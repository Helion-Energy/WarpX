/* This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#include "DarwinVacuumERecovery.H"
#include "DarwinVacuumAffineResponse.H"
#include "DarwinABoundary.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <ablastr/profiler/ProfilerWrapper.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>


#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
namespace
{
using Field = amrex::Array<amrex::MultiFab, 3>;
using View = ablastr::fields::VectorField;
inline View
view (Field& v)
{
    return {&v[0], &v[1], &v[2]};
}
} // namespace

static void
RecoverDarwinVacuumField (WarpX& warpx, HybridPICModel& hybrid, View const& endpoint,
                          View const& accepted, amrex::Real time, amrex::Real dt,
                          bool magnetic)
{
    ABLASTR_PROFILE("RecoverDarwinVacuumE()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::grid_type == GridType::Staggered,
                                     "Spatial vacuum electric recovery requires a staggered grid");
    using warpx::fields::FieldType;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.finestLevel() == 0, "Spatial vacuum electric recovery requires a single grid level");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(sizeof(amrex::Real) == sizeof(double),
                                     "Spatial vacuum electric recovery requires double precision");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(EB::enabled() && hybrid.m_use_conformal_eb),
        "Spatial vacuum electric recovery does not yet support conformal EB "
        "metrics");
    amrex::ParmParse pp("hybrid_pic_model");
    amrex::Real rtol = 1.e-12, atol = 1.e-10;
    int max_iterations = 2000;
    bool check_operator = false;
    pp.query("darwin_vacuum_e_relative_tolerance", rtol);
    pp.query("darwin_vacuum_e_absolute_tolerance", atol);
    pp.query("darwin_vacuum_e_max_iterations", max_iterations);
    pp.query("darwin_vacuum_e_check_operator", check_operator);
    if (magnetic)
    {
        rtol = hybrid.m_darwin_vacrec_rtol;
        atol = hybrid.m_darwin_vacrec_atol;
        // A native Krylov solve can require more iterations than the previous
        // global multigrid correction. Share the endpoint spatial solve budget.
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(rtol) && std::isfinite(atol) && rtol > 0. &&
                                         atol >= 0. && max_iterations > 0,
                                     "Invalid spatial vacuum electric recovery solver parameters");
    auto const& geom = warpx.Geom(0);
    bool const has_eb =
        !warpx.GetEBUpdateEFlag().empty() && warpx.GetEBUpdateEFlag()[0][0] != nullptr;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geom.isAllPeriodic() || hybrid.m_add_external_fields || has_eb,
        "Spatial vacuum electric recovery requires periodic boundaries or the "
        "Darwin A boundary pin");
#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes == 1,
                                     "Spatial RZ recovery supports only the axisymmetric mode");
    bool const axis = geom.ProbLo(0) == 0.;
    bool const flux_only = hybrid.m_darwin_vacuum_recovery_components == "flux";
#endif
    auto const e = warpx.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const b = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    auto const el = warpx.m_fields.get_alldirs("hybrid_E_long_fp", 0);
    View ext{nullptr, nullptr, nullptr};
    if (hybrid.m_add_external_fields)
    {
        ext = warpx.m_fields.get_alldirs(FieldType::hybrid_E_fp_external, 0);
    }
    auto const& rho = hybrid.m_darwin_vacuum_recovery_frozen_mask
                          ? *warpx.m_fields.get("hybrid_rho_vacmask_fp", 0)
                          : *warpx.m_fields.get(FieldType::rho_fp, 0);
    if (!magnetic && hybrid.m_add_external_fields)
    {
        hybrid.m_external_vector_potential->UpdateHybridExternalFields(time, dt);
    }
    Field base, input;
    amrex::Array<amrex::iMultiFab, 3> mask, trace_mask;
    amrex::Real diagonal = 0.;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        diagonal += 4. / (geom.CellSize(d) * geom.CellSize(d));
    }
    if (magnetic)
    {
        // The existing magnetic absolute tolerance is in raw Laplacian units.
        // Preserve its meaning when applying the normalized native operator.
        atol /= diagonal;
    }
    amrex::Real const floor =
        PhysConst::q_e * hybrid.m_n_floor * hybrid.m_darwin_vacuum_recovery_density_fraction;
    int const mode = hybrid.m_darwin_vacuum_recovery_mask == "global"
                         ? 2
                         : (hybrid.m_darwin_vacuum_recovery_mask == "transition" ? 1 : 0);
    auto const periodic = geom.periodicity();
    amrex::GpuArray<int, 3> per{1, 1, 1};
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{};
    for (int c = 0; c < AMREX_SPACEDIM; ++c)
    {
        per[c] = geom.isPeriodic(c);
        pmc_lo[c] = WarpX::field_boundary_lo[c] == FieldBoundaryType::PMC;
        pmc_hi[c] = WarpX::field_boundary_hi[c] == FieldBoundaryType::PMC;
    }
    for (int d = 0; d < 3; ++d)
    {
        auto const ba = e[d]->boxArray();
        auto const dm = e[d]->DistributionMap();
        base[d].define(ba, dm, 1, 0);
        base[d].setVal(0.);
        input[d].define(ba, dm, 1, e[d]->nGrowVect());
        mask[d].define(ba, dm, 1, 0);
        trace_mask[d].define(ba, dm, 1, 0);
        auto const domain = amrex::convert(geom.Domain(), e[d]->ixType());
        auto const lo = domain.smallEnd(), hi = domain.bigEnd();
        amrex::GpuArray<int, AMREX_SPACEDIM> nodal{};
        for (int c = 0; c < AMREX_SPACEDIM; ++c)
        {
            nodal[c] = e[d]->ixType().nodeCentered(c);
        }
        auto const* eb = EB::enabled() ? warpx.GetEBUpdateEFlag()[0][d].get() : nullptr;
        for (amrex::MFIter mfi(base[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto dst = base[d].array(mfi);
            auto m = mask[d].array(mfi);
            auto trace = trace_mask[d].array(mfi);
            auto r = rho.const_array(mfi);
            auto old = accepted[d]->const_array(mfi);
            auto end = endpoint[d]->const_array(mfi);
            amrex::Array4<int const> flag = eb ? eb->const_array(mfi) : amrex::Array4<int const>{};
            amrex::Array4<amrex::Real const> drive = hybrid.m_add_external_fields
                                                         ? ext[d]->const_array(mfi)
                                                         : amrex::Array4<amrex::Real const>{};
            amrex::ParallelFor(mfi.tilebox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   int p[3]{i, j, k};
                                   int q[3]{i, j, k};
#if defined(WARPX_DIM_RZ)
                                   q[0] += d == 0;
                                   q[1] += d == 2;
#else
                                   q[d]++;
#endif
                                   amrex::Real const density =
                                       .5 * (r(i, j, k) + r(q[0], q[1], q[2]));
                                   bool vacuum = mode == 2 || (mode == 0 && density < floor) ||
                                                 (mode == 1 && density > 0. && density < floor);
#if defined(WARPX_DIM_RZ)
                                   vacuum = vacuum && (!flux_only || d == 1);
#endif
                                   bool wall = false;
                                   for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                   {
                                       bool low_wall = p[c] <= lo[c];
#if defined(WARPX_DIM_RZ)
                                       if (c == 0 && axis)
                                       {
                                           low_wall = false;
                                       }
#endif
                                       // Only nodal points lie on the wall. The first/last
                                       // normal half-cell is an interior unknown, just as in
                                       // DarwinApplyABoundary; do not overwrite its endpoint E.
                                       if (!per[c] && nodal[c] &&
                                           ((low_wall && !pmc_lo[c]) ||
                                            (p[c] >= hi[c] && !pmc_hi[c])))
                                       {
                                           wall = true;
                                       }
                                   }
                                   bool conductor = flag && flag(i, j, k) == 0;
#if defined(WARPX_DIM_RZ)
                                   conductor = conductor || (axis && d == 1 && i == 0);
#endif
                                   m(i, j, k) = vacuum && !wall && !conductor;
                                   trace(i, j, k) = !vacuum && !wall && !conductor;
                                   dst(i, j, k) = (!magnetic && vacuum) ? old(i, j, k) : end(i, j, k);
                                   if (wall && !magnetic)
                                   {
                                       dst(i, j, k) = drive ? drive(i, j, k) : 0.;
                                   }
                                   if (conductor)
                                   {
                                       dst(i, j, k) = 0.;
                                   }
                               });
        }
        base[d].OverrideSync(periodic);
    }
    warpx::darwin::DarwinVacuumAffineResponse::Options options;
    options.relative_tolerance = rtol;
    options.absolute_tolerance = atol;
    options.max_iterations = max_iterations;
    options.magnetic = magnetic;
    options.check_operator = check_operator;
    options.pmc_lo = pmc_lo;
    options.pmc_hi = pmc_hi;
#if defined(WARPX_DIM_RZ)
    options.flux_only = flux_only;
#endif
    warpx::darwin::DarwinVacuumAffineResponse response(warpx, e, b, options);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        response.Freeze({&mask[0], &mask[1], &mask[2]},
                        {&trace_mask[0], &trace_mask[1], &trace_mask[2]}, view(base),
                        magnetic ? endpoint : ext),
        "Invalid frozen spatial recovery context");
    auto const result = response.Recover(endpoint);
    amrex::Print() << (magnetic ? "SPATIAL_A_RECOVERY iterations=" : "SPATIAL_RECOVERY iterations=")
                   << result.iterations << " initial=" << result.initial
                   << " residual=" << result.residual << " target=" << result.target << "\n";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(result,
        "Spatial vacuum recovery did not converge to its true residual tolerance");
    for (int d = 0; d < 3; ++d)
    {
        endpoint[d]->OverrideSync(periodic);
        if (magnetic)
        {
            // The caller restores A's gauge-shifted boundary pin and guards,
            // then reconstructs B. Do not touch E, EL, Aold or Je here.
            endpoint[d]->FillBoundary(periodic);
            continue;
        }
        input[d].setVal(0.);
        amrex::MultiFab::Copy(input[d], *endpoint[d], 0, 0, 1, 0);
        input[d].OverrideSync(periodic);
        input[d].FillBoundary(periodic);
#if defined(WARPX_DIM_RZ)
        // Fill each component here; all positive-r source values are already available.
        if (axis)
        {
            bool const node = input[d].ixType().nodeCentered(0);
            amrex::Real const sign = d == 2 ? 1. : -1.;
            for (amrex::MFIter mfi(input[d]); mfi.isValid(); ++mfi)
            {
                auto box = mfi.fabbox();
                if (box.smallEnd(0) >= 0)
                {
                    continue;
                }
                box.setBig(0, -1);
                auto a = input[d].array(mfi);
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   { a(i, j, k) = sign * a(-i - (node ? 0 : 1), j, k); });
            }
        }
#endif
        ApplyDarwinPMCVectorBoundary(input[d], geom, pmc_lo, pmc_hi, ext[d]);
        amrex::MultiFab::Copy(*e[d], input[d], 0, 0, 1, e[d]->nGrowVect());
        // Match the existing all-component boundary pin and continue its value
        // through physical guards.
        auto const domain = amrex::convert(geom.Domain(), e[d]->ixType());
        auto const lo = domain.smallEnd(), hi = domain.bigEnd();
        for (amrex::MFIter mfi(*e[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto dst = e[d]->array(mfi);
            auto const source = input[d].const_array(mfi);
            amrex::ParallelFor(mfi.growntilebox(e[d]->nGrowVect()),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   int p[3]{i, j, k};
                                   bool outside = false;
                                   for (int c = 0; c < AMREX_SPACEDIM; ++c)
                                   {
                                       if (!per[c] && (
#if defined(WARPX_DIM_RZ)
                                                          !(c == 0 && axis && p[c] < lo[c]) &&
#endif
                                                          ((p[c] < lo[c] && !pmc_lo[c]) ||
                                                           (p[c] > hi[c] && !pmc_hi[c]))))
                                       {
                                           p[c] = amrex::Clamp(p[c], lo[c], hi[c]);
                                           outside = true;
                                       }
                                   }
                                   if (outside)
                                   {
                                       dst(i, j, k) = source(p[0], p[1], p[2]);
                                   }
                               });
        }
        amrex::MultiFab::Add(*e[d], *el[d], 0, 0, 1, e[d]->nGrowVect());
        e[d]->OverrideSync(periodic);
        e[d]->FillBoundary(periodic);
        ApplyDarwinPMCVectorBoundary(*e[d], geom, pmc_lo, pmc_hi, ext[d]);
    }
}

void
RecoverDarwinVacuumE (WarpX& warpx, HybridPICModel& hybrid, View const& endpoint,
                      View const& accepted, amrex::Real time, amrex::Real dt)
{
    RecoverDarwinVacuumField(warpx, hybrid, endpoint, accepted, time, dt, false);
}

void
RecoverDarwinVacuumA (WarpX& warpx, HybridPICModel& hybrid, View const& endpoint)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(hybrid.m_darwin_vacrec_relax_time == 0.,
        "Spatial endpoint magnetic recovery requires instantaneous recovery");
    RecoverDarwinVacuumField(warpx, hybrid, endpoint, endpoint, 0., 0., true);
}

#else
void
RecoverDarwinVacuumE (WarpX&, HybridPICModel&, ablastr::fields::VectorField const&,
                      ablastr::fields::VectorField const&, amrex::Real, amrex::Real)
{
    WARPX_ABORT_WITH_MESSAGE("Spatial vacuum electric recovery requires 3D or RZ");
}
void
RecoverDarwinVacuumA (WarpX&, HybridPICModel&, ablastr::fields::VectorField const&)
{
    WARPX_ABORT_WITH_MESSAGE("Spatial vacuum magnetic recovery requires 3D or RZ");
}

#endif
