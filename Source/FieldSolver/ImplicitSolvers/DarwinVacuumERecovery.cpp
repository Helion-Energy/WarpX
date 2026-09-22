/* This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#include "DarwinVacuumERecovery.H"

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
    Field base, delta, residual, direction, action, input, curl, current;
    amrex::Array<amrex::iMultiFab, 3> mask;
    amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> owner;
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
    amrex::Real const scale = PhysConst::mu0 / diagonal;
    amrex::Real const floor =
        PhysConst::q_e * hybrid.m_n_floor * hybrid.m_darwin_vacuum_recovery_density_fraction;
    int const mode = hybrid.m_darwin_vacuum_recovery_mask == "global"
                         ? 2
                         : (hybrid.m_darwin_vacuum_recovery_mask == "transition" ? 1 : 0);
    auto const periodic = geom.periodicity();
    amrex::GpuArray<int, 3> per{1, 1, 1};
    for (int c = 0; c < AMREX_SPACEDIM; ++c)
    {
        per[c] = geom.isPeriodic(c);
    }
    for (int d = 0; d < 3; ++d)
    {
        auto const ba = e[d]->boxArray();
        auto const dm = e[d]->DistributionMap();
        for (auto* f : {&base, &delta, &residual, &direction, &action})
        {
            (*f)[d].define(ba, dm, 1, 0);
            (*f)[d].setVal(0.);
        }
        input[d].define(ba, dm, 1, e[d]->nGrowVect());
        current[d].define(ba, dm, 1, 1);
        curl[d].define(b[d]->boxArray(), b[d]->DistributionMap(), 1, b[d]->nGrowVect());
        mask[d].define(ba, dm, 1, 0);
        owner[d] = e[d]->OwnerMask(periodic);
        auto const domain = amrex::convert(geom.Domain(), e[d]->ixType());
        auto const lo = domain.smallEnd(), hi = domain.bigEnd();
        auto const* eb = EB::enabled() ? warpx.GetEBUpdateEFlag()[0][d].get() : nullptr;
        for (amrex::MFIter mfi(base[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto dst = base[d].array(mfi);
            auto m = mask[d].array(mfi);
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
                                       if (!per[c] && (low_wall || p[c] >= hi[c]))
                                       {
                                           wall = true;
                                       }
                                   }
                                   bool conductor = flag && flag(i, j, k) == 0;
#if defined(WARPX_DIM_RZ)
                                   conductor = conductor || (axis && d == 1 && i == 0);
#endif
                                   m(i, j, k) = vacuum && !wall && !conductor;
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
    // Cylindrical dual-volume weights make the native curl pair adjoint.
    // Divide all weights by dr^2: the axis Ez weight is then 1/8.
    auto component_dot =
        [&] (amrex::MultiFab const& x, amrex::MultiFab const& y, amrex::iMultiFab const& own)
    {
#if defined(WARPX_DIM_RZ)
        amrex::Real const r0 = geom.ProbLo(0) / geom.CellSize(0);
        bool const node = x.ixType().nodeCentered(0);
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi)
        {
            auto const a = x.const_array(mfi), b = y.const_array(mfi);
            auto const o = own.const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                    {
                        amrex::Real weight = r0 + i + (node ? 0. : .5);
                        if (axis && node && i == 0)
                        {
                            weight = .125;
                        }
                        return {o(i, j, k) ? weight * a(i, j, k) * b(i, j, k) : 0.};
                    });
        }
        return amrex::get<0>(data.value(op));
#else
        return amrex::MultiFab::Dot(own, x, 0, y, 0, 1, 0, true);
#endif
    };
    auto dot = [&] (Field const& x, Field const& y)
    {
        amrex::Real v = 0.;
        for (int d = 0; d < 3; ++d)
        {
            v += component_dot(x[d], y[d], *owner[d]);
        }
        amrex::ParallelAllReduce::Sum(v, amrex::ParallelContext::CommunicatorSub());
        return v;
    };
    auto apply = [&] (Field& out, Field const& x)
    {
        for (int d = 0; d < 3; ++d)
        {
            input[d].setVal(0.);
            amrex::MultiFab::Copy(input[d], x[d], 0, 0, 1, 0);
            input[d].OverrideSync(periodic);
            input[d].FillBoundary(periodic);
            curl[d].setVal(0.);
            current[d].setVal(0.);
        }
#if defined(WARPX_DIM_RZ)
        if (axis)
        {
            warpx.ApplyFieldBoundaryOnAxis(&input[0], &input[1], &input[2], 0);
        }
#endif
        auto cv = view(curl), iv = view(input), jv = view(current);
        warpx.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(cv, iv, warpx.GetEBUpdateBFlag()[0], 0);
        for (auto& f : curl)
        {
            f.OverrideSync(periodic);
            f.FillBoundary(periodic);
        }
#if defined(WARPX_DIM_RZ)
        if (axis)
        {
            warpx.ApplyFieldBoundaryOnAxis(&curl[0], &curl[1], &curl[2], 0);
        }
#endif
        warpx.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(jv, cv,
                                                                    warpx.GetEBUpdateEFlag()[0], 0);
        for (int d = 0; d < 3; ++d)
        {
            for (amrex::MFIter mfi(out[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto a = out[d].array(mfi);
                auto c = current[d].const_array(mfi);
                auto m = mask[d].const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   { a(i, j, k) = m(i, j, k) ? scale * c(i, j, k) : 0.; });
            }
            out[d].OverrideSync(periodic);
        }
    };
    if (check_operator)
    {
        Field u, v, ku, kv;
        for (int d = 0; d < 3; ++d)
        {
            for (auto* f : {&u, &v, &ku, &kv})
            {
                (*f)[d].define(base[d].boxArray(), base[d].DistributionMap(), 1, 0);
            }
            for (amrex::MFIter mfi(u[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto ua = u[d].array(mfi), va = v[d].array(mfi);
                auto m = mask[d].const_array(mfi);
                amrex::ParallelFor(
                    mfi.tilebox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        ua(i, j, k) =
                            m(i, j, k) ? std::sin(.731 * (i + 3 * j + 7 * k + 11 * d)) : 0.;
                        va(i, j, k) =
                            m(i, j, k) ? std::cos(.413 * (5 * i + 2 * j + 3 * k + 13 * d)) : 0.;
                    });
            }
        }
        apply(ku, u);
        amrex::Real energy = 0.;
        for (int d = 0; d < 3; ++d)
        {
            auto const own = curl[d].OwnerMask(periodic);
            energy += component_dot(curl[d], curl[d], *own) / diagonal;
        }
        amrex::ParallelAllReduce::Sum(energy, amrex::ParallelContext::CommunicatorSub());
        apply(kv, v);
        amrex::Real const uku = dot(u, ku), ukv = dot(u, kv), vku = dot(v, ku);
        amrex::Real const energy_error = std::abs(uku - energy) / std::max(1., std::abs(energy));
        amrex::Real const symmetry_error =
            std::abs(ukv - vku) / std::max(1., std::sqrt(dot(u, u) * dot(v, v)));
        amrex::Print() << "Spatial recovery operator: symmetry=" << symmetry_error
                       << " energy=" << energy_error << "\n";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            symmetry_error < 1.e-12 && energy_error < 1.e-12,
            "Spatial recovery operator violates its symmetry or curl-energy "
            "identity");
    }
    apply(action, base);
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::Copy(residual[d], action[d], 0, 0, 1, 0);
        residual[d].mult(-1., 0, 1, 0);
        amrex::MultiFab::Copy(direction[d], residual[d], 0, 0, 1, 0);
    }
    // With K = I_V C^T C I_V, zero-start unpreconditioned CG keeps every
    // correction in range(K). It preserves the baseline's null modes (accepted
    // electric modes or extrapolated endpoint A modes). Fixed plasma and wall
    // values never enter the correction space. A global inverse followed by
    // masking is not this projection and can amplify the interface defect.
    // An arbitrary preconditioner would invalidate the null-mode selection.
    amrex::Real rr = dot(residual, residual), initial = std::sqrt(rr);
    // A has no universal dimensional absolute floor. Stop at the native
    // operator's roundoff scale when an already projected field is revisited;
    // otherwise a second projection would demand a tolerance below cancellation
    // error. This bound scales with the actual field, not a fixed unit value.
    amrex::Real reference_norm = 0.;
    if (magnetic)
    {
#if defined(WARPX_DIM_RZ)
        if (flux_only)
        {
            // The axisymmetric toroidal block does not depend on Ar or Az.
            // Large unchanged poloidal fields must not loosen this solve.
            reference_norm = component_dot(base[1], base[1], *owner[1]);
            amrex::ParallelAllReduce::Sum(reference_norm,
                                         amrex::ParallelContext::CommunicatorSub());
        }
        else
#endif
        {
            reference_norm = dot(base, base);
        }
    }
    amrex::Real const roundoff =
        64. * std::numeric_limits<amrex::Real>::epsilon() * std::sqrt(reference_norm);
    amrex::Real const target = std::max({atol, rtol * initial, roundoff});
    amrex::Real truth = initial;
    int iter = 0;
    while (truth > target && iter < max_iterations)
    {
        apply(action, direction);
        amrex::Real const pap = dot(direction, action);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pap > 0. && std::isfinite(pap),
                                         "Spatial recovery CG curvature must be positive");
        amrex::Real const alpha = rr / pap;
        for (int d = 0; d < 3; ++d)
        {
            amrex::MultiFab::Saxpy(delta[d], alpha, direction[d], 0, 0, 1, 0);
            amrex::MultiFab::Saxpy(residual[d], -alpha, action[d], 0, 0, 1, 0);
        }
        ++iter;
        amrex::Real next = dot(residual, residual);
        truth = std::sqrt(next);
        bool restart = false;
        if (truth <= target)
        {
            for (int d = 0; d < 3; ++d)
            {
                amrex::MultiFab::LinComb(input[d], 1., base[d], 0, 1., delta[d], 0, 0, 1, 0);
            }
            // apply() owns input scratch: use direction only after its old
            // value is no longer needed.
            for (int d = 0; d < 3; ++d)
            {
                amrex::MultiFab::Copy(direction[d], input[d], 0, 0, 1, 0);
            }
            apply(action, direction);
            for (int d = 0; d < 3; ++d)
            {
                amrex::MultiFab::Copy(residual[d], action[d], 0, 0, 1, 0);
                residual[d].mult(-1., 0, 1, 0);
            }
            next = dot(residual, residual);
            truth = std::sqrt(next);
            restart = true;
        }
        if (truth <= target)
        {
            break;
        }
        for (int d = 0; d < 3; ++d)
        {
            amrex::MultiFab::LinComb(direction[d], 1., residual[d], 0, restart ? 0. : next / rr,
                                     direction[d], 0, 0, 1, 0);
        }
        rr = next;
    }
    amrex::Print() << (magnetic ? "SPATIAL_A_RECOVERY iterations=" : "SPATIAL_RECOVERY iterations=") << iter << " initial=" << initial
                   << " residual=" << truth << " target=" << target << "\n";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(truth <= target, "Spatial vacuum recovery did not converge "
                                                      "to its true residual tolerance");
    for (int d = 0; d < 3; ++d)
    {
        amrex::MultiFab::LinComb(*endpoint[d], 1., base[d], 0, 1., delta[d], 0, 0, 1, 0);
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
                                                          (p[c] < lo[c] || p[c] > hi[c])))
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
