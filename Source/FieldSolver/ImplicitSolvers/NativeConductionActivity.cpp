/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "NativeConductionActivity.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConductivity.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

namespace warpx::thermal {
namespace {
using amrex::Real;
void
images (amrex::MultiFab& f, const amrex::Geometry& geom) {
    f.OverrideSync(geom.periodicity());
    f.FillBoundary(geom.periodicity());
    auto const domain = amrex::convert(geom.Domain(), f.ixType());
    auto const lo = domain.smallEnd(), hi = domain.bigEnd();
    auto const nodal = f.ixType().toIntVect();
    auto const periodic = geom.isPeriodicArray();
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto const a = f.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), f.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                int p[3] = {i, j, k};
                bool image = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!periodic[d] && (p[d] < lo[d] || p[d] > hi[d])) {
                        p[d] = p[d] < lo[d] ? 2 * lo[d] - p[d] - (1 - nodal[d])
                                            : 2 * hi[d] - p[d] + (1 - nodal[d]);
                        image = true;
                    }
                }
                if (image) {
                    a(i, j, k, n) = a(p[0], p[1], p[2], n);
                }
            });
    }
}
} // namespace
NativeConductionActivity::NativeConductionActivity (
    const amrex::Geometry& geom, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& dm, ConductionActivityOptions options)
    : m_geometry(geom), m_cells(cells), m_distribution(dm), m_options(options),
      m_candidate(amrex::convert(cells, amrex::IntVect::TheNodeVector()), dm,
                  Count, 1),
      m_nodes(m_candidate.boxArray(), dm, Count, 1),
      m_cell_open(cells, dm, 1, 1), m_fraction(cells, dm, 1, 0) {
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
    amrex::Abort("NativeConductionActivity requires RZ or Cartesian 3D");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        options.closure == ConductionCellClosure::AllContributors,
        "NativeConductionActivity requires explicit AllContributors closure");
    AMREX_ALWAYS_ASSERT(std::isfinite(options.legacy_density_floor) &&
                        options.legacy_density_floor > 0 &&
                        std::isfinite(options.common_density_floor) &&
                        options.common_density_floor > 0);
    AMREX_ALWAYS_ASSERT(cells.ixType().cellCentered() && cells.isDisjoint() &&
                        cells.numPts() == geom.Domain().numPts());
    for (int b = 0; b < cells.size(); ++b) {
        AMREX_ALWAYS_ASSERT(geom.Domain().contains(cells[b]));
    }
#if defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT(geom.Coord() == 1 && geom.ProbLo(0) == 0 &&
                        geom.Domain().smallEnd(0) == 0 && !geom.isPeriodic(0));
#else
    AMREX_ALWAYS_ASSERT(geom.Coord() == 0);
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(geom.Domain().length(d) >= 2);
        auto type = amrex::IntVect::TheNodeVector();
        type[d] = 0;
        m_edges[d] = std::make_unique<amrex::MultiFab>(
            amrex::convert(cells, type), dm, 1, 0);
        m_faces[d] = std::make_unique<amrex::MultiFab>(
            amrex::convert(cells, amrex::IntVect::TheDimensionVector(d)), dm, 1,
            0);
    }
}
bool
NativeConductionActivity::Evaluate (const amrex::MultiFab& charge,
                                    int component,
                                    const amrex::MultiFab* pedestal) {
    auto check = [&] (const amrex::MultiFab& f, int c) {
        AMREX_ALWAYS_ASSERT(f.boxArray() == m_nodes.boxArray() &&
                            f.DistributionMap() == m_distribution && c >= 0 &&
                            c < f.nComp() && !f.hasEBFabFactory());
    };
    check(charge, component);
    if (pedestal) {
        check(*pedestal, 0);
    }
    if (charge.contains_nan(component, 1, 0) ||
        charge.contains_inf(component, 1, 0)) {
        return false;
    }
    if (pedestal && (pedestal->contains_nan(0, 1, 0) ||
                     pedestal->contains_inf(0, 1, 0) || pedestal->min(0) < 0)) {
        return false;
    }
    ThermalConductionDensityOptions p{m_options.legacy_density_floor,
                                      m_options.halo_unfreeze,
                                      pedestal != nullptr};
    Real const common = m_options.common_density_floor;
    for (amrex::MFIter mfi(m_candidate, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const rho = charge.const_array(mfi);
        auto const ped =
            pedestal ? pedestal->const_array(mfi) : amrex::Array4<const Real>{};
        auto const out = m_candidate.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // Preserve the maintained kernel's two independent divisions.
                Real const raw = rho(i, j, k, component) / PhysConst::q_e;
                Real const np = ped ? ped(i, j, k) / PhysConst::q_e : 0.;
                auto const value = EvaluateThermalConductionDensity(raw, np, p);
                Real const ncommon = amrex::max(raw + np, common);
                out(i, j, k, Open) = value.active ? 1. : 0.;
                out(i, j, k, LegacyCapacity) = value.effective_number_density;
                out(i, j, k, CommonCapacity) = ncommon;
                out(i, j, k, RelativeCapacityGap) =
                    std::abs(ncommon / value.effective_number_density - 1.);
                out(i, j, k, ActiveRelativeCapacityGap) =
                    value.active ? out(i, j, k, RelativeCapacityGap) : 0.;
            });
    }
    if (m_candidate.contains_nan(0, Count, 0) ||
        m_candidate.contains_inf(0, Count, 0)) {
        return false;
    }
    amrex::MultiFab::Copy(m_nodes, m_candidate, 0, 0, Count, 0);
    images(m_nodes, m_geometry);
#if defined(WARPX_DIM_RZ)
    int const rlo = m_geometry.Domain().smallEnd(0);
#endif
    for (amrex::MFIter mfi(m_cell_open, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const n = m_nodes.const_array(mfi);
        auto const out = m_cell_open.array(mfi);
        auto const f = m_fraction.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
#if defined(WARPX_DIM_RZ)
                Real const radial = static_cast<Real>(i - rlo);
                Real const wl =
                    (radial + .25) /
                    (2. * radial +
                     1.); // physical dual overlap, NOT deposition quadrature
                Real const base = n(i, j, k, Open);
                Real all = 1., fraction = 0.;
                for (int bits = 0; bits < 4; ++bits) {
                    int const di = bits & 1, dj = (bits >> 1) & 1;
                    Real const open = n(i + di, j + dj, k, Open);
                    all *= open;
                    fraction += (di == 0 ? wl : 1. - wl) * .5 * (open - base);
                }
#else
            Real const base=n(i,j,k,Open);
            Real all=1.,fraction=0.;
            for(int bits=0;bits<8;++bits) {
                int const di=bits&1,dj=(bits>>1)&1,dk=(bits>>2)&1;
                Real const open=n(i+di,j+dj,k+dk,Open);
                all*=open;fraction+=.125*(open-base);
            }
#endif
                out(i, j, k) = all;
                f(i, j, k) = base + fraction;
            });
    }
    images(m_cell_open, m_geometry);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int const di = d == 0, dj = d == 1, dk = d == 2;
        for (amrex::MFIter mfi(*m_edges[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const n = m_nodes.const_array(mfi);
            auto const out = m_edges[d]->array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    out(i, j, k) =
                        n(i, j, k, Open) * n(i + di, j + dj, k + dk, Open);
                });
        }
        for (amrex::MFIter mfi(*m_faces[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const c = m_cell_open.const_array(mfi);
            auto const out = m_faces[d]->array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    out(i, j, k) = c(i - di, j - dj, k - dk) * c(i, j, k);
                });
        }
        m_edges[d]->OverrideSync(m_geometry.periodicity());
        m_faces[d]->OverrideSync(m_geometry.periodicity());
    }
    m_valid = true;
    return true;
}
ConductionActivityMeasure
NativeConductionActivity::Measure () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
        op;
    amrex::ReduceData<Real, Real, Real> data(op);
    using Tuple = decltype(data)::Type;
    auto const dx = m_geometry.CellSizeArray();
    for (amrex::MFIter mfi(m_fraction); mfi.isValid(); ++mfi) {
        auto const f = m_fraction.const_array(mfi);
        auto const a = m_cell_open.const_array(mfi);
        op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
#if defined(WARPX_DIM_RZ)
                    Real const volume = 2. * 3.14159265358979323846 * (i + .5) *
                                        dx[0] * dx[0] * dx[1];
#else
            Real const volume=dx[0]*dx[1]*dx[2];
#endif
                    Real const fraction = f(i, j, k);
                    return {volume * fraction, volume * a(i, j, k),
                            volume * (fraction > 0 && fraction < 1 ? 1. : 0.)};
                });
    }
    auto result = data.value();
    Real sum[3] = {amrex::get<0>(result), amrex::get<1>(result),
                   amrex::get<2>(result)};
    amrex::ParallelDescriptor::ReduceRealSum(sum, 3);
    return {sum[0],
            sum[1],
            sum[2],
            sum[0] - sum[1],
            m_nodes.norm0(RelativeCapacityGap, 0),
            m_nodes.norm0(ActiveRelativeCapacityGap, 0)};
}
} // namespace warpx::thermal
