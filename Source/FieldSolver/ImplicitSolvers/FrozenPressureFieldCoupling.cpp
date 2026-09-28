/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FrozenPressureFieldCoupling.H"
#include "Utils/WarpXConst.H"
#ifdef WARPX_DIM_RZ
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include <AMReX_MFIter.H>
#include <utility>

namespace warpx::thermal {
namespace {
using amrex::Real;

amrex::IntVect
yee_type (int c) {
    auto type = amrex::IntVect::TheNodeVector();
#ifdef WARPX_DIM_RZ
    if (c != 1) {
        type[c / 2] = 0;
    }
#else
    if (c < AMREX_SPACEDIM) {
        type[c] = 0;
    }
#endif
    return type;
}

bool
finite (const amrex::MultiFab& f) {
    return !f.contains_nan(0, 1, 0) && !f.contains_inf(0, 1, 0);
}

// Match KineticThermalMoments' even scalar images, including periodic sync.
void
scalar_images (amrex::MultiFab& data, const amrex::Geometry& geometry) {
    data.OverrideSync(geometry.periodicity());
    data.FillBoundary(geometry.periodicity());
    auto const domain = amrex::convert(geometry.Domain(), data.ixType());
    auto const lo = domain.smallEnd();
    auto const hi = domain.bigEnd();
    auto const nodal = data.ixType().toIntVect();
    auto const periodic = geometry.isPeriodicArray();
    for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
        auto const a = data.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int src[3] = {i, j, k};
                bool image = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!periodic[d] && (src[d] < lo[d] || src[d] > hi[d])) {
                        src[d] = src[d] < lo[d]
                                     ? 2 * lo[d] - src[d] - (1 - nodal[d])
                                     : 2 * hi[d] - src[d] + (1 - nodal[d]);
                        image = true;
                    }
                }
                if (image) {
                    a(i, j, k) = a(src[0], src[1], src[2]);
                }
            });
    }
}

void
pressure_images (amrex::MultiFab& pressure, amrex::MultiFab& ohm,
                 const amrex::Geometry& geometry,
                 const FrozenPressureFieldOptions& options) {
    scalar_images(pressure, geometry);
    auto const lo = geometry.Domain().smallEnd();
    auto const hi =
        geometry.Domain().bigEnd() + amrex::IntVect::TheNodeVector();
    auto const periodic = geometry.isPeriodicArray();
    amrex::GpuArray<amrex::GpuArray<bool, 2>, AMREX_SPACEDIM> pec{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int s = 0; s < 2; ++s) {
            pec[d][s] = options.boundary[d][s] == MomentBoundary::PEC;
        }
    }
    // Immutable gather is also the production moments' corner/PEC rule.
    for (amrex::MFIter mfi(ohm); mfi.isValid(); ++mfi) {
        auto const in = pressure.const_array(mfi);
        auto const out = ohm.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               int src[3] = {i, j, k};
                               for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                   if (!periodic[d]) {
                                       if (src[d] < lo[d]) {
                                           src[d] = 2 * lo[d] - src[d];
                                       }
                                       if (src[d] > hi[d]) {
                                           src[d] = 2 * hi[d] - src[d];
                                       }
                                       if (pec[d][0] && src[d] == lo[d]) {
                                           ++src[d];
                                       }
                                       if (pec[d][1] && src[d] == hi[d]) {
                                           --src[d];
                                       }
                                   }
                               }
                               out(i, j, k) = in(src[0], src[1], src[2]);
                           });
    }
    ohm.OverrideSync(geometry.periodicity());
    ohm.FillBoundary(geometry.periodicity());
}
} // namespace

FrozenPressureFieldCoupling::FrozenPressureFieldCoupling (
    const amrex::Geometry& geometry, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& distribution,
    FrozenPressureFieldOptions options)
    : m_geometry(geometry), m_cells(cells), m_distribution(distribution),
      m_options(std::move(options)), m_cell_density(cells, distribution, 1, 0),
      m_node_density(amrex::convert(cells, amrex::IntVect::TheNodeVector()),
                     distribution, 1, 0),
      m_temperature(cells, distribution, 1, 1),
      m_node_temperature(m_node_density.boxArray(), distribution, 1, 0),
      m_pressure(m_node_density.boxArray(), distribution, 1, 1),
      m_ohm_pressure(m_node_density.boxArray(), distribution, 1, 1) {
#ifndef WARPX_DIM_RZ
    amrex::Abort("FrozenPressureFieldCoupling currently supports only RZ m=0");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_options.transformed_ohm_form,
        "FrozenPressureFieldCoupling requires physical E-form Ohm response");
    AMREX_ALWAYS_ASSERT(std::isfinite(m_options.gamma) &&
                        m_options.gamma > 1.0);
    AMREX_ALWAYS_ASSERT(cells.ixType().cellCentered() && cells.isDisjoint() &&
                        cells.numPts() == geometry.Domain().numPts());
    for (int b = 0; b < cells.size(); ++b) {
        AMREX_ALWAYS_ASSERT(geometry.Domain().contains(cells[b]));
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(geometry.Domain().length(d) >= 2);
        for (int s = 0; s < 2; ++s) {
            auto const bc = m_options.boundary[d][s];
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                geometry.isPeriodic(d)
                    ? bc == MomentBoundary::Periodic
                    : (bc == MomentBoundary::Axis ||
                       bc == MomentBoundary::PEC || bc == MomentBoundary::PMC),
                "FrozenPressureFieldCoupling needs explicit "
                "geometry-consistent pressure images");
            AMREX_ALWAYS_ASSERT(bc != MomentBoundary::Axis ||
                                (d == 0 && s == 0));
        }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 1 && geometry.ProbLo(0) == 0.0 &&
            geometry.Domain().smallEnd(0) == 0 && !geometry.isPeriodic(0) &&
            m_options.boundary[0][0] == MomentBoundary::Axis &&
            m_options.azimuthal_modes == 1,
        "FrozenPressureFieldCoupling requires normalized RZ geometry, r_lo=0, "
        "and m=0");
    for (int c = 0; c < 3; ++c) {
        m_weight[c] = std::make_unique<amrex::MultiFab>(
            amrex::convert(cells, yee_type(c)), distribution, 1, 0);
    }
}

void
FrozenPressureFieldCoupling::CheckLayout (const amrex::MultiFab& f,
                                          const amrex::IntVect& type) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        f.boxArray() == amrex::convert(m_cells, type) &&
            f.DistributionMap() == m_distribution && f.nComp() == 1 &&
            !f.hasEBFabFactory(),
        "FrozenPressureFieldCoupling requires matching single-component "
        "layouts without EB");
}

bool
FrozenPressureFieldCoupling::Freeze (const amrex::MultiFab& cell_density,
                                     const amrex::MultiFab& nodal_density,
                                     ConstFieldView weight) {
    CheckLayout(cell_density, amrex::IntVect::TheZeroVector());
    CheckLayout(nodal_density, amrex::IntVect::TheNodeVector());
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(weight[c]);
        CheckLayout(*weight[c], yee_type(c));
    }
    if (!finite(cell_density) || !finite(nodal_density) ||
        cell_density.min(0) <= 0.0 || nodal_density.min(0) <= 0.0) {
        return false;
    }
    for (int c = 0; c < 3; ++c) {
        if (!finite(*weight[c]) || weight[c]->min(0) < 0.0) {
            return false;
        }
    }
    amrex::MultiFab::Copy(m_cell_density, cell_density, 0, 0, 1, 0);
    amrex::MultiFab::Copy(m_node_density, nodal_density, 0, 0, 1, 0);
    m_node_density.OverrideSync(m_geometry.periodicity());
    for (int c = 0; c < 3; ++c) {
        amrex::MultiFab::Copy(*m_weight[c], *weight[c], 0, 0, 1, 0);
        m_weight[c]->OverrideSync(m_geometry.periodicity());
    }
    m_frozen = true;
    ++m_freezes;
    return true;
}

void
FrozenPressureFieldCoupling::Apply (FieldView output,
                                    const amrex::MultiFab& delta_energy) {
    AMREX_ALWAYS_ASSERT(m_frozen);
    CheckLayout(delta_energy, amrex::IntVect::TheZeroVector());
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(output[c] && output[c] != &delta_energy);
        CheckLayout(*output[c], yee_type(c));
    }
    Real const gm1 = m_options.gamma - 1.0;
    for (amrex::MFIter mfi(m_temperature, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const u = delta_energy.const_array(mfi);
        auto const n = m_cell_density.const_array(mfi);
        auto const t = m_temperature.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                t(i, j, k) = gm1 * u(i, j, k) / (PhysConst::kb * n(i, j, k));
            });
    }
    scalar_images(m_temperature, m_geometry);
    for (amrex::MFIter mfi(m_node_temperature, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const t = m_temperature.const_array(mfi);
        auto const nt = m_node_temperature.array(mfi);
        auto const n = m_node_density.const_array(mfi);
        auto const p = m_pressure.array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int k) {
            Real const base = t(i, j, k);
            Real sum = 0.0;
            for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
                int const di = bits & 1, dj = (bits >> 1) & 1;
#if AMREX_SPACEDIM == 3
                int const dk = (bits >> 2) & 1;
#else
                int const dk = 0;
#endif
                sum += t(i - di, j - dj, k - dk) - base;
            }
            nt(i, j, k) = base + sum / static_cast<Real>(1 << AMREX_SPACEDIM);
            p(i, j, k) = PhysConst::kb * n(i, j, k) * nt(i, j, k);
        });
    }
    pressure_images(m_pressure, m_ohm_pressure, m_geometry, m_options);
#ifdef WARPX_DIM_RZ
    auto const inverse_dx = m_geometry.InvCellSizeArray();
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*output[c], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const out = output[c]->array(mfi);
            auto const p = m_ohm_pressure.const_array(mfi);
            auto const w = m_weight[c]->const_array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (c == 1 || w(i, j, k) == 0.0) {
                        out(i, j, k) = 0.0;
                        return;
                    }
                    Real const grad =
                        c == 0 ? CylindricalYeeAlgorithm::UpwardDr(
                                     p, inverse_dx.data(), 1, i, j, k, 0)
                               : CylindricalYeeAlgorithm::UpwardDz(
                                     p, inverse_dx.data() + 1, 1, i, j, k, 0);
                    out(i, j, k) = w(i, j, k) * grad;
                });
        }
        output[c]->OverrideSync(m_geometry.periodicity());
    }
#endif
    ++m_applies;
}
} // namespace warpx::thermal
