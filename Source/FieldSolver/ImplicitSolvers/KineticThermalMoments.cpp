/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "KineticThermalMoments.H"
#include "ThermalCurrentRemainder.H"

#include "Utils/WarpXConst.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <utility>

namespace warpx::thermal {
namespace {
using amrex::Real;

amrex::IntVect
yee_type (int component) {
    amrex::IntVect result = amrex::IntVect::TheNodeVector();
#if defined(WARPX_DIM_RZ)
    if (component != 1) {
        result[component / 2] = 0;
    }
#else
    if (component < AMREX_SPACEDIM) {
        result[component] = 0;
    }
#endif
    return result;
}

int
physical_direction (int d) {
#if defined(WARPX_DIM_RZ)
    return 2 * d;
#else
    return d;
#endif
}

bool
finite (const amrex::MultiFab& field) {
    return !field.contains_nan(0, 1, 0) && !field.contains_inf(0, 1, 0);
}

// Physical-volume overlap of the right half of nodal dual cell i with
// primal cell i, divided by the primal cell volume. The other weight is 1-a.
// In RZ: a=(r_i+dr/4)/(2*r_i+dr), including a_axis=1/4.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
radial_left_weight (int i, int lo) {
#if defined(WARPX_DIM_RZ)
    Real const radius = static_cast<Real>(i - lo);
    return (radius + 0.25) / (2.0 * radius + 1.0);
#else
    amrex::ignore_unused(i, lo);
    return 0.5;
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
native_volume_ratio (int i, int hi, bool corrected_axis) {
#if defined(WARPX_DIM_RZ)
    if (i == 0) {
        return corrected_axis ? 4.0 / 3.0 : 1.0;
    }
    if (i == hi) {
        return static_cast<Real>(hi) / (static_cast<Real>(hi) - 0.25);
    }
#else
    amrex::ignore_unused(i, hi, corrected_axis);
#endif
    return 1.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
restrict_scalar (amrex::Array4<const Real> const& a, int i, int j, int k,
                 int radial_lo) {
    Real const left = radial_left_weight(i, radial_lo);
    // Anchor the convex interpolation to preserve the exact constant nullspace.
    Real const base = a(i, j, k);
    Real sum = 0.0;
    for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
        int const di = bits & 1;
        int const dj = (bits >> 1) & 1;
#if (AMREX_SPACEDIM == 3)
        int const dk = (bits >> 2) & 1;
#else
        int const dk = 0;
#endif
        Real const w = (di == 0 ? left : 1.0 - left) /
                       static_cast<Real>(1 << (AMREX_SPACEDIM - 1));
        sum += w * (a(i + di, j + dj, k + dk) - base);
    }
    return base + sum;
}

// Slot/weight association is shared by the ordinary and retained-increment
// maps. In particular 1-left and sequential metric products remain native
// Real operations, not algebraically reassociated coefficients.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
current_slot (int bits,int d,int i,int radial_lo,int radial_hi,bool corrected,int (&src)[3]) {
    Real weight=1.0;
                    for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                        int const b = (bits >> t) & 1;
                        if (t == d) {
                            src[t] += b - 1;
#if defined(WARPX_DIM_RZ)
                            if (d == 0) {
                                Real const r = static_cast<Real>(i - radial_lo);
                                Real const s_minus =
                                    i == radial_hi ? 0.5
                                                   : (r - 0.25) / (2.0 * r);
                                weight *= b == 0
                                              ? (1.0 - s_minus) * (r - 0.5) / r
                                              : s_minus * (r + 0.5) / r;
                            } else
#endif
                            {
                                weight *= 0.5;
                            }
                        } else {
                            src[t] += b;
                            Real const left =
                                t == 0 ? radial_left_weight(i, radial_lo) : 0.5;
                            weight *= b == 0 ? left : 1.0 - left;
                            if (t == 0)
                                weight *= native_volume_ratio(i + b, radial_hi,
                                                              corrected);
                        }
                    }
    return weight;
}

void
fill_scalar_images (amrex::MultiFab& data, const amrex::Geometry& m_geometry) {
    data.OverrideSync(m_geometry.periodicity());
    data.FillBoundary(m_geometry.periodicity());
    auto const domain = amrex::convert(m_geometry.Domain(), data.ixType());
    auto const lo = domain.smallEnd();
    auto const hi = domain.bigEnd();
    auto const nodal = data.ixType().toIntVect();
    auto const periodic = m_geometry.isPeriodicArray();
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
fill_current_images (amrex::MultiFab& data, int physical_component,
                     const amrex::Geometry& m_geometry,
                     const ThermalMomentOptions& m_options) {
    data.OverrideSync(m_geometry.periodicity());
    data.FillBoundary(m_geometry.periodicity());
    auto const domain = amrex::convert(m_geometry.Domain(), data.ixType());
    auto const lo = domain.smallEnd();
    auto const hi = domain.bigEnd();
    auto const nodal = data.ixType().toIntVect();
    auto const periodic = m_geometry.isPeriodicArray();
    amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM> signs{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int side = 0; side < 2; ++side) {
            bool const normal = physical_direction(d) == physical_component;
            bool const pec =
                m_options.current_boundary[d][side] == MomentBoundary::PEC;
            signs[d][side] = (normal != pec) ? -1 : 1;
#if defined(WARPX_DIM_RZ)
            if (d == 0 && side == 0) {
                signs[d][side] = physical_component == 2 ? 1 : -1;
            }
#endif
        }
    }
    for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
        auto const a = data.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int src[3] = {i, j, k};
                Real factor = 1.0;
                bool image = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!periodic[d] && (src[d] < lo[d] || src[d] > hi[d])) {
                        int const side = src[d] < lo[d] ? 0 : 1;
                        int const dst = src[d];
                        src[d] = side == 0 ? 2 * lo[d] - dst - (1 - nodal[d])
                                           : 2 * hi[d] - dst + (1 - nodal[d]);
                        factor *= signs[d][side];
#if defined(WARPX_DIM_RZ)
                        // SetJorRho's physical RZ wall image preserves r J. The
                        // axis image uses vector parity without a radial ratio.
                        if (d == 0 && side == 1) {
                            Real const shift = 0.5 * (1 - nodal[0]);
                            factor *= (src[0] + shift) / (dst + shift);
                        }
#endif
                        image = true;
                    }
                }
                if (image) {
                    a(i, j, k) = factor * a(src[0], src[1], src[2]);
                }
            });
    }
}

void
fill_current_pair_images (amrex::MultiFab& data, amrex::MultiFab& small, int physical_component,
                     const amrex::Geometry& m_geometry,
                     const ThermalMomentOptions& m_options) {
    small.OverrideSync(m_geometry.periodicity());
    small.FillBoundary(m_geometry.periodicity());
    data.OverrideSync(m_geometry.periodicity());
    data.FillBoundary(m_geometry.periodicity());
    auto const domain = amrex::convert(m_geometry.Domain(), data.ixType());
    auto const lo = domain.smallEnd();
    auto const hi = domain.bigEnd();
    auto const nodal = data.ixType().toIntVect();
    auto const periodic = m_geometry.isPeriodicArray();
    amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM> signs{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int side = 0; side < 2; ++side) {
            bool const normal = physical_direction(d) == physical_component;
            bool const pec =
                m_options.current_boundary[d][side] == MomentBoundary::PEC;
            signs[d][side] = (normal != pec) ? -1 : 1;
#if defined(WARPX_DIM_RZ)
            if (d == 0 && side == 0) {
                signs[d][side] = physical_component == 2 ? 1 : -1;
            }
#endif
        }
    }
    for (amrex::MFIter mfi(data); mfi.isValid(); ++mfi) {
        auto const a = data.array(mfi);
        auto const l = small.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int src[3] = {i, j, k};
                Real factor = 1.0;
                bool image = false;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (!periodic[d] && (src[d] < lo[d] || src[d] > hi[d])) {
                        int const side = src[d] < lo[d] ? 0 : 1;
                        int const dst = src[d];
                        src[d] = side == 0 ? 2 * lo[d] - dst - (1 - nodal[d])
                                           : 2 * hi[d] - dst + (1 - nodal[d]);
                        factor *= signs[d][side];
#if defined(WARPX_DIM_RZ)
                        // SetJorRho's physical RZ wall image preserves r J. The
                        // axis image uses vector parity without a radial ratio.
                        if (d == 0 && side == 1) {
                            Real const shift = 0.5 * (1 - nodal[0]);
                            factor *= (src[0] + shift) / (dst + shift);
                        }
#endif
                        image = true;
                    }
                }
                if (image) {
                    namespace dd=warpx::ohm::compensated;
                    auto const value=dd::Multiply({a(src[0],src[1],src[2]),l(src[0],src[1],src[2])},factor);
                    a(i,j,k)=value.hi; l(i,j,k)=value.lo;
                }
            });
    }
}

void
apply_pressure_images (amrex::MultiFab& m_pressure,
                       amrex::MultiFab& m_ohm_pressure,
                       const amrex::Geometry& m_geometry,
                       const ThermalMomentOptions& m_options) {
    // Gather FROM immutable thermodynamic pressure. This is the deterministic
    // PEC valid-node copy used by the production boundary operator, followed
    // by even scalar ghosts on the remaining axis/PMC sides. No scatter race.
    fill_scalar_images(m_pressure, m_geometry);
    auto const lo = m_geometry.Domain().smallEnd();
    auto hi = m_geometry.Domain().bigEnd();
    hi += amrex::IntVect::TheNodeVector();
    auto const periodic = m_geometry.isPeriodicArray();
    amrex::GpuArray<amrex::GpuArray<bool, 2>, AMREX_SPACEDIM> pec{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int s = 0; s < 2; ++s) {
            pec[d][s] = m_options.boundary[d][s] == MomentBoundary::PEC;
        }
    }
    for (amrex::MFIter mfi(m_ohm_pressure); mfi.isValid(); ++mfi) {
        auto const in = m_pressure.const_array(mfi);
        auto const out = m_ohm_pressure.array(mfi);
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
    m_ohm_pressure.OverrideSync(m_geometry.periodicity());
    m_ohm_pressure.FillBoundary(m_geometry.periodicity());
}

} // namespace

KineticThermalMoments::KineticThermalMoments (
    const amrex::Geometry& geometry, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& distribution,
    ThermalMomentOptions options)
    : m_geometry(geometry), m_cells(cells), m_distribution(distribution),
      m_options(std::move(options)),
      m_scalar(amrex::convert(cells, amrex::IntVect::TheNodeVector()),
               distribution, 1, 1),
      m_charge(m_scalar.boxArray(), distribution, 1, 1),
      m_pedestal(m_scalar.boxArray(), distribution, 1, 1),
      m_node_density(m_scalar.boxArray(), distribution, 1, 1),
      m_raw_density(cells, distribution, 1, 1),
      m_density(cells, distribution, 1, 1), m_active(cells, distribution, 1, 0),
      m_temperature(cells, distribution, 1, 1),
      m_node_temperature(m_scalar.boxArray(), distribution, 1,
                         m_options.nodal_ghosts),
      m_pressure(m_scalar.boxArray(), distribution, 1, m_options.nodal_ghosts),
      m_ohm_pressure(m_scalar.boxArray(), distribution, 1,
                     m_options.nodal_ghosts) {
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
    amrex::Abort("KineticThermalMoments supports only RZ m=0 and Cartesian 3D");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        cells.ixType().cellCentered() && cells.isDisjoint() &&
            cells.numPts() == geometry.Domain().numPts(),
        "KineticThermalMoments requires a complete, disjoint, single physical "
        "mesh");
    for (int b = 0; b < cells.size(); ++b) {
        AMREX_ALWAYS_ASSERT(geometry.Domain().contains(cells[b]));
    }
    auto const& o = m_options;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(o.nodal_ghosts >= 1,
                                     "thermal nodal ghosts must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(o.number_density_floor) && o.number_density_floor > 0.0 &&
            std::isfinite(o.active_density_floor) &&
            o.active_density_floor >= 0.0 && std::isfinite(o.gamma) &&
            o.gamma > 1.0 && std::isfinite(o.temperature_floor_kelvin) &&
            o.temperature_floor_kelvin >= 0.0,
        "KineticThermalMoments requires finite positive density floor and "
        "gamma>1");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(geometry.Domain().length(d) >=
                            std::max(2, o.nodal_ghosts));
        for (int side = 0; side < 2; ++side) {
            auto const bc = o.boundary[d][side];
            auto& current_bc = m_options.current_boundary[d][side];
            if (current_bc == MomentBoundary::Unspecified) {
                current_bc = bc;
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                current_bc == MomentBoundary::Periodic ||
                    current_bc == MomentBoundary::Axis ||
                    current_bc == MomentBoundary::PEC ||
                    current_bc == MomentBoundary::PMC,
                "Explicit current boundary must be periodic, axis, PEC or PMC");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                (geometry.isPeriodic(d) ==
                 (current_bc == MomentBoundary::Periodic)) &&
                    ((bc == MomentBoundary::Axis) ==
                     (current_bc == MomentBoundary::Axis)),
                "KineticThermalMoments needs geometry-consistent current "
                "images");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                (geometry.isPeriodic(d) && bc == MomentBoundary::Periodic) ||
                    (!geometry.isPeriodic(d) &&
                     (bc == MomentBoundary::PEC || bc == MomentBoundary::PMC ||
                      bc == MomentBoundary::Axis)),
                "KineticThermalMoments needs explicit, geometry-consistent "
                "boundary images");
#if defined(WARPX_DIM_RZ)
            AMREX_ALWAYS_ASSERT(bc != MomentBoundary::Axis ||
                                (d == 0 && side == 0));
#else
            AMREX_ALWAYS_ASSERT(bc != MomentBoundary::Axis);
#endif
        }
    }
#if defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 1 && geometry.ProbLo(0) == 0.0 &&
            geometry.Domain().smallEnd(0) == 0 && !geometry.isPeriodic(0) &&
            o.boundary[0][0] == MomentBoundary::Axis &&
            (o.boundary[0][1] == MomentBoundary::PEC ||
             o.boundary[0][1] == MomentBoundary::PMC) &&
            o.azimuthal_modes == 1,
        "KineticThermalMoments RZ requires m=0, r_lo=0 axis and radial "
        "PEC/PMC");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 0,
        "KineticThermalMoments 3D requires Cartesian geometry");
#endif
    for (int c = 0; c < 3; ++c) {
        m_current[c] = std::make_unique<amrex::MultiFab>(
            amrex::convert(cells, yee_type(c)), distribution, 1, 1);
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const faces =
            amrex::convert(cells, amrex::IntVect::TheDimensionVector(d));
        for (auto* family : {&m_face_density, &m_face_ion, &m_face_electron,
                             &m_face_velocity}) {
            (*family)[d] =
                std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
        }
    }
}

void
KineticThermalMoments::CheckLayout (const amrex::MultiFab& field,
                                    const amrex::IntVect& type,
                                    int component) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        field.boxArray() == amrex::convert(m_cells, type) &&
            field.DistributionMap() == m_distribution &&
            !field.hasEBFabFactory() && component >= 0 &&
            component < field.nComp(),
        "KineticThermalMoments requires matching native staggering/box "
        "order/DM, "
        "a valid component and no EB");
}

void
KineticThermalMoments::CopyNodal (const amrex::MultiFab& input, int component,
                                  amrex::MultiFab& output) const {
    CheckLayout(input, amrex::IntVect::TheNodeVector(), component);
    AMREX_ALWAYS_ASSERT(&input != &output);
    amrex::MultiFab::Copy(output, input, component, 0, 1, 0);
    output.OverrideSync(m_geometry.periodicity());
    output.FillBoundary(m_geometry.periodicity());
}

void
KineticThermalMoments::FillScalarImages (amrex::MultiFab& data) const {
    fill_scalar_images(data, m_geometry);
}

void
KineticThermalMoments::FillCurrentImages (amrex::MultiFab& data,
                                          int physical_component) const {
    fill_current_images(data, physical_component, m_geometry, m_options);
}

void
KineticThermalMoments::RestrictNodalScalar (const amrex::MultiFab& input,
                                            int component,
                                            amrex::MultiFab& cell_output) {
    CheckLayout(cell_output, amrex::IntVect::TheZeroVector(), 0);
    AMREX_ALWAYS_ASSERT(cell_output.nComp() == 1);
    CopyNodal(input, component, m_scalar);
    int const radial_lo = m_geometry.Domain().smallEnd(0);
    for (amrex::MFIter mfi(cell_output, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const a = m_scalar.const_array(mfi);
        auto const out = cell_output.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                out(i, j, k) = restrict_scalar(a, i, j, k, radial_lo);
            });
    }
    FillScalarImages(cell_output);
}

void
KineticThermalMoments::RestrictNativeMoment (const amrex::MultiFab& input,
                                             int component,
                                             amrex::MultiFab& cell_output) {
    CheckLayout(cell_output, amrex::IntVect::TheZeroVector(), 0);
    AMREX_ALWAYS_ASSERT(cell_output.nComp() == 1);
    CopyNodal(input, component, m_scalar);
    int const radial_hi = m_geometry.Domain().bigEnd(0) + 1;
    int const radial_lo = m_geometry.Domain().smallEnd(0);
    bool const corrected = m_options.verboncoeur_axis_correction;
    for (amrex::MFIter mfi(m_scalar, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const a = m_scalar.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                a(i, j, k) *= native_volume_ratio(i, radial_hi, corrected);
            });
    }
    m_scalar.OverrideSync(m_geometry.periodicity());
    m_scalar.FillBoundary(m_geometry.periodicity());
    for (amrex::MFIter mfi(cell_output, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const a = m_scalar.const_array(mfi);
        auto const out = cell_output.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                out(i, j, k) = restrict_scalar(a, i, j, k, radial_lo);
            });
    }
    FillScalarImages(cell_output);
}

void
KineticThermalMoments::RestrictCurrent (const YeeCurrentView& input,
                                        const ThermalFaceView& face_output) {
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(input[c]);
        CheckLayout(*input[c], yee_type(c), 0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            input[c]->nComp() == 1,
            "Thermal current map requires one physical m=0 component");
        amrex::MultiFab::Copy(*m_current[c], *input[c], 0, 0, 1, 0);
        FillCurrentImages(*m_current[c], c);
    }
    int const radial_hi = m_geometry.Domain().bigEnd(0) + 1;
    bool const corrected = m_options.verboncoeur_axis_correction;
    int const radial_lo = m_geometry.Domain().smallEnd(0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(face_output[d]);
        CheckLayout(*face_output[d], amrex::IntVect::TheDimensionVector(d), 0);
        AMREX_ALWAYS_ASSERT(face_output[d]->nComp() == 1);
        int const c = physical_direction(d);
        auto const low=m_geometry.Domain().smallEnd(d);
        auto const high=m_geometry.Domain().bigEnd(d)+1;
        bool const closed_low=m_options.current_boundary[d][0]==MomentBoundary::PMC;
        bool const closed_high=m_options.current_boundary[d][1]==MomentBoundary::PMC;
        for (amrex::MFIter mfi(*face_output[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const a = m_current[c]->const_array(mfi);
            auto const out = face_output[d]->array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                   int k) {
#if defined(WARPX_DIM_RZ)
                if (d == 0 && i == radial_lo) {
                    out(i, j, k) = 0.0;
                    return;
                }
#endif
                int const coordinate[3]={i,j,k};
                if ((closed_low && coordinate[d]==low) ||
                    (closed_high && coordinate[d]==high)) {
                    // Enforce the actual closed normal-current constraint.
                    // Summing metric odd ghosts leaves roundoff of either
                    // sign, which would look like physical thermal inflow.
                    // Ordinary PEC normal exchange and all interior rows stay
                    // on the conservative transfer below.
                    out(i,j,k)=0.0;
                    return;
                }
                Real sum = 0.0;
                // Each target face has two normal source edges and 2^(D-1)
                // tangential nodes. These coefficients integrate flux, not J.
                for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
                    int src[3] = {i, j, k};
                    Real const weight=current_slot(bits,d,i,radial_lo,radial_hi,corrected,src);
                    sum += weight * a(src[0], src[1], src[2]);
                }
                out(i, j, k) = sum;
            });
        }
        face_output[d]->OverrideSync(m_geometry.periodicity());
    }
}

bool KineticThermalMoments::ValidateCurrentRemainder(KineticThermalStateView const& state) const {
    int count=0;for(auto* p:state.current_remainder)count+=p!=nullptr;
    int minimum=count,maximum=count;amrex::ParallelDescriptor::ReduceIntMin(minimum);
    amrex::ParallelDescriptor::ReduceIntMax(maximum);
    if(minimum!=maximum || (count!=0 && count!=3)) return false;
    if(!count)return true;
    bool valid=remainder::ArithmeticSupported() &&
        (state.current_input==ThermalCurrentInput::SignedElectron || state.current_input==ThermalCurrentInput::TotalPlasma);
    for(int c=0;c<3;++c) {
        auto const* high=state.current[c];auto const* low=state.current_remainder[c];auto const* ion=state.ion_current[c];
        valid=valid && high && ion && low && high->nComp()==1 &&
            high->boxArray()==amrex::convert(m_cells,yee_type(c)) &&
            high->DistributionMap()==m_distribution;
        if(high && ion && low)valid=valid && remainder::Layout(*low,*high) && remainder::Layout(*ion,*high) &&
            !remainder::Overlap(*low,*high) && !remainder::Overlap(*low,*ion);
    }
    if(!remainder::All(valid))return false;
    for(int c=0;c<3;++c) if(!finite(*state.current[c]) || !finite(*state.current_remainder[c]) ||
        !finite(*state.ion_current[c])) return false;
    return true;
}
bool KineticThermalMoments::RestrictCurrentPair(YeeCurrentView const& input,YeeCurrentView const& low,
    ThermalFaceView const& represented,ThermalFaceView const& output) {
    KineticThermalStateView state{m_charge};state.current=input;state.current_remainder=low;
    state.ion_current=input;state.current_input=ThermalCurrentInput::SignedElectron;
    if(!ValidateCurrentRemainder(state))return false;
    bool valid=true;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        valid=valid && represented[d] && output[d];
        if(represented[d] && output[d]) {
            valid=valid && remainder::Layout(*represented[d],*m_face_electron[d]) &&
                remainder::Layout(*output[d],*represented[d]);
            for(int q=0;q<AMREX_SPACEDIM;++q) {
                if(represented[q])valid=valid && !remainder::Overlap(*output[d],*represented[q]);
                if(q!=d && output[q])valid=valid && !remainder::Overlap(*output[d],*output[q]);
            }
            for(int c=0;c<3;++c)if(input[c] && low[c]) valid=valid &&
                !remainder::Overlap(*output[d],*input[c]) && !remainder::Overlap(*output[d],*low[c]);
        }
    }
    if(!remainder::All(valid))return false;
    return RestrictCurrentRemainder(state,represented,output);
}
bool KineticThermalMoments::RestrictCurrentRemainder(KineticThermalStateView const& state,
    ThermalFaceView const& face_high,ThermalFaceView const& face_low,bool publish_pair) {
    bool const total=state.current_input==ThermalCurrentInput::TotalPlasma;
    for(int c=0;c<3;++c) {
        if(!m_current_pair[c])m_current_pair[c]=std::make_unique<amrex::MultiFab>(m_current[c]->boxArray(),m_distribution,1,1);
        if(!m_current_remainder[c])m_current_remainder[c]=std::make_unique<amrex::MultiFab>(m_current[c]->boxArray(),m_distribution,1,1);
        for(amrex::MFIter mfi(*m_current_pair[c]);mfi.isValid();++mfi) {
            auto const high=state.current[c]->const_array(mfi),small=state.current_remainder[c]->const_array(mfi),ion=state.ion_current[c]->const_array(mfi);
            auto const h=m_current_pair[c]->array(mfi),l=m_current_remainder[c]->array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                namespace dd=warpx::ohm::compensated;
                auto value=dd::Pair{high(i,j,k),small(i,j,k)};
                if(total)value=dd::Add(value,{-ion(i,j,k),0.});
                h(i,j,k)=value.hi;l(i,j,k)=value.lo;
            });
        }
        fill_current_pair_images(*m_current_pair[c],*m_current_remainder[c],c,m_geometry,m_options);
    }
    int const radial_hi = m_geometry.Domain().bigEnd(0) + 1;
    bool const corrected = m_options.verboncoeur_axis_correction;
    int const radial_lo = m_geometry.Domain().smallEnd(0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(face_low[d]);
        CheckLayout(*face_low[d], amrex::IntVect::TheDimensionVector(d), 0);
        AMREX_ALWAYS_ASSERT(face_low[d]->nComp() == 1);
        int const c = physical_direction(d);
        auto const low=m_geometry.Domain().smallEnd(d);
        auto const high=m_geometry.Domain().bigEnd(d)+1;
        bool const closed_low=m_options.current_boundary[d][0]==MomentBoundary::PMC;
        bool const closed_high=m_options.current_boundary[d][1]==MomentBoundary::PMC;
        for (amrex::MFIter mfi(*face_low[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const a = m_current_pair[c]->const_array(mfi);
            auto const l = m_current_remainder[c]->const_array(mfi);
            auto const represented = face_high[d]->const_array(mfi);
            auto const out = face_low[d]->array(mfi);
            auto const pair_high=face_high[d]->array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                   int k) {
#if defined(WARPX_DIM_RZ)
                if (d == 0 && i == radial_lo) {
                    out(i, j, k) = 0.0;
                    if(publish_pair)pair_high(i,j,k)=0.0;
                    return;
                }
#endif
                int const coordinate[3]={i,j,k};
                if ((closed_low && coordinate[d]==low) ||
                    (closed_high && coordinate[d]==high)) {
                    // Enforce the actual closed normal-current constraint.
                    // Summing metric odd ghosts leaves roundoff of either
                    // sign, which would look like physical thermal inflow.
                    // Ordinary PEC normal exchange and all interior rows stay
                    // on the conservative transfer below.
                    out(i,j,k)=0.0;
                    if(publish_pair)pair_high(i,j,k)=0.0;
                    return;
                }
                namespace dd=warpx::ohm::compensated;
                dd::Pair sum{0.,0.};
                // Each target face has two normal source edges and 2^(D-1)
                // tangential nodes. These coefficients integrate flux, not J.
                for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
                    int src[3] = {i, j, k};
                    Real const weight=current_slot(bits,d,i,radial_lo,radial_hi,corrected,src);
                    sum=dd::Add(sum,dd::Multiply({a(src[0],src[1],src[2]),l(src[0],src[1],src[2])},weight));
                }
                if(publish_pair){pair_high(i,j,k)=sum.hi;out(i,j,k)=sum.lo;}
                else out(i,j,k)=dd::WithHigh(sum,represented(i,j,k)).lo;
            });
        }
        face_low[d]->OverrideSync(m_geometry.periodicity());
        if(publish_pair)face_high[d]->OverrideSync(m_geometry.periodicity());
    }
    for(auto const* face:face_low) if(!finite(*face)) return false;
    if(publish_pair)for(auto const* face:face_high)if(!finite(*face))return false;
    if(!publish_pair)m_has_current_remainder=true;
    return true;
}

bool KineticThermalMoments::RestrictCurrentIncrement(YeeCurrentView const& high,
    YeeCurrentView const& low,ThermalFaceView const& face_high,ThermalFaceView const& face_low) {
    KineticThermalStateView state{m_charge};state.current=high;state.current_remainder=low;
    state.ion_current=high;state.current_input=ThermalCurrentInput::SignedElectron;
    if(!ValidateCurrentRemainder(state))return false;
    bool valid=remainder::ArithmeticSupported();
    for(int d=0;d<AMREX_SPACEDIM;++d) for(auto* output:{face_high[d],face_low[d]}) {
        valid=valid && output;
        if(!output)continue;
        valid=valid && remainder::Layout(*output,*m_face_electron[d]);
        for(int c=0;c<3;++c)valid=valid && !remainder::Overlap(*output,*high[c]) && !remainder::Overlap(*output,*low[c]);
        for(int q=0;q<AMREX_SPACEDIM;++q)for(auto* other:{face_high[q],face_low[q]})
            if(other && other!=output)valid=valid && !remainder::Overlap(*output,*other);
    }
    // Pointer equality needs an explicit test in addition to storage overlap.
    for(int d=0;d<AMREX_SPACEDIM;++d)for(int q=d;q<AMREX_SPACEDIM;++q) {
        valid=valid && face_high[d]!=face_low[q] && face_low[d]!=face_high[q];
        if(q!=d)valid=valid && face_high[d]!=face_high[q] && face_low[d]!=face_low[q];
    }
    if(!remainder::All(valid))return false;
    return RestrictCurrentRemainder(state,face_high,face_low,true);
}
bool KineticThermalMoments::RestrictNativeMomentIncrement(amrex::MultiFab const& high,
    amrex::MultiFab const& low,amrex::MultiFab& out,amrex::MultiFab& small) {
    bool valid=remainder::ArithmeticSupported() && remainder::Layout(high,m_node_density) &&
        remainder::Layout(low,high) && remainder::Layout(out,m_density) && remainder::Layout(small,out) &&
        out.nGrowVect()==small.nGrowVect() && out.nGrowVect().allLE(amrex::IntVect(1)) &&
        !remainder::Overlap(out,small);
    for(auto* dst:{&out,&small})for(auto const* src:{&high,&low})valid=valid && !remainder::Overlap(*dst,*src);
    if(!remainder::All(valid) || !finite(high) || !finite(low))return false;
    amrex::MultiFab h(m_node_density.boxArray(),m_distribution,1,1),l(h.boxArray(),m_distribution,1,1);
    CopyNodal(high,0,h);CopyNodal(low,0,l);
    int const radial_hi=m_geometry.Domain().bigEnd(0)+1,radial_lo=m_geometry.Domain().smallEnd(0);
    bool const corrected=m_options.verboncoeur_axis_correction;
    for(amrex::MFIter it(h);it.isValid();++it) {
        auto a=h.array(it),b=l.array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            auto value=remainder::Multiply({a(i,j,k),b(i,j,k)},native_volume_ratio(i,radial_hi,corrected));
            a(i,j,k)=value.hi;b(i,j,k)=value.lo;
        });
    }
    h.OverrideSync(m_geometry.periodicity());l.OverrideSync(m_geometry.periodicity());
    h.FillBoundary(m_geometry.periodicity());l.FillBoundary(m_geometry.periodicity());
    for(amrex::MFIter it(out);it.isValid();++it) {
        auto a=h.const_array(it),b=l.const_array(it);auto o=out.array(it),s=small.array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            using namespace remainder;
            Real const left=radial_left_weight(i,radial_lo);
            Pair const base{a(i,j,k),b(i,j,k)};Pair sum{0.,0.};
            for(int bits=0;bits<(1<<AMREX_SPACEDIM);++bits) {
                int const di=bits&1,dj=(bits>>1)&1;
#if AMREX_SPACEDIM==3
                int const dk=(bits>>2)&1;
#else
                int const dk=0;
#endif
                Real const weight=(di==0?left:1.-left)/static_cast<Real>(1<<(AMREX_SPACEDIM-1));
                sum=Add(sum,Multiply(Add({a(i+di,j+dj,k+dk),b(i+di,j+dj,k+dk)},Negate(base)),weight));
            }
            auto value=Add(base,sum);o(i,j,k)=value.hi;s(i,j,k)=value.lo;
        });
    }
    FillScalarImages(out);FillScalarImages(small);return finite(out)&&finite(small);
}

void
KineticThermalMoments::ApplyPressureImages () {
    apply_pressure_images(m_pressure, m_ohm_pressure, m_geometry, m_options);
}

void KineticThermalMoments::ApplyPressureIncrementImages(amrex::MultiFab& high,amrex::MultiFab& low,
    amrex::MultiFab& output_high,amrex::MultiFab& output_low) {
    apply_pressure_images(high,output_high,m_geometry,m_options);
    apply_pressure_images(low,output_low,m_geometry,m_options);
}

bool
KineticThermalMoments::Evaluate (const KineticThermalStateView& state) {
    if(m_options.retain_current_remainder) {
        if(!ValidateCurrentRemainder(state))return false; // collective before mutation
    } else {
        for(auto const* low:state.current_remainder)AMREX_ALWAYS_ASSERT(low==nullptr);
    }
    m_has_current_remainder=false;
    m_has_pressure = false;
    m_has_currents = false;
    AMREX_ALWAYS_ASSERT(
        state.current_input == ThermalCurrentInput::SignedElectron ||
        state.current_input == ThermalCurrentInput::TotalPlasma);
    // Inputs and derived outputs must have independent storage. In particular,
    // do not accept this object's previous density/temperature as a trial
    // input.
    auto independent = [&] (const amrex::MultiFab* input) {
        if (!input) {
            return;
        }
        for (const auto* owned :
             {&m_scalar, &m_charge, &m_pedestal, &m_node_density,
              &m_raw_density, &m_density, &m_active, &m_temperature,
              &m_node_temperature, &m_pressure, &m_ohm_pressure}) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                input != owned, "KineticThermalMoments trial inputs must be "
                                "independent of its outputs");
        }
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            for (const auto* family : {&m_face_density, &m_face_ion,
                                       &m_face_electron, &m_face_velocity}) {
                AMREX_ALWAYS_ASSERT(input != (*family)[d].get());
            }
        }
    };
    independent(&state.charge);
    independent(state.pedestal);
    independent(state.energy);
    for (int c = 0; c < 3; ++c) {
        independent(state.ion_current[c]);
        independent(state.current[c]);
    }
    CopyNodal(state.charge, state.charge_component, m_charge);
    if (!finite(m_charge)) {
        return false;
    }
    if (state.pedestal) {
        CopyNodal(*state.pedestal, 0, m_pedestal);
        if (!finite(m_pedestal) || m_pedestal.min(0) < 0.0) {
            return false;
        }
    } else {
        m_pedestal.setVal(0.0);
    }
    Real const floor = m_options.number_density_floor;
    for (amrex::MFIter mfi(m_node_density, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const rho = m_charge.const_array(mfi);
        auto const ped = m_pedestal.const_array(mfi);
        auto const n = m_node_density.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                n(i, j, k) = amrex::max(
                    (rho(i, j, k) + ped(i, j, k)) / PhysConst::q_e, floor);
            });
    }
    FillScalarImages(m_node_density);
    RestrictNativeMoment(m_charge, 0, m_raw_density);
    m_raw_density.mult(1.0 / PhysConst::q_e, 0, 1, 1);
    RestrictNativeMoment(m_node_density, 0, m_density);
    if (!finite(m_raw_density) || !finite(m_density)) {
        return false;
    }
    Real const gate = (state.pedestal || m_options.halo_unfreeze)
                          ? 0.0
                          : m_options.active_density_floor;
    for (amrex::MFIter mfi(m_active, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        auto const raw = m_raw_density.const_array(mfi);
        auto const active = m_active.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                active(i, j, k) = raw(i, j, k) > gate ? 1.0 : 0.0;
            });
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (amrex::MFIter mfi(*m_face_density[d], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            auto const n = m_density.const_array(mfi);
            auto const face = m_face_density[d]->array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int left[3] = {i, j, k};
                    --left[d];
                    Real const base = n(left[0], left[1], left[2]);
                    face(i, j, k) = base + 0.5 * (n(i, j, k) - base);
                });
        }
    }
    bool const currents = state.ion_current[0] || state.current[0];
    ThermalFaceView fi{}, fe{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        fi[d] = m_face_ion[d].get();
        fe[d] = m_face_electron[d].get();
        fi[d]->setVal(0.0);
        fe[d]->setVal(0.0);
        m_face_velocity[d]->setVal(0.0);
    }
    if (currents) {
        RestrictCurrent(state.ion_current, fi);
        RestrictCurrent(state.current, fe);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (state.current_input == ThermalCurrentInput::TotalPlasma) {
                amrex::MultiFab::Subtract(*fe[d], *fi[d], 0, 0, 1, 0);
            }
            if (!finite(*fi[d]) || !finite(*fe[d])) {
                return false;
            }
            for (amrex::MFIter mfi(*m_face_velocity[d],
                                   amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                auto const n = m_face_density[d]->const_array(mfi);
                auto const je = fe[d]->const_array(mfi);
                auto const u = m_face_velocity[d]->array(mfi);
                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(
                                                      int i, int j, int k) {
                    u(i, j, k) = -je(i, j, k) / (PhysConst::q_e * n(i, j, k));
                });
            }
            if (!finite(*m_face_velocity[d])) {
                return false;
            }
        }
        if(state.current_remainder[0]) {
            ThermalFaceView low{};
            for(int d=0;d<AMREX_SPACEDIM;++d) {
                if(!m_face_remainder[d])m_face_remainder[d]=std::make_unique<amrex::MultiFab>(
                    m_face_electron[d]->boxArray(),m_distribution,1,0);
                low[d]=m_face_remainder[d].get();
            }
            if(!RestrictCurrentRemainder(state,fe,low))return false;
        }
        m_has_currents = true;
    } else {
        for (int c = 0; c < 3; ++c) {
            AMREX_ALWAYS_ASSERT(state.ion_current[c] == nullptr &&
                                state.current[c] == nullptr);
        }
    }
    m_temperature.setVal(0.0);
    m_node_temperature.setVal(0.0);
    m_pressure.setVal(0.0);
    m_ohm_pressure.setVal(0.0);
    if (!state.energy) {
        return true;
    }
    CheckLayout(*state.energy, amrex::IntVect::TheZeroVector(), 0);
    if (!finite(*state.energy)) {
        return false;
    }
    Real const gamma_minus_one = m_options.gamma - 1.0;
    for (amrex::MFIter mfi(m_temperature, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const energy = state.energy->const_array(mfi);
        auto const n = m_density.const_array(mfi);
        auto const t = m_temperature.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               t(i, j, k) = gamma_minus_one * energy(i, j, k) /
                                            (PhysConst::kb * n(i, j, k));
                           });
    }
    if (!finite(m_temperature) ||
        m_temperature.min(0) < m_options.temperature_floor_kelvin) {
        return false;
    }
    FillScalarImages(m_temperature);
    // Pressure uses the same density ratio as k_B*n_node*average(T_cell),
    // without its temperature conversion round trip. Energy has no required
    // ghosts, so give this local copy the same scalar images as temperature.
    // Forming n_node/n_cell before multiplying U also avoids storing a tiny
    // U/n_cell intermediate. The returned temperature arithmetic is unchanged.
    amrex::MultiFab pressure_energy(m_cells, m_distribution, 1, 1);
    amrex::MultiFab::Copy(pressure_energy, *state.energy, 0, 0, 1, 0);
    FillScalarImages(pressure_energy);
    for (amrex::MFIter mfi(m_node_temperature, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const t = m_temperature.const_array(mfi);
        auto const node_t = m_node_temperature.array(mfi);
        auto const n = m_node_density.const_array(mfi);
        auto const cell_n = m_density.const_array(mfi);
        auto const u = pressure_energy.const_array(mfi);
        auto const p = m_pressure.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const base = t(i, j, k);
                Real sum = 0.0;
                Real const node_n = n(i, j, k);
                Real const pressure_base = u(i, j, k) *
                                           (node_n / cell_n(i, j, k));
                Real pressure_sum = 0.0;
                for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
                    int const di = bits & 1;
                    int const dj = (bits >> 1) & 1;
#if (AMREX_SPACEDIM == 3)
                    int const dk = (bits >> 2) & 1;
#else
                int const dk = 0;
#endif
                    sum += t(i - di, j - dj, k - dk) - base;
                    pressure_sum += u(i - di, j - dj, k - dk) *
                                        (node_n / cell_n(i - di, j - dj, k - dk)) -
                                    pressure_base;
                }
                node_t(i, j, k) =
                    base + sum / static_cast<Real>(1 << AMREX_SPACEDIM);
                p(i, j, k) = gamma_minus_one *
                    (pressure_base + pressure_sum /
                                         static_cast<Real>(1 << AMREX_SPACEDIM));
            });
    }
    if (!finite(m_pressure)) {
        return false;
    }
    FillScalarImages(m_node_temperature);
    ApplyPressureImages();
    m_has_pressure = true;
    return true;
}

} // namespace warpx::thermal
