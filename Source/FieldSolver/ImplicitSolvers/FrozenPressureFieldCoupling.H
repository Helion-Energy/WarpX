/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FROZEN_PRESSURE_FIELD_COUPLING_H_
#define WARPX_FROZEN_PRESSURE_FIELD_COUPLING_H_

#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>

namespace warpx::thermal {

// Scalar E-form Ohm pressure response at an actual native Yee point. All
// density parameters here are CHARGE density [C/m^3], not number density.
// Widths are absolute; map the model's dimensionless widths times charge_floor.
struct PressureResponseParameters {
    amrex::Real charge_floor = 1.0;
    amrex::Real floor_width = 0.0;
    amrex::Real holmstrom_width = 0.0;
    amrex::Real axis_radius = 0.0;
    amrex::Real axis_rolloff = 0.0;
    bool pressure_enabled = true;
    bool holmstrom = false;
    bool conductor_raw_gate = false;
};

// Interp always loops over all three array indices, including in RZ. Keep
// inactive dimensions identical to the nodal source (one), or Interp would
// average a nonexistent second k plane. Active entries use the actual Yee
// MultiFab centering, preserving the native Cartesian counterpart.
inline amrex::GpuArray<int, 3>
PressureResponseStagger (const amrex::IntVect& type) noexcept {
    amrex::GpuArray<int, 3> stagger{1, 1, 1};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        stagger[d] = type[d];
    }
    return stagger;
}

// raw and pedestal must be interpolated independently with the native Ohm
// interpolation, BEFORE this floor/gate. row_response is an explicit frozen
// multiplier in [0,1]: end-region gate times prescribed/recovery row response.
// A replaced row has zero response. This is NOT a thermal face density map.
// Parameters must be finite, charge_floor>0, widths/radii>=0, pedestal>=0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
PressureResponseWeight (amrex::Real raw, amrex::Real pedestal,
                        amrex::Real radius, PressureResponseParameters const& p,
                        amrex::Real row_response = 1.0) noexcept {
    if (!p.pressure_enabled || row_response == 0.0 ||
        (p.conductor_raw_gate && !(raw > 0.0))) {
        return 0.0;
    }
    bool const confined = p.holmstrom && p.axis_radius > 0.0;
    bool const smooth = p.holmstrom && p.holmstrom_width > 0.0;
    if (p.holmstrom && !smooth && raw < p.charge_floor &&
        (!confined || radius < p.axis_radius)) {
        return 0.0;
    }
    amrex::Real const total = raw + pedestal;
    amrex::Real const delta = total - p.charge_floor;
    amrex::Real const divisor =
        p.floor_width > 0.0
            ? 0.5 * (total + p.charge_floor +
                     std::sqrt(delta * delta + p.floor_width * p.floor_width))
            : (total > p.charge_floor ? total : p.charge_floor);
    amrex::Real gate = 1.0;
    if (smooth) {
        amrex::Real const g =
            0.5 * (1.0 + std::tanh((raw - p.charge_floor) / p.holmstrom_width));
        amrex::Real const axis =
            !confined ? 1.0
                      : (p.axis_rolloff > 0.0
                             ? 0.5 * (1.0 - std::tanh((radius - p.axis_radius) /
                                                      p.axis_rolloff))
                             : (radius < p.axis_radius ? 1.0 : 0.0));
        gate = confined ? 1.0 - (1.0 - g) * axis : g;
    }
    return gate * row_response / divisor;
}

struct FrozenPressureFieldOptions {
    amrex::Real gamma = 5.0 / 3.0;
    std::array<std::array<MomentBoundary, 2>, AMREX_SPACEDIM> boundary{};
    int azimuthal_modes = 1;
    // A numerator/curl-curl/tensor-transformed Ohm solve needs its actual
    // frozen inverse response after the gradient. It is not this operator.
    bool transformed_ohm_form = false;
};

// Frozen upper-right block B = dF_E/dU for the pressure-only part of
// F_E = E_T + E_L - E_Ohm [V/m]. Input delta_U is cell energy density [J/m^3].
//
// delta_T_C = (gamma-1) delta_U / (kB n_C),
// delta_P_N = kB n_N I_C->N(delta_T_C), with EXACT kinetic moment PEC images;
// B delta_U = +weight_Yee grad_Yee(delta_P_N). No theta*dt or norm scaling.
//
// The caller freezes all coefficients once per PC rebuild, then applies
// z_U=D^-1 r_U; z_E=A^-1(r_E-B z_U). Full residual/Jv remain live. Density,
// current, resistivity/viscosity response and the lower-left block are omitted.
// Only one full RZ m=0 mesh without EB/AMR is qualified. Other geometries and
// transformed Ohm forms reject explicitly; existing 3D operators are unchanged.
class FrozenPressureFieldCoupling {
  public:
    using FieldView = std::array<amrex::MultiFab*, 3>;
    using ConstFieldView = std::array<const amrex::MultiFab*, 3>;
    FrozenPressureFieldCoupling (const amrex::Geometry& geometry,
                                 const amrex::BoxArray& cells,
                                 const amrex::DistributionMapping& distribution,
                                 FrozenPressureFieldOptions options);

    // Deep-copy valid n_C [m^-3], n_N [m^-3], and native Yee g/rho [m^3/C].
    // Density must be finite/positive; weights finite/nonnegative. No borrowed
    // live stage/moment storage survives. Validation is transactional: false
    // retains the previous frozen operator. Layout mistakes are programmer
    // errors.
    void Invalidate () noexcept { m_frozen = false; }

    bool Freeze (const amrex::MultiFab& cell_density,
                 const amrex::MultiFab& nodal_density,
                 ConstFieldView pressure_response_weight);

    // Linear signed kernel; negative delta_U is allowed. Replaces valid output
    // values, including exactly zero toroidal response and zero masked rows.
    // Uses preallocated device scratch, neighbor exchanges, no coefficient
    // reductions, particle pushes, recovery, source callbacks or live model
    // reads. Caller owns finite-vector checks. Input and outputs must not
    // alias.
    void Apply (FieldView output, const amrex::MultiFab& delta_energy);

    bool
    IsFrozen () const {
        return m_frozen;
    }
    std::uint64_t
    FreezeCount () const {
        return m_freezes;
    }
    std::uint64_t
    ApplyCount () const {
        return m_applies;
    }
    const amrex::MultiFab&
    PressureIncrement () const {
        return m_ohm_pressure;
    }
    const amrex::MultiFab&
    TemperatureIncrement () const {
        return m_temperature;
    }

  private:
    void CheckLayout (const amrex::MultiFab& field,
                      const amrex::IntVect& type) const;
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells;
    amrex::DistributionMapping m_distribution;
    FrozenPressureFieldOptions m_options;
    amrex::MultiFab m_cell_density, m_node_density, m_temperature,
        m_node_temperature, m_pressure, m_ohm_pressure;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_weight;
    bool m_frozen = false;
    std::uint64_t m_freezes = 0, m_applies = 0;
};
} // namespace warpx::thermal
#endif
