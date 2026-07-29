/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef THETA_IMPLICIT_MHD_K_H_
#define THETA_IMPLICIT_MHD_K_H_

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

namespace theta_implicit_mhd {
struct FluxParameters {
    amrex::Real density_floor;
    amrex::Real charge_to_mass;
    amrex::Real gamma_e;
    amrex::Real gamma_i;
    amrex::Real reference_density;
    amrex::Real reference_ion_pressure;
    amrex::Real electron_pressure_floor;
    bool include_hall_term;
    bool use_rusanov;
};

struct CellState {
    amrex::Real density;
    amrex::Real safe_density;
    amrex::GpuArray<amrex::Real, 3> momentum;
    amrex::GpuArray<amrex::Real, 3> ion_velocity;
    amrex::Real electron_energy;
    amrex::Real ion_pressure;
    amrex::Real electron_pressure;
    amrex::Real electron_velocity_normal;
    amrex::Real wave_speed;
};

struct FaceFlux {
    amrex::Real mass;
    amrex::GpuArray<amrex::Real, 3> momentum;
    amrex::Real electron_energy;
    amrex::Real electron_velocity;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
admissible_step_fraction (const amrex::Real value, const amrex::Real old_value,
                          const amrex::Real direction,
                          const amrex::Real requested_step,
                          const amrex::Real floor,
                          const amrex::Real theta) noexcept {
    const amrex::Real lower_bound = (1.0 - theta) * old_value + theta * floor;
    const amrex::Real proposed_change = requested_step * direction;
    if (proposed_change >= 0.0) {
        return 1.0;
    }
    return std::max(0.0,
                    std::min(1.0, (value - lower_bound) / (-proposed_change)));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
ion_pressure (const amrex::Real density,
              const FluxParameters& parameters) noexcept {
    if (parameters.reference_ion_pressure == 0.0) {
        return 0.0;
    }
    return parameters.reference_ion_pressure *
           std::pow(parameters.density_floor > density
                        ? parameters.density_floor /
                              parameters.reference_density
                        : density / parameters.reference_density,
                    parameters.gamma_i);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state (const amrex::Array4<const amrex::Real>& density,
                 const amrex::Array4<const amrex::Real>& momentum,
                 const amrex::Array4<const amrex::Real>& electron_energy,
                 const amrex::Array4<const amrex::Real>& current, const int i,
                 const int j, const int k, const int normal_direction,
                 const FluxParameters& parameters) noexcept {
    CellState state;
    state.density = density(i, j, k);
    state.safe_density = std::max(state.density, parameters.density_floor);
    for (int component = 0; component < 3; ++component) {
        state.momentum[component] = momentum(i, j, k, component);
        state.ion_velocity[component] =
            state.momentum[component] / state.safe_density;
    }
    state.electron_energy = electron_energy(i, j, k);
    state.ion_pressure = ion_pressure(state.safe_density, parameters);
    state.electron_pressure =
        std::max((parameters.gamma_e - 1.0) * state.electron_energy,
                 parameters.electron_pressure_floor);

    const amrex::Real charge_density =
        parameters.charge_to_mass * state.safe_density;
    state.electron_velocity_normal =
        state.ion_velocity[normal_direction] -
        (parameters.include_hall_term
             ? current(i, j, k, normal_direction) / charge_density
             : 0.0);

    const amrex::Real sound_speed_squared =
        std::max((parameters.gamma_i * state.ion_pressure +
                  parameters.gamma_e * state.electron_pressure) /
                     state.safe_density,
                 0.0);
    state.wave_speed = std::max(std::abs(state.ion_velocity[normal_direction]) +
                                    std::sqrt(sound_speed_squared),
                                std::abs(state.electron_velocity_normal));
    return state;
}

/**
 * \brief Compute a centered or local Lax--Friedrichs flux at one cell face.
 *
 * The ion block is a conservative hydrodynamic flux with total fluid pressure
 * P_i+P_e. Magnetic induction remains in the staggered Ohm/Faraday operator,
 * so this is deliberately not described as a full MHD Riemann flux.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FaceFlux
face_flux (const amrex::Array4<const amrex::Real>& density,
           const amrex::Array4<const amrex::Real>& momentum,
           const amrex::Array4<const amrex::Real>& electron_energy,
           const amrex::Array4<const amrex::Real>& current, const int il,
           const int jl, const int kl, const int ir, const int jr, const int kr,
           const int normal_direction,
           const FluxParameters& parameters) noexcept {
    const CellState left =
        load_cell_state(density, momentum, electron_energy, current, il, jl, kl,
                        normal_direction, parameters);
    const CellState right =
        load_cell_state(density, momentum, electron_energy, current, ir, jr, kr,
                        normal_direction, parameters);
    const amrex::Real alpha = parameters.use_rusanov
                                  ? std::max(left.wave_speed, right.wave_speed)
                                  : 0.0;

    FaceFlux flux;
    flux.mass = 0.5 * (left.momentum[normal_direction] +
                       right.momentum[normal_direction] -
                       alpha * (right.density - left.density));
    for (int component = 0; component < 3; ++component) {
        const amrex::Real left_flux =
            left.momentum[normal_direction] * left.ion_velocity[component] +
            (component == normal_direction
                 ? left.ion_pressure + left.electron_pressure
                 : 0.0);
        const amrex::Real right_flux =
            right.momentum[normal_direction] * right.ion_velocity[component] +
            (component == normal_direction
                 ? right.ion_pressure + right.electron_pressure
                 : 0.0);
        flux.momentum[component] =
            0.5 *
            (left_flux + right_flux -
             alpha * (right.momentum[component] - left.momentum[component]));
    }
    flux.electron_energy =
        0.5 * (left.electron_energy * left.electron_velocity_normal +
               right.electron_energy * right.electron_velocity_normal -
               alpha * (right.electron_energy - left.electron_energy));
    flux.electron_velocity =
        0.5 * (left.electron_velocity_normal + right.electron_velocity_normal);
    return flux;
}
} // namespace theta_implicit_mhd

#endif
