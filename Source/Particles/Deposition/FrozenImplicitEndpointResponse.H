/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_FROZEN_IMPLICIT_ENDPOINT_RESPONSE_H_
#define WARPX_FROZEN_IMPLICIT_ENDPOINT_RESPONSE_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>

#include <cmath>

namespace warpx::particles
{

/** Exact derivative of the explicitly approximate frozen-gamma orbit map.
 * midpoint_velocity_direction is the native MM kernel's physical velocity
 * response, rotated to Cartesian. B, gather weights and implicit gamma are
 * held. Old momentum and endpoint position are fixed base inputs. The native
 * endpoint velocity conversion is differentiated, including its gamma.
 * This does not claim to differentiate the full Boris/Picard fixed point.
 */
AMREX_GPU_HOST_DEVICE inline bool
FrozenImplicitEndpointResponse (amrex::GpuArray<double, 3> const &endpoint_momentum,
                                amrex::GpuArray<double, 3> const &midpoint_velocity_direction,
                                double implicit_gamma_inverse, double dt, double inverse_c_squared,
                                amrex::GpuArray<double, 3> &endpoint_velocity,
                                amrex::GpuArray<double, 3> &position_direction,
                                amrex::GpuArray<double, 3> &velocity_direction) noexcept
{
    endpoint_velocity = {};
    position_direction = {};
    velocity_direction = {};
    if (!(implicit_gamma_inverse > 0.) || !std::isfinite(implicit_gamma_inverse) || !(dt > 0.) ||
        !std::isfinite(dt) || !(inverse_c_squared >= 0.) || !std::isfinite(inverse_c_squared))
    {
        return false;
    }
    double norm = 0., dot = 0.;
    amrex::GpuArray<double, 3> du{};
    for (int d = 0; d < 3; ++d)
    {
        if (!std::isfinite(endpoint_momentum[d]) || !std::isfinite(midpoint_velocity_direction[d]))
        {
            return false;
        }
        du[d] = 2. * midpoint_velocity_direction[d] / implicit_gamma_inverse;
        norm += endpoint_momentum[d] * endpoint_momentum[d];
        dot += endpoint_momentum[d] * du[d];
    }
    double const inverse_gamma = 1. / std::sqrt(1. + norm * inverse_c_squared);
    double const factor = inverse_c_squared * inverse_gamma * inverse_gamma * inverse_gamma * dot;
    for (int d = 0; d < 3; ++d)
    {
        endpoint_velocity[d] = endpoint_momentum[d] * inverse_gamma;
        position_direction[d] = dt * midpoint_velocity_direction[d];
        velocity_direction[d] = inverse_gamma * du[d] - endpoint_momentum[d] * factor;
        if (!std::isfinite(endpoint_velocity[d]) || !std::isfinite(position_direction[d]) ||
            !std::isfinite(velocity_direction[d]))
        {
            return false;
        }
    }
    return true;
}

/** Cartesian endpoint value/direction to cylindrical grid-coordinate map.
 * Basis derivatives are required even at fixed azimuthal Fourier mode m=0.
 * The origin is deliberately excluded; native endpoints require radius > 0.
 */
AMREX_GPU_HOST_DEVICE inline bool CylindricalEndpointCurrentDirection (
    amrex::GpuArray<double, 3> const &endpoint_position,
    amrex::GpuArray<double, 3> const &endpoint_velocity,
    amrex::GpuArray<double, 3> const &cartesian_position_direction,
    amrex::GpuArray<double, 3> const &cartesian_velocity_direction, double inverse_dr,
    double inverse_dz, amrex::GpuArray<double, 3> &velocity,
    amrex::GpuArray<double, 3> &position_direction, amrex::GpuArray<double, 3> &velocity_direction,
    double &transverse_velocity, double &transverse_direction) noexcept
{
    velocity = {};
    position_direction = {};
    velocity_direction = {};
    transverse_velocity = 0.;
    transverse_direction = 0.;
    double const r = std::sqrt(endpoint_position[0] * endpoint_position[0] +
                               endpoint_position[1] * endpoint_position[1]);
    if (!(r > 0.) || !std::isfinite(r) || !(inverse_dr > 0.) || !(inverse_dz > 0.) ||
        !std::isfinite(inverse_dr) || !std::isfinite(inverse_dz))
    {
        return false;
    }
    double const cs = endpoint_position[0] / r, sn = endpoint_position[1] / r;
    double const vr = cs * endpoint_velocity[0] + sn * endpoint_velocity[1];
    transverse_velocity = -sn * endpoint_velocity[0] + cs * endpoint_velocity[1];
    double const dangle =
        (-sn * cartesian_position_direction[0] + cs * cartesian_position_direction[1]) / r;
    velocity = {vr * inverse_dr, 0., endpoint_velocity[2] * inverse_dz};
    position_direction = {
        (cs * cartesian_position_direction[0] + sn * cartesian_position_direction[1]) * inverse_dr,
        0., cartesian_position_direction[2] * inverse_dz};
    velocity_direction = {(cs * cartesian_velocity_direction[0] +
                           sn * cartesian_velocity_direction[1] + transverse_velocity * dangle) *
                              inverse_dr,
                          0., cartesian_velocity_direction[2] * inverse_dz};
    transverse_direction =
        -sn * cartesian_velocity_direction[0] + cs * cartesian_velocity_direction[1] - vr * dangle;
    for (int d = 0; d < 3; ++d)
    {
        if (!std::isfinite(velocity[d]) || !std::isfinite(position_direction[d]) ||
            !std::isfinite(velocity_direction[d]))
        {
            return false;
        }
    }
    return std::isfinite(transverse_velocity) && std::isfinite(transverse_direction);
}
} // namespace warpx::particles
#endif
