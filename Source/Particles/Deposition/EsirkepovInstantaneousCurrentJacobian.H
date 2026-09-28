/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_INSTANTANEOUS_CURRENT_JACOBIAN_H_
#define WARPX_ESIRKEPOV_INSTANTANEOUS_CURRENT_JACOBIAN_H_

#include "EsirkepovCurrentRate.H"

namespace warpx::particles
{

/** Directional derivative of the instantaneous native Esirkepov map.
 * Position/direction and velocity/direction are independent grid-coordinate
 * quantities. Dim=2 uses coordinates 0,2 and physical transverse velocity.
 * The returned value uses the native current-rate implementation exactly.
 * This is a derivative, not a finite increment: no product of directions.
 * Caller supplies charge/weight, RZ metrics, boundary images and summation.
 */
template <int Order, int Dim>
AMREX_GPU_HOST_DEVICE bool EsirkepovInstantaneousCurrentJacobian (
    amrex::GpuArray<double, 3> const &position, amrex::GpuArray<double, 3> const &velocity,
    amrex::GpuArray<double, 3> const &position_direction,
    amrex::GpuArray<double, 3> const &velocity_direction, amrex::GpuArray<int, 3> const &node,
    double transverse_velocity, double transverse_direction, amrex::GpuArray<double, 3> &current,
    amrex::GpuArray<double, 3> &derivative) noexcept
{
    derivative = {};
    amrex::GpuArray<double, 3> unused{};
    if (!EsirkepovCurrentRate<Order, Dim>(position, velocity, {}, node, transverse_velocity, 0.,
                                          current, unused) ||
        !std::isfinite(transverse_velocity) || !std::isfinite(transverse_direction))
    {
        current = {};
        return false;
    }
    double shape[3]{1., 1., 1.}, dshape[3]{}, face[3]{}, dface[3]{};
    for (int d = 0; d < 3; ++d)
    {
        if constexpr (Dim == 2)
        {
            if (d == 1)
            {
                continue;
            }
        }
        if (!std::isfinite(position_direction[d]) || !std::isfinite(velocity_direction[d]))
        {
            current = {};
            return false;
        }
        int const center = int(position[d] + (Order % 2 == 0 ? .5 : 0.));
        int const slot = node[d] - (center - Order / 2);
        double const local = position[d] - center;
        auto const selected = instantaneous_esirkepov::Shape<Order>(slot, {local, 1., 0.});
        shape[d] = selected.value;
        dshape[d] = selected.first * position_direction[d];
        // The native prefix derivative is exactly the order-(p-1) face
        // spline. Evaluate that polynomial directly: summing nearly cancelling
        // nodal derivatives loses significant digits near a support edge.
        if constexpr (Order == 1)
        {
            face[d] = slot == 0 ? 1. : 0.;
        }
        else
        {
            auto const normal = instantaneous_esirkepov::Shape<Order - 1>(
                slot, {local + (Order % 2 == 0 ? .5 : -.5), 1., 0.});
            face[d] = normal.value;
            dface[d] = normal.first * position_direction[d];
        }
    }
    for (int d = 0; d < 3; ++d)
    {
        if constexpr (Dim == 2)
        {
            if (d == 1)
            {
                derivative[d] = transverse_direction * shape[0] * shape[2] +
                                transverse_velocity * (dshape[0] * shape[2] + shape[0] * dshape[2]);
                continue;
            }
        }
        int const e = (d + 1) % 3, f = (d + 2) % 3;
        derivative[d] =
            (face[d] * velocity_direction[d] + dface[d] * velocity[d]) * shape[e] * shape[f] +
            face[d] * velocity[d] * (dshape[e] * shape[f] + shape[e] * dshape[f]);
    }
    for (int d = 0; d < 3; ++d)
    {
        if (!std::isfinite(current[d]) || !std::isfinite(derivative[d]))
        {
            current = {};
            derivative = {};
            return false;
        }
    }
    return true;
}
} // namespace warpx::particles
#endif
