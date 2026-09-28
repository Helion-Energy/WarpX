/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_BORIS_LOCAL_RESPONSE_H_
#define WARPX_IMPLICIT_BORIS_LOCAL_RESPONSE_H_

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

namespace warpx::particles
{
/** Local derivative of the native full Boris momentum update and implicit
 * midpoint position velocity. Gather coordinates/weights are held. The first
 * three columns differentiate Cartesian E [V/m], the last three B [T].
 * Midpoint proper momentum and velocity must stay distinct: the velocity uses
 * 2/(gamma_n+gamma_endpoint), whose derivative is included here.
 * This does not include the spatial gather/Picard fixed-point derivative.
 */
struct ImplicitBorisLocalResponse
{
    amrex::GpuArray<amrex::GpuArray<double, 3>, 6> midpoint_momentum{};
    amrex::GpuArray<amrex::GpuArray<double, 3>, 6> midpoint_velocity{};
    amrex::GpuArray<amrex::GpuArray<double, 3>, 6> endpoint_velocity{};
};

AMREX_GPU_HOST_DEVICE inline bool EvaluateImplicitBorisLocalResponse (
    amrex::GpuArray<double, 3> const &old_u, amrex::GpuArray<double, 3> const &midpoint_u,
    amrex::GpuArray<double, 3> const &electric, amrex::GpuArray<double, 3> const &magnetic,
    double charge, double mass, double dt, double inverse_c_squared,
    ImplicitBorisLocalResponse &result) noexcept
{
    result = {};
    if (!(mass > 0.) || !(dt > 0.) || !(inverse_c_squared >= 0.) || !std::isfinite(charge) ||
        !std::isfinite(mass) || !std::isfinite(dt) || !std::isfinite(inverse_c_squared))
    {
        return false;
    }
    double const beta = .5 * charge * dt / mass;
    if (!std::isfinite(beta))
    {
        return false;
    }
    amrex::GpuArray<double, 3> w{}, t{}, s{}, p{}, endpoint{};
    double w2 = 0., old2 = 0., end2 = 0.;
    for (int d = 0; d < 3; ++d)
    {
        if (!std::isfinite(old_u[d]) || !std::isfinite(midpoint_u[d]) ||
            !std::isfinite(electric[d]) || !std::isfinite(magnetic[d]))
        {
            return false;
        }
        w[d] = old_u[d] + beta * electric[d];
        endpoint[d] = 2. * midpoint_u[d] - old_u[d];
        w2 += w[d] * w[d];
        old2 += old_u[d] * old_u[d];
        end2 += endpoint[d] * endpoint[d];
    }
    if (!std::isfinite(w2) || !std::isfinite(old2) || !std::isfinite(end2))
    {
        return false;
    }
    double const gamma_minus_inverse = 1. / std::sqrt(1. + w2 * inverse_c_squared);
    double const gamma_old = std::sqrt(1. + old2 * inverse_c_squared);
    double const gamma_endpoint = std::sqrt(1. + end2 * inverse_c_squared);
    if (!std::isfinite(gamma_old) || !std::isfinite(gamma_endpoint) || !(gamma_minus_inverse > 0.))
    {
        return false;
    }
    double const g = 2. / (gamma_old + gamma_endpoint);
    if (!(g > 0.))
    {
        return false;
    }
    double t2 = 0.;
    for (int d = 0; d < 3; ++d)
    {
        t[d] = beta * gamma_minus_inverse * magnetic[d];
        t2 += t[d] * t[d];
    }
    double const denominator = 1. + t2;
    if (!std::isfinite(denominator * denominator))
    {
        return false;
    }
    for (int d = 0; d < 3; ++d)
    {
        int const a = (d + 1) % 3, b = (d + 2) % 3;
        s[d] = 2. * t[d] / denominator;
        p[d] = w[d] + w[a] * t[b] - w[b] * t[a];
        if (!std::isfinite(p[d]))
        {
            return false;
        }
    }
    for (int column = 0; column < 6; ++column)
    {
        amrex::GpuArray<double, 3> dw{}, dtangent{}, ds{}, dp{}, du{};
        if (column < 3)
        {
            dw[column] = beta;
        }
        double wd = 0.;
        for (int d = 0; d < 3; ++d)
        {
            wd += w[d] * dw[d];
        }
        double const da = -inverse_c_squared * gamma_minus_inverse * gamma_minus_inverse *
                          gamma_minus_inverse * wd;
        double td = 0.;
        for (int d = 0; d < 3; ++d)
        {
            dtangent[d] = beta * (da * magnetic[d] + (column == d + 3 ? gamma_minus_inverse : 0.));
            td += t[d] * dtangent[d];
        }
        for (int d = 0; d < 3; ++d)
        {
            int const a = (d + 1) % 3, b = (d + 2) % 3;
            ds[d] = 2. * dtangent[d] / denominator - 4. * t[d] * td / (denominator * denominator);
            dp[d] = dw[d] + dw[a] * t[b] - dw[b] * t[a] + w[a] * dtangent[b] - w[b] * dtangent[a];
        }
        double ud = 0.;
        for (int d = 0; d < 3; ++d)
        {
            int const a = (d + 1) % 3, b = (d + 2) % 3;
            du[d] = dw[d] + dp[a] * s[b] - dp[b] * s[a] + p[a] * ds[b] - p[b] * ds[a] +
                    (column == d ? beta : 0.);
            ud += endpoint[d] * du[d];
        }
        double const dgamma = inverse_c_squared * ud / gamma_endpoint;
        double const dg = -.5 * g * g * dgamma;
        for (int d = 0; d < 3; ++d)
        {
            result.midpoint_momentum[column][d] = .5 * du[d];
            result.midpoint_velocity[column][d] = .5 * g * du[d] + midpoint_u[d] * dg;
            result.endpoint_velocity[column][d] =
                du[d] / gamma_endpoint - endpoint[d] * dgamma / (gamma_endpoint * gamma_endpoint);
            if (!std::isfinite(result.midpoint_momentum[column][d]) ||
                !std::isfinite(result.midpoint_velocity[column][d]) ||
                !std::isfinite(result.endpoint_velocity[column][d]))
            {
                result = {};
                return false;
            }
        }
    }
    return true;
}
} // namespace warpx::particles
#endif
