/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_YEELOOPKERNEL_H_
#define WARPX_CIRCUIT_YEELOOPKERNEL_H_

#include <ablastr/coils/RingKernel.H>

#include <AMReX_GpuQualifiers.H>

#include <cmath>

/** \file YeeLoopKernel.H
 *
 * The DISCRETE (mesh-sampling) convention of the loop vector potential:
 * what an RZ Yee mesh realizes for a circular filament. This is the
 * convention layer deliberately separated from the continuum kernels in
 * ablastr/coils/RingKernel.H -- the argument clipping and the quarter-cell
 * filament offset below belong HERE and only here, and quantities computed
 * in the two conventions (e.g. inductances) must never be mixed in one
 * matrix: the mesh-regularized discrete self-inductance is resolution
 * dependent (~ln(1/dr)) by construction and is the value a circuit port
 * coupled to fields on this mesh must use.
 */
namespace warpx::circuit
{
    /** Yee-convention filament placement: (+dr/4, +dz/4) off the nominal
     * coil position. A filament exactly on a mesh point makes the loop-A
     * elliptic integral singular there, and A is evaluated at every Yee
     * staggering (node and cell positions in both directions); the
     * quarter-cell offset is maximally distant from all of them. EVERY
     * discrete-convention routine places the filament here. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void QuarterOffset (const double coil_r, const double coil_z,
                        const double dr, const double dz,
                        double& r_off, double& z_off)
    {
        r_off = coil_r + 0.25 * dr;
        z_off = coil_z + 0.25 * dz;
    }

    /** Loop A_theta per unit current of the (already offset) filament at
     * (r_off, z_off), sampled at (r, z) in the Yee convention: the Legendre
     * parameter m = k^2 is clipped to [1e-15, 1-1e-12] (the lower clip
     * regularizes the 0/0 of f(m)/m on the axis, the upper clip keeps A
     * finite -- but large -- at points adjacent to the filament), and the
     * kernel is evaluated in the f(m)/m form
     *     psi = (mu0/4pi) * 4 r r_off / sqrt(d2) * f(m)/m,
     *     A_theta = psi / r  (exactly 0 on the axis).
     * The clip IS part of the convention, intentionally absent from the
     * continuum kernel. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double YeeLoopATheta (const double r, const double z,
                          const double r_off, const double z_off)
    {
        if (r <= 0.0) { return 0.0; }
        const double d2 = (r + r_off) * (r + r_off)
                          + (z - z_off) * (z - z_off);
        double m = 4.0 * r * r_off / d2;
        if (m < 1.0e-15) { m = 1.0e-15; }
        if (m > 1.0 - 1.0e-12) { m = 1.0 - 1.0e-12; }
        const double psi = ablastr::coils::mu0_ring
            / (4.0 * ablastr::coils::pi_ring)
            * 4.0 * r * r_off / std::sqrt(d2)
            * ablastr::coils::KernelBracket(m) / m;
        return psi / r;
    }

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_YEELOOPKERNEL_H_
