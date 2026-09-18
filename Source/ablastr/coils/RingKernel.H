/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_COILS_RINGKERNEL_H_
#define ABLASTR_COILS_RINGKERNEL_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

/** \file RingKernel.H
 *
 * Continuum (free-space) field kernels of an axisymmetric circular current
 * filament: the poloidal-flux ring Green's function and the loop fields
 * A_theta, B_r, B_z (von Hagenow & Lackner, Comput. Phys. Commun. 1975;
 * Jardin, Computational Methods in Plasma Physics, 2010, Sec. 4.6.3).
 *
 * CONVENTIONS -- read before using:
 *
 * - The complete elliptic integrals here take the Legendre PARAMETER
 *   m = k^2 (the SciPy convention). The AMReX expression parser's
 *   comp_ellint_1/2 take the MODULUS k = sqrt(m) instead: any expression
 *   cross-check must pass sqrt(m) there. This header is the single place
 *   that convention lives on the C++ side.
 *
 * - mu0 here is the classical exact value 4e-7*pi. WarpX's PhysConst::mu0
 *   is adjusted to the SI-2019/CODATA convention so that mu0*eps0*c^2 is
 *   exact; the two differ by ~1.3e-10 relative. The coil/inductance
 *   conventions (and their pinned reference values) are defined with the
 *   classical value; do not mix the two inside one calculation.
 *
 * - These are the CONTINUUM kernels: no argument clipping and no filament
 *   offsets. Evaluation exactly on the filament diverges (IEEE inf), by
 *   design. Discrete (mesh) conventions -- quarter-cell filament offsets,
 *   parameter clipping, discrete curls, staircase disk sums -- belong to
 *   the mesh-sampling layer, never here, and continuum and discrete
 *   quantities (e.g. inductances) must never be mixed in one matrix.
 *
 * All math is in double precision regardless of the WarpX precision.
 */
namespace ablastr::coils
{
    /** pi to double precision (kept local: this header is usable device-side
     * and independent of the WarpX constants headers). */
    constexpr double pi_ring = 3.14159265358979323846;

    /** Classical vacuum permeability 4e-7*pi [H/m] (see the file header for
     * the deliberate ~1.3e-10 relative difference vs PhysConst::mu0). */
    constexpr double mu0_ring = 4.0e-7 * pi_ring;

    /** Complete elliptic integrals K(m) and E(m) of the Legendre PARAMETER
     * m = k^2, via the arithmetic-geometric mean (Abramowitz & Stegun 17.6).
     * Requires 0 <= m < 1; K(1) = inf is produced by the AGM naturally
     * (a -> 0 never happens for m < 1). Converges quadratically: the
     * iteration cap is never reached for m <= 1 - 1e-12.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void CompleteEllipticIntegrals (const double m, double& K, double& E)
    {
        if (m >= 1.0) {
            // The logarithmic singularity: K(1) = inf, E(1) = 1. Reached
            // only by evaluation exactly on a filament.
            K = std::numeric_limits<double>::infinity();
            E = 1.0;
            return;
        }
        double a = 1.0;
        double b = std::sqrt(1.0 - m);
        double c = std::sqrt(m);
        double sum = 0.5 * m;   // n = 0 term: 2^{-1} c_0^2 with c_0 = k
        double pow2 = 1.0;
        for (int n = 0; n < 64; ++n) {
            if (std::abs(c) < 1.0e-16) { break; }
            const double an = 0.5 * (a + b);
            c = 0.5 * (a - b);
            b = std::sqrt(a * b);
            a = an;
            pow2 *= 2.0;
            sum += 0.5 * pow2 * c * c;   // 2^{n-1} c_n^2
        }
        K = pi_ring / (2.0 * a);
        E = K * (1.0 - sum);
    }

    /** The ring-kernel bracket f(m) = (2-m)*K(m) - 2*E(m), accurate for all
     * m in [0, 1). The direct form cancels catastrophically as m -> 0 (both
     * terms -> pi while f ~ pi*m^2/16), so below m <= 0.1 a power series
     * through m^16 is used (coefficients derived with exact rational
     * arithmetic; the truncation error at the branch point is ~1e-16
     * relative). f diverges logarithmically as m -> 1.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double KernelBracket (const double m)
    {
        if (m <= 0.1) {
            // f(m)/pi = sum_k c_k m^k, k = 2..16 (c_0 = c_1 = 0)
            constexpr double c[17] = {
                0.0,                                        // m^0
                0.0,                                        // m^1
                1.0 / 16.0,                                 // m^2
                3.0 / 64.0,                                 // m^3
                75.0 / 2048.0,                              // m^4
                245.0 / 8192.0,                             // m^5
                6615.0 / 262144.0,                          // m^6
                22869.0 / 1048576.0,                        // m^7
                1288287.0 / 67108864.0,                     // m^8
                4601025.0 / 268435456.0,                    // m^9
                265939245.0 / 17179869184.0,                // m^10
                969738055.0 / 68719476736.0,                // m^11
                28510298817.0 / 2199023255552.0,            // m^12
                105468168351.0 / 8796093022208.0,           // m^13
                3138933581875.0 / 281474976710656.0,        // m^14
                11734782467625.0 / 1125899906842624.0,      // m^15
                1409850293610375.0 / 144115188075855872.0,  // m^16
            };
            double poly = c[16];
            for (int k = 15; k >= 0; --k) { poly = poly * m + c[k]; }
            return pi_ring * poly;
        }
        double K, E;
        CompleteEllipticIntegrals(m, K, E);
        return (2.0 - m) * K - 2.0 * E;
    }

    /** Poloidal flux per radian per unit ring current: G0 such that
     * psi(rb, zb) = r*A_theta of a circular filament at (rs, zs) carrying
     * 1 A is
     *     G0 = (mu0/4pi) sqrt((zb-zs)^2 + (rb+rs)^2) f(m),
     *     m  = 4 rb rs / ((zb-zs)^2 + (rb+rs)^2).
     * The full-loop flux is 2*pi*G0*I and the coaxial (Maxwell) mutual
     * inductance of two rings is 2*pi*G0. Returns 0 for rb <= 0, rs <= 0 or
     * coincident degenerate geometry; +inf exactly on the filament.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double RingPsiPerAmp (const double rb, const double zb,
                          const double rs, const double zs)
    {
        const double d2 = (zb - zs) * (zb - zs) + (rb + rs) * (rb + rs);
        if (d2 <= 0.0 || rs <= 0.0 || rb <= 0.0) { return 0.0; }
        double m = 4.0 * rb * rs / d2;
        if (m > 1.0) { m = 1.0; }   // FP-roundoff clamp only, not a policy clip
        return mu0_ring / (4.0 * pi_ring) * std::sqrt(d2) * KernelBracket(m);
    }

    /** A_theta per unit ring current of a circular filament of radius Rc at
     * axial position zc, evaluated at (r, z): psi/r with the exact on-axis
     * limit A_theta(0, z) = 0. Diverges on the filament. Requires Rc > 0.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double LoopATheta (const double r, const double z,
                       const double Rc, const double zc)
    {
        if (r <= 0.0) { return 0.0; }
        return RingPsiPerAmp(r, z, Rc, zc) / r;
    }

    /** B_r per unit ring current of a circular filament (radius Rc at zc)
     * at (r, z):
     *     B_r = mu0/(2 pi) * (z-zc)/(r sqrt(d2)) * [ -K(m)
     *           + (Rc^2 + r^2 + (z-zc)^2)/a2 * E(m) ],
     *     a2 = (r-Rc)^2 + (z-zc)^2,  d2 = (r+Rc)^2 + (z-zc)^2,
     * with the exact on-axis limit B_r(0, z) = 0. Diverges (inf) exactly on
     * the filament. Requires r >= 0 and Rc > 0.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double LoopBr (const double r, const double z,
                   const double Rc, const double zc)
    {
        if (r <= 0.0) { return 0.0; }
        const double dz = z - zc;
        const double a2 = (r - Rc) * (r - Rc) + dz * dz;
        const double d2 = (r + Rc) * (r + Rc) + dz * dz;
        double m = 4.0 * r * Rc / d2;
        if (m > 1.0) { m = 1.0; }
        double K, E;
        CompleteEllipticIntegrals(m, K, E);
        return mu0_ring / (2.0 * pi_ring) * dz / (r * std::sqrt(d2))
               * (-K + (Rc * Rc + r * r + dz * dz) / a2 * E);
    }

    /** B_z per unit ring current of a circular filament (radius Rc at zc)
     * at (r, z):
     *     B_z = mu0/(2 pi) / sqrt(d2) * [ K(m)
     *           + (Rc^2 - r^2 - (z-zc)^2)/a2 * E(m) ],
     * with the exact on-axis limit
     *     B_z(0, z) = mu0 Rc^2 / (2 (Rc^2 + (z-zc)^2)^{3/2}).
     * Diverges (inf) exactly on the filament. Requires r >= 0 and Rc > 0.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double LoopBz (const double r, const double z,
                   const double Rc, const double zc)
    {
        const double dz = z - zc;
        if (r <= 0.0) {
            const double s2 = Rc * Rc + dz * dz;
            return mu0_ring * Rc * Rc / (2.0 * s2 * std::sqrt(s2));
        }
        const double a2 = (r - Rc) * (r - Rc) + dz * dz;
        const double d2 = (r + Rc) * (r + Rc) + dz * dz;
        double m = 4.0 * r * Rc / d2;
        if (m > 1.0) { m = 1.0; }
        double K, E;
        CompleteEllipticIntegrals(m, K, E);
        return mu0_ring / (2.0 * pi_ring) / std::sqrt(d2)
               * (K + (Rc * Rc - r * r - dz * dz) / a2 * E);
    }

} // namespace ablastr::coils

#endif // ABLASTR_COILS_RINGKERNEL_H_
