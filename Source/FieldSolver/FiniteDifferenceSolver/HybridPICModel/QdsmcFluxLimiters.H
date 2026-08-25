/* Copyright 2024-2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_FLUX_LIMITERS_H_
#define WARPX_QDSMC_FLUX_LIMITERS_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

/** \file QdsmcFluxLimiters.H
 *
 * Limiter primitives for the electron-energy transport/conduction
 * operators: the NVD face-value family (QUICK/SMART/upwind) used by the
 * advective sweeps, and the Sharma--Hammett (JCP 227, 2007) limited
 * transverse face gradient used by the anisotropic FD conduction
 * operator's cross-derivative fluxes.
 *
 * This header is deliberately self-contained (AMReX scalar types only,
 * no solver state) so that other solver lineages carrying a
 * chi_par/chi_perp anisotropic electron conduction operator (e.g.
 * implicit MHD or Darwin electron-energy variants) can adopt the same
 * limiters by including it and applying two call-site substitutions in
 * their cross-flux assembly:
 *   - replace the centered transverse derivative entering the cross
 *     flux with sh_face_gradient(...) built from the four one-sided
 *     differences around the face, and
 *   - drop any face-value reconstruction of that cross term (the SH
 *     bound acts on the flux ingredient, not the reconstructed value).
 * The nested minmod construction guarantees the cross flux through a
 * face vanishes when either adjacent node is a local extremum along
 * the transverse direction, so cross transport cannot amplify
 * extrema -- a property face-value (NVD) bounds do not provide.
 */

/** SMART-limited (Gaskell--Lau 1988) face value for advective fluxes,
 *  in the normalized-variable frame of the local upwind triple
 *  (tUU = far upwind, tU = upwind, tD = downwind): QUICK where smooth,
 *  blended to bounded legs near extrema, 1st-order upwind outside
 *  [0, 1] -- monotone by construction. limiter = 0 returns unlimited
 *  QUICK (attribution control, can overshoot), 1 returns 1st-order
 *  upwind. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qdsmc_fd_face_value (amrex::Real const tUU,
                                 amrex::Real const tU,
                                 amrex::Real const tD, int const limiter)
{
    using namespace amrex::literals;
    if (limiter == 1) { return tU; }
    if (limiter == 0) {
        return 0.375_rt*tD + 0.75_rt*tU - 0.125_rt*tUU;   // QUICK
    }
    amrex::Real const den = tD - tUU;
    if (den == 0.0_rt) { return tU; }
    amrex::Real const ph = (tU - tUU) / den;
    amrex::Real phf;
    if (ph <= 0.0_rt || ph >= 1.0_rt) { phf = ph; }
    else if (ph < 1.0_rt/6.0_rt)      { phf = 3.0_rt*ph; }
    else if (ph <= 5.0_rt/6.0_rt)     { phf = 0.375_rt + 0.75_rt*ph; }
    else                              { phf = 1.0_rt; }
    return tUU + phf*den;
}

/** minmod of two slopes: 0 on sign disagreement (an extremum), else
 *  the smaller-magnitude slope. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qdsmc_minmod2 (amrex::Real const a, amrex::Real const b)
{
    using namespace amrex::literals;
    if (a*b <= 0.0_rt) { return 0.0_rt; }
    return (std::abs(a) < std::abs(b)) ? a : b;
}

/** Monotonized-central (MC) slope from the two one-sided differences:
 *  minmod(2 dp, 2 dm, (dp + dm)/2). Equals the centered derivative in
 *  smooth regions (retains 2nd order), 0 at a local extremum. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qdsmc_mc_slope (amrex::Real const dp, amrex::Real const dm)
{
    using namespace amrex::literals;
    if (dp*dm <= 0.0_rt) { return 0.0_rt; }
    amrex::Real const c = 0.5_rt*(dp + dm);
    amrex::Real const two_min =
        2.0_rt*((std::abs(dp) < std::abs(dm)) ? dp : dm);
    return (std::abs(c) < std::abs(two_min)) ? c : two_min;
}

/** Sharma--Hammett limited transverse gradient at a face, from the two
 *  one-sided differences at each adjacent node ((dp0, dm0) at the low
 *  node, (dp1, dm1) at the high node), nested as
 *  minmod(L(dp0, dm0), L(dp1, dm1)) with the inner limiter L = MC
 *  (use_mc = true, 2nd order in smooth regions) or minmod (use_mc =
 *  false, maximally diffusive). The outer minmod zeroes the face's
 *  cross flux whenever either node is a transverse local extremum --
 *  the flux-level extrema bound. A missing one-sided difference (a
 *  clamped shift at a non-periodic wall) must be passed as 0, which
 *  kills the face's cross flux (adiabatic-wall consistent). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qdsmc_sh_face_gradient (amrex::Real const dp0,
                                    amrex::Real const dm0,
                                    amrex::Real const dp1,
                                    amrex::Real const dm1,
                                    bool const use_mc)
{
    amrex::Real const g0 = use_mc ? qdsmc_mc_slope(dp0, dm0)
                                  : qdsmc_minmod2(dp0, dm0);
    amrex::Real const g1 = use_mc ? qdsmc_mc_slope(dp1, dm1)
                                  : qdsmc_minmod2(dp1, dm1);
    return qdsmc_minmod2(g0, g1);
}

#endif // WARPX_QDSMC_FLUX_LIMITERS_H_
