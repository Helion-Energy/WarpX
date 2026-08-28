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
    amrex::Real ion_pressure_floor;
    bool include_hall_term;
    bool use_rusanov;
    bool use_hllc;
    bool total_energy_closure;
    // HLLC signal-speed closure: evaluate the acoustic wave-speed
    // estimates (Davis bounds, LLF alpha) from the polytropic p_i(rho)
    // instead of the recovered p_i(E_i), removing the E_i unknown from
    // the HLLC wave structure (total_energy closure only).
    bool barotropic_signal_speeds;
    // Smooth contact-side blending width kappa for the HLLC flux
    // (0 disables): the upwind-side selection at the contact wave is
    // blended over |s_star| ~ kappa (|s_left| + |s_right|) / 2 instead
    // of switching at s_star = 0.
    amrex::Real contact_blend;
    // Implicit time-centering parameter: the floor limiters gate on the
    // theta-extrapolated end-of-step values, matching the Newton
    // admissibility bound (see face_flux).
    amrex::Real theta;
    // ------------------------------------------------------------------
    // HLLD (Miyoshi & Kusano 2005) additions. Appended with in-struct
    // default initializers so the existing positional aggregate
    // initialization (ThetaImplicitMHD.cpp) keeps compiling unchanged and
    // hlld_flux stays inert until the conservative-form recast wires it
    // in.
    // ------------------------------------------------------------------
    // Vacuum permeability; hlld_flux rescales b = B/sqrt(mu0) internally
    // so the MK05 formulas apply verbatim with mu0 = 1. The default 1.0
    // is unit-agnostic; the recast must set PhysConst::mu0.
    amrex::Real mu0 = 1.0;
    // C-infinity smoothing widths (each 0 recovers the hard form):
    // Davis signal bounds: smooth min/max of width
    // w_s = kappa_signal (c_fL + c_fR).
    amrex::Real hlld_kappa_signal = 0.05;
    // Region selection: sign blends m(S) = (1 + S/sqrt(S^2 + w_c^2))/2
    // with w_c = kappa_contact (|S_L| + |S_R|)/2.
    amrex::Real hlld_kappa_contact = 0.05;
    // B_n -> 0 degeneracy: eps_B^2 = kappa_bn^2 mu0 mean(p_T) sets the
    // rotational-layer weight chi = B_n^2/(B_n^2 + eps_B^2) and the
    // smoothed sign(B_n).
    amrex::Real hlld_kappa_bn = 0.05;
    // Star-state denominator guard scale (C^1 smooth-max floor, same
    // idiom as the internal-energy floor in load_cell_state).
    amrex::Real hlld_kappa_denominator = 0.05;
    // Fan closure (implicit_mhd.hlld_fan_closure = barotropic): evaluate
    // the WAVE-FAN structure -- the contact-speed estimate S_M and the
    // star states built from it -- with the polytropic p_i(rho) instead
    // of the recovered p_i(E_i). The physical fluxes (and hence the jump
    // conditions, which telescope over the input states) keep the
    // consistent pressure, so this only re-centers the dissipation
    // structure. Rationale: p_i(E_i) feeding S_M couples the E_i block
    // into the star algebra, whose curvature at rotational/compound
    // structures (B_t through zero at finite B_n) defeats the Newton
    // line search; the Brio-Wu lock bisection localized the mechanism
    // to exactly this path.
    bool hlld_barotropic_fan = false;
    // Fan-consistent star construction for the ion-energy channel
    // (default); false selects the LLF enthalpy fallback.
    bool hlld_ion_energy_star = true;
    // LHLLD all-speed scaling of the star-pressure velocity-difference
    // term (Minoshima & Miyoshi 2021); false keeps the original HLLD
    // fast-speed scaling.
    bool hlld_all_speed = true;
    // CGL (bi-Maxwellian) ion closure: load_cell_state_cgl overwrites
    // ion_pressure with the effective isotropic pressure
    // p_eff = (p_par + 2 p_perp)/3 and hlld_flux adds the U_par/U_perp
    // advection channels. Mutually exclusive with total_energy_closure
    // (MakeFluxParameters keeps the latter false under cgl).
    bool cgl_closure = false;
    // Offset-density halo pedestal (0 = off): the per-step frozen pedestal
    // STATE (see ThetaImplicitMHD::m_halo_pedestal_fraction) -- an
    // f-scaled image of the peak state, one value per fluid block. Each
    // value replaces its block's floor as the anchor of the donor drain
    // gates, so the halo band is dynamically held at the pedestal -- an
    // interior point of the admissible set, far above every positivity
    // bound. The density pedestal additionally anchors the C^1 taper of
    // the reactive work and relaxation sources across the halo band.
    amrex::Real halo_pedestal = 0.0;
    amrex::Real halo_pedestal_electron_energy = 0.0;
    // Internal ion energy pedestal (total_energy closure): the E_i drain
    // gate keys on the KE-subtracted internal part, so the pedestal is
    // an internal-energy value.
    amrex::Real halo_pedestal_ion_internal = 0.0;
    amrex::Real halo_pedestal_ion_parallel = 0.0;
    amrex::Real halo_pedestal_ion_perp = 0.0;
    // Floor-independent corner width of the C^1 smooth-max internal-energy
    // floor in load_cell_state (0 = legacy: width = internal floor). At
    // production floor scales the legacy width is tiny, so a compressed
    // cell riding the corner presents a near-kink to the pressure recovery
    // and its local Jacobian is near-singular; a positive fraction widens
    // the corner to max(internal_floor, fraction * kinetic_energy), keying
    // the width to the LOCAL kinetic-energy scale of the recovery's
    // subtraction. Asymptotes stay exact (floor below, E_i - KE above);
    // the inflation is width/2, confined to near-corner cells.
    amrex::Real pressure_corner_width_fraction = 0.0;
    // Dual-energy ion closure (the reference code's "mixmaster" mix = -1 form):
    // BOTH the conservative ion total energy E_i (evolved exactly as
    // under total_energy_closure, which MakeFluxParameters keeps true)
    // and the auxiliary internal energy U_i are evolved, and the ion
    // pressure every consumer sees is the kinetic-fraction blend
    //     p_i = fk p_i(E_i) + (1 - fk) (gamma_i - 1) U_i,
    //     fk  = 1 - KE/(gamma_i E_i)   (smoothly rectified to [0, 1]),
    // so KE-dominated cells (fk -> 0) read their pressure from U_i and
    // never see the total-minus-kinetic catastrophic cancellation.
    // load_cell_state_dual overwrites CellState::ion_pressure /
    // ion_internal with the blend and refreshes the acoustic speeds
    // (the CGL pattern); hlld_flux/central_flux add the U_i advection
    // channel. Mutually exclusive with cgl_closure.
    bool dual_energy_closure = false;
    // Optional low-internal-energy cutoff (the reference code's mix = -2 variant;
    // 0 = off): cells whose STEP-OLD U_i sits below
    // cutoff * max_domain(U_i_old) get fk = 0 exactly (pure internal
    // pressure in the halo/dust). The mask and the max are per-solve
    // frozen (step-old), so the branch is a constant for JFNK probes.
    amrex::Real dual_energy_internal_cutoff = 0.0;
    amrex::Real dual_energy_internal_old_max = 0.0;
};

// C-infinity rectifier width of the dual-energy kinetic fraction: fk_raw
// = 1 - KE/(gamma_i E_i) <= 1 automatically (KE >= 0, guarded E_i > 0),
// and the lower clamp to 0 is the smooth_max of this width -- exact to
// O(w^2) one width away from fk = 0, so mid-range blends (an imploding
// sheet at fk ~ 0.4) are untouched while the kink at KE = gamma_i E_i
// (exactly the KE-dominated population the closure exists for) is
// smoothed away for the matrix-free Jacobian probes.
constexpr amrex::Real dual_energy_fk_width = 0.05;

struct CellState {
    amrex::Real density;
    amrex::Real safe_density;
    amrex::GpuArray<amrex::Real, 3> momentum;
    amrex::GpuArray<amrex::Real, 3> ion_velocity;
    amrex::Real electron_energy;
    amrex::Real ion_energy;
    amrex::Real ion_internal;
    amrex::Real ion_pressure;
    amrex::Real electron_pressure;
    amrex::Real electron_velocity_normal;
    amrex::Real sound_speed;
    amrex::Real wave_speed;
    // --- HLLD additions. load_cell_state fills hydro-limit values
    // (magnetic = 0, fast_speed = sound_speed); load_cell_state_hlld
    // loads the cell-centered TOTAL B and upgrades the fast speeds to
    // fast-magnetosonic. ---
    amrex::GpuArray<amrex::Real, 3> magnetic;
    // Fast magnetosonic speed along the face normal.
    amrex::Real fast_speed;
    // |u_n| + fast_speed: the LLF alpha scale for the ion-energy channel.
    amrex::Real fast_wave_speed;
    // --- CGL (bi-Maxwellian) additions. load_cell_state_cgl fills the
    // internal energies U_par = p_par/2 and U_perp = p_perp and overwrites
    // ion_pressure with p_eff = (p_par + 2 p_perp)/3; default zero so the
    // isotropic closures are undisturbed. ---
    amrex::Real ion_parallel_energy = 0.0;
    amrex::Real ion_perp_energy = 0.0;
    // --- Dual-energy addition: the EVOLVED auxiliary internal energy
    // U_i (load_cell_state_dual; ion_internal then holds the BLENDED
    // internal energy the pressure consumers see). Default zero so the
    // other closures are undisturbed. ---
    amrex::Real ion_internal_energy = 0.0;
};

struct FaceFlux {
    amrex::Real mass;
    amrex::GpuArray<amrex::Real, 3> momentum;
    amrex::Real electron_energy;
    amrex::Real ion_energy;
    amrex::Real electron_velocity;
    // CGL closure: advective fluxes of U_par and U_perp (populated by
    // hlld_flux only when FluxParameters::cgl_closure is set; the CGL
    // work terms are pointwise RHS sources, not fluxes).
    amrex::Real ion_parallel_energy = 0.0;
    amrex::Real ion_perp_energy = 0.0;
    // Dual-energy closure: advective flux of the auxiliary internal
    // energy U_i (populated by hlld_flux/central_flux only when
    // FluxParameters::dual_energy_closure is set; the blended PdV work
    // and the viscous heating are pointwise RHS sources, not fluxes).
    amrex::Real ion_internal_energy = 0.0;
    // --- HLLD additions (populated by hlld_flux only; default-zero so
    // the existing hllc/rusanov paths are undisturbed). ---
    // Maxwell-stress part of the momentum flux; momentum[] holds the
    // TOTAL, so subtracting this recovers the hydro-only part.
    amrex::GpuArray<amrex::Real, 3> momentum_magnetic = {0.0, 0.0, 0.0};
    // Tangential induction fluxes through the face, i.e. the edge-EMF
    // ingredients for the recast's constrained-transport update. With
    // t1 = (n+1)%3 and t2 = (n+2)%3 (cyclic), induction_t1 is the flux
    // of B_t1: u_n B_t1 - u_t1 B_n = -EMF_t2, and induction_t2 is the
    // flux of B_t2: u_n B_t2 - u_t2 B_n = +EMF_t1, with EMF = -u x B.
    // For a z-face (n = 2) in 1D: F_Bx = u_z Bx - u_x Bz = -EMF_y and
    // F_By = u_z By - u_y Bz = +EMF_x.
    amrex::Real induction_t1 = 0.0;
    amrex::Real induction_t2 = 0.0;
    // Smoothed Davis signal bounds, stored for upwind-constrained-
    // transport (UCT) edge assembly in the recast.
    amrex::Real signal_left = 0.0;
    amrex::Real signal_right = 0.0;
    // Rotational (Alfven) wave speeds of the HLLD fan, S_M -+ c_An*: the
    // scale at which the fan itself dissipates tangential-field jumps.
    // They collapse to S_M as B_n -> 0 (kappa_bn-smoothed), and S_M -> 0
    // for static pressure-balanced states, so corner-EMF dissipation
    // built on them vanishes for held equilibria instead of diffusing
    // them at the fast-speed scale.
    amrex::Real signal_alfven_left = 0.0;
    amrex::Real signal_alfven_right = 0.0;
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
floor_outflow_limiter (const amrex::Real value,
                       const amrex::Real floor) noexcept {
    // Smoothstep: 0 at value <= floor, 1 at value >= 2*floor (C^1, so
    // matrix-free Jacobian probes stay well-behaved). Applied to fluxes
    // and sources that DRAIN a near-floor cell, so the RHS itself
    // respects the positivity floors: the bounded Newton line search can
    // only reject steps, and a residual that demands sub-floor states
    // otherwise locks the solve with no admissible step.
    if (floor <= 0.0) {
        return 1.0;
    }
    const amrex::Real s = std::min(std::max(value / floor - 1.0, 0.0), 1.0);
    return s * s * (3.0 - 2.0 * s);
}

/**
 * \brief Minmod of the two one-sided tangential differences of one cell,
 * for the Braginskii cross-term stencil (Sharma & Hammett 2007): zero
 * whenever the differences disagree in sign (a tangential extremum), the
 * smaller-magnitude difference otherwise. Killing the cross-term flux at
 * tangential extrema restores the discrete maximum principle that the
 * centered corner average violates -- an unlimited cross term at a
 * steep oblique interface demands states below the local stencil
 * minimum, which the admissibility projection then pins (the
 * plasma-vacuum-edge electron_energy freeze of the formation ladder).
 * C^0 like every minmod; the value is continuous through zero.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
minmod_pair (const amrex::Real up_difference,
             const amrex::Real down_difference) noexcept {
    if (up_difference * down_difference <= 0.0) {
        return 0.0;
    }
    return std::abs(up_difference) < std::abs(down_difference)
               ? up_difference
               : down_difference;
}

// C-infinity smoothed min/max of absolute width >= 0 (exact hard min/max
// at width = 0): smin(a, b) = (a + b - sqrt((a - b)^2 + w^2))/2. The
// smoothing only widens the interval: smin <= min and smax >= max.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_min (const amrex::Real a, const amrex::Real b,
            const amrex::Real width) noexcept {
    return 0.5 * (a + b - std::sqrt((a - b) * (a - b) + width * width));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_max (const amrex::Real a, const amrex::Real b,
            const amrex::Real width) noexcept {
    return 0.5 * (a + b + std::sqrt((a - b) * (a - b) + width * width));
}

// C^2 soft upper clip: EXACT identity for value <= cap - width, exact
// asymptote to cap as value -> +infinity, tanh rolloff in between
// (value and its first two derivatives are continuous at the knee:
// tanh(0) = 0, tanh'(0) = 1, tanh''(0) = 0). Unlike smooth_min, states
// deep inside the admissible band are passed through UNCHANGED -- an
// equilibrium far from the cap acquires no spurious offset. width > 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
soft_upper_clip (const amrex::Real value, const amrex::Real cap,
                 const amrex::Real width) noexcept {
    const amrex::Real knee = cap - width;
    if (value <= knee) {
        return value;
    }
    return knee + width * std::tanh((value - knee) / width);
}

// C-infinity smoothed sign: s/sqrt(s^2 + w^2), i.e. +-1 for |s| >> w.
// Returns 0 at s = w = 0 (the hard-sign limit is undefined there).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_sign (const amrex::Real value, const amrex::Real width) noexcept {
    const amrex::Real magnitude =
        std::sqrt(value * value + width * width);
    return magnitude > 0.0 ? value / magnitude : 0.0;
}

// C-infinity smoothed |s| = s * smooth_sign(s) = s^2/sqrt(s^2 + w^2).
// Exactly 0 at s = 0 for any width: a wave sitting on the interface
// contributes no dissipation, which preserves exact static contacts.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_abs (const amrex::Real value, const amrex::Real width) noexcept {
    return value * smooth_sign(value, width);
}

// Smooth max(value, floor) for floor >= 0 -- the same C^1 smooth-max
// idiom as the internal-energy floor in load_cell_state (C-infinity for
// floor > 0; exact asymptotes, corner inflation floor/2 at value =
// floor). At floor = 0 it reduces to the exact identity for value > 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_positive_floor (const amrex::Real value,
                       const amrex::Real floor) noexcept {
    const amrex::Real excess = value - floor;
    return floor +
           0.5 * (excess + std::sqrt(excess * excess + floor * floor));
}

// Rectifier width of the floor-consistency relaxation source, as a
// fraction of the cell's theta-stage admissibility bound. Together with
// the supply-rate cap 1/(theta dt) it fixes the source's per-solve
// supply capacity at width_fraction/2 of the bound (see
// floor_consistency_deficit below), and through the exact-zero gate it
// fixes the guarantee that any state at or above (1 + 2 f) times its
// bound feels IDENTICALLY nothing -- at theta = 1/2 that is every cell
// at or above 1.5x its cell floor, so healthy decks are bit-identical
// with the source armed.
// Default for implicit_mhd.floor_consistency_width_fraction: the
// runtime value is a per-solve constant passed into
// floor_consistency_deficit; raising it raises the bound-riding supply
// CAPACITY (rate_eff * w/2) at the cost of a wider engagement band
// (the exact-zero guarantee moves to (1 + 2 f) x bound).
constexpr amrex::Real floor_consistency_width_fraction = 0.1;

/**
 * \brief Rectified deficit of the floor-consistency relaxation source
 * (implicit_mhd.floor_consistency_rate): the one-sided measure of how
 * far a bounded fluid block's theta-stage value sits BELOW its
 * admissibility bound, smoothly engaged so the residual source
 * S = rate_eff * floor_consistency_deficit(...) supplies the equations'
 * demanded sub-bound drain at bound-riding cells.
 *
 * The bound is the SAME theta-stage admissibility image the Newton
 * projection uses (ProjectAndLimitSolverStep):
 *     bound = (1 - theta) u_old + theta cell_floor,
 * with cell_floor the absolute floor extended by the one-way
 * temperature-coefficient ratchet (engaged only where the step-old
 * value already satisfied it) -- both per-solve constants of the
 * frozen step-old state.
 *
 * Construction (w = width_fraction * bound, x = bound - u):
 *     deficit = smooth_max(x, 0, w) * gate(x),
 *     gate    = floor_outflow_limiter(x + 3 w, w)
 * (1 for x >= -w, exactly 0 for x <= -2 w, C^1 between).
 *
 * Guarantees, stated once here for every caller:
 *  (a) ONE-SIDED: deficit >= 0 always, so the source only SUPPLIES the
 *      block (never drains). The gate dies before smooth_max's odd part
 *      could matter, and smooth_max(x, 0, w) >= 0 for all x.
 *  (b) NEWTON-LINEAR / JFNK-EXACT: the bound, width, and rate are
 *      per-solve constants (frozen-coefficient idiom), so the source
 *      depends on the Newton iterate only through u itself; it is
 *      exactly linear with slope -rate_eff throughout the engaged
 *      regime x >= w and a C^1 (C-infinity smooth-max times C^1 gate)
 *      diagonal blend elsewhere -- the matrix-free probes differentiate
 *      it exactly, with no hidden state dependence.
 *  (c) EXACT ZERO ABOVE THE BOUND: identically zero for u >= bound + 2w
 *      (the gate is exactly 0 there), i.e. for every cell at or above
 *      (1 + 2 * width_fraction) * bound. This EXCEEDS the documented
 *      sub-1e-12 tail requirement: healthy cells (>= 1.5x their cell
 *      floor at theta = 1/2) and healthy runs are bit-identical, not
 *      merely unchanged to round-off. Within (bound, bound + 2w) the
 *      engagement is bounded by w/2 -- the smooth on-bound handoff.
 *  (d) NO OVERSHOOT (with the caller's rate cap
 *      rate_eff = min(rate, 1/(theta dt))): deficit <= max(x, 0) + w/2
 *      (smooth_max widens by at most w/2), so the per-solve supplied
 *      increment theta dt * rate_eff * deficit <= max(bound - u, 0)
 *      + w/2 -- at most the deficit scale plus half a width. A
 *      sub-bound cell cannot be pushed above bound + w/2 in a single
 *      application, and the user's rate is an exact per-second cap on
 *      the supply rate below the 1/(theta dt) ceiling.
 *
 * The supply capacity at a bound-riding cell is rate_eff * w/2 per
 * second (the rectifier's value at x = 0 is w/2): a persistent sink
 * demand below that capacity finds a consistent equilibrium a fraction
 * of a width ABOVE the bound where supply balances sink -- the cell
 * rides the bound smoothly, unpinned, and the Newton residual closes
 * instead of wedging on the projection.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
floor_consistency_deficit (const amrex::Real value,
                           const amrex::Real old_value,
                           const amrex::Real old_density,
                           const amrex::Real floor,
                           const amrex::Real temperature_coefficient,
                           const amrex::Real theta,
                           const amrex::Real width_fraction) noexcept {
    // The projection's cell bound, term for term (see
    // ProjectAndLimitSolverStep): one-way temperature ratchet on the
    // step-old value, then the theta-stage image of the cell floor.
    const amrex::Real temperature_bound =
        temperature_coefficient * old_density;
    const amrex::Real cell_floor = old_value >= temperature_bound
                                       ? std::max(floor, temperature_bound)
                                       : floor;
    const amrex::Real bound = (1.0 - theta) * old_value + theta * cell_floor;
    if (bound <= 0.0) {
        // A non-positive bound never binds a non-negative block.
        return 0.0;
    }
    const amrex::Real width = width_fraction * bound;
    const amrex::Real deficit = bound - value;
    if (deficit <= -2.0 * width) {
        // Exact-zero fast path == the gate's own zero (guarantee (c)).
        return 0.0;
    }
    const amrex::Real gate =
        floor_outflow_limiter(deficit + 3.0 * width, width);
    return gate * smooth_max(deficit, 0.0, width);
}

/**
 * \brief Density-keyed vacuum resistivity of the FIELD advance:
 * a smooth, uncapped resistivity floor
 *
 *     eta_field = smooth-max(eta_user, eta_vac),
 *     eta_vac   = vacuum_scale * (rho_ref / rho)^2,
 *
 * with vacuum_scale = mu0 * D_vac (D_vac the magnetic diffusivity the
 * boost reaches AT the Ohm reference density rho_ref, in m^2/s) and
 * rho_ref the Ohm's-law division guard (hybrid n_floor equivalent) --
 * the same density at which Ohm's law stops trusting the fluid -- or,
 * under implicit_mhd.vacuum_reference_peak_fraction, the per-step
 * frozen dynamic reference max(guard, fraction * rho_peak) (the reference code's
 * en0; see ThetaImplicitMHD::VacuumReferenceMassDensity). The same
 * kernel also composes the conduction halo boost (chi_perp with
 * D_boost (rho_ref/rho)^2, vacuum_scale = D_boost without the mu0).
 *
 * First-principles motivation: below the Ohm reference the "fluid" is a
 * numerical halo/pedestal, not plasma; the physically correct field
 * behavior there is vacuum-like diffusion of the response currents on a
 * timescale short compared to the plasma dynamics. Keying eta to
 * (rho_ref/rho)^2 makes the magnetic diffusivity grow quadratically as
 * the density falls below rho_ref and vanish quadratically above it, so
 * the bulk plasma keeps the user resistivity while the halo relaxes
 * toward curl B = 0. Measured on the FRC free-boundary hold: a flat
 * halo resistivity of this magnitude relaxed the halo residual two
 * orders of magnitude in a single backward-Euler step, isolating the
 * bound-resident population as the only remaining blocker.
 *
 * Smoothness: the density division is guarded with the strictly
 * positive fluid positivity floor through smooth_positive_floor -- a
 * pure div-by-zero guard, orders of magnitude below the Ohm reference,
 * so the boost stays UNCAPPED over the whole physically reachable
 * density range (in particular across a halo pedestal band sitting
 * below the Ohm reference) while arbitrary Jacobian probe states remain
 * finite and C-infinity. The smooth max is the quadrature combination
 * sqrt(eta_user^2 + eta_vac^2): exact in both dominance limits (in
 * particular eta_field = eta_vac exactly where the user resistivity
 * vanishes, which the calibrated vacuum-diffusion test relies on), at
 * most a factor sqrt(2) above the hard max at the crossover, and
 * C-infinity wherever eta_vac > 0.
 *
 * FIELD advance only: Joule heating keeps the un-boosted user eta. In a
 * true vacuum there is no plasma to heat; depositing eta_vac |J|^2 into
 * the pedestal would pump unphysical energy into numerical mass.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
vacuum_keyed_resistivity (const amrex::Real user_resistivity,
                          const amrex::Real density,
                          const amrex::Real reference_density,
                          const amrex::Real division_guard,
                          const amrex::Real vacuum_scale) noexcept {
    if (vacuum_scale <= 0.0) {
        return user_resistivity;
    }
    const amrex::Real guarded_density =
        smooth_positive_floor(density, division_guard);
    const amrex::Real ratio = reference_density / guarded_density;
    const amrex::Real vacuum_resistivity = vacuum_scale * ratio * ratio;
    return std::sqrt(user_resistivity * user_resistivity +
                     vacuum_resistivity * vacuum_resistivity);
}

/**
 * \brief Frozen-coefficient Spitzer electron-ion temperature
 * equilibration rate [1/s], the reference code's verbatim eq_brate coefficient
 * (ntb.f90 658-700, the f_n = 0 electron-ion branch, z_eff = 1):
 *
 *     nu = 4.75e-15 lnLambda (m_p/m_i) n_i / Te[eV]^{3/2},
 *
 * with the NRL-style Coulomb-logarithm branches exactly as the reference code
 * writes them (their +3 ln 10 converts NRL's cm^-3 to the m^-3 density
 * used here; the branch point is the reference code's e^2 = 7.389 eV, not NRL's
 * 10 Z^2):
 *
 *     Te <= e^2 : lnLambda = 23 + 3 ln 10 - ln(n)/2 + 3 ln(Te)/2
 *     Te >  e^2 : lnLambda = 24 + 3 ln 10 - ln(n)/2 + ln(Te)
 *
 * The caller passes number_density and Te precomputed from the
 * STEP-OLD state (floored like every frozen coefficient), so the rate
 * is a per-solve constant: the pair term nu (p_i - p_e) it multiplies
 * stays Newton-linear in the live energies (the
 * conduction_coefficient_state = step_old idiom). The reference code evaluates the
 * same coefficient at its predictor/corrector iterate and applies a
 * backward-Euler pair update -- identical to first order in dt.
 *
 * Deviations from the Fortran, both documented defensive-only:
 * lnLambda is clamped at zero (the reference code has no clamp; a negative value
 * -- reachable only at extreme density/low Te -- would anti-relax),
 * and the rate is capped at rate_cap, the theta-scheme monotonicity
 * bound 1/((gamma_e + gamma_i - 2)(1 - theta) dt) precomputed by the
 * host (the reference code's backward Euler is unconditionally monotone and needs
 * no cap; at the cap the theta update reaches equilibrium in exactly
 * one step instead of overshooting). The reference code's en^2/enm density factor
 * is its P/C time-centering artifact, = n_i to O(dt): the single
 * step-old n_i is used here.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
electron_ion_equilibration_rate (const amrex::Real number_density,
                                 const amrex::Real electron_temperature_ev,
                                 const amrex::Real rate_coefficient,
                                 const amrex::Real rate_cap) noexcept {
    constexpr amrex::Real three_ln_ten = 6.907755278982137;
    constexpr amrex::Real branch_te_ev = 7.389056098930650; // e^2
    const amrex::Real log_n = std::log(number_density);
    const amrex::Real log_te = std::log(electron_temperature_ev);
    const amrex::Real coulomb_log = std::max(
        electron_temperature_ev <= branch_te_ev
            ? 23.0 + three_ln_ten - 0.5 * log_n + 1.5 * log_te
            : 24.0 + three_ln_ten - 0.5 * log_n + log_te,
        0.0);
    return std::min(rate_coefficient * coulomb_log * number_density /
                        (electron_temperature_ev *
                         std::sqrt(electron_temperature_ev)),
                    rate_cap);
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

// (Re)build the acoustic wave speeds from the CURRENT state.ion_pressure
// and electron pressure, leaving hydro-limit values in the HLLD members
// (fast_speed = sound_speed); finish_magnetized_state upgrades them.
// Shared by load_cell_state and load_cell_state_cgl, which overwrites
// ion_pressure with the CGL effective pressure and must then refresh
// every speed derived from it.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
finish_acoustic_speeds (CellState& state, const int normal_direction,
                        const FluxParameters& parameters) noexcept {
    // With the barotropic signal closure the wave-speed estimates use the
    // polytropic p_i(rho) instead of p_i(E_i): the signal speeds then do
    // not respond to E_i Jacobian probes at all, while the physical fluxes
    // and the contact speed keep the consistent (E_i-recovered) pressure,
    // preserving HLLC flux continuity across s_star = 0.
    const amrex::Real signal_ion_pressure =
        parameters.barotropic_signal_speeds
            ? ion_pressure(state.safe_density, parameters)
            : state.ion_pressure;
    const amrex::Real sound_speed_squared =
        std::max((parameters.gamma_i * signal_ion_pressure +
                  parameters.gamma_e * state.electron_pressure) /
                     state.safe_density,
                 0.0);
    state.sound_speed = std::sqrt(sound_speed_squared);
    state.wave_speed = std::max(std::abs(state.ion_velocity[normal_direction]) +
                                    state.sound_speed,
                                std::abs(state.electron_velocity_normal));
    state.fast_speed = state.sound_speed;
    state.fast_wave_speed =
        std::abs(state.ion_velocity[normal_direction]) + state.sound_speed;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state (const amrex::Array4<const amrex::Real>& density,
                 const amrex::Array4<const amrex::Real>& momentum,
                 const amrex::Array4<const amrex::Real>& electron_energy,
                 const amrex::Array4<const amrex::Real>& ion_energy,
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
    if (parameters.total_energy_closure) {
        amrex::Real kinetic_energy = 0.0;
        for (int component = 0; component < 3; ++component) {
            kinetic_energy +=
                state.momentum[component] * state.momentum[component];
        }
        kinetic_energy *= 0.5 / state.safe_density;
        state.ion_energy = ion_energy(i, j, k);
        const amrex::Real internal_floor =
            parameters.ion_pressure_floor / (parameters.gamma_i - 1.0);
        const amrex::Real excess =
            state.ion_energy - kinetic_energy - internal_floor;
        // C^1 smooth max against the internal-energy floor: a hard
        // clamp's kink makes the pressure recovery non-differentiable
        // exactly where floored halo cells sit, and the Newton line
        // search then cannot find a residual decrease. The legacy width
        // is the floor itself, which at production floor scales is tiny
        // against the kinetic energy subtracted just above -- a
        // compressed cell riding the corner then still presents a
        // near-kink and a near-singular local Jacobian; a positive
        // pressure_corner_width_fraction keys the width to the local
        // kinetic-energy scale instead. Asymptotes are exact (floor
        // below, E_i - KE above); the corner inflation is width/2,
        // confined to near-corner cells.
        const amrex::Real corner_width = std::max(
            internal_floor,
            parameters.pressure_corner_width_fraction * kinetic_energy);
        state.ion_internal =
            internal_floor +
            0.5 * (excess + std::sqrt(excess * excess +
                                      corner_width * corner_width));
        state.ion_pressure = (parameters.gamma_i - 1.0) * state.ion_internal;
    } else {
        state.ion_energy = 0.0;
        state.ion_internal = 0.0;
        state.ion_pressure = ion_pressure(state.safe_density, parameters);
    }
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

    finish_acoustic_speeds(state, normal_direction, parameters);
    // Hydro-limit values for the HLLD members; load_cell_state_hlld
    // upgrades them to the magnetized values.
    for (int component = 0; component < 3; ++component) {
        state.magnetic[component] = 0.0;
    }
    return state;
}

/**
 * \brief load_cell_state for the CGL (bi-Maxwellian) ion closure: the
 * separately evolved internal energies U_par = p_par/2 and U_perp = p_perp
 * replace the isotropic pressure closure.
 *
 * The base load_cell_state is called with the SAME parameters -- under cgl
 * MakeFluxParameters keeps total_energy_closure false, so it fills the
 * (unused) polytropic pressure -- and its ion pressure is then overwritten
 * with the effective isotropic pressure p_eff = (p_par + 2 p_perp)/3,
 * which is what the HLLD wave fan and total-pressure algebra see. The
 * recovery is LINEAR in the unknowns (no kinetic-energy subtraction);
 * p_par = 2 U_par and p_perp = U_perp are floored with the same C-infinity
 * smooth-max idiom as the total-energy pressure recovery. The trace-free
 * deviation stress (p_par - p_perp)(bhat bhat - I/3) is a pointwise RHS
 * source, not part of the face flux.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state_cgl (const amrex::Array4<const amrex::Real>& density,
                     const amrex::Array4<const amrex::Real>& momentum,
                     const amrex::Array4<const amrex::Real>& electron_energy,
                     const amrex::Array4<const amrex::Real>& ion_energy,
                     const amrex::Array4<const amrex::Real>& ion_parallel_energy,
                     const amrex::Array4<const amrex::Real>& ion_perp_energy,
                     const amrex::Array4<const amrex::Real>& current,
                     const int i, const int j, const int k,
                     const int normal_direction,
                     const FluxParameters& parameters) noexcept {
    CellState state =
        load_cell_state(density, momentum, electron_energy, ion_energy,
                        current, i, j, k, normal_direction, parameters);
    state.ion_parallel_energy = ion_parallel_energy(i, j, k);
    state.ion_perp_energy = ion_perp_energy(i, j, k);
    state.ion_internal =
        state.ion_parallel_energy + state.ion_perp_energy;
    const amrex::Real pressure_parallel = smooth_positive_floor(
        2.0 * state.ion_parallel_energy, parameters.ion_pressure_floor);
    const amrex::Real pressure_perp = smooth_positive_floor(
        state.ion_perp_energy, parameters.ion_pressure_floor);
    state.ion_pressure = (pressure_parallel + 2.0 * pressure_perp) / 3.0;
    // The acoustic speeds depend on ion_pressure and must be rebuilt.
    finish_acoustic_speeds(state, normal_direction, parameters);
    return state;
}

/**
 * \brief Load the cell-centered TOTAL magnetic field into the state and
 * upgrade the fast speeds (from the current state.sound_speed) to
 * fast-magnetosonic. Shared tail of load_cell_state_hlld and
 * load_cell_state_hlld_cgl.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
finish_magnetized_state (CellState& state,
                         const amrex::Array4<const amrex::Real>& magnetic,
                         const int i, const int j, const int k,
                         const int normal_direction,
                         const FluxParameters& parameters) noexcept {
    for (int component = 0; component < 3; ++component) {
        state.magnetic[component] = magnetic(i, j, k, component);
    }
    const int tangent1 = (normal_direction + 1) % 3;
    const int tangent2 = (normal_direction + 2) % 3;
    // Rescaled-unit Alfven speeds (b = B/sqrt(mu0)): c_An^2 = b_n^2/rho,
    // c_At^2 = b_t^2/rho.
    const amrex::Real inv_mu0_rho =
        1.0 / (parameters.mu0 * state.safe_density);
    const amrex::Real alfven_normal2 =
        state.magnetic[normal_direction] * state.magnetic[normal_direction] *
        inv_mu0_rho;
    const amrex::Real alfven_tangential2 =
        (state.magnetic[tangent1] * state.magnetic[tangent1] +
         state.magnetic[tangent2] * state.magnetic[tangent2]) *
        inv_mu0_rho;
    const amrex::Real sound2 = state.sound_speed * state.sound_speed;
    // c_f^2 = (a^2 + c_A^2 + sqrt((a^2 - c_A^2)^2 + 4 a^2 c_At^2))/2: the
    // radicand is a sum of non-negative terms, so no clamp is needed and
    // the hydro limit c_f = a is recovered to roundoff.
    const amrex::Real alfven2 = alfven_normal2 + alfven_tangential2;
    const amrex::Real difference = sound2 - alfven2;
    const amrex::Real fast2 =
        0.5 * (sound2 + alfven2 +
               std::sqrt(difference * difference +
                         4.0 * sound2 * alfven_tangential2));
    state.fast_speed = std::sqrt(fast2);
    state.fast_wave_speed =
        std::abs(state.ion_velocity[normal_direction]) + state.fast_speed;
}

/**
 * \brief load_cell_state plus the cell-centered TOTAL magnetic field and
 * the fast-magnetosonic wave speeds required by hlld_flux.
 *
 * The fast speed is evaluated from the cell's own field (including its
 * normal component); hlld_flux itself uses the single-valued face B_n for
 * the wave-fan structure. Existing load_cell_state call sites are
 * undisturbed: they keep hydro-limit values for the new members.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state_hlld (const amrex::Array4<const amrex::Real>& density,
                      const amrex::Array4<const amrex::Real>& momentum,
                      const amrex::Array4<const amrex::Real>& electron_energy,
                      const amrex::Array4<const amrex::Real>& ion_energy,
                      const amrex::Array4<const amrex::Real>& current,
                      const amrex::Array4<const amrex::Real>& magnetic,
                      const int i, const int j, const int k,
                      const int normal_direction,
                      const FluxParameters& parameters) noexcept {
    CellState state =
        load_cell_state(density, momentum, electron_energy, ion_energy,
                        current, i, j, k, normal_direction, parameters);
    finish_magnetized_state(state, magnetic, i, j, k, normal_direction,
                            parameters);
    return state;
}

/**
 * \brief load_cell_state_cgl plus the cell-centered TOTAL magnetic field
 * and the fast-magnetosonic wave speeds (built from the CGL effective
 * pressure p_eff) required by hlld_flux under the cgl closure.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state_hlld_cgl (const amrex::Array4<const amrex::Real>& density,
                          const amrex::Array4<const amrex::Real>& momentum,
                          const amrex::Array4<const amrex::Real>& electron_energy,
                          const amrex::Array4<const amrex::Real>& ion_energy,
                          const amrex::Array4<const amrex::Real>& ion_parallel_energy,
                          const amrex::Array4<const amrex::Real>& ion_perp_energy,
                          const amrex::Array4<const amrex::Real>& current,
                          const amrex::Array4<const amrex::Real>& magnetic,
                          const int i, const int j, const int k,
                          const int normal_direction,
                          const FluxParameters& parameters) noexcept {
    CellState state = load_cell_state_cgl(
        density, momentum, electron_energy, ion_energy, ion_parallel_energy,
        ion_perp_energy, current, i, j, k, normal_direction, parameters);
    finish_magnetized_state(state, magnetic, i, j, k, normal_direction,
                            parameters);
    return state;
}

/**
 * \brief Kinetic-fraction blend weight of the dual-energy ion closure
 * (the reference code's mixmaster mix = -1 weight, vp.f90: fk = 1 - w_k/(gam wik)):
 *
 *     fk = 1 - KE / (gamma_i * smooth-floor(E_i, U_i_floor)),
 *
 * smoothly rectified onto [0, 1] -- the upper bound holds automatically
 * (KE >= 0 and the guarded E_i > 0 give fk_raw <= 1), and the lower clamp
 * is the C-infinity smooth_max of width dual_energy_fk_width, so the
 * whole map is smooth in (E_i, KE) for the matrix-free Jacobian probes
 * (the task-critical population sits exactly at the KE ~ gamma_i E_i
 * corner a hard clamp would kink). The division guard is the ion
 * internal-energy floor, which total_energy/dual_energy require positive.
 *
 * The optional low-internal cutoff (the reference code's mix = -2 variant) zeroes
 * fk where the STEP-OLD U_i sits below cutoff * max_domain(U_i_old);
 * both the mask input and the max are per-solve frozen constants, so the
 * branch is state-independent for the probes.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
dual_energy_kinetic_fraction (const amrex::Real ion_energy,
                              const amrex::Real kinetic_energy,
                              const amrex::Real old_internal_energy,
                              const FluxParameters& parameters) noexcept {
    if (parameters.dual_energy_internal_cutoff > 0.0 &&
        old_internal_energy < parameters.dual_energy_internal_cutoff *
                                  parameters.dual_energy_internal_old_max) {
        return 0.0;
    }
    const amrex::Real internal_floor =
        parameters.ion_pressure_floor / (parameters.gamma_i - 1.0);
    const amrex::Real guarded_energy =
        smooth_positive_floor(ion_energy, internal_floor);
    const amrex::Real raw =
        1.0 - kinetic_energy / (parameters.gamma_i * guarded_energy);
    return smooth_max(raw, 0.0, dual_energy_fk_width);
}

/**
 * \brief Blended ion pressure of the dual-energy closure, the single
 * expression every consumer shares (loaders, RZ geometric sources, the
 * pointwise PdV work, conduction temperatures):
 *
 *     p_i = fk p(E_i) + (1 - fk) smooth-floor((gamma_i - 1) U_i, p_floor),
 *
 * with p(E_i) the SAME C^1 smooth-max recovery (and KE-scaled corner
 * width) as load_cell_state's total-energy branch, and fk the smoothly
 * rectified kinetic fraction above.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
dual_energy_blended_pressure (const amrex::Real ion_energy,
                              const amrex::Real kinetic_energy,
                              const amrex::Real internal_energy,
                              const amrex::Real old_internal_energy,
                              const FluxParameters& parameters) noexcept {
    const amrex::Real internal_floor =
        parameters.ion_pressure_floor / (parameters.gamma_i - 1.0);
    const amrex::Real excess =
        ion_energy - kinetic_energy - internal_floor;
    const amrex::Real corner_width = std::max(
        internal_floor,
        parameters.pressure_corner_width_fraction * kinetic_energy);
    const amrex::Real pressure_total =
        (parameters.gamma_i - 1.0) *
        (internal_floor +
         0.5 * (excess + std::sqrt(excess * excess +
                                   corner_width * corner_width)));
    const amrex::Real pressure_internal = smooth_positive_floor(
        (parameters.gamma_i - 1.0) * internal_energy,
        parameters.ion_pressure_floor);
    const amrex::Real fk = dual_energy_kinetic_fraction(
        ion_energy, kinetic_energy, old_internal_energy, parameters);
    return fk * pressure_total + (1.0 - fk) * pressure_internal;
}

/**
 * \brief load_cell_state for the dual-energy ion closure: the base
 * loader runs with total_energy_closure TRUE (MakeFluxParameters keeps it
 * set under dual_energy), so ion_pressure arrives as the smooth-floored
 * recovery p(E_i) = (gamma_i - 1) smooth-max(E_i - KE, U_i_floor); this
 * wrapper loads the evolved U_i, forms the kinetic-fraction blend
 *
 *     p_i = fk p(E_i) + (1 - fk) smooth-floor((gamma_i - 1) U_i, p_floor)
 *
 * (the reference code's step_velo pressure, vp.f90 ~585-595, with the internal
 * branch floored by the same C-infinity smooth-max idiom as the CGL
 * pressures), overwrites ion_pressure/ion_internal with it, and rebuilds
 * the acoustic speeds -- EVERY pressure consumer (momentum flux, wave
 * fan, signal bounds, conduction temperatures, drain-gate scales) then
 * sees the blend. E_i itself (state.ion_energy and its recovered-channel
 * machinery inside the E_i flux) keeps the total_energy assembly.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state_dual (const amrex::Array4<const amrex::Real>& density,
                      const amrex::Array4<const amrex::Real>& momentum,
                      const amrex::Array4<const amrex::Real>& electron_energy,
                      const amrex::Array4<const amrex::Real>& ion_energy,
                      const amrex::Array4<const amrex::Real>& ion_internal_energy,
                      const amrex::Array4<const amrex::Real>& old_ion_internal_energy,
                      const amrex::Array4<const amrex::Real>& current,
                      const int i, const int j, const int k,
                      const int normal_direction,
                      const FluxParameters& parameters) noexcept {
    CellState state =
        load_cell_state(density, momentum, electron_energy, ion_energy,
                        current, i, j, k, normal_direction, parameters);
    state.ion_internal_energy = ion_internal_energy(i, j, k);
    amrex::Real kinetic_energy = 0.0;
    for (int component = 0; component < 3; ++component) {
        kinetic_energy += state.momentum[component] * state.momentum[component];
    }
    kinetic_energy *= 0.5 / state.safe_density;
    state.ion_pressure = dual_energy_blended_pressure(
        state.ion_energy, kinetic_energy, state.ion_internal_energy,
        old_ion_internal_energy(i, j, k), parameters);
    state.ion_internal = state.ion_pressure / (parameters.gamma_i - 1.0);
    // The acoustic speeds depend on ion_pressure and must be rebuilt.
    finish_acoustic_speeds(state, normal_direction, parameters);
    return state;
}

/**
 * \brief load_cell_state_dual plus the cell-centered TOTAL magnetic field
 * and the fast-magnetosonic wave speeds (built from the blended pressure)
 * required by hlld_flux/central_flux under the dual-energy closure.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CellState
load_cell_state_hlld_dual (const amrex::Array4<const amrex::Real>& density,
                           const amrex::Array4<const amrex::Real>& momentum,
                           const amrex::Array4<const amrex::Real>& electron_energy,
                           const amrex::Array4<const amrex::Real>& ion_energy,
                           const amrex::Array4<const amrex::Real>& ion_internal_energy,
                           const amrex::Array4<const amrex::Real>& old_ion_internal_energy,
                           const amrex::Array4<const amrex::Real>& current,
                           const amrex::Array4<const amrex::Real>& magnetic,
                           const int i, const int j, const int k,
                           const int normal_direction,
                           const FluxParameters& parameters) noexcept {
    CellState state = load_cell_state_dual(
        density, momentum, electron_energy, ion_energy, ion_internal_energy,
        old_ion_internal_energy, current, i, j, k, normal_direction,
        parameters);
    finish_magnetized_state(state, magnetic, i, j, k, normal_direction,
                            parameters);
    return state;
}

// The side-dependent HLLC channels (the LLF ion-energy and centered
// electron-velocity channels of FaceFlux are side-independent).
struct HydroSideFlux {
    amrex::Real mass;
    amrex::GpuArray<amrex::Real, 3> momentum;
    amrex::Real electron_energy;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE HydroSideFlux
hllc_side_flux (const CellState& up, const amrex::Real s_outer,
                const amrex::Real s_star, const bool subsonic,
                const int normal_direction) noexcept
{
    // Side-complete HLLC flux: the physical flux of the given side plus,
    // when the outer wave has crossed the interface (subsonic), the jump
    // to the star state, F* = F_up + S_outer (U* - U_up).
    const amrex::Real u_up = up.ion_velocity[normal_direction];
    const amrex::Real p_up = up.ion_pressure + up.electron_pressure;

    HydroSideFlux flux;
    flux.mass = up.momentum[normal_direction];
    for (int component = 0; component < 3; ++component) {
        flux.momentum[component] =
            up.momentum[normal_direction] * up.ion_velocity[component] +
            (component == normal_direction ? p_up : 0.0);
    }
    flux.electron_energy = up.electron_energy * u_up;

    if (subsonic) {
        const amrex::Real fraction = (s_outer - u_up) / (s_outer - s_star);
        flux.mass += s_outer * (fraction * up.density - up.density);
        for (int component = 0; component < 3; ++component) {
            const amrex::Real star_momentum =
                (component == normal_direction)
                    ? fraction * up.density * s_star
                    : fraction * up.momentum[component];
            flux.momentum[component] +=
                s_outer * (star_momentum - up.momentum[component]);
        }
        flux.electron_energy +=
            s_outer * (fraction * up.electron_energy - up.electron_energy);
    }
    return flux;
}

/**
 * \brief HLLC flux for the hydro block {rho, rho u, U_e} and, with the
 * total-energy closure, the conservative ion total energy E_i.
 *
 * Contact-preserving: at a stationary contact (u = 0, continuous total
 * pressure) the flux reduces to the exact pressure flux with zero diffusive
 * term in mass, momentum, and electron energy, so a held equilibrium is not
 * eroded (Rusanov diffuses every variable at ~c_s there). The electron
 * energy is treated as a scalar advected with the ion contact wave, which
 * requires u_e = u_i (no Hall term); ThetaImplicitMHD::Define() enforces
 * this. Wave-speed estimates are Davis bounds from the two-temperature
 * effective sound speed; the electron-pressure floor keeps them strictly
 * separated, so the star-state denominators cannot vanish.
 *
 * Two JFNK-smoothing options (see FluxParameters): barotropic signal
 * speeds remove the E_i unknown from the wave-speed estimates, and the
 * contact blend replaces the hard upwind-side switch at s_star = 0 with a
 * C-infinity blend of the two side-complete star fluxes.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FaceFlux
hllc_flux (const CellState& left, const CellState& right,
           const int normal_direction,
           const FluxParameters& parameters) noexcept {
    const amrex::Real u_left = left.ion_velocity[normal_direction];
    const amrex::Real u_right = right.ion_velocity[normal_direction];
    const amrex::Real p_left = left.ion_pressure + left.electron_pressure;
    const amrex::Real p_right = right.ion_pressure + right.electron_pressure;

    const amrex::Real s_left =
        std::min(u_left - left.sound_speed, u_right - right.sound_speed);
    const amrex::Real s_right =
        std::max(u_left + left.sound_speed, u_right + right.sound_speed);
    // m_k = rho_k (S_k - u_k): m_left < 0 <= -rho c, m_right > 0, so the
    // denominator m_left - m_right is strictly negative.
    const amrex::Real m_left = left.safe_density * (s_left - u_left);
    const amrex::Real m_right = right.safe_density * (s_right - u_right);
    const amrex::Real s_star =
        (p_right - p_left + m_left * u_left - m_right * u_right) /
        (m_left - m_right);

    FaceFlux flux;
    // Component hybrid: the ion total energy uses a local Lax-Friedrichs
    // flux (enthalpy form, same total pressure as the momentum system)
    // rather than the HLLC star construction. Toro's energy star state
    // reads any pressure jump as wave-borne; across a MAGNETICALLY
    // balanced unresolved gradient (an FRC separatrix: u ~ 0, total
    // pressure jump held by J x B) it pumps O(p_total)-scale energy out
    // of the low-pressure side with no compensating source, overdrawing
    // the p_i-scale E_i there and locking the Newton line search. The
    // LLF form is exactly zero at stagnant faces (central part vanishes
    // at u = 0; the alpha term diffuses E_i itself, which is small where
    // it matters), at the cost of c_s-scale diffusion of the ion energy
    // channel only -- mass, momentum, and U_e keep the contact-preserving
    // HLLC flux.
    {
        const amrex::Real u_l = left.ion_velocity[normal_direction];
        const amrex::Real u_r = right.ion_velocity[normal_direction];
        const amrex::Real alpha_energy =
            std::max(left.wave_speed, right.wave_speed);
        flux.ion_energy =
            0.5 * (u_l * (left.ion_energy + left.ion_pressure +
                          left.electron_pressure) +
                   u_r * (right.ion_energy + right.ion_pressure +
                          right.electron_pressure) -
                   alpha_energy * (right.ion_energy - left.ion_energy));
    }
    flux.electron_velocity =
        0.5 * (left.electron_velocity_normal + right.electron_velocity_normal);

    // NO star correction for the ion energy: its LLF flux above is
    // complete (see the component-hybrid comment). The remaining channels
    // take the side-complete flux of the contact-upwind side.
    if (parameters.contact_blend > 0.0) {
        // C-infinity blend of the two side-complete star fluxes over
        // |s_star| ~ kappa (|s_left| + |s_right|) / 2, replacing the hard
        // switch at s_star = 0. At s_star = 0 the two branches carry the
        // identical flux (HLLC flux continuity across the contact), so
        // the blend is inert there and a static contact stays machine-
        // preserved; away from the blend width it reduces to the standard
        // upwind selection. The hard switch flips under matrix-free
        // Jacobian probes at near-stagnant balanced faces (an FRC
        // separatrix), which defeats the Newton line search.
        const HydroSideFlux left_flux = hllc_side_flux(
            left, s_left, s_star, s_left < 0.0, normal_direction);
        const HydroSideFlux right_flux = hllc_side_flux(
            right, s_right, s_star, s_right > 0.0, normal_direction);
        const amrex::Real width = parameters.contact_blend * 0.5 *
                                  (std::abs(s_left) + std::abs(s_right));
        const amrex::Real weight_left =
            0.5 *
            (1.0 + s_star / std::sqrt(s_star * s_star + width * width));
        flux.mass = weight_left * left_flux.mass +
                    (1.0 - weight_left) * right_flux.mass;
        for (int component = 0; component < 3; ++component) {
            flux.momentum[component] =
                weight_left * left_flux.momentum[component] +
                (1.0 - weight_left) * right_flux.momentum[component];
        }
        flux.electron_energy =
            weight_left * left_flux.electron_energy +
            (1.0 - weight_left) * right_flux.electron_energy;
    } else {
        // upwind side by the sign of the contact speed
        const bool from_left = (s_star >= 0.0);
        const HydroSideFlux up_flux =
            from_left ? hllc_side_flux(left, s_left, s_star, s_left < 0.0,
                                       normal_direction)
                      : hllc_side_flux(right, s_right, s_star,
                                       s_right > 0.0, normal_direction);
        flux.mass = up_flux.mass;
        flux.momentum = up_flux.momentum;
        flux.electron_energy = up_flux.electron_energy;
    }
    return flux;
}

/**
 * \brief HLLD flux (Miyoshi & Kusano, JCP 208 (2005) 315, "MK05") for the
 * conservative ideal-MHD ion block, in a smoothed, JFNK-friendly form.
 *
 * Kernel-only scope for the conservative-form recast: this function is
 * PURE -- it applies no floor_outflow_limiter guards (face_flux-level code
 * applies them post-flux with theta-extrapolated donor values, exactly as
 * face_flux does for hllc/rusanov today) and it is not yet wired into
 * face_flux.
 *
 * Units and inputs: the kernel works in rescaled field units
 * b = B/sqrt(mu0), so the MK05 formulas apply verbatim with mu0 = 1.
 * CellState.magnetic carries the cell-centered TOTAL B in physical units;
 * bn_face is the single-valued normal field on the face (physical units).
 * The two-temperature total pressure is p_T = p_i + p_e + |b|^2/2.
 *
 * Wave fan: S_L/S_R are SMOOTHED Davis bounds built from the fast
 * magnetosonic speeds (smooth_min/smooth_max of width
 * kappa_signal (c_fL + c_fR)); S_M and the star states are the standard
 * single-valued MK05 expressions (the star total pressure p_T*, MK05
 * eq. 41, is implied by S_M through the star construction and never needs
 * to be formed explicitly). Region selection uses NO if-chains: the flux
 * is the symmetric telescoped ("central") form of the wave fan,
 *
 *   F = (F_L + F_R)/2 - (1/2) sum_j q(S_j) (U_j - U_{j-1}),
 *
 * with q(S) = S^2/sqrt(S^2 + w_c^2), the C-infinity |S|. Under hard
 * switches (w_c = 0) this is identical to upwind region selection for the
 * exactly conservative channels; q(0) = 0 makes every blended channel
 * vanish identically at an exact static contact.
 *
 * B_n -> 0 degeneracy: the rotational (Alfven) layer -- the MK05
 * double-star states -- enters as an ADDITIVE correction weighted by
 * chi = B_n^2/(B_n^2 + eps_B^2), eps_B^2 = kappa_bn^2 mu0 mean(p_T), and
 * every sign(B_n) is smoothed to B_n/sqrt(B_n^2 + eps_B^2). At B_n = 0
 * the scheme reduces EXACTLY to the HLLC-like two-state fan (with
 * magnetic pressure in p_T and B_t advected with the contact); with
 * B = 0 and kappa_signal = kappa_contact = kappa_denominator = 0 it
 * reproduces hllc_flux to roundoff. Star denominators are guarded with
 * the C^1 smooth-max idiom of load_cell_state, and the star fractions are
 * arranged as 1 + (S_M - u)/(S - S_M) so that F(U, U) equals the exact
 * physical flux to machine precision independent of the guards.
 *
 * Channels: mass, momentum[3] (TOTAL, including Maxwell stress) with the
 * Maxwell-stress part returned separately in momentum_magnetic[3] (a
 * region-blended stress state, so momentum - momentum_magnetic is the
 * hydro part); U_e advected passively with the contact (as in hllc_flux);
 * the ion energy stays the LLF enthalpy channel (H = E_i + p_i + p_e, no
 * magnetic terms) with the fast-speed alpha.
 *
 * SIGN CONVENTION (induction channels): with t1 = (n+1)%3 and
 * t2 = (n+2)%3 (cyclic), induction_t1/induction_t2 are the conservative
 * fluxes of B_t1/B_t2 through the n-face,
 *   F_{B_t} = u_n B_t - u_t B_n,
 * equivalently F_{B_t1} = -EMF_t2 and F_{B_t2} = +EMF_t1 with
 * EMF = -u x B. For the z-face (n = 2) in 1D:
 * F_Bx = u_z Bx - u_x Bz = -EMF_y and F_By = u_z By - u_y Bz = +EMF_x.
 * The smoothed signal bounds are returned in signal_left/signal_right for
 * the recast's UCT edge assembly.
 *
 * Requires p_T > 0 on both sides (guaranteed by the pressure floors) so
 * the degeneracy scale eps_B^2 is strictly positive.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FaceFlux
hlld_flux (const CellState& left, const CellState& right,
           const amrex::Real bn_face, const int normal_direction,
           const FluxParameters& parameters) noexcept {
    const int tangent1 = (normal_direction + 1) % 3;
    const int tangent2 = (normal_direction + 2) % 3;

    // Rescaled field units b = B/sqrt(mu0).
    const amrex::Real sqrt_mu0 = std::sqrt(parameters.mu0);
    const amrex::Real inv_sqrt_mu0 = 1.0 / sqrt_mu0;
    const amrex::Real bn = bn_face * inv_sqrt_mu0;
    const amrex::Real bn2 = bn * bn;

    const amrex::Real u_left = left.ion_velocity[normal_direction];
    const amrex::Real u_right = right.ion_velocity[normal_direction];
    const amrex::Real bt1_left = left.magnetic[tangent1] * inv_sqrt_mu0;
    const amrex::Real bt2_left = left.magnetic[tangent2] * inv_sqrt_mu0;
    const amrex::Real bt1_right = right.magnetic[tangent1] * inv_sqrt_mu0;
    const amrex::Real bt2_right = right.magnetic[tangent2] * inv_sqrt_mu0;
    const amrex::Real p_total_left =
        left.ion_pressure + left.electron_pressure +
        0.5 * (bn2 + bt1_left * bt1_left + bt2_left * bt2_left);
    const amrex::Real p_total_right =
        right.ion_pressure + right.electron_pressure +
        0.5 * (bn2 + bt1_right * bt1_right + bt2_right * bt2_right);

    // Smoothed Davis bounds from the fast magnetosonic speeds.
    const amrex::Real signal_width =
        parameters.hlld_kappa_signal * (left.fast_speed + right.fast_speed);
    const amrex::Real s_left = smooth_min(
        u_left - left.fast_speed, u_right - right.fast_speed, signal_width);
    const amrex::Real s_right = smooth_max(
        u_left + left.fast_speed, u_right + right.fast_speed, signal_width);

    // Barotropic fan closure: the fan structure sees the polytropic
    // p_i(rho); the physical fluxes keep the consistent pressure (see
    // FluxParameters::hlld_barotropic_fan).
    amrex::Real p_fan_left = p_total_left;
    amrex::Real p_fan_right = p_total_right;
    if (parameters.hlld_barotropic_fan) {
        p_fan_left +=
            ion_pressure(left.safe_density, parameters) - left.ion_pressure;
        p_fan_right += ion_pressure(right.safe_density, parameters) -
                       right.ion_pressure;
    }

    // m_k = rho_k (S_k - u_k): m_left <= -rho_L c_fL < 0 and
    // m_right >= rho_R c_fR > 0 (smoothing only widens the fan), so the
    // S_M denominator is strictly negative. Same expression as hllc_flux
    // with p -> p_T (MK05 eq. 38).
    const amrex::Real m_left = left.safe_density * (s_left - u_left);
    const amrex::Real m_right = right.safe_density * (s_right - u_right);
    const amrex::Real s_mid =
        (p_fan_right - p_fan_left + m_left * u_left -
         m_right * u_right) /
        (m_left - m_right);

    // LHLLD all-speed pressure correction (Minoshima & Miyoshi, JCP 446
    // (2021) 110639, eqs. 14-16): the velocity-difference term inside
    // the fan's star total pressure scales with the fast speed and acts
    // as a normal-momentum diffusion ~ rho c_f (u_R - u_L); at low Mach
    // a velocity difference is not necessarily compression (rotational
    // flow), and the term degrades the solution and pumps noise-driven
    // pressure work into near-stagnant regions. Scale it by
    // phi = chi (2 - chi), chi = min(1, max(c_u,L, c_u,R)/c_f,max),
    // where c_u is the fast-speed formula with |u| in place of the
    // sound speed -- phi -> 1 for super-magnetosonic flow (original
    // HLLD) and -> 0 at stagnation -- assembled C-infinity with the
    // usual signal widths.
    amrex::Real phi_all_speed = 1.0;
    if (parameters.hlld_all_speed) {
        const auto modified_fast = [&] (const CellState& state,
                                        const amrex::Real bt1,
                                        const amrex::Real bt2) noexcept {
            const amrex::Real alfven2 =
                (bn2 + bt1 * bt1 + bt2 * bt2) / state.safe_density;
            const amrex::Real alfven_normal2 = bn2 / state.safe_density;
            amrex::Real speed2 = 0.0;
            for (int component = 0; component < 3; ++component) {
                speed2 += state.ion_velocity[component] *
                          state.ion_velocity[component];
            }
            const amrex::Real total = alfven2 + speed2;
            const amrex::Real radicand =
                total * total - 4.0 * speed2 * alfven_normal2;
            return std::sqrt(
                0.5 * (total + std::sqrt(std::max(radicand, 0.0))));
        };
        const amrex::Real signal_width_u =
            parameters.hlld_kappa_signal *
            (left.fast_speed + right.fast_speed);
        const amrex::Real c_u = smooth_max(
            modified_fast(left, bt1_left, bt2_left),
            modified_fast(right, bt1_right, bt2_right), signal_width_u);
        const amrex::Real c_f_max = smooth_max(
            left.fast_speed, right.fast_speed, signal_width_u);
        const amrex::Real chi_all_speed = smooth_min(
            1.0, c_u / c_f_max, parameters.hlld_kappa_signal);
        phi_all_speed = chi_all_speed * (2.0 - chi_all_speed);
    }
    // The (u_R - u_L) content of the star total pressure (the third term
    // of MK05 eq. 41 / Minoshima-Miyoshi eq. 7): m_L m_R < 0 and
    // m_R - m_L > 0 strictly, so no denominator guard is needed.
    const amrex::Real pressure_velocity_term =
        m_left * m_right * (u_right - u_left) / (m_right - m_left);

    // B_n -> 0 degeneracy scales; eps_b^2 = kappa_bn^2 mean(p_T) in
    // b-units (= kappa_bn^2 mu0 mean(p_T) in field units).
    const amrex::Real eps_b2 = parameters.hlld_kappa_bn *
                               parameters.hlld_kappa_bn * 0.5 *
                               (p_total_left + p_total_right);
    const amrex::Real chi = bn2 / (bn2 + eps_b2);
    const amrex::Real sign_bn = bn / std::sqrt(bn2 + eps_b2);
    const amrex::Real abs_bn = bn * sign_bn;

    // Guarded star denominators. The wave gaps S_M - S_L and S_R - S_M
    // are positive for admissible states; the MK05 tangential
    // denominators D = rho (S - u)(S - S_M) - b_n^2 can vanish at the
    // switch-on/switch-off degeneracy, so both get smooth-max floors.
    const amrex::Real guard_speed = parameters.hlld_kappa_denominator *
                                    0.5 *
                                    (left.fast_speed + right.fast_speed);
    const amrex::Real gap_left =
        smooth_positive_floor(s_mid - s_left, guard_speed);
    const amrex::Real gap_right =
        smooth_positive_floor(s_right - s_mid, guard_speed);
    const amrex::Real guard_dissipation =
        parameters.hlld_kappa_denominator * (bn2 + eps_b2);
    const amrex::Real d_left = smooth_positive_floor(
        m_left * (s_left - s_mid) - bn2, guard_dissipation);
    const amrex::Real d_right = smooth_positive_floor(
        m_right * (s_right - s_mid) - bn2, guard_dissipation);

    // Star fractions (S - u)/(S - S_M) written as 1 + (S_M - u)/(S - S_M)
    // and the star numerators arranged to carry an explicit (S_M - u)
    // factor: at S_M = u (consistency) every star state equals the input
    // state exactly, independent of the denominator guards.
    const amrex::Real slip_left = s_mid - u_left;
    const amrex::Real slip_right = s_mid - u_right;
    const amrex::Real fraction_left = 1.0 - slip_left / gap_left;
    const amrex::Real fraction_right = 1.0 + slip_right / gap_right;

    // MK05 single-star states (eqs. 39-47). The raw-density convention
    // mirrors hllc_side_flux so the B -> 0 limit reproduces hllc_flux.
    const amrex::Real rho_star_left = left.density * fraction_left;
    const amrex::Real rho_star_right = right.density * fraction_right;
    const amrex::Real ue_star_left =
        fraction_left * left.electron_energy;
    const amrex::Real ue_star_right =
        fraction_right * right.electron_energy;
    const amrex::Real ratio_left = slip_left / d_left;
    const amrex::Real ratio_right = slip_right / d_right;
    // v_t* = v_t - b_n b_t (S_M - u)/D (MK05 eqs. 42, 46).
    const amrex::Real vt1_star_left =
        left.ion_velocity[tangent1] - bn * bt1_left * ratio_left;
    const amrex::Real vt2_star_left =
        left.ion_velocity[tangent2] - bn * bt2_left * ratio_left;
    const amrex::Real vt1_star_right =
        right.ion_velocity[tangent1] - bn * bt1_right * ratio_right;
    const amrex::Real vt2_star_right =
        right.ion_velocity[tangent2] - bn * bt2_right * ratio_right;
    // b_t* = b_t (rho (S - u)^2 - b_n^2)/D = b_t (1 + m (S_M - u)/D)
    // (MK05 eqs. 45, 47).
    const amrex::Real bt1_star_left =
        bt1_left * (1.0 + m_left * ratio_left);
    const amrex::Real bt2_star_left =
        bt2_left * (1.0 + m_left * ratio_left);
    const amrex::Real bt1_star_right =
        bt1_right * (1.0 + m_right * ratio_right);
    const amrex::Real bt2_star_right =
        bt2_right * (1.0 + m_right * ratio_right);
    // Star momenta (raw-momentum convention of hllc_side_flux).
    const amrex::Real momn_star_left =
        fraction_left * left.density * s_mid;
    const amrex::Real momn_star_right =
        fraction_right * right.density * s_mid;
    const amrex::Real momt1_star_left =
        fraction_left * (left.momentum[tangent1] -
                         left.safe_density * bn * bt1_left * ratio_left);
    const amrex::Real momt2_star_left =
        fraction_left * (left.momentum[tangent2] -
                         left.safe_density * bn * bt2_left * ratio_left);
    const amrex::Real momt1_star_right =
        fraction_right * (right.momentum[tangent1] -
                          right.safe_density * bn * bt1_right * ratio_right);
    const amrex::Real momt2_star_right =
        fraction_right * (right.momentum[tangent2] -
                          right.safe_density * bn * bt2_right * ratio_right);

    // Rotational (Alfven) layer: MK05 double-star states (eqs. 59-62)
    // with sign(B_n) -> smoothed sign. sqrt arguments are smooth-max
    // guarded; the whole layer is chi-weighted, so at B_n = 0 it drops
    // out exactly and these guards never perturb the base fan.
    const amrex::Real density_guard =
        parameters.density_floor +
        parameters.hlld_kappa_denominator * 0.5 *
            (left.safe_density + right.safe_density);
    const amrex::Real sqrt_rho_left =
        std::sqrt(smooth_positive_floor(rho_star_left, density_guard));
    const amrex::Real sqrt_rho_right =
        std::sqrt(smooth_positive_floor(rho_star_right, density_guard));
    const amrex::Real inv_sqrt_sum = 1.0 / (sqrt_rho_left + sqrt_rho_right);
    const amrex::Real s_alfven_left = s_mid - abs_bn / sqrt_rho_left;
    const amrex::Real s_alfven_right = s_mid + abs_bn / sqrt_rho_right;
    const amrex::Real vt1_dstar =
        (sqrt_rho_left * vt1_star_left + sqrt_rho_right * vt1_star_right +
         (bt1_star_right - bt1_star_left) * sign_bn) *
        inv_sqrt_sum;
    const amrex::Real vt2_dstar =
        (sqrt_rho_left * vt2_star_left + sqrt_rho_right * vt2_star_right +
         (bt2_star_right - bt2_star_left) * sign_bn) *
        inv_sqrt_sum;
    const amrex::Real bt1_dstar =
        (sqrt_rho_left * bt1_star_right + sqrt_rho_right * bt1_star_left +
         sqrt_rho_left * sqrt_rho_right * (vt1_star_right - vt1_star_left) *
             sign_bn) *
        inv_sqrt_sum;
    const amrex::Real bt2_dstar =
        (sqrt_rho_left * bt2_star_right + sqrt_rho_right * bt2_star_left +
         sqrt_rho_left * sqrt_rho_right * (vt2_star_right - vt2_star_left) *
             sign_bn) *
        inv_sqrt_sum;

    // Physical fluxes of the two input states (b-units).
    const amrex::Real mass_flux_left = left.momentum[normal_direction];
    const amrex::Real mass_flux_right = right.momentum[normal_direction];
    const amrex::Real momn_flux_left =
        left.momentum[normal_direction] * u_left + p_total_left - bn2;
    const amrex::Real momn_flux_right =
        right.momentum[normal_direction] * u_right + p_total_right - bn2;
    const amrex::Real momt1_flux_left =
        left.momentum[normal_direction] * left.ion_velocity[tangent1] -
        bn * bt1_left;
    const amrex::Real momt2_flux_left =
        left.momentum[normal_direction] * left.ion_velocity[tangent2] -
        bn * bt2_left;
    const amrex::Real momt1_flux_right =
        right.momentum[normal_direction] * right.ion_velocity[tangent1] -
        bn * bt1_right;
    const amrex::Real momt2_flux_right =
        right.momentum[normal_direction] * right.ion_velocity[tangent2] -
        bn * bt2_right;
    const amrex::Real ue_flux_left = left.electron_energy * u_left;
    const amrex::Real ue_flux_right = right.electron_energy * u_right;
    const amrex::Real bt1_flux_left =
        u_left * bt1_left - left.ion_velocity[tangent1] * bn;
    const amrex::Real bt2_flux_left =
        u_left * bt2_left - left.ion_velocity[tangent2] * bn;
    const amrex::Real bt1_flux_right =
        u_right * bt1_right - right.ion_velocity[tangent1] * bn;
    const amrex::Real bt2_flux_right =
        u_right * bt2_right - right.ion_velocity[tangent2] * bn;

    // C-infinity region blends.
    const amrex::Real contact_width =
        parameters.hlld_kappa_contact * 0.5 *
        (std::abs(s_left) + std::abs(s_right));
    const amrex::Real q_left = smooth_abs(s_left, contact_width);
    const amrex::Real q_mid = smooth_abs(s_mid, contact_width);
    const amrex::Real q_right = smooth_abs(s_right, contact_width);
    const amrex::Real q_alfven_left =
        smooth_abs(s_alfven_left, contact_width);
    const amrex::Real q_alfven_right =
        smooth_abs(s_alfven_right, contact_width);
    // The rotational layer only redistributes dissipation inside the fan
    // (its jumps across S_L*, S_M, S_R* telescope to zero), so it enters
    // as (q(S*) - q(S_M)) times the Alfven-wave jumps, chi-weighted.
    const amrex::Real weight_alfven_left =
        0.5 * chi * (q_alfven_left - q_mid);
    const amrex::Real weight_alfven_right =
        0.5 * chi * (q_alfven_right - q_mid);

    // Symmetric telescoped fan flux.
    const auto central = [&] (const amrex::Real flux_l,
                              const amrex::Real flux_r,
                              const amrex::Real jump_l,
                              const amrex::Real jump_m,
                              const amrex::Real jump_r) noexcept {
        return 0.5 * (flux_l + flux_r) -
               0.5 * (q_left * jump_l + q_mid * jump_m + q_right * jump_r);
    };

    FaceFlux flux;
    flux.mass = central(mass_flux_left, mass_flux_right,
                        rho_star_left - left.density,
                        rho_star_right - rho_star_left,
                        right.density - rho_star_right);
    flux.momentum[normal_direction] =
        central(momn_flux_left, momn_flux_right,
                momn_star_left - left.momentum[normal_direction],
                momn_star_right - momn_star_left,
                right.momentum[normal_direction] - momn_star_right);
    // Tangential momentum: base fan plus the chi-weighted rotational
    // redistribution (Alfven jumps rho* (v** - v*)).
    flux.momentum[tangent1] =
        central(momt1_flux_left, momt1_flux_right,
                momt1_star_left - left.momentum[tangent1],
                momt1_star_right - momt1_star_left,
                right.momentum[tangent1] - momt1_star_right) -
        weight_alfven_left * rho_star_left * (vt1_dstar - vt1_star_left) -
        weight_alfven_right * rho_star_right * (vt1_star_right - vt1_dstar);
    flux.momentum[tangent2] =
        central(momt2_flux_left, momt2_flux_right,
                momt2_star_left - left.momentum[tangent2],
                momt2_star_right - momt2_star_left,
                right.momentum[tangent2] - momt2_star_right) -
        weight_alfven_left * rho_star_left * (vt2_dstar - vt2_star_left) -
        weight_alfven_right * rho_star_right * (vt2_star_right - vt2_dstar);
    flux.electron_energy =
        central(ue_flux_left, ue_flux_right,
                ue_star_left - left.electron_energy,
                ue_star_right - ue_star_left,
                right.electron_energy - ue_star_right);
    if (parameters.cgl_closure) {
        // CGL internal-energy channels: pure advection with the contact
        // (star values fraction * U, the U_e pattern). The CGL work terms
        // (-p_par bb:grad u, -p_perp (div u - bb:grad u)) and the
        // isotropization exchange are pointwise sources in the fluid RHS,
        // not fluxes: this is the internal-energy (non-conservative)
        // formulation.
        const amrex::Real upar_star_left =
            fraction_left * left.ion_parallel_energy;
        const amrex::Real upar_star_right =
            fraction_right * right.ion_parallel_energy;
        const amrex::Real uperp_star_left =
            fraction_left * left.ion_perp_energy;
        const amrex::Real uperp_star_right =
            fraction_right * right.ion_perp_energy;
        flux.ion_parallel_energy =
            central(left.ion_parallel_energy * u_left,
                    right.ion_parallel_energy * u_right,
                    upar_star_left - left.ion_parallel_energy,
                    upar_star_right - upar_star_left,
                    right.ion_parallel_energy - upar_star_right);
        flux.ion_perp_energy =
            central(left.ion_perp_energy * u_left,
                    right.ion_perp_energy * u_right,
                    uperp_star_left - left.ion_perp_energy,
                    uperp_star_right - uperp_star_left,
                    right.ion_perp_energy - uperp_star_right);
    }
    if (parameters.dual_energy_closure) {
        // Dual-energy auxiliary internal energy: pure advection with the
        // contact (star values fraction * U_i, the U_e/CGL pattern). The
        // blended PdV work -p_blend div u and the viscous heating are
        // pointwise sources in the fluid RHS, not fluxes: U_i is the
        // internal-energy (non-conservative) formulation riding next to
        // the conservative E_i channel above.
        const amrex::Real uint_star_left =
            fraction_left * left.ion_internal_energy;
        const amrex::Real uint_star_right =
            fraction_right * right.ion_internal_energy;
        flux.ion_internal_energy =
            central(left.ion_internal_energy * u_left,
                    right.ion_internal_energy * u_right,
                    uint_star_left - left.ion_internal_energy,
                    uint_star_right - uint_star_left,
                    right.ion_internal_energy - uint_star_right);
    }
    flux.induction_t1 =
        sqrt_mu0 *
        (central(bt1_flux_left, bt1_flux_right, bt1_star_left - bt1_left,
                 bt1_star_right - bt1_star_left, bt1_right - bt1_star_right) -
         weight_alfven_left * (bt1_dstar - bt1_star_left) -
         weight_alfven_right * (bt1_star_right - bt1_dstar));
    flux.induction_t2 =
        sqrt_mu0 *
        (central(bt2_flux_left, bt2_flux_right, bt2_star_left - bt2_left,
                 bt2_star_right - bt2_star_left, bt2_right - bt2_star_right) -
         weight_alfven_left * (bt2_dstar - bt2_star_left) -
         weight_alfven_right * (bt2_star_right - bt2_dstar));

    // Ion total energy. The fan-consistent star channel (default) builds
    // E_i* from the S_L/S_R jump conditions of the model equation
    // dE_i/dt + div(u (E_i + p_i + p_e)) = sources (the magnetic work
    // arrives through the SEPARATE discrete stress difference, so the
    // flux carries only the gas work): the gas star pressure is the
    // fan's single-valued total star pressure (MK05 eq. 41, evaluated
    // with the fan pressures) minus each side's star magnetic pressure.
    // Because S_M comes from the TOTAL pressure, a magnetically balanced
    // static gradient (an FRC separatrix or halo: u ~ 0, gas jump held
    // by J x B) gives S_M = 0 and star states equal to the inputs -- no
    // energy pumping (the Toro gas-star pathology that motivated the
    // hllc component hybrid does not arise) and only |S_M|-scale
    // dissipation of contact jumps, instead of the LLF channel's
    // fast-speed diffusion of every E_i gradient (which visibly drains
    // near-static ion energy along strong-B_n field lines). The LLF
    // enthalpy channel is retained as a fallback
    // (FluxParameters::hlld_ion_energy_star = false).
    if (parameters.hlld_ion_energy_star) {
        const amrex::Real p_t_star =
            0.5 * ((p_fan_left + m_left * slip_left) +
                   (p_fan_right + m_right * slip_right)) +
            (phi_all_speed - 1.0) * pressure_velocity_term;
        const amrex::Real p_gas_left =
            left.ion_pressure + left.electron_pressure;
        const amrex::Real p_gas_right =
            right.ion_pressure + right.electron_pressure;
        const amrex::Real p_gas_star_left =
            p_t_star - 0.5 * (bn2 + bt1_star_left * bt1_star_left +
                              bt2_star_left * bt2_star_left);
        const amrex::Real p_gas_star_right =
            p_t_star - 0.5 * (bn2 + bt1_star_right * bt1_star_right +
                              bt2_star_right * bt2_star_right);
        // Arranged with explicit slip/consistency factors: at S_M = u
        // (and hence p_gas* = p_gas) the star energy equals the input
        // state exactly, independent of the guarded gaps.
        const amrex::Real ei_star_left =
            left.ion_energy -
            (slip_left * left.ion_energy + p_gas_star_left * s_mid -
             p_gas_left * u_left) /
                gap_left;
        const amrex::Real ei_star_right =
            right.ion_energy +
            (slip_right * right.ion_energy + p_gas_star_right * s_mid -
             p_gas_right * u_right) /
                gap_right;
        flux.ion_energy =
            central(u_left * (left.ion_energy + p_gas_left),
                    u_right * (right.ion_energy + p_gas_right),
                    ei_star_left - left.ion_energy,
                    ei_star_right - ei_star_left,
                    right.ion_energy - ei_star_right);
    } else {
        // LLF enthalpy channel with the fast-speed alpha, assembled
        // C-infinity: both the |u_n| of each side (a hard kink exactly
        // where a fan-internal face crosses stagnation) and the
        // left/right max are smoothed with the signal width; the
        // smoothing only raises alpha, and smooth_abs(0) = 0 keeps
        // static contacts exact.
        const amrex::Real alpha_energy = smooth_max(
            smooth_abs(u_left, signal_width) + left.fast_speed,
            smooth_abs(u_right, signal_width) + right.fast_speed,
            signal_width);
        flux.ion_energy =
            0.5 * (u_left * (left.ion_energy + left.ion_pressure +
                             left.electron_pressure) +
                   u_right * (right.ion_energy + right.ion_pressure +
                              right.electron_pressure) -
                   alpha_energy * (right.ion_energy - left.ion_energy));
    }
    flux.electron_velocity =
        0.5 * (left.electron_velocity_normal + right.electron_velocity_normal);

    // Maxwell-stress part of the momentum flux: the region-blended stress
    // STATE (weights telescope with the smoothed region signs, so under
    // hard switches this is the stress of the region containing the
    // interface). Already in physical units: b_i b_j = B_i B_j / mu0.
    const amrex::Real r_left = smooth_sign(s_left, contact_width);
    const amrex::Real r_mid = smooth_sign(s_mid, contact_width);
    const amrex::Real r_right = smooth_sign(s_right, contact_width);
    const amrex::Real r_alfven_left =
        smooth_sign(s_alfven_left, contact_width);
    const amrex::Real r_alfven_right =
        smooth_sign(s_alfven_right, contact_width);
    // Apply the all-speed correction to the star total pressure implied
    // by the telescoped normal-momentum flux, weighted by the (smoothed)
    // star-region indicator: for S_L < 0 < S_R the interface sits in the
    // fan and the weight is ~1; it vanishes when the whole fan is swept
    // to one side.
    flux.momentum[normal_direction] +=
        0.5 * (r_right - r_left) * (phi_all_speed - 1.0) *
        pressure_velocity_term;
    const auto stress_blend = [&] (const amrex::Real x_left_state,
                                   const amrex::Real x_star_left,
                                   const amrex::Real x_dstar,
                                   const amrex::Real x_star_right,
                                   const amrex::Real x_right_state) noexcept {
        const amrex::Real base =
            0.5 * (x_left_state + x_right_state) -
            0.5 * (r_left * (x_star_left - x_left_state) +
                   r_mid * (x_star_right - x_star_left) +
                   r_right * (x_right_state - x_star_right));
        return base - 0.5 * chi *
                          ((r_alfven_left - r_mid) *
                               (x_dstar - x_star_left) +
                           (r_alfven_right - r_mid) *
                               (x_star_right - x_dstar));
    };
    const auto stress_normal = [&] (const amrex::Real b1,
                                    const amrex::Real b2) noexcept {
        return 0.5 * (b1 * b1 + b2 * b2) - 0.5 * bn2;
    };
    flux.momentum_magnetic[normal_direction] =
        stress_blend(stress_normal(bt1_left, bt2_left),
                     stress_normal(bt1_star_left, bt2_star_left),
                     stress_normal(bt1_dstar, bt2_dstar),
                     stress_normal(bt1_star_right, bt2_star_right),
                     stress_normal(bt1_right, bt2_right));
    flux.momentum_magnetic[tangent1] =
        stress_blend(-bn * bt1_left, -bn * bt1_star_left, -bn * bt1_dstar,
                     -bn * bt1_star_right, -bn * bt1_right);
    flux.momentum_magnetic[tangent2] =
        stress_blend(-bn * bt2_left, -bn * bt2_star_left, -bn * bt2_dstar,
                     -bn * bt2_star_right, -bn * bt2_right);

    flux.signal_left = s_left;
    flux.signal_right = s_right;
    flux.signal_alfven_left = s_alfven_left;
    flux.signal_alfven_right = s_alfven_right;
    return flux;
}

/**
 * \brief Central conservative flux for the conservative-form recast
 * (Chacon-style co-located central FV with explicit dissipation, JCP 526
 * (2025) 113789): every channel is the exact q, r -> 0 limit of
 * hlld_flux's telescoped wave fan -- the arithmetic mean of the two
 * sides' physical fluxes with NO Riemann dissipation and no fan algebra
 * on the JFNK matvec hot path. Nonlinear stability must come from the
 * explicit viscous face flux the caller adds (ComputeDirectionalFaceFluxes;
 * implicit_mhd.viscosity > 0 is enforced for fluid_flux = central) and
 * from resistivity on the field channels.
 *
 * Channel accounting matches hlld_flux exactly:
 * - momentum[] is the TOTAL flux (hydro plus Maxwell stress, with the
 *   total pressure p_i + p_e + |b|^2/2 and the single-valued face B_n in
 *   every b_n factor); momentum_magnetic[] is the centered Maxwell-stress
 *   state, so momentum - momentum_magnetic is the pure hydro flux and
 *   -div(momentum_magnetic) is the discrete J x B whose work the
 *   ion-energy equation integrates.
 * - ion_energy carries the GAS enthalpy channel u_n (E_i + p_i + p_e)
 *   ONLY: the magnetic work reaches E_i through the separate discrete
 *   stress difference (ComputeFluidRHSFromFaceFluxes), never through this
 *   flux, exactly as in the hlld star/llf channels.
 * - induction_t1/t2 are the centered ideal induction fluxes
 *   u_n B_t - u_t B_n (the face EMF ingredients of the Ohm assembly, same
 *   sign convention as hlld_flux).
 * - signal_left/right keep the smoothed Davis fast bounds: the donor
 *   drain gates and the RZ corner-EMF upwind weights need a strictly
 *   positive signal span for their C-infinity blend widths.
 * - signal_alfven_left/right stay exactly zero: the RZ corner-EMF
 *   dissipation coefficients built on them then vanish identically and
 *   the corner E_theta reduces to the weighted average of the four ideal
 *   EMF states (central, no field dissipation).
 * The CGL channels are not populated (the input parser rejects
 * fluid_flux = central with ion_closure = cgl); the dual-energy U_i
 * channel is (centered pure advection). The electron channels
 * (U_e advection and the div(u_e) velocity flux) use the electron
 * velocity, which carries -J/(rho_q) under the Hall term: central is the
 * Hall-capable recast flux, the induction channels staying in the ION
 * frame with the edge Hall EMF added by the Ohm assembly.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE FaceFlux
central_flux (const CellState& left, const CellState& right,
              const amrex::Real bn_face, const int normal_direction,
              const FluxParameters& parameters) noexcept {
    const int tangent1 = (normal_direction + 1) % 3;
    const int tangent2 = (normal_direction + 2) % 3;

    // Rescaled field units b = B/sqrt(mu0), as in hlld_flux.
    const amrex::Real sqrt_mu0 = std::sqrt(parameters.mu0);
    const amrex::Real inv_sqrt_mu0 = 1.0 / sqrt_mu0;
    const amrex::Real bn = bn_face * inv_sqrt_mu0;
    const amrex::Real bn2 = bn * bn;

    const amrex::Real u_left = left.ion_velocity[normal_direction];
    const amrex::Real u_right = right.ion_velocity[normal_direction];
    const amrex::Real bt1_left = left.magnetic[tangent1] * inv_sqrt_mu0;
    const amrex::Real bt2_left = left.magnetic[tangent2] * inv_sqrt_mu0;
    const amrex::Real bt1_right = right.magnetic[tangent1] * inv_sqrt_mu0;
    const amrex::Real bt2_right = right.magnetic[tangent2] * inv_sqrt_mu0;
    const amrex::Real p_gas_left =
        left.ion_pressure + left.electron_pressure;
    const amrex::Real p_gas_right =
        right.ion_pressure + right.electron_pressure;
    const amrex::Real p_total_left =
        p_gas_left +
        0.5 * (bn2 + bt1_left * bt1_left + bt2_left * bt2_left);
    const amrex::Real p_total_right =
        p_gas_right +
        0.5 * (bn2 + bt1_right * bt1_right + bt2_right * bt2_right);

    FaceFlux flux;
    flux.mass = 0.5 * (left.momentum[normal_direction] +
                       right.momentum[normal_direction]);
    flux.momentum[normal_direction] =
        0.5 * ((left.momentum[normal_direction] * u_left + p_total_left -
                bn2) +
               (right.momentum[normal_direction] * u_right + p_total_right -
                bn2));
    flux.momentum[tangent1] =
        0.5 * ((left.momentum[normal_direction] *
                    left.ion_velocity[tangent1] -
                bn * bt1_left) +
               (right.momentum[normal_direction] *
                    right.ion_velocity[tangent1] -
                bn * bt1_right));
    flux.momentum[tangent2] =
        0.5 * ((left.momentum[normal_direction] *
                    left.ion_velocity[tangent2] -
                bn * bt2_left) +
               (right.momentum[normal_direction] *
                    right.ion_velocity[tangent2] -
                bn * bt2_right));
    // Electron thermal energy advects with the ELECTRON velocity (Hall:
    // u_e = u_i - J/(rho_q); reduces bitwise to the ion velocity when the
    // Hall term is off), consistent with the div(u_e) pressure work below.
    flux.electron_energy =
        0.5 * (left.electron_energy * left.electron_velocity_normal +
               right.electron_energy * right.electron_velocity_normal);
    flux.ion_energy = 0.5 * (u_left * (left.ion_energy + p_gas_left) +
                             u_right * (right.ion_energy + p_gas_right));
    if (parameters.dual_energy_closure) {
        // Dual-energy auxiliary internal energy: centered pure advection
        // (the q -> 0 limit of the hlld_flux star channel); the blended
        // PdV work and the viscous heating are pointwise RHS sources.
        flux.ion_internal_energy =
            0.5 * (u_left * left.ion_internal_energy +
                   u_right * right.ion_internal_energy);
    }
    flux.electron_velocity =
        0.5 * (left.electron_velocity_normal + right.electron_velocity_normal);
    flux.induction_t1 =
        sqrt_mu0 * 0.5 *
        ((u_left * bt1_left - left.ion_velocity[tangent1] * bn) +
         (u_right * bt1_right - right.ion_velocity[tangent1] * bn));
    flux.induction_t2 =
        sqrt_mu0 * 0.5 *
        ((u_left * bt2_left - left.ion_velocity[tangent2] * bn) +
         (u_right * bt2_right - right.ion_velocity[tangent2] * bn));

    // Centered Maxwell-stress state (physical units: b_i b_j = B_i B_j/mu0).
    flux.momentum_magnetic[normal_direction] =
        0.5 * ((0.5 * (bt1_left * bt1_left + bt2_left * bt2_left) -
                0.5 * bn2) +
               (0.5 * (bt1_right * bt1_right + bt2_right * bt2_right) -
                0.5 * bn2));
    flux.momentum_magnetic[tangent1] =
        0.5 * (-bn * bt1_left - bn * bt1_right);
    flux.momentum_magnetic[tangent2] =
        0.5 * (-bn * bt2_left - bn * bt2_right);

    // Smoothed Davis fast bounds, identical to hlld_flux.
    const amrex::Real signal_width =
        parameters.hlld_kappa_signal * (left.fast_speed + right.fast_speed);
    flux.signal_left = smooth_min(
        u_left - left.fast_speed, u_right - right.fast_speed, signal_width);
    flux.signal_right = smooth_max(
        u_left + left.fast_speed, u_right + right.fast_speed, signal_width);
    return flux;
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
           const amrex::Array4<const amrex::Real>& ion_energy,
           const amrex::Array4<const amrex::Real>& current,
           const amrex::Array4<const amrex::Real>& old_density,
           const amrex::Array4<const amrex::Real>& old_momentum,
           const amrex::Array4<const amrex::Real>& old_electron_energy,
           const amrex::Array4<const amrex::Real>& old_ion_energy,
           const int il,
           const int jl, const int kl, const int ir, const int jr, const int kr,
           const int normal_direction,
           const FluxParameters& parameters) noexcept {
    const CellState left =
        load_cell_state(density, momentum, electron_energy, ion_energy, current,
                        il, jl, kl, normal_direction, parameters);
    const CellState right =
        load_cell_state(density, momentum, electron_energy, ion_energy, current,
                        ir, jr, kr, normal_direction, parameters);
    FaceFlux flux;
    if (parameters.use_hllc) {
        flux = hllc_flux(left, right, normal_direction, parameters);
    } else {
        const amrex::Real alpha =
            parameters.use_rusanov
                ? std::max(left.wave_speed, right.wave_speed)
                : 0.0;
        flux.mass = 0.5 * (left.momentum[normal_direction] +
                           right.momentum[normal_direction] -
                           alpha * (right.density - left.density));
        for (int component = 0; component < 3; ++component) {
            const amrex::Real left_flux =
                left.momentum[normal_direction] *
                    left.ion_velocity[component] +
                (component == normal_direction
                     ? left.ion_pressure + left.electron_pressure
                     : 0.0);
            const amrex::Real right_flux =
                right.momentum[normal_direction] *
                    right.ion_velocity[component] +
                (component == normal_direction
                     ? right.ion_pressure + right.electron_pressure
                     : 0.0);
            flux.momentum[component] =
                0.5 * (left_flux + right_flux -
                       alpha * (right.momentum[component] -
                                left.momentum[component]));
        }
        flux.electron_energy =
            0.5 * (left.electron_energy * left.electron_velocity_normal +
                   right.electron_energy * right.electron_velocity_normal -
                   alpha * (right.electron_energy - left.electron_energy));
        flux.ion_energy =
            0.5 * (left.ion_velocity[normal_direction] *
                       (left.ion_energy + left.ion_pressure +
                        left.electron_pressure) +
                   right.ion_velocity[normal_direction] *
                       (right.ion_energy + right.ion_pressure +
                        right.electron_pressure) -
                   alpha * (right.ion_energy - left.ion_energy));
        flux.electron_velocity =
            0.5 *
            (left.electron_velocity_normal + right.electron_velocity_normal);
    }
    if (parameters.use_hllc || parameters.use_rusanov) {
        // Positivity guard: genuine evacuation can drain a donor cell to
        // its floor. HLLC exposes this because it preserves contacts;
        // Rusanov's diffusive backfill masks weak evacuation but a
        // sustained real drain (e.g. the wall-band relaxation when an
        // open free-space boundary replaces the conducting wall the
        // loaded equilibrium assumed) goes through it as well, so both
        // Riemann fluxes get the guards. Smoothly close the advected
        // mass and energy fluxes as the DONOR cell approaches its floor;
        // the face-based form keeps the scheme conservative, and only
        // outflow from a near-floor donor is closed (refill fluxes have
        // the healthy cell as donor and pass unlimited). The momentum
        // flux (including its pressure part) is left alone: momentum has
        // no floor, and near-floor cells are dust whose velocity the
        // vacuum drag already bounds.
        //
        // The limiters gate on the theta-EXTRAPOLATED end-of-step donor
        // values, (U^theta - (1-theta) U^n)/theta: the Newton
        // admissibility bound holds the END-OF-STEP state above the
        // floor, so a stage-value gate saturates too late for theta < 1
        // (the stage sits at (1-theta) U^n + theta*floor when the
        // end-of-step value reaches the floor -- far above it while the
        // cell drains from a healthy old state) and the residual can
        // still demand inadmissible states, zeroing the line search. At
        // theta = 1 the two gates coincide and this reduces to the
        // stage-value form.
        const amrex::Real inv_theta = 1.0 / parameters.theta;
        const amrex::Real ext = (1.0 - parameters.theta) * inv_theta;
        const bool mass_from_left = (flux.mass >= 0.0);
        const int im = mass_from_left ? il : ir;
        const int jm = mass_from_left ? jl : jr;
        const int km = mass_from_left ? kl : kr;
        const amrex::Real rho_end =
            density(im, jm, km) * (1.0 + ext) - old_density(im, jm, km) * ext;
        // With the halo pedestal active the mass drain gate anchors at the
        // pedestal instead of the positivity floor: outflow from a donor
        // closes smoothly as it approaches rho_ped, making the pedestal
        // band {rho >= rho_ped} dynamically invariant while the Newton
        // admissibility bound stays at the (far lower) positivity floor.
        flux.mass *= floor_outflow_limiter(
            rho_end,
            std::max(parameters.density_floor, parameters.halo_pedestal));
        const bool ue_from_left = (flux.electron_energy >= 0.0);
        const int ie = ue_from_left ? il : ir;
        const int je = ue_from_left ? jl : jr;
        const int ke = ue_from_left ? kl : kr;
        const amrex::Real ue_end =
            electron_energy(ie, je, ke) * (1.0 + ext) -
            old_electron_energy(ie, je, ke) * ext;
        flux.electron_energy *= floor_outflow_limiter(
            ue_end,
            std::max(parameters.electron_pressure_floor /
                         (parameters.gamma_e - 1.0),
                     parameters.halo_pedestal_electron_energy));
        if (parameters.total_energy_closure) {
            const bool ei_from_left = (flux.ion_energy >= 0.0);
            const int ii = ei_from_left ? il : ir;
            const int ji = ei_from_left ? jl : jr;
            const int ki = ei_from_left ? kl : kr;
            const amrex::Real ei_end =
                ion_energy(ii, ji, ki) * (1.0 + ext) -
                old_ion_energy(ii, ji, ki) * ext;
            amrex::Real kinetic_end = 0.0;
            for (int component = 0; component < 3; ++component) {
                const amrex::Real mom_end =
                    momentum(ii, ji, ki, component) * (1.0 + ext) -
                    old_momentum(ii, ji, ki, component) * ext;
                kinetic_end += mom_end * mom_end;
            }
            const amrex::Real rho_end_donor = std::max(
                density(ii, ji, ki) * (1.0 + ext) -
                    old_density(ii, ji, ki) * ext,
                parameters.density_floor);
            kinetic_end *= 0.5 / rho_end_donor;
            flux.ion_energy *= floor_outflow_limiter(
                ei_end - kinetic_end,
                std::max(parameters.ion_pressure_floor /
                             (parameters.gamma_i - 1.0),
                         parameters.halo_pedestal_ion_internal));
        }
    }
    return flux;
}
} // namespace theta_implicit_mhd

#endif
