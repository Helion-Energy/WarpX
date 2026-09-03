#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Calibrated gates for offset-density advection.

``implicit_mhd.advection_density_offset_fraction`` is the flux-form
transliteration of the reference code's subtract / advect / re-add density advance
(ntb.f90:178-195, ``step_en``): the advective mass flux transports
the RECTIFIED PERTURBATION

    D(rho) = smooth-max(rho - rho_off, 0)

instead of ``rho``, with ``rho_off`` a fraction of the shared vacuum
reference density (their ``f_en_mn * en0``).  Above ``1.2 rho_off`` the
smoothing is exactly inactive, so ``D = rho - rho_off`` with no residue
surviving into the plasma; below it ``D -> 0``, so a sub-offset cell has
no advective mass flux at all and the halo is advectively static.

The deck is a cold, Mach-7 ENTROPY WAVE: uniform velocity, uniform total
pressure, density structure only, so the exact solution of the
continuous system is pure translation of the loaded profile at ``v0``.
The loaded profile is a two-cell-edged top hat six times the offset
riding on a featureless halo a fifth of the offset, and the run length
is chosen so the pulse translates exactly five cells.  The central flux
has no mass-channel dissipation, so those edges radiate a dispersive
wake that alternates sign cell to cell across the halo and advectively
EVACUATES it -- the production pathology (a halo driven onto its
positivity bound) in miniature.

Arms (all graded against the same knob-off null twin):

``off``
    The null twin itself.  Self-consistency only: exact mass
    conservation, plus a halo that is measurably drained AND whose
    error is genuinely grid-scale ripple (otherwise there is nothing
    for the ON arm to fix and the headline gates would be vacuous).

``on``
    Gate 1 -- exact conservation: the offset enters only as the density
    argument of a FACE flux, so the divergence still telescopes.  Total
    mass must be conserved to round-off and must match the null twin's
    to round-off.

    Gate 2 -- the point of the feature.  Four statements about the
    part of the halo the exact solution never moves: the error's
    high-wavenumber (2*dz) energy, its RMS and its peak must all drop
    substantially, and the halo must not be advectively evacuated.
    Because the exact answer in that window is "unchanged", the
    amplitude measures say the offset run is CLOSER TO THE TRUTH, not
    merely smoother.
    Meanwhile the resolved pulse must be transported no worse -- it
    must keep at least the null twin's peak amplitude and no larger a
    second moment, and land on the exact solution's centroid.  That
    pairing is what makes the metric ungameable by diffusion: added
    dissipation lowers the grid-scale content but also lowers the pulse
    peak and raises its second moment, so a diffusive impostor fails
    the signal half.

``zero``
    Knob at exactly 0: the early-return path, which must reproduce the
    null twin BITWISE.  This is the default-off contract.

``core``
    Gate 4, the core-class ``n >> rho_off`` no-harm requirement.  The
    knob is POSITIVE but with the offset four decades below every cell
    in the domain, on a QUIESCENT variant of the deck (three-cell pulse
    edges, so the halo never leaves its loaded value).  What the offset
    subtracts there is the background's own advective flux
    ``rho_off * u``, whose divergence is the background compression
    ``rho_off div(u)`` -- identically zero for this deck's uniform
    velocity, and ``O(rho_off/rho)`` in general.  The arm therefore
    asserts agreement with the null twin to solver precision rather
    than bit for bit, which is the honest statement of what the source
    scheme does.
"""

import sys

import numpy as np
import warpx_constants as constants
import yt

# ---------------------------------------------------------------- deck
NUMBER_OF_CELLS = 64
DOMAIN_LENGTH = 1.0
CELL_SIZE = DOMAIN_LENGTH / NUMBER_OF_CELLS
PEAK_NUMBER_DENSITY = 6.0e15
HALO_NUMBER_DENSITY = 2.0e14
REFERENCE_NUMBER_DENSITY = 1.0e16
OFFSET_FRACTION = 0.1
PULSE_LO = 0.15
PULSE_HI = 0.35
PULSE_EDGE = 2.0 * CELL_SIZE
# max_step * v0 * dt with the deck's Courant number 0.5.
TRANSLATION = 24 * 0.25 * CELL_SIZE

RHO_PEAK = PEAK_NUMBER_DENSITY * constants.proton_mass
RHO_HALO = HALO_NUMBER_DENSITY * constants.proton_mass
RHO_OFFSET = OFFSET_FRACTION * REFERENCE_NUMBER_DENSITY * constants.proton_mass

# Deck sanity: the halo is deeply sub-offset (fully rectified) and the
# pulse is above the exact-zero tail at 1.2 rho_off (bit-identical).
assert RHO_HALO < 0.5 * RHO_OFFSET
assert RHO_PEAK > 2.0 * RHO_OFFSET

Z = (np.arange(NUMBER_OF_CELLS) + 0.5) * CELL_SIZE


def loaded_profile(z):
    """The deck's smoothed top hat, evaluated at arbitrary positions."""
    return RHO_HALO + (RHO_PEAK - RHO_HALO) * 0.5 * (
        np.tanh((z - PULSE_LO) / PULSE_EDGE) - np.tanh((z - PULSE_HI) / PULSE_EDGE)
    )


def get_density(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return data["boxlib", "implicit_mhd_mass_density"].value.ravel().astype(np.float64)


def total_mass(density):
    return float(np.sum(density)) * CELL_SIZE


def halo_metrics(density, exact, loaded):
    """Halo error measures, on the part of the halo that never moves.

    The window is every cell that is within 1% of the halo value in BOTH
    the loaded state and the exact final state: the region the exact
    solution leaves flat for the whole run, so the correct answer there
    is "unchanged" and every deviation is numerical.  (Requiring the
    LOADED value too is what keeps the pulse's own sub-offset tail out
    of the window: the offset deliberately does not transport that
    tail, and grading it against a moving exact solution would be
    grading the feature for doing its job.)  The measured field is the
    ERROR ``e = (rho - rho_exact)/rho_halo``, identically zero for a
    perfect scheme.

    ``rms``, ``peak``
        Amplitude of the halo error.
    ``grid_scale``
        HIGH-WAVENUMBER ENERGY: the RMS of the error's second difference
        ``e_i - (e_(i-1) + e_(i+1))/2``, i.e. the discrete amplitude of
        the 2*dz mode, blind to any smooth drift.  Computed locally in
        the halo rather than spectrally over the whole domain -- a
        global FFT would be dominated by the deliberately steep pulse
        edges rather than by the halo.
    ``sign_changes``
        Cell-to-cell sign flips of the error: the same statement as a
        pure COUNT, which no rescaling can move.  Dispersive ripple
        alternates every cell; a smooth error does not.  Reported, not
        gated: the gate is the grid-scale energy above.
    ``undershoot``
        The lowest density anywhere in the domain, in units of the halo
        value.  The exact solution never goes below 1.

    Measures that need neighbours use the eroded window (cell and both
    neighbours inside); the grid is periodic, so the rolls wrap.
    """
    flat = (exact < 1.01 * RHO_HALO) & (loaded < 1.01 * RHO_HALO)
    interior = flat & np.roll(flat, 1) & np.roll(flat, -1)
    error = (density - exact) / RHO_HALO
    curvature = np.abs(error - 0.5 * (np.roll(error, 1) + np.roll(error, -1)))
    return (
        float(np.sqrt(np.mean(error[flat] ** 2))),
        float(np.abs(error[flat]).max()),
        float(np.sqrt(np.mean(curvature[interior] ** 2))),
        int(np.sum(np.diff(np.sign(error[flat])) != 0)),
        float(density.min() / RHO_HALO),
    )


PULSE_THRESHOLD = 0.5 * RHO_PEAK


def pulse_moments(density, exact):
    """Peak, mass, centroid and second moment of the resolved pulse.

    The window is the pulse CORE -- every cell whose exact final density
    is at least half the peak, i.e. three times the offset, well above
    the exact-zero tail of the rectifier.  Weighting by the excess above
    the window threshold keeps the moments insensitive to the shoulders.
    """
    window = exact > PULSE_THRESHOLD
    weight = np.maximum(density[window] - PULSE_THRESHOLD, 0.0)
    positions = Z[window]
    mass = float(weight.sum()) * CELL_SIZE
    centroid = float((weight * positions).sum() / weight.sum())
    variance = float((weight * (positions - centroid) ** 2).sum() / weight.sum())
    return float(density.max()), mass, centroid, variance


mode = sys.argv[1]
initial_density = get_density(sys.argv[2])
final_density = get_density(sys.argv[3])

initial_mass = total_mass(initial_density)
final_mass = total_mass(final_density)
mass_drift = abs(final_mass - initial_mass) / initial_mass
print(f"[{mode}] mass drift              = {mass_drift:.3e}")

# GATE 1 (every arm): the mass flux stays a face flux, so the divergence
# telescopes and a periodic domain conserves mass to round-off.  The
# subtract / re-add cancellation is structural (a linearity identity),
# not a numerical near-cancellation of large numbers.
assert mass_drift < 1.0e-12, f"mass drift {mass_drift:.3e} is not round-off"

if mode in ("zero", "core"):
    # No analytic profile is referenced here, so these arms are free to
    # run whichever deck variant suits them (the core arm uses the
    # quiescent twin).
    twin_start = get_density(sys.argv[4])
    twin_end = get_density(sys.argv[5])
    np.testing.assert_array_equal(initial_density, twin_start)
    if mode == "zero":
        # GATE 3: the early-return path must reproduce the null twin
        # BITWISE -- not merely close.  The default-off contract.
        np.testing.assert_array_equal(final_density, twin_end)
        print("[zero] bitwise identical to the null twin")
    else:
        # GATE 4: core class, n >> rho_off everywhere.  What the offset
        # removes there is the background's own advective flux, whose
        # divergence is the background compression rho_off div(u):
        # identically zero for this deck's uniform velocity, and
        # O(rho_off/rho) in general.  So the requirement is agreement to
        # solver precision, and the offset here is four decades below
        # every cell, i.e. a part in 1e4 of the smallest density.
        deviation = float(np.max(np.abs(final_density - twin_end)) / np.max(twin_end))
        print(f"[core] max |ON - OFF| / peak = {deviation:.3e}")
        assert deviation < 1.0e-9, (
            f"core-class deviation {deviation:.3e} is not solver precision"
        )
    sys.exit(0)

# The loaded state must be the deck's analytic profile.
np.testing.assert_allclose(initial_density, loaded_profile(Z), rtol=1.0e-12)

loaded = loaded_profile(Z)
exact_final = loaded_profile((Z - TRANSLATION) % DOMAIN_LENGTH)
rms, peak, grid_scale, sign_changes, undershoot = halo_metrics(
    final_density, exact_final, loaded
)
print(f"[{mode}] halo RMS error          = {rms:.6e}")
print(f"[{mode}] halo peak error         = {peak:.6e}")
print(f"[{mode}] halo grid-scale energy  = {grid_scale:.6e}")
print(f"[{mode}] halo error sign changes = {sign_changes}")
print(f"[{mode}] min density / rho_halo  = {undershoot:.6f}")

if mode == "off":
    # The null twin must actually be sick, otherwise the headline gates
    # below compare two healthy runs and prove nothing: the centred flux
    # must have filled the flat halo with sign-alternating error and
    # drained it below its loaded value.
    assert sign_changes >= 8, (
        f"null twin halo error only changes sign {sign_changes} times -- "
        "the deck no longer produces dispersive ripple"
    )
    assert undershoot < 0.99, (
        f"null twin min density {undershoot:.6f} rho_halo never dipped -- "
        "the deck no longer exercises the feature"
    )
    sys.exit(0)

twin_initial = get_density(sys.argv[4])
twin_final = get_density(sys.argv[5])

# Same loaded state in both twins.
np.testing.assert_array_equal(initial_density, twin_initial)

assert mode == "on", f"unknown mode {mode}"

# GATE 1, second half: the offset run's total mass matches the null
# twin's to round-off (the same conserved quantity, differently
# transported).
twin_mass = total_mass(twin_final)
twin_drift = abs(twin_mass - initial_mass) / initial_mass
cross_drift = abs(final_mass - twin_mass) / initial_mass
print(f"[on]  null-twin mass drift    = {twin_drift:.3e}")
print(f"[on]  ON vs OFF mass mismatch = {cross_drift:.3e}")
assert twin_drift < 1.0e-12
assert cross_drift < 1.0e-12

twin_rms, twin_peak, twin_grid, twin_signs, twin_undershoot = halo_metrics(
    twin_final, exact_final, loaded
)
print(f"[on]  halo RMS        ON / OFF = {rms:.6e} / {twin_rms:.6e}")
print(f"[on]  halo peak error ON / OFF = {peak:.6e} / {twin_peak:.6e}")
print(f"[on]  grid-scale enrg ON / OFF = {grid_scale:.6e} / {twin_grid:.6e}")
print(f"[on]  sign changes    ON / OFF = {sign_changes} / {twin_signs}")
print(f"[on]  min rho/rho_halo ON/OFF  = {undershoot:.6f} / {twin_undershoot:.6f}")

# GATE 2a: the HIGH-WAVENUMBER ENERGY of the halo error -- the
# dispersive-ripple signature of a centred flux with no mass-channel
# dissipation, which is what "noise in the low density region" means
# here.  The offset's advective mass flux vanishes below rho_off, so the
# halo has nothing left to ripple with.
assert grid_scale < 0.6 * twin_grid, (
    f"halo grid-scale energy {grid_scale:.3e} is not reduced against the "
    f"null twin's {twin_grid:.3e}"
)

# GATE 2b: the halo error itself, in RMS and in peak.  The exact answer
# in this window is "unchanged", so these say the offset run is simply
# CLOSER TO THE TRUTH, not merely smoother.
assert rms < 0.6 * twin_rms, (
    f"halo RMS error {rms:.3e} is not reduced against the null twin's {twin_rms:.3e}"
)
assert peak < 0.5 * twin_peak, (
    f"halo peak error {peak:.3e} is not reduced against the null twin's {twin_peak:.3e}"
)

# GATE 2c: the halo is not advectively evacuated.  This is the
# conservative image of the source scheme's invariant n >= f_en_mn en0:
# a sub-offset cell's advective mass flux vanishes, so the wake cannot
# drain the halo.  The null twin's does.
assert undershoot > twin_undershoot, (
    f"the offset run's min density {undershoot:.6f} rho_halo is not above "
    f"the null twin's {twin_undershoot:.6f}"
)
assert undershoot > 0.999, (
    f"the offset run drained the halo to {undershoot:.6f} rho_halo; with "
    "no advective mass flux below the offset it should not have moved"
)

# GATE 2c: NOT DIFFUSION.  The resolved pulse -- every cell of which
# sits above the rectifier's exact-zero tail -- must be transported no
# worse than in the null twin.  Any scheme that bought the reductions
# above with added dissipation would lower the peak, raise the second
# moment, and displace the centroid here.
peak_on, mass_on, centroid_on, variance_on = pulse_moments(final_density, exact_final)
peak_off, mass_off, centroid_off, variance_off = pulse_moments(twin_final, exact_final)
peak_ex, mass_ex, centroid_ex, variance_ex = pulse_moments(exact_final, exact_final)
print(
    f"[on]  pulse peak     ON/OFF/ex = {peak_on:.6e} / {peak_off:.6e} / {peak_ex:.6e}"
)
print(
    f"[on]  pulse mass     ON/OFF/ex = {mass_on:.6e} / {mass_off:.6e} / {mass_ex:.6e}"
)
print(
    f"[on]  pulse centroid ON/OFF/ex = {centroid_on:.6e} / {centroid_off:.6e}"
    f" / {centroid_ex:.6e}"
)
print(
    f"[on]  pulse variance ON/OFF/ex = {variance_on:.6e} / {variance_off:.6e}"
    f" / {variance_ex:.6e}"
)
assert peak_on > 0.995 * peak_off, "the offset damped the resolved pulse peak"
assert variance_on < 1.05 * variance_off, "the offset spread the resolved pulse"
np.testing.assert_allclose(mass_on, mass_off, rtol=5.0e-2)
# ... and the pulse must still be where the exact solution puts it.
np.testing.assert_allclose(centroid_on, centroid_ex, rtol=5.0e-3)
np.testing.assert_allclose(centroid_off, centroid_ex, rtol=5.0e-3)

print("[on]  all offset-density advection gates passed")
