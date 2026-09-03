#!/usr/bin/env python3
#
# --- Analysis for the anisotropic electron-viscosity manufactured solution
# --- (see inputs_test_2d_ohm_solver_electron_viscosity_picmi.py for the
# --- fixture and its derivation).
# ---
# --- Active arm (tilt = 45 deg): the field-aligned strain carries the whole
# --- rate of strain, so the DOMAIN-MEAN electron temperature must rise at the
# --- Braginskii rate
# ---
# ---     <dTe/dt> = 0.25 sin^2(2 theta) (m_e/kB) nu_par(Te) (U k)^2
# ---
# --- (the 0.25 is (gamma-1) * 3/4 * <sin^2(k x)> = (2/3)(3/4)(1/2)), integrated
# --- exactly in time because nu_par ~ Te^{5/2} steepens the ramp as the plasma
# --- heats. Both the mean rise and the sin^2(k x) profile are checked.
# ---
# --- Null arm (tilt = 0): B_x is zero everywhere, so b.W.b vanishes
# --- identically and a correct PARALLEL viscosity deposits no heat at all.
# --- This is the anisotropy gate: an isotropic viscosity, or one that
# --- contracts the strain with the wrong projector, heats here.

import argparse
import glob
import sys

import numpy as np
import yt

# Physical constants (SI, CODATA 2018 -- matched to WarpX's PhysConst).
Q_E = 1.602176634e-19
KB = 1.380649e-23
M_E = 9.1093837139e-31
MU0 = 1.2566370612685e-06
EPS0 = 8.8541878188e-12

# Deck constants (keep in sync with the inputs file).
TE0_EV = 100.0
N0 = 1.0e19
B1 = 0.005
LX = 0.3
N_WAVE = 2
COULOMB_LOG = 10.0

K = 2.0 * np.pi * N_WAVE / LX
U = B1 * K / (MU0 * Q_E * N0)


def braginskii_nu_par(te_eV):
    """Braginskii (1965) field-aligned kinematic electron viscosity [m^2/s]."""
    te_J = te_eV * Q_E
    four_pi_eps0 = 4.0 * np.pi * EPS0
    tau_e = (
        3.0
        * np.sqrt(M_E)
        * te_J**1.5
        * four_pi_eps0**2
        / (4.0 * np.sqrt(2.0 * np.pi) * N0 * Q_E**4 * COULOMB_LOG)
    )
    return 0.73 * te_J * tau_e / M_E


def sinc(u):
    return np.sin(u) / u if u != 0.0 else 1.0


def discrete_gain(dx):
    """Amplitude gain of the DISCRETE d u_ez/dx relative to the analytic one.

    Two centred differences stand between B and grad u_e, each contributing
    its own sinc factor at wavenumber k:
      * the Yee curl, J_z = (B_y[i+1/2] - B_y[i-1/2])/dx  ->  sinc(k dx / 2);
      * the nodal gradient, (u[i+1] - u[i-1])/(2 dx)      ->  sinc(k dx).
    The heating is quadratic in grad u_e, so the rate carries the SQUARE of
    this. VERIFIED against the code over a 4x resolution sweep (k dx = 0.393,
    0.196, 0.098): dividing the measured rate by this gain flattens it to
    0.9875, 0.9872, 0.9871 -- 4e-4 spread, i.e. the factor is exact and the
    remaining 1.3% is the fixture drift below, not a discretization error.
    """
    return sinc(K * dx / 2.0) * sinc(K * dx)


def predicted_te_K(te0_K, t_end, sin2_2theta, gain):
    """Exact solution of dTe/dt = A Te^{5/2}, the closed form with the
    Braginskii temperature scaling nu_par ~ Te^{5/2} folded in.

    A is fixed by matching the rate at Te0:
        dTe/dt|_0 = 0.25 sin^2(2 theta) (m_e/kB) nu_par(Te0) (U k gain)^2
    with <sin^2(k x)> = 1/2 already absorbed into the 0.25 prefactor.
    """
    rate0 = (
        0.25
        * sin2_2theta
        * (M_E / KB)
        * braginskii_nu_par(TE0_EV)
        * (U * K * gain) ** 2
    )
    if rate0 <= 0.0:
        return te0_K
    a = rate0 / te0_K**2.5
    # d(Te^{-3/2})/dt = -1.5 A  ->  Te(t) = [Te0^{-3/2} - 1.5 A t]^{-2/3}
    return (te0_K**-1.5 - 1.5 * a * t_end) ** (-2.0 / 3.0)


def load_te(path):
    """Return (x cell centres, z-averaged Te [K], time [s], dx [m])."""
    ds = yt.load(path)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    # yt presents the 2D openPMD mesh as (z, x, 1); the fixture is uniform in
    # z, so collapse it and keep the x profile.
    te = np.squeeze(np.asarray(grid["Te"]))
    te_x = te.mean(axis=0)
    nx = te_x.size
    x_lo = float(ds.domain_left_edge[1])
    dx = (float(ds.domain_right_edge[1]) - x_lo) / nx
    x = x_lo + dx * (np.arange(nx) + 0.5)
    return x, te_x, float(ds.current_time), dx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--null",
        action="store_true",
        help="null arm: assert the parallel channel deposits no heat",
    )
    # Measured agreement on this deck, once the discretization gain is taken
    # out: 1.3% low at every resolution, which is the whistler rotation of the
    # perturbation over the run (resolution independent, as expected -- it
    # depends only on omega_w t). 5% leaves ~4x margin on that while still
    # rejecting an isotropic contraction (which would read 33% high) and any
    # coefficient or units slip.
    parser.add_argument("--tol", type=float, default=0.05)
    # The null is structural, not approximate: B_x is set to zero everywhere,
    # so the field-aligned strain S = 2 b_x b_z G_xz vanishes and only the
    # ~1e-10-suppressed cross-field channel (plus whatever B_x the dynamics
    # generates over 20 steps) can heat. Measured |dTe|/Te0 = 2.9e-9, so 1e-5
    # leaves 3.5 decades of margin while sitting 3 decades BELOW what a
    # projector slip would produce -- an isotropic contraction heats this arm
    # at 1.3e-2, i.e. the full active-arm rate.
    parser.add_argument(
        "--null-tol",
        type=float,
        default=1.0e-5,
        help="max |dTe|/Te0 accepted on the null arm",
    )
    args = parser.parse_args()

    files = sorted(glob.glob("diags/field_diags/*.h5"))
    if not files:
        files = sorted(glob.glob("diags/field_diags*"))
    assert files, "no field diagnostic output found"

    _, te_first, t0, dx = load_te(files[0])
    x1, te_last, t1, _ = load_te(files[-1])
    assert t1 > t0, "need two distinct diagnostic times"

    te0_K = TE0_EV * Q_E / KB
    d_te = te_last.mean() - te_first.mean()
    print(f"[viscosity] t = {t0:.4e} -> {t1:.4e} s")
    print(f"[viscosity] <Te> = {te_first.mean():.6e} -> {te_last.mean():.6e} K")
    print(f"[viscosity] <dTe>/Te0 = {d_te / te0_K:.6e}")

    if args.null:
        # b_x = 0 everywhere, so S = b.W.b vanishes identically: the
        # field-aligned channel has nothing to dissipate. Anything above the
        # tolerance means the strain is being contracted with the wrong
        # projector (an isotropic viscosity heats at full rate here).
        rel = abs(d_te) / te0_K
        print(f"[viscosity] null arm |dTe|/Te0 = {rel:.3e} (tol {args.null_tol:.3e})")
        assert rel < args.null_tol, (
            f"parallel viscosity deposited heat where the field-aligned strain "
            f"is identically zero: |dTe|/Te0 = {rel:.3e}"
        )
        print("[viscosity] PASS (anisotropy null)")
        return

    # --- Active arm -------------------------------------------------------
    gain = discrete_gain(dx)
    te_pred = predicted_te_K(te0_K, t1 - t0, sin2_2theta=1.0, gain=gain)
    d_te_pred = te_pred - te0_K
    rel_err = abs(d_te - d_te_pred) / d_te_pred
    print(f"[viscosity] discrete grad u_e gain = {gain:.6f} (k dx = {K * dx:.4f})")
    print(
        f"[viscosity] dTe measured = {d_te:.6e} K, predicted = {d_te_pred:.6e} K, "
        f"rel err = {rel_err:.4f} (tol {args.tol:.3f})"
    )
    assert d_te > 0.0, "viscous heating must be positive-definite"
    assert rel_err < args.tol, (
        f"viscous heating rate off by {rel_err * 100:.1f}%: measured "
        f"{d_te:.4e} K vs Braginskii closed form {d_te_pred:.4e} K"
    )

    # Profile shape: Q_nu ~ sin^2(k x). Correlating against the analytic shape
    # is insensitive to the overall amplitude and to the cell-centring of the
    # diagnostic, and catches a heating term that is active but misplaced.
    profile = te_last - te_first
    shape = np.sin(K * x1) ** 2
    profile -= profile.mean()
    shape -= shape.mean()
    corr = float(
        np.dot(profile, shape)
        / np.sqrt(np.dot(profile, profile) * np.dot(shape, shape))
    )
    print(f"[viscosity] sin^2(k x) profile correlation = {corr:.4f}")
    assert corr > 0.95, f"viscous heating profile does not track sin^2(k x): {corr:.4f}"

    print("[viscosity] PASS")


if __name__ == "__main__":
    sys.exit(main())
