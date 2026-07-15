#!/usr/bin/env python3
#
# --- Analysis script for the vacuum inductive-ramp test. The whole domain
# --- sits below the Ohm's-law density floor, so the run isolates the
# --- external vector-potential drive: (1) the volume-averaged B_z must track
# --- B(0) + f(t) dB (the discrete Faraday advance of the step-averaged
# --- inductive field reproduces the external flux exactly in vacuum cells);
# --- (2) the electric field must be the azimuthal inductive field only --
# --- spurious E_r/E_z would indicate a bad longitudinal projection or
# --- density-floor artifacts; (3) the dilute ions E x B compress with the
# --- flux: <r^2> B is conserved for the ensemble (zero spurious drift).

import sys

import numpy as np
import yt

# Deck constants (keep in sync with the inputs file)
B0 = 0.1
DB = 0.1
M_P = 1.67262192369e-27
Q_E = 1.602176634e-19
W_CI = Q_E * B0 / M_P
T_CI = 2.0 * np.pi / W_CI
T0_RAMP = 1.0 * T_CI
TAU_RAMP = 4.0 * T_CI


def f_ramp(t):
    return 1.0 / (1.0 + np.exp(5.0 * (1.0 - (t - T0_RAMP) * np.sqrt(2.0) / TAU_RAMP)))


def load_series():
    import glob

    dirs = sorted(glob.glob("diags/diag1" + "[0-9]" * 6))
    if not dirs:
        raise RuntimeError("no plotfiles found under diags/")
    return dirs


def volume_mean_bz(ds):
    # yt reads the RZ plotfile on a uniform (r, z) index grid; the physical
    # cell volume is proportional to the cell-center radius.
    ad = ds.all_data()
    r = ad["boxlib", "x"].value
    bz = ad["boxlib", "Bz"].value
    return (bz * r).sum() / r.sum()


def weighted_r2(ds):
    # In RZ output the recorded x scales with the particle radius, so
    # <x^2> B is conserved by the compression exactly when <r^2> B is.
    ad = ds.all_data()
    x = ad["ions", "particle_position_x"].value
    w = ad["ions", "particle_weight"].value
    return (w * x**2).sum() / w.sum()


def main():
    dirs = load_series()
    yt.set_log_level(50)

    # --- flux tracking at every dump
    print(f"{'step':>6s} {'t/t_ci':>7s} {'f(t)':>7s} {'<Bz>':>9s} {'expected':>9s} {'rel err':>9s}")
    ds0 = yt.load(dirs[0])
    b_start = volume_mean_bz(ds0)
    max_flux_err = 0.0
    for d in dirs:
        ds = yt.load(d)
        t = float(ds.current_time)
        bz = volume_mean_bz(ds)
        expected = b_start + (f_ramp(t) - f_ramp(0.0)) * DB
        err = abs(bz - expected) / B0
        max_flux_err = max(max_flux_err, err)
        print(
            f"{d[-5:]:>6s} {t / T_CI:7.3f} {f_ramp(t):7.4f} "
            f"{bz:9.5f} {expected:9.5f} {err:9.2e}"
        )

    # --- E-field cleanliness at the dump nearest the peak drive
    ds_mid = yt.load(dirs[len(dirs) // 2])
    ad = ds_mid.all_data()
    r = ad["boxlib", "x"].value
    interior = (r > 0.05) & (r < 0.45)
    et_max = np.abs(ad["boxlib", "Et"].value[interior]).max()
    er_rms = np.sqrt(np.mean(ad["boxlib", "Er"].value[interior] ** 2))
    ez_rms = np.sqrt(np.mean(ad["boxlib", "Ez"].value[interior] ** 2))
    print(
        f"\nmid-ramp fields: max|Et| = {et_max:.4e}  "
        f"RMS(Er) = {er_rms:.4e}  RMS(Ez) = {ez_rms:.4e}"
    )

    # --- betatron compression: <r^2> B conserved for the ion ensemble
    r2_0 = weighted_r2(ds0)
    ds1 = yt.load(dirs[-1])
    r2_1 = weighted_r2(ds1)
    b_end = volume_mean_bz(ds1)
    invariant = (r2_1 * b_end) / (r2_0 * b_start)
    print(
        f"betatron: <r^2>B end/start = {invariant:.4f} "
        f"(<r^2> ratio {r2_1 / r2_0:.4f}, B ratio {b_end / b_start:.4f})"
    )

    assert max_flux_err < 5.0e-3, f"flux tracking error {max_flux_err:.2e} > 5e-3"
    assert er_rms < 0.05 * et_max, f"spurious Er: {er_rms:.2e} vs Et {et_max:.2e}"
    assert ez_rms < 0.05 * et_max, f"spurious Ez: {ez_rms:.2e} vs Et {et_max:.2e}"
    assert abs(invariant - 1.0) < 0.05, f"<r^2>B not conserved: {invariant:.4f}"
    print("\nAll vacuum-ramp checks passed.")


if __name__ == "__main__":
    sys.exit(main())
