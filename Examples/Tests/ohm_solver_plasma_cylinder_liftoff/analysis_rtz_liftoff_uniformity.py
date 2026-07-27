#!/usr/bin/env python3
"""Azimuthal-uniformity and implosion-trajectory analysis for the RTZ liftoff run.

Reads the openPMD (h5) field diagnostics of
inputs_test_rtz_ohm_solver_plasma_cylinder_liftoff_picmi.py and reports, per
output time:

  * the implosion trajectory: density-weighted mean radius and peak-|rho| radius;
  * theta-uniformity of rho and Bz: per-radius rms/mean over theta, aggregated
    over the plasma region (rho > 10% of its instantaneous max);
  * optionally (--m M) the azimuthal mode-M relative amplitude of rho and Bz
    from the theta FFT, for seeded-perturbation growth studies.

On the rotationally symmetric RTZ grid an unperturbed implosion must stay
theta-uniform to particle noise; --assert-uniform TOL turns that into a hard
check (used for validation runs, not CI).

Usage:
  python analysis_rtz_liftoff_uniformity.py [--diags diags] [--m 4]
      [--assert-uniform 0.2] [--png liftoff_uniformity.png]
"""

import argparse
import glob
import os
import sys

import numpy as np

import openpmd_api as io


def cylindrical_fields(it):
    """Return (rho, Bz) as (nr, ntheta) arrays averaged over z, plus r axis.

    RTZ openPMD output is a 3D block with axis labels (r, theta, z) [or
    (x, y, z) on older writers]; components of B are named z/t/r or with
    Cartesian fallbacks. Introspect rather than assume.
    """
    rho_mesh = it.meshes["rho"]
    rho_comp = rho_mesh[io.Mesh_Record_Component.SCALAR]
    rho = rho_comp.load_chunk()

    b_mesh = it.meshes["B"]
    bz_name = None
    for cand in ("z", "Bz"):
        if cand in b_mesh:
            bz_name = cand
            break
    if bz_name is None:
        raise RuntimeError(f"no axial component among {list(b_mesh)}")
    bz = b_mesh[bz_name].load_chunk()
    it.series_flush()

    labels = list(rho_mesh.axis_labels)
    gspace = rho_mesh.grid_spacing
    goffset = rho_mesh.grid_global_offset

    # openPMD store order matches axis_labels; identify the (r, theta, z) axes
    def axis(name, fallback):
        return labels.index(name) if name in labels else labels.index(fallback)

    ir = axis("r", "x")
    itheta = axis("theta", "y")
    iz = axis("z", "z")

    rho = np.moveaxis(rho, (ir, itheta, iz), (0, 1, 2))
    bz = np.moveaxis(bz, (ir, itheta, iz), (0, 1, 2))
    nr = rho.shape[0]
    r = goffset[ir] + gspace[ir] * (np.arange(nr) + 0.5)

    return rho.mean(axis=2), bz.mean(axis=2), r


def ring_metrics(f2d, r, plasma_mask):
    """Per-ring theta rms/|mean|, aggregated over masked rings."""
    mean = f2d.mean(axis=1)
    rms = f2d.std(axis=1)
    denom = np.maximum(np.abs(mean), 1e-30)
    rel = np.where(plasma_mask, rms / denom, 0.0)
    if plasma_mask.any():
        return float(np.max(rel[plasma_mask])), float(np.median(rel[plasma_mask]))
    return 0.0, 0.0


def mode_amplitude(f2d, m, plasma_mask):
    """Density-weighted relative amplitude of azimuthal mode m."""
    fhat = np.fft.rfft(f2d, axis=1)
    ntheta = f2d.shape[1]
    if m >= fhat.shape[1]:
        return 0.0
    amp = 2.0 * np.abs(fhat[:, m]) / ntheta
    base = np.maximum(np.abs(fhat[:, 0].real) / ntheta, 1e-30)
    rel = amp / base
    if plasma_mask.any():
        return float(np.max(rel[plasma_mask]))
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diags", default="diags", help="diagnostics directory")
    ap.add_argument("--m", type=int, default=0, help="track azimuthal mode m")
    ap.add_argument("--assert-uniform", type=float, default=None,
                    help="fail if the final max rho theta-nonuniformity "
                    "exceeds this")
    ap.add_argument("--png", default="liftoff_uniformity.png")
    args = ap.parse_args()

    h5s = sorted(glob.glob(os.path.join(args.diags, "field_diag", "*.h5")))
    if not h5s:
        # openPMD file-based series (one file per iteration under field_diag)
        raise SystemExit(f"no h5 outputs under {args.diags}/field_diag")
    import re

    base = os.path.basename(h5s[0])
    pattern = re.sub(r"\d+\.h5$", "%T.h5", base)
    series = io.Series(
        os.path.join(args.diags, "field_diag", pattern), io.Access.read_only
    )

    times, r_mean, r_peak = [], [], []
    rho_nonunif_max, rho_nonunif_med, bz_nonunif_max = [], [], []
    mode_rho, mode_bz = [], []

    for idx in series.iterations:
        it = series.iterations[idx]
        rho2d, bz2d, r = cylindrical_fields(it)
        arho = np.abs(rho2d)
        ring_rho = arho.mean(axis=1)
        mask = ring_rho > 0.1 * ring_rho.max()

        w = (arho * r[:, None]).sum(axis=1)
        times.append(it.time)
        r_mean.append(float((r * w).sum() / max(w.sum(), 1e-30)))
        r_peak.append(float(r[np.argmax(ring_rho)]))

        mx, med = ring_metrics(rho2d, r, mask)
        rho_nonunif_max.append(mx)
        rho_nonunif_med.append(med)
        bmx, _ = ring_metrics(bz2d, r, mask)
        bz_nonunif_max.append(bmx)

        if args.m > 0:
            mode_rho.append(mode_amplitude(rho2d, args.m, mask))
            mode_bz.append(mode_amplitude(bz2d, args.m, mask))

    times = np.asarray(times)
    print(f"{'time (us)':>10} {'r_mean (m)':>11} {'r_peak (m)':>11} "
          f"{'drho/rho max':>13} {'drho/rho med':>13} {'dBz/Bz max':>11}"
          + (f" {'mode-%d rho' % args.m:>11} {'mode-%d Bz' % args.m:>11}"
             if args.m > 0 else ""))
    for i, t in enumerate(times):
        row = (f"{t*1e6:10.3f} {r_mean[i]:11.4f} {r_peak[i]:11.4f} "
               f"{rho_nonunif_max[i]:13.4e} {rho_nonunif_med[i]:13.4e} "
               f"{bz_nonunif_max[i]:11.4e}")
        if args.m > 0:
            row += f" {mode_rho[i]:11.4e} {mode_bz[i]:11.4e}"
        print(row)

    implosion = r_peak[0] - r_peak[-1]
    print(f"\npeak-density radius moved {implosion:+.4f} m "
          f"({r_peak[0]:.4f} -> {r_peak[-1]:.4f})")
    print(f"final rho theta-nonuniformity: max {rho_nonunif_max[-1]:.3e}, "
          f"median {rho_nonunif_med[-1]:.3e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3 if args.m > 0 else 2,
                                figsize=(13 if args.m > 0 else 9, 3.6))
        axs[0].plot(times * 1e6, r_peak, label="peak-density radius")
        axs[0].plot(times * 1e6, r_mean, label="density-weighted radius")
        axs[0].set_xlabel("t (us)"); axs[0].set_ylabel("r (m)")
        axs[0].set_title("implosion trajectory"); axs[0].legend()
        axs[1].semilogy(times * 1e6, rho_nonunif_max, label="rho max")
        axs[1].semilogy(times * 1e6, rho_nonunif_med, label="rho median")
        axs[1].semilogy(times * 1e6, bz_nonunif_max, label="Bz max")
        axs[1].set_xlabel("t (us)"); axs[1].set_ylabel("theta rms/mean")
        axs[1].set_title("azimuthal nonuniformity"); axs[1].legend()
        if args.m > 0:
            axs[2].semilogy(times * 1e6, mode_rho, label=f"rho m={args.m}")
            axs[2].semilogy(times * 1e6, mode_bz, label=f"Bz m={args.m}")
            axs[2].set_xlabel("t (us)"); axs[2].set_ylabel("relative amplitude")
            axs[2].set_title("seeded mode"); axs[2].legend()
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print(f"wrote {args.png}")
    except ImportError:
        pass

    if args.assert_uniform is not None:
        assert rho_nonunif_max[-1] < args.assert_uniform, (
            f"theta nonuniformity {rho_nonunif_max[-1]:.3e} exceeds "
            f"{args.assert_uniform}")
        print("uniformity assertion PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
