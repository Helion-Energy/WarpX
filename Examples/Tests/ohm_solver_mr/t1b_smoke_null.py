#!/usr/bin/env python3
"""T1b E3 bulk-physics null: the seam band must be seam-local.

Runs the T0.3-lite quiet-drift smoke (t_smoke_inputs_2d_uniform, 500 steps,
production-like dt) twice -- band off and band on (both eta and eta_h
seam targets, W = 6, values referenced to this deck's coarse grid) -- and
compares
  (1) sigma(B)/B0 on the level-0 covering grid,
  (2) sigma(B)/B0 on the level-1 patch interior (10 edge cells excluded),
  (3) total particle energy drift (part_energy reduced diagnostic).
Writes t1b_null_sigma.png and prints a table.

Usage:  python t1b_smoke_null.py [--nprocs 4] [--skip-run]
"""

import argparse
import glob
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "t1b_runs")
sys.path.insert(0, HERE)
import t1_pltreader  # noqa: E402

B0 = 0.1  # T, deck background Bz

# this deck's coarse grid: dz = 0.073088.../256, w_ci = q B0 / m_ion
W_CI = 1.602176634e-19 * B0 / 9.1093837139e-29
DZ_C = 0.073088442495424816 / 256.0
K_NC = np.pi / DZ_C
ETA_H_STAR = 1.25663706212e-06 * W_CI / K_NC**4
BAND_ARGS = [
    "hybrid_pic_model.mr_seam_band_width=6",
    "hybrid_pic_model.mr_seam_eta=3e-7",
    f"hybrid_pic_model.mr_seam_eta_h={0.2 * ETA_H_STAR:.6e}",
]


def run(name, extra, nprocs):
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    exe = os.path.abspath(
        os.path.join(HERE, "..", "..", "..", "build", "bin", "warpx.2d")
    )
    deck = os.path.join(HERE, "t_smoke_inputs_2d_uniform")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    cmd = ["mpirun", "-np", str(nprocs), "--bind-to", "none", exe, deck] + extra
    print("[smoke-null]", name, ":", " ".join(cmd))
    with open(os.path.join(out, "run.log"), "w") as log:
        rc = subprocess.run(
            cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    assert rc == 0, f"{name} failed rc={rc}"


def series(name):
    """(t, sigma_lev0, sigma_lev1_interior) from every plotfile."""
    ts, s0, s1 = [], [], []
    for p in sorted(glob.glob(os.path.join(RUNS, name, "diags", "diag1??????"))):
        time, levels = t1_pltreader.read_plotfile(p, ["Bx", "By", "Bz"])
        a = levels[0]["arr"]
        fl = np.sqrt(np.mean(a["Bx"] ** 2 + a["By"] ** 2 + (a["Bz"] - B0) ** 2))
        ts.append(time)
        s0.append(fl / B0)
        b = levels[1]["arr"]
        sl = np.s_[10:-10, 10:-10]  # exclude band + margin
        fl1 = np.sqrt(
            np.mean(b["Bx"][sl] ** 2 + b["By"][sl] ** 2 + (b["Bz"][sl] - B0) ** 2)
        )
        s1.append(fl1 / B0)
    return np.array(ts), np.array(s0), np.array(s1)


def part_energy(name):
    a = np.loadtxt(
        os.path.join(RUNS, name, "diags/reducedfiles/part_energy.txt"), skiprows=1
    )
    return a[:, 1], a[:, 2]  # time, total particle energy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    if not args.skip_run:
        run("smoke_null_off", [], args.nprocs)
        run("smoke_null_on", BAND_ARGS, args.nprocs)

    t_off, s0_off, s1_off = series("smoke_null_off")
    t_on, s0_on, s1_on = series("smoke_null_on")
    tp_off, ep_off = part_energy("smoke_null_off")
    tp_on, ep_on = part_energy("smoke_null_on")

    print("\n# time(ns)  sig0_off  sig0_on  ratio0   sig1_off  sig1_on  ratio1")
    for i in range(len(t_off)):
        r0 = s0_on[i] / s0_off[i] if s0_off[i] > 0 else np.nan
        r1 = s1_on[i] / s1_off[i] if s1_off[i] > 0 else np.nan
        print(
            f"{t_off[i] * 1e9:8.3f}  {s0_off[i]:.6e} {s0_on[i]:.6e} "
            f"{r0:6.3f}   {s1_off[i]:.6e} {s1_on[i]:.6e} {r1:6.3f}"
        )
    heat_off = (ep_off[-1] - ep_off[0]) / ep_off[0]
    heat_on = (ep_on[-1] - ep_on[0]) / ep_on[0]
    print(
        f"\nfinal sigma(B)/B0 lev0: off {s0_off[-1]:.4e}  on {s0_on[-1]:.4e}"
        f"  ratio {s0_on[-1] / s0_off[-1]:.3f}"
    )
    print(
        f"final sigma(B)/B0 lev1 interior: off {s1_off[-1]:.4e}  "
        f"on {s1_on[-1]:.4e}  ratio {s1_on[-1] / s1_off[-1]:.3f}"
    )
    print(f"particle-energy drift: off {heat_off:+.3e}  on {heat_on:+.3e}")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(t_off * 1e9, s0_off, color="#c95d20", lw=2, label="band off")
    axs[0].plot(t_on * 1e9, s0_on, color="#3d65f5", lw=2, ls="--", label="band on")
    axs[0].set_ylabel(r"$\sigma(B)/B_0$ (level-0 view)")
    axs[1].plot(t_off * 1e9, s1_off, color="#c95d20", lw=2, label="band off")
    axs[1].plot(t_on * 1e9, s1_on, color="#3d65f5", lw=2, ls="--", label="band on")
    axs[1].set_ylabel(r"$\sigma(B)/B_0$ (level-1 patch interior)")
    for ax in axs:
        ax.set_xlabel("time (ns)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)
    fig.suptitle(
        "E3 bulk-physics null: quiet-drift smoke, seam band on vs off (identical seed)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t1b_null_sigma.png"), dpi=280)
    print("wrote t1b_null_sigma.png")


if __name__ == "__main__":
    main()
