#!/usr/bin/env python3
"""Hybrid-PIC MR smoke analysis: energy traces from the reduced diagnostics
and the div(B) audit trend parsed from the run log
(hybrid_pic_model.mr_check_div_b = 1).

Usage:
    python t_smoke_energy_divb.py RUN_DIR LOGFILE [CTL_RUN_DIR] [OUT_PREFIX]

RUN_DIR must contain diags/reducedfiles/{field_energy,part_energy}.txt.
Writes <OUT_PREFIX>_energy_divb.png (default prefix "t_smoke") and prints a
summary table.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FINE_RE = re.compile(
    r"fine lev (\d+): valid raw (\S+) rel (\S+) \| ghost ring raw (\S+) rel (\S+)"
    r" \| ghost band raw (\S+) rel (\S+)"
)
CRSE_RE = re.compile(
    r"crse lev (\d+): interior raw (\S+) rel (\S+) \| seam ring raw (\S+) rel (\S+)"
    r" \| exterior raw (\S+) rel (\S+)"
)


def load_energy(run_dir):
    fe = np.loadtxt(
        os.path.join(run_dir, "diags/reducedfiles/field_energy.txt"), skiprows=1
    )
    pe = np.loadtxt(
        os.path.join(run_dir, "diags/reducedfiles/part_energy.txt"), skiprows=1
    )
    return fe, pe


def main():
    run_dir, logfile = sys.argv[1], sys.argv[2]
    ctl_dir = sys.argv[3] if len(sys.argv) > 3 and os.path.isdir(sys.argv[3]) else None
    prefix = sys.argv[4] if len(sys.argv) > 4 else "t_smoke"

    fine, crse = [], []
    with open(logfile) as f:
        for line in f:
            m = FINE_RE.search(line)
            if m:
                fine.append([float(x) for x in m.groups()[1:]])
            m = CRSE_RE.search(line)
            if m:
                crse.append([float(x) for x in m.groups()[1:]])
    fine = np.array(fine)  # cols: valid raw/rel, ring raw/rel, band raw/rel
    crse = np.array(crse)  # cols: interior raw/rel, seam raw/rel, exterior raw/rel

    print("div(B) audit maxima over the run (relative to max|B|/dx):")
    print(f"  fine valid      max rel {fine[:, 1].max():.3e}")
    print(f"  fine ghost ring max rel {fine[:, 3].max():.3e}")
    print(f"  fine ghost band max rel {fine[:, 5].max():.3e}")
    print(f"  crse interior   max rel {crse[:, 1].max():.3e}")
    print(
        f"  crse seam ring  max rel {crse[:, 3].max():.3e}  (raw {crse[:, 2].max():.3e} T/m)"
    )
    print(f"  crse exterior   max rel {crse[:, 5].max():.3e}")

    fe, pe = load_energy(run_dir)
    print("energy trace (field total, particle total):")
    for i in [0, len(fe) // 2, len(fe) - 1]:
        print(
            f"  step {int(fe[i, 0]):5d}: field {fe[i, 2]:.6e} J  particle {pe[i, 2]:.6e} J"
        )
    print(f"  particle energy change: {(pe[-1, 2] / pe[0, 2] - 1) * 100:.2f} %")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    steps = np.arange(1, len(crse) + 1)
    axs[0].semilogy(steps, crse[:, 3], color="#3d65f5", lw=2, label="coarse seam ring")
    axs[0].semilogy(steps, fine[:, 3], color="#c95d20", lw=2, label="fine ghost ring")
    axs[0].semilogy(
        steps,
        np.maximum(crse[:, 1], 1e-18),
        color="#2e9e73",
        lw=1.5,
        label="coarse interior",
    )
    axs[0].semilogy(
        steps,
        np.maximum(fine[:, 1], 1e-18),
        color="#8d69cf",
        lw=1.5,
        label="fine valid",
    )
    axs[0].set_xlabel("step")
    axs[0].set_ylabel(r"max $|\nabla\cdot B|\,\Delta x / \max|B|$")
    axs[0].set_title("div(B) audit")
    axs[0].legend(frameon=False, fontsize=9)
    axs[0].grid(alpha=0.25, lw=0.5)

    axs[1].plot(
        pe[:, 0], pe[:, 2] / pe[0, 2], color="#3d65f5", lw=2, label="particle (MR)"
    )
    axs[1].plot(
        fe[:, 0], fe[:, 2] / fe[0, 2], color="#c95d20", lw=2, label="field (MR)"
    )
    if ctl_dir:
        fec, pec = load_energy(ctl_dir)
        axs[1].plot(
            pec[:, 0],
            pec[:, 2] / pec[0, 2],
            color="#3d65f5",
            lw=1.2,
            ls="--",
            label="particle (control)",
        )
        axs[1].plot(
            fec[:, 0],
            fec[:, 2] / fec[0, 2],
            color="#c95d20",
            lw=1.2,
            ls="--",
            label="field (control)",
        )
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("energy / initial")
    axs[1].set_title("reduced-diagnostics energy trace")
    axs[1].legend(frameon=False, fontsize=9)
    axs[1].grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}_energy_divb.png", dpi=270)
    print(f"wrote {prefix}_energy_divb.png")


if __name__ == "__main__":
    main()
