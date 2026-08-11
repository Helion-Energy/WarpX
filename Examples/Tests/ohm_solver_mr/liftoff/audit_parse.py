#!/usr/bin/env python
"""Parse the Hybrid MR div(B) audit blocks from a liftoff arm log and plot
the per-class raw |divB| maxima vs step (criterion 4: the machine-zero
classes must hold through the seam sweep).

Usage: audit_parse.py RUNLOG [--out PREFIX] [--dt-us 8.7459e-4]
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FINE_RE = re.compile(
    r"fine lev 1: valid raw ([\d.e+-]+) rel ([\d.e+-]+) \| ghost ring raw "
    r"([\d.e+-]+) rel ([\d.e+-]+) \| ghost band raw ([\d.e+-]+) rel "
    r"([\d.e+-]+) \| ghost wall raw ([\d.e+-]+)"
)
CRSE_RE = re.compile(
    r"crse lev 0: interior raw ([\d.e+-]+) rel ([\d.e+-]+) \| seam ring raw "
    r"([\d.e+-]+) rel ([\d.e+-]+) \| exterior raw ([\d.e+-]+) rel "
    r"([\d.e+-]+) \| wall seam raw ([\d.e+-]+) rel ([\d.e+-]+) \| wall all "
    r"raw ([\d.e+-]+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runlog")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--dt-us", type=float, default=8.745929994e-4)
    args = ap.parse_args()

    fine, crse = [], []
    with open(args.runlog) as f:
        for line in f:
            m = FINE_RE.search(line)
            if m:
                fine.append([float(x) for x in m.groups()])
            m = CRSE_RE.search(line)
            if m:
                crse.append([float(x) for x in m.groups()])
    fine = np.array(fine)
    crse = np.array(crse)
    n = min(len(fine), len(crse))
    print(f"{n} audit blocks parsed from {args.runlog}")
    if n == 0:
        return
    t = (1 + np.arange(n)) * args.dt_us

    classes = {
        "fine valid": fine[:n, 0],
        "fine ghost ring": fine[:n, 2],
        "fine ghost band": fine[:n, 4],
        "fine ghost wall": fine[:n, 6],
        "crse interior": crse[:n, 0],
        "crse seam ring": crse[:n, 2],
        "crse exterior": crse[:n, 4],
        "crse wall seam": crse[:n, 6],
        "crse wall all": crse[:n, 8],
    }
    zero_classes = [
        "fine valid",
        "fine ghost band",
        "crse interior",
        "crse seam ring",
        "crse exterior",
        "crse wall seam",
        "crse wall all",
    ]
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for nm, v in classes.items():
        lw = 2 if nm in zero_classes else 1.2
        ls = "-" if nm in zero_classes else "--"
        ax.semilogy(t, np.maximum(v, 1e-25), ls, lw=lw, label=nm)
    ax.set_xlabel("t (us)")
    ax.set_ylabel("max |div B| raw (T/m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Hybrid MR div(B) audit classes (solid = machine-zero classes)")
    fig.savefig(f"{args.out}_divb.png", dpi=280)
    print(f"saved {args.out}_divb.png")
    for nm in zero_classes:
        v = classes[nm]
        print(
            f"  {nm:16s}: max {v.max():.3e} at t={t[int(np.argmax(v))]:.2f} us, "
            f"final {v[-1]:.3e}"
        )


if __name__ == "__main__":
    main()
