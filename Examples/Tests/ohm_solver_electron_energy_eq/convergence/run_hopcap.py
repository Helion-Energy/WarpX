#!/usr/bin/env python3
"""C.7 hop-cap transport deficit: capped vs uncapped aligned conduction.

Sweeps qdsmc_conduction_max_hop on the aligned G3 deck (N=64, nsteps=64,
npts=3, chi0 = s0^2/(2 tf)) so the cap ceiling chi_cap =
(max_hop dx)^2 / (2 dt_c x_max^2) crosses chi0, and compares the measured
transport chi_par_meas/chi0 against the designed p=4 soft-min response
    chi_eff/chi0 = (1 + (chi0/chi_cap)^4)^(-1/4).
This quantifies the slowdown when the cap engages (the capped-fast-front
transport deficit) and verifies the soft-min in situ.

dt_c = dt/2: the pc advance runs the conduction substep as Strang halves.
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "hopcap_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)

N = 64
NSTEPS = 64
NPTS = 3
TFINAL = 1.0e-5
BLOB_SIGMA = 0.10
L = 1.0

CHI0 = (BLOB_SIGMA * L) ** 2 / (2.0 * TFINAL)
DT_C = TFINAL / NSTEPS / 2.0
XMAX = {2: 1.0, 3: np.sqrt(3.0), 5: 2.8570}[NPTS]
DX = L / N

HOPS = [4.0, 1.5, 1.0, 0.7, 0.5, 0.35]


def run_case(tag, max_hop):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.npz")
    if not os.path.exists(path):
        cmd = [
            PY,
            os.path.join(HERE, "qdsmc_conduction_test.py"),
            "--mode",
            "aligned",
            "--ncell",
            str(N),
            "--nsteps",
            str(NSTEPS),
            "--npts",
            str(NPTS),
            "--max-hop",
            str(max_hop),
            "--out",
            path,
        ]
        print("[run ]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=ENV, cwd=HERE)
    return np.load(path)


def main():
    print(
        f"\n## hop-cap deficit  [aligned, N={N}, ns={NSTEPS}, npts={NPTS}, "
        f"chi0={CHI0:.0f} m^2/s, hop needed l/dx = "
        f"{XMAX * np.sqrt(2 * CHI0 * DT_C) / DX:.2f}]\n"
    )
    print(
        "| max_hop | chi_cap/chi0 | chi_meas/chi0 | soft-min pred | meas/pred | relL2 |"
    )
    print(
        "|---------|--------------|---------------|---------------|-----------|-------|"
    )
    for mh in HOPS:
        d = run_case(f"hop{mh}", mh)
        chi_cap = (mh * DX) ** 2 / (2.0 * DT_C * XMAX**2)
        r = CHI0 / chi_cap
        pred = (1.0 + r**4) ** -0.25
        meas = float(d["chi_par_meas"]) / float(d["chi0"])
        print(
            f"| {mh:7.2f} | {chi_cap / CHI0:12.3f} | {meas:13.4f} "
            f"| {pred:13.4f} | {meas / pred:9.4f} | {float(d['rel_l2']):.3e} |"
        )


if __name__ == "__main__":
    main()
