#!/usr/bin/env python3
"""Scatter + Keys deposit gauntlet: exact conservation + high-order transfer.

The conservative counterpart of the layer form (plan doc C.7b follow-up,
Eric 2026-08-05: strict conservation is a no-go to give up): keep the
scatter deposit but replace hat+B1 with the Keys cubic-convolution kernel
(partition of unity exact -> conservation exact; quadratic reproduction ->
the remap moments vanish without any correction). Godunov price: negative
lobes -- the Zeldovich section measures the front ringing (undershoot).

Reference numbers printed inline: scatter-hat (the leak) and
layer-monocubic (the monotone non-conservative arm).
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "keys_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)


def run_case(script, tag, **kw):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(path):
        return np.load(path)
    cmd = [PY, os.path.join(HERE, script), "--out", path]
    for key, val in kw.items():
        cmd.append("--" + key.replace("_", "-"))
        if isinstance(val, (list, tuple)):
            cmd += [str(v) for v in val]
        else:
            cmd.append(str(val))
    print("[run ]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=ENV, cwd=HERE)
    return np.load(path)


def sumdrift(d):
    return float((d["te_sum1"] - d["te_sum0"]) / d["te_sum0"])


def wig_row(tag, d):
    leak = float(d["chi_perp_num"]) / float(d["chi0"])
    return (
        f"| {tag} | {int(d['ncell'])} | {int(d['nsteps'])} "
        f"| {float(d['eps']):.2f} | {'/'.join(str(int(v)) for v in d['npts'])} "
        f"| {leak:.3e} | {float(d['adrift']):.2e} "
        f"| {float(d['chi_par_meas']) / float(d['chi0']):.3f} "
        f"| {sumdrift(d):.2e} |"
    )


WIG_HDR = (
    "| arm | N | steps | eps | npts | leak/chi0 | adrift | chi_par | sum drift |\n"
    "|-----|---|-------|-----|------|-----------|--------|---------|-----------|"
)


def main(sections):
    if "aligned" in sections:
        print("\n## aligned straight-B order (scatter-keys)\n")
        print("| N | steps | rel L2 | order | chi/chi0 | sum drift |")
        print("|---|-------|--------|-------|----------|-----------|")
        prev = None
        for n in (32, 48, 64, 96, 128):
            steps = max(8, round(32 * (n / 32) ** 2))
            d = run_case(
                "qdsmc_conduction_test.py",
                f"al_keys_N{n}",
                mode="aligned",
                ncell=n,
                nsteps=steps,
                npts=[3],
                deposit_kernel="keys",
            )
            e = float(d["rel_l2"])
            o = f"{np.log(prev[1] / e) / np.log(n / prev[0]):.2f}" if prev else ""
            print(
                f"| {n} | {steps} | {e:.4e} | {o} "
                f"| {float(d['chi_par_meas'] / d['chi0']):.4f} "
                f"| {sumdrift(d):.2e} |"
            )
            prev = (n, e)
        print("(layer-monocubic: 1.97; scatter-hat+B1: 1.95)")

    if "wiggle" in sections:
        print("\n## wiggle: floor, leak point, dt-growth, liftoff point\n")
        print(WIG_HDR)
        d = run_case(
            "qdsmc_wiggle_test.py",
            "floor_keys_N64",
            ncell=64,
            nsteps=64,
            npts=[3],
            eps=0.0,
            mwiggle=2,
            deposit_kernel="keys",
        )
        print(wig_row("keys eps=0 floor", d))
        for ns in (64, 256):
            d = run_case(
                "qdsmc_wiggle_test.py",
                f"wig_keys_ns{ns}",
                ncell=64,
                nsteps=ns,
                npts=[3],
                eps=0.2,
                mwiggle=2,
                deposit_kernel="keys",
            )
            print(wig_row("keys eps=0.2", d))
        print(
            "(scatter-hat: 2.07e-1 growing to 3.44e-1; "
            "layer-monocubic-straight: 4.7e-4)"
        )
        for np_ in (2, 3):
            d = run_case(
                "qdsmc_wiggle_test.py",
                f"lift_keys_np{np_}",
                ncell=128,
                nsteps=224,
                npts=[np_],
                eps=0.45,
                mwiggle=2,
                deposit_kernel="keys",
            )
            print(wig_row("keys liftoff-pt", d))
        print(
            "(scatter-hat: np2 1.81, np3 0.56; layer-monocubic-straight: "
            "np2 2.56e-4, np3 3.95e-4)"
        )

    if "zeld" in sections:
        print("\n## Zeldovich front + ringing  [N=128, ns=256, scatter-keys]\n")
        d = run_case(
            "qdsmc_zeldovich_test.py",
            "zeld_keys",
            ncell=128,
            nsteps=256,
            deposit_kernel="keys",
        )
        print(
            f"relL2(prof) = {float(d['rel_l2']):.4e} "
            f"(scatter-hat plateau 1.32e-01; layer-monocubic+fixup 4.20e-02)\n"
            f"front_exp = {float(d['front_exp']):.3f} (reference 0.206)\n"
            f"undershoot (min Te below T0) = {float(d['undershoot']):.3e}\n"
            f"sum drift = {sumdrift(d):.3e}  (must be ~1e-8: exact scatter)"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["aligned", "wiggle", "zeld"])
