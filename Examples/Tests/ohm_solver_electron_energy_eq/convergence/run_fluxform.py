#!/usr/bin/env python3
"""Flux-form (Esirkepov) remap gauntlet: the C.7d gates GF1-GF4.

Runs the same measurement points as the layer gauntlet (run_layer.py) with
`qdsmc_conduction_form = fluxform`, so every row is directly comparable to
the recorded layer/scatter references.

Sections
--------
aligned : straight-B parabolic N sweep (npts=3) -- order and constant vs
          layer-monocubic (1.97) and scatter G3 (1.95).  [GF2]
floors  : wiggle eps = 0 straight-field floor (layer: 3.6e-6).  [GF1]
wiggle  : curvature-leak point eps=0.2/m=2/N=64 (layer-monocubic 4.7e-4,
          scatter 2.07e-1) + dt-growth (must be FLAT) + the liftoff
          dimensionless point N=128/ns=224/eps=0.45 at npts=2 and 3
          (layer 2.56e-4 / scatter 1.81 at npts=2).  [GF3]
zeld    : Zeldovich front, no fixup needed (conservation is structural);
          monotonicity = min(Te) must stay at the floor.  [GF4]

Every row reports Sigma(Te) drift, which for fluxform must sit at the
harness floor (~3e-8, the same as scatter) -- any excess is a bug.
"""

import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "fluxform_out")
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
    t0 = time.time()
    subprocess.run(cmd, check=True, env=ENV, cwd=HERE)
    print(f"[time] {tag}: {time.time() - t0:.1f} s", flush=True)
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
        print("\n## aligned straight-B order  [GF2: >= 1.9, constant vs layer 1.97]\n")
        print("| form | N | steps | rel L2 | order | chi/chi0 | sum drift |")
        print("|------|---|-------|--------|-------|----------|-----------|")
        prev = None
        for n in (32, 48, 64, 96, 128):
            steps = max(8, round(32 * (n / 32) ** 2))
            d = run_case(
                "qdsmc_conduction_test.py",
                f"al_N{n}",
                mode="aligned",
                ncell=n,
                nsteps=steps,
                npts=[3],
                form="fluxform",
            )
            e = float(d["rel_l2"])
            o = f"{np.log(prev[1] / e) / np.log(n / prev[0]):.2f}" if prev else ""
            print(
                f"| fluxform | {n} | {steps} | {e:.4e} | {o} "
                f"| {float(d['chi_par_meas'] / d['chi0']):.4f} "
                f"| {sumdrift(d):.2e} |"
            )
            prev = (n, e)

    if "tilted" in sections:
        print("\n## tilted 30deg full tensor  [GF2: split remap on oblique branches]\n")
        print("| form | N | steps | rel L2 | order | chi_perp_num/chi0 | sum drift |")
        print("|------|---|-------|--------|-------|-------------------|-----------|")
        prev = None
        for n in (48, 64, 96):
            steps = max(8, round(32 * (n / 32) ** 2))
            d = run_case(
                "qdsmc_conduction_test.py",
                f"tl_N{n}",
                mode="tilted",
                ncell=n,
                nsteps=steps,
                npts=[3],
                form="fluxform",
            )
            e = float(d["rel_l2"])
            o = f"{np.log(prev[1] / e) / np.log(n / prev[0]):.2f}" if prev else ""
            print(
                f"| fluxform | {n} | {steps} | {e:.4e} | {o} "
                f"| {float(d['chi_perp_meas'] / d['chi0']):.3e} "
                f"| {sumdrift(d):.2e} |"
            )
            prev = (n, e)
        print("(scatter G3 raw reference at N=64: 1.7e-3 = the moment-estimator")
        print(" floor; score the exact field with the same estimator to subtract)")

    if "floors" in sections:
        print("\n## wiggle eps = 0 floor  [GF1: layer floor 3.6e-6]\n")
        print(WIG_HDR)
        for n in (64, 128):
            d = run_case(
                "qdsmc_wiggle_test.py",
                f"floor_N{n}",
                ncell=n,
                nsteps=64,
                npts=[3],
                eps=0.0,
                mwiggle=2,
                form="fluxform",
            )
            print(wig_row("fluxform", d))

    if "wiggle" in sections:
        print("\n## curvature leak  [GF3; eps=0.2, m=2, N=64, ns=64]\n")
        print(WIG_HDR)
        base = dict(ncell=64, nsteps=64, npts=[3], eps=0.2, mwiggle=2)
        d = run_case("qdsmc_wiggle_test.py", "wig_fluxform", form="fluxform", **base)
        print(wig_row("fluxform", d))
        print("(references: layer-monocubic 4.7e-4 (curved 2.1e-1), scatter 2.072e-01)")

        print("\n## dt-growth signature  [GF3: must be FLAT]\n")
        print(WIG_HDR)
        for ns in (64, 256):
            d = run_case(
                "qdsmc_wiggle_test.py",
                f"wigdt_ns{ns}",
                **{**base, "nsteps": ns, "form": "fluxform"},
            )
            print(wig_row("fluxform", d))
        print("(scatter grew 0.207 -> 0.344 over the same span; layer stayed flat)")

        print("\n## liftoff dimensionless point  [GF3; N=128, ns=224, eps=0.45, m=2]\n")
        print(WIG_HDR)
        for np_ in (2, 3):
            d = run_case(
                "qdsmc_wiggle_test.py",
                f"lift_np{np_}",
                ncell=128,
                nsteps=224,
                npts=[np_],
                eps=0.45,
                mwiggle=2,
                form="fluxform",
            )
            print(wig_row("fluxform", d))
        print("(references at npts=2: layer-monocubic 2.56e-4, scatter 1.812e+00;")
        print(" chi_par ~ 0.89 here is the z-vs-arc estimator factor")
        print(" (1+eps^2/4)^-2 at eps=0.45, not a transport deficit)")

    if "zeld" in sections:
        print("\n## Zeldovich front  [GF4; N=128, ns=256, no fixup]\n")
        d = run_case(
            "qdsmc_zeldovich_test.py",
            "zeld_fluxform",
            ncell=128,
            nsteps=256,
            form="fluxform",
        )
        print(
            f"relL2(prof) = {float(d['rel_l2']):.4e} "
            f"(layer+fixup 4.2e-2, scatter plateau 1.32e-01), "
            f"front_exp = {float(d['front_exp']):.3f} (reference 0.206), "
            f"sum drift = {sumdrift(d):.3e}, "
            f"min(Te_final) = {float(np.min(d['te_final'])):.3e} K "
            f"(never-add: no undershoot below the ambient floor)"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["aligned", "tilted", "floors", "wiggle", "zeld"])
