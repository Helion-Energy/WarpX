#!/usr/bin/env python3
"""Layer-form (gather) conduction gauntlet: scatter vs layer x interpolant.

Sections
--------
aligned : straight-B order sanity for the layer form (parabolic N sweep,
          monocubic + keys) -- must not regress the scatter G3 result.
floors  : wiggle eps = 0 straight-field floors per interpolant.
wiggle  : the curvature-leak discrimination matrix at eps=0.2/m=2/N=64
          (scatter cached from run_wiggle) + the dt-growth signature
          (leak must NOT grow with 1/dt if the remap mechanism is gone)
          + the liftoff dimensionless point (N=128, ns=224, eps=0.45,
          npts=2 and 3).
zeld    : Zeldovich front, layer-monocubic vs the scatter 13% plateau.

Every row reports the run's Sigma(Te) drift -- the layer form's liability
(gather is not exactly conservative; fixup stays OFF here).
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "layer_out")
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
        print("\n## aligned straight-B order (layer must not regress G3)\n")
        print("| interp | N | steps | rel L2 | order | chi/chi0 | sum drift |")
        print("|--------|---|-------|--------|-------|----------|-----------|")
        for interp in ("monocubic", "keys"):
            prev = None
            for n in (32, 48, 64, 96, 128):
                steps = max(8, round(32 * (n / 32) ** 2))
                d = run_case(
                    "qdsmc_conduction_test.py",
                    f"al_{interp}_N{n}",
                    mode="aligned",
                    ncell=n,
                    nsteps=steps,
                    npts=[3],
                    form="layer",
                    interp=interp,
                )
                e = float(d["rel_l2"])
                o = f"{np.log(prev[1] / e) / np.log(n / prev[0]):.2f}" if prev else ""
                print(
                    f"| {interp} | {n} | {steps} | {e:.4e} | {o} "
                    f"| {float(d['chi_par_meas'] / d['chi0']):.4f} "
                    f"| {sumdrift(d):.2e} |"
                )
                prev = (n, e)

    if "floors" in sections:
        print("\n## wiggle eps = 0 floors (straight field)\n")
        print(WIG_HDR)
        for interp in ("linear", "monocubic", "keys"):
            for n in (64, 128):
                d = run_case(
                    "qdsmc_wiggle_test.py",
                    f"floor_{interp}_N{n}",
                    ncell=n,
                    nsteps=64,
                    npts=[3],
                    eps=0.0,
                    mwiggle=2,
                    form="layer",
                    interp=interp,
                )
                print(wig_row(f"layer-{interp}", d))

    if "wiggle" in sections:
        print("\n## curvature leak: scatter vs layer  [eps=0.2, m=2, N=64, ns=64]\n")
        print(WIG_HDR)
        base = dict(ncell=64, nsteps=64, npts=[3], eps=0.2, mwiggle=2)
        arms = [
            (
                "layer-linear-straight",
                dict(form="layer", interp="linear", curved_feet=0),
            ),
            ("layer-linear-curved", dict(form="layer", interp="linear", curved_feet=1)),
            (
                "layer-monocubic-straight",
                dict(form="layer", interp="monocubic", curved_feet=0),
            ),
            (
                "layer-monocubic-curved",
                dict(form="layer", interp="monocubic", curved_feet=1),
            ),
            ("layer-keys-curved", dict(form="layer", interp="keys", curved_feet=1)),
        ]
        for tag, kw in arms:
            d = run_case("qdsmc_wiggle_test.py", f"wig_{tag}", **{**base, **kw})
            print(wig_row(tag, d))
        print("(scatter reference at this point: leak/chi0 = 2.072e-01)")

        print("\n## dt-growth signature  [layer-monocubic-curved]\n")
        print(WIG_HDR)
        for ns in (64, 256):
            d = run_case(
                "qdsmc_wiggle_test.py",
                f"wigdt_ns{ns}",
                **{**base, "nsteps": ns, "form": "layer", "interp": "monocubic"},
            )
            print(wig_row("layer-monocubic-curved", d))
        print("(scatter grew 0.207 -> 0.344 over the same span)")

        print("\n## liftoff dimensionless point  [N=128, ns=224, eps=0.45, m=2]\n")
        print(WIG_HDR)
        for cf, cft in ((0, "straight"), (1, "curved")):
            for np_ in (2, 3):
                d = run_case(
                    "qdsmc_wiggle_test.py",
                    f"lift_mc_{cft}_np{np_}",
                    ncell=128,
                    nsteps=224,
                    npts=[np_],
                    eps=0.45,
                    mwiggle=2,
                    form="layer",
                    interp="monocubic",
                    curved_feet=cf,
                )
                print(wig_row(f"layer-monocubic-{cft}", d))
        print("(scatter reference: npts=2 1.812e+00, npts=3 5.623e-01;")
        print(" chi_par ~ 0.89 here is the z-vs-arc estimator factor")
        print(" (1+eps^2/4)^-2 at eps=0.45, not a transport deficit)")

    if "zeld" in sections:
        print("\n## Zeldovich front  [N=128, ns=256, layer-monocubic]\n")
        for fx in (0, 1):
            d = run_case(
                "qdsmc_zeldovich_test.py",
                "zeld_layer_monocubic" if fx == 0 else "zeld_layer_fixup",
                ncell=128,
                nsteps=256,
                form="layer",
                interp="monocubic",
                conserve_fixup=fx,
            )
            print(
                f"fixup={fx}: relL2(prof) = {float(d['rel_l2']):.4e} "
                f"(scatter plateau: 1.32e-01), "
                f"front_exp = {float(d['front_exp']):.3f} (reference 0.206), "
                f"sum drift = {sumdrift(d):.3e}"
            )


if __name__ == "__main__":
    main(sys.argv[1:] or ["aligned", "floors", "wiggle", "zeld"])
