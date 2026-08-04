#!/usr/bin/env python3
"""Run the three energy-equation CI cases under each qdsmc_time_advance
scheme (sources-on validation leg of the Thrust-A bake-off).

Each (case, scheme) pair runs in its own directory; the scheme is injected by
monkeypatching picmi.Simulation.initialize_warpx (the knob is not a deck
argument), then the case's physics analysis runs with the CMakeLists
tolerances. Checksums are intentionally not run (leapfrog/pc change physics;
euler is the bit-identical control).

Usage:
  python3 run_ci_matrix.py --python ~/.env/warpx-qdsmc/bin/python3 --outdir ci_matrix
"""

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TESTDIR = HERE.parent
DECK = TESTDIR / "inputs_test_2d_ohm_solver_electron_energy_picmi.py"

WRAPPER = r"""
import sys
scheme, deck = sys.argv[1], sys.argv[2]
sys.argv = [deck] + sys.argv[3:]
import pywarpx
from pywarpx import picmi
_orig = picmi.Simulation.initialize_warpx
def _patched(self, *a, **k):
    pywarpx.hybridpicmodel.qdsmc_time_advance = scheme
    return _orig(self, *a, **k)
picmi.Simulation.initialize_warpx = _patched
exec(compile(open(deck).read(), deck, "exec"), {"__name__": "__main__", "__file__": deck})
"""

CASES = {
    "adiabat": (
        ["--case", "adiabat", "--test"],
        "analysis_adiabat.py",
        ["--tol-median", "0.01", "--tol-max", "0.05"],
    ),
    "joule": (
        ["--case", "joule", "--test", "--eta-scale", "100"],
        "analysis_joule.py",
        ["--eta-scale", "100"],
    ),
    "qei": (["--case", "qei", "--test"], "analysis_qei.py", ["--nu-ei", "1e6"]),
}
SCHEMES = ("euler", "leapfrog", "pc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--outdir", default="ci_matrix")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir).resolve()
    results = {}
    for case, (deck_args, analysis, ana_args) in CASES.items():
        for scheme in SCHEMES:
            rundir = outdir / f"{case}_{scheme}"
            rundir.mkdir(parents=True, exist_ok=True)
            tag = f"{case}/{scheme}"

            run_log = rundir / "run.log"
            if not (rundir / "diags").exists():
                print(f"[run ] {tag}")
                res = subprocess.run(
                    [args.python, "-c", WRAPPER, scheme, str(DECK)] + deck_args,
                    cwd=rundir,
                    capture_output=True,
                    text=True,
                )
                run_log.write_text(res.stdout + "\n=== STDERR ===\n" + res.stderr)
                if res.returncode != 0:
                    results[tag] = ("RUN FAILED", "")
                    print(f"      RUN FAILED (see {run_log})")
                    continue
            else:
                print(f"[skip] {tag} (diags exist)")

            res = subprocess.run(
                [args.python, str(TESTDIR / analysis)] + ana_args,
                cwd=rundir,
                capture_output=True,
                text=True,
            )
            (rundir / "analysis.log").write_text(
                res.stdout + "\n=== STDERR ===\n" + res.stderr
            )
            status = "PASS" if res.returncode == 0 else "ANALYSIS FAIL"
            # keep the analysis' own summary lines for the report
            summary = " | ".join(
                ln.strip() for ln in res.stdout.splitlines()[-4:] if ln.strip()
            )
            results[tag] = (status, summary)
            print(f"      {status}")

    lines = [
        "# CI matrix: case x qdsmc_time_advance",
        "",
        "| case | scheme | status | analysis tail |",
        "|------|--------|--------|---------------|",
    ]
    for case in CASES:
        for scheme in SCHEMES:
            st, sm = results.get(f"{case}/{scheme}", ("?", ""))
            lines.append(f"| {case} | {scheme} | **{st}** | {sm[:180]} |")
    text = "\n".join(lines)
    (outdir / "ci_matrix_results.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
