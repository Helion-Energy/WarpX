#!/usr/bin/env python3
"""Phase-0 baseline sweep driver for the QDSMC advection harness.

Runs qdsmc_advection_test.py as subprocesses (one WarpX instance per run),
collects the saved .npz fields, computes error norms and convergence slopes,
and writes baseline_results.md.

Measurements (plan gate G0):
  1. at_rest      : per-step smoothing of a stationary blob (E5/E6).
  2. translate    : dx order vs the exact shifted blob, fixed CFL.
  3. rotate (dt)  : dt self-convergence at fixed N (expected slope ~1 for the
                    baseline forward-Euler scheme).

Usage:
  python3 run_baseline.py --python /path/to/venv/python3 --outdir baseline_out
  python3 run_baseline.py ... --only at_rest,translate
"""

import argparse
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DECK = HERE / "qdsmc_advection_test.py"

# Default qdsmc_time_advance for the three main sections, set from --advance.
# The recorded baseline in this file was measured at "euler"; production runs
# "pc", so the sweeps have to be runnable at either without editing call sites.
# Sections that deliberately pin a scheme (the bake-off) pass advance=
# explicitly and are left alone by the setdefault below.
ADVANCE = "euler"


def run_case(python, outdir, tag, **kw):
    kw.setdefault("advance", ADVANCE)
    # The scheme goes in the CACHE KEY, not just the command. run_case skips
    # when the npz exists, so without this a pc sweep into an outdir that
    # already holds euler results would silently "[skip]" every case and
    # report the euler numbers as if they were pc.
    tag = f"{tag}_{kw['advance']}"
    out = outdir / f"{tag}.npz"
    if out.exists():
        print(f"[skip] {tag} (exists)")
        return out
    cmd = [python, str(DECK), "--out", str(out)]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    print("[run ]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:])
        print(res.stderr[-3000:])
        raise RuntimeError(f"case {tag} failed (exit {res.returncode})")
    tail = [ln for ln in res.stdout.splitlines() if ln.startswith("[harness]")]
    print("      ", "\n       ".join(tail))
    return out


def l2(a, b):
    d = (a - b)[:-1, :-1]  # drop duplicated periodic nodal row/col
    return float(np.sqrt(np.mean(d**2)))


def wrap_blob(xg, zg, x0, z0, sb, amp, L):
    dxp = (xg - x0 + 0.5 * L) % L - 0.5 * L
    dzp = (zg - z0 + 0.5 * L) % L - 0.5 * L
    return amp * np.exp(-(dxp**2 + dzp**2) / (2.0 * sb**2))


def nodes(n, L):
    x = np.linspace(0.0, L, n)
    return x


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--outdir", default="baseline_out")
    ap.add_argument("--only", default="at_rest,translate,rotate")
    ap.add_argument(
        "--ncell-dt", type=int, default=128, help="fixed grid for the dt sweep"
    )
    ap.add_argument(
        "--advance",
        choices=["euler", "leapfrog", "pc"],
        default="euler",
        help="qdsmc_time_advance for the three main sections. Default euler "
        "reproduces the recorded baseline; production runs pc.",
    )
    args = ap.parse_args()

    global ADVANCE
    ADVANCE = args.advance

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    report = ["# QDSMC Phase-0 baseline measurements", ""]

    # ------------------------------------------------------------------
    # 1. at-rest smoothing
    # ------------------------------------------------------------------
    if "at_rest" in only:
        report += [
            "## at-rest smoothing (E5/E6)",
            "",
            "| N | steps | L2(Te(t)-Te(0)) | L2 per step |",
            "|---|-------|-----------------|-------------|",
        ]
        for N in (32, 64, 128):
            f = run_case(
                args.python, outdir, f"rest_N{N}", mode="at_rest", ncell=N, nsteps=64
            )
            d = np.load(f)
            e = l2(d["te_final"], d["te_initial"]) / (d["Te0_eV"] * 11604.5)
            report.append(f"| {N} | 64 | {e:.4e} | {e / 64:.4e} |")
        report.append("")

    # ------------------------------------------------------------------
    # 2. translate: dx order vs exact shifted blob (time-exact mode)
    # ------------------------------------------------------------------
    if "translate" in only:
        errs, hs = [], []
        report += [
            "## translation: spatial order vs exact solution",
            "",
            "| N | L2 error | order |",
            "|---|----------|-------|",
        ]
        prev = None
        for N in (32, 48, 64, 96, 128):
            f = run_case(
                args.python, outdir, f"tr_N{N}", mode="translate", ncell=N, nsteps=N
            )  # fixed CFL
            d = np.load(f)
            L = float(d["L"])
            xb, zb, sb, amp = d["blob"]
            v = float(d["vmax"]) / np.sqrt(2.0)
            t = float(d["tfinal"])
            n = d["te_final"].shape[0]
            xg, zg = np.meshgrid(nodes(n, L), nodes(n, L), indexing="ij")
            K_per_eV = 11604.5
            te_exact = (
                d["Te0_eV"]
                * K_per_eV
                * (1.0 + wrap_blob(xg, zg, xb + v * t, zb + v * t, sb, amp, L))
            )
            e = l2(d["te_final"], te_exact) / (d["Te0_eV"] * K_per_eV)
            o = (
                ""
                if prev is None
                else f"{np.log(prev[1] / e) / np.log(N / prev[0]):.2f}"
            )
            report.append(f"| {N} | {e:.4e} | {o} |")
            errs.append(e)
            hs.append(1.0 / N)
            prev = (N, e)
        report += ["", f"fit slope (in dx): **{slope(hs, errs):.2f}**", ""]

    # ------------------------------------------------------------------
    # 3. rotate: dt self-convergence at fixed N
    # ------------------------------------------------------------------
    if "rotate" in only:
        N = args.ncell_dt
        # Coarsest dt must keep the ion/marker CFL < 1 (Esirkepov + hybrid
        # segfaults on multi-cell crossings; QDSMC PushX assumes <= 1 cell).
        # At N=128, tfinal=0.5*L/vmax, realized |v|max ~ 0.6*vmax: 64 steps
        # gives CFL ~ 0.55.
        steps_list = (64, 128, 256, 512, 1024)
        fs = {
            s: run_case(
                args.python, outdir, f"rot_N{N}_s{s}", mode="rotate", ncell=N, nsteps=s
            )
            for s in steps_list
        }
        ref = np.load(fs[steps_list[-1]])["te_final"]
        report += [
            f"## rotation: dt self-convergence (N={N}, ref = {steps_list[-1]} steps)",
            "",
            "| steps | dt rel. | L2 vs ref | order |",
            "|-------|---------|-----------|-------|",
        ]
        errs, dts, prev = [], [], None
        for s in steps_list[:-1]:
            d = np.load(fs[s])
            e = l2(d["te_final"], ref) / (float(d["Te0_eV"]) * 11604.5)
            o = (
                ""
                if prev is None
                else f"{np.log(prev[1] / e) / np.log(prev[0] / (1.0 / s)):+.2f}"
            )
            report.append(f"| {s} | 1/{s} | {e:.4e} | {o} |")
            errs.append(e)
            dts.append(1.0 / s)
            prev = (1.0 / s, e)
        report += ["", f"fit slope (in dt): **{slope(dts, errs):.2f}**", ""]

    # ------------------------------------------------------------------
    # 3b. spatial order with a resolved blob (G2 instrument): sigma = 0.08L
    #     so the remap variance is small vs the blob variance over the sweep.
    # ------------------------------------------------------------------
    if "spatial2" in only:
        report += [
            "## translation, resolved blob (sigma = 0.08 L): spatial order",
            "",
            "| N | L2 error | order |",
            "|---|----------|-------|",
        ]
        errs2, hs2, prev = [], [], None
        for N in (32, 48, 64, 96, 128, 192):
            f = run_case(
                args.python,
                outdir,
                f"tr2_N{N}",
                mode="translate",
                ncell=N,
                nsteps=N,
                blob_sigma=0.08,
            )
            d = np.load(f)
            L = float(d["L"])
            xb, zb, sb, amp = d["blob"]
            v = float(d["vmax"]) / np.sqrt(2.0)
            t = float(d["tfinal"])
            n = d["te_final"].shape[0]
            xg, zg = np.meshgrid(nodes(n, L), nodes(n, L), indexing="ij")
            te_exact = (
                d["Te0_eV"]
                * 11604.5
                * (1.0 + wrap_blob(xg, zg, xb + v * t, zb + v * t, sb, amp, L))
            )
            e = l2(d["te_final"], te_exact) / (d["Te0_eV"] * 11604.5)
            o = (
                ""
                if prev is None
                else f"{np.log(prev[1] / e) / np.log(N / prev[0]):.2f}"
            )
            report.append(f"| {N} | {e:.4e} | {o} |")
            errs2.append(e)
            hs2.append(1.0 / N)
            prev = (N, e)
        report += ["", f"fit slope (in dx): **{slope(hs2, errs2):.2f}**", ""]

    # ------------------------------------------------------------------
    # 3c. B1 payoff: same resolved-blob translation sweep with the
    #     half-gradient-corrected deposit (expect slope -> ~2).
    # ------------------------------------------------------------------
    if "spatial2g" in only:
        report += [
            "## translation, resolved blob, GRADIENT deposit (B1)",
            "",
            "| N | L2 error | order |",
            "|---|----------|-------|",
        ]
        errs2, hs2, prev = [], [], None
        for N in (32, 48, 64, 96, 128, 192):
            f = run_case(
                args.python,
                outdir,
                f"tr2g_N{N}",
                mode="translate",
                ncell=N,
                nsteps=N,
                blob_sigma=0.08,
                grad_deposit=1,
            )
            d = np.load(f)
            L = float(d["L"])
            xb, zb, sb, amp = d["blob"]
            v = float(d["vmax"]) / np.sqrt(2.0)
            t = float(d["tfinal"])
            n = d["te_final"].shape[0]
            xg, zg = np.meshgrid(nodes(n, L), nodes(n, L), indexing="ij")
            te_exact = (
                d["Te0_eV"]
                * 11604.5
                * (1.0 + wrap_blob(xg, zg, xb + v * t, zb + v * t, sb, amp, L))
            )
            e = l2(d["te_final"], te_exact) / (d["Te0_eV"] * 11604.5)
            o = (
                ""
                if prev is None
                else f"{np.log(prev[1] / e) / np.log(N / prev[0]):.2f}"
            )
            report.append(f"| {N} | {e:.4e} | {o} |")
            errs2.append(e)
            hs2.append(1.0 / N)
            prev = (N, e)
        report += ["", f"fit slope (in dx): **{slope(hs2, errs2):.2f}**", ""]

    # ------------------------------------------------------------------
    # 4. bake-off: euler vs leapfrog vs pc (rotate, common fine reference)
    # ------------------------------------------------------------------
    if "bakeoff" in only:
        N = args.ncell_dt
        steps_list = (64, 128, 256, 512)
        schemes = ("euler", "leapfrog", "pc")
        # Common reference: all schemes converge to the same dt->0 limit at
        # fixed N (exact trajectories + the same per-remap smoothing), so one
        # ultra-fine run serves all three.
        ref_f = run_case(
            args.python,
            outdir,
            "bo_ref_leapfrog_s2048",
            mode="rotate",
            ncell=N,
            nsteps=2048,
            advance="leapfrog",
        )
        ref = np.load(ref_f)["te_final"]
        Te0K = float(np.load(ref_f)["Te0_eV"]) * 11604.5

        errs = {}
        for sch in schemes:
            for s in steps_list:
                f = run_case(
                    args.python,
                    outdir,
                    f"bo_{sch}_s{s}",
                    mode="rotate",
                    ncell=N,
                    nsteps=s,
                    advance=sch,
                )
                errs[(sch, s)] = l2(np.load(f)["te_final"], ref) / Te0K

        report += [
            f"## bake-off: rotate, N={N}, common ref = leapfrog @ 2048 steps",
            "",
            "| steps | euler | leapfrog | pc | euler/leapfrog | euler/pc |",
            "|-------|-------|----------|----|----------------|----------|",
        ]
        for s in steps_list:
            e, lf, pc = errs[("euler", s)], errs[("leapfrog", s)], errs[("pc", s)]
            report.append(
                f"| {s} | {e:.4e} | {lf:.4e} | {pc:.4e} | "
                f"{e / lf:.1f}x | {e / pc:.1f}x |"
            )
        report.append("")
        for sch in schemes:
            ys = [errs[(sch, s)] for s in steps_list]
            report.append(
                f"{sch}: fit slope (in dt) = **{slope([1.0 / s for s in steps_list], ys):.2f}**"
            )
        report.append("")

        # Same-discretization scheme differences: remap statistics cancel, so
        # euler-vs-2nd-order isolates euler's O(dt) time error, and the
        # pc-leapfrog difference is the half-step staggering offset
        # 0.5*dt*dTe/dt (leapfrog's state ends at t - dt/2) plus the schemes'
        # O(dt^2) disagreement. Subtracting a dt-scaled template of the
        # offset (from the finest pair) exposes the O(dt^2) residual.
        def fld(sch, s):
            return np.load(outdir / f"bo_{sch}_s{s}.npz")["te_final"]

        report += [
            "### same-discretization differences (remap cancels)",
            "",
            "| steps | euler-leapfrog | pc-leapfrog (raw) | pc-leapfrog minus dt-scaled offset |",
            "|-------|----------------|-------------------|-------------------------------------|",
        ]
        tmpl = fld("pc", steps_list[-1]) - fld("leapfrog", steps_list[-1])
        d_el, d_res = [], []
        for s in steps_list:
            de = l2(fld("euler", s), fld("leapfrog", s)) / Te0K
            raw = fld("pc", s) - fld("leapfrog", s)
            res = raw - (steps_list[-1] / s) * tmpl
            dr = l2(raw, 0.0 * raw) / Te0K
            rr = l2(res, 0.0 * res) / Te0K
            report.append(f"| {s} | {de:.4e} | {dr:.4e} | {rr:.4e} |")
            d_el.append(de)
            if s != steps_list[-1]:
                d_res.append(rr)
        report += [
            "",
            f"euler-leapfrog slope: **{slope([1.0 / s for s in steps_list], d_el):.2f}** "
            "(euler's isolated O(dt) time error)",
            f"offset-subtracted pc-leapfrog slope: "
            f"**{slope([1.0 / s for s in steps_list[:-1]], d_res):.2f}** "
            "(the 2nd-order candidates agree to O(dt^2))",
            "",
        ]

    text = "\n".join(report)
    (outdir / "baseline_results.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
