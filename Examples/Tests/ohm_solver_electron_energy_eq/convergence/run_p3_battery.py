#!/usr/bin/env python3
"""P3 robustness battery driver: GFD3 (wiggle leak), GFD4 (ZK front),
GFD6 (X-point/null maximum principle + integrator x limiter matrix), and
the density-cliff arm, for the FD conduction operator with SDE reference
rows.

Cases run as independent single-rank subprocesses, --jobs at a time
(each case is a small serial deck; the battery parallelizes across
cases, not within them). Results are cached as npz in p3_out/<tag>.npz;
delete a file to force a re-run.

Usage:
  python3 run_p3_battery.py [gfd3 gfd4 gfd6 cliff] --jobs 32
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "p3_out")
ENV = dict(
    os.environ,
    LD_LIBRARY_PATH="/usr/local/openmpi5/lib:/usr/local/hdf5/lib",
    OMP_NUM_THREADS="2",
)

parser = argparse.ArgumentParser()
parser.add_argument("sections", nargs="*", default=[])
parser.add_argument("--jobs", type=int, default=8)
args = parser.parse_args()
SECTIONS = args.sections or ["gfd3", "gfd4", "gfd6", "cliff", "a4"]

# tolerances
TOL_EXTREMUM = 1.0e-9  # max-principle violation threshold (rel Te0)
TOL_SUM = 1.0e-6  # Sigma drift threshold over a full run
TOL_MINT = 1.0e-11  # atrest cliff arm minted structure (FD exact-zero)


def build_cases():
    cases = []  # (section, script, tag, kwargs)

    if "gfd3" in SECTIONS:
        for n in (64, 128):
            for op in ("fd", "sde"):
                cases.append(
                    (
                        "gfd3",
                        "qdsmc_wiggle_test.py",
                        f"wig_{op}_N{n}",
                        dict(ncell=n, nsteps=64, eps=0.42, mwiggle=6, conduction_op=op),
                    )
                )
            cases.append(
                (
                    "gfd3",
                    "qdsmc_wiggle_test.py",
                    f"wig_fd_floor_N{n}",
                    dict(ncell=n, nsteps=64, eps=0.0, mwiggle=6, conduction_op="fd"),
                )
            )
        # rkf45 leg at the stress point
        cases.append(
            (
                "gfd3",
                "qdsmc_wiggle_test.py",
                "wig_fd_rkf45_N128",
                dict(
                    ncell=128,
                    nsteps=64,
                    eps=0.42,
                    mwiggle=6,
                    conduction_op="fd",
                    fd_time="rkf45",
                ),
            )
        )

    if "gfd4" in SECTIONS:
        for op in ("fd", "sde"):
            for n in (64, 128):
                cases.append(
                    (
                        "gfd4",
                        "qdsmc_zeldovich_test.py",
                        f"zk_{op}_N{n}",
                        dict(ncell=n, nsteps=256, conduction_op=op),
                    )
                )
        for lim in ("upwind1", "none"):
            cases.append(
                (
                    "gfd4",
                    "qdsmc_zeldovich_test.py",
                    f"zk_fd_{lim}_N128",
                    dict(ncell=128, nsteps=256, conduction_op="fd", fd_limiter=lim),
                )
            )
        cases.append(
            (
                "gfd4",
                "qdsmc_zeldovich_test.py",
                "zk_fd_rkf45_N128",
                dict(ncell=128, nsteps=256, conduction_op="fd", fd_time="rkf45"),
            )
        )
        # flux-limited leg (front speed bound must hold)
        cases.append(
            (
                "gfd4",
                "qdsmc_zeldovich_test.py",
                "zk_fd_flim_N128",
                dict(ncell=128, nsteps=256, conduction_op="fd", flux_limit=0.1),
            )
        )

    if "gfd6" in SECTIONS:
        fd_ops = [
            ("smart", "ssprk2"),
            ("smart", "rkf45"),
            ("upwind1", "ssprk2"),
            ("none", "ssprk2"),
            ("none", "rkf45"),
        ]
        for eps in (1.0e-4, 1.0e-7):
            for noise in (0.0, 0.1):
                etag = f"e{eps:.0e}_n{noise}"
                for lim, tint in fd_ops:
                    cases.append(
                        (
                            "gfd6",
                            "qdsmc_xpoint_test.py",
                            f"xp_fd_{lim}_{tint}_{etag}",
                            dict(
                                conduction_op="fd",
                                fd_limiter=lim,
                                fd_time=tint,
                                eps_perp=eps,
                                b_noise=noise,
                                kappa_form="spitzer",
                            ),
                        )
                    )
                cases.append(
                    (
                        "gfd6",
                        "qdsmc_xpoint_test.py",
                        f"xp_sde_{etag}",
                        dict(
                            conduction_op="sde",
                            eps_perp=eps,
                            b_noise=noise,
                            kappa_form="spitzer",
                        ),
                    )
                )
        # O-point arms + const-kappa sanity at the worst corner
        for op in ("fd", "sde"):
            cases.append(
                (
                    "gfd6",
                    "qdsmc_xpoint_test.py",
                    f"xp_{op}_opoint",
                    dict(
                        conduction_op=op,
                        blob_at="o",
                        eps_perp=1.0e-7,
                        b_noise=0.1,
                        kappa_form="spitzer",
                    ),
                )
            )
        cases.append(
            (
                "gfd6",
                "qdsmc_xpoint_test.py",
                "xp_fd_const_sanity",
                dict(
                    conduction_op="fd",
                    eps_perp=1.0e-7,
                    b_noise=0.1,
                    kappa_form="const",
                ),
            )
        )
        # conduction-OFF instrument floor
        cases.append(
            (
                "gfd6",
                "qdsmc_xpoint_test.py",
                "xp_floor_chi0",
                dict(conduction_op="fd", chi=0.0, b_noise=0.1),
            )
        )

    if "gfd6hi" in SECTIONS:
        # N=256 replicas of the worst corners (mitigation-scaling lesson:
        # verdicts must be re-read at 2x resolution)
        for lim, tint in (("smart", "ssprk2"), ("smart", "rkf45"), ("none", "ssprk2")):
            cases.append(
                (
                    "gfd6hi",
                    "qdsmc_xpoint_test.py",
                    f"xp256_fd_{lim}_{tint}",
                    dict(
                        ncell=256,
                        nsteps=256,
                        conduction_op="fd",
                        fd_limiter=lim,
                        fd_time=tint,
                        eps_perp=1.0e-7,
                        b_noise=0.1,
                        kappa_form="spitzer",
                    ),
                )
            )
        cases.append(
            (
                "gfd6hi",
                "qdsmc_xpoint_test.py",
                "xp256_sde",
                dict(
                    ncell=256,
                    nsteps=256,
                    conduction_op="sde",
                    eps_perp=1.0e-7,
                    b_noise=0.1,
                    kappa_form="spitzer",
                ),
            )
        )
        for arm in ("atrest", "blob"):
            for op in ("fd", "sde"):
                cases.append(
                    (
                        "gfd6hi",
                        "qdsmc_cliff_test.py",
                        f"cl256_{arm}_{op}",
                        dict(
                            ncell=256,
                            nsteps=256,
                            arm=arm,
                            ratio=20.0,
                            conduction_op=op,
                        ),
                    )
                )
            cases.append(
                (
                    "gfd6hi",
                    "qdsmc_cliff_test.py",
                    f"cl256_{arm}_floor",
                    dict(
                        ncell=256,
                        nsteps=256,
                        arm=arm,
                        ratio=20.0,
                        conduction_op="fd",
                        chi_dense=0.0,
                    ),
                )
            )

    if "a4" in SECTIONS:
        # m=4 conduction-anisotropy imprint (a4 star pattern): the FD
        # operator has no launch/deposit hop scale — expect a4 at the
        # iso-launch-corrected SDE level or below
        for op, iso in (("sde", 0), ("sde", 1), ("fd", 0)):
            cases.append(
                (
                    "a4",
                    "qdsmc_blob_iso_test.py",
                    f"a4_{op}_iso{iso}",
                    dict(conduction_op=op, iso_launch=iso),
                )
            )

    if "cliff" in SECTIONS:
        ops = [
            ("fd", dict()),
            ("fd", dict(fd_time="rkf45")),
            ("fd", dict(fd_limiter="none")),
            ("sde", dict(form="scatter")),
            ("sde", dict(form="fluxform")),
        ]
        for arm in ("atrest", "blob"):
            for ratio in (20.0, 100.0):
                # conduction-OFF floor: the transport/recovery arm mints
                # structure at the unresolved cliff on its own — the
                # conduction verdict is read AGAINST this floor
                cases.append(
                    (
                        "cliff",
                        "qdsmc_cliff_test.py",
                        f"cl_{arm}_r{ratio:.0f}_floor",
                        dict(arm=arm, ratio=ratio, conduction_op="fd", chi_dense=0.0),
                    )
                )
                for op, extra in ops:
                    sub = "_".join(f"{v}" for v in extra.values()) or "base"
                    cases.append(
                        (
                            "cliff",
                            "qdsmc_cliff_test.py",
                            f"cl_{arm}_r{ratio:.0f}_{op}_{sub}",
                            dict(arm=arm, ratio=ratio, conduction_op=op, **extra),
                        )
                    )

    return cases


def run_case(section, script, tag, kw):
    path = os.path.join(OUT, f"{tag}.npz")
    if os.path.exists(path):
        return tag, path, True
    log = os.path.join(OUT, f"{tag}.log")
    # qdsmc_blob_iso_test.py appends .npz to --out itself
    out_arg = path[:-4] if script == "qdsmc_blob_iso_test.py" else path
    cmd = [PY, os.path.join(HERE, script), "--out", out_arg]
    for key, val in kw.items():
        cmd.append("--" + key.replace("_", "-"))
        if isinstance(val, (list, tuple)):
            cmd += [str(v) for v in val]
        else:
            cmd.append(str(val))
    with open(log, "w") as fh:
        r = subprocess.run(cmd, env=ENV, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"[FAIL] {tag} exited {r.returncode} (see {log})", flush=True)
        return tag, None, False
    print(f"[done] {tag}", flush=True)
    return tag, path, True


def load(tag):
    path = os.path.join(OUT, f"{tag}.npz")
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def report_gfd3():
    print("\n## GFD3 wiggle leak (chi_perp_num/chi0; SDE reference rows)\n")
    print("| case | leak/chi0 | chi_par | sum drift | verdict |")
    print("|---|---|---|---|---|")
    for tag in sorted(
        t for t in os.listdir(OUT) if t.startswith("wig_") and t.endswith(".npz")
    ):
        d = load(tag[:-4])
        if d is None:
            continue
        leak = float(d["chi_perp_num"]) / float(d["chi0"])
        drift = (float(d["te_sum1"]) - float(d["te_sum0"])) / float(d["te_sum0"])
        v = ""
        if "_fd_" in tag and "floor" not in tag:
            v = "PASS" if abs(leak) < 2.3e-2 else "CHECK"
        print(
            f"| {tag[:-4]} | {leak:+.3e} | "
            f"{float(d['chi_par_meas']) / float(d['chi0']):.4f} | {drift:+.2e} | {v} |"
        )


def report_gfd4():
    print("\n## GFD4 Zeldovich front (reference relL2; undershoot = ringing)\n")
    print("| case | relL2 | front_exp | undershoot | vfront_max | verdict |")
    print("|---|---|---|---|---|---|")
    for tag in sorted(
        t for t in os.listdir(OUT) if t.startswith("zk_") and t.endswith(".npz")
    ):
        d = load(tag[:-4])
        if d is None:
            continue
        rel = float(d["rel_l2"])
        und = float(d["undershoot"])
        v = ""
        if "_fd_" in tag or tag.startswith("zk_fd"):
            ok = rel < 2.0e-2 and und > -1.0e-6
            v = "PASS" if ok else "CHECK"
        print(
            f"| {tag[:-4]} | {rel:.3e} | {float(d['front_exp']):.3f} "
            f"| {und:+.3e} | {float(d['vfront_max']):.3e} | {v} |"
        )


def report_gfd6():
    print("\n## GFD6 X-point/null maximum principle\n")
    print(
        "PASS (monotone arms) = overshoot <= "
        f"{TOL_EXTREMUM:g}, undershoot >= -{TOL_EXTREMUM:g}, "
        f"|sum drift| <= {TOL_SUM:g}; 'none' arms are the violation control.\n"
    )
    print("| case | overshoot | undershoot | peak rises | sum drift | verdict |")
    print("|---|---|---|---|---|---|")
    for tag in sorted(
        t for t in os.listdir(OUT) if t.startswith("xp") and t.endswith(".npz")
    ):
        d = load(tag[:-4])
        if d is None:
            continue
        ov = float(d["overshoot"])
        un = float(d["undershoot"])
        sd = float(d["sum_drift"])
        pr = int(d["peak_rise_steps"])
        monotone_arm = "_none_" not in tag and "_sde" not in tag[:6]
        ok = ov <= TOL_EXTREMUM and un >= -TOL_EXTREMUM and abs(sd) <= TOL_SUM
        if monotone_arm:
            v = "PASS" if ok else "VIOLATION"
        else:
            v = "clean" if ok else "violates (expected)"
        print(f"| {tag[:-4]} | {ov:+.3e} | {un:+.3e} | {pr} | {sd:+.2e} | {v} |")


def report_cliff():
    print("\n## Density-cliff arm (verdicts read against the conduction-OFF floor)\n")
    print(
        "PASS (fd smart arms) = metric within 3x the matching floor row "
        "(the transport arm mints structure at the unresolved cliff on its "
        "own; conduction must add ~nothing on top).\n"
    )
    print("| case | mint | overshoot | undershoot | sum(neTe) drift | verdict |")
    print("|---|---|---|---|---|---|")
    floors = {}
    for tag in sorted(
        t for t in os.listdir(OUT) if t.startswith("cl") and t.endswith("_floor.npz")
    ):
        d = load(tag[:-4])
        if d is not None:
            key = tag[: -len("_floor.npz")]
            floors[key] = dict(
                mint=float(d["mint"]),
                ov=float(d["overshoot"]),
                un=float(d["undershoot"]),
            )
    for tag in sorted(
        t for t in os.listdir(OUT) if t.startswith("cl") and t.endswith(".npz")
    ):
        d = load(tag[:-4])
        if d is None:
            continue
        mint = float(d["mint"])
        ov = float(d["overshoot"])
        un = float(d["undershoot"])
        ud = float(d["usum_drift"])
        arm = str(d["arm"])
        fl = next(
            (
                floors[k]
                for k in sorted(floors, key=len, reverse=True)
                if tag.startswith(k + "_")
            ),
            None,
        )
        if tag.endswith("_floor.npz"):
            v = "floor"
        elif fl is None:
            v = "no-floor"
        else:
            if arm == "atrest":
                ok = mint <= max(TOL_MINT, 3.0 * fl["mint"])
            else:
                ok = ov <= max(TOL_EXTREMUM, 3.0 * fl["ov"]) and un >= min(
                    -TOL_EXTREMUM, 3.0 * fl["un"]
                )
            fd_clean = "_fd_" in tag and "none" not in tag
            v = (
                ("PASS" if ok else "VIOLATION")
                if fd_clean
                else ("clean" if ok else "ref-violates")
            )
        print(f"| {tag[:-4]} | {mint:.3e} | {ov:+.3e} | {un:+.3e} | {ud:+.2e} | {v} |")


def report_a4():
    print("\n## a4 conduction-anisotropy imprint (m=4 star pattern)\n")
    print("| case | a4 | r* | sum drift | note |")
    print("|---|---|---|---|---|")
    vals = {}
    for tag in sorted(
        t for t in os.listdir(OUT) if t.startswith("a4_") and t.endswith(".npz")
    ):
        d = load(tag[:-4])
        if d is None:
            continue
        vals[tag[:-4]] = float(d["a4"])
        print(
            f"| {tag[:-4]} | {float(d['a4']):.4e} | {float(d['rstar']):.3f} "
            f"| {float(d['drift']):+.2e} | |"
        )
    if "a4_fd_iso0" in vals and "a4_sde_iso1" in vals:
        ok = vals["a4_fd_iso0"] <= 1.5 * vals["a4_sde_iso1"]
        print(
            f"\nFD a4 = {vals['a4_fd_iso0']:.3e} vs corrected-SDE "
            f"{vals['a4_sde_iso1']:.3e}: {'PASS' if ok else 'CHECK'} "
            "(FD must be at or below the iso-launch-corrected level)"
        )


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = build_cases()
    todo = [c for c in cases if not os.path.exists(os.path.join(OUT, c[2] + ".npz"))]
    print(f"[battery] {len(cases)} cases ({len(todo)} to run), jobs={args.jobs}")
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_case, *c) for c in todo]
        for f in cf.as_completed(futs):
            f.result()
    if "gfd3" in SECTIONS:
        report_gfd3()
    if "gfd4" in SECTIONS:
        report_gfd4()
    if "gfd6" in SECTIONS:
        report_gfd6()
    if "cliff" in SECTIONS:
        report_cliff()
    if "a4" in SECTIONS:
        report_a4()


if __name__ == "__main__":
    main()
