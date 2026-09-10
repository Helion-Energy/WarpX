#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Circuit probe region mask / physical-share weight / region report.

Reads the region report (diags/circuit_probe_region.txt) and the
CircuitCoupling reduced diagnostic (diags/reducedfiles/ccdiag.txt) of
inputs_test_rz_theta_implicit_mhd_circuit_probe_region and asserts,
per arm (sys.argv[1]; sys.argv[2] = the wall polyline CSV):

common (every arm)
  - the report's r_wall(z_j) table equals numpy.interp of the polyline
    BITWISE at every z node (the documented rule, both clamps and the
    sloped section);
  - lambda_total == interior + band + exterior and == w_plasma + w_mixed
    + w_boost to roundoff, on every coil row;
  - the accepting row's lambda_used equals the reduced diagnostic's
    lambda of the same step (the coupler's committed value) to roundoff.

domain_outside (probe_region = domain, outside bump, uniform plasma)
  - lambda_used == lambda_total (roundoff): the mask is off;
  - the FIRST row of step 1 (the pristine initial state, J exactly zero
    away from the bump): interior == band == 0.0 exactly, exterior ==
    total, rms_interior == rms_band == 0.0 exactly, rms_exterior > 0;
  - the boost is off: w == 1 exactly, so w_plasma == total and weighted
    == total bitwise (J * 1.0 == J), w_mixed == w_boost == 0.0 exactly.

wall_outside (probe_region = wall_interior, outside bump)
  - lambda_used == interior + band (roundoff) on every row;
  - the FIRST row of step 1: lambda_used == 0.0 EXACTLY (every inside
    node carries exactly zero current; masked nodes have weight exactly
    0), while lambda_total != 0;
  - accepting rows: |lambda_used| < 0.5 |lambda_total| (the exterior
    current still carries the majority after the evolution).

wall_inside (probe_region = wall_interior, inside bump)
  - lambda_used == interior + band (roundoff) on every row;
  - the FIRST row of step 1: band == 0.0 exactly and rms_band == 0.0
    exactly (the bump's nodes are strictly interior). The exterior is
    NOT exactly zero: the Green's open boundary reconstructs the r_hi
    ghost from the interior current, and the loaded compact bump is not
    the free-space field of its own curl, so the first iterate already
    carries a thin current sheet at the outermost node (the open face)
    -- an initial-condition artifact the mask correctly EXCLUDES from
    the back-EMF. Asserted small: |exterior| < 0.1 |total| per coil and
    rms_exterior < 1e-2 rms_interior.

weight_boost (probe_weight = physical_share, outside bump in the vacuum)
  - the FIRST row of step 1: w_boost == total (roundoff), w_plasma ==
    w_mixed == 0.0 exactly (every current-carrying node, the open-face
    sheet included, is boost-dominated), lambda_used / lambda_total ==
    W_BOOST (the analytic vacuum weight of the deck, to 1 %): the mock
    current contributes only its physical share;
  - every row: lambda_used == lambda_weighted (roundoff).

weight_plasma (probe_weight = physical_share, inside bump in the plasma)
  - the FIRST row of step 1: w_plasma == interior and w_boost ==
    exterior (roundoff; the bump is plasma, the open-face sheet sits in
    the vacuum), w_mixed == 0.0 exactly, and lambda_used == w_plasma *
    (1 - O(1e-6)) + W_BOOST * w_boost to 1e-5 of the total (w = 1 -
    eta_vac^2 / (2 eta^2) in the plasma);
  - every row: lambda_used == lambda_weighted (roundoff).
"""

import sys

import numpy as np

ARM = sys.argv[1] if len(sys.argv) > 1 else "domain_outside"
POLYLINE = sys.argv[2] if len(sys.argv) > 2 else "wall_probe_region_slope.csv"
REPORT = "diags/circuit_probe_region.txt"
REDUCED = "diags/reducedfiles/ccdiag.txt"
RTOL = 1.0e-12

# The boost arm's expected weight at the vacuum nodes: every node of the
# outside bump has all four surrounding cells at rho_vac = 1e-3 rho0 = the
# Ohm reference density (hybrid_pic_model.n_floor), so
#     eta_vac = mu0 D (rho_ref / rho_vac)^2 = mu0 * 1e3 * 1 = 1.2566e-3 ohm m,
#     w = eta_phys / sqrt(eta_phys^2 + eta_vac^2) = 1e-6 / 1.2566e-3 = 7.958e-4
# (the division guard at the positivity floor, 1e-5 rho0, perturbs this at
# the 1e-4 level); the weighted linkage of the bump is w times the
# unweighted one.
MU0 = 4.0e-7 * np.pi
ETA_PHYS = 1.0e-6
ETA_VAC = MU0 * 1.0e3
W_BOOST = ETA_PHYS / np.sqrt(ETA_PHYS**2 + ETA_VAC**2)

# report columns after "kind step t coil"
COLUMNS = [
    "total",
    "interior",
    "band",
    "exterior",
    "w_plasma",
    "w_mixed",
    "w_boost",
    "weighted",
    "used",
    "rms_interior",
    "rms_band",
    "rms_exterior",
    "weighted_inside",
]


def close(a, b, scale, rtol=RTOL):
    return abs(a - b) <= rtol * max(abs(scale), 1.0e-300)


def load_report(path):
    rows = []
    table = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# r_wall_table"):
                    table = line
                continue
            tok = line.split()
            assert len(tok) == 4 + len(COLUMNS), f"malformed row: {line}"
            vals = dict(zip(COLUMNS, (float(v) for v in tok[4:])))
            rows.append((tok[0], int(tok[1]), float(tok[2]), tok[3], vals))
    assert table is not None, "the report header lacks the r_wall table"
    return rows, table


def check_rule(table_line, polyline):
    """The report's r_wall(z_j) table vs numpy.interp, bitwise."""
    head, values = table_line.split("values:")
    meta = dict(kv.split("=") for kv in head.split()[2:])
    j_lo = int(meta["j_lo"])
    z_lo = float(meta["z_lo"])
    dz = float(meta["dz"])
    r_table = np.array([float(v) for v in values.split()])
    poly = np.loadtxt(polyline, delimiter=",", skiprows=1)
    z_poly, r_poly = poly[:, 0], poly[:, 1]
    j = np.arange(j_lo, j_lo + r_table.size)
    z_node = z_lo + j * dz
    r_numpy = np.interp(z_node, z_poly, r_poly)
    mismatch = np.nonzero(r_table != r_numpy)[0]
    assert mismatch.size == 0, (
        f"r_wall table differs from numpy.interp at nodes {mismatch.tolist()}: "
        f"{r_table[mismatch]} vs {r_numpy[mismatch]}"
    )
    # the polyline ends inside the domain: both clamps and the slope must
    # actually be exercised by this fixture
    assert z_node[0] < z_poly[0] and z_node[-1] > z_poly[-1], "clamps not exercised"
    assert np.unique(r_table).size > 2, "the sloped section was not sampled"
    print(f"r_wall table: {r_table.size} nodes bitwise equal to numpy.interp")


def load_reduced(path):
    with open(path) as f:
        header = f.readline()
    names = [h.split("]")[1].split("(")[0] for h in header.lstrip("#").split()]
    data = np.atleast_2d(np.loadtxt(path))
    return names, data


def main():
    rows, table = load_report(REPORT)
    check_rule(table, POLYLINE)
    assert rows, "empty report"
    coils = sorted({r[3] for r in rows})
    print(f"arm = {ARM}; coils in the report: {coils}")
    kinds = {r[0] for r in rows}
    assert kinds == {"first", "accept"}, f"unexpected row kinds {kinds}"

    # common: the two partitions of the total
    for kind, step, t, coil, v in rows:
        s_radial = v["interior"] + v["band"] + v["exterior"]
        s_class = v["w_plasma"] + v["w_mixed"] + v["w_boost"]
        assert close(v["total"], s_radial, v["total"]), (
            f"{kind} step {step} {coil}: total {v['total']!r} != interior + band + "
            f"exterior {s_radial!r}"
        )
        assert close(v["total"], s_class, v["total"]), (
            f"{kind} step {step} {coil}: total {v['total']!r} != w_plasma + w_mixed + "
            f"w_boost {s_class!r}"
        )

    # common: the accepting row's used value is the committed value the
    # reduced diagnostic recorded for that step
    names, red = load_reduced(REDUCED)
    accept_rows = [r for r in rows if r[0] == "accept"]
    assert accept_rows, "no accepting rows"
    for kind, step, t, coil, v in accept_rows:
        col = names.index(f"lambda_{coil}")
        row = np.nonzero(np.isclose(red[:, 1], t, rtol=1e-12, atol=0.0))[0]
        assert row.size == 1, f"no reduced-diag row at t = {t!r}"
        lam_red = red[row[0], col]
        assert close(v["used"], lam_red, v["total"]), (
            f"accept step {step} {coil}: used {v['used']!r} != reduced diag {lam_red!r}"
        )

    first_rows = [r for r in rows if r[0] == "first"]
    t_first = min(r[2] for r in first_rows)
    step1 = [r for r in first_rows if r[2] == t_first]
    assert len(step1) == len(coils)
    rms_int = step1[0][4]["rms_interior"]
    rms_band = step1[0][4]["rms_band"]
    rms_ext = step1[0][4]["rms_exterior"]
    print(
        f"step-1 first-iterate J_theta RMS: interior {rms_int:.6e}, band {rms_band:.6e}, "
        f"exterior {rms_ext:.6e}"
    )
    for kind, step, t, coil, v in step1:
        print(
            f"  step-1 first-iterate {coil}: used {v['used']:.10e} total {v['total']:.10e} "
            f"interior {v['interior']:.3e} band {v['band']:.3e} exterior {v['exterior']:.3e} "
            f"w_plasma {v['w_plasma']:.3e} w_mixed {v['w_mixed']:.3e} w_boost {v['w_boost']:.3e} "
            f"weighted {v['weighted']:.10e}"
        )
        assert v["total"] != 0.0, f"{coil}: the synthetic current links nothing"

    if ARM == "domain_outside":
        for kind, step, t, coil, v in rows:
            assert close(v["used"], v["total"], v["total"]), f"{coil}: mask leaked in"
            # boost off: w == 1 exactly everywhere
            assert v["w_plasma"] == v["total"] and v["w_mixed"] == 0.0 and v["w_boost"] == 0.0
            assert v["weighted"] == v["total"], f"{coil}: J * 1.0 != J"
        for kind, step, t, coil, v in step1:
            assert v["interior"] == 0.0 and v["band"] == 0.0, f"{coil}: interior current"
            assert v["exterior"] == v["total"]
        assert rms_int == 0.0 and rms_band == 0.0 and rms_ext > 0.0
    elif ARM == "wall_outside":
        for kind, step, t, coil, v in rows:
            assert close(v["used"], v["interior"] + v["band"], v["total"]), (
                f"{kind} step {step} {coil}: masked value {v['used']!r} != interior + band"
            )
        for kind, step, t, coil, v in step1:
            assert v["used"] == 0.0, (
                f"{coil}: masked linkage of an exterior current is {v['used']!r}, not 0.0"
            )
            assert v["total"] != 0.0
        for kind, step, t, coil, v in accept_rows:
            print(f"  accept step {step} {coil}: masked share {v['used'] / v['total']:.4f}")
            assert abs(v["used"]) < 0.5 * abs(v["total"]), (
                f"accept step {step} {coil}: the masked linkage {v['used']!r} is not a "
                f"minority of the total {v['total']!r}"
            )
    elif ARM == "wall_inside":
        for kind, step, t, coil, v in rows:
            assert close(v["used"], v["interior"] + v["band"], v["total"]), (
                f"{kind} step {step} {coil}: masked value {v['used']!r} != interior + band"
            )
        for kind, step, t, coil, v in step1:
            assert v["band"] == 0.0, f"{coil}: band current {v['band']!r}"
            share = abs(v["exterior"]) / abs(v["total"])
            print(f"  {coil}: open-face (exterior) share of the total {share:.3e}")
            assert share < 0.1, f"{coil}: the open-face sheet links {share:.3e} of the total"
        assert rms_band == 0.0 and rms_int > 0.0
        assert rms_ext < 1.0e-2 * rms_int, f"exterior RMS {rms_ext!r} vs interior {rms_int!r}"
    elif ARM in ("weight_boost", "weight_plasma"):
        for kind, step, t, coil, v in rows:
            assert close(v["used"], v["weighted"], v["total"]), (
                f"{kind} step {step} {coil}: weighted value {v['used']!r} != lambda_weighted "
                f"{v['weighted']!r}"
            )
        for kind, step, t, coil, v in step1:
            if ARM == "weight_boost":
                assert v["w_plasma"] == 0.0 and v["w_mixed"] == 0.0, f"{coil}: non-boost current"
                assert close(v["w_boost"], v["total"], v["total"])
                ratio = v["used"] / v["total"]
                print(
                    f"  {coil}: weighted / unweighted linkage of the boost-dominated current "
                    f"{ratio:.6e} (expected w = {W_BOOST:.6e})"
                )
                assert abs(ratio - W_BOOST) <= 1.0e-2 * W_BOOST, (
                    f"{coil}: the boost-dominated current's weighted share {ratio!r} is not "
                    f"the vacuum weight {W_BOOST!r}"
                )
            else:
                assert v["w_mixed"] == 0.0, f"{coil}: mixed-class current {v['w_mixed']!r}"
                assert close(v["w_plasma"], v["interior"], v["total"]), f"{coil}: plasma != interior"
                assert close(v["w_boost"], v["exterior"], v["total"]), f"{coil}: boost != exterior"
                expected = v["w_plasma"] + W_BOOST * v["w_boost"]
                deficit = 1.0 - v["used"] / v["w_plasma"]
                print(
                    f"  {coil}: plasma-current weight deficit 1 - used/w_plasma = {deficit:.3e}; "
                    f"used - (w_plasma + W_BOOST * w_boost) = {v['used'] - expected:.3e}"
                )
                assert abs(v["used"] - expected) <= 1.0e-5 * abs(v["total"]), (
                    f"{coil}: the weighted linkage {v['used']!r} is not w_plasma + W_BOOST * "
                    f"w_boost = {expected!r}"
                )
    else:
        raise SystemExit(f"unknown arm {ARM}")
    print("PASS")


if __name__ == "__main__":
    main()
