#!/usr/bin/env python3
"""Strict solver, finite-field and analytic-limit gates for the Darwin split."""

import sys
from pathlib import Path

import numpy as np
import yt

mode = sys.argv[1]
check_vacuum_gauge = mode.startswith("pc_")
rtol = 1e-8 if check_vacuum_gauge else 1e-6
mode = mode.removeprefix("pc_")
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
atol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
yt.set_log_level(50)
rows = np.loadtxt("newton.txt", ndmin=2)
assert rows.shape[1] == 18 and np.isfinite(rows).all()
status = rows[:, 10]
assert np.all(rows[:, 9] == status), "public and convergence status differ"
assert np.all(((status == 3) & (rows[:, 5] < rtol)) |
              ((status == 2) & (rows[:, 4] < atol))), "nonlinear convergence gate"
plots = sorted(p for p in Path("diags").glob("gate[0-9]*")
               if len(p.name) == 10 and p.name[4:].isdigit())
assert plots[-1].name == f"gate{steps:06d}", "required steps did not complete"
ds = yt.load(str(plots[-1]))
grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
for field in ds.field_list:
    assert np.isfinite(np.asarray(grid[field])).all(), field
if mode == "energy":
    # The periodic transverse wave has uniform density/temperature in this
    # short limit. Spurious longitudinal probes must not heat the electrons.
    te = np.asarray(grid["boxlib", "Te"])
    initial = yt.load(str(plots[0]))
    g0 = initial.covering_grid(0, initial.domain_left_edge, initial.domain_dimensions)
    te0 = np.asarray(g0["boxlib", "Te"])
    assert np.min(te) > 0
    assert np.max(np.abs(te-te0)) / np.max(te0) < 1e-8
elif mode in ("vacuum", "vacuum_energy"):
    # Boundary A imposes dBz/dt = 5e5 T/s. The entire volume must receive
    # the flux at eta <= 1e-6, beyond what resistive diffusion can deliver.
    expected = 0.1 + 5e5 * float(ds.current_time)
    error = np.max(np.abs(np.asarray(grid["boxlib", "Bz"])-expected))
    assert error < 3e-7, (error, expected)
    if mode == "vacuum_energy":
        te = np.asarray(grid["boxlib", "Te"])
        initial = yt.load(str(plots[0]))
        g0 = initial.covering_grid(0, initial.domain_left_edge, initial.domain_dimensions)
        te0 = np.asarray(g0["boxlib", "Te"])
        assert np.min(te) > 0
        # Cold ions and a short smooth drive must not produce the artificial
        # 16-fold temperature increase caused by sharing the field floor.
        assert np.max(te) < 1.01 * np.max(te0)
elif mode == "eb_vacuum":
    initial = yt.load(str(plots[0]))
    g0 = initial.covering_grid(0, initial.domain_left_edge, initial.domain_dimensions)
    bz = np.asarray(grid["boxlib", "Bz"])
    bz0 = np.asarray(g0["boxlib", "Bz"])
    x = np.asarray(grid["index", "x"])
    y = np.asarray(grid["index", "y"])
    radius = np.sqrt(x*x+y*y)
    inside = radius < 0.03
    exterior = (radius > 0.13) & (np.abs(x) < 0.2) & (np.abs(y) < 0.2)
    assert np.max(np.abs((bz-bz0)[inside])) < 1e-12
    assert np.ptp(bz[exterior]) < 3e-7
    assert np.min(bz[exterior]-bz0[exterior]) > 0.9*5e5*float(ds.current_time)
    dx = np.asarray(ds.domain_width) / ds.domain_dimensions
    flux_change = np.sum(bz-bz0, axis=(0, 1)) * dx[0]*dx[1]
    expected = 5e5*float(ds.current_time)*float(ds.domain_width[0]*ds.domain_width[1])
    # Discrete Stokes gate for the existing masked curl (not a claim about
    # cut-cell metric accuracy): the outer boundary A supplies this flux.
    assert np.max(np.abs(flux_change-expected)) < 1e-8*expected
    assert np.min(np.asarray(grid["boxlib", "Te"])[exterior]) > 0
else:
    assert mode == "eb"
if check_vacuum_gauge and mode in ("vacuum_energy", "eb_vacuum"):
    # These drives and initial states are invariant in z. A curl-free electric
    # error can leave every magnetic/flux gate above unchanged; test E directly.
    # 0.01 V/m is <2e-7 of the imposed in-plane inductive field scale.
    ex = np.asarray(grid["boxlib", "Ex"])
    ey = np.asarray(grid["boxlib", "Ey"])
    ez = np.asarray(grid["boxlib", "Ez"])
    assert np.max(np.abs(ez)) < 1e-2, "spurious longitudinal Ez"
    assert np.max(np.ptp(ex, axis=2)) < 1e-2, "Ex lost z symmetry"
    assert np.max(np.ptp(ey, axis=2)) < 1e-2, "Ey lost z symmetry"
    if mode == "vacuum_energy":
        x = np.asarray(grid["index", "x"])
        y = np.asarray(grid["index", "y"])
        assert np.max(np.abs(ex - 2.5e5*y)) < 1e-2, "incorrect inductive Ex"
        assert np.max(np.abs(ey + 2.5e5*x)) < 1e-2, "incorrect inductive Ey"
print("Darwin split", mode, "checks passed")
