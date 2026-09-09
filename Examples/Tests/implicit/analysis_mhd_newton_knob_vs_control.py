#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""A Newton-count knob against its control run on the same deck.

Used for the theta-implicit MHD solver knobs that change the Newton
iteration's path but must not change the converged physics beyond a stated
class: the linear predictor (implicit_mhd.newton_predictor = linear) and the
smooth-floor widths (implicit_mhd.pressure_floor_width_factor,
implicit_mhd.dual_energy_fk_width). The control is the same deck without the
knob (a test dependency). Gates, from the command line:

  --control <dir>            the control test directory (its diags/newton.txt
                             and the plotfile named like this run's)
  --newton-max <ratio>       Newton total of this run <= ratio x control
  --newton-min <ratio>       Newton total >= ratio x control (a knob that
                             must NOT help this much would be suspicious;
                             use with --newton-max to bracket)
  --newton-reduction <ratio> Newton total <= ratio x control (the knob MUST
                             reduce the work by this factor: the predictor)
  --gmres-max <ratio>        linear iterations total <= ratio x control
  --physics <tol>            relative L2 of every fluid field of the final
                             plotfile within tol of the control
  --fields a,b,c             the fields to compare (default: the MHD fluid
                             blocks present in both plotfiles)

Usage: analysis_mhd_newton_knob_vs_control.py <final plotfile> --control <dir> [gates]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

parser = argparse.ArgumentParser()
parser.add_argument("plotfile")
parser.add_argument("--control", required=True)
parser.add_argument("--newton-max", type=float, default=None)
parser.add_argument("--newton-min", type=float, default=None)
parser.add_argument("--newton-reduction", type=float, default=None)
parser.add_argument("--gmres-max", type=float, default=None)
parser.add_argument("--physics", type=float, default=None)
parser.add_argument("--fields", type=str, default=None)
parser.add_argument("--require-knob", type=str, default=None,
                    help="a 'key = value' line that must appear in warpx_used_inputs")
args = parser.parse_args()

if args.require_knob:
    used = Path("warpx_used_inputs").read_text()
    key, value = [t.strip() for t in args.require_knob.split("=", 1)]
    match = re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", used, re.MULTILINE)
    assert match is not None and match.group(1).strip('"') == value, (key, value, match)


def load_newton(directory):
    # newton.txt appends across reruns: keep the last session.
    sessions, current = [], []
    for line in open(f"{directory}/diags/newton.txt"):
        if line.startswith("#") or not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if current and values[0] <= current[-1][0]:
            sessions.append(current)
            current = []
        current.append(values)
    sessions.append(current)
    return np.array(sessions[-1])


mine = load_newton(".")
control = load_newton(args.control)
assert len(mine) == len(control), (len(mine), len(control))
newton_mine, newton_control = int(mine[:, 2].sum()), int(control[:, 2].sum())
gmres_mine, gmres_control = int(mine[:, 6].sum()), int(control[:, 6].sum())
print(f"Newton total {newton_mine} vs control {newton_control} (ratio {newton_mine / newton_control:.3f}); "
      f"linear iterations {gmres_mine} vs {gmres_control} (ratio {gmres_mine / gmres_control:.3f})")
if args.newton_max is not None:
    assert newton_mine <= args.newton_max * newton_control, (newton_mine, newton_control, args.newton_max)
if args.newton_min is not None:
    assert newton_mine >= args.newton_min * newton_control, (newton_mine, newton_control, args.newton_min)
if args.newton_reduction is not None:
    assert newton_mine <= args.newton_reduction * newton_control, (
        f"the knob did not reduce the Newton work: {newton_mine} vs control {newton_control} "
        f"(required <= {args.newton_reduction} x)")
if args.gmres_max is not None:
    assert gmres_mine <= args.gmres_max * gmres_control, (gmres_mine, gmres_control, args.gmres_max)


def fields(plotfile, names):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
    available = {name for kind, name in ds.field_list if kind == "boxlib"}
    if names is None:
        names = sorted(n for n in available if n.startswith("implicit_mhd_"))
    return {n: np.array(grid["boxlib", n]).ravel() for n in names if n in available}


if args.physics is not None:
    names = args.fields.split(",") if args.fields else None
    mine_fields = fields(args.plotfile, names)
    control_fields = fields(f"{args.control}/{args.plotfile}", list(mine_fields))
    assert mine_fields, "no fluid fields found in the plotfile"
    for name in sorted(mine_fields):
        ref = control_fields[name]
        scale = np.linalg.norm(ref)
        diff = np.linalg.norm(mine_fields[name] - ref) / scale if scale > 0 else np.linalg.norm(mine_fields[name])
        print(f"{name:40s} relative L2 difference from the control {diff:.3e}")
        assert diff <= args.physics, (name, diff, args.physics)
print("newton knob vs control: PASS")
