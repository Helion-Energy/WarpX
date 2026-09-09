#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Emission gate of the PICMI kwarg ``resistive_direct_cudss_options``.

The direct resistive block reads ``implicit_mhd.resistive_direct_cudss_options``
as a list of strings; ParmParse rejects an unquoted ``name=value`` token
("no values for definition"), so every option must reach the input deck as
its own double-quoted token. This runs without the compiled bindings: it
drives ``pywarpx.picmi.ThetaImplicitMHDEvolveScheme`` through the same
emission path the deck uses and checks the ``implicit_mhd`` bucket's
attribute list for the list form, the string form, the space-separated
string form, and the unset default; it also splits the emitted right-hand
side the way ParmParse does and checks every token is a name=value pair.
"""

import shlex

import pywarpx
from pywarpx import picmi


def emitted(**kwargs):
    bucket = pywarpx.warpx.get_bucket("implicit_mhd")
    bucket.argvattrs.clear()
    scheme = picmi.ThetaImplicitMHDEvolveScheme(
        nonlinear_solver=picmi.NewtonNonlinearSolver(),
        mass_density="1.0",
        electron_pressure="1.0",
        reference_mass_density=1.0,
        reference_magnetic_field=1.0,
        **kwargs,
    )
    scheme.solver_scheme_initialize_inputs()  # the deck's emission path
    lines = [
        line
        for line in bucket.attrlist()
        if line.startswith("implicit_mhd.resistive_direct_cudss_options ")
    ]
    assert len(lines) <= 1, lines
    return lines[0] if lines else None


def right_hand_side(line):
    name, equals, value = line.partition(" = ")
    assert equals and name == "implicit_mhd.resistive_direct_cudss_options", line
    return value


for kwargs, expected in (
    ({"resistive_direct_cudss_options": ["pivot_type=none", "nd_nlevels=16"]},
     '"pivot_type=none" "nd_nlevels=16"'),
    ({"resistive_direct_cudss_options": "pivot_type=none"}, '"pivot_type=none"'),
    ({"resistive_direct_cudss_options": "pivot_type=none nd_nlevels=16"},
     '"pivot_type=none" "nd_nlevels=16"'),
    ({"resistive_direct_cudss_options": ("pivot_type=none",)}, '"pivot_type=none"'),
):
    line = emitted(**kwargs)
    assert line is not None, kwargs
    value = right_hand_side(line)
    assert value == expected, (value, expected)
    # ParmParse tokenization: every token is a quoted name=value pair
    tokens = shlex.split(value)
    assert tokens and all("=" in token and " " not in token for token in tokens), tokens
    assert all(f'"{token}"' in value for token in tokens), (value, tokens)
    print(f"ok: {kwargs} -> {line}")

assert emitted() is None, "the knob must not be emitted when unset"
print("ok: unset kwarg emits nothing")
