#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""The change-of-variables pedestal injects nothing: ledger rows of zeros, injected fields zero.

The minimal contract of every change-of-variables run (here the single-box non-periodic RZ
column that serves as the fused-residual identity reference): the pedestal ledger has one
row per step with 0 raised cells and zero totals, the implicit_mhd_pedestal_injected_* fields
are identically zero, every cell sits at or above the background density and the run reached
its final plotfile.

Usage: analysis_mhd_pedestal_cov_ledger_zero.py <final_plotfile> <ledger>
"""

import re
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(50)

final_plotfile, ledger_file = sys.argv[1:3]
ds = yt.load(final_plotfile)
data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
names = [name for kind, name in ds.field_list if kind == "boxlib"]
fields = {name: np.array(data["boxlib", name].value) for name in names}

used_inputs = Path("warpx_used_inputs").read_text()
max_step = int(re.search(r"^max_step\s*=\s*(\d+)", used_inputs, re.MULTILINE).group(1))
ledger = np.loadtxt(ledger_file, ndmin=2)
print(f"ledger rows {ledger.shape[0]} (steps {max_step}); last row {ledger[-1].tolist()}")
assert ledger.shape == (max_step, 5), ledger.shape
assert np.all(ledger[:, 1] == 0), ledger[:, 1]
assert np.all(ledger[:, 2:] == 0.0), ledger[-1]
for name in ("implicit_mhd_pedestal_injected_mass",
             "implicit_mhd_pedestal_injected_electron_energy",
             "implicit_mhd_pedestal_injected_ion_energy"):
    assert np.all(fields[name] == 0.0), name
rho = fields["implicit_mhd_mass_density"]
print(f"density min {rho.min():.6e}, max {rho.max():.6e}")
print("change of variables: nothing injected")
