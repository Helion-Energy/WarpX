#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Exercise the Hall-MHD triangular preconditioner with evolving ions."""

import re
import sys
from pathlib import Path

import numpy as np
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


def input_value(name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\s]+)", used_inputs, re.MULTILINE)
    assert match is not None
    return match.group(1)


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])
used_inputs = Path("warpx_used_inputs").read_text()

initial_density = initial["boxlib", "implicit_mhd_mass_density"].value
final_density = final["boxlib", "implicit_mhd_mass_density"].value
final_energy = final["boxlib", "implicit_mhd_electron_energy"].value
initial_momentum = initial["boxlib", "implicit_mhd_momentum_density"].value
final_momentum = final["boxlib", "implicit_mhd_momentum_density"].value
initial_magnetic_field = np.stack(
    [initial["boxlib", component].value for component in ("Bx", "By")]
)
final_magnetic_field = np.stack(
    [final["boxlib", component].value for component in ("Bx", "By")]
)

# The ion fluid is initially at rest. Its nonzero response verifies that this
# test enters the evolving-ion residual rather than the frozen-ion/eMHD path.
np.testing.assert_array_equal(initial_momentum, 0.0)
assert np.max(np.abs(final_momentum)) > 1.0e-8
assert np.linalg.norm(final_magnetic_field - initial_magnetic_field) > (
    0.25 * np.linalg.norm(initial_magnetic_field)
)

# The periodic continuity equation conserves total ion mass. The weakly
# nonlinear broadband solve must also preserve a finite, admissible state.
np.testing.assert_allclose(
    np.sum(final_density),
    np.sum(initial_density),
    rtol=5.0e-13,
    atol=0.0,
)
for state in (final_density, final_energy, final_momentum, final_magnetic_field):
    assert np.all(np.isfinite(state))
assert np.min(final_density) > 0.0
assert np.min(final_energy) > 0.0

assert input_value("jacobian.pc_type") == "pc_mhd_block"
assert input_value("implicit_mhd.evolve_ion_fluid") == "true"
assert input_value("hybrid_pic_model.include_hall_term") == "true"
assert input_value("pc_mhd_block.include_ideal_mhd_coupling") == "true"
assert input_value("pc_mhd_block.include_hall_mhd_coupling") == "true"
assert int(input_value("pc_mhd_block.field_iterations")) == 2
assert int(input_value("pc_mhd_block.fluid_iterations")) == 1

newton_history = np.atleast_2d(np.loadtxt("diags/newton.txt"))
last_solve = newton_history[-1]
assert 1 <= last_solve[2] <= 8
assert (last_solve[4] <= 1.1e-12) or (last_solve[5] <= 1.1e-10)
# The triangular Hall path takes 200 total iterations on the CPU reference
# build. The failed ideal-wave/Hall factorization reached the 800-iteration
# cap, so retain platform margin while guarding the chosen factorization.
assert 0 < last_solve[7] <= 250

print(f"maximum ion momentum={np.max(np.abs(final_momentum)):.12e}")
print(f"Newton iterations={int(last_solve[2])}")
print(f"GMRES iterations={int(last_solve[7])}")
