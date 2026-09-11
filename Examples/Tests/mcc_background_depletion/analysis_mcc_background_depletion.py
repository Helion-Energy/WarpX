#!/usr/bin/env python3
"""Conservation check for the depleting MCC background.

Each electron-impact ionization event removes one neutral from the
mesh-resident background and creates one ion, so the neutral weight missing
from the background must equal the ion weight present. The cell volume is the
same for both sums and cancels, so the sums are compared directly.

This is the identity that a mismatch between the shape factor used to gather
the background density and the one used to deposit its depletion would break,
and that a missing guard-cell fold would break at the tile boundaries. It
holds exactly only while gas remains everywhere: the clamp at zero is a
deliberate non-conservation, so that is checked first.
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

from pywarpx import picmi

q_e = picmi.constants.q_e

# must match the input file
n_background_0 = 1.0e18

ts = OpenPMDTimeSeries("diags/diag2/")
iteration = ts.iterations[-1]

n_gas, info = ts.get_field(field="n_background_gas", iteration=iteration)
rho_ions, _ = ts.get_field(field="rho_ions", iteration=iteration)

# The clamp at zero is a legitimate loss of conservation, so the identity
# below is only meaningful while every cell still holds gas.
assert n_gas.min() > 0.0, (
    "the background was exhausted somewhere, so the conservation identity "
    "does not apply; lower n_elec or raise n_gas in the input file"
)

neutrals_removed = np.sum(n_background_0 - n_gas)
ions_created = np.sum(rho_ions) / q_e

# Guard against the test passing by simply not depleting anything.
depleted_fraction = neutrals_removed / (n_background_0 * n_gas.size)
assert depleted_fraction > 0.01, (
    f"only {depleted_fraction:.3%} of the fill was consumed, which is too "
    "little for this test to demonstrate anything"
)

relative_error = abs(neutrals_removed - ions_created) / ions_created

print(f"fraction of fill consumed : {depleted_fraction:.4%}")
print(f"neutrals removed          : {neutrals_removed:.10e}")
print(f"ions created              : {ions_created:.10e}")
print(f"relative difference       : {relative_error:.3e}")

assert relative_error < 1.0e-9, (
    f"neutrals removed and ions created differ by {relative_error:.3e}: the "
    "depletion deposit and the background gather do not agree"
)

# Both quantities above are numbers per cell volume, which is what lets the
# volume cancel between them. Confirm that reading against the ion
# macroparticle weights, so that the check above cannot be satisfied by two
# field sums being wrong in the same way.
cell_volume = info.dx * info.dz
ion_weight = ts.get_particle(var_list=["w"], species="ions", iteration=iteration)[0]
weight_error = abs(ions_created * cell_volume - ion_weight.sum()) / ion_weight.sum()

print(f"ion weight from rho       : {ions_created * cell_volume:.10e}")
print(f"ion weight from particles : {ion_weight.sum():.10e}")
print(f"relative difference       : {weight_error:.3e}")

assert weight_error < 1.0e-12, (
    f"the ion charge on the mesh and the ion macroparticle weights differ by "
    f"{weight_error:.3e}, so the sums above do not mean what the test assumes"
)
