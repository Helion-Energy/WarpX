#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Free-streaming conduction limiter: the cliff face flux sits on the cap.

One backward-Euler step across a 10x electron-temperature cliff with
chi_e sized so the unlimited conductive flux is ~100x the cap f q_fs.
The applied face flux, reconstructed from the energy update of the two
cells adjacent to the cliff, must match the analytic harmonic value
q_unl/(1 + q_unl/(f q_fs)) -- within a few % (the backward-Euler step
evaluates the flux at the slightly-relaxed end state) -- and must sit
~100x below the unlimited flux (the cap genuinely engaged).
"""

import sys

import numpy as np
import warpx_constants as constants
import yt


def get_data(plotfile):
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    return ds, data


initial_ds, initial = get_data(sys.argv[1])
final_ds, final = get_data(sys.argv[2])

n0 = 1.0e18
te_hot_ev = 20.0
te_cold_ev = 2.0
gamma_e = 5.0 / 3.0
number_of_cells = 64
domain_length = 1.0
dz = domain_length / number_of_cells
f_lim = 0.1
q_e = constants.elementary_charge
m_e = constants.m_e
m_p = constants.m_p

# the deck's sizing, reproduced exactly
te_face_ev = 0.5 * (te_hot_ev + te_cold_ev)
q_fs = n0 * q_e * te_face_ev * np.sqrt(q_e * te_face_ev / m_e)
de_spec = q_e * (te_hot_ev - te_cold_ev) / ((gamma_e - 1.0) * m_p)
chi0 = 100.0 * f_lim * q_fs * dz / (n0 * m_p * de_spec)
q_unlimited = chi0 * n0 * m_p * de_spec / dz
q_capped = q_unlimited / (1.0 + q_unlimited / (f_lim * q_fs))
dt = float(final_ds.current_time - initial_ds.current_time)

ue0 = initial["boxlib", "implicit_mhd_electron_energy"].value.ravel()
ue1 = final["boxlib", "implicit_mhd_electron_energy"].value.ravel()
due = ue1 - ue0

# the profile has TWO identical cliffs (z = zc and the periodic wrap),
# and the cold block is exactly symmetric about its own center, so the
# total heat gained by the cold block is 2 q_face dt/dz -- an EXACT
# telescoped measure of the applied face flux, immune to the secondary
# faces the backward-Euler end state opens next to each cliff.
cold = ue0 < 0.5 * (ue0.max() + ue0.min())
q_measured = 0.5 * np.sum(due[cold]) * dz / dt
q_hot_side = -0.5 * np.sum(due[~cold]) * dz / dt
print(f"cap engagement: q_unlimited/q_capped = {q_unlimited / q_capped:.1f}x")
print(f"analytic capped flux = {q_capped:.6e} W/m^2")
print(
    f"measured (cold-block gain) = {q_measured:.6e}, "
    f"(hot-block loss) = {q_hot_side:.6e}, "
    f"deviation = {q_measured / q_capped - 1.0:.3e}"
)

# the cap must have engaged massively (a limiter-off run would move
# ~100x more energy)
assert q_unlimited / q_capped > 20.0, "cap not engaged; sizing broken"
# exact conservation: the cold block gains exactly what the hot block
# loses (periodic telescoping)
np.testing.assert_allclose(q_measured, q_hot_side, rtol=1.0e-12)
# the applied flux sits on the analytic harmonic cap (backward-Euler
# end-state evaluation costs a few %)
np.testing.assert_allclose(q_measured, q_capped, rtol=0.05)

# the update concentrates at the cliffs (the implicit step's far-field
# tails decay geometrically and stay orders of magnitude down)
peak = np.abs(due).max()
cliff_cells = np.zeros_like(due, dtype=bool)
edges = np.nonzero(np.abs(np.diff(ue0)) > 0.5 * (ue0.max() - ue0.min()))[0]
for e in list(edges) + [len(due) - 1]:
    lo = max(e - 2, 0)
    hi = min(e + 4, len(due))
    cliff_cells[lo:hi] = True
    cliff_cells[:3] = True
far_peak = np.abs(due[~cliff_cells]).max() if (~cliff_cells).any() else 0.0
assert far_peak < 1.0e-3 * peak, (
    f"far-field update {far_peak:.3e} is not small vs the cliff {peak:.3e}"
)

print("PASS")
