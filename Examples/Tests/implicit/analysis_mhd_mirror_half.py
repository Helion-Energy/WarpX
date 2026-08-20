#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Half-domain twin of the z_lo mirror-symmetry boundary test.

Physics (the gold standard of the mirror boundary): this run covers
z in [0, Lz] with the mirror plane at z_lo (boundary.field_lo = pmc on
z paired with implicit_mhd.z_lo_boundary_fluid = symmetry) and must
reproduce the UPPER HALF of the full-domain reference run
(test_rz_theta_implicit_mhd_mirror_full, the test dependency)
field-for-field to solver roundoff: the parity ghost fills make the
half-domain residual the exact restriction of the full-domain residual
to the symmetric subspace, so the only differences are the
Newton/GMRES scalar reductions (tolerances 1e-12 in both decks).

Additionally asserted: (b) the half-domain mass is conserved to
roundoff (the mirror plane is a zero-flux face and nothing reaches the
z_hi outflow end within the run), (c) the tangential magnetic field on
the z_lo plane vanishes -- the first interior rows of Br and Btheta
equal MINUS the full run's rows just below its z = 0 plane -- and
(d) the axial velocity at the first interior row is
antisymmetry-consistent the same way.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import yt

FIELDS = [
    "implicit_mhd_mass_density",
    "implicit_mhd_electron_energy",
    "implicit_mhd_ion_energy",
    "implicit_mhd_momentum_r",
    "implicit_mhd_momentum_t",
    "implicit_mhd_momentum_z",
    "Br",
    "Bt",
    "Bz",
    "Er",
    "Et",
    "Ez",
    "jr",
    "jt",
    "jz",
]

# Half-vs-full equivalence budget: the residuals are exact mirror
# twins; the converged states differ only within the nonlinear/linear
# solver tolerances (1e-12 relative in both decks).
EQUIVALENCE_RTOL = 1.0e-10

BASELINE_DIRECTORY = "../test_rz_theta_implicit_mhd_mirror_full"


def load_fields(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    fields = {name: grid["boxlib", name].value[:, :, 0] for name in FIELDS}
    return ds, fields


def relative_error(difference, scale):
    if scale == 0.0:
        assert difference == 0.0
        return 0.0
    return difference / scale


def total_mass(ds, fields):
    # RZ cell volume 2 pi r dr dz with uniform dr, dz: sum(rho * r_i).
    nr = fields["implicit_mhd_mass_density"].shape[0]
    left = float(ds.domain_left_edge[0].value)
    right = float(ds.domain_right_edge[0].value)
    dr = (right - left) / nr
    radii = left + (np.arange(nr) + 0.5) * dr
    return float(np.sum(fields["implicit_mhd_mass_density"] * radii[:, None]))


def main():
    initial_plotfile, final_plotfile = sys.argv[1], sys.argv[2]

    # The mirror configuration is part of the regression contract.
    used_inputs = Path("warpx_used_inputs").read_text()
    assert re.search(
        r"^implicit_mhd\.z_lo_boundary_fluid\s*=\s*symmetry",
        used_inputs,
        re.MULTILINE,
    ), "the half-domain deck must select z_lo_boundary_fluid = symmetry"

    ds0, initial = load_fields(initial_plotfile)
    ds1, final = load_fields(final_plotfile)
    _, full_initial = load_fields(os.path.join(BASELINE_DIRECTORY, initial_plotfile))
    _, full_final = load_fields(os.path.join(BASELINE_DIRECTORY, final_plotfile))

    # (a) The half domain matches the full run's upper half.
    nz_half = final["Br"].shape[1]
    for label, half, full in (
        ("step 0", initial, full_initial),
        ("the final step", final, full_final),
    ):
        print(f"--- half vs full upper half at {label} ---")
        worst = 0.0
        for name in FIELDS:
            upper = full[name][:, nz_half:]
            assert upper.shape == half[name].shape
            error = relative_error(
                np.max(np.abs(half[name] - upper)),
                np.max(np.abs(full[name])),
            )
            print(f"{name:34s} relative error {error:.3e}")
            worst = max(worst, error)
            assert error < EQUIVALENCE_RTOL, (
                f"{name} deviates from the full run at {label}: {error:.3e}"
            )
        print(f"worst equivalence error at {label}: {worst:.3e}")

    # (b) Mass conservation: the mirror plane is a zero-flux face and
    # nothing reaches z_hi within the run.
    mass_initial = total_mass(ds0, initial)
    mass_final = total_mass(ds1, final)
    mass_change = abs(mass_final - mass_initial) / mass_initial
    print(f"relative mass change: {mass_change:.3e}")
    assert mass_change < 1.0e-11

    # (c) Tangential B vanishes on the mirror plane: the half run's
    # first interior row is the exact NEGATIVE of the full run's row
    # just below z = 0 (their average is the plane value).
    # (d) Same antisymmetry consistency for the axial velocity/momentum.
    for name in ("Br", "Bt", "implicit_mhd_momentum_z"):
        below = full_final[name][:, nz_half - 1]
        error = relative_error(
            np.max(np.abs(final[name][:, 0] + below)),
            np.max(np.abs(full_final[name])),
        )
        print(f"plane antisymmetry consistency of {name}: {error:.3e}")
        assert error < EQUIVALENCE_RTOL

    print("half-domain mirror test: all checks passed")


if __name__ == "__main__":
    main()
