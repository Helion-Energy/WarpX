#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Full-domain reference of the z_lo mirror-symmetry boundary test.

Physics: an axisymmetric adiabatic blob exactly symmetric about z = 0
(with an even azimuthal flow that winds up an odd Btheta, so every
field of the mirror parity table is nonzero by the final output) on
z in [-Lz, Lz] with both z ends outflow. The discrete operator must COMMUTE with the z-mirror: every
field keeps its mirror parity (even scalars, even u_r/u_theta and Bz,
odd u_z and Br/Btheta, even E_r/E_theta and J_r/J_theta, odd E_z/J_z)
to solver roundoff at every output. In particular the antisymmetric
fields must cancel between the two rows straddling z = 0 -- the implied
plane values of Br, Btheta, and u_z are zero -- which is the continuous
contract the half-domain twin (test_rz_theta_implicit_mhd_mirror_half)
realizes as a boundary condition.

Nothing may reach the z ends within the run, so the total mass is also
conserved to roundoff (axis, PEC wall, and untouched outflow ends leave
no exit).
"""

import sys

import numpy as np
import yt

# (field name, z-mirror parity)
FIELD_PARITY = [
    ("implicit_mhd_mass_density", +1),
    ("implicit_mhd_electron_energy", +1),
    ("implicit_mhd_ion_energy", +1),
    ("implicit_mhd_momentum_r", +1),
    ("implicit_mhd_momentum_t", +1),
    ("implicit_mhd_momentum_z", -1),
    ("Br", -1),
    ("Bt", -1),
    ("Bz", +1),
    ("Er", +1),
    ("Et", +1),
    ("Ez", -1),
    ("jr", +1),
    ("jt", +1),
    ("jz", -1),
]

# Solver-roundoff parity budget: the residual stencils are mirror-exact,
# so any asymmetry enters only through the Newton/GMRES scalar
# reductions (tolerances 1e-12 in the deck).
PARITY_RTOL = 1.0e-11


def load_fields(plotfile):
    ds = yt.load(plotfile)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    fields = {name: grid["boxlib", name].value[:, :, 0] for name, _ in FIELD_PARITY}
    return ds, fields


def relative_error(difference, scale):
    if scale == 0.0:
        assert difference == 0.0
        return 0.0
    return difference / scale


def check_parity(fields, label):
    print(f"--- z-mirror parity of the full run at {label} ---")
    worst = 0.0
    for name, parity in FIELD_PARITY:
        data = fields[name]
        nz = data.shape[1]
        assert nz % 2 == 0
        upper = data[:, nz // 2 :]
        lower_mirrored = data[:, : nz // 2][:, ::-1]
        error = relative_error(
            np.max(np.abs(upper - parity * lower_mirrored)),
            np.max(np.abs(data)),
        )
        print(f"{name:34s} parity {parity:+d} relative error {error:.3e}")
        worst = max(worst, error)
        assert error < PARITY_RTOL, (
            f"{name} violates its z-mirror parity at {label}: {error:.3e}"
        )
    print(f"worst parity error at {label}: {worst:.3e}")


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
    ds0, initial = load_fields(initial_plotfile)
    ds1, final = load_fields(final_plotfile)

    check_parity(initial, "step 0")
    check_parity(final, "the final step")

    # The run must have DONE something in every parity channel or the
    # checks above are vacuous.
    for name, _ in FIELD_PARITY:
        assert np.max(np.abs(final[name])) > 0.0, f"{name} stayed zero"

    # Straddling-row cancellation at z = 0 (the implied plane values of
    # the odd fields vanish).
    nz = final["Br"].shape[1]
    for name in ("Br", "Bt", "implicit_mhd_momentum_z"):
        straddle = relative_error(
            np.max(np.abs(final[name][:, nz // 2] + final[name][:, nz // 2 - 1])),
            np.max(np.abs(final[name])),
        )
        print(f"straddling-row cancellation of {name}: {straddle:.3e}")
        assert straddle < PARITY_RTOL

    # Mass conservation (nothing reached the z ends).
    mass_initial = total_mass(ds0, initial)
    mass_final = total_mass(ds1, final)
    mass_change = abs(mass_final - mass_initial) / mass_initial
    print(f"relative mass change: {mass_change:.3e}")
    assert mass_change < 1.0e-11

    print("full-domain mirror reference: all checks passed")


if __name__ == "__main__":
    main()
