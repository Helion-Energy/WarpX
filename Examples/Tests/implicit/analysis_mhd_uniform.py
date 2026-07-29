#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import sys

import numpy as np
import scipy.constants as constants
import yt

ds = yt.load(sys.argv[1])
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

n0 = 1.0e20
rho0 = n0 * constants.proton_mass
electron_pressure = n0 * 10.0 * constants.elementary_charge
electron_energy = electron_pressure / (5.0 / 3.0 - 1.0)

np.testing.assert_allclose(
    data["boxlib", "implicit_mhd_mass_density"].value,
    rho0,
    rtol=5.0e-14,
    atol=0.0,
)
np.testing.assert_allclose(
    data["boxlib", "implicit_mhd_electron_energy"].value,
    electron_energy,
    rtol=5.0e-14,
    atol=0.0,
)
np.testing.assert_allclose(
    data["boxlib", "Pe"].value,
    electron_pressure,
    rtol=5.0e-14,
    atol=0.0,
)

for field_name in ("Ex", "Ey", "Ez", "Bx", "By"):
    np.testing.assert_allclose(
        data["boxlib", field_name].value,
        0.0,
        rtol=0.0,
        atol=1.0e-14,
    )
np.testing.assert_allclose(
    data["boxlib", "Bz"].value,
    0.1,
    rtol=5.0e-14,
    atol=0.0,
)
