#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Physical constants pinned to WarpX's own values.

Mirrors ``Source/ablastr/constant.H`` (the values behind ``PhysConst``),
exposed under the ``scipy.constants`` attribute names the analysis
scripts use. Analyses must compare simulation output against the
constants the CODE ran with: a system ``scipy`` built against a
different CODATA release differs from WarpX in, e.g., the proton mass at
the 1e-9 relative level, which is far above the roundoff-level
tolerances of the strictest checks here. (``pywarpx.picmi`` carries the
same hardcoded mirror; it is not imported to keep the analyses free of
the ``picmistandard`` dependency.)
"""

# values from Source/ablastr/constant.H (CODATA as pinned by WarpX)
speed_of_light = 299792458.0
c = speed_of_light
mu_0 = 1.2566370612685e-06
epsilon_0 = 1.0 / (mu_0 * c**2)
elementary_charge = 1.602176634e-19
e = elementary_charge
electron_mass = 9.1093837139e-31
m_e = electron_mass
proton_mass = 1.67262192595e-27
m_p = proton_mass
k = 1.380649e-23  # Boltzmann
