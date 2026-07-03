#!/usr/bin/env python3
"""Analysis for test_2d_eb_specular_reflection.

A specular embedded-boundary wall must (a) not absorb or leak particles and (b) conserve total
kinetic energy (elastic bounce). Over the run, particles repeatedly strike the EB slab, so
constant particle number and constant total energy together demonstrate correct specular
reflection (the stock EB behavior would absorb particles, dropping the count).

Reads the ParticleNumber and ParticleEnergy reduced diagnostics; asserts both are constant.
"""

import numpy as np

npart = np.loadtxt("diags/reducedfiles/npart.txt")
pen = np.loadtxt("diags/reducedfiles/pen.txt")

N = npart[:, 2]  # total (weighted) particle number
E = pen[:, 2]  # total kinetic energy [J]

N_spread = (N.max() - N.min()) / N[0]
E_spread = (E.max() - E.min()) / E[0]

print(f"particle number: {N[0]:.6e} -> {N[-1]:.6e}   (max spread {N_spread:.3e})")
print(f"total energy:    {E[0]:.6e} -> {E[-1]:.6e} J (max spread {E_spread:.3e})")

# specular reflection loses no particles ...
assert N_spread < 5e-3, (
    "particle number not conserved -> EB is absorbing/leaking, not reflecting"
)
# ... and conserves kinetic energy
assert E_spread < 5e-3, (
    "kinetic energy not conserved -> specular reflection is not elastic"
)

print("PASS: specular EB reflection conserves particle number and kinetic energy.")
