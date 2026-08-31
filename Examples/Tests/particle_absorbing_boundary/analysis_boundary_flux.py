#!/usr/bin/env python3
"""Validate the ParticleBoundaryFlux reduced diagnostic on the ballistic
two-beam RZ deck: closure against ParticleEnergy at every step, correct
face attribution, charge/weight consistency, and monotone cumulatives."""

import re
import sys

import numpy as np

QE = 1.602176634e-19


def read_reduced(path):
    """Return (data, {column_name: index}) for a reduced-diags .txt file."""
    with open(path) as f:
        header = f.readline()
    names = re.findall(r"\[\d+\](\S+?)\(", header)
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data, {n: i for i, n in enumerate(names)}


pe, pe_c = read_reduced("diags/reducedfiles/PE.txt")
pbf, pbf_c = read_reduced("diags/reducedfiles/PBF.txt")

assert pe.shape[0] == pbf.shape[0], "PE/PBF row counts differ"
assert np.array_equal(pe[:, pe_c["step"]], pbf[:, pbf_c["step"]]), (
    "PE/PBF steps misaligned"
)

def col(name):
    return pbf[:, pbf_c[name]]

E_tot = pe[:, pe_c["total"]]
E0 = E_tot[0]
ke_lost = sum(
    col(f"{sp}_{ax}_{side}_KE_J")
    for sp in ("protons", "electrons")
    for ax in ("r", "z")
    for side in ("lo", "hi")
)

# (a) closure at EVERY row: in-domain energy + booked losses = initial energy
closure = np.max(np.abs(E_tot + ke_lost - E0)) / E0
print(f"closure max |E_tot + KE_lost - E0| / E0 = {closure:.3e}")
assert closure < 1.0e-9, f"closure failed: {closure:.3e}"

# all particles are gone at the end: everything is in the ledger
final_frac = E_tot[-1] / E0
print(f"final in-domain energy fraction = {final_frac:.3e}")
assert final_frac < 1.0e-12, "particles remain that should have exited"

# (b) face attribution: protons exit z_hi only, electrons z_lo only
p_zhi = col("protons_z_hi_KE_J")[-1]
e_zlo = col("electrons_z_lo_KE_J")[-1]
assert p_zhi > 0.0 and e_zlo > 0.0
for name in (
    "protons_z_lo_KE_J", "protons_r_lo_KE_J", "protons_r_hi_KE_J",
    "electrons_z_hi_KE_J", "electrons_r_lo_KE_J", "electrons_r_hi_KE_J",
):
    assert col(name)[-1] == 0.0, f"{name} expected exactly zero"
print(f"face split OK: protons z_hi {p_zhi:.6e} J, electrons z_lo {e_zlo:.6e} J")

# (c) charge/weight consistency: charge column = q * weight column
for sp, q in (("protons", QE), ("electrons", -QE)):
    w = col(f"{sp}_z_hi_count")[-1] + col(f"{sp}_z_lo_count")[-1]
    c = col(f"{sp}_z_hi_charge_C")[-1] + col(f"{sp}_z_lo_charge_C")[-1]
    rel = abs(c - q * w) / abs(q * w)
    print(f"{sp}: lost weight {w:.6e}, charge {c:.6e} C, q*w rel err {rel:.3e}")
    assert rel < 1.0e-12, f"{sp} charge/weight inconsistent: {rel:.3e}"

# analytic weight: constant density over an exactly cell-aligned cylinder
n0, rmax, zlen = 1.0e14, 0.05, 0.2
w_expected = n0 * np.pi * rmax**2 * zlen
for sp in ("protons", "electrons"):
    w = col(f"{sp}_z_hi_count")[-1] + col(f"{sp}_z_lo_count")[-1]
    rel = abs(w - w_expected) / w_expected
    print(f"{sp}: total lost weight vs n*V rel err {rel:.3e}")
    assert rel < 1.0e-6, f"{sp} weight vs analytic failed: {rel:.3e}"

# cumulative columns never decrease
for name, i in pbf_c.items():
    if name.endswith(("_count", "_KE_J")):
        assert np.all(np.diff(pbf[:, i]) >= -1e-300), f"{name} not monotone"

print("ParticleBoundaryFlux analysis: all checks passed")
sys.exit(0)
