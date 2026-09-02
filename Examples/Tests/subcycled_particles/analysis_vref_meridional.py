#!/usr/bin/env python3
"""Analytic gate for meridional sizing of the subcycle count.

Run by hand against a captured run log:
    warpx.rz inputs_test_rz_subcycled_vref_meridional > run.log 2>&1
    python3 analysis_vref_meridional.py run.log
The CI harness does not preserve the run stdout, so the companion ctest entry
is run-only.

Parses the "[subcycle]" witness line and checks the measured reference speed
and the resulting subcycle count against closed-form values derived from the
proper velocities in the input file. Also re-runs nothing: the quantities
under test are printed by the code itself.
"""
import math
import re
import sys

C = 299792458.0

# proper velocities u/c of the five probe particles, and their theta
UOC = [
    (0.0, 2.0013845711889124e-01, 0.0, 0.0),  # p0 theta=0, pure azimuthal
    (0.0, 0.0, 3.335640951981520e-02, 0.0),  # p1 theta=0, pure axial
    (6.671281903963041e-02, 0.0, 0.0, 0.0),  # p2 theta=0, pure radial
    (
        1.0006922855944562e-02,
        3.3356409519815204e-03,
        1.3342563807926083e-02,
        0.0,
    ),  # p3 mixed
    (2.0013845711889124e-01, 0.0, 0.0, 0.5 * 3.141592653589793),  # p4 theta=pi/2
]

DR = 0.5 / 64.0  # binding cell size (dr < dz)
CFL_GRID = 0.4
DT = 1.0e-9
MARGIN = 0.2


def components(uxc, uyc, uzc, theta):
    """Return (|v|, v_meridional, |v_theta|) in m/s from proper velocity."""
    usq = (uxc**2 + uyc**2 + uzc**2) * C * C
    inv_gamma = 1.0 / (1.0 + usq / (C * C)) ** 0.5
    ux, uy, uz = uxc * C, uyc * C, uzc * C
    ct, st = math.cos(theta), math.sin(theta)
    vr = (ux * ct + uy * st) * inv_gamma
    vt = (-ux * st + uy * ct) * inv_gamma
    vz = uz * inv_gamma
    vfull = usq**0.5 * inv_gamma
    return vfull, (vr * vr + vz * vz) ** 0.5, abs(vt)


full = max(components(*p)[0] for p in UOC)
merid = max(components(*p)[1] for p in UOC)
azim = max(components(*p)[2] for p in UOC)

v_ref_expected = merid + MARGIN * azim
dt_sub_expected = CFL_GRID * DR / v_ref_expected
# integer ceil of DT/dt_sub, with the same 1e-12 guard the code applies
n_expected = max(1, math.ceil(DT / dt_sub_expected * (1 - 1e-12)))

log = open(sys.argv[1] if len(sys.argv) > 1 else "run.log", errors="ignore").read()
m = re.search(
    r"\[subcycle\] probe: (\d+) subcycles/step .*?v_ref = ([0-9.eE+-]+) m/s "
    r"\(measured, (meridional|full speed)\)",
    log,
)
assert m, "no [subcycle] witness line with a measured meridional/full v_ref"
n_got, v_got, mode = int(m.group(1)), float(m.group(2)), m.group(3)

print(f"mode           : {mode}")
print(f"max |v|        : {full:.6e} m/s")
print(f"max meridional : {merid:.6e} m/s")
print(f"max |v_theta|  : {azim:.6e} m/s")
print(f"v_ref expected : {v_ref_expected:.6e} m/s   got {v_got:.6e}")
print(f"n_sub expected : {n_expected}                got {n_got}")

assert mode == "meridional", f"expected meridional sizing, got {mode}"
rel = abs(v_got - v_ref_expected) / v_ref_expected
assert rel < 1e-6, f"v_ref mismatch: rel err {rel:.3e}"
assert n_got == n_expected, f"subcycle count mismatch: {n_got} vs {n_expected}"

# The rotation test: p4 carries the largest speed in the box but is purely
# azimuthal, so the meridional maximum must come from p2, not p4.
assert merid < 0.5 * full, (
    "the meridional maximum should be far below the full maximum for this "
    "ensemble; if it is not, the azimuthal rotation was not applied"
)
print("PASSED")
