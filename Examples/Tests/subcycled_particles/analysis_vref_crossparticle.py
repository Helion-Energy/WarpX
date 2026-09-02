#!/usr/bin/env python3
"""Regression gate: the meridional reference speed must never exceed the
full-speed answer, even when the meridional and azimuthal maxima belong to
different particles.

Run by hand against a captured run log:
    warpx.rz inputs_test_rz_subcycled_vref_crossparticle > run.log 2>&1
    python3 analysis_vref_crossparticle.py run.log
The CI harness does not preserve the run stdout, so the companion ctest entry
is run-only.
"""
import re
import sys

C = 299792458.0
MARGIN = 0.2
UOC = 6.671281903963041e-02  # u/c of both particles

# Both particles have the same speed; p0 carries it meridionally, p1
# azimuthally, so the two component maxima come from different particles.
usq = (UOC * C) ** 2
v = (usq**0.5) / (1.0 + usq / (C * C)) ** 0.5
full_max, mer_max, az_max = v, v, v

old_form = mer_max + MARGIN * az_max  # sum of independent maxima (the bug)
new_form = max(mer_max + MARGIN * 0.0, 0.0 + MARGIN * az_max)  # per particle
new_form = min(new_form, full_max)  # clamp

log = open(sys.argv[1] if len(sys.argv) > 1 else "run.log", errors="ignore").read()
m = re.search(r"v_ref = ([0-9.eE+-]+) m/s \(measured, (meridional|full speed)\)", log)
assert m, "no [subcycle] witness line with a measured v_ref"
v_got, mode = float(m.group(1)), m.group(2)

print(f"mode              : {mode}")
print(f"max |v|           : {full_max:.6e} m/s")
print(f"max meridional    : {mer_max:.6e} m/s   (particle 0)")
print(f"max |v_theta|     : {az_max:.6e} m/s   (particle 1)")
print(f"OLD form (sum)    : {old_form:.6e} m/s  -> {old_form/full_max:.4f} x full")
print(f"NEW form (per-p)  : {new_form:.6e} m/s  -> {new_form/full_max:.4f} x full")
print(f"v_ref from run    : {v_got:.6e} m/s     -> {v_got/full_max:.4f} x full")

assert mode == "meridional", f"expected meridional sizing, got {mode}"
# The fixture must actually exhibit the bug condition, else it proves nothing.
assert old_form > full_max * 1.05, (
    f"fixture does not exercise the bug: old form {old_form:.6e} is not "
    f"meaningfully above the full speed {full_max:.6e}"
)
# The fix: never above the full-speed answer.
assert v_got <= full_max * (1.0 + 1e-9), (
    f"meridional reference speed {v_got:.6e} exceeds the full speed "
    f"{full_max:.6e}; sizing on it would do more work than not using the "
    f"option at all"
)
rel = abs(v_got - new_form) / new_form
assert rel < 1e-6, f"v_ref mismatch vs per-particle form: rel err {rel:.3e}"
print("PASSED")
