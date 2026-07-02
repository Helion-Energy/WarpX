#!/usr/bin/env python3
"""Analysis for the collocated conformal embedded-boundary cylinder-compression test.

This exercises the hybrid solver's collocated (nodal) conformal-EB path end to end:
a multi-step PIC evolution with grid_type='collocated' + use_conformal_eb=True. It
checks two things that the stair-case / staggered paths do not produce:

1. All fields are finite after the run. On the collocated grid the conformal update
   must SKIP the (staggered-only) ECT face-circulation recompute (EvolveECTRho); if
   that gate were wrong it would touch absent ECT geometry and yield NaN/abort.

2. The magnetic field is driven to ~0 deep inside the conductor by the level-set
   magnetic-parity EB fill (ApplyPECBoundaryToField, normal-odd / tangential-even,
   applied to B after the nodal Faraday push). The initial condition loads a finite
   Bz everywhere (including the conductor); only the B fill zeroes it deep inside.
   Without the fill the unmasked nodal Faraday leaves the field there ~ O(peak).
"""

import sys

import numpy as np
import yt

yt.set_log_level(50)

R_C = (
    0.5  # conducting-wall radius (EmbeddedBoundary implicit_function x**2+y**2-R_w**2)
)


def main():
    path = "diags/diag1000010"
    if "--path" in sys.argv:
        path = sys.argv[sys.argv.index("--path") + 1]

    ds = yt.load(path)
    dims = ds.domain_dimensions
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)

    fields = {
        f: np.asarray(cg["boxlib", f])
        for f in ("Bx", "By", "Bz", "Ex", "Ey", "Ez", "rho")
    }
    for name, arr in fields.items():
        assert np.all(np.isfinite(arr)), f"non-finite values in field {name}"

    bmag = np.sqrt(fields["Bx"] ** 2 + fields["By"] ** 2 + fields["Bz"] ** 2)
    peak = float(bmag.max())
    assert peak > 0.0, "magnetic field is identically zero"

    xc = np.asarray(cg["index", "x"])
    yc = np.asarray(cg["index", "y"])
    rr = np.sqrt(xc**2 + yc**2)
    dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]) / dims[0])

    # "deep" = at least two cells inside the conductor, past the one-cell mirror band.
    deep = rr > (R_C + 2.0 * dx)
    assert deep.any(), "no deep-conductor nodes in the domain"
    deep_max = float(bmag[deep].max())

    print(
        f"peak |B| = {peak:.6e} T ; deep-conductor (r > {R_C + 2 * dx:.3f} m) "
        f"max |B| = {deep_max:.6e} T ; ratio = {deep_max / peak:.3e}"
    )

    # With the level-set B fill the PLASMA field is zeroed in the deep conductor;
    # what remains in the plotfile there is the external (compression-coil) field,
    # which by design fills THROUGH the wall (the vacuum coil field penetrates a
    # resistive shell; the wall condition applies to the plasma response). Measured
    # 2026-07-02 (isotropic operators at their default-off): deep/peak = 1.16e-3 --
    # the external-field floor at the test's early ramp time. Without the
    # magnetic-parity fill the staircase leaves ~0.5*peak in the shallow conductor,
    # so a 5e-3 cap keeps ~100x discrimination against a fill-off regression while
    # sitting ~4x above the external floor.
    assert deep_max < 5.0e-3 * peak, (
        f"deep-conductor |B| ({deep_max:.3e} T) not suppressed by the level-set B "
        f"EB fill (peak {peak:.3e} T) -- the magnetic-parity fill may not be applied"
    )

    print(
        "PASS: collocated conformal-EB cylinder compression -- fields finite and B "
        "zeroed deep inside the conductor by the level-set magnetic-parity fill."
    )


if __name__ == "__main__":
    main()
