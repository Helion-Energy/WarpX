#!/usr/bin/env python3
"""Compare the coupled Cartesian XZ PETSc solve with AMReX GMRES."""

import glob
import math
import os
import re
import sys

AMREX_REF_DIR = os.path.join("..", "test_2d_hybrid_mag_diffusion_coupled")
PARITY_TOL = 2.0e-6


def last_plotfile(base_dir):
    plots = sorted(
        path
        for path in glob.glob(os.path.join(base_dir, "diags", "diag1*"))
        if os.path.isdir(path)
    )
    if not plots:
        raise RuntimeError(f"No plotfiles under {base_dir}/diags/")
    return plots[-1]


def component_minmax(diag_dir):
    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    with open(cell_h) as infile:
        text = infile.read()
    float_lines = []
    for line in text.splitlines():
        if not re.match(r"^-?[0-9]", line.strip()):
            continue
        try:
            values = [
                float(value) for value in line.strip().rstrip(",").split(",") if value
            ]
        except ValueError:
            continue
        if len(values) >= 3:
            float_lines.append(values)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse field min/max from {cell_h}")
    return float_lines[-2][:3], float_lines[-1][:3]


def magnetic_energy(base_dir):
    path = os.path.join(base_dir, "diags", "reduced_files", "field_energy.txt")
    with open(path) as infile:
        rows = [
            line
            for line in infile
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not rows:
        raise RuntimeError(f"No data rows in {path}")
    return float(rows[-1].split()[4])


def relative_error(a, b, scale):
    return abs(a - b) / max(scale, 1.0e-30)


def main():
    ref_plot = last_plotfile(AMREX_REF_DIR)
    petsc_plot = last_plotfile(".")
    ref_min, ref_max = component_minmax(ref_plot)
    petsc_min, petsc_max = component_minmax(petsc_plot)
    ref_energy = magnetic_energy(AMREX_REF_DIR)
    petsc_energy = magnetic_energy(".")

    values = ref_min + ref_max + petsc_min + petsc_max + [ref_energy, petsc_energy]
    if not all(math.isfinite(value) for value in values):
        print("FAILED: non-finite field extrema or magnetic energy")
        return 1

    names = ["By", "Bx", "Bz"]
    errors = []
    for component, name in enumerate(names):
        scale = max(
            abs(ref_min[component]),
            abs(ref_max[component]),
            abs(petsc_min[component]),
            abs(petsc_max[component]),
        )
        error = max(
            relative_error(ref_min[component], petsc_min[component], scale),
            relative_error(ref_max[component], petsc_max[component], scale),
        )
        errors.append(error)
        print(
            f"{name}: AMReX=[{ref_min[component]:.9e}, {ref_max[component]:.9e}] "
            f"PETSc=[{petsc_min[component]:.9e}, {petsc_max[component]:.9e}] "
            f"rel={error:.3e}"
        )

    energy_error = relative_error(
        ref_energy, petsc_energy, max(abs(ref_energy), abs(petsc_energy))
    )
    print(
        f"B-energy: AMReX={ref_energy:.12e} PETSc={petsc_energy:.12e} "
        f"rel={energy_error:.3e}"
    )

    if max(errors + [energy_error]) >= PARITY_TOL:
        print(f"FAILED: PETSc/AMReX parity error >= {PARITY_TOL:.0e}")
        return 1

    print(
        "PASSED: Cartesian XZ coupled PETSc/AMReX parity and exact-Pmat "
        f"field agreement < {PARITY_TOL:.0e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
