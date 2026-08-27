#!/usr/bin/env python3

# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Shared helpers of the dual-energy (the reference code's mixmaster) closure tests:
python replicas of the kernel's smooth pressure recovery, the kinetic
fraction fk, and the blended ion pressure (ThetaImplicitMHD_K.H)."""

import numpy as np

FK_WIDTH = 0.05  # theta_implicit_mhd::dual_energy_fk_width


def smooth_positive_floor(value, floor):
    excess = value - floor
    return floor + 0.5 * (excess + np.sqrt(excess**2 + floor**2))


def recovered_pressure(ion_energy, kinetic_energy, gamma_i, pressure_floor):
    """The C^1 smooth-max recovery p(E_i) of load_cell_state (legacy
    corner width: pressure_corner_width_fraction = 0)."""
    internal_floor = pressure_floor / (gamma_i - 1.0)
    excess = ion_energy - kinetic_energy - internal_floor
    internal = internal_floor + 0.5 * (excess + np.sqrt(excess**2 + internal_floor**2))
    return (gamma_i - 1.0) * internal


def kinetic_fraction(ion_energy, kinetic_energy, gamma_i, pressure_floor):
    """fk = 1 - KE/(gamma_i E_i), guarded and smoothly rectified exactly
    as dual_energy_kinetic_fraction (no cutoff: the tests keep it off)."""
    internal_floor = pressure_floor / (gamma_i - 1.0)
    guarded = smooth_positive_floor(ion_energy, internal_floor)
    raw = 1.0 - kinetic_energy / (gamma_i * guarded)
    return 0.5 * (raw + np.sqrt(raw**2 + FK_WIDTH**2))


def blended_pressure(
    ion_energy, kinetic_energy, internal_energy, gamma_i, pressure_floor
):
    """p_i = fk p(E_i) + (1 - fk) smooth-floor((gamma_i - 1) U_i)."""
    fk = kinetic_fraction(ion_energy, kinetic_energy, gamma_i, pressure_floor)
    pressure_internal = smooth_positive_floor(
        (gamma_i - 1.0) * internal_energy, pressure_floor
    )
    pressure_total = recovered_pressure(
        ion_energy, kinetic_energy, gamma_i, pressure_floor
    )
    return fk * pressure_total + (1.0 - fk) * pressure_internal


def load_fluid_state(plotfile):
    """Covering-grid arrays of the implicit-MHD fluid dumps."""
    import yt

    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    rho = data["boxlib", "implicit_mhd_mass_density"].value.ravel()
    momentum_z = data["boxlib", "implicit_mhd_momentum_z"].value.ravel()
    ion_energy = data["boxlib", "implicit_mhd_ion_energy"].value.ravel()
    momentum_square = momentum_z**2
    try:
        momentum_x = data["boxlib", "implicit_mhd_momentum_x"].value.ravel()
        momentum_square = momentum_square + momentum_x**2
    except Exception:
        pass
    fields = {
        "rho": rho,
        "momentum_z": momentum_z,
        "ion_energy": ion_energy,
        "kinetic_energy": 0.5 * momentum_square / rho,
    }
    try:
        fields["ion_internal_energy"] = data[
            "boxlib", "implicit_mhd_ion_internal_energy"
        ].value.ravel()
    except Exception:
        fields["ion_internal_energy"] = None
    return fields
