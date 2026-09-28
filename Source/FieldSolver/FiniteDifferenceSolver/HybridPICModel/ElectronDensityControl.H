/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_DENSITY_CONTROL_H_
#define WARPX_ELECTRON_DENSITY_CONTROL_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <cstdint>
#include <string>

namespace warpx::thermal {
// Immutable input configuration and accepted, restartable control state are
// distinct. The independently configured legacy activity gate stays separate.
struct ElectronDensityControl {
    amrex::Real initial_number_floor = 1., initial_gate_floor = 1.;
    amrex::Real initial_representation_floor = 1.;
    bool pedestal = false, pedestal_tracks_floor = false, halo_unfreeze = false;
    amrex::Real pedestal_taper_cells = 0.;
    std::string pedestal_profile;
    amrex::Real number_floor = 1., gate_floor = 1., representation_floor = 1.;
    std::uint64_t epoch = 0;
    amrex::Real floor_inventory_joule = 0., pedestal_inventory_joule = 0.;
    // A deck registers a canonical description of the controller parameters.
    // Every sample (including deadband no-ops) persists its EMA and step.
    std::string controller_configuration;
    amrex::Real controller_last = 0., controller_ema = 0.;
    bool controller_has_ema = false;
    std::int64_t controller_step = -1;
    std::string Encode () const;
    static ElectronDensityControl Decode (const std::string& text);
    void Validate () const;
    void ValidateConfiguration (const ElectronDensityControl& requested) const;
};

struct DensityControlInventory {
    bool valid = false;
    amrex::Real floor_joule = 0., pedestal_joule = 0., total_joule = 0.;
};
// Pure GPU-native candidate: preserve T_cell by U'=U*n'/n. The middle density
// changes only the common floor; the last also changes the tracked pedestal.
// Inventory entries sum to the actual U increment. These are external control
// adjustments, never residual source rates or ion/electron heat exchange.
DensityControlInventory RemapElectronDensityControl (
    const amrex::Geometry& geometry, const amrex::MultiFab& energy,
    const amrex::MultiFab& old_density, const amrex::MultiFab& floor_density,
    const amrex::MultiFab& new_density, amrex::MultiFab& candidate);
} // namespace warpx::thermal
#endif
