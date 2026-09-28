/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ABSORBING_ELECTRON_INVENTORY_H_
#define WARPX_ABSORBING_ELECTRON_INVENTORY_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace warpx::thermal {
struct AbsorbingElectronInventoryResult {
    bool valid = false;
    amrex::Real raw_requested_joule = 0.;
    amrex::Real represented_joule = 0.;
    amrex::Real floor_replacement_joule = 0.;
    amrex::Real absolute_raw_requested_joule = 0.;
    amrex::Real absolute_represented_joule = 0.;
    amrex::Real absolute_floor_replacement_joule = 0.;
    amrex::Real max_temperature_relative_error = 0.;
    amrex::Real max_survivor_density_relative_increase = 0.;
};

// Pure candidate for accepted deterministic particle absorption. All inputs
// are cell-centered, one component, with identical boxes/distribution. Uv is
// J/m^3. nv and ns are effective electron capacities [m^-3] obtained using the
// SAME native moment options and the SAME pedestal at this accepted boundary.
// raw_lost_n is the LINEAR native restriction of signed lost charge divided by
// q_e: never add a pedestal or apply a constitutive density floor to this
// input.
//
// Us = Uv*(ns/nv), with a bitwise Uv copy when ns == nv. Output is independent
// scratch; inputs never change. False rejects nonfinite/negative energy,
// nonpositive capacities, or unrepresentable results. Scratch on failure is
// unspecified and must not be committed. No particle, RNG, model, or ledger
// state is mutated. Ghost values are not read or written.
//
// Signed physical-volume integrals: raw=(Uv/nv)*raw_lost_n, represented=Uv-Us,
// replacement=raw-represented. No clipping of signed loss or ns>nv occurs.
// The latter is measured for the caller's separate pure-loss consistency gate.
AbsorbingElectronInventoryResult
RemapAbsorbingElectronInventory (const amrex::Geometry& geometry,
                                 const amrex::MultiFab& virtual_energy,
                                 const amrex::MultiFab& virtual_density,
                                 const amrex::MultiFab& survivor_density,
                                 const amrex::MultiFab& raw_lost_number_density,
                                 amrex::MultiFab& survivor_energy);
} // namespace warpx::thermal
#endif
