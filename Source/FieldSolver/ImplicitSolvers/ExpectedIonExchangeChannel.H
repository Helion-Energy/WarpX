/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXPECTED_ION_EXCHANGE_CHANNEL_H_
#define WARPX_EXPECTED_ION_EXCHANGE_CHANNEL_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>
namespace warpx::thermal {
// Exact QDSMCInitializeUe midpoint specialization, INTO PRIVATE nodal fields:
// V_e=-(Jp-Ji)/rho with the native raw-floor/pedestal convention. No registry.
bool BuildExpectedIonDrift (
    const amrex::Geometry&, const amrex::MultiFab& raw_charge,
    const amrex::MultiFab* pedestal,
    const std::array<const amrex::MultiFab*, 3>& ion_current,
    const std::array<const amrex::MultiFab*, 3>& plasma_current,
    amrex::Real density_floor,
    const std::array<amrex::MultiFab*, 3>& private_nodal_drift);
} // namespace warpx::thermal
#endif
