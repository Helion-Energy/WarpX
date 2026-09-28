/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_WORK_REDISTRIBUTION_H_
#define WARPX_THERMAL_WORK_REDISTRIBUTION_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>
#include <memory>

namespace warpx::thermal {

// Pure conservative work transfer on the actual Yee component staggering.
// Each signed physical edge J.E first uses its eligible original endpoints,
// then their immediate outer neighbours only if no original endpoint is
// eligible. In RZ theta has one endpoint and four outer neighbours; r/z have
// two endpoints and six neighbours. Cartesian edges have two endpoints and
// 2+4*(dimension-1) outer neighbours. Nothing is scattered or atomically added.
//
// eligibility (nodal, 1 component, 2 ghosts) and work (Yee, 4 components,
// 2 ghosts) are caller-owned disjoint scratch. Channels of work are extensive
// power [W]: inner share, outer share, signed unassigned, absolute unassigned.
// rates is nodal [W/m^3]. Only its three specified channels are overwritten.
// Physical ghosts are zero; periodic ghosts are exchanged. Boundary images
// of rho/Te/current/E are not used as additional physical heat receivers.
void RedistributeThermalWork (
    const amrex::Geometry& geometry, amrex::Real density_floor,
    const amrex::MultiFab& raw_charge, const amrex::MultiFab& temperature_kelvin,
    const amrex::MultiFab& heat_capacity_charge,
    const std::array<const amrex::MultiFab*, 3>& current,
    const std::array<const amrex::MultiFab*, 3>& electric,
    amrex::MultiFab& eligibility,
    std::array<std::unique_ptr<amrex::MultiFab>, 3>& work,
    amrex::MultiFab& rates, int accepted, int declined, int unassignable_abs);

} // namespace warpx::thermal
#endif
