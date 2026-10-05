/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRONINERTIACONVECTION_H_
#define WARPX_ELECTRONINERTIACONVECTION_H_

#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <array>

/** Spatial electron acceleration, evaluated from the current field stage.
 *
 * Adds -(m_e/e) (u_e . grad) u_e to the elliptic Ohm RHS, where
 * u_e = (J_i - J_plasma)/max(rho + rho_ped, e n_floor). No temporal
 * history or heat deposition is involved. The caller supplies synchronized
 * moment/current guards (one layer), then applies its physical E boundary
 * conditions and the curl-curl recovery after this addition.
 *
 * Currents are gathered to nodes. The rotational form uses centered curl
 * derivatives (second-order one-sided at physical faces) and a native Yee
 * gradient of |u|^2/2, avoiding a spurious curl of the scalar gradient.
 * All three vector components are retained in axisymmetric RZ, including
 * cylindrical connection terms. The velocity is rebuilt on every call;
 * only allocation is cached. This operator is second-order consistent, not
 * a claim of exact discrete moving-plasma energy conservation.
 */
class ElectronInertiaConvection
{
public:
    void AddToRHS (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& J_plasma,
        ablastr::fields::VectorField const& J_ion,
        amrex::MultiFab const& rho, amrex::MultiFab const* pedestal,
        amrex::Real rho_floor, amrex::Geometry const& geom,
        std::array<amrex::iMultiFab const*, 3> const& eb_mask = {nullptr, nullptr, nullptr});

private:
    amrex::MultiFab m_velocity;
    amrex::MultiFab m_acceleration;
};
#endif
