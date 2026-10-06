/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRONINERTIAMOMENTS_H_
#define WARPX_ELECTRONINERTIAMOMENTS_H_

#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <array>

class FiniteDifferenceSolver;

/** Ion-moment contributions to the explicit elliptic electron-inertia RHS.
 *
 * rho_i denotes positive ion charge density, not net plasma charge.
 * Quasineutrality and charge continuity give partial_t rho_i = -div(J_i).
 * Together with u_e=(J_i-J_plasma)/rho_eff, this adds
 *   -(m_e/(e rho_eff)) [partial_t J_i + u_e div(J_i)]
 * to the existing velocity-advection RHS before curl-curl recovery.
 * The total-current derivative is already in that recovery; do not add it here.
 *
 * The current slope is captured from consecutive leapfrog deposits before
 * their in-place averaging, and held through both magnetic halves and final
 * particle-gather E solve. No density history or thermal source is introduced.
 * A static pedestal has zero time derivative. Where a fixed floor clips the
 * density, partial_t rho_eff=0 and the compression contribution is suppressed.
 * Changes of the numerical floor/pedestal themselves are not physical sources.
 */
class ElectronInertiaMoments
{
public:
    void SetCurrentSlope (
        ablastr::fields::VectorField const& old_current,
        ablastr::fields::VectorField const& new_current, amrex::Real dt);

    void AddToRHS (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& J_plasma,
        ablastr::fields::VectorField const& J_ion,
        amrex::MultiFab const& rho_i, amrex::MultiFab const* pedestal,
        amrex::Real rho_floor, amrex::Geometry const& geom,
        FiniteDifferenceSolver& solver,
        std::array<amrex::iMultiFab const*, 3> const& eb_mask = {nullptr, nullptr, nullptr});

    /** Combined conservative momentum flux on the native current/E grid.
     * Uses the same edge density as the elliptic operator. The effective
     * continuity residual is retained at a clipped floor and for the RZ
     * staggered-volume geometry; no mass or heat source is deposited.
     */
    void AddMomentumFluxToRHS (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& J_plasma,
        ablastr::fields::VectorField const& J_ion,
        amrex::MultiFab const& rho_i, amrex::MultiFab const* pedestal,
        amrex::Real rho_floor, amrex::Geometry const& geom,
        FiniteDifferenceSolver& solver,
        std::array<amrex::iMultiFab const*, 3> const& eb_mask = {nullptr, nullptr, nullptr});

private:
    std::array<amrex::MultiFab, 3> m_flux_current;
    std::array<amrex::MultiFab, 3> m_flux_velocity;
    std::array<amrex::MultiFab, 3> m_current_slope;
    amrex::MultiFab m_div_ion_current;
    amrex::MultiFab m_rhs;
};
#endif
