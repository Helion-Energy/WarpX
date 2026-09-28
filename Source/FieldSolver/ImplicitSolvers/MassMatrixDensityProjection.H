/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_MASS_MATRIX_DENSITY_PROJECTION_H_
#define WARPX_MASS_MATRIX_DENSITY_PROJECTION_H_

#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_Array.H>
#include <cstdint>
#include <AMReX_MultiFab.H>

class WarpX;

/** Continuity response about a fully deposited nonlinear particle state.
 *
 * rho_mid(E0 + dE) = rho_mid,0 - interval * div_h(Ji(E0 + dE) - Ji0).
 * The caller supplies physical, synchronized current and density, and uses
 * interval = dt/2 for the midpoint density. With Esirkepov, this is the
 * average of the old and virtual endpoint deposits. Anchoring preserves the
 * nonlinear residual at zero perturbation. Legacy Direct deposits use the
 * same anchored predictor as an approximation to their density response.
 * All stored data and the application remain on the AMReX device.
 */
class MassMatrixDensityProjection
{
public:
    /** Divergence in the charge deposit's volume convention (including the
     * optional Verboncoeur axis volume), not the Maxwell field divergence. */
    static void ComputeDivergence (WarpX& simulation, int lev,
                                  ablastr::fields::VectorField const& current,
                                  amrex::MultiFab& divergence);

    void CaptureStepStart (amrex::MultiFab const& density, int component);
    void DepositEndpointAverage (WarpX& simulation, int lev,
                                 amrex::MultiFab& density, int component);

    enum class EndpointSubset { All, Survivors, Absorbed };
    /** Private endpoint shape deposit with the identical native volume/sync/BC
     * operations. Qualified deterministic absorption only; no particle, RNG,
     * boundary tally or registered field is changed. */
    static bool DepositEndpointSubset (WarpX& simulation, int lev,
                                       amrex::MultiFab& density,
                                       EndpointSubset subset);


    void Capture (amrex::MultiFab const& density, int component,
                  ablastr::fields::VectorField const& current);

    /** Update only the selected scalar density component from the held base.
     * Closed reflecting/periodic domain boundaries are required by the caller.
     * Vacuum stays zero. Positive charge may increase or decrease, including
     * below the Ohm density floor; only negative predicted charge is clipped.
     * Constitutive floors belong to the consumers, not to continuity.
     */
    void Apply (WarpX& simulation, int lev, amrex::MultiFab& density, int component,
                ablastr::fields::VectorField const& current,
                amrex::Real interval);

    /** The input contains delta Ji after the native metric, guard sum and
     * particle wall images. Retain it in existing scratch, then replace the
     * input by Ji0 + delta Ji. No allocation and no second boundary sum. */
    void CaptureCurrentIncrementAndCompose (ablastr::fields::VectorField const& current);
    void ApplyCurrentIncrement (WarpX& simulation, int lev,
                                amrex::MultiFab& density, int component,
                                amrex::Real interval);
    void InvalidateCurrentIncrement () noexcept;
    // Additive, read-only copy-out for a typed MM thermal consumer. A current
    // compose alone does not certify density: publication becomes ready only
    // after ApplyCurrentIncrement has completed its physical images.
    [[nodiscard]] bool DensityIncrementReady () const noexcept
    { return m_density_increment_ready; }
    [[nodiscard]] std::uint64_t BaseEpoch () const noexcept { return m_base_epoch; }
    [[nodiscard]] std::uint64_t PublicationEpoch () const noexcept { return m_publication_epoch; }
    bool CopyLinearizationBase (amrex::MultiFab& density,
        ablastr::fields::VectorField const& current, std::uint64_t expected_base) const;
    bool CopyRetainedIncrement (amrex::MultiFab& density,
        ablastr::fields::VectorField const& current,
        std::uint64_t expected_base, std::uint64_t expected_publication) const;
    [[nodiscard]] bool CurrentIncrementReady () const noexcept
    { return m_current_increment_ready; }
    [[nodiscard]] bool IsCaptured () const { return m_density.isDefined(); }

    // Public for CUDA extended-lambda instantiation; callers use Apply above.
    void ApplyStoredCurrentIncrement (WarpX&, int, amrex::MultiFab&, int,
                                      amrex::Real interval, bool difference_first);
private:
    bool m_current_increment_ready = false, m_density_increment_ready = false;
    std::uint64_t m_base_epoch = 0, m_publication_epoch = 0;
    amrex::MultiFab m_step_density, m_endpoint_density;
    amrex::MultiFab m_density;
    amrex::MultiFab m_delta_density;
    amrex::Array<amrex::MultiFab, 3> m_current;
    amrex::Array<amrex::MultiFab, 3> m_delta_current;
};

#endif
