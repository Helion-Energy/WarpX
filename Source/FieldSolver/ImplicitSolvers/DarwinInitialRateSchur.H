/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_INITIAL_RATE_SCHUR_H_
#define WARPX_DARWIN_INITIAL_RATE_SCHUR_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>

namespace warpx::darwin {
enum class InitialRateBoundary { Axis, PEC, PMC, Periodic };

struct InitialRateSchurOptions {
    std::array<InitialRateBoundary, AMREX_SPACEDIM> lower{};
    std::array<InitialRateBoundary, AMREX_SPACEDIM> upper{};
    amrex::IntVect output_ghosts{1};
    // These are qualification guards, not switches that omit physical terms.
    amrex::Real theta = .5;
    bool djedt_only = true;
    bool bdf2 = false;
    // Spatial equation control: direct diagonal Yee mass instead of M*kappa*N.
    bool compatible_yee = false;
    bool embedded_boundaries = false;
    int azimuthal_modes = 1;
    amrex::Real relative_tolerance = 1.e-10;
    amrex::Real absolute_tolerance = 0.;
    int max_iterations = 600;
    int restart_length = 100;
    int preconditioner_cycles = 2;
    // Match the E_L multigrid hierarchy; these affect only the preconditioner.
    int max_semicoarsening_levels = 0;
    int semicoarsening_direction = -1;
    int verbose = 0;
};
struct InitialRateSchurResult {
    bool converged = false;
    int iterations = 0;
    amrex::Real initial_residual = 0.;
    amrex::Real residual = 0.;
    amrex::Real target = 0.;
};

/** Shared diagonal inertia map on the native Yee current components.
 * kappa is dimensionless and nodal. A cell-centered component uses the
 * harmonic mean of its two endpoint kappa values (the reciprocal arithmetic
 * effective density); a nodal component uses its local kappa. A zero endpoint
 * closes that edge. The caller owns validity/taper semantics and native BCs.
 * Only valid values are read. Output ghosts use zero physical extension and
 * periodic/interior synchronization; the caller applies any PMC vector image.
 * out = scale * kappa_edge * in, with no node-vector round trip.
 */
void ApplyYeeInertiaMass (
    const amrex::Geometry& geometry, const amrex::MultiFab& kappa,
    const std::array<const amrex::MultiFab*, 3>& in,
    const std::array<amrex::MultiFab*, 3>& out, amrex::Real scale = 1.);

/** Same represented edge coefficient, multiply the unevaluated pair before final rounding. */
void ApplyYeeInertiaMassPaired (
    const amrex::Geometry& geometry, const amrex::MultiFab& kappa,
    const std::array<const amrex::MultiFab*, 3>& in,
    const std::array<const amrex::MultiFab*, 3>& low,
    const std::array<amrex::MultiFab*, 3>& out, amrex::Real scale = 1.,
    const std::array<amrex::MultiFab*, 3>* output_low = nullptr);

/** Diagnostic native initial-rate weighted projection, single level.
 * This derives from the qualified longitudinal Schur by removing its DG
 * identity term: the target is DM*kappa*NG, with no time-discretization term.
 * N/M are the production Yee/node interpolations and D/G the native E_L pair,
 * including RZ axis4 (independent of the particle charge quadrature).
 * All input fields are read-only. Outputs and intermediate ghost images are
 * owned. The parent must still grade the full unmodified nonlinear physical
 * constraint.
 */
class DarwinInitialRateSchur {
  public:
    // Public opaque declaration permits nvcc extended lambdas in the
    // implementation.
    struct Impl;
    using ConstVector = std::array<const amrex::MultiFab*, 3>;
    DarwinInitialRateSchur (const amrex::Geometry& geometry,
                             const amrex::BoxArray& cells,
                             const amrex::DistributionMapping& distribution,
                             const InitialRateSchurOptions& options);
    ~DarwinInitialRateSchur ();
    DarwinInitialRateSchur (const DarwinInitialRateSchur&) = delete;
    DarwinInitialRateSchur& operator=(const DarwinInitialRateSchur&) = delete;

    /** Copy nodal scalar kappa >= 0, dimensionless, from the CURRENT kinetic
     * context: epsilon0*m_e*omega_ref^2/(q_e*rho_lim), or zero in true
     * vacuum. Caller uses exactly its native deposited rho/pedestal/floor/taper
     * semantics. No caller ghosts are consumed. False for nonfinite/negative
     * valid values.
     */
    bool Freeze (const amrex::MultiFab& kappa);

    /** Solve (DM*kappa*NG) dphi = D(raw_EL-held_EL).
     * Both inputs are Yee longitudinal vectors in V/m on this cell layout.
     * Outputs remain private even on failure; inspect converged before
     * applying.
     */
    InitialRateSchurResult Correct (const ConstVector& raw_EL,
                                     const ConstVector& held_EL);

    /** Correct the linear defect D(raw_EL)-D(held_EL) with a caller's
     * remaining iteration budget. Separate divergences retain corrections
     * smaller than an ulp of raw_EL. The budget must be in [0, options.max_iterations]; tolerances and the fresh
     * operator residual are unchanged. It never updates the stored options.
     * Inputs obey Correct's independent-owned-output contract.
     */
    InitialRateSchurResult CorrectWithIterationBudget (
        const ConstVector& raw_EL, const ConstVector& held_EL, int remaining_iterations);

    /** Fresh native ||D(raw)-D(held)||, with the refinement solve's images
     * and norm. Do not round raw-held before applying the linear operator.
     * Inputs are read-only, scratch is private, and the last correction and
     * potential remain intact. This is an action check, not a solve or a
     * physical-current acceptance criterion.
     */
    amrex::Real DivergenceResidualNorm (const ConstVector& raw_EL,
                                       const ConstVector& held_EL);

    /** Nodal potential operator [DM*kappa*NG], output in potential/length^2.
     * Homogeneous PEC phi=0, PMC/axis Neumann and periodic images.
     * Dirichlet rows are eliminated (zero), not part of the Krylov space.
     */
    void ApplyPotential (amrex::MultiFab& out,
                         const amrex::MultiFab& potential);
    /** Solve this same potential equation from a nodal scalar RHS. */
    InitialRateSchurResult SolvePotential (const amrex::MultiFab& rhs);
    const amrex::MultiFab& CorrectionPotential () const;
    ConstVector CorrectionField () const;

  private:
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::darwin
#endif
