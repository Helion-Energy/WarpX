/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_LONGITUDINAL_SCHUR_H_
#define WARPX_DARWIN_LONGITUDINAL_SCHUR_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>

namespace warpx::darwin {
enum class LongitudinalBoundary { Axis, PEC, PMC, Periodic };

struct LongitudinalSchurOptions {
    std::array<LongitudinalBoundary, AMREX_SPACEDIM> lower{};
    std::array<LongitudinalBoundary, AMREX_SPACEDIM> upper{};
    amrex::IntVect output_ghosts{1};
    // These are qualification guards, not switches that omit physical terms.
    amrex::Real theta = .5;
    bool djedt_only = true;
    bool bdf2 = false;
    // Match the direct native Yee inertia spatial equation when selected.
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
struct LongitudinalSchurResult {
    bool converged = false;
    int iterations = 0;
    amrex::Real initial_residual = 0.;
    amrex::Real residual = 0.;
    amrex::Real target = 0.;
};

/** Exact frozen midpoint displacement-inertia Schur correction, single level.
 * N/M are the production Yee/node interpolations and D/G the native E_L pair,
 * including RZ axis4 (independent of the particle charge quadrature).
 * All input fields are read-only. Outputs and intermediate ghost images are
 * owned. The parent must still grade the full unmodified nonlinear physical
 * constraint.
 */
class DarwinLongitudinalSchur {
  public:
    // Public opaque declaration permits nvcc extended lambdas in the
    // implementation.
    struct Impl;
    using ConstVector = std::array<const amrex::MultiFab*, 3>;
    DarwinLongitudinalSchur (const amrex::Geometry& geometry,
                             const amrex::BoxArray& cells,
                             const amrex::DistributionMapping& distribution,
                             const LongitudinalSchurOptions& options);
    ~DarwinLongitudinalSchur ();
    DarwinLongitudinalSchur (const DarwinLongitudinalSchur&) = delete;
    DarwinLongitudinalSchur& operator=(const DarwinLongitudinalSchur&) = delete;

    /** Copy nodal scalar kappa >= 0, dimensionless, from the CURRENT kinetic
     * context: w_taper*epsilon0*m_e/(q_e*rho_lim*(theta*dt)^2), or zero in true
     * vacuum. Caller uses exactly its native deposited rho/pedestal/floor/taper
     * semantics. No caller ghosts are consumed. False for nonfinite/negative
     * valid values.
     */
    bool Freeze (const amrex::MultiFab& kappa);

    /** Own exact nonnegative coefficients already collocated on native Yee
     * edges. Applies DG+D(kappa_edge G). This is an auxiliary full-grid
     * correction in mixed vacuum, not a physical DG vacuum equation.
     * Invalid input leaves the previous frozen operator unchanged. */
    bool FreezeEdges (const ConstVector& kappa);

    /** Solve (DG+DM*kappa*NG) dphi = D(raw_EL-held_EL).
     * Both inputs are Yee longitudinal vectors in V/m on this cell layout.
     * Outputs remain private even on failure; inspect converged before
     * applying.
     */
    LongitudinalSchurResult Correct (const ConstVector& raw_EL,
                                     const ConstVector& held_EL);

    /** Same solve from an already formed physical raw-minus-held defect. */
    LongitudinalSchurResult CorrectDefect (const ConstVector& defect);

    /** Nodal potential operator [DG+DM*kappa*NG], output in potential/length^2.
     * Homogeneous PEC phi=0, PMC/axis Neumann and periodic images.
     * Dirichlet rows are eliminated (zero), not part of the Krylov space.
     */
    void ApplyPotential (amrex::MultiFab& out,
                         const amrex::MultiFab& potential);
    /** Native divergence with owned boundary images and eliminated PEC rows.
     * No elliptic inversion is performed. Input guards are never consumed. */
    void ApplyDivergence (amrex::MultiFab& out, const ConstVector& field);
    /** Apply exactly preconditioner_cycles multigrid cycles from zero.
     * This is an approximate inverse, never a physical convergence receipt. */
    bool PreconditionPotential (amrex::MultiFab& out, const amrex::MultiFab& rhs);
    /** Solve this same potential equation from a nodal scalar RHS. */
    LongitudinalSchurResult SolvePotential (const amrex::MultiFab& rhs);
    const amrex::MultiFab& CorrectionPotential () const;
    ConstVector CorrectionField () const;

  private:
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::darwin
#endif
