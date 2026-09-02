/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_ELECTRONINERTIAELLIPTIC_H_
#define WARPX_ELECTRONINERTIAELLIPTIC_H_

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

/**
 * \brief Electron inertia in the generalized Ohm's law, in elliptic form.
 *
 * The inertialess generalized Ohm's law used by the hybrid solver,
 *
 *     E = -u_e x B - grad(p_e)/(e n) + eta J,
 *
 * carries a whistler branch with no high-wavenumber cutoff: with the Hall
 * term alone the dispersion relation is omega = Omega_ce (k d_e)^2, which
 * grows without bound as the wavelength approaches the grid scale. The
 * fastest representable mode therefore sets the magnetic substep CFL, and
 * on a grid with k_max d_e > 1 that mode oscillates faster than the electron
 * cyclotron frequency -- a purely discrete artefact with no physical
 * counterpart.
 *
 * Retaining the electron inertia term restores the cutoff. The electron
 * momentum equation contributes
 *
 *     E = RHS_inertialess + (m_e / (e^2 n)) dJ/dt,
 *
 * and substituting Ampere's law (J = curl B / mu0) followed by Faraday's law
 * (dB/dt = -curl E) *before* discretising in time converts the inertia term
 * into a spatial operator acting on E itself:
 *
 *     E + d_e^2 curl(curl E) = RHS_inertialess,   d_e^2 = m_e / (mu0 e^2 n).
 *
 * The resulting whistler dispersion relation is
 *
 *     omega = Omega_ce (k d_e)^2 / (1 + (k d_e)^2)  ->  Omega_ce,
 *
 * i.e. bounded by the electron cyclotron frequency for every wavenumber the
 * grid can represent.
 *
 * The decisive property of this form is that it is ALGEBRAIC IN TIME. No
 * electron-current state is stored, nothing is lagged across a substep, and
 * nothing can go stale across a restart: each call solves a self-contained
 * elliptic problem for E given the inertialess right-hand side.
 *
 * DELIBERATE OMISSION -- the advective inertia term. The full convective
 * derivative would add -(m_e/e)(u_e . grad) u_e to Ohm's law. It is dropped
 * here on a direct measurement rather than for convenience: evaluated on
 * dumps of the magnetised, strongly sheared regime this solver is aimed at,
 * it came out three to five orders of magnitude below both the Hall term and
 * the total solved |E|, consistent with its structural suppression by the
 * ratio of the electron gyroradius to the gradient scale length. Anyone
 * porting this to a regime where that ratio is not small should re-measure
 * before relying on the omission. It is also the term whose omission
 * preserves the algebraic-in-time property above.
 *
 * DISCRETISATION. curl(curl E) is not assembled from a standalone stencil.
 * It is formed as the exact composition of the two discrete curls the hybrid
 * loop already applies: the E -> B curl of Faraday's law (upward differences)
 * followed by the B -> E curl of Ampere's law (downward differences),
 * including the on-axis regularisations in cylindrical geometry. Because the
 * operator is the same composition that generates the instability, the
 * bound on the whistler frequency holds for the *discrete* dispersion
 * relation on the actual grid, not merely in the continuum limit.
 *
 * SOLVER. The system is (I + d_e^2 C) E = RHS with C the discrete curl-curl.
 * It is solved for the correction dE = E - RHS, which obeys homogeneous
 * boundary conditions because RHS already satisfies the physical ones. The
 * iteration is Jacobi-preconditioned BiCGStab; the operator is very well
 * conditioned (after diagonal scaling the spread is set by max (k d_e)^2,
 * of order unity on production grids), so a handful of iterations suffices.
 *
 * BOUNDARY CONDITIONS. The iterate reuses WarpX's own E-field boundary
 * machinery rather than a separate mapping, which keeps the elliptic problem
 * consistent with the explicit solve by construction:
 *   Periodic  -> FillBoundary with the level periodicity.
 *   PEC       -> PEC::ApplyPECtoEfield (tangential E zeroed, guard cells
 *                mirrored). This is a linear, homogeneous operation, so
 *                applying it to the correction imposes homogeneous PEC.
 *   PMC /
 *   Neumann   -> PEC::ApplyPECtoBfield applied to E, likewise linear.
 *   None      -> no-op; this is the r = 0 cylindrical axis, where the
 *                regularisations are carried inside the curl stencils.
 * Every other FieldBoundaryType (PML, Damped, Absorbing_Silver_Mueller,
 * Open, PEC_Insulator) is REJECTED at setup with an explicit message rather
 * than silently approximated: those conditions are either affine in E or
 * carry their own auxiliary state, and neither has a homogeneous linear
 * counterpart that could be applied to a Krylov iterate.
 */
class ElectronInertiaElliptic
{
public:
    using EVec = amrex::Array<amrex::MultiFab, 3>;

    ElectronInertiaElliptic () = default;

    /** Allocate scratch and coefficient MultiFabs matching the E-field layout. */
    void Define (ablastr::fields::VectorField const& Efield, int lev);

    [[nodiscard]] bool IsDefined () const { return m_defined; }

    /**
     * \brief Rebuild d_e^2 and the Jacobi diagonal from the charge density.
     *
     * Called on every solve and rebuilt unconditionally: it is three grid
     * passes against a solve of order a hundred, and every cheap staleness
     * signature either moves by itself (MultiFab::sum is not
     * bit-reproducible across boxes) or can collide, which here would mean
     * silently solving with a stale d_e^2. See the implementation for the
     * measurement. Setup and solve costs are accumulated separately (see
     * m_setup_time / m_solve_time) so the split stays visible in the
     * runtime report.
     */
    void PrepareCoefficients (amrex::MultiFab const& rhofield,
                              amrex::Real rho_floor, int lev);

    /**
     * \brief Apply the inertia correction in place: E <- (I + d_e^2 C)^{-1} E.
     *
     * On entry Efield holds the inertialess Ohm's-law solution; on exit it
     * holds the field including electron inertia. Aborts if the iteration
     * fails to reach m_rtol within m_max_iter -- a partially converged E is
     * never accepted, because a loose solve silently restores the unbounded
     * branch this term exists to remove.
     */
    void Solve (ablastr::fields::VectorField const& Efield, int lev);

    /** Emit the accumulated iteration and timing statistics, then reset
     *  them. \p substeps is the magnetic substep count the step ran at,
     *  reported alongside the solve cost because per-substep cost cannot be
     *  recovered from a step wall time without it. */
    void ReportAndResetStats (int substeps);

    // ---- runtime knobs (see HybridPICModel::ReadParameters) ----
    amrex::Real m_rtol = 1.e-8;
    int m_max_iter = 200;
    int m_verbose = 0;
    /** Seed each solve with the previous substep's correction. E moves by
     *  O(dt_sub dE/dt) between substeps, so the previous correction is a
     *  much better starting guess than zero. */
    bool m_warm_start = true;

    // ---- statistics, accumulated between reports ----
    long m_n_solves = 0;
    long m_n_iters = 0;
    int m_max_iters_seen = 0;
    long m_n_setups = 0;
    double m_setup_time = 0.0;
    double m_solve_time = 0.0;
    /** Iterations of the solves started from a zero guess rather than from
     *  the previous substep's correction, so warm starting is measurable. */
    long m_n_cold_solves = 0;
    long m_n_cold_iters = 0;

    // Everything below is public because the member functions contain device
    // lambdas, which nvcc does not accept in private or protected members.

    /** out <- X + d_e^2 curl(curl X), with boundary conditions applied to X. */
    void ApplyOperator (EVec& X, EVec& out, int lev);

    /** Fill ghost cells of a Krylov iterate: periodic/MPI exchange plus the
     *  homogeneous form of the physical E boundary conditions. */
    void FillIterateBoundary (EVec& X, int lev);

    /** Impose the homogeneous physical boundary conditions on a vector,
     *  without a ghost exchange. Forced values (tangential E at PEC, normal
     *  E at PMC) are not unknowns of the elliptic system, so this is applied
     *  to the operator output and to the right-hand side as well as to the
     *  iterate -- otherwise those rows make the system inconsistent. */
    void ApplyHomogeneousBC (EVec& X, int lev);

    /** Validate that every field boundary has a homogeneous linear form. */
    static void CheckBoundarySupport ();

private:
    bool m_defined = false;
    bool m_coefs_valid = false;

    /** d_e^2 = m_e / (mu0 e rho_floored), at the E-field staggering. */
    EVec m_de2;
    /** Reciprocal of the operator diagonal, the Jacobi preconditioner. */
    EVec m_jac;
    /** Working vectors: the previous correction (warm start), and the
     *  BiCGStab state r, rhat, p, v, s, t, plus one operator scratch. */
    EVec m_x, m_r, m_rhat, m_p, m_v, m_s, m_t, m_b, m_ax;
    /** curl of the iterate, at the B-field staggering. */
    EVec m_curl;
};

#endif // WARPX_ELECTRONINERTIAELLIPTIC_H_
