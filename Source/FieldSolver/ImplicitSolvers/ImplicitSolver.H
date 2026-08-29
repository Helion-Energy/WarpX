/* Copyright 2024 Justin Angus, Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef Implicit_Solver_H_
#define Implicit_Solver_H_

#include "ImplicitSolver_fwd.H"

#include "FieldSolver/ImplicitSolvers/ImplicitOptions.H"

#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "NonlinearSolvers/NonlinearSolverLibrary.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_LO_BCTYPES.H>

/**
 * \brief Base class for implicit time solvers. The base functions are those
 *  needed by an implicit solver to be used through WarpX and those needed
 *  to use the nonlinear solvers, such as Picard or Newton (i.e., JFNK).
 */

class WarpX;
class HybridPICModel;
enum class FieldBoundaryType;
class ImplicitSolver
{
public:

    ImplicitSolver() = default;

    virtual ~ImplicitSolver() = default;

    // Prohibit Move and Copy operations
    ImplicitSolver(const ImplicitSolver&) = delete;
    ImplicitSolver& operator=(const ImplicitSolver&) = delete;
    ImplicitSolver(ImplicitSolver&&) = delete;
    ImplicitSolver& operator=(ImplicitSolver&&) = delete;

    //
    // the following routines are called by WarpX
    //

    /**
     * \brief Read user-provided parameters that control the implicit solver.
     * Allocate internal arrays for intermediate field values needed by the solver.
     */
    virtual void Define (WarpX* a_WarpX, bool from_restart) = 0;

    [[nodiscard]] bool IsDefined () const { return m_is_defined; }

    virtual void PrintParameters () const = 0;

    void CreateParticleAttributes () const;

    bool DoParticleSuborbits () const { return m_particle_suborbits; }

    /**
     * \brief Advance fields and particles by one time step using the specified implicit algorithm
     */
    virtual int OneStep (amrex::Real  a_time,
                         amrex::Real  a_dt,
                         int          a_step) = 0;

    //
    // the following routines are called by the linear and nonlinear solvers
    //

    /**
     * \brief Computes the RHS of the equation corresponding to the specified implicit algorithm.
     * The discrete equations corresponding to numerical integration of ODEs are often
     * written in the form U = b + RHS(U), where U is the variable to be solved for (e.g.,
     * the solution at the next time step), b is a constant (i.e., the solution from the
     * previous time step), and RHS(U) is the right-hand-side of the equation. Iterative
     * solvers, such as Picard and Newton, and higher-order Runge-Kutta methods, need to
     * compute RHS(U) multiple times for different values of U. Thus, a routine that
     * returns this value is needed.
     * e.g., Ebar - E^n = cvac^2*0.5*dt*(curl(Bbar) - mu0*Jbar(Ebar,Bbar))
     * Here, U = Ebar, b = E^n, and the expression on the right is RHS(U).
     */
    virtual void ComputeRHS ( WarpXSolverVec&  a_RHS,
                        const WarpXSolverVec&  a_E,
                              amrex::Real      a_time,
                              int              a_nl_iter,
                              bool             a_from_jacobian ) = 0;

    /**
     * Prepare for the linear solve. This includes depositing the mass matrices and updating the PC.
     */
    void PreLinearSolve ();

    [[nodiscard]] int numAMRLevels () const { return m_num_amr_levels; }

    [[nodiscard]] const amrex::Geometry& GetGeometry (int) const;
    [[nodiscard]] const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& GetFieldBoundaryLo () const;
    [[nodiscard]] const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>& GetFieldBoundaryHi () const;
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> GetLinOpBCLo () const;
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> GetLinOpBCHi () const;

    /**
     * \brief Return pointer to MultiFab array for mass matrix
     */
    [[nodiscard]] const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* GetMassMatricesCoeff() const {
        if (m_use_mass_matrices_pc) { return &m_mmpc_mfarrvec; }
        else { return nullptr; }
    }

    /**
     * \brief Return reference to the multifabs for curl curl mask
     */
    [[nodiscard]] virtual const amrex::MultiFab* GetCurl2BCmask (const int lev, const int dir) const
    {
        amrex::ignore_unused(lev,dir);
        return nullptr;
    }

    /**
     * \brief Return the space_dir number of components for the mass matrices used in
     *        the preconditioner for field direction field_dir
     */
    int GetMassMatricesPCnComp (const int field_dir, const int space_dir) const
    {
        if      (field_dir == 0) { return m_ncomp_pc_xx[space_dir]; }
        else if (field_dir == 1) { return m_ncomp_pc_yy[space_dir]; }
        else if (field_dir == 2) { return m_ncomp_pc_zz[space_dir]; }
        else { return -1; }
    }

    // This parameter is used for the time-step fraction in the PC for implicit
    // treatment of light waves in the curl-curl MLMG solver.
    // This function should return zero if light waves are not treated implicitly
    [[nodiscard]] virtual amrex::Real  GetThetaForPC () const = 0;

    /**
     * \brief Return the hybrid-PIC model for preconditioners that need the
     * generalized Ohm's-law coefficients (n_floor, resistivity and
     * hyper-resistivity parsers, Hall flags). Returns nullptr for solvers
     * that do not carry a hybrid model.
     */
    virtual const HybridPICModel* GetHybridPICModel () const { return nullptr; }

    /**
     * \brief Return per-direction pointers to the theta-midpoint magnetic
     * field B^{n+theta} on level lev for use in a preconditioner. Valid only
     * after the first residual evaluation of the current Newton iterate has
     * refreshed the field registry. Solvers that do not maintain a
     * theta-midpoint B return nullptrs.
     */
    virtual amrex::Array<const amrex::MultiFab*, 3> GetBfieldThetaForPC (int lev) const
    {
        amrex::ignore_unused(lev);
        return {nullptr, nullptr, nullptr};
    }

    /**
     * \brief Return the charge density holding the midpoint deposit
     * rho^{n+1/2} on level lev for use in a preconditioner (see the override
     * for the component convention). Valid only after the first residual
     * evaluation of the current Newton iterate has deposited it. Solvers
     * that do not maintain a midpoint density return nullptr.
     */
    virtual const amrex::MultiFab* GetRhoMidForPC (int lev) const
    {
        amrex::ignore_unused(lev);
        return nullptr;
    }

    /**
     * \brief Return the per-step-frozen rho^n snapshot the curlcurl_form
     * poloidal stage reads for its operator content and gates
     * (hybrid_rho_n_frozen, component 0), for preconditioners that mirror
     * that stage. Returns nullptr when the snapshot is not in play (e_form
     * decks, curlcurl_pol_frozen_rho = 0, or before the first capture of a
     * run) -- callers then fall back to the midpoint density, matching the
     * solve's own fallback.
     */
    virtual const amrex::MultiFab* GetRhoPolFrozenForPC (int lev) const
    {
        amrex::ignore_unused(lev);
        return nullptr;
    }

    /**
     * \brief Fill and return E-staggered beta coefficients for the
     * electron-inertia curl-curl preconditioner. The inertia operator in
     * divided form is beta(x) E + curl curl E = beta(x) b with
     * beta = 1/d_e^2 = mu0 e^2 n_lim / m_e_eff, so a non-null return puts
     * CurlCurlMLMGPC into hybrid-inertia mode (unit curl-curl coefficient,
     * MultiFab beta, RHS scaled by beta). Called by CurlCurlMLMGPC::Update()
     * at every preconditioner rebuild; solvers without an inertia operator
     * return nullptr (the default), keeping the electromagnetic behavior.
     */
    virtual const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>* FillInertiaBetaCoeff ()
    {
        return nullptr;
    }

    /**
     * \brief Return per-direction pointers to the ion (particle) current
     * deposit consumed by the Ohm solve on level lev, for preconditioners
     * that model the frozen motional/drift response (J - J_i) x delta_B.
     * During Jacobian probes the deposits are frozen, so this is exactly
     * the coefficient the linear solve sees. Solvers without an ion
     * current return nullptrs.
     */
    virtual amrex::Array<const amrex::MultiFab*, 3> GetIonCurrentForPC (int lev) const
    {
        amrex::ignore_unused(lev);
        return {nullptr, nullptr, nullptr};
    }

    /**
     * \brief Apply deposited mass matrices to a vector field.
     * Computes a_out += a_scale * S * (a_in - a_in_ref) [+ a_baseline],
     * where S is the mass matrix, accumulating into a_out.
     * If a_in_ref is null, a_in is used directly.
     * If a_baseline is null, no baseline term is added.
     * If a_zero_out_first is true, a_out is set to zero before accumulating,
     * so that it holds the result of this operation only; otherwise the result
     * is added on top of the existing contents of a_out.
     */
    void ApplyMassMatrices (
        ablastr::fields::MultiLevelVectorField& a_out,
        const ablastr::fields::MultiLevelVectorField& a_in,
        const ablastr::fields::MultiLevelVectorField* a_in_ref = nullptr,
        const ablastr::fields::MultiLevelVectorField* a_baseline = nullptr,
        amrex::Real a_scale = 1.0,
        bool a_zero_out_first = false );

    void ComputeJfromMassMatrices (bool a_J_from_MM_only);

    /**
     * \brief Finish filling the mass matrices after deposit
     */
    void FinishMassMatrices ();

protected:

    /**
     * \brief Pointer back to main WarpX class
     */
    WarpX* m_WarpX;

    bool m_is_defined = false;

    /**
     * \brief Number of AMR levels
     */
    int m_num_amr_levels = 1;

    /**
     * \brief Time step
     */
    mutable amrex::Real m_dt = 0.0;

    /**
     * \brief Time-biasing parameter for fields used on RHS to advance system.
     */
    amrex::Real m_theta = 0.5;

    /**
     * \brief Nonlinear solver type and object
     */
    NonlinearSolverType m_nlsolver_type;
    std::unique_ptr<NonlinearSolver<WarpXSolverVec,ImplicitSolver>> m_nlsolver;

    /**
     * \brief tolerance used by the iterative method used to obtain a self-consistent
     *  update of the particle positions and velocities for given E and B on the grid
     */
    amrex::ParticleReal m_particle_tolerance = amrex::ParticleReal(1.0e-10);

    /**
     * \brief maximum iterations for the iterative method used to obtain a self-consistent
     *  update of the particle positions and velocities for given E and B on the grid
     */
    int m_max_particle_iterations = 21;

    /**
     * \brief whether to use suborbits for particles that fail to converge in
     * m_max_particle_iterations iterations
     */
    bool m_particle_suborbits = false;

    /**
     * \brief whether to specularly reflect particles at the r-max domain boundary
     * inside the implicit per-particle Picard iteration, so that wall-crossing
     * orbits are reflected during the nonlinear solve rather than only once per
     * step after it. Only active in RZ geometry with a reflecting particle
     * boundary at r-hi.
     */
    bool m_reflect_particles_at_rmax = false;

    /**
     * \brief whether to overwrite the r-max guard cells of the particle-gather
     * electric field with the discrete energy-adjoint of the guard-current fold
     * applied to the deposited J at a reflecting r-max wall (per-component parity
     * mirror: E_r odd, E_theta/E_z even). Only active in RZ geometry with a PEC
     * field boundary and a Reflecting particle boundary at r-hi, where the fold
     * parity (particle-Reflecting precedence) is otherwise not adjoint to the
     * PEC-image ghosts the gather reads, making the wall band a standing
     * discrete energy source: Sum_p q v.E(x_p) != integral J.E there.
     */
    bool m_adjoint_gather_ghosts = false;

    /**
     * \brief whether the details of unconverged particles are printed out during
     * the particle evolve loops
     */
    bool m_print_unconverged_particle_details = false;

    /**
     * \brief whether to skip the full Picard update of particles on the initial newton step
     */
    bool m_skip_particle_picard_init = false;

    /**
     * \brief Whether to use mass matrices for the implicit solver
     */
    bool m_use_mass_matrices = false;

    /**
     * \brief Whether to use mass matrices to compute J during linear stage of JFNK
     */
    bool m_use_mass_matrices_jacobian = false;

    /**
     * \brief Whether to use mass matrices in the preconditioner
     */
    bool m_use_mass_matrices_pc = false;

    /**
     * \brief Width for mass matrices in the preconditioner
     */
    int m_mass_matrices_pc_width = -1;

    /**
     * \brief Whether to fold the physical-boundary response into the mass
     * matrices themselves. The deposit leaves raw band entries with guard
     * rows and guard columns at physical boundaries; when enabled, those are
     * folded into the interior band with the deposit-side guard-current
     * parities (row fold) and the gather-side ghost-image parities (column
     * remap), so the stored bands represent the composed interior operator.
     * Guard rows/columns are zeroed after folding, which leaves the
     * mass-matrix matvec unchanged (the folded content enters through
     * interior entries instead of through the vector-side guard fold and
     * ghost-image fill). Currently implemented for the RZ r-hi boundary with
     * field PEC + particle Reflecting; see ApplyMassMatricesBoundaryRows.
     */
    bool m_mass_matrices_boundary_rows = false;

    /**
     * \brief One-shot verification of the mass-matrix Jacobian action against
     * the true (particle) current response: when >= 0, at this step index the
     * first mass-matrix linear-stage evaluation is compared row-by-row in
     * radius against a full particle push/deposit at the identical field
     * state, the per-row relative errors are printed, and the run aborts.
     * Debug instrumentation; default off (-1).
     */
    int m_verify_mm_jvp_step = -1;

    /**
     * \brief Internal latch so the mass-matrix JVP verification runs once.
     */
    bool m_verify_mm_jvp_done = false;

    /**
     * \brief Direction-dependent component number of mass matrices elements
     */
    amrex::IntVect m_ncomp_xx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_xy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_xz = amrex::IntVect{0};
    amrex::IntVect m_ncomp_yx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_yy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_yz = amrex::IntVect{0};
    amrex::IntVect m_ncomp_zx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_zy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_zz = amrex::IntVect{0};

    /**
     * \brief Direction-dependent component number of mass matrices elements
     *        used for the preconditioner
     */
    amrex::IntVect m_ncomp_pc_xx = amrex::IntVect{0};
    amrex::IntVect m_ncomp_pc_yy = amrex::IntVect{0};
    amrex::IntVect m_ncomp_pc_zz = amrex::IntVect{0};

    /**
     * \brief Array of multifab pointers to mass matrix preconditioner
     */
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> m_mmpc_mfarrvec;

    /**
     * \brief parse nonlinear solver parameters (if one is used)
     */
    void parseNonlinearSolverParams ( const amrex::ParmParse&  pp );

    void SaveEoldMultifab ();

    /**
     * \brief Initialize the Mass Matrices used for plasma response in nonlinear Newton solver
     */
    void InitializeMassMatrices ();

    void PrintBaseImplicitSolverParameters () const;

    /**
     * \brief Perform operations needed before computing the Right Hand Side
     */
    void PreRHSOp ( amrex::Real  a_cur_time,
                    int          a_nl_iter,
                    bool         a_from_jacobian );

    // The next two members are public: they contain extended device
    // lambdas (ParallelFor), which nvcc 13 requires to live in public
    // member functions (the 2fc85ed9b rule).
    public:
    /**
     * \brief Overwrite the r-max guard cells of Efield_fp (the particle gather
     * field) with the energy-adjoint ghost fill of the reflecting-wall J fold.
     * Called from PreRHSOp between the E filter and the particle push; gated on
     * m_adjoint_gather_ghosts and the RZ PEC/Reflecting r-hi boundary combination.
     */
    void ApplyAdjointGatherGhosts ();

    /**
     * \brief Fold the physical-boundary response into the mass-matrix bands
     * at the RZ r-hi wall (field PEC + particle Reflecting). Composes the
     * guard-row current fold (J_r odd, J_theta/J_z even) with the active
     * gather ghost-image column convention (PEC conductor image, or the
     * fold-adjoint image when adjoint_gather_ghosts is on), then zeroes the
     * folded guard rows and columns. The wall-node self-fold doubling of
     * nodal-in-r rows is deliberately left to the downstream guard-current
     * fold acting on the matvec result (see the implementation comments), so
     * the stored wall-node rows keep the raw-deposit half-normalized
     * convention. Called at the end of FinishMassMatrices when
     * mass_matrices_boundary_rows is enabled; must run after the band
     * symmetry completion, which reads the raw guard rows.
     */
    void ApplyMassMatricesBoundaryRows ();
    protected:

    /**
     * \brief One-shot comparison of the mass-matrix linear-stage current
     * against a full particle push/deposit at the identical field state,
     * reported per radial row. Aborts on completion by design (debug
     * instrumentation, gated by verify_mm_jvp_step).
     */
    void VerifyMassMatricesJVP ( amrex::Real a_cur_time );

    /**
     * \brief Communicate Mass Matrices used for PC and apply boundary conditions
     */
    void SyncMassMatricesPCAndApplyBCs ();

    /**
     * \brief Scale mass matrices used for PC by c^2*mu0*theta*dt and add 1 to diagonal terms
     */
    void SetMassMatricesForPC ( amrex::Real a_theta_dt );

    /**
     * \brief Add J from particles included in mass matrices at start of each Newton step
     */
    void CumulateJ ();

    /**
     * \brief Save E at start of each Newton step
     */
    void SaveE ();

    /**
     * \brief Convert from WarpX FieldBoundaryType to amrex::LinOpBCType
     */
    [[nodiscard]] amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> convertFieldBCToLinOpBC ( const amrex::Array<FieldBoundaryType,AMREX_SPACEDIM>&, int bdry_side ) const;

};

#endif
