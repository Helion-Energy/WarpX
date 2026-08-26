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
#include "NonlinearSolvers/MHDResistiveStencil.H"
#include "NonlinearSolvers/NonlinearSolverLibrary.H"

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_REAL.H>
#include <AMReX_LO_BCTYPES.H>

#include <string>

/**
 * \brief Base class for implicit time solvers. The base functions are those
 *  needed by an implicit solver to be used through WarpX and those needed
 *  to use the nonlinear solvers, such as Picard or Newton (i.e., JFNK).
 */

class WarpX;
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

    /**
     * \brief Allocate scheme-specific fields while a WarpX level is created.
     *
     * Most implicit schemes need no additional state and use this no-op
     * implementation.
     */
    virtual void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) const;

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
    /**
     * \brief Restrict a nonlinear-solver update to the admissible state domain.
     *
     * The proposed candidate is
     * `a_state + returned_step * a_direction`. The base implementation leaves
     * the requested step unchanged. Fluid solvers can override this hook to
     * keep Newton updates and matrix-free Jacobian probes inside positivity
     * bounds without imposing those constraints on unrelated implicit schemes.
     */
    [[nodiscard]] virtual amrex::Real
    LimitSolverStep (const WarpXSolverVec& a_state,
                     const WarpXSolverVec& a_direction,
                     amrex::Real a_requested_step) const
    {
        amrex::ignore_unused(a_state, a_direction);
        return a_requested_step;
    }

    /**
     * \brief Like LimitSolverStep, but may additionally PROJECT the update
     * direction (active-set treatment): a cell resting exactly on its
     * admissibility bound with a descent direction component would
     * otherwise zero the GLOBAL step fraction and deadlock the Newton
     * line search; zeroing that component instead lets the rest of the
     * update proceed while the bound-resident cell stays put. Called only
     * on the Newton-update path -- never on matrix-free Jacobian probe
     * vectors, whose Krylov data must not be mutated. The base
     * implementation performs no projection.
     */
    [[nodiscard]] virtual amrex::Real
    ProjectAndLimitSolverStep (const WarpXSolverVec& a_state,
                               WarpXSolverVec& a_direction,
                               const amrex::Real a_requested_step) const
    {
        return LimitSolverStep(a_state, a_direction, a_requested_step);
    }

    /**
     * \brief Whether Newton updates should use residual-decreasing
     * backtracking.
     */
    [[nodiscard]] virtual bool
    UseNewtonLineSearch () const
    {
        return false;
    }

    /**
     * \brief L2 norm of a_residual restricted to the FREE subspace: every
     * component EXCEPT those whose direction components the most recent
     * ProjectAndLimitSolverStep call clamped onto their admissibility
     * bounds (the Newton active set). Also returns the complementary
     * PINNED-defect norm (the residual L2 norm over the clamped
     * components alone) and the global clamped-component count. The
     * Newton line search consults this ONLY after the full-norm Armijo
     * backtracking of iteration 0 has failed every backtrack: the full
     * norm can then be dominated by rows whose variables the projection
     * froze, which no admissible step is allowed to reduce, and the
     * solve would otherwise count a frozen step despite productive free
     * updates being available. The base implementation has
     * no projection, so its active set is empty by construction: the
     * free norm IS the full norm and the defect is zero -- solvers
     * without a projecting override keep bit-identical behavior.
     */
    [[nodiscard]] virtual amrex::Real
    FreeResidualNorm (const WarpXSolverVec& a_residual,
                      amrex::Real& a_pinned_defect,
                      amrex::Long& a_num_pinned) const
    {
        a_pinned_defect = 0.0;
        a_num_pinned = 0;
        return a_residual.norm2();
    }

    /**
     * \brief Human-readable per-block counts of the direction components
     * the most recent ProjectAndLimitSolverStep call clamped (e.g.
     * "mass 12, electron_energy 3, ion_energy 230"), for the Newton
     * freeze diagnostics. Empty when nothing was clamped or the solver
     * has no projection.
     */
    [[nodiscard]] virtual std::string
    PinnedComponentReport () const
    {
        return {};
    }

    /**
     * \brief Whether matrix-free Jacobian probe perturbations must be
     * limited to the admissible state domain. Operators whose residual
     * evaluation clamps every floored quantity pointwise can return
     * false: probe states are then evaluable even when transiently
     * inadmissible, which keeps the Jacobian exact for states resting on
     * their admissibility bounds.
     */
    [[nodiscard]] virtual bool
    ProbePerturbationsNeedLimiting () const
    {
        return true;
    }

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
     * \brief Return theta for the MHD block preconditioner.
     *
     * This is separate from GetThetaForPC(), whose established meaning is the
     * implicit light-wave fraction used by electromagnetic preconditioners.
     */
    [[nodiscard]] virtual amrex::Real
    GetMHDThetaForPC () const
    {
        return -1.0;
    }

    /**
     * \brief Evaluate the scalar reference resistivity used by the MHD block
     * preconditioner.
     *
     * The base sentinel makes selecting pc_mhd_block for an unsupported
     * implicit scheme fail during its first update with a descriptive
     * assertion.
     */
    [[nodiscard]] virtual amrex::Real
    GetMHDReferenceResistivityForPC (amrex::Real a_time) const
    {
        amrex::ignore_unused(a_time);
        return -1.0;
    }

    /**
     * \brief Return the resistive-stage implicit fraction theta_r used by the
     * MHD block preconditioner's resistive Helmholtz block.
     *
     * The dissipative Ohm terms of the theta-implicit MHD residual may be
     * centered at theta_r != theta (resistive backward Euler); the
     * preconditioner must weight its resistive approximation with the same
     * fraction. The negative sentinel makes selecting the resistive block for
     * an implicit scheme that does not expose it fail with a descriptive
     * assertion.
     */
    [[nodiscard]] virtual amrex::Real
    GetMHDResistiveThetaForPC () const
    {
        return -1.0;
    }

    /**
     * \brief Return the cell-centered field resistivity eta_field (user
     * resistivity through the density-keyed vacuum boost, with the Ohm floor
     * guards) evaluated at the preconditioner update state, read by the MHD
     * block preconditioner's resistive block gate under the
     * conservative-form recast.
     *
     * The null default keeps that block inactive for implicit schemes that do
     * not expose it.
     */
    [[nodiscard]] virtual const amrex::MultiFab*
    GetMHDFieldResistivityCCForPC (amrex::Real a_time) const
    {
        amrex::ignore_unused(a_time);
        return nullptr;
    }

    /**
     * \brief Return eta_field evaluated on the electric-field staggerings
     * (per physical component) at the preconditioner update state, with the
     * SAME density interpolation the residual's Ohm assembly uses: the MHD
     * block preconditioner's resistive block iterates on the discrete
     * resistive curl-curl, and eta_field is strongly nonlinear in the
     * density (the vacuum boost is quadratic in 1/rho), so averaging
     * cell-centered eta instead of evaluating eta at the interpolated
     * density misrepresents the operator by order unity across halo
     * density gradients.
     *
     * The null defaults keep that block inactive for implicit schemes that
     * do not expose it.
     */
    [[nodiscard]] virtual amrex::Array<const amrex::MultiFab*, 3>
    GetMHDFieldResistivityEdgeForPC (amrex::Real a_time) const
    {
        amrex::ignore_unused(a_time);
        return {nullptr, nullptr, nullptr};
    }

    /**
     * \brief Return whether the generalized Ohm law includes the
     * hyper-resistive term eta_H, for the MHD block preconditioner (static
     * per run: sizes the resistive block's stencil scratch and the banded
     * solver's bandwidth at Define time).
     */
    [[nodiscard]] virtual bool
    GetMHDIncludeHyperResistivityForPC () const
    {
        return false;
    }

    /**
     * \brief Return eta_H evaluated on the electric-field staggerings at
     * the preconditioner update state, with the residual's interpolated
     * charge-density and cell-centered |B| arguments frozen, read by the
     * MHD block preconditioner's resistive block. Null components when
     * the hyper-resistive term is off.
     */
    [[nodiscard]] virtual amrex::Array<const amrex::MultiFab*, 3>
    GetMHDHyperResistivityEdgeForPC (amrex::Real a_time) const
    {
        amrex::ignore_unused(a_time);
        return {nullptr, nullptr, nullptr};
    }

    /**
     * \brief Return the physical xyz reference magnetic field used by the MHD
     * wave preconditioner.
     *
     * The zero default keeps this coupling inactive for implicit schemes that
     * do not expose an MHD background field.
     */
    [[nodiscard]] virtual amrex::GpuArray<amrex::Real, 3>
    GetMHDReferenceMagneticFieldForPC () const
    {
        return {0.0, 0.0, 0.0};
    }

    /**
     * \brief Return whether the generalized Ohm law includes the Hall current
     * for the MHD block preconditioner.
     */
    [[nodiscard]] virtual bool
    GetMHDIncludeHallTermForPC () const
    {
        return false;
    }

    /**
     * \brief Return the frozen Hall coefficient vector Hb = B/rho_q (the
     * cell-centered total magnetic field over the Ohm-floored charge
     * density) evaluated on the electric-field staggerings with the
     * residual Ohm assembly's own interpolation stencils, read by the MHD
     * block preconditioner's Hall/whistler rows under the
     * conservative-form recast. Null components when the Hall term is off
     * (or for implicit schemes that do not expose it).
     */
    [[nodiscard]] virtual amrex::Array<const amrex::MultiFab*, 3>
    GetMHDHallCoefficientEdgeForPC () const
    {
        return {nullptr, nullptr, nullptr};
    }

    /**
     * \brief Return whether the generalized Ohm law includes the
     * electron-inertia term for the MHD block preconditioner.
     */
    [[nodiscard]] virtual bool
    GetMHDIncludeElectronInertiaForPC () const
    {
        return false;
    }

    /**
     * \brief Return the dt-independent stencil factor s = dt c_p1 of the
     * frozen electron-inertia response ((2 theta + 1)/2 with the
     * three-point dJe/dt stencil engaged, 1 for the two-point form), read
     * by the MHD block preconditioner's inertia rows.
     */
    [[nodiscard]] virtual amrex::Real
    GetMHDElectronInertiaStencilScaleForPC () const
    {
        return 0.0;
    }

    /**
     * \brief Return the frozen electron-inertia coefficient
     * C_i = m_e_eff/(e rho_q) (= mu0 d_e^2 at the Ohm-floored charge
     * density) evaluated on the electric-field staggerings with the
     * residual Ohm assembly's own interpolation stencils, read by the MHD
     * block preconditioner's inertia rows under the conservative-form
     * recast. Null components when the term is off (or for implicit
     * schemes that do not expose it).
     */
    [[nodiscard]] virtual amrex::Array<const amrex::MultiFab*, 3>
    GetMHDElectronInertiaCoefficientEdgeForPC () const
    {
        return {nullptr, nullptr, nullptr};
    }

    /**
     * \brief Return whether the implicit MHD state advances the ion fluid.
     *
     * When false, the MHD block preconditioner treats the ion density and
     * momentum rows as identities and applies only the electron-MHD field
     * block.
     */
    [[nodiscard]] virtual bool
    GetMHDEvolveIonFluidForPC () const
    {
        return true;
    }

    /**
     * \brief Return the positive reference ion charge density used by the Hall
     * field preconditioner.
     */
    [[nodiscard]] virtual amrex::Real
    GetMHDReferenceChargeDensityForPC () const
    {
        return -1.0;
    }

    /**
     * \brief Return the cell-centered magnetic field (including any external
     * contribution) at the preconditioner update state, read by the MHD
     * block preconditioner's triangular Faraday corrector under the
     * conservative-form recast.
     *
     * The null default keeps that coupling inactive for implicit schemes
     * that do not expose it.
     */
    [[nodiscard]] virtual const amrex::MultiFab*
    GetMHDMagneticFieldCCForPC () const
    {
        return nullptr;
    }

    /**
     * \brief Return the device view of the stair-step conducting wall
     * mask (implicit_mhd.wall_model = pec/pec_response) so the MHD block
     * preconditioner's resistive stencil emission drops the same masked
     * tangential-E rows the residual projects. The inactive default
     * keeps the emission unchanged for implicit schemes without a wall
     * -- and for wall_model = dielectric, whose EM-transparent standoff
     * reports an inactive view by design (no rows to drop).
     */
    [[nodiscard]] virtual warpx::mhd_pc::WallMaskView
    GetMHDWallMaskForPC () const
    {
        return {};
    }

    /**
     * \brief Return the device view of the shaped wall's rigid-conductor
     * FLUID freeze (implicit_mhd.wall_thermal_bc) so the MHD block
     * preconditioner's cell-centered fluid sweeps mirror the residual's
     * masked-band identity rows (wall_live). Active in EVERY wall_model
     * with a thermal wall BC -- including dielectric, whose FIELD-side
     * WallMaskView is inactive by design. The inactive default keeps the
     * fluid sweeps unchanged for schemes without a shaped wall.
     */
    [[nodiscard]] virtual warpx::mhd_pc::WallFluidFreezeView
    GetMHDWallFluidFreezeForPC () const
    {
        return {};
    }

    /**
     * \brief Return the device view of the shaped wall's FIELD freeze
     * (implicit_mhd.wall_field_freeze) so the MHD block preconditioner
     * mirrors the residual's frozen magnetic-field identity rows: the
     * resistive stencil emission drops the frozen rows (keeping their
     * columns in live rows -- the exact block-triangular Jacobian
     * structure) and the preconditioner restores dB = r_B on the frozen
     * faces after the field-block inverses. The inactive default keeps
     * schemes without the freeze unchanged.
     */
    [[nodiscard]] virtual warpx::mhd_pc::WallFieldFreezeView
    GetMHDWallFieldFreezeForPC () const
    {
        return {};
    }

    /**
     * \brief Return true when the z_lo domain face is the theta-implicit
     * MHD mirror-symmetry plane (boundary.field_lo = pmc on z paired with
     * implicit_mhd.z_lo_boundary_fluid = symmetry): the MHD block
     * preconditioner's resistive stencil emission then keeps the z_lo
     * boundary-plane tangential-E rows and folds their ghost
     * magnetic-field columns with the mirror parities (tangential B odd,
     * normal B even) instead of dropping them. The default keeps the
     * tangential-E-zero end-row convention.
     */
    [[nodiscard]] virtual bool GetMHDZLoMirrorForPC () const
    {
        return false;
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
