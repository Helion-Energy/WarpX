/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef NEWTON_SOLVER_H_
#define NEWTON_SOLVER_H_

#include "LinearSolverLibrary.H"
#include "NonlinearSolver.H"
#include "JacobianFunctionMF.H"
#include "Preconditioner.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Config.H>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <istream>
#include <filesystem>

/**
 * \brief Newton method to solve nonlinear equation of form:
 * F(U) = U - b - R(U) = 0. U is the solution vector, b is a constant,
 * and R(U) is some nonlinear function of U, which is computed in the
 * ComputeRHS() Ops function.
 */

template<class Vec, class Ops>
class NewtonSolver : public NonlinearSolver<Vec,Ops>
{
public:

    NewtonSolver() = default;

    ~NewtonSolver() override = default;

    // Prohibit Move and Copy operations
    NewtonSolver(const NewtonSolver&) = delete;
    NewtonSolver& operator=(const NewtonSolver&) = delete;
    NewtonSolver(NewtonSolver&&) noexcept = delete;
    NewtonSolver& operator=(NewtonSolver&&) noexcept = delete;

    void Define (const Vec&  a_U,
                       Ops*  a_ops) override;

    void Solve (Vec&   a_U,
          const Vec&   a_b,
          amrex::Real  a_time,
          amrex::Real  a_dt,
          int          a_step) const override;

    void GetSolverParams (amrex::Real&  a_rtol,
                          amrex::Real&  a_atol,
                          int&          a_maxits) override
    {
        a_rtol = m_rtol;
        a_atol = m_atol;
        a_maxits = m_maxits;
    }

    inline void CurTime ( amrex::Real  a_time ) const
    {
        m_cur_time = a_time;
        m_linear_function->curTime( a_time );
    }

    inline void CurTimeStep ( amrex::Real  a_dt ) const
    {
        m_dt = a_dt;
        m_linear_function->curTimeStep( a_dt );
    }

    PreconditionerType GetPreconditionerType () const override
    {
        return m_linear_function->pcType();
    }

    void PrintParams () const override
    {
        amrex::Print()     << "Newton verbose:             " << (this->m_verbose?"true":"false") << "\n";
        amrex::Print()     << "Newton max iterations:      " << m_maxits << "\n";
        amrex::Print()     << "Newton relative tolerance:  " << m_rtol << "\n";
        amrex::Print()     << "Newton absolute tolerance:  " << m_atol << "\n";
        amrex::Print()     << "Newton require convergence: " << (m_require_convergence?"true":"false") << "\n";
        amrex::Print()     << "Newton max frozen steps:    " << m_max_frozen_steps << "\n";
        amrex::Print()     << "JFNK probe epsilon:         " << m_jfnk_epsilon << "\n";
        amrex::Print()     << "JFNK probe epsilon mode:    "
                           << amrex::getEnumNameString(m_jfnk_epsilon_mode) << "\n";
        if (m_jfnk_epsilon_mode == JFNKEpsilonMode::component) {
            for (std::size_t ib = 0; ib < m_component_floor_names.size(); ++ib) {
                amrex::Print() << "  component floor [" << m_component_floor_names[ib]
                               << "] = " << m_component_floor_fractions[ib]
                               << " x block scale\n";
            }
        }
        if (!m_jfnk_probe_report_file.empty()) {
            amrex::Print() << "JFNK probe report file:     " << m_jfnk_probe_report_file << "\n";
        }
        if (m_active_set) {
            amrex::Print() << "Newton active set:          true (reduced-space solve on the free subspace)\n";
            amrex::Print() << "Active-set tolerance:       " << m_active_set_tolerance << "\n";
        }
        auto linsol_name = amrex::getEnumNameString(m_linear_solver_type);
        amrex::Print()     << "Newton linear solver:       " << linsol_name << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") verbose:            " << m_linsol_verbose_int << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") restart length:     " << m_linsol_restart_length << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") max iterations:     " << m_linsol_maxits << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") relative tolerance: " << m_linsol_rtol << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") absolute tolerance: " << m_linsol_atol << "\n";
        amrex::Print()     << "Adaptive forcing:           " << (m_adaptive_forcing ? "true" : "false") << "\n";
        if (m_adaptive_forcing) {
            amrex::Print() << "Forcing alpha/gamma/max:    " << m_forcing_alpha
                           << " " << m_forcing_gamma << " " << m_forcing_max << "\n";
        }
        amrex::Print()     << "Preconditioner type:      " << amrex::getEnumNameString(m_pc_type) << "\n";

        m_linear_function->printParams();
    }

private:

    /**
     * \brief Intermediate Vec containers used by the solver.
     */
    mutable Vec m_dU, m_F, m_R, m_trial_U, m_trial_F;
    // Active-set mode: the residual restricted to the free subspace (the
    // reduced system's right-hand side), and a second restriction used to
    // measure free norms without cancellation.
    mutable Vec m_F_free, m_F_split;

    /**
     * \brief Active-set mode: the free/pinned split of a residual with the
     * free norm computed DIRECTLY as the norm of the residual restricted
     * to the free subspace. The operator's FreeResidualNorm returns
     * sqrt(full^2 - pinned^2), which loses the free part entirely once
     * the pinned defect dominates the norm by more than ~1e-8 (the
     * difference underflows to exactly zero, which would pass any
     * convergence test); the pinned defect and the count are taken from
     * the operator (its pinned sum is a direct sum, hence exact).
     */
    amrex::Real ActiveSetResidualSplit (const Vec& a_F,
                                        amrex::Real& a_pinned_defect,
                                        amrex::Long& a_num_pinned) const
    {
        m_ops->FreeResidualNorm(a_F, a_pinned_defect, a_num_pinned);
        if (a_num_pinned == 0) { return a_F.norm2(); }
        m_F_split.Copy(a_F);
        m_ops->ZeroActiveSetComponents(m_F_split);
        return m_F_split.norm2();
    }

    /**
     * \brief Pointer to Ops class.
     */
    Ops* m_ops = nullptr;

    /**
     * \brief Flag to determine whether convergence is required.
     */
    bool m_require_convergence = true;

    /**
     * \brief Maximum consecutive zero-progress solves before aborting.
     * A line-search stagnation at iteration 0 returns the state the
     * solve was handed: the step is a no-op and, for a system without
     * time-dependent drives, every later solve is bit-identical -- the
     * simulation silently freezes while time marches on. After this
     * many consecutive frozen solves the run aborts instead. Raise it
     * only when time-dependent sources can change the residual between
     * steps.
     */
    int m_max_frozen_steps = 10;

    /**
     * \brief Relative size of the matrix-free Jacobian probe: the state is
     * perturbed by eps*dU with eps = jfnk_epsilon * ||U|| / ||dU||, i.e. a
     * perturbation of jfnk_epsilon times the GLOBAL state norm along the
     * Krylov direction (Pernice-Walker form). The global norm is set by the
     * large blocks (fields, core density and energy), so at cells whose
     * state is orders of magnitude smaller (a plasma-vacuum edge) the same
     * probe is a large relative perturbation and the difference quotient
     * is a secant across strongly nonlinear terms rather than a derivative.
     * The default reproduces the historical hard-coded value; the classic
     * choice is the square root of machine epsilon (~1.5e-8).
     */
    amrex::Real m_jfnk_epsilon = 1.0e-6;

    /**
     * \brief Sizing rule of the probe (newton.jfnk_epsilon_mode): global
     * (the above, default) or component, where eps is sized so that every
     * component is perturbed by jfnk_epsilon of its own magnitude
     * |U_i| + floor in the root-mean-square sense (see JacobianFunctionMF).
     */
    JFNKEpsilonMode m_jfnk_epsilon_mode = JFNKEpsilonMode::global;

    /**
     * \brief Additive floor of the component probe scale, as a fraction of
     * each field block's reference scale (newton.jfnk_component_floor, all
     * blocks) with per-block overrides newton.jfnk_component_floor_<block>.
     * The floor keeps components at or near zero from demanding a vanishing
     * absolute perturbation (and from dominating the scaled direction norm):
     * below the floor the perturbation is relative to the floor instead.
     */
    amrex::Real m_jfnk_component_floor = 1.0e-3;
    std::vector<std::string> m_component_floor_names;
    std::vector<amrex::Real> m_component_floor_fractions;

    /**
     * \brief Diagnostic file (newton.jfnk_probe_report_file, default none;
     * "none"/"off" disable): one row per Newton iteration with the
     * statistics of the effective relative perturbation
     * eps |v_i| / (|U_i| + floor) of the probe over the components the
     * Krylov direction touches, split into small components (below 1e-2 of
     * their block reference scale) and large. Costs three host copies of
     * the state and three reductions per Newton iteration when enabled;
     * a no-op (bit-identical) when unset.
     */
    std::string m_jfnk_probe_report_file;

    /**
     * \brief Reduced-space (active-set) Newton, newton.active_set = 1
     * (default 0: the plain projected Newton, bit-identical).
     *
     * The operator's admissibility projection makes the Newton update a
     * bound-constrained one: direction components that would leave the
     * admissible box are clamped onto their bounds. The plain method
     * computes the UNCONSTRAINED direction first and clamps afterwards,
     * so every free component's update was computed assuming the clamped
     * cells move fully -- once a population rides its floor under a
     * sustained drain (a wall row, a pedestal band) the clamped direction
     * is inconsistent, the full-norm Armijo test grades rows the
     * projection just froze, and the iteration degrades to damped linear
     * convergence or stagnates (measured on the production formation
     * deck: 100-300 pinned components, 17-57% of the steps within 1e-3,
     * 100-150 GMRES per solve, 10-80 s per step in the affected window).
     *
     * The active-set mode instead solves the complementarity problem the
     * projection implies, min(U_i - l_i, F_i) = 0: at every iterate the
     * ACTIVE SET A is identified (IdentifyActiveSet: components resting
     * on their bound whose residual demands a sub-bound update, F_i > 0,
     * plus the components the previous iteration clamped that still
     * demand it; a component whose residual points inward leaves the
     * set), its members are moved onto their bounds
     * (MoveActiveSetToBounds, with a residual re-evaluation when anything
     * moved) and the Newton system is solved in the FREE subspace only:
     * the matrix-free operator becomes P_F J P_F + P_A (pinned rows and
     * columns replaced by the identity), the preconditioner
     * P_F M^{-1} P_F + P_A, the right-hand side P_F F, so the correction
     * vanishes exactly on A and every free direction component is
     * computed with the pinned cells held where they are. The Armijo
     * search grades the FREE residual norm, convergence is declared on it
     * (newton.active_set_tolerance relative to its iteration-0 value, or
     * the absolute tolerance; exit status 4), the residual left on A --
     * the pinned defect, the sub-bound update the floors refused -- is
     * booked to the operator's floor ledger at every exit
     * (BookPinnedDefect), and newton.txt gains the columns free_norm_rel
     * and exit_status. Components the projection clamps DURING a free
     * solve (new entrants) join the set for that line search and are
     * held from the next iteration on; the set is rebuilt from the state
     * at iteration 0 of every solve. The iteration-0 free-subspace rescue
     * below is not used in this mode (the search already grades the free
     * norm). A fully pinned state whose free residual is at round-off
     * and that nothing moved still counts as a frozen step, so the
     * newton.max_frozen_steps guard keeps its meaning.
     */
    bool m_active_set = false;

    /**
     * \brief Relative tolerance of the free-subspace residual in the
     * active-set mode, measured against the free norm at iteration 0 of
     * the solve. Negative (the default) selects newton.relative_tolerance.
     */
    amrex::Real m_active_set_tolerance = -1.0;

    /**
     * \brief Consecutive zero-progress (frozen) solve counter.
     */
    mutable int m_frozen_steps = 0;

    /**
     * \brief Env-gated residual fingerprint: when the environment variable
     * WARPX_MHD_REPORT_RESIDUAL_BLOCKS is set (and not "0"), print the
     * per-solver-vec-block L2 norms of the residual at iteration 0 of every
     * solve, and the top-|residual| cells of any block whose squared-norm
     * share exceeds m_residual_block_share (environment variable
     * WARPX_MHD_RESIDUAL_BLOCK_SHARE, default 0.5).
     */
    bool m_report_residual_blocks = false;
    amrex::Real m_residual_block_share = 0.5;

    /**
     * \brief Relative tolerance for the Newton solver.
     */
    amrex::Real m_rtol = 1.0e-6;

    /**
     * \brief Absolute tolerance for the Newton solver.
     */
    amrex::Real m_atol = 0.;

    /**
     * \brief Maximum iterations for the Newton solver.
     */
    int m_maxits = 100;
    // Counter of budget-pegged exits driving the hotspot-report rate limit.
    mutable amrex::Long m_budget_pegged_exits = 0;

    /**
     * \brief Total nonlinear iterations for the diagnostic file
     */
    mutable int m_total_iters = 0;

    /**
     * \brief Inexact-Newton adaptive forcing (Eisenstat--Walker-style, the
     * prescription of Chacon & Knoll, JCP 188 (2003) 577): the linear
     * solve tolerance of iteration k is
     *
     *   zeta_A = gamma (||F_k|| / ||F_{k-1}||)^alpha,
     *   zeta_B = min[zeta_max, max(zeta_A, gamma zeta_{k-1}^alpha)],
     *   zeta_k = min[zeta_max, max(zeta_B, gamma eps_t / ||F_k||)],
     *
     * with eps_t the Newton convergence target. The first term gives
     * superlinear convergence, the second avoids volatile decreases, and
     * the third avoids oversolving the last iteration. When disabled
     * (default) the fixed m_linsol_rtol is used every iteration.
     */
    bool m_adaptive_forcing = false;
    amrex::Real m_forcing_alpha = 1.5;
    amrex::Real m_forcing_gamma = 0.9;
    amrex::Real m_forcing_max = 0.5;

    /**
     * \brief Relative tolerance for linear solver.
     */
    amrex::Real m_linsol_rtol = 1.0e-4;

    /**
     * \brief Absolute tolerance for linear solver.
     */
    amrex::Real m_linsol_atol = 0.;

    /**
     * \brief Maximum iterations for linear solver.
     */
    int m_linsol_maxits = 1000;

    /**
     * \brief Total linear iterations for the diagnostic file
     */
    mutable int m_total_linsol_iters = 0;

    /**
     * \brief Verbosity flag for linear solver.
     */
    int m_linsol_verbose_int = 2;

    /**
     * \brief Restart iteration for linear solver.
     */
    int m_linsol_restart_length = 30;

    /**
     * \brief Preconditioner type
     */
    PreconditionerType m_pc_type = PreconditionerType::none;

    mutable amrex::Real m_cur_time, m_dt;

    /**
     * \brief The linear function used by linear solver to compute A*v.
     * In the contect of JFNK, A = dF/dU (i.e., system Jacobian)
     */
    std::unique_ptr<JacobianFunctionMF<Vec,Ops>> m_linear_function;

    /**
     * \brief Choice of linear solver
     */
    LinearSolverType m_linear_solver_type = LinearSolverType::amrex_gmres;

    /**
     * \brief The linear solver object.
     */
    std::unique_ptr<LinearSolver<Vec,JacobianFunctionMF<Vec,Ops>>> m_linear_solver;

    void ParseParameters ();

    /**
     * \brief Compute the nonlinear residual: F(U) = U - b - R(U).
     * a_is_iterate distinguishes the evaluation at an accepted Newton
     * iterate (the F(U_k) driving the convergence test) from a
     * line-search trial evaluation; the operator is notified through
     * SetResidualIsNewtonIterate() before ComputeRHS() runs.
     */
    void EvalResidual ( Vec&         a_F,
                  const Vec&         a_U,
                  const Vec&         a_b,
                        amrex::Real  a_time,
                        int          a_iter,
                        bool         a_is_iterate ) const;

};

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::Define ( const Vec&  a_U,
                                     Ops*        a_ops )
{
    BL_PROFILE("NewtonSolver::Define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !this->m_is_defined,
        "Newton nonlinear solver object is already defined!");

    ParseParameters();

    m_dU.Define(a_U);
    m_F.Define(a_U); // residual function F(U) = U - b - R(U) = 0
    m_R.Define(a_U); // right hand side function R(U)
    m_trial_U.Define(a_U);
    m_trial_F.Define(a_U);

    m_ops = a_ops;

    m_linear_function = std::make_unique<JacobianFunctionMF<Vec,Ops>>();
    m_linear_function->define(m_F, m_ops, m_pc_type);
    m_linear_function->setJFNKEps(m_jfnk_epsilon);
    {
        // Component probe floors: fraction x block reference scale per
        // field block of the state vector (names/scales from the vector).
        const amrex::ParmParse pp_newton("newton");
        m_component_floor_names = a_U.blockNames();
        const std::vector<amrex::Real> block_scales = a_U.blockScales();
        std::vector<amrex::Real> floors;
        m_component_floor_fractions.clear();
        for (std::size_t ib = 0; ib < m_component_floor_names.size(); ++ib) {
            amrex::Real fraction = m_jfnk_component_floor;
            utils::parser::queryWithParser(
                pp_newton, ("jfnk_component_floor_" + m_component_floor_names[ib]).c_str(), fraction);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                fraction > 0.0,
                "newton.jfnk_component_floor[_<block>] must be positive");
            m_component_floor_fractions.push_back(fraction);
            floors.push_back(fraction * block_scales[ib]);
        }
        m_linear_function->setJFNKEpsMode(m_jfnk_epsilon_mode, floors);
        m_linear_function->setProbeReportFile(m_jfnk_probe_report_file);
    }
    this->m_usePC = m_linear_function->usePreconditioner();
    if (m_active_set) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ops->HasAdmissibilityProjection() && m_ops->UseNewtonLineSearch(),
            "newton.active_set = 1 requires an operator with an admissibility "
            "projection and a Newton line search (the theta-implicit MHD solver)");
        m_ops->SetActiveSetMode(true);
        m_linear_function->setActiveSetMasking(true);
        m_F_free.Define(a_U);
        m_F_split.Define(a_U);
    }

    if (m_linear_solver_type == LinearSolverType::amrex_gmres) {
        m_linear_solver = std::make_unique<AMReXGMRES<Vec,JacobianFunctionMF<Vec,Ops>>>();
    } else if (m_linear_solver_type == LinearSolverType::petsc_ksp) {
#ifdef AMREX_USE_PETSC
        m_linear_solver = std::make_unique<PETScKSP<Vec,JacobianFunctionMF<Vec,Ops>>>();
#else
        WARPX_ABORT_WITH_MESSAGE("NewtonSolver::Define(): must compile with PETSc to use petsc_ksp (AMREX_USE_PETSC must be defined)");
#endif
    } else {
        WARPX_ABORT_WITH_MESSAGE("NewtonSolver::Define(): invalid linear solver type");
    }
    m_linear_solver->define(*m_linear_function);
    m_linear_solver->setVerbose( m_linsol_verbose_int );
    m_linear_solver->setRestartLength( m_linsol_restart_length );
    m_linear_solver->setMaxIters( m_linsol_maxits );

    this->m_is_defined = true;

    // Create diagnostic file and write header
    if (amrex::ParallelDescriptor::IOProcessor()
        && !this->m_diagnostic_file.empty()
        && !amrex::FileExists(this->m_diagnostic_file)) {

        std::filesystem::path const diagnostic_path(this->m_diagnostic_file);
        std::filesystem::path const diagnostic_dir = diagnostic_path.parent_path();
        if (!diagnostic_dir.empty()) {
            std::filesystem::create_directories(diagnostic_dir);
        }

        std::ofstream diagnostic_file{this->m_diagnostic_file, std::ofstream::out | std::ofstream::trunc};
        int c = 0;
        diagnostic_file << "#";
        diagnostic_file << "[" << c++ << "]step()";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]time(s)";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]total_iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]norm_abs";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]norm_rel";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]gmres_iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]gmres_total_iters";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]gmres_last_res";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]free_norm_abs";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]pinned_defect";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]num_pinned";
        if (m_active_set) {
            diagnostic_file << " ";
            diagnostic_file << "[" << c++ << "]free_norm_rel";
            diagnostic_file << " ";
            diagnostic_file << "[" << c++ << "]exit_status";
            diagnostic_file << " ";
            diagnostic_file << "[" << c++ << "]max_pinned_direction";
            diagnostic_file << " ";
            diagnostic_file << "[" << c++ << "]max_bound_excess";
        }
        diagnostic_file << "\n";
        diagnostic_file.close();
    }
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::ParseParameters ()
{
    const amrex::ParmParse pp_newton("newton");
    pp_newton.query("verbose",             this->m_verbose);
    pp_newton.query("linear_solver",       m_linear_solver_type);
    pp_newton.query("absolute_tolerance",  m_atol);
    pp_newton.query("relative_tolerance",  m_rtol);
    pp_newton.query("max_iterations",      m_maxits);
    pp_newton.query("require_convergence", m_require_convergence);
    pp_newton.query("max_frozen_steps",    m_max_frozen_steps);
    utils::parser::queryWithParser(pp_newton, "jfnk_epsilon", m_jfnk_epsilon);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_jfnk_epsilon > 0.0,
        "newton.jfnk_epsilon must be positive");
    pp_newton.query("jfnk_epsilon_mode", m_jfnk_epsilon_mode);
    utils::parser::queryWithParser(pp_newton, "jfnk_component_floor", m_jfnk_component_floor);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_jfnk_component_floor > 0.0,
        "newton.jfnk_component_floor must be positive");
    pp_newton.query("jfnk_probe_report_file", m_jfnk_probe_report_file);
    // "none" (or "off") switches the diagnostic off, so a deck that sets a
    // file can be overridden from the command line.
    if (m_jfnk_probe_report_file == "none" || m_jfnk_probe_report_file == "off") {
        m_jfnk_probe_report_file.clear();
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_max_frozen_steps > 0,
        "newton.max_frozen_steps must be positive");
    pp_newton.query("active_set", m_active_set);
    utils::parser::queryWithParser(pp_newton, "active_set_tolerance",
                                   m_active_set_tolerance);
    if (m_active_set_tolerance < 0.0) { m_active_set_tolerance = m_rtol; }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_active_set || m_active_set_tolerance < 1.0,
        "newton.active_set_tolerance must be below 1");
    pp_newton.query("diagnostic_file",     this->m_diagnostic_file);
    pp_newton.query("diagnostic_interval", this->m_diagnostic_interval);
    pp_newton.query("adaptive_forcing",    m_adaptive_forcing);
    utils::parser::queryWithParser(pp_newton, "forcing_alpha", m_forcing_alpha);
    utils::parser::queryWithParser(pp_newton, "forcing_gamma", m_forcing_gamma);
    utils::parser::queryWithParser(pp_newton, "forcing_max",   m_forcing_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_forcing_alpha > 1.0 && m_forcing_gamma > 0.0 &&
            m_forcing_gamma <= 1.0 && m_forcing_max > 0.0 &&
            m_forcing_max < 1.0,
        "newton.forcing_alpha must exceed 1, newton.forcing_gamma must be "
        "in (0, 1], and newton.forcing_max must be in (0, 1)");

    const amrex::ParmParse pp_l(amrex::getEnumNameString(m_linear_solver_type));
    pp_l.query("verbose_int",         m_linsol_verbose_int);
    pp_l.query("restart_length",      m_linsol_restart_length);
    pp_l.query("absolute_tolerance",  m_linsol_atol);
    pp_l.query("relative_tolerance",  m_linsol_rtol);
    pp_l.query("max_iterations",      m_linsol_maxits);

    /* backward compatibility */
    const amrex::ParmParse pp_gmres("gmres");
    pp_gmres.query("verbose_int",         m_linsol_verbose_int);
    pp_gmres.query("restart_length",      m_linsol_restart_length);
    pp_gmres.query("absolute_tolerance",  m_linsol_atol);
    pp_gmres.query("relative_tolerance",  m_linsol_rtol);
    pp_gmres.query("max_iterations",      m_linsol_maxits);

    const amrex::ParmParse pp_jac("jacobian");
    pp_jac.query("pc_type", m_pc_type);

    if (const char* const report_env =
            std::getenv("WARPX_MHD_REPORT_RESIDUAL_BLOCKS")) {
        m_report_residual_blocks = (std::string(report_env) != "0");
    }
    if (const char* const share_env =
            std::getenv("WARPX_MHD_RESIDUAL_BLOCK_SHARE")) {
        m_residual_block_share =
            static_cast<amrex::Real>(std::atof(share_env));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_residual_block_share > 0.0 && m_residual_block_share < 1.0,
            "WARPX_MHD_RESIDUAL_BLOCK_SHARE must be in (0, 1)");
    }
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::Solve (Vec&         a_U,
                             const Vec&         a_b,
                                   amrex::Real  a_time,
                                   amrex::Real  a_dt,
                                   int          a_step) const
{
    BL_PROFILE("NewtonSolver::Solve()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        this->m_is_defined,
        "NewtonSolver::Solve() called on undefined object");
    using namespace amrex::literals;

    //
    // Newton routine to solve nonlinear equation of form:
    // F(U) = U - b - R(U) = 0
    //

    CurTime(a_time);
    CurTimeStep(a_dt);

    amrex::Real norm_abs = 0.;
    amrex::Real norm0 = 1._rt;
    amrex::Real norm_rel = 0.;
    amrex::Real norm_prev = 0.;
    amrex::Real forcing_prev = m_forcing_max;

    // Reduced-space (active-set) mode bookkeeping (see m_active_set);
    // inert at the default.
    amrex::Real free_norm = 0.;
    amrex::Real active_free_norm0 = 1._rt;
    amrex::Real free_rel = 0.;
    amrex::Real active_pinned_defect = 0.;
    amrex::Long active_num_pinned = 0;
    amrex::Long active_set_moves = 0;
    bool active_set_identified = false;
    // Self-checks of the mode, recorded per solve in newton.txt: the
    // largest norm of the Newton direction restricted to the active set
    // (exactly 0 by construction: masked RHS, masked operator with the
    // identity on the pinned rows, masked preconditioner) and the largest
    // distance of a pinned component above its landing point after the
    // move, in margins (<= 0.5 by construction).
    amrex::Real active_max_leak = 0.;
    amrex::Real active_max_excess = 0.;

    int iter;
    int linear_solver_iters = 0;
    for (iter = 0; iter < m_maxits;) {

        // Compute residual: F(U) = U - b - R(U)
        EvalResidual(m_F, a_U, a_b, a_time, iter, true);

        if (m_active_set) {
            // Reduced-space mode: identify the active set at this iterate
            // (bound-resident components whose residual demands a
            // sub-bound update; after iteration 0 also the components the
            // previous projection clamped, while they still demand it),
            // move its members onto their bounds, and re-evaluate the
            // residual there so the free solve below is consistent with
            // the pinned components held at the bound.
            amrex::Long entered = 0;
            amrex::Long released = 0;
            active_num_pinned = m_ops->IdentifyActiveSet(
                a_U, m_F, iter > 0, entered, released);
            active_set_identified = true;
            amrex::Long moved = 0;
            if (active_num_pinned > 0) {
                moved = m_ops->MoveActiveSetToBounds(a_U);
                if (moved > 0) {
                    active_set_moves += moved;
                    EvalResidual(m_F, a_U, a_b, a_time, iter, true);
                }
                const amrex::Real excess = m_ops->ActiveSetBoundExcess(a_U);
                active_max_excess = std::max(active_max_excess, excess);
                if (this->m_verbose && excess > 0.5_rt) {
                    amrex::Print() << "Newton: WARNING active-set member left "
                                   << excess << " margins above its bound after "
                                      "the move\n";
                }
            }
            if (this->m_verbose && (active_num_pinned > 0 || released > 0)) {
                amrex::Print() << "Newton: active set at iteration " << iter
                               << ": " << active_num_pinned << " pinned ("
                               << entered << " entered, " << released
                               << " released, " << moved
                               << " moved onto bounds)\n";
            }
        }

        // Compute norm of the residual
        norm_abs = m_F.norm2();
        if (iter == 0) {
            if (norm_abs > 0.) { norm0 = norm_abs; }
            else { norm0 = 1._rt; }
        }
        norm_rel = norm_abs/norm0;

        if (m_active_set) {
            free_norm = ActiveSetResidualSplit(m_F, active_pinned_defect,
                                               active_num_pinned);
            if (iter == 0) {
                active_free_norm0 = (free_norm > 0.) ? free_norm : 1._rt;
            }
            free_rel = free_norm / active_free_norm0;
        }

        // Check for convergence criteria
        if (this->m_verbose || iter == m_maxits) {
            amrex::Print() << "Newton: iteration = " << std::setw(3) << iter <<  ", norm = "
                           << std::scientific << std::setprecision(5) << norm_abs << " (abs.), "
                           << std::scientific << std::setprecision(5) << norm_rel << " (rel.)" << "\n";
        }
        if (m_active_set && active_num_pinned > 0 &&
            (this->m_verbose || iter == m_maxits)) {
            amrex::Print() << "Newton: free-subspace norm = "
                           << std::scientific << std::setprecision(5) << free_norm
                           << " (abs.), " << free_rel << " (rel.); pinned defect = "
                           << active_pinned_defect << " over " << active_num_pinned
                           << " components\n";
        }

        // Env-gated residual fingerprint (WARPX_MHD_REPORT_RESIDUAL_BLOCKS):
        // the per-block scaled L2 norms of the iteration-0 residual, plus
        // the top-|residual| cells of any block holding more than the
        // WARPX_MHD_RESIDUAL_BLOCK_SHARE fraction (default 0.5) of the
        // squared norm. Host reductions; off by default.
        if (iter == 0 && m_report_residual_blocks) {
            m_F.ReportBlockNorms(m_residual_block_share, 5);
        }

        if (norm_abs < m_atol) {
            this->m_exit_status = 2;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied absolute tolerance " << m_atol << "\n";
            }
            break;
        }

        if (norm_rel < m_rtol) {
            this->m_exit_status = 3;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied relative tolerance " << m_rtol << "\n";
            }
            break;
        }

        if (m_active_set && active_num_pinned > 0) {
            // Free-subspace convergence (the active-set exit): the residual
            // over the free components is within the active-set tolerance
            // of its iteration-0 value, or within the absolute tolerance,
            // or at round-off relative to the full norm (no free dynamics;
            // 1e-12, the rescue path's noise gate). The pinned defect is
            // not part of the test -- it is the sub-bound update the
            // floors refused and is booked to the ledger at the exit.
            const amrex::Real free_target = std::max(
                {m_atol, m_active_set_tolerance * active_free_norm0,
                 1.0e-12_rt * norm_abs});
            if (free_norm <= free_target) {
                this->m_exit_status = 4;
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied active-set tolerance " << m_active_set_tolerance
                               << " on the free subspace (free norm " << free_norm
                               << " abs., " << free_rel << " rel.; pinned defect "
                               << active_pinned_defect << " over " << active_num_pinned
                               << " components; full rel. norm " << norm_rel << ")\n";
                // Genuine free convergence (absolute or relative
                // tolerance) is a converged solve. An exit granted by the
                // round-off gate alone at iteration 0, with nothing moved,
                // returns the state it was handed: keep the freeze guard's
                // accounting so a fully pinned domain with no free
                // dynamics still aborts instead of silently advancing.
                const bool free_converged =
                    (free_norm <= m_atol) ||
                    (free_norm <= m_active_set_tolerance * active_free_norm0);
                if (iter == 0 && active_set_moves == 0 && !free_converged) {
                    m_frozen_steps++;
                    amrex::Print()
                        << "WARNING: Newton active-set exit at iter = 0 with "
                        << "the free residual at round-off and nothing moved; "
                        << "the state is UNCHANGED by this solve (frozen step "
                        << m_frozen_steps << " of at most " << m_max_frozen_steps
                        << " consecutive; pinned defect = " << active_pinned_defect
                        << ", " << active_num_pinned << " pinned)\n";
                    if (m_frozen_steps >= m_max_frozen_steps) {
                        std::stringstream freeze_msg;
                        freeze_msg
                            << "Newton made zero progress for "
                            << "newton.max_frozen_steps consecutive solves "
                            << "(active-set mode: free residual at round-off, "
                            << active_num_pinned << " pinned components";
                        const std::string pinned_report =
                            m_ops->PinnedComponentReport();
                        if (!pinned_report.empty()) {
                            freeze_msg << ": " << pinned_report;
                        }
                        freeze_msg
                            << "): the simulation state is frozen and, "
                               "absent time-dependent drives, cannot "
                               "recover. Aborting instead of silently "
                               "advancing time.";
                        WARPX_ABORT_WITH_MESSAGE(freeze_msg.str());
                    }
                } else {
                    m_frozen_steps = 0;
                }
                break;
            }
        }

        if (norm_abs > 100._rt*norm0) {
            this->m_exit_status = -9;
            amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                           << ". SOLVER DIVERGED! relative tolerance = " << m_rtol << "\n";
            std::stringstream convergenceMsg;
            convergenceMsg << "Newton: exiting at iteration " << std::setw(3) << iter <<
                              ". SOLVER DIVERGED! absolute norm = " << norm_abs <<
                              " has increased by 100X from that after first iteration.";
            WARPX_ABORT_WITH_MESSAGE(convergenceMsg.str());
        }

        // Prepare for the linear solve
        m_ops->PreLinearSolve();
        m_linear_function->setBaseSolution(a_U);
        m_linear_function->setBaseRHS(m_R);
        m_linear_function->updatePreCondMat(a_U);

        // Inexact-Newton adaptive forcing (see the member documentation):
        // loose linear tolerances far from the solution, superlinearly
        // tightening near it, floored so the last iteration is not
        // oversolved.
        amrex::Real linsol_rtol = m_linsol_rtol;
        if (m_adaptive_forcing) {
            // In the active-set mode the linear system is the free one:
            // the forcing history tracks the free norm and its target.
            const amrex::Real forcing_norm = m_active_set ? free_norm : norm_abs;
            const amrex::Real convergence_target =
                m_active_set ? std::max(m_atol, m_active_set_tolerance * active_free_norm0)
                             : std::max(m_atol, m_rtol * norm0);
            amrex::Real zeta = m_forcing_max;
            if (iter > 0 && norm_prev > 0._rt) {
                const amrex::Real zeta_a =
                    m_forcing_gamma *
                    std::pow(forcing_norm / norm_prev, m_forcing_alpha);
                const amrex::Real safeguard =
                    m_forcing_gamma * std::pow(forcing_prev, m_forcing_alpha);
                zeta = std::min(m_forcing_max, std::max(zeta_a, safeguard));
            }
            if (forcing_norm > 0._rt) {
                zeta = std::min(
                    m_forcing_max,
                    std::max(zeta, m_forcing_gamma * convergence_target /
                                       forcing_norm));
            }
            forcing_prev = zeta;
            norm_prev = forcing_norm;
            linsol_rtol = zeta;
            if (this->m_verbose) {
                amrex::Print() << "Newton: forcing term = " << zeta << "\n";
            }
        }

        // Solve linear system for Newton step [Jac]*dU = F
        m_dU.zero();
        if (m_active_set && active_num_pinned > 0) {
            // Reduced system: the right-hand side restricted to the free
            // subspace; the operator and the preconditioner are masked by
            // the linear function (identity on the active set), so the
            // correction is exactly zero on every pinned component.
            m_F_free.Copy(m_F);
            m_ops->ZeroActiveSetComponents(m_F_free);
            m_linear_solver->solve( m_dU, m_F_free, linsol_rtol, m_linsol_atol );
            // Self-check: the direction restricted to the active set must
            // vanish identically (every Krylov vector is zero on A).
            m_F_split.zero();
            m_ops->CopyActiveSetComponents(m_F_split, m_dU);
            const amrex::Real leak = m_F_split.norm2();
            active_max_leak = std::max(active_max_leak, leak);
            if (this->m_verbose) {
                amrex::Print() << "Newton: active-set direction leak = " << leak
                               << "\n";
            }
        } else {
            m_linear_solver->solve( m_dU, m_F, linsol_rtol, m_linsol_atol );
        }
        linear_solver_iters += m_linear_solver->getNumIters();

        // Update solution
        // Update solution, allowing the operator to enforce its admissible
        // state domain (for example, positive fluid density and energy).
        // The projecting variant may also zero descent components at
        // cells resting on their bound (active-set treatment), so a
        // single floor-resident cell cannot deadlock the line search.
        const amrex::Real solver_step =
            m_ops->ProjectAndLimitSolverStep(a_U, m_dU, -1.0_rt);
        if (m_ops->UseNewtonLineSearch()) {
            constexpr int max_backtracks = 12;
            constexpr amrex::Real reduction = 0.5_rt;
            constexpr amrex::Real armijo = 1.0e-4_rt;
            amrex::Real step_length = -solver_step;
            if (this->m_verbose && step_length != 1.0_rt) {
                amrex::Print() << "Newton: admissibility-limited trial step = "
                               << step_length << "\n";
            }
            // Armijo reference: the full residual norm; in the active-set
            // mode the FREE norm with respect to the set the projection
            // may just have enlarged (new entrants are graded as pinned).
            amrex::Real reference_norm = norm_abs;
            if (m_active_set) {
                reference_norm = ActiveSetResidualSplit(
                    m_F, active_pinned_defect, active_num_pinned);
                free_norm = reference_norm;
            }
            bool accepted = false;
            for (int backtrack = 0; backtrack <= max_backtracks; ++backtrack) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    step_length > 0.0_rt,
                    "Newton line search found no positive admissible step");
                m_trial_U.Copy(a_U);
                m_trial_U.increment(m_dU, -step_length);

                EvalResidual(m_trial_F, m_trial_U, a_b, a_time, iter, false);
                amrex::Real trial_norm;
                if (m_active_set) {
                    amrex::Real trial_defect = 0.0_rt;
                    amrex::Long trial_pinned = 0;
                    trial_norm = ActiveSetResidualSplit(m_trial_F, trial_defect,
                                                        trial_pinned);
                } else {
                    trial_norm = m_trial_F.norm2();
                }
                if (trial_norm <= (1.0_rt - armijo * step_length) * reference_norm) {
                    a_U.Copy(m_trial_U);
                    accepted = true;
                    if (this->m_verbose &&
                        (step_length != 1.0_rt || backtrack != 0)) {
                        amrex::Print()
                            << "Newton: accepted damped step = " << step_length
                            << " after " << backtrack << " backtracks\n";
                    }
                    break;
                }
                step_length *= reduction;
            }
            // Full-norm backtracking exhausted. With require_convergence
            // the solve aborts here -- BEFORE any rescue is attempted, so
            // the strict policy is unchanged.
            if (!accepted) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !m_require_convergence,
                    "Newton line search failed to find a residual-decreasing "
                    "admissible step (newton.require_convergence = true)");
            }

            // Active-set-aware line-search rescue, iteration 0 ONLY. The
            // admissibility projection can clamp direction components at
            // cells resting on their bounds (a persistent active set at
            // floor-resident cells with a sustained drain). The full-norm
            // Armijo test above then grades every trial on residual rows
            // whose VARIABLES the projection just froze: the free
            // subspace cannot reduce what it is not allowed to touch,
            // and once the pinned rows dominate the norm the search
            // stagnates at iteration 0 forever even though every free
            // cell still has a productive update. Rescue: retry the same
            // backtracking ladder against the FREE-subspace norm (the
            // residual excluding the clamped components). A pass on that
            // ladder is genuine progress on every row the solver may
            // still act on, so accept it -- the state changed, no frozen
            // step. The full-norm test remains the primary criterion:
            // this pass runs only after it exhausted all backtracks, and
            // only when the projection clamped something (with an empty
            // active set the free norm IS the full norm and the ladder
            // would fail identically, so it is skipped). Later-iteration
            // stagnations keep the unchanged tolerant policy below --
            // the iterate already carries this solve's accepted progress,
            // there is no freeze to prevent, and extending the rescue
            // there was measured to overdrive bound-riding drains (the
            // temperature-floor CI trajectory).
            amrex::Real pinned_defect = 0.0_rt;
            amrex::Long num_pinned = 0;
            if (!accepted && iter == 0 && !m_active_set) {
                const amrex::Real free_norm0 =
                    m_ops->FreeResidualNorm(m_F, pinned_defect, num_pinned);
                // Noise gate: when the free residual sits at round-off
                // relative to the full norm, "reducing" it is meaningless
                // and would defeat the freeze guard -- a fully frozen
                // state (everything pinned, no free dynamics) must still
                // abort rather than silently advance time on noise-level
                // acceptances (measured: an all-pedestal uniform variant
                // of the CI deck rode 106 such acceptances to completion
                // with its 0.03 defect never resolving). 1e-12 sits four
                // orders above double round-off and eight below any
                // physically meaningful free/pinned split observed (CI
                // deck ~1.4e-4, production corpse ~4e-3).
                const amrex::Real free_noise_floor = 1.0e-12_rt * norm_abs;
                if (num_pinned > 0 && free_norm0 > free_noise_floor) {
                    amrex::Real rescue_step = -solver_step;
                    for (int backtrack = 0; backtrack <= max_backtracks;
                         ++backtrack) {
                        m_trial_U.Copy(a_U);
                        m_trial_U.increment(m_dU, -rescue_step);
                        EvalResidual(m_trial_F, m_trial_U, a_b, a_time, iter,
                                     false);
                        amrex::Real trial_defect = 0.0_rt;
                        amrex::Long trial_pinned = 0;
                        const amrex::Real free_trial =
                            m_ops->FreeResidualNorm(m_trial_F, trial_defect,
                                                    trial_pinned);
                        if (free_trial <=
                            (1.0_rt - armijo * rescue_step) * free_norm0) {
                            a_U.Copy(m_trial_U);
                            accepted = true;
                            amrex::Print()
                                << "Newton: accepted free-subspace step = "
                                << rescue_step << " (pinned defect = "
                                << trial_defect << ", " << num_pinned
                                << " pinned)\n";
                            break;
                        }
                        rescue_step *= reduction;
                    }
                }
            }

            if (!accepted) {
                // The state is unchanged only when no update was ever
                // accepted AND (active-set mode) nothing was moved onto
                // its bound by this solve.
                const bool state_unchanged =
                    (iter == 0) && !(m_active_set && active_set_moves > 0);
                // Not even the free-subspace norm could be decreased.
                // This is treated like the max-iteration exit (the same
                // policy that already tolerates a non-converged step):
                // keep the current iterate -- which typically carries
                // most of the step's residual reduction -- warn, and
                // hand the state back to the time loop. Stagnation at
                // iteration 0 is different in kind: no update was ever
                // accepted, so the solve returns the state it was handed
                // and the time step is a no-op -- for a system without
                // time-dependent drives every later solve then repeats
                // bit-identically and the simulation silently freezes.
                // Count consecutive frozen solves and abort past
                // newton.max_frozen_steps.
                if (state_unchanged) {
                    m_frozen_steps++;
                    amrex::Print()
                        << "WARNING: Newton line search stagnated at iter = 0 "
                        << "with no residual reduction; the state is UNCHANGED "
                        << "by this solve (frozen step " << m_frozen_steps
                        << " of at most " << m_max_frozen_steps
                        << " consecutive; pinned defect = " << pinned_defect
                        << ", " << num_pinned << " pinned)\n";
                    if (m_frozen_steps >= m_max_frozen_steps) {
                        std::stringstream freeze_msg;
                        freeze_msg
                            << "Newton made zero progress for "
                            << "newton.max_frozen_steps consecutive solves "
                            << "(pinned defect = " << pinned_defect << ", "
                            << num_pinned << " pinned components";
                        const std::string pinned_report =
                            m_ops->PinnedComponentReport();
                        if (!pinned_report.empty()) {
                            freeze_msg << ": " << pinned_report;
                        }
                        freeze_msg
                            << "): the simulation state is frozen and, "
                               "absent time-dependent drives, cannot "
                               "recover. Aborting instead of silently "
                               "advancing time.";
                        WARPX_ABORT_WITH_MESSAGE(freeze_msg.str());
                    }
                } else {
                    // Grade the accepted fiction: how much of the residual
                    // lives on components the projection clamped (the
                    // pinned defect, which the free subspace cannot
                    // reduce) versus on free components (genuinely
                    // unconverged dynamics).
                    amrex::Real stagnation_defect = 0.0_rt;
                    amrex::Long stagnation_pinned = 0;
                    const amrex::Real stagnation_free_norm =
                        m_ops->FreeResidualNorm(m_F, stagnation_defect,
                                                stagnation_pinned);
                    amrex::Print()
                        << "WARNING: Newton line search stagnated at iter = "
                        << iter << " (rel. norm " << norm_rel
                        << "; free-subspace norm " << stagnation_free_norm
                        << ", pinned defect " << stagnation_defect
                        << " over " << stagnation_pinned
                        << " clamped components"
                        << "); accepting the current iterate (projected "
                           "onto the admissible set) and continuing\n";
                    // Stagnation-accept insurance: the accepted iterate
                    // is non-converged fiction; project it onto the
                    // operator's admissible set (floors, NaN scrub) so
                    // it cannot launder inadmissible values into the
                    // time loop (measured: a NEGATIVE 6e17 J wall-ledger
                    // booking rode this path on the formation ladder).
                    m_ops->ProjectStateToAdmissibleSet(a_U);
                }
                if (m_active_set) {
                    // 1 = stagnation exit: an unconverged iterate returned.
                    this->m_exit_status = 1;
                }
                break;
            }
            m_frozen_steps = 0;
        } else {
            a_U.increment(m_dU, solver_step);
            if (this->m_verbose && solver_step != -1.0_rt) {
                amrex::Print()
                    << "Newton: admissibility-limited step = " << -solver_step
                    << "\n";
            }
        }

        iter++;
        if (iter >= m_maxits) {
            if (m_active_set && m_ops->UseNewtonLineSearch()) {
                // The accepted trial's residual describes the returned
                // state: report and book from it rather than from the
                // previous iterate's residual.
                m_F.Copy(m_trial_F);
                norm_abs = m_F.norm2();
                norm_rel = norm_abs / norm0;
            }
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iter = " << std::setw(3) << iter
                               << ". Maximum iteration reached: iter = " << m_maxits << "\n";
            }
            // Budget-pegged exit: localize the residual (first exit and
            // every 25th after -- the report is a bounded device scan).
            if (m_budget_pegged_exits++ % 25 == 0) {
                m_ops->ResidualHotspotReport(m_F);
            }
            break;
        }

    }

    // Update total iteration count
    m_total_iters += iter;
    m_total_linsol_iters += linear_solver_iters;

    // Free-subspace split of the exit residual: the norm over the
    // components the last admissibility projection did NOT clamp, the
    // pinned defect (norm over the clamped components) and their count,
    // with free^2 + pinned^2 = norm_abs^2. A non-converged exit whose
    // residual is all pinned defect is a bound-resident population the
    // solve cannot act on (book-keeping for the floor ledger); one whose
    // free norm is above tolerance is genuinely unconverged dynamics. The
    // masks belong to this solve's last Newton direction; a solve that
    // exits at iteration 0 computed none, so it is reported as all-free.
    amrex::Real exit_free_norm = norm_abs;
    amrex::Real exit_pinned_defect = 0.0_rt;
    amrex::Long exit_num_pinned = 0;
    if (m_active_set) {
        if (active_set_identified) {
            exit_free_norm = ActiveSetResidualSplit(m_F, exit_pinned_defect,
                                                    exit_num_pinned);
        }
    } else if (iter > 0) {
        exit_free_norm =
            m_ops->FreeResidualNorm(m_F, exit_pinned_defect, exit_num_pinned);
    }
    if (m_active_set && exit_num_pinned > 0) {
        // Book the pinned defect -- the sub-bound update the floors
        // refused on the returned state -- to the operator's floor
        // ledger, and list the active set (rate-limited).
        m_ops->BookPinnedDefect(m_F);
        m_ops->ReportActiveSet();
    }

    if (m_rtol > 0. && iter == m_maxits) {
        std::stringstream convergenceMsg;
        convergenceMsg << "Newton solver failed to converge after " << iter <<
                          " iterations. Relative norm is " << norm_rel <<
                          " and the relative tolerance is " << m_rtol <<
                          ". Absolute norm is " << norm_abs <<
                          " and the absolute tolerance is " << m_atol <<
                          ". Free-subspace norm is " << exit_free_norm <<
                          ", pinned defect " << exit_pinned_defect <<
                          " over " << exit_num_pinned <<
                          " clamped components";
        if (this->m_verbose) { amrex::Print() << convergenceMsg.str() << "\n"; }
        ablastr::warn_manager::WMRecordWarning("NewtonSolver", convergenceMsg.str());

        if (m_require_convergence) { this->m_exit_status = -5; }
        else { this->m_exit_status = 0; }
    }

    if (!this->m_diagnostic_file.empty() && amrex::ParallelDescriptor::IOProcessor() &&
        ((a_step+1)%this->m_diagnostic_interval==0 || a_step==0)) {
        std::ofstream diagnostic_file{this->m_diagnostic_file, std::ofstream::out | std::ofstream::app};
        diagnostic_file << std::setprecision(14);
        diagnostic_file << a_step+1;
        diagnostic_file << " ";
        diagnostic_file << a_time + a_dt;
        diagnostic_file << " ";
        diagnostic_file << iter;
        diagnostic_file << " ";
        diagnostic_file << m_total_iters;
        diagnostic_file << " ";
        diagnostic_file << norm_abs;
        diagnostic_file << " ";
        diagnostic_file << norm_rel;
        diagnostic_file << " ";
        diagnostic_file << linear_solver_iters;
        diagnostic_file << " ";
        diagnostic_file << m_total_linsol_iters;
        diagnostic_file << " ";
        diagnostic_file << m_linear_solver->getResidualNorm();
        diagnostic_file << " ";
        diagnostic_file << exit_free_norm;
        diagnostic_file << " ";
        diagnostic_file << exit_pinned_defect;
        diagnostic_file << " ";
        diagnostic_file << exit_num_pinned;
        if (m_active_set) {
            diagnostic_file << " ";
            diagnostic_file << exit_free_norm / active_free_norm0;
            diagnostic_file << " ";
            diagnostic_file << this->m_exit_status;
            diagnostic_file << " ";
            diagnostic_file << active_max_leak;
            diagnostic_file << " ";
            diagnostic_file << active_max_excess;
        }
        diagnostic_file << "\n";
        diagnostic_file.close();
    }

}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::EvalResidual ( Vec&         a_F,
                                     const Vec&         a_U,
                                     const Vec&         a_b,
                                           amrex::Real  a_time,
                                           int          a_iter,
                                           bool         a_is_iterate ) const
{
    BL_PROFILE("NewtonSolver::EvalResidual()");
    m_ops->SetResidualIsNewtonIterate(a_is_iterate);
    m_ops->ComputeRHS( m_R, a_U, a_time, a_iter, false );

    // Compute residual: F(U) = U - b - R(U)
    a_F.Copy(a_U);
    a_F -= m_R;
    a_F -= a_b;

}

#endif
