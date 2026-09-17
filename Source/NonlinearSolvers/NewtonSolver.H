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
#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_Array4.H>
#include <AMReX_Config.H>
#include <AMReX_ParmParse.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <istream>
#include <filesystem>
#include <sstream>
#include <string>

namespace newton_solver_detail
{
    /** Named device functor for the D1 direction fill.  NVCC cannot lower an
     * extended device lambda whose enclosing member is private, while this
     * namespace-scope functor keeps the diagnostic orchestration private. */
    struct D1RZHashDirection
    {
        amrex::Array4<amrex::Real> values;
        int zlo;
        int zn;
        int component;
        bool zperiodic;

        AMREX_GPU_HOST_DEVICE
        void operator() (int i, int j, int k) const noexcept
        {
            int jw = j;
            if (zperiodic) {
                jw = zlo + (((j-zlo) % zn) + zn) % zn;
            }
            unsigned int h = (static_cast<unsigned int>(i) * 73856093u)
                           ^ (static_cast<unsigned int>(jw) * 19349663u)
                           ^ (static_cast<unsigned int>(component) * 83492791u);
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            values(i,j,k) = amrex::Real(h % 20011u)
                          / amrex::Real(20011.0) - amrex::Real(0.5);
        }
    };
}

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

    void SetConvergenceReferenceNorm (amrex::Real norm) override
    {
        m_convergence_reference_norm = norm;
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
        if (m_adaptive_forcing) {
            amrex::Print() << "Newton adaptive forcing:    true"
                           << " (gamma = "    << m_forcing_gamma
                           << ", xi = "       << m_forcing_xi
                           << ", zeta_max = " << m_forcing_zeta_max
                           << ", zeta_min = " << m_forcing_zeta_min << ")\n";
        }
        auto linsol_name = amrex::getEnumNameString(m_linear_solver_type);
        amrex::Print()     << "Newton linear solver:       " << linsol_name << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") verbose:            " << m_linsol_verbose_int << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") restart length:     " << m_linsol_restart_length << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") max iterations:     " << m_linsol_maxits << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") relative tolerance: " << m_linsol_rtol << "\n";
        amrex::Print()     << "Linear solver (" << linsol_name << ") absolute tolerance: " << m_linsol_atol << "\n";
        amrex::Print()     << "Preconditioner type:      " << amrex::getEnumNameString(m_pc_type) << "\n";

        m_linear_function->printParams();
    }

private:

    /**
     * \brief Intermediate Vec containers used by the solver.
     */
    mutable Vec m_dU, m_F, m_R;

    /**
     * \brief Pointer to Ops class.
     */
    Ops* m_ops = nullptr;

    /**
     * \brief Flag to determine whether convergence is required.
     */
    bool m_require_convergence = true;

    /**
     * \brief Relative tolerance for the Newton solver.
     */
    amrex::Real m_rtol = 1.0e-6;
    amrex::Real m_convergence_reference_norm = 0.0;

    /**
     * \brief Absolute tolerance for the Newton solver.
     */
    amrex::Real m_atol = 0.;

    /**
     * \brief Relative perturbation scale for the finite-difference JVP
     * (matrix-free Jacobian action) in the linear solve.
     */
    amrex::Real m_jfnk_epsilon = 1.0e-6;

    /** Default-off D1 common-direction operator partition.  This is a
     * destructive evidence diagnostic, not a timing mode: it adds full
     * particle residual evaluations and intentionally reports them in the
     * ordinary residual-evaluation count. */
    bool m_d1_operator_partition = false;
    int m_d1_step = -1;
    int m_d1_iteration = -1;
    bool m_d1_once = true;
    std::string m_d1_diagnostic_file;
    mutable bool m_d1_done = false;

    /**
     * \brief Maximum iterations for the Newton solver.
     */
    int m_maxits = 100;

    /**
     * \brief Total nonlinear iterations for the diagnostic file
     */
    mutable int m_total_iters = 0;

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
     */
    /** Optional backtracking line search (newton.line_search = 1):
     *  after the linear solve, accept the largest step fraction alpha in
     *  {1, 1/2, ..., 2^-max_halvings} that reduces the residual norm
     *  (keeping the best trial seen). Stabilizes Newton when the update
     *  overshoots -- stiff or state-dependent coefficients (vacuum
     *  resistivity ramps, flux limiters) and noisy residual landscapes. */
    bool m_use_line_search = false;
    int m_line_search_max_halvings = 6;
    /** newton.line_search_reuse_residual: when the accepted line-search trial is
     *  the last state evaluated, reuse its residual at the top of the next
     *  iteration instead of re-evaluating it (one particle push and deposit per
     *  Newton iteration on the particle-suppressed schemes). Default off: the
     *  re-evaluation at the same state agrees only to the deposition roundoff. */
    bool m_line_search_reuse_residual = false;
    /** Whether m_F (and the operator's RHS state) hold the residual at the
     *  current a_U, left by the accepted line-search trial. */
    mutable bool m_residual_current = false;   // Solve() is const
    /** Cap on the first line-search trial's update max-norm
     *  (newton.line_search_trial_max_norm, same units as the solver
     *  state; 0 = off). The trial state U - alpha*dU is evaluated through
     *  the full residual, and on kinetic solvers an unmoderated trial can
     *  hard-abort the run (particle displacements past the deposit guard
     *  range) before the line search gets a chance to reject it. When
     *  |dU|inf * alpha exceeds the cap, the search starts at
     *  alpha = cap/|dU|inf instead of 1. */
    amrex::Real m_line_search_trial_max_norm = 0.0;
    mutable Vec m_U_ls;

    /** Eisenstat-Walker adaptive forcing (newton.adaptive_forcing = 1):
     *  choose the GMRES relative tolerance each Newton iteration from the
     *  observed nonlinear residual decrease instead of a fixed value
     *  (Choice 2 with safeguards, Chacon JCP 2025 Eq. 21). Avoids
     *  oversolving the linear system when far from the root. Off by
     *  default; the fixed-tolerance path is unchanged. */
    bool m_adaptive_forcing = false;
    amrex::Real m_forcing_gamma = 0.9;
    amrex::Real m_forcing_xi = 1.5;
    amrex::Real m_forcing_zeta_max = 0.8;
    amrex::Real m_forcing_zeta_min = 1.0e-4;
    /** True when the user explicitly set the linear-solver relative
     *  tolerance; adaptive forcing then starts from that value on the
     *  first Newton iteration instead of zeta_max. */
    bool m_linsol_rtol_set = false;

    void EvalResidual ( Vec&         a_F,
                  const Vec&         a_U,
                  const Vec&         a_b,
                        amrex::Real  a_time,
                        int          a_iter ) const;

    void RunD1OperatorPartition (const Vec& a_U,
                                 const Vec& a_b,
                                 amrex::Real a_time,
                                 int a_step,
                                 int a_iter,
                                 int& a_residual_evals) const;

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
    m_U_ls.Define(a_U); // line-search base state
    m_F.Define(a_U); // residual function F(U) = U - b - R(U) = 0
    m_R.Define(a_U); // right hand side function R(U)

    m_ops = a_ops;

    m_linear_function = std::make_unique<JacobianFunctionMF<Vec,Ops>>();
    m_linear_function->define(m_F, m_ops, m_pc_type);
    m_linear_function->setJFNKEps(m_jfnk_epsilon);
    this->m_usePC = m_linear_function->usePreconditioner();

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
        diagnostic_file << "[" << c++ << "]exit_status";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]convergence_status";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]line_search_min_alpha";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]line_search_halvings";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]line_search_trials";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]linear_rtol_last";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]linear_rtol_min";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]linear_rtol_max";
        diagnostic_file << " ";
        diagnostic_file << "[" << c++ << "]residual_evals";
        diagnostic_file << "\n";
        diagnostic_file << "# exit_status is the public GetExitStatus value; "
                           "convergence_status: 2=absolute, 3=relative, "
                           "-5=tolerance-bearing nonconvergence, 0=fixed-iteration/no verdict.\n";
        diagnostic_file << "# line_search_min_alpha is the minimum accepted alpha (1 when off "
                           "or unused); counts are per solve. residual_evals counts full "
                           "nonlinear residual evaluations and excludes matrix-free JVP probes.\n";
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
    pp_newton.query("jfnk_epsilon",        m_jfnk_epsilon);
    pp_newton.query("d1_operator_partition", m_d1_operator_partition);
    pp_newton.query("d1_step",             m_d1_step);
    pp_newton.query("d1_iteration",        m_d1_iteration);
    pp_newton.query("d1_once",             m_d1_once);
    pp_newton.query("d1_diagnostic_file",  m_d1_diagnostic_file);
    pp_newton.query("line_search",          m_use_line_search);
    pp_newton.query("line_search_max_halvings", m_line_search_max_halvings);
    pp_newton.query("line_search_reuse_residual", m_line_search_reuse_residual);
    pp_newton.query("line_search_trial_max_norm", m_line_search_trial_max_norm);
    pp_newton.query("adaptive_forcing",    m_adaptive_forcing);
    pp_newton.query("forcing_gamma",       m_forcing_gamma);
    pp_newton.query("forcing_xi",          m_forcing_xi);
    pp_newton.query("forcing_zeta_max",    m_forcing_zeta_max);
    pp_newton.query("forcing_zeta_min",    m_forcing_zeta_min);
    pp_newton.query("diagnostic_file",     this->m_diagnostic_file);
    pp_newton.query("diagnostic_interval", this->m_diagnostic_interval);

    const amrex::ParmParse pp_l(amrex::getEnumNameString(m_linear_solver_type));
    pp_l.query("verbose_int",         m_linsol_verbose_int);
    pp_l.query("restart_length",      m_linsol_restart_length);
    pp_l.query("absolute_tolerance",  m_linsol_atol);
    if (pp_l.query("relative_tolerance", m_linsol_rtol)) { m_linsol_rtol_set = true; }
    pp_l.query("max_iterations",      m_linsol_maxits);

    /* backward compatibility */
    const amrex::ParmParse pp_gmres("gmres");
    pp_gmres.query("verbose_int",         m_linsol_verbose_int);
    pp_gmres.query("restart_length",      m_linsol_restart_length);
    pp_gmres.query("absolute_tolerance",  m_linsol_atol);
    if (pp_gmres.query("relative_tolerance", m_linsol_rtol)) { m_linsol_rtol_set = true; }
    pp_gmres.query("max_iterations",      m_linsol_maxits);

    const amrex::ParmParse pp_jac("jacobian");
    pp_jac.query("pc_type", m_pc_type);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_jfnk_epsilon > 0.0 && std::isfinite(m_jfnk_epsilon),
        "newton.jfnk_epsilon must be positive and finite");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_d1_step >= -1 && m_d1_iteration >= -1,
        "newton.d1_step and newton.d1_iteration must be -1 or non-negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_d1_operator_partition ||
        m_pc_type == PreconditionerType::pc_block_banded,
        "newton.d1_operator_partition requires jacobian.pc_type = pc_block_banded; "
        "C is not exposed for other or curl-curl preconditioners");
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
    amrex::Real solve_norm0 = 1._rt;
    amrex::Real norm_rel = 0.;

    // Eisenstat-Walker adaptive forcing state: per-iteration GMRES relative
    // tolerance zeta_k, plus the previous iteration's zeta and residual norm
    // (Choice 2 with safeguards, Chacon JCP 2025 Eq. 21). With adaptive
    // forcing off, linsol_rtol stays at the fixed m_linsol_rtol.
    amrex::Real linsol_rtol = m_linsol_rtol;
    amrex::Real zeta_prev = m_forcing_zeta_max;
    amrex::Real norm_prev = 1._rt;
    amrex::Real linear_rtol_last = 0._rt;
    amrex::Real linear_rtol_min = 0._rt;
    amrex::Real linear_rtol_max = 0._rt;
    bool linear_solve_performed = false;

    // Per-solve evidence counters. A line search accepts one alpha per
    // Newton update; retain its minimum and aggregate the explored ladder.
    // With line search disabled (or no update needed), alpha remains one and
    // both counts remain zero.
    amrex::Real line_search_min_alpha = 1._rt;
    int line_search_halvings = 0;
    int line_search_trials = 0;

    // Count the full nonlinear residual evaluations initiated by this
    // routine. Matrix-free JVP probes call ComputeRHS directly and are
    // intentionally excluded; their count is represented by the linear
    // iteration columns instead.
    int residual_evals = 0;
    auto eval_residual = [&] (Vec& a_F, const Vec& a_residual_U, int a_iter) {
        ++residual_evals;
        EvalResidual(a_F, a_residual_U, a_b, a_time, a_iter);
    };

    int iter;
    int linear_solver_iters = 0;
    // Solve() is const and the object is reused across time steps. Always
    // clear a prior success/failure before grading this solve.
    this->m_exit_status = 0;
    int convergence_status = 0;
    m_residual_current = false;
    for (iter = 0; iter <= m_maxits; ++iter) {

        // Compute residual: F(U) = U - b - R(U). This is the evaluation the
        // linear solve linearizes about: the operator may deposit its
        // mass matrices here. Skipped when the accepted line-search trial of
        // the previous iteration already evaluated this exact state.
        if (!(m_line_search_reuse_residual && m_residual_current)) {
            m_ops->SetMassMatricesWanted(true);
            eval_residual(m_F, a_U, iter);
        }
        m_residual_current = false;

        // Compute norm of the residual
        norm_abs = m_F.norm2();
        if (iter == 0) {
            solve_norm0 = norm_abs > 0. ? norm_abs : 1._rt;
            norm0 = m_convergence_reference_norm > 0.
                ? m_convergence_reference_norm : solve_norm0;
        }
        norm_rel = norm_abs/norm0;

        // Check for convergence criteria
        if (this->m_verbose) {
            amrex::Print() << "Newton: iteration = " << std::setw(3) << iter <<  ", norm = "
                           << std::scientific << std::setprecision(5) << norm_abs << " (abs.), "
                           << std::scientific << std::setprecision(5) << norm_rel << " (rel.)" << "\n";
        }

        if (norm_abs < m_atol) {
            this->m_exit_status = 2;
            convergence_status = 2;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied absolute tolerance " << m_atol << "\n";
            }
            break;
        }

        if (norm_rel < m_rtol) {
            this->m_exit_status = 3;
            convergence_status = 3;
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iteration = " << std::setw(3) << iter
                               << ". Satisfied relative tolerance " << m_rtol << "\n";
            }
            break;
        }

        // The last permitted Newton update has already been applied when
        // iter == m_maxits. Check its residual above before declaring that
        // the iteration limit was reached.
        if (iter == m_maxits) {
            if (this->m_verbose) {
                amrex::Print() << "Newton: exiting at iter = " << std::setw(3) << iter
                               << ". Maximum iteration reached: iter = " << m_maxits << "\n";
            }
            break;
        }

        // A shared outer convergence reference may be much smaller than
        // this inner solve's initial residual after its constraint changes.
        // Divergence means growth within THIS Newton solve, not growth from
        // a different outer problem. Preserve the same convergence target.
        if (norm_abs > 100._rt*solve_norm0) {
            this->m_exit_status = -9;
            convergence_status = -9;
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

        // Eisenstat-Walker Choice-2 forcing: set this iteration's GMRES
        // relative tolerance from the observed nonlinear residual decrease.
        if (m_adaptive_forcing) {
            amrex::Real zeta;
            if (iter == 0) {
                // No residual history yet: start from zeta_max, unless the
                // user pinned a fixed linear-solver relative tolerance (which
                // is honored exactly, below the zeta_min floor if so pinned).
                zeta = m_linsol_rtol_set ? m_linsol_rtol : m_forcing_zeta_max;
            } else {
                const amrex::Real eps_newton = m_rtol*norm0 + m_atol;
                const amrex::Real zeta_A = m_forcing_gamma
                    * std::pow(norm_abs/norm_prev, m_forcing_xi);
                const amrex::Real zeta_B = std::min( m_forcing_zeta_max,
                    std::max(zeta_A, m_forcing_gamma*std::pow(zeta_prev, m_forcing_xi)) );
                zeta = std::min( m_forcing_zeta_max,
                    std::max(zeta_B, m_forcing_gamma*eps_newton/norm_abs) );
                zeta = std::max(zeta, m_forcing_zeta_min);
            }
            zeta_prev = zeta;
            norm_prev = norm_abs;
            linsol_rtol = zeta;
            if (this->m_verbose) {
                amrex::Print() << "Newton: adaptive forcing zeta = "
                               << std::scientific << std::setprecision(5)
                               << zeta << "\n";
            }
        }

        // Solve linear system for Newton step [Jac]*dU = F
        linear_rtol_last = linsol_rtol;
        if (linear_solve_performed) {
            linear_rtol_min = std::min(linear_rtol_min, linsol_rtol);
            linear_rtol_max = std::max(linear_rtol_max, linsol_rtol);
        } else {
            linear_rtol_min = linsol_rtol;
            linear_rtol_max = linsol_rtol;
            linear_solve_performed = true;
        }
        m_dU.zero();
        m_linear_solver->solve( m_dU, m_F, linsol_rtol, m_linsol_atol );
        linear_solver_iters += m_linear_solver->getNumIters();

        const bool d1_selected = m_d1_operator_partition
            && (!m_d1_once || !m_d1_done)
            && (m_d1_step < 0 || m_d1_step == a_step)
            && (m_d1_iteration < 0 || m_d1_iteration == iter);
        if (d1_selected) {
            RunD1OperatorPartition(a_U, a_b, a_time, a_step, iter,
                                   residual_evals);
        }

        // Jacobian consistency instrument (WARPX_JAC_CHECK=1): along the
        // solved Newton direction, compare the linear operator GMRES used
        // (probe/linear-stage semantics) against true-residual secants over
        // an epsilon ladder (nonlinear-evaluation semantics). A converged
        // linear solve that yields a non-descent direction implies these
        // disagree; the epsilon dependence separates non-smoothness of the
        // map from a frozen/approximate operator model. Diagnostic only --
        // it mutates grid state like any residual evaluation, and the line
        // search / next iteration re-evaluate from the solver state anyway.
        if (std::getenv("WARPX_JAC_CHECK") != nullptr) {
            {
                const auto& dU_mfs = m_dU.getArrayVec();
                for (int dir = 0; dir < 3; ++dir) {
                    amrex::Print() << "JAC_CHECK: iter = " << iter
                        << ", dir " << dir << " |dU|inf = "
                        << std::scientific
                        << dU_mfs[0][dir]->norminf(0) << "\n";
                }
            }
            Vec v, Jv, Up, Fp, F0;
            v.Define(a_U); Jv.Define(a_U); Up.Define(a_U);
            Fp.Define(a_U); F0.Define(a_U);
            v.Copy(m_dU);
            const amrex::Real vnorm = v.norm2();
            if (vnorm > 0._rt) {
                v.scale(1._rt/vnorm);
                // Operator as the Krylov solve saw it (baseline vectors were
                // set before the solve and are unmodified).
                m_linear_function->apply(Jv, v);
                const amrex::Real Jvnorm = Jv.norm2();
                // Fresh nonlinear-semantics base for the secants.
                eval_residual(F0, a_U, iter);
                // Ladder tops out at the JFNK scale: larger rungs create
                // perturbed states wild enough to push particles past the
                // deposit guard range and abort the run.
                for (int p = -2; p <= 0; ++p) {
                    const amrex::Real eps = m_linear_function->perturbationScale(
                        v, std::pow(10._rt, 2*p) * m_jfnk_epsilon);
                    Up.linComb(1._rt, a_U, eps, v);
                    eval_residual(Fp, Up, iter);
                    Fp.linComb(1._rt/eps, Fp, -1._rt/eps, F0);
                    Fp.linComb(1._rt, Fp, -1._rt, Jv);
                    amrex::Print() << "JAC_CHECK: iter = " << iter
                        << ", eps = " << std::scientific << eps
                        << ", ||secant - Jv||/||Jv|| = "
                        << Fp.norm2()/Jvnorm << "\n";
                }
                // Restore the base-state evaluation for the update below.
                eval_residual(m_F, a_U, iter);
            }
        }

        // Update solution
        if (m_use_line_search) {
            // Backtracking: try alpha = 1, 1/2, ... and accept the first
            // trial that reduces the residual norm (falling back to the
            // best trial seen). Trial residuals reuse EvalResidual; the
            // accepted state is re-evaluated at the top of the next
            // Newton iteration as usual.
            m_U_ls.Copy(a_U);
            amrex::Real alpha = 1._rt;
            if (m_line_search_trial_max_norm > 0._rt) {
                const auto& dU_mfs = m_dU.getArrayVec();
                amrex::Real dU_inf = 0._rt;
                for (int dir = 0; dir < 3; ++dir) {
                    dU_inf = std::max(dU_inf, dU_mfs[0][dir]->norminf(0));
                }
                if (dU_inf * alpha > m_line_search_trial_max_norm) {
                    alpha = m_line_search_trial_max_norm / dU_inf;
                    if (this->m_verbose) {
                        amrex::Print() << "Newton: line search first trial "
                            "capped at alpha = " << alpha
                            << " (|dU|inf = " << std::scientific << dU_inf
                            << ")\n";
                    }
                }
            }
            amrex::Real best_alpha = alpha;
            amrex::Real best_norm = -1._rt;
            amrex::Real last_alpha = alpha;
            int trials_this_update = 0;
            m_ops->SetMassMatricesWanted(false);   // trials never deposit mass matrices
            for (int ls = 0; ls <= m_line_search_max_halvings; ++ls) {
                a_U.linComb(1._rt, m_U_ls, -alpha, m_dU);
                eval_residual(m_F, a_U, iter);
                ++trials_this_update;
                last_alpha = alpha;
                amrex::Real const trial_norm = m_F.norm2();
                if (best_norm < 0._rt || trial_norm < best_norm) {
                    best_norm = trial_norm;
                    best_alpha = alpha;
                }
                if (trial_norm < norm_abs) { break; }
                alpha *= 0.5_rt;
            }
            a_U.linComb(1._rt, m_U_ls, -best_alpha, m_dU);
            line_search_min_alpha = std::min(line_search_min_alpha, best_alpha);
            line_search_trials += trials_this_update;
            line_search_halvings += trials_this_update - 1;
            // m_F and the operator's RHS hold F at the last trial; that is the
            // accepted state exactly when the accepted alpha was the last tried.
            m_residual_current = (best_alpha == last_alpha);
            if (this->m_verbose && best_alpha < 1._rt) {
                amrex::Print() << "Newton: line search accepted alpha = "
                               << best_alpha << "\n";
            }
        } else {
            a_U -= m_dU;
        }

    }

    // Update total iteration count
    m_total_iters += iter;
    m_total_linsol_iters += linear_solver_iters;

    // The failure comparison is the exact complement of the strict success
    // tests above. A zero relative tolerance requests a fixed number of Newton
    // updates and intentionally suppresses the non-convergence warning.
    if (norm_rel >= m_rtol && norm_abs >= m_atol && m_rtol > 0.) {
        // Keep GetExitStatus behavior compatible: a permissive solve returns
        // zero so the caller can continue. The diagnostic convergence status
        // remains negative so those tolerated failures cannot be mistaken
        // for converged campaign evidence.
        convergence_status = -5;
        std::stringstream convergenceMsg;
        convergenceMsg << "Newton solver failed to converge after " << iter <<
                          " iterations. Relative norm is " << norm_rel <<
                          " and the relative tolerance is " << m_rtol <<
                          ". Absolute norm is " << norm_abs <<
                          " and the absolute tolerance is " << m_atol;
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
        diagnostic_file << this->m_exit_status;
        diagnostic_file << " ";
        diagnostic_file << convergence_status;
        diagnostic_file << " ";
        diagnostic_file << line_search_min_alpha;
        diagnostic_file << " ";
        diagnostic_file << line_search_halvings;
        diagnostic_file << " ";
        diagnostic_file << line_search_trials;
        diagnostic_file << " ";
        diagnostic_file << linear_rtol_last;
        diagnostic_file << " ";
        diagnostic_file << linear_rtol_min;
        diagnostic_file << " ";
        diagnostic_file << linear_rtol_max;
        diagnostic_file << " ";
        diagnostic_file << residual_evals;
        diagnostic_file << "\n";
        diagnostic_file.close();
    }

}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::RunD1OperatorPartition (
    const Vec& a_U, const Vec& a_b, amrex::Real a_time,
    int a_step, int a_iter, int& a_residual_evals) const
{
    using namespace amrex::literals;
    static constexpr int schema = 1;
    static constexpr int nrungs = 3;

    std::ofstream d1_file;
    if (amrex::ParallelDescriptor::IOProcessor()
        && !m_d1_diagnostic_file.empty()) {
        const std::filesystem::path path(m_d1_diagnostic_file);
        const std::filesystem::path dir = path.parent_path();
        if (!dir.empty()) { std::filesystem::create_directories(dir); }
        // A solver may emit several selected records when d1_once=false, but
        // the first selected record of each process must replace stale output
        // from an earlier run in the same directory.
        const auto mode = std::ofstream::out
            | (m_d1_done ? std::ofstream::app : std::ofstream::trunc);
        d1_file.open(m_d1_diagnostic_file, mode);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(d1_file.good(),
            "could not open newton.d1_diagnostic_file for append");
    }
    auto emit = [&] (const std::string& line) {
        amrex::Print() << line << "\n";
        if (d1_file.is_open()) { d1_file << line << "\n"; }
    };
    auto line_prefix = [&] (const char* record) {
        std::ostringstream os;
        os << record << " schema=" << schema
           << " step=" << a_step << " iteration=" << a_iter;
        return os;
    };

    // Retain the exact full base used to anchor the matrix-free operator.
    // GMRES owns independent copies, but these let us form B_g-B_s and
    // measure whether a later full evaluation repeats the saved base.
    Vec R_full_saved, F_full_saved;
    R_full_saved.Define(a_U); F_full_saved.Define(a_U);
    R_full_saved.Copy(m_R); F_full_saved.Copy(m_F);

    // One layout-independent deterministic RZ direction.  Periodic duplicate
    // z nodes hash to the same physical node; WarpXSolverVec's norm uses the
    // field dot mask, so normalization is invariant to ownership layout.
    Vec v;
    v.Define(a_U);
    v.zero();
    const auto& geom = m_ops->GetGeometry(0);
    const int zlo = geom.Domain().smallEnd(1);
    const int zn = geom.Domain().length(1);
    const bool zperiodic = geom.isPeriodic(1);
    auto& v_fields = v.getArrayVec();
    for (int comp = 0; comp < 3; ++comp) {
        amrex::MultiFab& mf = *(v_fields[0][comp]);
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            auto arr = mf.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(),
                newton_solver_detail::D1RZHashDirection{
                    arr, zlo, zn, comp, zperiodic});
        }
        mf.FillBoundaryAndSync(geom.periodicity());
    }
    const amrex::Real raw_vnorm = v.norm2();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(raw_vnorm) && raw_vnorm > 0._rt,
        "D1 deterministic direction has invalid norm");
    v.scale(1._rt/raw_vnorm);
    const amrex::Real vnorm = v.norm2();

    {
        auto os = line_prefix("D1_BEGIN");
        os << " direction=rz_periodic_hash_v1"
           << " direction_norm=" << std::scientific << std::setprecision(17)
           << vnorm << " epsilon_relative_top=" << m_jfnk_epsilon
           << " rungs=" << nrungs << " destructive=1 timing_valid=0";
        emit(os.str());
    }
    for (int comp = 0; comp < 3; ++comp) {
        const amrex::MultiFab& mf = *(v.getArrayVec()[0][comp]);
        auto os = line_prefix("D1_DIRECTION");
        os << " component=" << comp
           << " norm=" << std::scientific << std::setprecision(17)
           << mf.norm2(0) << " max_abs=" << mf.norminf(0);
        emit(os.str());
    }

    // Machine-readable metric emitter.  Every record has the same scalar
    // fields; component records also identify the worst absolute (r,z) row.
    auto emit_metric = [&] (const std::string& quantity,
                            const Vec& value, const Vec& reference,
                            int rung, amrex::Real eps_relative,
                            amrex::Real eps_absolute) {
        const amrex::Real n = value.norm2();
        const amrex::Real nr = reference.norm2();
        const amrex::Real rel = (nr > 0._rt) ? n/nr : n;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(n) && std::isfinite(nr) && std::isfinite(rel),
            "D1 produced a non-finite global metric");
        {
            auto os = line_prefix("D1_METRIC");
            os << " rung=" << rung
               << " eps_relative=" << std::scientific << std::setprecision(17)
               << eps_relative << " eps_absolute=" << eps_absolute
               << " quantity=" << quantity << " norm=" << n
               << " reference_norm=" << nr << " relative=" << rel;
            emit(os.str());
        }
        for (int comp = 0; comp < 3; ++comp) {
            const amrex::MultiFab& mf = *(value.getArrayVec()[0][comp]);
            const amrex::MultiFab& ref_mf = *(reference.getArrayVec()[0][comp]);
            const amrex::Real cn = mf.norm2(0);
            const amrex::Real cr = ref_mf.norm2(0);
            const amrex::Real crel = (cr > 0._rt) ? cn/cr : cn;
            amrex::MultiFab abs_mf(mf.boxArray(), mf.DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(abs_mf, mf, 0, 0, 1, 0);
            abs_mf.abs(0, 1, 0);
            const amrex::Real vmax = abs_mf.norminf(0);
            const amrex::IntVect loc = abs_mf.maxIndex(0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(cn) && std::isfinite(cr)
                && std::isfinite(crel) && std::isfinite(vmax),
                "D1 produced a non-finite component metric");
            auto os = line_prefix("D1_COMPONENT");
            os << " rung=" << rung
               << " eps_relative=" << std::scientific << std::setprecision(17)
               << eps_relative << " eps_absolute=" << eps_absolute
               << " quantity=" << quantity << " component=" << comp
               << " norm=" << cn << " reference_norm=" << cr
               << " relative=" << crel << " max_abs=" << vmax
               << " max_r=" << loc[0] << " max_z=" << loc[1];
            emit(os.str());
        }
    };

    // B_g must be sampled before any additional full residual changes the
    // held particle/mass-matrix state.  The same stored directions feed all
    // subsequent legs of the evidence chain.
    const std::array<amrex::Real, nrungs> eps_relative = {
        m_jfnk_epsilon*1.0e-4_rt,
        m_jfnk_epsilon*1.0e-2_rt,
        m_jfnk_epsilon
    };
    std::array<amrex::Real, nrungs> eps_absolute{};
    std::array<Vec, nrungs> B_g;
    for (int rung = 0; rung < nrungs; ++rung) {
        B_g[rung].Define(a_U);
        eps_absolute[rung] = m_linear_function->perturbationScale(
            v, eps_relative[rung]);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(eps_absolute[rung]) && eps_absolute[rung] > 0._rt,
            "D1 produced an invalid absolute epsilon");
        m_linear_function->applyWithJFNKEps(
            B_g[rung], v, eps_relative[rung]);
    }

    // C is the unfactored frozen BB action, not the inverse applied by GMRES.
    Vec C;
    C.Define(a_U);
    if (!m_linear_function->applyFrozenPreconditionerOperator(C, v)) {
        auto os = line_prefix("D1_STATUS");
        os << " status=failed reason=frozen_operator_unavailable";
        emit(os.str());
        WARPX_ABORT_WITH_MESSAGE(
            "D1 requires a built e-form pc_block_banded frozen operator");
    }

    // Stage base and repeat still use the intact linear-stage state.  These
    // complete B_s and measure its same-state noise before the first full
    // particle residual below invalidates that state.
    Vec R_stage0, R_stage1, stage_repeat;
    R_stage0.Define(a_U); R_stage1.Define(a_U); stage_repeat.Define(a_U);
    m_ops->ComputeRHS(R_stage0, a_U, a_time, -1, true);
    m_ops->ComputeRHS(R_stage1, a_U, a_time, -1, true);
    stage_repeat.Copy(R_stage1);
    stage_repeat.increment(R_stage0, -1._rt);

    // Two consecutive full evaluations establish the full-map repeat floor.
    // The saved-vs-fresh record additionally detects drift since the exact
    // full base used by GMRES was captured.
    Vec F_full0, F_full1, full_repeat, full_saved_repeat;
    F_full0.Define(a_U); F_full1.Define(a_U);
    full_repeat.Define(a_U); full_saved_repeat.Define(a_U);
    ++a_residual_evals;
    EvalResidual(F_full0, a_U, a_b, a_time, a_iter);
    ++a_residual_evals;
    EvalResidual(F_full1, a_U, a_b, a_time, a_iter);
    full_repeat.Copy(F_full1);
    full_repeat.increment(F_full0, -1._rt);
    full_saved_repeat.Copy(F_full0);
    full_saved_repeat.increment(F_full_saved, -1._rt);

    emit_metric("stage_repeat", stage_repeat, R_stage0,
                -1, 0._rt, 0._rt);
    emit_metric("full_repeat", full_repeat, F_full0,
                -1, 0._rt, 0._rt);
    emit_metric("full_saved_repeat", full_saved_repeat, F_full_saved,
                -1, 0._rt, 0._rt);

    Vec U_probe, F_probe, A, saved_base_drift_over_eps, anchor, B_s;
    Vec A_minus_B_s, B_s_minus_C, B_g_minus_B_s, A_minus_C, closure;
    U_probe.Define(a_U); F_probe.Define(a_U); A.Define(a_U);
    saved_base_drift_over_eps.Define(a_U);
    anchor.Define(a_U); B_s.Define(a_U);
    A_minus_B_s.Define(a_U); B_s_minus_C.Define(a_U);
    B_g_minus_B_s.Define(a_U); A_minus_C.Define(a_U);
    closure.Define(a_U);

    for (int rung = 0; rung < nrungs; ++rung) {
        const amrex::Real eps = eps_absolute[rung];

        // B_g-B_s = [R_full(U)-R_stage(U)]/eps.  Deriving B_s from
        // the already-captured B_g uses the identical stage probe and avoids
        // contaminating the partition with a second probe evaluation.
        anchor.linComb(1._rt/eps, R_full_saved,
                       -1._rt/eps, R_stage0);
        B_s.Copy(B_g[rung]);
        B_s.increment(anchor, -1._rt);

        U_probe.linComb(1._rt, a_U, eps, v);
        ++a_residual_evals;
        EvalResidual(F_probe, U_probe, a_b, a_time, a_iter);
        // Primary A is the full nonlinear derivative diagnostic: use the
        // fresh full-map base, matching the established WARPX_JAC_CHECK
        // semantics.  The saved GMRES/Newton base can cease to be
        // reproducible after the linear-stage state transition; report that
        // actual-base consistency defect separately at its 1/eps scale so it
        // cannot erase the differential A-B_s measurement.
        A.linComb(1._rt/eps, F_probe, -1._rt/eps, F_full0);
        saved_base_drift_over_eps.Copy(full_saved_repeat);
        saved_base_drift_over_eps.scale(1._rt/eps);

        A_minus_B_s.Copy(A);
        A_minus_B_s.increment(B_s, -1._rt);
        B_s_minus_C.Copy(B_s);
        B_s_minus_C.increment(C, -1._rt);
        B_g_minus_B_s.Copy(B_g[rung]);
        B_g_minus_B_s.increment(B_s, -1._rt);
        A_minus_C.Copy(A);
        A_minus_C.increment(C, -1._rt);

        // Four-operator closure:
        // (A-B_g) + (B_g-B_s) + (B_s-C) - (A-C) = 0.
        closure.Copy(A);
        closure.increment(B_g[rung], -1._rt);
        closure.increment(B_g_minus_B_s, 1._rt);
        closure.increment(B_s_minus_C, 1._rt);
        closure.increment(A_minus_C, -1._rt);

        emit_metric("A", A, A, rung, eps_relative[rung], eps);
        emit_metric("B_g", B_g[rung], B_g[rung], rung,
                    eps_relative[rung], eps);
        emit_metric("B_s", B_s, B_s, rung, eps_relative[rung], eps);
        emit_metric("C", C, C, rung, eps_relative[rung], eps);
        emit_metric("A_minus_B_s", A_minus_B_s, A, rung,
                    eps_relative[rung], eps);
        emit_metric("B_s_minus_C", B_s_minus_C, B_s, rung,
                    eps_relative[rung], eps);
        emit_metric("B_g_minus_B_s", B_g_minus_B_s, B_g[rung], rung,
                    eps_relative[rung], eps);
        emit_metric("A_minus_C", A_minus_C, A, rung,
                    eps_relative[rung], eps);
        emit_metric("full_saved_drift_over_eps", saved_base_drift_over_eps,
                    A, rung, eps_relative[rung], eps);
        emit_metric("closure", closure, A_minus_C, rung,
                    eps_relative[rung], eps);
    }

    // Leave the ordinary Newton update at a freshly evaluated full base, as
    // the legacy WARPX_JAC_CHECK path does.  This extra evaluation is part of
    // the diagnostic cost and therefore remains visible in residual_evals.
    ++a_residual_evals;
    EvalResidual(m_F, a_U, a_b, a_time, a_iter);

    {
        auto os = line_prefix("D1_END");
        os << " status=complete rungs=" << nrungs;
        emit(os.str());
    }
    if (d1_file.is_open()) { d1_file.flush(); }
    m_d1_done = true;
}

template <class Vec, class Ops>
void NewtonSolver<Vec,Ops>::EvalResidual ( Vec&         a_F,
                                     const Vec&         a_U,
                                     const Vec&         a_b,
                                           amrex::Real  a_time,
                                           int          a_iter ) const
{
    BL_PROFILE("NewtonSolver::EvalResidual()");
    m_ops->ComputeRHS( m_R, a_U, a_time, a_iter, false );

    // Compute residual: F(U) = U - b - R(U)
    a_F.Copy(a_U);
    a_F -= m_R;
    a_F -= a_b;

}

#endif
