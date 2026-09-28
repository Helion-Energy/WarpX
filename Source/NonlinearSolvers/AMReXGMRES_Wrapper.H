/* Copyright 2025 Debojyoti Ghosh
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef AMREX_GMRES_WRAPPER_H_
#define AMREX_GMRES_WRAPPER_H_

#include "LinearSolver.H"

#include <AMReX_GMRES.H>

#include <cmath>
#include <memory>

/**
 * \brief Wrapper for AMReX's GMRES solver
 *
 *    This class is a wrapper for AMReX's GMRES algorithm that inherits from
 *    WarpX's LinearSolver base class. See the documentation of AMReX's GMRES
 *    solver for more details. This class is templated on a vector class Vec,
 *    and an operator class Ops.
 *
 *    The Ops class must have the following function:
 *    ComputeRHS( R_vec, U_vec, time, nl_iter, from_jacobian ),
 *    where U_vec and R_vec are of type Vec.
 *
 *    The Vec class must have basic math operators, such as Copy, +=, -=,
 *    increment(), linComb(), scale(), etc.. See WarpXSolverVec.H for an example.
 */
template <typename Vec, typename Ops>
class AMReXGMRES : public LinearSolver<Vec,Ops>
{
public:

    using RT = typename Ops::RT; // double or float

    AMReXGMRES () = default;

    ~AMReXGMRES () override = default;

    // Prohibit Move and Copy operations
    AMReXGMRES(const AMReXGMRES&) = delete;
    AMReXGMRES& operator=(const AMReXGMRES&) = delete;
    AMReXGMRES(AMReXGMRES&&) noexcept = delete;
    AMReXGMRES& operator=(AMReXGMRES&&) noexcept = delete;

    //! Defines with a reference to Ops. It's the user's responsibility to
    //! keep the Ops object alive for GMRES to be functional. This function
    //! must be called before solve() can be called.
    AMREX_FORCE_INLINE
    void define (Ops& linop) override
    {
        m_checked = std::make_unique<CheckedOps>(linop);
        m_solver = std::make_unique<amrex::GMRES<Vec,CheckedOps>>();
        m_solver->define(*m_checked);
        m_failed_evaluation = false;
    }

    /**
     * \brief Solve the linear system
     *
     * \param a_sol     unknowns, i.e., x in A x = b.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \param a_its     optional argument specifying the maximum number of iterations.
     */
    AMREX_FORCE_INLINE
    void solve (Vec& a_sol,
                Vec const& a_rhs,
                RT a_tol_rel,
                RT a_tol_abs,
                int a_its=-1) override
    {
        // Reset scratch and diagnostics for every solve. This also discards an
        // interrupted Arnoldi cycle while retaining iteration/restart limits.
        m_solver->define(*m_checked);
        m_failed_evaluation = false;
        try {
            m_solver->solve(a_sol, a_rhs, a_tol_rel, a_tol_abs, a_its);
            m_checked->check(m_solver->getResidualNorm(), "final residual");
            m_checked->check(m_checked->norm2(a_sol), "solution");
        } catch (NonFiniteEvaluation const& failure) {
            // Do not turn a rejected residual into another Arnoldi direction or
            // expose a partial correction to Newton. This exception remains
            // inside the C++ AMReX solve; it never crosses a PETSc C callback.
            m_failed_evaluation = true;
            m_checked->setToZero(a_sol);
            amrex::Print() << "GMRES: rejected nonfinite " << failure.operation
                           << "; completed iterations=" << m_solver->getNumIters()
                           << ", last residual=" << m_solver->getResidualNorm() << '\n';
        }
    }

    //! Sets verbosity.
    AMREX_FORCE_INLINE
    void setVerbose (int v) override { m_solver->setVerbose(v); }

    //! Sets restart length. The default is 30.
    AMREX_FORCE_INLINE
    void setRestartLength (int rl) override { m_solver->setRestartLength(rl); }

    //! Sets the max number of iterations
    AMREX_FORCE_INLINE
    void setMaxIters (int niters) override { m_solver->setMaxIters(niters); }

    //! Gets the number of iterations.
    AMREX_FORCE_INLINE
    int getNumIters () const override { return m_solver->getNumIters(); }

    //! Gets the solver status.
    AMREX_FORCE_INLINE
    int getStatus () const override
    {
        return m_failed_evaluation ? 2 : m_solver->getStatus();
    }

    //! A rejected evaluation is fatal even when finite iteration-limit failure is tolerated.
    [[nodiscard]] bool failedEvaluation () const override { return m_failed_evaluation; }

    //! Gets the 2-norm of the residual.
    AMREX_FORCE_INLINE
    RT getResidualNorm () const override { return m_solver->getResidualNorm(); }

private:
    struct NonFiniteEvaluation
    {
        char const* operation;
    };

    // Ops::norm2/dotProduct must be collective over the solver communicator.
    // Every rank then takes the same exception path, including empty ranks.
    // Forward finite operations exactly; no arithmetic or tolerance is changed.
    struct CheckedOps
    {
        using RT = typename Ops::RT;
        Ops& op;
        explicit CheckedOps (Ops& value) : op(value) {}

        static RT check (RT value, char const* operation)
        {
            if (!std::isfinite(value)) { throw NonFiniteEvaluation{operation}; }
            return value;
        }
        Vec makeVecLHS () { return op.makeVecLHS(); }
        Vec makeVecRHS () { return op.makeVecRHS(); }
        void apply (Vec& out, Vec const& in)
        {
            check(op.norm2(in), "operator input");
            op.apply(out, in);
            check(op.norm2(out), "operator result (residual/Jacobian evaluation)");
        }
        void precond (Vec& out, Vec const& in)
        {
            check(op.norm2(in), "preconditioner input");
            op.precond(out, in);
            check(op.norm2(out), "preconditioner result");
        }
        RT norm2 (Vec const& value) { return check(op.norm2(value), "vector norm"); }
        RT dotProduct (Vec const& x, Vec const& y)
        {
            return check(op.dotProduct(x, y), "dot product");
        }
        void assign (Vec& out, Vec const& in) { op.assign(out, in); }
        void setToZero (Vec& value) { op.setToZero(value); }
        void scale (Vec& value, RT factor)
        {
            op.scale(value, check(factor, "scale factor"));
        }
        void increment (Vec& out, Vec const& in, RT factor)
        {
            op.increment(out, in, check(factor, "increment factor"));
        }
        void linComb (Vec& out, RT a, Vec const& x, RT b, Vec const& y)
        {
            op.linComb(out, check(a, "combination factor"), x,
                       check(b, "combination factor"), y);
        }
    };

    bool m_failed_evaluation = false;
    std::unique_ptr<CheckedOps> m_checked = nullptr;
    std::unique_ptr<amrex::GMRES<Vec,CheckedOps>> m_solver = nullptr;
};

#endif
