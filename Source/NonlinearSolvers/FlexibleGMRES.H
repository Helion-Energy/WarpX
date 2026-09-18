/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FLEXIBLE_GMRES_H_
#define WARPX_FLEXIBLE_GMRES_H_

#include "LinearSolver.H"

#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

/** Restarted right-preconditioned flexible GMRES.
 *
 * Store each actual preconditioned Arnoldi vector. This permits an iterative
 * preconditioner with RHS-dependent stopping or different actions between
 * applications. The matrix action must still represent one frozen linear
 * problem. Recompute the true residual after every solution update, including
 * an apparent happy breakdown, before reporting convergence.
 */
template <typename Vec, typename Ops>
class FlexibleGMRES : public LinearSolver<Vec, Ops>
{
public:
    using RT = typename Ops::RT;

    void define (Ops& linop) override { m_ops = &linop; }
    void setVerbose (int value) override { m_verbose = value; }
    void setRestartLength (int value) override
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(value > 0, "FGMRES restart length must be positive");
        m_restart = value;
    }
    void setMaxIters (int value) override
    {
        if (value >= 0) { m_max_iterations = value; }
    }
    [[nodiscard]] int getNumIters () const override { return m_iterations; }
    [[nodiscard]] int getStatus () const override { return m_status; }
    [[nodiscard]] RT getResidualNorm () const override { return m_residual; }

    void solve (Vec& solution, Vec const& rhs, RT relative_tolerance,
                RT absolute_tolerance, int maximum_iterations = -1) override
    {
        BL_PROFILE("FlexibleGMRES::solve()");
        AMREX_ALWAYS_ASSERT(m_ops != nullptr);
        AMREX_ALWAYS_ASSERT(relative_tolerance >= RT(0) && absolute_tolerance >= RT(0));
        int const limit = maximum_iterations < 0 ? m_max_iterations : maximum_iterations;
        m_iterations = 0;
        m_status = 1;
        m_ops->setToZero(solution); // Same zero-initial-guess contract as AMReX GMRES.
        Vec residual = m_ops->makeVecRHS();
        Vec action = m_ops->makeVecRHS();
        m_ops->assign(residual, rhs);
        RT const initial_norm = m_ops->norm2(residual);
        RT const target = std::max(absolute_tolerance, relative_tolerance * initial_norm);
        m_residual = initial_norm;
        if (!std::isfinite(initial_norm)) { m_status = 2; return; }
        if (m_residual <= target) { m_status = 0; return; }

        while (m_iterations < limit) {
            int const length = std::min(m_restart, limit - m_iterations);
            std::vector<Vec> basis;
            std::vector<Vec> preconditioned;
            basis.reserve(length + 1);
            preconditioned.reserve(length);
            basis.emplace_back(m_ops->makeVecRHS());
            m_ops->assign(basis[0], residual);
            m_ops->scale(basis[0], RT(1) / m_residual);
            std::vector<RT> h((length + 1) * length, RT(0));
            std::vector<RT> cosine(length), sine(length), g(length + 1, RT(0));
            auto H = [&] (int row, int col) -> RT& { return h[col * (length + 1) + row]; };
            g[0] = m_residual;
            int columns = 0;
            bool breakdown = false;

            for (int column = 0; column < length; ++column) {
                preconditioned.emplace_back(m_ops->makeVecLHS());
                m_ops->precond(preconditioned.back(), basis[column]);
                basis.emplace_back(m_ops->makeVecRHS());
                auto& next = basis.back();
                m_ops->apply(next, preconditioned.back());
                RT const action_norm = m_ops->norm2(next);
                if (!std::isfinite(action_norm)) { m_status = 2; return; }
                // Two-pass modified Gram-Schmidt keeps the small least-squares
                // residual reliable when the preconditioner changes direction.
                for (int pass = 0; pass < 2; ++pass) {
                    for (int row = 0; row <= column; ++row) {
                        RT const coefficient = m_ops->dotProduct(next, basis[row]);
                        H(row, column) += coefficient;
                        m_ops->increment(next, basis[row], -coefficient);
                    }
                }
                RT const remainder = m_ops->norm2(next);
                if (!std::isfinite(remainder)) { m_status = 2; return; }
                H(column + 1, column) = remainder;
                breakdown = remainder <= std::numeric_limits<RT>::epsilon() * action_norm;
                if (!breakdown) { m_ops->scale(next, RT(1) / remainder); }

                for (int row = 0; row < column; ++row) {
                    RT const upper = H(row, column);
                    RT const lower = H(row + 1, column);
                    H(row, column) = cosine[row] * upper + sine[row] * lower;
                    H(row + 1, column) = -sine[row] * upper + cosine[row] * lower;
                }
                RT const diagonal = std::hypot(H(column, column), H(column + 1, column));
                cosine[column] = diagonal == RT(0) ? RT(1) : H(column, column) / diagonal;
                sine[column] = diagonal == RT(0) ? RT(0) : H(column + 1, column) / diagonal;
                H(column, column) = diagonal;
                H(column + 1, column) = RT(0);
                g[column + 1] = -sine[column] * g[column];
                g[column] *= cosine[column];
                ++columns;
                ++m_iterations;
                if (m_verbose > 1) {
                    amrex::Print() << "FGMRES: iter = " << m_iterations
                                   << ", estimated residual = " << std::abs(g[column + 1]) << '\n';
                }
                if (breakdown || std::abs(g[column + 1]) <= target) { break; }
            }

            std::vector<RT> coefficients(columns, RT(0));
            for (int row = columns - 1; row >= 0; --row) {
                RT value = g[row];
                for (int col = row + 1; col < columns; ++col) {
                    value -= H(row, col) * coefficients[col];
                }
                if (H(row, row) != RT(0)) { coefficients[row] = value / H(row, row); }
            }
            for (int col = 0; col < columns; ++col) {
                m_ops->increment(solution, preconditioned[col], coefficients[col]);
            }
            m_ops->apply(action, solution);
            m_ops->linComb(residual, RT(1), rhs, RT(-1), action);
            m_residual = m_ops->norm2(residual);
            if (m_verbose > 0) {
                amrex::Print() << "FGMRES: iter = " << m_iterations
                               << ", true residual = " << m_residual << '\n';
            }
            if (!std::isfinite(m_residual)) { m_status = 2; return; }
            if (m_residual <= target) { m_status = 0; return; }
            if (breakdown) { m_status = 2; return; }
        }
    }

private:
    Ops* m_ops = nullptr;
    int m_verbose = 0;
    int m_restart = 30;
    int m_max_iterations = 1000;
    int m_iterations = 0;
    int m_status = -1; // 0 converged, 1 iteration limit, 2 breakdown/nonfinite action.
    RT m_residual = std::numeric_limits<RT>::max();
};

#endif
