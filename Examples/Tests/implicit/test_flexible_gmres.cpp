/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL.
 */
#include "NonlinearSolvers/FlexibleGMRES.H"

#include <AMReX.H>
#include <AMReX_GMRES.H>

#include <cmath>
#include <limits>
#include <vector>

namespace {
struct TestVector { std::vector<double> values; };

struct TestOperator
{
    using RT = double;
    std::vector<std::vector<double>> matrix;
    bool variable_pc = false;
    bool nonfinite_pc = false;
    int pc_calls = 0;
    bool directional_pc = false;

    TestVector makeVecRHS () const { return {std::vector<double>(matrix.size(), 0.)}; }
    TestVector makeVecLHS () const { return makeVecRHS(); }
    void assign (TestVector& x, TestVector const& y) const { x = y; }
    void setToZero (TestVector& x) const { for (auto& v : x.values) { v = 0.; } }
    void scale (TestVector& x, double a) const { for (auto& v : x.values) { v *= a; } }
    void increment (TestVector& x, TestVector const& y, double a) const
    {
        for (std::size_t i = 0; i < x.values.size(); ++i) { x.values[i] += a * y.values[i]; }
    }
    void linComb (TestVector& z, double a, TestVector const& x,
                  double b, TestVector const& y) const
    {
        for (std::size_t i = 0; i < z.values.size(); ++i) {
            z.values[i] = a * x.values[i] + b * y.values[i];
        }
    }
    double dotProduct (TestVector const& x, TestVector const& y) const
    {
        double sum = 0.;
        for (std::size_t i = 0; i < x.values.size(); ++i) { sum += x.values[i] * y.values[i]; }
        return sum;
    }
    double norm2 (TestVector const& x) const { return std::sqrt(dotProduct(x, x)); }
    void apply (TestVector& y, TestVector const& x) const
    {
        setToZero(y);
        for (std::size_t i = 0; i < matrix.size(); ++i) {
            for (std::size_t j = 0; j < matrix.size(); ++j) {
                y.values[i] += matrix[i][j] * x.values[j];
            }
        }
    }
    void precond (TestVector& y, TestVector const& x)
    {
        ++pc_calls;
        assign(y, x);
        if (variable_pc) { scale(y, pc_calls % 2 == 1 ? 1. : 2.); }
        if (directional_pc) {
            double const norm = norm2(x);
            for (std::size_t i = 0; i < y.values.size(); ++i) {
                y.values[i] *= 1. + .1 * (pc_calls % 3) * (i + 1)
                    + .3 * std::abs(x.values[i]) / std::max(norm, 1.e-30);
            }
        }
        if (nonfinite_pc) { y.values[0] = std::numeric_limits<double>::quiet_NaN(); }
    }
    double trueResidual (TestVector const& x, TestVector const& b) const
    {
        auto r = makeVecRHS(); apply(r, x); increment(r, b, -1.); return norm2(r);
    }
};
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // An exact one-dimensional Krylov space exposes reconstruction with
        // a fresh, different PC action. Ordinary GMRES reports zero but x is wrong.
        TestOperator identity{{{1., 0.}, {0., 1.}}, true};
        TestVector rhs{{2., -3.}}, x = identity.makeVecLHS();
        amrex::GMRES<TestVector, TestOperator> ordinary;
        ordinary.define(identity); ordinary.solve(x, rhs, 1.e-12, 1.e-14);
        AMREX_ALWAYS_ASSERT(ordinary.getStatus() == 0);
        AMREX_ALWAYS_ASSERT(identity.trueResidual(x, rhs) > 1.);
        identity.pc_calls = 0;
        FlexibleGMRES<TestVector, TestOperator> flexible;
        flexible.define(identity); flexible.solve(x, rhs, 1.e-12, 1.e-14);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 0 && flexible.getNumIters() == 1);
        AMREX_ALWAYS_ASSERT(identity.trueResidual(x, rhs) < 1.e-12);
        AMREX_ALWAYS_ASSERT(identity.pc_calls == 1);

        // A nonsymmetric matrix requires several stored preconditioned directions.
        TestOperator coupled{{{4., 1., 0.}, {-1., 3., 1.}, {0., -2., 5.}}, true};
        TestVector exact{{1., -2., .5}}, b = coupled.makeVecRHS();
        coupled.apply(b, exact); x = coupled.makeVecLHS();
        flexible.define(coupled); flexible.setRestartLength(3);
        flexible.solve(x, b, 1.e-12, 1.e-14);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 0 && flexible.getNumIters() <= 3);
        AMREX_ALWAYS_ASSERT(coupled.trueResidual(x, b) < 1.e-11);
        auto error = x; coupled.increment(error, exact, -1.);
        AMREX_ALWAYS_ASSERT(coupled.norm2(error) < 1.e-11);
        coupled.variable_pc = false; coupled.directional_pc = true;
        flexible.solve(x, b, 1.e-12, 1.e-14);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 0);
        AMREX_ALWAYS_ASSERT(coupled.trueResidual(x, b) < 1.e-11);
        flexible.setRestartLength(2); flexible.setMaxIters(100);
        flexible.solve(x, b, 1.e-10, 1.e-13);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 0 && flexible.getNumIters() > 2);
        AMREX_ALWAYS_ASSERT(coupled.trueResidual(x, b) < 1.e-10 * coupled.norm2(b));

        // Per-solve iteration caps must be honored even within a long restart.
        flexible.setRestartLength(20); flexible.solve(x, b, 1.e-14, 1.e-14, 1);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 1 && flexible.getNumIters() == 1);
        AMREX_ALWAYS_ASSERT(std::abs(flexible.getResidualNorm() - coupled.trueResidual(x, b)) < 1.e-13);
        flexible.solve(x, b, 1.e-14, 1.e-14, 0);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 1 && flexible.getNumIters() == 0);
        coupled.setToZero(b); flexible.solve(x, b, 1.e-14, 0.);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() == 0 && flexible.getNumIters() == 0);

        // A singular inconsistent problem must not report a happy convergence.
        TestOperator singular{{{0., 0.}, {0., 0.}}};
        flexible.define(singular); x = singular.makeVecLHS();
        flexible.solve(x, rhs, 1.e-12, 1.e-14);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() != 0);
        AMREX_ALWAYS_ASSERT(flexible.getResidualNorm() == singular.norm2(rhs));
        identity.nonfinite_pc = true; flexible.define(identity);
        flexible.solve(x, rhs, 1.e-12, 1.e-14);
        AMREX_ALWAYS_ASSERT(flexible.getStatus() != 0);
        amrex::Print() << "Flexible GMRES variable-PC, restart, residual and breakdown checks passed\n";
    }
    amrex::Finalize();
}
