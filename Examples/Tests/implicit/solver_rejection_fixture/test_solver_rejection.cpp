/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NonlinearSolvers/AMReXGMRES_Wrapper.H"

#include <AMReX.H>
#include <AMReX_GMRES.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <limits>
#include <string>

namespace {
using Vec = amrex::MultiFab;
using RT = amrex::Real;

struct NativeOperator {
    using RT = amrex::Real;
    amrex::BoxArray boxes;
    amrex::DistributionMapping mapping;
    std::string failure;
    int applications = 0;
    int preconditioners = 0;
    bool reject = false;
    bool invalid_input = false;

    Vec
    makeVecRHS () const {
        return Vec(boxes, mapping, 1, 0);
    }
    Vec
    makeVecLHS () const {
        return makeVecRHS();
    }
    RT
    norm2 (Vec const& x) const {
        return x.norm2();
    }
    RT
    dotProduct (Vec const& x, Vec const& y) const {
        return Vec::Dot(x, 0, y, 0, 1, 0);
    }
    void
    assign (Vec& out, Vec const& in) const {
        Vec::Copy(out, in, 0, 0, 1, 0);
    }
    void
    setToZero (Vec& out) const {
        out.setVal(0);
    }
    void
    scale (Vec& out, RT a) const {
        out.mult(a);
    }
    void
    increment (Vec& out, Vec const& in, RT a) const {
        Vec::Saxpy(out, a, in, 0, 0, 1, 0);
    }
    void
    linComb (Vec& out, RT a, Vec const& x, RT b, Vec const& y) const {
        Vec::LinComb(out, a, x, 0, b, y, 0, 0, 1, 0);
    }
    void
    poisonOwner (Vec& out) const {
        // Only rank zero injects a failure. The global norm must stop every
        // rank, including a rank with no local boxes in the box64 case.
        if (amrex::ParallelDescriptor::MyProc() == 0) {
            out.setVal(std::numeric_limits<RT>::quiet_NaN());
        }
    }
    void
    apply (Vec& out, Vec const& in) {
        ++applications;
        invalid_input = invalid_input || !std::isfinite(norm2(in));
        bool const overflow = reject && failure == "overflow";
        for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
            auto const x = in.const_array(mfi);
            auto const y = out.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    RT const diagonal = RT(2) + RT(0.02) * (i + 3 * j + 5 * k);
                    y(i, j, k) = overflow ? std::numeric_limits<RT>::max() *
                                                (RT(2) + x(i, j, k))
                                          : diagonal * x(i, j, k);
                });
        }
        if (reject && failure == "action" && applications == 3) {
            poisonOwner(out);
        }
    }
    void
    precond (Vec& out, Vec const& in) {
        ++preconditioners;
        invalid_input = invalid_input || !std::isfinite(norm2(in));
        assign(out, in);
        if (reject && failure == "preconditioner") {
            poisonOwner(out);
        }
    }
    void
    reset () {
        applications = 0;
        preconditioners = 0;
        invalid_input = false;
    }
};

void
require (bool condition, char const* message) {
    int success = condition ? 1 : 0;
    amrex::ParallelDescriptor::ReduceIntMin(success);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(success, message);
}

void
run () {
    int box = 4;
    std::string failure = "action";
    amrex::ParmParse pp("test");
    pp.query("box", box);
    pp.query("failure", failure);
    amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0), amrex::IntVect(7)));
    boxes.maxSize(box);
    NativeOperator op{boxes, amrex::DistributionMapping(boxes), failure};
    Vec rhs = op.makeVecRHS();
    Vec solution = op.makeVecLHS();
    Vec baseline = op.makeVecLHS();
    Vec difference = op.makeVecLHS();
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto const f = rhs.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                f(i, j, k) = RT(1) + RT(0.01) * (2 * i + j + 3 * k);
            });
    }

    amrex::GMRES<Vec, NativeOperator> reference;
    reference.define(op);
    reference.setRestartLength(8);
    reference.setMaxIters(100);
    reference.solve(baseline, rhs, RT(1.e-12), RT(0));
    require(reference.getStatus() == 0, "native reference must converge");
    int const reference_iterations = reference.getNumIters();
    RT const reference_residual = reference.getResidualNorm();
    int const reference_applications = op.applications;
    int const reference_preconditioners = op.preconditioners;

    AMReXGMRES<Vec, NativeOperator> checked;
    checked.define(op);
    checked.setRestartLength(8);
    checked.setMaxIters(100);
    for (int pass = 0; pass < 3; ++pass) {
        op.reset();
        op.reject = pass == 1;
        if (op.reject && failure == "rhs") {
            op.poisonOwner(rhs);
        }
        checked.solve(solution, rhs, RT(1.e-12), RT(0));
        if (op.reject) {
            require(checked.failedEvaluation() && checked.getStatus() == 2,
                    "invalid linear evaluation must retain explicit failure "
                    "status");
            require(solution.norm2() == RT(0),
                    "rejected solve must not expose a partial correction");
            require(!op.invalid_input,
                    "no invalid vector may reach apply or precond");
            if (failure == "rhs" || failure == "preconditioner") {
                require(op.applications == 0,
                        "rejection must precede operator evaluation");
            } else if (failure == "action") {
                require(op.applications == 3 && checked.getNumIters() == 2,
                        "stop immediately after rejected action, before "
                        "another Arnoldi direction");
                require(std::isfinite(checked.getResidualNorm()),
                        "retain last finite residual");
            } else {
                require(op.applications == 1,
                        "overflow must stop on its first action");
            }
            amrex::Print() << "REJECTION failure=" << failure
                           << " status=" << checked.getStatus()
                           << " iterations=" << checked.getNumIters()
                           << " applications=" << op.applications
                           << " last_residual=" << checked.getResidualNorm()
                           << '\n';
            // Reconstruct the same analytic RHS for the second valid solve.
            for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const f = rhs.array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        f(i, j, k) = RT(1) + RT(0.01) * (2 * i + j + 3 * k);
                    });
            }
        } else {
            require(!checked.failedEvaluation() && checked.getStatus() == 0,
                    "finite solve must succeed before and after rejection");
            op.linComb(difference, RT(1), baseline, RT(-1), solution);
            require(difference.norm0() == RT(0),
                    "finite solution must equal native AMReX result exactly");
            require(checked.getNumIters() == reference_iterations &&
                        checked.getResidualNorm() == reference_residual &&
                        op.applications == reference_applications &&
                        op.preconditioners == reference_preconditioners,
                    "finite solver iteration path must remain identical");
        }
    }
    amrex::Print() << "PASS solver rejection and native A/B/A box=" << box
                   << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
}
} // namespace

int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    run();
    amrex::Finalize();
}
