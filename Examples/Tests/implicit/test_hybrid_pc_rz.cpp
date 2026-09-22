/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "NonlinearSolvers/HybridPICRZOperator.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <iomanip>

namespace {
using Real = amrex::Real;
using Op = HybridPICRZOperator;
using Field = Op::Field;
constexpr Real pi = 3.14159265358979323846;

// An independently evaluated grid function, including physical correction
// extensions. No production sampling, curl or boundary helper is used here.
struct Reference {
    int nr = 16, nz = 32, pec = 1, held = 0, periodic = 0, pin_a = 0,
        identity = 0, weighted = 0;
    Real dr = 1. / 16., dz = 2. / 32., hall = .002, eta = .0003, hyper = 2.e-7;
    AMREX_GPU_HOST_DEVICE bool
    row (int c, int i, int j) const {
        return (c == 1 && i == 0) ||
               (pec && ((c != 0 && i == nr) ||
                        (!periodic && c != 2 && (j == 0 || j == nz))));
    }
    AMREX_GPU_HOST_DEVICE Real
    raw (int c, int i, int j, int w) const {
        Real const r = (i + (c == 0 ? .5 : 0.)) * dr,
                   z = (j + (c == 2 ? .5 : 0.)) * dz;
        Real const radial = c == 2 ? 1. - .3 * r * r : r * (1. - .2 * r * r);
        return (c + 1) * radial *
                   (std::cos(pi * z) +
                    (w ? .17 : .09) * std::sin(3. * pi * z)) +
               (w ? .13 : .21) * radial * std::cos(5. * pi * r) *
                   std::cos(2. * pi * z);
    }
    AMREX_GPU_HOST_DEVICE Real
    cf (int n, int c, int i, int j) const {
        Real const r = (i + (c == 0 ? .5 : 0.)) * dr,
                   z = (j + (c == 2 ? .5 : 0.)) * dz;
        Real const f = 1. + .1 * std::cos(2. * pi * z) + .15 * r;
        if (n == Op::RowWeight) {
            return weighted && c == 1 && i < nr / 2 ? 0. : 1.;
        }
        if (n == Op::Scale) {
            return .006 * (1. + .5 * r + .2 * std::sin(pi * z));
        }
        if (identity) {
            return 0.;
        }
        if (n == Op::Hall) {
            return hall * f;
        }
        if (n == Op::Eta) {
            return eta * f;
        }
        if (n == Op::Hyper) {
            return hyper * f;
        }
        if (n == Op::InvRho) {
            return .15 * f;
        }
        return .02 * f;
    }
    AMREX_GPU_HOST_DEVICE Real
    node (int n, int i, int j) const {
        Real const r = i * dr, z = j * dz;
        if (identity) {
            return 0.;
        }
        if (n == 0) {
            return .2 * r;
        }
        if (n == 1) {
            return .1 * r;
        }
        if (n == 2) {
            return 1. + .1 * std::cos(pi * z);
        }
        if (n == 3) {
            return .003 * r;
        }
        if (n == 4) {
            return .002 * r;
        }
        if (n == 5) {
            return .003 * std::cos(pi * z);
        }
        return .0004 * (1. + .1 * r);
    }
    // kind 0: E, 1: physical W, 2: local ion response, 3: curl source.
    AMREX_GPU_HOST_DEVICE Real
    value (int c, int i, int j, int kind) const {
        int const ir = c == 0 ? nr - 1 : nr, iz = c == 2 ? nz - 1 : nz;
        if (kind == 3 && pin_a &&
            (i >= ir || (!periodic && (j <= 0 || j >= iz)))) {
            return 0.;
        }
        Real sign = 1.;
        if (i < 0) {
            i = c == 0 ? -i - 1 : -i;
            if (c != 2) {
                sign = -sign;
            }
        }
        if (i > ir) {
            if (held) {
                return 0.;
            }
            i = 2 * ir - i + (c == 0 ? 1 : 0);
            if (pec ? c != 0 : c == 0) {
                sign = -sign;
            }
        }
        if (periodic) {
            j = (j % nz + nz) % nz;
        } else if (j < 0 || j > iz) {
            if (held) {
                return 0.;
            }
            j = j < 0 ? -j - (c == 2 ? 1 : 0) : 2 * iz - j + (c == 2 ? 1 : 0);
            if (pec ? c != 2 : c == 2) {
                sign = -sign;
            }
        }
        if (row(c, i, j)) {
            return 0.;
        }
        if (kind == 2) {
            return sign * cf(Op::Ion, c, i, j) *
                   (raw(c, i, j, 0) + cf(Op::Eta, c, i, j) * raw(c, i, j, 1) -
                    cf(Op::Hyper, c, i, j) * lap(c, i, j));
        }
        return sign * raw(c, i, j, kind == 1 ? 1 : 0);
    }
    AMREX_GPU_HOST_DEVICE Real
    lap (int c, int i, int j) const {
        Real const r = (i + (c == 0 ? .5 : 0.)) * dr, u = value(c, i, j, 1);
        Real v = (value(c, i, j + 1, 1) + value(c, i, j - 1, 1) - 2. * u) /
                 (dz * dz);
        if (r > 0.) {
            v += ((r + .5 * dr) * value(c, i + 1, j, 1) +
                  (r - .5 * dr) * value(c, i - 1, j, 1) - 2. * r * u) /
                 (r * dr * dr);
            if (c != 2) {
                v -= u / (r * r);
            }
        } else if (c == 2) {
            v += 2. * (value(c, -1, j, 1) + value(c, 1, j, 1) - 2. * u) /
                 (dr * dr);
        } else {
            v = 0.;
        }
        return v;
    }
    AMREX_GPU_HOST_DEVICE Real
    curl (int c, int i, int j) const {
        if (c == 0) {
            return i == 0 ? 0.
                          : -(value(1, i, j + 1, 3) - value(1, i, j, 3)) / dz;
        }
        if (c == 1) {
            return (value(0, i, j + 1, 3) - value(0, i, j, 3)) / dz -
                   (value(2, i + 1, j, 3) - value(2, i, j, 3)) / dr;
        }
        return ((i + 1.) * value(1, i + 1, j, 3) - i * value(1, i, j, 3)) /
               ((i + .5) * dr);
    }
    AMREX_GPU_HOST_DEVICE Real
    curlcurl (int c, int i, int j) const {
        if (c == 0) {
            return -(curl(1, i, j) - curl(1, i, j - 1)) / dz;
        }
        if (c == 1) {
            return i == 0 ? 0.
                          : (curl(0, i, j) - curl(0, i, j - 1)) / dz -
                                (curl(2, i, j) - curl(2, i - 1, j)) / dr;
        }
        return i == 0
                   ? 4. * curl(1, 0, j) / dr
                   : ((i + .5) * curl(1, i, j) - (i - .5) * curl(1, i - 1, j)) /
                         (i * dr);
    }
    AMREX_GPU_HOST_DEVICE Real
    nodal_value (int c, int i, int j, int kind) const {
        if (c == 0) {
            return .5 * (value(c, i - 1, j, kind) + value(c, i, j, kind));
        }
        if (c == 2) {
            return .5 * (value(c, i, j - 1, kind) + value(c, i, j, kind));
        }
        return value(c, i, j, kind);
    }
    AMREX_GPU_HOST_DEVICE Real
    nodal_curl (int c, int i, int j) const {
        if (c == 0) {
            return .5 * (curl(c, i, j - 1) + curl(c, i, j));
        }
        if (c == 2) {
            return .5 * (curl(c, i - 1, j) + curl(c, i, j));
        }
        return .25 * (curl(c, i - 1, j - 1) + curl(c, i, j - 1) +
                      curl(c, i - 1, j) + curl(c, i, j));
    }
    AMREX_GPU_HOST_DEVICE Real
    cross (int c, int i, int j, int kind) const {
        int const a = (c + 1) % 3, b = (c + 2) % 3;
        if (kind == 0) {
            return nodal_value(a, i, j, 1) * node(b, i, j) -
                   nodal_value(b, i, j, 1) * node(a, i, j);
        }
        if (kind == 1) {
            return nodal_value(c, i, j, 1) * node(6, i, j);
        }
        if (kind == 2) {
            return node(a + 3, i, j) * nodal_curl(b, i, j) -
                   node(b + 3, i, j) * nodal_curl(a, i, j);
        }
        return nodal_value(a, i, j, 2) * node(b, i, j) -
               nodal_value(b, i, j, 2) * node(a, i, j);
    }
    AMREX_GPU_HOST_DEVICE Real
    edge_cross (int c, int i, int j, int kind) const {
        if (c == 0) {
            return .5 * (cross(c, i, j, kind) + cross(c, i + 1, j, kind));
        }
        if (c == 2) {
            return .5 * (cross(c, i, j, kind) + cross(c, i, j + 1, kind));
        }
        return cross(c, i, j, kind);
    }
    AMREX_GPU_HOST_DEVICE Real
    action (int c, int i, int j, int n) const {
        if (row(c, i, j)) {
            return n ? cf(Op::Scale, c, i, j) * raw(c, i, j, 1)
                     : raw(c, i, j, 0);
        }
        if (n) {
            return cf(Op::Scale, c, i, j) *
                   (raw(c, i, j, 1) - curlcurl(c, i, j));
        }
        return raw(c, i, j, 0) +
               cf(Op::RowWeight, c, i, j) *
                   (cf(Op::Hall, c, i, j) * edge_cross(c, i, j, 0) +
                    edge_cross(c, i, j, 1) +
                    cf(Op::Eta, c, i, j) * raw(c, i, j, 1) -
                    cf(Op::Hyper, c, i, j) * lap(c, i, j) +
                    cf(Op::InvRho, c, i, j) *
                        (edge_cross(c, i, j, 3) - edge_cross(c, i, j, 2)));
    }
};

void
coefficients (Op& op, Reference const& ref) {
    auto& l = *op.levels[0];
    for (amrex::MFIter mfi(*l.nodes); mfi.isValid(); ++mfi) {
        auto a = l.nodes->array(mfi);
        amrex::ParallelFor(mfi.validbox(), Op::NN,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                               a(i, j, k, n) = ref.node(n, i, j);
                           });
    }
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*l.coeff[c]); mfi.isValid(); ++mfi) {
            auto a = l.coeff[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), Op::NC,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    a(i, j, k, n) = ref.cf(n, c, i, j);
                });
        }
    }
    op.ion_response = !ref.identity;
    op.PrepareCoefficients();
}

void
stencil (Op& op, Reference const& ref) {
    auto x = op.makeVecRHS(), actual = op.makeVecRHS(),
         expected = op.makeVecRHS(), error = op.makeVecRHS();
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*x[c]); mfi.isValid(); ++mfi) {
            auto a = x[c]->array(mfi), e = expected[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), 2,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    a(i, j, k, n) = ref.raw(c, i, j, n) *
                                    (n ? ref.cf(Op::Scale, c, i, j) : 1.);
                    e(i, j, k, n) = ref.action(c, i, j, n);
                });
        }
    }
    op.apply(actual, x);
    op.linComb(error, 1., actual, -1., expected);
    Real const relative = op.norm2(error) / op.norm2(expected);
    amrex::Print() << "HALL_RZ stencil pec=" << ref.pec << " held=" << ref.held
                   << " periodic=" << ref.periodic << " A_pins=" << ref.pin_a
                   << " error=" << relative << '\n';
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        relative < 2.e-13, "independent primal-pair stencil/BC mismatch");
    // Scale is only a change of variables: spatially changing it cannot
    // change the physical E equation, including Lap(W_hat / scale).
    auto& l = *op.levels[0];
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*x[c]); mfi.isValid(); ++mfi) {
            auto a = x[c]->array(mfi), cf = l.coeff[c]->array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   Real const factor = 1.1 + .03 * i + .017 * j;
                                   a(i, j, k, 1) *= factor;
                                   cf(i, j, k, Op::Scale) *= factor;
                               });
        }
    }
    op.PrepareCoefficients();
    op.apply(actual, x);
    for (int c = 0; c < 3; ++c) {
        amrex::MultiFab::Subtract(*actual[c], *expected[c], 0, 0, 1, 0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            actual[c]->norminf(0) < 2.e-12,
            "variable auxiliary scale changed E action");
    }
    coefficients(op, ref);
    // A homogeneous fixed-cycle PC must be linear, also on identity rows.
    auto a = op.makeVecRHS(), b = op.makeVecRHS(), sum = op.makeVecRHS(),
         sum_pc = op.makeVecRHS();
    op.assign(a, x);
    op.scale(a, .37);
    op.assign(b, expected);
    op.scale(b, .21);
    op.linComb(sum, 1., a, 1., b);
    op.precond(sum_pc, sum);
    op.precond(actual, a);
    op.precond(error, b);
    op.increment(actual, error, 1.);
    op.increment(actual, sum_pc, -1.);
    Real const linear = op.norm2(actual) / op.norm2(sum_pc);
    amrex::Print() << "HALL_RZ fixed_cycle_linearity=" << linear << '\n';
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(linear) && linear < 2.e-12,
                                     "fixed MG cycle is not linear");
}

void
inverse (Op& op, Reference const& ref) {
    auto exact = op.makeVecRHS(), rhs = op.makeVecRHS(), out = op.makeVecRHS(),
         action = op.makeVecRHS();
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*exact[c]); mfi.isValid(); ++mfi) {
            auto a = exact[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // Nonzero boundary values exercise identity rows; high
                    // modes expose averaging away the cell-centered
                    // checkerboard.
                    a(i, j, k) = ref.raw(c, i, j, 0) +
                                 .03 * std::cos(pi * i) * std::cos(pi * j);
                });
        }
    }
    op.ApplyField(rhs, exact);
    op.Solve(out, rhs, 63, 1.e-10, 1);
    op.ApplyField(action, out);
    op.linComb(action, 1., action, -1., rhs);
    Real const residual = op.norm2(action) / op.norm2(rhs);
    op.increment(out, exact, -1.);
    Real const error = op.norm2(out) / op.norm2(exact);
    amrex::Print() << "HALL_RZ inverse identity=" << ref.identity
                   << " iterations=" << op.last_iterations
                   << " pair_residual=" << op.last_residual
                   << " field_residual=" << residual
                   << " solution_error=" << error << '\n';
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(residual < 3.e-10 && error < 3.e-9,
                                     "manufactured field inverse failed");
}
// Exercise this approximation in its intended role: a variable inner PC
// inside an outer field solve. A tight 63-step solve of the scaled pair is
// not required for the outer field residual to converge at high Hall CFL.
struct FieldOps {
    using RT = Real;
    using Pair = Op::Pair;
    Op& op;
    bool use_pc;
    Pair
    makeVecRHS () const {
        return op.makeVecRHS();
    }
    Pair
    makeVecLHS () const {
        return op.makeVecLHS();
    }
    void
    apply (Pair& y, Pair const& x) {
        op.ApplyField(y, x);
    }
    void
    precond (Pair& y, Pair const& x) {
        if (use_pc) {
            op.Solve(y, x, 8, .3, 1);
        } else {
            op.assign(y, x);
        }
    }
    void
    assign (Pair& y, Pair const& x) {
        op.assign(y, x);
    }
    void
    setToZero (Pair& x) {
        op.setToZero(x);
    }
    void
    scale (Pair& x, RT a) {
        op.scale(x, a);
    }
    void
    increment (Pair& x, Pair const& y, RT a) {
        op.increment(x, y, a);
    }
    void
    linComb (Pair& z, RT a, Pair const& x, RT b, Pair const& y) {
        op.linComb(z, a, x, b, y);
    }
    RT
    dotProduct (Pair const& x, Pair const& y) {
        return op.dotProduct(x, y);
    }
    RT
    norm2 (Pair const& x) {
        return op.norm2(x);
    }
};

void
outer_inverse (Op& op, Reference const& ref) {
    auto exact = op.makeVecRHS(), rhs = op.makeVecRHS(), out = op.makeVecRHS(),
         check = op.makeVecRHS();
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*exact[c]); mfi.isValid(); ++mfi) {
            auto a = exact[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    a(i, j, k) = ref.raw(c, i, j, 0) +
                                 .03 * std::cos(pi * i) * std::cos(pi * j);
                });
        }
    }
    op.ApplyField(rhs, exact);
    int unpreconditioned_iterations = 0;
    for (bool pc : {false, true}) {
        FieldOps f{op, pc};
        FlexibleGMRES<Op::Pair, FieldOps> solver;
        solver.define(f);
        solver.setRestartLength(40);
        solver.solve(out, rhs, 1.e-8, 0., 300);
        op.ApplyField(check, out);
        op.increment(check, rhs, -1.);
        Real const residual = op.norm2(check) / op.norm2(rhs);
        op.increment(out, exact, -1.);
        Real const error = op.norm2(out) / op.norm2(exact);
        amrex::Print() << "HALL_RZ outer pc=" << pc
                       << " iterations=" << solver.getNumIters()
                       << " residual=" << residual << " error=" << error
                       << '\n';
        if (!pc) {
            unpreconditioned_iterations = solver.getNumIters();
        } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                solver.getStatus() == 0 && residual < 3.e-8 && error < 3.e-7 &&
                    solver.getNumIters() < unpreconditioned_iterations,
                "spatial Hall PC did not accelerate the stiff outer field "
                "solve");
        }
    }
}

} // namespace

int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        int grid = 8, solve = 1, outer = 0;
        amrex::ParmParse pp("hall_test");
        pp.query("grid", grid);
        pp.query("solve", solve);
        pp.query("outer", outer);
        for (int bc = 0; bc < (outer ? 1 : 4); ++bc) {
            Reference ref;
            pp.query("hall", ref.hall);
            ref.pec = bc == 1 || bc == 2;
            ref.held = bc == 3;
            ref.periodic = bc == 2;
            amrex::Box domain(amrex::IntVect(0, 0),
                              amrex::IntVect(ref.nr - 1, ref.nz - 1));
            amrex::RealBox physical({0., 0.}, {1., 2.});
            int periodic[2] = {0, ref.periodic};
            amrex::Geometry geom(domain, &physical, 1, periodic);
            amrex::BoxArray cells(domain);
            cells.maxSize(grid);
            amrex::DistributionMapping dm(cells);
            Op op;
            auto const wall = ref.held ? FieldBoundaryType::None
                                       : (ref.pec ? FieldBoundaryType::PEC
                                                  : FieldBoundaryType::Neumann);
            Op::BC lo = {FieldBoundaryType::None,
                         ref.periodic ? FieldBoundaryType::Periodic : wall};
            Op::BC hi = {wall,
                         ref.periodic ? FieldBoundaryType::Periodic : wall};
            op.Define(geom, cells, dm, lo, hi);
            pp.query("sigma", op.sigma);
            pp.query("sigma_w", op.sigma_w);
            pp.query("chi", op.whistler_defect);
            if (outer) {
                coefficients(op, ref);
                outer_inverse(op, ref);
                continue;
            }
            for (int pins = 0; pins <= 1; ++pins) {
                ref.pin_a = pins;
                ref.weighted = pins;
                op.pin_vector_potential = pins;
                coefficients(op, ref);
                stencil(op, ref);
                if (solve) {
                    inverse(op, ref);
                }
            }
            ref.identity = 1;
            op.pin_vector_potential = false;
            coefficients(op, ref);
            if (solve) {
                inverse(op, ref);
            }
        }
        amrex::Print() << "HALL_RZ PASS\n";
    }
    amrex::Finalize();
}
