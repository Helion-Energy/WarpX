/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/DarwinABoundary.H"
#include "FieldSolver/ImplicitSolvers/DarwinLongitudinalSchur.H"

#include <AMReX.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

using amrex::Real;
using namespace warpx::darwin;
using B = LongitudinalBoundary;
namespace {
struct Analytic {
    amrex::GpuArray<int, AMREX_SPACEDIM> lo{}, hi{}, per{}; // 0 PMC/axis, 1 PEC
    amrex::GpuArray<Real, AMREX_SPACEDIM> dx{};
    int n = 12, variable = 1;
    Real stiffness = 600.;
    AMREX_GPU_HOST_DEVICE Real
    Phi (int const* p, int mode = 0) const {
        constexpr Real pi = 3.14159265358979323846;
        Real value = 1.;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            Real x = Real(p[d]) / n;
            if ((lo[d] && p[d] == 0) || (hi[d] && p[d] == n)) {
                return 0.;
            }
            if (d == 0 && !per[d]) {
                value *= hi[d] ? std::pow(1 - x * x, mode + 2)
                               : std::cos((mode + 1) * pi * x);
            } else {
                value *= per[d] ? .3 + std::cos((mode + 1) * 2 * pi * x)
                                : (hi[d] ? std::sin((mode + 1) * pi * x)
                                         : std::cos((mode + 1) * pi * x));
            }
        }
        return value;
    }
    AMREX_GPU_HOST_DEVICE Real
    Kappa (int const* p) const {
        constexpr Real pi = 3.14159265358979323846;
        if (!variable) {
            return stiffness;
        }
        if (variable == 2 && p[0] > n / 3 && p[0] < 2 * n / 3) {
            return 0.;
        }
        Real x = Real(p[0]) / n, z = Real(p[1]) / n;
        // Density variation is reciprocal to kappa; variable=2 additionally
        // freezes an inactive interior band, independent of floor
        // representation.
        return stiffness * (1 + .3 * std::cos((per[0] ? 2 : 1) * pi * x)) *
               (1 + .2 * std::cos(2 * pi * z));
    }
    AMREX_GPU_HOST_DEVICE int
    Dim (int c) const {
#if defined(WARPX_DIM_RZ)
        return c == 1 ? -1 : c / 2;
#else
        return c;
#endif
    }
    AMREX_GPU_HOST_DEVICE Real
    Image (int* p, int c) const {
        int dnorm = Dim(c);
        Real sign = 1.;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            bool nodal = d != dnorm;
            int high = n - (nodal ? 0 : 1);
            if (per[d]) {
                if (p[d] < 0) {
                    p[d] += n;
                }
                if (p[d] > high) {
                    p[d] -= n;
                }
            } else if (p[d] < 0 || p[d] > high) {
                bool const lower = p[d] < 0;
#if defined(WARPX_DIM_RZ)
                bool const axis = d == 0 && lower;
#else
                bool const axis = false;
#endif
                if (axis || (lower ? lo[d] : hi[d])) {
                    return 0.;
                }
                p[d] = lower ? -(nodal ? 0 : 1) - p[d]
                             : 2 * high + (nodal ? 0 : 1) - p[d];
                if (!nodal) {
                    sign = -sign;
                }
            }
        }
        return sign;
    }
    AMREX_GPU_HOST_DEVICE Real
    G (int c, int i, int j, int k, int mode = 0) const {
        int d = Dim(c);
        if (d < 0) {
            return 0.;
        }
        int p[3] = {i, j, k};
        Real sign = Image(p, c);
        if (sign == 0) {
            return 0.;
        }
        Real low = Phi(p, mode);
        ++p[d];
        return sign * (Phi(p, mode) - low) / dx[d];
    }
    AMREX_GPU_HOST_DEVICE Real
    NG (int c, int* p, int mode = 0) const {
        int d = Dim(c);
        if (d < 0) {
            return 0.;
        }
        int q[3] = {p[0], p[1], p[2]};
        --q[d];
        return .5 *
               (G(c, p[0], p[1], p[2], mode) + G(c, q[0], q[1], q[2], mode)) *
               Kappa(p);
    }
    AMREX_GPU_HOST_DEVICE Real
    Flux (int c, int i, int j, int k, bool inertia = true, int mode = 0) const {
        int d = Dim(c);
        if (d < 0) {
            return 0.;
        }
        int p[3] = {i, j, k};
        Real sign = Image(p, c);
        if (sign == 0) {
            return 0.;
        }
        Real result = G(c, p[0], p[1], p[2], mode);
        if (inertia) {
            Real low = NG(c, p, mode);
            ++p[d];
            result += .5 * (low + NG(c, p, mode));
        }
        return sign * result;
    }
    AMREX_GPU_HOST_DEVICE Real
    Action (int i, int j, int k, bool inertia = true, int mode = 0) const {
        int const p[3] = {i, j, k};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if ((lo[d] && p[d] == 0) || (hi[d] && p[d] == n)) {
                return 0.;
            }
        }
        Real value = 0;
#if defined(WARPX_DIM_RZ)
        value = i == 0 ? 4 * Flux(0, i, j, k, inertia, mode) / dx[0]
                       : ((i + .5) * Flux(0, i, j, k, inertia, mode) -
                          (i - .5) * Flux(0, i - 1, j, k, inertia, mode)) /
                             (i * dx[0]);
        value += (Flux(2, i, j, k, inertia, mode) -
                  Flux(2, i, j - 1, k, inertia, mode)) /
                 dx[1];
#else
        for (int d = 0; d < 3; ++d) {
            int q[3] = {i, j, k};
            --q[d];
            value += (Flux(d, i, j, k, inertia, mode) -
                      Flux(d, q[0], q[1], q[2], inertia, mode)) /
                     dx[d];
        }
#endif
        return value;
    }
};
void
check (bool ok, const char* message) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok, message);
}
Real
difference (const amrex::MultiFab& a, const amrex::MultiFab& b,
            int ghosts = 0) {
    amrex::MultiFab error(a.boxArray(), a.DistributionMap(), 1, ghosts);
    amrex::MultiFab::LinComb(error, 1., a, 0, -1., b, 0, 0, 1, ghosts);
    return error.norminf(0, ghosts);
}
Real
potential_error (const amrex::MultiFab& a, const amrex::MultiFab& b,
                 bool singular, Real factor = 1.) {
    amrex::MultiFab error(a.boxArray(), a.DistributionMap(), 1, 0),
        one(a.boxArray(), a.DistributionMap(), 1, 0);
    amrex::MultiFab::LinComb(error, 1., a, 0, -factor, b, 0, 0, 1, 0);
    if (singular) {
        one.setVal(1);
        error.plus(-error.sum() / one.sum(), 0, 1, 0);
    }
    return error.norminf() / std::max(Real(1.e-30), factor * b.norminf());
}
} // namespace
int
main (int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        std::string wall = "pec", cap = "periodic";
        int box = 4, variable = 1, coord = 0, n = 12, all_periodic = 0;
        Real stiffness = 600.;
        pp.query("wall", wall);
        pp.query("cap", cap);
        pp.query("box", box);
        pp.query("variable", variable);
        pp.query("stiffness", stiffness);
        pp.query("coord", coord);
        pp.query("n", n);
        pp.query("all_periodic", all_periodic);
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(n - 1));
        amrex::RealBox physical(AMREX_D_DECL(0., 0., 0.),
                                AMREX_D_DECL(.25, 1., 1.));
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            periodic[d] = (d > 0 && cap == "periodic") || all_periodic;
        }
        amrex::Geometry geometry(domain, &physical, coord, periodic.data());
        amrex::BoxArray cells(domain);
        cells.maxSize(box);
        amrex::DistributionMapping dm(cells);
        auto nodes = amrex::convert(cells, amrex::IntVect(1));
        LongitudinalSchurOptions options;
        options.output_ghosts = amrex::IntVect(3);
        Analytic reference;
        reference.dx = geometry.CellSizeArray();
        reference.n = n;
        reference.variable = variable;
        reference.stiffness = stiffness;
        bool singular = all_periodic || (wall == "pmc" && cap != "pec");
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            options.lower[d] = periodic[d]    ? B::Periodic
                               : d == 0       ? B::PMC
                               : cap == "pec" ? B::PEC
                                              : B::PMC;
            options.upper[d] = periodic[d] ? B::Periodic
                               : d == 0    ? wall == "pec" ? B::PEC : B::PMC
                                           : options.lower[d];
            reference.lo[d] = options.lower[d] == B::PEC;
            reference.hi[d] = options.upper[d] == B::PEC;
            reference.per[d] = periodic[d];
        }
#if defined(WARPX_DIM_RZ)
        options.lower[0] = B::Axis;
#endif
        std::string rejected;
        pp.query("reject", rejected);
        if (rejected == "theta") {
            options.theta = .7;
        }
        if (rejected == "bdf") {
            options.bdf2 = true;
        }
        if (rejected == "advective") {
            options.djedt_only = false;
        }
        if (rejected == "eb") {
            options.embedded_boundaries = true;
        }
        if (rejected == "modes") {
            options.azimuthal_modes = 2;
        }
        DarwinLongitudinalSchur schur(geometry, cells, dm, options);
        amrex::MultiFab kappa(nodes, dm, 1, 3), phi(nodes, dm, 1, 3),
            phi2(nodes, dm, 1, 3), exact_rhs(nodes, dm, 1, 0);
        amrex::MultiFab input_kappa(nodes, dm, 1, 3),
            input_phi(nodes, dm, 1, 3), out(nodes, dm, 1, 0),
            diff(nodes, dm, 1, 0);
        kappa.setVal(-8.e9);
        phi.setVal(7.e9);
        phi2.setVal(-6.e9);
        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi), q = phi2.array(mfi),
                 kap = kappa.array(mfi), rhs = exact_rhs.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   int const index[3] = {i, j, k};
                                   p(i, j, k) = reference.Phi(index);
                                   q(i, j, k) = reference.Phi(index, 1);
                                   kap(i, j, k) = reference.Kappa(index);
                                   rhs(i, j, k) = reference.Action(i, j, k);
                               });
        }
        amrex::MultiFab::Copy(input_phi, phi, 0, 0, 1, 3);
        amrex::MultiFab::Copy(input_kappa, kappa, 0, 0, 1, 3);
        check(schur.Freeze(kappa), "Valid kappa rejected");
        schur.ApplyPotential(out, phi);
        if (singular) {
            auto owners = out.OwnerMask(geometry.periodicity());
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> op;
            amrex::ReduceData<Real, Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
                auto a = out.const_array(mfi);
                auto own = owners->const_array(mfi);
                op.eval(
                    mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        int const p[3] = {i, j, k};
                        Real weight = own(i, j, k);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            if (!reference.per[d] && (p[d] == 0 || p[d] == n)) {
                                weight *= .5;
                            }
                        }
#if defined(WARPX_DIM_RZ)
                        weight *= i == 0
                                      ? Real(.25)
                                      : Real(i) - (i == n ? Real(.5) : Real(0));
#endif
                        return {weight * a(i, j, k),
                                weight * std::abs(a(i, j, k))};
                    });
            }
            Real sum = amrex::get<0>(data.value()),
                 scale = amrex::get<1>(data.value());
            amrex::ParallelDescriptor::ReduceRealSum(sum);
            amrex::ParallelDescriptor::ReduceRealSum(scale);
            check(
                std::abs(sum) <= 2.e-13 * scale,
                "Native divergence lost its exact metric left-null quadrature");
        }
        Real operator_error = difference(out, exact_rhs) / exact_rhs.norminf();
        amrex::MultiFab::LinComb(diff, 1., out, 0, -1., exact_rhs, 0, 0, 1, 0);
        amrex::Print() << "operator_error=" << operator_error
                       << " max=" << diff.maxIndex(0)
                       << " min=" << diff.minIndex(0)
                       << " norms=" << out.norminf() << ","
                       << exact_rhs.norminf() << "\n";
        check(operator_error < 2.e-12, "Schur disagrees with independent "
                                       "native staggered/metric reference");
        amrex::MultiFab plus(nodes, dm, 1, 0), minus(nodes, dm, 1, 0),
            a_plus(nodes, dm, 1, 0), a_minus(nodes, dm, 1, 0),
            a_dir(nodes, dm, 1, 0);
        Real eps = 1.e-3;
        amrex::MultiFab::LinComb(plus, 1., phi, 0, eps, phi2, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(minus, 1., phi, 0, -eps, phi2, 0, 0, 1, 0);
        schur.ApplyPotential(a_plus, plus);
        schur.ApplyPotential(a_minus, minus);
        schur.ApplyPotential(a_dir, phi2);
        amrex::MultiFab::LinComb(diff, .5 / eps, a_plus, 0, -.5 / eps, a_minus,
                                 0, 0, 1, 0);
        Real directional_error = difference(diff, a_dir) / a_dir.norminf();
        check(directional_error < 2.e-10,
              "Manufactured Schur directional action mismatch");
        auto result = schur.SolvePotential(exact_rhs);
        check(result.converged,
              "Manufactured Schur solve failed true residual");
        Real solve_error =
            potential_error(schur.CorrectionPotential(), phi, singular);
        check(solve_error < 2.e-7, "Manufactured Schur potential mismatch");
        // Project the independently assembled frozen physical source with the
        // same native FD nodal Poisson class used by ComputeDarwinELong.
        amrex::MultiFab projected(nodes, dm, 1, 3),
            rhs_projection(nodes, dm, 1, 0);
        for (amrex::MFIter mfi(rhs_projection); mfi.isValid(); ++mfi) {
            auto r = rhs_projection.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    r(i, j, k) = .63 * reference.Action(i, j, k) +
                                 .37 * reference.Action(i, j, k, false);
                });
        }
        amrex::MLEBNodeFDLaplacian poisson({geometry}, {cells}, {dm},
                                           amrex::LPInfo{});
#if defined(WARPX_DIM_RZ)
        poisson.setRZ(true);
        poisson.setSigma({0., 1.});
#else
        poisson.setSigma({AMREX_D_DECL(1., 1., 1.)});
#endif
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> low, high;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            low[d] = reference.lo[d] ? amrex::LinOpBCType::Dirichlet
                     : periodic[d]   ? amrex::LinOpBCType::Periodic
                                     : amrex::LinOpBCType::Neumann;
            high[d] = reference.hi[d] ? amrex::LinOpBCType::Dirichlet
                      : periodic[d]   ? amrex::LinOpBCType::Periodic
                                      : amrex::LinOpBCType::Neumann;
        }
        poisson.setDomainBC(low, high);
        amrex::MLMG mg(poisson);
        mg.setVerbose(0);
        mg.setMaxIter(2000);
        projected.setVal(0);
        if (singular) {
            // MLEBNodeFDLaplacian declares isSingular=false even for this
            // constant-null problem. Use the explicit quotient Krylov solve,
            // then verify it with the independent production FD operator.
            auto projection_options = options;
            projection_options.relative_tolerance = 1.e-12;
            DarwinLongitudinalSchur projection(geometry, cells, dm,
                                               projection_options);
            amrex::MultiFab zero(nodes, dm, 1, 0);
            zero.setVal(0);
            check(projection.Freeze(zero), "Projection freeze");
            check(projection.SolvePotential(rhs_projection).converged,
                  "Constant-null projection solve");
            amrex::MultiFab::Copy(projected, projection.CorrectionPotential(),
                                  0, 0, 1, 0);
        } else {
            mg.solve({&projected}, {&rhs_projection}, 1.e-12, 0.);
        }
        poisson.prepareForSolve();
        poisson.apply(0, 0, out, projected, amrex::MLLinOp::BCMode::Homogeneous,
                      amrex::MLLinOp::StateMode::Solution);
        check(difference(out, rhs_projection) / rhs_projection.norminf() <
                  2.e-10,
              "Independent native Poisson action disagrees with reference "
              "projection");
        projected.FillBoundary(geometry.periodicity());
        std::array<amrex::MultiFab, 3> raw, held, saved_raw, saved_held;
        DarwinLongitudinalSchur::ConstVector raw_view{}, held_view{};
        auto const dx = geometry.CellSizeArray();
        for (int c = 0; c < 3; ++c) {
            int const d = reference.Dim(c);
            amrex::IntVect type(1);
            if (d >= 0) {
                type[d] = 0;
            }
            raw[c].define(amrex::convert(cells, type), dm, 1, 3);
            held[c].define(raw[c].boxArray(), dm, 1, 3);
            raw[c].setVal(991.);
            held[c].setVal(-885.);
            for (amrex::MFIter mfi(raw[c]); mfi.isValid(); ++mfi) {
                auto r = raw[c].array(mfi), h = held[c].array(mfi);
                auto p = projected.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    int q[3] = {i, j, k};
                    if (d >= 0) {
                        ++q[d];
                    }
                    r(i, j, k) =
                        d < 0 ? 0. : (p(q[0], q[1], q[2]) - p(i, j, k)) / dx[d];
                    h(i, j, k) = .37 * reference.G(c, i, j, k);
                });
            }
            saved_raw[c].define(raw[c].boxArray(), dm, 1, 3);
            saved_held[c].define(held[c].boxArray(), dm, 1, 3);
            amrex::MultiFab::Copy(saved_raw[c], raw[c], 0, 0, 1, 3);
            amrex::MultiFab::Copy(saved_held[c], held[c], 0, 0, 1, 3);
            raw_view[c] = &raw[c];
            held_view[c] = &held[c];
        }
        auto fixed = schur.Correct(raw_view, held_view);
        check(fixed.converged,
              "Frozen physical fixed-point correction solve failed");
        Real fixed_error =
            potential_error(schur.CorrectionPotential(), phi, singular, .63);
        check(fixed_error < 4.e-7,
              "Schur correction changes frozen physical fixed point");
        auto field = schur.CorrectionField();
        Real field_error = 0;
        for (int c = 0; c < 3; ++c) {
            amrex::MultiFab exact(field[c]->boxArray(), dm, 1, 3);
            for (amrex::MFIter mfi(exact); mfi.isValid(); ++mfi) {
                auto x = exact.array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        x(i, j, k) = .63 * reference.G(c, i, j, k);
                    });
            }
            field_error = std::max(field_error,
                                   difference(exact, *field[c], 3) /
                                       std::max(Real(1), exact.norminf(0, 3)));
            check(difference(raw[c], saved_raw[c], 3) == 0 &&
                      difference(held[c], saved_held[c], 3) == 0,
                  "Schur mutated raw/held E_L input or guards");
        }
        check(field_error < 4.e-7,
              "Correction field or physical/corner images mismatch");
        // A/B/A and poisoned unused input guards establish independence from
        // previous solves and caller ghost contents.
        auto copy = amrex::MultiFab(nodes, dm, 1, 3);
        amrex::MultiFab::Copy(copy, schur.CorrectionPotential(), 0, 0, 1, 3);
        check(schur.Freeze(kappa), "Second freeze failed");
        schur.SolvePotential(a_dir);
        auto again = schur.Correct(raw_view, held_view);
        check(again.converged, "A/B/A solve failed");
        Real purity = difference(copy, schur.CorrectionPotential(), 3);
        check(purity == 0., "A/B/A Schur output changed");
        check(difference(phi, input_phi, 3) == 0. &&
                  difference(kappa, input_kappa, 3) == 0.,
              "Schur mutated caller inputs");
        kappa.setVal(-1.);
        check(!schur.Freeze(kappa), "Negative kappa accepted");
        kappa.setVal(std::numeric_limits<Real>::quiet_NaN());
        check(!schur.Freeze(kappa), "Nonfinite kappa accepted");
        amrex::Print() << std::setprecision(13)
                       << "SCHUR_PASS geometry=" << AMREX_SPACEDIM
                       << " wall=" << wall << " cap=" << cap << " box=" << box
                       << " kappa=" << stiffness << " variable=" << variable
                       << " coord=" << coord
                       << " iterations=" << result.iterations
                       << " residual=" << result.residual
                       << " target=" << result.target
                       << " operator_error=" << operator_error
                       << " directional_error=" << directional_error
                       << " solve_error=" << solve_error
                       << " field_error=" << field_error
                       << " fixed_point_error=" << fixed_error
                       << " purity=" << purity << "\n";
    }
    amrex::Finalize();
}
