/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionFDOperator.H"
#include "FieldSolver/ImplicitSolvers/ThermalConductionPC.H"
#include "FieldSolver/ImplicitSolvers/ThermalStageInitialGuess.H"
#include "FieldSolver/ImplicitSolvers/ThermalStageSolver.H"
#include "WarpX.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <string>
using namespace warpx::thermal;
using Real = amrex::Real;
using MF = amrex::MultiFab;
namespace {
constexpr Real n0 = 1.e18, gamma_e = 5. / 3., gm1 = gamma_e - 1.,
               ev_energy = n0 * PhysConst::q_e / gm1;
constexpr char name[] = "hybrid_electron_energy_fp";
void
require (bool value, const char* text) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(value, text);
}
void
near (Real error, Real tolerance, const char* text) {
    amrex::Print() << "CARTESIAN " << text << " error=" << error
                   << " tolerance=" << tolerance << "\n";
    require(std::isfinite(error) && std::abs(error) <= tolerance, text);
}
struct Context {
    WarpX sim;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    MF old, n, b, q;
    WarpXSolverVec u;
    Context (int box, int nr = 12, bool periodic = true,
             bool anisometric = false)
        : ba(amrex::Box(amrex::IntVect(0), amrex::IntVect(nr - 1))) {
        amrex::RealBox rb({-.25, .3, 1.1},
                          {anisometric ? .55 : .75, anisometric ? 1.5 : 1.3,
                           anisometric ? 2.6 : 2.1});
        int per[3] = {periodic, periodic, periodic};
        sim.geometry = amrex::Geometry(ba.minimalBox(), &rb, 0, per);
        ba.maxSize(box);
        dm = amrex::DistributionMapping(ba);
        old.define(ba, dm, 1, 1);
        n.define(ba, dm, 1, 1);
        b.define(ba, dm, 3, 1);
        q.define(ba, dm, 1, 1);
        old.setVal(2. * ev_energy);
        n.setVal(n0);
        q.setVal(0.);
        for (int d = 0; d < 3; ++d)
            b.setVal(Real(d + 1), d, 1, 1);
        sim.m_fields.alloc_init(name, 0, ba, dm, 1, amrex::IntVect(1), 0.);
        u.Define(&sim, "none", "none", {{name, ev_energy}});
    }
    MF
    scalar (int ng = 0) const {
        return MF(ba, dm, 1, ng);
    }
};
Real
difference (const MF& a, const MF& b, int ng = 0) {
    MF tmp(a.boxArray(), a.DistributionMap(), 1, ng);
    MF::LinComb(tmp, 1, a, 0, -1, b, 0, 0, 1, ng);
    return tmp.norm0(0, ng);
}
Real
integral (const MF& x, const Context& c) {
    auto dx = c.sim.geometry.CellSizeArray();
    return x.sum(0) * dx[0] * dx[1] * dx[2];
}
void
wave (Context& c, Real amplitude = .1) {
    auto const dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto u = c.old.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const phase =
                    2. * MathConst::pi *
                    ((i + .5) * dx[0] + (j + .5) * dx[1] + (k + .5) * dx[2]);
                u(i, j, k) = ev_energy * (2. + amplitude * std::sin(phase));
            });
    }
}
ThermalStageOptions
options (int order = 2) {
    ThermalStageOptions o;
    o.dt = .02;
    o.order = order;
    o.cross_mode = 0;
    o.temperature_floor_ev = .01;
    o.kappa_parallel = "0.000001";
    o.kappa_perpendicular = "0.000001";
    return o;
}
struct Faces {
    std::array<std::unique_ptr<MF>, 3> values;
    ThermalFaceContext view;
    Faces (Context const& c, amrex::GpuArray<Real, 3> speed) {
        for (int d = 0; d < 3; ++d) {
            values[d] = std::make_unique<MF>(
                amrex::convert(c.ba, amrex::IntVect::TheDimensionVector(d)),
                c.dm, 1, 0);
            values[d]->setVal(speed[d]);
            view.velocity[d] = values[d].get();
        }
    }
};
Real
eigenvalue (int nr, int order, Real kp, Real kt,
            amrex::GpuArray<Real, 3> field) {
    Real const h = 1. / nr, x = 2. * MathConst::pi * h;
    Real const diag =
        order == 2 ? 4. * std::pow(std::sin(.5 * x), 2) / (h * h)
                   : (37. / 18. - 2. * std::cos(x) - .1 * std::cos(2. * x) +
                      2. / 45. * std::cos(3. * x)) /
                         (h * h);
    Real const der = order == 2 ? std::sin(x) / h
                                : std::sin(x) * (4. - std::cos(x)) / (3. * h);
    Real b2 = 0;
    for (auto v : field)
        b2 += v * v;
    Real result = 0;
    for (int d = 0; d < 3; ++d) {
        result += (kt + (kp - kt) * field[d] * field[d] / b2) * diag;
        for (int t = 0; t < 3; ++t)
            if (t != d)
                result += (kp - kt) * field[d] * field[t] / b2 * der * der;
    }
    return gm1 / (n0 * PhysConst::kb) * result;
}
void
constant (Context& c) {
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto u = c.old.array(mfi), n = c.n.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                n(i, j, k) =
                    n0 *
                    (1. + .3 * std::sin(2. * MathConst::pi * (i + .5) * dx[0]) *
                              std::cos(2. * MathConst::pi * (k + .5) * dx[2]));
                u(i, j, k) = n(i, j, k) * 2. * PhysConst::q_e / gm1;
            });
    }
    for (int order : {2, 4})
        for (int cross : {0, 1}) {
            auto o = options(order);
            o.cross_mode = cross;
            o.kappa_parallel = "0.00001";
            o.kappa_perpendicular = "0.000002";
            o.conduction_theta = 1.;
            o.free_streaming_fraction = .03;
            MF r = c.scalar();
            EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, o);
            require(stage.Residual(c.old, r), "constant state evaluates");
            near(r.norm0() / ev_energy, 3.e-13,
                 "variable-density uniform-T null");
            for (int d = 0; d < 3; ++d)
                near(stage.FaceFlux(d).norm0(), 1.e-14,
                     "all directions constant-null heat flux");
        }
}
void
native_tensor (Context& c) {
    // Translate native nodal sample locations by half a cell. In the interior,
    // the native FD operator and the FV adapter then sample the same affine T.
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto u = c.old.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const t = 2. + .07 * (i + .5) * dx[0] -
                               .11 * (j + .5) * dx[1] + .19 * (k + .5) * dx[2];
                u(i, j, k) = ev_energy * t;
            });
    }
    auto nodes = amrex::convert(c.ba, amrex::IntVect::TheNodeVector());
    MF temp(nodes, c.dm, 1, 3),
        context(nodes, c.dm, ConductionState::b_ncomp, 3),
        tensor(nodes, c.dm, 6, 2), rhs(nodes, c.dm, 1, 0),
        cache(nodes, c.dm, 3, 1), budget(nodes, c.dm, 1, 1);
    context.setVal(0);
    context.setVal(1., ConductionState::b_open, 1, 3);
    context.setVal(n0, ConductionState::b_ne, 1, 3);
    for (int d = 0; d < 3; ++d)
        context.setVal(Real(d + 1) / std::sqrt(14.), ConductionState::b_bx + d,
                       1, 3);
    context.setVal(14., ConductionState::b_B2, 1, 3);
    for (amrex::MFIter mfi(temp); mfi.isValid(); ++mfi) {
        auto t = temp.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                t(i, j, k) = PhysConst::q_e / PhysConst::kb *
                             (2. + .07 * (i + .5) * dx[0] -
                              .11 * (j + .5) * dx[1] + .19 * (k + .5) * dx[2]);
            });
    }
    auto parallel = [] AMREX_GPU_HOST_DEVICE(Real, Real, Real) {
        return Real(1.e-5);
    };
    auto perp = [] AMREX_GPU_HOST_DEVICE(Real, Real, Real) {
        return Real(2.e-6);
    };
    ConductionFDGeometry g(c.sim.geometry);
    for (int order : {2, 4}) {
        auto o = options(order);
        o.kappa_parallel = "0.00001";
        o.kappa_perpendicular = "0.000002";
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, o);
        auto r = c.scalar();
        require(stage.Residual(c.old, r), "Cartesian affine tensor residual");
        ConductionFDOptions native;
        native.fourth_order = order == 4;
        native.limiter = 3;
        native.flux_budget = true;
        native.budget_dt = 1.e-30;
        BuildConductionTensor(temp, context, tensor, g, native, parallel, perp,
                              0.);
        EvaluateConductionFDRHS(temp, context, tensor, rhs, cache, budget, g,
                                native, false);
        for (int d = 0; d < 3; ++d) {
            MF error(stage.FaceFlux(d).boxArray(), c.dm, 1, 0),
                analytic(error.boxArray(), c.dm, 1, 0);
            amrex::GpuArray<Real, 3> gradient{.07, -.11, .19};
            Real projection = .07 - .22 + .57;
            Real expected =
                -PhysConst::q_e / PhysConst::kb *
                (2.e-6 * gradient[d] + 8.e-6 * (d + 1) * projection / 14.);
            int const nr = c.sim.geometry.Domain().length(0);
            for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                auto e = error.array(mfi), a = analytic.array(mfi);
                auto f = stage.FaceFlux(d).const_array(mfi),
                     fn = cache.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        bool const interior = i >= 3 && j >= 3 && k >= 3 &&
                                              i < nr - 3 && j < nr - 3 &&
                                              k < nr - 3;
                        e(i, j, k) = interior ? f(i, j, k) + fn(i - (d == 0),
                                                                j - (d == 1),
                                                                k - (d == 2), d)
                                              : 0.;
                        a(i, j, k) = interior ? f(i, j, k) - expected : 0.;
                    });
            }
            near(analytic.norm0(), 2.e-14,
                 "full Cartesian tensor analytic face");
            near(error.norm0(), 2.e-14, "actual native 3D tensor face oracle");
        }
    }
}
void
spatial (int box) {
    for (int order : {2, 4}) {
        Real previous = 0;
        for (int nr : {8, 16, 32}) {
            Context c(box, nr);
            wave(c);
            auto o = options(order);
            o.kappa_parallel = "0.000005";
            o.kappa_perpendicular = "0.000001";
            EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, o);
            auto r = c.scalar();
            require(stage.Residual(c.old, r), "spatial manufactured residual");
            Real const rate = gm1 / (n0 * PhysConst::kb) * 4. * MathConst::pi *
                              MathConst::pi * (3.e-6 + 4.e-6 * 36. / 14.);
            Real const h = o.theta * o.dt;
            auto err = c.scalar();
            for (amrex::MFIter mfi(err); mfi.isValid(); ++mfi) {
                auto e = err.array(mfi);
                auto u = c.old.const_array(mfi), f = r.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        e(i, j, k) = f(i, j, k) / h -
                                     rate * (u(i, j, k) - 2. * ev_energy);
                    });
            }
            Real const error = err.norm0() / ev_energy;
            amrex::Print() << "SPATIAL order=" << order << " n=" << nr
                           << " error=" << error
                           << " ratio=" << (previous > 0 ? previous / error : 0)
                           << "\n";
            if (previous > 0)
                require(previous / error > (order == 2 ? 3.1 : 12.),
                        "Cartesian spatial design order");
            previous = error;
            near(integral(r, c) / ev_energy, 1.e-13,
                 "periodic 3D physical energy telescopes");
        }
    }
}
void
temporal (int box) {
    for (Real theta : {.5, 1.}) {
        Real previous = 0;
        for (int steps : {1, 2, 4}) {
            Context c(box);
            wave(c);
            auto o = options(2);
            o.theta = theta;
            o.conduction_theta = theta;
            o.dt = .05 / steps;
            Real const rate = eigenvalue(12, 2, 1.e-6, 1.e-6, {1., 2., 3.});
            MF initial = c.scalar();
            MF::Copy(initial, c.old, 0, 0, 1, 0);
            MF endpoint = c.scalar();
            for (int step = 0; step < steps; ++step) {
                EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q,
                                           o);
                MF::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
                auto result = SolveThermalStage(stage, c.u, endpoint, name);
                require(result.status == ThermalSolveStatus::Converged,
                        "temporal manufactured solve");
                MF::Copy(c.old, endpoint, 0, 0, 1, 0);
            }
            auto err = c.scalar();
            Real const decay = std::exp(-rate * .05);
            for (amrex::MFIter mfi(err); mfi.isValid(); ++mfi) {
                auto e = err.array(mfi);
                auto u = c.old.const_array(mfi), v = initial.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        e(i, j, k) = u(i, j, k) -
                                     (2. * ev_energy +
                                      decay * (v(i, j, k) - 2. * ev_energy));
                    });
            }
            Real error = err.norm0() / ev_energy;
            amrex::Print() << "TEMPORAL theta=" << theta << " steps=" << steps
                           << " error=" << error
                           << " ratio=" << (previous > 0 ? previous / error : 0)
                           << "\n";
            if (previous > 0)
                require(previous / error > (theta == .5 ? 3.7 : 1.75),
                        "Cartesian temporal design order");
            previous = error;
        }
    }
}
void
root (Context& c) {
    wave(c);
    auto o = options(4);
    o.kappa_parallel = "0.000005";
    o.kappa_perpendicular = "0.000001";
    Faces faces(c, {.4, -.2, .1});
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b, c.q,
                               faces.view, o);
    auto endpoint = c.scalar(), reference = c.scalar();
    Real const h = o.theta * o.dt,
               rate = eigenvalue(12, 4, 5.e-6, 1.e-6, {1., 2., 3.});
    Real const omega = .3 * std::sin(2. * MathConst::pi / 12.) * 12.;
    Real const a = 1. + h * rate, b = h * omega;
    auto expected = c.scalar();
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
        auto u = expected.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real phi = 2. * MathConst::pi *
                       ((i + .5) * dx[0] + (j + .5) * dx[1] + (k + .5) * dx[2]);
            u(i, j, k) =
                ev_energy * (2. + .1 * (a * std::sin(phi) - b * std::cos(phi)) /
                                      (a * a + b * b));
        });
    }
    for (bool pc : {true, false}) {
        MF::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
        ThermalSolveOptions solve;
        solve.use_preconditioner = pc;
        auto result = SolveThermalStage(stage, c.u, endpoint, name, solve);
        require(result.status == ThermalSolveStatus::Converged,
                "3D transport/conduction root");
        near(difference(c.u.getMultiFabBlock(name, 0), expected) / ev_energy,
             3.e-9, "Cartesian analytic advective diffusive root");
        near((integral(endpoint, c) - integral(c.old, c)) / ev_energy, 3.e-11,
             "3D endpoint energy balance");
        if (pc)
            MF::Copy(reference, c.u.getMultiFabBlock(name, 0), 0, 0, 1, 0);
        else
            near(difference(reference, c.u.getMultiFabBlock(name, 0)) /
                     ev_energy,
                 3.e-9, "Cartesian PC on/off same root");
        amrex::Print() << "ROOT pc=" << pc
                       << " newton=" << result.newton_iterations
                       << " gmres=" << result.linear_iterations
                       << " residual=" << result.residual << "\n";
    }
}
void
nonlinear (Context& c) {
    wave(c, .3);
    auto o = options(4);
    o.cross_mode = 1;
    o.conduction_theta = 1.;
    o.kappa_parallel = "0.000001*Te^2";
    o.kappa_perpendicular = "0.0000001";
    o.free_streaming_fraction = 1.e-7;
    Faces faces(c, {.15, -.2, .1});
    MF mask = c.scalar();
    for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
        auto a = mask.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                a(i, j, k) = i > 3 && i < 8 && j < 6 && k > 8 ? 0. : 1.;
            });
    }
    faces.view.conduction_active = &mask;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b, c.q,
                               faces.view, o);
    auto endpoint = c.scalar(), reference = c.scalar(), before = c.scalar(),
         after = c.scalar(), other = c.scalar();
    require(stage.Residual(c.old, before), "3D nonlinear base");
    MF::LinComb(other, 1.001, c.old, 0, 0., c.old, 0, 0, 1, 0);
    require(stage.Residual(other, after), "3D nonlinear trial");
    other.setVal(-1.);
    require(!stage.Residual(other, after), "3D negative trial rejects");
    require(stage.Residual(c.old, after), "3D base restored after probes");
    near(difference(before, after), 0, "3D residual A B A purity");
    for (bool pc : {true, false}) {
        MF::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
        ThermalSolveOptions solve;
        solve.use_preconditioner = pc;
        auto result = SolveThermalStage(stage, c.u, endpoint, name, solve);
        require(result.status == ThermalSolveStatus::Converged,
                "3D nonlinear SMART cap root");
        near((integral(endpoint, c) - integral(c.old, c)) / ev_energy, 2.e-10,
             "3D nonlinear masked endpoint energy");
        if (pc)
            MF::Copy(reference, c.u.getMultiFabBlock(name, 0), 0, 0, 1, 0);
        else
            near(difference(reference, c.u.getMultiFabBlock(name, 0)) /
                     ev_energy,
                 4.e-9, "3D nonlinear PC on/off root");
        amrex::Print() << "NONLINEAR3D pc=" << pc
                       << " newton=" << result.newton_iterations
                       << " gmres=" << result.linear_iterations
                       << " residual=" << result.residual << "\n";
    }
}
void
contact (Context& c) {
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(c.n); mfi.isValid(); ++mfi) {
        auto n = c.n.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            n(i, j, k) =
                n0 *
                (1. + .15 * std::sin(2. * MathConst::pi * (i + .5) * dx[0]) +
                 .1 * std::cos(2. * MathConst::pi * (j + .5) * dx[1]) +
                 .08 * std::sin(2. * MathConst::pi * (k + .5) * dx[2]));
        });
    }
    c.n.FillBoundary(c.sim.geometry.periodicity());
    Faces faces(c, {0., 0., 0.});
    amrex::GpuArray<Real, 3> speeds{.2, -.1, .15};
    for (int d = 0; d < 3; ++d) {
        faces.view.velocity[d] = nullptr;
        faces.view.number_flux[d] = faces.values[d].get();
        for (amrex::MFIter mfi(*faces.values[d]); mfi.isValid(); ++mfi) {
            auto n = c.n.const_array(mfi);
            auto f = faces.values[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                f(i, j, k) =
                    speeds[d] * .5 *
                    (n(i, j, k) + n(i - (d == 0), j - (d == 1), k - (d == 2)));
            });
        }
    }
    auto o = options(4);
    auto nold = c.scalar(), nend = c.scalar(), us = c.scalar(), r = c.scalar();
    Real const h = o.theta * o.dt, theta = o.theta,
               e = 2. * PhysConst::q_e / gm1;
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto old = c.old.array(mfi), ns = nold.array(mfi), ne = nend.array(mfi),
             u = us.array(mfi);
        auto n = c.n.const_array(mfi);
        amrex::GpuArray<amrex::Array4<const Real>, 3> f;
        for (int d = 0; d < 3; ++d)
            f[d] = faces.values[d]->const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real div = 0;
                for (int d = 0; d < 3; ++d) {
                    int q[3] = {i, j, k};
                    ++q[d];
                    div += (f[d](q[0], q[1], q[2]) - f[d](i, j, k)) / dx[d];
                }
                ns(i, j, k) = n(i, j, k) + h * div;
                ne(i, j, k) = (n(i, j, k) - (1. - theta) * ns(i, j, k)) / theta;
                old(i, j, k) = e * ns(i, j, k);
                u(i, j, k) = e * n(i, j, k);
            });
    }
    EulerianThermalStage stage(c.sim.geometry, c.old, nold, c.n, nend, c.b, c.q,
                               faces.view, o);
    require(stage.Residual(us, r), "3D conservative-number-flux contact");
    near(r.norm0() / ev_energy, 2.e-14,
         "3D translating constant-temperature contact");
    for (int d = 0; d < 3; ++d) {
        MF err(stage.FaceAdvectionFlux(d).boxArray(), c.dm, 1, 0);
        MF::LinComb(err, 1, stage.FaceAdvectionFlux(d), 0, -e, *faces.values[d],
                    0, 0, 1, 0);
        near(err.norm0() / ev_energy, 1.e-15,
             "common face density contact identity");
    }
}
void
rows (Context& c) {
    wave(c);
    auto o = options(2);
    Faces faces(c, {.8, -.7, .4});
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b, c.q,
                               faces.view, o);
    ThermalConductionPC pc(stage);
    require(pc.Freeze(c.old), "3D frozen PC");
    auto v = c.scalar(), direct = c.scalar(), rows = c.scalar(),
         before = c.scalar(), inverse = c.scalar(), plus = c.scalar(),
         minus = c.scalar(), fp = c.scalar(), fm = c.scalar();
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
        auto out = v.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                out(i, j, k) =
                    ev_energy *
                    (.3 +
                     std::sin(2. * MathConst::pi * (i + .5) * dx[0]) *
                         std::cos(2. * MathConst::pi * (j + .5) * dx[1]) +
                     .2 * std::sin(2. * MathConst::pi * (k + .5) * dx[2]));
            });
    }
    pc.ApplyOperator(direct, v);
    pc.ApplyRows(rows, v);
    near(difference(direct, rows) / ev_energy, 2.e-14,
         "3D MLMG and signed transport rows");
    Real eps = 1.e-5;
    MF::LinComb(plus, 1, c.old, 0, eps, v, 0, 0, 1, 0);
    MF::LinComb(minus, 1, c.old, 0, -eps, v, 0, 0, 1, 0);
    require(stage.Residual(plus, fp) && stage.Residual(minus, fm),
            "3D direct directional residual");
    MF::LinComb(fp, .5 / eps, fp, 0, -.5 / eps, fm, 0, 0, 1, 0);
    near(difference(direct, fp) / ev_energy, 2.e-9,
         "3D actual residual Jv oracle");
    MF::Copy(before, direct, 0, 0, 1, 0);
    pc.Apply(inverse, v);
    c.n.setVal(3. * n0);
    c.b.setVal(5.);
    c.q.setVal(.25);
    for (auto& f : faces.values)
        f->mult(-3.);
    require(stage.RefreshContext(c.n, c.n, c.b, c.q, faces.view),
            "live 3D context mutation");
    pc.ApplyOperator(direct, v);
    near(difference(before, direct), 0,
         "frozen 3D action survives context refresh");
    pc.ApplyRows(rows, v);
    near(difference(before, rows) / ev_energy, 2.e-14,
         "frozen 3D rows survive context refresh");
    pc.Apply(plus, v);
    near(difference(inverse, plus), 0,
         "frozen 3D inverse survives context refresh");
}
void
boundary (Context& c) {
    auto o = options(2);
    o.dt = .001;
    Real surface = 0;
    for (int d = 0; d < 3; ++d)
        for (int side = 0; side < 2; ++side) {
            o.boundary[d][side].kind = BoundaryKind::PrescribedFlux;
            o.boundary[d][side].value = (1 + d + side) * 1.e-4;
            Real area = 1;
            for (int t = 0; t < 3; ++t)
                if (t != d)
                    area *= c.sim.geometry.ProbLength(t);
            surface += area * o.boundary[d][side].value;
        }
    auto r = c.scalar();
    EulerianThermalStage prescribed(c.sim.geometry, c.old, c.n, c.b, c.q, o);
    require(prescribed.Residual(c.old, r), "six physical Cartesian faces");
    near(integral(r, c) - o.theta * o.dt * surface, 1.e-16,
         "Cartesian area volume wall balance");
    for (int kind : {0, 1}) {
        for (int d = 0; d < 3; ++d)
            for (int side = 0; side < 2; ++side) {
                auto& bc = o.boundary[d][side];
                bc.kind = kind ? BoundaryKind::Leg : BoundaryKind::Reservoir;
                bc.value = .5;
                bc.leg_length = .3;
                bc.flux_limit = kind ? .05 : 0.;
            }
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, o);
        ThermalConductionPC pc(stage);
        require(pc.Freeze(c.old), "3D physical wall PC");
        auto dir = c.scalar(), a = c.scalar(), b = c.scalar(),
             plus = c.scalar(), minus = c.scalar(), fp = c.scalar(),
             fm = c.scalar();
        dir.setVal(ev_energy);
        pc.ApplyOperator(a, dir);
        pc.ApplyRows(b, dir);
        near(difference(a, b) / ev_energy, 2.e-14,
             "Cartesian wall row emission");
        Real eps = 1.e-5;
        MF::LinComb(plus, 1, c.old, 0, eps, dir, 0, 0, 1, 0);
        MF::LinComb(minus, 1, c.old, 0, -eps, dir, 0, 0, 1, 0);
        require(stage.Residual(plus, fp) && stage.Residual(minus, fm),
                "wall physical derivative probes");
        MF::LinComb(fp, .5 / eps, fp, 0, -.5 / eps, fm, 0, 0, 1, 0);
        near(difference(a, fp) / ev_energy, 3.e-9,
             "all-axis physical wall derivative");
    }
}
void
compression (Context& c) {
    auto o = options(2);
    o.kappa_parallel = "0";
    o.kappa_perpendicular = "0";
    o.dt = 1.;
    Real const div = -.3, ns_value = n0 / (1. + o.theta * o.dt * div),
               ne_value = (ns_value - (1 - o.theta) * n0) / o.theta,
               target = 2. * ev_energy / (1. + o.theta * o.dt * gamma_e * div);
    for (int d = 0; d < 3; ++d)
        for (int side = 0; side < 2; ++side)
            o.boundary[d][side].inflow_temperature_ev =
                target * gm1 / (ns_value * PhysConst::q_e);
    Faces faces(c, {0., 0., 0.});
    auto dx = c.sim.geometry.CellSizeArray();
    for (int d = 0; d < 3; ++d)
        for (amrex::MFIter mfi(*faces.values[d]); mfi.isValid(); ++mfi) {
            auto out = faces.values[d]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int p[3] = {i, j, k};
                    out(i, j, k) = div / 3. * (p[d] * dx[d] - .5);
                });
        }
    auto ns = c.scalar(), ne = c.scalar(), seed = c.scalar(),
         endpoint = c.scalar(), expected = c.scalar();
    ns.setVal(ns_value);
    ne.setVal(ne_value);
    expected.setVal(target);
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, o);
    require(stage.RefreshContext(ns, ne, c.b, c.q, faces.view),
            "3D compression trial density");
    ThermalStageGuessOptions guess;
    guess.temperature_floor_ev = o.temperature_floor_ev;
    require(MakeThermalStageInitialGuess(seed, c.old, c.old, ns, ne, guess),
            "3D seed");
    for (bool pc : {true, false}) {
        MF::Copy(c.u.getMultiFabBlock(name, 0), seed, 0, 0, 1, 0);
        ThermalSolveOptions solve;
        solve.use_preconditioner = pc;
        auto result = SolveThermalStage(stage, c.u, endpoint, name, solve);
        require(result.status == ThermalSolveStatus::Converged,
                "3D compressive solve");
        near(difference(c.u.getMultiFabBlock(name, 0), expected) / ev_energy,
             3.e-9, "3D analytic signed compression");
        amrex::Print() << "COMPRESSION3D pc=" << pc
                       << " newton=" << result.newton_iterations
                       << " gmres=" << result.linear_iterations << "\n";
    }
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse p("test");
        std::string test = "constant";
        int box = 6;
        p.query("case", test);
        p.query("box", box);
        if (test == "spatial")
            spatial(box);
        else if (test == "temporal")
            temporal(box);
        else {
            bool const periodic = test != "native_tensor" &&
                                  test != "boundary" && test != "compression";
            Context c(box, 12, periodic, test == "boundary");
            if (test == "constant")
                constant(c);
            else if (test == "native_tensor")
                native_tensor(c);
            else if (test == "root")
                root(c);
            else if (test == "nonlinear")
                nonlinear(c);
            else if (test == "contact")
                contact(c);
            else if (test == "rows")
                rows(c);
            else if (test == "boundary")
                boundary(c);
            else if (test == "compression")
                compression(c);
            else
                amrex::Abort("unknown Cartesian case");
        }
        amrex::Print() << "PASS thermal_cartesian " << test << "\n";
    }
    amrex::Finalize();
}
