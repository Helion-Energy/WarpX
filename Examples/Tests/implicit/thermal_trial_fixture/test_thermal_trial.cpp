/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionFDOperator.H"
#include "FieldSolver/ImplicitSolvers/NativeConductionActivity.H"
#include "FieldSolver/ImplicitSolvers/ThermalStageInitialGuess.H"
#include "FieldSolver/ImplicitSolvers/ThermalStageSolver.H"
#include "WarpX.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <limits>
#include <string>

using namespace warpx::thermal;
using amrex::Real;
using MF = amrex::MultiFab;
namespace {
constexpr auto energy_name = "hybrid_electron_energy_fp";
constexpr Real density = 1.e18, gamma_e = 5. / 3.,
               unit_energy = density * PhysConst::q_e / (gamma_e - 1.);
void
require (bool b, const char* text) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(b, text);
}
void
near (Real error, Real tolerance, const char* text) {
    amrex::Print() << "TRIAL " << text << " error=" << error
                   << " tolerance=" << tolerance << "\n";
    require(std::isfinite(error) && std::abs(error) <= tolerance, text);
}
struct Context {
    WarpX sim;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    MF rho, ped, n, old, b, q;
    WarpXSolverVec u;
    Context (int box, int nr = 16)
        : ba(amrex::Box(amrex::IntVect(0),
                        amrex::IntVect(nr - 1, 2 * nr - 1))) {
        amrex::RealBox rb({0., 0.}, {1., 1.});
        int period[2] = {0, 1};
        sim.geometry = amrex::Geometry(ba.minimalBox(), &rb, 1, period);
        ba.maxSize(box);
        dm = amrex::DistributionMapping(ba);
        rho.define(amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm, 1,
                   1);
        ped.define(rho.boxArray(), dm, 1, 1);
        n.define(ba, dm, 1, 1);
        old.define(ba, dm, 1, 1);
        b.define(ba, dm, 3, 1);
        q.define(ba, dm, 1, 1);
        rho.setVal(PhysConst::q_e * density);
        ped.setVal(0);
        n.setVal(density);
        old.setVal(unit_energy);
        b.setVal(0);
        q.setVal(0);
        sim.m_fields.alloc_init(energy_name, 0, ba, dm, 1, amrex::IntVect(1),
                                0.);
        u.Define(&sim, "none", "none", {{energy_name, unit_energy}});
        u.getMultiFabBlock(energy_name, 0).setVal(unit_energy);
    }
    MF
    scalar (int ng = 0) const {
        return MF(ba, dm, 1, ng);
    }
};
Real
difference (const MF& a, const MF& b, int ng = 0) {
    MF t(a.boxArray(), a.DistributionMap(), 1, ng);
    MF::LinComb(t, 1., a, 0, -1., b, 0, 0, 1, ng);
    return t.norm0(0, ng);
}
Real
integral (const MF& a, const amrex::Geometry& geom) {
    MF t(a.boxArray(), a.DistributionMap(), 1, 0);
    auto dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(t); mfi.isValid(); ++mfi) {
        auto out = t.array(mfi);
        auto in = a.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               out(i, j, k) = in(i, j, k) * 2. * MathConst::pi *
                                              (i + .5) * dx[0] * dx[0] * dx[1];
                           });
    }
    return t.sum(0);
}
ConductionActivityOptions
activity_options () {
    ConductionActivityOptions o;
    o.closure = ConductionCellClosure::AllContributors;
    o.legacy_density_floor = density;
    o.common_density_floor = 2. * density;
    return o;
}
void
radial (Context& c, Real cutoff) {
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(c.rho); mfi.isValid(); ++mfi) {
        auto a = c.rho.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               a(i, j, k) = PhysConst::q_e * density *
                                            (1. + .4 * (cutoff - i * dx[0]));
                           });
    }
}

void
activity (Context& c) {
    auto o = activity_options();
    NativeConductionActivity map(c.sim.geometry, c.ba, c.dm, o);
    radial(c, .57);
    MF original(c.rho.boxArray(), c.dm, 1, 1);
    MF::Copy(original, c.rho, 0, 0, 1, 1);
    require(map.Evaluate(c.rho), "activity evaluates");
    near(difference(original, c.rho, 1), 0, "input ghosts unchanged");
    auto measures = map.Measure();
    require(measures.lost_open_volume > 0 && measures.mixed_cell_volume > 0,
            "mixed cells explicitly lost");
    auto dx = c.sim.geometry.CellSizeArray();
    int const nr = c.sim.geometry.Domain().length(0);
    int const last = int(.57 / dx[0]);
    MF err(c.ba, c.dm, 1, 0);
    for (amrex::MFIter mfi(err); mfi.isValid(); ++mfi) {
        auto e = err.array(mfi);
        auto a = map.CellOpen().const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               e(i, j, k) = a(i, j, k) - (i < last ? 1. : 0.);
                           });
    }
    near(err.norm0(), 0, "AND cell mask analytic");
    for (int d = 0; d < 2; ++d) {
        MF e(map.FaceOpen(d).boxArray(), c.dm, 1, 0);
        for (amrex::MFIter mfi(e); mfi.isValid(); ++mfi) {
            auto out = e.array(mfi);
            auto face = map.FaceOpen(d).const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    out(i, j, k) = face(i, j, k) - (i < last ? 1. : 0.);
                });
        }
        near(e.norm0(), 0, "FV face mask analytic");
    }
    near(measures.retained_open_volume -
             MathConst::pi * std::pow(Real(last) / nr, 2),
         1.e-14, "physical retained volume");
    MF saved(c.ba, c.dm, 1, 0);
    MF::Copy(saved, map.CellOpen(), 0, 0, 1, 0);
    c.ped.setVal(-1.);
    require(!map.Evaluate(c.rho, 0, &c.ped), "invalid pedestal rejects");
    near(difference(saved, map.CellOpen()), 0,
         "invalid activity preserves prior");
    c.rho.setVal(std::numeric_limits<Real>::quiet_NaN());
    require(!map.Evaluate(c.rho), "nonfinite charge rejects");
    near(difference(saved, map.CellOpen()), 0,
         "invalid charge preserves prior");
    amrex::Print() << "ACTIVITY native_volume=" << measures.native_open_volume
                   << " retained=" << measures.retained_open_volume
                   << " lost=" << measures.lost_open_volume
                   << " mixed=" << measures.mixed_cell_volume << "\n";
}
void
floors (Context& c) {
    auto o = activity_options();
    NativeConductionActivity normal(c.sim.geometry, c.ba, c.dm, o);
    c.rho.setVal(.5 * density * PhysConst::q_e);
    require(normal.Evaluate(c.rho), "low raw evaluates");
    require(normal.CellOpen().norm0() == 0, "standard low-density closure");
    auto m = normal.Measure();
    near(m.max_capacity_relative_gap - 1., 1.e-15,
         "inactive capacity discrepancy retained");
    near(m.max_active_capacity_relative_gap, 0,
         "inactive is not active mismatch");
    c.ped.setVal(0);
    require(normal.Evaluate(c.rho, 0, &c.ped),
            "zero pedestal pointer changes gate");
    require(normal.CellOpen().min(0) == 1,
            "positive halo with zero pedestal open");
    near(normal.Measure().max_active_capacity_relative_gap - 1., 1.e-15,
         "active capacity discrepancy explicit");
    c.rho.setVal(0);
    c.ped.setVal(.1 * density * PhysConst::q_e);
    require(normal.Evaluate(c.rho, 0, &c.ped), "pedestal-only opens");
    require(normal.CellOpen().min(0) == 1, "pedestal-only native gate");
    c.ped.setVal(0);
    require(normal.Evaluate(c.rho, 0, &c.ped), "zero total valid");
    require(normal.CellOpen().norm0() == 0, "zero total stays closed");
    c.rho.setVal(density * PhysConst::q_e);
    require(normal.Evaluate(c.rho), "threshold equality valid");
    require(normal.CellOpen().norm0() == 0, "strict native floor threshold");
    o.halo_unfreeze = true;
    NativeConductionActivity halo(c.sim.geometry, c.ba, c.dm, o);
    c.rho.setVal(.25 * density * PhysConst::q_e);
    require(halo.Evaluate(c.rho), "halo open");
    require(halo.CellOpen().min(0) == 1, "halo gate at zero");
    auto all = halo.Measure();
    near(all.native_open_volume - MathConst::pi, 1.e-14,
         "uniform open physical volume");
    near(all.lost_open_volume, 1.e-14, "all-open no erosion");
    near(all.mixed_cell_volume, 0, "all-open exact fraction");
}
void
refinement (int box) {
    Real previous = 0;
    for (int n : {16, 32, 64, 128}) {
        Context c(box, n);
        auto o = activity_options();
        NativeConductionActivity map(c.sim.geometry, c.ba, c.dm, o);
        radial(c, .6);
        require(map.Evaluate(c.rho), "refinement evaluates");
        auto m = map.Measure();
        Real const true_volume = MathConst::pi * .6 * .6;
        Real const error = true_volume - m.retained_open_volume;
        Real const h = 1. / n;
        require(error >= 0 && error <= 2. * MathConst::pi * .6 * h,
                "O(h) insulating erosion bound");
        require(m.lost_open_volume >= 0 &&
                    m.lost_open_volume <= m.mixed_cell_volume + 1.e-14,
                "lost measure bounded by mixed cells");
        if (previous > 0) {
            require(error <= previous + 1.e-14,
                    "retained domain improves under nested refinement");
        }
        previous = error;
        amrex::Print() << "REFINEMENT n=" << n << " h=" << h
                       << " true=" << true_volume
                       << " native=" << m.native_open_volume
                       << " retained=" << m.retained_open_volume
                       << " error=" << error << " error_over_h=" << error / h
                       << " lost=" << m.lost_open_volume << "\n";
    }
}
void
channels (int box) {
    for (int nr : {16, 32, 64}) {
        Context c(box, nr);
        auto o = activity_options();
        NativeConductionActivity map(c.sim.geometry, c.ba, c.dm, o);
        auto dx = c.sim.geometry.CellSizeArray();
        for (amrex::MFIter mfi(c.rho); mfi.isValid(); ++mfi) {
            auto a = c.rho.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    a(i, j, k) = PhysConst::q_e * density *
                                 (std::abs(i * dx[0] - .5) < .045 ? 2. : .5);
                });
        }
        require(map.Evaluate(c.rho), "thin channel activity");
        auto m = map.Measure();
        require(map.NativeEdgeOpen(1).norm0() == 1.,
                "native axial channel connects");
        require(m.native_open_volume > 0, "channel has physical nodal volume");
        if (nr == 16) {
            near(m.retained_open_volume, 0,
                 "unresolved FV channel explicitly closes");
            require(map.FaceOpen(1).norm0() == 0, "closed unresolved FV faces");
        } else {
            require(m.retained_open_volume > 0,
                    "refinement resolves open channel");
        }
        Real const exact = MathConst::pi * (.545 * .545 - .455 * .455);
        require(std::abs(exact - m.retained_open_volume) <=
                    2. * MathConst::pi / nr,
                "channel O(h) volume error");
        amrex::Print() << "CHANNEL n=" << nr
                       << " native=" << m.native_open_volume
                       << " retained=" << m.retained_open_volume
                       << " true=" << exact << " mixed=" << m.mixed_cell_volume
                       << " lost=" << m.lost_open_volume << "\n";
    }
}
void
native_edges (Context& c) {
    auto o = activity_options();
    NativeConductionActivity map(c.sim.geometry, c.ba, c.dm, o);
    radial(c, .57);
    require(map.Evaluate(c.rho), "edge map");
    MF context(c.rho.boxArray(), c.dm, ConductionState::b_ncomp, 3),
        t(c.rho.boxArray(), c.dm, 1, 3), tensor(c.rho.boxArray(), c.dm, 3, 2),
        rhs(c.rho.boxArray(), c.dm, 1, 0), cache(c.rho.boxArray(), c.dm, 2, 1),
        budget(c.rho.boxArray(), c.dm, 1, 1);
    context.setVal(0);
    MF::Copy(context, map.Nodes(), NativeConductionActivity::Open,
             ConductionState::b_open, 1, 0);
    MF::Copy(context, map.Nodes(), NativeConductionActivity::LegacyCapacity,
             ConductionState::b_ne, 1, 0);
    context.FillBoundary(c.sim.geometry.periodicity());
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(t); mfi.isValid(); ++mfi) {
        auto out = t.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                out(i, j, k) = 300. + 10. * i * dx[0] +
                               3. * std::sin(2. * MathConst::pi * j * dx[1]);
            });
    }
    auto kappa = [] AMREX_GPU_HOST_DEVICE(Real, Real, Real) {
        return Real(1.);
    };
    ConductionFDGeometry geometry(c.sim.geometry);
    ConductionFDOptions options;
    options.fourth_order = false;
    options.isotropic = true;
    BuildConductionTensor(t, context, tensor, geometry, options, kappa, kappa,
                          0.);
    // Legacy cache observation only. Vanishing budget dt keeps all ratios 1;
    // no legacy integrator or budget limiter is imported into the new residual.
    options.flux_budget = true;
    options.budget_dt = 1.e-30;
    EvaluateConductionFDRHS(t, context, tensor, rhs, cache, budget, geometry,
                            options, false);
    for (int d = 0; d < 2; ++d) {
        MF err(map.NativeEdgeOpen(d).boxArray(), c.dm, 1, 0);
        for (amrex::MFIter mfi(err); mfi.isValid(); ++mfi) {
            auto out = err.array(mfi);
            auto f = cache.const_array(mfi);
            auto open = map.NativeEdgeOpen(d).const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                out(i, j, k) = open(i, j, k) == 0
                                   ? std::abs(f(i, j, k, d))
                                   : (std::abs(f(i, j, k, d)) > 0 ? 0. : 1.);
            });
        }
        near(err.norm0(), 0, "actual native edge flux gates");
    }
}
void
initializer (Context& c) {
    ThermalStageGuessOptions o;
    o.temperature_floor_ev = 1.;
    MF ns = c.scalar(), ne = c.scalar(), seed = c.scalar(), out = c.scalar(),
       saved = c.scalar(1);
    MF::Copy(saved, c.old, 0, 0, 1, 1);
    ns.setVal(1.25 * density);
    ne.setVal(1.5 * density);
    seed.setVal(unit_energy);
    out.setVal(-789.);
    require(MakeThermalStageInitialGuess(out, c.old, seed, ns, ne, o),
            "compressive seed builds");
    ThermalStageOptions opts;
    opts.temperature_floor_ev = 1.;
    opts.order = 2;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, opts);
    require(stage.RefreshContext(ns, ne, c.b, c.q), "compressive context");
    require(!stage.Admissible(seed), "old seed inadmissible");
    require(stage.Admissible(out), "new seed endpoint admissible");
    require(stage.Residual(out, seed), "new seed production residual");
    near(difference(saved, c.old, 1), 0, "accepted old energy unchanged");
    MF::Copy(seed, out, 0, 0, 1, 0);
    require(MakeThermalStageInitialGuess(out, c.old, seed, ns, ne, o),
            "feasible copy");
    near(difference(out, seed), 0, "feasible seed bitwise unchanged");
    ne.setVal(-1.);
    require(!MakeThermalStageInitialGuess(out, c.old, seed, ns, ne, o),
            "invalid endpoint density");
    near(difference(out, seed), 0, "failed seed publication transactional");
    for (Real theta : {.5, 1.})
        for (Real floor : {0., 1.})
            for (Real ratio : {1., 1.e6, 1.e12}) {
                o.theta = theta;
                o.temperature_floor_ev = floor;
                ns.setVal(ratio * density);
                ne.setVal(2. * ratio * density);
                seed.setVal(0.);
                require(
                    MakeThermalStageInitialGuess(out, c.old, seed, ns, ne, o),
                    "large compression and zero floor");
                Real const result = out.min(0);
                auto point = PrepareThermalStageGuess(unit_energy, result,
                                                      ratio * density,
                                                      2. * ratio * density, o);
                require(point.valid && !point.changed,
                        "representably feasible output");
                require(
                    MakeThermalStageInitialGuess(seed, c.old, seed, ns, ne, o),
                    "in-place trial seed allowed");
                near(difference(out, seed), 0, "same in-place seed");
            }
    o.theta = .5;
    o.temperature_floor_ev = 1.e-100;
    auto roundoff = PrepareThermalStageGuess(1.e200, 0., 1.e18, 1.e18, o);
    require(roundoff.valid && roundoff.energy > 5.e199,
            "affine cancellation padding");
    o.temperature_floor_ev = 1.e300;
    require(!PrepareThermalStageGuess(1., 0., 1.e300, 1.e300, o).valid,
            "overflow rejects without physical repair");
    near(difference(saved, c.old, 1), 0, "old unchanged after all seeds");
}
void
compression (Context& c) {
    auto opts = ThermalStageOptions{};
    opts.temperature_floor_ev = 1.;
    opts.order = 2;
    opts.dt = 1.;
    opts.theta = .5;
    Real const div = -.4;
    Real const ns_value = density / (1. + opts.theta * opts.dt * div);
    Real const ne_value = (ns_value - (1. - opts.theta) * density) / opts.theta;
    Real const target =
        unit_energy / (1. + opts.theta * opts.dt * gamma_e * div);
    opts.boundary[0][1].inflow_temperature_ev =
        (gamma_e - 1.) * target / (ns_value * PhysConst::q_e);
    MF ns = c.scalar(), ne = c.scalar(), endpoint = c.scalar(),
       reference = c.scalar(), seed = c.scalar();
    ns.setVal(ns_value);
    ne.setVal(ne_value);
    std::array<std::unique_ptr<MF>, 2> velocity;
    ThermalFaceContext faces;
    auto dx = c.sim.geometry.CellSizeArray();
    for (int d = 0; d < 2; ++d) {
        velocity[d] = std::make_unique<MF>(
            amrex::convert(c.ba, amrex::IntVect::TheDimensionVector(d)), c.dm,
            1, 0);
        velocity[d]->setVal(0.);
        faces.velocity[d] = velocity[d].get();
    }
    for (amrex::MFIter mfi(*velocity[0]); mfi.isValid(); ++mfi) {
        auto v = velocity[0]->array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               v(i, j, k) = .5 * div * i * dx[0];
                           });
    }
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, opts);
    require(stage.RefreshContext(ns, ne, c.b, c.q, faces),
            "compressive stage refresh");
    ThermalStageGuessOptions guess;
    guess.temperature_floor_ev = 1.;
    amrex::ParmParse("test").query("margin", guess.relative_margin);
    require(MakeThermalStageInitialGuess(seed, c.old, c.old, ns, ne, guess),
            "compressive feasible initializer");
    for (bool pc : {true, false}) {
        MF::Copy(c.u.getMultiFabBlock(energy_name, 0), seed, 0, 0, 1, 0);
        ThermalSolveOptions solve;
        solve.use_preconditioner = pc;
        solve.relative_tolerance = 1.e-10;
        solve.absolute_tolerance = 1.e-11;
        auto result =
            SolveThermalStage(stage, c.u, endpoint, energy_name, solve);
        amrex::Print() << "COMPRESSION solve pc=" << pc
                       << " status=" << int(result.status)
                       << " newton=" << result.newton_iterations
                       << " gmres=" << result.linear_iterations
                       << " initial=" << result.initial_residual
                       << " residual=" << result.residual << "\n";
        require(result.status == ThermalSolveStatus::Converged,
                "compression root converges");
        MF expected = c.scalar();
        expected.setVal(target);
        near(difference(c.u.getMultiFabBlock(energy_name, 0), expected) /
                 unit_energy,
             3.e-9, "analytic compressive root");
        if (pc) {
            MF::Copy(reference, c.u.getMultiFabBlock(energy_name, 0), 0, 0, 1,
                     0);
        } else {
            near(difference(reference, c.u.getMultiFabBlock(energy_name, 0)) /
                     unit_energy,
                 3.e-9, "PC on/off compressive root");
        }
        amrex::Print() << "COMPRESSION pc=" << pc
                       << " newton=" << result.newton_iterations
                       << " gmres=" << result.linear_iterations
                       << " residual=" << result.residual << "\n";
    }
    near(c.old.min(0) - unit_energy, 0, "compression accepted old untouched");
}
void
masked_balance (Context& c) {
    auto ao = activity_options();
    NativeConductionActivity map(c.sim.geometry, c.ba, c.dm, ao);
    radial(c, .57);
    require(map.Evaluate(c.rho), "balance map");
    auto dx = c.sim.geometry.CellSizeArray();
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto u = c.old.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                u(i, j, k) =
                    unit_energy *
                    (1. + .2 * std::cos(2. * MathConst::pi * (j + .5) * dx[1]));
            });
    }
    ThermalStageOptions opts;
    opts.dt = .02;
    opts.temperature_floor_ev = .1;
    opts.order = 2;
    opts.kappa_parallel = "0.00002";
    opts.kappa_perpendicular = "0.00002";
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.q, opts);
    ThermalFaceContext faces;
    faces.conduction_active = &map.CellOpen();
    require(stage.RefreshContext(c.n, c.n, c.b, c.q, faces),
            "actual mapped mask stage");
    MF residual = c.scalar(), endpoint = c.scalar(), reference = c.scalar();
    require(stage.Residual(c.old, residual), "masked residual");
    near(integral(residual, c.sim.geometry) / unit_energy, 2.e-14,
         "masked residual physical energy telescopes");
    for (int d = 0; d < 2; ++d) {
        MF error(stage.FaceFlux(d).boxArray(), c.dm, 1, 0);
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            auto out = error.array(mfi);
            auto open = map.FaceOpen(d).const_array(mfi);
            auto flux = stage.FaceFlux(d).const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    out(i, j, k) =
                        open(i, j, k) == 0 ? std::abs(flux(i, j, k)) : 0.;
                });
        }
        near(error.norm0(), 0, "closed FV faces emit exactly zero");
    }
    for (bool pc : {true, false}) {
        MF::Copy(c.u.getMultiFabBlock(energy_name, 0), c.old, 0, 0, 1, 0);
        ThermalSolveOptions solve;
        solve.use_preconditioner = pc;
        auto result =
            SolveThermalStage(stage, c.u, endpoint, energy_name, solve);
        require(result.status == ThermalSolveStatus::Converged,
                "mapped mask root");
        near((integral(endpoint, c.sim.geometry) -
              integral(c.old, c.sim.geometry)) /
                 unit_energy,
             1.e-10, "masked extensive endpoint energy");
        if (pc) {
            MF::Copy(reference, c.u.getMultiFabBlock(energy_name, 0), 0, 0, 1,
                     0);
        } else {
            near(difference(reference, c.u.getMultiFabBlock(energy_name, 0)) /
                     unit_energy,
                 1.e-9, "mapped mask PC same root");
        }
        amrex::Print() << "MASKED pc=" << pc
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
        std::string name = "activity";
        int box = 8;
        p.query("case", name);
        p.query("box", box);
        Context c(box);
        if (name == "activity")
            activity(c);
        else if (name == "floors")
            floors(c);
        else if (name == "refinement")
            refinement(box);
        else if (name == "channels")
            channels(box);
        else if (name == "native_edges")
            native_edges(c);
        else if (name == "initializer")
            initializer(c);
        else if (name == "compression")
            compression(c);
        else if (name == "masked_balance")
            masked_balance(c);
        else if (name == "reject") {
            ConductionActivityOptions o;
            NativeConductionActivity map(c.sim.geometry, c.ba, c.dm, o);
        } else {
            amrex::Abort("unknown trial case");
        }
        amrex::Print() << "PASS thermal_trial " << name
                       << " boxes=" << c.ba.size() << "\n";
    }
    amrex::Finalize();
}
