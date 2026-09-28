/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalWall.H"
#include "FieldSolver/ImplicitSolvers/EulerianThermalStage.H"
#include "FieldSolver/ImplicitSolvers/ThermalConductionPC.H"
#include "FieldSolver/ImplicitSolvers/ThermalStageSolver.H"
#include "NonlinearSolvers/FlexibleGMRES.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace warpx::thermal;
using amrex::Real;
constexpr auto name = "hybrid_electron_energy_fp";
constexpr Real density = 1.e18;
constexpr Real gamma = 5.0 / 3.0;
constexpr Real temperature = 100;
constexpr Real specific = PhysConst::q_e * temperature / (gamma - 1);
constexpr Real energy = density * specific;
std::string
number (Real x) {
    std::ostringstream out;
    out << std::setprecision(17) << x;
    return out.str();
}
void
require (bool condition, const char* message) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(condition, message);
}
void
near (Real x, Real y, Real tolerance, const char* message) {
    if (!(std::isfinite(x) && std::abs(x - y) <= tolerance)) {
        amrex::Print() << message << ": " << std::setprecision(17) << x
                       << " vs " << y << " tol " << tolerance << '\n';
        amrex::Abort(message);
    }
}
struct Context {
    WarpX sim;
    amrex::MultiFab n, b, old, source;
    WarpXSolverVec u;
    Context (int max_grid, bool periodic, int nr = 16, int nz = 32,
             Real zlength = 1) {
        amrex::Box const domain(amrex::IntVect(0, 0),
                                amrex::IntVect(nr - 1, nz - 1));
        amrex::RealBox const physical({0., 0.}, {1., zlength});
        int periods[2] = {0, int(periodic)};
        sim.geometry = amrex::Geometry(domain, &physical, 1, periods);
        amrex::BoxArray ba(domain);
        ba.maxSize(max_grid);
        amrex::DistributionMapping dm(ba);
        n.define(ba, dm, 1, 3);
        b.define(ba, dm, 3, 3);
        old.define(ba, dm, 1, 3);
        source.define(ba, dm, 1, 3);
        n.setVal(density);
        b.setVal(0);
        b.setVal(1, 2, 1, 3);
        old.setVal(energy);
        source.setVal(0);
        sim.m_fields.alloc_init(name, 0, ba, dm, 1, amrex::IntVect(3), 0.0);
        u.Define(&sim, "none", "none", {{name, energy}});
        u.getMultiFabBlock(name, 0).setVal(-321);
        amrex::MultiFab::Copy(u.getMultiFabBlock(name, 0), old, 0, 0, 1, 0);
    }
    amrex::MultiFab
    scalar (int ghosts = 0) const {
        return {old.boxArray(), old.DistributionMap(), 1, ghosts};
    }
    void
    set_pattern (Real amplitude, bool radial = false,
                 bool variable_density = false) {
        for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
            auto const ua = old.array(mfi);
            auto const na = n.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                Real const z = (j + 0.5) / 32.;
                Real const r = (i + 0.5) / 16.;
                na(i, j, k) = density * (variable_density && i >= 8 ? 1024 : 1);
                Real const pattern = std::cos(2 * MathConst::pi * z) *
                                     (radial ? std::cos(MathConst::pi * r) : 1);
                ua(i, j, k) =
                    na(i, j, k) * specific * (1 + amplitude * pattern);
            });
        }
        amrex::MultiFab::Copy(u.getMultiFabBlock(name, 0), old, 0, 0, 1, 0);
    }
};
ThermalStageOptions
options () {
    ThermalStageOptions o;
    o.kappa_parallel =
        number(density * PhysConst::kb / (gamma - 1)); // chi_parallel=1 m^2/s
    o.kappa_perpendicular = "0";
    o.order = 2;
    return o;
}
Real
difference (const amrex::MultiFab& a, const amrex::MultiFab& b) {
    amrex::MultiFab diff(a.boxArray(), a.DistributionMap(), 1, 0);
    amrex::MultiFab::LinComb(diff, 1, a, 0, -1, b, 0, 0, 1, 0);
    return diff.norm0(0);
}
Real
integral (const amrex::MultiFab& x, const amrex::Geometry& geom) {
    amrex::MultiFab weighted(x.boxArray(), x.DistributionMap(), 1, 0);
    auto const dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi) {
        auto const a = x.const_array(mfi);
        auto const w = weighted.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               w(i, j, k) = a(i, j, k) * 2 * MathConst::pi *
                                            (i + 0.5) * dx[0] * dx[0] * dx[1];
                           });
    }
    return weighted.sum(0);
}
Real
boundary_power (const EulerianThermalStage& stage) {
    Real result = 0;
    auto const dx = stage.Geometry().CellSizeArray();
    for (int d = 0; d < 2; ++d) {
        if (stage.Geometry().isPeriodic(d)) {
            continue;
        }
        auto const hi = stage.Geometry().Domain().bigEnd(d) + 1;
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(stage.FaceFlux(d)); mfi.isValid(); ++mfi) {
            auto const q = stage.FaceFlux(d).const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        int const index = d == 0 ? i : j;
                        Real const area =
                            d == 0
                                ? 2 * MathConst::pi * i * dx[0] * dx[1]
                                : 2 * MathConst::pi * (i + 0.5) * dx[0] * dx[0];
                        return {(index == 0    ? -1
                                 : index == hi ? 1
                                               : 0) *
                                q(i, j, k) * area};
                    });
        }
        Real value = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealSum(value);
        result += value;
    }
    return result;
}
void
print_result (const char* label, const ThermalSolveResult& r) {
    amrex::Print() << "RESULT " << label << " status=" << int(r.status)
                   << " newton=" << r.newton_iterations
                   << " gmres=" << r.linear_iterations
                   << " residual=" << r.residual
                   << " initial=" << r.initial_residual
                   << " rejects=" << r.rejected_trials
                   << " pc_updates=" << r.pc_updates << '\n';
}
void
converged (const char* label, const ThermalSolveResult& r) {
    print_result(label, r);
    require(r.status == ThermalSolveStatus::Converged,
            "native thermal solve failed");
}
std::vector<Real>
snapshot (const amrex::MultiFab& x) {
    std::vector<Real> result;
    for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi) {
        auto const& fab = x[mfi];
        auto const start = result.size();
        result.resize(start + fab.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(),
                         fab.dataPtr() + fab.size(), result.begin() + start);
    }
    return result;
}
void
source_only (int box) {
    Context c(box, true);
    c.set_pattern(0.1, true);
    c.source.setVal(7.25);
    auto o = options();
    o.kappa_parallel = "0";
    o.kappa_perpendicular = "0";
    o.dt = 0.75;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto end = c.scalar(), expected = c.scalar();
    auto r = SolveThermalStage(stage, c.u, end, name);
    converged("source_only", r);
    amrex::MultiFab::LinComb(expected, 1, c.old, 0, o.dt, c.source, 0, 0, 1, 0);
    near(difference(end, expected) / energy, 0, 2.e-12,
         "exact constant source endpoint");
    amrex::MultiFab::LinComb(expected, 1, c.old, 0, o.theta * o.dt, c.source, 0,
                             0, 1, 0);
    near(difference(c.u.getMultiFabBlock(name, 0), expected) / energy, 0,
         2.e-12, "exact constant source U stage");
}
void
rejection (const std::string& test, int box) {
    Context c(box, true);
    auto o = options();
    if (test == "reject_leg") {
        o.boundary[0][1].kind = BoundaryKind::LegUnsupported;
    }
    if (test == "reject_bfloor") {
        o.magnetic_floor = 0;
    }
    if (test == "reject_density") {
        c.n.setVal(0);
    }
    if (test == "reject_periodic_wall") {
        o.boundary[1][0] = {BoundaryKind::Reservoir, 100};
    }
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    amrex::Abort("unexpected acceptance of invalid context");
}
void
invalid_trial (int box) {
    Context c(box, true);
    c.set_pattern(0.2, true);
    auto o = options();
    o.kappa_parallel = "-1";
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto end = c.scalar();
    end.setVal(12.5);
    auto before = snapshot(c.u.getMultiFabBlock(name, 0)),
         output = snapshot(end);
    auto r = SolveThermalStage(stage, c.u, end, name);
    require(r.status == ThermalSolveStatus::InvalidTrial,
            "negative physical kappa rejects solve");
    require(snapshot(c.u.getMultiFabBlock(name, 0)) == before &&
                snapshot(end) == output,
            "invalid conductivity rollback");
    o = options();
    EulerianThermalStage positive(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    c.u.getMultiFabBlock(name, 0).setVal(0.25 * energy);
    before = snapshot(c.u.getMultiFabBlock(name, 0));
    r = SolveThermalStage(positive, c.u, end, name);
    require(r.status == ThermalSolveStatus::InvalidTrial,
            "positive stage but negative endpoint rejected");
    require(snapshot(c.u.getMultiFabBlock(name, 0)) == before &&
                snapshot(end) == output,
            "negative endpoint rollback");
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto const value = c.u.getMultiFabBlock(name, 0).array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (i == 0 && j == 0) {
                    value(i, j, k) = std::numeric_limits<Real>::max();
                }
            });
    }
    require(!positive.Admissible(c.u.getMultiFabBlock(name, 0)),
            "one overflowing endpoint cannot hide in a global minimum");
}
void
eigen (int box) {
    for (Real theta_c : {Real(0.5), Real(1)}) {
        Context c(box, true);
        c.set_pattern(0.1);
        auto o = options();
        o.conduction_theta = theta_c;
        Real const lambda =
            4 * 32 * 32 * std::pow(std::sin(MathConst::pi / 32), 2);
        o.dt = 8000 / lambda;
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source,
                                   o);
        auto end = c.scalar();
        auto expected = c.scalar();
        auto r = SolveThermalStage(stage, c.u, end, name);
        converged("eigen_pc", r);
        Real const amplification =
            (1 - (1 - theta_c) * o.dt * lambda) / (1 + theta_c * o.dt * lambda);
        for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
            auto const a = expected.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    a(i, j, k) =
                        energy *
                        (1 + 0.1 * amplification *
                                 std::cos(2 * MathConst::pi * (j + 0.5) / 32));
                });
        }
        near(difference(end, expected) / energy, 0, 2.e-10,
             "linear eigenmode amplification");
        near(integral(end, c.sim.geometry), integral(c.old, c.sim.geometry),
             1.e-9, "eigen heat conservation");
        auto saved = c.scalar();
        amrex::MultiFab::Copy(saved, end, 0, 0, 1, 0);
        amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
        ThermalSolveOptions no;
        no.use_preconditioner = false;
        no.max_linear_iterations = 600;
        auto rn = SolveThermalStage(stage, c.u, end, name, no);
        converged("eigen_no_pc", rn);
        near(difference(end, saved) / energy, 0, 2.e-10,
             "PC on/off same eigen root");
    }
}
void
constant (int box) {
    Context c(box, true);
    c.set_pattern(0, false, true);
    c.b.setVal(0);
    auto o = options();
    o.kappa_perpendicular = o.kappa_parallel;
    o.dt = 1.e3;
    o.order = 4;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto f = c.scalar();
    require(stage.Residual(c.old, f), "zero-B residual finite");
    near(f.norm0(0), 0, 1.e-7, "variable density constant T nullspace");
    auto end = c.scalar();
    auto r = SolveThermalStage(stage, c.u, end, name);
    converged("constant_T", r);
    near(difference(end, c.old), 0, 1.e-7, "constant T unchanged");
}
void
nonlinear (int box, Real flux_fraction = 0.001) {
    Context c(box, true);
    c.set_pattern(0.35, true);
    c.b.setVal(0.45, 0, 1, 3);
    c.b.setVal(0.2, 1, 1, 3);
    auto o = options();
    o.kappa_parallel = "(" + o.kappa_parallel + ")*(Te/100)^2.5";
    o.kappa_perpendicular =
        number(0.03 * density * PhysConst::kb / (gamma - 1));
    o.dt = 0.02;
    o.order = 4;
    o.conduction_theta = 1;
    o.free_streaming_fraction = flux_fraction;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto end = c.scalar();
    auto saved = c.scalar();
    if (flux_fraction < 1.e-5) {
        auto fcap = c.scalar(), fraw = c.scalar();
        require(stage.Residual(c.old, fcap), "capped residual finite");
        auto uncapped_options = o;
        uncapped_options.free_streaming_fraction = 0;
        EulerianThermalStage uncapped(c.sim.geometry, c.old, c.n, c.b, c.source,
                                      uncapped_options);
        require(uncapped.Residual(c.old, fraw),
                "uncapped control residual finite");
        require(difference(fcap, fraw) > 0.05 * fraw.norm0(0),
                "fixture exercises a material flux cap");
    }
    auto r = SolveThermalStage(stage, c.u, end, name);
    converged("nonlinear_pc", r);
    near(integral(end, c.sim.geometry), integral(c.old, c.sim.geometry), 2.e-9,
         "nonlinear heat conservation");
    amrex::MultiFab::Copy(saved, end, 0, 0, 1, 0);
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    ThermalSolveOptions no;
    no.use_preconditioner = false;
    auto rn = SolveThermalStage(stage, c.u, end, name, no);
    converged("nonlinear_no_pc", rn);
    near(difference(end, saved) / energy, 0, 2.e-9,
         "PC on/off same nonlinear root");
    require(stage.Admissible(c.u.getMultiFabBlock(name, 0)),
            "positive accepted nonlinear endpoint");
}
void
density_jump (int box) {
    Context c(box, true);
    c.set_pattern(0.35, true, true);
    c.b.setVal(0);
    auto o = options();
    o.kappa_perpendicular = o.kappa_parallel;
    o.dt = 0.5;
    o.conduction_theta = 1;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto end = c.scalar();
    auto saved = c.scalar();
    auto r = SolveThermalStage(stage, c.u, end, name);
    converged("density_pc", r);
    amrex::MultiFab::Copy(saved, end, 0, 0, 1, 0);
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    ThermalSolveOptions no;
    no.use_preconditioner = false;
    no.max_linear_iterations = 1200;
    auto rn = SolveThermalStage(stage, c.u, end, name, no);
    converged("density_no_pc", rn);
    near(difference(end, saved) / energy, 0, 2.e-8,
         "PC on/off same density-jump root");
    near(integral(end, c.sim.geometry), integral(c.old, c.sim.geometry), 1.e-7,
         "density-jump heat conservation");
    require(r.linear_iterations < rn.linear_iterations,
            "weighted thermal PC reduces stiff density-jump Krylov work");
}
void
boundary (int box) {
    Context c(box, false);
    c.set_pattern(0.1, true);
    c.b.setVal(0);
    c.source.setVal(7);
    auto o = options();
    o.kappa_perpendicular = o.kappa_parallel;
    o.dt = 0.02;
    o.conduction_theta = 1;
    o.boundary[0][1] = {BoundaryKind::PrescribedFlux, 5};
    o.boundary[1][0] = {BoundaryKind::Reservoir, 80};
    o.boundary[1][1] = {BoundaryKind::Reservoir, 110};
    o.free_streaming_fraction = 1.e-6;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto end = c.scalar();
    auto f = c.scalar();
    auto r = SolveThermalStage(stage, c.u, end, name);
    converged("wall_source", r);
    require(stage.Residual(c.u.getMultiFabBlock(name, 0), f),
            "accepted wall residual");
    Real const delta =
        integral(end, c.sim.geometry) - integral(c.old, c.sim.geometry);
    Real const exchange =
        o.dt * (integral(c.source, c.sim.geometry) - boundary_power(stage));
    near(delta, exchange, 2.e-8,
         "physical heat balance including both boundary faces and source");
    amrex::Print() << "HEAT delta=" << delta << " exchange=" << exchange
                   << '\n';
}
void
rows (int box) {
    for (bool periodic : {true, false}) {
        Context c(box, periodic);
        c.set_pattern(0.08, false, true);
        c.b.setVal(0);
        auto o = options();
        o.kappa_perpendicular = o.kappa_parallel;
        o.dt = 0.7;
        if (!periodic) {
            o.boundary[0][1] = {BoundaryKind::Reservoir, 100};
            o.boundary[1][0] = {BoundaryKind::Reservoir, 100};
            o.boundary[1][1] = {BoundaryKind::PrescribedFlux, 2};
        }
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source,
                                   o);
        ThermalConductionPC pc(stage);
        require(pc.Freeze(c.old), "freeze thermal rows");
        auto v = c.scalar(), a = c.scalar(), b = c.scalar(), plus = c.scalar(),
             minus = c.scalar(), fp = c.scalar(), fm = c.scalar();
        for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
            auto const va = v.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    va(i, j, k) =
                        energy * (0.2 * std::sin(Real(2 * i + 3 * j)) + 0.05);
                });
        }
        pc.ApplyOperator(a, v);
        pc.ApplyRows(b, v);
        near(difference(a, b) / a.norm0(0), 0, 2.e-13,
             "MHD emitted rows equal distributed MLMG action");
        Real const eps = 1.e-4;
        amrex::MultiFab::LinComb(plus, 1, c.old, 0, eps, v, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(minus, 1, c.old, 0, -eps, v, 0, 0, 1, 0);
        require(stage.Residual(plus, fp) && stage.Residual(minus, fm),
                "row finite differences");
        amrex::MultiFab::LinComb(b, 0.5 / eps, fp, 0, -0.5 / eps, fm, 0, 0, 1,
                                 0);
        near(difference(a, b) / a.norm0(0), 0, 1.e-8,
             "density-weighted rows match compact physical Jacobian");
        // Residual probe changed its own emission arrays, never frozen PC data.
        pc.ApplyOperator(b, v);
        near(difference(a, b), 0, 0,
             "PC remains frozen across trial residual probes");
    }
}
void
rollback (int box) {
    Context c(box, true);
    c.set_pattern(0.2, true);
    auto o = options();
    o.dt = 0.02;
    auto const old_before = snapshot(c.old), n_before = snapshot(c.n),
               b_before = snapshot(c.b), src_before = snapshot(c.source);
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto a = c.scalar(), b = c.scalar(), f = c.scalar(), trial = c.scalar(),
         end = c.scalar(3);
    require(stage.Residual(c.old, a), "A residual");
    amrex::MultiFab::Copy(trial, c.old, 0, 0, 1, 0);
    trial.mult(1.01, 0, 1, 0);
    require(stage.Residual(trial, b) && stage.Residual(c.old, f),
            "B/A residuals");
    near(difference(a, f), 0, 0, "A/B/A residual equality");
    require(snapshot(c.old) == old_before && snapshot(c.n) == n_before &&
                snapshot(c.b) == b_before && snapshot(c.source) == src_before,
            "accepted inputs and ghosts unchanged during probes");
    auto const input = snapshot(c.u.getMultiFabBlock(name, 0));
    end.setVal(123.75);
    auto const output = snapshot(end);
    ThermalSolveOptions fail;
    fail.max_newton_iterations = 0;
    auto r = SolveThermalStage(stage, c.u, end, name, fail);
    require(r.status == ThermalSolveStatus::IterationLimit,
            "forced nonlinear rejection");
    require(snapshot(c.u.getMultiFabBlock(name, 0)) == input &&
                snapshot(end) == output,
            "full nonlinear rollback");
    fail.max_newton_iterations = 20;
    fail.max_linear_iterations = 0;
    r = SolveThermalStage(stage, c.u, end, name, fail);
    require(r.status == ThermalSolveStatus::LinearFailure,
            "forced Krylov rejection");
    require(snapshot(c.u.getMultiFabBlock(name, 0)) == input &&
                snapshot(end) == output,
            "full Krylov rollback");
    r = SolveThermalStage(stage, c.u, end, name);
    converged("retry", r);
    auto retry_end = c.scalar();
    amrex::MultiFab::Copy(retry_end, end, 0, 0, 1, 0);
    EulerianThermalStage fresh(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    r = SolveThermalStage(fresh, c.u, end, name);
    converged("fresh", r);
    near(difference(end, retry_end), 0, 0,
         "retry and fresh solve reach identical root");
    // External context writes after construction must not alter the stage.
    c.n.mult(2, 0, 1, 0);
    c.b.setVal(17);
    c.old.mult(1.5, 0, 1, 0);
    c.source.setVal(31);
    require(stage.Residual(stage.OldEnergy(), f),
            "deep-copied context residual");
    near(difference(a, f), 0, 0, "stage owns immutable context snapshots");
    amrex::MultiFab::Copy(c.old, stage.OldEnergy(), 0, 0, 1, 0);
    c.n.mult(0.5, 0, 1, 0);
    c.b.setVal(0);
    c.b.setVal(1, 2, 1, 3);
    // Impossible cooling demands a negative mean endpoint: no floor repair or
    // residual-free active-set acceptance can turn this into a successful step.
    c.source.setVal(-10 * energy / o.dt);
    EulerianThermalStage drain(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    auto const before = snapshot(c.u.getMultiFabBlock(name, 0));
    auto const end_before = snapshot(end);
    r = SolveThermalStage(drain, c.u, end, name);
    print_result("unreachable_drain", r);
    require(r.status != ThermalSolveStatus::Converged,
            "unreachable positive endpoint rejected");
    require(snapshot(c.u.getMultiFabBlock(name, 0)) == before &&
                snapshot(end) == end_before,
            "drain rollback");
}
#include "test_thermal_transport.H"
#include "test_thermal_transport_pc.H"
#include "test_thermal_conductivity.H"
} // namespace
int
main (int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        std::string test = "eigen";
        int box = 8;
        pp.query("case", test);
        pp.query("max_grid_size", box);
        if (test.rfind("reject_", 0) == 0) {
            rejection(test, box);
        } else if (test == "conductivity_zero_floor") {
            conductivity_zero_floor(box);
        } else if (test == "conductivity_laws") {
            conductivity_laws(box);
            conductivity_variable_field(box);
        } else if (test == "conductivity_active") {
            conductivity_active(box);
        } else if (test == "conductivity_roots") {
            conductivity_roots(box);
        } else if (test == "transport_pc") {
            transport_pc(box);
        } else if (test == "transport_pc_omission") {
            transport_pc_omission(box);
        } else if (test == "manufactured") {
            manufactured(box);
        } else if (test == "mg_controls") {
            mg_controls(box);
        } else if (test == "transport") {
            transport_identity(box);
        } else if (test == "compression") {
            compression_identity(box);
        } else if (test == "density_context") {
            density_context(box);
        } else if (test == "source_callback") {
            source_callback(box);
        } else if (test == "leg_law") {
            leg_law(box);
        } else if (test == "leg_root") {
            leg_root(box);
        } else if (test == "source") {
            source_only(box);
        } else if (test == "invalid") {
            invalid_trial(box);
        } else if (test == "eigen") {
            eigen(box);
        } else if (test == "constant") {
            constant(box);
        } else if (test == "nonlinear") {
            nonlinear(box);
        } else if (test == "cap") {
            nonlinear(box, 1.e-6);
        } else if (test == "density") {
            density_jump(box);
        } else if (test == "boundary") {
            boundary(box);
        } else if (test == "rows") {
            rows(box);
        } else if (test == "rollback") {
            rollback(box);
        } else {
            amrex::Abort("unknown fixture");
        }
        amrex::Print() << "PASS " << test << " box=" << box << '\n';
    }
    amrex::Finalize();
}
