/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "FieldSolver/ImplicitSolvers/EulerianStoppingTransfer.H"
#include "KineticThermalMoments.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

namespace {
using namespace warpx::thermal;
using MF = amrex::MultiFab;
using amrex::Real;
constexpr Real n0 = 1.e19;
void
require (bool b, const char* why) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(b, why);
}
void
near (Real a, Real b, Real tol, const char* why) {
    if (!(std::isfinite(a) && std::isfinite(b) &&
          std::abs(a - b) <=
              tol * std::max({Real(1), std::abs(a), std::abs(b)}))) {
        amrex::Print() << std::setprecision(17) << why << ": " << a << " vs "
                       << b << '\n';
        amrex::Abort(why);
    }
}
MF
clone (const MF& in) {
    MF out(in.boxArray(), in.DistributionMap(), 1, in.nGrowVect());
    MF::Copy(out, in, 0, 0, 1, in.nGrowVect());
    return out;
}
Real
difference (const MF& a, const MF& b, int ng = 0) {
    MF d(a.boxArray(), a.DistributionMap(), 1, ng);
    MF::LinComb(d, 1, a, 0, -1, b, 0, 0, 1, ng);
    return d.norm0(0, ng);
}
struct Context {
    amrex::Geometry g;
    amrex::BoxArray cells;
    amrex::DistributionMapping dm;
    MF impulse, rho, pedestal, before, capacity;
    std::unique_ptr<KineticThermalMoments> map;
    Context (int box, bool periodic = false, bool corrected = false) {
        amrex::Box dom(amrex::IntVect(0), amrex::IntVect(15));
        amrex::RealBox physical({0., 0.}, {1., 2.});
        int period[2] = {0, int(periodic)};
        g = amrex::Geometry(dom, &physical, 1, period);
        cells = amrex::BoxArray(dom);
        cells.maxSize(box);
        dm = amrex::DistributionMapping(cells);
        auto nodes = amrex::convert(cells, amrex::IntVect(1));
        impulse.define(nodes, dm, 1, 1);
        rho.define(nodes, dm, 1, 1);
        pedestal.define(nodes, dm, 1, 1);
        before.define(cells, dm, 1, 1);
        capacity.define(cells, dm, 1, 1);
        impulse.setVal(0);
        rho.setVal(n0 * PhysConst::q_e);
        pedestal.setVal(0);
        before.setVal(100);
        capacity.setVal(n0);
        ThermalMomentOptions o;
        o.number_density_floor = 1.e17;
        o.active_density_floor = 1.e17;
        o.verboncoeur_axis_correction = corrected;
        o.boundary[0] = {MomentBoundary::Axis, MomentBoundary::PEC};
        o.boundary[1] =
            periodic
                ? std::array{MomentBoundary::Periodic, MomentBoundary::Periodic}
                : std::array{MomentBoundary::PMC, MomentBoundary::PMC};
        o.current_boundary[0] = {MomentBoundary::Axis, MomentBoundary::PMC};
        map = std::make_unique<KineticThermalMoments>(g, cells, dm, o);
        refresh();
    }
    void
    refresh () {
        KineticThermalStateView state{rho};
        state.pedestal = &pedestal;
        state.energy = &before;
        require(map->Evaluate(state), "valid endpoint moment/capacity state");
        MF::Copy(capacity, map->NumberDensity(), 0, 0, 1, 0);
        capacity.FillBoundary(g.periodicity());
    }
    StoppingTransferOptions
    options () const {
        StoppingTransferOptions o;
        o.raw_density_floor = 1.e17;
        return o;
    }
    bool
    evaluate (EulerianStoppingTransfer& m) {
        return m.Evaluate(*map, impulse, rho, capacity, before);
    }
};
void
conservation (int box) {
    for (bool periodic : {false, true})
        for (bool corrected : {false, true}) {
            Context c(box, periodic, corrected);
            auto volume = MakeQdsmcVolumeElement(
                c.g, amrex::IndexType(amrex::IntVect(1)));
            Real expected = 0;
            // Caller-owned particle-like partial scatter: each cell belongs to
            // exactly one rank/FAB. The actual node volume matches collision
            // code.
            for (int p = 0; p < 173; ++p) {
                int const i = (3 * p) % 16, j = (7 * p + p / 16) % 16;
                Real const wr = .07 + .8 * std::fmod(p * .137, 1.),
                           wz = .11 + .7 * std::fmod(p * .191, 1.);
                Real const loss = .001 * (.4 + std::sin(Real(p)));
                expected += loss;
                for (amrex::MFIter mfi(c.before); mfi.isValid(); ++mfi) {
                    if (!mfi.validbox().contains(amrex::IntVect(i, j))) {
                        continue;
                    }
                    for (int di = 0; di < 2; ++di)
                        for (int dj = 0; dj < 2; ++dj) {
                            c.impulse[mfi](amrex::IntVect(i + di, j + dj)) +=
                                loss * (di ? wr : 1 - wr) * (dj ? wz : 1 - wz) /
                                volume(i + di, j + dj, 0);
                        }
                }
            }
            c.impulse.SumBoundary(
                c.g.periodicity()); // exactly once, caller-owned
            auto synced = clone(c.impulse);
            synced.OverrideSync(c.g.periodicity());
            amrex::Print() << "CONSOLIDATED_SEAM roundoff="
                           << difference(synced, c.impulse)
                           << " scale=" << c.impulse.norminf() << '\n';
            auto o = c.options();
            EulerianStoppingTransfer m(c.g, c.cells, c.dm, o);
            require(c.evaluate(m), "consolidated signed measured impulse");
            auto l = m.Ledger();
            near(l.requested, expected, 2.e-13,
                 "measured particle loss equals physical nodal impulse");
            near(l.delivered, expected, 2.e-13,
                 "signed physical nodal-to-cell stopping conservation");
            require(l.floor_declined == 0 && l.density_declined == 0,
                    "unrestricted transfer has no decline");
            near(l.map_defect, 0, 2.e-13, "physical restriction map defect");
            near(l.exchange_defect, 0, 2.e-13,
                 "realized stopping exchange defect");
            amrex::Print() << std::setprecision(16)
                           << "STOPPING_CONSERVATION periodic=" << periodic
                           << " native_axis3=" << corrected
                           << " requested=" << l.requested
                           << " delivered=" << l.delivered
                           << " map_defect=" << l.map_defect
                           << " exchange_defect=" << l.exchange_defect << '\n';
        }
}
void
gates (int box) {
    Context c(box);
    auto o = c.options();
    o.raw_density_floor = n0;
    for (amrex::MFIter mfi(c.rho); mfi.isValid(); ++mfi) {
        auto r = c.rho.array(mfi), s = c.impulse.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               r(i, j, k) = n0 * PhysConst::q_e *
                                            (i % 3 == 0   ? .5
                                             : i % 3 == 1 ? 1.
                                                          : 2.);
                               s(i, j, k) = i % 2 == 0 ? -1. : 3.;
                           });
    }
    c.refresh();
    EulerianStoppingTransfer m(c.g, c.cells, c.dm, o);
    require(c.evaluate(m), "raw gate transfer");
    auto candidate = clone(m.CandidateEnergy());
    auto ledger = m.Ledger();
    require(ledger.density_declined != 0, "signed raw gate decline exercised");
    for (amrex::MFIter mfi(c.rho); mfi.isValid(); ++mfi) {
        for (amrex::BoxIterator bi(mfi.validbox()); bi.ok(); ++bi) {
            auto iv = bi();
            Real expected =
                c.rho[mfi](iv) > PhysConst::q_e * n0 ? c.impulse[mfi](iv) : 0;
            require(m.NodalEligibleImpulse()[mfi](iv) == expected,
                    "strict raw gate includes equality in declined channel");
        }
    }
    c.pedestal.setVal(7 * n0 * PhysConst::q_e);
    c.refresh();
    require(c.evaluate(m), "pedestal transfer");
    require(difference(candidate, m.CandidateEnergy()) == 0,
            "pedestal changes capacity without amplifying source");
    near(m.Ledger().delivered, ledger.delivered, 0,
         "pedestal leaves unfloored delivered work unchanged");
    near(m.Ledger().exchange_defect, 0, 3.e-13,
         "raw gate signed energy ledger");
}
void
floor (int box) {
    Context c(box);
    auto o = c.options();
    o.temperature_floor_ev = 10;
    Real const factor = PhysConst::q_e * o.temperature_floor_ev / (o.gamma - 1);
    c.impulse.setVal(-200);
    for (amrex::MFIter mfi(c.before); mfi.isValid(); ++mfi) {
        auto u = c.before.array(mfi);
        auto n = c.capacity.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                u(i, j, k) = n(i, j, k) * factor * (j % 2 ? 3. : .5);
            });
    }
    EulerianStoppingTransfer m(c.g, c.cells, c.dm, o);
    require(c.evaluate(m), "cell cooling floor transfer");
    require(m.Ledger().floor_declined > 0,
            "positive cooling decline exercised");
    for (amrex::MFIter mfi(c.before); mfi.isValid(); ++mfi) {
        for (amrex::BoxIterator bi(mfi.validbox()); bi.ok(); ++bi) {
            auto iv = bi();
            Real const minimum =
                std::min(c.before[mfi](iv), c.capacity[mfi](iv) * factor);
            require(m.CandidateEnergy()[mfi](iv) == minimum,
                    "cell floor never lifts an already subfloor U");
        }
    }
    near(m.Ledger().exchange_defect, 0, 3.e-13,
         "floor decline signed exchange identity");
}
void
counterexample (int box) {
    Context c(box);
    auto o = c.options();
    o.temperature_floor_ev = (o.gamma - 1) / (n0 * PhysConst::q_e);
    c.capacity.setVal(n0);
    for (amrex::MFIter mfi(c.before); mfi.isValid(); ++mfi) {
        auto u = c.before.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               u(i, j, k) = j < 8 ? 1. : 100.;
                           });
    }
    for (amrex::MFIter mfi(c.impulse); mfi.isValid(); ++mfi) {
        auto x = c.impulse.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               x(i, j, k) = j == 8 ? -49.5 : 0.;
                           });
    }
    EulerianStoppingTransfer m(c.g, c.cells, c.dm, o);
    require(c.evaluate(m), "nodal floor counterexample");
    for (amrex::MFIter mfi(c.before); mfi.isValid(); ++mfi) {
        for (amrex::BoxIterator bi(mfi.validbox()); bi.ok(); ++bi) {
            auto iv = bi();
            if (iv[1] != 7 && iv[1] != 8) {
                continue;
            }
            near(m.RequestedCellImpulse()[mfi](iv), -24.75, 1.e-14,
                 "physical overlap of nodally admissible cooling");
            near(m.CandidateEnergy()[mfi](iv), iv[1] == 7 ? 1. : 75.25, 1.e-14,
                 "primary cell floor prevents negative U");
        }
    }
    require(m.Ledger().floor_declined > 0,
            "counterexample is explicitly declined, not hidden repair");
}
void
purity (int box) {
    Context c(box, true);
    c.impulse.setVal(2);
    EulerianStoppingTransfer m(c.g, c.cells, c.dm, c.options());
    auto u = clone(c.before), rho = clone(c.rho), impulse = clone(c.impulse),
         n = clone(c.capacity);
    auto density = clone(c.map->NumberDensity()),
         te = clone(c.map->NodalTemperature()),
         pe = clone(c.map->OhmPressure());
    require(c.evaluate(m), "purity A");
    auto candidate = clone(m.CandidateEnergy());
    auto l = m.Ledger();
    c.impulse.mult(1.7, 1);
    require(c.evaluate(m), "purity B");
    MF::Copy(c.impulse, impulse, 0, 0, 1, 1);
    require(c.evaluate(m), "purity A restored");
    require(difference(candidate, m.CandidateEnergy()) == 0,
            "A/B/A exact candidate");
    require(m.Ledger().delivered == l.delivered &&
                m.Ledger().requested == l.requested,
            "repeated ledger does not accumulate");
    require(difference(u, c.before, 1) == 0 && difference(rho, c.rho, 1) == 0 &&
                difference(impulse, c.impulse, 1) == 0 &&
                difference(n, c.capacity, 1) == 0,
            "all accepted inputs and ghosts untouched");
    require(difference(density, c.map->NumberDensity(), density.nGrow()) == 0 &&
                difference(te, c.map->NodalTemperature(), te.nGrow()) == 0 &&
                difference(pe, c.map->OhmPressure(), pe.nGrow()) == 0,
            "physical map restriction preserves moment outputs");
    // Parent-owned commit and staging clear; a second endpoint pass is a no-op.
    MF::Copy(c.before, candidate, 0, 0, 1, 0);
    c.impulse.setVal(0);
    require(c.evaluate(m), "consumed impulse");
    require(m.Ledger().requested == 0 && m.Ledger().delivered == 0 &&
                difference(c.before, m.CandidateEnergy()) == 0,
            "zeroed consumed impulse delivers no second transfer");
}
void
invalid (int box) {
    Context c(box, true);
    EulerianStoppingTransfer m(c.g, c.cells, c.dm, c.options());
    c.impulse.setVal(std::numeric_limits<Real>::quiet_NaN());
    require(!c.evaluate(m), "nonfinite impulse rejected");
    c.impulse.setVal(0);
    c.capacity.setVal(0);
    require(!c.evaluate(m), "zero capacity rejected");
    c.capacity.setVal(n0);
    c.before.setVal(-1);
    require(!c.evaluate(m), "negative accepted energy rejected");
    c.before.setVal(100);
    for (amrex::MFIter mfi(c.impulse); mfi.isValid(); ++mfi) {
        auto x = c.impulse.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) { x(i, j, k) = j; });
    }
    require(!c.evaluate(m), "unconsolidated periodic seam copies rejected");
    c.impulse.setVal(2);
    require(c.evaluate(m), "valid recovery after invalid input");
    near(m.Ledger().exchange_defect, 0, 3.e-13,
         "invalid evaluation leaves no accumulated ledger");
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int box = 4;
        std::string which = "conservation", reject;
        pp.query("case", which);
        pp.query("max_grid_size", box);
        pp.query("reject", reject);
        if (!reject.empty()) {
            Context c(box);
            auto o = c.options();
            if (reject == "eb") {
                o.embedded_boundary = true;
            }
            if (reject == "amr") {
                o.physical_levels = 2;
            }
            if (reject == "mode") {
                o.azimuthal_modes = 2;
            }
            if (reject == "gamma") {
                o.gamma = 1;
            }
            EulerianStoppingTransfer m(c.g, c.cells, c.dm, o);
            require(c.evaluate(m), "alias test baseline");
            if (reject == "alias") {
                m.Evaluate(*c.map, c.impulse, c.rho, c.capacity,
                           m.CandidateEnergy());
            }
            amrex::Abort("REJECTION DID NOT OCCUR");
        }
        if (which == "conservation") {
            conservation(box);
        } else if (which == "gates") {
            gates(box);
        } else if (which == "floor") {
            floor(box);
        } else if (which == "counterexample") {
            counterexample(box);
        } else if (which == "purity") {
            purity(box);
        } else if (which == "invalid") {
            invalid(box);
        } else {
            amrex::Abort("Unknown stopping case");
        }
        amrex::Print() << "STOPPING_PASS case=" << which << " box=" << box
                       << '\n';
    }
    amrex::Finalize();
}
