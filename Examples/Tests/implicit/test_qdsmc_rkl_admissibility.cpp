/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcRKIntegrator.H"
#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <cmath>
#include <limits>
#include <string>

using RK = QdsmcRKIntegrator;
using Real = amrex::Real;

// Controlled finite excursions use the stage plans and temperature ratios seen
// in three accepted conduction calls. This is a transaction regression, not a
// replay of the unavailable spatial state at the onset of that failure.
void
check (amrex::BoxArray const& ba, amrex::DistributionMapping const& dm,
       int plan, Real ratio, bool lower, bool guarded, int budget = 100000,
       bool nonfinite = false, int reject_count = 1)
{
    constexpr Real duration = 1.2166732870820377e-10;
    amrex::MultiFab y(ba, dm, 1, 0), saved(ba, dm, 1, 0), difference(ba, dm, 1, 0);
    y.setVal(1.0);
    RK::Auxiliary heat{{7.0}, {0.0}};
    Real raw = 0.0, saved_raw = 0.0, raw_min = 0.0;
    int stages = 0, injected = 0, restored = 0, calls = 0;
    Real saved_heat = 0.0, saved_rate = 0.0;
    RK::AttemptHooks hooks;
    hooks.save = [&] () {
        stages = 0;
        amrex::MultiFab::Copy(saved, y, 0, 0, 1, 0);
        saved_raw = raw; saved_heat = heat.value[0]; saved_rate = heat.rate[0];
    };
    hooks.restore = [&] () {
        ++restored;
        amrex::MultiFab::Copy(difference, saved, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(difference, y, 0, 0, 1, 0);
        AMREX_ALWAYS_ASSERT(difference.norminf() == 0.0);
        AMREX_ALWAYS_ASSERT(heat.value[0] == saved_heat && heat.rate[0] == saved_rate);
        raw = saved_raw;
    };
    auto rhs = [&] (amrex::MultiFab&, amrex::MultiFab& k) {
        ++calls;
        k.setVal(-0.05 / duration);
        heat.rate[0] = 0.05 / duration;
    };
    auto project = [&] (amrex::MultiFab& state, Real) {
        ++stages;
        if (stages == (plan == 2 ? 2 : 5) && injected < reject_count) {
            ++injected;
            if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::NProcs()-1) {
                Real const value = nonfinite ? std::numeric_limits<Real>::infinity()
                                             : (lower ? -ratio : ratio);
                state.setVal(value);
                raw += 13.0;
                heat.value[0] += 17.0;
            }
        }
        raw_min = state.min(0);
        if (lower) {
            for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
                auto const t = state.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    t(i,j,k) = amrex::max(t(i,j,k), Real(0.25));
                });
            }
        }
    };
    RK::Admissibility acceptance;
    if (guarded) {
        acceptance = [&] (amrex::MultiFab const& state, Real, int) {
            if (!state.is_finite() || !std::isfinite(raw_min)) {
                return RK::StageAcceptance::invalid;
            }
            return raw_min < 0.25 || state.max(0) > 1.0 + 1.e-10
                       ? RK::StageAcceptance::retry : RK::StageAcceptance::accept;
        };
    }
    Real const ceiling = duration*4.0/(0.9*(Real(plan)*(plan+1)-2))*1.000001;
    RK solver(RK::Scheme::RKL2, rhs, [=] () { return ceiling; }, 1.e-7, 1.e-12,
              0.9, 2.0, budget, project, nullptr, &heat, true, {}, plan, hooks, acceptance);
    auto const st = solver.Advance(y, duration);
    AMREX_ALWAYS_ASSERT(st.n_attempts == calls && calls <= budget);
    AMREX_ALWAYS_ASSERT(st.n_attempts == st.n_accepted + st.n_rejected_work);
    if (!guarded) {
        AMREX_ALWAYS_ASSERT(!st.failure && st.n_retries == 0 && injected == 1);
    } else if (nonfinite) {
        AMREX_ALWAYS_ASSERT(st.failure && st.n_retries == 0 && restored == 1);
        AMREX_ALWAYS_ASSERT(std::string(st.failure).find("nonfinite") != std::string::npos);
    } else if (reject_count > 32) {
        AMREX_ALWAYS_ASSERT(st.failure && st.t_done == 0.0 && st.n_accepted == 0);
        AMREX_ALWAYS_ASSERT(st.n_admissibility_retries == 33 && restored == 33);
        AMREX_ALWAYS_ASSERT(std::string(st.failure).find("retry budget") != std::string::npos);
    } else if (budget < 100000) {
        AMREX_ALWAYS_ASSERT(st.failure && st.t_done == 0.0 && st.n_accepted == 0);
        AMREX_ALWAYS_ASSERT(st.n_admissibility_retries > 0);
    } else {
        AMREX_ALWAYS_ASSERT(!st.failure && st.t_done == duration);
        AMREX_ALWAYS_ASSERT(st.n_admissibility_retries == reject_count &&
                            st.n_retries == reject_count && restored == reject_count);
        AMREX_ALWAYS_ASSERT(std::abs(y.min(0)-0.95) < 2.e-11 &&
                            std::abs(y.max(0)-0.95) < 2.e-11);
        AMREX_ALWAYS_ASSERT(std::abs(y.min(0)-1.0+heat.value[0]-7.0) < 2.e-11);
        AMREX_ALWAYS_ASSERT(raw == 0.0);
    }
    amrex::Print() << "RKL_ADMISSIBILITY plan=" << plan << " ratio=" << ratio
                   << " guarded=" << guarded << " lower=" << lower
                   << " retries=" << st.n_retries << " discarded=" << st.n_rejected_work
                   << " accepted=" << st.n_accepted << " completed=" << st.t_done
                   << " failure=" << (st.failure ? st.failure : "none") << "\n";
}

int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    {
        amrex::BoxArray ba(amrex::Box(amrex::IntVect(0), amrex::IntVect(7)));
        ba.maxSize(4);
        amrex::DistributionMapping dm(ba);
        for (bool guarded : {false, true}) {
            check(ba, dm, 168, 2887.983657/1204.641828, false, guarded);
            check(ba, dm, 168, 56768.84089/2887.983657, false, guarded);
            check(ba, dm, 176, 67408.60607/56768.84089, false, guarded);
        }
        check(ba, dm, 168, 2.0, true, true); // pre-floor undershoot
        check(ba, dm, 2, 2.0, false, true); // final stage must be checked
        check(ba, dm, 168, 2.0, false, true, 100000, false, 3);
        check(ba, dm, 2, 2.0, false, true, 100000, false, 40);
        check(ba, dm, 2, 2.0, false, true, 2); // failed work consumes budget
        check(ba, dm, 2, 2.0, false, true, 100000, true);
        amrex::Print() << "RKL_ADMISSIBILITY_PASS\n";
    }
    amrex::Finalize();
}
