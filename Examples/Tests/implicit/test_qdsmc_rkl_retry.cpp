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

// All boxes carry the same two scalar ODEs: a cooling cell held at its floor,
// and exponential cooling. Only the last MPI rank imposes the tighter bound.
// Local projection accounts are transactions, with an independent accepted
// projection total assembled by the fixture at each next super-step entry.
struct Result {
    QdsmcRKStats stats;
    Real temperature, closure, raw, count;
};

Result
run (amrex::BoxArray const& ba, amrex::DistributionMapping const& dm,
     int rejections, int budget, Real ceiling = 0.001, int reject_after = 0) {
    amrex::MultiFab y(ba, dm, 2, 1);
    y.setVal(0.0);
    y.setVal(1.0, 1, 1, 1);
    amrex::iMultiFab mask(ba, dm, 1, 0), saved_mask(ba, dm, 1, 0);
    mask.setVal(0);
    RK::Auxiliary heat{{3.0, 5.0, 7.0}, {0.0, 0.0, 0.0}};
    Real raw = 0.0, count = 0.0, saved_raw = 0.0, saved_count = 0.0;
    Real accepted_raw = 0.0, accepted_count = 0.0, pending_raw = 0.0,
         pending_count = 0.0;
    Real observed = ceiling;
    int stages = 0, rejected = 0, rhs_calls = 0, restores = 0;
    bool cache_valid = false;
    int attempt = 0;
    Real saved_temperature = 0.0;
    amrex::Vector<Real> saved_heat, saved_rate;
    RK::AttemptHooks hooks;
    hooks.save = [&] () {
        accepted_raw += pending_raw;
        accepted_count += pending_count;
        pending_raw = pending_count = 0.0;
        saved_raw = raw;
        saved_count = count;
        amrex::iMultiFab::Copy(saved_mask, mask, 0, 0, 1, 0);
        stages = 0;
        ++attempt;
        saved_temperature = y.min(1);
        saved_heat = heat.value;
        saved_rate = heat.rate;
    };
    hooks.restore = [&] () {
        ++restores;
        AMREX_ALWAYS_ASSERT(y.min(1) == saved_temperature);
        AMREX_ALWAYS_ASSERT(heat.value == saved_heat &&
                            heat.rate == saved_rate);
        // Nonzero raw projection activity must really have occurred before
        // this rejection, even though none may survive the rollback.
        AMREX_ALWAYS_ASSERT(stages < 5 || raw > saved_raw);
        raw = saved_raw;
        count = saved_count;
        pending_raw = pending_count = 0.0;
        amrex::iMultiFab::Copy(mask, saved_mask, 0, 0, 1, 0);
        cache_valid = false;
    };
    auto rhs = [&] (amrex::MultiFab& state, amrex::MultiFab& k) {
        ++stages;
        ++rhs_calls;
        state.FillBoundary();
        cache_valid = true;
        k.setVal(-1.0, 0, 1, 0);
        amrex::MultiFab::Copy(k, state, 1, 1, 1, 0);
        k.mult(-1.0, 1, 1, 0);
        heat.rate = {1.0, 0.0, state.min(1)};
    };
    auto cap = [&] () {
        AMREX_ALWAYS_ASSERT(cache_valid);
        Real bound = ceiling; // ordinary re-entry alone would undo shrinkage
        if (stages == 5 && rejected < rejections && attempt > reject_after) {
            observed *= 0.3;
            ++rejected;
            if (amrex::ParallelDescriptor::MyProc() ==
                amrex::ParallelDescriptor::NProcs() - 1) {
                bound = observed;
            }
        }
        amrex::ParallelDescriptor::ReduceRealMin(bound);
        return bound;
    };
    auto project = [&] (amrex::MultiFab& state, Real) {
        Real const correction = amrex::max(-state.min(0), 0.0);
        state.setVal(0.0, 0, 1, 0);
        heat.value[1] += correction;
        raw += correction;
        count += 1.0;
        pending_raw += correction;
        pending_count += 1.0;
        mask.setVal(1);
    };
    RK rk(RK::Scheme::RKL2, rhs, cap, 1.e-7, 1.e-12, 0.9, 2.0, budget, project,
          &mask, &heat, true, {}, 8, hooks);
    auto const st = rk.Advance(y, 0.02);
    AMREX_ALWAYS_ASSERT(st.n_attempts == rhs_calls && st.n_attempts <= budget);
    AMREX_ALWAYS_ASSERT(st.n_attempts == st.n_accepted + st.n_rejected_work);
    AMREX_ALWAYS_ASSERT(st.n_retries == rejected && restores == rejected);
    if (!st.failure) {
        accepted_raw += pending_raw;
        accepted_count += pending_count;
        AMREX_ALWAYS_ASSERT(std::abs(raw - accepted_raw) < 2.e-14 &&
                            count == accepted_count);
        AMREX_ALWAYS_ASSERT(count == st.n_accepted && st.t_done == 0.02);
        AMREX_ALWAYS_ASSERT(mask.min(0) == 1);
        AMREX_ALWAYS_ASSERT(std::abs(heat.value[0] - 3.0 - 0.02) < 2.e-12);
        AMREX_ALWAYS_ASSERT(std::abs(heat.value[1] - 5.0 - 0.02) < 2.e-12);
    } else {
        AMREX_ALWAYS_ASSERT(heat.value == amrex::Vector<Real>({3.0, 5.0, 7.0}));
    }
    Real const temperature = y.min(1);
    Real const closure = temperature - 1.0 + heat.value[2] - 7.0;
    amrex::Print() << "RKL_RETRY_CASE requested_rejections=" << rejections
                   << " retries=" << st.n_retries << " work=" << st.n_attempts
                   << " accepted=" << st.n_accepted
                   << " rejected_work=" << st.n_rejected_work
                   << " t_done=" << st.t_done << " temperature=" << temperature
                   << " closure=" << closure << " raw=" << raw
                   << " count=" << count
                   << " failure=" << (st.failure ? st.failure : "none") << "\n";
    return {st, temperature, closure, raw, count};
}

int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::BoxArray ba(amrex::Box(amrex::IntVect(0), amrex::IntVect(7)));
        ba.maxSize(4);
        amrex::DistributionMapping dm(ba);
        auto const reference = run(ba, dm, 0, 100000, 1.e-5);
        for (int rejects : {0, 1, 3}) {
            auto const r = run(ba, dm, rejects, 100000);
            AMREX_ALWAYS_ASSERT(!r.stats.failure &&
                                r.stats.n_retries == rejects);
            AMREX_ALWAYS_ASSERT(
                std::abs(r.temperature - reference.temperature) < 2.e-7);
            AMREX_ALWAYS_ASSERT(std::abs(r.temperature - std::exp(-0.02)) <
                                2.e-7);
            AMREX_ALWAYS_ASSERT(std::abs(r.closure) < 3.e-12);
            if (rejects) {
                AMREX_ALWAYS_ASSERT(r.stats.n_rejected_work == 5 * rejects);
            }
        }
        auto const later = run(ba, dm, 1, 100000, 0.001, 1);
        AMREX_ALWAYS_ASSERT(!later.stats.failure && later.stats.n_retries == 1);
        AMREX_ALWAYS_ASSERT(
            std::abs(later.temperature - reference.temperature) < 2.e-7);
        AMREX_ALWAYS_ASSERT(std::abs(later.closure) < 3.e-12);
        auto const exhausted = run(ba, dm, 1, 8);
        AMREX_ALWAYS_ASSERT(exhausted.stats.failure &&
                            exhausted.stats.t_done < 0.02);
        AMREX_ALWAYS_ASSERT(
            std::string(exhausted.stats.failure).find("budget") !=
            std::string::npos);

        // A genuinely state-dependent ceiling: reheating lowers the ceiling
        // inside the super-step, while rollback restores the larger entry cap.
        {
            amrex::MultiFab state(ba, dm, 1, 0);
            state.setVal(1.0);
            RK::Auxiliary heat{{0.0}, {0.0}};
            auto rhs = [&] (amrex::MultiFab&, amrex::MultiFab& k) {
                k.setVal(1.0);
                heat.rate[0] = -1.0;
            };
            RK solver(
                RK::Scheme::RKL2, rhs, [&] () { return 1.0 / state.min(0); },
                1.e-7, 1.e-12, 0.9, 2.0, 1000, {}, nullptr, &heat, true);
            auto const st = solver.Advance(state, 0.8);
            AMREX_ALWAYS_ASSERT(!st.failure && st.n_retries > 0 &&
                                st.t_done == 0.8);
            AMREX_ALWAYS_ASSERT(std::abs(state.min(0) - 1.8) < 1.e-14);
            AMREX_ALWAYS_ASSERT(std::abs(state.min(0) - 1.0 + heat.value[0]) <
                                1.e-14);
            amrex::Print() << "RKL_STATE_DEPENDENT_PASS retries="
                           << st.n_retries << "\n";
        }

        // Nonfinite RHS wins over a simultaneous ceiling violation. Neither
        // fused nor separate validation may misclassify this as recoverable.
        for (bool fused : {false, true}) {
            amrex::MultiFab state(ba, dm, 1, 0);
            state.setVal(1.0);
            RK::Auxiliary heat{{7.0}, {0.0}};
            int calls = 0;
            bool finite = true;
            auto rhs = [&] (amrex::MultiFab& value, amrex::MultiFab& k) {
                ++calls;
                k.setVal(calls == 5 ? std::numeric_limits<Real>::infinity()
                                    : 1.0);
                heat.rate[0] = 1.0;
                finite = value.is_finite() && k.is_finite();
            };
            RK solver(
                RK::Scheme::RKL2, rhs,
                [&] () { return calls == 5 ? 1.e-8 : 0.001; }, 1.e-7, 1.e-12,
                0.9, 2.0, 1000, {}, nullptr, &heat, true,
                fused ? RK::StageValidity([&] () { return finite; })
                      : RK::StageValidity{},
                8);
            auto const st = solver.Advance(state, 0.02);
            AMREX_ALWAYS_ASSERT(st.failure &&
                                std::string(st.failure).find("nonfinite") !=
                                    std::string::npos);
            AMREX_ALWAYS_ASSERT(st.n_retries == 0 && st.n_rejected_work == 5 &&
                                st.n_attempts == 5);
            AMREX_ALWAYS_ASSERT(st.t_done == 0.0 && st.n_accepted == 0);
            AMREX_ALWAYS_ASSERT(state.min(0) == 1.0 && heat.value[0] == 7.0);
        }

        // Consecutive recoverable rejections must terminate independently of
        // the work budget. Persistently hostile ceilings cannot loop forever.
        amrex::MultiFab y(ba, dm, 1, 0);
        y.setVal(1.0);
        int stage = 0;
        Real bad_cap = 1.0;
        RK::AttemptHooks hooks;
        hooks.save = [&] () { stage = 0; };
        auto rhs = [&] (amrex::MultiFab&, amrex::MultiFab& k) {
            ++stage;
            k.setVal(0.0);
        };
        RK retries(
            RK::Scheme::RKL2, rhs,
            [&] () {
                if (stage == 2) {
                    bad_cap *= 0.1;
                    return bad_cap;
                }
                return 1.0;
            },
            1.e-7, 1.e-12, 0.9, 2.0, 100000, {}, nullptr, nullptr, true, {}, 2,
            hooks);
        auto const retry_limit = retries.Advance(y, 0.8);
        AMREX_ALWAYS_ASSERT(retry_limit.failure && retry_limit.t_done == 0.0);
        AMREX_ALWAYS_ASSERT(
            std::string(retry_limit.failure).find("retry budget") !=
            std::string::npos);
        AMREX_ALWAYS_ASSERT(retry_limit.n_attempts == 66 &&
                            retry_limit.n_accepted == 0);

        // Tiny positive time, and lack of representable progress after a
        // completed super-step, fail explicitly without a false acceptance.
        RK tiny(
            RK::Scheme::RKL2, rhs, [] () { return 1.0; }, 1.e-7, 1.e-12, 0.9,
            2.0, 100, {}, nullptr, nullptr, true);
        auto const underflow =
            tiny.Advance(y, std::numeric_limits<Real>::denorm_min());
        AMREX_ALWAYS_ASSERT(underflow.failure && underflow.t_done == 0.0);
        AMREX_ALWAYS_ASSERT(std::string(underflow.failure).find("underflow") !=
                            std::string::npos);
        stage = 0;
        RK no_progress(
            RK::Scheme::RKL2, rhs,
            [&] () {
                return stage <= 2 ? 1.0
                                  : std::numeric_limits<Real>::denorm_min();
            },
            1.e-7, 1.e-12, 0.9, 2.0, 100, {}, nullptr, nullptr, true, {}, 2);
        auto const stalled = no_progress.Advance(y, 2.0);
        AMREX_ALWAYS_ASSERT(stalled.failure && stalled.t_done > 0.0 &&
                            stalled.t_done < 2.0);
        AMREX_ALWAYS_ASSERT(std::string(stalled.failure).find("underflow") !=
                            std::string::npos);
        for (Real invalid : {-1.0, std::numeric_limits<Real>::infinity(),
                             std::numeric_limits<Real>::quiet_NaN()}) {
            auto const st = tiny.Advance(y, invalid);
            AMREX_ALWAYS_ASSERT(st.failure && st.n_attempts == 0);
        }
        amrex::Print() << "RKL_RETRY_PASS\n";
    }
    amrex::Finalize();
}
