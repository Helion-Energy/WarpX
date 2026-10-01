/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcRKIntegrator.H"
#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <limits>

int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::BoxArray ba(amrex::Box(amrex::IntVect(0), amrex::IntVect(0)));
        amrex::DistributionMapping dm(ba);
        amrex::MultiFab y(ba, dm, 1, 0);
        using RK = QdsmcRKIntegrator;
        for (auto const scheme :
             {RK::Scheme::SSPRK2, RK::Scheme::RKF45, RK::Scheme::RKL2}) {
            for (amrex::Real const ceiling : {10.0, 0.01}) {
                y.setVal(1.0);
                RK::Auxiliary heat{{0.0, 0.0}, {0.0, 0.0}};
                amrex::Real raw_floor = 0.0;
                auto rhs = [&] (amrex::MultiFab&, amrex::MultiFab& k) {
                    k.setVal(-1.0);
                    heat.rate[0] = 1.0;
                    heat.rate[1] = 0.0;
                };
                auto project = [&] (amrex::MultiFab& state, amrex::Real) {
                    amrex::Real const delta = amrex::max(-state.min(0), 0.0);
                    if (delta > 0) {
                        state.setVal(0.0);
                    }
                    heat.value[1] += delta;
                    raw_floor += delta;
                };
                RK rk(
                    scheme, rhs, [=] () { return ceiling; }, 1.e-6, 1.e-10, 0.9,
                    2.0, 100000, project, nullptr, &heat, true);
                auto const st = rk.Advance(y, 2.0);
                AMREX_ALWAYS_ASSERT(!st.failure &&
                                    std::abs(st.t_done - 2.0) < 1.e-12);
                AMREX_ALWAYS_ASSERT(std::abs(y.min(0) - 1 + heat.value[0] -
                                             heat.value[1]) < 1.e-12);
                AMREX_ALWAYS_ASSERT(std::abs(heat.value[0] - 2) < 1.e-12);
                amrex::Print()
                    << "RK_ACCOUNT scheme=" << int(scheme) << " s=" << st.s_max
                    << " floor=" << heat.value[1] << " raw_floor=" << raw_floor
                    << "\n";
            }
        }
        // Nonlinear error rejection must discard provisional quadrature.
        for (auto const scheme : {RK::Scheme::SSPRK2, RK::Scheme::RKF45}) {
            y.setVal(1.0);
            RK::Auxiliary heat{{0.0}, {0.0}};
            auto rhs = [&] (amrex::MultiFab& state, amrex::MultiFab& k) {
                heat.rate[0] = state.min(0);
                amrex::MultiFab::Copy(k, state, 0, 0, 1, 0);
                k.mult(-1.0);
            };
            RK rk(
                scheme, rhs, [] () { return 10.0; }, 1.e-7, 1.e-12, 0.9, 2.0,
                100000, {}, nullptr, &heat, true);
            auto const st = rk.Advance(y, 1.0);
            AMREX_ALWAYS_ASSERT(!st.failure && st.n_attempts > st.n_accepted);
            AMREX_ALWAYS_ASSERT(std::abs(y.min(0) - 1 + heat.value[0]) <
                                1.e-12);
            AMREX_ALWAYS_ASSERT(std::abs(y.min(0) - std::exp(-1.0)) < 1.e-6);
        }
        // A finite but newly insufficient ceiling is a failure, not success.
        for (amrex::Real const bad :
             {0.0, 1.e-8, std::numeric_limits<amrex::Real>::quiet_NaN()}) {
            y.setVal(1.0);
            int calls = 0;
            RK::Auxiliary heat{{7.0}, {0.0}};
            auto rhs = [&] (amrex::MultiFab&, amrex::MultiFab& k) {
                ++calls;
                k.setVal(1.0);
                heat.rate[0] = 1.0;
            };
            RK rk(
                RK::Scheme::RKL2, rhs, [&] () { return calls < 2 ? 1.0 : bad; },
                1.e-6, 1.e-10, 0.9, 2.0, 100, {}, nullptr, &heat, true);
            auto const st = rk.Advance(y, 0.8);
            AMREX_ALWAYS_ASSERT(st.failure && st.t_done == 0.0 &&
                                heat.value[0] == 7.0);
        }
        amrex::Print() << "RK_ACCOUNTING_PASS\n";
    }
    amrex::Finalize();
}
