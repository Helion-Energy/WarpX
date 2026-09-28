/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        auto const period = sim.Geom(0).periodicity();
        auto const A = sim.m_fields.get_alldirs("hybrid_A_fp", 0);
        auto const old = sim.m_fields.get_alldirs("hybrid_A_old_fp", 0);
        auto const E =
            sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0);
        auto& rho = *sim.m_fields.get("hybrid_rho_vacmask_fp", 0);
        amrex::Real const floor = hp.m_n_floor * PhysConst::q_e;
        int const nr = sim.Geom(0).Domain().length(0);
        int const nz = sim.Geom(0).Domain().length(1);
        amrex::Real const pi = std::acos(-1.0);
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            auto const a = rho.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   a(i, j, k) = i < nr / 2 ? 2.0 * floor : 0.0;
                               });
        }
        for (int d = 0; d < 3; ++d) {
            old[d]->setVal(0.0);
        }
        amrex::MultiFab base(E[1]->boxArray(), E[1]->DistributionMap(), 1, 0);
        amrex::MultiFab probe(base.boxArray(), base.DistributionMap(), 1, 0);
        amrex::MultiFab full(base.boxArray(), base.DistributionMap(), 1, 0);
        amrex::MultiFab replay(base.boxArray(), base.DistributionMap(), 1, 0);
        amrex::Real const radius = sim.Geom(0).ProbHi(0);
        amrex::Real const dz = sim.Geom(0).CellSize(1);
        amrex::Real const dr = sim.Geom(0).CellSize(0);
        amrex::Real const inverse_scale = 4.0 / (dr * dr) + 4.0 / (dz * dz);
        constexpr amrex::Real dt = 1.e-9;
        auto evaluate = [&] (amrex::Real h, bool jac, amrex::MultiFab& output) {
            for (int d = 0; d < 3; ++d) {
                A[d]->setVal(0.0);
                E[d]->setVal(0.0);
            }
            for (amrex::MFIter mfi(*A[1]); mfi.isValid(); ++mfi) {
                auto const a = A[1]->array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        amrex::Real const r = static_cast<amrex::Real>(i) / nr;
                        amrex::Real const shape = r * (1.0 - r * r);
                        a(i, j, k) = -dt * shape *
                                     (std::sin(2.0 * pi * j / nz) +
                                      h * std::cos(4.0 * pi * j / nz));
                    });
            }
            A[1]->OverrideSync(period);
            A[1]->FillBoundary(period);
            hp.ComputeVacuumARecovery(jac, dt);
            hp.ApplyVacuumFaradayE(dt, false, jac, false);
            amrex::MultiFab::Copy(output, *E[1], 0, 0, 1, 0);
            if (hp.m_darwin_vacuum_recovery_operator == "edge_relaxation") {
                // Independent cylindrical oracle: native radial curls are
                // exact on r*(1-r^2/R^2), with -L_r(shape)=8*r/R^2.
                // The periodic axial second difference has its known sine
                // eigenvalue. Physical trace rows are restored by the caller
                // and are excluded from this direct interior-operator check.
                amrex::MultiFab error(base.boxArray(), base.DistributionMap(), 1, 0);
                for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                    auto const diff = error.array(mfi);
                    auto const value = output.const_array(mfi);
                    amrex::ParallelFor(mfi.tilebox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            diff(i, j, k) = 0.0;
                            if (i <= 0 || i >= nr) { return; }
                            amrex::Real expected = 0.0;
                            if (i >= nr / 2) {
                                amrex::Real const r = static_cast<amrex::Real>(i) / nr;
                                amrex::Real const shape = r * (1.0 - r * r);
                                for (int mode = 1; mode <= 2; ++mode) {
                                    amrex::Real const a = pi * mode / nz;
                                    amrex::Real const lambda = 4.0 * std::sin(a) *
                                        std::sin(a) / (dz * dz);
                                    amrex::Real const wave = mode == 1
                                        ? std::sin(2.0 * a * j)
                                        : h * std::cos(2.0 * a * j);
                                    expected += (shape - (8.0 * r / (radius * radius) +
                                        lambda * shape) / inverse_scale) * wave;
                                }
                            }
                            diff(i, j, k) = value(i, j, k) - expected;
                        });
                }
                amrex::Real const defect = error.norminf();
                amrex::Print() << "RECOVERY_EDGE_ORACLE h=" << h
                               << " absolute_defect=" << defect << "\n";
                AMREX_ALWAYS_ASSERT(defect < 1.e-12);
            }
        };
        bool expect_frozen = false;
        amrex::ParmParse("probe_test").query("expect_frozen", expect_frozen);
        for (amrex::Real h : {1.e-2, 1.e-4}) {
            evaluate(0.0, false, base);
            evaluate(h, true, probe);
            evaluate(h, false, full);
            evaluate(0.0, false, replay);
            amrex::MultiFab::Subtract(replay, base, 0, 0, 1, 0);
            amrex::Real const replay_defect = replay.norminf();
            amrex::Print() << "RECOVERY_REPLAY absolute_defect=" << replay_defect << "\n";
            AMREX_ALWAYS_ASSERT(replay_defect < 1.e-12);
            amrex::MultiFab::Subtract(probe, full, 0, 0, 1, 0);
            amrex::MultiFab::Subtract(full, base, 0, 0, 1, 0);
            amrex::Real const signal = full.norminf();
            amrex::Real const defect = probe.norminf() / signal;
            amrex::Print() << "RECOVERY_PROBE h=" << h << " signal=" << signal
                           << " relative_map_defect=" << defect << "\n";
            AMREX_ALWAYS_ASSERT(signal > 1.e-3 * h);
            if (expect_frozen) {
                AMREX_ALWAYS_ASSERT(defect > 0.1);
            } else {
                AMREX_ALWAYS_ASSERT(defect < 1.e-7);
            }
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
