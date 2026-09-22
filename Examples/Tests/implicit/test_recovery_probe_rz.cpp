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
        };
        bool expect_frozen = false;
        amrex::ParmParse("probe_test").query("expect_frozen", expect_frozen);
        for (amrex::Real h : {1.e-2, 1.e-4}) {
            evaluate(0.0, false, base);
            evaluate(h, true, probe);
            evaluate(h, false, full);
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
