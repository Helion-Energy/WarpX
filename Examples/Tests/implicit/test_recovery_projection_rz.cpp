/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/DarwinVacuumERecovery.H"
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
        bool const z_periodic = sim.Geom(0).isPeriodic(1);
        amrex::Real poloidal_scale = 1.;
        amrex::ParmParse("projection_test")
            .query("poloidal_scale", poloidal_scale);
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
                                          h * std::cos(4.0 * pi * j / nz)) +
                                     .2 * dt * r;
                    });
            }
            A[1]->OverrideSync(period);
            A[1]->FillBoundary(period);
            hp.ComputeVacuumARecovery(jac, dt);
            hp.ApplyVacuumFaradayE(dt, false, jac, false);
            amrex::MultiFab::Copy(output, *E[1], 0, 0, 1, 0);
        };
        // Repeated recovery must not alter the already recovered field.
        // This is a native operator check, independent of the particle solve.
        evaluate(0.0, false, base);
        amrex::MultiFab first(A[1]->boxArray(), A[1]->DistributionMap(), 1, 0);
        amrex::MultiFab change(first.boxArray(), first.DistributionMap(), 1, 0);
        auto B =
            sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp, 0);
        auto J = sim.m_fields.get_alldirs("hybrid_J_vac_fp", 0);
        auto vacuum_current = [&] () {
            A[1]->OverrideSync(period);
            A[1]->FillBoundary(period);
            sim.ApplyFieldBoundaryOnAxis(A[0], A[1], A[2], 0);
            sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(
                B, A, sim.GetEBUpdateBFlag()[0], 0);
            for (int d = 0; d < 3; ++d) {
                B[d]->setBndry(0.);
                B[d]->FillBoundary(period);
            }
            sim.ApplyFieldBoundaryOnAxis(B[0], B[1], B[2], 0);
            sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(
                J, B, sim.GetEBUpdateEFlag()[0], 0);
            for (amrex::MFIter mfi(change); mfi.isValid(); ++mfi) {
                auto dst = change.array(mfi);
                auto src = J[1]->const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        dst(i, j, k) = (i >= nr / 2 && i < nr &&
                                        (z_periodic || (j > 0 && j < nz)))
                                           ? src(i, j, k)
                                           : 0.;
                    });
            }
            return change.norminf();
        };
        // Nonzero poloidal traces expose accidental all-component updates.
        amrex::Array<amrex::MultiFab, 3> original;
        for (int d = 0; d < 3; ++d) {
            if (d != 1) {
                A[d]->setVal((d + 1) * 1.e-10 * poloidal_scale);
            }
            original[d].define(A[d]->boxArray(), A[d]->DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(original[d], *A[d], 0, 0, 1, 0);
        }
        // A recovery must not overwrite electric or current/history state.
        amrex::Vector<amrex::MultiFab*> protected_fields;
        for (auto field :
             {E, old, sim.m_fields.get_alldirs("hybrid_E_long_fp", 0)}) {
            for (auto* component : field) {
                protected_fields.push_back(component);
            }
        }
        for (auto const* key : {"hybrid_Je_n_nodal", "hybrid_Je_nm1_nodal",
                                "hybrid_Je_theta_nodal"}) {
            protected_fields.push_back(sim.m_fields.get(key, 0));
        }
        amrex::Vector<amrex::MultiFab> protected_copies(
            protected_fields.size());
        for (int n = 0; n < static_cast<int>(protected_fields.size()); ++n) {
            auto& f = *protected_fields[n];
            f.setVal(7. + n);
            protected_copies[n].define(f.boxArray(), f.DistributionMap(),
                                       f.nComp(), f.nGrowVect());
            amrex::MultiFab::Copy(protected_copies[n], f, 0, 0, f.nComp(),
                                  f.nGrowVect());
        }
        amrex::Real prior = vacuum_current();
        amrex::Real const initial_current = prior;
        bool legacy = false;
        amrex::ParmParse("projection_test").query("legacy", legacy);
        for (int pass = 0; pass < 4; ++pass) {
            amrex::MultiFab::Copy(first, *A[1], 0, 0, 1, 0);
            if (legacy) {
                hp.ComputeVacuumARecovery(false, dt);
            } else {
                RecoverDarwinVacuumA(sim, hp, A);
            }
            amrex::MultiFab::LinComb(change, 1., *A[1], 0, -1., first, 0, 0, 1,
                                     0);
            amrex::Real const relative_change =
                change.norminf() / first.norminf();
            amrex::Real const now = vacuum_current();
            amrex::Print() << "RECOVERY_PROJECTION pass=" << pass
                           << " relative_change=" << relative_change
                           << " vacuum_current=" << now
                           << " gain=" << now / prior << "\n";
            if (legacy) {
                AMREX_ALWAYS_ASSERT(now > 1.5 * prior);
            } else {
                AMREX_ALWAYS_ASSERT(now < 1.e-9 * initial_current);
                if (pass > 0) {
                    AMREX_ALWAYS_ASSERT(relative_change < 1.e-9);
                }
            }
            prior = now;
        }
        for (int d = 0; d < 3; ++d) {
            amrex::MultiFab fixed(original[d].boxArray(),
                                  original[d].DistributionMap(), 1, 0);
            for (amrex::MFIter mfi(fixed); mfi.isValid(); ++mfi) {
                auto f = fixed.array(mfi);
                auto a = A[d]->const_array(mfi),
                     before = original[d].const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        f(i, j, k) = (d != 1 || i < nr / 2 || i >= nr ||
                                      (!z_periodic && (j == 0 || j == nz)))
                                         ? a(i, j, k) - before(i, j, k)
                                         : 0.;
                    });
            }
            AMREX_ALWAYS_ASSERT(fixed.norminf() == 0.);
        }
        for (int n = 0; n < static_cast<int>(protected_fields.size()); ++n) {
            auto const& f = *protected_fields[n];
            auto& diff = protected_copies[n];
            amrex::MultiFab::Subtract(diff, f, 0, 0, f.nComp(), f.nGrowVect());
            AMREX_ALWAYS_ASSERT(diff.norminf(0, f.nComp(), f.nGrowVect()) ==
                                0.);
        }
        amrex::Print() << "RECOVERY_PROJECTION "
                          "fixed_trace_and_protected_state=unchanged\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
