/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <cmath>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        using ablastr::fields::Direction;
        auto& hybrid = *sim.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(sim.getistep(0) == 0 && sim.gett_new(0) == 0.);
        AMREX_ALWAYS_ASSERT(hybrid.m_inertia_history_initialized);
        AMREX_ALWAYS_ASSERT(hybrid.m_inertia_history_levels == 1);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const lo = geom.ProbLoArray();
        auto const hi = geom.Domain().bigEnd();
        auto const E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto const B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto const Ji = sim.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto const& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& Pe = *sim.m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        auto const& history = *sim.m_fields.get("hybrid_Je_n_nodal", 0);
        amrex::Real density_pedestal = 0.0;
        amrex::ParmParse("startup_test").query("density_pedestal", density_pedestal);
        // Bz = 0.13 + 0.2*x, including the initial external field exactly
        // once. Away from physical boundaries, Jy = -0.2/mu0 exactly on
        // either mesh, and E_x = (Jy*Bz - dPe/dx)/rho on its Yee stagger.
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                         amrex::ReduceOpMax> op;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(*E[0]); mfi.isValid(); ++mfi) {
            auto const e = E[0]->const_array(mfi), r = rho.const_array(mfi),
                       p = Pe.const_array(mfi), je = history.const_array(mfi);
            op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    bool interior = i >= 3 && i <= hi[0]-3 && j >= 3 && j <= hi[1]-3;
#if defined(WARPX_DIM_3D)
                    interior = interior && k >= 2 && k <= hi[2]-2;
#endif
                    if (!interior) { return {0., 0., 0.}; }
                    amrex::Real const x = lo[0] + (i+.5)*dx[0];
                    amrex::Real const current = -.2/PhysConst::mu0;
                    amrex::Real const charge = .5*(r(i,j,k)+r(i+1,j,k))
                        + PhysConst::q_e*density_pedestal;
                    amrex::Real const expected =
                        (current*(.13+.2*x) - (p(i+1,j,k)-p(i,j,k))/dx[0])/charge;
                    return {std::abs(e(i,j,k)-expected), std::abs(expected),
                            std::abs(je(i,j,k,1)-current)};
                });
        }
        auto value = data.value(op);
        amrex::Real v[3]{amrex::get<0>(value), amrex::get<1>(value), amrex::get<2>(value)};
        amrex::ParallelDescriptor::ReduceRealMax(v, 3);
        amrex::Print() << "STARTUP_ANALYTIC error=" << v[0] << " scale=" << v[1]
                       << " Je_error=" << v[2] << "\n";
        AMREX_ALWAYS_ASSERT(v[1] > 1.e3 && v[0] < 1.e-10*v[1]);
        AMREX_ALWAYS_ASSERT(v[2] < 1.e-10*.2/PhysConst::mu0);
        bool pressure_case = false;
        amrex::ParmParse("startup_test").query("pressure", pressure_case);
        if (pressure_case) {
            auto const longitudinal = sim.m_fields.get_alldirs("hybrid_E_long_fp", 0);
            AMREX_ALWAYS_ASSERT(longitudinal[2]->norminf() > .1);
            // Re-deposit the initial moving ions with an independently supplied
            // zero time offset. The old explicit half-step offset changes this
            // nonuniform current by O(dt), so it cannot satisfy this check.
            for (int d=0; d<3; ++d) {
                sim.m_fields.alloc_init("startup_current_reference", Direction{d}, 0,
                    Ji[d]->boxArray(), Ji[d]->DistributionMap(), 1, Ji[d]->nGrowVect(), 0.);
            }
            auto const reference = sim.m_fields.get_alldirs("startup_current_reference", 0);
            auto& ions = sim.GetPartContainer().GetParticleContainerFromName("ions");
            ions.DepositCurrent("startup_current_reference", 0, sim.getdt(0), 0.);
            for (int d=0; d<3; ++d) {
                amrex::Real const scale = std::max(Ji[d]->norminf(), 1.);
                amrex::MultiFab::Subtract(*reference[d], *Ji[d], 0, 0, 1, 0);
                amrex::Print() << "STARTUP_CURRENT dir=" << d << " error="
                               << reference[d]->norminf() << " scale=" << scale << "\n";
                AMREX_ALWAYS_ASSERT(reference[d]->norminf() < 1.e-11*scale);
            }
        }
        bool temperature_case = false;
        amrex::ParmParse startup("startup_test");
        startup.query("temperature", temperature_case);
        auto const& Te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        amrex::MultiFab saved_Te(Te.boxArray(), Te.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(saved_Te, Te, 0, 0, 1, 0);
        if (temperature_case) {
            // Check the prescribed initial profile independently of the model
            // members, including nodes below the distinct initialization floor.
            amrex::Real gamma_init = 5.0/3.0, floor_init = 1.e14;
            startup.query("gamma_init", gamma_init);
            startup.query("floor_init", floor_init);
            bool include_pedestal = true;
            startup.query("include_pedestal", include_pedestal);
            amrex::Real const seed_pedestal = include_pedestal ? density_pedestal : 0.0;
            AMREX_ALWAYS_ASSERT(hybrid.m_gamma == 5.0/3.0);
            AMREX_ALWAYS_ASSERT(hybrid.m_qdsmc_te_n_floor == 1.e14);
            AMREX_ALWAYS_ASSERT(hybrid.m_qdsmc_te_seeded);
            amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                             amrex::ReduceOpMax> temperature_op;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real>
                temperature_data(temperature_op);
            using TemperatureTuple = decltype(temperature_data)::Type;
            for (amrex::MFIter mfi(Te); mfi.isValid(); ++mfi) {
                auto const te = Te.const_array(mfi), r = rho.const_array(mfi),
                           p = Pe.const_array(mfi);
                temperature_op.eval(mfi.validbox(), temperature_data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> TemperatureTuple {
                        amrex::Real const density = amrex::max(
                            r(i,j,k)/PhysConst::q_e + seed_pedestal, floor_init);
                        amrex::Real const expected = 10.0*PhysConst::q_e/PhysConst::kb
                            * std::pow(density/1.e18, gamma_init-1.0);
                        bool interior = i >= 3 && i <= hi[0]-3 && j >= 3 && j <= hi[1]-3;
#if defined(WARPX_DIM_3D)
                        interior = interior && k >= 2 && k <= hi[2]-2;
#endif
                        amrex::Real const expected_pressure = amrex::max(
                            r(i,j,k)/PhysConst::q_e + density_pedestal, 1.e12)*PhysConst::kb*expected;
                        return {std::abs(te(i,j,k)-expected)/expected,
                                interior ? std::abs(p(i,j,k)-expected_pressure)/expected_pressure
                                         : 0.0,
                                r(i,j,k)/PhysConst::q_e < floor_init ? 1.0 : 0.0};
                    });
            }
            auto const tv = temperature_data.value(temperature_op);
            amrex::Real check[3]{amrex::get<0>(tv), amrex::get<1>(tv), amrex::get<2>(tv)};
            amrex::ParallelDescriptor::ReduceRealMax(check, 3);
            amrex::Print() << "STARTUP_TEMPERATURE relative_error=" << check[0]
                           << " pressure_error=" << check[1] << " floor_exercised=" << check[2] << "\n";
            AMREX_ALWAYS_ASSERT(check[0] < 1.e-12 && check[1] < 1.e-12);
            if (floor_init > 1.e14) { AMREX_ALWAYS_ASSERT(check[2] == 1.0); }
        }
        auto const& inertia = *sim.m_fields.get("hybrid_E_inertial_nodal", 0);
        for (int d=0; d<3; ++d) {
            AMREX_ALWAYS_ASSERT(inertia.norminf(d) == 0.);
            auto const& old = *sim.m_fields.get(FieldType::E_old, Direction{d}, 0);
            amrex::MultiFab error(old.boxArray(), old.DistributionMap(), 1, 0);
            amrex::MultiFab::LinComb(error, 1., old, 0, -1., *E[d], 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(error.norminf() == 0.);
        }
        // A repeated Evolve-entry deposit must not add the coil field again
        // or replace the initialized E/history.
        amrex::Array<amrex::MultiFab,3> saved_B, saved_E;
        for (int d=0; d<3; ++d) {
            saved_B[d].define(B[d]->boxArray(), B[d]->DistributionMap(), 1, 0);
            saved_E[d].define(E[d]->boxArray(), E[d]->DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(saved_B[d], *B[d], 0, 0, 1, 0);
            amrex::MultiFab::Copy(saved_E[d], *E[d], 0, 0, 1, 0);
        }
        amrex::MultiFab saved(history.boxArray(), history.DistributionMap(), 3, 1);
        amrex::MultiFab::Copy(saved, history, 0, 0, 3, 1);
        if (temperature_case) {
            auto& evolved_Te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
            evolved_Te.mult(1.125, 0, 1, 0);
            saved_Te.mult(1.125, 0, 1, 0);
        }
        sim.HybridPICInitializeRhoJandB();
        if (temperature_case) {
            // Only the evolved-temperature path owns Te as independent state.
            // The algebraic closure re-emits its diagnostic Te on re-entry.
            amrex::MultiFab::Subtract(saved_Te, Te, 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(saved_Te.norminf() == 0.0);
        }
        for (int d=0; d<3; ++d) {
            amrex::MultiFab::Subtract(saved_B[d], *B[d], 0, 0, 1, 0);
            amrex::MultiFab::Subtract(saved_E[d], *E[d], 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(saved_B[d].norminf() == 0. && saved_E[d].norminf() == 0.);
        }
        // Perturbed probes, nonlinear trials, and a second initialization
        // request cannot replace the committed current reference.
        auto& trial_rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        amrex::MultiFab::Copy(trial_rho, trial_rho, 0, trial_rho.nComp()/2,
                              1, trial_rho.nGrowVect());
        Ji[0]->plus(100., 0, 1, 1);
        hybrid.InitializeElectronInertiaHistory();
        amrex::MultiFab first(inertia.boxArray(), inertia.DistributionMap(), 3, 1);
        bool first_probe = true;
        for (bool jacobian : {true, false, true}) {
            hybrid.ComputeElectronInertiaNodal(.5, sim.getdt(0), jacobian);
            amrex::MultiFab error(history.boxArray(), history.DistributionMap(), 3, 1);
            amrex::MultiFab::LinComb(error, 1., saved, 0, -1., history, 0, 0, 3, 1);
            for (int d=0; d<3; ++d) { AMREX_ALWAYS_ASSERT(error.norminf(d, 1) == 0.); }
            if (first_probe) {
                amrex::MultiFab::Copy(first, inertia, 0, 0, 3, 1);
                first_probe = false;
            }
            else {
                amrex::MultiFab::LinComb(error, 1., first, 0, -1., inertia, 0, 0, 3, 1);
                for (int d=0; d<3; ++d) { AMREX_ALWAYS_ASSERT(error.norminf(d, 1) == 0.); }
            }
        }
        AMREX_ALWAYS_ASSERT(inertia.norminf(0) > 0.);
        AMREX_ALWAYS_ASSERT(hybrid.m_inertia_history_levels == 1);
        amrex::Print() << "DARWIN_INITIAL_STATE_PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
