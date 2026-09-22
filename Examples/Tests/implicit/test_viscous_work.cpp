/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>

namespace {
using namespace amrex::literals;
using amrex::MultiFab;
using warpx::fields::FieldType;
using Field = ablastr::fields::VectorField;

amrex::Real
integral (MultiFab const& a, int comp, amrex::Geometry const& geom) {
    auto const vol = MakeQdsmcVolumeElement(geom, a.ixType());
    MultiFab weighted(a.boxArray(), a.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(weighted); mfi.isValid(); ++mfi) {
        auto const u = a.const_array(mfi);
        auto const v = weighted.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               v(i, j, k) = vol(i, j, k) * u(i, j, k, comp);
                           });
    }
    return weighted.sum_unique(0, false, geom.periodicity());
}

void
require_close (amrex::Real a, amrex::Real b, char const* label,
               amrex::Real tol = 2.e-10) {
    amrex::Real const relative =
        std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1.e-30_rt});
    amrex::Print() << "VISCOUS_WORK " << label << " relative=" << relative
                   << " measured=" << a << " expected=" << b << '\n';
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(relative) && relative < tol,
                                     label);
}
} // namespace

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse hp_inputs("hybrid_pic_model");
        hp_inputs.add("solve_electron_energy_equation", 1);
        hp_inputs.add("include_joule_heating", 0);
        hp_inputs.add("include_hall_term", 0);
        hp_inputs.add("include_electron_pressure_term", 0);
        hp_inputs.add("elec_temp", 100.0);
        hp_inputs.add("n_floor", 1.e17);
        hp_inputs.add("qdsmc_viscosity_model", std::string("parser"));
        hp_inputs.add("qdsmc_viscosity_in_ohms_law", 1);
        hp_inputs.add("qdsmc_viscosity_limiter", std::string("none"));
        hp_inputs.add("qdsmc_viscosity_heating", std::string("work"));
        hp_inputs.add("qdsmc_nu_par(n,Te,B)", std::string("1.e4*Te/100"));
        hp_inputs.add("qdsmc_nu_perp(n,Te,B)", std::string("3.e3*Te/100"));
        hp_inputs.add("qdsmc_viscosity_flux_limit_factor", 0.0);
        hp_inputs.add("density_pedestal", 1);
        int boundary_mode = 0;
        amrex::ParmParse("viscous_test").query("boundary_mode", boundary_mode);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const lo = geom.ProbLoArray();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> length{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            length[d] = geom.ProbHi(d) - geom.ProbLo(d);
        }
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto& Te =
            *w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto* ped = const_cast<MultiFab*>(hp.DensityPedestal(0));
        AMREX_ALWAYS_ASSERT(ped != nullptr);
        ped->setVal(0.2_rt * 1.e19_rt * PhysConst::q_e);
        constexpr amrex::Real T0 = 100. * PhysConst::q_e / PhysConst::kb;
        Te.setVal(T0);
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            auto const a = rho.array(mfi);
            auto const nc = rho.nComp();
            amrex::ParallelFor(
                mfi.fabbox(), nc,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    amrex::Real const x = lo[0] + i * dx[0];
                    a(i, j, k, n) =
                        1.e19_rt * PhysConst::q_e *
                        (1.2_rt + 0.4_rt * std::cos(2 * MathConst::pi * x));
                });
        }
        Field J =
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        Field const Ji = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        Field const B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        Field const E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        std::array<MultiFab, 3> ev, full, push, before;
        for (int c = 0; c < 3; ++c) {
            Ji[c]->setVal(0.0);
            B[c]->setVal(c == 2 ? 1.0 : 0.0);
            auto const ix = J[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*J[c]); mfi.isValid(); ++mfi) {
                auto const a = J[c]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    amrex::Real const x =
                        lo[0] + (i + (ix[0] ? 0. : .5)) * dx[0];
#if defined(WARPX_DIM_RZ)
                    amrex::Real const z =
                        lo[1] + (j + (ix[1] ? 0. : .5)) * dx[1];
                    amrex::Real const r = x;
                    amrex::Real const env =
                        (boundary_mode ? 0.4_rt : 0.0_rt) +
                        std::pow(std::sin(MathConst::pi * r / length[0]), 4);
                    a(i, j, k) =
                        1.e6_rt * env * (c == 2 ? 1.0_rt : r) *
                        (std::sin(2 * MathConst::pi * z / length[1]) +
                         .3_rt * std::cos(4 * MathConst::pi * z / length[1]));
#else
                    amrex::Real const y=lo[1]+(j+(ix[1]?0.:.5))*dx[1];
                    amrex::Real const z=lo[2]+(k+(ix[2]?0.:.5))*dx[2];
                    a(i,j,k)=1.e6_rt*(c+1)*std::sin(2*MathConst::pi*x)*
                        (std::cos(2*MathConst::pi*y)+.3_rt*std::sin(4*MathConst::pi*z));
#endif
                });
            }
            for (auto* f : {&ev[c], &full[c], &push[c], &before[c]}) {
                f->define(E[c]->boxArray(), E[c]->DistributionMap(), 1,
                          E[c]->nGrowVect());
                f->setVal(0.0);
            }
        }
        Field EV{&ev[0], &ev[1], &ev[2]}, EF{&full[0], &full[1], &full[2]},
            EP{&push[0], &push[1], &push[2]};
        auto const& Pe =
            *w.m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        auto solve = [&] (Field const& target, bool dissipative,
                          Field const* mirror) {
            w.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(
                target, J, Ji, B, rho, Pe, w.GetEBUpdateEFlag()[0], 0, &hp,
                false, dissipative, nullptr, mirror);
        };
        solve(EF, true, &EV);
        solve(EP, false, nullptr);
        amrex::Real work = 0;
        for (int c = 0; c < 3; ++c) {
            MultiFab::Copy(before[c], ev[c], 0, 0, 1, 0);
            MultiFab::Subtract(full[c], push[c], 0, 0, 1, 0);
            MultiFab::Subtract(full[c], ev[c], 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(full[c].norminf(0) <
                                1.e-12_rt * std::max(ev[c].norminf(0), 1._rt));
            MultiFab product(J[c]->boxArray(), J[c]->DistributionMap(), 1, 0);
            MultiFab::Copy(product, *J[c], 0, 0, 1, 0);
            MultiFab::Multiply(product, ev[c], 0, 0, 1, 0);
            work += integral(product, 0, geom);
        }
        AMREX_ALWAYS_ASSERT(work > 0.0);
        hp.QDSMCAddViscousHeating(0, 0.0, rho, false);
        auto const& qnu = *w.m_fields.get("hybrid_qdsmc_visc_heating_fp", 0);
        require_close(integral(qnu, 0, geom), work, "stress adjoint");

        // Endpoint heat capacity can differ from the force's midpoint density.
        MultiFab capacity(rho.boxArray(), rho.DistributionMap(), 1,
                          rho.nGrowVect());
        MultiFab::Copy(capacity, rho, 0, 0, 1, rho.nGrowVect());
        capacity.mult(1.7);
        MultiFab q(Te.boxArray(), Te.DistributionMap(), 1, 0);
        amrex::Real clamp = 0.;
        amrex::Real const dt = 1.e-7;
        hp.QDSMCDepositDragWork(0, dt, rho, EV, q, clamp, true, true,
                                &capacity);
        require_close(integral(q, 0, geom), work, "edge work partition");
        MultiFab heat(Te.boxArray(), Te.DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(heat); mfi.isValid(); ++mfi) {
            auto const a = heat.array(mfi);
            auto const t = Te.const_array(mfi), n = capacity.const_array(mfi),
                       p = ped->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                a(i, j, k) = 1.5_rt * (n(i, j, k) + p(i, j, k)) /
                             PhysConst::q_e * PhysConst::kb * (t(i, j, k) - T0);
            });
        }
        require_close(integral(heat, 0, geom), dt * work,
                      "heat capacity and pedestal");
        AMREX_ALWAYS_ASSERT(clamp == 0.0);

        // A/B/A coefficient trials: neither live current nor the overwritten
        // temperature may alter a frozen viscous stage.
        Te.setVal(T0);
        hp.FreezeImplicitViscousCurrent();
        hp.CaptureImplicitDissipationCoefficients();
        for (int c = 0; c < 3; ++c) {
            MultiFab::Copy(*Ji[c], *J[c], 0, 0, 1, Ji[c]->nGrowVect());
            Ji[c]->mult(2.0);
        }
        B[0]->setVal(0.7);
        B[2]->setVal(0.2);
        Te.setVal(.5 * T0);
        solve(EF, true, &EV);
        for (int c = 0; c < 3; ++c) {
            MultiFab::Subtract(ev[c], before[c], 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(ev[c].norminf(0) <
                                1.e-12_rt *
                                    std::max(before[c].norminf(0), 1._rt));
        }
        hp.ReleaseImplicitViscousStage();
        // A negative work trial forces the floor clamp. Probe evaluation
        // must not increment the accepted-energy ledger.
        amrex::Real const old_clamp = hp.m_visc_clamp_J;
        for (int trial = 0; trial < 3; ++trial) {
            Te.setVal(T0);
            hp.QDSMCAddViscousDragWork(0, 1.e3, rho, false);
            AMREX_ALWAYS_ASSERT(hp.m_visc_clamp_J == old_clamp);
        }
        Te.setVal(T0);
        hp.QDSMCAddViscousDragWork(0, 1.e3, rho, true);
        AMREX_ALWAYS_ASSERT(hp.m_visc_clamp_J > old_clamp);
        amrex::Print()
            << "VISCOUS_WORK trial ledgers and stage freshness PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
