/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
// Manufactured thermal-ODE stage test. Prescribed density, not an FRC
// trajectory.
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <cmath>
#include <iomanip>
using namespace amrex::literals;
using warpx::fields::FieldType;
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse p("audit");
        int steps = 8;
        amrex::Real n0 = 1.e20_rt, n1 = 2.e18_rt, interval = 1.e-5_rt;
        p.query("steps", steps);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& h = *w.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(h.DensityPedestal(0) == nullptr);
        auto& te =
            *w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rn = *w.m_fields.get(FieldType::rho_fp, 0);
        auto& ro = *w.m_fields.get(FieldType::hybrid_rho_fp_temp, 0);
        auto jp =
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto ve =
            w.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, 0);
        auto B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        for (int c = 0; c < 3; ++c) {
            ve[c]->setVal(0);
            jp[c]->setVal(0);
            B[c]->setVal(c == 2 ? 1._rt : 0);
            w.m_fields.get_alldirs(FieldType::current_fp, 0)[c]->setVal(0);
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_temp, 0)[c]
                ->setVal(0);
            w.m_fields.get_alldirs("current_fp_ions", 0)[c]->setVal(0);
            w.m_fields.get_alldirs("current_fp_ions2", 0)[c]->setVal(0);
        }
        auto const t0 = 100._rt * PhysConst::q_e / PhysConst::kb,
                   dt = interval / steps;
        auto const rate = std::log(n1 / n0) / interval, gm1 = h.m_gamma - 1._rt;
        te.setVal(t0);
        for (int s = 0; s < steps; ++s) {
            auto const na = n0 * std::exp(rate * s * dt),
                       nb = n0 * std::exp(rate * (s + 1) * dt);
            ro.setVal(PhysConst::q_e * na);
            rn.setVal(PhysConst::q_e * nb);
            w.m_fields.get("rho_fp_ions", 0)
                ->setVal(.6_rt * PhysConst::q_e * nb);
            w.m_fields.get("rho_fp_ions2", 0)
                ->setVal(.4_rt * PhysConst::q_e * nb);
            h.AdvanceElectronEnergyQDSMC_PC(dt);
        }
        // Exact solution of dT/dt=(gamma-1)*a*T+(gamma-1)*Q/(kB*n0*exp(a*t)).
        // Q=-S is a constant parser source; both densities stay above all
        // gates/floors. The complete PC driver dispatches
        // conduction/source/transport/source/conduction.
        auto const homogeneous = t0 * std::exp(gm1 * rate * interval);
        auto const heat = (gm1 * 1.e5_rt / (PhysConst::kb * n0)) *
                          std::exp(gm1 * rate * interval) *
                          (-std::expm1(-h.m_gamma * rate * interval)) /
                          (h.m_gamma * rate);
        auto const exact = homogeneous + heat;
        amrex::MultiFab delta(te.boxArray(), te.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(delta, te, 0, 0, 1, 0);
        delta.plus(-exact, 0, 1, 0);
        auto const error = delta.norminf();
        amrex::VisMF::Write(te, "temperature");
        amrex::Print() << std::setprecision(17)
                       << "PC_SOURCE_ORDER {\"steps\":" << steps
                       << ",\"stage_matched_receiver\":"
                       << (h.m_qdsmc_source_stage_density ? "true" : "false")
                       << ",\"exact_temperature_K\":" << exact
                       << ",\"exact_source_temperature_K\":" << heat
                       << ",\"final_temperature_max_K\":" << te.norminf()
                       << ",\"max_error_K\":" << error
                       << ",\"relative_source_error\":" << error / heat
                       << "}\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return 0;
}
