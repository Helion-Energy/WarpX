/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
// Native source / entropy-remap capacity audit; no particle or field evolution.
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <cmath>
#include <iomanip>
#include <limits>
using namespace amrex::literals;
using warpx::fields::FieldType;
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    int result = 0;
    {
        amrex::ParmParse p("audit");
        amrex::Real nold = 1.e19_rt, nnew = 1.e19_rt, nped = 1.e18_rt,
                    dt = 1.e-5_rt;
        bool heat = true, old_receiver = false;
        amrex::Real eta_factor = 1._rt;
        p.query("nold", nold);
        p.query("nnew", nnew);
        p.query("nped", nped);
        p.query("dt", dt);
        p.query("heat", heat);
        p.query("old_receiver", old_receiver);
        p.query("eta_factor", eta_factor);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& h = *w.get_pointer_HybridPICModel();
        auto& te =
            *w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rn = *w.m_fields.get(FieldType::rho_fp, 0);
        auto& ro = *w.m_fields.get(FieldType::hybrid_rho_fp_temp, 0);
        auto& wt = *w.m_fields.get(FieldType::hybrid_qdsmc_weights_fp, 0);
        auto* pedestal = const_cast<amrex::MultiFab*>(h.DensityPedestal(0));
        if (pedestal)
            pedestal->setVal(nped * PhysConst::q_e);
        else
            nped = 0;
        rn.setVal(nnew * PhysConst::q_e);
        ro.setVal(nold * PhysConst::q_e);
        w.m_fields.get("rho_fp_ions", 0)->setVal(.6_rt * nnew * PhysConst::q_e);
        w.m_fields.get("rho_fp_ions2", 0)
            ->setVal(.4_rt * nnew * PhysConst::q_e);
        auto ve =
            w.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, 0);
        auto jp =
            w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        for (int c = 0; c < 3; ++c) {
            ve[c]->setVal(0);
            jp[c]->setVal(c == 2 ? 1.e5_rt : 0);
            B[c]->setVal(c == 2 ? 1._rt : 0);
            w.m_fields.get_alldirs("current_fp_ions", 0)[c]->setVal(0);
            w.m_fields.get_alldirs("current_fp_ions2", 0)[c]->setVal(0);
        }
        auto const& g = w.Geom(0);
        auto const dx = g.CellSizeArray();
        auto const lo = g.ProbLoArray();
        auto const R = g.ProbHi(0);
        auto const zhi = g.Domain().bigEnd(1) + 1;
        constexpr amrex::Real t0 = 100._rt * PhysConst::q_e / PhysConst::kb;
        amrex::MultiFab source_te(te.boxArray(), te.DistributionMap(), 1,
                                  te.nGrowVect());
        amrex::MultiFab control_te(te.boxArray(), te.DistributionMap(), 1, 0);
        amrex::MultiFab out(te.boxArray(), te.DistributionMap(), 7, 0);
        // Unheated native remap control, then an independent replay with
        // heating.
        te.setVal(t0);
        h.QdsmcTransportOnce(0, 0._rt, true);
        amrex::MultiFab::Copy(control_te, te, 0, 0, 1, 0);
        h.m_qdsmc_pc->ResetParticles(0);
        te.setVal(t0);
        if (heat) {
            if (old_receiver)
                h.ApplyQdsmcEnergySources(0, dt, true, ro);
            else
                h.ApplyQdsmcEnergySources(0, dt, true);
        }
        amrex::MultiFab::Copy(source_te, te, 0, 0, 1, te.nGrowVect());
        h.QdsmcTransportOnce(0, 0._rt, true);
        amrex::Real const conversion_floor = h.m_n_floor;
        auto const eff_old = amrex::max(nold + nped, conversion_floor);
        auto const eff_new = amrex::max(nnew + nped, conversion_floor);
        auto const gm1 = h.m_gamma - 1._rt;
        auto const cold = eff_old * PhysConst::kb / gm1,
                   cnew = eff_new * PhysConst::kb / gm1;
        auto const gate = amrex::max(h.m_joule_heating_n_min, h.m_n_floor);
        auto const heating_density =
            nnew; // physical coefficient/gate density is unchanged
        auto const q = heat && heating_density > gate
                           ? dt * 1.e-5_rt * 1.e10_rt * eta_factor
                           : 0._rt;
        auto const receiver_capacity = old_receiver ? cold : cnew;
        auto const expected_dT = q / receiver_capacity;
        auto const recover_factor =
            nold > 0 ? (nped + nold * std::pow(eff_new / eff_old, gm1)) /
                           (nped + nold)
                     : 1._rt;
        auto const te_expected = (t0 + expected_dT) * recover_factor;
        for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
            auto const a = out.array(mfi);
            auto const s = source_te.const_array(mfi), t = te.const_array(mfi);
            auto const c = control_te.const_array(mfi),
                       weights = wt.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    auto const r = lo[0] + i * dx[0];
                    auto const rl = amrex::max(0._rt, r - .5_rt * dx[0]),
                               rh = amrex::min(R, r + .5_rt * dx[0]);
                    auto const volume =
                        j == zhi ? 0._rt
                                 : MathConst::pi * (rh * rh - rl * rl) * dx[1];
                    a(i, j, k, 0) = volume * cold * (s(i, j, k) - t0);
                    a(i, j, k, 1) = volume * cnew * (s(i, j, k) - t0);
                    a(i, j, k, 2) = volume * cnew * (t(i, j, k) - c(i, j, k));
                    a(i, j, k, 3) = std::abs(s(i, j, k) - (t0 + expected_dT));
                    a(i, j, k, 4) = std::abs(t(i, j, k) - te_expected);
                    a(i, j, k, 5) = std::abs(weights(i, j, k) - nold) /
                                    (nold + nped + 1._rt);
                    a(i, j, k, 6) = std::abs(c(i, j, k) - t0 * recover_factor);
                });
        }
        auto const volume = MathConst::pi * R * R * g.ProbLength(1);
        auto const input = q * volume;
        auto const old_energy = out.sum_unique(0, false, g.periodicity());
        auto const new_energy = out.sum_unique(1, false, g.periodicity());
        auto const recovered = out.sum_unique(2, false, g.periodicity());
        auto const eps = std::numeric_limits<amrex::Real>::epsilon();
        auto const bound =
            512._rt * eps * volume * amrex::max(cold, cnew) * t0 +
            2.e-10_rt * input;
        bool const contract = std::abs(old_energy - input) <= bound;
        bool const map =
            out.norminf(3) <= 512._rt * eps * te_expected &&
            out.norminf(4) <= 512._rt * eps * te_expected &&
            out.norminf(6) <= 512._rt * eps * te_expected &&
            out.norminf(5) <= 512._rt * eps &&
            std::abs(old_energy - input * cold / receiver_capacity) <= bound &&
            std::abs(recovered - input * cnew / receiver_capacity *
                                     recover_factor) <= bound;
        amrex::Print() << std::setprecision(17)
                       << "ENERGY_RECOVERY {\"n_old\":" << nold
                       << ",\"n_new\":" << nnew << ",\"n_ped\":" << nped
                       << ",\"stage_matched_receiver\":"
                       << (old_receiver ? "true" : "false")
                       << ",\"physical_source_J\":" << input
                       << ",\"old_state_increment_J\":" << old_energy
                       << ",\"new_density_increment_J\":" << new_energy
                       << ",\"recovered_heat_difference_J\":" << recovered
                       << ",\"capacity_ratio_old_new\":" << cold / cnew
                       << ",\"recovery_temperature_factor\":" << recover_factor
                       << ",\"source_temperature_error_K\":" << out.norminf(3)
                       << ",\"recovery_temperature_error_K\":" << out.norminf(4)
                       << ",\"marker_density_error_relative\":"
                       << out.norminf(5) << ",\"bound_J\":" << bound
                       << ",\"source_on_old_state_contract_pass\":"
                       << (contract ? "true" : "false")
                       << ",\"native_map_verified\":"
                       << (map ? "true" : "false") << "}\n";
        result = map ? (contract ? 0 : 2) : 3;
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return result;
}
