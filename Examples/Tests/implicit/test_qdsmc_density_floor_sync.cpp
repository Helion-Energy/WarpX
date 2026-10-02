/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
// Floor changes across real explicit source, pressure, entropy and conduction operators.
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>

using namespace amrex::literals;
using warpx::fields::FieldType;

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        AMREX_ALWAYS_ASSERT(sim.evolve_scheme == EvolveScheme::Explicit);
        auto& hp = *sim.get_pointer_HybridPICModel();
        amrex::ParmParse pp("audit");
        std::string mode = "operators";
        pp.query("mode", mode);
        if (mode == "invalid") { hp.SetDensityFloor(0.0_rt); }
        if (mode == "conflict") { hp.SetQdsmcDensityFloor(2.e18_rt); }
        amrex::Real initial_cond = 1.e18_rt, initial_te = 1.e18_rt;
        pp.query("initial_cond", initial_cond);
        pp.query("initial_te", initial_te);
        AMREX_ALWAYS_ASSERT(hp.m_n_floor == 1.e18_rt);
        AMREX_ALWAYS_ASSERT(hp.m_qdsmc_n_floor == initial_cond);
        AMREX_ALWAYS_ASSERT(hp.m_qdsmc_te_n_floor == initial_te);
        auto& te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rn = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto& ro = *sim.m_fields.get(FieldType::hybrid_rho_fp_temp, 0);
        auto& entropy = *sim.m_fields.get(FieldType::hybrid_entropy_fp, 0);
        auto& pressure = *sim.m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        auto const radius = geom.ProbHi(0);
        auto const nr = geom.Domain().bigEnd(0) + 1;
        auto const volume = MathConst::pi * radius * radius * geom.ProbLength(1);
        amrex::MultiFab error(te.boxArray(), te.DistributionMap(), 1, 0);
        auto relative_error = [&] (amrex::MultiFab const& field, amrex::Real expected)
        {
            amrex::MultiFab::Copy(error, field, 0, 0, 1, 0);
            error.plus(-expected, 0, 1, 0);
            return error.norminf(0) / std::abs(expected);
        };
        constexpr amrex::Real nold = 2.e18_rt, nnew = 2.e19_rt;
        auto const t0 = 100.0_rt * PhysConst::q_e / PhysConst::kb;
        auto const gm1 = hp.m_gamma - 1.0_rt;
        auto const dt = 1.e-5_rt, heat = dt * 1.e-5_rt * 1.e10_rt;
        auto const cond_dt = 1.e-7_rt, flux = 1.e4_rt;
        auto const wall_area = 2.0_rt * MathConst::pi * radius * geom.ProbLength(1);
        auto const wall_shell = MathConst::pi *
            (radius * radius - std::pow(radius - 0.5_rt * dx[0], 2));
        auto const area_over_volume = 2.0_rt * MathConst::pi * radius / wall_shell;
        // The source reads species moments independently of total rho. Supply
        // the same prescribed physical state to both, as in the PC source test.
        sim.m_fields.get("rho_fp_ions", 0)->setVal(nnew * PhysConst::q_e);
        for (int c = 0; c < 3; ++c) {
            sim.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, 0)[c]->setVal(0);
            sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0)[c]
                ->setVal(c == 2 ? 1.e5_rt : 0.0_rt);
            sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0)[c]
                ->setVal(c == 2 ? 1.0_rt : 0.0_rt);
            sim.m_fields.get_alldirs("current_fp_ions", 0)[c]->setVal(0);
        }
        // Cache the initial pedestal image before updating the floor. A fixed
        // profile and the legacy floor-following option are separate test cases.
        auto const* pedestal = hp.DensityPedestal(0);
        auto const fixed_ped = pedestal ? pedestal->max(0) / PhysConst::q_e : 0.0_rt;
        amrex::Real worst = 0.0_rt;
        // Increase, decrease, restore a saved controller value, and return to boot.
        for (auto const floor : {8.e18_rt, 5.e17_rt, 8.e18_rt, 1.e18_rt}) {
            te.setVal(t0);
            hp.SetDensityFloor(floor);
            AMREX_ALWAYS_ASSERT(te.min(0) == t0 && te.max(0) == t0);
            auto const cond_floor = hp.m_qdsmc_sync_density_floors ? floor : initial_cond;
            auto const te_floor = hp.m_qdsmc_sync_density_floors ? floor : initial_te;
            AMREX_ALWAYS_ASSERT(hp.m_qdsmc_n_floor == cond_floor);
            AMREX_ALWAYS_ASSERT(hp.m_qdsmc_te_n_floor == te_floor);
            hp.SetQdsmcDensityFloor(cond_floor); // compatible redundant controller call
            pedestal = hp.DensityPedestal(0);
            auto const nped = hp.m_density_pedestal_track_floor ? floor : fixed_ped;
            if (pedestal) {
                AMREX_ALWAYS_ASSERT(relative_error(*pedestal, nped * PhysConst::q_e) < 1.e-14_rt);
            }
            auto const cold = std::max(nold + nped, floor) * PhysConst::kb / gm1;
            auto const cnew = std::max(nnew + nped, floor) * PhysConst::kb / gm1;
            ro.setVal(nold * PhysConst::q_e);
            rn.setVal(nnew * PhysConst::q_e);
            hp.ApplyQdsmcEnergySources(0, 0.5_rt * dt, false, ro);
            auto const source1 = relative_error(te, t0 + 0.5_rt * heat / cold);
            hp.ApplyQdsmcEnergySources(0, 0.5_rt * dt, false, rn);
            auto const source2 = relative_error(te, t0 + 0.5_rt * heat / cold +
                                                       0.5_rt * heat / cnew);
            te.setVal(t0);
            hp.QDSMCFillElectronPressureFromTe(0, ro);
            auto const pressure_error = relative_error(pressure, cold * gm1 * t0);
            hp.QDSMCInitializeKe(0, ro);
            auto const effective_old = std::max(nold + nped, te_floor);
            auto const effective_new = std::max(nnew + nped, te_floor);
            auto const entropy_error = relative_error(entropy,
                100.0_rt * std::pow(effective_old, -gm1));
            // Real stationary markers carry old-density entropy and recover it
            // at the new density. Both density floors matter to this ratio.
            hp.QdsmcTransportOnce(0, 0.0_rt, true);
            auto const recovered = t0 * (nped + nold *
                std::pow(effective_new / effective_old, gm1)) / (nped + nold);
            auto const transport_error = relative_error(te, recovered);
            hp.m_qdsmc_pc->ResetParticles(0);
            // A below-floor physical source gate must remain closed even when
            // its receiving capacity includes an additive pedestal.
            rn.setVal(0.5_rt * floor * PhysConst::q_e);
            te.setVal(t0);
            hp.ApplyQdsmcEnergySources(0, dt, false, ro);
            auto const gate_error = relative_error(te, t0);
            rn.setVal(nnew * PhysConst::q_e);
            // Zero interior conductivity isolates a prescribed inward face flux:
            // delta U = q A dt independently of the density floor or box layout.
            // Check temperature at every node and physical energy separately.
            hp.m_cond_bc[0][1] = 2;
            hp.m_cond_bc_q[0][1] = flux;
            auto const cond_capacity = 1.5_rt * PhysConst::kb *
                std::max(nold + nped, cond_floor);
            auto const delta_t = flux * cond_dt * area_over_volume / cond_capacity;
            te.setVal(t0);
            hp.QdsmcConductionOnceFDAtState(0, cond_dt, false, ro, 0.0_rt);
            hp.m_cond_bc[0][1] = 0;
            for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                auto const e = error.array(mfi);
                auto const t = te.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    e(i,j,k) = std::abs(t(i,j,k) - t0 - (i == nr ? delta_t : 0.0_rt));
                });
            }
            auto const conduction_error = error.norminf(0) / delta_t;
            for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                auto const e = error.array(mfi);
                auto const t = te.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    auto const r = i * dx[0];
                    auto const rl = amrex::max(0.0_rt, r - 0.5_rt * dx[0]);
                    auto const rh = amrex::min(radius, r + 0.5_rt * dx[0]);
                    e(i,j,k) = (t(i,j,k) - t0) * cond_capacity *
                        MathConst::pi * (rh * rh - rl * rl) * dx[1];
                });
            }
            auto const delivered = error.sum_unique(0, false, geom.periodicity());
            auto const energy_error = std::abs(delivered / (flux * wall_area * cond_dt) - 1);
            worst = std::max({worst, source1, source2, pressure_error, entropy_error,
                              transport_error, gate_error, conduction_error, energy_error});
            amrex::Print() << std::setprecision(17)
                << "FLOOR_SYNC_STEP {\"floor\":" << floor << ",\"pedestal\":" << nped
                << ",\"source1_relative_error\":" << source1
                << ",\"source2_relative_error\":" << source2
                << ",\"pressure_relative_error\":" << pressure_error
                << ",\"entropy_relative_error\":" << entropy_error
                << ",\"transport_relative_error\":" << transport_error
                << ",\"gate_relative_error\":" << gate_error
                << ",\"conduction_relative_error\":" << conduction_error
                << ",\"wall_heat_relative_error\":" << energy_error
                << ",\"volume\":" << volume << "}\n";
            AMREX_ALWAYS_ASSERT(worst < 1.e-10_rt);
        }
        amrex::Print() << std::setprecision(17)
            << "FLOOR_SYNC {\"pass\":true,\"max_relative_error\":" << worst << "}\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
