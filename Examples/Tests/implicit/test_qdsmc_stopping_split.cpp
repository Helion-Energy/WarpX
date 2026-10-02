/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
// Deposited stopping energy: two receivers, raw seam partials, periodic images,
// density/pedestal gates, signed cooling, and complete consumption.
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

using namespace amrex::literals;
using warpx::fields::FieldType;

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        auto& te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rn = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto& ro = *sim.m_fields.get(FieldType::hybrid_rho_fp_temp, 0);
        hp.m_has_energy_sink = false;
        hp.m_include_joule_heating = false;
        hp.m_has_electron_stopping = true;
        hp.m_cond_te_floor = 10.0_rt;
        auto const& geom = sim.Geom(0);
        auto const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
        AMREX_ALWAYS_ASSERT(geom.Domain().smallEnd() == amrex::IntVect(0));
        AMREX_ALWAYS_ASSERT(geom.isPeriodic(1));
        auto const K = PhysConst::q_e / PhysConst::kb;
        auto const t0 = 100.0_rt * K, gm1 = hp.m_gamma - 1.0_rt;
        auto const floor = hp.m_n_floor;
        auto const nold = 1.e19_rt, nnew = 4.e19_rt;
        auto* pedestal = const_cast<amrex::MultiFab*>(hp.DensityPedestal(0));
        auto const nped = pedestal ? 2.e18_rt : 0.0_rt;
        if (pedestal) { pedestal->setVal(nped * PhysConst::q_e); }
        amrex::ParmParse pp("audit");
        bool cooling = false;
        amrex::Real fraction = 0.5_rt;
        pp.query("cooling", cooling);
        pp.query("fraction", fraction);
        auto const unit = 0.2_rt * (nold + nped) * PhysConst::kb * t0 / gm1;
        te.setVal(t0);
        // Lazy, unallocated staging is a no-op, including a partial request.
        hp.QDSMCApplyFastIonHeating(0, nullptr, nullptr, 0.5_rt);
        AMREX_ALWAYS_ASSERT(te.min(0) == t0 && te.max(0) == t0);
        auto& stage = hp.GetFastIonHeatingStaging(0);
        amrex::MultiFab raw(stage.boxArray(), stage.DistributionMap(), 1, 0);
        amrex::MultiFab errors(stage.boxArray(), stage.DistributionMap(), 2, 0);
        auto const count = (nr + 1) * (nz + 1);
        std::vector<amrex::Real> expected(count), packet(count, 0.0_rt);
        for (int i = 0; i <= nr; ++i) {
            for (int j = 0; j <= nz; ++j) {
                auto const index = i * (nz + 1) + j;
                expected[index] = cooling && i == nr ? 5.0_rt * K : t0;
                // Independently sum the explicitly known deposit copies,
                // including the two physical representations of periodic z=0.
                for (int b = 0; b < stage.boxArray().size(); ++b) {
                    auto const& box = stage.boxArray()[b];
                    for (int jj : {j % nz, j % nz + nz}) {
                        if (box.contains(amrex::IntVect(AMREX_D_DECL(i, jj, 0)))) {
                            packet[index] += unit * (1.0_rt + 0.125_rt * b) *
                                (cooling && i >= nr / 2 ? -20.0_rt : 1.0_rt);
                        }
                    }
                }
            }
        }
        for (amrex::MFIter mfi(stage); mfi.isValid(); ++mfi) {
            auto const e = stage.array(mfi), temp = te.array(mfi);
            auto const old = ro.array(mfi), now = rn.array(mfi);
            auto const box_weight = 1.0_rt + 0.125_rt * mfi.index();
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                e(i,j,k) = unit * box_weight *
                    (cooling && i >= nr / 2 ? -20.0_rt : 1.0_rt);
                temp(i,j,k) = cooling && i == nr ? 5.0_rt * K : t0;
                old(i,j,k) = PhysConst::q_e * (cooling && i == 1 ? 0.0_rt : nold);
                now(i,j,k) = PhysConst::q_e * (cooling && i == 0 ? 0.0_rt : nnew);
            });
        }
        amrex::MultiFab::Copy(raw, stage, 0, 0, 1, 0);
        // Zero fraction must neither synchronize nor consume the raw packet.
        hp.QDSMCApplyFastIonHeating(0, nullptr, nullptr, 0.0_rt);
        amrex::Gpu::DeviceVector<amrex::Real> expected_device(count);
        amrex::Real declined = 0.0_rt, worst_error = 0.0_rt;
        auto const dx = geom.CellSizeArray();
        auto const radius = geom.ProbHi(0);
        for (int half = 0; half < 3; ++half) {
            auto const share = half == 0 ? fraction : (half == 1 ? 1.0_rt - fraction : 0.0_rt);
            for (int i = 0; i <= nr; ++i) {
                auto const nphys = cooling && i == 0 ? 0.0_rt : nnew;
                auto const nstate = half == 0 ? (cooling && i == 1 ? 0.0_rt : nold) : nnew;
                auto const capacity = std::max(nstate + nped, floor) * PhysConst::kb / gm1;
                auto const rl = std::max(0.0_rt, i * dx[0] - 0.5_rt * dx[0]);
                auto const rh = std::min(radius, i * dx[0] + 0.5_rt * dx[0]);
                auto const volume = MathConst::pi * (rh * rh - rl * rl) * dx[1];
                for (int j = 0; j <= nz; ++j) {
                    auto const index = i * (nz + 1) + j;
                    auto const energy = share * packet[index];
                    amrex::Real declined_density = 0.0_rt;
                    if (nphys <= floor) {
                        declined_density = energy;
                    } else {
                        auto const minimum = std::min(expected[index], 10.0_rt * K);
                        auto const trial = expected[index] + energy / capacity;
                        declined_density = std::max(0.0_rt, (minimum - trial) * capacity);
                        expected[index] = std::max(trial, minimum);
                    }
                    // Only one periodic endpoint contributes physical volume.
                    if (j < nz) { declined += volume * declined_density; }
                }
            }
            hp.ApplyQdsmcEnergySources(0, 0.0_rt, false, half == 0 ? ro : rn,
                                       half == 0 ? fraction : 1.0_rt);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, expected.begin(), expected.end(),
                             expected_device.begin());
            auto const* exact = expected_device.data();
            auto const remainder = half == 0 ? 1.0_rt - fraction : 0.0_rt;
            for (amrex::MFIter mfi(errors); mfi.isValid(); ++mfi) {
                auto const err = errors.array(mfi);
                auto const t = te.const_array(mfi);
                auto const left = stage.const_array(mfi), original = raw.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    err(i,j,k,0) = std::abs(t(i,j,k) - exact[i * (nz + 1) + j]);
                    err(i,j,k,1) = std::abs(left(i,j,k) - remainder * original(i,j,k));
                });
            }
            auto const bound = 4096.0_rt * std::numeric_limits<amrex::Real>::epsilon() *
                *std::max_element(expected.begin(), expected.end());
            worst_error = std::max(worst_error, errors.norminf(0));
            AMREX_ALWAYS_ASSERT(errors.norminf(0) < bound);
            AMREX_ALWAYS_ASSERT(errors.norminf(1) == 0.0_rt);
            AMREX_ALWAYS_ASSERT(std::abs(hp.m_stopping_declined_J - declined) <
                                1.e-11_rt * std::max(1.0_rt, std::abs(declined)));
        }
        amrex::Print() << std::setprecision(17)
            << "STOPPING_SPLIT {\"pass\":true,\"cooling\":" << (cooling ? "true" : "false")
            << ",\"fraction\":" << fraction << ",\"pedestal\":" << nped
            << ",\"boxes\":" << stage.boxArray().size()
            << ",\"max_temperature_error_K\":" << worst_error
            << ",\"declined_J\":" << hp.m_stopping_declined_J
            << ",\"expected_declined_J\":" << declined << "}\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
