/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Arena.H>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace amrex::literals;
using amrex::MultiFab;
using warpx::fields::FieldType;

namespace {
// Optional small field maps let the driver compare layouts, disabled controls,
// and diagnostic on/off exactly, without relying on only a maximum norm.
void dump (const MultiFab& mf, const std::string& name)
{
    std::ofstream out(name + "_rank" + std::to_string(amrex::ParallelDescriptor::MyProc()) + ".csv");
    out << std::setprecision(17) << "i,j,value\n";
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        amrex::FArrayBox host(mfi.fabbox(), 1, amrex::The_Pinned_Arena());
        amrex::Gpu::dtoh_memcpy_async(host.dataPtr(), mf[mfi].dataPtr(),
            mf[mfi].box().numPts()*sizeof(amrex::Real));
        amrex::Gpu::streamSynchronize();
        auto const a = host.const_array();
        auto const box = mfi.validbox();
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                out << i << ',' << j << ',' << a(i,j,0) << '\n';
            }
        }
    }
}
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    int result = 0;
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& pc = w.GetPartContainer();
        auto* rho = w.m_fields.get(FieldType::rho_fp, 0);
        auto current = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto const& geom = w.Geom(0);
        auto const& period = geom.periodicity();
        amrex::Real dt = w.getdt(0);
        int mode = 2;
        bool write_maps = false;
        int evolve_steps = 0;
        amrex::ParmParse test("continuity_test");
        test.query("mode", mode);
        test.query("write_maps", write_maps);
        test.query("evolve_steps", evolve_steps);
        if (evolve_steps > 0) {
            w.Evolve(evolve_steps);
            auto const* te = w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
            auto const* pe = w.m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
            auto const velocity = w.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp, 0);
            auto const electric = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
            auto const magnetic = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
            bool finite = te->is_finite() && pe->is_finite() && rho->is_finite();
            if (write_maps) { dump(*te, "Te"); dump(*pe, "Pe"); dump(*rho, "rho"); }
            for (int c = 0; c < 3; ++c) {
                finite = finite && velocity[c]->is_finite() && current[c]->is_finite()
                    && electric[c]->is_finite() && magnetic[c]->is_finite();
                if (write_maps) {
                    dump(*velocity[c], "Ve" + std::to_string(c));
                    dump(*current[c], "Ji" + std::to_string(c));
                    dump(*electric[c], "E" + std::to_string(c));
                    dump(*magnetic[c], "B" + std::to_string(c));
                }
            }
            amrex::Print() << std::setprecision(17)
                << "CONTINUITY_EVOLVE {\"steps\":" << w.getistep(0)
                << ",\"Te_max_K\":" << te->norminf()
                << ",\"finite\":" << (finite ? "true" : "false") << "}\n";
            if (!finite || w.getistep(0) != evolve_steps) { result = 4; }
        }
        bool corrected_axis = true;
        amrex::ParmParse("boundary").query("verboncoeur_axis_correction", corrected_axis);
        amrex::Real old_mass = 0._rt;
        MultiFab oldrho(rho->boxArray(), rho->DistributionMap(), 1, rho->nGrowVect());
        for (int endpoint = 0; endpoint < (evolve_steps > 0 ? 0 : 2); ++endpoint) {
            if (endpoint == 1) { pc.PushX(dt); pc.Redistribute(); }
            // Actual shape deposition at two endpoints and its half-time
            // Esirkepov current. Native sum/filter/physical BC follows.
            pc.DepositCharge({rho}, 0._rt);
            pc.DepositCurrent({current}, dt, -.5_rt*dt);
            w.SyncCurrentAndRho();
            rho->FillBoundary(period);
            for (auto* j : current) { j->FillBoundary(period); }
            auto const dx = geom.CellSizeArray();
            const int nr = geom.Domain().bigEnd(0)+1, nz = geom.Domain().bigEnd(1)+1;
            const bool periodic_z = geom.isPeriodic(1);
            MultiFab mass(rho->boxArray(), rho->DistributionMap(), 1, 0);
            for (amrex::MFIter mfi(mass); mfi.isValid(); ++mfi) {
                auto const a = mass.array(mfi);
                auto const rr = rho->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const auto weight = i == 0 ? (corrected_axis ? 1._rt/3._rt : .25_rt)
                        : (i == nr ? static_cast<amrex::Real>(i) : 2._rt*i);
                    const auto zweight = !periodic_z && (j == 0 || j == nz) ? .5_rt : 1._rt;
                    a(i,j,k) = rr(i,j,k)*MathConst::pi*dx[0]*dx[0]*dx[1]*weight*zweight;
                });
            }
            const auto total = mass.sum_unique(0, false, period);
            if (endpoint == 0) {
                old_mass = total;
                MultiFab::Copy(oldrho, *rho, 0, 0, 1, rho->nGrowVect());
            } else {
                MultiFab residual(rho->boxArray(), rho->DistributionMap(), 1, 0);
                w.ComputeRZContinuityResidual(residual, oldrho, *rho, current, dt);
                if (write_maps) {
                    dump(residual, "residual"); dump(oldrho, "rho_old"); dump(*rho, "rho_new");
                    for (int c = 0; c < 3; ++c) { dump(*current[c], "J" + std::to_string(c)); }
                }
                const auto peak = residual.norminf();
                const auto scale = std::max(rho->norminf(), oldrho.norminf());
                const auto error = peak*dt/scale;
                w.ComputeRZContinuityResidual(residual, oldrho, *rho, current, dt, true);
                amrex::Print() << std::setprecision(17)
                    << "CONTINUITY_TEST {\"mode\":" << mode
                    << ",\"relative_step_residual\":" << error
                    << ",\"yee_relative_step_residual\":" << residual.norminf()*dt/scale
                    << ",\"rho_scale\":" << scale
                    << ",\"old_charge_C\":" << old_mass
                    << ",\"new_charge_C\":" << total << "}\n";
                const auto eps = std::max(std::numeric_limits<amrex::Real>::epsilon(),
                    amrex::Real(std::numeric_limits<amrex::ParticleReal>::epsilon()));
                if (mode != 1 && error > 512._rt*eps) { result = 2; }
                if (!rho->is_finite() || !residual.is_finite()) { result = 3; }
            }
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
    return result;
}
