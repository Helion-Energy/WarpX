/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#ifdef WARPX_DIM_RZ
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Parallelization/WarpXSumGuardCells.H"
#include "Utils/TextMsg.H"
#include <AMReX_GpuLaunch.H>
#include <iomanip>

using namespace amrex::literals;
using amrex::MultiFab;
using warpx::fields::FieldType;

void WarpX::ComputeRZContinuityResidual (
    MultiFab& residual, const MultiFab& rho_old, const MultiFab& rho_new,
    const std::array<MultiFab*, 3>& current, const amrex::Real delta_t,
    const bool yee_axis) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(delta_t > 0._rt && n_rz_azimuthal_modes == 1
        && grid_type == GridType::Staggered && Geom(0).ProbLo(0) == 0._rt,
        "RZ continuity residual requires dt>0 and staggered m=0 RZ with r_lo=0");
    const auto dx = Geom(0).CellSizeArray();
    const int axis = Geom(0).Domain().smallEnd(0);
    const amrex::Real axis_factor = yee_axis || !m_verboncoeur_axis_correction
        ? 0.25_rt : 1._rt/3._rt;
    for (amrex::MFIter mfi(residual); mfi.isValid(); ++mfi) {
        const auto out = residual.array(mfi);
        const auto ro = rho_old.const_array(mfi), rn = rho_new.const_array(mfi);
        const auto jr = current[0]->const_array(mfi), jz = current[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real r = static_cast<amrex::Real>(i-axis);
            const amrex::Real divr = i == axis ? jr(i,j,k)/(axis_factor*dx[0])
                : ((r+0.5_rt)*jr(i,j,k)-(r-0.5_rt)*jr(i-1,j,k))/(r*dx[0]);
            out(i,j,k) = (rn(i,j,k)-ro(i,j,k))/delta_t + divr
                + (jz(i,j,k)-jz(i,j-1,k))/dx[1];
        });
    }
}

void WarpX::AuditRZContinuity (const bool filtered)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 0
        && evolve_scheme == EvolveScheme::Explicit && !do_current_centering
        && !EB::enabled() && !do_fluid_species
        && n_rz_azimuthal_modes == 1 && grid_type == GridType::Staggered
        && Geom(0).ProbLo(0) == 0._rt,
        "rz_continuity_audit_interval supports single-level explicit hybrid PIC without EB, "
        "current centering or fluid species");
    const auto& period = Geom(0).periodicity();
    auto const& source_rho = *m_fields.get(FieldType::rho_fp, 0);
    auto const source_j = m_fields.get_alldirs(FieldType::current_fp, 0);
    auto rho = std::make_unique<MultiFab>(source_rho.boxArray(),
        source_rho.DistributionMap(), 1, source_rho.nGrowVect());
    MultiFab::Copy(*rho, source_rho, 0, 0, 1, rho->nGrowVect());
    auto current = source_j;
    std::array<std::unique_ptr<MultiFab>, 3> raw_current;
    if (!filtered) {
        // Sum a separate copy of the *local* deposits. Never modify a solver
        // moment or apply filtering to this control. Match SyncCurrentAndRho
        // physical boundary handling, then fill interior/periodic ghosts.
        WarpXSumGuardCells(*rho, period, amrex::min(get_ng_depos_rho(), rho->nGrowVect()));
        for (int c = 0; c < 3; ++c) {
            raw_current[c] = std::make_unique<MultiFab>(source_j[c]->boxArray(),
                source_j[c]->DistributionMap(), 1, source_j[c]->nGrowVect());
            current[c] = raw_current[c].get();
            MultiFab::Copy(*current[c], *source_j[c], 0, 0, 1, current[c]->nGrowVect());
            WarpXSumGuardCells(*current[c], period,
                amrex::min(get_ng_depos_J(), current[c]->nGrowVect()));
        }
        ApplyRhofieldBoundary(0, rho.get(), PatchType::fine);
        ApplyJfieldBoundary(0, current[0], current[1], current[2], PatchType::fine);
        rho->FillBoundary(period);
        for (auto* j : current) { j->FillBoundary(period); }
    }
    auto& previous = m_rz_continuity_old_rho[filtered ? 1 : 0];
    // Init/restart only seeds the old endpoint. A regrid/load balance resets
    // it as well; comparing different layouts would not be a time derivative.
    const bool same_layout = previous && previous->boxArray() == rho->boxArray()
        && previous->DistributionMap() == rho->DistributionMap();
    if (same_layout && istep[0] % m_rz_continuity_audit_interval == 0) {
        MultiFab residual(rho->boxArray(), rho->DistributionMap(), 1, 0);
        ComputeRZContinuityResidual(residual, *previous, *rho, current, dt[0]);
        const amrex::Real peak = residual.norminf();
        const amrex::Real scale = std::max(previous->norminf(), rho->norminf());
        ComputeRZContinuityResidual(residual, *previous, *rho, current, dt[0], true);
        auto log = amrex::Print();
        log << std::setprecision(17)
            << "RZ_CONTINUITY {\"step\":" << istep[0]
            << ",\"stage\":\"" << (filtered ? "post_sync" : "raw")
            << "\",\"deposition_linf_C_m3_s\":" << peak
            << ",\"yee_linf_C_m3_s\":" << residual.norminf()
            << ",\"dt_s\":" << dt[0] << ",\"rho_scale_C_m3\":" << scale
            << ",\"relative_step_residual\":";
        if (scale > 0._rt) { log << dt[0]*peak/scale; }
        else { log << "null"; }
        log << "}\n";
    }
    previous = std::move(rho);
}
#endif
