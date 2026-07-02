/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HybridPICModel.H"

#include "EBJBoundary.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_Array4.H>
#include <AMReX_GpuControl.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <algorithm>

using namespace amrex;
using warpx::fields::FieldType;

void HybridPICModel::MarderCleanDivergence (
    ablastr::fields::VectorField const& field,
    std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
    warpx::hybrid::EBFillStatus* status_cache,
    const bool normal_odd,
    const bool fill_covered_centers,
    const amrex::Real alpha,
    const int max_iters,
    const amrex::Real clean_band_cells,
    const amrex::Real fill_band_cells,
    const int lev) const
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    if (alpha <= 0.0_rt || max_iters <= 0 || !EB::enabled()) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const bool nodal_grid =
        (WarpX::grid_type == ablastr::utils::enums::GridType::Collocated);

    // CFL-stable explicit grad(div) step: the discrete Laplacian peaks at ~1/h^2
    // (collocated wide) or ~4/h^2 (Yee compact), so alpha (a fraction of the CFL
    // cap) maps to the same damping on both grids via this scaling -- mirrors the
    // E-field Marder.
    Real h_min = dx[0], h_max = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        h_min = std::min(h_min, dx[d]);
        h_max = std::max(h_max, dx[d]);
    }
    const Real alpha_scaled = alpha * h_min * h_min * (nodal_grid ? 4.0_rt : 1.0_rt);
    const Real d_clean = clean_band_cells * h_max;
    // EB-aware band cutoffs, exposed as knobs. The defaults (inner_div=1,
    // inner_corr=2) trust the divergence only where its own +/-1 stencil is in
    // the fluid and apply the correction only where the full +/-2 grad(div)
    // stencil is fluid, leaving the immediate wall layer to the level-set
    // mirror fill -- that preserves the near-wall order on a smooth wall but
    // cannot reach a divergence mode living in the first two fluid layers.
    // Setting both to 0 extends the clean over every uncovered band node (the
    // wall-layer mode): the stencils then read mirror-filled covered nodes,
    // trading an O(h) near-wall error for dissipation exactly where a sharp
    // re-entrant corner (e.g. a wall radius step) pumps an exponentially
    // growing div(B). phi is the true signed distance, so these cutoffs hold
    // for any wall orientation. (No clean if d_clean <= d_stencil.)
    const Real d_div_keep = m_divb_clean_inner_div_cells * h_max;
    const Real d_stencil = m_divb_clean_inner_corr_cells * h_max;
    amrex::GpuArray<Real, AMREX_SPACEDIM> inv_dx{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { inv_dx[d] = 1.0_rt/dx[d]; }

    amrex::MultiFab const* phi_mf = warpx.m_fields.get(FieldType::distance_to_eb, lev);
    auto* fdtd = warpx.get_pointer_fdtd_solver_fp(lev);

    // Nodal divergence scratch; one ghost on the collocated grid because the
    // centered grad(div) update reaches one node past a box boundary.
    const IntVect div_ng = nodal_grid ? IntVect(1) : IntVect::TheZeroVector();
    auto const& nodal_ba = amrex::convert(field[0]->boxArray(), IntVect::TheNodeVector());
    MultiFab div_f(nodal_ba, field[0]->DistributionMap(), 1, div_ng);

    for (int it = 0; it < max_iters; ++it) {
        for (int c = 0; c < 3; ++c) {
            ablastr::utils::communication::FillBoundary(
                *field[c], IntVect(1), WarpX::do_single_precision_comms,
                geom.periodicity());
        }
        fdtd->ComputeDivE(field, div_f);

        // Keep the divergence only in the fluid-clean near-wall band
        // [d_div_keep, d_clean]: drop the bulk (already solenoidal) and, unless
        // the wall-layer mode is selected, the wall layer / covered nodes
        // (phi < d_div_keep), where the centered divergence would read the
        // mirror-filled covered nodes across the wall.
        for (MFIter mfi(div_f, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& d = div_f.array(mfi);
            Array4<Real const> const& phi = phi_mf->const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (phi(i, j, k) < d_div_keep || phi(i, j, k) > d_clean) { d(i, j, k) = 0.0_rt; }
                });
        }
        if (nodal_grid) {
            ablastr::utils::communication::FillBoundary(
                div_f, div_ng, WarpX::do_single_precision_comms, geom.periodicity());
        }

        // field += alpha_scaled * grad(div), on solution nodes inside the band.
        // grad of a scalar is curl-free, so curl(field) (= J for B) is unchanged
        // away from the band cutoffs (the band masking perturbs the curl at the
        // band edges; in the wall-layer mode the mirror re-fill below re-imposes
        // the wall parity on top each sweep).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*field[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> const& Fx = field[0]->array(mfi);
            Array4<Real> const& Fy = field[1]->array(mfi);
            Array4<Real> const& Fz = field[2]->array(mfi);
            Array4<Real const> const& d = div_f.const_array(mfi);
            Array4<Real const> const& phi = phi_mf->const_array(mfi);
            Array4<int const> const& ux = eb_update[0]->const_array(mfi);
            Array4<int const> const& uy = eb_update[1]->const_array(mfi);
            Array4<int const> const& uz = eb_update[2]->const_array(mfi);

            Box const& tx = mfi.tilebox(field[0]->ixType().toIntVect());
            Box const& ty = mfi.tilebox(field[1]->ixType().toIntVect());
            Box const& tz = mfi.tilebox(field[2]->ixType().toIntVect());

#if defined(WARPX_DIM_3D)
            amrex::ParallelFor(tx, ty, tz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (ux(i,j,k)==0 || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean) { return; }
                    Fx(i,j,k) += nodal_grid
                        ? alpha_scaled*(d(i+1,j,k)-d(i-1,j,k))*(0.5_rt*inv_dx[0])
                        : alpha_scaled*(d(i+1,j,k)-d(i,j,k))*inv_dx[0];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (uy(i,j,k)==0 || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean) { return; }
                    Fy(i,j,k) += nodal_grid
                        ? alpha_scaled*(d(i,j+1,k)-d(i,j-1,k))*(0.5_rt*inv_dx[1])
                        : alpha_scaled*(d(i,j+1,k)-d(i,j,k))*inv_dx[1];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){
                    if (uz(i,j,k)==0 || phi(i,j,k)<d_stencil || phi(i,j,k)>d_clean) { return; }
                    Fz(i,j,k) += nodal_grid
                        ? alpha_scaled*(d(i,j,k+1)-d(i,j,k-1))*(0.5_rt*inv_dx[2])
                        : alpha_scaled*(d(i,j,k+1)-d(i,j,k))*inv_dx[2];
                });
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
            // In-plane components only (out-of-plane has no divergence contribution).
            amrex::ignore_unused(Fy, uy, ty);
            amrex::ParallelFor(tx, tz,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    if (ux(i,j,0)==0 || phi(i,j,0)<d_stencil || phi(i,j,0)>d_clean) { return; }
                    Fx(i,j,0) += nodal_grid
                        ? alpha_scaled*(d(i+1,j,0)-d(i-1,j,0))*(0.5_rt*inv_dx[0])
                        : alpha_scaled*(d(i+1,j,0)-d(i,j,0))*inv_dx[0];
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
                    if (uz(i,j,0)==0 || phi(i,j,0)<d_stencil || phi(i,j,0)>d_clean) { return; }
                    Fz(i,j,0) += nodal_grid
                        ? alpha_scaled*(d(i,j+1,0)-d(i,j-1,0))*(0.5_rt*inv_dx[1])
                        : alpha_scaled*(d(i,j+1,0)-d(i,j,0))*inv_dx[1];
                });
#endif
        }

        // Re-impose the EB PEC condition with the field's own parity, using its
        // own fill band, so the covered band stays consistent with the main fill.
        warpx::hybrid::ApplyPECBoundaryToField(
            field, eb_update, *phi_mf, geom,
            m_eb_bc_rtol, m_eb_bc_max_iters, m_eb_bc_direct_fill,
            normal_odd, fill_covered_centers, status_cache, fill_band_cells,
            m_eb_cylindrical_correction, m_eb_cyl_axis);
    }

    for (int c = 0; c < 3; ++c) {
        ablastr::utils::communication::FillBoundary(
            *field[c], field[c]->nGrowVect(), WarpX::do_single_precision_comms,
            geom.periodicity());
    }
#else
    amrex::ignore_unused(field, eb_update, status_cache, normal_odd,
        fill_covered_centers, alpha, max_iters, clean_band_cells, fill_band_cells, lev);
#endif
}

void HybridPICModel::MarderCleanFieldsPerStep () const
{
    if (m_divb_clean_alpha <= 0.0_rt && m_divj_clean_alpha <= 0.0_rt) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        // B (magnetic parity).
        if (m_divb_clean_alpha > 0.0_rt) {
            if (static_cast<int>(m_eb_bc_status_B.size()) <= lev) { m_eb_bc_status_B.resize(lev+1); }
            MarderCleanDivergence(
                warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev),
                warpx.GetEBUpdateBFlag()[lev], &m_eb_bc_status_B[lev],
                /*normal_odd=*/true, /*fill_covered_centers=*/false,
                m_divb_clean_alpha, m_divb_clean_iters,
                m_divb_clean_band_cells, m_eb_b_fill_band_cells, lev);
        }
        // Total Ampere current (electric parity), never an ion species.
        if (m_divj_clean_alpha > 0.0_rt) {
            if (static_cast<int>(m_eb_bc_status_Jplasma.size()) <= lev) { m_eb_bc_status_Jplasma.resize(lev+1); }
            MarderCleanDivergence(
                warpx.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev),
                warpx.GetEBUpdateEFlag()[lev], &m_eb_bc_status_Jplasma[lev],
                /*normal_odd=*/false, /*fill_covered_centers=*/true,
                m_divj_clean_alpha, m_divb_clean_iters,
                m_divb_clean_band_cells, m_eb_fill_band_cells, lev);
        }
    }
}
