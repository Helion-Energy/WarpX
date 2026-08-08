/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 *
 * Mesh-refinement support for the hybrid-PIC (Ohm's law) solver:
 *  - divergence-free prolongation of B into fine-level coarse-fine ghosts,
 *  - fine->coarse face-averaged restriction of B with a sacrificial setback
 *    band under the fine-patch edge,
 *  - seam EMF matching: an edge-EMF flux register on the restriction commit
 *    boundary, so the coarse faces around the seam integrate the fine
 *    edge-EMF history (coarse div(B) preserved by construction),
 *  - an optional runtime div(B) audit of the transfer operators.
 */

#include "HybridPICModel.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBCellFlag.H>
#   include <AMReX_EBFabFactory.H>
#endif
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFInterpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <sstream>

using namespace amrex;
using warpx::fields::FieldType;

/** \brief Staggering-aware linear interpolation of a coarse array to the
 * fine index (j,k,l), for data with the same staggering on both levels.
 * Weight logic follows the same-type aux kernel in WarpXComm_K.H; reads
 * beyond the coarse array are zero-padded. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real hybrid_mr_interp_from_coarse (
    int j, int k, int l, int n,
    amrex::Array4<amrex::Real const> const& arr_coarse,
    amrex::IntVect const& arr_stag,
    amrex::IntVect const& rr)
{
    using namespace amrex;

    // Pad arr_coarse with zeros beyond ghost cells for out-of-bound accesses
    const auto arr_coarse_zeropad = [arr_coarse, n] (const int jj, const int kk, const int ll) noexcept
    {
        return arr_coarse.contains(jj, kk, ll) ? arr_coarse(jj, kk, ll, n) : 0.0_rt;
    };

    // Refinement ratio and staggering (0: cell-centered; 1: nodal);
    // unused dimensions are considered nodal.
    const int rj = rr[0];
    const int rk = (AMREX_SPACEDIM > 1) ? rr[1] : 1;
    const int rl = (AMREX_SPACEDIM > 2) ? rr[2] : 1;
    const int sj = arr_stag[0];
    const int sk = (AMREX_SPACEDIM > 1) ? arr_stag[1] : 1;
    const int sl = (AMREX_SPACEDIM > 2) ? arr_stag[2] : 1;

    const int nj = 2;
    const int nk = (AMREX_SPACEDIM > 1) ? 2 : 1;
    const int nl = (AMREX_SPACEDIM > 2) ? 2 : 1;

    const int jc = (sj == 0) ? amrex::coarsen(j - rj/2, rj) : amrex::coarsen(j, rj);
    const int kc = (sk == 0) ? amrex::coarsen(k - rk/2, rk) : amrex::coarsen(k, rk);
    const int lc = (sl == 0) ? amrex::coarsen(l - rl/2, rl) : amrex::coarsen(l, rl);

    const amrex::Real hj = (sj == 0) ? 0.5_rt : 0._rt;
    const amrex::Real hk = (sk == 0) ? 0.5_rt : 0._rt;
    const amrex::Real hl = (sl == 0) ? 0.5_rt : 0._rt;

    amrex::Real res = 0.0_rt;
    for         (int jj = 0; jj < nj; jj++) {
        for     (int kk = 0; kk < nk; kk++) {
            for (int ll = 0; ll < nl; ll++) {
                const amrex::Real wj = (rj - amrex::Math::abs(j + hj - (jc + jj + hj) * rj)) / static_cast<amrex::Real>(rj);
                const amrex::Real wk = (rk - amrex::Math::abs(k + hk - (kc + kk + hk) * rk)) / static_cast<amrex::Real>(rk);
                const amrex::Real wl = (rl - amrex::Math::abs(l + hl - (lc + ll + hl) * rl)) / static_cast<amrex::Real>(rl);
                res += wj * wk * wl * arr_coarse_zeropad(jc+jj, kc+kk, lc+ll);
            }
        }
    }
    return res;
}

/** \brief True if the cell (i,j,k) counts as "covered" for the restriction
 * keep-mask: either its mask value is 1, or it lies outside the domain in a
 * non-periodic direction (physical boundaries neither erode the mask nor
 * block the masked face copy). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool hybrid_mr_cell_covered (amrex::Array4<int const> const& m,
                             int i, int j, int k,
                             amrex::Box const& domain,
                             amrex::GpuArray<int, AMREX_SPACEDIM> const& is_per)
{
    const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!is_per[d] && (iv[d] < domain.smallEnd(d) || iv[d] > domain.bigEnd(d))) {
            return true;
        }
    }
    return m(i, j, k) == 1;
}

/** \brief True if the staggered point (j,k,l) touches a cell of the
 * sacrificial fine-edge band (edge_mask == 0). Nodal directions touch the
 * two adjacent cells, cell-centered directions only their own cell. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool hybrid_mr_point_in_band (int j, int k, int l,
                              amrex::IntVect const& stag,
                              amrex::Array4<int const> const& edge_mask)
{
    const int oj = (stag[0] == 1) ? 1 : 0;
    const int ok = (AMREX_SPACEDIM > 1 && stag[1] == 1) ? 1 : 0;
    const int ol = (AMREX_SPACEDIM > 2 && stag[2] == 1) ? 1 : 0;
    for         (int dj = -oj; dj <= 0; ++dj) {
        for     (int dk = -ok; dk <= 0; ++dk) {
            for (int dl = -ol; dl <= 0; ++dl) {
                if (edge_mask(j+dj, k+dk, l+dl) == 0) { return true; }
            }
        }
    }
    return false;
}

namespace
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    /** \brief Copy src into dst only on ghost cells of dst that lie outside
     * the owning box's valid region, restricted to the periodically grown
     * domain (ghost cells outside a non-periodic domain boundary are left to
     * the domain boundary conditions). */
    void CopyGhostRegion (amrex::MultiFab& dst, amrex::MultiFab const& src,
                          amrex::Geometry const& geom, amrex::IntVect const& ng)
    {
        const amrex::Box allowed = amrex::convert(
            geom.growPeriodicDomain(ng), dst.ixType());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box vbx = mfi.validbox();
            amrex::Box gbx = mfi.growntilebox(ng);
            gbx &= allowed;
            if (!gbx.ok()) { continue; }
            auto const& d = dst.array(mfi);
            auto const& s = src.const_array(mfi);
            amrex::ParallelFor(gbx, dst.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (!vbx.contains(i, j, k)) { d(i, j, k, n) = s(i, j, k, n); }
                });
        }
    }

    /** \brief Overwrite dst with src on the faces (or cells, for the 2D
     * out-of-plane cell-centered component) all of whose adjacent keep-mask
     * cells are covered. If face_gate is given (same staggering/layout as
     * dst; 1 = live, 0 = frozen on the coarse staircase), frozen faces are
     * never overwritten: they stay coarse-owned (EB ownership rule). */
    void MaskedCopyRestricted (amrex::MultiFab& dst, amrex::MultiFab const& src,
                               amrex::iMultiFab const& mask,
                               amrex::iMultiFab const* face_gate,
                               amrex::Geometry const& geom)
    {
        const amrex::IntVect itype = dst.ixType().toIntVect();
        int fdir = -1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (itype[d] == 1) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fdir == -1,
                    "MaskedCopyRestricted: expected face- or cell-centered data");
                fdir = d;
            }
        }
        const int ilo = (fdir == 0) ? 1 : 0;
        const int jlo = (fdir == 1) ? 1 : 0;
        const int klo = (fdir == 2) ? 1 : 0;
        const amrex::Box domain = geom.Domain();
        amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = geom.isPeriodic(d); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& d_arr = dst.array(mfi);
            auto const& s_arr = src.const_array(mfi);
            auto const& m_arr = mask.const_array(mfi);
            amrex::Array4<int const> g_arr = (face_gate != nullptr) ?
                face_gate->const_array(mfi) : amrex::Array4<int const>{};
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (g_arr && g_arr(i, j, k) == 0) { return; }
                    if (hybrid_mr_cell_covered(m_arr, i-ilo, j-jlo, k-klo, domain, is_per) &&
                        hybrid_mr_cell_covered(m_arr, i, j, k, domain, is_per))
                    {
                        d_arr(i, j, k) = s_arr(i, j, k);
                    }
                });
        }
    }

    /** \brief Max |div B| over selected cells of one level.
     *
     * The region per box is grow(validbox_cells, region_grow), intersected
     * with the periodically grown domain, minus grow(validbox_cells,
     * exclude_grow) if exclude_grow >= 0. If a cell keep-mask is given,
     * region_sel further selects: 0 = exterior (mask 0, no restricted
     * faces), 1 = seam ring (mask 1 with a mask-0 face neighbor: cells
     * mixing restricted and freely evolved faces), 2 = strict interior
     * (mask 1, all face neighbors covered: all faces restricted).
     *
     * If a wall indicator is given (cell-centered, 1 = the EB-gated transfer
     * granule holding this cell has a staircase-frozen face; sampled at
     * coarsen(cell, wall_ratio), so a coarsened indicator serves the fine
     * level), wall_mode selects: 0 = ignore, 1 = exclude wall cells, 2 =
     * wall cells only (region_sel is then ignored; with a keep-mask, the
     * exterior class stays excluded since it has no transferred faces).
     */
    amrex::Real MaxAbsDivB (
        ablastr::fields::VectorField const& B,
        amrex::Geometry const& geom,
        int region_grow, int exclude_grow,
        amrex::iMultiFab const* mask, int region_sel,
        amrex::iMultiFab const* wall = nullptr,
        amrex::IntVect const& wall_ratio = amrex::IntVect(1),
        int wall_mode = 0)
    {
        auto const dxi = geom.InvCellSizeArray();
        const amrex::Box allowed = geom.growPeriodicDomain(std::max(region_grow, 0));
        const amrex::Box domain = geom.Domain();
        amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = geom.isPeriodic(d); }
        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(*B[0], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box vcell = amrex::enclosedCells(mfi.validbox());
            amrex::Box bx = amrex::grow(vcell, region_grow);
            bx &= allowed;
            if (!bx.ok()) { continue; }
            const bool has_excl = (exclude_grow >= 0);
            const amrex::Box excl = has_excl ? amrex::grow(vcell, exclude_grow) : amrex::Box();
            auto const& Bx_arr = B[0]->const_array(mfi);
#if defined(WARPX_DIM_3D)
            auto const& By_arr = B[1]->const_array(mfi);
#endif
            auto const& Bz_arr = B[2]->const_array(mfi);
            amrex::Array4<int const> m_arr = (mask != nullptr) ?
                mask->const_array(mfi) : amrex::Array4<int const>{};
            amrex::Array4<int const> w_arr = (wall != nullptr) ?
                wall->const_array(mfi) : amrex::Array4<int const>{};
            const amrex::IntVect wr = wall_ratio;
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (has_excl && excl.contains(i, j, k)) { return {0._rt}; }
                    int region = 0;
                    if (m_arr) {
                        if (m_arr(i, j, k) == 1) {
                            bool all_nb = true;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                const int ii = (d == 0) ? 1 : 0;
                                const int jj = (d == 1) ? 1 : 0;
                                const int kk = (d == 2) ? 1 : 0;
                                all_nb = all_nb &&
                                    hybrid_mr_cell_covered(m_arr, i-ii, j-jj, k-kk, domain, is_per) &&
                                    hybrid_mr_cell_covered(m_arr, i+ii, j+jj, k+kk, domain, is_per);
                            }
                            region = all_nb ? 2 : 1;
                        }
                    }
                    int wall_v = 0;
                    if (w_arr) {
                        const amrex::IntVect civ = amrex::coarsen(
                            amrex::IntVect(AMREX_D_DECL(i, j, k)), wr);
                        wall_v = w_arr(civ[0], civ[1],
                                       (AMREX_SPACEDIM == 3) ? civ[AMREX_SPACEDIM-1] : 0);
                    }
                    if (wall_mode == 2) {
                        if (wall_v == 0) { return {0._rt}; }
                        if (m_arr && region == 0) { return {0._rt}; }
                    } else {
                        if (wall_mode == 1 && wall_v != 0) { return {0._rt}; }
                        if (m_arr && region != region_sel) { return {0._rt}; }
                    }
#if defined(WARPX_DIM_3D)
                    const amrex::Real d =
                          (Bx_arr(i+1, j, k) - Bx_arr(i, j, k)) * dxi[0]
                        + (By_arr(i, j+1, k) - By_arr(i, j, k)) * dxi[1]
                        + (Bz_arr(i, j, k+1) - Bz_arr(i, j, k)) * dxi[2];
#else
                    const amrex::Real d =
                          (Bx_arr(i+1, j, k) - Bx_arr(i, j, k)) * dxi[0]
                        + (Bz_arr(i, j+1, k) - Bz_arr(i, j, k)) * dxi[1];
#endif
                    return {std::abs(d)};
                });
        }
        amrex::Real r = amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelDescriptor::ReduceRealMax(r);
        return r;
    }

    /** \brief max|B| over the valid region of all components of one level. */
    amrex::Real MaxAbsB (ablastr::fields::VectorField const& B)
    {
        amrex::Real bmax = 0._rt;
        for (int idim = 0; idim < 3; ++idim) {
            bmax = std::max(bmax, B[idim]->norm0(0, 0));
        }
        return bmax;
    }

    /** \brief Cell-centered indicator on one level's cell layout (with ng
     * ghost cells, ng <= the gate ghost width): 1 where any face of the cell
     * is frozen on the level's staircase (eb_update_B == 0). Ghost values
     * are computed directly on the grown boxes from the gates' filled
     * ghosts. Only the AMREX_SPACEDIM face-staggered components enter (the
     * 2D out-of-plane component is cell-centered and does not enter divB). */
    amrex::iMultiFab MakeFrozenFaceCellIndicator (
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& gate,
        amrex::BoxArray const& ba_cell,
        amrex::DistributionMapping const& dm,
        int ng)
    {
        amrex::iMultiFab wall(ba_cell, dm, 1, ng);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(wall, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box gbx = mfi.growntilebox(ng);
            auto const& w_arr = wall.array(mfi);
            auto const& gx = gate[0]->const_array(mfi);
#if defined(WARPX_DIM_3D)
            auto const& gy = gate[1]->const_array(mfi);
#endif
            auto const& gz = gate[2]->const_array(mfi);
            amrex::ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    bool frozen = (gx(i, j, k) == 0) || (gx(i+1, j, k) == 0);
#if defined(WARPX_DIM_3D)
                    frozen = frozen || (gy(i, j, k) == 0) || (gy(i, j+1, k) == 0)
                                    || (gz(i, j, k) == 0) || (gz(i, j, k+1) == 0);
#else
                    frozen = frozen || (gz(i, j, k) == 0) || (gz(i, j+1, k) == 0);
#endif
                    w_arr(i, j, k) = frozen ? 1 : 0;
                });
        }
        return wall;
    }
#endif // Cartesian 2D/3D
}

void HybridPICModel::EnsureProlongGateMask (const int lev,
    amrex::BoxArray const& cpatch_ba, amrex::DistributionMapping const& fdm)
{
    auto& warpx = WarpX::GetInstance();

    if (static_cast<int>(m_mr_prolong_gate.size()) <= lev) {
        m_mr_prolong_gate.resize(lev + 1);
        m_mr_prolong_gate_ba.resize(lev + 1);
        m_mr_prolong_gate_fresh.resize(lev + 1, 0);
    }
    if (m_mr_prolong_gate[lev][0] &&
        m_mr_prolong_gate_ba[lev] == cpatch_ba &&
        m_mr_prolong_gate[lev][0]->DistributionMap() == fdm)
    {
        return;
    }
    m_mr_prolong_gate_fresh[lev] = 1;

    auto const& eb_gate = warpx.GetEBUpdateBFlag()[lev-1];
    const amrex::Geometry& cgeom = warpx.Geom(lev-1);
    for (int idim = 0; idim < 3; ++idim) {
        m_mr_prolong_gate[lev][idim] = std::make_unique<amrex::iMultiFab>(
            amrex::convert(cpatch_ba, eb_gate[idim]->ixType()), fdm, 1, 0);
        // Faces with no underlying coarse data (beyond a non-periodic domain
        // boundary) default to live; the corresponding fine ghost cells are
        // never committed back anyway.
        m_mr_prolong_gate[lev][idim]->setVal(1);
        m_mr_prolong_gate[lev][idim]->ParallelCopy(
            *eb_gate[idim], 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0),
            cgeom.periodicity());
    }
    m_mr_prolong_gate_ba[lev] = cpatch_ba;
}

void HybridPICModel::FillBfieldCoarseFineGhosts (
    const int lev, amrex::IntVect const& ng, std::optional<bool> nodal_sync)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();

    // EB ownership rule (P-1): coarse faces frozen by the coarse staircase
    // (m_eb_update_B[lev-1] == 0) are masked out of the divergence-free face
    // interpolation, so frozen coarse data never sources live fine ghosts.
    // The scratch is pre-seeded from the cached frozen ghost-fill state so
    // that the retained fine faces feed flux-consistent (static) data into
    // the interior closures.
#ifdef AMREX_USE_EB
    const bool gate_enabled = EB::enabled() && m_mr_eb_gate_prolong &&
        (warpx.GetEBUpdateBFlag()[lev-1][0] != nullptr);
#else
    const bool gate_enabled = false;
#endif
    bool masked_fill = false; // decided below, after the gate-mask cache check

    ablastr::fields::VectorField Bfine = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
    ablastr::fields::VectorField Bcrse = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev-1);

    const amrex::Geometry& cgeom = warpx.Geom(lev-1);
    const amrex::Geometry& fgeom = warpx.Geom(lev);
    const amrex::IntVect ratio = warpx.refRatio(lev-1);

    // Prolong the coarse solution onto a scratch copy of the fine patch;
    // only the coarse-fine ghost region is copied back below. The fill
    // region per fine box is its valid cell box grown by ng, padded up to
    // coarse alignment (FaceDivFree writes every fine face of each covered
    // coarse cell).
    amrex::IntVect ng_pad;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        ng_pad[d] = ((ng[d] + ratio[d] - 1) / ratio[d]) * ratio[d];
    }
    const amrex::BoxArray& fba_cell = warpx.boxArray(lev);
    std::array<amrex::MultiFab, 3> Btmp;
    for (int idim = 0; idim < 3; ++idim) {
        Btmp[idim] = amrex::MultiFab(
            Bfine[idim]->boxArray(), Bfine[idim]->DistributionMap(), 1, ng_pad);
        Btmp[idim].setVal(0.0_rt);
    }

    {
        // Import the coarse data onto per-box coarse patches covering
        // CoarseBox(fill region) = coarsen(fill region) grown by 1.
        // ParallelCopy with the coarse periodicity fills everything inside
        // the (periodically extended) domain; anything beyond a non-periodic
        // domain boundary stays zero, and the corresponding fine ghost cells
        // are excluded from the copy-back below (the domain BCs own them).
        amrex::BoxList cbl;
        for (int i = 0, N = static_cast<int>(fba_cell.size()); i < N; ++i) {
            cbl.push_back(
                amrex::grow(amrex::coarsen(amrex::grow(fba_cell[i], ng_pad), ratio), 1));
        }
        const amrex::BoxArray cpatch_ba(std::move(cbl));
        const amrex::DistributionMapping& fdm = Bfine[0]->DistributionMap();

        if (gate_enabled) {
            EnsureProlongGateMask(lev, cpatch_ba, fdm);
            // First fill after a gate (re)build: the fine coarse-fine
            // ghosts hold no live data yet (fresh init/restart), so a
            // masked fill would freeze zeros into the retained faces. That
            // one fill runs unmasked -- at init both levels hold the same
            // field, so nothing frozen leaks -- and its prolonged state is
            // captured below as the frozen ghost cache.
            masked_fill = !m_mr_prolong_gate_fresh[lev];
            if (masked_fill) {
                // Pre-seed the scratch from the frozen cache: fine faces
                // overlying a masked (frozen) coarse face are skipped by
                // the face interpolation and retain exactly this state.
                // (The live Bfield_fp ghosts are NOT a valid seed source:
                // the RK stage arithmetic manufactures stale-B_old mixtures
                // on ghost faces between fills.)
                for (int idim = 0; idim < 3; ++idim) {
                    amrex::MultiFab::Copy(Btmp[idim], *m_mr_prolong_frozen[lev][idim],
                                          0, 0, 1, ng_pad);
                }
            }
        }

        std::array<amrex::MultiFab, 3> cpatch;
        for (int idim = 0; idim < 3; ++idim) {
            cpatch[idim] = amrex::MultiFab(
                amrex::convert(cpatch_ba, Bcrse[idim]->ixType()), fdm, 1, 0);
            cpatch[idim].setVal(0.0_rt);
            cpatch[idim].ParallelCopy(*Bcrse[idim], 0, 0, 1, cgeom.periodicity());
        }

        // Drive the divergence-free face interpolater FAB by FAB, on the
        // AMREX_SPACEDIM face-staggered components together (in 2D these are
        // Bx and Bz). Its single-component interface intentionally aborts,
        // and the array-interface FillPatch wrapper cannot be used here: in
        // EB-enabled builds it unconditionally constructs an EB factory even
        // when no embedded boundary is in use (AMReX_FillPatchUtil_I.H,
        // InterpFace path).
        amrex::Vector<amrex::Array<amrex::BCRec, AMREX_SPACEDIM>> bcr(1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                bcr[0][d].setLo(dd, amrex::BCType::int_dir);
                bcr[0][d].setHi(dd, amrex::BCType::int_dir);
            }
        }
        const amrex::Array<amrex::IArrayBox*, AMREX_SPACEDIM> no_mask{AMREX_D_DECL(nullptr, nullptr, nullptr)};
        for (amrex::MFIter mfi(Btmp[0], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box fine_region = amrex::grow(fba_cell[mfi.index()], ng_pad);
#if defined(WARPX_DIM_3D)
            const amrex::Array<amrex::FArrayBox*, AMREX_SPACEDIM> crse_fabs{
                &cpatch[0][mfi], &cpatch[1][mfi], &cpatch[2][mfi]};
            const amrex::Array<amrex::FArrayBox*, AMREX_SPACEDIM> fine_fabs{
                &Btmp[0][mfi], &Btmp[1][mfi], &Btmp[2][mfi]};
#else
            const amrex::Array<amrex::FArrayBox*, AMREX_SPACEDIM> crse_fabs{
                &cpatch[0][mfi], &cpatch[2][mfi]};
            const amrex::Array<amrex::FArrayBox*, AMREX_SPACEDIM> fine_fabs{
                &Btmp[0][mfi], &Btmp[2][mfi]};
#endif
            amrex::Array<amrex::IArrayBox*, AMREX_SPACEDIM> mask_fabs = no_mask;
            if (masked_fill) {
#if defined(WARPX_DIM_3D)
                mask_fabs = {&(*m_mr_prolong_gate[lev][0])[mfi],
                             &(*m_mr_prolong_gate[lev][1])[mfi],
                             &(*m_mr_prolong_gate[lev][2])[mfi]};
#else
                mask_fabs = {&(*m_mr_prolong_gate[lev][0])[mfi],
                             &(*m_mr_prolong_gate[lev][2])[mfi]};
#endif
            }
            amrex::face_divfree_interp.interp_arr(
                crse_fabs, 0, fine_fabs, 0, 1, fine_region, ratio, mask_fabs,
                cgeom, fgeom, bcr, 0, 0, amrex::RunOn::Gpu);
        }

#if !defined(WARPX_DIM_3D)
        // In 2D the out-of-plane component By is cell-centered and does not
        // enter div(B): prolong it separately (conservative linear).
        amrex::Vector<amrex::BCRec> bcrec(1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            bcrec[0].setLo(d, amrex::BCType::int_dir);
            bcrec[0].setHi(d, amrex::BCType::int_dir);
        }
        amrex::InterpFromCoarseLevel(
            Btmp[1], ng, amrex::IntVect(0), *Bcrse[1], 0, 0, 1,
            cgeom, fgeom, ratio, &amrex::cell_cons_interp, bcrec, 0);
#endif
    }

    // Capture the frozen ghost cache at the first (unmasked) fill after a
    // gate rebuild: the full prolonged state, from which later masked fills
    // pre-seed their retained faces.
    if (gate_enabled && m_mr_prolong_gate_fresh[lev]) {
        if (static_cast<int>(m_mr_prolong_frozen.size()) <= lev) {
            m_mr_prolong_frozen.resize(lev + 1);
        }
        for (int idim = 0; idim < 3; ++idim) {
            m_mr_prolong_frozen[lev][idim] = std::make_unique<amrex::MultiFab>(
                Btmp[idim].boxArray(), Btmp[idim].DistributionMap(), 1, ng_pad);
            amrex::MultiFab::Copy(*m_mr_prolong_frozen[lev][idim], Btmp[idim],
                                  0, 0, 1, ng_pad);
        }
        m_mr_prolong_gate_fresh[lev] = 0;
    }

    // Copy back the ghost region only: interior valid data must not be
    // touched. Ghost cells covered by same-level valid data (including
    // periodic images) are clobbered here and restored by the FillBoundary
    // below, so only true coarse-fine ghost cells keep prolonged data.
    for (int idim = 0; idim < 3; ++idim) {
        ::CopyGhostRegion(*Bfine[idim], Btmp[idim], fgeom, ng);
    }
    warpx.FillBoundaryB(lev, ng, nodal_sync);
#else
    amrex::ignore_unused(lev, ng, nodal_sync);
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::EnsureRestrictKeepMask (const int clev, amrex::IntVect const& ratio)
{
    auto& warpx = WarpX::GetInstance();
    const amrex::BoxArray& fba = warpx.boxArray(clev + 1);
    const amrex::BoxArray& cba = warpx.boxArray(clev);
    const amrex::DistributionMapping& cdm = warpx.DistributionMap(clev);

    if (static_cast<int>(m_mr_keep_mask.size()) <= clev) {
        m_mr_keep_mask.resize(clev + 1);
        m_mr_keep_mask_fba.resize(clev + 1);
    }
    if (m_mr_keep_mask[clev] &&
        m_mr_keep_mask_fba[clev] == fba &&
        m_mr_keep_mask[clev]->boxArray() == cba &&
        m_mr_keep_mask[clev]->DistributionMap() == cdm)
    {
        return;
    }

    const amrex::Periodicity& period = warpx.Geom(clev).periodicity();
    auto mask = std::make_unique<amrex::iMultiFab>(
        amrex::makeFineMask(cba, cdm, amrex::IntVect(1), fba, ratio, period,
                            /*crse_value=*/0, /*fine_value=*/1));

    // Erode the covered region by m_mr_restrict_setback cells (Linf ball):
    // the band under the fine-patch edge is sacrificial, since its moments
    // are partial by design (buffer particles deposit to the coarse level).
    // Non-periodic domain boundaries do not erode.
    const amrex::Box domain = warpx.Geom(clev).Domain();
    amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = warpx.Geom(clev).isPeriodic(d); }

    amrex::iMultiFab tmp(cba, cdm, 1, 1);
    for (int pass = 0; pass < m_mr_restrict_setback; ++pass)
    {
        mask->FillBoundary(period);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*mask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& src = mask->const_array(mfi);
            auto const& dst = tmp.array(mfi);
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    int v = src(i, j, k);
                    if (v == 1) {
#if defined(WARPX_DIM_3D)
                        for (int dk = -1; dk <= 1 && v; ++dk) {
#else
                        const int dk = 0;
#endif
                        for (int dj = -1; dj <= 1 && v; ++dj) {
                        for (int di = -1; di <= 1 && v; ++di) {
                            if (!hybrid_mr_cell_covered(src, i+di, j+dj, k+dk, domain, is_per)) {
                                v = 0;
                            }
                        }}
#if defined(WARPX_DIM_3D)
                        }
#endif
                    }
                    dst(i, j, k) = v;
                });
        }
        amrex::iMultiFab::Copy(*mask, tmp, 0, 0, 1, 0);
    }
    mask->FillBoundary(period);

    m_mr_keep_mask[clev] = std::move(mask);
    m_mr_keep_mask_fba[clev] = fba;
}

void HybridPICModel::RestrictBfieldFineToCoarse (
    amrex::IntVect const& ng, std::optional<bool> nodal_sync)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();

    for (int flev = warpx.finestLevel(); flev >= 1; --flev)
    {
        const int clev = flev - 1;
        const amrex::IntVect ratio = warpx.refRatio(clev);
        EnsureRestrictKeepMask(clev, ratio);

        ablastr::fields::VectorField Bfine = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, flev);
        ablastr::fields::VectorField Bcrse = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, clev);

        // EB ownership rule (P-2): coarse faces frozen by the coarse
        // staircase are never overwritten by the fine face averages; they
        // stay coarse-owned/frozen.
#ifdef AMREX_USE_EB
        const bool use_eb_gate = EB::enabled() && m_mr_eb_gate_restrict &&
            (warpx.GetEBUpdateBFlag()[clev][0] != nullptr);
#else
        const bool use_eb_gate = false;
#endif

        for (int idim = 0; idim < 3; ++idim)
        {
            // Face-averaged (cell-averaged for the 2D out-of-plane component)
            // restriction onto the coarsened fine layout.
            amrex::MultiFab tmp_cf(
                amrex::coarsen(Bfine[idim]->boxArray(), ratio),
                Bfine[idim]->DistributionMap(), 1, 0);
            if (Bfine[idim]->ixType().cellCentered()) {
                amrex::average_down(*Bfine[idim], tmp_cf, 0, 1, ratio);
            } else {
                amrex::average_down_faces(*Bfine[idim], tmp_cf, ratio, 0);
            }

            // Move onto the coarse-level layout, then overwrite the coarse
            // field only where the keep-mask allows it.
            amrex::MultiFab tmp_c(
                Bcrse[idim]->boxArray(), Bcrse[idim]->DistributionMap(), 1, 0);
            tmp_c.setVal(0.0_rt);
            ablastr::utils::communication::ParallelCopy(
                tmp_c, tmp_cf, 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0),
                WarpX::do_single_precision_comms);
            const amrex::iMultiFab* face_gate = use_eb_gate ?
                warpx.GetEBUpdateBFlag()[clev][idim].get() : nullptr;
            ::MaskedCopyRestricted(*Bcrse[idim], tmp_c, *m_mr_keep_mask[clev],
                                   face_gate, warpx.Geom(clev));
        }

        if (m_mr_check_div_b) { CheckDivBAfterRestriction(clev); }
        warpx.FillBoundaryB(clev, ng, nodal_sync);
    }
#else
    amrex::ignore_unused(ng, nodal_sync);
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::EnsureEmfMatchingRegister (const int flev)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
    const int clev = flev - 1;
    const amrex::BoxArray& fba_cell = warpx.boxArray(flev);
    const amrex::DistributionMapping& fdm = warpx.DistributionMap(flev);
    const amrex::BoxArray& cba = warpx.boxArray(clev);
    const amrex::DistributionMapping& cdm = warpx.DistributionMap(clev);
    const amrex::IntVect ratio = warpx.refRatio(clev);

    if (static_cast<int>(m_mr_emf.size()) <= flev) {
        m_mr_emf.resize(flev + 1);
    }
    EmfMatchingLevel& L = m_mr_emf[flev];
    if (L.reg && L.fba == fba_cell && L.fdm == fdm && L.cba == cba && L.cdm == cdm) {
        return;
    }
    L = EmfMatchingLevel{};

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fba_cell.coarsenable(ratio),
        "Hybrid-PIC MR seam EMF matching: the fine-level BoxArray must be "
        "coarsenable by the refinement ratio.");

    // The commit boundary of the masked restriction is defined by the eroded
    // keep mask; derive the keep region from it (authoritative for multi-box
    // fine unions: the erosion acts on the union, never box by box).
    EnsureRestrictKeepMask(clev, ratio);
    const amrex::iMultiFab& kmask = *m_mr_keep_mask[clev];

    // Bounding box of the keep cells.
    amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMin, amrex::ReduceOpMin,
                     amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax> rops;
    amrex::ReduceData<int, int, int, int, int, int> rdata(rops);
    using RTuple = typename decltype(rdata)::Type;
    for (MFIter mfi(kmask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box tbx = mfi.tilebox();
        auto const& m_arr = kmask.const_array(mfi);
        rops.eval(tbx, rdata,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RTuple
            {
                if (m_arr(i, j, k) == 1) { return {i, j, k, i, j, k}; }
                constexpr int big = std::numeric_limits<int>::max();
                constexpr int small = std::numeric_limits<int>::lowest();
                return {big, big, big, small, small, small};
            });
    }
    auto rv = rdata.value(rops);
    int klo[3] = {amrex::get<0>(rv), amrex::get<1>(rv), amrex::get<2>(rv)};
    int khi[3] = {amrex::get<3>(rv), amrex::get<4>(rv), amrex::get<5>(rv)};
    amrex::ParallelDescriptor::ReduceIntMin(klo, 3);
    amrex::ParallelDescriptor::ReduceIntMax(khi, 3);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(klo[0] <= khi[0],
        "Hybrid-PIC MR seam EMF matching: the restriction keep region is "
        "empty (the setback erosion consumed the whole patch); enlarge the "
        "patch, reduce hybrid_pic_model.mr_restrict_setback, or set "
        "hybrid_pic_model.mr_emf_matching = 0.");
    const amrex::Box kbox(amrex::IntVect(AMREX_D_DECL(klo[0], klo[1], klo[2])),
                          amrex::IntVect(AMREX_D_DECL(khi[0], khi[1], khi[2])));

    // Register-basis safety check: the synthetic register basis below is
    // sound only if the keep region is exactly this rectangle (single static
    // rectangular patches -- all supported deck geometries). Verified bitwise
    // against the authoritative mask on every (re)build.
    amrex::ReduceOps<amrex::ReduceOpSum> vops;
    amrex::ReduceData<amrex::Long> vdata(vops);
    using VTuple = typename decltype(vdata)::Type;
    for (MFIter mfi(kmask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box tbx = mfi.tilebox();
        auto const& m_arr = kmask.const_array(mfi);
        vops.eval(tbx, vdata,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> VTuple
            {
                const bool in = kbox.contains(i, j, k);
                return {((m_arr(i, j, k) == 1) == in) ? amrex::Long(0) : amrex::Long(1)};
            });
    }
    amrex::Long mismatch = amrex::get<0>(vdata.value(vops));
    amrex::ParallelDescriptor::ReduceLongSum(mismatch);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mismatch == 0,
        "Hybrid-PIC MR seam EMF matching: the restriction keep region is not "
        "a single rectangle, which the synthetic edge-flux register basis "
        "requires (single static rectangular patches). Use one rectangular "
        "refinement patch per level, or set "
        "hybrid_pic_model.mr_emf_matching = 0.");

    // Synthetic fine BoxArray whose cell union is the keep region: the
    // register's "fine boundary" IS the commit boundary. Decompose along the
    // actual fine grids so every synthetic box lives inside its fine grid on
    // the same rank (FineAdd then reads the actual fine E fabs directly).
    // Internal per-box seams only register edges adjacent to committed
    // faces, which the restriction overwrites -- benign by construction.
    const amrex::BoxArray cfba = amrex::coarsen(fba_cell, ratio);
    amrex::BoxList sbl;
    amrex::Vector<int> pmap;
    for (int i = 0, N = static_cast<int>(cfba.size()); i < N; ++i) {
        const amrex::Box b = cfba[i] & kbox;
        if (b.ok()) {
            sbl.push_back(amrex::refine(b, ratio));
            pmap.push_back(fdm[i]);
            L.orig.push_back(i);
        }
    }
    const amrex::BoxArray sfba(std::move(sbl));
    const amrex::DistributionMapping sdm(pmap);
    L.reg = std::make_unique<amrex::EdgeFluxRegister>(
        sfba, cba, sdm, cdm, warpx.Geom(flev), warpx.Geom(clev), 1);
    L.iter = std::make_unique<amrex::iMultiFab>(
        sfba, sdm, 1, 0, amrex::MFInfo().SetAlloc(false));

    // EB source gating (applied delta must vanish identically on staircase-
    // frozen coarse edges): cache the per-direction fine-edge gates (parent
    // coarse edge's eb_update_E, imported from VALID coarse data only -- the
    // coarse-fine ghost region of that mask is never marked, the documented
    // trap) and the per-stage E scratch space.
#ifdef AMREX_USE_EB
    const bool gate_enabled = EB::enabled() &&
        (warpx.GetEBUpdateEFlag()[clev][0] != nullptr);
#else
    const bool gate_enabled = false;
#endif
    if (gate_enabled) {
        L.gated = true;
        const amrex::BoxArray csba = amrex::coarsen(sfba, ratio);
        for (int idim = 0; idim < 3; ++idim) {
#if !defined(WARPX_DIM_3D)
            // 2D: only the out-of-plane component (WarpX Ey) enters.
            if (idim != 1) { continue; }
#endif
            const amrex::iMultiFab& ceb = *warpx.GetEBUpdateEFlag()[clev][idim];
            const amrex::IndexType etype = ceb.ixType();
            amrex::iMultiFab cgate(amrex::convert(csba, etype), sdm, 1, 0);
            cgate.setVal(1);
            cgate.ParallelCopy(ceb, 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0),
                               warpx.Geom(clev).periodicity());
            L.gate_f[idim] = std::make_unique<amrex::iMultiFab>(
                amrex::convert(sfba, etype), sdm, 1, 0);
            L.scratch_f[idim] = std::make_unique<amrex::MultiFab>(
                amrex::convert(sfba, etype), sdm, 1, 0);
            L.scratch_c[idim] = std::make_unique<amrex::MultiFab>(
                amrex::convert(cba, etype), cdm, 1, 0);
            // Expand onto the fine edges: each fine edge inherits its parent
            // coarse edge's gate. FineAdd only reads fine edges exactly
            // overlying coarse edges (even transverse indices), where the
            // floor-division coarsening is exact.
            amrex::iMultiFab& gf = *L.gate_f[idim];
            const amrex::IntVect rr = ratio;
            for (MFIter mfi(gf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box tbx = mfi.tilebox();
                auto const& g = gf.array(mfi);
                auto const& cg = cgate.const_array(mfi);
                amrex::ParallelFor(tbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        const int ci = amrex::coarsen(i, rr[0]);
                        const int cj = amrex::coarsen(j, rr[1]);
#if defined(WARPX_DIM_3D)
                        const int ck = amrex::coarsen(k, rr[2]);
#else
                        const int ck = k;
#endif
                        g(i, j, k) = cg(ci, cj, ck);
                    });
            }
        }
    }

    L.fba = fba_cell;
    L.fdm = fdm;
    L.cba = cba;
    L.cdm = cdm;
#else
    amrex::ignore_unused(flev);
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::EmfMatchingReset ()
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
    for (int flev = 1; flev <= warpx.finestLevel(); ++flev) {
        EnsureEmfMatchingRegister(flev);
        m_mr_emf[flev].reg->reset();
    }
#else
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::EmfMatchingAccumulate (
    ablastr::fields::MultiLevelVectorField const& Efield, amrex::Real wdt)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
#if defined(WARPX_DIM_3D)
    const amrex::Real w = wdt;
#else
    // 2D sign: the out-of-plane WarpX component is Ey while the register
    // models dB/dt = -curl(Ez zhat) with its y axis equal to the WarpX z
    // axis (WarpX XZ Faraday: dBx/dt = +dEy/dz, dBz/dt = -dEy/dx).
    const amrex::Real w = -wdt;
#endif
    for (int flev = 1; flev <= warpx.finestLevel(); ++flev)
    {
        EmfMatchingLevel& L = m_mr_emf[flev];
        const int clev = flev - 1;

        if (L.gated) {
            // Refresh the gated stage sources: zero on staircase-frozen
            // coarse edges, so both sides accumulate identical zeros there
            // and the applied delta vanishes.
            for (int idim = 0; idim < 3; ++idim) {
                if (!L.scratch_c[idim]) { continue; }
                amrex::MultiFab& sc = *L.scratch_c[idim];
                const amrex::iMultiFab& ceb = *warpx.GetEBUpdateEFlag()[clev][idim];
                const amrex::MultiFab& Ec = *Efield[clev][idim];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(sc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const amrex::Box tbx = mfi.tilebox();
                    auto const& d = sc.array(mfi);
                    auto const& g = ceb.const_array(mfi);
                    auto const& s = Ec.const_array(mfi);
                    amrex::ParallelFor(tbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            d(i, j, k) = (g(i, j, k) != 0) ? s(i, j, k) : 0.0_rt;
                        });
                }
                amrex::MultiFab& sf = *L.scratch_f[idim];
                const amrex::MultiFab& Ef = *Efield[flev][idim];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(sf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const amrex::Box tbx = mfi.tilebox();
                    auto const& d = sf.array(mfi);
                    auto const& g = L.gate_f[idim]->const_array(mfi);
                    auto const& s = Ef[L.orig[mfi.index()]].const_array();
                    amrex::ParallelFor(tbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            d(i, j, k) = (g(i, j, k) != 0) ? s(i, j, k) : 0.0_rt;
                        });
                }
            }
        }

        // Coarse-side accumulation over the coarse grids (the register skips
        // grids without a coarse-fine interface). CrseAdd/FineAdd require
        // untiled MFIter loops and read valid regions only.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*Efield[clev][0], false); mfi.isValid(); ++mfi)
        {
#if defined(WARPX_DIM_3D)
            const amrex::Array<amrex::FArrayBox const*, 3> ec{
                L.gated ? &(*L.scratch_c[0])[mfi] : &(*Efield[clev][0])[mfi],
                L.gated ? &(*L.scratch_c[1])[mfi] : &(*Efield[clev][1])[mfi],
                L.gated ? &(*L.scratch_c[2])[mfi] : &(*Efield[clev][2])[mfi]};
            L.reg->CrseAdd(mfi, ec, w);
#else
            L.reg->CrseAdd(
                mfi,
                L.gated ? (*L.scratch_c[1])[mfi] : (*Efield[clev][1])[mfi],
                w);
#endif
        }

        // Fine-side accumulation over the synthetic (keep-region) boxes;
        // each lives inside its actual fine grid on the same rank, so the
        // fine E fabs are read directly (no communication).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*L.iter, false); mfi.isValid(); ++mfi)
        {
            const int gi = L.orig[mfi.index()];
#if defined(WARPX_DIM_3D)
            const amrex::Array<amrex::FArrayBox const*, 3> ef{
                L.gated ? &(*L.scratch_f[0])[mfi] : &(*Efield[flev][0])[gi],
                L.gated ? &(*L.scratch_f[1])[mfi] : &(*Efield[flev][1])[gi],
                L.gated ? &(*L.scratch_f[2])[mfi] : &(*Efield[flev][2])[gi]};
            L.reg->FineAdd(mfi, ef, w);
#else
            L.reg->FineAdd(
                mfi,
                L.gated ? (*L.scratch_f[1])[mfi] : (*Efield[flev][1])[gi],
                w);
#endif
        }
    }
    // CrseAdd/FineAdd are asynchronous on GPU builds; the stage E fields are
    // overwritten by the next stage's solve, so synchronize here.
    amrex::Gpu::streamSynchronize();
#else
    amrex::ignore_unused(Efield, wdt);
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::EmfMatchingReflux (
    ablastr::fields::MultiLevelVectorField const& Bfield,
    amrex::IntVect const& ng, std::optional<bool> nodal_sync)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
    for (int flev = warpx.finestLevel(); flev >= 1; --flev)
    {
        const int clev = flev - 1;
#if defined(WARPX_DIM_3D)
        m_mr_emf[flev].reg->Reflux(
            {Bfield[clev][0], Bfield[clev][1], Bfield[clev][2]});
#else
        // 2D: the out-of-plane By is cell-centered and not part of div(B).
        m_mr_emf[flev].reg->Reflux({Bfield[clev][0], Bfield[clev][2]});
#endif
        // With the per-substep restriction cadence (default) the restriction
        // that follows immediately refreshes the coarse ghosts; otherwise
        // refresh them here so the next stage's E solve sees the corrected
        // seam faces.
        if (!m_mr_restrict_every_substep) {
            warpx.FillBoundaryB(clev, ng, nodal_sync);
        }
    }
#else
    amrex::ignore_unused(Bfield, ng, nodal_sync);
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::EnsureFineEdgeMask (const int lev, amrex::IntVect const& ratio)
{
    auto& warpx = WarpX::GetInstance();
    const amrex::BoxArray& fba = warpx.boxArray(lev);
    const amrex::DistributionMapping& fdm = warpx.DistributionMap(lev);

    if (static_cast<int>(m_mr_fine_edge_mask.size()) <= lev) {
        m_mr_fine_edge_mask.resize(lev + 1);
        m_mr_fine_edge_mask_ba.resize(lev + 1);
    }
    if (m_mr_fine_edge_mask[lev] &&
        m_mr_fine_edge_mask_ba[lev] == fba &&
        m_mr_fine_edge_mask[lev]->DistributionMap() == fdm)
    {
        return;
    }

    // Band width: wide enough to cover both the restriction setback band and
    // the deposition-buffer depletion zone (buffer particles deposit to the
    // coarse level, plus one cell of shape-factor reach).
    int width = m_mr_restrict_setback;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { width = std::max(width, m_mr_restrict_setback * ratio[d]); }
    width = std::max(width, WarpX::n_current_deposition_buffer + 2);

    const amrex::Periodicity& period = warpx.Geom(lev).periodicity();
    const amrex::Box domain = warpx.Geom(lev).Domain();
    amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = warpx.Geom(lev).isPeriodic(d); }

    // Start from 1 on the fine union (ghost cells covered by other fine
    // boxes become 1 through FillBoundary; true coarse-fine ghosts stay 0),
    // then erode by `width` cells. Non-periodic domain boundaries do not
    // erode.
    auto mask = std::make_unique<amrex::iMultiFab>(fba, fdm, 1, 1);
    mask->setVal(0);
    mask->setVal(1, 0, 1, amrex::IntVect(0));

    amrex::iMultiFab tmp(fba, fdm, 1, 1);
    for (int pass = 0; pass < width; ++pass)
    {
        mask->FillBoundary(period);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*mask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& src = mask->const_array(mfi);
            auto const& dst = tmp.array(mfi);
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    int v = src(i, j, k);
                    if (v == 1) {
#if defined(WARPX_DIM_3D)
                        for (int dk = -1; dk <= 1 && v; ++dk) {
#else
                        const int dk = 0;
#endif
                        for (int dj = -1; dj <= 1 && v; ++dj) {
                        for (int di = -1; di <= 1 && v; ++di) {
                            if (!hybrid_mr_cell_covered(src, i+di, j+dj, k+dk, domain, is_per)) {
                                v = 0;
                            }
                        }}
#if defined(WARPX_DIM_3D)
                        }
#endif
                    }
                    dst(i, j, k) = v;
                });
        }
        amrex::iMultiFab::Copy(*mask, tmp, 0, 0, 1, 0);
    }
    mask->FillBoundary(period);

    m_mr_fine_edge_mask[lev] = std::move(mask);
    m_mr_fine_edge_mask_ba[lev] = fba;
}

void HybridPICModel::FillMomentsCoarseFineGhosts ()
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    for (int lev = 1; lev <= warpx.finestLevel(); ++lev)
    {
        const amrex::Geometry& cgeom = warpx.Geom(lev-1);
        const amrex::Geometry& fgeom = warpx.Geom(lev);
        const amrex::IntVect ratio = warpx.refRatio(lev-1);
        EnsureFineEdgeMask(lev, ratio);
        const amrex::iMultiFab& edge_mask = *m_mr_fine_edge_mask[lev];

        // EB ownership rule (P-3): the band fill never overwrites the fine
        // level's own staircase-frozen values -- J writes are skipped on
        // frozen fine edges (eb_update_E == 0) and rho writes on nodes
        // buried in the covered region (no uncovered adjacent cell), so
        // coarse in-wall (deposition-starved) moments cannot leak into the
        // fine level's wall band.
#ifdef AMREX_USE_EB
        const bool gate_moments = EB::enabled() && m_mr_eb_gate_moments &&
            (warpx.GetEBUpdateEFlag()[lev][0] != nullptr);
        const amrex::FabArray<amrex::EBCellFlagFab>* eb_flags = gate_moments ?
            &(warpx.fieldEBFactory(lev).getMultiEBCellFlagFab()) : nullptr;
#endif

        // rho (nodal): interpolate the coarse charge density onto a scratch
        // copy of the fine level with bilinear nodal interpolation, then copy
        // back the coarse-fine ghost region only.
        {
            amrex::MultiFab& rho_f = *warpx.m_fields.get(FieldType::rho_fp, lev);
            amrex::MultiFab& rho_c = *warpx.m_fields.get(FieldType::rho_fp, lev-1);
            const amrex::IntVect ng = rho_f.nGrowVect();
            const int ncomp = rho_f.nComp();

            amrex::MultiFab rtmp(rho_f.boxArray(), rho_f.DistributionMap(), ncomp, ng);
            rtmp.setVal(0.0_rt);
            amrex::Vector<amrex::BCRec> bcrec(ncomp);
            for (int n = 0; n < ncomp; ++n) {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    bcrec[n].setLo(d, amrex::BCType::int_dir);
                    bcrec[n].setHi(d, amrex::BCType::int_dir);
                }
            }
            amrex::InterpFromCoarseLevel(
                rtmp, ng, amrex::IntVect(0), rho_c, 0, 0, ncomp,
                cgeom, fgeom, ratio, &amrex::node_bilinear_interp, bcrec, 0);
            // Copy back the coarse-fine ghost region plus the valid cells of
            // the sacrificial edge band (edge_mask == 0).
            {
                const amrex::IntVect stag = rho_f.ixType().toIntVect();
                const amrex::Box allowed = amrex::convert(
                    fgeom.growPeriodicDomain(ng), rho_f.ixType());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(rho_f, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const amrex::Box vbx = mfi.validbox();
                    amrex::Box gbx = mfi.growntilebox(ng);
                    gbx &= allowed;
                    if (!gbx.ok()) { continue; }
                    auto const& d_arr = rho_f.array(mfi);
                    auto const& s_arr = rtmp.const_array(mfi);
                    auto const& e_arr = edge_mask.const_array(mfi);
#ifdef AMREX_USE_EB
                    amrex::Array4<amrex::EBCellFlag const> f_arr =
                        (eb_flags != nullptr) ? eb_flags->const_array(mfi)
                                              : amrex::Array4<amrex::EBCellFlag const>{};
                    const int oi = stag[0];
                    const int oj = (AMREX_SPACEDIM > 1) ? stag[1] : 0;
                    const int ol = (AMREX_SPACEDIM > 2) ? stag[2] : 0;
#endif
                    amrex::ParallelFor(gbx, ncomp,
                        [=] AMREX_GPU_DEVICE (int j, int k, int l, int n)
                        {
#ifdef AMREX_USE_EB
                            if (f_arr) {
                                // Node validity: any adjacent cell uncovered.
                                // Cells beyond the flag arrays count as
                                // uncovered (deep ghosts stay writable).
                                bool any_unc = false;
                                for         (int dl = -ol; dl <= 0; ++dl) {
                                    for     (int dk = -oj; dk <= 0; ++dk) {
                                        for (int dj = -oi; dj <= 0; ++dj) {
                                            if (!f_arr.contains(j+dj, k+dk, l+dl) ||
                                                !f_arr(j+dj, k+dk, l+dl).isCovered()) {
                                                any_unc = true;
                                            }
                                        }
                                    }
                                }
                                if (!any_unc) { return; }
                            }
#endif
                            if (!vbx.contains(j, k, l) ||
                                hybrid_mr_point_in_band(j, k, l, stag, e_arr)) {
                                d_arr(j, k, l, n) = s_arr(j, k, l, n);
                            }
                        });
                }
            }
            ablastr::utils::communication::FillBoundary(
                rho_f, ng, WarpX::do_single_precision_comms,
                fgeom.periodicity(), true);
        }

        // J (edge-staggered): AMReX offers no edge interpolater usable through
        // FillPatch (FaceLinear covers faces only, NodeBilinear nodes only),
        // so import the coarse current onto the coarsened fine layout and
        // apply a staggering-aware linear interpolation kernel by hand.
        for (int idim = 0; idim < 3; ++idim)
        {
            amrex::MultiFab& J_f = *warpx.m_fields.get(FieldType::current_fp, Direction{idim}, lev);
            const amrex::MultiFab& J_c = *warpx.m_fields.get(FieldType::current_fp, Direction{idim}, lev-1);
            const amrex::IntVect ng = J_f.nGrowVect();

            // Scratch on the coarsened fine layout with enough ghost cells
            // for the interpolation stencil under the fine ghost region.
            amrex::IntVect ngc;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { ngc[d] = ng[d]/ratio[d] + 2; }
            amrex::MultiFab jtmp(
                amrex::coarsen(J_f.boxArray(), ratio), J_f.DistributionMap(), 1, ngc);
            jtmp.setVal(0.0_rt);
            ablastr::utils::communication::ParallelCopy(
                jtmp, J_c, 0, 0, 1, amrex::IntVect(0), ngc,
                WarpX::do_single_precision_comms, cgeom.periodicity());

            const amrex::IntVect stag = J_f.ixType().toIntVect();
            const amrex::Box allowed = amrex::convert(
                fgeom.growPeriodicDomain(ng), J_f.ixType());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(J_f, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box vbx = mfi.validbox();
                amrex::Box gbx = mfi.growntilebox(ng);
                gbx &= allowed;
                if (!gbx.ok()) { continue; }
                auto const& j_arr = J_f.array(mfi);
                auto const& c_arr = jtmp.const_array(mfi);
                auto const& e_arr = edge_mask.const_array(mfi);
#ifdef AMREX_USE_EB
                amrex::Array4<amrex::EBCellFlag const> f_arr =
                    (eb_flags != nullptr) ? eb_flags->const_array(mfi)
                                          : amrex::Array4<amrex::EBCellFlag const>{};
                const int oi = stag[0];
                const int oj = (AMREX_SPACEDIM > 1) ? stag[1] : 0;
                const int ol = (AMREX_SPACEDIM > 2) ? stag[2] : 0;
#endif
                amrex::ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int j, int k, int l)
                    {
#ifdef AMREX_USE_EB
                        if (f_arr) {
                            // Staircase rule (same as MarkUpdateCellsStairCase,
                            // but from the geometric flags, which are valid in
                            // the coarse-fine ghost region where eb_update_E
                            // is never marked): an edge with any non-regular
                            // adjacent cell is frozen and keeps its own
                            // (zero) value. Cells beyond the flag arrays
                            // count as regular.
                            bool frozen = false;
                            for         (int dl = -ol; dl <= 0; ++dl) {
                                for     (int dk = -oj; dk <= 0; ++dk) {
                                    for (int dj = -oi; dj <= 0; ++dj) {
                                        if (f_arr.contains(j+dj, k+dk, l+dl) &&
                                            !f_arr(j+dj, k+dk, l+dl).isRegular()) {
                                            frozen = true;
                                        }
                                    }
                                }
                            }
                            if (frozen) { return; }
                        }
#endif
                        if (!vbx.contains(j, k, l) ||
                            hybrid_mr_point_in_band(j, k, l, stag, e_arr)) {
                            j_arr(j, k, l) = hybrid_mr_interp_from_coarse(
                                j, k, l, 0, c_arr, stag, ratio);
                        }
                    });
            }
            ablastr::utils::communication::FillBoundary(
                J_f, ng, WarpX::do_single_precision_comms,
                fgeom.periodicity(), true);
        }
    }
#else
    WARPX_ABORT_WITH_MESSAGE(
        "Hybrid-PIC mesh refinement is only implemented for Cartesian 2D/3D.");
#endif
}

void HybridPICModel::CheckDivBAfterGhostFill (const int lev, amrex::IntVect const& ng)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField B = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, lev);
    const amrex::Geometry& geom = warpx.Geom(lev);

    if (static_cast<int>(m_divb_fine_valid.size()) <= lev) {
        m_divb_fine_valid.resize(lev + 1);
        m_divb_fine_ring.resize(lev + 1);
        m_divb_fine_band.resize(lev + 1);
        m_divb_fine_wall.resize(lev + 1);
    }

    // EB wall-band classification: the P-1 commit granule is the coarse
    // face, so a fine ghost cell belongs to the wall band iff its PARENT
    // coarse cell has a staircase-frozen face. Cells there legitimately mix
    // retained (live fine) and prolonged faces and are reported in their own
    // class instead of polluting the ring/band statistics.
    std::unique_ptr<amrex::iMultiFab> wall_cf;
    amrex::IntVect wall_ratio(1);
#ifdef AMREX_USE_EB
    if (EB::enabled() && warpx.GetEBUpdateBFlag()[lev-1][0]) {
        const int clev = lev - 1;
        const amrex::IntVect ratio = warpx.refRatio(clev);
        const amrex::iMultiFab wall_crse = ::MakeFrozenFaceCellIndicator(
            warpx.GetEBUpdateBFlag()[clev], warpx.boxArray(clev),
            warpx.DistributionMap(clev), 1);
        amrex::IntVect ngc;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { ngc[d] = ng[d]/ratio[d] + 2; }
        wall_cf = std::make_unique<amrex::iMultiFab>(
            amrex::coarsen(warpx.boxArray(lev), ratio),
            warpx.DistributionMap(lev), 1, ngc);
        wall_cf->setVal(0);
        wall_cf->ParallelCopy(wall_crse, 0, 0, 1, amrex::IntVect(1), ngc,
                              warpx.Geom(clev).periodicity());
        wall_ratio = ratio;
    }
#endif

    const amrex::Real bmax = ::MaxAbsB(B);
    amrex::Real dxmin = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dxmin = std::min(dxmin, geom.CellSize(d)); }
    const amrex::Real scale = (bmax > 0._rt) ? dxmin / bmax : 1._rt;

    const int ngmin = ng.min();
    // Fine valid region: untouched by the fill, curl updates preserve div(B).
    const amrex::Real v = ::MaxAbsDivB(B, geom, 0, -1, nullptr, 0);
    // First ghost ring: cells mixing owned fine faces (patch boundary) with
    // prolonged faces; carries the coarse/fine face mismatch at the seam.
    const amrex::Real r1 = ::MaxAbsDivB(B, geom, 1, 0, nullptr, 0,
                                        wall_cf.get(), wall_ratio, 1);
    // Outer ghost band: fully prolonged cells, div(B) inherits the coarse
    // value (machine zero if the coarse level is divergence free).
    const amrex::Real rb = (ngmin >= 2) ?
        ::MaxAbsDivB(B, geom, ngmin - 1, 1, nullptr, 0,
                     wall_cf.get(), wall_ratio, 1) : 0._rt;
    // Wall band (ghost region only): mixed retained/prolonged faces.
    const amrex::Real rw = wall_cf ?
        ::MaxAbsDivB(B, geom, std::max(ngmin - 1, 1), 0, nullptr, 0,
                     wall_cf.get(), wall_ratio, 2) : 0._rt;

    m_divb_fine_valid[lev].update(v, v * scale);
    m_divb_fine_ring[lev].update(r1, r1 * scale);
    m_divb_fine_band[lev].update(rb, rb * scale);
    m_divb_fine_wall[lev].update(rw, rw * scale);

    if (v * scale > 1.e-12_rt || rb * scale > 1.e-12_rt) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "Hybrid MR div(B) audit: fine level " + std::to_string(lev) +
            " valid region or prolonged ghost band exceeds 1e-12 relative "
            "div(B) after the coarse-fine ghost fill.",
            ablastr::warn_manager::WarnPriority::high);
    }
#else
    amrex::ignore_unused(lev, ng);
#endif
}

void HybridPICModel::CheckDivBAfterRestriction (const int clev)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField B = warpx.m_fields.get_alldirs(FieldType::Bfield_fp, clev);
    const amrex::Geometry& geom = warpx.Geom(clev);

    if (static_cast<int>(m_divb_crse_interior.size()) <= clev) {
        m_divb_crse_interior.resize(clev + 1);
        m_divb_crse_seam.resize(clev + 1);
        m_divb_crse_exterior.resize(clev + 1);
        m_divb_crse_wall.resize(clev + 1);
    }

    // EB wall-seam classification: cells with a staircase-frozen face mix
    // frozen (never-restricted) and restricted faces; they are reported in
    // their own class so the interior/seam-ring statistics keep their
    // divergence-consistency meaning.
    std::unique_ptr<amrex::iMultiFab> wall_cc;
#ifdef AMREX_USE_EB
    if (EB::enabled() && warpx.GetEBUpdateBFlag()[clev][0]) {
        wall_cc = std::make_unique<amrex::iMultiFab>(
            ::MakeFrozenFaceCellIndicator(
                warpx.GetEBUpdateBFlag()[clev], warpx.boxArray(clev),
                warpx.DistributionMap(clev), 0));
    }
#endif

    const amrex::Real bmax = ::MaxAbsB(B);
    amrex::Real dxmin = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dxmin = std::min(dxmin, geom.CellSize(d)); }
    const amrex::Real scale = (bmax > 0._rt) ? dxmin / bmax : 1._rt;

    const amrex::iMultiFab* mask = m_mr_keep_mask[clev].get();
    const amrex::IntVect unit_ratio(1);
    // Strict interior (all faces restricted): face averages of a
    // divergence-free fine field, expected machine zero.
    const amrex::Real vi = ::MaxAbsDivB(B, geom, 0, -1, mask, 2,
                                        wall_cc.get(), unit_ratio, 1);
    // Seam ring (outermost keep-mask layer): cells mixing restricted and
    // freely evolved coarse faces show O(dt)-accumulated div(B) without EMF
    // matching (reported, not hidden -- EMF flux matching is deliberately a
    // later phase).
    const amrex::Real vs = ::MaxAbsDivB(B, geom, 0, -1, mask, 1,
                                        wall_cc.get(), unit_ratio, 1);
    // Exterior (no restricted faces): freely evolved, divergence preserving.
    const amrex::Real ve = ::MaxAbsDivB(B, geom, 0, -1, mask, 0);
    // Wall seam (interior or seam-ring cells with a frozen face).
    const amrex::Real vw = wall_cc ?
        ::MaxAbsDivB(B, geom, 0, -1, mask, -1, wall_cc.get(), unit_ratio, 2)
        : 0._rt;

    m_divb_crse_interior[clev].update(vi, vi * scale);
    m_divb_crse_seam[clev].update(vs, vs * scale);
    m_divb_crse_exterior[clev].update(ve, ve * scale);
    m_divb_crse_wall[clev].update(vw, vw * scale);

    if (vi * scale > 1.e-12_rt) {
        ablastr::warn_manager::WMRecordWarning(
            "HybridPIC",
            "Hybrid MR div(B) audit: coarse level " + std::to_string(clev) +
            " restricted interior exceeds 1e-12 relative div(B) after "
            "restriction.",
            ablastr::warn_manager::WarnPriority::high);
    }
#else
    amrex::ignore_unused(clev);
#endif
}

void HybridPICModel::PrintDivBDiagnostics ()
{
    amrex::Print() << "Hybrid MR div(B) audit (max over step; rel = |divB|*dx/max|B|):\n";
    for (int lev = 1; lev < static_cast<int>(m_divb_fine_valid.size()); ++lev) {
        amrex::Print() << "  fine lev " << lev
            << ": valid raw " << m_divb_fine_valid[lev].raw
            << " rel " << m_divb_fine_valid[lev].rel
            << " | ghost ring raw " << m_divb_fine_ring[lev].raw
            << " rel " << m_divb_fine_ring[lev].rel
            << " | ghost band raw " << m_divb_fine_band[lev].raw
            << " rel " << m_divb_fine_band[lev].rel
            << " | ghost wall raw " << m_divb_fine_wall[lev].raw
            << " rel " << m_divb_fine_wall[lev].rel << "\n";
    }
    for (int clev = 0; clev < static_cast<int>(m_divb_crse_interior.size()); ++clev) {
        amrex::Print() << "  crse lev " << clev
            << ": interior raw " << m_divb_crse_interior[clev].raw
            << " rel " << m_divb_crse_interior[clev].rel
            << " | seam ring raw " << m_divb_crse_seam[clev].raw
            << " rel " << m_divb_crse_seam[clev].rel
            << " | exterior raw " << m_divb_crse_exterior[clev].raw
            << " rel " << m_divb_crse_exterior[clev].rel
            << " | wall seam raw " << m_divb_crse_wall[clev].raw
            << " rel " << m_divb_crse_wall[clev].rel << "\n";
    }
    for (auto& s : m_divb_fine_valid)    { s = DivBStats{}; }
    for (auto& s : m_divb_fine_ring)     { s = DivBStats{}; }
    for (auto& s : m_divb_fine_band)     { s = DivBStats{}; }
    for (auto& s : m_divb_fine_wall)     { s = DivBStats{}; }
    for (auto& s : m_divb_crse_interior) { s = DivBStats{}; }
    for (auto& s : m_divb_crse_seam)     { s = DivBStats{}; }
    for (auto& s : m_divb_crse_exterior) { s = DivBStats{}; }
    for (auto& s : m_divb_crse_wall)     { s = DivBStats{}; }
}

void HybridPICModel::CheckMREBClearance (const bool verbose) const
{
#ifdef AMREX_USE_EB
    if (!EB::enabled()) { return; }
    auto& warpx = WarpX::GetInstance();
    if (warpx.finestLevel() < 1) { return; }

    // Refinement ratio (the MR validation block enforces 2 between all
    // levels; keep the max over levels for generality).
    int rmax = 2;
    for (int lev = 0; lev < warpx.finestLevel(); ++lev) {
        rmax = std::max(rmax, warpx.refRatio(lev).max());
    }

    // Required clearance (coarse cells): the max footprint, measured from the
    // coarse-fine patch boundary, over every EB-blind interlevel operator.
    //  - restriction: the sacrificial setback ring plus the face average
    //    reaching one cell past it;
    //  - moment band fill: the fine-edge band width, in coarse cells;
    //  - B ghost fill: the coarse patch feeding FaceDivFree covers
    //    coarsen(grow(fine box, ng_pad)) grown by one;
    //  - particle buffers: buffer particles within the deposition/gather
    //    buffer of the patch edge deposit to (gather from) the coarse level
    //    with the full particle shape.
    const int setback = m_mr_restrict_setback;
    const int c_restrict = setback + 1;
    const int dep_buf = std::max(WarpX::n_current_deposition_buffer, 0);
    const int gat_buf = std::max(WarpX::n_field_gather_buffer, 0);
    const int fine_band = std::max(setback * rmax, dep_buf + 2);
    const int c_moment = (fine_band + rmax - 1) / rmax;
    const int ng_fs = warpx.getngFieldSolver().max();
    const int c_bfill = (ng_fs + rmax - 1) / rmax + 1;
    const int c_buffers = (std::max(dep_buf, gat_buf) + WarpX::nox + rmax - 1) / rmax;
    const int n_clear_auto = std::max({c_restrict, c_moment, c_bfill, c_buffers, 4});

    const bool overridden = (m_mr_eb_clearance_cells >= 0);
    const int n_clear = overridden ? m_mr_eb_clearance_cells : n_clear_auto;

    if (verbose) {
        amrex::Print() << "Hybrid-PIC MR + EB clearance guard: N_clear = "
            << n_clear << " coarse cells"
            << (overridden ? " (user override; auto value " : " (auto: restriction ")
            << (overridden ? std::to_string(n_clear_auto)
                           : std::to_string(c_restrict) + ", moment band " +
                             std::to_string(c_moment) + ", B ghost fill " +
                             std::to_string(c_bfill) + ", particle buffers " +
                             std::to_string(c_buffers) + ", floor 4")
            << ")";
        if (m_mr_eb_clearance_waive != amrex::IntVect(0)) {
            amrex::Print() << "; waived dims:";
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (m_mr_eb_clearance_waive[d]) { amrex::Print() << " " << d; }
            }
        }
        amrex::Print() << "\n";
    }
    if (n_clear == 0) {
        if (verbose) {
            ablastr::warn_manager::WMRecordWarning(
                "HybridPIC",
                "hybrid_pic_model.mr_eb_clearance_cells = 0: the EB clearance "
                "guard is disabled. EB-blind coarse-fine operators touching a "
                "cut cell corrupt fields silently.",
                ablastr::warn_manager::WarnPriority::high);
        }
        return;
    }

    constexpr auto sentinel = std::numeric_limits<amrex::Long>::max();

    for (int lev = 1; lev <= warpx.finestLevel(); ++lev)
    {
        const int clev = lev - 1;
        const amrex::IntVect ratio = warpx.refRatio(clev);
        const amrex::BoxArray& cba = warpx.boxArray(clev);
        const amrex::DistributionMapping& cdm = warpx.DistributionMap(clev);
        const amrex::Geometry& cgeom = warpx.Geom(clev);
        const amrex::Box cdomain = cgeom.Domain();

        // Coarsened fine region; along waived dimensions each box is
        // stretched to the domain ends, so a fine/crse transition through a
        // patch face normal to a waived dimension disappears (no violation
        // there), while the other faces stay guarded and corners near
        // guarded faces still trigger.
        amrex::BoxArray cfba = amrex::coarsen(warpx.boxArray(lev), ratio);
        if (m_mr_eb_clearance_waive != amrex::IntVect(0)) {
            amrex::BoxList bl = cfba.boxList();
            for (auto& b : bl) {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (m_mr_eb_clearance_waive[d]) {
                        b.setSmall(d, cdomain.smallEnd(d));
                        b.setBig(d, cdomain.bigEnd(d));
                    }
                }
            }
            cfba = amrex::BoxArray(std::move(bl));
        }

        // Indicator of the (stretched) coarsened fine region on the coarse
        // layout, with N_clear ghost cells; FillBoundary makes the ghosts
        // correct across box seams and periodic boundaries. Ghost cells
        // beyond a non-periodic domain boundary keep 0 ("not fine"); the
        // ball scan below is clamped to the domain along non-periodic
        // dimensions, since no coarse-fine operator acts outside the domain
        // (with a stretched box reaching a non-periodic end, those default
        // ghosts would otherwise read as a false "crse" transition).
        amrex::iMultiFab fine_mask(cba, cdm, 1, n_clear);
        fine_mask.setVal(0);
        for (amrex::MFIter mfi(fine_mask); mfi.isValid(); ++mfi) {
            for (const auto& is : cfba.intersections(mfi.validbox())) {
                fine_mask[mfi].template setVal<amrex::RunOn::Device>(1, is.second, 0, 1);
            }
        }
        fine_mask.FillBoundary(cgeom.periodicity());

        auto const& flags = warpx.fieldEBFactory(clev).getMultiEBCellFlagFab();
        const amrex::Long dnpts = cdomain.numPts();
        amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = cgeom.isPeriodic(d); }
        const auto dlo = amrex::lbound(cdomain);
        const auto dhi = amrex::ubound(cdomain);

        // For every cut cell of the parent level, find the smallest radius
        // r <= N_clear whose Linf ball contains both fine-region and
        // non-fine-region cells (i.e. the distance to the patch boundary,
        // from either side); encode (r, cell) and reduce to the worst
        // offender. Regular and fully covered cells never violate.
        amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Long> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(fine_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& m_arr = fine_mask.const_array(mfi);
            auto const& f_arr = flags.const_array(mfi);
            const int nc = n_clear;
            reduce_op.eval(tbx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (!f_arr(i,j,k).isSingleValued()) { return {sentinel}; }
                    bool saw_fine = (m_arr(i,j,k) == 1);
                    bool saw_crse = !saw_fine;
                    for (int r = 1; r <= nc; ++r) {
                        const int jr = (AMREX_SPACEDIM >= 2) ? r : 0;
                        const int kr = (AMREX_SPACEDIM == 3) ? r : 0;
                        for         (int kk = -kr; kk <= kr; ++kk) {
                            for     (int jj = -jr; jj <= jr; ++jj) {
                                for (int ii = -r;  ii <= r;  ++ii) {
                                    const int i2 = i + ii;
                                    const int j2 = j + jj;
                                    const int k2 = k + kk;
                                    // Clamp to the domain along non-periodic
                                    // dimensions (see the indicator note).
                                    bool outside =
                                        (!is_per[0] && (i2 < dlo.x || i2 > dhi.x));
#if (AMREX_SPACEDIM >= 2)
                                    outside = outside ||
                                        (!is_per[1] && (j2 < dlo.y || j2 > dhi.y));
#endif
#if (AMREX_SPACEDIM == 3)
                                    outside = outside ||
                                        (!is_per[2] && (k2 < dlo.z || k2 > dhi.z));
#endif
                                    if (outside) { continue; }
                                    if (m_arr(i2, j2, k2) == 1) { saw_fine = true; }
                                    else                        { saw_crse = true; }
                                }
                            }
                        }
                        if (saw_fine && saw_crse) {
                            return {static_cast<amrex::Long>(r) * dnpts
                                    + cdomain.index(amrex::IntVect(AMREX_D_DECL(i,j,k)))};
                        }
                    }
                    return {sentinel};
                });
        }
        amrex::Long enc = amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelDescriptor::ReduceLongMin(enc);

        if (enc != sentinel) {
            const int r_meas = static_cast<int>(enc / dnpts);
            const amrex::IntVect iv = cdomain.atOffset(enc % dnpts);
            // Nearest coarsened fine box (Linf), for the message.
            amrex::Box near_box;
            int near_d = std::numeric_limits<int>::max();
            for (int ib = 0, nb = static_cast<int>(cfba.size()); ib < nb; ++ib) {
                const amrex::Box b = cfba[ib];
                int d = 0;
                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) {
                    d = std::max({d, b.smallEnd(dd) - iv[dd], iv[dd] - b.bigEnd(dd)});
                }
                if (d < near_d) { near_d = d; near_box = b; }
            }
            std::stringstream ss;
            ss << "Hybrid-PIC MR + EB clearance violation: EB cut cell " << iv
               << " on level " << clev << " lies " << r_meas
               << " coarse cell(s) (Linf) from the boundary of the level-" << lev
               << " patch " << near_box << " (coarsened index space), closer than "
               << "the required clearance of " << n_clear << " cells. The "
               << "coarse-fine operators (B ghost fill/restriction, moment band "
               << "fill, particle buffers) are EB-blind: keep every fine-patch "
               << "boundary at least N_clear cells away from the embedded "
               << "boundary (an EB strictly interior to the patch is fine). "
               << "Move or resize the refinement patch, or override "
               << "hybrid_pic_model.mr_eb_clearance_cells at your own risk.";
            amrex::Abort(ss.str());
        }
    }
#else
    amrex::ignore_unused(verbose);
#endif
}

void HybridPICModel::CheckFineLevelDensityFloor () const
{
    auto& warpx = WarpX::GetInstance();
    if (warpx.finestLevel() < 1) { return; }

    const amrex::Real rho_floor = PhysConst::q_e * m_n_floor;
    const amrex::Real rho_thresh = 1.05_rt * rho_floor;
#ifdef AMREX_USE_EB
    const bool eb_enabled = EB::enabled();
#endif

    for (int lev = 1; lev <= warpx.finestLevel(); ++lev)
    {
        const amrex::MultiFab& rho = *warpx.m_fields.get(FieldType::rho_fp, lev);
        const amrex::IndexType itype = rho.ixType();
        const int si = itype.nodeCentered(0) ? 1 : 0;
        const int sj = (AMREX_SPACEDIM >= 2 && itype.nodeCentered(1)) ? 1 : 0;
        const int sk = (AMREX_SPACEDIM == 3 && itype.nodeCentered(2)) ? 1 : 0;
#if !defined(AMREX_USE_EB)
        amrex::ignore_unused(si, sj, sk);
#endif

#ifdef AMREX_USE_EB
        const amrex::FabArray<amrex::EBCellFlagFab>* flags = eb_enabled ?
            &(warpx.fieldEBFactory(lev).getMultiEBCellFlagFab()) : nullptr;
#endif

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Long, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(rho, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& r_arr = rho.const_array(mfi);
#ifdef AMREX_USE_EB
            amrex::Array4<amrex::EBCellFlag const> f_arr = (flags != nullptr) ?
                flags->const_array(mfi) : amrex::Array4<amrex::EBCellFlag const>{};
#endif
            reduce_op.eval(tbx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    constexpr auto rbig = std::numeric_limits<amrex::Real>::max();
                    if (r_arr(i,j,k) > rho_thresh) { return {0, rbig}; }
#ifdef AMREX_USE_EB
                    if (f_arr) {
                        // Skip points buried in the covered region: they are
                        // frozen by the EB masks and never enter the E solve.
                        bool any_uncovered = false;
                        for         (int kk = k - sk; kk <= k; ++kk) {
                            for     (int jj = j - sj; jj <= j; ++jj) {
                                for (int ii = i - si; ii <= i; ++ii) {
                                    if (!f_arr(ii,jj,kk).isCovered()) { any_uncovered = true; }
                                }
                            }
                        }
                        if (!any_uncovered) { return {0, rbig}; }
                    }
#endif
                    return {1, r_arr(i,j,k)};
                });
        }
        auto rv = reduce_data.value(reduce_op);
        auto count = amrex::get<0>(rv);
        auto rho_min = amrex::get<1>(rv);
        amrex::ParallelDescriptor::ReduceLongSum(count);
        amrex::ParallelDescriptor::ReduceRealMin(rho_min);

        if (count > 0) {
            std::stringstream ss;
            ss << "Hybrid-PIC MR: " << count << " point(s) on refinement level "
               << lev << " start at or below 1.05x the density floor "
               << "hybrid_pic_model.n_floor (min n/n_floor = "
               << rho_min / rho_floor << "). Fine-resolution cells at the "
               << "density floor host a noise-seeded, physical-rate "
               << "instability that mesh refinement amplifies 2-4x (T1.7 "
               << "battery); stabilizing it required a plasma resistivity "
               << "floor ~1e-6 normalized (10-30x the wave-quiet floors). "
               << "Keep refinement patches out of floor regions, carry a "
               << "sufficient plasma_resistivity floor, or raise n_floor.";
            ablastr::warn_manager::WMRecordWarning(
                "HybridPIC", ss.str(),
                ablastr::warn_manager::WarnPriority::high);
        }
    }
}
