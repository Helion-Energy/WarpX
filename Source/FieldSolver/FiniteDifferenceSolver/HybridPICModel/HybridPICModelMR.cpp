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
 *  - an optional runtime div(B) audit of both operators.
 */

#include "HybridPICModel.H"

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFInterpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <memory>

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
     * cells are covered. */
    void MaskedCopyRestricted (amrex::MultiFab& dst, amrex::MultiFab const& src,
                               amrex::iMultiFab const& mask,
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
            amrex::ParallelFor(tbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
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
     */
    amrex::Real MaxAbsDivB (
        ablastr::fields::VectorField const& B,
        amrex::Geometry const& geom,
        int region_grow, int exclude_grow,
        amrex::iMultiFab const* mask, int region_sel)
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
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    if (has_excl && excl.contains(i, j, k)) { return {0._rt}; }
                    if (m_arr) {
                        int region = 0;
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
                        if (region != region_sel) { return {0._rt}; }
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
#endif // Cartesian 2D/3D
}

void HybridPICModel::FillBfieldCoarseFineGhosts (
    const int lev, amrex::IntVect const& ng, std::optional<bool> nodal_sync)
{
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    auto& warpx = WarpX::GetInstance();

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
            amrex::face_divfree_interp.interp_arr(
                crse_fabs, 0, fine_fabs, 0, 1, fine_region, ratio, no_mask,
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
            ::MaskedCopyRestricted(*Bcrse[idim], tmp_c, *m_mr_keep_mask[clev], warpx.Geom(clev));
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

void HybridPICModel::EnsureSeamBandRamp (const int lev)
{
    auto& warpx = WarpX::GetInstance();
    const amrex::BoxArray& fba = warpx.boxArray(lev);
    const amrex::DistributionMapping& fdm = warpx.DistributionMap(lev);

    if (static_cast<int>(m_mr_seam_ramp.size()) <= lev) {
        m_mr_seam_ramp.resize(lev + 1);
        m_mr_seam_ramp_ba.resize(lev + 1);
    }
    if (m_mr_seam_ramp[lev] &&
        m_mr_seam_ramp_ba[lev] == fba &&
        m_mr_seam_ramp[lev]->DistributionMap() == fdm)
    {
        return;
    }

    const int width = m_mr_seam_band_width;
    const amrex::Periodicity& period = warpx.Geom(lev).periodicity();
    const amrex::Box domain = warpx.Geom(lev).Domain();
    amrex::GpuArray<int, AMREX_SPACEDIM> is_per{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = warpx.Geom(lev).isPeriodic(d); }

    // Linf cell distance to the coarse-fine patch edge by repeated erosion
    // (same construction as EnsureFineEdgeMask): after pass p the mask is 1
    // only on cells more than p cells from the fine-union edge, so
    // accumulating the mask gives dist = min(d - 1, width) with d >= 1 the
    // cell distance to the nearest non-fine cell. Non-periodic domain
    // boundaries do not erode (they are not coarse-fine seams).
    amrex::iMultiFab mask(fba, fdm, 1, 1);
    mask.setVal(0);
    mask.setVal(1, 0, 1, amrex::IntVect(0));
    amrex::iMultiFab dist(fba, fdm, 1, 0);
    dist.setVal(0);
    amrex::iMultiFab tmp(fba, fdm, 1, 1);
    for (int pass = 0; pass < width; ++pass)
    {
        mask.FillBoundary(period);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box tbx = mfi.tilebox();
            auto const& src = mask.const_array(mfi);
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
        amrex::iMultiFab::Copy(mask, tmp, 0, 0, 1, 0);
        amrex::iMultiFab::Add(dist, mask, 0, 0, 1, 0);
    }

    // Half-cosine ramp: 1 at the patch-edge cells (dist = 0) falling to 0
    // at `width` cells inward. Ghost cells start at 1 so true coarse-fine
    // ghosts read as "at the edge"; ghosts covered by fine valid data are
    // restored by the FillBoundary, and ghosts beyond a non-periodic
    // domain boundary are filled by first-order extrapolation.
    auto ramp = std::make_unique<amrex::MultiFab>(fba, fdm, 1, 1);
    ramp->setVal(1.0_rt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*ramp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box tbx = mfi.tilebox();
        auto const& d_arr = dist.const_array(mfi);
        auto const& r_arr = ramp->array(mfi);
        const auto winv = 1.0_rt / static_cast<amrex::Real>(width);
        amrex::ParallelFor(tbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const amrex::Real s = amrex::min(
                    static_cast<amrex::Real>(d_arr(i, j, k)) * winv, 1.0_rt);
                r_arr(i, j, k) = 0.5_rt * (1.0_rt + std::cos(MathConst::pi * s));
            });
    }
    ramp->FillBoundary(period);
    if (!warpx.Geom(lev).isAllPeriodic()) {
        amrex::Vector<amrex::BCRec> bcr(1);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const auto bct = warpx.Geom(lev).isPeriodic(d)
                ? amrex::BCType::int_dir : amrex::BCType::foextrap;
            bcr[0].setLo(d, bct);
            bcr[0].setHi(d, bct);
        }
        amrex::FillDomainBoundary(*ramp, warpx.Geom(lev), bcr);
    }

    m_mr_seam_ramp[lev] = std::move(ramp);
    m_mr_seam_ramp_ba[lev] = fba;
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
        if (m_mr_seam_band_active) { EnsureSeamBandRamp(lev); }

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
                    amrex::ParallelFor(gbx, ncomp,
                        [=] AMREX_GPU_DEVICE (int j, int k, int l, int n)
                        {
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
                amrex::ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int j, int k, int l)
                    {
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
    }

    const amrex::Real bmax = ::MaxAbsB(B);
    amrex::Real dxmin = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dxmin = std::min(dxmin, geom.CellSize(d)); }
    const amrex::Real scale = (bmax > 0._rt) ? dxmin / bmax : 1._rt;

    const int ngmin = ng.min();
    // Fine valid region: untouched by the fill, curl updates preserve div(B).
    const amrex::Real v = ::MaxAbsDivB(B, geom, 0, -1, nullptr, 0);
    // First ghost ring: cells mixing owned fine faces (patch boundary) with
    // prolonged faces; carries the coarse/fine face mismatch at the seam.
    const amrex::Real r1 = ::MaxAbsDivB(B, geom, 1, 0, nullptr, 0);
    // Outer ghost band: fully prolonged cells, div(B) inherits the coarse
    // value (machine zero if the coarse level is divergence free).
    const amrex::Real rb = (ngmin >= 2) ? ::MaxAbsDivB(B, geom, ngmin - 1, 1, nullptr, 0) : 0._rt;

    m_divb_fine_valid[lev].update(v, v * scale);
    m_divb_fine_ring[lev].update(r1, r1 * scale);
    m_divb_fine_band[lev].update(rb, rb * scale);

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
    }

    const amrex::Real bmax = ::MaxAbsB(B);
    amrex::Real dxmin = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dxmin = std::min(dxmin, geom.CellSize(d)); }
    const amrex::Real scale = (bmax > 0._rt) ? dxmin / bmax : 1._rt;

    const amrex::iMultiFab* mask = m_mr_keep_mask[clev].get();
    // Strict interior (all faces restricted): face averages of a
    // divergence-free fine field, expected machine zero.
    const amrex::Real vi = ::MaxAbsDivB(B, geom, 0, -1, mask, 2);
    // Seam ring (outermost keep-mask layer): cells mixing restricted and
    // freely evolved coarse faces show O(dt)-accumulated div(B) without EMF
    // matching (reported, not hidden -- EMF flux matching is deliberately a
    // later phase).
    const amrex::Real vs = ::MaxAbsDivB(B, geom, 0, -1, mask, 1);
    // Exterior (no restricted faces): freely evolved, divergence preserving.
    const amrex::Real ve = ::MaxAbsDivB(B, geom, 0, -1, mask, 0);

    m_divb_crse_interior[clev].update(vi, vi * scale);
    m_divb_crse_seam[clev].update(vs, vs * scale);
    m_divb_crse_exterior[clev].update(ve, ve * scale);

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
            << " rel " << m_divb_fine_band[lev].rel << "\n";
    }
    for (int clev = 0; clev < static_cast<int>(m_divb_crse_interior.size()); ++clev) {
        amrex::Print() << "  crse lev " << clev
            << ": interior raw " << m_divb_crse_interior[clev].raw
            << " rel " << m_divb_crse_interior[clev].rel
            << " | seam ring raw " << m_divb_crse_seam[clev].raw
            << " rel " << m_divb_crse_seam[clev].rel
            << " | exterior raw " << m_divb_crse_exterior[clev].raw
            << " rel " << m_divb_crse_exterior[clev].rel << "\n";
    }
    for (auto& s : m_divb_fine_valid)    { s = DivBStats{}; }
    for (auto& s : m_divb_fine_ring)     { s = DivBStats{}; }
    for (auto& s : m_divb_fine_band)     { s = DivBStats{}; }
    for (auto& s : m_divb_crse_interior) { s = DivBStats{}; }
    for (auto& s : m_divb_crse_seam)     { s = DivBStats{}; }
    for (auto& s : m_divb_crse_exterior) { s = DivBStats{}; }
}
