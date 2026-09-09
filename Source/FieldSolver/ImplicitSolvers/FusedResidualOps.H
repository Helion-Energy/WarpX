/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_FUSED_RESIDUAL_OPS_H_
#define WARPX_FUSED_RESIDUAL_OPS_H_

#include "Utils/TextMsg.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Periodicity.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

/** \brief One launch per stage for the bookkeeping of the theta-implicit MHD
 * residual and its block preconditioner (``implicit_evolve.fused_residual``).
 *
 * On a 1024 x 128 RZ mesh (131k cells) one GMRES iteration of the implicit
 * MHD solver is launch-latency bound: a residual evaluation issues about 170
 * kernels, of which about ten do the physics (face fluxes, Ohm's law, the
 * fluid divergences, the curls, the Green's-function boundary) and the rest
 * are one-MultiFab-at-a-time copies, zero fills, ghost-row clamps/mirrors,
 * staggered interpolations and about fifty ``FillBoundary`` /
 * ``FillBoundaryAndSync`` calls that, on a single box in a non-periodic
 * domain, move no data at all but each cost several profiler regions (and,
 * with WarpX's GPU default ``tiny_profiler.device_synchronize_around_region``,
 * two stream synchronizations). This header provides the pieces that turn
 * those sequences into a handful of launches WITHOUT changing a single
 * floating-point expression:
 *
 *  - Level() / Enabled(): the knob (default 0: every existing path untouched).
 *  - NeedsGhostExchange(): true when a FillBoundary of a FabArray can move
 *    data (more than one box, or a periodic direction); the FillBoundary*()
 *    wrappers skip the call otherwise. AMReX's FillBoundary_nowait already
 *    returns after finding an empty copy-tag list in that situation, and
 *    OverrideSync has no shared nodal points on one box, so the skip is exact.
 *  - Op / Run(): a table of independent per-box operations (copy, set,
 *    masked copy/set, subtract, axial ghost clamp+mirror, radial ghost
 *    mirror/clamp, frozen-row restore) executed as ONE ParallelFor over the
 *    union of their index ranges. The per-element expressions are those of
 *    the AMReX MultiFab routines and of the ThetaImplicitMHD ghost passes
 *    they replace (pure copies, sign flips and constants), so the results
 *    are bit-identical; the caller groups the ops into waves such that no
 *    op of a wave reads what another op of the same wave writes.
 *
 * Verified bit-identical on the CPU implicit MHD suite (Newton diagnostic
 * files byte-equal, plotfile fields equal to the last bit); see the
 * ``fused_residual_identity`` tests.
 */
namespace warpx::fused_residual
{
    /** \brief ``implicit_evolve.fused_residual`` (0 or 1), read once. */
    inline int Level ()
    {
        static const int level = [] () {
            int value = 0;
            const amrex::ParmParse pp("implicit_evolve");
            pp.query("fused_residual", value);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                value == 0 || value == 1,
                "implicit_evolve.fused_residual must be 0 or 1");
            return value;
        }();
        return level;
    }

    inline bool Enabled () { return Level() >= 1; }

    /** \brief True when a FillBoundary of this FabArray can move data. */
    inline bool NeedsGhostExchange (const amrex::FabArrayBase& a_fa,
                                    const amrex::Periodicity& a_period)
    {
        return a_fa.boxArray().size() > 1 || a_period.isAnyPeriodic();
    }

    /** \brief mf.FillBoundaryAndSync(period), skipped when a_fused and nothing can move. */
    inline void FillBoundaryAndSync (amrex::MultiFab& a_mf,
                                     const amrex::Periodicity& a_period,
                                     const bool a_fused)
    {
        if (a_fused && !NeedsGhostExchange(a_mf, a_period)) { return; }
        a_mf.FillBoundaryAndSync(a_period);
    }

    /** \brief mf.FillBoundary(period), skipped when a_fused and nothing can move. */
    inline void FillBoundary (amrex::MultiFab& a_mf,
                              const amrex::Periodicity& a_period,
                              const bool a_fused)
    {
        if (a_fused && !NeedsGhostExchange(a_mf, a_period)) { return; }
        a_mf.FillBoundary(a_period);
    }

    enum OpKind : int
    {
        Copy = 0,      //!< dst(., dst_comp+n) = src(., src_comp+n)
        SetVal,        //!< dst(., dst_comp+n) = value
        MaskedCopy,    //!< where mask != 0: dst = src
        MaskedSetVal,  //!< where mask != 0: dst = value
        Subtract,      //!< dst(., dst_comp+n) -= src(., src_comp+n)
        ZGhost,        //!< axial domain ghosts: zero-gradient clamp then z_lo mirror
        RGhost,        //!< radial domain ghosts: axis mirror / wall mirror or clamp
        RowsCopy       //!< where i >= rows_first[clamp(j)]: dst = src
    };

    /** \brief One operation of a fused wave (one FAB of one MultiFab). */
    struct Op
    {
        amrex::Array4<amrex::Real> dst{};
        amrex::Array4<amrex::Real const> src{};
        amrex::Array4<int const> mask{};
        const int* rows_first = nullptr;
        amrex::Box box{};          //!< index range covered (per component)
        int ncomp = 1;
        int dst_comp = 0;
        int src_comp = 0;
        int kind = Copy;
        amrex::Real value = 0.0;
        // ZGhost: the sequential composition of ThetaImplicitMHD::ApplyNeumannZDomainGhosts
        // (rows j < clamp_lo and j > clamp_hi copy the anchor row; nodal-in-z data mirrors
        // evenly about the anchor node) and ApplyMirrorZLoDomainGhosts (rows j < plane take
        // parity * the interior reflection; an odd nodal component is zeroed ON the plane),
        // the mirror having the last word exactly as when the two passes run in sequence.
        int clamp_lo = 0;
        int clamp_hi = 0;
        int plane = 0;
        int zdomain_hi = 0;
        bool nodal_z = false;
        bool neumann = false;
        bool mirror = false;
        amrex::GpuArray<int, 3> parity{1, 1, 1};
        // RGhost: i < r_lo mirrors about the axis (index 2 r_lo - cc_offset - i) with
        // parity_lo; i > r_hi takes the clamp row r_hi (hi_clamp) or the wall mirror
        // 2 r_hi + cc_offset - i, with parity_hi.
        int r_lo = 0;
        int r_hi = 0;
        int cc_offset = 1;
        bool do_lo = false;
        bool do_hi = false;
        bool hi_clamp = false;
        amrex::GpuArray<int, 3> parity_lo{1, 1, 1};
        amrex::GpuArray<int, 3> parity_hi{1, 1, 1};
        // RowsCopy: rows_first is indexed by clamp(j, rows_z_lo, rows_z_hi)
        int rows_z_lo = INT_MIN;
        int rows_z_hi = INT_MAX;
    };

    inline Op CopyOp (amrex::MultiFab& a_dst, const amrex::MultiFab& a_src,
                      const amrex::MFIter& a_mfi, const amrex::Box& a_box,
                      const int a_ncomp, const int a_dst_comp = 0, const int a_src_comp = 0)
    {
        Op op;
        op.kind = Copy;
        op.dst = a_dst.array(a_mfi);
        op.src = a_src.const_array(a_mfi);
        op.box = a_box;
        op.ncomp = a_ncomp;
        op.dst_comp = a_dst_comp;
        op.src_comp = a_src_comp;
        return op;
    }

    inline Op SubtractOp (amrex::MultiFab& a_dst, const amrex::MultiFab& a_src,
                          const amrex::MFIter& a_mfi, const amrex::Box& a_box,
                          const int a_ncomp, const int a_dst_comp = 0, const int a_src_comp = 0)
    {
        Op op = CopyOp(a_dst, a_src, a_mfi, a_box, a_ncomp, a_dst_comp, a_src_comp);
        op.kind = Subtract;
        return op;
    }

    inline Op SetValOp (amrex::MultiFab& a_dst, const amrex::MFIter& a_mfi,
                        const amrex::Box& a_box, const int a_ncomp,
                        const amrex::Real a_value, const int a_dst_comp = 0)
    {
        Op op;
        op.kind = SetVal;
        op.dst = a_dst.array(a_mfi);
        op.box = a_box;
        op.ncomp = a_ncomp;
        op.dst_comp = a_dst_comp;
        op.value = a_value;
        return op;
    }

    inline Op MaskedCopyOp (amrex::MultiFab& a_dst, const amrex::MultiFab& a_src,
                            const amrex::iMultiFab& a_mask, const amrex::MFIter& a_mfi,
                            const amrex::Box& a_box, const int a_ncomp)
    {
        Op op = CopyOp(a_dst, a_src, a_mfi, a_box, a_ncomp);
        op.kind = MaskedCopy;
        op.mask = a_mask.const_array(a_mfi);
        return op;
    }

    inline Op MaskedSetValOp (amrex::MultiFab& a_dst, const amrex::iMultiFab& a_mask,
                              const amrex::MFIter& a_mfi, const amrex::Box& a_box,
                              const int a_ncomp, const amrex::Real a_value)
    {
        Op op = SetValOp(a_dst, a_mfi, a_box, a_ncomp, a_value);
        op.kind = MaskedSetVal;
        op.mask = a_mask.const_array(a_mfi);
        return op;
    }

    inline Op RowsCopyOp (amrex::MultiFab& a_dst, const amrex::MultiFab& a_src,
                          const amrex::MFIter& a_mfi, const amrex::Box& a_box,
                          const int a_ncomp, const int* const a_rows_first,
                          const int a_rows_z_lo = INT_MIN, const int a_rows_z_hi = INT_MAX)
    {
        Op op = CopyOp(a_dst, a_src, a_mfi, a_box, a_ncomp);
        op.kind = RowsCopy;
        op.rows_first = a_rows_first;
        op.rows_z_lo = a_rows_z_lo;
        op.rows_z_hi = a_rows_z_hi;
        return op;
    }

    /** \brief Apply one op at flat index a_local of its own range. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void apply_op (const Op& op, const amrex::Long a_local) noexcept
    {
        const amrex::Dim3 lo = amrex::lbound(op.box);
        const amrex::Dim3 len = amrex::length(op.box);
        const amrex::Long npts = static_cast<amrex::Long>(len.x) * len.y * len.z;
        const int n = static_cast<int>(a_local / npts);
        amrex::Long r = a_local - static_cast<amrex::Long>(n) * npts;
        const int i = lo.x + static_cast<int>(r % len.x);
        r /= len.x;
        const int j = lo.y + static_cast<int>(r % len.y);
        const int k = lo.z + static_cast<int>(r / len.y);
        switch (op.kind)
        {
        case Copy:
            op.dst(i, j, k, op.dst_comp + n) = op.src(i, j, k, op.src_comp + n);
            break;
        case SetVal:
            op.dst(i, j, k, op.dst_comp + n) = op.value;
            break;
        case MaskedCopy:
            if (op.mask(i, j, k) != 0) {
                op.dst(i, j, k, op.dst_comp + n) = op.src(i, j, k, op.src_comp + n);
            }
            break;
        case MaskedSetVal:
            if (op.mask(i, j, k) != 0) {
                op.dst(i, j, k, op.dst_comp + n) = op.value;
            }
            break;
        case Subtract:
            op.dst(i, j, k, op.dst_comp + n) -= op.src(i, j, k, op.src_comp + n);
            break;
        case ZGhost:
            if (op.mirror && j < op.plane) {
                const int source =
                    op.nodal_z ? amrex::min(2 * op.plane - j, op.zdomain_hi)
                               : amrex::min(2 * op.plane - 1 - j, op.zdomain_hi);
                op.dst(i, j, k, n) =
                    static_cast<amrex::Real>(op.parity[n]) * op.dst(i, source, k, n);
            } else if (op.mirror && op.nodal_z && j == op.plane && op.parity[n] < 0) {
                op.dst(i, j, k, n) = 0.0;
            } else if (op.neumann && j < op.clamp_lo) {
                const int source =
                    op.nodal_z ? amrex::min(2 * op.clamp_lo - j, op.clamp_hi)
                               : op.clamp_lo;
                op.dst(i, j, k, n) = op.dst(i, source, k, n);
            } else if (op.neumann && j > op.clamp_hi) {
                const int source =
                    op.nodal_z ? amrex::max(2 * op.clamp_hi - j, op.clamp_lo)
                               : op.clamp_hi;
                op.dst(i, j, k, n) = op.dst(i, source, k, n);
            }
            break;
        case RGhost:
            if (op.do_lo && i < op.r_lo) {
                const int mirror = 2 * op.r_lo - op.cc_offset - i;
                const amrex::Real v = op.dst(mirror, j, k, n);
                op.dst(i, j, k, n) = (op.parity_lo[n] < 0) ? -v : v;
            } else if (op.do_hi && i > op.r_hi) {
                const int mirror =
                    op.hi_clamp ? op.r_hi : 2 * op.r_hi + op.cc_offset - i;
                const amrex::Real v = op.dst(mirror, j, k, n);
                op.dst(i, j, k, n) = (op.parity_hi[n] < 0) ? -v : v;
            }
            break;
        case RowsCopy:
        {
            const int jc = amrex::max(op.rows_z_lo, amrex::min(op.rows_z_hi, j));
            if (i >= op.rows_first[jc]) {
                op.dst(i, j, k, op.dst_comp + n) = op.src(i, j, k, op.src_comp + n);
            }
            break;
        }
        default:
            break;
        }
    }

    /** \brief Table of at most NMAX ops passed to the kernel by value. */
    template <int NMAX>
    struct OpTable
    {
        amrex::GpuArray<Op, NMAX> op;
        amrex::GpuArray<amrex::Long, NMAX + 1> offset;
        int n = 0;
    };

    /** \brief Run the ops as one launch per NMAX of them (one for the usual wave).
     *
     * The caller guarantees that no op reads an element another op of the
     * same call writes (ops of one MultiFab may only read its interior).
     */
    template <int NMAX = 8>
    inline void Run (const std::vector<Op>& a_ops)
    {
        for (std::size_t first = 0; first < a_ops.size(); first += NMAX)
        {
            OpTable<NMAX> table;
            table.n = static_cast<int>(std::min<std::size_t>(NMAX, a_ops.size() - first));
            amrex::Long total = 0;
            for (int s = 0; s < table.n; ++s)
            {
                table.op[s] = a_ops[first + s];
                table.offset[s] = total;
                total += table.op[s].box.numPts() * table.op[s].ncomp;
            }
            table.offset[table.n] = total;
            if (total == 0) { continue; }
            amrex::ParallelFor(total, [=] AMREX_GPU_DEVICE (amrex::Long idx) noexcept
            {
                int s = 0;
                while (s + 1 < table.n && idx >= table.offset[s + 1]) { ++s; }
                apply_op(table.op[s], idx - table.offset[s]);
            });
        }
    }
} // namespace warpx::fused_residual

#endif // WARPX_FUSED_RESIDUAL_OPS_H_
