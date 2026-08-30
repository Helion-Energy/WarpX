/* Copyright 2026 The WarpX Community
 *
 * This file is part of ABLASTR.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ABLASTR_FIELDS_STAGGER_COUPLING_H_
#define ABLASTR_FIELDS_STAGGER_COUPLING_H_

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParReduce.H>
#include <AMReX_REAL.H>

/** Coupling between a staggered (Yee) field registration and the
 *  all-nodal registration, as an adjoint gather/scatter pair.
 *
 *  Properties (each is load-bearing for the consumers):
 *
 *  - SECOND ORDER: the gather is the equal-weight average over the
 *    2^d staggered points surrounding a node (d = number of offset
 *    dimensions), i.e. linear interpolation evaluated at the
 *    midpoint -- second-order accurate on a uniform grid. It is
 *    ablastr::coarsen::sample::Interp at coarsening ratio 1.
 *
 *  - MOMENTUM CONSERVING (PIC sense): every field gathered to a given
 *    node uses the same interpolation scheme and weights for the same
 *    source registration, so a coupled advance evaluated at the nodes
 *    introduces no spurious force from interpolation asymmetry
 *    between field components or between E and B.
 *
 *  - ADJOINT PAIR: the scatter is the exact transpose of the gather
 *    in the volume-weighted inner products <u, v>_V = sum_i V_i u_i
 *    v_i of the two registrations (V from the geometry: proportional
 *    to the registration radius in cylindrical coordinates, uniform
 *    in Cartesian). For target point s with gathering nodes n:
 *        (S g)_s = sum_{n : s in stencil(n)} w V_n g_n / V_s ,
 *    w the gather weight (2^-d). Adjointness makes the round trip
 *    S G symmetric positive semi-definite -- the coupling interface
 *    is non-amplifying and conservative: <G u, g>_N = <u, S g>_S
 *    exactly FOR THE UNCLAMPED PAIR. On a uniform Cartesian grid S
 *    reduces to the same equal-weight average as G.
 *    Face-clamp caveat: at clamped non-periodic domain faces (see
 *    GatherToNodes) the gather degrades to a one-sided sample while
 *    the scatter is unchanged, so exact adjointness and the SPSD
 *    round trip hold on the interior only -- the clamped face rings
 *    trade adjointness for never reading unmaintained ghosts. The
 *    convex-combination bound (non-amplifying in max norm) survives
 *    everywhere.
 *    Cylindrical caveat, documented deliberately: at the axis the
 *    node volume vanishes, so the innermost staggered ring receives
 *    its value from the first off-axis node alone -- an O(dr) local
 *    error on ONE ring, the price of exact adjointness. The m = 0
 *    parity fields vanish linearly there.
 *
 *  A runtime self-check (adjointness to roundoff) is provided; the
 *  consumers run it once at coupling boot. Three passes: the
 *  Cartesian-weighted identity, the cylindrical-weighted identity,
 *  and constant preservation with the r-hi clamp active -- see the
 *  AdjointSelfCheck doc for the measured coverage limits (single
 *  box; the axis-degenerate scatter fallback and ghost-trap
 *  detection are NOT exercised; z-face clamps share the r-face
 *  code path).
 */
namespace ablastr::fields
{

    /** Radial coordinate weight of an index at registration stag
     *  along dim 0 (cylindrical); 1 in Cartesian builds. */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real
    stagger_coupling_rweight (
        int i, int stag_r, amrex::Real rmin_over_dr, bool cyl)
    {
        using namespace amrex::literals;
        if (!cyl) { return 1.0_rt; }
        // node (stag 1): r = rmin + i dr ; staggered (stag 0):
        // r = rmin + (i + 1/2) dr -- in units of dr
        const amrex::Real r = rmin_over_dr
            + (stag_r ? static_cast<amrex::Real>(i)
                      : static_cast<amrex::Real>(i) + 0.5_rt);
        return amrex::Math::abs(r);
    }

    /** Gather a staggered MultiFab to the all-nodal registration:
     *  dst(node) = equal-weight average of the surrounding source
     *  points (see the header doc). Source ghosts must be valid to
     *  depth 1 in each offset dimension ACROSS INTERIOR and PERIODIC
     *  boundaries; at NON-PERIODIC domain faces the stencil is
     *  clamped to the source-valid domain (the ghost ring there is
     *  never written by exchanges -- reading it is the classic
     *  domain-ghost trap), which degrades the face ring to a
     *  one-sided first-order sample. That ring is boundary-condition
     *  territory for the consumers. Pass the CELL-CENTERED domain box
     *  and per-dim/per-face clamp masks (clamp exactly the faces
     *  whose ghost rings no exchange maintains: e.g. the cylindrical
     *  axis guards ARE parity-filled and must stay in the stencil,
     *  while the radial wall and unmaintained z caps are clamped).
     *  A default-constructed (invalid) domain box disables clamping.
     *  Fills the valid nodal box.
     */
    inline void
    GatherToNodes (
        amrex::MultiFab & dst_nodal,
        amrex::MultiFab const & src,
        amrex::Box const & domain_cell = amrex::Box(),
        amrex::GpuArray<int,3> const & clamp_lo = {0, 0, 0},
        amrex::GpuArray<int,3> const & clamp_hi = {0, 0, 0},
        const int scomp = 0,
        const int dcomp = 0)
    {
        using namespace amrex;
        AMREX_ALWAYS_ASSERT(dst_nodal.ixType().nodeCentered());
        GpuArray<int,3> sf{1,1,1}, sc{1,1,1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            sf[d] = src.ixType().nodeCentered(d) ? 1 : 0;
        }
        GpuArray<int,3> const cr{1,1,1};
        // source-valid index range of the domain, per dim/face, for
        // the unmaintained-ghost clamp (unclamped faces get a huge
        // range)
        GpuArray<int,3> slo{-2000000000, -2000000000, -2000000000};
        GpuArray<int,3> shi{2000000000, 2000000000, 2000000000};
        const bool have_dom = domain_cell.ok();
        if (have_dom) {
            const Box sdom = amrex::convert(domain_cell,
                                            src.ixType());
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (clamp_lo[d]) { slo[d] = sdom.smallEnd(d); }
                if (clamp_hi[d]) { shi[d] = sdom.bigEnd(d); }
            }
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(dst_nodal, TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            auto const& d4 = dst_nodal.array(mfi);
            auto const& s4 = src.const_array(mfi);
            ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                using namespace amrex::literals;
                // stencil bounds per dim: node gathers src at
                // (idx-1, idx) along offset dims, (idx) along
                // matching dims -- clamped to the source domain
                int i0 = sf[0] ? i : i - 1, i1 = i;
                int j0 = sf[1] ? j : j - 1, j1 = j;
                int k0 = sf[2] ? k : k - 1, k1 = k;
                i0 = amrex::max(i0, slo[0]); i1 = amrex::min(i1, shi[0]);
                j0 = amrex::max(j0, slo[1]); j1 = amrex::min(j1, shi[1]);
                k0 = amrex::max(k0, slo[2]); k1 = amrex::min(k1, shi[2]);
                amrex::Real acc = 0.0_rt;
                int cnt = 0;
                for (int kk = k0; kk <= k1; ++kk) {
                for (int jj = j0; jj <= j1; ++jj) {
                for (int ii = i0; ii <= i1; ++ii) {
                    acc += s4(ii, jj, kk, scomp);
                    ++cnt;
                }}}
                d4(i,j,k,dcomp) = acc
                    / static_cast<amrex::Real>(cnt);
            });
        }
    }

    /** Scatter a nodal MultiFab to a staggered registration as the
     *  EXACT ADJOINT of GatherToNodes in the volume-weighted inner
     *  products (see the header doc). Nodal source must be valid on
     *  every node adjacent to the target box (nodal boxes contain
     *  their staggered counterparts, so the valid region suffices).
     *  Fills the valid staggered box.
     *
     *  \param cyl_rmin_over_dr rmin/dr of the level (cylindrical
     *         radial weight); ignored when cyl = false.
     */
    inline void
    ScatterFromNodes (
        amrex::MultiFab & dst,
        amrex::MultiFab const & src_nodal,
        const bool cyl,
        const amrex::Real cyl_rmin_over_dr,
        const int scomp = 0,
        const int dcomp = 0)
    {
        using namespace amrex;
        using namespace amrex::literals;
        AMREX_ALWAYS_ASSERT(src_nodal.ixType().nodeCentered());
        // offset pattern of the target registration
        GpuArray<int,3> st{1,1,1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            st[d] = dst.ixType().nodeCentered(d) ? 1 : 0;
        }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const& d4 = dst.array(mfi);
            auto const& s4 = src_nodal.const_array(mfi);
            ParallelFor(mfi.tilebox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // average the 2^d adjacent nodes, volume weighted;
                // the target weight normalizes (adjoint form)
                amrex::Real num = 0.0_rt, den = 0.0_rt;
                amrex::Real unw = 0.0_rt;
                int cnt = 0;
                const int ni = st[0] ? 0 : 1;
                const int nj = st[1] ? 0 : 1;
                const int nk = st[2] ? 0 : 1;
                for (int kk = 0; kk <= nk; ++kk) {
                for (int jj = 0; jj <= nj; ++jj) {
                for (int ii = 0; ii <= ni; ++ii) {
                    const amrex::Real vn = stagger_coupling_rweight(
                        i + ii, 1, cyl_rmin_over_dr, cyl);
                    const amrex::Real gv =
                        s4(i + ii, j + jj, k + kk, scomp);
                    num += vn * gv;
                    den += vn;
                    unw += gv;
                    ++cnt;
                }}}
                // a target whose whole stencil sits AT the axis has
                // zero total volume weight (components nodal in r at
                // i = 0): the volume-weighted form is 0/0 there, and
                // the correct value is the plain (unweighted) average
                // -- for the no-offset case, the axis node itself
                d4(i,j,k,dcomp) = (den > 0.0_rt)
                    ? num / den
                    : unw / static_cast<amrex::Real>(cnt);
            });
        }
    }

    /** Self-check of the coupling pair on an internal single-box
     *  test grid (staggered like `staggered_proto`). Three passes,
     *  returning the WORST relative mismatch; consumers assert
     *  < 1e-12 at boot:
     *   1. Cartesian-weighted adjointness <G u, g> = <u, S g> with
     *      the nodal test field windowed to the strict interior
     *      (boundary stencils of a finite box are a property of the
     *      box, not the pair);
     *   2. the same identity under the CYLINDRICAL radial weights
     *      (rmin = 0). COVERAGE LIMIT: for an r-offset prototype
     *      (Er) the scatter denominators are >= 1, so the
     *      axis-degenerate den == 0 fallback never executes here,
     *      and the r-weights annihilate the axis column's
     *      contribution -- the fallback branch (hit in production
     *      by r-nodal components) rests on inspection, not on this
     *      check;
     *   3. constant preservation of BOTH maps with the r-hi domain
     *      CLAMP active on the gather: G(1) = 1 on every valid node
     *      including the clamped face ring (a clamped average of a
     *      constant is the constant), and S(1) = 1 on every valid
     *      staggered point including the axis ring. COVERAGE LIMIT:
     *      the constant fills ghosts too, so this catches clamp
     *      weight/count errors but NOT a clamp that wrongly reads
     *      unmaintained ghosts; only the r-hi face is exercised
     *      (the z-face clamps share the identical code path).
     *  Passes 2 and 3 extend the plain Cartesian identity toward
     *  the production (RZ, wall-clamped) configuration within the
     *  stated limits; single-box by design (interface-node
     *  double-counting is a property of plain dot products). */
    inline amrex::Real
    AdjointSelfCheck (
        amrex::MultiFab const & staggered_proto)
    {
        using namespace amrex;
        using namespace amrex::literals;
        Box cbox(IntVect(0), IntVect(AMREX_D_DECL(15, 15, 15)));
        BoxArray ba_c(cbox);
        DistributionMapping dm_c(ba_c);
        MultiFab u(convert(ba_c, staggered_proto.ixType()), dm_c, 1, 1);
        MultiFab g(convert(ba_c, IntVect::TheNodeVector()), dm_c, 1, 1);
        MultiFab Gu(g.boxArray(), dm_c, 1, 0);
        MultiFab Sg(u.boxArray(), dm_c, 1, 0);
        auto hashfill = [] (MultiFab & mf, int salt, bool windowed) {
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                auto const& a = mf.array(mfi);
                // window the NODAL side to the strict interior so
                // every stencil the two inner products count stays on
                // valid data: the identity is a property of the
                // operator pair, and an unwindowed finite box counts
                // boundary ghosts asymmetrically (a property of the
                // box, not the pair)
                const Box vb = mfi.validbox();
                ParallelFor(mfi.growntilebox(1),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (windowed) {
                        const amrex::IntVect p(AMREX_D_DECL(i, j, k));
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            if (p[d] < vb.smallEnd(d) + 2 ||
                                p[d] > vb.bigEnd(d) - 2) {
                                a(i,j,k) = 0.0_rt;
                                return;
                            }
                        }
                    }
                    // deterministic position hash in (-1, 1)
                    unsigned h = static_cast<unsigned>(
                        (i * 73856093) ^ (j * 19349663)
                        ^ (k * 83492791) ^ (salt * 2654435761u));
                    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
                    a(i,j,k) = 2.0_rt * (static_cast<amrex::Real>(
                        h & 0xffffffu) / 16777215.0_rt) - 1.0_rt;
                });
            }
        };
        hashfill(u, 1, false);
        hashfill(g, 2, true);

        // registration r-weight applied in place (pass 2 weighting)
        auto rweight = [] (MultiFab & mf, bool node_reg) {
            const int st_r = mf.ixType().nodeCentered(0) ? 1 : 0;
            amrex::ignore_unused(node_reg);
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                auto const& a4 = mf.array(mfi);
                ParallelFor(mfi.tilebox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    a4(i,j,k) *= stagger_coupling_rweight(
                        i, st_r, 0.0_rt, true);
                });
            }
        };

        // pass 1: Cartesian-weighted adjointness
        GatherToNodes(Gu, u);
        ScatterFromNodes(Sg, g, /*cyl=*/false, 0.0_rt);
        amrex::Real ip_n = MultiFab::Dot(Gu, 0, g, 0, 1, 0);
        amrex::Real ip_s = MultiFab::Dot(u, 0, Sg, 0, 1, 0);
        amrex::Real scale = amrex::max(
            amrex::Math::abs(ip_n), amrex::Math::abs(ip_s), 1e-300_rt);
        amrex::Real worst = amrex::Math::abs(ip_n - ip_s) / scale;

        // pass 2: cylindrical-weighted adjointness (rmin = 0): weight
        // each dot by its side's registration radius
        ScatterFromNodes(Sg, g, /*cyl=*/true, 0.0_rt);
        MultiFab Gu_w(Gu.boxArray(), dm_c, 1, 0);
        MultiFab::Copy(Gu_w, Gu, 0, 0, 1, 0);
        rweight(Gu_w, true);
        MultiFab Sg_w(Sg.boxArray(), dm_c, 1, 0);
        MultiFab::Copy(Sg_w, Sg, 0, 0, 1, 0);
        rweight(Sg_w, false);
        ip_n = MultiFab::Dot(Gu_w, 0, g, 0, 1, 0);
        ip_s = MultiFab::Dot(u, 0, Sg_w, 0, 1, 0);
        scale = amrex::max(
            amrex::Math::abs(ip_n), amrex::Math::abs(ip_s), 1e-300_rt);
        worst = amrex::max(worst,
            amrex::Math::abs(ip_n - ip_s) / scale);

        // pass 3: constant preservation with the r-hi clamp active
        // (gather) and cylindrical weights (scatter, axis ring)
        u.setVal(1.0_rt);
        g.setVal(1.0_rt);
        GpuArray<int,3> clo{0,0,0}, chi{0,0,0};
        chi[0] = 1;
        GatherToNodes(Gu, u, cbox, clo, chi);
        ScatterFromNodes(Sg, g, /*cyl=*/true, 0.0_rt);
        Gu.plus(-1.0_rt, 0, 1, 0);
        Sg.plus(-1.0_rt, 0, 1, 0);
        // Device-safe valid-region max: the Array4 points at device
        // memory on GPU builds, so a host loop over it is a crash or
        // a garbage read -- reduce on the device instead.
        auto vmax = [] (MultiFab const & mf) {
            amrex::ReduceOps<amrex::ReduceOpMax> rop;
            amrex::ReduceData<amrex::Real> rdata(rop);
            using RT = typename decltype(rdata)::Type;
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                auto const& a4 = mf.const_array(mfi);
                rop.eval(mfi.validbox(), rdata,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> RT
                    {
                        return {amrex::Math::abs(a4(i,j,k))};
                    });
            }
            return amrex::get<0>(rdata.value(rop));
        };
        worst = amrex::max(worst, vmax(Gu));
        worst = amrex::max(worst, vmax(Sg));
        amrex::ParallelDescriptor::ReduceRealMax(worst);
        return worst;
    }

} // namespace ablastr::fields

#endif // ABLASTR_FIELDS_STAGGER_COUPLING_H_
