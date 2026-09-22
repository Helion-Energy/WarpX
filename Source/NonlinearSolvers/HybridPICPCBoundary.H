/* Copyright 2026 Prabhat Kumar
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Implementation detail included by HybridPICPC.H. See its provenance.
 */
#ifndef WARPX_HYBRID_PIC_PC_BOUNDARY_H_
#define WARPX_HYBRID_PIC_PC_BOUNDARY_H_

template <class T, class Ops>
void HybridPICPC<T,Ops>::ApplyDomainClosure ( MF& a, int a_lev,
                                              ClosureClass a_cls,
                                              int a_scomp, int a_ncomp ) const
{
    using namespace amrex;

    // component range: the closure logic indexes components RELATIVE to
    // a_scomp (so a 6-comp Krylov block gets its W-half and E-half closed
    // with their own classes by two calls), the storage by a_scomp + n
    const int ncl = (a_ncomp < 0) ? a.nComp() : a_ncomp;
    const int sc = a_scomp;

    if (m_has_pec_wall) {
        const Box dom = amrex::convert(m_geom[a_lev].Domain(),
                                       a.boxArray().ixType());
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            for (int side = 0; side < 2; side++) {
                if (!m_wall_pec[d][side]) { continue; }
                // wall NODE index: the staggered containers are
                // cell-indexed, so the hi-side wall plane sits one past
                // the cell bigEnd
                const int w = (side == 0)
                    ? dom.smallEnd(d)
                    : dom.bigEnd(d) + (m_staggered ? 1 : 0);
                const int nc = m_dim2comp[d];
                const bool lo_side = (side == 0);
                const int cls = static_cast<int>(a_cls);
                const int dd = d;
                const bool stag = m_staggered;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box& bx = mfi.growntilebox();
                    // component window: relative index n, storage sc + n
                    Array4<Real> const& v = a.array(mfi, sc);
                    amrex::ParallelFor(bx, ncl,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                        {
                            const int id = (dd == 0) ? i : ((dd == 1) ? j : k);
                            const bool outside =
                                lo_side ? (id < w) : (id > w);
                            // Coef: comps 0-2 carry bhat (vector), the rest
                            // are scalars
                            const bool scalar =
                                (cls == 4/*Coef*/) && (n >= 3);
                            const bool is_norm = !scalar && (n == nc);
                            if (stag) {
                                // Yee: components NODAL along dd (wall-
                                // tangential E/J, wall-normal B) carry a
                                // wall-plane DOF that is zero at PEC (dead
                                // tangential-E columns, zero normal B, and
                                // zero mirror-image tangential J) and
                                // mirror ODD about the wall node;
                                // components CELL-centered along dd have no
                                // wall DOF and mirror EVEN about the wall
                                // plane. The node-built coefficients keep a
                                // smooth even cell-mirror (their ghost ring
                                // only shapes U there).
                                bool nodal_d;
                                Real sgn;
                                if (cls == 0/*EField*/ || cls == 2/*JField*/) {
                                    nodal_d = !is_norm;
                                    sgn = is_norm ? 1.0_rt : -1.0_rt;
                                } else if (cls == 4/*Coef*/) {
                                    nodal_d = false;
                                    sgn = 1.0_rt;
                                } else { // BField, BCoef
                                    nodal_d = is_norm;
                                    sgn = is_norm ? -1.0_rt : 1.0_rt;
                                }
                                int im;
                                if (nodal_d) {
                                    const bool beyond =
                                        lo_side ? (id < w) : (id > w);
                                    if (!beyond) {
                                        if (id == w) { v(i,j,k,n) = 0.0_rt; }
                                        return;
                                    }
                                    im = 2*w - id;
                                } else {
                                    const bool beyond =
                                        lo_side ? (id < w) : (id >= w);
                                    if (!beyond) { return; }
                                    im = 2*w - 1 - id;
                                }
                                const int mi = (dd == 0) ? im : i;
                                const int mj = (dd == 1) ? im : j;
                                const int mk = (dd == 2) ? im : k;
                                v(i,j,k,n) = sgn * v(mi,mj,mk,n);
                                return;
                            }
                            if (!outside) {
                                if (id != w) { return; }
                                // wall plane: EField zeroes tangential
                                // (identity rows / dead columns), BField
                                // zeroes the normal component
                                if (cls == 0/*EField*/ && !scalar
                                    && !is_norm) { v(i,j,k,n) = 0.0_rt; }
                                if (cls == 1/*BField*/ && is_norm) {
                                    v(i,j,k,n) = 0.0_rt;
                                }
                                return;
                            }
                            // ghost planes: signed mirror about the wall node
                            const int im = 2*w - id;
                            Real sign = 1.0_rt;
                            if (!scalar) {
                                if (cls == 0/*EField*/ || cls == 2/*JField*/) {
                                    sign = is_norm ? 1.0_rt : -1.0_rt;
                                } else { // BField, BCoef, Coef bhat comps
                                    sign = is_norm ? -1.0_rt : 1.0_rt;
                                }
                            }
                            const int mi = (dd == 0) ? im : i;
                            const int mj = (dd == 1) ? im : j;
                            const int mk = (dd == 2) ? im : k;
                            v(i,j,k,n) = sign * v(mi,mj,mk,n);
                        });
                }
            }
        }
    }

#if defined(WARPX_DIM_RZ)
    // Symmetry axis (r = 0): signed-mirror guards about the axis node --
    // r/theta components odd, z even, the same parities for E-, B- and
    // J-like objects at m = 0 (matching ApplyFieldBoundaryOnAxis). The Ohm
    // solve zeroes E_theta ON the axis, so EField-class objects get that
    // row zeroed here as well (restored as an identity row on the output).
    if (m_has_axis) {
        const Box adom = amrex::convert(m_geom[a_lev].Domain(),
                                        a.boxArray().ixType());
        const int w = adom.smallEnd(0);
        const int cls = static_cast<int>(a_cls);
        const bool stag = m_staggered;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.growntilebox();
            Array4<Real> const& v = a.array(mfi, sc);
            amrex::ParallelFor(bx, ncl,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (i > w) { return; }
                    const bool scalar = (cls == 4/*Coef*/) && (n >= 3);
                    // Yee: components CELL-centered in r (the E/J r
                    // component, the B theta/z components; never the
                    // node-built coefficients) have no axis DOF and mirror
                    // with the residual's ishift = 1
                    bool cell_r = false;
                    if (stag && cls != 4/*Coef*/) {
                        cell_r = (cls == 0 || cls == 2) ? (n == 0)
                                                        : (n != 0);
                    }
                    if (i == w) {
                        if (cls == 0/*EField*/ && n == 1) {
                            v(i,j,k,n) = 0.0_rt;
                        }
                        return;
                    }
                    const int im = cell_r ? (2*w - 1 - i) : (2*w - i);
                    const Real sign =
                        (scalar || n == 2) ? 1.0_rt : -1.0_rt;
                    v(i,j,k,n) = sign * v(im,j,k,n);
                });
        }
    }
#endif

    // EB (staircase): zero covered/frozen nodes so no stencil emits or
    // receives contributions there (the residual's masked Ohm solves skip
    // those locations). Fine level only; masked up to the flag's guards.
    // Exception: the pair's local W-hat scale (Coef comp 14) is a strictly
    // positive SCALING, divided by in the joint block -- zeroing it at
    // covered nodes makes 1/sW = inf there (NaN poisoning that the
    // level-0-only closure cannot clean off the coarse levels).
    if (a_lev == 0 && m_eb_flag != nullptr) {
        amrex::IntVect ngm = a.nGrowVect();
        ngm.min(m_eb_flag->nGrowVect());
        const bool is_coef = (a_cls == ClosureClass::Coef);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.growntilebox(ngm);
            Array4<Real> const& v = a.array(mfi, sc);
            Array4<const int> const& fl = m_eb_flag->const_array(mfi);
            amrex::ParallelFor(bx, ncl,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    if (is_coef && n == 14) { return; }
                    if (fl(i,j,k) == 0) { v(i,j,k,n) = 0.0_rt; }
                });
        }
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::OverrideIdentityRows ( T& a_out, const T& a_in ) const
{
    using namespace amrex;

    for (int d = 0; d < AMREX_SPACEDIM; d++) {
        for (int side = 0; side < 2; side++) {
            // PEC faces: exact identity tangential rows (both grids).
            // Staggered hi-side zero-extension faces: the nodal wall
            // plane is an unmodeled ghost plane of the cell-indexed
            // container -- the preconditioner would return zero there
            // (a singular preconditioner); make those rows identity.
            const bool open_stag = m_staggered && (side == 1)
                && m_wall_open[d][side];
            if (!m_wall_pec[d][side] && !open_stag) { continue; }
            const int nc = m_dim2comp[d];
            for (int n = 0; n < 3; n++) {
                if (n == nc) { continue; }  // tangential rows only
                MF& o = *a_out.getArrayVec()[0][n];
                const MF& s = *a_in.getArrayVec()[0][n];
                // wall plane in this component's own staggering (the
                // tangential components are nodal along d on both grids)
                const Box dom = amrex::convert(m_geom[0].Domain(),
                                               o.boxArray().ixType());
                const int w = (side == 0) ? dom.smallEnd(d) : dom.bigEnd(d);
                Box plane = dom;
                plane.setSmall(d, w);
                plane.setBig(d, w);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(o, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box bx = mfi.tilebox() & plane;
                    if (bx.isEmpty()) { continue; }
                    Array4<Real> const& ov = o.array(mfi);
                    Array4<const Real> const& sv = s.const_array(mfi);
                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        { ov(i,j,k) = sv(i,j,k); });
                }
            }
        }
    }

#if defined(WARPX_DIM_RZ)
    // E_theta on the axis is fixed to zero by the Ohm solve: identity row
    if (m_has_axis) {
        const Box adom = amrex::convert(m_geom[0].Domain(),
            a_out.getArrayVec()[0][1]->boxArray().ixType());
        Box plane = adom;
        plane.setBig(0, adom.smallEnd(0));
        MF& o = *a_out.getArrayVec()[0][1];
        const MF& si = *a_in.getArrayVec()[0][1];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(o, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box bx = mfi.tilebox() & plane;
            if (bx.isEmpty()) { continue; }
            Array4<Real> const& ov = o.array(mfi);
            Array4<const Real> const& sv = si.const_array(mfi);
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                { ov(i,j,k) = sv(i,j,k); });
        }
    }
#endif

    if (m_eb_flag != nullptr) {
        for (int n = 0; n < 3; n++) {
            MF& o = *a_out.getArrayVec()[0][n];
            const MF& s = *a_in.getArrayVec()[0][n];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(o, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();
                Array4<Real> const& ov = o.array(mfi);
                Array4<const Real> const& sv = s.const_array(mfi);
                Array4<const int> const& fl = m_eb_flag->const_array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (fl(i,j,k) == 0) { ov(i,j,k) = sv(i,j,k); }
                    });
            }
        }
    }
}

#endif
