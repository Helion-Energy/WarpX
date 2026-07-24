/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "AzimuthalFilter.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <vector>

using namespace amrex;

#ifdef WARPX_DIM_RTZ

void AzimuthalFilter::Init (const Geometry& geom, Real alpha)
{
    m_geom = geom;
    m_alpha = alpha;
    m_NT = geom.Domain().length(1);
    m_enabled = true;

    // Report the band-limit table for the cell-centered ring radii.
    const Real dr = geom.CellSize(0);
    const Real rlo = geom.ProbLo(0);
    amrex::Print() << "AzimuthalFilter: m <= max(1, floor("
                   << alpha << "*pi*r/dr)), identity for m_max >= "
                   << m_NT/2 << ". Filtered (cell-centered) rings:\n";
    for (int i = 0; i < geom.Domain().length(0); ++i) {
        const Real r = rlo + (i + 0.5_rt)*dr;
        const int m_max = std::max(1,
            static_cast<int>(std::floor(alpha*MathConst::pi*r/dr)));
        if (m_max >= m_NT/2) { break; }
        amrex::Print() << "  ring i=" << i << " (r=" << r << "): m_max="
                       << m_max << "\n";
    }
}

AzimuthalFilter::ShellData& AzimuthalFilter::get_shell (const IntVect& iv)
{
    const int key = iv[0] + 2*iv[1] + 4*iv[2];
    if (m_shells[key]) { return *m_shells[key]; }

    m_shells[key] = std::make_unique<ShellData>();
    ShellData& sd = *m_shells[key];

    const Box dom = m_geom.Domain();
    const Box sdom = amrex::convert(dom, IndexType(iv));
    const Real dr = m_geom.CellSize(0);
    const Real rlo = m_geom.ProbLo(0);
    const int NT = m_NT;

    // Ring radius at the component's radial staggering: nodal rings sit at
    // r = rlo + i*dr, cell-centered rings at r = rlo + (i+1/2)*dr.
    const Real r_off = (iv[0] == 1) ? 0.0_rt : 0.5_rt;

    // Count the filtered rings and record each ring's m_max.
    std::vector<int> m_max_ring;
    for (int i = sdom.smallEnd(0); i <= sdom.bigEnd(0); ++i) {
        const Real r = rlo + (static_cast<Real>(i - dom.smallEnd(0)) + r_off)*dr;
        const int m_max = std::max(1,
            static_cast<int>(std::floor(m_alpha*MathConst::pi*r/dr)));
        if (m_max >= NT/2) { break; }
        m_max_ring.push_back(m_max);
    }

    sd.i_lo = sdom.smallEnd(0);
    sd.n_ring = static_cast<int>(m_max_ring.size());
    sd.j_lo = sdom.smallEnd(1);
    sd.theta_nodal = (iv[1] == 1);
    sd.active = (sd.n_ring > 0);
    if (!sd.active) { return sd; }

    // Circulant Fourier-projection matrix per ring:
    //   P[j][l] = (1 + 2*sum_{m=1}^{m_max} cos(m*dtheta*(j-l))) / NT
    // (the Dirichlet kernel; exact projection onto modes m <= m_max on a ring
    // of NT uniform samples; row sums = 1 so the ring average is preserved).
    const Real dtheta = 2.0_rt*MathConst::pi/static_cast<Real>(NT);
    std::vector<Real> P_host(static_cast<std::size_t>(sd.n_ring)*NT*NT);
    for (int ring = 0; ring < sd.n_ring; ++ring) {
        const int m_max = m_max_ring[ring];
        for (int j = 0; j < NT; ++j) {
            for (int l = 0; l < NT; ++l) {
                Real acc = 1.0_rt;
                for (int m = 1; m <= m_max; ++m) {
                    acc += 2.0_rt*std::cos(m*dtheta*static_cast<Real>(j - l));
                }
                P_host[(static_cast<std::size_t>(ring)*NT + j)*NT + l] =
                    acc/static_cast<Real>(NT);
            }
        }
    }
    sd.P.resize(P_host.size());
    Gpu::copyAsync(Gpu::hostToDevice, P_host.begin(), P_host.end(), sd.P.begin());
    Gpu::streamSynchronize();

    // Full-theta, full-z shell over the filtered rings, chunked in z across
    // ranks. Keeping theta whole in every box is what makes the per-ring
    // mat-vec local.
    IntVect slo = sdom.smallEnd();
    IntVect shi = sdom.bigEnd();
    shi[0] = sd.i_lo + sd.n_ring - 1;
    Box shell_box(slo, shi, IndexType(iv));
    BoxArray ba(shell_box);
    const int nprocs = ParallelDescriptor::NProcs();
    const int zlen = shell_box.length(2);
    const int zchunk = std::max(8, (zlen + nprocs - 1)/nprocs);
    ba.maxSize(IntVect(sd.n_ring, shell_box.length(1), zchunk));
    const DistributionMapping dm{ba};

    sd.in = MultiFab(ba, dm, 1, 0);
    sd.out = MultiFab(ba, dm, 1, 0);
    return sd;
}

void AzimuthalFilter::ApplyFilter (MultiFab& mf, int scomp, int ncomp,
                                   bool fill_ghosts)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_enabled,
        "AzimuthalFilter::ApplyFilter called before Init");

    ShellData& sd = get_shell(mf.ixType().toIntVect());
    if (!sd.active) { return; }

    const int NT = m_NT;
    const int i_lo = sd.i_lo;
    const int j_lo = sd.j_lo;
    const bool theta_nodal = sd.theta_nodal;
    const Real* pmat = sd.P.data();
    const IntVect ng_dst = fill_ghosts ? mf.nGrowVect() : IntVect(0);

    for (int comp = scomp; comp < scomp + ncomp; ++comp) {
        sd.in.ParallelCopy(mf, comp, 0, 1);

        for (MFIter mfi(sd.in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box tb = mfi.tilebox();
            Array4<Real const> const& ain = sd.in.const_array(mfi);
            Array4<Real> const& aout = sd.out.array(mfi);
            amrex::ParallelFor(tb,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // theta-nodal grids carry the periodic image plane at
                    // j = j_lo + NT; it is set from the j_lo plane below.
                    if (j - j_lo >= NT) { return; }
                    const int ring = i - i_lo;
                    const Real* prow =
                        pmat + (static_cast<std::size_t>(ring)*NT + (j - j_lo))*NT;
                    Real acc = 0.0_rt;
                    for (int l = 0; l < NT; ++l) {
                        acc += prow[l]*ain(i, j_lo + l, k);
                    }
                    aout(i, j, k) = acc;
                });
            if (theta_nodal && tb.bigEnd(1) >= j_lo + NT) {
                Box image = tb;
                image.setSmall(1, j_lo + NT);
                amrex::ParallelFor(image,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        amrex::ignore_unused(j);
                        aout(i, j_lo + NT, k) = aout(i, j_lo, k);
                    });
            }
        }

        // The scatter fills valid cells and any ghost cells reachable from
        // the shell's valid data through the periodic images. For a periodic
        // NODAL direction the duplicated domain-boundary planes are also
        // periodic images of each other; WarpX keeps them synced, so the
        // (order-unspecified) overlapping writes carry identical values.
        mf.ParallelCopy(sd.out, 0, comp, 1, IntVect(0), ng_dst,
                        m_geom.periodicity());
    }
}

#else // not WARPX_DIM_RTZ

void AzimuthalFilter::Init (const Geometry& /*geom*/, Real /*alpha*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "AzimuthalFilter is only available for RTZ geometry");
}

AzimuthalFilter::ShellData& AzimuthalFilter::get_shell (const IntVect& /*iv*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "AzimuthalFilter is only available for RTZ geometry");
    if (!m_shells[0]) { m_shells[0] = std::make_unique<ShellData>(); }
    return *m_shells[0];
}

void AzimuthalFilter::ApplyFilter (MultiFab& /*mf*/, int /*scomp*/,
                                   int /*ncomp*/, bool /*fill_ghosts*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "AzimuthalFilter is only available for RTZ geometry");
}

#endif // WARPX_DIM_RTZ
