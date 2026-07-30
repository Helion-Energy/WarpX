/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GreensFunctionOpenBC.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

using namespace amrex;

#if defined(WARPX_DIM_RZ)

namespace
{
    /** Complete elliptic integrals K(k) and E(k) as functions of m = k^2,
     * via the arithmetic-geometric mean (Abramowitz & Stegun 17.6).
     * Host-only: used at kernel-assembly (setup) time.
     * Requires 0 <= m < 1 (evaluation points are separated from sources). */
    void CompleteEllipticIntegrals (amrex::Real m, amrex::Real& K, amrex::Real& E)
    {
        amrex::Real a = 1.0_rt;
        amrex::Real b = std::sqrt(1.0_rt - m);
        amrex::Real c = std::sqrt(m);
        amrex::Real sum = 0.5_rt * m;   // n = 0 term: 2^{-1} c_0^2 with c_0 = k
        amrex::Real pow2 = 1.0_rt;
        for (int n = 0; n < 64; ++n) {
            if (std::abs(c) < 1.0e-16_rt) { break; }
            const amrex::Real an = 0.5_rt * (a + b);
            c = 0.5_rt * (a - b);
            b = std::sqrt(a * b);
            a = an;
            pow2 *= 2.0_rt;
            sum += 0.5_rt * pow2 * c * c;   // 2^{n-1} c_n^2
        }
        K = MathConst::pi / (2.0_rt * a);
        E = K * (1.0_rt - sum);
    }

    /** Poloidal flux per unit ring current: psi(rb, zb) of a circular
     * filament at (rs, zs) carrying 1 A,
     *     G0 = (mu0/4pi) sqrt((zb-zs)^2 + (rb+rs)^2) [(2-k^2)K - 2E].
     * Host-only (setup). */
    amrex::Real RingPsiPerAmp (amrex::Real rb, amrex::Real zb,
                               amrex::Real rs, amrex::Real zs)
    {
        const amrex::Real d2 = (zb - zs) * (zb - zs) + (rb + rs) * (rb + rs);
        if (d2 <= 0.0_rt || rs <= 0.0_rt || rb <= 0.0_rt) { return 0.0_rt; }
        const amrex::Real m = 4.0_rt * rb * rs / d2;
        amrex::Real K, E;
        CompleteEllipticIntegrals(m, K, E);
        return PhysConst::mu0 / (4.0_rt * MathConst::pi)
               * std::sqrt(d2) * ((2.0_rt - m) * K - 2.0_rt * E);
    }
}

bool GreensFunctionOpenBC::IsActive ()
{
    return (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)
        && (WarpX::field_boundary_hi[0] == FieldBoundaryType::Open);
}

void GreensFunctionOpenBC::Define (ablastr::fields::VectorField const& Bfield,
                                   amrex::Geometry const& geom)
{
    const ParmParse pp_boundary("boundary");
    pp_boundary.query("open_bc_coarsening", m_coarsening);
    pp_boundary.query("open_bc_image_sum_rtol", m_image_sum_rtol);
    pp_boundary.query("open_bc_max_images", m_max_images);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_coarsening >= 1,
        "boundary.open_bc_coarsening must be >= 1");

    // Validate the supported configuration
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::n_rz_azimuthal_modes == 1,
        "The Green's-function open boundary only supports the m = 0 azimuthal mode.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geom.ProbLo(0) == 0.0_rt,
        "The Green's-function open boundary requires the RZ domain to include "
        "the axis (geometry.prob_lo[0] == 0).");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::field_boundary_lo[1] != FieldBoundaryType::Open &&
        WarpX::field_boundary_hi[1] != FieldBoundaryType::Open &&
        WarpX::field_boundary_lo[0] != FieldBoundaryType::Open,
        "The Green's-function open boundary is only implemented on the r_hi face.");

    const Box& domain = geom.Domain();
    m_nr = domain.length(0);
    m_nz = domain.length(1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        domain.smallEnd(0) == 0 && domain.smallEnd(1) == 0,
        "GreensFunctionOpenBC: unexpected domain index origin.");

    const IntVect ng = Bfield[0]->nGrowVect();
    for (int d = 0; d < 3; ++d) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            Bfield[d]->nGrowVect() == ng,
            "GreensFunctionOpenBC: B components must have matching ghost cells.");
    }
    m_ngr = ng[0];
    m_ngz = ng[1];

    m_dr = geom.CellSize(0);
    m_dz = geom.CellSize(1);
    m_zmin = geom.ProbLo(1);
    m_periodic_z = geom.isPeriodic(1);

    // psi table: nodal (i, j) with i in [m_nr, m_nr + m_ngr],
    // j in [-m_ngz, m_nz + m_ngz]
    m_psi_ni = m_ngr + 1;
    m_psi_nj = m_nz + 2 * m_ngz + 1;
    const int nrows = m_psi_ni * m_psi_nj;

    // Source bins over nodal source indices i in [0, m_nr-1], j in [0, m_nz-1].
    // (The r = 0 axis nodes carry no ring current; the face node i = m_nr and,
    // for non-periodic z, the top node j = m_nz are excluded from the source
    // support -- interior sources are assumed to stay away from the faces.)
    const int nbin_r = (m_nr - 1) / m_coarsening + 1;
    m_nbin_z = (m_nz - 1) / m_coarsening + 1;
    m_nbins = nbin_r * m_nbin_z;
    m_ncols = 3 * m_nbins;

    // Bin centers (geometric center of the bin's node range; the dipole
    // kernel entries account for in-bin offsets exactly to first order).
    Vector<amrex::Real> bin_r_h(m_nbins), bin_z_h(m_nbins);
    for (int bi = 0; bi < nbin_r; ++bi) {
        const int ilo = bi * m_coarsening;
        const int ihi = std::min((bi + 1) * m_coarsening - 1, m_nr - 1);
        for (int bj = 0; bj < m_nbin_z; ++bj) {
            const int jlo = bj * m_coarsening;
            const int jhi = std::min((bj + 1) * m_coarsening - 1, m_nz - 1);
            const int b = bi * m_nbin_z + bj;
            bin_r_h[b] = 0.5_rt * (ilo + ihi) * m_dr;
            bin_z_h[b] = m_zmin + 0.5_rt * (jlo + jhi) * m_dz;
        }
    }

    // Assemble the kernel on the host (elliptic integrals; setup only),
    // then copy to device. For periodic z, image rings at z' + n L_z are
    // summed in +/- pairs until the pair increment drops below
    // m_image_sum_rtol relative to the accumulated value.
    Vector<amrex::Real> kernel_h(static_cast<Long>(nrows) * m_ncols);
    const amrex::Real Lz = geom.ProbLength(1);
    const amrex::Real rtol = m_image_sum_rtol;
    const int max_images = m_max_images;
    const bool periodic_z = m_periodic_z;
    const int psi_nj = m_psi_nj;
    const int ngz = m_ngz;
    const int nr = m_nr;
    const amrex::Real dr = m_dr;
    const amrex::Real dz = m_dz;
    const amrex::Real zmin = m_zmin;

#ifdef AMREX_USE_OMP
#pragma omp parallel for
#endif
    for (int row = 0; row < nrows; ++row) {
        const int ip = row / psi_nj;             // 0 .. m_ngr
        const int jp = row % psi_nj;             // 0 .. m_psi_nj-1
        const amrex::Real rb = (nr + ip) * dr;
        const amrex::Real zb = zmin + (jp - ngz) * dz;
        for (int b = 0; b < m_nbins; ++b) {
            const amrex::Real rs = bin_r_h[b];
            const amrex::Real zs = bin_z_h[b];
            // finite-difference step for the dipole entries, scaled to the
            // n = 0 source-observer distance
            const amrex::Real d0 = std::sqrt((zb - zs) * (zb - zs)
                                             + (rb + rs) * (rb + rs));
            const amrex::Real h = 1.0e-5_rt * d0;
            amrex::Real g = 0.0_rt, gr = 0.0_rt, gz = 0.0_rt;
            auto add_image = [&] (amrex::Real zoff) -> amrex::Real {
                const amrex::Real zsi = zs + zoff;
                const amrex::Real g0 = RingPsiPerAmp(rb, zb, rs, zsi);
                g  += g0;
                gr += (RingPsiPerAmp(rb, zb, rs + h, zsi)
                       - RingPsiPerAmp(rb, zb, rs - h, zsi)) / (2.0_rt * h);
                gz += (RingPsiPerAmp(rb, zb, rs, zsi + h)
                       - RingPsiPerAmp(rb, zb, rs, zsi - h)) / (2.0_rt * h);
                return g0;
            };
            add_image(0.0_rt);
            if (periodic_z) {
                for (int n = 1; n <= max_images; ++n) {
                    amrex::Real dpair = add_image(n * Lz);
                    dpair += add_image(-n * Lz);
                    if (n >= 2 && std::abs(dpair) < rtol * std::abs(g)) { break; }
                }
            }
            const Long idx = static_cast<Long>(row) * m_ncols + 3 * b;
            kernel_h[idx + 0] = g;
            kernel_h[idx + 1] = gr;
            kernel_h[idx + 2] = gz;
        }
    }

    m_kernel.resize(kernel_h.size());
    m_bin_r.resize(m_nbins);
    m_bin_z.resize(m_nbins);
    m_src.resize(m_ncols);
    m_src_host.resize(m_ncols);
    m_psi.resize(nrows);
    Gpu::copyAsync(Gpu::hostToDevice, kernel_h.begin(), kernel_h.end(), m_kernel.begin());
    Gpu::copyAsync(Gpu::hostToDevice, bin_r_h.begin(), bin_r_h.end(), m_bin_r.begin());
    Gpu::copyAsync(Gpu::hostToDevice, bin_z_h.begin(), bin_z_h.end(), m_bin_z.begin());
    Gpu::streamSynchronize();

    amrex::Print() << "GreensFunctionOpenBC: open (free-space) boundary active on r_hi;"
                   << " kernel " << nrows << " x " << m_ncols
                   << " (coarsening " << m_coarsening << ", "
                   << (m_periodic_z ? "periodic" : "isolated") << " z)\n";

    m_defined = true;
}

void GreensFunctionOpenBC::ApplyToBfield (ablastr::fields::VectorField const& Bfield,
                                          amrex::Geometry const& geom, int lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0,
        "The Green's-function open boundary only supports a single level.");

    if (!m_defined) { Define(Bfield, geom); }

    // The interior box-boundary and periodic ghost values of B must be
    // current before differencing across box seams below (the caller
    // applies this BC right after the valid-cell Faraday update, before
    // its own FillBoundary).
    for (int d = 0; d < 3; ++d) {
        Bfield[d]->FillBoundary(geom.periodicity());
    }

    // ---- 1. Deposit the coarse source moments (I, I dr, I dz) per bin ----
    amrex::Real* const AMREX_RESTRICT src = m_src.data();
    const int ncols = m_ncols;
    amrex::ParallelFor(ncols, [=] AMREX_GPU_DEVICE (int c) { src[c] = 0.0_rt; });

    const amrex::Real dr = m_dr;
    const amrex::Real dz = m_dz;
    const amrex::Real zmin = m_zmin;
    const amrex::Real inv_dr = 1.0_rt / dr;
    const amrex::Real inv_dz = 1.0_rt / dz;
    const amrex::Real dA_over_mu0 = dr * dz / PhysConst::mu0;
    const int nr = m_nr;
    const int coarsening = m_coarsening;
    const int nbin_z = m_nbin_z;
    const amrex::Real* const AMREX_RESTRICT bin_r = m_bin_r.data();
    const amrex::Real* const AMREX_RESTRICT bin_z = m_bin_z.data();

    for (MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {
        const Box vbx_cc = amrex::enclosedCells(mfi.validbox());
        Array4<amrex::Real const> const& Br = Bfield[0]->const_array(mfi);
        Array4<amrex::Real const> const& Bz = Bfield[2]->const_array(mfi);
        // Each source node (i, j) is owned by the rank whose valid
        // cell-centered box contains cell (i, j); this counts every node
        // exactly once and drops the r_hi face node (i = m_nr) and, for
        // non-periodic z, the top node (j = m_nz).
        amrex::ParallelFor(vbx_cc,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                if (i < 1 || i > nr - 1) { return; }  // axis and face nodes
                // total azimuthal current at the psi node:
                // J_theta dA = (curl B)_theta dA / mu0, with the same
                // discrete curl as the Ohm's-law solver
                const amrex::Real i_ring =
                    ((Br(i, j, 0) - Br(i, j - 1, 0)) * inv_dz
                     - (Bz(i, j, 0) - Bz(i - 1, j, 0)) * inv_dr) * dA_over_mu0;
                const int b = (i / coarsening) * nbin_z + (j / coarsening);
                const amrex::Real rn = i * dr;
                const amrex::Real zn = zmin + j * dz;
                HostDevice::Atomic::Add(&src[3 * b + 0], i_ring);
                HostDevice::Atomic::Add(&src[3 * b + 1], i_ring * (rn - bin_r[b]));
                HostDevice::Atomic::Add(&src[3 * b + 2], i_ring * (zn - bin_z[b]));
            });
    }

    // ---- 2. Global reduction of the moments ----
    Gpu::copyAsync(Gpu::deviceToHost, m_src.begin(), m_src.end(), m_src_host.begin());
    Gpu::streamSynchronize();
    ParallelDescriptor::ReduceRealSum(m_src_host.data(), m_ncols);
    Gpu::copyAsync(Gpu::hostToDevice, m_src_host.begin(), m_src_host.end(), m_src.begin());
    Gpu::streamSynchronize();

    // ---- 3. GEMV: psi at the ghost psi-points ----
    const int nrows = m_psi_ni * m_psi_nj;
    const amrex::Real* const AMREX_RESTRICT kernel = m_kernel.data();
    amrex::Real* const AMREX_RESTRICT psi = m_psi.data();
    amrex::ParallelFor(nrows, [=] AMREX_GPU_DEVICE (int row) {
        amrex::Real s = 0.0_rt;
        const amrex::Real* const AMREX_RESTRICT krow =
            kernel + static_cast<Long>(row) * ncols;
        for (int c = 0; c < ncols; ++c) { s += krow[c] * src[c]; }
        psi[row] = s;
    });

    // ---- 4. Fill the r_hi ghost values of B ----
    // psi table lookup (device pointer): nodal point (i, j) with
    // i in [m_nr, m_nr+m_ngr] maps to psi[(i - nr) * psi_nj + (j + ngz_psi)]
    const int psi_nj = m_psi_nj;
    const int ngz_psi = m_ngz;
    const int nz = m_nz;
    const bool periodic_z = m_periodic_z;

    for (MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {
        const Box vbx_cc = amrex::enclosedCells(mfi.validbox());
        if (vbx_cc.bigEnd(0) != nr - 1) { continue; }   // not on the r_hi face

        Array4<amrex::Real> const& Br = Bfield[0]->array(mfi);
        Array4<amrex::Real> const& Bt = Bfield[1]->array(mfi);
        Array4<amrex::Real> const& Bz = Bfield[2]->array(mfi);

        const IntVect ngv_r = Bfield[0]->nGrowVect();
        const IntVect ngv_t = Bfield[1]->nGrowVect();
        const IntVect ngv_z = Bfield[2]->nGrowVect();

        const int jlo_cc = vbx_cc.smallEnd(1);
        const int jhi_cc = vbx_cc.bigEnd(1);

        // Br (nodal r, cc z): ghost nodes i in [m_nr+1, m_nr+ngr],
        // Br = -(1/r_i) (psi(i,j+1) - psi(i,j)) / dz
        const Box br_bx(IntVect(nr + 1, jlo_cc - ngv_r[1]),
                        IntVect(nr + ngv_r[0], jhi_cc + ngv_r[1]));
        // Bz (cc r, nodal z): ghost cells i in [m_nr, m_nr+ngr-1],
        // Bz = (psi(i+1,j) - psi(i,j)) / (r_{i+1/2} dr)
        const Box bz_bx(IntVect(nr, jlo_cc - ngv_z[1]),
                        IntVect(nr + ngv_z[0] - 1, jhi_cc + ngv_z[1] + 1));
        // Btheta (cc r, cc z): Ampere continuation r_c Btheta = const
        const Box bt_bx(IntVect(nr, jlo_cc - ngv_t[1]),
                        IntVect(nr + ngv_t[0] - 1, jhi_cc + ngv_t[1]));

        amrex::ParallelFor(br_bx, bz_bx, bt_bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                const amrex::Real r = i * dr;
                const amrex::Real psi_lo = psi[(i - nr) * psi_nj + (j + ngz_psi)];
                const amrex::Real psi_hi = psi[(i - nr) * psi_nj + (j + 1 + ngz_psi)];
                Br(i, j, 0) = -(psi_hi - psi_lo) * inv_dz / r;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                const amrex::Real rc = (i + 0.5_rt) * dr;
                const amrex::Real psi_lo = psi[(i - nr) * psi_nj + (j + ngz_psi)];
                const amrex::Real psi_hi = psi[(i + 1 - nr) * psi_nj + (j + ngz_psi)];
                Bz(i, j, 0) = (psi_hi - psi_lo) * inv_dr / rc;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                // With no poloidal current beyond the face, r_c B_theta is
                // constant: continue from the outermost valid cell. For
                // non-periodic z the row is clamped to the valid range
                // (periodic ghost rows hold current data via FillBoundary).
                const int jj = periodic_z ? j : amrex::Clamp(j, 0, nz - 1);
                const amrex::Real rc_last = (nr - 0.5_rt) * dr;
                const amrex::Real rc = (i + 0.5_rt) * dr;
                Bt(i, j, 0) = Bt(nr - 1, jj, 0) * rc_last / rc;
            });
    }
}

#else  // not WARPX_DIM_RZ

bool GreensFunctionOpenBC::IsActive () { return false; }

void GreensFunctionOpenBC::Define (ablastr::fields::VectorField const& /*Bfield*/,
                                   amrex::Geometry const& /*geom*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "The Green's-function open field boundary is only available in RZ geometry.");
}

void GreensFunctionOpenBC::ApplyToBfield (ablastr::fields::VectorField const& /*Bfield*/,
                                          amrex::Geometry const& /*geom*/, int /*lev*/)
{
    WARPX_ABORT_WITH_MESSAGE(
        "The Green's-function open field boundary is only available in RZ geometry.");
}

#endif // WARPX_DIM_RZ
