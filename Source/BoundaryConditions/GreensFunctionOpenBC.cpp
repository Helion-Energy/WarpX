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

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
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
#include <cctype>
#include <cmath>
#include <string>

using namespace amrex::literals;

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

    /** psi-table value of the nodal point (i, j): the r_hi band holds
     * i in [nr, nr+ngr] at every j, the appended cap bands hold
     * i in [0, nr-1] at the nodal-j ghost band of each open z face.
     * Sharing one table across faces makes the corner ghosts of adjacent
     * fills agree identically. */
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real PsiAt (amrex::Real const* AMREX_RESTRICT psi, int i, int j,
                       int nr, int psi_nj, int ngz, int nz,
                       int caplo_base, int caphi_base)
    {
        if (i >= nr) { return psi[(i - nr) * psi_nj + (j + ngz)]; }
        const int row = (j <= 0)
            ? caplo_base + i * (ngz + 1) + (j + ngz)
            : caphi_base + i * (ngz + 1) + (j - nz);
        return psi[row];
    }
}

bool GreensFunctionOpenBC::IsActive ()
{
    return (WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC)
        && (WarpX::field_boundary_hi[0] == FieldBoundaryType::Open ||
            WarpX::field_boundary_lo[1] == FieldBoundaryType::Open ||
            WarpX::field_boundary_hi[1] == FieldBoundaryType::Open);
}

void GreensFunctionOpenBC::Define (ablastr::fields::VectorField const& Bfield,
                                   amrex::Geometry const& geom)
{
    const amrex::ParmParse pp_boundary("boundary");
    pp_boundary.query("open_bc_coarsening", m_coarsening);
    pp_boundary.query("open_bc_image_sum_rtol", m_image_sum_rtol);
    pp_boundary.query("open_bc_max_images", m_max_images);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_coarsening >= 1,
        "boundary.open_bc_coarsening must be >= 1");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_max_images >= 1,
        "boundary.open_bc_max_images must be >= 1 (the periodic-z image sum "
        "cannot be disabled while z is periodic)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_image_sum_rtol > 0.0_rt,
        "boundary.open_bc_image_sum_rtol must be > 0");

    // Validate the supported configuration
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::n_rz_azimuthal_modes == 1,
        "The Green's-function open boundary only supports the m = 0 azimuthal mode.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geom.ProbLo(0) == 0.0_rt,
        "The Green's-function open boundary requires the RZ domain to include "
        "the axis (geometry.prob_lo[0] == 0).");
    m_r_open = (WarpX::field_boundary_hi[0] == FieldBoundaryType::Open);
    m_z_lo_open = (WarpX::field_boundary_lo[1] == FieldBoundaryType::Open);
    m_z_hi_open = (WarpX::field_boundary_hi[1] == FieldBoundaryType::Open);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::field_boundary_lo[0] != FieldBoundaryType::Open,
        "The Green's-function open boundary is only implemented on the r_hi, "
        "z_lo and z_hi faces (r_lo is the symmetry axis).");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_r_open || m_z_lo_open || m_z_hi_open,
        "GreensFunctionOpenBC::Define called without any open face selected.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(geom.isPeriodic(1) && (m_z_lo_open || m_z_hi_open)),
        "The Green's-function open boundary on a z face requires non-periodic z.");

    // Externally applied fields initialized directly into the evolved B
    // (warpx.B_ext_grid_init_style) are curl-free inside the domain, so the
    // curl-B source deposit below does not see them: the ghost fill would
    // silently erase them at the open face and drive a spurious azimuthal
    // wall current sheet ~ B_applied / (mu0 dr) that resistively erodes the
    // applied field from the wall inward. Applied fields must instead be
    // loaded through the hybrid solver's split external-field registers,
    // which provide their own ghost values. A "constant" init style is
    // curl-free by construction and is rejected outright; the other init
    // styles (parser, file, python loaders) may legitimately load a field
    // supported by interior currents -- e.g. the plasma-response flux of an
    // equilibrium whose vacuum part rides in the external registers -- so
    // they get a warning that any curl-free component will be erased.
    {
        const amrex::ParmParse pp_warpx("warpx");
        std::string B_ext_grid_s;
        pp_warpx.query("B_ext_grid_init_style", B_ext_grid_s);
        std::transform(B_ext_grid_s.begin(), B_ext_grid_s.end(),
                       B_ext_grid_s.begin(),
                       [] (unsigned char c) { return std::tolower(c); });
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            B_ext_grid_s != "constant",
            "warpx.B_ext_grid_init_style = constant is not compatible with the "
            "Green's-function open boundary: a curl-free applied field written into "
            "the evolved B would be erased at the open face (the ghost fill "
            "reconstructs only the field of the interior currents), creating a "
            "spurious wall current sheet. Load applied fields with "
            "hybrid_pic_model.add_external_fields = 1 and "
            "hybrid_pic_model.B[x/y/z]_external_grid_function instead.");
        if (!B_ext_grid_s.empty() && B_ext_grid_s != "default") {
            ablastr::warn_manager::WMRecordWarning(
                "GreensFunctionOpenBC",
                "warpx.B_ext_grid_init_style = " + B_ext_grid_s +
                    " writes into the evolved B: the Green's-function open "
                    "boundary reconstructs ghost values from the interior "
                    "currents only, so any curl-free (applied/vacuum) component "
                    "of the loaded field will be erased at the open face. Make "
                    "sure the loaded field is supported by interior currents "
                    "and carry applied fields in the split external registers.",
                ablastr::warn_manager::WarnPriority::high);
        }
    }

    const amrex::Box& domain = geom.Domain();
    m_nr = domain.length(0);
    m_nz = domain.length(1);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        domain.smallEnd(0) == 0 && domain.smallEnd(1) == 0,
        "GreensFunctionOpenBC: unexpected domain index origin.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_nr >= 2,
        "GreensFunctionOpenBC: at least two radial cells are required.");

    const amrex::IntVect ng = Bfield[0]->nGrowVect();
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

    // psi table, r_hi band: nodal (i, j) with i in [m_nr, m_nr + m_ngr],
    // j in [-m_ngz, m_nz + m_ngz]. When r_hi is not open only the face-node
    // column i = m_nr is kept: the z-cap fills difference psi across it, and
    // the r ghosts beyond it belong to the radial boundary condition.
    m_psi_ni = m_r_open ? m_ngr + 1 : 1;
    m_psi_nj = m_nz + 2 * m_ngz + 1;
    const int nrows_r = m_psi_ni * m_psi_nj;
    // psi table, z cap bands: nodal (i, j) with i in [0, m_nr - 1] and the
    // (m_ngz + 1) nodal-j ghost band of each open cap; the i >= m_nr corner
    // columns are shared with the r_hi band so corner ghosts are filled
    // from the same psi values by every face.
    m_ncaprows_lo = m_z_lo_open ? m_nr * (m_ngz + 1) : 0;
    m_ncaprows_hi = m_z_hi_open ? m_nr * (m_ngz + 1) : 0;
    const int nrows = nrows_r + m_ncaprows_lo + m_ncaprows_hi;

    // ---- Graded source bins ------------------------------------------------
    // Source nodes are i in [1, m_nr-1], j in [0, m_nz-1]. (The r = 0 axis
    // nodes carry no ring current; the face node i = m_nr and, for
    // non-periodic z, the top node j = m_nz are excluded from the source
    // support -- see the deposition below.)
    //
    // Uniform coarsening violates the multipole acceptance criterion for
    // bins adjacent to the open face (in-bin offset ~ evaluation distance),
    // so the bin size is GRADED: a radial bin whose outermost node sits at
    // a distance d from the first evaluation radius (node i = m_nr) may not
    // be wider than d/4 in either direction, and is never wider than the
    // requested interior coarsening. Bins within a few cells of the face
    // are then single nodes (no coarse-graining error at all), and the
    // in-bin offset over evaluation distance stays <= ~1/4 everywhere, so
    // the dipole-corrected remainder keeps the far-field O((h/d)^2) law up
    // to the face.
    constexpr int mac_inv_theta = 4;   // bin extent <= (distance to face) / 4
    amrex::Vector<int> bin_wz_h, bin_nzb_h, bin_col0_h;
    amrex::Vector<amrex::Real> bin_rc_h;
    amrex::Vector<amrex::Long> bin_koff_h;
    amrex::Vector<int> rbin_of_i_h(m_nr, -1);
    amrex::Long kernel_size = 0;
    {
        int nbins = 0;
        int ihi = m_nr - 1;
        while (ihi >= 1) {
            const amrex::Real d_face = (m_nr - ihi) * m_dr;
            const int wr = amrex::Clamp(
                static_cast<int>(d_face / (mac_inv_theta * m_dr)), 1, m_coarsening);
            const int wz = amrex::Clamp(
                static_cast<int>(d_face / (mac_inv_theta * m_dz)), 1, m_coarsening);
            const int ilo = std::max(1, ihi - wr + 1);
            const int nzb = (m_nz - 1) / wz + 1;
            const auto b = static_cast<int>(bin_rc_h.size());
            for (int i = ilo; i <= ihi; ++i) { rbin_of_i_h[i] = b; }
            bin_rc_h.push_back(0.5_rt * (ilo + ihi) * m_dr);
            bin_wz_h.push_back(wz);
            bin_nzb_h.push_back(nzb);
            bin_col0_h.push_back(nbins);
            bin_koff_h.push_back(kernel_size);
            nbins += nzb;
            // per-bin kernel block: [m_psi_ni][3][offset table of length L]
            kernel_size += static_cast<amrex::Long>(m_psi_ni) * 3
                           * (m_psi_nj + static_cast<amrex::Long>(wz) * (nzb - 1));
            ihi = ilo - 1;
        }
        m_nrbins = static_cast<int>(bin_rc_h.size());
        m_nbins = nbins;
        m_ncols = 3 * nbins;
    }

    // ---- Kernel assembly ---------------------------------------------------
    // Free space is translation-invariant in z (with or without the
    // periodic-z image sum), so the kernel is NOT stored as a dense
    // (psi-point x bin) matrix: for each radial bin it is a per-moment
    // table over the integer z-offset o = j_psi - wz * j_bin between the
    // psi point and the bin's nominal center. This factorization keeps the
    // kernel at O(n_radial_bins * nz) reals -- tens of MB at production
    // resolutions -- instead of the O(nz^2 nr) dense form (GBs), and
    // shrinks the setup-time elliptic-integral count by the same factor.
    //
    // Entries are assembled on the host (elliptic integrals; setup only),
    // then copied to device. For periodic z, image rings at z' + n L_z are
    // summed in +/- pairs until the pair increment drops below
    // m_image_sum_rtol relative to the accumulated value.
    amrex::Vector<amrex::Real> kernel_h(kernel_size);
    const amrex::Real Lz = geom.ProbLength(1);
    const amrex::Real rtol = m_image_sum_rtol;
    const int max_images = m_max_images;
    const bool periodic_z = m_periodic_z;
    const int psi_ni = m_psi_ni;
    const int psi_nj = m_psi_nj;
    const int ngz = m_ngz;
    const int nr = m_nr;
    const amrex::Real dr = m_dr;
    const amrex::Real dz = m_dz;

#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < m_nrbins; ++b) {
        const int wz = bin_wz_h[b];
        const int nzb = bin_nzb_h[b];
        const int L = psi_nj + wz * (nzb - 1);
        const int omin = -wz * (nzb - 1);
        const amrex::Real rs = bin_rc_h[b];
        for (int ip = 0; ip < psi_ni; ++ip) {
            const amrex::Real rb = (nr + ip) * dr;
            for (int io = 0; io < L; ++io) {
                // z of the psi point relative to the bin's nominal center
                const amrex::Real zb = (omin + io - ngz - 0.5_rt * (wz - 1)) * dz;
                // finite-difference step for the dipole entries, scaled to
                // the n = 0 source-observer distance
                const amrex::Real d0 = std::sqrt(zb * zb + (rb + rs) * (rb + rs));
                const amrex::Real h = 1.0e-5_rt * d0;
                amrex::Real g = 0.0_rt, gr = 0.0_rt, gz = 0.0_rt;
                auto add_image = [&] (amrex::Real zoff) -> amrex::Real {
                    const amrex::Real g0 = RingPsiPerAmp(rb, zb, rs, zoff);
                    g  += g0;
                    gr += (RingPsiPerAmp(rb, zb, rs + h, zoff)
                           - RingPsiPerAmp(rb, zb, rs - h, zoff)) / (2.0_rt * h);
                    gz += (RingPsiPerAmp(rb, zb, rs, zoff + h)
                           - RingPsiPerAmp(rb, zb, rs, zoff - h)) / (2.0_rt * h);
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
                const amrex::Long idx =
                    bin_koff_h[b] + (static_cast<amrex::Long>(ip) * 3) * L + io;
                kernel_h[idx        ] = g;
                kernel_h[idx +     L] = gr;
                kernel_h[idx + 2 * L] = gz;
            }
        }
    }

    // ---- Dense cap-row kernel ----------------------------------------------
    // The z-cap psi rows evaluate at a fixed nodal-j band while the source
    // bins span all of z, so the z-offset factorization above buys nothing
    // for them: their kernel is a dense [cap row][3 * nbins] block, columns
    // aligned with the source-moment vector, with the moments evaluated at
    // each bin's actual (r_c, z_c) center (same finite-difference dipole
    // recipe as the offset tables). Open z caps require non-periodic z, so
    // there is no image sum.
    //
    // Unlike the r_hi band (whose face node is dropped from the source
    // support), the boundary-plane source nodes coincide with cap psi
    // points: the filament kernel is log-singular there (K(m -> 1)), and a
    // raw entry turns any boundary-plane current -- including the
    // per-stage feedback seed of the fill's own ghost row -- into a
    // runaway. Regularize sub-cell separations as an equivalent filament
    // of minor radius a (cell-scale), which bounds the self entry at the
    // standard mu0 r (ln(8r/a) - 2) loop self-inductance scale.
    const int ncaprows = m_ncaprows_lo + m_ncaprows_hi;
    const amrex::Long cap_kernel_size =
        static_cast<amrex::Long>(ncaprows) * m_ncols;
    amrex::Vector<amrex::Real> cap_kernel_h(cap_kernel_size);
    if (ncaprows > 0) {
        const int nz = m_nz;
        const int ncaprows_lo = m_ncaprows_lo;
        const int ncols = m_ncols;
        const int nrbins = m_nrbins;
        const amrex::Real a_reg = 0.5_rt * std::min(dr, dz);
        const auto ring_psi_reg =
            [a_reg] (amrex::Real rb, amrex::Real zb,
                     amrex::Real rs, amrex::Real zs) -> amrex::Real
        {
            const amrex::Real dzs = zb - zs;
            const amrex::Real drs = rb - rs;
            if (dzs * dzs + drs * drs >= a_reg * a_reg) {
                return RingPsiPerAmp(rb, zb, rs, zs);
            }
            const amrex::Real dz_eff =
                std::sqrt(std::max(a_reg * a_reg - drs * drs, 0.0_rt));
            return RingPsiPerAmp(rb, zb, rs, zb - dz_eff);
        };
#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int row = 0; row < ncaprows; ++row) {
            const bool hi_side = (row >= ncaprows_lo);
            const int local = hi_side ? row - ncaprows_lo : row;
            const int i = local / (ngz + 1);
            const int jj = local % (ngz + 1);
            // nodal j of the psi point: [-ngz, 0] (lo cap), [nz, nz+ngz] (hi)
            const int j = hi_side ? nz + jj : jj - ngz;
            const amrex::Real rb = i * dr;
            amrex::Real* const krow =
                cap_kernel_h.data() + static_cast<amrex::Long>(row) * ncols;
            for (int b = 0; b < nrbins; ++b) {
                const amrex::Real rs = bin_rc_h[b];
                const int wz = bin_wz_h[b];
                const int nzb = bin_nzb_h[b];
                for (int bj = 0; bj < nzb; ++bj) {
                    // z of the psi point relative to the bin's actual center
                    const amrex::Real zb = (j - (wz * bj + 0.5_rt * (wz - 1))) * dz;
                    const amrex::Real d0 =
                        std::sqrt(zb * zb + (rb + rs) * (rb + rs));
                    const amrex::Real h = 1.0e-5_rt * d0;
                    const int c0 = 3 * (bin_col0_h[b] + bj);
                    krow[c0    ] = ring_psi_reg(rb, zb, rs, 0.0_rt);
                    krow[c0 + 1] = (ring_psi_reg(rb, zb, rs + h, 0.0_rt)
                                    - ring_psi_reg(rb, zb, rs - h, 0.0_rt))
                                   / (2.0_rt * h);
                    krow[c0 + 2] = (ring_psi_reg(rb, zb, rs, h)
                                    - ring_psi_reg(rb, zb, rs, -h))
                                   / (2.0_rt * h);
                }
            }
        }
    }

    m_kernel.resize(kernel_h.size());
    m_cap_kernel.resize(cap_kernel_h.size());
    m_rbin_of_i.resize(m_nr);
    m_bin_rc.resize(m_nrbins);
    m_bin_wz.resize(m_nrbins);
    m_bin_nzb.resize(m_nrbins);
    m_bin_col0.resize(m_nrbins);
    m_bin_koff.resize(m_nrbins);
    m_src.resize(m_ncols);
    m_src_host.resize(m_ncols);
    m_psi.resize(nrows);
    m_psi_host.resize(nrows);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        kernel_h.begin(), kernel_h.end(), m_kernel.begin());
    if (ncaprows > 0) {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
            cap_kernel_h.begin(), cap_kernel_h.end(), m_cap_kernel.begin());
    }
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        rbin_of_i_h.begin(), rbin_of_i_h.end(), m_rbin_of_i.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        bin_rc_h.begin(), bin_rc_h.end(), m_bin_rc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        bin_wz_h.begin(), bin_wz_h.end(), m_bin_wz.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        bin_nzb_h.begin(), bin_nzb_h.end(), m_bin_nzb.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        bin_col0_h.begin(), bin_col0_h.end(), m_bin_col0.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        bin_koff_h.begin(), bin_koff_h.end(), m_bin_koff.begin());
    amrex::Gpu::streamSynchronize();

    std::string faces;
    if (m_r_open) { faces += " r_hi"; }
    if (m_z_lo_open) { faces += " z_lo"; }
    if (m_z_hi_open) { faces += " z_hi"; }
    amrex::Print() << "GreensFunctionOpenBC: open (free-space) boundary active on" << faces
                   << "; " << m_nrbins << " graded radial bins, " << m_nbins
                   << " source bins, kernel " << kernel_size << " reals ("
                   << static_cast<double>(kernel_size * sizeof(amrex::Real))
                      / (1024.0 * 1024.0) << " MB,"
                   << " cap kernel " << cap_kernel_size << " reals ("
                   << static_cast<double>(cap_kernel_size * sizeof(amrex::Real))
                      / (1024.0 * 1024.0) << " MB),"
                   << " interior coarsening " << m_coarsening << ", "
                   << (m_periodic_z ? "periodic" : "isolated") << " z)\n";

    m_defined = true;
}

void GreensFunctionOpenBC::ApplyToBfield (ablastr::fields::VectorField const& Bfield,
                                          amrex::Geometry const& geom, int lev)
{
    BL_PROFILE("GreensFunctionOpenBC::ApplyToBfield()");
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
    const int* const AMREX_RESTRICT rbin_of_i = m_rbin_of_i.data();
    const amrex::Real* const AMREX_RESTRICT bin_rc = m_bin_rc.data();
    const int* const AMREX_RESTRICT bin_wz = m_bin_wz.data();
    const int* const AMREX_RESTRICT bin_nzb = m_bin_nzb.data();
    const int* const AMREX_RESTRICT bin_col0 = m_bin_col0.data();
    const amrex::Long* const AMREX_RESTRICT bin_koff = m_bin_koff.data();

    for (amrex::MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {
        const amrex::Box vbx_cc = amrex::enclosedCells(mfi.validbox());
        amrex::Array4<amrex::Real const> const& Br = Bfield[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Bz = Bfield[2]->const_array(mfi);
        // Each source node (i, j) is owned by the rank whose valid
        // cell-centered box contains cell (i, j); this counts every node
        // exactly once and drops the r_hi face node (i = m_nr) and, for
        // non-periodic z, the top node (j = m_nz). The bottom plane node
        // (j = 0) stays in the support: its curl straddles the boundary
        // plane and reads the cap ghost row (the previous application's
        // fill when the z_lo cap is open -- a per-stage-lagged contraction
        // that converges with the integrator, same as the design doc's
        // per-stage closure treatment). Excluding it instead makes any
        // REAL boundary-plane current (an open-field-line column crossing
        // the cap) invisible to the fill, an unstable feedback channel:
        // the ghost field then misses the plane sheet, the plane-row curl
        // misreads, and the Ohm E pumps the invisible current further
        // (observed as a boundary-plane B_theta/B_r runaway on the plugged
        // racetrack hold). Restarts are safe: checkpoints store the ghost
        // cells (VisMF), so the first post-restart deposit reads exactly
        // the pre-checkpoint fill values.
        amrex::ParallelFor(vbx_cc,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                if (i < 1 || i > nr - 1) { return; }  // axis and face nodes
                // total azimuthal current at the psi node:
                // J_theta dA = (curl B)_theta dA / mu0, with the same
                // discrete curl as the Ohm's-law solver
                const amrex::Real i_ring =
                    ((Br(i, j, 0) - Br(i, j - 1, 0)) * inv_dz
                     - (Bz(i, j, 0) - Bz(i - 1, j, 0)) * inv_dr) * dA_over_mu0;
                const int b = rbin_of_i[i];
                const int wz = bin_wz[b];
                const int bj = j / wz;
                const int c0 = 3 * (bin_col0[b] + bj);
                // moments are taken about the bin's nominal expansion center
                // (the same point the kernel's offset table assumes)
                const amrex::Real rn = i * dr;
                const amrex::Real zn = zmin + j * dz;
                const amrex::Real zc = zmin + (wz * bj + 0.5_rt * (wz - 1)) * dz;
                amrex::HostDevice::Atomic::Add(&src[c0 + 0], i_ring);
                amrex::HostDevice::Atomic::Add(&src[c0 + 1], i_ring * (rn - bin_rc[b]));
                amrex::HostDevice::Atomic::Add(&src[c0 + 2], i_ring * (zn - zc));
            });
    }

    // ---- 2. Global reduction of the moments ----
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
        m_src.begin(), m_src.end(), m_src_host.begin());
    amrex::Gpu::streamSynchronize();
    amrex::ParallelDescriptor::ReduceRealSum(m_src_host.data(), m_ncols);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        m_src_host.begin(), m_src_host.end(), m_src.begin());
    amrex::Gpu::streamSynchronize();

    // ---- 3. GEMV: psi at the ghost psi-points ----
    // The rows (psi points) are distributed over the MPI ranks -- the
    // kernel tables are replicated but each row is computed by exactly one
    // rank -- and the assembled psi vector (a few thousand reals) is then
    // summed across ranks alongside the existing source reduction. The
    // r_hi-band rows contract the per-radial-bin z-offset tables; the
    // appended z-cap rows are straight dot products with their dense
    // kernel block.
    const int nrows_r = m_psi_ni * m_psi_nj;
    const int nrows = nrows_r + m_ncaprows_lo + m_ncaprows_hi;
    const int psi_nj = m_psi_nj;
    const int nrbins = m_nrbins;
    const amrex::Real* const AMREX_RESTRICT kernel = m_kernel.data();
    const amrex::Real* const AMREX_RESTRICT cap_kernel = m_cap_kernel.data();
    amrex::Real* const AMREX_RESTRICT psi = m_psi.data();

    const int nprocs = amrex::ParallelDescriptor::NProcs();
    const int myproc = amrex::ParallelDescriptor::MyProc();
    const int rows_per_rank = (nrows + nprocs - 1) / nprocs;
    const int row0 = std::min(myproc * rows_per_rank, nrows);
    const int nlocal = std::min(rows_per_rank, nrows - row0);

    amrex::ParallelFor(nrows, [=] AMREX_GPU_DEVICE (int row) { psi[row] = 0.0_rt; });
    amrex::ParallelFor(nlocal, [=] AMREX_GPU_DEVICE (int idx) {
        const int row = row0 + idx;
        if (row >= nrows_r) {
            const amrex::Real* const AMREX_RESTRICT krow =
                cap_kernel + static_cast<amrex::Long>(row - nrows_r) * ncols;
            amrex::Real s = 0.0_rt;
            for (int c = 0; c < ncols; ++c) { s += krow[c] * src[c]; }
            psi[row] = s;
            return;
        }
        const int ip = row / psi_nj;
        const int jp = row % psi_nj;
        amrex::Real s = 0.0_rt;
        for (int b = 0; b < nrbins; ++b) {
            const int wz = bin_wz[b];
            const int nzb = bin_nzb[b];
            const int L = psi_nj + wz * (nzb - 1);
            const amrex::Real* const AMREX_RESTRICT blk =
                kernel + bin_koff[b] + (static_cast<amrex::Long>(ip) * 3) * L;
            const amrex::Real* const AMREX_RESTRICT sb = src + 3 * bin_col0[b];
            for (int bj = 0; bj < nzb; ++bj) {
                // offset-table index of (psi row jp, z bin bj)
                const int oo = jp + wz * (nzb - 1 - bj);
                s += blk[oo] * sb[3 * bj]
                     + blk[L + oo] * sb[3 * bj + 1]
                     + blk[2 * L + oo] * sb[3 * bj + 2];
            }
        }
        psi[row] = s;
    });

    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
        m_psi.begin(), m_psi.end(), m_psi_host.begin());
    amrex::Gpu::streamSynchronize();
    amrex::ParallelDescriptor::ReduceRealSum(m_psi_host.data(), nrows);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
        m_psi_host.begin(), m_psi_host.end(), m_psi.begin());
    amrex::Gpu::streamSynchronize();

    if (std::getenv("WARPX_GREENS_DEBUG") != nullptr) {
        amrex::Real src_max = 0.0_rt, psi_r = 0.0_rt, psi_lo = 0.0_rt, psi_hi = 0.0_rt;
        for (int c = 0; c < m_ncols; ++c) {
            src_max = std::max(src_max, std::abs(m_src_host[c]));
        }
        for (int row = 0; row < nrows_r; ++row) {
            psi_r = std::max(psi_r, std::abs(m_psi_host[row]));
        }
        for (int row = nrows_r; row < nrows_r + m_ncaprows_lo; ++row) {
            psi_lo = std::max(psi_lo, std::abs(m_psi_host[row]));
        }
        for (int row = nrows_r + m_ncaprows_lo; row < nrows; ++row) {
            psi_hi = std::max(psi_hi, std::abs(m_psi_host[row]));
        }
        amrex::Vector<amrex::Real> cap_check(m_cap_kernel.size());
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
            m_cap_kernel.begin(), m_cap_kernel.end(), cap_check.begin());
        amrex::Gpu::streamSynchronize();
        amrex::Real cap_max = 0.0_rt;
        double cap_sum = 0.0;
        for (const amrex::Real v : cap_check) {
            cap_max = std::max(cap_max, std::abs(v));
            cap_sum += static_cast<double>(v);
        }
        amrex::AllPrint() << "GreensDebug: max|src| " << src_max
                          << " max|psi| r-band " << psi_r
                          << " cap-lo " << psi_lo << " cap-hi " << psi_hi
                          << " | cap kernel max " << cap_max
                          << " sum " << cap_sum << "\n";
    }

    // ---- 4. Fill the open-face ghost values of B ----
    // psi table lookup (device pointer): the r_hi band maps nodal (i, j)
    // with i in [m_nr, m_nr+m_ngr] to psi[(i - nr) * psi_nj + (j + ngz_psi)];
    // the cap fills route interior radii through the appended cap rows
    // (see PsiAt). All fills read only valid B data and the shared psi
    // table, so they are order-independent and corner-consistent.
    const int ngz_psi = m_ngz;
    const int nz = m_nz;
    const bool periodic_z = m_periodic_z;
    const bool r_open = m_r_open;
    const bool z_lo_open = m_z_lo_open;
    const bool z_hi_open = m_z_hi_open;
    const int caplo_base = nrows_r;
    const int caphi_base = nrows_r + m_ncaprows_lo;

    for (amrex::MFIter mfi(*Bfield[0]); mfi.isValid(); ++mfi) {
        const amrex::Box vbx_cc = amrex::enclosedCells(mfi.validbox());

        amrex::Array4<amrex::Real> const& Br = Bfield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Bt = Bfield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Bz = Bfield[2]->array(mfi);

        const amrex::IntVect ngv_r = Bfield[0]->nGrowVect();
        const amrex::IntVect ngv_t = Bfield[1]->nGrowVect();
        const amrex::IntVect ngv_z = Bfield[2]->nGrowVect();

        const int jlo_cc = vbx_cc.smallEnd(1);
        const int jhi_cc = vbx_cc.bigEnd(1);
        const int ilo_cc = vbx_cc.smallEnd(0);
        const int ihi_cc = vbx_cc.bigEnd(0);

        if (r_open && ihi_cc == nr - 1) {
            // Br (nodal r, cc z): ghost nodes i in [m_nr+1, m_nr+ngr],
            // Br = -(1/r_i) (psi(i,j+1) - psi(i,j)) / dz
            const amrex::Box br_bx(amrex::IntVect(nr + 1, jlo_cc - ngv_r[1]),
                                   amrex::IntVect(nr + ngv_r[0], jhi_cc + ngv_r[1]));
            // Bz (cc r, nodal z): ghost cells i in [m_nr, m_nr+ngr-1],
            // Bz = (psi(i+1,j) - psi(i,j)) / (r_{i+1/2} dr)
            const amrex::Box bz_bx(amrex::IntVect(nr, jlo_cc - ngv_z[1]),
                                   amrex::IntVect(nr + ngv_z[0] - 1, jhi_cc + ngv_z[1] + 1));
            // Btheta (cc r, cc z): Ampere continuation r_c Btheta = const
            const amrex::Box bt_bx(amrex::IntVect(nr, jlo_cc - ngv_t[1]),
                                   amrex::IntVect(nr + ngv_t[0] - 1, jhi_cc + ngv_t[1]));

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
                    // (periodic ghost rows hold current data via FillBoundary;
                    // open-cap ghost rows get the same value from the z-cap
                    // fill's own z-invariant continuation below).
                    const int jj = periodic_z ? j : amrex::Clamp(j, 0, nz - 1);
                    const amrex::Real rc_last = (nr - 0.5_rt) * dr;
                    const amrex::Real rc = (i + 0.5_rt) * dr;
                    Bt(i, j, 0) = Bt(nr - 1, jj, 0) * rc_last / rc;
                });
        }

        // z-cap fills. The poloidal ghost components difference the shared
        // psi table exactly like the interior Yee stencil (discretely
        // divergence-free, corner values identical to the r_hi fill); the
        // toroidal component continues B_theta z-invariantly from the last
        // valid plane at fixed radius (Ampere: no radial current beyond
        // the cap, so d(r B_theta)/dz = 0). Radial extents are the box's
        // own grown range, clipped to the radii the psi table covers when
        // r_hi is open, or to the valid radii otherwise (the r ghosts then
        // belong to the radial BC, which is applied after this fill).
        for (int side = 0; side < 2; ++side) {
            const bool hi_side = (side == 1);
            if (hi_side ? !(z_hi_open && jhi_cc == nz - 1)
                        : !(z_lo_open && jlo_cc == 0)) { continue; }

            const int br_ilo = std::max(0, ilo_cc - ngv_r[0]);
            const int br_ihi = std::min(ihi_cc + 1 + ngv_r[0],
                                        r_open ? nr + ngv_r[0] : nr);
            const int bz_ilo = std::max(0, ilo_cc - ngv_z[0]);
            const int bz_ihi = std::min(ihi_cc + ngv_z[0],
                                        r_open ? nr + ngv_z[0] - 1 : nr - 1);
            const int bt_ilo = std::max(0, ilo_cc - ngv_t[0]);
            const int bt_ihi = std::min(ihi_cc + ngv_t[0], nr - 1);

            // ghost bands: Br cc-z rows, Bz nodal-z rows, Bt cc-z rows
            const amrex::Box br_bx(
                amrex::IntVect(br_ilo, hi_side ? nz : -ngv_r[1]),
                amrex::IntVect(br_ihi, hi_side ? nz - 1 + ngv_r[1] : -1));
            const amrex::Box bz_bx(
                amrex::IntVect(bz_ilo, hi_side ? nz + 1 : -ngv_z[1]),
                amrex::IntVect(bz_ihi, hi_side ? nz + ngv_z[1] : -1));
            const amrex::Box bt_bx(
                amrex::IntVect(bt_ilo, hi_side ? nz : -ngv_t[1]),
                amrex::IntVect(bt_ihi, hi_side ? nz - 1 + ngv_t[1] : -1));
            const int jt_valid = hi_side ? nz - 1 : 0;

            amrex::ParallelFor(br_bx, bz_bx, bt_bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                    if (i == 0) {
                        // on-axis Br vanishes for m = 0 (psi ~ r^2)
                        Br(i, j, 0) = 0.0_rt;
                        return;
                    }
                    const amrex::Real r = i * dr;
                    const amrex::Real psi_lo = PsiAt(psi, i, j, nr, psi_nj,
                                                     ngz_psi, nz, caplo_base, caphi_base);
                    const amrex::Real psi_hi = PsiAt(psi, i, j + 1, nr, psi_nj,
                                                     ngz_psi, nz, caplo_base, caphi_base);
                    Br(i, j, 0) = -(psi_hi - psi_lo) * inv_dz / r;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                    const amrex::Real rc = (i + 0.5_rt) * dr;
                    const amrex::Real psi_lo = PsiAt(psi, i, j, nr, psi_nj,
                                                     ngz_psi, nz, caplo_base, caphi_base);
                    const amrex::Real psi_hi = PsiAt(psi, i + 1, j, nr, psi_nj,
                                                     ngz_psi, nz, caplo_base, caphi_base);
                    Bz(i, j, 0) = (psi_hi - psi_lo) * inv_dr / rc;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) {
                    Bt(i, j, 0) = Bt(i, jt_valid, 0);
                });
        }
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
