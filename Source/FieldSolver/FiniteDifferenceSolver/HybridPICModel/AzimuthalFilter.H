/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_AZIMUTHALFILTER_H_
#define WARPX_AZIMUTHALFILTER_H_

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>
#include <AMReX_REAL.H>

#include <array>
#include <memory>

/**
 * \brief Azimuthal band-limit filter for RTZ (3D real-space cylindrical) grids.
 *
 * Near the axis the azimuthal arc length r*dtheta is much finer than the
 * radial grid scale dr, so the grid formally supports azimuthal wavenumbers
 * k_theta = m/r up to pi/(r*dtheta) >> pi/dr. Those modes carry no resolvable
 * physics but set the whistler stiffness (omega ~ k_theta^2 B / (mu0 n q)) of
 * the Ohm's-law hybrid solver's B-field substep integrators.
 *
 * This filter enforces the isotropic band limit
 *     m <= m_max(r) = max(1, floor(alpha * pi * r / dr))
 * per radial ring, i.e. k_theta <= alpha*pi/dr everywhere, by applying the
 * exact Fourier projection onto modes m <= m_max as a precomputed circulant
 * (Dirichlet-kernel) matrix per filtered ring. Rings with m_max >= NT/2 are
 * untouched, so the filter is the identity beyond r_c = NT*dr/(2*pi*alpha)
 * and on all modes it keeps; it preserves the ring average (m=0) exactly.
 * This is the real-space-theta analog of RZ azimuthal mode truncation.
 *
 * Decomposition handling: the projection needs the full ring, so the filtered
 * inner shell (all theta, all z, r < r_c) is gathered into an auxiliary
 * full-theta MultiFab via ParallelCopy, filtered there, and scattered back.
 * The scatter also fills destination ghost cells reachable from the shell's
 * valid data (theta- and z-periodic images and interior r ghosts), keeping
 * the ghost-evolved substep scheme consistent. Below-axis (r<0) guard cells
 * are not touched; their filtered-out content is amplitude-bounded by one
 * substep of nonlinear regeneration.
 */
class AzimuthalFilter
{
public:
    /** Build filter metadata from the level-0 geometry. RTZ only. */
    void Init (const amrex::Geometry& geom, amrex::Real alpha);

    [[nodiscard]] bool Enabled () const { return m_enabled; }

    /**
     * \brief Band-limit the azimuthal spectrum of mf in place.
     *
     * \param mf          field to filter (any staggering)
     * \param scomp       first component to filter
     * \param ncomp       number of components
     * \param fill_ghosts also fill mf ghost cells reachable from the shell's
     *                    valid data (theta/z periodic images); pass false for
     *                    MultiFabs with no ghost cells
     */
    void ApplyFilter (amrex::MultiFab& mf, int scomp = 0, int ncomp = 1,
                      bool fill_ghosts = true);

private:
    struct ShellData
    {
        amrex::MultiFab in;
        amrex::MultiFab out;
        /** Circulant projection matrices, [n_ring][NT][NT] */
        amrex::Gpu::DeviceVector<amrex::Real> P;
        int i_lo = 0;      /**< first filtered radial index */
        int n_ring = 0;    /**< number of filtered rings */
        int j_lo = 0;      /**< low theta index */
        bool theta_nodal = false;
        bool active = false;
    };

    /** Lazily build (or fetch) the shell for a staggering. */
    ShellData& get_shell (const amrex::IntVect& iv);

    bool m_enabled = false;
    amrex::Real m_alpha = 1.0;
    amrex::Geometry m_geom;
    int m_NT = 0;
    /** Shells keyed by staggering bits (ix + 2*iy + 4*iz) */
    std::array<std::unique_ptr<ShellData>, 8> m_shells;
};

#endif // WARPX_AZIMUTHALFILTER_H_
