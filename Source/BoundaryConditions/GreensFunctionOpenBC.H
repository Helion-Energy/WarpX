/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_GREENSFUNCTIONOPENBC_H_
#define WARPX_GREENSFUNCTIONOPENBC_H_

#include "GreensFunctionOpenBC_fwd.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

/**
 * \brief Free-space ("open") field boundary condition for the RZ (m = 0)
 * hybrid-PIC B-field advance.
 *
 * The ghost values of B seen by the curl stencils at the r_hi domain edge
 * are set to the free-space field of the interior sources, so that the
 * boundary carries no image currents and no reflected flux:
 *
 * - The poloidal ghost field (Br, Bz) is obtained by evaluating the
 *   poloidal flux function psi = r * A_theta at the ghost psi-points
 *   through the axisymmetric ring-current Green's function,
 *
 *       psi(x_b) = Int_domain G0(x_b; x') J_theta(x') dA',
 *       G0 = (mu0/4pi) sqrt((z-z')^2 + (r+r')^2) [(2-k^2) K(k) - 2 E(k)],
 *       k^2 = 4 r r' / ((z-z')^2 + (r+r')^2),
 *
 *   (dA' the poloidal cross-section area element; G0 maps a unit ring
 *   current to psi) and then differencing psi exactly as the interior
 *   Yee stencil would (Br = -(1/r) Dz psi, Bz = (1/(r_c dr)) Dr psi).
 *   This keeps the ghost-region field discretely divergence-free by
 *   construction, and corner ghost values are produced from the same psi
 *   table so adjacent fills agree identically.
 *
 * - The toroidal ghost field B_theta is closed with Ampere's law: with no
 *   poloidal current outside the domain, r * B_theta is constant beyond the
 *   boundary, so the ghost values continue r_c B_theta from the outermost
 *   valid cell (exactly mu0 I_enc / (2 pi r)).
 *
 * The azimuthal source current is the total (plasma + external) current
 * J_theta = (curl B)_theta / mu0, evaluated with the same discrete curl the
 * Ohm's-law solver uses, so the ghost fill is an instantaneous LINEAR map
 * of the B-field state at the same time level. It is applied through
 * WarpX::ApplyBfieldBoundary, i.e. inside every RHS/stage evaluation of the
 * subcycled B integrator (RK4/RKF45), inheriting the integrator's order.
 *
 * Cost control: the source is conservatively coarse-grained onto GRADED
 * bins -- single nodes next to the open face, growing with distance under
 * a multipole acceptance criterion (bin extent <= distance/4) up to the
 * interior linear size `boundary.open_bc_coarsening`. Each bin carries its
 * monopole (total ring current) and first (r,z) moments, and the
 * precomputed kernel stores [G0, dG0/dr', dG0/dz'] per (ghost-point, bin)
 * pair, which cancels the dipole error of monopole-only binning and gives
 * an O((h/d)^2) far-field error law with a fixed kernel
 * (Barnes-Hut/Salmon-Warren) that the grading extends up to the face.
 *
 * Because free space is translation-invariant in z, the kernel is stored
 * per radial bin as a table over the psi-point/bin z-offset rather than as
 * a dense matrix: O(n_radial_bins * nz) reals in total (tens of MB at
 * production RZ resolutions). Each application is one small GEMV whose
 * rows are distributed over the MPI ranks (followed by a reduction of the
 * psi vector). For periodic z the kernel is assembled with a
 * tolerance-driven image sum at setup time.
 *
 * Restrictions (checked with clear aborts): RZ geometry, m = 0 only,
 * hybrid-PIC solver, single level, Open selected on the r_hi face only,
 * domain including the axis (prob_lo[0] == 0), and no
 * warpx.B_ext_grid_init_style applied field (a curl-free field written
 * directly into the evolved B would be erased at the open face; use the
 * hybrid solver's split external fields instead).
 */
class GreensFunctionOpenBC
{
public:
    GreensFunctionOpenBC () = default;

    /** \brief Whether the Green's-function open BC is selected: RZ geometry,
     * hybrid-PIC field solver, and `boundary.field_hi = open ...` on the
     * radial face. Cheap; safe to call from any dimensionality (returns
     * false outside RZ). */
    [[nodiscard]] static bool IsActive ();

    /** \brief Fill the r_hi ghost values of B with the free-space field of
     * the interior sources. Collective (contains a global reduction of the
     * coarse source moments); must be called on all ranks.
     *
     * \param[in,out] Bfield  the (Br, Btheta, Bz) MultiFabs at level lev
     * \param[in] geom        geometry at level lev
     * \param[in] lev         refinement level (must be 0)
     */
    void ApplyToBfield (ablastr::fields::VectorField const& Bfield,
                        amrex::Geometry const& geom, int lev);

private:
    /** Read input parameters, validate the configuration and assemble the
     * coarse-grained Green's-function kernel. Called once, lazily, on the
     * first ApplyToBfield. */
    void Define (ablastr::fields::VectorField const& Bfield,
                 amrex::Geometry const& geom);

    bool m_defined = false;

    /** Linear coarsening factor of the interior source binning (input:
     * `boundary.open_bc_coarsening`, default 4; must be >= 1). Bins near
     * the open face are graded down to single nodes regardless. */
    int m_coarsening = 4;
    /** Relative tolerance for the periodic-z image sum at kernel assembly
     * (input: `boundary.open_bc_image_sum_rtol`; must be > 0). */
    amrex::Real m_image_sum_rtol = 1.0e-6;
    /** Cap on the number of periodic-z image pairs summed per kernel entry
     * (input: `boundary.open_bc_max_images`; must be >= 1). */
    int m_max_images = 200;

    // Cached grid parameters (single level, RZ)
    int m_nr = 0;              //!< number of cells in r
    int m_nz = 0;              //!< number of cells in z
    int m_ngr = 0;             //!< B ghost cells in r
    int m_ngz = 0;             //!< B ghost cells in z
    amrex::Real m_dr = 0.0;
    amrex::Real m_dz = 0.0;
    amrex::Real m_zmin = 0.0;  //!< z of nodal index 0
    bool m_periodic_z = false;

    // psi table layout: nodal points (i, j) with i in [m_nr, m_nr + m_ngr]
    // and j in [-m_ngz, m_nz + m_ngz]
    int m_psi_ni = 0;
    int m_psi_nj = 0;

    // Graded source bins (over nodal source indices i in [1, m_nr-1],
    // j in [0, m_nz-1]): a list of radial bins, fine (single-node) next to
    // the open face and up to m_coarsening wide in the interior, each with
    // its own z-bin width (also graded).
    int m_nrbins = 0;          //!< number of radial bins
    int m_nbins = 0;           //!< total number of (r,z) bins
    int m_ncols = 0;           //!< source-moment vector length = 3 * m_nbins

    /** Radial bin of source node i (device; size m_nr, entry 0 unused) */
    amrex::Gpu::DeviceVector<int> m_rbin_of_i;
    /** Per radial bin: expansion-center radius (device) */
    amrex::Gpu::DeviceVector<amrex::Real> m_bin_rc;
    /** Per radial bin: z-bin width in nodes (device) */
    amrex::Gpu::DeviceVector<int> m_bin_wz;
    /** Per radial bin: number of z bins (device) */
    amrex::Gpu::DeviceVector<int> m_bin_nzb;
    /** Per radial bin: index of its first (r,z) bin (device) */
    amrex::Gpu::DeviceVector<int> m_bin_col0;
    /** Per radial bin: offset of its kernel block (device) */
    amrex::Gpu::DeviceVector<amrex::Long> m_bin_koff;

    /** Kernel tables, one block per radial bin. Free space is
     * z-translation-invariant, so each block is laid out as
     * [m_psi_ni][3][z-offset], with the (G0, dG0/dr', dG0/dz') moments
     * tabulated over the integer z-offset between the psi point and the
     * bin's nominal center. */
    amrex::Gpu::DeviceVector<amrex::Real> m_kernel;
    /** Scratch: coarse source moments (device) and host mirror for the reduce */
    amrex::Gpu::DeviceVector<amrex::Real> m_src;
    amrex::Vector<amrex::Real> m_src_host;
    /** Scratch: psi values at the ghost psi-points (device) and host mirror
     * for the cross-rank reduction of the distributed GEMV */
    amrex::Gpu::DeviceVector<amrex::Real> m_psi;
    amrex::Vector<amrex::Real> m_psi_host;
};

#endif /* WARPX_GREENSFUNCTIONOPENBC_H_ */
