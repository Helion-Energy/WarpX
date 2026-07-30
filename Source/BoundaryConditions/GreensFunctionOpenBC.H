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
 * Cost control: the source is conservatively coarse-grained onto bins of
 * `boundary.open_bc_coarsening`^2 psi-nodes. Each bin carries its monopole
 * (total ring current) and first (r,z) moments, and the precomputed kernel
 * stores [G0, dG0/dr', dG0/dz'] per (ghost-point, bin) pair, which cancels
 * the dipole error of monopole-only binning and gives an O((h/d)^2)
 * far-field error law with a fixed kernel (Barnes-Hut/Salmon-Warren).
 * Each application is then one small GEMV. For periodic z the kernel is
 * assembled with a tolerance-driven image sum at setup time.
 *
 * Restrictions (checked with clear aborts): RZ geometry, m = 0 only,
 * hybrid-PIC solver, single level, Open selected on the r_hi face only,
 * domain including the axis (prob_lo[0] == 0).
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

    /** Linear coarsening factor of the source binning (input:
     * `boundary.open_bc_coarsening`, default 4). */
    int m_coarsening = 4;
    /** Relative tolerance for the periodic-z image sum at kernel assembly
     * (input: `boundary.open_bc_image_sum_rtol`). */
    amrex::Real m_image_sum_rtol = 1.0e-6;
    /** Cap on the number of periodic-z image pairs summed per kernel entry
     * (input: `boundary.open_bc_max_images`). */
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

    // Source bins (over nodal source indices i in [0, m_nr-1], j in [0, m_nz-1])
    int m_nbins = 0;           //!< total number of bins
    int m_nbin_z = 0;          //!< number of bins along z
    int m_ncols = 0;           //!< kernel columns = 3 * m_nbins

    /** Kernel matrix, row-major [psi-point][3*bin]: (G0, dG0/dr', dG0/dz') */
    amrex::Gpu::DeviceVector<amrex::Real> m_kernel;
    /** Bin center coordinates (device) */
    amrex::Gpu::DeviceVector<amrex::Real> m_bin_r;
    amrex::Gpu::DeviceVector<amrex::Real> m_bin_z;
    /** Scratch: coarse source moments (device) and host mirror for the reduce */
    amrex::Gpu::DeviceVector<amrex::Real> m_src;
    amrex::Vector<amrex::Real> m_src_host;
    /** Scratch: psi values at the ghost psi-points (device) */
    amrex::Gpu::DeviceVector<amrex::Real> m_psi;
};

#endif /* WARPX_GREENSFUNCTIONOPENBC_H_ */
