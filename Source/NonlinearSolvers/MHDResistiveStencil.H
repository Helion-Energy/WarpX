/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_RESISTIVE_STENCIL_H_
#define MHD_RESISTIVE_STENCIL_H_

#include "Utils/WarpXConst.H"

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::mhd_pc
{

/**
 * \brief Frozen geometric constants of the discrete resistive curl-curl
 *
 *     A d = theta_r dt curl((eta_field/mu0) curl d)
 *
 * on the residual's exact Yee staggering (see MHDBlockPC, resistive
 * block), with tangential E zeroed on every non-periodic domain boundary
 * plane (exact for PEC walls and the residual's zero axis EMF, the
 * identity-leaning convention for Open/None ends).
 *
 * The row emission below is the single source of truth for this stencil:
 * the matrix-free Chebyshev application and the direct-solver sparse-row
 * assembly both consume it, so the assembled matrix is the applied
 * operator by construction (validated to roundoff by the
 * pc_mhd_block.resistive_validate_assembly matvec check).
 */
struct ResistiveStencilConstants
{
    amrex::Real scale = 0.0; //!< theta_r dt / mu0
#if defined(WARPX_DIM_1D_Z)
    int axial_node_low = 0;
    int axial_node_high = 0;
    bool axial_periodic = false;
    amrex::Real inverse_dz = 0.0;
#elif defined(WARPX_DIM_RZ)
    int radial_face_low = 0;
    int radial_face_high = 0;
    int axial_face_low = 0;
    int axial_face_high = 0;
    bool axial_periodic = false;
    amrex::Real inverse_dr = 0.0;
    amrex::Real inverse_dz = 0.0;
    amrex::Real radial_lo = 0.0;
    amrex::Real radial_cell_size = 0.0;
#endif
};

/**
 * \brief eta_field on the electric-field staggerings (frozen at the
 * preconditioner update), per physical component. In 1D the transverse
 * components share the z-node array and the cell-centered component
 * never enters the curl.
 */
struct ResistiveStencilArrays
{
    amrex::Array4<const amrex::Real> eta[3];
};

/**
 * \brief Build the stencil constants from the level geometry and the
 * resistive implicit interval h_r = theta_r dt.
 */
inline ResistiveStencilConstants
MakeResistiveStencilConstants (const amrex::Geometry& a_geometry,
                               const amrex::Real a_resistive_h)
{
    ResistiveStencilConstants constants;
    constants.scale = a_resistive_h / amrex::Real(PhysConst::mu0);
    const amrex::Box domain = a_geometry.Domain();
    const auto inverse_cell_size = a_geometry.InvCellSizeArray();
#if defined(WARPX_DIM_1D_Z)
    constants.axial_node_low = domain.smallEnd(0);
    constants.axial_node_high = domain.bigEnd(0) + 1;
    constants.axial_periodic = a_geometry.isPeriodic(0);
    constants.inverse_dz = inverse_cell_size[0];
#elif defined(WARPX_DIM_RZ)
    constants.radial_face_low = domain.smallEnd(0);
    constants.radial_face_high = domain.bigEnd(0) + 1;
    constants.axial_face_low = domain.smallEnd(1);
    constants.axial_face_high = domain.bigEnd(1) + 1;
    constants.axial_periodic = a_geometry.isPeriodic(1);
    constants.inverse_dr = inverse_cell_size[0];
    constants.inverse_dz = inverse_cell_size[1];
    constants.radial_lo = static_cast<amrex::Real>(a_geometry.ProbLo(0));
    constants.radial_cell_size = static_cast<amrex::Real>(a_geometry.CellSize(0));
#else
    amrex::ignore_unused(domain, inverse_cell_size);
#endif
    return constants;
}

/**
 * \brief Emit the off-identity entries of one row of A (the resistive
 * curl-curl without the identity) as (column component, column index,
 * coefficient) pairs: sink(component, i, j, k, coefficient).
 *
 * The row is the magnetic-field face a_component at (i, j, k) on the
 * residual's staggering. Boundary-zeroed electric-field contributions are
 * simply not emitted, so out-of-domain column references occur only
 * through periodic images (the caller wraps them). The matrix-free
 * application accumulates coefficient * d(column); the sparse assembly
 * records the same pairs as matrix entries.
 */
template <typename Sink>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EmitResistiveRow (const ResistiveStencilConstants& c,
                  const ResistiveStencilArrays& a,
                  const int a_component, const int i, const int j, const int k,
                  Sink&& sink)
{
#if defined(WARPX_DIM_1D_Z)
    // Transverse components on z-cells; eta d_z d on the z-node between
    // cells node-1 and node (zero on non-periodic domain boundary nodes:
    // tangential E), then the row is -s d_z of the node flux.
    const auto emit_node_flux = [&] (const int node, const amrex::Real multiplier)
    {
        if (!c.axial_periodic &&
            (node == c.axial_node_low || node == c.axial_node_high))
        {
            return;
        }
        const amrex::Real coefficient =
            multiplier * a.eta[a_component](node, j, k) * c.inverse_dz;
        sink(a_component, node, j, k, coefficient);
        sink(a_component, node - 1, j, k, -coefficient);
    };
    emit_node_flux(i + 1, -c.scale * c.inverse_dz);
    emit_node_flux(i, c.scale * c.inverse_dz);
#elif defined(WARPX_DIM_RZ)
    // E_theta = eta (d_z d_r - d_r d_z) on the (r-node, z-node) corner,
    // zero on every non-periodic domain boundary plane (the r = 0 corner
    // reproduces the residual's zero axis EMF).
    const auto emit_corner_azimuthal =
        [&] (const int node_r, const int node_z, const amrex::Real multiplier)
    {
        if (node_r == c.radial_face_low || node_r == c.radial_face_high)
        {
            return;
        }
        if (!c.axial_periodic &&
            (node_z == c.axial_face_low || node_z == c.axial_face_high))
        {
            return;
        }
        const amrex::Real scaled_eta = multiplier * a.eta[1](node_r, node_z, k);
        sink(0, node_r, node_z, k, scaled_eta * c.inverse_dz);
        sink(0, node_r, node_z - 1, k, -scaled_eta * c.inverse_dz);
        sink(2, node_r, node_z, k, -scaled_eta * c.inverse_dr);
        sink(2, node_r - 1, node_z, k, scaled_eta * c.inverse_dr);
    };

    if (a_component == 0)
    {
        // Br on (r-node, z-cell): (curl E)_r = -d_z E_theta.
        emit_corner_azimuthal(i, j + 1, -c.scale * c.inverse_dz);
        emit_corner_azimuthal(i, j, c.scale * c.inverse_dz);
    }
    else if (a_component == 1)
    {
        // Bt on (r-cell, z-cell): (curl E)_t = d_z E_r - d_r E_z, with
        // E_r = -eta d_z d_theta on the (r-cell, z-node) face and
        // E_z = eta (1/r) d_r(r d_theta) on the (r-node, z-cell) face,
        // each zero on its non-periodic domain boundary planes
        // (tangential E; on the axis this also removes the 1/r
        // singularity).
        const auto emit_face_radial_electric =
            [&] (const int cell_r, const int node_z, const amrex::Real multiplier)
        {
            if (!c.axial_periodic &&
                (node_z == c.axial_face_low || node_z == c.axial_face_high))
            {
                return;
            }
            const amrex::Real coefficient =
                multiplier * a.eta[0](cell_r, node_z, k) * c.inverse_dz;
            sink(1, cell_r, node_z, k, -coefficient);
            sink(1, cell_r, node_z - 1, k, coefficient);
        };
        const auto emit_face_axial_electric =
            [&] (const int node_r, const int cell_z, const amrex::Real multiplier)
        {
            if (node_r == c.radial_face_low || node_r == c.radial_face_high)
            {
                return;
            }
            const amrex::Real radius_node =
                c.radial_lo + static_cast<amrex::Real>(node_r) * c.radial_cell_size;
            const amrex::Real radius_low =
                c.radial_lo +
                (static_cast<amrex::Real>(node_r) - amrex::Real(0.5)) * c.radial_cell_size;
            const amrex::Real radius_high =
                c.radial_lo +
                (static_cast<amrex::Real>(node_r) + amrex::Real(0.5)) * c.radial_cell_size;
            const amrex::Real coefficient = multiplier *
                                            a.eta[2](node_r, cell_z, k) *
                                            c.inverse_dr / radius_node;
            sink(1, node_r, cell_z, k, coefficient * radius_high);
            sink(1, node_r - 1, cell_z, k, -coefficient * radius_low);
        };
        emit_face_radial_electric(i, j + 1, c.scale * c.inverse_dz);
        emit_face_radial_electric(i, j, -c.scale * c.inverse_dz);
        emit_face_axial_electric(i + 1, j, -c.scale * c.inverse_dr);
        emit_face_axial_electric(i, j, c.scale * c.inverse_dr);
    }
    else
    {
        // Bz on (r-cell, z-node): (curl E)_z = (1/r) d_r(r E_theta), the
        // residual's r-weighted corner-EMF divergence.
        const amrex::Real radius_center =
            c.radial_lo + (static_cast<amrex::Real>(i) + amrex::Real(0.5)) *
                              c.radial_cell_size;
        const amrex::Real radius_node_low =
            c.radial_lo + static_cast<amrex::Real>(i) * c.radial_cell_size;
        const amrex::Real radius_node_high =
            c.radial_lo + (static_cast<amrex::Real>(i) + amrex::Real(1.0)) *
                              c.radial_cell_size;
        emit_corner_azimuthal(i + 1, j,
                              c.scale * c.inverse_dr * radius_node_high / radius_center);
        emit_corner_azimuthal(i, j,
                              -c.scale * c.inverse_dr * radius_node_low / radius_center);
    }
#else
    amrex::ignore_unused(c, a, a_component, i, j, k, sink);
#endif
}

} // namespace warpx::mhd_pc

#endif // MHD_RESISTIVE_STENCIL_H_
