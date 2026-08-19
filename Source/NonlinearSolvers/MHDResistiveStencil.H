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
 * \brief Device view of the static stair-step PEC wall mask of the
 * theta-implicit MHD solver (implicit_mhd.wall_model = pec, RZ only): a
 * shaped conducting wall represented by zeroing every tangential
 * electric-field component located on or outside the revolved wall
 * polyline (r >= r_wall(z)).
 *
 * Per electric-field staggering, the mask is the first masked radial
 * index at each axial index; a row (i, j) is masked iff
 * i >= first_masked_*[j]. The pointers are pre-offset so the axial index
 * j (nodal for er/et, cell for ez) is valid over the ghost-extended
 * range the owner built the mask for. The residual applies the same mask
 * as a linear projection of the assembled Ohm E, so dropping the masked
 * emissions here keeps the preconditioner rows exact for the wall (the
 * same convention as the non-periodic domain boundary planes).
 */
struct WallMaskView
{
    bool active = false;
    const int* first_masked_er = nullptr; //!< E_r rows: (r-cell, z-node)
    const int* first_masked_et = nullptr; //!< E_theta rows: (r-node, z-node)
    const int* first_masked_ez = nullptr; //!< E_z rows: (r-node, z-cell)
};

/**
 * \brief Frozen geometric constants of the discrete resistive curl-curl
 *
 *     A d = theta_r dt curl((eta_field/mu0) curl d)
 *          [- theta_r dt curl(eta_H lap(curl d)) / mu0 when include_hyper]
 *          [+ theta dt curl((curl d / mu0) x Hb) when include_hall]
 *          [+ s curl((C_i/mu0) curl d) when include_inertia]
 *
 * on the residual's exact Yee staggering (see MHDBlockPC, resistive
 * block), with tangential E zeroed on every non-periodic domain boundary
 * plane (exact for PEC walls and the residual's zero axis EMF, the
 * identity-leaning convention for Open/None ends). The optional
 * hyper-resistive chain reproduces the residual's E -= eta_H lap(J)
 * stencils (J = curl B / mu0 on the electric-field staggerings, the m = 0
 * cylindrical vector Laplacian with the residual's axis regularizations)
 * with eta_H frozen at the update.
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
    // Emit the hyper-resistive chain E -= eta_H laplacian(J), J = curl B /
    // mu0, alongside the resistive term (the same theta_r dt / mu0 scale
    // carries the single 1/mu0 of the current). Any current evaluation
    // whose curl-B stencil would read a magnetic-field ghost outside the
    // canonical unknowns is treated as zero (whole-point drop): exact for
    // the residual's Neumann-z current ghosts, the identity-leaning
    // convention elsewhere (same philosophy as the boundary-plane
    // tangential-E zeroing of the resistive term).
    bool include_hyper = false;
    // Emit the Hall/whistler coupling of the recast Ohm's law,
    // E += (curl dB / mu0) x Hb, linearized about the frozen coefficient
    // vector Hb = B/rho_q on the electric-field staggerings (see
    // ResistiveStencilArrays::hall) -- the whistler rotation of Chacon,
    // JCP 526 (2025) 113789, in B-form. The neglected (J0 x dB)/rho_q
    // piece of the exact Jacobian is O(|J0|/|B0| dx) smaller at whistler
    // stiffness (standard frozen-rotation linearization). The term is
    // REACTIVE, so its rows carry the global theta staging of the ideal
    // EMF through hall_scale = theta dt / mu0 (never the shifted
    // resistive theta_r of `scale`). First-order and skew: the resulting
    // block has complex conjugate eigenvalue pairs 1 +/- i h D_H kt^2,
    // outside any real Chebyshev interval -- consumers must invert it
    // directly (banded/direct LU).
    bool include_hall = false;
    amrex::Real hall_scale = 0.0; //!< theta dt / mu0
    // Emit the electron-inertia response of the recast Ohm's law: with
    // frozen density and ion current the theta-stage dJe/dt stencil
    // linearizes to dE = (C_i/mu0) (c_p1/theta) curl dB with
    // C_i = m_e_eff/(e rho_q) (see ResistiveStencilArrays::inertia), so
    // the frozen B-row gains s/mu0 curl(C_i curl dB) with the
    // dt-INDEPENDENT stencil factor s = dt c_p1 = (2 theta + 1)/2 (BDF2
    // engaged) or 1 (two-point form). An exact stencil twin of the
    // resistive rows -- the emission folds C_i into the eta coefficient
    // at inertia_scale, inheriting every boundary-plane and wall-mask
    // drop. The block is SPD and BOUNDS the skew whistler part: the
    // preconditioned symbol is the CAPPED branch,
    // (1 + s k^2 d_e^2) +/- i h D_H k^2 with d_e^2 = C_i/mu0. Reactive
    // like the Hall rows (never the shifted resistive theta_r), and
    // consumed with them by the exact banded/direct inverses.
    bool include_inertia = false;
    amrex::Real inertia_scale = 0.0; //!< s / mu0 (dt-independent)
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
    /** Stair-step PEC wall mask (inactive by default: identical rows). */
    WallMaskView wall;
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
    // eta_H on the same electric-field staggerings (frozen at the
    // preconditioner update); read only when include_hyper is set.
    amrex::Array4<const amrex::Real> eta_h[3];
    // Frozen Hall coefficient vector Hb = B/rho_q (three physical
    // components per array) on the same electric-field staggerings,
    // interpolated with the residual assembly's own stencils (total
    // cell-centered B averaged to the edge, Ohm-floored charge density);
    // read only when include_hall is set. In 1D the transverse
    // components share the z-node array and only Hb_z enters the curl.
    amrex::Array4<const amrex::Real> hall[3];
    // Frozen electron-inertia coefficient C_i = m_e_eff/(e rho_q) on the
    // same electric-field staggerings (the Ohm-floored charge density
    // with the eta stencils); read only when include_inertia is set. In
    // 1D the transverse components share the z-node array.
    amrex::Array4<const amrex::Real> inertia[3];
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
    //
    // mu0-scaled current proxy g(m) = d_z d at node m (whole-point drop
    // when its cell reads leave the canonical range, see include_hyper).
    const auto emit_gradient = [&] (const int node, const amrex::Real weight)
    {
        if (!c.axial_periodic &&
            (node <= c.axial_node_low || node >= c.axial_node_high))
        {
            return;
        }
        sink(a_component, node, j, k, weight * c.inverse_dz);
        sink(a_component, node - 1, j, k, -weight * c.inverse_dz);
    };
    const auto emit_node_flux = [&] (const int node, const amrex::Real multiplier)
    {
        if (!c.axial_periodic &&
            (node == c.axial_node_low || node == c.axial_node_high))
        {
            return;
        }
        // Electron inertia: an exact stencil twin of the eta term, folded
        // as the effective coefficient eta + (inertia_scale/scale) C_i
        // (see include_inertia) so it inherits the identical drops.
        amrex::Real edge_coefficient = a.eta[a_component](node, j, k);
        if (c.include_inertia)
        {
            edge_coefficient += (c.inertia_scale / c.scale) *
                                a.inertia[a_component](node, j, k);
        }
        const amrex::Real coefficient =
            multiplier * edge_coefficient * c.inverse_dz;
        sink(a_component, node, j, k, coefficient);
        sink(a_component, node - 1, j, k, -coefficient);
        if (c.include_hall)
        {
            // Hall EMF at the node: E = (curl dB x Hb)/mu0 couples the
            // transverse pair through Hb_z with the whistler skew sign
            // ((curl dB)_z is identically zero in 1D, so the Hb_x/Hb_y
            // content never enters). E_y = Hb_z d_z(dB_y)/mu0 feeds the
            // B_x row and E_x = Hb_z d_z(dB_x)/mu0 feeds the B_y row:
            // same node-flux form as the eta term, cross-component, with
            // opposite row signs (Chacon (2025) whistler rotation) and
            // the reactive theta staging of hall_scale.
            const int other_component = 1 - a_component;
            const amrex::Real hall_sign =
                (a_component == 0) ? amrex::Real(1.0) : amrex::Real(-1.0);
            const amrex::Real hall_coefficient =
                hall_sign * (c.hall_scale / c.scale) * multiplier *
                a.hall[a_component](node, j, k, 2) * c.inverse_dz;
            sink(other_component, node, j, k, hall_coefficient);
            sink(other_component, node - 1, j, k, -hall_coefficient);
        }
        if (c.include_hyper)
        {
            // Flux gains -eta_H lap_z(g) at the node (both transverse
            // components reduce to the same form, like the eta term).
            const amrex::Real hyper_weight = -multiplier *
                                             a.eta_h[a_component](node, j, k) *
                                             c.inverse_dz * c.inverse_dz;
            emit_gradient(node - 1, hyper_weight);
            emit_gradient(node, amrex::Real(-2.0) * hyper_weight);
            emit_gradient(node + 1, hyper_weight);
        }
    };
    emit_node_flux(i + 1, -c.scale * c.inverse_dz);
    emit_node_flux(i, c.scale * c.inverse_dz);
#elif defined(WARPX_DIM_RZ)
    const amrex::Real inverse_dr2 = c.inverse_dr * c.inverse_dr;
    const amrex::Real inverse_dz2 = c.inverse_dz * c.inverse_dz;
    // Canonical-range guards of the hyper-resistive current reads: a
    // derivative pair whose magnetic-field cells leave the canonical
    // range is dropped whole (exactly zero under the residual's
    // Neumann-z current ghosts, identity-leaning elsewhere).
    const auto radial_cell_in_range = [&] (const int cell)
    {
        return cell >= c.radial_face_low && cell < c.radial_face_high;
    };
    const auto axial_cell_in_range = [&] (const int cell)
    {
        return c.axial_periodic ||
               (cell >= c.axial_face_low && cell < c.axial_face_high);
    };
    // mu0 J_theta = -d_r Bz + d_z Br on the (r-node, z-node) corner
    // (native staggering); zero on the axis node like the residual's
    // on-axis current correction.
    const auto emit_current_azimuthal =
        [&] (const int node_r, const int node_z, const amrex::Real weight)
    {
        const amrex::Real radius =
            c.radial_lo + static_cast<amrex::Real>(node_r) * c.radial_cell_size;
        if (radius <= amrex::Real(0.5) * c.radial_cell_size)
        {
            return;
        }
        if (radial_cell_in_range(node_r) && radial_cell_in_range(node_r - 1))
        {
            sink(2, node_r, node_z, k, -weight * c.inverse_dr);
            sink(2, node_r - 1, node_z, k, weight * c.inverse_dr);
        }
        if (axial_cell_in_range(node_z) && axial_cell_in_range(node_z - 1))
        {
            sink(0, node_r, node_z, k, weight * c.inverse_dz);
            sink(0, node_r, node_z - 1, k, -weight * c.inverse_dz);
        }
    };
    // mu0 J_r = -d_z Btheta on the (r-cell, z-node) face.
    const auto emit_current_radial =
        [&] (const int cell_r, const int node_z, const amrex::Real weight)
    {
        if (!radial_cell_in_range(cell_r) || !axial_cell_in_range(node_z) ||
            !axial_cell_in_range(node_z - 1))
        {
            return;
        }
        sink(1, cell_r, node_z, k, -weight * c.inverse_dz);
        sink(1, cell_r, node_z - 1, k, weight * c.inverse_dz);
    };
    // mu0 J_z = (1/r) d_r(r Btheta) on the (r-node, z-cell) face, with
    // the residual's on-axis regularization mu0 J_z = 4 Btheta / dr.
    const auto emit_current_axial =
        [&] (const int node_r, const int cell_z, const amrex::Real weight)
    {
        if (!axial_cell_in_range(cell_z))
        {
            return;
        }
        const amrex::Real radius =
            c.radial_lo + static_cast<amrex::Real>(node_r) * c.radial_cell_size;
        if (radius <= amrex::Real(0.5) * c.radial_cell_size)
        {
            if (radial_cell_in_range(node_r))
            {
                sink(1, node_r, cell_z, k,
                     weight * amrex::Real(4.0) * c.inverse_dr);
            }
            return;
        }
        if (radial_cell_in_range(node_r) && radial_cell_in_range(node_r - 1))
        {
            const amrex::Real radius_high =
                radius + amrex::Real(0.5) * c.radial_cell_size;
            const amrex::Real radius_low =
                radius - amrex::Real(0.5) * c.radial_cell_size;
            sink(1, node_r, cell_z, k,
                 weight * c.inverse_dr * radius_high / radius);
            sink(1, node_r - 1, cell_z, k,
                 -weight * c.inverse_dr * radius_low / radius);
        }
    };
    // --- Hall/whistler edge EMFs (see include_hall): E += (g x Hb)/mu0
    // with g = mu0 J = curl dB, the frozen Hb read at the edge, and the
    // residual assembly's own cross-staggering current interpolations.
    // Each helper honors the SAME boundary-plane drops and wall-mask row
    // drops as its resistive counterpart. Callers pass multipliers built
    // on c.hall_scale (the reactive theta staging).
    //
    // E_theta += (g_z Hb_r - g_r Hb_z)/mu0 on the corner; the residual's
    // corner currents are the half-sums of the two adjacent native faces.
    const auto emit_corner_hall =
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
        if (c.wall.active && node_r >= c.wall.first_masked_et[node_z])
        {
            return;
        }
        const amrex::Real hall_radial = a.hall[1](node_r, node_z, k, 0);
        const amrex::Real hall_axial = a.hall[1](node_r, node_z, k, 2);
        emit_current_axial(node_r, node_z - 1,
                           multiplier * amrex::Real(0.5) * hall_radial);
        emit_current_axial(node_r, node_z,
                           multiplier * amrex::Real(0.5) * hall_radial);
        emit_current_radial(node_r - 1, node_z,
                            -multiplier * amrex::Real(0.5) * hall_axial);
        emit_current_radial(node_r, node_z,
                            -multiplier * amrex::Real(0.5) * hall_axial);
    };
    // E_r += (g_theta Hb_z - g_z Hb_theta)/mu0 on the (r-cell, z-node)
    // face; corner J_theta half-sum, four-point J_z average.
    const auto emit_face_radial_hall =
        [&] (const int cell_r, const int node_z, const amrex::Real multiplier)
    {
        if (!c.axial_periodic &&
            (node_z == c.axial_face_low || node_z == c.axial_face_high))
        {
            return;
        }
        if (c.wall.active && cell_r >= c.wall.first_masked_er[node_z])
        {
            return;
        }
        const amrex::Real hall_azimuthal = a.hall[0](cell_r, node_z, k, 1);
        const amrex::Real hall_axial = a.hall[0](cell_r, node_z, k, 2);
        emit_current_azimuthal(cell_r, node_z,
                               multiplier * amrex::Real(0.5) * hall_axial);
        emit_current_azimuthal(cell_r + 1, node_z,
                               multiplier * amrex::Real(0.5) * hall_axial);
        emit_current_axial(cell_r, node_z,
                           -multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_axial(cell_r + 1, node_z,
                           -multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_axial(cell_r, node_z - 1,
                           -multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_axial(cell_r + 1, node_z - 1,
                           -multiplier * amrex::Real(0.25) * hall_azimuthal);
    };
    // E_z += (g_r Hb_theta - g_theta Hb_r)/mu0 on the (r-node, z-cell)
    // face; four-point J_r average with the residual's axis clamp (the
    // clamp is exact there: Hb_theta and Hb_r vanish on the axis by the
    // m = 0 parities of the frozen coefficient fill) and corner J_theta
    // half-sum.
    const auto emit_face_axial_hall =
        [&] (const int node_r, const int cell_z, const amrex::Real multiplier)
    {
        if (node_r == c.radial_face_low || node_r == c.radial_face_high)
        {
            return;
        }
        if (c.wall.active && node_r >= c.wall.first_masked_ez[cell_z])
        {
            return;
        }
        const amrex::Real hall_radial = a.hall[2](node_r, cell_z, k, 0);
        const amrex::Real hall_azimuthal = a.hall[2](node_r, cell_z, k, 1);
        const int clamped_cell =
            (node_r - 1 < c.radial_face_low) ? node_r : node_r - 1;
        emit_current_radial(clamped_cell, cell_z,
                            multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_radial(node_r, cell_z,
                            multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_radial(clamped_cell, cell_z + 1,
                            multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_radial(node_r, cell_z + 1,
                            multiplier * amrex::Real(0.25) * hall_azimuthal);
        emit_current_azimuthal(node_r, cell_z,
                               -multiplier * amrex::Real(0.5) * hall_radial);
        emit_current_azimuthal(node_r, cell_z + 1,
                               -multiplier * amrex::Real(0.5) * hall_radial);
    };
    // E_theta -= eta_H lap J_theta at the corner (the m = 0 cylindrical
    // vector Laplacian of the residual, J_theta native so the stencil
    // needs no interpolation), dropped on every non-periodic domain
    // boundary plane like the resistive corner term.
    const auto emit_corner_hyper =
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
        if (c.wall.active && node_r >= c.wall.first_masked_et[node_z])
        {
            return;
        }
        const amrex::Real weight = -multiplier * a.eta_h[1](node_r, node_z, k);
        const amrex::Real radius =
            c.radial_lo + static_cast<amrex::Real>(node_r) * c.radial_cell_size;
        const amrex::Real radial_high =
            (radius + amrex::Real(0.5) * c.radial_cell_size) * inverse_dr2 / radius;
        const amrex::Real radial_low =
            (radius - amrex::Real(0.5) * c.radial_cell_size) * inverse_dr2 / radius;
        emit_current_azimuthal(node_r + 1, node_z, weight * radial_high);
        emit_current_azimuthal(node_r - 1, node_z, weight * radial_low);
        emit_current_azimuthal(node_r, node_z + 1, weight * inverse_dz2);
        emit_current_azimuthal(node_r, node_z - 1, weight * inverse_dz2);
        emit_current_azimuthal(node_r, node_z,
                               weight * (-radial_high - radial_low -
                                         amrex::Real(2.0) * inverse_dz2 -
                                         amrex::Real(1.0) / (radius * radius)));
    };
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
        if (c.wall.active && node_r >= c.wall.first_masked_et[node_z])
        {
            return;
        }
        // Electron-inertia fold: eta + (inertia_scale/scale) C_i (an
        // exact stencil twin; see include_inertia).
        amrex::Real edge_coefficient = a.eta[1](node_r, node_z, k);
        if (c.include_inertia)
        {
            edge_coefficient +=
                (c.inertia_scale / c.scale) * a.inertia[1](node_r, node_z, k);
        }
        const amrex::Real scaled_eta = multiplier * edge_coefficient;
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
        if (c.include_hall)
        {
            emit_corner_hall(i, j + 1, -c.hall_scale * c.inverse_dz);
            emit_corner_hall(i, j, c.hall_scale * c.inverse_dz);
        }
        if (c.include_hyper)
        {
            emit_corner_hyper(i, j + 1, -c.scale * c.inverse_dz);
            emit_corner_hyper(i, j, c.scale * c.inverse_dz);
        }
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
            if (c.wall.active && cell_r >= c.wall.first_masked_er[node_z])
            {
                return;
            }
            // Electron-inertia fold (see include_inertia).
            amrex::Real edge_coefficient = a.eta[0](cell_r, node_z, k);
            if (c.include_inertia)
            {
                edge_coefficient += (c.inertia_scale / c.scale) *
                                    a.inertia[0](cell_r, node_z, k);
            }
            const amrex::Real coefficient =
                multiplier * edge_coefficient * c.inverse_dz;
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
            if (c.wall.active && node_r >= c.wall.first_masked_ez[cell_z])
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
            // Electron-inertia fold (see include_inertia).
            amrex::Real edge_coefficient = a.eta[2](node_r, cell_z, k);
            if (c.include_inertia)
            {
                edge_coefficient += (c.inertia_scale / c.scale) *
                                    a.inertia[2](node_r, cell_z, k);
            }
            const amrex::Real coefficient = multiplier * edge_coefficient *
                                            c.inverse_dr / radius_node;
            sink(1, node_r, cell_z, k, coefficient * radius_high);
            sink(1, node_r - 1, cell_z, k, -coefficient * radius_low);
        };
        // E_r -= eta_H lap J_r on the (r-cell, z-node) face (radial part
        // of the m = 0 cylindrical vector Laplacian carries -1/r^2).
        const auto emit_face_radial_hyper =
            [&] (const int cell_r, const int node_z, const amrex::Real multiplier)
        {
            if (!c.axial_periodic &&
                (node_z == c.axial_face_low || node_z == c.axial_face_high))
            {
                return;
            }
            if (c.wall.active && cell_r >= c.wall.first_masked_er[node_z])
            {
                return;
            }
            const amrex::Real weight = -multiplier * a.eta_h[0](cell_r, node_z, k);
            const amrex::Real radius =
                c.radial_lo + (static_cast<amrex::Real>(cell_r) + amrex::Real(0.5)) *
                                  c.radial_cell_size;
            const amrex::Real radial_high =
                (radius + amrex::Real(0.5) * c.radial_cell_size) * inverse_dr2 / radius;
            const amrex::Real radial_low =
                (radius - amrex::Real(0.5) * c.radial_cell_size) * inverse_dr2 / radius;
            emit_current_radial(cell_r + 1, node_z, weight * radial_high);
            emit_current_radial(cell_r - 1, node_z, weight * radial_low);
            emit_current_radial(cell_r, node_z + 1, weight * inverse_dz2);
            emit_current_radial(cell_r, node_z - 1, weight * inverse_dz2);
            emit_current_radial(cell_r, node_z,
                                weight * (-radial_high - radial_low -
                                          amrex::Real(2.0) * inverse_dz2 -
                                          amrex::Real(1.0) / (radius * radius)));
        };
        // E_z -= eta_H lap J_z on the (r-node, z-cell) face (scalar-like
        // radial part, no -1/r^2; the node is interior here, and the
        // residual's axis form enters through the J_z(axis) neighbor).
        const auto emit_face_axial_hyper =
            [&] (const int node_r, const int cell_z, const amrex::Real multiplier)
        {
            if (node_r == c.radial_face_low || node_r == c.radial_face_high)
            {
                return;
            }
            if (c.wall.active && node_r >= c.wall.first_masked_ez[cell_z])
            {
                return;
            }
            const amrex::Real weight = -multiplier * a.eta_h[2](node_r, cell_z, k);
            const amrex::Real radius =
                c.radial_lo + static_cast<amrex::Real>(node_r) * c.radial_cell_size;
            const amrex::Real radial_high =
                (radius + amrex::Real(0.5) * c.radial_cell_size) * inverse_dr2 / radius;
            const amrex::Real radial_low =
                (radius - amrex::Real(0.5) * c.radial_cell_size) * inverse_dr2 / radius;
            emit_current_axial(node_r + 1, cell_z, weight * radial_high);
            emit_current_axial(node_r - 1, cell_z, weight * radial_low);
            emit_current_axial(node_r, cell_z + 1, weight * inverse_dz2);
            emit_current_axial(node_r, cell_z - 1, weight * inverse_dz2);
            emit_current_axial(node_r, cell_z,
                               weight * (-radial_high - radial_low -
                                         amrex::Real(2.0) * inverse_dz2));
        };
        emit_face_radial_electric(i, j + 1, c.scale * c.inverse_dz);
        emit_face_radial_electric(i, j, -c.scale * c.inverse_dz);
        emit_face_axial_electric(i + 1, j, -c.scale * c.inverse_dr);
        emit_face_axial_electric(i, j, c.scale * c.inverse_dr);
        if (c.include_hall)
        {
            emit_face_radial_hall(i, j + 1, c.hall_scale * c.inverse_dz);
            emit_face_radial_hall(i, j, -c.hall_scale * c.inverse_dz);
            emit_face_axial_hall(i + 1, j, -c.hall_scale * c.inverse_dr);
            emit_face_axial_hall(i, j, c.hall_scale * c.inverse_dr);
        }
        if (c.include_hyper)
        {
            emit_face_radial_hyper(i, j + 1, c.scale * c.inverse_dz);
            emit_face_radial_hyper(i, j, -c.scale * c.inverse_dz);
            emit_face_axial_hyper(i + 1, j, -c.scale * c.inverse_dr);
            emit_face_axial_hyper(i, j, c.scale * c.inverse_dr);
        }
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
        if (c.include_hall)
        {
            emit_corner_hall(i + 1, j,
                             c.hall_scale * c.inverse_dr * radius_node_high /
                                 radius_center);
            emit_corner_hall(i, j,
                             -c.hall_scale * c.inverse_dr * radius_node_low /
                                 radius_center);
        }
        if (c.include_hyper)
        {
            emit_corner_hyper(i + 1, j,
                              c.scale * c.inverse_dr * radius_node_high / radius_center);
            emit_corner_hyper(i, j,
                              -c.scale * c.inverse_dr * radius_node_low / radius_center);
        }
    }
#else
    amrex::ignore_unused(c, a, a_component, i, j, k, sink);
#endif
}

} // namespace warpx::mhd_pc

#endif // MHD_RESISTIVE_STENCIL_H_
