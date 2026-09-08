/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_CONDUCTION_STENCIL_H_
#define MHD_CONDUCTION_STENCIL_H_

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::mhd_pc
{

/**
 * \brief Domain-boundary kind of one side of one direction of a conduction
 * channel, in the cell-centered linear-operator convention of the block
 * preconditioner's stacked Helmholtz (MHDBlockPC::DefineSignalDiffusion):
 * a periodic direction wraps, a Neumann face carries no flux, a Dirichlet
 * face (the conductive z-end exchange) holds a zero boundary value on the
 * face itself.
 */
enum struct ConductionBoundary : int
{
    Periodic = 0,
    Neumann = 1,
    Dirichlet = 2
};

/**
 * \brief Frozen geometric constants of the conduction Helmholtz rows
 *
 *     (L u)_c = alpha a_c u_c - sum_d (1/dx_d^2) [ b_{d,c+1/2} (u_{c+1} - u_c)
 *                                                - b_{d,c-1/2} (u_c - u_{c-1}) ]
 *
 * of the block preconditioner's stacked cell-centered Helmholtz (an AMReX
 * MLABecLaplacian with alpha = beta = 1, per-component face coefficients b
 * and the cell coefficient a; in RZ the equation is multiplied through by
 * r, so a = r_cell and b carries the face/cell radius). The domain
 * boundaries are folded in exactly the way the MLMG operator applies
 * them: a Neumann ghost mirrors the boundary cell (no face flux), a
 * Dirichlet ghost is AMReX's polynomial extrapolation through the zero
 * boundary value ON the face -- with N = min(box length + 1, max_order)
 * interpolation points at the ghost centre, i.e. for the default
 * max_order 3 and a box of two or more cells
 *
 *     u_ghost = -2 u_c + (1/3) u_{c'}     (c' the next cell inward),
 *
 * and u_ghost = -u_c for a one-cell box. The row emission below is the
 * single source of truth of the assembled conduction operator: the direct
 * (cuDSS) and banded exact inverses both consume it, and MHDBlockPC
 * validates it to roundoff against the MLMG operator's own matrix-free
 * application (pc_mhd_block.conduction_validate_assembly).
 */
struct ConductionStencilConstants
{
    amrex::Real alpha = 1.0;
    //! beta / dx_d^2 per direction (beta = 1).
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> face_scale = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_low = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_high = {};
    //! Boundary kind per channel, per direction, per side (0 low, 1 high),
    //! as int(ConductionBoundary).
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<int, 2>, AMREX_SPACEDIM>, 3> boundary =
        {};
    //! MLLinOp::getMaxOrder() of the stacked operator (Dirichlet ghost order).
    int max_order = 3;
};

/**
 * \brief Device-copyable views of the frozen coefficients of one box: the
 * cell coefficient a (one component), the stacked face coefficients per
 * direction (one component per stacked block component), the stacked
 * component index of each conduction channel, and the box extent per
 * direction (AMReX's Dirichlet extrapolation order depends on it).
 */
struct ConductionStencilArrays
{
    amrex::Array4<const amrex::Real> acoef;
    amrex::GpuArray<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> bcoef = {};
    amrex::GpuArray<int, 3> component = {-1, -1, -1};
    amrex::GpuArray<int, AMREX_SPACEDIM> box_length = {};
};

/**
 * \brief Coefficients of AMReX's cell-centered Dirichlet ghost fill
 * (mllinop_apply_bc_*): u_ghost = c1 u_c + c2 u_{c'} for a zero boundary
 * value on the face, from the Lagrange interpolation of N points (face,
 * first cell, second cell) at the ghost centre. N = 2 gives (c1, c2) =
 * (-1, 0); N >= 3 gives (-2, 1/3). Evaluated with the same quotients
 * AMReX's poly_interp_coeff forms, so the values are bitwise those of the
 * matrix-free application.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
ConductionDirichletGhostCoefficients (const int a_points, amrex::Real& a_c1,
                                      amrex::Real& a_c2)
{
    // Points x = {0 (face), 0.5, 1.5}, interpolation at the ghost centre
    // -0.5 (in cell widths), see AMReX_MLLinOp_K.H.
    if (a_points >= 3)
    {
        a_c1 = ((amrex::Real(-0.5) - amrex::Real(0.0)) *
                (amrex::Real(-0.5) - amrex::Real(1.5))) /
               ((amrex::Real(0.5) - amrex::Real(0.0)) *
                (amrex::Real(0.5) - amrex::Real(1.5)));
        a_c2 = ((amrex::Real(-0.5) - amrex::Real(0.0)) *
                (amrex::Real(-0.5) - amrex::Real(0.5))) /
               ((amrex::Real(1.5) - amrex::Real(0.0)) *
                (amrex::Real(1.5) - amrex::Real(0.5)));
    }
    else
    {
        a_c1 = (amrex::Real(-0.5) - amrex::Real(0.0)) /
               (amrex::Real(0.5) - amrex::Real(0.0));
        a_c2 = amrex::Real(0.0);
    }
}

/**
 * \brief Emit the sparse row of one conduction channel at cell (i, j, k):
 * a_record(ic, jc, kc, value) is called once per contribution, with the
 * row's own cell repeated for every diagonal term (the consumers sum
 * duplicates in emission order, so the host and device assemblies stay
 * bitwise consistent). Column indices of a periodic direction may lie
 * one cell outside the domain; the consumer's index map wraps them.
 * Neumann faces contribute nothing (the mirrored ghost cancels the
 * face flux); Dirichlet faces fold the extrapolated ghost into the
 * diagonal and, for a three-point extrapolation, into the next cell
 * inward.
 */
template <class Record>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EmitConductionRow (const ConductionStencilConstants& a_constants,
                   const ConductionStencilArrays& a_arrays, const int a_channel,
                   const int i, const int j, const int k, Record&& a_record)
{
    const int component = a_arrays.component[a_channel];
    const int cell[3] = {i, j, k};

    a_record(i, j, k, a_constants.alpha * a_arrays.acoef(i, j, k));

    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        const amrex::Real scale = a_constants.face_scale[direction];
        for (int side = 0; side < 2; ++side)
        {
            int face[3] = {i, j, k};
            face[direction] += side;
            const amrex::Real coefficient =
                scale * a_arrays.bcoef[direction](face[0], face[1], face[2], component);
            const int step = (side == 0) ? -1 : 1;
            const bool at_domain_face =
                (side == 0) ? (cell[direction] == a_constants.domain_low[direction])
                            : (cell[direction] == a_constants.domain_high[direction]);
            const int kind = a_constants.boundary[a_channel][direction][side];
            if (!at_domain_face || kind == static_cast<int>(ConductionBoundary::Periodic))
            {
                int neighbor[3] = {i, j, k};
                neighbor[direction] += step;
                a_record(i, j, k, coefficient);
                a_record(neighbor[0], neighbor[1], neighbor[2], -coefficient);
            }
            else if (kind == static_cast<int>(ConductionBoundary::Dirichlet))
            {
                const int points =
                    amrex::min(a_arrays.box_length[direction] + 1, a_constants.max_order);
                amrex::Real c1 = 0.0;
                amrex::Real c2 = 0.0;
                ConductionDirichletGhostCoefficients(points, c1, c2);
                a_record(i, j, k, coefficient * (amrex::Real(1.0) - c1));
                if (points >= 3)
                {
                    int inward[3] = {i, j, k};
                    inward[direction] -= step;
                    a_record(inward[0], inward[1], inward[2], -coefficient * c2);
                }
            }
            // Neumann: the mirrored ghost cancels the face flux exactly.
        }
    }
}

} // namespace warpx::mhd_pc

#endif // MHD_CONDUCTION_STENCIL_H_
