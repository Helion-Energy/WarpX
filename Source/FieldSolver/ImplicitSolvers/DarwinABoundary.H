/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_A_BOUNDARY_H_
#define WARPX_DARWIN_A_BOUNDARY_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

/** Curl guards at PMC faces must be derived from the same reflected A.
 *  Leaving them at zero loses the odd B_t image in the next Ampere curl.
 */
inline amrex::IntVect DarwinPMCCurlGrow (amrex::Array<FieldBoundaryType, AMREX_SPACEDIM> const &lo,
                                         amrex::Array<FieldBoundaryType, AMREX_SPACEDIM> const &hi)
{
    amrex::IntVect grow(0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        grow[d] = lo[d] == FieldBoundaryType::PMC || hi[d] == FieldBoundaryType::PMC;
    }
    return grow;
}

/** Complete the prescribed-A boundary for components centered normal to a face.
 *  Their last valid point is half a cell inside the wall and must remain free.
 *  Odd reflection of A-A_bc in the ghost cells puts the Dirichlet trace on the
 *  physical face. Nodal boundary points retain the caller's prescribed values.
 *  PMC tangential nodes remain free and use even reflection of A-A_bc.
 *  Call after the nodal/EB pins and FillBoundary, so mirror values are current.
 */
inline void ApplyDarwinCellCenteredABoundary (amrex::MultiFab &A, amrex::MultiFab const &A_bc,
                                              amrex::Geometry const &geom,
                                              bool const skip_lower_axis,
                                              amrex::iMultiFab const *eb_flag = nullptr,
                                              amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo = {},
                                              amrex::GpuArray<int, AMREX_SPACEDIM> pmc_hi = {})
{
    auto const lo = amrex::lbound(amrex::convert(geom.Domain(), A.ixType()));
    auto const hi = amrex::ubound(amrex::convert(geom.Domain(), A.ixType()));
    amrex::GpuArray<int, AMREX_SPACEDIM> const lower{AMREX_D_DECL(lo.x, lo.y, lo.z)};
    amrex::GpuArray<int, AMREX_SPACEDIM> const upper{AMREX_D_DECL(hi.x, hi.y, hi.z)};
    amrex::GpuArray<int, AMREX_SPACEDIM> nodal{}, periodic{};
    int cell_centered = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        nodal[d] = A.ixType().nodeCentered(d);
        periodic[d] = geom.isPeriodic(d);
        cell_centered += !nodal[d];
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(cell_centered <= 1,
                                     "A must use the electric-field staggering");
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(A, false); mfi.isValid(); ++mfi)
    {
        auto const a = A.array(mfi);
        auto const bc = A_bc.const_array(mfi);
        auto const eb = eb_flag ? eb_flag->const_array(mfi) : amrex::Array4<int const>{};
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               // Covered conductor cells and guards retain the caller's
                               // gauge-zero pin, including physical boundary intersections.
                               if (eb && eb(i, j, k) == 0)
                               {
                                   return;
                               }
                               int const p[3] = {i, j, k};
                               int mirror[3] = {i, j, k};
                               bool reflected = false;
                               amrex::Real sign = 1.;
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   if (periodic[d])
                                   {
                                       continue;
                                   }
                                   bool const use_lower = !(skip_lower_axis && d == 0);
                                   // The caller already pins tangential/nodal boundary planes,
                                   // including their corners; do not replace those values.
                                   if (nodal[d])
                                   {
                                       if ((use_lower && p[d] <= lower[d] && !pmc_lo[d]) ||
                                           (p[d] >= upper[d] && !pmc_hi[d]))
                                       {
                                           return;
                                       }
                                       if (use_lower && p[d] < lower[d] && pmc_lo[d])
                                       {
                                           mirror[d] = 2 * lower[d] - p[d];
                                           reflected = true;
                                       }
                                       else if (p[d] > upper[d] && pmc_hi[d])
                                       {
                                           mirror[d] = 2 * upper[d] - p[d];
                                           reflected = true;
                                       }
                                   }
                                   else if (use_lower && p[d] < lower[d])
                                   {
                                       mirror[d] = 2 * lower[d] - 1 - p[d];
                                       reflected = true;
                                       sign = -sign;
                                   }
                                   else if (p[d] > upper[d])
                                   {
                                       mirror[d] = 2 * upper[d] + 1 - p[d];
                                       reflected = true;
                                       sign = -sign;
                                   }
                               }
                               if (reflected)
                               {
                                   // PMC tangential A is even about the imposed
                                   // drive; normal A remains odd, with its
                                   // trace at the face.
                                   a(i, j, k) =
                                       bc(i, j, k) + sign * (a(mirror[0], mirror[1], mirror[2]) -
                                                             bc(mirror[0], mirror[1], mirror[2]));
                               }
                           });
    }
}

/** Reflect a Yee vector component at PMC faces only. Nodal-in-normal components
 *  are even; cell-centered-in-normal components are odd. This gives E_n=0 and
 *  B_t=0 at the physical face, without pinning a normal half-cell. An optional
 *  imposed field makes the reflection act on the plasma response instead.
 */
inline void ApplyDarwinPMCVectorBoundary (amrex::MultiFab &field, amrex::Geometry const &geom,
                                          amrex::GpuArray<int, AMREX_SPACEDIM> const pmc_lo,
                                          amrex::GpuArray<int, AMREX_SPACEDIM> const pmc_hi,
                                          amrex::MultiFab const *reference = nullptr)
{
    bool any = false;
    amrex::GpuArray<int, AMREX_SPACEDIM> lower{}, upper{}, nodal{};
    auto const domain = amrex::convert(geom.Domain(), field.ixType());
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        any = any || pmc_lo[d] || pmc_hi[d];
        lower[d] = domain.smallEnd(d);
        upper[d] = domain.bigEnd(d);
        nodal[d] = field.ixType().nodeCentered(d);
    }
    if (!any)
    {
        return;
    }
    for (amrex::MFIter mfi(field, false); mfi.isValid(); ++mfi)
    {
        auto const a = field.array(mfi);
        auto const ref =
            reference ? reference->const_array(mfi) : amrex::Array4<amrex::Real const>{};
        amrex::ParallelFor(
            mfi.fabbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                int const p[3] = {i, j, k};
                int mirror[3] = {i, j, k};
                bool reflected = false;
                amrex::Real sign = 1.;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    if (pmc_lo[d] && p[d] < lower[d])
                    {
                        mirror[d] = 2 * lower[d] - (1 - nodal[d]) - p[d];
                        reflected = true;
                        if (!nodal[d])
                        {
                            sign = -sign;
                        }
                    }
                    else if (pmc_hi[d] && p[d] > upper[d])
                    {
                        mirror[d] = 2 * upper[d] + (1 - nodal[d]) - p[d];
                        reflected = true;
                        if (!nodal[d])
                        {
                            sign = -sign;
                        }
                    }
                }
                if (reflected)
                {
                    amrex::Real const value = a(mirror[0], mirror[1], mirror[2]);
                    a(i, j, k) =
                        ref ? ref(i, j, k) + sign * (value - ref(mirror[0], mirror[1], mirror[2]))
                            : sign * value;
                }
            });
    }
}
#endif
