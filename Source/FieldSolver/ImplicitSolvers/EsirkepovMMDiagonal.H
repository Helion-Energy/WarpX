/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ESIRKEPOV_MM_DIAGONAL_H_
#define WARPX_ESIRKEPOV_MM_DIAGONAL_H_

#include "Utils/WarpXAlgorithmSelection.H"
#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>

#if defined(WARPX_DIM_RZ)
namespace esirkepov_mm {
/** Diagonal of the composed Yee -> nodal MC gather -> current response.
 * The stored bands have nodal electric columns, so their central entry alone
 * is NOT the Yee diagonal. Include both centering weights and all electric
 * mirror images of the final current row. The caller subsequently applies the
 * ordinary current volume, exchange and boundary fold exactly once. Electric
 * parity is already included here; applying an even-coefficient fold again
 * would be wrong. No state or particle evaluation is involved.
 */
struct Diagonal {
    amrex::GpuArray<int,2> cells{};
    amrex::GpuArray<FieldBoundaryType,2> lo{}, hi{};

    AMREX_GPU_HOST_DEVICE amrex::Real map (int& i, int& j, int c,
                                         bool electric) const
    {
        int p[2] = {i,j};
        amrex::Real sign = 1.;
        for (int d = 0; d < 2; ++d) {
            int const centered = c == 2*d;
            int const last = cells[d]-centered;
            if (lo[d] == FieldBoundaryType::Periodic) {
                p[d] = (p[d]%cells[d]+cells[d])%cells[d];
                continue;
            }
            // Deposit/gather support is narrower than the physical domain.
            if (p[d] < 0 || p[d] > last) {
                bool const low = p[d] < 0;
                auto const bc = low ? lo[d] : hi[d];
                bool const axis = d == 0 && low;
                p[d] = low ? -centered-p[d] : 2*cells[d]-centered-p[d];
                bool const odd = axis ? c != 2
                    : bc == FieldBoundaryType::PMC ? centered : !centered;
                if (electric && odd) { sign = -sign; }
            }
            if (electric && !centered) {
                if ((p[d] == 0 && (d == 0 ? c == 1 : lo[d] == FieldBoundaryType::PEC)) ||
                    (p[d] == last && hi[d] == FieldBoundaryType::PEC)) {
                    sign = 0.;
                }
            }
        }
        i = p[0]; j = p[1];
        return sign;
    }

    AMREX_GPU_HOST_DEVICE amrex::Real operator() (
        amrex::Array4<amrex::Real const> const& matrix,
        amrex::Box const& electric_box, amrex::Box const& nodal_box,
        int i, int j, int c, int width) const
    {
        int target_i = i, target_j = j;
        map(target_i,target_j,c,false);
        int const radius = (width-1)/2;
        amrex::Real diagonal = 0.;
        for (int b = 0; b < width; ++b) {
            for (int a = 0; a < width; ++a) {
                int const ni = i+a-radius, nj = j+b-radius;
                if (!nodal_box.contains(amrex::IntVect(ni,nj))) { continue; }
                // Centering is second order; particle gather remains full order.
                for (int edge = 0; edge < (c == 1 ? 1 : 2); ++edge) {
                    int ei = ni-(c == 0 ? edge : 0);
                    int ej = nj-(c == 2 ? edge : 0);
                    if (!electric_box.contains(amrex::IntVect(ei,ej))) { continue; }
                    auto const sign = map(ei,ej,c,true);
                    if (ei == target_i && ej == target_j) {
                        diagonal += (c == 1 ? 1. : .5)*sign*matrix(i,j,0,a+width*b);
                    }
                }
            }
        }
        return diagonal;
    }
};
} // namespace esirkepov_mm
#endif
#endif
