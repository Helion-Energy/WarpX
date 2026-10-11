/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_MOTIONAL_FIELD_CARTESIAN_H_
#define WARPX_MOTIONAL_FIELD_CARTESIAN_H_

#include "QdsmcVolumeElement.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <cmath>

#if defined(WARPX_DIM_3D)
/** Work-compatible Lorentz force on Cartesian Yee edges.
 * u = (Ji-J)/rho_edge; F_node = -rho_node (I sqrt(g) P u) x B.
 * E = sqrt(g) P V_edge^-1 I^T V_node F_node / rho_edge.
 * Hence sum V_edge (Ji-J).E vanishes, also for discontinuous density.
 * P removes fixed PEC edges, including ECT cut/covered edges, on BOTH sides.
 * Volumes are the physical dual-cell volumes used by the hybrid work ledger;
 * ECT's face enlargement is not a substitute for this electric-work metric.
 */
struct MotionalFieldCartesian
{
    amrex::GpuArray<int, 3> lo{}, hi{}, periodic{}, pec_lo{}, pec_hi{}, pmc_lo{}, pmc_hi{};
    amrex::GpuArray<amrex::Real, 3> dx{};
    amrex::GpuArray<amrex::Array4<int const>, 3> update{};
    amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> length{};
    QdsmcVolumeElement vn;
    amrex::GpuArray<QdsmcVolumeElement, 3> ve{};
    amrex::Real floor = 0.0, inv_width = 0.0;
    bool vacuum = false, smooth = false;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
    active (int i, int j, int k, int c) const
    {
        int const index[3] = {i, j, k};
        for (int d = 0; d < 3; ++d) {
            if (!periodic[d] &&
                (index[d] < lo[d] || index[d] > hi[d] + (d != c))) {
                return false;
            }
            if (d != c && ((pec_lo[d] && index[d] == lo[d]) ||
                           (pec_hi[d] && index[d] == hi[d] + 1))) {
                return false;
            }
        }
        // Same criterion/tolerance as HybridPICModel::ZeroConductorEdges.
        return !update[c] || (update[c](i,j,k) != 0 &&
                              length[c](i,j,k) >= (1.0 - 1.e-6) * dx[c]);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    weight (int i, int j, int k, int c, int offset) const
    {
        int const index[3] = {i, j, k};
        // Odd PMC normal-current image: project its nodal average to zero.
        if ((pmc_lo[c] && index[c] == lo[c]) ||
            (pmc_hi[c] && index[c] == hi[c] + 1)) { return 0.0; }
        if (!active(i + (c == 0 ? offset : 0), j + (c == 1 ? offset : 0),
                    k + (c == 2 ? offset : 0), c)) { return 0.0; }
        return !periodic[c] && (index[c] == lo[c] || index[c] == hi[c] + 1)
            ? 1.0 : 0.5;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    raw_density (amrex::Array4<amrex::Real const> const& rho,
                 int i, int j, int k, int c) const
    {
        return 0.5 * (rho(i,j,k) + rho(i+(c==0),j+(c==1),k+(c==2)));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    density (amrex::Array4<amrex::Real const> const& rho,
             amrex::Array4<amrex::Real const> const& ped,
             bool has_ped, int i, int j, int k, int c) const
    {
        return amrex::max(floor, raw_density(rho,i,j,k,c) +
                          (has_ped ? raw_density(ped,i,j,k,c) : 0.0));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    gate_sqrt (amrex::Array4<amrex::Real const> const& rho,
               int i, int j, int k, int c) const
    {
        if (!vacuum) { return 1.0; }
        amrex::Real const raw = raw_density(rho,i,j,k,c);
        if (!smooth) { return raw < floor ? 0.0 : 1.0; }
        return std::sqrt(0.5 * (1.0 + std::tanh((raw-floor)*inv_width)));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    gather (amrex::Array4<amrex::Real const> const& ji,
            amrex::Array4<amrex::Real const> const& total,
            amrex::Array4<amrex::Real const> const& rho,
            amrex::Array4<amrex::Real const> const& ped,
            bool has_ped, bool hall, int i, int j, int k, int c) const
    {
        amrex::Real value = 0.0;
        for (int offset = -1; offset <= 0; ++offset) {
            amrex::Real const w = weight(i,j,k,c,offset);
            if (w == 0.0) { continue; }
            int const ei = i + (c == 0 ? offset : 0);
            int const ej = j + (c == 1 ? offset : 0);
            int const ek = k + (c == 2 ? offset : 0);
            value += w * gate_sqrt(rho,ei,ej,ek,c) *
                (ji(ei,ej,ek) - (hall ? total(ei,ej,ek) : 0.0)) /
                density(rho,ped,has_ped,ei,ej,ek,c);
        }
        return value;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    scatter (amrex::Array4<amrex::Real const> const& f,
             int i, int j, int k, int c) const
    {
        if (!active(i,j,k,c)) { return 0.0; }
        int const ni = i+(c==0), nj = j+(c==1), nk = k+(c==2);
        return (weight(i,j,k,c,0)*vn(i,j,k)*f(i,j,k,c) +
                weight(ni,nj,nk,c,-1)*vn(ni,nj,nk)*f(ni,nj,nk,c)) / ve[c](i,j,k);
    }
};
#endif
#endif
