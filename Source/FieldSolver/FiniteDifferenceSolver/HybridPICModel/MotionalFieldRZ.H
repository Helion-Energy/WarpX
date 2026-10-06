/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_MOTIONAL_FIELD_RZ_H_
#define WARPX_MOTIONAL_FIELD_RZ_H_

#include "ElectronViscosityPoint.H"
#include "QdsmcVolumeElement.H"

#if defined(WARPX_DIM_RZ)
/** Work-compatible Lorentz force on physical RZ Yee edges.
 * q = Ji-J and u = q/rho_edge. Gather u to nodes with I, form
 * F = -rho_node (I u) x B, and scatter with V_edge^-1 I^T V_node.
 * Dividing the result by rho_edge gives sum V_edge q.E = 0. Vacuum gates enter as sqrt(g) on both sides.
 * Boundary projections belong on BOTH sides of the transpose pair.
 */
struct MotionalFieldRZ
{
    ElectronViscosityRZEdges edges;
    QdsmcVolumeElement vn, vr, vz;
    amrex::Real floor;
    bool vacuum, smooth, axis_confined, conductor;
    amrex::Real inv_width, axis_radius, axis_inv_width, dr;

    // Apply sqrt(g) on both sides so a spatially uniform vacuum gate g
    // retains its original strength while the Lorentz work remains zero.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    gate_sqrt (amrex::Array4<amrex::Real const> const& rho,
               int i, int j, int c) const
    {
        amrex::Real raw=rho(i,j,0);
        if (c!=1) { raw=0.5*(raw+rho(i+(c==0),j+(c==2),0)); }
        if (conductor && !(raw>0.0)) { return 0.0; }
        if (!vacuum) { return 1.0; }
        amrex::Real const r=(i-edges.lo[0]+(c==0 ? 0.5 : 0.0))*dr;
        if (!smooth) { return raw<floor && (!axis_confined || r<axis_radius) ? 0.0 : 1.0; }
        amrex::Real const g=0.5*(1.0+std::tanh((raw-floor)*inv_width));
        amrex::Real const mask=axis_inv_width>0.0 ?
            0.5*(1.0-std::tanh((r-axis_radius)*axis_inv_width)) : (r<axis_radius ? 1.0 : 0.0);
        return std::sqrt(axis_confined ? 1.0-(1.0-g)*mask : g);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    weight (int i, int j, int c, int offset) const
    {
        if ((c != 2 && i == edges.lo[0]) ||
            (c == 0 && edges.pmc_hi[0] && i == edges.hi[0]+1) ||
            !edges.nodal_active(i,j,c)) { return 0.0; }
        int const ei = i + (c == 0 ? offset : 0);
        int const ej = j + (c == 2 ? offset : 0);
        if (!edges.active(ei,ej,c)) { return 0.0; }
        if (c == 1) { return 1.0; }
        int const d = c == 0 ? 0 : 1;
        int const index = d == 0 ? i : j;
        return !edges.periodic[d] &&
               (index == edges.lo[d] || index == edges.hi[d]+1) ? 1.0 : 0.5;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    density (amrex::Array4<amrex::Real const> const& rho,
             amrex::Array4<amrex::Real const> const& ped,
             bool has_ped, int i, int j, int c) const
    {
        amrex::Real value = rho(i,j,0) + (has_ped ? ped(i,j,0) : 0.0);
        if (c != 1) {
            int const ni = i + (c == 0), nj = j + (c == 2);
            value = 0.5*(value + rho(ni,nj,0) + (has_ped ? ped(ni,nj,0) : 0.0));
        }
        return amrex::max(floor,value);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    gather (amrex::Array4<amrex::Real const> const& ji,
            amrex::Array4<amrex::Real const> const& total,
            amrex::Array4<amrex::Real const> const& rho,
            amrex::Array4<amrex::Real const> const& ped,
            bool has_ped, bool hall, int i, int j, int c) const
    {
        amrex::Real value = 0.0;
        for (int offset = (c == 1 ? 0 : -1); offset <= 0; ++offset) {
            auto const w = weight(i,j,c,offset);
            if (w == 0.0) { continue; }
            int const ei = i + (c == 0 ? offset : 0);
            int const ej = j + (c == 2 ? offset : 0);
            value += w * gate_sqrt(rho,ei,ej,c) * (ji(ei,ej,0) - (hall ? total(ei,ej,0) : 0.0)) /
                     density(rho,ped,has_ped,ei,ej,c);
        }
        return value;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    scatter (amrex::Array4<amrex::Real const> const& f, int i, int j, int c) const
    {
        if (!edges.active(i,j,c)) { return 0.0; }
        if (c == 1) { return f(i,j,0,c); }
        int const ni = i + (c == 0), nj = j + (c == 2);
        auto const volume = c == 0 ? vr(i,j) : vz(i,j);
        return (weight(i,j,c,0)*vn(i,j)*f(i,j,0,c) +
                weight(ni,nj,c,-1)*vn(ni,nj)*f(ni,nj,0,c))/volume;
    }
};
#endif
#endif
