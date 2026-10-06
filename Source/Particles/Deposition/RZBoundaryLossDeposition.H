/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_RZ_BOUNDARY_LOSS_DEPOSITION_H_
#define WARPX_RZ_BOUNDARY_LOSS_DEPOSITION_H_

#include "Particles/ShapeFactors.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuLaunch.H>

/** Deposit the unreflected endpoint cloud of each particle actually absorbed
 *  by the native boundary decision. Components 0/1 belong to the lower/upper
 *  axial endcap. This is an event source, not a continuity residual. */
template <int order, class Position>
void DepositRZBoundaryLoss (
    Position const& get_position, unsigned char const* absorbed,
    amrex::ParticleReal const* weight, int const* ion_level, long count,
    amrex::Array4<amrex::Real> const& loss, amrex::Real charge,
    amrex::XDim3 const& inverse_cell_size, amrex::XDim3 const& lower,
    amrex::Dim3 const& index_lo)
{
    using namespace amrex::literals;
    auto const invvol=inverse_cell_size.x*inverse_cell_size.y*inverse_cell_size.z;
    // Different particles can hit the same node. For prevents CPU SIMD
    // reordering; HostDevice atomics also cover concurrent OpenMP tiles.
    amrex::For(count,[=] AMREX_GPU_DEVICE (long ip) {
        if (absorbed[ip]==0) { return; }
        int const face=absorbed[ip]-1;
        amrex::ParticleReal x,y,z;
        get_position(ip,x,y,z);
        auto const r=std::sqrt(x*x+y*y);
        amrex::Real sr[order+1]={0._rt}, sz[order+1]={0._rt};
        Compute_shape_factor<order> const shape;
        int const ir=shape(sr,(r-lower.x)*inverse_cell_size.x);
        int const iz=shape(sz,(z-lower.z)*inverse_cell_size.z);
        auto const blo=amrex::lbound(loss), bhi=amrex::ubound(loss);
        AMREX_ALWAYS_ASSERT(index_lo.x+ir>=blo.x && index_lo.y+iz>=blo.y
            && index_lo.x+ir+order<=bhi.x && index_lo.y+iz+order<=bhi.y);
        auto const q=charge*weight[ip]*invvol*(ion_level ? ion_level[ip] : 1);
        for (int j=0;j<=order;++j) {
            for (int i=0;i<=order;++i) {
                amrex::HostDevice::Atomic::Add(
                    &loss(index_lo.x+ir+i,index_lo.y+iz+j,0,face),q*sr[i]*sz[j]);
            }
        }
    });
}
#endif
