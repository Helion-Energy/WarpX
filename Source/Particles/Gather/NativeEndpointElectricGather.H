/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ENDPOINT_ELECTRIC_GATHER_H_
#define WARPX_NATIVE_ENDPOINT_ELECTRIC_GATHER_H_
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "Particles/ShapeFactors.H"
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>
#include <cstdint>
#include <limits>

namespace warpx::particles {
using EndpointElectricPair = amrex::GpuArray<warpx::ohm::compensated::Pair,3>;
using EndpointElectricArrays = amrex::GpuArray<amrex::Array4<amrex::Real const>,3>;

/** Full-order MC gather of an already centered, owner-validated electric pair.
 * Coordinates, represented shape factors and tensor-product order match the
 * native nodal gather. Keep both operands through contraction and RZ rotation;
 * callers may round only at a scalar output boundary. This routine neither
 * selects a field epoch nor grants a stale endpoint permission to act.
 * False leaves output untouched and never reads outside either input array.
 */
template<int Order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool GatherEndpointElectricPair(amrex::ParticleReal xp,amrex::ParticleReal yp,
    amrex::ParticleReal zp,EndpointElectricArrays const& high,
    EndpointElectricArrays const& low,amrex::XDim3 const& inverse,
    amrex::XDim3 const& origin,amrex::Dim3 const& lower,
    EndpointElectricPair& output) noexcept
{
    static_assert(Order>=1&&Order<=4);
    namespace ar=warpx::ohm::compensated;
    using R=amrex::Real;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    R radius=0.;
#if defined(WARPX_DIM_RZ)
    radius=std::sqrt(xp*xp+yp*yp);
    R const qx=(radius-origin.x)*inverse.x;
#else
    R const qx=(xp-origin.x)*inverse.x;
#endif
    R const qz=(zp-origin.z)*inverse.z;
#if defined(WARPX_DIM_3D)
    R const qy=(yp-origin.y)*inverse.y;
#else
    R const qy=0.;
#endif
    R const limit=R(std::numeric_limits<int>::max()-Order-4);
    if(!std::isfinite(qx)||!std::isfinite(qy)||!std::isfinite(qz)||
       std::abs(qx)>limit||std::abs(qy)>limit||std::abs(qz)>limit)return false;
    R sx[Order+1]{},sy[Order+1]{},sz[Order+1]{};
    Compute_shape_factor<Order> shape;
    std::int64_t const wide_x=std::int64_t(lower.x)+shape(sx,qx);
#if defined(WARPX_DIM_3D)
    std::int64_t const wide_y=std::int64_t(lower.y)+shape(sy,qy);
    std::int64_t const wide_z=std::int64_t(lower.z)+shape(sz,qz);
    constexpr int ycount=Order+1;
#else
    std::int64_t const wide_y=0,wide_z=std::int64_t(lower.y)+shape(sz,qz);
    constexpr int ycount=1;sy[0]=1.;
#endif
    for(auto v:{wide_x,wide_y,wide_z})
        if(v<std::numeric_limits<int>::min()||v>std::numeric_limits<int>::max()-Order)return false;
    int const ix=int(wide_x),iy=int(wide_y),iz=int(wide_z);
    for(int c=0;c<3;++c)for(int part=0;part<2;++part){
        auto const f=part?low[c]:high[c];
#if defined(WARPX_DIM_3D)
        if(!f||f.nComp()<1||!f.contains(ix,iy,iz)||!f.contains(ix+Order,iy+Order,iz+Order))return false;
#else
        if(!f||f.nComp()<1||!f.contains(ix,iz,0)||!f.contains(ix+Order,iz+Order,0))return false;
#endif
    }
    EndpointElectricPair result{{{0.,0.},{0.,0.},{0.,0.}}};
    for(int kz=0;kz<=Order;++kz)for(int jy=0;jy<ycount;++jy)for(int jx=0;jx<=Order;++jx){
        // Each weight is the same rounded native product. Pair arithmetic
        // preserves contraction of those coefficients with the retained field.
#if defined(WARPX_DIM_3D)
        R const weight=sx[jx]*sy[jy]*sz[kz];
        int const i=ix+jx,j=iy+jy,k=iz+kz;
#else
        R const weight=sx[jx]*sz[kz];
        int const i=ix+jx,j=iz+kz,k=0;
#endif
        for(int c=0;c<3;++c){
            R const h=high[c](i,j,k),l=low[c](i,j,k);
            if(!std::isfinite(h)||!std::isfinite(l))return false;
            result[c]=ar::Add(result[c],ar::Multiply({h,l},weight));
        }
    }
#if defined(WARPX_DIM_RZ)
    R const cosine=radius>0.?xp/radius:1.,sine=radius>0.?yp/radius:0.;
    auto const radial=result[0],azimuthal=result[1];
    result[0]=ar::Add(ar::Multiply(radial,cosine),ar::Multiply(azimuthal,-sine));
    result[1]=ar::Add(ar::Multiply(radial,sine),ar::Multiply(azimuthal,cosine));
#else
    amrex::ignore_unused(radius,iy);
#endif
    for(auto const& value:result)if(!std::isfinite(value.hi)||!std::isfinite(value.lo))return false;
    output=result;return true;
#else
    amrex::ignore_unused(xp,yp,zp,high,low,inverse,origin,lower,output);
    return false;
#endif
}
} // namespace warpx::particles
#endif
