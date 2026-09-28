/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_AXIAL_ENDPOINT_H_
#define WARPX_IMPLICIT_AXIAL_ENDPOINT_H_
#include "Utils/WarpXConst.H"
#include "Particles/ParticleBoundaries_K.H"
#include <AMReX_Geometry.H>
#include <cmath>
namespace warpx::implicit {
// Private flat-cap extension of the deterministic endpoint image contract.
// It does not fold the pusher/deposition coordinates or mutate a particle.
struct AxialEndpointBoundary {
    amrex::Real radial_low=0.,radial_high=0.,low=0.,high=0.;
    ParticleBoundaries::ParticleBoundariesData native{};
    bool enabled=false,lower=false,upper=false;
};
struct AxialEndpointImage {
    amrex::GpuArray<amrex::ParticleReal,3> stored{},momentum{};
    bool valid=false,reflected=false;
    int side=-1;
};
inline AxialEndpointBoundary MakeAxialEndpointBoundary(
    amrex::Geometry const& g,ParticleBoundaries::ParticleBoundariesData const& native,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const& lo,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const& hi,bool enabled)
{
    AxialEndpointBoundary b;b.native=native;
#ifdef WARPX_DIM_RZ
    b.radial_low=g.ProbLo(0);b.radial_high=g.ProbHi(0);
    b.low=g.ProbLo(1);b.high=g.ProbHi(1);
    b.lower=lo[1]==FieldBoundaryType::PMC && native.zmin_bc==ParticleBoundaryType::Reflecting;
    b.upper=hi[1]==FieldBoundaryType::PMC && native.zmax_bc==ParticleBoundaryType::Reflecting;
    b.enabled=enabled && !g.isPeriodic(1) && !native.reflect_all_velocities &&
        std::isfinite(b.low) && std::isfinite(b.high) && b.low<b.high &&
        b.radial_low==0. && b.radial_high>0. && b.lower && b.upper;
#else
    amrex::ignore_unused(g,lo,hi,enabled);
#endif
    return b;
}
// Input is the actual stored endpoint (RZ r,theta,z). Only the native axial
// Reflecting branch and its normal-momentum sign are invoked. The probability
// arguments are unused by that branch; no parser/RNG/tally/buffer is executed.
AMREX_GPU_HOST_DEVICE inline AxialEndpointImage MapStoredAxialEndpoint(
    amrex::GpuArray<amrex::ParticleReal,3> const& stored,
    amrex::GpuArray<amrex::ParticleReal,3> const& momentum,
    AxialEndpointBoundary const& b)
{
    AxialEndpointImage out;out.stored=stored;out.momentum=momentum;
    for(int d=0;d<3;++d)if(!std::isfinite(stored[d])||!std::isfinite(momentum[d]))return out;
#ifdef WARPX_DIM_RZ
    if(!b.enabled || !(stored[0]>0. && stored[0]>=b.radial_low && stored[0]<b.radial_high) ||
        stored[2]==b.low || stored[2]==b.high)return out;
    bool const below=stored[2]<b.low,above=stored[2]>b.high;
    if(!below&&!above){out.valid=true;return out;}
    if((below&&!b.lower)||(above&&!b.upper))return out;
    bool flip=false,thermal=false,lost=false;
#ifdef AMREX_USE_GPU
    amrex::RandomEngine const unused_engine{nullptr};
#else
    amrex::RandomEngine const unused_engine{};
#endif
    ApplyParticleBoundaries::apply_boundary(out.stored[2],b.low,b.high,
        flip,thermal,lost,b.native.zmin_bc,b.native.zmax_bc,0.,0.,unused_engine);
    if(flip)out.momentum[2]=-out.momentum[2];
    // A second wall encounter, exact wall, or non-specular action is not
    // inferred from a single endpoint. Its request remains a decline.
    out.valid=flip&&!thermal&&!lost&&std::isfinite(out.stored[2])&&
        out.stored[2]>b.low&&out.stored[2]<b.high;
    out.reflected=out.valid;out.side=out.valid?int(above):-1;
#else
    amrex::ignore_unused(b);
#endif
    return out;
}
// Exact counterpart of the native finisher's SetPosition in RZ, followed by
// MapStoredAxialEndpoint. Its caller separately certifies the unfolded path.
AMREX_GPU_HOST_DEVICE inline AxialEndpointImage MapAxialEndpoint(
    amrex::GpuArray<amrex::ParticleReal,3> const& position,
    amrex::GpuArray<amrex::ParticleReal,3> const& momentum,
    AxialEndpointBoundary const& b)
{
    auto stored=position;
#ifdef WARPX_DIM_RZ
    stored={std::sqrt(position[0]*position[0]+position[1]*position[1]),
            std::atan2(position[1],position[0]),position[2]};
#endif
    return MapStoredAxialEndpoint(stored,momentum,b);
}
} // namespace warpx::implicit
#endif
