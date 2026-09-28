/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_RADIAL_ENDPOINT_H_
#define WARPX_IMPLICIT_RADIAL_ENDPOINT_H_

#include "Utils/WarpXConst.H"
#include "Particles/ParticleBoundaries_K.H"
#include <cmath>

namespace warpx::implicit {
// A native endpoint image, not a segmented/specular orbit reconstruction.
// Stored coordinates are (r,theta,z) in RZ. Proper momentum is Cartesian.
struct RadialEndpointImage {
    amrex::GpuArray<amrex::ParticleReal,3> stored{}, momentum{};
    bool valid = false;
    bool reflected = false;
};

// Pure counterpart of FinishImplicitParticleUpdate's SetPosition followed by
// the native endpoint-only radial boundary action. The caller must separately
// qualify unreflected gather/deposit reach and exclude in-push reflection.
// Axial absorption is classified using the ORIGINAL momentum, as native BCs
// evaluate every face's loss decision before their final momentum reflection.
AMREX_GPU_HOST_DEVICE inline RadialEndpointImage MapRadialEndpoint (
    amrex::GpuArray<amrex::ParticleReal,3> const& position,
    amrex::GpuArray<amrex::ParticleReal,3> const& momentum,
    amrex::Real low, amrex::Real high,
    ParticleBoundaries::ParticleBoundariesData const& native,
    bool allow_reflection)
{
    RadialEndpointImage image;
    image.momentum=momentum;
    for (int d=0;d<3;++d) {
        if (!std::isfinite(position[d]) || !std::isfinite(momentum[d])) { return image; }
    }
#ifdef WARPX_DIM_RZ
    image.stored={std::sqrt(position[0]*position[0]+position[1]*position[1]),
                  std::atan2(position[1],position[0]),position[2]};
    auto& r=image.stored[0];
    if (!std::isfinite(r) || low!=0 || !(high>0) || r==high) { return image; }
    if (r<high) { image.valid=true; return image; }
    if (!allow_reflection || native.xmax_bc!=ParticleBoundaryType::Reflecting ||
        !(2*high-r>=low && 2*high-r<high)) { return image; }
    // Invoke the actual native kernel on private scalar values. Periodic axial
    // flags suppress unrelated axial actions. A deterministic Reflecting face
    // never consumes the random engine; no stochastic/thermal face is enabled.
    auto radial=native;
    radial.xmin_bc=ParticleBoundaryType::None;
    radial.zmin_bc=radial.zmax_bc=ParticleBoundaryType::Periodic;
    bool lost=false;
#ifdef AMREX_USE_GPU
    amrex::RandomEngine const unused_engine{nullptr};
#else
    amrex::RandomEngine const unused_engine{};
#endif
    ApplyParticleBoundaries::apply_boundaries(
        image.stored[0],image.stored[1],image.stored[2],{low,0,0},{high,0,0},
        image.momentum[0],image.momentum[1],image.momentum[2],lost,radial,
        unused_engine);
    image.valid=!lost && std::isfinite(r) && r>=low && r<high;
    for (int d=0;d<3;++d) { image.valid=image.valid && std::isfinite(image.momentum[d]); }
    image.reflected=image.valid;
#else
    amrex::ignore_unused(low,high,native,allow_reflection);
    image.stored=position;
    image.valid=true;
#endif
    return image;
}
} // namespace warpx::implicit
#endif
