/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_ABSORBING_ENDPOINT_H_
#define WARPX_IMPLICIT_ABSORBING_ENDPOINT_H_

#include "Particles/ParticleBoundaries.H"
#include "ImplicitRadialEndpoint.H"
#include <AMReX_Geometry.H>
#include <cmath>

namespace warpx::implicit {
// Match deterministic native absorption without applying a boundary, drawing
// random numbers, moving an accepted particle or publishing a loss tally.
// Opt-in native endpoint-only RZ reflection is mapped without RNG.
// Stochastic reflection, rethermalization and in-push reflection remain outside
// this contract. All charged species participate in the density map.
enum class AbsorbingEndpointStatus { Interior, Absorbed, Unsupported, Nonfinite };
struct AbsorbingEndpointResult {
    AbsorbingEndpointStatus status = AbsorbingEndpointStatus::Interior;
    unsigned faces = 0;
    unsigned reflected_faces = 0;
};
struct AbsorbingEndpointBoundary {
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> low{}, high{};
    ParticleBoundaries::ParticleBoundariesData native{};
    unsigned qualified_faces = 0;
    bool allow_radial_reflection = false;
};
inline AbsorbingEndpointBoundary MakeAbsorbingEndpointBoundary (
    amrex::Geometry const& geometry,
    ParticleBoundaries::ParticleBoundariesData const& native,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const& low_field,
    amrex::Array<FieldBoundaryType,AMREX_SPACEDIM> const& high_field,
    bool allow_radial_reflection = false)
{
    AbsorbingEndpointBoundary result{geometry.ProbLoArray(),geometry.ProbHiArray(),native,0};
    for (int d=0;d<AMREX_SPACEDIM;++d) {
#ifdef WARPX_DIM_RZ
        if (d==0) { continue; } // only the canonical axial PMC caps
#endif
        if (low_field[d]==FieldBoundaryType::PMC) { result.qualified_faces |= 1u<<(2*d); }
        if (high_field[d]==FieldBoundaryType::PMC) { result.qualified_faces |= 1u<<(2*d+1); }
    }
#ifdef WARPX_DIM_RZ
    result.allow_radial_reflection=allow_radial_reflection && geometry.ProbLo(0)==0 &&
        high_field[0]==FieldBoundaryType::PEC && native.xmax_bc==ParticleBoundaryType::Reflecting;
#else
    amrex::ignore_unused(allow_radial_reflection);
#endif
    return result;
}

// Cartesian physical position and proper momentum, as stored by the implicit
// step-start/midpoint bridge. The native RZ boundary kernel compares r and z,
// but evaluates its x reflection parser with Cartesian ux; retain that exact
// convention instead of substituting radial momentum here.
AMREX_GPU_HOST_DEVICE inline AbsorbingEndpointResult InspectAbsorbingEndpoint (
    amrex::GpuArray<amrex::ParticleReal,3> const& position,
    amrex::GpuArray<amrex::ParticleReal,3> const& momentum,
    AbsorbingEndpointBoundary const& boundary)
{
    AbsorbingEndpointResult result;
    for (int c=0;c<3;++c) {
        if (!std::isfinite(position[c]) || !std::isfinite(momentum[c])) {
            return {AbsorbingEndpointStatus::Nonfinite,0};
        }
    }
#if defined(WARPX_DIM_RZ)
    amrex::GpuArray<amrex::ParticleReal,AMREX_SPACEDIM> x{
        std::sqrt(position[0]*position[0]+position[1]*position[1]),position[2]};
    amrex::GpuArray<int,AMREX_SPACEDIM> coordinate{0,2};
#elif defined(WARPX_DIM_3D)
    auto const& x=position;
    amrex::GpuArray<int,AMREX_SPACEDIM> coordinate{0,1,2};
#else
    amrex::ignore_unused(boundary);
    return {AbsorbingEndpointStatus::Unsupported,0};
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    auto const& b=boundary.native;
    amrex::GpuArray<ParticleBoundaryType,3> lower{b.xmin_bc,b.ymin_bc,b.zmin_bc};
    amrex::GpuArray<ParticleBoundaryType,3> upper{b.xmax_bc,b.ymax_bc,b.zmax_bc};
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        int const c=coordinate[d];
        if (!std::isfinite(x[d])) { return {AbsorbingEndpointStatus::Nonfinite,0}; }
        bool const below=x[d]<boundary.low[d], above=x[d]>boundary.high[d];
        auto const bc=below ? lower[c] : upper[c];
        if (bc==ParticleBoundaryType::Periodic) { continue; }
        // Native BC uses a strict >hi loss, while AMReX location is half-open.
        // Do not silently discard an exact-hi endpoint during Redistribute.
        if (x[d]==boundary.high[d]) { return {AbsorbingEndpointStatus::Unsupported,0}; }
        if (!below && !above) { continue; }
#ifdef WARPX_DIM_RZ
        if (d==0 && above && boundary.allow_radial_reflection &&
            bc==ParticleBoundaryType::Reflecting) {
            auto const image=MapRadialEndpoint(position,momentum,boundary.low[0],
                boundary.high[0],b,true);
            if (!image.valid || !image.reflected) { return {AbsorbingEndpointStatus::Unsupported,0}; }
            result.reflected_faces |= 1u<<1;
            continue;
        }
#endif
        if (!(boundary.qualified_faces & (1u<<(2*d+int(above))))) {
            return {AbsorbingEndpointStatus::Unsupported,0};
        }
        bool absorbed=bc==ParticleBoundaryType::Open;
        if (bc==ParticleBoundaryType::Absorbing) {
            amrex::Real probability=0;
            if (c==0) { probability=below ? b.reflection_model_xlo(-momentum[0]) : b.reflection_model_xhi(momentum[0]); }
            if (c==1) { probability=below ? b.reflection_model_ylo(-momentum[1]) : b.reflection_model_yhi(momentum[1]); }
            if (c==2) { probability=below ? b.reflection_model_zlo(-momentum[2]) : b.reflection_model_zhi(momentum[2]); }
            if (!std::isfinite(probability)) { return {AbsorbingEndpointStatus::Nonfinite,0}; }
            absorbed=probability==0;
        }
        if (!absorbed) { return {AbsorbingEndpointStatus::Unsupported,0}; }
        result.status=AbsorbingEndpointStatus::Absorbed;
        result.faces |= 1u<<(2*d+int(above));
    }
#endif
    return result;
}
} // namespace warpx::implicit
#endif
