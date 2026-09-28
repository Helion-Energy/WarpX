/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_STOPPING_VELOCITY_IMAGES_H_
#define WARPX_STOPPING_VELOCITY_IMAGES_H_
#include "NativeAcceptedStoppingContext.H"

namespace warpx::thermal {
/** Shared index/parity map for the stopping velocity extension and its
 * transpose. Periodic and inter-FAB images are handled by native communication.
 * Valid boundary rows are left unchanged, exactly as in the forward operator.
 */
struct StoppingVelocityImages {
    amrex::IntVect lo{}, hi{}, node{};
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 2>, AMREX_SPACEDIM> sign{};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (amrex::IntVect& source) const
    {
        amrex::Real parity=1.;
        for (int d=0; d<AMREX_SPACEDIM; ++d) {
            if (periodic[d]) { continue; }
            if (source[d]<lo[d]) {
                source[d]=2*lo[d]-source[d]-(1-node[d]); parity*=sign[d][0];
            } else if (source[d]>hi[d]) {
                source[d]=2*hi[d]-source[d]+(1-node[d]); parity*=sign[d][1];
            }
        }
        return parity;
    }
};
inline StoppingVelocityImages MakeStoppingVelocityImages (
    amrex::Geometry const& geometry, AcceptedStoppingOptions const& options,
    amrex::IndexType const type, int component)
{
    StoppingVelocityImages image;
    auto const domain=amrex::convert(geometry.Domain(),type);
    image.lo=domain.smallEnd(); image.hi=domain.bigEnd(); image.node=type.toIntVect();
    image.periodic=geometry.isPeriodicArray();
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
#if defined(WARPX_DIM_RZ)
        bool const normal=component==2*d;
#else
        bool const normal=component==d;
#endif
        for (int side=0; side<2; ++side) {
            auto const p=side?options.particle_hi[d]:options.particle_lo[d];
            auto const f=side?options.field_hi[d]:options.field_lo[d];
            bool const reflected=p==ParticleBoundaryType::Reflecting || f==FieldBoundaryType::PMC;
            image.sign[d][side]=reflected?(normal?-1.:1.):(normal?1.:-1.);
        }
    }
#if defined(WARPX_DIM_RZ)
    image.sign[0][0]=component==2?1.:-1.;
#endif
    return image;
}
} // namespace warpx::thermal
#endif
