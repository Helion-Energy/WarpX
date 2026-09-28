/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_BORIS_PUSH_RECORDER_H_
#define WARPX_BORIS_PUSH_RECORDER_H_
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <cmath>

namespace warpx::particles {
/** One local, freshly invalidated record of the ACTUAL native Boris execution.
 * The latent impulse uses its native primed momentum and rotation coefficients.
 * It is not a subtraction of two rounded endpoint momenta. */
struct BorisPushRecord {
    amrex::GpuArray<amrex::ParticleReal,3> full{}, impulse{};
    bool valid=false;
};
struct ActualBorisPushRecorder {
    BorisPushRecord* output=nullptr;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void Invalidate() const noexcept { if(output) { *output={}; } }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void Capture(bool supported, amrex::ParticleReal econst,
        amrex::ParticleReal ex, amrex::ParticleReal ey, amrex::ParticleReal ez,
        amrex::ParticleReal px, amrex::ParticleReal py, amrex::ParticleReal pz,
        amrex::ParticleReal sx, amrex::ParticleReal sy, amrex::ParticleReal sz,
        amrex::ParticleReal ux, amrex::ParticleReal uy, amrex::ParticleReal uz) const noexcept
    {
        using P=amrex::ParticleReal;
        Invalidate();
        if(!output||!supported||!std::isfinite(econst)) { return; }
        amrex::GpuArray<P,3> const electric{ex,ey,ez}, prime{px,py,pz},
            rotation_coefficient{sx,sy,sz}, full{ux,uy,uz};
        for(int c=0;c<3;++c) {
            if(!std::isfinite(electric[c])||!std::isfinite(prime[c])||
               !std::isfinite(rotation_coefficient[c])||!std::isfinite(full[c])) { return; }
        }
        amrex::GpuArray<P,3> const rotation{py*sz-pz*sy,pz*sx-px*sz,px*sy-py*sx};
        amrex::GpuArray<P,3> impulse{};
        for(int c=0;c<3;++c) {
            P const kick=econst*electric[c];
            impulse[c]=(kick+rotation[c])+kick;
            if(!std::isfinite(impulse[c])) { return; }
        }
        output->full=full;
        output->impulse=impulse;
        output->valid=true;
    }
};
}
#endif
