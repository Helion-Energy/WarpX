/* Diagnostic prototype. Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_IMPULSE_RECORD_H_
#define WARPX_NATIVE_IMPULSE_RECORD_H_
#include "NativeBorisImpulse.H"
namespace warpx::particles {
/** Transient record written only by the last converged native orbit iteration.
 * It retains the latent proper-momentum impulse from the ACTUAL gathered force.
 * No new gather, particle update, reflection, or rounded-endpoint subtraction.
 */
struct NativeImpulseRecord {
    amrex::GpuArray<amrex::ParticleReal*,3> impulse{};
    int* valid=nullptr;
    AMREX_GPU_HOST_DEVICE explicit operator bool() const noexcept { return valid!=nullptr; }
    AMREX_GPU_HOST_DEVICE void Invalidate(long p) const noexcept {
        if(valid) { valid[p]=0;for(int c=0;c<3;++c)impulse[c][p]=0.; }
    }
    AMREX_GPU_HOST_DEVICE void Store(long p,amrex::GpuArray<amrex::ParticleReal,3> const& du) const noexcept {
        if(valid) { for(int c=0;c<3;++c)impulse[c][p]=du[c];valid[p]=1; }
    }
};
}
#endif
