/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_FINAL_GATHER_H_
#define WARPX_IMPLICIT_FINAL_GATHER_H_
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <limits>

namespace warpx::particles {
// Cartesian SI coordinates, including in RZ. Scratch belongs to one full
// implicit particle evaluation; it must be consumed before redistribution.
inline constexpr char const* FinalGatherAttributeNames[3] = {
    "implicit_final_gather_x", "implicit_final_gather_y", "implicit_final_gather_z"};
using FinalGatherPoint = amrex::GpuArray<amrex::ParticleReal, 3>;
struct FinalGatherRecord {
    amrex::GpuArray<amrex::ParticleReal*, 3> position{};
    AMREX_GPU_HOST_DEVICE explicit operator bool () const { return position[0] != nullptr; }
    AMREX_GPU_HOST_DEVICE void Invalidate (long ip) const {
        if (*this) {
            for (int d=0; d<3; ++d) {
                position[d][ip] = std::numeric_limits<amrex::ParticleReal>::quiet_NaN();
            }
        }
    }
    AMREX_GPU_HOST_DEVICE void Store (long ip, FinalGatherPoint const& point) const {
        if (*this) {
            for (int d=0; d<3; ++d) { position[d][ip] = point[d]; }
        }
    }
};
} // namespace warpx::particles
#endif
