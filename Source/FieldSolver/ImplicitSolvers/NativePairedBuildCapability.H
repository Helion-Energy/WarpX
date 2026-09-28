#ifndef WARPX_NATIVE_PAIRED_BUILD_CAPABILITY_H_
#define WARPX_NATIVE_PAIRED_BUILD_CAPABILITY_H_

#include <AMReX_Config.H>

// The macro alone cannot create this capability: the generated header is a
// build dependency produced by the complete precise-command audit. GNUmake,
// ordinary CUDA, HIP and SYCL builds retain the default rejection.
#if defined(AMREX_USE_CUDA) && defined(WARPX_NATIVE_PAIRED_PRECISE_CUDA_BUILD)
#include <FieldSolver/ImplicitSolvers/NativePairedCudaBuild.H>
#else
namespace warpx::darwin {
inline constexpr bool NativePairedPreciseCudaBuild = false;
}
#endif

#endif
