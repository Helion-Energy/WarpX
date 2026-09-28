/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ENDPOINT_ARITHMETIC_H_
#define WARPX_NATIVE_ENDPOINT_ARITHMETIC_H_
#include "NativePairedBuildCapability.H"
#include "ThermalCurrentRemainder.H"
#include <AMReX_REAL.H>
#include <cfenv>
#include <limits>
#include <type_traits>
#if defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif
namespace warpx::darwin {
// Endpoint-only arithmetic capability. The generated precise CUDA receipt
// binds every native consumer to no reassociation/contraction, gradual
// underflow and precise division/sqrt; this is not runtime qualification.
// Internal linkage keeps preprocessing checks local to each consuming TU.
[[maybe_unused]] static inline bool NativeEndpointArithmeticSupported() noexcept {
#if !defined(AMREX_USE_GPU)
    // Preserve the original CPU F-certificate predicate byte for byte.
    return warpx::thermal::remainder::ArithmeticSupported();
#elif defined(WARPX_DIM_RZ) && defined(AMREX_USE_CUDA) && defined(__CUDACC__) && \
      !defined(__HIPCC__) && !defined(SYCL_LANGUAGE_VERSION) && \
      !defined(__FAST_MATH__) && \
      (!defined(__FINITE_MATH_ONLY__) || __FINITE_MATH_ONLY__ == 0) && \
      defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && \
      defined(__SSE2__) && defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ == 0
    // Host coefficient/bound assembly must keep its declared environment.
    // Device RN/FTZ/contraction semantics are bound by the build receipt,
    // not inferred from the host floating-point environment or a test macro.
    return NativePairedPreciseCudaBuild &&
        std::is_same_v<amrex::Real,double> && std::is_same_v<amrex::ParticleReal,double> &&
        std::numeric_limits<amrex::Real>::is_iec559 &&
        std::numeric_limits<amrex::Real>::radix==2 &&
        std::numeric_limits<amrex::Real>::digits==53 &&
        std::fegetround()==FE_TONEAREST && (_mm_getcsr() & 0xe040u)==0;
#else
    return false;
#endif
}
} // namespace warpx::darwin
#endif
