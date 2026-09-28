/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_CURRENT_REMAINDER_H_
#define WARPX_THERMAL_CURRENT_REMAINDER_H_
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <cstdint>
#include <vector>
#include <utility>
#include <cfenv>
#include <limits>
#include <type_traits>
#if defined(__x86_64__) && defined(__SSE2__) && (!defined(AMREX_USE_GPU) || defined(WARPX_PRIVATE_TEST_THERMAL_FINITE_CUDA))
#include <xmmintrin.h>
#endif
namespace warpx::thermal::remainder {
using namespace warpx::ohm::compensated;
// The first remainder capability is the qualified GNU/SSE2 CPU arithmetic.
// In particular __FAST_MATH__ alone does not identify GCC's independent unsafe
// or reciprocal transformations. CUDA/HIP option state is not reliably exposed
// by host preprocessing, so a successful GPU compile is not admission.
// Keep internal linkage: the predicate must describe each consuming TU's flags.
[[maybe_unused]] static constexpr bool CompiledArithmeticSupported() noexcept {
#if defined(WARPX_PRIVATE_TEST_THERMAL_FINITE_CUDA)
    // Private qualification scope. Every consumer is compiled with the pinned
    // precise CUDA flags; this macro is never defined by a public build.
#if !defined(AMREX_USE_CUDA) || !defined(__CUDACC__) || defined(__HIPCC__) || \
    defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    return false;
#else
    return std::is_same_v<Real,double> && std::numeric_limits<Real>::is_iec559 &&
        std::numeric_limits<Real>::radix==2 && std::numeric_limits<Real>::digits==53;
#endif
#else
#if defined(AMREX_USE_GPU) || defined(__CUDACC__) || defined(__HIPCC__) || \
    defined(SYCL_LANGUAGE_VERSION) || defined(__FAST_MATH__) || \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0) || \
    !defined(__GNUC__) || defined(__clang__) || !defined(__GCC_IEC_559) || \
    __GCC_IEC_559 < 2 || !defined(__x86_64__) || !defined(__SSE2_MATH__) || \
    !defined(__FLT_EVAL_METHOD__) || __FLT_EVAL_METHOD__ != 0 || \
    defined(__FMA__) || defined(__FMA4__) || defined(__FP_FAST_FMA) || \
    defined(__AVX512F__)
    return false;
#else
    // With no FMA target, GNU's optional implicit contraction has no scalar
    // multiply-add instruction to use. Explicit std::fma remains correctly
    // rounded; it must not be replaced by an unqualified device/fast path.
    return std::is_same_v<Real,double> && std::numeric_limits<Real>::is_iec559 &&
        std::numeric_limits<Real>::radix==2 && std::numeric_limits<Real>::digits==53;
#endif
#endif
}
[[maybe_unused]] static inline bool ArithmeticSupported() noexcept {
#if defined(__x86_64__) && defined(__SSE2__) && (!defined(AMREX_USE_GPU) || defined(WARPX_PRIVATE_TEST_THERMAL_FINITE_CUDA))
    // Reject changed rounding and flush/denormals-are-zero modes without
    // changing a caller's floating-point environment.
    return CompiledArithmeticSupported() && std::fegetround()==FE_TONEAREST &&
        (_mm_getcsr() & 0xe040u)==0;
#else
    return false;
#endif
}
inline bool All(bool p) { int valid=p; amrex::ParallelDescriptor::ReduceIntMin(valid); return valid!=0; }
inline bool Layout(amrex::MultiFab const& a,amrex::MultiFab const& b,int grow=0) {
    return a.boxArray()==b.boxArray() && a.DistributionMap()==b.DistributionMap() &&
        a.nComp()==b.nComp() && a.nGrowVect().allGE(amrex::IntVect(grow)) && !a.hasEBFabFactory();
}
// Local storage test, followed by the caller's all-rank preflight.
inline bool Overlap(amrex::MultiFab const& a,amrex::MultiFab const& b) {
    std::vector<std::pair<std::uintptr_t,std::uintptr_t>> ranges;
    for(amrex::MFIter i(a);i.isValid();++i) {
        auto const lo=reinterpret_cast<std::uintptr_t>(a[i].dataPtr());
        ranges.emplace_back(lo,lo+a[i].size()*sizeof(Real));
    }
    for(amrex::MFIter j(b);j.isValid();++j) {
        auto const lo=reinterpret_cast<std::uintptr_t>(b[j].dataPtr());
        auto const hi=lo+b[j].size()*sizeof(Real);
        for(auto const& range:ranges) if(range.first<hi && lo<range.second) return true;
    }
    return false;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Divide(Pair a,Real b) noexcept {
    Real const q=a.hi/b;
    return Sum(q,(std::fma(-q,b,a.hi)+a.lo)/b);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Product(Pair a,Pair b) noexcept {
    return Add(Multiply(a,b.hi),Multiply(a,b.lo));
}
inline bool Difference(amrex::MultiFab& out,amrex::MultiFab const& p,
    amrex::MultiFab const& pl,amrex::MultiFab const& n,amrex::MultiFab const& nl,Real factor)
{
    bool valid=ArithmeticSupported() && std::isfinite(factor) && Layout(out,p) && Layout(pl,p) && Layout(n,p) && Layout(nl,p);
    for(auto const* input:{&p,&pl,&n,&nl}) valid=valid && !Overlap(out,*input);
    if(!All(valid)) return false;
    for(amrex::MFIter mfi(out);mfi.isValid();++mfi) {
        auto const a=p.const_array(mfi),al=pl.const_array(mfi),b=n.const_array(mfi),bl=nl.const_array(mfi);
        auto const o=out.array(mfi);
        amrex::ParallelFor(mfi.validbox(),out.nComp(),[=] AMREX_GPU_DEVICE(int i,int j,int k,int c) {
            o(i,j,k,c)=Multiply(Add({a(i,j,k,c),al(i,j,k,c)},Negate({b(i,j,k,c),bl(i,j,k,c)})),factor).hi;
        });
    }
    return out.is_finite(0,out.nComp(),0);
}
}
#endif
