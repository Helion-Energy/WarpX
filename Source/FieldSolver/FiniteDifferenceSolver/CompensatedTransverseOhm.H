/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_COMPENSATED_TRANSVERSE_OHM_H_
#define WARPX_COMPENSATED_TRANSVERSE_OHM_H_
#include <AMReX_Array4.H>
#include <AMReX_REAL.H>
#include <cmath>
namespace warpx::ohm::compensated {
using Real=amrex::Real;
struct Pair { Real hi; Real lo; };
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Sum(Real a,Real b) noexcept {
    Real const s=a+b, v=s-a;
    return {s,(a-(s-v))+(b-v)};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Add(Pair a,Pair b) noexcept {
    Pair const s=Sum(a.hi,b.hi);
    return Sum(s.hi,(s.lo+a.lo)+b.lo);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Negate(Pair a) noexcept { return {-a.hi,-a.lo}; }
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Multiply(Pair a,Real b) noexcept {
    Real const p=a.hi*b;
    return Sum(p,std::fma(a.hi,b,-p)+a.lo*b);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Average(amrex::Array4<Real const> const& f,int i,int j,int k,
             int component,int axis) noexcept {
    if(axis<0) return {f(i,j,k,component),0.};
    return Multiply(Sum(f(i,j,k,component),
        f(i+int(axis==0),j+int(axis==1),k+int(axis==2),component)),.5);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Gradient(amrex::Array4<Real const> const& f,int i,int j,int k,
              int axis,Real inverse_cell) noexcept {
    if(axis<0) return {0.,0.};
    return Multiply(Sum(f(i+int(axis==0),j+int(axis==1),k+int(axis==2)),
                        -f(i,j,k)),inverse_cell);
}
// The native floor/gate branch is already selected by the caller. Keep its
// represented high part and carry the active average's otherwise lost part.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair WithHigh(Pair value,Real represented) noexcept {
    return {represented,(value.hi-represented)+value.lo};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real OffsetQuotient(Pair hall,Pair gradient,Pair density,
                    Real gate,Real longitudinal) noexcept {
    if(gate==0.) return -longitudinal;
    Pair const force=Multiply(Add(hall,Negate(gradient)),gate);
    Pair const numerator=Add(force,Negate(Multiply(density,longitudinal)));
    Real const quotient=numerator.hi/density.hi;
    Real const remainder=std::fma(-quotient,density.hi,numerator.hi)
        +(numerator.lo-quotient*density.lo);
    return quotient+remainder/density.hi;
}
}
#endif
