/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_JOINT_CURL_ARITHMETIC_H_
#define WARPX_NATIVE_JOINT_CURL_ARITHMETIC_H_
#include "ThermalCurrentRemainder.H"
#include "Utils/WarpXConst.H"
namespace warpx::thermal::joint_arithmetic {
namespace dd=warpx::ohm::compensated;
using Real=amrex::Real;
using Pair=dd::Pair;
struct Operand {
    amrex::Array4<Real const> high,low;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Pair operator()(int i,int j) const noexcept {return {high(i,j,0),low?low(i,j,0):0.};}
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Difference(Pair a,Pair b) noexcept {return dd::Add(a,dd::Negate(b));}
// Native CylindricalYeeAlgorithm coefficients are formed in double, in the
// same order. Only field arithmetic is retained; this is not a nodal Laplacian
// or an idealized metric replacement. Axis rows are the native m=0 rows.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Curl(amrex::GpuArray<Operand,3> const& A,int c,int i,int j,Real dr,Real dz) noexcept {
    Real const ir=1./dr,iz=1./dz;
    if(c==0)return i==0?Pair{0.,0.}:dd::Negate(dd::Multiply(Difference(A[1](i,j+1),A[1](i,j)),iz));
    if(c==1)return Difference(dd::Multiply(Difference(A[0](i,j+1),A[0](i,j)),iz),
                               dd::Multiply(Difference(A[2](i+1,j),A[2](i,j)),ir));
    Real const r=(i+.5)*dr,plus=r+.5*dr,minus=r-.5*dr,coefficient=(1./r)*ir;
    return dd::Multiply(Difference(dd::Multiply(A[1](i+1,j),plus),
                                   dd::Multiply(A[1](i,j),minus)),coefficient);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Ampere(amrex::GpuArray<Operand,3> const& B,int c,int i,int j,Real dr,Real dz) noexcept {
    Real const ir=1./dr,iz=1./dz,mu=1./PhysConst::mu0;
    if(c==0)return dd::Multiply(dd::Negate(dd::Multiply(Difference(B[1](i,j),B[1](i,j-1)),iz)),mu);
    Real const r=i*dr;
    if(c==1)return r>.5*dr?dd::Multiply(Difference(
        dd::Multiply(Difference(B[0](i,j),B[0](i,j-1)),iz),
        dd::Multiply(Difference(B[2](i,j),B[2](i-1,j)),ir)),mu):Pair{0.,0.};
    if(r<=.5*dr)return remainder::Divide(dd::Multiply(B[1](i,j),mu*4.),dr);
    Real const plus=r+.5*dr,minus=r-.5*dr,coefficient=(1./r)*ir;
    return dd::Multiply(dd::Multiply(Difference(dd::Multiply(B[1](i,j),plus),
        dd::Multiply(B[1](i-1,j),minus)),coefficient),mu);
}
}
#endif
