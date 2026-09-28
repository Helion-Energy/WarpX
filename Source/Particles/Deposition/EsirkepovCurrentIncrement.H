/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_CURRENT_INCREMENT_H_
#define WARPX_ESIRKEPOV_CURRENT_INCREMENT_H_
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#include <AMReX_Array.H>
#include <cmath>
#include <limits>

namespace warpx::particles {
/** A value and a finite increment, not an infinitesimal derivative.
 * Product includes da*db, so this preserves nonlinear finite-difference algebra.
 * Unlike evaluating b-a after two complete deposits, every term contains an
 * input increment. No field/current or accepted-state mutation occurs here.
 */
struct EsirkepovIncrement {
    double value=0.,delta=0.;
    AMREX_GPU_HOST_DEVICE constexpr EsirkepovIncrement()=default;
    AMREX_GPU_HOST_DEVICE constexpr EsirkepovIncrement(double v,double d=0.):value(v),delta(d){}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE friend EsirkepovIncrement operator+(EsirkepovIncrement a,EsirkepovIncrement b) noexcept {return {a.value+b.value,a.delta+b.delta};}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE friend EsirkepovIncrement operator-(EsirkepovIncrement a,EsirkepovIncrement b) noexcept {return {a.value-b.value,a.delta-b.delta};}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE friend EsirkepovIncrement operator-(EsirkepovIncrement a) noexcept {return {-a.value,-a.delta};}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE friend EsirkepovIncrement operator*(EsirkepovIncrement a,EsirkepovIncrement b) noexcept {return {a.value*b.value,a.value*b.delta+a.delta*b.value+a.delta*b.delta};}
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE friend EsirkepovIncrement operator/(EsirkepovIncrement a,double b) noexcept {return {a.value/b,a.delta/b};}
};

/** Grid-coordinate orbit and a correlated second orbit at the same interval h.
 * chord is the FULL physical interval chord, captured before adding an absolute
 * position. old.delta and chord.delta must likewise come from a correlated
 * displacement/increment, not a subtraction of already rounded positions.
 * For RZ the caller supplies the radial chord derived from the Cartesian chord,
 * and z; y is inactive. transverse_velocity is the native azimuthal velocity.
 * This is an m=0 two-/three-dimensional Esirkepov algebra helper. No new current
 * model, position reconstruction, RZ volume division, or boundary image is used.
 */
struct EsirkepovOrbitIncrement {
    amrex::GpuArray<EsirkepovIncrement,3> old{},chord{};
    EsirkepovIncrement transverse_velocity{};
    double dt=0.;
};

namespace esirkepov_increment_detail {
template<int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EsirkepovIncrement Shape(int slot,EsirkepovIncrement a) noexcept {
    if(slot<0||slot>N)return {};
    if constexpr(N==1)return slot==0?1.-a:a;
    if constexpr(N==2) {
        if(slot==0)return (.5-a)*(.5-a)*.5;
        if(slot==1)return .75-a*a;
        return (.5+a)*(.5+a)*.5;
    }
    if constexpr(N==3) {
        if(slot==0)return (1.-a)*(1.-a)*(1.-a)/6.;
        if(slot==1)return 2./3.-a*a*(1.-a/2.);
        if(slot==2)return 2./3.-(1.-a)*(1.-a)*(1.-(1.-a)/2.);
        return a*a*a/6.;
    }
    if constexpr(N==4) {
        if(slot==0)return (.5-a)*(.5-a)*(.5-a)*(.5-a)/24.;
        if(slot==1)return (4.75-11.*a+4.*a*a*(1.5+a-a*a))/24.;
        if(slot==2)return (14.375+6.*a*a*(a*a-2.5))/24.;
        if(slot==3)return (4.75+11.*a+4.*a*a*(1.5-a-a*a))/24.;
        return (.5+a)*(.5+a)*(.5+a)*(.5+a)/24.;
    }
    return {};
}
template<int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EsirkepovIncrement Difference(int slot,EsirkepovIncrement a,EsirkepovIncrement d) noexcept {
    if(slot<0||slot>N)return {};
    auto const b=a+d,s=a+b,q=a*a+a*b+b*b;
    if constexpr(N==1)return slot==0?d:-d;
    if constexpr(N==2) {
        if(slot==0)return d*(.5-.5*s);
        if(slot==1)return d*s;
        return -d*(.5+.5*s);
    }
    if constexpr(N==3) {
        if(slot==0){auto const u=1.-a,v=1.-b;return d*(u*u+u*v+v*v)/6.;}
        if(slot==1)return d*(s-.5*q);
        if(slot==2)return -d*(.5+.5*s-.5*q);
        return -d*q/6.;
    }
    if constexpr(N==4) {
        auto const t=(a+b)*(a*a+b*b);
        if(slot==0||slot==4){auto const u=slot==0?.5-a:.5+a,v=slot==0?.5-b:.5+b;return (slot==0?d:-d)*(u+v)*(u*u+v*v)/24.;}
        if(slot==1)return d*(11.-6.*s-4.*q+4.*t)/24.;
        if(slot==2)return d*(15.*s-6.*t)/24.;
        return d*(-11.-6.*s+4.*q+4.*t)/24.;
    }
    return {};
}
// This first adapter explicitly rejects a knot crossing in either orbit or
// between orbits. Callers must use the maintained deposit for that case. The
// conservative epsilon margin also rejects an unresolved support decision.
template<int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CommonSupport(EsirkepovIncrement a,EsirkepovIncrement d,int& first) noexcept {
    double const end=a.value+d.value;
    double const values[4]={a.value,end,a.value+a.delta,end+a.delta+d.delta};
    if(!(a.value>=0.) || !std::isfinite(a.value))return false;
    int const center=static_cast<int>(a.value+(N%2==0?.5:0.));first=center-N/2;
    double const lower=double(center)-(N%2==0?.5:0.);
    for(double x:values) {
        double const guard=64.*std::numeric_limits<double>::epsilon()*(1.+std::abs(x));
        if(!std::isfinite(x)||!(x>lower+guard&&x<lower+1.-guard))return false;
    }
    return true;
}
}

/** Native unweighted grid current at a face, and its finite increment.
 * Dim=2 corresponds to (r,z) / (x,z), native components0,2; Dim=3 is Cartesian.
 * Caller applies qw*transverse inverse cell areas and native RZ volume/mode
 * handling exactly as in the maintained deposit. Returns false for unsupported
 * support changes, nonpositive/nonfinite dt; outputs are then zero. A fixed
 * interval is deliberate: no implicit variable-dt/history policy is introduced.
 */
template<int N,int Dim> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool EsirkepovCurrentPair(EsirkepovOrbitIncrement const& orbit,
    amrex::GpuArray<int,3> const& node,amrex::GpuArray<EsirkepovIncrement,3>& current) noexcept
{
    static_assert(N>=1&&N<=4);static_assert(Dim==2||Dim==3);current={};
    if(!(orbit.dt>0.)||!std::isfinite(orbit.dt))return false;
    int first[3]{};EsirkepovIncrement old[3]{},now[3]{},difference[3]{};
    for(int d=0;d<3;++d) {
        if constexpr(Dim==2){if(d==1)continue;}
        if(!esirkepov_increment_detail::CommonSupport<N>(orbit.old[d],orbit.chord[d],first[d]))return false;
        auto const a=orbit.old[d]-double(first[d]+N/2);
        int const slot=node[d]-first[d];
        old[d]=esirkepov_increment_detail::Shape<N>(slot,a);
        now[d]=esirkepov_increment_detail::Shape<N>(slot,a+orbit.chord[d]);
        for(int s=0;s<=N&&s<=slot;++s)difference[d]=difference[d]+esirkepov_increment_detail::Difference<N>(s,a,orbit.chord[d]);
    }
    if constexpr(Dim==3) {
        for(int d=0;d<3;++d){int const e=(d+1)%3,f=(d+2)%3;current[d]=difference[d]/orbit.dt*((now[e]*now[f]+old[e]*old[f])/3.+(now[e]*old[f]+old[e]*now[f])/6.);}
    } else {
        current[0]=difference[0]/orbit.dt*(now[2]+old[2])*.5;
        current[2]=difference[2]/orbit.dt*(now[0]+old[0])*.5;
        current[1]=orbit.transverse_velocity*((now[0]*now[2]+old[0]*old[2])/3.+(now[0]*old[2]+old[0]*now[2])/6.);
    }
    return true;
}
}
#endif
