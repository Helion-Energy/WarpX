/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_CURRENT_RATE_H_
#define WARPX_ESIRKEPOV_CURRENT_RATE_H_
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>
#include <limits>
namespace warpx::particles {
namespace instantaneous_esirkepov {
// A true second derivative jet. Unlike a finite increment, its product
// deliberately has no products of third/higher infinitesimal order.
struct Jet {
    double value=0.,first=0.,second=0.;
    AMREX_GPU_HOST_DEVICE constexpr Jet()=default;
    AMREX_GPU_HOST_DEVICE constexpr Jet(double v,double d=0.,double dd=0.):value(v),first(d),second(dd){}
    AMREX_GPU_HOST_DEVICE friend Jet operator+(Jet a,Jet b) noexcept {return {a.value+b.value,a.first+b.first,a.second+b.second};}
    AMREX_GPU_HOST_DEVICE friend Jet operator-(Jet a,Jet b) noexcept {return {a.value-b.value,a.first-b.first,a.second-b.second};}
    AMREX_GPU_HOST_DEVICE friend Jet operator-(Jet a) noexcept {return {-a.value,-a.first,-a.second};}
    AMREX_GPU_HOST_DEVICE friend Jet operator*(Jet a,Jet b) noexcept {return {a.value*b.value,a.first*b.value+a.value*b.first,a.second*b.value+2.*a.first*b.first+a.value*b.second};}
    AMREX_GPU_HOST_DEVICE friend Jet operator/(Jet a,double b) noexcept {return {a.value/b,a.first/b,a.second/b};}
};
template<int N> AMREX_GPU_HOST_DEVICE Jet Shape(int s,Jet x) noexcept {
    if(s<0||s>N)return {};
    if constexpr(N==1)return s==0?1.-x:x;
    if constexpr(N==2){if(s==0)return (.5-x)*(.5-x)/2.;if(s==1)return .75-x*x;return (.5+x)*(.5+x)/2.;}
    if constexpr(N==3){if(s==0)return (1.-x)*(1.-x)*(1.-x)/6.;if(s==1)return 2./3.-x*x*(1.-x/2.);if(s==2)return 2./3.-(1.-x)*(1.-x)*(1.-(1.-x)/2.);return x*x*x/6.;}
    if constexpr(N==4){if(s==0)return (.5-x)*(.5-x)*(.5-x)*(.5-x)/24.;if(s==1)return (4.75-11.*x+4.*x*x*(1.5+x-x*x))/24.;if(s==2)return (14.375+6.*x*x*(x*x-2.5))/24.;if(s==3)return (4.75+11.*x+4.*x*x*(1.5-x-x*x))/24.;return (.5+x)*(.5+x)*(.5+x)*(.5+x)/24.;}
    return {};
}
}
/** Exact h->0 current and its physical time derivative of the native
 * Esirkepov straight-chord polynomial. x,v,a are position, velocity and
 * acceleration in grid coordinates; no timestep or rounded-position
 * subtraction enters this operation. For Dim2, d=0,2 are active and the
 * out-of-plane velocity/acceleration are in physical m/s and m/s^2.
 * Caller owns q*w*inverse transverse area (0,2), inverse volume(theta),
 * cylindrical velocity/basis derivatives and native volume/BC/sum operations.
 * At a spline knot this initial adapter rejects, since order1/2 derivatives
 * can depend on the side. No physical orbit or history is changed here.
 */
template<int N,int Dim> AMREX_GPU_HOST_DEVICE bool EsirkepovCurrentRate(
    amrex::GpuArray<double,3> const& x,amrex::GpuArray<double,3> const& v,
    amrex::GpuArray<double,3> const& acceleration,amrex::GpuArray<int,3> const& node,
    double transverse_velocity,double transverse_acceleration,
    amrex::GpuArray<double,3>& current,amrex::GpuArray<double,3>& rate) noexcept
{
    static_assert(N>=1&&N<=4);static_assert(Dim==2||Dim==3);
    current={};rate={};double s[3]{1.,1.,1.},sd[3]{},prefix[3]{},prefix_d[3]{};
    for(int d=0;d<3;++d){
        if constexpr(Dim==2){if(d==1)continue;}
        if(!std::isfinite(x[d])||!std::isfinite(v[d])||!std::isfinite(acceleration[d])||x[d]<0.||x[d]>double(std::numeric_limits<int>::max()-8))return false;
        int const center=int(x[d]+(N%2==0?.5:0.)),first=center-N/2;
        double const local=x[d]-center,lower=N%2==0?-.5:0.;
        double const margin=64.*std::numeric_limits<double>::epsilon()*(1.+std::abs(x[d]));
        if(!(local>lower+margin&&local<lower+1.-margin))return false;
        int const slot=node[d]-first;
        auto const selected=instantaneous_esirkepov::Shape<N>(slot,{local,1.,0.});
        s[d]=selected.value;sd[d]=selected.first*v[d];
        for(int k=0;k<N&&k<=slot;++k){auto const q=instantaneous_esirkepov::Shape<N>(k,{local,1.,0.});prefix[d]-=q.first;prefix_d[d]-=q.second;}
        // Beyond the complete shape support its prefix is identically zero.
        if(slot>=N){prefix[d]=0.;prefix_d[d]=0.;}
    }
    for(int d=0;d<3;++d){
        if constexpr(Dim==2){if(d==1){current[1]=transverse_velocity*s[0]*s[2];rate[1]=transverse_acceleration*s[0]*s[2]+transverse_velocity*(sd[0]*s[2]+s[0]*sd[2]);continue;}}
        int const e=(d+1)%3,f=(d+2)%3;
        current[d]=prefix[d]*v[d]*s[e]*s[f];
        rate[d]=(prefix[d]*acceleration[d]+prefix_d[d]*v[d]*v[d])*s[e]*s[f]+prefix[d]*v[d]*(sd[e]*s[f]+s[e]*sd[f]);
    }
    return true;
}
}
#endif
