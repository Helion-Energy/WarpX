/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_THETA_REGISTRATION_H_
#define WARPX_NATIVE_STOPPING_THETA_REGISTRATION_H_
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <cmath>
namespace warpx::thermal {
struct NativeStoppingThetaRegistration {
    amrex::Real initial=0.,endpoint=0.,difference_hi=0.,difference_lo=0.,correction=0.;
};
/** Register the correlated theta increment to the actual represented endpoints.
 * This uses the same native Cartesian division and RZ frame order as full
 * instantaneous deposition. It never changes momentum or the correlated map.
 * Caller deposits correction through the native fixed-position theta shape
 * and owns all metric, scatter and image arithmetic remainders.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool ComputeNativeStoppingThetaRegistration (
    amrex::GpuArray<amrex::Real,3> const& position,
    amrex::GpuArray<amrex::Real,3> const& old,
    amrex::GpuArray<amrex::Real,3> const& next,
    amrex::Real correlated,amrex::Real light_speed,
    NativeStoppingThetaRegistration& output) noexcept
{
    using R=amrex::Real;
    namespace pair=warpx::ohm::compensated;
    output={};
    for(int c=0;c<3;++c){
        if(!std::isfinite(position[c])||!std::isfinite(old[c])||!std::isfinite(next[c]))return false;
    }
    if(!std::isfinite(correlated)||!std::isfinite(light_speed)||light_speed<=0.)return false;
    R const xp=position[0],yp=position[1],rr=std::sqrt(xp*xp+yp*yp);
    if(!std::isfinite(rr)||rr<=0.)return false;
    R const g0=std::sqrt(1.+(old[0]*old[0]+old[1]*old[1]+old[2]*old[2])/(light_speed*light_speed));
    R const g1=std::sqrt(1.+(next[0]*next[0]+next[1]*next[1]+next[2]*next[2])/(light_speed*light_speed));
    R const vx0=old[0]/g0,vy0=old[1]/g0,vx1=next[0]/g1,vy1=next[1]/g1;
    R const initial=(-yp*vx0+xp*vy0)/rr,endpoint=(-yp*vx1+xp*vy1)/rr;
    auto const difference=pair::Sum(endpoint,-initial);
    auto const correction=pair::Add(difference,{-correlated,0.});
    R const registered=correction.hi+correction.lo;
    if(!std::isfinite(g0)||!std::isfinite(g1)||!std::isfinite(initial)||
       !std::isfinite(endpoint)||!std::isfinite(difference.hi)||
       !std::isfinite(difference.lo)||!std::isfinite(registered))return false;
    output={initial,endpoint,difference.hi,difference.lo,registered};
    return true;
}
} // namespace warpx::thermal
#endif
