/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_AXIAL_CURRENT_INCREMENT_H_
#define WARPX_ESIRKEPOV_AXIAL_CURRENT_INCREMENT_H_
#include "EsirkepovInstantaneousCurrentIncrement.H"

namespace warpx::particles {
// A separate private RZ/cubic extension. The ordinary same-cell primitive is
// unchanged. Its caller owns a fresh axial PMC path/produced-support lease.
struct CubicAxialPointSupport {
    int radial_center=0,radial_first=0;
    int axial_center=0,axial_first=0,axial_last=0;
    int knot=0,direction=0;
    double tail=0.;
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool OneAxialKnotSupport(EsirkepovInstantaneousPointIncrement const& p,
                        int cap_knot,CubicAxialPointSupport& out) noexcept
{
    out={};
    if(!instantaneous_increment_detail::CommonPointSupport<3>(
            p.position[0],out.radial_center,out.radial_first) ||
       !instantaneous_increment_detail::FinitePair(p.position[2]))return false;
    auto const& z=p.position[2];
    double const end=z.value+z.delta;
    double const upper=double(std::numeric_limits<int>::max()-8);
    if(z.value<0.||z.value>upper||end<0.||end>upper)return false;
    int const c0=int(z.value),c1=int(end);
    int const direction=c1>c0?1:-1;
    if((direction>0&&c1!=c0+1)||(direction<0&&c0!=c1+1))return false;
    int const knot=direction>0?c1:c0;
    if(knot!=cap_knot)return false;
    double const q[2]{z.value,end};int const c[2]{c0,c1};
    for(int n=0;n<2;++n){
        double const guard=64.*std::numeric_limits<double>::epsilon()*(1.+std::abs(q[n]));
        if(!(q[n]>double(c[n])+guard&&q[n]<double(c[n])+1.-guard))return false;
    }
    // Keep the retained distance beyond the knot. The rounded endpoint above
    // is used only for the integer/branch decision, never for this difference.
    double const tail=z.delta-(double(knot)-z.value);
    if(!std::isfinite(tail)||!(direction*tail>0.&&direction*tail<1.))return false;
    out.axial_center=c0;out.axial_first=(c0<c1?c0:c1)-1;
    out.axial_last=(c0>c1?c0:c1)+2;
    out.knot=knot;out.direction=direction;out.tail=tail;
    return true;
}

// Infer the adjacent axial knot only after the same bounded conversion
// preflight. The caller separately certifies an interior knot or the actual
// physical cap and retains all produced/allocation support requirements.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool OneAxialKnotSupport(EsirkepovInstantaneousPointIncrement const& p,
                        CubicAxialPointSupport& out) noexcept
{
    out={};auto const& z=p.position[2];
    if(!instantaneous_increment_detail::FinitePair(z))return false;
    double const end=z.value+z.delta;
    double const upper=double(std::numeric_limits<int>::max()-8);
    if(z.value<0.||z.value>upper||end<0.||end>upper)return false;
    int const c0=int(z.value),c1=int(end);
    return OneAxialKnotSupport(p,c0>c1?c0:c1,out);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool EsirkepovAxialInstantaneousCurrentPair(
    EsirkepovInstantaneousPointIncrement const& p,int cap_knot,
    amrex::GpuArray<int,3> const& node,
    amrex::GpuArray<EsirkepovIncrement,3>& current) noexcept
{
    current={};CubicAxialPointSupport support;
    if(!OneAxialKnotSupport(p,cap_knot,support) ||
       !instantaneous_increment_detail::FinitePair(p.velocity[0]) ||
       !instantaneous_increment_detail::FinitePair(p.velocity[2]) ||
       !instantaneous_increment_detail::FinitePair(p.azimuthal_velocity))return false;
    int const first[3]{support.radial_first,0,support.axial_center-1};
    int const center[3]{support.radial_center,0,support.axial_center};
    amrex::GpuArray<int,3> safe{};
    EsirkepovIncrement nodal[3]{{1.,0.},{1.,0.},{1.,0.}},face[3]{};
    for(int d=0;d<3;d+=2){
        safe[d]=node[d]<first[d]?first[d]-1:
                node[d]>first[d]+3?first[d]+4:node[d];
        int const slot=safe[d]-first[d];
        auto const local=p.position[d]-double(center[d]);
        nodal[d]=esirkepov_increment_detail::Shape<3>(slot,local);
        face[d]=esirkepov_increment_detail::Shape<2>(slot,local-.5);
    }
    // Cardinal derivative jumps supply precisely the newly active polynomial.
    // Outside old support its old polynomial is zero. No absolute endpoint
    // current/shape is subtracted, including for entering and leaving rows.
    long long const s3=static_cast<long long>(support.knot)-node[2]+2;
    long long const s2=static_cast<long long>(support.knot)-node[2]+1;
    constexpr int c3[5]{1,-4,6,-4,1},c2[4]{1,-3,3,-1};
    double const t=support.tail;
    if(s3>=0&&s3<5)nodal[2].delta+=support.direction*c3[s3]*((t*t)*t/6.);
    if(s2>=0&&s2<4)face[2].delta+=support.direction*c2[s2]*(t*t/2.);
    // A row that leaves the new support has an exactly zero new shape.
    // Use that structural identity, avoiding cancellation between an
    // extrapolated polynomial and its knot correction when the old shape is
    // tiny. Entering rows already use only the retained tail power above.
    int const new_first=support.axial_center+support.direction-1;
    if(node[2]<new_first||node[2]>new_first+3)nodal[2].delta=-nodal[2].value;
    if(node[2]<new_first||node[2]>new_first+2)face[2].delta=-face[2].value;
    amrex::GpuArray<double,3> x{p.position[0].value,0.,p.position[2].value};
    amrex::GpuArray<double,3> v{p.velocity[0].value,0.,p.velocity[2].value};
    amrex::GpuArray<double,3> zero{},base{},rate{};
    if(!EsirkepovCurrentRate<3,2>(x,v,zero,safe,p.azimuthal_velocity.value,0.,base,rate))return false;
    amrex::GpuArray<EsirkepovIncrement,3> candidate{
        face[0]*p.velocity[0]*nodal[2],
        p.azimuthal_velocity*nodal[0]*nodal[2],
        face[2]*p.velocity[2]*nodal[0]};
    for(int c=0;c<3;++c){
        candidate[c].value=base[c];
        if(!std::isfinite(candidate[c].value)||!std::isfinite(candidate[c].delta)||
           !std::isfinite(candidate[c].value+candidate[c].delta))return false;
    }
    current=candidate;return true;
}
} // namespace warpx::particles
#endif
