/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_BORIS_IMPULSE_H_
#define WARPX_NATIVE_BORIS_IMPULSE_H_
#include "Utils/WarpXConst.H"
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#include <AMReX_REAL.H>
#include <cmath>
namespace warpx::particles {
/** Pure full-Boris reference, with the kick retained before adding it to u_old.
 * This does not mutate the production pusher. full/midpoint/finished reproduce
 * native operations; impulse retains the sum of the two electric half-kicks
 * and magnetic kick. In exact arithmetic finished = old + impulse. In floating
 * point their difference is a REPRESENTATION DEFECT, not an exact identity.
 * No radiation, QED, ionization, suborbit or wall action is included here.
 */
struct NativeBorisImpulse {
    amrex::GpuArray<amrex::ParticleReal,3> full{}, midpoint{}, finished{}, impulse{};
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool FullBorisImpulse(amrex::GpuArray<amrex::ParticleReal,3> const& old,
    amrex::GpuArray<amrex::ParticleReal,3> const& electric,
    amrex::GpuArray<amrex::ParticleReal,3> const& magnetic,
    amrex::ParticleReal charge, amrex::ParticleReal mass, amrex::Real dt,
    NativeBorisImpulse& result) noexcept
{
    using P=amrex::ParticleReal;
    result={};
    if (!(mass>0.) || !(dt>=0.) || !std::isfinite(mass) ||
        !std::isfinite(dt) || !std::isfinite(charge)) { return false; }
    for (int c=0;c<3;++c) {
        if (!std::isfinite(old[c]) || !std::isfinite(electric[c]) ||
            !std::isfinite(magnetic[c])) { return false; }
    }
    P const econst=P(.5)*charge*dt/mass;
    amrex::GpuArray<P,3> kick{}, u{};
    for (int c=0;c<3;++c) { kick[c]=econst*electric[c]; u[c]=old[c]+kick[c]; }
    constexpr auto inv_c2=PhysConst::inv_c2_v<P>;
    P const inv_gamma=P(1.)/std::sqrt(P(1.)+(u[0]*u[0]+u[1]*u[1]+u[2]*u[2])*inv_c2);
    P const tx=econst*inv_gamma*magnetic[0];
    P const ty=econst*inv_gamma*magnetic[1];
    P const tz=econst*inv_gamma*magnetic[2];
    P const tsqi=P(2.)/(P(1.)+tx*tx+ty*ty+tz*tz);
    P const sx=tx*tsqi, sy=ty*tsqi, sz=tz*tsqi;
    P const px=u[0]+u[1]*tz-u[2]*ty;
    P const py=u[1]+u[2]*tx-u[0]*tz;
    P const pz=u[2]+u[0]*ty-u[1]*tx;
    amrex::GpuArray<P,3> const rotation{py*sz-pz*sy,pz*sx-px*sz,px*sy-py*sx};
    for (int c=0;c<3;++c) {
        result.full[c]=(u[c]+rotation[c])+kick[c];
        result.midpoint[c]=P(.5)*(result.full[c]+old[c]);
        result.finished[c]=P(2.)*result.midpoint[c]-old[c];
        result.impulse[c]=(kick[c]+rotation[c])+kick[c];
        if (!std::isfinite(result.full[c]) || !std::isfinite(result.midpoint[c]) ||
            !std::isfinite(result.finished[c]) || !std::isfinite(result.impulse[c])) {
            result={}; return false;
        }
    }
    return true;
}

/** Finite relativistic velocity increment from a correlated momentum impulse.
 * The base uses the instantaneous-current map u/sqrt(1+|u|^2/c^2).
 * gamma1-gamma0 is rationalized; no subtraction of two rounded velocities.
 * The delta describes old+impulse in exact arithmetic, not a promise that the
 * rounded native finisher retained the sub-ulp impulse. All finite cross terms
 * are retained. Caller separately validates the represented-endpoint defect.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool RelativisticVelocityIncrement(amrex::GpuArray<double,3> const& old,
    amrex::GpuArray<double,3> const& impulse, amrex::GpuArray<double,3>& value,
    amrex::GpuArray<double,3>& delta) noexcept
{
    value={}; delta={};
    double square0=0.,square1=0.,increment=0.;
    for (int c=0;c<3;++c) {
        if (!std::isfinite(old[c]) || !std::isfinite(impulse[c])) { return false; }
        double const endpoint=old[c]+impulse[c];
        square0+=old[c]*old[c]; square1+=endpoint*endpoint;
        increment+=(2.*old[c]+impulse[c])*impulse[c];
    }
    double const c2=PhysConst::c*PhysConst::c;
    double const g0=std::sqrt(1.+square0/c2),g1=std::sqrt(1.+square1/c2);
    double const dg=(increment/c2)/(g0+g1);
    for (int c=0;c<3;++c) {
        value[c]=old[c]/g0;
        delta[c]=impulse[c]/g1-old[c]*dg/(g0*g1);
        if (!std::isfinite(value[c]) || !std::isfinite(delta[c])) {
            value={};delta={};return false;
        }
    }
    return true;
}

/** RZ basis/position finite increment from an already captured Cartesian chord.
 * Native old and endpoint radius are used only in well-conditioned sums and
 * denominators. This preserves tiny dr and basis rotations that disappear in
 * subtracting two absolute RZ points. Axis-crossing chords are explicitly unsupported; no wall image is supplied.
 */
struct RZPointIncrement {
    double radius=0., radial_delta=0.;
    amrex::GpuArray<double,2> basis{}, basis_delta{};
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool CylindricalPointIncrement(amrex::GpuArray<double,2> const& old,
    amrex::GpuArray<double,2> const& chord,RZPointIncrement& out) noexcept
{
    out={};
    for(int c=0;c<2;++c)if(!std::isfinite(old[c])||!std::isfinite(chord[c]))return false;
    double const r0=std::sqrt(old[0]*old[0]+old[1]*old[1]);
    double const x1=old[0]+chord[0],y1=old[1]+chord[1];
    double const r1=std::sqrt(x1*x1+y1*y1);
    if(!(r0>0.)||!(r1>0.)||!std::isfinite(r0)||!std::isfinite(r1))return false;
    double const chord2=chord[0]*chord[0]+chord[1]*chord[1];
    double const dot=old[0]*chord[0]+old[1]*chord[1];
    double const cross=old[0]*chord[1]-old[1]*chord[0];
    if(chord2>0. && cross==0. && dot<0. && -dot<chord2)return false;
    out.radius=r0;
    out.radial_delta=((2.*old[0]+chord[0])*chord[0]+(2.*old[1]+chord[1])*chord[1])/(r1+r0);
    for(int c=0;c<2;++c){out.basis[c]=old[c]/r0;out.basis_delta[c]=(chord[c]-out.basis[c]*out.radial_delta)/r1;}
    if(!std::isfinite(out.radial_delta)||!std::isfinite(out.basis_delta[0])||!std::isfinite(out.basis_delta[1])){out={};return false;}
    return true;
}
}
#endif
