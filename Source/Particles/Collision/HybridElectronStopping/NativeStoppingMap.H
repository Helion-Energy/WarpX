/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_MAP_H_
#define WARPX_NATIVE_STOPPING_MAP_H_
#include "Utils/WarpXConst.H"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuContainers.H>
#include <cmath>

namespace warpx::particles {
/** Actual native ion-on-electron slowing-down rate. T is k_B T_e [J].
 * Eligibility and the 1e-3 eV temperature floor remain caller-owned.
 * The rate uses physical electron mass, independently of inertia-model mass.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real NativeStoppingRate (amrex::Real n_e, amrex::Real T_e,
    amrex::ParticleReal species_charge, amrex::ParticleReal species_mass,
    amrex::ParticleReal clog)
{
    using namespace amrex::literals;
    amrex::Real constexpr pi = MathConst::pi;
    amrex::Real constexpr ep0 = PhysConst::epsilon_0;
    amrex::Real constexpr q_e2 = PhysConst::q_e * PhysConst::q_e;
    amrex::Real constexpr ep02 = ep0 * ep0;
    amrex::Real const pi32 = pi * std::sqrt(pi);
    amrex::Real const q2 = species_charge * species_charge;
    amrex::Real const T32 = T_e * std::sqrt(T_e);
    return std::sqrt(2._rt) * n_e * q2 * q_e2 * std::sqrt(PhysConst::m_e) * clog
        / (12._rt * pi32 * ep02 * species_mass * T32);
}
struct NativeStoppingKickResult {
    amrex::GpuArray<amrex::ParticleReal,3> proper{};
    amrex::Real weighted_lab_loss = 0.;
};
/** Unchanged native proper->physical exponential velocity map. All components
 * share the caller's Cartesian/collision frame. No RNG, eligibility or clamp.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
NativeStoppingKickResult NativeStoppingKick (
    amrex::GpuArray<amrex::ParticleReal,3> u,
    amrex::GpuArray<amrex::ParticleReal,3> electron_velocity,
    amrex::Real nu_s, amrex::Real dt, amrex::ParticleReal weight,
    amrex::ParticleReal species_mass)
{
    using namespace amrex::literals;
    amrex::ParticleReal constexpr inv_c2 = 1._prt / (PhysConst::c * PhysConst::c);
    amrex::ParticleReal const u2b = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    amrex::ParticleReal const gb = std::sqrt(1._prt + u2b*inv_c2);
    amrex::ParticleReal const fac = std::exp(-nu_s*dt);
    amrex::ParticleReal const vxn = electron_velocity[0] + (u[0]/gb-electron_velocity[0])*fac;
    amrex::ParticleReal const vyn = electron_velocity[1] + (u[1]/gb-electron_velocity[1])*fac;
    amrex::ParticleReal const vzn = electron_velocity[2] + (u[2]/gb-electron_velocity[2])*fac;
    amrex::ParticleReal const v2n = vxn*vxn + vyn*vyn + vzn*vzn;
    amrex::ParticleReal const ga = 1._prt/std::sqrt(1._prt - v2n*inv_c2);
    NativeStoppingKickResult result;
    result.proper = {ga*vxn,ga*vyn,ga*vzn};
    amrex::ParticleReal const u2a = ga*ga*v2n;
    result.weighted_lab_loss = weight * species_mass
        * (u2b/(gb+1._prt) - u2a/(ga+1._prt));
    return result;
}
/** Finite proper-velocity increment for the same exponential stopping map.
 * The caller supplies one common collision frame and owns the same finite,
 * subluminal and proper-speed admission checks as NativeStoppingKick.
 * Form the small change without subtracting full velocities or gammas.
 * Callers must materialize their particle once and use that represented state
 * for reaction, current and work; this increment is not a separate reaction.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::ParticleReal,3> NativeStoppingProperIncrement (
    amrex::GpuArray<amrex::ParticleReal,3> u,
    amrex::GpuArray<amrex::ParticleReal,3> electron_velocity,
    amrex::Real nu_s, amrex::Real dt)
{
    using namespace amrex::literals;
    if (nu_s == 0._rt || dt == 0._rt) { return {}; }
    amrex::ParticleReal constexpr inv_c2 = 1._prt / (PhysConst::c * PhysConst::c);
    amrex::ParticleReal const g0 = std::sqrt(1._prt +
        (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*inv_c2);
    amrex::ParticleReal const f = std::expm1(-nu_s*dt);
    amrex::GpuArray<amrex::ParticleReal,3> v{}, dv{};
    for (int c=0; c<3; ++c) {
        v[c] = u[c]/g0;
        dv[c] = (v[c]-electron_velocity[c])*f;
    }
    amrex::ParticleReal const q = (dv[0]*(2._prt*v[0]+dv[0]) +
        dv[1]*(2._prt*v[1]+dv[1]) + dv[2]*(2._prt*v[2]+dv[2]))*inv_c2;
    amrex::ParticleReal const a = g0*g0*q;
    amrex::ParticleReal const s = std::sqrt(1._prt-a);
    amrex::ParticleReal const dg = g0*a/(s*(1._prt+s));
    amrex::GpuArray<amrex::ParticleReal,3> result{};
    for (int c=0; c<3; ++c) {
        result[c] = g0*dv[c] + dg*(v[c]+dv[c]);
    }
    return result;
}
} // namespace warpx::particles
#endif
