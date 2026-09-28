/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ION_ELECTRIC_WORK_REFERENCE_H_
#define WARPX_ION_ELECTRIC_WORK_REFERENCE_H_
#include "ImplicitIonElectricWork.H"
#include <limits>
namespace warpx::thermal {
struct IonElectricWorkReference {
    amrex::Real old_kinetic_energy=0, endpoint_kinetic_energy=0;
    // eps(ParticleReal)*(K_old+K_endpoint), J. A scale reference only:
    // NOT an error bound for Boris, the nonlinear solve, gather or roundoff,
    // and never an energy-conservation or accept/reject criterion.
    amrex::Real momentum_roundoff_reference=0;
    // Cauchy bound on this component's work, using the SAME gathered field
    // and rounded endpoint as ComputeIonElectricWork; J, nonnegative.
    amrex::Real component_cauchy_work_bound=0;
};
// Pure optional acceptance diagnostic. Same SI/endpoint convention and
// nonnegative mass/weight contract as ComputeIonElectricWork. No RNG, state
// mutation, communication or rejection. Stable K=m*u^2/(gamma+1) avoids
// subtracting nearly equal gamma and one. Does not alter the work ledger ABI.
AMREX_GPU_HOST_DEVICE inline IonElectricWorkReference
ComputeIonElectricWorkReference (
    amrex::Real mass, amrex::Real charge, amrex::Real weight, amrex::Real dt,
    IonWorkPoint const& old_u, IonWorkPoint const& midpoint_u,
    IonWorkPoint const& component_e)
{
    amrex::Real s0=0, s1=0, sum_squared=0, field_squared=0;
    for (int d=0;d<3;++d) {
        amrex::ParticleReal const end_u=2*midpoint_u[d]-old_u[d];
        amrex::Real const a=old_u[d], b=end_u;
        s0+=a*a; s1+=b*b;
        sum_squared+=(a+b)*(a+b);
        amrex::Real const e=component_e[d];
        field_squared+=e*e;
    }
    auto const g0=std::sqrt(1+s0/(PhysConst::c*PhysConst::c));
    auto const g1=std::sqrt(1+s1/(PhysConst::c*PhysConst::c));
    IonElectricWorkReference result;
    result.old_kinetic_energy=weight*mass*s0/(g0+1);
    result.endpoint_kinetic_energy=weight*mass*s1/(g1+1);
    result.momentum_roundoff_reference=
        std::numeric_limits<amrex::ParticleReal>::epsilon()*
        (result.old_kinetic_energy+result.endpoint_kinetic_energy);
    result.component_cauchy_work_bound=
        weight*std::abs(charge*dt)*std::sqrt(field_squared)*
        std::sqrt(sum_squared)/(g0+g1);
    return result;
}
} // namespace warpx::thermal
#endif
