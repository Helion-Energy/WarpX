/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_ION_ELECTRIC_WORK_H_
#define WARPX_IMPLICIT_ION_ELECTRIC_WORK_H_
#include "Utils/WarpXConst.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
class WarpX;
namespace warpx::thermal {
enum class IonElectricGatherContract {
    Unspecified,
    // Actual final Boris gather coordinates, AFTER any native crop.
    RecordedFinalGather,
    // Native midpoint is updated AFTER the last gather; this reconstruction
    // must not be labelled an exact force-work receipt.
    FinalMidpointReconstruction
};
enum class NativeIonWorkContract {
    Unspecified,
    // Caller attests all charged species fully push/gather with Boris, without
    // suborbits, radiation or extra momentum changes. The recorded bridge also
    // validates actual native species flags; the reconstruction path relies on
    // this attestation and does not qualify cropped gathers.
    FullBorisGatherWithoutAdditionalMomentumChanges
};
struct IonElectricWorkComponent {
    enum : int {
        ComponentWork, ComponentWorkAbs, ImpulseX, ImpulseY, ImpulseZ,
        KineticChange, KineticChangeAbs, TotalFieldWork, TotalFieldWorkAbs,
        TotalWorkDefect, TotalWorkDefectAbs, Weight, NonrelativisticWork,
        NonrelativisticKineticChange, Count
    };
};
using IonElectricWorkLedger = std::array<amrex::Real, IonElectricWorkComponent::Count>;
struct IonElectricWorkDiagnostic {
    enum : int {
        OldKineticEnergy, EndpointKineticEnergy, MomentumRoundoffReference,
        ComponentCauchyWorkBound, RemainderWork, RemainderWorkAbs,
        NonrelativisticRemainderWork, RemainderCauchyWorkBound, Count
    };
};
using IonElectricWorkDiagnostics = std::array<amrex::Real, IonElectricWorkDiagnostic::Count>;
using IonWorkPoint = amrex::GpuArray<amrex::ParticleReal, 3>;

// SI: proper u [m/s], electric field [V/m], charge [C], mass [kg], dt [s],
// macro weight [physical particles]. Extensive work [J], Cartesian impulse
// [kg m/s]. Boris's norm-preserving magnetic rotation gives the exact total
// electric work identity; total_e must include any particle-external E.
AMREX_GPU_HOST_DEVICE inline IonElectricWorkLedger
ComputeIonElectricWork (amrex::Real mass, amrex::Real charge, amrex::Real weight,
                        amrex::Real dt, IonWorkPoint const& old_u,
                        IonWorkPoint const& midpoint_u,
                        IonWorkPoint const& component_e,
                        IonWorkPoint const& total_e, bool has_total)
{
    using C = IonElectricWorkComponent;
    IonElectricWorkLedger out{};
    amrex::Real s0=0, s1=0, delta_s=0, dot_component=0, dot_total=0;
    for (int d=0;d<3;++d) {
        // Match native endpoint extrapolation in ParticleReal first.
        amrex::ParticleReal const end_u=2*midpoint_u[d]-old_u[d];
        amrex::Real const a=old_u[d], b=end_u;
        s0+=a*a; s1+=b*b;
        delta_s+=(b-a)*(b+a);
        dot_component+=component_e[d]*(b+a);
        dot_total+=total_e[d]*(b+a);
        out[C::ImpulseX+d]=weight*charge*dt*component_e[d];
    }
    auto const gamma_sum=std::sqrt(1+s0/(PhysConst::c*PhysConst::c))+
                         std::sqrt(1+s1/(PhysConst::c*PhysConst::c));
    out[C::ComponentWork]=weight*charge*dt*dot_component/gamma_sum;
    out[C::ComponentWorkAbs]=std::abs(out[C::ComponentWork]);
    out[C::KineticChange]=weight*mass*delta_s/gamma_sum;
    out[C::KineticChangeAbs]=std::abs(out[C::KineticChange]);
    out[C::Weight]=weight;
    out[C::NonrelativisticWork]=.5*weight*charge*dt*dot_component;
    out[C::NonrelativisticKineticChange]=.5*weight*mass*delta_s;
    if (has_total) {
        out[C::TotalFieldWork]=weight*charge*dt*dot_total/gamma_sum;
        out[C::TotalFieldWorkAbs]=std::abs(out[C::TotalFieldWork]);
        out[C::TotalWorkDefect]=out[C::KineticChange]-out[C::TotalFieldWork];
        out[C::TotalWorkDefectAbs]=std::abs(out[C::TotalWorkDefect]);
    }
    return out;
}
struct IonElectricWorkSpecies {
    std::string name;
    amrex::Real mass=0, charge=0;
};
struct IonElectricWorkParticles {
    std::size_t species=0;
    int grid=0, tile=0;
    long count=0;
    const std::uint64_t* idcpu=nullptr;
    const amrex::ParticleReal* weight=nullptr;
    // Cartesian coordinates/proper u, also in RZ.
    std::array<const amrex::ParticleReal*,3> gather_position{}, old_momentum{},
        midpoint_momentum{};
};
struct IonElectricWorkFields {
    // Immutable ACTUALLY projected component after the SAME native auxiliary
    // centering. RZ (Er,Etheta,Ez); 3D (Ex,Ey,Ez), all nodal V/m. No parser or
    // accepted registry is invoked. Caller owns copies and their lifetime.
    std::array<const amrex::MultiFab*,3> component{};
    // Optional full applied field, all or none. Particle-external E cannot be
    // inferred from these arrays; omitting it limits the total-work comparison.
    std::array<const amrex::MultiFab*,3> total{};
    // Optional applied-minus-nores field at the actual auxiliary gather layout.
    // Preserve the affine nores field before adding the iterate correction;
    // subtract after native synchronization/centering, without a new projector.
    std::array<const amrex::MultiFab*,3> remainder{};
    amrex::IntVect filled_ghosts{0};
};
struct IonElectricWorkResult {
    bool valid=false, has_total_field=false, has_remainder_field=false;
    IonElectricGatherContract gather_contract=IonElectricGatherContract::Unspecified;
    amrex::Long particles=0, invalid_particles=0;
    std::vector<IonElectricWorkLedger> species;
    // Acceptance-only references, not solver acceptance/error bounds. The
    // component Cauchy bounds bound work of the supplied gathered fields only.
    std::vector<IonElectricWorkDiagnostics> diagnostics;
};
// Collective acceptance diagnostic, pure/repeatable; no RNG, particle/field
// writes or accepted tally. Each charged species (including excluded alpha)
// has a row. Shape3 nodal MC/Direct gather, RZ m0 or Cartesian3D. Coordinates
// must already include any native crop. Filled guards are a caller contract.
IonElectricWorkResult MeasureImplicitIonElectricWork (
    amrex::Geometry const&, IonElectricWorkFields const&,
    std::vector<IonElectricWorkSpecies> const&,
    std::vector<IonElectricWorkParticles> const&, amrex::Real dt,
    IonElectricGatherContract);
// Native recorded bridge requires audit_ion_electric_work scratch from the
// final materialized particle push and returns RecordedFinalGather. Cropped
// coordinates are already recorded. Missing capture fails; no reconstruction.
// Call BEFORE endpoint extrapolation, boundary loss, redistribution, collisions
// or OU momentum changes. The caller must match the actual final gather fields:
// scratch alone does not identify the nonlinear iterate or field generation.
IonElectricWorkResult MeasureNativeImplicitIonElectricWork (
    WarpX&, int lev, amrex::Real dt, IonElectricWorkFields const&,
    NativeIonWorkContract);
// Explicit diagnostic approximation retained for comparisons. It always
// returns FinalMidpointReconstruction, never an exact final-gather receipt.
IonElectricWorkResult MeasureNativeReconstructedIonElectricWork (
    WarpX&, int lev, amrex::Real dt, IonElectricWorkFields const&,
    NativeIonWorkContract);
} // namespace warpx::thermal
#endif
