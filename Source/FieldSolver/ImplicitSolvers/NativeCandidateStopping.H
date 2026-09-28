/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_CANDIDATE_STOPPING_H_
#define WARPX_NATIVE_CANDIDATE_STOPPING_H_
#include "NativeAcceptedStepCandidate.H"
#include "NativeRZSpatialStoppingEvent.H"
#include "NativeEndpointField.H"
#include "NativeRZMidpointStoppingEvent.H"
namespace warpx::implicit {
// The physical interval comes only from the candidate; no independent kick dt.
struct NativeCandidateStoppingOptions {
    std::string fast_species;
    amrex::Real coulomb_log=0.;
    amrex::Real proper_speed_cap=0.;
    amrex::Real relative_convention_budget=0.;
};
// The two private source intervals are exactly half of the owned macrostep.
struct NativeSymmetricStoppingOptions {
    bool material_support=false; // Default-OFF accepted-origin S1/F/transition S2.
    NativeCandidateStoppingOptions physical;
    warpx::thermal::NativeStoppingThermalOptions thermal=[] {
        warpx::thermal::NativeStoppingThermalOptions options;
        options.nonlinear.use_preconditioner=true;
        return options;
    }();
};
struct NativeCandidateStoppingReceipt {
    amrex::Real endpoint_time=0., interval=0.;
    std::uint64_t endpoint_epoch=0;
    warpx::thermal::RZSpatialStoppingLedger work;
    warpx::thermal::RZStoppingBackgroundLedger background;
    NativeEndpointResult endpoint;
    warpx::thermal::NativeStoppingCircuitWork circuit_work;
    warpx::thermal::NativeStoppingCircuitPublication circuit_publication;
};
struct NativeMidpointStoppingReceipt {
    NativeCandidateStoppingReceipt physical;
    warpx::thermal::NativeStoppingThermalReceipt thermal;
};
struct NativeCandidateStoppingResult {
    NativeCandidateStatus status=NativeCandidateStatus::Unsupported;
    std::string reason;
    // Provisional values only. No accepted receipt until owner.Finalize().
    NativeCandidateStoppingReceipt receipt;
};
}
#endif
