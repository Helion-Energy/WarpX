/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeRetainedAcceptance.H"
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
namespace warpx::implicit {
namespace { NativeAcceptanceValidator validator; }
bool NativeRetainedAcceptanceEnabled() {
    bool enabled=false;
    amrex::ParmParse("implicit_evolve").queryAdd("native_retained_acceptance",enabled);
    return enabled;
}
void SetNativeAcceptanceValidator(NativeAcceptanceValidator value) { validator=value; }
bool CheckNativeAcceptance(WarpX& simulation, AcceptancePhase phase) {
    bool accepted=!validator.check || validator.check(simulation,phase,validator.context);
    amrex::ParallelDescriptor::ReduceBoolAnd(accepted);
    return accepted;
}
}
