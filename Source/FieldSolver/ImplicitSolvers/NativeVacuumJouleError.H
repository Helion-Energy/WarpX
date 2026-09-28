/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_VACUUM_JOULE_ERROR_H_
#define WARPX_NATIVE_VACUUM_JOULE_ERROR_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>
class WarpX;
namespace warpx::thermal {
class EulerianThermalSources;
class NativeJouleWork;
struct ThermalSourceLedger;
using VacuumWorkVector = std::array<const amrex::MultiFab*,3>;

// An independently specified projection/roundoff budget, NOT a measured-current
// fit. The implementation initially admits only the derived axis/PEC, periodic-z
// 12x6 RZ operator. Fixed envelopes are checked at old and accepted-stage states.
struct VacuumJouleErrorBound {
    bool valid=false;
    amrex::Real projection=0, current=0, eta=0, dt=0, summation_gamma=0;
};
VacuumJouleErrorBound DeriveVacuumJouleErrorBound(
    const amrex::Geometry&, amrex::Real dt, amrex::Real theta, int old_step,
    amrex::Real el_rtol, amrex::Real el_atol, amrex::Real eta);

enum class VacuumJouleErrorStatus {
    Ready, Unsupported, Envelope, InvalidData, PopulatedOrIonCurrent,
    NotRecovery, CurrentResidual, WorkResidual, LedgerMismatch
};
struct VacuumJouleErrorReceipt {
    VacuumJouleErrorStatus status=VacuumJouleErrorStatus::Unsupported;
    amrex::Real allowance=0, signed_orphan=0, absolute_orphan=0;
    amrex::Real max_current=0, current_bound=0;
    bool Ready() const noexcept { return status==VacuumJouleErrorStatus::Ready; }
};

// Every input is read-only. work is the EXACT four-channel edge scratch from
// RedistributeThermalWork, not reconstructed eligibility. On nonzero orphan
// rows, all physical contributing nodes and Ji must be exactly zero. Populated
// ineligible rows retain the exact policy. No heat/current clipping, source
// masking, new thermal receiver, RHS projection or accepted write occurs.
// Check actual edge Jp independently: the nodal vacuum-A residual by itself
// does not bound the near-null native edge current. Physical current is zero
// only in this explicitly checked empty/no-external-current capability.
VacuumJouleErrorReceipt CheckVacuumJouleError(
    const amrex::Geometry&, const VacuumJouleErrorBound&,
    const amrex::MultiFab& physical_rho, const amrex::MultiFab& recovery_rho,
    amrex::Real recovery_threshold, const VacuumWorkVector& current,
    const VacuumWorkVector& ion_current, const VacuumWorkVector& electric,
    const VacuumWorkVector& work);

bool NativeVacuumJouleErrorEnabled();
void ValidateNativeVacuumJouleErrorScope(WarpX&);
// Called at ordinary step entry, after accepted storage exists, before mutation.
void ValidateNativeVacuumJouleErrorEntry(WarpX&);
// Acceptance-only scalar reductions/capture. Residual and Jv are untouched.
// Reports full signed recovery-removal work separately from this allowance.
VacuumJouleErrorReceipt PrepareNativeVacuumJouleError(
    WarpX&, const EulerianThermalSources&, const NativeJouleWork&,
    const ThermalSourceLedger&, amrex::Real dt, amrex::Real theta);
}
#endif
