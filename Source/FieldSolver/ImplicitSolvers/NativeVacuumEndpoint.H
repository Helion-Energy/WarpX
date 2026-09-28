/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_VACUUM_ENDPOINT_H_
#define WARPX_NATIVE_VACUUM_ENDPOINT_H_
#include <ablastr/fields/MultiFabRegister.H>
class WarpX;
bool NativeVacuumEndpointEnabled();
// Collective read-only predicate after native external initialization. All
// named resolved profiles/templates and materialized driven E/B must be zero
// drive; external current is excluded. The static background B is untouched.
// This is not retained/PMC/MM capability admission.
bool NativeZeroExternalDriveSupported(WarpX &);
// Static source-free physical envelope shared by retained MM0 and MM1.
// MM1 additionally requires its independently validated density projection.
bool NativeStaticVacuumEndpointSupported(WarpX &);
// Retained first slice additionally restricts external fields to the exact
// static zero scale used to carry the native A boundary pin.
bool NativeRetainedStaticVacuumSupported(WarpX &);
// Source-free RZ MM0 combined caller; the native circuit transaction phase is
// checked separately immediately before the actual endpoint constraint.
bool NativeRetainedCircuitVacuumSupported(WarpX &);
void ValidateNativeVacuumEndpointCapabilities(WarpX &);
char const *NativeVacuumSupportRecordName();
// Exact accepted transverse origin for the native circuit/vacuum split.
char const *NativeVacuumTransverseRecordName();
void AllocateNativeVacuumSupportStorage(WarpX &);
void ValidateNativeVacuumRestart(WarpX &);
void ValidateNativeVacuumEntryDensity(WarpX &);
// Read-only native gradient check; ordinary-step observations require write_fields.
void AuditNativeVacuumPotential(WarpX &, char const *label, bool required = false);
// Numerical rejection restores temporary fields and publishes no endpoint.
// An optional private destination receives the exact solved transverse field
// only on success. Its storage may be the read-only baseline, but cannot alias
// any registered physical field or another output component.
bool TryConstrainNativeVacuumEndpointField(
    WarpX &, amrex::Real time, bool initial,
    ablastr::fields::VectorField const &accepted_transverse,
    ablastr::fields::VectorField const *exact_transverse = nullptr);
void ConstrainNativeVacuumEndpointField(
    WarpX &, amrex::Real time, bool initial,
    ablastr::fields::VectorField const &accepted_transverse,
    ablastr::fields::VectorField const *exact_transverse = nullptr);
void CompleteNativeVacuumLongitudinalSource(WarpX &, amrex::Real interval);
// Shared exact companion mirror, implemented beside its original native helper.
void CopyNativeEndpointCurrentMirror(WarpX &,
                                     ablastr::fields::VectorField const &,
                                     amrex::MultiFab &);
#endif
