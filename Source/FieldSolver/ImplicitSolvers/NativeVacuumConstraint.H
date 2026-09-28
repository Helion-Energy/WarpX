/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_VACUUM_CONSTRAINT_H_
#define WARPX_NATIVE_VACUUM_CONSTRAINT_H_
#include "NativeInertiaSupport.H"
#include "NativePairedDarwinFields.H"
#include <ablastr/fields/MultiFabRegister.H>
class WarpX;
class CircuitCoupler;
namespace warpx::darwin {
struct NativeVacuumConstraintResult {
  amrex::Real vacuum_current_relative_error = 0.;
  bool compatible = false, converged = false;
  int iterations = 0, actions = 0, graph_iterations = 0;
  amrex::Real null_defect = 0., plasma_error = 0., divergence_error = 0.,
              residual = 0.;
};
NativeInertiaSupportOptions NativeVacuumSupportOptions(WarpX &);
/** Read-only accepted-field compatibility. No solve, projection or history
 * reset. */
NativeVacuumConstraintResult
CheckNativeVacuumAcceptedConstraint(WarpX &,
                                    amrex::MultiFab const &physical_density,
                                    amrex::MultiFab const &frozen_mask_density,
                                    ablastr::fields::VectorField const *reference = nullptr,
                                    ablastr::fields::VectorField const *accepted_transverse = nullptr);
bool ApplyNativeVacuumInertia(WarpX &, amrex::MultiFab const &density,
                              int component,
                              ablastr::fields::VectorField const &rate,
                              ablastr::fields::VectorField const &result);
/** Transactional output. E_T baseline is caller-selected accepted state; it is
 * distinct from endpoint A and never reset to force range compatibility.
 * F0 is the native non-inertial force; full current rate is evaluated
 * read-only. Returned Ei is an edge field; its nodal diagnostic mirror is a
 * separate map. If nores_force is supplied, the frozen affine particle field
 * is (E-force)+nores_force before native images/gather. Both force views are
 * copied before any solve action; the homogeneous response remains L*deltaE.
 * The caller must recapture the native pair for each live endpoint state. */
NativeVacuumConstraintResult SolveNativeVacuumConstraint(
    WarpX &, amrex::MultiFab const &physical_density,
    amrex::MultiFab const &mask_density,
    ablastr::fields::VectorField const &accepted_transverse,
    ablastr::fields::VectorField const &imposed_reference,
    ablastr::fields::VectorField const &force,
    ablastr::fields::VectorField const &held_longitudinal,
    ablastr::fields::VectorField const &out_transverse,
    ablastr::fields::VectorField const &out_inertia,
    ablastr::fields::VectorField const *nores_force = nullptr,
    NativePairedDarwinFields::EndpointDraft *endpoint = nullptr);
/** Same owned retained endpoint solve with a complete paired transverse
 * origin. It copies both parts before action and uses the identical live
 * rate/mass/divergence consumers. No high-only success fallback is allowed;
 * the ready draft still needs the owning phase's exact Bind/Publish contract.
 * This API does not open accepted/source/circuit or changed-support guards. */
NativeVacuumConstraintResult SolveNativeVacuumPairedEndpoint(
    WarpX &, amrex::MultiFab const &physical_density,
    amrex::MultiFab const &mask_density,
    ablastr::fields::VectorField const &accepted_transverse_high,
    ablastr::fields::VectorField const &accepted_transverse_low,
    ablastr::fields::VectorField const &imposed_reference,
    ablastr::fields::VectorField const &force,
    ablastr::fields::VectorField const &held_longitudinal,
    ablastr::fields::VectorField const &out_transverse,
    ablastr::fields::VectorField const &out_inertia,
    NativePairedDarwinFields::EndpointDraft &endpoint);
/** Circuit-coupled endpoint constraint. The accepted circuit/time and density
 * support are frozen for this call. Every action completes the zero-reference
 * vacuum field once and uses the resulting native Bdot/Cdot and electric wall
 * reference. The port map never consumes its own completed field as input.
 * Outputs must be mutually independent matching E-layout scratch and must not
 * alias any input. Invalid inputs or failed solves leave all outputs unchanged.
 * No accepted field, particle, circuit history or plugin state is advanced.
 */
NativeVacuumConstraintResult SolveNativeVacuumCircuitConstraint(
    WarpX &, CircuitCoupler &, amrex::Real time,
    amrex::MultiFab const &physical_density,
    amrex::MultiFab const &mask_density,
    ablastr::fields::VectorField const &accepted_transverse,
    ablastr::fields::VectorField const &force,
    ablastr::fields::VectorField const &held_longitudinal,
    ablastr::fields::VectorField const &out_transverse,
    ablastr::fields::VectorField const &out_inertia,
    ablastr::fields::VectorField const &out_reference,
    ablastr::fields::VectorField const *nores_force = nullptr);
/** Frozen outer iteration correction only. The raw full-source physical
 * acceptance test remains caller-owned and unchanged. No DG vacuum term. */
NativeVacuumConstraintResult CorrectNativeVacuumLongitudinal(
    WarpX &, amrex::MultiFab const &density, int component,
    amrex::Real interval, ablastr::fields::VectorField const &raw,
    ablastr::fields::VectorField const &held, amrex::MultiFab &potential_delta,
    ablastr::fields::VectorField const &field_delta);
/** Opt-in native verification; restores every registered field before return.
 */
bool VerifyNativeVacuumFiniteSchur(WarpX &, amrex::MultiFab const &density,
                                   int component, amrex::Real interval);
} // namespace warpx::darwin
#endif
