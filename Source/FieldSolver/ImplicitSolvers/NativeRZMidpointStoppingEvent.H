/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_RZ_MIDPOINT_STOPPING_EVENT_H_
#define WARPX_NATIVE_RZ_MIDPOINT_STOPPING_EVENT_H_
#include "NativeRZSpatialStoppingEvent.H"
#include "ThermalStageSolver.H"
#include "Circuit/ExternalCircuitImpulse.h"
#include <cstdint>

namespace warpx::darwin { class NativeCircuitFieldImpulse; }
namespace warpx::implicit { class NativeAcceptedStepCandidate; }
namespace warpx::thermal {
/** Explicit borrowed source context. No constructor prepares/advances the
 * provider. Only the pure circuit overload consumes this binding. */
struct NativeStoppingCircuitBinding {
    enum class Phase { PreField, PostField };
    warpx::darwin::NativeCircuitFieldImpulse* field=nullptr;
    amrex::Real source_time=0.;
    Phase phase=Phase::PreField;
};
/** Separate circuit work receipt. The network has not been committed here:
 * actual network storage and provider-rounding work are deliberately absent.
 * Signed wall/reciprocity terms are measurements, not closure allowances. */
struct NativeStoppingCircuitWork {
    bool available=false, network_change_available=false, publication_available=false;
    std::uint64_t generation=0, action_evaluations=0;
    amrex::Real source_time=0.;
    NativeStoppingCircuitBinding::Phase phase=NativeStoppingCircuitBinding::Phase::PreField;
    amrex::Real initial_port_current=0., predicted_port_increment=0., port_impulse=0.;
    amrex::Real port_midpoint_work=0., coil_initial_energy=0.;
    amrex::Real coil_magnetic_self_work=0., coil_magnetic_cross_work=0.;
    amrex::Real coil_magnetic_change=0., total_coil_current_work=0.;
    amrex::Real coil_impulse_current_work=0., response_coil_current_work=0.;
    amrex::Real coil_curl_work=0., reciprocity_work=0., wall_work=0.;
    amrex::Real inventory_without_network=0., named_work_without_network=0.;
    amrex::Real accounting_defect=0., positive_work_scale=0., arithmetic_bound=0.;
};

/** Provisional joined receipt from the original whole-step owner. The provider
 * storage is measured by its native impulse API. Finite PIC spatial, coil,
 * reciprocity and wall work remain separate in CircuitWork/Ledger; none is heat.
 * This receipt never grants an endpoint lease or finalizes the transaction. */
struct NativeStoppingCircuitPublication {
    bool provider_attempted=false, provider_committed=false;
    bool fields_published=false, linkage_rebased=false;
    std::uint64_t application=0;
    WarpxCircuitImpulseReceiptV1 provider{};
    amrex::Real network_change=0., port_rounding_work=0.;
    amrex::Real inventory_change=0., named_work=0., accounting_defect=0.;
    amrex::Real arithmetic_bound=0.;
};

struct NativeStoppingThermalOptions {
    enum class InitialGuess { Zero, PrivateHalfStep };
    InitialGuess initial_guess=InitialGuess::Zero;
    // Numerical PC only. Physical residual/Jv and source law are unchanged.
    bool analytic_temperature_column=false;
    ThermalSolveOptions nonlinear=[] {
        ThermalSolveOptions o;
        o.relative_tolerance=0.;o.absolute_tolerance=1.e-12;
        o.linear_relative_tolerance=1.e-5;o.probe_relative_size=1.e-5;
        o.max_newton_iterations=32;o.max_linear_iterations=200;
        o.restart_length=60;o.max_backtracks=10;o.use_preconditioner=false;
        return o;
    }();
};
struct NativeStoppingThermalReceipt {
    ThermalSolveResult nonlinear;
    amrex::Real energy_scale=0.;
    amrex::Real maximum_residual=0., signed_residual=0., l1_residual=0.;
    amrex::Real residual_bound=0., delivered_heat=0., restricted_heat=0.;
    amrex::Real heat_map_error=0., storage_error=0., residual_identity_error=0.;
    amrex::Real roundoff_bound=0.;
    int predictor_preparations=0, predictor_failures=0;
    int pc_freezes=0, pc_temperature_probes=0, pc_projection_calls=0;
    int pc_applications=0, pc_projection_iterations=0;
    int physical_projection_calls=0, physical_projection_iterations=0;
    int pc_temperature_actions=0, pc_temperature_projection_calls=0;
    int pc_temperature_projection_iterations=0;
    amrex::Long pc_temperature_freeze_point_visits=0;
    amrex::Long pc_temperature_action_point_visits=0;
    amrex::Real pc_temperature_projection_residual=0.;
    int predictor_native_evaluations=0, predictor_linear_iterations=0;
    amrex::Real pc_projection_residual=0.;
    amrex::Real pc_beta=0., pc_electron_drag=0., pc_ion_response=0.;
};
/** Pure fixed-density source solve for native Psi, mean Je, and cell DeltaU.
 * No outer event admission. The unchanged native rate sees T(U0+DeltaU/2)
 * at every residual/Jv. All physical destinations remain unchanged until
 * CommitOnce. The original fixed-temperature helper remains a separate API.
 */
class NativeRZMidpointStoppingEvent {
public:
    using InvalidationHook=NativeRZSpatialStoppingEvent::InvalidationHook;
    NativeRZMidpointStoppingEvent(WarpX&,NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin,amrex::MultiFab& cell_energy,
        SpatialStoppingOptions const&,SpatialStoppingFields const&,
        NativeStoppingThermalOptions const& = {});
    NativeRZMidpointStoppingEvent(WarpX&,NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin,amrex::MultiFab& cell_energy,
        SpatialStoppingOptions const&,SpatialStoppingFields const&,
        RZStoppingBackground const&,NativeStoppingThermalOptions const& = {});
    // Pure circuit-aware residual/solve. Public CommitOnce stays disabled.
    // Only the owned retained candidate may join provider and field publication.
    // Circuit Capture/DefineState/Residual/Solve/PC/CopySolution are collective.
    // Prepared is a local read-only query; actions independently preflight on
    // all ranks. A captured accepted-material generation cannot be re-prepared
    // in place; stale material declines before changing output arrays/action.
    // Read-only operands/receipts retain historical data, not live readiness.
    NativeRZMidpointStoppingEvent(WarpX&,NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin,amrex::MultiFab& cell_energy,
        SpatialStoppingOptions const&,SpatialStoppingFields const&,
        RZStoppingBackground const&,NativeStoppingCircuitBinding const&,
        NativeStoppingThermalOptions const& = {});
    ~NativeRZMidpointStoppingEvent();
    NativeRZMidpointStoppingEvent(NativeRZMidpointStoppingEvent const&)=delete;
    bool Prepare();
    // Independent pure residual surface. Capture/Residual revoke readiness;
    // only a successful fresh Solve can restore CommitOnce capability.
    bool Capture();
    bool DefineState(WarpXSolverVec&) const;
    bool Residual(WarpXSolverVec& output,WarpXSolverVec const& state);
    bool Solve();
    // Pure numerical PC surface; never grants CommitOnce readiness.
    bool FreezePreconditioner(WarpXSolverVec const&);
    bool Precondition(WarpXSolverVec&,WarpXSolverVec const&);
    bool CopySolution(WarpXSolverVec&) const;
    bool Prepared() const;
    char const* Failure() const;
    RZSpatialStoppingLedger const& Ledger() const;
    NativeStoppingThermalReceipt const& ThermalReceipt() const;
    NativeStoppingCircuitWork const& CircuitWork() const;
    NativeStoppingCircuitPublication const& CircuitPublication() const;
    static bool ValidateCircuitWork(RZSpatialStoppingLedger const&,
        NativeStoppingCircuitWork const&,SpatialStoppingOptions const&);
    amrex::MultiFab const& CandidateEnergy() const;
    ablastr::fields::ConstVectorField CandidateElectronCurrent() const;
    ablastr::fields::ConstVectorField ElectricImpulse() const;
    ablastr::fields::ConstVectorField MeanElectronCurrent() const;
    // Read-only native operands for independent residual and work oracles.
    RZStoppingBackgroundLedger const& BackgroundLedger() const;
    ablastr::fields::ConstVectorField CandidatePotential() const;
    ablastr::fields::ConstVectorField CandidateMagnetic() const;
    ablastr::fields::ConstVectorField CandidateDisplacement() const;
    ablastr::fields::ConstVectorField CandidateConductorCurrent() const;
    ablastr::fields::ConstVectorField StoppingVelocity() const;
    ablastr::fields::ConstVectorField StoppingImpulseDensity() const;
    ablastr::fields::ConstVectorField ElectricAuxiliary() const;
    ablastr::fields::ConstVectorField ElectronReactionImpulse() const;
    ablastr::fields::ConstVectorField CandidateIonCurrent() const;
    ablastr::fields::ConstVectorField InitialIonCurrent() const;
    amrex::MultiFab const& MidpointTemperature() const;
    amrex::MultiFab const& HeatDensity() const;
    amrex::MultiFab const& EnergyIncrement() const;
    amrex::MultiFab const& EnergyResidual() const;
    bool CommitOnce(ablastr::fields::VectorField const&,InvalidationHook);
    // Collective: requires committed, open, same-owner source provenance.
    // This records the saved final current-projection potential, not PC scratch.
    bool AppendDisplacementCertificate(warpx::darwin::NativeLongitudinalProducerCertificate const& old,
        warpx::darwin::NativeLongitudinalProducerCertificate& updated) const;
    bool Rollback(InvalidationHook);
    void Finalize();
    struct Impl;
private:
    // Requires the candidate's original complete particle/field/provider rollback.
    // Any false return after provider_attempted mandates whole-step Cancel.
    bool CommitCircuitOnce(ablastr::fields::VectorField const&,InvalidationHook);
    friend class warpx::implicit::NativeAcceptedStepCandidate;
    std::unique_ptr<Impl> m_impl;
};
}
#endif
