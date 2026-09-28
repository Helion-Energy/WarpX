/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_THERMAL_ADVANCE_H_
#define WARPX_DARWIN_THERMAL_ADVANCE_H_
#include "NativePairedDarwinFields.H"
#include "NativeStoppingCarryCertificate.H"
#include "KineticThermalMoments.H"
#include <cstdint>
#include "KineticThermalSpecies.H"
#include "AcceptedIonExchange.H"
#include "ImplicitIonEndpointTrial.H"
#include "ExpectedIonExchangeChannel.H"
#include "IonExchangeMomentAudit.H"
#include "ExpectedIonThermalPartner.H"
#include "FrozenPressureFieldCoupling.H"
#include "EulerianThermalSources.H"
#include "NativeJouleWork.H"
#include "EulerianSourceDiagonal.H"
#include "NativeConductionActivity.H"
#include "ThermalStageInitialGuess.H"
#include "EulerianDissipation.H"
#include "ThermalStageSolver.H"
#include "ThermalEnergyBalance.H"
#include "AbsorbingElectronInventory.H"
#include "ImplicitIonElectricWork.H"
#include "ImplicitFieldRollback.H"
#include "NonlinearSolvers/JacobianFunctionMF.H"

class ImplicitSolver;
class WarpX;
class HybridPICModel;
class ThetaImplicitHybrid;
class NativeEndpointAmpereOrigin;

namespace warpx::thermal {
class DarwinVacuumJointSolve;
struct NativeStoppingFieldWork;

// Owns one physical step's scratch. Accepted U/Te/Pe remain in HybridPICModel's
// field register and are written only by Commit. Both modes use exactly the
// same full field and Eulerian thermal equations and fixed step references.
class DarwinThermalAdvance final : public ThermalNonlinearOperator {
  public:
    explicit DarwinThermalAdvance (ThetaImplicitHybrid& solver);
    ~DarwinThermalAdvance () override;
    void SnapshotStepStart ();
    void BeginStep (amrex::Real time, amrex::Real dt);
    void PublishEndpointDensity ();
    int Solve (amrex::Real field_reference, int step);
    bool JointVacuumRequested() const noexcept { return m_joint_vacuum_requested; }
    bool JointVacuumActive() const noexcept;
    int SolveJointVacuum(int step); // 4 delegates the all-positive control to the old route.
    bool PublishJointVacuumStage();
    bool CaptureJointVacuumEndpointStage();
    bool FinishJointVacuumParticles(amrex::Real endpoint_time);
    bool CheckJointVacuumEndpointSupport();
    bool PrepareJointVacuumEndpointAmpere(amrex::Real endpoint_time);
    bool VerifyJointVacuumEndpointAmpere(amrex::Real endpoint_time);
    std::shared_ptr<NativeEndpointAmpereOrigin const> RetainJointVacuumEndpointAmpereOrigin(amrex::Real endpoint_time);
    // Opt-in capture only: failure declines the material capability, never
    // silently changes a field solve or publishes a stopping event.
    void RequestStoppingCarry(bool);
    void RequestStoppingCarry(NativeStoppingCarryRequest const&);
    void BindStoppingCarryStage();
    void BindStoppingCarryRotation(amrex::Real theta);
    std::shared_ptr<NativeStoppingCarryCertificate const> BindStoppingCarryEndpoint(amrex::Real,std::uint64_t);
    bool RefreshPressure ();
    bool AuditParticleEndpoints (amrex::Long* absorbed=nullptr, amrex::Long* total=nullptr,
                                 amrex::Long* reflected=nullptr) const;
    // Accepted native boundary/tally transaction, before any random ion kick.
    bool CommitParticleBoundaries ();
    bool HasOwnedAxialReflection () const noexcept;
    bool RefreshSpeciesContext ();
    bool CanCancel () const;
    bool PrepareAcceptance ();
    void Commit ();
    // Retain fields/private solver state through endpoint verification. These
    // calls are used only by the explicitly guarded native positive branch.
    bool CanRetainAcceptance () const;
    void PublishCandidate ();
    void FinalizeCandidate ();
    ThermalEnergyBalance const& CandidateEnergyBalance () const;
    bool CaptureMaterialFieldWork(NativeEndpointAmpereOrigin const&,NativeStoppingFieldWork&,
        ablastr::fields::ConstVectorField const* post_source_initial=nullptr) const;
    void InvalidateBorrowedStepViews () noexcept;
    bool CommitIonExchange ();
    void Cancel ();

    bool Residual (WarpXSolverVec&, const WarpXSolverVec&, int, bool) override;
    amrex::Real StepBound (const WarpXSolverVec&, const WarpXSolverVec&) const override;
    bool Freeze (const WarpXSolverVec&, int, bool, ThermalSolveResult&) override;
    bool Precondition (WarpXSolverVec&, const WarpXSolverVec&) override;
    void RestoreInput (const WarpXSolverVec&) override;
    bool HasResidualRemainder () const noexcept override { return m_current_remainder_requested; }
    bool CopyResidualRemainder (WarpXSolverVec&) const override;
    bool DifferenceResidualRemainder (WarpXSolverVec&,WarpXSolverVec const&,
        WarpXSolverVec const&,WarpXSolverVec const&,WarpXSolverVec const&,amrex::Real) const override;
    // Borrowed only inside FieldResidual -> native displacement producer.
    ablastr::fields::VectorField CurrentRemainderOutput(amrex::Real interval);
    void PublishCurrentRemainder();
    void InvalidateCurrentRemainder() noexcept;
    void CurrentRemainderFieldsChanged() noexcept {
        if(!m_current_remainder_collecting)InvalidateCurrentRemainder();
    }
    class CurrentRemainderMap {
      public:
        explicit CurrentRemainderMap(DarwinThermalAdvance&);
        ~CurrentRemainderMap();
        bool Finish();
      private:
        DarwinThermalAdvance& a;bool active=false,finished=false;
    };

    // These methods contain CUDA device lambdas and must be public for nvcc.
    bool RefreshContext ();
    bool FreezePressureResponse ();
    bool Source (const amrex::MultiFab& energy, amrex::MultiFab& rates,
                 amrex::MultiFab& derivative);
    bool PrepareInitialThermalGuess (bool& changed);
    void CellMagneticField ();
    void InitializePushContext ();
    void InitializeAdditiveResistivity ();
    void RefreshDensityDependentContext ();
    bool PrepareAbsorbingBoundary ();
    void InitializeIonElectricWork ();
    ablastr::fields::VectorField const* IonElectricWorkOutput ();
    void CaptureIonElectricWorkBase (ablastr::fields::VectorField const&);
    void PrepareIonElectricWorkField ();
    void CaptureIonElectricWorkTotal ();
    bool PrepareIonElectricWorkReceipt ();
    void PrintIonElectricWorkReceipt () const;
    void RefreshAbsorbingLongitudinalField ();
    bool SetPushThermodynamics ();
    bool PushContextConverged ();
    int PushIterations () const { return m_push_iterations; }
    EulerianDissipation* Dissipation () const { return m_dissipation.get(); }
    const amrex::MultiFab& PushCharge () const { return m_push_charge; }

  private:
    friend class DarwinVacuumJointSolve;
    bool m_current_remainder_requested=false,m_current_remainder_collecting=false;
    std::uint64_t m_current_remainder_generation=0,m_current_remainder_ready=0;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_current_remainder;
    std::array<std::unique_ptr<amrex::MultiFab>,AMREX_SPACEDIM> m_number_flux_remainder;
    friend class NativeStoppingCarryCertificate;
    std::shared_ptr<NativeStoppingCarryCertificate> m_stopping_carry;
    bool m_joint_vacuum_requested = false;
    std::unique_ptr<DarwinVacuumJointSolve> m_joint_vacuum;
    static constexpr const char* energy_name = "hybrid_electron_energy_fp";
    ThetaImplicitHybrid& m_solver;
    WarpX& m_simulation;
    HybridPICModel& m_model;
    amrex::Geometry m_geometry;
    bool m_coupled = false, m_open = false, m_receipt_ready = false;
    bool m_candidate_published = false;
    bool m_pressure_coupling = false, m_source_diagonal_enabled = true;
    int m_source_diagonal_fallbacks = 0;
    bool m_expected_ou_audit = false, m_population_thermal_partner = false;
    bool m_relativistic_reference_audit = false;
    bool m_ion_electric_work_enabled=false, m_ion_electric_work_ready=false;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_ion_work_raw, m_ion_work_aux, m_ion_work_total;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_ion_work_base_fp, m_ion_work_remainder;
    bool m_ion_work_base_ready=false;
    ablastr::fields::VectorField m_ion_work_raw_view{};
    IonElectricWorkResult m_ion_work_receipt;
    amrex::Real m_population_thermal_energy = 0, m_population_electron_energy = 0;
    amrex::MultiFab m_legacy_relaxation;
    std::unique_ptr<ExpectedIonThermalPartner> m_thermal_partner;
    ExpectedIonEnergyOptions m_expected_options;
    std::unique_ptr<ExpectedIonExchangeSource> m_expected_exchange;
    std::unique_ptr<ImplicitIonEndpointTrial> m_expected_endpoints;
    std::unique_ptr<IonExchangeMomentAudit> m_exchange_moments;
    std::vector<std::array<amrex::Real,IonExchangeMomentComponent::Count>> m_exchange_moment_ledger;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_exchange_drift;
    amrex::MultiFab m_exchange_budget, m_expected_endpoint_work;
    std::vector<ExpectedIonWorkLedger> m_expected_ledger;
    ThermalStageGuessOptions m_guess_options;
    ThermalEnergyBalance m_balance;
    ThermalSourceLedger m_source_balance;
    amrex::Real m_time = 0, m_dt = 0;
    amrex::Long m_absorbed_particles=0, m_endpoint_population=0, m_reflected_particles=0;
    bool m_axial_boundary_completed=false;
    bool m_particle_boundary_committed=false;
    amrex::Real m_absorbed_charge_integral=0;
    AbsorbingElectronInventoryResult m_absorbing_inventory;
    amrex::MultiFab m_survivor_charge, m_absorbed_charge, m_survivor_energy;
    std::vector<std::array<amrex::Real,3>> m_boundary_tallies_before;
    amrex::Real m_energy_reference = -1;
    amrex::Real m_electric_scale = 1.e5, m_energy_scale = 1;
    int m_outer_iterations = 30, m_push_iterations = 30;
    amrex::Real m_push_tolerance = 1.e-12;
    ThermalSolveOptions m_options;
    ThermalStageOptions m_stage_options;
    WarpXSolverVec m_energy, m_combined, m_field, m_field_rhs, m_field_work, m_pressure_work;
    WarpXSolverVec m_saved_field, m_saved_field_old, m_saved_field_prev;
    warpx::implicit::FieldRollback m_field_rollback;
    struct StepFlags {
        amrex::Real dt=0;
        bool have_old=false,have_prev=false,rho_captured=false,jp_captured=false,
            plasma_valid=false,native_circuit_open=false,history_initialized=false;
        int history_levels=0;
    } m_saved_flags;
    warpx::darwin::NativePairedDarwinFields::Snapshot m_saved_paired_fields;
    warpx::darwin::NativePairedDarwinFields::Snapshot m_saved_endpoint_pair;
    amrex::Vector<amrex::Real> m_saved_circuit_scales, m_saved_external_initial;
    amrex::MultiFab m_old_energy, m_endpoint, m_old_charge, m_endpoint_charge, m_bound_density;
    amrex::MultiFab m_magnetic, m_source, m_nodal_rates, m_ion_rates, m_additive_eta;
    amrex::MultiFab m_push_charge, m_push_difference;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_old_current, m_push_current;
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_number_flux;
    std::unique_ptr<KineticThermalMoments> m_old_moments, m_stage_moments, m_endpoint_moments;
    std::shared_ptr<ThermalStageLifetime> m_stage_lifetime=std::make_shared<ThermalStageLifetime>();
    std::unique_ptr<EulerianThermalStage> m_stage;
    std::unique_ptr<ThermalConductionPC> m_pc;
    std::unique_ptr<EulerianThermalSources> m_sources;
    std::unique_ptr<NativeJouleWork> m_joule_work;
    std::unique_ptr<EulerianSourceDiagonal> m_source_diagonal;
    std::unique_ptr<NativeConductionActivity> m_conduction_activity;
    std::unique_ptr<KineticThermalSpecies> m_species;
    std::unique_ptr<EulerianDissipation> m_dissipation;
    std::unique_ptr<AcceptedIonExchange> m_ion_exchange;
    std::unique_ptr<FrozenPressureFieldCoupling> m_pressure_pc;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_pressure_weight;
    JacobianFunctionMF<WarpXSolverVec, ImplicitSolver> m_field_pc;
    std::uint64_t m_density_epoch = 0;
    void CopyFields (WarpXSolverVec& destination, const WarpXSolverVec& source) const;
    bool FieldResidual (WarpXSolverVec& output, const WarpXSolverVec& electric,
                        int iteration, bool probe);
    bool ThermalResidual (amrex::MultiFab& output);
};
} // namespace warpx::thermal
#endif
