/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_PAIRED_DARWIN_FIELDS_H_
#define WARPX_NATIVE_PAIRED_DARWIN_FIELDS_H_
#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
class WarpX;
class ThetaImplicitHybrid;
namespace warpx::thermal { class DarwinThermalAdvance; }
namespace warpx::implicit { class NativeAcceptedStepCandidate; }
namespace warpx::darwin {
struct NativeVacuumConstraintResult;
struct NativeEndpointPairProducer;
/** Opt-in finite-stage arithmetic owner. The nonlinear unknown remains the
 * native transverse high; its low is an immutable affine coordinate during
 * each inner solve. All physical consumers share this owner. No low is an
 * independent accepted field or a physical energy/source inventory. The
 * exclusive endpoint role retains a pending instantaneous representation;
 * it cannot yet finalize or advance a source/history consumer. */
class NativePairedDarwinFields {
public:
    using R=amrex::Real;
    using V=ablastr::fields::VectorField;
    using CV=ablastr::fields::ConstVectorField;
    enum class Phase : int { Idle, FiniteStage, Materialized, EndpointPending,
        EndpointAccepted, EndpointInitializing, EndpointSourceOrigin, EndpointSourcePending };
    enum class Role : int { FiniteStage, Endpoint };
    enum class EndpointPart : int { TransverseHigh, TransverseLow, ElectricHigh,
        ElectricLow, InertiaHigh, InertiaLow, ElectronRateHigh, ElectronRateLow,
        IonRateHigh, IonRateLow, Count };
    static constexpr int endpoint_parts=static_cast<int>(EndpointPart::Count);
    struct EndpointView {
        std::array<CV,endpoint_parts> fields{};
        amrex::MultiFab const* nodal_inertia_high=nullptr;
        amrex::MultiFab const* nodal_inertia_low=nullptr;
    };
    /** Local receipt introspection only; owner methods perform collective
     * preflight before a physical action. Revocation is never rolled back. */
    class EndpointReceipt {
    public:
        bool Valid() const noexcept;
        struct Token;
    private:
        std::weak_ptr<Token const> m_token;
        std::uint64_t m_generation=0;
        friend class NativePairedDarwinFields;
    };
    /** Deep-owned native solve, frozen inputs and represented outputs. Only the
     * joined producer can make it ready. It never borrows a stack-local Core. */
    class EndpointDraft {
    public:
        EndpointDraft();
        ~EndpointDraft();
        EndpointDraft(EndpointDraft&&) noexcept;
        EndpointDraft& operator=(EndpointDraft&&) noexcept;
        EndpointDraft(EndpointDraft const&)=delete;
        EndpointDraft& operator=(EndpointDraft const&)=delete;
        bool Ready() const noexcept;
        struct Impl;
    private:
        std::unique_ptr<Impl> m_impl;
        bool Bind(WarpX&,R,std::uint64_t,bool initial=false,bool restored=false,bool source_origin=false);
        void WriteCheckpoint(std::string const&) const;
        static EndpointDraft ReadCheckpoint(WarpX&,std::string const&,R,std::uint64_t);
        bool Matches(WarpX&,EndpointView const&) const;
        bool MatchesTransverse(CV const&,CV const&) const;
        bool BindAcceptedBoundary(WarpX&,EndpointView const&);
        bool RedistributeAccepted(WarpX&,EndpointView const&,bool,amrex::IntVect const&,
                                  bool,amrex::IntVect const&,bool,amrex::IntVect const&);
        bool PublishAuxiliary(WarpX&,V const&,V const&);
        bool MatchesAuxiliary(WarpX&,CV const&,CV const&) const;
        NativeVacuumConstraintResult Check(WarpX&,EndpointView const&);
        bool Publish(WarpX&,std::array<V,endpoint_parts> const&,
                     amrex::MultiFab&,amrex::MultiFab&);
        friend class NativePairedDarwinFields;
        friend struct NativeEndpointPairProducer;
    };
    struct Publication {
        std::array<R,3> longitudinal_low{},transverse_low{},potential_low{},
            magnetic_difference{},current_difference{},displacement_low{},rate_low{};
        R scalar_low=0.;
        // Signed after-minus-before numerical inventory differences, using
        // native dual volumes and one owner for each periodic alias.
        R magnetic_inventory_change=0.,longitudinal_inventory_change=0.;
        R held_electron_inventory_change=0.;
        R endpoint_ion_quadrature_change=0.,endpoint_electron_quadrature_change=0.;
        R endpoint_current_quadrature_change=0.;
        R stage_ion_terminal_rounding_work=0.,stage_electron_terminal_rounding_work=0.;
        R stage_inertia_terminal_rounding_work=0.;
        R stage_assembly_mismatch=0.;
        std::array<R,3> signed_current_change{},electric_publication_change{};
        bool stage_captured=false,materialized=false,complete=false,finite=true;
    };
    struct Snapshot {
        Phase phase=Phase::Idle;
        R dt=0.,time=0.;
        std::uint64_t epoch=0,trials=0;
        bool magnetic_ready=false,current_ready=false;
        Publication publication;
        bool endpoint_step_open=false,endpoint_history_captured=false;
        bool endpoint_origin_ready=false,endpoint_previous_ready=false;
        bool endpoint_origin_paired=false,endpoint_boundary_bound=false;
        bool endpoint_auxiliary_ready=false;
    };
    static bool Requested();
    static bool EndpointRequested();
    static bool EndpointAcceptanceRequested();
    explicit NativePairedDarwinFields(WarpX&,Role=Role::FiniteStage,bool accepted_consumers=false);
    ~NativePairedDarwinFields();
    NativePairedDarwinFields(NativePairedDarwinFields const&)=delete;
    bool Active() const noexcept;
    bool EndpointPending() const noexcept;
    bool EndpointAccepted() const noexcept;
    bool EndpointPreSourceReclosure() const noexcept;
    // Exact registered old/previous transverse references. They are origins
    // and nonlinear guesses only, never a time-difference field recovery.
    struct EndpointHistoryView { CV high{},low{},previous_high{},previous_low{};
        bool paired=false,previous_ready=false; };
    bool EndpointHistory(EndpointHistoryView&) const;
    bool MatchesEndpointOrigin(CV const&) const;
    struct EndpointAuxiliaryView { CV high{},low{}; };
    bool EndpointAuxiliary(EndpointReceipt const&,EndpointAuxiliaryView&) const;
    EndpointReceipt EndpointLease() const noexcept;
    EndpointView EndpointFields() const;
    bool MatchesEndpoint(EndpointReceipt const&) const;
    NativeVacuumConstraintResult CheckEndpoint(amrex::MultiFab const& density,
        amrex::MultiFab const& mask, V const* transverse=nullptr);
    bool AdoptEndpoint(EndpointDraft&&,R time,std::uint64_t epoch);
    // Captured owner capability, never a live parser grant. Initial publication
    // retains the same physical pair at the current clock without a step.
    bool InitialEndpointSelected() const noexcept;
    bool AdoptInitialEndpoint(EndpointDraft&&,R time);
    bool InitialEndpointPending() const noexcept;
    bool CopyEndpointTransverse(V const&);
    std::uint64_t LocalMemory() const noexcept;
    Snapshot State() const noexcept;
    void RestoreState(Snapshot const&) noexcept;
    void InvalidateBorrowedViews() noexcept;
    void BeginField(R dt,R theta,R time,std::uint64_t epoch);
    void SaveOuter();
    CV ConstraintDefect(V const& raw,CV const& held);
    void ApplySchur(amrex::MultiFab const& delta_phi,CV const& delta,
                    CV const& held,amrex::MultiFab const& held_phi,R omega,V const& transverse);
    void ConditionTransverse();
    void BuildPotential(R theta_dt);
    void BuildMagnetic();
    bool OwnsMagnetic(V const&) const noexcept;
    void RefreshAmpere(V const& current);
    void SubtractDisplacement(V const& current,V const* thermal_remainder=nullptr);
    void BuildDisplacement();
    void BuildCorrelatedRate(CV const& cdot,CV const& delta_ion,CV const& old_displacement,
                             V const& rate,V const& displacement);
    void CompletePECStageRateLow();
    void AssembleTotalElectric();
    void FormTransverseOhmRHS(bool already_subtracted_high);
    void Extrapolate(R theta,V const& transverse,CV const& old_transverse);
    bool CaptureStageAccounting();
    Publication MaterializeEndpoint();
    Publication CompletePublication();
    CV LongitudinalLow() const;
    CV PreviousLongitudinalLow() const;
    CV TransverseLow() const;
    CV ConditionedTransverseLow() const;
    CV DisplacementLow() const;
    CV RateLow() const;
    V InertiaElectricLow();
    V CurlRateLow();
    V CurlRatePotentialScratch();
    V CurlRateMagneticScratch();
    amrex::MultiFab const& ScalarLow() const;
    Publication const& LastPublication() const noexcept;
    // Public opaque name allows NVCC extended device lambdas in its methods.
    struct Impl;
private:
    // Only the existing whole-step owners can rotate, promote or rebind the
    // accepted boundary. Adding these consumers does not open Finalize.
    bool CompleteInitialEndpoint();
    bool CheckpointReady() const;
    void WriteEndpointCheckpoint(std::string const&) const;
    void ReadEndpointCheckpoint(std::string const&);
    bool BeginEndpointStep(R time,std::uint64_t epoch);
    // Candidate-only source mutation seam. Copy the entry pair first. The
    // original snapshot/history/prior Core stay with the same transaction.
    bool BeginEndpointSourceReclosure(EndpointReceipt const&);
    bool BeginEndpointPreSource();
    bool FinishEndpointPreSource(EndpointReceipt const&);
    bool CaptureEndpointHistory(V const& old,V const& previous,bool keep_previous);
    bool PublishEndpointAuxiliary();
    bool CanAcceptEndpoint(EndpointReceipt const&);
    void AcceptEndpoint() noexcept;
    bool BindAcceptedEndpointBoundary();
    bool RedistributeAcceptedEndpoint(bool local,amrex::IntVect const& max_cells,
        bool sort,amrex::IntVect const& bins,bool deposition,amrex::IntVect const& index_type);
    friend class ::ThetaImplicitHybrid;
    friend class warpx::thermal::DarwinThermalAdvance;
    friend class warpx::implicit::NativeAcceptedStepCandidate;
    friend bool BindNativeAcceptedEndpointBoundary(WarpX&);
    friend bool RedistributeNativeAcceptedEndpoint(WarpX&,bool,amrex::IntVect const&,
        bool,amrex::IntVect const&,bool,amrex::IntVect const&);
    std::unique_ptr<Impl> m_impl;
};
// Owned by the actual Theta solver. Never infer readiness from parser flags or
// low norms; initialization and the DTA phase transaction own this capability.
NativePairedDarwinFields* NativeActivePairedFields(WarpX&) noexcept;
// Exclusive endpoint role. This never enables positive finite-stage hooks.
NativePairedDarwinFields* NativeEndpointPairedFields(WarpX&) noexcept;
bool NativeEndpointPairSelected(WarpX&) noexcept;
void RestoreNativeAcceptedEndpointCheckpoint(WarpX&,std::string const&);
// Captured numerical reconstruction/validated restart context, not a parser grant.
bool NativeEndpointPairRestartContext(WarpX&) noexcept;
// Read-only actual owner selection; callers collectively decide before action.
bool NativeEndpointPMCQualificationSelected(WarpX&);
// Local captured startup capability, available before owner allocation.
// This grants no endpoint receipt, source action or physical readiness.
bool NativeEndpointCudaQualificationSelected(WarpX&) noexcept;
// Explicit once-only caller handoff after an accepted physical step and clock
// advance. A changed population/field/configuration declines before any action.
// This helper does not advance the clock or authorize a source event.
bool BindNativeAcceptedEndpointBoundary(WarpX&);
// The owner revokes the old lease before invoking native redistribution and
// optional sorting. It accepts only the predicted particle relocation with
// exact per-ID attribute bytes and unchanged fields/configuration/clock.
// Failure leaves the lease revoked; the accepted driver must stop.
bool RedistributeNativeAcceptedEndpoint(WarpX&,bool local,amrex::IntVect const& max_cells,
    bool sort,amrex::IntVect const& bins,bool deposition,amrex::IntVect const& index_type);
} // namespace warpx::darwin
#endif
