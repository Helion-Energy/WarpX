/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_MATERIAL_SOURCE_H_
#define WARPX_NATIVE_STOPPING_MATERIAL_SOURCE_H_
#include "NativeStoppingMaterialOwner.H"
#include "NativeRZMidpointStoppingEvent.H"
#include "NativeEndpointField.H"
#include "ThermalEnergyBalance.H"
#include "ImplicitIonElectricWork.H"
#include <memory>
namespace warpx::thermal {
// Value-only field work copied before source mutation invalidates its producer.
struct NativeStoppingFieldWork {
    bool available=false;
    amrex::Real time=0.,interval=0.;
    std::uint64_t epoch=0;
    ThermalEnergyBalance thermal;
    IonElectricWorkResult ions;
    NativeEndpointMagneticWork magnetic;
};
struct NativeStoppingSourceOptions {
    NativeStoppingOwnerOptions physical;
    amrex::Real relative_convention_budget=0.; // Explicit, never fitted.
    NativeStoppingThermalOptions thermal;
    bool endpoint_acceptance=false; // Additive; all existing callers remain provisional.
    bool stable_drag_increment=false; // Additive arithmetic selection, agreed at capture.
    bool refine_current_projection=false; // One bounded same-map defect correction.
    bool register_native_theta=false; // Actual represented endpoint frame, setup-agreed.
};
struct NativeStoppingProjectionReceipt {
    bool refined=false,converged=false,action_evaluated=false;
    int solves=0,iterations=0;
    amrex::Real initial_residual=0.,first_residual=0.,original_target=0.;
    amrex::Real correction_initial=0.,correction_residual=0.,correction_target=0.;
    amrex::Real action_residual=0.;
};
struct NativeStoppingSourceWork {
    RZSpatialStoppingLedger physical;
    RZStoppingBackgroundLedger background;
    StoppingCarryReport carry;
    NativeStoppingThermalReceipt thermal;
    // Last completed/declined private projection actions: current, then Psi.
    std::array<NativeStoppingProjectionReceipt,2> projections;
    // Identity/provenance and preservation of a finite inherited defect do
    // not resolve the field endpoint discrepancy or turn it into heat.
    amrex::Real background_identity_error=0.,background_identity_bound=0.;
    amrex::Real preserved_defect_error=0.,preserved_defect_bound=0.;
    amrex::Real inherited_ampere_work=0.,source_increment_constraint_work=0.;
    bool field_transition_background=false,accepted_origin_background=false,field_endpoint_ampere_closed=false;
    // B_static is the native time-independent part in B_static + curl(A).
    // These are a split of existing full-B curl work, never additional heat.
    amrex::Real static_magnetic_cross_work=0.,static_magnetic_cross_scale=0.;
    amrex::Real static_current_work=0.,static_curl_work=0.,dynamic_curl_work=0.;
    bool static_background=false;
    // Explicit finite transfers, all distinct from heat and residual allowances.
    amrex::Real material_endpoint_rounding=0.;
    amrex::Real ion_increment_remainder=0.,electron_increment_remainder=0.;
    amrex::Real current_base_remainder=0.,current_representation_work=0.;
    amrex::Real energy_storage_remainder=0.;
    // Signed diagnostics only; neither a heat change nor a physical allowance.
    amrex::Real correlated_ion_increment_remainder=0.;
    amrex::Real correlated_current_representation_work=0.,theta_registration_work=0.;
    bool fresh=false,physical_gates=false,publication_available=false;
};
struct NativeStoppingSourceView {
    bool captured=false,action=false,solved=false;
    std::string failure;
    NativeStoppingOwnerView material;
    ablastr::fields::ConstVectorField potential_increment{},magnetic_increment{},curl_increment{};
    ablastr::fields::ConstVectorField displacement_increment{},conductor_increment{};
    ablastr::fields::ConstVectorField ion_increment{},ion_base{},electron_increment{};
    // Present only for selected complete actions; all borrowed allocations
    // remain subject to the same collective overlap and freshness checks.
    ablastr::fields::ConstVectorField correlated_ion_increment{},theta_registration{};
    ablastr::fields::ConstVectorField longitudinal_current{},longitudinal_impulse{};
    // Read-only completed action operands. A converged numerical root can still
    // fail the separate physical work gates; these views never confer readiness.
    WarpXSolverVec const* residual=nullptr;
    WarpXSolverVec const* root_candidate=nullptr;
    // Candidate inventories currently expose valid rows only. No guard-band
    // publication or endpoint reclosure capability is granted by this view.
    ablastr::fields::ConstVectorField potential{},magnetic{},displacement{},conductor_current{};
    // conductor_current is a source-local incremental reaction accumulator;
    // it is not a pre-existing or published vacuum endpoint W inventory.
    amrex::MultiFab const* energy=nullptr;
    amrex::MultiFab const* energy_increment=nullptr;
    NativeStoppingSourceWork work;
};
struct NativeStoppingSourceAcceptance {
    bool ready=false,includes_field=true;
    NativeEndpointAmpereCheck field_current,published_current;
    NativeStoppingFieldWork field;
    amrex::Real source_ion=0.,source_electron=0.,source_internal=0.,source_magnetic=0.;
    amrex::Real initial_internal=0.,final_internal=0.;
    amrex::Real magnetic_materialization=0.,magnetic_materialization_bound=0.;
    amrex::Real source_inventory_error=0.,source_inventory_bound=0.;
    amrex::Real field_ion_work_defect=0.,field_change=0.,source_change=0.;
    amrex::Real combined_change=0.,named_work=0.,accounting_error=0.,accounting_bound=0.;
};
enum class NativeStoppingSourcePublicationStatus { Unsupported, Declined, Provisional, Terminal, Acceptable };
struct NativeStoppingSourcePublication {
    NativeStoppingSourcePublicationStatus status=NativeStoppingSourcePublicationStatus::Unsupported;
    NativeStoppingSourceWork work;
    NativeStoppingSourceAcceptance acceptance;
    amrex::Real time=0.,interval=0.;
    std::uint64_t epoch=0;
    int publication_residual_evaluations=0;
    amrex::Real endpoint_plasma_error=0.,endpoint_divergence_error=0.,endpoint_null_defect=0.;
    amrex::Real published_current_rounding=0.,published_current_bound=0.;
    bool attempted=false,fields_written=false,endpoint_reclosed=false,cancelled=false;
    bool source_finalization_available=false,field_endpoint_ampere_closed=false;
    std::string reason;
};
/** Actual retained-owner, fixed-position live-U source equations on P and V.
 * Only the candidate may capture it. Public numerical actions remain pure;
 * no public numerical method grants a finalizable endpoint or changes physical fields.
 * Outputs are published only after a successful complete private action.
 * The default positive-only helper and its arithmetic are separate, unchanged.
 */
class NativeStoppingMaterialSource final {
public:
    struct Impl;
    ~NativeStoppingMaterialSource();
    NativeStoppingMaterialSource(NativeStoppingMaterialSource const&)=delete;
    bool DefineState(WarpXSolverVec&) const;
    bool Residual(WarpXSolverVec&,WarpXSolverVec const&);
    bool Solve();
    NativeStoppingSourceView View();
    void Invalidate() noexcept;
private:
    NativeStoppingMaterialSource(WarpX&,AcceptedStoppingOptions const&,
        NativeStoppingSourceOptions const&,amrex::Real,std::uint64_t,amrex::Real,
        std::shared_ptr<NativeStoppingCarryCertificate const>,
        std::shared_ptr<NativeEndpointAmpereOrigin const> = {},
        NativeStoppingFieldWork const* = nullptr);
    bool Capture();
    // Two private phases within the candidate's non-interleaved transaction.
    // Preparation is pure; publication requires its pair lease already revoked.
    bool PreparePublication(NativeStoppingSourcePublication&);
    bool PublishPrepared();
    bool CheckPublication(NativeStoppingSourcePublication&);
    bool CheckAcceptance(NativeStoppingSourcePublication&);
    std::unique_ptr<Impl> m_impl;
    friend class warpx::implicit::NativeAcceptedStepCandidate;
};
}
#endif
