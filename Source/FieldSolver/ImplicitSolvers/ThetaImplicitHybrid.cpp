#include "NativeEndpointCheckpointIO.H"
#include "DarwinDisplacementCurrent.H"
/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "NativePairedBuildCapability.H"
#include "NativeEndpointArithmetic.H"
#include "NativeLongitudinalIncrement.H"
#include "NativeLongitudinalProducerCertificate.H"

#include "Fields.H"
#include "Circuit/CircuitCoupling.H"
#include "ThetaImplicitHybrid.H"
#include "NativeRetainedAcceptance.H"
#include "NativeCandidateStopping.H"
#include "NativeStoppingMaterialOwner.H"
#include "NativeStoppingMaterialSource.H"
#include "Particles/Collision/NativeCollisionTransaction.H"
#include "NativeEndpointField.H"
#include "NativePECPlasma.H"
#include "NativeCoilField.H"
#include "NativeCircuitFieldImpulse.H"
#include "NativeVacuumEndpoint.H"
#include "NativeVacuumConstraint.H"
#include "NativeLongitudinalAitken.H"
#include "NativeSplitAmpere.H"
#include "NativeInstantaneousForce.H"
#include "EulerianDissipation.H"
#include "DarwinLongitudinalSchur.H"
#include "ThermalFixedPointAcceleration.H"
#include "ThermalFiniteFluxResponse.H"
#include "DarwinThermalAdvance.H"
#include "DarwinVacuumJointSolve.H"
#include "DarwinABoundary.H"
#include "DarwinVacuumERecovery.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Python/callbacks.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <ablastr/utils/UsedInputsFile.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <AMReX_VisMF.H>

using warpx::fields::FieldType;
using namespace amrex::literals;

namespace {
// B used by a particle gather needs more physical PMC guards than the
// next Ampere curl alone. With MC shape 3 and order-2 centering, a particle
// next to the wall reads the second B ghost: nodal shape support plus one
// low centering row. Derive that band from the SAME completed A, preserving
// B_static and the gauge-shifted imposed drive without a second B wall law.
// ComputeCurlA checks that A owns the additional stencil row before launch.
amrex::IntVect DarwinParticleCurlGrow ()
{
    auto grow=DarwinPMCCurlGrow(WarpX::field_boundary_lo, WarpX::field_boundary_hi);
    if (EB::enabled()) { return grow; } // EB boundary qualification is separate.
    bool const centered=WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving;
#if AMREX_SPACEDIM == 3
    amrex::IntVect const centering(WarpX::field_centering_nox/2,
        WarpX::field_centering_noy/2, WarpX::field_centering_noz/2);
#elif AMREX_SPACEDIM == 2
    amrex::IntVect const centering(WarpX::field_centering_nox/2, WarpX::field_centering_noz/2);
#else
    amrex::IntVect const centering(WarpX::field_centering_noz/2);
#endif
    int support=1;
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        int const required=centered ? WarpX::nox/2+centering[d] : (WarpX::nox+1)/2;
        support=std::max(support,required);
    }
    // The same width is needed tangentially at a physical/FAB intersection.
    // CurlEvaluationBox completes that local corner without growing a PEC face.
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        if (grow[d]) { grow[d]=std::max(grow[d],support); }
    }
    return grow;
}

// Private R82 qualification switch. Full nonlinear/endpoint acceptance remains active.
bool NativePECMMQualificationEnabled() {
    bool enabled=false;
    amrex::ParmParse("pec_mm_qualification").query("enable",enabled);
    return enabled;
}
bool FillDarwinLongitudinalSchurKappa (amrex::MultiFab& kappa,
    const amrex::MultiFab& rho, HybridPICModel& model, amrex::Real interval)
{
    if(model.UseCompatibleYeeInertia())
        return FillNativeYeeSchurKappa(WarpX::GetInstance(),rho,kappa,interval);
    auto const* pedestal=model.DensityPedestal(0);
    int const mid=rho.nComp()/2;
    amrex::Real const floor=PhysConst::q_e*model.m_n_floor;
    amrex::Real const width=model.m_n_floor_smooth_width*floor;
    amrex::Real const taper=model.m_electron_inertia_floor_taper*floor;
    amrex::Real const coefficient=model.m_include_electron_inertia
        ? PhysConst::epsilon_0*model.m_electron_inertia_mass/
            (PhysConst::q_e*interval*interval) : 0.;
    for (amrex::MFIter mfi(kappa);mfi.isValid();++mfi) {
        auto const out=kappa.array(mfi); auto const raw=rho.const_array(mfi);
        auto const ped=pedestal ? pedestal->const_array(mfi) : amrex::Array4<const amrex::Real>{};
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            amrex::Real const charge=raw(i,j,k,mid);
            amrex::Real const gate=taper>0 ? .5*(1.+std::tanh((charge-floor)/taper)) : 1.;
            out(i,j,k)=charge>0 ? coefficient*gate/HybridSmoothFloor(
                charge+(ped ? ped(i,j,k) : amrex::Real(0)),floor,width) : 0.;
        });
    }
    return true;
}
// TEMPORARY diagnosis instrument (WARPX_EXT_LEDGER): valid-region sums of the
// Bz state and the stored external Bz at each strip/add choreography point.
void ExtLedgerPrint (WarpX* a_wx, const char* a_tag)
{
    if (std::getenv("WARPX_EXT_LEDGER") == nullptr) { return; }
    using ablastr::fields::Direction;
    const amrex::Real bsum =
        a_wx->m_fields.get(FieldType::Bfield_fp, Direction{2}, 0)->sum(0);
    amrex::Real esum = 0.0_rt;
    if (a_wx->m_fields.has(FieldType::hybrid_B_fp_external, Direction{2}, 0)) {
        esum = a_wx->m_fields.get(
            FieldType::hybrid_B_fp_external, Direction{2}, 0)->sum(0);
    }
    amrex::Print() << "EXT_LEDGER[" << a_tag << "] sumBz=" << bsum
                   << " sumBextz=" << esum << "\n";
}
}

namespace warpx::implicit {
struct NativeEndpointCandidateLease::Token {
    bool active=true;
    std::uint64_t generation=1;
    amrex::Real time=0.;
    std::uint64_t epoch=0;
};
struct NativeAcceptedStepCandidate::Impl {
    WarpX* simulation=nullptr;
    ThetaImplicitHybrid* solver=nullptr;
    warpx::thermal::DarwinThermalAdvance* thermal=nullptr;
    std::unique_ptr<warpx::particles::NativeCollisionTransaction> particles;
    MultiReducedDiags::MidStepSnapshot diagnostics;
    std::shared_ptr<NativeEndpointCandidateLease::Token> token;
    amrex::Real start_time=0., endpoint_time=0., interval=0.;
    int step=0;
    bool circuit=false, active=false, endpoint_ready=false;
    warpx::darwin::NativePairedDarwinFields::EndpointReceipt endpoint_pair;
    enum class Phase { Captured, Field, FieldReady, Ready };
    Phase phase=Phase::Captured;
    bool symmetric=false,material_symmetric=false,material_pre_source=false;
    bool MaterialSourcePhase() const {return phase==Phase::Ready||(material_pre_source&&phase==Phase::Captured);}
    std::unique_ptr<warpx::thermal::NativeStoppingSourcePublication> material_pre_publication;
    std::array<amrex::MultiFab,3> material_field_magnetic_origin;
    mutable amrex::Real symmetric_change=0.,symmetric_named=0.,symmetric_error=0.,symmetric_bound=0.;
    mutable amrex::Real symmetric_binding_error=0.,symmetric_binding_bound=0.;
    std::unique_ptr<NativeCandidateStoppingReceipt> stopping;
    std::unique_ptr<warpx::thermal::NativeStoppingMaterialOwner> material_owner;
    std::unique_ptr<warpx::thermal::NativeStoppingMaterialSource> material_source;
    bool material_source_published=false,material_acceptance_requested=false;
    warpx::thermal::NativeStoppingSourcePublication material_publication;
    std::shared_ptr<warpx::thermal::NativeStoppingCarryCertificate const> stopping_carry;
    std::unique_ptr<NativeMidpointStoppingReceipt> pre_source,post_source;
    std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate> longitudinal;
};

bool NativeEndpointCandidateLease::Valid() const noexcept
{
    return m_token && m_token->active && m_generation==m_token->generation;
}
amrex::Real NativeEndpointCandidateLease::EndpointTime() const noexcept
{ return m_token ? m_token->time : 0.; }
std::uint64_t NativeEndpointCandidateLease::EndpointEpoch() const noexcept
{ return m_token ? m_token->epoch : 0; }

NativeAcceptedStepCandidate::NativeAcceptedStepCandidate(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl))
{
    m_impl->token=std::make_shared<NativeEndpointCandidateLease::Token>();
    m_impl->token->time=m_impl->endpoint_time;
    m_impl->token->epoch=static_cast<std::uint64_t>(m_impl->step)+1;
}
NativeAcceptedStepCandidate::NativeAcceptedStepCandidate(NativeAcceptedStepCandidate&&) noexcept=default;
NativeAcceptedStepCandidate::~NativeAcceptedStepCandidate()
{
    // No hidden MPI operation or best-effort rollback in destruction.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!Active(),
        "Destroying an unresolved native endpoint candidate; explicitly Cancel or Finalize it");
    if(m_impl && m_impl->token)m_impl->token->active=false;
}
bool NativeAcceptedStepCandidate::Active() const noexcept
{ return m_impl && m_impl->active; }
NativeEndpointCandidateLease NativeAcceptedStepCandidate::EndpointLease() const
{
    NativeEndpointCandidateLease lease;
    if(Active() && m_impl->endpoint_ready) {
        lease.m_token=m_impl->token;
        lease.m_generation=m_impl->token->generation;
    }
    return lease;
}
bool NativeAcceptedStepCandidate::ValidateLease(NativeEndpointCandidateLease const& lease) const
{
    bool valid=Active() && m_impl->endpoint_ready && lease.Valid() &&
        lease.m_token.get()==m_impl->token.get() &&
        m_impl->simulation->gett_new(0)==m_impl->start_time &&
        m_impl->simulation->getistep(0)==m_impl->step;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid)return false;
    if(m_impl->solver->m_native_endpoint_pair_requested){
        auto* owner=m_impl->solver->m_native_endpoint_pair.get();bool present=owner!=nullptr;
        amrex::ParallelDescriptor::ReduceBoolAnd(present);if(!present)return false;
        return owner->MatchesEndpoint(m_impl->endpoint_pair);
    }
    return true;
}
std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate const>
NativeAcceptedStepCandidate::StoppingProducerCertificate(NativeEndpointCandidateLease const& lease)const
{
    if(!ValidateLease(lease))return {};
    auto const& state=*m_impl;
    bool ready=static_cast<bool>(state.longitudinal);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return {};
    if(!state.longitudinal->Matches(*state.simulation,lease.EndpointTime(),lease.EndpointEpoch()))return {};
    return state.longitudinal;
}
bool NativeAcceptedStepCandidate::CanCancel() const
{
    // First agree that all local owners exist, then execute every owner's
    // collective preflight, even when another local predicate is false.
    bool active=Active();
    amrex::ParallelDescriptor::ReduceBoolAnd(active);
    if(!active)return false;
    auto& state=*m_impl;
    bool const thermal=state.thermal->CanCancel();
    bool valid=state.simulation->gett_new(0)==state.start_time &&
        state.simulation->getistep(0)==state.step && thermal;
    auto const particles=state.particles->CanRestore(*state.simulation);
    bool const diagnostics=state.simulation->reduced_diags->CanRestoreMidStepState(state.diagnostics);
    valid=valid && particles==warpx::particles::NativeCollisionTransaction::Status::Success && diagnostics;
    if(state.circuit)valid=state.solver->NativeCircuitCoupler().RetainedNativeStepCancelable() && valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}
NativeCandidateStatus NativeAcceptedStepCandidate::Cancel()
{
    bool active=Active();
    amrex::ParallelDescriptor::ReduceBoolAnd(active);
    if(!active)return NativeCandidateStatus::Unsupported;
    if(!CanCancel())return NativeCandidateStatus::Terminal;
    auto& state=*m_impl;
    if(state.circuit) {
        bool restored=state.solver->NativeCircuitCoupler().CancelRetainedNativeStep();
        amrex::ParallelDescriptor::ReduceBoolAnd(restored);
        if(!restored)return NativeCandidateStatus::Terminal;
    }
    // No borrowed material view survives the complete particle-map swap.
    state.material_source.reset();
    state.material_owner.reset();
    auto const particles=state.particles->Restore(*state.simulation,{
        [](void* context) noexcept {
            static_cast<warpx::thermal::DarwinThermalAdvance*>(context)->InvalidateBorrowedStepViews();
        },state.thermal});
    if(particles!=warpx::particles::NativeCollisionTransaction::Status::Success)
        return NativeCandidateStatus::Terminal;
    state.thermal->Cancel();
    if(!state.simulation->reduced_diags->RestoreMidStepState(state.diagnostics))
        return NativeCandidateStatus::Terminal;
    state.active=false; state.endpoint_ready=false; state.token->active=false;
    state.solver->m_native_candidate_active=false;
    state.solver->m_native_longitudinal_pending.reset();state.longitudinal.reset();
    return NativeCandidateStatus::Success;
}
bool NativeAcceptedStepCandidate::CanFinalize() const
{
    bool ready=Active() && m_impl->endpoint_ready && m_impl->phase==Impl::Phase::Ready &&
        (!m_impl->material_source || (m_impl->material_source_published&&
            m_impl->material_acceptance_requested&&m_impl->material_publication.acceptance.ready)) &&
        (!m_impl->symmetric || (m_impl->material_symmetric?
            (m_impl->material_pre_publication&&m_impl->material_source&&m_impl->material_source_published):
            (m_impl->pre_source && m_impl->post_source))) &&
        (!m_impl->solver->m_native_endpoint_pair_requested ||
         m_impl->solver->m_native_endpoint_pair_accept_requested);
    if(ready && m_impl->solver->m_native_endpoint_pair_requested)
        ready=warpx::darwin::NativePairedDarwinFields::EndpointRequested()&&
            warpx::darwin::NativePairedDarwinFields::EndpointAcceptanceRequested();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return false;
    // Agree presence before calling a collective source method. This is an
    // acceptance-only preflight, never a residual/Jv or default field probe.
    int source_min=bool(m_impl->material_source),source_max=source_min;
    amrex::ParallelDescriptor::ReduceIntMin(source_min);
    amrex::ParallelDescriptor::ReduceIntMax(source_max);
    if(source_min!=source_max)return false;
    // This calls all rollback-owner checks before any irreversible publication.
    bool valid=CanCancel();
    if(source_min){
        auto checked=m_impl->material_publication;
        bool const source=m_impl->material_source->CheckAcceptance(checked);
        checked.status=source?warpx::thermal::NativeStoppingSourcePublicationStatus::Acceptable:
            warpx::thermal::NativeStoppingSourcePublicationStatus::Declined;
        m_impl->material_publication=std::move(checked);valid=source&&valid;
        if(m_impl->material_symmetric){bool const combined=CheckMaterialSymmetricWork();valid=combined&&valid;}
    }
    if(m_impl->solver->m_native_endpoint_pair_requested){
        bool present=bool(m_impl->solver->m_native_endpoint_pair);
        amrex::ParallelDescriptor::ReduceBoolAnd(present);if(!present)return false;
        bool const endpoint=m_impl->solver->m_native_endpoint_pair->CanAcceptEndpoint(m_impl->endpoint_pair);
        valid=endpoint&&valid;
    }
    if(m_impl->longitudinal) {
        bool const proof=m_impl->longitudinal->Matches(*m_impl->simulation,
            m_impl->endpoint_time,static_cast<std::uint64_t>(m_impl->step)+1);
        valid=proof&&valid;
    }
    if(m_impl->circuit){
        valid=m_impl->solver->NativeCircuitCoupler().RetainedNativeStepFinalizable()&&valid;
        if(m_impl->symmetric)for(auto const* source:{m_impl->pre_source.get(),m_impl->post_source.get()}){
            auto const& receipt=source->physical.circuit_publication;
            valid=receipt.provider_committed&&receipt.fields_published&&receipt.linkage_rebased&&valid;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid;
}
NativeCandidateStatus NativeAcceptedStepCandidate::Finalize()
{
    bool active=Active();
    amrex::ParallelDescriptor::ReduceBoolAnd(active);
    if(!active)return NativeCandidateStatus::Unsupported;
    if(!CanFinalize())return CanCancel() ? NativeCandidateStatus::Declined : NativeCandidateStatus::Terminal;
    auto& state=*m_impl;
    if(state.circuit) {
        bool finalized=state.solver->NativeCircuitCoupler().FinalizeRetainedNativeStep();
        amrex::ParallelDescriptor::ReduceBoolAnd(finalized);
        if(!finalized)return NativeCandidateStatus::Terminal;
    }
    // Copy value receipts before releasing the only source/publication owner.
    // No borrowed particle/field pointer survives this once-only boundary.
    auto const accepted_material=state.material_publication;
    bool const accepted_source=state.material_source&&state.material_acceptance_requested;
    state.material_source.reset();
    state.material_owner.reset();
    // The successful preflight freshly checks the retained spatial n+1
    // constraint and every produced image. Promotion changes ownership only;
    // it does not materialize a high-only field or alter accepted Je/D.
    if(state.solver->m_native_endpoint_pair_requested)
        state.solver->m_native_endpoint_pair->AcceptEndpoint();
    // Preserve the field-stage thermal account before once-only publication.
    warpx::thermal::ThermalEnergyBalance field_balance;
    if(state.symmetric&&!state.material_symmetric)field_balance=state.thermal->CandidateEnergyBalance();
    // No numerical acceptance check follows the circuit's irreversible commit.
    state.thermal->FinalizeCandidate();
    if(accepted_source){
        auto const& r=accepted_material;auto const& a=r.acceptance;auto const& l=r.work.physical;
        auto const& m=r.work.carry.moving_metric;
        amrex::Print()<<std::setprecision(17)<<"Native accepted material source [J]: epoch="<<r.epoch
            <<" time="<<r.time<<" interval="<<r.interval
            <<" field_change="<<a.field_change<<" source_change="<<a.source_change
            <<" combined_change="<<a.combined_change<<" named_work="<<a.named_work
            <<" accounting_error="<<a.accounting_error<<" accounting_bound="<<a.accounting_bound
            <<" source_ion="<<a.source_ion<<" source_electron="<<a.source_electron
            <<" source_internal="<<a.source_internal<<" source_magnetic="<<a.source_magnetic
            <<" magnetic_materialization="<<a.magnetic_materialization
            <<" magnetic_materialization_bound="<<a.magnetic_materialization_bound
            <<" source_inventory_error="<<a.source_inventory_error
            <<" source_inventory_bound="<<a.source_inventory_bound
            <<" field_ion_work_defect="<<a.field_ion_work_defect
            <<" field_magnetic_curl_transfer="<<a.field.magnetic.curl_transfer
            <<" field_magnetic_materialization="<<a.field.magnetic.represented_faraday_work
            <<" field_metric_work="<<m.metric_work<<" field_rotation_work="<<m.rotation_work
            <<" source_transition_work="<<r.work.carry.transition_work
            <<" source_vacuum_work="<<r.work.carry.free_work<<" source_fixed_work="<<r.work.carry.fixed_work
            <<" heat="<<l.heat<<" delivered_heat="<<r.work.thermal.delivered_heat
            <<" spatial_transfer="<<l.spatial_transfer<<" convention="<<l.relativistic_defect
            <<" current_entry_error_A_per_m2="<<a.published_current.maximum_error
            <<" current_entry_bound_A_per_m2="<<a.published_current.maximum_bound
            <<" current_entry_ratio="<<a.published_current.maximum_ratio
            <<" current_entry_rows="<<a.published_current.rows<<"\n";
    }
    if(state.material_symmetric){
        auto const& pre=*state.material_pre_publication;
        amrex::Print()<<std::setprecision(17)<<"Native accepted material symmetric [J]: epoch="<<accepted_material.epoch
            <<" macro_interval="<<state.interval<<" pre_interval="<<pre.interval
            <<" post_interval="<<accepted_material.interval<<" pre_time="<<pre.time<<" post_time="<<accepted_material.time
            <<" pre_change="<<pre.acceptance.source_change<<" field_change="<<accepted_material.acceptance.field_change
            <<" post_change="<<accepted_material.acceptance.source_change
            <<" pre_heat="<<pre.work.physical.heat<<" pre_delivered_heat="<<pre.work.thermal.delivered_heat
            <<" post_heat="<<accepted_material.work.physical.heat<<" post_delivered_heat="<<accepted_material.work.thermal.delivered_heat
            <<" pre_vacuum_work="<<pre.work.carry.accepted_work
            <<" pre_current_ratio="<<pre.acceptance.published_current.maximum_ratio
            <<" post_current_ratio="<<accepted_material.acceptance.published_current.maximum_ratio
            <<" pre_spatial_transfer="<<pre.work.physical.spatial_transfer
            <<" post_spatial_transfer="<<accepted_material.work.physical.spatial_transfer
            <<" combined_change="<<state.symmetric_change<<" named_work="<<state.symmetric_named
            <<" accounting_error="<<state.symmetric_error<<" accounting_bound="<<state.symmetric_bound
            <<" binding_error="<<state.symmetric_binding_error<<" binding_bound="<<state.symmetric_binding_bound<<"\n";
    }
    if(state.stopping) {
        auto const& r=*state.stopping;
        auto const& l=r.work;
        amrex::Print()<<std::setprecision(17)<<"Native accepted stopping [J]: epoch="<<r.endpoint_epoch
            <<" time="<<r.endpoint_time<<" interval="<<r.interval
            <<" ion="<<l.ion_energy_change<<" electron_bulk="<<l.electron_energy_change
            <<" magnetic="<<l.magnetic_energy_change<<" heat="<<l.heat
            <<" actual_defect="<<l.actual_energy_defect<<" accounting_defect="<<l.accounting_defect
            <<" spatial_transfer="<<l.spatial_transfer<<" convention="<<l.relativistic_defect
            <<" displacement="<<l.displacement_work<<" conductor="<<l.conductor_work
            <<" curl_transfer="<<l.curl_work<<" transpose="<<l.transpose_work
            <<" heat_transfer="<<l.heat_transfer_error<<" arithmetic_bound="<<l.arithmetic_bound
            <<" constraint_work="<<l.constraint_work<<" mean_current_work="<<l.mean_current_work<<"\n";
    }
    if(state.symmetric&&!state.material_symmetric) {
        for(auto const* source:{state.pre_source.get(),state.post_source.get()}) {
            auto const& r=source->physical;auto const& l=r.work;auto const& t=source->thermal;
            amrex::Print()<<std::setprecision(17)<<"Native accepted midpoint source [J]: epoch="<<r.endpoint_epoch
                <<" time="<<r.endpoint_time<<" interval="<<r.interval
                <<" ion="<<l.ion_energy_change<<" electron_bulk="<<l.electron_energy_change
                <<" magnetic="<<l.magnetic_energy_change<<" heat="<<l.heat
                <<" actual_defect="<<l.actual_energy_defect<<" accounting_defect="<<l.accounting_defect
                <<" spatial_transfer="<<l.spatial_transfer<<" convention="<<l.relativistic_defect
                <<" displacement="<<l.displacement_work<<" conductor="<<l.conductor_work
                <<" constraint_work="<<l.constraint_work<<" mean_current_work="<<l.mean_current_work
                <<" curl_transfer="<<l.curl_work<<" transpose="<<l.transpose_work
                <<" heat_transfer="<<l.heat_transfer_error<<" arithmetic_bound="<<l.arithmetic_bound
                <<" signed_U_residual="<<t.signed_residual<<" U_residual_bound="<<t.residual_bound
                <<" delivered_heat="<<t.delivered_heat<<" restricted_heat="<<t.restricted_heat
                <<" pc_projection_calls="<<t.pc_projection_calls<<" pc_projection_iterations="<<t.pc_projection_iterations
                <<" physical_projection_calls="<<t.physical_projection_calls
                <<" physical_projection_iterations="<<t.physical_projection_iterations
                <<" pc_applications="<<t.pc_applications
                <<" pc_temperature_probes="<<t.pc_temperature_probes
                <<" pc_temperature_actions="<<t.pc_temperature_actions
                <<" pc_temperature_projection_calls="<<t.pc_temperature_projection_calls
                <<" pc_temperature_projection_iterations="<<t.pc_temperature_projection_iterations
                <<" pc_temperature_projection_residual="<<t.pc_temperature_projection_residual
                <<" pc_temperature_freeze_point_visits="<<t.pc_temperature_freeze_point_visits
                <<" pc_temperature_action_point_visits="<<t.pc_temperature_action_point_visits<<"\n";
            if(state.circuit){
                auto const& c=r.circuit_work;auto const& p=r.circuit_publication;
                amrex::Print()<<std::setprecision(17)<<"Native accepted circuit source [J]: epoch="<<r.endpoint_epoch
                    <<" time="<<r.endpoint_time<<" application="<<p.application
                    <<" network_change="<<p.network_change<<" coil_magnetic_change="<<c.coil_magnetic_change
                    <<" provider_port_work="<<p.provider.port_work<<" port_rounding_work="<<p.port_rounding_work
                    <<" coil_curl_work="<<c.coil_curl_work<<" reciprocity_work="<<c.reciprocity_work
                    <<" wall_work="<<c.wall_work<<" inventory_change="<<p.inventory_change
                    <<" named_work="<<p.named_work<<" accounting_defect="<<p.accounting_defect
                    <<" arithmetic_bound="<<p.arithmetic_bound<<" linkage_rebased="<<p.linkage_rebased<<"\n";
            }
        }
    }
    if(state.symmetric&&!state.material_symmetric) {
        auto const& pre=*state.pre_source;auto const& post=*state.post_source;
        auto const& f=field_balance;
        auto const delivered=pre.thermal.delivered_heat+post.thermal.delivered_heat;
        auto const collision=pre.physical.work.heat+post.physical.work.heat;
        amrex::Print()<<std::setprecision(17)<<"Native accepted symmetric thermal balance [J]: "
            <<"field_old="<<f.initial<<" field_new="<<f.final
            <<" pre_delivered_heat="<<pre.thermal.delivered_heat
            <<" post_delivered_heat="<<post.thermal.delivered_heat
            <<" collision_heat="<<collision<<" field_source="<<f.source
            <<" conduction="<<f.conduction<<" advection="<<f.advection<<" compression="<<f.compression
            <<" composed_change="<<f.final-f.initial+delivered
            <<" physical_defect="<<f.defect+delivered-collision
            <<" signed_U_residual="<<pre.thermal.signed_residual+post.thermal.signed_residual
            <<" heat_map_error="<<pre.thermal.heat_map_error+post.thermal.heat_map_error
            <<" storage_error="<<pre.thermal.storage_error+post.thermal.storage_error<<"\n";
    }
    if(state.longitudinal)state.solver->m_native_longitudinal_accepted=state.longitudinal;
    state.solver->m_native_longitudinal_pending.reset();
    state.particles->Commit();
    state.simulation->reduced_diags->CommitMidStepState(state.diagnostics);
    state.simulation->DiscardSavedImplicitParticleState();
    state.active=false; state.endpoint_ready=false; state.token->active=false;
    state.solver->m_native_candidate_active=false;
    return NativeCandidateStatus::Success;
}
NativeCandidateStatus NativeAcceptedStepCandidate::InvalidateEndpointLease()
{
    bool ready=Active() && m_impl->endpoint_ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return NativeCandidateStatus::Unsupported;
    m_impl->material_source.reset();
    m_impl->material_owner.reset();
    m_impl->endpoint_ready=false;
    ++m_impl->token->generation;
    return NativeCandidateStatus::Success;
}


std::shared_ptr<warpx::thermal::NativeStoppingCarryCertificate const>
NativeAcceptedStepCandidate::StoppingCarryCertificate(NativeEndpointCandidateLease const& lease) const {
    if(!ValidateLease(lease))return {};
    return m_impl->stopping_carry;
}
warpx::thermal::NativeStoppingOwnerResult NativeAcceptedStepCandidate::CaptureStoppingMaterial(
    NativeEndpointCandidateLease const& lease,
    warpx::thermal::NativeStoppingOwnerOptions const& options)
{
    using namespace warpx::thermal;
    NativeStoppingOwnerResult result;
    bool const endpoint=ValidateLease(lease);
    bool const cancellable=CanCancel();
    if(!endpoint||!cancellable){result.reason="material capture requires this cancellable endpoint";return result;}
    auto& state=*m_impl;auto& sim=*state.simulation;auto& model=*sim.get_pointer_HybridPICModel();
    bool supported=false;
#if defined(WARPX_DIM_RZ)
    supported=state.phase==Impl::Phase::Ready&&!state.symmetric&&!state.stopping&&!state.circuit&&
        !sim.get_pointer_CircuitCoupling()&&!state.solver->m_native_paired_requested&&
        !state.solver->m_use_mass_matrices_jacobian&&!state.solver->m_use_mass_matrices_pc&&
        state.solver->NativeRetainedScopeSupported()&&!NativePrescribedDriveEnabled()&&
        !model.m_has_external_current&&!model.m_density_pedestal&&!model.DensityPedestal(0)&&
        !model.m_end_region.holmstrom&&model.m_electron_inertia_mass==PhysConst::m_e&&
        sim.Geom(0).isPeriodic(1)&&!model.m_darwin_checkpoint_restored&&
        std::isfinite(state.interval)&&state.interval>0.&&
        !options.fast_species.empty()&&options.fast_species.size()<256&&
        std::isfinite(options.coulomb_log)&&options.coulomb_log>0.&&
        std::isfinite(options.proper_speed_cap)&&options.proper_speed_cap>0.&&
        options.proper_speed_cap<.01*PhysConst::c;
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported){result.reason="pure material owner requires source-free periodic RZ, MM off, no circuit/paired/restart";return result;}
    amrex::Real low[2]={options.coulomb_log,options.proper_speed_cap},high[2]={low[0],low[1]};
    amrex::ParallelDescriptor::ReduceRealMin(low,2);amrex::ParallelDescriptor::ReduceRealMax(high,2);
    int length=static_cast<int>(options.fast_species.size());
    amrex::ParallelDescriptor::Bcast(&length,1,amrex::ParallelDescriptor::IOProcessorNumber());
    std::string fast=options.fast_species;fast.resize(length);
    amrex::ParallelDescriptor::Bcast(fast.data(),length,amrex::ParallelDescriptor::IOProcessorNumber());
    bool agreement=fast==options.fast_species&&low[0]==high[0]&&low[1]==high[1];
    amrex::ParallelDescriptor::ReduceBoolAnd(agreement);
    if(!agreement){result.reason="material physical options differ across ranks";return result;}
    if(state.stopping_carry){
        bool const ready=state.stopping_carry->ValidateSource(sim,state.endpoint_time,static_cast<std::uint64_t>(state.step)+1);
        if(!ready){
            result.status=NativeStoppingOwnerStatus::Invalid;
            result.reason=state.stopping_carry->Report().status==StoppingCarryStatus::DeclinedTransition
                ? "instantaneous empty material rows are not contained in certified field vacuum"
                : "field producer did not supply a live stopping carry certificate";
            return result;
        }
    }
    if(state.stopping_carry&&state.stopping_carry->Report().mode==NativeStoppingCarryMode::FieldTransition){
        result.reason="transition carry requires the owned live source residual and endpoint reclosure";return result;
    }
    // This first endpoint representation supports the real transition census
    // above and whole-step Cancel. A T=0 surprise cannot expose a source action
    // until accepted paired histories/gather and source reclosure are qualified.
    if(state.solver->m_native_endpoint_pair_requested){
        result.reason="pending paired endpoint has no accepted source consumer";return result;
    }
    auto const& rho=*sim.m_fields.get(FieldType::rho_fp,0);
    AcceptedStoppingOptions material;
    material.number_density_floor=model.m_n_floor;material.reference_number_density=model.m_n0_ref;
    material.model_electron_mass=model.m_electron_inertia_mass;material.ghosts=rho.nGrowVect();
    material.field_lo=WarpX::field_boundary_lo;material.field_hi=WarpX::field_boundary_hi;
    material.particle_lo=WarpX::particle_boundary_lo;material.particle_hi=WarpX::particle_boundary_hi;
    material.levels=sim.finestLevel()+1;material.azimuthal_modes=WarpX::ncomps;
    material.shape_order=WarpX::nox;material.embedded_boundary=EB::enabled();
    material.moving_window=sim.getdo_moving_window();
    material.galilean=std::any_of(sim.m_v_galilean.begin(),sim.m_v_galilean.end(),
        [](amrex::Real x){return x!=0.;});
    material.filter=WarpX::use_filter;material.current_centering=sim.do_current_centering;
    material.single_precision_communications=WarpX::do_single_precision_comms;
    material.galerkin_interpolation=WarpX::galerkin_interpolation;
    // Only this explicit branch allocates the source owner. Its private source
    // time is the lease time; the actual global clock and step remain unchanged.
    state.material_source.reset();
    state.material_owner.reset();
    state.material_owner=std::unique_ptr<NativeStoppingMaterialOwner>(new NativeStoppingMaterialOwner(
        sim,material,options,state.endpoint_time,static_cast<std::uint64_t>(state.step)+1,.5*state.interval,state.stopping_carry));
    return state.material_owner->Capture();
}
bool NativeAcceptedStepCandidate::CaptureStoppingMaterialSource(
    NativeEndpointCandidateLease const& lease,
    warpx::thermal::NativeStoppingSourceOptions const& source_options)
{
    using namespace warpx::thermal;
    auto const& options=source_options.physical;
    bool const endpoint=ValidateLease(lease);
    bool const cancellable=CanCancel();
    if(!endpoint||!cancellable)return false;
    auto& state=*m_impl;auto& sim=*state.simulation;auto& model=*sim.get_pointer_HybridPICModel();
    bool supported=false;
#if defined(WARPX_DIM_RZ)
    supported=state.MaterialSourcePhase()&&(!state.symmetric||state.material_symmetric)&&!state.stopping&&!state.circuit&&
        !state.material_owner&&!state.material_source&&state.solver->m_native_endpoint_pair_requested&&
        state.solver->m_native_endpoint_pair_accept_requested&&
        !sim.get_pointer_CircuitCoupling()&&!state.solver->m_native_paired_requested&&
        !state.solver->m_use_mass_matrices_jacobian&&!state.solver->m_use_mass_matrices_pc&&
        state.solver->NativeRetainedScopeSupported()&&!NativePrescribedDriveEnabled()&&
        !model.m_has_external_current&&!model.m_density_pedestal&&!model.DensityPedestal(0)&&
        !model.m_end_region.holmstrom&&model.m_electron_inertia_mass==PhysConst::m_e&&
        sim.Geom(0).isPeriodic(1)&&!model.m_darwin_checkpoint_restored&&
        std::isfinite(state.interval)&&state.interval>0.&&
        !options.fast_species.empty()&&options.fast_species.size()<256&&
        std::isfinite(options.coulomb_log)&&options.coulomb_log>0.&&
        std::isfinite(options.proper_speed_cap)&&options.proper_speed_cap>0.&&
        options.proper_speed_cap<.01*PhysConst::c;
#if defined(AMREX_USE_GPU)
    supported=supported&&warpx::darwin::NativeEndpointCudaQualificationSelected(sim);
#endif
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported)return false;
    amrex::Real low[2]={options.coulomb_log,options.proper_speed_cap},high[2]={low[0],low[1]};
    amrex::ParallelDescriptor::ReduceRealMin(low,2);amrex::ParallelDescriptor::ReduceRealMax(high,2);
    int length=static_cast<int>(options.fast_species.size());
    amrex::ParallelDescriptor::Bcast(&length,1,amrex::ParallelDescriptor::IOProcessorNumber());
    std::string fast=options.fast_species;fast.resize(length);
    amrex::ParallelDescriptor::Bcast(fast.data(),length,amrex::ParallelDescriptor::IOProcessorNumber());
    bool agreement=fast==options.fast_species&&low[0]==high[0]&&low[1]==high[1];
    amrex::ParallelDescriptor::ReduceBoolAnd(agreement);
    if(!agreement)return false;
    int selection_min[4]={int(source_options.endpoint_acceptance),int(source_options.stable_drag_increment),
        int(source_options.refine_current_projection),int(source_options.register_native_theta)};
    int selection_max[4]={selection_min[0],selection_min[1],selection_min[2],selection_min[3]};
    amrex::ParallelDescriptor::ReduceIntMin(selection_min,4);
    amrex::ParallelDescriptor::ReduceIntMax(selection_max,4);
    if(selection_min[0]!=selection_max[0]||selection_min[1]!=selection_max[1]||
       selection_min[2]!=selection_max[2]||selection_min[3]!=selection_max[3])return false;
    auto const source_time=state.material_pre_source?state.start_time:state.endpoint_time;
    auto const source_epoch=static_cast<std::uint64_t>(state.step)+(state.material_pre_source?0:1);
    bool carry=state.stopping_carry&&
        (state.material_pre_source?
         (state.stopping_carry->Report().mode==NativeStoppingCarryMode::AcceptedOrigin&&state.stopping_carry->Report().status==StoppingCarryStatus::AcceptedReady):
         (state.stopping_carry->Report().mode==NativeStoppingCarryMode::FieldTransition&&state.stopping_carry->Report().status==StoppingCarryStatus::TransitionReady));
    amrex::ParallelDescriptor::ReduceBoolAnd(carry);if(!carry)return false;
    if(!state.stopping_carry->ValidateSource(sim,source_time,source_epoch))return false;
    auto const& rho=*sim.m_fields.get(FieldType::rho_fp,0);
    AcceptedStoppingOptions material;
    material.number_density_floor=model.m_n_floor;material.reference_number_density=model.m_n0_ref;
    material.model_electron_mass=model.m_electron_inertia_mass;material.ghosts=rho.nGrowVect();
    material.field_lo=WarpX::field_boundary_lo;material.field_hi=WarpX::field_boundary_hi;
    material.particle_lo=WarpX::particle_boundary_lo;material.particle_hi=WarpX::particle_boundary_hi;
    material.levels=sim.finestLevel()+1;material.azimuthal_modes=WarpX::ncomps;
    material.shape_order=WarpX::nox;material.embedded_boundary=EB::enabled();
    material.moving_window=sim.getdo_moving_window();
    material.galilean=std::any_of(sim.m_v_galilean.begin(),sim.m_v_galilean.end(),
        [](amrex::Real x){return x!=0.;});
    material.filter=WarpX::use_filter;material.current_centering=sim.do_current_centering;
    material.single_precision_communications=WarpX::do_single_precision_comms;
    material.galerkin_interpolation=WarpX::galerkin_interpolation;
    std::shared_ptr<NativeEndpointAmpereOrigin const> origin;
    NativeStoppingFieldWork field;
    if(source_options.endpoint_acceptance){
        origin=state.material_pre_source?
            NativeEndpointAmpereOrigin::CaptureAccepted(sim,source_time,source_epoch,state.token->generation):
            state.thermal->RetainJointVacuumEndpointAmpereOrigin(state.endpoint_time);
        bool present=bool(origin);amrex::ParallelDescriptor::ReduceBoolAnd(present);
        if(!present)return false;
        if(!state.material_pre_source){
            ablastr::fields::ConstVectorField initial{&state.material_field_magnetic_origin[0],
                &state.material_field_magnetic_origin[1],&state.material_field_magnetic_origin[2]};
            if(!state.thermal->CaptureMaterialFieldWork(*origin,field,state.material_symmetric?&initial:nullptr))return false;
        }
    }
    state.material_acceptance_requested=source_options.endpoint_acceptance;
    state.material_source=std::unique_ptr<NativeStoppingMaterialSource>(new NativeStoppingMaterialSource(
        sim,material,source_options,source_time,source_epoch,
        .5*state.interval,state.stopping_carry,std::move(origin),source_options.endpoint_acceptance?&field:nullptr));
    return state.material_source->Capture();
}
bool NativeAcceptedStepCandidate::DefineStoppingMaterialSourceState(
    NativeEndpointCandidateLease const& lease,WarpXSolverVec& x) const {
    bool ready=Active()&&m_impl->MaterialSourcePhase()&&bool(m_impl->material_source)&&
        (!m_impl->material_source_published||!m_impl->material_acceptance_requested);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!ValidateLease(lease))return false;
    return m_impl->material_source->DefineState(x);
}
bool NativeAcceptedStepCandidate::StoppingMaterialSourceResidual(
    NativeEndpointCandidateLease const& lease,WarpXSolverVec& out,WarpXSolverVec const& x) {
    bool ready=Active()&&m_impl->MaterialSourcePhase()&&bool(m_impl->material_source)&&
        (!m_impl->material_source_published||!m_impl->material_acceptance_requested);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!ValidateLease(lease)){m_impl->material_source->Invalidate();return false;}
    return m_impl->material_source->Residual(out,x);
}
bool NativeAcceptedStepCandidate::SolveStoppingMaterialSource(NativeEndpointCandidateLease const& lease) {
    bool ready=Active()&&m_impl->MaterialSourcePhase()&&bool(m_impl->material_source)&&
        (!m_impl->material_source_published||!m_impl->material_acceptance_requested);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    bool const endpoint=ValidateLease(lease),cancel=CanCancel();
    if(!endpoint||!cancel){m_impl->material_source->Invalidate();return false;}
    return m_impl->material_source->Solve();
}
warpx::thermal::NativeStoppingSourceView NativeAcceptedStepCandidate::StoppingMaterialSourceView(
    NativeEndpointCandidateLease const& lease) {
    bool ready=Active()&&m_impl->MaterialSourcePhase()&&bool(m_impl->material_source)&&
        (!m_impl->material_source_published||!m_impl->material_acceptance_requested);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return {};
    if(!ValidateLease(lease)){m_impl->material_source->Invalidate();return {};}
    return m_impl->material_source->View();
}

warpx::thermal::NativeStoppingSourcePublication NativeAcceptedStepCandidate::PublishStoppingMaterialSource(
    NativeEndpointCandidateLease const& lease)
{
    using namespace warpx::thermal;
    using Status=NativeStoppingSourcePublicationStatus;
    NativeStoppingSourcePublication result;
    bool ready=Active()&&m_impl->endpoint_ready&&m_impl->MaterialSourcePhase()&&
        m_impl->material_source&&!m_impl->material_source_published&&
        !m_impl->longitudinal&&!m_impl->circuit&&(!m_impl->symmetric||m_impl->material_symmetric)&&
        m_impl->solver->m_native_endpoint_pair_requested&&
        m_impl->solver->m_native_endpoint_pair_accept_requested&&
        m_impl->token->generation<std::numeric_limits<std::uint64_t>::max();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready){result.reason="provisional source publication unavailable or already consumed";return result;}
    bool const endpoint=ValidateLease(lease),cancellable=CanCancel();
    if(!endpoint||!cancellable){result.reason="provisional source requires the exact cancellable endpoint";return result;}
    auto& state=*m_impl;auto& sim=*state.simulation;auto& model=*sim.get_pointer_HybridPICModel();
    if(!state.material_source->PreparePublication(result)){
        result.status=Status::Declined;result.reason="fresh source publication preparation failed";return result;
    }
    auto decline=[&](char const* reason){
        result.reason=reason;
        auto const cancelled=Cancel();result.cancelled=cancelled==NativeCandidateStatus::Success;
        result.status=result.cancelled?Status::Declined:Status::Terminal;
        return result;
    };
    // Retain the original macrostep snapshot and old/previous pair history.
    // InvalidateEndpointLease() would destroy this source owner, so perform
    // only the same token revocation here before the private source writes.
    state.endpoint_ready=false;++state.token->generation;result.attempted=true;
    state.material_source_published=true;
    auto* paired=state.solver->m_native_endpoint_pair.get();
    if(!paired->BeginEndpointSourceReclosure(state.endpoint_pair))
        return decline("paired source-origin revocation failed");
    if(!state.material_source->PublishPrepared())return decline("material source publication failed");
    result.fields_written=true;
    // Borrowed field-stage closures cannot survive the momentum/U change.
    // This preserves the original rollback snapshot and retained pair history.
    state.thermal->InvalidateBorrowedStepViews();
    model.m_inertia_rho_n_captured=false;model.m_inertia_jpold_captured=false;
    model.m_qdsmc_J_plasma_valid=false;model.ClearElectronThermalTrials();
    auto const accepted=sim.m_fields.get_alldirs(NativeAcceptedStoppingContext::AcceptedCurrentName,0);
    CopyNativeEndpointCurrentMirror(sim,accepted,*sim.m_fields.get("hybrid_Je_n_nodal",0));
    auto const& rho=*sim.m_fields.get(FieldType::rho_fp,0);
    model.RefreshEulerianElectronThermodynamics(0,rho);
    auto const& baseline=state.solver->m_Eold.getArrayVec()[0];
    ablastr::fields::VectorField source_baseline{baseline[0],baseline[1],baseline[2]};
    if(state.material_pre_source){
        warpx::darwin::NativePairedDarwinFields::EndpointHistoryView history;
        if(!paired->EndpointHistory(history))return decline("pre-source paired spatial origin unavailable");
        for(int c=0;c<3;++c)source_baseline[c]=const_cast<amrex::MultiFab*>(history.high[c]);
    }
    if(!TryConstrainNativeVacuumEndpointField(sim,state.material_pre_source?state.start_time:state.endpoint_time,false,
        source_baseline))return decline("source CHIEF paired endpoint reclosure failed");
    auto exact=state.solver->m_E.getArrayVec()[0];
    ablastr::fields::VectorField transverse{exact[0],exact[1],exact[2]};
    if(!paired->CopyEndpointTransverse(transverse))return decline("source exact transverse publication failed");
    amrex::MultiFab density(rho,amrex::make_alias,0,1);
    auto const check=paired->CheckEndpoint(density,*sim.m_fields.get("hybrid_rho_vacmask_fp",0),&transverse);
    result.endpoint_plasma_error=check.plasma_error;
    result.endpoint_divergence_error=check.divergence_error;
    result.endpoint_null_defect=check.null_defect;
    if(!check.compatible||!check.converged)return decline("fresh source endpoint constraint failed");
    if(!paired->PublishEndpointAuxiliary())return decline("source paired auxiliary publication failed");
    if(!state.material_source->CheckPublication(result))return decline("source endpoint changed held inventories or population");
    if(!CanCancel())return decline("source endpoint lost whole-step cancellation");
    state.endpoint_pair=paired->EndpointLease();state.endpoint_ready=true;
    result.endpoint_reclosed=true;result.status=Status::Provisional;
    if(state.material_acceptance_requested){
        if(!state.material_source->CheckAcceptance(result))
            return decline("fresh source current or combined F/source work acceptance failed");
        result.status=Status::Acceptable;
    }
    state.material_publication=result;
    // The default path remains provisional. The additive path still requires
    // all fresh pair/current/work and whole-Cancel checks in CanFinalize.
    return result;
}

NativeCandidateStatus NativeAcceptedStepCandidate::ApplyMaterialHalf(
    bool pre,NativeSymmetricStoppingOptions const& options)
{
    using namespace warpx::thermal;
    auto& state=*m_impl;auto& sim=*state.simulation;auto* pair=state.solver->m_native_endpoint_pair.get();
    bool ready=Active()&&state.material_symmetric&&options.material_support&&pair&&
        !state.material_source&&!state.material_owner&&
        state.token->generation<std::numeric_limits<std::uint64_t>::max()-4&&
        (pre?state.phase==Impl::Phase::Captured:state.phase==Impl::Phase::Ready);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready){amrex::Print()<<"Material S"<<(pre?1:2)<<" declined at phase entry\n";return NativeCandidateStatus::Unsupported;}
    auto decline=[&](char const* reason){
        amrex::Print()<<"Material S"<<(pre?1:2)<<" declined at "<<reason<<"\n";
        return Cancel()==NativeCandidateStatus::Success?NativeCandidateStatus::Declined:NativeCandidateStatus::Terminal;
    };
    state.material_pre_source=pre;
    if(pre){
        if(!pair->BeginEndpointPreSource())return decline("accepted endpoint origin");
        state.endpoint_pair=pair->EndpointLease();state.endpoint_ready=true;
        state.token->time=state.start_time;state.token->epoch=static_cast<std::uint64_t>(state.step);
        state.stopping_carry=NativeStoppingCarryCertificate::CaptureAcceptedOrigin(
            sim,state.start_time,state.token->epoch,state.token->generation);
        bool present=bool(state.stopping_carry);amrex::ParallelDescriptor::ReduceBoolAnd(present);
        if(!present)return decline("accepted current carry");
    }
    NativeStoppingSourceOptions source;source.physical={options.physical.fast_species,
        options.physical.coulomb_log,options.physical.proper_speed_cap};
    source.relative_convention_budget=options.physical.relative_convention_budget;source.thermal=options.thermal;
    source.endpoint_acceptance=true;source.stable_drag_increment=true;
    source.refine_current_projection=true;source.register_native_theta=true;
    auto const lease=EndpointLease();
    if(!CaptureStoppingMaterialSource(lease,source))return decline("source capture");
    if(!SolveStoppingMaterialSource(lease))return decline("source solve");
    auto receipt=PublishStoppingMaterialSource(lease);
    if(receipt.status!=NativeStoppingSourcePublicationStatus::Acceptable){
        if(!Active())return receipt.cancelled?NativeCandidateStatus::Declined:NativeCandidateStatus::Terminal;
        return decline("source publication");
    }
    if(pre){
        if(!pair->FinishEndpointPreSource(state.endpoint_pair))return decline("post-source field origin");
        state.material_pre_publication=std::make_unique<NativeStoppingSourcePublication>(receipt);
        state.material_source.reset();state.stopping_carry.reset();state.material_source_published=false;
        state.material_acceptance_requested=false;state.material_publication={};
        state.material_pre_source=false;state.endpoint_ready=false;++state.token->generation;
        state.token->time=state.endpoint_time;state.token->epoch=static_cast<std::uint64_t>(state.step)+1;
        for(int c=0;c<3;++c){auto const& in=*sim.m_fields.get(FieldType::Bfield_fp,ablastr::fields::Direction{c},0);
            auto& out=state.material_field_magnetic_origin[c];out.define(in.boxArray(),in.DistributionMap(),in.nComp(),in.nGrowVect());
            amrex::MultiFab::Copy(out,in,0,0,in.nComp(),in.nGrowVect());}
        state.phase=Impl::Phase::Field;
    }else if(!CheckMaterialSymmetricWork())return decline("combined source/field work");
    return NativeCandidateStatus::Success;
}
bool NativeAcceptedStepCandidate::CheckMaterialSymmetricWork() const
{
    using namespace warpx::thermal;auto& s=*m_impl;
    bool ready=s.material_symmetric&&!s.material_pre_source&&s.material_pre_publication&&
        s.material_pre_publication->acceptance.ready&&s.material_publication.acceptance.ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    auto const& pre=*s.material_pre_publication;auto const& post=s.material_publication;
    auto const& a=pre.acceptance;auto const& b=post.acceptance;auto const& f=b.field;
    auto const& m=post.work.carry.moving_metric;
    ready=!a.includes_field&&b.includes_field&&f.available&&m.available&&
        pre.time==s.start_time&&post.time==s.endpoint_time&&pre.interval==.5*s.interval&&
        post.interval==.5*s.interval&&pre.epoch==std::uint64_t(s.step)&&post.epoch==std::uint64_t(s.step)+1;
    amrex::Real field_ion_initial=0.;
    for(auto const& d:f.ions.diagnostics)field_ion_initial+=d[IonElectricWorkDiagnostic::OldKineticEnergy];
    amrex::Real const expected_ion=pre.work.physical.ion_initial_energy+a.source_ion;
    amrex::Real const expected_electron=pre.work.physical.electron_initial_energy+a.source_electron;
    amrex::Real const expected_magnetic=pre.work.physical.magnetic_initial_energy+a.source_magnetic;
    amrex::Real const scale=std::abs(field_ion_initial)+std::abs(expected_ion)+std::abs(m.initial_kinetic)+
        std::abs(expected_electron)+std::abs(f.thermal.initial)+std::abs(a.final_internal)+
        std::abs(f.magnetic.initial)+std::abs(expected_magnetic);
    // Independent phase binding at the represented post-S1 inventories. The
    // operation/reduction allowances are retained from the actual S and F
    // receipts, with only the final four scalar sums added here.
    amrex::Real const eps=std::numeric_limits<amrex::Real>::epsilon(),g=16.*eps/(1.-16.*eps);
    s.symmetric_binding_bound=std::nextafter(a.source_inventory_bound+m.arithmetic_bound+
        f.magnetic.arithmetic_bound+g*scale,std::numeric_limits<amrex::Real>::infinity());
    s.symmetric_binding_error=std::max({std::abs(field_ion_initial-expected_ion),
        std::abs(m.initial_kinetic-expected_electron),std::abs(f.thermal.initial-a.final_internal),
        std::abs(f.magnetic.initial-expected_magnetic)});
    s.symmetric_change=a.source_change+b.combined_change;s.symmetric_named=a.named_work+b.named_work;
    s.symmetric_error=s.symmetric_change-s.symmetric_named;
    s.symmetric_bound=std::nextafter(a.accounting_bound+b.accounting_bound+
        g*(std::abs(a.source_change)+std::abs(b.combined_change)+std::abs(a.named_work)+std::abs(b.named_work)),
        std::numeric_limits<amrex::Real>::infinity());
    ready=ready&&std::isfinite(s.symmetric_binding_bound)&&std::isfinite(s.symmetric_bound)&&
        std::isfinite(s.symmetric_binding_error)&&std::isfinite(s.symmetric_error)&&
        s.symmetric_binding_error<=s.symmetric_binding_bound&&std::abs(s.symmetric_error)<=s.symmetric_bound;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);return ready;
}

warpx::thermal::NativeStoppingSourcePublication NativeAcceptedStepCandidate::StoppingMaterialAcceptance(
    NativeEndpointCandidateLease const& lease) const
{
    warpx::thermal::NativeStoppingSourcePublication result;
    bool ready=Active()&&m_impl->endpoint_ready&&m_impl->material_source_published&&
        m_impl->material_acceptance_requested&&bool(m_impl->material_source);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready||!ValidateLease(lease))return result;
    bool const accepted=CanFinalize();
    result=m_impl->material_publication;
    if(!accepted){result.acceptance.ready=false;result.source_finalization_available=false;
        result.status=warpx::thermal::NativeStoppingSourcePublicationStatus::Declined;}
    return result;
}

warpx::thermal::NativeStoppingOwnerResult NativeAcceptedStepCandidate::EvaluateStoppingMaterial(
    warpx::thermal::NativeStoppingOwnerLease const& lease,
    warpx::thermal::NativeStoppingOwnerTrial const& trial)
{
    using namespace warpx::thermal;
    NativeStoppingOwnerResult result;result.status=NativeStoppingOwnerStatus::Stale;
    bool ready=Active()&&m_impl->endpoint_ready&&m_impl->phase==Impl::Phase::Ready&&
        static_cast<bool>(m_impl->material_owner);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready){result.reason="retained material owner unavailable";return result;}
    bool const endpoint=ValidateLease(EndpointLease());bool const cancel=CanCancel();
    if(!endpoint||!cancel){m_impl->material_owner->Invalidate();result.reason="retained material phase expired";return result;}
    return m_impl->material_owner->Evaluate(lease,trial);
}
warpx::thermal::NativeStoppingOwnerView NativeAcceptedStepCandidate::StoppingMaterialView(
    warpx::thermal::NativeStoppingOwnerLease const& lease)
{
    bool ready=Active()&&m_impl->endpoint_ready&&m_impl->phase==Impl::Phase::Ready&&
        static_cast<bool>(m_impl->material_owner);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return {};
    if(!ValidateLease(EndpointLease())){m_impl->material_owner->Invalidate();return {};}
    return m_impl->material_owner->View(lease);
}

NativeCandidateStoppingResult NativeAcceptedStepCandidate::ApplyStopping(
    NativeEndpointCandidateLease const& lease,NativeCandidateStoppingOptions const& options)
{
    return ApplySource(SourcePhase::FixedTemperaturePost,&lease,options,nullptr);
}

NativeCandidateStoppingResult NativeAcceptedStepCandidate::ApplySource(
    SourcePhase phase,NativeEndpointCandidateLease const* lease,
    NativeCandidateStoppingOptions const& options,
    warpx::thermal::NativeStoppingThermalOptions const* thermal_options)
{
    using namespace warpx::thermal;
    using ablastr::fields::Direction;
    NativeCandidateStoppingResult result;
    bool endpoint_source_allowed=Active()&&!m_impl->solver->m_native_endpoint_pair_requested;
    amrex::ParallelDescriptor::ReduceBoolAnd(endpoint_source_allowed);
    if(!endpoint_source_allowed){result.reason="pending paired endpoint has no accepted source consumer";return result;}
    bool const midpoint=phase!=SourcePhase::FixedTemperaturePost;
    bool const before=phase==SourcePhase::MidpointPre;
    // Source phases are private to the owned advance; neither intermediate
    // state can publish a lease or pass CanFinalize.
    bool phase_valid=false;
    if(!midpoint) {
        NativeEndpointCandidateLease const empty;
        phase_valid=ValidateLease(lease ? *lease : empty);
    }
    else {
        phase_valid=Active() && m_impl->symmetric && thermal_options && !m_impl->endpoint_ready &&
            (before ? m_impl->phase==Impl::Phase::Captured && !m_impl->pre_source
                    : m_impl->phase==Impl::Phase::FieldReady && m_impl->pre_source && !m_impl->post_source);
        amrex::ParallelDescriptor::ReduceBoolAnd(phase_valid);
    }
    bool const cancellable=CanCancel();
    if(!phase_valid || !cancellable) {
        result.reason="Stopping requires this active validated endpoint lease and complete rollback";
        return result;
    }
    auto& state=*m_impl;
    bool current_scope=state.thermal && !state.thermal->HasResidualRemainder();
    amrex::ParallelDescriptor::ReduceBoolAnd(current_scope);
    if(!current_scope) {
        result.reason="Thermal current remainder does not support native stopping events";
        return result;
    }
    auto& sim=*state.simulation;
    auto& model=*sim.get_pointer_HybridPICModel();
    bool const circuit_source=midpoint&&state.circuit;
    bool supported=false;
#if defined(WARPX_DIM_RZ)
    auto const& names=sim.GetPartContainer().GetSpeciesNames();
    bool const driven_source=circuit_source&&state.solver->m_circuit_native&&
        warpx::darwin::NativeCoilCurrentEnabled()&&
        state.solver->NativeCircuitCoupler().SupportsSourceImpulse()&&
        !state.solver->m_use_mass_matrices_jacobian&&!state.solver->m_use_mass_matrices_pc;
    supported=!state.stopping && (midpoint || !state.symmetric) &&
        (driven_source || (!state.circuit && !sim.get_pointer_CircuitCoupling())) &&
        !NativeVacuumEndpointEnabled() && !NativePrescribedDriveEnabled() &&
        warpx::darwin::NativePECPlasmaEnabled() && NativeFullOhmLongitudinalEnabled() &&
        state.solver->NativeRetainedScopeSupported() &&
        ((!state.solver->m_use_mass_matrices_jacobian && !state.solver->m_use_mass_matrices_pc) || NativePECMMQualificationEnabled()) &&
        (!model.m_add_external_fields || driven_source) && !model.m_has_external_current &&
        !model.m_density_pedestal && !model.DensityPedestal(0) &&
        model.m_electron_inertia_mass==PhysConst::m_e && sim.Geom(0).isPeriodic(1) &&
        std::find(names.begin(),names.end(),options.fast_species)!=names.end() &&
        !options.fast_species.empty() && options.fast_species.size()<256 &&
        std::isfinite(options.coulomb_log) && options.coulomb_log>0. &&
        std::isfinite(options.proper_speed_cap) && options.proper_speed_cap>0. &&
        options.proper_speed_cap<.01*PhysConst::c &&
        std::isfinite(options.relative_convention_budget) && options.relative_convention_budget>0.;
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported) {
        result.reason="Stopping bridge requires positive fixed-density periodic RZ PEC and finite native options; circuit midpoint sources require the original native transaction and MM off";
        return result;
    }
    // The typed caller must supply the same model parameters on all ranks.
    amrex::Real lower[3]={options.coulomb_log,options.proper_speed_cap,options.relative_convention_budget};
    amrex::Real upper[3]={lower[0],lower[1],lower[2]};
    amrex::ParallelDescriptor::ReduceRealMin(lower,3);
    amrex::ParallelDescriptor::ReduceRealMax(upper,3);
    int length=static_cast<int>(options.fast_species.size());
    amrex::ParallelDescriptor::Bcast(&length,1,amrex::ParallelDescriptor::IOProcessorNumber());
    std::string species=options.fast_species;species.resize(length);
    amrex::ParallelDescriptor::Bcast(species.data(),length,amrex::ParallelDescriptor::IOProcessorNumber());
    supported=species==options.fast_species;
    for(int i=0;i<3;++i)supported=supported && lower[i]==upper[i];
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported) { result.reason="Stopping options differ across ranks";return result; }

    std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate const> certificate;
    amrex::Real const source_time=before ? state.start_time : state.endpoint_time;
    std::uint64_t const source_epoch=static_cast<std::uint64_t>(state.step)+(before ? 0 : 1);
    if(state.solver->m_native_stopping_certificate) {
        if(midpoint) {
            bool present=static_cast<bool>(state.longitudinal);
            amrex::ParallelDescriptor::ReduceBoolAnd(present);
            if(present && state.longitudinal->Matches(sim,source_time,source_epoch))certificate=state.longitudinal;
        } else certificate=StoppingProducerCertificate(*lease);
        if(!certificate) { result.reason="Stopping producer certificate is missing, stale or unsupported";return result; }
    }
    auto& rho=*sim.m_fields.get(FieldType::rho_fp,0);
    AcceptedStoppingOptions context_options;
    context_options.number_density_floor=model.m_n_floor;
    context_options.reference_number_density=model.m_n0_ref;
    context_options.model_electron_mass=model.m_electron_inertia_mass;
    context_options.ghosts=rho.nGrowVect();
    context_options.field_lo=WarpX::field_boundary_lo;context_options.field_hi=WarpX::field_boundary_hi;
    context_options.particle_lo=WarpX::particle_boundary_lo;context_options.particle_hi=WarpX::particle_boundary_hi;
    context_options.levels=sim.finestLevel()+1;context_options.azimuthal_modes=WarpX::ncomps;
    context_options.shape_order=WarpX::nox;context_options.embedded_boundary=EB::enabled();
    context_options.moving_window=sim.getdo_moving_window();
    context_options.galilean=std::any_of(sim.m_v_galilean.begin(),sim.m_v_galilean.end(),
        [](amrex::Real x){return x!=0.;});
    context_options.filter=WarpX::use_filter;context_options.current_centering=sim.do_current_centering;
    context_options.single_precision_communications=WarpX::do_single_precision_comms;
    context_options.galerkin_interpolation=WarpX::galerkin_interpolation;
    auto context=std::make_unique<NativeAcceptedStoppingContext>(
        model.ElectronThermalGeometry(),sim.boxArray(0),sim.DistributionMap(0),context_options);
    auto binding=NativeAcceptedStoppingContext::Bind(sim.m_fields,false,
        source_time,source_epoch);
    std::unique_ptr<NativeRZSpatialStoppingEvent> event;
    std::unique_ptr<NativeRZMidpointStoppingEvent> thermal_event;
    std::unique_ptr<warpx::darwin::NativeCircuitFieldImpulse> circuit_action;
    auto decline=[&](std::string reason) {
        // Drop all particle/context views before the whole-map swap. The
        // helper's fixed-topology rollback must never run after this restore.
        thermal_event.reset();event.reset();circuit_action.reset();context.reset();
        result.reason=std::move(reason);
        result.status=Cancel()==NativeCandidateStatus::Success
            ? NativeCandidateStatus::Declined : NativeCandidateStatus::Terminal;
        return result;
    };
    if(midpoint && !CheckNativeAcceptance(sim,before ? AcceptancePhase::BeforePreSource : AcceptancePhase::BeforePostSource))
        return decline("Symmetric source declined before preparation");
    bool prepared=context->Prepare(sim.m_fields,binding);
    amrex::ParallelDescriptor::ReduceBoolAnd(prepared);
    if(!prepared)return decline(context->Failure());
    SpatialStoppingOptions physical;
    physical.fast_species=options.fast_species;physical.coulomb_log=options.coulomb_log;
    physical.proper_speed_cap=options.proper_speed_cap;
    physical.relative_convention_budget=options.relative_convention_budget;
    physical.interval=midpoint ? .5*state.interval : state.interval;
    SpatialStoppingFields fields;
    RZStoppingBackground background;
    background.longitudinal_certificate=certificate;
    for(int c=0;c<3;++c) {
        fields.potential[c]=sim.m_fields.get("hybrid_A_fp",Direction{c},0);
        fields.magnetic[c]=sim.m_fields.get(FieldType::Bfield_fp,Direction{c},0);
        fields.displacement[c]=sim.m_fields.get("diagnostic_D_endpoint_fp",Direction{c},0);
        background.conductor_current[c]=sim.m_fields.get(warpx::darwin::PECWallCurrentName,Direction{c},0);
    }
    auto const& temperature=*sim.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
    auto& energy=*sim.m_fields.get(FieldType::hybrid_electron_energy_fp,0);
    NativeStoppingThermalReceipt thermal_receipt;
    if(midpoint) {
        if(circuit_source){
            auto& coupler=state.solver->NativeCircuitCoupler();
            circuit_action=std::make_unique<warpx::darwin::NativeCircuitFieldImpulse>(sim,coupler);
            auto const circuit_phase=before ? CircuitCoupler::SourceImpulsePhase::PreField
                                           : CircuitCoupler::SourceImpulsePhase::PostField;
            std::string error;
            if(!circuit_action->Prepare(source_time,circuit_phase,error))return decline(error);
            NativeStoppingCircuitBinding circuit_binding{circuit_action.get(),source_time,
                before ? NativeStoppingCircuitBinding::Phase::PreField : NativeStoppingCircuitBinding::Phase::PostField};
            thermal_event=std::make_unique<NativeRZMidpointStoppingEvent>(sim,*context,
                temperature,energy,physical,fields,background,circuit_binding,*thermal_options);
        }else{
            thermal_event=std::make_unique<NativeRZMidpointStoppingEvent>(sim,*context,
                temperature,energy,physical,fields,background,*thermal_options);
        }
        if(!thermal_event->Prepare())return decline(thermal_event->Failure());
        result.receipt.work=thermal_event->Ledger();result.receipt.background=thermal_event->BackgroundLedger();
        thermal_receipt=thermal_event->ThermalReceipt();
        if(circuit_source)result.receipt.circuit_work=thermal_event->CircuitWork();
    } else {
        event=std::make_unique<NativeRZSpatialStoppingEvent>(sim,*context,
            temperature,energy,physical,fields,background);
        if(!event->Prepare())return decline(event->Failure());
        result.receipt.work=event->Ledger();result.receipt.background=event->BackgroundLedger();
    }
    result.receipt.endpoint_time=source_time;
    result.receipt.interval=physical.interval;result.receipt.endpoint_epoch=source_epoch;
    if(midpoint && !CheckNativeAcceptance(sim,before ? AcceptancePhase::AfterPreSourcePrepare : AcceptancePhase::AfterPostSourcePrepare))
        return decline("Symmetric source declined after preparation");
    bool const source_work_valid=circuit_source ?
        NativeRZMidpointStoppingEvent::ValidateCircuitWork(result.receipt.work,result.receipt.circuit_work,physical) :
        NativeRZSpatialStoppingEvent::ValidateLedger(result.receipt.work,physical);
    if(!source_work_valid)
        return decline("Stopping source ledger failed before commit");
    // No accepted storage is changed before every outstanding lease is revoked.
    if(midpoint)++state.token->generation;
    else if(InvalidateEndpointLease()!=NativeCandidateStatus::Success)
        return decline("Stopping lease invalidation failed");
    struct Invalidate {
        NativeAcceptedStoppingContext* context;
        DarwinThermalAdvance* thermal;
        HybridPICModel* model;
    } invalidate{context.get(),state.thermal,&model};
    auto const accepted=sim.m_fields.get_alldirs(NativeAcceptedStoppingContext::AcceptedCurrentName,0);
    NativeRZSpatialStoppingEvent::InvalidationHook hook{[](void* pointer) noexcept {
        auto& x=*static_cast<Invalidate*>(pointer);
        x.context->Invalidate();x.thermal->InvalidateBorrowedStepViews();
        x.model->m_inertia_rho_n_captured=false;x.model->m_inertia_jpold_captured=false;
        x.model->m_qdsmc_J_plasma_valid=false;
    },&invalidate};
    bool const committed=circuit_source ? thermal_event->CommitCircuitOnce(accepted,hook) :
        midpoint ? thermal_event->CommitOnce(accepted,hook) : event->CommitOnce(accepted,hook);
    if(circuit_source)result.receipt.circuit_publication=thermal_event->CircuitPublication();
    if(!committed)return decline(midpoint ? thermal_event->Failure() : event->Failure());
    if(midpoint && !CheckNativeAcceptance(sim,before ? AcceptancePhase::AfterPreSourceCommit : AcceptancePhase::AfterPostSourceCommit))
        return decline("Symmetric source declined after commit");
    if(state.longitudinal) {
        bool const appended=midpoint ? thermal_event->AppendDisplacementCertificate(*certificate,*state.longitudinal)
                                     : event->AppendDisplacementCertificate(*certificate,*state.longitudinal);
        if(!appended)return decline("Stopping displacement producer certificate invalidated");
    }
    // Preserve interval J/Jp, nm1/theta and all stage histories. Only the
    // independently updated accepted Je receives its corresponding n mirror.
    CopyNativeEndpointCurrentMirror(sim,accepted,*sim.m_fields.get("hybrid_Je_n_nodal",0));
    model.RefreshEulerianElectronThermodynamics(0,rho);
    if(!midpoint && !CheckNativeAcceptance(sim,AcceptancePhase::BeforeEndpoint))
        return decline("Stopping candidate declined before endpoint recovery");
    auto const& output=state.solver->m_E.getArrayVec()[0];
    ablastr::fields::VectorField exact_transverse{output[0],output[1],output[2]};
    result.receipt.endpoint=TryConstrainNativeEndpointField(sim,source_time,[&]() {
        NativeInstantaneousCurlRate(sim,source_time,true);
    },&exact_transverse,circuit_source ?
        (before ? NativeEndpointContext::PreSource : NativeEndpointContext::PostSource) :
        NativeEndpointContext::RetainedFieldEnd);
    if(!result.receipt.endpoint)return decline(result.receipt.endpoint.reason);
    sim.UpdateAuxiliaryData();sim.FillBoundaryAux(sim.getngUpdateAux());
    auto const after_phase=midpoint ? (before ? AcceptancePhase::AfterPreSourceReclosure : AcceptancePhase::AfterPostSourceReclosure)
                                    : AcceptancePhase::AfterEndpoint;
    if(!CheckNativeAcceptance(sim,after_phase))
        return decline("Stopping candidate declined after endpoint recovery");
    if(state.longitudinal && !state.longitudinal->Matches(sim,source_time,source_epoch))
        return decline("Stopping endpoint changed a held certified field");
    if(!CanCancel())return decline("Stopping candidate lost its whole-step rollback capability");
    // Internal event rollback is now subsumed by the still-live whole-step
    // owner. No accepted receipt or global time publication has occurred.
    if(midpoint) {
        thermal_event->Finalize();thermal_event.reset();
        auto receipt=std::make_unique<NativeMidpointStoppingReceipt>();
        receipt->physical=result.receipt;receipt->thermal=thermal_receipt;
        if(before)state.pre_source=std::move(receipt);else state.post_source=std::move(receipt);
        state.phase=before ? Impl::Phase::Field : Impl::Phase::Ready;
        state.endpoint_ready=!before;
    } else {
        event->Finalize();event.reset();
        state.stopping=std::make_unique<NativeCandidateStoppingReceipt>(result.receipt);
        state.phase=Impl::Phase::Ready;state.endpoint_ready=true;
    }
    circuit_action.reset();context.reset();
    result.status=NativeCandidateStatus::Success;result.reason="stopping endpoint reclosed";
    return result;
}

NativeStepCandidateResult AdvanceNativeStepCandidate(
    WarpX& simulation, amrex::Real start_time, amrex::Real dt, int step,
    NativeSymmetricStoppingOptions const* symmetric_source,
    warpx::thermal::NativeStoppingCarryRequest const* stopping_carry)
{
    NativeStepCandidateResult result;
    int selection_min=symmetric_source ? (symmetric_source->material_support?2:1) : 0,selection_max=selection_min;
    amrex::ParallelDescriptor::ReduceIntMin(selection_min);
    amrex::ParallelDescriptor::ReduceIntMax(selection_max);
    if(selection_min!=selection_max) {
        result.reason="Symmetric source selection differs across ranks";return result;
    }
    // Encode presence and the explicit policy in the existing two reductions.
    // Default-OFF and the original VacuumResidualOnly policy retain their path.
    int carry_min=stopping_carry?(stopping_carry->mode==warpx::thermal::NativeStoppingCarryMode::VacuumResidualOnly?1:
        stopping_carry->mode==warpx::thermal::NativeStoppingCarryMode::FieldTransition?2:3):0,carry_max=carry_min;
    amrex::ParallelDescriptor::ReduceIntMin(carry_min);amrex::ParallelDescriptor::ReduceIntMax(carry_max);
    if(carry_min!=carry_max){result.reason="Stopping carry request differs across ranks";return result;}
    if(carry_min==3){result.reason="Stopping carry policy is unsupported";return result;}
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    bool supported=solver && NativeRetainedAcceptanceEnabled() &&
        !solver->m_native_candidate_active && std::isfinite(start_time) &&
        std::isfinite(dt) && dt>0. && step>=0 &&
        simulation.gett_new(0)==start_time && simulation.getistep(0)==step;
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported) {
        result.reason="Candidate requires a quiescent retained Theta solver and its unchanged accepted clock";
        return result;
    }
    // The thermal owner can be constructed lazily by OneStepImpl. An existing
    // owner is checked here; the deferred owner is checked before any snapshot.
    supported=!symmetric_source || !solver->m_eulerian_energy ||
        !solver->m_eulerian_energy->HasResidualRemainder();
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported) {
        result.reason="Thermal current remainder does not support native stopping events";
        return result;
    }
    supported=solver->NativePMCJointConfigurationValid(true);
    if(!supported) {
        result.reason="Private PMC joint configuration or actual zero-drive ownership changed";
        return result;
    }
    supported=solver->NativeRetainedScopeSupported() &&
        (!solver->m_circuit_native || solver->NativeCircuitCoupler().SupportsNativeTransaction());
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported) {
        result.reason="Candidate requires the supported source-free native Yee retained scope and native circuit transaction capability";
        return result;
    }
    bool const material_symmetric=symmetric_source&&symmetric_source->material_support;
    bool material_scope=true;
    if(material_symmetric){
        bool const scope=solver->NativeEndpointPairScopeSupported();
        material_scope=scope&&solver->m_native_endpoint_pair_requested&&solver->m_native_endpoint_pair_accept_requested&&
            solver->m_native_endpoint_pair&&solver->m_native_endpoint_pair->EndpointAccepted()&&
            !solver->m_hybrid_pic_model->m_darwin_checkpoint_restored&&stopping_carry&&
            stopping_carry->mode==warpx::thermal::NativeStoppingCarryMode::FieldTransition;
    }
    supported=!symmetric_source || (material_symmetric?material_scope:
        (solver->m_native_stopping_certificate&&solver->NativeStoppingCertificateScopeSupported()&&solver->m_theta==.5));
#if defined(AMREX_USE_GPU)
    supported=supported&&(!material_symmetric||
        warpx::darwin::NativeEndpointCudaQualificationSelected(simulation));
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported) { result.reason="Symmetric source requires the initialized midpoint producer scope";return result; }
    bool carry_joint_vacuum=false;
    if(stopping_carry) {
        // The thermal owner is normally constructed inside OneStepImpl. For
        // its first step use the same false default and query as its deferred
        // constructor; an existing owner's selection remains authoritative.
        int owner_min=solver->m_eulerian_energy ? 1 : 0,owner_max=owner_min;
        amrex::ParallelDescriptor::ReduceIntMin(owner_min);
        amrex::ParallelDescriptor::ReduceIntMax(owner_max);
        if(owner_min!=owner_max) {
            result.reason="Stopping carry thermal-owner presence differs across ranks";return result;
        }
        if(owner_min)carry_joint_vacuum=solver->m_eulerian_energy->JointVacuumRequested();
        else amrex::ParmParse("implicit_evolve.thermal").query("joint_vacuum",carry_joint_vacuum);
        int joint_min=carry_joint_vacuum ? 1 : 0,joint_max=joint_min;
        amrex::ParallelDescriptor::ReduceIntMin(joint_min);
        amrex::ParallelDescriptor::ReduceIntMax(joint_max);
        if(joint_min!=joint_max) {
            result.reason="Stopping carry joint-vacuum selection differs across ranks";return result;
        }
    }
    supported=!stopping_carry||((!symmetric_source||material_symmetric)&&!solver->m_circuit_native&&!solver->m_native_paired_fields&&
        solver->m_theta==.5&&((!solver->m_use_mass_matrices_jacobian&&!solver->m_use_mass_matrices_pc)||
            (!symmetric_source&&solver->NativeEndpointPairMassMatrixScopeSupported()))&&
        simulation.Geom(0).isPeriodic(1)&&(!solver->m_hybrid_pic_model->m_darwin_checkpoint_restored||
            solver->m_native_endpoint_restart_validated)&&
        solver->m_hybrid_pic_model->UsesEulerianElectronEnergy()&&carry_joint_vacuum);
#if !defined(WARPX_DIM_RZ)
    supported=supported&&!stopping_carry;
#endif
    supported=supported&&(!stopping_carry||
        stopping_carry->mode!=warpx::thermal::NativeStoppingCarryMode::FieldTransition||
        (solver->m_native_endpoint_pair_accept_requested&&solver->m_native_endpoint_pair&&
         warpx::darwin::NativePairedDarwinFields::EndpointRequested()&&
         warpx::darwin::NativePairedDarwinFields::EndpointAcceptanceRequested()));
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported){result.reason="Stopping carry capture requires source-free periodic RZ joint vacuum, theta=1/2, no circuit/MM/finite pair or unvalidated restart; transition provenance also requires the accepted endpoint pair owner";return result;}
    result.implicit_exit_status=solver->OneStepImpl(start_time,dt,step,&result.candidate,symmetric_source,stopping_carry);
    if(result.implicit_exit_status<0) {
        result.status=NativeCandidateStatus::Declined;
        result.reason="Native advance declined and restored before candidate publication";
    } else if(result.candidate && result.candidate->Active()) {
        result.status=NativeCandidateStatus::Success;
    } else {
        result.status=NativeCandidateStatus::Terminal;
        result.reason="Native advance returned without the required retained candidate";
    }
    return result;
}
} // namespace warpx::implicit

ThetaImplicitHybrid::ThetaImplicitHybrid() = default;
ThetaImplicitHybrid::~ThetaImplicitHybrid()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_candidate_active,
        "Theta solver destroyed with an unresolved native endpoint candidate");
}

void ThetaImplicitHybrid::Define ( WarpX* const a_WarpX, const bool a_from_restart )
{
    BL_PROFILE("ThetaImplicitHybrid::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "ThetaImplicitHybrid object is already defined!");

    m_WarpX = a_WarpX;
    m_num_amr_levels = 1;

    // Bounded qualification: ordinary endpoints keep their existing path.
    // A selected certificate must succeed; there is no fallback after failure.
    amrex::ParmParse("endpoint_diagnostic").query(
        "owned_endpoint_ampere",m_native_owned_endpoint_ampere_requested);
    int ampere_min=m_native_owned_endpoint_ampere_requested?1:0,ampere_max=ampere_min;
    amrex::ParallelDescriptor::ReduceIntMin(ampere_min);
    amrex::ParallelDescriptor::ReduceIntMax(ampere_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ampere_min==ampere_max,
        "Owned endpoint Ampere selector must agree on all ranks");
#if !defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_owned_endpoint_ampere_requested,
        "Owned endpoint Ampere qualification currently requires precise CPU RZ");
#endif

    amrex::ParmParse("pmc_joint_qualification").query("enable",m_native_pmc_joint_requested);
    amrex::ParmParse("pmc_joint_qualification").query("thermal",m_native_pmc_thermal_requested);
    amrex::ParmParse("pmc_joint_qualification").query("mass_matrix",m_native_pmc_mm_requested);
    amrex::ParmParse("pmc_joint_qualification").query("finite_conduction",m_native_pmc_finite_conduction_requested);
    int finite_min=m_native_pmc_finite_conduction_requested?1:0,finite_max=finite_min;
    amrex::ParallelDescriptor::ReduceIntMin(finite_min);
    amrex::ParallelDescriptor::ReduceIntMax(finite_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finite_min==finite_max,
        "Private PMC finite-conduction selector must agree on all ranks");
    int mm_min=m_native_pmc_mm_requested ? 1 : 0,mm_max=mm_min;
    amrex::ParallelDescriptor::ReduceIntMin(mm_min);
    amrex::ParallelDescriptor::ReduceIntMax(mm_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mm_min==mm_max,
        "Private PMC mass-matrix selector must agree on all ranks");
    int thermal_min=m_native_pmc_thermal_requested ? 1 : 0,thermal_max=thermal_min;
    amrex::ParallelDescriptor::ReduceIntMin(thermal_min);
    amrex::ParallelDescriptor::ReduceIntMax(thermal_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(thermal_min==thermal_max,
        "Private PMC thermal selector must agree on all ranks");
    int pmc_min=m_native_pmc_joint_requested ? 1 : 0,pmc_max=pmc_min;
    amrex::ParallelDescriptor::ReduceIntMin(pmc_min);
    amrex::ParallelDescriptor::ReduceIntMax(pmc_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pmc_min==pmc_max,
        "Private PMC joint selector must agree on all ranks");
    // Publish the immutable dispatch receipt only after all three startup
    // selectors agree, and before any selector-dependent allocation. A later
    // solver replacement may rebind the same value, never toggle this scope.
    auto const selected=m_native_pmc_mm_requested ? NativePrivatePMCMassMatrixSelection::On
                                                 : NativePrivatePMCMassMatrixSelection::Off;
    bool binding=a_WarpX->m_private_pmc_mm_selection==NativePrivatePMCMassMatrixSelection::Unbound ||
        a_WarpX->m_private_pmc_mm_selection==selected;
    auto const generation=a_WarpX->m_private_pmc_mm_owner_generation;
    bool const generation_ok=generation<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max());
    amrex::Long generation_low=generation_ok ? static_cast<amrex::Long>(generation) : 0;
    amrex::Long generation_high=generation_low;
    amrex::ParallelDescriptor::ReduceLongMin(generation_low);
    amrex::ParallelDescriptor::ReduceLongMax(generation_high);
    binding=binding && generation_ok && generation_low==generation_high;
    amrex::ParallelDescriptor::ReduceBoolAnd(binding);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(binding,
        "Private PMC mass-matrix selection is immutable for the simulation lifetime");
    a_WarpX->m_private_pmc_mm_selection=selected;
    a_WarpX->m_private_pmc_mm_owner_generation=generation+1;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_pmc_thermal_requested || m_native_pmc_joint_requested,
        "Private PMC thermal qualification requires the private joint owner");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_pmc_mm_requested || m_native_pmc_joint_requested,
        "Private PMC mass-matrix qualification requires the private joint owner");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_pmc_finite_conduction_requested ||
        (m_native_pmc_joint_requested && m_native_pmc_thermal_requested),
        "Private PMC finite-conduction material requires the joint thermal owner");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_pmc_joint_requested || !a_from_restart,
        "Private PMC joint qualification requires fresh initialization");
    amrex::ParmParse("endpoint_diagnostic").query("stopping_producer_certificate",m_native_stopping_certificate);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_stopping_certificate || !a_from_restart,
        "Stopping producer certificate requires fresh native initialization; restart proof is unavailable");

    int paired_low=warpx::darwin::NativePairedDarwinFields::Requested()?1:0,paired_high=paired_low;
    amrex::ParallelDescriptor::ReduceIntMin(paired_low);
    amrex::ParallelDescriptor::ReduceIntMax(paired_high);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(paired_low==paired_high,
        "Positive paired finite-stage selection must agree on every rank");
    m_native_paired_requested=paired_low!=0;
    bool endpoint_mm=false;
    amrex::ParmParse("endpoint_diagnostic").query("retained_endpoint_pair_mm",endpoint_mm);
    int endpoint_low=(warpx::darwin::NativePairedDarwinFields::EndpointRequested()?1:0)|
        (warpx::darwin::NativePairedDarwinFields::EndpointAcceptanceRequested()?2:0)|
        (endpoint_mm?4:0);
    int endpoint_high=endpoint_low;
    amrex::ParallelDescriptor::ReduceIntMin(endpoint_low);amrex::ParallelDescriptor::ReduceIntMax(endpoint_high);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(endpoint_low==endpoint_high,
        "Retained endpoint pair selection must agree on every rank");
    m_native_endpoint_pair_requested=(endpoint_low&1)!=0;
    m_native_endpoint_pair_accept_requested=(endpoint_low&2)!=0;
    m_native_endpoint_pair_mm_requested=(endpoint_low&4)!=0;
    if(m_native_endpoint_pair_mm_requested) {
        bool supported=false;
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
        supported=m_native_endpoint_pair_requested&&m_native_endpoint_pair_accept_requested&&
            m_native_owned_endpoint_ampere_requested&&warpx::darwin::NativeEndpointArithmeticSupported();
#endif
        amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "Accepted paired MM qualification requires precise CPU RZ and captured pair/accept/owned-Ampere selectors");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_endpoint_pair_accept_requested||m_native_endpoint_pair_requested,
        "Accepted retained endpoint consumers require the retained endpoint pair owner");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_endpoint_pair_requested||(!m_native_paired_requested&&
            (!a_from_restart||m_native_endpoint_pair_accept_requested)),
        "Retained endpoint restart requires accepted consumers and excludes positive paired finite-stage mode");
#if defined(AMREX_USE_GPU)
    // All selection bits were agreed above, before constrained initialization.
    // The actual endpoint owner is allocated only after that initialization.
    if(m_native_endpoint_pair_requested||m_native_owned_endpoint_ampere_requested){
        bool supported=m_native_endpoint_pair_requested&&m_native_endpoint_pair_accept_requested&&
            m_native_owned_endpoint_ampere_requested&&!m_native_paired_requested&&
            warpx::darwin::NativeEndpointArithmeticSupported();
        amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "CUDA retained endpoint qualification requires the captured accepted pair and owned Ampere selectors with an audited precise RZ CUDA build");
    }
#endif
#if defined(__FAST_MATH__) || \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_paired_requested,
        "Positive paired arithmetic requires a pinned precise CPU build; GPU or fast-math is unsupported");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_endpoint_pair_requested,
        "Retained endpoint pair requires a pinned precise CPU build; GPU or fast-math is unsupported");
#elif defined(AMREX_USE_GPU)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_paired_requested ||
        warpx::darwin::NativePairedPreciseCudaBuild,
        "Positive paired arithmetic requires a pinned precise CPU build; GPU or fast-math is unsupported");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_paired_requested || !a_from_restart,
        "Positive paired finite-stage arithmetic currently requires fresh initialization");
    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitHybrid solver requires hybrid PIC model to be defined");

    m_darwin = m_hybrid_pic_model->m_darwin;
    if (a_from_restart && (m_darwin || m_hybrid_pic_model->m_include_electron_inertia)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx::darwin::NativeCoilCurrentEnabled(),
            "Driven PEC current restart awaits validated free transverse origin reconstruction");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin &&
            m_hybrid_pic_model->UsesEulerianElectronEnergy() &&
            m_hybrid_pic_model->m_darwin_checkpoint_restored,
            "Implicit Darwin/inertia restart requires validated Eulerian field/history checkpoint state");
        m_darwin_initialized=true;
    }

    // Vacuum vector-potential recovery cadence (see HybridPICModel.H):
    // "half" applies inside every residual evaluation at the theta-stage
    // field and once at the end-of-step state; "full" end-of-step only.
    m_vacuum_recovery = m_darwin && m_hybrid_pic_model->m_darwin_vacuum_recovery;
    m_vacuum_recovery_half = m_vacuum_recovery
        && (m_hybrid_pic_model->m_darwin_vacuum_recovery_cadence == "half");

    // External vector-potential fields use the split-field convention here,
    // like the explicit scheme: the solver state carries the plasma fields
    // (OneStep strips B_ext/E_ext at entry; Bfield_fp holds totals between
    // steps for diagnostics), the external flux advances analytically from
    // A(t) so wall boundary conditions cannot exclude the programmed coil
    // flux, and the Ohm kernels subtract the inductive E_ext from plasma
    // cells (inside the plasma the generalized Ohm's law IS the electric
    // field). Clearing m_external_split only tells the kernels that the
    // Hall-term field they receive is already the total.
    if (m_hybrid_pic_model->m_add_external_fields) {
        if (m_darwin) {
            // Unified drive: the external vector potential enters through
            // the boundary values of the evolved A (DarwinApplyABoundary);
            // the kernels and the split-field machinery stay out of it.
            m_hybrid_pic_model->m_external_unified = true;
        } else {
            m_hybrid_pic_model->m_external_split = false;
        }
    }

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    if (m_vacuum_recovery) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_darwin_vacrec_relax_time == 0._rt,
            "Spatial vacuum electric recovery requires instantaneous magnetic recovery");
    }
#endif

    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );
    m_Eprev.Define( m_E );
    if (m_native_endpoint_pair_requested) {
        // Inactive histories have defined numeric storage for accepted IO and
        // exact rollback. Keep both readiness flags false; the physical E
        // field still comes from constrained initialization, never this zero.
        m_Eold.zero();
        m_Eprev.zero();
    }

    // Set initial values for E and Eold vectors
    m_E.Copy(FieldType::Efield_fp);
    m_Eold.Copy(a_from_restart ? FieldType::E_old : FieldType::Efield_fp, FieldType::None, true);

    // Define B_old MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& Bfp_x = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
        const auto& dm = Bfp_x->DistributionMap();
        const amrex::IntVect ngb = Bfp_x->nGrowVect();

        for (int dir = 0; dir < 3; ++dir) {
            const auto& ba = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev)->boxArray();
            m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{dir}, lev, ba, dm, 1, ngb, 0.0_rt);
        }
    }

    // Scratch for the resistive push-field correction assembled in every
    // residual evaluation (see ComputeRHS). Only allocated when the opt-in
    // momentum-consistent push field is enabled and a resistive term is
    // configured.
    m_use_resistive_push_correction = m_hybrid_pic_model->HasResistivity()
        && m_hybrid_pic_model->m_implicit_push_excludes_resistive_field;
    if (m_use_resistive_push_correction) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                const auto& Efp = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                m_WarpX->m_fields.alloc_init("hybrid_E_resistive_fp", Direction{dir}, lev,
                    Efp->boxArray(), Efp->DistributionMap(), Efp->nComp(),
                    Efp->nGrowVect(), 0.0_rt);
            }
        }
    }

    // curlcurl_form: the correction's per-evaluation Ohm-solve pair runs
    // BEFORE the inertia assembly that used to allocate the elliptic
    // scratch lazily, so allocate it up front (no-op on e_form). Stale
    // scratch content cancels exactly in the correction's two-pass
    // difference; unallocated arrays do not.
    m_hybrid_pic_model->EnsureCurlCurlScratch();

    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    pp.query("extrapolate_initial_guess", m_extrapolate_initial_guess);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta >= 0.5 && m_theta <= 1.0,
        "theta parameter must be between 0.5 and 1.0");

    // The density response belongs to the hybrid MM Jacobian, not to the
    // electromagnetic MM path. Parse it before selecting the thermal split.
    parseNonlinearSolverParams(pp);
    m_mass_matrices_density_projection = m_use_mass_matrices_jacobian;
    pp.query("mass_matrices_density_projection", m_mass_matrices_density_projection);
    m_continuity_density = WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov;
    m_live_ion_density = m_mass_matrices_density_projection || m_continuity_density;
    m_hybrid_pic_model->m_implicit_continuity_density = m_continuity_density;
    if (m_mass_matrices_density_projection) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_use_mass_matrices_jacobian && !m_particle_suborbits,
            "Hybrid MM density projection requires a MM Jacobian without particle suborbits");
    }
    if (m_live_ion_density) {
        // The radial volume-weighted density/current filters do not yet
        // form a commuting pair with the discrete continuity divergence.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_continuity_density || !WarpX::use_filter,
            "Implicit Esirkepov continuity density currently requires warpx.use_filter=0");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_continuity_density ||
            (WarpX::nox >= 2 && !m_particle_suborbits && m_theta == 0.5_rt &&
             WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving),
            "Implicit conservative density requires shape>=2, momentum gather, theta=0.5, no suborbits");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_num_amr_levels == 1 && !EB::enabled(),
            "Hybrid MM density projection currently requires one grid level without EB");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->m_fields.get(FieldType::rho_fp, 0)->nComp() == 2,
            "Hybrid MM density projection requires a scalar density (m=0 in RZ)");
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (m_WarpX->Geom(0).isPeriodic(d)) { continue; }
            bool axis = false;
#if defined(WARPX_DIM_RZ)
            axis = d == 0 && m_WarpX->Geom(0).ProbLo(0) == 0.0;
#endif
            // Neumann is the PMC alias. Its charge/current deposit folds
            // have the same parity as a reflecting particle wall, including
            // when an absorbing particle boundary is applied at acceptance.
            auto const supported_wall = [](ParticleBoundaryType particles,
                                           FieldBoundaryType fields) {
                return particles == ParticleBoundaryType::Reflecting ||
                    (particles == ParticleBoundaryType::Absorbing &&
                     fields == FieldBoundaryType::PMC);
            };
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (axis || supported_wall(WarpX::particle_boundary_lo[d],
                                        WarpX::field_boundary_lo[d])) &&
                supported_wall(WarpX::particle_boundary_hi[d], WarpX::field_boundary_hi[d]),
                "Hybrid MM density projection requires reflecting/periodic particles, "
                "or absorbing particles with PMC (Neumann) field boundaries");
        }
        m_mass_matrix_density.resize(m_num_amr_levels);
    }

    {
        // Default: re-evaluate the generalized Ohm's law at the delivered
        // end-of-step state. The theta extrapolation of the ALGEBRAIC E is
        // a -(1-theta)/theta recursion on the stored field (marginal at
        // theta = 1/2) whose error grows linearly under a steady drift.
        std::string e_finisher = "reevaluate";
        const bool user_set = pp.query("hybrid_e_finisher", e_finisher);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            e_finisher == "extrapolate" || e_finisher == "reevaluate",
            "implicit_evolve.hybrid_e_finisher must be 'extrapolate' or "
            "'reevaluate'");
        m_e_finisher_reevaluate = (e_finisher == "reevaluate");
        if (m_darwin && m_e_finisher_reevaluate) {
            // Not implemented for the Darwin field split (E_L comes from
            // the ambipolar constraint, E_T from the vector potential);
            // fall back to the legacy extrapolation unless the user asked
            // for the re-evaluated finisher explicitly.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!user_set,
                "implicit_evolve.hybrid_e_finisher = reevaluate is not "
                "implemented for the Darwin field split");
            m_e_finisher_reevaluate = false;
        }
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_hybrid_pic_model->m_include_thermal_conduction || m_darwin ||
            !m_hybrid_pic_model->m_add_external_fields,
        "Implicit thermal conduction with external fields requires unified Darwin fields");

    // Segregated midpoint-iterated solve for the QDSMC electron-energy
    // stage (see the member documentation in the header).
    // Temperature-dependent push and Ohm coefficients must share a frozen
    // thermal stage during every inner residual and Jacobian evaluation.
    bool const eulerian_energy = m_hybrid_pic_model->UsesEulerianElectronEnergy();
    m_qdsmc_segregated_solve = !eulerian_energy && (
        m_hybrid_pic_model->m_resistivity_has_Te_dependence ||
        m_hybrid_pic_model->m_visc_in_ohms_law ||
        (m_live_ion_density && m_hybrid_pic_model->m_solve_electron_energy_equation));
    pp.query("qdsmc_segregated_solve", m_qdsmc_segregated_solve);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!eulerian_energy || !m_qdsmc_segregated_solve,
        "Eulerian electron energy uses its own implicit solve; disable qdsmc_segregated_solve");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(eulerian_energy || !m_live_ion_density ||
        !m_hybrid_pic_model->m_solve_electron_energy_equation || m_qdsmc_segregated_solve,
        "MM density projection with electron energy requires the segregated thermal solve");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        eulerian_energy || !(m_hybrid_pic_model->m_resistivity_has_Te_dependence ||
          m_hybrid_pic_model->m_visc_in_ohms_law) ||
            m_qdsmc_segregated_solve,
        "Temperature-dependent implicit drag requires qdsmc_segregated_solve");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_hybrid_pic_model->m_resistivity_has_Te_dependence ||
          m_hybrid_pic_model->m_visc_in_ohms_law ||
          m_hybrid_pic_model->m_include_thermal_conduction) ||
            m_theta == 0.5_rt,
        "Implicit live-temperature drag and split conduction require "
        "theta=0.5");
    if (m_qdsmc_segregated_solve) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_solve_electron_energy_equation,
            "implicit_evolve.qdsmc_segregated_solve requires "
            "hybrid_pic_model.solve_electron_energy_equation = true");
        pp.query("qdsmc_outer_max_iterations", m_qdsmc_outer_max_iterations);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_qdsmc_outer_max_iterations >= 1,
            "implicit_evolve.qdsmc_outer_max_iterations must be >= 1");
        pp.query("qdsmc_outer_relative_tolerance", m_qdsmc_outer_relative_tolerance);
        pp.query("qdsmc_outer_require_convergence", m_qdsmc_outer_require_convergence);
        pp.query("qdsmc_outer_verbose", m_qdsmc_outer_verbose);
        pp.query("qdsmc_outer_acceleration", m_qdsmc_outer_acceleration);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_qdsmc_outer_acceleration || m_num_amr_levels == 1,
            "Thermal outer acceleration is qualified on one physical level only");
    }

    pp.query("darwin_segregated_solve", m_darwin_segregated_solve);
    if (m_darwin_segregated_solve) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin, "darwin_segregated_solve requires Darwin");
        pp.query("darwin_outer_max_iterations", m_darwin_outer_max_iterations);
        pp.query("darwin_outer_relative_tolerance", m_darwin_outer_rtol);
        pp.query("darwin_outer_absolute_tolerance", m_darwin_outer_atol);
        pp.query("darwin_outer_verbose", m_darwin_outer_verbose);
        pp.query("darwin_outer_relaxation", m_darwin_outer_relaxation);
        pp.query("darwin_outer_relaxation_start", m_darwin_outer_relaxation_start);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_outer_relaxation > 0.0_rt
            && m_darwin_outer_relaxation <= 1.0_rt
            && m_darwin_outer_relaxation_start >= 0,
            "darwin_outer_relaxation must be in (0,1] and its start iteration nonnegative");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_outer_max_iterations > 0
            && m_darwin_outer_rtol >= 0 && m_darwin_outer_atol >= 0
            && (m_darwin_outer_rtol > 0 || m_darwin_outer_atol > 0),
            "Darwin outer iterations and tolerances must be positive");
    }

    pp.query("darwin_vacuum_pc_regularization", m_darwin_vacuum_pc_regularization);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_vacuum_pc_regularization >= 0.0_rt,
        "darwin_vacuum_pc_regularization must be nonnegative (0 disables it)");
    if (m_darwin_vacuum_pc_regularization > 0.0_rt) {
        bool preserve_rows = false;
        amrex::ParmParse("pc_curl_curl_mlmg").query("preserve_dirichlet_rows", preserve_rows);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_hybrid_pic_model->m_include_electron_inertia && preserve_rows,
            "Darwin vacuum PC requires electron inertia and pc_curl_curl_mlmg.preserve_dirichlet_rows=1");
    }

    pp.query("darwin_vacuum_gauge_projection", m_darwin_vacuum_gauge_projection);
    pp.query("darwin_vacuum_gauge_pc_relative_tolerance", m_darwin_vacuum_gauge_pc_rtol);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_vacuum_gauge_pc_rtol > 0.0_rt
        && m_darwin_vacuum_gauge_pc_rtol < 1.0_rt,
        "darwin_vacuum_gauge_pc_relative_tolerance must lie strictly between 0 and 1");
    if (m_darwin_vacuum_gauge_projection) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 && m_darwin_segregated_solve
            && m_darwin_vacuum_pc_regularization > 0.0_rt,
            "darwin_vacuum_gauge_projection requires 3D split and native vacuum PC");
    }

    // Circuit-in-the-residual coupling (see the member documentation in
    // the header).
    pp.query("external_field_iteration", m_external_field_iteration);
    bool native_circuit_geometry = AMREX_SPACEDIM == 3;
    bool default_consistent_stage = AMREX_SPACEDIM == 3;
#if defined(WARPX_DIM_RZ)
    // The native disk probe integrates scalar Bz with cylindrical weights
    // and zero-based indices. Preserve the legacy RZ callback default.
    auto const& circuit_geometry = m_WarpX->Geom(0);
    native_circuit_geometry = WarpX::n_rz_azimuthal_modes == 1 && !EB::enabled()
        && !WarpX::do_moving_window
        && circuit_geometry.Domain().smallEnd() == amrex::IntVect(0)
        && circuit_geometry.ProbLo(0) == 0._rt && !circuit_geometry.isPeriodic(0);
    std::string requested_circuit_driver;
    pp.query("circuit_driver", requested_circuit_driver);
    default_consistent_stage = requested_circuit_driver == "native";
#endif
    m_darwin_circuit_consistent_stage =
        native_circuit_geometry && default_consistent_stage && m_WarpX->maxLevel() == 0
        && m_darwin && m_darwin_segregated_solve && m_external_field_iteration;
    pp.query("darwin_circuit_consistent_stage", m_darwin_circuit_consistent_stage);
    pp.query("darwin_circuit_max_iterations", m_darwin_circuit_max_iterations);
    pp.query("darwin_circuit_scale_tolerance", m_darwin_circuit_scale_tolerance);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_circuit_max_iterations > 0
        && std::isfinite(m_darwin_circuit_scale_tolerance)
        && m_darwin_circuit_scale_tolerance > 0._rt
        && m_darwin_circuit_scale_tolerance < 1._rt,
        "Darwin circuit stage requires positive iteration limit and scale tolerance in (0,1)");
    if (m_darwin_circuit_consistent_stage) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(native_circuit_geometry && m_WarpX->maxLevel() == 0
            && m_darwin && m_darwin_segregated_solve && m_external_field_iteration,
            "darwin_circuit_consistent_stage requires single-level segregated Darwin coupling "
            "in 3D or non-EB, fixed, zero-based, on-axis RZ m=0 geometry");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_vacuum_recovery
            || (m_hybrid_pic_model->m_darwin_vacrec_relax_time == 0._rt
                && m_hybrid_pic_model->m_darwin_vacuum_recovery_frozen_mask),
            "Darwin circuit stage requires instantaneous recovery with a frozen mask");
    }


    // Redistribute ahead of the end-of-step deposits (see the member
    // documentation in the header).
    pp.query("redistribute_before_end_deposits",
             m_redistribute_before_end_deposits);
    if (m_external_field_iteration) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->m_add_external_fields,
            "implicit_evolve.external_field_iteration requires "
            "hybrid_pic_model.add_external_fields");
        if (m_vacuum_recovery) {
            // The recovery's frozen-probe shortcut reuses the correction
            // from the last non-Jacobian evaluation; with the coil scales
            // changing inside every evaluation that would make the probed
            // map inconsistent with the iterate map (the frozen-probe-lag
            // failure class). Force the live-probe recovery; run with a
            // tight darwin_vacuum_recovery_relative_tolerance.
            m_hybrid_pic_model->m_vacuum_recovery_live_probes = true;
        }
    }

    {
        std::string driver = m_darwin_circuit_consistent_stage ? "native" : "python";
        pp.query("circuit_driver", driver);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(driver == "native" || driver == "python",
            "implicit_evolve.circuit_driver must be native or python");
        m_circuit_native = driver == "native";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_circuit_native
            || (native_circuit_geometry && m_WarpX->maxLevel() == 0 && m_darwin
                && m_darwin_segregated_solve && m_external_field_iteration
                && m_darwin_circuit_consistent_stage && sizeof(amrex::Real) == sizeof(double)),
            "Native circuit driver requires single-level double-precision segregated Darwin "
            "with consistent circuit stages in 3D or non-EB, fixed, zero-based, on-axis RZ m=0 geometry");
    }

    m_nlsolver->Define(m_E, this);

    // Resolve the complete selected MM scope collectively before tensor storage
    // or constrained initialization. No endpoint owner exists at this point.
    if(m_native_endpoint_pair_mm_requested) {
        bool supported=NativeEndpointPairMassMatrixScopeSupported();
        amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "Accepted paired MM requires fresh source-free periodic RZ current+density/thermal MM with the original native supports");
    }

    // Preserve the original setup order outside the separate private PMC-MM
    // request. Its complete material/scope preflight precedes tensor storage.
    if (m_use_mass_matrices && !m_native_pmc_mm_requested) { InitializeMassMatrices(); }

    if(m_native_pmc_finite_conduction_requested) {
        // Check the complete owned input grammar before private tensor storage.
        // HPM later compiles its native hard-cap wrapper; bind and parse that
        // actual compiled owner separately, before field initialization.
        using Law=warpx::thermal::finite_flux::MaterialLaw;
        Law parallel,perpendicular;
        auto const& model=*m_hybrid_pic_model;
        bool material=warpx::thermal::remainder::ArithmeticSupported() &&
            warpx::thermal::finite_flux::ParseMaterial(model.m_kappa_par_expression,parallel) &&
            warpx::thermal::finite_flux::ParseMaterial(model.m_kappa_perp_expression,perpendicular);
        if(model.m_cond_chi_max>0.)material=material &&
            parallel.family==Law::PowerFiveHalves && perpendicular.family==Law::PowerFiveHalves;
        amrex::ParallelDescriptor::ReduceBoolAnd(material);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(material,
            "Private PMC finite conduction requires the precise supported owned material grammar");
    }
    if(m_native_pmc_thermal_requested) {
        m_native_pmc_thermal_descriptor=NativePMCJointThermalDescriptor();
        bool length_ok=m_native_pmc_thermal_descriptor.size()<=
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        amrex::ParallelDescriptor::ReduceBoolAnd(length_ok);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(length_ok,"Private PMC material descriptor is too large");
        int length=static_cast<int>(m_native_pmc_thermal_descriptor.size());
        int const root=amrex::ParallelDescriptor::IOProcessorNumber();
        amrex::ParallelDescriptor::Bcast(&length,1,root);
        std::string reference(static_cast<std::size_t>(length),'\0');
        if(amrex::ParallelDescriptor::IOProcessor())reference=m_native_pmc_thermal_descriptor;
        amrex::ParallelDescriptor::Bcast(reference.data(),length,root);
        bool same=reference==m_native_pmc_thermal_descriptor;
        amrex::ParallelDescriptor::ReduceBoolAnd(same);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(same,
            "Private PMC material and thermal boundary descriptor must agree on all ranks");
    }
    // Thermal material executors are compiled later by HybridPICModel::InitData.
    // Construct the thermal advance at first OneStep, after that initialization.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(NativePMCJointConfigurationValid(false),
        "Private PMC joint qualification requires source-free RZ PMC/reflecting caps, "
        "radial PEC, retained all-component Poisson recovery, joint thermal ownership, "
        "MM-Jv/PC0 or the private owner-bound current+density MM capability, "
        "no circuit, paired arithmetic, remainder, restart or end layer");
    if(m_native_pmc_mm_requested) {
        InitializeMassMatrices();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(NativePMCJointConfigurationValid(false),
            "Private PMC mass-matrix setup changed an admitted consumer");
    }
    if(a_from_restart&&m_native_endpoint_pair_requested) {
        // Allocate an empty owner only. InitData restores numerical context
        // after field parsers/stagger caches exist; no lease is granted here.
        m_native_endpoint_pair=std::make_unique<warpx::darwin::NativePairedDarwinFields>(*m_WarpX,
            warpx::darwin::NativePairedDarwinFields::Role::Endpoint,true);
    }
    m_is_defined = true;
}

warpx::darwin::NativeCoilField* ThetaImplicitHybrid::NativeCoilCurrentContext()
{
    if (!warpx::darwin::NativeCoilCurrentEnabled()) return nullptr;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin && NativeCircuitDriveEnabled() &&
        NativeEndpointEnabled() && warpx::darwin::NativePECPlasmaEnabled() && NativeFullOhmLongitudinalEnabled(),
        "Native coil-current convention requires the native Darwin circuit path");
    if (!m_native_coil_field) {
        m_native_coil_field=std::make_unique<warpx::darwin::NativeCoilField>(*m_WarpX);
        auto const e=m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp,0);
        for(int c=0;c<3;++c)m_native_longitudinal_gather[c].define(
            e[c]->boxArray(),e[c]->DistributionMap(),1,e[c]->nGrowVect());
    }
    return m_native_coil_field.get();
}

void ThetaImplicitHybrid::CalculateNativePlasmaCurrent(
    ablastr::fields::MultiLevelVectorField const& total_B,amrex::Real time)
{
    if (auto* coil=NativeCoilCurrentContext()) {
        AMREX_ALWAYS_ASSERT(total_B.size()==1 && std::isfinite(time));
        coil->CalculatePlasmaCurrent(total_B[0],time);
    } else m_hybrid_pic_model->CalculatePlasmaCurrent(total_B,m_WarpX->GetEBUpdateEFlag());
}

void ThetaImplicitHybrid::InitializeFields (amrex::Real time, amrex::Real dt)
{
    if (!m_darwin || m_darwin_initialized) { return; }
    BL_PROFILE("ThetaImplicitHybrid::InitializeFields()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_defined && std::isfinite(dt) && dt > 0.0_rt,
        "Darwin initial fields require a defined solver and positive time interval");
    using ablastr::fields::Direction;
    // HPM InitData has compiled the immutable native parser storage. Bind it
    // before the first physical initialization write, never inside a residual.
    if(m_native_pmc_thermal_requested)BindNativePMCJointThermalExecutors();
    m_dt = dt;

    // This deposits the committed particles without pushing them, initializes
    // Pe/Te, and adds the initial external B once. Esirkepov uses a straight
    // orbit centered at the synchronized particle time for the entry current.
    m_WarpX->HybridPICInitializeRhoJandB();
    auto const B = m_WarpX->m_fields.get_mr_levels_alldirs(
        FieldType::Bfield_fp, m_num_amr_levels - 1);
    auto const E = m_WarpX->m_fields.get_mr_levels_alldirs(
        FieldType::Efield_fp, m_num_amr_levels - 1);
    auto const Ji = m_WarpX->m_fields.get_mr_levels_alldirs(
        FieldType::current_fp, m_num_amr_levels - 1);
    auto const rho = m_WarpX->m_fields.get_mr_levels(
        FieldType::rho_fp, m_num_amr_levels - 1);
    m_WarpX->FillBoundaryB(B[0][0]->nGrowVect(), true);
    CalculateNativePlasmaCurrent(B,time);
    if (m_hybrid_pic_model->m_pec_conductor_wall_rows) {
        // This closure applies moment boundary conditions inside the Ohm
        // wrapper. Establish those rows before seeding Je or either density
        // slot; the final solve below then includes the initial inertia.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_hybrid_pic_model->HybridPICSolveE(E[lev], Ji[lev], B[lev], *rho[lev],
                m_WarpX->GetEBUpdateEFlag()[lev], lev, false, true);
        }
    }
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        // There is one physical density at initialization. The inertia
        // assembly must see that same state in both density slots.
        amrex::MultiFab::Copy(*rho[lev], *rho[lev], 0, rho[lev]->nComp()/2,
                              1, rho[lev]->nGrowVect());
        if (m_vacuum_recovery && m_hybrid_pic_model->m_darwin_vacuum_recovery_frozen_mask) {
            auto& mask = *m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev);
            amrex::MultiFab::Copy(mask, *rho[lev], 0, 0, 1,
                                  amrex::min(mask.nGrowVect(), rho[lev]->nGrowVect()));
        }
        for (int d = 0; d < 3; ++d) {
            auto& fixed = *m_WarpX->m_fields.get("hybrid_B_static_fp", Direction{d}, lev);
            amrex::MultiFab::Copy(fixed, *B[lev][d], 0, 0, fixed.nComp(), fixed.nGrowVect());
        }
    }
    DarwinApplyABoundary(time);
    if (m_hybrid_pic_model->m_include_electron_inertia) {
        m_hybrid_pic_model->InitializeElectronInertiaHistory();
        // Stationary initial history: dJe/dt = drho/dt = 0, while any
        // enabled spatial/advective inertia is evaluated from the t0 moments.
        // No particle, field or clock is advanced by this assembly.
        m_hybrid_pic_model->ComputeElectronInertiaNodal(1.0_rt, dt, false);
    }
    bool const full_longitudinal = NativeFullOhmLongitudinalEnabled();
    if (full_longitudinal) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_segregated_solve,
            "Full Ohm longitudinal projection requires the segregated native constraint");
        auto dissipation = warpx::darwin::MakeNativeInstantaneousDissipation(*m_WarpX);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_hybrid_pic_model->HybridPICSolveE(E[lev], Ji[lev], B[lev], *rho[lev],
                m_WarpX->GetEBUpdateEFlag()[lev], lev, false, true, dissipation.get());
        }
    }
    m_hybrid_pic_model->ComputeDarwinELong(
        rho, time, true, full_longitudinal ? &E : nullptr);
    bool paired_unchanged=warpx::darwin::NativePairedDarwinFields::Requested()==m_native_paired_requested;
    amrex::ParallelDescriptor::ReduceBoolAnd(paired_unchanged);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(paired_unchanged,
        "Positive paired finite-stage selection changed during initialization");
    if(m_native_stopping_certificate) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(NativeStoppingCertificateScopeSupported(),
            "Stopping producer certificate supports only fresh plain-midpoint positive RZ PEC/periodic source-free retained fields");
        m_native_longitudinal_accepted=warpx::darwin::NativeLongitudinalProducerCertificate::Seed(*m_WarpX);
    }
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        if (!full_longitudinal) {
            m_hybrid_pic_model->HybridPICSolveE(E[lev], Ji[lev], B[lev], *rho[lev],
                m_WarpX->GetEBUpdateEFlag()[lev], lev, false, true);
        }
        for (int d = 0; d < 3; ++d) {
            auto const& longitudinal = *m_WarpX->m_fields.get(
                "hybrid_E_long_fp", Direction{d}, lev);
            auto& old_longitudinal = *m_WarpX->m_fields.get(
                "hybrid_E_long_old_fp", Direction{d}, lev);
            amrex::MultiFab::Copy(old_longitudinal, longitudinal, 0, 0,
                                  longitudinal.nComp(), longitudinal.nGrowVect());
            amrex::MultiFab::Subtract(*E[lev][d], longitudinal, 0, 0,
                                      E[lev][d]->nComp(), E[lev][d]->nGrowVect());
        }
    }
    m_WarpX->FillBoundaryE(E[0][0]->nGrowVect(), true);
    m_E.Copy(FieldType::Efield_fp);
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    if (m_vacuum_recovery) {
        auto const& transverse = m_E.getArrayVec()[0];
        RecoverDarwinVacuumE(*m_WarpX, *m_hybrid_pic_model,
            {transverse[0], transverse[1], transverse[2]},
            {transverse[0], transverse[1], transverse[2]}, time, dt);
    } else
#endif
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int d = 0; d < 3; ++d) {
                auto const& longitudinal = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{d}, lev);
                amrex::MultiFab::Add(*E[lev][d], longitudinal, 0, 0,
                                     E[lev][d]->nComp(), E[lev][d]->nGrowVect());
            }
        }
    }
    bool endpoint_unchanged=warpx::darwin::NativePairedDarwinFields::EndpointRequested()==m_native_endpoint_pair_requested&&
        warpx::darwin::NativePairedDarwinFields::EndpointAcceptanceRequested()==m_native_endpoint_pair_accept_requested;
    amrex::ParallelDescriptor::ReduceBoolAnd(endpoint_unchanged);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(endpoint_unchanged,"Retained endpoint pair selection changed during initialization");
    if(m_native_endpoint_pair_requested){
        bool supported=NativeEndpointPairScopeSupported();amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "Retained endpoint pair requires precise fresh native RZm0 axis/PEC joined vacuum MM0, periodic z or the owned CPU PMC scope; no driven/circuit/restart source");
        m_native_endpoint_pair=std::make_unique<warpx::darwin::NativePairedDarwinFields>(*m_WarpX,
            warpx::darwin::NativePairedDarwinFields::Role::Endpoint,m_native_endpoint_pair_accept_requested);
    }
    if (NativeEndpointEnabled()) { DiagnosticConstrainEndpoint(time,true); }
    if(m_native_longitudinal_accepted) {
        bool const zero=m_native_longitudinal_accepted->BindInitialDisplacement(time,m_WarpX->getistep(0));
        bool const same=m_native_longitudinal_accepted->Matches(*m_WarpX,time,m_WarpX->getistep(0));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(zero&&same,"Initial native longitudinal producer is not certifiable");
    }
    if(m_native_paired_requested) {
        bool supported=NativePairedFieldScopeSupported();
        amrex::ParallelDescriptor::ReduceBoolAnd(supported);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "Positive paired arithmetic requires fresh proof-bound positive RZm0 axis/PEC/periodic native full-Ohm Schur MM0; no external/circuit/vacuum/PMC/EB/filter/restart");
        m_native_paired_fields=std::make_unique<warpx::darwin::NativePairedDarwinFields>(*m_WarpX);
    }
    // Full physical E is now ready for diagnostics and the first gather.
    // OneStep takes the transverse old-field snapshot from this same state.
    SaveEoldMultifab();
    m_WarpX->UpdateAuxiliaryData();
    m_WarpX->FillBoundaryAux(m_WarpX->getngUpdateAux());
    if(m_native_endpoint_pair&&m_native_endpoint_pair->InitialEndpointSelected()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_endpoint_pair->CompleteInitialEndpoint(),
            "Initial retained endpoint lost its physical pair, origin or auxiliary publication");
    }
    m_darwin_initialized = true;
    if (m_WarpX->Verbose()) {
        amrex::Print() << "Darwin initial fields: committed moments, E/EL and Je history at t="
                       << time << "; max|E|=" << E[0][0]->norminf() << "/"
                       << E[0][1]->norminf() << "/" << E[0][2]->norminf() << "\n";
    }
}

void ThetaImplicitHybrid::DiagnosticConstrainEndpoint (amrex::Real time, bool initial)
{
    using ablastr::fields::Direction;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(NativePMCJointConfigurationValid(true),
        "Private PMC joint initial endpoint lost its zero-drive or supported owner scope");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin && m_theta==0.5_rt && (!m_circuit_native || NativeCircuitDriveEnabled())
        && (!m_vacuum_recovery || NativeVacuumEndpointEnabled()) && (!m_hybrid_pic_model->m_darwin_checkpoint_restored ||
            NativeEndpointRestartSupported())
        && m_hybrid_pic_model->UsesEulerianElectronEnergy(),
        "Endpoint requires midpoint Eulerian Darwin and an explicitly supported circuit/vacuum map");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!NativeCircuitDriveEnabled() ||
        (m_circuit_native && (!m_vacuum_recovery || NativeVacuumEndpointEnabled()) && NativeCircuitCoupler().DeviceTrials() &&
         NativeCircuitCoupler().SupportsNativeEndpointRate() &&
         (!m_hybrid_pic_model->m_darwin_checkpoint_restored ||
          m_WarpX->get_pointer_CircuitCoupling()->NativeRestartBindingValidated())),
        "Native circuit endpoint requires the closed-rate capability, supported joined vacuum "
        "and validated immutable restart binding");
    if (!NativeCircuitDriveEnabled() && m_hybrid_pic_model->m_add_external_fields) {
        auto& external=*m_hybrid_pic_model->m_external_vector_potential;
        for(int i=0;i<external.nFields();++i) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!external.DeviceDriven(i) && !external.UsesPythonScale(i) &&
                (NativePrescribedDriveEnabled() ? external.HasAnalyticTimeProfile(i) :
                 (external.TimeScale(i,time)==external.TimeScale(i,time+1.) &&
                  external.TimeScale(i,time)==external.TimeScale(i,time-1.))),
                "Endpoint prototype currently requires static prescribed external fields");
        }
    }
    if (NativeVacuumEndpointEnabled()) {
        auto const& baseline=(initial?m_E:m_Eold).getArrayVec()[0];
        auto exact_free=m_E.getArrayVec()[0];
        ConstrainNativeVacuumEndpointField(*m_WarpX,time,initial,
            {baseline[0],baseline[1],baseline[2]},
            m_native_pmc_joint_requested ? &exact_free : nullptr);
        if (NativeCircuitDriveEnabled()) {
            for (int d=0;d<3;++d) amrex::MultiFab::Copy(*m_E.getArrayVec()[0][d],
                *m_WarpX->m_fields.get(NativeVacuumTransverseRecordName(),Direction{d},0),0,0,1,0);
        } else if(initial&&m_native_endpoint_pair&&m_native_endpoint_pair->InitialEndpointSelected()) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_endpoint_pair->CopyEndpointTransverse(m_E.getArrayVec()[0]),
                "Initial retained transverse field must come from its physical pair owner");
        } else if(!m_native_pmc_joint_requested) {
            m_E.Copy(FieldType::Efield_fp);
            for(int d=0;d<3;++d) amrex::MultiFab::Subtract(*m_E.getArrayVec()[0][d],
                *m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{d},0),0,0,1,0);
        }
        return;
    }
    auto exact_free=m_E.getArrayVec()[0];
    ConstrainNativeEndpointField(*m_WarpX,time,initial,[&]() {
        auto A=m_WarpX->m_fields.get_alldirs("hybrid_A_fp",0);
        for(int d=0;d<3;++d) {
            auto const& Et=*m_WarpX->m_fields.get(FieldType::Efield_fp,Direction{d},0);
            amrex::MultiFab::Copy(*A[d],Et,0,0,1,A[d]->nGrowVect());A[d]->mult(-1.);
        }
        // Static imposed drive has zero derivative: native A images are homogeneous.
        DarwinApplyABoundary(time);
        auto Bd=m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp,0);
        m_WarpX->get_pointer_fdtd_solver_fp(0)->ComputeCurlA(Bd,A,
            m_WarpX->GetEBUpdateBFlag()[0],0,
            DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
        m_WarpX->FillBoundaryB(Bd[0]->nGrowVect(),true);
        m_hybrid_pic_model->CalculatePlasmaCurrent(Bd,m_WarpX->GetEBUpdateEFlag()[0],0);
    },warpx::darwin::NativeCoilCurrentEnabled()?&exact_free:nullptr);
    if(!warpx::darwin::NativeCoilCurrentEnabled()) {
        m_E.Copy(FieldType::Efield_fp);
        for(int d=0;d<3;++d) amrex::MultiFab::Subtract(*m_E.getArrayVec()[0][d],
            *m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{d},0),0,0,1,0);
    }
}

void ThetaImplicitHybrid::PrintParameters () const
{
    BL_PROFILE("ThetaImplicitHybrid::PrintParameters()");

    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "-------- THETA IMPLICIT HYBRID PIC SOLVER PARAMETERS ------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "Time-bias parameter theta:           " << m_theta << "\n";
    if (m_qdsmc_segregated_solve) {
        amrex::Print() << "QDSMC segregated solve:              on\n";
        amrex::Print() << "  outer max iterations:              " << m_qdsmc_outer_max_iterations << "\n";
        amrex::Print() << "  outer relative tolerance:          " << m_qdsmc_outer_relative_tolerance << "\n";
        amrex::Print() << "  outer require convergence:         " << (m_qdsmc_outer_require_convergence?"true":"false") << "\n";
    }
    amrex::Print() << "MM density projection:              "
                   << (m_mass_matrices_density_projection ? "on" : "off") << "\n";
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

namespace warpx::darwin {
NativePairedDarwinFields* NativeActivePairedFields(WarpX& simulation) noexcept {
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    auto* owner=solver?solver->m_native_paired_fields.get():nullptr;
    return owner&&owner->Active()?owner:nullptr;
}
NativePairedDarwinFields* NativeEndpointPairedFields(WarpX& simulation) noexcept {
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    return solver?solver->m_native_endpoint_pair.get():nullptr;
}
bool NativeEndpointPairSelected(WarpX& simulation) noexcept {
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    return solver&&solver->m_native_endpoint_pair_requested;
}
bool NativeEndpointPMCQualificationSelected(WarpX& simulation) {
#if defined(WARPX_DIM_RZ) && !defined(AMREX_USE_GPU)
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    return solver&&solver->m_native_pmc_joint_requested&&
        solver->m_native_endpoint_pair_requested&&solver->m_native_endpoint_pair_accept_requested&&
        solver->m_native_owned_endpoint_ampere_requested&&
        NativeEndpointArithmeticSupported()&&solver->NativePMCJointRequestSelectorsCurrent()&&
        solver->NativePMCJointScopeSupported()&&solver->NativeEndpointPairScopeSupported();
#else
    amrex::ignore_unused(simulation);return false;
#endif
}
bool NativeEndpointCudaQualificationSelected(WarpX& simulation) noexcept {
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_GPU)
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    return solver&&solver->m_native_endpoint_pair_requested&&
        solver->m_native_endpoint_pair_accept_requested&&
        solver->m_native_owned_endpoint_ampere_requested&&!solver->m_native_paired_requested&&
        NativeEndpointArithmeticSupported();
#else
    amrex::ignore_unused(simulation);return false;
#endif
}
bool BindNativeAcceptedEndpointBoundary(WarpX& simulation){
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    bool ready=solver&&solver->m_native_endpoint_pair_requested&&
        solver->m_native_endpoint_pair_accept_requested&&solver->m_native_endpoint_pair&&
        !solver->m_native_candidate_active&&NativePairedDarwinFields::EndpointRequested()&&
        NativePairedDarwinFields::EndpointAcceptanceRequested();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    // The owner validates exactly one next-clock boundary and all original
    // population, configuration and produced-field bytes before rebinding.
    // General Redistribute has no receipt here and remains unsupported.
    return solver->m_native_endpoint_pair->BindAcceptedEndpointBoundary();
}
}
bool ThetaImplicitHybrid::NativeEndpointPairMassMatrixScopeSupported() const {
#if !defined(WARPX_DIM_RZ) || defined(AMREX_USE_GPU)
    return false;
#else
    if(!m_native_endpoint_pair_mm_requested || !m_hybrid_pic_model)return false;
    bool requested=false,jacobian=false,pc=false,density=false,companion=false,remainder=false;
    amrex::ParmParse("endpoint_diagnostic").query("retained_endpoint_pair_mm",requested);
    amrex::ParmParse implicit("implicit_evolve");
    implicit.query("use_mass_matrices_jacobian",jacobian);
    implicit.query("use_mass_matrices_pc",pc);
    implicit.query("mass_matrices_density_projection",density);
    implicit.query("darwin_joint_mm_thermal_increment",companion);
    amrex::ParmParse("implicit_evolve.thermal").query("current_remainder",remainder);
    auto const& model=*m_hybrid_pic_model;
    auto const zero=[](std::string const& law){return law=="0"||law=="0.0";};
    bool const zero_conduction=!model.m_include_thermal_conduction ||
        (zero(model.m_kappa_par_expression)&&zero(model.m_kappa_perp_expression));
    bool adiabatic=true;
    for(int d=0;d<AMREX_SPACEDIM;++d)for(int side=0;side<2;++side)
        adiabatic=adiabatic&&model.m_cond_bc[d][side]==0;
    return requested&&m_native_endpoint_pair_requested&&m_native_endpoint_pair_accept_requested&&
        m_native_owned_endpoint_ampere_requested&&warpx::darwin::NativeEndpointArithmeticSupported()&&
        jacobian&&pc&&density&&companion&&!remainder&&
        m_use_mass_matrices&&m_use_mass_matrices_jacobian&&m_use_mass_matrices_pc&&
        m_esirkepov_mass_matrices&&m_mass_matrices_density_projection&&
        m_live_ion_density&&m_continuity_density&&m_num_amr_levels==1&&
        m_mass_matrices_deposit_interval==1&&!m_mass_matrices_reuse_within_step&&
        !m_fused_mass_matrices_deposit&&!m_particle_suborbits&&!m_skip_particle_picard_init&&
        !m_reflect_particles_at_rmax&&!m_mass_matrices_boundary_rows&&m_verify_mm_jvp_step<0&&
        !m_native_pmc_joint_requested&&!m_native_paired_requested&&!m_native_paired_fields&&
        !m_native_stopping_certificate&&!m_circuit_native&&!m_WarpX->get_pointer_CircuitCoupling()&&
        !model.m_has_electron_stopping&&!model.m_end_region.holmstrom&&
        !model.m_darwin_checkpoint_restored&&zero_conduction&&adiabatic&&
        (!m_eulerian_energy || (m_eulerian_energy->JointVacuumRequested()&&
                               !m_eulerian_energy->HasResidualRemainder()))&&
        WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov&&
        m_nlsolver&&m_nlsolver->GetPreconditionerType()==PreconditionerType::pc_hybrid_pic&&
        NativeVacuumMassMatrixSupported(*m_WarpX);
#endif
}
ThetaImplicitHybrid const* NativeEndpointPairMassMatrixOwnerLocal(WarpX& simulation) noexcept {
    auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    return solver&&solver->NativeEndpointPairMassMatrixScopeSupported()?solver:nullptr;
}
bool ThetaImplicitHybrid::NativeEndpointPairScopeSupported()const {
#if !defined(WARPX_DIM_RZ) || defined(__FAST_MATH__) || \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    return false;
#else
#if defined(AMREX_USE_GPU)
    bool arithmetic=warpx::darwin::NativeEndpointCudaQualificationSelected(*m_WarpX);
    amrex::ParallelDescriptor::ReduceBoolAnd(arithmetic);if(!arithmetic)return false;
#endif
    // Call the collective physical scope unconditionally on this selected path.
    bool const native=NativeRetainedScopeSupported();
    bool const mm_scope=m_native_endpoint_pair_mm_requested ? NativeEndpointPairMassMatrixScopeSupported() :
        !m_use_mass_matrices_jacobian&&!m_use_mass_matrices_pc;
    bool joined=false;int retained=0;
    amrex::ParmParse("implicit_evolve.thermal").query("joint_vacuum",joined);
    amrex::ParmParse("endpoint_diagnostic").query("retained_joined_recovery",retained);
    auto const& g=m_WarpX->Geom(0);
    bool pmc=false;
#if !defined(AMREX_USE_GPU)
    pmc=m_native_pmc_joint_requested&&m_native_owned_endpoint_ampere_requested&&
        warpx::darwin::NativeEndpointArithmeticSupported()&&NativePMCJointScopeSupported();
#endif
    return native&&joined&&retained==1&&m_vacuum_recovery&&NativeVacuumEndpointEnabled()&&
        warpx::implicit::NativeRetainedAcceptanceEnabled()&&!m_circuit_native&&!m_native_paired_requested&&
        !m_native_stopping_certificate&&mm_scope&&
        !m_hybrid_pic_model->m_has_external_current&&
        !m_hybrid_pic_model->m_end_region.holmstrom&&!m_hybrid_pic_model->m_electron_inertia_bdf2&&
        !m_extrapolate_initial_guess&&!m_particle_suborbits&&!WarpX::use_filter&&!EB::enabled()&&
        !m_WarpX->do_current_centering&&!WarpX::do_single_precision_comms&&m_WarpX->finestLevel()==0&&
        WarpX::ncomps==1&&sizeof(amrex::Real)==sizeof(double)&&WarpX::nox==3&&WarpX::noz==3&&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving&&
        WarpX::field_centering_nox==2&&WarpX::field_centering_noz==2&&
        g.ProbLo(0)==0.&&!g.isPeriodic(0)&&(g.isPeriodic(1)||pmc)&&
        WarpX::field_boundary_lo[0]==FieldBoundaryType::None&&WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC;
#endif
}
bool ThetaImplicitHybrid::NativePairedFieldScopeSupported() const {
#if defined(__FAST_MATH__) || \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    return false;
#else
#if defined(AMREX_USE_GPU)
    if (!warpx::darwin::NativePairedPreciseCudaBuild) { return false; }
#endif
    bool schur=false;
    bool joint_vacuum=false;
    amrex::ParmParse("implicit_evolve").query("darwin_longitudinal_schur",schur);
    amrex::ParmParse("implicit_evolve.thermal").query("joint_vacuum",joint_vacuum);
    // The circuit stopping certificate has a broader scope than the paired
    // field implementation. Preserve the latter's original positive-density,
    // source-free scope until the combined paths are implemented and qualified.
    return m_native_stopping_certificate && NativeStoppingCertificateScopeSupported() &&
        !m_circuit_native && !m_hybrid_pic_model->m_add_external_fields && !joint_vacuum &&
        !m_use_mass_matrices_jacobian && !m_use_mass_matrices_pc &&
        schur && !WarpX::use_filter && !m_WarpX->do_current_centering &&
        !m_extrapolate_initial_guess && !m_particle_suborbits &&
        !m_hybrid_pic_model->m_electron_inertia_bdf2 &&
        !m_hybrid_pic_model->m_darwin_checkpoint_restored;
#endif
}


std::string ThetaImplicitHybrid::NativePMCJointThermalDescriptor() const
{
    auto const& m=*m_hybrid_pic_model;
    amrex::Real conduction_theta=m_theta;
    amrex::ParmParse("implicit_evolve.thermal").query("conduction_theta",conduction_theta);
    std::ostringstream out;
    out<<std::hexfloat<<std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
       <<std::quoted(m.m_kappa_par_expression)<<' '<<std::quoted(m.m_kappa_perp_expression)<<' '
       <<m.m_include_thermal_conduction<<' '<<m.m_cond_operator<<' '<<m.m_cond_fd_order<<' '
       <<m.m_cond_fd_limiter<<' '<<m.m_cond_fd_limiter_width<<' '
       <<m.m_cond_chi_max<<' '<<m.m_cond_chi_par_max<<' '
       <<m.m_cond_isotropic<<' '<<m.m_cond_iso_B<<' '<<m.m_cond_flux_limit_factor<<' '
       <<m.m_gamma<<' '<<m.m_cond_te_floor<<' '<<m_theta<<' '<<conduction_theta<<' '
       <<m.m_qdsmc_n_floor<<' '<<m.m_n_floor<<' '<<m.m_qdsmc_halo_unfreeze<<' '
       <<m.m_cond_closed_floor_faces<<' '<<m.m_cond_wall_flux_limit<<' '
       <<m.m_cond_leg_flux_limit<<' '<<m.m_cond_wall_flux_cap_form<<' '
       <<m.m_cond_leg_length<<' '<<m.m_cond_leg_Te_wall<<' ';
    for(int d=0;d<AMREX_SPACEDIM;++d)for(int side=0;side<2;++side)
        out<<m.m_cond_bc[d][side]<<' '<<m.m_cond_bc_Te[d][side]<<' '<<m.m_cond_bc_q[d][side]<<' ';
    if(m_native_pmc_finite_conduction_requested)out<<"finite_conduction=1";
    return out.str();
}

bool ThetaImplicitHybrid::NativePMCJointThermalExecutorsValid() const
{
    auto const& m=*m_hybrid_pic_model;
    if(!m_native_pmc_thermal_parsers[0] || !m_native_pmc_thermal_parsers[1] ||
       !m.m_kappa_par_parser || !m.m_kappa_perp_parser ||
       !m.m_kappa_par || !m.m_kappa_perp) { return false; }
    std::array<char const*,4> current{m.m_kappa_par.m_host_executor,
        m.m_kappa_perp.m_host_executor,nullptr,nullptr};
#ifdef AMREX_USE_GPU
    current[2]=m.m_kappa_par.m_device_executor;
    current[3]=m.m_kappa_perp.m_device_executor;
    if(!current[2] || !current[3]) { return false; }
#endif
    // Local identities only. Shared Parser Data keeps captured bytecode alive,
    // so a supported redefinition cannot recycle its address. AMReX forbids
    // constants/variables from changing compiled Data; define creates new Data.
    // User functions and external symbolic constants are excluded at binding.
    return current==m_native_pmc_thermal_executors &&
        m.m_kappa_par_parser->expr()==m_native_pmc_thermal_parsers[0]->expr() &&
        m.m_kappa_perp_parser->expr()==m_native_pmc_thermal_parsers[1]->expr();
}

void ThetaImplicitHybrid::BindNativePMCJointThermalExecutors()
{
    auto const& m=*m_hybrid_pic_model;
    bool valid=NativePMCJointConfigurationValid(false);
    valid=valid && !m_native_pmc_thermal_parsers[0] && !m_native_pmc_thermal_parsers[1] &&
        m.m_kappa_par_parser && m.m_kappa_perp_parser && m.m_kappa_par && m.m_kappa_perp;
    // Numeric material laws may depend only on n, Te and t in this private
    // epoch. Parse symbols once, without evaluating/compiling a new executor.
    // Otherwise equal raw strings could hide different rank-local constants.
    if(valid) {
        auto const variables_only=[](std::string const& expression) {
            amrex::Parser raw(expression);
            auto const symbols=raw.symbols();
            return raw.userFunctions().empty() && std::all_of(symbols.begin(),symbols.end(),
                [](std::string const& symbol) {return symbol=="n" || symbol=="Te" || symbol=="t";});
        };
        valid=m.m_kappa_par_parser->userFunctions().empty() &&
            m.m_kappa_perp_parser->userFunctions().empty() &&
            variables_only(m.m_kappa_par_expression) && variables_only(m.m_kappa_perp_expression);
    }
    if(m_native_pmc_finite_conduction_requested && valid) {
        warpx::thermal::finite_flux::MaterialLaw parallel,perpendicular;
        valid=warpx::thermal::remainder::ArithmeticSupported() &&
            warpx::thermal::finite_flux::ParseMaterial(m.m_kappa_par_parser->expr(),parallel) &&
            warpx::thermal::finite_flux::ParseMaterial(m.m_kappa_perp_parser->expr(),perpendicular);
    }
#ifdef AMREX_USE_GPU
    valid=valid && m.m_kappa_par.m_device_executor && m.m_kappa_perp.m_device_executor;
#endif
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
        "Private PMC thermal compilation/lifetime binding failed before field initialization");
    m_native_pmc_thermal_parsers[0]=std::make_unique<amrex::Parser>(*m.m_kappa_par_parser);
    m_native_pmc_thermal_parsers[1]=std::make_unique<amrex::Parser>(*m.m_kappa_perp_parser);
    m_native_pmc_thermal_executors={m.m_kappa_par.m_host_executor,
        m.m_kappa_perp.m_host_executor,nullptr,nullptr};
#ifdef AMREX_USE_GPU
    m_native_pmc_thermal_executors[2]=m.m_kappa_par.m_device_executor;
    m_native_pmc_thermal_executors[3]=m.m_kappa_perp.m_device_executor;
#endif
}

bool ThetaImplicitHybrid::NativePMCJointThermalScopeSupported() const
{
#if defined(WARPX_DIM_RZ)
    auto const& m=*m_hybrid_pic_model;
    amrex::Real conduction_theta=m_theta;
    amrex::ParmParse("implicit_evolve.thermal").query("conduction_theta",conduction_theta);
    if(!m_native_pmc_thermal_requested || m_native_pmc_thermal_descriptor.empty() ||
       NativePMCJointThermalDescriptor()!=m_native_pmc_thermal_descriptor ||
       !m.m_include_thermal_conduction || m.m_cond_operator!=1 || m.m_cond_fd_order!=4 ||
       conduction_theta!=0.5_rt || m_theta!=0.5_rt ||
       (m_native_pmc_finite_conduction_requested
            ? (m.m_cond_flux_limit_factor!=0. && m.m_cond_flux_limit_factor!=0.1_rt)
            : m.m_cond_flux_limit_factor!=0.) ||
       m.m_cond_wall_flux_limit!=0. || m.m_cond_leg_flux_limit!=0. || m.m_cond_wall_flux_cap_form!=0 ||
       !(std::isfinite(m.m_cond_chi_max) &&
            (m_native_pmc_finite_conduction_requested || m.m_cond_chi_max<=0.)) ||
       !(std::isfinite(m.m_cond_chi_par_max) &&
            (m_native_pmc_finite_conduction_requested || m.m_cond_chi_par_max<=0.)) ||
       !(std::isfinite(m.m_cond_fd_limiter_width) && m.m_cond_fd_limiter_width>0.) ||
       !std::isfinite(m.m_cond_iso_B) ||
       !(std::isfinite(m.m_gamma) && m.m_gamma>1.) ||
       !(std::isfinite(m.m_cond_te_floor) && m.m_cond_te_floor>0.) ||
       !(std::isfinite(m.m_qdsmc_n_floor) && m.m_qdsmc_n_floor>0.) ||
       !(std::isfinite(m.m_n_floor) && m.m_n_floor>0.) ||
       !(std::isfinite(m.m_cond_leg_length) && m.m_cond_leg_length>0.) ||
       !(std::isfinite(m.m_cond_leg_Te_wall) && m.m_cond_leg_Te_wall>=m.m_cond_te_floor) ||
       m.m_qdsmc_halo_unfreeze || !m.m_cond_closed_floor_faces ||
       m.m_cond_bc[0][0]!=0 ||
       (m.m_cond_bc[0][1]!=0 && !(m_native_pmc_finite_conduction_requested && m.m_cond_bc[0][1]==1))) { return false; }
    if(m_native_pmc_finite_conduction_requested &&
       (!warpx::thermal::remainder::ArithmeticSupported() || m.m_cond_fd_limiter!=2 ||
        (m.m_cond_chi_max<=0. && m.m_cond_chi_par_max>0.))) { return false; }
    for(int d=0;d<AMREX_SPACEDIM;++d)for(int side=0;side<2;++side) {
        int const kind=m.m_cond_bc[d][side];
        if(kind<0 || kind>3 || (m_native_pmc_finite_conduction_requested && kind==2) ||
           !std::isfinite(m.m_cond_bc_Te[d][side]) ||
           !std::isfinite(m.m_cond_bc_q[d][side]) ||
           (kind==1 && m.m_cond_bc_Te[d][side]<m.m_cond_te_floor)) { return false; }
    }
    // Before InitData the owned descriptor is sufficient. Once captured,
    // actual local compiled storage must stay identical for the whole owner.
    return (!m_native_pmc_thermal_parsers[0] && !m_native_pmc_thermal_parsers[1]) ||
        NativePMCJointThermalExecutorsValid();
#else
    return false;
#endif
}

bool ThetaImplicitHybrid::NativePMCJointMassMatrixScopeSupported() const
{
#if defined(WARPX_DIM_RZ)
    bool requested_jv=false,requested_pc=false,companion=false;
    amrex::ParmParse implicit("implicit_evolve");
    implicit.query("use_mass_matrices_jacobian",requested_jv);
    implicit.query("use_mass_matrices_pc",requested_pc);
    implicit.query("darwin_joint_mm_thermal_increment",companion);
    return NativePrivatePMCMassMatrixSelectionOf(*m_WarpX)==NativePrivatePMCMassMatrixSelection::On &&
        m_native_pmc_mm_requested && requested_jv && requested_pc &&
        (m_native_pmc_finite_conduction_requested ? companion : !companion) &&
        m_use_mass_matrices && m_use_mass_matrices_jacobian && m_use_mass_matrices_pc &&
        m_esirkepov_mass_matrices && m_mass_matrices_density_projection &&
        m_live_ion_density && m_continuity_density && m_num_amr_levels==1 &&
        m_mass_matrices_deposit_interval==1 && !m_mass_matrices_reuse_within_step &&
        !m_fused_mass_matrices_deposit && !m_particle_suborbits &&
        !m_skip_particle_picard_init && !m_reflect_particles_at_rmax &&
        !m_mass_matrices_boundary_rows && m_verify_mm_jvp_step<0 &&
        WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov &&
        m_nlsolver && m_nlsolver->GetPreconditionerType()==PreconditionerType::pc_hybrid_pic;
#else
    return false;
#endif
}

NativePrivatePMCMassMatrixSelection NativePrivatePMCMassMatrixSelectionOf(
    WarpX const& simulation) noexcept
{
    return simulation.m_private_pmc_mm_selection;
}

std::uint64_t NativePrivatePMCMassMatrixOwnerGeneration(WarpX const& simulation) noexcept
{
    return simulation.m_private_pmc_mm_owner_generation;
}

ThetaImplicitHybrid const* NativePrivatePMCMassMatrixOwnerLocal(WarpX& simulation)
{
    if(NativePrivatePMCMassMatrixSelectionOf(simulation)!=NativePrivatePMCMassMatrixSelection::On)
        return nullptr;
    auto const* owner=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
    if(!owner || !owner->m_is_defined || owner->m_WarpX!=&simulation ||
       !owner->m_native_pmc_mm_requested) return nullptr;
    // All predicates are local metadata/owned-executor reads. Actual zero
    // drive remains checked collectively at the existing owner boundaries.
    return owner->NativePMCJointRequestSelectorsCurrent() &&
        owner->NativePMCJointScopeSupported() ? owner : nullptr;
}

ThetaImplicitHybrid const* NativePrivatePMCMassMatrixOwner(WarpX& simulation)
{
    // This branch is independent of a stale/replaced local solver pointer.
    // Raw ParmParse/member mutation cannot turn an immutable Off receipt On.
    if(NativePrivatePMCMassMatrixSelectionOf(simulation)==NativePrivatePMCMassMatrixSelection::Off)
        return nullptr;
    auto const* owner=NativePrivatePMCMassMatrixOwnerLocal(simulation);
    bool ready=owner!=nullptr;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    return ready ? owner : nullptr;
}

bool ThetaImplicitHybrid::NativePMCJointScopeSupported() const
{
#if defined(WARPX_DIM_RZ)
    if(!m_native_pmc_joint_requested) { return false; }
    auto const& m=*m_hybrid_pic_model;
    auto const& g=m_WarpX->Geom(0);
    bool joint=false,remainder=false,smooth=false,requested_mm_jv=false,requested_mm_pc=false;
    amrex::ParmParse implicit("implicit_evolve");
    implicit.query("use_mass_matrices_jacobian",requested_mm_jv);
    implicit.query("use_mass_matrices_pc",requested_mm_pc);
    bool retained_request=false;
    implicit.query("native_retained_acceptance",retained_request);
    amrex::ParmParse thermal("implicit_evolve.thermal");
    thermal.query("joint_vacuum",joint);
    thermal.query("current_remainder",remainder);
    amrex::ParmParse endpoint("endpoint_diagnostic");
    endpoint.query("smooth_force",smooth);
    // This scope is a non-accepting query: reject excluded raw requests before
    // a validating public accessor can assert on their unrelated prerequisites.
    bool circuit_request=false,prescribed_request=false;
    bool endpoint_request=false,ion_request=false,correlated_request=false;
    bool full_ohm_request=false,publish_request=true,segregated_request=false;
    bool yee_request=false;
    endpoint.query("native_circuit",circuit_request);
    endpoint.query("prescribed_drive",prescribed_request);
    endpoint.query("enable",endpoint_request);
    endpoint.query("ion_quadrature",ion_request);
    endpoint.query("correlated_increment",correlated_request);
    endpoint.query("full_ohm_longitudinal",full_ohm_request);
    endpoint.query("publish_field",publish_request);
    endpoint.query("yee_inertia",yee_request);
    implicit.query("darwin_segregated_solve",segregated_request);
    if(circuit_request || prescribed_request || !endpoint_request ||
       !ion_request || !correlated_request || !full_ohm_request ||
       !publish_request || !segregated_request || !yee_request ||
       !m.m_include_electron_inertia || !m.m_darwin ||
       !m.m_electron_inertia_djedt_only || m.m_electron_inertia_bdf2 ||
       m.m_esolve_tensor || m.m_esolve_curlcurl) { return false; }
    std::string policy,restart;
    endpoint.query("vacuum_edge_policy",policy);
    amrex::ParmParse("amr").query("restart",restart);
    std::vector<std::string> collisions;
    amrex::ParmParse("collisions").queryarr("collision_names",collisions);
    // The native flag records parser presence, including an identically zero
    // law. Its owned expressions are authoritative already in the constructor;
    // InitData later compiles these same expressions without clearing the flag.
    auto const literal_zero=[](std::string const& value) {
        return value=="0" || value=="0.0";
    };
    bool const zero_conduction=!m.m_include_thermal_conduction ||
        (literal_zero(m.m_kappa_par_expression) &&
         literal_zero(m.m_kappa_perp_expression));
    // Prescribed wall heat can remain nonzero even when both coefficients vanish.
    bool adiabatic=true;
    for(int d=0;d<AMREX_SPACEDIM;++d)
        for(int side=0;side<2;++side) adiabatic=adiabatic && m.m_cond_bc[d][side]==0;
    bool mm_companion=false;
    implicit.query("darwin_joint_mm_thermal_increment",mm_companion);
    bool const mm_scope=m_native_pmc_mm_requested ? NativePMCJointMassMatrixScopeSupported() :
        !requested_mm_jv && !requested_mm_pc && !m_mass_matrices_density_projection &&
        !m_use_mass_matrices_jacobian && !m_use_mass_matrices_pc && !mm_companion;
    return joint && !remainder && mm_scope &&
        (!m_eulerian_energy || (m_eulerian_energy->JointVacuumRequested() &&
                               !m_eulerian_energy->HasResidualRemainder())) &&
        retained_request &&
        m_darwin && m_darwin_segregated_solve && m_theta==0.5_rt &&
        m_vacuum_recovery && m_vacuum_recovery_half &&
        m_continuity_density &&
        !m_particle_suborbits && !m_external_field_iteration &&
        !m_circuit_native && !m_WarpX->get_pointer_CircuitCoupling() &&
        !m_native_stopping_certificate && !m_native_paired_requested &&
        !m_native_paired_fields && restart.empty() && !m.m_darwin_checkpoint_restored &&
        m_WarpX->maxLevel()==0 && m_WarpX->finestLevel()==0 && !EB::enabled() &&
        !m_WarpX->getdo_moving_window() && !WarpX::do_single_precision_comms &&
        WarpX::grid_type==GridType::Staggered && WarpX::nox==3 && WarpX::ncomps==1 &&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving &&
        WarpX::field_centering_nox==2 && WarpX::field_centering_noz==2 &&
        !WarpX::use_filter && sizeof(amrex::Real)==sizeof(double) &&
        g.Domain().smallEnd()==amrex::IntVect(0) && g.ProbLo(0)==0. &&
        !g.isPeriodic(0) && !g.isPeriodic(1) &&
        WarpX::field_boundary_lo[0]==FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC &&
        WarpX::field_boundary_lo[1]==FieldBoundaryType::PMC &&
        WarpX::field_boundary_hi[1]==FieldBoundaryType::PMC &&
        WarpX::particle_boundary_lo[1]==ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[1]==ParticleBoundaryType::Reflecting &&
        WarpX::particle_boundary_hi[0]==ParticleBoundaryType::Reflecting &&
        NativeVacuumEndpointEnabled() && NativeEndpointEnabled() &&
        NativeFullOhmLongitudinalEnabled() && NativeIonQuadratureEnabled() &&
        NativeCorrelatedIncrementEnabled() && warpx::darwin::increment::Enabled() &&
        !circuit_request && !prescribed_request &&
        !warpx::darwin::NativePECPlasmaEnabled() && !smooth &&
        m.UseCompatibleYeeInertia() && m.UsesEulerianElectronEnergy() &&
        !m.m_electron_inertia_bdf2 && !m.m_esolve_tensor &&
        !m.HasResistivity() && m.m_eta_per_species.empty() && !m.m_density_pedestal &&
        !m.m_has_external_current && !m.m_end_region.holmstrom &&
        !m.m_include_joule_heating && !m.m_include_temperature_relaxation &&
        (m_native_pmc_thermal_requested ? NativePMCJointThermalScopeSupported()
                                      : zero_conduction && adiabatic) &&
        !m.m_include_electron_viscosity &&
        !m.m_include_hyper_resistivity_term && !m.m_has_energy_sink &&
        !m.m_has_electron_stopping && !m.m_pec_conductor_wall_rows && collisions.empty() &&
        policy=="native_edge_candidate" && m.m_darwin_vacuum_recovery &&
        m.m_darwin_vacuum_recovery_cadence=="half" && m.m_darwin_vacuum_recovery_frozen_mask &&
        m.m_darwin_vacuum_recovery_components=="all" &&
        m.m_darwin_vacuum_recovery_mask=="vacuum" && m.m_darwin_vacrec_relax_time==0. &&
        m.m_darwin_vacuum_recovery_operator=="poisson" && m.m_vacuum_recovery_live_probes;
#else
    return false;
#endif
}

bool ThetaImplicitHybrid::NativePMCJointRequestSelectorsCurrent() const
{
    bool requested=false;
    amrex::ParmParse("pmc_joint_qualification").query("enable",requested);
    bool thermal_requested=false;
    amrex::ParmParse("pmc_joint_qualification").query("thermal",thermal_requested);
    bool mm_requested=false;
    amrex::ParmParse("pmc_joint_qualification").query("mass_matrix",mm_requested);
    bool finite_requested=false;
    amrex::ParmParse("pmc_joint_qualification").query("finite_conduction",finite_requested);
    auto const selected=m_native_pmc_mm_requested ? NativePrivatePMCMassMatrixSelection::On
                                                 : NativePrivatePMCMassMatrixSelection::Off;
    return requested==m_native_pmc_joint_requested &&
        thermal_requested==m_native_pmc_thermal_requested &&
        mm_requested==m_native_pmc_mm_requested &&
        finite_requested==m_native_pmc_finite_conduction_requested &&
        NativePrivatePMCMassMatrixSelectionOf(*m_WarpX)==selected;
}

bool ThetaImplicitHybrid::NativePMCJointConfigurationValid(bool check_drive) const
{
    bool valid=NativePMCJointRequestSelectorsCurrent();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid) { return false; }
    if(!m_native_pmc_joint_requested) { return true; }
    valid=NativePMCJointScopeSupported();
    if(check_drive && m_native_pmc_thermal_requested)
        valid=valid && NativePMCJointThermalExecutorsValid();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid) { return false; }
    // Called only after native external initialization. This collective checks
    // actual owned zero profiles, unit layouts and materialized driven fields.
    return !check_drive || NativeZeroExternalDriveSupported(*m_WarpX);
}

bool ThetaImplicitHybrid::NativeRetainedScopeSupported() const
{
        bool const static_vacuum=NativeRetainedStaticVacuumSupported(*m_WarpX);
        bool const circuit_vacuum=NativeRetainedCircuitVacuumSupported(*m_WarpX);
        bool const pmc_vacuum=NativePMCJointScopeSupported();
        bool const supported=m_hybrid_pic_model->UsesEulerianElectronEnergy() && m_darwin && m_theta==0.5_rt &&
            NativeEndpointEnabled() && NativeIonQuadratureEnabled() &&
            NativeCorrelatedIncrementEnabled() && m_hybrid_pic_model->UseCompatibleYeeInertia() &&
            ((!m_vacuum_recovery && !NativeVacuumEndpointEnabled()) ||
             (!m_circuit_native && (static_vacuum || pmc_vacuum)) ||
             (m_circuit_native && circuit_vacuum)) &&
            (m_circuit_native==NativeCircuitDriveEnabled()) &&
            (!m_circuit_native || NativeCircuitCoupler().DeviceTrials()) &&
            !NativePrescribedDriveEnabled() &&
            (!m_hybrid_pic_model->m_add_external_fields || static_vacuum || pmc_vacuum || m_circuit_native) && !m_hybrid_pic_model->m_esolve_tensor &&
            !m_hybrid_pic_model->HasResistivity() && !m_hybrid_pic_model->m_include_joule_heating &&
            !m_hybrid_pic_model->m_include_temperature_relaxation &&
            !m_hybrid_pic_model->m_joule_redirect_to_ions &&
            !m_hybrid_pic_model->m_include_electron_viscosity && !m_hybrid_pic_model->m_has_energy_sink;
    return supported;
}

bool ThetaImplicitHybrid::NativeStoppingCertificateScopeSupported()const
{
#if !defined(WARPX_DIM_RZ)
    return false;
#else
    auto const& geom=m_WarpX->Geom(0);
    bool const circuit_source=m_circuit_native&&warpx::darwin::NativeCoilCurrentEnabled()&&
        NativeCircuitCoupler().SupportsSourceImpulse()&&!m_use_mass_matrices_jacobian&&!m_use_mass_matrices_pc;
    return !m_hybrid_pic_model->m_end_region.holmstrom && NativeRetainedScopeSupported()&&warpx::implicit::NativeRetainedAcceptanceEnabled()&&
        m_darwin_segregated_solve&&NativeFullOhmLongitudinalEnabled()&&
        warpx::darwin::NativePECPlasmaEnabled()&&!warpx::darwin::increment::Enabled()&&
        !NativeVacuumEndpointEnabled()&&!m_vacuum_recovery&&(!m_circuit_native||circuit_source)&&
        (!m_hybrid_pic_model->m_add_external_fields||circuit_source)&&!m_hybrid_pic_model->m_has_external_current&&
        ((!m_use_mass_matrices_jacobian&&!m_use_mass_matrices_pc)||NativePECMMQualificationEnabled())&&
        !WarpX::do_single_precision_comms&&!EB::enabled()&&m_WarpX->finestLevel()==0&&
        WarpX::ncomps==1&&WarpX::grid_type==GridType::Staggered&&sizeof(amrex::Real)==sizeof(double)&&
        WarpX::field_centering_nox==2&&WarpX::field_centering_noz==2&&
        geom.ProbLo(0)==0.&&geom.isPeriodic(1)&&
        WarpX::field_boundary_lo[0]==FieldBoundaryType::None&&
        WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC;
#endif
}

int ThetaImplicitHybrid::OneStep (const amrex::Real start_time,
    const amrex::Real a_dt, const int a_step)
{
    return OneStepImpl(start_time,a_dt,a_step,nullptr);
}

int ThetaImplicitHybrid::OneStepImpl (const amrex::Real start_time,
    const amrex::Real a_dt, const int a_step,
    std::unique_ptr<warpx::implicit::NativeAcceptedStepCandidate>* deferred,
    warpx::implicit::NativeSymmetricStoppingOptions const* symmetric_source,
    warpx::thermal::NativeStoppingCarryRequest const* stopping_carry)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_candidate_active,
        "A native endpoint candidate must be canceled or finalized before another step");
    BL_PROFILE("ThetaImplicitHybrid::OneStep()");
    if(!NativePMCJointConfigurationValid(true)) {
        amrex::Print()<<"Private PMC joint configuration declined before snapshot\n";
        return -14;
    }
    bool paired_configuration=warpx::darwin::NativePairedDarwinFields::Requested()==bool(m_native_paired_fields);
    if(m_native_paired_fields)paired_configuration=NativePairedFieldScopeSupported()&&paired_configuration;
    amrex::ParallelDescriptor::ReduceBoolAnd(paired_configuration);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(paired_configuration,
        "Positive paired arithmetic option or supported scope changed after native initialization");
    bool const material_symmetric=symmetric_source&&symmetric_source->material_support;
    bool endpoint_mm=false;
    amrex::ParmParse("endpoint_diagnostic").query("retained_endpoint_pair_mm",endpoint_mm);
    bool endpoint_configuration=endpoint_mm==m_native_endpoint_pair_mm_requested&&
        warpx::darwin::NativePairedDarwinFields::EndpointRequested()==m_native_endpoint_pair_requested&&
        warpx::darwin::NativePairedDarwinFields::EndpointAcceptanceRequested()==m_native_endpoint_pair_accept_requested&&
        bool(m_native_endpoint_pair)==m_native_endpoint_pair_requested;
    if(m_native_endpoint_pair_requested){
        bool const scope=NativeEndpointPairScopeSupported();
        endpoint_configuration=scope&&endpoint_configuration&&m_native_endpoint_pair&&
            !m_native_endpoint_pair->EndpointPending()&&deferred&&(!symmetric_source||material_symmetric)&&
            (stopping_carry||(!symmetric_source&&warpx::darwin::NativeEndpointPMCQualificationSelected(*m_WarpX)));
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(endpoint_configuration);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(endpoint_configuration,
        "Retained endpoint pair requires unchanged scope and the private deferred carry/cancel path");
    // Recheck capabilities before any snapshot/trial mutation: callbacks and
    // mutable source flags can be registered after InitData completed.
    if (NativeEndpointEnabled()) {
        ValidateNativeEndpointCapabilities(*m_WarpX);
        ValidateNativeVacuumEntryDensity(*m_WarpX);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(NativeCircuitDriveEnabled() && NativeVacuumEndpointEnabled()) ||
            (warpx::implicit::NativeRetainedAcceptanceEnabled() &&
             NativeRetainedCircuitVacuumSupported(*m_WarpX)),
            "Live native vacuum/circuit requires qualified source-free RZ MM0 retained endpoint acceptance");
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!NativeEndpointEnabled() ||
        !m_hybrid_pic_model->m_darwin_checkpoint_restored || NativeEndpointRestartSupported(),
        "Native endpoint restart requires the validated direct Yee companion checkpoint");

    if (NativeIonQuadratureEnabled()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_theta == 0.5_rt,
            "Native ion quadrature requires midpoint time centering");
        if (m_use_mass_matrices_jacobian) {
            ValidateNativeEndpointMassMatrixCapabilities(*m_WarpX);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass_matrices_density_projection &&
                m_esirkepov_mass_matrices && !m_particle_suborbits &&
                m_mass_matrices_deposit_interval == 1 && !m_mass_matrices_reuse_within_step,
                "Endpoint MM requires native interval-current density projection and a fresh "
                "Esirkepov tangent at every linearization; suborbits/reuse are unsupported");
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_mass_matrices_density_projection,
                "Native full-particle Jv requires native deposited density");
        }
    }
    if (m_hybrid_pic_model->UsesEulerianElectronEnergy() && !m_eulerian_energy) {
        m_eulerian_energy = std::make_unique<warpx::thermal::DarwinThermalAdvance>(*this);
        // Thermal controls are first consumed by this deferred construction,
        // after InitData wrote the initial used-input file. Include explicit
        // values and recorded defaults before any trial or possible failure.
        std::string used_inputs_file = "warpx_used_inputs";
        amrex::ParmParse("warpx").queryAdd("used_inputs_file", used_inputs_file);
        ablastr::utils::write_used_inputs_file(used_inputs_file, false);
    }
    // Complete the typed entry preflight after deferred owner construction and
    // before particle, field, diagnostic or circuit snapshots/source mutation.
    bool current_source_scope=!symmetric_source || !m_eulerian_energy ||
        !m_eulerian_energy->HasResidualRemainder();
    amrex::ParallelDescriptor::ReduceBoolAnd(current_source_scope);
    if(!current_source_scope) {
        amrex::Print()<<"Thermal current remainder does not support native stopping events\n";
        return -14;
    }
    // Check the actual deferred thermal owner before any rollback snapshot.
    if(!NativePMCJointConfigurationValid(true)) {
        amrex::Print()<<"Private PMC joint owner declined before snapshot\n";
        return -14;
    }
    if(stopping_carry) {
        // Validate the actual deferred constructor result collectively before
        // any retained field/particle/diagnostic snapshot or carry request.
        bool owner_ready=m_eulerian_energy&&m_eulerian_energy->JointVacuumRequested();
        amrex::ParallelDescriptor::ReduceBoolAnd(owner_ready);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(owner_ready,
            "Stopping carry lost its selected joint-vacuum thermal owner before snapshot");
    }
    bool const retained_acceptance=warpx::implicit::NativeRetainedAcceptanceEnabled();
    if (m_eulerian_energy && m_circuit_native) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(retained_acceptance
            ? NativeCircuitCoupler().SupportsNativeTransaction()
            : NativeCircuitCoupler().SupportsNativeRejection(),
            "Eulerian Darwin requires the selected native pre-accept or retained post-accept capability");
    }
    // EL and its scalar representation are both accepted endpoint state.
    // Keep this local to the step so rejected trials cannot rotate its origin.
    std::unique_ptr<amrex::MultiFab> vacuum_phi_old;
    if (NativeVacuumEndpointEnabled()) {
        auto const& phi = *m_WarpX->m_fields.get("hybrid_phi_darwin_fp", 0);
        vacuum_phi_old = std::make_unique<amrex::MultiFab>(
            phi.boxArray(), phi.DistributionMap(), phi.nComp(), phi.nGrowVect());
        amrex::MultiFab::Copy(*vacuum_phi_old, phi, 0, 0, phi.nComp(), phi.nGrowVect());
        AuditNativeVacuumPotential(*m_WarpX, "old");
    }
    using ParticleTransaction=warpx::particles::NativeCollisionTransaction;
    using Candidate=warpx::implicit::NativeAcceptedStepCandidate;
    std::unique_ptr<Candidate> retained_candidate;
    if (retained_acceptance) {
        bool const supported=NativeRetainedScopeSupported();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,
            "Retained acceptance requires source-free native Yee Eulerian Darwin with positive "
            "density and an optional retained native circuit, qualified static joined vacuum, "
            "or source-free RZ MM0 native circuit/vacuum; prescribed drive and accepted sources remain unsupported");
        auto owner=std::make_unique<Candidate::Impl>();
        owner->simulation=m_WarpX; owner->solver=this;
        owner->thermal=m_eulerian_energy.get(); owner->circuit=m_circuit_native;
        owner->start_time=start_time; owner->endpoint_time=start_time+a_dt; owner->interval=a_dt;
        owner->step=a_step;owner->symmetric=symmetric_source!=nullptr;owner->material_symmetric=material_symmetric;
        if(m_native_stopping_certificate) {
            bool supported=NativeStoppingCertificateScopeSupported()&&static_cast<bool>(m_native_longitudinal_accepted);
            amrex::ParallelDescriptor::ReduceBoolAnd(supported);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(supported,"Stopping certificate lost its supported native producer scope");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_longitudinal_accepted->Matches(*m_WarpX,start_time,a_step),
                "Stopping certificate does not match this accepted epoch and physical fields");
            m_native_longitudinal_pending=m_native_longitudinal_accepted->Fork(
                symmetric_source ? start_time : start_time+a_dt,
                symmetric_source ? a_step : a_step+1);
            owner->longitudinal=m_native_longitudinal_pending;
        }
        owner->particles=std::make_unique<ParticleTransaction>();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(owner->particles->Capture(*m_WarpX)==ParticleTransaction::Status::Success,
            "Complete native particle snapshot unavailable for retained acceptance");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_WarpX->reduced_diags->CaptureMidStepState(owner->diagnostics),
            "Midpoint diagnostic snapshot unavailable for retained acceptance");
        retained_candidate=std::unique_ptr<Candidate>(new Candidate(std::move(owner)));
    }
    if (m_eulerian_energy) { m_eulerian_energy->SnapshotStepStart();
        if(stopping_carry&&!material_symmetric)m_eulerian_energy->RequestStoppingCarry(*stopping_carry); }
    if (retained_candidate && m_circuit_native) {
        auto& coupler=NativeCircuitCoupler();
        if(!m_native_circuit_configured) {
            coupler.ConfigureDarwinMagneticResponse();m_native_circuit_configured=true;
        }
        // The original rollback origin precedes S1 and the advertisement of
        // a cancellable candidate. F measures its own post-S1 physical origin;
        // it must never replace this whole-map snapshot.
        bool reversible=coupler.SnapshotRetainedNativeStep(*m_hybrid_pic_model->m_external_vector_potential);
        amrex::ParallelDescriptor::ReduceBoolAnd(reversible);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(reversible,
            "Unable to snapshot the native circuit before the retained source/field map");
    }
    if (retained_candidate) {
        retained_candidate->m_impl->active=true;
        m_native_candidate_active=true;
    }
    auto cancel_retained=[&]() {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            retained_candidate->Cancel()==warpx::implicit::NativeCandidateStatus::Success,
            "Retained candidate restoration failed; step is terminal");
    };
    if(material_symmetric){
        auto const status=retained_candidate->ApplyMaterialHalf(true,*symmetric_source);
        if(status!=warpx::implicit::NativeCandidateStatus::Success){
            if(status==warpx::implicit::NativeCandidateStatus::Unsupported)cancel_retained();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(status!=warpx::implicit::NativeCandidateStatus::Terminal,
                "Material S1 failed without whole-step cancellation");return -14;
        }
        m_eulerian_energy->RequestStoppingCarry(*stopping_carry);
        if(!warpx::implicit::CheckNativeAcceptance(*m_WarpX,warpx::implicit::AcceptancePhase::BeforeSymmetricField)){
            cancel_retained();return -14;
        }
    } else if(symmetric_source) {
        auto const source=retained_candidate->ApplySource(Candidate::SourcePhase::MidpointPre,
            nullptr,symmetric_source->physical,&symmetric_source->thermal);
        if(source.status!=warpx::implicit::NativeCandidateStatus::Success) {
            if(source.status==warpx::implicit::NativeCandidateStatus::Unsupported)cancel_retained();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(source.status!=warpx::implicit::NativeCandidateStatus::Terminal,
                "Symmetric pre-source failed without a restorable step");
            amrex::Print()<<"Symmetric pre-source declined: "<<source.reason<<"\n";return -14;
        }
        // AppendEvent closes its EL producer trace. Start a new field trace
        // from the actual source-updated accepted state, retaining its D origin.
        m_native_longitudinal_pending=retained_candidate->m_impl->longitudinal->Fork(start_time+a_dt,a_step+1);
        retained_candidate->m_impl->longitudinal=m_native_longitudinal_pending;
        if(!warpx::implicit::CheckNativeAcceptance(*m_WarpX,warpx::implicit::AcceptancePhase::BeforeSymmetricField)) {
            cancel_retained();return -14;
        }
    } else if(retained_candidate)retained_candidate->m_impl->phase=Candidate::Impl::Phase::Field;
    m_dt = a_dt;
    // Every physical advance (including a retried step) owns a fresh tangent.
    InvalidateMassMatrices();

    if (m_darwin_circuit_consistent_stage) {
        auto const& external = *m_hybrid_pic_model->m_external_vector_potential;
        m_darwin_circuit_accepted_scales.resize(external.nFields());
        for (int coil = 0; coil < external.nFields(); ++coil) {
            auto const value = external.TimeScale(coil, start_time);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(value),
                "Nonfinite accepted circuit scale");
            m_darwin_circuit_accepted_scales[coil] = value;
        }
    }

    // tensor_form: publish the theta interval and snapshot the step-start
    // plasma current Jp^n = curl(B^n)/mu0 - J_ext (Bfield_fp holds the
    // committed B^n here). Both are per-step-frozen inputs of the
    // stateless Je elimination, constant through every residual
    // evaluation of the step.
    if (m_hybrid_pic_model->m_esolve_tensor) {
        m_hybrid_pic_model->m_tensor_dt_eff = m_theta * m_dt;
        m_hybrid_pic_model->CaptureTensorStepStart();
    }

    // curlcurl_form + frozen gates: capture the per-step-frozen rho^n
    // snapshot HERE, from the committed entry deposit in component 0 of
    // rho_fp (the tensor/vacmask capture family), BEFORE any residual
    // evaluation of the step. The resistive push-field correction runs
    // its Ohm passes before the first midpoint deposit and before the
    // inertia assembly's lazy capture, so the first elliptic solves of
    // the run otherwise read an EMPTY midpoint density -- beta = 0
    // everywhere and every anchored RHS row zeroed against a nonzero
    // resistive warm start, an unreachable tolerance (measured: seeded
    // vacuum-column decks cap the toroidal CG at the very first solve).
    // The inertia assembly's own capture becomes a per-step no-op (latch
    // already set); the drho/dt leg, the gates, and the toroidal fold all
    // read this same entry snapshot.
    if (m_hybrid_pic_model->m_esolve_curlcurl
        && m_hybrid_pic_model->m_curlcurl_pol_frozen_rho) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab const & rho_fp =
                *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            amrex::MultiFab & rho_n_frozen =
                *m_WarpX->m_fields.get("hybrid_rho_n_frozen", lev);
            amrex::MultiFab::Copy(rho_n_frozen, rho_fp, 0, 0, 1,
                                  amrex::min(rho_n_frozen.nGrowVect(),
                                             rho_fp.nGrowVect()));
        }
        m_hybrid_pic_model->m_inertia_rho_n_captured = true;
    }

    // Non-Darwin inertia uses the committed current as well. Darwin seeded
    // it with its initial field solve, so no trial evaluation can seed history.
    // Use total B and the t0 external current before the split-field update.
    if (m_hybrid_pic_model->m_include_electron_inertia
        && !m_hybrid_pic_model->m_inertia_history_initialized) {
        CalculateNativePlasmaCurrent(
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
            start_time);
        m_hybrid_pic_model->InitializeElectronInertiaHistory();
    }

    // External vector-potential drive, split-field form: the solver state
    // carries the PLASMA fields only, so the field boundary conditions act
    // on the plasma response while the imposed external field rides
    // through the wall unchanged (a conducting boundary must not exclude
    // the programmed coil flux; advancing the external flux through the
    // discrete Faraday/vector-potential update would pin it to the wall
    // value of E or A). Strip the external field at t^n here; the field
    // assembly in UpdateWarpXFields re-adds B_ext at the theta-time and
    // the step-averaged inductive E_ext on top of the plasma fields, and
    // FinishFieldUpdate restores end-of-step totals.
    if (m_hybrid_pic_model->m_add_external_fields && !m_darwin) {
        using ablastr::fields::Direction;
        auto & ext = *m_hybrid_pic_model->m_external_vector_potential;
        ext.UpdateHybridExternalFields(start_time, a_dt);
        ExtLedgerPrint(m_WarpX, "entry pre-strip");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Subtract(B, B_ext, 0, 0, B.nComp(), B.nGrowVect());
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & E_ext = *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Subtract(E, E_ext, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        ExtLedgerPrint(m_WarpX, "entry post-strip");
        // Mid-step values used throughout the nonlinear solve: B_ext at
        // t^{n+theta}, and E_ext = -[f(t^{n+theta}+dt/2) -
        // f(t^{n+theta}-dt/2)]/dt * A for the push field (at theta = 1/2
        // this is the exact step mean of the inductive field).
        ext.UpdateHybridExternalFields(start_time + m_theta*a_dt, a_dt);
        ExtLedgerPrint(m_WarpX, "entry post-theta-eval");
    }

    // Save particle state at t^n
    m_WarpX->SaveParticlesAtImplicitStepStart(bool(m_eulerian_energy));
    if (m_continuity_density) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            auto& predictor = m_mass_matrix_density[lev];
            if (!predictor) { predictor = std::make_unique<MassMatrixDensityProjection>(); }
            predictor->CaptureStepStart(*m_WarpX->m_fields.get(FieldType::rho_fp,lev),0);
        }
    }

    // Save E^n
    SaveEoldMultifab();
    if (m_darwin) {
        using ablastr::fields::Direction;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_initialized,
            "Darwin fields must be initialized before advancing the first step");

        // E_L^n from the entry state (Pe^n and the Evolve-entry deposit
        // rho^n in component 0 of rho_fp); save it and A^n for the theta
        // reconstructions, then strip E_L from Efield_fp so the solver
        // state (seeded from Efield_fp below) is the transverse field E_T.
        ablastr::fields::MultiLevelScalarField rho_n_alias;
        amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_n_store(m_num_amr_levels);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab & rho_fp = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            rho_n_store[lev] = std::make_unique<amrex::MultiFab>(rho_fp, amrex::make_alias, 0, 1);
            rho_n_alias.push_back(rho_n_store[lev].get());
        }
        // Frozen vacuum-recovery mask density: snapshot the same committed
        // entry rho the E_L^n solve below consumes, so the recovery/
        // Faraday-overwrite partition is constant through every residual
        // evaluation of the step (rho_fp component 0 is a pre-push deposit
        // that follows the iterate from the second evaluation on -- see
        // m_darwin_vacuum_recovery_frozen_mask).
        if (m_vacuum_recovery
            && m_hybrid_pic_model->m_darwin_vacuum_recovery_frozen_mask) {
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                amrex::MultiFab const & rho_fp =
                    *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
                amrex::MultiFab & rho_mask =
                    *m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev);
                amrex::MultiFab::Copy(rho_mask, rho_fp, 0, 0, 1,
                                      amrex::min(rho_mask.nGrowVect(),
                                                 rho_fp.nGrowVect()));
            }
        }
        // With electron inertia, the E_L source reads the inertial field
        // at its last converged assembly (t^{n-1+theta}) here -- a
        // half-step staleness of the same order as the ion half-step
        // offset in the Je history; a t^n reassembly from the histories
        // is a possible refinement.
        bool endpoint_audit=false,endpoint_dumps=false;
        amrex::ParmParse endpoint_options("endpoint_diagnostic");
        endpoint_options.query("audit",endpoint_audit);
        endpoint_options.query("write_fields",endpoint_dumps);
        std::array<std::unique_ptr<amrex::MultiFab>,3> diagnostic_old_EL;
        std::string const diagnostic_prefix="ENTRY_"+std::to_string(a_step)+"_";
        if(NativeEndpointEnabled() && (endpoint_audit || endpoint_dumps)) {
            for(int c=0;c<3;++c) {
                auto const& f=*m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0);
                if(endpoint_audit) {
                diagnostic_old_EL[c]=std::make_unique<amrex::MultiFab>(f.boxArray(),f.DistributionMap(),1,0);
                amrex::MultiFab::Copy(*diagnostic_old_EL[c],f,0,0,1,0);
                }
                if(endpoint_dumps)amrex::VisMF::Write(f,diagnostic_prefix+"EL_before_"+std::to_string(c));
            }
            if(endpoint_dumps)amrex::VisMF::Write(*rho_n_alias[0],diagnostic_prefix+"rho");
            if(endpoint_dumps)amrex::VisMF::Write(m_hybrid_pic_model->ElectronPressureForSolve(0),diagnostic_prefix+"Pe");
        }
        if (!NativeFullOhmLongitudinalEnabled()) {
            m_hybrid_pic_model->ComputeDarwinELong(rho_n_alias, start_time);
        }
        // Full-Ohm EL and its verified scalar representation are accepted
        // endpoint state. A new approximate projection would change the
        // origin used for D and Je; retain them exactly across step entry.
        if(NativeEndpointEnabled() && (endpoint_audit || endpoint_dumps)) {
            std::array<amrex::Real,3> jump{};
            for(int c=0;c<3;++c) {
                auto const& f=*m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0);
                if(endpoint_dumps)amrex::VisMF::Write(f,diagnostic_prefix+"EL_after_"+std::to_string(c));
                if(endpoint_audit) {
                amrex::MultiFab::Subtract(*diagnostic_old_EL[c],f,0,0,1,0);
                jump[c]=diagnostic_old_EL[c]->norminf();
                }
            }
            if(endpoint_audit && amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream log("ENTRY_ORIGIN.jsonl",std::ios::app);
                log<<std::setprecision(17)<<"{\"step\":"<<a_step<<",\"time\":"<<start_time<<",\"EL_jump_V_per_m\":["<<jump[0]<<","<<jump[1]<<","<<jump[2]<<"]}\n";
            }
        }
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab & EL_old = *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
                amrex::MultiFab::Copy(EL_old, EL, 0, 0, EL.nComp(), EL.nGrowVect());
                amrex::MultiFab const & A = *m_WarpX->m_fields.get("hybrid_A_fp", Direction{dir}, lev);
                amrex::MultiFab & A_old = *m_WarpX->m_fields.get("hybrid_A_old_fp", Direction{dir}, lev);
                amrex::MultiFab::Copy(A_old, A, 0, 0, A.nComp(), A.nGrowVect());
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                if (NativeVacuumEndpointEnabled() && NativeCircuitDriveEnabled()) {
                    amrex::MultiFab::Copy(E, *m_WarpX->m_fields.get(
                        NativeVacuumTransverseRecordName(),Direction{dir},lev),
                        0,0,E.nComp(),E.nGrowVect());
                } else if(!warpx::darwin::NativeCoilCurrentEnabled()) {
                    amrex::MultiFab::Subtract(E, EL, 0, 0, E.nComp(), E.nGrowVect());
                }
                // The driven route retains physical E here. Its free solver
                // origin is the exact private m_E supplied by endpoint closure.
            }
        }
        // Boundary-driven external flux: pin A^n (idempotent re-pin of the
        // end-of-last-step values, and the gauge reference on step one).
        DarwinApplyABoundary(start_time);

        // The transverse state at t^n (Efield_fp = E^n - E_L^n here).
        // Keep the previous step's E^n as E^{n-1} for the extrapolated guess.
        if (m_extrapolate_initial_guess && m_have_Eold) {
            m_Eprev.Copy(m_Eold);
            m_have_Eprev = true;
        }
        if(warpx::darwin::NativeCoilCurrentEnabled())m_Eold.Copy(m_E);
        else m_Eold.Copy(FieldType::Efield_fp);
        m_have_Eold = true;
        if(m_native_endpoint_pair){
            auto const& old=m_Eold.getArrayVec()[0];auto const& previous=m_Eprev.getArrayVec()[0];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_endpoint_pair->CaptureEndpointHistory(
                {old[0],old[1],old[2]},{previous[0],previous[1],previous[2]},m_have_Eprev),
                "Retained endpoint history lost its captured transverse pair");
        }
    } else {
        // Non-Darwin path: E^n sits in the E_old register. Same E^{n-1} bookkeeping
        // (a restart starts with m_have_Eold false, so its first step uses E^n).
        if (m_extrapolate_initial_guess && m_have_Eold) {
            m_Eprev.Copy(m_Eold);
            m_have_Eprev = true;
        }
        m_Eold.Copy(FieldType::E_old, FieldType::None, true);
        m_have_Eold = true;
    }

    // Save B^n
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Bfp = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField B_old = m_WarpX->m_fields.get_alldirs(FieldType::B_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*B_old[n], *Bfp[n], 0, 0,
                                  B_old[n]->nComp(), B_old[n]->nGrowVect());
        }
    }

    // Save the electron-energy start-of-step state (T_e^n, J_plasma(B^n),
    // rho^n, frozen T_i^n deposits). B currently holds B^n, so refresh the
    // plasma current from it first.
    if (m_eulerian_energy) {
        m_eulerian_energy->BeginStep(start_time, m_dt);
    } else if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        CalculateNativePlasmaCurrent(
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
            start_time);
        m_hybrid_pic_model->QDSMCSaveImplicitStepStart(m_dt, start_time);
        if (m_qdsmc_segregated_solve)
        {
            m_qdsmc_rho_frozen.resize(m_num_amr_levels);
            for (int lev = 0; lev < m_num_amr_levels; ++lev)
            {
                auto const& rho = *m_WarpX->m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
                auto& frozen = m_qdsmc_rho_frozen[lev];
                if (!frozen || frozen->boxArray() != rho.boxArray() ||
                    frozen->DistributionMap() != rho.DistributionMap())
                {
                    frozen = std::make_unique<amrex::MultiFab>(
                        rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
                }
                amrex::MultiFab::Copy(*frozen, rho, 0, 0, 1, rho.nGrowVect());
            }
        }
    }

    if (m_circuit_native) {
        auto& coupler = NativeCircuitCoupler();
        if (!m_native_circuit_configured) {
            coupler.ConfigureDarwinMagneticResponse();
            m_native_circuit_configured = true;
        }
        if (m_eulerian_energy) {
            bool reversible=retained_acceptance
                ? coupler.RetainedNativeStepCancelable()
                : coupler.SnapshotNativeStep(*m_hybrid_pic_model->m_external_vector_potential);
            amrex::ParallelDescriptor::ReduceBoolAnd(reversible);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(reversible,
                "Unable to snapshot the native circuit before a rejectable step");
        }
        // These are the actual accepted B^n registers, before any trial update.
        coupler.MeasureDarwinLinkages(start_time);
        coupler.BeginStepMeasured(start_time, m_dt);
        if (coupler.DeviceTrials()) {
            if (m_fext_init.empty()) {
                auto const& ext = *m_hybrid_pic_model->m_external_vector_potential;
                for (int i = 0; i < ext.nFields(); ++i) {
                    m_fext_init.push_back(ext.TimeScale(i,start_time));
                }
            }
            if(retained_candidate) {
                std::string preparation_error;
                bool prepared=coupler.TryPrepareDarwinDeviceStep(start_time,m_dt,m_theta,
                    m_darwin_circuit_scale_tolerance,preparation_error);
                amrex::ParallelDescriptor::ReduceBoolAnd(prepared);
                if(!prepared) {
                    cancel_retained();
                    amrex::Print()<<"Native circuit preparation declined: "
                        <<(preparation_error.empty()?"unsupported on another rank":preparation_error)<<"\n";
                    return -13;
                }
            }else{
                coupler.PrepareDarwinDeviceStep(start_time,m_dt,m_theta,m_darwin_circuit_scale_tolerance);
            }
        }
        m_native_circuit_step_open = true;
    }

    // Initial guess: E^{n+theta} = E^n, or the linear extrapolation of the
    // field history (1 + theta) E^n - theta E^{n-1} when opted in and E^{n-1}
    // exists (saves the Newton iteration that otherwise rebuilds the step's
    // change from scratch; the converged state does not depend on the guess).
    if (m_extrapolate_initial_guess && m_have_Eprev) {
        m_E.linComb(1.0_rt + m_theta, m_Eold, -m_theta, m_Eprev);
    } else {
        m_E.Copy(m_Eold);
    }

    // Solve nonlinear system for E^{n+theta} (and eventually Pe^{n+theta})
    int exit_status = 0;
    if (m_darwin_segregated_solve) {
        exit_status = SolveDarwinSegregated(start_time, a_step);
    } else if (m_qdsmc_segregated_solve) {
        exit_status = SolveSegregated( start_time, a_step );
    } else {
        m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );
        exit_status = m_nlsolver->GetExitStatus();
    }
    if (exit_status >= 0 && m_eulerian_energy && !m_eulerian_energy->PrepareAcceptance()) {
        exit_status=-9;
    }
    if (exit_status >= 0 && retained_acceptance && !m_eulerian_energy->CanRetainAcceptance()) {
        exit_status=-10; // no native boundary/source mutation has occurred
    }
    if (exit_status < 0) {
        if (retained_acceptance) {
            cancel_retained();
            return exit_status;
        }
        if (m_eulerian_energy) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_eulerian_energy->CanCancel(),
                "Implicit field topology changed before rejection rollback");
            if (m_circuit_native) {
                auto& coupler=NativeCircuitCoupler();
                bool reversible=coupler.NativeStepCancelable();
                amrex::ParallelDescriptor::ReduceBoolAnd(reversible);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(reversible,
                    "Native circuit has crossed its acceptance barrier; step cannot be retried");
                bool restored=coupler.CancelNativeStep();
                amrex::ParallelDescriptor::ReduceBoolAnd(restored);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(restored,
                    "Native circuit cancellation failed; no reversible step is available");
            }
            m_WarpX->RestoreParticlesAtImplicitStepStart();
            m_eulerian_energy->Cancel();
        }
        return exit_status;
    }

    // The verified joint lease spans this publication seam. Final-time recovery
    // remains separate and unchanged. A failed publication is still reversible.
    if(m_eulerian_energy && m_eulerian_energy->JointVacuumActive()) {
        if(!m_eulerian_energy->PublishJointVacuumStage()) {
            AMREX_ALWAYS_ASSERT(retained_acceptance);
            cancel_retained(); return -12;
        }
    } else {
        UpdateWarpXFields( m_E, false, start_time );
    }
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);
    if (NativeEndpointEnabled()) {
        bool private_joint_capture=false;
        if(m_native_pmc_joint_requested) {
            int lower=m_eulerian_energy && m_eulerian_energy->JointVacuumActive(),upper=lower;
            amrex::ParallelDescriptor::ReduceIntMin(lower);
            amrex::ParallelDescriptor::ReduceIntMax(upper);
            if(lower!=upper) {
                AMREX_ALWAYS_ASSERT(retained_acceptance);
                cancel_retained();return -12;
            }
            private_joint_capture=lower!=0;
        }
        if(private_joint_capture) {
            if(!m_eulerian_energy->CaptureJointVacuumEndpointStage()) {
                AMREX_ALWAYS_ASSERT(retained_acceptance);
                cancel_retained();return -12;
            }
        } else {
            CaptureNativeEndpointStage(*m_WarpX,start_time+m_theta*m_dt,m_dt,NativeCoilCurrentContext());
        }
    }
    if(stopping_carry)m_eulerian_energy->BindStoppingCarryStage();
    if(m_native_paired_fields && !m_native_paired_fields->CaptureStageAccounting()) {
        cancel_retained();return -17;
    }
    if(m_native_longitudinal_pending) {
        if(m_native_paired_fields)m_native_longitudinal_pending->PairedCaptureStage(m_dt,m_theta);
        else m_native_longitudinal_pending->CaptureStage(m_dt,m_theta);
    }

    // Accepted-source booking must use the same midpoint temperature and
    // magnetic direction as the force, before endpoint field extrapolation.
    if (!m_eulerian_energy) { m_hybrid_pic_model->CaptureImplicitDissipationCoefficients(); }

    const amrex::Real new_time = start_time + m_dt;

    // Advance particles from t^{n+1/2} to t^{n+1}. The joint transaction owns
    // its verified-stage/finished-state transition around the native call.
    if(m_eulerian_energy && m_eulerian_energy->JointVacuumActive()) {
        if(!m_eulerian_energy->FinishJointVacuumParticles(new_time)) {
            AMREX_ALWAYS_ASSERT(retained_acceptance);
            cancel_retained(); return -12;
        }
    } else {
        m_WarpX->FinishImplicitParticleUpdate(new_time);
    }
    if (m_eulerian_energy) {
        // The accepted-only ion consumer uses endpoint NGP ownership, not
        // midpoint tile ownership. All source coefficients were frozen before
        // extrapolation, and no RNG is used in residuals or PrepareAcceptance.
        bool const boundary_complete=m_eulerian_energy->CommitParticleBoundaries();
        if(!boundary_complete && retained_acceptance && m_eulerian_energy->CanCancel()) {
            // The owned axial image has no tally/RNG/buffer barrier. A general
            // irreversible boundary action still makes CanCancel false.
            cancel_retained();return -12;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(boundary_complete,
            "Accepted particle boundary mismatch after native tallies/buffer mutation; terminal step failure");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_eulerian_energy->CommitIonExchange(),
            "Accepted native ion exchange failed; stochastic failure cannot be retried without RNG restoration");
    }

    if (m_circuit_native) {
        if(retained_acceptance) {
            std::string error;
            bool candidate=NativeCircuitCoupler().TryCommitRetainedDarwinDeviceStep(
                start_time,m_dt,m_theta,m_darwin_circuit_scale_tolerance,error);
            amrex::ParallelDescriptor::ReduceBoolAnd(candidate);
            if(!candidate) {
                amrex::Print()<<"Native circuit candidate declined: "<<error<<"\n";
                cancel_retained();return -13;
            }
            m_native_circuit_step_open=false;
        } else { CommitNativeCircuitStage(start_time); }
    }

    warpx::darwin::increment::Audit(*m_WarpX,"stage");
    AuditNativeVacuumPotential(*m_WarpX, "stage");
    // Only the actual verified/finished transaction can preserve the affine
    // magnetic endpoint. The current certificate is checked after the literal
    // Je/D rotations below; every failure remains inside whole-owner Cancel.
    bool owned_endpoint_ampere=false;
#if defined(WARPX_DIM_RZ)
    if(m_native_owned_endpoint_ampere_requested) {
        bool owner=m_eulerian_energy&&m_eulerian_energy->JointVacuumActive();
        amrex::ParallelDescriptor::ReduceBoolAnd(owner);
        if(!owner || !m_eulerian_energy->PrepareJointVacuumEndpointAmpere(new_time)) {
            AMREX_ALWAYS_ASSERT(retained_acceptance);
            cancel_retained();return -12;
        }
        owned_endpoint_ampere=true;
    }
#endif
    // Advance fields from t^{n+theta} to t^{n+1}
    FinishFieldUpdate( new_time, owned_endpoint_ampere );
    if(m_native_longitudinal_pending) {
        if(m_native_paired_fields) {
            using ablastr::fields::Direction;
            m_native_longitudinal_pending->PairedExtrapolateElectric(
                m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",0),m_theta);
            m_native_longitudinal_pending->MaterializeElectric();
            auto const receipt=m_native_paired_fields->MaterializeEndpoint();
            if(!receipt.materialized) {cancel_retained();return -17;}
            // Original high-only field representation is now published. The
            // following endpoint solve receives no private low shadow.
            m_WarpX->SetElectricFieldAndApplyBCs(m_E,new_time);
            for(int c=0;c<3;++c)amrex::MultiFab::Add(
                *m_WarpX->m_fields.get(FieldType::Efield_fp,Direction{c},0),
                *m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0),0,0,1,
                m_WarpX->m_fields.get(FieldType::Efield_fp,Direction{c},0)->nGrowVect());
            m_WarpX->FillBoundaryE(m_WarpX->m_fields.get(FieldType::Efield_fp,Direction{0},0)->nGrowVect(),true);
            m_WarpX->ApplyEfieldBoundary(0,PatchType::fine,new_time);
            auto const publication=m_native_paired_fields->CompletePublication();
            if(!publication.complete) {cancel_retained();return -17;}
        } else m_native_longitudinal_pending->ExtrapolateElectric(
            m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",0),m_theta);
    }
    if (vacuum_phi_old) {
        // Match FinishFieldUpdate's EL coefficients and all allocated guards.
        // Linear gradient commutes with this update; never reproject EL here.
        auto& phi = *m_WarpX->m_fields.get("hybrid_phi_darwin_fp", 0);
        amrex::Real const c0 = 1.0_rt / m_theta;
        amrex::Real const c1 = 1.0_rt - c0;
        if(warpx::darwin::increment::Enabled())warpx::darwin::increment::Extrapolate(
            phi,*vacuum_phi_old,*m_WarpX->m_fields.get(warpx::darwin::increment::scalar_low,0),m_theta);
        else amrex::MultiFab::LinComb(phi, c0, phi, 0, c1, *vacuum_phi_old, 0,
                                0, phi.nComp(), phi.nGrowVect());
        AuditNativeVacuumPotential(*m_WarpX, "endpoint");
        warpx::darwin::increment::DiscardEndpointLow(*m_WarpX,m_dt,m_theta);
    }

    // Opt-in: apply the particle boundary conditions and re-bin before the
    // end-of-step deposits below (see the member documentation). The rho
    // deposition guard range covers displacements of up to
    // (nox + particles.max_grid_crossings - nox/2 - 1) cells from the home
    // tile (GuardCellManager: rho band nox + max_grid_crossings, J band one
    // less; default max_grid_crossings 1), and the full-dt extrapolation
    // above can exceed it for warm boundary populations. The midpoint
    // deposits inside every nonlinear iteration are bound by the same band
    // and are NOT covered by this knob: widen particles.max_grid_crossings
    // for large time steps.
    if (m_redistribute_before_end_deposits) {
        m_WarpX->GetPartContainer().Redistribute();
    }

    // Complete the electron-energy step: apply the stochastic ion-heating
    // realization once with converged states, refresh Pe^{n+1}, and reset
    // the QDSMC markers.
    if (m_eulerian_energy) {
        if (retained_acceptance) { m_eulerian_energy->PublishCandidate(); }
        else { m_eulerian_energy->Commit(); }
    } else if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        m_hybrid_pic_model->QDSMCFinishImplicitStep(m_dt, m_theta, new_time);
    } else if (!m_darwin) {
        // Closure path: re-evaluate Pe^{n+1} (and the diagnostic T_e
        // mirror) from a true end-of-step density deposit. The in-solve
        // closure evaluations consumed midpoint-position deposits, so
        // without this the dumped Pe/Te lag the ion state by half a step
        // -- a first-order error in Te-based convergence metrics on a
        // scheme whose dynamics are second order. The energy-equation
        // branch above already ends with an equivalent end-of-step
        // recovery inside QDSMCFinishImplicitStep. Skipped under darwin:
        // the entry E_L^n solve consumes the as-left Pe together with the
        // as-left rho_fp, and re-labeling only one of that pair (or both)
        // changes validated darwin evolution -- the entry-state
        // time-labeling there is a separate, jointly-decided item.
        m_hybrid_pic_model->CalculateElectronPressureAtStepEnd();
    }

    // Refresh the per-species temperature deposits from the end-of-step
    // particle state (after the ion-heating realization above). The
    // explicit scheme deposits these every step; without this call the
    // T_<species> diagnostics would hold their initialization values for
    // the whole run. Species without do_temperature_deposition are
    // skipped inside.
    m_WarpX->GetPartContainer().DepositTemperatures(m_WarpX->m_fields, 0.0_rt);

    // Leave hybrid_current_fp_plasma holding the delivered end-of-step
    // Ampere closure, J_plasma^{n+1} = curl(B^{n+1})/mu0 - J_ext. The
    // residual evaluations left the theta-stage value; everything that
    // reads the register between steps must observe the same t^{n+1}
    // state the explicit loop ends on: the coherent afterEpush /
    // afterEsolve python callbacks (e.g. a segregated circuit coupler
    // measuring the plasma flux linkage), the particle-level resistive
    // drag (collisions run after OneStep under the implicit schemes),
    // and the displacement-current diagnostic. Runs after
    // QDSMCFinishImplicitStep, which consumes the theta-stage value.
    // Split-field externals: Bfield_fp holds end-of-step TOTALS here, so
    // strip the external field around the curl -- the plasma-current
    // register must stay response-only (the explicit loop's final refresh
    // runs before its external add-back; a totals-frame curl would leak
    // the coil field's O(h^2) discrete curl into the circuit flux-linkage
    // probes and the resistive drag).
    const bool strip_ext =
        m_hybrid_pic_model->m_add_external_fields && !m_darwin;
    if (strip_ext) { AddSplitExternalFields(-1.0_rt); }
    if (!m_darwin || !warpx::darwin::TryCalculateNativeSplitDarwinAmpere(*m_WarpX)) {
        CalculateNativePlasmaCurrent(
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
            start_time + m_dt);
    }
    if (strip_ext) { AddSplitExternalFields(1.0_rt); }
    if (m_darwin)
    {
        ApplyDarwinDisplacementCurrent(m_dt);
    }

    // Re-evaluated E finisher: overwrite the extrapolated E^{n+1} with the
    // generalized Ohm's law evaluated at the DELIVERED end-of-step state
    // (total B^{n+1}, the delivered plasma current refreshed above, the
    // same ion-deposit family the theta-stage used), as the explicit
    // hybrid loop finishes its step. The extrapolated finisher is a
    // -(1-theta)/theta recursion on the stored algebraic field.
    if (m_e_finisher_reevaluate) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_darwin,
            "implicit_evolve.hybrid_e_finisher = reevaluate is not "
            "implemented for the Darwin field split");
        if (!m_hybrid_pic_model->m_solve_electron_energy_equation) {
            m_hybrid_pic_model->CalculateElectronPressure();
        }
        // per-level variant: the multi-level HybridPICSolveE fires the
        // afterEpush python callback, which the implicit step already
        // fires exactly once at the delivered state (WarpX::OneStep) --
        // the finisher solve must not add a second firing per step
        {
            ablastr::fields::MultiLevelVectorField E_fp =
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
            ablastr::fields::MultiLevelVectorField J_fp =
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
            ablastr::fields::MultiLevelVectorField B_fp =
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
            ablastr::fields::MultiLevelScalarField r_fp =
                m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                m_hybrid_pic_model->HybridPICSolveE(
                    E_fp[lev], J_fp[lev], B_fp[lev], *r_fp[lev],
                    m_WarpX->GetEBUpdateEFlag()[lev], lev,
                    false /* solve_for_Faraday */,
                    true /* include_resistivity */);
            }
        }
        {
            using ablastr::fields::Direction;
            amrex::IntVect const ngE = m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{0}, 0)->nGrowVect();
            m_WarpX->FillBoundaryE(ngE, true /* sync nodal points */);
        }
        // keep the solver vector consistent with the delivered field
        m_E.Copy(FieldType::Efield_fp);
    }

    // Density-band statistics feeding (DSMC-style split in depleted
    // cells): runs at step boundaries only, never inside the residual.
    // Band limits are configured per species in units of the hybrid
    // n_floor; the merge relief valve is the stock velocity-coincidence
    // resampler.
    {
        auto& mpc = m_WarpX->GetPartContainer();
        const amrex::Real rho_floor =
            m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
        for (int isp = 0; isp < mpc.nSpecies(); ++isp) {
            auto* pc = dynamic_cast<PhysicalParticleContainer*>(
                &mpc.GetParticleContainer(isp));
            if (pc == nullptr) { continue; }
            const int interval = pc->HybridSplitInterval();
            if (interval <= 0 || ((a_step + 1) % interval != 0)) { continue; }
            const amrex::MultiFab& rho0 =
                *m_WarpX->m_fields.get(FieldType::rho_fp, 0);
            pc->SplitDepletedBand(rho0,
                pc->HybridSplitBandLo()*rho_floor,
                pc->HybridSplitBandHi()*rho_floor, 0);
            pc->Redistribute();
        }
    }

    // Electron inertia: rotate the per-step nodal Je history from the
    // MEASURED delivered state -- hybrid_current_fp_plasma now holds
    // J_plasma^{n+1} (including the Darwin displacement piece above), and
    // current_fp still holds the same ion-deposit family the theta-stage
    // assemblies used. Runs here (not in FinishFieldUpdate) so the stored
    // value is a measurement, not an extrapolation of a stored value.
    if (m_hybrid_pic_model->m_include_electron_inertia) {
        m_hybrid_pic_model->RotateElectronInertiaHistory(m_theta);
        if (NativeEndpointEnabled()) { RotateNativeEndpointCurrent(*m_WarpX,m_theta);
        if(stopping_carry)m_eulerian_energy->BindStoppingCarryRotation(m_theta);
        if(m_native_longitudinal_pending)m_native_longitudinal_pending->RotateDisplacement(m_theta); }
    }
    if(owned_endpoint_ampere &&
       !m_eulerian_energy->VerifyJointVacuumEndpointAmpere(new_time)) {
        amrex::Print()<<"Owned endpoint physical Ampere check declined\n";
        cancel_retained();return -12;
    }
    if (m_eulerian_energy) {
        m_eulerian_energy->PublishEndpointDensity();
        if (retained_acceptance) {
            using warpx::implicit::AcceptancePhase;
            using ablastr::fields::Direction;
            if (!warpx::implicit::CheckNativeAcceptance(*m_WarpX,AcceptancePhase::BeforeEndpoint)) {
                cancel_retained();
                return -12;
            }
            // Use the original accepted transverse baseline and native
            // endpoint maps. The retained path changes only publication and
            // rollback ownership, never the physical time level or gauge.
            bool endpoint_ok=false;
            char const* endpoint_reason="Native static vacuum candidate declined";
            if(NativeVacuumEndpointEnabled()) {
                if(m_circuit_native) {
                    bool ready=NativeCircuitCoupler().RetainedNativeEndpointReady();
                    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
                    if(!ready) {
                        amrex::Print()<<"Native vacuum/circuit candidate lacks a retained closed endpoint phase\n";
                        cancel_retained();return -13;
                    }
                    endpoint_reason="Native vacuum/circuit endpoint candidate declined";
                }
                auto const& baseline=m_Eold.getArrayVec()[0];
                auto exact_free=m_E.getArrayVec()[0];
                endpoint_ok=TryConstrainNativeVacuumEndpointField(*m_WarpX,new_time,false,
                    {baseline[0],baseline[1],baseline[2]},
                    m_native_pmc_joint_requested ? &exact_free : nullptr);
            } else {
            // Coil and paired paths retain the native transverse output.
            // Other paths keep the existing full-E minus EL publication below.
            auto exact_free=m_E.getArrayVec()[0];
            auto const endpoint=TryConstrainNativeEndpointField(*m_WarpX,new_time,[&]() {
                auto A=m_WarpX->m_fields.get_alldirs("hybrid_A_fp",0);
                for (int d=0;d<3;++d) {
                    auto const& Et=*m_WarpX->m_fields.get(FieldType::Efield_fp,Direction{d},0);
                    amrex::MultiFab::Copy(*A[d],Et,0,0,1,A[d]->nGrowVect()); A[d]->mult(-1.);
                }
                DarwinApplyABoundary(new_time);
                auto Bd=m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp,0);
                m_WarpX->get_pointer_fdtd_solver_fp(0)->ComputeCurlA(Bd,A,
                    m_WarpX->GetEBUpdateBFlag()[0],0,
                    DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
                m_WarpX->FillBoundaryB(Bd[0]->nGrowVect(),true);
                m_hybrid_pic_model->CalculatePlasmaCurrent(Bd,m_WarpX->GetEBUpdateEFlag()[0],0);
            },(warpx::darwin::NativeCoilCurrentEnabled() || m_native_paired_fields)?&exact_free:nullptr);
            endpoint_ok=bool(endpoint);endpoint_reason=endpoint.reason;
            }
            if (!endpoint_ok) {
                amrex::Print()<<"Native endpoint candidate declined: "<<endpoint_reason<<"\n";
                cancel_retained();
                return -11;
            }
            if(m_native_endpoint_pair_requested){
                auto exact=m_E.getArrayVec()[0];
                if(!m_native_endpoint_pair->CopyEndpointTransverse({exact[0],exact[1],exact[2]})){
                    cancel_retained();return -11;
                }
                auto const& raw=*m_WarpX->m_fields.get(FieldType::rho_fp,0);
                amrex::MultiFab density(raw,amrex::make_alias,0,1);
                ablastr::fields::VectorField th{exact[0],exact[1],exact[2]};
                auto checked=m_native_endpoint_pair->CheckEndpoint(density,
                    *m_WarpX->m_fields.get("hybrid_rho_vacmask_fp",0),&th);
                if(!checked.compatible||!checked.converged){cancel_retained();return -11;}
                if(m_native_endpoint_pair_accept_requested&&!m_native_endpoint_pair->PublishEndpointAuxiliary()){
                    cancel_retained();return -11;
                }
                retained_candidate->m_impl->endpoint_pair=m_native_endpoint_pair->EndpointLease();
            } else if(NativeVacuumEndpointEnabled() && NativeCircuitDriveEnabled()) {
                // The full electric publication can hide the much smaller Et
                // below an EL ULP. Preserve the exact native solved origin,
                // as initialization, restart and the next step entry do.
                for(int d=0;d<3;++d) amrex::MultiFab::Copy(*m_E.getArrayVec()[0][d],
                    *m_WarpX->m_fields.get(NativeVacuumTransverseRecordName(),Direction{d},0),0,0,1,0);
            } else if(!warpx::darwin::NativeCoilCurrentEnabled() && !m_native_paired_fields &&
                      !m_native_pmc_joint_requested) {
                m_E.Copy(FieldType::Efield_fp);
                for (int d=0;d<3;++d) amrex::MultiFab::Subtract(*m_E.getArrayVec()[0][d],
                    *m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{d},0),0,0,1,0);
            }
            if (m_eulerian_energy->JointVacuumActive() &&
                !m_eulerian_energy->CheckJointVacuumEndpointSupport()) {
                cancel_retained(); return -12;
            }
            if (!warpx::implicit::CheckNativeAcceptance(*m_WarpX,AcceptancePhase::AfterEndpoint)) {
                cancel_retained();
                return -12;
            }
            // All validated candidate state and snapshots outlive this stack
            // frame on the explicit deferred route. No receipt is published.
            if(m_native_longitudinal_pending) {
                bool const matches=m_native_longitudinal_pending->Matches(*m_WarpX,new_time,a_step+1);
                auto const proof=m_native_longitudinal_pending->Statistics();
                amrex::Print()<<std::setprecision(17)<<"Native stopping producer certificate: step="<<a_step
                    <<" valid="<<matches<<" E_radius="<<proof.electric_radius<<" D_radius="<<proof.displacement_radius
                    <<" operations="<<proof.operations<<" local_bytes="<<proof.local_bytes<<"\n";
            }
            if(material_symmetric){
                retained_candidate->m_impl->phase=Candidate::Impl::Phase::Ready;
                retained_candidate->m_impl->endpoint_ready=true;
                retained_candidate->m_impl->stopping_carry=m_eulerian_energy->BindStoppingCarryEndpoint(new_time,a_step+1);
                if(!warpx::implicit::CheckNativeAcceptance(*m_WarpX,AcceptancePhase::AfterSymmetricField)){
                    cancel_retained();return -14;
                }
                auto const status=retained_candidate->ApplyMaterialHalf(false,*symmetric_source);
                if(status!=warpx::implicit::NativeCandidateStatus::Success){
                    if(status==warpx::implicit::NativeCandidateStatus::Unsupported)cancel_retained();
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(status!=warpx::implicit::NativeCandidateStatus::Terminal,
                        "Material S2 failed without whole-step cancellation");return -14;
                }
            } else if(symmetric_source) {
                retained_candidate->m_impl->phase=Candidate::Impl::Phase::FieldReady;
                if(!warpx::implicit::CheckNativeAcceptance(*m_WarpX,AcceptancePhase::AfterSymmetricField)) {
                    cancel_retained();return -14;
                }
                auto const source=retained_candidate->ApplySource(Candidate::SourcePhase::MidpointPost,
                    nullptr,symmetric_source->physical,&symmetric_source->thermal);
                if(source.status!=warpx::implicit::NativeCandidateStatus::Success) {
                    if(source.status==warpx::implicit::NativeCandidateStatus::Unsupported)cancel_retained();
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(source.status!=warpx::implicit::NativeCandidateStatus::Terminal,
                        "Symmetric post-source failed without a restorable step");
                    amrex::Print()<<"Symmetric post-source declined: "<<source.reason<<"\n";return -14;
                }
            } else {
                retained_candidate->m_impl->phase=Candidate::Impl::Phase::Ready;
                retained_candidate->m_impl->endpoint_ready=true;
            }
            if(stopping_carry&&!material_symmetric)retained_candidate->m_impl->stopping_carry=
                m_eulerian_energy->BindStoppingCarryEndpoint(new_time,a_step+1);
            if (deferred) {
                *deferred=std::move(retained_candidate);
                return exit_status;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                retained_candidate->Finalize()==warpx::implicit::NativeCandidateStatus::Success,
                "Retained candidate finalization failed; step is terminal");
        } else if (NativeEndpointEnabled()) { DiagnosticConstrainEndpoint(new_time,false); }
        if (!retained_acceptance) { m_WarpX->DiscardSavedImplicitParticleState(); }
    }

    return exit_status;
}

void
ThetaImplicitHybrid::ApplyDarwinDisplacementCurrent (amrex::Real interval)
{
    // Called only after Ampere rebuilds Jp. Both push correction and Ohm
    // consume J = curl(B)/mu0 - epsilon0*dEL/dt at the same stage.
    using ablastr::fields::Direction;
    auto thermal_low=m_eulerian_energy ? m_eulerian_energy->CurrentRemainderOutput(interval)
        : ablastr::fields::VectorField{};
    if(auto* paired=warpx::darwin::NativeActivePairedFields(*m_WarpX)) {
        AMREX_ALWAYS_ASSERT(interval==.5*m_dt);
        paired->SubtractDisplacement(m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0),thermal_low[0]?&thermal_low:nullptr);
        if(thermal_low[0])m_eulerian_energy->PublishCurrentRemainder();
        return;
    }
    if(m_joint_stage_owner&&m_joint_stage_owner->CurrentArithmeticActive()) {
        m_joint_stage_owner->SubtractArithmeticDisplacement(interval,thermal_low[0]?&thermal_low:nullptr);
        if(thermal_low[0])m_eulerian_energy->PublishCurrentRemainder();
        return;
    }
    amrex::Real const inv_thetadt = 1.0_rt / interval;
    for (int lev = 0; lev < m_num_amr_levels; ++lev)
    {
        for (int dir = 0; dir < 3; ++dir)
        {
            amrex::MultiFab& Jp =
                *m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma, Direction{dir}, lev);
            amrex::MultiFab const& EL =
                *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
            amrex::MultiFab const& EL_old =
                *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
            warpx::darwin::SubtractDisplacementCurrent(
                Jp, EL, EL_old, PhysConst::epsilon_0 * inv_thetadt,
                warpx::darwin::increment::Low(*m_WarpX,dir),lev==0?thermal_low[dir]:nullptr);
        }
    }
    if(thermal_low[0])m_eulerian_energy->PublishCurrentRemainder();
    if (EB::enabled())
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            auto current = m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
            auto& flags = m_WarpX->GetEBUpdateEFlag()[lev];
            if (m_hybrid_pic_model->m_use_conformal_eb &&
                m_hybrid_pic_model->m_conformal_wall_conductor)
            {
                m_hybrid_pic_model->ZeroConductorEdges(current, flags, lev);
            }
            else
            {
                for (int d = 0; d < 3; ++d)
                {
                    for (amrex::MFIter mfi(*current[d], amrex::TilingIfNotGPU()); mfi.isValid();
                         ++mfi)
                    {
                        auto const jp = current[d]->array(mfi);
                        auto const open = flags[d]->const_array(mfi);
                        amrex::ParallelFor(mfi.tilebox(),
                                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                           {
                                               if (!open(i, j, k))
                                               {
                                                   jp(i, j, k) = 0.0_rt;
                                               }
                                           });
                    }
                }
            }
            for (int d = 0; d < 3; ++d)
            {
                current[d]->FillBoundary(m_WarpX->Geom(lev).periodicity());
            }
        }
    }
}

bool ThetaImplicitHybrid::UseMassMatrixCurrentIncrement () const
{
    return m_use_mass_matrices_jacobian && m_mass_matrices_density_projection &&
           !m_particle_suborbits && NativeCorrelatedIncrementEnabled();
}

void ThetaImplicitHybrid::ComposeMassMatrixCurrentIncrement ()
{
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass_matrix_density[lev] &&
            m_mass_matrix_density[lev]->IsCaptured(),
            "MM current increment requires its fully deposited nonlinear base");
        m_mass_matrix_density[lev]->CaptureCurrentIncrementAndCompose(
            m_WarpX->m_fields.get_alldirs(FieldType::current_fp,lev));
    }
}

void ThetaImplicitHybrid::InvalidateMassMatrixCurrentIncrement ()
{
    for (auto& predictor : m_mass_matrix_density) {
        if (predictor) { predictor->InvalidateCurrentIncrement(); }
    }
}

void ThetaImplicitHybrid::UpdateMassMatrixDensity (bool const from_jacobian)
{
    if (!m_live_ion_density) { return; }
    if(m_native_pmc_mm_requested) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(NativePrivatePMCMassMatrixOwner(*m_WarpX)==this,
            "Private PMC density response lost its live owner/material capability");
    }
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        auto& rho = *m_WarpX->m_fields.get(FieldType::rho_fp,lev);
        int const midpoint = rho.nComp()/2;
        auto const current = m_WarpX->m_fields.get_alldirs(FieldType::current_fp,lev);
        auto& predictor = m_mass_matrix_density[lev];
        if (!predictor) { predictor = std::make_unique<MassMatrixDensityProjection>(); }
        if (!from_jacobian || !m_use_mass_matrices_jacobian) {
            if (m_continuity_density) {
                predictor->DepositEndpointAverage(*m_WarpX,lev,rho,midpoint);
            }
            if (m_mass_matrices_density_projection) { predictor->Capture(rho,midpoint,current); }
        } else if (m_mass_matrices_density_projection) {
            if (UseMassMatrixCurrentIncrement()) {
                predictor->ApplyCurrentIncrement(*m_WarpX,lev,rho,midpoint,0.5*m_dt);
            } else {
                predictor->Apply(*m_WarpX,lev,rho,midpoint,current,0.5*m_dt);
            }
        }
    }
}

void ThetaImplicitHybrid::RefreshDarwinELong (amrex::Real theta_time)
{
    ablastr::fields::MultiLevelScalarField rho_half_alias;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_half_store(m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        auto& rho = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
        rho_half_store[lev] = std::make_unique<amrex::MultiFab>(
            rho, amrex::make_alias, rho.nComp()/2, 1);
        rho_half_alias.push_back(rho_half_store[lev].get());
    }
    ablastr::fields::MultiLevelVectorField full_source;
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> source_store;
    bool const full_longitudinal = NativeFullOhmLongitudinalEnabled();
    if (full_longitudinal) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_darwin_segregated_solve && m_eulerian_energy,
            "Full Ohm longitudinal stage requires native Eulerian segregated closure");
        full_source.resize(m_num_amr_levels);
        source_store.resize(m_num_amr_levels);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            auto const electric = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
            for (int c = 0; c < 3; ++c) {
                auto const& field = *electric[c];
                source_store[lev][c] = std::make_unique<amrex::MultiFab>(
                    field.boxArray(), field.DistributionMap(), field.nComp(),
                    field.nGrowVect());
                amrex::MultiFab::Copy(*source_store[lev][c], field, 0, 0,
                    field.nComp(), field.nGrowVect());
                // The native Ohm wrapper applies physical BCs through the
                // registered E fields. Save and restore them in full; a
                // private output pointer alone would miss those BCs.
                full_source[lev][c] = electric[c];
            }
            auto const ions = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
            auto const magnetic = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
            // Component zero of the registry may still be the previous
            // trial. Match ComputeRHS: consume this trial's midpoint alias.
            auto const& rho = *rho_half_alias[lev];
            m_hybrid_pic_model->HybridPICSolveE(full_source[lev], ions, magnetic, rho,
                m_WarpX->GetEBUpdateEFlag()[lev], lev, false, true,
                m_eulerian_energy->Dissipation());
            if(NativeVacuumEndpointEnabled()) {
                CompleteNativeVacuumLongitudinalSource(*m_WarpX,m_theta*m_dt);
            }
        }
    }
    m_hybrid_pic_model->ComputeDarwinELong(rho_half_alias, theta_time,
        m_darwin_segregated_solve, full_longitudinal ? &full_source : nullptr);
    if (full_longitudinal) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int c = 0; c < 3; ++c) {
                auto& field = *full_source[lev][c];
                amrex::MultiFab::Copy(field, *source_store[lev][c], 0, 0,
                    field.nComp(), field.nGrowVect());
            }
        }
    }
}

int ThetaImplicitHybrid::SolveDarwinSegregated (amrex::Real start_time, int a_step)
{
    BL_PROFILE("ThetaImplicitHybrid::SolveDarwinSegregated()");
    warpx::darwin::increment::Begin(*m_WarpX);
    if(m_eulerian_energy && m_eulerian_energy->JointVacuumRequested()) {
        int const joint=m_eulerian_energy->SolveJointVacuum(a_step);
        if(joint!=4)return joint; // Empty V: unchanged positive-density route below.
    }
    if(m_native_paired_fields){m_native_paired_fields->BeginField(m_dt,m_theta,start_time,a_step);m_native_longitudinal_pending->PairedBeginField();}
    // The density mask is frozen only within this step. Rebuild its scalar
    // correction space before projecting the first guess and Krylov updates.
    m_vacuum_gauge_solver.reset();
    m_vacuum_gauge_op.reset();
    ProjectDarwinVacuumGauge(m_E);
    using ablastr::fields::Direction;
    // Keep ghosts too: a rejected candidate must not alter the field gathered
    // by the converged particle/energy stage. Scratch is local to this step.
    amrex::Vector<amrex::Array<std::unique_ptr<amrex::MultiFab>, 3>> previous(
        m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0; dir < 3; ++dir) {
            auto const& field = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
            previous[lev][dir] = std::make_unique<amrex::MultiFab>(
                field.boxArray(), field.DistributionMap(), field.nComp(), field.nGrowVect());
        }
    }
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_phi(m_num_amr_levels);
    for (int lev=0;lev<m_num_amr_levels;++lev) {
        auto const& phi=*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",lev);
        previous_phi[lev]=std::make_unique<amrex::MultiFab>(
            phi.boxArray(),phi.DistributionMap(),phi.nComp(),phi.nGrowVect());
    }
    // This opt-in changes the outer iteration, never the physical residual or
    // its independent raw constraint acceptance test.
    bool use_schur=false;
    amrex::ParmParse pp("implicit_evolve");
    pp.query("darwin_longitudinal_schur",use_schur);
    std::unique_ptr<warpx::darwin::DarwinLongitudinalSchur> schur;
    std::unique_ptr<amrex::MultiFab> kappa;
    if (use_schur && !NativeVacuumEndpointEnabled()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_num_amr_levels==1 &&
            !m_hybrid_pic_model->m_esolve_tensor && !m_hybrid_pic_model->m_esolve_curlcurl,
            "Darwin longitudinal Schur requires one physical level and E-form Ohm law");
        auto const& geom=m_WarpX->Geom(0);
        auto const& rho=*m_WarpX->m_fields.get(FieldType::rho_fp,0);
        auto const& phi=*previous_phi[0];
        auto const cells=amrex::convert(rho.boxArray(),amrex::IntVect(0));
        warpx::darwin::LongitudinalSchurOptions options;
        options.theta=m_theta;
        options.compatible_yee=m_hybrid_pic_model->UseCompatibleYeeInertia();
        options.max_semicoarsening_levels=m_hybrid_pic_model->m_darwin_poisson_semicoarsening;
        options.semicoarsening_direction=m_hybrid_pic_model->m_darwin_poisson_semicoarsening_direction;
        options.verbose=m_hybrid_pic_model->m_darwin_poisson_verbosity;
        options.djedt_only=m_hybrid_pic_model->m_electron_inertia_djedt_only;
        options.bdf2=m_hybrid_pic_model->m_electron_inertia_bdf2;
        options.embedded_boundaries=EB::enabled();
        options.azimuthal_modes=WarpX::ncomps;
        options.output_ghosts=phi.nGrowVect();
        for (int c=0;c<3;++c) {
            options.output_ghosts=amrex::max(options.output_ghosts,previous[0][c]->nGrowVect());
        }
        pp.query("darwin_schur_relative_tolerance",options.relative_tolerance);
        pp.query("darwin_schur_absolute_tolerance",options.absolute_tolerance);
        pp.query("darwin_schur_max_iterations",options.max_iterations);
        pp.query("darwin_schur_restart_length",options.restart_length);
        pp.query("darwin_schur_pc_cycles",options.preconditioner_cycles);
        for (int d=0;d<AMREX_SPACEDIM;++d) {
            for (int side=0;side<2;++side) {
                auto const boundary=side ? WarpX::field_boundary_hi[d] : WarpX::field_boundary_lo[d];
                auto& selected=side ? options.upper[d] : options.lower[d];
                using Boundary=warpx::darwin::LongitudinalBoundary;
                if (geom.isPeriodic(d)) { selected=Boundary::Periodic; }
#if defined(WARPX_DIM_RZ)
                else if (d==0 && side==0) { selected=Boundary::Axis; }
#endif
                else if (boundary==FieldBoundaryType::PEC) { selected=Boundary::PEC; }
                else if (boundary==FieldBoundaryType::PMC) { selected=Boundary::PMC; }
                else { amrex::Abort("Darwin longitudinal Schur requires axis/PEC/PMC/periodic boundaries"); }
            }
        }
        schur=std::make_unique<warpx::darwin::DarwinLongitudinalSchur>(
            geom,cells,rho.DistributionMap(),options);
        kappa=std::make_unique<amrex::MultiFab>(rho.boxArray(),rho.DistributionMap(),1,0);
    }
    amrex::Real field_rtol, field_atol;
    int field_maxits;
    m_nlsolver->GetSolverParams(field_rtol, field_atol, field_maxits);
    amrex::ignore_unused(field_maxits);
    WarpXSolverVec residual;
    residual.Define(m_E);
    bool use_aitken=false;
    pp.query("darwin_outer_aitken",use_aitken);
    bool clamp_positive_secant=false;
    amrex::ParmParse("endpoint_diagnostic").query("longitudinal_aitken_clamp",clamp_positive_secant);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!clamp_positive_secant||use_aitken,
        "Positive secant clamp requires native longitudinal Aitken");
    std::unique_ptr<warpx::darwin::NativeLongitudinalAitken> outer_inverse;
    if(use_aitken) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(use_schur&&NativeVacuumEndpointEnabled()&&m_num_amr_levels==1,
            "Native longitudinal Aitken requires the single-level joined-vacuum Schur route");
        amrex::GpuArray<int,AMREX_SPACEDIM> low{},high{};
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            low[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
            high[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
        }
        outer_inverse=std::make_unique<warpx::darwin::NativeLongitudinalAitken>(
            m_E.getArrayVec()[0],m_WarpX->Geom(0),low,high);
    }
    // Local memory is discarded on success, failure and physical-step retry.
    amrex::Real field_reference = 0.0_rt;
    for (int outer = 0; outer < m_darwin_outer_max_iterations; ++outer) {
        amrex::Real outer_omega=outer>=m_darwin_outer_relaxation_start ? m_darwin_outer_relaxation : 1.;
        // Grade the coupled solve against this STEP's initial field residual.
        // Starting a fresh relative Newton solve after every tiny E_L update
        // would tighten the physical tolerance repeatedly, down to roundoff.
        ComputeRHS(residual, m_E, start_time, 0, false);
        residual.increment(m_Eold, 1.0_rt);
        residual.increment(m_E, -1.0_rt);
        amrex::Real const field_norm = residual.norm2();
        if (!std::isfinite(field_norm)) { return -8; }
        if (outer == 0 || field_reference == 0.0_rt) { field_reference = field_norm; }
        amrex::Real const field_target = std::max(field_atol, field_rtol * field_reference);
        if (m_darwin_outer_verbose) {
            amrex::Print() << "Darwin field: outer=" << outer
                << " residual=" << field_norm << " target=" << field_target << "\n";
        }
        int status = 3;
        if (m_eulerian_energy) {
            status = m_eulerian_energy->Solve(field_reference, a_step);
            if (status < 0) { return status; }
        }
        else if (m_qdsmc_segregated_solve)
        {
            // The thermal update is required even when the initial field
            // residual already passes. E_L stays fixed throughout this solve.
            status = SolveSegregated(start_time, a_step, field_reference);
            if (status < 0)
            {
                return status;
            }
        }
        else if (!(field_norm == 0.0_rt || field_norm < field_target))
        {
            // E_L stays fixed for every Jv and line search in this solve.
            m_nlsolver->SetConvergenceReferenceNorm(field_reference);
            m_nlsolver->Solve(m_E, m_Eold, start_time, m_dt, a_step);
            m_nlsolver->SetConvergenceReferenceNorm(0.0_rt);
            status = m_nlsolver->GetExitStatus();
            if (status < 0) { return status; }
            // A permissive/fixed-iteration inner solve is not evidence that
            // the field equation converged. The split requires both gates.
            if (status != 2 && status != 3) { return -8; }
        }
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                auto const& field = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Copy(*previous[lev][dir], field, 0, 0,
                                      field.nComp(), field.nGrowVect());
            }
        }
        for (int lev=0;lev<m_num_amr_levels;++lev) {
            auto const& phi=*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",lev);
            amrex::MultiFab::Copy(*previous_phi[lev],phi,0,0,phi.nComp(),phi.nGrowVect());
        }
        // The successful solver leaves the particle deposits, pressure and
        // inertial field at its accepted state. Test the constraint there.
        if(m_native_paired_fields)m_native_paired_fields->SaveOuter();
        if(m_native_longitudinal_pending)m_native_longitudinal_pending->BeginOuter();
        RefreshDarwinELong(start_time + m_theta*m_dt);
        amrex::Real change = 0.0_rt;
        amrex::Real scale = 0.0_rt;
        ablastr::fields::ConstVectorField paired_defect{};
        if(m_native_paired_fields) {
            paired_defect=m_native_paired_fields->ConstraintDefect(
                m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",0),
                {previous[0][0].get(),previous[0][1].get(),previous[0][2].get()});
        }
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                auto const& field = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{dir}, lev);
                auto const& old = *previous[lev][dir];
                if (!field.is_finite()) { return -8; }
                amrex::MultiFab diff(field.boxArray(), field.DistributionMap(),
                                    field.nComp(), 0);
                if(m_native_paired_fields)amrex::MultiFab::Copy(diff,*paired_defect[dir],0,0,1,0);
                else {
                amrex::MultiFab::Copy(diff, field, 0, 0, field.nComp(), 0);
                amrex::MultiFab::Subtract(diff, old, 0, 0, field.nComp(), 0);
                }
                change = std::max(change, diff.norminf());
                scale = std::max(scale, std::max(field.norminf(), old.norminf()));
            }
        }
        amrex::Real const target = m_darwin_outer_atol + m_darwin_outer_rtol * scale;
        if (m_darwin_outer_verbose) {
            amrex::Print() << "Darwin segregated: outer=" << outer
                << " constraint_change=" << change << " target=" << target << "\n";
        }
        if (change <= target) {
            // Accept the pair actually solved by Newton. Its independently
            // measured longitudinal defect is bounded by the outer tolerance.
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    auto& field = *m_WarpX->m_fields.get(
                        "hybrid_E_long_fp", Direction{dir}, lev);
                    amrex::MultiFab::Copy(field, *previous[lev][dir], 0, 0,
                                          field.nComp(), field.nGrowVect());
                }
            }
            for (int lev=0;lev<m_num_amr_levels;++lev) {
                auto& phi=*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",lev);
                amrex::MultiFab::Copy(phi,*previous_phi[lev],0,0,phi.nComp(),phi.nGrowVect());
            }
            return status;
        }
        if (use_schur && NativeVacuumEndpointEnabled()) {
            if(m_native_longitudinal_pending)m_native_longitudinal_pending->Invalidate();
            auto const& rho=*m_WarpX->m_fields.get(FieldType::rho_fp,0);
            auto& phi=*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",0);
            amrex::MultiFab delta_phi(phi.boxArray(),phi.DistributionMap(),1,phi.nGrowVect());
            ablastr::fields::VectorField raw,held,delta;
            std::array<std::unique_ptr<amrex::MultiFab>,3> store;
            for(int c=0;c<3;++c) {
                raw[c]=m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0);
                held[c]=previous[0][c].get();
                store[c]=std::make_unique<amrex::MultiFab>(raw[c]->boxArray(),raw[c]->DistributionMap(),1,raw[c]->nGrowVect());
                delta[c]=store[c].get();
            }
            auto correction=warpx::darwin::CorrectNativeVacuumLongitudinal(
                *m_WarpX,rho,rho.nComp()/2,m_theta*m_dt,raw,held,delta_phi,delta);
            amrex::Print()<<"Native vacuum finite Schur: compatible="<<correction.compatible
                <<" converged="<<correction.converged<<" iterations="<<correction.iterations
                <<" residual="<<correction.residual<<" null="<<correction.null_defect<<"\n";
            if(!correction.compatible||!correction.converged)return -8;
            if(outer_inverse) {
                auto const choice=outer_inverse->Select(delta,outer_omega,outer>=m_darwin_outer_relaxation_start,clamp_positive_secant);
                if(!choice.valid)return -8;
                outer_omega=choice.omega;
                if(m_darwin_outer_verbose)amrex::Print()<<"Native longitudinal Aitken omega="<<outer_omega
                    <<" secant="<<choice.secant<<"\n";
            }
            if(warpx::darwin::increment::Enabled()) {
                warpx::darwin::increment::Correct(*m_WarpX,*previous_phi[0],held,delta_phi,delta,outer_omega);
                outer_omega=1.; // The common field/potential omega is already applied.
            } else {
            for(int c=0;c<3;++c)amrex::MultiFab::LinComb(*raw[c],1.,*held[c],0,1.,*delta[c],0,0,1,raw[c]->nGrowVect());
            amrex::MultiFab::LinComb(phi,1.,*previous_phi[0],0,1.,delta_phi,0,0,1,phi.nGrowVect());
            }
        } else if (schur) {
            if(!FillDarwinLongitudinalSchurKappa(*kappa,
                *m_WarpX->m_fields.get(FieldType::rho_fp,0),*m_hybrid_pic_model,m_theta*m_dt))
                return -8;
            if (!schur->Freeze(*kappa)) { return -8; }
            warpx::darwin::DarwinLongitudinalSchur::ConstVector raw,held;
            for (int c=0;c<3;++c) {
                raw[c]=m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0);
                held[c]=previous[0][c].get();
            }
            auto const correction=m_native_paired_fields
                ? schur->CorrectDefect(paired_defect) : schur->Correct(raw,held);
            if (m_darwin_outer_verbose) {
                amrex::Print()<<"Darwin longitudinal Schur: iterations="<<correction.iterations
                    <<" residual="<<correction.residual<<" target="<<correction.target<<"\n";
            }
            if (!correction.converged) { return -8; }
            auto const delta=schur->CorrectionField();
            if(m_native_paired_fields) {
                m_native_paired_fields->ApplySchur(schur->CorrectionPotential(),delta,held,
                    *previous_phi[0],outer_omega,m_E.getArrayVec()[0]);
                m_native_longitudinal_pending->PairedSchurAddition(schur->CorrectionPotential(),delta,
                    m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",0),outer_omega);
                // Phi/L/T use the same omega already. Skip unpaired relaxation
                // and the large-offset warm start below, including at omega=1.
                continue;
            }
            for (int c=0;c<3;++c) {
                auto& field=*m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0);
                amrex::MultiFab::LinComb(field,1.,*previous[0][c],0,1.,*delta[c],0,
                    0,field.nComp(),field.nGrowVect());
            }
            auto& phi=*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",0);
            amrex::MultiFab::LinComb(phi,1.,*previous_phi[0],0,1.,schur->CorrectionPotential(),0,
                0,phi.nComp(),phi.nGrowVect());
            if(m_native_longitudinal_pending)m_native_longitudinal_pending->SchurAddition(
                schur->CorrectionPotential(),delta,m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",0));
        } else if(m_native_longitudinal_pending) {
            m_native_longitudinal_pending->RawGradient(*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",0),
                m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",0));
        }
        if (outer_omega != 1.0_rt) {
            // Grade the unrelaxed constraint defect above. Damping controls
            // the iteration only; it must never reduce the acceptance test.
            for (int lev=0;lev<m_num_amr_levels;++lev) {
                auto& phi=*m_WarpX->m_fields.get("hybrid_phi_darwin_fp",lev);
                amrex::MultiFab::LinComb(phi,outer_omega,phi,0,
                    1.-outer_omega,*previous_phi[lev],0,0,phi.nComp(),phi.nGrowVect());
            }
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    auto& field = *m_WarpX->m_fields.get(
                        "hybrid_E_long_fp", Direction{dir}, lev);
                    amrex::MultiFab::LinComb(field,
                        outer_omega, field, 0,
                        1.0_rt - outer_omega, *previous[lev][dir], 0,
                        0, field.nComp(), field.nGrowVect());
                }
            }
        }
        if(m_native_longitudinal_pending && outer_omega!=1.) {
            ablastr::fields::VectorField held{previous[0][0].get(),previous[0][1].get(),previous[0][2].get()};
            m_native_longitudinal_pending->Relax(held,outer_omega);
        }
        // Warm-start at unchanged E_total: E_T(new) = E_T(old) - delta E_L.
        // The next Newton solve handles any BC-induced change of the A/B map.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                auto& transverse = *m_E.getArrayVec()[lev][dir];
                auto const& field = *m_WarpX->m_fields.get(
                    "hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(transverse, *previous[lev][dir], 0, 0,
                                     transverse.nComp(), 0);
                amrex::MultiFab::Subtract(transverse, field, 0, 0,
                                          transverse.nComp(), 0);
            }
        }
        // E_L changes can inject a vacuum gradient through the total-E
        // warm start. Select the same gauge used by all Krylov corrections.
        ProjectDarwinVacuumGauge(m_E);
    }
    ablastr::warn_manager::WMRecordWarning("ThetaImplicitHybrid",
        "Darwin segregated longitudinal constraint failed to converge");
    return -8;
}

int
ThetaImplicitHybrid::SolveSegregated (const amrex::Real start_time, const int a_step,
                                      amrex::Real field_reference)
{
    BL_PROFILE("ThetaImplicitHybrid::SolveSegregated()");

    // Previous-iterate pressure scratch for the outer convergence test
    // (re-allocated if a load balance moved the field distribution).
    if (m_qdsmc_Pe_prev.empty()) { m_qdsmc_Pe_prev.resize(m_num_amr_levels); }
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        amrex::MultiFab const & Pe =
            *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        if (!m_qdsmc_Pe_prev[lev]
            || m_qdsmc_Pe_prev[lev]->boxArray() != Pe.boxArray()
            || m_qdsmc_Pe_prev[lev]->DistributionMap() != Pe.DistributionMap()) {
            m_qdsmc_Pe_prev[lev] = std::make_unique<amrex::MultiFab>(
                Pe.boxArray(), Pe.DistributionMap(), Pe.nComp(), amrex::IntVect(0));
        }
    }

    // Alternate {inner nonlinear solve at frozen Pe, one re-entrant stage
    // pass} until the emitted pressure stops changing between outer
    // iterations. The stage pass runs on the grid state as left by the
    // inner solver's last (non-probe) residual evaluation -- the plasma
    // current at B^{n+theta}, the midpoint particle deposits, and the
    // frozen step-start states are all mutually consistent there, and
    // nothing may be re-pushed or re-deposited (the as-left discipline of
    // CalculateElectronPressureAtStepEnd). Each outer iteration ENDS on the
    // stage pass, so the markers and the electron velocity are consistent
    // with the accepted field state and QDSMCFinishImplicitStep completes
    // the characteristic unchanged; the O(tolerance) pressure-field
    // mismatch of the final pair is the explicit segregation error.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> previous_temperature(m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev)
    {
        auto const& Te = *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
        previous_temperature[lev] =
            std::make_unique<amrex::MultiFab>(Te.boxArray(), Te.DistributionMap(), Te.nComp(), 0);
    }
    WarpXSolverVec residual;
    residual.Define(m_E);
    amrex::Real field_rtol, field_atol;
    int field_maxits;
    m_nlsolver->GetSolverParams(field_rtol, field_atol, field_maxits);
    amrex::ignore_unused(field_maxits);
    auto residual_norm = [&] ()
    {
        ComputeRHS(residual, m_E, start_time, 0, false);
        residual.increment(m_Eold, 1.0_rt);
        residual.increment(m_E, -1.0_rt);
        return residual.norm2();
    };
    auto capture_thermal = [&] ()
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            amrex::MultiFab const & Pe =
                *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
            amrex::MultiFab::Copy(*m_qdsmc_Pe_prev[lev], Pe, 0, 0, Pe.nComp(),
                                  amrex::IntVect(0));
            auto const& Te = *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            amrex::MultiFab::Copy(*previous_temperature[lev], Te, 0, 0, Te.nComp(), 0);
        }
    };
    auto thermal_defect = [&] ()
    {
        amrex::Real dPe_rel = 0.0_rt;
        amrex::Real dTe_rel = 0.0_rt;
        amrex::Real drho_rel = 0.0_rt;
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            amrex::MultiFab const & Pe =
                *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
            amrex::MultiFab & dPe = *m_qdsmc_Pe_prev[lev];
            amrex::MultiFab::Subtract(dPe, Pe, 0, 0, Pe.nComp(), amrex::IntVect(0));
            amrex::Real const norm_Pe = Pe.norm2(0);
            amrex::Real const norm_dPe = dPe.norm2(0);
            dPe_rel = std::max(dPe_rel,
                norm_dPe / std::max(norm_Pe, std::numeric_limits<amrex::Real>::min()));
            auto const& Te = *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            auto& diff = *previous_temperature[lev];
            for (amrex::MFIter mfi(diff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto const delta = diff.array(mfi);
                auto const now = Te.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       amrex::Real const old = delta(i, j, k);
                                       delta(i, j, k) = std::abs(now(i, j, k) - old) /
                                                        amrex::max(std::abs(now(i, j, k)),
                                                                   std::abs(old), 1.0_rt);
                                   });
            }
            dTe_rel = std::max(dTe_rel, diff.norminf());
            auto const& rho = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            auto const& frozen = *m_qdsmc_rho_frozen[lev];
            int const midpoint = rho.nComp() / 2;
            amrex::Real const floor = PhysConst::q_e * m_hybrid_pic_model->m_n_floor;
            for (amrex::MFIter mfi(diff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto const delta = diff.array(mfi);
                auto const now = rho.const_array(mfi);
                auto const old = frozen.const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       delta(i, j, k) =
                                           std::abs(now(i, j, k, midpoint) - old(i, j, k)) /
                                           amrex::max(std::abs(now(i, j, k, midpoint)),
                                                      std::abs(old(i, j, k)), floor, 1.e-100_rt);
                                   });
            }
            drho_rel = std::max(drho_rel, diff.norminf());
        }
        return std::array<amrex::Real, 3>{dPe_rel, dTe_rel, drho_rel};
    };
    auto refresh_density = [&] ()
    {
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            auto const& rho = *m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            auto& frozen = *m_qdsmc_rho_frozen[lev];
            amrex::MultiFab::Copy(frozen, rho, rho.nComp() / 2, 0, 1, frozen.nGrowVect());
        }
    };
    auto current_defect = [&] () {
        amrex::Real defect = 0.0_rt;
        amrex::Real reference = 1.e-30_rt;
        if (m_hybrid_pic_model->m_visc_in_ohms_law) {
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                auto const live =
                    m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
                auto const plasma = m_WarpX->m_fields.get_alldirs(
                    FieldType::hybrid_current_fp_plasma, lev);
                for (int d = 0; d < 3; ++d) {
                    auto const& frozen =
                        *m_hybrid_pic_model->m_implicit_visc_current.at(lev)[d];
                    amrex::MultiFab delta(frozen.boxArray(),
                                          frozen.DistributionMap(), 1, 0);
                    amrex::MultiFab::LinComb(delta, 1.0_rt, *live[d], 0,
                                             -1.0_rt, frozen, 0, 0, 1, 0);
                    defect = std::max(defect, delta.norminf(0));
                    // Stress depends on J_i-J, so both currents set its scale.
                    // Normalizing by J_i alone cannot converge when ions are
                    // at rest and only floating-point push noise remains.
                    reference =
                        std::max({reference, live[d]->norminf(0),
                                  frozen.norminf(0), plasma[d]->norminf(0)});
                }
            }
        }
        return defect / reference;
    };
    ThermalFixedPointAcceleration thermal_inverse;
    int exit_status = 3;
    amrex::Real dPe_rel = std::numeric_limits<amrex::Real>::max();
    int outer = 0;
    for (; outer < m_qdsmc_outer_max_iterations; ++outer)
    {

        if (m_qdsmc_outer_acceleration) {
            thermal_inverse.Capture(*m_WarpX->m_fields.get(
                FieldType::hybrid_electron_temperature_fp,0));
        }
        m_hybrid_pic_model->FreezeImplicitViscousCurrent();

        // Inner solve: E^{n+theta} at frozen electron pressure. Warm-started
        // from the previous outer iterate (m_E carries through).
        amrex::Real const initial_norm = residual_norm();
        if (!std::isfinite(initial_norm))
        {
            return -7;
        }
        if (field_reference == 0.0_rt)
        {
            field_reference = initial_norm;
        }
        amrex::Real const field_target = std::max(field_atol, field_rtol * field_reference);
        if (!(initial_norm == 0.0_rt || initial_norm < field_target))
        {
            m_nlsolver->SetConvergenceReferenceNorm(field_reference);
            m_nlsolver->Solve(m_E, m_Eold, start_time, m_dt, a_step);
            m_nlsolver->SetConvergenceReferenceNorm(0.0_rt);
            exit_status = m_nlsolver->GetExitStatus();
            if (exit_status < 0)
            {
                return exit_status;
            }
            if (exit_status != 2 && exit_status != 3)
            {
                return -7;
            }
        }

        capture_thermal();

        // One stage pass against the converged fields (re-entrant: restarts
        // from the saved t^n state and re-solves the midpoint entropy
        // transport; runs non-probe by construction, so the per-species
        // source deposits refresh once per outer iteration).
        m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, true);

        auto const defect = thermal_defect();
        refresh_density();
        dPe_rel = defect[0];
        amrex::Real const dTe_rel = defect[1];

        amrex::Real const dJi_rel = current_defect();
        if (m_qdsmc_outer_verbose)
        {
            amrex::Print() << "QDSMC segregated: outer iteration = " << outer
                           << ", dPe/Pe = " << std::scientific << dPe_rel
                           << ", max relative dTe = " << dTe_rel
                           << ", max relative drho = " << defect[2]
                           << ", relative dJi (viscous) = " << dJi_rel << "\n";
        }
        dPe_rel = std::max({dPe_rel, dTe_rel, defect[2], dJi_rel});
        if (dPe_rel < m_qdsmc_outer_relative_tolerance)
        {
            // Grade the field equation with the newly emitted thermal stage.
            // This residual skips the energy advance; reconstruct it once
            // afterwards so the finisher owns markers from these deposits.
            capture_thermal();
            amrex::Real const final_norm = residual_norm();
            if (!std::isfinite(final_norm))
            {
                return -7;
            }
            m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, true);
            auto const final_defect = thermal_defect();
            dPe_rel = std::max({final_defect[0], final_defect[1],
                                final_defect[2], current_defect()});
            refresh_density();
            if ((final_norm == 0.0_rt || final_norm < field_target) &&
                dPe_rel < m_qdsmc_outer_relative_tolerance)
            {
                break;
            }
        }
        if (m_qdsmc_outer_acceleration && outer+1 < m_qdsmc_outer_max_iterations) {
            auto const omega = thermal_inverse.Apply(
                *m_WarpX->m_fields.get(FieldType::hybrid_electron_temperature_fp,0),
                *m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp,0));
            if (m_qdsmc_outer_verbose) {
                amrex::Print() << "QDSMC thermal secant omega=" << omega << "\n";
            }
        }
    }

    if (outer == m_qdsmc_outer_max_iterations || dPe_rel >= m_qdsmc_outer_relative_tolerance)
    {
        std::stringstream convergenceMsg;
        convergenceMsg << "QDSMC segregated outer loop failed to converge after "
                       << outer << " iterations. Relative pressure change is "
                       << dPe_rel << " and the outer relative tolerance is "
                       << m_qdsmc_outer_relative_tolerance;
        ablastr::warn_manager::WMRecordWarning(
            "ThetaImplicitHybrid", convergenceMsg.str());
        if (m_qdsmc_outer_require_convergence || m_darwin_segregated_solve)
        {
            return -7;
        }
    }

    return exit_status;
}

void ThetaImplicitHybrid::ComputeRHS ( WarpXSolverVec&        a_RHS,
                                       const WarpXSolverVec&  a_E,
                                       amrex::Real            start_time,
                                       int                    a_nl_iter,
                                       bool                   a_from_jacobian )
{
    BL_PROFILE("ThetaImplicitHybrid::ComputeRHS()");

    std::unique_ptr<warpx::thermal::DarwinThermalAdvance::CurrentRemainderMap> thermal_remainder_map;
    if(m_eulerian_energy && m_eulerian_energy->HasResidualRemainder())
        thermal_remainder_map=std::make_unique<warpx::thermal::DarwinThermalAdvance::CurrentRemainderMap>(*m_eulerian_energy);
    // The circuit and field stage must agree before particles gather B/E.
    // Replaying only after the push leaves both the moments and Jp stale.
    if (m_darwin_circuit_consistent_stage) {
        ConvergeDarwinCircuitStage(a_E, start_time, a_from_jacobian);
    } else {
        UpdateWarpXFields(a_E, a_from_jacobian, start_time);
    }

    // Split-field circuit-in-the-residual coupling, run BEFORE the
    // particle stage: python measures the flux linkage of THIS iterate's
    // plasma response (disk probes read the theta-stage response B just
    // assembled above), re-advances the coupled external circuit against
    // it, and pushes updated coil scale segments (SetScale); the external
    // fields are then refreshed at the new scales so the particle push
    // AND the Ohm's law below see circuit-consistent fields. Running the
    // coupling after the push (as the darwin branch below does for its
    // boundary pin) would make the gathered E_ext lag the iterate by one
    // evaluation -- hidden history that shows up as an
    // epsilon-independent noise floor in Jacobian secants once the drive
    // is active. Reciprocity (J-based) probes see THIS evaluation's
    // plasma current (refreshed in UpdateWarpXFields from the response
    // field); the circuit ports in use are disk-flux based.
    if (m_external_field_iteration && !m_darwin) {
        AddSplitExternalFields(-1.0_rt);
        ExecutePythonCallback("externalcoiltheta");
        m_hybrid_pic_model->m_external_vector_potential
            ->UpdateHybridExternalFields(start_time + m_theta * m_dt, m_dt);
        AddSplitExternalFields(1.0_rt);
    }

    const amrex::Real theta_time = start_time + m_theta * m_dt;
    auto push_and_deposit = [&] () {
    // Momentum-consistent particle push field: the ions gather Efield_fp,
    // and the solver state deliberately includes the resistive eta*J term
    // (Faraday's law needs it), but the resistive friction must not
    // accelerate the ions through E -- the explicit scheme pushes ions with
    // the no-resistivity Ohm field, and the resistive electron-ion friction
    // is a separate (optional) collision operator. B above already used the
    // full E.
    //
    // The correction is assembled HERE, in every residual evaluation, from
    // THIS iterate: E_push = E_iterate - (E_full - E_nores), with both Ohm
    // solves running INTO Efield_fp so the solve wrapper's boundary stack
    // (PEC image, conformal-EB conductor edges, resistive-shell wall rows)
    // lands identically on both passes -- every term except eta*J (and the
    // hyper-resistive term) then cancels exactly in the difference,
    // including all boundary-controlled rows. Solving the no-resistivity
    // pass into a scratch field instead would leave those rows untreated
    // (the wrapper applies the field boundary to the registered Efield_fp,
    // not to the passed output) and the difference would carry O(1)
    // boundary-row garbage at any eta. A correction lagged from the
    // previous residual evaluation is hidden state that shifts the residual
    // between the base and probe evaluations of the difference Jacobian and
    // stalls Newton at an O(1) relative norm. The pre-push moments feeding
    // the two passes enter the difference only through the eta(rho, |J|)
    // and eta_h(rho, |B|) parametrizations (|J| is the Ampere current, a
    // pure function of the iterate); constant coefficients make the
    // correction a pure function of the iterate.
    if (m_use_resistive_push_correction) {
        using ablastr::fields::Direction;

        // The Darwin branch computes the plasma current after the particle
        // stage; the correction needs it at the iterate's B (already
        // rebuilt above). The later recomputation sees the same B.
        if (m_darwin) {
            if (!warpx::darwin::TryCalculateNativeSplitDarwinAmpere(*m_WarpX)) {
                CalculateNativePlasmaCurrent(
                    m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
                    theta_time);
            }
            ApplyDarwinDisplacementCurrent(m_theta * m_dt);
        }

        ablastr::fields::MultiLevelVectorField E_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelVectorField J_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelVectorField B_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelScalarField rho_pre =
            m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);

        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            // Park the assembled push field (iterate + BCs + E_ext).
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_res = *m_WarpX->m_fields.get(
                    "hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab const& E = *E_fp[lev][dir];
                amrex::MultiFab::Copy(E_res, E, 0, 0, E.nComp(), E.nGrowVect());
            }
            // Per-level solves: no callback fires inside residual
            // evaluations (see the RHS Ohm solve below).
            m_hybrid_pic_model->HybridPICSolveE(E_fp[lev], J_fp[lev], B_fp[lev],
                                                m_eulerian_energy ? m_eulerian_energy->PushCharge() :
                                                (m_qdsmc_segregated_solve ? *m_qdsmc_rho_frozen[lev]
                                                                         : *rho_pre[lev]),
                                                m_WarpX->GetEBUpdateEFlag()[lev], lev,
                                                false, // solve_for_Faraday (retain grad(Pe))
                                                true,  // include_resistivity
                                                m_eulerian_energy ? m_eulerian_energy->Dissipation() : nullptr
            );
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_res = *m_WarpX->m_fields.get(
                    "hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab const& E = *E_fp[lev][dir];
                amrex::MultiFab::Subtract(E_res, E, 0, 0, E.nComp(), E.nGrowVect());
            }
            m_hybrid_pic_model->HybridPICSolveE(
                E_fp[lev], J_fp[lev], B_fp[lev],
                m_eulerian_energy ? m_eulerian_energy->PushCharge() :
                (m_qdsmc_segregated_solve ? *m_qdsmc_rho_frozen[lev] : *rho_pre[lev]),
                m_WarpX->GetEBUpdateEFlag()[lev], lev,
                false, // solve_for_Faraday (retain grad(Pe))
                false, // include_resistivity: no-resistivity push field
                m_eulerian_energy ? m_eulerian_energy->Dissipation() : nullptr,
                m_eulerian_energy ? m_eulerian_energy->IonElectricWorkOutput() : nullptr
            );
            if (m_eulerian_energy) { m_eulerian_energy->CaptureIonElectricWorkBase(E_fp[lev]); }
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E_push = *E_fp[lev][dir];
                amrex::MultiFab const& E_res = *m_WarpX->m_fields.get(
                    "hybrid_E_resistive_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(E_push, E_res, 0, 0, E_push.nComp(), E_push.nGrowVect());
            }
        }
        // The Ohm kernels write valid cells only, so box-boundary ghosts
        // still hold the uncorrected iterate; the gather reads those ghosts.
        amrex::IntVect const ngE = E_fp[0][0]->nGrowVect();
        m_WarpX->FillBoundaryE(ngE, true /* sync nodal points */);
    }

    if (m_darwin && m_hybrid_pic_model->m_darwin_poisson_verbosity > 1) {
        using ablastr::fields::Direction;
        auto ni = [&](const char* nm, int dir) {
            return m_WarpX->m_fields.get(nm, Direction{dir}, 0)->norminf();
        };
        amrex::Print() << "[darwin-eval] iter " << a_nl_iter
            << (a_from_jacobian ? " (jac)" : "")
            << " max|E_push| = " << m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->norminf()
            << "/" << m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, 0)->norminf()
            << " max|B| = " << m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, 0)->norminf()
            << " max|A| = " << ni("hybrid_A_fp", 0) << "/" << ni("hybrid_A_fp", 2)
            << "\n";
    }

    // Capture the same projected/centered force used by the native push; no
    // reduction or accepted bookkeeping is performed in residuals or Jv.
    if (m_eulerian_energy) {
        if (!m_use_resistive_push_correction) {
            // With no nores correction, use the native push field itself as
            // base: its applied remainder is only synchronization roundoff.
            m_eulerian_energy->CaptureIonElectricWorkBase(
                m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp,0));
        }
        m_eulerian_energy->PrepareIonElectricWorkField();
    }
    // Advance particles and deposit J^{n+1/2}, rho^{n+1/2}
    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );
    if (m_eulerian_energy && (!a_from_jacobian || !m_use_mass_matrices_jacobian)) {
        m_eulerian_energy->CaptureIonElectricWorkTotal();
    }
    if (m_eulerian_energy && (!a_from_jacobian || !m_use_mass_matrices_jacobian) &&
        !m_eulerian_energy->AuditParticleEndpoints()) { return false; }
    UpdateMassMatrixDensity(a_from_jacobian);
    return true;

    };
    if (m_eulerian_energy && m_use_resistive_push_correction) {
        m_eulerian_energy->InitializePushContext();
        bool consistent = false;
        for (int inner=0;inner<m_eulerian_energy->PushIterations();++inner) {
            // Rebuild the uncorrected field before every elimination sweep.
            // The old particle coordinates are fixed by the implicit pusher.
            if (m_darwin_circuit_consistent_stage) {
                ConvergeDarwinCircuitStage(a_E,start_time,a_from_jacobian);
            } else { UpdateWarpXFields(a_E,a_from_jacobian,start_time); }
            if (!m_eulerian_energy->SetPushThermodynamics()) { break; }
            if (!push_and_deposit()) { break; }
            if (m_eulerian_energy->PushContextConverged()) { consistent=true; break; }
        }
        if (!consistent) {
            a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
            return;
        }
    } else if (!push_and_deposit()) {
        a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        return;
    }
    if (m_eulerian_energy && !m_eulerian_energy->RefreshSpeciesContext()) {
        a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
        return;
    }

    // Get field arrays at all levels
    ablastr::fields::MultiLevelVectorField Efield_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField Bfield_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField current_fp =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelScalarField rho_fp =
        m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);

    // The split residual must consume THIS evaluation's midpoint density.
    // Component zero is deposited before the particle push and, after the
    // first evaluation, contains the previous evaluation's midpoint state.
    // Ohm's law and the algebraic pressure closure read component zero of
    // their argument, so expose the new midpoint component through an alias.
    // Leave the registered two-time-level density intact for inertia/history.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_half_store(m_num_amr_levels);
    if (m_darwin_segregated_solve || m_live_ion_density) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            auto& rho = *rho_fp[lev];
            rho_half_store[lev] = std::make_unique<amrex::MultiFab>(
                rho, amrex::make_alias, rho.nComp()/2, 1);
            rho_fp[lev] = rho_half_store[lev].get();
        }
    }

    // Compute J_plasma = curl(B^{n+theta})/mu_0. The split-field (non-
    // darwin) branch computed it in UpdateWarpXFields from the plasma-
    // response field, BEFORE the external assembly -- recomputing it here
    // from the total field would re-introduce the spurious O(h^2) external
    // curl. The darwin branch computes it here from its derived
    // B = B_static + curl A, whose curl is discretely consistent (the
    // external flux enters through the evolved A, not a sampled field).
    if(m_darwin&&m_joint_stage_owner&&m_joint_stage_owner->CurrentArithmeticActive())
        m_joint_stage_owner->PublishArithmeticCurrent();
    else if (m_darwin && !warpx::darwin::TryCalculateNativeSplitDarwinAmpere(*m_WarpX)) {
        CalculateNativePlasmaCurrent(Bfield_fp,theta_time);
    }

    if (m_darwin)
    {
        ApplyDarwinDisplacementCurrent(m_theta * m_dt);
    }

    // Circuit-in-the-residual coupling, darwin (unified-drive) branch:
    // python measures the flux linkage of THIS iterate's plasma response,
    // re-advances the coupled external circuit against it, and pushes
    // updated coil scale segments (SetScale). The scales enter through
    // the boundary pin and the vacuum band (interior A is state-only, so
    // the particle stage above needs no re-run). Re-impose the pin at
    // the updated scales and re-derive B. The split-field branch of this
    // coupling runs BEFORE the particle stage (top of this function): its
    // scales enter the gathered push field, which must not lag the
    // iterate.
    if (m_external_field_iteration && m_darwin && !m_darwin_circuit_consistent_stage) {
        if (m_circuit_native) {
            auto& coupler = NativeCircuitCoupler();
            coupler.MeasureDarwinLinkages(theta_time);
            coupler.EvaluateInterval(start_time, start_time + m_dt, false, m_theta*m_dt);
        } else {
            ExecutePythonCallback("externalcoiltheta");
        }
        DarwinApplyABoundary(theta_time);
        if (m_vacuum_recovery_half) {
            // Recompute the recovery against the re-imposed boundary
            // values (live in Jacobian probes too -- Define forced the
            // live-probe mode), then restore the exact pin. NOTE: this
            // is the second recovery application of the evaluation, so
            // a finite darwin_vacuum_recovery_relaxation_time would be
            // double-applied -- circuit decks should run the default
            // (instant) recovery.
            m_hybrid_pic_model->ComputeVacuumARecovery(
                a_from_jacobian, m_theta * m_dt);
            DarwinApplyABoundary(theta_time);
        }
        DarwinDeriveB();
    }

    // The eliminated PEC momentum row depends on this trial pressure.
    // The original branch retains its existing ordering unchanged.
    if(warpx::darwin::NativePECPlasmaEnabled() && m_eulerian_energy &&
       !m_eulerian_energy->RefreshPressure()) {
        a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());return;
    }

    // Electron inertia: assemble the nodal inertial field from the
    // theta-stage state, ahead of both the E_L constraint solve (which
    // takes its longitudinal part into the source) and the Ohm E-solve
    // (which adds it per component). Refreshed in every evaluation
    // including Jacobian probes -- a smooth function of the state; the Je
    // histories are frozen per step.
    if (m_hybrid_pic_model->m_include_electron_inertia) {
        amrex::MultiFab const* inertia_stage=nullptr;
        amrex::MultiFab const* inertia_rate=nullptr;
        if (NativeIonQuadratureEnabled() && a_from_jacobian && m_use_mass_matrices_jacobian &&
            (!m_endpoint_current_response || !m_endpoint_current_response->IsValid())) {
            a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
            return;
        }
        if (NativeIonQuadratureEnabled() &&
            !PrepareNativeInertiaStage(*m_WarpX,m_dt,inertia_stage,inertia_rate,
                a_from_jacobian && m_use_mass_matrices_jacobian
                    ? m_endpoint_current_response.get() : nullptr,NativeCoilCurrentContext(),theta_time)) {
            a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
            return;
        }
        m_hybrid_pic_model->ComputeElectronInertiaNodal(m_theta, m_dt,
                                                        a_from_jacobian,inertia_stage,inertia_rate);
        PublishNativeYeeInertiaMirror(*m_WarpX);
    }

    // Electron pressure at t^{n+theta}: either the theta-centered QDSMC
    // electron-energy stage (re-entrant; re-run from the saved t^n state in
    // every residual evaluation, so the nonlinear solver converges the
    // coupled E/T_e system by elimination) or the algebraic adiabatic
    // closure. Per-species deposits feeding the multi-species sources are
    // refreshed once per Newton iteration and frozen during Jacobian
    // evaluations. Under the segregated solve the stage runs once per OUTER
    // iteration from SolveSegregated instead, and the residual consumes the
    // held temperature. With MM density projection, pressure follows the
    // live density at that temperature; otherwise it remains outer-frozen.
    if (m_eulerian_energy) {
        if (!m_eulerian_energy->RefreshPressure()) {
            a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
            return;
        }
    } else if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        if (!m_qdsmc_segregated_solve) {
            m_hybrid_pic_model->AdvanceElectronEnergyQDSMCTheta(m_dt, m_theta, !a_from_jacobian);
        } else if (m_live_ion_density) {
            // Keep the expensive thermal marker stage outside GMRES, while
            // Pe = ne*kB*Te follows THIS density at the held temperature.
            // Nonlinear evaluations use their true particle deposit; the
            // anchored continuity predictor agrees at zero perturbation.
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                if (m_theta == 0.5_rt) {
                    m_hybrid_pic_model->QDSMCFillElectronPressureFromTe(lev, *rho_fp[lev]);
                } else {
                    m_hybrid_pic_model->QDSMCFillElectronPressureTheta(lev, m_theta);
                }
            }
        }
    } else {
        if (m_darwin_segregated_solve || m_live_ion_density) {
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                m_hybrid_pic_model->CalculateElectronPressure(lev, *rho_fp[lev]);
            }
        } else {
            m_hybrid_pic_model->CalculateElectronPressure();
        }
    }

    // Darwin: refresh the longitudinal constraint field from this
    // evaluation's electron pressure and midpoint density. The refresh runs
    // in EVERY residual evaluation (including finite-difference Jacobian
    // probes): E_L is a smooth function of the state, so folding it into
    // the probed residual keeps the FD Jacobian consistent with the actual
    // iterate-to-iterate map -- freezing it (as is done for the noisy
    // per-species particle deposits) makes Newton chase a moving target
    // and stall near 50% residuals. The particle push of this evaluation
    // used the previous evaluation's E_L; both agree at convergence. The
    // opt-in segregated solve instead updates E_L only between complete
    // nonlinear solves, so no residual or Jv consumes hidden E_L history.
    if (m_darwin && !m_darwin_segregated_solve) {
        RefreshDarwinELong(theta_time);
    }

    // Optional same-equation arithmetic: keep the small transverse residual
    // from losing Ei when the pressure and held longitudinal field cancel.
    bool stable_transverse_ohm=false;
    amrex::ParmParse("endpoint_diagnostic").query("stable_transverse_ohm",stable_transverse_ohm);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!stable_transverse_ohm || m_darwin,
        "Stable transverse Ohm grouping requires Darwin field variables");

    // Solve Ohm's law: E_ohm = f(B^{n+theta}, J_ion^{n+1/2}, rho^{n+1/2}, Pe)
    // Result stored in Efield_fp.
    //
    // The converged E^{n+theta} serves BOTH roles in the implicit scheme: it
    // drives Faraday's law (so the resistive terms must be included or the
    // magnetic field never decays resistively) and it is gathered by the
    // particles (so grad(Pe) must be included for the pressure coupling).
    // The explicit scheme separates these into two E-solves gated by
    // solve_for_Faraday; here the full Ohm E is assembled by overriding the
    // resistive gate, and the resistive part is subtracted again from the
    // push field at the top of this function.
    //
    // Use the per-level HybridPICSolveE: this runs inside every nonlinear
    // residual evaluation (including Jacobian probes), so the multi-level
    // variant's afterEpush python callback must not fire here. A single
    // afterEpush is fired at the converged state after
    // ImplicitSolver::OneStep completes (see WarpX::OneStep).
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        auto const held_EL=m_darwin
            ? m_WarpX->m_fields.get_alldirs("hybrid_E_long_fp",lev)
            : ablastr::fields::VectorField{};
        m_hybrid_pic_model->HybridPICSolveE(
            Efield_fp[lev], current_fp[lev], Bfield_fp[lev],
            m_qdsmc_segregated_solve && !m_live_ion_density
                ? *m_qdsmc_rho_frozen[lev] : *rho_fp[lev],
            m_WarpX->GetEBUpdateEFlag()[lev], lev, PatchType::fine,
            false, // solve_for_Faraday (retain grad(Pe))
            true,  // include_resistivity (retain eta*J for the B-update)
            m_eulerian_energy ? m_eulerian_energy->Dissipation() : nullptr,
            nullptr, nullptr, nullptr, stable_transverse_ohm ? &held_EL : nullptr
        );
    }

    auto* paired_rhs=warpx::darwin::NativeActivePairedFields(*m_WarpX);
    if(paired_rhs)paired_rhs->FormTransverseOhmRHS(stable_transverse_ohm);
    if(stable_transverse_ohm && warpx::darwin::increment::Enabled())
        for(int c=0;c<3;++c)warpx::darwin::increment::SubtractLow(*Efield_fp[0][c],warpx::darwin::increment::Low(*m_WarpX,c));

    // The Ohm kernels above returned the stored (plasma) field convention:
    // E_ohm computed from the total B, with the inductive E_ext subtracted
    // in plasma cells (inside the plasma the generalized Ohm's law IS the
    // electric field; the external drive reaches it through B). The push
    // field assembly in UpdateWarpXFields re-adds E_ext on top.

    // Darwin: the state is the transverse field, so the fixed point is
    // E_T = E_ohm - E_L. Subtracting E_L from the assembled Ohm field
    // groups the (nearly cancelling) pressure and longitudinal-constraint
    // terms into one small object, and the converged full field
    // E = E_T + E_L satisfies the complete generalized Ohm's law
    // independently of the Helmholtz-projection quality of E_L.
    if (m_darwin) {
        using ablastr::fields::Direction;
        if (!stable_transverse_ohm && !paired_rhs) for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab::Subtract(E, EL, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        // In the vacuum band the field is defined by the recovered vector
        // potential, E_T = -(A_rec - A^n)/(theta dt), not by the (invalid
        // there) generalized Ohm's law. The FD-Jacobian must not
        // difference through the iterative recovery solve, whose
        // rtol-level noise enters these rows amplified by 1/(theta dt):
        // probe evaluations reuse the correction and the stored target
        // (identity Jacobian rows), and the outer Newton iteration lags
        // the recovery (contraction at the recovery's small leak factor).
        if (m_vacuum_recovery_half && m_joint_stage_purpose==JointStagePurpose::Standard) {
            m_hybrid_pic_model->ApplyVacuumFaradayE(
                m_theta * m_dt, false, a_from_jacobian, false);
        }
    }

    if(m_joint_stage_owner&&m_joint_stage_owner->CurrentArithmeticActive())
        m_joint_stage_owner->CaptureArithmeticOhm();
    // Return RHS = E_ohm - E_old
    // Framework computes residual = E - E_old - RHS = E - E_ohm
    // Convergence: E = E_ohm
    if (std::getenv("WARPX_DEBUG_RESID") != nullptr) {
        using ablastr::fields::Direction;
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab diff(m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, 0)->boxArray(),
                                 m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, 0)->DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(diff, *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, 0), 0, 0, 1, 0);
            amrex::MultiFab::Subtract(diff, *a_E.getArrayVec()[0][dir], 0, 0, 1, 0);
            auto imax = diff.maxIndex(0);
            amrex::AllPrintToFile("resid_probe") << "iter " << a_nl_iter
                << " dir " << dir << " |dE|max " << diff.norminf(0)
                << " at " << imax << "\n";
        }
    }
    a_RHS.Copy(FieldType::Efield_fp);         // a_RHS = E_ohm
    a_RHS.linComb(1.0, a_RHS, -1.0, m_Eold);  // a_RHS = E_ohm - E_old
    if(thermal_remainder_map && !thermal_remainder_map->Finish())
        a_RHS.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
}

CircuitCoupler& ThetaImplicitHybrid::NativeCircuitCoupler () const
{
    auto* coupling = m_WarpX->get_pointer_CircuitCoupling();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(coupling != nullptr && coupling->Coupler() != nullptr
        && coupling->Coupler()->Plugin() != nullptr,
        "Native Darwin circuit driver requires circuit.coils, engine=external and plugin_library");
    return *coupling->Coupler();
}

void ThetaImplicitHybrid::CommitNativeCircuitStage (amrex::Real start_time)
{
    BL_PROFILE("ThetaImplicitHybrid::CommitNativeCircuitStage()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_native_circuit_step_open,
        "Native circuit step is not open");
    if (NativeCircuitCoupler().DeviceTrials()) {
        NativeCircuitCoupler().CommitDarwinDeviceStep(start_time,m_dt,m_theta,
                                                     m_darwin_circuit_scale_tolerance);
        m_native_circuit_step_open = false;
        return;
    }
    auto& external = *m_hybrid_pic_model->m_external_vector_potential;
    auto const time = start_time + m_theta*m_dt;
    amrex::Vector<amrex::Real> candidate(external.nFields());
    for (int field = 0; field < external.nFields(); ++field) {
        candidate[field] = external.TimeScale(field, time);
    }
    auto& coupler = NativeCircuitCoupler();
    coupler.MeasureDarwinLinkages(time);
    // The same stage EMF drives a full-step candidate and its exact acceptance.
    // Do not reinterpret the theta linkage as an endpoint measurement.
    coupler.EvaluateInterval(start_time, start_time + m_dt, true, m_theta*m_dt);
    for (int field = 0; field < external.nFields(); ++field) {
        auto const accepted = external.TimeScale(field, time);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(accepted)
            && std::abs(accepted - candidate[field]) <= m_darwin_circuit_scale_tolerance
                *std::max({1._rt, std::abs(accepted), std::abs(candidate[field])}),
            "Exact circuit acceptance changed the converged stage; rejecting inconsistent commit");
    }
    coupler.FinishStep();
    m_native_circuit_step_open = false;
}

void ThetaImplicitHybrid::RefreshDarwinCircuitCurrent ()
{
    BL_PROFILE("ThetaImplicitHybrid::RefreshDarwinCircuitCurrent()");
    AMREX_ALWAYS_ASSERT(!warpx::darwin::NativeCoilCurrentEnabled());
    using ablastr::fields::Direction;
    m_hybrid_pic_model->CalculatePlasmaCurrent(
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
        m_WarpX->GetEBUpdateEFlag());
    amrex::Real const factor = PhysConst::epsilon_0 / (m_theta * m_dt);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0; dir < 3; ++dir) {
            auto& current = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{dir}, lev);
            auto const& longitudinal = *m_WarpX->m_fields.get(
                "hybrid_E_long_fp", Direction{dir}, lev);
            auto const& old = *m_WarpX->m_fields.get(
                "hybrid_E_long_old_fp", Direction{dir}, lev);
            warpx::darwin::SubtractDisplacementCurrent(
                current, longitudinal, old, factor,warpx::darwin::increment::Low(*m_WarpX,dir));
        }
    }
}

void ThetaImplicitHybrid::ConvergeDarwinCircuitStage (
    WarpXSolverVec const& electric_field, amrex::Real start_time, bool from_jacobian)
{
    BL_PROFILE("ThetaImplicitHybrid::ConvergeDarwinCircuitStage()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_circuit_native
        || IsPythonCallbackInstalled("externalcoiltheta"),
        "Darwin circuit stage requires its native plugin or compatibility callback");
    auto& external = *m_hybrid_pic_model->m_external_vector_potential;
    amrex::Real const stage_time = start_time + m_theta * m_dt;
    if (m_circuit_native && NativeCircuitCoupler().DeviceTrials()) {
        auto& coupler = NativeCircuitCoupler();
        coupler.ResetDarwinDeviceTrial();
        // Fixed launch count: convergence and map validity are checked on
        // device, with no circuit vector/status fetch inside the residual.
        for (int iteration = 0; iteration < coupler.DeviceIterations(); ++iteration) {
            UpdateWarpXFields(electric_field,from_jacobian,start_time);
            coupler.AdvanceDarwinDeviceTrial();
        }
        coupler.RequireDarwinDeviceConvergence();
        UpdateWarpXFields(electric_field,from_jacobian,start_time);
        return;
    }

    auto const& accepted = m_darwin_circuit_accepted_scales;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(accepted.size() == external.nFields(),
        "Circuit scales must be captured at the start of the step");
    // A warm start from the preceding residual makes finite fixed-point error
    // depend on probe order. Rebuild each native trial from accepted scales.
    if (m_circuit_native) { NativeCircuitCoupler().ResetDarwinTrialScales(start_time, m_dt); }
    amrex::Vector<amrex::Real> previous(external.nFields());
    for (int iteration = 0; iteration < m_darwin_circuit_max_iterations; ++iteration) {
        // Always rebuild from the same trial E and accepted A, including Jv probes.
        // No particles or circuit accepted-state commits occur inside this loop.
        UpdateWarpXFields(electric_field, from_jacobian, start_time);
        if (!m_circuit_native) { RefreshDarwinCircuitCurrent(); }
        for (int coil = 0; coil < external.nFields(); ++coil) {
            previous[coil] = external.TimeScale(coil, stage_time);
        }
        if (m_circuit_native) {
            auto& coupler = NativeCircuitCoupler();
            coupler.MeasureDarwinLinkages(stage_time);
            coupler.EvaluateInterval(start_time, start_time + m_dt, false, m_theta*m_dt);
        } else {
            ExecutePythonCallback("externalcoiltheta");
        }
        amrex::Real defect = 0._rt;
        for (int coil = 0; coil < external.nFields(); ++coil) {
            amrex::Real const current = external.TimeScale(coil, stage_time);
            amrex::Real const start = external.TimeScale(coil, start_time);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(current) && std::isfinite(start)
                && std::isfinite(previous[coil]), "Nonfinite trial circuit scale");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(start - accepted[coil])
                <= m_darwin_circuit_scale_tolerance * std::max(1._rt, std::abs(accepted[coil])),
                "externalcoiltheta changed the accepted start-of-step coil scale");
            defect = std::max(defect, std::abs(current - previous[coil])
                / std::max({1._rt, std::abs(current), std::abs(previous[coil])}));
        }
        if (defect <= m_darwin_circuit_scale_tolerance) {
            // Use the final published segment, not the preceding fixed-point iterate.
            UpdateWarpXFields(electric_field, from_jacobian, start_time);
            if (!m_circuit_native) { RefreshDarwinCircuitCurrent(); }
            return;
        }
    }
    WARPX_ABORT_WITH_MESSAGE("Darwin circuit stage failed to converge its coil scales");
}

void ThetaImplicitHybrid::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                                bool a_from_jacobian,
                                                amrex::Real start_time )
{
    BL_PROFILE("ThetaImplicitHybrid::UpdateWarpXFields()");
    if(m_eulerian_energy && m_eulerian_energy->HasResidualRemainder())
        m_eulerian_energy->CurrentRemainderFieldsChanged();

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    // Set E^{n+theta} in WarpX (the transverse part E_T on the Darwin path)
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // Assemble the external contributions on top of the plasma fields after
    // the respective plasma-field updates below: B_ext at t^{n+theta} for
    // the Ohm kernels and the particle push, and the step-averaged
    // inductive E_ext for the push field. Deliberately applied AFTER the
    // boundary treatments -- the imposed external field must not be
    // altered by the wall conditions.
    auto add_external = [&](warpx::fields::FieldType ftype,
                            warpx::fields::FieldType ext_type) {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & F = *m_WarpX->m_fields.get(ftype, Direction{dir}, lev);
                amrex::MultiFab const & F_ext = *m_WarpX->m_fields.get(ext_type, Direction{dir}, lev);
                amrex::MultiFab::Add(F, F_ext, 0, 0, F.nComp(), F.nGrowVect());
            }
        }
    };
    const bool has_external = m_hybrid_pic_model->m_add_external_fields;
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{};
    bool any_pmc = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
        pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
        any_pmc = any_pmc || pmc_lo[d] || pmc_hi[d];
    }
    auto apply_pmc_response = [&] ()
    {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev)
        {
            for (int dir = 0; dir < 3; ++dir)
            {
                auto& E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                auto const* reference = has_external
                    ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{dir}, lev)
                    : nullptr;
                ApplyDarwinPMCVectorBoundary(E, m_WarpX->Geom(lev), pmc_lo, pmc_hi, reference);
            }
        }
    };

    if (m_darwin) {
        // Reflect the plasma response about the current imposed drive. The
        // generic field BC above reflects the total field and loses the
        // external axial variation. Refresh from this trial circuit segment
        // so residual/Jacobian evaluations do not inherit another stage's drive.
        if (any_pmc)
        {
            if (has_external)
            {
                m_hybrid_pic_model->m_external_vector_potential
                    ->UpdateHybridExternalFields(theta_time, m_dt);
            }
            apply_pmc_response();
        }
        // Darwin: advance the vector potential with the transverse field and
        // rebuild B = B_static + curl A (Faraday integrated through A, so B
        // stays solenoidal by construction), then assemble the full
        // E = E_T + E_L for the particle push. E_L is the constraint field
        // refreshed once per nonlinear iteration in ComputeRHS.
        using ablastr::fields::Direction;
        DarwinUpdateA_B( m_theta * m_dt, start_time, a_from_jacobian );
        if(warpx::darwin::NativeCoilCurrentEnabled()) {
            for(int c=0;c<3;++c)amrex::MultiFab::Copy(m_native_longitudinal_gather[c],
                *m_WarpX->m_fields.get("hybrid_E_long_fp",Direction{c},0),0,0,1,0);
            warpx::darwin::CompleteNativeElectricGatherImages(*m_WarpX,
                {&m_native_longitudinal_gather[0],&m_native_longitudinal_gather[1],&m_native_longitudinal_gather[2]});
        }
        auto* paired=warpx::darwin::NativeActivePairedFields(*m_WarpX);
        if(paired)paired->AssembleTotalElectric();
        bool const joint_arithmetic=m_joint_stage_owner&&m_joint_stage_owner->CurrentArithmeticActive();
        if(joint_arithmetic)m_joint_stage_owner->AssembleArithmeticElectric();
        if(!paired&&!joint_arithmetic)for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                if(warpx::darwin::NativeCoilCurrentEnabled())
                    amrex::MultiFab::Add(E,m_native_longitudinal_gather[dir],0,0,1,E.nGrowVect());
                else warpx::darwin::increment::AddFull(E,EL,warpx::darwin::increment::Low(*m_WarpX,dir));
            }
        }
        // Re-apply the field boundary treatment to the ASSEMBLED field: the
        // boundary/ghost values applied to the transverse state above do not
        // survive the E_L addition at PEC faces,
        // and wall-adjacent particles gather from those ghost layers. With
        // stale ghosts the wall layer picks up O(E_L) spurious kicks whose
        // density response feeds back through grad(Pe) into E_L -- a
        // divergent per-iteration wall loop in non-periodic directions.
        amrex::IntVect const ngE =
            m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->nGrowVect();
        m_WarpX->FillBoundaryE(ngE, true /* sync nodal points */);
        if(!warpx::darwin::NativeCoilCurrentEnabled()) {
            m_WarpX->ApplyEfieldBoundary(0, PatchType::fine, theta_time);
            apply_pmc_response();
        }
        // The driven branch already has completed -A_rate plus independently
        // completed EL. A second generic PEC pass would erase its drive.
    } else {
        // Compute B^{n+theta} = B^n - theta*dt*curl(E^{n+theta}) via Faraday's law
        ablastr::fields::MultiLevelVectorField const& B_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, m_num_amr_levels - 1);
        m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta * m_dt, start_time );

        // Plasma current from the RESPONSE field, before the external
        // assembly below: J_plasma = curl(B_plasma)/mu0, matching the
        // explicit advance (which computes it from the stripped field).
        // The discrete curl of the stored external field is only O(h^2)
        // zero for a spatially varying coil field, so computing the plasma
        // current from the total field deposits that truncation artifact
        // as a spurious near-boundary current, proportional to the coil
        // scale, whose Hall/resistive Ohm response integrates secularly
        // through Faraday (measured as a linear ride-through drift on the
        // coil-pair vacuum deck; a uniform external A is blind to the
        // defect since its discrete curl vanishes identically).
        m_hybrid_pic_model->CalculatePlasmaCurrent(
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1),
            m_WarpX->GetEBUpdateEFlag());
    }

    if (has_external && !m_darwin) {
        add_external(FieldType::Bfield_fp, FieldType::hybrid_B_fp_external);
        add_external(FieldType::Efield_fp, FieldType::hybrid_E_fp_external);
    }
    if(m_joint_stage_owner)m_joint_stage_owner->CapturePushField();
}

void ThetaImplicitHybrid::DarwinUpdateA_B ( amrex::Real a_thetadt, amrex::Real a_time,
                                            bool a_from_jacobian )
{
    const amrex::Real pin_time = a_time + a_thetadt;
    BL_PROFILE("ThetaImplicitHybrid::DarwinUpdateA_B()");

    if(m_joint_stage_owner)m_joint_stage_owner->CaptureTransverse();
    if(auto* paired=warpx::darwin::NativeActivePairedFields(*m_WarpX)) {
        paired->ConditionTransverse();
        PrepareNativeCurlRate(*m_WarpX,a_time,a_thetadt/m_theta);
        paired->BuildPotential(a_thetadt);
        paired->BuildMagnetic();
        return;
    }
    if(m_joint_stage_owner&&m_joint_stage_owner->CurrentArithmeticActive()) {
        m_joint_stage_owner->BuildArithmeticFields(a_thetadt,DarwinParticleCurlGrow());
        return;
    }
    // Efield_fp is the native boundary-conditioned transverse trial here.
    // Capture its double curl before EL assembly or addition of B_static.
    if(NativeCorrelatedIncrementEnabled()) { PrepareNativeCurlRate(*m_WarpX,a_time,a_thetadt/m_theta,NativeCoilCurrentContext()); }

    using ablastr::fields::Direction;

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        // A^{n+theta} = A_old - theta*dt * E_T (Efield_fp holds E_T here)
        ablastr::fields::VectorField A = m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
        ablastr::fields::VectorField B = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab const & A_old = *m_WarpX->m_fields.get("hybrid_A_old_fp", Direction{dir}, lev);
            amrex::MultiFab const & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
            amrex::MultiFab::LinComb(*A[dir], 1.0_rt, A_old, 0, -a_thetadt, E, 0,
                                     0, A[dir]->nComp(), A[dir]->nGrowVect());
        }
        // Boundary-driven external flux and embedded conductors act on A
        // itself (see DarwinApplyABoundary).
        DarwinApplyABoundary(pin_time);
        // In-residual vacuum recovery: replace A in masked (vacuum) cells
        // with the magnetostatic solution before deriving B, then restore
        // the exact boundary pin. Runs in every residual evaluation
        // including FD-Jacobian probes -- like E_L, the recovery is a
        // smooth function of the state and freezing it per iteration would
        // make the Jacobian inconsistent with the iterate map.
        if (m_vacuum_recovery_half && m_joint_stage_purpose==JointStagePurpose::Standard) {
            m_hybrid_pic_model->ComputeVacuumARecovery(
                a_from_jacobian, a_thetadt);
            DarwinApplyABoundary(pin_time);
        }
        // B = B_static + curl A
        m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            B, A, m_WarpX->GetEBUpdateBFlag()[lev], lev,
            DarwinParticleCurlGrow());
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab const & Bs = *m_WarpX->m_fields.get("hybrid_B_static_fp", Direction{dir}, lev);
            amrex::MultiFab::Add(*B[dir], Bs, 0, 0, B[dir]->nComp(), B[dir]->nGrowVect());
        }
    }
    // B is DERIVED here (B = B_static + curl A): boundary conditions act on
    // A (DarwinApplyABoundary) and must not re-condition the curl, or the
    // wall ring picks up values inconsistent with the enclosed-flux pin.
    amrex::IntVect const ngB =
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->nGrowVect();
    m_WarpX->FillBoundaryB(ngB, true /* sync nodal points */);
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitHybrid::GetBfieldThetaForPC ( const int lev ) const
{
    // During the nonlinear solve, UpdateWarpXFields (called from every
    // residual evaluation) leaves the Bfield_fp registry holding the TOTAL
    // theta-midpoint field B^{n+theta} of the current iterate: the
    // Faraday-advanced plasma field with B_ext^{n+theta} assembled on top
    // (split-field external drive), or B_static + curl A^{n+theta} on the
    // Darwin path. Valid only after the first residual evaluation of the
    // current Newton iterate; before that (and between steps) the registry
    // holds the end-of-step totals B^{n+1} (= B^n at the next entry).
    using ablastr::fields::Direction;
    return { m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
             m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
             m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev) };
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitHybrid::GetIonCurrentForPC ( const int lev ) const
{
    // The Ohm solve consumes current_fp as the ion (particle) current
    // (see the HybridPICSolveE call in ComputeRHS). PreRHSOp deposits it
    // each residual evaluation and freezes it during Jacobian probes, so
    // between updatePreCondMat and the GMRES solve it holds exactly the
    // frozen drift-leg coefficient (J - J_i) x delta_B needs.
    using ablastr::fields::Direction;
    return { m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev),
             m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev),
             m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev) };
}

const amrex::MultiFab*
ThetaImplicitHybrid::GetRhoMidForPC ( const int lev ) const
{
    // The rho_fp registry carries two time slots of WarpX::ncomps components
    // each: component 0 holds the pre-push deposit (rho(x^n) only in the
    // first evaluation of a step; the previous evaluation's midpoint
    // positions afterwards), and component nComp()/2 holds the
    // midpoint-position deposit rho^{n+1/2} of the current iterate, written
    // by PreRHSOp's PushParticlesandDeposit in every residual evaluation.
    // Consumers should read component nComp()/2 (the same convention as
    // rho_mid_comp in HybridPICModel and the Darwin rho_half alias in
    // ComputeRHS). Valid only after the first residual evaluation of the
    // current Newton iterate has deposited it.
    return m_WarpX->m_fields.get(FieldType::rho_fp, lev);
}

std::pair<const amrex::MultiFab*, int>
ThetaImplicitHybrid::GetOhmDensityForPC (const int lev) const
{
    if (m_qdsmc_segregated_solve && !m_live_ion_density) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_qdsmc_rho_frozen.at(lev) != nullptr,
            "Ohm PC density requested before the thermal stage is frozen");
        return {m_qdsmc_rho_frozen[lev].get(), 0};
    }
    auto const* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    return {rho, (m_darwin_segregated_solve || m_live_ion_density)
                     ? rho->nComp()/2 : 0};
}

const amrex::MultiFab*
ThetaImplicitHybrid::GetRhoPolFrozenForPC ( const int lev ) const
{
    // Mirror of the poloidal stage's own density-source selection
    // (HybridPICModel::HybridPICSolveE): the per-step-frozen rho^n
    // snapshot once captured this step, otherwise nullptr so PC callers
    // fall back to the midpoint density exactly as the solve does.
    const HybridPICModel* hybrid = m_hybrid_pic_model;
    if (hybrid == nullptr
        || !hybrid->m_esolve_curlcurl
        || !hybrid->m_curlcurl_pol_frozen_rho
        || !hybrid->m_inertia_rho_n_captured) {
        return nullptr;
    }
    return m_WarpX->m_fields.get("hybrid_rho_n_frozen", lev);
}

namespace
{
    /** Per-node beta = 1/d_e^2 for the divided electron-inertia curl-curl
     *  operator, mirroring the E-solve kernel's floor and taper. Rows where
     *  the kernel zeroes the inertia term (no deposit, or fully tapered)
     *  are identity rows, approximated with the finite ceiling a_beta_id. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real InertiaBetaNode (amrex::Real a_rho, amrex::Real a_rho_floor,
                                 amrex::Real a_floor_w, amrex::Real a_taper_w,
                                 amrex::Real a_beta_fac, amrex::Real a_beta_id)
    {
        using amrex::Real;
        if (a_rho <= Real(0.0)) { return a_beta_id; }
        const Real rho_lim = HybridSmoothFloor(a_rho, a_rho_floor, a_floor_w);
        Real b = a_beta_fac*rho_lim;
        if (a_taper_w > Real(0.0)) {
            const Real tp = Real(0.5)*(Real(1.0)
                + std::tanh((a_rho - a_rho_floor)/a_taper_w));
            b /= amrex::max(tp, Real(1.0e-4));
        }
        // the physical branch needs no ceiling (finite rho -> finite beta;
        // the taper amplification is already capped at 1e4x local); the
        // identity ceiling a_beta_id applies only to the rho <= 0 rows
        return b;
    }
}

const amrex::Vector<amrex::Array<amrex::MultiFab*,3>>*
ThetaImplicitHybrid::FillInertiaBetaCoeff ()
{
    using namespace amrex;

    const HybridPICModel* hybrid = m_hybrid_pic_model;
    if (hybrid == nullptr || !hybrid->m_include_electron_inertia) {
        return nullptr;
    }

#if defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "jacobian.pc_type = pc_curl_curl_mlmg is not available for the "
        "hybrid solver in RZ (the curl-curl operator carries no cylindrical "
        "metric) - use jacobian.pc_type = pc_block_banded instead.");
    return nullptr; // unreachable
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        hybrid->m_electron_inertia_djedt_only
            && !hybrid->m_electron_inertia_bdf2,
        "pc_curl_curl_mlmg with the hybrid solver models the operator form "
        "of the electron-inertia term: set "
        "hybrid_pic_model.electron_inertia_djedt_only = 1 and "
        "hybrid_pic_model.electron_inertia_bdf2 = 0 (the form of Amano et "
        "al., J. Comput. Phys. 275, 197 (2014)), or use another "
        "preconditioner.");

    const bool vacuum_pc = m_darwin_vacuum_pc_regularization > 0.0_rt;
    if (vacuum_pc) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 && m_darwin_segregated_solve
            && m_vacuum_recovery_half && !m_external_field_iteration
            && hybrid->m_darwin_vacuum_recovery_operator == "edge_relaxation"
            && hybrid->m_darwin_vacuum_recovery_frozen_mask,
            "Darwin vacuum PC requires 3D split, native half-cadence recovery, frozen mask and no circuit iteration");
    }
    const Real rho_floor = static_cast<Real>(hybrid->m_n_floor)*PhysConst::q_e;
    const Real floor_w =
        static_cast<Real>(hybrid->m_n_floor_smooth_width)*rho_floor;
    const Real taper_w =
        static_cast<Real>(hybrid->m_electron_inertia_floor_taper)*rho_floor;
    const Real me_eff = static_cast<Real>(hybrid->m_electron_inertia_mass);
    // divided operator: beta(x) E + curl curl E = beta(x) b with
    // beta = 1/d_e^2 = mu0 q_e rho_lim / m_e_eff
    const Real beta_fac = PhysConst::mu0*PhysConst::q_e/me_eff;

    if (m_inertia_beta.empty()) {
        m_inertia_beta_owned.resize(m_num_amr_levels);
        m_inertia_beta.resize(m_num_amr_levels);
        if (vacuum_pc) {
            m_inertia_rhs_scale_owned.resize(m_num_amr_levels);
            m_inertia_rhs_scale.resize(m_num_amr_levels);
        }
        const auto& e_mfarrvec = m_E.getArrayVec();
        for (int lev = 0; lev < m_num_amr_levels; lev++) {
            for (int c = 0; c < 3; c++) {
                const MultiFab& emf = *e_mfarrvec[lev][c];
                m_inertia_beta_owned[lev][c] = std::make_unique<MultiFab>(
                    emf.boxArray(), emf.DistributionMap(), 1, 0);
                m_inertia_beta[lev][c] = m_inertia_beta_owned[lev][c].get();
                if (vacuum_pc) {
                    m_inertia_rhs_scale_owned[lev][c] = std::make_unique<MultiFab>(
                        emf.boxArray(), emf.DistributionMap(), 1, 0);
                    m_inertia_rhs_scale[lev][c] = m_inertia_rhs_scale_owned[lev][c].get();
                }
            }
        }
    }

    for (int lev = 0; lev < m_num_amr_levels; lev++) {
        Real vacuum_scale = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const Real dx = m_WarpX->Geom(lev).CellSize(d);
            vacuum_scale += 4.0_rt / (dx*dx);
        }
        const Real vacuum_beta = m_darwin_vacuum_pc_regularization * vacuum_scale;
        const Real mask_floor = rho_floor * hybrid->m_darwin_vacuum_recovery_density_fraction;
        const int mask_mode = hybrid->m_darwin_vacuum_recovery_mask == "global" ? 2
            : (hybrid->m_darwin_vacuum_recovery_mask == "transition" ? 1 : 0);
        const MultiFab* mask_rho = vacuum_pc
            ? m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev) : nullptr;
        const MultiFab* rho_mf = GetRhoMidForPC(lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho_mf != nullptr,
            "FillInertiaBetaCoeff: no midpoint density available");
        const int rho_comp = rho_mf->nComp()/2;
        // identity-row ceiling: anchored on the LARGEST density present
        // (anchoring on the floor breaks on decks whose n_floor is tiny
        // relative to the plasma - the ceiling must sit above every
        // physical beta on the level)
        const Real rho_max = rho_mf->max(rho_comp);
        const Real beta_id =
            beta_fac*amrex::max(rho_max, rho_floor)*Real(1.0e4);

        // conformal-wall mirror: the residual zeroes E on covered and cut
        // edges (ZeroConductorEdges inside every Ohm solve), making those
        // Jacobian rows identity -- mirror them with the identity-row beta
        const bool eb_mirror = EB::enabled()
            && hybrid->m_use_conformal_eb
            && hybrid->m_conformal_wall_conductor;
        const auto& eb_update_E = m_WarpX->GetEBUpdateEFlag();

        for (int c = 0; c < 3; c++) {
            MultiFab& bmf = *m_inertia_beta[lev][c];
            // this E component is cell-centered in at most one direction;
            // there the edge value is the harmonic mean of the two
            // neighboring density nodes (harmonic: the depleted node
            // dominates at the density-contrast edge)
            const IntVect etype = bmf.ixType().toIntVect();
            const int ox = (etype[0] == 0) ? 1 : 0;
            const int oy = (AMREX_SPACEDIM >= 2 && etype[1] == 0) ? 1 : 0;
            const int oz = (AMREX_SPACEDIM == 3 && etype[2] == 0) ? 1 : 0;
            const bool has_cc = (ox + oy + oz > 0);
            const Box pc_domain = amrex::convert(m_WarpX->Geom(lev).Domain(), etype);
            const auto pc_lo = pc_domain.smallEnd();
            const auto pc_hi = pc_domain.bigEnd();
            GpuArray<int,AMREX_SPACEDIM> pin_normal{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                pin_normal[d] = !etype[d] && !m_WarpX->Geom(lev).isPeriodic(d)
                    && (hybrid->m_add_external_fields || EB::enabled());
            }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(bmf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box bx = mfi.tilebox();
                const auto beta_arr = bmf.array(mfi);
                const auto rhs_scale = vacuum_pc ? m_inertia_rhs_scale[lev][c]->array(mfi)
                    : amrex::Array4<Real>{};
                const auto mask_arr = vacuum_pc ? mask_rho->const_array(mfi)
                    : amrex::Array4<Real const>{};
                const auto rho_arr = rho_mf->const_array(mfi, rho_comp);
                const auto eb_arr = eb_mirror
                    ? eb_update_E[lev][c]->const_array(mfi)
                    : amrex::Array4<int const>{};
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (eb_mirror && eb_arr(i,j,k) == 0) {
                        beta_arr(i,j,k) = beta_id;
                        if (vacuum_pc) { rhs_scale(i,j,k) = beta_id; }
                        return;
                    }
                    const Real ba = InertiaBetaNode(rho_arr(i,j,k),
                        rho_floor, floor_w, taper_w, beta_fac, beta_id);
                    Real bv = ba;
                    if (has_cc) {
                        const Real bb = InertiaBetaNode(
                            rho_arr(i+ox, j+oy, k+oz),
                            rho_floor, floor_w, taper_w, beta_fac, beta_id);
                        bv = Real(2.0)*ba*bb/(ba + bb);
                    }
                    beta_arr(i,j,k) = bv;
                    if (vacuum_pc) {
                        // Match the arithmetic nodal-to-edge mask interpolation
                        // used by native recovery and the Faraday overwrite.
                        const Real rho_edge = has_cc ? Real(0.5)*(mask_arr(i,j,k)
                            + mask_arr(i+ox,j+oy,k+oz)) : mask_arr(i,j,k);
                        const bool in_mask = mask_mode == 2
                            || (mask_mode == 0 && rho_edge < mask_floor)
                            || (mask_mode == 1 && rho_edge > 0.0_rt && rho_edge < mask_floor);
                        // The native vacuum Jacobian is K/Lambda. Regularize
                        // only its PC: (K + eps Lambda I)x = Lambda b.
                        beta_arr(i,j,k) = in_mask ? vacuum_beta : bv;
                        rhs_scale(i,j,k) = in_mask ? vacuum_scale : bv;
                        if (in_mask) {
                            // Darwin A pins include the first/last normal,
                            // cell-centered DOFs; MLCurlCurl PEC does not.
                            const IntVect iv(AMREX_D_DECL(i,j,k));
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                if (pin_normal[d] && (iv[d] == pc_lo[d] || iv[d] == pc_hi[d])) {
                                    const Real pin_beta = amrex::max(beta_id, Real(1.0e4)*vacuum_scale);
                                    beta_arr(i,j,k) = pin_beta;
                                    rhs_scale(i,j,k) = pin_beta;
                                }
                            }
                        }
                    }
                });
            }
        }
    }
    return &m_inertia_beta;
#endif
}

void ThetaImplicitHybrid::ProjectDarwinVacuumGauge (WarpXSolverVec& field, bool preconditioner)
{
    if (!m_darwin_vacuum_gauge_projection) { return; }
#if defined(WARPX_DIM_3D)
    BL_PROFILE("ThetaImplicitHybrid::ProjectDarwinVacuumGauge()");
    using namespace amrex;
    constexpr int lev = 0;
    auto const& geom = m_WarpX->Geom(lev);
    auto const& period = geom.periodicity();
    auto const dx = geom.CellSizeArray();
    auto& rho = *m_WarpX->m_fields.get("hybrid_rho_vacmask_fp", lev);
    auto const& shape = *m_WarpX->m_fields.get("hybrid_phi_darwin_fp", lev);
    auto const& vectors = field.getArrayVec()[lev];
    if (!m_vacuum_gauge_solver) {
        rho.OverrideSync(period);
        rho.FillBoundary(period);
        m_vacuum_gauge_mask = std::make_unique<iMultiFab>(
            shape.boxArray(), shape.DistributionMap(), 1, 0);
        m_vacuum_gauge_phi = std::make_unique<MultiFab>(
            shape.boxArray(), shape.DistributionMap(), 1, 1);
        m_vacuum_gauge_rhs = std::make_unique<MultiFab>(
            shape.boxArray(), shape.DistributionMap(), 1, 0);
        Real const floor = PhysConst::q_e * m_hybrid_pic_model->m_n_floor
            * m_hybrid_pic_model->m_darwin_vacuum_recovery_density_fraction;
        int const mode = m_hybrid_pic_model->m_darwin_vacuum_recovery_mask == "global" ? 2
            : (m_hybrid_pic_model->m_darwin_vacuum_recovery_mask == "transition" ? 1 : 0);
        auto const dom = surroundingNodes(geom.Domain());
        auto const lo = dom.smallEnd();
        auto const hi = dom.bigEnd();
        GpuArray<int,3> per{geom.isPeriodic(0),geom.isPeriodic(1),geom.isPeriodic(2)};
        bool const use_eb = EB::enabled();
        auto const& flags = m_WarpX->GetEBUpdateEFlag();
        for (MFIter mfi(*m_vacuum_gauge_mask); mfi.isValid(); ++mfi) {
            auto const mask = m_vacuum_gauge_mask->array(mfi);
            auto const rr = rho.const_array(mfi);
            GpuArray<Array4<int const>,3> eb{};
            if (use_eb) {
                for (int d = 0; d < 3; ++d) { eb[d] = flags[lev][d]->const_array(mfi); }
            }
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                IntVect const iv(i,j,k);
                // phi=0 on the wall AND its neighboring nodal layer keeps
                // grad(phi)=0 on all pinned tangential and normal A rows.
                for (int d = 0; d < 3; ++d) {
                    if (!per[d] && (iv[d] <= lo[d]+1 || iv[d] >= hi[d]-1)) {
                        mask(iv) = 0; return;
                    }
                }
                // Permit a potential only if all incident edges belong to
                // the unconstrained vacuum. Its gradient is then zero on
                // plasma/covered rows and lies in the native curl nullspace.
                for (int d = 0; d < 3; ++d) {
                    IntVect off(0); off[d] = 1;
                    for (int side = 0; side < 2; ++side) {
                        IntVect const edge = side ? iv-off : iv;
                        Real const re = Real(0.5)*(rr(edge)+rr(edge+off));
                        bool const vac = mode == 2 || (mode == 0 && re < floor)
                            || (mode == 1 && re > 0.0_rt && re < floor);
                        if (!vac || (use_eb && eb[d](edge) == 0)) {
                            mask(iv) = 0; return;
                        }
                    }
                }
                mask(iv) = 1;
            });
        }
        m_vacuum_gauge_mask->OverrideSync(period);
        LPInfo const info;
        m_vacuum_gauge_op = std::make_unique<MLNodeTensorLaplacian>(
            Vector<Geometry>{geom}, Vector<BoxArray>{m_WarpX->boxArray(lev)},
            Vector<DistributionMapping>{m_WarpX->DistributionMap(lev)}, info);
        m_vacuum_gauge_op->setSigma({1._rt,0._rt,0._rt,1._rt,0._rt,1._rt});
        Array<LinOpBCType,3> lo_bc, hi_bc;
        for (int d = 0; d < 3; ++d) {
            lo_bc[d] = hi_bc[d] = per[d] ? LinOpBCType::Periodic : LinOpBCType::Dirichlet;
        }
        m_vacuum_gauge_op->setDomainBC(lo_bc,hi_bc);
        // With no known nodes in a fully periodic vacuum, leave AMReX's
        // singular/nullspace handling active rather than attaching an all-1 mask.
        if (m_vacuum_gauge_mask->min(0) == 0) {
            m_vacuum_gauge_op->setOversetMask(lev,*m_vacuum_gauge_mask);
        }
        m_vacuum_gauge_solver = std::make_unique<MLMG>(*m_vacuum_gauge_op);
        m_vacuum_gauge_solver->setVerbose(0);
        m_vacuum_gauge_solver->setMaxIter(200);
    }
    // Solver vectors have no ghosts. Divergence needs neighboring edges,
    // including those across box/rank boundaries, so use owned ghosted scratch.
    for (int d = 0; d < 3; ++d) {
        auto& edge = m_vacuum_gauge_edges[d];
        if (!edge.isDefined()) {
            edge.define(vectors[d]->boxArray(), vectors[d]->DistributionMap(), 1, 1);
        }
        edge.setVal(0.0_rt);
        MultiFab::Copy(edge,*vectors[d],0,0,1,0);
        edge.OverrideSync(period);
        edge.FillBoundary(period);
    }
    ablastr::fields::VectorField vf{
        &m_vacuum_gauge_edges[0],&m_vacuum_gauge_edges[1],&m_vacuum_gauge_edges[2]};
    auto& rhs = *m_vacuum_gauge_rhs;
    auto& phi = *m_vacuum_gauge_phi;
    m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeDivE(vf,rhs);
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto const rr = rhs.array(mfi);
        auto const mask = m_vacuum_gauge_mask->const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
            if (mask(i,j,k) == 0) { rr(i,j,k) = 0.0_rt; }
        });
    }
    rhs.OverrideSync(period);
    phi.setVal(0.0_rt);
    // Only preconditioner corrections may use a cheaper projection. The split
    // solution and warm start retain the strict gauge used by the field gates.
    Real const rtol = preconditioner ? m_darwin_vacuum_gauge_pc_rtol : 1.e-12_rt;
    m_vacuum_gauge_solver->solve({&phi},{&rhs},rtol,0.0_rt);
    phi.OverrideSync(period);
    phi.FillBoundary(period);
    for (int d = 0; d < 3; ++d) {
        auto& v = *vectors[d];
        IntVect off(0); off[d] = 1;
        Real const idx = 1.0_rt/dx[d];
        for (MFIter mfi(v); mfi.isValid(); ++mfi) {
            auto const vv = v.array(mfi);
            auto const pp = phi.const_array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                IntVect const iv(i,j,k);
                vv(iv) -= (pp(iv+off)-pp(iv))*idx;
            });
        }
        v.OverrideSync(period);
        v.FillBoundary(period);
    }
#else
    amrex::ignore_unused(field, preconditioner);
    WARPX_ABORT_WITH_MESSAGE("darwin_vacuum_gauge_projection requires 3D");
#endif
}

void ThetaImplicitHybrid::DarwinDeriveB ()
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField A =
            m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
        ablastr::fields::VectorField B =
            m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
            B, A, m_WarpX->GetEBUpdateBFlag()[lev], lev,
            DarwinParticleCurlGrow());
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab const & Bs = *m_WarpX->m_fields.get(
                "hybrid_B_static_fp", Direction{dir}, lev);
            amrex::MultiFab::Add(*B[dir], Bs, 0, 0,
                                 B[dir]->nComp(), B[dir]->nGrowVect());
        }
    }
    // Derived B: boundary conditions act on A (DarwinApplyABoundary), so
    // only exchange/sync the ghosts of the derived field.
    amrex::IntVect const ngB =
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->nGrowVect();
    m_WarpX->FillBoundaryB(ngB, true /* sync nodal points */);
}

void ThetaImplicitHybrid::DarwinApplyABoundary ( amrex::Real a_time )
{
    using ablastr::fields::Direction;
    constexpr int NODE = amrex::IndexType::NODE;

    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
        pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
    }
    const bool has_external = m_hybrid_pic_model->m_add_external_fields;
    const bool has_eb = !m_WarpX->GetEBUpdateEFlag().empty()
        && m_WarpX->GetEBUpdateEFlag()[0][0] != nullptr;
    bool any_pmc = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        any_pmc = any_pmc || pmc_lo[d] || pmc_hi[d];
    }
    if (!has_external && !has_eb && !any_pmc)
    {
        return;
    }

    // Gauge: A was zeroed at initialization, so boundary values impose the
    // CHANGE of the external vector potential since then.
    amrex::Vector<amrex::Real> scales;
    if (has_external) {
        auto & ext = *m_hybrid_pic_model->m_external_vector_potential;
        if (m_fext_init.empty()) {
            for (int i = 0; i < ext.nFields(); ++i) {
                m_fext_init.push_back(ext.TimeScale(i, a_time));
            }
        }
        for (int i = 0; i < ext.nFields(); ++i) {
            scales.push_back(ext.DeviceDriven(i) ? 0. : ext.TimeScale(i, a_time) - m_fext_init[i]);
        }
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const amrex::Box& domain = m_WarpX->Geom(lev).Domain();
        const amrex::Periodicity& period = m_WarpX->Geom(lev).periodicity();

        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & A = *m_WarpX->m_fields.get("hybrid_A_fp", Direction{dir}, lev);

            // Sum of the (gauge-shifted) external vector potentials on this
            // component's staggering.
            amrex::MultiFab A_bc(A.boxArray(), A.DistributionMap(), 1, A.nGrowVect());
            A_bc.setVal(0.0_rt);
            if (has_external) {
                auto & ext = *m_hybrid_pic_model->m_external_vector_potential;
                for (int i = 0; i < ext.nFields(); ++i) {
                    amrex::MultiFab const & Aext = *m_WarpX->m_fields.get(
                        ext.FieldName(i) + "_Aext", Direction{dir}, lev);
                    auto const ng = amrex::min(A.nGrowVect(), Aext.nGrowVect());
                    if (ext.DeviceDriven(i)) {
                        auto const device = ext.DeviceScales();
                        double const initial = m_fext_init[i];
                        int const field = i;
                        for (amrex::MFIter mfi(A_bc,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                            auto const dst = A_bc.array(mfi); auto const unit = Aext.const_array(mfi);
                            amrex::ParallelFor(mfi.growntilebox(ng),
                                [=] AMREX_GPU_DEVICE(int ii,int jj,int kk) noexcept {
                                    dst(ii,jj,kk) += (device.Value(field,a_time)-initial)*unit(ii,jj,kk);
                                });
                        }
                    } else {
                        amrex::MultiFab::Saxpy(A_bc,scales[i],Aext,0,0,1,ng);
                    }
                }
            }

            if (std::getenv("WARPX_DEBUG_ABC") != nullptr) {
                amrex::Print() << "[A-bc] t=" << a_time << " dir " << dir
                    << " |A_bc|max = " << A_bc.norminf(0)
                    << " scale0 = " << (scales.empty() ? 0.0 : scales[0])
                    << " |A|max = " << A.norminf(0) << "\n";
            }

            const amrex::iMultiFab* eb_flag = has_eb
                ? m_WarpX->GetEBUpdateEFlag()[lev][dir].get() : nullptr;

            for (amrex::MFIter mfi(A, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box tb = mfi.tilebox();
                tb.grow(A.nGrowVect());
                const amrex::Box domain_t = amrex::convert(domain, A.ixType().toIntVect());

                amrex::Array4<amrex::Real> const& a = A.array(mfi);
                amrex::Array4<amrex::Real const> const& abc = A_bc.const_array(mfi);
                amrex::Array4<int const> eb;
                if (eb_flag) { eb = eb_flag->const_array(mfi); }
                const bool use_eb = (eb_flag != nullptr);

                amrex::GpuArray<int, 3> dlo{{0, 0, 0}};
                amrex::GpuArray<int, 3> dhi{{0, 0, 0}};
                amrex::GpuArray<int, 3> per{{1, 1, 1}};
                amrex::GpuArray<int, 3> nodal{{0, 0, 0}};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    dlo[d] = domain_t.smallEnd(d);
                    dhi[d] = domain_t.bigEnd(d);
                    per[d] = period.isPeriodic(d) ? 1 : 0;
                    nodal[d] = A.ixType().nodeCentered(d);
                }

#if defined(WARPX_DIM_RZ)
                bool const on_axis = m_WarpX->Geom(lev).ProbLo(0) == 0.;
#endif
                amrex::ParallelFor(tb,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
#if defined(WARPX_DIM_RZ)
                                       if (on_axis && dir == 1 && i == 0)
                                       {
                                           a(i, j, k) = 0.;
                                           return;
                                       }
#endif
                                       // Embedded conductors: hold A at the gauge zero inside
                                       // masked cells (frozen enclosed flux; the interior field
                                       // stays at B_static).
                                       if (use_eb && eb(i, j, k) == 0)
                                       {
                                           a(i, j, k) = 0.0_rt;
                                           return;
                                       }
                                       // PEC tangential nodes carry the imposed
                                       // drive. PMC tangential nodes are free:
                                       // their response is even. Pin only
                                       // points that lie on a physical wall
                                       // node. A normal component is half a
                                       // cell inside the face; its ghost
                                       // reflection below supplies the wall
                                       // trace.
                                       const int idx[3] = {i, j, k};
                                       bool on_boundary = false;
                                       for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                       {
                                           if (per[d] || !nodal[d])
                                           {
                                               continue;
                                           }
                                           bool lower = idx[d] <= dlo[d] && !pmc_lo[d];
#if defined(WARPX_DIM_RZ)
                                           if (d == 0 && on_axis)
                                           {
                                               lower = false;
                                           }
#endif
                                           if (lower || (idx[d] >= dhi[d] && !pmc_hi[d]))
                                           {
                                               on_boundary = true;
                                           }
                                       }
                                       if (on_boundary)
                                       {
                                           // Clamp the imposed value to the domain edge:
                                           // ghosts continue the wall value rather than the
                                           // (growing) exterior vector potential, so the wall
                                           // ring carries no spurious curl sheet.
                                           int ic[3] = {i, j, k};
                                           for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                           {
                                               if (per[d])
                                               {
                                                   continue;
                                               }
#if defined(WARPX_DIM_RZ)
                                               if (d == 0 && on_axis && ic[d] < dlo[d])
                                               {
                                                   continue;
                                               }
#endif
                                               ic[d] = amrex::Clamp(ic[d], dlo[d], dhi[d]);
                                           }
                                           a(i, j, k) = abc(ic[0], ic[1], ic[2]);
                                       }
                                   });
            }
            A.FillBoundary(m_WarpX->Geom(lev).periodicity());
#if defined(WARPX_DIM_RZ)
            bool const skip_lower_axis = m_WarpX->Geom(lev).ProbLo(0) == 0.;
#else
            bool const skip_lower_axis = false;
#endif
            ApplyDarwinCellCenteredABoundary(A, A_bc, m_WarpX->Geom(lev), skip_lower_axis, eb_flag,
                                             pmc_lo, pmc_hi);
        }
#if defined(WARPX_DIM_RZ)
        if (m_WarpX->Geom(lev).ProbLo(0) == 0.)
        {
            auto A = m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
            m_WarpX->ApplyFieldBoundaryOnAxis(A[0], A[1], A[2], lev);
        }
#endif
    }
    amrex::ignore_unused(NODE);
}

void ThetaImplicitHybrid::FinishFieldUpdate( amrex::Real end_time, bool owned_ampere )
{
    BL_PROFILE("ThetaImplicitHybrid::FinishFieldUpdate()");

    if(auto* paired=warpx::darwin::NativeActivePairedFields(*m_WarpX)) {
        auto const& old=m_Eold.getArrayVec()[0];
        paired->Extrapolate(m_theta,m_E.getArrayVec()[0],{old[0],old[1],old[2]});
        m_WarpX->SetElectricFieldAndApplyBCs(m_E,end_time);
        paired->ConditionTransverse();
        paired->BuildMagnetic();
        paired->RefreshAmpere(m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0));
        paired->AssembleTotalElectric();
        return;
    }
    // Extrapolate from t^{n+theta} to t^{n+1}:
    // F^{n+1} = (1/theta)*F^{n+theta} + (1 - 1/theta)*F^n
    const amrex::Real c0 = 1.0_rt / m_theta;
    const amrex::Real c1 = 1.0_rt - c0;

    // E^{n+1} (the transverse part on the Darwin path).
    // On the guarded driven-current route this extrapolation is provisional:
    // retained endpoint closure publishes physical -A_rate plus imaged EL
    // before accepted callbacks/source leases. BeforeEndpoint observers and
    // endpoint audit publication_jump see this provisional extrapolation, not
    // an accepted physical electric inventory. No particle push occurs here.
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );

    if (m_darwin) {
        using ablastr::fields::Direction;
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                // A^{n+1} = A_old + (A^{n+theta} - A_old)/theta
                //         = A_old - dt E_T^{n+theta}
                amrex::MultiFab & A = *m_WarpX->m_fields.get("hybrid_A_fp", Direction{dir}, lev);
                amrex::MultiFab const & A_old = *m_WarpX->m_fields.get("hybrid_A_old_fp", Direction{dir}, lev);
                amrex::MultiFab::LinComb(A, c0, A, 0, c1, A_old, 0,
                                         0, A.nComp(), A.nGrowVect());
                // E_L^{n+1} = (E_L^{n+theta} - (1-theta) E_L^n)/theta
                amrex::MultiFab & EL = *m_WarpX->m_fields.get("hybrid_E_long_fp", Direction{dir}, lev);
                amrex::MultiFab const & EL_old = *m_WarpX->m_fields.get("hybrid_E_long_old_fp", Direction{dir}, lev);
                if(warpx::darwin::increment::Enabled())warpx::darwin::increment::Extrapolate(
                    EL,EL_old,*m_WarpX->m_fields.get(warpx::darwin::increment::field_low,Direction{dir},lev),m_theta);
                else amrex::MultiFab::LinComb(EL, c0, EL, 0, c1, EL_old, 0,
                                         0, EL.nComp(), EL.nGrowVect());
                // Full E^{n+1} = E_T^{n+1} + E_L^{n+1} (SetElectricFieldAndApplyBCs
                // above wrote the transverse part into Efield_fp)
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                warpx::darwin::increment::AddFull(E,EL,warpx::darwin::increment::Low(*m_WarpX,dir));
            }
        }
        if (m_external_field_iteration && !m_circuit_native) {
            // Final circuit pass against the converged theta-stage plasma
            // current (hybrid_current_fp_plasma as left by the last
            // residual evaluation), leaving the circuit state -- and the
            // coil scale segments the pin below reads -- at t^{n+1}.
            ExecutePythonCallback("externalcoilfinish");
        }
        DarwinApplyABoundary(end_time);
        // Vacuum recovery at the full-step state (both cadences: in "half"
        // mode the theta-stage was recovered inside the solve, and the
        // extrapolated end state gets the same treatment so the delivered
        // field is exactly recovered; in "full" mode this is the only
        // application). Restore the exact boundary pin afterwards.
        if (m_vacuum_recovery) {
            // Half cadence already relaxed over theta*dt inside the
            // solve; the end application covers the remaining
            // (1 - theta)*dt so the configured tau is the effective
            // response time in both cadences.
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            // The stage correction defines a preconditioned fixed-point
            // residual. At the endpoint one application of that global inverse
            // followed by a mask is not a vacuum projection: it can amplify
            // interface errors. Solve the actual constrained native operator.
            // A homogeneous vacuum-current projection is incompatible with
            // the owned implicit Je/D endpoint. Preserve its affine A producer;
            // the fresh physical all-free-row check follows current rotation.
            if(!owned_ampere)RecoverDarwinVacuumA(*m_WarpX, *m_hybrid_pic_model,
                m_WarpX->m_fields.get_alldirs("hybrid_A_fp", 0));
#else
            m_hybrid_pic_model->ComputeVacuumARecovery(false,
                m_vacuum_recovery_half ? (1.0_rt - m_theta) * m_dt : m_dt);
#endif
            DarwinApplyABoundary(end_time);
            // Recover the endpoint transverse electric field spatially in 3D/RZ.
            // The implicit current and theta-stage updates are unchanged.
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            auto const& endpoint = m_E.getArrayVec()[0];
            auto const& accepted = m_Eold.getArrayVec()[0];
            RecoverDarwinVacuumE(*m_WarpX, *m_hybrid_pic_model,
                {endpoint[0], endpoint[1], endpoint[2]},
                {accepted[0], accepted[1], accepted[2]}, end_time, m_dt);
#else
            m_hybrid_pic_model->ApplyVacuumFaradayE(m_dt, true, false,
                                                    true /* BDF2 */);
            // Rotate the A history for the next step's BDF2: A^n becomes
            // A^{n-1} (A_old still holds A^n here; it is rewritten from the
            // end state at the next OneStep entry).
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    amrex::MultiFab & Anm1 = *m_WarpX->m_fields.get(
                        "hybrid_A_vac_nm1_fp", Direction{dir}, lev);
                    amrex::MultiFab const & Aold = *m_WarpX->m_fields.get(
                        "hybrid_A_old_fp", Direction{dir}, lev);
                    amrex::MultiFab::Copy(Anm1, Aold, 0, 0, Anm1.nComp(),
                                          Anm1.nGrowVect());
                }
            }
#endif
        }
        // B^{n+1} = B_static + curl A^{n+1}
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            ablastr::fields::VectorField A = m_WarpX->m_fields.get_alldirs("hybrid_A_fp", lev);
            ablastr::fields::VectorField B = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
            m_WarpX->get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(
                B, A, m_WarpX->GetEBUpdateBFlag()[lev], lev,
                DarwinParticleCurlGrow());
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab const & Bs = *m_WarpX->m_fields.get("hybrid_B_static_fp", Direction{dir}, lev);
                amrex::MultiFab::Add(*B[dir], Bs, 0, 0, B[dir]->nComp(), B[dir]->nGrowVect());
            }
        }
        // Derived B: no independent boundary conditioning (see
        // DarwinUpdateA_B).
        amrex::IntVect const ngB =
            m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0)->nGrowVect();
        m_WarpX->FillBoundaryB(ngB, true /* sync nodal points */);
    } else {
        // The residual evaluations (and the post-solve UpdateWarpXFields)
        // leave Bfield_fp holding the TOTAL theta-time field, with
        // B_ext^{n+theta} assembled on top of the Faraday-advanced plasma
        // field, while B_old holds the plasma response alone. Strip the
        // still-stored theta-time external before the theta-extrapolation:
        // extrapolating the total against the plasma B_old amplifies
        // B_ext^{n+theta} by 1/theta, and with B_ext^{n+1} restored below
        // the carried plasma field would gain a spurious
        // (1/theta)*B_ext^{n+theta} every step (the vacuum-ramp deck
        // integrates the programmed ramp to a ~300x overshoot).
        ExtLedgerPrint(m_WarpX, "finish pre-theta-strip");
        if (m_hybrid_pic_model->m_add_external_fields) {
            using ablastr::fields::Direction;
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                for (int dir = 0; dir < 3; ++dir) {
                    amrex::MultiFab & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                    amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{dir}, lev);
                    amrex::MultiFab::Subtract(B, B_ext, 0, 0, B.nComp(), B.nGrowVect());
                }
            }
        }
        ExtLedgerPrint(m_WarpX, "finish post-theta-strip");
        // B^{n+1}
        ablastr::fields::MultiLevelVectorField const& B_old =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
        m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );
        ExtLedgerPrint(m_WarpX, "finish post-extrap");
    }

    // (The electron-inertia Je history rotates in Advance, AFTER the
    // delivered-state plasma-current refresh: the stored history values
    // must be MEASURED end-of-step assemblies, never extrapolations --
    // storing the extrapolation (Je^theta - (1-theta) Je^n)/theta feeds
    // the stored value back into itself with eigenvalue -(1-theta)/theta,
    // which is marginal (-1) at theta = 1/2 and rings at period 2 where
    // the inertia term dominates the Ohm law.)

    // Restore end-of-step totals: the analytic external flux advance means
    // Bfield_fp = B_plasma^{n+1} + f(t^{n+1}) curl A_ext exactly, for any
    // ramp shape (OneStep strips the same values at the next entry).
    // Split-field form only: under the Darwin unified drive the external
    // flux already lives inside A through its boundary values, and adding
    // E_ext here would poison the saved E^n and re-inject the drive
    // volumetrically through the A rebuild (doubling the programmed flux).
    if (m_hybrid_pic_model->m_add_external_fields && !m_darwin) {
        using ablastr::fields::Direction;
        if (m_external_field_iteration) {
            // Final circuit pass against the accepted end-of-step plasma
            // response (Bfield_fp still holds the response-only field
            // here), leaving the circuit state -- and the coil segments
            // the refresh below reads -- at t^{n+1}.
            ExecutePythonCallback("externalcoilfinish");
        }
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            end_time, m_dt);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            for (int dir = 0; dir < 3; ++dir) {
                amrex::MultiFab & B = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev);
                amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Add(B, B_ext, 0, 0, B.nComp(), B.nGrowVect());
                amrex::MultiFab & E = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{dir}, lev);
                amrex::MultiFab const & E_ext = *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{dir}, lev);
                amrex::MultiFab::Add(E, E_ext, 0, 0, E.nComp(), E.nGrowVect());
            }
        }
        ExtLedgerPrint(m_WarpX, "finish post-restore");
    }
}

void ThetaImplicitHybrid::AddSplitExternalFields ( amrex::Real a_sign )
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab & B = *m_WarpX->m_fields.get(
                FieldType::Bfield_fp, Direction{dir}, lev);
            amrex::MultiFab const & B_ext = *m_WarpX->m_fields.get(
                FieldType::hybrid_B_fp_external, Direction{dir}, lev);
            amrex::MultiFab::Saxpy(B, a_sign, B_ext, 0, 0,
                                   B.nComp(), B.nGrowVect());
            amrex::MultiFab & E = *m_WarpX->m_fields.get(
                FieldType::Efield_fp, Direction{dir}, lev);
            amrex::MultiFab const & E_ext = *m_WarpX->m_fields.get(
                FieldType::hybrid_E_fp_external, Direction{dir}, lev);
            amrex::MultiFab::Saxpy(E, a_sign, E_ext, 0, 0,
                                   E.nComp(), E.nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::WriteDarwinDriveReference (std::string const& directory) const
{
    if (!m_darwin || !m_hybrid_pic_model->UsesEulerianElectronEnergy() ||
        !amrex::ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out(directory+"/darwin_drive_reference.dat");
    out<<"WarpXDarwinDrive 1\n"<<m_fext_init.size()<<'\n'
       <<std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    auto const* ext=m_hybrid_pic_model->m_external_vector_potential.get();
    for (std::size_t i=0;i<m_fext_init.size();++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ext && int(i)<ext->nFields() &&
            std::isfinite(m_fext_init[i]), "Invalid accepted Darwin external reference");
        out<<ext->FieldName(int(i))<<' '<<m_fext_init[i]<<'\n';
    }
    out.close();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(out),"Unable to write Darwin drive reference");
}
void ThetaImplicitHybrid::ReadDarwinDriveReference (std::string const& directory)
{
    if (!m_darwin || !m_hybrid_pic_model->UsesEulerianElectronEnergy()) { return; }
    std::ifstream in(directory+"/darwin_drive_reference.dat");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in),
        "Eulerian Darwin restart is missing the accepted external drive reference");
    std::string magic;int version=0,count=-1;in>>magic>>version>>count;
    auto const* ext=m_hybrid_pic_model->m_external_vector_potential.get();
    int const expected=ext ? ext->nFields() : 0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in) && magic=="WarpXDarwinDrive" && version==1 &&
        (count==expected || (count==0 && m_WarpX->getistep(0)==0)),
        "Darwin external drive checkpoint has incompatible field count or version");
    amrex::Vector<amrex::Real> reference(count);
    for(int i=0;i<count;++i) {
        std::string name;in>>name>>reference[i];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(in) && name==ext->FieldName(i) &&
            std::isfinite(reference[i]),"Invalid or reordered Darwin external drive checkpoint");
    }
    in>>std::ws;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in.eof(),"Trailing Darwin external drive checkpoint data");
    m_fext_init=std::move(reference);
}


bool ThetaImplicitHybrid::NativeEndpointCheckpointReady() const {
    if(!m_native_endpoint_pair_requested)return true;
    bool const scope=NativeEndpointPairScopeSupported();
    bool valid=m_native_endpoint_pair_accept_requested&&m_native_endpoint_pair&&
        !m_native_candidate_active&&scope;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    return valid&&m_native_endpoint_pair->CheckpointReady();
}
void ThetaImplicitHybrid::WriteAcceptedEndpointCheckpoint(std::string const& directory) const {
    if(!m_native_endpoint_pair_requested)return;
    namespace io=warpx::darwin::endpoint_checkpoint;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(NativeEndpointCheckpointReady(),
        "Cannot checkpoint an unaccepted retained endpoint transaction");
    std::string const root=directory+"/native_endpoint_pair";
    bool created=true;if(amrex::ParallelDescriptor::IOProcessor())created=amrex::UtilCreateDirectory(root,0755);
    amrex::ParallelDescriptor::ReduceBoolAnd(created);WARPX_ALWAYS_ASSERT_WITH_MESSAGE(created,"Cannot create endpoint checkpoint directory");
    amrex::ParallelDescriptor::Barrier();
    // Accepted paired restart needs these exact coordinates before any lazy
    // PIC deposition or thermal rebuild. Preserve every component and guard;
    // standard PIC checkpoints omit rho/Pe and conditionally save Te/Ji.
    for(auto const& entry:{std::pair<char const*,FieldType>{"physical_rho",FieldType::rho_fp},
        {"physical_Te",FieldType::hybrid_electron_temperature_fp},
        {"physical_Pe",FieldType::hybrid_electron_pressure_fp}})
        io::Write(root+"/"+entry.first,*m_WarpX->m_fields.get(entry.second,0));
    for(int d=0;d<3;++d)io::Write(root+"/physical_Ji"+std::to_string(d),
        *m_WarpX->m_fields.get_alldirs(FieldType::current_fp,0)[d]);
    for(auto const& entry:{std::pair<char const*,WarpXSolverVec const*>{"private_ET",&m_E},
        {"private_old",&m_Eold},{"private_previous",&m_Eprev}})
        for(int d=0;d<3;++d)io::Write(root+"/"+entry.first+std::to_string(d),*entry.second->getArrayVec()[0][d]);
    m_native_endpoint_pair->WriteEndpointCheckpoint(root);
    io::WriteText(root+"/configuration",NativeEndpointCheckpointConfiguration(*m_WarpX));
    std::ostringstream flags;flags<<m_have_Eold<<' '<<m_have_Eprev<<'\n';
    io::WriteText(root+"/solver",flags.str());
    // AMReX writes the global maximum ID counter. Keep the per-birth-rank
    // counters too, so same-rank restarts reproduce future particle creation.
    // Read storage directly: the zero-argument NextID() advances the counter.
    auto const next_ids=amrex::ParallelDescriptor::Gather(
        WarpXParticleContainer::ParticleType::the_next_id,
        amrex::ParallelDescriptor::IOProcessorNumber());
    std::ostringstream counters;counters<<amrex::ParallelDescriptor::NProcs()<<'\n';
    for(auto const next:next_ids)counters<<next<<'\n';
    io::WriteText(root+"/particle_next_ids",counters.str());
}
void ThetaImplicitHybrid::ReadAcceptedEndpointCheckpoint(std::string const& directory) {
    if(!m_native_endpoint_pair_requested)return;
    namespace io=warpx::darwin::endpoint_checkpoint;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_defined&&m_darwin_initialized&&
        m_hybrid_pic_model->m_darwin_checkpoint_restored&&m_native_endpoint_pair&&
        m_native_endpoint_pair_accept_requested&&!m_native_candidate_active&&NativeEndpointPairScopeSupported(),
        "Unsupported retained endpoint restart configuration");
    std::string const root=directory+"/native_endpoint_pair";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(io::ReadText(root+"/configuration")==NativeEndpointCheckpointConfiguration(*m_WarpX),
        "Accepted endpoint physical configuration changed on restart");
    std::istringstream flags(io::ReadText(root+"/solver"));int old=-1,previous=-1;flags>>old>>previous;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(flags)&&old==(m_WarpX->getistep(0)>0)&&previous==0,
        "Invalid accepted endpoint private history flags");
    flags>>std::ws;WARPX_ALWAYS_ASSERT_WITH_MESSAGE(flags.eof(),"Trailing private endpoint history flags");
    std::istringstream counters(io::ReadText(root+"/particle_next_ids"));
    int ranks=0;counters>>ranks;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(counters)&&ranks==amrex::ParallelDescriptor::NProcs(),
        "Accepted endpoint particle ID restart requires the original rank count");
    std::vector<amrex::Long> next_ids(ranks);
    for(auto& next:next_ids) {
        counters>>next;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(counters)&&next>0&&
            next<=amrex::LongParticleIds::LastParticleID+1,
            "Invalid accepted endpoint particle ID counter");
    }
    counters>>std::ws;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(counters.eof()&&
        *std::max_element(next_ids.begin(),next_ids.end())==
            WarpXParticleContainer::ParticleType::the_next_id,
        "Accepted endpoint particle ID counters disagree with the native checkpoint");
    // Accepted paired restart needs these exact coordinates before any lazy
    // PIC deposition or thermal rebuild. Preserve every component and guard;
    // standard PIC checkpoints omit rho/Pe and conditionally save Te/Ji.
    for(auto const& entry:{std::pair<char const*,FieldType>{"physical_rho",FieldType::rho_fp},
        {"physical_Te",FieldType::hybrid_electron_temperature_fp},
        {"physical_Pe",FieldType::hybrid_electron_pressure_fp}})
        io::Read(root+"/"+entry.first,*m_WarpX->m_fields.get(entry.second,0));
    for(int d=0;d<3;++d)io::Read(root+"/physical_Ji"+std::to_string(d),
        *m_WarpX->m_fields.get_alldirs(FieldType::current_fp,0)[d]);
    for(auto const& entry:{std::pair<char const*,WarpXSolverVec*>{"private_ET",&m_E},
        {"private_old",&m_Eold},{"private_previous",&m_Eprev}})
        for(int d=0;d<3;++d)io::Read(root+"/"+entry.first+std::to_string(d),*entry.second->getArrayVec()[0][d]);
    auto const& support=*m_WarpX->m_fields.get(NativeVacuumSupportRecordName(),0);
    auto& mask=*m_WarpX->m_fields.get("hybrid_rho_vacmask_fp",0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(support.nComp()==2&&support.nGrowVect().allGE(mask.nGrowVect()),
        "Invalid retained endpoint support checkpoint");
    amrex::MultiFab::Copy(mask,support,1,0,1,mask.nGrowVect());
    // Check private solver coordinates before the owner can grant a lease.
    for(int d=0;d<3;++d){auto const& field=*m_E.getArrayVec()[0][d];
        auto const& shape=*m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp,0)[d];
        amrex::MultiFab expected(shape.boxArray(),shape.DistributionMap(),1,shape.nGrowVect());
        io::Read(root+"/part0_"+std::to_string(d),expected);
        amrex::MultiFab delta(field.boxArray(),field.DistributionMap(),1,0);
        amrex::MultiFab::Copy(delta,field,0,0,1,0);
        amrex::MultiFab::Subtract(delta,expected,0,0,1,0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(delta.norminf(0)==0.,"Private transverse restart field differs from retained physical high");}
    // Particle identity includes birth rank, which Redistribute preserves.
    // Changing box owners with the same rank numbering does not change it.
    WarpXParticleContainer::ParticleType::NextID(next_ids[amrex::ParallelDescriptor::MyProc()]);
    m_have_Eold=old;m_have_Eprev=previous;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_native_endpoint_restart_reconstructing&&!m_native_endpoint_restart_validated,
        "Accepted endpoint numeric reconstruction is once-only");
    struct ResetReconstruction {bool& active;~ResetReconstruction(){active=false;}} reset{m_native_endpoint_restart_reconstructing};
    m_native_endpoint_restart_reconstructing=true;
    m_native_endpoint_pair->ReadEndpointCheckpoint(root);
    m_native_endpoint_restart_validated=true;
    amrex::Print()<<"restart: retained endpoint, history and gather reconstructed from numeric fields without elliptic solve\n";
}
void warpx::darwin::RestoreNativeAcceptedEndpointCheckpoint(WarpX& simulation,std::string const& directory) {
    if(auto* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver()))
        solver->ReadAcceptedEndpointCheckpoint(directory);
}

bool warpx::darwin::NativeEndpointPairRestartContext(WarpX& simulation) noexcept {
    auto const* solver=dynamic_cast<ThetaImplicitHybrid const*>(simulation.get_pointer_ImplicitSolver());
    return solver&&solver->m_native_endpoint_pair_requested&&solver->m_native_endpoint_pair_accept_requested&&
        solver->m_native_endpoint_pair&&solver->m_hybrid_pic_model&&
        solver->m_hybrid_pic_model->m_darwin_checkpoint_restored&&
        (solver->m_native_endpoint_restart_reconstructing||solver->m_native_endpoint_restart_validated);
}
