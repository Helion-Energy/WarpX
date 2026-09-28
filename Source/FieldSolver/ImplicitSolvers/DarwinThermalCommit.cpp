#include <iomanip>
/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitHybrid.H"
#include "DarwinThermalAdvance.H"
#include "DarwinVacuumJointSolve.H"
#include "ThermalSourceAcceptance.H"
#include "NativeStoppingMaterialSource.H"
#include "MassMatrixDensityProjection.H"
#include "NativeVacuumJouleError.H"
#include "ImplicitParticleEndpointAudit.H"
#include "Particles/MultiParticleContainer.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include <ablastr/coarsen/sample.H>
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>

namespace warpx::thermal {
using Real = amrex::Real;
using Vec = WarpXSolverVec;
using warpx::fields::FieldType;
using ablastr::fields::Direction;

bool DarwinThermalAdvance::PrepareAcceptance ()
{
    AMREX_ALWAYS_ASSERT(m_open && !m_receipt_ready);
    auto const& energy=m_energy.getMultiFabBlock(energy_name,0);
    if (!AuditParticleEndpoints(&m_absorbed_particles,&m_endpoint_population,&m_reflected_particles) ||
        !m_stage->Admissible(energy)) { return false; }
    if (!PrepareIonElectricWorkReceipt()) { return false; }
    // No residual reevaluation here: its emitted source/flux arrays belong to
    // the accepted midpoint. Endpoint field updates must not change this heat.
    m_stage->Endpoint(energy,m_endpoint);
    m_balance=MeasureThermalEnergyBalance(*m_stage,energy);
    m_source_balance=m_sources ? m_sources->Ledger(m_nodal_rates,m_ion_rates,
        m_species ? static_cast<int>(m_species->Descriptors().size()) : 0) : ThermalSourceLedger{};
    // Inspect the converged, already reduced work ledger before exchange
    // preparation, receipt publication, endpoint particle updates or any kick.
    // Never cancel signed orphan cooling against orphan heating.
    amrex::Real verified_joule_error=0;
    if (NativeVacuumJouleErrorEnabled()) {
        if (!m_sources || !m_joule_work) { return false; }
        auto const numerical=PrepareNativeVacuumJouleError(m_simulation,*m_sources,
            *m_joule_work,m_source_balance,m_dt,m_solver.m_theta);
        if (!numerical.Ready()) { return false; }
        verified_joule_error=numerical.allowance;
    }
    auto const source_status=InspectThermalSourceAcceptance(m_source_balance,verified_joule_error);
    if (source_status!=ThermalSourceAcceptanceStatus::Ready) {
        amrex::Print()<<"Eulerian source acceptance rejected: status="<<static_cast<int>(source_status)
            <<" joule_orphan_signed="<<m_source_balance.energy[SourceComponent::DeclinedJouleWork]
            <<" joule_orphan_abs="<<m_source_balance.energy[SourceComponent::UnassignableJouleWorkAbs]
            <<" viscous_orphan_signed="<<m_source_balance.energy[SourceComponent::DeclinedViscousWork]
            <<" viscous_orphan_abs="<<m_source_balance.energy[SourceComponent::UnassignableViscousWorkAbs]
            <<" hyper_orphan_signed="<<m_source_balance.energy[SourceComponent::DeclinedHyperResistive]
            <<" hyper_orphan_abs="<<m_source_balance.energy[SourceComponent::UnassignableHyperResistiveAbs]
            <<" [J]; applied work requires an eligible thermal receiver\n";
        return false;
    }
    if (m_joule_work && !m_joule_work->PrepareEndReceipt(m_dt)) { return false; }
    if(m_expected_exchange) {
        if(!m_expected_exchange->Valid() || m_ion_exchange->Status()!=IonExchangeStatus::Prepared)return false;
        m_expected_ledger=m_expected_exchange->Ledger();
        m_exchange_moment_ledger=m_exchange_moments->Ledger();
        if(m_thermal_partner) {
            if(!m_thermal_partner->Valid())return false;
            m_population_thermal_energy=m_thermal_partner->ThermalEnergy();
            m_population_electron_energy=m_thermal_partner->ElectronEnergy();
            m_source_balance.energy[SourceComponent::ElectronTotal]+=
                m_population_electron_energy-m_source_balance.energy[SourceComponent::Relaxation];
            m_source_balance.energy[SourceComponent::Relaxation]=m_population_electron_energy;
        }
        amrex::MultiFab::Copy(m_expected_endpoint_work,m_expected_exchange->SpeciesWork(),0,0,m_expected_endpoint_work.nComp(),0);
    } else if (m_ion_exchange) {
        auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
        amrex::MultiFab half(rho,amrex::make_alias,rho.nComp()/2,1);
        auto const ji=m_simulation.m_fields.get_alldirs(FieldType::current_fp,0);
        auto const jp=m_simulation.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        // This shared native helper only forms V_e=-(Jp-Ji)/rho on nodes;
        // it does not run the legacy marker or thermal integrator.
        m_model.QDSMCInitializeUe(0,half,ji,jp,jp,1.);
        ThermalSourceState source;
        source.raw_charge=&half;
        source.temperature_kelvin=&m_stage_moments->NodalTemperature();
        source.species=m_species->Species();
        std::array<const amrex::MultiFab*,3> drift;
        for (int c=0;c<3;++c) {
            drift[c]=m_simulation.m_fields.get(FieldType::hybrid_electron_velocity_fp,Direction{c},0);
        }
        if (!m_ion_exchange->Prepare(source,m_ion_rates,drift)) { return false; }
    }
    if (!PrepareAbsorbingBoundary()) { return false; }
    m_receipt_ready=true;
    return true;
}

bool DarwinThermalAdvance::CommitIonExchange ()
{
    AMREX_ALWAYS_ASSERT(m_open && m_receipt_ready);
    if (!m_ion_exchange) { return true; }
    std::vector<ExpectedIonWorkLedger> relativistic_reference;
    if(m_expected_exchange) {
        // Verify the private endpoint map against the ACTUAL redistributed
        // endpoint before any accepted random draw. This is not a probe tally.
        if(!EvaluateNativeExpectedIonExchangeSource(*m_expected_exchange,*m_ion_exchange,
            m_simulation.GetPartContainer(),m_exchange_budget,
            ExpectedIonEndpointContract::CompleteRedistributedPreCollisionEndpoint))return false;
        auto const& actual=m_expected_exchange->SpeciesWork();
        amrex::MultiFab::Subtract(m_expected_endpoint_work,actual,0,0,actual.nComp(),0);
        amrex::Real error=0;
        for(int c=0;c<actual.nComp();++c)
            error=std::max(error,m_expected_endpoint_work.norminf(c)/std::max(amrex::Real(1.e-30),actual.norminf(c)));
        amrex::Print()<<"Expected OU endpoint work relative mismatch="<<error<<"\n";
        if(!std::isfinite(error) || error>1.e-11)return false;
    }
    if(m_relativistic_reference_audit) {
        // One optional acceptance-only reference separates the NR convention from
        // sampling. All residual/Jv evaluations retain the bounded NR path.
        // This is still pre-RNG and writes only private diagnostic scratch.
        amrex::Gpu::streamSynchronize();
        auto const start=amrex::second();
        auto options=m_expected_options;
        options.mode=ExpectedIonEnergyMode::RelativisticQuadrature;
        ExpectedIonExchangeSource reference(*m_ion_exchange,options);
        if(!EvaluateNativeExpectedIonExchangeSource(reference,*m_ion_exchange,
            m_simulation.GetPartContainer(),m_exchange_budget,
            ExpectedIonEndpointContract::CompleteRedistributedPreCollisionEndpoint))return false;
        relativistic_reference=reference.Ledger();
        auto seconds=amrex::second()-start;
        amrex::ParallelDescriptor::ReduceRealMax(seconds);
        amrex::Long count=0;
        for(auto const& view:m_expected_endpoints->Views())count+=view.count;
        amrex::ParallelDescriptor::ReduceLongSum(count);
        amrex::Print()<<"Expected OU relativistic reference cost: endpoint_particles="<<count
            <<" passes=1 seconds_max="<<seconds<<"\n";
    }
    if (!CommitNativeIonExchangeOnce(*m_ion_exchange,m_simulation.GetPartContainer())) { return false; }
    auto const& ledgers=m_ion_exchange->Ledger();
    for (std::size_t s=0;s<ledgers.size();++s) {
        auto const& l=ledgers[s];
        amrex::Print()<<"Eulerian accepted ion exchange [J]: species="<<m_species->Descriptors()[s].name
            <<" requested_relaxation="<<l.requested_relaxation
            <<" requested_redirect="<<l.requested_redirect
            <<" expected_relaxation="<<l.expected_relaxation
            <<" expected_redirect="<<l.expected_redirect
            <<" realized_total="<<l.realized_total
            <<" expectation_defect="<<l.expectation_defect
            <<" realization_defect="<<l.realization_defect<<"\n";
        if(m_relativistic_reference_audit) {
            auto const& r=relativistic_reference[s];using C=ExpectedIonWorkComponent;
            auto const mean=r[C::Relaxation]+r[C::Redirect];
            auto const nr=m_expected_ledger[s][C::NonrelativisticRelaxation]+
                m_expected_ledger[s][C::NonrelativisticRedirect];
            amrex::Print()<<"Expected OU accepted convention/sample [J]: species="<<m_species->Descriptors()[s].name
                <<" relativistic_expected="<<mean<<" nr_convention="<<nr-mean
                <<" measured_sampling_fluctuation="<<l.realized_total-mean
                <<" quadrature_estimate="<<r[C::QuadratureError]<<"\n";
        }
        if(m_expected_exchange) {
            auto const& w=m_expected_ledger[s];using C=ExpectedIonWorkComponent;
            auto const selected=w[C::Relaxation]+w[C::Redirect];
            amrex::Print()<<"Expected OU endpoint audit [J]: species="<<m_species->Descriptors()[s].name
                <<" relaxation="<<w[C::Relaxation]<<" redirect="<<w[C::Redirect]
                <<" nr_Ve_dot_deltaP="<<w[C::NonrelativisticDriftWork]
                <<" nr_relative_relaxation="<<w[C::NonrelativisticRelaxation]-w[C::NonrelativisticDriftWork]
                <<" nr_minus_selected="<<w[C::NonrelativisticRelaxation]+w[C::NonrelativisticRedirect]-selected
                <<" quadrature_estimate="<<w[C::QuadratureError]
                <<" convention_lower="<<w[C::ConventionLower]<<" convention_upper="<<w[C::ConventionUpper]
                <<" realized_minus_selected="<<l.realized_total-selected<<"\n";
            auto const& a=m_exchange_moment_ledger[s];using M=IonExchangeMomentComponent;
            amrex::Print()<<"Expected OU NR moment work [J]: species="<<m_species->Descriptors()[s].name
                <<" deterministic_bulk="<<a[M::DeterministicBulk]
                <<" thermal_relaxation="<<a[M::ThermalRelaxation]
                <<" thermal_minus_requested="<<a[M::ThermalRelaxation]-l.requested_relaxation
                <<" redirect="<<a[M::Redirect]
                <<" finite_sample_bulk_noise="<<a[M::FiniteSampleBulkNoise]
                <<" decomposition_defect="<<a[M::DeterministicBulk]+a[M::ThermalRelaxation]+a[M::Redirect]
                    -w[C::NonrelativisticRelaxation]-w[C::NonrelativisticRedirect]<<"\n";
        }
    }
    if(m_thermal_partner) {
        amrex::Print()<<"Expected OU population thermal partner [J]: electron_relaxation="
            <<m_population_electron_energy<<" ion_population_thermal="<<m_population_thermal_energy
            <<" paired_mean_defect="<<m_population_electron_energy+m_population_thermal_energy<<"; deterministic bulk/eta-electric work remain separate; "
            <<"finite-sample mean noise is a partition, not another debit\n";
    } else if(m_expected_exchange) {
        amrex::Print()<<"Expected OU audit: physical electron Q unchanged; no aggregate force/work closure claim\n";
    }
    return true;
}

void DarwinThermalAdvance::Commit ()
{
    PublishCandidate();
    FinalizeCandidate();
}

bool DarwinThermalAdvance::CanRetainAcceptance () const
{
    bool ready=m_open && m_receipt_ready && !m_particle_boundary_committed &&
        !m_absorbed_particles && !m_ion_exchange &&
        (!m_reflected_particles || (m_joint_vacuum && m_joint_vacuum->AxialReflectionPrepared()));
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);return ready;
}

void DarwinThermalAdvance::PublishCandidate ()
{
    AMREX_ALWAYS_ASSERT(m_open && m_receipt_ready && !m_candidate_published);
    auto& accepted=*m_simulation.m_fields.get(FieldType::hybrid_electron_energy_fp,0);
    amrex::MultiFab::Copy(accepted,m_absorbed_particles ? m_survivor_energy : m_endpoint,0,0,1,0);
    m_model.ClearElectronThermalTrials();
    m_model.RefreshEulerianElectronThermodynamics(0,
        m_absorbed_particles ? m_survivor_charge : m_endpoint_charge);
    m_candidate_published=true;
}

ThermalEnergyBalance const& DarwinThermalAdvance::CandidateEnergyBalance () const
{
    AMREX_ALWAYS_ASSERT(m_open && m_receipt_ready && m_candidate_published);
    return m_balance;
}

bool DarwinThermalAdvance::CaptureMaterialFieldWork(
    NativeEndpointAmpereOrigin const& origin,NativeStoppingFieldWork& output,
    ablastr::fields::ConstVectorField const* post_source_initial) const
{
    bool ready=m_open&&m_receipt_ready&&m_candidate_published&&m_joint_vacuum&&
        m_ion_electric_work_enabled&&m_ion_electric_work_ready&&m_ion_work_receipt.valid&&
        m_ion_work_receipt.has_total_field&&m_ion_work_receipt.has_remainder_field&&
        m_ion_work_receipt.gather_contract==IonElectricGatherContract::RecordedFinalGather&&
        !m_absorbed_particles&&!m_reflected_particles&&!m_ion_exchange;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    auto const epoch=std::uint64_t(m_simulation.getistep(0))+1;
    if(!origin.Matches(m_simulation,m_time+m_dt,epoch)||!CanCancel())return false;
    ablastr::fields::ConstVectorField initial{};
    for(int c=0;c<3;++c){auto const* field=m_simulation.m_fields.get(FieldType::Bfield_fp,Direction{c},0);
        initial[c]=post_source_initial?(*post_source_initial)[c]:m_field_rollback.SavedField(field);
        ready=bool(initial[c])&&ready;}
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    NativeStoppingFieldWork result;result.time=m_time+m_dt;result.interval=m_dt;result.epoch=epoch;
    result.thermal=m_balance;result.ions=m_ion_work_receipt;
    result.magnetic=origin.FieldMagneticWork(m_simulation,initial);
    if(!result.magnetic.valid)return false;
    result.available=true;output=std::move(result);return true;
}

void DarwinThermalAdvance::FinalizeCandidate ()
{
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    AMREX_ALWAYS_ASSERT(m_open && m_receipt_ready && m_candidate_published);
    amrex::Print()<<"Eulerian accepted energy [J]: old="<<m_balance.initial
        <<" new="<<m_balance.final-m_absorbing_inventory.represented_joule
        <<" trajectory_endpoint="<<m_balance.final
        <<" absorption="<<-m_absorbing_inventory.represented_joule<<" source="<<m_balance.source
        <<" conduction="<<m_balance.conduction<<" advection="<<m_balance.advection
        <<" compression="<<m_balance.compression<<" defect="<<m_balance.defect
        <<" absolute_defect="<<m_balance.absolute_defect<<"\n";
    if (m_absorbed_particles) {
        amrex::Print()<<"Eulerian accepted absorption [J]: raw_requested="<<m_absorbing_inventory.raw_requested_joule
            <<" represented="<<m_absorbing_inventory.represented_joule
            <<" floor_replacement="<<m_absorbing_inventory.floor_replacement_joule
            <<" cell_T_relative_error="<<m_absorbing_inventory.max_temperature_relative_error<<"\n";
    }
    PrintIonElectricWorkReceipt();
    if(m_solver.m_native_paired_fields) {
        auto const& p=m_solver.m_native_paired_fields->LastPublication();
        AMREX_ALWAYS_ASSERT(p.complete);
        amrex::Print()<<std::setprecision(17)<<"Native paired publication [J]: magnetic="<<p.magnetic_inventory_change
            <<" longitudinal="<<p.longitudinal_inventory_change
            <<" held_Je="<<p.held_electron_inventory_change
            <<" endpoint_ion_quadrature="<<p.endpoint_ion_quadrature_change
            <<" endpoint_electron_quadrature="<<p.endpoint_electron_quadrature_change
            <<" endpoint_current_quadrature="<<p.endpoint_current_quadrature_change
            <<" stage_ion_terminal_rounding="<<p.stage_ion_terminal_rounding_work
            <<" stage_electron_terminal_rounding="<<p.stage_electron_terminal_rounding_work
            <<" stage_inertia_terminal_rounding="<<p.stage_inertia_terminal_rounding_work
            <<" stage_assembly_mismatch_V_per_m="<<p.stage_assembly_mismatch<<"\n";
    }
    if (m_joule_work) { m_joule_work->CommitEndReceipt(); }
    // Publish only the accepted receipt, once. Source evaluation/Jv never
    // accumulates or reports these channels.
    for (int c=0;c<SourceComponent::Count;++c) {
        if (!m_joule_work && c>=SourceComponent::DeclinedJouleWork) { continue; }
        amrex::Print()<<"Eulerian accepted source [J]: channel="<<ThermalSourceChannelNames[c]
            <<" energy="<<m_source_balance.energy[c]<<"\n";
    }
    for (std::size_t s=0;s<m_source_balance.ion_relaxation.size();++s) {
        amrex::Print()<<"Eulerian accepted requested ion source [J]: species="
            <<m_species->Descriptors()[s].name
            <<" relaxation="<<m_source_balance.ion_relaxation[s]
            <<" redirect="<<m_source_balance.ion_redirect[s]<<"\n";
    }
    if (m_conduction_activity) {
        auto const activity=m_conduction_activity->Measure();
        amrex::Print()<<"Eulerian conduction activity [m^3]: native_open="<<activity.native_open_volume
            <<" retained="<<activity.retained_open_volume<<" mixed="<<activity.mixed_cell_volume
            <<" lost="<<activity.lost_open_volume<<" capacity_gap="<<activity.max_capacity_relative_gap
            <<" active_capacity_gap="<<activity.max_active_capacity_relative_gap<<"\n";
    }
    amrex::Print()<<"Eulerian source PC: unavailable_diagonal_evaluations="<<m_source_diagonal_fallbacks<<"\n";
    if(m_stopping_carry){m_stopping_carry->Invalidate();m_stopping_carry.reset();}
    if(m_joint_vacuum)m_joint_vacuum->Invalidate();
    m_field_rollback.Discard();
    m_open=false; m_receipt_ready=false; m_candidate_published=false;
    m_model.EndDensityControlStep();
}
void DarwinThermalAdvance::PublishEndpointDensity ()
{
    // Keep rho_half intact until all accepted midpoint/history consumers finish.
    // A loss has a separately validated survivor image and a named electron
    // inventory debit; the trajectory density is retained until this point.
    auto& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    amrex::MultiFab::Copy(rho,m_absorbed_particles ? m_survivor_charge : m_endpoint_charge,
        0,0,1,rho.nGrowVect());
    if (m_absorbed_particles) { RefreshAbsorbingLongitudinalField(); }
}
void DarwinThermalAdvance::InvalidateBorrowedStepViews () noexcept
{
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    if(m_stopping_carry){m_stopping_carry->Invalidate();m_stopping_carry.reset();}
    if(m_joint_vacuum)m_joint_vacuum->Invalidate();
    // Called by whole-map Restore before particles are swapped. No collectives
    // or accepted-field writes belong here. Each next BeginStep rebuilds trial
    // closures; no source/endpoint borrowed particle view survives rejection.
    m_thermal_partner.reset(); m_exchange_moments.reset();
    m_expected_endpoints.reset(); m_expected_exchange.reset(); m_ion_exchange.reset();
    m_source_diagonal.reset(); m_sources.reset(); m_pc.reset(); m_stage.reset();
    m_ion_electric_work_ready=false; m_ion_work_base_ready=false;
    m_field_pc = JacobianFunctionMF<WarpXSolverVec, ImplicitSolver>{};
    if (m_pressure_pc) { m_pressure_pc->Invalidate(); }
    for (auto& density : m_solver.m_mass_matrix_density) { density.reset(); }
    m_solver.m_inertia_beta.clear(); m_solver.m_inertia_rhs_scale.clear();
    m_solver.m_inertia_beta_owned.clear(); m_solver.m_inertia_rhs_scale_owned.clear();
    // Transformed/tensor electric closures are excluded from retained scope.
    // Their inactive cached coefficient views must nevertheless not be reused.
    if(m_solver.m_native_paired_fields)m_solver.m_native_paired_fields->InvalidateBorrowedViews();
    if(m_solver.m_native_endpoint_pair)m_solver.m_native_endpoint_pair->InvalidateBorrowedViews();
    m_model.m_tensor_rho_captured=false;
    m_model.m_tensor_rho_old.reset();
    for (auto& current : m_model.m_tensor_jp_old) { current.reset(); }
    m_solver.InvalidateMassMatrices();
}

bool DarwinThermalAdvance::CanCancel () const
{
    // CanRestore is collective even when a rank has already crossed a local
    // boundary barrier. Never short-circuit it on a rank-local predicate.
    bool const fields=m_field_rollback.CanRestore(m_simulation.m_fields);
    bool can_cancel=!m_particle_boundary_committed && fields;
    amrex::ParallelDescriptor::ReduceBoolAnd(can_cancel);
    return can_cancel;
}
void DarwinThermalAdvance::Cancel ()
{
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    if (m_joule_work) { m_joule_work->DiscardEndReceipt(); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_particle_boundary_committed,
        "Native boundary tallies/buffers have crossed acceptance; this step cannot be retried");
    m_model.ClearElectronThermalTrials();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_field_rollback.Restore(m_simulation.m_fields),
        "Implicit field topology changed inside a rejectable thermal step");
    CopyFields(m_solver.m_E,m_saved_field);
    CopyFields(m_solver.m_Eold,m_saved_field_old);
    CopyFields(m_solver.m_Eprev,m_saved_field_prev);
    if(m_solver.m_native_paired_fields)m_solver.m_native_paired_fields->RestoreState(m_saved_paired_fields);
    if(m_solver.m_native_endpoint_pair)m_solver.m_native_endpoint_pair->RestoreState(m_saved_endpoint_pair);
    m_solver.m_dt=m_saved_flags.dt;
    m_solver.m_have_Eold=m_saved_flags.have_old;
    m_solver.m_have_Eprev=m_saved_flags.have_prev;
    m_model.m_inertia_rho_n_captured=m_saved_flags.rho_captured;
    m_model.m_inertia_jpold_captured=m_saved_flags.jp_captured;
    m_model.m_qdsmc_J_plasma_valid=m_saved_flags.plasma_valid;
    m_solver.m_native_circuit_step_open=m_saved_flags.native_circuit_open;
    m_model.m_inertia_history_initialized=m_saved_flags.history_initialized;
    m_model.m_inertia_history_levels=m_saved_flags.history_levels;
    // The PC stamp is a monotone derived-cache generation, not checkpointed
    // history. Do not rewind it and accidentally revive candidate factors.
    m_solver.m_darwin_circuit_accepted_scales=m_saved_circuit_scales;
    m_solver.m_fext_init=m_saved_external_initial;
    m_field_rollback.Discard();
    m_solver.InvalidateMassMatrices();
    m_open=false; m_receipt_ready=false; m_candidate_published=false;
    m_model.EndDensityControlStep();
}

} // namespace warpx::thermal
