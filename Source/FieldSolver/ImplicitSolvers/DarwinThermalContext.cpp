/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitHybrid.H"
#include "DarwinThermalAdvance.H"
#include "ThermalCurrentRemainder.H"
#include "DarwinVacuumJointSolve.H"
#include "ThermalSourceAcceptance.H"
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

void DarwinThermalAdvance::CopyFields (Vec& dst,const Vec& src) const
{
    for (int d=0;d<3;++d) {
        amrex::MultiFab::Copy(*dst.getArrayVec()[0][d],*src.getArrayVec()[0][d],0,0,1,0);
    }
}

void DarwinThermalAdvance::SnapshotStepStart ()
{
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    AMREX_ALWAYS_ASSERT(!m_open && !m_field_rollback.Open());
    // Esirkepov trajectory support can grow after the first nonlinear push.
    // ResizeEsirkepovMassMatrices then rebuilds these tangent arrays in place.
    // They are derived scratch, not accepted physical fields. Keeping their
    // old storage identity in the transaction would prevent a failed step from
    // restoring E/B/rho/U and particles. Cancel invalidates the tangent before
    // any retry; every other registered field retains the strict topology check.
    std::vector<amrex::MultiFab const*> tangent_scratch;
    for (int lev=0;lev<=m_simulation.finestLevel();++lev) {
        for (auto type : {FieldType::MassMatrices_X, FieldType::MassMatrices_Y,
                          FieldType::MassMatrices_Z}) {
            for (int d=0;d<3;++d) {
                if (m_simulation.m_fields.has(type,Direction{d},lev)) {
                    tangent_scratch.push_back(m_simulation.m_fields.get(type,Direction{d},lev));
                }
            }
        }
    }
    m_field_rollback.Capture(m_simulation.m_fields,tangent_scratch);
    CopyFields(m_saved_field,m_solver.m_E);
    CopyFields(m_saved_field_old,m_solver.m_Eold);
    CopyFields(m_saved_field_prev,m_solver.m_Eprev);
    m_saved_flags={m_solver.m_dt,m_solver.m_have_Eold,m_solver.m_have_Eprev,
        m_model.m_inertia_rho_n_captured,m_model.m_inertia_jpold_captured,
        m_model.m_qdsmc_J_plasma_valid,m_solver.m_native_circuit_step_open,
        m_model.m_inertia_history_initialized,m_model.m_inertia_history_levels};
    if(m_solver.m_native_paired_fields)m_saved_paired_fields=m_solver.m_native_paired_fields->State();
    if(m_solver.m_native_endpoint_pair){
        m_saved_endpoint_pair=m_solver.m_native_endpoint_pair->State();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_solver.m_native_endpoint_pair->BeginEndpointStep(
            m_simulation.gett_new(0),static_cast<std::uint64_t>(m_simulation.getistep(0))),
            "Retained endpoint origin is not valid at the original step snapshot");
    }
    m_saved_circuit_scales=m_solver.m_darwin_circuit_accepted_scales;
    m_saved_external_initial=m_solver.m_fext_init;
}

void DarwinThermalAdvance::BeginStep (Real time,Real dt)
{
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    m_joint_vacuum.reset();
    AMREX_ALWAYS_ASSERT(!m_open);
    if(m_density_epoch!=m_model.DensityControlEpoch()) {
        RefreshDensityDependentContext();
        m_solver.InvalidateMassMatrices();
    }
    m_model.BeginDensityControlStep();
    m_time=time; m_dt=dt; m_energy_reference=-1; m_receipt_ready=false;
    m_candidate_published=false;
    m_absorbed_particles=0; m_reflected_particles=0; m_endpoint_population=0; m_particle_boundary_committed=false;
    m_axial_boundary_completed=false;
    m_absorbed_charge_integral=0; m_absorbing_inventory={}; m_boundary_tallies_before.clear();
    m_source_diagonal_fallbacks=0;
    m_ion_electric_work_ready=false; m_ion_work_base_ready=false; m_ion_work_receipt={};
    m_thermal_partner.reset(); m_population_thermal_energy=0; m_population_electron_energy=0;
    m_exchange_moments.reset(); m_expected_endpoints.reset(); m_expected_exchange.reset();
    m_ion_exchange.reset(); m_expected_ledger.clear();
    if (m_model.m_include_temperature_relaxation ||
        (m_model.m_include_joule_heating && m_model.m_joule_redirect_to_ions)) {
        AMREX_ALWAYS_ASSERT(m_species);
        IonExchangeOptions exchange;
        exchange.dt=dt; exchange.time=time+m_solver.m_theta*dt;
        exchange.raw_density_floor=m_model.m_n_floor;
        exchange.relaxation=m_model.m_include_temperature_relaxation;
        exchange.redirect=m_model.m_include_joule_heating && m_model.m_joule_redirect_to_ions;
        exchange.redirect_kick_cap=exchange.redirect && m_model.m_joule_redirect_kick_cap_vth_frac>=0;
        // This implicit lane historically never calls the explicit temperature
        // shunt. A configured threshold is not an active exchange request.
        exchange.temperature_shunt=false;
        auto const& u=m_energy.getMultiFabBlock(energy_name,0);
        m_ion_exchange=std::make_unique<AcceptedIonExchange>(m_geometry,u.boxArray(),
            u.DistributionMap(),m_species->Descriptors(),exchange);
        if(m_expected_ou_audit) {
            m_expected_exchange=std::make_unique<ExpectedIonExchangeSource>(*m_ion_exchange,m_expected_options);
            m_expected_endpoints=std::make_unique<ImplicitIonEndpointTrial>(*m_ion_exchange);
            m_exchange_moments=std::make_unique<IonExchangeMomentAudit>(*m_ion_exchange);
            if(m_population_thermal_partner) {
                m_thermal_partner=std::make_unique<ExpectedIonThermalPartner>(*m_ion_exchange);
                if(!m_legacy_relaxation.isDefined())m_legacy_relaxation.define(u.boxArray(),u.DistributionMap(),1,0);
            }
            if(!m_exchange_budget.isDefined())m_exchange_budget.define(u.boxArray(),u.DistributionMap(),1,0);
            if(!m_expected_endpoint_work.isDefined())m_expected_endpoint_work.define(u.boxArray(),u.DistributionMap(),
                static_cast<int>(m_species->Descriptors().size())*ExpectedIonWorkComponent::Count,0);
            for(auto& v:m_exchange_drift)if(!v)v=std::make_unique<amrex::MultiFab>(m_old_charge.boxArray(),m_old_charge.DistributionMap(),1,1);
        }
    }
    if (m_species && m_model.m_include_temperature_relaxation) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(FreezeNativeOldSpeciesTemperature(*m_species,
            m_simulation.GetPartContainer()),"Invalid accepted native old ion temperature");
    }
    m_energy.CopyMultiFabBlocksFromFields();
    amrex::MultiFab::Copy(m_old_energy,m_energy.getMultiFabBlock(energy_name,0),0,0,1,0);
    auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    amrex::MultiFab::Copy(m_old_charge,rho,0,0,1,rho.nGrowVect());
    for (int d=0;d<3;++d) {
        auto const& j=*m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        amrex::MultiFab::Copy(*m_old_current[d],j,0,0,1,j.nGrowVect());
    }
    KineticThermalStateView old{m_old_charge};
    old.pedestal=m_model.DensityPedestal(0); old.energy=&m_old_energy;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_old_moments->Evaluate(old),"Invalid accepted Eulerian thermal state");
    m_source.setVal(0);
    CellMagneticField();
    m_stage_options.dt=dt; m_stage_options.time=time+m_solver.m_theta*dt;
    m_stage_options.lifetime=m_stage_lifetime;
    m_stage=std::make_unique<EulerianThermalStage>(m_geometry,m_old_energy,
        m_old_moments->NumberDensity(),m_old_moments->NumberDensity(),
        m_old_moments->NumberDensity(),m_magnetic,m_source,ThermalFaceContext{},m_stage_options);
    ThermalSourceOptions sources;
    sources.time=m_stage_options.time; sources.physical_dt=dt;
    sources.density_floor=m_model.m_n_floor;
    sources.joule=m_model.m_include_joule_heating;
    sources.joule_applied_edge_work=bool(m_joule_work);
    sources.relaxation=m_model.m_include_temperature_relaxation;
    sources.field_eta_uses_kelvin=m_model.m_resistivity_has_Te_dependence;
    sources.use_heating_eta=m_model.m_has_heating_resistivity;
    sources.field_eta=m_model.m_eta;
    sources.field_eta_kelvin=m_model.m_eta_te;
    sources.heating_eta=m_model.m_eta_heating;
    sources.relaxation_rate=m_model.m_nu_ei;
    sources.joule_density_gate=std::max(m_model.m_joule_heating_n_min,m_model.m_n_floor);
    sources.joule_taper=m_model.m_joule_heating_taper;
    sources.redirect_joule=m_model.m_joule_redirect_to_ions;
    sources.redirect_temperature_ev=m_model.m_joule_redirect_to_ions ? m_model.m_joule_redirect_Te_eV : Real(0);
    sources.redirect_density_floor_factor=m_model.m_joule_redirect_n_min_factor;
    sources.pedestal_temperature_cap=m_model.m_qdsmc_te_pedestal_cap_eV>=0 || m_model.m_qdsmc_te_pedestal_image;
    sources.redirect_kick_cap=m_model.m_joule_redirect_kick_cap_vth_frac>=0;
    sources.external_sink=m_model.m_has_energy_sink; sources.sink=m_model.m_energy_sink;
    sources.hyperresistive_work=m_model.m_hyper_res_heating;
    if (m_model.m_include_electron_viscosity) {
        sources.viscosity=m_model.m_visc_heating_work ? ViscousSourceKind::AppliedWork : ViscousSourceKind::Strain;
    }
    sources.source_taper_density=std::max(Real(0),m_model.m_qdsmc_source_taper_n);
#if defined(WARPX_DIM_RZ)
    m_sources=std::make_unique<EulerianThermalSources>(m_geometry,m_old_charge,sources);
    if (m_source_diagonal_enabled && (sources.joule || sources.relaxation)) {
        m_source_diagonal=std::make_unique<EulerianSourceDiagonal>(m_geometry,m_old_charge,sources);
    } else { m_source_diagonal.reset(); }
#else
    // Cartesian transport qualification starts with the source-free system.
    // Pressure and Te below still refresh on every residual and Jv probe.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!sources.joule && !sources.relaxation &&
        !sources.external_sink && !sources.hyperresistive_work &&
        sources.viscosity==ViscousSourceKind::None && !m_dissipation,
        "Cartesian implicit energy sources require their native source adapter");
    m_sources.reset(); m_source_diagonal.reset();
#endif
    m_stage->SetSourceEvaluator([this](const amrex::MultiFab& u,const amrex::MultiFab&,
                                     const amrex::MultiFab&,amrex::MultiFab& q,amrex::MultiFab& dq) {
        return Source(u,q,dq);
    }, !sources.joule && !sources.relaxation && !sources.external_sink &&
       !sources.hyperresistive_work && sources.viscosity==ViscousSourceKind::None && !m_dissipation);
    m_pc=std::make_unique<ThermalConductionPC>(*m_stage,*m_options.preconditioner);
    if (!m_field_pc.isDefined()) {
        m_field_pc.define(m_field, &m_solver, m_solver.m_nlsolver->GetPreconditionerType());
    }
    m_field_pc.curTime(m_stage_options.time); m_field_pc.curTimeStep(dt);
    m_open=true;
}

void DarwinThermalAdvance::InitializePushContext ()
{
    // Every full residual and every signed Jv probe starts at the same accepted
    // physical state. Previous residual deposits must never seed this map.
    amrex::MultiFab::Copy(m_push_charge,m_old_charge,0,0,1,m_old_charge.nGrowVect());
    for (int d=0;d<3;++d) {
        auto& j=*m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        amrex::MultiFab::Copy(j,*m_old_current[d],0,0,1,j.nGrowVect());
        amrex::MultiFab::Copy(*m_push_current[d],j,0,0,1,j.nGrowVect());
    }
}

bool DarwinThermalAdvance::SetPushThermodynamics ()
{
    KineticThermalStateView state{m_push_charge};
    state.pedestal=m_model.DensityPedestal(0);
    state.energy=&m_energy.getMultiFabBlock(energy_name,0);
    if (!m_stage_moments->Evaluate(state)) { return false; }
    m_model.SetElectronThermalTrial(0,m_stage_moments->NodalTemperature(),m_stage_moments->OhmPressure());
    return true;
}

bool DarwinThermalAdvance::PushContextConverged ()
{
    auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    amrex::MultiFab::LinComb(m_push_difference,1.,rho,rho.nComp()/2,-1.,m_push_charge,0,0,1,0);
    Real defect=m_push_difference.norminf()/std::max({rho.norminf(rho.nComp()/2),
        m_push_charge.norminf(),PhysConst::q_e*m_model.m_n_floor});
    amrex::MultiFab::Copy(m_push_charge,rho,rho.nComp()/2,0,1,m_push_charge.nGrowVect());
    // Current is one vector block. Normalizing each component separately
    // forces a physically zero component to converge relative to roundoff
    // (for example, azimuthal cancellation in an axisymmetric axial flow).
    // Keep charge on its own dimensional scale and use a common current norm.
    amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax> op;
    amrex::ReduceData<Real,Real,Real> data(op);
    using Tuple=decltype(data)::Type;
    for (int d=0;d<3;++d) {
        auto const& now=*m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        auto const& old=*m_push_current[d];
        for (amrex::MFIter mfi(now);mfi.isValid();++mfi) {
            auto const a=now.const_array(mfi), b=old.const_array(mfi);
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) -> Tuple {
                Real const x=a(i,j,k), y=b(i,j,k), change=x-y;
                // A maximum reduction must not hide a NaN in one component.
                bool const valid=std::isfinite(x) && std::isfinite(y) && std::isfinite(change);
                return valid ? Tuple{amrex::max(std::abs(x),std::abs(y)),std::abs(change),0.}
                             : Tuple{0.,0.,1.};
            });
        }
    }
    auto const reduced=data.value();
    std::array<Real,3> norms{amrex::get<0>(reduced),amrex::get<1>(reduced),amrex::get<2>(reduced)};
    amrex::ParallelDescriptor::ReduceRealMax(norms.data(),int(norms.size()));
    if (norms[2]>0.) { return false; }
    Real const current_reference=std::max(Real(1.e-30),norms[0]);
    Real const current_difference=std::max(Real(0.),norms[1]);
    for (int d=0;d<3;++d) {
        auto const& now=*m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        auto& old=*m_push_current[d];
        amrex::MultiFab::Copy(old,now,0,0,1,old.nGrowVect());
    }
    defect=std::max(defect,current_difference/current_reference);
    return std::isfinite(defect) && defect<=m_push_tolerance;
}

void DarwinThermalAdvance::CellMagneticField ()
{
    for (int c=0;c<3;++c) {
        auto const& b=*m_simulation.m_fields.get(FieldType::Bfield_fp,Direction{c},0);
        auto const type=b.ixType().toIntVect();
        for (amrex::MFIter mfi(m_magnetic);mfi.isValid();++mfi) {
            auto const in=b.const_array(mfi); auto const out=m_magnetic.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real sum=0;
                for (int bits=0;bits<(1<<AMREX_SPACEDIM);++bits) {
                    int s[3]={i,j,k};
                    for (int d=0;d<AMREX_SPACEDIM;++d) { s[d]+=type[d]*((bits>>d)&1); }
                    sum+=in(s[0],s[1],s[2]);
                }
                out(i,j,k,c)=sum/Real(1<<AMREX_SPACEDIM);
            });
        }
    }
}

bool DarwinThermalAdvance::RefreshPressure ()
{
    auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    KineticThermalStateView state{rho}; state.charge_component=rho.nComp()/2;
    state.pedestal=m_model.DensityPedestal(0);
    state.energy=&m_energy.getMultiFabBlock(energy_name,0);
    state.current_input=ThermalCurrentInput::TotalPlasma;
    for (int d=0;d<3;++d) {
        state.ion_current[d]=m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        state.current[d]=m_simulation.m_fields.get(FieldType::hybrid_current_fp_plasma,Direction{d},0);
        if(m_current_remainder_requested && m_current_remainder_ready!=0 &&
           m_current_remainder_ready==m_current_remainder_generation)
            state.current_remainder[d]=m_current_remainder[d].get();
    }
    if (!m_stage_moments->Evaluate(state)) { return false; }
    m_model.SetElectronThermalTrial(0,m_stage_moments->NodalTemperature(),m_stage_moments->OhmPressure());
    return true;
}

bool DarwinThermalAdvance::AuditParticleEndpoints (amrex::Long* absorbed, amrex::Long* total,
                                                   amrex::Long* reflected) const
{
    warpx::implicit::EndpointAuditFields fields;
    fields.allow_deterministic_absorption=true;
    fields.allow_radial_reflection=true;
    fields.allow_axial_reflection=m_joint_vacuum && m_joint_vacuum->AllowsAxialReflection();
    fields.endpoint_density=&m_endpoint_charge;
    // Single-level MC UpdateAuxiliaryData centers the entire auxiliary fab
    // from already-filled source fields; FillBoundaryAux skips the finest
    // level. Qualify only the native gather band, not merely allocated guards.
    fields.gather_filled_ghosts=m_simulation.get_ng_fieldgather();
    fields.current_deposit_ghosts=m_simulation.get_ng_depos_J();
    for (int d=0;d<3;++d) {
        fields.current[d]=m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
        fields.gather_e[d]=m_simulation.m_fields.get(FieldType::Efield_aux,Direction{d},0);
        fields.gather_b[d]=m_simulation.m_fields.get(FieldType::Bfield_aux,Direction{d},0);
    }
    auto const audit=warpx::implicit::AuditImplicitParticleEndpoints(m_simulation,0,fields);
    if (absorbed) { *absorbed=audit.count(warpx::implicit::EndpointIssue::SupportedAbsorption); }
    if (total) { *total=audit.particles; }
    if (reflected) { *reflected=audit.count(warpx::implicit::EndpointIssue::SupportedReflection); }
    if (!audit.ok()) { amrex::Print()<<"Eulerian particle endpoint rejected: "<<audit.summary()<<"\n"; }
    return audit.ok();
}

bool DarwinThermalAdvance::RefreshSpeciesContext ()
{
    if (!m_species) { return true; }
    return RefreshNativeThermalSpecies(*m_species,m_simulation.GetPartContainer(),
        m_simulation.getdt(0),MaterializedSpeciesTrial::FullParticleState);
}

bool DarwinThermalAdvance::RefreshContext ()
{
    if(m_current_remainder_requested && !remainder::All(remainder::ArithmeticSupported() && m_current_remainder_ready!=0 &&
        m_current_remainder_ready==m_current_remainder_generation))return false;
    if (!RefreshPressure()) { return false; }
    auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    amrex::MultiFab::LinComb(m_endpoint_charge,2.,rho,rho.nComp()/2,-1.,m_old_charge,0,0,1,m_endpoint_charge.nGrowVect());
    KineticThermalStateView endpoint{m_endpoint_charge}; endpoint.pedestal=m_model.DensityPedestal(0);
    if (!m_endpoint_moments->Evaluate(endpoint)) { return false; }
    CellMagneticField();
    ThermalFaceContext faces;
    if (m_conduction_activity) {
        if (!m_conduction_activity->Evaluate(rho,rho.nComp()/2,m_model.DensityPedestal(0))) {
            return false;
        }
        faces.conduction_active=&m_conduction_activity->CellOpen();
    }
    for (int d=0;d<AMREX_SPACEDIM;++d) {
        amrex::MultiFab::Copy(*m_number_flux[d],m_stage_moments->FaceElectronCurrent(d),0,0,1,0);
        m_number_flux[d]->mult(-1./PhysConst::q_e,0,1,0);
        faces.number_flux[d]=m_number_flux[d].get();
        if(m_current_remainder_requested) {
            if(!m_stage_moments->HasCurrentRemainder())return false;
            if(!m_number_flux_remainder[d])m_number_flux_remainder[d]=std::make_unique<amrex::MultiFab>(
                m_number_flux[d]->boxArray(),m_number_flux[d]->DistributionMap(),1,0);
            Real const factor=-1./PhysConst::q_e;
            for(amrex::MFIter mfi(*m_number_flux_remainder[d]);mfi.isValid();++mfi) {
                auto const j=m_stage_moments->FaceElectronCurrent(d).const_array(mfi);
                auto const l=m_stage_moments->FaceElectronCurrentRemainder(d).const_array(mfi);
                auto const h=m_number_flux[d]->const_array(mfi);
                auto const o=m_number_flux_remainder[d]->array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j0,int k) {
                    o(i,j0,k)=remainder::WithHigh(remainder::Multiply({j(i,j0,k),l(i,j0,k)},factor),h(i,j0,k)).lo;
                });
            }
            faces.number_flux_remainder[d]=m_number_flux_remainder[d].get();
        }
    }
    return m_stage->RefreshContext(m_stage_moments->NumberDensity(),m_endpoint_moments->NumberDensity(),
                                    m_magnetic,m_source,faces);
}

bool DarwinThermalAdvance::Source (const amrex::MultiFab& u,amrex::MultiFab& q,amrex::MultiFab& dq)
{
    // Fixed kinetic context in a scalar subsolve; in the combined equation
    // the field residual immediately preceding this call refreshes it.
    auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    KineticThermalStateView state{rho}; state.charge_component=rho.nComp()/2;
    state.pedestal=m_model.DensityPedestal(0); state.energy=&u;
    if (!m_stage_moments->Evaluate(state)) { return false; }
    m_model.SetElectronThermalTrial(0,m_stage_moments->NodalTemperature(),m_stage_moments->OhmPressure());
    dq.setVal(0.);
    if (!m_sources) {
        q.setVal(0.); m_nodal_rates.setVal(0.); m_ion_rates.setVal(0.);
        return true;
    }
    amrex::MultiFab rho_alias(rho,amrex::make_alias,rho.nComp()/2,1);
    ThermalSourceState source;
    if (m_species) { source.species=m_species->Species(); }
    source.additive_resistivity=&m_additive_eta;
    source.raw_charge=&rho_alias; source.temperature_kelvin=&m_stage_moments->NodalTemperature();
    for (int d=0;d<3;++d) {
        source.plasma_current[d]=m_simulation.m_fields.get(FieldType::hybrid_current_fp_plasma,Direction{d},0);
        source.magnetic_field[d]=m_simulation.m_fields.get(FieldType::Bfield_fp,Direction{d},0);
    }
    if (m_dissipation) {
        TrialDissipationState trial;
        trial.charge=&rho_alias; trial.temperature_kelvin=source.temperature_kelvin;
        for (int c=0;c<3;++c) {
            trial.plasma_current[c]=source.plasma_current[c];
            trial.magnetic_field[c]=source.magnetic_field[c];
            trial.ion_current[c]=m_simulation.m_fields.get(FieldType::current_fp,Direction{c},0);
        }
        if (!m_dissipation->Evaluate(trial)) { return false; }
        source.viscous_electric=m_dissipation->ViscousField();
        source.hyper_electric=m_dissipation->HyperField();
        source.viscous_strain_power=&m_dissipation->StrainPower();
    }
    if (m_joule_work) {
        if (!m_joule_work->Evaluate(m_simulation,rho_alias,m_dissipation.get())) { return false; }
        source.joule_electric=m_joule_work->Electric();
    }
    if (!m_sources->Evaluate(source,m_nodal_rates,m_ion_rates)) { return false; }
    if(m_expected_exchange) {
        std::array<const amrex::MultiFab*,3> ji,drift;
        std::array<amrex::MultiFab*,3> output;
        for(int d=0;d<3;++d){
            ji[d]=m_simulation.m_fields.get(FieldType::current_fp,Direction{d},0);
            output[d]=m_exchange_drift[d].get();drift[d]=output[d];
        }
        if(!BuildExpectedIonDrift(m_geometry,rho_alias,m_model.DensityPedestal(0),ji,
            source.plasma_current,m_model.m_n_floor,output) ||
            !m_ion_exchange->Prepare(source,m_ion_rates,drift) ||
            !RefreshNativeImplicitIonEndpointTrial(*m_expected_endpoints,m_simulation.GetPartContainer()))return false;
        // Zero is an explicitly unused diagnostic budget: Source() below is
        // NEVER deposited. Only private expected ion work is evaluated.
        m_exchange_budget.setVal(0.);
        if(!m_expected_exchange->Evaluate(*m_ion_exchange,m_expected_endpoints->Views(),m_exchange_budget,
            ExpectedIonEndpointContract::CompleteRedistributedPreCollisionEndpoint) ||
            !m_exchange_moments->Evaluate(*m_expected_endpoints,*m_ion_exchange))return false;
    }
    m_stage_moments->RestrictNodalScalar(m_nodal_rates,SourceComponent::ElectronTotal,q);
    if(m_thermal_partner) {
        if(!m_thermal_partner->Evaluate(*m_exchange_moments,*m_expected_exchange))return false;
        m_stage_moments->RestrictNodalScalar(m_nodal_rates,SourceComponent::Relaxation,m_legacy_relaxation);
        amrex::MultiFab::Subtract(q,m_legacy_relaxation,0,0,1,0);
        amrex::MultiFab::Add(q,m_thermal_partner->Source(),0,0,1,0);
    }
    // The pure audit leaves Q unchanged. The opt-in partner replaces ONLY
    // relaxation; this legacy damping diagonal is an explicit PC surrogate,
    // frozen independently of the physical full residual/Jv.
    if (m_source_diagonal) {
        if (m_source_diagonal->Freeze(source,m_stage_moments->NumberDensity(),m_model.m_gamma)) {
            amrex::MultiFab::Copy(dq,m_source_diagonal->PCDiagonal(),0,0,1,0);
        } else {
            // An unavailable PC derivative never changes the physical source.
            ++m_source_diagonal_fallbacks;
        }
    }
    return true;
}

} // namespace warpx::thermal
