/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "CircuitCoupler.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "WarpX.H"
#include "Utils/TextMsg.H"
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
bool ValidImpulseApi(WarpxCircuitImpulseApiV1 const* api)
{
    return api && api->struct_bytes==sizeof(*api) && api->api_version==WARPX_CIRCUIT_IMPULSE_API_V1 &&
        api->capabilities==WARPX_CIRCUIT_IMPULSE_SERIES_RLC_V1 && api->prepare && api->commit && api->release;
}
bool AllRanks(bool valid,std::string& error)
{
    int ok=valid?1:0;amrex::ParallelDescriptor::ReduceIntMin(ok);
    if(!ok && valid)error="Native source impulse failed on another rank";
    return ok!=0;
}
bool Finite(double const* x,std::size_t n)
{
    if(n && !x)return false;
    for(std::size_t i=0;i<n;++i)if(!std::isfinite(x[i]))return false;
    return true;
}
struct RateLease {
    WarpxCircuitRateApiV1 const* api;
    ExternalCircuit* plugin;
    std::uint64_t token=0;
    ~RateLease(){if(token)api->release(plugin,token);}
};
}

bool CircuitCoupler::SetSourceImpulseCapability(WarpxCircuitImpulseApiV1 const* api)
{
    if(m_source_capability_attached || !m_transaction.Idle() || !m_rejection.Idle() ||
       (api && !ValidImpulseApi(api)))return false;
    m_source_impulse_api=api;m_source_capability_attached=true;return true;
}
bool CircuitCoupler::SupportsSourceImpulse() const
{
    return SupportsNativeEndpointRate() && SupportsNativeTransaction() &&
        ValidImpulseApi(m_source_impulse_api) && m_params.device_affine &&
        m_coils.size()==1 && m_coils.coil(0).n_turns==1;
}
bool CircuitCoupler::RetainedSourcePhaseReady(SourceImpulsePhase phase) const
{
    if(!RetainedNativeStepCancelable() || !SupportsSourceImpulse() || m_source_rebase_pending ||
       m_transaction_external->DeviceScales().start || m_transaction_external->DeviceScales().end)return false;
    WarpxCircuitTransactionViewV1 view{};
    if(!m_transaction.Query(view))return false;
    if(phase==SourceImpulsePhase::PreField)
        return view.phase==WARPX_TRANSACTION_SNAPSHOTTED && m_transaction_device_trials<0;
    if(phase==SourceImpulsePhase::PostField)
        return view.phase==WARPX_TRANSACTION_ACCEPTED_CLOSED && m_transaction_device_trials>=0;
    return false;
}
void CircuitCoupler::InvalidateNativeSourceImpulse()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_source_impulse_generation!=std::numeric_limits<std::uint64_t>::max(),
        "Native source impulse generation exhausted");
    ++m_source_impulse_generation;
    m_source_impulse_ready=false;m_source_impulse_applied=false;m_source_impulse_application=0;
    if(m_source_impulse_token) {
        m_source_impulse_api->release(m_plugin.get(),m_source_impulse_token);
        m_source_impulse_token=0;
    }
}
bool CircuitCoupler::PrepareNativeSourceImpulse(amrex::Real time,SourceImpulsePhase phase,std::string& error)
{
    BL_PROFILE("CircuitCoupler::PrepareNativeSourceImpulse");
    InvalidateNativeEndpointRate();error.clear();
    bool valid=std::isfinite(time) && RetainedSourcePhaseReady(phase);
    if(!valid)error="Source impulse requires its retained pre-field or closed post-field phase";
    if(!AllRanks(valid,error))return false;
    amrex::Real lo=time,hi=time;amrex::ParallelDescriptor::ReduceRealMin(lo);amrex::ParallelDescriptor::ReduceRealMax(hi);
    int phase_lo=static_cast<int>(phase),phase_hi=phase_lo;
    amrex::ParallelDescriptor::ReduceIntMin(phase_lo);amrex::ParallelDescriptor::ReduceIntMax(phase_hi);
    if(lo!=hi || phase_lo!=phase_hi){error="Source phase or clock differs across ranks";return false;}
    auto& simulation=WarpX::GetInstance();auto const& external=*m_transaction_external;
    auto const& bz=*simulation.m_fields.get(warpx::fields::FieldType::Bfield_fp,ablastr::fields::Direction{2},0);
    valid=NativeEndpointGeometryMatches(bz) && external.nFields()==m_coils.size();
    if(!valid)error="Source impulse requires immutable H/Q geometry and only circuit-owned fields";
    if(!AllRanks(valid,error))return false;
    std::vector<int> fields(m_coils.size(),-1),measured(m_coils.size());
    std::vector<double> scales(external.nFields()),iref(m_coils.size()),normalization(m_coils.size());
    for(int f=0;f<external.nFields();++f)scales[f]=external.TimeScale(f,time);
    for(int c=0;c<m_coils.size();++c){auto const& coil=m_coils.coil(c);iref[c]=coil.I_ref;
        normalization[c]=coil.I_ref*coil.n_turns;measured[c]=m_probes[c]==warpx::circuit::ProbeKind::disk;
        for(int f=0;f<external.nFields();++f)if(external.FieldName(f)==coil.field_name)fields[c]=f;
    }
    WarpxCircuitTransactionViewV1 transaction{};valid=m_transaction.Query(transaction);
    if(!AllRanks(valid,error))return false;
    WarpxCircuitImpulseViewV1 packet{};packet.struct_bytes=sizeof(packet);char message[512]{};
    int status=WARPX_IMPULSE_INTERNAL_ERROR;
    try{status=m_source_impulse_api->prepare(m_plugin.get(),transaction.token,time,
            static_cast<std::uint32_t>(phase),&packet,message,sizeof(message));}
    catch(...){error="Source impulse provider threw during preparation";}
    message[sizeof(message)-1]='\0';m_source_impulse_token=packet.token;
    valid=status==WARPX_IMPULSE_OK && packet.n_port==1 && packet.n_state==5 &&
        Finite(packet.current,1) && Finite(packet.g,1) && Finite(packet.i_ref,1) &&
        Finite(packet.state,5) && Finite(packet.delta_x,5) && Finite(packet.zeta,5);
    if(!valid && error.empty())error="Source impulse provider: "+std::to_string(status)+" "+message;
    if(!AllRanks(valid,error)){InvalidateNativeSourceImpulse();return false;}
    // Provider state is replicated. Check the entire small response packet
    // before a distributed field operator can consume inconsistent coefficients.
    std::vector<double> values{packet.current[0],packet.g[0],packet.i_ref[0]};
    for(auto const* p:{packet.state,packet.delta_x,packet.zeta})values.insert(values.end(),p,p+5);
    auto reference=values;amrex::ParallelDescriptor::Bcast(reference.data(),static_cast<int>(reference.size()),
        amrex::ParallelDescriptor::IOProcessorNumber());
    valid=values==reference;
    if(!valid)error="Replicated source impulse state/coefficients differ across ranks";
    if(!AllRanks(valid,error)){InvalidateNativeSourceImpulse();return false;}
    if(!m_source_impulse)m_source_impulse=std::make_unique<warpx::circuit::DeviceCircuitRate>();
    std::vector<double> q(m_darwin_unit_response.begin(),m_darwin_unit_response.end());
    valid=m_source_impulse->PrepareImpulse(packet,transaction.token,time,static_cast<std::uint32_t>(phase),
        fields,scales,iref,normalization,measured,m_endpoint_boundary_response,q,error);
    if(!AllRanks(valid,error)){InvalidateNativeSourceImpulse();return false;}
    m_source_entry_current.assign(packet.current,packet.current+packet.n_port);
    m_a_ext_scratch.assign(m_coils.size(),{nullptr,nullptr,nullptr});
    warpx::circuit::VectorFieldPtrs const unused{nullptr,nullptr,nullptr};
    m_batch.BuildPack(m_coils,m_probes,m_probe_exclusion,m_a_ext_scratch,&bz,unused);
    m_source_time=time;m_source_phase=phase;m_source_impulse_ready=true;return true;
}
void CircuitCoupler::ApplyNativeSourceImpulse(amrex::MultiFab const& homogeneous_delta_bz)
{
    BL_PROFILE("CircuitCoupler::ApplyNativeSourceImpulse");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_source_impulse_ready && m_source_impulse &&
        NativeEndpointGeometryMatches(homogeneous_delta_bz),"Source impulse response is stale or layout changed");
    warpx::circuit::VectorFieldPtrs const unused{nullptr,nullptr,nullptr};
    auto const& linkage=m_batch.MeasureDevice(m_coils,m_probes,m_probe_exclusion,
        m_a_ext_scratch,&homogeneous_delta_bz,unused);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_source_impulse_application!=std::numeric_limits<std::uint64_t>::max(),
        "Native source impulse application identity exhausted");
    m_source_impulse->Apply(linkage,false);m_source_impulse_applied=true;
    ++m_source_impulse_application;
}
amrex::Gpu::DeviceVector<double> const& CircuitCoupler::SourceFieldIncrements() const
{
    AMREX_ALWAYS_ASSERT(m_source_impulse_ready && m_source_impulse_applied);return m_source_impulse->FieldRates();
}
amrex::Gpu::DeviceVector<double> const& CircuitCoupler::SourcePortImpulses() const
{
    AMREX_ALWAYS_ASSERT(m_source_impulse_ready && m_source_impulse_applied);return m_source_impulse->Emf();
}
bool CircuitCoupler::CommitNativeSourceImpulse(WarpxCircuitImpulseReceiptV1& receipt,std::string& error)
{
    BL_PROFILE("CircuitCoupler::CommitNativeSourceImpulse");
    receipt={};receipt.struct_bytes=sizeof(receipt);error.clear();
    bool valid=m_source_impulse_ready && m_source_impulse_applied && RetainedSourcePhaseReady(m_source_phase);
    if(!valid)error="Source impulse has no current final device action in its retained phase";
    if(!AllRanks(valid,error))return false;
    std::vector<double> chi(m_coils.size()),increments(m_transaction_external->nFields());
    auto const& dc=SourcePortImpulses();auto const& ds=SourceFieldIncrements();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,dc.begin(),dc.end(),chi.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,ds.begin(),ds.end(),increments.begin());
    valid=Finite(chi.data(),chi.size()) && Finite(increments.data(),increments.size());
    auto reference=chi;amrex::ParallelDescriptor::Bcast(reference.data(),static_cast<int>(reference.size()),amrex::ParallelDescriptor::IOProcessorNumber());
    valid=valid && reference==chi;
    if(!valid)error="Final source port impulse is nonfinite or differs across ranks";
    if(!AllRanks(valid,error))return false;
    auto const token=m_source_impulse_token;WarpxCircuitTransactionViewV1 transaction{};
    valid=m_transaction.Query(transaction);if(!AllRanks(valid,error))return false;
    // From this point any failure requires the original owner to cancel.
    // Prevent finalization or a stale source endpoint from bypassing it.
    m_source_rebase_pending=true;m_source_endpoint_phase=0;
    char message[512]{};int status=WARPX_IMPULSE_INTERNAL_ERROR;
    try{status=m_source_impulse_api->commit(m_plugin.get(),token,chi.data(),chi.size(),&receipt,message,sizeof(message));}
    catch(...){error="Source provider threw during provisional publication; whole-step cancellation required";}
    message[sizeof(message)-1]='\0';
    valid=status==WARPX_IMPULSE_OK && receipt.struct_bytes==sizeof(receipt) && receipt.token==token &&
        receipt.transaction_token==transaction.token && receipt.phase==static_cast<std::uint32_t>(m_source_phase) && receipt.time_sim==m_source_time;
    double const work[]={receipt.coil_energy_change,receipt.local_inductor_energy_change,
        receipt.capacitor_energy_change,receipt.port_work,receipt.resistor_impulse_work,
        receipt.energy_defect,receipt.arithmetic_bound};
    valid=valid && Finite(work,7) && receipt.arithmetic_bound>=0. &&
        std::abs(receipt.energy_defect)<=receipt.arithmetic_bound &&
        receipt.capacitor_energy_change==0. && receipt.resistor_impulse_work==0.;
    if(!valid && error.empty())error="Source provider publication: "+std::to_string(status)+" "+message;
    if(!AllRanks(valid,error)){InvalidateNativeEndpointRate();return false;}
    // The actual native current is authoritative. A fresh regular rate packet
    // exposes it without another advance; no algebraic impulse is added to volts.
    WarpxCircuitRateViewV1 packet{};packet.struct_bytes=sizeof(packet);message[0]='\0';
    int rate_status=WARPX_RATE_INTERNAL_ERROR;
    try{rate_status=m_rate_api->prepare(m_plugin.get(),m_source_time,&packet,message,sizeof(message));}catch(...){}
    RateLease lease{m_rate_api,m_plugin.get(),packet.token};
    valid=rate_status==WARPX_RATE_OK && packet.struct_bytes==sizeof(packet) && packet.n_port==chi.size() &&
        Finite(packet.current,chi.size()) && Finite(packet.i_ref,chi.size());
    if(!valid)error="Fresh native current unavailable after source publication; cancel whole step";
    if(!AllRanks(valid,error)){InvalidateNativeEndpointRate();return false;}
    auto& ext=*m_transaction_external;std::vector<double> exact(chi.size());
    bool const zero=std::all_of(chi.begin(),chi.end(),[](double x){return x==0.;});
    for(int c=0;c<m_coils.size();++c){auto const& coil=m_coils.coil(c);int field=-1;
        for(int f=0;f<ext.nFields();++f)if(ext.FieldName(f)==coil.field_name)field=f;
        valid=valid && field>=0 && packet.i_ref[c]==coil.I_ref && coil.I_ref!=0.;
        if(!valid)break;
        exact[c]=packet.current[c]/coil.I_ref;
        double const predicted=m_source_entry_current[c]/coil.I_ref+increments[field];
        double const scale=std::abs(exact[c])+std::abs(m_source_entry_current[c]/coil.I_ref)+std::abs(increments[field]);
        valid=std::isfinite(exact[c]) && std::abs(exact[c]-predicted)<=2560.*std::numeric_limits<double>::epsilon()*scale &&
            (!zero || packet.current[c]==m_source_entry_current[c]);
    }
    if(!valid)error="Exact native current disagrees with final device impulse; cancel whole step";
    if(!AllRanks(valid,error)){InvalidateNativeEndpointRate();return false;}
    if(!zero)for(int c=0;c<m_coils.size();++c)ext.SetScale(m_coils.coil(c).field_name,exact[c],exact[c],m_source_time,m_source_time);
    m_source_rebase_pending=!zero;m_source_endpoint_phase=static_cast<std::uint32_t>(m_source_phase);
    InvalidateNativeEndpointRate();return true;
}
bool CircuitCoupler::RebaseNativeSourceLinkage(std::string& error)
{
    bool valid=m_source_endpoint_phase!=0 && RetainedNativeStepCancelable();
    if(!valid)error="Source linkage rebase has no retained source publication";
    if(!AllRanks(valid,error))return false;
    if(m_source_rebase_pending){MeasureDarwinLinkages(m_source_time);m_lambda_accepted=m_lambda;
        m_have_lambda_accepted=true;m_source_rebase_pending=false;}
    error.clear();return true;
}
bool CircuitCoupler::RetainedSourceEndpointReady(SourceImpulsePhase phase,amrex::Real time) const
{
    return m_source_endpoint_phase==static_cast<std::uint32_t>(phase) && time==m_source_time &&
        !m_source_rebase_pending && RetainedSourcePhaseReady(phase);
}
