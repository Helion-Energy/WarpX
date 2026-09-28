#include "NativeEndpointCheckpointIO.H"
#include <iomanip>
#include <sstream>
/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativePECPlasma.H"
#include "DarwinInitialRateSchur.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>
#include "NativePairedDarwinFields.H"
#include "ThermalCurrentRemainder.H"
#include "NativeVacuumConstraint.H"
#include "DarwinABoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <cmath>
#include <limits>
namespace warpx::darwin {
namespace {
using R=amrex::Real;using MF=amrex::MultiFab;
using V=ablastr::fields::VectorField;using CV=ablastr::fields::ConstVectorField;
using D=ablastr::fields::Direction;using warpx::fields::FieldType;
namespace dd=warpx::ohm::compensated;
CV Const(V const& x){return {x[0],x[1],x[2]};}
void Copy(V const& a,CV const& b){for(int c=0;c<3;++c)MF::Copy(*a[c],*b[c],0,0,1,amrex::min(a[c]->nGrowVect(),b[c]->nGrowVect()));}
void Sync(WarpX& w,V const& a){for(auto* f:a){f->OverrideSync(w.Geom(0).periodicity());f->FillBoundary(w.Geom(0).periodicity());}}
void ScalarImages(WarpX& w,MF& p){p.setBndry(0.);p.OverrideSync(w.Geom(0).periodicity());p.FillBoundary(w.Geom(0).periodicity());}
void LongImages(WarpX& w,V const& a){for(auto* f:a)f->setBndry(0.);Sync(w,a);}
void ElectricImages(WarpX& w,V const& a){
    // Match SetElectricFieldAndApplyBCs in the admitted no-PML, native-precision
    // scope: owner/periodic completion first, then only the native gather band.
    // A later sync would change the PEC/axis corner images; allocation-wide PEC
    // images would populate guards that the native high-field path never writes.
    const amrex::Vector<MF*> fields(a.begin(),a.end());
    amrex::FillBoundaryAndSync_nowait(fields,w.Geom(0).periodicity());
    amrex::FillBoundaryAndSync_finish(fields);
    amrex::Vector<amrex::IntVect> ratios;
    PEC::ApplyPECtoEfield(a,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PEC,w.get_ng_fieldgather(),w.Geom(0),0,ablastr::utils::enums::PatchType::fine,ratios);
#if defined(WARPX_DIM_RZ)
    w.ApplyFieldBoundaryOnAxis(a[0],a[1],a[2],0);
#endif
}
void PairExtrapolate(MF& high,MF& low,MF const& old,R theta){
    for(amrex::MFIter mfi(high);mfi.isValid();++mfi){auto h=high.array(mfi),l=low.array(mfi);auto o=old.const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto change=dd::Add({h(i,j,k),l(i,j,k)},{-o(i,j,k),0.});
            auto result=dd::Add({o(i,j,k),0.},dd::Multiply(change,1./theta));
            h(i,j,k)=result.hi;l(i,j,k)=result.lo;
        });}
}
}
struct NativePairedDarwinFields::EndpointReceipt::Token {
    bool active=false,exhausted=false;
    std::uint64_t generation=0;
};
bool NativePairedDarwinFields::EndpointReceipt::Valid()const noexcept {
    auto t=m_token.lock();return t&&t->active&&!t->exhausted&&t->generation==m_generation;
}
struct NativePairedDarwinFields::Impl {
    WarpX* simulation;
    Snapshot state;
    Role role;
    bool accepted_consumers=false;
    V endpoint_auxiliary_high{},endpoint_auxiliary_low{};
    std::array<V,endpoint_parts> endpoint{};
    MF *endpoint_nodal_high=nullptr,*endpoint_nodal_low=nullptr;
    std::unique_ptr<EndpointDraft> endpoint_draft,endpoint_prior,endpoint_field_origin;
    // Private S1 reference and post-S1 spatial Core. The original prior Core
    // stays immutable until whole-step Accept or Cancel.
    std::array<MF,3> endpoint_pre_high,endpoint_pre_low;
    bool endpoint_pre_reclosure=false;
    EndpointDraft* CurrentEndpoint() const noexcept {
        return state.phase==Phase::EndpointSourceOrigin?endpoint_prior.get():endpoint_draft.get();
    }
    EndpointDraft* FieldOrigin() const noexcept {
        return endpoint_field_origin?endpoint_field_origin.get():endpoint_prior.get();
    }
    Snapshot endpoint_step_snapshot;
    V endpoint_old_high{},endpoint_old_low{},endpoint_previous_high{},endpoint_previous_low{};
    bool endpoint_accept_ready=false;
    std::shared_ptr<EndpointReceipt::Token> endpoint_token;
    void RevokeEndpoint() noexcept {
        if(!endpoint_token)return;
        endpoint_token->active=false;
        if(endpoint_token->generation==std::numeric_limits<std::uint64_t>::max())
            endpoint_token->exhausted=true;
        else ++endpoint_token->generation;
    }
    V el_low,old_el_low,t_low,t_image,a_low,b_low,b_scratch,c_low,c_scratch,
      d_low,rate_low,cdot_low,rate_a,rate_b,defect,old_a,saved_b,saved_c,
      ei_low,total_low,total_high,stage_ji,stage_je,stage_force_je,saved_e,held_je,account_edge,
      saved_bl,saved_cl,saved_ll,saved_tl,saved_al,saved_dl,saved_rl,
      t_high,saved_th,saved_ah,saved_lh;
    std::uint64_t local_memory=0;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> edge_owners,magnetic_owners;
    MF *phi_low=nullptr,*old_phi_low=nullptr,*phi_origin=nullptr,*saved_phi=nullptr,*saved_phi_low=nullptr;
    std::vector<MF*> allocations;
    explicit Impl(WarpX& w,Role r,bool accepted):simulation(&w),role(r),accepted_consumers(accepted){
        AMREX_ALWAYS_ASSERT(!accepted_consumers||role==Role::Endpoint);
        auto electric=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
        auto el=w.m_fields.get_alldirs("hybrid_E_long_fp",0);
        auto potential=w.m_fields.get_alldirs("hybrid_A_fp",0);
        auto magnetic=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
        auto current=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto allocate=[&](char const* name,V const& shape,int grow=-1){
            V out;
            for(int c=0;c<3;++c){auto const& f=*shape[c];
                AMREX_ALWAYS_ASSERT(!w.m_fields.has(name,D{c},0));
                w.m_fields.alloc_init(name,D{c},0,f.boxArray(),f.DistributionMap(),1,
                    grow<0?f.nGrowVect():amrex::IntVect(grow),0.,true,true,false);
                out[c]=w.m_fields.get(name,D{c},0);allocations.push_back(out[c]);}
            return out;
        };
        if(role==Role::Endpoint) {
            // All endpoint arrays exist before the original DTA snapshot. No
            // finite-stage A/EL/D/history low storage is allocated for this role.
            char const* names[]={"native_endpoint_pair_ET_high_fp","native_endpoint_pair_ET_low_fp",
                "native_endpoint_pair_E_high_fp","native_endpoint_pair_E_low_fp",
                "native_endpoint_pair_Ei_high_fp","native_endpoint_pair_Ei_low_fp",
                "native_endpoint_pair_dJe_high_fp","native_endpoint_pair_dJe_low_fp",
                "native_endpoint_pair_dJi_high_fp","native_endpoint_pair_dJi_low_fp"};
            for(int k=0;k<endpoint_parts;++k)endpoint[k]=allocate(names[k],electric);
            auto const& nodal=*w.m_fields.get("hybrid_E_inertial_nodal",0);
            auto scalar=[&](char const* name){
                AMREX_ALWAYS_ASSERT(!w.m_fields.has(name,0));
                w.m_fields.alloc_init(name,0,nodal.boxArray(),nodal.DistributionMap(),
                    nodal.nComp(),nodal.nGrowVect(),0.,true,true,false);
                auto* out=w.m_fields.get(name,0);allocations.push_back(out);return out;
            };
            endpoint_nodal_high=scalar("native_endpoint_pair_Ei_high_nodal");
            endpoint_nodal_low=scalar("native_endpoint_pair_Ei_low_nodal");
            endpoint_token=std::make_shared<EndpointReceipt::Token>();
            // References need valid values only; physical E/Ei images remain
            // in the existing endpoint holders. Allocate before DTA capture.
            endpoint_old_high=allocate("native_endpoint_pair_ET_old_high_fp",electric,0);
            endpoint_old_low=allocate("native_endpoint_pair_ET_old_low_fp",electric,0);
            endpoint_previous_high=allocate("native_endpoint_pair_ET_previous_high_fp",electric,0);
            endpoint_previous_low=allocate("native_endpoint_pair_ET_previous_low_fp",electric,0);
            if(accepted_consumers){
                auto const auxiliary=w.m_fields.get_alldirs(FieldType::Efield_aux,0);
                endpoint_auxiliary_high=allocate("native_endpoint_pair_E_high_aux",auxiliary);
                endpoint_auxiliary_low=allocate("native_endpoint_pair_E_low_aux",auxiliary);
            }
            for(auto const* f:allocations)for(amrex::MFIter it(*f);it.isValid();++it)
                local_memory+=(*f)[it].size()*sizeof(R);
            return;
        }
        el_low=allocate("native_pair_EL_low_fp",el);
        old_el_low=allocate("native_pair_EL_previous_low_fp",el);
        t_low=allocate("native_pair_ET_low_fp",electric);
        t_image=allocate("native_pair_ET_image_fp",electric);
        a_low=allocate("native_pair_A_low_fp",potential);
        b_low=allocate("native_pair_B_low_fp",magnetic);
        b_scratch=allocate("native_pair_B_curl_scratch_fp",magnetic);
        c_low=allocate("native_pair_C_low_fp",current,1);
        c_scratch=allocate("native_pair_C_scratch_fp",current,1);
        d_low=allocate("native_pair_D_low_fp",current,1);
        rate_low=allocate("native_pair_dJe_low_fp",current,1);
        cdot_low=allocate("native_pair_Cdot_low_fp",current,1);
        rate_a=allocate("native_pair_rate_A_fp",potential);
        rate_b=allocate("native_pair_rate_B_fp",magnetic);
        defect=allocate("native_pair_constraint_defect_fp",el);
        old_a=allocate("native_pair_stage_A_origin_fp",potential);
        saved_b=allocate("native_pair_publication_B_fp",magnetic);
        saved_c=allocate("native_pair_publication_C_fp",current,1);
        ei_low=allocate("native_pair_Ei_low_fp",current,1);
        total_low=allocate("native_pair_E_sum_low_fp",electric);
        total_high=allocate("native_pair_E_sum_high_fp",electric);
        t_high=allocate("native_pair_ET_physical_high_fp",electric);
        saved_th=allocate("native_pair_publication_ET_high_fp",electric);
        saved_ah=allocate("native_pair_publication_A_high_fp",potential);
        saved_lh=allocate("native_pair_publication_EL_high_fp",el);
        stage_ji=allocate("native_pair_stage_interval_Ji_fp",current,1);
        stage_je=allocate("native_pair_stage_inertia_Je_fp",current,1);
        stage_force_je=allocate("native_pair_stage_force_Je_fp",current,1);
        saved_e=allocate("native_pair_publication_E_fp",electric);
        held_je=allocate("native_pair_publication_held_Je_fp",current,1);
        account_edge=allocate("native_pair_publication_edge_scratch_fp",current,1);
        saved_bl=allocate("native_pair_publication_B_low_fp",magnetic);
        saved_cl=allocate("native_pair_publication_C_low_fp",current,1);
        saved_ll=allocate("native_pair_publication_EL_low_fp",el);
        saved_tl=allocate("native_pair_publication_ET_low_fp",electric);
        saved_al=allocate("native_pair_publication_A_low_fp",potential);
        saved_dl=allocate("native_pair_publication_D_low_fp",current,1);
        saved_rl=allocate("native_pair_publication_dJe_low_fp",current,1);
        for(int c=0;c<3;++c){edge_owners[c]=current[c]->OwnerMask(w.Geom(0).periodicity());magnetic_owners[c]=magnetic[c]->OwnerMask(w.Geom(0).periodicity());}
        auto const& phi=*w.m_fields.get("hybrid_phi_darwin_fp",0);
        auto scalar=[&](char const* name){
            AMREX_ALWAYS_ASSERT(!w.m_fields.has(name,0));
            w.m_fields.alloc_init(name,0,phi.boxArray(),phi.DistributionMap(),1,phi.nGrowVect(),0.,true,true,false);
            auto* out=w.m_fields.get(name,0);allocations.push_back(out);return out;
        };
        phi_low=scalar("native_pair_phi_low_fp");
        old_phi_low=scalar("native_pair_phi_previous_low_fp");
        phi_origin=scalar("native_pair_phi_origin_fp");
        saved_phi=scalar("native_pair_publication_phi_high_fp");
        saved_phi_low=scalar("native_pair_publication_phi_low_fp");
        for(auto const* f:allocations)for(amrex::MFIter mfi(*f);mfi.isValid();++mfi)local_memory+=(*f)[mfi].size()*sizeof(R);
        for(int c=0;c<3;++c)for(auto const* f:{edge_owners[c].get(),magnetic_owners[c].get()})for(amrex::MFIter mfi(*f);mfi.isValid();++mfi)local_memory+=(*f)[mfi].size()*sizeof(int);
    }
    WarpX& Sim() const {return *simulation;}
    void RequireFiniteStageRole() const {AMREX_ALWAYS_ASSERT(role==Role::FiniteStage);}
    void RequireActive() const {RequireFiniteStageRole();AMREX_ALWAYS_ASSERT(state.phase==Phase::FiniteStage);}
    bool Finite() const {
        bool valid=true;
        // Every owned allocation is initialized, including private ghosts.
        // Local checks never skip a collective on a rank with a bad value.
        for(auto const* f:allocations)valid=f->is_finite(0,1,f->nGrowVect(),true)&&valid;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);return valid;
    }
    V EL() const{return Sim().m_fields.get_alldirs("hybrid_E_long_fp",0);}
    V Electric() const{return Sim().m_fields.get_alldirs(FieldType::Efield_fp,0);}
    void Curl(V const& in,V out){
        for(auto* f:out)f->setVal(0.);
        Sim().get_pointer_fdtd_solver_fp(0)->ComputeCurlA(out,in,Sim().GetEBUpdateBFlag()[0],0,
            DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
        Sync(Sim(),out);
    }
    void Ampere(V const& in,V out){
        Sim().get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(out,in,Sim().GetEBUpdateEFlag()[0],0);
    }
};
bool NativePairedDarwinFields::Requested(){bool value=false;amrex::ParmParse("endpoint_diagnostic").query("positive_paired_increment",value);return value;}
bool NativePairedDarwinFields::EndpointRequested(){bool value=false;amrex::ParmParse("endpoint_diagnostic").query("retained_endpoint_pair",value);return value;}
bool NativePairedDarwinFields::EndpointAcceptanceRequested(){bool value=false;amrex::ParmParse("endpoint_diagnostic").query("retained_endpoint_pair_accept",value);return value;}
NativePairedDarwinFields::NativePairedDarwinFields(WarpX& w,Role role,bool accepted):m_impl(std::make_unique<Impl>(w,role,accepted)){}
NativePairedDarwinFields::~NativePairedDarwinFields()=default;
bool NativePairedDarwinFields::Active() const noexcept{return m_impl->state.phase==Phase::FiniteStage;}
std::uint64_t NativePairedDarwinFields::LocalMemory()const noexcept{return m_impl->local_memory;}
NativePairedDarwinFields::Snapshot NativePairedDarwinFields::State()const noexcept{return m_impl->state;}
void NativePairedDarwinFields::RestoreState(Snapshot const& s)noexcept{
    if(m_impl->role==Role::Endpoint){
        auto& p=*m_impl;
        AMREX_ALWAYS_ASSERT(p.state.endpoint_step_open&&!s.endpoint_step_open&&
            (s.phase==Phase::Idle||s.phase==Phase::EndpointAccepted)&&
            s.phase==p.endpoint_step_snapshot.phase&&s.time==p.endpoint_step_snapshot.time&&
            s.epoch==p.endpoint_step_snapshot.epoch&&
            s.endpoint_boundary_bound==p.endpoint_step_snapshot.endpoint_boundary_bound);
        // Registered bytes were already restored by FieldRollback. Restore
        // the old immutable Core, not a new solve or a reconstructed field.
        p.RevokeEndpoint();p.endpoint_draft=std::move(p.endpoint_prior);
        p.endpoint_field_origin.reset();p.endpoint_pre_reclosure=false;
        for(auto* v:{&p.endpoint_pre_high,&p.endpoint_pre_low})for(auto& f:*v)f.clear();
        p.endpoint_accept_ready=false;
        if(s.phase==Phase::EndpointAccepted){
            AMREX_ALWAYS_ASSERT(p.endpoint_draft&&p.endpoint_draft->Ready());
            p.endpoint_token->active=!p.endpoint_token->exhausted;
        }else AMREX_ALWAYS_ASSERT(!p.endpoint_draft);
    }
    m_impl->state=s;
}
void NativePairedDarwinFields::InvalidateBorrowedViews()noexcept{
    m_impl->state.magnetic_ready=false;m_impl->state.current_ready=false;
    if(m_impl->role==Role::Endpoint){m_impl->RevokeEndpoint();m_impl->endpoint_accept_ready=false;}
}
bool NativePairedDarwinFields::EndpointPending()const noexcept {
    return m_impl->role==Role::Endpoint&&m_impl->state.phase==Phase::EndpointPending;
}
bool NativePairedDarwinFields::EndpointAccepted()const noexcept {
    return m_impl->role==Role::Endpoint&&m_impl->state.phase==Phase::EndpointAccepted;
}
bool NativePairedDarwinFields::EndpointPreSourceReclosure()const noexcept {
    return m_impl->role==Role::Endpoint&&m_impl->endpoint_pre_reclosure;
}
NativePairedDarwinFields::EndpointReceipt NativePairedDarwinFields::EndpointLease()const noexcept {
    EndpointReceipt value;
    if((EndpointPending()||EndpointAccepted()||m_impl->state.phase==Phase::EndpointSourceOrigin||
        m_impl->state.phase==Phase::EndpointSourcePending)&&m_impl->endpoint_token->active){
        value.m_token=m_impl->endpoint_token;value.m_generation=m_impl->endpoint_token->generation;
    }
    return value;
}
NativePairedDarwinFields::EndpointView NativePairedDarwinFields::EndpointFields()const {
    EndpointView value;
    if(m_impl->role!=Role::Endpoint)return value;
    for(int k=0;k<endpoint_parts;++k)value.fields[k]=Const(m_impl->endpoint[k]);
    value.nodal_inertia_high=m_impl->endpoint_nodal_high;
    value.nodal_inertia_low=m_impl->endpoint_nodal_low;return value;
}
bool NativePairedDarwinFields::MatchesEndpoint(EndpointReceipt const& receipt)const {
    auto token=receipt.m_token.lock();
    bool ready=(EndpointPending()||EndpointAccepted()||m_impl->state.phase==Phase::EndpointSourceOrigin||
        m_impl->state.phase==Phase::EndpointSourcePending)&&receipt.Valid()&&token.get()==m_impl->endpoint_token.get()&&
        bool(m_impl->CurrentEndpoint());
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return false;
    return m_impl->CurrentEndpoint()->Matches(m_impl->Sim(),EndpointFields());
}
bool NativePairedDarwinFields::AdoptEndpoint(EndpointDraft&& draft,R time,std::uint64_t epoch){
    auto& s=*m_impl;
    bool const pre=s.endpoint_pre_reclosure;
    int pre_min=pre,pre_max=pre;amrex::ParallelDescriptor::ReduceIntMin(pre_min);amrex::ParallelDescriptor::ReduceIntMax(pre_max);
    if(pre_min!=pre_max)return false;
    bool ready=s.role==Role::Endpoint&&s.state.phase==Phase::Idle&&
        s.state.endpoint_step_open&&(pre?!s.state.endpoint_history_captured:s.state.endpoint_history_captured)&&draft.Ready()&&
        s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()&&
        std::isfinite(time)&&(pre?time==s.Sim().gett_new(0):time>s.Sim().gett_new(0))&&
        epoch==std::uint64_t(s.Sim().getistep(0))+(pre?0:1);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!draft.Bind(s.Sim(),time,epoch,false,false,pre))return false;
    if(!draft.Publish(s.Sim(),s.endpoint,*s.endpoint_nodal_high,*s.endpoint_nodal_low))return false;
    s.endpoint_draft=std::make_unique<EndpointDraft>(std::move(draft));
    s.state.phase=pre?Phase::EndpointSourcePending:Phase::EndpointPending;s.state.time=time;s.state.epoch=epoch;
    s.state.endpoint_boundary_bound=false;s.state.endpoint_auxiliary_ready=false;s.endpoint_accept_ready=false;
    ++s.endpoint_token->generation;s.endpoint_token->active=true;return true;
}
bool NativePairedDarwinFields::InitialEndpointSelected()const noexcept {
    return m_impl->role==Role::Endpoint&&m_impl->accepted_consumers;
}
bool NativePairedDarwinFields::InitialEndpointPending()const noexcept {
    return m_impl->role==Role::Endpoint&&m_impl->state.phase==Phase::EndpointInitializing;
}
bool NativePairedDarwinFields::AdoptInitialEndpoint(EndpointDraft&& draft,R time){
    auto& s=*m_impl;
    bool ready=InitialEndpointSelected()&&s.state.phase==Phase::Idle&&
        !s.state.endpoint_step_open&&!s.state.endpoint_history_captured&&
        !s.state.endpoint_origin_ready&&!s.state.endpoint_previous_ready&&
        !s.endpoint_draft&&!s.endpoint_prior&&draft.Ready()&&
        s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()&&
        std::isfinite(time)&&time==s.Sim().gett_new(0)&&s.Sim().getistep(0)==0&&
        !s.Sim().get_pointer_HybridPICModel()->m_darwin_checkpoint_restored;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!draft.Bind(s.Sim(),time,0,true))return false;
    if(!draft.Publish(s.Sim(),s.endpoint,*s.endpoint_nodal_high,*s.endpoint_nodal_low))return false;
    s.endpoint_draft=std::make_unique<EndpointDraft>(std::move(draft));
    s.state.phase=Phase::EndpointInitializing;s.state.time=time;s.state.epoch=0;
    s.state.endpoint_boundary_bound=false;s.state.endpoint_auxiliary_ready=false;
    s.endpoint_accept_ready=false;
    // No receipt is exposed until native auxiliary initialization is complete.
    // The first real step will capture this pair as its origin, with no prior
    // timestep history and no artificial clock or epoch transition.
    return true;
}
bool NativePairedDarwinFields::CompleteInitialEndpoint(){
    auto& s=*m_impl;
    bool ready=InitialEndpointPending()&&s.accepted_consumers&&s.endpoint_draft&&
        !s.state.endpoint_step_open&&!s.state.endpoint_history_captured&&
        !s.state.endpoint_origin_ready&&!s.state.endpoint_previous_ready&&
        !s.state.endpoint_boundary_bound&&!s.state.endpoint_auxiliary_ready&&
        s.endpoint_token&&!s.endpoint_token->active&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()&&
        s.state.time==s.Sim().gett_new(0)&&s.state.epoch==0&&s.Sim().getistep(0)==0;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!s.endpoint_draft->Matches(s.Sim(),EndpointFields()))return false;
    auto const result=s.endpoint_draft->Check(s.Sim(),EndpointFields());
    ready=result.compatible&&result.converged;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!s.endpoint_draft->PublishAuxiliary(s.Sim(),s.endpoint_auxiliary_high,s.endpoint_auxiliary_low))return false;
    if(!s.endpoint_draft->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    s.state.phase=Phase::EndpointAccepted;s.state.endpoint_boundary_bound=true;
    s.state.endpoint_auxiliary_ready=true;
    ++s.endpoint_token->generation;s.endpoint_token->active=true;return true;
}
namespace {
bool EndpointSameValid(MF const& a,MF const& b){
    bool good=a.boxArray()==b.boxArray()&&a.DistributionMap()==b.DistributionMap()&&a.nComp()>=1&&b.nComp()>=1;
    amrex::ParallelDescriptor::ReduceBoolAnd(good);if(!good)return false;
    amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
    for(amrex::MFIter it(a);it.isValid();++it){auto x=a.const_array(it),y=b.const_array(it);
        amrex::For(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto const* p=reinterpret_cast<unsigned char const*>(&x(i,j,k));
            auto const* q=reinterpret_cast<unsigned char const*>(&y(i,j,k));
            bool mismatch=false;for(std::size_t z=0;z<sizeof(R);++z)mismatch=mismatch||(p[z]!=q[z]);
            if(mismatch)amrex::Gpu::Atomic::Exch(bad,1);
        });
    }
    good=invalid.dataValue()==0;amrex::ParallelDescriptor::ReduceBoolAnd(good);return good;
}
bool EndpointShares(MF const& a,MF const& b){
    using Range=std::pair<std::uintptr_t,std::uintptr_t>;
    auto collect=[](MF const& f,std::vector<Range>& out){
        for(amrex::MFIter it(f);it.isValid();++it){
            auto begin=reinterpret_cast<std::uintptr_t>(f[it].dataPtr());
            auto n=static_cast<std::uint64_t>(f[it].size());
            if(n>std::numeric_limits<std::uintptr_t>::max()/sizeof(R))return false;
            auto bytes=std::uintptr_t(n*sizeof(R));
            if(begin>std::numeric_limits<std::uintptr_t>::max()-bytes||(!begin&&bytes))return false;
            out.emplace_back(begin,begin+bytes);
        }
        return true;
    };
    std::vector<Range> left,right;bool const x=collect(a,left),y=collect(b,right);
    if(!x||!y)return true;
    for(auto const& l:left)for(auto const& r:right)if(l.first<r.second&&r.first<l.second)return true;
    return false;
}
}
bool NativePairedDarwinFields::CheckpointReady() const {
    auto const& s=*m_impl;
    bool good=EndpointAccepted()&&s.accepted_consumers&&!s.state.endpoint_step_open&&
        s.state.endpoint_boundary_bound&&s.state.endpoint_auxiliary_ready&&s.endpoint_draft&&
        !s.endpoint_prior&&s.state.time==s.Sim().gett_new(0)&&s.state.epoch==std::uint64_t(s.Sim().getistep(0));
    amrex::ParallelDescriptor::ReduceBoolAnd(good);if(!good)return false;
    if(!MatchesEndpoint(EndpointLease())||!s.endpoint_draft->MatchesAuxiliary(s.Sim(),
        Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    auto const result=s.endpoint_draft->Check(s.Sim(),EndpointFields());
    return result.compatible&&result.converged;
}
void NativePairedDarwinFields::WriteEndpointCheckpoint(std::string const& directory) const {
    namespace io=endpoint_checkpoint;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(CheckpointReady(),"Endpoint checkpoint requires a current accepted boundary");
    auto const& s=*m_impl;
    s.endpoint_draft->WriteCheckpoint(directory);
    for(auto const& entry:{std::pair<char const*,V const*>{"old_high",&s.endpoint_old_high},
        {"old_low",&s.endpoint_old_low},{"previous_high",&s.endpoint_previous_high},
        {"previous_low",&s.endpoint_previous_low},{"aux_high",&s.endpoint_auxiliary_high},
        {"aux_low",&s.endpoint_auxiliary_low}})
        for(int d=0;d<3;++d)io::Write(directory+"/"+entry.first+std::to_string(d),*(*entry.second)[d]);
    std::ostringstream out;out<<std::setprecision(std::numeric_limits<R>::max_digits10)
        <<s.state.time<<' '<<s.state.epoch<<' '<<s.state.dt<<' '<<s.state.trials<<'\n'
        <<s.state.endpoint_history_captured<<' '<<s.state.endpoint_origin_ready<<' '
        <<s.state.endpoint_previous_ready<<' '<<s.state.endpoint_origin_paired<<'\n';
    io::WriteText(directory+"/history",out.str());
}
void NativePairedDarwinFields::ReadEndpointCheckpoint(std::string const& directory) {
    namespace io=endpoint_checkpoint;auto& s=*m_impl;
    bool empty=s.role==Role::Endpoint&&s.accepted_consumers&&s.state.phase==Phase::Idle&&
        !s.state.endpoint_step_open&&!s.endpoint_draft&&!s.endpoint_prior&&s.endpoint_token&&
        !s.endpoint_token->active&&!s.endpoint_token->exhausted&&s.endpoint_token->generation==0&&
        s.Sim().get_pointer_HybridPICModel()->m_darwin_checkpoint_restored;
    amrex::ParallelDescriptor::ReduceBoolAnd(empty);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(empty,"Endpoint checkpoint can initialize only a fresh restart owner");
    Snapshot restored;std::array<int,4> flags{};
    std::istringstream in(io::ReadText(directory+"/history"));
    in>>restored.time>>restored.epoch>>restored.dt>>restored.trials;
    for(int& value:flags)in>>value;
    bool valid=bool(in)&&std::isfinite(restored.time)&&std::isfinite(restored.dt)&&restored.dt>=0.&&
        restored.time==s.Sim().gett_new(0)&&restored.epoch==std::uint64_t(s.Sim().getistep(0));
    for(int value:flags)valid=(value==0||value==1)&&valid;
    valid=valid&&flags[0]==(restored.epoch>0)&&flags[1]==(restored.epoch>0)&&
        flags[2]==(restored.epoch>1)&&flags[3]==(restored.epoch>0);
    in>>std::ws;valid=in.eof()&&valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,"Invalid accepted endpoint clock/history checkpoint");
    restored.endpoint_history_captured=flags[0];restored.endpoint_origin_ready=flags[1];
    restored.endpoint_previous_ready=flags[2];restored.endpoint_origin_paired=flags[3];
    for(auto const& entry:{std::pair<char const*,V*>{"old_high",&s.endpoint_old_high},
        {"old_low",&s.endpoint_old_low},{"previous_high",&s.endpoint_previous_high},
        {"previous_low",&s.endpoint_previous_low},{"aux_high",&s.endpoint_auxiliary_high},
        {"aux_low",&s.endpoint_auxiliary_low}})
        for(int d=0;d<3;++d)io::Read(directory+"/"+entry.first+std::to_string(d),*(*entry.second)[d]);
    std::array<std::vector<int>,6> auxiliary;
    for(int d=0;d<3;++d){auxiliary[d]=io::FieldChecksum(*s.endpoint_auxiliary_high[d]);
        auxiliary[d+3]=io::FieldChecksum(*s.endpoint_auxiliary_low[d]);}
    auto draft=EndpointDraft::ReadCheckpoint(s.Sim(),directory,restored.time,restored.epoch);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(draft.Publish(s.Sim(),s.endpoint,*s.endpoint_nodal_high,*s.endpoint_nodal_low),
        "Cannot bind freshly validated endpoint numeric representation");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(draft.PublishAuxiliary(s.Sim(),s.endpoint_auxiliary_high,s.endpoint_auxiliary_low),
        "Cannot reconstruct accepted endpoint gather fields");
    for(int d=0;d<3;++d){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(auxiliary[d]==io::FieldChecksum(*s.endpoint_auxiliary_high[d])&&
            auxiliary[d+3]==io::FieldChecksum(*s.endpoint_auxiliary_low[d]),
            "Restarted endpoint gather differs from its saved physical pair");}
    // Only current numerical validation grants a new session-local lease.
    // No old generation, token, pointer, phase or ready bit is deserialized.
    s.endpoint_draft=std::make_unique<EndpointDraft>(std::move(draft));
    restored.phase=Phase::EndpointAccepted;restored.endpoint_boundary_bound=true;
    restored.endpoint_auxiliary_ready=true;s.state=restored;
    ++s.endpoint_token->generation;s.endpoint_token->active=true;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(CheckpointReady(),"Restored endpoint boundary is not current");
}
bool NativePairedDarwinFields::BeginEndpointStep(R time,std::uint64_t epoch){
    auto& s=*m_impl;
    bool ready=s.role==Role::Endpoint&&!s.state.endpoint_step_open&&!s.endpoint_prior&&!s.endpoint_field_origin&&!s.endpoint_pre_reclosure&&
        (s.state.phase==Phase::Idle||s.state.phase==Phase::EndpointAccepted)&&
        s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()&&
        std::isfinite(time)&&time==s.Sim().gett_new(0)&&epoch==std::uint64_t(s.Sim().getistep(0));
    if(s.state.phase==Phase::EndpointAccepted)
        ready=s.state.endpoint_boundary_bound&&s.state.time==time&&s.state.epoch==epoch&&ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    int lo=s.state.phase==Phase::EndpointAccepted,hi=lo;
    amrex::ParallelDescriptor::ReduceIntMin(lo);amrex::ParallelDescriptor::ReduceIntMax(hi);
    if(lo!=hi)return false;
    if(lo&&!MatchesEndpoint(EndpointLease()))return false;
    s.endpoint_step_snapshot=s.state;s.endpoint_prior=std::move(s.endpoint_draft);
    s.RevokeEndpoint();s.endpoint_accept_ready=false;s.state.phase=Phase::Idle;
    s.state.endpoint_step_open=true;s.state.endpoint_history_captured=false;
    s.state.endpoint_boundary_bound=false;s.state.endpoint_auxiliary_ready=false;return true;
}
bool NativePairedDarwinFields::BeginEndpointPreSource(){
    auto& s=*m_impl;
    bool ready=s.role==Role::Endpoint&&s.accepted_consumers&&s.state.phase==Phase::Idle&&
        s.state.endpoint_step_open&&!s.state.endpoint_history_captured&&!s.endpoint_draft&&
        s.endpoint_prior&&!s.endpoint_field_origin&&!s.endpoint_pre_reclosure&&
        s.endpoint_step_snapshot.phase==Phase::EndpointAccepted&&
        s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()-2&&
        !s.Sim().get_pointer_HybridPICModel()->m_darwin_checkpoint_restored;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!s.endpoint_prior->Matches(s.Sim(),EndpointFields()))return false;
    s.state.phase=Phase::EndpointSourceOrigin;s.state.time=s.Sim().gett_new(0);
    s.state.epoch=std::uint64_t(s.Sim().getistep(0));
    s.state.endpoint_auxiliary_ready=s.endpoint_step_snapshot.endpoint_auxiliary_ready;
    s.endpoint_token->active=true;return true;
}
bool NativePairedDarwinFields::FinishEndpointPreSource(EndpointReceipt const& receipt){
    auto& s=*m_impl;
    bool ready=s.state.phase==Phase::EndpointSourcePending&&s.endpoint_pre_reclosure&&
        s.state.endpoint_step_open&&!s.state.endpoint_history_captured&&
        s.state.endpoint_auxiliary_ready&&s.endpoint_prior&&s.endpoint_draft&&!s.endpoint_field_origin;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!MatchesEndpoint(receipt))return false;
    if(!s.endpoint_draft->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    auto const check=s.endpoint_draft->Check(s.Sim(),EndpointFields());
    if(!check.compatible||!check.converged)return false;
    s.RevokeEndpoint();s.endpoint_field_origin=std::move(s.endpoint_draft);
    s.endpoint_pre_reclosure=false;s.state.phase=Phase::Idle;
    s.state.endpoint_auxiliary_ready=false;s.state.endpoint_boundary_bound=false;
    for(auto* v:{&s.endpoint_pre_high,&s.endpoint_pre_low})for(auto& f:*v)f.clear();
    return true;
}
bool NativePairedDarwinFields::BeginEndpointSourceReclosure(EndpointReceipt const& receipt){
    auto& s=*m_impl;
    bool const pre=s.state.phase==Phase::EndpointSourceOrigin;
    bool ready=s.role==Role::Endpoint&&s.accepted_consumers&&(EndpointPending()||pre)&&
        s.state.endpoint_step_open&&(pre?!s.state.endpoint_history_captured:s.state.endpoint_history_captured)&&
        s.state.endpoint_auxiliary_ready&&s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()-1;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready||!MatchesEndpoint(receipt))return false;
    if(pre){
        for(int c=0;c<3;++c)for(int part=0;part<2;++part){
            auto& out=part?s.endpoint_pre_low[c]:s.endpoint_pre_high[c];
            auto const& in=*s.endpoint[int(part?EndpointPart::TransverseLow:EndpointPart::TransverseHigh)][c];
            out.define(in.boxArray(),in.DistributionMap(),1,0);MF::Copy(out,in,0,0,1,0);
        }
        s.endpoint_pre_reclosure=true;
    }
    // Revoke before physical source writes. BeginEndpointStep would replace
    // the original rollback boundary and rotate history, so do not call it.
    s.RevokeEndpoint();s.endpoint_draft.reset();s.endpoint_accept_ready=false;
    s.state.phase=Phase::Idle;s.state.endpoint_boundary_bound=false;
    s.state.endpoint_auxiliary_ready=false;return true;
}
bool NativePairedDarwinFields::CaptureEndpointHistory(V const& old,V const& previous,bool keep_previous){
    auto& s=*m_impl;
    bool ready=s.role==Role::Endpoint&&s.state.phase==Phase::Idle&&
        s.state.endpoint_step_open&&!s.state.endpoint_history_captured;
    int policy=keep_previous?1:0,minimum=policy,maximum=policy;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
    ready=minimum==maximum&&ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    for(int c=0;c<3;++c)for(auto const* f:{old[c],previous[c]})
        ready=f&&f->nComp()==1&&f->boxArray()==s.endpoint_old_high[c]->boxArray()&&
            f->DistributionMap()==s.endpoint_old_high[c]->DistributionMap()&&ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    std::vector<MF*> outputs{old[0],old[1],old[2],previous[0],previous[1],previous[2]};
    for(std::size_t i=0;i<outputs.size();++i){
        for(std::size_t j=0;j<i;++j)ready=!EndpointShares(*outputs[i],*outputs[j])&&ready;
        for(auto const& name:s.Sim().m_fields.list())
            ready=!EndpointShares(*outputs[i],*s.Sim().m_fields.internal_get(name))&&ready;
    }
    for(auto const* f:old)ready=f->is_finite(0,1,0,true)&&ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    bool const warm=bool(s.FieldOrigin());
    int warm_lo=warm,warm_hi=warm;
    amrex::ParallelDescriptor::ReduceIntMin(warm_lo);amrex::ParallelDescriptor::ReduceIntMax(warm_hi);
    if(warm_lo!=warm_hi)return false;
    if(warm&&!s.FieldOrigin()->MatchesTransverse(
        Const(s.endpoint[int(EndpointPart::TransverseHigh)]),
        Const(s.endpoint[int(EndpointPart::TransverseLow)])))return false;
    bool const have_previous=s.state.endpoint_origin_ready;
    if(have_previous){
        Copy(s.endpoint_previous_high,Const(s.endpoint_old_high));
        Copy(s.endpoint_previous_low,Const(s.endpoint_old_low));
    }
    if(warm){
        Copy(s.endpoint_old_high,Const(s.endpoint[int(EndpointPart::TransverseHigh)]));
        Copy(s.endpoint_old_low,Const(s.endpoint[int(EndpointPart::TransverseLow)]));
        Copy(old,Const(s.endpoint_old_high));
    }else{
        Copy(s.endpoint_old_high,Const(old));
        for(auto* f:s.endpoint_old_low)f->setVal(0.);
    }
    if(keep_previous&&have_previous)Copy(previous,Const(s.endpoint_previous_high));
    s.state.endpoint_origin_ready=true;s.state.endpoint_previous_ready=have_previous;
    s.state.endpoint_origin_paired=warm;s.state.endpoint_history_captured=true;return true;
}
bool NativePairedDarwinFields::EndpointHistory(EndpointHistoryView& result)const {
    auto const& s=*m_impl;
    int pre_min=s.endpoint_pre_reclosure,pre_max=pre_min;
    amrex::ParallelDescriptor::ReduceIntMin(pre_min);amrex::ParallelDescriptor::ReduceIntMax(pre_max);
    if(pre_min!=pre_max)return false;
    // S1 recovery uses its accepted transverse pair without rotating the
    // registered old/previous F references. No physical E_L is a multiplier.
    if(s.endpoint_pre_reclosure){
        bool ready=s.role==Role::Endpoint&&s.state.endpoint_step_open&&!s.state.endpoint_history_captured&&
            s.endpoint_prior&&s.state.phase==Phase::Idle;
        amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
        CV high{&s.endpoint_pre_high[0],&s.endpoint_pre_high[1],&s.endpoint_pre_high[2]};
        CV low{&s.endpoint_pre_low[0],&s.endpoint_pre_low[1],&s.endpoint_pre_low[2]};
        if(!s.endpoint_prior->MatchesTransverse(high,low))return false;
        EndpointHistoryView v;v.high=high;v.low=low;v.paired=true;result=v;return true;
    }
    bool ready=s.role==Role::Endpoint&&s.state.endpoint_step_open&&s.state.endpoint_history_captured;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    int paired=s.state.endpoint_origin_paired,minimum=paired,maximum=paired;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
    if(minimum!=maximum)return false;
    if(paired){
        ready=bool(s.FieldOrigin());amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
        if(!s.FieldOrigin()->MatchesTransverse(Const(s.endpoint_old_high),Const(s.endpoint_old_low)))return false;
    }else{
        bool finite=true;for(auto const* f:s.endpoint_old_low){
            finite=f->is_finite(0,1,0,true)&&finite;
            finite=f->norminf(0,0,true)==0.&&finite;
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(finite);if(!finite)return false;
    }
    EndpointHistoryView value;value.high=Const(s.endpoint_old_high);value.low=Const(s.endpoint_old_low);
    value.previous_high=Const(s.endpoint_previous_high);value.previous_low=Const(s.endpoint_previous_low);
    value.paired=paired!=0;value.previous_ready=s.state.endpoint_previous_ready;result=value;return true;
}
bool NativePairedDarwinFields::MatchesEndpointOrigin(CV const& high)const {
    EndpointHistoryView history;if(!EndpointHistory(history))return false;
    bool ready=true;for(auto const* f:high)ready=bool(f)&&ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    for(int c=0;c<3;++c){bool const same=EndpointSameValid(*high[c],*history.high[c]);ready=same&&ready;}
    return ready;
}
bool NativePairedDarwinFields::PublishEndpointAuxiliary(){
    auto& s=*m_impl;
    bool ready=(EndpointPending()||s.state.phase==Phase::EndpointSourcePending)&&s.accepted_consumers&&!s.state.endpoint_auxiliary_ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!MatchesEndpoint(EndpointLease()))return false;
    if(!s.endpoint_draft->PublishAuxiliary(s.Sim(),s.endpoint_auxiliary_high,s.endpoint_auxiliary_low))return false;
    s.state.endpoint_auxiliary_ready=true;return true;
}
bool NativePairedDarwinFields::EndpointAuxiliary(EndpointReceipt const& receipt,EndpointAuxiliaryView& out)const {
    auto const& s=*m_impl;
    bool ready=s.role==Role::Endpoint&&s.accepted_consumers&&s.state.endpoint_auxiliary_ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!MatchesEndpoint(receipt))return false;
    if(!s.CurrentEndpoint()->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    EndpointAuxiliaryView value{Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)};
    out=value;return true;
}
bool NativePairedDarwinFields::CanAcceptEndpoint(EndpointReceipt const& receipt){
    auto& s=*m_impl;s.endpoint_accept_ready=false;
    bool ready=EndpointPending()&&s.accepted_consumers&&s.state.endpoint_step_open&&
        s.state.endpoint_history_captured&&s.state.endpoint_auxiliary_ready&&
        s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<=std::numeric_limits<std::uint64_t>::max()-2;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!MatchesEndpoint(receipt))return false;
    if(!s.endpoint_draft->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    auto const result=s.endpoint_draft->Check(s.Sim(),EndpointFields());
    ready=result.compatible&&result.converged;amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    s.endpoint_accept_ready=ready;return ready;
}
void NativePairedDarwinFields::AcceptEndpoint()noexcept {
    auto& s=*m_impl;
    AMREX_ALWAYS_ASSERT(EndpointPending()&&s.state.endpoint_step_open&&s.endpoint_accept_ready);
    s.RevokeEndpoint();s.endpoint_prior.reset();s.endpoint_field_origin.reset();s.state.phase=Phase::EndpointAccepted;
    s.state.endpoint_step_open=false;s.state.endpoint_boundary_bound=false;
    s.endpoint_accept_ready=false;s.endpoint_token->active=!s.endpoint_token->exhausted;
}
bool NativePairedDarwinFields::BindAcceptedEndpointBoundary(){
    auto& s=*m_impl;
    bool ready=EndpointAccepted()&&!s.state.endpoint_step_open&&!s.state.endpoint_boundary_bound&&
        bool(s.endpoint_draft)&&s.endpoint_token&&!s.endpoint_token->exhausted&&
        s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max()&&
        s.Sim().gett_new(0)==s.state.time&&std::uint64_t(s.Sim().getistep(0))==s.state.epoch;
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!s.endpoint_draft->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    if(!s.endpoint_draft->BindAcceptedBoundary(s.Sim(),EndpointFields()))return false;
    s.RevokeEndpoint();s.state.endpoint_boundary_bound=true;s.endpoint_token->active=true;return true;
}
bool NativePairedDarwinFields::RedistributeAcceptedEndpoint(bool local,
    amrex::IntVect const& max_cells,bool sort,amrex::IntVect const& bins,
    bool deposition,amrex::IntVect const& index_type)
{
    auto& s=*m_impl;
    bool ready=EndpointAccepted()&&!s.state.endpoint_step_open&&s.state.endpoint_boundary_bound&&
        s.state.endpoint_auxiliary_ready&&s.accepted_consumers&&s.endpoint_draft&&s.endpoint_token&&
        !s.endpoint_token->exhausted&&s.endpoint_token->generation<std::numeric_limits<std::uint64_t>::max();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    if(!MatchesEndpoint(EndpointLease())||
       !s.endpoint_draft->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    // No borrowed lease is valid across a mutation, including a failed one.
    s.RevokeEndpoint();
    if(!s.endpoint_draft->RedistributeAccepted(s.Sim(),EndpointFields(),local,max_cells,
        sort,bins,deposition,index_type))return false;
    if(!s.endpoint_draft->MatchesAuxiliary(s.Sim(),Const(s.endpoint_auxiliary_high),Const(s.endpoint_auxiliary_low)))return false;
    s.endpoint_token->active=true;return true;
}
bool RedistributeNativeAcceptedEndpoint(WarpX& w,bool local,amrex::IntVect const& max_cells,
    bool sort,amrex::IntVect const& bins,bool deposition,amrex::IntVect const& index_type)
{
    auto* owner=NativeEndpointPairedFields(w);
    bool ready=NativeEndpointPairSelected(w)&&owner&&NativePairedDarwinFields::EndpointRequested()&&
        NativePairedDarwinFields::EndpointAcceptanceRequested();
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
    return owner->RedistributeAcceptedEndpoint(local,max_cells,sort,bins,deposition,index_type);
}
NativeVacuumConstraintResult NativePairedDarwinFields::CheckEndpoint(MF const& density,MF const& mask,V const* transverse){
    if(!MatchesEndpoint(EndpointLease()))return {};
    auto& s=*m_impl;auto& w=s.Sim();
    int lo=transverse?1:0,hi=lo;amrex::ParallelDescriptor::ReduceIntMin(lo);amrex::ParallelDescriptor::ReduceIntMax(hi);
    if(lo!=hi)return {};
    bool valid=EndpointSameValid(density,*w.m_fields.get(FieldType::rho_fp,0));
    bool const masks=EndpointSameValid(mask,*w.m_fields.get("hybrid_rho_vacmask_fp",0));valid=masks&&valid;
    if(transverse){
        bool layout=true;for(int c=0;c<3;++c)layout=bool((*transverse)[c])&&layout;
        amrex::ParallelDescriptor::ReduceBoolAnd(layout);if(!layout)return {};
        for(int c=0;c<3;++c){bool const same=EndpointSameValid(*(*transverse)[c],*s.endpoint[int(EndpointPart::TransverseHigh)][c]);valid=same&&valid;}
    }
    if(!valid)return {};
    return s.CurrentEndpoint()->Check(w,EndpointFields());
}
bool NativePairedDarwinFields::CopyEndpointTransverse(V const& out){
    auto& s=*m_impl;
    int initial=InitialEndpointPending()?1:0,minimum=initial,maximum=initial;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
    if(minimum!=maximum)return false;
    if(initial){
        bool ready=s.accepted_consumers&&bool(s.endpoint_draft);
        amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
        if(!s.endpoint_draft->Matches(s.Sim(),EndpointFields()))return false;
    }else if(!MatchesEndpoint(EndpointLease()))return false;
    auto const high=s.endpoint[int(EndpointPart::TransverseHigh)];
    bool valid=true;
    for(int c=0;c<3;++c)valid=out[c]&&out[c]->nComp()==1&&out[c]->boxArray()==high[c]->boxArray()&&
        out[c]->DistributionMap()==high[c]->DistributionMap()&&valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);if(!valid)return false;
    for(int c=0;c<3;++c){
        for(int d=0;d<c;++d)valid=!EndpointShares(*out[c],*out[d])&&valid;
        for(auto const* f:s.allocations)valid=!EndpointShares(*out[c],*f)&&valid;
        // A private transverse destination must not share any registered
        // physical, current/history, proof or paired allocation. Checking a
        // short named list misses valid-layout current_fp/Jp make_alias views.
        for(auto const& name:s.Sim().m_fields.list())
            valid=!EndpointShares(*out[c],*s.Sim().m_fields.internal_get(name))&&valid;
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);if(!valid)return false;
    for(int c=0;c<3;++c)MF::Copy(*out[c],*high[c],0,0,1,0);
    return true;
}
void NativePairedDarwinFields::BeginField(R dt,R theta,R time,std::uint64_t epoch){
    auto& s=*m_impl;bool valid=s.role==Role::FiniteStage&&s.state.phase!=Phase::FiniteStage&&theta==.5&&std::isfinite(dt)&&dt>0.&&std::isfinite(time);
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,"Invalid paired finite-stage lifecycle");
    for(auto* field:s.allocations)field->setVal(0.);
    s.state={Phase::FiniteStage,dt,time,epoch,0,false,false,{}};
    auto const& origin=*s.Sim().m_fields.get("hybrid_phi_darwin_fp",0);
    MF::Copy(*s.phi_origin,origin,0,0,1,origin.nGrowVect());
    Copy(s.old_a,Const(s.Sim().m_fields.get_alldirs("hybrid_A_old_fp",0)));
}
void NativePairedDarwinFields::SaveOuter(){auto& s=*m_impl;s.RequireActive();Copy(s.old_el_low,Const(s.el_low));MF::Copy(*s.old_phi_low,*s.phi_low,0,0,1,s.phi_low->nGrowVect());}
NativePairedDarwinFields::CV NativePairedDarwinFields::ConstraintDefect(V const& raw,CV const& held){
    auto& s=*m_impl;s.RequireActive();
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*s.defect[c]);mfi.isValid();++mfi){auto o=s.defect[c]->array(mfi);auto a=raw[c]->const_array(mfi),b=held[c]->const_array(mfi),l=s.old_el_low[c]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){o(i,j,k)=dd::Add({a(i,j,k),0.},{-b(i,j,k),-l(i,j,k)}).hi;});}
    LongImages(s.Sim(),s.defect);return Const(s.defect);
}
void NativePairedDarwinFields::ApplySchur(MF const& delta_phi,CV const& delta,CV const& held,MF const& held_phi,R omega,V const& transverse){
    auto& s=*m_impl;s.RequireActive();AMREX_ALWAYS_ASSERT(omega>0.&&omega<=1.);
    auto& phi=*s.Sim().m_fields.get("hybrid_phi_darwin_fp",0);
    for(amrex::MFIter mfi(phi);mfi.isValid();++mfi){auto h=phi.array(mfi),l=s.phi_low->array(mfi);auto old=held_phi.const_array(mfi),ol=s.old_phi_low->const_array(mfi),d=delta_phi.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto y=dd::Add({old(i,j,k),ol(i,j,k)},dd::Multiply({d(i,j,k),0.},omega));h(i,j,k)=y.hi;l(i,j,k)=y.lo;});}
    ScalarImages(s.Sim(),phi);ScalarImages(s.Sim(),*s.phi_low);
    auto out=s.EL();auto spacing=s.Sim().Geom(0).InvCellSizeArray();
    for(int c=0;c<3;++c){
#if defined(WARPX_DIM_RZ)
        int axis=c==0?0:c==2?1:-1;
#else
        int axis=c;
#endif
        R inverse=axis<0?0.:spacing[axis];
        for(amrex::MFIter mfi(*out[c]);mfi.isValid();++mfi){auto h=out[c]->array(mfi),l=s.el_low[c]->array(mfi),t=transverse[c]->array(mfi),tl=s.t_low[c]->array(mfi);auto old=held[c]->const_array(mfi),ol=s.old_el_low[c]->const_array(mfi),d=delta[c]->const_array(mfi),p=delta_phi.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                auto change=dd::Multiply(dd::WithHigh(dd::Gradient(p,i,j,k,axis,inverse),d(i,j,k)),omega);
                auto before=dd::Pair{old(i,j,k),ol(i,j,k)};auto after=dd::Add(before,change);
                auto shift=dd::Add(before,dd::Negate(after));auto next_t=dd::Add({t(i,j,k),tl(i,j,k)},shift);
                h(i,j,k)=after.hi;l(i,j,k)=after.lo;t(i,j,k)=next_t.hi;tl(i,j,k)=next_t.lo;
            });}
    }
    LongImages(s.Sim(),out);LongImages(s.Sim(),s.el_low);
    s.state.magnetic_ready=false;s.state.current_ready=false;
}
void NativePairedDarwinFields::ConditionTransverse(){auto& s=*m_impl;s.RequireActive();Copy(s.t_image,Const(s.t_low));ElectricImages(s.Sim(),s.t_image);}
void NativePairedDarwinFields::BuildPotential(R interval){
    auto& s=*m_impl;s.RequireActive();AMREX_ALWAYS_ASSERT(interval==.5*s.state.dt);
    auto A=s.Sim().m_fields.get_alldirs("hybrid_A_fp",0);auto high=s.Electric();
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*A[c]);mfi.isValid();++mfi){auto a=A[c]->array(mfi),al=s.a_low[c]->array(mfi);auto a0=s.old_a[c]->const_array(mfi),e=high[c]->const_array(mfi),l=s.t_image[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto v=dd::Add({a0(i,j,k),0.},dd::Multiply({e(i,j,k),l(i,j,k)},-interval));a(i,j,k)=v.hi;al(i,j,k)=v.lo;});}
    // Exact native branch here: no external/EB/PMC means A-boundary is a no-op.
    ++s.state.trials;s.state.magnetic_ready=false;s.state.current_ready=false;
}
void NativePairedDarwinFields::BuildMagnetic(){
    auto& s=*m_impl;s.RequireActive();auto A=s.Sim().m_fields.get_alldirs("hybrid_A_fp",0),B=s.Sim().m_fields.get_alldirs(FieldType::Bfield_fp,0),Bs=s.Sim().m_fields.get_alldirs("hybrid_B_static_fp",0);
    s.Curl(A,B);s.Curl(s.a_low,s.b_scratch);
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*B[c]);mfi.isValid();++mfi){auto h=B[c]->array(mfi),l=s.b_low[c]->array(mfi);auto small=s.b_scratch[c]->const_array(mfi),fixed=Bs[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto v=dd::Add(dd::Sum(h(i,j,k),small(i,j,k)),{fixed(i,j,k),0.});h(i,j,k)=v.hi;l(i,j,k)=v.lo;});}
    Sync(s.Sim(),B);Sync(s.Sim(),s.b_low);s.state.magnetic_ready=true;s.state.current_ready=false;
}
bool NativePairedDarwinFields::OwnsMagnetic(V const& a)const noexcept{auto expected=m_impl->Sim().m_fields.get_alldirs(FieldType::Bfield_fp,0);return a==expected;}
void NativePairedDarwinFields::RefreshAmpere(V const& current){
    auto& s=*m_impl;s.RequireActive();AMREX_ALWAYS_ASSERT(s.state.magnetic_ready);
    s.Ampere(s.Sim().m_fields.get_alldirs(FieldType::Bfield_fp,0),current);s.Ampere(s.b_low,s.c_scratch);
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*s.c_low[c]);mfi.isValid();++mfi){auto h=current[c]->array(mfi),l=s.c_low[c]->array(mfi);auto small=s.c_scratch[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto v=dd::Sum(h(i,j,k),small(i,j,k));h(i,j,k)=v.hi;l(i,j,k)=v.lo;});}
    s.state.current_ready=true;
}
void NativePairedDarwinFields::BuildDisplacement(){
    auto& s=*m_impl;s.RequireActive();auto high=s.EL(),old=s.Sim().m_fields.get_alldirs("hybrid_E_long_old_fp",0),stage=s.Sim().m_fields.get_alldirs("diagnostic_D_stage_fp",0);R factor=PhysConst::epsilon_0/(.5*s.state.dt);
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*s.d_low[c]);mfi.isValid();++mfi){auto dh=stage[c]->array(mfi),dl=s.d_low[c]->array(mfi);auto e=high[c]->const_array(mfi),l=s.el_low[c]->const_array(mfi),o=old[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto v=dd::Multiply(dd::Add({e(i,j,k),l(i,j,k)},{-o(i,j,k),0.}),factor);dh(i,j,k)=v.hi;dl(i,j,k)=v.lo;});}
}
void NativePairedDarwinFields::SubtractDisplacement(V const& current,V const* thermal_remainder){
    auto& s=*m_impl;s.RequireActive();AMREX_ALWAYS_ASSERT(s.state.current_ready);
    if(thermal_remainder) {
        bool valid=warpx::thermal::remainder::ArithmeticSupported();for(int c=0;c<3;++c)valid=valid && (*thermal_remainder)[c] &&
            warpx::thermal::remainder::Layout(*(*thermal_remainder)[c],*current[c],current[c]->nGrow()) &&
            !warpx::thermal::remainder::Overlap(*(*thermal_remainder)[c],*current[c]);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(warpx::thermal::remainder::All(valid),"Invalid paired thermal current output");
    }
    BuildDisplacement();auto stage=s.Sim().m_fields.get_alldirs("diagnostic_D_stage_fp",0);
    if(!thermal_remainder) {
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*s.c_low[c]);mfi.isValid();++mfi){auto j=current[c]->array(mfi);auto cl=s.c_low[c]->const_array(mfi),dh=stage[c]->const_array(mfi),dl=s.d_low[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int k,int l){j(i,k,l)=dd::Add({j(i,k,l),cl(i,k,l)},{-dh(i,k,l),-dl(i,k,l)}).hi;});}
    } else {
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*s.c_low[c]);mfi.isValid();++mfi){auto j=current[c]->array(mfi);auto cl=s.c_low[c]->const_array(mfi),dh=stage[c]->const_array(mfi),dl=s.d_low[c]->const_array(mfi);
        bool const keep=thermal_remainder!=nullptr;
        auto const low=keep?(*thermal_remainder)[c]->array(mfi):amrex::Array4<R>{};
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int k,int l){
            auto value=dd::Add({j(i,k,l),cl(i,k,l)},{-dh(i,k,l),-dl(i,k,l)});
            j(i,k,l)=value.hi;if(keep)low(i,k,l)=value.lo;
        });}
    }
    s.state.current_ready=false;
}
void NativePairedDarwinFields::BuildCorrelatedRate(CV const& cdot,CV const& delta_ion,CV const& old_displacement,V const& rate,V const& displacement){
    auto& s=*m_impl;s.RequireActive();BuildDisplacement();R inv=1./s.state.dt;
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*rate[c]);mfi.isValid();++mfi){auto out=rate[c]->array(mfi),low=s.rate_low[c]->array(mfi);auto C=cdot[c]->const_array(mfi),Cl=s.cdot_low[c]->const_array(mfi),di=delta_ion[c]->const_array(mfi),d=displacement[c]->const_array(mfi),dl=s.d_low[c]->const_array(mfi),d0=old_displacement[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto dv=dd::Multiply(dd::Add({d(i,j,k),dl(i,j,k)},{-d0(i,j,k),0.}),2.*inv);
            auto v=dd::Add(dd::Add({C(i,j,k),Cl(i,j,k)},dd::Multiply({di(i,j,k),0.},-inv)),dd::Negate(dv));
            out(i,j,k)=v.hi;low(i,j,k)=v.lo;
        });}
}
void NativePairedDarwinFields::CompletePECStageRateLow(){
    auto& s=*m_impl;s.RequireActive();
    SetNativePECTangential(s.Sim(),s.rate_low,nullptr,0.);Sync(s.Sim(),s.rate_low);
}
void NativePairedDarwinFields::AssembleTotalElectric(){
    auto& s=*m_impl;s.RequireActive();auto total=s.Electric(),longitudinal=s.EL();
    Copy(s.t_high,Const(total));
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*total[c]);mfi.isValid();++mfi){auto e=total[c]->array(mfi),lo=s.total_low[c]->array(mfi),hi=s.total_high[c]->array(mfi);auto tl=s.t_image[c]->const_array(mfi),lh=longitudinal[c]->const_array(mfi),ll=s.el_low[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto value=dd::Add({e(i,j,k),tl(i,j,k)},{lh(i,j,k),ll(i,j,k)});e(i,j,k)=value.hi;hi(i,j,k)=value.hi;lo(i,j,k)=value.lo;});}
}
void NativePairedDarwinFields::FormTransverseOhmRHS(bool grouped){
    auto& s=*m_impl;s.RequireActive();auto result=s.Electric(),longitudinal=s.EL();
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(*result[c]);mfi.isValid();++mfi){auto e=result[c]->array(mfi);auto tl=s.t_low[c]->const_array(mfi),lh=longitudinal[c]->const_array(mfi),ll=s.el_low[c]->const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto v=dd::Pair{e(i,j,k),0.};if(!grouped)v=dd::Add(v,{-lh(i,j,k),0.});
            e(i,j,k)=dd::Add(v,dd::Add({-ll(i,j,k),0.},{-tl(i,j,k),0.})).hi;
        });}
}
void NativePairedDarwinFields::Extrapolate(R theta,V const& transverse,CV const& old_transverse){
    auto& s=*m_impl;s.RequireActive();AMREX_ALWAYS_ASSERT(theta==.5);
    auto L=s.EL(),L0=s.Sim().m_fields.get_alldirs("hybrid_E_long_old_fp",0),A=s.Sim().m_fields.get_alldirs("hybrid_A_fp",0);
    for(int c=0;c<3;++c){
        // Solver vectors own valid cells only; use low's valid alias for Th.
        for(amrex::MFIter mfi(*transverse[c]);mfi.isValid();++mfi){auto h=transverse[c]->array(mfi),l=s.t_low[c]->array(mfi);auto o=old_transverse[c]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto v=dd::Add({o(i,j,k),0.},dd::Multiply(dd::Add({h(i,j,k),l(i,j,k)},{-o(i,j,k),0.}),1./theta));h(i,j,k)=v.hi;l(i,j,k)=v.lo;});}
        PairExtrapolate(*L[c],*s.el_low[c],*L0[c],theta);
        PairExtrapolate(*A[c],*s.a_low[c],*s.old_a[c],theta);
    }
    auto& phi=*s.Sim().m_fields.get("hybrid_phi_darwin_fp",0);PairExtrapolate(phi,*s.phi_low,*s.phi_origin,theta);
    s.state.current_ready=false;s.state.magnetic_ready=false;
}
namespace {
AMREX_GPU_HOST_DEVICE R Product(dd::Pair a,dd::Pair b){return dd::Add(dd::Multiply(a,b.hi),dd::Multiply(a,b.lo)).hi;}
// Namespace-scope value/factory types keep the reduction compatible with NVCC.
// Each case retains the previous cell expression and operation order.
enum class MeasureKind { Terminal, Magnetic, Longitudinal, Current, CurrentWork,
                         HeldElectron, ElectricWork };
struct MeasureValue {
    amrex::Array4<R const> a,b,c,d;
    R factor;
    MeasureKind kind;
    AMREX_GPU_HOST_DEVICE R operator()(int i,int j,int k) const noexcept {
        switch(kind){
        case MeasureKind::Terminal: return -factor*a(i,j,k)*b(i,j,k);
        case MeasureKind::Magnetic: {
            dd::Pair old{a(i,j,k),b(i,j,k)},now{c(i,j,k),0.};
            return Product(dd::Add(now,dd::Negate(old)),dd::Add(now,old))/(2.*PhysConst::mu0);
        }
        case MeasureKind::Longitudinal: return -PhysConst::epsilon_0*b(i,j,k)*(a(i,j,k)+.5*b(i,j,k));
        case MeasureKind::Current: return dd::Add({c(i,j,k),0.},{-a(i,j,k),-b(i,j,k)}).hi;
        case MeasureKind::CurrentWork: return factor*d(i,j,k)*dd::Add({c(i,j,k),0.},{-a(i,j,k),-b(i,j,k)}).hi;
        case MeasureKind::HeldElectron: return (b(i,j,k)-a(i,j,k))*c(i,j,k);
        case MeasureKind::ElectricWork: return factor*c(i,j,k)*(a(i,j,k)-b(i,j,k));
        }
        return 0.;
    }
};
struct MeasureFields {
    MeasureKind kind;
    R factor;
    MF const *a,*b,*c=nullptr,*d=nullptr;
    MeasureValue operator()(amrex::MFIter const& mfi) const {
        return {a->const_array(mfi),b->const_array(mfi),
            c?c->const_array(mfi):amrex::Array4<R const>{},
            d?d->const_array(mfi):amrex::Array4<R const>{},factor,kind};
    }
};
template<class Function>
R Measure(MF const& layout,amrex::iMultiFab const& owner,amrex::Geometry const& geometry,Function const& function){
    auto volume=MakeQdsmcVolumeElement(geometry,layout.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> operation;amrex::ReduceData<R> data(operation);using Tuple=typename decltype(data)::Type;
    for(amrex::MFIter mfi(layout);mfi.isValid();++mfi){auto own=owner.const_array(mfi);auto f=function(mfi);
        operation.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple{return {own(i,j,k)?volume(i,j,k)*f(i,j,k):0.};});}
    R answer=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(answer);return answer;
}
}
bool NativePairedDarwinFields::CaptureStageAccounting(){
    auto& s=*m_impl;s.RequireActive();auto& p=s.state.publication;AMREX_ALWAYS_ASSERT(!p.stage_captured);
    p.finite=s.Finite();if(!p.finite)return false;
    Copy(s.stage_ji,Const(s.Sim().m_fields.get_alldirs(FieldType::current_fp,0)));
    Copy(s.stage_je,Const(s.Sim().m_fields.get_alldirs("diagnostic_Je_stage_fp",0)));
    Copy(s.stage_force_je,Const(s.Sim().m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0)));
    for(int c=0;c<3;++c)MF::Subtract(*s.stage_force_je[c],*s.stage_ji[c],0,0,1,1);
    // Apply the SAME homogeneous physical images to both terminal parts.
    // A constrained wall row contributes zero; normal plasma current is kept.
    ElectricImages(s.Sim(),s.total_high);ElectricImages(s.Sim(),s.total_low);
    auto E=s.Electric();R dt=s.state.dt;
    for(int c=0;c<3;++c){
        p.stage_ion_terminal_rounding_work+=Measure(*s.stage_ji[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::Terminal,dt,s.stage_ji[c],s.total_low[c]});
        p.stage_electron_terminal_rounding_work+=Measure(*s.stage_force_je[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::Terminal,dt,s.stage_force_je[c],s.total_low[c]});
        p.stage_inertia_terminal_rounding_work+=Measure(*s.stage_je[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::Terminal,dt,s.stage_je[c],s.ei_low[c]});
        MF::Copy(*s.saved_e[c],*s.total_high[c],0,0,1,s.saved_e[c]->nGrowVect());MF::Subtract(*s.saved_e[c],*E[c],0,0,1,s.saved_e[c]->nGrowVect());
        p.stage_assembly_mismatch=std::max(p.stage_assembly_mismatch,s.saved_e[c]->norminf());
    }
    p.finite=s.Finite()&&std::isfinite(p.stage_ion_terminal_rounding_work)&&
        std::isfinite(p.stage_electron_terminal_rounding_work)&&
        std::isfinite(p.stage_inertia_terminal_rounding_work)&&std::isfinite(p.stage_assembly_mismatch);
    amrex::ParallelDescriptor::ReduceBoolAnd(p.finite);
    p.stage_captured=p.finite;return p.stage_captured;
}
NativePairedDarwinFields::Publication NativePairedDarwinFields::MaterializeEndpoint(){
    auto& s=*m_impl;s.RequireActive();auto& p=s.state.publication;AMREX_ALWAYS_ASSERT(p.stage_captured&&!p.materialized);
    p.finite=s.Finite();if(!p.finite)return p;
    auto B=s.Sim().m_fields.get_alldirs(FieldType::Bfield_fp,0),C=s.Sim().m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    Copy(s.saved_b,Const(B));Copy(s.saved_c,Const(C));Copy(s.saved_e,Const(s.Electric()));
    Copy(s.saved_th,Const(s.t_high));Copy(s.saved_ah,Const(s.Sim().m_fields.get_alldirs("hybrid_A_fp",0)));Copy(s.saved_lh,Const(s.EL()));
    MF::Copy(*s.saved_phi,*s.Sim().m_fields.get("hybrid_phi_darwin_fp",0),0,0,1,s.saved_phi->nGrowVect());
    MF::Copy(*s.saved_phi_low,*s.phi_low,0,0,1,s.saved_phi_low->nGrowVect());
    Copy(s.saved_bl,Const(s.b_low));Copy(s.saved_cl,Const(s.c_low));Copy(s.saved_ll,Const(s.el_low));Copy(s.saved_tl,Const(s.t_low));Copy(s.saved_al,Const(s.a_low));Copy(s.saved_dl,Const(s.d_low));Copy(s.saved_rl,Const(s.rate_low));
    Copy(s.held_je,Const(s.Sim().m_fields.get_alldirs("diagnostic_Je_endpoint_fp",0)));
    for(int c=0;c<3;++c){p.longitudinal_low[c]=s.el_low[c]->norminf();p.transverse_low[c]=s.t_low[c]->norminf();p.potential_low[c]=s.a_low[c]->norminf();p.displacement_low[c]=s.d_low[c]->norminf();p.rate_low[c]=s.rate_low[c]->norminf();}
    p.scalar_low=s.phi_low->norminf();auto L=s.EL();
    auto A=s.Sim().m_fields.get_alldirs("hybrid_A_fp",0),Bs=s.Sim().m_fields.get_alldirs("hybrid_B_static_fp",0);
    s.Curl(A,B);for(int c=0;c<3;++c)MF::Add(*B[c],*Bs[c],0,0,1,B[c]->nGrowVect());Sync(s.Sim(),B);s.Ampere(B,C);
    for(int c=0;c<3;++c){
        p.magnetic_inventory_change+=Measure(*B[c],*s.magnetic_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::Magnetic,0.,s.saved_b[c],s.saved_bl[c],B[c]});
        p.longitudinal_inventory_change+=Measure(*L[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::Longitudinal,0.,L[c],s.saved_ll[c]});
        p.signed_current_change[c]=Measure(*C[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::Current,0.,s.saved_c[c],s.saved_cl[c],C[c]});
        p.endpoint_current_quadrature_change+=Measure(*C[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::CurrentWork,s.state.dt,s.saved_c[c],s.saved_cl[c],C[c],s.saved_e[c]});
        // Same pair-to-high difference and valid rows as the signed inventory.
        for(amrex::MFIter mfi(*B[c]);mfi.isValid();++mfi){auto out=s.b_scratch[c]->array(mfi);auto before=s.saved_b[c]->const_array(mfi),lo=s.saved_bl[c]->const_array(mfi),after=B[c]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=dd::Add({after(i,j,k),0.},{-before(i,j,k),-lo(i,j,k)}).hi;});}
        for(amrex::MFIter mfi(*C[c]);mfi.isValid();++mfi){auto out=s.c_scratch[c]->array(mfi);auto before=s.saved_c[c]->const_array(mfi),lo=s.saved_cl[c]->const_array(mfi),after=C[c]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=dd::Add({after(i,j,k),0.},{-before(i,j,k),-lo(i,j,k)}).hi;});}
        p.magnetic_difference[c]=s.b_scratch[c]->norminf();p.current_difference[c]=s.c_scratch[c]->norminf();
    }
    for(auto fields:{s.el_low,s.old_el_low,s.t_low,s.t_image,s.a_low,s.b_low,s.c_low,s.d_low,s.rate_low,s.cdot_low,s.ei_low,s.total_low})for(auto* f:fields)f->setVal(0.);
    s.phi_low->setVal(0.);s.old_phi_low->setVal(0.);
    s.state.phase=Phase::Materialized;s.state.magnetic_ready=false;s.state.current_ready=false;p.materialized=true;return p;
}
NativePairedDarwinFields::Publication NativePairedDarwinFields::CompletePublication(){
    auto& s=*m_impl;auto& p=s.state.publication;AMREX_ALWAYS_ASSERT(s.state.phase==Phase::Materialized&&p.materialized&&!p.complete);
    auto E=s.Electric(),je=s.Sim().m_fields.get_alldirs("diagnostic_Je_endpoint_fp",0);R dt=s.state.dt;
    // The independent accepted Je is held through materialization. Measure its
    // actual inventory difference with the same native frozen stage mass;
    // do not infer it by reseeding from the changed Ampere current.
    for(int c=0;c<3;++c)MF::LinComb(*s.account_edge[c],.5,*je[c],0,.5,*s.held_je[c],0,0,1,0);
    auto const& model=*s.Sim().get_pointer_HybridPICModel();R scale=model.m_electron_inertia_mass/(PhysConst::q_e*PhysConst::q_e*model.m_n0_ref);
    ApplyYeeInertiaMass(s.Sim().Geom(0),*s.Sim().m_fields.get("diagnostic_inertia_kappa_nodal",0),Const(s.account_edge),s.c_scratch,scale);
    for(int c=0;c<3;++c){
        p.held_electron_inventory_change+=Measure(*je[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::HeldElectron,0.,s.held_je[c],je[c],s.c_scratch[c]});
        p.endpoint_ion_quadrature_change+=Measure(*s.stage_ji[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::ElectricWork,dt,E[c],s.saved_e[c],s.stage_ji[c]});
        p.endpoint_electron_quadrature_change+=Measure(*s.stage_force_je[c],*s.edge_owners[c],s.Sim().Geom(0),MeasureFields{MeasureKind::ElectricWork,dt,E[c],s.saved_e[c],s.stage_force_je[c]});
        MF::Copy(*s.account_edge[c],*E[c],0,0,1,0);MF::Subtract(*s.account_edge[c],*s.saved_e[c],0,0,1,0);p.electric_publication_change[c]=s.account_edge[c]->norminf();
    }
    // These endpoint quadratures are diagnostic counterfactuals, not heat or
    // source additions. U and particles are untouched by this transition.
    p.finite=s.Finite()&&p.finite;
    for(auto value:{p.magnetic_inventory_change,p.longitudinal_inventory_change,
            p.held_electron_inventory_change,p.endpoint_ion_quadrature_change,
            p.endpoint_electron_quadrature_change,p.endpoint_current_quadrature_change})
        p.finite=std::isfinite(value)&&p.finite;
    for(auto const& values:{p.magnetic_difference,p.current_difference,
            p.signed_current_change,p.electric_publication_change})
        for(auto value:values)p.finite=std::isfinite(value)&&p.finite;
    amrex::ParallelDescriptor::ReduceBoolAnd(p.finite);
    p.complete=p.finite;return p;
}
NativePairedDarwinFields::CV NativePairedDarwinFields::LongitudinalLow()const{m_impl->RequireFiniteStageRole();return Const(m_impl->el_low);}
NativePairedDarwinFields::CV NativePairedDarwinFields::PreviousLongitudinalLow()const{m_impl->RequireFiniteStageRole();return Const(m_impl->old_el_low);}
NativePairedDarwinFields::CV NativePairedDarwinFields::TransverseLow()const{m_impl->RequireFiniteStageRole();return Const(m_impl->t_low);}
NativePairedDarwinFields::CV NativePairedDarwinFields::ConditionedTransverseLow()const{m_impl->RequireFiniteStageRole();return Const(m_impl->t_image);}
NativePairedDarwinFields::CV NativePairedDarwinFields::DisplacementLow()const{m_impl->RequireFiniteStageRole();return Const(m_impl->d_low);}
NativePairedDarwinFields::CV NativePairedDarwinFields::RateLow()const{m_impl->RequireFiniteStageRole();return Const(m_impl->rate_low);}
NativePairedDarwinFields::V NativePairedDarwinFields::InertiaElectricLow(){m_impl->RequireFiniteStageRole();return m_impl->ei_low;}
NativePairedDarwinFields::V NativePairedDarwinFields::CurlRateLow(){m_impl->RequireFiniteStageRole();return m_impl->cdot_low;}
NativePairedDarwinFields::V NativePairedDarwinFields::CurlRatePotentialScratch(){m_impl->RequireFiniteStageRole();return m_impl->rate_a;}
NativePairedDarwinFields::V NativePairedDarwinFields::CurlRateMagneticScratch(){m_impl->RequireFiniteStageRole();return m_impl->rate_b;}
amrex::MultiFab const& NativePairedDarwinFields::ScalarLow()const{m_impl->RequireFiniteStageRole();return *m_impl->phi_low;}
NativePairedDarwinFields::Publication const& NativePairedDarwinFields::LastPublication()const noexcept{return m_impl->state.publication;}
} // namespace warpx::darwin
