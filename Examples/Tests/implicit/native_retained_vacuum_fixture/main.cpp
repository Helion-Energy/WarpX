/* Static joined-vacuum retained acceptance; no driver or physical event admission. */
#include "NativeBytes.H"
#include "FieldSolver/ImplicitSolvers/NativeEndpointField.H"
#include "FieldSolver/ImplicitSolvers/NativeVacuumEndpoint.H"
#include <limits>
#include <algorithm>
#include "FieldSolver/ImplicitSolvers/ImplicitFieldRollback.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/NativeAcceptedStoppingContext.H"
#include "Python/callbacks.H"
#include "Initialization/WarpXInit.H"
#include <AMReX_ParmParse.H>
#include <fstream>
#include <filesystem>
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "Diagnostics/ReducedDiags/FieldPoyntingFlux.H"
using R=amrex::Real; using P=amrex::ParticleReal; using V=amrex::GpuArray<R,3>;
using warpx::fields::FieldType; using ablastr::fields::Direction;
using namespace warpx::thermal;
R norm(V const& x){return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
RawState RegistryBytes(WarpX& w){
    RawState state;
    std::vector<amrex::MultiFab const*> tangent_scratch;
    for(int lev=0;lev<=w.finestLevel();++lev)
        for(auto type:{FieldType::MassMatrices_X,FieldType::MassMatrices_Y,FieldType::MassMatrices_Z})
            for(int c=0;c<3;++c)if(w.m_fields.has(type,Direction{c},lev))
                tangent_scratch.push_back(w.m_fields.get(type,Direction{c},lev));
    for(auto const& name:w.m_fields.list()){
        auto const* pointer=w.m_fields.internal_get(name);
        if(std::find(tangent_scratch.begin(),tangent_scratch.end(),pointer)!=tangent_scratch.end())continue;
        auto const& field=*pointer;
        for(amrex::MFIter mfi(field);mfi.isValid();++mfi)
            state[name+"/"+std::to_string(mfi.index())]=ReadBytes(field[mfi].dataPtr(),field[mfi].size());
    }
    return state;
}
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "FieldSolver/ImplicitSolvers/NativeRetainedAcceptance.H"
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include <AMReX_LayoutData.H>
#include <set>

// Explicit-instantiation access is fixture-only, with no altered class layout
// or ODR definition. It lets this independent oracle inspect pre-existing
// private vectors/flags without a production observer/accessor seam.
template<class Tag, typename Tag::type member> struct PrivateMember {
    friend typename Tag::type access(Tag) { return member; }
};
#define PRIVATE_TAG(tag, klass, typ, mem) \
struct tag { using type=typ klass::*; friend type access(tag); }; \
template struct PrivateMember<tag,&klass::mem>;
PRIVATE_TAG(FieldTag,ThetaImplicitHybrid,WarpXSolverVec,m_E)
PRIVATE_TAG(OldFieldTag,ThetaImplicitHybrid,WarpXSolverVec,m_Eold)
PRIVATE_TAG(PrevFieldTag,ThetaImplicitHybrid,WarpXSolverVec,m_Eprev)
PRIVATE_TAG(HaveOldTag,ThetaImplicitHybrid,bool,m_have_Eold)
PRIVATE_TAG(HavePrevTag,ThetaImplicitHybrid,bool,m_have_Eprev)
PRIVATE_TAG(DtTag,ImplicitSolver,amrex::Real,m_dt)
using TimeVector=amrex::Vector<amrex::Real>;
PRIVATE_TAG(OldTimeTag,WarpX,TimeVector,t_old)
PRIVATE_TAG(CurrentTag,ImplicitSolver,bool,m_mass_matrices_current)
PRIVATE_TAG(CachedTag,ImplicitSolver,bool,m_mass_matrices_cached)
using DensityVector=amrex::Vector<std::unique_ptr<MassMatrixDensityProjection>>;
PRIVATE_TAG(DensityTag,ThetaImplicitHybrid,DensityVector,m_mass_matrix_density)
using EndpointResponsePtr=std::unique_ptr<warpx::particles::NativeEndpointCurrentResponse>;
PRIVATE_TAG(EndpointResponseTag,ImplicitSolver,EndpointResponsePtr,m_endpoint_current_response)
PRIVATE_TAG(MidpointFlagTag,FieldPoyntingFlux,bool,use_mid_step_value)
#undef PRIVATE_TAG

RawState PrivateState(WarpX& w) {
    auto& solver=*dynamic_cast<ThetaImplicitHybrid*>(w.get_pointer_ImplicitSolver());
    RawState out;
    auto collect=[&](char const* name,WarpXSolverVec const& vec) {
        auto const& levels=vec.getArrayVec();
        for(std::size_t lev=0;lev<levels.size();++lev)for(int c=0;c<3;++c){
            auto const* field=levels[lev][c];if(!field)continue;
            for(amrex::MFIter mfi(*field);mfi.isValid();++mfi){auto const& fab=(*field)[mfi];
                out[std::string(name)+"/"+std::to_string(lev)+"/"+std::to_string(c)+"/"+std::to_string(mfi.index())]=ReadBytes(fab.dataPtr(),fab.size());}
        }
    };
    collect("E",solver.*access(FieldTag{}));collect("Eold",solver.*access(OldFieldTag{}));collect("Eprev",solver.*access(PrevFieldTag{}));
    out["have_old"]=ReadScalar(solver.*access(HaveOldTag{}));out["have_prev"]=ReadScalar(solver.*access(HavePrevTag{}));
    out["dt"]=ReadScalar(static_cast<ImplicitSolver&>(solver).*access(DtTag{}));
    auto const& model=*w.get_pointer_HybridPICModel();
    out["rho_captured"]=ReadScalar(model.m_inertia_rho_n_captured);out["jp_captured"]=ReadScalar(model.m_inertia_jpold_captured);
    out["plasma_valid"]=ReadScalar(model.m_qdsmc_J_plasma_valid);
    out["history_initialized"]=ReadScalar(model.m_inertia_history_initialized);
    out["history_levels"]=ReadScalar(model.m_inertia_history_levels);
    return out;
}
RawState CompleteState(WarpX& w) {
    RawState out=RegistryBytes(w);
    for(auto const& [k,v]:ParticleBytes(w))out["particles/"+k]=v;
    for(auto const& [k,v]:PrivateState(w))out["solver/"+k]=v;
    auto const& diagnostics=*w.reduced_diags;
    for(std::size_t i=0;i<diagnostics.m_multi_rd.size();++i) {
        auto const& diag=*diagnostics.m_multi_rd[i];
        auto const prefix="reduced/"+diagnostics.m_rd_names[i];
        auto const* begin=reinterpret_cast<unsigned char const*>(diag.m_data.data());
        out[prefix+"/data"]=diag.m_data.empty()?RawBytes{}:
            RawBytes(begin,begin+diag.m_data.size()*sizeof(R));
        if(auto const* poynting=dynamic_cast<FieldPoyntingFlux const*>(&diag))
            out[prefix+"/use_mid_step_value"]=ReadScalar(*poynting.*access(MidpointFlagTag{}));
    }
    out["old_time"]=ReadScalar(w.gett_old(0));out["time"]=ReadScalar(w.gett_new(0));out["step"]=ReadScalar(w.getistep(0));
    if(auto const* costs=WarpX::getCosts(0))out["costs"]=ReadBytes(costs->data(),costs->local_size());
    return out;
}
void WriteState(RawState const& state,std::string const& label) {
    auto& w=WarpX::GetInstance();
    if(w.reduced_diags->m_plot_rd) {
        auto const directory="diagnostic-"+label;
        if(amrex::ParallelDescriptor::IOProcessor())std::filesystem::create_directory(directory);
        amrex::ParallelDescriptor::Barrier();
        w.reduced_diags->WriteCheckpointData(directory);
        amrex::ParallelDescriptor::Barrier();
    }
    std::ofstream f(label+".rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".bin",std::ios::binary);
    for(auto const& [key,value]:state){std::uint64_t n=key.size(),m=value.size();f.write(reinterpret_cast<char*>(&n),sizeof(n));f.write(key.data(),n);f.write(reinterpret_cast<char*>(&m),sizeof(m));f.write(reinterpret_cast<char const*>(value.data()),m);}
}
std::set<std::uint64_t> LocalIDs(WarpX& w) {
    std::set<std::uint64_t> out;
    for(auto const& pc:w.GetPartContainer())for(auto const& [key,tile]:pc->GetParticles(0)) {
        amrex::ignore_unused(key);auto const& a=tile.GetStructOfArrays().GetIdCPUData();std::vector<std::uint64_t> v(a.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,a.begin(),a.end(),v.begin());out.insert(v.begin(),v.end());
    }
    return out;
}
void TagNativeParticles(WarpX& w) {
    // The pinned native injector owns positions, momenta, weights and RNG.
    // Add only deterministic migratable attributes for the complete-map oracle.
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()) {
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        pc.AddRealComp("retained_reference",true);pc.AddIntComp("retained_tag",true);
        for(WarpXParIter pti(pc,0);pti.isValid();++pti) {
            auto* real=pti.GetAttribs("retained_reference").data();
            auto* integer=pti.GetiAttribs("retained_tag").data();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n) {
                real[n]=n+.125;integer[n]=17+int(n);
            });
        }
    }
}


struct Validator {
    std::string fault;bool decline=false,relocate=false;int veto_rank=-1;int before=0,after=0;std::set<std::uint64_t> old_ids;amrex::Long migrated=0;
};
bool ValidateCandidate(WarpX& w,warpx::implicit::AcceptancePhase phase,void* context) {
    auto& v=*static_cast<Validator*>(context);
    if(phase==warpx::implicit::AcceptancePhase::BeforeEndpoint){
        bool check_diagnostics=false;
        amrex::ParmParse("fixture").query("diagnostics",check_diagnostics);
        if(check_diagnostics) {
            R integrated_abs=0.;int count=0;
            for(auto const& ptr:w.reduced_diags->m_multi_rd)
                if(auto const* diag=dynamic_cast<FieldPoyntingFlux const*>(ptr.get())) {
                    ++count;AMREX_ALWAYS_ASSERT(*diag.*access(MidpointFlagTag{}));
                    for(int i=2*AMREX_SPACEDIM;i<4*AMREX_SPACEDIM;++i)
                        integrated_abs+=std::abs(diag->m_data[i]);
                }
            AMREX_ALWAYS_ASSERT(count==1 && std::isfinite(integrated_abs) && integrated_abs>0.);
            amrex::Print()<<"RETAINED_MIDPOINT_DIAGNOSTIC integrated_abs="<<integrated_abs
                <<" native_sample=1\n";
        }
        ++v.before;auto ids=LocalIDs(w);amrex::Long changed=0;for(auto id:v.old_ids)changed+=!ids.contains(id);
        amrex::ParallelDescriptor::ReduceLongSum(changed);v.migrated=std::max(v.migrated,changed);
        if(v.decline && (v.fault=="density" || v.fault=="nonfinite")) {
            auto& rho=*w.m_fields.get(FieldType::rho_fp,0);
            R const bad=v.fault=="density"?-1.:std::numeric_limits<R>::quiet_NaN();
            // One owner, finite negative or explicit nonfinite valid node.
            // The numerical Try must reject before constructing its operator.
            bool selected=false;
            if(amrex::ParallelDescriptor::MyProc()==0)
                for(amrex::MFIter mfi(rho);mfi.isValid()&&!selected;++mfi) {
                    selected=true;auto arr=rho.array(mfi);auto const box=mfi.validbox();auto index=box.smallEnd();
                    amrex::ParallelFor(1,[=] AMREX_GPU_DEVICE(int){arr(index[0],index[1],0,0)=bad;});
                }
        }
        return !v.decline || v.fault!="before" ||
            (v.veto_rank>=0 && amrex::ParallelDescriptor::MyProc()!=v.veto_rank);
    }
    ++v.after;
    if(v.decline && v.relocate) {
        // Deliberately change the covered candidate map AFTER physical endpoint
        // verification. This rejected-only injection tests complete ownership
        // rollback; it is not an accepted orbit crossing qualification.
        auto& pc=w.GetPartContainer().GetParticleContainerFromName("ions");
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){
            auto* radius=pti.GetAttribs(PIdx::r).data();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){if(radius[n]<.25){radius[n]+=.25;}});
        }
        pc.Redistribute();
        auto ids=LocalIDs(w);amrex::Long changed=0;
        for(auto id:v.old_ids)changed+=!ids.contains(id);
        amrex::ParallelDescriptor::ReduceLongSum(changed);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(changed>0,"Injected redistribution did not change rank ownership");
        v.migrated=std::max(v.migrated,changed);
        amrex::Print()<<"RETAINED_INJECTED_REDISTRIBUTION migrated="<<changed<<" accepted_physical_crossing=0\n";
    }
    return !v.decline || (v.veto_rank>=0 && amrex::ParallelDescriptor::MyProc()!=v.veto_rank);
}
int main(int argc,char** argv) {
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        InstallPythonCallback("beforeInitEsolve",[](){TagNativeParticles(WarpX::GetInstance());});
        auto& w=WarpX::GetInstance();w.InitData();w.HybridPICInitializeRhoJandB();
        bool preflight=false;amrex::ParmParse("fixture").query("preflight",preflight);
        if(preflight) {
            auto& theta=*dynamic_cast<ThetaImplicitHybrid*>(w.get_pointer_ImplicitSolver());
            auto const& baseline=(theta.*access(FieldTag{})).getArrayVec()[0];
            ablastr::fields::VectorField fields{baseline[0],baseline[1],baseline[2]};
            auto original=CompleteState(w);
            for(int fault=0;fault<5;++fault) {
                warpx::implicit::FieldRollback saved;saved.Capture(w.m_fields);
                auto* target=fault<2?w.m_fields.get(FieldType::rho_fp,0):
                    w.m_fields.get("hybrid_rho_vacmask_fp",0);
                if(fault<4)target->setVal(fault%2?-std::numeric_limits<R>::infinity():-1.,0,1,0);
                auto before=CompleteState(w);
                R const time=fault==4?std::numeric_limits<R>::quiet_NaN():w.gett_new(0);
                bool const ok=TryConstrainNativeVacuumEndpointField(w,time,false,fields);
                AMREX_ALWAYS_ASSERT(!ok && CompleteState(w)==before);
                AMREX_ALWAYS_ASSERT(saved.Restore(w.m_fields));saved.Discard();
                AMREX_ALWAYS_ASSERT(CompleteState(w)==original);
            }
            auto wrong=fields;wrong[0]=nullptr;
            AMREX_ALWAYS_ASSERT(!TryConstrainNativeVacuumEndpointField(w,w.gett_new(0),false,wrong));
            AMREX_ALWAYS_ASSERT(CompleteState(w)==original);
            amrex::Print()<<"RETAINED_VACUUM_PREFLIGHT_PASS density=2 mask=2 time=1 layout=1 exact=1\n";
        }
        bool reject=false,seam=false;int steps=1;std::string fault="decision";
        amrex::ParmParse pp("fixture");pp.query("reject",reject);pp.query("seam",seam);pp.query("steps",steps);pp.query("fault",fault);
        int reject_step=0;pp.query("reject_step",reject_step);
        auto* solver=w.get_pointer_ImplicitSolver();Validator validator;validator.fault=fault;validator.relocate=seam;pp.query("veto_rank",validator.veto_rank);
        warpx::implicit::SetNativeAcceptanceValidator({ValidateCandidate,&validator});
        int rejected=0;
        for(int step=0;step<steps;++step){
            R const time=w.gett_new(0),dt=w.getdt(0);int const initial_step=w.getistep(0);
            validator.old_ids=LocalIDs(w);auto before=CompleteState(w);WriteState(before,"old_"+std::to_string(step));
            if(reject && step==reject_step){
                int const pc_stamp_before=w.get_pointer_HybridPICModel()->m_esolve_pc_stamp;
                validator.decline=true;int const status=solver->OneStep(time,dt,initial_step);
                int const expected=(fault=="density"||fault=="nonfinite")?-11:-12;AMREX_ALWAYS_ASSERT(status==expected);
                auto restored=CompleteState(w);WriteState(restored,"restored_"+std::to_string(step));bool exact=before==restored;
                if(!exact)for(auto const& [k,v]:before){auto it=restored.find(k);if(it==restored.end()||it->second!=v)amrex::AllPrint()<<"ROLLBACK_DIFFERENCE rank="<<amrex::ParallelDescriptor::MyProc()<<" key="<<k<<"\n";}
                amrex::ParallelDescriptor::ReduceBoolAnd(exact);AMREX_ALWAYS_ASSERT(exact);
                int const pc_stamp_after=w.get_pointer_HybridPICModel()->m_esolve_pc_stamp;
                AMREX_ALWAYS_ASSERT(pc_stamp_after>=pc_stamp_before);
                amrex::Print()<<"RETAINED_DERIVED_PC_STAMP before="<<pc_stamp_before
                    <<" after_restore="<<pc_stamp_after<<" checkpoint_history_exact=1\n";
                auto& theta=*dynamic_cast<ThetaImplicitHybrid*>(solver);
                AMREX_ALWAYS_ASSERT(!(static_cast<ImplicitSolver&>(theta).*access(CurrentTag{})) && !(static_cast<ImplicitSolver&>(theta).*access(CachedTag{})));
                for(auto const& density:theta.*access(DensityTag{}))AMREX_ALWAYS_ASSERT(!density);
                auto const& response=static_cast<ImplicitSolver&>(theta).*access(EndpointResponseTag{});
                AMREX_ALWAYS_ASSERT(!response || !response->IsValid());
                ++rejected;amrex::Print()<<"RETAINED_ROLLBACK_PASS status="<<status<<" step="<<step<<" raw_fields_particles_rng_private_time_exact=1 caches_invalid=1 migrated="<<validator.migrated<<"\n";
                validator.decline=false;
            }
            int const status=solver->OneStep(time,dt,initial_step);AMREX_ALWAYS_ASSERT(status>=0);
            AMREX_ALWAYS_ASSERT(w.gett_new(0)==time&&w.getistep(0)==initial_step);
            WriteState(CompleteState(w),"accepted_"+std::to_string(step));
            // The driver publishes time/step only after success, as Evolve does.
            (w.*access(OldTimeTag{}))[0]=time;w.sett_new(0,time+dt);w.setistep(0,initial_step+1);
        }
        if(seam && reject)AMREX_ALWAYS_ASSERT_WITH_MESSAGE(validator.migrated>0,"Seam fixture did not exercise a rank ownership change");
        amrex::Print()<<"RETAINED_VACUUM_ACCEPTANCE_PASS steps="<<steps<<" rejected="<<rejected<<" before_calls="<<validator.before<<" after_calls="<<validator.after<<" migrated="<<validator.migrated<<"\n";
        warpx::implicit::SetNativeAcceptanceValidator({});WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();return 0;
}
