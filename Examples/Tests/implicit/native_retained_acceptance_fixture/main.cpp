/* Native retained-acceptance fixture; no live stopping/event guard is enabled. */
#include "NativeBytes.H"
#include "FieldSolver/ImplicitSolvers/NativeEndpointField.H"
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
    for(auto const& name:w.m_fields.list()){
        auto const& field=*w.m_fields.internal_get(name);
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
void LoadRetainedParticles(WarpX& w){
    auto const dx=w.Geom(0).CellSizeArray();auto const lengths=w.Geom(0).Domain().length();
    bool longitudinal=false;amrex::ParmParse("fixture").query("longitudinal",longitudinal);
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        pc.AddRealComp("event_reference",true);pc.AddIntComp("event_tag",true);
        bool const fast=name=="fast";R const density=fast?1.e18:9.e18;
        amrex::Vector<P> x,y,z,ux,uy,uz,weight;amrex::Vector<amrex::Vector<int>> integer(pc.NumIntComps());
        for(int j=0;j<lengths[1];++j){for(int i=0;i<lengths[0];++i){for(int ring=0;ring<4;++ring){
            R const radius=(i+.5)*dx[0],axial=(j+.5)*dx[1];
            R const angle=.37+ring*MathConst::pi/2.,co=std::cos(angle),si=std::sin(angle);
            x.push_back(radius*co);y.push_back(radius*si);z.push_back(axial);
            R const phase=2.*MathConst::pi*axial;
            R const vr=fast?4000.*radius*(1.-radius)*std::sin(phase):0.;
            R const vt=fast&&!longitudinal?(20000.+5000.*std::cos(phase))*radius:0.;
            R const vz=fast?12000.+6000.*std::sin(phase):0.;
            V const v{co*vr-si*vt,si*vr+co*vt,vz};
            R const gamma=1./std::sqrt(1.-norm(v)*norm(v)/(PhysConst::c*PhysConst::c));
            ux.push_back(gamma*v[0]);uy.push_back(gamma*v[1]);uz.push_back(gamma*v[2]);
            R const volume=MathConst::pi*((i+1.)*(i+1.)-i*i)*dx[0]*dx[0]*dx[1];
            weight.push_back(density*volume*.25*(1.+.1*radius*std::cos(phase)));
            for(auto& a:integer){a.push_back(0);}integer[pc.GetIntCompIndex("event_tag")].back()=1+ring+4*(i+lengths[0]*j);
        }}}
        pc.AddNParticles(0,x.size(),x,y,z,ux,uy,uz,1,amrex::Vector<amrex::Vector<P>>{weight},pc.NumIntComps(),integer,0);
        pc.Redistribute();
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){auto* extra=pti.GetAttribs("event_reference").data();auto const* tag=pti.GetiAttribs("event_tag").data();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){extra[n]=tag[n]+.125;});}
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
        if(v.decline && v.fault=="companion") {
            auto* j=w.m_fields.get(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{0},0);
            for(amrex::MFIter mfi(*j);mfi.isValid();++mfi){auto a=j->array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int k,int z){a(i,k,z)+=1.;});}
        }
        return true;
    }
    ++v.after;
    if(v.decline && v.relocate) {
        // Deliberately change the covered candidate map AFTER physical endpoint
        // verification. This rejected-only injection tests complete ownership
        // rollback; it is not an accepted orbit crossing qualification.
        auto& pc=w.GetPartContainer().GetParticleContainerFromName("fast");
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){
            auto* z=pti.GetAttribs(PIdx::z).data();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){if(z[n]<.5){z[n]+=.5;}});
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
        InstallPythonCallback("beforeInitEsolve",[](){LoadRetainedParticles(WarpX::GetInstance());});
        auto& w=WarpX::GetInstance();w.InitData();w.HybridPICInitializeRhoJandB();
        bool preflight=false;amrex::ParmParse("fixture").query("preflight",preflight);
        if(preflight) {
            // Capability and density declines must not invoke the operator or
            // mutate any candidate field/particle/time state.
            auto original=CompleteState(w);int calls=0;
            auto& model=*w.get_pointer_HybridPICModel();
            bool const external=model.m_add_external_fields;model.m_add_external_fields=true;
            auto unsupported=TryConstrainNativeEndpointField(w,w.gett_new(0),[&](){++calls;});
            model.m_add_external_fields=external;
            AMREX_ALWAYS_ASSERT(!unsupported && unsupported.failure==NativeEndpointFailure::UnsupportedScope);
            AMREX_ALWAYS_ASSERT(CompleteState(w)==original && calls==0);
            warpx::implicit::FieldRollback snapshot;snapshot.Capture(w.m_fields);
            w.m_fields.get(FieldType::rho_fp,0)->setVal(0.);
            auto zero=CompleteState(w);
            auto invalid=TryConstrainNativeEndpointField(w,w.gett_new(0),[&](){++calls;});
            AMREX_ALWAYS_ASSERT(!invalid && invalid.failure==NativeEndpointFailure::InvalidDensity);
            AMREX_ALWAYS_ASSERT(CompleteState(w)==zero && calls==0);
            AMREX_ALWAYS_ASSERT(snapshot.Restore(w.m_fields));snapshot.Discard();
            AMREX_ALWAYS_ASSERT(CompleteState(w)==original);
            amrex::Print()<<"RETAINED_PREFLIGHT_PASS unsupported=1 invalid_density=1 exact=1 operator_calls=0\n";
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
                int const expected=fault=="companion"?-11:-12;AMREX_ALWAYS_ASSERT(status==expected);
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
        amrex::Print()<<"RETAINED_ACCEPTANCE_PASS steps="<<steps<<" rejected="<<rejected<<" before_calls="<<validator.before<<" after_calls="<<validator.after<<" migrated="<<validator.migrated<<"\n";
        warpx::implicit::SetNativeAcceptanceValidator({});WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();return 0;
}
