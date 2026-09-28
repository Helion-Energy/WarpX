#include "FieldSolver/ImplicitSolvers/NativePECPlasma.H"
#include "FieldSolver/ImplicitSolvers/NativeEndpointField.H"
#include "FieldSolver/ImplicitSolvers/ImplicitFieldRollback.H"
#include "Python/callbacks.H"
#include <AMReX_VisMF.H>
/* Native physical stopping event oracle. No Evolve/event guard is enabled. */
#include "NativeBytes.H"

#include "FieldSolver/ImplicitSolvers/NativeRZSpatialStoppingEvent.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/ImplicitSolvers/DarwinInitialRateSchur.H"
#include "FieldSolver/ImplicitSolvers/NativeInstantaneousIonCurrent.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/Collision/HybridElectronStopping/HybridElectronStopping.H"
#include "Particles/Collision/HybridElectronStopping/NativeStoppingMap.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Initialization/WarpXInit.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include <ablastr/particles/NodalFieldGather.H>
#include <AMReX_ParmParse.H>
#include <fstream>
#include <iomanip>
#include <limits>

using R=amrex::Real;using P=amrex::ParticleReal;
using V=amrex::GpuArray<R,3>;
using warpx::fields::FieldType;
using ablastr::fields::Direction;
using namespace warpx::thermal;
constexpr R ne=1.e19,temperature_ev=100.,cap=70000.,accuracy=1.e-6;
constexpr R eps=std::numeric_limits<R>::epsilon();
R norm(V const& x){return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
void Load(WarpX& w){
    auto const dx=w.Geom(0).CellSizeArray();auto const lengths=w.Geom(0).Domain().length();
    bool longitudinal=false;amrex::ParmParse("fixture").query("longitudinal",longitudinal);
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        pc.AddRealComp("event_reference",false);pc.AddIntComp("event_tag",false);
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

void DumpParticles(WarpX& w,std::string const& filename){
    std::ofstream f(filename+".rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");f<<std::setprecision(17);
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){std::array<std::vector<P>,7> a;
            int const indices[7]={PIdx::r,PIdx::theta,PIdx::z,PIdx::ux,PIdx::uy,PIdx::uz,PIdx::w};
            for(int c=0;c<7;++c){auto const& v=pti.GetAttribs(indices[c]);a[c].resize(v.size());amrex::Gpu::copy(amrex::Gpu::deviceToHost,v.begin(),v.end(),a[c].begin());}
            std::vector<std::uint64_t> tags(pti.numParticles());auto const& source=pti.GetStructOfArrays().GetIdCPUData();amrex::Gpu::copy(amrex::Gpu::deviceToHost,source.begin(),source.end(),tags.begin());
            for(long n=0;n<pti.numParticles();++n){f<<name<<' '<<tags[n]<<' '<<pc.getMass()<<' '<<pc.getCharge();for(auto const& v:a){f<<' '<<v[n];}f<<'\n';}
        }
    }
}
void DumpFields(WarpX& w,amrex::MultiFab const& energy,std::string const& name,std::array<amrex::MultiFab,3> const& ion){
    auto const& g=w.Geom(0);std::ofstream f(name+".rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");f<<std::setprecision(17);
    std::vector<std::pair<std::string,amrex::MultiFab const*>> fields{{"rho",w.m_fields.get(FieldType::rho_fp,0)},{"U",&energy}};
    for(int c=0;c<3;++c){fields.emplace_back("J"+std::to_string(c),w.m_fields.get(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0));}
    for(int c=0;c<3;++c){
        fields.emplace_back("I"+std::to_string(c),&ion[c]);
        fields.emplace_back("A"+std::to_string(c),w.m_fields.get("hybrid_A_fp",Direction{c},0));
        fields.emplace_back("B"+std::to_string(c),w.m_fields.get(FieldType::Bfield_fp,Direction{c},0));
        fields.emplace_back("D"+std::to_string(c),w.m_fields.get("diagnostic_D_endpoint_fp",Direction{c},0));
    }
    for(auto const& [label,field]:fields){auto owner=field->OwnerMask(g.periodicity());auto const measure=MakeQdsmcVolumeElement(g,field->ixType());
        for(amrex::MFIter mfi(*field);mfi.isValid();++mfi){
            auto const& fab=(*field)[mfi];auto const& own=(*owner)[mfi];
            std::vector<R> a(fab.size());std::vector<int> mask(own.size());
            amrex::Gpu::dtoh_memcpy(a.data(),fab.dataPtr(),a.size()*sizeof(R));amrex::Gpu::dtoh_memcpy(mask.data(),own.dataPtr(),mask.size()*sizeof(int));
            for(amrex::BoxIterator it(mfi.validbox());it.ok();++it){auto index=it();if(!mask[own.box().index(index)]){continue;}
                auto const value=a[fab.box().index(index)];for(int d=0;d<AMREX_SPACEDIM;++d){if(g.isPeriodic(d)){index[d]=(index[d]%g.Domain().length(d)+g.Domain().length(d))%g.Domain().length(d);}}
                f<<label<<' '<<index[0]<<' '<<index[1]<<' '<<0<<' '<<value<<' '<<field->ixType()[0]<<' '<<field->ixType()[1]<<' '<<measure(index[0],index[1],0)<<'\n';
            }
        }
    }
}
void WriteLedger(RZSpatialStoppingLedger const& l,SpatialStoppingOptions const& o){
    if(!amrex::ParallelDescriptor::IOProcessor()){return;}
    std::ofstream f("LEDGER.json");f<<std::setprecision(17)<<"{\n";
#define VALUE(x) f<<"\"" #x "\":"<<l.x<<",\n"
    VALUE(conductor_current_norm);VALUE(conductor_work);VALUE(axis_reaction_conjugate);VALUE(axis_reaction_work);
    VALUE(axial_momentum_defect);VALUE(axial_electric_transfer);VALUE(axial_image_transfer);
    VALUE(ion_initial_energy);VALUE(electron_initial_energy);VALUE(ion_energy_change);VALUE(electron_energy_change);VALUE(heat);
    VALUE(drag_energy_change);VALUE(drag_bulk_work);VALUE(electric_particle_work);VALUE(endpoint_particle_work);
    VALUE(relativistic_defect);VALUE(relativistic_bound);VALUE(spatial_transfer);VALUE(constraint_work);VALUE(mean_current_work);VALUE(transpose_work);
    VALUE(heat_transfer_error);VALUE(accounting_defect);VALUE(actual_energy_defect);VALUE(positive_energy_scale);VALUE(arithmetic_bound);
    VALUE(current_scale);VALUE(spatial_residual);VALUE(grid_residual);VALUE(grid_bound);VALUE(uniform_error);VALUE(uniform_bound);VALUE(current_initial_error);
    VALUE(current_range_mean);VALUE(current_range_bound);VALUE(impulse_range_mean);VALUE(impulse_range_bound);VALUE(range_ratio);VALUE(range_calls);VALUE(current_range_scale);VALUE(impulse_range_scale);
    VALUE(electric_momentum_transfer);VALUE(faraday_error);VALUE(faraday_bound);VALUE(displacement_curl);VALUE(displacement_curl_bound);VALUE(magnetic_energy_change);VALUE(curl_work);VALUE(displacement_work);VALUE(gauge_error);VALUE(magnetic_norm);VALUE(projection_residual);VALUE(linear_iterations);VALUE(evaluations);
    VALUE(positive_momentum_scale);VALUE(momentum_error);VALUE(momentum_bound);VALUE(absolute_work);VALUE(contribution_count);VALUE(operations);VALUE(iterations);
#undef VALUE
#define VECTOR(x) f<<"\"" #x "\":["<<l.x[0]<<','<<l.x[1]<<','<<l.x[2]<<"],\n"
    VECTOR(unrepresented_particle_momentum);VECTOR(electric_impulse);VECTOR(mean_current_increment);VECTOR(ion_momentum_change);VECTOR(reaction_momentum);VECTOR(electron_momentum_change);
#undef VECTOR
    f<<"\"interval\":"<<o.interval<<",\"cap\":"<<o.proper_speed_cap<<",\"relative_budget\":"<<o.relative_convention_budget<<",\"q_e\":"<<PhysConst::q_e<<",\"m_e\":"<<PhysConst::m_e<<",\"c\":"<<PhysConst::c<<",\"mu0\":"<<PhysConst::mu0<<",\"pass\":true}\n";
}

using warpx::darwin::PECWallCurrentName;
RawState RegistryBytes(WarpX& w){
    RawState state;
    for(auto const& name:w.m_fields.list()){
        auto const& f=*w.m_fields.internal_get(name);
        for(amrex::MFIter mfi(f);mfi.isValid();++mfi){
            state[name+"/"+std::to_string(mfi.index())]=ReadBytes(f[mfi].dataPtr(),f[mfi].size());
        }
    }
    return state;
}
void DumpRegistry(WarpX& w,std::string const& label){
    amrex::UtilCreateDirectory(label,0755);amrex::ParallelDescriptor::Barrier();
    std::ofstream list(label+"/fields.rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");
    for(auto const& name:w.m_fields.list()){
        auto const& f=*w.m_fields.internal_get(name);list<<name<<" "<<f.nComp()<<" "<<f.nGrowVect()<<"\n";
        amrex::VisMF::Write(f,label+"/"+name);
    }
    DumpParticles(w,label+"/particles");
}
#include "KernelOracle.H"
void Event(WarpX& w){
    AMREX_ALWAYS_ASSERT(warpx::darwin::NativePECPlasmaEnabled());
    auto& model=*w.get_pointer_HybridPICModel();
    auto const& geometry=model.ElectronThermalGeometry();
    auto* rho=w.m_fields.get(FieldType::rho_fp,0);
    auto* temperature=w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
    auto* energy=w.m_fields.get(FieldType::hybrid_electron_energy_fp,0);
    AcceptedStoppingOptions opt;opt.number_density_floor=1.e16;opt.reference_number_density=ne;
    opt.ghosts=rho->nGrowVect();opt.field_lo=WarpX::field_boundary_lo;opt.field_hi=WarpX::field_boundary_hi;
    opt.particle_lo=WarpX::particle_boundary_lo;opt.particle_hi=WarpX::particle_boundary_hi;
    NativeAcceptedStoppingContext context(geometry,w.boxArray(0),w.DistributionMap(0),opt);
    auto binding=NativeAcceptedStoppingContext::Bind(w.m_fields,false,w.gett_new(0),1);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(context.Prepare(w.m_fields,binding),context.Failure());
    SpatialStoppingOptions options;options.fast_species="fast";options.coulomb_log=10.;
    options.proper_speed_cap=cap;options.relative_convention_budget=accuracy;
    auto const& fast=w.GetPartContainer().GetParticleContainerFromName("fast");
    options.interval=.2/warpx::particles::NativeStoppingRate(ne,temperature_ev*PhysConst::q_e,fast.getCharge(),fast.getMass(),10.);
    bool zero=false;amrex::ParmParse("fixture").query("zero_event",zero);if(zero)options.interval=0.;
    SpatialStoppingFields destinations;std::array<amrex::MultiFab,3> private_d,wall;
    auto E=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto accepted=w.m_fields.get_alldirs(NativeAcceptedStoppingContext::AcceptedCurrentName,0);
    for(int c=0;c<3;++c){
        AMREX_ALWAYS_ASSERT(accepted[c]->nGrowVect()==amrex::IntVect(1));
        private_d[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,E[c]->nGrowVect());private_d[c].setVal(0.);
        wall[c].define(accepted[c]->boxArray(),accepted[c]->DistributionMap(),1,1);wall[c].setVal(0.);
        destinations.potential[c]=w.m_fields.get("hybrid_A_fp",Direction{c},0);
        destinations.displacement[c]=&private_d[c];destinations.magnetic[c]=w.m_fields.get(FieldType::Bfield_fp,Direction{c},0);
    }
    auto const before=ParticleBytes(w),registry=RegistryBytes(w);
    NativeRZSpatialStoppingEvent event(w,context,*temperature,*energy,options,destinations);
    bool const ready=event.Prepare();
    amrex::Print()<<std::setprecision(17)<<"PEC_EVENT_PREPARE ready="<<ready<<" reason="<<event.Failure()<<" residual="<<event.Ledger().spatial_residual<<" current0="<<event.Ledger().current_initial_error<<" uniform="<<event.Ledger().uniform_error<<"\n";
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before&&RegistryBytes(w)==registry);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ready,event.Failure());WriteLedger(event.Ledger(),options);
    auto const prepared_wall=event.CandidateConductorCurrent();
    for(int c=0;c<3;++c)amrex::MultiFab::Copy(wall[c],*prepared_wall[c],0,0,1,0);
    NativeRZSpatialStoppingEvent::InvalidationHook hook{[](void* p)noexcept{static_cast<NativeAcceptedStoppingContext*>(p)->Invalidate();},&context};
    AMREX_ALWAYS_ASSERT(event.CommitOnce(accepted,hook));
    auto const committed=ParticleBytes(w),registered=RegistryBytes(w);
    AMREX_ALWAYS_ASSERT(!event.CommitOnce(accepted,hook));
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==committed&&RegistryBytes(w)==registered);
    AMREX_ALWAYS_ASSERT(event.Rollback(hook));
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before&&RegistryBytes(w)==registry);
    AMREX_ALWAYS_ASSERT(context.Prepare(w.m_fields,NativeAcceptedStoppingContext::Bind(w.m_fields,false,w.gett_new(0),2)));
    NativeRZSpatialStoppingEvent retry(w,context,*temperature,*energy,options,destinations);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(retry.Prepare(),retry.Failure());
    AMREX_ALWAYS_ASSERT(retry.CommitOnce(accepted,hook));
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==committed&&RegistryBytes(w)==registered);
    auto D=w.m_fields.get_alldirs("diagnostic_D_endpoint_fp",0);
    for(int c=0;c<3;++c)amrex::MultiFab::Copy(*D[c],private_d[c],0,0,1,1);
    warpx::darwin::PublishNativePECEventWall(w,{&wall[0],&wall[1],&wall[2]});
    model.RefreshEulerianElectronThermodynamics(0,*rho);
    DumpRegistry(w,"event_before_recovery");
    ConstrainNativeEndpointField(w,w.gett_new(0),false,[&](){NativeInstantaneousCurlRate(w,w.gett_new(0),true);});
    // Recovery holds accepted companions. The next OneStep takes its field
    // seed directly from the accepted E registry, exactly as at restart.
    retry.Finalize();DumpRegistry(w,"event_after_recovery");
    amrex::Print()<<"PEC_EVENT_COMMIT_PASS prepare_registry_exact=1 particle_rng_exact=1 rollback_exact=1 retry_exact=1 once_only=1\n";
}
int main(int argc,char** argv){
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        InstallPythonCallback("beforeInitEsolve",[](){Load(WarpX::GetInstance());});
        auto& w=WarpX::GetInstance();w.InitData();
        // Same native moment prelude used by Evolve. Capture only after it.
        w.HybridPICInitializeRhoJandB();
        DumpRegistry(w,"initialized");
        bool event=false;amrex::ParmParse("fixture").query("event",event);if(event)Event(w);
        bool kernel=false;amrex::ParmParse("fixture").query("kernel_oracle",kernel);if(kernel)KernelOracle(w);

        bool poison=false;amrex::ParmParse("fixture").query("poison_ampere_outer",poison);
        if(poison){
            auto jp=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
            for(int c=0;c<3;++c)for(amrex::MFIter mfi(*jp[c]);mfi.isValid();++mfi){
                auto const a=jp[c]->array(mfi);auto const known=amrex::grow(mfi.validbox(),1);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    if(!known.contains(amrex::IntVect(i,j))){a(i,j,k)=(c+1.)*1.e90;}
                });
            }
        }
        InstallPythonCallback("afterstep",[&](){DumpRegistry(w,"accepted_"+std::to_string(w.getistep(0)));});
        w.Evolve();amrex::Print()<<"PEC_PLASMA_CONTINUATION_PASS steps="<<w.getistep(0)<<"\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
