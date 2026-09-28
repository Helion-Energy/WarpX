/* Native physical stopping event oracle. No Evolve/event guard is enabled. */
#include "NativeBytes.H"
#include "FieldSolver/ImplicitSolvers/NativeUniformStoppingEvent.H"
#include "FieldSolver/ImplicitSolvers/NativeInstantaneousIonCurrent.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/Collision/HybridElectronStopping/HybridElectronStopping.H"
#include "Particles/Collision/HybridElectronStopping/NativeStoppingMap.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Initialization/WarpXInit.H"
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
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        pc.AddRealComp("event_reference",false);pc.AddIntComp("event_tag",false);
        bool const fast=name=="fast";R const density=fast?1.e18:9.e18;
        V const v=fast?V{30000.,12000.,-6000.}:V{0.,0.,0.};
        R const gamma=1./std::sqrt(1.-norm(v)*norm(v)/(PhysConst::c*PhysConst::c));
        amrex::Vector<P> x,y,z,ux,uy,uz,weight;amrex::Vector<amrex::Vector<int>> integer(pc.NumIntComps());
        for(int k=0;k<lengths[2];++k){for(int j=0;j<lengths[1];++j){for(int i=0;i<lengths[0];++i){
            x.push_back((i+.5)*dx[0]);y.push_back((j+.5)*dx[1]);z.push_back((k+.5)*dx[2]);
            ux.push_back(gamma*v[0]);uy.push_back(gamma*v[1]);uz.push_back(gamma*v[2]);weight.push_back(density*dx[0]*dx[1]*dx[2]);
            for(auto& a:integer){a.push_back(0);}integer[pc.GetIntCompIndex("event_tag")].back()=i+lengths[0]*j+lengths[0]*lengths[1]*k+1;
        }}}
        pc.AddNParticles(0,x.size(),x,y,z,ux,uy,uz,1,amrex::Vector<amrex::Vector<P>>{weight},pc.NumIntComps(),integer,0);
        pc.Redistribute();
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){auto* extra=pti.GetAttribs("event_reference").data();auto const* tag=pti.GetiAttribs("event_tag").data();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){extra[n]=tag[n]+.125;});}
    }
}

// Verbatim algebraic oracle for the PRE-FACTORIZATION native rate and map.
AMREX_GPU_HOST_DEVICE auto LegacyKick(V u,V ve,R rho,R te,P mass,P charge,R dt,P weight){
    using namespace amrex::literals;
    R const n_e=rho*(1._rt/PhysConst::q_e),T_e=te*PhysConst::kb;
    R constexpr pi=MathConst::pi,ep0=PhysConst::epsilon_0,q_e2=PhysConst::q_e*PhysConst::q_e,ep02=ep0*ep0;
    R const pi32=pi*std::sqrt(pi),q2=charge*charge,T32=T_e*std::sqrt(T_e);
    R const nu_s=std::sqrt(2._rt)*n_e*q2*q_e2*std::sqrt(PhysConst::m_e)*10._prt/(12._rt*pi32*ep02*mass*T32);
    P constexpr inv_c2=1._prt/(PhysConst::c*PhysConst::c);
    P const u2b=u[0]*u[0]+u[1]*u[1]+u[2]*u[2],gb=std::sqrt(1._prt+u2b*inv_c2),fac=std::exp(-nu_s*dt);
    P const vxn=ve[0]+(u[0]/gb-ve[0])*fac,vyn=ve[1]+(u[1]/gb-ve[1])*fac,vzn=ve[2]+(u[2]/gb-ve[2])*fac;
    P const v2n=vxn*vxn+vyn*vyn+vzn*vzn,ga=1._prt/std::sqrt(1._prt-v2n*inv_c2),u2a=ga*ga*v2n;
    return amrex::GpuArray<R,4>{ga*vxn,ga*vyn,ga*vzn,weight*mass*(u2b/(gb+1._prt)-u2a/(ga+1._prt))};
}
void NativeOracle(WarpX& w,NativeAcceptedStoppingContext const& context,R interval){
    auto before=ParticleBytes(w);auto const& geometry=w.Geom(0);
    auto& pc=w.GetPartContainer().GetParticleContainerFromName("fast");
    auto* model=w.get_pointer_HybridPICModel();auto& staging=model->GetFastIonHeatingStaging(0);staging.setVal(0.);
    auto ve=w.m_fields.get_alldirs("Ve_fp",0),magnetic=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
    for(int c=0;c<3;++c){amrex::MultiFab::Copy(*ve[c],*context.Velocity()[c],0,0,1,ve[c]->nGrowVect());}
    std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<amrex::GpuArray<R,4>>> expected;
    auto const* te=w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
    auto const dxi=geometry.InvCellSizeArray(),plo=geometry.ProbLoArray();
    for(WarpXParIter pti(pc,0);pti.isValid();++pti){
        auto& values=expected[{pti.index(),pti.LocalTileIndex()}];values.resize(pti.numParticles());auto* out=values.data();
        auto const get=GetParticlePosition<PIdx>(pti);auto const* ux=pti.GetAttribs(PIdx::ux).data();auto const* uy=pti.GetAttribs(PIdx::uy).data();auto const* uz=pti.GetAttribs(PIdx::uz).data();auto const* weight=pti.GetAttribs(PIdx::w).data();
        auto const r=context.Binding().raw_charge->const_array(pti),t=te->const_array(pti);
        auto const gather=context.GatherView(pti,{magnetic[0],magnetic[1],magnetic[2]});R const mass=pc.getMass(),charge=pc.getCharge();
        amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){P x,y,z;get(n,x,y,z);
            R const rho=ablastr::particles::doGatherScalarFieldNodal(x,y,z,r,dxi,plo);
            R const temperature=amrex::max(ablastr::particles::doGatherScalarFieldNodal(x,y,z,t,dxi,plo),1.e-3*PhysConst::q_e/PhysConst::kb);
            auto const v=gather(x,y,z);out[n]=LegacyKick({ux[n],uy[n],uz[n]},v.collision_frame,rho,temperature,mass,charge,interval,weight[n]);
        });
    }
    HybridElectronStopping native("stopping");native.doCollisions(0.,interval,&w.GetPartContainer());
    amrex::Gpu::DeviceScalar<int> bad(0);int* b=bad.dataPtr();
    for(WarpXParIter pti(pc,0);pti.isValid();++pti){auto const* e=expected.at({pti.index(),pti.LocalTileIndex()}).data();
        auto const* ux=pti.GetAttribs(PIdx::ux).data();auto const* uy=pti.GetAttribs(PIdx::uy).data();auto const* uz=pti.GetAttribs(PIdx::uz).data();
        amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){
            V actual{ux[n],uy[n],uz[n]};for(int c=0;c<3;++c){auto const* a=reinterpret_cast<unsigned char const*>(&actual[c]);auto const* x=reinterpret_cast<unsigned char const*>(&e[n][c]);
                for(std::size_t i=0;i<sizeof(P);++i){if(a[i]!=x[i]){amrex::HostDevice::Atomic::Add(b,1);}}}
        });
    }
    amrex::Gpu::synchronize();int errors=bad.dataValue();amrex::ParallelDescriptor::ReduceIntSum(errors);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(errors==0,"unfactored/native finite stopping kick changed bytes");
    RestoreParticleBytes(w,before);staging.setVal(0.);
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before);
    amrex::Print()<<"NATIVE_STOPPING_ZERO_PSI_ORACLE exact=1\n";
}
void DumpParticles(WarpX& w,std::string const& filename){
    std::ofstream f(filename+".rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");f<<std::setprecision(17);
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){std::array<std::vector<P>,7> a;
            int const indices[7]={PIdx::x,PIdx::y,PIdx::z,PIdx::ux,PIdx::uy,PIdx::uz,PIdx::w};
            for(int c=0;c<7;++c){auto const& v=pti.GetAttribs(indices[c]);a[c].resize(v.size());amrex::Gpu::copy(amrex::Gpu::deviceToHost,v.begin(),v.end(),a[c].begin());}
            std::vector<std::uint64_t> tags(pti.numParticles());auto const& source=pti.GetStructOfArrays().GetIdCPUData();amrex::Gpu::copy(amrex::Gpu::deviceToHost,source.begin(),source.end(),tags.begin());
            for(long n=0;n<pti.numParticles();++n){f<<name<<' '<<tags[n]<<' '<<pc.getMass()<<' '<<pc.getCharge();for(auto const& v:a){f<<' '<<v[n];}f<<'\n';}
        }
    }
}
void DumpFields(WarpX& w,amrex::MultiFab const& energy,std::string const& name){
    auto const& g=w.Geom(0);std::ofstream f(name+".rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");f<<std::setprecision(17);
    std::vector<std::pair<std::string,amrex::MultiFab const*>> fields{{"rho",w.m_fields.get(FieldType::rho_fp,0)},{"U",&energy}};
    for(int c=0;c<3;++c){fields.emplace_back("J"+std::to_string(c),w.m_fields.get(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0));}
    for(auto const& [label,field]:fields){auto owner=field->OwnerMask(g.periodicity());
        for(amrex::MFIter mfi(*field);mfi.isValid();++mfi){
            auto const& fab=(*field)[mfi];auto const& own=(*owner)[mfi];
            std::vector<R> a(fab.size());std::vector<int> mask(own.size());
            amrex::Gpu::dtoh_memcpy(a.data(),fab.dataPtr(),a.size()*sizeof(R));amrex::Gpu::dtoh_memcpy(mask.data(),own.dataPtr(),mask.size()*sizeof(int));
            for(amrex::BoxIterator it(mfi.validbox());it.ok();++it){auto index=it();if(!mask[own.box().index(index)]){continue;}
                auto const value=a[fab.box().index(index)];for(int d=0;d<3;++d){index[d]=(index[d]%g.Domain().length(d)+g.Domain().length(d))%g.Domain().length(d);}
                f<<label<<' '<<index[0]<<' '<<index[1]<<' '<<index[2]<<' '<<value<<'\n';
            }
        }
    }
}
void WriteLedger(UniformStoppingLedger const& l,UniformStoppingOptions const& o){
    if(!amrex::ParallelDescriptor::IOProcessor()){return;}
    std::ofstream f("LEDGER.json");f<<std::setprecision(17)<<"{\n";
#define VALUE(x) f<<"\"" #x "\":"<<l.x<<",\n"
    VALUE(ion_initial_energy);VALUE(electron_initial_energy);VALUE(ion_energy_change);VALUE(electron_energy_change);VALUE(heat);
    VALUE(drag_energy_change);VALUE(drag_bulk_work);VALUE(electric_particle_work);VALUE(endpoint_particle_work);
    VALUE(relativistic_defect);VALUE(relativistic_bound);VALUE(spatial_transfer);VALUE(constraint_work);VALUE(mean_current_work);VALUE(transpose_work);
    VALUE(heat_transfer_error);VALUE(accounting_defect);VALUE(actual_energy_defect);VALUE(positive_energy_scale);VALUE(arithmetic_bound);
    VALUE(current_scale);VALUE(harmonic_residual);VALUE(grid_residual);VALUE(grid_bound);VALUE(uniform_error);VALUE(uniform_bound);VALUE(current_initial_error);
    VALUE(positive_momentum_scale);VALUE(momentum_error);VALUE(momentum_bound);VALUE(absolute_work);VALUE(contribution_count);VALUE(operations);VALUE(iterations);
#undef VALUE
#define VECTOR(x) f<<"\"" #x "\":["<<l.x[0]<<','<<l.x[1]<<','<<l.x[2]<<"],\n"
    VECTOR(electric_impulse);VECTOR(mean_current_increment);VECTOR(ion_momentum_change);VECTOR(reaction_momentum);VECTOR(electron_momentum_change);
#undef VECTOR
    f<<"\"interval\":"<<o.interval<<",\"cap\":"<<o.proper_speed_cap<<",\"relative_budget\":"<<o.relative_convention_budget<<",\"q_e\":"<<PhysConst::q_e<<",\"m_e\":"<<PhysConst::m_e<<",\"c\":"<<PhysConst::c<<",\"pass\":true}\n";
}
void Run(WarpX& w){
#if !defined(WARPX_DIM_3D)
    amrex::Abort("Physical uniform stopping fixture requires Cartesian3D");
#else
    Load(w);auto const& geometry=w.Geom(0);
    auto charge=w.m_fields.get_mr_levels(FieldType::rho_fp,0);w.GetPartContainer().DepositCharge(charge,0.);
    w.SyncRho(charge,{},{});charge[0]->OverrideSync(geometry.periodicity());charge[0]->FillBoundary(geometry.periodicity());
    auto* temperature=w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);temperature->setVal(temperature_ev*PhysConst::q_e/PhysConst::kb);
    std::array<amrex::MultiFab,3> current;
    auto const native=w.m_fields.get_alldirs(FieldType::current_fp,0);ablastr::fields::VectorField accepted{};
    for(int c=0;c<3;++c){current[c].define(native[c]->boxArray(),native[c]->DistributionMap(),1,native[c]->nGrowVect());
        accepted[c]=w.m_fields.alloc_init(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0,native[c]->boxArray(),native[c]->DistributionMap(),1,native[c]->nGrowVect(),0.);}
    AMREX_ALWAYS_ASSERT(warpx::particles::DepositNativeInstantaneousIonCurrent(w,warpx::particles::InstantaneousIonState::Current,{&current[0],&current[1],&current[2]}));
    for(int c=0;c<3;++c){amrex::MultiFab::Copy(*accepted[c],current[c],0,0,1,current[c].nGrowVect());accepted[c]->mult(-1.,0,1,accepted[c]->nGrow());}
    AcceptedStoppingOptions opts;opts.number_density_floor=1.e16;opts.reference_number_density=ne;opts.ghosts=charge[0]->nGrowVect();
    NativeAcceptedStoppingContext context(geometry,w.boxArray(0),w.DistributionMap(0),opts);
    AMREX_ALWAYS_ASSERT(context.Prepare(w.m_fields,NativeAcceptedStoppingContext::Bind(w.m_fields,false,0.,1)));
    amrex::MultiFab energy(w.boxArray(0),w.DistributionMap(0),1,1);energy.setVal(1.5*ne*temperature_ev*PhysConst::q_e);
    UniformStoppingOptions options;options.fast_species="fast";options.coulomb_log=10.;options.proper_speed_cap=cap;options.relative_convention_budget=accuracy;
    auto& fast=w.GetPartContainer().GetParticleContainerFromName("fast");
    R const rate=warpx::particles::NativeStoppingRate(ne,temperature_ev*PhysConst::q_e,fast.getCharge(),fast.getMass(),10.);
    options.interval=.2/rate;bool zero=false;amrex::ParmParse("fixture").query("zero",zero);if(zero){options.interval=0.;}
    NativeOracle(w,context,.2/rate);
    std::string negative;amrex::ParmParse("fixture").query("negative",negative);
    if(negative=="budget"){options.relative_convention_budget=1.e-20;}
    if(negative=="speed"){options.proper_speed_cap=1000.;}
    if(negative=="nonuniform"){
        for(amrex::MFIter mfi(*temperature);mfi.isValid();++mfi){auto const a=temperature->array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){if(i==0){a(i,j,k)*=1.01;}});}
        temperature->OverrideSync(geometry.periodicity());temperature->FillBoundary(geometry.periodicity());
    }
    auto before=ParticleBytes(w);std::vector<amrex::MultiFab const*> fields{charge[0],temperature,&energy,accepted[0],accepted[1],accepted[2]};
    for(auto key:{FieldType::Efield_fp,FieldType::Bfield_fp}){for(auto* field:w.m_fields.get_alldirs(key,0)){fields.push_back(field);}}
    auto const field_before=FieldBytes(fields);DumpParticles(w,"before");DumpFields(w,energy,"fields_before");
    NativeUniformStoppingEvent event(w,context,*temperature,energy,options);
    bool const prepared=event.Prepare();
    amrex::Print()<<std::setprecision(17)<<"EVENT_PREPARE valid="<<prepared<<" reason="<<event.Failure()<<" residual="<<event.Ledger().harmonic_residual<<" grid="<<event.Ledger().grid_residual<<" bound="<<event.Ledger().grid_bound<<" heat="<<event.Ledger().heat<<" defect="<<event.Ledger().accounting_defect<<" arithmetic="<<event.Ledger().arithmetic_bound<<"\n";
    if(!negative.empty()){
        AMREX_ALWAYS_ASSERT(!prepared&&ParticleBytes(w)==before&&FieldBytes(fields)==field_before);
        amrex::Print()<<"NATIVE_UNIFORM_STOPPING_NEGATIVE_PASS case="<<negative<<" reason="<<event.Failure()<<" purity=exact\n";return;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prepared,event.Failure());
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before&&FieldBytes(fields)==field_before);
    auto ledger=event.Ledger();WriteLedger(ledger,options);
    auto corrupt=ledger;corrupt.momentum_error=2.*ledger.momentum_bound+1.;AMREX_ALWAYS_ASSERT(!NativeUniformStoppingEvent::ValidateLedger(corrupt,options));
    corrupt=ledger;corrupt.heat=-ledger.heat-10.*ledger.arithmetic_bound-1.;
    AMREX_ALWAYS_ASSERT(!NativeUniformStoppingEvent::ValidateLedger(corrupt,options));
    corrupt=ledger;corrupt.accounting_defect=10.*ledger.arithmetic_bound+1.;AMREX_ALWAYS_ASSERT(!NativeUniformStoppingEvent::ValidateLedger(corrupt,options));
    corrupt=ledger;corrupt.grid_residual=2.*ledger.grid_bound+1.;AMREX_ALWAYS_ASSERT(!NativeUniformStoppingEvent::ValidateLedger(corrupt,options));
    struct Cache{int calls=0;NativeAcceptedStoppingContext* context;};Cache cache{0,&context};
    NativeUniformStoppingEvent::InvalidationHook hook{[](void* p)noexcept{auto& c=*static_cast<Cache*>(p);++c.calls;c.context->Invalidate();},&cache};
    AMREX_ALWAYS_ASSERT(!event.CommitOnce(accepted,{}));AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before&&FieldBytes(fields)==field_before);
    // Stale arbitrary real attributes reject before any accepted publication.
    for(WarpXParIter pti(fast,0);pti.isValid();++pti){auto* a=pti.GetAttribs("event_reference").data();
        amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){a[n]+=1.;});}
    auto const stale=ParticleBytes(w);AMREX_ALWAYS_ASSERT(!event.CommitOnce(accepted,hook));
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==stale&&FieldBytes(fields)==field_before&&cache.calls==0);
    RestoreParticleBytes(w,before);
    AMREX_ALWAYS_ASSERT(event.CommitOnce(accepted,hook));auto const after=ParticleBytes(w);auto const fields_after=FieldBytes(fields);
    AMREX_ALWAYS_ASSERT(!event.CommitOnce(accepted,hook));AMREX_ALWAYS_ASSERT(ParticleBytes(w)==after&&FieldBytes(fields)==fields_after);
    DumpParticles(w,"after");DumpFields(w,energy,"fields_after");
    AMREX_ALWAYS_ASSERT(warpx::particles::DepositNativeInstantaneousIonCurrent(w,warpx::particles::InstantaneousIonState::Current,{&current[0],&current[1],&current[2]}));
    R row=0.;for(int c=0;c<3;++c){amrex::MultiFab::Add(current[c],*accepted[c],0,0,1,0);row=std::max(row,current[c].norm0());}
    AMREX_ALWAYS_ASSERT(row<=ledger.grid_bound);
    AMREX_ALWAYS_ASSERT(event.Rollback(hook));AMREX_ALWAYS_ASSERT(ParticleBytes(w)==before&&FieldBytes(fields)==field_before);
    AMREX_ALWAYS_ASSERT(!context.Valid());
    AMREX_ALWAYS_ASSERT(context.Prepare(w.m_fields,NativeAcceptedStoppingContext::Bind(w.m_fields,false,0.,2)));
    NativeUniformStoppingEvent retry(w,context,*temperature,energy,options);AMREX_ALWAYS_ASSERT(retry.Prepare());AMREX_ALWAYS_ASSERT(retry.CommitOnce(accepted,hook));
    AMREX_ALWAYS_ASSERT(ParticleBytes(w)==after&&FieldBytes(fields)==fields_after);retry.Finalize();
    AMREX_ALWAYS_ASSERT(!retry.Rollback(hook));AMREX_ALWAYS_ASSERT(cache.calls==3);
    if(!zero){AMREX_ALWAYS_ASSERT(norm(ledger.electric_impulse)>0.&&norm(ledger.reaction_momentum)>0.&&ledger.heat>0.);}
    amrex::Print()<<"NATIVE_UNIFORM_STOPPING_PASS purity=exact rollback=exact retry=exact once_only=1 stale=reject cache_invalidation=3 wrong_heat=reject wrong_work=reject wrong_current=reject actual_row="<<row<<"\n";
#endif
}
int main(int argc,char** argv){
    warpx::initialization::initialize_external_libraries(argc,argv);
    {auto& w=WarpX::GetInstance();w.InitData();Run(w);WarpX::Finalize();}
    warpx::initialization::finalize_external_libraries();
}
