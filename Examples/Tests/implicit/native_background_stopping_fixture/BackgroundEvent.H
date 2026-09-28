#include "FieldSolver/ImplicitSolvers/NativeVacuumEndpoint.H"
void DumpVector(ablastr::fields::ConstVectorField const& v,std::string const& label){for(int c=0;c<3;++c)amrex::VisMF::Write(*v[c],label+std::to_string(c));}
void DumpInstantBackground(WarpX& w,std::string const& label){
    auto const shape=w.m_fields.get_alldirs(FieldType::current_fp,0);std::array<amrex::MultiFab,3> out;
    for(int c=0;c<3;++c)out[c].define(shape[c]->boxArray(),shape[c]->DistributionMap(),1,shape[c]->nGrowVect());
    AMREX_ALWAYS_ASSERT(warpx::particles::DepositNativeInstantaneousIonCurrent(w,warpx::particles::InstantaneousIonState::Current,{&out[0],&out[1],&out[2]}));
    DumpVector({&out[0],&out[1],&out[2]},label);
}
void DumpPreparedBackground(WarpX& w,NativeAcceptedStoppingContext const& context,NativeRZSpatialStoppingEvent const& event){
    auto const before=ParticleBytes(w),registry=RegistryBytes(w);
    DumpVector(event.ElectricImpulse(),"background_Psi_");DumpVector(event.ElectricAuxiliary(),"background_Aux_");
    DumpVector(event.StoppingVelocity(),"background_Ve_");DumpVector(event.StoppingImpulseDensity(),"background_Pi_");
    DumpVector(event.MeanElectronCurrent(),"background_Mean_");DumpVector(event.CandidateElectronCurrent(),"background_Je1_");
    DumpVector(event.CandidatePotential(),"background_A1_");DumpVector(event.CandidateMagnetic(),"background_B1_");
    DumpVector(event.CandidateDisplacement(),"background_D1_");DumpVector(event.CandidateConductorCurrent(),"background_W1_");
    DumpVector(event.PotentialIncrement(),"background_dA_");DumpVector(event.MagneticIncrement(),"background_dB_");
    DumpVector(event.DisplacementIncrement(),"background_dD_");DumpVector(event.ConductorIncrement(),"background_dW_");
    amrex::VisMF::Write(context.Kappa(),"background_Kappa");amrex::VisMF::Write(event.CandidateEnergy(),"background_U1");
    DumpInstantBackground(w,"background_I0_");
    auto B=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);auto E=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
    std::array<amrex::MultiFab,3> one,mass,C0,C1;
    for(int c=0;c<3;++c){one[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,E[c]->nGrowVect());one[c].setVal(1.);
        mass[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,E[c]->nGrowVect());
        C0[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,1);C0[c].setVal(0.);
        C1[c].define(E[c]->boxArray(),E[c]->DistributionMap(),1,1);C1[c].setVal(0.);}
    warpx::darwin::ApplyYeeInertiaMass(context.Geometry(),context.Kappa(),{&one[0],&one[1],&one[2]},
        {&mass[0],&mass[1],&mass[2]},PhysConst::m_e/(PhysConst::q_e*PhysConst::q_e*context.Options().reference_number_density));
    ablastr::fields::VectorField cv0{&C0[0],&C0[1],&C0[2]},cv1{&C1[0],&C1[1],&C1[2]};
    w.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv0,B,w.GetEBUpdateEFlag()[0],0);
    auto cb=event.MagneticIncrement();
    std::array<amrex::MultiFab,3> private_B;
    for(int c=0;c<3;++c){private_B[c].define(cb[c]->boxArray(),cb[c]->DistributionMap(),1,cb[c]->nGrowVect());amrex::MultiFab::Copy(private_B[c],*cb[c],0,0,1,cb[c]->nGrowVect());}
    w.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv1,{&private_B[0],&private_B[1],&private_B[2]},w.GetEBUpdateEFlag()[0],0);
    DumpVector({&mass[0],&mass[1],&mass[2]},"background_M_");DumpVector({&C0[0],&C0[1],&C0[2]},"background_C0_");DumpVector({&C1[0],&C1[1],&C1[2]},"background_dC_");
    std::ofstream f("background_points.rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");f<<std::setprecision(17);
    struct Point {std::uint64_t id;V position,psi,ve;};
    for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
        auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto const* ids=pti.GetStructOfArrays().GetIdCPUData().data();
            auto eg=context.GatherView(pti,{B[0],B[1],B[2]}),vg=eg;
            auto auxiliary=event.ElectricAuxiliary(),velocity=event.StoppingVelocity();
            for(int c=0;c<3;++c){eg.velocity[c]=auxiliary[c]->const_array(pti);eg.velocity_type[c]=auxiliary[c]->ixType();vg.velocity[c]=velocity[c]->const_array(pti);}
            amrex::Gpu::DeviceVector<Point> device(pti.numParticles());auto* out=device.data();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                P x,y,z;pos(n,x,y,z);auto const e=eg(x,y,z),v=vg(x,y,z);out[n]={ids[n],{x,y,z},e.cartesian,v.cartesian};
            });
            std::vector<Point> host(pti.numParticles());amrex::Gpu::copy(amrex::Gpu::deviceToHost,device.begin(),device.end(),host.begin());
            for(auto const& p:host){f<<name<<' '<<p.id;for(auto x:p.position)f<<' '<<x;for(auto x:p.psi)f<<' '<<x;for(auto x:p.ve)f<<' '<<x;f<<'\n';}
        }
    }
    AMREX_ALWAYS_ASSERT(before==ParticleBytes(w)&&registry==RegistryBytes(w));
}
void BackgroundEvent(WarpX& w){
    auto& model=*w.get_pointer_HybridPICModel();auto const& geometry=model.ElectronThermalGeometry();
    auto* rho=w.m_fields.get(FieldType::rho_fp,0);auto* temperature=w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
    auto* energy=w.m_fields.get(FieldType::hybrid_electron_energy_fp,0);
    AcceptedStoppingOptions opt;opt.number_density_floor=1.e16;opt.reference_number_density=ne;opt.ghosts=rho->nGrowVect();
    opt.field_lo=WarpX::field_boundary_lo;opt.field_hi=WarpX::field_boundary_hi;opt.particle_lo=WarpX::particle_boundary_lo;opt.particle_hi=WarpX::particle_boundary_hi;
    NativeAcceptedStoppingContext context(geometry,w.boxArray(0),w.DistributionMap(0),opt);
    AMREX_ALWAYS_ASSERT(context.Prepare(w.m_fields,NativeAcceptedStoppingContext::Bind(w.m_fields,false,w.gett_new(0),3)));
    SpatialStoppingOptions options;options.fast_species="fast";options.coulomb_log=10.;options.proper_speed_cap=cap;options.relative_convention_budget=accuracy;
    auto const& fast=w.GetPartContainer().GetParticleContainerFromName("fast");
    options.interval=.2/warpx::particles::NativeStoppingRate(ne,temperature_ev*PhysConst::q_e,fast.getCharge(),fast.getMass(),10.);
    SpatialStoppingFields destinations;RZStoppingBackground bg;
    for(int c=0;c<3;++c){destinations.potential[c]=w.m_fields.get("hybrid_A_fp",Direction{c},0);destinations.magnetic[c]=w.m_fields.get(FieldType::Bfield_fp,Direction{c},0);
        destinations.displacement[c]=w.m_fields.get("diagnostic_D_endpoint_fp",Direction{c},0);bg.conductor_current[c]=w.m_fields.get(PECWallCurrentName,Direction{c},0);}
    auto accepted=w.m_fields.get_alldirs(NativeAcceptedStoppingContext::AcceptedCurrentName,0);
    auto const before=ParticleBytes(w),registered_before=RegistryBytes(w);DumpRegistry(w,"background_before");
    std::string negative;amrex::ParmParse("fixture").query("background_negative",negative);
    bool zero_background=false;amrex::ParmParse("fixture").query("background_zero",zero_background);
    warpx::implicit::FieldRollback corruption;
    if(!negative.empty()){
        AMREX_ALWAYS_ASSERT(negative=="ampere"||negative=="stale"||negative=="magnetic"||negative=="wall"||negative=="alias");
        corruption.Capture(w.m_fields);
        if(negative=="ampere")accepted[2]->plus(1000.,0,1,0);
        if(negative=="magnetic")destinations.magnetic[1]->plus(0.001,0,1,0);
        if(negative=="wall")bg.conductor_current[0]->plus(1.,0,1,0);
        if(negative=="alias")bg.conductor_current[0]=accepted[0];
    }
    NativeRZSpatialStoppingEvent event(w,context,*temperature,*energy,options,destinations,bg);
    bool ready=event.Prepare();auto const& l=event.BackgroundLedger();
    amrex::Print()<<std::setprecision(17)<<"BACKGROUND_PREPARE ready="<<ready<<" reason="<<event.Failure()<<" residual="<<event.Ledger().spatial_residual
        <<" current="<<l.current_residual<<" bound="<<l.current_bound<<" magnetic="<<l.magnetic_residual<<" magnetic_bound="<<l.magnetic_bound<<" displacement="<<l.displacement_residual<<" displacement_bound="<<l.displacement_bound<<"\n";
    if(!negative.empty()&&negative!="stale"){
        AMREX_ALWAYS_ASSERT(!ready&&before==ParticleBytes(w));
        AMREX_ALWAYS_ASSERT(corruption.Restore(w.m_fields));corruption.Discard();
        AMREX_ALWAYS_ASSERT(registered_before==RegistryBytes(w));
        amrex::Print()<<"BACKGROUND_NEGATIVE_PASS kind="<<negative<<" rejected_before_commit=1 restored_exact=1\n";return;
    }
    AMREX_ALWAYS_ASSERT(before==ParticleBytes(w)&&registered_before==RegistryBytes(w));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ready,event.Failure());
    if(negative=="stale"){
        destinations.potential[1]->plus(1.e-6,0,1,0);
        NativeRZSpatialStoppingEvent::InvalidationHook invalidation{[](void*)noexcept{},nullptr};
        auto const altered=RegistryBytes(w);
        AMREX_ALWAYS_ASSERT(!event.CommitOnce(accepted,invalidation));
        AMREX_ALWAYS_ASSERT(altered==RegistryBytes(w)&&before==ParticleBytes(w));
        AMREX_ALWAYS_ASSERT(corruption.Restore(w.m_fields));corruption.Discard();
        AMREX_ALWAYS_ASSERT(registered_before==RegistryBytes(w));
        amrex::Print()<<"BACKGROUND_NEGATIVE_PASS kind=stale rejected_before_commit=1 restored_exact=1\n";return;
    }
    if(zero_background)AMREX_ALWAYS_ASSERT(l.magnetic_cross_work==0.&&event.Ledger().magnetic_initial_energy==0.);
    else AMREX_ALWAYS_ASSERT(l.magnetic_cross_work!=0.&&event.Ledger().magnetic_initial_energy>0.);

    std::vector<amrex::MultiFab const*> prepared;
    for(auto const& v:{event.CandidateElectronCurrent(),event.CandidatePotential(),event.CandidateMagnetic(),event.CandidateDisplacement(),event.CandidateConductorCurrent()})
        for(auto const* f:v)prepared.push_back(f);
    prepared.push_back(&event.CandidateEnergy());auto const prepared_bytes=FieldBytes(prepared);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(event.Prepare(),event.Failure());
    AMREX_ALWAYS_ASSERT(prepared_bytes==FieldBytes(prepared)&&before==ParticleBytes(w)&&registered_before==RegistryBytes(w));
    DumpPreparedBackground(w,context,event);WriteLedger(event.Ledger(),options);
    if(amrex::ParallelDescriptor::IOProcessor()){
        std::ofstream f("BACKGROUND_LEDGER.json");f<<std::setprecision(17)<<"{\n";
#define BL(x) f<<"\"" #x "\":"<<l.x<<",\n"
        BL(current_residual);BL(current_bound);BL(magnetic_residual);BL(magnetic_bound);BL(displacement_residual);BL(displacement_bound);
        BL(magnetic_cross_work);BL(magnetic_self_work);BL(magnetic_cross_scale);BL(initial_constraint_work);BL(initial_displacement_work);BL(initial_conductor_work);
#undef BL
        f<<"\"magnetic_initial_energy\":"<<event.Ledger().magnetic_initial_energy<<",\"pass\":true}\n";
    }
    auto bad=event.Ledger();bad.actual_energy_defect+=2.*std::abs(l.magnetic_cross_work)+100.*bad.arithmetic_bound;bad.accounting_defect=bad.actual_energy_defect;
    AMREX_ALWAYS_ASSERT(!NativeRZSpatialStoppingEvent::ValidateLedger(bad,options));
    NativeRZSpatialStoppingEvent::InvalidationHook hook{[](void* p)noexcept{static_cast<NativeAcceptedStoppingContext*>(p)->Invalidate();},&context};
    AMREX_ALWAYS_ASSERT(event.CommitOnce(accepted,hook));DumpInstantBackground(w,"background_I1_");DumpParticles(w,"background_after_particles");
    auto const committed=ParticleBytes(w),registered_commit=RegistryBytes(w);AMREX_ALWAYS_ASSERT(!event.CommitOnce(accepted,hook));
    warpx::implicit::FieldRollback endpoint_fields;endpoint_fields.Capture(w.m_fields);
    CopyNativeEndpointCurrentMirror(w,accepted,*w.m_fields.get("hybrid_Je_n_nodal",0));
    model.RefreshEulerianElectronThermodynamics(0,*rho);
    ConstrainNativeEndpointField(w,w.gett_new(0),false,[&](){NativeInstantaneousCurlRate(w,w.gett_new(0),true);});
    auto const recovered=RegistryBytes(w);DumpRegistry(w,"background_after_recovery");
    // Controlled rejection after candidate publication and native recovery.
    // Restore mirrors/recovery first, then the event's exact particle/field map.
    AMREX_ALWAYS_ASSERT(endpoint_fields.Restore(w.m_fields));endpoint_fields.Discard();
    AMREX_ALWAYS_ASSERT(registered_commit==RegistryBytes(w)&&committed==ParticleBytes(w));
    AMREX_ALWAYS_ASSERT(event.Rollback(hook));
    AMREX_ALWAYS_ASSERT(before==ParticleBytes(w)&&registered_before==RegistryBytes(w));
    AMREX_ALWAYS_ASSERT(context.Prepare(w.m_fields,NativeAcceptedStoppingContext::Bind(w.m_fields,false,w.gett_new(0),4)));
    NativeRZSpatialStoppingEvent retry(w,context,*temperature,*energy,options,destinations,bg);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(retry.Prepare(),retry.Failure());AMREX_ALWAYS_ASSERT(retry.CommitOnce(accepted,hook));
    AMREX_ALWAYS_ASSERT(committed==ParticleBytes(w)&&registered_commit==RegistryBytes(w));
    CopyNativeEndpointCurrentMirror(w,accepted,*w.m_fields.get("hybrid_Je_n_nodal",0));
    model.RefreshEulerianElectronThermodynamics(0,*rho);
    ConstrainNativeEndpointField(w,w.gett_new(0),false,[&](){NativeInstantaneousCurlRate(w,w.gett_new(0),true);});
    AMREX_ALWAYS_ASSERT(recovered==RegistryBytes(w));retry.Finalize();
    amrex::Print()<<"BACKGROUND_EVENT_PASS nonzero_background="<<(!zero_background)<<" raw_prepare_exact=1 prepared_ABA_exact=1 cross_work=1 once_only=1 after_candidate_rollback_exact=1 retry_exact=1\n";
}
