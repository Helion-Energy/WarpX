#ifndef PEC_KERNEL_ORACLE_H_
#define PEC_KERNEL_ORACLE_H_
// Fixture only: actual native wrapper, complete registry restore, no new solver law.
void KernelOracle(WarpX& w){
    auto const before_particles=ParticleBytes(w),before_fields=RegistryBytes(w);
    warpx::implicit::FieldRollback snapshot;snapshot.Capture(w.m_fields);
    auto& model=*w.get_pointer_HybridPICModel();auto E=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto B=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);auto Ji=w.m_fields.get_alldirs(FieldType::current_fp,0);
    auto Jp=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    auto Je=w.m_fields.get_alldirs(NativeAcceptedStoppingContext::AcceptedCurrentName,0);
    auto const& rho=*w.m_fields.get(FieldType::rho_fp,0);
    std::array<amrex::MultiFab,3> ordinary,captured,raw,reference;
    for(int c=0;c<3;++c){
        for(auto* f:{&ordinary[c],&captured[c],&raw[c],&reference[c]})f->define(E[c]->boxArray(),E[c]->DistributionMap(),1,E[c]->nGrowVect());
        raw[c].setVal(std::numeric_limits<R>::quiet_NaN());
    }
    auto& EB=w.GetEBUpdateEFlag()[0];
    std::array<amrex::MultiFab,3> deposited;
    for(int c=0;c<3;++c)deposited[c].define(Ji[c]->boxArray(),Ji[c]->DistributionMap(),1,Ji[c]->nGrowVect());
    AMREX_ALWAYS_ASSERT(warpx::particles::DepositNativeInstantaneousIonCurrent(w,warpx::particles::InstantaneousIonState::Current,{&deposited[0],&deposited[1],&deposited[2]}));
    for(int c=0;c<3;++c)amrex::MultiFab::Copy(*Ji[c],deposited[c],0,0,1,Ji[c]->nGrowVect());
    for(int c=0;c<3;++c)amrex::MultiFab::LinComb(*Jp[c],1.,*Ji[c],0,1.,*Je[c],0,0,1,1);
    model.HybridPICSolveE(E,Ji,B,rho,EB,0,false,true);
    for(int c=0;c<3;++c)amrex::MultiFab::Copy(ordinary[c],*E[c],0,0,1,E[c]->nGrowVect());
    ablastr::fields::VectorField F{&raw[0],&raw[1],&raw[2]};
    std::string negative;amrex::ParmParse("fixture").query("capture_negative",negative);
    std::unique_ptr<amrex::MultiFab> undersized;
    if(negative=="alias")F=E;
    if(negative=="width"){
        undersized=std::make_unique<amrex::MultiFab>(E[0]->boxArray(),E[0]->DistributionMap(),1,0);F[0]=undersized.get();
    }
    model.HybridPICSolveE(E,Ji,B,rho,EB,0,false,true,nullptr,nullptr,nullptr,&F);
    AMREX_ALWAYS_ASSERT(negative.empty());
    for(int c=0;c<3;++c){
        AMREX_ALWAYS_ASSERT(FieldBytes({&ordinary[c]})==FieldBytes({E[c]}));
        AMREX_ALWAYS_ASSERT(raw[c].is_finite(0,1,raw[c].nGrowVect()));
        amrex::MultiFab::Copy(reference[c],raw[c],0,0,1,raw[c].nGrowVect());
    }
    int const wall=w.Geom(0).Domain().bigEnd(0)+1;
    for(int c=1;c<3;++c)for(amrex::MFIter mfi(*Jp[c]);mfi.isValid();++mfi){
        auto const j=Jp[c]->array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int k,int l){if(i==wall)j(i,k,l)+=(c+1.)*1.e5;});
    }
    for(auto* f:Jp){f->OverrideSync(w.Geom(0).periodicity());f->FillBoundary(w.Geom(0).periodicity());}
    model.HybridPICSolveE(E,Ji,B,rho,EB,0,false,true,nullptr,nullptr,nullptr,&F);
    R tangent=0.,normal=0.,Bn=0.;
    for(int c=0;c<3;++c){
        amrex::MultiFab diff(E[c]->boxArray(),E[c]->DistributionMap(),1,0);diff.setVal(0.);
        for(amrex::MFIter mfi(diff);mfi.isValid();++mfi){auto const d=diff.array(mfi);auto const a=raw[c].const_array(mfi),b=reference[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){d(i,j,k)=(c==0||i==wall)?a(i,j,k)-b(i,j,k):0.;});}
        if(c==0)normal=diff.norm0();else tangent=std::max(tangent,diff.norm0());
    }
    amrex::MultiFab br(B[0]->boxArray(),B[0]->DistributionMap(),1,0);br.setVal(0.);
    for(amrex::MFIter mfi(br);mfi.isValid();++mfi){auto const d=br.array(mfi),b=B[0]->array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){d(i,j,k)=i==wall?b(i,j,k):0.;});}
    Bn=br.norm0();AMREX_ALWAYS_ASSERT(Bn==0.&&tangent==0.);
    AMREX_ALWAYS_ASSERT(snapshot.Restore(w.m_fields));snapshot.Discard();
    AMREX_ALWAYS_ASSERT(RegistryBytes(w)==before_fields&&ParticleBytes(w)==before_particles);
    amrex::Print()<<std::setprecision(17)<<"PEC_KERNEL_ORACLE_PASS outputs_exact=1 registry_restore_exact=1 particle_rng_exact=1 tangent_independence="<<tangent<<" normal_response="<<normal<<" Br_wall="<<Bn<<"\n";
}
#endif
