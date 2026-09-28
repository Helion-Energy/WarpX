/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include "FieldSolver/ImplicitSolvers/ImplicitAbsorbingEndpoint.H"
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include <AMReX_Reduce.H>
#include <AMReX_ParmParse.H>
#include <iomanip>
using Real=amrex::Real;
using warpx::fields::FieldType;
using namespace warpx::thermal;
Real integral(amrex::MultiFab const& f,amrex::Geometry const& g) {
    auto const dx=g.CellSizeArray(),lo=g.ProbLoArray();
    bool const rz=g.IsRZ();
    Real const v=AMREX_D_TERM(dx[0],*dx[1],*dx[2]);
    amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<Real> data(op);
    for(amrex::MFIter mfi(f);mfi.isValid();++mfi){auto a=f.const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k){
            return amrex::GpuTuple<Real>{a(i,j,k)*v*(rz ? 2*MathConst::pi*(lo[0]+(i+.5)*dx[0]) : 1.)};});}
    Real value=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(value);return value;
}
void deposit(WarpX& w,amrex::MultiFab& rho){
    ablastr::fields::MultiLevelScalarField r{&rho};w.GetPartContainer().DepositCharge(r,0.);
    w.SyncRho(r,{},{});w.ApplyRhofieldBoundary(0,&rho,PatchType::fine);rho.FillBoundary(w.Geom(0).periodicity());
}
int main(int argc,char** argv){
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w=WarpX::GetInstance();w.InitData();auto& mpc=w.GetPartContainer();
        auto& model=*w.get_pointer_HybridPICModel();
        auto const& g=w.Geom(0);auto const lo=g.ProbLoArray(),hi=g.ProbHiArray();
        Real const dt=w.getdt(0),dz=g.CellSize(WARPX_ZINDEX);
        Real displacement=.4;amrex::ParmParse("absorption_test").query("displacement_cells",displacement);
        Real const v=displacement*dz/dt,uz=v/std::sqrt(1-v*v/(PhysConst::c*PhysConst::c));
        auto const& layout=*w.m_fields.get(FieldType::rho_fp,0);
        auto make=[&](){return amrex::MultiFab(layout.boxArray(),layout.DistributionMap(),1,layout.nGrowVect());};
        auto old=make(),virt=make(),survivor=make(),lost=make(),actual=make(),cv=make(),cs=make(),difference=make();
        deposit(w,old);
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto* u=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                amrex::ParticleReal x,y,z;pos(i,x,y,z);u[i]=z<.5*(lo[WARPX_ZINDEX]+hi[WARPX_ZINDEX]) ? -uz : uz;});}
        w.SaveParticlesAtImplicitStepStart(true);
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto set=SetParticlePosition(pti);auto const* u=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                amrex::ParticleReal x,y,z;pos(i,x,y,z);z+=.5*dt*u[i]/std::sqrt(1+u[i]*u[i]/(PhysConst::c*PhysConst::c));set(i,x,y,z);});}
        auto current=w.m_fields.get_alldirs(FieldType::current_fp,0);for(auto* c:current)c->setVal(0.);
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            pc->DepositCurrent(pti,pti.GetAttribs(PIdx::w),pti.GetAttribs(PIdx::ux),pti.GetAttribs(PIdx::uy),
                pti.GetAttribs(PIdx::uz),nullptr,current[0],current[1],current[2],0,pti.numParticles(),0,0,0,dt,0.,PushType::Implicit);}
#ifdef WARPX_DIM_RZ
        w.ApplyInverseVolumeScalingToCurrentDensity(current[0],current[1],current[2],0);
#endif
        w.SyncCurrent("current_fp");w.ApplyJfieldBoundary(0,current[0],current[1],current[2],PatchType::fine);
        for(auto* c:current)c->FillBoundary(g.periodicity());
        using Subset=MassMatrixDensityProjection::EndpointSubset;
        AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,virt,Subset::All));
        AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,survivor,Subset::Survivors));
        AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,lost,Subset::Absorbed));
        amrex::MultiFab::LinComb(difference,1.,survivor,0,1.,lost,0,0,1,0);
        amrex::MultiFab::Subtract(difference,virt,0,0,1,0);
        Real const partition=difference.norminf()/std::max(virt.norminf(),Real(1.e-30));
        MassMatrixDensityProjection::ComputeDivergence(w,0,current,cv);
        cv.mult(dt,0,1,0);amrex::MultiFab::Copy(cs,cv,0,0,1,0);
        amrex::MultiFab::Add(cv,virt,0,0,1,0);amrex::MultiFab::Subtract(cv,old,0,0,1,0);
        amrex::MultiFab::Add(cs,survivor,0,0,1,0);amrex::MultiFab::Add(cs,lost,0,0,1,0);amrex::MultiFab::Subtract(cs,old,0,0,1,0);
        Real const continuity=cv.norminf()/old.norminf(), sink_continuity=cs.norminf()/old.norminf();
        auto const cells=amrex::convert(layout.boxArray(),amrex::IntVect(0));
        KineticThermalMoments moments(model.ElectronThermalGeometry(),cells,layout.DistributionMap(),model.EulerianMomentOptions());
        amrex::MultiFab loss_cell(cells,layout.DistributionMap(),1,0);
        moments.RestrictNativeMoment(lost,0,loss_cell);Real const charge=integral(loss_cell,model.ElectronThermalGeometry());
        // Independent virtual particle ledger, including neutral weight/energy
        // and excluded alpha charge. Accepted native tallies must match it.
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Long,Real,Real,Real> data(op);
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto const* xn=pti.GetAttribs("x_n").dataPtr();
            auto const* yn=pti.GetAttribs("y_n").dataPtr();auto const* zn=pti.GetAttribs("z_n").dataPtr();
            auto const* u=pti.GetAttribs(PIdx::uz).dataPtr();auto const* weight=pti.GetAttribs(PIdx::w).dataPtr();
            auto const q=pc->getCharge(),mass=pc->getMass();
            auto boundary=warpx::implicit::MakeAbsorbingEndpointBoundary(g,pc->GetParticleBoundaryData(),WarpX::field_boundary_lo,WarpX::field_boundary_hi);
            op.eval(pti.numParticles(),data,[=] AMREX_GPU_DEVICE(long i)->decltype(data)::Type{
                amrex::GpuArray<amrex::ParticleReal,3>x{},momentum{0,0,u[i]};pos(i,x[0],x[1],x[2]);
                x={2*x[0]-xn[i],2*x[1]-yn[i],2*x[2]-zn[i]};
                if(warpx::implicit::InspectAbsorbingEndpoint(x,momentum,boundary).status!=warpx::implicit::AbsorbingEndpointStatus::Absorbed)return {0,0,0,0};
                Real const k=weight[i]*mass*u[i]*u[i]/(1+std::sqrt(1+u[i]*u[i]/(PhysConst::c*PhysConst::c)));
                return {1,weight[i],q*weight[i],k};});}
        auto prediction=data.value();amrex::Long lost_count=amrex::get<0>(prediction);amrex::ParallelDescriptor::ReduceLongSum(lost_count);
        Real predicted[3]={amrex::get<1>(prediction),amrex::get<2>(prediction),amrex::get<3>(prediction)};amrex::ParallelDescriptor::ReduceRealSum(predicted,3);
        w.FinishImplicitParticleUpdate(dt);mpc.ApplyBoundaryConditions();
        auto& buffer=w.GetParticleBoundaryBuffer();buffer.gatherParticlesFromDomainBoundaries(mpc,dt);
        mpc.Redistribute();deposit(w,actual);
        amrex::MultiFab::Subtract(actual,survivor,0,0,1,0);
        Real const survivor_error=actual.norminf()/old.norminf();
        Real tallies[3]={0,0,0};amrex::Long buffered=0;
        for(auto const& pc:mpc)for(int d=0;d<AMREX_SPACEDIM;++d)for(int side=0;side<2;++side){
            tallies[0]+=pc->GetBoundaryAbsorbedWeight(d,side);tallies[1]+=pc->GetBoundaryAbsorbedCharge(d,side);tallies[2]+=pc->GetBoundaryAbsorbedEnergy(d,side);
            if(d==WARPX_ZINDEX)buffered+=buffer.getNumParticlesInContainer(pc->getName(),2*d+side,true);}
        amrex::ParallelDescriptor::ReduceRealSum(tallies,3);amrex::ParallelDescriptor::ReduceLongSum(buffered);
        for(int c=0;c<3;++c)AMREX_ALWAYS_ASSERT(std::abs(tallies[c]-predicted[c])<=1.e-12*std::max(std::abs(predicted[c]),Real(1.e-30)));
        AMREX_ALWAYS_ASSERT(buffered==lost_count && partition<1.e-12 && continuity<1.e-11 && sink_continuity<1.e-11 && survivor_error<1.e-12);
        AMREX_ALWAYS_ASSERT(std::abs(charge-predicted[1])<=1.e-11*std::max(std::abs(predicted[1]),Real(1.e-30)));
        if(displacement>.25)AMREX_ALWAYS_ASSERT(lost_count>0);else AMREX_ALWAYS_ASSERT(lost_count==0);
        // The normal later boundary hook must not tally or scrape these ids twice.
        mpc.ApplyBoundaryConditions();buffer.gatherParticlesFromDomainBoundaries(mpc,dt);
        Real repeated[3]={0,0,0};amrex::Long buffer_again=0;
        for(auto const& pc:mpc)for(int d=0;d<AMREX_SPACEDIM;++d)for(int side=0;side<2;++side){
            repeated[0]+=pc->GetBoundaryAbsorbedWeight(d,side);repeated[1]+=pc->GetBoundaryAbsorbedCharge(d,side);repeated[2]+=pc->GetBoundaryAbsorbedEnergy(d,side);
            if(d==WARPX_ZINDEX)buffer_again+=buffer.getNumParticlesInContainer(pc->getName(),2*d+side,true);}
        amrex::ParallelDescriptor::ReduceRealSum(repeated,3);amrex::ParallelDescriptor::ReduceLongSum(buffer_again);
        for(int c=0;c<3;++c)AMREX_ALWAYS_ASSERT(repeated[c]==tallies[c]);AMREX_ALWAYS_ASSERT(buffer_again==buffered);
        amrex::Print()<<std::setprecision(17)<<"NATIVE_ABSORPTION_PASS lost="<<lost_count<<" buffers="<<buffered
            <<" partition="<<partition<<" continuity_virtual="<<continuity<<" continuity_survivor_sink="<<sink_continuity
            <<" survivor="<<survivor_error<<" deposited_charge="<<charge<<" native_charge="<<tallies[1]<<"\n";
        w.DiscardSavedImplicitParticleState();
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
