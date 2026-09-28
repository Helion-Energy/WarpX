/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include "FieldSolver/ImplicitSolvers/ImplicitAbsorbingEndpoint.H"
#include "FieldSolver/ImplicitSolvers/ImplicitParticleEndpointAudit.H"
#include "FieldSolver/ImplicitSolvers/ImplicitIonEndpointTrial.H"
#include <AMReX_Random.H>
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include <AMReX_Reduce.H>
#include <AMReX_ParmParse.H>
#include <iomanip>
#include "reflecting_endpoint_fixture/NativeStateBytes.H"
#include "reflecting_endpoint_fixture/PositiveDepositRoundoff.H"
#include "reflecting_endpoint_fixture/EndpointMomentDiagnostic.H"
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
        Real displacement=0.,radial_displacement=.4;
        amrex::ParmParse("radial_test").query("axial_displacement_cells",displacement);
        amrex::ParmParse("radial_test").query("displacement_cells",radial_displacement);
        Real const v=displacement*dz/dt,vr=radial_displacement*g.CellSize(0)/dt;
        Real const wall=hi[0],dr=g.CellSize(0);
        for(auto const& pc:mpc)for(auto name:{"wall_r","wall_theta","wall_ux","wall_uy","wall_uz","wall_reflected"})
            pc->AddRealComp(name,1);
        auto const& layout=*w.m_fields.get(FieldType::rho_fp,0);
        auto make=[&](){return amrex::MultiFab(layout.boxArray(),layout.DistributionMap(),1,layout.nGrowVect());};
        auto old=make(),virt=make(),survivor=make(),lost=make(),actual=make(),cv=make(),cs=make(),difference=make();
        radial_fixture::PositiveDepositRoundoff deposition(w,old);
        deposition.ValidateConstituents(w,old,false);
        deposit(w,old);
        auto old_repeat=make();deposit(w,old_repeat);
        amrex::MultiFab::Subtract(old_repeat,old,0,0,1,0);
        Real const old_repeat_error=old_repeat.norminf();
        amrex::Print()<<std::setprecision(17)<<"NATIVE_DEPOSIT_REPEAT phase=old absolute="
            <<old_repeat_error<<" reference="<<old.norminf()<<std::endl;
        deposition.Check(old_repeat,old,"old_repeat");
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto* ux=pti.GetAttribs(PIdx::ux).dataPtr();
            auto* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto* uz=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                amrex::ParticleReal x,y,z;pos(i,x,y,z);Real const r=std::sqrt(x*x+y*y);
                Real const radial=r>wall-.5*dr ? vr : 0, tangential=.17*radial;
                Real const vx=(radial*x-tangential*y)/r,vy=(radial*y+tangential*x)/r;
                Real const vz=z<.5*(lo[WARPX_ZINDEX]+hi[WARPX_ZINDEX]) ? -v : v;
                Real const gamma=1/std::sqrt(1-(vx*vx+vy*vy+vz*vz)/(PhysConst::c*PhysConst::c));
                ux[i]=gamma*vx;uy[i]=gamma*vy;uz[i]=gamma*vz;});}
        w.SaveParticlesAtImplicitStepStart(true);
        // Current deposition and wall_* fixture records change intentionally;
        // the rollback check covers every other raw particle byte.
        auto const accepted_particles=radial_fixture::NativeStateBytes::Capture(w,false,false);
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto set=SetParticlePosition(pti);
            auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();
            auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                amrex::ParticleReal x,y,z;pos(i,x,y,z);
                Real const inv=1/std::sqrt(1+(ux[i]*ux[i]+uy[i]*uy[i]+uz[i]*uz[i])/(PhysConst::c*PhysConst::c));
                set(i,x+.5*dt*ux[i]*inv,y+.5*dt*uy[i]*inv,z+.5*dt*uz[i]*inv);});}
        warpx::implicit::EndpointAuditFields audit_fields;
        audit_fields.allow_deterministic_absorption=true;audit_fields.allow_radial_reflection=true;
        audit_fields.endpoint_density=&virt;audit_fields.gather_filled_ghosts=w.get_ng_fieldgather();
        audit_fields.current_deposit_ghosts=w.get_ng_depos_J();
        for(int c=0;c<3;++c){
            audit_fields.current[c]=w.m_fields.get(FieldType::current_fp,ablastr::fields::Direction{c},0);
            audit_fields.gather_e[c]=w.m_fields.get(FieldType::Efield_aux,ablastr::fields::Direction{c},0);
            audit_fields.gather_b[c]=w.m_fields.get(FieldType::Bfield_aux,ablastr::fields::Direction{c},0);}
        auto const audit=warpx::implicit::AuditImplicitParticleEndpoints(w,0,audit_fields);
        amrex::Print()<<"REFLECTION_ENDPOINT_AUDIT "<<audit.summary()<<"\n";AMREX_ALWAYS_ASSERT(audit.ok());
        auto const reflected_count=audit.count(warpx::implicit::EndpointIssue::SupportedReflection);
        AMREX_ALWAYS_ASSERT(reflected_count>0);
        // Record the pure mapped endpoint by native particle identity. The
        // extra fixture attributes redistribute with the actual particles.
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);
            auto const* xn=pti.GetAttribs("x_n").dataPtr();auto const* yn=pti.GetAttribs("y_n").dataPtr();
            auto const* zn=pti.GetAttribs("z_n").dataPtr();
            auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::GpuArray<amrex::ParticleReal*,6> output{pti.GetAttribs("wall_r").dataPtr(),pti.GetAttribs("wall_theta").dataPtr(),
                pti.GetAttribs("wall_ux").dataPtr(),pti.GetAttribs("wall_uy").dataPtr(),pti.GetAttribs("wall_uz").dataPtr(),pti.GetAttribs("wall_reflected").dataPtr()};
            auto native=pc->GetParticleBoundaryData();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                amrex::GpuArray<amrex::ParticleReal,3>x{},u{ux[i],uy[i],uz[i]};pos(i,x[0],x[1],x[2]);
                x={2*x[0]-xn[i],2*x[1]-yn[i],2*x[2]-zn[i]};
                auto const image=warpx::implicit::MapRadialEndpoint(x,u,0,wall,native,true);
                AMREX_ALWAYS_ASSERT(image.valid);output[0][i]=image.stored[0];output[1][i]=image.stored[1];
                for(int d=0;d<3;++d)output[2+d][i]=image.momentum[d];output[5][i]=image.reflected;
            });}
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
        deposition.ValidateConstituents(w,virt,true);
        AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,virt,Subset::All));
        auto const before_repeat=radial_fixture::NativeStateBytes::Capture(w);
        auto virtual_repeat=make();
        AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,virtual_repeat,Subset::All));
        before_repeat.RequireUnchanged(w,"repeat_endpoint_deposit");
        amrex::MultiFab::Subtract(virtual_repeat,virt,0,0,1,0);
        Real const virtual_repeat_error=virtual_repeat.norminf();
        amrex::Print()<<std::setprecision(17)<<"NATIVE_DEPOSIT_REPEAT phase=virtual absolute="
            <<virtual_repeat_error<<" reference="<<virt.norminf()<<std::endl;
        deposition.Check(virtual_repeat,virt,"virtual_repeat");
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
            auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();
            auto const* u=pti.GetAttribs(PIdx::uz).dataPtr();auto const* weight=pti.GetAttribs(PIdx::w).dataPtr();
            auto const q=pc->getCharge(),mass=pc->getMass();
            auto boundary=warpx::implicit::MakeAbsorbingEndpointBoundary(g,pc->GetParticleBoundaryData(),WarpX::field_boundary_lo,WarpX::field_boundary_hi,true);
            op.eval(pti.numParticles(),data,[=] AMREX_GPU_DEVICE(long i)->decltype(data)::Type{
                amrex::GpuArray<amrex::ParticleReal,3>x{},momentum{ux[i],uy[i],u[i]};pos(i,x[0],x[1],x[2]);
                x={2*x[0]-xn[i],2*x[1]-yn[i],2*x[2]-zn[i]};
                if(warpx::implicit::InspectAbsorbingEndpoint(x,momentum,boundary).status!=warpx::implicit::AbsorbingEndpointStatus::Absorbed)return {0,0,0,0};
                Real const u2=ux[i]*ux[i]+uy[i]*uy[i]+u[i]*u[i];
                Real const k=weight[i]*mass*u2/(1+std::sqrt(1+u2/(PhysConst::c*PhysConst::c)));
                return {1,weight[i],q*weight[i],k};});}
        auto prediction=data.value();amrex::Long lost_count=amrex::get<0>(prediction);amrex::ParallelDescriptor::ReduceLongSum(lost_count);
        Real predicted[3]={amrex::get<1>(prediction),amrex::get<2>(prediction),amrex::get<3>(prediction)};amrex::ParallelDescriptor::ReduceRealSum(predicted,3);
        // Actual production private endpoint bridge: its NGP moments must
        // match the accepted native endpoint AFTER reflection/absorption.
        std::vector<KineticSpeciesDescriptor> descriptors;
        for(auto const& pc:mpc)if(pc->getCharge()!=0) {
            KineticSpeciesDescriptor desc;desc.name=pc->getName();desc.mass=pc->getMass();
            desc.charge_number=pc->getCharge()/PhysConst::q_e;
            desc.relaxation_excluded=desc.name=="alpha";descriptors.push_back(desc);}
        IonExchangeOptions exchange_options;exchange_options.dt=dt;
        AcceptedIonExchange exchange(model.ElectronThermalGeometry(),cells,layout.DistributionMap(),descriptors,exchange_options);
        ImplicitIonEndpointTrial private_endpoint(exchange);
        auto const before_private=radial_fixture::NativeStateBytes::Capture(w);
#ifndef AMREX_USE_GPU
        amrex::ResetRandomSeed(712);Real const next_random=amrex::Random();amrex::ResetRandomSeed(712);
#endif
        AMREX_ALWAYS_ASSERT(RefreshNativeImplicitIonEndpointTrial(private_endpoint,mpc));
#ifndef AMREX_USE_GPU
        AMREX_ALWAYS_ASSERT(amrex::Random()==next_random);
#endif
        amrex::MultiFab private_moments(cells,layout.DistributionMap(),7*int(descriptors.size()),0);
        amrex::MultiFab accepted_moments(cells,layout.DistributionMap(),private_moments.nComp(),0);
        amrex::MultiFab private_repeat(cells,layout.DistributionMap(),private_moments.nComp(),0);
        amrex::MultiFab accepted_repeat(cells,layout.DistributionMap(),private_moments.nComp(),0);
        private_moments.setVal(0.);accepted_moments.setVal(0.);private_repeat.setVal(0.);accepted_repeat.setVal(0.);
        auto const private_inputs=radial_fixture::EndpointSourceTuples::Private(private_endpoint);
        for(int repeat=0;repeat<2;++repeat) {
        for(auto const& v:private_endpoint.Views()) {
            auto out=(repeat ? private_repeat : private_moments)[v.grid_index].array();int const base=7*int(v.species);
            amrex::For(v.count,[=] AMREX_GPU_DEVICE(long i) {
                auto const cell=v.cell[i];Real const weight=v.weight[i];
                amrex::Gpu::Atomic::AddNoRet(&out(cell,base),weight);
                for(int d=0;d<3;++d) {
                    amrex::Gpu::Atomic::AddNoRet(&out(cell,base+1+d),weight*v.momentum[d][i]);
                    amrex::Gpu::Atomic::AddNoRet(&out(cell,base+4+d),weight*v.momentum[d][i]*v.momentum[d][i]);}
            });}
        }
        AMREX_ALWAYS_ASSERT(RefreshNativeImplicitIonEndpointTrial(private_endpoint,mpc));
        private_inputs.RequireEqual(radial_fixture::EndpointSourceTuples::Private(private_endpoint),"private_refresh_repeat");
        // Optional negative fixture: one changed weight bit must fail the raw
        // snapshot even when its density change would lie below any norm gate.
        bool inject_byte=false;amrex::ParmParse("radial_test").query("inject_byte_mutation",inject_byte);
        if(inject_byte) {
            for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti) {
                auto* weight=pti.GetAttribs(PIdx::w).dataPtr();
                amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                    if(i==0)weight[i]=std::nextafter(weight[i],std::numeric_limits<Real>::infinity());});
            }
        }
        before_private.RequireUnchanged(w,"private_endpoint_twice");
        auto pure_charge=make();AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,pure_charge,Subset::All));
        amrex::MultiFab::Subtract(pure_charge,virt,0,0,1,0);Real const input_density_error=pure_charge.norminf();
        amrex::Print()<<std::setprecision(17)<<"NATIVE_DEPOSIT_AFTER_PRIVATE absolute="<<input_density_error
            <<" independent_repeat="<<virtual_repeat_error<<" reference="<<virt.norminf()<<std::endl;
        deposition.Check(pure_charge,virt,"after_private");
        // Synthetic field negatives exercise the bound independently of the
        // byte snapshot. No accepted state is changed by either control.
        std::string inject_density;amrex::ParmParse("radial_test").query("inject_density_mutation",inject_density);
        if(!inject_density.empty()) {
            auto corrupt=make(),reference=make();corrupt.setVal(0.);reference.setVal(0.);
            if(inject_density=="nonzero") {
                reference.setVal(1.);corrupt.setVal(2*deposition.relative_bound());
            } else {
                AMREX_ALWAYS_ASSERT(inject_density=="empty");corrupt.setVal(1.e-30);
            }
            deposition.Check(corrupt,reference,"injected_"+inject_density);
        }
        // A rejected trial can still restore exact accepted particles after
        // every pure endpoint/map evaluation. Recreate the same midpoint A.
        w.RestoreParticlesAtImplicitStepStart();
        accepted_particles.RequireUnchanged(w,"rollback_particles",false,false);
        auto restored_charge=make();deposit(w,restored_charge);
        amrex::MultiFab::Subtract(restored_charge,old,0,0,1,0);
        Real const rollback_density_error=restored_charge.norminf();
        deposition.Check(restored_charge,old,"rollback");
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto set=SetParticlePosition(pti);
            auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();
            auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i){
                amrex::ParticleReal x,y,z;pos(i,x,y,z);
                Real const inv=1/std::sqrt(1+(ux[i]*ux[i]+uy[i]*uy[i]+uz[i]*uz[i])/(PhysConst::c*PhysConst::c));
                set(i,x+.5*dt*ux[i]*inv,y+.5*dt*uy[i]*inv,z+.5*dt*uz[i]*inv);});}
        before_private.RequireUnchanged(w,"recreated_midpoint");
        auto retry_charge=make();AMREX_ALWAYS_ASSERT(MassMatrixDensityProjection::DepositEndpointSubset(w,0,retry_charge,Subset::All));
        amrex::MultiFab::Subtract(retry_charge,virt,0,0,1,0);
        Real const retry_density_error=retry_charge.norminf();
        deposition.Check(retry_charge,virt,"retry");
        w.FinishImplicitParticleUpdate(dt);mpc.ApplyBoundaryConditions();
        auto& buffer=w.GetParticleBoundaryBuffer();buffer.gatherParticlesFromDomainBoundaries(mpc,dt);
        mpc.Redistribute();deposit(w,actual);
        amrex::ReduceOps<amrex::ReduceOpMax> image_op;amrex::ReduceData<int> image_data(image_op);
        image_op.eval(1,image_data,[] AMREX_GPU_DEVICE(int){return amrex::GpuTuple<int>{0};});
        for(auto const& pc:mpc)for(WarpXParIter pti(*pc,0);pti.isValid();++pti){
            auto pos=GetParticlePosition(pti);auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();
            auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
            amrex::GpuArray<const amrex::ParticleReal*,5> expected{pti.GetAttribs("wall_r").dataPtr(),pti.GetAttribs("wall_theta").dataPtr(),
                pti.GetAttribs("wall_ux").dataPtr(),pti.GetAttribs("wall_uy").dataPtr(),pti.GetAttribs("wall_uz").dataPtr()};
            image_op.eval(pti.numParticles(),image_data,[=] AMREX_GPU_DEVICE(long i){
                amrex::ParticleReal r,theta,z;pos.AsStored(i,r,theta,z);
                return amrex::GpuTuple<int>{int(r!=expected[0][i] || theta!=expected[1][i] ||
                    ux[i]!=expected[2][i] || uy[i]!=expected[3][i] || uz[i]!=expected[4][i])};});}
        int image_error=amrex::get<0>(image_data.value());amrex::ParallelDescriptor::ReduceIntMax(image_error);
        AMREX_ALWAYS_ASSERT(image_error==0);
        auto const native_inputs=radial_fixture::EndpointSourceTuples::Native(w,descriptors);
        private_inputs.RequireEqual(native_inputs,"private_actual_native");
        for(int repeat=0;repeat<2;++repeat) {
        std::size_t species_index=0;
        for(auto const& pc:mpc) {
            if(pc->getCharge()==0)continue;
            auto const& desc=descriptors[species_index];int const base=7*int(species_index++);
            if(desc.relaxation_excluded)continue;
            auto const plo=g.ProbLoArray(),dxi=g.InvCellSizeArray();
            for(WarpXParIter pti(*pc,0);pti.isValid();++pti) {
                auto pos=GetParticlePosition(pti);auto out=(repeat ? accepted_repeat : accepted_moments)[pti.index()].array();
                auto const* weight=pti.GetAttribs(PIdx::w).dataPtr();
                amrex::GpuArray<const amrex::ParticleReal*,3> u{pti.GetAttribs(PIdx::ux).dataPtr(),
                    pti.GetAttribs(PIdx::uy).dataPtr(),pti.GetAttribs(PIdx::uz).dataPtr()};
                amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i) {
                    amrex::ParticleReal r,theta,z;pos.AsStored(i,r,theta,z);
                    amrex::IntVect const cell(int(std::floor((r-plo[0])*dxi[0])),int(std::floor((z-plo[1])*dxi[1])));
                    amrex::Gpu::Atomic::AddNoRet(&out(cell,base),weight[i]);
                    for(int d=0;d<3;++d) {
                        amrex::Gpu::Atomic::AddNoRet(&out(cell,base+1+d),weight[i]*u[d][i]);
                        amrex::Gpu::Atomic::AddNoRet(&out(cell,base+4+d),weight[i]*u[d][i]*u[d][i]);}
                });}
        }
        }
        radial_fixture::ReportMomentDifferences(private_moments,private_repeat,descriptors,private_inputs,"private_repeat");
        radial_fixture::ReportMomentDifferences(accepted_moments,accepted_repeat,descriptors,native_inputs,"native_repeat");
        radial_fixture::ReportMomentDifferences(private_moments,accepted_moments,descriptors,private_inputs,"private_native");
        private_inputs.CheckMomentSummation(private_moments,private_repeat,"private_repeat");
        native_inputs.CheckMomentSummation(accepted_moments,accepted_repeat,"native_repeat");
        private_inputs.CheckMomentSummation(private_moments,accepted_moments,"private_native");
        std::string inject_moment;amrex::ParmParse("radial_test").query("inject_moment_mutation",inject_moment);
        if(!inject_moment.empty()) {
            amrex::MultiFab corrupt(cells,layout.DistributionMap(),accepted_moments.nComp(),0);
            amrex::MultiFab::Copy(corrupt,accepted_moments,0,0,corrupt.nComp(),0);
            private_inputs.InjectMomentCorruption(corrupt,inject_moment);
            private_inputs.CheckMomentSummation(corrupt,accepted_moments,"injected_"+inject_moment);
        }
        Real private_error=0,positive_moment_error=0;
        for(int c=0;c<private_moments.nComp();++c) {
            Real const reference=std::max(accepted_moments.norminf(c),Real(1.e-30));
            amrex::MultiFab::Subtract(private_moments,accepted_moments,c,c,1,0);
            Real const error=private_moments.norminf(c)/reference;
            private_error=std::max(private_error,error);
            if(c%7==0 || c%7>=4)positive_moment_error=std::max(positive_moment_error,error);
        }
#ifndef AMREX_USE_GPU
        AMREX_ALWAYS_ASSERT(private_error<2.e-13);
#else
        // The exact tuple gate and count-derived signed-sum bound replace
        // the invalid GPU normalization by a cancelling first-moment sum.
        // Preserve the original relative gate for noncancelling moments.
        AMREX_ALWAYS_ASSERT(positive_moment_error<2.e-13);
#endif
        amrex::Print()<<std::setprecision(17)<<"ENDPOINT_MOMENT_REFERENCE_RATIO all="<<private_error
            <<" noncancelling="<<positive_moment_error<<std::endl;
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
        amrex::Print()<<std::setprecision(17)<<"NATIVE_REFLECTION_PASS reflected="<<reflected_count<<" image_bitwise_error="<<image_error<<" lost="<<lost_count<<" buffers="<<buffered
            <<" private_endpoint_reference_ratio="<<private_error<<" input_density_error="<<input_density_error
            <<" rollback_density_error="<<rollback_density_error<<" retry_density_error="<<retry_density_error
            <<" partition="<<partition<<" continuity_virtual="<<continuity<<" continuity_survivor_sink="<<sink_continuity
            <<" survivor="<<survivor_error<<" deposited_charge="<<charge<<" native_charge="<<tallies[1]<<"\n";
        w.DiscardSavedImplicitParticleState();
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
