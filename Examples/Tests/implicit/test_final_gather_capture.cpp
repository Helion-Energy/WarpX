/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Diagnostics/MultiDiagnostics.H"
#include "Initialization/WarpXInit.H"
#ifndef FINAL_GATHER_BASELINE
#include "FieldSolver/ImplicitSolvers/ImplicitIonElectricWork.H"
#endif
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/ImplicitFinalGather.H"
#include "Particles/Pusher/PushSelector.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <fstream>
#include <iomanip>

using namespace amrex::literals;
using PR=amrex::ParticleReal;
using warpx::particles::FinalGatherAttributeNames;

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w=WarpX::GetInstance();
        w.InitData();
        auto& pc=dynamic_cast<PhysicalParticleContainer&>(w.GetPartContainer().GetParticleContainerFromName("ions"));
        amrex::ParmParse pp("gather_test");
        int iterations=1; bool cropped=false, checkpoint=false;
        std::string negative;
        pp.query("iterations",iterations); pp.query("cropped",cropped);
        pp.query("checkpoint",checkpoint); pp.query("negative",negative);
        auto const& names=pc.GetRealSoANames();
        bool const recorded=std::find(names.begin(),names.end(),FinalGatherAttributeNames[0])!=names.end();
#ifndef FINAL_GATHER_BASELINE
        if (negative=="late_no_push") { pc.setDoNotPush(true); pc.ValidateImplicitIonElectricWorkCapture(); }
        if (negative=="late_no_gather") { pc.setDoNotGather(true); }
        if (negative=="direct_suborbit") { pc.AddIntComp("nsuborbits",0); }
        if (negative=="variable_charge") { pc.AddIntComp("ionizationLevel",0); pc.ValidateImplicitIonElectricWorkCapture(); }
#endif
        for (auto const* name:FinalGatherAttributeNames) {
            auto it=std::find(names.begin(),names.end(),name);
            AMREX_ALWAYS_ASSERT((it!=names.end())==recorded);
            if (recorded) { AMREX_ALWAYS_ASSERT(pc.h_redistribute_real_comp[it-names.begin()]==0); }
        }
        auto const& geom=w.Geom(0);
        auto const dx=geom.CellSizeArray(), plo=geom.ProbLoArray(), phi=geom.ProbHiArray();
        auto const origin_index=geom.Domain().smallEnd();
        auto const dinv=WarpX::InvCellSize(0);
        amrex::MultiFab f(amrex::convert(pc.ParticleBoxArray(0),amrex::IntVect::TheNodeVector()),pc.ParticleDistributionMap(0),6,5);
        for (amrex::MFIter mfi(f);mfi.isValid();++mfi) {
            auto arr=f.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto x=plo[0]+(i-origin_index[0])*dx[0];
#if defined(WARPX_DIM_RZ)
                auto z=plo[1]+(j-origin_index[1])*dx[1];
                auto y=0.;
#else
                auto y=plo[1]+(j-origin_index[1])*dx[1];
                auto z=plo[2]+(k-origin_index[2])*dx[2];
#endif
                arr(i,j,k,0)=1.e9*(.3+.2*x+.1*z);
                arr(i,j,k,1)=1.e9*(-.12+.1*y-.05*z);
                arr(i,j,k,2)=1.e9*(.08+.3*z+.07*x);
                arr(i,j,k,3)=.03+.01*x;
                arr(i,j,k,4)=-.01+.03*z;
                arr(i,j,k,5)=.1+.01*y;
            });
        }
        std::ofstream dump("physics-rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".txt");
        dump<<std::hexfloat;
        amrex::Real max_error=0., max_midpoint_gap=0.;
        amrex::Long tested=0, crop_tested=0;
        for (WarpXParIter pti(pc,0);pti.isValid();++pti) {
            auto& tile=pti.GetParticleTile(); tile.resize(7);
            auto data=tile.getParticleTileData();
            auto set=SetParticlePosition<PIdx>(pti); auto get=GetParticlePosition<PIdx>(pti);
            auto const tilebox=pti.tilebox();
            auto lo=tilebox.smallEnd(), hi=tilebox.bigEnd();
            bool const at_wall=hi[0]==geom.Domain().bigEnd(0);
            amrex::GpuArray<PR*,3> oldx{pti.GetAttribs("x_n").dataPtr(),pti.GetAttribs("y_n").dataPtr(),pti.GetAttribs("z_n").dataPtr()};
            amrex::GpuArray<PR*,3> oldu{pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),pti.GetAttribs("uz_n").dataPtr()};
            amrex::GpuArray<PR*,3> capture{};
            if (recorded) for(int d=0;d<3;++d) { capture[d]=pti.GetAttribs(FinalGatherAttributeNames[d]).dataPtr(); }
            auto const grid=pti.index(); auto const rank=amrex::ParallelDescriptor::MyProc();
            amrex::ParallelFor(7,[=] AMREX_GPU_DEVICE(long p) {
                for(int d=0;d<PIdx::nattribs;++d) { data.rdata(d)[p]=0.; }
                auto particle=data[p]; particle.id()=100*grid+p+1; particle.cpu()=rank;
                PR x=plo[0]+(.5*(lo[0]+hi[0]+1)-origin_index[0]+.003*p)*dx[0];
#if defined(WARPX_DIM_RZ)
                PR z=plo[1]+(.5*(lo[1]+hi[1]+1)-origin_index[1]+.002*p)*dx[1];
                PR angle=.13+.17*p;
                if(cropped&&at_wall) { x=phi[0]-.01*dx[0]; }
                PR y=x*std::sin(angle); x*=std::cos(angle);
#else
                PR y=plo[1]+(.5*(lo[1]+hi[1]+1)-origin_index[1]+.002*p)*dx[1];
                PR z=plo[2]+(.5*(lo[2]+hi[2]+1)-origin_index[2]+.004*p)*dx[2];
                if(cropped&&at_wall) { x=phi[0]-.01*dx[0]; }
#endif
                set(p,x,y,z); get(p,oldx[0][p],oldx[1][p],oldx[2][p]);
                PR ux=2.e6+.1e6*p, uy=-1.e6, uz=.5e6;
                if(cropped&&at_wall) {
#if defined(WARPX_DIM_RZ)
                    ux=1.5e7*std::cos(angle); uy=1.5e7*std::sin(angle);
#else
                    ux=1.5e7;
#endif
                }
                oldu[0][p]=ux; oldu[1][p]=uy; oldu[2][p]=uz;
                data.rdata(PIdx::ux)[p]=ux; data.rdata(PIdx::uy)[p]=uy; data.rdata(PIdx::uz)[p]=uz;
                data.rdata(PIdx::w)[p]=3.+p;
                if(capture[0]) for(int d=0;d<3;++d) { capture[d][p]=999.; }
            });
            std::array<amrex::FArrayBox,6> fields;
            for(int c=0;c<6;++c) { fields[c]=amrex::FArrayBox(f[pti],amrex::make_alias,c,1); }
            ImplicitOptions options; options.max_particle_iterations=iterations; options.particle_tolerance=1.e-10_prt;
            if(negative=="late_reflection") { options.reflect_particles_at_rmax=true; }
            if(negative=="zero_iterations") { options.max_particle_iterations=0; }
            if(negative=="late_suborbit") { options.evolve_suborbit_particles_only=true; }
            long unconverged=0; amrex::Gpu::DeviceVector<long> indices;
            amrex::Gpu::DeviceVector<PR> weights;
            constexpr amrex::Real dt=2.e-10;
            // The native empty-range exit must not inspect particle attrs or
            // change its existing behavior, even for inconsistent options.
            ImplicitOptions empty_options; empty_options.max_particle_iterations=0;
            empty_options.reflect_particles_at_rmax=true;
            pc.ImplicitPushXP(pti,&fields[0],&fields[1],&fields[2],&fields[3],&fields[4],&fields[5],
                &empty_options,amrex::IntVect(5),1,0,0,0,dt,unconverged,indices,weights);
            if (negative=="direct_suborbit") {
                pc.ImplicitPushXPSubOrbits(pti,w.m_fields,&fields[0],&fields[1],&fields[2],&fields[3],&fields[4],&fields[5],
                    &options,amrex::IntVect(5),nullptr,nullptr,nullptr,0,0,0,dt,true,1,indices,weights);
            }
            pc.ImplicitPushXP(pti,&fields[0],&fields[1],&fields[2],&fields[3],&fields[4],&fields[5],
                &options,amrex::IntVect(5),1,5,0,0,dt,unconverged,indices,weights);
            auto origin=WarpX::LowerCorner(amrex::grow(pti.tilebox(),5),0,0.);
            auto lower=amrex::lbound(amrex::grow(pti.tilebox(),5));
            amrex::GpuArray<amrex::Array4<const amrex::Real>,6> arrays{};
            for(int c=0;c<6;++c) { arrays[c]=fields[c].const_array(); }
            auto type=amrex::IndexType(amrex::IntVect::TheNodeVector());
            PR const mass=pc.getMass(),charge=pc.getCharge();
            amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpSum,amrex::ReduceOpMax> op;
            amrex::ReduceData<amrex::Real,amrex::Real,amrex::Long,int> rd(op);
            op.eval(7,rd,[=] AMREX_GPU_DEVICE(long p) -> decltype(rd)::Type {
                if(p==0||p==6) {
                    int bad=0;
                    for(int d=0;d<3;++d) { if(capture[0]) bad|=capture[d][p]!=999.; }
                    bad|=data.rdata(PIdx::ux)[p]!=oldu[0][p]||data.rdata(PIdx::uy)[p]!=oldu[1][p]||data.rdata(PIdx::uz)[p]!=oldu[2][p];
                    return {0.,0.,0,bad};
                }
                if(!capture[0]) { return {0.,0.,0,0}; }
                PR gx=capture[0][p],gy=capture[1][p],gz=capture[2][p];
                int bad=!std::isfinite(gx)||!std::isfinite(gy)||!std::isfinite(gz);
                PR ex=0.,ey=0.,ez=0.,bx=0.,by=0.,bz=0.;
                doGatherShapeN<3,0>(gx,gy,gz,ex,ey,ez,bx,by,bz,arrays[0],arrays[1],arrays[2],arrays[3],arrays[4],arrays[5],
                    type,type,type,type,type,type,dinv,origin,lower,1);
                PR ux=oldu[0][p],uy=oldu[1][p],uz=oldu[2][p];
                doParticleMomentumPush<0>(ux,uy,uz,ex,ey,ez,bx,by,bz,1,mass,charge,ParticlePusherAlgo::Boris,false,
#ifdef WARPX_QED
                    0.,
#endif
                    dt,MomentumPushType::Full);
                amrex::Real err=std::max({std::abs(.5*(ux+oldu[0][p])-data.rdata(PIdx::ux)[p]),
                    std::abs(.5*(uy+oldu[1][p])-data.rdata(PIdx::uy)[p]),std::abs(.5*(uz+oldu[2][p])-data.rdata(PIdx::uz)[p])}) /
                    std::max({amrex::Real(1.),std::abs(ux),std::abs(uy),std::abs(uz)});
                PR mx,my,mz; get(p,mx,my,mz);
                auto gap=std::max({std::abs(mx-gx),std::abs(my-gy),std::abs(mz-gz)});
                amrex::Long ncrop=0;
                if(cropped&&at_wall) {
#if defined(WARPX_DIM_RZ)
                    auto radius=std::sqrt(gx*gx+gy*gy);
#else
                    auto radius=gx;
#endif
                    bad|=std::abs(radius-phi[0])>32*std::numeric_limits<PR>::epsilon()*phi[0]; ++ncrop;
                } else if(iterations==1) {
                    PR ddx=0.,ddy=0.,ddz=0.;
                    UpdatePositionImplicit(ddx,ddy,ddz,oldu[0][p],oldu[1][p],oldu[2][p],oldu[0][p],oldu[1][p],oldu[2][p],.5*dt);
                    bad|=gx!=oldx[0][p]+ddx||gy!=oldx[1][p]+ddy||gz!=oldx[2][p]+ddz;
                }
                return {err,gap,ncrop,bad};
            });
            auto result=rd.value(); AMREX_ALWAYS_ASSERT(amrex::get<3>(result)==0);
            max_error=std::max(max_error,amrex::get<0>(result));
            max_midpoint_gap=std::max(max_midpoint_gap,amrex::get<1>(result));
            crop_tested+=amrex::get<2>(result); tested+=5;
            // The offset check above leaves end particles untouched. Complete
            // those two separately so the native all-particle bridge can read
            // one recorded final gather for every charged particle.
            for(long offset : {0L,6L}) {
                pc.ImplicitPushXP(pti,&fields[0],&fields[1],&fields[2],&fields[3],&fields[4],&fields[5],
                    &options,amrex::IntVect(5),offset,1,0,0,dt,unconverged,indices,weights);
            }
            for(int c=0;c<PIdx::nattribs;++c) {
                amrex::Gpu::HostVector<PR> host(7);
                auto const* src=data.rdata(c);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,src,src+7,host.begin());
                for(int p=0;p<7;++p) { dump<<grid<<' '<<p<<' '<<c<<' '<<host[p]<<'\n'; }
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(max_error);
        amrex::ParallelDescriptor::ReduceRealMax(max_midpoint_gap);
        amrex::ParallelDescriptor::ReduceLongSum(tested);
        amrex::ParallelDescriptor::ReduceLongSum(crop_tested);
        AMREX_ALWAYS_ASSERT(tested>0);
        if(recorded) {
            AMREX_ALWAYS_ASSERT(max_error<128*std::numeric_limits<PR>::epsilon());
            if(iterations==1) { AMREX_ALWAYS_ASSERT(max_midpoint_gap>1.e-9); }
            if(cropped) { AMREX_ALWAYS_ASSERT(crop_tested>0); }
        }
        amrex::Print()<<std::setprecision(17)<<"FINAL_GATHER_PASS recorded="<<recorded<<" particles="<<tested
            <<" cropped="<<crop_tested<<" Boris_relative_error="<<max_error<<" stored_midpoint_gap_m="<<max_midpoint_gap<<'\n';
#ifndef FINAL_GATHER_BASELINE
        std::array<amrex::MultiFab,3> electric;
        warpx::thermal::IonElectricWorkFields fields;
        for(int d=0;d<3;++d) {
            electric[d]=amrex::MultiFab(f,amrex::make_alias,d,1);
            fields.component[d]=&electric[d]; fields.total[d]=&electric[d];
        }
        fields.filled_ghosts=amrex::IntVect(2);
        using namespace warpx::thermal;
        auto const contract=NativeIonWorkContract::FullBorisGatherWithoutAdditionalMomentumChanges;
        auto result=MeasureNativeImplicitIonElectricWork(w,0,w.getdt(0),fields,contract);
        AMREX_ALWAYS_ASSERT(result.gather_contract==IonElectricGatherContract::RecordedFinalGather);
        AMREX_ALWAYS_ASSERT(result.valid==recorded);
        if(recorded) {
            using C=IonElectricWorkComponent;
            AMREX_ALWAYS_ASSERT(result.species.size()==1);
            auto const& work=result.species[0];
            auto scale=std::max(work[C::KineticChangeAbs],work[C::TotalFieldWorkAbs]);
            auto const relative=work[C::TotalWorkDefectAbs]/scale;
            AMREX_ALWAYS_ASSERT(relative<2.e-12);
            auto replay=MeasureNativeImplicitIonElectricWork(w,0,w.getdt(0),fields,contract);
            AMREX_ALWAYS_ASSERT(replay.valid && result.species==replay.species);
            auto reconstructed=MeasureNativeReconstructedIonElectricWork(w,0,w.getdt(0),fields,contract);
            AMREX_ALWAYS_ASSERT(reconstructed.gather_contract==IonElectricGatherContract::FinalMidpointReconstruction);
            if(!cropped && iterations==1) {
                AMREX_ALWAYS_ASSERT(reconstructed.valid);
                AMREX_ALWAYS_ASSERT(reconstructed.species[0][C::TotalWorkDefectAbs]>1.e-6*scale);
            }
            amrex::Print()<<"FINAL_GATHER_NATIVE_BRIDGE exact_relative_work_defect="<<relative
                <<" reconstructed_absolute_defect="<<(reconstructed.valid?reconstructed.species[0][C::TotalWorkDefectAbs]:-1.)<<'\n';
        }
#endif
        if(checkpoint) { w.GetMultiDiags().NewIteration(); w.GetMultiDiags().FilterComputePackFlush(0,true); }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
