/* Diagnostic adapter. Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL. */
#include "NativeInstantaneousIonCurrent.H"
#include "DarwinVacuumJointSolve.H"
#include "ImplicitParticleEndpointAudit.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Deposition/EsirkepovCurrentRate.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
namespace warpx::particles {
bool DepositNativeInstantaneousIonCurrent(WarpX& sim,InstantaneousIonState state,
                                          ablastr::fields::VectorField const& current){
 using Real=amrex::Real;using P=amrex::ParticleReal;using warpx::fields::FieldType;
 using Joint=warpx::thermal::DarwinVacuumJointSolve;
 auto const axial_lease=Joint::CaptureAxialProducerLease(sim);
#if defined(WARPX_DIM_RZ)
 constexpr int Dim=2;
#elif defined(WARPX_DIM_3D)
 constexpr int Dim=3;
#else
 amrex::Abort("Instantaneous ion companion supports RZ/Cartesian3D only");constexpr int Dim=3;
#endif
 AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxLevel()==0&&!EB::enabled()&&!sim.getdo_moving_window()&&WarpX::grid_type==GridType::Staggered&&WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov&&WarpX::nox==3&&WarpX::ncomps==1&&!WarpX::use_filter&&!WarpX::do_shared_mem_current_deposition,
 "Instantaneous ion companion requires single-level static no-EB m0/order3 unfiltered native Esirkepov Yee layout");
 auto const original=sim.m_fields.get_alldirs(FieldType::current_fp,0);
 for(int c=0;c<3;++c){AMREX_ALWAYS_ASSERT(current[c]&&current[c]!=original[c]&&current[c]->boxArray()==original[c]->boxArray()&&current[c]->DistributionMap()==original[c]->DistributionMap()&&current[c]->nComp()==1&&current[c]->nGrowVect().allGE(sim.get_ng_depos_J()));current[c]->setVal(0.);}
 if(!axial_lease.valid)return false;bool metadata_ok=true;
 auto const& geom=sim.Geom(0);auto const inverse=WarpX::InvCellSize(0);bool const old=state==InstantaneousIonState::Old,endpoint=state==InstantaneousIonState::VirtualEndpoint;
 auto const plo=geom.ProbLoArray(),phi=geom.ProbHiArray();amrex::GpuArray<int,AMREX_SPACEDIM> periodic{};for(int d=0;d<AMREX_SPACEDIM;++d)periodic[d]=geom.isPeriodic(d);
 amrex::Gpu::DeviceScalar<int> invalid(0);int* bad=invalid.dataPtr();
 for(auto const& name:sim.GetPartContainer().GetSpeciesNames()){
  auto&pc=sim.GetPartContainer().GetParticleContainerFromName(name);Real const q=pc.getCharge();if(q==0.)continue;
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dynamic_cast<PhysicalParticleContainer*>(&pc)&&!dynamic_cast<SubcycledParticleContainer*>(&pc)&&!pc.DoFieldIonization()&&!pc.do_not_deposit,
   "Instantaneous ion companion requires deposited fixed-charge physical species; no subcycled container");
  if(old||endpoint){auto const names=pc.GetRealSoANames();for(auto const* n:{"x_n","y_n","z_n","ux_n","uy_n","uz_n"})AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::find(names.begin(),names.end(),n)!=names.end(),"Instantaneous ion companion requires saved native old phase state");}
  bool const axial_enabled=axial_lease.enabled&&endpoint;
  auto const axial=warpx::implicit::MakeAxialEndpointBoundary(geom,pc.GetParticleBoundaryData(),
      WarpX::field_boundary_lo,WarpX::field_boundary_hi,axial_enabled);
  if(axial_enabled){
   auto const names=pc.GetRealSoANames();bool local=axial.enabled&&pc.HasiAttrib("diagnostic_impulse_valid")&&pc.HasiAttrib("esirkepov_chord_valid");
   for(auto const* name:{"implicit_final_gather_x","implicit_final_gather_y","implicit_final_gather_z"})local=local&&std::find(names.begin(),names.end(),name)!=names.end();
   if(!local){metadata_ok=false;continue;}
  }
  for(WarpXParIter pti(pc,0);pti.isValid();++pti){
   auto const pos=GetParticlePosition<PIdx>(pti);auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const*uy=pti.GetAttribs(PIdx::uy).dataPtr();auto const*uz=pti.GetAttribs(PIdx::uz).dataPtr();auto const*w=pti.GetAttribs(PIdx::w).dataPtr();
   amrex::GpuArray<P const*,3> xn{},un{};if(old||endpoint){xn={pti.GetAttribs("x_n").dataPtr(),pti.GetAttribs("y_n").dataPtr(),pti.GetAttribs("z_n").dataPtr()};un={pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),pti.GetAttribs("uz_n").dataPtr()};}
   amrex::GpuArray<P const*,3> gather{};warpx::implicit::EndpointTileContract axial_support;
   int const* impulse_valid=nullptr;int const* chord_valid=nullptr;
   if(axial_enabled){
    gather={pti.GetAttribs("implicit_final_gather_x").dataPtr(),pti.GetAttribs("implicit_final_gather_y").dataPtr(),pti.GetAttribs("implicit_final_gather_z").dataPtr()};
    axial_support=warpx::implicit::MakeNativeAxialCurrentContract(sim,pti.tilebox(),pti.index());
    impulse_valid=pti.GetiAttribs("diagnostic_impulse_valid").dataPtr();chord_valid=pti.GetiAttribs("esirkepov_chord_valid").dataPtr();
   }
   int const* suborbits=pc.HasiAttrib("nsuborbits")?pti.GetiAttribs("nsuborbits").dataPtr():nullptr;
   auto tile=pti.tilebox();tile.grow(sim.get_ng_depos_J());auto const origin=WarpX::LowerCorner(tile,0,0.);auto const lo=amrex::lbound(tile);
   auto const j0=current[0]->array(pti),j1=current[1]->array(pti),j2=current[2]->array(pti);
   amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long p){
    P xp,yp,zp;pos(p,xp,yp,zp);P up=ux[p],vp=uy[p],wp=uz[p];
    if(old){xp=xn[0][p];yp=xn[1][p];zp=xn[2][p];up=un[0][p];vp=un[1][p];wp=un[2][p];}
    else if(endpoint){xp=2.*xp-xn[0][p];yp=2.*yp-xn[1][p];zp=2.*zp-xn[2][p];up=2.*up-un[0][p];vp=2.*vp-un[1][p];wp=2.*wp-un[2][p];
#if defined(WARPX_DIM_RZ)
     // Native Finish calls SetPosition; the next Current read calls GetPosition.
     P const radius=std::sqrt(xp*xp+yp*yp),angle=std::atan2(yp,xp);xp=radius*std::cos(angle);yp=radius*std::sin(angle);
#endif
    }
    bool ok=std::isfinite(xp)&&std::isfinite(yp)&&std::isfinite(zp)&&std::isfinite(up)&&std::isfinite(vp)&&std::isfinite(wp)&&std::isfinite(w[p])&&w[p]>=0.&&(!suborbits||suborbits[p]<=1);
    P const gamma=std::sqrt(1.+(up*up+vp*vp+wp*wp)/(PhysConst::c*PhysConst::c));P const vx=up/gamma,vy=vp/gamma,vz=wp/gamma;
    amrex::GpuArray<double,3>x{},v{},acceleration{};double vt=0.;
#if defined(WARPX_DIM_RZ)
    bool axial_image=false;
    if(axial_enabled&&!periodic[1]&&!(zp>=plo[1]&&zp<phi[1])&&impulse_valid[p]&&chord_valid[p]){
     P const r_old=std::sqrt(xn[0][p]*xn[0][p]+xn[1][p]*xn[1][p]);
     warpx::implicit::AxialEndpointImage image;
     axial_image=r_old>0.&&r_old>=plo[0]&&r_old<phi[0]&&xn[2][p]>=plo[1]&&xn[2][p]<phi[1]&&
        warpx::implicit::endpoint_detail::axialCurrentFits({xp,yp,zp},{up,vp,wp},
            {gather[0][p],gather[1][p],gather[2][p]},axial_support,axial,image);
    }
    P const rr=std::sqrt(xp*xp+yp*yp);ok=ok&&rr>0.&&rr>=plo[0]&&rr<phi[0]&&(periodic[1]||(zp>=plo[1]&&zp<phi[1])||axial_image);
    x={(rr-origin.x)*inverse.x,0.,(zp-origin.z)*inverse.z};v={(xp*vx+yp*vy)/rr*inverse.x,0.,vz*inverse.z};vt=(-yp*vx+xp*vy)/rr;
#else
    ok=ok&&(periodic[0]||(xp>=plo[0]&&xp<phi[0]))&&(periodic[1]||(yp>=plo[1]&&yp<phi[1]))&&(periodic[2]||(zp>=plo[2]&&zp<phi[2]));
    x={(xp-origin.x)*inverse.x,(yp-origin.y)*inverse.y,(zp-origin.z)*inverse.z};v={vx*inverse.x,vy*inverse.y,vz*inverse.z};
#endif
    if(!ok){amrex::Gpu::Atomic::Max(bad,1);return;}
    constexpr int N=3;int first[3]{};for(int d=0;d<3;++d)first[d]=int(x[d])-N/2;
    for(int k=first[2];k<=first[2]+N;++k)for(int j=(Dim==3?first[1]:0);j<=(Dim==3?first[1]+N:0);++j)for(int i=first[0];i<=first[0]+N;++i){
     amrex::GpuArray<double,3>value{},unused{};if(!EsirkepovCurrentRate<N,Dim>(x,v,acceleration,{i,j,k},vt,0.,value,unused)){amrex::Gpu::Atomic::Max(bad,1);return;}
     int const ii=i+lo.x,jj=Dim==3?j+lo.y:k+lo.y,kk=Dim==3?k+lo.z:0;
     Real const a=q*w[p]*inverse.y*inverse.z,b=q*w[p]*(Dim==3?inverse.x*inverse.z:inverse.x*inverse.y*inverse.z),c=q*w[p]*inverse.x*inverse.y;
     if(i<first[0]+N){if(!j0.contains(ii,jj,kk)){amrex::Gpu::Atomic::Max(bad,1);return;}amrex::Gpu::Atomic::AddNoRet(&j0(ii,jj,kk),a*value[0]);}
     if(Dim!=3||j<first[1]+N){if(!j1.contains(ii,jj,kk)){amrex::Gpu::Atomic::Max(bad,1);return;}amrex::Gpu::Atomic::AddNoRet(&j1(ii,jj,kk),b*value[1]);}
     if(k<first[2]+N){if(!j2.contains(ii,jj,kk)){amrex::Gpu::Atomic::Max(bad,1);return;}amrex::Gpu::Atomic::AddNoRet(&j2(ii,jj,kk),c*value[2]);}
    }
   });
  }
 }
 int bad_host=invalid.dataValue()||!metadata_ok||!Joint::AxialProducerLeaseCurrentLocal(sim,axial_lease);amrex::ParallelDescriptor::ReduceIntMax(bad_host);if(bad_host){for(auto*f:current)f->setVal(0.);return false;}
#if defined(WARPX_DIM_RZ)
 sim.ApplyInverseVolumeScalingToCurrentDensity(current[0],current[1],current[2],0);
#endif
 for(auto*f:current)ablastr::utils::communication::SumBoundary(*f,0,1,f->nGrowVect(),f->nGrowVect(),WarpX::do_single_precision_comms,geom.periodicity());
 sim.ApplyJfieldBoundary(0,current[0],current[1],current[2],PatchType::fine);
 for(auto*f:current){f->OverrideSync(geom.periodicity());f->FillBoundary(geom.periodicity());}
 return true;
}
}
