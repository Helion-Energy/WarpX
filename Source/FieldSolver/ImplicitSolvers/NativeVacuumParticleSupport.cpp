/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeVacuumParticleSupport.H"
#include "NativeEndpointField.H"
#include "Particles/Deposition/EsirkepovAdjacentInstantaneousCurrentIncrement.H"
#include "ImplicitAxialEndpoint.H"
#include "ImplicitParticleEndpointAudit.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Deposition/EsirkepovInstantaneousCurrentIncrement.H"
#include "Particles/Deposition/EsirkepovAxialCurrentIncrement.H"
#include "Particles/Pusher/NativeBorisImpulse.H"
#include "Particles/Pusher/ImplicitFinalGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "WarpX.H"
#include <AMReX_GpuAtomic.H>
#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::darwin {
namespace {
using R=amrex::Real;using P=amrex::ParticleReal;
using Point=amrex::GpuArray<double,3>;using Index=amrex::GpuArray<int,3>;
using Masks=amrex::GpuArray<amrex::Array4<int const>,3>;
#if defined(WARPX_DIM_RZ)
constexpr int Dim=2;
#else
constexpr int Dim=3;
#endif
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Point Coordinates(Point const& p,amrex::XDim3 o,amrex::XDim3 inv) {
#if defined(WARPX_DIM_RZ)
 R const radius=std::sqrt(p[0]*p[0]+p[1]*p[1]);
 return {(radius-o.x)*inv.x,0.,(p[2]-o.z)*inv.z};
#else
 return {(p[0]-o.x)*inv.x,(p[1]-o.y)*inv.y,(p[2]-o.z)*inv.z};
#endif
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool Start(Point const& q,Index& first) {
 // Invoke the existing positive branch-margin contract before every int cast.
 Point out{},rate{},zero{};
 if(!warpx::particles::EsirkepovCurrentRate<3,Dim>(q,zero,zero,{0,0,0},0.,0.,out,rate))return false;
 first={int(q[0])-1,Dim==3?int(q[1])-1:0,int(q[2])-1};return true;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int Clear(amrex::Array4<int const> const& mask,Index low,Index high,amrex::Dim3 offset) {
 for(int k=low[2];k<=high[2];++k)for(int j=(Dim==3?low[1]:0);j<=(Dim==3?high[1]:0);++j)for(int i=low[0];i<=high[0];++i){
  int const ii=i+offset.x,jj=Dim==3?j+offset.y:k+offset.y,kk=Dim==3?k+offset.z:0;
  if(!mask.contains(ii,jj,kk))return 8;
  if(mask(ii,jj,kk))return 1;
 }return 0;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int CurrentClear(Masks const& masks,Index low,Index high,amrex::Dim3 offset) {
 for(int c=0;c<3;++c){Index upper=high;if(Dim==3||c!=1)--upper[c];int const issue=Clear(masks[c],low,upper,offset);if(issue)return issue;}return 0;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Index Upper(Index first) {for(int d=0;d<3;++d)if(Dim==3||d!=1)first[d]+=3;return first;}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool InDomain(Point const& p,amrex::GpuArray<R,AMREX_SPACEDIM> lo,
              amrex::GpuArray<R,AMREX_SPACEDIM> hi,amrex::GpuArray<int,AMREX_SPACEDIM> per) {
 for(double v:p)if(!std::isfinite(v))return false;
#if defined(WARPX_DIM_RZ)
 R const radius=std::sqrt(p[0]*p[0]+p[1]*p[1]);
 return radius>0.&&radius>=lo[0]&&radius<hi[0]&&(per[1]||(p[2]>=lo[1]&&p[2]<hi[1]));
#else
 for(int d=0;d<3;++d)if(!per[d]&&(p[d]<lo[d]||p[d]>=hi[d]))return false;return true;
#endif
}
std::vector<std::size_t> Layout(WarpX& w,std::vector<std::string> const& names) {
 std::vector<std::size_t> out;
 for(auto const& name:names){auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);if(pc.getCharge()==0.)continue;
  out.push_back(pc.GetParticles(0).size());for(auto const& [key,tile]:pc.GetParticles(0)){
   out.insert(out.end(),{std::size_t(key.first),std::size_t(key.second),std::size_t(tile.numParticles()),std::size_t(tile.numNeighborParticles()),std::size_t(tile.NumRealComps()),std::size_t(tile.NumIntComps()),reinterpret_cast<std::size_t>(tile.GetStructOfArrays().GetIdCPUData().data())});
  }
 }return out;
}
bool AxialMetadata(WarpX& w,std::vector<std::string> const& names,bool enabled) {
 if(!enabled)return true;
 bool ready=true;
 for(auto const& name:names){auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
  ready=warpx::implicit::MakeAxialEndpointBoundary(w.Geom(0),pc.GetParticleBoundaryData(),
      WarpX::field_boundary_lo,WarpX::field_boundary_hi,true).enabled&&ready;
 }
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);return ready;
}
bool Metadata(WarpX& w,std::vector<std::string> const& names) {
 bool ok=names==w.GetPartContainer().GetSpeciesNames();amrex::ParallelDescriptor::ReduceBoolAnd(ok);if(!ok)return false;
 for(auto const& name:names){auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);if(pc.getCharge()==0.)continue;
  ok=ok&&dynamic_cast<PhysicalParticleContainer*>(&pc)&&!dynamic_cast<SubcycledParticleContainer*>(&pc)&&!pc.DoFieldIonization()&&!pc.do_not_deposit&&!pc.HasiAttrib("nsuborbits");
  auto const attributes=pc.GetRealSoANames();
  for(auto const* a:{"x_n","y_n","z_n","ux_n","uy_n","uz_n","esirkepov_chord_x","esirkepov_chord_y","esirkepov_chord_z","diagnostic_du_x","diagnostic_du_y","diagnostic_du_z","implicit_final_gather_x","implicit_final_gather_y","implicit_final_gather_z"})ok=ok&&std::find(attributes.begin(),attributes.end(),a)!=attributes.end();
  ok=ok&&pc.HasiAttrib("diagnostic_impulse_valid")&&pc.HasiAttrib("esirkepov_chord_valid");
 }
 amrex::ParallelDescriptor::ReduceBoolAnd(ok);return ok;
}
}

bool NativeVacuumParticleSupport::FillPullback(WarpX& w,amrex::iMultiFab& f) {
 auto const& g=w.Geom(0);auto const lo=g.Domain().smallEnd(),hi=g.Domain().bigEnd();auto const type=f.ixType().toIntVect();auto const per=g.isPeriodicArray();
 bool ok=f.nComp()==1;for(int d=0;d<AMREX_SPACEDIM;++d)ok=ok&&f.nGrowVect()[d]<=g.Domain().length(d);
 amrex::ParallelDescriptor::ReduceBoolAnd(ok);if(!ok)return false;
 f.OverrideSync(g.periodicity());f.FillBoundary(g.periodicity());
 // Unknown halo cells remain forbidden. Pull back all physical mirrors with
 // coefficient magnitudes; zero native coefficients may be over-covered.
 for(amrex::MFIter it(f);it.isValid();++it){auto q=f.array(it);
  amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){
   int src[3]={i,j,k};bool image=false;
   for(int d=0;d<AMREX_SPACEDIM;++d)if(!per[d]){
    if(src[d]<lo[d]){src[d]=2*lo[d]-(1-type[d])-src[d];image=true;}
    else if(src[d]>hi[d]+type[d]){src[d]=2*(hi[d]+1)-(1-type[d])-src[d];image=true;}
   }
   if(image)q(i,j,k)=q.contains(src[0],src[1],src[2])?q(src[0],src[1],src[2]):1;
  });
 }return true;
}

bool NativeVacuumParticleSupport::Prepare(WarpX& w,amrex::Array<amrex::iMultiFab const*,3> const& vacuum,amrex::MultiFab const& raw,std::string& error,bool allow_axial_reflection) {
 bool adjacent=false;
 if(!TryNativeAdjacentInstantaneousPolicy(w,adjacent)){error="Unsupported or noncollective adjacent current policy";return false;}
 bool ok=!m_ready&&sizeof(R)==sizeof(double)&&sizeof(P)==sizeof(double)&&w.maxLevel()==0&&!EB::enabled()&&!w.getdo_moving_window()&&WarpX::grid_type==GridType::Staggered&&WarpX::nox==3&&WarpX::ncomps==1&&!WarpX::use_filter&&!WarpX::do_shared_mem_current_deposition&&WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov;
 for (auto x : w.m_v_galilean) ok = ok && x == 0.;
 // ShiftGalileanBoundary stores NaN in the unused Cartesian y slot in RZ.
 // Only represented spatial shifts constrain the static support geometry.
#if defined(WARPX_DIM_RZ)
 ok = ok && w.m_galilean_shift[0] == 0. && w.m_galilean_shift[2] == 0.;
#else
 for (auto x : w.m_galilean_shift) ok = ok && x == 0.;
#endif
 auto const& g=w.Geom(0);
 for(int d=0;d<AMREX_SPACEDIM;++d)if(!g.isPeriodic(d)){
  bool lower=WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC||WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
#if defined(WARPX_DIM_RZ)
  lower=lower||(d==0&&g.ProbLo(0)==0.&&WarpX::field_boundary_lo[d]==FieldBoundaryType::None);
#endif
  bool upper=WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC||WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
  ok=ok&&lower&&upper;
 }
 auto const primary=w.m_fields.get_alldirs(warpx::fields::FieldType::current_fp,0);auto const& rho=*w.m_fields.get(warpx::fields::FieldType::rho_fp,0);
 ok=ok&&raw.nComp()==1&&raw.boxArray()==rho.boxArray()&&raw.DistributionMap()==rho.DistributionMap();
 for(int c=0;c<3;++c)ok=ok&&vacuum[c]&&vacuum[c]->boxArray()==primary[c]->boxArray()&&vacuum[c]->DistributionMap()==primary[c]->DistributionMap()&&vacuum[c]->nComp()==1;
 amrex::ParallelDescriptor::ReduceBoolAnd(ok);if(!ok){error="Unsupported support layout/dispatch/boundary";return false;}
 int axial_min=int(allow_axial_reflection),axial_max=axial_min;
 amrex::ParallelDescriptor::ReduceIntMin(axial_min);amrex::ParallelDescriptor::ReduceIntMax(axial_max);
 if(axial_min!=axial_max){error="Inconsistent private axial reflection capability";return false;}
 m_axial_reflection=allow_axial_reflection;
 m_species=w.GetPartContainer().GetSpeciesNames();if(!Metadata(w,m_species)){error="Missing unsplit native record metadata";return false;}
 if(!AxialMetadata(w,m_species,m_axial_reflection)){error="Unsupported private axial boundary metadata";return false;}
 for(int c=0;c<3;++c){m_v[c].define(vacuum[c]->boxArray(),vacuum[c]->DistributionMap(),1,primary[c]->nGrowVect());m_v[c].setVal(1);amrex::iMultiFab::Copy(m_v[c],*vacuum[c],0,0,1,0);if(!FillPullback(w,m_v[c])){error="Current support halo exceeds domain";return false;}}
 m_z.define(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());m_z.setVal(1);
 for(amrex::MFIter it(m_z);it.isValid();++it){auto z=m_z.array(it);auto r=raw.const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){z(i,j,k)=!std::isfinite(r(i,j,k))||r(i,j,k)<=0.;});}
 if(!FillPullback(w,m_z)){error="Density support halo exceeds domain";return false;}
 m_layout=Layout(w,m_species);m_counts.resize(9);m_step=w.getistep(0);m_time=w.gett_new(0);m_adjacent=adjacent;m_ready=true;return true;
}

NativeVacuumParticleSupport::Report NativeVacuumParticleSupport::Check(WarpX& w,Phase phase) {
 Report report;bool adjacent=false;
 if(!TryNativeAdjacentInstantaneousPolicy(w,adjacent))return report;
 bool ready=m_ready&&m_adjacent==adjacent&&m_step==w.getistep(0)&&m_time==w.gett_new(0);
 auto const primary=w.m_fields.get_alldirs(warpx::fields::FieldType::current_fp,0);auto const& rho=*w.m_fields.get(warpx::fields::FieldType::rho_fp,0);
 for(int c=0;c<3;++c)ready=ready&&m_v[c].isDefined()&&m_v[c].boxArray()==primary[c]->boxArray()&&m_v[c].DistributionMap()==primary[c]->DistributionMap()&&m_v[c].nGrowVect()==primary[c]->nGrowVect();
 ready=ready&&m_z.isDefined()&&m_z.boxArray()==rho.boxArray()&&m_z.DistributionMap()==rho.DistributionMap()&&m_z.nGrowVect()==rho.nGrowVect();
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!Metadata(w,m_species))return report;
 ready=m_layout==Layout(w,m_species);amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return report;
 if(!AxialMetadata(w,m_species,m_axial_reflection))return report;
 auto* counts=m_counts.data();amrex::ParallelFor(9,[=]AMREX_GPU_DEVICE(int i){counts[i]=0;});
 auto const inverse=WarpX::InvCellSize(0);auto const& geom=w.Geom(0);auto plo=geom.ProbLoArray(),phi=geom.ProbHiArray();auto per=geom.isPeriodicArray();bool const trial=phase==Phase::StageTrial;
 for(auto const& name:m_species){auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);if(pc.getCharge()==0.)continue;
  auto const axial=warpx::implicit::MakeAxialEndpointBoundary(geom,pc.GetParticleBoundaryData(),
      WarpX::field_boundary_lo,WarpX::field_boundary_hi,m_axial_reflection);
  for(WarpXParIter pti(pc,0);pti.isValid();++pti){auto position=GetParticlePosition<PIdx>(pti);
   amrex::GpuArray<P const*,3> old{pti.GetAttribs("x_n").dataPtr(),pti.GetAttribs("y_n").dataPtr(),pti.GetAttribs("z_n").dataPtr()},chord{pti.GetAttribs("esirkepov_chord_x").dataPtr(),pti.GetAttribs("esirkepov_chord_y").dataPtr(),pti.GetAttribs("esirkepov_chord_z").dataPtr()},gather{pti.GetAttribs("implicit_final_gather_x").dataPtr(),pti.GetAttribs("implicit_final_gather_y").dataPtr(),pti.GetAttribs("implicit_final_gather_z").dataPtr()};
   amrex::GpuArray<P const*,9> momentum{pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),pti.GetAttribs("uz_n").dataPtr(),pti.GetAttribs(PIdx::ux).dataPtr(),pti.GetAttribs(PIdx::uy).dataPtr(),pti.GetAttribs(PIdx::uz).dataPtr(),pti.GetAttribs("diagnostic_du_x").dataPtr(),pti.GetAttribs("diagnostic_du_y").dataPtr(),pti.GetAttribs("diagnostic_du_z").dataPtr()};
   auto const* valid=pti.GetiAttribs("diagnostic_impulse_valid").dataPtr();auto const* cv=pti.GetiAttribs("esirkepov_chord_valid").dataPtr();auto const* weight=pti.GetAttribs(PIdx::w).dataPtr();
   auto tile=pti.tilebox();tile.grow(w.get_ng_depos_J());auto origin=WarpX::LowerCorner(tile,0,0.);auto offset=amrex::lbound(tile);
   auto rhotile=pti.tilebox();rhotile.grow(w.get_ng_depos_rho());auto rho_tile_origin=WarpX::LowerCorner(rhotile,0,0.);auto rho_tile_offset=amrex::lbound(rhotile);
   auto const fab=m_z[pti].box();auto rho_origin=WarpX::LowerCorner(fab,0,0.);auto rho_offset=amrex::lbound(fab);
   Masks masks{m_v[0].const_array(pti),m_v[1].const_array(pti),m_v[2].const_array(pti)};auto empty=m_z.const_array(pti);
   warpx::implicit::EndpointAuditFields audit_fields;
   warpx::implicit::EndpointTileContract audit;
   if(m_axial_reflection){
    audit_fields.endpoint_density=&rho;audit_fields.gather_filled_ghosts=w.get_ng_fieldgather();
    audit_fields.current_deposit_ghosts=w.get_ng_depos_J();
    for(int c=0;c<3;++c){audit_fields.current[c]=primary[c];
        audit_fields.gather_e[c]=w.m_fields.get(warpx::fields::FieldType::Efield_aux,ablastr::fields::Direction{c},0);
        audit_fields.gather_b[c]=w.m_fields.get(warpx::fields::FieldType::Bfield_aux,ablastr::fields::Direction{c},0);}
    audit=warpx::implicit::MakeEndpointTileContract(geom,pti.tilebox(),pti.index(),audit_fields,
        WarpX::particle_boundary_lo,WarpX::particle_boundary_hi);
    warpx::implicit::RestrictAxialGatherToProduced(w,pti.index(),audit);
   }
   bool const axial_scope=m_axial_reflection;
   int const cap_low=geom.Domain().smallEnd(AMREX_SPACEDIM-1),cap_high=geom.Domain().bigEnd(AMREX_SPACEDIM-1)+1;
   amrex::For(pti.numParticles(),[=]AMREX_GPU_DEVICE(long p){
    amrex::Gpu::Atomic::AddNoRet(counts,amrex::Long(1));P xp,yp,zp;position(p,xp,yp,zp);Point stored{xp,yp,zp},xn{old[0][p],old[1][p],old[2][p]},dx{chord[0][p],chord[1][p],chord[2][p]},endpoint=stored;
    if(trial)for(int d=0;d<3;++d)endpoint[d]=2.*stored[d]-xn[d];
    warpx::implicit::AxialEndpointImage image;
    if(axial_scope&&trial){
     Point end_u{momentum[3][p],momentum[4][p],momentum[5][p]};
     for(int d=0;d<3;++d)end_u[d]=2.*end_u[d]-momentum[d][p];
     image=warpx::implicit::MapAxialEndpoint(endpoint,end_u,axial);
    }
    bool const physical_endpoint=InDomain(endpoint,plo,phi,per) ||
        (trial&&axial_scope&&image.valid&&image.reflected);
    bool finite=std::isfinite(weight[p])&&weight[p]>=0.&&InDomain(xn,plo,phi,per)&&physical_endpoint;
    for(int d=0;d<3;++d)finite=finite&&std::isfinite(dx[d])&&std::isfinite(stored[d]);
    for(auto const* component:momentum)finite=finite&&std::isfinite(component[p]);
    if(!finite){amrex::Gpu::Atomic::AddNoRet(counts+1,amrex::Long(1));return;}
    bool records=valid[p]&&cv[p];for(int d=0;d<3;++d)records=records&&std::isfinite(gather[d][p]);
    if(!records){amrex::Gpu::Atomic::AddNoRet(counts+6,amrex::Long(1));return;}
    if(axial_scope&&trial){
     Point const actual_gather{gather[0][p],gather[1][p],gather[2][p]};
     auto const mesh=warpx::implicit::endpoint_detail::mesh(actual_gather);
     for(auto const& band:audit.gather)if(!warpx::implicit::endpoint_detail::shapeFits(mesh,band)){
      amrex::Gpu::Atomic::AddNoRet(counts+7,amrex::Long(1));return;}
    }
    Index begin{},end{};Point x0=Coordinates(xn,origin,inverse),x1=Coordinates(endpoint,origin,inverse);
    if(!Start(x0,begin)||!Start(x1,end)){amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;}
    if(trial){Index low=begin,high=end;for(int d=0;d<3;++d){low[d]=amrex::min(begin[d],end[d]);high[d]=amrex::max(begin[d],end[d]);}high=Upper(high);
     int issue=CurrentClear(masks,low,high,offset);if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:2),amrex::Long(1));return;}
    }
    // The direct endpoint-density producer uses the FAB box; ordinary charge
    // deposition uses a grown particle tile. Cover both actual origins.
    for(int point=0;point<2;++point)for(int producer=0;producer<2;++producer){Index first{};Point const q=Coordinates(point?endpoint:xn,producer?rho_tile_origin:rho_origin,inverse);
     if(!Start(q,first)){amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;}
     int issue=Clear(empty,first,Upper(first),producer?rho_tile_offset:rho_offset);if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:3),amrex::Long(1));return;}
    }
    // Folded physical endpoint support is additional; all unfolded interval,
    // virtual density and point-current checks above/below are still required.
#if defined(WARPX_DIM_RZ)
    if(trial&&image.reflected){
     Point const folded{image.stored[0]*std::cos(image.stored[1]),image.stored[0]*std::sin(image.stored[1]),image.stored[2]};
     for(int producer=0;producer<2;++producer){Index first{};
      if(!Start(Coordinates(folded,producer?rho_tile_origin:rho_origin,inverse),first)){
       amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;}
      int issue=Clear(empty,first,Upper(first),producer?rho_tile_offset:rho_offset);
      if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:3),amrex::Long(1));return;}
     }
     Index first{};if(!Start(Coordinates(folded,origin,inverse),first)){
      amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;}
     int issue=CurrentClear(masks,first,Upper(first),offset);
     if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:4),amrex::Long(1));return;}
    }
#endif
    // Native VirtualEndpoint performs Finish's SetPosition/GetPosition RZ
    // roundtrip. FinishedEndpoint already reads the actual stored endpoint.
    Point represented=endpoint;
#if defined(WARPX_DIM_RZ)
    if(trial){P const radius=std::sqrt(represented[0]*represented[0]+represented[1]*represented[1]),angle=std::atan2(represented[1],represented[0]);represented[0]=radius*std::cos(angle);represented[1]=radius*std::sin(angle);}
#endif
    for(int point=0;point<2;++point){Index first{};if(!Start(Coordinates(point?represented:xn,origin,inverse),first)){amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;}
     int issue=CurrentClear(masks,first,Upper(first),offset);if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:4),amrex::Long(1));return;}
    }
    if(trial){warpx::particles::EsirkepovInstantaneousPointIncrement point{};
#if defined(WARPX_DIM_RZ)
     warpx::particles::RZPointIncrement rz{};if(!warpx::particles::CylindricalPointIncrement({xn[0],xn[1]},{dx[0],dx[1]},rz)){amrex::Gpu::Atomic::AddNoRet(counts+1,amrex::Long(1));return;}
     point.position={warpx::particles::EsirkepovIncrement{(rz.radius-origin.x)*inverse.x,rz.radial_delta*inverse.x},{0.,0.},{(xn[2]-origin.z)*inverse.z,dx[2]*inverse.z}};
#else
     double const o[3]{origin.x,origin.y,origin.z},iv[3]{inverse.x,inverse.y,inverse.z};for(int d=0;d<3;++d)point.position[d]={(xn[d]-o[d])*iv[d],dx[d]*iv[d]};
#endif
     amrex::GpuArray<warpx::particles::EsirkepovIncrement,3> unused{};
     if(adjacent){
      // This additional image uses the captured chord. The interval/density/
      // finisher images above deliberately keep their represented endpoints.
      warpx::particles::CubicAdjacentPointSupport support{};
      if(!warpx::particles::PrepareCubicAdjacentPointSupport<Dim>(point,support)||
         !warpx::particles::EsirkepovAdjacentInstantaneousCurrentPair<Dim>(point,{0,0,0},unused)){
       amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;
      }
      int issue=CurrentClear(masks,support.lower,support.upper,offset);if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:5),amrex::Long(1));return;}
     }else{
      bool pair_ok=warpx::particles::EsirkepovInstantaneousCurrentPair<3,Dim>(point,{0,0,0},unused);
      bool use_axial_pair=false;warpx::particles::CubicAxialPointSupport union_support;
 #if defined(WARPX_DIM_RZ)
      if(!pair_ok&&axial_scope){
       warpx::implicit::AxialEndpointImage retained_image;
       bool const path=warpx::implicit::endpoint_detail::axialCurrentFits(
           {xn[0]+dx[0],xn[1]+dx[1],xn[2]+dx[2]},
           {momentum[0][p]+momentum[6][p],momentum[1][p]+momentum[7][p],momentum[2][p]+momentum[8][p]},
           {gather[0][p],gather[1][p],gather[2][p]},audit,axial,retained_image,false);
       bool const one=path&&image.valid&&retained_image.reflected==image.reflected&&
           warpx::particles::OneAxialKnotSupport(point,union_support);
       long long const knot=static_cast<long long>(union_support.knot)+offset.y;
       bool const admitted=one&&(image.reflected?
           (retained_image.side==image.side&&knot==(image.side==0?cap_low:cap_high)):
           (knot>cap_low&&knot<cap_high));
       use_axial_pair=admitted&&warpx::particles::EsirkepovAxialInstantaneousCurrentPair(
           point,union_support.knot,{0,0,0},unused);
       pair_ok=use_axial_pair;
      }
 #else
      amrex::ignore_unused(cap_low,cap_high);
 #endif
      if(!pair_ok){amrex::Gpu::Atomic::AddNoRet(counts+8,amrex::Long(1));return;}
      Index first{int(point.position[0].value)-1,Dim==3?int(point.position[1].value)-1:0,
          use_axial_pair?union_support.axial_first:int(point.position[2].value)-1};
      auto high=Upper(first);if(use_axial_pair)high[2]=union_support.axial_last;
      int issue=CurrentClear(masks,first,high,offset);if(issue){amrex::Gpu::Atomic::AddNoRet(counts+(issue==8?7:5),amrex::Long(1));return;}
     }
    }
   });
  }
 }
 amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_counts.begin(),m_counts.end(),report.counts.begin());
 amrex::ParallelDescriptor::ReduceLongSum(report.counts.data(),9);report.valid=true;for(int i=1;i<9;++i)report.valid=report.valid&&report.counts[i]==0;return report;
}
}
