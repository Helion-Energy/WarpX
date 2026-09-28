/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "DarwinVacuumJointSolve.H"
#include "NativeJointCurrentArithmetic.H"
#include "ImplicitAxialEndpoint.H"
#include "ThermalCurrentRemainder.H"
#include "ThermalMassMatrixResponse.H"
#include "DarwinThermalAdvance.H"
#include "ThetaImplicitHybrid.H"
#include "NativeVacuumConstraint.H"
#include "NativeVacuumEndpoint.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "NativeEndpointField.H"
#include "NativeEndpointCurrentResponse.H"
#include "NativeVacuumMassMatrixCapabilities.H"
#include "NativeRetainedAcceptance.H"
#include "NativeLongitudinalIncrement.H"
#include "EulerianThermalStageUtils.H"
#include "ThermalRandomCheckpoint.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Gather/ImplicitGatherSafety.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <ablastr/coarsen/sample.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_Random.H>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace warpx::thermal {
namespace {
using MF=amrex::MultiFab;using Real=amrex::Real;using View=ablastr::fields::VectorField;
using Field=amrex::Array<MF,3>;using FT=warpx::fields::FieldType;using Dir=ablastr::fields::Direction;
void Copy(MF& dst,MF const& src) { MF::Copy(dst,src,0,0,1,amrex::min(dst.nGrowVect(),src.nGrowVect())); }
void Define(MF& dst,MF const& src,amrex::IntVect ng) {dst.define(src.boxArray(),src.DistributionMap(),1,ng);dst.setVal(0.);}
void Define(Field& dst,View const& src,bool guards=true) {for(int c=0;c<3;++c)Define(dst[c],*src[c],guards?src[c]->nGrowVect():amrex::IntVect(0));}
void Copy(Field& dst,View const& src) {for(int c=0;c<3;++c)Copy(dst[c],*src[c]);}
int Axis(int c) {
#if defined(WARPX_DIM_RZ)
 return c==0?0:c==2?1:-1;
#else
 return c;
#endif
}
bool Equal(MF const& x,MF const& y) {
 bool layout=x.isDefined()&&y.isDefined();
 if(layout)layout=x.boxArray()==y.boxArray()&&x.DistributionMap()==y.DistributionMap()&&x.nGrowVect()==y.nGrowVect()&&x.nComp()==y.nComp();
 amrex::ParallelDescriptor::ReduceBoolAnd(layout);if(!layout)return false;
 amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
 for(amrex::MFIter it(x);it.isValid();++it){auto p=x.const_array(it);auto q=y.const_array(it);op.eval(it.fabbox(),x.nComp(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k,int n)->T{return {p(i,j,k,n)!=q(i,j,k,n)};});}
 int bad=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceIntMax(bad);return bad==0;
}
// Read-only accepted-support decision. The no-V route must delegate before
// a predictor, longitudinal materialization or thermal-seed repair occurs.
bool AcceptedVacuumPresence(WarpX& w,MF const& rho,bool& vacuum) {
 auto const e=w.m_fields.get_alldirs(FT::Efield_fp,0);bool layout=rho.isDefined()&&rho.nComp()==1;
 for(auto const* f:e)layout=layout&&f&&f->nComp()==1&&rho.DistributionMap()==f->DistributionMap()&&rho.boxArray()==amrex::convert(f->boxArray(),amrex::IntVect::TheNodeVector());
 amrex::ParallelDescriptor::ReduceBoolAnd(layout);if(!layout)return false;
 auto const& g=w.Geom(0);auto lo=g.Domain().smallEnd();auto hi=g.Domain().bigEnd()+amrex::IntVect(1);
 amrex::GpuArray<int,AMREX_SPACEDIM> dl{},dh{};
 for(int d=0;d<AMREX_SPACEDIM;++d){dl[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC;dh[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC;}
 int bad=0,found=0;
 for(int c=0;c<3;++c){auto type=e[c]->ixType().toIntVect();
  amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax> op;amrex::ReduceData<int,int> data(op);using T=decltype(data)::Type;
  for(amrex::MFIter it(*e[c]);it.isValid();++it){auto r=rho.const_array(it);
   op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T {
    amrex::IntVect q(AMREX_D_DECL(i,j,k));bool fixed=false;for(int d=0;d<AMREX_SPACEDIM;++d)fixed=fixed||(type[d]&&((dl[d]&&q[d]==lo[d])||(dh[d]&&q[d]==hi[d])));
#if defined(WARPX_DIM_RZ)
    fixed=fixed||(c==1&&i==0);
#endif
    amrex::GpuArray<int,3> const node{1,1,1},ratio{1,1,1};amrex::GpuArray<int,3> stagger{type[0],type[1],1};
#if defined(WARPX_DIM_3D)
    stagger[2]=type[2];
#endif
    Real value=ablastr::coarsen::sample::Interp(r,node,stagger,ratio,i,j,k,0);
    return {!std::isfinite(value)||value<0.,!fixed&&value==0.};
   });
  }auto v=data.value();bad=std::max(bad,amrex::get<0>(v));found=std::max(found,amrex::get<1>(v));
 }
 amrex::ParallelDescriptor::ReduceIntMax(bad);amrex::ParallelDescriptor::ReduceIntMax(found);vacuum=found!=0;return bad==0;
}
struct Scope {
 // The owning class supplies the typed values; restoring is exception-safe.
 std::function<void()> restore;
 ~Scope(){restore();}
};
}
DarwinVacuumJointSolve::DarwinVacuumJointSolve(DarwinThermalAdvance& owner)
 :a(owner),s(owner.m_solver),w(owner.m_simulation) {}
DarwinVacuumJointSolve::~DarwinVacuumJointSolve()=default;
void DarwinVacuumJointSolve::InvalidateConvergenceReceipt() noexcept {
 m_endpoint_ampere.reset();m_endpoint_ampere_generation=0;
 InvalidateAxialReflection();
 m_convergence_input=nullptr;m_convergence_generation=0;m_convergence_part=-1;
}
void DarwinVacuumJointSolve::Invalidate() noexcept {InvalidateConvergenceReceipt();InvalidateMassMatrixLease();m_mm_full_base_current=false;m_verified=false;m_active=false;m_particle_phase=ParticlePhase::Inactive;if(m_arithmetic)m_arithmetic->Invalidate();if(m_hodge)m_hodge->Invalidate();if(m_particle_support)m_particle_support->Invalidate();}

void DarwinVacuumJointSolve::InvalidateMassMatrixLease() noexcept
{
    m_mm_linear_ready=false;
    if(m_mm_thermal)m_mm_thermal->Invalidate();
    m_mm_thermal_density_epoch=0;m_mm_thermal_publication=0;
    m_mm_thermal_copies.clear();m_remainder_generation=0;
    m_mm_response=nullptr;
    m_mm_density=nullptr;
    // Epochs/generations are not rolled back. Surviving allocations never
    // grant permission to reuse a response after a full map or Cancel.
}


void DarwinVacuumJointSolve::InvalidateAxialReflection() noexcept {
 m_axial_prepared=false;m_axial_finished=false;m_axial_generation=0;m_axial_count=0;
 m_axial_particles.clear();m_axial_changed.clear();
}
int DarwinVacuumJointSolve::AxialProducerState(WarpX& simulation,
    DarwinVacuumJointSolve const*& current)
{
 current=nullptr;
 auto const* solver=dynamic_cast<ThetaImplicitHybrid*>(simulation.get_pointer_ImplicitSolver());
 if(!solver || !solver->m_native_pmc_joint_requested ||
    solver->m_joint_stage_purpose==ThetaImplicitHybrid::JointStagePurpose::Standard)return 1;
 auto const* advance=solver->m_eulerian_energy.get();
 auto const* owner=advance?advance->m_joint_vacuum.get():nullptr;
 bool const ready=owner && owner==solver->m_joint_stage_owner &&
     owner->AllowsAxialReflection() &&
     owner->m_generation>0 && owner->m_generation<std::numeric_limits<std::uint64_t>::max() &&
     owner->m_time==advance->m_time && owner->m_dt==advance->m_dt &&
     owner->m_physical_step==simulation.getistep(0) && simulation.gett_new(0)==owner->m_time &&
     ((solver->m_joint_stage_purpose==ThetaImplicitHybrid::JointStagePurpose::Residual &&
       owner->m_particle_phase==ParticlePhase::Trial) ||
      (solver->m_joint_stage_purpose==ThetaImplicitHybrid::JointStagePurpose::PublishedCurrent &&
       owner->m_particle_phase==ParticlePhase::Published &&
       owner->m_generation==owner->m_verified_generation &&
       (!advance->m_reflected_particles || owner->AxialReflectionPrepared())));
 if(!ready)return 0;
 current=owner;return 2;
}
DarwinVacuumJointSolve::AxialProducerLease
DarwinVacuumJointSolve::CaptureAxialProducerLease(WarpX& simulation)
{
 AxialProducerLease lease;DarwinVacuumJointSolve const* owner=nullptr;
 int const local=AxialProducerState(simulation,owner);int low=local,high=local;
 amrex::ParallelDescriptor::ReduceIntMin(low);amrex::ParallelDescriptor::ReduceIntMax(high);
 lease.valid=low==high&&low>0;lease.enabled=lease.valid&&low==2;
 if(lease.enabled){lease.owner=owner;lease.generation=owner->m_generation;lease.lifetime=owner->m_axial_lifetime;}
 return lease;
}
bool DarwinVacuumJointSolve::AxialProducerLeaseCurrentLocal(WarpX& simulation,
    AxialProducerLease const& lease)
{
 DarwinVacuumJointSolve const* owner=nullptr;int const state=AxialProducerState(simulation,owner);
 if(!lease.valid)return false;
 if(!lease.enabled)return state==1;
 return state==2&&owner==lease.owner&&lease.lifetime==owner->m_axial_lifetime&&
     lease.generation==owner->m_generation;
}
bool DarwinVacuumJointSolve::AllowsAxialReflection() const noexcept {
 return m_axial_scope&&m_active&&a.m_open&&a.m_joint_vacuum.get()==this&&
     s.m_native_pmc_joint_requested&&!s.ReflectsParticlesInsidePush();
}
bool DarwinVacuumJointSolve::AxialReflectionPrepared() const noexcept {
 return AllowsAxialReflection()&&m_axial_prepared&&m_axial_count>0&&
     m_axial_generation==m_generation&&m_generation==m_verified_generation&&
     (m_particle_phase==ParticlePhase::Trial||m_particle_phase==ParticlePhase::Published)&&
     m_time==a.m_time&&m_dt==a.m_dt&&m_physical_step==w.getistep(0)&&w.gett_new(0)==m_time;
}
bool DarwinVacuumJointSolve::AxialReflectionFinished() const noexcept {
 return AllowsAxialReflection()&&m_axial_prepared&&m_axial_finished&&m_axial_count>0&&
     m_axial_generation==m_generation&&m_generation==m_verified_generation&&
     m_particle_phase==ParticlePhase::Finished&&m_time==a.m_time&&m_dt==a.m_dt&&
     m_physical_step==w.getistep(0)&&w.gett_new(0)==m_time;
}
bool DarwinVacuumJointSolve::PrepareAxialReflection(amrex::Long reflected) {
 InvalidateAxialReflection();
 bool const config=s.NativePMCJointConfigurationValid(true);
 bool ready=config&&AllowsAxialReflection()&&m_verified&&m_particle_phase==ParticlePhase::Trial&&
     m_generation==m_verified_generation&&reflected>0&&!a.m_absorbed_particles&&!a.m_ion_exchange;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!CheckReceipt())return false;
#ifdef WARPX_DIM_RZ
 // Clone the existing device-resident complete receipt. All fields, IDs,
 // dynamic attributes and RNG remain byte protected; only exact expected
 // finisher/image coordinates and momentum are replaced in this copy.
 for(auto const& old:m_particles){Bytes q;q.pointer=old.pointer;q.size=old.size;q.saved.resize(q.size);
  if(q.size)amrex::Gpu::dtod_memcpy(q.saved.data(),old.saved.data(),q.size);
  m_axial_particles.push_back(std::move(q));}
 using P=amrex::ParticleReal;using Point=amrex::GpuArray<P,3>;
 auto saved=[&](void const* pointer,std::size_t size)->P*{
  for(auto& q:m_axial_particles)if(q.pointer==pointer&&q.size>=size){
   m_axial_changed.push_back(pointer);return reinterpret_cast<P*>(q.saved.data());}
  return nullptr;};
 amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
 amrex::Gpu::DeviceScalar<amrex::Long> count(0);auto* reflected_count=count.dataPtr();
 auto const& geom=w.Geom(0);auto const lo=geom.ProbLoArray(),inv=geom.InvCellSizeArray();
 auto const domain_lo=geom.Domain().smallEnd();
 for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
  auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
  auto const boundary=warpx::implicit::MakeAxialEndpointBoundary(geom,pc.GetParticleBoundaryData(),
      WarpX::field_boundary_lo,WarpX::field_boundary_hi,true);
  ready=ready&&boundary.enabled;
  for(WarpXParIter pti(pc,0);pti.isValid();++pti){
   auto const np=pti.numParticles();if(!np)continue;
   ready=ready&&pti.numNeighborParticles()==0;
   auto const get=GetParticlePosition<PIdx>(pti);
   auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
   amrex::GpuArray<P const*,6> old{pti.GetAttribs("x_n").dataPtr(),pti.GetAttribs("y_n").dataPtr(),pti.GetAttribs("z_n").dataPtr(),
       pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),pti.GetAttribs("uz_n").dataPtr()};
   amrex::GpuArray<P*,6> expected{saved(get.m_x,np*sizeof(P)),saved(get.m_theta,np*sizeof(P)),saved(get.m_z,np*sizeof(P)),
       saved(ux,np*sizeof(P)),saved(uy,np*sizeof(P)),saved(uz,np*sizeof(P))};
   bool pointers=true;for(auto* p:expected)pointers=pointers&&p;
   ready=ready&&pointers;if(!pointers)continue;
   auto const tile=pti.tilebox();
   auto const tilelo=tile.smallEnd(),tilehi=tile.bigEnd();
   amrex::ParallelFor(np,[=]AMREX_GPU_DEVICE(long p){
    P x,y,z;get(p,x,y,z);Point const end{2.*x-old[0][p],2.*y-old[1][p],2.*z-old[2][p]};
    Point const momentum{2.*ux[p]-old[3][p],2.*uy[p]-old[4][p],2.*uz[p]-old[5][p]};
    auto const image=warpx::implicit::MapAxialEndpoint(end,momentum,boundary);
    // Prevent ownership migration before the later mandatory Redistribute.
    // There is no clamp or rebin in the rejectable particle transaction.
    P const rcell=(image.stored[0]-lo[0])*inv[0]+domain_lo[0];
    P const zcell=(image.stored[2]-lo[1])*inv[1]+domain_lo[1];
    bool const fits=image.valid&&rcell>=tilelo[0]&&rcell<tilehi[0]+1&&zcell>=tilelo[1]&&zcell<tilehi[1]+1;
    if(!fits){amrex::Gpu::Atomic::Max(bad,1);return;}
    for(int d=0;d<3;++d){expected[d][p]=image.stored[d];expected[d+3][p]=image.momentum[d];}
    if(image.reflected)amrex::Gpu::Atomic::AddNoRet(reflected_count,amrex::Long(1));
   });
  }
 }
 amrex::Gpu::synchronize();int local_bad=invalid.dataValue();amrex::Long actual=count.dataValue();
 amrex::ParallelDescriptor::ReduceIntMax(local_bad);amrex::ParallelDescriptor::ReduceLongSum(actual);
 ready=ready&&!local_bad&&actual==reflected;amrex::ParallelDescriptor::ReduceBoolAnd(ready);
 if(!ready){InvalidateAxialReflection();return false;}
 m_axial_count=actual;m_axial_generation=m_generation;m_axial_prepared=true;
 amrex::Print()<<"JOINT_VACUUM axial_reflection_prepared="<<actual<<" generation="<<m_generation<<" native_unfolded_deposition=1\n";
 return true;
#else
 amrex::ignore_unused(reflected);return false;
#endif
}
bool DarwinVacuumJointSolve::CompleteAxialReflection() {
 bool const config=s.NativePMCJointConfigurationValid(true);
 bool ready=config&&AllowsAxialReflection()&&m_axial_prepared&&!m_axial_finished&&
     m_axial_generation==m_generation&&m_generation==m_verified_generation&&
     m_particle_phase==ParticlePhase::Finishing&&m_axial_count==a.m_reflected_particles;
 // Structural/RNG preflight remains complete after the unchanged native
 // finisher. It precedes any private reflection write on every rank.
 std::vector<std::size_t> schema;std::size_t index=0;
 VisitReceipt([&](void const* pointer,std::size_t size){
  if(index>=m_axial_particles.size()){ready=false;return;}
  auto const& q=m_axial_particles[index++];ready=ready&&q.pointer==pointer&&q.size==size;
 },schema);
 std::ostringstream rng;amrex::SaveRandomState(rng);
 ready=ready&&schema==m_particle_schema&&index==m_axial_particles.size()&&
     m_next_id==WarpXParticleContainer::ParticleType::the_next_id&&rng.str()==m_host_rng;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
#ifdef WARPX_DIM_RZ
 using P=amrex::ParticleReal;
 auto saved=[&](void const* pointer)->P const*{
  for(auto const& q:m_axial_particles)if(q.pointer==pointer)return reinterpret_cast<P const*>(q.saved.data());
  return nullptr;};
 amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
 for(auto const& q:m_axial_particles){
  if(std::find(m_axial_changed.begin(),m_axial_changed.end(),q.pointer)!=m_axial_changed.end())continue;
  auto const* current=static_cast<unsigned char const*>(q.pointer);auto const* expected=q.saved.data();
  amrex::ParallelFor(q.size,[=]AMREX_GPU_DEVICE(amrex::Long j){if(current[j]!=expected[j])amrex::Gpu::Atomic::Max(bad,1);});
 }
 auto const& geom=w.Geom(0);
 // First pass computes the proposed native image without changing a byte.
 for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
  auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
  auto const boundary=warpx::implicit::MakeAxialEndpointBoundary(geom,pc.GetParticleBoundaryData(),
      WarpX::field_boundary_lo,WarpX::field_boundary_hi,true);
  ready=ready&&boundary.enabled;
  for(WarpXParIter pti(pc,0);pti.isValid();++pti){if(!pti.numParticles())continue;
   auto const get=GetParticlePosition<PIdx>(pti);
   auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();
   amrex::GpuArray<P const*,6> expected{saved(get.m_x),saved(get.m_theta),saved(get.m_z),saved(ux),saved(uy),saved(uz)};
   bool pointers=true;for(auto* p:expected)pointers=pointers&&p;ready=ready&&pointers;if(!pointers)continue;
   amrex::ParallelFor(pti.numParticles(),[=]AMREX_GPU_DEVICE(long p){
    P r,theta,z;get.AsStored(p,r,theta,z);
    auto const image=warpx::implicit::MapStoredAxialEndpoint({r,theta,z},{ux[p],uy[p],uz[p]},boundary);
    bool ok=image.valid;for(int d=0;d<3;++d)ok=ok&&image.stored[d]==expected[d][p]&&image.momentum[d]==expected[d+3][p];
    if(!ok)amrex::Gpu::Atomic::Max(bad,1);
   });
  }
 }
 amrex::Gpu::synchronize();int local_bad=invalid.dataValue();amrex::ParallelDescriptor::ReduceIntMax(local_bad);
 ready=ready&&!local_bad;amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 // The collective decision above authorizes exactly the native deterministic
 // axial operation. There is no general boundary/RNG/tally/scrape call here.
 for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
  auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);
  auto const boundary=warpx::implicit::MakeAxialEndpointBoundary(geom,pc.GetParticleBoundaryData(),
      WarpX::field_boundary_lo,WarpX::field_boundary_hi,true);
  for(WarpXParIter pti(pc,0);pti.isValid();++pti){
   auto const get=GetParticlePosition<PIdx>(pti);auto const set=SetParticlePosition<PIdx>(pti);
   auto* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto* uz=pti.GetAttribs(PIdx::uz).dataPtr();
   amrex::ParallelFor(pti.numParticles(),[=]AMREX_GPU_DEVICE(long p){
    P r,theta,z;get.AsStored(p,r,theta,z);
    auto const image=warpx::implicit::MapStoredAxialEndpoint({r,theta,z},{ux[p],uy[p],uz[p]},boundary);
    if(image.reflected){set.AsStored(p,image.stored[0],image.stored[1],image.stored[2]);uz[p]=image.momentum[2];}
   });
  }
 }
 for(auto const& q:m_axial_particles){auto const* current=static_cast<unsigned char const*>(q.pointer);auto const* expected=q.saved.data();
  amrex::ParallelFor(q.size,[=]AMREX_GPU_DEVICE(amrex::Long j){if(current[j]!=expected[j])amrex::Gpu::Atomic::Max(bad,1);});}
 amrex::Gpu::synchronize();local_bad=invalid.dataValue();amrex::ParallelDescriptor::ReduceIntMax(local_bad);
 if(local_bad)return false;
 m_axial_finished=true;
 amrex::Print()<<"JOINT_VACUUM axial_reflection_finished="<<m_axial_count<<" exact_expected_state=1 reversible=1\n";
 return true;
#else
 return false;
#endif
}

bool DarwinVacuumJointSolve::MassMatrixScopeSupported() const
{
#if defined(WARPX_DIM_RZ)
    auto const& model=a.m_model;
    auto const literal_zero=[](std::string const& expression) {
        return expression=="0" || expression=="0.0";
    };
    // The model flag denotes parser presence, including an owned literal
    // zero law. Preserve its executors; do not sample a dynamic law.
    bool const zero_conduction=!model.m_include_thermal_conduction ||
        (literal_zero(model.m_kappa_par_expression) &&
         literal_zero(model.m_kappa_perp_expression));
    bool adiabatic=true;
    for(int d=0;d<AMREX_SPACEDIM;++d)for(int side=0;side<2;++side)
        adiabatic=adiabatic && model.m_cond_bc[d][side]==0;
    // Evaluate the collective private capability before a short-circuit public
    // predicate. It binds this exact Theta owner; a finite companion requires
    // its separately captured private conduction capability.
    auto const* pmc_owner=NativePrivatePMCMassMatrixOwner(w);
    return s.m_use_mass_matrices && s.m_use_mass_matrices_jacobian &&
        s.m_use_mass_matrices_pc && s.m_esirkepov_mass_matrices &&
        s.m_mass_matrices_density_projection && s.m_live_ion_density &&
        s.m_continuity_density && s.m_num_amr_levels==1 &&
        s.m_mass_matrices_deposit_interval==1 && !s.m_mass_matrices_reuse_within_step &&
        !s.m_fused_mass_matrices_deposit && !s.m_particle_suborbits &&
        !s.m_skip_particle_picard_init && !s.m_native_paired_requested &&
        !s.m_native_paired_fields && !a.HasResidualRemainder() &&
        !s.m_circuit_native && !w.get_pointer_CircuitCoupling() &&
        !model.m_has_electron_stopping && !model.m_end_region.holmstrom &&
        ((zero_conduction && adiabatic && NativeVacuumMassMatrixSupported(w)) || pmc_owner==&s);
#else
    return false;
#endif
}

bool DarwinVacuumJointSolve::MassMatrixBaseMatches(Vec const& x) const
{
    bool ready=m_mm_full_base_current && m_mm_base.IsDefined() && x.IsDefined() &&
        x.hasMultiFabBlock(U) && x.hasMultiFabBlock(Phi) &&
        x.getArrayVec().size()==1 && m_mm_base.getArrayVec().size()==1 &&
        x.blockNames()==m_mm_base.blockNames() && x.blockScales()==m_mm_base.blockScales();
    if(ready)for(int c=0;c<3;++c)
        ready=ready && x.getArrayVec()[0][c] && m_mm_base.getArrayVec()[0][c];
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return false;
    // Equal performs collective layout preflight before reading data. Keep
    // every comparison collective even when an earlier component differs.
    for(int c=0;c<3;++c)
        ready=Equal(*x.getArrayVec()[0][c],*m_mm_base.getArrayVec()[0][c]) && ready;
    ready=Equal(x.getMultiFabBlock(U,0),m_mm_base.getMultiFabBlock(U,0)) && ready;
    ready=Equal(x.getMultiFabBlock(Phi,0),m_mm_base.getMultiFabBlock(Phi,0)) && ready;
    return ready;
}

bool DarwinVacuumJointSolve::MassMatrixCacheReady() const
{
    bool const captured=s.m_mass_matrix_density.size()==1 &&
        s.m_mass_matrix_density[0] && s.m_mass_matrix_density[0]->IsCaptured();
    return s.m_use_mass_matrices_jacobian && s.m_mass_matrices_density_projection &&
        s.m_live_ion_density && s.m_mass_matrices_cached && s.m_mass_matrices_current &&
        s.m_mass_matrices_cached_dt==m_dt && s.m_mass_matrices_cached_time==m_time &&
        s.m_mass_matrices_calls_since_deposit==1 && captured &&
        s.m_endpoint_current_response && s.m_endpoint_current_response->IsValid() &&
        s.m_endpoint_current_response->LocalBorisResponse();
}

bool DarwinVacuumJointSolve::MassMatrixLeaseReady() const
{
    auto const* pmc_owner=NativePrivatePMCMassMatrixOwner(w);
    bool ready=(!s.m_native_pmc_mm_requested || pmc_owner==&s) &&
        m_mm_requested && m_mm_linear_ready && m_active && m_support_ready &&
        m_particle_phase==ParticlePhase::Trial && a.m_open &&
        m_mm_base_generation!=0 && m_mm_lease_base_generation==m_mm_base_generation &&
        m_time==a.m_time && m_dt==a.m_dt && m_physical_step==w.getistep(0) &&
        w.gett_new(0)==m_time && s.m_joint_stage_owner==nullptr &&
        MassMatrixCacheReady();
    if(ready)ready=m_mm_response==s.m_endpoint_current_response.get() &&
        m_mm_endpoint_epoch==s.m_endpoint_current_response->Epoch() &&
        m_mm_density==s.m_mass_matrix_density[0].get() &&
        m_mm_interval_width==s.m_ncomp_xx[0];
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    if(!ready)return false;
    // A MM map holds the base particle geometry. This is not a fresh
    // trajectory certificate and never permits final particle publication.
    return CheckReceipt();
}

bool DarwinVacuumJointSolve::MassMatrixThermalCompanionReady() const
{
    // Cached helper/copy receipts do not survive a native invalidation, even
    // at the same clock and without another joint map. Revalidate the actual
    // response/projection owner and particle receipt before consuming bytes.
    if(!MassMatrixLeaseReady())return false;
    bool ready=remainder::ArithmeticSupported() && m_mm_thermal_requested &&
        m_mm_thermal && m_mm_thermal->Ready() && m_remainder_generation!=0 &&
        m_remainder_generation==m_generation && m_mm_thermal_density_epoch!=0 &&
        m_mm_thermal_publication!=0;
    if(ready) {
        auto const& projection=*s.m_mass_matrix_density[0];
        ready=projection.BaseEpoch()==m_mm_thermal_density_epoch &&
            projection.PublicationEpoch()==m_mm_thermal_publication &&
            projection.CurrentIncrementReady() && projection.DensityIncrementReady();
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    return ready;
}

void DarwinVacuumJointSolve::CanonicalScalar(MF& phi) {
 auto const& g=w.Geom(0);auto lo=g.Domain().smallEnd();auto hi=g.Domain().bigEnd()+amrex::IntVect(1);
 amrex::GpuArray<int,AMREX_SPACEDIM> dl{},dh{};
 for(int d=0;d<AMREX_SPACEDIM;++d){dl[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC;dh[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC;}
 for(amrex::MFIter it(phi);it.isValid();++it){auto p=phi.array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));for(int d=0;d<AMREX_SPACEDIM;++d)if((dl[d]&&q[d]==lo[d])||(dh[d]&&q[d]==hi[d]))p(q)=0.;});}
 phi.setBndry(0.);phi.OverrideSync(g.periodicity());phi.FillBoundary(g.periodicity());
}
void DarwinVacuumJointSolve::Gradient(View const& output,MF const& input) {
 MF::Copy(m_scalar_work,input,0,0,1,0);CanonicalScalar(m_scalar_work);auto dx=w.Geom(0).CellSizeArray();
 for(int c=0;c<3;++c){int d=Axis(c);if(d<0){output[c]->setVal(0.);continue;}auto off=amrex::IntVect::TheDimensionVector(d);Real inv=1./dx[d];
  for(amrex::MFIter it(*output[c]);it.isValid();++it){auto e=output[c]->array(it);auto p=m_scalar_work.const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){amrex::IntVect q(AMREX_D_DECL(i,j,k));e(q)=(p(q+off)-p(q))*inv;});}
  warpx::darwin::increment::Images(w,*output[c]);
 }
}
void DarwinVacuumJointSolve::Materialize(Vec const& x) {
 InvalidateConvergenceReceipt();
 using namespace warpx::darwin::increment;
 MF::Copy(m_phi,x.getMultiFabBlock(Phi,0),0,0,1,0);CanonicalScalar(m_phi);Gradient(V(m_delta),m_phi);
 // Always reconstruct from this attempt's origin, not the previous probe.
 auto el=w.m_fields.get_alldirs("hybrid_E_long_fp",0);
 for(int c=0;c<3;++c){Copy(*el[c],m_origin_EL[c]);w.m_fields.get(field_low,Dir{c},0)->setVal(0.);}
 Copy(*w.m_fields.get(Phi,0),m_origin_phi);w.m_fields.get(scalar_low,0)->setVal(0.);
 Correct(w,m_origin_phi,V(m_origin_EL),m_phi,V(m_delta),1.);
}
void DarwinVacuumJointSolve::CaptureTransverse() {
 AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this);
 Copy(m_transverse,w.m_fields.get_alldirs(FT::Efield_fp,0));
}
bool DarwinVacuumJointSolve::BeginInputSolve(Vec const& x) {return !m_arithmetic||m_arithmetic->BeginSolve(x);}
bool DarwinVacuumJointSolve::FormInput(Vec& out,Vec const& base,Real factor,Vec const& direction,bool probe) {return !m_arithmetic||m_arithmetic->FormInput(out,base,factor,direction,probe,m_generation+1);}
bool DarwinVacuumJointSolve::AcceptInput(Vec const& x) {return !m_arithmetic||m_arithmetic->AcceptInput(x,m_generation);}
void DarwinVacuumJointSolve::EndInputSolve(bool success) noexcept {if(m_arithmetic)m_arithmetic->EndSolve(success);}
void DarwinVacuumJointSolve::BuildArithmeticFields(Real h,amrex::IntVect const& grow) {AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this&&m_arithmetic);m_arithmetic->BuildFields(h,grow,m_generation);}
void DarwinVacuumJointSolve::PublishArithmeticCurrent() {AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this&&m_arithmetic);m_arithmetic->PublishCurrent(m_generation);}
void DarwinVacuumJointSolve::SubtractArithmeticDisplacement(Real h,View const* low) {AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this&&m_arithmetic);m_arithmetic->SubtractDisplacement(h,low,m_generation);}
void DarwinVacuumJointSolve::AssembleArithmeticElectric() {AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this&&m_arithmetic);m_arithmetic->AssembleElectric(w.m_fields.get_alldirs(FT::Efield_fp,0),m_generation);}
void DarwinVacuumJointSolve::CaptureArithmeticOhm() {AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this&&m_arithmetic);m_arithmetic->CaptureOhm(w.m_fields.get_alldirs(FT::Efield_fp,0),m_generation);}
void DarwinVacuumJointSolve::CapturePushField() {
 AMREX_ALWAYS_ASSERT(m_active&&s.m_joint_stage_owner==this);
 Copy(m_push,w.m_fields.get_alldirs(FT::Efield_fp,0));
}

namespace {
bool ZeroDriveReference(WarpX& w,amrex::Vector<amrex::Real> const& reference) {
 auto const& model=*w.get_pointer_HybridPICModel();
 int const count=model.m_add_external_fields && model.m_external_vector_potential
     ? model.m_external_vector_potential->nFields() : 0;
 bool valid=reference.empty() || (count>0 && reference.size()==static_cast<std::size_t>(count));
 for(auto value:reference)valid=std::isfinite(value)&&value==0.&&valid;
 amrex::ParallelDescriptor::ReduceBoolAnd(valid);return valid;
}
}
bool DarwinVacuumJointSolve::Prepare() {
 m_time=a.m_time;m_dt=a.m_dt;m_physical_step=w.getistep(0);m_h=s.m_theta*m_dt;
 m_alpha=m_h/PhysConst::epsilon_0;m_b=m_h*m_h/(PhysConst::epsilon_0*PhysConst::mu0);
 auto const& g=w.Geom(0);m_length=0.;for(int d=0;d<AMREX_SPACEDIM;++d)m_length=std::max(m_length,g.ProbLength(d));
 auto& model=a.m_model;std::string restart;amrex::ParmParse("amr").query("restart",restart);
 bool const zero_drive=NativeZeroExternalDriveSupported(w);
 bool const zero_reference=ZeroDriveReference(w,s.m_fext_init);
 int mm_min=s.m_use_mass_matrices_jacobian?1:0,mm_max=mm_min;
 amrex::ParallelDescriptor::ReduceIntMin(mm_min);amrex::ParallelDescriptor::ReduceIntMax(mm_max);
 if(mm_min!=mm_max){amrex::Print()<<"JOINT_VACUUM rejected inconsistent MM selector before native map\n";return false;}
 m_mm_requested=mm_max!=0;InvalidateMassMatrixLease();m_mm_full_base_current=false;
 bool thermal_increment=false;amrex::ParmParse("implicit_evolve").query("darwin_joint_mm_thermal_increment",thermal_increment);
 int thermal_min=thermal_increment?1:0,thermal_max=thermal_min;
 amrex::ParallelDescriptor::ReduceIntMin(thermal_min);amrex::ParallelDescriptor::ReduceIntMax(thermal_max);
 if(thermal_min!=thermal_max){amrex::Print()<<"JOINT_VACUUM rejected inconsistent MM thermal selector before native map\n";return false;}
 m_mm_thermal_requested=thermal_max!=0;
 bool thermal_scope=!m_mm_thermal_requested || (m_mm_requested && MassMatrixScopeSupported() &&
     remainder::ArithmeticSupported() && a.m_stage_options.transport.fluid_reconstruction==0 &&
     a.m_stage_options.transport.central_dissipation_entropy==0);
 thermal_scope=thermal_scope && (!s.m_native_pmc_finite_conduction_requested ||
        (s.m_native_pmc_thermal_requested && a.m_stage_options.use_compiled_conductivity &&
         (!m_mm_requested || m_mm_thermal_requested)));
 amrex::ParallelDescriptor::ReduceBoolAnd(thermal_scope);
 if(!thermal_scope){amrex::Print()<<"JOINT_VACUUM rejected MM thermal increment capability before native map\n";return false;}
 bool const mm_scope=!m_mm_requested || MassMatrixScopeSupported();
 bool scope=zero_drive&&zero_reference&&warpx::implicit::NativeRetainedAcceptanceEnabled()&&w.maxLevel()==0&&!EB::enabled()&&!w.getdo_moving_window()&&WarpX::grid_type==GridType::Staggered&&WarpX::nox==3&&WarpX::ncomps==1&&!WarpX::use_filter&&s.m_theta==.5&&s.m_continuity_density&&mm_scope&&!s.m_circuit_native&&!s.m_external_field_iteration&&!s.m_native_stopping_certificate&&(restart.empty()||s.m_native_endpoint_restart_validated)&&NativeVacuumEndpointEnabled()&&NativeFullOhmLongitudinalEnabled()&&warpx::darwin::increment::Enabled()&&!NativePrescribedDriveEnabled()&&!NativeCircuitDriveEnabled()&&!model.HasResistivity()&&!model.m_include_joule_heating&&!model.m_include_temperature_relaxation&&!model.m_include_electron_viscosity&&!model.m_include_hyper_resistivity_term&&!model.m_has_energy_sink&&!model.DensityPedestal(0);
 std::vector<std::string> collisions;amrex::ParmParse("collisions").queryarr("collision_names",collisions);scope=scope&&collisions.empty();
 amrex::ParallelDescriptor::ReduceBoolAnd(scope);if(!scope){amrex::Print()<<"JOINT_VACUUM rejected capability before native map\n";return false;}
 m_axial_scope=s.m_native_pmc_joint_requested && s.NativePMCJointScopeSupported() &&
     !s.ReflectsParticlesInsidePush();
 bool vacuum=false;if(!AcceptedVacuumPresence(w,a.m_old_charge,vacuum))return false;
 if(!vacuum){m_active=false;amrex::Print()<<"JOINT_VACUUM delegated no_V before_mutation=1 predictor_maps=0\n";return true;}
 auto e=w.m_fields.get_alldirs(FT::Efield_fp,0);auto el=w.m_fields.get_alldirs("hybrid_E_long_fp",0);auto mag=w.m_fields.get_alldirs(FT::Bfield_fp,0);
 auto const& phi=*w.m_fields.get(Phi,0);auto ng=phi.nGrowVect();for(auto* f:e)ng=amrex::max(ng,f->nGrowVect()+amrex::IntVect(1));
 for(auto* f:{&m_origin_phi,&m_phi,&m_held_phi,&m_held_phi_low,&m_phi_a,&m_phi_w,&m_scalar_work})Define(*f,phi,ng);
 Copy(m_origin_phi,phi);
 for(auto* f:{&m_origin_EL,&m_delta,&m_transverse,&m_push,&m_held_EL,&m_held_low,&m_saved_E,&m_current,&m_pc_current,&m_pc_d,&m_pc_h,&m_pc_e,&m_pc_k,&m_kappa,&m_zero,&m_verified_EL,&m_verified_Et,&m_verified_push})Define(*f,e);
 Define(m_verified_A,w.m_fields.get_alldirs("hybrid_A_fp",0));Define(m_verified_B,mag);Copy(m_origin_EL,el);
 if(m_mm_thermal_requested)m_mm_thermal=std::make_unique<ThermalMassMatrixResponse>(
     *a.m_stage_moments,s.m_native_pmc_finite_conduction_requested);
 auto const& u=a.m_energy.getMultiFabBlock(U,0);
 for(auto* f:{&m_raw_u_field,&m_correction})Define(*f,u,amrex::IntVect(0));
 for(auto* f:{&m_U_images,&m_frozen_U,&m_frozen_n})Define(*f,u,amrex::IntVect(3));
 for(int d=0;d<AMREX_SPACEDIM;++d)for(auto* f:{&m_face[d],&m_velocity[d],&m_advection[d]})Define(*f,*a.m_number_flux[d],amrex::IntVect(0));
 auto const& rho=*w.m_fields.get(FT::rho_fp,0);Define(m_support_rho,rho,rho.nGrowVect());
 m_state.Define(&w,"Efield_fp","none",{{U,a.m_energy_scale},{Phi,a.m_electric_scale*m_length}},a.m_electric_scale);
 m_pc_rhs.Define(a.m_combined);m_pc_out.Define(a.m_combined);
 m_residual.Define(m_state);m_raw.Define(m_state);m_last.Define(m_state);if(m_mm_requested)m_mm_base.Define(m_state);m_state.setVal(0.);a.CopyFields(m_state,s.m_E);MF::Copy(m_state.getMultiFabBlock(U,0),u,0,0,1,0);
 for(int c=0;c<3;++c){m_P[c].define(e[c]->boxArray(),e[c]->DistributionMap(),1,0);m_V[c].define(e[c]->boxArray(),e[c]->DistributionMap(),1,0);m_P[c].setVal(0);m_V[c].setVal(0);}
 auto geom=g;
#if defined(WARPX_DIM_RZ)
 geom=amrex::Geometry(g.Domain(),g.ProbDomain(),1,g.isPeriodic());
#endif
 auto so=warpx::darwin::NativeVacuumSupportOptions(w);m_support=std::make_unique<warpx::darwin::NativeInertiaSupport>(geom,w.boxArray(0),w.DistributionMap(0),so);
 // Automatic only inside the already admitted private PMC owner. The
 // source-derived arithmetic scope is agreed before allocation; unavailable
 // builds and other physical paths keep their original implementation.
 // Reuse the owned parser/field certificate above, including zero A reference.
 // The external-field enable flag alone does not describe a nonzero drive.
 bool arithmetic=m_axial_scope&&zero_drive&&zero_reference&&
     NativeJointCurrentArithmetic::SupportedLocal(w);
 amrex::ParallelDescriptor::ReduceBoolAnd(arithmetic);
 if(arithmetic)m_arithmetic=std::make_unique<NativeJointCurrentArithmetic>(w,m_state);
 m_active=true;m_particle_phase=ParticlePhase::Trial;
 // One accounted predictor from the saved old population; support is frozen
 // only after its endpoint-average density and actual native gathers exist.
 bool ready=false;
 for(int attempt=0;attempt<4;++attempt){
  Materialize(m_state);a.CopyFields(a.m_field,m_state);MF::Copy(a.m_energy.getMultiFabBlock(U,0),m_state.getMultiFabBlock(U,0),0,0,1,0);
  if(m_arithmetic&&!m_arithmetic->BindInput(m_state,false,m_generation+1))return false;
  ++m_generation;bool ok=false;
  {auto previous=s.m_joint_stage_purpose;auto* owner=s.m_joint_stage_owner;s.m_joint_stage_owner=this;s.m_joint_stage_purpose=ThetaImplicitHybrid::JointStagePurpose::Residual;
   Scope context{[&](){s.m_joint_stage_purpose=previous;s.m_joint_stage_owner=owner;}};
   ok=a.FieldResidual(a.m_field_rhs,a.m_field,0,false);++m_maps;
  }
  bool changed=false;if(!a.PrepareInitialThermalGuess(changed))return false;
  if(changed){MF::Copy(m_state.getMultiFabBlock(U,0),a.m_energy.getMultiFabBlock(U,0),0,0,1,0);continue;}
  if(!ok)return false;ready=true;break;
 }
 if(!ready||!CheckSupport(true))return false;
 amrex::Long count=0;for(int c=0;c<3;++c)count+=m_V[c].sum(0,0); // topology count only; owner count is reported by Hodge.
 if(count==0){amrex::Print()<<"JOINT_VACUUM support changed to empty V after predictor\n";return false;}
 m_particle_support=std::make_unique<warpx::darwin::NativeVacuumParticleSupport>();std::string support_error;
 if(!m_particle_support->Prepare(w,{&m_V[0],&m_V[1],&m_V[2]},m_support_rho,support_error,AllowsAxialReflection())){amrex::Print()<<"JOINT_VACUUM particle support rejected "<<support_error<<'\n';return false;}
 if(!CheckParticleSupport())return false;
 warpx::darwin::LongitudinalSchurOptions options;options.compatible_yee=true;options.output_ghosts=ng;options.max_semicoarsening_levels=model.m_darwin_poisson_semicoarsening;options.semicoarsening_direction=model.m_darwin_poisson_semicoarsening_direction;
 amrex::ParmParse pp("implicit_evolve");pp.query("darwin_schur_relative_tolerance",options.relative_tolerance);pp.query("darwin_schur_absolute_tolerance",options.absolute_tolerance);pp.query("darwin_schur_max_iterations",options.max_iterations);pp.query("darwin_schur_restart_length",options.restart_length);pp.query("darwin_schur_pc_cycles",options.preconditioner_cycles);
 for(int d=0;d<AMREX_SPACEDIM;++d){auto map=[](warpx::darwin::InitialRateBoundary q){using A=warpx::darwin::InitialRateBoundary;using B=warpx::darwin::LongitudinalBoundary;return q==A::Axis?B::Axis:q==A::PEC?B::PEC:q==A::PMC?B::PMC:B::Periodic;};options.lower[d]=map(so.lower[d]);options.upper[d]=map(so.upper[d]);}
 m_schur=std::make_unique<warpx::darwin::DarwinLongitudinalSchur>(g,w.boxArray(0),w.DistributionMap(0),options);
 warpx::darwin::DarwinVacuumHodgePC::Options ho;ho.b=m_b;ho.length=m_length;ho.relative_tolerance=options.relative_tolerance;ho.absolute_tolerance=options.absolute_tolerance;ho.max_iterations=options.max_iterations;ho.restart_length=options.restart_length;ho.use_auxiliary_preconditioner=true;pp.query("darwin_schur_green_auxiliary",ho.use_green_auxiliary_inverse);
 pp.query("darwin_schur_yee_green_interface",ho.use_yee_green_interface_inverse);
 pp.query("darwin_schur_interface_max_dofs",ho.max_interface_dofs);
 m_hodge=std::make_unique<warpx::darwin::DarwinVacuumHodgePC>(w,e,mag,ho);std::string error;
 if(!m_hodge->Prepare({&m_P[0],&m_P[1],&m_P[2]},{&m_V[0],&m_V[1],&m_V[2]},error)){amrex::Print()<<"JOINT_VACUUM topology rejected "<<error<<'\n';return false;}
 amrex::Print()<<"JOINT_VACUUM prepared V="<<m_hodge->Stats().vacuum_edges<<" W="<<m_hodge->Stats().potential_dofs<<" floating="<<m_hodge->Stats().floating_components<<" Hodge_bytes="<<m_hodge->Stats().local_owned_bytes<<" cap="<<ho.max_iterations<<" bottom_sweeps=8 green_auxiliary="<<ho.use_green_auxiliary_inverse<<" yee_green_interface="<<ho.use_yee_green_interface_inverse<<" interface_dofs="<<m_hodge->Stats().interface_dofs<<"\n";
 return true;
}

bool DarwinVacuumJointSolve::CheckSupport(bool prepare,bool finished) {
 auto const& g=w.Geom(0);auto const& charge=*w.m_fields.get(FT::rho_fp,0);int const mid=charge.nComp()/2;
 for(amrex::MFIter it(m_support_rho);it.isValid();++it){auto dst=m_support_rho.array(it);auto now=charge.const_array(it);auto old=a.m_old_charge.const_array(it);amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){Real stage=now(i,j,k,mid),end=finished?now(i,j,k,0):2.*stage-old(i,j,k);dst(i,j,k)=amrex::max(old(i,j,k),stage,end);});}
 if(prepare && !m_support->FreezeEdges(m_support_rho,*w.m_fields.get("hybrid_rho_vacmask_fp",0)))return false;
 auto lo=g.Domain().smallEnd();auto hi=g.Domain().bigEnd()+amrex::IntVect(1);amrex::GpuArray<int,AMREX_SPACEDIM> dl{},dh{};
 for(int d=0;d<AMREX_SPACEDIM;++d){dl[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC;dh[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC;}
 int bad=0;amrex::Long charged_recovery=0,empty_recovery=0;
 for(int c=0;c<3;++c){auto type=m_P[c].ixType().toIntVect();auto const& ji=*w.m_fields.get(FT::current_fp,Dir{c},0);
  amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpSum,amrex::ReduceOpSum> op;amrex::ReduceData<int,amrex::Long,amrex::Long> data(op);using T=decltype(data)::Type;
  for(amrex::MFIter it(m_P[c]);it.isValid();++it){auto p=m_P[c].array(it);auto v=m_V[c].array(it);auto raw=m_support_rho.const_array(it);auto rec=m_support->RecoveryMask(c).const_array(it);auto jion=ji.const_array(it);
   op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T {
    amrex::IntVect q(AMREX_D_DECL(i,j,k));bool fixed=false;for(int d=0;d<AMREX_SPACEDIM;++d)fixed=fixed||(type[d]&&((dl[d]&&q[d]==lo[d])||(dh[d]&&q[d]==hi[d])));
#if defined(WARPX_DIM_RZ)
    fixed=fixed||(c==1&&i==0);
#endif
    amrex::GpuArray<int,3> const node{1,1,1},ratio{1,1,1};
    amrex::GpuArray<int,3> stagger{type[0],type[1],1};
#if defined(WARPX_DIM_3D)
    stagger[2]=type[2];
#endif
    Real raw_edge=ablastr::coarsen::sample::Interp(raw,node,stagger,ratio,i,j,k,0);
    int P=!fixed&&raw_edge>0.,V=!fixed&&raw_edge==0.;int issue=!std::isfinite(raw_edge)||raw_edge<0.;
    if(prepare){p(q)=P;v(q)=V;}else issue=issue||(p(q)!=P||v(q)!=V);
    // Actual interval deposition, including boundary sums, is the support
    // oracle. Current magnitude never defines or expands the charged mask.
    issue=issue||(V&&jion(q)!=0.)||!std::isfinite(jion(q));
    return {issue,amrex::Long(P&&rec(q)),amrex::Long(V&&rec(q))};
   });
  }
  auto values=data.value();bad=std::max(bad,amrex::get<0>(values));charged_recovery+=amrex::get<1>(values);empty_recovery+=amrex::get<2>(values);
 }
 amrex::ParallelDescriptor::ReduceIntMax(bad);amrex::ParallelDescriptor::ReduceLongSum(charged_recovery);amrex::ParallelDescriptor::ReduceLongSum(empty_recovery);
 if(prepare||bad)amrex::Print()<<"JOINT_VACUUM support prepare="<<prepare<<" bad="<<bad<<" charged_recovery_local_rows_sum="<<charged_recovery<<" empty_recovery_local_rows_sum="<<empty_recovery<<'\n';
 m_support_ready=bad==0;return bad==0;
}

bool DarwinVacuumJointSolve::CheckParticleSupport() {
 bool ready=m_active&&m_particle_phase==ParticlePhase::Trial&&m_particle_support&&m_particle_support->Ready();
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 auto result=m_particle_support->Check(w,warpx::darwin::NativeVacuumParticleSupport::Phase::StageTrial);
 if(!result.valid){amrex::Print()<<"JOINT_VACUUM particle support failed phase=trial counts=";for(auto n:result.counts)amrex::Print()<<n<<',';amrex::Print()<<'\n';}
 return result.valid;
}

bool DarwinVacuumJointSolve::CapturePublishedEndpointStage() {
 // This exact published midpoint still has unfolded particles. Its already
 // verified receipt and typed borrower are required by the correlated pair.
 bool const config=s.NativePMCJointConfigurationValid(true);
 bool ready=config&&AllowsAxialReflection()&&a.m_receipt_ready&&
     m_particle_phase==ParticlePhase::Published&&m_generation==m_verified_generation&&
     m_generation>0&&m_generation<std::numeric_limits<std::uint64_t>::max()&&
     m_time==a.m_time&&m_dt==a.m_dt&&m_physical_step==w.getistep(0)&&w.gett_new(0)==m_time&&
     s.m_joint_stage_owner==nullptr&&s.m_joint_stage_purpose==ThetaImplicitHybrid::JointStagePurpose::Standard&&
     (!a.m_reflected_particles||AxialReflectionPrepared());
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!CheckReceipt())return false;
 auto const purpose=s.m_joint_stage_purpose;auto* owner=s.m_joint_stage_owner;
 s.m_joint_stage_owner=this;s.m_joint_stage_purpose=ThetaImplicitHybrid::JointStagePurpose::PublishedCurrent;
 {
  Scope restore{[&](){s.m_joint_stage_purpose=purpose;s.m_joint_stage_owner=owner;}};
  ready=TryCaptureNativeEndpointStage(w,m_time+s.m_theta*m_dt,m_dt,s.NativeCoilCurrentContext());
 }
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 ready=CheckReceipt();amrex::ParallelDescriptor::ReduceBoolAnd(ready);
 amrex::Print()<<"JOINT_VACUUM published_current_capture exact="<<ready<<" generation="<<m_generation<<"\n";
 return ready;
}

bool DarwinVacuumJointSolve::FinishParticles(Real endpoint_time) {
 bool const zero_drive=NativeZeroExternalDriveSupported(w);
 bool const zero_reference=ZeroDriveReference(w,s.m_fext_init);
 bool ready=zero_drive&&zero_reference&&m_active&&m_particle_phase==ParticlePhase::Published&&m_generation==m_verified_generation&&m_time==a.m_time&&m_dt==a.m_dt&&m_physical_step==w.getistep(0)&&w.gett_new(0)==m_time&&endpoint_time==m_time+m_dt&&s.m_joint_stage_owner==nullptr;
 ready=ready&&(!a.m_reflected_particles||AxialReflectionPrepared());
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!CheckReceipt())return false;
 if(m_mm_thermal_requested)InvalidateMassMatrixLease();
 m_particle_phase=ParticlePhase::Finishing;
 // Exactly one unchanged native finisher; the object owns this transition.
 w.FinishImplicitParticleUpdate(endpoint_time);
 if(a.m_reflected_particles&&!CompleteAxialReflection())return false;
 m_particle_phase=ParticlePhase::Finished;CaptureReceipt(false);
 return CheckReceipt();
}

bool DarwinVacuumJointSolve::EndpointAmpereOwnerCurrent(Real endpoint_time) const {
 bool ready=m_active&&a.m_open&&a.m_receipt_ready&&a.m_joint_vacuum.get()==this&&
     m_particle_phase==ParticlePhase::Finished&&m_generation==m_verified_generation&&
     m_generation>0&&m_generation<std::numeric_limits<std::uint64_t>::max()&&
     m_time==a.m_time&&m_dt==a.m_dt&&m_physical_step==w.getistep(0)&&
     w.gett_new(0)==m_time&&endpoint_time==m_time+m_dt&&s.m_theta==.5&&
     s.m_joint_stage_owner==nullptr&&
     s.m_joint_stage_purpose==ThetaImplicitHybrid::JointStagePurpose::Standard&&
     !s.m_circuit_native&&!s.m_native_paired_fields&&m_support_ready;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);
 return ready&&CheckReceipt();
}
bool DarwinVacuumJointSolve::PrepareEndpointAmpere(Real endpoint_time) {
 m_endpoint_ampere.reset();m_endpoint_ampere_generation=0;
 bool ready=EndpointAmpereOwnerCurrent(endpoint_time);
 bool const drive=NativeZeroExternalDriveSupported(w);
 bool const reference=ZeroDriveReference(w,s.m_fext_init);
 ready=ready&&drive&&reference;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 auto candidate=std::make_unique<NativeOwnedEndpointAmpere>(w);
 if(!candidate->Capture(m_dt,{&m_P[0],&m_P[1],&m_P[2]},
                            {&m_V[0],&m_V[1],&m_V[2]})||!CheckReceipt())return false;
 m_endpoint_ampere=std::move(candidate);m_endpoint_ampere_generation=m_generation;
 return true;
}
bool DarwinVacuumJointSolve::VerifyEndpointAmpere(Real endpoint_time) {
 bool ready=EndpointAmpereOwnerCurrent(endpoint_time);
 ready=ready&&m_endpoint_ampere&&m_endpoint_ampere_generation==m_generation;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 return m_endpoint_ampere->Verify({&m_P[0],&m_P[1],&m_P[2]},
                                  {&m_V[0],&m_V[1],&m_V[2]})&&CheckReceipt();
}

std::shared_ptr<NativeEndpointAmpereOrigin const>
DarwinVacuumJointSolve::RetainEndpointAmpereOrigin(Real time) {
 bool ready=EndpointAmpereOwnerCurrent(time);
 ready=ready&&m_endpoint_ampere&&m_endpoint_ampere_generation==m_generation;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return {};
 return m_endpoint_ampere->RetainVerifiedOrigin(time,std::uint64_t(m_physical_step)+1,
     m_generation,{&m_P[0],&m_P[1],&m_P[2]},{&m_V[0],&m_V[1],&m_V[2]});
}

bool DarwinVacuumJointSolve::CheckFinishedSupport() {
 bool ready=m_active&&m_particle_phase==ParticlePhase::Finished&&m_generation==m_verified_generation&&m_particle_support&&m_particle_support->Ready();
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready||!CheckReceipt())return false;
 auto result=m_particle_support->Check(w,warpx::darwin::NativeVacuumParticleSupport::Phase::FinishedEndpoint);
 if(!result.valid){amrex::Print()<<"JOINT_VACUUM particle support failed phase=finished counts=";for(auto n:result.counts)amrex::Print()<<n<<',';amrex::Print()<<'\n';}
 return result.valid&&CheckSupport(false,true);
}

bool DarwinVacuumJointSolve::Transform(MF& output,View const& current,MF const& energy,MF const& density) {
 YeeCurrentView input{current[0],current[1],current[2]};ThermalFaceView faces;
 for(int d=0;d<AMREX_SPACEDIM;++d)faces[d]=&m_face[d];
 a.m_stage_moments->RestrictCurrent(input,faces);
 for(int d=0;d<AMREX_SPACEDIM;++d){
  for(amrex::MFIter it(m_face[d]);it.isValid();++it){auto f=m_face[d].const_array(it);auto n=density.const_array(it);auto u=energy.const_array(it);auto v=m_velocity[d].array(it);auto adv=m_advection[d].array(it);
   amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){int il=i-(d==0),jl=j-(d==1),kl=k-(d==2);Real nf=.5*(n(i,j,k)+n(il,jl,kl)),uf=.5*(u(i,j,k)+u(il,jl,kl));v(i,j,k)=f(i,j,k)/nf;adv(i,j,k)=uf*v(i,j,k);});
  }
  m_velocity[d].OverrideSync(w.Geom(0).periodicity());m_advection[d].OverrideSync(w.Geom(0).periodicity());
 }
 auto dx=w.Geom(0).CellSizeArray();Real rlo=w.Geom(0).ProbLo(0),gm=a.m_stage_options.gamma-1.,factor=m_h/PhysConst::q_e;
 for(amrex::MFIter it(output);it.isValid();++it){auto q=output.array(it);auto u=energy.const_array(it);auto vr=m_velocity[0].const_array(it);auto vz=m_velocity[1].const_array(it);auto fr=m_advection[0].const_array(it);auto fz=m_advection[1].const_array(it);
#if defined(WARPX_DIM_3D)
  auto vt=m_velocity[2].const_array(it);auto ft=m_advection[2].const_array(it);
#endif
  amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){
#if defined(WARPX_DIM_RZ)
   Real r=rlo+(i+.5)*dx[0];Real divv=((rlo+(i+1)*dx[0])*vr(i+1,j,k)-(rlo+i*dx[0])*vr(i,j,k))/(r*dx[0])+(vz(i,j+1,k)-vz(i,j,k))/dx[1];Real divf=((rlo+(i+1)*dx[0])*fr(i+1,j,k)-(rlo+i*dx[0])*fr(i,j,k))/(r*dx[0])+(fz(i,j+1,k)-fz(i,j,k))/dx[1];
#else
   Real divv=(vr(i+1,j,k)-vr(i,j,k))/dx[0]+(vz(i,j+1,k)-vz(i,j,k))/dx[1]+(vt(i,j,k+1)-vt(i,j,k))/dx[2];Real divf=(fr(i+1,j,k)-fr(i,j,k))/dx[0]+(fz(i,j+1,k)-fz(i,j,k))/dx[1]+(ft(i,j,k+1)-ft(i,j,k))/dx[2];
#endif
   q(i,j,k)=factor*(divf+gm*u(i,j,k)*divv);
  });
 }
 return output.is_finite();
}

bool DarwinVacuumJointSolve::Project(Vec& out) {
 using namespace warpx::darwin::increment;
 auto E=w.m_fields.get_alldirs(FT::Efield_fp,0);auto EL=w.m_fields.get_alldirs("hybrid_E_long_fp",0);
 Copy(m_saved_E,E);Copy(m_held_EL,EL);Copy(m_held_phi,*w.m_fields.get(Phi,0));Copy(m_held_phi_low,*w.m_fields.get(scalar_low,0));
 for(int c=0;c<3;++c)Copy(m_held_low[c],*w.m_fields.get(field_low,Dir{c},0));
 Scope restore{[&](){for(int c=0;c<3;++c){Copy(*E[c],m_saved_E[c]);Copy(*EL[c],m_held_EL[c]);Copy(*w.m_fields.get(field_low,Dir{c},0),m_held_low[c]);}Copy(*w.m_fields.get(Phi,0),m_held_phi);Copy(*w.m_fields.get(scalar_low,0),m_held_phi_low);}};
 auto& allrho=*w.m_fields.get(FT::rho_fp,0);MF rho(allrho,amrex::make_alias,allrho.nComp()/2,1);
 a.m_model.HybridPICSolveE(E,w.m_fields.get_alldirs(FT::current_fp,0),w.m_fields.get_alldirs(FT::Bfield_fp,0),rho,w.GetEBUpdateEFlag()[0],0,false,true,a.Dissipation());
 for(int c=0;c<3;++c){Copy(m_delta[c],m_transverse[c]);
  if(m_arithmetic)m_arithmetic->AddTransverseLow(m_delta[c],m_held_EL[c],&m_held_low[c],c,m_generation);
  else AddFull(m_delta[c],m_held_EL[c],&m_held_low[c]);
  for(amrex::MFIter it(*E[c]);it.isValid();++it){auto src=E[c]->array(it);auto full=m_delta[c].const_array(it);auto mask=m_V[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(mask(i,j,k))src(i,j,k)=full(i,j,k);});}
  Images(w,*E[c]);
 }
 ablastr::fields::MultiLevelVectorField source{E};ablastr::fields::MultiLevelScalarField densities{&rho};
 a.m_model.ComputeDarwinELong(densities,m_time+m_h,true,&source);
 m_raw_h=0.;Real scale=0.;
 for(int c=0;c<3;++c){auto& defect=m_delta[c];auto const& projected=*EL[c];
  for(amrex::MFIter it(defect);it.isValid();++it){auto q=defect.array(it);auto p=projected.const_array(it);auto h=m_held_EL[c].const_array(it);auto l=m_held_low[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto x=arithmetic::Add(arithmetic::Sum(p(i,j,k),-h(i,j,k)),{-l(i,j,k),0.});q(i,j,k)=x.hi+x.lo;});}
  m_raw_h=std::max(m_raw_h,defect.norminf(0));scale=std::max({scale,projected.norminf(0),m_held_EL[c].norminf(0)});
 }
 auto& residual=out.getMultiFabBlock(Phi,0);auto const& projected=*w.m_fields.get(Phi,0);
 for(amrex::MFIter it(residual);it.isValid();++it){auto r=residual.array(it);auto p=projected.const_array(it);auto h=m_held_phi.const_array(it);auto l=m_held_phi_low.const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto x=arithmetic::Add(arithmetic::Sum(p(i,j,k),-h(i,j,k)),{-l(i,j,k),0.});r(i,j,k)=x.hi+x.lo;});}
 CanonicalScalar(residual);m_h_target=s.m_darwin_outer_atol+s.m_darwin_outer_rtol*scale;
 return residual.is_finite()&&std::isfinite(m_raw_h);
}

bool DarwinVacuumJointSolve::Residual(Vec& out,Vec const& x,int iteration,bool probe) {
 InvalidateConvergenceReceipt();
 bool ready=m_active&&m_support_ready&&m_particle_phase==ParticlePhase::Trial&&m_generation<std::numeric_limits<std::uint64_t>::max()&&a.m_open&&m_maps<5000&&std::chrono::duration<Real>(std::chrono::steady_clock::now()-m_started).count()<=1800.&&m_time==a.m_time&&m_dt==a.m_dt&&m_physical_step==w.getistep(0)&&w.gett_new(0)==m_time&&s.m_joint_stage_owner==nullptr;
 if(m_mm_requested && !probe)ready=ready && m_mm_base_generation<std::numeric_limits<std::uint64_t>::max();
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);
 if(!ready){if(m_mm_requested){InvalidateMassMatrixLease();m_mm_full_base_current=false;}return false;}
 if(m_mm_requested && probe && !MassMatrixLeaseReady()) {
  InvalidateMassMatrixLease();m_mm_full_base_current=false;
  amrex::Print()<<"JOINT_VACUUM rejected stale MM lease before native map\n";return false;
 }
 if(m_mm_thermal_requested && probe) {
  bool valid=m_mm_thermal && m_mm_thermal_density_epoch!=0;
  amrex::ParallelDescriptor::ReduceBoolAnd(valid);
  if(!valid || !m_mm_thermal->CanSample({m_mm_base_generation,m_mm_endpoint_epoch,m_mm_thermal_density_epoch},m_generation+1)) {
   InvalidateMassMatrixLease();m_mm_full_base_current=false;return false;
  }
 }
 if(m_mm_requested) {
  m_mm_full_base_current=false;
  if(!probe){InvalidateMassMatrixLease();++m_mm_base_generation;}
 }
 bool completed=false;
 Scope failed_mm{[&](){if(m_mm_requested && !completed){InvalidateMassMatrixLease();m_mm_full_base_current=false;}}};
 m_remainder_generation=0;
 if(m_arithmetic&&!m_arithmetic->BindInput(x,probe,m_generation+1))return false;
 m_verified=false;++m_maps;++m_generation;Materialize(x);a.CopyFields(a.m_field,x);MF::Copy(a.m_energy.getMultiFabBlock(U,0),x.getMultiFabBlock(U,0),0,0,1,0);
 auto purpose=s.m_joint_stage_purpose;auto* owner=s.m_joint_stage_owner;s.m_joint_stage_owner=this;s.m_joint_stage_purpose=ThetaImplicitHybrid::JointStagePurpose::Residual;
 Scope context{[&](){s.m_joint_stage_purpose=purpose;s.m_joint_stage_owner=owner;}};
 if(!a.FieldResidual(a.m_field_rhs,a.m_field,iteration,probe)||!CheckSupport(false)||!CheckParticleSupport())return false;
 if(m_mm_requested && probe) {
  bool unchanged=MassMatrixCacheReady() && m_mm_response==s.m_endpoint_current_response.get() &&
      m_mm_endpoint_epoch==s.m_endpoint_current_response->Epoch() &&
      m_mm_density==s.m_mass_matrix_density[0].get() &&
      s.m_mass_matrix_density[0]->CurrentIncrementReady();
  amrex::ParallelDescriptor::ReduceBoolAnd(unchanged);
  if(!unchanged || !CheckReceipt())return false;
 }
 out.setVal(0.);a.CopyFields(out,a.m_field_rhs);Real alpha=m_alpha;
 for(int c=0;c<3;++c){auto const& jp=*w.m_fields.get(FT::hybrid_current_fp_plasma,Dir{c},0);auto const& ji=*w.m_fields.get(FT::current_fp,Dir{c},0);
  for(amrex::MFIter it(m_current[c]);it.isValid();++it){auto j=m_current[c].array(it);auto r=out.getArrayVec()[0][c]->array(it);auto p=jp.const_array(it);auto ion=ji.const_array(it);auto mask=m_V[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j0,int k){Real value=mask(i,j0,k)?p(i,j0,k)-ion(i,j0,k):0.;j(i,j0,k)=value;if(mask(i,j0,k))r(i,j0,k)=alpha*value;});}
 }
 if(m_arithmetic)m_arithmetic->FormResidual(out,x,w.m_fields.get_alldirs(FT::current_fp,0),m_V,V(m_current),m_alpha,m_generation);
 if(!a.ThermalResidual(m_raw_u_field))return false;
 MF::Copy(out.getMultiFabBlock(U,0),m_raw_u_field,0,0,1,0);
 if(!Project(out))return false;
 m_raw.Copy(out);auto const norms=m_raw.blockNorms();m_raw_e=norms[0]*a.m_electric_scale;m_raw_u=norms[1];
 Copy(m_U_images,x.getMultiFabBlock(U,0));stage_detail::fill_images(m_U_images,a.m_geometry);
 if(!Transform(m_correction,V(m_current),m_U_images,a.m_stage->Density()))return false;
 auto& ru=out.getMultiFabBlock(U,0);MF::Add(ru,m_correction,0,0,1,0);
 // A priori operation-bound certificate for the exact inverse row map.
 m_inverse_error=0.;Real bound=0.;
 for(amrex::MFIter it(ru);it.isValid();++it){auto transformed=ru.const_array(it);auto correction=m_correction.const_array(it);auto raw=m_raw_u_field.const_array(it);amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax> op;amrex::ReduceData<Real,Real> data(op);using T=decltype(data)::Type;op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{Real back=transformed(i,j,k)-correction(i,j,k);return {std::abs(back-raw(i,j,k)),4.*std::numeric_limits<Real>::epsilon()*(std::abs(raw(i,j,k))+std::abs(correction(i,j,k)))};});auto v=data.value();m_inverse_error=std::max(m_inverse_error,amrex::get<0>(v));bound=std::max(bound,amrex::get<1>(v));}
 amrex::ParallelDescriptor::ReduceRealMax(m_inverse_error);amrex::ParallelDescriptor::ReduceRealMax(bound);
 if(m_inverse_error>bound)return false;
 if(m_part==1)ru.setVal(0.); // Held-U field subsolve only; raw complete U stays live above.
 bool ok=std::isfinite(out.norm2());
 if(ok && m_mm_thermal_requested)ok=CompleteMassMatrixThermalIncrement(out,x,probe);
 else if(ok && HasResidualRemainder())ok=CompleteResidualRemainder(out);
 if(ok&&m_arithmetic)m_arithmetic->CompleteInput(probe,m_generation);
 if(ok)m_last.Copy(x);
 if(ok && m_mm_requested && !probe){m_mm_base.Copy(x);m_mm_full_base_current=true;}
 completed=ok;
 if(ok && !probe){m_convergence_input=&x;m_convergence_generation=m_generation;m_convergence_part=m_part;}
 if(!probe)amrex::Print()<<std::setprecision(17)<<"JOINT_VACUUM residual map="<<m_maps<<" part="<<m_part<<" E="<<m_raw_e<<" U="<<m_raw_u<<" h="<<m_raw_h<<" h_target="<<m_h_target<<" inverse="<<m_inverse_error<<'\n';
 return ok;
}
bool DarwinVacuumJointSolve::HasResidualRemainder() const noexcept {return m_mm_thermal_requested || a.HasResidualRemainder();}
bool DarwinVacuumJointSolve::CompleteMassMatrixThermalIncrement(Vec const& represented,Vec const& input,bool probe) {
 if(!remainder::All(remainder::ArithmeticSupported() && m_mm_thermal_requested && m_mm_requested && m_mm_thermal!=nullptr))return false;
 if(!m_residual_remainder.IsDefined())m_residual_remainder.Define(represented);
 m_residual_remainder.zero();
 if(probe) {
  auto const& projection=*s.m_mass_matrix_density[0];auto const publication=projection.PublicationEpoch();
  auto const jp=w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0);
  ThermalMassMatrixResponse::Binding const binding{m_mm_base_generation,m_mm_endpoint_epoch,m_mm_thermal_density_epoch};
  if(!m_mm_thermal->Apply(binding,publication,m_generation,projection,input.getMultiFabBlock(U,0),
       {jp[0],jp[1],jp[2]},*a.m_stage,
       m_arithmetic?m_arithmetic->CurrentRemainder(m_generation):YeeCurrentView{}))return false;
  if(!m_mm_thermal->CopyCompanion(m_residual_remainder.getMultiFabBlock(U,0),represented.getMultiFabBlock(U,0),binding,publication,m_generation))return false;
  m_mm_thermal_publication=publication;
 }
 if(m_part==1)m_residual_remainder.getMultiFabBlock(U,0).setVal(0.);
 m_remainder_generation=m_generation;return true;
}
bool DarwinVacuumJointSolve::CopyResidualRemainder(Vec& out) const {
 if(!remainder::All(HasResidualRemainder() && m_active && m_remainder_generation!=0 &&
    m_remainder_generation==m_generation && m_residual_remainder.IsDefined()))return false;
 if(m_mm_thermal_requested) {
  if(!MassMatrixThermalCompanionReady())return false;
  bool valid=out.IsDefined() && out.blockNames()==m_residual_remainder.blockNames() &&
      out.getArrayVec().size()==m_residual_remainder.getArrayVec().size();
  if(valid) {
   for(auto const* block:{U,Phi})valid=valid && remainder::Layout(out.getMultiFabBlock(block,0),m_residual_remainder.getMultiFabBlock(block,0)) &&
       !remainder::Overlap(out.getMultiFabBlock(block,0),m_residual_remainder.getMultiFabBlock(block,0));
   for(int c=0;c<3;++c)valid=valid && remainder::Layout(*out.getArrayVec()[0][c],*m_residual_remainder.getArrayVec()[0][c]) &&
       !remainder::Overlap(*out.getArrayVec()[0][c],*m_residual_remainder.getArrayVec()[0][c]);
  }
  if(!remainder::All(valid))return false;
 }
 out.Copy(m_residual_remainder);
 if(m_mm_thermal_requested)m_mm_thermal_copies[&out]={m_mm_base_generation,m_generation,m_mm_thermal_publication};
 return true;
}
bool DarwinVacuumJointSolve::DifferenceResidualRemainder(Vec& out,Vec const& p,Vec const& pl,
    Vec const& n,Vec const& nl,Real factor) const {
 if(m_mm_thermal_requested) {
  if(!MassMatrixThermalCompanionReady())return false;
  auto plus=m_mm_thermal_copies.find(&pl),minus=m_mm_thermal_copies.find(&nl);
  bool valid=remainder::ArithmeticSupported() && m_active && m_mm_linear_ready && m_remainder_generation==m_generation &&
      plus!=m_mm_thermal_copies.end() && minus!=m_mm_thermal_copies.end();
  if(valid)valid=plus->second[0]==m_mm_base_generation && minus->second[0]==m_mm_base_generation &&
      plus->second[1]<std::numeric_limits<std::uint64_t>::max() &&
      plus->second[1]+1==minus->second[1] && minus->second[1]==m_generation &&
      plus->second[2]<minus->second[2] && minus->second[2]==m_mm_thermal_publication;
  if(!remainder::All(valid))return false;
 }
 if(!remainder::Difference(out.getMultiFabBlock(U,0),p.getMultiFabBlock(U,0),pl.getMultiFabBlock(U,0),
     n.getMultiFabBlock(U,0),nl.getMultiFabBlock(U,0),factor))return false;
 // The new option corrects U only. In particular, preserve the caller's
 // existing ordinary E/h differences and bypass the older V-row companion.
 if(m_mm_thermal_requested)return std::isfinite(out.norm2());
 for(int c=0;c<3;++c)for(amrex::MFIter it(*out.getArrayVec()[0][c]);it.isValid();++it) {
    auto dst=out.getArrayVec()[0][c]->array(it);auto mask=m_V[c].const_array(it);
    auto ph=p.getArrayVec()[0][c]->const_array(it),ps=pl.getArrayVec()[0][c]->const_array(it);
    auto nh=n.getArrayVec()[0][c]->const_array(it),ns=nl.getArrayVec()[0][c]->const_array(it);
    amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
       if(mask(i,j,k))dst(i,j,k)=remainder::Multiply(remainder::Add({ph(i,j,k),ps(i,j,k)},
           remainder::Negate({nh(i,j,k),ns(i,j,k)})),factor).hi;
    });
 }
 return std::isfinite(out.norm2());
}
bool DarwinVacuumJointSolve::CompleteResidualRemainder(Vec const& represented) {
 if(!remainder::All(remainder::ArithmeticSupported() && HasResidualRemainder() && a.m_current_remainder_ready!=0 &&
     a.m_current_remainder_ready==a.m_current_remainder_generation))return false;
 if(!m_residual_remainder.IsDefined())m_residual_remainder.Define(represented);
 m_residual_remainder.zero();
 auto& ul=m_residual_remainder.getMultiFabBlock(U,0);
 if(!a.m_stage->CopyResidualRemainder(ul))return false;
 if(!m_current_remainder) {
     m_current_remainder=std::make_unique<Field>();
     for(int c=0;c<3;++c)(*m_current_remainder)[c].define(m_current[c].boxArray(),m_current[c].DistributionMap(),1,m_current[c].nGrowVect());
 }
 Real const alpha=m_alpha;
 for(int c=0;c<3;++c) {
     auto const& jp=*w.m_fields.get(FT::hybrid_current_fp_plasma,Dir{c},0);
     auto const& ji=*w.m_fields.get(FT::current_fp,Dir{c},0);
     for(amrex::MFIter it(m_current[c]);it.isValid();++it) {
         auto high=m_current[c].const_array(it),p=jp.const_array(it),ion=ji.const_array(it);
         auto pl=a.m_current_remainder[c]->const_array(it);auto mask=m_V[c].const_array(it);
         auto small=(*m_current_remainder)[c].array(it),rl=m_residual_remainder.getArrayVec()[0][c]->array(it);
         auto rh=represented.getArrayVec()[0][c]->const_array(it);
         amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
             auto value=mask(i,j,k)?remainder::Add({p(i,j,k),pl(i,j,k)},{-ion(i,j,k),0.}):remainder::Pair{0.,0.};
             small(i,j,k)=remainder::WithHigh(value,high(i,j,k)).lo;
             if(mask(i,j,k))rl(i,j,k)=remainder::WithHigh(remainder::Multiply(value,alpha),rh(i,j,k)).lo;
         });
     }
 }
 ThermalFaceView high{},low{};
 for(int d=0;d<AMREX_SPACEDIM;++d) {
     if(!m_face_remainder[d])m_face_remainder[d]=std::make_unique<MF>(m_face[d].boxArray(),m_face[d].DistributionMap(),1,0);
     if(!m_velocity_remainder[d])m_velocity_remainder[d]=std::make_unique<MF>(m_face[d].boxArray(),m_face[d].DistributionMap(),1,0);
     if(!m_advection_remainder[d])m_advection_remainder[d]=std::make_unique<MF>(m_face[d].boxArray(),m_face[d].DistributionMap(),1,0);
     high[d]=&m_face[d];low[d]=m_face_remainder[d].get();
 }
 if(!a.m_stage_moments->RestrictCurrentPair({&m_current[0],&m_current[1],&m_current[2]},
     {&(*m_current_remainder)[0],&(*m_current_remainder)[1],&(*m_current_remainder)[2]},high,low))return false;
 auto const& density=a.m_stage->Density();
 for(int d=0;d<AMREX_SPACEDIM;++d) {
     for(amrex::MFIter it(m_face[d]);it.isValid();++it) {
         auto f=m_face[d].const_array(it),fl=m_face_remainder[d]->const_array(it);
         auto n=density.const_array(it),u=m_U_images.const_array(it);
         auto v=m_velocity[d].const_array(it),adv=m_advection[d].const_array(it);
         auto vl=m_velocity_remainder[d]->array(it),al=m_advection_remainder[d]->array(it);
         amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
             int const il=i-(d==0),jl=j-(d==1),kl=k-(d==2);
             Real const nf=.5*(n(i,j,k)+n(il,jl,kl));
             auto velocity=remainder::Divide({f(i,j,k),fl(i,j,k)},nf);
             auto uf=remainder::Multiply(remainder::Sum(u(i,j,k),u(il,jl,kl)),.5);
             vl(i,j,k)=remainder::WithHigh(velocity,v(i,j,k)).lo;
             al(i,j,k)=remainder::WithHigh(remainder::Product(uf,velocity),adv(i,j,k)).lo;
         });
     }
     m_velocity_remainder[d]->OverrideSync(w.Geom(0).periodicity());
     m_advection_remainder[d]->OverrideSync(w.Geom(0).periodicity());
 }
 if(!m_correction_remainder)m_correction_remainder=std::make_unique<MF>(m_correction.boxArray(),m_correction.DistributionMap(),1,0);
 auto const dx=w.Geom(0).CellSizeArray();Real const rlo=w.Geom(0).ProbLo(0);
 Real const gm=a.m_stage_options.gamma-1.,factor=m_h/PhysConst::q_e;
 for(amrex::MFIter it(ul);it.isValid();++it) {
     auto u=m_U_images.const_array(it),raw=m_raw_u_field.const_array(it),ch=m_correction.const_array(it);
     auto total=represented.getMultiFabBlock(U,0).const_array(it);
     auto out=ul.array(it),cl=m_correction_remainder->array(it);
     amrex::GpuArray<amrex::Array4<Real const>,AMREX_SPACEDIM> v,vl,f,fl;
     for(int d=0;d<AMREX_SPACEDIM;++d) {
         v[d]=m_velocity[d].const_array(it);vl[d]=m_velocity_remainder[d]->const_array(it);
         f[d]=m_advection[d].const_array(it);fl[d]=m_advection_remainder[d]->const_array(it);
     }
     amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
         namespace dd=warpx::thermal::remainder;
         dd::Pair divv{0.,0.},divf{0.,0.};
         for(int d=0;d<AMREX_SPACEDIM;++d) {
             int const ip=i+(d==0),jp=j+(d==1),kp=k+(d==2);
             dd::Pair av{v[d](ip,jp,kp),vl[d](ip,jp,kp)},bv{v[d](i,j,k),vl[d](i,j,k)};
             dd::Pair af{f[d](ip,jp,kp),fl[d](ip,jp,kp)},bf{f[d](i,j,k),fl[d](i,j,k)};
             Real denominator=dx[d];
#if defined(WARPX_DIM_RZ)
             if(d==0) {
                 Real const rp=rlo+(i+1)*dx[0],rm=rlo+i*dx[0],r=rlo+(i+.5)*dx[0];
                 av=dd::Multiply(av,rp);bv=dd::Multiply(bv,rm);af=dd::Multiply(af,rp);bf=dd::Multiply(bf,rm);denominator=r*dx[0];
             }
#endif
             divv=dd::Add(divv,dd::Divide(dd::Add(av,dd::Negate(bv)),denominator));
             divf=dd::Add(divf,dd::Divide(dd::Add(af,dd::Negate(bf)),denominator));
         }
         auto correction=dd::Multiply(dd::Add(divf,dd::Product(dd::Multiply({gm,0.},u(i,j,k)),divv)),factor);
         cl(i,j,k)=dd::WithHigh(correction,ch(i,j,k)).lo;
         out(i,j,k)=dd::WithHigh(dd::Add({raw(i,j,k),out(i,j,k)},correction),total(i,j,k)).lo;
     });
 }
 if(m_part==1)ul.setVal(0.);
 if(!std::isfinite(m_residual_remainder.norm2()))return false;
 m_remainder_generation=m_generation;return true;
}
Real DarwinVacuumJointSolve::StepBound(Vec const& state,Vec const& direction) const {return a.StepBound(state,direction);}
bool DarwinVacuumJointSolve::RawAccepted() const {return std::isfinite(m_raw_e)&&std::isfinite(m_raw_u)&&std::isfinite(m_raw_h)&&m_raw_e<=m_field_target&&m_raw_u<=m_energy_target&&m_raw_h<=m_h_target;}
bool DarwinVacuumJointSolve::PhysicalConverged(Vec const& input) const {
 bool ready=m_active && m_support_ready && m_particle_phase==ParticlePhase::Trial &&
     m_convergence_input==&input && m_convergence_generation!=0 &&
     m_convergence_generation==m_generation && m_convergence_part==m_part &&
     a.m_open && m_time==a.m_time && m_dt==a.m_dt &&
     m_physical_step==w.getistep(0) && w.gett_new(0)==m_time &&
     std::isfinite(m_raw_e) && std::isfinite(m_raw_h) &&
     m_raw_e<=m_field_target && m_raw_h<=m_h_target &&
     (m_part==1 || (m_part==0 && std::isfinite(m_raw_u) && m_raw_u<=m_energy_target));
 // Addresses/generations are local identities; only validity is reduced.
 // The physical norms already came from the current FULL map reductions.
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);return ready;
}
void DarwinVacuumJointSolve::RestoreInput(Vec const& input) {if(m_arithmetic&&!m_arithmetic->RestoreInput(input)){Invalidate();return;}if(m_mm_requested){InvalidateMassMatrixLease();m_mm_full_base_current=false;}Residual(m_residual,input,0,false);m_remainder_generation=0;if(HasResidualRemainder())a.InvalidateCurrentRemainder();s.InvalidateMassMatrices();InvalidateConvergenceReceipt();}

bool DarwinVacuumJointSolve::Freeze(Vec const& x,int iteration,bool use_pc,ThermalSolveResult& stats) {
 InvalidateConvergenceReceipt();
 bool ready=m_active&&m_support_ready;
 if(m_mm_requested)ready=ready && use_pc && m_particle_phase==ParticlePhase::Trial &&
     a.m_open && m_time==a.m_time && m_dt==a.m_dt && m_physical_step==w.getistep(0) &&
     w.gett_new(0)==m_time && s.m_joint_stage_owner==nullptr;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);
 if(!ready){if(m_mm_requested)InvalidateMassMatrixLease();return false;}
 if(m_mm_requested) {
  InvalidateMassMatrixLease();
  bool scope=MassMatrixScopeSupported();amrex::ParallelDescriptor::ReduceBoolAnd(scope);
  if(!scope)return false;
  if(!MassMatrixBaseMatches(x) || !CheckSupport(false) || !CheckParticleSupport()) {
   amrex::Print()<<"JOINT_VACUUM rejected MM Freeze without current full particle base\n";return false;
  }
 }
 auto purpose=s.m_joint_stage_purpose;auto* owner=s.m_joint_stage_owner;s.m_joint_stage_owner=this;s.m_joint_stage_purpose=ThetaImplicitHybrid::JointStagePurpose::Residual;
 Scope context{[&](){s.m_joint_stage_purpose=purpose;s.m_joint_stage_owner=owner;}};
 if(m_arithmetic&&!m_arithmetic->FreezeInput(x,m_generation))return false;
 if(!a.Freeze(x,iteration,use_pc,stats))return false;
 Copy(m_frozen_U,m_U_images);Copy(m_frozen_n,a.m_stage->Density());
 auto& allrho=*w.m_fields.get(FT::rho_fp,0);MF midpoint(allrho,amrex::make_alias,allrho.nComp()/2,1);
 if(!m_support->FreezeEdges(midpoint,*w.m_fields.get("hybrid_rho_vacmask_fp",0)))return false;
 Real beta=PhysConst::epsilon_0*a.m_model.m_electron_inertia_mass/(PhysConst::q_e*PhysConst::q_e*a.m_model.m_n0_ref*m_h*m_h);
 for(int c=0;c<3;++c)for(amrex::MFIter it(m_kappa[c]);it.isValid();++it){auto out=m_kappa[c].array(it);auto k=m_support->KappaEdge(c).const_array(it);auto p=m_P[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int l){out(i,j,l)=p(i,j,l)?beta*k(i,j,l):0.;});}
 bool const frozen=m_schur->FreezeEdges({&m_kappa[0],&m_kappa[1],&m_kappa[2]});
 if(!m_mm_requested)return frozen;
 bool valid=frozen && MassMatrixCacheReady();
 amrex::ParallelDescriptor::ReduceBoolAnd(valid);
 if(!valid){InvalidateMassMatrixLease();return false;}
 // Particle-only storage is mutually exclusive with the later verified
 // stage receipt. Final acceptance always replaces it after a full push.
 CaptureReceipt(false);
 if(!CheckReceipt()){InvalidateMassMatrixLease();return false;}
 m_mm_response=s.m_endpoint_current_response.get();
 m_mm_density=s.m_mass_matrix_density[0].get();
 m_mm_endpoint_epoch=m_mm_response->Epoch();m_mm_interval_width=s.m_ncomp_xx[0];
 m_mm_lease_base_generation=m_mm_base_generation;m_mm_linear_ready=true;
 if(m_mm_thermal_requested) {
  m_mm_thermal_density_epoch=m_mm_density->BaseEpoch();
  auto const ji=w.m_fields.get_alldirs(FT::current_fp,0),jp=w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0);
  ThermalMassMatrixResponse::ConstFaces gamma;for(int d=0;d<AMREX_SPACEDIM;++d)gamma[d]=a.m_number_flux[d].get();
  ThermalMassMatrixResponse::Base const base{{m_mm_base_generation,m_mm_endpoint_epoch,m_mm_thermal_density_epoch},
      midpoint,x.getMultiFabBlock(U,0),m_raw_u_field,m_correction,{ji[0],ji[1],ji[2]},
      {jp[0],jp[1],jp[2]},{&m_V[0],&m_V[1],&m_V[2]},gamma,*a.m_stage_moments,*a.m_stage,*m_mm_density,
      m_arithmetic?m_arithmetic->CurrentRemainder(m_generation):YeeCurrentView{},
      m_arithmetic?YeeCurrentView{&m_current[0],&m_current[1],&m_current[2]}:YeeCurrentView{}};
  if(!m_mm_thermal->Capture(base)){InvalidateMassMatrixLease();return false;}
 }
 return true;
}
bool DarwinVacuumJointSolve::Precondition(Vec& out,Vec const& rhs) {
 InvalidateConvergenceReceipt();
 if(m_mm_requested && !MassMatrixLeaseReady()){InvalidateMassMatrixLease();return false;}
 bool ready=m_active&&m_hodge&&m_hodge->Ready();amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;++m_pc_calls;
 Real alpha=m_alpha;
 for(int c=0;c<3;++c)for(amrex::MFIter it(m_pc_current[c]);it.isValid();++it){auto j=m_pc_current[c].array(it);auto r=rhs.getArrayVec()[0][c]->const_array(it);auto v=m_V[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j0,int k){j(i,j0,k)=v(i,j0,k)?r(i,j0,k)/alpha:0.;});}
 if(!Transform(m_correction,V(m_pc_current),m_frozen_U,m_frozen_n))return false;
 m_pc_rhs.setVal(0.);a.CopyFields(m_pc_rhs,rhs);
 for(int c=0;c<3;++c)for(amrex::MFIter it(*m_pc_rhs.getArrayVec()[0][c]);it.isValid();++it){auto f=m_pc_rhs.getArrayVec()[0][c]->array(it);auto p=m_P[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(!p(i,j,k))f(i,j,k)=0.;});}
 MF::LinComb(m_pc_rhs.getMultiFabBlock(U,0),1.,rhs.getMultiFabBlock(U,0),0,-1.,m_correction,0,0,1,0);
 if(m_part==1)m_pc_rhs.getMultiFabBlock(U,0).setVal(0.);
 if(!a.Precondition(m_pc_out,m_pc_rhs))return false;
 out.setVal(0.);a.CopyFields(out,m_pc_out);MF::Copy(out.getMultiFabBlock(U,0),m_pc_out.getMultiFabBlock(U,0),0,0,1,0);
 if(m_part==1)out.getMultiFabBlock(U,0).setVal(0.);
 for(int c=0;c<3;++c)for(amrex::MFIter it(*out.getArrayVec()[0][c]);it.isValid();++it){auto f=out.getArrayVec()[0][c]->array(it);auto p=m_P[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if(!p(i,j,k))f(i,j,k)=0.;});}
 // Complementary nodal auxiliary response. Approximation belongs only to PC;
 // the saddle RHS below retains its complete gradient and K_VP contribution.
 auto const& hrhs=rhs.getMultiFabBlock(Phi,0);
 m_hodge->CanonicalPotential(m_phi_w,hrhs);
 MF::LinComb(m_phi_a,1.,hrhs,0,-1.,m_phi_w,0,0,1,0);CanonicalScalar(m_phi_a);Gradient(V(m_pc_h),m_phi_a);
 auto auxiliary=m_schur->Correct({&m_pc_h[0],&m_pc_h[1],&m_pc_h[2]},{&m_zero[0],&m_zero[1],&m_zero[2]});m_auxiliary_iterations+=auxiliary.iterations;
 if(!auxiliary.converged||!std::isfinite(auxiliary.residual)||auxiliary.residual>auxiliary.target){amrex::Print()<<"JOINT_VACUUM PC auxiliary failed residual="<<auxiliary.residual<<" target="<<auxiliary.target<<'\n';return false;}
 Copy(m_phi_a,m_schur->CorrectionPotential());m_phi_a.mult(-1.,0,1,0);
 m_hodge->CanonicalPotential(m_phi_w,m_phi_a);MF::Subtract(m_phi_a,m_phi_w,0,0,1,0);CanonicalScalar(m_phi_a);
 Gradient(V(m_pc_d),m_phi_a);m_hodge->Curl(V(m_pc_k),out.getArrayVec()[0]);Real b=m_b;
 for(int c=0;c<3;++c)for(amrex::MFIter it(m_pc_d[c]);it.isValid();++it){auto d=m_pc_d[c].array(it);auto r=rhs.getArrayVec()[0][c]->const_array(it);auto k=m_pc_k[c].const_array(it);auto v=m_V[c].const_array(it);amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int l){d(i,j,l)=v(i,j,l)?r(i,j,l)+b*k(i,j,l)+d(i,j,l):0.;});}
 Gradient(V(m_pc_h),hrhs);
 auto result=m_hodge->Solve(V(m_pc_e),m_phi_w,V(m_pc_d),V(m_pc_h));m_saddle_iterations+=result.iterations;
 // Explicit fresh incomplete-PC status is never a root/physical receipt.
 if(!result.converged||!result.finite||result.residual>result.target){amrex::Print()<<"JOINT_VACUUM PC saddle failed status="<<result.status<<" iterations="<<result.iterations<<" fresh="<<result.residual<<" target="<<result.target<<'\n';return false;}
 for(int c=0;c<3;++c)MF::Add(*out.getArrayVec()[0][c],m_pc_e[c],0,0,1,0);
 MF::LinComb(out.getMultiFabBlock(Phi,0),1.,m_phi_a,0,1.,m_phi_w,0,0,1,0);CanonicalScalar(out.getMultiFabBlock(Phi,0));
 return std::isfinite(out.norm2());
}

int DarwinVacuumJointSolve::Solve(int step) {
 m_started=std::chrono::steady_clock::now();m_step=step;
 if(!Prepare()){Invalidate();return -12;}if(!m_active)return 4;
 if(!Residual(m_residual,m_state,0,false)){Invalidate();return -12;}
 Real field_rtol,field_atol;int field_maxits;s.m_nlsolver->GetSolverParams(field_rtol,field_atol,field_maxits);
 Real field_reference=m_raw_e;if(a.m_energy_reference<0.)a.m_energy_reference=m_raw_u;
 m_field_target=std::max(field_atol,field_rtol*field_reference);m_energy_target=std::max(a.m_options.absolute_tolerance,a.m_options.relative_tolerance*a.m_energy_reference);
 auto solve=[&](int part)->bool {
  m_part=part;auto options=a.m_options;
  options.max_newton_iterations=part==1?field_maxits:a.m_options.max_newton_iterations;
  options.block_reference_norms={field_reference/a.m_electric_scale,a.m_energy_reference,m_h_target/a.m_electric_scale};
  options.block_relative_tolerances={field_rtol,part==1?0.:a.m_options.relative_tolerance,0.};
  options.block_absolute_tolerances={field_atol/a.m_electric_scale,part==1?1.:a.m_options.absolute_tolerance,m_h_target/a.m_electric_scale};
  auto result=SolveThermalSystem(*this,m_state,options);
  amrex::Print()<<"JOINT_VACUUM solve part="<<part<<" status="<<int(result.status)<<" Newton="<<result.newton_iterations<<" GMRES="<<result.linear_iterations<<" maps="<<m_maps<<" PC_calls="<<m_pc_calls<<" scalar_iterations="<<m_auxiliary_iterations<<" saddle_iterations="<<m_saddle_iterations<<'\n';
  return result.status==ThermalSolveStatus::Converged;
 };
 bool converged=false;
 if(a.m_coupled)converged=solve(0);
 else for(int outer=0;outer<a.m_outer_iterations;++outer){
  if(!solve(1))break;
  // The held-field scalar solve has no native MM particle response. Revoke
  // that lease explicitly; the next joint map/Freeze must rebuild it.
  if(m_mm_thermal_requested){InvalidateMassMatrixLease();m_mm_full_base_current=false;}
  // Native thermal stage at this fresh particle/current/density context.
  auto options=a.m_options;options.block_reference_norms={a.m_energy_reference};
  auto thermal=SolveThermalStage(*a.m_stage,a.m_energy,a.m_endpoint,U,options);
  if(thermal.status!=ThermalSolveStatus::Converged)break;
  MF::Copy(m_state.getMultiFabBlock(U,0),a.m_energy.getMultiFabBlock(U,0),0,0,1,0);m_part=0;
  if(!Residual(m_residual,m_state,0,false))break;
  amrex::Print()<<"JOINT_VACUUM decoupled outer="<<outer<<" E="<<m_raw_e<<" U="<<m_raw_u<<" h="<<m_raw_h<<'\n';
  if(RawAccepted()){converged=true;break;}
 }
 m_part=0;
 if(!converged||!Residual(m_residual,m_state,0,false)||!RawAccepted()){
  amrex::Print()<<"JOINT_VACUUM rejected raw root E="<<m_raw_e<<" target="<<m_field_target<<" U="<<m_raw_u<<" target="<<m_energy_target<<" h="<<m_raw_h<<" target="<<m_h_target<<'\n';Invalidate();return -12;
 }
 a.CopyFields(s.m_E,m_state);MF::Copy(a.m_energy.getMultiFabBlock(U,0),m_state.getMultiFabBlock(U,0),0,0,1,0);
 CaptureReceipt();m_verified=true;m_verified_generation=m_generation;
 if(a.m_stopping_carry)a.m_stopping_carry->ArmAtVerifiedRoot(*this);
 amrex::Print()<<"JOINT_VACUUM raw_root=1 fresh_full_push=1 whole_step_accepted=0 maps="<<m_maps<<" E="<<m_raw_e<<" U="<<m_raw_u<<" h="<<m_raw_h<<'\n';return 2;
}

void DarwinVacuumJointSolve::VisitReceipt(std::function<void(void const*,std::size_t)> const& visit,std::vector<std::size_t>& schema) const {
 for(auto const& name:w.GetPartContainer().GetSpeciesNames()){
  auto& pc=w.GetPartContainer().GetParticleContainerFromName(name);schema.push_back(pc.GetParticles().size());
  for(std::size_t lev=0;lev<pc.GetParticles().size();++lev){schema.push_back(pc.GetParticles(lev).size());
   for(auto const& [key,tile]:pc.GetParticles(lev)){
    schema.insert(schema.end(),{std::size_t(key.first),std::size_t(key.second),std::size_t(tile.numParticles()),std::size_t(tile.numNeighborParticles()),std::size_t(tile.NumRealComps()),std::size_t(tile.NumIntComps())});
    auto const& soa=tile.GetStructOfArrays();auto const& ids=soa.GetIdCPUData();visit(ids.data(),ids.size()*sizeof(*ids.data()));
    for(int c=0;c<tile.NumRealComps();++c){auto const& x=soa.GetRealData(c);visit(x.data(),x.size()*sizeof(*x.data()));}
    for(int c=0;c<tile.NumIntComps();++c){auto const& x=soa.GetIntData(c);visit(x.data(),x.size()*sizeof(*x.data()));}
   }
  }
 }
 for(auto* field:m_receipt_fields){schema.push_back(field->size());schema.push_back(field->nComp());for(amrex::MFIter it(*field);it.isValid();++it){auto const& f=(*field)[it];schema.push_back(it.index());visit(f.dataPtr(),f.size()*sizeof(Real));}}
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
 visit(amrex::getRandState(),ThermalRandomDeviceBytes());
#endif
}
void DarwinVacuumJointSolve::CaptureReceipt(bool include_fields) {
 m_particles.clear();m_particle_schema.clear();m_receipt_fields.clear();
 if(include_fields)for(auto const* name:{"hybrid_A_fp","Bfield_fp","hybrid_E_long_fp",warpx::darwin::increment::field_low,"current_fp","hybrid_current_fp_plasma"})for(auto* f:w.m_fields.get_alldirs(name,0))m_receipt_fields.push_back(f);
 if(include_fields)for(auto const* name:{Phi,warpx::darwin::increment::scalar_low,"rho_fp","hybrid_electron_pressure_fp"})m_receipt_fields.push_back(w.m_fields.get(name,0));
 if(include_fields&&m_arithmetic)m_arithmetic->AppendReceipt(m_receipt_fields);
 VisitReceipt([&](void const* pointer,std::size_t size){Bytes record;record.pointer=pointer;record.size=size;record.saved.resize(size);if(size)amrex::Gpu::dtod_memcpy(record.saved.data(),pointer,size);m_particles.push_back(std::move(record));},m_particle_schema);
 amrex::Gpu::synchronize();m_next_id=WarpXParticleContainer::ParticleType::the_next_id;std::ostringstream rng;amrex::SaveRandomState(rng);m_host_rng=rng.str();
 Copy(m_verified_Et,V(m_transverse));Copy(m_verified_push,V(m_push));
 std::size_t bytes=0;for(auto const& b:m_particles)bytes+=b.size;
 amrex::Print()<<"JOINT_VACUUM receipt local_bytes="<<bytes<<" storage_count="<<m_particles.size()<<" device_resident=1\n";
}
bool DarwinVacuumJointSolve::CheckReceipt() const {
 std::vector<std::size_t> schema;std::size_t index=0;bool ok=true;
 // Complete structural preflight on every rank before reading saved pointers.
 VisitReceipt([&](void const* pointer,std::size_t size){if(index>=m_particles.size()){ok=false;return;}auto const& b=m_particles[index++];ok=ok&&pointer==b.pointer&&size==b.size;},schema);
 ok=ok&&index==m_particles.size()&&schema==m_particle_schema&&m_next_id==WarpXParticleContainer::ParticleType::the_next_id;
 std::ostringstream rng;amrex::SaveRandomState(rng);ok=ok&&rng.str()==m_host_rng;
 amrex::ParallelDescriptor::ReduceBoolAnd(ok);if(!ok)return false;
 amrex::Gpu::DeviceScalar<int> mismatch(0);auto* failed=mismatch.dataPtr();
 for(auto const& b:m_particles){auto const* x=static_cast<unsigned char const*>(b.pointer);auto const* old=b.saved.data();amrex::For(b.size,[=]AMREX_GPU_DEVICE(amrex::Long i){if(x[i]!=old[i])amrex::Gpu::Atomic::Max(failed,1);});}
 int bad=mismatch.dataValue();amrex::ParallelDescriptor::ReduceIntMax(bad);return bad==0;
}
bool DarwinVacuumJointSolve::PublishVerifiedStage() {
 bool const zero_drive=NativeZeroExternalDriveSupported(w);
 bool const zero_reference=ZeroDriveReference(w,s.m_fext_init);
 bool ready=zero_drive&&zero_reference&&m_active&&m_verified&&m_particle_phase==ParticlePhase::Trial&&m_generation==m_verified_generation&&a.m_open&&a.m_receipt_ready&&m_time==a.m_time&&m_dt==a.m_dt&&m_physical_step==w.getistep(0)&&w.gett_new(0)==m_time&&s.m_joint_stage_owner==nullptr;
 amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 for(int c=0;c<3;++c){auto const& input=*s.m_E.getArrayVec()[0][c];auto const& verified=*m_last.getArrayVec()[0][c];ready=Equal(input,verified)&&ready;}
 ready=Equal(a.m_energy.getMultiFabBlock(U,0),m_last.getMultiFabBlock(U,0))&&ready;
 ready=CheckReceipt()&&ready;amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return false;
 auto purpose=s.m_joint_stage_purpose;auto* owner=s.m_joint_stage_owner;s.m_joint_stage_owner=this;s.m_joint_stage_purpose=ThetaImplicitHybrid::JointStagePurpose::VerifiedPublication;
 {Scope restore{[&](){s.m_joint_stage_purpose=purpose;s.m_joint_stage_owner=owner;}};s.UpdateWarpXFields(s.m_E,false,m_time);}
 for(int c=0;c<3;++c)ready=Equal(m_transverse[c],m_verified_Et[c])&&Equal(m_push[c],m_verified_push[c])&&ready;
 ready=CheckReceipt()&&ready;amrex::ParallelDescriptor::ReduceBoolAnd(ready);
 amrex::Print()<<"JOINT_VACUUM stage_publication exact="<<ready<<" no_particle_push=1 legacy_half_recovery=0 endpoint_recovery_unchanged=1\n";
 m_verified=false;if(ready)m_particle_phase=ParticlePhase::Published;return ready;
}
} // namespace warpx::thermal
