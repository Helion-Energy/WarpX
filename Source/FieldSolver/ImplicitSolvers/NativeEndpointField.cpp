/* Diagnostic prototype. Copyright 2026 The WarpX Community. License BSD-3-Clause-LBNL. */

#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "NativePairedDarwinFields.H"
#include "NativeLongitudinalIncrement.H"

#include "NativeEndpointField.H"
#include "NativeCurrentTolerance.H"
#include "NativeRZMidpointStoppingEvent.H"
#include "ThermalCurrentRemainder.H"
#include "NativeEndpointArithmetic.H"
#include "NativePECPlasma.H"
#include "NativeCoilField.H"
#include "NativeCircuitFieldRate.H"
#include "Circuit/CircuitCoupler.H"
#include "Circuit/CircuitCoupling.H"
#include "NativeVacuumEndpoint.H"
#include "NativeVacuumMassMatrixCapabilities.H"
#include "NativeVacuumConstraint.H"
#include "NativeEndpointCurrentResponse.H"
#include "NativeInstantaneousForce.H"
#include "NativeJouleWork.H"
#include "NativeVacuumJouleError.H"
#include "NativeSplitAmpere.H"
#include "NativeInstantaneousIonCurrent.H"
#include "NativeInstantaneousIonRate.H"
#include "NativeInstantaneousIonIncrement.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "Initialization/PlasmaInjector.H"
#include "Python/callbacks.H"
#include "Particles/Gather/GetExternalFields.H"
#include "EmbeddedBoundary/Enabled.H"
#include "DarwinInitialRateSchur.H"
#include "DarwinLongitudinalSchur.H"
#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include "DarwinABoundary.H"
#include "ImplicitFieldRollback.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "WarpXSolverVec.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "NonlinearSolvers/FlexibleGMRES.H"
#include "Particles/Deposition/EsirkepovCurrentRate.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include "ablastr/coarsen/sample.H"
#include "ablastr/utils/Communication.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
using Real=amrex::Real;
using MF=amrex::MultiFab;
using Vector=std::array<MF,3>;
using View=ablastr::fields::VectorField;
using ablastr::fields::Direction;
using warpx::fields::FieldType;
namespace {
// Private R82 qualification switch. Full nonlinear/endpoint acceptance remains active.
bool NativePECMMQualificationEnabled() {
    bool enabled=false;
    amrex::ParmParse("pec_mm_qualification").query("enable",enabled);
    return enabled;
}
struct EndpointNumericalFailure {
    NativeEndpointFailure failure;
    char const* reason;
};
void EndpointRequire(bool condition, NativeEndpointFailure failure,
                     bool recoverable, char const* reason) {
    if (!condition && recoverable) { throw EndpointNumericalFailure{failure,reason}; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(condition,reason);
}
// Expensive development evidence is explicit. Physical solve/acceptance and
// registry restoration remain active independently of this observer switch.
bool EndpointAuditEnabled()
{
    bool audit = false;
    amrex::ParmParse("endpoint_diagnostic").query("audit", audit);
    return audit;
}
View Pointers(Vector& f){return {&f[0],&f[1],&f[2]};}
Vector Allocate(View const& shape){Vector x;for(int d=0;d<3;++d)x[d].define(shape[d]->boxArray(),shape[d]->DistributionMap(),1,shape[d]->nGrowVect());return x;}
Vector CurrentAllocate(View const& shape) {
    if(!warpx::darwin::NativePECPlasmaEnabled())return Allocate(shape);
    Vector result;for(int c=0;c<3;++c)result[c].define(shape[c]->boxArray(),shape[c]->DistributionMap(),1,1);return result;
}
Vector CurrentClone(View const& shape) {
    auto result=CurrentAllocate(shape);for(int c=0;c<3;++c)MF::Copy(result[c],*shape[c],0,0,1,result[c].nGrowVect());return result;
}
Vector Clone(View const& shape){auto x=Allocate(shape);for(int d=0;d<3;++d)MF::Copy(x[d],*shape[d],0,0,1,shape[d]->nGrowVect());return x;}
void Copy(View const& a,View const& b,int ghosts=-1){for(int d=0;d<3;++d)MF::Copy(*a[d],*b[d],0,0,1,ghosts<0?amrex::min(a[d]->nGrowVect(),b[d]->nGrowVect()):amrex::IntVect(ghosts));}
// The native source uses zero physical extension except at PMC faces.
// A scalar and a Yee component can use the existing staggering-aware helper;
// a fully nodal vector needs explicit odd parity for its normal component.
void PMCImages(MF& f)
{
    auto const& geometry=WarpX::GetInstance().Geom(0);
    amrex::GpuArray<int,AMREX_SPACEDIM> lower{},upper{};
    bool any=false;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        lower[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        upper[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
        any=any||lower[d]||upper[d];
    }
    if(!any) return; // Preserve the qualified periodic route exactly.
    f.OverrideSync(geometry.periodicity());
    f.FillBoundary(geometry.periodicity());
    if(f.nComp()==1) {
        ApplyDarwinPMCVectorBoundary(f,geometry,lower,upper);
        return;
    }
    AMREX_ALWAYS_ASSERT(f.nComp()==3 && f.ixType().nodeCentered());
    auto const domain=amrex::convert(geometry.Domain(),f.ixType());
    auto const lo=domain.smallEnd(),hi=domain.bigEnd();
    for(amrex::MFIter mfi(f);mfi.isValid();++mfi) {
        auto const a=f.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),3,
            [=] AMREX_GPU_DEVICE(int i,int j,int k,int c) {
                int const point[3]={i,j,k};
                int mirror[3]={i,j,k};
                bool reflected=false;
                Real sign=1.;
                for(int d=0;d<AMREX_SPACEDIM;++d) {
                    bool const low=lower[d] && point[d]<lo[d];
                    bool const high=upper[d] && point[d]>hi[d];
                    if(low||high) {
                        mirror[d]=2*(low?lo[d]:hi[d])-point[d];
                        reflected=true;
#if defined(WARPX_DIM_RZ)
                        int const normal=d==0?0:2;
#else
                        int const normal=d;
#endif
                        if(c==normal) sign=-sign;
                    }
                }
                if(reflected) a(i,j,k,c)=sign*a(mirror[0],mirror[1],mirror[2],c);
            });
    }
}
void Images(MF& f)
{
    f.setBndry(0.);
    f.OverrideSync(WarpX::GetInstance().Geom(0).periodicity());
    f.FillBoundary(WarpX::GetInstance().Geom(0).periodicity());
    PMCImages(f);
}


// The same nodal coefficient feeds every direct Yee map: live midpoint,
// initial/endpoint constrained solve, and the frozen potential operator.
// This candidate deliberately rejects masked/tapered contexts instead of
// silently redefining their physical support. The old nodal route is unchanged.
bool BuildNativeInertiaKappa(WarpX& sim, MF const& density, int component, MF& kappa)
{
    auto const& model = *sim.get_pointer_HybridPICModel();
    AMREX_ALWAYS_ASSERT(density.boxArray() == kappa.boxArray() &&
        density.DistributionMap() == kappa.DistributionMap() &&
        component >= 0 && component < density.nComp());
    Real const floor = PhysConst::q_e * model.m_n_floor;
    if (!density.is_finite(component, 1, 0) || density.min(component, 0) <= floor) {
        return false;
    }
    auto const* pedestal = model.DensityPedestal(0);
    bool const ped = pedestal != nullptr;
    Real const ref = PhysConst::q_e * model.m_n0_ref;
    for (amrex::MFIter mfi(kappa); mfi.isValid(); ++mfi) {
        auto const out = kappa.array(mfi);
        auto const rho = density.const_array(mfi);
        amrex::Array4<Real const> p;
        if (ped) { p = pedestal->const_array(mfi); }
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            out(i,j,k) = ref / amrex::max(floor,
                rho(i,j,k,component) + (ped ? p(i,j,k) : 0.));
        });
    }
    Images(kappa);
    return kappa.is_finite(0, 1, 0) && kappa.min(0) > 0.;
}

enum class DriveRate { Homogeneous, Instantaneous, Secant, Curvature };
void NativeCurlRate(WarpX&,View const&,View const&,View,View,
                    Real,Real,DriveRate);
void BuildDriveReference(WarpX&,MF&,int,Real,Real,DriveRate);

// Deposit the exact instantaneous limit and current rate. All particles are
// read-only. The native momentum-conserving zero-orbit gather uses centered auxiliary fields.
template<int N> void DepositRate(WarpX& sim,View E,View B,View current,View rate,bool linear_only){
#if defined(WARPX_DIM_RZ)
 constexpr int Dim=2;
#elif defined(WARPX_DIM_3D)
 constexpr int Dim=3;
#else
 amrex::Abort("Native startup rate supports RZ/Cartesian3D only");constexpr int Dim=3;
#endif
 for(auto* x:current)x->setVal(0.);for(auto* x:rate)x->setVal(0.);
 auto const inverse=WarpX::InvCellSize(0);auto const& geom=sim.Geom(0);
 auto const lower=geom.ProbLoArray();
 amrex::XDim3 const origin{lower[0],Dim==3?lower[1]:0.,lower[AMREX_SPACEDIM-1]};
 amrex::Dim3 const zero{0,0,0};
 amrex::GpuArray<amrex::GpuArray<double,2>,AMREX_SPACEDIM> domain{};
 amrex::GpuArray<amrex::GpuArray<bool,2>,AMREX_SPACEDIM> crop{};
 for(auto const& name:sim.GetPartContainer().GetSpeciesNames()){
  auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!pc.DoFieldIonization(),"Startup oracle requires nonionizing species");
  auto const* physical=dynamic_cast<PhysicalParticleContainer const*>(&pc);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(physical,"Native endpoint diagnostic requires physical species");
  auto const& all_particles=sim.GetPartContainer();
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    (all_particles.m_E_ext_particle_s=="none"||all_particles.m_E_ext_particle_s=="constant")&&
    (all_particles.m_B_ext_particle_s=="none"||all_particles.m_B_ext_particle_s=="constant"),
    "Native endpoint diagnostic does not support external particle field parsers, files or lenses");
  for(auto const value:pc.m_E_external_particle)AMREX_ALWAYS_ASSERT_WITH_MESSAGE(value==0.,"Native endpoint diagnostic does not support constant external particle E");
  for(auto const value:pc.m_B_external_particle)AMREX_ALWAYS_ASSERT_WITH_MESSAGE(value==0.,"Native endpoint diagnostic does not support constant external particle B");
  Real const q=pc.getCharge();
  if(q==0.) {
    // Native massive neutrals free stream in the ordinary particle advance.
    // They supply neither instantaneous current nor its field response, and
    // their constructor intentionally disables deposition and gathering.
    // Native trajectory switches remain owned by the ordinary pusher; they
    // cannot add charge or a Lorentz current response to an exact neutral.
    // The native constructor rejects classical radiation for every nonlepton;
    // neutralized leptons and all QED processes remain outside this route.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !dynamic_cast<SubcycledParticleContainer const*>(&pc) &&
        std::isfinite(pc.getMass()) && pc.getMass()>0. &&
        pc.do_not_deposit && !pc.HasiAttrib("ionizationLevel") &&
        !pc.AmIA<PhysicalSpecies::electron>() && !pc.AmIA<PhysicalSpecies::positron>()
#ifdef WARPX_QED
        && !pc.has_quantum_sync() && !pc.has_breit_wheeler()
        && !pc.has_virtual_photons() && !pc.has_virtual_photons_beam_size_effect()
#endif
        ,
        "Native endpoint rate requires passive massive physical neutrals without subcycling");
    continue;
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!pc.do_not_deposit,"Startup oracle requires deposited charged species");
  physical->ValidateImplicitIonElectricWorkCapture();
  Real const qm=q/pc.getMass();
  for(WarpXParIter pti(pc,0);pti.isValid();++pti){
   AMREX_ALWAYS_ASSERT_WITH_MESSAGE(GetExternalEBField(pti).isNoOp(),"Native endpoint diagnostic does not support an accelerator lattice or external particle force");
   auto const pos=GetParticlePosition<PIdx>(pti);
   auto const* ux=pti.GetAttribs(PIdx::ux).dataPtr();auto const* uy=pti.GetAttribs(PIdx::uy).dataPtr();auto const* uz=pti.GetAttribs(PIdx::uz).dataPtr();auto const* w=pti.GetAttribs(PIdx::w).dataPtr();
   auto const ex=E[0]->const_array(pti),ey=E[1]->const_array(pti),ez=E[2]->const_array(pti),bx=B[0]->const_array(pti),by=B[1]->const_array(pti),bz=B[2]->const_array(pti);
   auto const et0=E[0]->ixType(),et1=E[1]->ixType(),et2=E[2]->ixType(),bt0=B[0]->ixType(),bt1=B[1]->ixType(),bt2=B[2]->ixType();
   auto const j0=current[0]->array(pti),j1=current[1]->array(pti),j2=current[2]->array(pti),r0=rate[0]->array(pti),r1=rate[1]->array(pti),r2=rate[2]->array(pti);
   amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long p){
    amrex::ParticleReal xp,yp,zp;pos(p,xp,yp,zp);
    amrex::ParticleReal ep0=0.,ep1=0.,ep2=0.,bp0=0.,bp1=0.,bp2=0.;
    auto const result=doGatherShapeNImplicit(xp,yp,zp,xp,yp,zp,ep0,ep1,ep2,bp0,bp1,bp2,ex,ey,ez,bx,by,bz,et0,et1,et2,bt0,bt1,bt2,inverse,origin,domain,crop,zero,1,N,CurrentDepositionAlgo::Direct);
    AMREX_ALWAYS_ASSERT(bool(result));
    Real const g=std::sqrt(1.+(ux[p]*ux[p]+uy[p]*uy[p]+uz[p]*uz[p])/(PhysConst::c*PhysConst::c));
    Real const vx=ux[p]/g,vy=uy[p]/g,vz=uz[p]/g;
    Real const fx=qm*(ep0+(linear_only?0.:vy*bp2-vz*bp1)),fy=qm*(ep1+(linear_only?0.:vz*bp0-vx*bp2)),fz=qm*(ep2+(linear_only?0.:vx*bp1-vy*bp0));
    Real const dot=(vx*fx+vy*fy+vz*fz)/(PhysConst::c*PhysConst::c);
    Real const ax=(fx-vx*dot)/g,ay=(fy-vy*dot)/g,az=(fz-vz*dot)/g;
    amrex::GpuArray<double,3>x{},v{},a{};double vt=0.,at=0.;
#if defined(WARPX_DIM_RZ)
    Real const rr=std::sqrt(xp*xp+yp*yp);AMREX_ALWAYS_ASSERT(rr>0.);
    Real const co=xp/rr,si=yp/rr,vr=co*vx+si*vy,veltheta=-si*vx+co*vy;
    x={(rr-origin.x)*inverse.x,0.,(zp-origin.z)*inverse.z};
    v={linear_only?0.:vr*inverse.x,0.,linear_only?0.:vz*inverse.z};
    a={(co*ax+si*ay+(linear_only?0.:veltheta*veltheta/rr))*inverse.x,0.,az*inverse.z};
    vt=linear_only?0.:veltheta;at=-si*ax+co*ay-(linear_only?0.:vr*veltheta/rr);
#else
    x={(xp-origin.x)*inverse.x,(yp-origin.y)*inverse.y,(zp-origin.z)*inverse.z};
    v={linear_only?0.:vx*inverse.x,linear_only?0.:vy*inverse.y,linear_only?0.:vz*inverse.z};
    a={ax*inverse.x,ay*inverse.y,az*inverse.z};
#endif
    int first[3]{};for(int d=0;d<3;++d)first[d]=int(x[d]+(N%2==0?.5:0.))-N/2;
    for(int k=first[2];k<=first[2]+N;++k)for(int j=(Dim==3?first[1]:0);j<=(Dim==3?first[1]+N:0);++j)for(int i=first[0];i<=first[0]+N;++i){
     amrex::GpuArray<double,3>value{},derivative{};
     AMREX_ALWAYS_ASSERT((warpx::particles::EsirkepovCurrentRate<N,Dim>(x,v,a,{i,j,k},vt,at,value,derivative)));
     int const jj=Dim==3?j:k,kk=Dim==3?k:0;
     Real const w0=q*w[p]*inverse.y*inverse.z,w2=q*w[p]*inverse.x*inverse.y,w1=q*w[p]*(Dim==3?inverse.x*inverse.z:inverse.x*inverse.y*inverse.z);
     // Flux prefix vanishes after N, so skip its absent final face.
     if(i<first[0]+N){amrex::Gpu::Atomic::AddNoRet(&j0(i,jj,kk),w0*value[0]);amrex::Gpu::Atomic::AddNoRet(&r0(i,jj,kk),w0*derivative[0]);}
     if(Dim!=3||j<first[1]+N){amrex::Gpu::Atomic::AddNoRet(&j1(i,jj,kk),w1*value[1]);amrex::Gpu::Atomic::AddNoRet(&r1(i,jj,kk),w1*derivative[1]);}
     if(k<first[2]+N){amrex::Gpu::Atomic::AddNoRet(&j2(i,jj,kk),w2*value[2]);amrex::Gpu::Atomic::AddNoRet(&r2(i,jj,kk),w2*derivative[2]);}
    }
   });
  }
 }
 for(auto const& f:{current,rate}){
#if defined(WARPX_DIM_RZ)
  sim.ApplyInverseVolumeScalingToCurrentDensity(f[0],f[1],f[2],0);
#endif
  for(auto* x:f) ablastr::utils::communication::SumBoundary(*x,0,1,x->nGrowVect(),x->nGrowVect(),WarpX::do_single_precision_comms,geom.periodicity());
  sim.ApplyJfieldBoundary(0,f[0],f[1],f[2],PatchType::fine);
  for(auto* x:f){x->OverrideSync(geom.periodicity());x->FillBoundary(geom.periodicity());}
 }
}
void ValidateNativeEndpointScope(WarpX& s, HybridPICModel const& model)
{
  warpx::thermal::ValidateNativeJouleWork(s);
  warpx::darwin::ValidateNativePECPlasma(s);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx::darwin::NativeCoilCurrentEnabled() ||
      warpx::darwin::NativePECPlasmaEnabled(),
      "Native coil-current qualification requires the guarded PEC plasma path");
  bool smooth_force=false;
  amrex::ParmParse("endpoint_diagnostic").query("smooth_force",smooth_force);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s.maxLevel()==0&&WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving&&!model.m_esolve_curlcurl&&model.m_electron_inertia_djedt_only&&!model.m_electron_inertia_bdf2&&model.m_electron_inertia_extrapolated_history&&!EB::enabled()&&(!model.HasResistivity()||smooth_force)&&!model.m_pec_conductor_wall_rows&&!model.m_holmstrom_vacuum_region&&!model.m_has_external_current&&model.m_n_floor_smooth_width==0.&&model.m_electron_inertia_floor_taper==0.&&!WarpX::use_filter&&WarpX::ncomps==1&&WarpX::nox==3,"Native initial-rate diagnostic guard: one level,m0,MC-gather,e-form,dJe-only,no filters/tapers/conductor closure/external current");
#if defined(WARPX_DIM_RZ)
  bool const axial_pmc=model.UseCompatibleYeeInertia() &&
      WarpX::field_boundary_lo[1]==FieldBoundaryType::PMC &&
      WarpX::field_boundary_hi[1]==FieldBoundaryType::PMC;
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      (s.Geom(0).isPeriodic(1)||axial_pmc) &&
      WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC,
      "Native endpoint requires radial PEC and periodic z, or positive direct-Yee inertia with axial PMC ends");
#else
  amrex::Abort("Native endpoint prototype is currently qualified only in RZ");
#endif
  // Current companion contract has no post-interval event map. These are
  // capability checks only; ordinary initial loading and read-only observers
  // remain allowed. Other callbacks must honor the no-state-mutation contract.
  amrex::Vector<std::string> collisions;
  amrex::ParmParse("collisions").queryarr("collision_names",collisions);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(collisions.empty() && !model.m_has_electron_stopping,
      "Native endpoint prototype does not cover outer collisions or electron stopping");
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!s.getdo_moving_window() && WarpX::gamma_boost==1. &&
      s.m_v_galilean[0]==0. && s.m_v_galilean[1]==0. && s.m_v_galilean[2]==0.,
      "Native endpoint prototype requires a fixed unboosted laboratory grid");
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!IsPythonCallbackInstalled("particleinjection") &&
      !IsPythonCallbackInstalled("particlescraper"),
      "Native endpoint prototype does not cover particle injection/scraping callbacks");
  for(auto const& name:s.GetPartContainer().GetSpeciesNames()) {
      auto& pc=s.GetPartContainer().GetParticleContainerFromName(name);
      auto const* physical=dynamic_cast<PhysicalParticleContainer const*>(&pc);
      int resampling=0;
      amrex::ParmParse(name).query("do_resampling",resampling);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(physical && !resampling &&
          !pc.doContinuousInjection() && physical->HybridSplitInterval()<=0,
          "Native endpoint prototype does not cover resampling, continuous injection or hybrid splitting");
      for(int i=0;;++i) {
          auto const* injector=pc.GetPlasmaInjector(i);
          if(!injector)break;
          AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!injector->doFluxInjection(),
              "Native endpoint prototype does not cover particle flux injection");
      }
  }
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(smooth_force ||
      (!model.m_include_joule_heating && !model.m_include_temperature_relaxation &&
       !model.m_include_electron_viscosity && !model.m_has_energy_sink),
      "Native endpoint prototype requires source-free model unless smooth_force is enabled");
  if (smooth_force) {
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(model.UsesEulerianElectronEnergy() &&
          !model.m_include_temperature_relaxation && !model.m_joule_redirect_to_ions &&
          !model.m_has_energy_sink && !model.m_has_per_species_eta && !model.m_esolve_tensor,
          "Native smooth force excludes OU/relaxation/redirect/sink/species/transformed closures");
      // Native Ohm evaluates eta at gett_new, which remains the step-start
      // clock during endpoint recovery. A separate stage-time API is needed
      // before a time-dependent coefficient can satisfy an endpoint equation.
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          !model.m_resistivity_parser->symbols().contains("t") &&
          (!model.m_include_joule_heating ||
           !model.m_heating_resistivity_parser->symbols().contains("t")),
          "Native smooth endpoint force requires time-independent eta and heating parsers");
  }
  for(int d=0;d<AMREX_SPACEDIM;++d)AMREX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::particle_boundary_lo[d]!=ParticleBoundaryType::Absorbing&&WarpX::particle_boundary_hi[d]!=ParticleBoundaryType::Absorbing,"Native endpoint prototype does not cover absorbing endpoint state splits");
}

void NegateFullAllocation(MF& field) {
 for(amrex::MFIter mfi(field);mfi.isValid();++mfi) {
  auto const a=field.array(mfi);
  amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=-a(i,j,k);});
 }
}
struct Startup {
 WarpX& sim;HybridPICModel& model;Real time;std::function<void()> curl_rate;
 View E,B,Ji,C;
 Vector B0,C0,EL0,F0,J0,Jrate,curl,raw,projected,zero;
 Vector nores, force_entry, pressure_hall;
 std::unique_ptr<warpx::thermal::EulerianDissipation> dissipation;
 bool smooth_force=false, force_ready=false, correct_force=false;
 bool recoverable=false;
 MF kappa,nodal,rate_node,density;
 Vector target,rate_edge,drive_reference,longitudinal_gather;
 std::unique_ptr<warpx::darwin::DarwinInitialRateSchur> projection;
 std::unique_ptr<warpx::darwin::NativeCircuitFieldRate> circuit_rate;
 std::unique_ptr<warpx::darwin::NativeCoilField> coil_current;
 WarpXSolverVec layout;
 Real omega2=0.;int actions=0,projection_iterations=0;
 Startup(WarpX& s,Real t,std::function<void()> callback,bool allow_decline=false):sim(s),model(*s.get_pointer_HybridPICModel()),time(t),curl_rate(std::move(callback)),E(s.m_fields.get_alldirs(FieldType::Efield_fp,0)),B(s.m_fields.get_alldirs(FieldType::Bfield_fp,0)),Ji(s.m_fields.get_alldirs(FieldType::current_fp,0)),C(s.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0)),B0(Clone(B)),C0(CurrentClone(C)),EL0(Clone(s.m_fields.get_alldirs("hybrid_E_long_fp",0))),F0(Allocate(E)),J0(Allocate(Ji)),Jrate(Allocate(Ji)),curl(CurrentAllocate(C)),raw(Allocate(E)),projected(Allocate(E)),zero(Allocate(E)),target(Allocate(E)),rate_edge(Allocate(E)),drive_reference(Allocate(E)){
  recoverable=allow_decline;
  ValidateNativeEndpointScope(s,model);
  if(warpx::darwin::NativePECPlasmaEnabled())pressure_hall=Allocate(E);
  if(warpx::darwin::NativeCoilCurrentEnabled()) {
   coil_current=std::make_unique<warpx::darwin::NativeCoilField>(sim);
   longitudinal_gather=Clone(Pointers(EL0));
   warpx::darwin::CompleteNativeElectricGatherImages(sim,Pointers(longitudinal_gather));
  }
  amrex::ParmParse("endpoint_diagnostic").query("smooth_force",smooth_force);
  if(smooth_force) {
      dissipation=warpx::darwin::MakeNativeInstantaneousDissipation(sim);
      nores=Allocate(E);force_entry=Allocate(E);
      correct_force=model.HasResistivity() && model.m_implicit_push_excludes_resistive_field;
  }
  auto const& supplied=*sim.m_fields.get(FieldType::rho_fp,0);
  density.define(supplied.boxArray(),supplied.DistributionMap(),1,supplied.nGrowVect());MF::Copy(density,supplied,0,0,1,supplied.nGrowVect());
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(density.min(0,0)>PhysConst::q_e*model.m_n_floor,"Native initial-rate diagnostic currently requires every native node above the vacuum threshold; recovery remains unmodified");
  kappa.define(density.boxArray(),density.DistributionMap(),1,1);nodal.define(density.boxArray(),density.DistributionMap(),3,1);rate_node.define(density.boxArray(),density.DistributionMap(),3,1);
  for(auto& f:zero)f.setVal(0.);for(auto& f:target)f.setVal(0.);
  if(NativeCircuitDriveEnabled()) {
    auto* subsystem=sim.get_pointer_CircuitCoupling();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(subsystem && subsystem->Coupler() &&
        subsystem->Coupler()->DeviceTrials() &&
        (!model.m_darwin_checkpoint_restored || subsystem->NativeRestartBindingValidated()),
        "Native endpoint circuit requires initialized device-affine coupling and validated restart binding");
    circuit_rate=std::make_unique<warpx::darwin::NativeCircuitFieldRate>(sim,*subsystem->Coupler());
    std::string error;
    bool const prepared=circuit_rate->Prepare(time,error);
    if(!prepared && recoverable) {
        amrex::Print()<<"Native candidate circuit rate declined: "<<error<<"\n";
        throw EndpointNumericalFailure{NativeEndpointFailure::CircuitRate,
            "Native candidate closed circuit rate is unavailable"};
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(prepared,error);
  } else for(int c=0;c<3;++c) {
    BuildDriveReference(sim,drive_reference[c],c,time,0.,DriveRate::Instantaneous);
    NegateFullAllocation(drive_reference[c]);
  }
  omega2=PhysConst::q_e*PhysConst::q_e*model.m_n0_ref/(model.m_electron_inertia_mass*PhysConst::epsilon_0);
  FreezeCapacity(density);
  warpx::darwin::InitialRateSchurOptions opts;using BC=warpx::darwin::InitialRateBoundary;
  for(int d=0;d<AMREX_SPACEDIM;++d){opts.lower[d]=s.Geom(0).isPeriodic(d)?BC::Periodic:WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC?BC::PEC:BC::PMC;opts.upper[d]=s.Geom(0).isPeriodic(d)?BC::Periodic:WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC?BC::PEC:BC::PMC;}
#if defined(WARPX_DIM_RZ)
  opts.lower[0]=BC::Axis;
#endif
  opts.compatible_yee=model.UseCompatibleYeeInertia();
  opts.max_semicoarsening_levels=model.m_darwin_poisson_semicoarsening;
  opts.semicoarsening_direction=model.m_darwin_poisson_semicoarsening_direction;
  opts.verbose=model.m_darwin_poisson_verbosity;
  opts.relative_tolerance=1.e-12;opts.absolute_tolerance=1.e-12;opts.max_iterations=1000;opts.restart_length=150;
  projection=std::make_unique<warpx::darwin::DarwinInitialRateSchur>(s.Geom(0),s.boxArray(0),s.DistributionMap(0),opts);EndpointRequire(projection->Freeze(kappa),NativeEndpointFailure::Projection,recoverable,"Native endpoint coefficient freeze failed");
  layout.Define(&s,"Efield_fp");
 }
 void FreezeCapacity(MF const& density){
  EndpointRequire(BuildNativeInertiaKappa(sim,density,0,kappa),
      NativeEndpointFailure::InvalidDensity,recoverable,
      "Native endpoint coefficient requires finite all-active density");
 }
 void PhysicalElectric(View const& out,bool linear) {
  AMREX_ALWAYS_ASSERT(coil_current && circuit_rate && circuit_rate->Ready());
  for(int c=0;c<3;++c) {
   MF::Copy(*out[c],circuit_rate->PotentialRate(c),0,0,1,out[c]->nGrowVect());
   NegateFullAllocation(*out[c]);
   if(!linear)MF::Add(*out[c],longitudinal_gather[c],0,0,1,out[c]->nGrowVect());
  }
 }
 void GatherImages(bool affine=true,bool circuit_images=false){
  if(coil_current && circuit_images){PhysicalElectric(E,false);return;}
  sim.FillBoundaryE(E[0]->nGrowVect(),true);
  sim.ApplyEfieldBoundary(0,PatchType::fine,time);
  amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
  for(int d=0;d<AMREX_SPACEDIM;++d){lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
  // The circuit reference includes a homogeneous E-dependent response.
  // Deposit consumes the reference prepared for its exact transverse input;
  // algebra/output vectors keep their original homogeneous valid-node map.
  for(int c=0;c<3;++c)ApplyDarwinPMCVectorBoundary(*E[c],sim.Geom(0),lo,hi,
      circuit_images && circuit_rate ? &circuit_rate->ElectricReference(c) :
      affine&&NativePrescribedDriveEnabled()?&drive_reference[c]:nullptr);
 }
 void CurlRate(bool affine){
  if(circuit_rate) {
    circuit_rate->Apply(E,affine);
    for(int c=0;c<3;++c) {
      MF::Copy(*B[c],circuit_rate->MagneticRate(c),0,0,1,B[c]->nGrowVect());
      MF::Copy(*C[c],circuit_rate->CurrentRate(c),0,0,1,C[c]->nGrowVect());
    }
    if(coil_current) {
      coil_current->SampleRates(sim.get_pointer_CircuitCoupling()->Coupler()->EndpointFieldRates());
      coil_current->SubtractCurrent(C);
    }
  } else if(NativePrescribedDriveEnabled())NativeInstantaneousCurlRate(sim,time,affine);
  else curl_rate();
 }
 void Deposit(bool linear){
  Copy(B,Pointers(B0));
  // The first point-current deposit precedes assembling the frozen Ohm state.
  // Homogeneous electric responses never include its affine removed force.
  bool const corrected=force_ready && correct_force && !linear;
  if(corrected){Copy(Pointers(force_entry),E);warpx::darwin::ApplyNativeInstantaneousForce(E,Pointers(F0),Pointers(nores),true);}
  if(coil_current)PhysicalElectric(E,linear);else GatherImages(!linear,bool(circuit_rate));
  sim.UpdateAuxiliaryData();sim.FillBoundaryAux(sim.getngUpdateAux());
  DepositRate<3>(sim,sim.m_fields.get_alldirs(FieldType::Efield_aux,0),sim.m_fields.get_alldirs(FieldType::Bfield_aux,0),Pointers(J0),Pointers(Jrate),linear);
  if(corrected)Copy(E,Pointers(force_entry));
 }
 void Node(View const& v,MF& out){using ablastr::coarsen::sample::Interp;amrex::GpuArray<int,3>const node{1,1,1},ratio{1,1,1};for(int d=0;d<3;++d){auto const type=d==0?model.Jx_IndexType:d==1?model.Jy_IndexType:model.Jz_IndexType;for(amrex::MFIter mfi(out);mfi.isValid();++mfi){auto const a=out.array(mfi);auto const b=v[d]->const_array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k,d)=Interp(b,type,node,ratio,i,j,k,0);});}}Images(out);}
 void MapNode(MF const& in,View const& out,bool capacity){using ablastr::coarsen::sample::Interp;amrex::GpuArray<int,3>const node{1,1,1},ratio{1,1,1};MF weighted(in.boxArray(),in.DistributionMap(),3,1);Real const scalar=1./(PhysConst::epsilon_0*omega2);for(amrex::MFIter mfi(in);mfi.isValid();++mfi){auto const a=weighted.array(mfi);auto const b=in.const_array(mfi),kap=kappa.const_array(mfi);amrex::ParallelFor(mfi.validbox(),3,[=] AMREX_GPU_DEVICE(int i,int j,int k,int d){a(i,j,k,d)=b(i,j,k,d)*(capacity?kap(i,j,k)*scalar:1.);});}Images(weighted);for(int d=0;d<3;++d){auto const type=d==0?model.Ex_IndexType:d==1?model.Ey_IndexType:model.Ez_IndexType;for(amrex::MFIter mfi(*out[d]);mfi.isValid();++mfi){auto const a=out[d]->array(mfi);auto const b=weighted.const_array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=Interp(b,node,type,ratio,i,j,k,d);});}Images(*out[d]);}}
 void MapYee(View const& in, View const& out, Real scale) {
  warpx::darwin::ApplyYeeInertiaMass(sim.Geom(0),kappa,
      {in[0],in[1],in[2]},out,scale);
  for(auto* f:out) PMCImages(*f);
 }
 void RatesToRaw() {
  if(model.UseCompatibleYeeInertia()) {
   for(int c=0;c<3;++c){MF::Subtract(curl[c],Jrate[c],0,0,1,0);Images(curl[c]);}
   if(warpx::darwin::NativePECPlasmaEnabled())warpx::darwin::SetNativePECTangential(sim,Pointers(curl));
   MapYee(Pointers(curl),Pointers(raw),1./(PhysConst::epsilon_0*omega2));
  } else {
   Node(Pointers(curl),nodal);Node(Pointers(Jrate),rate_node);
   MF::Subtract(nodal,rate_node,0,0,3,0);Images(nodal);MapNode(nodal,Pointers(raw),true);
  }
 }
 void Project(View const& input,View const& output,bool affine=false) {
  auto const result=projection->Correct({input[0],input[1],input[2]},
      affine?std::array<MF const*,3>{&target[0],&target[1],&target[2]}
            :std::array<MF const*,3>{&zero[0],&zero[1],&zero[2]});
  projection_iterations+=result.iterations;
  EndpointRequire(result.converged,NativeEndpointFailure::Projection,recoverable,"Native initial-rate weighted projection failed");
  auto const grad=projection->CorrectionField();
  if(model.UseCompatibleYeeInertia()) {
   warpx::darwin::ApplyYeeInertiaMass(sim.Geom(0),kappa,grad,output);
   for(int c=0;c<3;++c){MF::Copy(rate_edge[c],*grad[c],0,0,1,1);
       rate_edge[c].mult(PhysConst::epsilon_0*omega2);}
  } else {
   Node({const_cast<MF*>(grad[0]),const_cast<MF*>(grad[1]),const_cast<MF*>(grad[2])},rate_node);
   rate_node.mult(PhysConst::epsilon_0*omega2);MapNode(rate_node,output,true);
  }
  for(int d=0;d<3;++d) {
   MF::LinComb(*output[d],1.,*input[d],0,-1.,*output[d],0,0,1,0);
   PMCImages(*output[d]);
  }
 }
 void Response(WarpXSolverVec& out,WarpXSolverVec const& in){++actions;Copy(E,in.getArrayVec()[0],0);GatherImages(false);CurlRate(false);Copy(Pointers(curl),C);Copy(E,in.getArrayVec()[0],0);Deposit(true);RatesToRaw();Project(Pointers(raw),Pointers(projected));Copy(E,Pointers(projected),0);GatherImages(false);for(int d=0;d<3;++d)MF::Copy(*out.getArrayVec()[0][d],*E[d],0,0,1,0);}
 struct Ops{
  using RT=Real;Startup& s;
  WarpXSolverVec makeVecLHS()const{WarpXSolverVec x;x.Define(s.layout);return x;}WarpXSolverVec makeVecRHS()const{return makeVecLHS();}
  void setToZero(WarpXSolverVec&x)const{x.zero();}void assign(WarpXSolverVec&x,WarpXSolverVec const&y)const{x.Copy(y);}Real norm2(WarpXSolverVec const&x)const{return x.norm2();}Real dotProduct(WarpXSolverVec const&x,WarpXSolverVec const&y)const{return x.dotProduct(y);}void scale(WarpXSolverVec&x,Real a)const{x.scale(a);}void increment(WarpXSolverVec&x,WarpXSolverVec const&y,Real a)const{x.increment(y,a);}void linComb(WarpXSolverVec&z,Real a,WarpXSolverVec const&x,Real b,WarpXSolverVec const&y)const{z.linComb(a,x,b,y);}void precond(WarpXSolverVec&x,WarpXSolverVec const&b)const{x.Copy(b);}void apply(WarpXSolverVec&out,WarpXSolverVec const&in){s.Response(out,in);out.linComb(1.,in,-1.,out);}
 };
};
}

namespace warpx::particles {
void EvaluateNativeInstantaneousIonRate(WarpX& sim,View const& electric_aux,
    View const& magnetic_aux,View const& current,View const& rate,bool linear_only)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxLevel()==0 && WarpX::nox==3 && WarpX::ncomps==1 &&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving,
        "Native instantaneous rate adapter requires one level, shape3, m0 and native MC gather");
    auto const native_e=sim.m_fields.get_alldirs(FieldType::Efield_aux,0);
    auto const native_b=sim.m_fields.get_alldirs(FieldType::Bfield_aux,0);
    auto const native_j=sim.m_fields.get_alldirs(FieldType::current_fp,0);
    for(int c=0;c<3;++c) {
        for(auto pair:{std::pair{electric_aux[c],native_e[c]},
                       std::pair{magnetic_aux[c],native_b[c]},
                       std::pair{current[c],native_j[c]},std::pair{rate[c],native_j[c]}}) {
            AMREX_ALWAYS_ASSERT(pair.first && pair.first->boxArray()==pair.second->boxArray() &&
                pair.first->DistributionMap()==pair.second->DistributionMap() &&
                pair.first->nComp()==1 && pair.first->nGrowVect().allGE(pair.second->nGrowVect()));
        }
    }
    std::array<MF*,6> outputs{current[0],current[1],current[2],rate[0],rate[1],rate[2]};
    for(std::size_t i=0;i<outputs.size();++i) {
        for(int c=0;c<3;++c)
            AMREX_ALWAYS_ASSERT(outputs[i]!=electric_aux[c] && outputs[i]!=magnetic_aux[c]);
        for(std::size_t j=0;j<i;++j) AMREX_ALWAYS_ASSERT(outputs[i]!=outputs[j]);
    }
    DepositRate<3>(sim,electric_aux,magnetic_aux,current,rate,linear_only);
}
}

namespace {
template<class T> std::uint64_t AddBytes(std::uint64_t h,T const* ptr,std::size_t count){std::vector<T> host(count);amrex::Gpu::copy(amrex::Gpu::deviceToHost,ptr,ptr+count,host.begin());for(auto b:std::span<unsigned char const>(reinterpret_cast<unsigned char const*>(host.data()),count*sizeof(T)))h=(h^b)*1099511628211ULL;return h;}
std::map<std::string,std::uint64_t> FieldHashes(WarpX& sim){std::map<std::string,std::uint64_t> result;amrex::Gpu::streamSynchronize();for(auto const&name:sim.m_fields.list()){auto const& f=*sim.m_fields.internal_get(name);std::uint64_t h=1469598103934665603ULL;for(amrex::MFIter mfi(f);mfi.isValid();++mfi)h=AddBytes(h,f[mfi].dataPtr(),f[mfi].size());result[name]=h;}return result;}
std::uint64_t ParticleHash(WarpX& sim){std::uint64_t h=1469598103934665603ULL;for(auto const&name:sim.GetPartContainer().GetSpeciesNames()){auto&pc=sim.GetPartContainer().GetParticleContainerFromName(name);for(WarpXParIter pti(pc,0);pti.isValid();++pti)for(auto const&data:pti.GetAttribs())h=AddBytes(h,data.dataPtr(),data.size());}return h;}
Real ElectricEnergy(WarpX&sim,View const&field){Real result=0.;auto&model=*sim.get_pointer_HybridPICModel();for(auto*f:field){MF energy(f->boxArray(),f->DistributionMap(),1,0);for(amrex::MFIter mfi(energy);mfi.isValid();++mfi){auto const a=energy.array(mfi);auto const e=f->const_array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=.5*PhysConst::epsilon_0*e(i,j,k)*e(i,j,k);});}result+=model.EnergyVolumeIntegral(energy,0,0);}return result;}
constexpr char const* current_name="diagnostic_Je_endpoint_fp";
constexpr char const* stage_name="diagnostic_Je_stage_fp";
constexpr char const* ion_old_name="diagnostic_Ji_old_instantaneous_fp";
constexpr char const* ion_virtual_name="diagnostic_Ji_virtual_instantaneous_fp";
constexpr char const* inertia_stage_name="diagnostic_Je_inertia_stage_fp";
constexpr char const* inertia_node_name="diagnostic_Je_inertia_stage_nodal";
constexpr char const* delta_base_name="diagnostic_Ji_increment_base_fp";
constexpr char const* delta_ion_name="diagnostic_delta_Ji_fp";
constexpr char const* curl_rate_name="diagnostic_Cdot_fp";
constexpr char const* drive_curvature_name="diagnostic_C_drive_curvature_fp";
constexpr char const* rate_a_name="diagnostic_Adot_fp";
constexpr char const* rate_b_name="diagnostic_Bdot_fp";
constexpr char const* displacement_old_name="diagnostic_D_endpoint_fp";
constexpr char const* displacement_stage_name="diagnostic_D_stage_fp";
constexpr char const* rate_yee_name="diagnostic_dJe_fp";
constexpr char const* rate_node_name="diagnostic_dJe_nodal";

void EnsureCompanion(WarpX& sim) {
    auto& fields=sim.m_fields;
    if(sim.get_pointer_HybridPICModel()->UseCompatibleYeeInertia()) {
        AMREX_ALWAYS_ASSERT(fields.has("diagnostic_inertia_kappa_nodal",0));
        for(int c=0;c<3;++c)
            AMREX_ALWAYS_ASSERT(fields.has("hybrid_E_inertial_fp",Direction{c},0));
    }
    auto const ji=fields.get_alldirs(FieldType::current_fp,0);
    bool const pec=warpx::darwin::NativePECPlasmaEnabled();
    warpx::darwin::AllocateNativePECPlasma(sim);
    if(NativePrescribedDriveEnabled()||NativeCircuitDriveEnabled())for(int c=0;c<3;++c)
        if(!fields.has(drive_curvature_name,Direction{c},0))
            fields.alloc_init(drive_curvature_name,Direction{c},0,ji[c]->boxArray(),
                ji[c]->DistributionMap(),1,ji[c]->nGrowVect(),0.);
    for(auto name:{current_name,stage_name,ion_old_name,ion_virtual_name,inertia_stage_name})
        for(int c=0;c<3;++c) if(!fields.has(name,Direction{c},0))
            fields.alloc_init(name,Direction{c},0,ji[c]->boxArray(),
                ji[c]->DistributionMap(),1,pec && (name==current_name || name==stage_name || name==inertia_stage_name) ? amrex::IntVect(1) : ji[c]->nGrowVect(),0.,
                true,true,name==current_name);
    if(!fields.has(inertia_node_name,0)) {
        auto const& ei=*fields.get("hybrid_E_inertial_nodal",0);
        fields.alloc_init(inertia_node_name,0,ei.boxArray(),ei.DistributionMap(),3,amrex::IntVect(1),0.);
    }
    if(NativeCorrelatedIncrementEnabled()) {
        for(auto name:{delta_base_name,delta_ion_name,curl_rate_name,
                       displacement_old_name,displacement_stage_name,rate_yee_name})
            for(int c=0;c<3;++c)if(!fields.has(name,Direction{c},0))
                fields.alloc_init(name,Direction{c},0,ji[c]->boxArray(),
                    ji[c]->DistributionMap(),1,pec && (name==curl_rate_name || name==displacement_old_name || name==displacement_stage_name || name==rate_yee_name) ? amrex::IntVect(1) : ji[c]->nGrowVect(),0.,
                    true,true,name==displacement_old_name);
        auto const A=fields.get_alldirs("hybrid_A_fp",0);
        for(int c=0;c<3;++c)if(!fields.has(rate_a_name,Direction{c},0))
            fields.alloc_init(rate_a_name,Direction{c},0,A[c]->boxArray(),
                A[c]->DistributionMap(),1,A[c]->nGrowVect(),0.);
        auto const B=fields.get_alldirs(FieldType::Bfield_fp,0);
        for(int c=0;c<3;++c)if(!fields.has(rate_b_name,Direction{c},0))
            fields.alloc_init(rate_b_name,Direction{c},0,B[c]->boxArray(),
                B[c]->DistributionMap(),1,B[c]->nGrowVect(),0.);
        if(!fields.has(rate_node_name,0)) {
            auto const& ei=*fields.get("hybrid_E_inertial_nodal",0);
            fields.alloc_init(rate_node_name,0,ei.boxArray(),ei.DistributionMap(),3,amrex::IntVect(1),0.);
        }
    }
}
void NodeCurrent(WarpX& sim,View const& v,MF& out){auto& model=*sim.get_pointer_HybridPICModel();using ablastr::coarsen::sample::Interp;amrex::GpuArray<int,3>const node{1,1,1},ratio{1,1,1};for(int c=0;c<3;++c){auto const type=c==0?model.Jx_IndexType:c==1?model.Jy_IndexType:model.Jz_IndexType;for(amrex::MFIter mfi(out);mfi.isValid();++mfi){auto const a=out.array(mfi);auto const b=v[c]->const_array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k,c)=Interp(b,type,node,ratio,i,j,k,0);});}}Images(out);}
Real CompanionError(WarpX&sim,View const&v,char const* refname,bool recoverable=false){auto const& ref=*sim.m_fields.get(refname,0);MF node(ref.boxArray(),ref.DistributionMap(),3,1);NodeCurrent(sim,v,node);MF::Subtract(node,ref,0,0,3,0);Real error=0.,scale=0.;for(int c=0;c<3;++c){error=std::max(error,node.norminf(c));scale=std::max(scale,ref.norminf(c));}EndpointRequire(std::isfinite(error)&&error<=1e-11*std::max(1.,scale),NativeEndpointFailure::Companion,recoverable,"Native Yee companion does not match the unchanged nodal theta Je history");return error/std::max(1.,scale);}
}
bool FillNativeYeeSchurKappa(WarpX& sim,MF const& rho,MF& kappa,Real interval) {
    AMREX_ALWAYS_ASSERT(sim.get_pointer_HybridPICModel()->UseCompatibleYeeInertia() &&
                        interval>0. && std::isfinite(interval));
    if(!BuildNativeInertiaKappa(sim,rho,rho.nComp()/2,kappa)) return false;
    auto const& model=*sim.get_pointer_HybridPICModel();
    kappa.mult(PhysConst::epsilon_0*model.m_electron_inertia_mass/
        (PhysConst::q_e*PhysConst::q_e*model.m_n0_ref*interval*interval));
    return kappa.is_finite(0,1,0);
}
bool NativePrescribedDriveEnabled(){
    bool enabled=false;amrex::ParmParse("endpoint_diagnostic").query("prescribed_drive",enabled);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!enabled ||
        WarpX::GetInstance().get_pointer_HybridPICModel()->UseCompatibleYeeInertia(),
        "Prescribed drive requires compatible Yee correlated endpoint inertia");
    return enabled;
}
bool NativeCircuitDriveEnabled(){
    bool enabled=false;amrex::ParmParse("endpoint_diagnostic").query("native_circuit",enabled);
    if(enabled) {
        std::string driver,backend;
        amrex::ParmParse("implicit_evolve").query("circuit_driver",driver);
        amrex::ParmParse("circuit").query("trial_backend",backend);
        auto const& model=*WarpX::GetInstance().get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(driver=="native" && backend=="device_affine" &&
            model.UseCompatibleYeeInertia() && NativeCorrelatedIncrementEnabled() &&
            !NativePrescribedDriveEnabled() &&
            (!NativeVacuumEndpointEnabled() || NativeFullOhmLongitudinalEnabled()),
            "Native endpoint circuit requires explicit native device-affine driver, "
            "correlated Yee inertia and full-Ohm projection for joined vacuum");
    }
    return enabled;
}
void ValidateNativeCircuitEndpointRestart(WarpX& sim,Real time)
{
    if(!NativeEndpointEnabled() || !NativeCircuitDriveEnabled() ||
       !sim.get_pointer_HybridPICModel()->m_darwin_checkpoint_restored) { return; }
    auto* subsystem=sim.get_pointer_CircuitCoupling();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativeEndpointRestartSupported() && subsystem &&
        subsystem->NativeRestartBindingValidated() && subsystem->Coupler() &&
        subsystem->Coupler()->DeviceTrials() && subsystem->Coupler()->SupportsNativeEndpointRate(),
        "Native circuit endpoint restart requires collective model binding and restored device-rate capability");
    // The query above is deliberately late: InitData has attached/restored the
    // provider. This new generation is derived from its closed accepted state;
    // neither a serialized device map nor a finite-interval secant is reused.
    warpx::darwin::NativeCircuitFieldRate rate(sim,*subsystem->Coupler());
    std::string error;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rate.Prepare(time,error),error);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rate.Ready(),
        "Native circuit endpoint restart rate generation is stale");
    if (NativeVacuumEndpointEnabled()) ValidateNativeVacuumRestart(sim);
    amrex::Print()<<"restart: native circuit endpoint closed rate validated at t="
                  <<time<<"; accepted fields and companions unchanged\n";
}
bool NativeEndpointEnabled(){bool enabled=false;amrex::ParmParse("endpoint_diagnostic").query("enable",enabled);return enabled;}
bool NativeIonQuadratureEnabled() {
    bool enabled=false;
    amrex::ParmParse("endpoint_diagnostic").query("ion_quadrature",enabled);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!enabled || NativeEndpointEnabled(),
        "Native ion quadrature requires the constrained initialization and Yee history companion");
    return enabled;
}
bool NativeCorrelatedIncrementEnabled() {
    bool enabled=false;amrex::ParmParse("endpoint_diagnostic").query("correlated_increment",enabled);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!enabled||NativeIonQuadratureEnabled(),
        "Correlated arithmetic requires the guarded live endpoint-ion-trapezoid inertia");
    return enabled;
}
bool TryNativeAdjacentInstantaneousPolicy(WarpX& sim,bool& enabled,bool direct_deposit)
{
    enabled=false;
    int selection=0;
    amrex::ParmParse endpoint("endpoint_diagnostic");
    endpoint.query("adjacent_instantaneous_current",selection);
    bool const requested=selection==1;
    bool valid=selection==0 || requested;
    if(requested) {
#if !defined(WARPX_DIM_RZ) || defined(__FAST_MATH__) ||     (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
        valid=false;
#else
        bool native=false,quadrature=false,correlated=false,drive=false,circuit=false,paired=false;
        endpoint.query("enable",native);endpoint.query("ion_quadrature",quadrature);
        endpoint.query("correlated_increment",correlated);endpoint.query("prescribed_drive",drive);
        endpoint.query("native_circuit",circuit);endpoint.query("positive_paired_increment",paired);
        bool mm_jacobian=false,mm_pc=false;
        amrex::ParmParse implicit("implicit_evolve");
        implicit.query("use_mass_matrices_jacobian",mm_jacobian);
        implicit.query("use_mass_matrices_pc",mm_pc);
        auto const* model=sim.get_pointer_HybridPICModel();
        auto const& geometry=sim.Geom(0);
        bool const axis=geometry.ProbLo(0)==0. && WarpX::field_boundary_lo[0]==FieldBoundaryType::None;
        // Selected FULL maps keep the native Adjacent deposit. A linear MM
        // map consumes its separately validated response/lease below; this
        // policy grants no cached-current or source publication permission.
        bool const paired_mm=NativeEndpointPairMassMatrixOwnerLocal(sim)!=nullptr;
        valid=valid && sizeof(amrex::Real)==sizeof(double) &&
            sizeof(amrex::ParticleReal)==sizeof(double) && native && quadrature && correlated &&
            ((direct_deposit && !mm_jacobian && !mm_pc) || paired_mm) &&
            !drive && !circuit && !paired &&
            model && model->UseCompatibleYeeInertia() && (!model->m_darwin_checkpoint_restored ||
                warpx::darwin::NativeEndpointPairRestartContext(sim)) &&
            sim.maxLevel()==0 && !EB::enabled() && !sim.getdo_moving_window() &&
            WarpX::grid_type==GridType::Staggered && WarpX::nox==3 && WarpX::ncomps==1 &&
            !WarpX::use_filter && !WarpX::do_shared_mem_current_deposition &&
            WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov &&
            geometry.isPeriodic(1) &&
            (axis || WarpX::field_boundary_lo[0]==FieldBoundaryType::PEC) &&
            WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC;
        for(auto x:sim.m_v_galilean)valid=valid && x==0.;
        // ShiftGalileanBoundary does not produce a physical y shift in RZ.
        // Its inactive Cartesian slot is not part of this geometry contract.
        valid=valid && sim.m_galilean_shift[0]==0. && sim.m_galilean_shift[2]==0.;
#if defined(AMREX_USE_GPU)
        valid=valid && warpx::darwin::NativeEndpointCudaQualificationSelected(sim);
#endif
#endif
    }
    // Both selection bits survive only for a unanimous selection. No local
    // predicate may bypass this collective, including the compile capability.
    int agreement[3]{valid?1:0,requested?1:0,requested?0:1};
    amrex::ParallelDescriptor::ReduceIntMin(agreement,3);
    if(agreement[0]==0 || agreement[1]+agreement[2]!=1)return false;
    enabled=agreement[1]!=0;
    return true;
}
bool NativeFullOhmLongitudinalEnabled()
{
    bool enabled = false;
    amrex::ParmParse("endpoint_diagnostic").query("full_ohm_longitudinal", enabled);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!enabled ||
        (NativeCorrelatedIncrementEnabled() &&
         WarpX::GetInstance().get_pointer_HybridPICModel()->UseCompatibleYeeInertia()),
        "Full Ohm longitudinal projection requires native correlated Yee inertia");
    if (enabled) {
        auto const& model = *WarpX::GetInstance().get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!model.m_esolve_tensor && !model.m_esolve_curlcurl,
            "Full Ohm longitudinal projection requires the algebraic native E-form solve");
        bool publish = true, segregated = false;
        amrex::ParmParse("endpoint_diagnostic").query("publish_field", publish);
        amrex::ParmParse("implicit_evolve").query("darwin_segregated_solve", segregated);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(publish && segregated,
            "Full Ohm longitudinal projection requires published endpoint fields "
            "and segregated constraint solves, including on restart");
    }
    return enabled;
}
namespace {
// This metadata-only check also runs while checkpoint parsers are initialized,
// before CircuitCoupling::InitData attaches the actual native provider.
bool ConfiguredCircuitField(std::string const& field) {
    amrex::ParmParse pp("circuit");amrex::Vector<std::string> names;pp.queryarr("coils",names);
    for(auto const& name:names) {
        std::string mapped=name;pp.query(name+".field_name",mapped);
        if(mapped==field)return true;
    }
    return false;
}
void ValidateCorrelatedStaticDrive(WarpX& sim) {
    auto const& model=*sim.get_pointer_HybridPICModel();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!model.m_has_external_current&&!EB::enabled(),
        "Correlated curl rate does not support external current or EB");
    bool const circuit=NativeCircuitDriveEnabled();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!circuit || model.m_add_external_fields,
        "Native endpoint circuit requires configured external coil fields");
    if(model.m_add_external_fields){
        auto const& ext=*model.m_external_vector_potential;
        amrex::ParmParse pp("external_vector_potential");
        for(int f=0;f<ext.nFields();++f){
            if(circuit && ConfiguredCircuitField(ext.FieldName(f))) {
                // Native provider owns this segment; python_scale is the
                // legacy storage name, not permission to execute a callback.
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ext.UsesPythonScale(f),
                    "Native circuit field requires native-owned segment storage");
                continue;
            }
            std::string expression="1.0";
            pp.query(ext.FieldName(f)+".A_time_external_function(t)",expression);
            amrex::Parser parser(expression);
            bool const analytic=NativePrescribedDriveEnabled();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!ext.DeviceDriven(f)&&!ext.UsesPythonScale(f)&&
                (analytic ? ext.HasAnalyticTimeProfile(f) :
                 (!parser.symbols().contains("t") || (circuit&&ext.HasAnalyticTimeProfile(f)))),
                "Correlated curl rate requires static or analytic prescribed fields, "
                "or the explicitly selected native circuit field map");
        }
    }
}
}
bool NativeEndpointRestartSupported()
{
    return NativeEndpointEnabled() && NativeIonQuadratureEnabled() &&
        NativeCorrelatedIncrementEnabled() &&
        WarpX::GetInstance().get_pointer_HybridPICModel()->UseCompatibleYeeInertia();
}

void AllocateNativeEndpointStorage(WarpX& sim)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativeEndpointRestartSupported(),
        "Native endpoint restart requires direct Yee correlated current convention");
    EnsureCompanion(sim);
    AllocateNativeVacuumSupportStorage(sim);
    warpx::darwin::increment::Allocate(sim);
}

void ValidateNativeEndpointCapabilities(WarpX& sim)
{
    ValidateNativeEndpointScope(sim,*sim.get_pointer_HybridPICModel());
    if(NativeCorrelatedIncrementEnabled()) { ValidateCorrelatedStaticDrive(sim); }
}

std::function<void()> NativeEndpointRestartStorageValidator(
    WarpX& sim,std::string const& checkpoint_prefix)
{
    struct Layout {
        std::string name;
        int direction;
        amrex::BoxArray boxes;
        amrex::DistributionMapping owners;
        amrex::IntVect ghosts;
        int components;
    };
    std::vector<Layout> expected;
    auto remember=[&](char const* name,int direction) {
        auto const* field=direction<0 ? sim.m_fields.get(name,0) :
            sim.m_fields.get(name,Direction{direction},0);
        auto const registry_name=direction<0 ? sim.m_fields.mf_name(name,0) :
            sim.m_fields.mf_name(name,Direction{direction},0);
        auto const file_name=checkpoint_prefix+registry_name;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::VisMF::Exist(file_name),
            "Native endpoint checkpoint missing required field: "+registry_name);
        amrex::Vector<char> bytes;
        amrex::VisMF::ReadFAHeader(file_name,bytes);
        std::istringstream input(std::string(bytes.data()));
        amrex::VisMF::Header saved;
        input>>saved;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bool(input),
            "Native endpoint checkpoint header is invalid: "+registry_name);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(saved.m_ba==field->boxArray(),
            "Native endpoint checkpoint BoxArray mismatch: "+registry_name);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(saved.m_ncomp==field->nComp(),
            "Native endpoint checkpoint component count mismatch: "+registry_name);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(saved.m_ngrow==field->nGrowVect(),
            "Native endpoint checkpoint ghost width mismatch: "+registry_name);
        expected.push_back({name,direction,field->boxArray(),field->DistributionMap(),
                            field->nGrowVect(),field->nComp()});
    };
    for(auto const* name:{current_name,displacement_old_name,"hybrid_E_inertial_fp"}) {
        for(int c=0;c<3;++c) { remember(name,c); }
    }
    if(warpx::darwin::NativePECPlasmaEnabled())for(int c=0;c<3;++c)remember(warpx::darwin::PECWallCurrentName,c);
    for(auto const* name:{"hybrid_Je_n_nodal","hybrid_Je_nm1_nodal",
                         "hybrid_Je_theta_nodal","hybrid_E_inertial_nodal"}) {
        remember(name,-1);
    }
    if(NativeVacuumEndpointEnabled()) {
        remember(NativeVacuumSupportRecordName(),-1);
        remember("hybrid_phi_darwin_fp",-1);
        if (NativeCircuitDriveEnabled())
            for (int c=0;c<3;++c) remember(NativeVacuumTransverseRecordName(),c);
    }
    return [&sim,expected=std::move(expected)]() {
        for(auto const& shape:expected) {
            auto const* field=shape.direction<0 ? sim.m_fields.get(shape.name,0) :
                sim.m_fields.get(shape.name,Direction{shape.direction},0);
            auto const label=shape.name+" direction="+std::to_string(shape.direction);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field->boxArray()==shape.boxes,
                "Native endpoint checkpoint BoxArray mismatch: "+label);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field->DistributionMap()==shape.owners,
                "Native endpoint checkpoint DistributionMapping mismatch: "+label);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field->nComp()==shape.components,
                "Native endpoint checkpoint component count mismatch: "+label);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(field->nGrowVect()==shape.ghosts,
                "Native endpoint checkpoint ghost width mismatch: "+label);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                field->is_finite(0,shape.components,shape.ghosts),
                "Native endpoint checkpoint contains nonfinite values: "+label);
        }
        amrex::Print()<<"restart: native endpoint storage validated before solver construction\n";
    };
}

std::string NativeEndpointCheckpointConfiguration(WarpX& sim)
{
    auto const& model=*sim.get_pointer_HybridPICModel();
    // Solver modes may switch; physical boundary/drive data must not. Serialize
    // expressions and the resolved values of their named constants, so changing
    // a constant behind an unchanged expression cannot change a restarted drive.
    std::ostringstream out;
    bool joint_vacuum=false;
    amrex::ParmParse("implicit_evolve.thermal").query("joint_vacuum",joint_vacuum);
    if(joint_vacuum)out << " joint_vacuum_stage_v1 raw_support_fixed full_particle_joint_phi ";
    if(warpx::darwin::NativePairedDarwinFields::EndpointRequested()){
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(warpx::darwin::NativePairedDarwinFields::EndpointAcceptanceRequested(),
            "Pending-only endpoint pairs cannot be checkpointed");
        out<<" retained_endpoint_numeric_v3 all_particle_attributes private_ET_history accepted_rho_Ji_Te_Pe per_rank_next_id ";
    }
    if(NativeVacuumEndpointEnabled()) {
        std::string policy;
        amrex::ParmParse("endpoint_diagnostic").query("vacuum_edge_policy",policy);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(policy=="native_edge_candidate",
            "Native vacuum checkpoint requires explicit native_edge_candidate support policy");
        out << " joined_vacuum_candidate_v3 support_record_v1 phi_endpoint_theta_v1 "
            << policy << " interval_mask_saved_entry_rebuild ";
        bool smooth = false;
        amrex::ParmParse("endpoint_diagnostic").query("smooth_force", smooth);
        if (smooth) {
            out << " joined_vacuum_native_force_affine_v1 ";
        }
    }
    out<<std::setprecision(std::numeric_limits<Real>::max_digits10)
       <<"native_boundary_drive_source_v2";
    if(warpx::darwin::NativePECPlasmaEnabled())out<<" pec_plasma_constitutive_pressure_hall_v1 independent_Je_D_W_grow1";
    if(warpx::darwin::NativeCoilCurrentEnabled())out<<" native_coil_Cp_Ctotal_minus_Jcoil_v1 explicit_stage_clock device_rate_secant driven_PEC_E_v1";
    if(warpx::darwin::NativePairedDarwinFields::Requested())
        out<<" positive_paired_finite_stage_v1 high_endpoint_materialization_v1 fresh_start_only";
    if (NativeFullOhmLongitudinalEnabled()) {
        out << " full_ohm_longitudinal_v2";
    }
    if(NativeCircuitDriveEnabled()) {
        out << " native_circuit_endpoint_v1 closed_rate_clock_v1 device_interval_secant_v1";
        if (NativeVacuumEndpointEnabled()) out << " joined_vacuum_circuit_closed_response_v2 accepted_transverse_origin_v1";
    }
    if (warpx::thermal::NativeJouleWorkEnabled()) {
        out << " joule_applied_edge_work_v1";
        if (warpx::thermal::NativeVacuumJouleErrorEnabled()) {
            out << " vacuum_joule_projection_error_rz12x6_v1 S100000_Phi1000_B0.2";
        }
        if (model.m_end_region.holmstrom || model.m_end_region.resistivity>0.) {
            out << " numerical_end_signed_sinks_v1";
        }
    }
    auto token=[&](std::string const& value) {
        char const* hex="0123456789abcdef";
        out<<' '<<value.size()<<':';
        for(unsigned char c:value) { out<<hex[c>>4]<<hex[c&15]; }
    };
    auto expression=[&](std::string const& value,
                        amrex::Vector<std::string> const& variables) {
        token(value);
        amrex::Parser parsed(value);
        auto const symbols=parsed.symbols();
        for(auto const& symbol:symbols) {
            if(std::find(variables.begin(),variables.end(),symbol)!=variables.end()) { continue; }
            auto constant=amrex::ParmParse().makeParser(symbol,{});
            Real const resolved=constant.compileHost<0>()();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(resolved),
                "Native endpoint checkpoint physical parser constant must be finite");
            token(symbol);out<<' '<<resolved;
        }
        out<<" expression_end";
    };
    auto const& geometry=sim.Geom(0);
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        out<<' '<<static_cast<int>(WarpX::field_boundary_lo[d])
           <<' '<<static_cast<int>(WarpX::field_boundary_hi[d])
           <<' '<<static_cast<int>(WarpX::particle_boundary_lo[d])
           <<' '<<static_cast<int>(WarpX::particle_boundary_hi[d])
           <<' '<<geometry.ProbLo(d)<<' '<<geometry.ProbHi(d)
           <<' '<<geometry.isPeriodic(d);
    }
    bool smooth_force=false,split_ampere=false;
    amrex::ParmParse endpoint("endpoint_diagnostic");
    endpoint.query("smooth_force",smooth_force);
    endpoint.query("split_ampere",split_ampere);
    bool vacuum_split=false;
    endpoint.query("vacuum_split_ampere",vacuum_split);
    if(vacuum_split) out<<" vacuum_split_ampere_static_rz_v1 ";
    // Record requested and effective maps. A literal zero parser can disable
    // a configured current adapter; that distinction is part of the history.
    out<<" native_maps "<<NativePrescribedDriveEnabled()<<' '<<smooth_force<<' '
       <<split_ampere<<' '<<(split_ampere&&model.HasResistivity())<<' '
       <<model.HasResistivity()<<' '<<model.m_implicit_push_excludes_resistive_field;
    out<<" physical_sources "<<model.m_resistivity_has_Te_dependence<<' '
       <<model.m_resistivity_has_J_dependence<<' '<<model.m_include_joule_heating<<' '
       <<model.m_include_temperature_relaxation<<' '<<model.m_joule_redirect_to_ions<<' '
       <<model.m_has_energy_sink<<' '<<model.m_has_per_species_eta<<' '
       <<model.m_n_floor<<' '<<model.m_n_floor_smooth_width;
    token(model.m_resistivity_has_Te_dependence?"eta(rho,J,Te[K],t)":"eta(rho,J,t)");
    expression(model.m_eta_expression,model.m_resistivity_has_Te_dependence?
        amrex::Vector<std::string>{"rho","J","Te","t"}:
        amrex::Vector<std::string>{"rho","J","t"});
    token("heating_eta(rho,J,Te[eV],t)");
    expression(model.m_eta_heating_expression,{"rho","J","Te","t"});
    out<<" hyper "<<model.m_include_hyper_resistivity_term<<' '
       <<model.m_hyper_resistivity_has_B_dependence<<' '<<model.m_hyper_res_heating<<' '
       <<model.m_hyper_res_curl_curl<<' '<<model.m_hyper_resistivity_curlcurl;
    expression(model.m_eta_h_expression,{"rho","B"});
    out<<" viscosity "<<model.m_visc_model<<' '<<model.m_include_electron_viscosity<<' '
       <<model.m_visc_limiter<<' '<<model.m_visc_in_ohms_law<<' '<<model.m_visc_heating_work<<' '
       <<model.m_visc_coulomb_log<<' '<<model.m_visc_Z_eff<<' '
       <<model.m_visc_flux_limit_factor<<' '<<model.m_visc_nu_max<<' '<<model.m_visc_taper_n;
    expression(model.m_visc_nu_par_expression,{"n","Te","B"});
    expression(model.m_visc_nu_perp_expression,{"n","Te","B"});
    out<<" end_region "<<model.m_end_region.holmstrom<<' '<<model.m_end_region.resistivity;
    for(int side=0;side<2;++side) {
        out<<' '<<model.m_end_region.width[side]<<' '<<model.m_end_region.rolloff[side];
    }
    if(model.m_add_external_fields) {
        auto const& external=*model.m_external_vector_potential;
        amrex::ParmParse pp("external_vector_potential");
        bool clean=true;pp.query("do_diva_cleaning",clean);
        out<<" prescribed "<<external.nFields()<<' '<<clean;
        for(int i=0;i<external.nFields();++i) {
            auto const& name=external.FieldName(i);token(name);
            bool from_file=false,in_initial=false;
            pp.queryWithParser(name+".read_from_file",from_file);
            pp.queryWithParser(name+".in_initial_field",in_initial);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!from_file,
                "Native endpoint checkpoints require analytic prescribed fields; file content identity is not supported by this convention");
            out<<' '<<in_initial;
            std::string value="1.0";
            pp.query(name+".A_time_external_function(t)",value);expression(value,{"t"});
            value="0";
            bool const has_derivative=pp.query(
                name+".A_time_derivative_external_function(t)",value);
            out<<' '<<has_derivative;expression(value,{"t"});
            value="0";
            bool const has_increment=pp.query(
                name+".A_time_increment_external_function(t,dt)",value);
            out<<' '<<has_increment;expression(value,{"t","dt"});
            for(auto const* component:{"Ax","Ay","Az"}) {
                value="0.0";
                pp.query(name+"."+component+"_external_grid_function(x,y,z)",value);
                expression(value,{"x","y","z"});
            }
        }
    }
    return out.str();
}

void ValidateNativeEndpointRestart(WarpX& sim)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativeEndpointRestartSupported(),
        "Unsupported native endpoint checkpoint convention");
    auto const& model=*sim.get_pointer_HybridPICModel();
    ValidateNativeEndpointScope(sim,model);
    ValidateCorrelatedStaticDrive(sim);
    auto const shape=sim.m_fields.get_alldirs(FieldType::current_fp,0);
    bool dumps=false;
    amrex::ParmParse("endpoint_diagnostic").query("write_fields",dumps);
    for(auto const* name:{current_name,displacement_old_name,"hybrid_E_inertial_fp"}) {
        for(int c=0;c<3;++c) {
            auto const& f=*sim.m_fields.get(name,Direction{c},0);
            if(dumps) amrex::VisMF::Write(f,
                std::string("RESTART_")+name+"_"+std::to_string(c));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.ixType()==shape[c]->ixType() &&
                f.nComp()==1 && f.is_finite(0,1,f.nGrowVect()),
                "Invalid accepted native endpoint checkpoint field");
        }
    }
    // Verify the stored nodal companion, but never recreate independent D by
    // subtracting the much larger rounded C, I and Je fields.
    CompanionError(sim,sim.m_fields.get_alldirs(current_name,0),"hybrid_Je_n_nodal");
    // Circuit-dependent images require the restored provider. InitData invokes
    // the late circuit restart validator immediately after provider attachment.
    if (!NativeCircuitDriveEnabled()) ValidateNativeVacuumRestart(sim);
    warpx::darwin::ValidateNativePECWallState(sim);
    amrex::Print()<<"restart: accepted Yee Je, independent D and direct Ei validated without reseed\n";
}

namespace {
void BuildDriveReference(WarpX& sim,MF& out,int component,Real time,Real h,DriveRate mode) {
    out.setVal(0.);
    bool const circuit=NativeCircuitDriveEnabled();
    if((!NativePrescribedDriveEnabled()&&!circuit) || mode==DriveRate::Homogeneous)return;
    auto const& model=*sim.get_pointer_HybridPICModel();
    if(!model.m_add_external_fields)return;
    auto const& ext=*model.m_external_vector_potential;
    for(int coil=0;coil<ext.nFields();++coil) {
        if(circuit && ConfiguredCircuitField(ext.FieldName(coil))) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mode!=DriveRate::Instantaneous && ext.DeviceDriven(coil),
                "Native circuit interval rate requires an active device segment; "
                "accepted instantaneous rates require the prepared field context");
            auto const scales=ext.DeviceScales();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(scales.start && scales.end && h>0. &&
                scales.t0==time && scales.dt==h && std::isfinite(time) && std::isfinite(h),
                "Native circuit rate interval differs from its borrowed device segment");
            if(mode==DriveRate::Curvature)continue; // Native segment is exactly affine in time.
            auto const& unit=*sim.m_fields.get(ext.FieldName(coil)+"_Aext",Direction{component},0);
            AMREX_ALWAYS_ASSERT(out.boxArray()==unit.boxArray()&&
                out.DistributionMap()==unit.DistributionMap()&&unit.nGrowVect().allGE(out.nGrowVect()));
            for(amrex::MFIter mfi(out);mfi.isValid();++mfi) {
                auto const a=out.array(mfi);auto const u=unit.const_array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    a(i,j,k)+=scales.Slope(coil)*u(i,j,k);
                });
            }
            continue;
        }
        if(circuit && !ext.HasAnalyticTimeProfile(coil))continue; // Validated static parser.
        auto const profile=ext.TimeProfile(coil);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(profile.ConsistentIncrement(time,h)&&
            profile.ConsistentIncrement(time,.5*h),
            "Nonfinite or inconsistent prescribed native drive finite increment");
        auto const& unit=*sim.m_fields.get(ext.FieldName(coil)+"_Aext",Direction{component},0);
        AMREX_ALWAYS_ASSERT(out.boxArray()==unit.boxArray()&&
            out.DistributionMap()==unit.DistributionMap()&&
            unit.nGrowVect().allGE(out.nGrowVect()));
        for(amrex::MFIter mfi(out);mfi.isValid();++mfi) {
            auto const a=out.array(mfi);auto const u=unit.const_array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                Real const factor=mode==DriveRate::Instantaneous?profile.derivative(time):
                    mode==DriveRate::Secant?profile.increment(time,h)/h:
                    profile.MidpointCurvature(time,h);
                a(i,j,k)+=factor*u(i,j,k);
            });
        }
    }
}
void NativeCurlRate(WarpX& sim,View const& E,View const& A,View B,View C,
                    Real time,Real h,DriveRate mode) {
    AMREX_ALWAYS_ASSERT(NativeCorrelatedIncrementEnabled());
    auto const& model=*sim.get_pointer_HybridPICModel();
    ValidateCorrelatedStaticDrive(sim);

    for(int c=0;c<3;++c) {
        if(mode==DriveRate::Curvature)A[c]->setVal(0.);
        else MF::Copy(*A[c],*E[c],0,0,1,A[c]->nGrowVect());
        // The native affine A update acts on the entire allocation. Negating
        // valid cells only leaves the wrong sign at axis/PMC ghost corners.
        for(amrex::MFIter mfi(*A[c]);mfi.isValid();++mfi) {
            auto const a=A[c]->array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                a(i,j,k)=-a(i,j,k);
            });
        }
        B[c]->setVal(0.);C[c]->setVal(0.);
    }
    // Exact homogeneous derivative of native DarwinApplyABoundary. The
    // prescribed static A change is zero. Native PMC parity is required even
    // without an external object; normal half-cells remain free.
    amrex::GpuArray<int,AMREX_SPACEDIM> pmc_lo{},pmc_hi{};
    bool any_pmc=false;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        pmc_lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        pmc_hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
        any_pmc=any_pmc||pmc_lo[d]||pmc_hi[d];
    }
    if(model.m_add_external_fields||any_pmc){
#if defined(WARPX_DIM_RZ)
        AMREX_ALWAYS_ASSERT(sim.Geom(0).ProbLo(0)==0. &&
            (sim.Geom(0).isPeriodic(1) ||
             (model.UseCompatibleYeeInertia() && pmc_lo[1] && pmc_hi[1])) &&
            WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC);
        for(int c=0;c<3;++c){
            MF boundary(A[c]->boxArray(),A[c]->DistributionMap(),1,A[c]->nGrowVect());
            BuildDriveReference(sim,boundary,c,time,h,mode);
            bool const nodal=A[c]->ixType().nodeCentered(0);
            auto const domain=amrex::convert(sim.Geom(0).Domain(),A[c]->ixType());
            int const high=domain.bigEnd(0),zlo=domain.smallEnd(1),zhi=domain.bigEnd(1);
            bool const periodic_z=sim.Geom(0).isPeriodic(1);
            for(amrex::MFIter mfi(*A[c]);mfi.isValid();++mfi){auto const a=A[c]->array(mfi);auto const bc=boundary.const_array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    if(c==1&&i==0)a(i,j,k)=0.;
                    else if(nodal&&i>=high)a(i,j,k)=bc(high,periodic_z?j:amrex::Clamp(j,zlo,zhi),k);
                });
            }
            A[c]->FillBoundary(sim.Geom(0).periodicity());
            ApplyDarwinCellCenteredABoundary(*A[c],boundary,sim.Geom(0),true,
                                                nullptr,pmc_lo,pmc_hi);
        }
        sim.ApplyFieldBoundaryOnAxis(A[0],A[1],A[2],0);
#else
        amrex::Abort("Correlated native A-boundary adapter is qualified only in RZ");
#endif
    }
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(B,A,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    amrex::Vector<MF*> vec{B.begin(),B.end()};
    amrex::FillBoundaryAndSync_nowait(vec,sim.Geom(0).periodicity());
    amrex::FillBoundaryAndSync_finish(vec);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(C,B,sim.GetEBUpdateEFlag()[0],0);
    for(auto* f:C) PMCImages(*f);
}
} // namespace
void PrepareNativeCurlRate(WarpX& sim,Real time,Real h,warpx::darwin::NativeCoilField* coil) {
    auto& fields=sim.m_fields;
    AMREX_ALWAYS_ASSERT(fields.has(rate_node_name,0));
    auto const E=fields.get_alldirs(FieldType::Efield_fp,0);
    auto const A=fields.get_alldirs(rate_a_name,0),B=fields.get_alldirs(rate_b_name,0);
    auto const C=fields.get_alldirs(curl_rate_name,0);
    bool const drive=NativePrescribedDriveEnabled()||NativeCircuitDriveEnabled();
    AMREX_ALWAYS_ASSERT(!drive || (h>0.&&std::isfinite(time)&&std::isfinite(h)));
    if(drive)NativeCurlRate(sim,E,A,B,fields.get_alldirs(drive_curvature_name,0),
                           time,h,DriveRate::Curvature);
    NativeCurlRate(sim,E,A,B,C,time,h,drive?DriveRate::Secant:DriveRate::Homogeneous);
    if(warpx::darwin::NativeCoilCurrentEnabled()) {
        AMREX_ALWAYS_ASSERT(coil && NativeCircuitDriveEnabled());
        coil->SampleSecant(time,h);coil->SubtractCurrent(C);
        for(int c=0;c<3;++c){MF::Copy(*E[c],*A[c],0,0,1,E[c]->nGrowVect());NegateFullAllocation(*E[c]);}
    }
    if(auto* paired=warpx::darwin::NativeActivePairedFields(sim)) {
        AMREX_ALWAYS_ASSERT(!drive);
        auto const low=paired->ConditionedTransverseLow();
        View input{const_cast<MF*>(low[0]),const_cast<MF*>(low[1]),const_cast<MF*>(low[2])};
        auto small=paired->CurlRateLow();
        NativeCurlRate(sim,input,paired->CurlRatePotentialScratch(),paired->CurlRateMagneticScratch(),
            small,time,h,DriveRate::Homogeneous);
        for(int c=0;c<3;++c)for(amrex::MFIter mfi(*small[c]);mfi.isValid();++mfi) {
            auto hi=C[c]->array(mfi),lo=small[c]->array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto value=warpx::ohm::compensated::Sum(hi(i,j,k),lo(i,j,k));
                hi(i,j,k)=value.hi;lo(i,j,k)=value.lo;
            });
        }
    }
}
void NativeInstantaneousCurlRate(WarpX& sim,Real time,bool affine) {
    auto const E=sim.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto const A=sim.m_fields.get_alldirs(rate_a_name,0),B=sim.m_fields.get_alldirs(rate_b_name,0);
    auto const C=sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    NativeCurlRate(sim,E,A,B,C,time,0.,affine?DriveRate::Instantaneous:DriveRate::Homogeneous);
}
bool NativeConstantEtaMassMatrixSupported(WarpX& sim) {
#if defined(WARPX_DIM_RZ)
    auto const& model=*sim.get_pointer_HybridPICModel();
    bool smooth_force=false;
    amrex::ParmParse("endpoint_diagnostic").query("smooth_force",smooth_force);
    if (!(NativeEndpointEnabled() && NativeIonQuadratureEnabled() &&
          NativeCorrelatedIncrementEnabled() && model.UseCompatibleYeeInertia() &&
          model.UsesEulerianElectronEnergy() && sim.finestLevel()==0 &&
          sim.Geom(0).isPeriodic(1) && !NativePrescribedDriveEnabled() && !NativeCircuitDriveEnabled() && smooth_force &&
          model.m_implicit_push_excludes_resistive_field && model.m_include_joule_heating &&
          model.m_resistivity_parser && model.m_resistivity_parser->symbols().empty() &&
          !model.m_has_heating_resistivity && !model.m_has_per_species_eta &&
          model.m_end_region.resistivity==0. && !model.m_include_temperature_relaxation &&
          !model.m_joule_redirect_to_ions && !model.m_include_electron_viscosity &&
          !model.m_visc_in_ohms_law && !model.m_include_hyper_resistivity_term &&
          !model.m_hyper_res_heating && !model.m_has_energy_sink &&
          !model.m_has_electron_stopping && !model.m_esolve_tensor && !model.m_esolve_curlcurl)) {
        return false;
    }
    auto const& names=sim.GetPartContainer().GetSpeciesNames();
    if (names.size()!=1) { return false; }
    auto const& species=sim.GetPartContainer().GetParticleContainerFromName(names[0]);
    if (!(species.getCharge()>0.) || species.do_not_deposit) { return false; }
    Real const eta=model.m_resistivity_has_Te_dependence
        ? model.m_eta_te(1.,0.,1.,0.) : model.m_eta(1.,0.,0.);
    return std::isfinite(eta) && eta>=0.;
#else
    amrex::ignore_unused(sim);
    return false;
#endif
}
void ValidateNativeEndpointMassMatrixCapabilities(WarpX& sim) {
    auto const* pmc_owner=NativePrivatePMCMassMatrixOwner(sim);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx::darwin::NativePECPlasmaEnabled() || NativePECMMQualificationEnabled(),
        "PEC plasma boundary constitutive response requires full particle Jv until its MM tangent is qualified");
    auto const& model=*sim.get_pointer_HybridPICModel();
    bool smooth_force=false;
    amrex::ParmParse("endpoint_diagnostic").query("smooth_force",smooth_force);
#if defined(WARPX_DIM_RZ)
    bool const source_free=NativeEndpointEnabled() && NativeIonQuadratureEnabled() &&
        NativeCorrelatedIncrementEnabled() && model.UseCompatibleYeeInertia() &&
        model.UsesEulerianElectronEnergy() && sim.Geom(0).isPeriodic(1) &&
        !NativePrescribedDriveEnabled() && !smooth_force && !model.HasResistivity() &&
        !model.m_include_temperature_relaxation && !model.m_include_joule_heating &&
        !model.m_include_electron_viscosity && !model.m_has_energy_sink &&
        !model.m_has_electron_stopping;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source_free || NativeConstantEtaMassMatrixSupported(sim) || pmc_owner,
        "Endpoint MM requires source-free direct-Yee/correlated RZ with periodic z, or "
        "single-species constant-global-eta smooth-force Joule-only scope, or "
        "the private owner-bound PMC current/density capability; "
        "other sources, temperature-dependent pushes and prescribed drive are unsupported");
#else
    amrex::ignore_unused(model,smooth_force,pmc_owner);
    amrex::Abort("Stored native endpoint MM response currently requires RZ");
#endif
}
namespace {
bool NativeEndpointResponseDensityAdmissible(WarpX& sim) {
    auto const& rho=*sim.m_fields.get(FieldType::rho_fp,0);
    Real const floor=PhysConst::q_e*sim.get_pointer_HybridPICModel()->m_n_floor;
    // The source-free vacuum tensor has no density inverse. Its live native
    // inertia/pressure consumers own the floor; continuity retains zero charge.
    // Outside the qualified fixed-mask scope preserve the prior positive gate.
    auto const* pmc_owner=NativePrivatePMCMassMatrixOwner(sim);
    if (NativeVacuumMassMatrixSupported(sim) || pmc_owner) {
        return rho.is_finite(0,rho.nComp(),0) && rho.min(0)>=0. &&
            rho.min(rho.nComp()/2)>=0.;
    }
    return rho.is_finite(0,rho.nComp(),0) && rho.min(0)>floor &&
        rho.min(rho.nComp()/2)>floor;
}
}
bool FreezeNativeInertiaResponse(WarpX& sim,
    warpx::particles::NativeEndpointCurrentResponse& response,Real dt,int interval_width) {
    AMREX_ALWAYS_ASSERT(NativeIonQuadratureEnabled() && NativeCorrelatedIncrementEnabled());
    if (!NativeEndpointResponseDensityAdmissible(sim)) {
        response.Invalidate();
        return false;
    }
    auto& fields=sim.m_fields;
    return response.Freeze(sim,dt,interval_width,
        fields.get_alldirs(FieldType::Efield_fp_save,0),
        fields.get_alldirs(ion_old_name,0),fields.get_alldirs(ion_virtual_name,0),
        fields.get_alldirs(delta_ion_name,0),true);
}
bool PrepareNativeInertiaStage(WarpX& sim,Real dt,MF const*& current,MF const*& rate,
    warpx::particles::NativeEndpointCurrentResponse* response,
    warpx::darwin::NativeCoilField* coil,std::optional<Real> stage_time) {
    current=nullptr;rate=nullptr;
    bool adjacent=false;
    if(!TryNativeAdjacentInstantaneousPolicy(sim,adjacent,response==nullptr))return false;
    bool const direct_yee=sim.get_pointer_HybridPICModel()->UseCompatibleYeeInertia();
    if(direct_yee)for(auto* f:sim.m_fields.get_alldirs("hybrid_E_inertial_fp",0))f->setVal(0.);
    AMREX_ALWAYS_ASSERT(NativeIonQuadratureEnabled());
    // All allocations are established by constrained initialization, before
    // the solver's step rollback captures the registered field set.
    auto& fields=sim.m_fields;
    AMREX_ALWAYS_ASSERT(fields.has(inertia_node_name,0));
    auto const old=fields.get_alldirs(ion_old_name,0);
    auto const endpoint=fields.get_alldirs(ion_virtual_name,0);
    auto const jp=fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    auto const corrected=fields.get_alldirs(inertia_stage_name,0);
    auto& node=*fields.get(inertia_node_name,0);
    using warpx::particles::InstantaneousIonState;
    using warpx::particles::DepositNativeInstantaneousIonCurrent;
    bool const correlated = NativeCorrelatedIncrementEnabled();
    bool old_valid = false;
    bool endpoint_valid = false;
    if (response) {
        AMREX_ALWAYS_ASSERT(correlated);
        if (!NativeEndpointResponseDensityAdmissible(sim)) { return false; }
        old_valid=response->Apply(sim,dt,fields.get_alldirs(FieldType::Efield_aux,0),
            old,endpoint,fields.get_alldirs(delta_ion_name,0));
        endpoint_valid=old_valid;
        if(old_valid)Copy(fields.get_alldirs(delta_base_name,0),old);
    } else if (correlated) {
        auto const base = fields.get_alldirs(delta_base_name,0);
        auto const delta = fields.get_alldirs(delta_ion_name,0);
        // One fresh represented old point map owns the pair. Independent GPU
        // evaluations and atomic scatters of equal real formulas need not be
        // bitwise equal. Preserve the native finite increment and share its
        // base with the endpoint trapezoid; no accepted history is read here.
        old_valid = adjacent
            ? warpx::particles::DepositNativeAdjacentInstantaneousIonIncrement(sim,old,delta)
            : warpx::particles::DepositNativeInstantaneousIonIncrement(sim,old,delta);
        if (old_valid) { Copy(base,old); }
        else {
            fields.get(rate_node_name,0)->setVal(0.);
            amrex::Print()<<"Correlated inertia rejected missing kick/chord or common-support condition\n";
        }
    } else {
        old_valid = DepositNativeInstantaneousIonCurrent(sim,InstantaneousIonState::Old,old);
    }
    if(!response && old_valid) {
        endpoint_valid=DepositNativeInstantaneousIonCurrent(
            sim,InstantaneousIonState::VirtualEndpoint,endpoint);
    }
    if(!old_valid || !endpoint_valid) {
        for(auto* f:corrected) f->setVal(0.);
        node.setVal(0.);
        return false;
    }
    warpx::darwin::RebuildNativePECStageCurrent(sim,dt,coil,stage_time);
    // Jp includes the existing interval displacement. Preserve primary Ji and
    // the force/Hall Je; only the inertial stage uses the endpoint trapezoid.
    for(int c=0;c<3;++c) {
        MF::LinComb(*corrected[c],-.5,*old[c],0,-.5,*endpoint[c],0,0,1,
                    corrected[c]->nGrowVect());
        MF::Add(*corrected[c],*jp[c],0,0,1,corrected[c]->nGrowVect());
        if(NativePrescribedDriveEnabled()||NativeCircuitDriveEnabled())MF::Add(*corrected[c],
            *fields.get(drive_curvature_name,Direction{c},0),0,0,1,corrected[c]->nGrowVect());
        corrected[c]->OverrideSync(sim.Geom(0).periodicity());
        corrected[c]->FillBoundary(sim.Geom(0).periodicity());
        PMCImages(*corrected[c]);
    }
    if(warpx::darwin::NativePECPlasmaEnabled()) {
        auto& kappa=*fields.get("diagnostic_inertia_kappa_nodal",0);
        auto const& rho=*fields.get(FieldType::rho_fp,0);
        if(!BuildNativeInertiaKappa(sim,rho,rho.nComp()/2,kappa) ||
           !warpx::darwin::PrepareNativePECStage(sim,dt,kappa,old,endpoint,corrected))return false;
    }
    NodeCurrent(sim,corrected,node);
    if(correlated) {
        AMREX_ALWAYS_ASSERT(dt>0.&&std::isfinite(dt));
        auto const base=fields.get_alldirs(delta_base_name,0),delta=fields.get_alldirs(delta_ion_name,0);
        auto const d0=fields.get_alldirs(displacement_old_name,0),dbar=fields.get_alldirs(displacement_stage_name,0);
        auto const cdot=fields.get_alldirs(curl_rate_name,0),derivative=fields.get_alldirs(rate_yee_name,0);
        auto const EL=fields.get_alldirs("hybrid_E_long_fp",0),EL0=fields.get_alldirs("hybrid_E_long_old_fp",0);
        auto& stable=*fields.get(rate_node_name,0);
        bool const audit=EndpointAuditEnabled();
        Real fresh_base_copy_error=0.,endpoint_remainder=0.;
        for(int c=0;c<3;++c){
            if(audit) {
                MF error(base[c]->boxArray(),base[c]->DistributionMap(),1,0);
                MF::LinComb(error,1.,*base[c],0,-1.,*old[c],0,0,1,0);
                fresh_base_copy_error=std::max(fresh_base_copy_error,error.norminf());
                MF::LinComb(error,1.,*endpoint[c],0,-1.,*old[c],0,0,1,0);
                MF::Subtract(error,*delta[c],0,0,1,0);
                endpoint_remainder=std::max(endpoint_remainder,error.norminf());
            }
            if(!warpx::darwin::NativeActivePairedFields(sim)) {
            Real const factor=PhysConst::epsilon_0/(.5*dt),inv_dt=1./dt;
            for(amrex::MFIter mfi(*derivative[c]);mfi.isValid();++mfi){
                auto const out=derivative[c]->array(mfi),ds=dbar[c]->array(mfi);
                auto const C=cdot[c]->const_array(mfi),di=delta[c]->const_array(mfi),d=d0[c]->const_array(mfi),e=EL[c]->const_array(mfi),e0=EL0[c]->const_array(mfi);
                auto const* low=warpx::darwin::increment::Low(sim,c);
                auto const remainder=low?low->const_array(mfi):amrex::Array4<Real const>{};
                bool const compensated=low!=nullptr;
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    Real const displacement=factor*((e(i,j,k)-e0(i,j,k))+(compensated?remainder(i,j,k):0.));
                    ds(i,j,k)=displacement;
                    out(i,j,k)=C(i,j,k)-inv_dt*di(i,j,k)-2.*inv_dt*(displacement-d(i,j,k));
                });
            }
            }
        }
        if(auto* paired=warpx::darwin::NativeActivePairedFields(sim)) {
            paired->BuildCorrelatedRate({cdot[0],cdot[1],cdot[2]},
                {delta[0],delta[1],delta[2]},{d0[0],d0[1],d0[2]},derivative,dbar);
        }
        warpx::darwin::CompleteNativePECStageRate(sim,derivative);
        if(auto* paired=warpx::darwin::NativeActivePairedFields(sim))paired->CompletePECStageRateLow();
        for(int c=0;c<3;++c) {
            PMCImages(*dbar[c]);
            PMCImages(*derivative[c]);
        }
        if(audit) AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fresh_base_copy_error==0.,
            "Fresh shared correlated old-base copy differs within one stage evaluation");
        NodeCurrent(sim,derivative,stable);
        if(audit) {
            MF representation(node.boxArray(),node.DistributionMap(),3,0);
            auto const& old_je=*fields.get("hybrid_Je_n_nodal",0);
            MF::LinComb(representation,1.,node,0,-1.,old_je,0,0,3,0);
            representation.mult(2./dt);MF::Subtract(representation,stable,0,0,3,0);
            Real rate_remainder=0.;for(int c=0;c<3;++c)rate_remainder=std::max(rate_remainder,representation.norminf(c));
            if(amrex::ParallelDescriptor::IOProcessor()){
                std::ofstream log("INERTIA_INCREMENT_REMAINDER.jsonl",std::ios::app);
                log<<std::setprecision(17)<<"{\"step\":"<<sim.getistep(0)<<",\"dt\":"<<dt
                    <<",\"shared_fresh_old_base\":true,\"independent_base_comparison_performed\":false"
                    <<",\"fresh_base_copy_error_A_per_m2\":"<<fresh_base_copy_error
                    <<",\"represented_endpoint_current_remainder_A_per_m2\":"<<endpoint_remainder
                    <<",\"represented_rate_remainder_A_per_m2_s\":"<<rate_remainder
                    <<",\"equivalent_stage_current_remainder_A_per_m2\":"<<.5*dt*rate_remainder<<"}\n";
            }
        }
        if(direct_yee) {
            auto& kap=*fields.get("diagnostic_inertia_kappa_nodal",0);
            auto const& rho=*fields.get(FieldType::rho_fp,0);
            if(NativeVacuumEndpointEnabled()) {
                if(!warpx::darwin::ApplyNativeVacuumInertia(sim,rho,rho.nComp()/2,derivative,
                    fields.get_alldirs("hybrid_E_inertial_fp",0))) return false;
            } else {
            if(!BuildNativeInertiaKappa(sim,rho,rho.nComp()/2,kap)) return false;
            auto const& model=*sim.get_pointer_HybridPICModel();
            Real const scale=model.m_electron_inertia_mass/
                (PhysConst::q_e*PhysConst::q_e*model.m_n0_ref);
            if(auto* paired=warpx::darwin::NativeActivePairedFields(sim)) {
                auto low=paired->InertiaElectricLow();
                warpx::darwin::ApplyYeeInertiaMassPaired(sim.Geom(0),kap,
                    {derivative[0],derivative[1],derivative[2]},paired->RateLow(),
                    fields.get_alldirs("hybrid_E_inertial_fp",0),scale,&low);
            } else {
            warpx::darwin::ApplyYeeInertiaMass(sim.Geom(0),kap,
                {derivative[0],derivative[1],derivative[2]},
                fields.get_alldirs("hybrid_E_inertial_fp",0),scale);
            }
            for(auto* f:fields.get_alldirs("hybrid_E_inertial_fp",0)) PMCImages(*f);
            }
        }
        rate=&stable;
    }
    current=&node;
    return true;
}
void CopyNativeEndpointCurrentMirror(WarpX& sim,View const& current,MF& out) {
    NodeCurrent(sim,current,out);
}
void PublishNativeYeeInertiaMirror(WarpX& sim) {
    if(sim.get_pointer_HybridPICModel()->UseCompatibleYeeInertia())
        NodeCurrent(sim,sim.m_fields.get_alldirs("hybrid_E_inertial_fp",0),
            *sim.m_fields.get("hybrid_E_inertial_nodal",0));
}
bool TryCaptureNativeEndpointStage(WarpX& sim,Real time,Real dt,warpx::darwin::NativeCoilField* coil){
 EnsureCompanion(sim);
 auto const jp=sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0),
            ji=sim.m_fields.get_alldirs(FieldType::current_fp,0),
            dst=sim.m_fields.get_alldirs(stage_name,0);
 if(NativeIonQuadratureEnabled()) {
    MF const* corrected=nullptr;MF const* corrected_rate=nullptr;
    bool prepared=PrepareNativeInertiaStage(sim,dt,corrected,corrected_rate,nullptr,coil,time);
    amrex::ParallelDescriptor::ReduceBoolAnd(prepared);
    if(!prepared)return false;
    Copy(dst,sim.m_fields.get_alldirs(inertia_stage_name,0));
    if(EndpointAuditEnabled()) {
        auto force=CurrentAllocate(ji);
        for(int c=0;c<3;++c) MF::LinComb(force[c],1.,*jp[c],0,-1.,*ji[c],0,0,1,force[c].nGrowVect());
        MF mismatch(corrected->boxArray(),corrected->DistributionMap(),3,1);
        NodeCurrent(sim,Pointers(force),mismatch);
        MF::Subtract(mismatch,*corrected,0,0,3,0);
        MF work(mismatch.boxArray(),mismatch.DistributionMap(),1,0);
        auto const& ei=*sim.m_fields.get("hybrid_E_inertial_nodal",0);
        for(amrex::MFIter mfi(work);mfi.isValid();++mfi) {
            auto const w=work.array(mfi);
            auto const j=mismatch.const_array(mfi),e=ei.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int k,int l) {
                w(i,k,l)=dt*(j(i,k,l,0)*e(i,k,l,0)+j(i,k,l,1)*e(i,k,l,1)+j(i,k,l,2)*e(i,k,l,2));
            });
        }
        Real const mismatch_work=sim.get_pointer_HybridPICModel()->EnergyVolumeIntegral(work,0,0);
        if(sim.get_pointer_HybridPICModel()->UseCompatibleYeeInertia()) {
            auto const Ei=sim.m_fields.get_alldirs("hybrid_E_inertial_fp",0);
            Real edge_work=0.;
            for(int c=0;c<3;++c) {
                MF ew(force[c].boxArray(),force[c].DistributionMap(),1,0);
                for(amrex::MFIter mfi(ew);mfi.isValid();++mfi) {
                    auto const out=ew.array(mfi);
                    auto const jf=force[c].const_array(mfi),
                        jt=dst[c]->const_array(mfi),e=Ei[c]->const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                        out(i,j,k)=dt*(jf(i,j,k)-jt(i,j,k))*e(i,j,k);
                    });
                }
                edge_work+=sim.get_pointer_HybridPICModel()->EnergyVolumeIntegral(ew,0,0);
            }
            if(amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream f("INERTIA_YEE_QUADRATURE_WORK.jsonl",std::ios::app);
                f<<std::setprecision(17)<<"{\"step\":"<<sim.getistep(0)<<",\"time\":"<<time
                 <<",\"edge_force_minus_inertia_work_J\":"<<edge_work
                 <<",\"nodal_mirror_force_minus_inertia_work_J\":"<<mismatch_work<<"}\n";
            }
        }
        if(amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream f("INERTIA_QUADRATURE_WORK.jsonl",std::ios::app);
            f<<std::setprecision(17)<<"{\"step\":"<<sim.getistep(0)<<",\"time\":"<<time
             <<",\"force_minus_inertia_work_J\":"<<mismatch_work<<"}\n";
        }
    }
 } else {
    for(int c=0;c<3;++c) {
        MF::LinComb(*dst[c],1.,*jp[c],0,-1.,*ji[c],0,0,1,dst[c]->nGrowVect());
        dst[c]->OverrideSync(sim.Geom(0).periodicity());
        dst[c]->FillBoundary(sim.Geom(0).periodicity());
    }
 }
 auto const e=CompanionError(sim,dst,"hybrid_Je_theta_nodal");amrex::Print()<<"Native endpoint companion stage relative error="<<e<<"\n";bool dumps=false;amrex::ParmParse("endpoint_diagnostic").query("write_fields",dumps);if(dumps){std::string const prefix="MIDPOINT_"+std::to_string(sim.getistep(0))+"_";for(int c=0;c<3;++c){amrex::VisMF::Write(*sim.m_fields.get(FieldType::Efield_fp,Direction{c},0),prefix+"E_"+std::to_string(c));amrex::VisMF::Write(*sim.m_fields.get(FieldType::Bfield_fp,Direction{c},0),prefix+"B_"+std::to_string(c));amrex::VisMF::Write(*sim.m_fields.get("hybrid_A_fp",Direction{c},0),prefix+"A_"+std::to_string(c));amrex::VisMF::Write(*ji[c],prefix+"Ji_"+std::to_string(c));amrex::VisMF::Write(*dst[c],prefix+"Je_"+std::to_string(c));}amrex::VisMF::Write(*sim.m_fields.get(FieldType::rho_fp,0),prefix+"rho");if(amrex::ParallelDescriptor::IOProcessor()){std::ofstream f("MIDPOINT_TIMES.jsonl",std::ios::app);f<<std::setprecision(17)<<"{\"step\":"<<sim.getistep(0)<<",\"time\":"<<time<<"}\n";}}
 return true;
}
void CaptureNativeEndpointStage(WarpX& sim,Real time,Real dt,warpx::darwin::NativeCoilField* coil){
 AMREX_ALWAYS_ASSERT_WITH_MESSAGE(TryCaptureNativeEndpointStage(sim,time,dt,coil),
     "Accepted endpoint-quadrature current must be freshly admissible");
}
void RotateNativeEndpointCurrent(WarpX& sim,Real theta){
 warpx::darwin::RotateNativePECWall(sim);
 if(NativeCorrelatedIncrementEnabled()) {
    AMREX_ALWAYS_ASSERT(theta==.5);
    auto const old=sim.m_fields.get_alldirs(displacement_old_name,0);
    auto const stage=sim.m_fields.get_alldirs(displacement_stage_name,0);
    for(int c=0;c<3;++c){MF::LinComb(*old[c],2.,*stage[c],0,-1.,*old[c],0,0,1,old[c]->nGrowVect());old[c]->OverrideSync(sim.Geom(0).periodicity());old[c]->FillBoundary(sim.Geom(0).periodicity());PMCImages(*old[c]);}
 }
 if(NativeIonQuadratureEnabled()) {
    auto const virtual_ion=sim.m_fields.get_alldirs(ion_virtual_name,0);
    auto actual=Allocate(virtual_ion);
    AMREX_ALWAYS_ASSERT(warpx::particles::DepositNativeInstantaneousIonCurrent(
        sim,warpx::particles::InstantaneousIonState::Current,Pointers(actual)));
    Real error=0.,scale=0.;
    for(int c=0;c<3;++c) {
        scale=std::max(scale,actual[c].norminf());
        MF::Subtract(actual[c],*virtual_ion[c],0,0,1,0);
        error=std::max(error,actual[c].norminf());
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error<=3.e-13*std::max(1.,scale),
        "Virtual instantaneous current differs from actual native accepted endpoint");
    amrex::Print()<<"Native endpoint ion virtual/accepted relative error="<<error/std::max(1.,scale)<<"\n";
 }
 auto const old=sim.m_fields.get_alldirs(current_name,0),stage=sim.m_fields.get_alldirs(stage_name,0);for(int c=0;c<3;++c){MF::LinComb(*old[c],1./theta,*stage[c],0,1.-1./theta,*old[c],0,0,1,old[c]->nGrowVect());old[c]->OverrideSync(sim.Geom(0).periodicity());old[c]->FillBoundary(sim.Geom(0).periodicity());PMCImages(*old[c]);}auto const e=CompanionError(sim,old,"hybrid_Je_n_nodal");amrex::Print()<<"Native endpoint companion accepted relative error="<<e<<"\n";}
namespace owned_ampere_detail {
bool ArithmeticSupported(WarpX& sim) {
 bool supported=warpx::darwin::NativeEndpointArithmeticSupported();
#if defined(AMREX_USE_GPU)
 supported=supported&&warpx::darwin::NativeEndpointCudaQualificationSelected(sim);
#else
 amrex::ignore_unused(sim);
#endif
 return supported;
}
// Bound arithmetic only. The physical producer is never reevaluated through
// these expressions. Reject overflow, nonnormal operands and lost positive
// products; no FTZ/underflow allowance is silently invented.
AMREX_GPU_HOST_DEVICE bool Normal(Real x) {
 return std::isfinite(x)&&(x==0.||std::abs(x)>=std::numeric_limits<Real>::min());
}
AMREX_GPU_HOST_DEVICE Real Up(Real x) {
 if(x==0.)return 0.;
 if(!Normal(x)||x<0.)return std::numeric_limits<Real>::infinity();
 return std::nextafter(x,std::numeric_limits<Real>::infinity());
}
AMREX_GPU_HOST_DEVICE Real Add(Real a,Real b) {
 if(!Normal(a)||!Normal(b)||a<0.||b<0.)return std::numeric_limits<Real>::infinity();
 return Up(a+b);
}
AMREX_GPU_HOST_DEVICE Real Mul(Real a,Real b) {
 if(!Normal(a)||!Normal(b)||a<0.||b<0.)return std::numeric_limits<Real>::infinity();
 if(a==0.||b==0.)return 0.;
 Real const x=a*b;return x==0.?std::numeric_limits<Real>::infinity():Up(x);
}
AMREX_GPU_HOST_DEVICE Real Gamma(int n) {
 Real const x=n*std::numeric_limits<Real>::epsilon();return Up(x/(1.-x));
}
AMREX_GPU_HOST_DEVICE Real Abs(Real x) {
 return Normal(x)?std::abs(x):std::numeric_limits<Real>::infinity();
}
AMREX_GPU_HOST_DEVICE Real Sum4(Real a,Real b,Real c,Real d) {
 return Add(Add(a,b),Add(c,d));
}
using Arrays=amrex::GpuArray<amrex::Array4<Real const>,3>;
#if defined(WARPX_DIM_RZ)
// Positive companion of the actual m=0 order-two Yee curl of A. Geometry
// coefficients use the native radial locations; outward divisions dominate
// the rounded native coefficients. PMC/periodic/box images are consumed from
// the actual conditioned A halos, never inferred from allocated storage.
AMREX_GPU_HOST_DEVICE Real CurlMagnitude(Arrays const& a,int c,int i,int j,Real dr,Real dz) {
 Real const ir=Up(1./dr),iz=Up(1./dz);
 if(c==0)return i==0?0.:Mul(iz,Add(Abs(a[1](i,j+1,0)),Abs(a[1](i,j,0))));
 if(c==1)return Add(Mul(ir,Add(Abs(a[2](i+1,j,0)),Abs(a[2](i,j,0)))),
                       Mul(iz,Add(Abs(a[0](i,j+1,0)),Abs(a[0](i,j,0)))));
 Real const r=(i+.5)*dr;
 return Mul(Mul(ir,Up(1./r)),Add(Mul(Up(std::abs(r+.5*dr)),Abs(a[1](i+1,j,0))),
                                       Mul(Up(std::abs(r-.5*dr)),Abs(a[1](i,j,0)))));
}
AMREX_GPU_HOST_DEVICE Real CurrentMagnitude(Arrays const& a,int c,int i,int j,Real dr,Real dz) {
 Real const ir=Up(1./dr),iz=Up(1./dz),im=Up(1./PhysConst::mu0);
 if(c==0)return Mul(Mul(im,iz),Add(CurlMagnitude(a,1,i,j,dr,dz),CurlMagnitude(a,1,i,j-1,dr,dz)));
 if(c==1)return i==0?0.:Mul(im,Add(
     Mul(ir,Add(CurlMagnitude(a,2,i,j,dr,dz),CurlMagnitude(a,2,i-1,j,dr,dz))),
     Mul(iz,Add(CurlMagnitude(a,0,i,j,dr,dz),CurlMagnitude(a,0,i,j-1,dr,dz)))));
 if(i==0)return Mul(Mul(im,Mul(4.,ir)),CurlMagnitude(a,1,i,j,dr,dz));
 Real const r=i*dr;
 return Mul(Mul(Mul(im,ir),Up(1./r)),Add(
     Mul(Up(std::abs(r+.5*dr)),CurlMagnitude(a,1,i,j,dr,dz)),
     Mul(Up(std::abs(r-.5*dr)),CurlMagnitude(a,1,i-1,j,dr,dz))));
}
// Keep the launch outside the private origin factory: NVCC extended-device
// lambdas require an enclosing free function or publicly accessible method.
void FillAcceptedOriginRows (amrex::iMultiFab& free, MF& bound, View const& A,
    MF const& current, MF const& ion, MF const& electron, MF const& displacement,
    int c, int wall, Real dr, Real dz, NativeCurrentTolerance tolerance)
{
    for (amrex::MFIter it(free); it.isValid(); ++it) {
        Arrays potential{};
        for (int d=0; d<3; ++d) { potential[d]=A[d]->const_array(it); }
        auto out=free.array(it);
        auto limit=bound.array(it);
        auto cf=current.const_array(it), ji=ion.const_array(it),
             je=electron.const_array(it), dd=displacement.const_array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            out(i,j,k)=!((c!=0&&i==wall)||(c==1&&i==0));
            if (out(i,j,k)) {
                limit(i,j,k)=tolerance.Bound(Mul(Gamma(128),Add(CurrentMagnitude(potential,c,i,j,dr,dz),
                    Sum4(Abs(cf(i,j,k)),Abs(ji(i,j,k)),Abs(je(i,j,k)),Abs(dd(i,j,k))))));
            }
        });
    }
}
#endif
bool All(bool ready){amrex::ParallelDescriptor::ReduceBoolAnd(ready);return ready;}
}

amrex::Real NativeEndpointCurrentAbsoluteTolerance(WarpX& sim)
{
    Real absolute = 0.;
    bool scales_valid = true;
    bool const specified = amrex::ParmParse("implicit_evolve").query(
        "darwin_current_absolute_tolerance", absolute);
    if (!specified) {
        bool material = false;
        amrex::ParmParse source("native_stopping_event");
        source.query("material_support", material);
        if (material) {
            Real speed = 0.;
            source.get("proper_speed_cap", speed);
            Real const density = sim.get_pointer_HybridPICModel()->m_n0_ref;
            Real const scaled = warpx::thermal::NativeStoppingThermalOptions{}
                .nonlinear.absolute_tolerance;
            absolute = scaled * (PhysConst::q_e * density * speed);
            scales_valid = std::isfinite(density) && density > 0. &&
                std::isfinite(speed) && speed > 0. && absolute > 0.;
        }
    }
    bool valid = scales_valid && std::isfinite(absolute) && absolute >= 0.;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(valid,
        "implicit_evolve.darwin_current_absolute_tolerance must be finite and nonnegative (A/m^2)");
    Real low = absolute, high = absolute;
    amrex::ParallelDescriptor::ReduceRealMin(low);
    amrex::ParallelDescriptor::ReduceRealMax(high);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(low == high,
        "Native endpoint current tolerance must agree across ranks");
    return absolute;
}

struct NativeOwnedEndpointAmpere::Impl {
 WarpX& sim;
 Vector old_C,stage_C,virtual_I,expected_Je,expected_D,static_B;
 Vector linear_scale,base_bound,je_bound,d_bound;
 bool captured=false,verified=false;
 Real dt=0.,old_error=0.,stage_error=0.;
 NativeCurrentTolerance tolerance;
 explicit Impl(WarpX& w):sim(w){}
 bool Layout(Masks const& P,Masks const& V) const {
  bool ok=true;auto const shape=sim.m_fields.get_alldirs(current_name,0);
  for(int c=0;c<3;++c)ok=ok&&P[c]&&V[c]&&P[c]->nComp()==1&&V[c]->nComp()==1&&
      P[c]->boxArray()==shape[c]->boxArray()&&V[c]->boxArray()==shape[c]->boxArray()&&
      P[c]->DistributionMap()==shape[c]->DistributionMap()&&V[c]->DistributionMap()==shape[c]->DistributionMap();
  return owned_ampere_detail::All(ok);
 }
};
NativeOwnedEndpointAmpere::NativeOwnedEndpointAmpere(WarpX& w):m_impl(std::make_unique<Impl>(w)){}
NativeOwnedEndpointAmpere::~NativeOwnedEndpointAmpere()=default;

bool NativeOwnedEndpointAmpere::Capture(Real dt,Masks const& P,Masks const& V) {
 using namespace owned_ampere_detail;auto& r=*m_impl;auto& sim=r.sim;
 r.captured=false;r.verified=false;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(dt,P,V);return false;
#else
 bool split=false;amrex::ParmParse("endpoint_diagnostic").query("vacuum_split_ampere",split);
 auto const& model=*sim.get_pointer_HybridPICModel();
 bool ready=ArithmeticSupported(sim)&&Normal(dt)&&dt>0.&&
     NativeEndpointEnabled()&&NativeCorrelatedIncrementEnabled()&&NativeIonQuadratureEnabled()&&
     split&&model.m_darwin&&model.m_darwin_vacuum_recovery&&model.UseCompatibleYeeInertia()&&
     !model.m_has_external_current&&!warpx::darwin::NativeActivePairedFields(sim)&&
     !warpx::darwin::NativePECPlasmaEnabled()&&sim.maxLevel()==0&&!sim.DoPML()&&
     !EB::enabled()&&!WarpX::do_single_precision_comms&&WarpX::ncomps==1&&
     sim.Geom(0).ProbLo(0)==0.&&WarpX::grid_type==GridType::Staggered&&
     (!WarpX::getCosts(0)||WarpX::load_balance_costs_update_algo!=LoadBalanceCostsUpdateAlgo::Timers);
 if(!All(ready)||!r.Layout(P,V))return false;
 auto const A0=sim.m_fields.get_alldirs("hybrid_A_old_fp",0),At=sim.m_fields.get_alldirs("hybrid_A_fp",0);
 auto const B=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 auto const J0=sim.m_fields.get_alldirs(current_name,0),Jt=sim.m_fields.get_alldirs(stage_name,0);
 auto const D0=sim.m_fields.get_alldirs(displacement_old_name,0),Dt=sim.m_fields.get_alldirs(displacement_stage_name,0);
 auto const I0=sim.m_fields.get_alldirs(ion_old_name,0),Iv=sim.m_fields.get_alldirs(ion_virtual_name,0);
 auto const E=sim.m_fields.get_alldirs("hybrid_E_long_fp",0),E0=sim.m_fields.get_alldirs("hybrid_E_long_old_fp",0);
 // This first certificate admits the native constant axial background only.
 // Its current is exactly zero; the large uniform Bz must not become a fitted
 // curl-cancellation allowance. Nonconstant/static imposed fields stay closed.
 for(int c=0;c<3;++c) {
  ready=ready&&B[c]->is_finite(0,1,B[c]->nGrow());
  Real lo=B[c]->min(0,B[c]->nGrow()),hi=B[c]->max(0,B[c]->nGrow());
  ready=ready&&Normal(lo)&&lo==hi&&(c==2||lo==0.);
 }
 for(int c=0;c<3;++c) {
  auto const& shape=*J0[c];
  for(auto const* f:{A0[c],At[c],Jt[c],D0[c],Dt[c],I0[c],Iv[c],E[c],E0[c]})
   ready=ready&&f&&f->nComp()==1&&f->boxArray()==shape.boxArray()&&
       f->DistributionMap()==shape.DistributionMap()&&f->nGrowVect().allGE(1);
  auto const* low=warpx::darwin::increment::Low(sim,c);
  if(low)ready=ready&&low->boxArray()==shape.boxArray()&&low->DistributionMap()==shape.DistributionMap()&&low->nComp()==1;
 }
 if(!All(ready))return false;
 r.tolerance.absolute=NativeEndpointCurrentAbsoluteTolerance(sim);
 r.dt=dt;r.old_C=Allocate(J0);r.stage_C=Allocate(J0);r.virtual_I=Clone(Iv);r.static_B=Clone(B);
 r.expected_Je=Allocate(J0);r.expected_D=Allocate(D0);
 for(auto* v:{&r.linear_scale,&r.base_bound,&r.je_bound,&r.d_bound})*v=Allocate(J0);
 for(auto* v:{&r.old_C,&r.stage_C})for(auto& f:*v)f.setVal(0.);
 warpx::darwin::CalculateNativeSplitAmpere(sim,A0,B,Pointers(r.old_C));
 warpx::darwin::CalculateNativeSplitAmpere(sim,At,B,Pointers(r.stage_C));
 for(int c=0;c<3;++c){MF::LinComb(r.expected_Je[c],2.,*Jt[c],0,-1.,*J0[c],0,0,1,0);
  MF::LinComb(r.expected_D[c],2.,*Dt[c],0,-1.,*D0[c],0,0,1,0);}
 Real const dr=sim.Geom(0).CellSize(0),dz=sim.Geom(0).CellSize(1);
 Real const fJ=PhysConst::epsilon_0*(1./(.5*dt)),fD=PhysConst::epsilon_0/(.5*dt);
 if(!All(Normal(fJ)&&Normal(fD)&&fJ>0.&&fD>0.))return false;
 NativeCurrentTolerance const tolerance=r.tolerance;
 int bad=0;Real old_peak=0.,stage_peak=0.,old_ratio=0.,stage_ratio=0.;amrex::Long rows=0;
 for(int c=0;c<3;++c) {
  auto const* low=warpx::darwin::increment::Low(sim,c);bool const have_low=low!=nullptr;
  amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpSum> op;
  amrex::ReduceData<int,Real,Real,Real,Real,amrex::Long> data(op);using T=decltype(data)::Type;
  for(amrex::MFIter it(r.old_C[c]);it.isValid();++it) {
   Arrays a0{},at{};for(int d=0;d<3;++d){a0[d]=A0[d]->const_array(it);at[d]=At[d]->const_array(it);}
   auto p=P[c]->const_array(it),v=V[c]->const_array(it);
   auto c0=r.old_C[c].const_array(it),ct=r.stage_C[c].const_array(it),j0=J0[c]->const_array(it),jt=Jt[c]->const_array(it),d0=D0[c]->const_array(it),ds=Dt[c]->const_array(it),i0=I0[c]->const_array(it),iv=Iv[c]->const_array(it),e=E[c]->const_array(it),e0=E0[c]->const_array(it);
   auto elow=have_low?low->const_array(it):amrex::Array4<Real const>{};
   auto lm=r.linear_scale[c].array(it),base=r.base_bound[c].array(it),jr=r.je_bound[c].array(it),dd=r.d_bound[c].array(it);
   op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T {
    if(!(p(i,j,k)||v(i,j,k)))return {0,0.,0.,0.,0.,0};
    Real const m0=CurrentMagnitude(a0,c,i,j,dr,dz),mt=CurrentMagnitude(at,c,i,j,dr,dz);
    Real const current0=Sum4(Abs(c0(i,j,k)),Abs(i0(i,j,k)),Abs(j0(i,j,k)),Abs(d0(i,j,k)));
    // Incoming endpoint may retain the finite source-solve error. Carry its
    // fixed absolute budget through the field-stage identity below.
    Real const b0=tolerance.Bound(Mul(Gamma(128),Add(m0,current0)));
    Real const er0=((c0(i,j,k)-i0(i,j,k))-j0(i,j,k))-d0(i,j,k);
    Real const em=Add(Add(Abs(e(i,j,k)),Abs(e0(i,j,k))),have_low?Abs(elow(i,j,k)):0.);
    Real const coeff=Mul(Up(std::abs(fJ-fD)),em);
    Real const displacement=Mul(Gamma(8),Mul(Add(Abs(fJ),Abs(fD)),em));
    Real const stage_sum=Add(Sum4(Abs(ct(i,j,k)),Abs(ds(i,j,k)),Abs(jt(i,j,k)),
        Mul(.5,Add(Abs(i0(i,j,k)),Abs(iv(i,j,k))))),Mul(Abs(fJ),em));
    Real const bq=Add(Mul(Gamma(128),mt),Add(coeff,Add(displacement,Mul(Gamma(16),stage_sum))));
    Real const eq=jt(i,j,k)-((ct(i,j,k)-ds(i,j,k))-.5*(i0(i,j,k)+iv(i,j,k)));
    Real const bj=Mul(Gamma(3),Add(Mul(2.,Abs(jt(i,j,k))),Abs(j0(i,j,k))));
    Real const bd=Mul(Gamma(3),Add(Mul(2.,Abs(ds(i,j,k))),Abs(d0(i,j,k))));
    lm(i,j,k)=Add(m0,Mul(2.,mt));jr(i,j,k)=bj;dd(i,j,k)=bd;
    base(i,j,k)=Add(Add(b0,Mul(2.,bq)),Add(bj,bd));
    bool const finite=Normal(m0)&&Normal(mt)&&Normal(b0)&&Normal(bq)&&Normal(er0)&&Normal(eq)&&Normal(bj)&&Normal(bd)&&Normal(base(i,j,k));
    int issue=!finite?1:((std::abs(er0)>b0?2:0)|(std::abs(eq)>bq?4:0));
    return {issue,std::abs(er0),std::abs(eq),b0?std::abs(er0)/b0:0.,bq?std::abs(eq)/bq:0.,1};
   });
  }
  auto value=data.value();bad=std::max(bad,amrex::get<0>(value));old_peak=std::max(old_peak,amrex::get<1>(value));stage_peak=std::max(stage_peak,amrex::get<2>(value));old_ratio=std::max(old_ratio,amrex::get<3>(value));stage_ratio=std::max(stage_ratio,amrex::get<4>(value));rows+=amrex::get<5>(value);
 }
 amrex::ParallelDescriptor::ReduceIntMax(bad);amrex::ParallelDescriptor::ReduceRealMax(old_peak);amrex::ParallelDescriptor::ReduceRealMax(stage_peak);amrex::ParallelDescriptor::ReduceRealMax(old_ratio);amrex::ParallelDescriptor::ReduceRealMax(stage_ratio);amrex::ParallelDescriptor::ReduceLongSum(rows);
 amrex::Print()<<std::setprecision(17)<<"OWNED_ENDPOINT_AMPERE capture bad="<<bad<<" free_local_rows_sum="<<rows<<" old_A_per_m2="<<old_peak<<" old_bound_ratio="<<old_ratio<<" stage_A_per_m2="<<stage_peak<<" stage_bound_ratio="<<stage_ratio<<"\n";
 r.old_error=old_peak;r.stage_error=stage_peak;r.captured=bad==0&&rows>0;return r.captured;
#endif
}

bool NativeOwnedEndpointAmpere::Verify(Masks const& P,Masks const& V) {
 using namespace owned_ampere_detail;auto& r=*m_impl;auto& sim=r.sim;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(P,V);return false;
#else
 if(!All(r.captured&&!r.verified&&ArithmeticSupported(sim))||!r.Layout(P,V))return false;
 auto const A=sim.m_fields.get_alldirs("hybrid_A_fp",0),B=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 auto const J=sim.m_fields.get_alldirs(current_name,0),D=sim.m_fields.get_alldirs(displacement_old_name,0);
 bool unchanged=true;
 for(int c=0;c<3;++c){MF difference(B[c]->boxArray(),B[c]->DistributionMap(),1,B[c]->nGrowVect());
  MF::LinComb(difference,1.,*B[c],0,-1.,r.static_B[c],0,0,1,difference.nGrowVect());
  unchanged=unchanged&&difference.is_finite(0,1,difference.nGrow())&&difference.norminf(0,difference.nGrow())==0.;}
 if(!All(unchanged))return false;
 auto C=Allocate(J),I=Allocate(J);for(auto& f:C)f.setVal(0.);
 warpx::darwin::CalculateNativeSplitAmpere(sim,A,B,Pointers(C));
 bool ion=warpx::particles::DepositNativeInstantaneousIonCurrent(sim,
     warpx::particles::InstantaneousIonState::Current,Pointers(I));
 if(!All(ion))return false;
 Real scale=0.;for(auto const& f:I)scale=std::max(scale,f.norminf());
 Real const ion_bound=Mul(3.e-13,std::max(1.,scale));
 Real const dr=sim.Geom(0).CellSize(0),dz=sim.Geom(0).CellSize(1);
 int bad=0;Real peak=0.,ratio=0.,center_peak=0.,bound_peak=0.;amrex::Long rows=0;
 for(int c=0;c<3;++c){
  amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpSum> op;
  amrex::ReduceData<int,Real,Real,Real,Real,amrex::Long> data(op);using T=decltype(data)::Type;
  for(amrex::MFIter it(C[c]);it.isValid();++it){
   Arrays potential{};for(int d=0;d<3;++d)potential[d]=A[d]->const_array(it);
   auto p=P[c]->const_array(it),v=V[c]->const_array(it);
   auto c1=C[c].const_array(it),i1=I[c].const_array(it),j1=J[c]->const_array(it),d1=D[c]->const_array(it),c0=r.old_C[c].const_array(it),ct=r.stage_C[c].const_array(it),iv=r.virtual_I[c].const_array(it);
   auto je=r.expected_Je[c].const_array(it),de=r.expected_D[c].const_array(it),lm=r.linear_scale[c].const_array(it),base=r.base_bound[c].const_array(it),jr=r.je_bound[c].const_array(it),dd=r.d_bound[c].const_array(it);
   op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T {
    if(!(p(i,j,k)||v(i,j,k)))return {0,0.,0.,0.,0.,0};
    // 128 dominates two composed RZ curls, affine A construction, metric
    // coefficient formation and native owner/image copies (exact signs).
    // Static B is unchanged and has identically zero current in this scope.
    Real const bc=Mul(Gamma(128),Add(lm(i,j,k),CurrentMagnitude(potential,c,i,j,dr,dz)));
    Real const center=(c1(i,j,k)-2.*ct(i,j,k))+c0(i,j,k);
    Real const center_eval=Mul(Gamma(4),Add(Abs(c1(i,j,k)),Add(Mul(2.,Abs(ct(i,j,k))),Abs(c0(i,j,k)))));
    Real const sum=Sum4(Abs(c1(i,j,k)),Abs(i1(i,j,k)),Abs(j1(i,j,k)),Abs(d1(i,j,k)));
    Real const bound=Add(Add(base(i,j,k),bc),Add(ion_bound,Mul(Gamma(4),sum)));
    Real const error=((c1(i,j,k)-i1(i,j,k))-j1(i,j,k))-d1(i,j,k);
    Real const jd=std::abs(j1(i,j,k)-je(i,j,k)),dchange=std::abs(d1(i,j,k)-de(i,j,k));
    Real const ion_error=std::abs(i1(i,j,k)-iv(i,j,k));
    bool const finite=Normal(bc)&&Normal(center)&&Normal(center_eval)&&Normal(bound)&&Normal(error)&&Normal(jd)&&Normal(dchange)&&Normal(ion_error);
    int issue=!finite?1:((std::abs(center)>Add(bc,center_eval)?2:0)|
        (jd>Mul(2.,jr(i,j,k))?4:0)|(dchange>Mul(2.,dd(i,j,k))?8:0)|
        (ion_error>ion_bound?16:0)|(std::abs(error)>bound?32:0));
    return {issue,std::abs(error),bound?std::abs(error)/bound:0.,std::abs(center),bound,1};
   });
  }
  auto value=data.value();bad=std::max(bad,amrex::get<0>(value));peak=std::max(peak,amrex::get<1>(value));ratio=std::max(ratio,amrex::get<2>(value));center_peak=std::max(center_peak,amrex::get<3>(value));bound_peak=std::max(bound_peak,amrex::get<4>(value));rows+=amrex::get<5>(value);
 }
 amrex::ParallelDescriptor::ReduceIntMax(bad);amrex::ParallelDescriptor::ReduceRealMax(peak);amrex::ParallelDescriptor::ReduceRealMax(ratio);amrex::ParallelDescriptor::ReduceRealMax(center_peak);amrex::ParallelDescriptor::ReduceRealMax(bound_peak);amrex::ParallelDescriptor::ReduceLongSum(rows);
 amrex::Print()<<std::setprecision(17)<<"OWNED_ENDPOINT_AMPERE verify bad="<<bad<<" free_local_rows_sum="<<rows<<" defect_A_per_m2="<<peak<<" maximum_row_bound_A_per_m2="<<bound_peak<<" maximum_row_ratio="<<ratio<<" affine_current_gap_A_per_m2="<<center_peak<<" ion_agreement_bound_A_per_m2="<<ion_bound<<"\n";
 r.verified=bad==0&&rows>0;return r.verified;
#endif
}

struct NativeEndpointAmpereOrigin::Impl {
 WarpX* simulation=nullptr;
 amrex::Geometry geometry;
 std::uint64_t generation=0,epoch=0;
 Real time=0.,clock=0.;int step=0;
 NativeCurrentTolerance tolerance;
 std::array<int,4*AMREX_SPACEDIM> boundary{};
 int shape=0,gather=0,centering_r=0,centering_z=0;
 Vector potential,potential_old,magnetic,static_B,current,ion,electron,displacement,bound;
 std::array<amrex::iMultiFab,3> free;
 bool available=false;
 NativeEndpointAmpereCheck CheckRows(ablastr::fields::VectorField const&,
     ablastr::fields::ConstVectorField const&,ablastr::fields::ConstVectorField const&,
     ablastr::fields::ConstVectorField const&,ablastr::fields::ConstVectorField const&) const;
 explicit Impl(WarpX& sim):simulation(&sim),geometry(sim.Geom(0)),clock(sim.gett_new(0)),
     step(sim.getistep(0)),shape(WarpX::nox),gather(int(WarpX::field_gathering_algo)),
     centering_r(WarpX::field_centering_nox),centering_z(WarpX::field_centering_noz) {
  for(int d=0;d<AMREX_SPACEDIM;++d){boundary[d]=int(WarpX::field_boundary_lo[d]);
   boundary[AMREX_SPACEDIM+d]=int(WarpX::field_boundary_hi[d]);
   boundary[2*AMREX_SPACEDIM+d]=int(WarpX::particle_boundary_lo[d]);
   boundary[3*AMREX_SPACEDIM+d]=int(WarpX::particle_boundary_hi[d]);}
 }
 bool Context(WarpX& sim) const {
  bool ok=available&&simulation==&sim&&sim.gett_new(0)==clock&&sim.getistep(0)==step&&
      geometry.Domain()==sim.Geom(0).Domain()&&geometry.Coord()==sim.Geom(0).Coord()&&
      shape==WarpX::nox&&gather==int(WarpX::field_gathering_algo)&&
      centering_r==WarpX::field_centering_nox&&centering_z==WarpX::field_centering_noz;
  for(int d=0;d<AMREX_SPACEDIM;++d)ok=ok&&
      geometry.ProbLo(d)==sim.Geom(0).ProbLo(d)&&geometry.ProbHi(d)==sim.Geom(0).ProbHi(d)&&
      geometry.isPeriodic(d)==sim.Geom(0).isPeriodic(d)&&
      boundary[d]==int(WarpX::field_boundary_lo[d])&&
      boundary[AMREX_SPACEDIM+d]==int(WarpX::field_boundary_hi[d])&&
      boundary[2*AMREX_SPACEDIM+d]==int(WarpX::particle_boundary_lo[d])&&
      boundary[3*AMREX_SPACEDIM+d]==int(WarpX::particle_boundary_hi[d]);
  return owned_ampere_detail::All(ok);
 }
 static bool Same(Vector const& saved,ablastr::fields::VectorField const& live) {
  bool ok=true;
  for(int c=0;c<3;++c)ok=ok&&live[c]&&saved[c].boxArray()==live[c]->boxArray()&&
      saved[c].DistributionMap()==live[c]->DistributionMap()&&saved[c].nComp()==live[c]->nComp()&&
      saved[c].nGrowVect()==live[c]->nGrowVect();
  if(!owned_ampere_detail::All(ok))return false;
  amrex::Gpu::DeviceScalar<int> mismatch(0);auto* bad=mismatch.dataPtr();
  for(int c=0;c<3;++c)for(amrex::MFIter it(saved[c]);it.isValid();++it){
   auto const* a=reinterpret_cast<unsigned char const*>(saved[c][it].dataPtr());
   auto const* b=reinterpret_cast<unsigned char const*>((*live[c])[it].dataPtr());
   auto const bytes=saved[c][it].nBytes();
   amrex::For(bytes,[=]AMREX_GPU_DEVICE(amrex::Long n){
    if(a[n]!=b[n])amrex::Gpu::Atomic::Exch(bad,1);
   });
  }
  return owned_ampere_detail::All(mismatch.dataValue()==0);
 }
};
NativeEndpointAmpereOrigin::NativeEndpointAmpereOrigin(std::unique_ptr<Impl> p):m_impl(std::move(p)){}
NativeEndpointAmpereOrigin::~NativeEndpointAmpereOrigin()=default;
NativeEndpointAmpereOriginView NativeEndpointAmpereOrigin::View() const {
 NativeEndpointAmpereOriginView v;auto const& r=*m_impl;if(!r.available)return v;
 v.available=true;v.generation=r.generation;v.epoch=r.epoch;v.time=r.time;
 for(int c=0;c<3;++c){v.current[c]=&r.current[c];v.ion[c]=&r.ion[c];v.electron[c]=&r.electron[c];
  v.displacement[c]=&r.displacement[c];v.bound[c]=&r.bound[c];v.free_rows[c]=&r.free[c];}
 return v;
}
bool NativeEndpointAmpereOrigin::Matches(WarpX& sim,Real time,std::uint64_t epoch) const {
 auto const& r=*m_impl;
 if(!owned_ampere_detail::All(r.available&&r.time==time&&r.epoch==epoch)||!r.Context(sim))return false;
 bool same=true;
 for(auto const& item:std::initializer_list<std::pair<Vector const*,char const*>>{
     {&r.potential,"hybrid_A_fp"},{&r.potential_old,"hybrid_A_old_fp"},
     {&r.magnetic,"Bfield_fp"},{&r.static_B,"hybrid_B_static_fp"},
     {&r.electron,current_name},{&r.displacement,displacement_old_name}}){
  bool const value=Impl::Same(*item.first,sim.m_fields.get_alldirs(item.second,0));same=value&&same;
 }
 return same;
}
NativeEndpointAmpereCheck NativeEndpointAmpereOrigin::Impl::CheckRows(
 ablastr::fields::VectorField const& A,ablastr::fields::ConstVectorField const& C,
 ablastr::fields::ConstVectorField const& I,ablastr::fields::ConstVectorField const& J,
 ablastr::fields::ConstVectorField const& D) const {
 using namespace owned_ampere_detail;
 NativeEndpointAmpereCheck result;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(A,C,I,J,D);return result;
#else
 auto const& r=*this;
 NativeCurrentTolerance const tolerance=r.tolerance;
 result.absolute_tolerance=tolerance.absolute;
 bool write_maps=false;
 amrex::ParmParse("endpoint_diagnostic").query("current_check_maps",write_maps);
 std::array<amrex::MultiFab,3> maps;
 if(write_maps)for(int c=0;c<3;++c){
  maps[c].define(C[c]->boxArray(),C[c]->DistributionMap(),9,0);maps[c].setVal(0.);
 }
 int bad=0;Real peak=0.,bound_peak=0.,ratio=0.;amrex::Long rows=0;
 Real const dr=r.geometry.CellSize(0),dz=r.geometry.CellSize(1);
 for(int c=0;c<3;++c){
  amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,
      amrex::ReduceOpMax,amrex::ReduceOpSum> op;
  amrex::ReduceData<int,Real,Real,Real,amrex::Long> data(op);using T=decltype(data)::Type;
  for(amrex::MFIter it(*C[c]);it.isValid();++it){
   Arrays potential{};for(int d=0;d<3;++d)potential[d]=A[d]->const_array(it);
   auto mask=r.free[c].const_array(it);auto a=C[c]->const_array(it),b=I[c]->const_array(it),
       jv=J[c]->const_array(it),dv=D[c]->const_array(it);
   auto output=write_maps?maps[c].array(it):amrex::Array4<Real>{};
   op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{
    if(write_maps){
     output(i,j,k,0)=a(i,j,k);output(i,j,k,1)=b(i,j,k);
     output(i,j,k,2)=jv(i,j,k);output(i,j,k,3)=dv(i,j,k);
     output(i,j,k,7)=mask(i,j,k);
    }
    if(!mask(i,j,k))return {0,0.,0.,0.,0};
    // Same absolute-plus-arithmetic closure as incoming field capture.
    Real const scale=Sum4(Abs(a(i,j,k)),Abs(b(i,j,k)),Abs(jv(i,j,k)),Abs(dv(i,j,k)));
    Real const roundoff=Mul(Gamma(128),Add(CurrentMagnitude(potential,c,i,j,dr,dz),scale));
    Real const limit=tolerance.Bound(roundoff);
    Real const error=((a(i,j,k)-b(i,j,k))-jv(i,j,k))-dv(i,j,k);
    bool const finite=Normal(limit)&&Normal(error);
    if(write_maps){
     output(i,j,k,4)=error;output(i,j,k,5)=limit;
     output(i,j,k,6)=limit?std::abs(error)/limit:0.;
     output(i,j,k,8)=CurrentMagnitude(potential,c,i,j,dr,dz);
    }
    return {!finite||!tolerance.Accepts(error,roundoff)?1:0,std::abs(error),limit,
            limit?std::abs(error)/limit:0.,1};
   });
  }
  auto const v=data.value();bad=std::max(bad,amrex::get<0>(v));peak=std::max(peak,amrex::get<1>(v));
  bound_peak=std::max(bound_peak,amrex::get<2>(v));ratio=std::max(ratio,amrex::get<3>(v));rows+=amrex::get<4>(v);
 }
 amrex::ParallelDescriptor::ReduceIntMax(bad);amrex::ParallelDescriptor::ReduceRealMax(peak);
 amrex::ParallelDescriptor::ReduceRealMax(bound_peak);amrex::ParallelDescriptor::ReduceRealMax(ratio);
 amrex::ParallelDescriptor::ReduceLongSum(rows);
 if(write_maps&&bad){
  static std::uint64_t sequence=0;
  std::string const prefix="NATIVE_CURRENT_CHECK_"+std::to_string(sequence++);
  for(int c=0;c<3;++c){
   amrex::VisMF::Write(maps[c],prefix+"_c"+std::to_string(c));
   amrex::VisMF::Write(*A[c],prefix+"_A"+std::to_string(c));
  }
  amrex::Print()<<"Native current map "<<prefix<<": rows="<<rows<<" error="<<peak
      <<" maximum_row_ratio="<<ratio<<" columns=C,Ji,Je,D,error,bound,ratio,free,curl_operand_scale\n";
 }
 result.valid=bad==0&&rows>0;result.rows=rows;result.maximum_error=peak;
 result.maximum_bound=bound_peak;result.maximum_ratio=ratio;return result;
#endif
}
NativeEndpointAmpereCheck NativeEndpointAmpereOrigin::CheckCurrent(WarpX& sim) const {
 using namespace owned_ampere_detail;
 NativeEndpointAmpereCheck result;auto const& r=*m_impl;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(sim);return result;
#else
 if(!r.Context(sim)||!All(ArithmeticSupported(sim)))return result;
 auto const A=sim.m_fields.get_alldirs("hybrid_A_fp",0),B=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 auto const J=sim.m_fields.get_alldirs(current_name,0),D=sim.m_fields.get_alldirs(displacement_old_name,0);
 if(!Impl::Same(r.static_B,B))return result;
 bool layout=true;
 for(int c=0;c<3;++c)for(auto const* f:{A[c],J[c],D[c]})layout=layout&&f&&f->nComp()==1&&
      f->boxArray()==r.free[c].boxArray()&&f->DistributionMap()==r.free[c].DistributionMap()&&
      f->nGrowVect().allGE(1);
 if(!All(layout))return result;
 auto C=Allocate(J),I=Allocate(J);for(auto* v:{&C,&I})for(auto& f:*v)f.setVal(0.);
 warpx::darwin::CalculateNativeSplitAmpere(sim,A,B,Pointers(C));
 bool const ion=warpx::particles::DepositNativeInstantaneousIonCurrent(sim,
     warpx::particles::InstantaneousIonState::Current,Pointers(I));
 if(!All(ion))return result;
 return r.CheckRows(A,{&C[0],&C[1],&C[2]},
     {&I[0],&I[1],&I[2]},{J[0],J[1],J[2]},{D[0],D[1],D[2]});
#endif
}
NativeEndpointAmpereCheck NativeEndpointAmpereOrigin::CheckCandidateCurrent(
 WarpX& sim,ablastr::fields::VectorField const& A,
 ablastr::fields::ConstVectorField const& I,ablastr::fields::ConstVectorField const& J,
 ablastr::fields::ConstVectorField const& D) const {
 using namespace owned_ampere_detail;
 NativeEndpointAmpereCheck result;auto const& r=*m_impl;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(sim,A,I,J,D);return result;
#else
 if(!r.Context(sim)||!All(ArithmeticSupported(sim)))return result;
 auto const B=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 if(!Impl::Same(r.static_B,B))return result;
 bool layout=true;
 for(int c=0;c<3;++c){
  layout=layout&&A[c]&&A[c]->nComp()==1&&A[c]->boxArray()==r.free[c].boxArray()&&
      A[c]->DistributionMap()==r.free[c].DistributionMap()&&A[c]->nGrowVect().allGE(1);
  for(auto const* field:{I[c],J[c],D[c]})layout=layout&&field&&field->nComp()==1&&
      field->boxArray()==r.free[c].boxArray()&&field->DistributionMap()==r.free[c].DistributionMap()&&
      field->nGrowVect().allGE(1);
 }
 if(!All(layout))return result;
 Vector C;for(int c=0;c<3;++c){
  C[c].define(J[c]->boxArray(),J[c]->DistributionMap(),1,J[c]->nGrowVect());C[c].setVal(0.);
 }
 warpx::darwin::CalculateNativeSplitAmpere(sim,A,B,Pointers(C));
 return r.CheckRows(A,{&C[0],&C[1],&C[2]},I,J,D);
#endif
}

std::shared_ptr<NativeEndpointAmpereOrigin const> NativeOwnedEndpointAmpere::RetainVerifiedOrigin(
 Real time,std::uint64_t epoch,std::uint64_t generation,Masks const& P,Masks const& V) {
 using namespace owned_ampere_detail;auto& r=*m_impl;auto& sim=r.sim;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(time,epoch,generation,P,V);return {};
#else
 bool ready=r.verified&&generation>0&&generation<std::numeric_limits<std::uint64_t>::max()&&
     sim.getistep(0)>=0&&epoch==std::uint64_t(sim.getistep(0))+1&&time==sim.gett_new(0)+r.dt;
 if(!All(ready)||!r.Layout(P,V))return {};
 // One capture-only recheck through the unchanged original verification body.
 // A failed repeated verification revokes the ability to export provenance.
 r.verified=false;if(!Verify(P,V))return {};
 auto p=std::make_unique<NativeEndpointAmpereOrigin::Impl>(sim);
 p->tolerance=r.tolerance;
 p->time=time;p->epoch=epoch;p->generation=generation;
 auto const A=sim.m_fields.get_alldirs("hybrid_A_fp",0),B=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 auto const J=sim.m_fields.get_alldirs(current_name,0),D=sim.m_fields.get_alldirs(displacement_old_name,0);
 p->potential=Clone(A);p->potential_old=Clone(sim.m_fields.get_alldirs("hybrid_A_old_fp",0));
 p->static_B=Clone(B);p->magnetic=Clone(sim.m_fields.get_alldirs("Bfield_fp",0));
 p->electron=Clone(J);p->displacement=Clone(D);p->current=Allocate(J);p->ion=Allocate(J);p->bound=Allocate(J);
 for(auto* array:{&p->current,&p->ion,&p->bound})for(auto& f:*array)f.setVal(0.);
 warpx::darwin::CalculateNativeSplitAmpere(sim,A,B,Pointers(p->current));
 if(!All(warpx::particles::DepositNativeInstantaneousIonCurrent(sim,
     warpx::particles::InstantaneousIonState::Current,Pointers(p->ion))))return {};
 NativeCurrentTolerance const tolerance=p->tolerance;
 Real const dr=sim.Geom(0).CellSize(0),dz=sim.Geom(0).CellSize(1);
 for(int c=0;c<3;++c){
  p->free[c].define(P[c]->boxArray(),P[c]->DistributionMap(),1,0);p->free[c].setVal(0);
  for(amrex::MFIter it(p->free[c]);it.isValid();++it){
   Arrays potential{};for(int d=0;d<3;++d)potential[d]=A[d]->const_array(it);
   auto out=p->free[c].array(it);auto a=P[c]->const_array(it),b=V[c]->const_array(it);
   auto limit=p->bound[c].array(it);auto cf=p->current[c].const_array(it),ji=p->ion[c].const_array(it),
       je=J[c]->const_array(it),dd=D[c]->const_array(it);
   amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){
    out(i,j,k)=a(i,j,k)||b(i,j,k);
    if(out(i,j,k))limit(i,j,k)=tolerance.Bound(Mul(Gamma(128),Add(CurrentMagnitude(potential,c,i,j,dr,dz),
        Sum4(Abs(cf(i,j,k)),Abs(ji(i,j,k)),Abs(je(i,j,k)),Abs(dd(i,j,k))))));
   });
  }
 }
 p->available=true;
 auto origin=std::shared_ptr<NativeEndpointAmpereOrigin const>(new NativeEndpointAmpereOrigin(std::move(p)));
 if(!origin->Matches(sim,time,epoch)||!origin->CheckCurrent(sim).valid)return {};
 return origin;
#endif
}

std::shared_ptr<NativeEndpointAmpereOrigin const> NativeEndpointAmpereOrigin::CaptureAccepted(
 WarpX& sim,Real time,std::uint64_t epoch,std::uint64_t generation) {
 using namespace owned_ampere_detail;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(sim,time,epoch,generation);return {};
#else
 // In this periodic RZ/axis/PEC scope, Joint's P union V is exactly every
 // valid row except tangential PEC and odd theta-axis rows. Normal wall
 // current remains free. This is the original CheckRows predicate/budget.
 bool ready=ArithmeticSupported(sim)&&sim.getistep(0)>=0&&time==sim.gett_new(0)&&
     epoch==std::uint64_t(sim.getistep(0))&&generation>0&&
     generation<std::numeric_limits<std::uint64_t>::max()&&
     sim.finestLevel()==0&&sim.Geom(0).ProbLo(0)==0.&&sim.Geom(0).isPeriodic(1)&&
     !sim.Geom(0).isPeriodic(0)&&WarpX::field_boundary_lo[0]==FieldBoundaryType::None&&
     WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC&&
     !sim.get_pointer_HybridPICModel()->m_darwin_checkpoint_restored;
 if(!All(ready))return {};
 auto const fixed=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 for(int c=0;c<3;++c){
  ready=ready&&fixed[c]->is_finite(0,1,fixed[c]->nGrow());
  Real const lo=fixed[c]->min(0,fixed[c]->nGrow()),hi=fixed[c]->max(0,fixed[c]->nGrow());
  ready=ready&&Normal(lo)&&lo==hi&&(c==2||lo==0.);
 }
 if(!All(ready))return {};
 auto p=std::make_unique<NativeEndpointAmpereOrigin::Impl>(sim);
 p->tolerance.absolute=NativeEndpointCurrentAbsoluteTolerance(sim);
 p->time=time;p->epoch=epoch;p->generation=generation;
 auto const A=sim.m_fields.get_alldirs("hybrid_A_fp",0),B=sim.m_fields.get_alldirs("hybrid_B_static_fp",0);
 auto const J=sim.m_fields.get_alldirs(current_name,0),D=sim.m_fields.get_alldirs(displacement_old_name,0);
 p->potential=Clone(A);p->potential_old=Clone(sim.m_fields.get_alldirs("hybrid_A_old_fp",0));
 p->static_B=Clone(B);p->magnetic=Clone(sim.m_fields.get_alldirs("Bfield_fp",0));
 p->electron=Clone(J);p->displacement=Clone(D);p->current=Allocate(J);p->ion=Allocate(J);p->bound=Allocate(J);
 for(auto* array:{&p->current,&p->ion,&p->bound})for(auto& f:*array)f.setVal(0.);
 warpx::darwin::CalculateNativeSplitAmpere(sim,A,B,Pointers(p->current));
 if(!All(warpx::particles::DepositNativeInstantaneousIonCurrent(sim,
     warpx::particles::InstantaneousIonState::Current,Pointers(p->ion))))return {};
 NativeCurrentTolerance const tolerance=p->tolerance;
 Real const dr=sim.Geom(0).CellSize(0),dz=sim.Geom(0).CellSize(1);
 for(int c=0;c<3;++c){
  p->free[c].define(J[c]->boxArray(),J[c]->DistributionMap(),1,0);p->free[c].setVal(0);
  FillAcceptedOriginRows(p->free[c],p->bound[c],A,p->current[c],p->ion[c],*J[c],*D[c],
      c,sim.Geom(0).Domain().bigEnd(0)+1,dr,dz,tolerance);
 }
 p->available=true;
 auto origin=std::shared_ptr<NativeEndpointAmpereOrigin const>(new NativeEndpointAmpereOrigin(std::move(p)));
 if(!origin->Matches(sim,time,epoch)||!origin->CheckCurrent(sim).valid)return {};
 return origin;
#endif
}

NativeEndpointMagneticWork NativeEndpointAmpereOrigin::FieldMagneticWork(
 WarpX& sim,ablastr::fields::ConstVectorField const& B0) const {
 using namespace owned_ampere_detail;
 NativeEndpointMagneticWork result;auto const& r=*m_impl;
#if !defined(WARPX_DIM_RZ)
 amrex::ignore_unused(sim,B0);return result;
#else
 if(!Matches(sim,r.time,r.epoch))return result;
 auto const A=sim.m_fields.get_alldirs("hybrid_A_fp",0),B=sim.m_fields.get_alldirs("Bfield_fp",0);
 auto const J=sim.m_fields.get_alldirs(current_name,0);
 bool valid=true;for(int c=0;c<3;++c)valid=valid&&B0[c]&&B0[c]->nComp()==1&&
     B0[c]->boxArray()==B[c]->boxArray()&&B0[c]->DistributionMap()==B[c]->DistributionMap()&&
     B0[c]->nGrowVect()==B[c]->nGrowVect();
 if(!All(valid))return result;
 auto deltaA=Allocate(A),deltaB=Allocate(B),C0=Allocate(J),C1=Allocate(J);
 for(int c=0;c<3;++c){
  MF::LinComb(deltaA[c],1.,r.potential[c],0,-1.,r.potential_old[c],0,0,1,deltaA[c].nGrowVect());
  deltaB[c].setVal(0.);C0[c].setVal(0.);C1[c].setVal(0.);
 }
 auto av=Pointers(deltaA),bv=Pointers(deltaB);auto oldB=Allocate(B);
 for(int c=0;c<3;++c)MF::Copy(oldB[c],*B0[c],0,0,1,oldB[c].nGrowVect());
 auto oldbv=Pointers(oldB),c0v=Pointers(C0),c1v=Pointers(C1);
 sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(bv,av,sim.GetEBUpdateBFlag()[0],0,
     DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
 sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(c0v,oldbv,sim.GetEBUpdateEFlag()[0],0);
 sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(c1v,B,sim.GetEBUpdateEFlag()[0],0);
 auto const geometry=sim.get_pointer_HybridPICModel()->ElectronThermalGeometry();
 Real const dr=geometry.CellSize(0),dz=geometry.CellSize(1);
 Real values[8]={};int bad=0;amrex::Long rows=0;Real row_error=0.,row_bound=0.;
 for(int c=0;c<3;++c){
  auto owner=B[c]->OwnerMask(geometry.periodicity());auto vol=MakeQdsmcVolumeElement(geometry,B[c]->ixType());
  amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
      amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpMax,
      amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpSum> op;
  amrex::ReduceData<Real,Real,Real,Real,Real,Real,int,Real,Real,amrex::Long> data(op);
  using T=decltype(data)::Type;
  for(amrex::MFIter it(*B[c]);it.isValid();++it){
   auto own=owner->const_array(it);auto old=B0[c]->const_array(it),now=B[c]->const_array(it),db=deltaB[c].const_array(it);
   Arrays a0{},a1{};for(int d=0;d<3;++d){a0[d]=r.potential_old[d].const_array(it);a1[d]=r.potential[d].const_array(it);}
   op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{
    if(!own(i,j,k))return {0.,0.,0.,0.,0.,0.,0,0.,0.,0};
    Real const b0=old(i,j,k),b1=now(i,j,k),d=db(i,j,k),v=vol(i,j,k)/PhysConst::mu0;
    Real const mid=.5*(b0+b1),change=b1-b0,remainder=change-d;
    Real const limit=Mul(Gamma(128),Add(Add(Abs(b0),Abs(b1)),
        Add(CurlMagnitude(a0,c,i,j,dr,dz),CurlMagnitude(a1,c,i,j,dr,dz))));
    Real const e0=.5*b0*b0*v,e1=.5*b1*b1*v,delta=mid*change*v;
    Real const operator_work=mid*d*v,material=mid*remainder*v,material_bound=Mul(Mul(Abs(mid),limit),Abs(v));
    bool const finite=Normal(limit)&&Normal(e0)&&Normal(e1)&&Normal(delta)&&Normal(operator_work)&&Normal(material)&&Normal(material_bound);
    return {e0,e1,delta,operator_work,material,material_bound,
        !finite||std::abs(remainder)>limit?1:0,std::abs(remainder),limit,1};
   });
  }
  auto x=data.value();values[0]+=amrex::get<0>(x);values[1]+=amrex::get<1>(x);values[2]+=amrex::get<2>(x);
  values[3]+=amrex::get<3>(x);values[4]+=amrex::get<4>(x);values[5]+=amrex::get<5>(x);
  bad=std::max(bad,amrex::get<6>(x));row_error=std::max(row_error,amrex::get<7>(x));
  row_bound=std::max(row_bound,amrex::get<8>(x));rows+=amrex::get<9>(x);
  auto co=C0[c].OwnerMask(geometry.periodicity());auto cv=MakeQdsmcVolumeElement(geometry,C0[c].ixType());
  amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> cop;
  amrex::ReduceData<Real,Real> cd(cop);using CT=decltype(cd)::Type;
  for(amrex::MFIter it(C0[c]);it.isValid();++it){auto own=co->const_array(it);auto a=C0[c].const_array(it),b=C1[c].const_array(it),d=deltaA[c].const_array(it);
   cop.eval(it.validbox(),cd,[=]AMREX_GPU_DEVICE(int i,int j,int k)->CT{
    Real const work=own(i,j,k)?.5*(a(i,j,k)+b(i,j,k))*d(i,j,k)*cv(i,j,k):0.;
    return {work,std::abs(work)};
   });
  }
  auto cx=cd.value();values[6]+=amrex::get<0>(cx);values[7]+=amrex::get<1>(cx);
 }
 amrex::ParallelDescriptor::ReduceRealSum(values,8);amrex::ParallelDescriptor::ReduceIntMax(bad);
 amrex::ParallelDescriptor::ReduceRealMax(row_error);amrex::ParallelDescriptor::ReduceRealMax(row_bound);
 amrex::ParallelDescriptor::ReduceLongSum(rows);
 Real const ops=128.+16.*Real(rows),e=ops*std::numeric_limits<Real>::epsilon();
 Real const positive=values[0]+values[1]+std::abs(values[2])+std::abs(values[3])+std::abs(values[4])+values[7];
 Real const arithmetic=e<.01?Up(4.*e/(1.-e)*positive):std::numeric_limits<Real>::infinity();
 result.initial=values[0];result.final=values[1];result.change=values[2];result.curl_work=values[6];
 result.curl_transfer=values[3]-values[6];result.represented_faraday_work=values[4];
 result.represented_faraday_bound=Up(values[5]*(1.+4.*e/(1.-e)));
 result.identity_error=values[2]-(values[3]+values[4]);result.arithmetic_bound=arithmetic;
 result.faraday_error=row_error;result.faraday_bound=row_bound;result.rows=rows;
 result.valid=bad==0&&rows>0&&Normal(arithmetic)&&std::abs(result.identity_error)<=arithmetic&&
     std::abs(values[4])<=result.represented_faraday_bound+arithmetic;
 return result;
#endif
}

static NativeEndpointResult ConstrainNativeEndpointFieldImpl(WarpX& sim,Real time,bool initial,
    std::function<void()>const&curl_rate,bool recoverable,View const* exact_transverse=nullptr){
 NativeEndpointResult result;
 if(NativeCorrelatedIncrementEnabled())ValidateCorrelatedStaticDrive(sim);
 EnsureCompanion(sim);Startup s(sim,time,curl_rate,recoverable);
 bool const audit=EndpointAuditEnabled();
 std::map<std::string,std::uint64_t> before_fields;
 std::uint64_t before_particles=0;
 Real before_energy=0.;
 if(audit) {
     before_fields=FieldHashes(sim);
     before_particles=ParticleHash(sim);
     before_energy=ElectricEnergy(sim,s.E);
 }
 warpx::implicit::FieldRollback saved;saved.Capture(sim.m_fields);
 // Pure endpoint C is independent of the interval displacement in the delivered Jp.
 if(s.coil_current)s.coil_current->CalculatePlasmaCurrent(Pointers(s.B0),time);
 else if(!warpx::darwin::TryCalculateNativeSplitDarwinAmpere(sim))
     s.model.CalculatePlasmaCurrent(Pointers(s.B0),sim.GetEBUpdateEFlag()[0],0);
 for(auto* f:s.C) PMCImages(*f);
 Copy(Pointers(s.C0),s.C);
 if(!s.coil_current)s.Deposit(false);
 auto instantaneous=Clone(Pointers(s.J0));
 // Use the identical point-map arithmetic for initialization, every trial's
 // saved Old state, and actual accepted endpoint. Global/local grid origins
 // are algebraically equal but an O(eps*I) mismatch is amplified by1/h.
 if(NativeIonQuadratureEnabled()) {
    AMREX_ALWAYS_ASSERT(warpx::particles::DepositNativeInstantaneousIonCurrent(
        sim,warpx::particles::InstantaneousIonState::Current,Pointers(instantaneous)));
 }
 Copy(s.Ji,Pointers(instantaneous));Copy(s.B,Pointers(s.B0));
 auto companion=sim.m_fields.get_alldirs(current_name,0);
 auto endpoint_je=warpx::darwin::NativePECPlasmaEnabled()?Allocate(companion):Allocate(s.Ji);
 for(int c=0;c<3;++c){if(initial)MF::LinComb(endpoint_je[c],1.,s.C0[c],0,-1.,instantaneous[c],0,0,1,endpoint_je[c].nGrowVect());else MF::Copy(endpoint_je[c],*companion[c],0,0,1,endpoint_je[c].nGrowVect());
  MF::LinComb(*s.C[c],1.,instantaneous[c],0,1.,endpoint_je[c],0,0,1,warpx::darwin::NativePECPlasmaEnabled()?amrex::IntVect(1):s.C[c]->nGrowVect());}
 auto physical_ohm_current=CurrentClone(s.C);
 sim.m_fields.get("hybrid_E_inertial_nodal",0)->setVal(0.);
 if(s.model.UseCompatibleYeeInertia())
    for(auto* f:sim.m_fields.get_alldirs("hybrid_E_inertial_fp",0))f->setVal(0.);
 if(s.smooth_force && s.model.HasResistivity()) {
    EndpointRequire(warpx::darwin::CaptureNativeInstantaneousForce(
        sim,s.density,Pointers(s.F0),Pointers(s.nores),s.dissipation.get()),
        NativeEndpointFailure::PhysicalConstraint,recoverable,
        "Native endpoint full/nores force is nonfinite");
    s.force_ready=true;
 } else {
    auto ph=warpx::darwin::NativePECPlasmaEnabled()?Pointers(s.pressure_hall):View{};
    s.model.HybridPICSolveE(s.E,s.Ji,s.B,s.density,sim.GetEBUpdateEFlag()[0],0,false,true,s.dissipation.get(),nullptr,nullptr,warpx::darwin::NativePECPlasmaEnabled()?&ph:nullptr);Copy(Pointers(s.F0),s.E);
    if(s.coil_current)warpx::darwin::SetNativePECTangential(sim,Pointers(s.F0),&ph,1.);
 }
 // The chosen longitudinal source establishes the affine inertial target.
 // F0 is the actual non-inertial native Ohm field with the same dissipation
 // context as the corrected ion force; it is not reconstructed from aliases.
 ablastr::fields::MultiLevelScalarField endpoint_density{&s.density};
 ablastr::fields::MultiLevelVectorField full_source{Pointers(s.F0)};
 s.model.ComputeDarwinELong(endpoint_density,time,true,
     NativeFullOhmLongitudinalEnabled() ? &full_source : nullptr);
 auto const pressure_long=sim.m_fields.get_alldirs("hybrid_E_long_fp",0);
 for(int c=0;c<3;++c)MF::LinComb(s.target[c],1.,s.EL0[c],0,-1.,*pressure_long[c],0,0,1,0);
 Copy(pressure_long,Pointers(s.EL0));
 if(s.coil_current) {
    // The affine device response must precede the ion rate that consumes it.
    Copy(s.E,Pointers(s.zero));s.GatherImages(false);s.CurlRate(true);
    Copy(Pointers(s.curl),s.C);s.Deposit(false);s.RatesToRaw();
 } else {
 Copy(s.E,Pointers(s.EL0));s.Deposit(false);
 if(NativePrescribedDriveEnabled()||s.circuit_rate) {
    Copy(s.E,Pointers(s.zero));s.GatherImages();s.CurlRate(true);
    Copy(Pointers(s.curl),s.C);s.RatesToRaw();
 } else if(s.model.UseCompatibleYeeInertia()) {
    s.MapYee(Pointers(s.Jrate),Pointers(s.raw),-1./(PhysConst::epsilon_0*s.omega2));
 } else {
    s.Node(Pointers(s.Jrate),s.nodal);s.nodal.mult(-1.);s.MapNode(s.nodal,Pointers(s.raw),true);
 }
 }
 s.Project(Pointers(s.raw),Pointers(s.projected),true);
 Startup::Ops ops{s};auto rhs=ops.makeVecRHS();for(int c=0;c<3;++c){auto&b=*rhs.getArrayVec()[0][c];MF::LinComb(b,1.,s.F0[c],0,-1.,s.EL0[c],0,0,1,0);MF::Add(b,s.projected[c],0,0,1,0);}
 Copy(s.E,rhs.getArrayVec()[0],0);s.GatherImages(false);Copy(rhs.getArrayVec()[0],s.E,0);
 Real oracle_aba=0.,oracle_linearity=0.,oracle_zero=0.;bool run_oracles=false;
 amrex::ParmParse("endpoint_diagnostic").query("oracles",run_oracles);
 if(run_oracles){
  auto a=ops.makeVecLHS(),b=ops.makeVecLHS(),ab=ops.makeVecLHS(),fa=ops.makeVecRHS(),fb=ops.makeVecRHS(),again=ops.makeVecRHS(),fab=ops.makeVecRHS();
  auto const dx=sim.Geom(0).CellSizeArray();
  for(int c=0;c<3;++c){auto& af=*a.getArrayVec()[0][c];auto& bf=*b.getArrayVec()[0][c];auto const st=af.ixType().toIntVect();for(amrex::MFIter mfi(af);mfi.isValid();++mfi){auto aa=af.array(mfi),bb=bf.array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){Real const x=(i+.5*(1-st[0]))*dx[0],z=(j+.5*(1-st[1]))*dx[1];aa(i,j,k)=(c+1.)*std::sin(2.3*x+.4*z);bb(i,j,k)=std::cos(.9*x-1.7*z)*(c-.3);});}}
  Copy(s.E,a.getArrayVec()[0],0);s.GatherImages(false);Copy(a.getArrayVec()[0],s.E,0);
  Copy(s.E,b.getArrayVec()[0],0);s.GatherImages(false);Copy(b.getArrayVec()[0],s.E,0);
  ops.apply(fa,a);ops.apply(fb,b);ops.apply(again,a);again.increment(fa,-1.);oracle_aba=again.norm2()/std::max(1.,fa.norm2());
  ab.linComb(1.,a,1.,b);ops.apply(fab,ab);fab.increment(fa,-1.);fab.increment(fb,-1.);oracle_linearity=fab.norm2()/std::max(1.,fa.norm2()+fb.norm2());
  ab.zero();ops.apply(fab,ab);oracle_zero=fab.norm2();
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(oracle_aba<1.e-10&&oracle_linearity<1.e-10&&oracle_zero<1.e-12,"Native startup A/B/A, linearity or homogeneous-zero oracle failed");
 }
 auto solution=ops.makeVecLHS();FlexibleGMRES<WarpXSolverVec,Startup::Ops>solver;solver.define(ops);solver.setMaxIters(1000);solver.setRestartLength(150);solver.setVerbose(0);solver.solve(solution,rhs,1e-12,1e-9);
 auto action=ops.makeVecRHS();ops.apply(action,solution);action.increment(rhs,-1.);Real const norm=action.norm2();result.linear_residual=norm;result.linear_bound=std::max(1e-9,1e-12*rhs.norm2());EndpointRequire(solver.getStatus()==0&&std::isfinite(norm)&&norm<=result.linear_bound,NativeEndpointFailure::LinearSolve,recoverable,"Native constrained endpoint field linear solve failed");
 Copy(s.E,solution.getArrayVec()[0],0);s.GatherImages();
 // Preserve the already completed native root, before full E is assembled.
 // Do not recover it later by subtracting two rounded physical fields.
 std::unique_ptr<std::array<MF,3>> transverse_root;
 if(exact_transverse)transverse_root=std::make_unique<std::array<MF,3>>(Clone(s.E));
 s.CurlRate(true);Copy(Pointers(s.curl),s.C);Copy(s.E,solution.getArrayVec()[0],0);for(int c=0;c<3;++c)MF::Add(*s.E[c],s.EL0[c],0,0,1,0);s.Deposit(false);
 s.RatesToRaw();s.Project(Pointers(s.raw),Pointers(s.projected),true);
 MF inertia(s.nodal.boxArray(),s.nodal.DistributionMap(),3,1);
 if(s.model.UseCompatibleYeeInertia()) {
    for(int c=0;c<3;++c){MF::Subtract(s.curl[c],s.rate_edge[c],0,0,1,0);Images(s.curl[c]);}
    s.MapYee(Pointers(s.curl),Pointers(s.raw),1./(PhysConst::epsilon_0*s.omega2));
    if(warpx::darwin::NativePECPlasmaEnabled()) {
        auto ph=Pointers(s.pressure_hall);
        if(s.coil_current)warpx::darwin::CombineNativePECTangential(sim,Pointers(s.raw),s.E,1.,ph,-1.);
        else warpx::darwin::SetNativePECTangential(sim,Pointers(s.raw),&ph,-1.);
        for(auto& f:s.raw)PMCImages(f);
    }
    s.Node(Pointers(s.raw),inertia);
 } else {
    MF::Copy(inertia,s.nodal,0,0,3,1);MF::Subtract(inertia,s.rate_node,0,0,3,0);s.MapNode(inertia,Pointers(s.raw),true);
    Real const scale=1./(PhysConst::epsilon_0*s.omega2);for(amrex::MFIter mfi(inertia);mfi.isValid();++mfi){auto a=inertia.array(mfi);auto k=s.kappa.const_array(mfi);amrex::ParallelFor(mfi.validbox(),3,[=] AMREX_GPU_DEVICE(int i,int j,int z,int c){a(i,j,z,c)*=k(i,j,z)*scale;});}Images(inertia);
 }
 auto const inertia_yee=Clone(Pointers(s.raw));
 auto accepted=Clone(Pointers(s.F0));for(int c=0;c<3;++c)MF::Add(accepted[c],s.projected[c],0,0,1,0);Copy(s.E,Pointers(accepted),0);s.GatherImages(true,bool(s.circuit_rate));Copy(Pointers(accepted),s.E);
 // Independent native endpoint Ohm and selected longitudinal projection checks.
 Copy(s.C,Pointers(physical_ohm_current));Copy(s.Ji,Pointers(instantaneous));Copy(s.B,Pointers(s.B0));MF::Copy(*sim.m_fields.get("hybrid_E_inertial_nodal",0),inertia,0,0,3,1);
 if(s.model.UseCompatibleYeeInertia())
    for(int c=0;c<3;++c)MF::Copy(*sim.m_fields.get("hybrid_E_inertial_fp",Direction{c},0),inertia_yee[c],0,0,1,1);
 auto fresh_ph=warpx::darwin::NativePECPlasmaEnabled()?Pointers(s.pressure_hall):View{};
 s.model.HybridPICSolveE(s.E,s.Ji,s.B,s.density,sim.GetEBUpdateEFlag()[0],0,false,true,s.dissipation.get(),nullptr,nullptr,s.coil_current?&fresh_ph:nullptr);
 std::unique_ptr<Vector> observed_ohm;
 View measured_ohm=s.E;
 if(s.coil_current) {
    observed_ohm=std::make_unique<Vector>(Clone(s.E));
    measured_ohm=Pointers(*observed_ohm);
    // Independent raw constitutive verification, before imposed E boundaries.
    // The kernel's fresh pressure/Hall capture and the actual registered Yee
    // inertia form a measured physical source; no drive is painted onto it.
    auto registered_inertia=sim.m_fields.get_alldirs("hybrid_E_inertial_fp",0);
    warpx::darwin::CombineNativePECTangential(sim,measured_ohm,fresh_ph,1.,registered_inertia,1.);
 }
 Real ohm_error=0.,el_error=0.;for(int c=0;c<3;++c){MF e(s.E[c]->boxArray(),s.E[c]->DistributionMap(),1,0);MF::LinComb(e,1.,*measured_ohm[c],0,-1.,accepted[c],0,0,1,0);ohm_error=std::max(ohm_error,e.norminf());}
 full_source[0] = measured_ohm;
 s.model.ComputeDarwinELong(endpoint_density,time,true,
     NativeFullOhmLongitudinalEnabled() ? &full_source : nullptr);
 for(int c=0;c<3;++c){MF e(s.EL0[c].boxArray(),s.EL0[c].DistributionMap(),1,0);MF::LinComb(e,1.,*pressure_long[c],0,-1.,s.EL0[c],0,0,1,0);el_error=std::max(el_error,e.norminf());}
 result.ohm_error=ohm_error;result.longitudinal_error=el_error;
 EndpointRequire(std::isfinite(ohm_error)&&std::isfinite(el_error)&&ohm_error<1e-8&&el_error<1e-8,
     NativeEndpointFailure::PhysicalConstraint,recoverable,"Native endpoint physical constraint verification failed");
 // The verified scalar represents the independently accepted EL. Preserve
 // it across the purity restore; FinishFieldUpdate extrapolates EL but its
 // scalar scratch otherwise remains at the midpoint. Do not replace EL.
 std::unique_ptr<MF> accepted_phi;
 if (NativeFullOhmLongitudinalEnabled()) {
     auto const& phi = *sim.m_fields.get("hybrid_phi_darwin_fp", 0);
     accepted_phi = std::make_unique<MF>(phi.boxArray(), phi.DistributionMap(),
         phi.nComp(), phi.nGrowVect());
     MF::Copy(*accepted_phi, phi, 0, 0, phi.nComp(), phi.nGrowVect());
 }
 AMREX_ALWAYS_ASSERT(saved.Restore(sim.m_fields));saved.Discard();
 if(audit) {
     bool restored=FieldHashes(sim)==before_fields&&ParticleHash(sim)==before_particles;amrex::ParallelDescriptor::ReduceBoolAnd(restored);AMREX_ALWAYS_ASSERT_WITH_MESSAGE(restored,"Endpoint diagnostic changed saved fields or physical particles before publication");

     std::string const digest_path="ENDPOINT_PRIMARY_STATE.rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".jsonl";{std::ofstream f(digest_path,std::ios::app);f<<std::setprecision(17)<<"{\"initial\":"<<(initial?"true":"false")<<",\"time\":"<<time<<",\"physical_particles_hash\":\""<<before_particles<<"\",\"field_hashes\":{";bool comma=false;for(auto const&[name,h]:before_fields){if(comma)f<<",";comma=true;f<<"\""<<name<<"\":\""<<h<<"\"";}f<<"}}\n";}
 }
 bool publish=true;amrex::ParmParse("endpoint_diagnostic").query("publish_field",publish);
 if(initial||publish){
     Copy(s.E,Pointers(accepted));s.GatherImages(true,bool(s.circuit_rate));
     if (accepted_phi) {
         auto& phi = *sim.m_fields.get("hybrid_phi_darwin_fp", 0);
         MF::Copy(phi, *accepted_phi, 0, 0, phi.nComp(), phi.nGrowVect());
     }
 }
 // Both causal publication twins refresh this derived inertia scratch so next
 // entry reprojects the SAME accepted EL origin; no current history is changed.
 MF::Copy(*sim.m_fields.get("hybrid_E_inertial_nodal",0),inertia,0,0,3,1);
 if(s.model.UseCompatibleYeeInertia())
    for(int c=0;c<3;++c)MF::Copy(*sim.m_fields.get("hybrid_E_inertial_fp",Direction{c},0),inertia_yee[c],0,0,1,1);
 if(initial){Copy(s.Ji,Pointers(instantaneous));Copy(companion,Pointers(endpoint_je));for(auto name:{"hybrid_Je_n_nodal","hybrid_Je_nm1_nodal","hybrid_Je_theta_nodal"})NodeCurrent(sim,companion,*sim.m_fields.get(name,0));}
 if(initial&&NativeCorrelatedIncrementEnabled())for(auto* f:sim.m_fields.get_alldirs(displacement_old_name,0))f->setVal(0.);

 auto const map_error=CompanionError(sim,companion,"hybrid_Je_n_nodal",recoverable);
 result.companion_error=map_error;
 auto displacement=Clone(Pointers(s.C0));for(int c=0;c<3;++c){MF::Subtract(displacement[c],instantaneous[c],0,0,1,displacement[c].nGrowVect());MF::Subtract(displacement[c],*companion[c],0,0,1,displacement[c].nGrowVect());
    if(warpx::darwin::NativePECPlasmaEnabled())MF::Subtract(displacement[c],*sim.m_fields.get(warpx::darwin::PECWallCurrentName,Direction{c},0),0,0,1,displacement[c].nGrowVect());
 }
 if(audit && NativeCorrelatedIncrementEnabled()){
    auto const saved=sim.m_fields.get_alldirs(displacement_old_name,0);Real remainder=0.,scale=0.;
    for(int c=0;c<3;++c){MF diff(displacement[c].boxArray(),displacement[c].DistributionMap(),1,0);MF::LinComb(diff,1.,displacement[c],0,-1.,*saved[c],0,0,1,0);remainder=std::max(remainder,diff.norminf());scale=std::max(scale,saved[c]->norminf());}
    if(amrex::ParallelDescriptor::IOProcessor()){std::ofstream log("INERTIA_INCREMENT_HISTORY.jsonl",std::ios::app);log<<std::setprecision(17)<<"{\"initial\":"<<(initial?"true":"false")<<",\"time\":"<<time<<",\"stored_D_norm_A_per_m2\":"<<scale<<",\"C_minus_I_minus_Je_minus_D_remainder_A_per_m2\":"<<remainder<<"}\n";}
 }
 // D-gradient decomposition and electric publication energy are observers.
 std::unique_ptr<warpx::darwin::DarwinLongitudinalSchur> dproj;
 std::array<MF const*,3> dg{};
 std::array<Real,3> d_norm{},d_defect{};
 Real energy_jump=0.;
 if(audit) {
     warpx::darwin::LongitudinalSchurOptions dopts;using LBC=warpx::darwin::LongitudinalBoundary;for(int d=0;d<AMREX_SPACEDIM;++d){dopts.lower[d]=sim.Geom(0).isPeriodic(d)?LBC::Periodic:WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC?LBC::PEC:LBC::PMC;dopts.upper[d]=sim.Geom(0).isPeriodic(d)?LBC::Periodic:WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC?LBC::PEC:LBC::PMC;}
    #if defined(WARPX_DIM_RZ)
     dopts.lower[0]=LBC::Axis;
    #endif
     dopts.max_semicoarsening_levels=s.model.m_darwin_poisson_semicoarsening;
     dopts.semicoarsening_direction=s.model.m_darwin_poisson_semicoarsening_direction;
     dopts.verbose=s.model.m_darwin_poisson_verbosity;
     dopts.relative_tolerance=1e-12;dopts.absolute_tolerance=1e-12;dopts.max_iterations=1000;dopts.restart_length=150;
     dproj=std::make_unique<warpx::darwin::DarwinLongitudinalSchur>(sim.Geom(0),sim.boxArray(0),sim.DistributionMap(0),dopts);MF dk(s.kappa.boxArray(),s.kappa.DistributionMap(),1,0);dk.setVal(0.);AMREX_ALWAYS_ASSERT(dproj->Freeze(dk));auto const dr=dproj->Correct({&displacement[0],&displacement[1],&displacement[2]},{&s.zero[0],&s.zero[1],&s.zero[2]});EndpointRequire(dr.converged,NativeEndpointFailure::Projection,recoverable,"Endpoint displacement observer projection failed");dg=dproj->CorrectionField();for(int c=0;c<3;++c){d_norm[c]=displacement[c].norminf();MF defect(displacement[c].boxArray(),displacement[c].DistributionMap(),1,0);MF::LinComb(defect,1.,displacement[c],0,-1.,*dg[c],0,0,1,0);d_defect[c]=defect.norminf();}
     energy_jump=ElectricEnergy(sim,Pointers(accepted))-before_energy;
 }

 bool dumps=false;amrex::ParmParse("endpoint_diagnostic").query("write_fields",dumps);std::string const prefix=std::string(initial?"INITIAL_":"ENDPOINT_")+std::to_string(sim.getistep(0))+"_";
 if(dumps){for(int c=0;c<3;++c){if(s.model.UseCompatibleYeeInertia())amrex::VisMF::Write(inertia_yee[c],prefix+"Ei_yee_"+std::to_string(c));amrex::VisMF::Write(accepted[c],prefix+"E_"+std::to_string(c));amrex::VisMF::Write(s.C0[c],prefix+"C_"+std::to_string(c));amrex::VisMF::Write(instantaneous[c],prefix+"Ji_"+std::to_string(c));amrex::VisMF::Write(*companion[c],prefix+"Je_"+std::to_string(c));amrex::VisMF::Write(s.EL0[c],prefix+"EL_"+std::to_string(c));amrex::VisMF::Write(s.target[c],prefix+"inertial_longitudinal_target_"+std::to_string(c));amrex::VisMF::Write(s.F0[c],prefix+"noninertial_ohm_"+std::to_string(c));amrex::VisMF::Write(displacement[c],prefix+"D_"+std::to_string(c));if(audit)amrex::VisMF::Write(*dg[c],prefix+"D_gradient_"+std::to_string(c));}amrex::VisMF::Write(inertia,prefix+"Ei");amrex::VisMF::Write(s.density,prefix+"rho");amrex::VisMF::Write(s.model.ElectronPressureForSolve(0),prefix+"Pe");}
 if(amrex::ParallelDescriptor::IOProcessor()) {
    std::ofstream f("ENDPOINT_CONSTRAINT.jsonl",std::ios::app);
    f<<std::setprecision(17)<<"{\"initial\":"<<(initial?"true":"false")
     <<",\"time\":"<<time<<",\"published\":"<<((initial||publish)?"true":"false")
     <<",\"iterations\":"<<solver.getNumIters()<<",\"residual_V_per_m\":"<<norm
     <<",\"native_ohm_error_V_per_m\":"<<ohm_error
     <<",\"native_EL_error_V_per_m\":"<<el_error
     <<",\"companion_nodal_error_relative\":"<<map_error
     <<",\"oracle_ABA\":"<<oracle_aba<<",\"oracle_linearity\":"<<oracle_linearity
     <<",\"oracle_zero\":"<<oracle_zero
     <<",\"state_audit_performed\":"<<(audit?"true":"false")
     <<",\"all_registered_fields_restored_exact\":"<<(audit?"true":"null")
     <<",\"physical_particle_attributes_unchanged\":"<<(audit?"true":"null");
    if(audit) {
        f<<",\"electric_energy_publication_jump_J\":"<<energy_jump
         <<",\"displacement_norm_A_per_m2\":["<<d_norm[0]<<","<<d_norm[1]<<","<<d_norm[2]
         <<"],\"displacement_gradient_defect_A_per_m2\":["<<d_defect[0]<<","<<d_defect[1]<<","<<d_defect[2]<<"]";
    } else {
        f<<",\"electric_energy_publication_jump_J\":null,\"displacement_norm_A_per_m2\":null"
         <<",\"displacement_gradient_defect_A_per_m2\":null";
    }
    f<<"}\n";
 }

 if(exact_transverse)for(int c=0;c<3;++c)
     MF::Copy(*(*exact_transverse)[c],(*transverse_root)[c],0,0,1,(*exact_transverse)[c]->nGrowVect());
 return result;
}

namespace {
bool NativePrivateTransverseOutputSupported(WarpX& sim,View const* exact_transverse) {
    int lo=exact_transverse?1:0,hi=lo;
    amrex::ParallelDescriptor::ReduceIntMin(lo);amrex::ParallelDescriptor::ReduceIntMax(hi);
    if(lo!=hi)return false;
    if(!exact_transverse)return true;
        bool valid=true;
        std::vector<std::pair<std::uintptr_t,std::uintptr_t>> registered,outputs;
        for(auto const& name:sim.m_fields.list()) {
            auto const& f=*sim.m_fields.internal_get(name);
            for(amrex::MFIter mfi(f);mfi.isValid();++mfi) {
                auto const lo=reinterpret_cast<std::uintptr_t>(f[mfi].dataPtr());
                registered.emplace_back(lo,lo+f[mfi].size()*sizeof(Real));
            }
        }
        for(int c=0;c<3;++c) {
            auto const* out=(*exact_transverse)[c];
            auto const& e=*sim.m_fields.get(FieldType::Efield_fp,Direction{c},0);
            valid=valid && out && out->boxArray()==e.boxArray() &&
                out->DistributionMap()==e.DistributionMap() && out->nComp()==1 &&
                e.nGrowVect().allGE(out->nGrowVect());
            if(!out)continue;
            for(amrex::MFIter mfi(*out);mfi.isValid();++mfi) {
                auto const lo=reinterpret_cast<std::uintptr_t>((*out)[mfi].dataPtr());
                auto const hi=lo+(*out)[mfi].size()*sizeof(Real);
                for(auto const& range:registered)valid=valid && (hi<=range.first || lo>=range.second);
                for(auto const& range:outputs)valid=valid && (hi<=range.first || lo>=range.second);
                outputs.emplace_back(lo,hi);
            }
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        return valid;

}
}
void ConstrainNativeEndpointField(WarpX& sim,Real time,bool initial,
    std::function<void()> const& curl_rate,View const* exact_transverse) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativePrivateTransverseOutputSupported(sim,exact_transverse),
        "Initial exact transverse output requires private native storage on every rank");
    (void)ConstrainNativeEndpointFieldImpl(sim,time,initial,curl_rate,false,exact_transverse);
}

NativeEndpointResult TryConstrainNativeEndpointField(WarpX& sim,Real time,
    std::function<void()> const& curl_rate,View const* exact_transverse,NativeEndpointContext context) {
    auto const& model=*sim.get_pointer_HybridPICModel();
    bool const circuit=NativeCircuitDriveEnabled();
    auto* coupling=sim.get_pointer_CircuitCoupling();
    bool circuit_ready=!circuit;
    if(circuit && coupling && coupling->Coupler()) {
        auto& coupler=*coupling->Coupler();
        using Phase=CircuitCoupler::SourceImpulsePhase;
        switch(context) {
        case NativeEndpointContext::RetainedFieldEnd:
            circuit_ready=coupler.RetainedNativeEndpointReady();break;
        case NativeEndpointContext::PreSource:
            circuit_ready=coupler.RetainedSourceEndpointReady(Phase::PreField,time);break;
        case NativeEndpointContext::PostSource:
            circuit_ready=coupler.RetainedSourceEndpointReady(Phase::PostField,time);break;
        }
    }
    int context_lo=static_cast<int>(context),context_hi=context_lo;
    amrex::ParallelDescriptor::ReduceIntMin(context_lo);
    amrex::ParallelDescriptor::ReduceIntMax(context_hi);
    int output_lo=exact_transverse ? 1 : 0,output_hi=output_lo;
    amrex::ParallelDescriptor::ReduceIntMin(output_lo);
    amrex::ParallelDescriptor::ReduceIntMax(output_hi);
    bool supported=context_lo==context_hi && output_lo==output_hi &&
        (context==NativeEndpointContext::RetainedFieldEnd || context==NativeEndpointContext::PreSource ||
         context==NativeEndpointContext::PostSource) &&
        NativeEndpointEnabled() && !NativeVacuumEndpointEnabled() && !NativePrescribedDriveEnabled() &&
        (!model.m_add_external_fields || circuit) && model.UseCompatibleYeeInertia() && circuit_ready;
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if (!supported) {
        return {NativeEndpointFailure::UnsupportedScope,
            "retained endpoint requires positive source-free native Yee state and any circuit must be retained/closed"};
    }
    auto const& rho=*sim.m_fields.get(FieldType::rho_fp,0);
    if (!rho.is_finite() || !(rho.min(0)>PhysConst::q_e*model.m_n_floor)) {
        return {NativeEndpointFailure::InvalidDensity,"retained endpoint requires positive finite density"};
    }
    if(!NativePrivateTransverseOutputSupported(sim,exact_transverse))
        return {NativeEndpointFailure::UnsupportedScope,
            "exact transverse output must have native layout and private nonaliasing storage"};
    // Includes constructor/derived coefficient writes, and restores even a
    // failure detected after provisional field publication. No time rewind.
    warpx::implicit::FieldRollback saved;saved.Capture(sim.m_fields);
    try {
        auto result=ConstrainNativeEndpointFieldImpl(sim,time,false,curl_rate,true,exact_transverse);
        saved.Discard();
        return result;
    } catch (EndpointNumericalFailure const& error) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(saved.Restore(sim.m_fields),
            "Retained endpoint field topology changed during numerical decline");
        saved.Discard();
        return {error.failure,error.reason};
    }
}
