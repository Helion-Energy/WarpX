/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeStoppingMaterialSource.H"
#include "NativeStoppingThetaRegistration.H"
#include "NativePairedDarwinFields.H"
#include "NativeEndpointArithmetic.H"
#include "NativePECPlasma.H"
#include "DarwinInitialRateSchur.H"
#include "DarwinABoundary.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Particles/Deposition/EsirkepovInstantaneousCurrentIncrement.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace warpx::thermal {
namespace {
using R=amrex::Real;
using V=amrex::GpuArray<R,3>;
using CV=ablastr::fields::ConstVectorField;
using FV=ablastr::fields::VectorField;
using Fields=std::array<amrex::MultiFab,3>;
using Pair=warpx::ohm::compensated::Pair;
namespace pair=warpx::ohm::compensated;
constexpr R eps=std::numeric_limits<R>::epsilon();
constexpr char const* energy_name="hybrid_electron_energy_fp";
bool All(bool x){amrex::ParallelDescriptor::ReduceBoolAnd(x);return x;}
CV Const(Fields const& a){return {&a[0],&a[1],&a[2]};}
FV Mutable(Fields& a){return {&a[0],&a[1],&a[2]};}
void Clone(amrex::MultiFab& a,amrex::MultiFab const& b){
    a.define(b.boxArray(),b.DistributionMap(),1,b.nGrowVect());a.setVal(0.);
    amrex::MultiFab::Copy(a,b,0,0,1,b.nGrowVect());
}
void Allocate(Fields& a,CV const& b){for(int c=0;c<3;++c){Clone(a[c],*b[c]);a[c].setVal(0.);}}
void Sync(Fields& a,amrex::Geometry const& g){
    for(auto& f:a){f.OverrideSync(g.periodicity());f.FillBoundary(g.periodicity());}
}
// A borrowed native static field is part of the source origin. Compare all
// captured bytes on device before any action; no per-trial host field copy.
bool Same(Fields const& saved,WarpX& sim,char const* name){
    using ablastr::fields::Direction;
    bool valid=true;
    for(int c=0;c<3;++c){
        if(!sim.m_fields.has(name,Direction{c},0)){valid=false;continue;}
        auto const& live=*sim.m_fields.get(name,Direction{c},0);
        valid=live.boxArray()==saved[c].boxArray()&&
            live.DistributionMap()==saved[c].DistributionMap()&&
            live.nComp()==saved[c].nComp()&&live.nGrowVect()==saved[c].nGrowVect()&&valid;
    }
    if(!All(valid))return false;
    amrex::Gpu::DeviceScalar<int> mismatch(0);auto* bad=mismatch.dataPtr();
    for(int c=0;c<3;++c){auto const& live=*sim.m_fields.get(name,Direction{c},0);
        for(amrex::MFIter it(saved[c]);it.isValid();++it){
            auto const* a=reinterpret_cast<unsigned char const*>(saved[c][it].dataPtr());
            auto const* b=reinterpret_cast<unsigned char const*>(live[it].dataPtr());
            auto const count=saved[c][it].nBytes();
            amrex::For(count,[=] AMREX_GPU_DEVICE(amrex::Long n){
                if(a[n]!=b[n])amrex::Gpu::Atomic::Exch(bad,1);
            });
        }
    }
    return All(mismatch.dataValue()==0);
}
bool Equal(amrex::MultiFab const& a,amrex::MultiFab const& b){
    bool same=a.boxArray()==b.boxArray()&&a.DistributionMap()==b.DistributionMap()&&
        a.nComp()==b.nComp()&&a.nGrowVect()==b.nGrowVect();
    if(!All(same))return false;
    amrex::Gpu::DeviceScalar<int> mismatch(0);auto* bad=mismatch.dataPtr();
    for(amrex::MFIter it(a);it.isValid();++it){
        auto const* x=reinterpret_cast<unsigned char const*>(a[it].dataPtr());
        auto const* y=reinterpret_cast<unsigned char const*>(b[it].dataPtr());
        amrex::For(a[it].nBytes(),[=] AMREX_GPU_DEVICE(amrex::Long n){
            if(x[n]!=y[n])amrex::Gpu::Atomic::Exch(bad,1);
        });
    }
    return All(mismatch.dataValue()==0);
}
bool Finite(Fields const& a){bool good=true;for(auto const& f:a){bool const b=f.is_finite(0,1,0);good=good&&b;}return All(good);}
R Integral(amrex::MultiFab const& a,amrex::Geometry const& g){
    auto owner=a.OwnerMask(g.periodicity());auto const volume=MakeQdsmcVolumeElement(g,a.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);using T=decltype(data)::Type;
    for(amrex::MFIter it(a);it.isValid();++it){auto x=a.const_array(it);auto o=owner->const_array(it);
        op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T{
            return {o(i,j,k)?x(i,j,k)*volume(i,j,k):0.};});}
    R x=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(x);return x;
}
// Compare actual allocation intervals, including make_alias and differing
// BoxArrays. Collect each side in its own MFIter scope (nested MFIter is illegal).
using Bytes=std::pair<std::uintptr_t,std::size_t>;
std::vector<Bytes> Ranges(amrex::MultiFab const& a){
    std::vector<Bytes> r;for(amrex::MFIter it(a);it.isValid();++it)
        r.emplace_back(reinterpret_cast<std::uintptr_t>(a[it].dataPtr()),a[it].size()*sizeof(R));
    return r;
}
bool Overlap(amrex::MultiFab const& a,amrex::MultiFab const& b){
    auto const x=Ranges(a),y=Ranges(b);
    for(auto const& [p,n]:x)for(auto const& [q,m]:y)
        if(n&&m&&(p<=q?q-p<n:p-q<m))return true;
    return false;
}
AMREX_GPU_HOST_DEVICE R Gamma(V const& u){return std::sqrt(1.+(u[0]*u[0]+u[1]*u[1]+u[2]*u[2])/(PhysConst::c*PhysConst::c));}
AMREX_GPU_HOST_DEVICE Pair Product(Pair a,Pair b){return pair::Add(pair::Multiply(a,b.hi),pair::Multiply(a,b.lo));}
AMREX_GPU_HOST_DEVICE Pair Divide(Pair a,R b){
    R const q=a.hi/b;return pair::Sum(q,(std::fma(-q,b,a.hi)+a.lo)/b);
}
// Actual represented endpoints are the operands. Retain their subtraction
// before gamma/current scaling; do not regenerate next from rounded next-old.
// The independently measured second-native-evaluation remainder remains in
// the receipt. This is finite increment algebra, not a directional tangent.
AMREX_GPU_HOST_DEVICE V VelocityDelta(V const& old,V const& next){
    Pair d[3],n{0.,0.};for(int c=0;c<3;++c){d[c]=pair::Sum(next[c],-old[c]);
        n=pair::Add(n,Product(d[c],pair::Sum(next[c],old[c])));}
    R const g0=Gamma(old),g1=Gamma(next);
    Pair const dg=Divide(Divide(n,PhysConst::c*PhysConst::c),g0+g1);
    V result{};for(int c=0;c<3;++c){auto x=pair::Add(Divide(d[c],g1),
        pair::Negate(Divide(pair::Multiply(dg,old[c]),g0*g1)));result[c]=x.hi+x.lo;}
    return result;
}
}

struct NativeStoppingMaterialSource::Impl {
    WarpX& sim;AcceptedStoppingOptions material_options;NativeStoppingSourceOptions options;
    R time=0.,interval=0.;std::uint64_t epoch=0;
    amrex::Geometry geometry;
    std::shared_ptr<NativeStoppingCarryCertificate const> carry;
    std::shared_ptr<NativeEndpointAmpereOrigin const> field_origin;
    NativeStoppingFieldWork field_work;
    NativeEndpointAmpereCheck field_current;
    bool publication_work_verified=false;
    std::unique_ptr<NativeStoppingMaterialOwner> owner;
    NativeStoppingOwnerLease owner_lease;
    NativeStoppingOwnerCensus census;
    NativeStoppingOwnerView material;
    warpx::darwin::NativePairedDarwinFields* paired=nullptr;
    warpx::darwin::NativePairedDarwinFields::EndpointReceipt endpoint;
    std::unique_ptr<warpx::darwin::DarwinInitialRateSchur> projection;
    int projection_iteration_limit=0;
    Fields a0,b0,bstatic,cstatic,d0,w0,c0,stored0,j0,psi,mean,zero,di,ibase,dj,delta,pl,plpsi;
    Fields da,a_zero,db,dc,dd,dw,full_a,full_b,full_d,full_w,scratch,bcheck;
    Fields published_je,published_c;
    Fields correlated_di,theta_registration;
    struct Held {std::string name;int component=-1;amrex::MultiFab const* original=nullptr;amrex::MultiFab copy;};
    std::vector<std::unique_ptr<Held>> held;
    amrex::MultiFab u0,u1,du,unit;
    WarpXSolverVec solution,residual;
    std::array<std::string,3> current_names;
    R rho_scale=0.,psi_scale=0.,current_scale=0.,energy_scale=0.,arithmetic_factor=0.,volume=0.;
    NativeStoppingSourceWork work;
    bool captured=false,action=false,solved=false,metric_checked=false,root_action=false;
    bool publication_prepared=false,published=false;
    std::string failure="unprepared";
    Impl(WarpX& s,AcceptedStoppingOptions const& m,NativeStoppingSourceOptions const& o,
        R t,std::uint64_t n,R h,std::shared_ptr<NativeStoppingCarryCertificate const> c)
        :sim(s),material_options(m),options(o),time(t),interval(h),epoch(n),
         geometry(s.get_pointer_HybridPICModel()->ElectronThermalGeometry()),carry(std::move(c)){}
    bool Fail(char const* s){publication_prepared=false;action=false;solved=false;work.fresh=false;work.physical_gates=false;failure=s;return false;}
    bool AcceptedOrigin()const{return carry&&carry->Report().mode==NativeStoppingCarryMode::AcceptedOrigin;}
    bool Initialize();
    bool Matches(){
        if(!All(captured&&paired&&owner&&carry))return Fail("source owner unavailable");
        if(!paired->MatchesEndpoint(endpoint))return Fail("paired source origin changed");
        if(!owner->Matches(owner_lease))return Fail("material source origin changed");
        if(!Same(bstatic,sim,"hybrid_B_static_fp"))return Fail("source static magnetic origin changed");
        if(options.endpoint_acceptance&&(!field_origin||!field_origin->Matches(sim,time,epoch)))
            return Fail("verified F origin changed before source action");
        return true;
    }
    bool Type(WarpXSolverVec const&) const;
    std::vector<amrex::MultiFab const*> Blocks(WarpXSolverVec const&) const;
    bool Disjoint(WarpXSolverVec const&,WarpXSolverVec const&) const;
    bool Evaluate(WarpXSolverVec&,WarpXSolverVec const&);
    bool DepositIncrement();
    void AImages(Fields&);
    void Curl();
    bool Project(CV const&,Fields&,int);
    bool Work();
    bool PreparePublishedImages();
    bool CaptureHeldPublication();
    bool HeldPublicationMatches();
    bool AcceptanceWork(NativeStoppingSourceAcceptance&);
    R StepBound(WarpXSolverVec const&,WarpXSolverVec const&) const;
};
NativeStoppingMaterialSource::NativeStoppingMaterialSource(WarpX& s,
    AcceptedStoppingOptions const& m,NativeStoppingSourceOptions const& o,R t,
    std::uint64_t n,R h,std::shared_ptr<NativeStoppingCarryCertificate const> c,
    std::shared_ptr<NativeEndpointAmpereOrigin const> origin,NativeStoppingFieldWork const* field)
    :m_impl(std::make_unique<Impl>(s,m,o,t,n,h,std::move(c))){
    m_impl->field_origin=std::move(origin);if(field)m_impl->field_work=*field;
}
NativeStoppingMaterialSource::~NativeStoppingMaterialSource(){Invalidate();}
void NativeStoppingMaterialSource::Invalidate() noexcept{
    auto& p=*m_impl;p.captured=false;p.action=false;p.solved=false;p.root_action=false;
    p.publication_prepared=false;p.publication_work_verified=false;
    p.work.fresh=false;p.work.physical_gates=false;if(p.owner)p.owner->Invalidate();
}
bool NativeStoppingMaterialSource::Capture(){
    auto& p=*m_impl;Invalidate();
    p.paired=warpx::darwin::NativeEndpointPairedFields(p.sim);
    bool valid=p.paired&&p.carry&&
        ((p.carry->Report().mode==NativeStoppingCarryMode::FieldTransition&&p.carry->Report().status==StoppingCarryStatus::TransitionReady)||
         (p.AcceptedOrigin()&&p.carry->Report().status==StoppingCarryStatus::AcceptedReady))&&
        std::isfinite(p.interval)&&p.interval>0.;
    if(!All(valid))return p.Fail("source requires producer-bound accepted/transition carry and paired endpoint");
    if(p.options.endpoint_acceptance){
        bool ready=p.field_origin&&(p.AcceptedOrigin()?!p.field_work.available:
            (p.field_work.available&&p.field_work.time==p.time&&p.field_work.epoch==p.epoch&&p.field_work.interval==2.*p.interval));
        if(!All(ready))return p.Fail("source acceptance requires actual F current/work provenance");
        if(!p.field_origin->Matches(p.sim,p.time,p.epoch))return p.Fail("F source origin changed");
        p.field_current=p.field_origin->CheckCurrent(p.sim);
        if(!p.field_current.valid)return p.Fail("F origin does not satisfy unchanged next-F entry current");
    }
    p.endpoint=p.paired->EndpointLease();
    if(!p.paired->MatchesEndpoint(p.endpoint)||!p.carry->ValidateSource(p.sim,p.time,p.epoch))
        return p.Fail("source endpoint/transition provenance invalid");
    p.owner.reset(new NativeStoppingMaterialOwner(p.sim,p.material_options,p.options.physical,
        p.time,p.epoch,p.interval,p.carry,p.options.stable_drag_increment));
    auto const r=p.owner->Capture();p.owner_lease=r.lease;p.census=r.census;
    if(r.status!=NativeStoppingOwnerStatus::Ready)return p.Fail(r.reason.c_str());
    if(!p.Initialize())return false;
    p.captured=true;p.failure.clear();return true;
}
bool NativeStoppingMaterialSource::Impl::Initialize(){
#if !defined(WARPX_DIM_RZ)
    return Fail("transition source prototype requires RZ");
#else
#if defined(AMREX_USE_GPU)
    // Admit only the existing audited precise CUDA endpoint capability. The
    // source keeps particle and field state on device; scalar ledger reductions
    // remain collective. Ordinary CUDA/HIP/SYCL builds still reject this path.
    if (!All(warpx::darwin::NativeEndpointArithmeticSupported() &&
             warpx::darwin::NativeEndpointCudaQualificationSelected(sim))) {
        return Fail("transition source requires the precise CUDA endpoint capability");
    }
#endif
    auto const& o=options.thermal.nonlinear;
    bool valid=options.thermal.initial_guess==NativeStoppingThermalOptions::InitialGuess::Zero&&
        !options.thermal.analytic_temperature_column&&!o.use_preconditioner&&!o.adaptive_forcing&&
        o.relative_tolerance==0.&&o.absolute_tolerance==1.e-12&&
        o.linear_relative_tolerance==1.e-5&&o.probe_relative_size==1.e-5&&
        o.max_newton_iterations==32&&o.max_linear_iterations==200&&o.restart_length==60&&
        o.max_backtracks==10&&o.block_relative_tolerances.empty()&&o.block_absolute_tolerances.empty()&&
        o.block_reference_norms.empty()&&std::isfinite(options.relative_convention_budget)&&
        options.relative_convention_budget>=0.&&geometry.isPeriodic(1);
    R lo[3]={options.relative_convention_budget,R(o.probe_rhs_scale_floor),R(o.linear_verbosity)},hi[3];
    std::copy(lo,lo+3,hi);amrex::ParallelDescriptor::ReduceRealMin(lo,3);amrex::ParallelDescriptor::ReduceRealMax(hi,3);
    for(int i=0;i<3;++i)valid=valid&&lo[i]==hi[i];
    using ablastr::fields::Direction;
    auto const history=carry->SourceOrigin();
    valid=valid&&history.ready;
    for(auto const* name:{"hybrid_A_fp","Bfield_fp","hybrid_B_static_fp","diagnostic_D_endpoint_fp"})
        for(int c=0;c<3;++c)valid=valid&&sim.m_fields.has(name,Direction{c},0);
    valid=valid&&sim.m_fields.has(energy_name,0)&&!warpx::darwin::NativePECPlasmaEnabled();
    for(int c=0;c<3;++c)valid=valid&&!sim.m_fields.has(warpx::darwin::PECWallCurrentName,Direction{c},0);
    if(!All(valid))return Fail("unsupported source options or physical field inventory");
    for(int c=0;c<3;++c){
        auto const& total=*sim.m_fields.get("Bfield_fp",Direction{c},0);
        auto const& fixed=*sim.m_fields.get("hybrid_B_static_fp",Direction{c},0);
        valid=total.nComp()==1&&fixed.nComp()==1&&total.boxArray()==fixed.boxArray()&&
            total.DistributionMap()==fixed.DistributionMap()&&total.nGrowVect()==fixed.nGrowVect()&&valid;
    }
    if(!All(valid))return Fail("unsupported static magnetic field layout");
    for(int c=0;c<3;++c){
        Clone(a0[c],*sim.m_fields.get("hybrid_A_fp",Direction{c},0));
        Clone(b0[c],*sim.m_fields.get("Bfield_fp",Direction{c},0));
        Clone(bstatic[c],*sim.m_fields.get("hybrid_B_static_fp",Direction{c},0));
        Clone(d0[c],*sim.m_fields.get("diagnostic_D_endpoint_fp",Direction{c},0));
        // No stored W exists here. This buffer counts only the new source
        // reaction, so its entry value is zero by definition of an increment.
        Clone(w0[c],d0[c]);w0[c].setVal(0.);
        Clone(c0[c],*history.current[c]);
        Clone(stored0[c],*sim.m_fields.get(NativeAcceptedStoppingContext::AcceptedCurrentName,Direction{c},0));
        Clone(j0[c],*carry->MaterialCurrent()[c]);
    }
    for(auto* a:{&psi,&mean,&zero,&di,&ibase,&dj,&delta,&pl,&plpsi,&dc,&dd,&dw,&scratch})Allocate(*a,Const(j0));
    if(options.register_native_theta){Allocate(correlated_di,Const(j0));Allocate(theta_registration,Const(j0));}
    // The A boundary helper reads its reference over the complete A FAB,
    // including every reflected corner. A current-layout zero has fewer
    // guards and is not a valid reference even when consumed curl rows pass.
    for(auto* a:{&da,&a_zero,&full_a})Allocate(*a,Const(a0));
    for(auto* a:{&db,&bcheck,&full_b})Allocate(*a,Const(b0));
    Allocate(full_d,Const(d0));Allocate(full_w,Const(w0));
    Allocate(published_je,Const(stored0));
    for(int c=0;c<3;++c){Clone(published_c[c],*sim.m_fields.get("hybrid_current_fp_plasma",Direction{c},0));published_c[c].setVal(0.);}
    Allocate(cstatic,Const(c0));
    // Native C remains curl of the TOTAL physical B. The static part is
    // captured only for representation validation and separate signed work.
    auto static_current=Mutable(cstatic),static_field=Mutable(bstatic);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(
        static_current,static_field,sim.GetEBUpdateEFlag()[0],0);
    Sync(cstatic,geometry);
    Clone(u0,*sim.m_fields.get(energy_name,0));Clone(u1,u0);Clone(du,u0);du.setVal(0.);
    bool const fu=u0.is_finite(0,1,0);R const minimum=u0.min(0);
    rho_scale=PhysConst::q_e*material_options.reference_number_density;
    current_scale=rho_scale*options.physical.proper_speed_cap;
    psi_scale=PhysConst::m_e*options.physical.proper_speed_cap/PhysConst::q_e;
    R mass_per_charge=0.;for(auto const& name:sim.GetPartContainer().GetSpeciesNames()){
        auto const& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        if(pc.getCharge()>0.)mass_per_charge=std::max(mass_per_charge,pc.getMass()/pc.getCharge());}
    energy_scale=u0.norm0()+.5*rho_scale*mass_per_charge*options.physical.proper_speed_cap*options.physical.proper_speed_cap;
    R const operations=4096.+4096.*R(census.charged);
    arithmetic_factor=4.*operations*eps/(1.-operations*eps);
    valid=fu&&minimum>0.&&std::isfinite(current_scale)&&current_scale>0.&&
        std::isfinite(psi_scale)&&psi_scale>0.&&std::isfinite(energy_scale)&&energy_scale>0.&&
        std::isfinite(arithmetic_factor)&&arithmetic_factor>0.&&operations*eps<1.;
    if(!All(valid))return Fail("invalid physical source scales");
    amrex::MultiFab ones(u0.boxArray(),u0.DistributionMap(),1,0);ones.setVal(1.);volume=Integral(ones,geometry);
    using BC=warpx::darwin::InitialRateBoundary;
    warpx::darwin::InitialRateSchurOptions po;po.compatible_yee=true;
    po.lower={BC::Axis,BC::Periodic};po.upper={BC::PEC,BC::Periodic};
    po.relative_tolerance=1.e-12;po.absolute_tolerance=0.;po.output_ghosts=j0[0].nGrowVect();
    projection_iteration_limit=po.max_iterations;
    projection=std::make_unique<warpx::darwin::DarwinInitialRateSchur>(geometry,sim.boxArray(0),sim.DistributionMap(0),po);
    unit.define(amrex::convert(sim.boxArray(0),amrex::IntVect::TheNodeVector()),sim.DistributionMap(0),1,1);unit.setVal(1.);
    if(!projection->Freeze(unit))return Fail("native source projector freeze");
    std::vector<WarpXSolverVec::MultiFabBlockSpec> specs;
    for(int c=0;c<3;++c){current_names[c]=std::string(NativeAcceptedStoppingContext::AcceptedCurrentName)+
        "[dir="+static_cast<std::string>(Direction{c})+"]";
        valid=valid&&sim.m_fields.has(current_names[c],0);specs.push_back({current_names[c],current_scale});}
    if(!All(valid))return Fail("native current state templates unavailable");
    specs.push_back({energy_name,energy_scale});
    solution.Define(&sim,"Efield_fp","none",specs,psi_scale);solution.zero();
    residual.Define(solution);residual.zero();work.thermal.energy_scale=energy_scale;
    // Use the owned native field-phase curl producer. Work retains its full
    // finite endpoint defect and the distinct field-origin provenance.
    return Finite(c0)&&Finite(a0)&&Finite(b0)&&Finite(bstatic)&&Finite(cstatic)&&Finite(d0)&&Finite(w0)&&Finite(j0)&&Finite(stored0);
#endif
}
std::vector<amrex::MultiFab const*> NativeStoppingMaterialSource::Impl::Blocks(WarpXSolverVec const& x)const{
    std::vector<amrex::MultiFab const*> a;for(auto* p:x.getArrayVec()[0])a.push_back(p);
    for(auto const& n:current_names)a.push_back(&x.getMultiFabBlock(n,0));
    a.push_back(&x.getMultiFabBlock(energy_name,0));return a;
}
bool NativeStoppingMaterialSource::Impl::Type(WarpXSolverVec const& x)const{
    if(!x.IsDefined()||x.getVectorType()!=solution.getVectorType()||x.getScalarType()!=solution.getScalarType()||
       x.blockNames()!=solution.blockNames()||x.blockScales()!=solution.blockScales()||x.getArrayVec().size()!=1)return false;
    if(x.getArrayVec()[0].size()!=3)return false;
    auto a=Blocks(x),b=Blocks(solution);for(std::size_t n=0;n<a.size();++n)
        if(a[n]->nComp()!=1||a[n]->boxArray()!=b[n]->boxArray()||a[n]->DistributionMap()!=b[n]->DistributionMap())return false;
    return true;
}
bool NativeStoppingMaterialSource::Impl::Disjoint(WarpXSolverVec const& out,WarpXSolverVec const& x)const{
    auto a=Blocks(out),b=Blocks(x);std::vector<amrex::MultiFab const*> owned{&u0,&u1,&du,&unit};
    // Borrowed completed-action views are owned storage too. In particular,
    // using the internal residual as a const trial must decline before zero().
    for(auto const* state:{&solution,&residual}){auto const blocks=Blocks(*state);
        owned.insert(owned.end(),blocks.begin(),blocks.end());}
    for(auto const* fields:{&a0,&b0,&bstatic,&cstatic,&d0,&w0,&c0,&stored0,&j0,&psi,&mean,&zero,&di,&ibase,&dj,&delta,&pl,&plpsi,
        &da,&a_zero,&db,&dc,&dd,&dw,&full_a,&full_b,&full_d,&full_w,&scratch,&bcheck,&published_je,&published_c})for(auto const& f:*fields)owned.push_back(&f);
    if(options.register_native_theta)for(auto const* fields:{&correlated_di,&theta_registration})
        for(auto const& f:*fields)owned.push_back(&f);
    if(material.ready){for(auto const& fields:{material.impulse,material.mean_current,material.velocity,
        material.initial_electron_current,material.mass,material.inertia_impulse,material.momentum,
        material.raw_momentum,material.electron_current,material.mean_residual,material.ion_initial,material.ion_endpoint,
        material.stored_electron_current,material.carried_current})for(auto const* f:fields)if(f)owned.push_back(f);
        for(auto const* f:{material.midpoint_energy,material.nodal_temperature,material.nodal_heat,material.cell_heat,material.thermal_residual})if(f)owned.push_back(f);}
    for(std::size_t i=0;i<a.size();++i){for(std::size_t j=i+1;j<a.size();++j)if(Overlap(*a[i],*a[j]))return false;
        for(auto const* q:b)if(Overlap(*a[i],*q))return false;}
    auto const history=carry->TransitionView();
    for(auto const& field:{history.old_current,history.stage_current,history.endpoint_current,
        history.old_mass,history.stage_mass,history.endpoint_mass,history.root_residual})
        for(auto const* f:field)if(f)owned.push_back(f);
    for(int phase=0;phase<3;++phase)for(auto const& field:{history.phase_curl_current[phase],
        history.phase_ion[phase],history.phase_displacement[phase]})
        for(auto const* f:field)if(f)owned.push_back(f);
    auto const source_origin=carry->SourceOrigin();
    for(auto const& field:{source_origin.current,source_origin.ion,source_origin.electron,
        source_origin.displacement,source_origin.mass})for(auto const* f:field)if(f)owned.push_back(f);
    if(field_origin){auto const view=field_origin->View();
        for(auto const& field:{view.current,view.ion,view.electron,view.displacement,view.bound})
            for(auto const* f:field)if(f)owned.push_back(f);
    }
    for(auto const& name:sim.m_fields.list())owned.push_back(sim.m_fields.internal_get(name));
    for(auto const* q:owned){for(auto const* f:a)if(Overlap(*q,*f))return false;for(auto const* f:b)if(Overlap(*q,*f))return false;}
    return true;
}
bool NativeStoppingMaterialSource::DefineState(WarpXSolverVec& x)const{
    auto& p=*m_impl;if(!All(p.captured&&!x.IsDefined())||!p.Matches())return false;
    x.Define(p.solution);x.zero();return true;
}
bool NativeStoppingMaterialSource::Residual(WarpXSolverVec& out,WarpXSolverVec const& x){
    auto& p=*m_impl;if(!All(p.captured&&p.Type(out)&&p.Type(x)))return p.Fail("invalid source state layout");
    if(!p.Matches()||!All(p.Disjoint(out,x)))return p.Fail("source state aliases owned inputs or outputs");
    return p.Evaluate(out,x);
}
bool NativeStoppingMaterialSource::Impl::Project(CV const& in,Fields& out,int slot){
    if(options.refine_current_projection)work.projections[slot]={};
    auto const result=projection->Correct(in,Const(zero));++work.thermal.physical_projection_calls;
    work.thermal.physical_projection_iterations+=result.iterations;
    work.physical.projection_residual=std::max(work.physical.projection_residual,result.residual);
    if(options.refine_current_projection){auto& receipt=work.projections[slot];
        receipt.solves=1;receipt.iterations=result.iterations;
        receipt.initial_residual=result.initial_residual;receipt.first_residual=result.residual;
        receipt.original_target=result.target;}
    if(!All(result.converged))return Fail("native source projection failed");
    auto v=projection->CorrectionField();for(int c=0;c<3;++c){out[c].setVal(0.);
        amrex::MultiFab::Copy(out[c],*v[c],0,0,1,amrex::min(out[c].nGrowVect(),v[c]->nGrowVect()));}
    Sync(out,geometry);
    if(!options.refine_current_projection)return Finite(out);
    auto& receipt=work.projections[slot];
    if(!Finite(out))return Fail("nonfinite first source projection");
    // The original Psi projection already meets its physical row budget.
    // Preserve that map exactly; only current has the witnessed defect.
    if(slot==1){receipt.converged=true;return true;}
    receipt.refined=true;
    // Unit kappa was frozen above. Accumulate a second gradient directly in
    // this owned vector; rounding a combined potential would discard it.
    int const remaining=projection_iteration_limit-result.iterations;
    if(!All(remaining>=0))return Fail("source projection iteration budget exhausted");
    auto const correction=projection->CorrectWithIterationBudget(in,Const(out),remaining);
    ++work.thermal.physical_projection_calls;
    work.thermal.physical_projection_iterations+=correction.iterations;
    receipt.solves=2;receipt.iterations+=correction.iterations;
    receipt.correction_initial=correction.initial_residual;
    receipt.correction_residual=correction.residual;receipt.correction_target=correction.target;
    if(!All(correction.converged&&receipt.iterations<=projection_iteration_limit))
        return Fail("native source projection correction failed");
    v=projection->CorrectionField();
    for(int c=0;c<3;++c)amrex::MultiFab::Saxpy(out[c],1.,*v[c],0,0,1,
        amrex::min(out[c].nGrowVect(),v[c]->nGrowVect()));
    Sync(out,geometry);
    if(!Finite(out))return Fail("nonfinite corrected source projection");
    receipt.action_residual=projection->DivergenceResidualNorm(in,Const(out));
    receipt.action_evaluated=true;
    receipt.converged=All(std::isfinite(receipt.action_residual)&&
        receipt.action_residual<=receipt.original_target);
    return receipt.converged||Fail("fresh corrected source projection action failed");
}
void NativeStoppingMaterialSource::Impl::AImages(Fields& a){
#if defined(WARPX_DIM_RZ)
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    for(int c=0;c<3;++c){auto const domain=amrex::convert(geometry.Domain(),a[c].ixType());
        int const high=domain.bigEnd(0);bool const nodal=a[c].ixType().nodeCentered(0);
        for(amrex::MFIter it(a[c]);it.isValid();++it){auto x=a[c].array(it);
            amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                if((c==1&&i==0)||(nodal&&i>=high))x(i,j,k)=0.;});}
        a[c].FillBoundary(geometry.periodicity());
        ApplyDarwinCellCenteredABoundary(a[c],a_zero[c],geometry,true,nullptr,lo,hi);
    }
    auto v=Mutable(a);sim.ApplyFieldBoundaryOnAxis(v[0],v[1],v[2],0);
#else
    amrex::ignore_unused(a);
#endif
}
void NativeStoppingMaterialSource::Impl::Curl(){
    for(int c=0;c<3;++c){da[c].setVal(0.);amrex::MultiFab::Copy(da[c],psi[c],0,0,1,0);
        da[c].mult(-1.,0,1,da[c].nGrow());db[c].setVal(0.);dc[c].setVal(0.);}
    Sync(da,geometry);AImages(da);auto av=Mutable(da),bv=Mutable(db),cv=Mutable(dc);
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(bv,av,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    Sync(db,geometry);sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv,bv,sim.GetEBUpdateEFlag()[0],0);Sync(dc,geometry);
}
bool NativeStoppingMaterialSource::Impl::DepositIncrement(){
#if !defined(WARPX_DIM_RZ)
    return false;
#else
    bool population=true;
    for(auto const& view:material.particles){auto& pc=sim.GetPartContainer().GetParticleContainerFromName(view.species);
        // The owned map retains empty tiles; WarpXParIter intentionally
        // skips them. Bind every captured key/count before nonempty deposit.
        auto const& tiles=pc.GetParticles(0);auto const it=tiles.find({view.grid,view.tile});
        population=(it!=tiles.end()&&it->second.numParticles()==view.count)&&population;
    }
    if(!All(population))return Fail("source point population or tile changed");
    for(auto& f:di)f.setVal(0.);
    for(auto& f:ibase)f.setVal(0.);
    bool const register_theta=options.register_native_theta;
    if(register_theta)for(auto* fields:{&correlated_di,&theta_registration})for(auto& f:*fields)f.setVal(0.);
    amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
    for(auto const& view:material.particles){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(view.species);
        for(WarpXParIter it(pc,0);it.isValid();++it){
            if(it.index()!=view.grid||it.LocalTileIndex()!=view.tile)continue;
            auto box=it.tilebox();box.grow(sim.get_ng_depos_J());auto const origin=WarpX::LowerCorner(box,0,0.);
            auto const lower=amrex::lbound(box);auto const inverse=WarpX::InvCellSize(0);
            auto const bx=ibase[0].array(it),by=ibase[1].array(it),bz=ibase[2].array(it);
            auto const dx=di[0].array(it),dy=di[1].array(it),dz=di[2].array(it);
            auto const registration_array=register_theta?theta_registration[1].array(it):amrex::Array4<R>{};
            auto const* points=view.device_points;
            amrex::For(view.count,[=] AMREX_GPU_DEVICE(amrex::Long n){
                auto const p=points[n];R const g=Gamma(p.old);V v{},dv=VelocityDelta(p.old,p.next);
                for(int c=0;c<3;++c)v[c]=p.old[c]/g;
                R const xp=p.position[0],yp=p.position[1],rr=std::sqrt(xp*xp+yp*yp);
                warpx::particles::EsirkepovInstantaneousPointIncrement point;
                point.position={{{(rr-origin.x)*inverse.x,0.},{0.,0.},{(p.position[2]-origin.z)*inverse.z,0.}}};
                point.velocity={{{(xp*v[0]+yp*v[1])/rr*inverse.x,(xp*dv[0]+yp*dv[1])/rr*inverse.x},
                    {0.,0.},{v[2]*inverse.z,dv[2]*inverse.z}}};
                point.azimuthal_velocity={(-yp*v[0]+xp*v[1])/rr,(-yp*dv[0]+xp*dv[1])/rr};
                NativeStoppingThetaRegistration registration;
                if(register_theta&&!ComputeNativeStoppingThetaRegistration(p.position,p.old,p.next,
                    point.azimuthal_velocity.delta,PhysConst::c,registration)){
                    amrex::Gpu::Atomic::Exch(bad,1);return;}
                R const bound=R(std::numeric_limits<int>::max()-8);
                if(!std::isfinite(rr)||rr<=0.||!std::isfinite(point.position[0].value)||
                    !std::isfinite(point.position[2].value)||point.position[0].value<0.||point.position[0].value>bound||
                    point.position[2].value<0.||point.position[2].value>bound){amrex::Gpu::Atomic::Exch(bad,1);return;}
                int const ir0=int(point.position[0].value)-1,iz0=int(point.position[2].value)-1;
                R const scale[3]={p.charge*p.weight*inverse.y*inverse.z,
                    p.charge*p.weight*inverse.x*inverse.y*inverse.z,p.charge*p.weight*inverse.x*inverse.y};
                for(int iz=iz0;iz<=iz0+3;++iz)for(int ir=ir0;ir<=ir0+3;++ir){
                    amrex::GpuArray<warpx::particles::EsirkepovIncrement,3> q{};
                    if(!warpx::particles::EsirkepovInstantaneousCurrentPair<3,2>(point,{ir,0,iz},q)){
                        amrex::Gpu::Atomic::Exch(bad,1);continue;}
                    int const i=ir+lower.x,j=iz+lower.y;
                    if(register_theta){
                        amrex::GpuArray<R,3> const position{point.position[0].value,0.,point.position[2].value};
                        amrex::GpuArray<R,3> correction{},unused{};
                        if(!warpx::particles::EsirkepovCurrentRate<3,2>(position,{},{},{ir,0,iz},
                            registration.correction,0.,correction,unused)||
                            !std::isfinite(scale[1]*correction[1])){
                            amrex::Gpu::Atomic::Exch(bad,1);continue;}
                        amrex::HostDevice::Atomic::Add(&registration_array(i,j,0),scale[1]*correction[1]);
                    }
                    for(int c=0;c<3;++c){if((c==0&&ir==ir0+3)||(c==2&&iz==iz0+3))continue;
                        R const a=scale[c]*q[c].value,b=scale[c]*q[c].delta;
                        if(!std::isfinite(a)||!std::isfinite(b)){amrex::Gpu::Atomic::Exch(bad,1);continue;}
                        auto const ba=c==0?bx:(c==1?by:bz),increment_array=c==0?dx:(c==1?dy:dz);
                        amrex::HostDevice::Atomic::Add(&ba(i,j,0),a);amrex::HostDevice::Atomic::Add(&increment_array(i,j,0),b);
                    }
                }
            });
        }
    }
    if(!All(invalid.dataValue()==0)){for(auto& f:di)f.setVal(0.);
    for(auto& f:ibase)f.setVal(0.);
        if(register_theta)for(auto* fields:{&correlated_di,&theta_registration})for(auto& f:*fields)f.setVal(0.);
        return Fail("correlated fixed-position current outside native polynomial support");}
    for(auto* f:{&ibase,&di}){
        sim.ApplyInverseVolumeScalingToCurrentDensity(&(*f)[0],&(*f)[1],&(*f)[2],0);
        for(auto& a:*f)ablastr::utils::communication::SumBoundary(a,0,1,a.nGrowVect(),a.nGrowVect(),false,geometry.periodicity());
        sim.ApplyJfieldBoundary(0,&(*f)[0],&(*f)[1],&(*f)[2],PatchType::fine);Sync(*f,geometry);
    }
    if(register_theta){
        // Keep the old correlated map as an explicit diagnostic. Only the
        // represented theta correction enters the selected physical increment.
        for(int c=0;c<3;++c)amrex::MultiFab::Copy(correlated_di[c],di[c],0,0,1,di[c].nGrowVect());
        sim.ApplyInverseVolumeScalingToCurrentDensity(&theta_registration[0],&theta_registration[1],&theta_registration[2],0);
        for(auto& a:theta_registration)ablastr::utils::communication::SumBoundary(a,0,1,a.nGrowVect(),a.nGrowVect(),false,geometry.periodicity());
        sim.ApplyJfieldBoundary(0,&theta_registration[0],&theta_registration[1],&theta_registration[2],PatchType::fine);
        Sync(theta_registration,geometry);
        if(!Finite(theta_registration))return Fail("nonfinite represented theta registration");
        amrex::MultiFab::Add(di[1],theta_registration[1],0,0,1,di[1].nGrowVect());
    }
    return Finite(ibase)&&Finite(di);
#endif
}
bool NativeStoppingMaterialSource::Impl::Evaluate(WarpXSolverVec& out,WarpXSolverVec const& x){
    publication_prepared=false;action=false;solved=false;root_action=false;work.fresh=false;work.physical_gates=false;
    bool finite=true;for(auto const* f:Blocks(x)){bool const ok=f->is_finite(0,1,0);finite=finite&&ok;}
    if(!All(finite))return Fail("nonfinite source trial");
    for(int c=0;c<3;++c){psi[c].setVal(0.);amrex::MultiFab::Copy(psi[c],*x.getArrayVec()[0][c],0,0,1,0);
        amrex::MultiFab::LinComb(mean[c],1.,j0[c],0,1.,x.getMultiFabBlock(current_names[c],0),0,0,1,0);}
    du.setVal(0.);amrex::MultiFab::Copy(du,x.getMultiFabBlock(energy_name,0),0,0,1,0);
    NativeStoppingOwnerTrial trial{Const(psi),Const(mean),&du};
    auto const r=owner->Evaluate(owner_lease,trial);owner_lease=r.lease;
    if(!All(r.status==NativeStoppingOwnerStatus::Ready))return Fail(r.reason.c_str());
    material=owner->View(owner_lease);if(!All(material.ready))return Fail("source material view unavailable");
    if(!metric_checked){
        auto const history=carry->SourceOrigin();
        if(!All(history.ready))return Fail("source endpoint metric origin unavailable");
        amrex::Gpu::DeviceScalar<int> mismatch(0);auto* bad=mismatch.dataPtr();
        for(int c=0;c<3;++c)for(amrex::MFIter it(*material.mass[c]);it.isValid();++it){
            auto a=material.mass[c]->const_array(it),b=history.mass[c]->const_array(it);
            amrex::For(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                if(a(i,j,k)!=b(i,j,k))amrex::Gpu::Atomic::Exch(bad,1);
            });
        }
        if(!All(mismatch.dataValue()==0))return Fail("field endpoint and source kinetic metrics differ");
        metric_checked=true;
    }
    // Use the actual completed native impulse images, including axis/PEC.
    for(int c=0;c<3;++c)amrex::MultiFab::Copy(psi[c],*material.impulse[c],0,0,1,
        amrex::min(psi[c].nGrowVect(),material.impulse[c]->nGrowVect()));
    if(!DepositIncrement())return false;
    for(int c=0;c<3;++c)for(amrex::MFIter it(dj[c]);it.isValid();++it){
        auto outj=dj[c].array(it),outd=delta[c].array(it);
        auto p=material.physical_support[c]->const_array(it);auto m=material.mass[c]->const_array(it);
        auto f=psi[c].const_array(it),b=material.inertia_impulse[c]->const_array(it),ion=di[c].const_array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            R const kick=p(i,j,k)?(f(i,j,k)+b(i,j,k))/m(i,j,k):0.;
            outj(i,j,k)=kick;outd(i,j,k)=ion(i,j,k)+kick;});
    }
    Sync(dj,geometry);Sync(delta,geometry);
    if(!Finite(dj)||!Finite(delta)||!Project(Const(delta),pl,0)||!Project(Const(psi),plpsi,1))return false;
    Curl();residual.zero();R maximum=0.;
    for(int c=0;c<3;++c){amrex::MultiFab::Copy(dd[c],pl[c],0,0,1,dd[c].nGrowVect());dd[c].mult(-1.,0,1,dd[c].nGrow());
        for(amrex::MFIter it(dd[c]);it.isValid();++it){
            auto f=residual.getArrayVec()[0][c]->array(it),j=residual.getMultiFabBlock(current_names[c],0).array(it);
            auto wall=dw[c].array(it);auto kick=dj[c].const_array(it),change=delta[c].const_array(it);
            auto curl=dc[c].const_array(it),disp=dd[c].const_array(it),lp=plpsi[c].const_array(it);
            auto xp=x.getArrayVec()[0][c]->const_array(it),xj=x.getMultiFabBlock(current_names[c],0).const_array(it);
            int const high=geometry.Domain().bigEnd(0)+1;R const ps=psi_scale,cs=current_scale;
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int k,int l){
                bool const q=c!=0&&i==high,axis=c==1&&i==0;
                wall(i,k,l)=q?curl(i,k,l)-change(i,k,l)-disp(i,k,l):0.;
                f(i,k,l)=(q||axis)?xp(i,k,l):ps/cs*(change(i,k,l)+disp(i,k,l)-curl(i,k,l))+lp(i,k,l);
                j(i,k,l)=axis?xj(i,k,l):xj(i,k,l)-.5*kick(i,k,l);
            });
        }
        maximum=std::max({maximum,residual.getArrayVec()[0][c]->norm0()/psi_scale,
            residual.getMultiFabBlock(current_names[c],0).norm0()/current_scale});
        amrex::MultiFab::LinComb(full_a[c],1.,a0[c],0,1.,da[c],0,0,1,0);
        amrex::MultiFab::LinComb(full_b[c],1.,b0[c],0,1.,db[c],0,0,1,0);
        amrex::MultiFab::LinComb(full_d[c],1.,d0[c],0,1.,dd[c],0,0,1,0);
        amrex::MultiFab::LinComb(full_w[c],1.,w0[c],0,1.,dw[c],0,0,1,0);
    }
    amrex::MultiFab::Copy(residual.getMultiFabBlock(energy_name,0),*material.thermal_residual,0,0,1,0);
    work.thermal.maximum_residual=material.thermal_residual->norm0()/energy_scale;
    maximum=std::max(maximum,work.thermal.maximum_residual);
    amrex::MultiFab::LinComb(u1,1.,u0,0,1.,du,0,0,1,0);
    bool good=std::isfinite(maximum);for(auto const* f:Blocks(residual)){bool const b=f->is_finite(0,1,0);good=good&&b;}
    if(!All(good))return Fail("nonfinite full source residual");
    work.physical.spatial_residual=maximum;work.carry=material.work.carry;
    work.thermal.signed_residual=material.work.signed_thermal_residual;
    action=true;failure.clear();out.Copy(residual);return true;
}
R NativeStoppingMaterialSource::Impl::StepBound(WarpXSolverVec const& x,WarpXSolverVec const& d)const{
    auto const& u=x.getMultiFabBlock(energy_name,0);auto const& v=d.getMultiFabBlock(energy_name,0);
    amrex::ReduceOps<amrex::ReduceOpMin> op;amrex::ReduceData<R> data(op);using T=decltype(data)::Type;
    for(amrex::MFIter it(u);it.isValid();++it){auto a=u.const_array(it),b=v.const_array(it),old=u0.const_array(it);
        op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T{
            R const end=old(i,j,k)+a(i,j,k),step=b(i,j,k);
            if(!std::isfinite(end)||!std::isfinite(step)||end<=0.||old(i,j,k)<=0.)return {R(0)};
            return {step<0.?std::min(R(1),.99*end/(-step)):R(1)};});}
    R result=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealMin(result);
    return std::isfinite(result)?std::max(R(0),std::min(R(1),result)):R(0);
}
namespace {
struct SourceOperator final:ThermalNonlinearOperator{
    NativeStoppingMaterialSource::Impl& p;
    // This callback is private to SolveThermalSystem. Its contract supplies
    // the same state immediately after a successful fresh Residual action.
    WarpXSolverVec const* evaluated_state=nullptr;
    mutable NativeEndpointAmpereCheck last_current;
    explicit SourceOperator(NativeStoppingMaterialSource::Impl& q):p(q){}
    bool Residual(WarpXSolverVec& out,WarpXSolverVec const& x,int,bool)override{
        evaluated_state=nullptr;
        bool const ready=p.Matches()&&p.Evaluate(out,x);
        if(ready)evaluated_state=&x;
        return ready;
    }
    bool PhysicalConverged(WarpXSolverVec const& x)const override{
        // No additional work/reduction or changed convergence on legacy OFF.
        if(!p.options.endpoint_acceptance)return true;
        if(!All(evaluated_state==&x&&p.action&&p.material.ready&&bool(p.field_origin)))return false;
        if(!p.Matches()||!p.PreparePublishedImages())return false;
        // Private completed images and already-current complete-particle i1.
        // The original fresh live companion is still required after publish.
        last_current=p.field_origin->CheckCandidateCurrent(p.sim,Mutable(p.full_a),
            p.material.ion_endpoint,Const(p.published_je),Const(p.full_d));
        return last_current.valid;
    }
    R StepBound(WarpXSolverVec const& x,WarpXSolverVec const& d)const override{return p.StepBound(x,d);}
    bool Freeze(WarpXSolverVec const&,int,bool pc,ThermalSolveResult&)override{evaluated_state=nullptr;return !pc;}
    bool Precondition(WarpXSolverVec&,WarpXSolverVec const&)override{return false;}
    void RestoreInput(WarpXSolverVec const&)override{evaluated_state=nullptr;p.action=false;p.solved=false;p.work.fresh=false;}
};
}
bool NativeStoppingMaterialSource::Solve(){
    auto& p=*m_impl;if(!p.Matches())return false;
    p.solution.zero();p.work={};p.work.thermal.energy_scale=p.energy_scale;
    SourceOperator op(p);auto const result=SolveThermalSystem(op,p.solution,p.options.thermal.nonlinear);
    p.work.thermal.nonlinear=result;
    if(result.status!=ThermalSolveStatus::Converged||!p.action){
        amrex::Print()<<"Material source nonlinear solve declined: status="<<int(result.status)
            <<" Newton="<<result.newton_iterations<<" linear="<<result.linear_iterations
            <<" residual="<<result.residual<<" reason="<<p.failure<<"\n";
        amrex::Print()<<"Material source candidate current: rows="<<op.last_current.rows
            <<" error="<<op.last_current.maximum_error<<" bound="<<op.last_current.maximum_bound
            <<" maximum_row_ratio="<<op.last_current.maximum_ratio<<"\n";
        for (int slot=0;slot<2;++slot) {
            auto const& projection=p.work.projections[slot];
            amrex::Print()<<"Material source projection "<<slot
                <<": initial="<<projection.initial_residual
                <<" target="<<projection.original_target
                <<" first="<<projection.first_residual
                <<" correction_initial="<<projection.correction_initial
                <<" correction_residual="<<projection.correction_residual
                <<" action="<<projection.action_residual<<"\n";
        }
        return p.Fail("live-U source solve failed");
    }
    amrex::Print()<<"Material source nonlinear solve converged: Newton="<<result.newton_iterations
        <<" linear="<<result.linear_iterations<<" residual="<<result.residual
        <<" current_ratio="<<op.last_current.maximum_ratio
        <<" current_atol_A_per_m2="<<op.last_current.absolute_tolerance<<"\n";
    p.root_action=true;
    if(!p.Work()){
        amrex::Print()<<"Material source work declined: "<<p.failure<<"\n";
        // The last full residual action remains available for diagnostics when
        // only the independent work gate rejects. Freshness/physical readiness
        // stay false, and a stale origin exposes no action at all.
        if(p.Matches())p.action=true;
        return false;
    }
    p.solved=true;p.work.fresh=true;p.work.physical_gates=true;return true;
}
NativeStoppingSourceView NativeStoppingMaterialSource::View(){
    auto& p=*m_impl;NativeStoppingSourceView v;
    if(!All(p.captured)){v.failure=p.failure;v.work=p.work;return v;}
    if(!p.Matches()){v.failure=p.failure;return v;}
    v.captured=p.captured;v.action=p.action;v.solved=p.solved;v.failure=p.failure;v.work=p.work;
    if(!All(p.action))return v;
    v.material=p.owner->View(p.owner_lease);
    v.potential_increment=Const(p.da);v.magnetic_increment=Const(p.db);v.curl_increment=Const(p.dc);
    v.displacement_increment=Const(p.dd);v.conductor_increment=Const(p.dw);
    v.ion_increment=Const(p.di);v.ion_base=Const(p.ibase);v.electron_increment=Const(p.dj);
    if(p.options.register_native_theta){v.correlated_ion_increment=Const(p.correlated_di);
        v.theta_registration=Const(p.theta_registration);}
    v.longitudinal_current=Const(p.pl);v.longitudinal_impulse=Const(p.plpsi);
    v.residual=&p.residual;v.root_candidate=p.root_action?&p.solution:nullptr;v.energy_increment=&p.du;
    v.potential=Const(p.full_a);v.magnetic=Const(p.full_b);v.displacement=Const(p.full_d);v.conductor_current=Const(p.full_w);v.energy=&p.u1;
    return v;
}
bool NativeStoppingMaterialSource::Impl::CaptureHeldPublication(){
    using ablastr::fields::Direction;
    held.clear();bool present=true;
    auto add=[&](char const* name,int component){
        bool const exists=component<0?sim.m_fields.has(name,0):sim.m_fields.has(name,Direction{component},0);
        present=exists&&present;if(!exists)return;
        auto f=std::make_unique<Held>();f->name=name;f->component=component;
        f->original=component<0?sim.m_fields.get(name,0):sim.m_fields.get(name,Direction{component},0);
        held.push_back(std::move(f));
    };
    for(auto const* name:{"rho_fp","hybrid_rho_vacmask_fp","hybrid_phi_darwin_fp",
        "hybrid_Je_nm1_nodal","hybrid_Je_theta_nodal"})add(name,-1);
    for(auto const* name:{"hybrid_E_long_fp","hybrid_E_long_old_fp","hybrid_A_old_fp",
        "diagnostic_Je_stage_fp","diagnostic_D_stage_fp","current_fp"})for(int c=0;c<3;++c)add(name,c);
    if(!All(present)){held.clear();return Fail("source held endpoint/history inventory unavailable");}
    for(auto& f:held){auto const& b=*f->original;auto& a=f->copy;
        a.define(b.boxArray(),b.DistributionMap(),b.nComp(),b.nGrowVect());
        amrex::MultiFab::Copy(a,b,0,0,b.nComp(),b.nGrowVect());
    }
    return true;
}
bool NativeStoppingMaterialSource::Impl::HeldPublicationMatches(){
    using ablastr::fields::Direction;
    bool valid=!held.empty();
    for(auto const& f:held){bool const exists=f->component<0?sim.m_fields.has(f->name,0):
        sim.m_fields.has(f->name,Direction{f->component},0);
        auto const* now=exists?(f->component<0?sim.m_fields.get(f->name,0):sim.m_fields.get(f->name,Direction{f->component},0)):nullptr;
        valid=(now==f->original)&&valid;
    }
    if(!All(valid))return false;
    for(auto const& f:held){bool const same=Equal(f->copy,*f->original);valid=same&&valid;}
    bool const magnetic=Same(bstatic,sim,"hybrid_B_static_fp");return magnetic&&valid;
}
bool NativeStoppingMaterialSource::Impl::PreparePublishedImages(){
#if !defined(WARPX_DIM_RZ)
    return false;
#else
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(full_a[c],1.,a0[c],0,1.,da[c],0,0,1,full_a[c].nGrowVect());
        amrex::MultiFab::Copy(full_b[c],b0[c],0,0,1,full_b[c].nGrowVect());
        // Both operands retain the native curl-A physical images. Preserve
        // their complete common guards before exchanging shared/periodic rows.
        amrex::MultiFab::Add(full_b[c],db[c],0,0,1,
            amrex::min(full_b[c].nGrowVect(),db[c].nGrowVect()));
        amrex::MultiFab::Copy(full_d[c],d0[c],0,0,1,full_d[c].nGrowVect());
        amrex::MultiFab::Add(full_d[c],dd[c],0,0,1,amrex::min(full_d[c].nGrowVect(),dd[c].nGrowVect()));
        // Native endpoint current has zero physical extension and periodic
        // owner images. Its valid normal electrode current is never clamped.
        published_je[c].setVal(0.);
        amrex::MultiFab::Copy(published_je[c],*material.stored_electron_current[c],0,0,1,0);
        published_c[c].setVal(0.);
    }
    Sync(full_a,geometry);Sync(full_b,geometry);Sync(full_d,geometry);Sync(published_je,geometry);
    // Derived B uses the boundary conditions already applied to A. Repainting
    // it with generic B parity changes the wall current (DarwinDeriveB uses
    // the same exchange/sync-only rule).
    auto bv=Mutable(full_b);auto cv=Mutable(published_c);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv,bv,sim.GetEBUpdateEFlag()[0],0);
    Sync(published_c,geometry);
    bool finite=u1.is_finite(0,1,u1.nGrowVect());
    for(auto const* array:{&full_a,&full_b,&full_d,&published_je,&published_c})
        for(auto const& f:*array){bool const good=f.is_finite(0,1,f.nGrowVect());finite=good&&finite;}
    return All(finite);
#endif
}
bool NativeStoppingMaterialSource::PreparePublication(NativeStoppingSourcePublication& receipt){
    auto& p=*m_impl;
    bool const ready=All(p.captured&&p.solved&&p.work.fresh&&p.work.physical_gates&&!p.published);
    if(!ready||!p.Matches())return false;
    // Preserve solve work counters, but rebuild every signed work accumulator
    // from this fresh action (including the static-background partition).
    auto const nonlinear=p.work.thermal.nonlinear;
    p.work={};p.work.thermal.energy_scale=p.energy_scale;p.work.thermal.nonlinear=nonlinear;
    WarpXSolverVec fresh;fresh.Define(p.solution);fresh.zero();
    ++receipt.publication_residual_evaluations;
    if(!p.Evaluate(fresh,p.solution))return false;
    p.root_action=true;if(!p.Work())return false;
    p.solved=true;p.work.fresh=true;p.work.physical_gates=true;
    if(!p.CaptureHeldPublication()||!p.PreparePublishedImages())return p.Fail("source publication images unavailable");
    p.publication_prepared=true;p.publication_work_verified=true;
    receipt.work=p.work;receipt.time=p.time;receipt.epoch=p.epoch;receipt.interval=p.interval;
    return true;
}
bool NativeStoppingMaterialSource::PublishPrepared(){
    auto& p=*m_impl;
    // The candidate has revoked its source-origin pair. Do not call Matches()
    // here: it must reject that old receipt. The complete population owner has
    // its own last collective preflight before any physical momentum write.
    if(!All(p.publication_prepared&&!p.published&&p.paired&&!p.paired->EndpointPending()))return false;
    p.publication_prepared=false;p.captured=false;p.action=false;p.solved=false;p.root_action=false;
    if(!p.owner->PublishOnce(p.owner_lease))return false;
    using ablastr::fields::Direction;
    for(int c=0;c<3;++c){
        auto copy=[&](char const* name,amrex::MultiFab const& from){
            auto& to=*p.sim.m_fields.get(name,Direction{c},0);
            amrex::MultiFab::Copy(to,from,0,0,1,to.nGrowVect());
        };
        copy("hybrid_A_fp",p.full_a[c]);copy("Bfield_fp",p.full_b[c]);
        copy("diagnostic_D_endpoint_fp",p.full_d[c]);
        copy(NativeAcceptedStoppingContext::AcceptedCurrentName,p.published_je[c]);
        copy("hybrid_current_fp_plasma",p.published_c[c]);
    }
    // Keep the solved DeltaU endpoint. Native thermal publication updates valid
    // energy cells; the physical moment map produces the temperature/pressure.
    amrex::MultiFab::Copy(*p.sim.m_fields.get(energy_name,0),p.u1,0,0,1,0);
    p.published=true;p.work.fresh=false;p.work.physical_gates=false;return true;
}
bool NativeStoppingMaterialSource::CheckPublication(NativeStoppingSourcePublication& receipt){
    auto& p=*m_impl;if(!All(p.published))return false;
    bool valid=p.owner->PublishedPopulationMatches();
    bool const held=p.HeldPublicationMatches();valid=held&&valid;
    for(auto const& item:std::initializer_list<std::pair<Fields const*,char const*>>{
        {&p.full_a,"hybrid_A_fp"},{&p.full_b,"Bfield_fp"},{&p.full_d,"diagnostic_D_endpoint_fp"},
        {&p.published_je,NativeAcceptedStoppingContext::AcceptedCurrentName},{&p.published_c,"hybrid_current_fp_plasma"}}){
        bool const same=Same(*item.first,p.sim,item.second);valid=same&&valid;
    }
    bool const energy=Equal(p.u1,*p.sim.m_fields.get(energy_name,0));valid=energy&&valid;
    R error=0.;
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(p.scratch[c],1.,p.published_c[c],0,-1.,p.c0[c],0,0,1,0);
        amrex::MultiFab::Subtract(p.scratch[c],p.dc[c],0,0,1,0);
        error=std::max(error,p.scratch[c].norm0());
    }
    receipt.published_current_rounding=error;
    receipt.published_current_bound=p.work.background.current_bound+p.work.physical.grid_bound;
    valid=std::isfinite(error)&&error<=receipt.published_current_bound&&valid;
    return All(valid);
}
bool NativeStoppingMaterialSource::CheckAcceptance(NativeStoppingSourcePublication& receipt){
    auto& p=*m_impl;receipt.acceptance.ready=false;
    receipt.source_finalization_available=false;receipt.field_endpoint_ampere_closed=false;
    receipt.work.field_endpoint_ampere_closed=false;
    bool ready=p.options.endpoint_acceptance&&p.published&&p.publication_work_verified&&
        p.field_origin&&(p.AcceptedOrigin()||p.field_work.available);
    if(!All(ready)||!CheckPublication(receipt))return false;
    auto check=p.field_origin->CheckCurrent(p.sim);
    receipt.acceptance.published_current=check;
    if(!check.valid){receipt.reason="published source fails next-F absolute-plus-relative current entry";return false;}
    if(!p.AcceptanceWork(receipt.acceptance)){
        receipt.reason="actual represented F/source work identity failed";return false;
    }
    receipt.acceptance.published_current=check;receipt.acceptance.ready=true;
    receipt.source_finalization_available=!p.AcceptedOrigin();receipt.field_endpoint_ampere_closed=true;
    receipt.work.field_endpoint_ampere_closed=true;
    return true;
}
bool NativeStoppingMaterialSource::Impl::AcceptanceWork(NativeStoppingSourceAcceptance& result){
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(result);return false;
#else
#if defined(AMREX_USE_GPU)
    if (!All(warpx::darwin::NativeEndpointArithmeticSupported() &&
             warpx::darwin::NativeEndpointCudaQualificationSelected(sim))) return false;
#endif
    result.ready=false;result.includes_field=!AcceptedOrigin();result.field=field_work;result.field_current=field_current;
    auto const& f=field_work;auto const& l=work.physical;auto const& metric=work.carry.moving_metric;
    bool valid=AcceptedOrigin()?(!f.available&&!metric.available):
        (f.available&&f.magnetic.valid&&f.ions.valid&&f.ions.has_total_field&&
        f.ions.gather_contract==IonElectricGatherContract::RecordedFinalGather&&
        f.ions.particles==census.charged&&metric.available&&f.ions.species.size()==f.ions.diagnostics.size());
    if(!All(valid))return false;
    R field_ion_delta=0.,field_ion_work=0.,field_ion_defect=0.,field_ion_initial=0.,field_ion_final=0.;
    using IC=IonElectricWorkComponent;using ID=IonElectricWorkDiagnostic;
    for(std::size_t i=0;i<f.ions.species.size();++i){auto const& a=f.ions.species[i];auto const& b=f.ions.diagnostics[i];
        field_ion_delta+=a[IC::KineticChange];field_ion_work+=a[IC::TotalFieldWork];
        field_ion_defect+=a[IC::TotalWorkDefect];field_ion_initial+=b[ID::OldKineticEnergy];field_ion_final+=b[ID::EndpointKineticEnergy];
    }
    R const ops=128.+128.*(R(census.charged)+R(geometry.Domain().numPts())*24.);
    R const e=ops*eps;
    if(!All(std::isfinite(e)&&e>0.&&e<.01))return false;
    R const gamma=e/(1.-e),factor=4.*gamma;
    R const old_u=Integral(u0,geometry),new_u=Integral(u1,geometry);
    R const binding_scale=std::abs(old_u)+std::abs(f.thermal.final)+l.ion_initial_energy+
        std::abs(field_ion_final)+l.electron_initial_energy+std::abs(metric.final_kinetic);
    R const binding_bound=std::nextafter(factor*binding_scale,std::numeric_limits<R>::infinity());
    valid=AcceptedOrigin()||(std::abs(old_u-f.thermal.final)<=binding_bound&&
        std::abs(l.ion_initial_energy-field_ion_final)<=binding_bound&&
        std::abs(l.electron_initial_energy-metric.final_kinetic)<=binding_bound);
    if(!All(valid))return false;
    R ion=0.,ion_abs=0.;
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> op;
    amrex::ReduceData<R,R> data(op);using T=decltype(data)::Type;
    for(auto const& view:material.particles){auto const* points=view.device_points;
        op.eval(view.count,data,[=]AMREX_GPU_DEVICE(amrex::Long n)->T{
            auto const p=points[n];Pair difference{0.,0.};
            for(int c=0;c<3;++c)difference=pair::Add(difference,
                Product(pair::Sum(p.next[c],-p.old[c]),pair::Sum(p.next[c],p.old[c])));
            auto value=Divide(pair::Multiply(difference,p.weight*p.mass),Gamma(p.old)+Gamma(p.next));
            R const q=value.hi+value.lo;return {q,std::abs(q)};
        });}
    auto v=data.value();ion=amrex::get<0>(v);ion_abs=amrex::get<1>(v);
    amrex::ParallelDescriptor::ReduceRealSum(ion);amrex::ParallelDescriptor::ReduceRealSum(ion_abs);
    R electron=0.,magnetic=0.,materialization=0.,material_bound=0.,magnetic_scale=0.;
    int bad=0;
    for(int c=0;c<3;++c){
        auto eo=stored0[c].OwnerMask(geometry.periodicity());auto ev=MakeQdsmcVolumeElement(geometry,stored0[c].ixType());
        amrex::ReduceOps<amrex::ReduceOpSum> eop;amrex::ReduceData<R> ed(eop);using ET=decltype(ed)::Type;
        for(amrex::MFIter it(stored0[c]);it.isValid();++it){auto own=eo->const_array(it);auto old=stored0[c].const_array(it),
            next=published_je[c].const_array(it),mass=material.mass[c]->const_array(it);
            eop.eval(it.validbox(),ed,[=]AMREX_GPU_DEVICE(int i,int j,int k)->ET{
                if(!own(i,j,k))return {R(0)};
                auto q=Product(pair::Sum(next(i,j,k),-old(i,j,k)),pair::Sum(next(i,j,k),old(i,j,k)));
                q=pair::Multiply(q,.5*mass(i,j,k)*ev(i,j,k));return {q.hi+q.lo};
            });}
        electron+=amrex::get<0>(ed.value());
        auto bo=b0[c].OwnerMask(geometry.periodicity());auto bv=MakeQdsmcVolumeElement(geometry,b0[c].ixType());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpMax> bop;
        amrex::ReduceData<R,R,R,R,int> bd(bop);using BT=decltype(bd)::Type;
        for(amrex::MFIter it(b0[c]);it.isValid();++it){auto own=bo->const_array(it);auto old=b0[c].const_array(it),
            next=full_b[c].const_array(it),inc=db[c].const_array(it);
            bop.eval(it.validbox(),bd,[=]AMREX_GPU_DEVICE(int i,int j,int k)->BT{
                if(!own(i,j,k))return {R(0),R(0),R(0),R(0),0};
                R const a=old(i,j,k),b=next(i,j,k),d=inc(i,j,k),volume=bv(i,j,k)/PhysConst::mu0;
                auto diff=pair::Sum(b,-a);auto mb=pair::Multiply(Product(diff,pair::Sum(b,a)),.5*volume);
                auto rem=pair::Add(diff,Pair{-d,0.});R const rounding=rem.hi+rem.lo;
                R const g8=8.*eps/(1.-8.*eps);
                R const rb=std::nextafter(g8*(std::abs(a)+std::abs(d))+8.*std::numeric_limits<R>::denorm_min(),std::numeric_limits<R>::infinity());
                auto mr=pair::Add(Product(pair::Sum(a,d),rem),pair::Multiply(Product(rem,rem),.5));
                mr=pair::Multiply(mr,volume);
                R const bound=((std::abs(a)+std::abs(d))*rb+.5*rb*rb)*volume;
                R const scale=(a*a+b*b+std::abs(a*d)+d*d)*std::abs(volume);
                bool const ok=std::isfinite(rounding)&&std::isfinite(rb)&&std::isfinite(bound)&&
                    std::isfinite(scale)&&std::abs(rounding)<=rb;
                return {mb.hi+mb.lo,mr.hi+mr.lo,bound,scale,ok?0:1};
            });}
        auto bx=bd.value();magnetic+=amrex::get<0>(bx);materialization+=amrex::get<1>(bx);
        material_bound+=amrex::get<2>(bx);magnetic_scale+=amrex::get<3>(bx);bad=std::max(bad,amrex::get<4>(bx));
    }
    R sums[5]={electron,magnetic,materialization,material_bound,magnetic_scale};amrex::ParallelDescriptor::ReduceRealSum(sums,5);
    amrex::ParallelDescriptor::ReduceIntMax(bad);electron=sums[0];magnetic=sums[1];materialization=sums[2];
    material_bound=std::nextafter(sums[3]*(1.+factor),std::numeric_limits<R>::infinity());magnetic_scale=sums[4];
    amrex::MultiFab cells(u0.boxArray(),u0.DistributionMap(),1,0);
    amrex::MultiFab::LinComb(cells,1.,u1,0,-1.,u0,0,0,1,0);R const internal=Integral(cells,geometry);
    R const source_actual=ion+electron+internal+magnetic;
    R const source_expected=l.actual_energy_defect+materialization;
    R const inventory_bound=std::nextafter(factor*(l.ion_initial_energy+l.electron_initial_energy+
        std::abs(old_u)+std::abs(new_u)+ion_abs+magnetic_scale),std::numeric_limits<R>::infinity());
    valid=bad==0&&std::isfinite(inventory_bound)&&
        std::abs(ion-l.ion_energy_change)<=inventory_bound&&
        std::abs(electron-l.electron_energy_change)<=inventory_bound&&
        std::abs(internal-work.thermal.delivered_heat)<=inventory_bound&&
        std::abs(magnetic-l.magnetic_energy_change-materialization)<=inventory_bound&&
        std::abs(materialization)<=material_bound+factor*std::abs(materialization)&&
        std::abs(source_actual-source_expected)<=inventory_bound;
    R const field_change=AcceptedOrigin()?0.:field_ion_delta+(metric.final_kinetic-metric.initial_kinetic)+
        (f.thermal.final-f.thermal.initial)+f.magnetic.change;
    R const field_named=AcceptedOrigin()?0.:field_ion_work+field_ion_defect+metric.stage_work+metric.metric_work+
        metric.rotation_work+metric.identity_remainder+f.thermal.source+f.thermal.conduction+
        f.thermal.advection+f.thermal.compression+f.thermal.defect+
        f.magnetic.curl_work+f.magnetic.curl_transfer+f.magnetic.represented_faraday_work;
    R source_named=l.relativistic_defect+l.spatial_transfer+l.constraint_work+l.mean_current_work+
        l.transpose_work+work.material_endpoint_rounding+l.heat_transfer_error+l.curl_work+
        l.displacement_work+l.conductor_work+work.carry.free_work+work.carry.transition_work+
        work.carry.fixed_work+l.accounting_defect+materialization;
    if(AcceptedOrigin())source_named+=work.carry.accepted_work;
    R const combined=field_change+source_actual,named=field_named+source_named;
    R const account_bound=std::nextafter(inventory_bound+metric.arithmetic_bound+f.magnetic.arithmetic_bound+
        l.arithmetic_bound+factor*(binding_scale+std::abs(field_ion_initial)+std::abs(f.thermal.initial)+
        std::abs(field_named)+std::abs(source_named)),std::numeric_limits<R>::infinity());
    valid=valid&&std::isfinite(combined)&&std::isfinite(named)&&std::isfinite(account_bound)&&
        std::abs(combined-named)<=account_bound;
    result.initial_internal=old_u;result.final_internal=new_u;
    result.source_ion=ion;result.source_electron=electron;result.source_internal=internal;result.source_magnetic=magnetic;
    result.magnetic_materialization=materialization;result.magnetic_materialization_bound=material_bound;
    result.source_inventory_error=source_actual-source_expected;result.source_inventory_bound=inventory_bound;
    result.field_ion_work_defect=field_ion_defect;result.field_change=field_change;result.source_change=source_actual;
    result.combined_change=combined;result.named_work=named;result.accounting_error=combined-named;result.accounting_bound=account_bound;
    return All(valid);
#endif
}

bool NativeStoppingMaterialSource::Impl::Work(){
#if !defined(WARPX_DIM_RZ)
    return Fail("RZ work unavailable");
#else
    auto& l=work.physical;auto& bg=work.background;auto const& q=material.work;
    R const spatial=l.spatial_residual,projection_residual=l.projection_residual;l={};bg={};
    l.spatial_residual=spatial;l.projection_residual=projection_residual;
    l.current_scale=current_scale;l.contribution_count=R(census.charged);l.operations=4096.+4096.*l.contribution_count;
    l.ion_initial_energy=q.ion_before;l.ion_energy_change=q.ion_increment;
    l.electron_initial_energy=q.material.kinetic_before;l.electron_energy_change=q.material.kinetic_increment;
    l.drag_energy_change=q.drag;l.drag_bulk_work=q.bulk;l.electric_particle_work=q.electric;
    l.endpoint_particle_work=q.endpoint_particle_work;l.heat=q.heat_nominal;
    l.relativistic_defect=q.convention;l.relativistic_bound=q.convention_bound;l.absolute_work=q.absolute_particle_work;
    l.spatial_transfer=q.spatial_transfer;l.transpose_work=q.bulk+q.material.mean_reaction;
    l.axis_reaction_work=q.axis_reaction_work;work.material_endpoint_rounding=q.material.endpoint_rounding;
    l.uniform_error=census.density_difference/rho_scale;l.uniform_bound=census.density_bound/rho_scale;
    work.ion_increment_remainder=0.;work.electron_increment_remainder=0.;work.current_base_remainder=0.;
    work.current_representation_work=0.;
    work.correlated_ion_increment_remainder=0.;work.correlated_current_representation_work=0.;work.theta_registration_work=0.;
    work.background_identity_error=0.;work.preserved_defect_error=0.;
    work.inherited_ampere_work=0.;work.source_increment_constraint_work=0.;
    // Original momentum map, with the material covector restricted only on P.
    // Actual old/next points and drag impulses supply independent particle sums.
    amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
        amrex::ReduceOpSum,amrex::ReduceOpSum> mop;
    amrex::ReduceData<R,R,R,R,R,R,R,R> md(mop);using MT=decltype(md)::Type;
    for(auto const& v:material.particles){auto const* points=v.device_points;
        mop.eval(v.count,md,[=] AMREX_GPU_DEVICE(amrex::Long n)->MT{
            auto const p=points[n];R norm=0.;for(int c=0;c<3;++c)norm+=p.old[c]*p.old[c];
            return {p.weight*p.mass*(p.next[0]-p.old[0]),p.weight*p.mass*(p.next[1]-p.old[1]),
                p.weight*p.mass*(p.next[2]-p.old[2]),p.weight*p.mass*(p.drag[0]-p.minus[0]),
                p.weight*p.mass*(p.drag[1]-p.minus[1]),p.weight*p.mass*(p.drag[2]-p.minus[2]),
                p.weight*p.mass*std::sqrt(norm),p.weight*p.charge*p.psi[2]};});}
    auto mt=md.value();R sums[8]={amrex::get<0>(mt),amrex::get<1>(mt),amrex::get<2>(mt),amrex::get<3>(mt),
        amrex::get<4>(mt),amrex::get<5>(mt),amrex::get<6>(mt),amrex::get<7>(mt)};
    amrex::ParallelDescriptor::ReduceRealSum(sums,8);
    l.positive_momentum_scale=sums[6]+volume*rho_scale*PhysConst::m_e/PhysConst::q_e*options.physical.proper_speed_cap;
    l.momentum_bound=arithmetic_factor*l.positive_momentum_scale;
    auto const history=carry->SourceOrigin();
    if(!All(history.ready))return Fail("source lost physical Ampere origin");
    work.field_transition_background=history.field_transition;
    work.accepted_origin_background=!history.field_transition;work.field_endpoint_ampere_closed=false;
    R coefficient_scale=0.,background_scale=current_scale;
    R const derivative=4./std::min(geometry.CellSize(0),geometry.CellSize(1));
    R a_scale=0.,b_scale=0.,d_scale=0.,da_scale=0.,dd_scale=0.;
    for(int c=0;c<3;++c){
        l.ion_momentum_change[c]=sums[c];l.reaction_momentum[c]=-sums[c+3];
        l.unrepresented_particle_momentum[c]=c<2?sums[c]:0.;
        amrex::ReduceOps<amrex::ReduceOpMin> minop;amrex::ReduceData<R> minval(minop);using MinT=decltype(minval)::Type;
        for(amrex::MFIter it(scratch[c]);it.isValid();++it){auto mask=material.physical_support[c]->const_array(it);auto m=material.mass[c]->const_array(it);
            minop.eval(it.validbox(),minval,[=] AMREX_GPU_DEVICE(int i,int j,int k)->MinT{
                return {mask(i,j,k)?m(i,j,k):std::numeric_limits<R>::max()};});}
        R minimum=amrex::get<0>(minval.value());amrex::ParallelDescriptor::ReduceRealMin(minimum);
        if(!All(std::isfinite(minimum)&&minimum>0.))return Fail("invalid physical-row mass in work gate");
        coefficient_scale+=psi[c].norm0()/minimum+dc[c].norm0();
        background_scale+=material.ion_initial[c]->norm0()+stored0[c].norm0()+d0[c].norm0()+w0[c].norm0()+c0[c].norm0();
        a_scale+=derivative*a0[c].norm0();b_scale+=b0[c].norm0();d_scale+=derivative*d0[c].norm0();
        da_scale+=derivative*da[c].norm0();dd_scale+=derivative*dd[c].norm0();
        // Every term is computed from the represented final endpoint. The
        // stable kick/current increment is a distinct diagnostic operand.
        for(int channel=0;channel<(options.register_native_theta?27:24);++channel){
            for(amrex::MFIter it(scratch[c]);it.isValid();++it){auto out=scratch[c].array(it);
                auto old=j0[c].const_array(it),next=material.electron_current[c]->const_array(it),st=material.stored_electron_current[c]->const_array(it);
                auto oldst=stored0[c].const_array(it),trial=material.mean_current[c]->const_array(it);
                auto ion0=material.ion_initial[c]->const_array(it),ion1=material.ion_endpoint[c]->const_array(it);
                auto base=ibase[c].const_array(it),inc=di[c].const_array(it),kick=dj[c].const_array(it);
                auto correlated=options.register_native_theta?correlated_di[c].const_array(it):inc;
                auto p=psi[c].const_array(it),m=material.mass[c]->const_array(it);auto mask=material.physical_support[c]->const_array(it);
                auto b=material.inertia_impulse[c]->const_array(it),pi=material.raw_momentum[c]->const_array(it);
                auto velocity=material.velocity[c]->const_array(it),C=c0[c].const_array(it),D=d0[c].const_array(it),W=w0[c].const_array(it);
                auto dC=dc[c].const_array(it),dD=dd[c].const_array(it),dW=dw[c].const_array(it);
                auto cs=cstatic[c].const_array(it);
                auto cf=history.current[c]->const_array(it),ifield=history.ion[c]->const_array(it),
                    jf=history.electron[c]->const_array(it),df=history.displacement[c]->const_array(it);
                int const wall_index=geometry.Domain().bigEnd(0)+1;
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    R const mid=.5*(old(i,j,k)+next(i,j,k)),smid=.5*(oldst(i,j,k)+st(i,j,k));
                    R const imid=.5*(ion0(i,j,k)+ion1(i,j,k)),ps=p(i,j,k);
                    R z=0.;
                    if(channel==0)z=(imid+smid+D(i,j,k)+.5*dD(i,j,k)+W(i,j,k)+.5*dW(i,j,k)-C(i,j,k)-.5*dC(i,j,k))*ps;
                    else if(channel==1)z=(mid-trial(i,j,k))*b(i,j,k);
                    else if(channel==2)z=(next(i,j,k)-old(i,j,k));
                    else if(channel==3)z=mask(i,j,k)?PhysConst::m_e/PhysConst::q_e*ps/m(i,j,k):0.;
                    else if(channel==4)z=pi(i,j,k);
                    else if(channel==5)z=(C(i,j,k)+.5*dC(i,j,k))*ps;
                    else if(channel==6)z=-(D(i,j,k)+.5*dD(i,j,k))*ps;
                    else if(channel==7)z=-(W(i,j,k)+.5*dW(i,j,k))*ps;
                    else if(channel==8)z=ion1(i,j,k)+st(i,j,k)+dD(i,j,k)+dW(i,j,k)-dC(i,j,k)-ion0(i,j,k)-oldst(i,j,k);
                    else if(channel==9)z=trial(i,j,k)-mid;
                    else if(channel==10)z=base(i,j,k)-ion0(i,j,k);
                    else if(channel==11)z=ion1(i,j,k)-(base(i,j,k)+inc(i,j,k));
                    else if(channel==12)z=next(i,j,k)-(old(i,j,k)+kick(i,j,k));
                    else if(channel==13)z=.5*((ion1(i,j,k)-ion0(i,j,k))-inc(i,j,k)+(next(i,j,k)-old(i,j,k))-kick(i,j,k))*ps;
                    else if(channel==14)z=ion0(i,j,k)+oldst(i,j,k)+D(i,j,k)+W(i,j,k)-C(i,j,k);
                    else if(channel==15)z=ps;
                    else if(channel==16)z=trial(i,j,k)-old(i,j,k);
                    else if(channel==17)z=i==0&&c==1?pi(i,j,k):0.;
                    else if(channel==18)z=i==0&&c==1?pi(i,j,k)*velocity(i,j,k):0.;
                    else if(channel==19)z=(ion0(i,j,k)+oldst(i,j,k)+D(i,j,k)+W(i,j,k)-C(i,j,k))*ps;
                    else if(channel==20)z=(c!=0&&i==wall_index)?0.:W(i,j,k);
                    else if(channel==23)z=cs(i,j,k)*ps;
                    else if(channel==24)z=ion1(i,j,k)-(base(i,j,k)+correlated(i,j,k));
                    else if(channel==25)z=.5*((ion1(i,j,k)-ion0(i,j,k))-correlated(i,j,k)+
                        (next(i,j,k)-old(i,j,k))-kick(i,j,k))*ps;
                    else if(channel==26)z=.5*(inc(i,j,k)-correlated(i,j,k))*ps;
                    else {
                        R const inherited=cf(i,j,k)-ifield(i,j,k)-jf(i,j,k)-df(i,j,k);
                        if(channel==21)z=(C(i,j,k)-ion0(i,j,k)-oldst(i,j,k)-D(i,j,k))-inherited;
                        else z=((C(i,j,k)+dC(i,j,k))-ion1(i,j,k)-st(i,j,k)-
                            (D(i,j,k)+dD(i,j,k))-dW(i,j,k))-inherited;
                    }
                    out(i,j,k)=z;
                });
            }
            bool const finite=scratch[c].is_finite(0,1,0);if(!All(finite))return Fail("nonfinite actual endpoint work operand");
            R const integral=Integral(scratch[c],geometry),maximum=scratch[c].norm0();
            if(channel==0)l.constraint_work+=integral;
            else if(channel==1)l.mean_current_work+=integral;
            else if(channel==2)l.electron_momentum_change[c]=-PhysConst::m_e/PhysConst::q_e*integral;
            else if(channel==3&&c==2)l.axial_electric_transfer=sums[7]-integral;
            else if(channel==4&&c==2)l.axial_image_transfer=sums[5]-integral;
            else if(channel==5)l.curl_work+=integral;
            else if(channel==6)l.displacement_work+=integral;
            else if(channel==7)l.conductor_work+=integral;
            else if(channel==8||channel==9)l.grid_residual=std::max(l.grid_residual,maximum);
            else if(channel==10)work.current_base_remainder=std::max(work.current_base_remainder,maximum);
            else if(channel==11)work.ion_increment_remainder=std::max(work.ion_increment_remainder,maximum);
            else if(channel==12)work.electron_increment_remainder=std::max(work.electron_increment_remainder,maximum);
            else if(channel==13)work.current_representation_work+=integral;
            else if(channel==14)bg.current_residual=std::max(bg.current_residual,maximum);
            else if(channel==15)l.electric_impulse[c]=integral/volume;
            else if(channel==16)l.mean_current_increment[c]=integral/volume;
            else if(channel==17)l.axis_reaction_conjugate+=integral;
            else if(channel==18)l.axis_reaction_work+=integral;
            else if(channel==19)bg.initial_constraint_work+=integral;
            else if(channel==20&&maximum!=0.)return Fail("initial conductor current outside tangential wall");
            else if(channel==21)work.background_identity_error=std::max(work.background_identity_error,maximum);
            else if(channel==22)work.preserved_defect_error=std::max(work.preserved_defect_error,maximum);
            else if(channel==23)work.static_current_work+=integral;
            else if(channel==24)work.correlated_ion_increment_remainder=std::max(work.correlated_ion_increment_remainder,maximum);
            else if(channel==25)work.correlated_current_representation_work+=integral;
            else if(channel==26)work.theta_registration_work+=integral;
        }
        l.conductor_current_norm=std::max(l.conductor_current_norm,dw[c].norm0());
        l.gauge_error=std::max(l.gauge_error,plpsi[c].norm0()/psi_scale);
        l.magnetic_norm=std::max(l.magnetic_norm,db[c].norm0());
        amrex::MultiFab term(db[c].boxArray(),db[c].DistributionMap(),1,0);
        for(int channel=0;channel<6;++channel){for(amrex::MFIter it(term);it.isValid();++it){auto out=term.array(it);
            auto b=b0[c].const_array(it),d=db[c].const_array(it),fixed=bstatic[c].const_array(it);
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                R const cross=b(i,j,k)*d(i,j,k)/PhysConst::mu0;
                out(i,j,k)=channel==0?.5*d(i,j,k)*d(i,j,k)/PhysConst::mu0:
                    channel==1?cross:channel==2?std::abs(cross):
                    channel==3?.5*b(i,j,k)*b(i,j,k)/PhysConst::mu0:
                    channel==4?fixed(i,j,k)*d(i,j,k)/PhysConst::mu0:
                    std::abs(fixed(i,j,k)*d(i,j,k)/PhysConst::mu0);});}
            bool const finite=term.is_finite(0,1,0);if(!All(finite))return Fail("nonfinite magnetic work");
            R const value=Integral(term,geometry);
            if(channel==0)bg.magnetic_self_work+=value;else if(channel==1)bg.magnetic_cross_work+=value;
            else if(channel==2)bg.magnetic_cross_scale+=value;
            else if(channel==3)l.magnetic_initial_energy+=value;
            else if(channel==4)work.static_magnetic_cross_work+=value;
            else work.static_magnetic_cross_scale+=value;
        }
    }
    l.axial_momentum_defect=l.ion_momentum_change[2]+l.electron_momentum_change[2];
    l.momentum_error=std::abs(l.axial_momentum_defect-l.axial_electric_transfer-l.axial_image_transfer);
    l.electric_momentum_transfer=std::abs(l.axial_electric_transfer);
    l.grid_bound=arithmetic_factor*(current_scale+coefficient_scale);
    bg.current_bound=arithmetic_factor*background_scale;bg.magnetic_bound=4096.*eps*(a_scale+b_scale);
    bg.displacement_bound=4096.*eps*d_scale;l.faraday_bound=4096.*eps*da_scale;l.displacement_curl_bound=4096.*eps*dd_scale;
    // Fresh native unmasked curls, with the same bound. Native Darwin owns
    // B = B_static + curl(A); B_static is not part of A and is never removed
    // from total magnetic energy/current work. A root grants no D allowance.
    for(int channel=0;channel<4;++channel){for(auto& f:bcheck)f.setVal(0.);
        auto out=Mutable(bcheck),in=channel==0?Mutable(a0):channel==1?Mutable(d0):channel==2?Mutable(da):Mutable(dd);
        sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(out,in,sim.GetEBUpdateBFlag()[0],0);
        for(int c=0;c<3;++c){if(channel==0){
                amrex::MultiFab::Add(bcheck[c],bstatic[c],0,0,1,0);
                amrex::MultiFab::Subtract(bcheck[c],b0[c],0,0,1,0);
            }
            if(channel==2)amrex::MultiFab::Subtract(bcheck[c],db[c],0,0,1,0);
            R const value=bcheck[c].norm0();bool const finite=bcheck[c].is_finite(0,1,0);
            if(!All(finite))return Fail("nonfinite native work curl");
            if(channel==0)bg.magnetic_residual=std::max(bg.magnetic_residual,value);
            else if(channel==1)bg.displacement_residual=std::max(bg.displacement_residual,value);
            else if(channel==2)l.faraday_error=std::max(l.faraday_error,value);
            else l.displacement_curl=std::max(l.displacement_curl,value);}
    }
    work.background_identity_bound=bg.current_bound;
    work.preserved_defect_bound=bg.current_bound+l.grid_bound;
    work.inherited_ampere_work=bg.initial_constraint_work;
    work.source_increment_constraint_work=l.constraint_work-bg.initial_constraint_work;
    // The positive helper keeps its nearly closed background gate. This
    // typed adapter preserves the measured inherited defect instead. Its
    // finite magnitude and signed work remain outside all heat allowances.
    if(work.background_identity_error>work.background_identity_bound||
       work.preserved_defect_error>work.preserved_defect_bound||
       bg.magnetic_residual>bg.magnetic_bound||bg.displacement_residual>bg.displacement_bound)
        return Fail("actual source field-origin identity or background curl gate");
    l.magnetic_energy_change=bg.magnetic_cross_work+bg.magnetic_self_work;l.curl_work+=l.magnetic_energy_change;
    work.static_background=true;
    work.static_curl_work=work.static_current_work+work.static_magnetic_cross_work;
    work.dynamic_curl_work=l.curl_work-work.static_curl_work;
    amrex::MultiFab cell(u0.boxArray(),u0.DistributionMap(),1,0);
    amrex::MultiFab::LinComb(cell,1.,u1,0,-1.,u0,0,0,1,0);R const delivered=Integral(cell,geometry);
    R const increment=Integral(du,geometry);work.energy_storage_remainder=delivered-increment;
    l.heat_transfer_error=delivered-l.heat;l.positive_energy_scale=l.ion_initial_energy+l.electron_initial_energy;
    l.actual_energy_defect=l.ion_energy_change+l.electron_energy_change+l.magnetic_energy_change+delivered;
    R carry_work=work.carry.free_work+work.carry.transition_work+work.carry.fixed_work;
    if(AcceptedOrigin())carry_work+=work.carry.accepted_work;
    l.accounting_defect=l.actual_energy_defect-(l.relativistic_defect+l.spatial_transfer+l.constraint_work+
        l.mean_current_work+l.transpose_work+work.material_endpoint_rounding+l.heat_transfer_error+
        l.curl_work+l.displacement_work+l.conductor_work+carry_work);
    l.arithmetic_bound=arithmetic_factor*(l.positive_energy_scale+Integral(u0,geometry)+l.absolute_work+
        std::abs(l.electron_energy_change)+l.magnetic_initial_energy+bg.magnetic_self_work+bg.magnetic_cross_scale);
    auto& t=work.thermal;t.delivered_heat=delivered;t.restricted_heat=q.heat_delivered;
    t.heat_map_error=t.restricted_heat-l.heat;t.storage_error=work.energy_storage_remainder;
    t.signed_residual=q.signed_thermal_residual;
    amrex::MultiFab::Copy(cell,*material.thermal_residual,0,0,1,0);cell.abs(0,1,0);t.l1_residual=Integral(cell,geometry);
    t.residual_identity_error=t.delivered_heat-t.restricted_heat-t.signed_residual-t.storage_error;
    t.residual_bound=std::nextafter(volume*energy_scale*1.e-12,std::numeric_limits<R>::infinity());
    R positive=0.;for(auto const* f:std::array<amrex::MultiFab const*,5>{&u0,&u1,&du,material.cell_heat,material.thermal_residual}){
        amrex::MultiFab::Copy(cell,*f,0,0,1,0);cell.abs(0,1,0);positive+=Integral(cell,geometry);}
    R const operations=64.+16.*(R(geometry.Domain().numPts())+l.contribution_count);
    if(!All(std::isfinite(positive)&&std::isfinite(operations)&&operations*eps<.01))return Fail("thermal accounting arithmetic scope");
    t.roundoff_bound=std::nextafter(4.*operations*eps/(1.-operations*eps)*positive+
        operations*std::numeric_limits<R>::denorm_min(),std::numeric_limits<R>::infinity());
    bool thermal=std::isfinite(t.residual_identity_error)&&std::isfinite(t.roundoff_bound)&&
        t.l1_residual<=t.residual_bound+t.roundoff_bound&&std::abs(t.residual_identity_error)<=t.roundoff_bound&&
        std::abs(delivered-l.heat-t.heat_map_error-t.storage_error-t.signed_residual)<=t.roundoff_bound;
    SpatialStoppingOptions physical;physical.fast_species=options.physical.fast_species;physical.interval=interval;
    physical.coulomb_log=options.physical.coulomb_log;physical.proper_speed_cap=options.physical.proper_speed_cap;
    physical.relative_convention_budget=options.relative_convention_budget;
    l.iterations=t.nonlinear.newton_iterations;l.linear_iterations=t.nonlinear.linear_iterations;l.evaluations=t.nonlinear.residual_evaluations;
    // PEC/axis scalar Dirichlet removes the Cartesian periodic constant mode;
    // no Cartesian mean subtraction or omitted-current allowance is used.
    if(!All(thermal&&l.gauge_error<=1.e-12&&NativeRZSpatialStoppingEvent::ValidateLedger(l,physical)))
        return Fail("actual source physical work/current gate");
    return Matches();
#endif
}
} // namespace warpx::thermal
