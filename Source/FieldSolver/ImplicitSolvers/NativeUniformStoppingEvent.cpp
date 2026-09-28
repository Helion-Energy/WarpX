/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeUniformStoppingEvent.H"
#include "DarwinInitialRateSchur.H"
#include "NativeInstantaneousIonCurrent.H"
#include "NativeStoppingReactionTranspose.H"
#include "ThermalRandomCheckpoint.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Collision/HybridElectronStopping/NativeStoppingMap.H"
#include "Particles/Deposition/EsirkepovCurrentRate.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "WarpX.H"
#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/utils/Communication.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace warpx::thermal {
namespace {
using R=amrex::Real;
using P=amrex::ParticleReal;
using V=amrex::GpuArray<R,3>;
using PC=WarpXParticleContainer;
using CV=ablastr::fields::ConstVectorField;
using FV=ablastr::fields::VectorField;
constexpr R epsilon=std::numeric_limits<R>::epsilon();
AMREX_GPU_HOST_DEVICE R Norm (V const& a) { return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
AMREX_GPU_HOST_DEVICE R Dot (V const& a,V const& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
AMREX_GPU_HOST_DEVICE R Gamma (V const& u) { return std::sqrt(1.+Dot(u,u)/(PhysConst::c*PhysConst::c)); }
AMREX_GPU_HOST_DEVICE R DeltaK (V const& a,V const& b,R m) {
    V d{},s{};for(int c=0;c<3;++c){d[c]=b[c]-a[c];s[c]=b[c]+a[c];}
    return m*Dot(d,s)/(Gamma(a)+Gamma(b));
}
R Roundoff (R k) { return k*epsilon/(1.-k*epsilon); }
bool Collective (bool value) { amrex::ParallelDescriptor::ReduceBoolAnd(value);return value; }
CV Const (std::array<amrex::MultiFab,3> const& a) { return {&a[0],&a[1],&a[2]}; }
FV Mutable (std::array<amrex::MultiFab,3>& a) { return {&a[0],&a[1],&a[2]}; }
void Copy (amrex::MultiFab& to,amrex::MultiFab const& from) {
    amrex::MultiFab::Copy(to,from,0,0,from.nComp(),from.nGrowVect());
}
void AllocateLike (amrex::MultiFab& a,amrex::MultiFab const& b) {
    a.define(b.boxArray(),b.DistributionMap(),b.nComp(),b.nGrowVect());Copy(a,b);
}
void ByteCompare (void const* a,void const* b,std::size_t n,int* bad) {
    auto const* x=static_cast<unsigned char const*>(a);auto const* y=static_cast<unsigned char const*>(b);
    amrex::For(n,[=] AMREX_GPU_DEVICE(std::size_t i){
        if(x[i]!=y[i]){amrex::HostDevice::Atomic::Add(bad,1);}
    });
}
bool Equal (amrex::MultiFab const& a,amrex::MultiFab const& b) {
    if(a.boxArray()!=b.boxArray()||a.DistributionMap()!=b.DistributionMap()||
       a.nComp()!=b.nComp()||a.nGrowVect()!=b.nGrowVect()){return false;}
    amrex::Gpu::DeviceScalar<int> invalid(0);
    for(amrex::MFIter mfi(a);mfi.isValid();++mfi){
        ByteCompare(a[mfi].dataPtr(),b[mfi].dataPtr(),a[mfi].size()*sizeof(R),invalid.dataPtr());
    }
    amrex::Gpu::synchronize();return invalid.dataValue()==0;
}
R Integral (amrex::MultiFab const& f,amrex::Geometry const& geom) {
    auto owner=f.OwnerMask(geom.periodicity());
    R const volume=AMREX_D_TERM(geom.CellSize(0),*geom.CellSize(1),*geom.CellSize(2));
    amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);
    using T=typename decltype(data)::Type;
    for(amrex::MFIter mfi(f);mfi.isValid();++mfi){
        auto const a=f.const_array(mfi);auto const own=owner->const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
            return {own(i,j,k)?a(i,j,k)*volume:0.};
        });
    }
    R result=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(result);return result;
}
void Sync (std::array<amrex::MultiFab,3>& f,amrex::Geometry const& geom) {
    for(auto& a:f){a.OverrideSync(geom.periodicity());a.FillBoundary(geom.periodicity());}
}
std::string HostRandom () { std::ostringstream s;amrex::SaveRandomState(s);return s.str(); }
struct Point {
    V position{},old{},minus{},drag{},next{},ve{},psi{};
    R weight=0.,mass=0.,charge=0.,rate=0.,heat=0.;
};
}

struct NativeUniformStoppingEvent::Impl {
    struct Species {
        PC* pc=nullptr;
        std::vector<std::string> real_names,int_names;
        PC::ParticleLevel old;
        std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<Point>> points;
        std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<StoppingIonImpulse>> impulses;
        R mass=0.,charge=0.;bool fast=false;
    };
    WarpX& sim;NativeAcceptedStoppingContext const& context;
    amrex::MultiFab const& temperature;amrex::MultiFab& energy;UniformStoppingOptions options;
    std::vector<Species> species;std::vector<std::string> names;
    std::array<amrex::MultiFab,3> j0,j1,mean,ve,mass,impulse,i0,i1,scratch;
    amrex::MultiFab rho0,t0,u0,u1,heat;
    std::unique_ptr<NativeStoppingReactionTranspose> reaction;
    amrex::Gpu::DeviceVector<R> sums;
    amrex::Gpu::DeviceVector<unsigned char> device_rng;
    std::string host_rng;std::uint64_t generation=0;amrex::Long next_id=0;
    R time=0.,volume=0.,reference_rho=0.,reference_temperature=0.,current_scale=0.,psi_scale=0.;
    R arithmetic_factor=0.;int step=0;bool ready=false,committed=false,closed=false;
    char const* failure="unprepared";UniformStoppingLedger ledger;
    AcceptedStoppingBinding binding;FV published{};
    Impl(WarpX& s,NativeAcceptedStoppingContext const& c,amrex::MultiFab const& t,
         amrex::MultiFab& u,UniformStoppingOptions const& o):sim(s),context(c),temperature(t),energy(u),options(o){}
    bool Reject(char const* why){ready=false;failure=why;return false;}
    bool Scope();bool Capture();bool Unchanged(bool after);bool Evaluate(std::array<R,6> const&,std::array<R,6>&);
    bool Solve();bool Work();bool Publish(FV const&,NativeUniformStoppingEvent::InvalidationHook);
    void Deposit(Species&,WarpXParIter const&,Point const*,long);
};

bool NativeUniformStoppingEvent::Impl::Scope () {
#if !defined(WARPX_DIM_3D) || defined(AMREX_SINGLE_PRECISION_PARTICLES)
    return false;
#else
    bool ok=context.Valid()&&context.Options().model_electron_mass==PhysConst::m_e&&
        !context.Binding().pedestal&&sim.finestLevel()==0&&sim.maxLevel()==0&&!EB::enabled()&&
        !sim.getdo_moving_window()&&!sim.get_load_balance_intervals().isActivated()&&
        sim.Geom(0).Domain().smallEnd()==amrex::IntVect(0)&&WarpX::grid_type==GridType::Staggered&&WarpX::nox==3&&
        !WarpX::use_filter&&!WarpX::do_single_precision_comms&&!sim.do_current_centering&&
        !WarpX::galerkin_interpolation&&energy.is_finite()&&energy.min(0)>=0.&&WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov&&
        std::isfinite(options.interval)&&options.interval>=0.&&options.coulomb_log>0.&&
        std::isfinite(options.coulomb_log)&&options.proper_speed_cap>0.&&
        options.proper_speed_cap<.01*PhysConst::c&&options.relative_convention_budget>0.&&
        std::isfinite(options.relative_convention_budget)&&!options.fast_species.empty();
    for(int d=0;d<3;++d){ok=ok&&sim.Geom(0).isPeriodic(d);}
    auto const magnetic=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    for(auto const* field:magnetic){ok=ok&&field->is_finite()&&field->norm0()==0.;}
    ok=ok&&energy.ixType().cellCentered()&&energy.nComp()==1&&
        energy.boxArray()==context.Cells()&&energy.DistributionMap()==context.Distribution()&&
        temperature.ixType().nodeCentered()&&temperature.nComp()==1&&
        temperature.boxArray()==amrex::convert(context.Cells(),amrex::IntVect(1))&&
        temperature.DistributionMap()==context.Distribution()&&temperature.nGrowVect().allGE(1);
    return ok;
#endif
}
bool NativeUniformStoppingEvent::Impl::Capture () {
    auto const& g=context.Geometry();volume=g.ProbSize();generation=context.Generation();binding=context.Binding();
    time=sim.gett_new(0);step=sim.getistep(0);next_id=PC::ParticleType::the_next_id;
    names=sim.GetPartContainer().GetSpeciesNames();species.clear();
    R count=0.;bool found=false,species_valid=true;
    for(auto const& name:names){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        if(!dynamic_cast<PhysicalParticleContainer*>(&pc)||pc.finestLevel()!=0||
           pc.DoFieldIonization()||pc.do_not_deposit||pc.getCharge()<=0.||pc.getMass()<=0.){
            species_valid=false;
        }
        species.emplace_back();auto& s=species.back();s.pc=&pc;s.mass=pc.getMass();s.charge=pc.getCharge();
        s.fast=name==options.fast_species;found=found||s.fast;
        s.real_names=pc.GetRealSoANames();s.int_names=pc.GetIntSoANames();
        for(auto const& [key,tile]:pc.GetParticles(0)){
            if(tile.numNeighborParticles()!=0){species_valid=false;}
            auto& copy=s.old[key];copy.define(tile.NumRealComps()-PIdx::nattribs,
                tile.NumIntComps()-IntIdx::nattribs,nullptr,nullptr,pc.arena());
            copy.GetStructOfArrays()=tile.GetStructOfArrays();
            s.points[key].resize(tile.numParticles());s.impulses[key].resize(tile.numParticles());
            count+=tile.numParticles();
        }
    }
    if(!Collective(found&&species_valid)){return Reject("unsupported or missing fast species");}
    amrex::ParallelDescriptor::ReduceRealSum(count);if(count<=0){return Reject("empty event");}
    ledger.contribution_count=count;
    // Every point can contribute 64 cubic terms, with at most eight periodic
    // images and eight shared-FAB owners. 4096 covers both product evaluations,
    // scalar/cubic gather, mass arithmetic and each reduction path. Bounds use
    // ALL global points, not occupancy estimates or observed cancellation.
    ledger.operations=4096.+4096.*count;
    if(ledger.operations*epsilon>=.01){return Reject("unsupported arithmetic depth");}
    arithmetic_factor=4.*Roundoff(ledger.operations);
    AllocateLike(rho0,*binding.raw_charge);AllocateLike(t0,temperature);
    AllocateLike(u0,energy);AllocateLike(u1,energy);
    heat.define(t0.boxArray(),t0.DistributionMap(),1,t0.nGrowVect());heat.setVal(0.);
    for(int c=0;c<3;++c){
        auto const& a=*binding.electron_current[c];
        for(auto* f:{&j0[c],&j1[c],&mean[c],&ve[c],&mass[c],&impulse[c],&i0[c],&i1[c],&scratch[c]}){
            AllocateLike(*f,a);
        }
        impulse[c].setVal(1.);
    }
    warpx::darwin::ApplyYeeInertiaMass(g,context.Kappa(),Const(impulse),Mutable(mass),
        PhysConst::m_e/(PhysConst::q_e*PhysConst::q_e*context.Options().reference_number_density));
    if(!warpx::particles::DepositNativeInstantaneousIonCurrent(sim,
        warpx::particles::InstantaneousIonState::Current,Mutable(i0))){return Reject("initial current");}
    reference_rho=Integral(rho0,g)/volume;reference_temperature=Integral(t0,g)/volume;
    if(!std::isfinite(reference_rho)||reference_rho<=PhysConst::q_e*context.Options().number_density_floor||
       !std::isfinite(reference_temperature)||reference_temperature<=0.){return Reject("invalid density/temperature");}
    ledger.uniform_error=0.;ledger.uniform_bound=arithmetic_factor;
    amrex::MultiFab rtmp(rho0.boxArray(),rho0.DistributionMap(),1,0);
    for(amrex::MFIter mfi(rtmp);mfi.isValid();++mfi){
        auto const a=rtmp.array(mfi);auto const r=rho0.const_array(mfi),t=t0.const_array(mfi);
        R const rref=reference_rho,tref=reference_temperature;
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            a(i,j,k)=amrex::max(std::abs(r(i,j,k)/rref-1.),std::abs(t(i,j,k)/tref-1.));
        });
    }
    ledger.uniform_error=rtmp.norm0();
    amrex::MultiFab deposited(rho0.boxArray(),rho0.DistributionMap(),rho0.nComp(),rho0.nGrowVect());
    deposited.setVal(0.);sim.GetPartContainer().DepositCharge({&deposited},0.);
    sim.SyncRho({&deposited},{},{});
    amrex::MultiFab::LinComb(rtmp,1.,deposited,0,-1.,rho0,0,0,1,0);
    ledger.uniform_error=std::max(ledger.uniform_error,rtmp.norm0()/reference_rho);
    current_scale=reference_rho*options.proper_speed_cap;
    psi_scale=PhysConst::m_p*options.proper_speed_cap/PhysConst::q_e;
    ledger.current_scale=current_scale;
    ledger.current_initial_error=0.;
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(scratch[c],1.,j0[c],0,1.,i0[c],0,0,1,0);
        ledger.current_initial_error=std::max(ledger.current_initial_error,scratch[c].norm0());
        R const average=Integral(j0[c],g)/volume;
        Copy(scratch[c],j0[c]);scratch[c].plus(-average,0,1,0);
        ledger.uniform_error=std::max(ledger.uniform_error,scratch[c].norm0()/current_scale);
    }
    if(ledger.uniform_error>ledger.uniform_bound||ledger.current_initial_error>arithmetic_factor*current_scale){
        return Reject("nonuniform or unconstrained harmonic context");
    }
    host_rng=HostRandom();
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    device_rng.resize(ThermalRandomDeviceBytes());
    amrex::Gpu::dtod_memcpy(device_rng.data(),amrex::getRandState(),device_rng.size());
#endif
    reaction=std::make_unique<NativeStoppingReactionTranspose>(context);sums.resize(32);
    amrex::Gpu::synchronize();return true;
}

bool NativeUniformStoppingEvent::Impl::Unchanged (bool after) {
    bool ok=(after||(context.Valid()&&context.Generation()==generation))&&sim.gett_new(0)==time&&
        sim.boxArray(0)==context.Cells()&&sim.DistributionMap(0)==context.Distribution()&&
        sim.getistep(0)==step&&PC::ParticleType::the_next_id==next_id&&HostRandom()==host_rng&&
        names==sim.GetPartContainer().GetSpeciesNames()&&Equal(rho0,*binding.raw_charge)&&
        Equal(t0,temperature)&&Equal(after?u1:u0,energy);
    for(int c=0;c<3;++c){ok=ok&&Equal(after?j1[c]:j0[c],*binding.electron_current[c]);}
    amrex::Gpu::DeviceScalar<int> bad(0);
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    if(device_rng.size()!=ThermalRandomDeviceBytes()){ok=false;}
    else{ByteCompare(device_rng.data(),amrex::getRandState(),device_rng.size(),bad.dataPtr());}
#endif
    for(auto const& s:species){
        auto const& pc=*s.pc;
        if(pc.GetRealSoANames()!=s.real_names||pc.GetIntSoANames()!=s.int_names||
           pc.getMass()!=s.mass||pc.getCharge()!=s.charge||pc.GetParticles(0).size()!=s.old.size()){
            ok=false;continue;
        }
        for(auto const& [key,old]:s.old){
            auto const it=pc.GetParticles(0).find(key);if(it==pc.GetParticles(0).end()){ok=false;continue;}
            auto const& now=it->second;
            if(now.numParticles()!=old.numParticles()||now.numNeighborParticles()!=0||
               now.NumRealComps()!=old.NumRealComps()||now.NumIntComps()!=old.NumIntComps()){
                ok=false;continue;
            }
            auto const& a=now.GetStructOfArrays();auto const& b=old.GetStructOfArrays();
            ByteCompare(a.GetIdCPUData().data(),b.GetIdCPUData().data(),a.GetIdCPUData().size()*sizeof(std::uint64_t),bad.dataPtr());
            for(int c=0;c<now.NumRealComps();++c){
                if(after&&(c==PIdx::ux||c==PIdx::uy||c==PIdx::uz)){
                    auto const* p=s.points.at(key).data();auto const* u=a.GetRealData(c).data();
                    int const component=c-PIdx::ux;auto* invalid=bad.dataPtr();
                    amrex::For(now.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                        auto const* expected=reinterpret_cast<unsigned char const*>(&p[n].next[component]);
                        auto const* actual=reinterpret_cast<unsigned char const*>(u+n);
                        for(std::size_t b0=0;b0<sizeof(P);++b0){if(expected[b0]!=actual[b0]){amrex::HostDevice::Atomic::Add(invalid,1);}}
                    });
                }else{ByteCompare(a.GetRealData(c).data(),b.GetRealData(c).data(),a.GetRealData(c).size()*sizeof(P),bad.dataPtr());}
            }
            for(int c=0;c<now.NumIntComps();++c){ByteCompare(a.GetIntData(c).data(),b.GetIntData(c).data(),a.GetIntData(c).size()*sizeof(int),bad.dataPtr());}
        }
    }
    amrex::Gpu::synchronize();return Collective(ok&&bad.dataValue()==0);
}

void NativeUniformStoppingEvent::Impl::Deposit (Species& s,WarpXParIter const& pti,Point const* p,long n) {
#if defined(WARPX_DIM_3D)
    auto box=pti.tilebox();box.grow(sim.get_ng_depos_J());auto const origin=WarpX::LowerCorner(box,0,0.);
    auto const lower=amrex::lbound(box);auto const inverse=WarpX::InvCellSize(0);
    auto const jx=i1[0].array(pti),jy=i1[1].array(pti),jz=i1[2].array(pti);
    amrex::ignore_unused(s);
    amrex::For(n,[=] AMREX_GPU_DEVICE(long n0){
        auto const point=p[n0];auto const u=point.next;R const gamma=Gamma(u);
        V x{(point.position[0]-origin.x)*inverse.x,(point.position[1]-origin.y)*inverse.y,
            (point.position[2]-origin.z)*inverse.z};
        V v{u[0]/gamma*inverse.x,u[1]/gamma*inverse.y,u[2]/gamma*inverse.z};
        int first[3]{};for(int c=0;c<3;++c){first[c]=int(x[c])-1;}
        for(int k=first[2];k<=first[2]+3;++k){for(int j=first[1];j<=first[1]+3;++j){for(int i=first[0];i<=first[0]+3;++i){
            V value{},unused{};
            warpx::particles::EsirkepovCurrentRate<3,3>(x,v,V{},amrex::GpuArray<int,3>{i,j,k},0.,0.,value,unused);
            int const ii=i+lower.x,jj=j+lower.y,kk=k+lower.z;
            R const a=point.charge*point.weight*inverse.y*inverse.z;
            R const b=point.charge*point.weight*inverse.x*inverse.z;
            R const c=point.charge*point.weight*inverse.x*inverse.y;
            if(i<first[0]+3){amrex::HostDevice::Atomic::Add(&jx(ii,jj,kk),a*value[0]);}
            if(j<first[1]+3){amrex::HostDevice::Atomic::Add(&jy(ii,jj,kk),b*value[1]);}
            if(k<first[2]+3){amrex::HostDevice::Atomic::Add(&jz(ii,jj,kk),c*value[2]);}
        }}}
    });
#else
    amrex::ignore_unused(s,pti,p,n);
#endif
}

bool NativeUniformStoppingEvent::Impl::Evaluate (std::array<R,6> const& x,std::array<R,6>& residual) {
#if !defined(WARPX_DIM_3D)
    amrex::ignore_unused(x,residual);return false;
#else
    auto const& g=context.Geometry();V psi{},increment{};
    for(int c=0;c<3;++c){psi[c]=x[c]*psi_scale;increment[c]=x[c+3]*current_scale;
        Copy(mean[c],j0[c]);mean[c].plus(increment[c],0,1,mean[c].nGrow());impulse[c].setVal(psi[c]);
        i1[c].setVal(0.);
    }
    warpx::darwin::ApplyYeeInertiaMass(g,context.Kappa(),Const(mean),Mutable(ve),
        PhysConst::m_e/(PhysConst::q_e*PhysConst::q_e*context.Options().reference_number_density));
    for(auto& a:ve){a.mult(-PhysConst::q_e/PhysConst::m_e,0,1,a.nGrow());}
    Sync(ve,g);heat.setVal(0.);R* init=sums.data();amrex::ParallelFor(int(sums.size()),[=] AMREX_GPU_DEVICE(int n){init[n]=0.;});
    amrex::Gpu::DeviceScalar<int> invalid(0);int* bad=invalid.dataPtr();R* totals=sums.data();
    if(!reaction->Begin()){return false;}
    auto const magnetic=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    CV bfield{magnetic[0],magnetic[1],magnetic[2]};
    auto const inverse=g.InvCellSizeArray(),plo=g.ProbLoArray(),dx=g.CellSizeArray();
    R const node_volume=dx[0]*dx[1]*dx[2],interval=options.interval,clog=options.coulomb_log;
    R const cap=options.proper_speed_cap,floor=PhysConst::q_e*context.Options().number_density_floor;
    for(auto& s:species){
        for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
            auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());
            auto const& old=s.old.at(key).GetStructOfArrays();
            GetParticlePosition<PIdx> position;
            position.m_x=old.GetRealData(PIdx::x).data();position.m_y=old.GetRealData(PIdx::y).data();position.m_z=old.GetRealData(PIdx::z).data();
            auto const* ux=old.GetRealData(PIdx::ux).data();auto const* uy=old.GetRealData(PIdx::uy).data();
            auto const* uz=old.GetRealData(PIdx::uz).data();auto const* weight=old.GetRealData(PIdx::w).data();
            auto gather=context.GatherView(pti,bfield),egather=gather;
            for(int c=0;c<3;++c){gather.velocity[c]=ve[c].const_array(pti);egather.velocity[c]=impulse[c].const_array(pti);}
            auto const r=rho0.const_array(pti),t=t0.const_array(pti);auto const h=heat.array(pti);
            auto* points=s.points.at(key).data();auto* impulses=s.impulses.at(key).data();
            R const m=s.mass,q=s.charge;bool const fast=s.fast;
            amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                Point p;p.mass=m;p.charge=q;p.weight=weight[n];p.old={ux[n],uy[n],uz[n]};
                position(n,p.position[0],p.position[1],p.position[2]);
                auto const vel=gather(p.position[0],p.position[1],p.position[2]);
                auto const electric=egather(p.position[0],p.position[1],p.position[2]);
                bool ok=vel.valid&&electric.valid&&p.weight>=0.&&std::isfinite(p.weight);
                p.ve=vel.cartesian;p.psi=electric.cartesian;
                R const rho=ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],r,inverse,plo);
                R const te=amrex::max(ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],t,inverse,plo),1.e-3*PhysConst::q_e/PhysConst::kb);
                ok=ok&&rho>floor&&std::isfinite(te)&&Norm(p.old)<=cap;
                for(int c=0;c<3;++c){p.minus[c]=p.old[c]+q*p.psi[c]/(2.*m);}
                p.drag=p.minus;
                if(fast&&interval>0.){p.rate=warpx::particles::NativeStoppingRate(rho*(1./PhysConst::q_e),te*PhysConst::kb,q,m,clog);
                    p.drag=warpx::particles::NativeStoppingKick(p.minus,p.ve,p.rate,interval,p.weight,m).proper;}
                StoppingIonImpulse tuple;tuple.position=p.position;tuple.weight=p.weight;
                for(int c=0;c<3;++c){p.next[c]=p.drag[c]+q*p.psi[c]/(2.*m);tuple.impulse[c]=m*(p.drag[c]-p.minus[c]);}
                ok=ok&&Norm(p.minus)<=cap&&Norm(p.drag)<=cap&&Norm(p.next)<=cap&&std::isfinite(Norm(p.next));
                R const dki=p.weight*DeltaK(p.old,p.next,m);
                R const drag=p.weight*DeltaK(p.minus,p.drag,m);
                R const bulk=p.weight*Dot(tuple.impulse,p.ve);
                R const electric_work=p.weight*(DeltaK(p.old,p.minus,m)+DeltaK(p.drag,p.next,m));
                V vmean{};R const g0=Gamma(p.old),g1=Gamma(p.next);
                for(int c=0;c<3;++c){vmean[c]=.5*(p.old[c]/g0+p.next[c]/g1);}
                R const endpoint_work=p.weight*q*Dot(p.psi,vmean);
                p.heat=-(drag-bulk);
                ok=ok&&std::isfinite(p.heat)&&p.heat>=0.;
                if(!ok){amrex::HostDevice::Atomic::Add(bad,1);return;}
                points[n]=p;impulses[n]=tuple;
                R values[17]={p.weight*m*Dot(p.old,p.old)/(g0+1.),dki,drag,bulk,electric_work,
                    endpoint_work,p.heat,electric_work-endpoint_work,
                    p.weight*std::abs(q)*Norm(p.psi)*cap*cap*cap/(PhysConst::c*PhysConst::c),
                    std::abs(dki)+std::abs(drag)+std::abs(bulk)+std::abs(electric_work),
                    p.weight*m*(p.next[0]-p.old[0]),p.weight*m*(p.next[1]-p.old[1]),p.weight*m*(p.next[2]-p.old[2]),
                    p.weight*tuple.impulse[0],p.weight*tuple.impulse[1],p.weight*tuple.impulse[2],p.weight*m*cap};
                for(int c=0;c<17;++c){amrex::HostDevice::Atomic::Add(totals+c,values[c]);}
                int ii=0,jj=0,kk=0;R weights[3][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(p.position[0],p.position[1],p.position[2],plo,inverse,ii,jj,kk,weights);
                for(int dz=0;dz<2;++dz){for(int dy=0;dy<2;++dy){for(int dx0=0;dx0<2;++dx0){
                    amrex::HostDevice::Atomic::Add(&h(ii+dx0,jj+dy,kk+dz),p.heat*weights[0][dx0]*weights[1][dy]*weights[2][dz]/node_volume);
                }}}
            });
            amrex::Gpu::streamSynchronize();
            if(invalid.dataValue()==0){reaction->DepositTile(pti,impulses,pti.numParticles());Deposit(s,pti,points,pti.numParticles());}
        }
    }
    amrex::Gpu::synchronize();if(!Collective(invalid.dataValue()==0)){return false;}
    if(!reaction->Finish()){return false;}
    for(auto& a:i1){ablastr::utils::communication::SumBoundary(a,0,1,a.nGrowVect(),a.nGrowVect(),false,g.periodicity());}
    Sync(i1,g);
    ablastr::utils::communication::SumBoundary(heat,0,1,heat.nGrowVect(),heat.nGrowVect(),false,g.periodicity());
    heat.OverrideSync(g.periodicity());heat.FillBoundary(g.periodicity());
    auto const reaction_impulse=reaction->InertiaImpulse();
    for(int c=0;c<3;++c){
        for(amrex::MFIter mfi(j1[c]);mfi.isValid();++mfi){
            auto const a=j1[c].array(mfi);auto const old=j0[c].const_array(mfi),coefficient=mass[c].const_array(mfi),ri=reaction_impulse[c]->const_array(mfi);
            R const value=psi[c];
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=old(i,j,k)+(value+ri(i,j,k))/coefficient(i,j,k);});
        }
        amrex::MultiFab::LinComb(scratch[c],1.,i1[c],0,1.,j1[c],0,0,1,0);
        residual[c]=Integral(scratch[c],g)/(volume*current_scale);
        for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
            auto const a=scratch[c].array(mfi);auto const old=j0[c].const_array(mfi),next=j1[c].const_array(mfi),trial=mean[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=trial(i,j,k)-.5*(old(i,j,k)+next(i,j,k));});
        }
        residual[c+3]=Integral(scratch[c],g)/(volume*current_scale);
    }
    Sync(j1,g);ledger.electric_impulse=psi;ledger.mean_current_increment=increment;
    return std::all_of(residual.begin(),residual.end(),[](R r){return std::isfinite(r);});
#endif
}

bool NativeUniformStoppingEvent::Impl::Solve () {
    std::array<R,6> x{},f{};
    auto norm=[](auto const& a){R n=0.;for(R v:a){n=std::max(n,std::abs(v));}return n;};
    for(int iteration=0;iteration<32;++iteration){
        if(!Evaluate(x,f)){return Reject("event candidate inadmissible");}
        ledger.iterations=iteration;ledger.harmonic_residual=norm(f);
        if(ledger.harmonic_residual<=1.e-12){return true;}
        std::array<std::array<R,7>,6> matrix{};
        for(int c=0;c<6;++c){
            R const h=1.e-5*std::max(R(1),std::abs(x[c]));
            auto plus=x,minus=x;plus[c]+=h;minus[c]-=h;
            std::array<R,6> fp{},fm{};
            if(!Evaluate(plus,fp)||!Evaluate(minus,fm)){return Reject("event derivative inadmissible");}
            for(int r=0;r<6;++r){matrix[r][c]=(fp[r]-fm[r])/(2*h);}
        }
        for(int r=0;r<6;++r){matrix[r][6]=-f[r];}
        for(int c=0;c<6;++c){
            int pivot=c;for(int r=c+1;r<6;++r){if(std::abs(matrix[r][c])>std::abs(matrix[pivot][c])){pivot=r;}}
            if(!std::isfinite(matrix[pivot][c])||std::abs(matrix[pivot][c])<1.e-16){return Reject("singular harmonic event");}
            std::swap(matrix[c],matrix[pivot]);R const divisor=matrix[c][c];
            for(int k=c;k<=6;++k){matrix[c][k]/=divisor;}
            for(int r=0;r<6;++r){if(r==c){continue;}R const a=matrix[r][c];for(int k=c;k<=6;++k){matrix[r][k]-=a*matrix[c][k];}}
        }
        bool found=false;
        for(R length=1.;length>=1./1024.;length*=.5){
            auto trial=x;for(int c=0;c<6;++c){trial[c]+=length*matrix[c][6];}
            std::array<R,6> next{};
            if(Evaluate(trial,next)&&norm(next)<norm(f)){x=trial;found=true;break;}
        }
        if(!found){return Reject("harmonic event no descent");}
    }
    return Reject("harmonic event maximum iterations");
}

bool NativeUniformStoppingEvent::Impl::Work () {
    auto const& g=context.Geometry();
    std::vector<R> host(sums.size());amrex::Gpu::copy(amrex::Gpu::deviceToHost,sums.begin(),sums.end(),host.begin());
    amrex::ParallelDescriptor::ReduceRealSum(host.data(),host.size());
    ledger.ion_initial_energy=host[0];ledger.ion_energy_change=host[1];ledger.drag_energy_change=host[2];
    ledger.drag_bulk_work=host[3];ledger.electric_particle_work=host[4];ledger.endpoint_particle_work=host[5];
    ledger.heat=host[6];ledger.relativistic_defect=host[7];ledger.relativistic_bound=host[8];
    ledger.absolute_work=host[9];ledger.positive_momentum_scale=host[16]+
        volume*reference_rho*PhysConst::m_e/PhysConst::q_e*options.proper_speed_cap;
    for(int c=0;c<3;++c){ledger.ion_momentum_change[c]=host[10+c];ledger.reaction_momentum[c]=-host[13+c];}
    auto const b=reaction->InertiaImpulse();
    ledger.momentum_error=0.;ledger.momentum_bound=arithmetic_factor*ledger.positive_momentum_scale;
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(scratch[c],1.,j1[c],0,-1.,j0[c],0,0,1,0);
        ledger.electron_momentum_change[c]=-PhysConst::m_e/PhysConst::q_e*Integral(scratch[c],g);
        ledger.momentum_error=std::max(ledger.momentum_error,std::abs(ledger.ion_momentum_change[c]+ledger.electron_momentum_change[c]));
    }
    R gridwork=0.,trialb=0.;ledger.electron_initial_energy=0.;ledger.electron_energy_change=0.;
    ledger.constraint_work=0.;ledger.mean_current_work=0.;ledger.grid_residual=0.;
    for(int c=0;c<3;++c){
        R const psi=ledger.electric_impulse[c];
        for(int channel=0;channel<6;++channel){
            for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
                auto const a=scratch[c].array(mfi);auto const old=j0[c].const_array(mfi),next=j1[c].const_array(mfi);
                auto const trial=mean[c].const_array(mfi),m=mass[c].const_array(mfi),ri=b[c]->const_array(mfi);
                auto const ion0=i0[c].const_array(mfi),ion1=i1[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    R const midpoint=.5*(old(i,j,k)+next(i,j,k));
                    if(channel==0){a(i,j,k)=.5*old(i,j,k)*m(i,j,k)*old(i,j,k);}
                    else if(channel==1){a(i,j,k)=midpoint*m(i,j,k)*(next(i,j,k)-old(i,j,k));}
                    else if(channel==2){a(i,j,k)=.5*(ion0(i,j,k)+ion1(i,j,k))*psi;}
                    else if(channel==3){a(i,j,k)=(.5*(ion0(i,j,k)+ion1(i,j,k))+midpoint)*psi;}
                    else if(channel==4){a(i,j,k)=(midpoint-trial(i,j,k))*ri(i,j,k);}
                    else{a(i,j,k)=trial(i,j,k)*ri(i,j,k);}
                });
            }
            R const value=Integral(scratch[c],g);
            if(channel==0){ledger.electron_initial_energy+=value;}
            else if(channel==1){ledger.electron_energy_change+=value;}
            else if(channel==2){gridwork+=value;}
            else if(channel==3){ledger.constraint_work+=value;}
            else if(channel==4){ledger.mean_current_work+=value;}
            else{trialb+=value;}
        }
        amrex::MultiFab::LinComb(scratch[c],1.,i1[c],0,1.,j1[c],0,0,1,0);
        ledger.grid_residual=std::max(ledger.grid_residual,scratch[c].norm0());
        for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
            auto const a=scratch[c].array(mfi);auto const old=j0[c].const_array(mfi),next=j1[c].const_array(mfi),trial=mean[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=trial(i,j,k)-.5*(old(i,j,k)+next(i,j,k));});
        }
        ledger.grid_residual=std::max(ledger.grid_residual,scratch[c].norm0());
    }
    ledger.spatial_transfer=ledger.endpoint_particle_work-gridwork;
    ledger.transpose_work=ledger.drag_bulk_work+trialb;
    Copy(u1,u0);
#if defined(WARPX_DIM_3D)
    for(amrex::MFIter mfi(u1);mfi.isValid();++mfi){
        auto const a=u1.array(mfi);auto const q=heat.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            R value=0.;for(int dz=0;dz<2;++dz){for(int dy=0;dy<2;++dy){for(int dx=0;dx<2;++dx){value+=q(i+dx,j+dy,k+dz)/8.;}}}
            a(i,j,k)+=value;
        });
    }
#endif
    u1.FillBoundary(g.periodicity());
    amrex::MultiFab change(u1.boxArray(),u1.DistributionMap(),1,0);
    amrex::MultiFab::LinComb(change,1.,u1,0,-1.,u0,0,0,1,0);
    R const delivered=Integral(change,g);
    ledger.heat_transfer_error=delivered-ledger.heat;
    ledger.positive_energy_scale=ledger.ion_initial_energy+ledger.electron_initial_energy;
    ledger.actual_energy_defect=ledger.ion_energy_change+ledger.electron_energy_change+delivered;
    ledger.accounting_defect=ledger.actual_energy_defect-(ledger.relativistic_defect+
        ledger.spatial_transfer+ledger.constraint_work+ledger.mean_current_work+
        ledger.transpose_work+ledger.heat_transfer_error);
    // Roundoff scales are positive input/storage/work magnitudes. The reported
    // signed defects never set a tolerance or get deposited as extra heat.
    R coefficient_scale=0.;
    for(int c=0;c<3;++c){coefficient_scale+=std::abs(ledger.electric_impulse[c])/mass[c].min(0);}
    ledger.grid_bound=arithmetic_factor*(current_scale+coefficient_scale);
    ledger.arithmetic_bound=arithmetic_factor*(ledger.positive_energy_scale+
        Integral(u0,g)+ledger.absolute_work+std::abs(ledger.electron_energy_change));
    if(ledger.momentum_error>ledger.momentum_bound){return Reject("event momentum gate");}
    if(!NativeUniformStoppingEvent::ValidateLedger(ledger,options)){return Reject("event work/capability gate");}
    return true;
}

NativeUniformStoppingEvent::NativeUniformStoppingEvent(WarpX& w,NativeAcceptedStoppingContext const& c,
    amrex::MultiFab const& t,amrex::MultiFab& u,UniformStoppingOptions const& o)
    :m_impl(std::make_unique<Impl>(w,c,t,u,o)){}
NativeUniformStoppingEvent::~NativeUniformStoppingEvent()=default;
bool NativeUniformStoppingEvent::Prepare () {
    auto& p=*m_impl;
    if(p.committed||p.closed){return p.Reject("event transaction closed");}
    p.ready=false;p.ledger={};
    if(!Collective(p.Scope())){return p.Reject("unsupported harmonic stopping scope");}
    if(!p.Capture()||!p.Solve()||!p.Work()){return false;}
    if(!p.Unchanged(false)){return p.Reject("accepted state changed during preparation");}
    p.ready=true;p.failure="none";return true;
}
bool NativeUniformStoppingEvent::Impl::Publish(FV const& current,NativeUniformStoppingEvent::InvalidationHook hook){
    bool valid=ready&&!committed&&!closed&&hook.function;
    for(int c=0;c<3;++c){valid=valid&&current[c]==binding.electron_current[c];}
    if(!Collective(valid)||!Unchanged(false)){failure="stale or duplicate event commit";return false;}
    for(auto& s:species){for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
        auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());auto const* points=s.points.at(key).data();
        auto* ux=pti.GetAttribs(PIdx::ux).data();auto* uy=pti.GetAttribs(PIdx::uy).data();auto* uz=pti.GetAttribs(PIdx::uz).data();
        amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){ux[n]=points[n].next[0];uy[n]=points[n].next[1];uz[n]=points[n].next[2];});
    }}
    for(int c=0;c<3;++c){Copy(*current[c],j1[c]);}Copy(energy,u1);
    amrex::Gpu::synchronize();committed=true;ready=false;published=current;hook.function(hook.context);return true;
}
bool NativeUniformStoppingEvent::CommitOnce(FV const& current,InvalidationHook hook){return m_impl->Publish(current,hook);}
bool NativeUniformStoppingEvent::Rollback(InvalidationHook hook){
    auto& p=*m_impl;
    if(!Collective(p.committed&&!p.closed&&hook.function)||!p.Unchanged(true)){p.failure="stale event rollback";return false;}
    for(auto& s:p.species){for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
        auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());auto const& soa=s.old.at(key).GetStructOfArrays();
        for(int c:{PIdx::ux,PIdx::uy,PIdx::uz}){auto const& source=soa.GetRealData(c);auto& target=pti.GetAttribs(c);
            amrex::Gpu::copy(amrex::Gpu::deviceToDevice,source.begin(),source.end(),target.begin());}
    }}
    for(int c=0;c<3;++c){Copy(*p.published[c],p.j0[c]);}
    Copy(p.energy,p.u0);amrex::Gpu::synchronize();p.committed=false;p.closed=true;hook.function(hook.context);return true;
}
void NativeUniformStoppingEvent::Finalize(){m_impl->closed=true;m_impl->ready=false;}
bool NativeUniformStoppingEvent::Prepared()const{return m_impl->ready;}
char const* NativeUniformStoppingEvent::Failure()const{return m_impl->failure;}
UniformStoppingLedger const& NativeUniformStoppingEvent::Ledger()const{return m_impl->ledger;}
CV NativeUniformStoppingEvent::CandidateElectronCurrent()const{return Const(m_impl->j1);}
amrex::MultiFab const& NativeUniformStoppingEvent::CandidateEnergy()const{return m_impl->u1;}
bool NativeUniformStoppingEvent::ValidateLedger(UniformStoppingLedger const& l,UniformStoppingOptions const& o){
    R const positive=l.positive_energy_scale,bound=l.arithmetic_bound;
    return std::isfinite(positive)&&positive>0.&&std::isfinite(bound)&&bound>=0.&&
        std::isfinite(l.actual_energy_defect)&&std::isfinite(l.accounting_defect)&&
        l.heat>=0.&&l.relativistic_bound>=0.&&
        l.relativistic_bound<=o.relative_convention_budget*positive&&
        std::abs(l.relativistic_defect)<=l.relativistic_bound+bound&&
        std::abs(l.accounting_defect)<=bound&&
        std::abs(l.heat+l.drag_energy_change-l.drag_bulk_work)<=bound&&
        l.harmonic_residual<=1.e-12&&l.grid_residual<=l.grid_bound&&
        l.uniform_error<=l.uniform_bound&&l.momentum_error<=l.momentum_bound;
}
} // namespace warpx::thermal
