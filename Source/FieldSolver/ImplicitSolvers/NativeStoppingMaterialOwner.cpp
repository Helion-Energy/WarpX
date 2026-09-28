/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeStoppingMaterialOwner.H"
#include "NativeStoppingMaterialReaction.H"
#include "NativeInertiaSupport.H"
#include "NativePECPlasma.H"
#include "NativeInstantaneousIonCurrent.H"
#include "KineticThermalMoments.H"
#include "ThermalRandomCheckpoint.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Particles/Collision/HybridElectronStopping/NativeStoppingMap.H"
#include "Particles/Deposition/EsirkepovCurrentRate.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "WarpX.H"
#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/utils/Communication.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace warpx::thermal {
namespace {
using R=amrex::Real;
using P=amrex::ParticleReal;
using V=amrex::GpuArray<R,3>;
using CV=ablastr::fields::ConstVectorField;
using FV=ablastr::fields::VectorField;
using Fields=std::array<amrex::MultiFab,3>;
using PC=WarpXParticleContainer;
using Key=std::pair<int,int>;
using Point=NativeStoppingOwnerPoint;
using Status=NativeStoppingOwnerStatus;
bool All(bool x){amrex::ParallelDescriptor::ReduceBoolAnd(x);return x;}
CV Const(Fields const& f){return{&f[0],&f[1],&f[2]};}
FV Mutable(Fields& f){return{&f[0],&f[1],&f[2]};}
AMREX_GPU_HOST_DEVICE R Dot(V const& a,V const& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
AMREX_GPU_HOST_DEVICE R Norm(V const& x){return std::sqrt(Dot(x,x));}
AMREX_GPU_HOST_DEVICE R Gamma(V const& u){return std::sqrt(1.+Dot(u,u)/(PhysConst::c*PhysConst::c));}
AMREX_GPU_HOST_DEVICE R DeltaK(V const& a,V const& b,R m){
    V d{},s{};for(int c=0;c<3;++c){d[c]=b[c]-a[c];s[c]=b[c]+a[c];}
    return m*Dot(d,s)/(Gamma(a)+Gamma(b));
}
void Clone(amrex::MultiFab& a,amrex::MultiFab const& b){
    a.define(b.boxArray(),b.DistributionMap(),b.nComp(),b.nGrowVect());
    amrex::MultiFab::Copy(a,b,0,0,b.nComp(),b.nGrowVect());
}
bool Layout(amrex::MultiFab const* a,amrex::MultiFab const& b){
    return a&&a->boxArray()==b.boxArray()&&a->DistributionMap()==b.DistributionMap()&&a->nComp()==1;
}
// Input/output layout checks already bind each component to its native box
// array. An alias MultiFab has a different C++ address but the same FAB data.
// Inspect only storage addresses, never copy field values to the host.
bool SharesStorage(amrex::MultiFab const& a,amrex::MultiFab const& b){
    if(a.boxArray()!=b.boxArray()||a.DistributionMap()!=b.DistributionMap())return false;
    for(amrex::MFIter it(a);it.isValid();++it){
        auto const& x=a[it];auto const& y=b[it];
        auto const xb=reinterpret_cast<std::uintptr_t>(x.dataPtr());
        auto const yb=reinterpret_cast<std::uintptr_t>(y.dataPtr());
        auto const xn=x.size()*sizeof(R),yn=y.size()*sizeof(R);
        if(xb<=yb ? yb-xb<xn : xb-yb<yn)return true;
    }
    return false;
}
void Bytes(void const* a,void const* b,std::size_t n,int* bad){
    auto x=static_cast<unsigned char const*>(a);auto y=static_cast<unsigned char const*>(b);
    amrex::For(n,[=] AMREX_GPU_DEVICE(std::size_t i){
        if(x[i]!=y[i])amrex::Gpu::Atomic::Exch(bad,1);
    });
}
std::string Random(){std::ostringstream s;amrex::SaveRandomState(s);return s.str();}
R Integral(amrex::MultiFab const& f,amrex::Geometry const& g){
    auto owner=f.OwnerMask(g.periodicity());auto volume=MakeQdsmcVolumeElement(g,f.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);
    using T=typename decltype(data)::Type;
    for(amrex::MFIter it(f);it.isValid();++it){auto a=f.const_array(it);auto own=owner->const_array(it);
        op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T{
            return{own(i,j,k)?a(i,j,k)*volume(i,j,k):0.};
        });
    }
    R x=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(x);return x;
}
void Sync(Fields& a,amrex::Geometry const& g){
    for(auto& f:a){f.OverrideSync(g.periodicity());f.FillBoundary(g.periodicity());}
}
}
struct NativeStoppingOwnerToken {bool active=false;std::uint64_t generation=0;};
bool NativeStoppingOwnerLease::Valid() const noexcept {
    return m_token&&m_token->active&&m_generation==m_token->generation;
}
struct NativeStoppingMaterialOwner::Impl {
    struct Species {
        std::string name;PC* pc=nullptr;R mass=0.,charge=0.;bool fast=false;
        std::vector<std::string> real_names,int_names;
        PC::ParticleLevel old;
        std::map<Key,amrex::Gpu::DeviceVector<StoppingMaterialPoint>> footprint;
        std::map<Key,amrex::Gpu::DeviceVector<Point>> points;
        std::map<Key,amrex::Gpu::DeviceVector<StoppingIonImpulse>> impulses;
    };
    struct ProtectedField {std::string name;int component=-1;
        amrex::MultiFab const* original=nullptr;amrex::MultiFab copy;};
    WarpX& sim;AcceptedStoppingOptions material_options;NativeStoppingOwnerOptions options;
    bool const stable_drag_increment;
    amrex::Real source_time=0.,interval=0.,clock=0.,thermal_gamma=0.;std::uint64_t epoch=0;
    int step=0;amrex::Long next_id=0;
    amrex::Geometry geometry;amrex::BoxArray cells;amrex::DistributionMapping distribution;
    std::vector<std::string> names;std::vector<Species> species;
    std::vector<std::unique_ptr<ProtectedField>> protected_fields;
    std::string host_rng;amrex::Gpu::DeviceVector<unsigned char> device_rng;
    AcceptedStoppingBinding binding;
    std::unique_ptr<NativeStoppingMaterialSupport> material;
    StoppingMaterialLease material_lease;
    std::unique_ptr<NativeStoppingMaterialReaction> reaction;
    std::unique_ptr<KineticThermalMoments> moments;
    Fields impulse,mean,zero,aux,i0,i1,physical_pi,stored_endpoint;
    std::shared_ptr<NativeStoppingCarryCertificate const> carry;
    amrex::MultiFab u0,umid,uend,du,heat,cell_heat,thermal_residual;
    amrex::Gpu::DeviceVector<R> sums;
    NativeStoppingOwnerCensus census;NativeStoppingOwnerWork work;
    Status status=Status::Unsupported;std::string reason="unprepared";
    bool captured=false,action=false,published=false;
    Impl(WarpX& w,AcceptedStoppingOptions const& m,NativeStoppingOwnerOptions const& o,
         R time,std::uint64_t n,R dt,std::shared_ptr<NativeStoppingCarryCertificate const> certificate,
         bool stable_increment)
        :sim(w),material_options(m),options(o),stable_drag_increment(stable_increment),
         source_time(time),interval(dt),epoch(n),
         geometry(w.get_pointer_HybridPICModel()->ElectronThermalGeometry()),
         cells(w.boxArray(0)),distribution(w.DistributionMap(0)),carry(std::move(certificate)){}
    bool Fail(Status s,std::string why){status=s;reason=std::move(why);action=false;return false;}
    bool Capture();bool Unchanged(bool after_publication=false);bool Action(NativeStoppingOwnerTrial const&);
    void PublishMomenta();
    void Protect(std::string const& name,int c=-1){
        auto p=std::make_unique<ProtectedField>();p->name=name;p->component=c;
        p->original=c<0?sim.m_fields.get(name,0):
            sim.m_fields.get(name,ablastr::fields::Direction{c},0);
        Clone(p->copy,*p->original);protected_fields.push_back(std::move(p));
    }
    void ElectricImages();
    void Deposit(WarpXParIter const&,Point const*,amrex::Long);
};
NativeStoppingMaterialOwner::NativeStoppingMaterialOwner(WarpX& w,
    AcceptedStoppingOptions const& m,NativeStoppingOwnerOptions const& o,R t,
    std::uint64_t epoch,R dt,std::shared_ptr<NativeStoppingCarryCertificate const> carry,
    bool stable_increment)
    :m_impl(std::make_unique<Impl>(w,m,o,t,epoch,dt,std::move(carry),stable_increment)),
    m_token(std::make_shared<NativeStoppingOwnerToken>()){}
NativeStoppingMaterialOwner::~NativeStoppingMaterialOwner(){Invalidate();}
void NativeStoppingMaterialOwner::Invalidate() noexcept {
    m_token->active=false;
    if(m_token->generation<std::numeric_limits<std::uint64_t>::max())++m_token->generation;
    m_impl->captured=false;m_impl->action=false;
    if(m_impl->material)m_impl->material->Invalidate();
}
NativeStoppingOwnerLease NativeStoppingMaterialOwner::Lease() const {
    NativeStoppingOwnerLease out;
    if(m_impl->captured&&m_token->active){out.m_token=m_token;out.m_generation=m_token->generation;}
    return out;
}
NativeStoppingOwnerResult NativeStoppingMaterialOwner::State() const {
    return{m_impl->status,m_impl->reason,Lease(),m_impl->census};
}
NativeStoppingOwnerResult NativeStoppingMaterialOwner::Capture(){
    Invalidate();if(m_impl->Capture()){m_token->active=true;++m_token->generation;}
    return State();
}
bool NativeStoppingMaterialOwner::Matches(NativeStoppingOwnerLease const& lease){
    bool const local=lease.Valid()&&lease.m_token.get()==m_token.get()&&m_impl->captured;
    if(!All(local)){m_impl->Fail(Status::Stale,"material owner lease expired");Invalidate();return false;}
    bool const material=m_impl->material->Matches(m_impl->material_lease);
    bool const unchanged=m_impl->Unchanged();
    bool const carried=!m_impl->carry||m_impl->carry->ValidateSource(m_impl->sim,m_impl->source_time,m_impl->epoch);
    if(!material||!unchanged||!carried){m_impl->Fail(Status::Stale,"captured material or population changed");Invalidate();return false;}
    return true;
}
NativeStoppingOwnerResult NativeStoppingMaterialOwner::Evaluate(
    NativeStoppingOwnerLease const& lease,NativeStoppingOwnerTrial const& trial){
    if(!Matches(lease))return State();
    ++m_token->generation;m_impl->action=false;
    if(!m_impl->Action(trial))return State();
    if(!m_impl->Unchanged()){
        m_impl->Fail(Status::Terminal,"pure material action changed captured state");
        Invalidate();return State();
    }
    m_impl->status=Status::Ready;m_impl->reason.clear();m_impl->action=true;return State();
}
NativeStoppingOwnerView NativeStoppingMaterialOwner::View(NativeStoppingOwnerLease const& lease){
    if(!Matches(lease))return {};
    if(!All(m_impl->action))return {};
    auto& p=*m_impl;NativeStoppingOwnerView v;v.ready=true;
    v.impulse=Const(p.impulse);v.mean_current=Const(p.mean);v.velocity=p.material->Velocity();
    v.initial_electron_current=p.material->CapturedCurrent();v.mass=p.material->Mass();
    v.inertia_impulse=p.material->InertiaImpulse();
    for(int c=0;c<3;++c)v.physical_support[c]=&p.material->PhysicalSupport(c);
    v.momentum=Const(p.physical_pi);v.raw_momentum=p.reaction->Momentum();v.electron_current=p.material->EndpointCurrent();
    v.stored_electron_current=p.carry?Const(p.stored_endpoint):v.electron_current;
    v.carried_current=p.carry?p.carry->CarriedCurrent():CV{};
    v.mean_residual=p.material->MeanResidual();v.ion_initial=Const(p.i0);v.ion_endpoint=Const(p.i1);
    v.midpoint_energy=&p.umid;v.nodal_temperature=&p.moments->NodalTemperature();
    v.nodal_heat=&p.heat;v.cell_heat=&p.cell_heat;v.thermal_residual=&p.thermal_residual;v.work=p.work;
    for(auto const& s:p.species){if(s.charge==0.)continue;
        for(auto const& [key,points]:s.points){auto const& old=s.old.at(key).GetStructOfArrays();
            v.particles.push_back({s.name,key.first,key.second,static_cast<amrex::Long>(points.size()),
                old.GetIdCPUData().data(),points.data()});
        }
    }
    return v;
}
bool NativeStoppingMaterialOwner::PublishOnce(NativeStoppingOwnerLease const& lease){
    auto& p=*m_impl;
    if(!All(p.action&&!p.published&&m_token->generation<std::numeric_limits<std::uint64_t>::max()))return false;
    if(!Matches(lease))return false;
    // Revoke borrowed point/material views before changing a physical particle.
    m_token->active=false;++m_token->generation;p.captured=false;p.action=false;
    p.material->Invalidate();p.PublishMomenta();p.published=true;return true;
}
void NativeStoppingMaterialOwner::Impl::PublishMomenta(){
    for(auto& s:species){if(s.charge==0.)continue;
        for(WarpXParIter it(*s.pc,0);it.isValid();++it){
            auto const* point=s.points.at({it.index(),it.LocalTileIndex()}).data();
            auto* x=it.GetAttribs(PIdx::ux).data();auto* y=it.GetAttribs(PIdx::uy).data();auto* z=it.GetAttribs(PIdx::uz).data();
            amrex::ParallelFor(it.numParticles(),[=] AMREX_GPU_DEVICE(amrex::Long n){
                x[n]=point[n].next[0];y[n]=point[n].next[1];z[n]=point[n].next[2];
            });
        }
    }
}
bool NativeStoppingMaterialOwner::PublishedPopulationMatches(){
    if(!All(m_impl->published))return false;
    return m_impl->Unchanged(true);
}
bool NativeStoppingMaterialOwner::Impl::Capture(){
#if !defined(WARPX_DIM_RZ)
    return Fail(Status::Unsupported,"material owner action requires RZ m=0");
#else
    clock=sim.gett_new(0);step=sim.getistep(0);next_id=PC::ParticleType::the_next_id;
    thermal_gamma=sim.get_pointer_HybridPICModel()->m_gamma;
    names=sim.GetPartContainer().GetSpeciesNames();host_rng=Random();
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    device_rng.resize(ThermalRandomDeviceBytes());
    amrex::Gpu::dtod_memcpy(device_rng.data(),amrex::getRandState(),device_rng.size());
#endif
    bool valid=true,found=false;amrex::Long counts[4]={0,0,0,0};
    for(auto const& name:names){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        valid=valid&&dynamic_cast<PhysicalParticleContainer*>(&pc)&&
            !dynamic_cast<SubcycledParticleContainer*>(&pc)&&pc.finestLevel()==0&&
            !pc.DoFieldIonization()&&pc.do_not_deposit==(pc.getCharge()==0.)&&
            std::isfinite(pc.getCharge())&&pc.getCharge()>=0.&&
            std::isfinite(pc.getMass())&&pc.getMass()>0.;
        species.emplace_back();auto& s=species.back();
        s.name=name;s.pc=&pc;s.mass=pc.getMass();s.charge=pc.getCharge();
        s.fast=name==options.fast_species;found=found||s.fast;valid=valid&&!(s.fast&&s.charge==0.);
        s.real_names=pc.GetRealSoANames();s.int_names=pc.GetIntSoANames();
        for(auto const& [key,tile]:pc.GetParticles(0)){
            valid=valid&&tile.numNeighborParticles()==0;
            auto& old=s.old[key];old.define(tile.NumRealComps()-PIdx::nattribs,
                tile.NumIntComps()-IntIdx::nattribs,nullptr,nullptr,pc.arena());
            old.GetStructOfArrays()=tile.GetStructOfArrays();
            counts[0]+=tile.numParticles();
            if(s.charge==0.){counts[2]+=tile.numParticles();continue;}
            counts[1]+=tile.numParticles();++counts[3];
            s.points[key].resize(tile.numParticles());s.impulses[key].resize(tile.numParticles());
            s.footprint[key].resize(tile.numParticles());
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(counts,4);
    census.particles=counts[0];census.charged=counts[1];census.neutral=counts[2];census.charged_tiles=counts[3];
    if(!All(valid&&found&&counts[1]>0))return Fail(Status::Unsupported,"unsupported real species role or empty charged population");
    binding=NativeAcceptedStoppingContext::Bind(sim.m_fields,false,source_time,epoch);
    using ablastr::fields::Direction;
    valid=binding.raw_charge&&binding.raw_charge->nComp()>=1;
    for(auto const* name:{"hybrid_electron_energy_fp","hybrid_electron_temperature_fp"})
        valid=valid&&sim.m_fields.has(name,0);
    for(int c=0;c<3;++c){valid=valid&&binding.electron_current[c]&&
        sim.m_fields.has("Bfield_fp",Direction{c},0)&&sim.m_fields.has("Efield_fp",Direction{c},0);}
    if(!All(valid))return Fail(Status::Unsupported,"retained material fields unavailable");
    material=std::make_unique<NativeStoppingMaterialSupport>(sim,material_options);
    bool const prepared=carry?material->PrepareCarried(binding,*carry):material->Prepare(binding);census.material_failure=material->Failure();
    if(!prepared&&census.material_failure!=StoppingMaterialFailure::EmptyCurrent)
        return Fail(Status::Invalid,"strict material scope/binding/density declined");
    Protect("rho_fp");Protect("hybrid_electron_energy_fp");Protect("hybrid_electron_temperature_fp");
    for(int c=0;c<3;++c){
        Protect(NativeAcceptedStoppingContext::AcceptedCurrentName,c);Protect("Bfield_fp",c);Protect("Efield_fp",c);
        for(auto const* name:{"hybrid_A_fp","hybrid_E_long_fp","diagnostic_D_endpoint_fp",warpx::darwin::PECWallCurrentName})
            if(sim.m_fields.has(name,Direction{c},0))Protect(name,c);
    }
    // Diagnose the actual P/V/fixed-row inventory before the strict primitive
    // declines. This is not a carry certificate or a private current reset.
    amrex::MultiFab rho(binding.raw_charge->boxArray(),distribution,1,1);rho.setVal(0.);
    amrex::MultiFab::Copy(rho,*binding.raw_charge,0,0,1,0);
    warpx::darwin::NativeInertiaSupportOptions so;
    so.coefficient=warpx::darwin::InertiaEdgePolicy::NativeEdgeCandidate;
    so.recovery=warpx::darwin::InertiaRecoveryMask::None;
    so.components=warpx::darwin::InertiaRecoveryComponents::All;
    so.charge_floor=PhysConst::q_e*material_options.number_density_floor;
    so.reference_charge_density=PhysConst::q_e*material_options.reference_number_density;
    using BC=warpx::darwin::InitialRateBoundary;
    so.lower={BC::Axis,BC::Periodic};so.upper={BC::PEC,BC::Periodic};
    warpx::darwin::NativeInertiaSupport support(geometry,cells,distribution,so);
    if(!support.FreezeEdges(rho,rho))return Fail(Status::Invalid,"invalid raw support density");
    for(int c=0;c<3;++c){
        auto const& mask=support.PhysicalSupport(c);auto owner=binding.electron_current[c]->OwnerMask(geometry.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
            amrex::ReduceOpSum,amrex::ReduceOpMax,amrex::ReduceOpMax,
            amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpMax> op;
        amrex::ReduceData<amrex::Long,amrex::Long,amrex::Long,amrex::Long,R,R,amrex::Long,amrex::Long,R> data(op);
        using T=typename decltype(data)::Type;int const wall=geometry.Domain().bigEnd(0)+1;
        for(amrex::MFIter it(mask);it.isValid();++it){auto p=mask.const_array(it);auto own=owner->const_array(it);
            auto j=binding.electron_current[c]->const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int k,int l)->T{
                bool const empty=own(i,k,l)&&!p(i,k,l),fixed=c!=0&&i==wall,axis=c==1&&i==0;
                auto const free=amrex::Long(empty&&!fixed&&!axis),wallrow=amrex::Long(empty&&fixed);
                auto const axisrow=amrex::Long(empty&&axis);
                return{free,wallrow,free*(j(i,k,l)!=0.),wallrow*(j(i,k,l)!=0.),
                    free?std::abs(j(i,k,l)):0.,wallrow?std::abs(j(i,k,l)):0.,
                    axisrow,axisrow*(j(i,k,l)!=0.),axisrow?std::abs(j(i,k,l)):0.};
            });
        }
        auto t=data.value();amrex::Long n[6]={amrex::get<0>(t),amrex::get<1>(t),amrex::get<2>(t),amrex::get<3>(t),amrex::get<6>(t),amrex::get<7>(t)};
        R mx[3]={amrex::get<4>(t),amrex::get<5>(t),amrex::get<8>(t)};
        amrex::ParallelDescriptor::ReduceLongSum(n,6);amrex::ParallelDescriptor::ReduceRealMax(mx,3);
        census.empty_free_rows+=n[0];census.empty_fixed_rows+=n[1];
        census.nonzero_free_rows+=n[2];census.nonzero_fixed_rows+=n[3];
        census.free_current_max=std::max(census.free_current_max,mx[0]);
        census.fixed_current_max=std::max(census.fixed_current_max,mx[1]);
        census.empty_axis_rows+=n[4];census.nonzero_axis_rows+=n[5];
        census.axis_current_max=std::max(census.axis_current_max,mx[2]);
    }
    if(!prepared)return Fail(Status::Invalid,"strict material preparation declined; stored current is unchanged");
    material_lease=material->Lease();
    if(!material->BeginFootprint(material_lease))return Fail(Status::Invalid,"material footprint begin");
    amrex::Long visited_count=0;
    for(auto& s:species){if(s.charge==0.)continue;std::set<Key> visited;
        for(WarpXParIter it(*s.pc,0);it.isValid();++it){
            auto key=Key(it.index(),it.LocalTileIndex());auto old=s.old.find(key);
            if(old==s.old.end()||!visited.insert(key).second||old->second.numParticles()!=it.numParticles()){
                valid=false;continue;
            }
            auto const& a=old->second.GetStructOfArrays();GetParticlePosition<PIdx> position;
            position.m_x=a.GetRealData(PIdx::r).data();position.m_theta=a.GetRealData(PIdx::theta).data();
            position.m_z=a.GetRealData(PIdx::z).data();auto weight=a.GetRealData(PIdx::w).data();
            auto out=s.footprint.at(key).data();R const charge=s.charge;
            amrex::ParallelFor(it.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                StoppingMaterialPoint point;position(n,point.position[0],point.position[1],point.position[2]);
                point.charge=charge;point.weight=weight[n];out[n]=point;
            });
            material->AddFootprintTile(it,it.tilebox(),out,it.numParticles());visited_count+=it.numParticles();
        }
        for(auto const& [key,tile]:s.old)valid=valid&&(tile.numParticles()==0||visited.count(key)==1);
    }
    amrex::ParallelDescriptor::ReduceLongSum(visited_count);
    bool const footprint=material->FinishFootprint(material_lease);census.footprint=material->Footprint();
    census.footprint_points=visited_count;
    if(!All(valid&&footprint&&visited_count==census.charged&&census.footprint.counts[0]==census.charged))
        return Fail(Status::Invalid,"actual charged population footprint incomplete");
    auto const& energy=*sim.m_fields.get("hybrid_electron_energy_fp",0);
    Clone(u0,energy);bool const finite_u=u0.is_finite();R const min_u=u0.min(0);
    if(!All(finite_u&&min_u>0.&&u0.nComp()==1))return Fail(Status::Invalid,"source energy must be positive");
    for(auto* f:{&umid,&uend,&du,&cell_heat,&thermal_residual}){
        f->define(u0.boxArray(),distribution,1,u0.nGrowVect());f->setVal(0.);
    }
    auto const& temperature=*sim.m_fields.get("hybrid_electron_temperature_fp",0);
    heat.define(temperature.boxArray(),distribution,1,temperature.nGrowVect());heat.setVal(0.);
    auto current=sim.m_fields.get_alldirs(warpx::fields::FieldType::current_fp,0);
    auto electric=sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp,0);
    auto native_aux=sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_aux,0);
    for(int c=0;c<3;++c){
        auto grow=amrex::max(material_options.ghosts,amrex::max(current[c]->nGrowVect(),electric[c]->nGrowVect()));
        for(auto* f:{&impulse[c],&mean[c],&zero[c],&i0[c],&i1[c],&physical_pi[c]}){
            f->define(binding.electron_current[c]->boxArray(),distribution,1,grow);f->setVal(0.);
        }
        Clone(aux[c],*native_aux[c]);aux[c].setVal(0.);
        if(carry){stored_endpoint[c].define(binding.electron_current[c]->boxArray(),distribution,1,0);stored_endpoint[c].setVal(0.);}
    }
    if(!warpx::particles::DepositNativeInstantaneousIonCurrent(sim,
        warpx::particles::InstantaneousIonState::Current,Mutable(i0)))return Fail(Status::Invalid,"initial native ion current");
    // The source support is bound to this real population's density, using the
    // old count/positive-operand bound, never a measured discrepancy allowance.
    amrex::MultiFab deposited(binding.raw_charge->boxArray(),distribution,binding.raw_charge->nComp(),binding.raw_charge->nGrowVect());
    deposited.setVal(0.);sim.GetPartContainer().DepositCharge({&deposited},0.);
    sim.SyncRho({&deposited},{},{});sim.ApplyRhofieldBoundary(0,&deposited,PatchType::fine);
    amrex::MultiFab difference(rho.boxArray(),distribution,1,0);
    amrex::MultiFab::LinComb(difference,1.,deposited,0,-1.,rho,0,0,1,0);
    amrex::MultiFab one(cells,distribution,1,0);one.setVal(1.);
    R const reference_rho=Integral(rho,geometry)/Integral(one,geometry);
    R const operations=4096.+4096.*R(census.charged),u=std::numeric_limits<R>::epsilon();
    census.density_difference=difference.norm0();census.density_bound=4.*operations*u/(1.-operations*u)*reference_rho;
    if(!All(std::isfinite(reference_rho)&&reference_rho>so.charge_floor&&operations*u<.01&&
        census.density_difference<=census.density_bound))return Fail(Status::Invalid,"real population/raw density mismatch");
    ThermalMomentOptions mo;mo.number_density_floor=material_options.number_density_floor;
    mo.active_density_floor=mo.number_density_floor;mo.gamma=thermal_gamma;
    mo.nodal_ghosts=temperature.nGrow();mo.verboncoeur_axis_correction=sim.UseVerboncoeurAxisCorrection();
    mo.boundary[0]={MomentBoundary::Axis,MomentBoundary::PEC};
    mo.current_boundary[0]={MomentBoundary::Axis,MomentBoundary::PMC};
    mo.boundary[1]={MomentBoundary::Periodic,MomentBoundary::Periodic};
    moments=std::make_unique<KineticThermalMoments>(geometry,cells,distribution,mo);
    reaction=std::make_unique<NativeStoppingMaterialReaction>(*material,material_lease,
        geometry,cells,distribution,material_options);sums.resize(20);
    captured=true;if(!Unchanged()){captured=false;return Fail(Status::Terminal,"capture modified input state");}
    status=Status::Ready;reason.clear();return true;
#endif
}
bool NativeStoppingMaterialOwner::Impl::Unchanged(bool after_publication){
    bool good=sim.gett_new(0)==clock&&sim.getistep(0)==step&&
        sim.boxArray(0)==cells&&sim.DistributionMap(0)==distribution&&
        sim.GetPartContainer().GetSpeciesNames()==names&&PC::ParticleType::the_next_id==next_id&&
        Random()==host_rng;
    auto const* model=sim.get_pointer_HybridPICModel();
    good=good&&model&&model->m_n_floor==material_options.number_density_floor&&
        model->m_n0_ref==material_options.reference_number_density&&
        model->m_electron_inertia_mass==material_options.model_electron_mass&&model->m_gamma==thermal_gamma;
    for(auto const& f:protected_fields){
        bool const exists=f->component<0?sim.m_fields.has(f->name,0):
            sim.m_fields.has(f->name,ablastr::fields::Direction{f->component},0);
        auto const* now=exists?(f->component<0?sim.m_fields.get(f->name,0):
            sim.m_fields.get(f->name,ablastr::fields::Direction{f->component},0)):nullptr;
        good=good&&now==f->original;
        if(now)good=good&&now->boxArray()==f->copy.boxArray()&&
            now->DistributionMap()==f->copy.DistributionMap()&&now->nComp()==f->copy.nComp()&&
            now->nGrowVect()==f->copy.nGrowVect();
    }
    // Complete structural checks before reading any live particle data pointer.
    if(!All(good))return false;
    for(auto const& s:species){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(s.name);
        good=good&&(&pc==s.pc)&&pc.finestLevel()==0&&pc.GetRealSoANames()==s.real_names&&
            pc.GetIntSoANames()==s.int_names&&pc.getMass()==s.mass&&pc.getCharge()==s.charge&&
            !pc.DoFieldIonization()&&pc.do_not_deposit==(s.charge==0.)&&pc.GetParticles(0).size()==s.old.size();
        for(auto const& [key,old]:s.old){auto it=pc.GetParticles(0).find(key);
            if(it==pc.GetParticles(0).end()){good=false;continue;}
            auto const& now=it->second;
            good=good&&now.numParticles()==old.numParticles()&&now.numNeighborParticles()==0&&
                now.NumRealComps()==old.NumRealComps()&&now.NumIntComps()==old.NumIntComps();
            auto const& a=now.GetStructOfArrays();auto const& b=old.GetStructOfArrays();
            good=good&&a.GetIdCPUData().size()==b.GetIdCPUData().size();
            if(now.NumRealComps()==old.NumRealComps())for(int c=0;c<now.NumRealComps();++c)
                good=good&&a.GetRealData(c).size()==b.GetRealData(c).size();
            if(now.NumIntComps()==old.NumIntComps())for(int c=0;c<now.NumIntComps();++c)
                good=good&&a.GetIntData(c).size()==b.GetIntData(c).size();
        }
    }
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    good=good&&device_rng.size()==ThermalRandomDeviceBytes();
#endif
    if(!All(good))return false;
    amrex::Gpu::DeviceScalar<int> invalid(0);auto* bad=invalid.dataPtr();
    // Source publication separately checks exact new fields and held histories.
    // Population/RNG protection below still covers every byte.
    if(!after_publication)for(auto const& f:protected_fields)for(amrex::MFIter it(*f->original);it.isValid();++it)
        Bytes((*f->original)[it].dataPtr(),f->copy[it].dataPtr(),f->copy[it].size()*sizeof(R),bad);
    for(auto const& s:species)for(auto const& [key,old]:s.old){
        auto const& now=s.pc->GetParticles(0).at(key);
        auto const& a=now.GetStructOfArrays();auto const& b=old.GetStructOfArrays();
        Bytes(a.GetIdCPUData().data(),b.GetIdCPUData().data(),a.GetIdCPUData().size()*sizeof(std::uint64_t),bad);
        for(int c=0;c<now.NumRealComps();++c){
            int const axis=c==PIdx::ux?0:(c==PIdx::uy?1:(c==PIdx::uz?2:-1));
            if(after_publication&&s.charge!=0.&&axis>=0){
                auto const* values=a.GetRealData(c).data();auto const* points=s.points.at(key).data();
                amrex::For(now.numParticles(),[=] AMREX_GPU_DEVICE(amrex::Long n){
                    P const expected=static_cast<P>(points[n].next[axis]);
                    auto const* x=reinterpret_cast<unsigned char const*>(values+n);
                    auto const* y=reinterpret_cast<unsigned char const*>(&expected);
                    for(std::size_t j=0;j<sizeof(P);++j)if(x[j]!=y[j])amrex::Gpu::Atomic::Exch(bad,1);
                });
            }else Bytes(a.GetRealData(c).data(),b.GetRealData(c).data(),a.GetRealData(c).size()*sizeof(P),bad);
        }
        for(int c=0;c<now.NumIntComps();++c)
            Bytes(a.GetIntData(c).data(),b.GetIntData(c).data(),a.GetIntData(c).size()*sizeof(int),bad);
    }
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    Bytes(device_rng.data(),amrex::getRandState(),device_rng.size(),bad);
#endif
    return All(invalid.dataValue()==0);
}
void NativeStoppingMaterialOwner::Impl::ElectricImages(){
#if defined(WARPX_DIM_RZ)
    Sync(impulse,geometry);auto v=Mutable(impulse);amrex::Vector<amrex::IntVect> ratios;
    PEC::ApplyPECtoEfield(v,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PEC,impulse[0].nGrowVect(),geometry,0,PatchType::fine,ratios);
    PEC::ApplyPECtoBfield(v,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PMC,impulse[0].nGrowVect(),geometry,0,PatchType::fine,ratios);
    sim.ApplyFieldBoundaryOnAxis(v[0],v[1],v[2],0);Sync(impulse,geometry);
#endif
}
void NativeStoppingMaterialOwner::Impl::Deposit(WarpXParIter const& it,Point const* points,amrex::Long count){
#if defined(WARPX_DIM_RZ)
    auto box=it.tilebox();box.grow(sim.get_ng_depos_J());auto const origin=WarpX::LowerCorner(box,0,0.);
    auto const lower=amrex::lbound(box);auto const inverse=WarpX::InvCellSize(0);
    auto const jx=i1[0].array(it),jy=i1[1].array(it),jz=i1[2].array(it);
    auto const h=heat.array(it);auto const plo=geometry.ProbLoArray();auto const inv=geometry.InvCellSizeArray();
    auto const volume=MakeQdsmcVolumeElement(geometry,heat.ixType());
    amrex::For(count,[=] AMREX_GPU_DEVICE(amrex::Long n){
        auto const p=points[n];auto const u=p.next;R const gamma=Gamma(u);
        R const xp=p.position[0],yp=p.position[1],rr=std::sqrt(xp*xp+yp*yp);
        R const vx=u[0]/gamma,vy=u[1]/gamma,vz=u[2]/gamma;
        V x{(rr-origin.x)*inverse.x,0.,(p.position[2]-origin.z)*inverse.z};
        V v{(xp*vx+yp*vy)/rr*inverse.x,0.,vz*inverse.z};R const vt=(-yp*vx+xp*vy)/rr;
        int const first_r=int(x[0])-1,first_z=int(x[2])-1;
        for(int iz=first_z;iz<=first_z+3;++iz)for(int ir=first_r;ir<=first_r+3;++ir){
            V value{},unused{};
            warpx::particles::EsirkepovCurrentRate<3,2>(x,v,V{},amrex::GpuArray<int,3>{ir,0,iz},vt,0.,value,unused);
            int const ii=ir+lower.x,jj=iz+lower.y;
            R const a=p.charge*p.weight*inverse.y*inverse.z;
            R const b=p.charge*p.weight*inverse.x*inverse.y*inverse.z;
            R const c=p.charge*p.weight*inverse.x*inverse.y;
            if(ir<first_r+3)amrex::HostDevice::Atomic::Add(&jx(ii,jj,0),a*value[0]);
            amrex::HostDevice::Atomic::Add(&jy(ii,jj,0),b*value[1]);
            if(iz<first_z+3)amrex::HostDevice::Atomic::Add(&jz(ii,jj,0),c*value[2]);
        }
        int ii=0,jj=0,kk=0;R weights[3][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(p.position[0],p.position[1],p.position[2],plo,inv,ii,jj,kk,weights);
        for(int dz=0;dz<2;++dz)for(int dr=0;dr<2;++dr)
            amrex::HostDevice::Atomic::Add(&h(ii+dr,jj+dz,0),
                p.heat*weights[0][dr]*weights[1][dz]/volume(ii+dr,jj+dz,0));
    });
#else
    amrex::ignore_unused(it,points,count);
#endif
}
bool NativeStoppingMaterialOwner::Impl::Action(NativeStoppingOwnerTrial const& trial){
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(trial);return Fail(Status::Unsupported,"RZ action unavailable");
#else
    bool layout=Layout(trial.energy_increment,u0);
    for(int c=0;c<3;++c)layout=layout&&Layout(trial.impulse[c],impulse[c])&&Layout(trial.mean_current[c],mean[c]);
    if(!All(layout))return Fail(Status::Invalid,"trial layout differs from retained owner");
    bool finite=trial.energy_increment->is_finite(0,1,0);
    for(int c=0;c<3;++c){bool const a=trial.impulse[c]->is_finite(0,1,0);
        bool const b=trial.mean_current[c]->is_finite(0,1,0);finite=finite&&a&&b;}
    if(!All(finite))return Fail(Status::Invalid,"nonfinite trial");
    // Check the supplied fixed-wall/axis impulse before ElectricImages can
    // impose a homogeneous boundary value. Never hide boundary work by paint.
    if(carry&&!carry->ValidateTrial(trial.impulse))return Fail(Status::Invalid,"carried fixed-current rows require exact homogeneous impulse");
    // Copy all valid trial operands before overwriting any private action field.
    // Caller-owned views must not alias this owner's published scratch.
    bool aliases=false;
    auto const v=material->ActionReady()?material->Velocity():CV{};
    auto const e=material->ActionReady()?material->EndpointCurrent():CV{};
    auto const r=material->ActionReady()?material->MeanResidual():CV{};
    auto const raw=Const(reaction->m_momentum);
    for(int c=0;c<3;++c)for(auto const* a:{trial.impulse[c],trial.mean_current[c]}){
        for(int d=0;d<3;++d){
            for(auto const* f:{&impulse[d],&mean[d],&zero[d],&aux[d],&i0[d],&i1[d],&physical_pi[d]})
                aliases=SharesStorage(*a,*f)||aliases;
            if(carry)aliases=SharesStorage(*a,stored_endpoint[d])||aliases;
            for(auto const* f:{v[d],e[d],r[d],raw[d]})if(f)
                aliases=SharesStorage(*a,*f)||aliases;
        }
    }
    for(auto const* f:{&u0,&umid,&uend,&du,&heat,&cell_heat,&thermal_residual})
        aliases=SharesStorage(*trial.energy_increment,*f)||aliases;
    if(!All(!aliases))return Fail(Status::Invalid,"trial aliases private owner output");
    for(int c=0;c<3;++c){impulse[c].setVal(0.);mean[c].setVal(0.);i1[c].setVal(0.);
        amrex::MultiFab::Copy(impulse[c],*trial.impulse[c],0,0,1,0);
        amrex::MultiFab::Copy(mean[c],*trial.mean_current[c],0,0,1,0);}
    du.setVal(0.);amrex::MultiFab::Copy(du,*trial.energy_increment,0,0,1,0);
    // Same admissible regular subspace as the old native map.
    for(amrex::MFIter it(mean[1]);it.isValid();++it){auto a=mean[1].array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){if(i==0)a(i,j,k)=0.;});}
    Sync(mean,geometry);ElectricImages();
    amrex::MultiFab::LinComb(umid,1.,u0,0,.5,du,0,0,1,0);
    amrex::MultiFab::LinComb(uend,1.,u0,0,1.,du,0,0,1,0);
    bool const finite_mid=umid.is_finite(0,1,0),finite_end=uend.is_finite(0,1,0);
    R const min_mid=umid.min(0),min_end=uend.min(0);
    if(!All(finite_mid&&finite_end&&min_mid>0.&&min_end>0.))return Fail(Status::Invalid,"midpoint/endpoint energy is not positive");
    KineticThermalStateView thermal{*binding.raw_charge};thermal.energy=&umid;
    if(!moments->Evaluate(thermal))return Fail(Status::Invalid,"live midpoint thermal state");
    ++work.actions;++work.material_actions;
    if(!material->Apply(material_lease,Const(zero),Const(zero),Const(mean)))return Fail(Status::Invalid,"material velocity action");
    auto electric=Mutable(impulse),electric_aux=Mutable(aux);
    sim.InterpolateLevelZeroFieldToAux(electric_aux,electric);
    heat.setVal(0.);amrex::Gpu::fillAsync(sums.begin(),sums.end(),[] AMREX_GPU_DEVICE(R& x,amrex::Long){x=0.;});
    amrex::Gpu::DeviceScalar<int> invalid(0);int* bad=invalid.dataPtr();R* totals=sums.data();
    auto native_b=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    CV magnetic{native_b[0],native_b[1],native_b[2]};
    auto const inverse=geometry.InvCellSizeArray(),plo=geometry.ProbLoArray();
    R const dt=interval,clog=options.coulomb_log,cap=options.proper_speed_cap;
    bool const stable_increment=stable_drag_increment;
    R const floor=PhysConst::q_e*material_options.number_density_floor;
    for(auto& s:species){if(s.charge==0.)continue;
        for(WarpXParIter it(*s.pc,0);it.isValid();++it){auto key=Key(it.index(),it.LocalTileIndex());
            auto const& old=s.old.at(key).GetStructOfArrays();GetParticlePosition<PIdx> position;
            position.m_x=old.GetRealData(PIdx::r).data();position.m_theta=old.GetRealData(PIdx::theta).data();position.m_z=old.GetRealData(PIdx::z).data();
            auto ux=old.GetRealData(PIdx::ux).data(),uy=old.GetRealData(PIdx::uy).data(),uz=old.GetRealData(PIdx::uz).data();
            auto weight=old.GetRealData(PIdx::w).data(),angle=old.GetRealData(PIdx::theta).data();
            auto gather=material->GatherView(it,magnetic),egather=gather;
            for(int c=0;c<3;++c){egather.velocity[c]=aux[c].const_array(it);egather.velocity_type[c]=aux[c].ixType();}
            auto rho=binding.raw_charge->const_array(it);auto t=moments->NodalTemperature().const_array(it);
            auto points=s.points.at(key).data();auto tuples=s.impulses.at(key).data();
            R const m=s.mass,q=s.charge;bool const fast=s.fast;
            amrex::For(it.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                Point p;p.mass=m;p.charge=q;p.weight=weight[n];p.old={ux[n],uy[n],uz[n]};
                position(n,p.position[0],p.position[1],p.position[2]);
                auto const velocity=gather(p.position[0],p.position[1],p.position[2]);
                auto const force=egather(p.position[0],p.position[1],p.position[2]);
                bool ok=velocity.valid&&force.valid&&p.weight>=0.&&std::isfinite(p.weight)&&
                    std::sqrt(p.position[0]*p.position[0]+p.position[1]*p.position[1])>0.;
                p.ve=velocity.cartesian;p.psi=force.cartesian;
                R const density=ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],rho,inverse,plo);
                R const te=amrex::max(ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],t,inverse,plo),1.e-3*PhysConst::q_e/PhysConst::kb);
                ok=ok&&density>floor&&std::isfinite(te)&&Norm(p.old)<=cap;
                for(int c=0;c<3;++c)p.minus[c]=p.old[c]+q*p.psi[c]/(2.*m);
                p.drag=p.minus;
                if(fast&&dt>0.){
                    p.rate=warpx::particles::NativeStoppingRate(density*(1./PhysConst::q_e),te*PhysConst::kb,q,m,clog);
                    R const theta=angle[n],cm=std::cos(-theta),sm=std::sin(-theta);
                    V const local{p.minus[0]*cm-p.minus[1]*sm,p.minus[0]*sm+p.minus[1]*cm,p.minus[2]};
                    if(stable_increment){
                        auto const delta=warpx::particles::NativeStoppingProperIncrement(
                            local,velocity.collision_frame,p.rate,dt);
                        R const co=std::cos(theta),si=std::sin(theta);
                        V const cartesian{delta[0]*co-delta[1]*si,
                            delta[0]*si+delta[1]*co,delta[2]};
                        // One actual represented drag state. The unchanged tuple,
                        // current and all particle work below consume this state.
                        for(int c=0;c<3;++c)p.drag[c]=p.minus[c]+cartesian[c];
                    }else{
                        auto const kick=warpx::particles::NativeStoppingKick(local,velocity.collision_frame,p.rate,dt,p.weight,m).proper;
                        R const co=std::cos(theta),si=std::sin(theta);
                        p.drag={kick[0]*co-kick[1]*si,kick[0]*si+kick[1]*co,kick[2]};
                    }
                }
                StoppingIonImpulse tuple;tuple.position=p.position;tuple.weight=p.weight;
                for(int c=0;c<3;++c){p.next[c]=p.drag[c]+q*p.psi[c]/(2.*m);tuple.impulse[c]=m*(p.drag[c]-p.minus[c]);}
                ok=ok&&Norm(p.minus)<=cap&&Norm(p.drag)<=cap&&Norm(p.next)<=cap&&std::isfinite(Norm(p.next));
                R const dki=p.weight*DeltaK(p.old,p.next,m),drag=p.weight*DeltaK(p.minus,p.drag,m);
                R const bulk=p.weight*Dot(tuple.impulse,p.ve);
                R const ew=p.weight*(DeltaK(p.old,p.minus,m)+DeltaK(p.drag,p.next,m));
                V vm{};R const g0=Gamma(p.old),g1=Gamma(p.next);
                for(int c=0;c<3;++c)vm[c]=.5*(p.old[c]/g0+p.next[c]/g1);
                R const ep=p.weight*q*Dot(p.psi,vm);p.heat=-(drag-bulk);
                ok=ok&&std::isfinite(p.heat)&&p.heat>=0.;
                if(!ok){amrex::Gpu::Atomic::Exch(bad,1);return;}
                points[n]=p;tuples[n]=tuple;
                R values[10]={p.weight*m*Dot(p.old,p.old)/(g0+1.),dki,drag,bulk,ew,ep,p.heat,ew-ep,
                    p.weight*std::abs(q)*Norm(p.psi)*cap*cap*cap/(PhysConst::c*PhysConst::c),
                    std::abs(dki)+std::abs(drag)+std::abs(bulk)+std::abs(ew)};
                for(int c=0;c<10;++c)amrex::HostDevice::Atomic::Add(totals+c,values[c]);
            });
        }
    }
    // One collective scalar check for all tuples; no per-tile host read/sync.
    ++work.tuple_validation_preflights;
    if(!All(invalid.dataValue()==0))return Fail(Status::Invalid,"native source tuple outside fixed physical scope");
    if(!reaction->Begin())return Fail(Status::Invalid,"material reaction begin");
    for(auto& s:species){if(s.charge==0.)continue;
        for(WarpXParIter it(*s.pc,0);it.isValid();++it){auto key=Key(it.index(),it.LocalTileIndex());
            reaction->DepositTile(it,s.impulses.at(key).data(),it.numParticles());
            Deposit(it,s.points.at(key).data(),it.numParticles());
        }
    }
    if(!reaction->Finish())return Fail(Status::Invalid,"material reaction finish");
    sim.ApplyInverseVolumeScalingToCurrentDensity(&i1[0],&i1[1],&i1[2],0);
    for(auto& f:i1)ablastr::utils::communication::SumBoundary(f,0,1,f.nGrowVect(),f.nGrowVect(),false,geometry.periodicity());
    sim.ApplyJfieldBoundary(0,&i1[0],&i1[1],&i1[2],PatchType::fine);Sync(i1,geometry);
    ablastr::utils::communication::SumBoundary(heat,0,1,heat.nGrowVect(),heat.nGrowVect(),false,geometry.periodicity());
    heat.OverrideSync(geometry.periodicity());heat.FillBoundary(geometry.periodicity());
    auto raw_pi=reaction->Momentum();
    for(int c=0;c<3;++c){physical_pi[c].setVal(0.);
        amrex::MultiFab::Copy(physical_pi[c],*raw_pi[c],0,0,1,0);}
    // Native regular-velocity covector restriction, separate from PEC carry.
    for(amrex::MFIter it(physical_pi[1]);it.isValid();++it){auto p=physical_pi[1].array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){if(i==0)p(i,j,k)=0.;});}
    amrex::MultiFab axis_work(physical_pi[1].boxArray(),distribution,1,0);
    auto velocity=material->Velocity();
    for(amrex::MFIter it(axis_work);it.isValid();++it){auto out=axis_work.array(it);
        auto pi=raw_pi[1]->const_array(it),v=velocity[1]->const_array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=i==0?pi(i,j,k)*v(i,j,k):0.;});}
    work.axis_reaction_work=Integral(axis_work,geometry);
    if(!All(work.axis_reaction_work==0.))return Fail(Status::Invalid,"nonzero axis reaction work");
    ++work.material_actions;
    if(!material->Apply(material_lease,Const(impulse),Const(physical_pi),Const(mean)))
        return Fail(Status::Invalid,"material constitutive action or empty reaction row");
    // Independently verify the actual deposited ion current, not just its
    // earlier footprint, on every V row; no inverse-zero mass or row reset.
    amrex::Gpu::DeviceScalar<int> empty_bad(0);auto* eb=empty_bad.dataPtr();
    for(int c=0;c<3;++c)for(amrex::MFIter it(i1[c]);it.isValid();++it){
        auto mask=material->PhysicalSupport(c).const_array(it);auto a=i1[c].const_array(it),b=i0[c].const_array(it);
        amrex::For(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            if(!mask(i,j,k)&&(a(i,j,k)!=0.||b(i,j,k)!=0.))amrex::HostDevice::Atomic::Add(eb,1);
        });
    }
    if(!All(empty_bad.dataValue()==0))return Fail(Status::Invalid,"actual ion current outside physical support");
    moments->RestrictNodalScalar(heat,0,cell_heat);
    amrex::MultiFab::LinComb(thermal_residual,1.,du,0,-1.,cell_heat,0,0,1,0);
    R values[20];amrex::Gpu::copy(amrex::Gpu::deviceToHost,sums.begin(),sums.end(),values);
    amrex::ParallelDescriptor::ReduceRealSum(values,20);
    work.material=material->Work();work.reaction=reaction->Balance();
    work.ion_before=values[0];work.ion_increment=values[1];work.drag=values[2];work.bulk=values[3];
    work.electric=values[4];work.endpoint_particle_work=values[5];work.heat_nominal=values[6];
    work.convention=values[7];work.convention_bound=values[8];work.absolute_particle_work=values[9];
    work.heat_delivered=Integral(cell_heat,geometry);work.heat_transfer=work.heat_delivered-work.heat_nominal;
    work.signed_thermal_residual=Integral(thermal_residual,geometry);work.current_grid_work=0.;
    for(int c=0;c<3;++c){amrex::MultiFab field(i1[c].boxArray(),distribution,1,0);
        for(amrex::MFIter it(field);it.isValid();++it){auto f=field.array(it);auto a=i0[c].const_array(it),b=i1[c].const_array(it),p=impulse[c].const_array(it);
            amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){f(i,j,k)=.5*(a(i,j,k)+b(i,j,k))*p(i,j,k);});}
        work.current_grid_work+=Integral(field,geometry);
    }
    work.spatial_transfer=work.endpoint_particle_work-work.current_grid_work;
    if(carry){
        if(!carry->MeasureWork(Const(impulse),work.carry))return Fail(Status::Invalid,"carried current work/provenance invalid");
        auto const carried=carry->CarriedCurrent(),endpoint=material->EndpointCurrent();
        for(int c=0;c<3;++c)for(amrex::MFIter it(stored_endpoint[c]);it.isValid();++it){
            auto out=stored_endpoint[c].array(it);auto p=material->PhysicalSupport(c).const_array(it);
            auto a=endpoint[c]->const_array(it),b=carried[c]->const_array(it);
            amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=p(i,j,k)?a(i,j,k):b(i,j,k);});
        }
    }
    bool const finite_heat=cell_heat.is_finite(0,1,0),finite_residual=thermal_residual.is_finite(0,1,0);
    bool good=finite_heat&&finite_residual;
    for(R x:values)good=good&&std::isfinite(x);
    return All(good)||Fail(Status::Invalid,"nonfinite private material work");
#endif
}
} // namespace warpx::thermal
