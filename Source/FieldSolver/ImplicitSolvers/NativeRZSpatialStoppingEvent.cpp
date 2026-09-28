/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativeRZSpatialStoppingEvent.H"
#include "NativeRZMidpointStoppingEvent.H"
#include "NativeCircuitFieldImpulse.H"
#include "Circuit/CircuitCoupling.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "NativeLongitudinalProducerCertificate.H"
#include "DarwinInitialRateSchur.H"
#include "DarwinABoundary.H"
#include "KineticThermalMoments.H"
#include "StoppingVelocityImages.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#if defined(WARPX_DIM_3D)
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "NonlinearSolvers/FlexibleGMRES.H"
#include "NativeInstantaneousIonCurrent.H"
#include "NativeStoppingReactionTranspose.H"
#include "ThermalRandomCheckpoint.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Collision/HybridElectronStopping/NativeStoppingMap.H"
#include "Particles/Deposition/EsirkepovCurrentRate.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Gather/GetExternalFields.H"
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
    auto const measure=MakeQdsmcVolumeElement(geom,f.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);
    using T=typename decltype(data)::Type;
    for(amrex::MFIter mfi(f);mfi.isValid();++mfi){
        auto const a=f.const_array(mfi);auto const own=owner->const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
            return {own(i,j,k)?a(i,j,k)*measure(i,j,k):0.};
        });
    }
    R result=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(result);return result;
}
void Sync (std::array<amrex::MultiFab,3>& f,amrex::Geometry const& geom) {
    for(auto& a:f){a.OverrideSync(geom.periodicity());a.FillBoundary(geom.periodicity());}
}
std::string HostRandom () { std::ostringstream s;amrex::SaveRandomState(s);return s.str(); }
struct SpatialVector { std::array<amrex::MultiFab,6> a; };
struct Point {
    V position{},old{},minus{},drag{},next{},ve{},psi{};
    R weight=0.,mass=0.,charge=0.,rate=0.,heat=0.;
};
// Frozen numerical-PC operands, separate from mutable physical Point scratch.
struct MidpointTemperaturePoint {
    V position{},proper_derivative{},velocity_derivative{};
    R weight=0.,mass=0.,charge=0.;
};
}

struct NativeRZSpatialStoppingEvent::Impl {
    struct Species {
        PC* pc=nullptr;
        std::vector<std::string> real_names,int_names;
        PC::ParticleLevel old;
        std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<Point>> points;
        std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<StoppingIonImpulse>> impulses;
        std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<MidpointTemperaturePoint>> temperature_points;
        std::map<std::pair<int,int>,amrex::Gpu::DeviceVector<StoppingIonImpulse>> temperature_impulses;
        R mass=0.,charge=0.;bool fast=false;
    };
    struct CircuitState {
        NativeStoppingCircuitBinding binding;
        std::uint64_t generation=0, evaluations=0, application=0;
        std::array<amrex::MultiFab,3> free_impulse,total0,total_delta,static_b;
        std::array<amrex::MultiFab,3> coil_b0,coil_j0,coil_b_delta,coil_j_delta,coil_a_delta,project_input;
        amrex::Gpu::DeviceVector<double> port_impulse,field_increment;
        R initial_port_current=0.,reference_current=0.;
        NativeStoppingCircuitWork work;
        NativeStoppingCircuitPublication publication;
    };
    std::unique_ptr<CircuitState> circuit;
    WarpX& sim;NativeAcceptedStoppingContext const& context;
    amrex::MultiFab const& temperature;amrex::MultiFab& energy;SpatialStoppingOptions options;
    std::vector<Species> species;std::vector<std::string> names;
    std::array<amrex::MultiFab,3> j0,j1,mean,ve,mass,impulse,i0,i1,scratch,reaction_b,wall;
    std::array<amrex::MultiFab,3> a0,a1,b0,b1,d0,d1,c1,aux,zero,plpsi,delta,e0,bcheck;
    SpatialStoppingFields destinations;
    bool background=false;
    RZStoppingBackground background_destinations;
    RZStoppingBackgroundLedger background_ledger;
    std::array<amrex::MultiFab,3> c0,w0,full_a,full_b,full_d,full_w;

    std::unique_ptr<warpx::darwin::DarwinInitialRateSchur> projection;
    std::unique_ptr<amrex::MultiFab> displacement_potential;
    std::unique_ptr<KineticThermalMoments> heat_map;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> owners;
    amrex::MultiFab rho0,t0,u0,u1,heat,projection_rhs;
    std::unique_ptr<NativeStoppingReactionTranspose> reaction;
    amrex::Gpu::DeviceVector<R> sums;
    amrex::Gpu::DeviceVector<unsigned char> device_rng;
    std::string host_rng;std::uint64_t generation=0;amrex::Long next_id=0;
    R time=0.,volume=0.,reference_rho=0.,reference_temperature=0.,current_scale=0.,psi_scale=0.;
    R arithmetic_factor=0.;int step=0;bool ready=false,committed=false,closed=false;
    char const* failure="unprepared";RZSpatialStoppingLedger ledger;
    AcceptedStoppingBinding binding;FV published{};
    bool narrow_current=false;
    bool midpoint_thermal=false,midpoint_captured=false;
    NativeStoppingThermalOptions midpoint_options;
    NativeStoppingThermalReceipt midpoint_receipt;
    std::unique_ptr<KineticThermalMoments> midpoint_moments;
    amrex::MultiFab midpoint_energy,midpoint_increment,midpoint_heat,midpoint_residual;
    WarpXSolverVec midpoint_solution;
    WarpXSolverVec midpoint_pc_base,midpoint_pc_direction,midpoint_pc_plus,midpoint_pc_minus;
    WarpXSolverVec midpoint_pc_fplus,midpoint_pc_fminus,midpoint_pc_response;
    bool midpoint_pc_frozen=false;
    bool midpoint_pc_temperature_frozen=false;
    std::uint64_t midpoint_pc_temperature_generation=0;
    amrex::Long midpoint_pc_temperature_points=0;
    R midpoint_pc_temperature_gamma=0.;
    amrex::MultiFab midpoint_pc_temperature_density,midpoint_pc_temperature_cell,midpoint_pc_temperature_node;
    std::array<amrex::MultiFab,3> midpoint_pc_temperature_current,midpoint_pc_temperature_electron,midpoint_pc_temperature_sum;
    std::unique_ptr<NativeStoppingReactionTranspose> midpoint_pc_temperature_reaction;
    amrex::Gpu::DeviceVector<int> midpoint_pc_temperature_bad;
    bool InitializeMidpointTemperaturePC();
    bool MidpointTemperatureContextValid(bool frozen) const;
    bool FreezeMidpointTemperaturePC();
    bool ApplyMidpointTemperatureColumn(amrex::MultiFab const&);
    bool FreezeMidpointPC(WarpXSolverVec const&);
    bool ApplyMidpointPC(WarpXSolverVec&,WarpXSolverVec const&);
    SpatialVector midpoint_trial,midpoint_spatial_residual;
    std::array<std::string,3> midpoint_current_names;
    R midpoint_energy_scale=0.;
    bool InitializeMidpoint();
    bool MidpointResidual(WarpXSolverVec&,WarpXSolverVec const&);
    bool SolveMidpoint();
    bool MidpointWork();
    bool DefineMidpointState(WarpXSolverVec&) const;
    Impl(WarpX& s,NativeAcceptedStoppingContext const& c,amrex::MultiFab const& t,
         amrex::MultiFab& u,SpatialStoppingOptions const& o,SpatialStoppingFields const& f):sim(s),context(c),temperature(t),energy(u),options(o),destinations(f){}
    bool Reject(char const* why){ready=false;failure=why;return false;}
    bool Scope();bool Capture();bool Unchanged(bool after);bool Evaluate(SpatialVector const&,SpatialVector&);
    bool CaptureBackground();void AssembleBackground();
    bool CircuitContextMatches() const;
    bool CircuitFinalActionMatches() const;
    bool CaptureCircuit();bool ApplyCircuit();bool CircuitWork();

    SpatialVector Vector() const;
    R VectorDot(SpatialVector const&,SpatialVector const&) const;
    R VectorMax(SpatialVector const&) const;
    bool Project(CV const&,std::array<amrex::MultiFab,3>&,bool current);
    void Curl();
    void CompleteNarrowCurrentGuards();
    void AImages(std::array<amrex::MultiFab,3>&);
    void ElectricImages(std::array<amrex::MultiFab,3>&);
    void VelocityImages(std::array<amrex::MultiFab,3>&);
    bool Solve();bool Work();bool Publish(FV const&,NativeRZSpatialStoppingEvent::InvalidationHook,bool owned_circuit=false);
    void Deposit(Species&,WarpXParIter const&,Point const*,long);
};


bool NativeRZSpatialStoppingEvent::Impl::CircuitContextMatches() const {
    if(!circuit||!circuit->binding.field)return false;
    auto const& b=circuit->binding;
    bool const phase=b.phase==NativeStoppingCircuitBinding::Phase::PreField||
        b.phase==NativeStoppingCircuitBinding::Phase::PostField;
    auto const native_phase=b.phase==NativeStoppingCircuitBinding::Phase::PreField?
        CircuitCoupler::SourceImpulsePhase::PreField:CircuitCoupler::SourceImpulsePhase::PostField;
    // Capture owns rho/mass/current and reaction operands. A same-time
    // re-Prepare is a new material epoch even if its circuit phase is unchanged.
    return context.Valid()&&(generation==0||context.Generation()==generation)&&
        phase&&std::isfinite(b.source_time)&&b.source_time==context.Binding().endpoint_time&&
        b.field->MatchesSourceContext(sim,b.source_time,native_phase)&&
        (circuit->generation==0||circuit->generation==b.field->Generation());
}

bool NativeRZSpatialStoppingEvent::Impl::CircuitFinalActionMatches() const {
    // Preparation generation alone does not distinguish residual/Jv actions.
    // CopySolution/publication must refer to the final action of this solve.
    return CircuitContextMatches() && circuit->application!=0 &&
        circuit->binding.field->Applied() &&
        circuit->application==circuit->binding.field->Application();
}

bool NativeRZSpatialStoppingEvent::Impl::CaptureCircuit() {
    auto& q=*circuit;auto& action=*q.binding.field;
    auto* subsystem=sim.get_pointer_CircuitCoupling();
    bool valid=CircuitContextMatches()&&subsystem&&subsystem->Coupler()&&
        subsystem->GetCoilSet().size()==1;
    if(!Collective(valid))return Reject("circuit source binding unavailable");
    auto const* ext=sim.get_pointer_HybridPICModel()->m_external_vector_potential.get();
    auto const& coil=subsystem->GetCoilSet().coil(0);
    valid=ext&&ext->nFields()==1&&ext->FieldName(0)==coil.field_name&&
        std::isfinite(coil.I_ref)&&coil.I_ref!=0.&&action.InitialPortCurrents().size()==1;
    for(int c=0;c<3;++c){
        valid=valid&&sim.m_fields.has("hybrid_B_static_fp",ablastr::fields::Direction{c},0)&&
            j0[c].nGrowVect()==amrex::IntVect(1);
    }
    if(!Collective(valid))return Reject("circuit source requires native one-port and exact grow1 plasma current");
    q.initial_port_current=action.InitialPortCurrents()[0];q.reference_current=coil.I_ref;
    valid=std::isfinite(q.initial_port_current);
    for(int c=0;c<3;++c){
        auto const& bs=*sim.m_fields.get("hybrid_B_static_fp",ablastr::fields::Direction{c},0);
        auto const& bc=action.InitialCoilMagneticField(c);auto const& jc=action.InitialCoilCurrent(c);
        valid=valid&&bs.boxArray()==b0[c].boxArray()&&bc.boxArray()==b0[c].boxArray()&&
            bs.DistributionMap()==context.Distribution()&&bc.DistributionMap()==context.Distribution()&&
            bs.nComp()==1&&bc.nComp()==1&&jc.nComp()==1&&
            jc.boxArray()==j0[c].boxArray()&&jc.DistributionMap()==context.Distribution()&&
            jc.nGrowVect().allGE(1)&&bc.nGrowVect().allGE(2);
    }
    if(!Collective(valid))return Reject("circuit initial field/current layout mismatch");
    R low[4]={q.initial_port_current,q.reference_current,q.binding.source_time,R(q.binding.phase==NativeStoppingCircuitBinding::Phase::PostField)};
    R high[4];std::copy(low,low+4,high);amrex::ParallelDescriptor::ReduceRealMin(low,4);amrex::ParallelDescriptor::ReduceRealMax(high,4);
    for(int n=0;n<4;++n)valid=valid&&low[n]==high[n];
    for(int c=0;c<3;++c){
        auto const& bs=*sim.m_fields.get("hybrid_B_static_fp",ablastr::fields::Direction{c},0);
        auto const& bc=action.InitialCoilMagneticField(c);auto const& jc=action.InitialCoilCurrent(c);
        bool const fs=bs.is_finite(0,1,bs.nGrowVect());
        bool const fb=bc.is_finite(0,1,bc.nGrowVect());
        bool const fj=jc.is_finite(0,1,jc.nGrowVect());valid=valid&&fs&&fb&&fj;
        AllocateLike(q.static_b[c],bs);AllocateLike(q.coil_b0[c],bc);AllocateLike(q.coil_j0[c],jc);
        AllocateLike(q.free_impulse[c],impulse[c]);q.free_impulse[c].setVal(0.);
        AllocateLike(q.total0[c],j0[c]);AllocateLike(q.total_delta[c],j0[c]);
        AllocateLike(q.coil_b_delta[c],bc);AllocateLike(q.coil_j_delta[c],jc);
        AllocateLike(q.coil_a_delta[c],a1[c]);AllocateLike(q.project_input[c],j0[c]);
    }
    if(!Collective(valid))return Reject("nonfinite or rank-inconsistent circuit source inventory");
    q.port_impulse.resize(1);q.field_increment.resize(1);q.generation=action.Generation();
    q.work={};q.publication={};q.evaluations=0;q.application=0;return true;
}

bool NativeRZSpatialStoppingEvent::Impl::ApplyCircuit() {
    auto& q=*circuit;q.work.available=false;
    if(!Collective(CircuitContextMatches()))return Reject("stale circuit source action");
    for(int c=0;c<3;++c){q.free_impulse[c].setVal(0.);amrex::MultiFab::Copy(q.free_impulse[c],impulse[c],0,0,1,0);}
    auto& action=*q.binding.field;std::string error;
    if(!action.TryApply(Mutable(q.free_impulse),error))return Reject("circuit source field action declined");
    auto* subsystem=sim.get_pointer_CircuitCoupling();auto* coupler=subsystem->Coupler();
    bool valid=coupler->SourceFieldIncrements().size()==1&&coupler->SourcePortImpulses().size()==1;
    for(int c=0;c<3;++c){
        auto compatible=[](amrex::MultiFab const& from,amrex::MultiFab const& to,amrex::IntVect ng){
            return from.boxArray()==to.boxArray()&&from.DistributionMap()==to.DistributionMap()&&
                from.nComp()==1&&from.nGrowVect().allGE(ng);};
        valid=valid&&compatible(action.TotalImpulse(c),impulse[c],impulse[c].nGrowVect())&&
            compatible(action.PotentialIncrement(c),a1[c],a1[c].nGrowVect())&&
            compatible(action.MagneticIncrement(c),b1[c],b1[c].nGrowVect())&&
            compatible(action.TotalCurrentIncrement(c),c1[c],amrex::IntVect(1))&&
            compatible(action.CoilCurrentIncrement(c),c1[c],amrex::IntVect(1))&&
            compatible(action.CoilPotentialIncrement(c),q.coil_a_delta[c],q.coil_a_delta[c].nGrowVect())&&
            compatible(action.CoilMagneticIncrement(c),q.coil_b_delta[c],q.coil_b_delta[c].nGrowVect());
    }
    if(!Collective(valid))return Reject("circuit action does not cover native source gather/work support");
    for(int c=0;c<3;++c){
        amrex::MultiFab::Copy(impulse[c],action.TotalImpulse(c),0,0,1,impulse[c].nGrowVect());
        amrex::MultiFab::Copy(a1[c],action.PotentialIncrement(c),0,0,1,a1[c].nGrowVect());
        amrex::MultiFab::Copy(b1[c],action.MagneticIncrement(c),0,0,1,b1[c].nGrowVect());
        amrex::MultiFab::Copy(q.total_delta[c],action.TotalCurrentIncrement(c),0,0,1,1);
        amrex::MultiFab::Copy(q.coil_j_delta[c],action.CoilCurrentIncrement(c),0,0,1,1);
        amrex::MultiFab::Copy(q.coil_a_delta[c],action.CoilPotentialIncrement(c),0,0,1,q.coil_a_delta[c].nGrowVect());
        amrex::MultiFab::Copy(q.coil_b_delta[c],action.CoilMagneticIncrement(c),0,0,1,q.coil_b_delta[c].nGrowVect());
        amrex::MultiFab::LinComb(c1[c],1.,q.total_delta[c],0,-1.,q.coil_j_delta[c],0,0,1,1);
        bool const fi=impulse[c].is_finite(0,1,impulse[c].nGrowVect());
        bool const fc=c1[c].is_finite(0,1,1);
        bool const fb=b1[c].is_finite(0,1,b1[c].nGrowVect());
        bool const fcb=q.coil_b_delta[c].is_finite(0,1,q.coil_b_delta[c].nGrowVect());
        bool const fca=q.coil_a_delta[c].is_finite(0,1,q.coil_a_delta[c].nGrowVect());
        valid=valid&&fi&&fc&&fb&&fcb&&fca;
    }
    auto const& chi=coupler->SourcePortImpulses();auto const& scale=coupler->SourceFieldIncrements();
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice,chi.begin(),chi.end(),q.port_impulse.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice,scale.begin(),scale.end(),q.field_increment.begin());
    q.application=action.Application();++q.evaluations;
    return Collective(valid)?true:Reject("nonfinite circuit source action");
}

bool NativeRZSpatialStoppingEvent::Impl::Scope () {
#if !defined(WARPX_DIM_RZ) || defined(AMREX_SINGLE_PRECISION_PARTICLES)
    return false;
#else
    auto const* model=sim.get_pointer_HybridPICModel();
    auto const& particles=sim.GetPartContainer();
    bool const source_scope=circuit ?
        (model&&background&&midpoint_thermal&&sim.get_pointer_CircuitCoupling()&&
         model->m_add_external_fields&&!model->m_has_external_current&&
         circuit->binding.field&&CircuitContextMatches()&&context.Geometry().isPeriodic(1)&&
         midpoint_options.initial_guess==NativeStoppingThermalOptions::InitialGuess::Zero) :
        (model&&!sim.get_pointer_CircuitCoupling()&&!model->m_add_external_fields&&!model->m_has_external_current);
    bool ok=source_scope&&
        (particles.m_E_ext_particle_s=="none"||particles.m_E_ext_particle_s=="constant")&&
        (particles.m_B_ext_particle_s=="none"||particles.m_B_ext_particle_s=="constant")&&
        context.Valid()&&context.Options().model_electron_mass==PhysConst::m_e&&
        !context.Binding().pedestal&&sim.finestLevel()==0&&sim.maxLevel()==0&&!EB::enabled()&&
        !sim.getdo_moving_window()&&!sim.get_load_balance_intervals().isActivated()&&
        sim.Geom(0).Domain().smallEnd()==amrex::IntVect(0)&&WarpX::grid_type==GridType::Staggered&&WarpX::nox==3&&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving&&WarpX::ncomps==1&&
        !WarpX::do_shared_mem_current_deposition&&WarpX::field_centering_nox==2&&
        WarpX::field_centering_noy==2&&WarpX::field_centering_noz==2&&
        !WarpX::use_filter&&!WarpX::do_single_precision_comms&&!sim.do_current_centering&&
        !WarpX::galerkin_interpolation&&WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov&&
        std::isfinite(options.interval)&&options.interval>=0.&&options.coulomb_log>0.&&
        std::isfinite(options.coulomb_log)&&options.proper_speed_cap>0.&&
        options.proper_speed_cap<.01*PhysConst::c&&options.relative_convention_budget>0.&&
        std::isfinite(options.relative_convention_budget)&&!options.fast_species.empty();
    for(auto const& name:particles.GetSpeciesNames()){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        for(auto const value:pc.m_E_external_particle){ok=ok&&value==0.;}
        for(auto const value:pc.m_B_external_particle){ok=ok&&value==0.;}
        for(WarpXParIter pti(pc,0);pti.isValid();++pti){ok=ok&&GetExternalEBField(pti).isNoOp();}
    }
    auto const& g=context.Geometry();auto const& bc=context.Options();
    bool const periodic=g.isPeriodic(1);
    ok=ok&&g.IsRZ()&&g.ProbLo(0)==0.&&!g.isPeriodic(0)&&
        bc.field_lo[0]==FieldBoundaryType::None&&bc.field_hi[0]==FieldBoundaryType::PEC&&
        bc.particle_lo[0]==ParticleBoundaryType::None&&bc.particle_hi[0]==ParticleBoundaryType::Reflecting;
    for(int side=0;side<2;++side){
        auto const f=side?bc.field_hi[1]:bc.field_lo[1];
        auto const p=side?bc.particle_hi[1]:bc.particle_lo[1];
        ok=ok&&(periodic?(f==FieldBoundaryType::Periodic&&p==ParticleBoundaryType::Periodic):
            (f==FieldBoundaryType::PMC&&p==ParticleBoundaryType::Reflecting));
    }
    int bg_min=int(background)+2*int(bool(circuit)),bg_max=bg_min;
    amrex::ParallelDescriptor::ReduceIntMin(bg_min);amrex::ParallelDescriptor::ReduceIntMax(bg_max);
    if(bg_min!=bg_max)return false;
    auto const magnetic=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    ok=ok&&energy.ixType().cellCentered()&&energy.nComp()==1&&
        energy.boxArray()==context.Cells()&&energy.DistributionMap()==context.Distribution()&&
        temperature.ixType().nodeCentered()&&temperature.nComp()==1&&
        temperature.boxArray()==amrex::convert(context.Cells(),amrex::IntVect(1))&&
        temperature.DistributionMap()==context.Distribution()&&temperature.nGrowVect().allGE(1);
    // No MultiFab collective is reached before every rank confirms local
    // capabilities and layouts. A rank-local decline cannot skip a peer's
    // finite/norm reduction or dereference a missing destination.
    if(!Collective(ok))return false;
    auto const E=sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp,0);
    for(int c=0;c<3;++c){
        ok=ok&&destinations.potential[c]&&destinations.magnetic[c]&&destinations.displacement[c];
        if(!Collective(ok))return false;
        ok=destinations.magnetic[c]==magnetic[c]&&
            destinations.potential[c]->boxArray()==E[c]->boxArray()&&
            destinations.displacement[c]->boxArray()==E[c]->boxArray()&&
            destinations.potential[c]->DistributionMap()==context.Distribution()&&
            destinations.displacement[c]->DistributionMap()==context.Distribution()&&
            destinations.potential[c]->nGrowVect()==E[c]->nGrowVect()&&
            (destinations.displacement[c]->nGrowVect()==E[c]->nGrowVect()||
             (background&&destinations.displacement[c]->nGrowVect()==amrex::IntVect(1)))&&
            destinations.potential[c]->nComp()==1&&destinations.displacement[c]->nComp()==1;
        if(background){
            auto const* wall_field=background_destinations.conductor_current[c];
            ok=ok&&wall_field&&wall_field->boxArray()==E[c]->boxArray()&&
                wall_field->DistributionMap()==context.Distribution()&&wall_field->nComp()==1&&
                wall_field->nGrowVect()==context.Binding().electron_current[c]->nGrowVect();
        }
    }
    if(background){
        std::vector<amrex::MultiFab const*> outputs;
        for(int c=0;c<3;++c){
            for(auto const* f:std::array<amrex::MultiFab const*,5>{destinations.potential[c],destinations.magnetic[c],
                destinations.displacement[c],background_destinations.conductor_current[c],
                context.Binding().electron_current[c]}){
                ok=ok&&f!=&temperature&&f!=&energy&&f!=context.Binding().raw_charge&&
                    std::find(outputs.begin(),outputs.end(),f)==outputs.end();
                outputs.push_back(f);
            }
        }
    }
    if(!Collective(ok))return false;
    R low[4]={options.interval,options.coulomb_log,options.proper_speed_cap,options.relative_convention_budget};
    R high[4];std::copy(low,low+4,high);
    amrex::ParallelDescriptor::ReduceRealMin(low,4);amrex::ParallelDescriptor::ReduceRealMax(high,4);
    for(int i=0;i<4;++i)ok=ok&&low[i]==high[i];
    int const root=amrex::ParallelDescriptor::IOProcessorNumber();
    int length=static_cast<int>(options.fast_species.size());amrex::ParallelDescriptor::Bcast(&length,1,root);
    std::vector<char> name(length);
    if(amrex::ParallelDescriptor::IOProcessor())std::copy(options.fast_species.begin(),options.fast_species.end(),name.begin());
    amrex::ParallelDescriptor::Bcast(name.data(),length,root);
    ok=ok&&options.fast_species==std::string(name.begin(),name.end());
    if(!Collective(ok))return false;
    bool const finite_energy=energy.is_finite();R const minimum_energy=energy.min(0);
    ok=finite_energy&&minimum_energy>=0.;
    for(auto const* field:magnetic){bool const finite=field->is_finite();R const maximum=field->norm0();ok=ok&&finite&&(background||maximum==0.);}
    bool const finite_temperature=temperature.is_finite();R const minimum_temperature=temperature.min(0);
    ok=ok&&(!background||(finite_temperature&&minimum_temperature>0.));
    for(int c=0;c<3;++c){
        bool const finite_a=destinations.potential[c]->is_finite();
        bool const finite_d=destinations.displacement[c]->is_finite();
        R const norm_a=destinations.potential[c]->norm0(),norm_d=destinations.displacement[c]->norm0();
        ok=ok&&finite_a&&finite_d&&(background||(norm_a==0.&&norm_d==0.));
        if(background){bool const finite_w=background_destinations.conductor_current[c]->is_finite();ok=ok&&finite_w;}
    }
    return Collective(ok);
#endif
}
bool NativeRZSpatialStoppingEvent::Impl::Capture () {
    auto const& g=context.Geometry();generation=context.Generation();binding=context.Binding();
    amrex::MultiFab one_cell(context.Cells(),context.Distribution(),1,0);one_cell.setVal(1.);
    volume=Integral(one_cell,g);
    time=sim.gett_new(0);step=sim.getistep(0);next_id=PC::ParticleType::the_next_id;
    names=sim.GetPartContainer().GetSpeciesNames();species.clear();
    R count=0.;bool found=false,species_valid=true;
    // Native physical neutrals are non-depositing; charged species must deposit.
    for(auto const& name:names){
        auto& pc=sim.GetPartContainer().GetParticleContainerFromName(name);
        if(!dynamic_cast<PhysicalParticleContainer*>(&pc)||dynamic_cast<SubcycledParticleContainer*>(&pc)||pc.finestLevel()!=0||
           pc.DoFieldIonization()||pc.do_not_deposit!=(pc.getCharge()==0.)||!std::isfinite(pc.getCharge())||
           pc.getCharge()<0.||!std::isfinite(pc.getMass())||pc.getMass()<=0.){
            species_valid=false;
        }
        species.emplace_back();auto& s=species.back();s.pc=&pc;s.mass=pc.getMass();s.charge=pc.getCharge();
        s.fast=name==options.fast_species;found=found||s.fast;
        if(s.fast&&s.charge==0.){species_valid=false;}
        s.real_names=pc.GetRealSoANames();s.int_names=pc.GetIntSoANames();
        for(auto const& [key,tile]:pc.GetParticles(0)){
            if(tile.numNeighborParticles()!=0){species_valid=false;}
            auto& copy=s.old[key];copy.define(tile.NumRealComps()-PIdx::nattribs,
                tile.NumIntComps()-IntIdx::nattribs,nullptr,nullptr,pc.arena());
            copy.GetStructOfArrays()=tile.GetStructOfArrays();
            // Neutrals remain in the complete state snapshot, but have no
            // event point, gather, reaction or convention-budget contribution.
            if(s.charge!=0.){
                s.points[key].resize(tile.numParticles());s.impulses[key].resize(tile.numParticles());
                count+=tile.numParticles();
            }
        }
    }
    if(!Collective(found&&species_valid)){return Reject("unsupported or missing fast species");}
    amrex::ParallelDescriptor::ReduceRealSum(count);if(count<=0){return Reject("empty event");}
    ledger.contribution_count=count;
    // Every point can contribute 64 cubic terms, with at most eight periodic
    // images and eight shared-FAB owners. 4096 covers both product evaluations,
    // scalar/cubic gather, mass arithmetic and each reduction path. Bounds use
    // ALL global charged points, not occupancy estimates or observed cancellation.
    ledger.operations=4096.+4096.*count;
    if(ledger.operations*epsilon>=.01){return Reject("unsupported arithmetic depth");}
    arithmetic_factor=4.*Roundoff(ledger.operations);
    AllocateLike(rho0,*binding.raw_charge);AllocateLike(t0,temperature);
    projection_rhs.define(rho0.boxArray(),context.Distribution(),1,1);projection_rhs.setVal(0.);
    AllocateLike(u0,energy);AllocateLike(u1,energy);
    heat.define(t0.boxArray(),t0.DistributionMap(),1,t0.nGrowVect());heat.setVal(0.);
    auto const native_current=sim.m_fields.get_alldirs(warpx::fields::FieldType::current_fp,0);
    auto const native_electric=sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp,0);
    narrow_current=false;
    for(int c=0;c<3;++c){
        if(!binding.electron_current[c]->nGrowVect().allGE(1)){
            return Reject("accepted current lacks native grow1 support");
        }
        narrow_current=narrow_current||!binding.electron_current[c]->nGrowVect().allGE(native_current[c]->nGrowVect());
    }
    for(int c=0;c<3;++c){
        auto const& a=*binding.electron_current[c];
        if(narrow_current&&a.nGrowVect()!=amrex::IntVect(1)){
            return Reject("narrow accepted current must use exact native grow1 storage");
        }
        AllocateLike(j0[c],a);AllocateLike(j1[c],a);
        auto const private_ghosts=amrex::max(a.nGrowVect(),amrex::max(context.Options().ghosts,
            amrex::max(native_current[c]->nGrowVect(),native_electric[c]->nGrowVect())));
        for(auto* f:{&mean[c],&ve[c],&mass[c],&impulse[c],&i0[c],&i1[c],&scratch[c],&reaction_b[c],&wall[c]}){
            if(narrow_current){
                f->define(a.boxArray(),a.DistributionMap(),a.nComp(),private_ghosts);
                f->setVal(0.);amrex::MultiFab::Copy(*f,a,0,0,a.nComp(),a.nGrowVect());
            }else{AllocateLike(*f,a);}
        }
        if(narrow_current){wall[c].setVal(0.);}
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
        R const tref=reference_temperature;
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            amrex::ignore_unused(r);a(i,j,k)=std::abs(t(i,j,k)/tref-1.);
        });
    }
    ledger.uniform_error=background?0.:rtmp.norm0();
    amrex::MultiFab deposited(rho0.boxArray(),rho0.DistributionMap(),rho0.nComp(),rho0.nGrowVect());
    deposited.setVal(0.);sim.GetPartContainer().DepositCharge({&deposited},0.);
    sim.SyncRho({&deposited},{},{});
    sim.ApplyRhofieldBoundary(0,&deposited,PatchType::fine);
    amrex::MultiFab::LinComb(rtmp,1.,deposited,0,-1.,rho0,0,0,1,0);
    ledger.uniform_error=std::max(ledger.uniform_error,rtmp.norm0()/reference_rho);
    current_scale=reference_rho*options.proper_speed_cap;
    psi_scale=PhysConst::m_e*options.proper_speed_cap/PhysConst::q_e;
    ledger.current_scale=current_scale;
    ledger.current_initial_error=0.;
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(scratch[c],1.,j0[c],0,1.,i0[c],0,0,1,0);
        ledger.current_initial_error=std::max(ledger.current_initial_error,scratch[c].norm0());
    }
    if(ledger.uniform_error>ledger.uniform_bound||(!background&&ledger.current_initial_error>arithmetic_factor*current_scale)){
        return Reject("nonuniform temperature, raw-density mismatch or unconstrained spatial context");
    }
    host_rng=HostRandom();
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    device_rng.resize(ThermalRandomDeviceBytes());
    amrex::Gpu::dtod_memcpy(device_rng.data(),amrex::getRandState(),device_rng.size());
#endif
    for(int c=0;c<3;++c){
        AllocateLike(a0[c],*destinations.potential[c]);AllocateLike(a1[c],a0[c]);
        AllocateLike(b0[c],*destinations.magnetic[c]);AllocateLike(b1[c],b0[c]);AllocateLike(bcheck[c],b0[c]);
        AllocateLike(d0[c],*destinations.displacement[c]);AllocateLike(d1[c],d0[c]);
        for(auto* f:{&c1[c],&plpsi[c],&delta[c]}){AllocateLike(*f,j0[c]);f->setVal(0.);}
        AllocateLike(zero[c],a0[c]);zero[c].setVal(0.);
        auto const* native_aux=sim.m_fields.get(warpx::fields::FieldType::Efield_aux,ablastr::fields::Direction{c},0);
        AllocateLike(aux[c],*native_aux);
        AllocateLike(e0[c],*sim.m_fields.get(warpx::fields::FieldType::Efield_fp,ablastr::fields::Direction{c},0));
        owners[c]=j0[c].OwnerMask(g.periodicity());
    }
    warpx::darwin::InitialRateSchurOptions po;po.compatible_yee=true;
    po.relative_tolerance=1.e-12;po.absolute_tolerance=0.;po.output_ghosts=j0[0].nGrowVect();
    using BC=warpx::darwin::InitialRateBoundary;
    po.lower[0]=BC::Axis;po.upper[0]=BC::PEC;
    po.lower[1]=po.upper[1]=g.isPeriodic(1)?BC::Periodic:BC::PMC;
    projection=std::make_unique<warpx::darwin::DarwinInitialRateSchur>(g,context.Cells(),context.Distribution(),po);
    amrex::MultiFab unit(context.Kappa().boxArray(),context.Distribution(),1,context.Kappa().nGrowVect());unit.setVal(1.);
    if(!projection->Freeze(unit)){return Reject("projection context");}
    if(background_destinations.longitudinal_certificate) {
        auto const& phi=projection->CorrectionPotential();
        displacement_potential=std::make_unique<amrex::MultiFab>(phi.boxArray(),phi.DistributionMap(),1,phi.nGrowVect());
        displacement_potential->setVal(0.);
    }
    ThermalMomentOptions mo;mo.number_density_floor=context.Options().number_density_floor;
    mo.active_density_floor=mo.number_density_floor;mo.nodal_ghosts=t0.nGrow();
#if defined(WARPX_DIM_RZ)
    mo.verboncoeur_axis_correction=sim.UseVerboncoeurAxisCorrection();
#endif
    mo.boundary[0]={MomentBoundary::Axis,MomentBoundary::PEC};
    mo.current_boundary[0]={MomentBoundary::Axis,MomentBoundary::PMC};
    mo.boundary[1]={g.isPeriodic(1)?MomentBoundary::Periodic:MomentBoundary::PMC,
        g.isPeriodic(1)?MomentBoundary::Periodic:MomentBoundary::PMC};
    heat_map=std::make_unique<KineticThermalMoments>(g,context.Cells(),context.Distribution(),mo);
    reaction=std::make_unique<NativeStoppingReactionTranspose>(context);sums.resize(32);
    if(circuit&&!CaptureCircuit()){return false;}
    if(background&&!CaptureBackground()){return false;}
    amrex::Gpu::synchronize();return true;
}

bool NativeRZSpatialStoppingEvent::Impl::CaptureBackground () {
    auto const& g=context.Geometry();
    R curl_scale=0.,field_scale=0.,displacement_scale=0.;
    R const derivative_scale=4./std::min(g.CellSize(0),g.CellSize(1));
    for(int c=0;c<3;++c){
        AllocateLike(c0[c],j0[c]);c0[c].setVal(0.);
        AllocateLike(w0[c],*background_destinations.conductor_current[c]);
        AllocateLike(full_a[c],a0[c]);AllocateLike(full_b[c],b0[c]);
        AllocateLike(full_d[c],d0[c]);AllocateLike(full_w[c],w0[c]);
        curl_scale+=derivative_scale*a0[c].norm0();field_scale+=b0[c].norm0();
        displacement_scale+=derivative_scale*d0[c].norm0();
        bcheck[c].setVal(0.);
        // A conductor reaction is a tangential covector, never normal plasma
        // current. Unsupported support is rejected, not projected away.
        for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
            auto const out=scratch[c].array(mfi);auto const w=w0[c].const_array(mfi);
            int const high=g.Domain().bigEnd(0)+1;
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                out(i,j,k)=(c!=0&&i==high)?0.:std::abs(w(i,j,k));
            });
        }
        if(scratch[c].norm0()!=0.){return Reject("background conductor support");}
    }
    auto magnetic=Mutable(bcheck),potential=Mutable(a0);
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(magnetic,potential,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    if(circuit)for(int c=0;c<3;++c){
        amrex::MultiFab::Add(bcheck[c],circuit->static_b[c],0,0,1,0);
        field_scale+=circuit->static_b[c].norm0();
    }
    background_ledger.magnetic_bound=4096.*epsilon*(curl_scale+field_scale);
    for(int c=0;c<3;++c){
        amrex::MultiFab::Subtract(bcheck[c],b0[c],0,0,1,0);
        background_ledger.magnetic_residual=std::max(background_ledger.magnetic_residual,bcheck[c].norm0());
        bcheck[c].setVal(0.);
    }
    if(background_ledger.magnetic_residual>background_ledger.magnetic_bound){return Reject("background magnetic representation");}
    auto displacement=Mutable(d0);
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(magnetic,displacement,sim.GetEBUpdateBFlag()[0],0);
    background_ledger.displacement_bound=4096.*epsilon*displacement_scale;
    for(int c=0;c<3;++c){background_ledger.displacement_residual=std::max(background_ledger.displacement_residual,bcheck[c].norm0());}
    if(background_destinations.longitudinal_certificate) {
        auto const& proof=*background_destinations.longitudinal_certificate;
        bool const bound=proof.Matches(sim,binding.endpoint_time,binding.endpoint_epoch);
        bool const curl=proof.ValidateCurl(Const(bcheck),background_ledger.displacement_certificate_bound);
        background_ledger.displacement_certified=bound&&curl;
        if(!background_ledger.displacement_certified)return Reject("background displacement producer certificate");
    } else if(background_ledger.displacement_residual>background_ledger.displacement_bound){return Reject("background displacement curl");}
    auto current=Mutable(c0),old_b=Mutable(b0);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(current,old_b,sim.GetEBUpdateEFlag()[0],0);
    Sync(c0,g);
    if(circuit)for(int c=0;c<3;++c){
        amrex::MultiFab::Copy(circuit->total0[c],c0[c],0,0,1,1);
        amrex::MultiFab::Subtract(c0[c],circuit->coil_j0[c],0,0,1,1);
    }
    R scale=current_scale;
    for(int c=0;c<3;++c){
        scale+=c0[c].norm0()+i0[c].norm0()+j0[c].norm0()+d0[c].norm0()+w0[c].norm0();
        amrex::MultiFab::LinComb(scratch[c],1.,i0[c],0,1.,j0[c],0,0,1,0);
        amrex::MultiFab::Add(scratch[c],d0[c],0,0,1,0);
        amrex::MultiFab::Add(scratch[c],w0[c],0,0,1,0);
        amrex::MultiFab::Subtract(scratch[c],c0[c],0,0,1,0);
        background_ledger.current_residual=std::max(background_ledger.current_residual,scratch[c].norm0());
    }
    background_ledger.current_bound=arithmetic_factor*scale;
    ledger.current_initial_error=background_ledger.current_residual;
    if(background_ledger.current_residual>background_ledger.current_bound){return Reject("background Ampere constraint");}
    return true;
}

void NativeRZSpatialStoppingEvent::Impl::AssembleBackground () {
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(full_a[c],1.,a0[c],0,1.,a1[c],0,0,1,full_a[c].nGrowVect());
        amrex::MultiFab::LinComb(full_b[c],1.,b0[c],0,1.,b1[c],0,0,1,full_b[c].nGrowVect());
        amrex::MultiFab::LinComb(full_d[c],1.,d0[c],0,1.,d1[c],0,0,1,full_d[c].nGrowVect());
        amrex::MultiFab::LinComb(full_w[c],1.,w0[c],0,1.,wall[c],0,0,1,full_w[c].nGrowVect());
    }
}

bool NativeRZSpatialStoppingEvent::Impl::Unchanged (bool after) {
    bool ok=(!circuit||CircuitContextMatches())&&
        (after||(context.Valid()&&context.Generation()==generation))&&sim.gett_new(0)==time&&
        sim.boxArray(0)==context.Cells()&&sim.DistributionMap(0)==context.Distribution()&&
        sim.getistep(0)==step&&PC::ParticleType::the_next_id==next_id&&HostRandom()==host_rng&&
        names==sim.GetPartContainer().GetSpeciesNames()&&Equal(rho0,*binding.raw_charge)&&
        Equal(t0,temperature)&&Equal(after?u1:u0,energy);
    for(int c=0;c<3;++c){ok=ok&&Equal(after?j1[c]:j0[c],*binding.electron_current[c])&&
        Equal(after?(background?full_a[c]:a1[c]):a0[c],*destinations.potential[c])&&
        Equal(after?(background?full_b[c]:b1[c]):b0[c],*destinations.magnetic[c])&&
        Equal(after?(background?full_d[c]:d1[c]):d0[c],*destinations.displacement[c])&&
        (!background||Equal(after?full_w[c]:w0[c],*background_destinations.conductor_current[c]))&&
        Equal(e0[c],*sim.m_fields.get(warpx::fields::FieldType::Efield_fp,ablastr::fields::Direction{c},0));}
    if(circuit)for(int c=0;c<3;++c){
        ok=ok&&Equal(circuit->static_b[c],*sim.m_fields.get("hybrid_B_static_fp",ablastr::fields::Direction{c},0))&&
            Equal(circuit->coil_b0[c],circuit->binding.field->InitialCoilMagneticField(c))&&
            Equal(circuit->coil_j0[c],circuit->binding.field->InitialCoilCurrent(c));
    }
    amrex::Gpu::DeviceScalar<int> bad(0);
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    if(device_rng.size()!=ThermalRandomDeviceBytes()){ok=false;}
    else{ByteCompare(device_rng.data(),amrex::getRandState(),device_rng.size(),bad.dataPtr());}
#endif
    for(auto const& s:species){
        auto const& pc=*s.pc;
        if(pc.GetRealSoANames()!=s.real_names||pc.GetIntSoANames()!=s.int_names||
           pc.getMass()!=s.mass||pc.getCharge()!=s.charge||pc.DoFieldIonization()||
           pc.do_not_deposit!=(s.charge==0.)||pc.GetParticles(0).size()!=s.old.size()){
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
                if(after&&s.charge!=0.&&(c==PIdx::ux||c==PIdx::uy||c==PIdx::uz)){
                    auto const* p=s.points.at(key).data();auto const* u=a.GetRealData(c).data();
                    int const component=c-PIdx::ux;auto* invalid=bad.dataPtr();
                    amrex::For(now.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                        auto const* expected=reinterpret_cast<unsigned char const*>(&p[n].next[component]);
                        auto const* actual=reinterpret_cast<unsigned char const*>(u+n);
                        for(std::size_t byte=0;byte<sizeof(P);++byte){if(expected[byte]!=actual[byte]){amrex::HostDevice::Atomic::Add(invalid,1);}}
                    });
                }else{ByteCompare(a.GetRealData(c).data(),b.GetRealData(c).data(),a.GetRealData(c).size()*sizeof(P),bad.dataPtr());}
            }
            for(int c=0;c<now.NumIntComps();++c){ByteCompare(a.GetIntData(c).data(),b.GetIntData(c).data(),a.GetIntData(c).size()*sizeof(int),bad.dataPtr());}
        }
    }
    amrex::Gpu::synchronize();return Collective(ok&&bad.dataValue()==0);
}

void NativeRZSpatialStoppingEvent::Impl::Deposit (Species& s,WarpXParIter const& pti,Point const* p,long n) {
#if defined(WARPX_DIM_RZ)
    auto box=pti.tilebox();box.grow(sim.get_ng_depos_J());auto const origin=WarpX::LowerCorner(box,0,0.);
    auto const lower=amrex::lbound(box);auto const inverse=WarpX::InvCellSize(0);
    auto const jx=i1[0].array(pti),jy=i1[1].array(pti),jz=i1[2].array(pti);
    amrex::ignore_unused(s);
    amrex::For(n,[=] AMREX_GPU_DEVICE(long n0){
        auto const point=p[n0];auto const u=point.next;R const gamma=Gamma(u);
        R const xp=point.position[0],yp=point.position[1],rr=std::sqrt(xp*xp+yp*yp);
        R const vx=u[0]/gamma,vy=u[1]/gamma,vz=u[2]/gamma;
        V x{(rr-origin.x)*inverse.x,0.,(point.position[2]-origin.z)*inverse.z};
        V v{(xp*vx+yp*vy)/rr*inverse.x,0.,vz*inverse.z};
        R const vt=(-yp*vx+xp*vy)/rr;
        int const first_r=int(x[0])-1,first_z=int(x[2])-1;
        for(int iz=first_z;iz<=first_z+3;++iz){for(int ir=first_r;ir<=first_r+3;++ir){
            V value{},unused{};
            warpx::particles::EsirkepovCurrentRate<3,2>(x,v,V{},amrex::GpuArray<int,3>{ir,0,iz},vt,0.,value,unused);
            int const ii=ir+lower.x,jj=iz+lower.y;
            R const a=point.charge*point.weight*inverse.y*inverse.z;
            R const b=point.charge*point.weight*inverse.x*inverse.y*inverse.z;
            R const c=point.charge*point.weight*inverse.x*inverse.y;
            if(ir<first_r+3){amrex::HostDevice::Atomic::Add(&jx(ii,jj,0),a*value[0]);}
            amrex::HostDevice::Atomic::Add(&jy(ii,jj,0),b*value[1]);
            if(iz<first_z+3){amrex::HostDevice::Atomic::Add(&jz(ii,jj,0),c*value[2]);}
        }}
    });
#else
    amrex::ignore_unused(s,pti,p,n);
#endif
}

SpatialVector NativeRZSpatialStoppingEvent::Impl::Vector()const{
    SpatialVector v;for(int c=0;c<6;++c){v.a[c].define(j0[c%3].boxArray(),context.Distribution(),1,j0[c%3].nGrowVect());v.a[c].setVal(0.);}return v;
}
R NativeRZSpatialStoppingEvent::Impl::VectorDot(SpatialVector const& a,SpatialVector const& b)const{
    R result=0.;
    for(int c=0;c<6;++c){amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<R> data(op);using T=typename decltype(data)::Type;
        for(amrex::MFIter mfi(a.a[c]);mfi.isValid();++mfi){auto const x=a.a[c].const_array(mfi),y=b.a[c].const_array(mfi);auto const own=owners[c%3]->const_array(mfi);auto const measure=MakeQdsmcVolumeElement(context.Geometry(),a.a[c].ixType());
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T{return{own(i,j,k)?x(i,j,k)*y(i,j,k)*measure(i,j,k):0.};});}
        result+=amrex::get<0>(data.value());}
    amrex::ParallelDescriptor::ReduceRealSum(result);return result/volume;
}
R NativeRZSpatialStoppingEvent::Impl::VectorMax(SpatialVector const& v)const{
    R result=0.;for(auto const& f:v.a){result=std::max(result,f.norm0());}return result;
}
bool NativeRZSpatialStoppingEvent::Impl::Project(CV const& in,std::array<amrex::MultiFab,3>& out,bool current){
    // Radial PEC scalar Dirichlet removes the constant null mode. No periodic
    // Cartesian mean subtraction is applicable to this physical boundary map.
    auto const result=projection->Correct(in,Const(zero));
    if(midpoint_thermal){++midpoint_receipt.physical_projection_calls;midpoint_receipt.physical_projection_iterations+=result.iterations;}
    ledger.projection_residual=std::max(ledger.projection_residual,result.residual);
    if(!result.converged){return false;}
    if(current && displacement_potential) {
        auto const& phi=projection->CorrectionPotential();
        amrex::MultiFab::Copy(*displacement_potential,phi,0,0,1,phi.nGrowVect());
    }
    auto field=projection->CorrectionField();
    for(int c=0;c<3;++c){out[c].setVal(0.);amrex::MultiFab::Copy(out[c],*field[c],0,0,1,
        amrex::min(out[c].nGrowVect(),field[c]->nGrowVect()));}
    Sync(out,context.Geometry());return true;
}
void NativeRZSpatialStoppingEvent::Impl::AImages(std::array<amrex::MultiFab,3>& a){
#if defined(WARPX_DIM_RZ)
    auto const& g=context.Geometry();amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    lo[1]=hi[1]=!g.isPeriodic(1);
    for(int c=0;c<3;++c){
        auto const domain=amrex::convert(g.Domain(),a[c].ixType());int const high=domain.bigEnd(0);
        bool const nodal=a[c].ixType().nodeCentered(0);
        for(amrex::MFIter mfi(a[c]);mfi.isValid();++mfi){auto const v=a[c].array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                if((c==1&&i==0)||(nodal&&i>=high)){v(i,j,k)=0.;}
            });}
        a[c].FillBoundary(g.periodicity());
        ApplyDarwinCellCenteredABoundary(a[c],zero[c],g,true,nullptr,lo,hi);
    }
    auto v=Mutable(a);sim.ApplyFieldBoundaryOnAxis(v[0],v[1],v[2],0);
#else
    amrex::ignore_unused(a);
#endif
}
void NativeRZSpatialStoppingEvent::Impl::ElectricImages(std::array<amrex::MultiFab,3>& a){
#if defined(WARPX_DIM_RZ)
    auto const& g=context.Geometry();Sync(a,g);auto v=Mutable(a);
    amrex::Vector<amrex::IntVect> ratios;
    PEC::ApplyPECtoEfield(v,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PEC,a[0].nGrowVect(),g,0,PatchType::fine,ratios);
    PEC::ApplyPECtoBfield(v,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PMC,a[0].nGrowVect(),g,0,PatchType::fine,ratios);
    sim.ApplyFieldBoundaryOnAxis(v[0],v[1],v[2],0);Sync(a,g);
#else
    amrex::ignore_unused(a);
#endif
}
void NativeRZSpatialStoppingEvent::Impl::VelocityImages(std::array<amrex::MultiFab,3>& a){
    Sync(a,context.Geometry());
    for(int c=0;c<3;++c){auto const image=MakeStoppingVelocityImages(context.Geometry(),context.Options(),a[c].ixType(),c);
        for(amrex::MFIter mfi(a[c]);mfi.isValid();++mfi){auto const v=a[c].array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                amrex::IntVect source(AMREX_D_DECL(i,j,k));auto const original=source;
                R const parity=image(source);if(source!=original){v(i,j,k)=parity*v(source);}
            });}
    }
}
void NativeRZSpatialStoppingEvent::Impl::Curl(){
    auto const& g=context.Geometry();
    for(int c=0;c<3;++c){a1[c].setVal(0.);amrex::MultiFab::Copy(a1[c],impulse[c],0,0,1,0);
        a1[c].mult(-1.,0,1,a1[c].nGrow());b1[c].setVal(0.);c1[c].setVal(0.);}
    Sync(a1,g);AImages(a1);
    auto av=Mutable(a1),bv=Mutable(b1),cv=Mutable(c1);
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(bv,av,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    Sync(b1,g);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv,bv,sim.GetEBUpdateEFlag()[0],0);
    Sync(c1,g);
}
bool NativeRZSpatialStoppingEvent::Impl::Evaluate (SpatialVector const& x,SpatialVector& residual) {
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(x,residual);return false;
#else
    ++ledger.evaluations;
    auto const& g=context.Geometry();
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(mean[c],1.,j0[c],0,current_scale,x.a[c+3],0,0,1,0);
        amrex::MultiFab::Copy(impulse[c],x.a[c],0,0,1,0);impulse[c].mult(psi_scale,0,1,0);
        i1[c].setVal(0.);
    }
    for(amrex::MFIter mfi(mean[1]);mfi.isValid();++mfi){auto const a=mean[1].array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){if(i==0){a(i,j,k)=0.;}});}
    Sync(mean,g);
    if(circuit){if(!ApplyCircuit())return false;}else{ElectricImages(impulse);}
    auto av=Mutable(aux),iv=Mutable(impulse);sim.InterpolateLevelZeroFieldToAux(av,iv);
    warpx::darwin::ApplyYeeInertiaMass(g,context.Kappa(),Const(mean),Mutable(ve),
        PhysConst::m_e/(PhysConst::q_e*PhysConst::q_e*context.Options().reference_number_density));
    for(auto& a:ve){a.mult(-PhysConst::q_e/PhysConst::m_e,0,1,a.nGrow());}
    VelocityImages(ve);heat.setVal(0.);R* init=sums.data();amrex::ParallelFor(int(sums.size()),[=] AMREX_GPU_DEVICE(int n){init[n]=0.;});
    amrex::Gpu::DeviceScalar<int> invalid(0);int* bad=invalid.dataPtr();R* totals=sums.data();
    if(!reaction->Begin()){return false;}
    auto const magnetic=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    CV bfield{magnetic[0],magnetic[1],magnetic[2]};
    auto const inverse=g.InvCellSizeArray(),plo=g.ProbLoArray(),dx=g.CellSizeArray();
    auto const node_volume=MakeQdsmcVolumeElement(g,heat.ixType());
    R const interval=options.interval,clog=options.coulomb_log;amrex::ignore_unused(dx);
    R const cap=options.proper_speed_cap,floor=PhysConst::q_e*context.Options().number_density_floor;
    for(auto& s:species){
        if(s.charge==0.){continue;}
        for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
            auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());
            auto const& old=s.old.at(key).GetStructOfArrays();
            GetParticlePosition<PIdx> position;
            position.m_x=old.GetRealData(PIdx::r).data();position.m_theta=old.GetRealData(PIdx::theta).data();position.m_z=old.GetRealData(PIdx::z).data();
            auto const* ux=old.GetRealData(PIdx::ux).data();auto const* uy=old.GetRealData(PIdx::uy).data();
            auto const* uz=old.GetRealData(PIdx::uz).data();auto const* weight=old.GetRealData(PIdx::w).data();
            auto const* angle=old.GetRealData(PIdx::theta).data();
            auto gather=context.GatherView(pti,bfield),egather=gather;
            for(int c=0;c<3;++c){gather.velocity[c]=ve[c].const_array(pti);egather.velocity[c]=aux[c].const_array(pti);egather.velocity_type[c]=aux[c].ixType();}
            auto const r=rho0.const_array(pti);
            auto const t=midpoint_thermal?midpoint_moments->NodalTemperature().const_array(pti):t0.const_array(pti);
            auto const h=heat.array(pti);
            auto* points=s.points.at(key).data();auto* impulses=s.impulses.at(key).data();
            R const m=s.mass,q=s.charge;bool const fast=s.fast;
            amrex::For(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){
                Point p;p.mass=m;p.charge=q;p.weight=weight[n];p.old={ux[n],uy[n],uz[n]};
                position(n,p.position[0],p.position[1],p.position[2]);
                auto const vel=gather(p.position[0],p.position[1],p.position[2]);
                auto const electric=egather(p.position[0],p.position[1],p.position[2]);
                bool ok=vel.valid&&electric.valid&&p.weight>=0.&&std::isfinite(p.weight)&&
                    std::sqrt(p.position[0]*p.position[0]+p.position[1]*p.position[1])>0.;
                p.ve=vel.cartesian;p.psi=electric.cartesian;
                R const rho=ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],r,inverse,plo);
                R const te=amrex::max(ablastr::particles::doGatherScalarFieldNodal(p.position[0],p.position[1],p.position[2],t,inverse,plo),1.e-3*PhysConst::q_e/PhysConst::kb);
                ok=ok&&rho>floor&&std::isfinite(te)&&Norm(p.old)<=cap;
                for(int c=0;c<3;++c){p.minus[c]=p.old[c]+q*p.psi[c]/(2.*m);}
                p.drag=p.minus;
                if(fast&&interval>0.){p.rate=warpx::particles::NativeStoppingRate(rho*(1./PhysConst::q_e),te*PhysConst::kb,q,m,clog);
                    // Preserve the native collision-frame rotations for the fast
                    // species, without rotating any accepted particle array.
                    R const theta=angle[n],cm=std::cos(-theta),sm=std::sin(-theta);
                    V const local{p.minus[0]*cm-p.minus[1]*sm,p.minus[0]*sm+p.minus[1]*cm,p.minus[2]};
                    auto const kick=warpx::particles::NativeStoppingKick(local,vel.collision_frame,p.rate,interval,p.weight,m).proper;
                    R const co=std::cos(theta),si=std::sin(theta);
                    p.drag={kick[0]*co-kick[1]*si,kick[0]*si+kick[1]*co,kick[2]};}
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
                R values[20]={p.weight*m*Dot(p.old,p.old)/(g0+1.),dki,drag,bulk,electric_work,
                    endpoint_work,p.heat,electric_work-endpoint_work,
                    p.weight*std::abs(q)*Norm(p.psi)*cap*cap*cap/(PhysConst::c*PhysConst::c),
                    std::abs(dki)+std::abs(drag)+std::abs(bulk)+std::abs(electric_work),
                    p.weight*m*(p.next[0]-p.old[0]),p.weight*m*(p.next[1]-p.old[1]),p.weight*m*(p.next[2]-p.old[2]),
                    p.weight*tuple.impulse[0],p.weight*tuple.impulse[1],p.weight*tuple.impulse[2],p.weight*m*cap,p.weight*q*p.psi[0],p.weight*q*p.psi[1],p.weight*q*p.psi[2]};
                for(int c=0;c<20;++c){amrex::HostDevice::Atomic::Add(totals+c,values[c]);}
                int ii=0,jj=0,kk=0;R weights[3][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(p.position[0],p.position[1],p.position[2],plo,inverse,ii,jj,kk,weights);
                for(int dz=0;dz<2;++dz){for(int dr=0;dr<2;++dr){
                    amrex::HostDevice::Atomic::Add(&h(ii+dr,jj+dz,0),
                        p.heat*weights[0][dr]*weights[1][dz]/node_volume(ii+dr,jj+dz,0));
                }}
            });
            amrex::Gpu::streamSynchronize();
            if(invalid.dataValue()==0){reaction->DepositTile(pti,impulses,pti.numParticles());Deposit(s,pti,points,pti.numParticles());}
        }
    }
    amrex::Gpu::synchronize();if(!Collective(invalid.dataValue()==0)){return false;}
    if(!reaction->Finish()){return false;}
    sim.ApplyInverseVolumeScalingToCurrentDensity(&i1[0],&i1[1],&i1[2],0);
    for(auto& a:i1){ablastr::utils::communication::SumBoundary(a,0,1,a.nGrowVect(),a.nGrowVect(),false,g.periodicity());}
    sim.ApplyJfieldBoundary(0,&i1[0],&i1[1],&i1[2],PatchType::fine);Sync(i1,g);
    ablastr::utils::communication::SumBoundary(heat,0,1,heat.nGrowVect(),heat.nGrowVect(),false,g.periodicity());
    heat.OverrideSync(g.periodicity());heat.FillBoundary(g.periodicity());
    auto const raw_reaction=reaction->InertiaImpulse();
    for(int c=0;c<3;++c){amrex::MultiFab::Copy(reaction_b[c],*raw_reaction[c],0,0,1,0);}
    // Restrict the covector to the regular velocity space: Vtheta(axis)=0.
    // This is a zero-work axis constraint, not conductor current or heat.
    for(amrex::MFIter mfi(reaction_b[1]);mfi.isValid();++mfi){auto const a=reaction_b[1].array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){if(i==0){a(i,j,k)=0.;}});}
    Sync(reaction_b,g);auto const reaction_impulse=Const(reaction_b);
    for(int c=0;c<3;++c){
        for(amrex::MFIter mfi(j1[c]);mfi.isValid();++mfi){
            auto const a=j1[c].array(mfi);auto const old=j0[c].const_array(mfi),coefficient=mass[c].const_array(mfi),ri=reaction_impulse[c]->const_array(mfi);
            auto const psi=impulse[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=old(i,j,k)+(psi(i,j,k)+ri(i,j,k))/coefficient(i,j,k);});
        }
    }
    Sync(j1,g);if(!circuit){Curl();}
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(delta[c],1.,i1[c],0,1.,j1[c],0,0,1,0);
        amrex::MultiFab::Subtract(delta[c],i0[c],0,0,1,0);amrex::MultiFab::Subtract(delta[c],j0[c],0,0,1,0);
    }
    Sync(delta,g);
    CV projection_input=Const(delta);
    if(circuit){
        for(int c=0;c<3;++c){amrex::MultiFab::LinComb(circuit->project_input[c],1.,delta[c],0,-1.,c1[c],0,0,1,0);}
        Sync(circuit->project_input,g);projection_input=Const(circuit->project_input);
    }
    if(!Project(projection_input,d1,true)||!Project(Const(impulse),plpsi,false)){return false;}
    for(int c=0;c<3;++c){
        d1[c].mult(-1.,0,1,d1[c].nGrow());
        for(amrex::MFIter mfi(residual.a[c]);mfi.isValid();++mfi){
            auto const out=residual.a[c].array(mfi),outm=residual.a[c+3].array(mfi);
            auto const dc=c1[c].const_array(mfi),di=delta[c].const_array(mfi),dd=d1[c].const_array(mfi),lp=plpsi[c].const_array(mfi);
            auto const old=j0[c].const_array(mfi),next=j1[c].const_array(mfi),trial=mean[c].const_array(mfi);
            R const cs=current_scale,ps=psi_scale;int const radial_high=g.Domain().bigEnd(0)+1;
            auto const w=wall[c].array(mfi);auto const unknown=x.a[c].const_array(mfi),unknown_mean=x.a[c+3].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                bool const conductor=c!=0&&i==radial_high;
                bool const axis=c==1&&i==0;
                w(i,j,k)=conductor?dc(i,j,k)-di(i,j,k)-dd(i,j,k):0.;
                out(i,j,k)=(conductor||axis)?unknown(i,j,k):
                    (di(i,j,k)+dd(i,j,k)-dc(i,j,k))/cs+lp(i,j,k)/ps;
                outm(i,j,k)=axis?unknown_mean(i,j,k):
                    (trial(i,j,k)-.5*(old(i,j,k)+next(i,j,k)))/cs;
            });
        }
        if(!residual.a[c].is_finite()||!residual.a[c+3].is_finite()){return false;}
    }
    CompleteNarrowCurrentGuards();
    return true;
#endif
}

void NativeRZSpatialStoppingEvent::Impl::CompleteNarrowCurrentGuards () {
#if defined(WARPX_DIM_RZ)
    if(!narrow_current){return;}
    auto const& g=context.Geometry();
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    lo[1]=hi[1]=!g.isPeriodic(1);
    // The physical conductor covector is separate from the plasma current.
    // Its cap images retain the same support; no PEC electric projector acts
    // on these nonzero constitutive rows.
    Sync(wall,g);
    for(auto& w:wall){ApplyDarwinPMCVectorBoundary(w,g,lo,hi);}
    for(int c=0;c<3;++c){
        auto const domain=amrex::convert(g.Domain(),j1[c].ixType());
        for(amrex::MFIter mfi(j1[c]);mfi.isValid();++mfi){
            auto const q=j1[c].array(mfi);
            auto const C=c1[c].const_array(mfi),I=i1[c].const_array(mfi),D=d1[c].const_array(mfi),W=wall[c].const_array(mfi);
            auto const C0=background?c0[c].const_array(mfi):C,D0=background?d0[c].const_array(mfi):D;
            auto const W0=background?w0[c].const_array(mfi):W;bool const bg=background;
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                bool const exterior=i<domain.smallEnd(0)||i>domain.bigEnd(0);
                if(exterior){q(i,j,k)=bg?C0(i,j,k)+C(i,j,k)-I(i,j,k)-D0(i,j,k)-D(i,j,k)-W0(i,j,k)-W(i,j,k):
                    C(i,j,k)-I(i,j,k)-D(i,j,k)-W(i,j,k);}
            });
        }
    }
    Sync(j1,g);
    // The independent valid plasma unknown carries the finite solve residual.
    // Complete cap ghosts from that same unknown, exactly as the live current
    // companion does. Reconstructing them independently from C-I-D-W would
    // replace the residual by a different cap value at the next phase rotation.
    // This is the native PMC vector extension, never deposited-ion folding;
    // all valid rows and the radial/axis producer values remain untouched.
    for(auto& q:j1){ApplyDarwinPMCVectorBoundary(q,g,lo,hi);}
#endif
}

bool NativeRZSpatialStoppingEvent::Impl::Solve () {
    auto x=Vector(),f=Vector();
    auto assign=[](SpatialVector& to,SpatialVector const& from){for(int c=0;c<6;++c){amrex::MultiFab::Copy(to.a[c],from.a[c],0,0,1,0);}};
    struct Ops {
        using RT=R;Impl& p;SpatialVector const& base;
        SpatialVector makeVecRHS(){return p.Vector();}SpatialVector makeVecLHS(){return p.Vector();}
        void assign(SpatialVector& a,SpatialVector const& b){for(int c=0;c<6;++c){amrex::MultiFab::Copy(a.a[c],b.a[c],0,0,1,0);}}
        void setToZero(SpatialVector& a){for(auto& f:a.a){f.setVal(0.);}}
        void scale(SpatialVector& a,R b){for(auto& f:a.a){f.mult(b,0,1,0);}}
        void increment(SpatialVector& a,SpatialVector const& b,R alpha){for(int c=0;c<6;++c){amrex::MultiFab::Saxpy(a.a[c],alpha,b.a[c],0,0,1,0);}}
        void linComb(SpatialVector& a,R alpha,SpatialVector const& b,R beta,SpatialVector const& c){for(int d=0;d<6;++d){amrex::MultiFab::LinComb(a.a[d],alpha,b.a[d],0,beta,c.a[d],0,0,1,0);}}
        R dotProduct(SpatialVector const& a,SpatialVector const& b){return p.VectorDot(a,b);}
        R norm2(SpatialVector const& a){R const v=dotProduct(a,a);return std::isfinite(v)?std::sqrt(std::max(R(0),v)):v;}
        void precond(SpatialVector& a,SpatialVector const& b){assign(a,b);}
        void apply(SpatialVector& out,SpatialVector const& direction){
            R const n=norm2(direction);if(n==0.){setToZero(out);return;}
            R const h=1.e-5*std::max(R(1),norm2(base))/n;
            auto plus=p.Vector(),minus=p.Vector(),fp=p.Vector(),fm=p.Vector();
            linComb(plus,1.,base,h,direction);linComb(minus,1.,base,-h,direction);
            if(!p.Evaluate(plus,fp)||!p.Evaluate(minus,fm)){for(auto& f:out.a){f.setVal(std::numeric_limits<R>::quiet_NaN());}return;}
            linComb(out,1./(2*h),fp,-1./(2*h),fm);
        }
    };
    for(int iteration=0;iteration<32;++iteration){
        if(!Evaluate(x,f)){return Reject("event candidate inadmissible");}
        ledger.iterations=iteration;ledger.spatial_residual=VectorMax(f);
        if(ledger.spatial_residual<=1.e-12){return true;}
        Ops op{*this,x};FlexibleGMRES<SpatialVector,Ops> solver;solver.define(op);solver.setMaxIters(200);solver.setRestartLength(60);
        auto rhs=Vector(),correction=Vector();assign(rhs,f);op.scale(rhs,-1.);solver.solve(correction,rhs,1.e-5,0.);
        ledger.linear_iterations+=solver.getNumIters();if(solver.getStatus()!=0){return Reject("spatial event linear failure");}
        bool found=false;R const old=VectorMax(f);
        for(R length=1.;length>=1./1024.;length*=.5){
            auto trial=Vector(),next=Vector();op.linComb(trial,1.,x,length,correction);
            if(Evaluate(trial,next)&&VectorMax(next)<old){assign(x,trial);found=true;break;}
        }
        if(!found){return Reject("spatial event no descent");}
    }
    return Reject("spatial event maximum iterations");
}

bool NativeRZSpatialStoppingEvent::Impl::Work () {
    auto const& g=context.Geometry();
    std::vector<R> host(sums.size());amrex::Gpu::copy(amrex::Gpu::deviceToHost,sums.begin(),sums.end(),host.begin());
    amrex::ParallelDescriptor::ReduceRealSum(host.data(),host.size());
    ledger.ion_initial_energy=host[0];ledger.ion_energy_change=host[1];ledger.drag_energy_change=host[2];
    ledger.drag_bulk_work=host[3];ledger.electric_particle_work=host[4];ledger.endpoint_particle_work=host[5];
    ledger.heat=host[6];ledger.relativistic_defect=host[7];ledger.relativistic_bound=host[8];
    ledger.absolute_work=host[9];ledger.positive_momentum_scale=host[16]+
        volume*reference_rho*PhysConst::m_e/PhysConst::q_e*options.proper_speed_cap;
    for(int c=0;c<3;++c){ledger.ion_momentum_change[c]=host[10+c];ledger.reaction_momentum[c]=-host[13+c];}
    auto const b=Const(reaction_b);
    ledger.electric_momentum_transfer=0.;
    for(int c=0;c<3;++c){for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){auto const out=scratch[c].array(mfi);auto const ps=impulse[c].const_array(mfi),m=mass[c].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=PhysConst::m_e/PhysConst::q_e*ps(i,j,k)/m(i,j,k);});}
        if(c==2){ledger.electric_momentum_transfer=std::abs(host[19]-Integral(scratch[c],g));}}
    ledger.faraday_error=0.;ledger.displacement_curl=0.;R impulse_scale=0.,displacement_scale=0.;
    for(int c=0;c<3;++c){R const inverse=4./std::min(g.CellSize(0),g.CellSize(1));
        impulse_scale+=inverse*a1[c].norm0();displacement_scale+=inverse*d1[c].norm0();}
    ledger.faraday_bound=4096.*epsilon*impulse_scale;ledger.displacement_curl_bound=4096.*epsilon*displacement_scale;
    for(int channel=0;channel<2;++channel){for(auto& f:bcheck){f.setVal(0.);}
        auto out=Mutable(bcheck),input=channel==0?Mutable(a1):Mutable(d1);
        sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(out,input,sim.GetEBUpdateBFlag()[0],0);
        for(int c=0;c<3;++c){if(channel==0){amrex::MultiFab::Subtract(bcheck[c],b1[c],0,0,1,0);ledger.faraday_error=std::max(ledger.faraday_error,bcheck[c].norm0());}
            else{ledger.displacement_curl=std::max(ledger.displacement_curl,bcheck[c].norm0());}}
    }
    if(ledger.faraday_error>ledger.faraday_bound){return Reject("event Faraday gate");}
    if(ledger.displacement_curl>ledger.displacement_curl_bound){return Reject("event longitudinal displacement gate");}
    ledger.momentum_error=0.;ledger.momentum_bound=arithmetic_factor*ledger.positive_momentum_scale;
    for(int c=0;c<3;++c){
        amrex::MultiFab::LinComb(scratch[c],1.,j1[c],0,-1.,j0[c],0,0,1,0);
        ledger.electron_momentum_change[c]=-PhysConst::m_e/PhysConst::q_e*Integral(scratch[c],g);
        ledger.unrepresented_particle_momentum[c]=c<2?ledger.ion_momentum_change[c]:0.;
        if(c==2){ledger.axial_momentum_defect=ledger.ion_momentum_change[c]+ledger.electron_momentum_change[c];}
    }
    for(amrex::MFIter mfi(scratch[2]);mfi.isValid();++mfi){auto const a=scratch[2].array(mfi);
        auto const p=impulse[2].const_array(mfi),m=mass[2].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=PhysConst::m_e/PhysConst::q_e*p(i,j,k)/m(i,j,k);});}
    ledger.axial_electric_transfer=host[19]-Integral(scratch[2],g);
    ledger.axial_image_transfer=host[15]-Integral(*reaction->ConjugateMomentumDensity()[2],g);
    ledger.momentum_error=std::abs(ledger.axial_momentum_defect-ledger.axial_electric_transfer-ledger.axial_image_transfer);
    ledger.axis_reaction_conjugate=0.;ledger.axis_reaction_work=0.;
    for(int channel=0;channel<2;++channel){for(amrex::MFIter mfi(scratch[1]);mfi.isValid();++mfi){
        auto const a=scratch[1].array(mfi);auto const pi=reaction->ConjugateMomentumDensity()[1]->const_array(mfi),velocity=ve[1].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=i==0?pi(i,j,k)*(channel==0?1.:velocity(i,j,k)):0.;});}
        auto const value=Integral(scratch[1],g);if(channel==0){ledger.axis_reaction_conjugate=value;}else{ledger.axis_reaction_work=value;}}
    R gridwork=0.,trialb=0.;ledger.electron_initial_energy=0.;ledger.electron_energy_change=0.;
    ledger.constraint_work=0.;ledger.mean_current_work=0.;ledger.grid_residual=0.;
    for(int c=0;c<3;++c){
        for(int channel=0;channel<6;++channel){
            for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
                auto const a=scratch[c].array(mfi);auto const old=j0[c].const_array(mfi),next=j1[c].const_array(mfi);
                auto const trial=mean[c].const_array(mfi),m=mass[c].const_array(mfi),ri=b[c]->const_array(mfi);
                auto const ion0=i0[c].const_array(mfi),ion1=i1[c].const_array(mfi);
                auto const psi=impulse[c].const_array(mfi),dc=c1[c].const_array(mfi),dd=d1[c].const_array(mfi),jw=wall[c].const_array(mfi);
                auto const C0=background?c0[c].const_array(mfi):dc,D0=background?d0[c].const_array(mfi):dd,W0=background?w0[c].const_array(mfi):jw;
                bool const bg=background;
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    R const midpoint=.5*(old(i,j,k)+next(i,j,k));
                    if(channel==0){a(i,j,k)=.5*old(i,j,k)*m(i,j,k)*old(i,j,k);}
                    else if(channel==1){a(i,j,k)=midpoint*m(i,j,k)*(next(i,j,k)-old(i,j,k));}
                    else if(channel==2){a(i,j,k)=.5*(ion0(i,j,k)+ion1(i,j,k))*psi(i,j,k);}
                    else if(channel==3){
                        R const old_expression=.5*(ion0(i,j,k)+ion1(i,j,k))+midpoint+.5*dd(i,j,k)+.5*jw(i,j,k)-.5*dc(i,j,k);
                        a(i,j,k)=(bg?old_expression+D0(i,j,k)+W0(i,j,k)-C0(i,j,k):old_expression)*psi(i,j,k);
                    }
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
        amrex::MultiFab::Add(scratch[c],d1[c],0,0,1,0);amrex::MultiFab::Add(scratch[c],wall[c],0,0,1,0);amrex::MultiFab::Subtract(scratch[c],c1[c],0,0,1,0);
        if(background){
            amrex::MultiFab::Subtract(scratch[c],i0[c],0,0,1,0);
            amrex::MultiFab::Subtract(scratch[c],j0[c],0,0,1,0);
        }
        ledger.grid_residual=std::max(ledger.grid_residual,scratch[c].norm0());
        for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
            auto const a=scratch[c].array(mfi);auto const old=j0[c].const_array(mfi),next=j1[c].const_array(mfi),trial=mean[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=trial(i,j,k)-.5*(old(i,j,k)+next(i,j,k));});
        }
        ledger.grid_residual=std::max(ledger.grid_residual,scratch[c].norm0());
    }
    ledger.magnetic_energy_change=0.;ledger.curl_work=0.;ledger.displacement_work=0.;ledger.gauge_error=0.;ledger.magnetic_norm=0.;
    for(int c=0;c<3;++c){
        amrex::MultiFab term(b1[c].boxArray(),context.Distribution(),1,0);
        for(amrex::MFIter mfi(term);mfi.isValid();++mfi){auto const out=term.array(mfi);auto const x=b1[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=.5*x(i,j,k)*x(i,j,k)/PhysConst::mu0;});}
        R const self=Integral(term,g);ledger.magnetic_energy_change+=self;
        if(background){
            background_ledger.magnetic_self_work+=self;
            for(int channel=0;channel<3;++channel){
                for(amrex::MFIter mfi(term);mfi.isValid();++mfi){auto const out=term.array(mfi);auto const old=b0[c].const_array(mfi),x=b1[c].const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                        R const product=old(i,j,k)*x(i,j,k)/PhysConst::mu0;
                        out(i,j,k)=channel==0?product:(channel==1?std::abs(product):.5*old(i,j,k)*old(i,j,k)/PhysConst::mu0);
                    });}
                R const value=Integral(term,g);
                if(channel==0){background_ledger.magnetic_cross_work+=value;ledger.magnetic_energy_change+=value;}
                else if(channel==1){background_ledger.magnetic_cross_scale+=value;}
                else{ledger.magnetic_initial_energy+=value;}
            }
        }
ledger.magnetic_norm=std::max(ledger.magnetic_norm,b1[c].norm0());
        ledger.gauge_error=std::max(ledger.gauge_error,plpsi[c].norm0()/psi_scale);
        for(int channel=0;channel<2;++channel){for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){
            auto const out=scratch[c].array(mfi);auto const p=impulse[c].const_array(mfi),
                C=(circuit?circuit->total_delta[c]:c1[c]).const_array(mfi),D=d1[c].const_array(mfi);
            auto const C0=background?(circuit?circuit->total0[c]:c0[c]).const_array(mfi):C,D0=background?d0[c].const_array(mfi):D;bool const bg=background;
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                out(i,j,k)=bg?p(i,j,k)*(channel==0?C0(i,j,k)+.5*C(i,j,k):-D0(i,j,k)-.5*D(i,j,k)):
                    .5*p(i,j,k)*(channel==0?C(i,j,k):-D(i,j,k));});}
            auto const value=Integral(scratch[c],g);if(channel==0){ledger.curl_work+=value;}else{ledger.displacement_work+=value;}}
        ledger.electric_impulse[c]=Integral(impulse[c],g)/volume;
        amrex::MultiFab::LinComb(scratch[c],1.,mean[c],0,-1.,j0[c],0,0,1,0);
        ledger.mean_current_increment[c]=Integral(scratch[c],g)/volume;
    }
    ledger.curl_work+=ledger.magnetic_energy_change;
    ledger.conductor_work=0.;ledger.conductor_current_norm=0.;
    for(int c=0;c<3;++c){ledger.conductor_current_norm=std::max(ledger.conductor_current_norm,wall[c].norm0());
        for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){auto const a=scratch[c].array(mfi);
            auto const p=impulse[c].const_array(mfi),jw=wall[c].const_array(mfi);
            auto const W0=background?w0[c].const_array(mfi):jw;bool const bg=background;
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                a(i,j,k)=bg?-p(i,j,k)*(W0(i,j,k)+.5*jw(i,j,k)):-.5*p(i,j,k)*jw(i,j,k);});}
        ledger.conductor_work+=Integral(scratch[c],g);
    }
    ledger.spatial_transfer=ledger.endpoint_particle_work-gridwork;
    ledger.transpose_work=ledger.drag_bulk_work+trialb;
    if(!midpoint_thermal) {
        Copy(u1,u0);
#if defined(WARPX_DIM_RZ)
        amrex::MultiFab heat_cell(u1.boxArray(),u1.DistributionMap(),1,u1.nGrowVect());
        heat_map->RestrictNodalScalar(heat,0,heat_cell);
        amrex::MultiFab::Add(u1,heat_cell,0,0,1,0);
#endif
        u1.FillBoundary(g.periodicity());
    }
    amrex::MultiFab change(u1.boxArray(),u1.DistributionMap(),1,0);
    amrex::MultiFab::LinComb(change,1.,u1,0,-1.,u0,0,0,1,0);
    R const delivered=Integral(change,g);
    ledger.heat_transfer_error=delivered-ledger.heat;
    ledger.positive_energy_scale=ledger.ion_initial_energy+ledger.electron_initial_energy;
    // Preserve the particle/electron convention budget even with a large B0.
    ledger.actual_energy_defect=ledger.ion_energy_change+ledger.electron_energy_change+ledger.magnetic_energy_change+delivered;
    ledger.accounting_defect=ledger.actual_energy_defect-(ledger.relativistic_defect+
        ledger.spatial_transfer+ledger.constraint_work+ledger.mean_current_work+
        ledger.transpose_work+ledger.heat_transfer_error+ledger.curl_work+ledger.displacement_work+ledger.conductor_work);
    // Roundoff scales are positive input/storage/work magnitudes. The reported
    // signed defects never set a tolerance or get deposited as extra heat.
    R coefficient_scale=0.;
    for(int c=0;c<3;++c){coefficient_scale+=impulse[c].norm0()/mass[c].min(0)+c1[c].norm0();}
    ledger.grid_bound=arithmetic_factor*(current_scale+coefficient_scale);
    ledger.arithmetic_bound=arithmetic_factor*(ledger.positive_energy_scale+
        Integral(u0,g)+ledger.absolute_work+std::abs(ledger.electron_energy_change)+
        (background?ledger.magnetic_initial_energy+background_ledger.magnetic_self_work+background_ledger.magnetic_cross_scale:ledger.magnetic_energy_change));
    if(background){
        for(int c=0;c<3;++c){for(int channel=0;channel<3;++channel){
            for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){auto const out=scratch[c].array(mfi);
                auto const I=i0[c].const_array(mfi),J=j0[c].const_array(mfi),D=d0[c].const_array(mfi),W=w0[c].const_array(mfi),C=c0[c].const_array(mfi),P=impulse[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    out(i,j,k)=P(i,j,k)*(channel==0?I(i,j,k)+J(i,j,k)+D(i,j,k)+W(i,j,k)-C(i,j,k):(channel==1?-D(i,j,k):-W(i,j,k)));
                });}
            R const value=Integral(scratch[c],g);
            if(channel==0){background_ledger.initial_constraint_work+=value;}
            else if(channel==1){background_ledger.initial_displacement_work+=value;}
            else{background_ledger.initial_conductor_work+=value;}
        }}
        AssembleBackground();
    }
    if(ledger.gauge_error>1.e-12){return Reject("event transverse gauge gate");}
    if(ledger.momentum_error>ledger.momentum_bound){return Reject("event momentum gate");}
    if(circuit){
        if(!CircuitWork()||!NativeRZMidpointStoppingEvent::ValidateCircuitWork(ledger,circuit->work,options))
            return Reject("circuit event work/capability gate");
    }else if(!NativeRZSpatialStoppingEvent::ValidateLedger(ledger,options)){return Reject("event work/capability gate");}
    return true;
}


bool NativeRZSpatialStoppingEvent::Impl::CircuitWork() {
    auto& q=*circuit;auto& w=q.work;w={};
    if(!Collective(CircuitContextMatches()))return Reject("stale final circuit source work");
    w.generation=q.generation;w.action_evaluations=q.evaluations;
    w.source_time=q.binding.source_time;w.phase=q.binding.phase;
    std::array<double,1> chi{},scale{};
    // Only final receipt extraction copies these two scalar port operands.
    // No residual/Jv has a host field bounce or calls the provider.
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,q.port_impulse.begin(),q.port_impulse.end(),chi.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,q.field_increment.begin(),q.field_increment.end(),scale.begin());
    w.initial_port_current=q.initial_port_current;w.predicted_port_increment=q.reference_current*scale[0];
    w.port_impulse=chi[0];w.port_midpoint_work=(w.initial_port_current+.5*w.predicted_port_increment)*w.port_impulse;
    auto const& g=context.Geometry();
    for(int c=0;c<3;++c){
        amrex::MultiFab term(q.coil_b0[c].boxArray(),context.Distribution(),1,0);
        for(int channel=0;channel<4;++channel){
            for(amrex::MFIter mfi(term);mfi.isValid();++mfi){auto const out=term.array(mfi);
                auto const old=q.coil_b0[c].const_array(mfi),db=q.coil_b_delta[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    R const cross=old(i,j,k)*db(i,j,k)/PhysConst::mu0;
                    out(i,j,k)=channel==0?.5*old(i,j,k)*old(i,j,k)/PhysConst::mu0:
                        channel==1?.5*db(i,j,k)*db(i,j,k)/PhysConst::mu0:channel==2?cross:std::abs(cross);
                });}
            R const value=Integral(term,g);
            if(channel==0)w.coil_initial_energy+=value;
            else if(channel==1)w.coil_magnetic_self_work+=value;
            else if(channel==2)w.coil_magnetic_cross_work+=value;
            else w.positive_work_scale+=value;
        }
        for(int channel=0;channel<6;++channel){
            for(amrex::MFIter mfi(scratch[c]);mfi.isValid();++mfi){auto const out=scratch[c].array(mfi);
                auto const psi=impulse[c].const_array(mfi),ac=q.coil_a_delta[c].const_array(mfi);
                auto const j0c=q.coil_j0[c].const_array(mfi),dj=q.coil_j_delta[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    R const jbar=j0c(i,j,k)+.5*dj(i,j,k);
                    int const part=channel%3;
                    R const value=(part==0?psi(i,j,k):part==1?-ac(i,j,k):psi(i,j,k)+ac(i,j,k))*jbar;
                    out(i,j,k)=channel<3?value:std::abs(value);
                });}
            R const value=Integral(scratch[c],g);
            if(channel==0)w.total_coil_current_work+=value;
            else if(channel==1)w.coil_impulse_current_work+=value;
            else if(channel==2)w.response_coil_current_work+=value;
            else w.positive_work_scale+=value;
        }
    }
    w.coil_magnetic_change=w.coil_magnetic_self_work+w.coil_magnetic_cross_work;
    w.coil_curl_work=w.coil_magnetic_change+w.coil_impulse_current_work;
    w.reciprocity_work=-w.response_coil_current_work-w.port_midpoint_work;
    w.wall_work=ledger.conductor_work;
    w.inventory_without_network=ledger.actual_energy_defect-w.coil_magnetic_change;
    R const old_named=ledger.relativistic_defect+ledger.spatial_transfer+ledger.constraint_work+
        ledger.mean_current_work+ledger.transpose_work+ledger.heat_transfer_error+ledger.curl_work+
        ledger.displacement_work+ledger.conductor_work;
    w.named_work_without_network=old_named-w.coil_curl_work+w.port_midpoint_work+w.reciprocity_work;
    w.accounting_defect=w.inventory_without_network-w.named_work_without_network;
    w.positive_work_scale+=w.coil_initial_energy+w.coil_magnetic_self_work+
        std::abs(w.initial_port_current*w.port_impulse)+.5*std::abs(w.predicted_port_increment*w.port_impulse);
    w.arithmetic_bound=ledger.arithmetic_bound+arithmetic_factor*w.positive_work_scale;
    w.available=true;
    return true;
}

bool NativeRZMidpointStoppingEvent::ValidateCircuitWork(RZSpatialStoppingLedger const& l,
    NativeStoppingCircuitWork const& w,SpatialStoppingOptions const& o) {
    R const positive=l.positive_energy_scale,bound=l.arithmetic_bound;
    bool finite=true;
    for(R const value:{w.source_time,w.initial_port_current,w.predicted_port_increment,w.port_impulse,
        w.port_midpoint_work,w.coil_initial_energy,w.coil_magnetic_self_work,w.coil_magnetic_cross_work,
        w.coil_magnetic_change,w.total_coil_current_work,w.coil_impulse_current_work,w.response_coil_current_work,
        w.coil_curl_work,w.reciprocity_work,w.wall_work,w.inventory_without_network,w.named_work_without_network,
        w.accounting_defect,w.positive_work_scale,w.arithmetic_bound})finite=finite&&std::isfinite(value);
    // Keep the physical source and thermal gates distinct from finite external
    // work. No finite reciprocity/wall term is reclassified as a roundoff bound.
    return w.available&&!w.network_change_available&&!w.publication_available&&w.action_evaluations>0&&finite&&
        w.wall_work==l.conductor_work&&w.positive_work_scale>=0.&&w.arithmetic_bound>=bound&&
        std::abs(w.accounting_defect)<=w.arithmetic_bound&&
        std::abs(l.accounting_defect+w.total_coil_current_work)<=w.arithmetic_bound&&
        std::isfinite(positive)&&positive>0.&&std::isfinite(bound)&&bound>=0.&&
        std::isfinite(l.actual_energy_defect)&&std::isfinite(l.accounting_defect)&&
        std::isfinite(l.conductor_current_norm)&&std::isfinite(l.axis_reaction_conjugate)&&
        std::isfinite(l.axial_momentum_defect)&&std::isfinite(l.axial_electric_transfer)&&std::isfinite(l.axial_image_transfer)&&
        std::isfinite(l.conductor_work)&&l.axis_reaction_work==0.&&
        l.heat>=0.&&l.relativistic_bound>=0.&&
        l.relativistic_bound<=o.relative_convention_budget*positive&&
        std::abs(l.relativistic_defect)<=l.relativistic_bound+bound&&
        std::abs(l.heat+l.drag_energy_change-l.drag_bulk_work)<=bound&&
        l.spatial_residual<=1.e-12&&l.grid_residual<=l.grid_bound&&
        l.uniform_error<=l.uniform_bound&&l.momentum_error<=l.momentum_bound&&
        l.faraday_error<=l.faraday_bound&&l.displacement_curl<=l.displacement_curl_bound&&
        l.range_ratio<=1.&&std::abs(l.current_range_mean)<=l.current_range_bound&&
        std::abs(l.impulse_range_mean)<=l.impulse_range_bound;
}

NativeRZSpatialStoppingEvent::NativeRZSpatialStoppingEvent(WarpX& w,NativeAcceptedStoppingContext const& c,
    amrex::MultiFab const& t,amrex::MultiFab& u,SpatialStoppingOptions const& o,SpatialStoppingFields const& f)
    :m_impl(std::make_unique<Impl>(w,c,t,u,o,f)){}
NativeRZSpatialStoppingEvent::NativeRZSpatialStoppingEvent(WarpX& w,NativeAcceptedStoppingContext const& c,
    amrex::MultiFab const& t,amrex::MultiFab& u,SpatialStoppingOptions const& o,SpatialStoppingFields const& f,
    RZStoppingBackground const& bg):m_impl(std::make_unique<Impl>(w,c,t,u,o,f)){
    m_impl->background=true;m_impl->background_destinations=bg;
}
NativeRZSpatialStoppingEvent::~NativeRZSpatialStoppingEvent()=default;
bool NativeRZSpatialStoppingEvent::Prepare () {
    auto& p=*m_impl;
    if(p.committed||p.closed){return p.Reject("event transaction closed");}
    p.ready=false;p.ledger={};p.background_ledger={};
    if(!Collective(p.Scope())){return p.Reject("unsupported spatial stopping scope");}
    if(!p.Capture()||!p.Solve()||!p.Work()){return false;}
    if(!p.Unchanged(false)){return p.Reject("accepted state changed during preparation");}
    p.ready=true;p.failure="none";return true;
}
bool NativeRZSpatialStoppingEvent::Impl::Publish(FV const& current,
    NativeRZSpatialStoppingEvent::InvalidationHook hook,bool owned_circuit){
    // A bare helper cannot publish circuit state. The retained candidate alone
    // owns the complete rollback required after any provisional provider call.
    bool valid=ready&&!committed&&!closed&&hook.function&&
        (circuit ? owned_circuit&&CircuitFinalActionMatches() : !owned_circuit);
    for(int c=0;c<3;++c){valid=valid&&current[c]==binding.electron_current[c];}
    if(!Collective(valid)||!Collective(Scope())||!Unchanged(false)){failure="stale or duplicate event commit";return false;}
    if(circuit){
        auto& q=*circuit;auto& publication=q.publication;auto& coupler=*sim.get_pointer_CircuitCoupling()->Coupler();
        bool const cancellable=coupler.RetainedNativeStepCancelable();
        if(!Collective(cancellable&&NativeRZMidpointStoppingEvent::ValidateCircuitWork(ledger,q.work,options)))
            return Reject("circuit source publication lost original retained owner or work receipt");
        publication.application=q.application;publication.provider_attempted=true;
        std::string error;
        bool const stored=coupler.CommitNativeSourceImpulse(publication.provider,error);
        if(!Collective(stored))return Reject("native source provider publication failed; cancel whole step");
        publication.provider_committed=true;
        auto const& r=publication.provider;auto const& work=q.work;
        publication.network_change=r.coil_energy_change+r.local_inductor_energy_change+r.capacitor_energy_change;
        publication.port_rounding_work=r.port_work+work.port_midpoint_work;
        publication.inventory_change=work.inventory_without_network+publication.network_change;
        publication.named_work=work.named_work_without_network-work.port_midpoint_work+
            r.energy_defect+publication.port_rounding_work;
        publication.accounting_defect=publication.inventory_change-publication.named_work;
        // Six signed sums and their final subtraction. The positive operands
        // bound representation error; the measured defect never sets its bound.
        R const positive=std::abs(r.coil_energy_change)+std::abs(r.local_inductor_energy_change)+
            std::abs(r.capacitor_energy_change)+std::abs(r.port_work)+
            2.*std::abs(work.port_midpoint_work)+std::abs(work.inventory_without_network)+
            std::abs(work.named_work_without_network)+std::abs(r.energy_defect);
        publication.arithmetic_bound=work.arithmetic_bound+r.arithmetic_bound+Roundoff(16)*positive;
        bool finite=true;
        for(R x:{publication.network_change,publication.port_rounding_work,publication.inventory_change,
            publication.named_work,publication.accounting_defect,publication.arithmetic_bound,positive})
            finite=finite&&std::isfinite(x);
        bool const receipt=finite&&publication.arithmetic_bound>=0.&&
            std::abs(publication.accounting_defect)<=publication.arithmetic_bound&&
            r.time_sim==q.binding.source_time&&r.token!=0&&r.transaction_token!=0&&
            r.phase==static_cast<std::uint32_t>(q.binding.phase==NativeStoppingCircuitBinding::Phase::PreField?
                CircuitCoupler::SourceImpulsePhase::PreField:CircuitCoupler::SourceImpulsePhase::PostField);
        if(!Collective(receipt))return Reject("joined source work receipt failed; cancel whole step");
    }
    for(auto& s:species){if(s.charge==0.){continue;}for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
        auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());auto const* points=s.points.at(key).data();
        auto* ux=pti.GetAttribs(PIdx::ux).data();auto* uy=pti.GetAttribs(PIdx::uy).data();auto* uz=pti.GetAttribs(PIdx::uz).data();
        amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long n){ux[n]=points[n].next[0];uy[n]=points[n].next[1];uz[n]=points[n].next[2];});
    }}
    for(int c=0;c<3;++c){Copy(*current[c],j1[c]);Copy(*destinations.potential[c],background?full_a[c]:a1[c]);
        Copy(*destinations.magnetic[c],background?full_b[c]:b1[c]);Copy(*destinations.displacement[c],background?full_d[c]:d1[c]);
        if(background){Copy(*background_destinations.conductor_current[c],full_w[c]);}
    }Copy(energy,u1);
    amrex::Gpu::synchronize();committed=true;ready=false;published=current;hook.function(hook.context);
    if(circuit){
        auto& q=*circuit;q.publication.fields_published=true;
        auto& coupler=*sim.get_pointer_CircuitCoupling()->Coupler();std::string error;
        bool const rebased=coupler.RebaseNativeSourceLinkage(error);
        if(!Collective(rebased))return Reject("source linkage rebase failed; cancel whole step");
        q.publication.linkage_rebased=true;
    }
    return true;
}
bool NativeRZSpatialStoppingEvent::CommitOnce(FV const& current,InvalidationHook hook){return m_impl->Publish(current,hook);}
namespace {
bool RollbackRZEvent(NativeRZSpatialStoppingEvent::Impl& p,
    NativeRZSpatialStoppingEvent::InvalidationHook hook){
    // Circuit rollback belongs to the original retained macrostep, including
    // its provider, scale segments, linkage, particles and all field histories.
    if(!Collective(!p.circuit&&p.committed&&!p.closed&&hook.function)||!p.Unchanged(true)){p.failure="stale event rollback or circuit requires whole-step cancellation";return false;}
    for(auto& s:p.species){if(s.charge==0.){continue;}for(WarpXParIter pti(*s.pc,0);pti.isValid();++pti){
        auto const key=std::make_pair(pti.index(),pti.LocalTileIndex());auto const& soa=s.old.at(key).GetStructOfArrays();
        for(int c:{PIdx::ux,PIdx::uy,PIdx::uz}){auto const& source=soa.GetRealData(c);auto& target=pti.GetAttribs(c);
            amrex::Gpu::copy(amrex::Gpu::deviceToDevice,source.begin(),source.end(),target.begin());}
    }}
    for(int c=0;c<3;++c){Copy(*p.published[c],p.j0[c]);Copy(*p.destinations.potential[c],p.a0[c]);Copy(*p.destinations.magnetic[c],p.b0[c]);Copy(*p.destinations.displacement[c],p.d0[c]);
        if(p.background){Copy(*p.background_destinations.conductor_current[c],p.w0[c]);}
    }
    Copy(p.energy,p.u0);amrex::Gpu::synchronize();p.committed=false;p.closed=true;hook.function(hook.context);return true;
}
} // namespace
bool NativeRZSpatialStoppingEvent::Rollback(InvalidationHook hook){return RollbackRZEvent(*m_impl,hook);}
void NativeRZSpatialStoppingEvent::Finalize(){m_impl->closed=true;m_impl->ready=false;}
bool NativeRZSpatialStoppingEvent::Prepared()const{return m_impl->ready;}
char const* NativeRZSpatialStoppingEvent::Failure()const{return m_impl->failure;}
RZSpatialStoppingLedger const& NativeRZSpatialStoppingEvent::Ledger()const{return m_impl->ledger;}
CV NativeRZSpatialStoppingEvent::CandidateElectronCurrent()const{return Const(m_impl->j1);}
amrex::MultiFab const& NativeRZSpatialStoppingEvent::CandidateEnergy()const{return m_impl->u1;}
CV NativeRZSpatialStoppingEvent::CandidateMagnetic()const{return Const(m_impl->background?m_impl->full_b:m_impl->b1);}
CV NativeRZSpatialStoppingEvent::CandidatePotential()const{return Const(m_impl->background?m_impl->full_a:m_impl->a1);}
CV NativeRZSpatialStoppingEvent::CandidateDisplacement()const{return Const(m_impl->background?m_impl->full_d:m_impl->d1);}
CV NativeRZSpatialStoppingEvent::ElectricImpulse()const{return Const(m_impl->impulse);}
CV NativeRZSpatialStoppingEvent::CandidateConductorCurrent()const{return Const(m_impl->background?m_impl->full_w:m_impl->wall);}
CV NativeRZSpatialStoppingEvent::MeanElectronCurrent()const{return Const(m_impl->mean);}
CV NativeRZSpatialStoppingEvent::StoppingVelocity()const{return Const(m_impl->ve);}
CV NativeRZSpatialStoppingEvent::StoppingImpulseDensity()const{return m_impl->reaction->ConjugateMomentumDensity();}
CV NativeRZSpatialStoppingEvent::ElectricAuxiliary()const{return Const(m_impl->aux);}
bool NativeRZSpatialStoppingEvent::AppendDisplacementCertificate(
    warpx::darwin::NativeLongitudinalProducerCertificate const& old,
    warpx::darwin::NativeLongitudinalProducerCertificate& updated)const
{
    auto const& s=*m_impl;
    bool supported=s.committed&&!s.closed&&s.displacement_potential&&
        s.background_destinations.longitudinal_certificate.get()==&old&&&old==&updated;
    amrex::ParallelDescriptor::ReduceBoolAnd(supported);
    if(!supported)return false;
    return updated.AppendEvent(*s.displacement_potential,Const(s.d1));
}
RZStoppingBackgroundLedger const& NativeRZSpatialStoppingEvent::BackgroundLedger()const{return m_impl->background_ledger;}
CV NativeRZSpatialStoppingEvent::PotentialIncrement()const{return Const(m_impl->a1);}
CV NativeRZSpatialStoppingEvent::MagneticIncrement()const{return Const(m_impl->b1);}
CV NativeRZSpatialStoppingEvent::DisplacementIncrement()const{return Const(m_impl->d1);}
CV NativeRZSpatialStoppingEvent::ConductorIncrement()const{return Const(m_impl->wall);}
bool NativeRZSpatialStoppingEvent::ValidateLedger(RZSpatialStoppingLedger const& l,SpatialStoppingOptions const& o){
    R const positive=l.positive_energy_scale,bound=l.arithmetic_bound;
    return std::isfinite(positive)&&positive>0.&&std::isfinite(bound)&&bound>=0.&&
        std::isfinite(l.actual_energy_defect)&&std::isfinite(l.accounting_defect)&&
        std::isfinite(l.conductor_current_norm)&&std::isfinite(l.axis_reaction_conjugate)&&
        std::isfinite(l.axial_momentum_defect)&&std::isfinite(l.axial_electric_transfer)&&std::isfinite(l.axial_image_transfer)&&
        l.conductor_work==0.&&l.axis_reaction_work==0.&&
        l.heat>=0.&&l.relativistic_bound>=0.&&
        l.relativistic_bound<=o.relative_convention_budget*positive&&
        std::abs(l.relativistic_defect)<=l.relativistic_bound+bound&&
        std::abs(l.accounting_defect)<=bound&&
        std::abs(l.heat+l.drag_energy_change-l.drag_bulk_work)<=bound&&
        l.spatial_residual<=1.e-12&&l.grid_residual<=l.grid_bound&&
        l.uniform_error<=l.uniform_bound&&l.momentum_error<=l.momentum_bound&&
        l.faraday_error<=l.faraday_bound&&l.displacement_curl<=l.displacement_curl_bound&&
        l.range_ratio<=1.&&std::abs(l.current_range_mean)<=l.current_range_bound&&
        std::abs(l.impulse_range_mean)<=l.impulse_range_bound;
}
#include "NativeRZMidpointTemperaturePC.H"
#include "NativeRZMidpointStoppingEventImpl.H"
} // namespace warpx::thermal
