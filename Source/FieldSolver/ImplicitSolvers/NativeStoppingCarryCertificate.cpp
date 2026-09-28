/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeStoppingCarryCertificate.H"
#include "DarwinVacuumJointSolve.H"
#include "DarwinThermalAdvance.H"
#include "NativeEndpointField.H"
#include "NativeInstantaneousIonCurrent.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "NativeInertiaSupport.H"
#include "NativeVacuumConstraint.H"
#include "KineticThermalMoments.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "WarpXSolverVec.H"
#include "WarpXSolverDOF.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_Reduce.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_iMultiFab.H>
#include <cmath>
#include <limits>
namespace warpx::thermal {
namespace {
using R=amrex::Real;using MF=amrex::MultiFab;using IM=amrex::iMultiFab;
using Field=std::array<MF,3>;using CV=ablastr::fields::ConstVectorField;
using FT=warpx::fields::FieldType;using Dir=ablastr::fields::Direction;
constexpr char const* Je="diagnostic_Je_endpoint_fp";
constexpr char const* Stage="diagnostic_Je_stage_fp";
constexpr char const* Corrected="diagnostic_Je_inertia_stage_fp";
constexpr char const* IonOld="diagnostic_Ji_old_instantaneous_fp";
constexpr char const* IonVirtual="diagnostic_Ji_virtual_instantaneous_fp";
bool All(bool b){amrex::ParallelDescriptor::ReduceBoolAnd(b);return b;}
CV View(Field const& f){return{&f[0],&f[1],&f[2]};}
void Clone(MF& a,MF const& b){a.define(b.boxArray(),b.DistributionMap(),b.nComp(),b.nGrowVect());MF::Copy(a,b,0,0,b.nComp(),b.nGrowVect());}
void Clone(IM& a,IM const& b){a.define(b.boxArray(),b.DistributionMap(),b.nComp(),b.nGrowVect());IM::Copy(a,b,0,0,b.nComp(),b.nGrowVect());}
bool Layout(MF const& a,MF const& b){return a.isDefined()&&b.isDefined()&&a.boxArray()==b.boxArray()&&a.DistributionMap()==b.DistributionMap()&&a.nComp()==b.nComp()&&a.nGrowVect()==b.nGrowVect();}
bool Same(MF const& a,MF const& b){
    if(!All(Layout(a,b)))return false;
    amrex::Gpu::DeviceScalar<amrex::Long> mismatch(0);auto* bad=mismatch.dataPtr();
    for(amrex::MFIter it(a);it.isValid();++it){
        auto const* x=reinterpret_cast<unsigned char const*>(a[it].dataPtr());
        auto const* y=reinterpret_cast<unsigned char const*>(b[it].dataPtr());auto n=a[it].nBytes();
        amrex::For(n,[=]AMREX_GPU_DEVICE(amrex::Long k){if(x[k]!=y[k])amrex::HostDevice::Atomic::Add(bad,amrex::Long(1));});
    }
    return All(mismatch.dataValue()==0);
}
bool SameValid(MF const& a,MF const& b){
    bool const layout=a.isDefined()&&b.isDefined()&&a.boxArray()==b.boxArray()&&
        a.DistributionMap()==b.DistributionMap()&&a.nComp()==b.nComp();
    if(!All(layout))return false;
    amrex::Gpu::DeviceScalar<int> mismatch(0);auto* bad=mismatch.dataPtr();
    for(amrex::MFIter it(a);it.isValid();++it){auto x=a.const_array(it),y=b.const_array(it);
        amrex::For(it.validbox(),a.nComp(),[=]AMREX_GPU_DEVICE(int i,int j,int k,int c){
            auto const* px=reinterpret_cast<unsigned char const*>(&x(i,j,k,c));
            auto const* py=reinterpret_cast<unsigned char const*>(&y(i,j,k,c));
            bool differs=false;for(std::size_t n=0;n<sizeof(R);++n)differs=differs||(px[n]!=py[n]);
            if(differs)amrex::Gpu::Atomic::Exch(bad,1);
        });
    }
    return All(mismatch.dataValue()==0);
}
bool SameOptions(warpx::darwin::NativeInertiaSupportOptions const& a,
                 warpx::darwin::NativeInertiaSupportOptions const& b){
    return a.lower==b.lower&&a.upper==b.upper&&a.coefficient==b.coefficient&&
        a.recovery==b.recovery&&a.components==b.components&&a.charge_floor==b.charge_floor&&
        a.reference_charge_density==b.reference_charge_density&&
        a.recovery_density_fraction==b.recovery_density_fraction;
}
R Up(R x){return std::nextafter(x,std::numeric_limits<R>::infinity());}
// Only the transition metric receipt uses this conservative arithmetic regime.
// It rejects underflow before a later large factor could amplify a lost product.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
R MetricProduct(R a,R b,bool& invalid){
    R const p=a*b;
    invalid=invalid||!std::isfinite(p)||
        (a!=0.&&b!=0.&&std::abs(p)<std::numeric_limits<R>::min());
    return p;
}

bool Scope(WarpX const& w){
#if defined(WARPX_DIM_RZ)
    return w.finestLevel()==0&&w.Geom(0).isPeriodic(1)&&!w.Geom(0).isPeriodic(0)&&
        WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC&&
        w.Geom(0).ProbLo(0)==0.&&NativeIonQuadratureEnabled()&&NativeCorrelatedIncrementEnabled();
#else
    amrex::ignore_unused(w);return false;
#endif
}
}
struct NativeStoppingCarryCertificate::Impl {
    WarpX& w;amrex::Geometry geometry;amrex::BoxArray cells;amrex::DistributionMapping distribution;
    R time=0.,dt=0.,endpoint_time=0.;int step=0,native_coord=0;std::uint64_t epoch=0;
    StoppingCarryReport report;
    Field root_e,root_j,root_jp,root_ion,old,stage,endpoint,material,carry;
    std::array<IM,3> vacuum,norm_mask,physical;
    MF rho;
    struct Transition {
        std::array<Field,3> mass;
        std::array<MF,3> density,mask_density;
        std::array<IM,3> field_physical,mask;
        Field origin_current;
        struct AmperePhase { Field potential,magnetic,ion,current,displacement;bool ready=false; };
        std::array<AmperePhase,3> ampere;
        warpx::darwin::NativeInertiaSupportOptions options;
        R electron_mass=0.,reference_number_density=0.;
        std::array<bool,3> metric_ready{false,false,false};
        explicit Transition(WarpX& w):options(warpx::darwin::NativeVacuumSupportOptions(w)),
            electron_mass(w.get_pointer_HybridPICModel()->m_electron_inertia_mass),
            reference_number_density(w.get_pointer_HybridPICModel()->m_n0_ref){}
        bool ConfigurationMatches(WarpX& w)const {
            return SameOptions(options,warpx::darwin::NativeVacuumSupportOptions(w))&&
                electron_mass==w.get_pointer_HybridPICModel()->m_electron_inertia_mass&&
                reference_number_density==w.get_pointer_HybridPICModel()->m_n0_ref;
        }
        bool CaptureAmpere(WarpX& w,int phase){
            bool valid=phase>=0&&phase<3&&!ampere[phase].ready&&ConfigurationMatches(w);
            char const* displacement=phase==1?"diagnostic_D_stage_fp":"diagnostic_D_endpoint_fp";
            for(int c=0;c<3;++c)for(auto const* name:{"hybrid_A_fp","Bfield_fp",displacement})
                valid=w.m_fields.has(name,Dir{c},0)&&valid;
            if(phase==1)for(int c=0;c<3;++c)valid=w.m_fields.has(IonVirtual,Dir{c},0)&&valid;
            if(!All(valid))return false;
            auto& a=ampere[phase];
            for(int c=0;c<3;++c){
                Clone(a.potential[c],*w.m_fields.get("hybrid_A_fp",Dir{c},0));
                Clone(a.magnetic[c],*w.m_fields.get(FT::Bfield_fp,Dir{c},0));
                Clone(a.displacement[c],*w.m_fields.get(displacement,Dir{c},0));
                auto const& like=*w.m_fields.get(FT::current_fp,Dir{c},0);
                a.ion[c].define(like.boxArray(),like.DistributionMap(),1,like.nGrowVect());a.ion[c].setVal(0.);
                a.current[c].define(like.boxArray(),like.DistributionMap(),1,1);a.current[c].setVal(0.);
                if(phase==1)MF::Copy(a.ion[c],*w.m_fields.get(IonVirtual,Dir{c},0),0,0,1,
                    amrex::min(a.ion[c].nGrowVect(),w.m_fields.get(IonVirtual,Dir{c},0)->nGrowVect()));
            }
            if(phase!=1&&!warpx::particles::DepositNativeInstantaneousIonCurrent(w,
                warpx::particles::InstantaneousIonState::Current,{&a.ion[0],&a.ion[1],&a.ion[2]}))return false;
            ablastr::fields::VectorField current{&a.current[0],&a.current[1],&a.current[2]},
                magnetic{&a.magnetic[0],&a.magnetic[1],&a.magnetic[2]};
            w.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(current,magnetic,w.GetEBUpdateEFlag()[0],0);
            bool finite=true;
            for(int c=0;c<3;++c){a.current[c].OverrideSync(w.Geom(0).periodicity());a.current[c].FillBoundary(w.Geom(0).periodicity());
                for(auto const* f:{&a.potential[c],&a.magnetic[c],&a.ion[c],&a.current[c],&a.displacement[c]})
                    finite=f->is_finite(0,1,0)&&finite;}
            if(!All(finite))return false;a.ready=true;return true;
        }
        bool CaptureMetric(WarpX& w,int phase,MF const& raw,int component){
            bool valid=phase>=0&&phase<3&&component>=0&&component<raw.nComp()&&
                !metric_ready[phase]&&ConfigurationMatches(w)&&std::isfinite(electron_mass)&&
                electron_mass>0.&&std::isfinite(reference_number_density)&&reference_number_density>0.&&
                w.m_fields.has("hybrid_rho_vacmask_fp",0);
            if(!All(valid))return false;
            MF physical(raw,amrex::make_alias,component,1);
            auto const& masks=*w.m_fields.get("hybrid_rho_vacmask_fp",0);
            Clone(density[phase],physical);Clone(mask_density[phase],masks);
            warpx::darwin::NativeInertiaSupport support(
                w.get_pointer_HybridPICModel()->ElectronThermalGeometry(),
                w.boxArray(0),w.DistributionMap(0),options);
            if(!support.FreezeEdges(density[phase],mask_density[phase]))return false;
            Field unit;
            for(int c=0;c<3;++c){auto const& like=support.KappaEdge(c);
                unit[c].define(like.boxArray(),like.DistributionMap(),1,0);unit[c].setVal(1.);
                mass[phase][c].define(like.boxArray(),like.DistributionMap(),1,0);}
            R const scale=electron_mass/(PhysConst::q_e*PhysConst::q_e*reference_number_density);
            if(!All(std::isfinite(scale)&&scale>0.))return false;
            support.ApplyMass({&unit[0],&unit[1],&unit[2]},
                {&mass[phase][0],&mass[phase][1],&mass[phase][2]},scale);
            bool finite=true;for(auto const& x:mass[phase])finite=x.is_finite()&&finite;
            if(!All(finite))return false;
            metric_ready[phase]=true;return true;
        }
    };
    std::unique_ptr<Transition> transition;
    explicit Impl(WarpX& sim,NativeStoppingCarryMode mode):w(sim),geometry(sim.get_pointer_HybridPICModel()->ElectronThermalGeometry()),cells(sim.boxArray(0)),distribution(sim.DistributionMap(0)),native_coord(sim.Geom(0).Coord()){report.mode=mode;}
    bool Ready()const{return Phase(StoppingCarryStatus::Ready)||Phase(StoppingCarryStatus::TransitionReady)||Phase(StoppingCarryStatus::AcceptedReady);}

    bool MeasureAmpereHistory(){
        if(!All(transition&&transition->ampere[0].ready&&transition->ampere[1].ready&&transition->ampere[2].ready))return false;
        auto const& phases=transition->ampere;
        R peaks[9]={};amrex::Long rows=0;int invalid=0;
        for(int c=0;c<3;++c){auto owner=endpoint[c].OwnerMask(geometry.periodicity());
            amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,
                amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,amrex::ReduceOpMax,
                amrex::ReduceOpMax,amrex::ReduceOpSum,amrex::ReduceOpMax> op;
            amrex::ReduceData<R,R,R,R,R,R,R,R,R,amrex::Long,int> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(endpoint[c]);it.isValid();++it){auto own=owner->const_array(it);
                auto c0=phases[0].current[c].const_array(it),ct=phases[1].current[c].const_array(it),c1=phases[2].current[c].const_array(it);
                auto i0=phases[0].ion[c].const_array(it),i1=phases[2].ion[c].const_array(it);
                auto d0=phases[0].displacement[c].const_array(it),dt=phases[1].displacement[c].const_array(it),d1=phases[2].displacement[c].const_array(it);
                auto j0=old[c].const_array(it),jt=stage[c].const_array(it),j1=endpoint[c].const_array(it);
                op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{
                    if(!own(i,j,k))return{0.,0.,0.,0.,0.,0.,0.,0.,0.,0,0};
                    R const e0=c0(i,j,k)-i0(i,j,k)-j0(i,j,k)-d0(i,j,k);
                    R const e1=c1(i,j,k)-i1(i,j,k)-j1(i,j,k)-d1(i,j,k);
                    R const dc=ct(i,j,k)-.5*(c0(i,j,k)+c1(i,j,k));
                    R const dq=jt(i,j,k)-(ct(i,j,k)-dt(i,j,k)-.5*(i0(i,j,k)+i1(i,j,k)));
                    R const dj=j1(i,j,k)-(2.*jt(i,j,k)-j0(i,j,k));
                    R const dd=d1(i,j,k)-(2.*dt(i,j,k)-d0(i,j,k));
                    R const scale=std::abs(c0(i,j,k))+std::abs(ct(i,j,k))+std::abs(c1(i,j,k))+
                        std::abs(i0(i,j,k))+std::abs(i1(i,j,k))+std::abs(j0(i,j,k))+std::abs(jt(i,j,k))+
                        std::abs(j1(i,j,k))+std::abs(d0(i,j,k))+std::abs(dt(i,j,k))+std::abs(d1(i,j,k));
                    R const error=std::abs(e1-(-e0-2.*dc-2.*dq-dj-dd));
                    R const gamma=128.*std::numeric_limits<R>::epsilon()/(1.-128.*std::numeric_limits<R>::epsilon());
                    R const bound=gamma*scale+128.*std::numeric_limits<R>::denorm_min();
                    bool const bad=!std::isfinite(scale)||!std::isfinite(error)||!std::isfinite(bound)||error>bound;
                    return{std::abs(e0),std::abs(e1),std::abs(dc),std::abs(dq),std::abs(dj),std::abs(dd),error,bound,scale,1,int(bad)};
                });}
            auto v=data.value();R p[9]={amrex::get<0>(v),amrex::get<1>(v),amrex::get<2>(v),amrex::get<3>(v),amrex::get<4>(v),amrex::get<5>(v),amrex::get<6>(v),amrex::get<7>(v),amrex::get<8>(v)};
            for(int z=0;z<9;++z)peaks[z]=std::max(peaks[z],p[z]);rows+=amrex::get<9>(v);invalid=std::max(invalid,amrex::get<10>(v));
        }
        amrex::ParallelDescriptor::ReduceRealMax(peaks,9);amrex::ParallelDescriptor::ReduceLongSum(rows);amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if(invalid)return false;auto& r=report.ampere_history;r.available=true;r.rows=rows;
        r.initial_defect=peaks[0];r.endpoint_defect=peaks[1];r.curl_centering=peaks[2];r.stage_quadrature=peaks[3];
        r.current_rotation=peaks[4];r.displacement_rotation=peaks[5];r.identity_error=peaks[6];r.identity_bound=peaks[7];r.physical_scale=peaks[8];return true;
    }
    bool MeasureMovingMetric(){
        if(!All(transition&&transition->metric_ready[0]&&transition->metric_ready[1]&&transition->metric_ready[2]))return false;
        R values[8]={};amrex::Long rows=0;int invalid=0;
        for(int c=0;c<3;++c){auto owner=endpoint[c].OwnerMask(geometry.periodicity());
            auto volume=MakeQdsmcVolumeElement(geometry,endpoint[c].ixType());
            amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
                amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,
                amrex::ReduceOpSum,amrex::ReduceOpMax> op;
            amrex::ReduceData<R,R,R,R,R,R,R,R,amrex::Long,int> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(endpoint[c]);it.isValid();++it){auto j0=old[c].const_array(it),jt=stage[c].const_array(it),j1=endpoint[c].const_array(it);
                auto m0=transition->mass[0][c].const_array(it),mt=transition->mass[1][c].const_array(it),m1=transition->mass[2][c].const_array(it);
                auto own=owner->const_array(it),t=transition->mask[c].const_array(it);
                op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{
                    if(!own(i,j,k))return{0.,0.,0.,0.,0.,0.,0.,0.,0,0};
                    R const a=j0(i,j,k),b=j1(i,j,k),mid=jt(i,j,k),v=volume(i,j,k);
                    R const x=m0(i,j,k),y=mt(i,j,k),z=m1(i,j,k);
                    bool bad=x<0.||y<0.||z<0.||!(v>0.);
                    R const aa=MetricProduct(a,a,bad);
                    R const bb=MetricProduct(b,b,bad);
                    R const mean=MetricProduct(.5,a+b,bad),delta=b-a;
                    R const half_x=MetricProduct(.5,x,bad);
                    R const old_density=MetricProduct(half_x,aa,bad);
                    R const old_k=MetricProduct(old_density,v,bad);
                    R const half_z=MetricProduct(.5,z,bad);
                    R const new_density=MetricProduct(half_z,bb,bad);
                    R const new_k=MetricProduct(new_density,v,bad);
                    R const yd=MetricProduct(y,delta,bad);
                    R const ydv=MetricProduct(yd,v,bad);
                    R const work=MetricProduct(mean,ydv,bad);
                    R const stage_work=MetricProduct(mid,ydv,bad);
                    R const metric_new=MetricProduct(z-y,bb,bad);
                    R const metric_old=MetricProduct(x-y,aa,bad);
                    R const metric_density=MetricProduct(.5,metric_new-metric_old,bad);
                    R const metric=MetricProduct(metric_density,v,bad);
                    R const rotation=MetricProduct(mean-mid,ydv,bad);
                    // Include un-subtracted metric products in the roundoff
                    // scale. Each multiplication checks before later factors.
                    R const abs_new=MetricProduct(std::abs(z)+std::abs(y),bb,bad);
                    R const abs_old=MetricProduct(std::abs(x)+std::abs(y),aa,bad);
                    R const abs_density=MetricProduct(.5,abs_new+abs_old,bad);
                    R const abs_metric=MetricProduct(abs_density,v,bad);
                    R const abs_work=MetricProduct(std::abs(mean)+std::abs(mid),std::abs(ydv),bad);
                    R const scale=abs_metric+std::abs(old_k)+std::abs(new_k)+abs_work+
                        std::abs(work)+std::abs(stage_work)+std::abs(metric)+std::abs(rotation);
                    bad=bad||!std::isfinite(scale);
                    return{old_k,new_k,work,stage_work,metric,t(i,j,k)?metric:0.,rotation,scale,1,int(bad)};
                });
            }
            auto result=data.value();values[0]+=amrex::get<0>(result);values[1]+=amrex::get<1>(result);
            values[2]+=amrex::get<2>(result);values[3]+=amrex::get<3>(result);values[4]+=amrex::get<4>(result);
            values[5]+=amrex::get<5>(result);values[6]+=amrex::get<6>(result);values[7]+=amrex::get<7>(result);
            rows+=amrex::get<8>(result);invalid=std::max(invalid,amrex::get<9>(result));
        }
        amrex::ParallelDescriptor::ReduceRealSum(values,8);amrex::ParallelDescriptor::ReduceLongSum(rows);
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        bool finite=invalid==0;for(R x:values)finite=std::isfinite(x)&&finite;
        R const n=R(128)*(R(rows)+R(amrex::ParallelDescriptor::NProcs())+16.);
        R const ne=n*std::numeric_limits<R>::epsilon();
        if(!All(finite&&std::isfinite(ne)&&ne<.01))return false;
        StoppingMovingMetricReport r;r.available=true;r.rows=rows;
        r.initial_kinetic=values[0];r.final_kinetic=values[1];r.endpoint_mean_work=values[2];
        r.stage_work=values[3];r.metric_work=values[4];r.transition_metric_work=values[5];
        r.rotation_work=values[6];r.absolute_scale=values[7];
        r.identity_remainder=(r.final_kinetic-r.initial_kinetic)-(r.stage_work+r.metric_work+r.rotation_work);
        r.arithmetic_bound=Up(Up(ne/(1.-ne))*r.absolute_scale+Up(n*std::numeric_limits<R>::denorm_min()));
        if(!All(std::isfinite(r.identity_remainder)&&std::isfinite(r.arithmetic_bound)&&
            std::abs(r.identity_remainder)<=r.arithmetic_bound))return false;
        report.moving_metric=r;return true;
    }
    bool Phase(StoppingCarryStatus p)const{return report.status==p;}
    bool Fail(StoppingCarryStatus p){report.status=p;return false;}
    bool Clock()const {
        bool good=Scope(w)&&w.gett_new(0)==time&&w.getistep(0)==step&&
            w.boxArray(0)==cells&&w.DistributionMap(0)==distribution&&w.Geom(0).Coord()==native_coord&&
            (!transition||transition->ConfigurationMatches(w));
        for(int d=0;d<AMREX_SPACEDIM;++d)good=good&&w.Geom(0).ProbLo(d)==geometry.ProbLo(d)&&
            w.Geom(0).ProbHi(d)==geometry.ProbHi(d)&&w.Geom(0).CellSize(d)==geometry.CellSize(d)&&
            w.Geom(0).isPeriodic(d)==geometry.isPeriodic(d);
        return good;
    }
};
NativeStoppingCarryCertificate::NativeStoppingCarryCertificate(WarpX& w,NativeStoppingCarryMode mode):m_impl(std::make_unique<Impl>(w,mode)){}
NativeStoppingCarryCertificate::~NativeStoppingCarryCertificate()=default;
void NativeStoppingCarryCertificate::Invalidate() noexcept {m_impl->report.status=StoppingCarryStatus::Stale;}
StoppingCarryReport const& NativeStoppingCarryCertificate::Report()const noexcept{return m_impl->report;}
std::shared_ptr<NativeStoppingCarryCertificate const>
NativeStoppingCarryCertificate::CaptureAcceptedOrigin(WarpX& w,R time,
    std::uint64_t epoch,std::uint64_t generation){
    bool ready=Scope(w)&&w.getistep(0)>=0&&time==w.gett_new(0)&&
        epoch==std::uint64_t(w.getistep(0))&&generation>0&&
        generation<std::numeric_limits<std::uint64_t>::max();
    if(!All(ready))return {};
    auto owner=std::shared_ptr<NativeStoppingCarryCertificate>(
        new NativeStoppingCarryCertificate(w,NativeStoppingCarryMode::AcceptedOrigin));
    auto& s=*owner->m_impl;s.time=time;s.step=w.getistep(0);
    s.report.accepted_generation=generation;
    s.transition=std::make_unique<Impl::Transition>(w);
    for(int c=0;c<3;++c){Clone(s.endpoint[c],*w.m_fields.get(Je,Dir{c},0));
        // This is not a field-vacuum certificate. The uniform local flag only
        // keeps the common P/V decomposition from inventing transition rows.
        s.vacuum[c].define(s.endpoint[c].boxArray(),s.distribution,1,0);
        s.vacuum[c].setVal(1);}
    if(!owner->BindSourceEndpoint(time,epoch))return {};
    return owner;
}
bool NativeStoppingCarryCertificate::CaptureFieldOrigin(){
    auto& s=*m_impl;auto& w=s.w;
    bool ready=s.report.mode==NativeStoppingCarryMode::FieldTransition&&
        s.Phase(StoppingCarryStatus::Unavailable)&&!s.transition&&Scope(w)&&
        w.m_fields.has(FT::rho_fp,0)&&w.m_fields.has("hybrid_rho_vacmask_fp",0);
    for(int c=0;c<3;++c)ready=w.m_fields.has(Je,Dir{c},0)&&ready;
    if(!All(ready))return s.Fail(StoppingCarryStatus::Unsupported);
    s.time=w.gett_new(0);s.step=w.getistep(0);
    s.transition=std::make_unique<Impl::Transition>(w);
    if(!s.transition->CaptureMetric(w,0,*w.m_fields.get(FT::rho_fp,0),0))
        return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    for(int c=0;c<3;++c)Clone(s.transition->origin_current[c],*w.m_fields.get(Je,Dir{c},0));
    if(!s.transition->CaptureAmpere(w,0))return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    return true;
}
bool NativeStoppingCarryCertificate::ArmAtVerifiedRoot(DarwinVacuumJointSolve const& root){
    auto& s=*m_impl;auto& w=s.w;auto& r=s.report;
    bool ready=Scope(w)&& &root.w==&w&&root.m_verified&&root.m_active&&root.m_support_ready&&
        root.m_part==0&&root.m_generation==root.m_verified_generation&&root.RawAccepted()&&
        root.m_dt>0.&&root.m_h==.5*root.m_dt&&root.m_alpha>0.&&std::isfinite(root.m_alpha);
    if(!All(ready))return s.Fail(StoppingCarryStatus::Unsupported);
    if(r.mode==NativeStoppingCarryMode::FieldTransition){
        ready=s.transition&&s.transition->metric_ready[0]&&s.Clock()&&
            s.Phase(StoppingCarryStatus::Unavailable)&&root.m_time==s.time&&root.m_physical_step==s.step;
        if(!All(ready))return s.Fail(StoppingCarryStatus::Stale);
        bool exact=SameValid(s.transition->density[0],root.a.m_old_charge);
        for(int c=0;c<3;++c){bool same=Same(s.transition->origin_current[c],*w.m_fields.get(Je,Dir{c},0));exact=same&&exact;}
        if(!exact)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
        auto const& density=*w.m_fields.get(FT::rho_fp,0);
        if(!s.transition->CaptureMetric(w,1,density,density.nComp()/2))
            return s.Fail(StoppingCarryStatus::DeclinedIdentity);
        for(int c=0;c<3;++c)Clone(s.transition->field_physical[c],root.m_P[c]);
    }
    auto const scales=root.m_raw.blockScales();auto const dofs=root.m_raw.getDOFsObject();
    ready=ready&&!scales.empty()&&dofs&&dofs->m_array_masks.size()==1;
    for(int c=0;c<3;++c)ready=ready&&w.m_fields.has(Je,Dir{c},0);
    if(!All(ready))return s.Fail(StoppingCarryStatus::Unsupported);
    s.time=root.m_time;s.dt=root.m_dt;s.step=root.m_physical_step;
    r.root_generation=root.m_verified_generation;r.native_norm=root.m_raw_e;r.native_target=root.m_field_target;
    r.array_scale=scales[0];r.electric_scale=root.a.m_electric_scale;r.alpha=root.m_alpha;
    R const min=std::numeric_limits<R>::min(),scale_square=r.array_scale*r.array_scale;
    bool arithmetic=std::isfinite(r.native_norm)&&r.native_norm>=0.&&r.native_norm<=r.native_target&&
        std::isfinite(r.array_scale)&&r.array_scale>0.&&std::isfinite(r.electric_scale)&&r.electric_scale>0.&&
        std::isfinite(scale_square)&&scale_square>=min&&r.array_scale==r.electric_scale;
    if(!All(arithmetic))return s.Fail(StoppingCarryStatus::Unsupported);
    amrex::Long rows=0;int invalid=0;
    for(int c=0;c<3;++c){
        Clone(s.root_e[c],*root.m_raw.getArrayVec()[0][c]);Clone(s.root_j[c],root.m_current[c]);
        Clone(s.root_jp[c],*w.m_fields.get(FT::hybrid_current_fp_plasma,Dir{c},0));
        Clone(s.root_ion[c],*w.m_fields.get(FT::current_fp,Dir{c},0));
        Clone(s.old[c],*w.m_fields.get(Je,Dir{c},0));Clone(s.vacuum[c],root.m_V[c]);Clone(s.norm_mask[c],*dofs->m_array_masks[0][c]);
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpMax> op;amrex::ReduceData<amrex::Long,int> data(op);using T=decltype(data)::Type;
        R const alpha=r.alpha;
        for(amrex::MFIter it(s.root_e[c]);it.isValid();++it){auto e=s.root_e[c].const_array(it),j=s.root_j[c].const_array(it),jp=s.root_jp[c].const_array(it),ji=s.root_ion[c].const_array(it);auto v=s.vacuum[c].const_array(it),mask=s.norm_mask[c].const_array(it);
            op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int k,int z)->T{
                R const x=e(i,k,z),square=x*x,scaled=square/scale_square;
                bool bad=!std::isfinite(x)||(mask(i,k,z)&&x!=0.&&
                    (!std::isfinite(square)||square<min||!std::isfinite(scaled)||scaled<min));
                if(v(i,k,z))bad=bad||ji(i,k,z)!=0.||j(i,k,z)!=jp(i,k,z)-ji(i,k,z)||x!=alpha*j(i,k,z)||!std::isfinite(j(i,k,z));
                return{mask(i,k,z)?1:0,int(bad)};
            });
        }
        auto t=data.value();rows+=amrex::get<0>(t);invalid=std::max(invalid,amrex::get<1>(t));
    }
    amrex::ParallelDescriptor::ReduceLongSum(rows);amrex::ParallelDescriptor::ReduceIntMax(invalid);
    // Native masked Euclidean norm is retained, never changed to a volume/RMS norm.
    // Reject unsafe underflow/overflow rather than silently dropping squared rows.
    R const n=R(16)*(R(rows)+R(amrex::ParallelDescriptor::NProcs())+16.);
    R const ne=n*std::numeric_limits<R>::epsilon();
    if(invalid||!std::isfinite(ne)||ne>=.01)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    r.norm_rows=rows;
    r.norm_arithmetic_bound=Up(Up(ne/(1.-ne))*std::abs(r.native_norm)+
        Up(n*std::numeric_limits<R>::denorm_min()));
    if(!std::isfinite(r.norm_arithmetic_bound))return s.Fail(StoppingCarryStatus::Unsupported);
    r.status=StoppingCarryStatus::Root;return true;
}
bool NativeStoppingCarryCertificate::BindCapturedStage(){
    auto& s=*m_impl;auto& w=s.w;
    bool ready=s.Phase(StoppingCarryStatus::Root)&&s.Clock();
    for(int c=0;c<3;++c)for(auto const* name:{Stage,Corrected,IonOld,IonVirtual})ready=ready&&w.m_fields.has(name,Dir{c},0);
    if(!All(ready))return s.Fail(StoppingCarryStatus::Stale);
    bool old_same=true;for(int c=0;c<3;++c){bool same=Same(s.old[c],*w.m_fields.get(Je,Dir{c},0));old_same=same&&old_same;}
    if(!old_same)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    if(s.transition){
        auto const& charge=*w.m_fields.get(FT::rho_fp,0);MF density(charge,amrex::make_alias,charge.nComp()/2,1);
        bool const same_rho=SameValid(s.transition->density[1],density);
        bool const same_mask=SameValid(s.transition->mask_density[1],*w.m_fields.get("hybrid_rho_vacmask_fp",0));
        if(!same_rho||!same_mask)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    }
    int invalid=0;
    for(int c=0;c<3;++c){Clone(s.stage[c],*w.m_fields.get(Stage,Dir{c},0));
        auto const& oldion=*w.m_fields.get(IonOld,Dir{c},0);auto const& virtualion=*w.m_fields.get(IonVirtual,Dir{c},0);
        auto const& corrected=*w.m_fields.get(Corrected,Dir{c},0);auto const& jp=*w.m_fields.get(FT::hybrid_current_fp_plasma,Dir{c},0);
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
        for(amrex::MFIter it(s.stage[c]);it.isValid();++it){auto st=s.stage[c].const_array(it),co=corrected.const_array(it),a=oldion.const_array(it),b=virtualion.const_array(it),p=jp.const_array(it),j=s.root_j[c].const_array(it),p0=s.root_jp[c].const_array(it);auto v=s.vacuum[c].const_array(it);
            op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int k,int z)->T{
                bool bad=!std::isfinite(st(i,k,z));
                if(v(i,k,z))bad=bad||a(i,k,z)!=0.||b(i,k,z)!=0.||p(i,k,z)!=p0(i,k,z)||
                    st(i,k,z)!=co(i,k,z)||st(i,k,z)!=j(i,k,z);
                return{int(bad)};
            });
        }invalid=std::max(invalid,amrex::get<0>(data.value()));
    }
    amrex::ParallelDescriptor::ReduceIntMax(invalid);if(invalid)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    if(s.transition&&!s.transition->CaptureAmpere(w,1))return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    s.report.status=StoppingCarryStatus::Stage;return true;
}
bool NativeStoppingCarryCertificate::BindRotatedEndpoint(R theta){
    auto& s=*m_impl;if(!All(s.Phase(StoppingCarryStatus::Stage)&&s.Clock()&&theta==.5))return s.Fail(StoppingCarryStatus::Unsupported);
    // Reproduce the actual native LinComb and periodic synchronization on private
    // storage. This checks representation and aliases, never rewrites native Je.
    R const a=1./theta,b=1.-1./theta;bool good=true;
    for(int c=0;c<3;++c){Clone(s.endpoint[c],s.old[c]);
        MF::LinComb(s.endpoint[c],a,s.stage[c],0,b,s.old[c],0,0,1,s.endpoint[c].nGrowVect());
        s.endpoint[c].OverrideSync(s.geometry.periodicity());s.endpoint[c].FillBoundary(s.geometry.periodicity());
        bool same=Same(s.endpoint[c],*s.w.m_fields.get(Je,Dir{c},0));good=same&&good;
    }
    if(!good)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    s.report.rotation_coefficient=a;s.report.status=StoppingCarryStatus::Rotated;return true;
}
bool NativeStoppingCarryCertificate::BindSourceEndpoint(R time,std::uint64_t epoch){
    auto& s=*m_impl;auto& w=s.w;
    bool const accepted=s.report.mode==NativeStoppingCarryMode::AcceptedOrigin;
    bool const field_transition=s.report.mode==NativeStoppingCarryMode::FieldTransition;
    bool const phase=accepted?s.Phase(StoppingCarryStatus::Unavailable):s.Phase(StoppingCarryStatus::Rotated);
    if(!All(phase&&s.Clock()&&time==s.time+(accepted?0.:s.dt)&&
        epoch==std::uint64_t(s.step)+(accepted?0:1)))return s.Fail(StoppingCarryStatus::Stale);
    bool good=true;for(int c=0;c<3;++c){bool same=Same(s.endpoint[c],*w.m_fields.get(Je,Dir{c},0));good=same&&good;}
    if(!good)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
#if !defined(WARPX_DIM_RZ)
    return s.Fail(StoppingCarryStatus::Unsupported);
#else
    auto& rho=*w.m_fields.get(FT::rho_fp,0);Clone(s.rho,rho);
    auto& model=*w.get_pointer_HybridPICModel();warpx::darwin::NativeInertiaSupportOptions o;
    using namespace warpx::darwin;o.lower={InitialRateBoundary::Axis,InitialRateBoundary::Periodic};o.upper={InitialRateBoundary::PEC,InitialRateBoundary::Periodic};
    o.coefficient=InertiaEdgePolicy::NativeEdgeCandidate;o.recovery=InertiaRecoveryMask::None;o.components=InertiaRecoveryComponents::All;
    o.charge_floor=PhysConst::q_e*model.m_n_floor;o.reference_charge_density=PhysConst::q_e*model.m_n0_ref;
    NativeInertiaSupport support(s.geometry,s.cells,s.distribution,o);
    MF density(rho,amrex::make_alias,0,1);if(!support.FreezeEdges(density,density))return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    if(s.transition&&!s.transition->CaptureMetric(w,2,rho,0))return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    amrex::Long counts[4]={0,0,0,0};int const wall=s.geometry.Domain().bigEnd(0)+1;
    for(int c=0;c<3;++c){Clone(s.physical[c],support.PhysicalSupport(c));Clone(s.material[c],s.endpoint[c]);Clone(s.carry[c],s.endpoint[c]);
        auto owner=s.endpoint[c].OwnerMask(s.geometry.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;amrex::ReduceData<amrex::Long,amrex::Long,amrex::Long,amrex::Long> data(op);using T=decltype(data)::Type;
        for(amrex::MFIter it(s.material[c]);it.isValid();++it){auto p=s.physical[c].const_array(it),v=s.vacuum[c].const_array(it),own=owner->const_array(it);auto m=s.material[c].array(it),r=s.carry[c].array(it);
            op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int k,int z)->T{
                bool const empty=!p(i,k,z),fixed=c!=0&&i==wall,axis=c==1&&i==0;
                if(empty)m(i,k,z)=0.;else r(i,k,z)=0.;
                if(!own(i,k,z))return{0,0,0,0};
                return{empty&&!fixed&&!axis?1:0,empty&&fixed?1:0,empty&&axis?1:0,
                    empty&&!fixed&&!axis&&!v(i,k,z)?1:0};
            });
        }
        auto t=data.value();counts[0]+=amrex::get<0>(t);counts[1]+=amrex::get<1>(t);counts[2]+=amrex::get<2>(t);counts[3]+=amrex::get<3>(t);
        if(field_transition){
            auto& mask=s.transition->mask[c];mask.define(s.physical[c].boxArray(),s.distribution,1,0);
            amrex::ReduceOps<amrex::ReduceOpMax> check;amrex::ReduceData<int> checks(check);using CT=decltype(checks)::Type;
            for(amrex::MFIter it(mask);it.isValid();++it){auto p=s.physical[c].const_array(it),v=s.vacuum[c].const_array(it),pf=s.transition->field_physical[c].const_array(it);auto t=mask.array(it);
                check.eval(it.validbox(),checks,[=]AMREX_GPU_DEVICE(int i,int j,int k)->CT{
                    bool const free=!(c!=0&&i==wall)&&!(c==1&&i==0);
                    bool const transition=free&&!p(i,j,k)&&pf(i,j,k);
                    t(i,j,k)=int(transition);
                    bool const bad=free&&((pf(i,j,k)!=0&&v(i,j,k)!=0)||(!pf(i,j,k)&&!v(i,j,k)));
                    return{int(bad)};
                });
            }
            int invalid=amrex::get<0>(checks.value());amrex::ParallelDescriptor::ReduceIntMax(invalid);
            if(invalid)return s.Fail(StoppingCarryStatus::DeclinedIdentity);
        }
        // Material/carry are valid-row decompositions. The strict material owner
        // creates its own complete gather images; these buffers are never gathered.
        s.material[c].setBndry(0.);s.carry[c].setBndry(0.);
        s.material[c].OverrideSync(s.geometry.periodicity());s.material[c].FillBoundary(s.geometry.periodicity());
        s.carry[c].OverrideSync(s.geometry.periodicity());s.carry[c].FillBoundary(s.geometry.periodicity());
    }
    amrex::ParallelDescriptor::ReduceLongSum(counts,4);s.report.free_rows=counts[0];s.report.fixed_rows=counts[1];s.report.axis_rows=counts[2];s.report.transition_rows=counts[3];
    if(counts[3]&&!s.transition)return s.Fail(StoppingCarryStatus::DeclinedTransition);
    if(field_transition){
        if(!s.transition->CaptureAmpere(w,2)||!s.MeasureAmpereHistory()||!s.MeasureMovingMetric())
            return s.Fail(StoppingCarryStatus::DeclinedIdentity);
        s.report.free_rows=counts[0]-counts[3];
    }
    if(accepted&&!s.transition->CaptureAmpere(w,2))return s.Fail(StoppingCarryStatus::DeclinedIdentity);
    s.endpoint_time=time;s.epoch=epoch;
    s.report.status=accepted?StoppingCarryStatus::AcceptedReady:
        field_transition?StoppingCarryStatus::TransitionReady:StoppingCarryStatus::Ready;return true;
#endif
}
bool NativeStoppingCarryCertificate::ValidateSource(WarpX const& w,R time,std::uint64_t epoch)const{
    auto const& s=*m_impl;
    if(!All(&w==&s.w&&s.Ready()&&s.Clock()&&time==s.endpoint_time&&epoch==s.epoch))return false;
    bool good=Same(s.rho,*s.w.m_fields.get(FT::rho_fp,0));
    for(int c=0;c<3;++c){bool same=Same(s.endpoint[c],*s.w.m_fields.get(Je,Dir{c},0));good=same&&good;}
    if(s.transition){
        bool present=s.transition->metric_ready[2]&&s.w.m_fields.has("hybrid_rho_vacmask_fp",0);
        if(!All(present))return false;
        // FreezeEdges consumes valid mask density and completes private
        // images. Match precisely that producer input, never caller ghosts.
        bool const same=SameValid(s.transition->mask_density[2],
            *s.w.m_fields.get("hybrid_rho_vacmask_fp",0));
        good=same&&good;
        bool ready=s.transition->ampere[2].ready&&(s.report.mode==NativeStoppingCarryMode::AcceptedOrigin||s.report.ampere_history.available);
        if(!All(ready))return false;
        for(int c=0;c<3;++c){auto const& phase=s.transition->ampere[2];
            bool const a=Same(phase.potential[c],*s.w.m_fields.get("hybrid_A_fp",Dir{c},0));
            bool const b=Same(phase.magnetic[c],*s.w.m_fields.get(FT::Bfield_fp,Dir{c},0));
            bool const d=Same(phase.displacement[c],*s.w.m_fields.get("diagnostic_D_endpoint_fp",Dir{c},0));
            good=a&&b&&d&&good;
        }
    }
    return good;
}
bool NativeStoppingCarryCertificate::ValidateTrial(CV const& psi)const{
    auto const& s=*m_impl;bool ready=s.Ready()&&s.Clock();
    for(int c=0;c<3;++c)ready=ready&&psi[c]&&psi[c]->boxArray()==s.endpoint[c].boxArray()&&psi[c]->DistributionMap()==s.distribution&&psi[c]->nComp()==1;
    if(!All(ready))return false;
    int invalid=0;int const wall=s.geometry.Domain().bigEnd(0)+1;
    for(int c=0;c<3;++c){amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
        for(amrex::MFIter it(*psi[c]);it.isValid();++it){auto x=psi[c]->const_array(it),r=s.carry[c].const_array(it);auto p=s.physical[c].const_array(it);
            op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int k,int z)->T{
                bool const fixed=c!=0&&i==wall,axis=c==1&&i==0;
                return{int(!std::isfinite(x(i,k,z))||(!p(i,k,z)&&(fixed||axis)&&x(i,k,z)!=0.)||!std::isfinite(r(i,k,z)))};
            });
        }invalid=std::max(invalid,amrex::get<0>(data.value()));
    }
    amrex::ParallelDescriptor::ReduceIntMax(invalid);return invalid==0;
}
bool NativeStoppingCarryCertificate::MeasureWork(CV const& psi,StoppingCarryReport& out)const{
    auto const& s=*m_impl;if(!ValidateSource(s.w,s.endpoint_time,s.epoch)||!ValidateTrial(psi))return false;
    if(s.report.mode==NativeStoppingCarryMode::AcceptedOrigin){
        R signed_work=0.,absolute_work=0.,fixed_work=0.;int invalid=0;
        int const wall=s.geometry.Domain().bigEnd(0)+1;
        for(int c=0;c<3;++c){auto own=s.endpoint[c].OwnerMask(s.geometry.periodicity());
            auto volume=MakeQdsmcVolumeElement(s.geometry,s.endpoint[c].ixType());
            amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpMax> op;
            amrex::ReduceData<R,R,R,int> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(s.endpoint[c]);it.isValid();++it){auto x=psi[c]->const_array(it),j=s.carry[c].const_array(it);
                auto p=s.physical[c].const_array(it),o=own->const_array(it);
                op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int k,int z)->T{
                    if(!o(i,k,z)||p(i,k,z))return{0.,0.,0.,0};
                    R const work=-j(i,k,z)*(x(i,k,z)*volume(i,k,z));
                    bool const fixed=(c!=0&&i==wall)||(c==1&&i==0);
                    return{fixed?0.:work,std::abs(work),fixed?work:0.,
                        int(!std::isfinite(work)||(fixed&&work!=0.))};
                });}
            auto v=data.value();signed_work+=amrex::get<0>(v);absolute_work+=amrex::get<1>(v);
            fixed_work+=amrex::get<2>(v);invalid=std::max(invalid,amrex::get<3>(v));
        }
        amrex::ParallelDescriptor::ReduceRealSum(signed_work);amrex::ParallelDescriptor::ReduceRealSum(absolute_work);
        amrex::ParallelDescriptor::ReduceRealSum(fixed_work);amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if(!All(invalid==0&&std::isfinite(signed_work)&&std::isfinite(absolute_work)&&std::isfinite(fixed_work)))return false;
        out=s.report;out.accepted_work=signed_work;out.accepted_absolute_work=absolute_work;
        out.fixed_work=fixed_work;out.absolute_work=absolute_work;return true;
    }
    R values[7]={0.,0.,0.,0.,0.,0.,0.};int const wall=s.geometry.Domain().bigEnd(0)+1;
    for(int c=0;c<3;++c){auto owner=s.endpoint[c].OwnerMask(s.geometry.periodicity());auto volume=MakeQdsmcVolumeElement(s.geometry,s.endpoint[c].ixType());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpMax> op;
        amrex::ReduceData<R,R,R,R,R,R,int> data(op);using T=decltype(data)::Type;R const alpha=s.report.alpha,a=s.report.rotation_coefficient;
        bool const has_transition=static_cast<bool>(s.transition);
        for(amrex::MFIter it(s.endpoint[c]);it.isValid();++it){auto x=psi[c]->const_array(it),r=s.carry[c].const_array(it),e=s.root_e[c].const_array(it),old=s.old[c].const_array(it);auto p=s.physical[c].const_array(it),own=owner->const_array(it),mask=s.norm_mask[c].const_array(it);
            amrex::Array4<int const> transition;if(has_transition)transition=s.transition->mask[c].const_array(it);
            op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int k,int z)->T{
                if(!own(i,k,z)||p(i,k,z)||(has_transition&&transition(i,k,z)))return{0.,0.,0.,0.,0.,0.,0};
                R const vol=volume(i,k,z),q=x(i,k,z)*vol,work=-r(i,k,z)*q;
                bool const fixed=c!=0&&i==wall,axis=c==1&&i==0;
                if(fixed||axis)return{0.,work,0.,0.,std::abs(work),0.,int(work!=0.)};
                R const dual=q/alpha,root=-a*e(i,k,z)*dual,history=-(1.-a)*old(i,k,z)*q;
                R const square=dual*dual;
                bool bad=mask(i,k,z)!=1||!std::isfinite(work)||!std::isfinite(root)||!std::isfinite(history)||!std::isfinite(square)||
                    (dual!=0.&&square<std::numeric_limits<R>::min());
                return{work,0.,root,history,std::abs(work)+std::abs(root)+std::abs(history),square,int(bad)};
            });
        }
        auto t=data.value();values[0]+=amrex::get<0>(t);values[1]+=amrex::get<1>(t);values[2]+=amrex::get<2>(t);values[3]+=amrex::get<3>(t);values[4]+=amrex::get<4>(t);values[5]+=amrex::get<5>(t);values[6]+=amrex::get<6>(t);
    }
    amrex::ParallelDescriptor::ReduceRealSum(values,7);bool good=values[6]==0.;for(R x:values)good=good&&std::isfinite(x);if(!All(good))return false;
    auto result=s.report;result.free_work=values[0];result.fixed_work=values[1];result.root_work=values[2];result.history_work=values[3];result.represented_work_remainder=values[0]-(values[2]+values[3]);result.absolute_work=values[4];result.dual_norm=std::sqrt(values[5]);
    // The producer norm has equal stored array/electric scales (checked above).
    // Enclose reduction/multiply/divide/sqrt rounding of the dual and dot.
    R const n=R(32)*(R(result.norm_rows)+R(amrex::ParallelDescriptor::NProcs())+16.);
    R const ne=n*std::numeric_limits<R>::epsilon();
    if(!All(std::isfinite(ne)&&ne<.01))return false;
    R const gamma=Up(ne/(1.-ne));
    result.dual_norm=Up(Up(result.dual_norm/(1.-gamma))+Up(n*std::numeric_limits<R>::denorm_min()));
    result.root_dual_bound=Up(Up(Up(std::abs(result.rotation_coefficient)*Up(result.native_norm+result.norm_arithmetic_bound))*result.dual_norm)/(1.-gamma));
    if(!All(std::isfinite(result.root_dual_bound)))return false;
    if(s.transition){
        R signed_work=0.,absolute_work=0.;int invalid=0;
        for(int c=0;c<3;++c){auto owner=s.endpoint[c].OwnerMask(s.geometry.periodicity());auto volume=MakeQdsmcVolumeElement(s.geometry,s.endpoint[c].ixType());
            amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpMax> op;
            amrex::ReduceData<R,R,int> data(op);using T=decltype(data)::Type;
            for(amrex::MFIter it(s.endpoint[c]);it.isValid();++it){auto x=psi[c]->const_array(it),r=s.carry[c].const_array(it);
                auto own=owner->const_array(it),transition=s.transition->mask[c].const_array(it);
                op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{
                    if(!own(i,j,k)||!transition(i,j,k))return{0.,0.,0};
                    R const work=-r(i,j,k)*(x(i,j,k)*volume(i,j,k));
                    return{work,std::abs(work),int(!std::isfinite(work))};
                });
            }
            auto v=data.value();signed_work+=amrex::get<0>(v);absolute_work+=amrex::get<1>(v);invalid=std::max(invalid,amrex::get<2>(v));
        }
        amrex::ParallelDescriptor::ReduceRealSum(signed_work);amrex::ParallelDescriptor::ReduceRealSum(absolute_work);
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if(!All(invalid==0&&std::isfinite(signed_work)&&std::isfinite(absolute_work)))return false;
        result.transition_work=signed_work;result.transition_absolute_work=absolute_work;
        result.absolute_work+=absolute_work;
        if(!All(std::isfinite(result.absolute_work)))return false;
    }
    out=result;return true;
}
NativeStoppingCarryCertificate::CV NativeStoppingCarryCertificate::MaterialCurrent()const{return m_impl->Ready()?View(m_impl->material):CV{};}
NativeStoppingCarryCertificate::CV NativeStoppingCarryCertificate::CarriedCurrent()const{return m_impl->Ready()?View(m_impl->carry):CV{};}
NativeStoppingSourceOriginView NativeStoppingCarryCertificate::SourceOrigin()const{
    NativeStoppingSourceOriginView v;auto const& s=*m_impl;
    bool const f=s.Phase(StoppingCarryStatus::TransitionReady),a=s.Phase(StoppingCarryStatus::AcceptedReady);
    if((!f&&!a)||!s.transition||!s.Clock()||!s.transition->ampere[2].ready||
       !s.transition->metric_ready[2]||(f&&!s.report.ampere_history.available))return v;
    v.ready=true;v.field_transition=f;auto const& p=s.transition->ampere[2];
    v.current=View(p.current);v.ion=View(p.ion);v.electron=View(s.endpoint);
    v.displacement=View(p.displacement);v.mass=View(s.transition->mass[2]);return v;
}
NativeStoppingTransitionView NativeStoppingCarryCertificate::TransitionView()const{
    NativeStoppingTransitionView view;auto const& s=*m_impl;
    if(!s.Phase(StoppingCarryStatus::TransitionReady)||!s.transition||!s.Clock())return view;
    view.ready=true;view.old_current=View(s.old);view.stage_current=View(s.stage);view.endpoint_current=View(s.endpoint);
    view.old_mass=View(s.transition->mass[0]);view.stage_mass=View(s.transition->mass[1]);
    view.endpoint_mass=View(s.transition->mass[2]);view.root_residual=View(s.root_e);
    for(int c=0;c<3;++c){view.field_physical[c]=&s.transition->field_physical[c];
        view.field_vacuum[c]=&s.vacuum[c];view.transition[c]=&s.transition->mask[c];
        view.phase_density[c]=&s.transition->density[c];view.phase_mask_density[c]=&s.transition->mask_density[c];
        view.phase_curl_current[c]=View(s.transition->ampere[c].current);
        view.phase_ion[c]=View(s.transition->ampere[c].ion);
        view.phase_displacement[c]=View(s.transition->ampere[c].displacement);}
    return view;
}
void DarwinThermalAdvance::RequestStoppingCarry(bool requested){
    if(m_stopping_carry)m_stopping_carry->Invalidate();
    m_stopping_carry.reset();
    if(requested)m_stopping_carry=std::shared_ptr<NativeStoppingCarryCertificate>(new NativeStoppingCarryCertificate(m_simulation,NativeStoppingCarryMode::VacuumResidualOnly));
}
void DarwinThermalAdvance::RequestStoppingCarry(NativeStoppingCarryRequest const& request){
    if(request.mode==NativeStoppingCarryMode::VacuumResidualOnly){RequestStoppingCarry(true);return;}
    if(m_stopping_carry)m_stopping_carry->Invalidate();
    m_stopping_carry.reset();
    if(request.mode!=NativeStoppingCarryMode::FieldTransition)return;
    m_stopping_carry=std::shared_ptr<NativeStoppingCarryCertificate>(new NativeStoppingCarryCertificate(m_simulation,request.mode));
    m_stopping_carry->CaptureFieldOrigin();
}
void DarwinThermalAdvance::BindStoppingCarryStage(){if(m_stopping_carry)m_stopping_carry->BindCapturedStage();}
void DarwinThermalAdvance::BindStoppingCarryRotation(R theta){if(m_stopping_carry)m_stopping_carry->BindRotatedEndpoint(theta);}
std::shared_ptr<NativeStoppingCarryCertificate const> DarwinThermalAdvance::BindStoppingCarryEndpoint(R time,std::uint64_t epoch){
    if(m_stopping_carry)m_stopping_carry->BindSourceEndpoint(time,epoch);
    return m_stopping_carry;
}

}
