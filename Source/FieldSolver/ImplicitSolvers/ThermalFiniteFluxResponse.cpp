/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ThermalFiniteFluxResponse.H"
#include "ThermalFiniteConduction.H"
#include "ThermalFiniteWall.H"
#include "EulerianThermalStage.H"
#include "EulerianThermalStageUtils.H"
#include "Utils/WarpXConst.H"
#include <AMReX_Reduce.H>
#include <sstream>
#include <iomanip>
namespace warpx::thermal {
namespace {
using MF=amrex::MultiFab;using Real=amrex::Real;
using Field=ThermalFiniteFluxResponse::Field;
namespace f=finite_flux;
using f::Ball;using f::Value;
void Define(MF& a,MF const& b,int components,int grow) {
    if(!a.isDefined())a.define(b.boxArray(),b.DistributionMap(),components,grow);
    AMREX_ALWAYS_ASSERT(a.boxArray()==b.boxArray()&&a.DistributionMap()==b.DistributionMap()&&a.nComp()==components&&a.nGrow()==grow);
}
void Define(Field& a,MF const& b,int components,int grow) {
    Define(a.high,b,components,grow);Define(a.low,b,components,grow);Define(a.radius,b,components,grow);
}
void Images(Field& a,amrex::Geometry const& g) {
    stage_detail::fill_images(a.high,g);stage_detail::fill_images(a.low,g);stage_detail::fill_images(a.radius,g);
}
bool Finite(Field const& a) {
    int const n=a.high.nComp(),g=a.high.nGrow();
    return a.high.is_finite(0,n,g)&&a.low.is_finite(0,n,g)&&a.radius.is_finite(0,n,g)&&a.radius.min(0)>=0.;
}
bool Layout(MF const& a,MF const& b,int components=1) {
    return a.nComp()==components&&!a.hasEBFabFactory()&&a.boxArray()==b.boxArray()&&a.DistributionMap()==b.DistributionMap();
}
// Endpoint invalidation epochs identify each rank's local response lifetime.
// A last-rank invalidation may leave their numbers different after a rebuild;
// validate every local epoch collectively, then retain exact local binding.
bool LocalResponseEpoch(std::uint64_t value) {
    return remainder::All(value>0 && value<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max()));
}
bool Agreed(std::uint64_t value) {
    if(!remainder::All(value>0&&value<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max())))return false;
    amrex::Long lo=static_cast<amrex::Long>(value),hi=lo;
    amrex::ParallelDescriptor::ReduceLongMin(lo);amrex::ParallelDescriptor::ReduceLongMax(hi);return lo==hi;
}
bool AgreedContext(EulerianThermalStage const& stage) {
    auto const& o=stage.Options();auto const& g=stage.Geometry();
    std::ostringstream text;text<<std::hexfloat<<o.dt<<' '<<o.theta<<' '<<o.conduction_theta<<' '<<o.time<<' '<<o.gamma<<' '
        <<o.temperature_floor_ev<<' '<<o.magnetic_floor<<' '<<o.limiter_width<<' '<<o.free_streaming_fraction<<' '
        <<o.order<<' '<<o.cross_mode<<' '<<o.conductivity.isotropic<<' '<<o.conductivity.isotropic_B<<' '
        <<o.conductivity.unmagnetized_parallel<<' '<<o.use_compiled_conductivity<<' '
        <<std::quoted(o.use_compiled_conductivity?o.parallel_owner->expr():o.kappa_parallel)<<' '
        <<std::quoted(o.use_compiled_conductivity?o.perpendicular_owner->expr():o.kappa_perpendicular)<<' ';
    for(int d=0;d<AMREX_SPACEDIM;++d){text<<g.Domain().smallEnd(d)<<' '<<g.Domain().bigEnd(d)<<' '<<g.ProbLo(d)<<' '<<g.ProbHi(d)<<' '<<g.isPeriodic(d)<<' ';
        for(auto const& bc:o.boundary[d])text<<int(bc.kind)<<' '<<bc.value<<' '<<bc.leg_length<<' '<<bc.flux_limit<<' '<<int(bc.cap_form)<<' '<<bc.ion_mass<<' '<<bc.drain_only<<' '<<bc.inflow_temperature_ev<<' ';
    }
    std::string const local=text.str();if(!remainder::All(local.size()<1048576))return false;
    int length=static_cast<int>(local.size());int const root=amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&length,1,root);std::string reference(static_cast<std::size_t>(length),' ');
    if(amrex::ParallelDescriptor::IOProcessor())reference=local;
    amrex::ParallelDescriptor::Bcast(reference.data(),length,root);return remainder::All(reference==local);
}
struct Read {
    amrex::Array4<Real const> high,low,radius;
    AMREX_GPU_HOST_DEVICE Ball operator()(int i,int j,int k,int c=0) const noexcept {
        return {{high(i,j,k,c),low(i,j,k,c)},radius(i,j,k,c)};
    }
};
struct Write {
    amrex::Array4<Real> high,low,radius;
    AMREX_GPU_HOST_DEVICE void operator()(int i,int j,int k,Ball x,int c=0) const noexcept {
        high(i,j,k,c)=x.value.hi;low(i,j,k,c)=x.value.lo;radius(i,j,k,c)=x.radius;
    }
};
Read Reader(Field const& x,amrex::MFIter const& it){return {x.high.const_array(it),x.low.const_array(it),x.radius.const_array(it)};}
Write Writer(Field& x,amrex::MFIter const& it){return {x.high.array(it),x.low.array(it),x.radius.array(it)};}
AMREX_GPU_HOST_DEVICE Ball Anchor(Ball x,Real high) noexcept {
    auto const difference=remainder::Sum(x.value.hi,-high);
    auto const tail=f::Add({difference,0.},{{x.value.lo,0.},x.radius});
    // Preserve the requested high plus a rounded low and its operation bound.
    auto const low=remainder::Sum(tail.value.hi,tail.value.lo);
    return {{high,low.hi},f::Upper(tail.radius+std::abs(low.lo))};
}
AMREX_GPU_HOST_DEVICE Value Anchor(Value x,Real base_high) noexcept {
    x.base=Anchor(x.base,base_high);return f::Check(x);
}
}
void ThermalFiniteFluxResponse::Invalidate() noexcept {
    m_captured=false;m_ready=false;m_publication=0;m_sample=0;m_base={};m_current={};
}
bool ThermalFiniteFluxResponse::Scope(EulerianThermalStage const& stage) const {
    auto const& o=stage.m_options;
    bool valid=remainder::ArithmeticSupported()&&stage.CompletedReceipt().Current()&&stage.m_zero_source_provider&&
        !stage.m_remainder_context&&o.conduction_theta==o.theta&&
        (o.free_streaming_fraction==0.||o.free_streaming_fraction==.1);
    if(o.use_compiled_conductivity){
        valid=valid&&o.parallel_owner&&o.perpendicular_owner;
        if(o.parallel_owner&&o.perpendicular_owner)valid=valid&&
            o.parallel_owner->compileHost<3>().m_host_executor==stage.m_parallel.m_host_executor&&
            o.perpendicular_owner->compileHost<3>().m_host_executor==stage.m_perpendicular.m_host_executor&&
            o.parallel_owner->userFunctions().empty()&&o.perpendicular_owner->userFunctions().empty();
    }
    for(auto const& sides:o.boundary)for(auto const& bc:sides)valid=valid&&
        bc.flux_limit==0.&&!bc.drain_only&&bc.kind!=BoundaryKind::LegUnsupported;
#if !defined(WARPX_DIM_RZ)
    valid=false; // full 3D compile remains; physical scope is precise RZ only
#endif
    if(!remainder::All(valid))return false;
    return stage.m_live_source.norminf(0)==0.&&stage.m_source_derivative.norminf(0)==0.;
}
bool ThermalFiniteFluxResponse::Capture(EulerianThermalStage const& stage,Binding binding) {
    Invalidate();
    if(!Agreed(binding.field_base)||!LocalResponseEpoch(binding.particle_response)||!Agreed(binding.density_base)||!Scope(stage)||!AgreedContext(stage))return false;
    finite_flux::MaterialLaw parallel,perpendicular;
    auto const& o=stage.m_options;
    std::string const par=o.use_compiled_conductivity?o.parallel_owner->expr():o.kappa_parallel;
    std::string const perp=o.use_compiled_conductivity?o.perpendicular_owner->expr():o.kappa_perpendicular;
    bool const material=f::ParseMaterial(par,parallel)&&f::ParseMaterial(perp,perpendicular);
    // A provider allocation belongs to one native partition. Reuse the same
    // storage only for that partition; different layouts decline collectively
    // before Define/copy. The caller can construct a fresh provider instead.
    bool const layout=!m_u0.isDefined()||Layout(stage.m_stage,m_u0);
    if(!remainder::All(material&&layout))return false;
    m_geometry=stage.m_geometry;m_parallel=parallel;m_perpendicular=perpendicular;
    Define(m_u0,stage.m_stage,1,3);Define(m_n0,stage.m_density,1,3);
    Define(m_b0,stage.m_magnetic,3,3);Define(m_kappa0,stage.m_kappa,2,3);
    Define(m_activity0,stage.m_conduction_active,1,3);Define(m_e0,m_u0,1,3);Define(m_te0,m_u0,1,3);
    for(auto const& item:std::array<std::pair<MF*,MF const*>,5>{{{&m_u0,&stage.m_stage},{&m_n0,&stage.m_density},{&m_b0,&stage.m_magnetic},{&m_kappa0,&stage.m_kappa},{&m_activity0,&stage.m_conduction_active}}})
        MF::Copy(*item.first,*item.second,0,0,item.first->nComp(),3);
    for(auto* x:{&m_du,&m_dn,&m_de,&m_dte,&m_ebase,&m_tebase})Define(*x,m_u0,1,3);
    Define(m_db,m_u0,3,3);Define(m_dkappa,m_u0,2,3);Define(m_kbase,m_u0,2,3);
    Define(m_dsource,m_u0,1,0);Define(m_dr,m_u0,1,0);
    for(int d=0;d<AMREX_SPACEDIM;++d){
        Define(m_q0[d],*stage.m_flux[d],1,0);MF::Copy(m_q0[d],*stage.m_flux[d],0,0,1,0);
        Define(m_dq[d],*stage.m_flux[d],1,0);Define(m_leg[d],*stage.m_flux[d],4,0);
    }
    m_binding=binding;m_base=stage.CompletedReceipt();m_captured=true;return true;
}
bool ThermalFiniteFluxResponse::Ready(Binding b,std::uint64_t p,std::uint64_t sample) const noexcept {
    return m_captured&&m_ready&&m_binding==b&&p&&sample&&m_publication==p&&m_sample==sample&&
        m_current.Current()&&m_current.SameInstance(m_base);
}
bool ThermalFiniteFluxResponse::Apply(EulerianThermalStage const& stage,Binding binding,
    std::uint64_t publication,std::uint64_t sample,PairView energy,PairView density) {
    m_ready=false;
    if(!Agreed(binding.field_base)||!LocalResponseEpoch(binding.particle_response)||!Agreed(binding.density_base)||!Agreed(publication)||!Agreed(sample)){Invalidate();return false;}
    bool valid=m_captured&&binding==m_binding&&publication&&sample&&
        (m_sample==0||(sample>m_sample&&publication>m_publication))&&
        stage.CompletedReceipt().Current()&&stage.CompletedReceipt().SameInstance(m_base)&&
        Layout(energy.high,m_u0)&&Layout(energy.low,m_u0)&&Layout(density.high,m_n0)&&Layout(density.low,m_n0);
    // Inputs are borrowed and remain immutable. Reject every overlap with
    // owned storage before copying or overwriting any finite work array.
    for(auto const* input:{&energy.high,&energy.low,&density.high,&density.low}) {
        for(auto const* x:{&m_u0,&m_n0,&m_b0,&m_kappa0,&m_activity0,&m_e0,&m_te0})
            valid=valid&&!remainder::Overlap(*input,*x);
        for(auto const* field:{&m_du,&m_dn,&m_db,&m_de,&m_dte,&m_dkappa,&m_dsource,&m_dr,&m_ebase,&m_tebase,&m_kbase})
            for(auto const* x:{&field->high,&field->low,&field->radius})valid=valid&&!remainder::Overlap(*input,*x);
        for(int d=0;d<AMREX_SPACEDIM;++d)for(auto const* x:{&m_q0[d],&m_leg[d],&m_dq[d].high,&m_dq[d].low,&m_dq[d].radius})
            valid=valid&&!remainder::Overlap(*input,*x);
    }
    if(!remainder::All(valid)||!Scope(stage)){Invalidate();return false;}
    for(auto const* x:{&energy.high,&energy.low,&density.high,&density.low})if(!x->is_finite()){Invalidate();return false;}
    MF::Copy(m_du.high,energy.high,0,0,1,0);MF::Copy(m_du.low,energy.low,0,0,1,0);m_du.radius.setVal(0.);
    MF::Copy(m_dn.high,density.high,0,0,1,0);MF::Copy(m_dn.low,density.low,0,0,1,0);m_dn.radius.setVal(0.);
    // The exact Stage scalar image map operates on BOTH parts, including
    // periodic/FAB/corner images and its insulated physical reflection.
    Images(m_du,m_geometry);Images(m_dn,m_geometry);
    for(auto* x:{&m_dsource.high,&m_dsource.low,&m_dsource.radius})x->setVal(0.);
    if(!CellActions(stage)||!FaceActions(stage)||!Divergence(stage)){Invalidate();return false;}
    m_current=stage.CompletedReceipt();m_publication=publication;m_sample=sample;m_ready=true;return true;
}
bool ThermalFiniteFluxResponse::CellActions(EulerianThermalStage const& stage) {
    Real const gm=stage.m_options.gamma-1.;auto const options=stage.m_options.conductivity;
    auto const parallel=m_parallel,perpendicular=m_perpendicular;
    auto const& o=stage.m_options;bool const same_law=o.use_compiled_conductivity?o.parallel_owner->expr()==o.perpendicular_owner->expr():o.kappa_parallel==o.kappa_perpendicular;
    amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using Tuple=decltype(data)::Type;
    for(amrex::MFIter it(m_u0);it.isValid();++it){
        auto const u0=m_u0.const_array(it),n0=m_n0.const_array(it),b0=m_b0.const_array(it),kap0=m_kappa0.const_array(it);
        auto const b1=stage.m_magnetic.const_array(it),a0=m_activity0.const_array(it),a1=stage.m_conduction_active.const_array(it);
        auto const du=Reader(m_du,it),dn=Reader(m_dn,it);auto db=Writer(m_db,it);
        auto de=Writer(m_de,it),dte=Writer(m_dte,it),dk=Writer(m_dkappa,it);
        auto eb=Writer(m_ebase,it),tb=Writer(m_tebase,it),kb=Writer(m_kbase,it);
        auto const e0=m_e0.array(it),te0=m_te0.array(it);
        op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
            int bad=a0(i,j,k)!=a1(i,j,k);Value const u{{{u0(i,j,k),0.},0.},du(i,j,k),&bad};
            Value const n{{{n0(i,j,k),0.},0.},dn(i,j,k),&bad};
            auto const e=Anchor(u/n,u0(i,j,k)/n0(i,j,k));
            // Native coefficient Te order is gm*U/(n*qe), not Kelvin*kb/qe.
            auto const te=Anchor(Value(gm)*u/(n*Value(PhysConst::q_e)),gm*u0(i,j,k)/(n0(i,j,k)*PhysConst::q_e));
            de(i,j,k,e.delta);eb(i,j,k,e.base);dte(i,j,k,te.delta);tb(i,j,k,te.base);
            e0(i,j,k)=e.base.value.hi;te0(i,j,k)=te.base.value.hi;
            Value magnetic[3];
            for(int c=0;c<3;++c){Ball const delta{remainder::Sum(b1(i,j,k,c),-b0(i,j,k,c)),0.};db(i,j,k,delta,c);magnetic[c]={{{b0(i,j,k,c),0.},0.},delta,&bad};}
            if(a0(i,j,k)==0){for(int c=0;c<2;++c){dk(i,j,k,{},c);kb(i,j,k,{},c);}return {bad};}
            auto const b2=magnetic[0]*magnetic[0]+magnetic[1]*magnetic[1]+magnetic[2]*magnetic[2];
            auto kp=parallel.Evaluate(n,te),kt=perpendicular.Evaluate(n,te);
            if(options.isotropic||same_law)kp=kt;
            else if(options.isotropic_B>0.){
                Value const weight=(b2>Value(0.))?Value(1.)/(Value(1.)+Value(options.isotropic_B*options.isotropic_B)/b2):Value(0.);
                kp=kt+(kp-kt)*weight;
            }else if(options.unmagnetized_parallel && !(b2>Value(0.)))kt=kp;
            kp=Anchor(kp,kap0(i,j,k,0));kt=Anchor(kt,kap0(i,j,k,1));
            kb(i,j,k,kp.base,0);kb(i,j,k,kt.base,1);dk(i,j,k,kp.delta,0);dk(i,j,k,kt.delta,1);
            if(f::Sign(n.base)!=1||f::Sign(f::Add(n.base,n.delta))!=1||f::Sign(e.base)!=1||f::Sign(f::Add(e.base,e.delta))!=1)bad=1;
            return {bad};
        });
    }
    int bad=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceIntMax(bad);if(bad)return false;
    for(auto* x:{&m_db,&m_de,&m_dte,&m_dkappa,&m_ebase,&m_tebase,&m_kbase}){Images(*x,m_geometry);if(!Finite(*x))return false;}
    stage_detail::fill_images(m_e0,m_geometry);stage_detail::fill_images(m_te0,m_geometry);return true;
}
bool ThermalFiniteFluxResponse::FaceActions(EulerianThermalStage const& stage) {
#if defined(WARPX_DIM_RZ)
    auto const dx=m_geometry.CellSizeArray();auto const domain=m_geometry.Domain();Real const rlo=m_geometry.ProbLo(0);
    Real const gm=stage.m_options.gamma-1.,bfloor2=stage.m_options.magnetic_floor*stage.m_options.magnetic_floor;
    auto const law=m_parallel;auto const& options=stage.m_options;
    Real const flux_limit=options.free_streaming_fraction;
    bool const same_law=options.use_compiled_conductivity?options.parallel_owner->expr()==options.perpendicular_owner->expr():options.kappa_parallel==options.kappa_perpendicular;
    bool const isotropic=options.conductivity.isotropic||same_law;
    bool const weak_blend=options.conductivity.isotropic_B>0.,unmagnetized=options.conductivity.unmagnetized_parallel;
    for(int d=0;d<AMREX_SPACEDIM;++d){
        mhd::ChaconFDParams p;p.normal=d==0?0:2;p.tangential=d==0?2:0;p.order=stage.m_options.order;p.cross_mode=stage.m_options.cross_mode;
        p.radial_faces=d==0;p.periodic_normal=m_geometry.isPeriodic(d);p.periodic_tangential=m_geometry.isPeriodic(1-d);
        p.domain_lo_normal=domain.smallEnd(d);p.domain_hi_normal=domain.bigEnd(d);p.domain_lo_tangential=domain.smallEnd(1-d);p.domain_hi_tangential=domain.bigEnd(1-d);
        p.radial_lower=rlo;p.radial_cell_size=dx[0];p.inverse_normal_size=1/dx[d];p.inverse_tangential_size=1/dx[1-d];p.limiter_width=stage.m_options.limiter_width;p.b2_floor=bfloor2;
        auto const low=stage.m_options.boundary[d][0],high=stage.m_options.boundary[d][1];Real const floor=stage.m_options.temperature_floor_ev;
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using Tuple=decltype(data)::Type;
        for(amrex::MFIter it(m_dq[d].high);it.isValid();++it){
            auto const n0=m_n0.const_array(it),b0=m_b0.const_array(it),active=m_activity0.const_array(it);
            auto const dn=Reader(m_dn,it),db=Reader(m_db,it),eb=Reader(m_ebase,it),de=Reader(m_de,it),kb=Reader(m_kbase,it),dk=Reader(m_dkappa,it);
            auto out=Writer(m_dq[d],it);auto const cert=m_leg[d].array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                int bad=0;out(i,j,k,{});for(int c=0;c<4;++c)cert(i,j,k,c)=0.;
                if(d==0&&rlo+i*dx[0]==0.)return {0};
                int const il=i-(d==0),jl=j-(d==1),kl=k;
                if(active(il,jl,kl)==0.||active(i,j,k)==0.)return {0};
                int const index=d==0?i:j;bool const atlo=!p.periodic_normal&&index==p.domain_lo_normal,athi=!p.periodic_normal&&index==p.domain_hi_normal+1;
                auto const e=[&](int ii,int jj,int kk){return Value(eb(ii,jj,kk),de(ii,jj,kk),&bad);};
                auto const n=[&](int ii,int jj,int kk){return Value({{n0(ii,jj,kk),0.},0.},dn(ii,jj,kk),&bad);};
                auto const b=[&](int ii,int jj,int kk,int c){return Value({{b0(ii,jj,kk,c),0.},0.},db(ii,jj,kk,c),&bad);};
                auto const kap=[&](int ii,int jj,int kk,int c){return Value(kb(ii,jj,kk,c),dk(ii,jj,kk,c),&bad);};
                // Exact zero-B constitutive branches also identify the two
                // channels. This is proved from base and retained increment,
                // never inferred from numerically close kappa enclosures.
                auto const scalar=[&](int ii,int jj,int kk){
                    if(isotropic)return true;
                    if(!(weak_blend||unmagnetized))return false;
                    for(int c=0;c<3;++c)if(b0(ii,jj,kk,c)!=0.||!f::ExactZero(db(ii,jj,kk,c)))return false;
                    return true;
                };
                auto const masked=[&](int ii,int jj,int kk){return active(ii,jj,kk)==0.||
                    (!p.periodic_normal&&(mhd::index_along(ii,jj,kk,p.normal)<p.domain_lo_normal||mhd::index_along(ii,jj,kk,p.normal)>p.domain_hi_normal))||
                    (!p.periodic_tangential&&(mhd::index_along(ii,jj,kk,p.tangential)<p.domain_lo_tangential||mhd::index_along(ii,jj,kk,p.tangential)>p.domain_hi_tangential));};
                Value q,cap_density,cap_energy;bool bulk_cap=true;
                if(atlo||athi){
                    auto const bc=atlo?low:high;Real const sign=atlo?-1.:1.;
                    if(bc.kind==BoundaryKind::Adiabatic||bc.kind==BoundaryKind::PrescribedFlux)return {0};
                    int const ic=atlo?i:il,jc=atlo?j:jl,kc=k;
                    if(bc.kind==BoundaryKind::Leg){
                        // Native wall converts specific energy directly with
                        // gm/qe. This is deliberately distinct from the cell
                        // coefficient Te order above, as in the high producer.
                        Value const te=Value(gm/PhysConst::q_e)*e(ic,jc,kc);
                        LegFaceParameters const wall{bc.value,bc.leg_length,.5*dx[d],0.,bc.cap_form==WallCapForm::Sonic,gm+1.,bc.ion_mass,floor};
                        auto const leg=f::Leg(te,n(ic,jc,kc),wall,law);q=Value(sign)*leg.flux;bulk_cap=false;
                        cert(i,j,k,0)=leg.base_width;cert(i,j,k,1)=leg.trial_width;cert(i,j,k,2)=leg.base_residual;cert(i,j,k,3)=leg.trial_residual;
                    }else if(bc.kind==BoundaryKind::Reservoir){
                        auto const bx=b(ic,jc,kc,0),by=b(ic,jc,kc,1),bz=b(ic,jc,kc,2),bn=b(ic,jc,kc,p.normal);
                        auto const b2=f::smooth_positive_floor(bx*bx+by*by+bz*bz,Value(bfloor2));
                        auto const anisotropy=scalar(ic,jc,kc)?Value(0.):kap(ic,jc,kc,0)-kap(ic,jc,kc,1);
                        auto const knn=kap(ic,jc,kc,1)+anisotropy*bn*bn/b2;
                        auto const nf=n(ic,jc,kc),chi=Value(gm)*knn/(nf*Value(PhysConst::kb));
                        Value const bath(PhysConst::q_e*bc.value/gm);
                        auto const bracket=Value(-sign)*Value(2.)*chi*(e(ic,jc,kc)-bath)/Value(dx[d]);q=-nf*bracket;
                        cap_density=nf;cap_energy=e(ic,jc,kc);
                    }else bad=1;
                }else{
                    auto const nf=Value(.5)*(n(il,jl,kl)+n(i,j,k));
                    auto const factor=Value(gm)/(nf*Value(PhysConst::kb));
                    auto const kp=factor*Value(.5)*(kap(il,jl,kl,0)+kap(i,j,k,0));
                    auto const kt=factor*Value(.5)*(kap(il,jl,kl,1)+kap(i,j,k,1));
                    Value chi;auto const bracket=f::chacon_fd_face_flux(e,masked,b,il,jl,kl,i,j,k,kt,(scalar(il,jl,kl)&&scalar(i,j,k))?Value(0.):kp-kt,p,chi);q=-nf*bracket;
                    cap_density=nf;cap_energy=Value(.5)*(e(il,jl,kl)+e(i,j,k));
                }
                // Native ComposeConductionFlux applies this smooth cap once
                // after stage mixing. theta_c==theta is part of Scope; the
                // leg returns earlier in the high producer and bypasses it.
                if(bulk_cap&&flux_limit>0.){
                    auto const kbt=Value(gm)*cap_energy;
                    auto const limit=Value(flux_limit)*cap_density*kbt*f::sqrt(kbt/Value(PhysConst::m_e));
                    if(f::Sign(limit.base)!=1||f::Sign(f::Add(limit.base,limit.delta))!=1)bad=1;
                    q=q/(Value(1.)+f::ConstitutiveAbs(q)/limit);
                }
                if(!f::Valid(q.base)||!f::Valid(q.delta))bad=1;
                out(i,j,k,q.delta);return {bad};
            });
        }
        int bad=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceIntMax(bad);if(bad)return false;
        for(auto* x:{&m_dq[d].high,&m_dq[d].low,&m_dq[d].radius})x->OverrideSync(m_geometry.periodicity());
        if(!Finite(m_dq[d]))return false;
    }return true;
#else
    amrex::ignore_unused(stage);return false;
#endif
}
bool ThermalFiniteFluxResponse::Divergence(EulerianThermalStage const& stage) {
    auto const dx=m_geometry.CellSizeArray();Real const h=stage.m_options.theta*stage.m_options.dt;
    [[maybe_unused]] Real const rlo=m_geometry.ProbLo(0);
    for(amrex::MFIter it(m_dr.high);it.isValid();++it){
        amrex::GpuArray<Read,AMREX_SPACEDIM> flux;for(int d=0;d<AMREX_SPACEDIM;++d)flux[d]=Reader(m_dq[d],it);
        auto const source=Reader(m_dsource,it);auto out=Writer(m_dr,it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            Ball div;
            for(int d=0;d<AMREX_SPACEDIM;++d){
                auto plus=flux[d](i+(d==0),j+(d==1),k+(d==2)),minus=flux[d](i,j,k);Real denom=dx[d];
#if defined(WARPX_DIM_RZ)
                if(d==0){plus=f::Scale(plus,rlo+(i+1)*dx[0]);minus=f::Scale(minus,rlo+i*dx[0]);denom=(rlo+(i+.5)*dx[0])*dx[0];}
#endif
                div=f::Add(div,f::Div(f::Add(plus,f::Neg(minus)),{{denom,0.},0.}));
            }
            out(i,j,k,f::Scale(f::Add(div,f::Neg(source(i,j,k))),h));
        });
    }return Finite(m_dr);
}
bool ThermalFiniteFluxResponse::AddResidual(MF& high,MF& low,Binding binding,std::uint64_t publication,std::uint64_t sample) const {
    bool valid=remainder::ArithmeticSupported()&&Ready(binding,publication,sample)&&Layout(high,m_dr.high)&&Layout(low,m_dr.high)&&!remainder::Overlap(high,low);
    for(auto const* x:{&m_u0,&m_n0,&m_b0,&m_kappa0,&m_activity0,&m_e0,&m_te0})valid=valid&&!remainder::Overlap(high,*x)&&!remainder::Overlap(low,*x);
    for(auto const* field:{&m_du,&m_dn,&m_db,&m_de,&m_dte,&m_dkappa,&m_dsource,&m_dr,&m_ebase,&m_tebase,&m_kbase})
        for(auto const* x:{&field->high,&field->low,&field->radius})valid=valid&&!remainder::Overlap(high,*x)&&!remainder::Overlap(low,*x);
    for(int d=0;d<AMREX_SPACEDIM;++d)for(auto const* x:{&m_q0[d],&m_leg[d],&m_dq[d].high,&m_dq[d].low,&m_dq[d].radius})
        valid=valid&&!remainder::Overlap(high,*x)&&!remainder::Overlap(low,*x);
    if(!remainder::All(valid))return false;
    for(amrex::MFIter it(high);it.isValid();++it){
        auto const h=high.array(it),l=low.array(it);
        auto const dh=m_dr.high.const_array(it),dl=m_dr.low.const_array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            auto const sum=remainder::Add({h(i,j,k),l(i,j,k)},{dh(i,j,k),dl(i,j,k)});h(i,j,k)=sum.hi;l(i,j,k)=sum.lo;
        });
    }return high.is_finite()&&low.is_finite();
}
}
