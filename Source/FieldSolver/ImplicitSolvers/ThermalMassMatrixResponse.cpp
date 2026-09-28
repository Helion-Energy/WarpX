/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ThermalMassMatrixResponse.H"
#include "ThermalFiniteFluxResponse.H"
#include "MassMatrixDensityProjection.H"
#include "EulerianThermalStage.H"
#include "Utils/WarpXConst.H"
#include <AMReX_Reduce.H>
#include <limits>
namespace warpx::thermal {
namespace {
using MF=amrex::MultiFab;using Real=amrex::Real;using PF=ThermalMassMatrixResponse::PairField;
namespace dd=mm_thermal_detail;
void Define(MF& out,MF const& in,int grow) {
    if(!out.isDefined())out.define(in.boxArray(),in.DistributionMap(),1,grow);
    AMREX_ALWAYS_ASSERT(remainder::Layout(out,in) && out.nGrowVect()==amrex::IntVect(grow));
    out.setVal(0.);
}
void Define(PF& out,MF const& in,int grow) {Define(out.high,in,grow);Define(out.low,in,grow);}
void Copy(MF& out,MF const& in) {MF::Copy(out,in,0,0,1,amrex::min(out.nGrowVect(),in.nGrowVect()));}
// Endpoint invalidation epochs identify each rank's local response lifetime.
// A last-rank invalidation may leave their numbers different after a rebuild;
// validate every local epoch collectively, then retain exact local binding.
bool LocalResponseEpoch(std::uint64_t value) {
    return remainder::All(value>0 && value<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max()));
}
bool Agreed(std::uint64_t value) {
    bool ok=value>0 && value<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max());
    if(!remainder::All(ok))return false;
    amrex::Long lo=static_cast<amrex::Long>(value),hi=lo;
    amrex::ParallelDescriptor::ReduceLongMin(lo);amrex::ParallelDescriptor::ReduceLongMax(hi);return lo==hi;
}
bool Finite(PF const& f) {return f.high.is_finite(0,1,f.high.nGrowVect()) && f.low.is_finite(0,1,f.low.nGrowVect());}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool Positive(dd::Pair value) noexcept {return value.hi>0. || (value.hi==0. && value.lo>0.);}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
dd::Pair Get(amrex::Array4<Real const> const& high,amrex::Array4<Real const> const& low,int i,int j,int k) {
    return {high(i,j,k),low(i,j,k)};
}
}
ThermalMassMatrixResponse::ThermalMassMatrixResponse(KineticThermalMoments const& map,bool finite_conduction)
 :m_map(map.m_geometry,map.m_cells,map.m_distribution,map.m_options) {
    int lo=finite_conduction,hi=lo;
    amrex::ParallelDescriptor::ReduceIntMin(lo);amrex::ParallelDescriptor::ReduceIntMax(hi);
    m_finite_selector_valid=lo==hi;
    if(m_finite_selector_valid&&finite_conduction)m_finite_flux=std::make_unique<ThermalFiniteFluxResponse>();
}
ThermalMassMatrixResponse::~ThermalMassMatrixResponse()=default;
bool ThermalMassMatrixResponse::Ready() const noexcept {
    return m_ready&&(!m_finite_flux||m_finite_flux->Ready({m_binding.field_base,m_binding.particle_response,m_binding.density_base},m_publication,m_sample));
}
void ThermalMassMatrixResponse::Invalidate() noexcept {
    if(m_finite_flux)m_finite_flux->Invalidate();
    m_captured=false;m_ready=false;m_retained_current=false;m_projection=nullptr;m_publication=0;m_sample=0;
    if(m_generation<std::numeric_limits<std::uint64_t>::max())++m_generation;
}
bool ThermalMassMatrixResponse::SameBinding(Binding value) const {
    bool const agreement=Agreed(value.field_base) && LocalResponseEpoch(value.particle_response) && Agreed(value.density_base);
    return agreement && remainder::All(m_captured && value.field_base==m_binding.field_base &&
        value.particle_response==m_binding.particle_response && value.density_base==m_binding.density_base);
}
bool ThermalMassMatrixResponse::CanSample(Binding binding,std::uint64_t sample) const {
    if(!SameBinding(binding)||!Agreed(sample))return false;
    bool valid=remainder::ArithmeticSupported() && m_projection && sample>m_sample;
    if(m_projection)valid=valid && m_projection->BaseEpoch()==binding.density_base;
    return remainder::All(valid);
}
bool ThermalMassMatrixResponse::ContextSupported(EulerianThermalStage const& stage) const {
    bool valid=remainder::ArithmeticSupported() && stage.m_transport && !stage.m_remainder_context &&
        stage.m_options.transport.fluid_reconstruction==0 && stage.m_options.transport.central_dissipation_entropy==0 &&
        stage.m_options.gamma==m_map.m_options.gamma && stage.m_options.dt>0. && stage.m_options.theta>0. &&
        stage.m_geometry.Domain()==m_map.m_geometry.Domain() && stage.m_geometry.Coord()==m_map.m_geometry.Coord() &&
        remainder::Layout(stage.m_density,m_map.m_density,1);
    for(int d=0;d<AMREX_SPACEDIM;++d)valid=valid &&
        stage.m_geometry.ProbLo(d)==m_map.m_geometry.ProbLo(d) && stage.m_geometry.ProbHi(d)==m_map.m_geometry.ProbHi(d) &&
        stage.m_geometry.isPeriodic(d)==m_map.m_geometry.isPeriodic(d);
#if !defined(WARPX_DIM_RZ)
    valid=false; // Cartesian implementation compiles; runtime admission is separate.
#endif
    if(!remainder::All(valid))return false;
    if(!stage.m_density.is_finite() || stage.m_density.min(0)<=0.)return false;
    if(!stage.m_live_source.is_finite() || !stage.m_source_derivative.is_finite() || !stage.m_kappa.is_finite())return false;
    if(stage.m_live_source.norminf(0)!=0. || stage.m_source_derivative.norminf(0)!=0.)return false;
    if(!m_finite_flux) {
        // The original companion remains source/conduction-free. Only the
        // separately owned provider supplies every finite conductive term.
        if(stage.m_kappa.norminf(0)!=0. || stage.m_kappa.norminf(1)!=0.)return false;
        for(auto const& q:stage.m_flux)if(q->norminf(0)!=0.)return false;
    } else {
        if(!remainder::All(stage.CompletedReceipt().Current()))return false;
        for(auto const& q:stage.m_flux)if(!q->is_finite())return false;
        // Capture/Apply below additionally validate the owned compiled laws,
        // exact material/BC receipt, source-zero producer and grow-3 images.
    }
    return true;
}
bool ThermalMassMatrixResponse::Inputs(MF const& energy,YeeCurrentView const& plasma,YeeCurrentView const& low) const {
    bool valid=remainder::ArithmeticSupported() && remainder::Layout(energy,m_map.m_density);
    for(int c=0;c<3;++c)valid=valid && plasma[c] &&
        (plasma[c] && remainder::Layout(*plasma[c],*m_map.m_current[c]));
    for(int c=0;c<3;++c) {
        valid=valid && bool(low[c])==bool(low[0]);
        if(low[c] && plasma[c])valid=valid && remainder::Layout(*low[c],*plasma[c]);
    }
    if(!remainder::All(valid))return false;
    if(!energy.is_finite())return false;
    for(auto const* p:plasma)if(!p->is_finite())return false;
    // Local checks have one collective result even if a caller supplies a
    // different optional-view selection on an empty rank.
    for(auto const* p:low)if(p)valid=p->is_finite(0,1,0,true)&&valid;
    return remainder::All(valid);
}
bool ThermalMassMatrixResponse::Capture(Base const& base) {
    Invalidate();
    if(!m_finite_selector_valid)return false; // immutable uniform constructor receipt
    if(!Agreed(base.binding.field_base) || !LocalResponseEpoch(base.binding.particle_response) ||
        !Agreed(base.binding.density_base) || !ContextSupported(base.stage) || !Inputs(base.energy,base.plasma,base.plasma_low))return false;
    bool valid=remainder::Layout(base.rho,m_map.m_node_density) && remainder::Layout(base.raw,m_map.m_density) &&
        remainder::Layout(base.correction,m_map.m_density) && remainder::Layout(base.moments.NodalNumberDensity(),m_map.m_node_density) &&
        base.projection.IsCaptured() && base.projection.BaseEpoch()==base.binding.density_base;
    for(int c=0;c<3;++c) {
        valid=valid && bool(base.vacuum_current[c])==bool(base.plasma_low[c]);
        if(base.vacuum_current[c])valid=valid && remainder::Layout(*base.vacuum_current[c],*base.plasma[c]);
        valid=valid && base.ion[c] && base.vacuum[c];
        if(base.ion[c] && base.vacuum[c])valid=valid && remainder::Layout(*base.ion[c],*m_map.m_current[c]) &&
            base.vacuum[c]->nComp()==1 && base.vacuum[c]->boxArray()==base.ion[c]->boxArray() &&
            base.vacuum[c]->DistributionMap()==base.ion[c]->DistributionMap();
    }
    for(int d=0;d<AMREX_SPACEDIM;++d)valid=valid && base.gamma[d] &&
        (base.gamma[d] && remainder::Layout(*base.gamma[d],base.stage.FaceVelocity(d)));
    if(!remainder::All(valid))return false;
    m_retained_current=base.plasma_low[0]!=nullptr;
    Define(m_rho0,base.rho,base.rho.nGrow());Define(m_delta_rho,base.rho,base.rho.nGrow());
    for(int c=0;c<3;++c) {
        Define(m_ji0[c],*base.ion[c],base.ion[c]->nGrow());Define(m_delta_ji[c],*base.ion[c],base.ion[c]->nGrow());
        Define(m_jp0[c],*base.plasma[c],0);Copy(m_jp0[c],*base.plasma[c]);
        Define(m_jp0_low[c],*base.plasma[c],0);
        if(base.plasma_low[c])Copy(m_jp0_low[c],*base.plasma_low[c]);
        Define(m_dj[c],*base.plasma[c],0);Define(m_djv[c],*base.plasma[c],0);
        if(!m_vacuum[c].isDefined())m_vacuum[c].define(base.vacuum[c]->boxArray(),base.vacuum[c]->DistributionMap(),1,0);
        amrex::iMultiFab::Copy(m_vacuum[c],*base.vacuum[c],0,0,1,0);
    }
    if(!base.projection.CopyLinearizationBase(m_rho0,{&m_ji0[0],&m_ji0[1],&m_ji0[2]},base.binding.density_base))return false;
    Define(m_node0,m_map.m_node_density,1);Copy(m_node0,base.moments.NodalNumberDensity());m_map.FillScalarImages(m_node0);
    Define(m_n0,m_map.m_density,1);Copy(m_n0,base.stage.Density());m_map.FillScalarImages(m_n0);
    Define(m_u0,base.energy,1);Copy(m_u0,base.energy);m_map.FillScalarImages(m_u0);
    Define(m_raw0,base.raw,0);Copy(m_raw0,base.raw);Define(m_correction0,base.correction,0);Copy(m_correction0,base.correction);
    for(auto* f:{&m_dn_node})Define(*f,m_node0,1);
    for(auto* f:{&m_dn,&m_du,&m_dt})Define(*f,m_n0,1);
    for(auto* f:{&m_dp,&m_dp_ohm})Define(*f,m_map.m_pressure,m_map.m_options.nodal_ghosts);
    for(auto* f:{&m_dru,&m_dcorrection})Define(*f,m_raw0,0);
    // Anchor the transform to the actual native vacuum residual. In retained
    // PMC arithmetic it is high((Jp_high,Jp_low)-Ji), not Jp_high-Ji.
    for(int c=0;c<3;++c)for(amrex::MFIter it(m_djv[c].high);it.isValid();++it) {
        auto out=m_djv[c].high.array(it);auto p=m_jp0[c].const_array(it),ion=m_ji0[c].const_array(it);auto mask=m_vacuum[c].const_array(it);
        auto actual=base.vacuum_current[c]?base.vacuum_current[c]->const_array(it):amrex::Array4<Real const>{};
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){out(i,j,k)=mask(i,j,k)?(actual?actual(i,j,k):p(i,j,k)-ion(i,j,k)):0.;});
    }
    ThermalFaceView jv;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        Define(m_gamma0[d],*base.gamma[d],0);Copy(m_gamma0[d],*base.gamma[d]);
        Define(m_nf0[d],*base.gamma[d],0);Define(m_jv_face0[d],*base.gamma[d],0);jv[d]=&m_jv_face0[d];
        for(auto* f:{&m_v0[d],&m_w0[d],&m_dgamma[d],&m_djv_face[d],&m_dnf[d],&m_dv[d],&m_dw[d],&m_df[d],&m_dtransform_flux[d]})Define(*f,*base.gamma[d],0);
    }
    m_map.RestrictCurrent({&m_djv[0].high,&m_djv[1].high,&m_djv[2].high},jv);
    // First scope has periodic or closed normal particle-current faces.
    // A nonzero physical wall flux declines here; it is never zeroed.
    int bad=0;auto const domain=m_map.m_geometry.Domain();
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        bool const periodic=m_map.m_geometry.isPeriodic(d);
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
        for(amrex::MFIter it(m_nf0[d]);it.isValid();++it) {
            auto n=m_n0.const_array(it),gamma=m_gamma0[d].const_array(it),jv0=m_jv_face0[d].const_array(it);
            auto vh=base.stage.FaceVelocity(d).const_array(it);auto nf=m_nf0[d].array(it);
            auto v=m_v0[d].high.array(it),vl=m_v0[d].low.array(it),wh=m_w0[d].high.array(it),wl=m_w0[d].low.array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                int const il=i-(d==0),jl=j-(d==1),kl=k-(d==2);int const q[3]={i,j,k};
                Real const face=.5*(n(i,j,k)+n(il,jl,kl));nf(i,j,k)=face;
                // Stage's stored high is retained; the division remainder is
                // explicitly reconciled rather than assumed zero. All finite
                // products below use BOTH parts of this same base quotient.
                auto velocity=dd::WithHigh(dd::Divide({gamma(i,j,k),0.},face),vh(i,j,k));
                v(i,j,k)=velocity.hi;vl(i,j,k)=velocity.lo;
                auto w=dd::Divide({jv0(i,j,k),0.},face);wh(i,j,k)=w.hi;wl(i,j,k)=w.lo;
                bool const boundary=!periodic && (q[d]==domain.smallEnd(d)||q[d]==domain.bigEnd(d)+1);
                return {!std::isfinite(face)||face<=0.||vh(i,j,k)!=gamma(i,j,k)/face||
                    (boundary&&(gamma(i,j,k)!=0.||jv0(i,j,k)!=0.))};
            });
        }
        bad=std::max(bad,amrex::get<0>(data.value()));
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);if(bad)return false;
    m_h=base.stage.Options().theta*base.stage.Options().dt;m_gm=base.stage.Options().gamma-1.;
    if(m_finite_flux&&!m_finite_flux->Capture(base.stage,{base.binding.field_base,base.binding.particle_response,base.binding.density_base}))return false;
    m_binding=base.binding;m_projection=&base.projection;m_captured=true;return true;
}
bool ThermalMassMatrixResponse::Apply(Binding binding,std::uint64_t publication,std::uint64_t sample,
    MassMatrixDensityProjection const& projection,MF const& energy,YeeCurrentView const& plasma,EulerianThermalStage const& stage,YeeCurrentView const& plasma_low) {
    m_ready=false;
    if(!SameBinding(binding)||!Agreed(publication)||!Agreed(sample)||!remainder::All(
        m_projection==&projection && bool(plasma_low[0])==m_retained_current && stage.Options().theta*stage.Options().dt==m_h &&
        stage.Options().gamma-1.==m_gm && (m_sample==0 || (publication>m_publication && sample>m_sample))) ||
        !ContextSupported(stage)||!Inputs(energy,plasma,plasma_low)) {Invalidate();return false;}
    if(!projection.CopyRetainedIncrement(m_delta_rho,{&m_delta_ji[0],&m_delta_ji[1],&m_delta_ji[2]},binding.density_base,publication)) {Invalidate();return false;}
    for(int c=0;c<3;++c)for(amrex::MFIter it(m_dj[c].high);it.isValid();++it) {
        auto p=plasma[c]->const_array(it),p0=m_jp0[c].const_array(it),ion=m_delta_ji[c].const_array(it);auto mask=m_vacuum[c].const_array(it);
        auto h=m_dj[c].high.array(it),l=m_dj[c].low.array(it),vh=m_djv[c].high.array(it),vl=m_djv[c].low.array(it);
        auto pl=plasma_low[c]?plasma_low[c]->const_array(it):amrex::Array4<Real const>{};
        auto pl0=m_jp0_low[c].const_array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            auto delta=dd::CurrentIncrement({p(i,j,k),pl?pl(i,j,k):0.},
                {p0(i,j,k),pl0(i,j,k)},ion(i,j,k));
            h(i,j,k)=delta.published.hi;l(i,j,k)=delta.published.lo;
            vh(i,j,k)=mask(i,j,k)?delta.vacuum.hi:0.;vl(i,j,k)=mask(i,j,k)?delta.vacuum.lo:0.;
        });
    }
    ThermalFaceView gh,gl,jh,jl;
    for(int d=0;d<AMREX_SPACEDIM;++d){gh[d]=&m_dgamma[d].high;gl[d]=&m_dgamma[d].low;jh[d]=&m_djv_face[d].high;jl[d]=&m_djv_face[d].low;}
    if(!m_map.RestrictCurrentIncrement({&m_dj[0].high,&m_dj[1].high,&m_dj[2].high},{&m_dj[0].low,&m_dj[1].low,&m_dj[2].low},gh,gl)||
        !m_map.RestrictCurrentIncrement({&m_djv[0].high,&m_djv[1].high,&m_djv[2].high},{&m_djv[0].low,&m_djv[1].low,&m_djv[2].low},jh,jl)) {Invalidate();return false;}
    Real const factor=-1./PhysConst::q_e; // the actual owned DTA coefficient
    for(int d=0;d<AMREX_SPACEDIM;++d)for(amrex::MFIter it(m_dgamma[d].high);it.isValid();++it) {
        auto h=m_dgamma[d].high.array(it),l=m_dgamma[d].low.array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto x=dd::Multiply({h(i,j,k),l(i,j,k)},factor);h(i,j,k)=x.hi;l(i,j,k)=x.lo;});
    }
    if(!DensityAndEOS(energy)||!FaceActions()||!ResidualActions()){Invalidate();return false;}
    if(m_finite_flux){
        ThermalFiniteFluxResponse::Binding const flux_binding{binding.field_base,binding.particle_response,binding.density_base};
        if(!m_finite_flux->Apply(stage,flux_binding,publication,sample,{m_du.high,m_du.low},{m_dn.high,m_dn.low}) ||
           !m_finite_flux->AddResidual(m_dru.high,m_dru.low,flux_binding,publication,sample)){Invalidate();return false;}
    }
    m_publication=publication;m_sample=sample;m_ready=true;return true;
}
bool ThermalMassMatrixResponse::DensityAndEOS(MF const& energy) {
    Real const floor=m_map.m_options.number_density_floor;
    int bad=0;
    amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
    for(amrex::MFIter it(m_dn_node.high);it.isValid();++it) {
        auto rho=m_rho0.const_array(it),dr=m_delta_rho.const_array(it),base=m_node0.const_array(it);
        auto h=m_dn_node.high.array(it),l=m_dn_node.low.array(it);
        op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
            Real const raw=rho(i,j,k),change=dr(i,j,k),n=raw/PhysConst::q_e;
            auto next=dd::Divide(dd::Sum(raw,change),PhysConst::q_e);
            auto gap=dd::Add(next,{-floor,0.});bool const active=n>floor;
            bool invalid=!std::isfinite(raw)||!std::isfinite(change)||raw<0.||base(i,j,k)!=amrex::max(n,floor)||
                (raw==0.&&change!=0.)||(raw>0.&&!Positive(dd::Sum(raw,change)))||
                (active&&!Positive(gap))||(!active&&Positive(gap))||(n==floor&&change!=0.);
            auto delta=active?dd::Divide({change,0.},PhysConst::q_e):dd::Pair{0.,0.};
            h(i,j,k)=delta.hi;l(i,j,k)=delta.lo;return {invalid};
        });
    }
    bad=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceIntMax(bad);if(bad)return false;
    m_map.FillScalarImages(m_dn_node.high);m_map.FillScalarImages(m_dn_node.low);
    if(!m_map.RestrictNativeMomentIncrement(m_dn_node.high,m_dn_node.low,m_dn.high,m_dn.low))return false;
    {
        amrex::ReduceOps<amrex::ReduceOpMax> positive_op;amrex::ReduceData<int> positive_data(positive_op);
        for(amrex::MFIter it(m_dn.high);it.isValid();++it) {
            auto n=m_n0.const_array(it),h=m_dn.high.const_array(it),l=m_dn.low.const_array(it);
            positive_op.eval(it.validbox(),positive_data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                return {!Positive(dd::Add({n(i,j,k),0.},Get(h,l,i,j,k)))};
            });
        }
        bad=amrex::get<0>(positive_data.value());amrex::ParallelDescriptor::ReduceIntMax(bad);if(bad)return false;
    }
    Real const gm=m_gm;
    for(amrex::MFIter it(m_du.high);it.isValid();++it) {
        auto u=energy.const_array(it),old=m_u0.const_array(it),n=m_n0.const_array(it);
        auto nh=m_dn.high.const_array(it),nl=m_dn.low.const_array(it);
        auto h=m_du.high.array(it),l=m_du.low.array(it),th=m_dt.high.array(it),tl=m_dt.low.array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            auto du=dd::Sum(u(i,j,k),-old(i,j,k));auto dn=Get(nh,nl,i,j,k);
            h(i,j,k)=du.hi;l(i,j,k)=du.lo;
            auto numerator=dd::Add(dd::Multiply(du,n(i,j,k)),dd::Negate(dd::Multiply(dn,old(i,j,k))));
            auto denominator=dd::Multiply(dd::Add({n(i,j,k),0.},dn),n(i,j,k));
            auto temperature=dd::Divide(dd::Multiply(dd::DividePair(numerator,denominator),gm),PhysConst::kb);
            th(i,j,k)=temperature.hi;tl(i,j,k)=temperature.lo;
        });
    }
    for(auto* f:{&m_du,&m_dt}){m_map.FillScalarImages(f->high);m_map.FillScalarImages(f->low);}
    // Pressure retains the actual native anchored mean and ratio-first finite
    // product. It is an independent EOS output, not a new low field-force path.
    for(amrex::MFIter it(m_dp.high);it.isValid();++it) {
        auto node=m_node0.const_array(it),n=m_n0.const_array(it),u=m_u0.const_array(it);
        auto dnh=m_dn_node.high.const_array(it),dnl=m_dn_node.low.const_array(it);
        auto nch=m_dn.high.const_array(it),ncl=m_dn.low.const_array(it),uh=m_du.high.const_array(it),ul=m_du.low.const_array(it);
        auto ph=m_dp.high.array(it),pl=m_dp.low.array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            auto term=[=] AMREX_GPU_DEVICE(int ci,int cj,int ck) {
                auto dnode=Get(dnh,dnl,i,j,k),dc=Get(nch,ncl,ci,cj,ck),du=Get(uh,ul,ci,cj,ck);
                auto numerator=dd::Add(dd::Multiply(dnode,n(ci,cj,ck)),dd::Negate(dd::Multiply(dc,node(i,j,k))));
                auto denominator=dd::Multiply(dd::Add({n(ci,cj,ck),0.},dc),n(ci,cj,ck));
                auto ratio=dd::Divide({node(i,j,k),0.},n(ci,cj,ck));
                auto dratio=dd::DividePair(numerator,denominator);
                return dd::FluxIncrement(du,dratio,{u(ci,cj,ck),0.},ratio);
            };
            auto base=term(i,j,k);dd::Pair sum{0.,0.};
            for(int bits=0;bits<(1<<AMREX_SPACEDIM);++bits) {
                int const di=bits&1,dj=(bits>>1)&1;
#if AMREX_SPACEDIM==3
                int const dk=(bits>>2)&1;
#else
                int const dk=0;
#endif
                sum=dd::Add(sum,dd::Add(term(i-di,j-dj,k-dk),dd::Negate(base)));
            }
            auto value=dd::Multiply(dd::Add(base,dd::Multiply(sum,1./static_cast<Real>(1<<AMREX_SPACEDIM))),gm);
            ph(i,j,k)=value.hi;pl(i,j,k)=value.lo;
        });
    }
    m_map.ApplyPressureIncrementImages(m_dp.high,m_dp.low,m_dp_ohm.high,m_dp_ohm.low);
    for(auto const* f:{&m_dn_node,&m_dn,&m_du,&m_dt,&m_dp,&m_dp_ohm})if(!Finite(*f))return false;
    return true;
}
bool ThermalMassMatrixResponse::FaceActions() {
    auto const domain=m_map.m_geometry.Domain();int bad=0;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        bool const periodic=m_map.m_geometry.isPeriodic(d);
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
        for(amrex::MFIter it(m_dv[d].high);it.isValid();++it) {
            auto n0=m_nf0[d].const_array(it),u0=m_u0.const_array(it);
            auto nh=m_dn.high.const_array(it),nl=m_dn.low.const_array(it),uh=m_du.high.const_array(it),ul=m_du.low.const_array(it);
            auto gh=m_dgamma[d].high.const_array(it),gl=m_dgamma[d].low.const_array(it);
            auto ch=m_djv_face[d].high.const_array(it),cl=m_djv_face[d].low.const_array(it);
            auto v0h=m_v0[d].high.const_array(it),v0l=m_v0[d].low.const_array(it),w0h=m_w0[d].high.const_array(it),w0l=m_w0[d].low.const_array(it);
            auto nfh=m_dnf[d].high.array(it),nfl=m_dnf[d].low.array(it);
            auto vh=m_dv[d].high.array(it),vl=m_dv[d].low.array(it),wh=m_dw[d].high.array(it),wl=m_dw[d].low.array(it);
            auto fh=m_df[d].high.array(it),fl=m_df[d].low.array(it),th=m_dtransform_flux[d].high.array(it),tl=m_dtransform_flux[d].low.array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                int const il=i-(d==0),jl=j-(d==1),kl=k-(d==2);int const q[3]={i,j,k};
                auto dn=dd::Multiply(dd::Add(Get(nh,nl,i,j,k),Get(nh,nl,il,jl,kl)),.5);
                auto du=dd::Multiply(dd::Add(Get(uh,ul,i,j,k),Get(uh,ul,il,jl,kl)),.5);
                auto uf=dd::Multiply(dd::Sum(u0(i,j,k),u0(il,jl,kl)),.5);
                auto v0=Get(v0h,v0l,i,j,k),w0=Get(w0h,w0l,i,j,k);
                auto gamma=Get(gh,gl,i,j,k),current=Get(ch,cl,i,j,k);
                bool const boundary=!periodic && (q[d]==domain.smallEnd(d)||q[d]==domain.bigEnd(d)+1);
                bool const invalid=!Positive(dd::Add({n0(i,j,k),0.},dn))||
                    (boundary&&(gamma.hi!=0.||gamma.lo!=0.||current.hi!=0.||current.lo!=0.));
                auto dv=dd::QuotientIncrement(dn,gamma,v0,n0(i,j,k));
                auto dw=dd::QuotientIncrement(dn,current,w0,n0(i,j,k));
                auto df=dd::FluxIncrement(du,dv,uf,v0),dt=dd::FluxIncrement(du,dw,uf,w0);
                nfh(i,j,k)=dn.hi;nfl(i,j,k)=dn.lo;vh(i,j,k)=dv.hi;vl(i,j,k)=dv.lo;
                wh(i,j,k)=dw.hi;wl(i,j,k)=dw.lo;fh(i,j,k)=df.hi;fl(i,j,k)=df.lo;th(i,j,k)=dt.hi;tl(i,j,k)=dt.lo;
                return {invalid};
            });
        }
        bad=std::max(bad,amrex::get<0>(data.value()));
        for(auto* f:{&m_dnf[d],&m_dv[d],&m_dw[d],&m_df[d],&m_dtransform_flux[d]}) {
            f->high.OverrideSync(m_map.m_geometry.periodicity());f->low.OverrideSync(m_map.m_geometry.periodicity());
            if(!Finite(*f))bad=1;
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);return bad==0;
}
bool ThermalMassMatrixResponse::ResidualActions() {
    auto const dx=m_map.m_geometry.CellSizeArray();Real const gm=m_gm,h=m_h,correction_factor=m_h/PhysConst::q_e;
    [[maybe_unused]] Real const rlo=m_map.m_geometry.ProbLo(0);
    for(amrex::MFIter it(m_dru.high);it.isValid();++it) {
        auto u=m_u0.const_array(it),uh=m_du.high.const_array(it),ul=m_du.low.const_array(it);
        auto rh=m_dru.high.array(it),rl=m_dru.low.array(it),ch=m_dcorrection.high.array(it),cl=m_dcorrection.low.array(it);
        // Six fields: base v, delta v, delta F; base w, delta w, delta(Uface*w).
        amrex::GpuArray<amrex::GpuArray<amrex::Array4<Real const>,AMREX_SPACEDIM>,6> hi{},lo{};
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            PF const* fields[6]={&m_v0[d],&m_dv[d],&m_df[d],&m_w0[d],&m_dw[d],&m_dtransform_flux[d]};
            for(int q=0;q<6;++q){hi[q][d]=fields[q]->high.const_array(it);lo[q][d]=fields[q]->low.const_array(it);}
        }
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            dd::Pair div[6]{};
            for(int q=0;q<6;++q)for(int d=0;d<AMREX_SPACEDIM;++d) {
                int const ip=i+(d==0),jp=j+(d==1),kp=k+(d==2);
                auto positive=Get(hi[q][d],lo[q][d],ip,jp,kp),negative=Get(hi[q][d],lo[q][d],i,j,k);Real denominator=dx[d];
#if defined(WARPX_DIM_RZ)
                if(d==0) {
                    Real const rp=rlo+(i+1)*dx[0],rm=rlo+i*dx[0],r=rlo+(i+.5)*dx[0];
                    positive=dd::Multiply(positive,rp);negative=dd::Multiply(negative,rm);denominator=r*dx[0];
                }
#endif
                div[q]=dd::Add(div[q],dd::Divide(dd::Add(positive,dd::Negate(negative)),denominator));
            }
            auto du=Get(uh,ul,i,j,k);
            auto work=dd::Add(dd::Add(dd::Multiply(div[1],u(i,j,k)),dd::Product(du,div[0])),dd::Product(du,div[1]));
            auto dwork=dd::Add(dd::Add(dd::Multiply(div[4],u(i,j,k)),dd::Product(du,div[3])),dd::Product(du,div[4]));
            auto raw=dd::Add(du,dd::Multiply(dd::Add(div[2],dd::Multiply(work,gm)),h));
            auto correction=dd::Multiply(dd::Add(div[5],dd::Multiply(dwork,gm)),correction_factor);
            rh(i,j,k)=raw.hi;rl(i,j,k)=raw.lo;ch(i,j,k)=correction.hi;cl(i,j,k)=correction.lo;
        });
    }
    return Finite(m_dru)&&Finite(m_dcorrection);
}
bool ThermalMassMatrixResponse::CopyCompanion(MF& output,MF const& represented,Binding binding,
    std::uint64_t publication,std::uint64_t sample) const {
    if(!SameBinding(binding)||!Agreed(publication)||!Agreed(sample))return false;
    bool valid=remainder::ArithmeticSupported() && m_ready && m_projection && m_sample==sample && m_publication==publication &&
        remainder::Layout(output,m_raw0) && remainder::Layout(represented,m_raw0) && !remainder::Overlap(output,represented);
    if(m_projection)valid=valid && m_projection->BaseEpoch()==binding.density_base && m_projection->PublicationEpoch()==publication && m_projection->DensityIncrementReady();
    if(m_finite_flux)valid=valid&&m_finite_flux->Ready({binding.field_base,binding.particle_response,binding.density_base},publication,sample);
    for(auto const* input:{&m_raw0,&m_correction0,&m_dru.high,&m_dru.low,&m_dcorrection.high,&m_dcorrection.low})valid=valid && !remainder::Overlap(output,*input);
    if(!remainder::All(valid) || !represented.is_finite())return false;
    for(amrex::MFIter it(output);it.isValid();++it) {
        auto raw=m_raw0.const_array(it),correction=m_correction0.const_array(it),high=represented.const_array(it);
        auto rh=m_dru.high.const_array(it),rl=m_dru.low.const_array(it),ch=m_dcorrection.high.const_array(it),cl=m_dcorrection.low.const_array(it);auto o=output.array(it);
        amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            Real const base=raw(i,j,k)+correction(i,j,k); // exact ordinary base publication
            auto delta=dd::Add(Get(rh,rl,i,j,k),Get(ch,cl,i,j,k));
            o(i,j,k)=dd::WithHigh(dd::Add({base,0.},delta),high(i,j,k)).lo;
        });
    }
    return output.is_finite();
}
} // namespace warpx::thermal
