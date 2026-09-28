/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#include "NativePairedDarwinFields.H"
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "NativeLongitudinalProducerCertificate.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <limits>

namespace warpx::darwin {
namespace {
using R=amrex::Real;using MF=amrex::MultiFab;
using CV=NativeLongitudinalProducerCertificate::ConstVector;
using V=NativeLongitudinalProducerCertificate::Vector;
using Triple=std::array<MF,3>;
constexpr R eps=std::numeric_limits<R>::epsilon();
AMREX_GPU_HOST_DEVICE bool Normal(R x) { return std::isfinite(x)&&(x==0.||std::abs(x)>=std::numeric_limits<R>::min()); }
// All bound arithmetic rounds outward. FTZ/underflow cannot silently erase
// a positive term: unsupported nonzero subnormals invalidate this certificate.
AMREX_GPU_HOST_DEVICE R Up(R x) {
    if(x==0.)return 0.;
    if(!Normal(x)||x<0.)return std::numeric_limits<R>::infinity();
    return std::nextafter(x,std::numeric_limits<R>::infinity());
}
AMREX_GPU_HOST_DEVICE R Add(R a,R b) {
    if(!Normal(a)||!Normal(b)||a<0.||b<0.)return std::numeric_limits<R>::infinity();
    return Up(a+b);
}
AMREX_GPU_HOST_DEVICE R Mul(R a,R b) {
    if(!Normal(a)||!Normal(b)||a<0.||b<0.)return std::numeric_limits<R>::infinity();
    if(a==0.||b==0.)return 0.;
    R x=a*b;return x==0.?std::numeric_limits<R>::infinity():Up(x);
}
AMREX_GPU_HOST_DEVICE R Gamma(int n) { return Up((n*eps)/(1.-n*eps)); }
// Exact coefficient closure for the represented native relaxation weights.
// TwoSum preserves the low part of a+b; this is bound arithmetic only, not
// the excluded compensated physical EL update. Native weights are untouched.
R CoefficientClosure(R a,R b) {
    if(!Normal(a)||!Normal(b)||a<=0.||a>1.||b!=(1.-a))return std::numeric_limits<R>::infinity();
    volatile R const hi=a+b;
    volatile R const virtual_b=hi-a;
    volatile R const virtual_a=hi-virtual_b;
    volatile R const da=a-virtual_a,db=b-virtual_b;
    R const low=da+db;
    if(hi!=1.||!Normal(virtual_b)||!Normal(virtual_a)||!Normal(da)||!Normal(db)||!Normal(low))
        return std::numeric_limits<R>::infinity();
    return std::abs(low);
}
AMREX_GPU_HOST_DEVICE R AffineRound(R a,R x,R b,R z) {
    return Mul(Gamma(3),Add(Mul(std::abs(a),std::abs(x)),Mul(std::abs(b),std::abs(z))));
}
AMREX_GPU_HOST_DEVICE int Axis(int component) {
#if defined(WARPX_DIM_RZ)
    return component==0?0:component==2?1:-1;
#else
    return component;
#endif
}
CV Const(Triple const& a){return {&a[0],&a[1],&a[2]};}
CV Const(V const& a){return {a[0],a[1],a[2]};}
void Copy(MF& a,MF const& b){MF::Copy(a,b,0,0,1,amrex::min(a.nGrowVect(),b.nGrowVect()));}
void Copy(Triple& a,CV const& b){for(int c=0;c<3;++c)Copy(a[c],*b[c]);}
void Negate(MF& field){
    for(amrex::MFIter mfi(field);mfi.isValid();++mfi){auto a=field.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=-a(i,j,k);});}
}
bool Equal(MF const& a,MF const& b){
    if(a.boxArray()!=b.boxArray()||a.DistributionMap()!=b.DistributionMap()||a.nComp()!=b.nComp()||a.nGrowVect()!=b.nGrowVect())return false;
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(amrex::MFIter mfi(a);mfi.isValid();++mfi){auto const* x=reinterpret_cast<unsigned char const*>(a[mfi].dataPtr());auto const* y=reinterpret_cast<unsigned char const*>(b[mfi].dataPtr());
        amrex::For(a[mfi].size()*sizeof(R),[=] AMREX_GPU_DEVICE(std::size_t n){if(x[n]!=y[n])amrex::HostDevice::Atomic::Add(invalid,1);});}
    amrex::Gpu::synchronize();return bad.dataValue()==0;
}
// Exact-zero channels must reject NaN explicitly: max norms can discard it.
bool Zero(MF const& field,amrex::IntVect const& grow){
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(amrex::MFIter mfi(field);mfi.isValid();++mfi){auto a=field.const_array(mfi);
        amrex::ParallelFor(amrex::grow(mfi.validbox(),grow),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            if(!std::isfinite(a(i,j,k))||a(i,j,k)!=0.)amrex::HostDevice::Atomic::Add(invalid,1);
        });
    }
    amrex::Gpu::synchronize();return bad.dataValue()==0;
}
bool Collective(bool x){amrex::ParallelDescriptor::ReduceBoolAnd(x);return x;}
}
struct NativeLongitudinalProducerCertificate::Impl {
    WarpX* sim;amrex::Geometry geometry;
    Triple e,er,d,dr,old_e,old_er,old_d,old_dr,previous_er,gradient_radius;
    Triple delta_er,previous_delta_er,paired_low;
    bool paired=NativePairedDarwinFields::Requested(),paired_scalar_ready=false;
    MF paired_phi,paired_phi_low,paired_phi_origin;
    enum class Relative { None, Increment, Blend, Endpoint };
    MF scalar_synced,scalar_difference,curl_bounds;
    bool valid=true,initial_bound=false,stage=false,event_appended=false;R time=0.;std::uint64_t epoch=0,operations=0;
    explicit Impl(WarpX& w):sim(&w),geometry(w.get_pointer_HybridPICModel()->ElectronThermalGeometry()){
        auto el=w.m_fields.get_alldirs("hybrid_E_long_fp",0);
        auto const& phi=*w.m_fields.get("hybrid_phi_darwin_fp",0);
        scalar_synced.define(phi.boxArray(),phi.DistributionMap(),1,phi.nGrowVect());
        scalar_difference.define(phi.boxArray(),phi.DistributionMap(),1,0);
        if(paired)for(auto* f:{&paired_phi,&paired_phi_low,&paired_phi_origin}){f->define(phi.boxArray(),phi.DistributionMap(),1,phi.nGrowVect());f->setVal(0.);}
        auto const& bt=*w.m_fields.get(warpx::fields::FieldType::Bfield_fp,ablastr::fields::Direction{1},0);
        curl_bounds.define(bt.boxArray(),bt.DistributionMap(),1,0);
        for(int c=0;c<3;++c){
            for(auto* f:{&e[c],&er[c],&old_e[c],&old_er[c],&previous_er[c],&gradient_radius[c],&delta_er[c],&previous_delta_er[c]}){f->define(el[c]->boxArray(),el[c]->DistributionMap(),1,el[c]->nGrowVect());f->setVal(0.);}
            if(paired){paired_low[c].define(el[c]->boxArray(),el[c]->DistributionMap(),1,el[c]->nGrowVect());paired_low[c].setVal(0.);}
            for(auto* f:{&d[c],&dr[c],&old_d[c],&old_dr[c]}){f->define(el[c]->boxArray(),el[c]->DistributionMap(),1,1);f->setVal(0.);}
        }
    }
    bool OpenProducer(){valid=Collective(valid&&!event_appended);return valid;}
    void Finish(amrex::Gpu::DeviceScalar<int>& bad){amrex::Gpu::synchronize();valid=Collective(valid&&bad.dataValue()==0);++operations;}
    void Sync(Triple& radius){for(auto& f:radius){f.OverrideSync(geometry.periodicity());f.FillBoundary(geometry.periodicity());}}
    // The recorded scalar must be one global nodal producer, including exact
    // shared/periodic owner aliases. Compare private copies; never sync inputs.
    bool ScalarOwners(MF const& phi){
        Copy(scalar_synced,phi);scalar_synced.OverrideSync(geometry.periodicity());
        amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
        for(amrex::MFIter mfi(phi);mfi.isValid();++mfi){auto a=phi.const_array(mfi),b=scalar_synced.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                if(!Normal(a(i,j,k))||!Normal(b(i,j,k))||a(i,j,k)!=b(i,j,k))
                    amrex::HostDevice::Atomic::Add(invalid,1);
            });
        }
        amrex::Gpu::synchronize();return bad.dataValue()==0;
    }
    void Gradient(MF const& phi,CV const& value){
        bool supported=phi.ixType().nodeCentered()&&phi.nComp()==1&&
            phi.boxArray()==scalar_synced.boxArray()&&phi.DistributionMap()==scalar_synced.DistributionMap()&&
            phi.nGrowVect().allGE(0);
        valid=Collective(valid&&supported);if(!valid)return;
        bool const owners=ScalarOwners(phi);valid=Collective(valid&&owners);if(!valid)return;
        // A valid E edge requires phi at that node and its one forward node.
        // These are both valid nodal rows of the same cell FAB, including the
        // physical axis/rmax and both periodic seam images; no ghost extension.
        for(int c=0;c<3;++c){
            supported=value[c]&&value[c]->boxArray()==e[c].boxArray()&&
                value[c]->DistributionMap()==e[c].DistributionMap()&&value[c]->nComp()==1;
            valid=Collective(valid&&supported);if(!valid)return;
            int axis=Axis(c);if(axis<0)continue;
            for(amrex::MFIter mfi(e[c]);mfi.isValid();++mfi){auto needed=mfi.validbox();needed.growHi(axis,1);
                supported=phi.boxArray()[mfi.index()].contains(needed)&&supported;}
        }
        valid=Collective(valid&&supported);if(!valid)return;
        amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();auto const spacing=geometry.InvCellSizeArray();
        for(int c=0;c<3;++c){int const axis=Axis(c);R const inverse=axis<0?0.:spacing[axis];gradient_radius[c].setVal(0.);
            for(amrex::MFIter mfi(gradient_radius[c]);mfi.isValid();++mfi){auto out=gradient_radius[c].array(mfi);auto p=phi.const_array(mfi),actual=value[c]->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    if(axis<0){if(actual(i,j,k)!=0.)amrex::HostDevice::Atomic::Add(invalid,1);return;}
                    amrex::IntVect here(AMREX_D_DECL(i,j,k)),next=here;next[axis]++;
                    R const a=p(here),b=p(next),difference=b-a,y=difference*inverse;
                    R const bound=Mul(Gamma(2),Mul(std::abs(inverse),Add(std::abs(a),std::abs(b))));
                    if(!Normal(a)||!Normal(b)||!Normal(difference)||!Normal(y)||!Normal(actual(i,j,k))||!Normal(bound)||actual(i,j,k)!=y)amrex::HostDevice::Atomic::Add(invalid,1);
                    // No measured mismatch is used to enlarge the radius.
                    out(i,j,k)=bound;
                });
            }
        }
        Finish(bad);Sync(gradient_radius);
    }
    void RawRelativeRadius(){
        amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
        for(int c=0;c<3;++c)for(amrex::MFIter mfi(delta_er[c]);mfi.isValid();++mfi){
            auto out=delta_er[c].array(mfi);auto current=er[c].const_array(mfi),origin=old_er[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                R const bound=Add(current(i,j,k),origin(i,j,k));out(i,j,k)=bound;
                if(!Normal(bound))amrex::HostDevice::Atomic::Add(invalid,1);
            });
        }
        Finish(bad);Sync(delta_er);
    }
    void Affine(Triple& values,Triple& radii,CV const& x,CV const& xr,CV const& z,CV const& zr,CV const& actual,R a,R b,
                Relative relative=Relative::None){
        R const closure=relative==Relative::Blend?CoefficientClosure(a,b):0.;
        valid=Collective(valid&&Normal(a)&&Normal(b)&&Normal(closure));if(!valid)return;
        amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
        for(int c=0;c<3;++c){
            for(amrex::MFIter mfi(radii[c]);mfi.isValid();++mfi){auto out=radii[c].array(mfi);auto xx=x[c]->const_array(mfi),zz=z[c]->const_array(mfi),rx=xr[c]->const_array(mfi),rz=zr[c]->const_array(mfi),y=actual[c]->const_array(mfi);
                auto dr_out=delta_er[c].array(mfi);auto dr_x=delta_er[c].const_array(mfi),dr_z=previous_delta_er[c].const_array(mfi),origin=old_er[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){R const left=xx(i,j,k),right=zz(i,j,k);
                    volatile R const ax=a*left,bz=b*right;
                    R const expected=ax+bz,fa=std::fma(a,left,bz),fb=std::fma(b,right,ax);
                    R const round=AffineRound(a,left,b,right);
                    R bound=Add(Add(Mul(std::abs(a),rx(i,j,k)),Mul(std::abs(b),rz(i,j,k))),round);
                    bool relative_valid=true;
                    if(relative!=Relative::None){
                        // Both proofs use the SAME ideal a*psi_x+b*psi_z.
                        // With O=G(psi0)+e0, the anchored error is
                        // a*delta_x+b*delta_z+(a+b-1)*e0+rounding.
                        // Increment uses psi_z for the added gradient; endpoint
                        // uses z=O, delta_z=0 and exact coefficient sum one.
                        R const left_radius=dr_x(i,j,k);
                        R const right_radius=relative==Relative::Increment?rz(i,j,k):
                            relative==Relative::Blend?dr_z(i,j,k):0.;
                        R const delta=Add(Add(Mul(std::abs(a),left_radius),Mul(std::abs(b),right_radius)),
                            Add(Mul(closure,origin(i,j,k)),round));
                        R const anchored=Add(origin(i,j,k),delta);
                        relative_valid=Normal(delta)&&Normal(anchored)&&Normal(bound);
                        dr_out(i,j,k)=delta;bound=std::min(bound,anchored);
                    }
                    bool const expression=y(i,j,k)==expected||y(i,j,k)==fa||y(i,j,k)==fb;
                    if(!Normal(left)||!Normal(right)||!Normal(ax)||!Normal(bz)||
                       (a!=0.&&left!=0.&&ax==0.)||(b!=0.&&right!=0.&&bz==0.)||
                       !Normal(expected)||!Normal(y(i,j,k))||!Normal(bound)||!expression||!relative_valid)amrex::HostDevice::Atomic::Add(invalid,1);
                    out(i,j,k)=bound;});
            }
        }
        Finish(bad);Copy(values,actual);Sync(radii);if(relative!=Relative::None)Sync(delta_er);
    }
};
NativeLongitudinalProducerCertificate::NativeLongitudinalProducerCertificate(WarpX& w):m_impl(std::make_unique<Impl>(w)){}
NativeLongitudinalProducerCertificate::~NativeLongitudinalProducerCertificate()=default;
bool NativeLongitudinalProducerCertificate::Valid()const noexcept{return m_impl&&m_impl->valid;}
void NativeLongitudinalProducerCertificate::Invalidate()noexcept{m_impl->valid=false;}
std::shared_ptr<NativeLongitudinalProducerCertificate> NativeLongitudinalProducerCertificate::Seed(WarpX& w){
    auto p=std::shared_ptr<NativeLongitudinalProducerCertificate>(new NativeLongitudinalProducerCertificate(w));
    p->RawGradient(*w.m_fields.get("hybrid_phi_darwin_fp",0),w.m_fields.get_alldirs("hybrid_E_long_fp",0));return p;
}
bool NativeLongitudinalProducerCertificate::BindInitialDisplacement(R time,std::uint64_t epoch){
    auto& s=*m_impl;auto d=s.sim->m_fields.get_alldirs("diagnostic_D_endpoint_fp",0);bool zero=true;
    for(int c=0;c<3;++c)zero=Zero(*d[c],d[c]->nGrowVect())&&zero;
    s.valid=Collective(s.valid&&zero);Copy(s.d,Const(d));s.time=time;s.epoch=epoch;s.initial_bound=s.valid;return s.valid;
}
std::shared_ptr<NativeLongitudinalProducerCertificate> NativeLongitudinalProducerCertificate::Fork(R time,std::uint64_t epoch)const{
    auto const& source=*m_impl;auto p=std::shared_ptr<NativeLongitudinalProducerCertificate>(new NativeLongitudinalProducerCertificate(*source.sim));auto& s=*p->m_impl;
    Copy(s.e,Const(source.e));Copy(s.er,Const(source.er));Copy(s.d,Const(source.d));Copy(s.dr,Const(source.dr));
    Copy(s.old_e,Const(source.e));Copy(s.old_er,Const(source.er));Copy(s.old_d,Const(source.d));Copy(s.old_dr,Const(source.dr));
    if(s.paired){Copy(s.paired_low,Const(source.paired_low));s.paired_scalar_ready=false;}
    s.valid=source.valid;s.initial_bound=source.initial_bound;s.operations=source.operations;s.time=time;s.epoch=epoch;return p;
}
bool NativeLongitudinalProducerCertificate::MatchesElectric()const{
    auto const& s=*m_impl;bool ok=s.valid;
    auto actual=s.sim->m_fields.get_alldirs("hybrid_E_long_fp",0);
    for(int c=0;c<3;++c)ok=Equal(s.e[c],*actual[c])&&ok;
    if(s.paired) {
        auto* owner=NativeActivePairedFields(*s.sim);
        for(int c=0;c<3;++c)ok=(owner?Equal(s.paired_low[c],*owner->LongitudinalLow()[c]):Zero(s.paired_low[c],s.paired_low[c].nGrowVect()))&&ok;
        if(owner&&s.paired_scalar_ready){bool const phi=Equal(s.paired_phi,*s.sim->m_fields.get("hybrid_phi_darwin_fp",0));bool const low=Equal(s.paired_phi_low,owner->ScalarLow());ok=phi&&low&&ok;}
    }
    return Collective(ok);
}
void NativeLongitudinalProducerCertificate::BeginOuter(){auto& s=*m_impl;if(!s.OpenProducer())return;s.valid=MatchesElectric();Copy(s.previous_er,Const(s.er));Copy(s.previous_delta_er,Const(s.delta_er));}
void NativeLongitudinalProducerCertificate::RawGradient(MF const& phi,V const& actual){auto& s=*m_impl;if(!s.OpenProducer())return;s.Gradient(phi,Const(actual));Copy(s.er,Const(s.gradient_radius));Copy(s.e,Const(actual));s.RawRelativeRadius();}
void NativeLongitudinalProducerCertificate::SchurAddition(MF const& phi,CV const& gradient,V const& actual){
    auto& s=*m_impl;if(!s.OpenProducer())return;s.Gradient(phi,gradient);s.Affine(s.e,s.er,Const(s.e),Const(s.er),gradient,Const(s.gradient_radius),Const(actual),1.,1.,Impl::Relative::Increment);
}
void NativeLongitudinalProducerCertificate::Relax(V const& previous,R omega){
    auto& s=*m_impl;if(!s.OpenProducer())return;auto actual=s.sim->m_fields.get_alldirs("hybrid_E_long_fp",0);
    s.Affine(s.e,s.er,Const(s.e),Const(s.er),Const(previous),Const(s.previous_er),Const(actual),omega,1.-omega,Impl::Relative::Blend);
}
namespace {
// NVCC requires the parent of an extended device lambda to be accessible.
// This free helper retains the exact original private method kernel body.
void ProducerStage(NativeLongitudinalProducerCertificate::Impl& s,R dt,R theta){
    if(!s.OpenProducer())return;s.valid=Collective(s.valid&&theta==.5&&dt>0.&&Normal(dt));if(!s.valid)return;
    auto actual=s.sim->m_fields.get_alldirs("diagnostic_D_stage_fp",0);R const factor=PhysConst::epsilon_0/(theta*dt);
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int c=0;c<3;++c){for(amrex::MFIter mfi(s.dr[c]);mfi.isValid();++mfi){auto out=s.dr[c].array(mfi);auto e=s.e[c].const_array(mfi),e0=s.old_e[c].const_array(mfi),er=s.er[c].const_array(mfi),er0=s.old_er[c].const_array(mfi),delta=s.delta_er[c].const_array(mfi),a=actual[c]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){R const diff=e(i,j,k)-e0(i,j,k),expected=factor*diff;
            R const round=Mul(Gamma(2),Mul(std::abs(factor),Add(std::abs(e(i,j,k)),std::abs(e0(i,j,k)))));
            R const absolute=Add(Mul(std::abs(factor),Add(er(i,j,k),er0(i,j,k))),round);
            R const anchored=Add(Mul(std::abs(factor),delta(i,j,k)),round);
            R const bound=std::min(absolute,anchored);
            if(!Normal(absolute)||!Normal(anchored)||!Normal(diff)||!Normal(expected)||!Normal(a(i,j,k))||!Normal(bound)||a(i,j,k)!=expected)
                amrex::HostDevice::Atomic::Add(invalid,1);
            out(i,j,k)=bound;});}
    }
    s.Finish(bad);Copy(s.d,Const(actual));s.Sync(s.dr);s.stage=true;
}
}
namespace {
namespace paired_arithmetic=warpx::ohm::compensated;
using Pair=paired_arithmetic::Pair;
// This bounded twofold producer rejects operands so small that a 106-bit
// product remainder could underflow even though its high is normal. The
// bound is derived from binary64 min/epsilon^2; it is not a physical floor.
AMREX_GPU_HOST_DEVICE bool PairOperand(R x) {
    R const threshold=std::numeric_limits<R>::min()/(eps*eps);
    return Normal(x)&&(x==0.||std::abs(x)>=threshold);
}
AMREX_GPU_HOST_DEVICE bool PairFinite(Pair x) {return Normal(x.hi)&&Normal(x.lo);}
AMREX_GPU_HOST_DEVICE R PairAddRound(Pair a,Pair b) {
    using namespace paired_arithmetic;
    if(!PairOperand(a.hi)||!PairOperand(a.lo)||!PairOperand(b.hi)||!PairOperand(b.lo))return std::numeric_limits<R>::infinity();
    auto first=Sum(a.hi,b.hi);R const t=first.lo+a.lo,q=t+b.lo;auto result=Sum(first.hi,q);
    if(!PairFinite(first)||!Normal(t)||!Normal(q)||!PairFinite(result))return std::numeric_limits<R>::infinity();
    // Both TwoSum operations are exact in the admitted range. Only the two
    // explicitly ordered low additions contribute forward rounding.
    return Add(Mul(Gamma(1),Add(std::abs(first.lo),std::abs(a.lo))),
               Mul(Gamma(1),Add(std::abs(t),std::abs(b.lo))));
}
AMREX_GPU_HOST_DEVICE R PairMultiplyRound(Pair a,R b) {
    if(!PairOperand(a.hi)||!PairOperand(a.lo)||!PairOperand(b))return std::numeric_limits<R>::infinity();
    R const product=a.hi*b,small=a.lo*b,error=std::fma(a.hi,b,-product),sum=error+small;
    if(!PairOperand(product)||!Normal(small)||!Normal(error)||!Normal(sum)||
        (a.hi!=0.&&b!=0.&&product==0.)||(a.lo!=0.&&b!=0.&&small==0.))return std::numeric_limits<R>::infinity();
    return Add(Mul(Gamma(1),Mul(std::abs(a.lo),std::abs(b))),
               Mul(Gamma(1),Add(std::abs(error),std::abs(small))));
}
void RecordPairedScalar(NativeLongitudinalProducerCertificate::Impl& s,MF const* increment,R multiplier){
    auto* owner=NativeActivePairedFields(*s.sim);
    s.valid=Collective(s.valid&&s.paired_scalar_ready&&owner);if(!s.valid)return;
    auto const& actual=*s.sim->m_fields.get("hybrid_phi_darwin_fp",0);auto const& actual_low=owner->ScalarLow();
    bool const owners=s.ScalarOwners(actual),low_owners=s.ScalarOwners(actual_low);
    s.valid=Collective(s.valid&&owners&&low_owners);if(!s.valid)return;
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(amrex::MFIter mfi(s.paired_phi);mfi.isValid();++mfi){auto before=s.paired_phi.const_array(mfi),lo=s.paired_phi_low.const_array(mfi),origin=s.paired_phi_origin.const_array(mfi),a=actual.const_array(mfi),al=actual_low.const_array(mfi);auto change=increment?increment->const_array(mfi):amrex::Array4<R const>{};bool add=increment!=nullptr;
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            Pair old{before(i,j,k),lo(i,j,k)};Pair next;
            if(add)next=paired_arithmetic::Add(old,paired_arithmetic::Multiply({change(i,j,k),0.},multiplier));
            else next=paired_arithmetic::Add({origin(i,j,k),0.},paired_arithmetic::Multiply(paired_arithmetic::Add(old,{-origin(i,j,k),0.}),multiplier));
            if(!PairFinite(next)||a(i,j,k)!=next.hi||al(i,j,k)!=next.lo)amrex::HostDevice::Atomic::Add(invalid,1);
        });}
    s.Finish(bad);Copy(s.paired_phi,actual);Copy(s.paired_phi_low,actual_low);
}
void RecordPairedSchur(NativeLongitudinalProducerCertificate::Impl& s,MF const& phi,CV const& gradient,V const& actual,R omega){
    if(!s.OpenProducer())return;
    auto* owner=NativeActivePairedFields(*s.sim);
    s.valid=Collective(s.valid&&s.paired&&owner&&Normal(omega)&&omega>0.&&omega<=1.);if(!s.valid)return;
    s.Gradient(phi,gradient);if(!s.valid)return;
    RecordPairedScalar(s,&phi,omega);if(!s.valid)return;
    auto low=owner->LongitudinalLow();auto spacing=s.geometry.InvCellSizeArray();
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int c=0;c<3;++c){int axis=Axis(c);R inverse=axis<0?0.:spacing[axis];
        for(amrex::MFIter mfi(s.er[c]);mfi.isValid();++mfi){auto radius=s.er[c].array(mfi),delta=s.delta_er[c].array(mfi);auto x=s.e[c].const_array(mfi),xl=s.paired_low[c].const_array(mfi),g=gradient[c]->const_array(mfi),gr=s.gradient_radius[c].const_array(mfi),p=phi.const_array(mfi),y=actual[c]->const_array(mfi),yl=low[c]->const_array(mfi),origin=s.old_er[c].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                using namespace paired_arithmetic;
                Pair before{x(i,j,k),xl(i,j,k)};
                Pair raw=WithHigh(Gradient(p,i,j,k,axis,inverse),g(i,j,k));
                Pair correction=paired_arithmetic::Multiply(raw,omega),after=paired_arithmetic::Add(before,correction);
                // Native gradient high has its predeclared arithmetic radius.
                // Adding the known constructed low yields a conservative
                // bound for the SAME scalar producer, without a residual fit.
                R const cr=warpx::darwin::Add(Mul(std::abs(omega),warpx::darwin::Add(gr(i,j,k),std::abs(raw.lo))),PairMultiplyRound(raw,omega));
                R const rounding=PairAddRound(before,correction);
                R const dr=warpx::darwin::Add(delta(i,j,k),warpx::darwin::Add(cr,rounding));
                R const absolute=warpx::darwin::Add(radius(i,j,k),warpx::darwin::Add(cr,rounding));
                R const bound=std::min(absolute,warpx::darwin::Add(origin(i,j,k),dr));
                if(!PairFinite(raw)||!PairFinite(correction)||!PairFinite(after)||!Normal(cr)||!Normal(rounding)||!Normal(dr)||!Normal(bound)||y(i,j,k)!=after.hi||yl(i,j,k)!=after.lo)amrex::HostDevice::Atomic::Add(invalid,1);
                radius(i,j,k)=bound;delta(i,j,k)=dr;
            });}
    }
    s.Finish(bad);Copy(s.e,Const(actual));Copy(s.paired_low,low);s.Sync(s.er);s.Sync(s.delta_er);
}
void RecordPairedStage(NativeLongitudinalProducerCertificate::Impl& s,R dt,R theta){
    if(!s.OpenProducer())return;auto* owner=NativeActivePairedFields(*s.sim);
    s.valid=Collective(s.valid&&s.paired&&owner&&theta==.5&&Normal(dt)&&dt>0.);if(!s.valid)return;
    R factor=PhysConst::epsilon_0/(theta*dt);auto actual=s.sim->m_fields.get_alldirs("diagnostic_D_stage_fp",0);auto low=owner->DisplacementLow();
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(s.dr[c]);mfi.isValid();++mfi){auto out=s.dr[c].array(mfi);auto e=s.e[c].const_array(mfi),el=s.paired_low[c].const_array(mfi),o=s.old_e[c].const_array(mfi),r=s.er[c].const_array(mfi),r0=s.old_er[c].const_array(mfi),dr=s.delta_er[c].const_array(mfi),a=actual[c]->const_array(mfi),al=low[c]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            Pair left{e(i,j,k),el(i,j,k)},right{-o(i,j,k),0.};auto difference=paired_arithmetic::Add(left,right);auto expected=paired_arithmetic::Multiply(difference,factor);
            R const rounding=Add(Mul(std::abs(factor),PairAddRound(left,right)),Add(PairMultiplyRound(difference,factor),std::abs(expected.lo)));
            R const absolute=Add(Mul(std::abs(factor),Add(r(i,j,k),r0(i,j,k))),rounding);
            R const anchored=Add(Mul(std::abs(factor),dr(i,j,k)),rounding);R bound=std::min(absolute,anchored);
            if(!PairFinite(difference)||!PairFinite(expected)||!Normal(absolute)||!Normal(anchored)||!Normal(bound)||a(i,j,k)!=expected.hi||al(i,j,k)!=expected.lo)amrex::HostDevice::Atomic::Add(invalid,1);
            out(i,j,k)=bound;
        });}
    s.Finish(bad);Copy(s.d,Const(actual));s.Sync(s.dr);s.stage=true;
}
void RecordPairedExtrapolation(NativeLongitudinalProducerCertificate::Impl& s,V const& actual,R theta){
    if(!s.OpenProducer())return;auto* owner=NativeActivePairedFields(*s.sim);
    s.valid=Collective(s.valid&&s.paired&&owner&&theta==.5);if(!s.valid)return;
    auto low=owner->LongitudinalLow();R a=1./theta;
    RecordPairedScalar(s,nullptr,a);if(!s.valid)return;
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(s.er[c]);mfi.isValid();++mfi){auto radius=s.er[c].array(mfi),dr=s.delta_er[c].array(mfi);auto x=s.e[c].const_array(mfi),xl=s.paired_low[c].const_array(mfi),o=s.old_e[c].const_array(mfi),r0=s.old_er[c].const_array(mfi),y=actual[c]->const_array(mfi),yl=low[c]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            Pair before{x(i,j,k),xl(i,j,k)},origin{o(i,j,k),0.};auto diff=paired_arithmetic::Add(before,paired_arithmetic::Negate(origin));auto change=paired_arithmetic::Multiply(diff,a);auto expected=paired_arithmetic::Add(origin,change);
            R const round=Add(Mul(a,PairAddRound(before,paired_arithmetic::Negate(origin))),Add(PairMultiplyRound(diff,a),PairAddRound(origin,change)));
            R const delta=Add(Mul(a,dr(i,j,k)),round);
            R const absolute=Add(Add(Mul(a,radius(i,j,k)),Mul(std::abs(1.-a),r0(i,j,k))),round);
            R const bound=std::min(absolute,Add(r0(i,j,k),delta));
            if(!PairFinite(diff)||!PairFinite(change)||!PairFinite(expected)||!Normal(round)||!Normal(delta)||!Normal(bound)||y(i,j,k)!=expected.hi||yl(i,j,k)!=expected.lo)amrex::HostDevice::Atomic::Add(invalid,1);
            radius(i,j,k)=bound;dr(i,j,k)=delta;
        });}
    s.Finish(bad);Copy(s.e,Const(actual));Copy(s.paired_low,low);s.Sync(s.er);s.Sync(s.delta_er);
}
void RecordPairedMaterialization(NativeLongitudinalProducerCertificate::Impl& s){
    if(!s.OpenProducer())return;auto* owner=NativeActivePairedFields(*s.sim);
    s.valid=Collective(s.valid&&s.paired&&owner);if(!s.valid)return;
    auto actual=s.sim->m_fields.get_alldirs("hybrid_E_long_fp",0);auto low=owner->LongitudinalLow();
    bool same=true;for(int c=0;c<3;++c)same=Equal(s.e[c],*actual[c])&&Equal(s.paired_low[c],*low[c])&&same;
    bool const phi=Equal(s.paired_phi,*s.sim->m_fields.get("hybrid_phi_darwin_fp",0));
    bool const phi_low=Equal(s.paired_phi_low,owner->ScalarLow());
    s.valid=Collective(s.valid&&s.paired_scalar_ready&&same&&phi&&phi_low);if(!s.valid)return;
    amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(s.er[c]);mfi.isValid();++mfi){auto radius=s.er[c].array(mfi),dr=s.delta_er[c].array(mfi);auto lo=s.paired_low[c].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){R magnitude=std::abs(lo(i,j,k));R r=Add(radius(i,j,k),magnitude),d=Add(dr(i,j,k),magnitude);if(!Normal(magnitude)||!Normal(r)||!Normal(d))amrex::HostDevice::Atomic::Add(invalid,1);radius(i,j,k)=r;dr(i,j,k)=d;});}
    s.Finish(bad);for(auto& f:s.paired_low)f.setVal(0.);s.paired_phi_low.setVal(0.);s.paired_scalar_ready=false;s.Sync(s.er);s.Sync(s.delta_er);
}
}
void NativeLongitudinalProducerCertificate::PairedBeginField(){
    auto& s=*m_impl;if(!s.OpenProducer())return;auto* owner=NativeActivePairedFields(*s.sim);
    s.valid=Collective(s.valid&&s.paired&&owner&&!s.paired_scalar_ready);if(!s.valid)return;
    auto const& phi=*s.sim->m_fields.get("hybrid_phi_darwin_fp",0);
    // This is an exact physical representation record, NOT a reseed of the
    // ideal longitudinal producer or any E/D error radius.
    bool const owners=s.ScalarOwners(phi),low_owners=s.ScalarOwners(owner->ScalarLow());
    s.valid=Collective(s.valid&&owners&&low_owners);if(!s.valid)return;
    Copy(s.paired_phi,phi);Copy(s.paired_phi_origin,phi);Copy(s.paired_phi_low,owner->ScalarLow());s.paired_scalar_ready=true;
}
void NativeLongitudinalProducerCertificate::PairedSchurAddition(MF const& p,CV const& g,V const& a,R omega){RecordPairedSchur(*m_impl,p,g,a,omega);}
void NativeLongitudinalProducerCertificate::PairedCaptureStage(R dt,R theta){RecordPairedStage(*m_impl,dt,theta);}
void NativeLongitudinalProducerCertificate::PairedExtrapolateElectric(V const& a,R theta){RecordPairedExtrapolation(*m_impl,a,theta);}
void NativeLongitudinalProducerCertificate::MaterializeElectric(){RecordPairedMaterialization(*m_impl);}
void NativeLongitudinalProducerCertificate::CaptureStage(R dt,R theta){ProducerStage(*m_impl,dt,theta);}
void NativeLongitudinalProducerCertificate::ExtrapolateElectric(V const& actual,R theta){
    auto& s=*m_impl;if(!s.OpenProducer())return;s.valid=Collective(s.valid&&theta==.5);if(!s.valid)return;
    s.Affine(s.e,s.er,Const(s.e),Const(s.er),Const(s.old_e),Const(s.old_er),Const(actual),1./theta,1.-1./theta,Impl::Relative::Endpoint);
}
void NativeLongitudinalProducerCertificate::RotateDisplacement(R theta){
    auto& s=*m_impl;if(!s.OpenProducer())return;s.valid=Collective(s.valid&&s.stage&&theta==.5);if(!s.valid)return;
    auto actual=s.sim->m_fields.get_alldirs("diagnostic_D_endpoint_fp",0);
    s.Affine(s.d,s.dr,Const(s.d),Const(s.dr),Const(s.old_d),Const(s.old_dr),Const(actual),2.,-1.);
}
bool NativeLongitudinalProducerCertificate::AppendEvent(MF const& potential,CV const& increment){
    auto& s=*m_impl;if(!s.OpenProducer())return false;
    // Event increment is minus the native projection gradient. Reuse the
    // existing preallocated electric scratch for that positive gradient only.
    // All EL producers are finished. Seal their phase below; the next Fork
    // constructs a fresh origin from accepted e, with both delta radii zero.
    for(int c=0;c<3;++c){
        s.old_e[c].setVal(0.);Copy(s.old_e[c],*increment[c]);
        Negate(s.old_e[c]);
    }
    s.Gradient(potential,Const(s.old_e));
    auto actual=s.sim->m_fields.get_alldirs("diagnostic_D_endpoint_fp",0);
    s.Affine(s.d,s.dr,Const(s.d),Const(s.dr),increment,Const(s.gradient_radius),Const(actual),1.,1.);s.event_appended=true;return s.valid;
}
bool NativeLongitudinalProducerCertificate::Matches(WarpX& w,R time,std::uint64_t epoch)const{
    auto const& s=*m_impl;auto const& geometry=w.get_pointer_HybridPICModel()->ElectronThermalGeometry();
    bool ok=s.valid&&s.initial_bound&&s.sim==&w&&s.time==time&&s.epoch==epoch&&
        s.geometry.Domain()==geometry.Domain()&&s.geometry.Coord()==geometry.Coord();
    for(int axis=0;axis<AMREX_SPACEDIM;++axis)ok=ok&&
        s.geometry.ProbLo(axis)==geometry.ProbLo(axis)&&s.geometry.ProbHi(axis)==geometry.ProbHi(axis)&&
        s.geometry.isPeriodic(axis)==geometry.isPeriodic(axis);
    if(s.paired) {
        ok=NativeActivePairedFields(w)==nullptr&&ok;
        for(int c=0;c<3;++c)ok=Zero(s.paired_low[c],s.paired_low[c].nGrowVect())&&ok;
    }
    auto el=w.m_fields.get_alldirs("hybrid_E_long_fp",0),d=w.m_fields.get_alldirs("diagnostic_D_endpoint_fp",0);
    for(int c=0;c<3;++c){ok=Equal(s.e[c],*el[c])&&ok;ok=Equal(s.d[c],*d[c])&&ok;}
    return Collective(ok);
}
bool NativeLongitudinalProducerCertificate::ValidateCurl(CV const& curl,R& maximum_bound)const{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(curl);maximum_bound=0.;return false;
#else
    auto const& s=*m_impl;bool ok=s.valid&&s.geometry.IsRZ()&&s.geometry.ProbLo(0)==0.&&s.geometry.isPeriodic(1);
    auto const dx=s.geometry.InvCellSizeArray();auto const global=s.geometry.Domain();amrex::Gpu::DeviceScalar<int> bad(0);auto* invalid=bad.dataPtr();
    auto& bounds=m_impl->curl_bounds;
    for(int c=0;c<3;++c)ok=ok&&curl[c]&&curl[c]->nComp()==1;
    if(!Collective(ok)) { maximum_bound=0.;return false; }
    ok=curl[1]->boxArray()==bounds.boxArray()&&curl[1]->DistributionMap()==bounds.DistributionMap();
    // Btheta is cell-centred: D_r and D_z forward contributors are valid
    // edges of that same cell FAB, even at axis/PEC and periodic aliases.
    for(amrex::MFIter mfi(bounds);mfi.isValid();++mfi){
        auto radial=mfi.validbox();radial.growHi(1,1);
        auto axial=mfi.validbox();axial.growHi(0,1);
        ok=s.d[0].boxArray()[mfi.index()].contains(radial)&&s.d[2].boxArray()[mfi.index()].contains(axial)&&ok;
    }
    if(!Collective(ok)) { maximum_bound=0.;return false; }
    bounds.setVal(0.);
    for(amrex::MFIter mfi(bounds);mfi.isValid();++mfi){auto out=bounds.array(mfi);auto y=curl[1]->const_array(mfi),dr=s.d[0].const_array(mfi),dz=s.d[2].const_array(mfi),rr=s.dr[0].const_array(mfi),rz=s.dr[2].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            R const error=Add(Mul(dx[0],Add(rz(i+1,j,k),rz(i,j,k))),Mul(dx[1],Add(rr(i,j+1,k),rr(i,j,k))));
            R const terms=Add(Mul(dx[0],Add(std::abs(dz(i+1,j,k)),std::abs(dz(i,j,k)))),Mul(dx[1],Add(std::abs(dr(i,j+1,k)),std::abs(dr(i,j,k)))));
            R const bound=Add(error,Mul(Gamma(5),terms));out(i,j,k)=bound;
            if(!global.contains(amrex::IntVect(AMREX_D_DECL(i,j,k)))||!Normal(bound)||!Normal(y(i,j,k))||std::abs(y(i,j,k))>bound)amrex::HostDevice::Atomic::Add(invalid,1);
        });
    }
    // m0 G has no azimuthal component, so these channels are exact zeros.
    bool const theta=Zero(s.d[1],amrex::IntVect(0));
    bool const radial=Zero(*curl[0],amrex::IntVect(0));
    bool const axial=Zero(*curl[2],amrex::IntVect(0));
    ok=theta&&radial&&axial&&ok;
    amrex::Gpu::synchronize();maximum_bound=bounds.norm0();return Collective(ok&&bad.dataValue()==0);
#endif
}
NativeLongitudinalProducerCertificate::Stats NativeLongitudinalProducerCertificate::Statistics()const{
    auto const& s=*m_impl;Stats result;result.operations=s.operations;
    for(int c=0;c<3;++c){result.electric_radius=std::max(result.electric_radius,s.er[c].norm0());result.displacement_radius=std::max(result.displacement_radius,s.dr[c].norm0());
        for(auto const* f:{&s.e[c],&s.er[c],&s.d[c],&s.dr[c],&s.old_e[c],&s.old_er[c],&s.old_d[c],&s.old_dr[c],&s.previous_er[c],&s.gradient_radius[c],&s.delta_er[c],&s.previous_delta_er[c]})for(amrex::MFIter mfi(*f);mfi.isValid();++mfi)result.local_bytes+=(*f)[mfi].size()*sizeof(R);}
    if(s.paired){for(auto const& f:s.paired_low)for(amrex::MFIter mfi(f);mfi.isValid();++mfi)result.local_bytes+=f[mfi].size()*sizeof(R);
        for(auto const* f:{&s.paired_phi,&s.paired_phi_low,&s.paired_phi_origin})for(amrex::MFIter mfi(*f);mfi.isValid();++mfi)result.local_bytes+=(*f)[mfi].size()*sizeof(R);}
    for(auto const* f:{&s.scalar_synced,&s.scalar_difference,&s.curl_bounds})
        for(amrex::MFIter mfi(*f);mfi.isValid();++mfi)result.local_bytes+=(*f)[mfi].size()*sizeof(R);
    return result;
}
NativeLongitudinalProducerCertificate::ConstVector NativeLongitudinalProducerCertificate::ElectricRadii()const{return Const(m_impl->er);}
NativeLongitudinalProducerCertificate::ConstVector NativeLongitudinalProducerCertificate::DisplacementRadii()const{return Const(m_impl->dr);}
NativeLongitudinalProducerCertificate::ConstVector NativeLongitudinalProducerCertificate::ElectricIncrementRadii()const{return Const(m_impl->delta_er);}
} // namespace warpx::darwin
