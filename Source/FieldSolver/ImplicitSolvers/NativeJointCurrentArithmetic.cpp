/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeJointCurrentArithmetic.H"
#include "NativeJointCurlArithmetic.H"
#include "NativeEndpointField.H"
#include "NativeLongitudinalIncrement.H"
#include "NativePairedDarwinFields.H"
#include "NativePECPlasma.H"
#include "DarwinABoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <limits>
namespace warpx::thermal {
namespace {
using MF=amrex::MultiFab;using Real=amrex::Real;using Vec=WarpXSolverVec;
using View=ablastr::fields::VectorField;using Field=amrex::Array<MF,3>;
using FT=warpx::fields::FieldType;using D=ablastr::fields::Direction;
namespace dd=warpx::ohm::compensated;
View V(Field& f){return {&f[0],&f[1],&f[2]};}
void Define(Field& f,View const& shape,int grow=-1){for(int c=0;c<3;++c){f[c].define(shape[c]->boxArray(),shape[c]->DistributionMap(),1,grow<0?shape[c]->nGrowVect():amrex::IntVect(grow));f[c].setVal(0.);}}
void Copy(Field& f,View const& in){for(int c=0;c<3;++c)MF::Copy(f[c],*in[c],0,0,1,amrex::min(f[c].nGrowVect(),in[c]->nGrowVect()));}
void Copy(Field& f,Field& in){Copy(f,V(in));}
bool Finite(Field const& f){bool ok=true;for(auto const& a:f)ok=a.is_finite(0,1,0,true)&&ok;return ok;}
bool Matches(Vec const& x,Field const& saved){
 bool ok=x.IsDefined()&&x.getArrayVecType()==FT::Efield_fp&&x.numAMRLevels()==1;
 if(ok)for(int c=0;c<3;++c){auto const& a=*x.getArrayVec()[0][c];auto const& b=saved[c];
  if(a.boxArray()!=b.boxArray()||a.DistributionMap()!=b.DistributionMap()||a.nComp()!=1){ok=false;continue;}
  amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);using T=decltype(data)::Type;
  for(amrex::MFIter it(a);it.isValid();++it){auto p=a.const_array(it),q=b.const_array(it);op.eval(it.validbox(),data,[=]AMREX_GPU_DEVICE(int i,int j,int k)->T{return {p(i,j,k)!=q(i,j,k)};});}
  ok=(amrex::get<0>(data.value())==0)&&ok;
 }return ok;
}
void Sync(WarpX& w,View const& fields){amrex::Vector<MF*> f(fields.begin(),fields.end());amrex::FillBoundaryAndSync_nowait(f,w.Geom(0).periodicity());amrex::FillBoundaryAndSync_finish(f);}
void PMC(WarpX& w,View const& fields){amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};for(int d=0;d<AMREX_SPACEDIM;++d){lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
 for(auto* f:fields){f->OverrideSync(w.Geom(0).periodicity());f->FillBoundary(w.Geom(0).periodicity());ApplyDarwinPMCVectorBoundary(*f,w.Geom(0),lo,hi);}}
// Exact homogeneous native A boundary in this certified zero-drive, no-EB RZ scope.
// Apply it to BOTH parts: neither allocated ghosts nor an E boundary alone
// certify the A producer. The fixed nodal PEC and axis traces have zero low.
void PotentialImages(WarpX& w,View const& fields,Field& zero){
#if defined(WARPX_DIM_RZ)
 amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};lo[1]=hi[1]=1;
 for(int c=0;c<3;++c){auto& f=*fields[c];auto domain=amrex::convert(w.Geom(0).Domain(),f.ixType());bool node=f.ixType().nodeCentered(0);int outer=domain.bigEnd(0);
  for(amrex::MFIter it(f);it.isValid();++it){auto a=f.array(it);amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){if((c==1&&i==0)||(node&&i>=outer))a(i,j,k)=0.;});}
  f.FillBoundary(w.Geom(0).periodicity());ApplyDarwinCellCenteredABoundary(f,zero[c],w.Geom(0),true,nullptr,lo,hi);
 }w.ApplyFieldBoundaryOnAxis(fields[0],fields[1],fields[2],0);
#else
 amrex::ignore_unused(w,fields,zero);amrex::Abort("Joint current arithmetic requires RZ");
#endif
}
void Curl(WarpX& w,View const& high,View const& low,Field& bh,Field& bl,amrex::IntVect const& physical){
#if defined(WARPX_DIM_RZ)
 for(int c=0;c<3;++c){bh[c].setVal(0.);bl[c].setVal(0.);}
 int width=1;for(int d=0;d<AMREX_SPACEDIM;++d)width=std::max(width,physical[d]);
 auto grow=physical;for(int d=0;d<AMREX_SPACEDIM;++d)if(w.Geom(0).isPeriodic(d))grow[d]=width;
 Real dr=w.Geom(0).CellSize(0),dz=w.Geom(0).CellSize(1);
 for(int c=0;c<3;++c)for(amrex::MFIter it(bh[c]);it.isValid();++it){
  auto box=amrex::grow(it.validbox(),width)&amrex::grow(amrex::convert(w.Geom(0).Domain(),bh[c].ixType()),grow);box.setSmall(0,std::max(0,box.smallEnd(0)));
  AMREX_ALWAYS_ASSERT(bh[c][it].box().contains(box)&&bl[c][it].box().contains(box));
  amrex::GpuArray<joint_arithmetic::Operand,3> a{};
  for(int d=0;d<3;++d){a[d]={high[d]->const_array(it),low[d]?low[d]->const_array(it):amrex::Array4<Real const>{}};
   if(d==c)continue;auto required=box;required.setType(high[d]->ixType());int axis=3-c-d;axis=axis==2?1:axis==0?0:-1;if(axis>=0)required.growHi(axis,1);
   AMREX_ALWAYS_ASSERT((*high[d])[it].box().contains(required)&&(!low[d]||(*low[d])[it].box().contains(required)));
  }auto h=bh[c].array(it),l=bl[c].array(it);
  amrex::ParallelFor(box,[=]AMREX_GPU_DEVICE(int i,int j,int k){auto value=joint_arithmetic::Curl(a,c,i,j,dr,dz);h(i,j,k)=value.hi;l(i,j,k)=value.lo;});
 }Sync(w,V(bh));Sync(w,V(bl));
#else
 amrex::ignore_unused(w,high,low,bh,bl,physical);amrex::Abort("Joint current arithmetic requires RZ");
#endif
}
void Ampere(WarpX& w,View const& high,View const& low,Field& ch,Field& cl){
#if defined(WARPX_DIM_RZ)
 Real dr=w.Geom(0).CellSize(0),dz=w.Geom(0).CellSize(1);
 for(int c=0;c<3;++c)for(amrex::MFIter it(ch[c]);it.isValid();++it){amrex::GpuArray<joint_arithmetic::Operand,3> b{};
  for(int d=0;d<3;++d)b[d]={high[d]->const_array(it),low[d]?low[d]->const_array(it):amrex::Array4<Real const>{}};
  auto h=ch[c].array(it),l=cl[c].array(it);amrex::ParallelFor(amrex::grow(it.validbox(),1),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto value=joint_arithmetic::Ampere(b,c,i,j,dr,dz);h(i,j,k)=value.hi;l(i,j,k)=value.lo;});
 }PMC(w,V(ch));PMC(w,V(cl));
#else
 amrex::ignore_unused(w,high,low,ch,cl);amrex::Abort("Joint current arithmetic requires RZ");
#endif
}
}
struct NativeJointCurrentArithmetic::Impl {
 WarpX& w;
 Field accepted_high,accepted_low,origin_high,origin_low,frozen_high,frozen_low,pending_low,formed_high,current_low,e_image;
 Field a_low,zero_a,fixed_B,b_high,b_low,rate_a_low,rate_b_high,rate_b_low,current_high,current_remainder,static_high,static_remainder,ohm;
 Vec const* pending=nullptr;Vec const* bound=nullptr;
 bool open=false,pending_probe=false,from_pending=false,magnetic_ready=false,current_ready=false,ohm_ready=false;
 std::uint64_t pending_generation=0,generation=0;
 explicit Impl(WarpX& sim,Vec const& initial):w(sim){
  auto e=w.m_fields.get_alldirs(FT::Efield_fp,0),a=w.m_fields.get_alldirs("hybrid_A_fp",0),b=w.m_fields.get_alldirs(FT::Bfield_fp,0),c=w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0);
  for(auto* f:{&accepted_high,&accepted_low,&origin_high,&origin_low,&frozen_high,&frozen_low,&pending_low,&formed_high,&ohm})Define(*f,e,0);
  for(auto* f:{&current_low,&e_image})Define(*f,e);
  for(auto* f:{&a_low,&zero_a,&rate_a_low})Define(*f,a);
  for(auto* f:{&fixed_B,&b_high,&b_low,&rate_b_high,&rate_b_low})Define(*f,b);
  for(auto* f:{&current_high,&current_remainder,&static_high,&static_remainder})Define(*f,c,1);
  Copy(accepted_high,initial.getArrayVec()[0]);
 }
};
bool NativeJointCurrentArithmetic::SupportedLocal(WarpX& w){
#if defined(WARPX_DIM_RZ)
 auto const& model=*w.get_pointer_HybridPICModel();bool split=false;amrex::ParmParse("endpoint_diagnostic").query("vacuum_split_ampere",split);
 return remainder::ArithmeticSupported()&&split&&NativeCorrelatedIncrementEnabled()&&
  w.maxLevel()==0&&WarpX::grid_type==GridType::Staggered&&WarpX::ncomps==1&&
  w.Geom(0).ProbLo(0)==0.&&WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC&&
  WarpX::field_boundary_lo[1]==FieldBoundaryType::PMC&&WarpX::field_boundary_hi[1]==FieldBoundaryType::PMC&&
  // The caller has already certified the owned constant-zero time profile
  // and zero initial reference; an allocated external-field object is allowed.
  !model.m_has_external_current&&!model.m_resistive_wall&&
  !warpx::darwin::NativeActivePairedFields(w);
#else
 amrex::ignore_unused(w);return false;
#endif
}
NativeJointCurrentArithmetic::NativeJointCurrentArithmetic(WarpX& w,Vec const& x):m(std::make_unique<Impl>(w,x)){}
NativeJointCurrentArithmetic::~NativeJointCurrentArithmetic()=default;
void NativeJointCurrentArithmetic::Invalidate() noexcept {m->generation=0;m->bound=m->pending=nullptr;m->open=m->magnetic_ready=m->current_ready=m->ohm_ready=false;}
bool NativeJointCurrentArithmetic::BeginSolve(Vec const& input){auto& q=*m;bool ok=!q.open&&Matches(input,q.accepted_high)&&remainder::ArithmeticSupported();if(!remainder::All(ok))return false;Copy(q.origin_high,q.accepted_high);Copy(q.origin_low,q.accepted_low);q.open=true;q.pending=nullptr;return true;}
bool NativeJointCurrentArithmetic::FormInput(Vec& output,Vec const& base,Real factor,Vec const& direction,bool probe,std::uint64_t next){auto& q=*m;
 bool ok=q.open&& &output!=&base&& &output!=&direction&&next>q.generation&&next<std::numeric_limits<std::uint64_t>::max()&&std::isfinite(factor)&&remainder::ArithmeticSupported()&&Matches(base,probe?q.frozen_high:q.accepted_high);
 if(!remainder::All(ok))return false;
 auto& low=probe?q.frozen_low:q.accepted_low;
 for(int c=0;c<3;++c)for(amrex::MFIter it(*output.getArrayVec()[0][c]);it.isValid();++it){auto h=output.getArrayVec()[0][c]->array(it),l=q.pending_low[c].array(it);auto bh=base.getArrayVec()[0][c]->const_array(it),bl=low[c].const_array(it),d=direction.getArrayVec()[0][c]->const_array(it);
  amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto value=dd::Add({bh(i,j,k),bl(i,j,k)},dd::Multiply({d(i,j,k),0.},factor));h(i,j,k)=value.hi;l(i,j,k)=value.lo;});}
 Copy(q.formed_high,output.getArrayVec()[0]);ok=Finite(q.formed_high)&&Finite(q.pending_low);if(!remainder::All(ok)){q.pending=nullptr;return false;}
 q.pending=&output;q.pending_probe=probe;q.pending_generation=next;return true;
}
bool NativeJointCurrentArithmetic::BindInput(Vec const& input,bool probe,std::uint64_t generation){auto& q=*m;bool prepared=q.pending!=nullptr;
 bool const same=Matches(input,q.accepted_high);
 // Direct owner users pass an explicit high-only vector. Outside a managed
 // solve that remains a supported input; only the unchanged committed high
 // retains its already-owned low. Managed probes require the affine receipt.
 bool ok=remainder::ArithmeticSupported()&&generation>0&&generation<std::numeric_limits<std::uint64_t>::max()&&
  (prepared?(q.pending==&input&&q.pending_probe==probe&&q.pending_generation==generation&&Matches(input,q.formed_high)):
   (q.open?(!probe&&same):(input.IsDefined()&&input.getArrayVecType()==FT::Efield_fp&&input.numAMRLevels()==1)));
 if(!remainder::All(ok)){q.pending=nullptr;return false;}
 for(auto& f:q.current_low)f.setVal(0.);
 if(prepared)Copy(q.current_low,q.pending_low);else if(same)Copy(q.current_low,q.accepted_low);
 q.bound=&input;q.generation=generation;q.from_pending=prepared;q.pending=nullptr;q.magnetic_ready=q.current_ready=q.ohm_ready=false;return true;
}
bool NativeJointCurrentArithmetic::AcceptInput(Vec const& input,std::uint64_t generation){auto& q=*m;bool ok=q.open&&q.bound==&input&&q.generation==generation&&q.from_pending&&q.ohm_ready&&remainder::ArithmeticSupported();if(!remainder::All(ok))return false;
 Copy(q.accepted_high,input.getArrayVec()[0]);Copy(q.accepted_low,q.current_low);q.from_pending=false;return true;
}
void NativeJointCurrentArithmetic::EndSolve(bool) noexcept {m->open=false;m->pending=nullptr;}
bool NativeJointCurrentArithmetic::RestoreInput(Vec const& input){auto& q=*m;bool ok=q.open?Matches(input,q.origin_high):Matches(input,q.accepted_high);if(!remainder::All(ok))return false;if(q.open){Copy(q.accepted_high,q.origin_high);Copy(q.accepted_low,q.origin_low);}q.pending=nullptr;return true;}
void NativeJointCurrentArithmetic::CompleteInput(bool probe,std::uint64_t generation){auto& q=*m;AMREX_ALWAYS_ASSERT(Ready(generation));if(!q.open&&!probe){Copy(q.accepted_high,q.bound->getArrayVec()[0]);Copy(q.accepted_low,q.current_low);}}
bool NativeJointCurrentArithmetic::FreezeInput(Vec const& input,std::uint64_t generation){auto& q=*m;bool ok=Ready(generation)&&Matches(input,q.accepted_high)&&remainder::ArithmeticSupported();if(!remainder::All(ok))return false;Copy(q.frozen_high,input.getArrayVec()[0]);Copy(q.frozen_low,q.accepted_low);Copy(q.current_low,q.accepted_low);q.bound=&input;return true;}
bool NativeJointCurrentArithmetic::Ready(std::uint64_t generation) const noexcept{return m->bound&&m->generation==generation&&generation!=0;}
void NativeJointCurrentArithmetic::BuildFields(Real interval,amrex::IntVect const& physical,std::uint64_t generation){auto& q=*m;auto& w=q.w;AMREX_ALWAYS_ASSERT(Ready(generation)&&interval>0.&&remainder::ArithmeticSupported());q.magnetic_ready=q.current_ready=q.ohm_ready=false;
 auto E=w.m_fields.get_alldirs(FT::Efield_fp,0),A=w.m_fields.get_alldirs("hybrid_A_fp",0),A0=w.m_fields.get_alldirs("hybrid_A_old_fp",0),B=w.m_fields.get_alldirs(FT::Bfield_fp,0),fixed=w.m_fields.get_alldirs("hybrid_B_static_fp",0);
 Copy(q.e_image,q.current_low);warpx::darwin::CompleteNativeElectricGatherImages(w,V(q.e_image));
 auto rate=w.m_fields.get_alldirs("diagnostic_Adot_fp",0);
 for(int c=0;c<3;++c)for(amrex::MFIter it(*A[c]);it.isValid();++it){auto ah=A[c]->array(it),al=q.a_low[c].array(it),rh=rate[c]->array(it),rl=q.rate_a_low[c].array(it);auto e=E[c]->const_array(it),el=q.e_image[c].const_array(it),old=A0[c]->const_array(it);
  amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto value=dd::Add({old(i,j,k),0.},dd::Multiply({e(i,j,k),el(i,j,k)},-interval));ah(i,j,k)=value.hi;al(i,j,k)=value.lo;rh(i,j,k)=-e(i,j,k);rl(i,j,k)=-el(i,j,k);});}
 PotentialImages(w,A,q.zero_a);PotentialImages(w,V(q.a_low),q.zero_a);PotentialImages(w,rate,q.zero_a);PotentialImages(w,V(q.rate_a_low),q.zero_a);
 Curl(w,A,V(q.a_low),q.b_high,q.b_low,physical);
 Curl(w,rate,V(q.rate_a_low),q.rate_b_high,q.rate_b_low,DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
 Ampere(w,V(q.rate_b_high),V(q.rate_b_low),q.current_high,q.current_remainder);
 auto cdot=w.m_fields.get_alldirs("diagnostic_Cdot_fp",0);auto bdot=w.m_fields.get_alldirs("diagnostic_Bdot_fp",0);
 for(int c=0;c<3;++c){cdot[c]->setVal(0.);MF::Copy(*cdot[c],q.current_high[c],0,0,1,1);MF::Copy(*bdot[c],q.rate_b_high[c],0,0,1,bdot[c]->nGrowVect());}
 // Split fixed current is added once, never by curling a cancellation-prone
 // total B. Owner-copy maps apply separately to each represented operand.
 Copy(q.fixed_B,fixed);Sync(w,V(q.fixed_B));Ampere(w,V(q.fixed_B),View{},q.static_high,q.static_remainder);
 Ampere(w,V(q.b_high),V(q.b_low),q.current_high,q.current_remainder);
 for(int c=0;c<3;++c){for(amrex::MFIter it(q.current_high[c]);it.isValid();++it){auto h=q.current_high[c].array(it),l=q.current_remainder[c].array(it);auto sh=q.static_high[c].const_array(it),sl=q.static_remainder[c].const_array(it);amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto v=dd::Add({h(i,j,k),l(i,j,k)},{sh(i,j,k),sl(i,j,k)});h(i,j,k)=v.hi;l(i,j,k)=v.lo;});}
  for(amrex::MFIter it(*B[c]);it.isValid();++it){auto b=B[c]->array(it);auto h=q.b_high[c].const_array(it),l=q.b_low[c].const_array(it),f=q.fixed_B[c].const_array(it);amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){b(i,j,k)=dd::Add({h(i,j,k),l(i,j,k)},{f(i,j,k),0.}).hi;});}}
 Sync(w,B);q.magnetic_ready=true;
}
void NativeJointCurrentArithmetic::PublishCurrent(std::uint64_t generation){auto& q=*m;AMREX_ALWAYS_ASSERT(Ready(generation)&&q.magnetic_ready&&!q.current_ready);auto out=q.w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0);for(int c=0;c<3;++c)MF::Copy(*out[c],q.current_high[c],0,0,1,1);q.current_ready=true;}
void NativeJointCurrentArithmetic::SubtractDisplacement(Real interval,View const* thermal_low,std::uint64_t generation){auto& q=*m;AMREX_ALWAYS_ASSERT(Ready(generation)&&q.current_ready);auto high=q.w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0),e=q.w.m_fields.get_alldirs("hybrid_E_long_fp",0),old=q.w.m_fields.get_alldirs("hybrid_E_long_old_fp",0);Real factor=PhysConst::epsilon_0*(1./interval);
 for(int c=0;c<3;++c)for(amrex::MFIter it(q.current_high[c]);it.isValid();++it){auto h=q.current_high[c].array(it),l=q.current_remainder[c].array(it),out=high[c]->array(it);auto a=e[c]->const_array(it),b=old[c]->const_array(it);auto const* small=warpx::darwin::increment::Low(q.w,c);auto al=small?small->const_array(it):amrex::Array4<Real const>{};auto keep=thermal_low?(*thermal_low)[c]->array(it):amrex::Array4<Real>{};
  amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto delta=dd::Add(dd::Sum(a(i,j,k),-b(i,j,k)),{al?al(i,j,k):0.,0.});auto v=dd::Add({h(i,j,k),l(i,j,k)},dd::Negate(dd::Multiply(delta,factor)));h(i,j,k)=v.hi;l(i,j,k)=v.lo;out(i,j,k)=v.hi;if(keep)keep(i,j,k)=v.lo;});}
}
void NativeJointCurrentArithmetic::AddTransverseLow(MF& total,MF const& longitudinal,MF const* longitudinal_low,int c,std::uint64_t generation) const {auto& q=*m;AMREX_ALWAYS_ASSERT(Ready(generation));
 for(amrex::MFIter it(total);it.isValid();++it){auto h=total.array(it);auto tl=q.e_image[c].const_array(it),a=longitudinal.const_array(it),al=longitudinal_low?longitudinal_low->const_array(it):amrex::Array4<Real const>{};amrex::ParallelFor(it.fabbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){h(i,j,k)=dd::Add({h(i,j,k),tl(i,j,k)},{a(i,j,k),al?al(i,j,k):0.}).hi;});}
}
void NativeJointCurrentArithmetic::AssembleElectric(View const& field,std::uint64_t generation) const {auto el=m->w.m_fields.get_alldirs("hybrid_E_long_fp",0);for(int c=0;c<3;++c)AddTransverseLow(*field[c],*el[c],warpx::darwin::increment::Low(m->w,c),c,generation);}
void NativeJointCurrentArithmetic::CaptureOhm(View const& field,std::uint64_t generation){AMREX_ALWAYS_ASSERT(Ready(generation)&&m->current_ready);Copy(m->ohm,field);m->ohm_ready=true;}
std::array<NativeJointCurrentArithmetic::MF const*,3>
NativeJointCurrentArithmetic::CurrentRemainder(std::uint64_t generation) const {
 AMREX_ALWAYS_ASSERT(Ready(generation)&&m->current_ready);
 return {&m->current_remainder[0],&m->current_remainder[1],&m->current_remainder[2]};
}
void NativeJointCurrentArithmetic::FormResidual(Vec& out,Vec const& input,View const& ion,amrex::Array<amrex::iMultiFab,3> const& vacuum,View const& current,Real alpha,std::uint64_t generation) const {auto& q=*m;AMREX_ALWAYS_ASSERT(Ready(generation)&&q.ohm_ready&&q.current_ready&&q.bound==&input);
 for(int c=0;c<3;++c)for(amrex::MFIter it(*out.getArrayVec()[0][c]);it.isValid();++it){auto r=out.getArrayVec()[0][c]->array(it),jv=current[c]->array(it);auto e=input.getArrayVec()[0][c]->const_array(it),el=q.current_low[c].const_array(it),ohm=q.ohm[c].const_array(it),ch=q.current_high[c].const_array(it),cl=q.current_remainder[c].const_array(it),ji=ion[c]->const_array(it);auto v=vacuum[c].const_array(it);
  amrex::ParallelFor(it.validbox(),[=]AMREX_GPU_DEVICE(int i,int j,int k){auto delta=dd::Add({ch(i,j,k),cl(i,j,k)},{-ji(i,j,k),0.});jv(i,j,k)=v(i,j,k)?delta.hi:0.;r(i,j,k)=v(i,j,k)?dd::Multiply(delta,alpha).hi:dd::Add({e(i,j,k),el(i,j,k)},{-ohm(i,j,k),0.}).hi;});}
}
void NativeJointCurrentArithmetic::AppendReceipt(std::vector<MF*>& fields) const {for(auto* f:{&m->current_low,&m->e_image,&m->a_low})for(auto& c:*f)fields.push_back(&c);}
}
