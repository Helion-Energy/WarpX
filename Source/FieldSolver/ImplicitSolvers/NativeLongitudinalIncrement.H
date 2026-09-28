/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_LONGITUDINAL_INCREMENT_H_
#define WARPX_NATIVE_LONGITUDINAL_INCREMENT_H_
#include "DarwinABoundary.H"
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <iomanip>
#include <limits>
namespace warpx::darwin::increment {
using MF=amrex::MultiFab;using Real=amrex::Real;using View=ablastr::fields::VectorField;
using Dir=ablastr::fields::Direction;
namespace arithmetic=warpx::ohm::compensated;
constexpr char const* field_low="diagnostic_EL_roundoff_fp";
constexpr char const* scalar_low="diagnostic_phi_roundoff_fp";
constexpr char const* field_origin="diagnostic_EL_roundoff_origin_fp";
constexpr char const* scalar_origin="diagnostic_phi_roundoff_origin_fp";
inline bool Enabled(){bool value=false;amrex::ParmParse("endpoint_diagnostic").query("compensated_longitudinal_increment",value);return value;}
inline void Allocate(WarpX& w){
 if(!Enabled())return;
 for(int c=0;c<3;++c){auto const& f=*w.m_fields.get("hybrid_E_long_fp",Dir{c},0);for(auto name:{field_low,field_origin})if(!w.m_fields.has(name,Dir{c},0))w.m_fields.alloc_init(name,Dir{c},0,f.boxArray(),f.DistributionMap(),1,f.nGrowVect(),0.,true,true,false);}
 auto const& p=*w.m_fields.get("hybrid_phi_darwin_fp",0);for(auto name:{scalar_low,scalar_origin})if(!w.m_fields.has(name,0))w.m_fields.alloc_init(name,0,p.boxArray(),p.DistributionMap(),1,p.nGrowVect(),0.,true,true,false);
}
inline MF const* Low(WarpX& w,int c){return Enabled()&&w.m_fields.has(field_low,Dir{c},0)?w.m_fields.get(field_low,Dir{c},0):nullptr;}
inline void Images(WarpX& w,MF& f){
 amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};for(int d=0;d<AMREX_SPACEDIM;++d){lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;}
 f.setBndry(0.);f.OverrideSync(w.Geom(0).periodicity());f.FillBoundary(w.Geom(0).periodicity());ApplyDarwinPMCVectorBoundary(f,w.Geom(0),lo,hi);
}
inline void Begin(WarpX& w){
 if(!Enabled())return;
 bool joined=false,stable=false,full=false;amrex::ParmParse pp("endpoint_diagnostic");pp.query("joined_vacuum",joined);pp.query("stable_transverse_ohm",stable);pp.query("vacuum_schur_full_grid",full);
 AMREX_ALWAYS_ASSERT_WITH_MESSAGE(joined&&stable&&full,"Compensated longitudinal candidate requires joined vacuum, stable transverse Ohm and full-grid Schur correction");
 AMREX_ALWAYS_ASSERT(w.finestLevel()==0 && w.m_fields.has(scalar_low,0));
 for(int c=0;c<3;++c){auto& low=*w.m_fields.get(field_low,Dir{c},0);AMREX_ALWAYS_ASSERT(low.norminf(0,1,low.nGrowVect())==0.);auto& origin=*w.m_fields.get(field_origin,Dir{c},0);MF::Copy(origin,*w.m_fields.get("hybrid_E_long_fp",Dir{c},0),0,0,1,origin.nGrowVect());}
 auto& low=*w.m_fields.get(scalar_low,0);AMREX_ALWAYS_ASSERT(low.norminf(0,1,low.nGrowVect())==0.);auto& origin=*w.m_fields.get(scalar_origin,0);MF::Copy(origin,*w.m_fields.get("hybrid_phi_darwin_fp",0),0,0,1,origin.nGrowVect());
}
inline void Correct(WarpX& w,MF const& held_phi,View const& held,MF const& delta_phi,View const& delta,Real omega){
 using namespace arithmetic;
 auto& phi=*w.m_fields.get("hybrid_phi_darwin_fp",0);auto& pl=*w.m_fields.get(scalar_low,0);
 for(amrex::MFIter mfi(phi);mfi.isValid();++mfi){auto h=phi.array(mfi),l=pl.array(mfi);auto old=held_phi.const_array(mfi),d=delta_phi.const_array(mfi);amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto v=Add({old(i,j,k),l(i,j,k)},Multiply({d(i,j,k),0.},omega));h(i,j,k)=v.hi;l(i,j,k)=v.lo;});}
 pl.setBndry(0.);pl.FillBoundary(w.Geom(0).periodicity());
 auto out=w.m_fields.get_alldirs("hybrid_E_long_fp",0);auto dx=w.Geom(0).CellSizeArray();
 for(int c=0;c<3;++c){
#if defined(WARPX_DIM_3D)
  int const axis=c;
#else
  int const axis=c==0?0:c==2?1:-1;
#endif
  Real const inverse=axis<0?0.:1./dx[axis];auto& low=*w.m_fields.get(field_low,Dir{c},0);
  for(amrex::MFIter mfi(*out[c]);mfi.isValid();++mfi){auto h=out[c]->array(mfi),l=low.array(mfi);auto old=held[c]->const_array(mfi),d=delta[c]->const_array(mfi),p=delta_phi.const_array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto correction=WithHigh(Gradient(p,i,j,k,axis,inverse),d(i,j,k));auto v=Add({old(i,j,k),l(i,j,k)},Multiply(correction,omega));h(i,j,k)=v.hi;l(i,j,k)=v.lo;});}
  Images(w,*out[c]);Images(w,low);
 }
}
inline void AddFull(MF& total,MF const& longitudinal,MF const* low){
 if(!low){MF::Add(total,longitudinal,0,0,1,total.nGrowVect());return;}
 using namespace arithmetic;
 for(amrex::MFIter mfi(total);mfi.isValid();++mfi){auto e=total.array(mfi);auto h=longitudinal.const_array(mfi),l=low->const_array(mfi);amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){e(i,j,k)=Add(Sum(e(i,j,k),h(i,j,k)),{l(i,j,k),0.}).hi;});}
}
inline void SubtractLow(MF& field,MF const* low){if(low)MF::Subtract(field,*low,0,0,1,field.nGrowVect());}
inline void Extrapolate(MF& high,MF const& old,MF& low,Real theta){
 using namespace arithmetic;
 for(amrex::MFIter mfi(high);mfi.isValid();++mfi){auto h=high.array(mfi),l=low.array(mfi);auto o=old.const_array(mfi);amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto change=Add(Sum(h(i,j,k),-o(i,j,k)),{l(i,j,k),0.});auto value=Add({o(i,j,k),0.},Multiply(change,1./theta));h(i,j,k)=value.hi;l(i,j,k)=value.lo;});}
}
// This audit never changes a physical row or state. The initial Gphi-EL
// representation is retained; only the accumulated correction is compared.
inline void Audit(WarpX& w,char const* phase){
 if(!Enabled())return;
 using namespace arithmetic;
 auto const& p=*w.m_fields.get("hybrid_phi_darwin_fp",0);auto const& p0=*w.m_fields.get(scalar_origin,0);auto const& pl=*w.m_fields.get(scalar_low,0);auto dx=w.Geom(0).CellSizeArray();Real error=0.,scale=0.,low_peak=0.;
 for(int c=0;c<3;++c){
#if defined(WARPX_DIM_3D)
  int const axis=c;
#else
  int const axis=c==0?0:c==2?1:-1;
#endif
  Real const inv=axis<0?0.:1./dx[axis];auto const& high=*w.m_fields.get("hybrid_E_long_fp",Dir{c},0);auto const& origin=*w.m_fields.get(field_origin,Dir{c},0);auto const& low=*w.m_fields.get(field_low,Dir{c},0);MF difference(high.boxArray(),high.DistributionMap(),1,0);
  for(amrex::MFIter mfi(difference);mfi.isValid();++mfi){auto d=difference.array(mfi);auto h=high.const_array(mfi),o=origin.const_array(mfi),l=low.const_array(mfi),a=p.const_array(mfi),b=p0.const_array(mfi),al=pl.const_array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){auto gradient=Add(Add(Gradient(a,i,j,k,axis,inv),Negate(Gradient(b,i,j,k,axis,inv))),Gradient(al,i,j,k,axis,inv));auto field=Add(Sum(h(i,j,k),-o(i,j,k)),{l(i,j,k),0.});auto defect=Add(field,Negate(gradient));d(i,j,k)=defect.hi+defect.lo;});}
  error=std::max(error,difference.norminf());scale=std::max({scale,high.norminf(),origin.norminf()});low_peak=std::max(low_peak,low.norminf());
 }
 // Scalar extrapolation later changes the origin; grade this stage identity
 // before finish. The independent offline Decimal oracle uses the same dumps.
 Real const bound=4096.*std::numeric_limits<Real>::epsilon()*std::numeric_limits<Real>::epsilon()*std::max(1.,scale);
 amrex::Print()<<std::setprecision(17)<<"COMPENSATED_EL_GRADIENT phase="<<phase<<" error="<<error<<" bound="<<bound<<" low="<<low_peak<<"\n";
 AMREX_ALWAYS_ASSERT(error<=bound);
 bool dumps=false;amrex::ParmParse("endpoint_diagnostic").query("write_fields",dumps);if(dumps){std::string prefix=std::string("COMPENSATED_")+phase+"_"+std::to_string(w.getistep(0))+"_";for(int c=0;c<3;++c){amrex::VisMF::Write(*w.m_fields.get("hybrid_E_long_fp",Dir{c},0),prefix+"EL_hi_"+std::to_string(c));amrex::VisMF::Write(*w.m_fields.get(field_low,Dir{c},0),prefix+"EL_lo_"+std::to_string(c));amrex::VisMF::Write(*w.m_fields.get(field_origin,Dir{c},0),prefix+"EL_origin_"+std::to_string(c));}amrex::VisMF::Write(p,prefix+"phi_hi");amrex::VisMF::Write(pl,prefix+"phi_lo");amrex::VisMF::Write(p0,prefix+"phi_origin");}
}
inline void DiscardEndpointLow(WarpX& w,Real dt,Real theta){
 if(!Enabled())return;
 Real observed=0.,bound=0.;Real const unit=std::numeric_limits<Real>::epsilon();
 for(int c=0;c<3;++c){auto& low=*w.m_fields.get(field_low,Dir{c},0);auto const& h=*w.m_fields.get("hybrid_E_long_fp",Dir{c},0);observed=std::max(observed,low.norminf(0,1,low.nGrowVect()));bound=std::max(bound,unit*h.norminf(0,1,h.nGrowVect())+std::numeric_limits<Real>::denorm_min());}
 AMREX_ALWAYS_ASSERT(observed<=bound);
 amrex::Print()<<std::setprecision(17)<<"COMPENSATED_EL_PUBLICATION low_V_per_m="<<observed<<" independent_bound_V_per_m="<<bound<<" interval_D_bound_A_per_m2="<<PhysConst::epsilon_0/dt*bound<<" endpoint_D_Je_bound_A_per_m2="<<PhysConst::epsilon_0/(theta*dt)*bound<<"\n";
 // Stage capture is complete. D/Je will rotate from their captured stage,
 // untouched. The completed endpoint and every restart use published highs.
 for(int c=0;c<3;++c)w.m_fields.get(field_low,Dir{c},0)->setVal(0.);w.m_fields.get(scalar_low,0)->setVal(0.);
}
}
#endif
