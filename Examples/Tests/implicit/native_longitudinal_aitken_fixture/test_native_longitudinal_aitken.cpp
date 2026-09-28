/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/NativeLongitudinalAitken.H"
#include "FieldSolver/ImplicitSolvers/DarwinVacuumAffineResponse.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <iomanip>
using MF=amrex::MultiFab;using Field=amrex::Array<MF,3>;using View=amrex::Array<MF*,3>;
using Aitken=warpx::darwin::NativeLongitudinalAitken;using Core=warpx::darwin::DarwinVacuumAffineResponse;
View V(Field& f){return {&f[0],&f[1],&f[2]};}
Field Clone(View const& f){Field a;for(int c=0;c<3;++c){a[c].define(f[c]->boxArray(),f[c]->DistributionMap(),1,1);a[c].setVal(0.);}return a;}
void Fill(Field& f,double a,double b){for(int c=0;c<3;++c)for(amrex::MFIter mfi(f[c]);mfi.isValid();++mfi){auto x=f[c].array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){x(i,j,k)=(c==0?a:(c==2?b:0.))*(1.+.0625*i+.03125*j+.015625*k);});}}
void Test(WarpX& w){
 auto e=w.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp,0),b=w.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
 amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};Core::Options opt;
 for(int d=0;d<AMREX_SPACEDIM;++d){lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;opt.pmc_lo[d]=lo[d];opt.pmc_hi[d]=hi[d];}
 Aitken accel(e,w.Geom(0),lo,hi);Core native(w,e,b,opt);
 auto x=Clone(e),y=Clone(e);Fill(x,.375,-.125);Fill(y,-.25,.5);
 double const dot=accel.WeightedDot(V(x),V(y)),reference=native.MetricDot(V(x),V(y));
 AMREX_ALWAYS_ASSERT(std::abs(dot-reference)<=2.e-14*std::max(1.,std::abs(reference)));
 // Single affine mode gives the exact inverse slope, including owner seams.
 Fill(x,1.,0.);auto first=accel.Select(V(x),.125,false);AMREX_ALWAYS_ASSERT(first.valid&&!first.secant&&first.omega==.125);
 Fill(x,-.5,0.);auto next=accel.Select(V(x),.125,true);AMREX_ALWAYS_ASSERT(next.valid&&next.secant&&std::abs(next.omega-1./12.)<1.e-14);
 accel.Reset();Fill(x,.25,-.75);auto reset=accel.Select(V(x),.125,true);AMREX_ALWAYS_ASSERT(reset.valid&&!reset.secant&&reset.omega==.125);
 auto same=accel.Select(V(x),.125,true);AMREX_ALWAYS_ASSERT(same.valid&&!same.secant&&same.omega==.125);
 // Exact slow positive slope: stationary omega=2 exceeds the allowed
 // upper bound. OFF retains fallback; ON takes the constrained minimum 1.
 Aitken capped(e,w.Geom(0),lo,hi),uncapped(e,w.Geom(0),lo,hi);
 Fill(x,1.,0.);capped.Select(V(x),.125,false,true);uncapped.Select(V(x),.125,false);
 Fill(x,.9375,0.);auto cap=capped.Select(V(x),.125,true,true),off=uncapped.Select(V(x),.125,true);
 AMREX_ALWAYS_ASSERT(cap.valid&&cap.secant&&cap.omega==1.&&off.valid&&!off.secant&&off.omega==.125);
 // Non-descent inverse secants and exactly unresolved differences keep the
 // old fallback, even when the positive clamp is enabled.
 capped.Reset();Fill(x,1.,0.);capped.Select(V(x),.125,false,true);
 Fill(x,1.0625,0.);auto negative=capped.Select(V(x),.125,true,true);
 AMREX_ALWAYS_ASSERT(negative.valid&&!negative.secant&&negative.omega==.125);
 auto unresolved=capped.Select(V(x),.125,true,true);
 AMREX_ALWAYS_ASSERT(unresolved.valid&&!unresolved.secant&&unresolved.omega==.125);
 Fill(x,.25,-.75);
 // Failure must not overwrite secant memory; compare with a clean twin.
 Aitken twin(e,w.Geom(0),lo,hi);twin.Select(V(x),.125,false);
 for(auto& f:x) { f.setVal(std::numeric_limits<double>::quiet_NaN()); }
 auto bad=accel.Select(V(x),.125,true);AMREX_ALWAYS_ASSERT(!bad.valid);
 Fill(x,.125,-.375);auto a=accel.Select(V(x),.125,true),z=twin.Select(V(x),.125,true);AMREX_ALWAYS_ASSERT(a.valid&&z.valid&&a.omega==z.omega);
 // Two positive slopes represent the measured difficult PMC iteration,
 // but use exact 4 and12 independent of the retained application fit.
 accel.Reset();double xa=1.,xb=1.;int iterations=0;
 for(;iterations<20;++iterations){if(std::max(std::abs(xa),std::abs(xb))<1.e-12)break;Fill(x,-12.*xa,-4.*xb);auto q=accel.Select(V(x),.125,true);AMREX_ALWAYS_ASSERT(q.valid);xa+=q.omega*(-12.*xa);xb+=q.omega*(-4.*xb);}
 AMREX_ALWAYS_ASSERT(iterations<20&&std::max(std::abs(xa),std::abs(xb))<1.e-12);
 amrex::Print()<<std::setprecision(17)<<"AITKEN_NATIVE_PASS dot="<<dot<<" reference="<<reference<<" two_mode_iterations="<<iterations<<" error="<<std::max(std::abs(xa),std::abs(xb))<<"\n";
}
int main(int argc,char** argv){warpx::initialization::initialize_external_libraries(argc,argv);{auto& w=WarpX::GetInstance();w.InitData();Test(w);WarpX::Finalize();}warpx::initialization::finalize_external_libraries();}
