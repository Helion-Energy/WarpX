/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL. */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>
#include <limits>
using namespace amrex::literals;
int main(int argc,char** argv) {
 warpx::initialization::initialize_external_libraries(argc,argv);
 int result=0;
 {
  auto& w=WarpX::GetInstance();w.InitData();auto& hp=*w.get_pointer_HybridPICModel();
  using warpx::fields::FieldType;
  auto& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
  auto& rho=*w.m_fields.get(FieldType::rho_fp,0);
  auto& oldrho=*w.m_fields.get(FieldType::hybrid_rho_fp_temp,0);
  auto vel=w.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp,0);
  auto const& g=w.Geom(0);auto const dx=g.CellSizeArray(),lo=g.ProbLoArray();
  constexpr amrex::Real nf=3.317472633125625e20_rt;
  constexpr amrex::Real t0=100._rt*PhysConst::q_e/PhysConst::kb;
  for(int d=0;d<3;++d)vel[d]->setVal(0._rt);
  for(amrex::MFIter mfi(te);mfi.isValid();++mfi) {
   auto const tt=te.array(mfi),rr=rho.array(mfi),ro=oldrho.array(mfi);
   amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
    amrex::Real const x=lo[0]+i*dx[0],y=lo[1]+j*dx[1],z=lo[2]+k*dx[2];
    // A smooth, stationary density crosses the physical floor in all three
    // directions, including internal MPI box seams. Uniform Te must survive.
    amrex::Real const n=nf*(1.05_rt+.70_rt*std::sin(2._rt*MathConst::pi*x/.6_rt)*std::cos(2._rt*MathConst::pi*y/.6_rt)*std::cos(2._rt*MathConst::pi*z/.6_rt));
    rr(i,j,k)=PhysConst::q_e*n;ro(i,j,k)=rr(i,j,k);tt(i,j,k)=t0;
   });
  }
  rho.FillBoundary(g.periodicity());oldrho.FillBoundary(g.periodicity());te.FillBoundary(g.periodicity());hp.EnsureDensityPedestal();
  int iterations=10;amrex::ParmParse("identity").query("iterations",iterations);
  amrex::MultiFab error(te.boxArray(),te.DistributionMap(),1,0);
  for(int step=1;step<=iterations;++step) {
   hp.QdsmcTransportOnce(0,0._rt,true);hp.m_qdsmc_pc->ResetParticles(0);
   for(amrex::MFIter mfi(te);mfi.isValid();++mfi) {
    auto const a=te.const_array(mfi);auto const e=error.array(mfi);
    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
     amrex::Real const x=lo[0]+i*dx[0],y=lo[1]+j*dx[1];
     e(i,j,k)=x*x+y*y<.10_rt*.10_rt?std::abs(a(i,j,k)-t0):0._rt;
    });
   }
   amrex::Real const err=error.norminf()/t0;
   amrex::Print()<<std::setprecision(17)<<"PEDESTAL_TRANSPORT_IDENTITY step="<<step<<" relative_error="<<err<<" maxTe_eV="<<te.max(0)*PhysConst::kb/PhysConst::q_e<<"\n";
   if(!te.is_finite() || err>512._rt*step*std::numeric_limits<amrex::Real>::epsilon())result=1;
  }
 }
 WarpX::ResetInstance();warpx::initialization::finalize_external_libraries();return result;
}
