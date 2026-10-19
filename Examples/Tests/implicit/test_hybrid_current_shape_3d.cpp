/* Native charge/current continuity across a controlled EB shape interface. */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/WarpXConst.H"
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <AMReX_Utility.H>
#include <algorithm>
#include <array>
#include <memory>
#include <iomanip>
using namespace amrex::literals;
void position(WarpX& sim, amrex::Real shift, int axis) {
 auto& pc=sim.GetPartContainer().GetParticleContainerFromName("ions");
 for(WarpXParIter pti(pc,0);pti.isValid();++pti) {
  auto set=SetParticlePosition(pti);
  amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE (int i) noexcept {
   amrex::ParticleReal x=0.0123_prt,y=0.0147_prt,z=0.0161_prt;
   if(axis==0)x=0._prt+shift;
   if(axis==1)y=0._prt+shift;
   if(axis==2)z=0._prt+shift;
   set(i,x,y,z);
  });
 }
}

using MF=amrex::MultiFab;
using Currents=std::array<std::unique_ptr<MF>,3>;
std::unique_ptr<MF> copy(MF const& src) {
 auto dst=std::make_unique<MF>(src.boxArray(),src.DistributionMap(),1,src.nGrowVect());
 MF::Copy(*dst,src,0,0,1,src.nGrowVect());return dst;
}
amrex::Real difference(MF const& a,MF const& b) {
 MF delta(a.boxArray(),a.DistributionMap(),1,0);MF::Copy(delta,a,0,0,1,0);MF::Subtract(delta,b,0,0,1,0);return delta.norm0();
}
amrex::Real current_norm(Currents const& j) {
 return std::max({j[0]->norm0(),j[1]->norm0(),j[2]->norm0()});
}
amrex::Real current_difference(Currents const& a,Currents const& b) {
 return std::max({difference(*a[0],*b[0]),difference(*a[1],*b[1]),difference(*a[2],*b[2])});
}
int main(int argc,char**argv) {
 warpx::initialization::initialize_external_libraries(argc,argv);bool passed=true;
 {
  auto& sim=WarpX::GetInstance();sim.InitData();
  int axis=0;amrex::ParmParse("audit").get("axis",axis);AMREX_ALWAYS_ASSERT(axis>=0&&axis<3);
  auto& pc=sim.GetPartContainer().GetParticleContainerFromName("ions");
  AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles()==1);
  auto& mask=*sim.GetEBReduceParticleShapeFlag()[0];
  auto const u=0.001_rt*PhysConst::c;
  auto const v=u/std::sqrt(1._rt+u*u/(PhysConst::c*PhysConst::c));
  auto const qwv=PhysConst::q_e*1.e12_rt*v;
  using warpx::fields::FieldType;using ablastr::fields::Direction;
  auto& rho=*sim.m_fields.get(FieldType::rho_fp,0);
  auto const dx=sim.Geom(0).CellSizeArray();auto const volume=dx[0]*dx[1]*dx[2];
  auto const domain=amrex::convert(sim.Geom(0).Domain(),amrex::IntVect(1));
  auto const lo=amrex::lbound(domain),hi=amrex::ubound(domain);
  std::array<std::array<Currents,3>,2> snapshots;
  for(int mode:{1,0}) {
   // A controlled plane isolates the shape switch from wall geometry. The
   // trajectory and entire support lie in the open interior of the EB cylinder.
   for(amrex::MFIter mfi(mask);mfi.isValid();++mfi){auto a=mask.array(mfi);
    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k)noexcept{
     int q=axis==0?i:axis==1?j:k;a(i,j,k)=mode&&q<8;
    });
   }
   for(int level=0;level<3;++level) {
    auto const dt=1.e-9_rt/(1<<level);sim.setdt(dt);position(sim,0._rt,axis);
    sim.HybridPICDepositRhoAndJ();
    auto& j=snapshots[mode][level];
    for(int c=0;c<3;++c)j[c]=copy(*sim.m_fields.get(FieldType::current_fp,Direction{c},0));
    auto const rho_new=copy(rho);
    position(sim,-dt*v,axis);sim.HybridPICDepositRhoAndJ();
    MF rate(rho.boxArray(),rho.DistributionMap(),1,0);
    MF div(rho.boxArray(),rho.DistributionMap(),1,0);
    MF residual(rho.boxArray(),rho.DistributionMap(),1,0);
    for(amrex::MFIter mfi(rho);mfi.isValid();++mfi) {
     auto const rn=rho_new->const_array(mfi),ro=rho.const_array(mfi);
     auto const jx=j[0]->const_array(mfi),jy=j[1]->const_array(mfi),jz=j[2]->const_array(mfi);
     auto const a=rate.array(mfi),b=div.array(mfi),e=residual.array(mfi);
     amrex::ParallelFor(mfi.validbox(),[=]AMREX_GPU_DEVICE(int i,int k,int l)noexcept{
      a(i,k,l)=b(i,k,l)=e(i,k,l)=0._rt;
      if(i<=lo.x||i>=hi.x||k<=lo.y||k>=hi.y||l<=lo.z||l>=hi.z)return;
      a(i,k,l)=(rn(i,k,l)-ro(i,k,l))/dt;
      b(i,k,l)=(jx(i,k,l)-jx(i-1,k,l))/dx[0]+(jy(i,k,l)-jy(i,k-1,l))/dx[1]+(jz(i,k,l)-jz(i,k,l-1))/dx[2];
      e(i,k,l)=a(i,k,l)+b(i,k,l);
     });
    }
    auto const closure=residual.norm0()/std::max({rate.norm0(),div.norm0(),1.e-300_rt});
    amrex::Real integrated_error=0._rt;
    for(int c=0;c<3;++c) {
     auto const total=j[c]->sum_unique(0,false,sim.Geom(0).periodicity())*volume;
     integrated_error=std::max(integrated_error,std::abs(total-(c==axis?qwv:0._rt))/qwv);
    }
    bool const ok=std::isfinite(closure)&&closure<1.e-10_rt&&integrated_error<1.e-10_rt;
    passed=passed&&ok;
    amrex::Print()<<std::setprecision(17)<<"HYBRID_CURRENT_CONTINUITY {\"axis\":"<<axis<<",\"order\":"<<WarpX::nox<<",\"mask\":"<<mode<<",\"factor\":"<<(1<<level)<<",\"continuity_relative_max\":"<<closure<<",\"current_integral_relative_max\":"<<integrated_error<<",\"passed\":"<<(ok?"true":"false")<<"}\n";
   }
  }
  for(int level=0;level<3;++level) {
   auto const error=current_difference(snapshots[0][level],snapshots[1][level])/std::max(current_norm(snapshots[0][level]),1.e-300_rt);
   passed=passed&&std::isfinite(error)&&error<1.e-11_rt;
   amrex::Print()<<"HYBRID_CURRENT_MASK {\"factor\":"<<(1<<level)<<",\"relative_max\":"<<error<<"}\n";
  }
  for(int mode:{0,1}) {
   auto const scale=std::max(current_norm(snapshots[mode][2]),1.e-300_rt);
   auto const coarse=current_difference(snapshots[mode][0],snapshots[mode][1])/scale;
   auto const fine=current_difference(snapshots[mode][1],snapshots[mode][2])/scale;
   bool const ok=std::isfinite(coarse)&&std::isfinite(fine)&&(fine<coarse||std::max(coarse,fine)<1.e-11_rt);
   passed=passed&&ok;
   amrex::Print()<<"HYBRID_CURRENT_REFINEMENT {\"mask\":"<<mode<<",\"coarse\":"<<coarse<<",\"fine\":"<<fine<<",\"passed\":"<<(ok?"true":"false")<<"}\n";
  }
  amrex::Print()<<"HYBRID_CURRENT_RESULT {\"passed\":"<<(passed?"true":"false")<<"}\n";
 }
 WarpX::ResetInstance();warpx::initialization::finalize_external_libraries();return passed?0:2;
}
