/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/BraginskiiConductivity.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>

// Independent continuum reference: direct rational form and fine-coordinate
// flux differences. The native operator is never used to construct its oracle.
AMREX_GPU_HOST_DEVICE amrex::Real temp(amrex::Real r, amrex::Real z) {
    return 1000.0*(1.0+0.01*std::cos(2.0*MathConst::pi*r)*std::cos(2.0*MathConst::pi*z));
}
AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real,2>
flux(amrex::Real r, amrex::Real z, int mode) {
    amrex::Real const T=temp(r,z), twopi=2*MathConst::pi;
    amrex::Real const tr=-10*twopi*std::sin(twopi*r)*std::cos(twopi*z)*PhysConst::q_e/PhysConst::kb;
    amrex::Real const tz=-10*twopi*std::cos(twopi*r)*std::sin(twopi*z)*PhysConst::q_e/PhysConst::kb;
    // null: divergence-free poloidal field with null at (r=0,z=0.5).
    amrex::Real const br=mode==0?0.0:mode==1?0.00006:0.001*r;
    amrex::Real const bz=mode==0?0.0:mode==1?0.00008:0.001*(1-2*z);
    amrex::Real const b2=br*br+bz*bz;
    amrex::Real const tau=3.44e5*std::pow(T,1.5)/(3.e15*10.0);
    amrex::Real const x2=std::pow(PhysConst::q_e*tau/PhysConst::m_e,2)*b2;
    amrex::Real const scale=3.e21*PhysConst::kb*PhysConst::q_e*T*tau/PhysConst::m_e;
    amrex::Real const kp=scale*(4.664*x2+11.92)/(x2*x2+14.79*x2+3.7703);
    amrex::Real const kl=amrex::min(scale*11.92/3.7703,1.e5*kp);
    amrex::Real const diff=b2>0?(kl-kp)/b2:0;
    return {(kp+diff*br*br)*tr+diff*br*bz*tz,
            diff*br*bz*tr+(kp+diff*bz*bz)*tz};
}
AMREX_GPU_HOST_DEVICE amrex::Real reference(amrex::Real r,amrex::Real z,int mode,amrex::Real capfactor) {
    amrex::Real constexpr h=1.e-5;
    auto const rp=flux(r+h,z,mode),rm=flux(r-h,z,mode);
    auto const zp=flux(r,z+h,mode),zm=flux(r,z-h,mode);
    return (( (r+h)*rp[0]-(r-h)*rm[0])/(2*h*r)+(zp[1]-zm[1])/(2*h)) /
           (1.5*3.e21*capfactor*PhysConst::kb);
}
int main(int argc,char** argv) {
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& sim=WarpX::GetInstance(); sim.InitData();
        auto& hp=*sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te=*sim.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto& rho=*sim.m_fields.get(FieldType::rho_fp,0);
        auto const& geom=sim.Geom(0); auto dx=geom.CellSizeArray(); auto plo=geom.ProbLoArray();
        int mode=0; amrex::Real capfactor=1;
        amrex::ParmParse pp("physical_test");pp.query("mode",mode);pp.query("capacity_factor",capfactor);
        hp.m_qdsmc_n_floor=3.e21*capfactor;
        hp.m_qdsmc_halo_unfreeze=true;
        hp.m_cond_te_floor=0;
        for(int d=0;d<AMREX_SPACEDIM;++d)for(int s=0;s<2;++s)hp.m_cond_bc[d][s]=0;
        for(int d=0;d<3;++d){
            auto& f=*sim.m_fields.get(FieldType::Bfield_fp,ablastr::fields::Direction{d},0);
            amrex::Real const ro=f.ixType().nodeCentered(0)?0:.5, zo=f.ixType().nodeCentered(1)?0:.5;
            for(amrex::MFIter mfi(f);mfi.isValid();++mfi){auto a=f.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    amrex::Real const r=plo[0]+(i+ro)*dx[0],z=plo[1]+(j+zo)*dx[1];
                    a(i,j,k)=d==0?(mode==0?0:mode==1?.00006:.001*r):d==2?(mode==0?0:mode==1?.00008:.001*(1-2*z)):0;
                });
            }
        }
        for(amrex::MFIter mfi(te);mfi.isValid();++mfi){auto t=te.array(mfi);auto n=rho.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){t(i,j,k)=temp(plo[0]+i*dx[0],plo[1]+j*dx[1])*PhysConst::q_e/PhysConst::kb;});
            amrex::ParallelFor(rho[mfi].box(),[=] AMREX_GPU_DEVICE(int i,int j,int k){n(i,j,k)=3.e21*PhysConst::q_e;});
        }
        amrex::MultiFab initial(te.boxArray(),te.DistributionMap(),1,te.nGrowVect());
        amrex::MultiFab::Copy(initial,te,0,0,1,te.nGrowVect());
        amrex::MultiFab result(te.boxArray(),te.DistributionMap(),3,0);
        auto const c=EvaluateBraginskiiConductivity(3.e21,1010,0,1.e5,10);
        amrex::Real const dt=.005*dx[0]*dx[0]/(c.parallel_physical/(1.5*3.e21*capfactor*PhysConst::kb));
        amrex::Real residual=0;int stages=0;
        for(int n=0;n<3;++n){
            amrex::MultiFab::Copy(te,initial,0,0,1,te.nGrowVect());
            hp.QdsmcConductionOnceFDAtState(0,dt/std::pow(2,n),true,rho,0);
            auto const& rep=hp.m_cond_last_report;
            residual=amrex::max(residual,std::abs(rep.residual)/rep.energy_before[0]);stages+=rep.accepted;
            amrex::MultiFab::Copy(result,te,0,n,1,0);
        }
        auto own=amrex::OwnerMask(te,geom.periodicity());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real,amrex::Real,amrex::Real> data(op);
        using Tuple=typename decltype(data)::Type;
        for(amrex::MFIter mfi(te);mfi.isValid();++mfi){auto a=result.const_array(mfi);auto b=initial.const_array(mfi);auto o=own->const_array(mfi);
            op.eval(mfi.tilebox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple{
                amrex::Real const r=plo[0]+i*dx[0],z=plo[1]+j*dx[1];
                if(!o(i,j,k)||r<.2||r>.8||z<.2||z>.8)return {0.,0.,0.};
                amrex::Real const exact=reference(r,z,mode,capfactor);
                amrex::Real const coarse=(4*(a(i,j,k,1)-b(i,j,k))-(a(i,j,k,0)-b(i,j,k)))/dt;
                amrex::Real const fine=(4*(a(i,j,k,2)-b(i,j,k))-(a(i,j,k,1)-b(i,j,k)))/(dt*.5);
                return {r*(fine-exact)*(fine-exact),r*exact*exact,r*(fine-coarse)*(fine-coarse)};
            });
        }
        auto v=data.value(op);amrex::Real sums[3]={amrex::get<0>(v),amrex::get<1>(v),amrex::get<2>(v)};
        amrex::ParallelDescriptor::ReduceRealSum(sums,3);
        amrex::Print()<<"PHYSICAL_OPERATOR mode="<<mode<<" n="<<geom.Domain().length(0)<<" capacity_factor="<<capfactor
          <<" relative_L2="<<std::sqrt(sums[0]/sums[1])<<" temporal_sensitivity="<<std::sqrt(sums[2]/sums[1])
          <<" energy_residual="<<residual<<" stages="<<stages<<" min_Te_K="<<te.min(0)<<"\n";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::sqrt(sums[0]/sums[1])<.03 && residual<1.e-11 && te.min(0)>0,"Physical continuum/conservation test failed");
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
