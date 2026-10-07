/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace amrex::literals;
using warpx::fields::FieldType;

int main (int argc,char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w=WarpX::GetInstance();
        w.InitData();
        auto& h=*w.get_pointer_HybridPICModel();
        auto& guard=h.m_source_guard;
        auto& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto& rho=*w.m_fields.get(FieldType::rho_fp,0);
        auto& mpc=w.GetPartContainer();
        auto const& geom=w.Geom(0);
        auto const K=PhysConst::q_e/PhysConst::kb;
        amrex::ParmParse pp("source_test"), gp("hybrid_pic_model.source_guard");
        int minimum=16, axis=1, ppc=16;
        gp.query("minimum_resident_markers",minimum);
        gp.query("axis_cells",axis);
        pp.query("ppc",ppc);
        amrex::Real n=1.e19_rt,bad_n=1.e17_rt,dt=1.e-9_rt;
        bool defect=true,redirect=false,thermal=true,stopping=true,unequal=false,changed_mask=false;
        pp.query("n",n); pp.query("bad_n",bad_n); pp.query("defect",defect);
        pp.query("redirect",redirect); pp.query("thermal",thermal); pp.query("stopping",stopping);
        pp.query("unequal",unequal); pp.query("changed_mask",changed_mask);
        amrex::Real drift=0; pp.query("electron_drift",drift);
        if(unequal) {
            for(auto const* name:{"ions","ions2"}) {
                auto& pc=mpc.GetParticleContainerFromName(name);
                for(WarpXParIter pti(pc,0);pti.isValid();++pti) {
                    auto* weights=pti.GetAttribs(PIdx::w).dataPtr();
                    auto const* theta=pti.GetAttribs(PIdx::theta).dataPtr();
                    amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long p) {
                        weights[p]*=std::sin(theta[p])>0?10:1;
                    });
                }
            }
        }
        auto prescribe=[&]() {
            rho.setVal(n*PhysConst::q_e);
            for(amrex::MFIter mfi(rho);mfi.isValid();++mfi) {
                auto const r=rho.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    if(defect && i==8) r(i,j,k)=bad_n*PhysConst::q_e;
                });
            }
            rho.OverrideSync(geom.periodicity()); rho.FillBoundary(geom.periodicity());
            w.m_fields.get("rho_fp_ions",0)->setVal(.5_rt*n*PhysConst::q_e);
            w.m_fields.get("rho_fp_ions2",0)->setVal(.5_rt*n*PhysConst::q_e);
            w.m_fields.get("rho_fp_alpha",0)->setVal(0);
            for(int c=0;c<3;++c) {
                w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0)[c]->setVal(c==2?1.e5_rt:0);
                w.m_fields.get_alldirs("Ve_fp",0)[c]->setVal(c==2?drift:0);
                for(auto const* name:{"ions","ions2","alpha"})
                    w.m_fields.get_alldirs(std::string("current_fp_")+name,0)[c]->setVal(0);
            }
            te.setVal(100*K);
        };
        prescribe();
        if(guard.enabled()) {
            guard.Prepare(0,rho,h);
            amrex::MultiFab error(te.boxArray(),te.DistributionMap(),2,0);
            auto const floor=h.m_n_floor;
            for(amrex::MFIter mfi(error);mfi.isValid();++mfi) {
                auto const a=guard.nodes().const_array(mfi);
                auto const out=error.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    int expected=(ppc<minimum?2:0)|(i<=axis && axis>0?4:0);
                    if(n<=floor || (defect && bad_n<=floor && i>=7 && i<=9)) expected|=1;
                    out(i,j,k,0)=std::abs(a(i,j,k,0)-expected);
                    out(i,j,k,1)=std::abs(a(i,j,k,1)-expected);
                });
            }
            AMREX_ALWAYS_ASSERT(error.norminf(0)==0 && error.norminf(1)==0);
            auto const counts=guard.counts().norminf(0);
            AMREX_ALWAYS_ASSERT(counts==ppc);
            // Independent support points around the axis cutoff, including a
            // zero-weight excluded corner at r=(axis+1)*dr.
            if(ppc>=minimum && n>h.m_n_floor && axis>0) {
                amrex::MultiFab probes(te.boxArray(),te.DistributionMap(),1,0); probes.setVal(0);
                auto const lo=geom.ProbLoArray(), inv=geom.InvCellSizeArray(), dx=geom.CellSizeArray();
                for(amrex::MFIter mfi(probes);mfi.isValid();++mfi) {
                    auto const mask=guard.nodes().const_array(mfi);
                    auto const out=probes.array(mfi);
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        if(i!=axis) return;
                        auto const z=lo[1]+j*dx[1];
                        bool const in=HybridSourceEligibility::StoppingEligible(
                            (axis-.01_rt)*dx[0],0,z,mask,lo,inv);
                        bool const neighbor=HybridSourceEligibility::StoppingEligible(
                            (axis+.01_rt)*dx[0],0,z,mask,lo,inv);
                        // Test exact cutoff only where the array includes it.
                        bool const edge=HybridSourceEligibility::StoppingEligible(
                            (axis+1)*dx[0],0,z,mask,lo,inv);
                        out(i,j,k)=(in||neighbor||!edge)?1:0;
                    });
                }
                AMREX_ALWAYS_ASSERT(probes.norminf(0)==0);
            }
        }
        amrex::MultiFab redirected(te.boxArray(),te.DistributionMap(),3,0);
        redirected.setVal(0);
        h.QDSMCAddJouleHeating(0,dt,rho,redirect?&redirected:nullptr);
        amrex::MultiFab error(te.boxArray(),te.DistributionMap(),1,0);
        auto const guarded=guard.guards(HybridSourceEligibility::Ohmic);
        auto const* pedestal=h.DensityPedestal(0);
        for(amrex::MFIter mfi(error);mfi.isValid();++mfi) {
            auto const e=error.array(mfi);
            auto const t=te.const_array(mfi), r=rho.const_array(mfi);
            auto const mask=guarded?guard.nodes().const_array(mfi):amrex::Array4<amrex::Real const>{};
            auto const ped=pedestal?pedestal->const_array(mfi):amrex::Array4<amrex::Real const>{};
            bool const has_ped=pedestal!=nullptr;
            auto const red=redirected.const_array(mfi);
            auto const floor=h.m_n_floor;
            auto const gm1=h.m_gamma-1;
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                bool const blocked=r(i,j,k)<=floor*PhysConst::q_e || (guarded && mask(i,j,k,0)!=0);
                auto const cap=(r(i,j,k)+(has_ped?ped(i,j,k):0))*PhysConst::kb/(PhysConst::q_e*gm1);
                auto const expected=100*K+(!blocked&&!redirect?dt*1.e5_rt/cap:0);
                e(i,j,k)=std::abs(t(i,j,k)-expected);
                if(blocked && (red(i,j,k,0)!=0||red(i,j,k,1)!=0||red(i,j,k,2)!=0)) e(i,j,k)=1.e30;
            });
        }
        auto const error_K=error.norminf(0);
        AMREX_ALWAYS_ASSERT(error_K<=1024*std::numeric_limits<amrex::Real>::epsilon()*te.norminf(0));
        AMREX_ALWAYS_ASSERT(h.m_eta(1.0_rt,1.0_rt,0.0_rt)==1.e-5_rt);
        if(redirect) {
            auto const relax=h.m_include_temperature_relaxation;
            h.m_include_temperature_relaxation=false;
            h.QDSMCApplyIonHeating(0,dt,&redirected,nullptr);
            h.m_include_temperature_relaxation=relax;
            if(guard.guards(HybridSourceEligibility::Ohmic) && ppc<minimum)
                AMREX_ALWAYS_ASSERT(guard.ionEnergy()==0);
        }
        if(thermal) {
            te.setVal(100*K);
            std::map<std::string,std::unique_ptr<amrex::MultiFab>> owned;
            std::map<std::string,amrex::MultiFab*> ti;
            for(auto const* name:{"ions","ions2"}) {
                auto& a=owned[name];
                a=std::make_unique<amrex::MultiFab>(amrex::convert(te.boxArray(),
                    amrex::IntVect::TheCellVector()),te.DistributionMap(),1,0);
                a->setVal(10); ti[name]=a.get();
            }
            h.QDSMCAddTemperatureRelaxation(0,dt,rho,ti);
            if(guard.guards(HybridSourceEligibility::Thermal)) {
                for(amrex::MFIter mfi(error);mfi.isValid();++mfi) {
                    auto const out=error.array(mfi);
                    auto const t=te.const_array(mfi);
                    auto const mask=guard.nodes().const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        out(i,j,k)=mask(i,j,k,1)!=0?std::abs(t(i,j,k)-100*K):0;
                    });
                }
                AMREX_ALWAYS_ASSERT(error.norminf(0)==0);
            }
            h.QDSMCApplyIonHeating(0,dt,nullptr,&ti);
            if(guard.guards(HybridSourceEligibility::Thermal) && (ppc<minimum || n<=h.m_n_floor)) {
                AMREX_ALWAYS_ASSERT(guard.electronEnergy(HybridSourceEligibility::Thermal)==0);
                AMREX_ALWAYS_ASSERT(guard.ionEnergy()==0);
            }
            guard.Phase(0,"fixture_thermal",h);
        }
        if(stopping) {
            prescribe();
            auto& alpha=mpc.GetParticleContainerFromName("alpha");
            auto const before=alpha.sumParticleEnergy();
            mpc.doCollisions(0,0.0_rt,1.e-3_rt);
            auto const after=alpha.sumParticleEnergy(), loss=before-after;
            auto& stage=h.GetFastIonHeatingStaging(0);
            amrex::MultiFab sum(stage.boxArray(),stage.DistributionMap(),1,0);
            amrex::MultiFab::Copy(sum,stage,0,0,1,0); sum.SumBoundary(geom.periodicity());
            auto const staged=h.EnergyVolumeIntegral(sum,0,0);
            auto const tol=4096*std::numeric_limits<amrex::Real>::epsilon()*before;
            AMREX_ALWAYS_ASSERT(std::abs(staged-loss)<=tol);
            if(guard.guards(HybridSourceEligibility::Stopping) && (ppc<minimum||n<=h.m_n_floor))
                AMREX_ALWAYS_ASSERT(loss==0 && staged==0);
            amrex::MultiFab capacity(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
            capacity.setVal(2.e19_rt*PhysConst::q_e);
            if(changed_mask) { rho.setVal(0); guard.Prepare(0,rho,h); }
            h.QDSMCApplyFastIonHeating(0,nullptr,&capacity,.5_rt);
            auto const half=guard.electronEnergy(HybridSourceEligibility::Stopping);
            capacity.setVal(4.e19_rt*PhysConst::q_e);
            h.QDSMCApplyFastIonHeating(0,nullptr,&capacity,1.0_rt);
            auto const full=half+guard.electronEnergy(HybridSourceEligibility::Stopping);
            AMREX_ALWAYS_ASSERT(stage.norminf(0)==0);
            if(guard.guards(HybridSourceEligibility::Stopping)) AMREX_ALWAYS_ASSERT(std::abs(full-loss)<=1.e-8_rt*std::max(std::abs(loss),1.e-20_rt)+tol);
            amrex::Print()<<std::setprecision(17)<<"SOURCE_GUARD_STOPPING loss="<<loss
                <<" staged="<<staged<<" delivered="<<full<<" residual="<<full-loss<<'\n';
        }
        amrex::Print()<<std::setprecision(17)<<"SOURCE_GUARD_PASS epoch="<<guard.epoch()
            <<" joule_error_K="<<error_K<<" declined_J="<<guard.declinedOhmic()
            <<" unresolved_species_nodes="<<guard.unevaluableOhmic()<<'\n';
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
