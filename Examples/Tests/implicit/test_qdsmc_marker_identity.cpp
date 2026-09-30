/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 *
 * Native RZ raw-marker identity and ownership regression. No physical time
 * advances: the analytic grid state is held fixed except for marker recovery.
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fluids/QdsmcParticleContainer.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleUtil.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_FArrayBox.H>
#include <array>
#include <algorithm>
#include <limits>
#include <AMReX_VisMF.H>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse pp("identity");
        int profile=1, iterations=800, mode=0;
        pp.query("profile",profile);
        pp.query("iterations",iterations);
        pp.query("mode",mode);
        int require_identity=0;
        pp.query("require_identity",require_identity);
        int ownership_push=0;
        pp.query("ownership_push",ownership_push);
        auto& w=WarpX::GetInstance();
        w.InitData();
        auto& hp=*w.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,0);
        auto& rho=*w.m_fields.get(FieldType::rho_fp,0);
        auto& oldrho=*w.m_fields.get(FieldType::hybrid_rho_fp_temp,0);
        auto& ke=*w.m_fields.get(FieldType::hybrid_entropy_fp,0);
        auto& weights=*w.m_fields.get(FieldType::hybrid_qdsmc_weights_fp,0);
        auto vel=w.m_fields.get_alldirs(FieldType::hybrid_electron_velocity_fp,0);
        auto const& g=w.Geom(0);
        auto const dx=g.CellSizeArray();
        auto const lo=g.ProbLoArray();
        auto const R=g.ProbHi(0);
        auto const Lz=g.ProbLength(1);
        for(int d=0;d<AMREX_SPACEDIM;++d) amrex::Print()<<std::setprecision(17)
            <<"BOUNDS "<<d<<" "<<g.ProbLo(d)<<" "<<g.ProbHi(d)<<" "
            <<g.ProbLoArrayInParticleReal()[d]<<" "<<g.ProbHiArrayInParticleReal()[d]<<"\n";
        double const kelvin=PhysConst::q_e/PhysConst::kb;
        for(int c=0;c<3;++c) vel[c]->setVal(0.);
        for(amrex::MFIter mfi(te);mfi.isValid();++mfi){
            auto const tt=te.array(mfi), rr=rho.array(mfi), ro=oldrho.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                auto const r=(lo[0]+i*dx[0])/R;
                rr(i,j,k)=PhysConst::q_e*1.e20*((profile==2||profile==3)?1.+.25*r*r:1.);
                ro(i,j,k)=rr(i,j,k);
                auto const z=(j*dx[1])/Lz;
                tt(i,j,k)=kelvin*(100.+((profile==1||profile==3||profile>=4)?10.*r*r:0.)
                    +(profile==4 ? 5.*std::sin(2.*3.14159265358979323846*z) : 0.)
                    +(profile==5 ? 5.*z*z : 0.));
            });
        }
        te.FillBoundary(g.periodicity());
        rho.FillBoundary(g.periodicity());
        oldrho.FillBoundary(g.periodicity());
        amrex::MultiFab initial(te.boxArray(),te.DistributionMap(),1,te.nGrowVect());
        amrex::MultiFab::Copy(initial,te,0,0,1,te.nGrowVect());
        int const owned_r=g.Domain().length(0)+(g.isPeriodic(0)?0:1);
        int const owned_z=g.Domain().length(1)+(g.isPeriodic(1)?0:1);
        amrex::Long const expected_count=amrex::Long(owned_r)*owned_z;
        AMREX_ALWAYS_ASSERT(hp.m_qdsmc_pc->TotalNumberOfParticles()==expected_count);
        auto markers=[&](std::string const& label){
            amrex::Vector<int> home_counts(expected_count,0);
            std::ofstream f("markers_"+label+"_rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".csv");
            f<<std::setprecision(17)<<"grid,tile,x,z,x_home,z_home,cell_i,cell_j,entropy,weight\n";
            for(QdsmcParticleContainer::iterator pti(*hp.m_qdsmc_pc,0);pti.isValid();++pti){
                auto const& a=pti.GetStructOfArrays().GetRealData();
                std::array<amrex::Gpu::PinnedVector<amrex::ParticleReal>,QdsmcPIdx::nattribs> h;
                for(int comp=0;comp<QdsmcPIdx::nattribs;++comp){
                    h[comp].resize(pti.numParticles());
                    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,a[comp].begin(),
                        a[comp].begin()+pti.numParticles(),h[comp].begin());
                }
                amrex::Gpu::streamSynchronize();
                struct HostParticle {
                    amrex::GpuArray<amrex::ParticleReal,AMREX_SPACEDIM> position;
                    amrex::ParticleReal pos(int d) const { return position[d]; }
                };
                for(amrex::Long ip=0;ip<pti.numParticles();++ip){
                    HostParticle const particle{{h[QdsmcPIdx::x][ip],h[QdsmcPIdx::z][ip]}};
                    auto const iv=amrex::DefaultAssignor{}(particle,g.ProbLoArray(),g.InvCellSizeArray(),g.Domain());
                    AMREX_ALWAYS_ASSERT(g.Domain().contains(iv));
                    int const hi=int(std::lround((h[QdsmcPIdx::x_node][ip]-g.ProbLo(0))*g.InvCellSize(0)));
                    int const hj=int(std::lround((h[QdsmcPIdx::z_node][ip]-g.ProbLo(1))*g.InvCellSize(1)));
                    AMREX_ALWAYS_ASSERT(hi>=0 && hi<owned_r && hj>=0 && hj<owned_z);
                    ++home_counts[hi+owned_r*hj];
                    if(label!="-1") {
                        AMREX_ALWAYS_ASSERT(particle.pos(0)==h[QdsmcPIdx::x_node][ip]);
                        AMREX_ALWAYS_ASSERT(particle.pos(1)==h[QdsmcPIdx::z_node][ip]);
                    }
                    f<<pti.index()<<","<<pti.LocalTileIndex()<<","<<particle.pos(0)<<","<<particle.pos(1)<<","
                     <<h[QdsmcPIdx::x_node][ip]<<","<<h[QdsmcPIdx::z_node][ip]<<","<<iv[0]<<","<<iv[1]<<","
                     <<h[QdsmcPIdx::entropy][ip]<<","<<h[QdsmcPIdx::np_real][ip]<<"\n";
                }
            }
            amrex::ParallelDescriptor::ReduceIntSum(home_counts.data(),int(home_counts.size()));
            for(int const count:home_counts) AMREX_ALWAYS_ASSERT(count==1);
        };
        auto snapshot=[&](int step){
            markers(std::to_string(step));
            amrex::VisMF::Write(te,"temperature_"+std::to_string(step));
            std::ofstream f("field_"+std::to_string(step)+"_rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".csv");
            f<<std::setprecision(17)<<"i,j,initial_eV,Te_eV\n";
            for(amrex::MFIter mfi(te);mfi.isValid();++mfi){
                amrex::FArrayBox ht(te[mfi].box(),1,amrex::The_Pinned_Arena());
                amrex::FArrayBox ht0(initial[mfi].box(),1,amrex::The_Pinned_Arena());
                amrex::Gpu::dtoh_memcpy_async(ht.dataPtr(),te[mfi].dataPtr(),te[mfi].size()*sizeof(amrex::Real));
                amrex::Gpu::dtoh_memcpy_async(ht0.dataPtr(),initial[mfi].dataPtr(),initial[mfi].size()*sizeof(amrex::Real));
                amrex::Gpu::streamSynchronize();
                auto const t=ht.const_array(), t0=ht0.const_array();
                auto const b=mfi.validbox();
                for(int j=b.smallEnd(1);j<=b.bigEnd(1);++j)
                    for(int i=b.smallEnd(0);i<=b.bigEnd(0);++i)
                        f<<i<<","<<j<<","<<t0(i,j,0)/kelvin<<","<<t(i,j,0)/kelvin<<"\n";
            }
            amrex::MultiFab delta(te.boxArray(),te.DistributionMap(),1,0);
            amrex::MultiFab::Copy(delta,te,0,0,1,0);
            amrex::MultiFab::Subtract(delta,initial,0,0,1,0);
            auto const actual_count=hp.m_qdsmc_pc->TotalNumberOfParticles();
            AMREX_ALWAYS_ASSERT(actual_count==expected_count);
            amrex::Real const max_error=delta.norminf();
            amrex::Real const epsilon=amrex::max(
                amrex::Real(std::numeric_limits<amrex::Real>::epsilon()),
                amrex::Real(std::numeric_limits<amrex::ParticleReal>::epsilon()));
            // The new null-test bound scales only with roundoff and calls;
            // it is not a relaxation of a transport accuracy requirement.
            amrex::Real const bound=128*epsilon*std::max(step,1)*initial.norminf();
            if(require_identity) AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_error<=bound,
                "Stationary raw-marker transport is not the identity within accumulated roundoff");
            amrex::Print()<<std::setprecision(17)<<"IDENTITY {\"step\":"<<step
                <<",\"profile\":"<<profile<<",\"mode\":"<<mode
                <<",\"max_abs_eV\":"<<max_error/kelvin
                <<",\"particle_count\":"<<actual_count<<"}\n";
            AMREX_ALWAYS_ASSERT(te.is_finite());
        };
        snapshot(0);
        if(ownership_push!=0){
            vel[0]->setVal(ownership_push*dx[0]);
            vel[2]->setVal(ownership_push*dx[1]);
            hp.QDSMCInitializeKe(0);
            hp.m_qdsmc_pc->SetV(0,*vel[0],*vel[1],*vel[2]);
            hp.m_qdsmc_pc->SetK(0,ke,oldrho);
            hp.m_qdsmc_pc->PushX(0,3.,1.);
            snapshot(-1);  // ownership-only displacement; no temperature update
            hp.m_qdsmc_pc->ResetParticles(0);
            snapshot(-2);
        }
        for(int n=1;n<=iterations;++n){
            if(mode==0){
                hp.QdsmcTransportOnce(0,0.,true);
            } else {
                // Native seed/gather/deposit/recovery, optionally omitting
                // PushX(0) to separate the ownership inset from the pusher.
                hp.QDSMCInitializeKe(0);
                hp.m_qdsmc_pc->SetV(0,*vel[0],*vel[1],*vel[2]);
                hp.m_qdsmc_pc->SetK(0,ke,oldrho);
                if(n==1) markers("gather1");
                if(mode==2) hp.m_qdsmc_pc->PushX(0,0.,1.);
                hp.m_qdsmc_pc->DepositK(0,ke,hp.m_qdsmc_gradient_deposit);
                hp.m_qdsmc_pc->DepositField(0,weights,hp.m_qdsmc_gradient_deposit);
                hp.QDSMCUpdateTe(0);
            }
            hp.m_qdsmc_pc->ResetParticles(0);
            if(n==1||n==10||n==200||n==400||n==800) snapshot(n);
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
