/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Test the applied native Ohm operator, with no reconstructed force formula.
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

using namespace amrex::literals;
using warpx::fields::FieldType;

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray(), lo = geom.ProbLoArray();
        auto const radius = geom.ProbHi(0), length = geom.ProbHi(1)-geom.ProbLo(1);
        auto& rho = *w.m_fields.get(FieldType::rho_fp,0);
        auto* ped = const_cast<amrex::MultiFab*>(hp.DensityPedestal(0));
        auto J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp,0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp,0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
        amrex::MultiFab pressure(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
        pressure.setVal(0.0_rt);
        hp.m_include_electron_pressure_term = false;
        hp.m_add_external_fields = false;
        hp.m_include_hall_term = true;
        amrex::Real largest_legacy = 0.0_rt;
        amrex::Print() << std::setprecision(17);
        for (amrex::Real phase : {0.0_rt,0.25_rt,0.5_rt,0.75_rt}) {
            for (amrex::MFIter mfi(rho);mfi.isValid();++mfi) {
                auto const out = rho.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),rho.nComp(),
                    [=] AMREX_GPU_DEVICE(int i,int j,int k,int n) {
                        auto const r = std::abs(lo[0]+i*dx[0]);
                        auto const z = (j*dx[1])/length;
                        out(i,j,k,n) = PhysConst::q_e *
                            (1.e16_rt + (r < 0.517_rt*radius+phase*dx[0] ?
                             3.e19_rt*(1.0_rt+0.2_rt*std::cos(2.0_rt*MathConst::pi*z)) : 0.0_rt));
                    });
            }
            if (ped) { ped->setVal(5.e17_rt*PhysConst::q_e); }
            for (int c=0;c<3;++c) {
                auto const it=Ji[c]->ixType().toIntVect(), bt=B[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(*Ji[c]);mfi.isValid();++mfi) {
                    auto const ji=Ji[c]->array(mfi), total=J[c]->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        auto const r=(lo[0]+(i+(it[0]?0.0_rt:0.5_rt))*dx[0])/radius;
                        auto const z=(j+(it[1]?0.0_rt:0.5_rt))*dx[1]/length;
                        auto const q=1.e5_rt*(c==0 ? std::sin(2.0_rt*MathConst::pi*r)*(1.0_rt+0.3_rt*std::cos(2.0_rt*MathConst::pi*z)) :
                            c==1 ? r*(1.0_rt+0.2_rt*std::sin(2.0_rt*MathConst::pi*z)) :
                            (0.4_rt+r*r)*(1.0_rt+0.1_rt*std::cos(2.0_rt*MathConst::pi*z)));
                        total(i,j,k)=0.3_rt*q;
                        ji(i,j,k)=1.3_rt*q;
                    });
                }
                for (amrex::MFIter mfi(*B[c]);mfi.isValid();++mfi) {
                    auto const b=B[c]->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        auto const r=(lo[0]+(i+(bt[0]?0.0_rt:0.5_rt))*dx[0])/radius;
                        auto const z=(j+(bt[1]?0.0_rt:0.5_rt))*dx[1]/length;
                        b(i,j,k)=c==0 ? 0.1_rt*r*std::sin(2.0_rt*MathConst::pi*z) :
                            c==1 ? 0.2_rt*r : 0.3_rt*(1.0_rt+0.2_rt*r*r);
                    });
                }
            }
            for (bool compatible : {false,true}) {
                hp.m_energy_conserving_motional=compatible;
                for (int c=0;c<3;++c) { E[c]->setVal(0.0_rt); }
                w.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(
                    E,J,Ji,B,rho,pressure,w.GetEBUpdateEFlag()[0],0,&hp,false,false);
                amrex::Real work=0.0_rt, scale=0.0_rt;
                for (int c=0;c<3;++c) {
                    amrex::MultiFab measured(E[c]->boxArray(),E[c]->DistributionMap(),2,0);
                    auto const vol=MakeQdsmcVolumeElement(geom,E[c]->ixType());
                    for (amrex::MFIter mfi(measured);mfi.isValid();++mfi) {
                        auto const out=measured.array(mfi);
                        auto const e=E[c]->const_array(mfi);
                        auto const ji=Ji[c]->const_array(mfi), total=J[c]->const_array(mfi);
                        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                            out(i,j,k,0)=vol(i,j,k)*(ji(i,j,k)-total(i,j,k))*e(i,j,k);
                            out(i,j,k,1)=std::abs(out(i,j,k,0));
                        });
                    }
                    work+=measured.sum_unique(0,false,geom.periodicity());
                    scale+=measured.sum_unique(1,false,geom.periodicity());
                }
                auto const relative=std::abs(work)/scale;
                amrex::Print()<<"MOTIONAL_WORK {\"phase\":"<<phase<<",\"compatible\":"<<compatible
                    <<",\"work\":"<<work<<",\"absolute_work\":"<<scale
                    <<",\"relative\":"<<relative<<"}\n";
                AMREX_ALWAYS_ASSERT(std::isfinite(relative));
                if (compatible) { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(relative<2.e-13_rt,
                    "Native motional force creates electron energy"); }
                else { largest_legacy=amrex::max(largest_legacy,relative); }
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(largest_legacy>1.e-5_rt,
            "Negative control must expose the legacy Hall work defect");
    }
    warpx::initialization::finalize_external_libraries();
}
