/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Independent extensive-work identity through the real deposition postprocessing
// (axis volume/fold, filter, overlap sum, wall fold) and the particle E gather.
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_Reduce.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

using namespace amrex::literals;
using warpx::fields::FieldType;
int main (int argc,char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& w=WarpX::GetInstance();
        w.InitData();
        auto const& geom=w.Geom(0);
        auto const dx=geom.CellSizeArray();
        auto const radius=geom.ProbHi(0),length=geom.ProbLength(1);
        auto const J=w.m_fields.get_alldirs(FieldType::current_fp,0);
        auto const E=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
        auto const gather=w.m_fields.get_alldirs(FieldType::Efield_aux,0);
        std::array<std::unique_ptr<amrex::MultiFab>,3> raw;
        for (int c=0;c<3;++c) {
            raw[c]=std::make_unique<amrex::MultiFab>(J[c]->boxArray(),J[c]->DistributionMap(),1,J[c]->nGrowVect());
            raw[c]->setVal(0.0_rt);
            auto const it=J[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*raw[c]);mfi.isValid();++mfi) {
                auto const deposit=raw[c]->array(mfi);
                auto const e=E[c]->array(mfi);
                int const box=mfi.index();
                auto const region=amrex::grow(mfi.validbox(),w.get_ng_depos_J());
                amrex::ParallelFor(region,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    // Each box owns distinct deposits, including deposits into
                    // its neighbors' guards and across the physical boundaries.
                    deposit(i,j,k)=std::sin(0.17_rt*(i+3*j+7*box+c+1)) +
                                  0.3_rt*std::cos(0.43_rt*(2*i-j+box+4*c));
                });
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const r=(i+(it[0]?0.0_rt:0.5_rt))*dx[0]/radius;
                    auto const z=(j+(it[1]?0.0_rt:0.5_rt))*dx[1]/length;
                    e(i,j,k)=(0.4_rt+std::sin(3.0_rt*r+c)) *
                        (1.0_rt+0.2_rt*std::cos(2.0_rt*MathConst::pi*z));
                });
            }
            amrex::MultiFab::Copy(*J[c],*raw[c],0,0,1,J[c]->nGrowVect());
            E[c]->FillBoundary(geom.periodicity());
        }
        w.ApplyInverseVolumeScalingToCurrentDensity(J[0],J[1],J[2],0);
        w.SyncCurrent("current_fp");
        w.ApplyJfieldBoundary(0,J[0],J[1],J[2],PatchType::fine);
        amrex::Real physical=0.0_rt,particle=0.0_rt,absolute=0.0_rt;
        for (int c=0;c<3;++c) {
            w.ApplyRZAdjointGather(*gather[c],*E[c],0,c);
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<amrex::Real> data(op);
            for (amrex::MFIter mfi(*raw[c]);mfi.isValid();++mfi) {
                auto const dep=raw[c]->const_array(mfi), e=gather[c]->const_array(mfi);
                op.eval(amrex::grow(mfi.validbox(),w.get_ng_depos_J()),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)
                    -> amrex::GpuTuple<amrex::Real> { return {dx[0]*dx[1]*dep(i,j,k)*e(i,j,k)}; });
            }
            amrex::Real local=amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealSum(local);
            particle+=local;
            auto const volume=MakeQdsmcVolumeElement(geom,J[c]->ixType());
            amrex::MultiFab measured(J[c]->boxArray(),J[c]->DistributionMap(),2,0);
            for (amrex::MFIter mfi(measured);mfi.isValid();++mfi) {
                auto const j=J[c]->const_array(mfi),e=E[c]->const_array(mfi);
                auto const out=measured.array(mfi);
                amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j0,int k) {
                    out(i,j0,k,0)=volume(i,j0,k)*j(i,j0,k)*e(i,j0,k);
                    out(i,j0,k,1)=std::abs(out(i,j0,k,0));
                });
            }
            physical+=measured.sum_unique(0,false,geom.periodicity());
            absolute+=measured.sum_unique(1,false,geom.periodicity());
        }
        auto const relative=std::abs(physical-particle)/absolute;
        amrex::Print()<<std::setprecision(17)<<"FILTER_WORK {\"physical\":"<<physical
            <<",\"particle\":"<<particle<<",\"absolute\":"<<absolute
            <<",\"relative\":"<<relative<<"}\n";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(relative) && relative<2.e-13_rt,
            "Current filtering and particle gather must preserve the physical work functional");
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
