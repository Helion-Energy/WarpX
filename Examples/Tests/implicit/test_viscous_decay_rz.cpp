/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

namespace {
using namespace amrex::literals;
using Field = ablastr::fields::VectorField;
using warpx::fields::FieldType;

amrex::Real magnetic_energy (Field const& fields, amrex::Geometry const& geom)
{
    amrex::Real result = 0.0_rt;
    for (int c = 0; c < 3; ++c) {
        auto const volume = MakeQdsmcVolumeElement(geom, fields[c]->ixType());
        amrex::MultiFab density(fields[c]->boxArray(), fields[c]->DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
            auto const b = fields[c]->const_array(mfi);
            auto const out = density.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                out(i,j,k) = volume(i,j,k) * b(i,j,k) * b(i,j,k) / (2.0_rt*PhysConst::mu0);
            });
        }
        result += density.sum_unique(0, false, geom.periodicity());
    }
    return result;
}
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        int steps = 20;
        amrex::Real tolerance = 2.e-5_rt;
        amrex::ParmParse options("viscous_decay");
        options.query("steps", steps);
        options.query("relative_tolerance", tolerance);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& hp = *w.get_pointer_HybridPICModel();
        auto const& geom = w.Geom(0);
        auto const lo = geom.ProbLoArray();
        auto const dx = geom.CellSizeArray();
        auto const rwall = geom.ProbHi(0);
        auto const zlength = geom.ProbHi(1)-geom.ProbLo(1);
        auto const magnetic = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        // Bt is regular at the axis, zero on both PMC ends, and has zero
        // radial derivative at the PEC wall. Bz has zero radial derivative
        // at both the axis and wall. Its radial mode supplies Jtheta; Bt
        // supplies Jr and Jz, exercising every cylindrical work component.
        for (int c = 0; c < 3; ++c) {
            auto const nodal = magnetic[c]->ixType().toIntVect();
            for (amrex::MFIter mfi(*magnetic[c]); mfi.isValid(); ++mfi) {
                auto const b = magnetic[c]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    auto const r=lo[0]+(i+(nodal[0]?0.0_rt:0.5_rt))*dx[0];
                    auto const z=lo[1]+(j+(nodal[1]?0.0_rt:0.5_rt))*dx[1];
                    auto const rr=r/rwall;
                    if (c==0) { b(i,j,k)=0.0_rt; }
                    else if (c==1) {
                        b(i,j,k)=0.005_rt*rr*(3.0_rt-rr*rr)*
                            std::sin(2.0_rt*MathConst::pi*(z-lo[1])/zlength);
                    } else {
                        b(i,j,k)=0.1_rt+0.005_rt*std::cos(MathConst::pi*rr);
                    }
                });
            }
            magnetic[c]->FillBoundary(geom.periodicity());
        }
        w.ApplyBfieldBoundary(0,PatchType::fine,SubcyclingHalf::None,0.0_rt);
        auto const before=magnetic_energy(magnetic,geom);
        auto const heat_before=hp.m_ebud_visc_bulk+hp.m_ebud_visc_band;
        auto const work_before=hp.m_ebud_visc_work_bulk+hp.m_ebud_visc_work_band;
        w.Evolve(steps);
        auto const loss=before-magnetic_energy(magnetic,geom);
        auto const heat=hp.m_ebud_visc_bulk+hp.m_ebud_visc_band-heat_before;
        auto const work=hp.m_ebud_visc_work_bulk+hp.m_ebud_visc_work_band-work_before;
        auto const defect=std::abs(loss-heat)/std::max(std::abs(loss),1.e-30_rt);
        amrex::Print()<<std::setprecision(17)<<"VISCOUS_DECAY steps="<<steps
            <<" field_loss_J="<<loss<<" booked_heat_J="<<heat
            <<" booked_work_J="<<work<<" relative_field_heat_error="<<defect<<'\n';
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(loss>0.0_rt,"viscous magnetic energy must decay");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(defect)&&defect<tolerance,
                                         "closed RZ viscous field/heat balance");
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
