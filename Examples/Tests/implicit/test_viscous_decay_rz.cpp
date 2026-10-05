/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <ablastr/coarsen/sample.H>
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

// With fixed ions and density the inertia operator stores electron-current
// kinetic energy. Use its exact edge density/floor and the Yee dual volumes.
amrex::Real electron_current_energy (WarpX& w, HybridPICModel& hp, Field const& magnetic)
{
    hp.CalculatePlasmaCurrent(magnetic,w.GetEBUpdateEFlag()[0],0);
    auto const current=w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma,0);
    auto const& rho=*w.m_fields.get(FieldType::rho_fp,0);
    amrex::MultiFab effective(rho.boxArray(),rho.DistributionMap(),1,rho.nGrowVect());
    amrex::MultiFab::Copy(effective,rho,0,0,1,rho.nGrowVect());
    if (auto const* ped=hp.DensityPedestal(0)) {
        amrex::MultiFab::Add(effective,*ped,0,0,1,rho.nGrowVect());
    }
    auto const& geom=w.Geom(0);
    auto const floor=hp.m_n_floor*PhysConst::q_e;
    amrex::GpuArray<int,3> const nodal={1,1,1}, coarsen={1,1,1};
    amrex::Real result=0.0_rt;
    for (int c=0;c<3;++c) {
        auto const volume=MakeQdsmcVolumeElement(geom,current[c]->ixType());
        auto const it=current[c]->ixType().toIntVect();
        amrex::GpuArray<int,3> const stagger={it[0],it[1],1};
        amrex::MultiFab density(current[c]->boxArray(),current[c]->DistributionMap(),1,0);
        for (amrex::MFIter mfi(density);mfi.isValid();++mfi) {
            auto const jfield=current[c]->const_array(mfi);
            auto const reff=effective.const_array(mfi);
            auto const out=density.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto const rho_edge=amrex::max(floor,ablastr::coarsen::sample::Interp(
                    reff,nodal,stagger,coarsen,i,j,k,0));
                out(i,j,k)=volume(i,j,k)*PhysConst::m_e*jfield(i,j,k)*jfield(i,j,k)
                    /(2.0_rt*PhysConst::q_e*rho_edge);
            });
        }
        result+=density.sum_unique(0,false,geom.periodicity());
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
        bool inertia_energy=false;
        options.query("include_inertia_energy",inertia_energy);
        bool closed_electron_flow=false;
        options.query("closed_electron_flow",closed_electron_flow);
        auto& w = WarpX::GetInstance();
        w.InitData();
        // The explicit hybrid deposits rho on its first step. Populate that
        // fixed-ion density before measuring the density-dependent current
        // energy; the initial uniform B has no drag during this bootstrap.
        if (inertia_energy) { w.Evolve(1); }
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
                        amrex::Real const profile = closed_electron_flow
                            ? rr*std::pow(1.0_rt-rr*rr,3) : rr*(3.0_rt-rr*rr);
                        auto const wave=std::sin(2.0_rt*MathConst::pi*(z-lo[1])/zlength);
                        // The convection certificate isolates closed flow:
                        // both poloidal velocity components vanish at the
                        // endcaps as well as at the radial conductor.
                        b(i,j,k)=0.005_rt*profile*(closed_electron_flow ? wave*wave*wave : wave);
                    } else {
                        b(i,j,k)=0.1_rt+0.005_rt*(closed_electron_flow
                            ? std::pow(1.0_rt-rr*rr,4) : std::cos(MathConst::pi*rr));
                    }
                });
            }
            magnetic[c]->FillBoundary(geom.periodicity());
        }
        w.ApplyBfieldBoundary(0,PatchType::fine,SubcyclingHalf::None,0.0_rt);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(inertia_energy==hp.m_include_electron_inertia_elliptic,
            "The decay energy must include current kinetic energy when inertia is on");
        auto const current_before=inertia_energy?electron_current_energy(w,hp,magnetic):0.0_rt;
        auto const before=magnetic_energy(magnetic,geom)+current_before;
        auto const heat_before=hp.m_ebud_visc_bulk+hp.m_ebud_visc_band;
        auto const work_before=hp.m_ebud_visc_work_bulk+hp.m_ebud_visc_work_band;
        w.Evolve(steps);
        auto const current_after=inertia_energy?electron_current_energy(w,hp,magnetic):0.0_rt;
        auto const loss=before-magnetic_energy(magnetic,geom)-current_after;
        auto const heat=hp.m_ebud_visc_bulk+hp.m_ebud_visc_band-heat_before;
        auto const work=hp.m_ebud_visc_work_bulk+hp.m_ebud_visc_work_band-work_before;
        auto const defect=std::abs(loss-heat)/std::max(std::abs(loss),1.e-30_rt);
        amrex::Print()<<std::setprecision(17)<<"VISCOUS_DECAY steps="<<steps
            <<" inertia="<<inertia_energy<<" current_energy_loss_J="<<current_before-current_after
            <<" field_loss_J="<<loss<<" booked_heat_J="<<heat
            <<" booked_work_J="<<work<<" relative_field_heat_error="<<defect<<'\n';
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(loss>0.0_rt,"viscous magnetic energy must decay");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(defect)&&defect<tolerance,
                                         "closed RZ viscous field/heat balance");
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
