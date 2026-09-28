/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeSplitAmpere.H"
#include "NativeEndpointField.H"
#include "NativeVacuumEndpoint.H"
#include "DarwinABoundary.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>

namespace warpx::darwin {
namespace {
using MF=amrex::MultiFab;
using View=ablastr::fields::VectorField;
View Pointers(std::array<MF,3>& a){return {&a[0],&a[1],&a[2]};}
void Sync(View const& fields, amrex::Periodicity const& period)
{
    amrex::Vector<MF*> vector{fields.begin(),fields.end()};
    amrex::FillBoundaryAndSync_nowait(vector,period);
    amrex::FillBoundaryAndSync_finish(vector);
}
}
void CalculateNativeSplitAmpere(WarpX& sim,View const& potential,
    View const& static_B,View const& current)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxLevel()==0 && !sim.DoPML() &&
        !WarpX::do_single_precision_comms && !EB::enabled() && WarpX::ncomps==1,
        "Split native Ampere requires one level, m0, DP communication, no EB/PML");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!WarpX::getCosts(0) ||
        WarpX::load_balance_costs_update_algo!=LoadBalanceCostsUpdateAlgo::Timers,
        "Split native Ampere does not update load-balancing timer tallies");
    std::array<MF,3> response,fixed,fixed_current;
    auto const native_B=sim.m_fields.get_alldirs(warpx::fields::FieldType::Bfield_fp,0);
    auto const native_C=sim.m_fields.get_alldirs(warpx::fields::FieldType::hybrid_current_fp_plasma,0);
    auto const native_A=sim.m_fields.get_alldirs("hybrid_A_fp",0);
    for(int c=0;c<3;++c){
        AMREX_ALWAYS_ASSERT(potential[c]->boxArray()==native_A[c]->boxArray() &&
            potential[c]->DistributionMap()==native_A[c]->DistributionMap() &&
            potential[c]->nGrowVect()==native_A[c]->nGrowVect() && potential[c]->nComp()==1);
        AMREX_ALWAYS_ASSERT(static_B[c]->boxArray()==native_B[c]->boxArray() &&
            static_B[c]->DistributionMap()==native_B[c]->DistributionMap() &&
            static_B[c]->nGrowVect()==native_B[c]->nGrowVect() && static_B[c]->nComp()==1);
        AMREX_ALWAYS_ASSERT(current[c]->boxArray()==native_C[c]->boxArray() &&
            current[c]->DistributionMap()==native_C[c]->DistributionMap() &&
            current[c]->nGrow()>=1 && current[c]->nComp()==1);
        for(int d=0;d<3;++d){
            AMREX_ALWAYS_ASSERT(current[c]!=potential[d] && current[c]!=static_B[d] &&
                (c==d || current[c]!=current[d]));
        }
        response[c].define(static_B[c]->boxArray(),static_B[c]->DistributionMap(),1,static_B[c]->nGrowVect());
        fixed[c].define(static_B[c]->boxArray(),static_B[c]->DistributionMap(),1,static_B[c]->nGrowVect());
        MF::Copy(fixed[c],*static_B[c],0,0,1,fixed[c].nGrowVect());
        fixed_current[c].define(current[c]->boxArray(),current[c]->DistributionMap(),1,1);
        fixed_current[c].setVal(0.);
    }
    auto response_view=Pointers(response);
    auto* solver=sim.get_pointer_fdtd_solver_fp(0);
    solver->ComputeCurlA(response_view,potential,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    // DP sync is the linear owner-copy map used by derived native B. Sync
    // both operands before its second curl, including box/physical corners.
    Sync(response_view,sim.Geom(0).periodicity());
    Sync(Pointers(fixed),sim.Geom(0).periodicity());
    auto fixed_view=Pointers(fixed_current);
    solver->CalculateCurrentAmpere(fixed_view,Pointers(fixed),sim.GetEBUpdateEFlag()[0],0);
    auto output=current;
    solver->CalculateCurrentAmpere(output,response_view,sim.GetEBUpdateEFlag()[0],0);
    for(int c=0;c<3;++c) MF::Add(*current[c],fixed_current[c],0,0,1,1);
    // Complete the same physical current image as the independent Cdot map.
    // The raw second curl lacks its reflected cap ghost on a PMC face.
    // Reuse the now-consumed fixed-current scratch to retain Ampere's grow1
    // write contract even when the caller provides wider output storage.
    amrex::GpuArray<int,AMREX_SPACEDIM> lower{},upper{};
    bool any_pmc=false;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        lower[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        upper[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
        any_pmc=any_pmc||lower[d]||upper[d];
    }
    if(any_pmc) {
        for(int c=0;c<3;++c) {
            MF::Copy(fixed_current[c],*current[c],0,0,1,1);
            fixed_current[c].OverrideSync(sim.Geom(0).periodicity());
            fixed_current[c].FillBoundary(sim.Geom(0).periodicity());
            ApplyDarwinPMCVectorBoundary(fixed_current[c],sim.Geom(0),lower,upper);
            MF::Copy(*current[c],fixed_current[c],0,0,1,1);
        }
    }
}

bool TryCalculateNativeSplitDarwinAmpere(WarpX& sim)
{
    bool enabled=false, vacuum_split=false;
    amrex::ParmParse endpoint("endpoint_diagnostic");
    endpoint.query("split_ampere",enabled);
    endpoint.query("vacuum_split_ampere",vacuum_split);
    auto const& model=*sim.get_pointer_HybridPICModel();
    if((!enabled && !vacuum_split) || (!model.HasResistivity() && !vacuum_split)) return false;
    if(vacuum_split) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NativeVacuumEndpointEnabled() &&
            model.m_darwin_vacuum_recovery && !NativePrescribedDriveEnabled(),
            "Vacuum split Ampere requires the guarded static joined-vacuum path");
        ValidateNativeVacuumEndpointCapabilities(sim);
    }
    bool smooth=false;amrex::ParmParse("endpoint_diagnostic").query("smooth_force",smooth);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE((smooth || (vacuum_split && !model.HasResistivity())) && NativeEndpointEnabled() &&
        NativeCorrelatedIncrementEnabled() && model.UseCompatibleYeeInertia() &&
        model.m_darwin && !model.m_has_external_current && (!model.m_darwin_vacuum_recovery || vacuum_split) &&
        !model.m_pec_conductor_wall_rows && !model.m_has_per_species_eta,
        "Split native Ampere requires the guarded smooth-force compatible-Yee Darwin prototype");
#if defined(WARPX_DIM_RZ)
    bool const axial_pmc = WarpX::field_boundary_lo[1] == FieldBoundaryType::PMC &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::PMC;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.Geom(0).ProbLo(0)==0. &&
        (sim.Geom(0).isPeriodic(1) || axial_pmc) &&
        WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC,
        "Split native Ampere requires RZ axis/PEC radial and periodic z or axial PMC");
#else
    amrex::Abort("Split native Ampere prototype is currently qualified only in RZ");
#endif
    CalculateNativeSplitAmpere(sim,sim.m_fields.get_alldirs("hybrid_A_fp",0),
        sim.m_fields.get_alldirs("hybrid_B_static_fp",0),
        sim.m_fields.get_alldirs(warpx::fields::FieldType::hybrid_current_fp_plasma,0));
    return true;
}
}
