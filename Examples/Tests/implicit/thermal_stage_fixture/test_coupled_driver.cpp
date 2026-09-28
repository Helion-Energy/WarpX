/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
// Manufactured E/U systems exercise the shared Newton driver on native Yee and
// cell layouts. They are solver-contract tests, not a Darwin physics result.
#include "WarpX.H"
#include "FieldSolver/ImplicitSolvers/ThermalStageSolver.H"
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuLaunch.H>
#include <cmath>
#include <limits>
#include <string>

using namespace warpx::thermal;
using Real = amrex::Real;
using Vec = WarpXSolverVec;
using warpx::fields::FieldType;
constexpr Real e_scale = 1.e5;
constexpr Real u_scale = 1.e-3;
constexpr const char* energy_name = "driver_energy";

struct Manufactured final : ThermalNonlinearOperator {
    WarpX sim;
    Vec state, base, visible;
    amrex::MultiFab ughost;
    std::string fail;
    bool weak = false;
    int restores = 0, probe_calls = 0, nonlinear_calls = 0, freeze_calls = 0;
    int near_root_calls = 0;
    bool mixed_probe = false;

    Manufactured (int max_box, std::string failure, bool weak_block)
        : fail(std::move(failure)), weak(weak_block)
    {
        amrex::Box dom(amrex::IntVect(0,0), amrex::IntVect(7,15));
        amrex::RealBox rb({0.,0.},{1.,1.});
        int periodic[2] = {0,0};
        sim.geometry = amrex::Geometry(dom, &rb, 1, periodic);
        amrex::BoxArray ba(dom); ba.maxSize(max_box);
        amrex::DistributionMapping dm(ba);
        amrex::IntVect stagger[3] = {{0,1},{1,1},{1,0}};
        for (int d = 0; d < 3; ++d) {
            sim.m_fields.alloc_init(FieldType::Efield_fp, ablastr::fields::Direction{d},
                                    0, amrex::convert(ba,stagger[d]), dm, 1,
                                    amrex::IntVect(1), 0.);
        }
        sim.m_fields.alloc_init(energy_name, 0, ba, dm, 1, amrex::IntVect(1), 0.);
        state.Define(&sim, "Efield_fp", "none", {{energy_name,u_scale}}, e_scale);
        state.zero();
        state.getMultiFabBlock(energy_name,0).setVal(u_scale);
        base.Define(state); visible.Copy(state);
        ughost.define(ba,dm,1,1);
    }

    void FillU (const Vec& x)
    {
        amrex::MultiFab::Copy(ughost,x.getMultiFabBlock(energy_name,0),0,0,1,0);
        ughost.FillBoundary(sim.geometry.periodicity());
    }

    bool Residual (Vec& out, const Vec& x, int, bool probe) override
    {
        visible.Copy(x); // stands in for caller-owned trial scratch
        if (probe) {
            ++probe_calls;
            Vec diff; diff.Copy(x); diff.increment(base,-1);
            auto const norms = diff.blockNorms();
            mixed_probe = mixed_probe || (norms[0]>0 && norms[1]>0);
        } else { ++nonlinear_calls; }
        if ((fail=="initial" && nonlinear_calls==1) ||
            (fail=="probe" && probe)) { return false; }
        FillU(x);
        auto const dom = sim.geometry.Domain();
        Real const coupling = weak ? 0. : 0.1;
        Real const thermal_weight = weak ? 1.e-8 : 1.;
        for (int d = 0; d < 3; ++d) {
            for (amrex::MFIter mfi(*out.getArrayVec()[0][d]); mfi.isValid(); ++mfi) {
                auto const e=x.getArrayVec()[0][d]->const_array(mfi);
                auto const u=ughost.const_array(mfi);
                auto const f=out.getArrayVec()[0][d]->array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    int const il = d==0 ? i : amrex::max(0,i-1);
                    int const ih = amrex::min(i,dom.bigEnd(0));
                    int const jl = d==2 ? j : amrex::max(0,j-1);
                    int const jh = amrex::min(j,dom.bigEnd(1));
                    Real const uv=.25*(u(il,jl,k)+u(ih,jl,k)+u(il,jh,k)+u(ih,jh,k))/u_scale;
                    f(i,j,k)=e_scale*(e(i,j,k)/e_scale-(d+1)+coupling*(uv*uv-4));
                });
            }
        }
        auto& fU=out.getMultiFabBlock(energy_name,0);
        for (amrex::MFIter mfi(fU);mfi.isValid();++mfi) {
            auto const u=x.getMultiFabBlock(energy_name,0).const_array(mfi);
            auto const er=x.getArrayVec()[0][0]->const_array(mfi);
            auto const et=x.getArrayVec()[0][1]->const_array(mfi);
            auto const ez=x.getArrayVec()[0][2]->const_array(mfi);
            auto const f=fU.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real const ec=(.5*(er(i,j,k)+er(i,j+1,k))+
                               .25*(et(i,j,k)+et(i+1,j,k)+et(i,j+1,k)+et(i+1,j+1,k))+
                               .5*(ez(i,j,k)+ez(i+1,j,k)))/e_scale;
                Real const uv=u(i,j,k)/u_scale;
                f(i,j,k)=u_scale*(thermal_weight*(uv*uv-4)+coupling*(ec-6));
            });
        }
        if (fail=="nonfinite") { fU.setVal(std::numeric_limits<Real>::quiet_NaN()); }
        if (fail=="final" && !probe && out.norm2()<1.e-8) {
            if (++near_root_calls==2) { return false; }
        }
        return x.getMultiFabBlock(energy_name,0).min(0)>0;
    }

    Real StepBound (const Vec&,const Vec&) const override { return 1.; }
    bool Freeze (const Vec& x,int,bool use_pc,ThermalSolveResult& stats) override
    {
        ++freeze_calls; base.Copy(x);
        if (use_pc) { ++stats.pc_updates; }
        return fail!="freeze";
    }
    bool Precondition (Vec& out,const Vec& rhs) override
    {
        if (fail=="pc") { return false; }
        out.Copy(rhs);
        auto& z=out.getMultiFabBlock(energy_name,0);
        Real const w=weak ? 1.e-8 : 1.;
        for (amrex::MFIter mfi(z);mfi.isValid();++mfi) {
            auto const a=z.array(mfi);
            auto const u=base.getMultiFabBlock(energy_name,0).const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                a(i,j,k)/=2*w*u(i,j,k)/u_scale;
            });
        }
        return true;
    }
    void RestoreInput (const Vec& input) override { ++restores; visible.Copy(input); }
};

int main (int argc,char** argv)
{
    amrex::Initialize(argc,argv);
    {
        amrex::ParmParse pp("test");
        int max_box=4; std::string which="coupled";
        pp.query("max_grid_size",max_box); pp.query("case",which);
        bool const success=which=="coupled" || which=="pc_off" || which=="reference";
        Manufactured op(max_box,success ? "" : which,which=="weak");
        Vec input; input.Copy(op.state);
        ThermalSolveOptions opts;
        opts.max_newton_iterations=15;
        opts.absolute_tolerance=1.e-15;
        opts.relative_tolerance=1.e-10;
        opts.linear_relative_tolerance=1.e-10;
        // E starts at zero while its residual is large. Preserve the same
        // RHS-based probe scale used by the maintained Darwin field solver.
        opts.probe_rhs_scale_floor=true;
        opts.use_preconditioner=which!="pc_off";
        pp.query("linear_verbose",opts.linear_verbosity);
        pp.query("adaptive_forcing",opts.adaptive_forcing);
        if (which=="limit") { opts.max_newton_iterations=0; }
        if (which=="weak") {
            // This is an acceptance adversary, deliberately stopped after
            // one Newton step: the aggregate norm passes, the small U block
            // does not. A scalar aggregate gate would silently accept it.
            opts.max_newton_iterations=1;
            opts.relative_tolerance=1.e-6;
            opts.block_relative_tolerances={1.e-6,1.e-10};
        }
        if (which=="reference") {
            // A previous outer sweep has already reduced both equations below
            // the fixed physical-step target. Restarting relative references
            // here would demand extra work down toward roundoff.
            opts.block_reference_norms={1.e14,1.e14};
        }
        auto result=SolveThermalSystem(op,op.state,opts);
        amrex::Print()<<"DRIVER_RESULT case="<<which<<" status="<<int(result.status)
                      <<" Newton="<<result.newton_iterations<<" GMRES="<<result.linear_iterations
                      <<" residual="<<result.residual<<" linear_status="<<result.linear_status
                      <<" linear_residual="<<result.linear_residual
                      <<" linear_solves="<<result.linear_solves
                      <<" linear_rtol_min="<<result.linear_tolerance_min
                      <<" linear_rtol_max="<<result.linear_tolerance_max<<"\n";
        AMREX_ALWAYS_ASSERT((result.status==ThermalSolveStatus::Converged)==success);
        AMREX_ALWAYS_ASSERT(result.linear_solves==int(result.linear_relative_tolerances.size()));
        if (result.linear_solves>0) {
            if (opts.adaptive_forcing) {
                AMREX_ALWAYS_ASSERT(result.linear_relative_tolerances.front()==opts.forcing_max);
                AMREX_ALWAYS_ASSERT(result.linear_tolerance_min>0 &&
                                    result.linear_tolerance_max<=opts.forcing_max);
                if (success) { AMREX_ALWAYS_ASSERT(result.linear_tolerance_min<opts.forcing_max); }
            } else {
                AMREX_ALWAYS_ASSERT(result.linear_tolerance_min==opts.linear_relative_tolerance &&
                                    result.linear_tolerance_max==opts.linear_relative_tolerance);
            }
        }
        if (which=="reference") {
            AMREX_ALWAYS_ASSERT(op.restores==0 && op.probe_calls==0 &&
                                op.nonlinear_calls==2 && result.newton_iterations==0);
            Vec diff; diff.Copy(op.state); diff.increment(input,-1);
            AMREX_ALWAYS_ASSERT(diff.norm2()==0);
            for (int b=0;b<2;++b) {
                AMREX_ALWAYS_ASSERT(result.block_targets[b]==1.e4 &&
                                    result.block_residuals[b]<=result.block_targets[b]);
            }
        } else if (success) {
            AMREX_ALWAYS_ASSERT(op.restores==0 && op.probe_calls>0 && op.mixed_probe);
            for (int d=0;d<3;++d) {
                amrex::MultiFab check(op.state.getArrayVec()[0][d]->boxArray(),
                                     op.state.getArrayVec()[0][d]->DistributionMap(),1,0);
                amrex::MultiFab::Copy(check,*op.state.getArrayVec()[0][d],0,0,1,0);
                check.plus(-e_scale*(d+1),0,1,0);
                AMREX_ALWAYS_ASSERT(check.norm0(0)/e_scale<2.e-9);
            }
            auto const& u=op.state.getMultiFabBlock(energy_name,0);
            AMREX_ALWAYS_ASSERT(std::abs(u.min(0)/u_scale-2)<2.e-9);
            AMREX_ALWAYS_ASSERT(std::abs(u.max(0)/u_scale-2)<2.e-9);
            AMREX_ALWAYS_ASSERT(result.block_residuals.size()==2);
            for (int b=0;b<2;++b) {
                AMREX_ALWAYS_ASSERT(result.block_residuals[b]<=
                    std::max(opts.absolute_tolerance,
                             opts.relative_tolerance*result.initial_block_residuals[b]));
            }
            Vec visible; visible.Copy(op.visible); visible.increment(op.state,-1);
            AMREX_ALWAYS_ASSERT(visible.norm2()==0);
        } else {
            if (which=="weak") {
                AMREX_ALWAYS_ASSERT(result.status==ThermalSolveStatus::IterationLimit);
                AMREX_ALWAYS_ASSERT(result.newton_iterations==1);
                AMREX_ALWAYS_ASSERT(result.residual<opts.relative_tolerance*result.initial_residual);
                AMREX_ALWAYS_ASSERT(result.block_residuals[1]>std::max(opts.absolute_tolerance,
                                    1.e-10*result.initial_block_residuals[1]));
            }
            AMREX_ALWAYS_ASSERT(op.restores==1);
            Vec diff;diff.Copy(op.state);diff.increment(input,-1);
            AMREX_ALWAYS_ASSERT(diff.norm2()==0);
            diff.Copy(op.visible);diff.increment(input,-1);
            AMREX_ALWAYS_ASSERT(diff.norm2()==0);
        }
        amrex::Print()<<"DRIVER_PASS case="<<which<<" boxes="<<max_box
                      <<" Newton="<<result.newton_iterations<<" GMRES="<<result.linear_iterations
                      <<" restores="<<op.restores<<"\n";
    }
    amrex::Finalize();
}
