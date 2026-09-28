/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ThermalStageSolver.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <iomanip>
#include <cmath>
#include <limits>
using namespace warpx::thermal;
using Vec=WarpXSolverVec;
using Real=amrex::Real;
constexpr char const* energy="hybrid_electron_energy_fp";
struct Manufactured final : ThermalNonlinearOperator {
    std::string final_failure;
    int near_root=0,restores=0,physical_calls=0;
    Vec visible;
    Real target=0.;
    bool Residual(Vec& out,Vec const& x,int,bool probe) override {
        visible.Copy(x);out.Copy(x);
        for(auto& level:out.getArrayVec())for(auto* field:level)
            for(amrex::MFIter mfi(*field);mfi.isValid();++mfi){auto a=field->array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=(a(i,j,k)-1.e-4)+1.005e-18;});}
        auto& U=out.getMultiFabBlock(energy,0);
        for(amrex::MFIter mfi(U);mfi.isValid();++mfi){auto a=U.array(mfi);amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=(a(i,j,k)-8.)+3.e-16;});}
        if(!probe){++physical_calls;if(target>0. && out.blockNorms()[0]<=target){++near_root;if(near_root==2){if(final_failure=="invalid")return false;if(final_failure=="thermal")U.setVal(1.);if(final_failure=="field")for(auto& level:out.getArrayVec())for(auto* field:level)field->setVal(1.);if(final_failure=="nonfinite")U.setVal(std::numeric_limits<Real>::quiet_NaN());}}}
        return true;
    }
    Real StepBound(Vec const&,Vec const&)const override{return 1.;}
    bool Freeze(Vec const&,int,bool,ThermalSolveResult&)override{return true;}
    bool Precondition(Vec& out,Vec const& rhs)override{out.Copy(rhs);return true;}
    void RestoreInput(Vec const& x)override{++restores;visible.Copy(x);}
};
int main(int argc,char** argv){
 warpx::initialization::initialize_external_libraries(argc,argv);
 {
  auto& w=WarpX::GetInstance();w.InitData();
  Vec x;x.Define(&w,"Efield_fp","none",{{energy,1.}});
  for(auto& level:x.getArrayVec())for(auto* f:level)f->setVal(1.e-4);
  x.getMultiFabBlock(energy,0).setVal(8.);
  Vec initial;initial.Copy(x);
  Manufactured op;amrex::ParmParse("test").query("final_failure",op.final_failure);
  Vec r;r.Define(x);AMREX_ALWAYS_ASSERT(op.Residual(r,x,0,false));
  ThermalSolveOptions o;o.relative_tolerance=0.;o.absolute_tolerance=1.e-12;
  o.block_absolute_tolerances={.99*r.blockNorms()[0],1.e-12};o.block_relative_tolerances={0.,0.};
  o.max_newton_iterations=3;o.max_backtracks=4;o.linear_relative_tolerance=1.e-8;o.use_preconditioner=false;
  op.target=o.block_absolute_tolerances[0];
  auto result=SolveThermalSystem(op,x,o);
  bool enabled=true;amrex::ParmParse("endpoint_diagnostic").query("accept_converged_trial",enabled);
  bool const expected=enabled && op.final_failure.empty();
  amrex::Print()<<std::setprecision(17)<<"BLOCK_ACCEPT_RESULT enabled="<<enabled<<" final_failure="<<op.final_failure<<" status="<<int(result.status)<<" Newton="<<result.newton_iterations<<" rejected="<<result.rejected_trials<<" roots="<<op.near_root<<" restores="<<op.restores<<" physical_calls="<<op.physical_calls<<" residual_calls="<<result.residual_evaluations<<" final_attempted="<<result.final_evaluation_attempted<<" final_valid="<<result.final_evaluation_valid<<" final_finite="<<result.final_residual_finite<<"\n";
  AMREX_ALWAYS_ASSERT((result.status==ThermalSolveStatus::Converged)==expected);
  if(expected){AMREX_ALWAYS_ASSERT(op.restores==0 && op.near_root>=2);for(int b=0;b<2;++b)AMREX_ALWAYS_ASSERT(result.block_residuals[b]<=result.block_targets[b]);}
  else{AMREX_ALWAYS_ASSERT(op.restores==1);Vec difference;difference.Copy(x);difference.increment(initial,-1.);AMREX_ALWAYS_ASSERT(difference.norm2()==0.);difference.Copy(op.visible);difference.increment(initial,-1.);AMREX_ALWAYS_ASSERT(difference.norm2()==0.);}
  if(expected){AMREX_ALWAYS_ASSERT(result.final_evaluation_attempted && result.final_evaluation_valid && result.final_residual_finite);}
  if(!enabled){AMREX_ALWAYS_ASSERT(!result.final_evaluation_attempted && !result.final_evaluation_valid && !result.final_residual_finite);}
  if(enabled && !op.final_failure.empty()){
    AMREX_ALWAYS_ASSERT(result.status==ThermalSolveStatus::InvalidTrial && result.final_evaluation_attempted && op.near_root==2);
    if(op.final_failure=="invalid"){
      AMREX_ALWAYS_ASSERT(!result.final_evaluation_valid && !result.final_residual_finite && std::isnan(result.residual));
      for(auto norm:result.block_residuals)AMREX_ALWAYS_ASSERT(std::isnan(norm));
    }else{
      AMREX_ALWAYS_ASSERT(result.final_evaluation_valid);
      if(op.final_failure=="nonfinite")AMREX_ALWAYS_ASSERT(!result.final_residual_finite && !std::isfinite(result.block_residuals[1]));
      else{
        AMREX_ALWAYS_ASSERT(result.final_residual_finite);
        int const failed=op.final_failure=="field"?0:1;
        AMREX_ALWAYS_ASSERT(result.block_residuals[failed]>result.block_targets[failed]);
        Vec expected_residual;expected_residual.Define(x);expected_residual.zero();
        if(failed==0)for(auto& level:expected_residual.getArrayVec())for(auto* field:level)field->setVal(1.);
        else expected_residual.getMultiFabBlock(energy,0).setVal(1.);
        AMREX_ALWAYS_ASSERT(result.block_residuals[failed]==expected_residual.blockNorms()[failed]);
      }
    }
  }
  WarpX::Finalize();
 }
 warpx::initialization::finalize_external_libraries();
}
