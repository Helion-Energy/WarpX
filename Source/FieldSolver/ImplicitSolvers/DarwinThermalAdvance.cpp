/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitHybrid.H"
#include "DarwinThermalAdvance.H"
#include "DarwinVacuumJointSolve.H"
#include "ThermalSourceAcceptance.H"
#include "ImplicitParticleEndpointAudit.H"
#include "Particles/MultiParticleContainer.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include <ablastr/coarsen/sample.H>
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>

namespace warpx::thermal {
using Real = amrex::Real;
using Vec = WarpXSolverVec;
using warpx::fields::FieldType;
using ablastr::fields::Direction;

bool DarwinThermalAdvance::JointVacuumActive() const noexcept { return m_joint_vacuum && m_joint_vacuum->Active(); }
int DarwinThermalAdvance::SolveJointVacuum(int step) {
    m_joint_vacuum=std::make_unique<DarwinVacuumJointSolve>(*this);
    return m_joint_vacuum->Solve(step);
}
bool DarwinThermalAdvance::PublishJointVacuumStage() {
    return m_joint_vacuum && m_joint_vacuum->PublishVerifiedStage();
}

bool DarwinThermalAdvance::CaptureJointVacuumEndpointStage() {
    bool ready=bool(m_joint_vacuum);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    return ready && m_joint_vacuum->CapturePublishedEndpointStage();
}

bool DarwinThermalAdvance::FinishJointVacuumParticles(amrex::Real endpoint_time) {
    return m_joint_vacuum && m_joint_vacuum->FinishParticles(endpoint_time);
}

bool DarwinThermalAdvance::PrepareJointVacuumEndpointAmpere(amrex::Real time) {
    bool ready=bool(m_joint_vacuum);amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    return ready&&m_joint_vacuum->PrepareEndpointAmpere(time);
}
std::shared_ptr<NativeEndpointAmpereOrigin const>
DarwinThermalAdvance::RetainJointVacuumEndpointAmpereOrigin(amrex::Real time) {
    bool ready=m_open&&m_receipt_ready&&m_candidate_published&&bool(m_joint_vacuum);
    amrex::ParallelDescriptor::ReduceBoolAnd(ready);if(!ready)return {};
    return m_joint_vacuum->RetainEndpointAmpereOrigin(time);
}
bool DarwinThermalAdvance::VerifyJointVacuumEndpointAmpere(amrex::Real time) {
    bool ready=bool(m_joint_vacuum);amrex::ParallelDescriptor::ReduceBoolAnd(ready);
    return ready&&m_joint_vacuum->VerifyEndpointAmpere(time);
}

bool DarwinThermalAdvance::CheckJointVacuumEndpointSupport() {
    return m_joint_vacuum && m_joint_vacuum->CheckFinishedSupport();
}

bool DarwinThermalAdvance::FieldResidual (Vec& out,const Vec& electric,int iteration,bool probe)
{
    m_solver.ComputeRHS(out,electric,m_time,iteration,probe);
    if(m_solver.m_native_endpoint_pair){
        // ComputeRHS has left the native transverse Ohm field in Efield_fp.
        // Cancel the common old (high+low) solver offset algebraically before
        // rounding this row: E - Eold - (Eohm - Eold) = E - Eohm. The existing
        // RHS API and every OFF consumer retain their original arithmetic.
        // No old field enters a BDF/time-difference recovery through this seam.
        out.Copy(FieldType::Efield_fp);
        out.linComb(1.,electric,-1.,out);
    }else{
        out.increment(m_solver.m_Eold,1.);
        out.scale(-1.);
        out.increment(electric,1.);
    }
    return std::isfinite(out.norm2());
}
bool DarwinThermalAdvance::ThermalResidual (amrex::MultiFab& out)
{
    if (!RefreshContext()) {
        amrex::Print()<<"Eulerian rejected thermal context refresh\n";
        return false;
    }
    bool const ok=m_stage->Residual(m_energy.getMultiFabBlock(energy_name,0),out);
    if (!ok) { amrex::Print()<<"Eulerian rejected thermal residual\n"; }
    return ok;
}

bool DarwinThermalAdvance::Residual (Vec& out,const Vec& state,int iteration,bool probe)
{
    CopyFields(m_field,state);
    amrex::MultiFab::Copy(m_energy.getMultiFabBlock(energy_name,0),state.getMultiFabBlock(energy_name,0),0,0,1,0);
    if (!FieldResidual(m_field_rhs,m_field,iteration,probe)) { return false; }
    CopyFields(out,m_field_rhs);
    return ThermalResidual(out.getMultiFabBlock(energy_name,0));
}

Real DarwinThermalAdvance::StepBound (const Vec& state,const Vec& direction) const
{
    auto const theta=m_solver.m_theta;
    auto const floor=PhysConst::q_e*m_stage_options.temperature_floor_ev/(m_stage_options.gamma-1);
    auto const& energy=state.getMultiFabBlock(energy_name,0);
    auto const& delta=direction.getMultiFabBlock(energy_name,0);
    amrex::ReduceOps<amrex::ReduceOpMin> op;
    amrex::ReduceData<Real> data(op);
    using Tuple=decltype(data)::Type;
    for (amrex::MFIter mfi(energy);mfi.isValid();++mfi) {
        auto const u=energy.const_array(mfi),du=delta.const_array(mfi);
        auto const old=m_old_energy.const_array(mfi),n=m_bound_density.const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
            Real const margin=u(i,j,k)-(1-theta)*old(i,j,k)-theta*n(i,j,k)*floor;
            return {du(i,j,k)<0 ? amrex::max(Real(0),amrex::min(Real(1),margin/-du(i,j,k))) : Real(1)};
        });
    }
    Real bound=amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealMin(bound);
    return bound;
}
bool DarwinThermalAdvance::Freeze (const Vec& state,int iteration,bool use_pc,ThermalSolveResult& stats)
{
    // Jv changes live endpoint density. Positivity bounds must use this
    // immutable Newton base, with actual trial density checked in Residual.
    amrex::MultiFab::Copy(m_bound_density,m_stage->EndpointDensity(),0,0,1,0);
    m_solver.PreLinearSolve(iteration);
    if (!use_pc) { return true; }
    ++stats.pc_updates;
    m_field_pc.updatePreCondMat(m_field);
    return m_pc->Freeze(state.getMultiFabBlock(energy_name,0)) && FreezePressureResponse();
}
bool DarwinThermalAdvance::Precondition (Vec& out,const Vec& rhs)
{
    m_pc->Apply(out.getMultiFabBlock(energy_name,0),rhs.getMultiFabBlock(energy_name,0));
    CopyFields(m_field_work,rhs);
    if (m_pressure_pc) {
        FrozenPressureFieldCoupling::FieldView response;
        for (int c=0;c<3;++c) { response[c]=m_pressure_work.getArrayVec()[0][c]; }
        m_pressure_pc->Apply(response,out.getMultiFabBlock(energy_name,0));
        // Physical fields and energy are stored unscaled. Norm scales do not
        // enter the pressure block of F_E=E_T+E_L-E_Ohm.
        m_field_work.increment(m_pressure_work,-1.);
    }
    m_field_pc.precond(m_field_rhs,m_field_work);
    CopyFields(out,m_field_rhs);
    return std::isfinite(out.norm2());
}
bool DarwinThermalAdvance::FreezePressureResponse ()
{
    if (!m_pressure_pc) { return true; }
    using ablastr::coarsen::sample::Interp;
    auto& rho_register=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    // The native Ohm denominator consumes the deposited midpoint component.
    // Vacuum replacement rows instead use component zero of the full rho
    // register, matching ApplyVacuumFaradayE and the native field PC. Keep
    // these two density views distinct; a frozen mask has one component.
    amrex::MultiFab rho(rho_register,amrex::make_alias,rho_register.nComp()/2,1);
    auto const* pedestal=m_model.DensityPedestal(0);
    auto const* mask=m_solver.m_vacuum_recovery_half && !JointVacuumActive()
        ? (m_model.m_darwin_vacuum_recovery_frozen_mask
            ? m_simulation.m_fields.get("hybrid_rho_vacmask_fp",0) : &rho_register) : nullptr;
    bool const flux_only=m_model.m_darwin_vacuum_recovery_components=="flux";
    int const mask_mode=m_model.m_darwin_vacuum_recovery_mask=="global" ? 2 :
        (m_model.m_darwin_vacuum_recovery_mask=="transition" ? 1 : 0);
    Real const mask_floor=PhysConst::q_e*m_model.m_n_floor*m_model.m_darwin_vacuum_recovery_density_fraction;
    PressureResponseParameters p;
    p.charge_floor=PhysConst::q_e*m_model.m_n_floor;
    p.floor_width=m_model.m_n_floor_smooth_width*p.charge_floor;
    p.holmstrom_width=m_model.m_holmstrom_transition_width*p.charge_floor;
    p.axis_radius=m_model.m_holmstrom_axis_radius;
    p.axis_rolloff=m_model.m_holmstrom_axis_rolloff;
    p.pressure_enabled=m_model.m_include_electron_pressure_term;
    p.holmstrom=m_model.m_holmstrom_vacuum_region;
    p.conductor_raw_gate=m_model.m_pec_conductor_wall_rows;
    auto const end=m_model.EndRegion(0);
    auto const dx=m_geometry.CellSizeArray(),lo=m_geometry.ProbLoArray();
    amrex::GpuArray<int,3> const nodal{1,1,1},coarsen{1,1,1};
    FrozenPressureFieldCoupling::FieldView weights;
    FrozenPressureFieldCoupling::ConstFieldView coefficients;
    for (int c=0;c<3;++c) {
        auto& weight=*m_pressure_weight[c];
        weights[c]=&weight; coefficients[c]=&weight;
        auto const iv=weight.ixType().toIntVect();
        auto const stagger=PressureResponseStagger(iv);
        bool const replace=mask && (!flux_only || c==1);
        for (amrex::MFIter mfi(weight);mfi.isValid();++mfi) {
            auto const w=weight.array(mfi); auto const raw=rho.const_array(mfi);
            auto const ped=pedestal ? pedestal->const_array(mfi) : amrex::Array4<const Real>{};
            auto const vac=mask ? mask->const_array(mfi) : amrex::Array4<const Real>{};
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real response=end.holmstrom ? 1.-end.Weight(i,j,k,stagger) : 1.;
                if (replace) {
                    Real const n=Interp(vac,nodal,stagger,coarsen,i,j,k,0);
                    if (mask_mode==2 || (mask_mode==0 && n<mask_floor) ||
                        (mask_mode==1 && n>0 && n<mask_floor)) { response=0.; }
                }
                w(i,j,k)=PressureResponseWeight(
                    Interp(raw,nodal,stagger,coarsen,i,j,k,0),
                    ped ? Interp(ped,nodal,stagger,coarsen,i,j,k,0) : Real(0),
                    lo[0]+(i+Real(0.5)*(1-stagger[0]))*dx[0],p,response);
            });
        }
    }
    // Match the actual native Ohm boundary rows. These operations only zero
    // constrained valid rows; all coefficient ghosts have width zero.
    amrex::Vector<amrex::IntVect> const ratios;
    PEC::ApplyPECtoEfield(weights,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PEC,amrex::IntVect(0),m_simulation.Geom(0),0,
        ablastr::utils::enums::PatchType::fine,ratios);
    PEC::ApplyPECtoBfield(weights,WarpX::field_boundary_lo,WarpX::field_boundary_hi,
        FieldBoundaryType::PMC,amrex::IntVect(0),m_simulation.Geom(0),0,
        ablastr::utils::enums::PatchType::fine,ratios);
    return m_pressure_pc->Freeze(m_stage_moments->NumberDensity(),
        m_stage_moments->NodalNumberDensity(),coefficients);
}
void DarwinThermalAdvance::RestoreInput (const Vec& input)
{
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    amrex::MultiFab::Copy(m_energy.getMultiFabBlock(energy_name,0),input.getMultiFabBlock(energy_name,0),0,0,1,0);
    CopyFields(m_field,input);
    // Rebuild only trial state. No accepted U, particles, RNG or source ledger
    // is committed by ComputeRHS. Full step failure is handled by Cancel.
    FieldResidual(m_field_rhs,m_field,0,false);
    if(m_current_remainder_requested)InvalidateCurrentRemainder();
    m_solver.InvalidateMassMatrices();
}

bool DarwinThermalAdvance::PrepareInitialThermalGuess (bool& changed)
{
    auto const& rho=*m_simulation.m_fields.get(FieldType::rho_fp,0);
    KineticThermalStateView stage_density{rho};
    stage_density.charge_component=rho.nComp()/2;
    stage_density.pedestal=m_model.DensityPedestal(0);
    if (!m_stage_moments->Evaluate(stage_density)) { return false; }
    amrex::MultiFab::LinComb(m_endpoint_charge,2.,rho,rho.nComp()/2,-1.,
        m_old_charge,0,0,1,m_endpoint_charge.nGrowVect());
    KineticThermalStateView end_density{m_endpoint_charge};
    end_density.pedestal=m_model.DensityPedestal(0);
    if (!m_endpoint_moments->Evaluate(end_density)) { return false; }
    auto& u=m_energy.getMultiFabBlock(energy_name,0);
    amrex::MultiFab original(u.boxArray(),u.DistributionMap(),1,0);
    amrex::MultiFab::Copy(original,u,0,0,1,0);
    if (!MakeThermalStageInitialGuess(u,m_old_energy,u,
            m_stage_moments->NumberDensity(),m_endpoint_moments->NumberDensity(),m_guess_options)) {
        return false;
    }
    amrex::MultiFab::Subtract(original,u,0,0,1,0);
    changed=original.norminf()>0.;
    return true;
}

int DarwinThermalAdvance::Solve (Real field_reference,int step)
{
    AMREX_ALWAYS_ASSERT(m_open);
    Real field_rtol,field_atol; int maxits;
    m_solver.m_nlsolver->GetSolverParams(field_rtol,field_atol,maxits);
    Vec thermal_residual; thermal_residual.Define(m_energy);
    // A changed kinetic density may make the carried U an infeasible INITIAL
    // guess. Repair only this seed, before entering Newton. Physical residual,
    // Jv and accepted states never clip energy or alter their density floors.
    bool initialized=false;
    for (int attempt=0;attempt<4;++attempt) {
        bool const field_ok=FieldResidual(m_field_rhs,m_solver.m_E,0,false);
        bool changed=false;
        if (!PrepareInitialThermalGuess(changed)) { return -9; }
        if (changed) { continue; }
        if (!field_ok) {
            amrex::Print()<<"Eulerian rejected initial field residual\n"; return -9;
        }
        initialized=true; break;
    }
    if (!initialized || !ThermalResidual(thermal_residual.getMultiFabBlock(energy_name,0))) { return -9; }
    if (m_energy_reference<0) { m_energy_reference=thermal_residual.norm2(); }
    auto const energy_target=std::max(m_options.absolute_tolerance,m_options.relative_tolerance*m_energy_reference);
    auto const field_target=std::max(field_atol,field_rtol*field_reference);
    if (m_coupled) {
        CopyFields(m_combined,m_solver.m_E);
        amrex::MultiFab::Copy(m_combined.getMultiFabBlock(energy_name,0),
                              m_energy.getMultiFabBlock(energy_name,0),0,0,1,0);
        auto options=m_options;
        options.block_reference_norms={field_reference/m_electric_scale,m_energy_reference};
        options.block_relative_tolerances={field_rtol,m_options.relative_tolerance};
        options.block_absolute_tolerances={field_atol/m_electric_scale,m_options.absolute_tolerance};
        auto const result=SolveThermalSystem(*this,m_combined,options);
        amrex::Print()<<"Eulerian coupled: Newton="<<result.newton_iterations
                      <<" GMRES="<<result.linear_iterations<<" residual="<<result.residual
                      <<" status="<<static_cast<int>(result.status)
                      <<" linear_status="<<result.linear_status
                      <<" linear_residual="<<result.linear_residual
                      <<" linear_solves="<<result.linear_solves
                      <<" linear_rtol_min="<<result.linear_tolerance_min
                      <<" linear_rtol_max="<<result.linear_tolerance_max
                      <<" residual_calls="<<result.residual_evaluations
                      <<" rejected_trials="<<result.rejected_trials<<"\n";
        if (result.status!=ThermalSolveStatus::Converged) {
            for (std::size_t b=0;b<result.block_residuals.size();++b) {
                amrex::Print()<<"Eulerian failed block "<<b<<": residual="<<result.block_residuals[b]
                    <<" target="<<result.block_targets[b]<<"\n";
            }
        }
        if (result.status!=ThermalSolveStatus::Converged) { return -9; }
        CopyFields(m_solver.m_E,m_combined);
        amrex::MultiFab::Copy(m_energy.getMultiFabBlock(energy_name,0),
                              m_combined.getMultiFabBlock(energy_name,0),0,0,1,0);
        return 2;
    }
    for (int outer=0;outer<m_outer_iterations;++outer) {
        m_solver.m_nlsolver->SetConvergenceReferenceNorm(field_reference);
        m_solver.m_nlsolver->Solve(m_solver.m_E,m_solver.m_Eold,m_time,m_dt,step);
        m_solver.m_nlsolver->SetConvergenceReferenceNorm(0.);
        auto const status=m_solver.m_nlsolver->GetExitStatus();
        if (status!=2 && status!=3) { return -9; }
        if (!RefreshContext()) { return -9; }
        auto options=m_options; options.block_reference_norms={m_energy_reference};
        auto const result=SolveThermalStage(*m_stage,m_energy,m_endpoint,energy_name,options);
        amrex::Print()<<"Eulerian thermal: outer="<<outer
                      <<" Newton="<<result.newton_iterations
                      <<" GMRES="<<result.linear_iterations<<" residual="<<result.residual
                      <<" status="<<static_cast<int>(result.status)
                      <<" linear_status="<<result.linear_status
                      <<" linear_residual="<<result.linear_residual
                      <<" linear_solves="<<result.linear_solves
                      <<" linear_rtol_min="<<result.linear_tolerance_min
                      <<" linear_rtol_max="<<result.linear_tolerance_max
                      <<" residual_calls="<<result.residual_evaluations
                      <<" rejected_trials="<<result.rejected_trials<<"\n";
        if (result.status!=ThermalSolveStatus::Converged) { return -9; }
        // A thermal solve changes pressure and potentially the particle force.
        // Re-evaluate BOTH equations with those updated kinetic moments.
        if (!FieldResidual(m_field_rhs,m_solver.m_E,0,false) ||
            !ThermalResidual(thermal_residual.getMultiFabBlock(energy_name,0))) { return -9; }
        auto const ef=m_field_rhs.norm2(),uf=thermal_residual.norm2();
        amrex::Print()<<"Eulerian decoupled: outer="<<outer<<" E="<<ef<<" target="<<field_target
                      <<" U="<<uf<<" target="<<energy_target<<"\n";
        if (ef<=field_target && uf<=energy_target) { return 2; }
    }
    return -9;
}

} // namespace warpx::thermal
