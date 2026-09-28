/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_STAGE_SOLVER_H_
#define WARPX_THERMAL_STAGE_SOLVER_H_
#include "EulerianThermalStage.H"
#include "WarpXSolverVec.H"
#include "ThermalConductionPC.H"
#include <optional>

namespace warpx::thermal {
struct ThermalSolveOptions {
    amrex::Real relative_tolerance = 1.e-10;
    amrex::Real absolute_tolerance = 1.e-12; // scaled WarpXSolverVec norm
    amrex::Real linear_relative_tolerance = 1.e-8;
    // Same safeguarded inexact-Newton forcing prescription as the MHD driver.
    // Final block tolerances and physical acceptance are never changed.
    bool adaptive_forcing = false;
    amrex::Real forcing_alpha = 1.5;
    amrex::Real forcing_gamma = 0.9;
    amrex::Real forcing_max = 0.5;
    amrex::Real probe_relative_size = 6.e-6;
    int max_newton_iterations = 20;
    int max_linear_iterations = 300;
    int restart_length = 60;
    int max_backtracks = 24;
    int pc_cycles = 2;
    // When present, all PC settings (including cycles) come from this object.
    std::optional<ThermalPCOptions> preconditioner;
    int linear_verbosity = 0;
    // Match the field JFNK rhs_scale_floor option: a zero/small state must
    // not force an unresolvably small probe of a large physical equation.
    bool probe_rhs_scale_floor = false;
    bool use_preconditioner = true;
    // Empty inherits the scalar tolerances. Otherwise each entry follows
    // WarpXSolverVec::blockNames(); every block must converge independently.
    std::vector<amrex::Real> block_relative_tolerances;
    std::vector<amrex::Real> block_absolute_tolerances;
    // Optional references held across outer field/thermal/E_L sweeps. An
    // inner solve must not repeatedly tighten the same physical step target.
    std::vector<amrex::Real> block_reference_norms;
};
enum class ThermalSolveStatus {
    Converged,
    InvalidTrial,
    LinearFailure,
    NoDescent,
    IterationLimit
};
struct ThermalSolveResult {
    ThermalSolveStatus status = ThermalSolveStatus::InvalidTrial;
    // A candidate root is checked by a fresh full evaluation. These flags
    // distinguish a rejected verification from failures before that check.
    bool final_evaluation_attempted = false;
    bool final_evaluation_valid = false;
    bool final_residual_finite = false;
    // After an attempted verification, residual/block_residuals describe that
    // fresh evaluation; a false callback leaves them explicitly unavailable.
    amrex::Real initial_residual = 0.0;
    amrex::Real residual = 0.0;
    int newton_iterations = 0;
    int linear_iterations = 0;
    int linear_solves = 0;
    amrex::Real linear_tolerance_min = 0.0;
    amrex::Real linear_tolerance_max = 0.0;
    std::vector<amrex::Real> linear_relative_tolerances;
    int residual_evaluations = 0;
    int rejected_trials = 0;
    int pc_updates = 0;
    int linear_status = -1;
    amrex::Real linear_residual = 0.0;
    std::vector<amrex::Real> initial_block_residuals;
    std::vector<amrex::Real> block_residuals;
    std::vector<amrex::Real> block_targets;
};

/** Physics adapter for the shared safeguarded Newton/FGMRES driver.
 * Residual must refresh every live consumer of this trial, including mixed
 * field/energy Jacobian probes. All arrays remain native MultiFabs. Accepted
 * physical state, RNG and energy/circuit ledgers must not change in any hook.
 * A failed solve calls RestoreInput exactly once, after the last trial. A
 * successful solve leaves a fresh non-Jacobian evaluation at the returned
 * state; committing physical endpoints remains the caller's responsibility.
 */
class ThermalNonlinearOperator {
  public:
    virtual ~ThermalNonlinearOperator () = default;
    virtual bool Residual (WarpXSolverVec& residual, const WarpXSolverVec& state,
                           int nonlinear_iteration, bool from_jacobian) = 0;
    // A conservative bound for a physical (already scaled) direction. A live
    // density-dependent constraint still has to be checked in Residual.
    virtual amrex::Real StepBound (const WarpXSolverVec& state,
                                   const WarpXSolverVec& direction) const = 0;
    // Freeze the Newton/MM base even with PC disabled. The PC must own its
    // snapshots: later live residuals/Jv may change the thermal context.
    virtual bool Freeze (const WarpXSolverVec& state, int nonlinear_iteration,
                         bool use_pc, ThermalSolveResult& statistics) = 0;
    virtual bool Precondition (WarpXSolverVec& output,
                               const WarpXSolverVec& rhs) = 0;
    virtual void RestoreInput (const WarpXSolverVec& input) = 0;
    // Called only just after a successful non-Jacobian Residual at this exact
    // state. Transformed block norms need not certify every physical gate.
    // This pure query must not evaluate, mutate or publish physical state.
    // Other operators retain their original block-only stopping contract.
    virtual bool PhysicalConverged (const WarpXSolverVec&) const { return true; }
    // Optional representation of the SAME affine field input. Default hooks
    // leave every other operator's vector arithmetic and lifecycle unchanged.
    // The physical owner may normalize E's high and retain its low; U/phi,
    // directions, epsilon, line search and convergence targets are unchanged.
    virtual bool BeginInputSolve(WarpXSolverVec const&) { return true; }
    virtual bool FormInput(WarpXSolverVec&,WarpXSolverVec const&,amrex::Real,
                           WarpXSolverVec const&,bool) { return true; }
    virtual bool AcceptInput(WarpXSolverVec const&) { return true; }
    virtual void EndInputSolve(bool) noexcept {}
    // Optional owned copy-out from the most recent successful physical map.
    // The driver copies PLUS before MINUS can overwrite producer scratch.
    virtual bool HasResidualRemainder () const noexcept { return false; }
    virtual bool CopyResidualRemainder (WarpXSolverVec&) const { return false; }
    virtual bool DifferenceResidualRemainder (WarpXSolverVec&,
        WarpXSolverVec const&,WarpXSolverVec const&,WarpXSolverVec const&,
        WarpXSolverVec const&,amrex::Real) const { return false; }
};

// Shared scalar/coupled engine. Solution is unchanged on every failure. Probe
// sizes respect each block's fixed reference scale, and convergence is checked
// block by block after a fresh physical residual, never from a Krylov estimate.
ThermalSolveResult SolveThermalSystem (ThermalNonlinearOperator& physical,
                                       WarpXSolverVec& solution,
                                       const ThermalSolveOptions& options = {});

// A transactional scalar driver on native WarpXSolverVec + FlexibleGMRES.
// The field Newton/Jacobian templates hard-instantiate field PC and MM hooks;
// this bounded driver avoids changes to their physics. Only the named U block
// is allowed. On every failure BOTH output arguments remain bitwise unchanged.
// On success both are written only after a fresh full residual and positive
// endpoint gate. No registry export, wall tally, RNG or accepted-state hook.
ThermalSolveResult SolveThermalStage (EulerianThermalStage& stage,
                                      WarpXSolverVec& solution,
                                      amrex::MultiFab& endpoint,
                                      const std::string& energy_name,
                                      const ThermalSolveOptions& options = {});
} // namespace warpx::thermal
#endif
