/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThermalStageSolver.H"
#include "ThermalCurrentRemainder.H"

#include "NonlinearSolvers/FlexibleGMRES.H"
#include "ThermalConductionPC.H"

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::thermal {
namespace {
using RT = amrex::Real;
using Vec = WarpXSolverVec;

// Native FGMRES Ops with a fixed Newton base and independently frozen PC.
// Only scalar reductions cross ranks; work vectors stay on the native backend.
class ThermalJacobian {
  public:
    using RT = amrex::Real;
    ThermalJacobian (ThermalNonlinearOperator& physical, const Vec& model,
                     const ThermalSolveOptions& options, ThermalSolveResult& stats)
        : m_physical(physical), m_options(options), m_stats(stats)
    {
        m_base.Define(model);
        m_plus.Define(model);
        m_minus.Define(model);
        m_fplus.Define(model);
        m_fminus.Define(model);
        m_negative.Define(model);
    }

    bool Freeze (const Vec& state, const Vec& residual, int iteration)
    {
        m_base.Copy(state);
        m_base_norms = state.blockNorms();
        if (m_options.probe_rhs_scale_floor) {
            auto const rhs_norms = residual.blockNorms();
            for (std::size_t b = 0; b < m_base_norms.size(); ++b) {
                m_base_norms[b] = std::max(m_base_norms[b], rhs_norms[b]);
            }
        }
        m_iteration = iteration;
        m_failed = false;
        return m_physical.Freeze(state, iteration, m_options.use_preconditioner, m_stats);
    }

    bool Evaluate (Vec& out, const Vec& state, int iteration, bool from_jacobian = false)
    {
        ++m_stats.residual_evaluations;
        return m_physical.Residual(out, state, iteration, from_jacobian);
    }

    bool EvaluateAffine(Vec& out,Vec& input,RT factor,Vec const& direction) {
        return m_physical.FormInput(input,m_base,factor,direction,true) &&
            Evaluate(out,input,m_iteration,true);
    }
    Vec makeVecLHS () const { Vec x; x.Define(m_base); return x; }
    Vec makeVecRHS () const { return makeVecLHS(); }
    void setToZero (Vec& x) const { x.zero(); }
    void assign (Vec& x, const Vec& y) const { x.Copy(y); }
    void SetTargetWeights (std::vector<RT> const& weights) {
        auto const scales=m_base.blockScales();
        AMREX_ALWAYS_ASSERT(scales.size()==weights.size());
        for(std::size_t b=0;b<weights.size();++b) {
            RT const ratio=weights[b]/scales[b];
            RT const square=ratio*ratio;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(square) && square>0.,
                "Target-weighted Krylov metric is not representable");
            m_dot_weights.push_back(square);
        }
    }
    RT norm2 (const Vec& x) const {
        return m_dot_weights.empty() ? x.norm2() : std::sqrt(dotProduct(x,x));
    }
    RT dotProduct (const Vec& x, const Vec& y) const {
        if(m_dot_weights.empty()) return x.dotProduct(y);
        x.assertSameType(y);
        RT sum=0.;auto const& dofs=*x.getDOFsObject();
        for(int lev=0;lev<x.numAMRLevels();++lev) {
            std::size_t block=0;
            auto add=[&](amrex::MultiFab const& a,amrex::MultiFab const& b,
                         amrex::iMultiFab const& owner) {
                sum+=m_dot_weights[block]*amrex::MultiFab::Dot(
                    owner,a,0,b,0,a.nComp(),0,true);
            };
            if(x.getArrayVecType()!=warpx::fields::FieldType::None) {
                for(int c=0;c<3;++c) add(*x.getArrayVec()[lev][c],
                    *y.getArrayVec()[lev][c],*dofs.m_array_masks[lev][c]);
                ++block;
            }
            if(x.getScalarVecType()!=warpx::fields::FieldType::None) {
                add(*x.getScalarVec()[lev],*y.getScalarVec()[lev],
                    *dofs.m_scalar_masks[lev]);++block;
            }
            auto const& specs=x.getMultiFabBlockSpecs();
            for(std::size_t b=0;b<specs.size();++b,++block)
                add(x.getMultiFabBlock(specs[b].name,lev),
                    y.getMultiFabBlock(specs[b].name,lev),
                    *dofs.m_multifab_blocks[b].masks[lev]);
            AMREX_ALWAYS_ASSERT(block==m_dot_weights.size());
        }
        amrex::ParallelAllReduce::Sum(sum,amrex::ParallelContext::CommunicatorSub());
        return sum;
    }
    void scale (Vec& x, RT a) const { x.scale(a); }
    void increment (Vec& x, const Vec& y, RT a) const { x.increment(y, a); }
    void linComb (Vec& z, RT a, const Vec& x, RT b, const Vec& y) const
    { z.linComb(a, x, b, y); }

    void precond (Vec& out, const Vec& rhs)
    {
        if (m_options.use_preconditioner) {
            if (!m_physical.Precondition(out, rhs)) { Fail(out); }
        } else {
            out.Copy(rhs);
        }
    }

    void apply (Vec& out, const Vec& direction)
    {
        auto const norms = direction.blockNorms();
        bool nonzero = false;
        RT eps = std::numeric_limits<RT>::infinity();
        // A strong E block must not set an oversized finite-difference probe
        // for a small thermal block (or vice versa). Scales are fixed by the
        // vector layout for the entire solve. This reduces to the scalar M3
        // rule when only U is present.
        for (std::size_t b = 0; b < norms.size(); ++b) {
            if (!std::isfinite(norms[b]) || !std::isfinite(m_base_norms[b])) {
                Fail(out);
                return;
            }
            if (norms[b] > 0) {
                nonzero = true;
                eps = std::min(eps, m_options.probe_relative_size *
                                   std::max(RT(1), m_base_norms[b]) / norms[b]);
            }
        }
        if (!nonzero) { out.zero(); return; }
        m_plus.Copy(direction);
        m_plus.scale(eps);
        m_negative.Copy(m_plus);
        m_negative.scale(-1);
        // Bound the physical epsilon*direction, not an unscaled Krylov vector.
        RT const bound = std::min(m_physical.StepBound(m_base, m_plus),
                                  m_physical.StepBound(m_base, m_negative));
        if (!std::isfinite(bound) || bound < 0 || bound > 1) { Fail(out); return; }
        if (bound < 1) { eps *= RT(0.5) * bound; }
        if (!std::isfinite(eps) || eps <= 0) { Fail(out); return; }
        m_plus.linComb(1, m_base, eps, direction);
        m_minus.linComb(1, m_base, -eps, direction);
        if(!m_physical.HasResidualRemainder()) {
        if (!EvaluateAffine(m_fplus,m_plus,eps,direction) ||
            !EvaluateAffine(m_fminus,m_minus,-eps,direction)) {
            Fail(out);
            return;
        }
        out.linComb(RT(0.5) / eps, m_fplus, -RT(0.5) / eps, m_fminus);
        } else {
            if(!m_plus_remainder.IsDefined()) {
                m_plus_remainder.Define(m_base);m_minus_remainder.Define(m_base);
            }
            if(!EvaluateAffine(m_fplus,m_plus,eps,direction) ||
               !m_physical.CopyResidualRemainder(m_plus_remainder) ||
               !EvaluateAffine(m_fminus,m_minus,-eps,direction) ||
               !m_physical.CopyResidualRemainder(m_minus_remainder)) { Fail(out);return; }
            out.linComb(RT(0.5)/eps,m_fplus,-RT(0.5)/eps,m_fminus);
            if(!m_physical.DifferenceResidualRemainder(out,m_fplus,m_plus_remainder,
                m_fminus,m_minus_remainder,RT(0.5)/eps)) {Fail(out);return;}
        }
    }

    bool Failed () const { return m_failed; }

  private:
    void Fail (Vec& out)
    {
        m_failed = true;
        out.setVal(std::numeric_limits<RT>::quiet_NaN());
    }
    ThermalNonlinearOperator& m_physical;
    const ThermalSolveOptions& m_options;
    ThermalSolveResult& m_stats;
    Vec m_base, m_plus, m_minus, m_fplus, m_fminus, m_negative;
    Vec m_plus_remainder,m_minus_remainder;
    std::vector<RT> m_base_norms;
    std::vector<RT> m_dot_weights;
    int m_iteration = 0;
    bool m_failed = false;
};

class ScalarThermalOperator final : public ThermalNonlinearOperator {
  public:
    ScalarThermalOperator (EulerianThermalStage& stage, std::string name, ThermalPCOptions pc)
        : m_stage(stage), m_name(std::move(name)), m_pc(stage, pc) {}

    bool Residual (Vec& out, const Vec& state, int, bool) override
    { return m_stage.Residual(state.getMultiFabBlock(m_name, 0),
                              out.getMultiFabBlock(m_name, 0)); }

    RT StepBound (const Vec& state, const Vec& direction) const override
    { return m_stage.StepBound(state.getMultiFabBlock(m_name, 0),
                               direction.getMultiFabBlock(m_name, 0)); }

    bool Freeze (const Vec& state, int, bool use_pc, ThermalSolveResult& stats) override
    {
        if (!use_pc) { return true; }
        ++stats.residual_evaluations; // PC emits coefficients from the live residual.
        bool const ok = m_pc.Freeze(state.getMultiFabBlock(m_name, 0));
        stats.pc_updates = m_pc.Updates();
        return ok;
    }
    bool Precondition (Vec& out, const Vec& rhs) override
    {
        m_pc.Apply(out.getMultiFabBlock(m_name, 0), rhs.getMultiFabBlock(m_name, 0));
        return true;
    }
    bool HasResidualRemainder() const noexcept override {return m_stage.HasResidualRemainder();}
    bool CopyResidualRemainder(Vec& out) const override {
        out.zero();return m_stage.CopyResidualRemainder(out.getMultiFabBlock(m_name,0));
    }
    bool DifferenceResidualRemainder(Vec& out,Vec const& p,Vec const& pl,Vec const& n,Vec const& nl,RT factor) const override {
        return remainder::Difference(out.getMultiFabBlock(m_name,0),p.getMultiFabBlock(m_name,0),
            pl.getMultiFabBlock(m_name,0),n.getMultiFabBlock(m_name,0),nl.getMultiFabBlock(m_name,0),factor);
    }
    // The scalar stage owns scratch only; it never exports into physical state.
    void RestoreInput (const Vec&) override {}

  private:
    EulerianThermalStage& m_stage;
    std::string m_name;
    ThermalConductionPC m_pc;
};

struct RestoreOnFailure {
    ThermalNonlinearOperator& physical;
    const Vec& input;
    bool success = false;
    ~RestoreOnFailure () { if (!success) { physical.RestoreInput(input); } physical.EndInputSolve(success); }
};

bool Converged (const std::vector<RT>& norms, const std::vector<RT>& targets)
{
    for (std::size_t b = 0; b < norms.size(); ++b) {
        if (!std::isfinite(norms[b]) || norms[b] > targets[b]) { return false; }
    }
    return true;
}
} // namespace

ThermalSolveResult SolveThermalSystem (ThermalNonlinearOperator& physical,
                                       WarpXSolverVec& solution,
                                       const ThermalSolveOptions& o)
{
    auto const block_count = solution.blockNames().size();
    bool accept_converged_trial=true;
    amrex::ParmParse("endpoint_diagnostic").query("accept_converged_trial",accept_converged_trial);
    bool block_target_merit=false;
    amrex::ParmParse("endpoint_diagnostic").query("block_target_merit",block_target_merit);
    block_target_merit=block_target_merit && block_count>1;
    bool block_target_linear=false;
    amrex::ParmParse("endpoint_diagnostic").query("block_target_linear",block_target_linear);
    block_target_linear=block_target_linear && block_count>1;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!block_target_linear || block_target_merit,
        "Target-weighted Krylov requires the same target-weighted merit");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solution.IsDefined() && block_count > 0 &&
        std::isfinite(o.relative_tolerance) && o.relative_tolerance >= 0 &&
        std::isfinite(o.absolute_tolerance) && o.absolute_tolerance >= 0 &&
        o.relative_tolerance + o.absolute_tolerance > 0 &&
        std::isfinite(o.linear_relative_tolerance) && o.linear_relative_tolerance > 0 &&
        o.linear_relative_tolerance < 1 &&
        std::isfinite(o.forcing_alpha) && o.forcing_alpha > 1 &&
        std::isfinite(o.forcing_gamma) && o.forcing_gamma > 0 && o.forcing_gamma < 1 &&
        std::isfinite(o.forcing_max) && o.forcing_max > 0 && o.forcing_max < 1 &&
        std::isfinite(o.probe_relative_size) && o.probe_relative_size > 0 &&
        o.max_newton_iterations >= 0 && o.max_linear_iterations >= 0 &&
        o.restart_length > 0 && o.max_backtracks >= 0 &&
        (o.block_relative_tolerances.empty() ||
         o.block_relative_tolerances.size() == block_count) &&
        (o.block_absolute_tolerances.empty() ||
         o.block_absolute_tolerances.size() == block_count) &&
        (o.block_reference_norms.empty() || o.block_reference_norms.size() == block_count),
        "Invalid thermal nonlinear solver options/layout");
    std::vector<RT> rtols(block_count, o.relative_tolerance);
    std::vector<RT> atols(block_count, o.absolute_tolerance);
    if (!o.block_relative_tolerances.empty()) { rtols = o.block_relative_tolerances; }
    if (!o.block_absolute_tolerances.empty()) { atols = o.block_absolute_tolerances; }
    for (std::size_t b = 0; b < block_count; ++b) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(rtols[b]) && rtols[b] >= 0 && std::isfinite(atols[b]) &&
            atols[b] >= 0 && rtols[b] + atols[b] > 0,
            "Invalid thermal nonlinear block tolerances");
    }
    ThermalSolveResult result;
    RestoreOnFailure restore{physical, solution};
    if(!physical.BeginInputSolve(solution)) return result;
    Vec state, residual, direction, trial, trial_residual;
    state.Copy(solution);
    residual.Define(solution);
    direction.Define(solution);
    trial.Define(solution);
    trial_residual.Define(solution);
    ThermalJacobian ops(physical, solution, o, result);
    if (!ops.Evaluate(residual, state, 0)) { return result; }
    result.initial_residual = residual.norm2();
    result.residual = result.initial_residual;
    result.initial_block_residuals = residual.blockNorms();
    result.block_residuals = result.initial_block_residuals;
    if (!std::isfinite(result.residual) ||
        std::any_of(result.block_residuals.begin(), result.block_residuals.end(),
                    [](RT value) { return !std::isfinite(value); })) { return result; }
    std::vector<RT> targets(block_count);
    for (std::size_t b = 0; b < block_count; ++b) {
        RT const reference = o.block_reference_norms.empty()
            ? result.initial_block_residuals[b] : o.block_reference_norms[b];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(reference) && reference >= 0,
            "Thermal block reference must be finite and nonnegative");
        targets[b] = std::max(atols[b], rtols[b] * reference);
    }
    result.block_targets = targets;
    std::vector<RT> merit_weights;
    if(block_target_merit) {
        RT const smallest=*std::min_element(targets.begin(),targets.end());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(smallest>0.,
            "Target-weighted merit requires strictly positive fixed block targets");
        for(auto target:targets) merit_weights.push_back(smallest/target);
    }
    auto merit=[&](std::vector<RT> const& norms) {
        RT value=0.;
        for(std::size_t b=0;b<norms.size();++b)
            value=std::hypot(value,merit_weights[b]*norms[b]);
        return value;
    };

    if(block_target_linear) ops.SetTargetWeights(merit_weights);
    FlexibleGMRES<Vec, ThermalJacobian> linear;
    linear.define(ops);
    linear.setVerbose(o.linear_verbosity);
    linear.setMaxIters(o.max_linear_iterations);
    linear.setRestartLength(o.restart_length);
    RT forcing_previous = o.forcing_max;
    RT norm_previous = block_target_linear ? merit(result.initial_block_residuals)
                                           : result.initial_residual;
    // A composite norm target could hide a smaller unconverged block. Use
    // the smallest fixed block target for the MHD last-iteration safeguard;
    // the actual convergence test remains the independent per-block check.
    RT const forcing_target = *std::min_element(targets.begin(), targets.end());
    for (int iteration = 0;; ++iteration) {
        result.newton_iterations = iteration;
        // Each physical block must pass after a fresh full evaluation. An
        // aggregate norm may otherwise hide a failing smaller equation.
        if (std::isfinite(result.residual) && Converged(result.block_residuals, targets)) {
            bool physical_pending = false;
            result.final_evaluation_attempted = true;
            result.final_evaluation_valid = ops.Evaluate(trial_residual, state, iteration);
            if (result.final_evaluation_valid) {
                auto const norms = trial_residual.blockNorms();
                result.residual = trial_residual.norm2();
                result.block_residuals = norms;
                result.final_residual_finite = std::isfinite(result.residual) &&
                    std::all_of(norms.begin(), norms.end(),
                                [](RT value) { return std::isfinite(value); });
                if (Converged(norms, targets)) {
                    if (physical.PhysicalConverged(state)) {
                        solution.Copy(state);
                        restore.success = true;
                        result.status = ThermalSolveStatus::Converged;
                        return result;
                    }
                    // The fresh transformed blocks pass, but a distinct
                    // physical row (e.g. grad(phi), not phi) still fails.
                    // Continue Newton from this same fresh residual under
                    // the original targets, forcing, line search and caps.
                    residual.Copy(trial_residual);
                    physical_pending = true;
                }
            } else {
                // A false callback may leave a partial residual. Do not reduce
                // it or report the preceding, apparently passing trial norms.
                result.residual = std::numeric_limits<RT>::quiet_NaN();
                result.block_residuals.assign(block_count, result.residual);
            }
            if (!physical_pending) {
                amrex::Print() << "Thermal final verification rejected: attempted="
                               << result.final_evaluation_attempted
                               << " valid=" << result.final_evaluation_valid
                               << " finite=" << result.final_residual_finite << "\n";
                result.status = ThermalSolveStatus::InvalidTrial;
                return result;
            }
            amrex::Print() << "Thermal physical convergence pending iteration="
                           << iteration << "\n";
        }
        if (!std::isfinite(result.residual)) { return result; }
        if (iteration == o.max_newton_iterations) {
            result.status = ThermalSolveStatus::IterationLimit;
            return result;
        }
        if (!ops.Freeze(state, residual, iteration)) { return result; }
        RT linear_tolerance = o.linear_relative_tolerance;
        RT const forcing_norm=block_target_linear ? merit(result.block_residuals)
                                                 : result.residual;
        if (o.adaptive_forcing) {
            // MHD's safeguarded forcing: loosen intermediate linear solves,
            // then tighten from the accepted nonlinear residual decrease.
            RT zeta = o.forcing_max;
            if (iteration > 0 && norm_previous > 0) {
                RT const predicted = o.forcing_gamma *
                    std::pow(forcing_norm / norm_previous, o.forcing_alpha);
                RT const safeguard = o.forcing_gamma *
                    std::pow(forcing_previous, o.forcing_alpha);
                zeta = std::min(o.forcing_max, std::max(predicted, safeguard));
            }
            if (forcing_norm > 0) {
                zeta = std::min(o.forcing_max,
                    std::max(zeta, o.forcing_gamma * forcing_target / forcing_norm));
            }
            if (!std::isfinite(zeta) || zeta <= 0 || zeta >= 1) {
                result.status = ThermalSolveStatus::LinearFailure;
                return result;
            }
            forcing_previous = zeta;
            norm_previous = forcing_norm;
            linear_tolerance = zeta;
        }
        ++result.linear_solves;
        result.linear_relative_tolerances.push_back(linear_tolerance);
        if (result.linear_solves == 1) {
            result.linear_tolerance_min = result.linear_tolerance_max = linear_tolerance;
        } else {
            result.linear_tolerance_min = std::min(result.linear_tolerance_min, linear_tolerance);
            result.linear_tolerance_max = std::max(result.linear_tolerance_max, linear_tolerance);
        }
        linear.solve(direction, residual, linear_tolerance, 0);
        result.linear_iterations += linear.getNumIters();
        result.linear_status = linear.getStatus();
        result.linear_residual = linear.getResidualNorm();
        if (linear.getStatus() != 0 || ops.Failed()) {
            result.status = ThermalSolveStatus::LinearFailure;
            return result;
        }
        direction.scale(-1);
        RT const bound = physical.StepBound(state, direction);
        if (!std::isfinite(bound) || bound < 0 || bound > 1) { return result; }
        RT alpha = bound < 1 ? RT(0.99) * bound : RT(1);
        bool accepted = false;
        for (int backtrack = 0;
             backtrack <= o.max_backtracks && alpha > RT(1.e-12); ++backtrack) {
            trial.linComb(1, state, alpha, direction);
            bool const valid = physical.FormInput(trial,state,alpha,direction,false) &&
                ops.Evaluate(trial_residual, trial, iteration + 1);
            RT const trial_norm = valid ? trial_residual.norm2()
                                       : std::numeric_limits<RT>::infinity();
            // A valid trial passing every physical block gate is already a
            // candidate root. A converged block's roundoff floor must not make
            // aggregate Armijo reject it. The loop still performs its fresh
            // full physical reevaluation before reporting convergence.
            bool const trial_converged = accept_converged_trial && valid &&
                Converged(trial_residual.blockNorms(),targets) &&
                physical.PhysicalConverged(trial);
            RT const old_merit=block_target_merit ? merit(result.block_residuals)
                                                 : result.residual;
            RT const trial_merit=block_target_merit && valid
                ? merit(trial_residual.blockNorms()) : trial_norm;
            if (std::isfinite(trial_norm) &&
                (trial_converged ||
                 trial_merit <= (1 - RT(1.e-4) * alpha) * old_merit)) {
                if(!physical.AcceptInput(trial)) return result;
                state.Copy(trial);
                residual.Copy(trial_residual);
                result.residual = trial_norm;
                result.block_residuals = residual.blockNorms();
                accepted = true;
                break;
            }
            ++result.rejected_trials;
            alpha *= RT(0.5);
        }
        if (!accepted) {
            result.status = ThermalSolveStatus::NoDescent;
            return result;
        }
    }
}

ThermalSolveResult SolveThermalStage (EulerianThermalStage& stage, WarpXSolverVec& solution,
                                      amrex::MultiFab& endpoint, const std::string& energy_name,
                                      const ThermalSolveOptions& options)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solution.getArrayVecType() == warpx::fields::FieldType::None &&
        solution.getScalarVecType() == warpx::fields::FieldType::None &&
        solution.numAMRLevels() == 1 && solution.getMultiFabBlockSpecs().size() == 1 &&
        solution.hasMultiFabBlock(energy_name),
        "Thermal scalar solve requires only the named U block");
    auto const& input_energy = solution.getMultiFabBlock(energy_name, 0);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        &endpoint != &input_energy && endpoint.is_cell_centered() && endpoint.nComp() == 1 &&
        !endpoint.hasEBFabFactory() && endpoint.boxArray() == input_energy.boxArray() &&
        endpoint.DistributionMap() == input_energy.DistributionMap(),
        "Thermal endpoint output must have the U layout and must not alias the stage output");
    ThermalPCOptions pc;
    pc.cycles = options.pc_cycles;
    if (options.preconditioner) { pc = *options.preconditioner; }
    ScalarThermalOperator physical(stage, energy_name, pc);
    auto result = SolveThermalSystem(physical, solution, options);
    if (result.status == ThermalSolveStatus::Converged) {
        stage.Endpoint(solution.getMultiFabBlock(energy_name, 0), endpoint);
    }
    return result;
}
} // namespace warpx::thermal
