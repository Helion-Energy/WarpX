/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CircuitCoupler.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Python/callbacks.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <utility>

using namespace amrex;
using namespace amrex::literals;
using warpx::fields::FieldType;

CircuitCoupler::CircuitCoupler (warpx::circuit::CoilSet const& coils,
                                std::vector<warpx::circuit::ProbeKind> probes,
                                std::vector<double> probe_exclusion,
                                Params params,
                                std::unique_ptr<ExternalCircuit> plugin,
                                const warpx::circuit::ProbeRegionTable* region)
    : m_coils(coils),
      m_probes(std::move(probes)),
      m_probe_exclusion(std::move(probe_exclusion)),
      m_params(std::move(params)),
      m_plugin(std::move(plugin)),
      m_region(region)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(m_probes.size()) == m_coils.size() &&
            static_cast<int>(m_probe_exclusion.size()) == m_coils.size(),
        "CircuitCoupler: probe kind / exclusion vectors must match the coil set");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_params.probe_region == ProbeRegion::domain &&
         m_params.probe_region_report.empty()) ||
            (m_region != nullptr && m_region->IsDefined()),
        "CircuitCoupler: circuit.probe_region = wall_interior and "
        "circuit.probe_region_report need the wall-polyline region table");
}

const amrex::MultiFab*
CircuitCoupler::LinkageWeightField () const
{
    auto& warpx = WarpX::GetInstance();
    if (!warpx.m_fields.has(LinkageWeightName, 0)) { return nullptr; }
    return warpx.m_fields.get(LinkageWeightName, 0);
}

void
CircuitCoupler::RefreshWeightedCurrent (const amrex::MultiFab& j_theta,
                                        const amrex::MultiFab& weight)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        weight.boxArray() == j_theta.boxArray() &&
            weight.DistributionMap() == j_theta.DistributionMap(),
        "circuit_linkage_weight must live on the J_theta (node r, node z) "
        "layout");
    if (!m_j_weighted.ok() ||
        !(m_j_weighted.boxArray() == j_theta.boxArray()) ||
        !(m_j_weighted.DistributionMap() == j_theta.DistributionMap())) {
        m_j_weighted.define(j_theta.boxArray(), j_theta.DistributionMap(),
                            1, 0);
    }
    // Valid nodes only: the probes read owner-masked valid nodes.
    amrex::MultiFab::Copy(m_j_weighted, j_theta, 0, 0, 1, 0);
    amrex::MultiFab::Multiply(m_j_weighted, weight, 0, 0, 1, 0);
}

warpx::circuit::ProbeNodeFilter
CircuitCoupler::MeasurementFilter () const
{
    warpx::circuit::ProbeNodeFilter filter;
    if (m_params.probe_region == ProbeRegion::wall_interior) {
        filter.region = m_region;
        filter.radial = warpx::circuit::RadialRegion::inside_wall;
    }
    return filter;
}

void
CircuitCoupler::MeasureLinkages (const bool refresh_plasma_current)
{
    using namespace warpx::circuit;
    auto& warpx = WarpX::GetInstance();
    auto* hybrid = warpx.get_pointer_HybridPICModel();

    // J-based rows (reciprocity on the coil's unit field, or the analytic
    // loop probe) need the plasma current; disk rows read B_z.
    bool any_reciprocity = false;
    bool any_disk = false;
    for (const ProbeKind kind : m_probes) {
        if (kind == ProbeKind::reciprocity || kind == ProbeKind::loop) {
            any_reciprocity = true;
        }
        if (kind == ProbeKind::disk) { any_disk = true; }
    }
    if (any_reciprocity && refresh_plasma_current) {
        // J_plasma = curl B / mu0 - J_ext of the CURRENT plasma-frame B.
        // Implicit consumers whose residual evaluation just computed it
        // pass refresh_plasma_current = false and skip this full-grid
        // curl.
        hybrid->CalculatePlasmaCurrent(
            warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp,
                                                 warpx.finestLevel()),
            warpx.GetEBUpdateEFlag());
    }

    // Batched measurement: one device pass, one stream synchronization,
    // one all-reduce over the coil vector (the per-coil single probes
    // remain the validated reference below).
    const amrex::MultiFab* bz = any_disk
        ? warpx.m_fields.get(FieldType::Bfield_fp,
                             ablastr::fields::Direction{2}, 0)
        : nullptr;
    const amrex::MultiFab* j_theta = any_reciprocity
        ? warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                             ablastr::fields::Direction{1}, 0)
        : nullptr;
    m_a_theta_scratch.assign(m_coils.size(), nullptr);
    for (int ic = 0; ic < m_coils.size(); ++ic) {
        if (m_probes[ic] == ProbeKind::reciprocity) {
            m_a_theta_scratch[ic] = warpx.m_fields.get(
                m_coils.coil(ic).field_name + "_Aext",
                ablastr::fields::Direction{1}, 0);
        }
    }
    // Physical-share weighting (Params::probe_weight): the J-based probes
    // integrate w * J_theta, formed once here on a scratch copy so the
    // batched tables and the reference probes below see the identical
    // weighted current. The register is refreshed by the solver right
    // before this call (NeedsLinkageWeight).
    const amrex::MultiFab* j_measure = j_theta;
    if (any_reciprocity &&
        m_params.probe_weight == ProbeWeight::physical_share) {
        const amrex::MultiFab* weight = LinkageWeightField();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(weight != nullptr,
            "circuit.probe_weight = physical_share requires the "
            "theta-implicit MHD solver's circuit_linkage_weight register "
            "(algo.evolve_scheme = theta_implicit_mhd)");
        RefreshWeightedCurrent(*j_theta, *weight);
        j_measure = &m_j_weighted;
    }
    // Region mask (Params::probe_region): folded into the batched weight
    // tables and applied by the reference probes through the same filter.
    const ProbeNodeFilter filter = MeasurementFilter();
    m_batch.Measure(m_coils, m_probes, m_probe_exclusion, m_a_theta_scratch,
                    bz, j_measure, m_lambda_scratch, filter);
    for (int ic = 0; ic < m_coils.size(); ++ic) {
        if (m_probes[ic] == ProbeKind::none) { continue; }
        m_lambda[m_coils.coil(ic).name] = m_lambda_scratch[ic];
    }

    if (m_params.probe_crosscheck) {
        // Validation mode: re-measure through the single-coil reference
        // probes and pin the batched values against them.
        for (int ic = 0; ic < m_coils.size(); ++ic) {
            const Coil& c = m_coils.coil(ic);
            const ProbeKind kind = m_probes[ic];
            if (kind == ProbeKind::none) { continue; }
            const amrex::Real reference =
                (kind == ProbeKind::disk) ? DiskFluxLinkage(c, *bz)
                : (kind == ProbeKind::loop)
                    ? LoopLinkage(c, *j_measure, m_probe_exclusion[ic], filter)
                    : ReciprocityLinkage(*m_a_theta_scratch[ic], *j_measure,
                                         filter);
            const amrex::Real batched = m_lambda_scratch[ic];
            const amrex::Real scale = std::max(
                std::abs(reference), std::abs(batched));
            const amrex::Real delta = std::abs(batched - reference);
            amrex::Print() << "circuit probe_crosscheck: coil " << c.name
                           << " batched = " << batched
                           << " reference = " << reference
                           << " |delta| = " << delta << "\n";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                delta <= m_params.crosscheck_rtol * std::max(scale, 1.0e-300),
                "circuit.probe_crosscheck: batched linkage of coil '" +
                c.name + "' disagrees with the single-coil reference probe");
        }
    }
}

void
CircuitCoupler::RefreshCircuitFields (const amrex::Real t0, const amrex::Real t1)
{
    // B at the interval midpoint of the linear segments; E carries the
    // exact constant slope regardless of the evaluation time.
    auto& ext = *WarpX::GetInstance().get_pointer_HybridPICModel()
                     ->m_external_vector_potential;
    ext.UpdateHybridExternalFields(0.5_rt * (t0 + t1), t1 - t0,
        ExternalVectorPotential::RefreshMode::CircuitOnly);
}

std::vector<amrex::Real>
CircuitCoupler::CoupledScales (const amrex::Real t) const
{
    auto& warpx = WarpX::GetInstance();
    auto& ext = *warpx.get_pointer_HybridPICModel()->m_external_vector_potential;
    std::vector<amrex::Real> s;
    for (int ic = 0; ic < m_coils.size(); ++ic) {
        if (m_probes[ic] == warpx::circuit::ProbeKind::none) { continue; }
        s.push_back(ext.GetScale(m_coils.coil(ic).field_name, t));
    }
    return s;
}

void
CircuitCoupler::FireEngine (char const* hook, const bool accept)
{
    if (accept) {
        // The accepting evaluation's field registers hold the accepted
        // state the linkages were just measured on: the region report's
        // once-per-step (per accepted substep) row set.
        WriteRegionReport("accept", m_interval.t1);
    }
    if (!m_plugin) {
        ExecutePythonCallback(hook);
        return;
    }
    // Compiled-engine dispatch of the identical contract.
    const std::string h(hook);
    if (h == "circuitbeginstep") {
        m_plugin->BeginStep(m_interval.t0, m_interval.t1 - m_interval.t0);
    } else if (h == "circuitfinish") {
        m_plugin->FinishStep();
    } else {
        // Per-coil EMF estimates in volts: the linkage registers hold
        // lambda_phys * I_ref * n_turns, so the port EMF is
        // d lambda / dt / (I_ref * n_turns). Unmeasured (probe = none)
        // coils get zero; engines keep their own held/smoothed EMF.
        //
        // Optional one-pole low-pass (Params::eps_lowpass_tau > 0): the
        // fresh interval-averaged EMF is blended with the per-coil memory
        // committed at the previous ACCEPTED evaluation,
        //     e = sigma * e_raw + (1 - sigma) * e_mem,
        //     sigma = dt_step / (dt_step + tau),
        // dt_step being the coupling step (BeginStep*), not this
        // sub-interval. The memory moves ONLY here on accept = true: the
        // filtered value of the accepting evaluation becomes e_mem for
        // the next step, and every non-accepted evaluation in between
        // reads the same frozen memory -- the python reference's
        // "pending value committed by the finish hook" contract. (A step
        // replayed after a failed solve never accepted, so its memory is
        // untouched, exactly like its interval-entry linkages.) The
        // filter is linear, so applying it to the volts value here equals
        // the reference's filter on the linkage rate before its division
        // by the reference current up to roundoff.
        std::vector<amrex::Real> eps;
        // The EMF differencing span: the interval length, or the caller's
        // eps_interval (the theta-stage span while the engine advances
        // the full step, Params::residual_advance_full_step).
        const amrex::Real dt_sub = (m_eps_interval > 0.0_rt)
            ? m_eps_interval : (m_interval.t1 - m_interval.t0);
        const bool lowpass =
            (m_params.eps_lowpass_tau > 0.0_rt && m_step_dt > 0.0_rt);
        const amrex::Real sigma = lowpass
            ? m_step_dt / (m_step_dt + m_params.eps_lowpass_tau) : 1.0_rt;
        for (int ic = 0; ic < m_coils.size(); ++ic) {
            const warpx::circuit::Coil& c = m_coils.coil(ic);
            const bool measured =
                (m_probes[ic] != warpx::circuit::ProbeKind::none);
            amrex::Real e = 0.0_rt;
            // An open-loop step (no linkage reference yet, accepted mode)
            // hands eps = 0 raw; the low-pass still runs on it so the
            // memory evolves exactly as the reference's does.
            if (measured && !m_open_loop_step && dt_sub > 0.0_rt &&
                m_lambda.count(c.name) > 0) {
                const amrex::Real lam0 = m_lambda_start.count(c.name)
                    ? m_lambda_start.at(c.name) : m_lambda.at(c.name);
                e = (m_lambda.at(c.name) - lam0) / dt_sub
                    / (c.I_ref * c.n_turns);
            }
            if (lowpass && measured) {
                const auto mem = m_eps_filt.find(c.name);
                const amrex::Real e_mem =
                    (mem != m_eps_filt.end()) ? mem->second : 0.0_rt;
                e = sigma * e + (1.0_rt - sigma) * e_mem;
                if (accept) { m_eps_filt[c.name] = e; }
            }
            eps.push_back(e);
        }

        std::vector<amrex::Real> scales;
        m_plugin->AdvanceInterval(m_interval.t0, m_interval.t1, eps, accept,
                                  scales);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(scales.size()) == m_coils.size(),
            "ExternalCircuit::AdvanceInterval returned " +
            std::to_string(scales.size()) + " scales for " +
            std::to_string(m_coils.size()) + " coils");
        if (accept) {
            // The accepting evaluation's linkages ARE the accepted state's:
            // the next step's EMF reference under linkage_reference =
            // accepted (the reference coupler caches lambda^n here, in its
            // finish hook, because between steps the field registers hold
            // totals and cannot be measured).
            m_lambda_accepted = m_lambda;
            m_have_lambda_accepted = true;
        }

        // Realize the engine's scales on the field registers as linear
        // segments over the interval (the plugin ABI is self-contained:
        // the engine calls no WarpX symbols). GetScale(t0) of the live
        // segment is the interval-entry scale and stays fixed across
        // repeated (corrector) re-pushes of the same interval.
        auto& ext = *WarpX::GetInstance().get_pointer_HybridPICModel()
                         ->m_external_vector_potential;
        for (int ic = 0; ic < m_coils.size(); ++ic) {
            const std::string& fname = m_coils.coil(ic).field_name;
            const amrex::Real s_old = ext.GetScale(fname, m_interval.t0);
            ext.SetScale(fname, s_old, scales[ic],
                         m_interval.t0, m_interval.t1);
        }
    }
}

void
CircuitCoupler::BeginStep (const amrex::Real t0, const amrex::Real dt)
{
    // The EMF low-pass commits its memory on every accepting evaluation.
    // Under this (explicit-hybrid) substep protocol the engine accepts once
    // per SUBSTEP while sigma is formed from the step dt, which would make
    // the effective time constant tau / n_substeps: the filter is defined
    // for the measured-step protocol (BeginStepMeasured, one accept per
    // step) only, and is refused here rather than applied with the wrong
    // weight.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_params.eps_lowpass_tau <= 0.0_rt,
        "circuit.eps_lowpass_tau is defined for the measured-step coupling "
        "protocol (the theta-implicit MHD residual hooks) only; the explicit "
        "substep protocol accepts once per substep and does not implement "
        "the filter. Set circuit.eps_lowpass_tau = 0 for this solver.");
    m_interval = Interval{t0, t0 + dt, -1, 0};
    m_substep_count = 0;
    m_step_dt = dt;
    // The substep protocol measures its own interval-entry linkages
    // (PredictSubstep) and differences over the substep.
    m_open_loop_step = false;
    m_eps_interval = -1.0_rt;
    // The evolved fields hold the plasma response here (called after the
    // split-field subtraction): seed the linkage registers at t^n.
    MeasureLinkages(true);
    m_lambda_start = m_lambda;
    FireEngine("circuitbeginstep", false);
}

void
CircuitCoupler::PredictSubstep (const amrex::Real t0, const amrex::Real t1)
{
    m_interval = Interval{t0, t1, m_substep_count, 0};
    MeasureLinkages(true);
    m_lambda_start = m_lambda;
    FireEngine("circuitpredict", false);
    RefreshCircuitFields(t0, t1);
}

bool
CircuitCoupler::CorrectSubstep (const amrex::Real t0, const amrex::Real t1)
{
    MeasureLinkages(true);   // lambda(t1) of the current field iterate
    m_interval.iteration += 1;
    const std::vector<amrex::Real> s_prev = CoupledScales(t1);
    FireEngine("circuitcorrect", false);
    const std::vector<amrex::Real> s_new = CoupledScales(t1);

    amrex::Real delta = 0.0;
    for (std::size_t i = 0; i < s_new.size(); ++i) {
        const amrex::Real denom =
            std::max(std::abs(s_new[i]), m_params.scale_floor);
        delta = std::max(delta, std::abs(s_new[i] - s_prev[i]) / denom);
    }
    if (delta < m_params.corrector_rtol) {
        // The re-advanced scales match the segments the fields already
        // used: converged, no field re-run needed.
        return true;
    }
    if (m_interval.iteration >= m_params.corrector_iterations) {
        ablastr::warn_manager::WMRecordWarning(
            "CircuitCoupler",
            "circuit corrector did not converge to corrector_rtol within "
            "corrector_iterations passes; the last correction is applied "
            "unverified",
            ablastr::warn_manager::WarnPriority::medium);
    }
    RefreshCircuitFields(t0, t1);
    return false;
}

void
CircuitCoupler::AcceptSubstep (const amrex::Real t0, const amrex::Real t1)
{
    m_interval = Interval{t0, t1, m_substep_count, m_interval.iteration};
    ++m_substep_count;
    // Final linkages of the accepted substep: the engine's next predictor
    // reads these as its interval-entry values.
    MeasureLinkages(true);
    if (m_plugin) {
        FireEngine("circuitaccept", true);
    }
}

void
CircuitCoupler::FinishStep ()
{
    m_interval.iteration = 0;
    FireEngine("circuitfinish", false);
}

void
CircuitCoupler::BeginStepMeasured (const amrex::Real t0, const amrex::Real dt)
{
    m_interval = Interval{t0, t0 + dt, -1, 0};
    m_substep_count = 0;
    m_step_dt = dt;
    m_eps_interval = -1.0_rt;
    if (m_params.linkage_reference_accepted) {
        // The step's EMF reference is the previous accepting evaluation's
        // linkage (the accepted t^n state as the finish hook saw it).
        // None yet -> the step runs open loop, eps = 0 throughout (the
        // reference coupler's first-step convention). A step replayed
        // after a failed solve re-enters here with the same reference.
        m_open_loop_step = !m_have_lambda_accepted;
        if (m_have_lambda_accepted) {
            m_lambda_start = m_lambda_accepted;
        } else {
            m_lambda_start.clear();
        }
    } else {
        // The caller measured the committed t^n state (MeasureLinkages):
        // seed the interval-entry linkage registers from it. The engine
        // snapshots its accepted state; the first EvaluateInterval of the
        // step then sees eps = 0 exactly (the ABI's predictor convention).
        m_open_loop_step = false;
        m_lambda_start = m_lambda;
    }
    FireEngine("circuitbeginstep", false);
}

void
CircuitCoupler::EvaluateInterval (const amrex::Real t0, const amrex::Real t1,
                                  const bool accept,
                                  const amrex::Real eps_interval)
{
    m_interval.t0 = t0;
    m_interval.t1 = t1;
    m_interval.substep = 0;
    m_eps_interval = eps_interval;
    if (!accept) {
        m_interval.iteration += 1;
        if (m_interval.iteration == 1) {
            // The first non-accepting evaluation of the step: the Newton
            // initial iterate, i.e. the committed t^n state (the residual
            // has just computed its plasma current) -- the report's
            // exact committed-state split, before any evolution.
            WriteRegionReport("first", t0);
        }
    }
    // accept = false: restore-and-re-advance from the interval entry (a
    // pure function of the caller's latest measurement). accept = true:
    // exactly once per step, on the accepted final state -- discontinuous
    // engine transitions latch here, in committed time.
    FireEngine(accept ? "circuitaccept" : "circuitcorrect", accept);
}

amrex::Real
CircuitCoupler::CoilLinkage (std::string const& coil_name) const
{
    const auto it = m_lambda.find(coil_name);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(it != m_lambda.end(),
        "CoilLinkage: no measured linkage for coil '" + coil_name +
        "' (unknown coil, or its probe is 'none')");
    return it->second;
}

amrex::Real
CircuitCoupler::CoilLinkageOr (std::string const& coil_name,
                               const amrex::Real fallback) const
{
    const auto it = m_lambda.find(coil_name);
    return (it != m_lambda.end()) ? it->second : fallback;
}

void
CircuitCoupler::InitRegionReport (const bool restarting)
{
    if (m_params.probe_region_report.empty()) { return; }
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream ofs{m_params.probe_region_report,
                      restarting ? (std::ofstream::out | std::ofstream::app)
                                 : std::ofstream::out};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ofs.good(),
        "circuit.probe_region_report: cannot open '" +
        m_params.probe_region_report + "'");
    if (restarting) {
        ofs << "# restart: rows below continue a restarted run\n";
        m_report_header_written = true;
        return;
    }
    ofs << "# circuit probe region report: one row per J-based coil "
           "(reciprocity / loop probes; disk-probe coils are not current "
           "integrals and are absent) per tagged evaluation\n"
        << "# kind: first = the first residual evaluation of the step (the "
           "Newton initial iterate = the committed t^n state; t = t^n); "
           "accept = the accepting evaluation (once per step, or per "
           "accepted substep under the explicit protocol; t = t^{n+1}); "
           "step = the step being computed (1-based)\n"
        << "# columns: kind step t coil lambda_total lambda_interior "
           "lambda_band lambda_exterior lambda_w_plasma lambda_w_mixed "
           "lambda_w_boost lambda_weighted lambda_used jtheta_rms_interior "
           "jtheta_rms_band jtheta_rms_exterior lambda_weighted_inside\n"
        << "#   lambda_total    = the unmasked, unweighted integral over "
           "the whole domain; = interior + band + exterior = w_plasma + "
           "w_mixed + w_boost to roundoff\n"
        << "#   interior / band / exterior: nodes with r < r_wall(z) - 2 dr, "
           "r_wall(z) - 2 dr <= r < r_wall(z), r >= r_wall(z) (node "
           "(r_i, z_j) inside the wall iff r_i < r_wall(z_j), r_wall = "
           "piecewise-linear, end-clamped interpolation of the wall "
           "polyline; numpy: r_node < np.interp(z_node, z_poly, r_poly))\n"
        << "#   w_plasma / w_mixed / w_boost: nodes with linkage weight "
           "w = eta_phys/eta_field > 0.9, 0.1 <= w <= 0.9, w < 0.1 (the "
           "register circuit_linkage_weight; without it w == 1: w_plasma = "
           "total, w_mixed = w_boost = 0)\n"
        << "#   lambda_weighted = Int w A_theta J_theta dV over the whole "
           "domain; lambda_weighted_inside = the same over r < r_wall(z) "
           "(what probe_weight = physical_share with probe_region = "
           "wall_interior measures)\n"
        << "#   lambda_used     = the value the coupler used (its "
           "probe_region mask and probe_weight applied)\n"
        << "#   jtheta_rms_*    = RMS of the (unweighted) nodal J_theta over "
           "the owner-masked nodes of each radial region (per evaluation, "
           "repeated on every coil row)\n"
        << "# lambda in Wb*A (lambda_phys * I_ref * n_turns), all reals "
           "%.17g\n";
    // The tabulated rule, for an exact cross-check against
    // numpy.interp: r_wall at every z node of the domain.
    {
        auto& warpx = WarpX::GetInstance();
        const auto& geom = warpx.Geom(0);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", geom.ProbLo(1));
        ofs << "# r_wall_table j_lo=" << m_region->JLo() << " z_lo=" << buf;
        std::snprintf(buf, sizeof(buf), "%.17g", geom.CellSize(1));
        ofs << " dz=" << buf;
        std::snprintf(buf, sizeof(buf), "%.17g", geom.CellSize(0));
        ofs << " dr=" << buf << " values:";
        for (const double v : m_region->WallRadiusHost()) {
            std::snprintf(buf, sizeof(buf), " %.17g", v);
            ofs << buf;
        }
        ofs << "\n";
    }
    m_report_header_written = true;
}

void
CircuitCoupler::WriteRegionReport (const char* kind, const amrex::Real time)
{
    if (m_params.probe_region_report.empty()) { return; }
    using namespace warpx::circuit;
    BL_PROFILE("CircuitCoupler::WriteRegionReport");
    auto& warpx = WarpX::GetInstance();

    bool any_j = false;
    for (const ProbeKind k : m_probes) {
        if (k == ProbeKind::reciprocity || k == ProbeKind::loop) {
            any_j = true;
        }
    }
    if (!any_j) { return; }

    const amrex::MultiFab* j_theta = warpx.m_fields.get(
        FieldType::hybrid_current_fp_plasma, ablastr::fields::Direction{1}, 0);
    const amrex::MultiFab* weight = LinkageWeightField();
    if (weight != nullptr) { RefreshWeightedCurrent(*j_theta, *weight); }
    const amrex::MultiFab* j_weighted = (weight != nullptr) ? &m_j_weighted
                                                            : j_theta;

    // One row per J-based coil: the reference (single-coil) probes under
    // the radial, weight-class and weighted variants. Correctness over
    // speed -- a diagnostic, a handful of reductions per coil per step.
    struct Row
    {
        std::string coil;
        std::array<double, 10> v{};
    };
    std::vector<Row> rows;
    for (int ic = 0; ic < m_coils.size(); ++ic) {
        const ProbeKind kind_ic = m_probes[ic];
        if (kind_ic != ProbeKind::reciprocity && kind_ic != ProbeKind::loop) {
            continue;
        }
        const Coil& c = m_coils.coil(ic);
        const amrex::MultiFab* a_theta = (kind_ic == ProbeKind::reciprocity)
            ? warpx.m_fields.get(c.field_name + "_Aext",
                                 ablastr::fields::Direction{1}, 0)
            : nullptr;
        const double excl = m_probe_exclusion[ic];
        auto measure = [&] (const amrex::MultiFab& j,
                            const ProbeNodeFilter& f) -> double
        {
            return static_cast<double>(
                (kind_ic == ProbeKind::loop)
                    ? LoopLinkage(c, j, excl, f)
                    : ReciprocityLinkage(*a_theta, j, f));
        };
        auto radial = [&] (const RadialRegion r) {
            ProbeNodeFilter f;
            f.region = m_region;
            f.radial = r;
            return f;
        };
        auto wclass = [&] (const WeightClass w) {
            ProbeNodeFilter f;
            f.weight = weight;
            f.wclass = w;
            return f;
        };
        // Column order of the report (after kind step t coil):
        // total interior band exterior w_plasma w_mixed w_boost weighted
        // used [rms x3 appended by the writer] weighted_inside.
        Row row;
        row.coil = c.name;
        const double total = measure(*j_theta, ProbeNodeFilter{});
        row.v[0] = total;
        row.v[1] = measure(*j_theta, radial(RadialRegion::interior));
        row.v[2] = measure(*j_theta, radial(RadialRegion::band));
        row.v[3] = measure(*j_theta, radial(RadialRegion::exterior));
        if (weight != nullptr) {
            row.v[4] = measure(*j_theta, wclass(WeightClass::plasma));
            row.v[5] = measure(*j_theta, wclass(WeightClass::mixed));
            row.v[6] = measure(*j_theta, wclass(WeightClass::boost));
        } else {
            row.v[4] = total;
            row.v[5] = 0.0;
            row.v[6] = 0.0;
        }
        row.v[7] = measure(*j_weighted, ProbeNodeFilter{});
        row.v[8] = static_cast<double>(m_lambda_scratch.empty()
                                           ? 0.0 : m_lambda_scratch[ic]);
        row.v[9] = measure(*j_weighted, radial(RadialRegion::inside_wall));
        rows.push_back(std::move(row));
    }
    const std::array<double, 3> rms = NodalRmsByRegion(*j_theta, *m_region);

    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream ofs{m_params.probe_region_report,
                      std::ofstream::out | std::ofstream::app};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ofs.good(),
        "circuit.probe_region_report: cannot append to '" +
        m_params.probe_region_report + "'");
    // The step being computed (1-based): WarpX's counter holds the number
    // of completed steps while a step is in progress, for both row kinds.
    const int step = warpx.getistep(0) + 1;
    char buf[64];
    auto put = [&] (const double v) {
        std::snprintf(buf, sizeof(buf), " %.17g", v);
        ofs << buf;
    };
    for (const Row& row : rows) {
        ofs << kind << " " << step;
        put(static_cast<double>(time));
        ofs << " " << row.coil;
        for (int q = 0; q < 9; ++q) { put(row.v[q]); }
        for (const double v : rms) { put(v); }
        put(row.v[9]);
        ofs << "\n";
    }
}

namespace
{
    /** Bit-exact text form of a double (C99 hexfloat) for the memory
     * checkpoint; read back with strtod. */
    std::string HexDouble (const double v)
    {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%a", v);
        return buf;
    }

    void WriteMemoryBlock (std::ofstream& ofs, const char* key,
                           std::map<std::string, amrex::Real> const& block)
    {
        ofs << key << " " << block.size() << "\n";
        for (const auto& [name, value] : block) {
            ofs << name << " " << HexDouble(static_cast<double>(value)) << "\n";
        }
    }
}

void
CircuitCoupler::WriteMemoryCheckpoint (std::string const& dir) const
{
    const std::string path = dir + "/circuit_coupler_memory.dat";
    std::ofstream ofs{path, std::ofstream::out};
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ofs.good(),
        "CircuitCoupler: cannot write '" + path + "'");
    // Block-structured: "<key> <count>" then <count> "<name> <hexfloat>"
    // lines; a reader skips nothing and aborts on an unknown key, so the
    // format can only grow by adding blocks.
    ofs << "version 1\n";
    WriteMemoryBlock(ofs, "eps_filt", m_eps_filt);
    if (m_have_lambda_accepted) {
        WriteMemoryBlock(ofs, "lambda_accepted", m_lambda_accepted);
    }
}

void
CircuitCoupler::ReadMemoryCheckpoint (std::string const& dir)
{
    const std::string path = dir + "/circuit_coupler_memory.dat";
    std::ifstream ifs{path, std::ifstream::in};
    if (!ifs.good()) {
        // Checkpoints from before the memory file existed: the memory
        // restarts from zero (a filter warm-up transient, disclosed) and
        // an accepted-linkage reference is absent (open-loop first step).
        amrex::Print() << "Circuit coupler memory: no " << path
                       << " in the checkpoint; the EMF low-pass memory "
                       << "restarts from zero and no accepted linkage "
                       << "reference is available\n";
        return;
    }
    std::string token;
    int version = 0;
    ifs >> token >> version;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(token == "version" && version == 1,
        "unsupported circuit_coupler_memory.dat format in '" + path + "'");
    std::string key;
    std::size_t count = 0;
    while (ifs >> key >> count) {
        std::map<std::string, amrex::Real>* block = nullptr;
        if (key == "eps_filt") { block = &m_eps_filt; }
        if (key == "lambda_accepted") {
            block = &m_lambda_accepted;
            m_have_lambda_accepted = true;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(block != nullptr,
            "circuit_coupler_memory.dat: unknown block '" + key + "'");
        block->clear();
        for (std::size_t i = 0; i < count; ++i) {
            std::string name, hex;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bool(ifs >> name >> hex),
                "circuit_coupler_memory.dat: truncated block '" + key + "'");
            (*block)[name] = static_cast<amrex::Real>(
                std::strtod(hex.c_str(), nullptr));
        }
    }
    amrex::Print() << "Circuit coupler memory: restored from " << path
                   << " (" << m_eps_filt.size() << " EMF low-pass entries, "
                   << (m_have_lambda_accepted
                           ? std::to_string(m_lambda_accepted.size())
                                 + " accepted-linkage entries"
                           : std::string("no accepted-linkage block"))
                   << ")\n";
}
