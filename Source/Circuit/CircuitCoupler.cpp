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

#include <algorithm>
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
                                std::unique_ptr<ExternalCircuit> plugin)
    : m_coils(coils),
      m_probes(std::move(probes)),
      m_probe_exclusion(std::move(probe_exclusion)),
      m_params(params),
      m_plugin(std::move(plugin))
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(m_probes.size()) == m_coils.size() &&
            static_cast<int>(m_probe_exclusion.size()) == m_coils.size(),
        "CircuitCoupler: probe kind / exclusion vectors must match the coil set");
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
    m_batch.Measure(m_coils, m_probes, m_probe_exclusion, m_a_theta_scratch,
                    bz, j_theta, m_lambda_scratch);
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
                    ? LoopLinkage(c, *j_theta, m_probe_exclusion[ic])
                    : ReciprocityLinkage(*m_a_theta_scratch[ic], *j_theta);
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
    if (!accept) { m_interval.iteration += 1; }
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
