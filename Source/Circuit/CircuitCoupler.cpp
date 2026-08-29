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
#include <utility>

using namespace amrex;
using namespace amrex::literals;
using warpx::fields::FieldType;

CircuitCoupler::CircuitCoupler (warpx::circuit::CoilSet const& coils,
                                std::vector<warpx::circuit::ProbeKind> probes,
                                Params params,
                                std::unique_ptr<ExternalCircuit> plugin)
    : m_coils(coils),
      m_probes(std::move(probes)),
      m_params(params),
      m_plugin(std::move(plugin))
{}

void
CircuitCoupler::MeasureLinkages ()
{
    using namespace warpx::circuit;
    auto& warpx = WarpX::GetInstance();
    auto* hybrid = warpx.get_pointer_HybridPICModel();

    bool any_reciprocity = false;
    for (const ProbeKind kind : m_probes) {
        if (kind == ProbeKind::reciprocity) { any_reciprocity = true; }
    }
    if (any_reciprocity) {
        // J_plasma = curl B / mu0 - J_ext of the CURRENT plasma-frame B
        hybrid->CalculatePlasmaCurrent(
            warpx.m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp,
                                                 warpx.finestLevel()),
            warpx.GetEBUpdateEFlag());
    }

    for (int ic = 0; ic < m_coils.size(); ++ic) {
        const Coil& c = m_coils.coil(ic);
        const ProbeKind kind = m_probes[ic];
        if (kind == ProbeKind::none) { continue; }
        amrex::Real lam = 0.0;
        if (kind == ProbeKind::disk) {
            const amrex::MultiFab& bz = *warpx.m_fields.get(
                FieldType::Bfield_fp, ablastr::fields::Direction{2}, 0);
            lam = DiskFluxLinkage(c, bz);
        } else {
            const amrex::MultiFab& a_theta = *warpx.m_fields.get(
                c.field_name + "_Aext", ablastr::fields::Direction{1}, 0);
            const amrex::MultiFab& j_theta = *warpx.m_fields.get(
                FieldType::hybrid_current_fp_plasma,
                ablastr::fields::Direction{1}, 0);
            lam = ReciprocityLinkage(a_theta, j_theta);
        }
        m_lambda[c.name] = lam;
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
        std::vector<amrex::Real> eps;
        const amrex::Real dt_sub = m_interval.t1 - m_interval.t0;
        for (int ic = 0; ic < m_coils.size(); ++ic) {
            const warpx::circuit::Coil& c = m_coils.coil(ic);
            amrex::Real e = 0.0_rt;
            if (m_probes[ic] != warpx::circuit::ProbeKind::none &&
                dt_sub > 0.0_rt && m_lambda.count(c.name) > 0) {
                const amrex::Real lam0 = m_lambda_start.count(c.name)
                    ? m_lambda_start.at(c.name) : m_lambda.at(c.name);
                e = (m_lambda.at(c.name) - lam0) / dt_sub
                    / (c.I_ref * c.n_turns);
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
    // The evolved fields hold the plasma response here (called after the
    // split-field subtraction): seed the linkage registers at t^n.
    MeasureLinkages();
    m_lambda_start = m_lambda;
    FireEngine("circuitbeginstep", false);
}

void
CircuitCoupler::PredictSubstep (const amrex::Real t0, const amrex::Real t1)
{
    m_interval = Interval{t0, t1, m_substep_count, 0};
    MeasureLinkages();
    m_lambda_start = m_lambda;
    FireEngine("circuitpredict", false);
    RefreshCircuitFields(t0, t1);
}

bool
CircuitCoupler::CorrectSubstep (const amrex::Real t0, const amrex::Real t1)
{
    MeasureLinkages();   // lambda(t1) of the current field iterate
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
    MeasureLinkages();
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
