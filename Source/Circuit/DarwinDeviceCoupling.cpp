/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "CircuitCoupler.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"
#include <AMReX_Print.H>
#include <algorithm>
#include <cmath>

using warpx::fields::FieldType;

void CircuitCoupler::PrepareDarwinDeviceStep (amrex::Real t0, amrex::Real dt,
                                             amrex::Real theta, amrex::Real tolerance)
{
    BL_PROFILE("CircuitCoupler::PrepareDarwinDeviceStep");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_params.device_affine && m_plugin && m_affine_api
        && m_affine_api->prepare && m_affine_api->release,
        "Device circuit preparation requires the affine plugin capability");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(sizeof(amrex::Real) == sizeof(double),
        "Device circuit affine packets require double precision");
    auto& ext = *WarpX::GetInstance().get_pointer_HybridPICModel()->m_external_vector_potential;
    std::vector<int> fields(m_coils.size()), measured(m_coils.size());
    std::vector<double> reference(m_coils.size()), normalization(m_coils.size());
    std::vector<double> memory(m_coils.size()), iref(m_coils.size());
    std::vector<double> start(ext.nFields()), fixed(ext.nFields());
    for (int f = 0; f < ext.nFields(); ++f) {
        start[f] = ext.TimeScale(f,t0); fixed[f] = ext.TimeScale(f,t0+theta*dt);
    }
    for (int c = 0; c < m_coils.size(); ++c) {
        auto const& coil = m_coils.coil(c);
        auto const entry = m_lambda_start.find(coil.name);
        reference[c] = entry == m_lambda_start.end() ? 0. : entry->second;
        auto const old = m_eps_filt.find(coil.name);
        memory[c] = old == m_eps_filt.end() ? 0. : old->second;
        normalization[c] = coil.I_ref*coil.n_turns;
        iref[c] = coil.I_ref; measured[c] = m_probes[c] != warpx::circuit::ProbeKind::none;
        fields[c] = -1;
        for (int f = 0; f < ext.nFields(); ++f) {
            if (ext.FieldName(f) == coil.field_name) { fields[c] = f; break; }
        }
    }
    WarpxCircuitAffineViewV1 packet{}; packet.struct_bytes = sizeof(packet);
    char error[512]{};
    auto const status = m_affine_api->prepare(m_plugin.get(),t0,t0+dt,&packet,error,sizeof(error));
    error[sizeof(error)-1] = '\0';
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(status == WARPX_AFFINE_OK,
        "Device circuit interval is unsupported; no host residual fallback: "
        + std::to_string(status) + " " + std::string(error));
    if (!m_device) { m_device = std::make_unique<warpx::circuit::DeviceCircuit>(); }
    double const sigma = m_params.eps_lowpass_tau > 0.
        ? dt/(dt+m_params.eps_lowpass_tau) : 1.;
    std::vector<double> q(m_darwin_unit_response.begin(),m_darwin_unit_response.end());
    m_device->Prepare(packet,t0,dt,theta,tolerance,sigma,fields,start,fixed,reference,
                      normalization,memory,q,measured,iref);
    // Prepare uses completed copies; no borrowed packet memory survives release.
    m_affine_api->release(m_plugin.get(),packet.token);
    ext.SetDeviceScaleSegments(m_device->Start(),m_device->End(),t0,(t0+dt)-t0,fields);
    amrex::Print() << "Device circuit prepared: ports=" << m_coils.size()
                   << " fixed_iterations=" << m_params.device_iterations << "\n";
}

void CircuitCoupler::ResetDarwinDeviceTrial ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_device != nullptr, "Device circuit step is not prepared");
    m_device->ResetTrial();
}

void CircuitCoupler::AdvanceDarwinDeviceTrial (bool force)
{
    BL_PROFILE("CircuitCoupler::AdvanceDarwinDeviceTrial");
    auto const* bz = WarpX::GetInstance().m_fields.get(FieldType::Bfield_fp,
        ablastr::fields::Direction{2},0);
    warpx::circuit::VectorFieldPtrs const unused{nullptr,nullptr,nullptr};
    auto const& total = m_batch.MeasureDevice(m_coils,m_probes,m_probe_exclusion,
        m_a_ext_scratch,bz,unused);
    m_device->Evaluate(total,force);
}

void CircuitCoupler::RequireDarwinDeviceConvergence () { m_device->RequireConverged(); }

void CircuitCoupler::CommitDarwinDeviceStep (amrex::Real t0, amrex::Real dt,
                                            amrex::Real theta, amrex::Real tolerance)
{
    BL_PROFILE("CircuitCoupler::CommitDarwinDeviceStep");
    // Re-measure the final consistent fields before the sole accepting call.
    AdvanceDarwinDeviceTrial(true);
    RequireDarwinDeviceConvergence();
    std::vector<double> candidate, emf, response;
    m_device->ReadAccepted(candidate,emf,response);
    auto& ext = *WarpX::GetInstance().get_pointer_HybridPICModel()->m_external_vector_potential;
    ext.ClearDeviceScaleSegments();
    std::vector<amrex::Real> input(emf.begin(),emf.end()), exact;
    m_plugin->AdvanceInterval(t0,t0+dt,input,true,exact);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(exact.size() == static_cast<std::size_t>(m_coils.size()),
        "Circuit accepted scale count mismatch");
    for (int c = 0; c < m_coils.size(); ++c) {
        auto const& coil = m_coils.coil(c);
        int field = -1;
        for (int f = 0; f < ext.nFields(); ++f) {
            if (ext.FieldName(f) == coil.field_name) { field = f; break; }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field >= 0 && std::isfinite(exact[c])
            && std::abs(exact[c]-candidate[field]) <= tolerance
                *std::max({1.,std::abs(double(exact[c])),std::abs(candidate[field])}),
            "Exact circuit acceptance disagrees with the device affine candidate");
        auto const entry = ext.TimeScale(field,t0);
        ext.SetScale(coil.field_name,entry,exact[c],t0,t0+dt);
        if (m_probes[c] != warpx::circuit::ProbeKind::none) {
            m_lambda[coil.name] = response[c];
            if (m_params.eps_lowpass_tau > 0.) { m_eps_filt[coil.name] = input[c]; }
        }
    }
    m_lambda_accepted = m_lambda; m_have_lambda_accepted = true;
    m_interval.t0 = t0; m_interval.t1 = t0+dt; m_eps_interval = theta*dt;
    FinishStep();
    amrex::Print() << "Device circuit accepted: trials=" << m_device->Trials()
                   << " host_trial_calls=0\n";
}
