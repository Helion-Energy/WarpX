/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "CircuitCoupler.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "Utils/TextMsg.H"
#include <AMReX_Print.H>
#include <cmath>

bool CircuitCoupler::SetTransactionCapability(WarpxCircuitTransactionApiV1 const* api)
{
    if(!m_transaction.Idle() || m_transaction_external || !m_rejection.Idle() || m_rejection_external)return false;
    if(api && !warpx::circuit::NativeCircuitTransaction::Supports(api))return false;
    m_transaction_api=api;return true;
}
bool CircuitCoupler::SupportsNativeTransaction() const
{
    return m_plugin && warpx::circuit::NativeCircuitTransaction::Supports(m_transaction_api);
}
bool CircuitCoupler::SnapshotRetainedNativeStep(ExternalVectorPotential& external)
{
    if(!SupportsNativeTransaction() || !m_params.device_affine || !m_transaction.Idle() ||
       m_transaction_external || !m_rejection.Idle() || m_rejection_external ||
       external.DeviceScales().start || external.DeviceScales().end)return false;
    auto saved=std::make_unique<RejectionSnapshot>();
    saved->interval=m_interval;saved->substep_count=m_substep_count;
    saved->step_dt=m_step_dt;saved->eps_interval=m_eps_interval;
    saved->lambda=m_lambda;saved->lambda_start=m_lambda_start;
    saved->eps_filt=m_eps_filt;saved->lambda_accepted=m_lambda_accepted;
    saved->have_lambda_accepted=m_have_lambda_accepted;saved->open_loop_step=m_open_loop_step;
    for(int f=0;f<external.nFields();++f) {
        RejectionSnapshot::Segment segment;segment.name=external.FieldName(f);
        segment.python=external.GetScaleSegment(f,segment.old_scale,segment.new_scale,
                                                segment.old_time,segment.new_time);
        // This first live caller admits circuit-owned fields only. Analytic
        // prescribed fields and all combined vacuum/event branches stay gated.
        bool coupled=false;for(auto const& coil:m_coils.coils())coupled=coupled||coil.field_name==segment.name;
        if(!coupled || !segment.python || !std::isfinite(segment.old_scale) ||
           !std::isfinite(segment.new_scale) || !std::isfinite(segment.old_time) ||
           !std::isfinite(segment.new_time))return false;
        saved->segments.push_back(std::move(segment));
    }
    if(!m_transaction.Snapshot(m_plugin.get(),m_transaction_api))return false;
    InvalidateNativeEndpointRate();
    m_transaction_snapshot=std::move(saved);m_transaction_external=&external;
    m_transaction_device_trials=-1;
    m_source_rebase_pending=false;m_source_endpoint_phase=0;return true;
}
bool CircuitCoupler::RetainedExternalMatches() const
{
    if(!m_transaction_external || !m_transaction_snapshot)return false;
    auto const& ext=*m_transaction_external;auto const& saved=*m_transaction_snapshot;
    if(ext.nFields()!=static_cast<int>(saved.segments.size()))return false;
    for(int f=0;f<ext.nFields();++f) {
        if(ext.FieldName(f)!=saved.segments[f].name || ext.UsesPythonScale(f)!=saved.segments[f].python)return false;
    }
    return true;
}
bool CircuitCoupler::RetainedNativeStepCancelable() const
{
    return RetainedExternalMatches() && m_transaction.CanCancel();
}
bool CircuitCoupler::RetainedNativeStepFinalizable() const
{
    return RetainedExternalMatches() && !m_source_rebase_pending && m_transaction_device_trials>=0 && m_transaction.CanFinalize() &&
        !m_transaction_external->DeviceScales().start && !m_transaction_external->DeviceScales().end;
}
bool CircuitCoupler::RetainedNativeEndpointReady() const
{
    return RetainedNativeStepFinalizable();
}
bool CircuitCoupler::CancelRetainedNativeStep()
{
    if(!RetainedNativeStepCancelable())return false;
    amrex::Gpu::synchronize();
    InvalidateNativeEndpointRate();
    auto& ext=*m_transaction_external;auto& saved=*m_transaction_snapshot;
    ext.ClearDeviceScaleSegments();m_device.reset();
    if(!m_transaction.Cancel())return false;
    for(auto const& segment:saved.segments) {
        ext.SetScale(segment.name,segment.old_scale,segment.new_scale,segment.old_time,segment.new_time);
    }
    m_interval=saved.interval;m_substep_count=saved.substep_count;
    m_step_dt=saved.step_dt;m_eps_interval=saved.eps_interval;
    m_lambda=std::move(saved.lambda);m_lambda_start=std::move(saved.lambda_start);
    m_eps_filt=std::move(saved.eps_filt);m_lambda_accepted=std::move(saved.lambda_accepted);
    m_have_lambda_accepted=saved.have_lambda_accepted;m_open_loop_step=saved.open_loop_step;
    m_transaction_snapshot.reset();m_transaction_external=nullptr;m_transaction_device_trials=-1;
    m_source_rebase_pending=false;m_source_endpoint_phase=0;
    return true;
}
bool CircuitCoupler::FinalizeRetainedNativeStep()
{
    if(!RetainedNativeStepFinalizable())return false;
    amrex::Gpu::synchronize();
    InvalidateNativeEndpointRate();
    if(!m_transaction.Finalize())return false;
    auto const trials=m_transaction_device_trials;
    m_transaction_snapshot.reset();m_transaction_external=nullptr;m_transaction_device_trials=-1;
    m_source_rebase_pending=false;m_source_endpoint_phase=0;
    amrex::Print()<<"Device circuit accepted: trials="<<trials<<" host_trial_calls=0\n";
    return true;
}
