/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DEVICE_CIRCUIT_H_
#define WARPX_DEVICE_CIRCUIT_H_
#include "ExternalCircuitAffine.h"
#include <AMReX_GpuContainers.H>
#include <vector>

namespace warpx::circuit
{
/** Immutable response copied at step preparation; all trial arithmetic and
 * convergence checks stay on device. Host reads occur only at acceptance.
 * A fixed iteration budget avoids per-iteration convergence transfers. */
class DeviceCircuit
{
public:
    void Prepare (WarpxCircuitAffineViewV1 const& packet, double t0, double dt,
                  double theta, double tolerance, double filter_weight,
                  std::vector<int> const& port_fields,
                  std::vector<double> const& start_scales,
                  std::vector<double> const& fixed_stage_scales,
                  std::vector<double> const& reference,
                  std::vector<double> const& normalization,
                  std::vector<double> const& filter_memory,
                  std::vector<double> const& unit_response,
                  std::vector<int> const& measured,
                  std::vector<double> const& reference_currents);
    void ResetTrial ();
    // Freeze the first converged candidate until ResetTrial, matching the host
    // loop. Acceptance remeasures the final fields with force=true.
    void Evaluate (amrex::Gpu::DeviceVector<double> const& total_linkage, bool force = false);
    void RequireConverged ();
    void ReadAccepted (std::vector<double>& endpoint, std::vector<double>& emf,
                       std::vector<double>& response) const;
    [[nodiscard]] double const* Start () const { return m_start.data(); }
    [[nodiscard]] double const* End () const { return m_end.data(); }
    [[nodiscard]] int Fields () const { return m_nf; }
    [[nodiscard]] unsigned long Trials () const { return m_trials; }
private:
    int m_np = 0, m_nf = 0, m_ng = 0;
    double m_theta = 0., m_dt = 0., m_tolerance = 0., m_sigma = 1., m_band = 0.;
    double m_stage_offset = 0., m_segment_dt = 0.;
    unsigned long m_trials = 0;
    amrex::Gpu::DeviceVector<int> m_port_field, m_field_port, m_measured, m_sign;
    amrex::Gpu::DeviceVector<double> m_p0, m_g, m_iref, m_guard0, m_guard_g;
    amrex::Gpu::DeviceVector<double> m_start, m_end, m_fixed, m_reference, m_norm, m_memory, m_q;
    amrex::Gpu::DeviceVector<double> m_eps, m_response, m_candidate, m_defect;
};
}
#endif
