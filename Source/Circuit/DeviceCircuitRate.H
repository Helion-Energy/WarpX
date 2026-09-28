/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_DEVICE_CIRCUIT_RATE_H_
#define WARPX_DEVICE_CIRCUIT_RATE_H_
#include "ExternalCircuitRate.h"
#include "ExternalCircuitImpulse.h"
#include "ExternalCircuitRateClock.h"
#include <AMReX_GpuContainers.H>
#include <string>
#include <cstdint>
#include <vector>

namespace warpx::circuit {
/** Fixed accepted-state circuit/probe Schur block. Geometry rows use the same
 * linkage convention as CircuitCoupler. Preparation owns completed device
 * copies; Apply performs no host vector reads, plugin calls or time advance.
 * Apply input must not alias either output vector. The caller must reject a
 * nonzero EMF low-pass time: this instantaneous packet has no filter state.
 * A supplied clock certificate binds the packet to its unique native lattice
 * point. Without it only the original unshifted synchronized scope is admitted.
 * The Schur conditioning test is a backward-error/resolvability guard, not a
 * bound on physical forward error. The caller must check uncondensed equations.
 * This class alone does not qualify the geometry or the field endpoint caller.
 */
class DeviceCircuitRate {
public:
    bool Prepare(WarpxCircuitRateViewV1 const& packet, double time,
        std::vector<int> const& port_fields,
        std::vector<double> const& accepted_scales,
        std::vector<double> const& reference_currents,
        std::vector<double> const& linkage_normalization,
        std::vector<int> const& measured,
        std::vector<double> const& boundary_response,
        std::vector<double> const& unit_response,
        std::vector<double> const& prescribed_rates, std::string& error,
        WarpxCircuitRateClockViewV1 const* clock = nullptr);
    /** Pure source response in impulse units. The provider has checked the
     * native clock and phase; the caller binds the original retained token.
     * No autonomous offset, prescribed drive or artificial time interval is
     * introduced. Shares only the geometric Schur algebra with rates. */
    bool PrepareImpulse(WarpxCircuitImpulseViewV1 const& packet,
        std::uint64_t transaction, double time, std::uint32_t phase,
        std::vector<int> const& port_fields,
        std::vector<double> const& accepted_scales,
        std::vector<double> const& reference_currents,
        std::vector<double> const& linkage_normalization,
        std::vector<int> const& measured,
        std::vector<double> const& boundary_response,
        std::vector<double> const& unit_response, std::string& error);
    void Apply(amrex::Gpu::DeviceVector<double> const& homogeneous_linkage_rate,
               bool affine);
    [[nodiscard]] bool Ready() const noexcept { return m_ready; }
    [[nodiscard]] int FieldCount() const noexcept { return m_nf; }
    [[nodiscard]] int PortCount() const noexcept { return m_np; }
    [[nodiscard]] amrex::Gpu::DeviceVector<double> const& FieldRates() const noexcept
    { return m_rates; }
    [[nodiscard]] amrex::Gpu::DeviceVector<double> const& Emf() const noexcept
    { return m_emf; }
private:
    bool PrepareResponse(double const* current, double const* offset,
        double const* response, double const* packet_reference,
        std::vector<int> const& port_fields,
        std::vector<double> const& accepted_scales,
        std::vector<double> const& reference_currents,
        std::vector<double> const& linkage_normalization,
        std::vector<int> const& measured,
        std::vector<double> const& boundary_response,
        std::vector<double> const& unit_response,
        std::vector<double> const& prescribed_rates, std::string& error);
    bool m_ready = false;
    int m_np = 0, m_nf = 0;
    amrex::Gpu::DeviceVector<double> m_offset, m_response, m_k, m_t, m_rates, m_emf;
};
} // namespace warpx::circuit
#endif
