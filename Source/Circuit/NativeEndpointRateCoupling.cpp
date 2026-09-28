/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "CircuitCoupler.H"

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
    bool ValidRateApi (WarpxCircuitRateApiV1 const* api)
    {
        return api && api->struct_bytes == sizeof(*api)
            && api->api_version == WARPX_CIRCUIT_RATE_API_V1
            && api->capabilities == WARPX_CIRCUIT_RATE_FROZEN_ACCEPTED_COEFFICIENTS_V1
            && api->prepare && api->release;
    }

    bool ValidClockApi (WarpxCircuitRateClockApiV1 const* api)
    {
        return api && api->struct_bytes == sizeof(*api)
            && api->api_version == WARPX_CIRCUIT_RATE_CLOCK_API_V1
            && api->capabilities == WARPX_CIRCUIT_RATE_CLOCK_NATIVE_OPERAND_ENCLOSURE_V1
            && api->query;
    }

    bool AllRanks (bool local, std::string& error)
    {
        int valid = local ? 1 : 0;
        amrex::ParallelDescriptor::ReduceIntMin(valid);
        if (!valid && local) { error = "Native endpoint rate preparation failed on another rank"; }
        return valid != 0;
    }

    struct RateLease
    {
        WarpxCircuitRateApiV1 const* api;
        ExternalCircuit* instance;
        std::uint64_t token = 0;
        ~RateLease () { if (token) { api->release(instance, token); } }
    };
}

bool CircuitCoupler::SetNativeEndpointRateCapabilities (
    WarpxCircuitRateApiV1 const* rate, WarpxCircuitRateClockApiV1 const* clock)
{
    if (m_endpoint_capabilities_attached || !m_rejection.Idle() || m_rejection_external
        || (rate && !ValidRateApi(rate)) || (clock && !ValidClockApi(clock))) { return false; }
    m_rate_api = rate;
    m_rate_clock_api = clock;
    m_endpoint_capabilities_attached = true;
    InvalidateNativeEndpointRate();
    return true;
}

bool CircuitCoupler::SupportsNativeEndpointRate () const
{
#if (!defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)) || \
    (defined(AMREX_USE_GPU) && !defined(AMREX_USE_CUDA) && !defined(AMREX_USE_HIP))
    return false;
#else
    return m_plugin && ValidRateApi(m_rate_api) && ValidClockApi(m_rate_clock_api)
        && sizeof(amrex::Real) == sizeof(double) && m_params.eps_lowpass_tau == 0.
        && m_coils.size() > 0
        && std::all_of(m_probes.begin(), m_probes.end(), [](auto kind) {
            return kind == warpx::circuit::ProbeKind::disk
                || kind == warpx::circuit::ProbeKind::none;
        });
#endif
}

void CircuitCoupler::InvalidateNativeEndpointRate ()
{
    InvalidateNativeSourceImpulse();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_endpoint_rate_generation != std::numeric_limits<std::uint64_t>::max(),
        "Native endpoint circuit rate generation exhausted");
    ++m_endpoint_rate_generation;
    // Keep immutable H and allocated maps. Any outstanding device work must be
    // completed by the caller before destroying/replacing the owning coupler.
    m_endpoint_rate_ready = false;
    m_endpoint_rate_applied = false;
}

bool CircuitCoupler::NativeEndpointGeometryMatches (amrex::MultiFab const& bz) const
{
    if (m_endpoint_boundary_response.empty()) { return false; }
    auto const& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    auto const& external = *simulation.get_pointer_HybridPICModel()->m_external_vector_potential;
    if (simulation.finestLevel() != 0 || geometry.Domain() != m_endpoint_domain
        || geometry.Coord() != m_endpoint_coord || bz.nComp() < 1
        || bz.boxArray() != m_endpoint_ba || bz.DistributionMap() != m_endpoint_dm
        || external.nFields() != static_cast<int>(m_endpoint_field_names.size())) { return false; }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (geometry.ProbLo(d) != m_endpoint_prob_lo[d]
            || geometry.ProbHi(d) != m_endpoint_prob_hi[d]
            || geometry.isPeriodic(d) != m_endpoint_periodic[d]) { return false; }
    }
    for (int f = 0; f < external.nFields(); ++f) {
        if (external.FieldName(f) != m_endpoint_field_names[f]) { return false; }
    }
    return true;
}

bool CircuitCoupler::SetNativeEndpointBoundaryResponse (
    std::vector<double> const& response, std::string& error)
{
    InvalidateNativeEndpointRate();
    auto fail = [&](char const* message) { error = message; return false; };
    auto const count = static_cast<std::size_t>(m_coils.size());
    if (m_darwin_external_count <= 0 || count == 0
        || count > std::numeric_limits<std::size_t>::max()/m_darwin_external_count
        || response.size() != count*m_darwin_external_count
        || m_darwin_unit_response.size() != response.size()
        || !std::all_of(response.begin(), response.end(), [](double x) { return std::isfinite(x); })) {
        return fail("Native endpoint H must match the configured finite coil-by-field Q layout");
    }
    auto const& simulation = WarpX::GetInstance();
    auto const& geometry = simulation.Geom(0);
    auto const& external = *simulation.get_pointer_HybridPICModel()->m_external_vector_potential;
    if (simulation.finestLevel() != 0 || external.nFields() != m_darwin_external_count) {
        return fail("Native endpoint geometry requires a configured fixed level-zero field set");
    }
    auto const& bz = *simulation.m_fields.get(warpx::fields::FieldType::Bfield_fp,
        ablastr::fields::Direction{2}, 0);
    m_endpoint_boundary_response = response;
    m_endpoint_domain = geometry.Domain();
    m_endpoint_coord = geometry.Coord();
    m_endpoint_ba = bz.boxArray();
    m_endpoint_dm = bz.DistributionMap();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        m_endpoint_prob_lo[d] = geometry.ProbLo(d);
        m_endpoint_prob_hi[d] = geometry.ProbHi(d);
        m_endpoint_periodic[d] = geometry.isPeriodic(d);
    }
    m_endpoint_field_names.clear();
    for (int f = 0; f < external.nFields(); ++f) {
        m_endpoint_field_names.push_back(external.FieldName(f));
    }
    error.clear();
    return true;
}

bool CircuitCoupler::PrepareNativeEndpointRate (amrex::Real time,
    std::vector<double> const& prescribed_field_rates, std::string& error)
{
    return PrepareNativeEndpointRate(time, prescribed_field_rates, error, nullptr);
}

bool CircuitCoupler::PrepareNativeEndpointRate (amrex::Real time,
    std::vector<double> const& prescribed_field_rates, std::string& error,
    std::vector<double> const* boundary_response_override)
{
    BL_PROFILE("CircuitCoupler::PrepareNativeEndpointRate");
    InvalidateNativeEndpointRate();
    error.clear();
    bool valid = SupportsNativeEndpointRate() && std::isfinite(time);
    if (!valid) { error = "Native endpoint rate requires valid rate/clock APIs, disk/none, and no lowpass"; }
    if (!AllRanks(valid, error)) { return false; }
    auto& simulation = WarpX::GetInstance();
    auto const& external = *simulation.get_pointer_HybridPICModel()->m_external_vector_potential;
    auto const& bz = *simulation.m_fields.get(warpx::fields::FieldType::Bfield_fp,
        ablastr::fields::Direction{2}, 0);
    valid = NativeEndpointGeometryMatches(bz)
        && !external.DeviceScales().start && !external.DeviceScales().end
        && prescribed_field_rates.size() == static_cast<std::size_t>(external.nFields());
    if (!valid) { error = "Native endpoint rate geometry changed or interval device views are active"; }
    if (!AllRanks(valid, error)) { return false; }
    // Every rank must follow the same optional-override branch before its
    // validation collective. A rank-local null pointer must fail collectively
    // rather than shift the following native prepare/clock collectives.
    int override_ranks = boundary_response_override ? 1 : 0;
    amrex::ParallelDescriptor::ReduceIntSum(override_ranks);
    if (override_ranks != 0 && override_ranks != amrex::ParallelDescriptor::NProcs()) {
        error = "Native endpoint response override presence differs across MPI ranks";
        return false;
    }
    auto const& boundary_response = boundary_response_override
        ? *boundary_response_override : m_endpoint_boundary_response;
    if (boundary_response_override) {
        valid = boundary_response.size() == m_endpoint_boundary_response.size()
            && boundary_response.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())
            && std::all_of(boundary_response.begin(), boundary_response.end(),
                           [](double value) { return std::isfinite(value); });
        if (!valid) { error = "Native endpoint response override must match the finite coil-by-field H layout"; }
        if (!AllRanks(valid, error)) { return false; }
        // H is a replicated, small coil-by-field geometric response. Different
        // finite copies would give each rank a different circuit action even
        // though every local validation passed. Validate the count first so the
        // following broadcast has the same extent on every rank.
        int count_min = static_cast<int>(boundary_response.size());
        int count_max = count_min;
        amrex::ParallelDescriptor::ReduceIntMin(count_min);
        amrex::ParallelDescriptor::ReduceIntMax(count_max);
        if (count_min != count_max) {
            error = "Native endpoint response override size differs across MPI ranks";
            return false;
        }
        std::vector<double> reference(boundary_response);
        amrex::ParallelDescriptor::Bcast(reference.data(), count_min,
            amrex::ParallelDescriptor::IOProcessorNumber());
        valid = std::equal(reference.begin(), reference.end(), boundary_response.begin());
        if (!valid) { error = "Native endpoint response override values differ across MPI ranks"; }
        if (!AllRanks(valid, error)) { return false; }
    }
    std::vector<int> fields(m_coils.size(), -1), measured(m_coils.size());
    std::vector<double> iref(m_coils.size()), normalization(m_coils.size());
    std::vector<double> scales(external.nFields());
    for (int f = 0; f < external.nFields(); ++f) { scales[f] = external.TimeScale(f, time); }
    for (int c = 0; c < m_coils.size(); ++c) {
        auto const& coil = m_coils.coil(c);
        iref[c] = coil.I_ref;
        normalization[c] = coil.I_ref*coil.n_turns;
        measured[c] = m_probes[c] == warpx::circuit::ProbeKind::disk;
        for (int f = 0; f < external.nFields(); ++f) {
            if (external.FieldName(f) == coil.field_name) { fields[c] = f; break; }
        }
    }
    WarpxCircuitRateViewV1 packet{};
    packet.struct_bytes = sizeof(packet);
    char message[512]{};
    auto const status = m_rate_api->prepare(m_plugin.get(), time, &packet, message, sizeof(message));
    RateLease lease{m_rate_api, m_plugin.get(), packet.token};
    message[sizeof(message)-1] = '\0';
    valid = status == WARPX_RATE_OK;
    if (!valid) { error = "Native endpoint rate provider: " + std::to_string(status) + " " + message; }
    if (!AllRanks(valid, error)) { return false; }
    WarpxCircuitRateClockViewV1 clock{};
    clock.struct_bytes = sizeof(clock);
    message[0] = '\0';
    auto const clock_status = m_rate_clock_api->query(m_plugin.get(), packet.token, time,
                                                    &clock, message, sizeof(message));
    message[sizeof(message)-1] = '\0';
    valid = clock_status == WARPX_RATE_CLOCK_OK;
    if (!valid) { error = "Native endpoint rate clock: " + std::to_string(clock_status) + " " + message; }
    if (!AllRanks(valid, error)) { return false; }
    if (!m_endpoint_rate) { m_endpoint_rate = std::make_unique<warpx::circuit::DeviceCircuitRate>(); }
    std::vector<double> q(m_darwin_unit_response.begin(), m_darwin_unit_response.end());
    valid = m_endpoint_rate->Prepare(packet, time, fields, scales, iref, normalization,
        measured, boundary_response, q, prescribed_field_rates, error, &clock);
    if (!AllRanks(valid, error)) { return false; }
    // Warm exactly the disk/none measurement layout before any operator call.
    // BuildPack and the packet copies complete before the lease is released.
    m_a_ext_scratch.assign(m_coils.size(), {nullptr, nullptr, nullptr});
    warpx::circuit::VectorFieldPtrs const unused{nullptr, nullptr, nullptr};
    m_batch.BuildPack(m_coils, m_probes, m_probe_exclusion, m_a_ext_scratch, &bz, unused);
    m_endpoint_rate_ready = true;
    return true;
}

void CircuitCoupler::ApplyNativeEndpointRate (amrex::MultiFab const& homogeneous_Bdot_z,
                                             bool affine)
{
    BL_PROFILE("CircuitCoupler::ApplyNativeEndpointRate");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_endpoint_rate_ready && m_endpoint_rate
        && NativeEndpointGeometryMatches(homogeneous_Bdot_z),
        "Native endpoint circuit rate is unprepared or its fixed geometry/layout changed");
    warpx::circuit::VectorFieldPtrs const unused{nullptr, nullptr, nullptr};
    auto const& linkage = m_batch.MeasureDevice(m_coils, m_probes, m_probe_exclusion,
        m_a_ext_scratch, &homogeneous_Bdot_z, unused);
    m_endpoint_rate->Apply(linkage, affine);
    m_endpoint_rate_applied = true;
}

amrex::Gpu::DeviceVector<double> const& CircuitCoupler::EndpointFieldRates () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_endpoint_rate_ready && m_endpoint_rate_applied,
        "Native endpoint field rates require a current prepared and applied map");
    return m_endpoint_rate->FieldRates();
}

amrex::Gpu::DeviceVector<double> const& CircuitCoupler::EndpointEmf () const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_endpoint_rate_ready && m_endpoint_rate_applied,
        "Native endpoint EMF requires a current prepared and applied map");
    return m_endpoint_rate->Emf();
}
