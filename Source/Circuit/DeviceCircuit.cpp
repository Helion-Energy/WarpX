/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "DeviceCircuit.H"
#include "Utils/TextMsg.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace warpx::circuit
{
namespace
{
    // AMReX device assertions/Abort are disabled in release builds. These
    // correctness refusals must also terminate a release executable.
    AMREX_GPU_HOST_DEVICE void Require (bool condition)
    {
        if (condition) { return; }
        AMREX_IF_ON_DEVICE((
            printf("Device circuit rejected invalid arithmetic, map guard, or stage convergence\n");
        ))
#if defined(__CUDA_ARCH__)
        __trap();
#elif defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        amrex::Abort("Device circuit rejected invalid arithmetic, map guard, or stage convergence");
#endif
    }

    AMREX_GPU_HOST_DEVICE double AddProduct (double sum, double a, double b)
    {
        // Match a plugin compiled without FMA contraction, including guards.
        volatile double product = a*b;
        return sum + product;
    }

    template <class T>
    void Upload (amrex::Gpu::DeviceVector<T>& destination, T const* source, std::size_t count)
    {
        destination.resize(count);
        if (count) {
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, source, source+count, destination.begin());
        }
    }
    template <class T>
    void Upload (amrex::Gpu::DeviceVector<T>& destination, std::vector<T> const& source)
    {
        Upload(destination, source.data(), source.size());
    }
    bool Finite (double const* values, std::size_t count)
    {
        if (count && !values) { return false; }
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) { return false; }
        }
        return true;
    }
}

void DeviceCircuit::Prepare (WarpxCircuitAffineViewV1 const& p, double t0, double dt,
    double theta, double tolerance, double filter_weight, std::vector<int> const& port_fields,
    std::vector<double> const& start_scales, std::vector<double> const& fixed_stage_scales,
    std::vector<double> const& reference, std::vector<double> const& normalization,
    std::vector<double> const& filter_memory, std::vector<double> const& unit_response,
    std::vector<int> const& measured, std::vector<double> const& reference_currents)
{
    auto const np = port_fields.size(), nf = start_scales.size();
    auto const cap = static_cast<std::size_t>(std::numeric_limits<int>::max());
    bool const valid_packet = p.struct_bytes == sizeof(p)
        && p.scalar_kind == WARPX_CIRCUIT_AFFINE_F64
        && p.guard_kind == WARPX_CIRCUIT_AFFINE_SIGN_GUARD_V1 && p.reserved == 0
        && p.token != 0 && p.n_port == np && np > 0 && np <= cap && nf <= cap
        && p.n_guard <= cap/np && np <= cap/np && nf <= cap/np
        && p.t0_sim == t0 && p.t1_sim == t0+dt && p.t1_sim > p.t0_sim
        && std::isfinite(dt) && dt > 0. && theta >= 0.5 && theta <= 1.
        && std::isfinite(tolerance) && tolerance > 0. && tolerance < 1.
        && std::isfinite(filter_weight) && filter_weight > 0. && filter_weight <= 1.
        && std::isfinite(p.guard_relative_band) && p.guard_relative_band >= 0.
        && p.guard_absolute_band == 0.;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(valid_packet, "Invalid device circuit affine packet or interval");
    m_np = static_cast<int>(np); m_nf = static_cast<int>(nf); m_ng = static_cast<int>(p.n_guard);
    m_theta = theta; m_dt = dt; m_tolerance = tolerance; m_sigma = filter_weight;
    m_band = p.guard_relative_band;
    m_stage_offset = (t0+theta*dt)-t0; m_segment_dt = (t0+dt)-t0;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(reference.size() == np && normalization.size() == np
        && filter_memory.size() == np && measured.size() == np && reference_currents.size() == np
        && fixed_stage_scales.size() == nf && unit_response.size() == np*nf,
        "Device circuit preparation array lengths disagree");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Finite(p.p0,np) && Finite(p.g,np*np)
        && Finite(p.i_ref,np) && Finite(p.entry_current,np)
        && Finite(p.guard_v0,p.n_guard) && Finite(p.guard_g,p.n_guard*np)
        && (p.n_guard == 0 || p.guard_sign != nullptr), "Nonfinite device circuit coefficients");
    for (auto const* v : {&start_scales, &fixed_stage_scales, &reference, &normalization,
                         &filter_memory, &unit_response, &reference_currents}) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Finite(v->data(),v->size()),
            "Nonfinite device circuit preparation data");
    }
    std::vector<int> field_port(nf,-1);
    for (std::size_t c = 0; c < np; ++c) {
        auto const f = port_fields[c];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(f >= 0 && f < m_nf && field_port[f] == -1
            && p.i_ref[c] == reference_currents[c] && p.i_ref[c] != 0.
            && normalization[c] != 0. && (measured[c] == 0 || measured[c] == 1),
            "Invalid device circuit port mapping or normalization");
        double const entry = p.entry_current[c]/p.i_ref[c];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(entry)
            && std::abs(entry-start_scales[f]) <= tolerance
                *std::max({1., std::abs(entry), std::abs(start_scales[f])}),
            "Circuit plugin entry current disagrees with accepted external field scale");
        field_port[f] = static_cast<int>(c);
    }
    std::vector<int> signs(m_ng);
    for (int g = 0; g < m_ng; ++g) {
        signs[g] = p.guard_sign[g];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(signs[g] == ((p.guard_v0[g]>0.)-(p.guard_v0[g]<0.)),
            "Affine guard sign disagrees with its reference");
    }
    Upload(m_port_field,port_fields); Upload(m_field_port,field_port); Upload(m_measured,measured);
    Upload(m_sign,signs); Upload(m_p0,p.p0,np); Upload(m_g,p.g,np*np); Upload(m_iref,p.i_ref,np);
    Upload(m_guard0,p.guard_v0,p.n_guard); Upload(m_guard_g,p.guard_g,p.n_guard*np);
    Upload(m_start,start_scales); Upload(m_fixed,fixed_stage_scales); Upload(m_reference,reference);
    Upload(m_norm,normalization); Upload(m_memory,filter_memory); Upload(m_q,unit_response);
    m_end.resize(nf); m_eps.resize(np); m_response.resize(np); m_candidate.resize(np); m_defect.resize(1);
    m_trials = 0;
    ResetTrial();
}

void DeviceCircuit::ResetTrial ()
{
    auto* end = m_end.data(); auto const* start = m_start.data(); auto* defect = m_defect.data();
    amrex::ParallelFor(m_nf, [=] AMREX_GPU_DEVICE (int f) noexcept { end[f] = start[f]; });
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept { defect[0] = 1.; });
    ++m_trials;
}

void DeviceCircuit::Evaluate (amrex::Gpu::DeviceVector<double> const& total_linkage, bool force)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(total_linkage.size() == static_cast<std::size_t>(m_np),
        "Device circuit linkage length mismatch");
    int const np = m_np, nf = m_nf, ng = m_ng;
    double const interval = m_dt*m_theta, sigma = m_sigma, band = m_band, tolerance = m_tolerance;
    double const offset = m_stage_offset, segment_dt = m_segment_dt;
    auto const* total = total_linkage.data(); auto const* q = m_q.data();
    auto const* start = m_start.data(); auto* end = m_end.data();
    auto const* fixed = m_fixed.data(); auto const* field_port = m_field_port.data();
    auto const* port_field = m_port_field.data(); auto const* measured = m_measured.data();
    auto const* reference = m_reference.data(); auto const* norm = m_norm.data();
    auto const* memory = m_memory.data(); auto const* p0 = m_p0.data(); auto const* g = m_g.data();
    auto const* iref = m_iref.data(); auto const* g0 = m_guard0.data();
    auto const* gg = m_guard_g.data(); auto const* signs = m_sign.data();
    auto* eps = m_eps.data(); auto* response = m_response.data();
    auto* candidate = m_candidate.data(); auto* defect = m_defect.data();
    // Small dense circuit: ordered sums preserve the plugin's guard arithmetic.
    // A single device thread also avoids a host convergence/validity round trip.
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept {
        if (!force && defect[0] <= tolerance) { return; }
        for (int c = 0; c < np; ++c) {
            double value = total[c];
            for (int f = 0; f < nf; ++f) {
                double const scale = field_port[f] < 0 ? fixed[f]
                    : AddProduct(start[f], (end[f]-start[f])/segment_dt, offset);
                value = AddProduct(value, -scale, q[c*nf+f]);
            }
            response[c] = measured[c] ? value : 0.;
            double const raw = measured[c] ? (value-reference[c])/interval/norm[c] : 0.;
            eps[c] = measured[c] ? AddProduct(sigma*raw, 1.-sigma, memory[c]) : 0.;
            Require(amrex::Math::isfinite(eps[c]) && amrex::Math::isfinite(value));
        }
        for (int row = 0; row < ng; ++row) {
            double dv = 0.;
            for (int c = 0; c < np; ++c) { dv = AddProduct(dv,gg[row*np+c],eps[c]); }
            double const value = g0[row]+dv;
            Require(amrex::Math::isfinite(dv) && amrex::Math::isfinite(value)
                && ((value>0.)-(value<0.)) == signs[row]
                && ((g0[row] == 0. && dv == 0.)
                    || std::abs(value) > band*(std::abs(g0[row])+std::abs(dv))));
        }
        double maximum = 0.;
        for (int c = 0; c < np; ++c) {
            double value = p0[c];
            for (int j = 0; j < np; ++j) { value = AddProduct(value,g[c*np+j],eps[j]); }
            candidate[c] = value/iref[c];
            int const f = port_field[c];
            double const before = AddProduct(start[f],(end[f]-start[f])/segment_dt,offset);
            double const after = AddProduct(start[f],(candidate[c]-start[f])/segment_dt,offset);
            Require(amrex::Math::isfinite(after) && amrex::Math::isfinite(candidate[c]));
            maximum = amrex::max(maximum, std::abs(after-before)
                /amrex::max(1.,amrex::max(std::abs(before),std::abs(after))));
        }
        for (int c = 0; c < np; ++c) { end[port_field[c]] = candidate[c]; }
        defect[0] = maximum;
    });
}

void DeviceCircuit::RequireConverged ()
{
    auto const* defect = m_defect.data(); double const tolerance = m_tolerance;
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept {
        if (!(amrex::Math::isfinite(defect[0]) && defect[0] <= tolerance)) {
            printf("Device circuit convergence defect %.17g exceeds tolerance %.17g\n",
                   defect[0],tolerance);
            Require(false);
        }
    });
}

void DeviceCircuit::ReadAccepted (std::vector<double>& endpoint, std::vector<double>& emf,
                                  std::vector<double>& response) const
{
    endpoint.resize(m_nf); emf.resize(m_np); response.resize(m_np);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_end.begin(),m_end.end(),endpoint.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_eps.begin(),m_eps.end(),emf.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,m_response.begin(),m_response.end(),response.begin());
}
}
