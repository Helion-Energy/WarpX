/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "EulerianSourceDiagonal.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace warpx::thermal {
namespace {
using amrex::Real;
using MF = amrex::MultiFab;
ThermalSourceOptions
LocalOptions (ThermalSourceOptions o) {
    o.viscosity = ViscousSourceKind::None;
    o.hyperresistive_work = false;
    o.external_sink = false;
    return o;
}
bool
Collective (bool ok) {
    int result = ok;
    amrex::ParallelDescriptor::ReduceIntMin(result);
    return result != 0;
}
bool
Aliases (const MF& a, const MF& b) {
    using Range = std::pair<std::uintptr_t, std::uintptr_t>;
    std::vector<Range> ranges;
    for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto lo = reinterpret_cast<std::uintptr_t>(a[mfi].dataPtr());
        ranges.emplace_back(lo, lo + a[mfi].box().numPts() * a.nComp() *
                                         sizeof(Real));
    }
    std::sort(ranges.begin(), ranges.end());
    for (amrex::MFIter mfi(b); mfi.isValid(); ++mfi) {
        auto lo = reinterpret_cast<std::uintptr_t>(b[mfi].dataPtr());
        auto hi = lo + b[mfi].box().numPts() * b.nComp() * sizeof(Real);
        auto next =
            std::lower_bound(ranges.begin(), ranges.end(), Range{hi, 0});
        if (next != ranges.begin() && std::prev(next)->second > lo)
            return true;
    }
    return false;
}
void
ProbeTemperatures (const MF& input, MF& plus, MF& minus, MF& span,
                   SourceDiagonalOptions o) {
    for (amrex::MFIter mfi(plus); mfi.isValid(); ++mfi) {
        auto t = input.const_array(mfi);
        auto p = plus.array(mfi), m = minus.array(mfi), delta = span.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real const value = t(i, j, k);
            Real const h =
                value > 0
                    ? amrex::min(.25 * value,
                                 amrex::max(std::abs(value) *
                                                o.relative_temperature_step,
                                            o.absolute_temperature_step_kelvin))
                    : 0;
            p(i, j, k) = value + h;
            m(i, j, k) = value - h;
            delta(i, j, k) = p(i, j, k) - m(i, j, k);
        });
    }
}
void
NodeSlopes (const MF& plus, const MF& minus, const MF& span, const MF& tp,
            const MF& tm, MF& derivative, MF& masks, ThermalSourceOptions o) {
    for (amrex::MFIter mfi(derivative); mfi.isValid(); ++mfi) {
        auto p = plus.const_array(mfi), m = minus.const_array(mfi),
             delta = span.const_array(mfi), high = tp.const_array(mfi),
             low = tm.const_array(mfi);
        auto out = derivative.array(mfi), mask = masks.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real const lo = low(i, j, k) * PhysConst::kb / PhysConst::q_e,
                       hi = high(i, j, k) * PhysConst::kb / PhysConst::q_e;
            bool const redir = o.joule && o.redirect_joule &&
                               lo <= o.redirect_temperature_ev &&
                               hi >= o.redirect_temperature_ev;
            bool const rate_floor = o.relaxation && lo <= 1.e-3 && hi >= 1.e-3;
            mask(i, j, k, 0) = redir;
            mask(i, j, k, 1) = rate_floor;
            Real const width = delta(i, j, k);
            out(i, j, k, 0) = width > 0 && !redir
                                  ? (p(i, j, k, SourceComponent::Joule) -
                                     m(i, j, k, SourceComponent::Joule)) /
                                        width
                                  : 0;
            out(i, j, k, 1) = width > 0 && !rate_floor
                                  ? (p(i, j, k, SourceComponent::Relaxation) -
                                     m(i, j, k, SourceComponent::Relaxation)) /
                                        width
                                  : 0;
            out(i, j, k, 2) = out(i, j, k, 0) + out(i, j, k, 1);
        });
    }
}
// Count occurrences of one physical cell among a node's two CC neighbors,
// after the exact even/periodic temperature images used by the moment map.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
ImageMultiplicity (int node, int cell, int high, bool periodic) {
    int count = 0;
    for (int side = 0; side < 2; ++side) {
        int image = node - side;
        if (periodic) {
            if (image < 0)
                image += high + 1;
            if (image > high)
                image -= high + 1;
        } else
            image = amrex::max(0, amrex::min(high, image));
        count += image == cell;
    }
    return count;
}
void
CellDiagonal (const MF& slopes, const MF& capacity, MF& raw, MF& pc,
              const amrex::Geometry& g, Real gamma) {
    auto high = g.Domain().bigEnd();
    auto periodic = g.isPeriodicArray();
    Real const factor = (gamma - 1) / PhysConst::kb;
    for (amrex::MFIter mfi(raw); mfi.isValid(); ++mfi) {
        auto s = slopes.const_array(mfi), n = capacity.const_array(mfi);
        auto out = raw.array(mfi), dpc = pc.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real const radial = (i + .25) / (2 * i + 1.);
            Real diagonal = 0;
            for (int di = 0; di < 2; ++di)
                for (int dj = 0; dj < 2; ++dj) {
                    Real const restriction =
                        (di == 0 ? radial : 1 - radial) * .5;
                    Real const interpolation =
                        .25 *
                        ImageMultiplicity(i + di, i, high[0], periodic[0]) *
                        ImageMultiplicity(j + dj, j, high[1], periodic[1]);
                    diagonal +=
                        restriction * s(i + di, j + dj, k, 2) * interpolation;
                }
            out(i, j, k) = factor * diagonal / n(i, j, k);
            dpc(i, j, k) = amrex::min(Real(0), out(i, j, k));
        });
    }
}
} // namespace
EulerianSourceDiagonal::EulerianSourceDiagonal (
    const amrex::Geometry& geometry, const MF& nodes,
    ThermalSourceOptions source_options, SourceDiagonalOptions options)
    : m_geometry(geometry), m_nodes(nodes.boxArray()),
      m_cells(amrex::convert(m_nodes, amrex::IntVect::TheCellVector())),
      m_distribution(nodes.DistributionMap()), m_source_options(source_options),
      m_options(options),
      m_source(geometry, nodes, LocalOptions(source_options)),
      m_plus_temperature(m_nodes, m_distribution, 1, 0),
      m_minus_temperature(m_nodes, m_distribution, 1, 0),
      m_span(m_nodes, m_distribution, 1, 0),
      m_plus_rates(m_nodes, m_distribution, SourceComponent::Count, 0),
      m_minus_rates(m_nodes, m_distribution, SourceComponent::Count, 0),
      m_derivative(m_nodes, m_distribution, 3, 0),
      m_kinks(m_nodes, m_distribution, 2, 0),
      m_raw(m_cells, m_distribution, 1, 0),
      m_pc(m_cells, m_distribution, 1, 0) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 1 && geometry.ProbLo(0) == 0 &&
            geometry.Domain().smallEnd() == amrex::IntVect::TheZeroVector(),
        "EulerianSourceDiagonal needs the qualified zero-based physical RZ "
        "moment map");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(options.relative_temperature_step) &&
            options.relative_temperature_step > 0 &&
            options.relative_temperature_step < .25 &&
            std::isfinite(options.absolute_temperature_step_kelvin) &&
            options.absolute_temperature_step_kelvin > 0,
        "EulerianSourceDiagonal invalid finite-difference step");
}
bool
EulerianSourceDiagonal::Freeze (const ThermalSourceState& state,
                                const MF& capacity, Real gamma) {
    m_valid = false;
    bool ok = state.temperature_kelvin &&
              state.temperature_kelvin->boxArray() == m_nodes &&
              state.temperature_kelvin->DistributionMap() == m_distribution &&
              state.temperature_kelvin->nComp() >= 1 &&
              !state.temperature_kelvin->hasEBFabFactory() &&
              capacity.boxArray() == m_cells &&
              capacity.DistributionMap() == m_distribution &&
              capacity.nComp() == 1 && !capacity.hasEBFabFactory() &&
              std::isfinite(gamma) && gamma > 1;
    if (!Collective(ok))
        return false;
    std::vector<const MF*> inputs{&capacity,
                                  state.raw_charge,
                                  state.temperature_kelvin,
                                  state.additive_resistivity,
                                  state.stopping_impulse,
                                  state.viscous_strain_power};
    for (auto const* vector : {&state.plasma_current, &state.magnetic_field,
                               &state.viscous_electric, &state.hyper_electric})
        for (auto* input : *vector)
            inputs.push_back(input);
    for (auto const& species : state.species) {
        inputs.push_back(species.raw_charge);
        inputs.push_back(species.ion_temperature_ev);
        for (auto* input : species.raw_current)
            inputs.push_back(input);
    }
    for (auto const* input : inputs)
        if (input)
            for (auto const* output : {&m_raw, &m_pc, &m_derivative, &m_kinks})
                ok = ok && !Aliases(*input, *output);
    if (!Collective(ok))
        return false;
    if (!capacity.is_finite(0, 1, 0) || capacity.min(0) <= 0 ||
        !state.temperature_kelvin->is_finite(0, 1, 0))
        return false;
    int const components =
        std::max(1, int(state.species.size()) * IonSourceComponent::Count);
    if (!m_ion_rates || m_ion_rates->nComp() != components)
        m_ion_rates =
            std::make_unique<MF>(m_nodes, m_distribution, components, 0);
    ProbeTemperatures(*state.temperature_kelvin, m_plus_temperature,
                      m_minus_temperature, m_span, m_options);
    auto local = state;
    local.viscous_electric = {};
    local.hyper_electric = {};
    local.viscous_strain_power = nullptr;
    local.stopping_impulse = nullptr;
    local.temperature_kelvin = &m_plus_temperature;
    if (!m_source.Evaluate(local, m_plus_rates, *m_ion_rates))
        return false;
    local.temperature_kelvin = &m_minus_temperature;
    if (!m_source.Evaluate(local, m_minus_rates, *m_ion_rates))
        return false;
    NodeSlopes(m_plus_rates, m_minus_rates, m_span, m_plus_temperature,
               m_minus_temperature, m_derivative, m_kinks, m_source_options);
    if (!m_derivative.is_finite(0, 3, 0))
        return false;
    m_derivative.OverrideSync(m_geometry.periodicity());
    CellDiagonal(m_derivative, capacity, m_raw, m_pc, m_geometry, gamma);
    m_valid = m_raw.is_finite(0, 1, 0) && m_pc.is_finite(0, 1, 0);
    return m_valid;
}
const MF&
EulerianSourceDiagonal::NodalDerivative () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_derivative;
}
const MF&
EulerianSourceDiagonal::KinkMask () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_kinks;
}
const MF&
EulerianSourceDiagonal::RawDiagonal () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_raw;
}
const MF&
EulerianSourceDiagonal::PCDiagonal () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_pc;
}
} // namespace warpx::thermal
