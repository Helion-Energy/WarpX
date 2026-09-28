/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "AcceptedIonExchange.H"
#include "AcceptedIonExchange_K.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "ablastr/coarsen/sample.H"
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
namespace warpx::thermal {
namespace {
using amrex::Real;
using P = amrex::ParticleReal;
using MF = amrex::MultiFab;
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
    for (amrex::MFIter first(a); first.isValid(); ++first) {
        auto lo = reinterpret_cast<std::uintptr_t>(a[first].dataPtr());
        ranges.emplace_back(lo, lo + a[first].box().numPts() * a.nComp() *
                                         sizeof(Real));
    }
    std::sort(ranges.begin(), ranges.end());
    for (amrex::MFIter second(b); second.isValid(); ++second) {
        auto begin = reinterpret_cast<std::uintptr_t>(b[second].dataPtr());
        auto end = begin + b[second].box().numPts() * b.nComp() * sizeof(Real);
        auto next =
            std::lower_bound(ranges.begin(), ranges.end(), Range{end, 0});
        if (next != ranges.begin() && std::prev(next)->second > begin)
            return true;
    }
    return false;
}
Real
Integral (const MF& f, int component, const amrex::Geometry& g,
          const amrex::iMultiFab& owner) {
    auto volume = MakeQdsmcVolumeElement(g, f.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<Real> data(op);
    using Tuple = decltype(data)::Type;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a = f.const_array(mfi);
        auto mask = owner.const_array(mfi);
        op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    return {mask(i, j, k)
                                ? volume(i, j, k) * a(i, j, k, component)
                                : Real(0)};
                });
    }
    Real v = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(v);
    return v;
}
void
PrepareCells (MF& coefficient, const MF& rho, const MF& te, const MF& rates,
              const std::array<const MF*, 3>& drift, const MF* ti,
              std::size_t species, IonExchangeOptions options,
              amrex::ParserExecutor<4> oracle, bool reference) {
    amrex::GpuArray<int, 3> node{1, 1, 1}, cc{0, 0, 0}, one{1, 1, 1};
    for (int d = AMREX_SPACEDIM; d < 3; ++d)
        node[d] = 0;
    int const offset = species * IonSourceComponent::Count;
    for (amrex::MFIter mfi(coefficient); mfi.isValid(); ++mfi) {
        auto out = coefficient.array(mfi);
        auto r = rho.const_array(mfi), t = te.const_array(mfi),
             q = rates.const_array(mfi);
        std::array<amrex::Array4<const Real>, 3> v{drift[0]->const_array(mfi),
                                                   drift[1]->const_array(mfi),
                                                   drift[2]->const_array(mfi)};
        auto old = ti ? ti->const_array(mfi) : amrex::Array4<const Real>{};
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            auto avg = [&] (amrex::Array4<const Real> a, int n = 0) {
                return ablastr::coarsen::sample::Interp(a, node, cc, one, i, j,
                                                        k, n);
            };
            Real const density = avg(r);
            if (density <= PhysConst::q_e * options.raw_density_floor)
                return;
            out(i, j, k, IonExchangeCoefficient::TemperatureKelvin) = avg(t);
            if (options.relaxation) {
                out(i, j, k, IonExchangeCoefficient::Nu) =
                    avg(q, offset + IonSourceComponent::CollisionFrequency);
                for (int d = 0; d < 3; ++d)
                    out(i, j, k, IonExchangeCoefficient::DriftR + d) =
                        avg(v[d]);
                if (reference)
                    out(i, j, k, IonExchangeCoefficient::ReferenceNu) = oracle(
                        density,
                        amrex::max(avg(t) * PhysConst::kb / PhysConst::q_e,
                                   Real(1e-3)),
                        old(i, j, k), options.time);
            }
            if (options.redirect)
                out(i, j, k, IonExchangeCoefficient::RedirectEnergy) =
                    options.dt *
                    avg(q, offset + IonSourceComponent::RedirectVarianceRate);
        });
    }
}
} // namespace
AcceptedIonExchange::AcceptedIonExchange (
    const amrex::Geometry& geometry, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& dm,
    std::vector<KineticSpeciesDescriptor> species, IonExchangeOptions options)
    : m_geometry(geometry), m_cells(cells),
      m_nodes(amrex::convert(cells, amrex::IntVect::TheNodeVector())),
      m_distribution(dm), m_descriptors(std::move(species)), m_options(options),
      m_ledger(m_descriptors.size()) {
#ifndef WARPX_DIM_RZ
    amrex::Abort("AcceptedIonExchange supports new RZ only; existing 3D "
                 "remains unchanged");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !options.embedded_boundary && options.physical_levels == 1 &&
            options.azimuthal_modes == 1,
        "AcceptedIonExchange requires single-level m0/no-EB");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !options.redirect_kick_cap && !options.temperature_shunt,
        "AcceptedIonExchange kick-cap/shunt transforms unsupported");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(options.dt) && options.dt > 0 &&
            std::isfinite(options.time) &&
            std::isfinite(options.raw_density_floor) &&
            options.raw_density_floor >= 0 && geometry.Coord() == 1 &&
            !geometry.isPeriodic(0) && geometry.ProbLo(0) == 0 &&
            geometry.Domain().smallEnd() == amrex::IntVect::TheZeroVector() &&
            cells.ixType().cellCentered() &&
            cells.minimalBox() == geometry.Domain() &&
            cells.numPts() == geometry.Domain().numPts(),
        "AcceptedIonExchange invalid options/physical RZ mesh");
    for (auto const& s : m_descriptors) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !s.name.empty() && std::isfinite(s.mass) && s.mass > 0 &&
                std::isfinite(s.charge_number) && s.charge_number > 0,
            "AcceptedIonExchange invalid species metadata");
        m_coefficients.push_back(
            std::make_unique<MF>(cells, dm, IonExchangeCoefficient::Count, 0));
    }
    MF nodes(m_nodes, dm, 1, 0);
    m_owner = nodes.OwnerMask(geometry.periodicity());
}
bool
AcceptedIonExchange::Prepare (const ThermalSourceState& state, const MF& rates,
                              const std::array<const MF*, 3>& drift,
                              const std::vector<const MF*>& old_ti,
                              const amrex::ParserExecutor<4>* reference) {
    if (m_status != IonExchangeStatus::Unprepared &&
        m_status != IonExchangeStatus::Prepared)
        return false;
    m_status = IonExchangeStatus::Unprepared;
    m_ledger.assign(m_descriptors.size(), {});
    auto layout = [&] (const MF* p, bool nodal, int components) {
        return p && !p->hasEBFabFactory() &&
               p->boxArray() == (nodal ? m_nodes : m_cells) &&
               p->DistributionMap() == m_distribution &&
               p->nComp() >= components;
    };
    bool ok = layout(state.raw_charge, true, 1) &&
              layout(state.temperature_kelvin, true, 1) &&
              layout(&rates, true,
                     std::max(1, int(m_descriptors.size()) *
                                     IonSourceComponent::Count)) &&
              state.species.size() == m_descriptors.size();
    for (auto* d : drift)
        ok = ok && layout(d, true, 1);
    if (reference)
        ok =
            ok && m_options.relaxation && old_ti.size() == m_descriptors.size();
    if (!Collective(ok))
        return false;
    ok = state.raw_charge->is_finite(0, 1, 0) &&
         state.temperature_kelvin->is_finite(0, 1, 0) &&
         rates.is_finite(0, rates.nComp(), 0) &&
         state.temperature_kelvin->min(0) >= 0;
    for (auto* d : drift)
        ok = ok && d->is_finite(0, 1, 0);
    if (!Collective(ok))
        return false;
    for (std::size_t s = 0; s < m_descriptors.size(); ++s) {
        auto const& desc = m_descriptors[s];
        auto const& input = state.species[s];
        ok = input.charge_number == desc.charge_number &&
             input.relaxation_excluded == desc.relaxation_excluded &&
             input.raw_charge;
        auto* ti = reference && !desc.relaxation_excluded ? old_ti[s] : nullptr;
        if (reference && !desc.relaxation_excluded) {
            ok = ok && layout(ti, false, 1);
            if (ok)
                for (auto const& owned : m_coefficients)
                    ok = ok && !Aliases(*ti, *owned);
        }
        if (!Collective(ok))
            return false;
        int const b = s * IonSourceComponent::Count;
        ok = ok && rates.min(b + IonSourceComponent::RedirectPower) >= 0 &&
             rates.min(b + IonSourceComponent::RedirectVarianceRate) >= 0 &&
             rates.min(b + IonSourceComponent::CollisionFrequency) >= 0;
        if (desc.relaxation_excluded || !m_options.relaxation)
            ok = ok &&
                 rates.norminf(b + IonSourceComponent::RelaxationPower) == 0 &&
                 rates.norminf(b + IonSourceComponent::CollisionFrequency) == 0;
        if (desc.relaxation_excluded || !m_options.redirect)
            ok = ok &&
                 rates.norminf(b + IonSourceComponent::RedirectPower) == 0 &&
                 rates.norminf(b + IonSourceComponent::RedirectVarianceRate) ==
                     0;
        if (ti)
            ok = ok && ti->is_finite(0, 1, 0) && ti->min(0) >= 0;
        if (!Collective(ok))
            return false;
        auto& c = *m_coefficients[s];
        c.setVal(0);
        if (!desc.relaxation_excluded)
            PrepareCells(c, *state.raw_charge, *state.temperature_kelvin, rates,
                         drift, ti, s, m_options,
                         reference ? *reference : amrex::ParserExecutor<4>{},
                         reference != nullptr);
        if (!c.is_finite(0, c.nComp(), 0) ||
            c.min(IonExchangeCoefficient::Nu) < 0 ||
            c.min(IonExchangeCoefficient::ReferenceNu) < 0 ||
            c.min(IonExchangeCoefficient::RedirectEnergy) < 0)
            return false;
        auto& ledger = m_ledger[s];
        ledger.requested_relaxation =
            m_options.dt * Integral(rates,
                                    b + IonSourceComponent::RelaxationPower,
                                    m_geometry, *m_owner);
        ledger.requested_redirect =
            m_options.dt * Integral(rates,
                                    b + IonSourceComponent::RedirectPower,
                                    m_geometry, *m_owner);
        if (reference && !desc.relaxation_excluded) {
            MF diff(m_cells, m_distribution, 1, 0);
            MF::LinComb(diff, 1, c, IonExchangeCoefficient::Nu, -1, c,
                        IonExchangeCoefficient::ReferenceNu, 0, 1, 0);
            ledger.reference_nu_max_difference = diff.norminf();
        }
        if (!std::isfinite(ledger.requested_relaxation) ||
            !std::isfinite(ledger.requested_redirect))
            return false;
    }
    m_status = IonExchangeStatus::Prepared;
    return true;
}
bool
AcceptedIonExchange::CommitOnce (
    const std::vector<IonExchangeParticleView>& views) {
    if (m_status != IonExchangeStatus::Prepared)
        return false;
    m_status = IonExchangeStatus::Rejected;
    // Validate layout and non-overlapping storage BEFORE any device
    // access/draw.
    bool ok = true;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
    for (auto const& v : views) {
        ok = ok && v.species < m_descriptors.size() && v.count >= 0;
        if (!ok)
            break;
        if (m_descriptors[v.species].relaxation_excluded || v.count == 0)
            continue;
        auto const& c = *m_coefficients[v.species];
        auto const& indices = c.IndexArray();
        ok = ok &&
             std::find(indices.begin(), indices.end(), v.grid_index) !=
                 indices.end() &&
             v.cell && v.theta && v.weight;
        for (auto* p : v.momentum) {
            ok = ok && p;
            auto lo = reinterpret_cast<std::uintptr_t>(p);
            ranges.emplace_back(lo, lo + v.count * sizeof(P));
        }
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        ok = ok && ranges[i - 1].second <= ranges[i].first;
    if (!Collective(ok))
        return false;
    int bad = 0;
    for (auto const& v : views) {
        if (m_descriptors[v.species].relaxation_excluded || v.count == 0)
            continue;
        auto box = m_cells[v.grid_index];
        auto coeff = m_coefficients[v.species]->const_array(v.grid_index);
        auto mass = m_descriptors[v.species].mass;
        auto dt = m_options.dt;
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<int> data(op);
        using Tuple = decltype(data)::Type;
        op.eval(v.count, data, [=] AMREX_GPU_DEVICE(long p) -> Tuple {
            auto cell = v.cell[p];
            if (!box.contains(cell) || !std::isfinite(v.theta[p]) ||
                !std::isfinite(v.weight[p]) || v.weight[p] < 0)
                return {1};
            for (int d = 0; d < 3; ++d)
                if (!std::isfinite(v.momentum[d][p]))
                    return {1};
            auto const [i, j, k] = cell.dim3();
            P const nu = coeff(i, j, k, IonExchangeCoefficient::Nu),
                    te = coeff(i, j, k,
                               IonExchangeCoefficient::TemperatureKelvin),
                    en = coeff(i, j, k, IonExchangeCoefficient::RedirectEnergy);
            P const variance =
                (-PhysConst::kb * te * std::expm1(-2 * nu * dt) + en) / mass;
            Real const energy = Algorithms::KineticEnergy<Real>(
                v.momentum[0][p], v.momentum[1][p], v.momentum[2][p], mass);
            return {!std::isfinite(variance) || variance < 0 ||
                    !std::isfinite(energy) ||
                    !std::isfinite(v.weight[p] * energy)};
        });
        bad = std::max(bad, amrex::get<0>(data.value()));
    }
    if (!Collective(bad == 0))
        return false;
    struct Candidate {
        amrex::Gpu::DeviceVector<P> momenta;
        amrex::Gpu::DeviceVector<Real> audit;
    };
    std::vector<Candidate> candidates(views.size());
    // Allocate all staging first. No accepted particle changes until every
    // candidate on every rank has passed finite validation.
    for (std::size_t t = 0; t < views.size(); ++t)
        if (!m_descriptors[views[t].species].relaxation_excluded) {
            candidates[t].momenta.resize(3 * views[t].count);
            candidates[t].audit.resize(5 * views[t].count);
        }
    m_status = IonExchangeStatus::FailedAfterDraw;
    std::vector<std::array<Real, 5>> sums(m_descriptors.size());
    bad = 0;
    for (std::size_t t = 0; t < views.size(); ++t) {
        auto v = views[t];
        if (m_descriptors[v.species].relaxation_excluded || v.count == 0)
            continue;
        auto c = m_coefficients[v.species]->const_array(v.grid_index);
        P const mass = m_descriptors[v.species].mass;
        Real const dt = m_options.dt;
        auto out = candidates[t].momenta.data();
        auto audit = candidates[t].audit.data();
        amrex::ParallelForRNG(v.count, [=] AMREX_GPU_DEVICE(
                                           long p,
                                           amrex::RandomEngine const& engine) {
            auto const [i, j, k] = v.cell[p].dim3();
            P const nu = c(i, j, k, IonExchangeCoefficient::Nu),
                    te = c(i, j, k, IonExchangeCoefficient::TemperatureKelvin),
                    en = c(i, j, k, IonExchangeCoefficient::RedirectEnergy);
            amrex::GpuArray<P, 3> u{v.momentum[0][p], v.momentum[1][p],
                                    v.momentum[2][p]},
                drift{c(i, j, k, 1), c(i, j, k, 2), c(i, j, k, 3)},
                normal{0, 0, 0};
            P const drag = -std::expm1(-nu * dt),
                    variance =
                        (-PhysConst::kb * te * std::expm1(-2 * nu * dt) + en) /
                        mass;
            IonExchangeKick kick;
            kick.momentum = u;
            if (drag > 0 || variance > 0) {
                for (int d = 0; d < 3; ++d)
                    normal[d] = amrex::RandomNormal(P(0), P(1), engine);
                kick = ApplyIonExchangeKick(u, drift, v.theta[p], nu, te, en,
                                            mass, v.weight[p], dt, normal);
            }
            for (int d = 0; d < 3; ++d)
                out[3 * p + d] = kick.momentum[d];
            audit[5 * p] = kick.expected_relaxation;
            audit[5 * p + 1] = kick.expected_redirect;
            audit[5 * p + 2] = kick.realized_relaxation;
            audit[5 * p + 3] = kick.realized_redirect;
            audit[5 * p + 4] = kick.realized_total;
        });
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<int> valid(op);
        using Valid = decltype(valid)::Type;
        op.eval(v.count, valid, [=] AMREX_GPU_DEVICE(long p) -> Valid {
            for (int d = 0; d < 3; ++d)
                if (!std::isfinite(out[3 * p + d]))
                    return {1};
            for (int d = 0; d < 5; ++d)
                if (!std::isfinite(audit[5 * p + d]))
                    return {1};
            return {0};
        });
        bad = std::max(bad, amrex::get<0>(valid.value()));
        for (int d = 0; d < 5; ++d) {
            amrex::ReduceOps<amrex::ReduceOpSum> sum;
            amrex::ReduceData<Real> value(sum);
            using Value = decltype(value)::Type;
            sum.eval(v.count, value, [=] AMREX_GPU_DEVICE(long p) -> Value {
                return {audit[5 * p + d]};
            });
            sums[v.species][d] += amrex::get<0>(value.value());
        }
    }
    if (!Collective(bad == 0))
        return false;
    for (std::size_t s = 0; s < sums.size(); ++s) {
        amrex::ParallelDescriptor::ReduceRealSum(sums[s].data(), 5);
        for (auto x : sums[s])
            ok = ok && std::isfinite(x);
    }
    if (!Collective(ok))
        return false;
    for (std::size_t t = 0; t < views.size(); ++t) {
        auto v = views[t];
        if (m_descriptors[v.species].relaxation_excluded || v.count == 0)
            continue;
        auto out = candidates[t].momenta.data();
        amrex::ParallelFor(v.count, [=] AMREX_GPU_DEVICE(long p) {
            for (int d = 0; d < 3; ++d)
                v.momentum[d][p] = out[3 * p + d];
        });
    }
    amrex::Gpu::streamSynchronize();
    for (std::size_t s = 0; s < sums.size(); ++s) {
        auto& l = m_ledger[s];
        l.expected_relaxation = sums[s][0];
        l.expected_redirect = sums[s][1];
        l.realized_relaxation = sums[s][2];
        l.realized_redirect = sums[s][3];
        l.realized_total = sums[s][4];
        l.expectation_defect = l.expected_relaxation + l.expected_redirect -
                               l.requested_relaxation - l.requested_redirect;
        l.realization_defect =
            l.realized_total - l.expected_relaxation - l.expected_redirect;
    }
    m_status = IonExchangeStatus::Committed;
    return true;
}
const std::vector<IonExchangeSpeciesLedger>&
AcceptedIonExchange::Ledger () const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_status == IonExchangeStatus::Prepared ||
                                         m_status ==
                                             IonExchangeStatus::Committed,
                                     "AcceptedIonExchange ledger unavailable");
    return m_ledger;
}
const MF&
AcceptedIonExchange::Coefficients (std::size_t s) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_status == IonExchangeStatus::Prepared ||
         m_status == IonExchangeStatus::Committed) &&
            s < m_coefficients.size(),
        "AcceptedIonExchange coefficients unavailable");
    return *m_coefficients[s];
}
} // namespace warpx::thermal
