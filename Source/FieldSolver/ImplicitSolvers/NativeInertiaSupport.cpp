/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "NativeInertiaSupport.H"

#include "DarwinABoundary.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace warpx::darwin
{
namespace
{
using amrex::Real;
using Boundary = InitialRateBoundary;
int Dimension(int c)
{
#if defined(WARPX_DIM_RZ)
    return c == 1 ? -1 : c / 2;
#else
    return c;
#endif
}
amrex::IntVect YeeType(int c)
{
    amrex::IntVect t(1);
    int const d = Dimension(c);
    if (d >= 0)
    {
        t[d] = 0;
    }
    return t;
}
void Layout(const amrex::MultiFab &f, const amrex::BoxArray &ba,
            const amrex::DistributionMapping &dm)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.boxArray() == ba && f.DistributionMap() == dm &&
                                         f.nComp() == 1,
                                     "Native inertia support layout mismatch");
}
bool FiniteDensity(const amrex::MultiFab &f)
{
    return !f.contains_nan(0, 1, 0) && !f.contains_inf(0, 1, 0) && f.min(0) >= 0.;
}
} // namespace

struct NativeInertiaSupport::State
{
    std::array<amrex::MultiFab, 3> raw, capacity, kappa;
    std::array<amrex::iMultiFab, 3> physical, recovery;
    amrex::iMultiFab labels, gauges;
    amrex::MultiFab weights;
    InertiaSupportStats stats;
    bool topology = false;
    State(const amrex::BoxArray &cells, const amrex::DistributionMapping &dm)
        : labels(amrex::convert(cells, amrex::IntVect(1)), dm, 1, 0),
          gauges(labels.boxArray(), dm, 1, 0), weights(labels.boxArray(), dm, 1, 0)
    {
        for (int c = 0; c < 3; ++c)
        {
            auto const ba = amrex::convert(cells, YeeType(c));
            raw[c].define(ba, dm, 1, 0);
            capacity[c].define(ba, dm, 1, 0);
            kappa[c].define(ba, dm, 1, 0);
            physical[c].define(ba, dm, 1, 0);
            recovery[c].define(ba, dm, 1, 0);
        }
    }
};

namespace
{
struct Graph
{
    amrex::IntVect lo, count;
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
    int nodes = 1;
    explicit Graph(const amrex::Geometry &g)
        : lo(g.Domain().smallEnd()), count(g.Domain().length()), periodic(g.isPeriodicArray())
    {
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            count[d] += !periodic[d];
            AMREX_ALWAYS_ASSERT(count[d] <= std::numeric_limits<int>::max() / nodes);
            nodes *= count[d];
        }
        AMREX_ALWAYS_ASSERT(nodes <= std::numeric_limits<int>::max() / AMREX_SPACEDIM);
    }
    int Index(amrex::IntVect p) const
    {
        int index = 0, stride = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            int q = p[d] - lo[d];
            if (periodic[d])
            {
                q = (q % count[d] + count[d]) % count[d];
            }
            AMREX_ALWAYS_ASSERT(q >= 0 && q < count[d]);
            index += stride * q;
            stride *= count[d];
        }
        return index;
    }
    amrex::IntVect Point(int n) const
    {
        auto p = lo;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            p[d] += n % count[d];
            n /= count[d];
        }
        return p;
    }
};

void BuildTopology(NativeInertiaSupport::State &s, const amrex::Geometry &g,
                   const NativeInertiaSupportOptions &o)
{
    Graph const graph(g);
    int const n = graph.nodes;
    // Setup-only host copy. No application or residual path calls this routine.
    amrex::Vector<int> edges(static_cast<std::size_t>(AMREX_SPACEDIM) * n, 0);
    for (int c = 0; c < 3; ++c)
    {
        int const d = Dimension(c);
        if (d < 0)
        {
            continue;
        }
        for (amrex::MFIter mfi(s.kappa[c]); mfi.isValid(); ++mfi)
        {
            auto const &src = s.kappa[c][mfi];
            amrex::FArrayBox host(src.box(), 1, amrex::The_Pinned_Arena());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, src.dataPtr(), src.dataPtr() + src.size(),
                             host.dataPtr());
            auto const a = host.const_array();
            for (amrex::BoxIterator bit(src.box()); bit.ok(); ++bit)
            {
                auto const iv = bit();
                if (a(iv) > 0.)
                {
                    edges[AMREX_SPACEDIM * graph.Index(iv) + d] = 1;
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(edges.data(), static_cast<int>(edges.size()));
    std::vector<int> root(n), size(n, 1), anchored(n, 0), label(n, -1), gauge(n, 0);
    std::iota(root.begin(), root.end(), 0);
    auto find = [&root](int q)
    {
        while (root[q] != q)
        {
            root[q] = root[root[q]];
            q = root[q];
        }
        return q;
    };
    for (int p = 0; p < n; ++p)
    {
        auto const iv = graph.Point(p);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            if (!edges[AMREX_SPACEDIM * p + d])
            {
                continue;
            }
            auto next = iv;
            ++next[d];
            int a = find(p), b = find(graph.Index(next));
            if (a == b)
            {
                continue;
            }
            if (a > b)
            {
                std::swap(a, b);
            }
            root[b] = a;
            size[a] += size[b];
        }
    }
    auto const upper = g.Domain().bigEnd() + amrex::IntVect(1);
    for (int p = 0; p < n; ++p)
    {
        auto const iv = graph.Point(p);
        bool fixed = false;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            fixed = fixed || (o.lower[d] == Boundary::PEC && iv[d] == graph.lo[d]) ||
                    (o.upper[d] == Boundary::PEC && iv[d] == upper[d]);
        }
        if (fixed)
        {
            anchored[find(p)] = 1;
            ++s.stats.dirichlet_nodes;
        }
    }
    std::map<int, int> floating;
    for (int p = 0; p < n; ++p)
    {
        if (find(p) != p)
        {
            continue;
        }
        if (anchored[p])
        {
            ++s.stats.anchored_components;
        }
        else if (size[p] == 1)
        {
            ++s.stats.isolated_nodes;
        }
        else
        {
            floating.emplace(p, static_cast<int>(floating.size()));
        }
    }
    for (int p = 0; p < n; ++p)
    {
        int const r = find(p);
        if (anchored[r])
        {
            label[p] = -1;
        }
        else if (size[r] == 1)
        {
            label[p] = -2;
            gauge[p] = 1;
        }
        else
        {
            label[p] = floating.at(r);
            gauge[p] = p == r;
        }
    }
    s.stats.canonical_nodes = n;
    s.stats.floating_components = static_cast<int>(floating.size());
    s.stats.replicated_graph_bytes = (edges.size() + 5 * static_cast<std::size_t>(n)) * sizeof(int);
    for (amrex::MFIter mfi(s.labels); mfi.isValid(); ++mfi)
    {
        amrex::IArrayBox host_label(mfi.validbox(), 1, amrex::The_Pinned_Arena());
        amrex::IArrayBox host_gauge(mfi.validbox(), 1, amrex::The_Pinned_Arena());
        for (amrex::BoxIterator bit(mfi.validbox()); bit.ok(); ++bit)
        {
            int const p = graph.Index(bit());
            host_label(bit()) = label[p];
            host_gauge(bit()) = gauge[p];
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_label.dataPtr(),
                         host_label.dataPtr() + host_label.size(), s.labels[mfi].dataPtr());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_gauge.dataPtr(),
                         host_gauge.dataPtr() + host_gauge.size(), s.gauges[mfi].dataPtr());
    }
}

InertiaRangeResult Range(const NativeInertiaSupport::State &s, const amrex::MultiFab &rhs,
                         Real absolute_tolerance, Real relative_tolerance)
{
    Layout(rhs, s.weights.boxArray(), s.weights.DistributionMap());
    AMREX_ALWAYS_ASSERT(std::isfinite(absolute_tolerance) && absolute_tolerance >= 0. &&
                        std::isfinite(relative_tolerance) && relative_tolerance >= 0. &&
                        relative_tolerance < 1.);
    InertiaRangeResult result;
    result.finite = !rhs.contains_nan(0, 1, 0) && !rhs.contains_inf(0, 1, 0);
    if (!result.finite)
    {
        return result;
    }
    result.rhs_norm = rhs.norminf(0);
    result.target = absolute_tolerance + relative_tolerance * result.rhs_norm;
    amrex::ReduceOps<amrex::ReduceOpMax> op;
    amrex::ReduceData<Real> data(op);
    using Tuple = decltype(data)::Type;
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        auto const r = rhs.const_array(mfi), w = s.weights.const_array(mfi);
        auto const l = s.labels.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                { return {l(i, j, k) == -2 && w(i, j, k) > 0. ? std::abs(r(i, j, k)) : 0.}; });
    }
    result.isolated_defect = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealMax(result.isolated_defect);
    result.max_null_mean = result.isolated_defect;
    for (int c = 0; c < s.stats.floating_components; ++c)
    {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> sum;
        amrex::ReduceData<Real, Real> reduced(sum);
        using Pair = decltype(reduced)::Type;
        for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
        {
            auto const r = rhs.const_array(mfi), w = s.weights.const_array(mfi);
            auto const l = s.labels.const_array(mfi);
            sum.eval(mfi.validbox(), reduced,
                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Pair
                     {
                         Real const weight = l(i, j, k) == c ? w(i, j, k) : 0.;
                         return {weight * r(i, j, k), weight};
                     });
        }
        auto const pair = reduced.value();
        Real values[2]{amrex::get<0>(pair), amrex::get<1>(pair)};
        amrex::ParallelDescriptor::ReduceRealSum(values, 2);
        AMREX_ALWAYS_ASSERT(values[1] > 0.);
        Real const mean = values[0] / values[1];
        result.floating.push_back({c, values[1], values[0], mean});
        result.max_null_mean = std::max(result.max_null_mean, std::abs(mean));
    }
    result.compatible = result.max_null_mean <= result.target;
    return result;
}
} // namespace

NativeInertiaSupport::NativeInertiaSupport(const amrex::Geometry &g, const amrex::BoxArray &cells,
                                           const amrex::DistributionMapping &dm,
                                           const NativeInertiaSupportOptions &o)
    : m_geometry(g), m_cells(cells), m_distribution(dm), m_options(o)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        cells.ixType().cellCentered() && cells.isDisjoint() && cells.minimalBox() == g.Domain() &&
            cells.numPts() == g.Domain().numPts(),
        "Native inertia support requires one rectangular fully covered level");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        o.coefficient != InertiaEdgePolicy::Unspecified &&
            o.recovery != InertiaRecoveryMask::Unspecified &&
            o.components != InertiaRecoveryComponents::Unspecified &&
            std::isfinite(o.charge_floor) && o.charge_floor > 0. &&
            std::isfinite(o.reference_charge_density) && o.reference_charge_density > 0. &&
            std::isfinite(o.recovery_density_fraction) && o.recovery_density_fraction >= 0.,
        "Native inertia support needs explicit policies and finite physical coefficients");
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        AMREX_ALWAYS_ASSERT((o.lower[d] == Boundary::Periodic) == g.isPeriodic(d) &&
                            (o.upper[d] == Boundary::Periodic) == g.isPeriodic(d));
    }
#if defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT(g.ProbLo(0) == 0. && g.Domain().smallEnd(0) == 0 && !g.isPeriodic(0) &&
                        o.lower[0] == Boundary::Axis && o.upper[0] == Boundary::PEC);
#else
    AMREX_ALWAYS_ASSERT(g.Coord() == 0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        AMREX_ALWAYS_ASSERT(o.lower[d] != Boundary::Axis && o.upper[d] != Boundary::Axis);
    }
#endif
}
NativeInertiaSupport::~NativeInertiaSupport() = default;

bool NativeInertiaSupport::Freeze(const amrex::MultiFab &endpoint, const amrex::MultiFab &recovery,
                                  const InertiaRangeTarget *required)
{
    return FreezeImpl(endpoint, recovery, required, true);
}
bool NativeInertiaSupport::FreezeEdges(const amrex::MultiFab &endpoint,
                                      const amrex::MultiFab &recovery)
{
    return FreezeImpl(endpoint, recovery, nullptr, false);
}
bool NativeInertiaSupport::FreezeImpl(const amrex::MultiFab &endpoint,
                                     const amrex::MultiFab &recovery,
                                     const InertiaRangeTarget *required, bool build_topology)
{
    auto const nodes = amrex::convert(m_cells, amrex::IntVect(1));
    Layout(endpoint, nodes, m_distribution);
    Layout(recovery, nodes, m_distribution);
    m_failure = InertiaSupportFailure::None;
    m_last_range = {};
    if (!FiniteDensity(endpoint) || !FiniteDensity(recovery))
    {
        m_failure = InertiaSupportFailure::InvalidDensity;
        return false;
    }
    auto next = std::make_unique<State>(m_cells, m_distribution);
    amrex::MultiFab rho(nodes, m_distribution, 1, 0), mask_rho(nodes, m_distribution, 1, 0);
    amrex::MultiFab nodal_kappa(nodes, m_distribution, 1, 0);
    amrex::MultiFab::Copy(rho, endpoint, 0, 0, 1, 0);
    amrex::MultiFab::Copy(mask_rho, recovery, 0, 0, 1, 0);
    rho.OverrideSync(m_geometry.periodicity());
    mask_rho.OverrideSync(m_geometry.periodicity());
    Real const floor = m_options.charge_floor, reference = m_options.reference_charge_density;
    for (amrex::MFIter mfi(nodal_kappa); mfi.isValid(); ++mfi)
    {
        auto const r = rho.const_array(mfi);
        auto const k = nodal_kappa.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int z)
            { k(i, j, z) = r(i, j, z) > 0. ? reference / amrex::max(r(i, j, z), floor) : 0.; });
    }
    std::array<amrex::MultiFab, 3> unit;
    for (int c = 0; c < 3; ++c)
    {
        unit[c].define(next->kappa[c].boxArray(), m_distribution, 1, 0);
        unit[c].setVal(1.);
    }
    ApplyYeeInertiaMass(m_geometry, nodal_kappa, {&unit[0], &unit[1], &unit[2]},
                        {&next->kappa[0], &next->kappa[1], &next->kappa[2]});
    auto const lower = m_geometry.Domain().smallEnd();
    auto const upper = m_geometry.Domain().bigEnd() + amrex::IntVect(1);
    amrex::GpuArray<int, AMREX_SPACEDIM> dir_lo{}, dir_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        dir_lo[d] = m_options.lower[d] == Boundary::PEC;
        dir_hi[d] = m_options.upper[d] == Boundary::PEC;
    }
    int const mode = static_cast<int>(m_options.recovery);
    bool const candidate = m_options.coefficient == InertiaEdgePolicy::NativeEdgeCandidate;
    bool const flux_only = m_options.components == InertiaRecoveryComponents::Flux;
    Real const threshold = floor * m_options.recovery_density_fraction;
    for (int c = 0; c < 3; ++c)
    {
        int const d = Dimension(c);
        auto const off = d < 0 ? amrex::IntVect(0) : amrex::IntVect::TheDimensionVector(d);
        auto const type = YeeType(c);
        for (amrex::MFIter mfi(next->raw[c]); mfi.isValid(); ++mfi)
        {
            auto const r = rho.const_array(mfi), m = mask_rho.const_array(mfi);
            auto const raw = next->raw[c].array(mfi), capacity = next->capacity[c].array(mfi);
            auto const kappa = next->kappa[c].array(mfi);
            auto const physical = next->physical[c].array(mfi), mask = next->recovery[c].array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                                   Real const a = r(iv), b = r(iv + off), density = .5 * (a + b);
                                   Real const mask_density = .5 * (m(iv) + m(iv + off));
                                   raw(iv) = density;
                                   capacity(iv) = amrex::max(density, floor);
                                   physical(iv) = density > 0.;
                                   if (candidate && !(a > floor && b > floor))
                                   {
                                       kappa(iv) = density > 0. ? reference / capacity(iv) : 0.;
                                   }
                                   bool vacuum =
                                       mode == static_cast<int>(InertiaRecoveryMask::Global) ||
                                       (mode == static_cast<int>(InertiaRecoveryMask::Vacuum) &&
                                        mask_density < threshold) ||
                                       (mode == static_cast<int>(InertiaRecoveryMask::Transition) &&
                                        mask_density > 0. && mask_density < threshold);
#if defined(WARPX_DIM_RZ)
                                   vacuum = vacuum && (!flux_only || c == 1);
#else
                amrex::ignore_unused(flux_only);
#endif
                                   bool fixed = false;
                                   for (int q = 0; q < AMREX_SPACEDIM; ++q)
                                   {
                                       fixed =
                                           fixed || (type[q] && ((dir_lo[q] && iv[q] == lower[q]) ||
                                                                 (dir_hi[q] && iv[q] == upper[q])));
                                   }
#if defined(WARPX_DIM_RZ)
                                   fixed = fixed || (c == 1 && i == 0);
#endif
                                   mask(iv) = vacuum && !fixed;
                               });
        }
        next->kappa[c].OverrideSync(m_geometry.periodicity());
        if (next->kappa[c].contains_nan(0, 1, 0) || next->kappa[c].contains_inf(0, 1, 0) ||
            next->raw[c].contains_nan(0, 1, 0) || next->raw[c].contains_inf(0, 1, 0) ||
            next->capacity[c].contains_nan(0, 1, 0) || next->capacity[c].contains_inf(0, 1, 0))
        {
            m_failure = InertiaSupportFailure::InvalidDensity;
            return false;
        }
    }
    if (build_topology) {
        BuildTopology(*next, m_geometry, m_options);
        next->topology = true;
    }
    auto const owner = next->weights.OwnerMask(m_geometry.periodicity());
    auto const periodic = m_geometry.isPeriodicArray();
    auto const dx = m_geometry.CellSizeArray();
    Real volume = 1.;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        volume *= dx[d];
    }
#if defined(WARPX_DIM_RZ)
    volume *= 2. * std::acos(-1.) * dx[0];
#endif
    for (amrex::MFIter mfi(next->weights); mfi.isValid(); ++mfi)
    {
        auto const w = next->weights.array(mfi);
        auto const own = owner->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                               Real value = volume;
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   if ((dir_lo[d] && iv[d] == lower[d]) ||
                                       (dir_hi[d] && iv[d] == upper[d]))
                                   {
                                       value = 0.;
                                   }
                                   if (!periodic[d] && (iv[d] == lower[d] || iv[d] == upper[d]))
                                   {
                                       value *= .5;
                                   }
                               }
#if defined(WARPX_DIM_RZ)
                               value *= i == 0 ? .25 : Real(i) - (i == upper[0] ? .5 : 0.);
#endif
                               w(iv) = value * own(iv);
                           });
    }
    if (required && required->rhs)
    {
        m_last_range = Range(*next, *required->rhs, required->absolute_tolerance,
                             required->relative_tolerance);
        if (!m_last_range.compatible)
        {
            m_failure = InertiaSupportFailure::IncompatibleRange;
            return false;
        }
    }
    m_state = std::move(next);
    return true;
}

bool NativeInertiaSupport::IsFrozen() const noexcept { return bool(m_state); }
InertiaSupportFailure NativeInertiaSupport::LastFailure() const noexcept { return m_failure; }
const InertiaRangeResult &NativeInertiaSupport::LastFreezeRange() const noexcept
{
    return m_last_range;
}
const InertiaSupportStats &NativeInertiaSupport::Stats() const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->stats;
}
const amrex::MultiFab &NativeInertiaSupport::RawEdge(int c) const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->raw.at(c);
}
const amrex::MultiFab &NativeInertiaSupport::CapacityEdge(int c) const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->capacity.at(c);
}
const amrex::MultiFab &NativeInertiaSupport::KappaEdge(int c) const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->kappa.at(c);
}
const amrex::iMultiFab &NativeInertiaSupport::PhysicalSupport(int c) const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->physical.at(c);
}
const amrex::iMultiFab &NativeInertiaSupport::RecoveryMask(int c) const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->recovery.at(c);
}
const amrex::iMultiFab &NativeInertiaSupport::ComponentLabels() const
{
    AMREX_ALWAYS_ASSERT(m_state && m_state->topology);
    return m_state->labels;
}
const amrex::iMultiFab &NativeInertiaSupport::GaugeNodes() const
{
    AMREX_ALWAYS_ASSERT(m_state && m_state->topology);
    return m_state->gauges;
}
const amrex::MultiFab &NativeInertiaSupport::NodeWeights() const
{
    AMREX_ALWAYS_ASSERT(m_state);
    return m_state->weights;
}

void NativeInertiaSupport::ApplyMass(const ConstVector &in, const Vector &out, Real scale) const
{
    AMREX_ALWAYS_ASSERT(m_state && std::isfinite(scale));
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        pmc_lo[d] = m_options.lower[d] == Boundary::PMC;
        pmc_hi[d] = m_options.upper[d] == Boundary::PMC;
    }
    for (int c = 0; c < 3; ++c)
    {
        auto const &coefficient = m_state->kappa[c];
        Layout(*in[c], coefficient.boxArray(), m_distribution);
        Layout(*out[c], coefficient.boxArray(), m_distribution);
        for (amrex::MFIter mfi(*out[c]); mfi.isValid(); ++mfi)
        {
            auto const a = out[c]->array(mfi);
            auto const x = in[c]->const_array(mfi);
            auto const k = coefficient.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int z)
                               { a(i, j, z) = scale * k(i, j, z) * x(i, j, z); });
        }
        out[c]->setBndry(0.);
        out[c]->OverrideSync(m_geometry.periodicity());
        out[c]->FillBoundary(m_geometry.periodicity());
        ApplyDarwinPMCVectorBoundary(*out[c], m_geometry, pmc_lo, pmc_hi);
    }
}
InertiaRangeResult NativeInertiaSupport::CheckRange(const amrex::MultiFab &rhs, Real absolute,
                                                    Real relative) const
{
    AMREX_ALWAYS_ASSERT(m_state && m_state->topology);
    return Range(*m_state, rhs, absolute, relative);
}
} // namespace warpx::darwin
