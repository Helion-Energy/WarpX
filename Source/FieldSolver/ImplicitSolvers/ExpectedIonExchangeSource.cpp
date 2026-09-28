/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ExpectedIonExchangeSource.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cstdint>
namespace warpx::thermal {
namespace {
bool Collective (bool ok) {
    int fail = !ok;
    amrex::ParallelDescriptor::ReduceIntMax(fail);
    return fail == 0;
}
bool
Aliases (const amrex::MultiFab& a, const amrex::MultiFab& b) {
    using Range = std::pair<std::uintptr_t, std::uintptr_t>;
    std::vector<Range> ranges;
    for (amrex::MFIter first(a); first.isValid(); ++first) {
        auto lo = reinterpret_cast<std::uintptr_t>(a[first].dataPtr());
        ranges.emplace_back(lo, lo + a[first].box().numPts() * a.nComp() *
                                         sizeof(amrex::Real));
    }
    std::sort(ranges.begin(), ranges.end());
    for (amrex::MFIter second(b); second.isValid(); ++second) {
        auto begin = reinterpret_cast<std::uintptr_t>(b[second].dataPtr());
        auto end = begin + b[second].box().numPts() * b.nComp() * sizeof(amrex::Real);
        auto next =
            std::lower_bound(ranges.begin(), ranges.end(), Range{end, 0});
        if (next != ranges.begin() && std::prev(next)->second > begin)
            return true;
    }
    return false;
}
}
ExpectedIonExchangeSource::ExpectedIonExchangeSource (
    const AcceptedIonExchange& layout, ExpectedIonEnergyOptions options)
    : m_geometry(layout.Geometry()), m_cells(layout.Cells()),
      m_distribution(layout.Distribution()), m_descriptors(layout.Descriptors()),
      m_options(options), m_source(m_cells,m_distribution,1,0),
      m_work(m_cells,m_distribution,
             std::max(1,int(m_descriptors.size())*ExpectedIonWorkComponent::Count),0) {
    // The supplied accepted-map layout already enforces RZ m0/single-level/noEB.
    m_source.setVal(0); m_work.setVal(0);
}
bool ExpectedIonExchangeSource::Evaluate (
    const AcceptedIonExchange& prepared,
    const std::vector<IonExchangeParticleView>& views,
    const amrex::MultiFab& budget, ExpectedIonEndpointContract contract) {
    using R = amrex::Real;
    m_valid = false;
    bool ok = contract == ExpectedIonEndpointContract::CompleteRedistributedPreCollisionEndpoint
        && prepared.Status() == IonExchangeStatus::Prepared
        && prepared.Cells() == m_cells && prepared.Distribution() == m_distribution
        && prepared.Descriptors().size() == m_descriptors.size()
        && !budget.hasEBFabFactory() && budget.boxArray() == m_cells
        && budget.DistributionMap() == m_distribution && budget.nComp() == 1
        && (m_options.mode == ExpectedIonEnergyMode::RelativisticQuadrature
            || m_options.mode == ExpectedIonEnergyMode::NonrelativisticBounded)
        && std::isfinite(m_options.absolute_tolerance_joule)
        && m_options.absolute_tolerance_joule >= 0
        && std::isfinite(m_options.relative_component_tolerance)
        && m_options.relative_component_tolerance >= 0;
    auto const& geom = prepared.Geometry();
    ok = ok && geom.Domain() == m_geometry.Domain() && geom.Coord() == m_geometry.Coord();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        ok = ok && geom.ProbLo(d) == m_geometry.ProbLo(d)
            && geom.ProbHi(d) == m_geometry.ProbHi(d)
            && geom.isPeriodic(d) == m_geometry.isPeriodic(d);
    if (!Collective(ok)) return false;
    if (!Collective(!Aliases(budget,m_source) && !Aliases(budget,m_work))) return false;
    if (!Collective(budget.is_finite(0,1,0))) return false;
    for (std::size_t s = 0; s < m_descriptors.size(); ++s) {
        auto const& a = m_descriptors[s]; auto const& b = prepared.Descriptors()[s];
        ok = ok && a.name == b.name && a.mass == b.mass
            && a.charge_number == b.charge_number
            && a.relaxation_excluded == b.relaxation_excluded;
    }
    // Duplicate particle views would double the extensive work. Reject them
    // before device access, as the accepted once-only consumer does.
    std::vector<std::pair<std::uintptr_t,std::uintptr_t>> ranges;
    for (auto const& v : views) {
        ok = ok && v.species < m_descriptors.size() && v.count >= 0;
        if (!ok) break;
        if (m_descriptors[v.species].relaxation_excluded || v.count == 0) continue;
        auto const& indices = m_work.IndexArray();
        ok = ok && std::find(indices.begin(),indices.end(),v.grid_index) != indices.end()
            && v.cell && v.theta && v.weight;
        for (auto* p : v.momentum) {
            ok = ok && p;
            auto lo = reinterpret_cast<std::uintptr_t>(p);
            ranges.emplace_back(lo,lo+v.count*sizeof(amrex::ParticleReal));
        }
    }
    std::sort(ranges.begin(),ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
        ok = ok && ranges[i-1].second <= ranges[i].first;
    for (auto const* owned : {&m_source,&m_work})
        for (amrex::MFIter mfi(*owned); mfi.isValid(); ++mfi) {
            auto lo = reinterpret_cast<std::uintptr_t>((*owned)[mfi].dataPtr());
            auto hi = lo+(*owned)[mfi].box().numPts()*owned->nComp()*sizeof(R);
            for (auto const& range : ranges)
                ok = ok && (range.second <= lo || range.first >= hi);
        }
    if (!Collective(ok)) return false;
    m_source.setVal(0); m_work.setVal(0);
    R const dt = prepared.Options().dt;
    auto const options = m_options;
    amrex::ReduceOps<amrex::ReduceOpMax> op;
    amrex::ReduceData<int> data(op);
    using Tuple = decltype(data)::Type;
    // One local validation result, including empty ranks; no per-tile readback.
    op.eval(1,data,[=] AMREX_GPU_DEVICE(long) -> Tuple { return {0}; });
    for (auto const& v : views) {
        if (m_descriptors[v.species].relaxation_excluded || v.count == 0) continue;
        auto box = m_cells[v.grid_index];
        auto c = prepared.Coefficients(v.species).const_array(v.grid_index);
        auto out = m_work.array(v.grid_index);
        auto mass = m_descriptors[v.species].mass;
        int const offset = int(v.species)*ExpectedIonWorkComponent::Count;
        op.eval(v.count,data,[=] AMREX_GPU_DEVICE(long p) -> Tuple {
            auto const cell = v.cell[p];
            if (!box.contains(cell) || !std::isfinite(v.weight[p]) || v.weight[p] < 0)
                return {1};
            auto const [i,j,k] = cell.dim3();
            amrex::GpuArray<amrex::ParticleReal,3> const u{
                v.momentum[0][p],v.momentum[1][p],v.momentum[2][p]};
            amrex::GpuArray<amrex::ParticleReal,3> const drift{
                c(i,j,k,1),c(i,j,k,2),c(i,j,k,3)};
            auto const w = ExpectedIonExchangeWork(u,drift,v.theta[p],
                c(i,j,k,IonExchangeCoefficient::Nu),
                c(i,j,k,IonExchangeCoefficient::TemperatureKelvin),
                c(i,j,k,IonExchangeCoefficient::RedirectEnergy),mass,dt,options);
            if (!w.valid) return {1};
            amrex::GpuArray<R,ExpectedIonWorkComponent::Count> const work{
                w.relaxation,w.redirect,w.nr_relaxation,w.nr_redirect,w.nr_drift_work,
                w.convention_lower,w.convention_upper,w.quadrature_error_estimate};
            for (int n = 0; n < ExpectedIonWorkComponent::Count; ++n)
                if (!std::isfinite(R(v.weight[p])*work[n])) return {1};
            for (int n = 0; n < ExpectedIonWorkComponent::Count; ++n)
                amrex::Gpu::Atomic::Add(&out(i,j,k,offset+n),R(v.weight[p])*work[n]);
            return {0};
        });
    }
    if (!Collective(amrex::get<0>(data.value()) == 0)) return false;
    int const species = m_descriptors.size();
    auto const dx = m_geometry.CellSizeArray(), lo = m_geometry.ProbLoArray();
    for (amrex::MFIter mfi(m_source); mfi.isValid(); ++mfi) {
        auto out = m_source.array(mfi);
        auto work = m_work.const_array(mfi), b = budget.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            R const volume = 2*MathConst::pi*(lo[0]+(R(i)+.5)*dx[0])*dx[0]*dx[1];
            R energy = b(i,j,k);
            for (int s = 0; s < species; ++s) {
                int const n = s*ExpectedIonWorkComponent::Count;
                energy -= work(i,j,k,n+ExpectedIonWorkComponent::Relaxation)
                    +work(i,j,k,n+ExpectedIonWorkComponent::Redirect);
            }
            out(i,j,k) = energy/(volume*dt);
        });
    }
    bool const finite_source = m_source.is_finite(0,1,0);
    bool const finite_work = m_work.is_finite(0,m_work.nComp(),0);
    m_valid = Collective(finite_source && finite_work);
    return m_valid;
}
const amrex::MultiFab& ExpectedIonExchangeSource::Source () const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_valid,"Expected OU source unavailable");
    return m_source;
}
const amrex::MultiFab& ExpectedIonExchangeSource::SpeciesWork () const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_valid,"Expected OU work unavailable");
    return m_work;
}
std::vector<ExpectedIonWorkLedger> ExpectedIonExchangeSource::Ledger () const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_valid,"Expected OU ledger unavailable");
    std::vector<ExpectedIonWorkLedger> out(m_descriptors.size());
    for (std::size_t s = 0; s < out.size(); ++s)
        for (int n = 0; n < ExpectedIonWorkComponent::Count; ++n)
            out[s][n] = m_work.sum(int(s)*ExpectedIonWorkComponent::Count+n,true);
    if (!out.empty()) amrex::ParallelDescriptor::ReduceRealSum(out[0].data(),
        out.size()*ExpectedIonWorkComponent::Count);
    return out;
}
} // namespace warpx::thermal
