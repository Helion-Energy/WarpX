/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ExpectedIonThermalPartner.H"
#include <AMReX_MFIter.H>
#include <AMReX_Reduce.H>
#include <limits>
namespace warpx::thermal
{
ExpectedIonThermalPartner::ExpectedIonThermalPartner (const AcceptedIonExchange& e)
    : m_geometry(e.Geometry()), m_dt(e.Options().dt),
      m_species(static_cast<int>(e.Descriptors().size())),
      m_source(e.Cells(), e.Distribution(), 1, 0), m_thermal_work(e.Cells(), e.Distribution(), 1, 0)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        sizeof(amrex::Real) == sizeof(double) && sizeof(amrex::ParticleReal) == sizeof(double),
        "ExpectedIonThermalPartner currently requires double field and particle precision");
}
bool ExpectedIonThermalPartner::Evaluate (const IonExchangeMomentAudit& moments,
                                          const ExpectedIonExchangeSource& bounded)
{
    m_valid = false;
    bool ok = moments.Valid() && bounded.Valid() &&
              bounded.Options().mode == ExpectedIonEnergyMode::NonrelativisticBounded;
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok)
    {
        return false;
    }
    using M = IonExchangeMomentComponent;
    using W = ExpectedIonWorkComponent;
    auto const& input = moments.Moments();
    auto const& work = bounded.SpeciesWork();
    ok = input.boxArray() == m_source.boxArray() &&
         input.DistributionMap() == m_source.DistributionMap() &&
         input.nComp() == std::max(1, m_species * M::Count) &&
         work.boxArray() == m_source.boxArray() &&
         work.DistributionMap() == m_source.DistributionMap() &&
         work.nComp() == std::max(1, m_species * W::Count);
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok)
    {
        return false;
    }
    auto const dx = m_geometry.CellSizeArray(), lo = m_geometry.ProbLoArray();
    auto const dt = m_dt;
    int const species = m_species;
    amrex::Real constexpr tol = 256 * std::numeric_limits<amrex::Real>::epsilon();
    amrex::ReduceOps<amrex::ReduceOpMax> op;
    amrex::ReduceData<int> data(op);
    using Tuple = decltype(data)::Type;
    op.eval(1, data, [=] AMREX_GPU_DEVICE(int) -> Tuple { return {0}; });
    for (amrex::MFIter it(m_source); it.isValid(); ++it)
    {
        auto const a = input.const_array(it), w = work.const_array(it);
        auto q = m_source.array(it), heat = m_thermal_work.array(it);
        op.eval(it.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                {
                    amrex::Real sum = 0;
                    for (int s = 0; s < species; ++s)
                    {
                        int const m = s * M::Count, n = s * W::Count;
                        auto const bulk = a(i, j, k, m + M::DeterministicBulk);
                        auto const thermal = a(i, j, k, m + M::ThermalRelaxation);
                        auto const redirect = a(i, j, k, m + M::Redirect);
                        auto const expected = w(i, j, k, n + W::NonrelativisticRelaxation) +
                                              w(i, j, k, n + W::NonrelativisticRedirect);
                        auto const scale = std::abs(bulk) + std::abs(thermal) + std::abs(redirect) +
                                           std::abs(expected);
                        if (std::abs(bulk + thermal + redirect - expected) > tol * scale)
                        {
                            return {1};
                        }
                        sum += thermal;
                    }
                    auto const volume =
                        2 * MathConst::pi * (lo[0] + (i + .5) * dx[0]) * dx[0] * dx[1];
                    heat(i, j, k) = sum;
                    q(i, j, k) = -sum / (volume * dt);
                    return {std::isfinite(q(i, j, k)) ? 0 : 1};
                });
    }
    ok = amrex::get<0>(data.value()) == 0;
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    m_valid = ok;
    return ok;
}
const amrex::MultiFab& ExpectedIonThermalPartner::Source () const
{
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_source;
}
const amrex::MultiFab& ExpectedIonThermalPartner::ThermalWork () const
{
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_thermal_work;
}
amrex::Real ExpectedIonThermalPartner::ThermalEnergy () const
{
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_thermal_work.sum(0, false);
}
amrex::Real ExpectedIonThermalPartner::ElectronEnergy () const
{
    AMREX_ALWAYS_ASSERT(m_valid);
    auto const dx = m_geometry.CellSizeArray(), lo = m_geometry.ProbLoArray();
    auto const dt = m_dt;
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<amrex::Real> data(op);
    using Tuple = decltype(data)::Type;
    op.eval(1, data, [=] AMREX_GPU_DEVICE(int) -> Tuple { return {0.}; });
    for (amrex::MFIter it(m_source); it.isValid(); ++it)
    {
        auto const q = m_source.const_array(it);
        op.eval(it.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                {
                    auto const volume =
                        2 * MathConst::pi * (lo[0] + (i + .5) * dx[0]) * dx[0] * dx[1];
                    return {q(i, j, k) * volume * dt};
                });
    }
    auto energy = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(energy);
    return energy;
}
} // namespace warpx::thermal
