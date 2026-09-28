/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "IonExchangeMomentAudit.H"
#include "ExpectedIonThermalPartner.H"
#include "Utils/WarpXConst.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_MFIter.H>
#include <cmath>
namespace warpx::thermal {
using C = IonExchangeMomentComponent;
IonExchangeMomentAudit::IonExchangeMomentAudit (const AcceptedIonExchange& e)
    : m_moments(e.Cells(), e.Distribution(),
                std::max(1, int(e.Descriptors().size()) * C::Count), 0) {}
bool
IonExchangeMomentAudit::Evaluate (const ImplicitIonEndpointTrial& trial,
                                  const AcceptedIonExchange& exchange) {
    m_valid = false;
    bool ok = trial.Valid() && &trial.Layout() == &exchange &&
              exchange.Status() == IonExchangeStatus::Prepared;
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok)
        return false;
    m_moments.setVal(0.);
    for (auto const& v : trial.Views()) {
        auto a = m_moments.array(v.grid_index);
        int const offset = v.species * C::Count;
        amrex::For(v.count, [=] AMREX_GPU_DEVICE(long p) {
            auto const iv = v.cell[p];
            int const i = iv[0], j = iv[1];
#if AMREX_SPACEDIM == 3
            int const k = iv[2];
#else
            int const k=0;
#endif
            auto const cs = std::cos(v.theta[p]), sn = std::sin(v.theta[p]),
                       w = v.weight[p];
            amrex::Gpu::Atomic::Add(&a(i, j, k, offset + C::Number), w);
            amrex::Gpu::Atomic::Add(&a(i, j, k, offset + C::WeightSquare),
                                    w * w);
            amrex::Gpu::Atomic::Add(
                &a(i, j, k, offset + C::MeanR),
                w * (cs * v.momentum[0][p] + sn * v.momentum[1][p]));
            amrex::Gpu::Atomic::Add(
                &a(i, j, k, offset + C::MeanTheta),
                w * (-sn * v.momentum[0][p] + cs * v.momentum[1][p]));
            amrex::Gpu::Atomic::Add(&a(i, j, k, offset + C::MeanZ),
                                    w * v.momentum[2][p]);
        });
    }
    int const ns = exchange.Descriptors().size();
    for (amrex::MFIter it(m_moments); it.isValid(); ++it) {
        auto a = m_moments.array(it);
        amrex::ParallelFor(it.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               for (int s = 0; s < ns; ++s) {
                                   int const o = s * C::Count;
                                   auto const n = a(i, j, k, o + C::Number);
                                   if (n > 0)
                                       for (int d = 0; d < 3; ++d)
                                           a(i, j, k, o + C::MeanR + d) /= n;
                               }
                           });
    }
    for (auto const& v : trial.Views()) {
        auto a = m_moments.array(v.grid_index);
        int const o = v.species * C::Count;
        amrex::For(v.count, [=] AMREX_GPU_DEVICE(long p) {
            auto const iv = v.cell[p];
            int const i = iv[0], j = iv[1];
#if AMREX_SPACEDIM == 3
            int const k = iv[2];
#else
            int const k=0;
#endif
            auto const cs = std::cos(v.theta[p]), sn = std::sin(v.theta[p]);
            auto const vr = cs * v.momentum[0][p] + sn * v.momentum[1][p] -
                            a(i, j, k, o + C::MeanR);
            auto const vt = -sn * v.momentum[0][p] + cs * v.momentum[1][p] -
                            a(i, j, k, o + C::MeanTheta);
            auto const vz = v.momentum[2][p] - a(i, j, k, o + C::MeanZ);
            amrex::Gpu::Atomic::Add(&a(i, j, k, o + C::VarianceSum),
                                    v.weight[p] *
                                        (vr * vr + vt * vt + vz * vz));
        });
    }
    amrex::Real const dt = exchange.Options().dt;
    for (int s = 0; s < ns; ++s) {
        amrex::Real const mass = exchange.Descriptors()[s].mass;
        int const o = s * C::Count;
        for (amrex::MFIter it(m_moments); it.isValid(); ++it) {
            auto a = m_moments.array(it);
            auto c = exchange.Coefficients(s).const_array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                   int k) {
                auto const n = a(i, j, k, o + C::Number);
                if (n <= 0)
                    return;
                auto const nu = c(i, j, k, IonExchangeCoefficient::Nu);
                auto const b = -std::expm1(-nu * dt),
                           beta = -std::expm1(-2 * nu * dt);
                amrex::Real bulk = 0;
                for (int d = 0; d < 3; ++d) {
                    auto const mean = a(i, j, k, o + C::MeanR + d);
                    auto const delta =
                        -b *
                        (mean - c(i, j, k, IonExchangeCoefficient::DriftR + d));
                    bulk += delta * (2 * mean + delta);
                }
                auto const noise =
                    PhysConst::kb *
                    c(i, j, k, IonExchangeCoefficient::TemperatureKelvin) *
                    beta;
                auto const redirect =
                    c(i, j, k, IonExchangeCoefficient::RedirectEnergy);
                a(i, j, k, o + C::DeterministicBulk) = .5 * mass * n * bulk;
                a(i, j, k, o + C::ThermalRelaxation) =
                    PopulationThermalOUIncrement(
                        n, a(i, j, k, o + C::VarianceSum), mass,
                        c(i, j, k, IonExchangeCoefficient::TemperatureKelvin),
                        nu, dt);
                a(i, j, k, o + C::Redirect) = 1.5 * n * redirect;
                a(i, j, k, o + C::FiniteSampleBulkNoise) =
                    1.5 * (noise + redirect) * a(i, j, k, o + C::WeightSquare) /
                    n;
            });
        }
    }
    m_valid = m_moments.is_finite(0, m_moments.nComp(), 0);
    return m_valid;
}
const amrex::MultiFab&
IonExchangeMomentAudit::Moments () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_moments;
}
std::vector<std::array<amrex::Real, C::Count>>
IonExchangeMomentAudit::Ledger () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    std::vector<std::array<amrex::Real, C::Count>> out(m_moments.nComp() /
                                                       C::Count);
    for (int s = 0; s < int(out.size()); ++s)
        for (int c = 0; c < C::Count; ++c)
            out[s][c] = m_moments.sum(s * C::Count + c, false);
    return out;
}
} // namespace warpx::thermal
