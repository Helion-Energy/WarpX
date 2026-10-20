#include "Particles/Deposition/TemperatureDeposition.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <iomanip>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

struct Case
{
    amrex::GpuArray<amrex::Real, 64> weights{};
    amrex::GpuArray<amrex::Real, 64> velocities{};
    amrex::Real expected = 0;
    int count = 0;
    int single = 0;
};

int main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    int result = 0;
    {
        std::vector<Case> cases;
        for (double scale : {1.e-150, 1., 1.e150}) {
            for (double epsilon : {1., 1.e-4, 1.e-8, 1.e-12, 1.e-14}) {
                for (int count : {2, 3, 8, 64}) {
                    for (int reverse : {0, 1}) {
                        Case c;
                        c.count = count;
                        for (int n = 0; n < count; ++n) {
                            int const j = reverse ? count-1-n : n;
                            c.weights[n] = scale * (j == 0 ? 1. : epsilon);
                            c.velocities[n] = j == 0 ? -0.3 : 1. + 0.02*j;
                        }
                        // Independent pairwise oracle: no W^2 - sum(w^2).
                        long double pair = 0, numerator = 0;
                        for (int i = 0; i < count; ++i) {
                            for (int j = i+1; j < count; ++j) {
                                auto const weight = static_cast<long double>(c.weights[i])
                                    * static_cast<long double>(c.weights[j]);
                                auto const dv = static_cast<long double>(c.velocities[i])
                                    - static_cast<long double>(c.velocities[j]);
                                pair += weight;
                                numerator += weight*dv*dv;
                            }
                        }
                        c.expected = static_cast<amrex::Real>(numerator / (2*pair));
                        cases.push_back(c);
                        // Raw single-pass cancellation is separate from the
                        // reliability repair. Exercise it where well-conditioned.
                        if (epsilon >= 1.e-4) {
                            c.single = 1;
                            cases.push_back(c);
                        }
                    }
                }
            }
        }
        amrex::Gpu::DeviceVector<Case> input(cases.size());
        amrex::Gpu::DeviceVector<amrex::Real> errors(cases.size());
        amrex::Gpu::DeviceVector<amrex::Real> old_errors(cases.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, cases.begin(), cases.end(), input.begin());
        auto const* in = input.data();
        auto* error = errors.data();
        auto* old_error = old_errors.data();
        amrex::ParallelFor(static_cast<int>(cases.size()),
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                auto const& c = in[i];
                amrex::Real w = 0, wv = 0;
                for (int j = 0; j < c.count; ++j) {
                    w += c.weights[j];
                    wv += c.weights[j]*c.velocities[j];
                }
                amrex::Real tail = 0, tail_sq = 0, dominant = 0, wv2 = 0, wsq = 0;
                for (int j = 0; j < c.count; ++j) {
                    amrex::Real const a = c.weights[j]/w;
                    wsq += a*a;
                    if (a > 0.75) {
                        dominant += a;
                    } else {
                        tail += a;
                        tail_sq += a*a;
                    }
                    amrex::Real const dv = c.velocities[j] - (c.single ? 0 : wv/w);
                    wv2 += c.weights[j]*dv*dv;
                }
                amrex::Real mean = wv, variance = 0;
                warpx::particles::deposition::normalizeTemperatureMoments(
                    w, tail, tail_sq, dominant, wv2, mean, variance, c.single != 0);
                error[i] = std::isfinite(variance)
                    ? std::abs(variance-c.expected) / std::max(amrex::Real(1), c.expected)
                    : std::numeric_limits<amrex::Real>::infinity();
                amrex::Real old_variance = wv2/w;
                if (c.single) { old_variance -= (wv/w)*(wv/w); }
                old_variance /= 1-wsq;
                old_error[i] = std::isfinite(old_variance)
                    ? std::abs(old_variance-c.expected) / std::max(amrex::Real(1), c.expected)
                    : std::numeric_limits<amrex::Real>::infinity();
            });
        std::vector<amrex::Real> observed(cases.size()), predecessor(cases.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, errors.begin(), errors.end(), observed.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, old_errors.begin(), old_errors.end(), predecessor.begin());
        auto worst = *std::max_element(observed.begin(), observed.end());
        auto negative = *std::max_element(predecessor.begin(), predecessor.end());
        amrex::ParallelDescriptor::ReduceRealMax(worst);
        amrex::ParallelDescriptor::ReduceRealMax(negative);
        // Original moment/replay precision bound. No tolerance increase.
        bool const passed = std::isfinite(worst) && worst <= 1.e-12 && negative > 1.e-8;
        amrex::Print() << std::setprecision(17) << "TEMPERATURE_STABILITY_RESULT cases=" << cases.size()
                       << " error=" << worst << " predecessor_error=" << negative << "\n";
        amrex::Print() << (passed ? "TEMPERATURE_STABILITY_PASS\n" : "TEMPERATURE_STABILITY_FAIL\n");
        result = passed ? 0 : 1;
    }
    amrex::Finalize();
    return result;
}
