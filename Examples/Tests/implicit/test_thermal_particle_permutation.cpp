#include "Particles/Algorithms/ParticleIdentityOrder.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

struct Particle
{
    std::uint64_t id;
    amrex::Real ux, uy, uz;
    amrex::Real drag, sigma;
};

// The oracle starts with particles physically stored in identity order and
// executes the ordinary array-indexed random update. Permuted trials must
// preserve the per-identity result while leaving their storage order untouched.
std::vector<Particle> update (
    std::vector<Particle> const& input, bool const canonical, unsigned long const seed)
{
    int const np = static_cast<int>(input.size());
    amrex::Gpu::DeviceVector<Particle> particles(np);
    amrex::Gpu::DeviceVector<std::uint64_t> identities(np);
    amrex::Gpu::DeviceVector<int> order(np);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, input.begin(), input.end(), particles.begin());
    auto* data = particles.data();
    auto* ids = identities.data();
    auto* sorted = order.data();
    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) { ids[i] = data[i].id; });
    amrex::ResetRandomSeed(seed, seed);
    if (canonical) {
        warpx::particles::ParticleIdentityOrder(np, ids, sorted);
    }
    amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int draw, amrex::RandomEngine const& engine) {
        int const ip = canonical ? sorted[draw] : draw;
        auto& p = data[ip];
        // Inactive particles do not consume draws, as in the native source.
        if (p.drag <= 0 && p.sigma <= 0) { return; }
        p.ux += -p.drag*(p.ux-amrex::Real(0.3))
            + p.sigma*amrex::RandomNormal(amrex::Real(0), amrex::Real(1), engine);
        p.uy += -p.drag*(p.uy+amrex::Real(0.2))
            + p.sigma*amrex::RandomNormal(amrex::Real(0), amrex::Real(1), engine);
        p.uz += -p.drag*(p.uz-amrex::Real(0.1))
            + p.sigma*amrex::RandomNormal(amrex::Real(0), amrex::Real(1), engine);
    });
    std::vector<Particle> result(np);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, particles.begin(), particles.end(), result.begin());
    for (int i = 0; i < np; ++i) {
        AMREX_ALWAYS_ASSERT(result[i].id == input[i].id);
    }
    std::sort(result.begin(), result.end(),
              [](Particle const& a, Particle const& b) { return a.id < b.id; });
    return result;
}

int main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    {
        int cases = 0, rejected = 0;
        for (int np : {0, 1, 2, 17, 257, 4099}) {
            for (int masked : {0, 1}) {
                std::vector<Particle> original;
                for (int i = 0; i < np; ++i) {
                    // Keys vary in both words, including the unsigned high bit.
                    auto const id = (std::uint64_t(i+1) << 32)
                        | (std::uint64_t(i*31+7) & 0xffffffffULL)
                        | ((i%3 == 0) ? (std::uint64_t(1) << 63) : 0);
                    bool const active = !masked || i%7 != 0;
                    original.push_back({id, 0.01*i, -0.03*i, 0.07*i,
                                        active ? 0.02 : 0., active ? 0.1+0.001*i : 0.});
                }
                std::sort(original.begin(), original.end(),
                          [](Particle const& a, Particle const& b) { return a.id < b.id; });
                auto const seed = 157UL + amrex::ParallelDescriptor::MyProc();
                auto const reference = update(original, false, seed);
                for (int permutation : {0, 1, 2}) {
                    auto input = original;
                    if (permutation == 1) { std::reverse(input.begin(), input.end()); }
                    if (permutation == 2 && np > 0) {
                        std::rotate(input.begin(), input.begin()+np/2, input.end());
                    }
                    auto const result = update(input, true, seed);
                    auto const old_result = update(input, false, seed);
                    bool old_equal = true;
                    for (int i = 0; i < np; ++i) {
                        auto const& a = reference[i];
                        auto const& b = result[i];
                        auto const& c = old_result[i];
                        AMREX_ALWAYS_ASSERT(a.id == b.id && a.ux == b.ux
                                            && a.uy == b.uy && a.uz == b.uz);
                        old_equal = old_equal && a.ux == c.ux && a.uy == c.uy && a.uz == c.uz;
                    }
                    rejected += !old_equal;
                    ++cases;
                }
            }
        }
        AMREX_ALWAYS_ASSERT(cases == 36 && rejected > 0);
        amrex::Print() << "THERMAL_IDENTITY_PASS cases=" << cases
                       << " storage_index_negative_controls_rejected=" << rejected << "\n";
    }
    amrex::Finalize();
    return 0;
}
