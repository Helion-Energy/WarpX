/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Exercise the native contact functor on measured or manufactured nodal walls.
#include "EmbeddedBoundary/ParticleBoundaryProcess.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace
{
using Real = amrex::ParticleReal;
struct Data
{
    amrex::GpuArray<Real *, PIdx::nattribs> m_rdata;
};
// SetParticlePosition only requires this SoA view. The backing allocations are
// real device vectors, and the unmodified production functor runs in the
// kernel.
struct Component
{
    Real *pointer;
    Real *dataPtr() const { return pointer; }
};
struct Storage
{
    Data data;
    Component GetRealData(int component) const { return {data.m_rdata[component]}; }
};
struct Tile
{
    Storage storage;
    Storage const &GetStructOfArrays() const { return storage; }
};
} // namespace

int main(int argc, char **argv)
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("wall_test");
        std::string field_path, particle_path;
        pp.query("distance_file", field_path);
        pp.query("particle_file", particle_path);
        int steps = 1000;
        bool require_pass = true;
        bool plane = false;
        bool reject_invalid = false;
        amrex::Real dt = 1.9560085406973426e-10;
        pp.query("steps", steps);
        pp.query("require_pass", require_pass);
        pp.query("plane", plane);
        pp.query("reject_invalid", reject_invalid);
        pp.query("dt", dt);
        amrex::GpuArray<amrex::Real, 3> const lo{-2.0 / 9, -2.0 / 9, 0};
        amrex::GpuArray<amrex::Real, 3> const dxi{360, 360, 1};
        amrex::Box const box(amrex::IntVect(0, 0, 0), amrex::IntVect(160, 160, 2));
        amrex::FArrayBox host_phi(box, 1, amrex::The_Pinned_Arena());
        auto h = host_phi.array();
        std::ifstream file;
        if (!field_path.empty())
        {
            file.open(field_path);
            AMREX_ALWAYS_ASSERT(file.good());
        }
        for (int i = 0; i <= 160; ++i)
        {
            for (int j = 0; j <= 160; ++j)
            {
                amrex::Real value = .2 - std::hypot(lo[0] + i / dxi[0], lo[1] + j / dxi[1]);
                if (plane)
                {
                    value = -(lo[0] + i / dxi[0]);
                }
                if (file.is_open())
                {
                    file >> value;
                    AMREX_ALWAYS_ASSERT(!file.fail());
                }
                for (int k = 0; k <= 2; ++k)
                {
                    h(i, j, k) = value;
                }
            }
        }
        auto distance_host = [&](Real x, Real y)
        {
            int i, j, k;
            amrex::Real weights[3][2];
            ablastr::particles::compute_weights<amrex::IndexType::NODE>(x, y, 1., lo, dxi, i, j, k,
                                                                        weights);
            return ablastr::particles::interp_field_nodal(i, j, k, weights, host_phi.const_array());
        };
        std::vector<std::array<Real, 7>> particles;
        if (!particle_path.empty())
        {
            std::ifstream source(particle_path);
            AMREX_ALWAYS_ASSERT(source.good());
            std::array<Real, 7> p;
            while (source >> p[0] >> p[1] >> p[2] >> p[3] >> p[4] >> p[5] >> p[6])
            {
                p[2] = 1.; // Cylinder fixture is invariant in z.
                particles.push_back(p);
            }
            AMREX_ALWAYS_ASSERT(source.eof() && !particles.empty());
        }
        else if (plane)
        {
            for (int a = 0; a < 512; ++a)
            {
                particles.push_back({-(a % 2 ? 1.e-12 : 0.), -.15 + .3 * a / 512., 1., 2., 1.e5,
                                     (a % 2 ? -1. : 1.) * 1.e4, 1.e4});
            }
        }
        else
        {
            for (int a = 0; a < 512; ++a)
            {
                Real const angle = 2 * std::acos(-1.) * (a + .17) / 512;
                Real const c = std::cos(angle), s = std::sin(angle);
                Real low = .199, high = .201;
                for (int b = 0; b < 50; ++b)
                {
                    Real const mid = .5 * (low + high);
                    if (distance_host(mid * c, mid * s) >= 1.e-12)
                    {
                        low = mid;
                    }
                    else
                    {
                        high = mid;
                    }
                }
                for (Real radial : {1.e5, 1.e3, 1., 0., -1.})
                {
                    particles.push_back({low * c, low * s, 1., 2., radial * c - 1.e5 * s,
                                         radial * s + 1.e5 * c, 1.e4});
                }
            }
        }
        int const count = static_cast<int>(particles.size());
        if (reject_invalid)
        {
            for (auto &p : particles)
            {
                p[0] = .205;
                p[1] = 0.;
                p[4] = 1.;
                p[5] = 0.;
            }
        }
        amrex::FArrayBox phi(box, 1);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_phi.dataPtr(),
                         host_phi.dataPtr() + host_phi.size(), phi.dataPtr());
        auto const distance = phi.const_array();
        std::array<amrex::Gpu::DeviceVector<Real>, PIdx::nattribs> components;
        Tile tile;
        for (int c = 0; c < PIdx::nattribs; ++c)
        {
            std::vector<Real> host(count);
            for (int p = 0; p < count; ++p)
            {
                host[p] = particles[p][c];
            }
            components[c].resize(count);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(),
                             components[c].begin());
            tile.storage.data.m_rdata[c] = components[c].data();
        }
        auto const data = tile.storage.data;
        SetParticlePosition<PIdx> const set_position(tile);
        amrex::Gpu::DeviceVector<int> status(count, 0), contacts(count, 0);
        amrex::Gpu::DeviceVector<Real> energy_error(count, 0.);
        amrex::Gpu::DeviceVector<Real> failure_depth(count, 0.);
        auto *code = status.data();
        auto *hits = contacts.data();
        auto *error = energy_error.data();
        auto *depth = failure_depth.data();
        ParticleBoundaryProcess::ParticleBoundaryInteraction const reflect{
            dt, 2 * PhysConst::m_p, ParticleBoundaryType::Reflecting};
        amrex::ParallelForRNG(
            count,
            [=] AMREX_GPU_DEVICE(int p, amrex::RandomEngine const &rng)
            {
                auto value = [&](Real x, Real y, Real z)
                {
                    int i, j, k;
                    amrex::Real weights[3][2];
                    ablastr::particles::compute_weights<amrex::IndexType::NODE>(x, y, z, lo, dxi, i,
                                                                                j, k, weights);
                    return ablastr::particles::interp_field_nodal(i, j, k, weights, distance);
                };
                Real const u2_initial = data.m_rdata[PIdx::ux][p] * data.m_rdata[PIdx::ux][p] +
                                        data.m_rdata[PIdx::uy][p] * data.m_rdata[PIdx::uy][p] +
                                        data.m_rdata[PIdx::uz][p] * data.m_rdata[PIdx::uz][p];
                if (reject_invalid)
                {
                    auto mutable_data = data;
                    bool const accepted = reflect(
                        mutable_data, p, set_position, data.m_rdata[PIdx::x][p],
                        data.m_rdata[PIdx::y][p], data.m_rdata[PIdx::z][p], distance, lo, dxi, rng);
                    code[p] = accepted ? 4 : 0;
                    return;
                }
                for (int step = 0; step < steps; ++step)
                {
                    Real x = data.m_rdata[PIdx::x][p], y = data.m_rdata[PIdx::y][p];
                    Real z = data.m_rdata[PIdx::z][p];
                    if (value(x, y, z) < 0)
                    {
                        code[p] = 1;
                        break;
                    }
                    UpdatePosition(x, y, z, data.m_rdata[PIdx::ux][p], data.m_rdata[PIdx::uy][p],
                                   data.m_rdata[PIdx::uz][p], dt, 2 * PhysConst::m_p);
                    set_position(p, x, y, z);
                    if (value(x, y, z) < 0)
                    {
                        auto mutable_data = data;
                        if (!reflect(mutable_data, p, set_position, x, y, z, distance, lo, dxi,
                                     rng))
                        {
                            code[p] = 3;
                            break;
                        }
                        ++hits[p];
                        x = data.m_rdata[PIdx::x][p];
                        y = data.m_rdata[PIdx::y][p];
                        z = data.m_rdata[PIdx::z][p];
                        if (value(x, y, z) < 0)
                        {
                            code[p] = 2;
                            depth[p] = -value(x, y, z);
                            break;
                        }
                    }
                }
                Real const u2 = data.m_rdata[PIdx::ux][p] * data.m_rdata[PIdx::ux][p] +
                                data.m_rdata[PIdx::uy][p] * data.m_rdata[PIdx::uy][p] +
                                data.m_rdata[PIdx::uz][p] * data.m_rdata[PIdx::uz][p];
                error[p] = std::abs(u2 - u2_initial) / u2_initial;
            });
        std::vector<int> codes(count), hit_counts(count);
        std::vector<Real> errors(count);
        std::vector<Real> depths(count);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, status.begin(), status.end(), codes.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, contacts.begin(), contacts.end(),
                         hit_counts.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, energy_error.begin(), energy_error.end(),
                         errors.begin());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, failure_depth.begin(), failure_depth.end(),
                         depths.begin());
        int bad = 0, total_hits = 0;
        Real max_error = 0.;
        Real max_depth = 0.;
        for (int p = 0; p < count; ++p)
        {
            if (codes[p])
            {
                ++bad;
            }
            total_hits += hit_counts[p];
            max_error = std::max(max_error, errors[p]);
            max_depth = std::max(max_depth, depths[p]);
        }
        Real max_position_error = 0.;
        Real max_momentum_error = 0.;
        if (plane || reject_invalid)
        {
            for (int c = 0; c < PIdx::nattribs; ++c)
            {
                std::vector<Real> values(count);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, components[c].begin(),
                                 components[c].end(), values.begin());
                for (int p = 0; p < count; ++p)
                {
                    auto const initial = particles[p];
                    Real expected = initial[c];
                    if (plane && !reject_invalid)
                    {
                        Real const gamma =
                            std::sqrt(1. + (initial[4] * initial[4] + initial[5] * initial[5] +
                                            initial[6] * initial[6]) /
                                               PhysConst::c2);
                        if (c == PIdx::x)
                        {
                            expected = -initial[0] - initial[4] / gamma * dt * steps;
                        }
                        if (c == PIdx::y)
                        {
                            expected += initial[5] / gamma * dt * steps;
                        }
                        if (c == PIdx::z)
                        {
                            expected += initial[6] / gamma * dt * steps;
                        }
                        if (c == PIdx::ux)
                        {
                            expected = -expected;
                        }
                    }
                    if (c < 3)
                    {
                        max_position_error =
                            std::max(max_position_error, std::abs(values[p] - expected));
                    }
                    else
                    {
                        max_momentum_error =
                            std::max(max_momentum_error, std::abs(values[p] - expected));
                    }
                }
            }
        }
        amrex::Print() << std::setprecision(17) << "CONTACT_RESULT particles=" << count
                       << " steps=" << steps << " invalid=" << bad << " contacts=" << total_hits
                       << " max_energy_error=" << max_error << " max_solid_depth=" << max_depth
                       << " max_position_error=" << max_position_error
                       << " max_momentum_error=" << max_momentum_error << "\n";
        if (require_pass)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bad == 0,
                                             "Native EB reflection leaves particles in solid");
            AMREX_ALWAYS_ASSERT(max_error < 1.e-11);
            AMREX_ALWAYS_ASSERT(total_hits > 0 || reject_invalid);
            AMREX_ALWAYS_ASSERT(max_position_error < 5.e-12);
            AMREX_ALWAYS_ASSERT(max_momentum_error == 0.);
        }
    }
    amrex::Finalize();
}
