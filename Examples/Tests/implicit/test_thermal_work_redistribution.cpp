/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/EulerianThermalSources.H"
#include "FieldSolver/ImplicitSolvers/ThermalWorkRedistribution.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

namespace {
using amrex::Real;
using namespace warpx::thermal;
using Point = std::array<int, 3>;
constexpr int size = 8;
constexpr int points = (size + 1) * (size + 1) * (AMREX_SPACEDIM == 3 ? size + 1 : 1);
constexpr Real floor_n = 1.e17;

AMREX_GPU_HOST_DEVICE int index (int i, int j, int k) {
    return i + (size + 1) * (j + (size + 1) * k);
}
AMREX_GPU_HOST_DEVICE int canon (int n, bool periodic) {
    return periodic ? (n % size + size) % size : n;
}
AMREX_GPU_HOST_DEVICE Real density (int scenario, int i, int j, int k) {
    bool eligible = true;
    if (scenario == 1) { eligible = (i + j + k) % 2 == 0; }
    if (scenario == 2) { eligible = i == 4 && j == 4 && k == (AMREX_SPACEDIM == 3 ? 4 : 0); }
    if (scenario == 3) { eligible = false; }
    if (scenario == 5) { eligible = (i == 0 || i == size) && (j == 0 || j == size); }
    return (eligible ? 3 : 1) * PhysConst::q_e * floor_n;
}
AMREX_GPU_HOST_DEVICE Real capacity_value (int scenario, int i, int j, int k) {
    if (scenario == 4) {
        int const n = (i + 2 * j + k) % 4;
        if (n == 0) { return 0; }
        if (n == 1) { return -1; }
        if (n == 2) { return std::numeric_limits<Real>::infinity(); }
    }
    return 2 * density(scenario, i, j, k);
}
AMREX_GPU_HOST_DEVICE Real current_value (int c, int i, int j, int k) {
    return (1 + c) * (1 + .13 * i + .07 * j + .11 * k);
}
AMREX_GPU_HOST_DEVICE Real field_value (int c, int i, int j, int k) {
    return ((i + j + k + c) % 2 ? -1 : 1) * (.9 + .21 * c + .03 * i + .04 * j);
}
int direction (int c) {
#if AMREX_SPACEDIM == 3
    return c;
#else
    return c == 0 ? 0 : (c == 2 ? 1 : -1);
#endif
}
bool eligible_host (int scenario, Point p) {
    return density(scenario, p[0], p[1], p[2]) > PhysConst::q_e * floor_n &&
           std::isfinite(capacity_value(scenario, p[0], p[1], p[2])) &&
           capacity_value(scenario, p[0], p[1], p[2]) > 0;
}
Real volume_host (Point p, int cell_direction, bool periodic_z) {
    Real v = 1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Real const center = (p[d] + (d == cell_direction ? .5 : 0)) / size;
        bool const periodic = d == AMREX_SPACEDIM - 1 && periodic_z;
        Real const lo = periodic ? center - .5 / size : std::max(Real(0), center - .5 / size);
        Real const hi = periodic ? center + .5 / size : std::min(Real(1), center + .5 / size);
#if defined(WARPX_DIM_RZ)
        v *= d == 0 ? MathConst::pi * (hi * hi - lo * lo) : hi - lo;
#else
        v *= hi - lo;
#endif
    }
    return v;
}
struct Oracle {
    std::vector<Real> nodal = std::vector<Real>(3 * points, 0);
    Real signed_edge = 0, absolute_orphan = 0;
    std::array<int, 3> counts{};
};
Oracle oracle (int scenario, bool periodic_z) {
    Oracle result;
    std::vector<Point> nodes;
    for (int k = 0; k <= (AMREX_SPACEDIM == 3 ? size : 0); ++k) {
        for (int j = 0; j <= size; ++j) {
            for (int i = 0; i <= size; ++i) {
                Point const p{i, j, k};
                if (periodic_z && p[AMREX_SPACEDIM - 1] == size) { continue; }
                nodes.push_back(p);
            }
        }
    }
    // Independent reference: enumerate physical edges and SCATTER integrated
    // powers into global nodes. Select outer support by graph distance, never
    // by the production offset list or gather routine.
    for (int c = 0; c < 3; ++c) {
        int const dir = direction(c);
        for (auto const edge : nodes) {
            if (dir >= 0 && edge[dir] == size) { continue; }
            std::vector<Point> endpoints{edge};
            if (dir >= 0) {
                auto p = edge;
                ++p[dir];
                if (periodic_z && dir == AMREX_SPACEDIM - 1) { p[dir] %= size; }
                endpoints.push_back(p);
            }
            Real const power = volume_host(edge, dir, periodic_z) *
                               current_value(c, edge[0], edge[1], edge[2]) *
                               field_value(c, edge[0], edge[1], edge[2]);
            result.signed_edge += power;
            std::vector<Point> receivers;
            for (auto const p : endpoints) {
                if (eligible_host(scenario, p)) { receivers.push_back(p); }
            }
            int kind = 0;
            if (receivers.empty()) {
                kind = 1;
                for (auto const p : nodes) {
                    if (!eligible_host(scenario, p)) { continue; }
                    int distance = 100;
                    for (auto const e : endpoints) {
                        int manhattan = 0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            int delta = std::abs(p[d] - e[d]);
                            if (periodic_z && d == AMREX_SPACEDIM - 1) {
                                delta = std::min(delta, size - delta);
                            }
                            manhattan += delta;
                        }
                        distance = std::min(distance, manhattan);
                    }
                    if (distance == 1) { receivers.push_back(p); }
                }
            }
            if (receivers.empty()) {
                kind = 2;
                result.absolute_orphan += std::abs(power);
                for (auto const p : endpoints) {
                    int const n = index(p[0], p[1], p[2]);
                    result.nodal[points + n] += power / endpoints.size();
                    result.nodal[2 * points + n] += std::abs(power) / endpoints.size();
                }
            } else {
                for (auto const p : receivers) {
                    result.nodal[index(p[0], p[1], p[2])] += power / receivers.size();
                }
            }
            ++result.counts[kind];
        }
    }
    for (auto const p : nodes) {
        for (int c = 0; c < 3; ++c) {
            result.nodal[c * points + index(p[0], p[1], p[2])] /= volume_host(p, -1, periodic_z);
        }
    }
    return result;
}
Real difference (const amrex::MultiFab& a, const amrex::MultiFab& b, int ncomp, int ng) {
    amrex::MultiFab d(a.boxArray(), a.DistributionMap(), ncomp, ng);
    amrex::MultiFab::LinComb(d, 1, a, 0, -1, b, 0, 0, ncomp, ng);
    Real result = 0;
    for (int n = 0; n < ncomp; ++n) { result = std::max(result, d.norm0(n, ng)); }
    return result;
}
void run (int scenario, int box_size, bool periodic_z) {
    amrex::Box domain(amrex::IntVect(0), amrex::IntVect(size - 1));
    amrex::RealBox physical({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
    int periods[AMREX_SPACEDIM]{};
    periods[AMREX_SPACEDIM - 1] = periodic_z;
    amrex::Geometry geom(domain, &physical, AMREX_SPACEDIM == 2 ? 1 : 0, periods);
    amrex::BoxArray cells(domain);
    cells.maxSize(box_size);
    amrex::DistributionMapping dm(cells);
    auto const nodes = amrex::convert(cells, amrex::IntVect(1));
    amrex::MultiFab rho(nodes, dm, 1, 2), te(nodes, dm, 1, 2), cap(nodes, dm, 1, 2);
    amrex::MultiFab mask(nodes, dm, 1, 2), rates(nodes, dm, 3, 2), saved(nodes, dm, 3, 2);
    std::array<amrex::MultiFab, 3> j, e;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> work;
    std::array<const amrex::MultiFab*, 3> jp{}, ep{};
    auto fill_state = [&] (int which) {
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            auto const r = rho.array(mfi), t = te.array(mfi), c = cap.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int x, int y, int z) {
                int yy = y, zz = z;
#if AMREX_SPACEDIM == 2
                yy = canon(y, periodic_z);
#else
                zz = canon(z, periodic_z);
#endif
                // Input physical ghosts deliberately look eligible: the helper
                // must never turn them into additional physical heat receivers.
                r(x, y, z) = density(which, x, yy, zz);
                t(x, y, z) = 1.e5;
                c(x, y, z) = capacity_value(which, x, yy, zz);
            });
        }
        rho.FillBoundary(geom.periodicity()); te.FillBoundary(geom.periodicity());
        cap.FillBoundary(geom.periodicity());
    };
    // Preserve original array indices while evaluating periodic analytic data.
    for (int c = 0; c < 3; ++c) {
        amrex::IntVect type(1);
        if (direction(c) >= 0) { type[direction(c)] = 0; }
        auto const ba = amrex::convert(cells, type);
        j[c].define(ba, dm, 1, 2); e[c].define(ba, dm, 1, 2);
        work[c] = std::make_unique<amrex::MultiFab>(ba, dm, 4, 2);
        for (amrex::MFIter mfi(j[c]); mfi.isValid(); ++mfi) {
            auto const ja = j[c].array(mfi), ea = e[c].array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int x, int y, int z) {
                int yy = y, zz = z;
#if AMREX_SPACEDIM == 2
                yy = canon(y, periodic_z);
#else
                zz = canon(z, periodic_z);
#endif
                ja(x, y, z) = current_value(c, x, yy, zz);
                ea(x, y, z) = field_value(c, x, yy, zz);
            });
        }
        jp[c] = &j[c]; ep[c] = &e[c];
    }
    fill_state(scenario);
    std::vector<const amrex::MultiFab*> inputs{&rho, &te, &cap};
    for (int c = 0; c < 3; ++c) { inputs.push_back(jp[c]); inputs.push_back(ep[c]); }
    std::vector<std::unique_ptr<amrex::MultiFab>> copies;
    for (auto const* field : inputs) {
        auto copy = std::make_unique<amrex::MultiFab>(
            field->boxArray(), field->DistributionMap(), 1, 2);
        amrex::MultiFab::Copy(*copy, *field, 0, 0, 1, 2);
        copies.push_back(std::move(copy));
    }
    auto evaluate = [&] () {
        rates.setVal(0);
#ifdef TEST_ORIGINAL_WORK
        ThermalSourceState original_state;
        original_state.raw_charge = &rho; original_state.temperature_kelvin = &te;
        original_state.plasma_current = jp; original_state.magnetic_field = jp;
        original_state.viscous_electric = ep;
        ThermalSourceOptions original_options;
        original_options.density_floor = floor_n;
        original_options.viscosity = ViscousSourceKind::AppliedWork;
        EulerianThermalSources original(geom, rho, original_options);
        amrex::MultiFab oq(nodes, dm, SourceComponent::Count, 2), oi(nodes, dm, 1, 2);
        AMREX_ALWAYS_ASSERT(original.Evaluate(original_state, oq, oi));
        amrex::MultiFab::Copy(rates, oq, SourceComponent::ViscousWork, 0, 1, 0);
        amrex::MultiFab::Copy(rates, oq, SourceComponent::DeclinedViscousWork, 1, 1, 0);
#else
        RedistributeThermalWork(geom, floor_n, rho, te, cap, jp, ep, mask, work,
                               rates, 0, 1, 2);
#endif
        rates.FillBoundary(geom.periodicity());
    };
    evaluate();
    auto const ref = oracle(scenario, periodic_z);
    amrex::Gpu::DeviceVector<Real> reference(ref.nodal.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, ref.nodal.begin(), ref.nodal.end(), reference.begin());
    auto const* expected = reference.data();
    amrex::MultiFab error(nodes, dm, 3, 0);
    for (amrex::MFIter mfi(rates); mfi.isValid(); ++mfi) {
        auto const q = rates.const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(mfi.validbox(), 3, [=] AMREX_GPU_DEVICE(int x, int y, int z, int c) {
            int yy = y, zz = z;
#if AMREX_SPACEDIM == 2
            yy = canon(y, periodic_z);
#else
            zz = canon(z, periodic_z);
#endif
            err(x, y, z, c) = q(x, y, z, c) - expected[c * points + index(x, yy, zz)];
        });
    }
    Real max_error = 0, scale = 1;
    for (int c = 0; c < 3; ++c) {
        max_error = std::max(max_error, error.norm0(c));
        scale = std::max(scale, rates.norm0(c));
    }
    amrex::Print() << std::setprecision(17) << "WORK_ORACLE scenario=" << scenario
                   << " box=" << box_size << " periodic=" << periodic_z
                   << " ranks=" << amrex::ParallelDescriptor::NProcs()
                   << " boxes=" << cells.size() << " error=" << max_error
                   << " scale=" << scale << " signed_edge=" << ref.signed_edge
                   << " absolute_orphan=" << ref.absolute_orphan
                   << " preferred=" << ref.counts[0] << " outer=" << ref.counts[1]
                   << " orphan=" << ref.counts[2] << '\n';
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_error <= 4.e-13 * scale, "independent physical scatter oracle");
    if (scenario == 1) { AMREX_ALWAYS_ASSERT(ref.counts[2] == 0); }
    if (scenario == 2) { AMREX_ALWAYS_ASSERT(ref.counts[1] > 0 && ref.counts[2] > 0); }
    if (scenario == 3) { AMREX_ALWAYS_ASSERT(ref.absolute_orphan > std::abs(ref.signed_edge)); }
    amrex::MultiFab::Copy(saved, rates, 0, 0, 3, 2);
    fill_state((scenario + 1) % 6); evaluate(); fill_state(scenario); evaluate();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(difference(saved, rates, 3, 2) == 0, "work A/B/A including ghosts");
#if defined(WARPX_DIM_RZ) && !defined(TEST_ORIGINAL_WORK)
    ThermalSourceState state;
    state.raw_charge = &rho; state.temperature_kelvin = &te;
    state.heat_capacity_charge = &cap;
    state.plasma_current = jp; state.magnetic_field = jp;
    state.viscous_electric = ep; state.hyper_electric = ep;
    ThermalSourceOptions options;
    options.density_floor = floor_n; options.viscosity = ViscousSourceKind::AppliedWork;
    options.hyperresistive_work = true;
    EulerianThermalSources source(geom, rho, options);
    amrex::MultiFab q(nodes, dm, SourceComponent::Count, 2), ions(nodes, dm, 1, 2);
    AMREX_ALWAYS_ASSERT(source.Evaluate(state, q, ions));
    int const components[3]{SourceComponent::ViscousWork, SourceComponent::DeclinedViscousWork,
                            SourceComponent::UnassignableViscousWorkAbs};
    for (int c = 0; c < 3; ++c) {
        amrex::MultiFab view(q, amrex::make_alias, components[c], 1);
        amrex::MultiFab want(rates, amrex::make_alias, c, 1);
        AMREX_ALWAYS_ASSERT(difference(view, want, 1, 0) == 0);
    }
    auto ledger = source.Ledger(q, ions, 0);
    Real const total = ledger.energy[SourceComponent::ViscousWork] +
                       ledger.energy[SourceComponent::DeclinedViscousWork];
    AMREX_ALWAYS_ASSERT(std::abs(total - ref.signed_edge) < 3.e-13 * std::max(Real(1), ref.absolute_orphan));
    AMREX_ALWAYS_ASSERT(std::abs(ledger.energy[SourceComponent::UnassignableViscousWorkAbs] - ref.absolute_orphan) <
                        3.e-13 * std::max(Real(1), ref.absolute_orphan));
#endif
    for (std::size_t n = 0; n < inputs.size(); ++n) {
        auto const& input = *inputs[n];
        amrex::MultiFab changed(input.boxArray(), input.DistributionMap(), 1, 2);
        for (amrex::MFIter mfi(input); mfi.isValid(); ++mfi) {
            auto const x = input.const_array(mfi), y = copies[n]->const_array(mfi);
            auto const flag = changed.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int ix, int iy, int iz) {
                flag(ix, iy, iz) = x(ix, iy, iz) == y(ix, iy, iz) ? 0 : 1;
            });
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(changed.norm0(0, 2) == 0, "all borrowed inputs unchanged");
    }
}
} // namespace
int main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        int scenario = 1, box = 4, periodic = 0;
        amrex::ParmParse pp("test");
        pp.query("scenario", scenario); pp.query("box", box); pp.query("periodic", periodic);
        run(scenario, box, periodic != 0);
        amrex::Print() << "PASS signed work redistribution\n";
    }
    amrex::Finalize();
}
