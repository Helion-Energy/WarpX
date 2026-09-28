/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "EulerianDissipation.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#ifdef WARPX_DIM_RZ
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#endif
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace warpx::thermal {
namespace {
using amrex::Real;
using MF = amrex::MultiFab;
using Vec = std::array<MF, 3>;
using A3 = amrex::GpuArray<amrex::Array4<const Real>, 3>;
using S3 = amrex::GpuArray<amrex::GpuArray<int, 3>, 3>;
DissipationVector
views (const Vec& a) {
    return {&a[0], &a[1], &a[2]};
}
A3
arrays (const Vec& a, const amrex::MFIter& mfi) {
    return {a[0].const_array(mfi), a[1].const_array(mfi),
            a[2].const_array(mfi)};
}
S3
staggers (const Vec& a) {
    S3 result;
    for (int c = 0; c < 3; ++c) {
        result[c] = {1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[c][d] = a[c].ixType().nodeCentered(d);
        }
    }
    return result;
}
bool
finite (const MF& a, int ng = 0) {
    return !a.contains_nan(0, a.nComp(), ng) &&
           !a.contains_inf(0, a.nComp(), ng);
}
void
copy (MF& to, const MF& from, const amrex::Periodicity& period) {
    MF::Copy(to, from, 0, 0, to.nComp(), 1);
    to.OverrideSync(period);
    to.FillBoundary(period);
}
bool
overlaps (const MF& a, const MF& b) {
    // MFIter instances may not be nested. Sorted disjoint FAB allocation
    // ranges also avoid a quadratic comparison for many local boxes.
    using Range = std::pair<std::uintptr_t, std::uintptr_t>;
    std::vector<Range> ranges;
    for (amrex::MFIter ai(a); ai.isValid(); ++ai) {
        auto const begin = reinterpret_cast<std::uintptr_t>(a[ai].dataPtr());
        ranges.emplace_back(begin, begin + a[ai].box().numPts() * a.nComp() *
                                               sizeof(Real));
    }
    std::sort(ranges.begin(), ranges.end());
    for (amrex::MFIter bi(b); bi.isValid(); ++bi) {
        auto const begin = reinterpret_cast<std::uintptr_t>(b[bi].dataPtr());
        auto const end =
            begin + b[bi].box().numPts() * b.nComp() * sizeof(Real);
        auto const next =
            std::lower_bound(ranges.begin(), ranges.end(), Range{end, 0});
        if (next != ranges.begin() && std::prev(next)->second > begin) {
            return true;
        }
    }
    return false;
}
#ifdef WARPX_DIM_RZ
ElectronViscosityRZEdges
edges (const amrex::Geometry& g, const TrialDissipationOptions& o) {
    ElectronViscosityRZEdges e;
    for (int d = 0; d < 2; ++d) {
        e.lo[d] = g.Domain().smallEnd(d);
        e.hi[d] = g.Domain().bigEnd(d);
        e.periodic[d] = g.isPeriodic(d);
        e.pec_lo[d] = o.boundary[d][0] == DissipationBoundary::PEC;
        e.pec_hi[d] = o.boundary[d][1] == DissipationBoundary::PEC;
    }
    return e;
}
// Extracted from FillNodalElectronVelocity's matched RZ branch: exclude only
// constrained/exterior edges, then divide by the SAME floored nodal rho.
void
velocity (MF& u, const MF& rho, const Vec& J, const Vec& Ji,
          ElectronViscosityRZEdges e, Real floor) {
    u.setVal(0);
    for (amrex::MFIter mfi(u); mfi.isValid(); ++mfi) {
        auto out = u.array(mfi);
        auto r = rho.const_array(mfi);
        auto jp = arrays(J, mfi), ji = arrays(Ji, mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int) {
            auto current = [=] AMREX_GPU_DEVICE(int ii, int jj, int c) {
                return e.active(ii, jj, c) ? ji[c](ii, jj, 0) - jp[c](ii, jj, 0)
                                           : Real(0);
            };
            Real const n = amrex::max(r(i, j, 0), floor);
            out(i, j, 0, 0) =
                i == e.lo[0]
                    ? 0
                    : .5 * (current(i - 1, j, 0) + current(i, j, 0)) / n;
            out(i, j, 0, 1) = current(i, j, 1) / n;
            out(i, j, 0, 2) =
                .5 * (current(i, j - 1, 2) + current(i, j, 2)) / n;
        });
    }
}
void
stress (MF& P, MF& q, MF& clamps, MF& valid, const MF& u, const MF& rho,
        const MF& te, const Vec& B, ElectronViscosityParams p,
        amrex::Box interior) {
    auto const bs = staggers(B);
    for (amrex::MFIter mfi(P); mfi.isValid(); ++mfi) {
        auto out = P.array(mfi), heat = q.array(mfi), audit = clamps.array(mfi),
             ok = valid.array(mfi);
        auto v = u.const_array(mfi), r = rho.const_array(mfi),
             t = te.const_array(mfi);
        auto b = arrays(B, mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            if (!interior.contains(amrex::IntVect(i, j)) ||
                r(i, j, k) <= p.rho_floor || t(i, j, k) <= 0) {
                return;
            }
            ElectronViscosityPoint pt;
            if (!electron_viscosity_point(i, j, k, v, r, t, b[0], b[1], b[2],
                                          bs[0], bs[1], bs[2], p, pt)) {
                ok(i, j, k) = 0;
                return;
            }
            Real packed[6];
            braginskii_stress(pt.mu_par, pt.mu_perp, pt.G, pt.b, pt.S, packed);
            for (int c = 0; c < 6; ++c) {
                out(i, j, k, c) = packed[c];
            }
            heat(i, j, k) =
                braginskii_viscous_heating(pt.mu_par, pt.mu_perp, pt.G, pt.S);
            audit(i, j, k, 0) = pt.nu_par_clamped;
            audit(i, j, k, 1) = pt.nu_perp_clamped;
        });
    }
}
// Exact physical-dual-volume weighted transpose, including boundary-node
// actions and cylindrical metric terms. Do NOT replace by arithmetic div(P).
void
drag (MF& F, const MF& P, const MF& rho, const amrex::Geometry& g, Real floor) {
    auto const dx = g.CellSizeArray();
    auto const lo = g.Domain().smallEnd(), hi = g.Domain().bigEnd();
    auto const volume = MakeQdsmcVolumeElement(g, F.ixType());
    bool const periodic = g.isPeriodic(1);
    for (amrex::MFIter mfi(F); mfi.isValid(); ++mfi) {
        auto f = F.array(mfi);
        auto p = P.const_array(mfi), density = rho.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int) {
            auto weighted = [=] AMREX_GPU_DEVICE(int ii, int jj, int c) {
                if (ii < lo[0] || ii > hi[0] + 1 ||
                    (!periodic && (jj < lo[1] || jj > hi[1] + 1))) {
                    return Real(0);
                }
                return volume(ii, jj) * p(ii, jj, 0, c);
            };
            Real const vol = volume(i, j), r = (i - lo[0]) * dx[0];
            Real const inv = 1 / amrex::max(density(i, j, 0), floor);
            Real const dr =
                ((weighted(i + 1, j, 0) - weighted(i - 1, j, 0)) / (2 * dx[0]) +
                 (weighted(i, j + 1, 4) - weighted(i, j - 1, 4)) /
                     (2 * dx[1])) /
                vol;
            Real const dt =
                ((weighted(i + 1, j, 3) - weighted(i - 1, j, 3)) / (2 * dx[0]) +
                 (weighted(i, j + 1, 5) - weighted(i, j - 1, 5)) /
                     (2 * dx[1])) /
                vol;
            Real const dz =
                ((weighted(i + 1, j, 4) - weighted(i - 1, j, 4)) / (2 * dx[0]) +
                 (weighted(i, j + 1, 2) - weighted(i, j - 1, 2)) /
                     (2 * dx[1])) /
                vol;
            f(i, j, 0, 0) = r > 0 ? -(dr - p(i, j, 0, 1) / r) * inv : 0;
            f(i, j, 0, 1) = r > 0 ? -(dt + p(i, j, 0, 3) / r) * inv : 0;
            f(i, j, 0, 2) = -dz * inv;
        });
    }
}
void
project_viscosity (Vec& E, const MF& F, const amrex::Geometry& g,
                   ElectronViscosityRZEdges e) {
    auto const vn = MakeQdsmcVolumeElement(g, F.ixType());
    for (int c = 0; c < 3; ++c) {
        auto const ve = MakeQdsmcVolumeElement(g, E[c].ixType());
        for (amrex::MFIter mfi(E[c]); mfi.isValid(); ++mfi) {
            auto out = E[c].array(mfi);
            auto f = F.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int) {
                    if (!e.active(i, j, c)) {
                        return;
                    }
                    out(i, j, 0) =
                        c == 0   ? .5 *
                                       (vn(i, j) * f(i, j, 0, c) +
                                        vn(i + 1, j) * f(i + 1, j, 0, c)) /
                                       ve(i, j)
                        : c == 2 ? .5 *
                                       (vn(i, j) * f(i, j, 0, c) +
                                        vn(i, j + 1) * f(i, j + 1, 0, c)) /
                                       ve(i, j)
                                 : f(i, j, 0, c);
                });
        }
    }
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
eta_at (amrex::ParserExecutor<2> eta, bool has_B, amrex::Array4<const Real> rho,
        A3 b, S3 bs, amrex::GpuArray<int, 3> target, int i, int j, int k) {
    using ablastr::coarsen::sample::Interp;
    amrex::GpuArray<int, 3> const nodal{1, 1, 1}, coarsen{1, 1, 1};
    Real mag2 = 0;
    if (has_B) {
        for (int c = 0; c < 3; ++c) {
            Real const v = Interp(b[c], bs[c], target, coarsen, i, j, k, 0);
            mag2 += v * v;
        }
    }
    return eta(Interp(rho, nodal, target, coarsen, i, j, k, 0),
               std::sqrt(mag2));
}
void
hyper_curl (Vec& K, const Vec& J, const Vec& B, const MF& rho,
            TrialDissipationOptions o, const amrex::Geometry& g) {
    using Y = CylindricalYeeAlgorithm;
    auto const bs = staggers(B);
    auto const dx = g.CellSizeArray();
    bool const ampere = o.hyper == HyperMode::AmpereCurlCurl;
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(K[c]); mfi.isValid(); ++mfi) {
            auto out = K[c].array(mfi);
            auto jv = arrays(J, mfi), b = arrays(B, mfi);
            auto r = rho.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                Real const cr[1] = {1 / dx[0]}, cz[1] = {1 / dx[1]};
                Real const coefficient = eta_at(o.eta_h, o.eta_h_depends_on_B,
                                                r, b, bs, bs[c], i, j, k);
                Real curl = 0;
                if (c == 0) {
                    curl = (ampere && i == 0)
                               ? 0
                               : -Y::UpwardDz(jv[1], cz, 1, i, j, k, 0);
                }
                if (c == 1) {
                    curl = Y::UpwardDz(jv[0], cz, 1, i, j, k, 0) -
                           Y::UpwardDr(jv[2], cr, 1, i, j, k, 0);
                }
                if (c == 2) {
                    curl = Y::UpwardDrr_over_r(jv[1], (i + .5) * dx[0], dx[0],
                                               cr, 1, i, j, k, 0);
                }
                // A negative coefficient is unphysical even where curl=0.
                out(i, j, k) = coefficient >= 0 && std::isfinite(coefficient)
                                   ? coefficient * curl
                                   : std::numeric_limits<Real>::quiet_NaN();
            });
        }
        K[c].FillBoundary(g.periodicity());
    }
}
void
hyper_electric (Vec& E, const Vec& K, const Vec& J, const Vec& B, const MF& rho,
                TrialDissipationOptions o, const amrex::Geometry& g) {
    using Y = CylindricalYeeAlgorithm;
    auto const es = staggers(E), bs = staggers(B);
    auto const dx = g.CellSizeArray();
    bool const lap = o.hyper == HyperMode::Laplacian;
    bool const interior_cc = o.hyper == HyperMode::InteriorCurlCurl;
    for (int c = 0; c < 3; ++c) {
        auto interior = amrex::convert(g.Domain(), E[c].ixType());
        if (interior_cc) {
            interior.growHi(0, -1);
            if (!g.isPeriodic(1)) {
                interior.grow(1, -1);
            }
        }
        for (amrex::MFIter mfi(E[c]); mfi.isValid(); ++mfi) {
            auto out = E[c].array(mfi);
            auto jv = arrays(J, mfi), b = arrays(B, mfi), kv = arrays(K, mfi);
            auto density = rho.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                if (c == 1 && i == 0) {
                    return;
                } // actual m=0 Etheta pin
                if (!interior.contains(amrex::IntVect(i, j))) {
                    return;
                }
                Real const cr[1] = {1 / dx[0]}, cz[1] = {1 / dx[1]};
                Real const r = (i + (c == 0 ? Real(.5) : Real(0))) * dx[0];
                Real value = 0;
                if (lap) {
                    Real const eta = eta_at(o.eta_h, o.eta_h_depends_on_B,
                                            density, b, bs, es[c], i, j, k);
                    Real laplace = Y::Dzz(jv[c], cz, 1, i, j, k, 0);
                    if (c == 2 && i == 0) {
                        laplace += 4 * (jv[c](i + 1, j, k) - jv[c](i, j, k)) /
                                   (dx[0] * dx[0]);
                    } else {
                        laplace += Y::Dr_rDr_over_r(jv[c], r, dx[0], cr, 1, i,
                                                    j, k, 0);
                    }
                    if (c != 2) {
                        laplace -= jv[c](i, j, k) / (r * r);
                    }
                    value = eta >= 0 && std::isfinite(eta)
                                ? -eta * laplace
                                : std::numeric_limits<Real>::quiet_NaN();
                } else {
                    if (c == 0) {
                        value = -Y::DownwardDz(kv[1], cz, 1, i, j, k, 0);
                    }
                    if (c == 1) {
                        value = Y::DownwardDz(kv[0], cz, 1, i, j, k, 0) -
                                Y::DownwardDr(kv[2], cr, 1, i, j, k, 0);
                    }
                    if (c == 2) {
                        value = i == 0
                                    ? 4 * kv[1](i, j, k) / dx[0]
                                    : Y::DownwardDrr_over_r(kv[1], r, dx[0], cr,
                                                            1, i, j, k, 0);
                    }
                    if (o.hyper == HyperMode::AmpereCurlCurl) {
                        value = (1 / PhysConst::mu0 * value) * PhysConst::mu0;
                    }
                }
                out(i, j, k) = value;
            });
        }
        E[c].FillBoundary(g.periodicity());
    }
}
// Homogeneous part of the standard electric boundary stack. The affine shell
// overwrite at fixed B has exactly the PEC homogeneous response. PMC uses
// the production ApplyPECtoBfield electric response: normal odd, tangential
// even, with normal nodal constraints only. Corner
// reflections use interior sources, so this never reads concurrently written
// ghosts. Parent still applies its full affine boundary to total E.
void
project_boundary (std::array<MF*, 3> field, const amrex::Geometry& g,
                  TrialDissipationOptions o) {
    auto const cell_lo = g.Domain().smallEnd(), cell_hi = g.Domain().bigEnd();
    amrex::GpuArray<int, 2> lo{}, hi{}, periodic{};
    for (int d = 0; d < 2; ++d) {
        lo[d] = int(o.boundary[d][0]);
        hi[d] = int(o.boundary[d][1]);
        periodic[d] = g.isPeriodic(d);
    }
    for (int c = 0; c < 3; ++c) {
        auto& f = *field[c];
        auto const ix = f.ixType().toIntVect();
        // Valid constrained nodes first. The copy below is read-only while
        // all images, including corner images, are filled.
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto a = f.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                int const v[2] = {i, j};
                bool zero = c == 1 && i == 0;
                for (int d = 0; d < 2; ++d) {
                    int const normal = d == 0 ? 0 : 2;
                    auto constrained = [=] AMREX_GPU_DEVICE(int kind) {
                        return (kind == int(DissipationBoundary::PEC) &&
                                c != normal) ||
                               (kind == int(DissipationBoundary::PMC) &&
                                c == normal);
                    };
                    zero = zero ||
                           (ix[d] &&
                            ((constrained(lo[d]) && v[d] == cell_lo[d]) ||
                             (constrained(hi[d]) && v[d] == cell_hi[d] + 1)));
                }
                if (zero) {
                    a(i, j, k) = 0;
                }
            });
        }
        f.OverrideSync(g.periodicity());
        f.FillBoundary(g.periodicity());
        MF saved(f.boxArray(), f.DistributionMap(), 1, 1);
        MF::Copy(saved, f, 0, 0, 1, 1);
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto a = f.array(mfi);
            auto s = saved.const_array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int v[2] = {i, j};
                    Real sign = 1;
                    bool changed = false;
                    for (int d = 0; d < 2; ++d) {
                        if (periodic[d]) {
                            continue;
                        }
                        int const last = cell_hi[d] + ix[d];
                        bool const lower = v[d]<cell_lo[d], upper = v[d]> last;
                        if (!lower && !upper) {
                            continue;
                        }
                        int const kind = lower ? lo[d] : hi[d];
                        if (kind == int(DissipationBoundary::Open)) {
                            continue;
                        }
                        int const normal = d == 0 ? 0 : 2;
                        bool const odd =
                            kind == int(DissipationBoundary::Axis)
                                ? c != 2
                                : (kind == int(DissipationBoundary::PMC)
                                       ? c == normal
                                       : c != normal);
                        if (odd) {
                            sign = -sign;
                        }
                        v[d] = lower ? 2 * cell_lo[d] - v[d] - (1 - ix[d])
                                     : 2 * last - v[d] + (1 - ix[d]);
                        changed = true;
                    }
                    if (changed) {
                        a(i, j, k) = sign * s(v[0], v[1], k);
                    }
                });
        }
    }
}
#endif
} // namespace

EulerianDissipation::EulerianDissipation (
    const amrex::Geometry& geometry, const amrex::BoxArray& cells,
    const amrex::DistributionMapping& distribution,
    TrialDissipationOptions options)
    : m_geometry(geometry), m_cells(cells),
      m_nodes(amrex::convert(cells, amrex::IntVect::TheNodeVector())),
      m_distribution(distribution), m_options(std::move(options)),
      m_rho(m_nodes, distribution, 1, 1),
      m_temperature(m_nodes, distribution, 1, 1),
      m_velocity(m_nodes, distribution, 3, 1),
      m_stress(m_nodes, distribution, 6, 1),
      m_drag(m_nodes, distribution, 3, 1),
      m_strain(m_nodes, distribution, 1, 1),
      m_clamps(m_nodes, distribution, 2, 1),
      m_valid(m_nodes, distribution, 1, 0) {
#ifndef WARPX_DIM_RZ
    amrex::Abort("EulerianDissipation is qualified only for RZ; existing 3D is "
                 "unchanged");
#endif
    auto const& o = m_options;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !o.embedded_boundary && o.physical_levels == 1 &&
            o.azimuthal_modes == 1 && !o.transformed_electric_solve,
        "EulerianDissipation requires single-level m0/no-EB and additive (not "
        "tensor/curlcurl) E solve");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        cells.ixType().cellCentered() &&
            cells.minimalBox() == geometry.Domain() &&
            cells.numPts() == geometry.Domain().numPts() &&
            geometry.Coord() == 1 &&
            geometry.Domain().smallEnd() == amrex::IntVect(0) &&
            geometry.ProbLo(0) == 0 && !geometry.isPeriodic(0),
        "EulerianDissipation requires complete zero-based RZ Yee mesh with r=0 "
        "axis");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        o.hyper == HyperMode::Off || o.hyper == HyperMode::Laplacian ||
            o.hyper == HyperMode::InteriorCurlCurl ||
            o.hyper == HyperMode::AmpereCurlCurl,
        "Invalid EulerianDissipation hyper mode");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(o.hyper == HyperMode::Off || bool(o.eta_h),
                                     "Hyperresistivity needs a live parser");
    auto& p = m_options.viscous;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !o.viscosity || !p.mc_limited,
        "Matched viscosity requires centered gradients");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !o.viscosity ||
            (!p.mc_limited && p.rho_floor > 0 && std::isfinite(p.rho_floor) &&
             (p.model == 1 || p.model == 2) &&
             (p.model != 2 || (bool(p.nu_par_pars) && bool(p.nu_perp_pars))) &&
             std::isfinite(p.coulomb_log) && p.coulomb_log > 0 &&
             std::isfinite(p.Z_eff) && p.Z_eff > 0 &&
             std::isfinite(p.flux_limit_f) && std::isfinite(p.nu_max) &&
             p.nu_max >= 0 && std::isfinite(p.taper_n) && p.taper_n >= 0),
        "Matched viscosity requires centered gradients and physical "
        "coefficients/parsers");
#ifdef WARPX_DIM_RZ
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        o.boundary[0][0] == DissipationBoundary::Axis &&
            (o.boundary[0][1] == DissipationBoundary::PEC ||
             o.boundary[0][1] == DissipationBoundary::Open),
        "Unsupported radial dissipation boundary");
    for (int side = 0; side < 2; ++side) {
        auto const b = o.boundary[1][side];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            geometry.isPeriodic(1) ? b == DissipationBoundary::Periodic
                                   : (b == DissipationBoundary::PEC ||
                                      b == DissipationBoundary::PMC ||
                                      b == DissipationBoundary::Open),
            "Unsupported axial dissipation boundary");
    }
    p.dx = geometry.CellSizeArray();
    p.dim_to_dir = {0, 2};
    p.problo_r = 0;
    for (int c = 0; c < 3; ++c) {
        amrex::IntVect es(1), bs(0);
        if (c == 0) {
            es[0] = 0;
            bs[0] = 1;
        }
        if (c == 2) {
            es[1] = 0;
            bs[1] = 1;
        }
        for (auto* a : {&m_current[c], &m_ion[c], &m_viscous[c], &m_hyper[c],
                        &m_hyper_raw[c]}) {
            a->define(amrex::convert(cells, es), distribution, 1, 1);
        }
        for (auto* a : {&m_magnetic[c], &m_curl[c]}) {
            a->define(amrex::convert(cells, bs), distribution, 1, 1);
        }
    }
#endif
}
void
EulerianDissipation::CheckInput (const MF* a, const amrex::BoxArray& ba,
                                 int components) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        a && !a->hasEBFabFactory() && a->boxArray() == ba &&
            a->DistributionMap() == m_distribution &&
            a->nComp() >= components && a->nGrow() >= 1,
        "EulerianDissipation input layout/ghosts/EB mismatch");
    bool alias = false;
    for (auto const* out : {&m_rho, &m_temperature, &m_velocity, &m_stress,
                            &m_drag, &m_strain, &m_clamps, &m_valid}) {
        alias = alias || overlaps(*a, *out);
    }
    for (int c = 0; c < 3; ++c) {
        for (auto const* out :
             {&m_current[c], &m_ion[c], &m_magnetic[c], &m_viscous[c],
              &m_hyper[c], &m_hyper_raw[c], &m_curl[c]}) {
            alias = alias || overlaps(*a, *out);
        }
    }
    int any = alias;
    amrex::ParallelDescriptor::ReduceIntMax(any);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !any, "EulerianDissipation inputs must not alias owned scratch");
}
bool
EulerianDissipation::Evaluate (const TrialDissipationState& state) {
    CheckInput(state.charge, m_nodes, 1);
    if (m_options.viscosity) {
        CheckInput(state.temperature_kelvin, m_nodes, 1);
    }
    for (int c = 0; c < 3; ++c) {
        CheckInput(state.plasma_current[c], m_current[c].boxArray(), 1);
        CheckInput(state.magnetic_field[c], m_magnetic[c].boxArray(), 1);
        if (m_options.viscosity) {
            CheckInput(state.ion_current[c], m_ion[c].boxArray(), 1);
        }
    }
    auto const& period = m_geometry.periodicity();
    copy(m_rho, *state.charge, period);
    if (m_options.viscosity) {
        copy(m_temperature, *state.temperature_kelvin, period);
    }
    bool ok =
        finite(m_rho, 1) && (!m_options.viscosity || finite(m_temperature, 1));
    for (int c = 0; c < 3; ++c) {
        copy(m_current[c], *state.plasma_current[c], period);
        copy(m_magnetic[c], *state.magnetic_field[c], period);
        if (m_options.viscosity) {
            copy(m_ion[c], *state.ion_current[c], period);
        }
        ok = finite(m_current[c], 1) && finite(m_magnetic[c], 1) &&
             (!m_options.viscosity || finite(m_ion[c], 1)) && ok;
    }
    for (auto* f : {&m_velocity, &m_stress, &m_drag, &m_strain, &m_clamps}) {
        f->setVal(0);
    }
    m_valid.setVal(1);
    for (int c = 0; c < 3; ++c) {
        for (auto* f :
             {&m_viscous[c], &m_hyper[c], &m_hyper_raw[c], &m_curl[c]}) {
            f->setVal(0);
        }
    }
    if (!ok) {
        return false;
    }
#ifdef WARPX_DIM_RZ
    if (m_options.viscosity) {
        auto const e = edges(m_geometry, m_options);
        velocity(m_velocity, m_rho, m_current, m_ion, e,
                 m_options.viscous.rho_floor);
        m_velocity.FillBoundary(period);
        auto interior = amrex::convert(m_geometry.Domain(),
                                       amrex::IntVect::TheNodeVector());
        for (int d = 0; d < 2; ++d) {
            if (!m_geometry.isPeriodic(d)) {
                interior.grow(d, -1);
            }
        }
        stress(m_stress, m_strain, m_clamps, m_valid, m_velocity, m_rho,
               m_temperature, m_magnetic, m_options.viscous, interior);
        m_stress.FillBoundary(period);
        drag(m_drag, m_stress, m_rho, m_geometry, m_options.viscous.rho_floor);
        m_drag.FillBoundary(period);
        project_viscosity(m_viscous, m_drag, m_geometry, e);
        ProjectElectricIncrement({&m_viscous[0], &m_viscous[1], &m_viscous[2]});
        m_strain.FillBoundary(period);
        m_clamps.FillBoundary(period);
    }
    if (m_options.hyper != HyperMode::Off) {
        if (m_options.hyper != HyperMode::Laplacian) {
            hyper_curl(m_curl, m_current, m_magnetic, m_rho, m_options,
                       m_geometry);
        }
        hyper_electric(m_hyper_raw, m_curl, m_current, m_magnetic, m_rho,
                       m_options, m_geometry);
        for (int c = 0; c < 3; ++c) {
            MF::Copy(m_hyper[c], m_hyper_raw[c], 0, 0, 1, 1);
        }
        ProjectElectricIncrement({&m_hyper[0], &m_hyper[1], &m_hyper[2]});
    }
#endif
    ok = m_valid.min(0) == 1;
    for (auto const* f :
         {&m_velocity, &m_stress, &m_drag, &m_strain, &m_clamps}) {
        ok = finite(*f) && ok;
    }
    for (int c = 0; c < 3; ++c) {
        ok = finite(m_viscous[c], 1) && finite(m_hyper[c], 1) &&
             finite(m_hyper_raw[c]) && finite(m_curl[c]) && ok;
    }
    return ok;
}
void
EulerianDissipation::ProjectElectricIncrement (std::array<MF*, 3> field) const {
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            field[c] && field[c]->boxArray() == m_current[c].boxArray() &&
                field[c]->DistributionMap() == m_distribution &&
                field[c]->nComp() == 1 && field[c]->nGrow() == 1 &&
                !field[c]->hasEBFabFactory(),
            "Dissipation electric projection needs matching one-ghost scratch");
    }
#ifdef WARPX_DIM_RZ
    project_boundary(field, m_geometry, m_options);
#endif
}
DissipationVector
EulerianDissipation::ViscousField () const {
    return views(m_viscous);
}
DissipationVector
EulerianDissipation::HyperField () const {
    return views(m_hyper);
}
DissipationVector
EulerianDissipation::RawHyperField () const {
    return views(m_hyper_raw);
}
} // namespace warpx::thermal
