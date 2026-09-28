/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "DarwinInitialRateSchur.H"
#include "DarwinPoissonMG.H"

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
#include "DarwinABoundary.H"
#if defined(WARPX_DIM_RZ)
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "NonlinearSolvers/FlexibleGMRES.H"

#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <ablastr/coarsen/sample.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::darwin {
namespace {
using amrex::Real;
using Boundary = InitialRateBoundary;
amrex::IntVect
yee_type (int c) {
    amrex::IntVect t(1);
#if defined(WARPX_DIM_RZ)
    if (c != 1) {
        t[c / 2] = 0;
    }
#else
    t[c] = 0;
#endif
    return t;
}
int
dimension (int c) {
#if defined(WARPX_DIM_RZ)
    return c == 1 ? -1 : c / 2;
#else
    return c;
#endif
}
amrex::GpuArray<int, 3>
stagger (int c) {
    amrex::GpuArray<int, 3> t{1, 1, 1};
    int const d = dimension(c);
    if (d >= 0) {
        t[d] = 0;
    }
    return t;
}
} // namespace

void ApplyYeeInertiaMass (
    const amrex::Geometry& geometry, const amrex::MultiFab& kappa,
    const std::array<const amrex::MultiFab*, 3>& in,
    const std::array<amrex::MultiFab*, 3>& out, Real scale)
{
    AMREX_ALWAYS_ASSERT(kappa.ixType().nodeCentered() && kappa.nComp() == 1 &&
                        std::isfinite(scale));
    for (int c = 0; c < 3; ++c) {
        auto& destination = *out[c];
        auto const& source = *in[c];
        AMREX_ALWAYS_ASSERT(source.boxArray() == destination.boxArray() &&
            source.DistributionMap() == destination.DistributionMap() &&
            source.DistributionMap() == kappa.DistributionMap() &&
            source.nComp() == 1 && destination.nComp() == 1 &&
            source.ixType().toIntVect() == yee_type(c) &&
            amrex::convert(source.boxArray(), amrex::IntVect(1)) == kappa.boxArray());
        int const d = dimension(c);
        auto const off = d < 0 ? amrex::IntVect(0)
                              : amrex::IntVect::TheDimensionVector(d);
        for (amrex::MFIter mfi(destination); mfi.isValid(); ++mfi) {
            auto const a = destination.array(mfi);
            auto const x = source.const_array(mfi), kap = kappa.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                Real coefficient = kap(iv);
                if (d >= 0) {
                    Real const other = kap(iv + off);
                    Real const lo = amrex::min(coefficient, other);
                    Real const hi = amrex::max(coefficient, other);
                    coefficient = lo > 0. ? 2. * lo / (1. + lo / hi) : 0.;
                }
                a(iv) = scale * coefficient * x(iv);
            });
        }
        destination.setBndry(0.);
        destination.OverrideSync(geometry.periodicity());
        destination.FillBoundary(geometry.periodicity());
    }
}

void ApplyYeeInertiaMassPaired (
    const amrex::Geometry& geometry, const amrex::MultiFab& kappa,
    const std::array<const amrex::MultiFab*, 3>& in,
    const std::array<const amrex::MultiFab*, 3>& low,
    const std::array<amrex::MultiFab*, 3>& out, Real scale,
    const std::array<amrex::MultiFab*, 3>* output_low)
{
    AMREX_ALWAYS_ASSERT(kappa.ixType().nodeCentered() && kappa.nComp() == 1 &&
                        std::isfinite(scale));
    for (int c = 0; c < 3; ++c) {
        auto& destination = *out[c];
        auto const& source = *in[c];
        AMREX_ALWAYS_ASSERT(source.boxArray() == destination.boxArray() &&
            source.DistributionMap() == destination.DistributionMap() &&
            source.DistributionMap() == kappa.DistributionMap() &&
            source.nComp() == 1 && destination.nComp() == 1 &&
            source.ixType().toIntVect() == yee_type(c) &&
            amrex::convert(source.boxArray(), amrex::IntVect(1)) == kappa.boxArray());
        AMREX_ALWAYS_ASSERT(low[c] && low[c]->boxArray()==source.boxArray() &&
            low[c]->DistributionMap()==source.DistributionMap() && low[c]->nComp()==1);
        if(output_low) {
            auto const* remainder=(*output_low)[c];
            AMREX_ALWAYS_ASSERT(remainder && remainder!=out[c] &&
                remainder->boxArray()==destination.boxArray() &&
                remainder->DistributionMap()==destination.DistributionMap() &&
                remainder->nComp()==1);
        }
        int const d = dimension(c);
        auto const off = d < 0 ? amrex::IntVect(0)
                              : amrex::IntVect::TheDimensionVector(d);
        for (amrex::MFIter mfi(destination); mfi.isValid(); ++mfi) {
            auto const a = destination.array(mfi);
            auto const x = source.const_array(mfi), kap = kappa.const_array(mfi);
            auto const small=low[c]->const_array(mfi);
            auto const remainder=output_low?(*output_low)[c]->array(mfi):amrex::Array4<Real>{};
            bool const retain=output_low!=nullptr;
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                Real coefficient = kap(iv);
                if (d >= 0) {
                    Real const other = kap(iv + off);
                    Real const lo = amrex::min(coefficient, other);
                    Real const hi = amrex::max(coefficient, other);
                    coefficient = lo > 0. ? 2. * lo / (1. + lo / hi) : 0.;
                }
                auto const value=warpx::ohm::compensated::Multiply({x(iv),small(iv)},scale*coefficient);
                a(iv)=value.hi;if(retain)remainder(iv)=value.lo;
            });
        }
        destination.setBndry(0.);
        destination.OverrideSync(geometry.periodicity());
        destination.FillBoundary(geometry.periodicity());
        if(output_low){auto& l=*(*output_low)[c];l.setBndry(0.);l.OverrideSync(geometry.periodicity());l.FillBoundary(geometry.periodicity());}
    }
}

struct DarwinInitialRateSchur::Impl {
    amrex::Geometry geometry;
    amrex::BoxArray cells, nodes;
    amrex::DistributionMapping distribution;
    InitialRateSchurOptions options;
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{}, dir_lo{}, dir_hi{};
    amrex::MultiFab kappa, potential, scratch_phi, nodal_response, weights,
        pc_sigma;
    std::array<amrex::MultiFab, 3> gradient, flux, correction;
    std::unique_ptr<amrex::MLEBNodeFDLaplacian> pc_operator;
    std::unique_ptr<amrex::MLMG> pc;
    bool singular = true, frozen = false;
    Real weight_sum = 0;

    Impl (const amrex::Geometry& g, const amrex::BoxArray& ba,
          const amrex::DistributionMapping& dm,
          const InitialRateSchurOptions& o)
        : geometry(g), cells(ba), nodes(amrex::convert(ba, amrex::IntVect(1))),
          distribution(dm), options(o), kappa(nodes, dm, 1, 1),
          potential(nodes, dm, 1, o.output_ghosts),
          scratch_phi(nodes, dm, 1, 1), nodal_response(nodes, dm, 3, 1),
          weights(nodes, dm, 1, 0), pc_sigma(ba, dm, 1, 0) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            ba.ixType().cellCentered() &&
                ba.minimalBox() == geometry.Domain() && ba.isDisjoint() &&
                ba.numPts() == geometry.Domain().numPts(),
            "InitialRate Schur requires one fully covered rectangular level");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            o.theta == .5 && o.djedt_only && !o.bdf2 &&
                !o.embedded_boundaries && o.azimuthal_modes == 1,
            "InitialRate Schur qualifies midpoint dJe/dt-only, BDF2 off, no "
            "EB and m=0 only");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            o.relative_tolerance > 0 && o.relative_tolerance < 1 &&
                o.absolute_tolerance >= 0 &&
                std::isfinite(o.absolute_tolerance) && o.max_iterations > 0 &&
                o.restart_length > 0 && o.preconditioner_cycles > 0 &&
                o.output_ghosts.allGE(amrex::IntVect(1)),
            "Invalid longitudinal Schur options");
#if defined(WARPX_DIM_RZ)
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            g.ProbLo(0) == 0 && g.Domain().smallEnd(0) == 0 &&
                !g.isPeriodic(0) && o.lower[0] == Boundary::Axis,
            "InitialRate Schur RZ requires an axis-containing domain");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            o.upper[0] == Boundary::PEC &&
                ((g.isPeriodic(1) && o.lower[1] == Boundary::Periodic &&
                  o.upper[1] == Boundary::Periodic) ||
                 (o.compatible_yee && !g.isPeriodic(1) &&
                  o.lower[1] == Boundary::PMC && o.upper[1] == Boundary::PMC)),
            "InitialRate Schur requires radial PEC and periodic z; direct Yee "
            "also supports axial PMC with positive connected inertia support");
#else
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            g.Coord() == 0,
            "Cartesian longitudinal Schur requires Cartesian geometry");
#endif
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lo{}, hi{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                o.output_ghosts[d] <= g.Domain().length(d),
                "InitialRate Schur output guards exceed physical domain");
            for (int side = 0; side < 2; ++side) {
                auto b = side ? o.upper[d] : o.lower[d];
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (b == Boundary::Periodic) == g.isPeriodic(d),
                    "InitialRate Schur periodic geometry/BC mismatch");
                bool axis_allowed = false;
#if defined(WARPX_DIM_RZ)
                axis_allowed = d == 0 && side == 0;
#endif
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    b != Boundary::Axis || axis_allowed,
                    "Unsupported longitudinal axis face");
                (side ? dir_hi[d] : dir_lo[d]) = b == Boundary::PEC;
                (side ? pmc_hi[d] : pmc_lo[d]) = b == Boundary::PMC;
                singular = singular && b != Boundary::PEC;
                (side ? hi[d] : lo[d]) =
                    b == Boundary::PEC        ? amrex::LinOpBCType::Dirichlet
                    : b == Boundary::Periodic ? amrex::LinOpBCType::Periodic
                                              : amrex::LinOpBCType::Neumann;
            }
        }
        for (int c = 0; c < 3; ++c) {
            auto const edges = amrex::convert(ba, yee_type(c));
            gradient[c].define(edges, dm, 1, 1);
            flux[c].define(edges, dm, 1, 1);
            correction[c].define(edges, dm, 1, o.output_ghosts);
        }
        DefineWeights();
        potential.setVal(0);
        for (auto& f : correction) {
            f.setVal(0);
        }
        amrex::LPInfo info;
        int const semi_levels = ConfigureDarwinPoissonSemicoarsening(
            info, g, o.max_semicoarsening_levels, o.semicoarsening_direction);
        if (o.verbose > 0) {
            amrex::Print() << "[darwin] DarwinInitialRateSchur PC: semicoarsening_levels="
                           << semi_levels << " retained_direction="
                           << info.semicoarsening_direction << " agglomeration="
                           << info.do_agglomeration << " consolidation="
                           << info.do_consolidation << "\n";
        }
        pc_operator = std::make_unique<amrex::MLEBNodeFDLaplacian>(
            amrex::Vector<amrex::Geometry>{g},
            amrex::Vector<amrex::BoxArray>{ba},
            amrex::Vector<amrex::DistributionMapping>{dm}, info);
#if defined(WARPX_DIM_RZ)
        pc_operator->setRZ(true);
        pc_operator->setSigma({0., 1.});
#else
        pc_operator->setSigma({AMREX_D_DECL(1., 1., 1.)});
#endif
        pc_operator->setDomainBC(lo, hi);
    }

    void
    DefineWeights () {
        auto const& g = geometry;
        auto owner = weights.OwnerMask(g.periodicity());
        auto const lower = g.Domain().smallEnd(),
                   upper = g.Domain().bigEnd() + amrex::IntVect(1);
        auto const dl = dir_lo, dh = dir_hi;
        auto const per = g.isPeriodicArray();
        for (amrex::MFIter mfi(weights); mfi.isValid(); ++mfi) {
            auto const w = weights.array(mfi);
            auto const own = owner->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                int const p[3] = {i, j, k};
                Real v = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if ((dl[d] && p[d] == lower[d]) ||
                        (dh[d] && p[d] == upper[d])) {
                        v = 0;
                    }
                    if (!per[d] && (p[d] == lower[d] || p[d] == upper[d])) {
                        v *= .5;
                    }
                }
#if defined(WARPX_DIM_RZ)
                // Left null quadrature of native axis4/PMC nodal FD
                // divergence.
                // At a PMC radial wall D uses an odd normal edge image:
                // the final row is -2 F_(N-1/2)/dr. Its left-null
                // weight is (N-1/2)/2, not the trapezoidal N/2.
                v *= i == 0 ? Real(.25)
                            : Real(i) - (i == upper[0] ? Real(.5) : Real(0));
#endif
                w(i, j, k) = v * own(i, j, k);
            });
        }
        weight_sum = weights.sum(0);
        AMREX_ALWAYS_ASSERT(weight_sum > 0);
    }

    void
    Layout (const amrex::MultiFab& f, const amrex::BoxArray& ba) const {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.boxArray() == ba && f.DistributionMap() == distribution &&
                f.nComp() == 1,
            "InitialRate Schur input layout mismatch");
    }
    Real
    Dot (const amrex::MultiFab& a, const amrex::MultiFab& b) const {
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
            auto const x = a.const_array(mfi), y = b.const_array(mfi),
                       w = weights.const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        return {w(i, j, k) * x(i, j, k) * y(i, j, k)};
                    });
        }
        Real sum = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealSum(sum);
        return sum;
    }
    void
    Canonical (amrex::MultiFab& f, bool remove_mean) const {
        Real mean = 0;
        if (singular && remove_mean) {
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
                auto const x = f.const_array(mfi), w = weights.const_array(mfi);
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            return {w(i, j, k) * x(i, j, k)};
                        });
            }
            mean = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceRealSum(mean);
            mean /= weight_sum;
        }
        auto const dl = dir_lo, dh = dir_hi;
        auto const lo = geometry.Domain().smallEnd(),
                   hi = geometry.Domain().bigEnd() + amrex::IntVect(1);
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto const a = f.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int const p[3] = {i, j, k};
                    bool fixed = false;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        fixed = fixed || (dl[d] && p[d] == lo[d]) ||
                                (dh[d] && p[d] == hi[d]);
                    }
                    a(i, j, k) = fixed ? Real(0) : a(i, j, k) - mean;
                });
        }
        f.setBndry(0);
        f.OverrideSync(geometry.periodicity());
        f.FillBoundary(geometry.periodicity());
    }
    void
    Images (amrex::MultiFab& f) const {
        // Exactly ComputeDarwinELong's source/gradient convention: zero
        // physical extension, then PMC vector images. No PEC particle parity.
        f.setBndry(0);
        f.OverrideSync(geometry.periodicity());
        f.FillBoundary(geometry.periodicity());
        ApplyDarwinPMCVectorBoundary(f, geometry, pmc_lo, pmc_hi);
    }
    void
    Gradient (std::array<amrex::MultiFab, 3>& out, const amrex::MultiFab& phi) {
        amrex::MultiFab::Copy(scratch_phi, phi, 0, 0, 1, 0);
        Canonical(scratch_phi, true);
        auto const dx = geometry.CellSizeArray();
        for (int c = 0; c < 3; ++c) {
            int const d = dimension(c);
            if (d < 0) {
                out[c].setVal(0);
                continue;
            }
            auto const off = amrex::IntVect::TheDimensionVector(d);
            Real const idx = 1 / dx[d];
            for (amrex::MFIter mfi(out[c]); mfi.isValid(); ++mfi) {
                auto const a = out[c].array(mfi);
                auto const p = scratch_phi.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                        a(iv) = (p(iv + off) - p(iv)) * idx;
                    });
            }
            Images(out[c]);
        }
    }
    void
    Divergence (amrex::MultiFab& out,
                const std::array<amrex::MultiFab, 3>& in) const {
        auto const dx = geometry.CellSizeArray();
        for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
            auto const a = out.array(mfi);
            auto const x = in[0].const_array(mfi);
            auto const y = in[1].const_array(mfi), z = in[2].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                Real const ix = 1 / dx[0];
#if defined(WARPX_DIM_RZ)
                Real const iz = 1 / dx[1], r = i * dx[0];
                a(i, j, k) =
                    (i == 0 ? 4 * x(i, j, k) / dx[0]
                            : CylindricalYeeAlgorithm::DownwardDrr_over_r(
                                  x, r, dx[0], &ix, 1, i, j, k, 0)) +
                    CylindricalYeeAlgorithm::DownwardDz(z, &iz, 1, i, j, k, 0);
                amrex::ignore_unused(y);
#else
                Real const iy=1/dx[1], iz=1/dx[2];
                a(i,j,k)=CartesianYeeAlgorithm::DownwardDx(x,&ix,1,i,j,k,0)+
                    CartesianYeeAlgorithm::DownwardDy(y,&iy,1,i,j,k,0)+
                    CartesianYeeAlgorithm::DownwardDz(z,&iz,1,i,j,k,0);
#endif
            });
        }
        Canonical(out, false);
    }
    // Evaluate the linear defect without materializing raw-held. A tiny
    // longitudinal correction can be below an ulp of a solenoidal current;
    // subtracting the vectors first would erase that correction entirely.
    void
    DivergenceDifference (amrex::MultiFab& out, const ConstVector& raw,
                          const ConstVector& held) {
        amrex::MultiFab held_divergence(nodes, distribution, 1, 0);
        for (int pass = 0; pass < 2; ++pass) {
            auto const& input = pass == 0 ? raw : held;
            for (int c = 0; c < 3; ++c) {
                AMREX_ALWAYS_ASSERT(input[c]);
                Layout(*input[c], flux[c].boxArray());
                amrex::MultiFab::Copy(flux[c], *input[c], 0, 0, 1, 0);
                Images(flux[c]);
            }
            Divergence(pass == 0 ? out : held_divergence, flux);
        }
        amrex::MultiFab::Subtract(out, held_divergence, 0, 0, 1, 0);
    }

    void
    Apply (amrex::MultiFab& out, const amrex::MultiFab& phi) {
        AMREX_ALWAYS_ASSERT(frozen && &out != &phi);
        Layout(out, nodes);
        Layout(phi, nodes);
        Gradient(gradient, phi);
        if (options.compatible_yee) {
            ApplyYeeInertiaMass(geometry, kappa,
                {&gradient[0], &gradient[1], &gradient[2]},
                {&flux[0], &flux[1], &flux[2]});
            for (auto& f : flux) { Images(f); }
            Divergence(out, flux);
            return;
        }
        using ablastr::coarsen::sample::Interp;
        amrex::GpuArray<int, 3> const node{1, 1, 1}, ratio{1, 1, 1};
        for (int c = 0; c < 3; ++c) {
            auto const edge = stagger(c);
            for (amrex::MFIter mfi(nodal_response); mfi.isValid(); ++mfi) {
                auto const n = nodal_response.array(mfi);
                auto const a = gradient[c].const_array(mfi);
                auto const kap = kappa.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    n(i, j, k, c) =
                        kap(i, j, k) * Interp(a, edge, node, ratio, i, j, k, 0);
                });
            }
        }
        nodal_response.setBndry(0);
        nodal_response.OverrideSync(geometry.periodicity());
        nodal_response.FillBoundary(geometry.periodicity());
        for (int c = 0; c < 3; ++c) {
            if (dimension(c) < 0) {
                flux[c].setVal(0);
                continue;
            }
            auto const edge = stagger(c);
            for (amrex::MFIter mfi(flux[c]); mfi.isValid(); ++mfi) {
                auto const f = flux[c].array(mfi);

                auto const n = nodal_response.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        f(i, j, k) = Interp(n, node, edge, ratio, i, j, k, c);
                    });
            }
            Images(flux[c]);
        }
        Divergence(out, flux);
    }
    bool
    Freeze (const amrex::MultiFab& input) {
        Layout(input, nodes);
        frozen = false;
        if (!input.is_finite(0, 1, 0) || input.min(0) < 0) {
            return false;
        }
        amrex::MultiFab::Copy(kappa, input, 0, 0, 1, 0);
        kappa.setBndry(0);
        kappa.OverrideSync(geometry.periodicity());
        kappa.FillBoundary(geometry.periodicity());
        for (amrex::MFIter mfi(pc_sigma); mfi.isValid(); ++mfi) {
            auto const sigma = pc_sigma.array(mfi);
            auto const kap = kappa.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real sum = 0;
                    for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
                        sum += kap(i + (bits & 1), j + ((bits >> 1) & 1),
                                   k + ((bits >> 2) & 1));
                    }
                    sigma(i, j, k) = sum / Real(1 << AMREX_SPACEDIM);
                });
        }
        pc.reset();
        pc_operator->setSigma(0, pc_sigma);
        pc = std::make_unique<amrex::MLMG>(*pc_operator);
        pc->setVerbose(0);
        pc->setBottomVerbose(0);
        pc->setFixedIter(options.preconditioner_cycles);
        pc->setMaxIter(options.preconditioner_cycles);
        pc->setBottomSolver(amrex::BottomSolver::smoother);
        pc->setBottomMaxIter(12);
        frozen = true;
        return true;
    }
    struct Ops {
        using RT = Real;
        using Vec = amrex::MultiFab;
        Impl& self;
        Vec
        makeVecLHS () const {
            return Vec(self.nodes, self.distribution, 1, 1);
        }
        Vec
        makeVecRHS () const {
            return makeVecLHS();
        }
        void
        setToZero (Vec& x) const {
            x.setVal(0);
        }
        void
        assign (Vec& x, const Vec& y) const {
            Vec::Copy(x, y, 0, 0, 1, 0);
        }
        RT
        norm2 (const Vec& x) const {
            RT const squared = self.Dot(x, x);
            return std::isfinite(squared) && squared >= RT(0)
                ? std::sqrt(squared) : std::numeric_limits<RT>::infinity();
        }
        RT
        dotProduct (const Vec& x, const Vec& y) const {
            return self.Dot(x, y);
        }
        void
        scale (Vec& x, RT a) const {
            x.mult(a, 0, 1, 0);
        }
        void
        increment (Vec& x, const Vec& y, RT a) const {
            Vec::Saxpy(x, a, y, 0, 0, 1, 0);
        }
        void
        linComb (Vec& z, RT a, const Vec& x, RT b, const Vec& y) const {
            Vec::LinComb(z, a, x, 0, b, y, 0, 0, 1, 0);
        }
        void
        apply (Vec& out, const Vec& in) {
            self.Apply(out, in);
        }
        void
        precond (Vec& out, const Vec& rhs) {
            out.setVal(0);
            self.pc->solve({&out}, {&rhs}, 0., 0.);
            self.Canonical(out, true);
        }
    };
    InitialRateSchurResult
    Solve (const amrex::MultiFab& rhs, int remaining_iterations = -1) {
        AMREX_ALWAYS_ASSERT(frozen);
        Layout(rhs, nodes);
        amrex::MultiFab b(nodes, distribution, 1, 1),
            action(nodes, distribution, 1, 0);
        amrex::MultiFab::Copy(b, rhs, 0, 0, 1, 0);
        Canonical(b, false);
        Ops ops{*this};
        InitialRateSchurResult result;
        result.initial_residual = ops.norm2(b);
        result.target =
            std::max(options.absolute_tolerance,
                     options.relative_tolerance * result.initial_residual);
        if (!rhs.is_finite(0, 1, 0)) {
            result.residual = std::numeric_limits<Real>::infinity();
            return result;
        }
        FlexibleGMRES<amrex::MultiFab, Ops> solver;
        solver.define(ops);
        solver.setVerbose(options.verbose);
        int const limit = remaining_iterations < 0 ? options.max_iterations : remaining_iterations;
        AMREX_ALWAYS_ASSERT(limit >= 0 && limit <= options.max_iterations);
        solver.setMaxIters(limit);
        solver.setRestartLength(options.restart_length);
        solver.solve(potential, b, options.relative_tolerance,
                     options.absolute_tolerance);
        result.iterations = solver.getNumIters();
        Canonical(potential, true);
        Apply(action, potential);
        amrex::MultiFab::Subtract(action, b, 0, 0, 1, 0);
        result.residual = ops.norm2(action);
        result.converged = solver.getStatus() == 0 &&
                           std::isfinite(result.residual) &&
                           result.residual <= result.target;
        Gradient(correction, potential);
        return result;
    }
};
DarwinInitialRateSchur::DarwinInitialRateSchur (
    const amrex::Geometry& g, const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm, const InitialRateSchurOptions& o)
    : m_impl(std::make_unique<Impl>(g, ba, dm, o)) {}
DarwinInitialRateSchur::~DarwinInitialRateSchur () = default;
bool
DarwinInitialRateSchur::Freeze (const amrex::MultiFab& k) {
    return m_impl->Freeze(k);
}
void
DarwinInitialRateSchur::ApplyPotential (amrex::MultiFab& out,
                                         const amrex::MultiFab& phi) {
    m_impl->Apply(out, phi);
}
InitialRateSchurResult
DarwinInitialRateSchur::SolvePotential (const amrex::MultiFab& rhs) {
    return m_impl->Solve(rhs);
}
InitialRateSchurResult
DarwinInitialRateSchur::Correct (const ConstVector& raw,
                                  const ConstVector& held) {
    auto& s = *m_impl;
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(raw[c] && held[c]);
        for (int d = 0; d < 3; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                raw[c] != &s.correction[d] && held[c] != &s.correction[d],
                "Correct inputs must be independent of owned output fields");
        }
        s.Layout(*raw[c], s.flux[c].boxArray());
        s.Layout(*held[c], s.flux[c].boxArray());
        amrex::MultiFab::LinComb(s.flux[c], 1., *raw[c], 0, -1., *held[c], 0, 0,
                                 1, 0);
        s.Images(s.flux[c]);
    }
    amrex::MultiFab rhs(s.nodes, s.distribution, 1, 0);
    s.Divergence(rhs, s.flux);
    return s.Solve(rhs);
}
InitialRateSchurResult
DarwinInitialRateSchur::CorrectWithIterationBudget (const ConstVector& raw,
    const ConstVector& held, int remaining_iterations) {
    auto& s = *m_impl;
    AMREX_ALWAYS_ASSERT(remaining_iterations >= 0 &&
                        remaining_iterations <= s.options.max_iterations);
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(raw[c] && held[c]);
        for (int d = 0; d < 3; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                raw[c] != &s.correction[d] && held[c] != &s.correction[d],
                "Correct inputs must be independent of owned output fields");
        }
    }
    amrex::MultiFab rhs(s.nodes, s.distribution, 1, 0);
    s.DivergenceDifference(rhs, raw, held);
    return s.Solve(rhs, remaining_iterations);
}
amrex::Real
DarwinInitialRateSchur::DivergenceResidualNorm (const ConstVector& raw,
                                              const ConstVector& held) {
    auto& s = *m_impl;
    AMREX_ALWAYS_ASSERT(s.frozen);
    amrex::MultiFab rhs(s.nodes, s.distribution, 1, 0);
    s.DivergenceDifference(rhs, raw, held);
    if (!rhs.is_finite(0, 1, 0)) {
        return std::numeric_limits<amrex::Real>::infinity();
    }
    Impl::Ops ops{s};
    return ops.norm2(rhs);
}
const amrex::MultiFab&
DarwinInitialRateSchur::CorrectionPotential () const {
    return m_impl->potential;
}
DarwinInitialRateSchur::ConstVector
DarwinInitialRateSchur::CorrectionField () const {
    return {&m_impl->correction[0], &m_impl->correction[1],
            &m_impl->correction[2]};
}
} // namespace warpx::darwin
#else
namespace warpx::darwin {
struct DarwinInitialRateSchur::Impl {};
DarwinInitialRateSchur::DarwinInitialRateSchur (
    const amrex::Geometry&, const amrex::BoxArray&,
    const amrex::DistributionMapping&, const InitialRateSchurOptions&) {
    amrex::Abort("InitialRate Schur supports RZ/3D only");
}
DarwinInitialRateSchur::~DarwinInitialRateSchur () = default;
bool
DarwinInitialRateSchur::Freeze (const amrex::MultiFab&) {
    return false;
}
void
DarwinInitialRateSchur::ApplyPotential (amrex::MultiFab&,
                                         const amrex::MultiFab&) {
    amrex::Abort("Unsupported geometry");
}
InitialRateSchurResult
DarwinInitialRateSchur::Correct (const ConstVector&, const ConstVector&) {
    return {};
}
InitialRateSchurResult
DarwinInitialRateSchur::SolvePotential (const amrex::MultiFab&) {
    return {};
}
InitialRateSchurResult
DarwinInitialRateSchur::CorrectWithIterationBudget (
    const ConstVector&, const ConstVector&, int) { return {}; }
amrex::Real
DarwinInitialRateSchur::DivergenceResidualNorm (
    const ConstVector&, const ConstVector&) {
    amrex::Abort("Unsupported geometry");
    return 0.;
}
const amrex::MultiFab&
DarwinInitialRateSchur::CorrectionPotential () const {
    amrex::Abort("Unsupported geometry");
    throw 0;
}
DarwinInitialRateSchur::ConstVector
DarwinInitialRateSchur::CorrectionField () const {
    return {};
}
} // namespace warpx::darwin
#endif
