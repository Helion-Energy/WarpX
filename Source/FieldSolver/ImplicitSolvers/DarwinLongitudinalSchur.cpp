/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "DarwinLongitudinalSchur.H"
#include "DarwinPoissonMG.H"
#include "DarwinInitialRateSchur.H"

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
#include "DarwinABoundary.H"
#if defined(WARPX_DIM_RZ)
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include "NonlinearSolvers/FlexibleGMRES.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
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
using Boundary = LongitudinalBoundary;
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

struct DarwinLongitudinalSchur::Impl {
    amrex::Geometry geometry;
    amrex::BoxArray cells, nodes;
    amrex::DistributionMapping distribution;
    LongitudinalSchurOptions options;
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{}, dir_lo{}, dir_hi{};
    amrex::MultiFab kappa, potential, scratch_phi, nodal_response, weights,
        pc_sigma;
    std::array<amrex::MultiFab, 3> gradient, flux, correction, edge_kappa;
    bool edge_coefficients = false;
    std::unique_ptr<amrex::MLEBNodeFDLaplacian> pc_operator;
    std::unique_ptr<amrex::MLMG> pc;
    bool singular = true, frozen = false;
    bool batched_orthogonalization = false;
    Real weight_sum = 0;

    Impl (const amrex::Geometry& g, const amrex::BoxArray& ba,
          const amrex::DistributionMapping& dm,
          const LongitudinalSchurOptions& o)
        : geometry(g), cells(ba), nodes(amrex::convert(ba, amrex::IntVect(1))),
          distribution(dm), options(o), kappa(nodes, dm, 1, 1),
          potential(nodes, dm, 1, o.output_ghosts),
          scratch_phi(nodes, dm, 1, 1), nodal_response(nodes, dm, 3, 1),
          weights(nodes, dm, 1, 0), pc_sigma(ba, dm, 1, 0) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            ba.ixType().cellCentered() &&
                ba.minimalBox() == geometry.Domain() && ba.isDisjoint() &&
                ba.numPts() == geometry.Domain().numPts(),
            "Longitudinal Schur requires one fully covered rectangular level");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            o.theta == .5 && o.djedt_only && !o.bdf2 &&
                !o.embedded_boundaries && o.azimuthal_modes == 1,
            "Longitudinal Schur qualifies midpoint dJe/dt-only, BDF2 off, no "
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
            "Longitudinal Schur RZ requires an axis-containing domain");
#else
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            g.Coord() == 0,
            "Cartesian longitudinal Schur requires Cartesian geometry");
#endif
        // Default off until the same physical roots and backend controls pass.
        int batch = 0;
        amrex::ParmParse("implicit_evolve").query("darwin_scalar_cgs2", batch);
        int batch_min = batch, batch_max = batch;
        amrex::ParallelDescriptor::ReduceIntMin(batch_min);
        amrex::ParallelDescriptor::ReduceIntMax(batch_max);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            batch_min == batch_max && (batch == 0 || batch == 1),
            "darwin_scalar_cgs2 must be 0 or 1 and agree across ranks");
        batched_orthogonalization = batch == 1;
        amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lo{}, hi{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                o.output_ghosts[d] <= g.Domain().length(d),
                "Longitudinal Schur output guards exceed physical domain");
            for (int side = 0; side < 2; ++side) {
                auto b = side ? o.upper[d] : o.lower[d];
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (b == Boundary::Periodic) == g.isPeriodic(d),
                    "Longitudinal Schur periodic geometry/BC mismatch");
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
            edge_kappa[c].define(edges, dm, 1, 1);
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
            amrex::Print() << "[darwin] DarwinLongitudinalSchur PC: semicoarsening_levels="
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
            "Longitudinal Schur input layout mismatch");
    }
    Real
    Dot (const amrex::MultiFab& a, const amrex::MultiFab& b) const {
        BL_PROFILE("DarwinLongitudinalSchur::Dot");
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
        BL_PROFILE("DarwinLongitudinalSchur::Canonical");
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
        BL_PROFILE("DarwinLongitudinalSchur::Gradient");
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
        BL_PROFILE("DarwinLongitudinalSchur::Divergence");
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
    void
    Apply (amrex::MultiFab& out, const amrex::MultiFab& phi) {
        BL_PROFILE("DarwinLongitudinalSchur::Apply");
        AMREX_ALWAYS_ASSERT(frozen && &out != &phi);
        Layout(out, nodes);
        Layout(phi, nodes);
        Gradient(gradient, phi);
        if (edge_coefficients) {
            for (int c=0;c<3;++c) {
                for (amrex::MFIter mfi(flux[c]);mfi.isValid();++mfi) {
                    auto f=flux[c].array(mfi);auto g=gradient[c].const_array(mfi);
                    auto k=edge_kappa[c].const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int l) {
                        f(i,j,l)=k(i,j,l)*g(i,j,l);
                    });
                }
                amrex::MultiFab::Add(flux[c],gradient[c],0,0,1,0);
                Images(flux[c]);
            }
            Divergence(out,flux);
            return;
        }
        if (options.compatible_yee) {
            ApplyYeeInertiaMass(geometry, kappa,
                {&gradient[0], &gradient[1], &gradient[2]},
                {&flux[0], &flux[1], &flux[2]});
            for (int c = 0; c < 3; ++c) {
                amrex::MultiFab::Add(flux[c], gradient[c], 0, 0, 1, 0);
                Images(flux[c]);
            }
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
                auto const g = gradient[c].const_array(mfi);
                auto const n = nodal_response.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        f(i, j, k) = g(i, j, k) +
                                     Interp(n, node, edge, ratio, i, j, k, c);
                    });
            }
            Images(flux[c]);
        }
        Divergence(out, flux);
    }
    bool
    Freeze (const amrex::MultiFab& input) {
        BL_PROFILE("DarwinLongitudinalSchur::Freeze");
        Layout(input, nodes);
        frozen = false;
        edge_coefficients = false;
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
                    sigma(i, j, k) = 1 + sum / Real(1 << AMREX_SPACEDIM);
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
    bool
    FreezeEdges (const ConstVector& input) {
        BL_PROFILE("DarwinLongitudinalSchur::FreezeEdges");
        AMREX_ALWAYS_ASSERT(options.compatible_yee);
        for(int c=0;c<3;++c) {
            AMREX_ALWAYS_ASSERT(input[c]);Layout(*input[c],edge_kappa[c].boxArray());
            if(!input[c]->is_finite(0,1,0)||input[c]->min(0)<0.)return false;
        }
        // Only the multigrid preconditioner uses a nodal/cell surrogate.
        // The action below owns and uses each exact input edge coefficient.
        std::array<amrex::MultiFab,3> temporary;
        for(int c=0;c<3;++c) {
            temporary[c].define(edge_kappa[c].boxArray(),distribution,1,1);
            temporary[c].setVal(0.);
            amrex::MultiFab::Copy(temporary[c],*input[c],0,0,1,0);
            temporary[c].OverrideSync(geometry.periodicity());
            temporary[c].FillBoundary(geometry.periodicity());
        }
        amrex::MultiFab surrogate(nodes,distribution,1,0);
        for(amrex::MFIter mfi(surrogate);mfi.isValid();++mfi) {
            auto out=surrogate.array(mfi);
            amrex::GpuArray<amrex::Array4<amrex::Real const>,AMREX_SPACEDIM> edge{};
            for(int d=0;d<AMREX_SPACEDIM;++d) {
#if defined(WARPX_DIM_RZ)
                int const c=d==0?0:2;
#else
                int const c=d;
#endif
                edge[d]=temporary[c].const_array(mfi);
            }
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                amrex::IntVect q(AMREX_D_DECL(i,j,k));Real value=0.;
                for(int d=0;d<AMREX_SPACEDIM;++d) {
                    auto off=amrex::IntVect::TheDimensionVector(d);
                    value+=(edge[d](q)+edge[d](q-off))/(2.*AMREX_SPACEDIM);
                }
                out(q)=value;
            });
        }
        if(!surrogate.is_finite(0,1,0))return false;
        if(!Freeze(surrogate))return false;
        for(int c=0;c<3;++c)amrex::MultiFab::Copy(edge_kappa[c],temporary[c],0,0,1,1);
        edge_coefficients=true;
        return true;
    }
    struct Ops {
        using RT = Real;
        using Vec = amrex::MultiFab;
        Impl& self;
        static constexpr int batch_chunk = 8;
        static constexpr int dot_threads = 128;
        std::vector<int> dot_blocks, dot_offsets;
        int dot_block_count = 0;
        amrex::Gpu::DeviceVector<Real> dot_device;
        amrex::Gpu::PinnedVector<Real> dot_host;

        explicit Ops (Impl& owner) : self(owner) {}

        void dotProducts (Vec const& x, std::vector<Vec> const& basis,
                          int count, std::vector<RT>& result)
        {
            BL_PROFILE("DarwinLongitudinalSchur::BatchDot");
            AMREX_ALWAYS_ASSERT(count > 0 && count <= int(basis.size()));
            result.assign(count, RT(0));
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
            if (amrex::Gpu::inLaunchRegion()) {
                if (dot_blocks.empty()) {
                    for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi) {
                        int const blocks = int(std::min<amrex::Long>(
                            64, (mfi.validbox().numPts() + dot_threads - 1) / dot_threads));
                        AMREX_ALWAYS_ASSERT(dot_block_count <=
                                            std::numeric_limits<int>::max() - blocks);
                        dot_offsets.push_back(dot_block_count);
                        dot_blocks.push_back(blocks);
                        dot_block_count += blocks;
                    }
                }
                std::size_t const size = std::size_t(count) * dot_block_count;
                if (dot_device.size() < size) {
                    dot_device.resize(size);
                    dot_host.resize(size);
                }
                Real* partial = dot_device.data();
                int const total = dot_block_count;
                // The host consumes every FAB's partials below. Complete all
                // producing streams even if the caller is in a NoSyncRegion.
                for (amrex::MFIter mfi(x, amrex::MFItInfo().SetDeviceSync(true));
                     mfi.isValid(); ++mfi) {
                    auto const bx = mfi.validbox();
                    auto const a = x.const_array(mfi);
                    auto const weight = self.weights.const_array(mfi);
                    auto const lo = amrex::lbound(bx), len = amrex::length(bx);
                    amrex::Long const points = bx.numPts();
                    int const blocks = dot_blocks[mfi.LocalIndex()];
                    int const offset = dot_offsets[mfi.LocalIndex()];
                    // Small descriptor chunks avoid duplicating the whole Krylov
                    // basis or moving fields to the host. All chunks share one
                    // reduction transfer and one collective per CGS pass.
                    for (int first = 0; first < count; first += batch_chunk) {
                        int const slots = std::min(batch_chunk, count - first);
                        amrex::GpuArray<amrex::Array4<Real const>, batch_chunk> v{};
                        for (int n = 0; n < slots; ++n) {
                            v[n] = basis[first + n].const_array(mfi);
                        }
                        amrex::launch<dot_threads>(blocks, amrex::Gpu::gpuStream(),
                            [=] AMREX_GPU_DEVICE () noexcept {
                                int const tid = int(threadIdx.x);
                                amrex::Long const stride =
                                    amrex::Long(blockDim.x) * gridDim.x;
                                amrex::Long const start =
                                    amrex::Long(blockDim.x) * blockIdx.x + tid;
                                amrex::Long const plane = amrex::Long(len.x) * len.y;
                                Real sums[batch_chunk]{};
                                for (amrex::Long q = start; q < points; q += stride) {
                                    int const k = int(q / plane);
                                    amrex::Long const rem = q - amrex::Long(k) * plane;
                                    int const j = int(rem / len.x);
                                    int const i = int(rem - amrex::Long(j) * len.x);
                                    int const ii = i + lo.x, jj = j + lo.y, kk = k + lo.z;
                                    Real const value = weight(ii,jj,kk) * a(ii,jj,kk);
                                    for (int n = 0; n < slots; ++n) {
                                        sums[n] += value * v[n](ii,jj,kk);
                                    }
                                }
                                // Uniform control flow is required at every
                                // block reduction, including the final chunk.
                                for (int n = 0; n < batch_chunk; ++n) {
                                    Real const sum =
                                        amrex::Gpu::blockReduceSum<dot_threads>(sums[n]);
                                    if (tid == 0 && n < slots) {
                                        partial[std::size_t(first + n) * total +
                                                offset + blockIdx.x] = sum;
                                    }
                                }
                            });
                    }
                }
                if (size != 0) {
                    amrex::Gpu::dtoh_memcpy(dot_host.data(), partial, size * sizeof(Real));
                }
                for (int n = 0; n < count; ++n) {
                    for (int b = 0; b < total; ++b) {
                        result[n] += dot_host[std::size_t(n) * total + b];
                    }
                }
            } else
#endif
            {
                // CPU/SYCL fallback preserves the exact weighted nodal mask.
                // No global reduction is issued until all local dots are ready.
                for (int n = 0; n < count; ++n) {
                    amrex::ReduceOps<amrex::ReduceOpSum> op;
                    amrex::ReduceData<Real> data(op);
                    using Tuple = decltype(data)::Type;
                    for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi) {
                        auto const a = x.const_array(mfi);
                        auto const b = basis[n].const_array(mfi);
                        auto const weight = self.weights.const_array(mfi);
                        op.eval(mfi.validbox(), data,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple {
                                return {weight(i,j,k) * a(i,j,k) * b(i,j,k)};
                            });
                    }
                    result[n] = amrex::get<0>(data.value());
                }
            }
            amrex::ParallelDescriptor::ReduceRealSum(result.data(), count);
        }

        void subtractProjection (Vec& x, std::vector<Vec> const& basis,
                                 int count, std::vector<RT> const& coefficients) const
        {
            BL_PROFILE("DarwinLongitudinalSchur::BatchProjection");
            AMREX_ALWAYS_ASSERT(count > 0 && count <= int(basis.size()) &&
                                count == int(coefficients.size()));
            for (int first = 0; first < count; first += batch_chunk) {
                int const slots = std::min(batch_chunk, count - first);
                amrex::GpuArray<Real, batch_chunk> c{};
                for (int n = 0; n < slots; ++n) { c[n] = coefficients[first + n]; }
                for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi) {
                    amrex::GpuArray<amrex::Array4<Real const>, batch_chunk> v{};
                    for (int n = 0; n < slots; ++n) {
                        v[n] = basis[first + n].const_array(mfi);
                    }
                    auto const out = x.array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                            Real value = out(i,j,k);
                            for (int n = 0; n < slots; ++n) {
                                value -= c[n] * v[n](i,j,k);
                            }
                            out(i,j,k) = value;
                        });
                }
            }
        }

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
            return std::sqrt(std::max(RT(0), self.Dot(x, x)));
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
            BL_PROFILE("DarwinLongitudinalSchur::Precondition");
            out.setVal(0);
            self.pc->solve({&out}, {&rhs}, 0., 0.);
            self.Canonical(out, true);
        }
    };
    LongitudinalSchurResult
    Solve (const amrex::MultiFab& rhs) {
        BL_PROFILE_REGION("DarwinLongitudinalSchur");
        BL_PROFILE("DarwinLongitudinalSchur::Solve");
        AMREX_ALWAYS_ASSERT(frozen);
        Layout(rhs, nodes);
        amrex::MultiFab b(nodes, distribution, 1, 1),
            action(nodes, distribution, 1, 0);
        amrex::MultiFab::Copy(b, rhs, 0, 0, 1, 0);
        Canonical(b, false);
        Ops ops{*this};
        LongitudinalSchurResult result;
        result.initial_residual = ops.norm2(b);
        result.target =
            std::max(options.absolute_tolerance,
                     options.relative_tolerance * result.initial_residual);
        if (!rhs.is_finite(0, 1, 0)) {
            result.residual = std::numeric_limits<Real>::infinity();
            return result;
        }
        int status = -1;
        auto solve = [&](auto& solver) {
            solver.define(ops);
            solver.setVerbose(options.verbose);
            solver.setMaxIters(options.max_iterations);
            solver.setRestartLength(options.restart_length);
            solver.solve(potential, b, options.relative_tolerance,
                         options.absolute_tolerance);
            result.iterations = solver.getNumIters();
            status = solver.getStatus();
        };
        if (batched_orthogonalization) {
            FlexibleGMRES<amrex::MultiFab, Ops, true> solver;
            solve(solver);
        } else {
            FlexibleGMRES<amrex::MultiFab, Ops> solver;
            solve(solver);
        }
        Canonical(potential, true);
        Apply(action, potential);
        amrex::MultiFab::Subtract(action, b, 0, 0, 1, 0);
        result.residual = ops.norm2(action);
        result.converged = status == 0 &&
                           std::isfinite(result.residual) &&
                           result.residual <= result.target;
        Gradient(correction, potential);
        return result;
    }
};
DarwinLongitudinalSchur::DarwinLongitudinalSchur (
    const amrex::Geometry& g, const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm, const LongitudinalSchurOptions& o)
    : m_impl(std::make_unique<Impl>(g, ba, dm, o)) {}
DarwinLongitudinalSchur::~DarwinLongitudinalSchur () = default;
bool
DarwinLongitudinalSchur::Freeze (const amrex::MultiFab& k) {
    return m_impl->Freeze(k);
}
bool
DarwinLongitudinalSchur::FreezeEdges (const ConstVector& kappa) {
    return m_impl->FreezeEdges(kappa);
}
void
DarwinLongitudinalSchur::ApplyPotential (amrex::MultiFab& out,
                                         const amrex::MultiFab& phi) {
    m_impl->Apply(out, phi);
}
void
DarwinLongitudinalSchur::ApplyDivergence (amrex::MultiFab& out, const ConstVector& field)
{
    auto& s = *m_impl;
    s.Layout(out, s.nodes);
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(field[c]);
        s.Layout(*field[c], s.flux[c].boxArray());
        amrex::MultiFab::Copy(s.flux[c], *field[c], 0, 0, 1, 0);
        s.Images(s.flux[c]);
    }
    s.Divergence(out, s.flux);
}
bool
DarwinLongitudinalSchur::PreconditionPotential (amrex::MultiFab& out,
                                               const amrex::MultiFab& rhs)
{
    auto& s = *m_impl;
    AMREX_ALWAYS_ASSERT(s.frozen && &out != &rhs);
    s.Layout(out, s.nodes);
    s.Layout(rhs, s.nodes);
    if (!rhs.is_finite(0, 1, 0)) { return false; }
    amrex::MultiFab::Copy(s.scratch_phi, rhs, 0, 0, 1, 0);
    s.Canonical(s.scratch_phi, false);
    Impl::Ops ops{s};
    ops.precond(out, s.scratch_phi);
    return out.is_finite(0, 1, 0);
}
LongitudinalSchurResult
DarwinLongitudinalSchur::SolvePotential (const amrex::MultiFab& rhs) {
    return m_impl->Solve(rhs);
}
LongitudinalSchurResult
DarwinLongitudinalSchur::Correct (const ConstVector& raw,
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
LongitudinalSchurResult
DarwinLongitudinalSchur::CorrectDefect (const ConstVector& defect) {
    auto& s=*m_impl;
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(defect[c]);
        for(int d=0;d<3;++d)AMREX_ALWAYS_ASSERT(defect[c]!=&s.correction[d]);
        s.Layout(*defect[c],s.flux[c].boxArray());
        amrex::MultiFab::Copy(s.flux[c],*defect[c],0,0,1,0);
        s.Images(s.flux[c]);
    }
    amrex::MultiFab rhs(s.nodes,s.distribution,1,0);
    s.Divergence(rhs,s.flux);
    return s.Solve(rhs);
}

const amrex::MultiFab&
DarwinLongitudinalSchur::CorrectionPotential () const {
    return m_impl->potential;
}
DarwinLongitudinalSchur::ConstVector
DarwinLongitudinalSchur::CorrectionField () const {
    return {&m_impl->correction[0], &m_impl->correction[1],
            &m_impl->correction[2]};
}
} // namespace warpx::darwin
#else
namespace warpx::darwin {
struct DarwinLongitudinalSchur::Impl {};
DarwinLongitudinalSchur::DarwinLongitudinalSchur (
    const amrex::Geometry&, const amrex::BoxArray&,
    const amrex::DistributionMapping&, const LongitudinalSchurOptions&) {
    amrex::Abort("Longitudinal Schur supports RZ/3D only");
}
DarwinLongitudinalSchur::~DarwinLongitudinalSchur () = default;
bool
DarwinLongitudinalSchur::Freeze (const amrex::MultiFab&) {
    return false;
}
bool
DarwinLongitudinalSchur::FreezeEdges (const ConstVector&) {
    return false;
}

void
DarwinLongitudinalSchur::ApplyPotential (amrex::MultiFab&,
                                         const amrex::MultiFab&) {
    amrex::Abort("Unsupported geometry");
}
LongitudinalSchurResult
DarwinLongitudinalSchur::Correct (const ConstVector&, const ConstVector&) {
    return {};
}
LongitudinalSchurResult
DarwinLongitudinalSchur::CorrectDefect (const ConstVector&) { return {}; }
void
DarwinLongitudinalSchur::ApplyDivergence (amrex::MultiFab&, const ConstVector&)
{
    amrex::Abort("Unsupported geometry");
}
bool
DarwinLongitudinalSchur::PreconditionPotential (amrex::MultiFab&, const amrex::MultiFab&)
{
    return false;
}
LongitudinalSchurResult
DarwinLongitudinalSchur::SolvePotential (const amrex::MultiFab&) {
    return {};
}
const amrex::MultiFab&
DarwinLongitudinalSchur::CorrectionPotential () const {
    amrex::Abort("Unsupported geometry");
    throw 0;
}
DarwinLongitudinalSchur::ConstVector
DarwinLongitudinalSchur::CorrectionField () const {
    return {};
}
} // namespace warpx::darwin
#endif
