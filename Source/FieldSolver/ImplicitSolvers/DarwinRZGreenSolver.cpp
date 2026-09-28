/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "DarwinRZGreenSolver.H"

#include "Utils/TextMsg.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
#include <AMReX_FFT_R2C.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <utility>

struct DarwinRZGreenSolver::Impl
{
    using Complex = amrex::GpuComplex<amrex::Real>;
    using ComplexField = amrex::FabArray<amrex::BaseFab<Complex>>;
    int m_nr, m_nz, m_length, m_mode_count, m_radial_start, m_radial_count;
    bool m_pmc_zlo, m_pmc_zhi, m_radial_neumann;
    amrex::MultiFab m_nodes, m_extended, m_factors;
    ComplexField m_modes;
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_fft;
    std::unique_ptr<Impl> m_axial, m_radial;

    explicit Impl (amrex::Geometry const &geom, bool pmc_zlo, bool pmc_zhi,
                   bool vector_component = true, bool radial_neumann = false)
        : m_nr(geom.Domain().length(0)), m_nz(geom.Domain().length(1)),
          m_length((pmc_zlo == pmc_zhi ? 2 : 4) * m_nz), m_mode_count(m_length / 2 + 1),
          m_radial_start(vector_component ? 1 : 0),
          m_radial_count(m_nr - m_radial_start + int(radial_neumann)),
          m_pmc_zlo(pmc_zlo), m_pmc_zhi(pmc_zhi), m_radial_neumann(radial_neumann)
    {
        ABLASTR_PROFILE("DarwinRZGreenSolver::Setup()");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            geom.ProbLo(0) == 0. && geom.Domain().smallEnd() == amrex::IntVect(0) &&
                !geom.isPeriodic(0) && !geom.isPeriodic(1) && m_nr > 1 && m_nz > 1,
            "RZ greens recovery requires an r=0 axis, zero-based grid and "
            "nonperiodic r/z");

        // RZ is selected by WarpX's compile-time geometry. Its AMReX
        // Geometry may retain Coord()==0; the radial metric is explicit here.
#ifdef AMREX_USE_GPU
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::ParallelContext::NProcsSub() == 1 ||
                                             amrex::ParallelDescriptor::UseGpuAwareMpi(),
                                         "Distributed GPU greens recovery requires GPU-aware MPI");
#endif
        // Complete axial lines, distributed by radius. Matching indices in
        // these two BoxArrays permit a local GPU transpose/odd extension.
        int const ranks = amrex::ParallelContext::NProcsSub();
        int const radial_count = m_radial_count;
        int const radial_parts = std::min(ranks, radial_count);
        amrex::BoxList node_boxes(amrex::IndexType(amrex::IntVect(1)));
        amrex::BoxList real_boxes;
        amrex::Vector<int> radial_owners;
        for (int p = 0; p < radial_parts; ++p) {
            int const lo = m_radial_start + p * radial_count / radial_parts;
            int const hi = m_radial_start + (p + 1) * radial_count / radial_parts - 1;
            node_boxes.push_back(
                amrex::Box(amrex::IntVect(lo, 0), amrex::IntVect(hi, m_nz), amrex::IntVect(1)));
            real_boxes.push_back(
                amrex::Box(amrex::IntVect(0, lo - m_radial_start),
                           amrex::IntVect(m_length - 1, hi - m_radial_start)));
            radial_owners.push_back(amrex::ParallelContext::local_to_global_rank(p));
        }
        amrex::DistributionMapping radial_dm(radial_owners);
        m_nodes.define(amrex::BoxArray(std::move(node_boxes)), radial_dm, 1, 0);
        m_extended.define(amrex::BoxArray(std::move(real_boxes)), radial_dm, 1, 0);

        // Complete radial lines per axial mode. These are globally coupled
        // through FFT redistribution, never independent conducting tile walls.
        int const mode_parts = std::min(ranks, m_mode_count);
        amrex::BoxList mode_boxes;
        amrex::Vector<int> mode_owners;
        for (int p = 0; p < mode_parts; ++p) {
            int const lo = p * m_mode_count / mode_parts;
            int const hi = (p + 1) * m_mode_count / mode_parts - 1;
            mode_boxes.push_back(amrex::Box(amrex::IntVect(lo, 0), amrex::IntVect(hi, radial_count - 1)));
            mode_owners.push_back(amrex::ParallelContext::local_to_global_rank(p));
        }
        amrex::BoxArray mode_ba(std::move(mode_boxes));
        amrex::DistributionMapping mode_dm(mode_owners);
        m_modes.define(mode_ba, mode_dm, 1, 0);
        m_factors.define(mode_ba, mode_dm, 3, 0);
        BuildFactors(geom.CellSize(0), geom.CellSize(1));
        amrex::FFT::Info info;
        info.setOneDMode(true);
        m_fft = std::make_unique<amrex::FFT::R2C<amrex::Real>>(
            amrex::Box(amrex::IntVect(0), amrex::IntVect(m_length - 1, radial_count - 1)), info);
        amrex::Gpu::streamSynchronize();
        amrex::Print() << "[darwin] cached bounded RZ Green response: " << m_nr << "x" << m_nz
                       << (vector_component ? ", vector" : ", scalar")
                       << ", outer radial=" << (m_radial_neumann ? "Neumann" : "Dirichlet")
                       << ", nonperiodic z=" << (m_pmc_zlo ? "Neumann" : "Dirichlet") << "/"
                       << (m_pmc_zhi ? "Neumann" : "Dirichlet") << ", " << ranks << " ranks\n";
    }

    // NVCC extended device lambdas need an addressable parent function,
    // so initialization kernels live here rather than in the constructor.
    void
    BuildFactors (amrex::Real const dr, amrex::Real const dz)
    {
        amrex::Real const pi = std::acos(-1.);
        int const radial_start = m_radial_start;
        int const radial_count = m_radial_count, transform_length = m_length;
        int const outer_node = m_nr;
        bool const radial_neumann = m_radial_neumann;
        for (amrex::MFIter mfi(m_factors); mfi.isValid(); ++mfi) {
            auto const f = m_factors.array(mfi);
            amrex::Box lines = mfi.validbox();
            lines.setBig(1, 0);
            amrex::ParallelFor(lines,
                               [=] AMREX_GPU_DEVICE(int mode, int, int)
                               {
                                   amrex::Real const sine = std::sin(pi * mode / transform_length);
                                   amrex::Real const lambda = 4. * sine * sine / (dz * dz);
                                   amrex::Real previous_inverse = 0., previous_upper = 0.;
                                   for (int ri = 0; ri < radial_count; ++ri)
                                   {
                                       int const radial_node = ri + radial_start;
                                       amrex::Real const r = radial_node * dr;
                                       // The alpha=0 scalar axis is a free row:
                                       // L_r u(0)=4*(u(1)-u(0))/dr^2. Vector
                                       // components exclude their zero axis trace.
                                       bool const axis = radial_node == 0;
                                       bool const outer = radial_neumann && radial_node == outer_node;
                                       // Native nodal Neumann reflects u(N+1)=u(N-1).
                                       // The two radial coefficients add to 2/dr^2;
                                       // the cylindrical first derivative cancels.
                                       amrex::Real const lower = axis ? 0. : outer ?
                                           -2. / (dr * dr) : -(1. - .5 * dr / r) / (dr * dr);
                                       amrex::Real const upper = outer ? 0. : axis ?
                                           -4. / (dr * dr) : -(1. + .5 * dr / r) / (dr * dr);
                                       amrex::Real const metric = radial_start ? 1. / (r * r) : 0.;
                                       amrex::Real const multiplier = lower * previous_inverse;
                                       amrex::Real const diagonal = (axis ? 4. : 2.) / (dr * dr) +
                                           metric + lambda - multiplier * previous_upper;
                                       f(mode, ri, 0, 0) = 1. / diagonal;
                                       f(mode, ri, 0, 1) = multiplier;
                                       f(mode, ri, 0, 2) = upper;
                                       previous_inverse = 1. / diagonal;
                                       previous_upper = upper;
                                   }
                               });
        }
    }

    void
    Solve (amrex::MultiFab& solution, amrex::MultiFab const& rhs)
    {
        ABLASTR_PROFILE("DarwinRZGreenSolver::Solve()");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rhs.ixType().nodeCentered() &&
                                             solution.ixType().nodeCentered() && rhs.nComp() == 1 &&
                                             solution.nComp() == 1,
                                         "RZ greens recovery requires one nodal component");
        m_nodes.ParallelCopy(rhs, 0, 0, 1);
        int const radial_start = m_radial_start;
        int const axial_count = m_nz, radial_count = m_radial_count;
        bool const pmc_lo = m_pmc_zlo, pmc_hi = m_pmc_zhi;
        bool const mixed = pmc_lo != pmc_hi;
        for (amrex::MFIter mfi(m_extended); mfi.isValid(); ++mfi) {
            auto const f = m_nodes.const_array(mfi.index());
            auto const e = m_extended.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int z, int ri, int k) {
                // K=-L is positive; the requested L u=rhs needs -rhs.
                int const half = z / (2 * axial_count);
                int const index = z % (2 * axial_count);
                int const source = index <= axial_count ? index : 2 * axial_count - index;
                amrex::Real sign = index <= axial_count || pmc_hi ? 1. : -1.;
                if (mixed && half)
                {
                    sign = -sign;
                }
                bool const constrained =
                    (source == 0 && !pmc_lo) || (source == axial_count && !pmc_hi);
                e(z, ri, k) = constrained ? 0. : -sign * f(ri + radial_start, source, k);
            });
        }
        m_fft->forward(m_extended, m_modes);
        for (amrex::MFIter mfi(m_modes); mfi.isValid(); ++mfi) {
            auto const u = m_modes.array(mfi);
            auto const f = m_factors.const_array(mfi);
            amrex::Box lines = mfi.validbox();
            lines.setBig(1, 0);
            amrex::ParallelFor(lines,
                               [=] AMREX_GPU_DEVICE(int mode, int, int)
                               {
                                   // Odd/odd removes the endpoint modes; mixed parity has only
                                   // odd Fourier modes on its doubled extension. Even/even keeps
                                   // the constant and Nyquist axial modes. Each radial
                                   // block is anchored by Dirichlet or the alpha/r^2 term.
                                   if ((!pmc_lo && !pmc_hi && (mode == 0 || mode == axial_count)) ||
                                       (mixed && mode % 2 == 0))
                                   {
                                       for (int ri = 0; ri < radial_count; ++ri)
                                       {
                                           u(mode, ri, 0) = Complex(0., 0.);
                                       }
                                       return;
                                   }
                                   for (int ri = 1; ri < radial_count; ++ri)
                                   {
                                       u(mode, ri, 0) -= f(mode, ri, 0, 1) * u(mode, ri - 1, 0);
                                   }
                                   u(mode, radial_count - 1, 0) *= f(mode, radial_count - 1, 0, 0);
                                   for (int ri = radial_count - 2; ri >= 0; --ri)
                                   {
                                       u(mode, ri, 0) = (u(mode, ri, 0) -
                                                         f(mode, ri, 0, 2) * u(mode, ri + 1, 0)) *
                                                        f(mode, ri, 0, 0);
                                   }
                               });
        }
        m_fft->backward(m_modes, m_extended);
        amrex::Real const normalization = m_fft->scalingFactor();
        for (amrex::MFIter mfi(m_nodes); mfi.isValid(); ++mfi) {
            auto const n = m_nodes.array(mfi);
            auto const e = m_extended.const_array(mfi.index());
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int r, int z, int k)
                               {
                                   n(r, z, k) =
                                       ((z == 0 && !pmc_lo) || (z == axial_count && !pmc_hi))
                                           ? 0.
                                           : normalization * e(z, r - radial_start, k);
                               });
        }
        // Fixed traces are excluded. The scalar axis and optional radial
        // Neumann outer trace are solved rows. Reset all other values and ghosts,
        // including after a zero RHS, before publishing the cached pencils.
        solution.setVal(0.);
        solution.ParallelCopy(m_nodes, 0, 0, 1);
    }
};
#else
struct DarwinRZGreenSolver::Impl
{
};
#endif

DarwinRZGreenSolver::DarwinRZGreenSolver (amrex::Geometry const &geometry, bool pmc_zlo,
                                          bool pmc_zhi)
    : DarwinRZGreenSolver(geometry, pmc_zlo, pmc_zhi, false)
{}

DarwinRZGreenSolver::DarwinRZGreenSolver (amrex::Geometry const &geometry, bool pmc_zlo,
                                       bool pmc_zhi, bool all_components)
    : DarwinRZGreenSolver(geometry, pmc_zlo, pmc_zhi, all_components,
                         BoundaryFamily::VectorPotentialRecovery)
{}

DarwinRZGreenSolver::DarwinRZGreenSolver (amrex::Geometry const &geometry, bool pmc_zlo,
                                       bool pmc_zhi, bool all_components,
                                       BoundaryFamily boundary_family)
{
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
    bool const electric_auxiliary = boundary_family == BoundaryFamily::ElectricAuxiliary;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!electric_auxiliary || all_components,
        "Electric auxiliary Green inverse requires all-component initialization");
    m_impl = std::make_unique<Impl>(geometry, pmc_zlo, pmc_zhi);
    if (all_components) {
        // Recovery A_z stays Dirichlet at both caps. The electric Hodge
        // auxiliary has the opposite normal/tangential parity at a PEC cap.
        m_impl->m_axial = std::make_unique<Impl>(geometry,
            electric_auxiliary && !pmc_zlo, electric_auxiliary && !pmc_zhi, false);
    }
    if (electric_auxiliary) {
        m_impl->m_radial = std::make_unique<Impl>(geometry, pmc_zlo, pmc_zhi, true, true);
    }
#else
    amrex::ignore_unused(geometry, pmc_zlo, pmc_zhi, all_components, boundary_family);
    WARPX_ABORT_WITH_MESSAGE("RZ greens recovery requires RZ and AMReX FFT "
                             "support (ABLASTR_FFT=ON)");
#endif
}

DarwinRZGreenSolver::~DarwinRZGreenSolver () = default;

void
DarwinRZGreenSolver::Solve (amrex::MultiFab& solution, amrex::MultiFab const& rhs)
{
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
    m_impl->Solve(solution, rhs);
#else
    amrex::ignore_unused(solution, rhs);
    WARPX_ABORT_WITH_MESSAGE("RZ greens recovery requires RZ and AMReX FFT "
                             "support (ABLASTR_FFT=ON)");
#endif
}

void
DarwinRZGreenSolver::SolveComponent (amrex::MultiFab& solution, amrex::MultiFab const& rhs,
                                    int component)
{
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(component >= 0 && component < 3,
                                     "RZ Green component must be r, theta or z");
    if (component == 0 && m_impl->m_radial) {
        m_impl->m_radial->Solve(solution, rhs);
    } else if (component < 2) {
        m_impl->Solve(solution, rhs);
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_impl->m_axial != nullptr,
            "Axial Green recovery requires all-component initialization");
        m_impl->m_axial->Solve(solution, rhs);
    }
#else
    amrex::ignore_unused(solution, rhs, component);
    WARPX_ABORT_WITH_MESSAGE("RZ greens recovery requires RZ and AMReX FFT "
                             "support (ABLASTR_FFT=ON)");
#endif
}
