/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "DarwinRZYeeGreenSolver.H"

#include "Utils/TextMsg.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
#include <AMReX_FFT_R2C.H>
#include <AMReX_ParallelContext.H>

#include <algorithm>
#include <cmath>
#include <utility>

struct DarwinRZYeeGreenSolver::Impl
{
    using Complex = amrex::GpuComplex<amrex::Real>;
    using ComplexField = amrex::FabArray<amrex::BaseFab<Complex>>;
    int m_component, m_nr, m_nz, m_radial_start, m_radial_count;
    amrex::IntVect m_type;
    amrex::MultiFab m_pencils, m_extended, m_factors;
    ComplexField m_modes;
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> m_fft;

    Impl (amrex::Geometry const& geom, int component)
        : m_component(component), m_nr(geom.Domain().length(0)),
          m_nz(geom.Domain().length(1)), m_radial_start(component == 1 ? 1 : 0),
          m_radial_count(m_nr - m_radial_start),
          m_type(component == 0 ? 0 : 1, component == 2 ? 0 : 1)
    {
        ABLASTR_PROFILE("DarwinRZYeeGreenSolver::Setup()");
        int const ranks = amrex::ParallelContext::NProcsSub();
        int const radial_parts = std::min(ranks, m_radial_count);
        int const length = 2 * m_nz;
        amrex::BoxList pencil_boxes{amrex::IndexType(m_type)};
        amrex::BoxList real_boxes;
        amrex::Vector<int> radial_owners;
        for (int p = 0; p < radial_parts; ++p) {
            int const lo = m_radial_start + p * m_radial_count / radial_parts;
            int const hi = m_radial_start + (p + 1) * m_radial_count / radial_parts - 1;
            pencil_boxes.push_back(amrex::Box(
                amrex::IntVect(lo, 0), amrex::IntVect(hi, m_nz - (component == 2)), m_type));
            real_boxes.push_back(amrex::Box(
                amrex::IntVect(0, lo - m_radial_start),
                amrex::IntVect(length - 1, hi - m_radial_start)));
            radial_owners.push_back(amrex::ParallelContext::local_to_global_rank(p));
        }
        amrex::DistributionMapping radial_dm(radial_owners);
        m_pencils.define(amrex::BoxArray(std::move(pencil_boxes)), radial_dm, 1, 0);
        m_extended.define(amrex::BoxArray(std::move(real_boxes)), radial_dm, 1, 0);

        int const modes = m_nz + 1;
        int const mode_parts = std::min(ranks, modes);
        amrex::BoxList mode_boxes;
        amrex::Vector<int> mode_owners;
        for (int p = 0; p < mode_parts; ++p) {
            int const lo = p * modes / mode_parts;
            int const hi = (p + 1) * modes / mode_parts - 1;
            mode_boxes.push_back(amrex::Box(
                amrex::IntVect(lo, 0), amrex::IntVect(hi, m_radial_count - 1)));
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
            amrex::Box(amrex::IntVect(0), amrex::IntVect(length - 1, m_radial_count - 1)),
            info);
        amrex::Gpu::streamSynchronize();
    }

    void BuildFactors (amrex::Real dr, amrex::Real dz)
    {
        int const component = m_component, nr = m_nr, nz = m_nz;
        int const start = m_radial_start, count = m_radial_count;
        amrex::Real const pi = std::acos(-1.);
        for (amrex::MFIter it(m_factors); it.isValid(); ++it) {
            auto const factors = m_factors.array(it);
            amrex::Box lines = it.validbox();
            lines.setBig(1, 0);
            amrex::ParallelFor(lines, [=] AMREX_GPU_DEVICE (int mode, int, int) {
                amrex::Real const sine = std::sin(pi * mode / (2 * nz));
                amrex::Real const axial = 4. * sine * sine / (dz * dz);
                amrex::Real previous_inverse = 0., previous_upper = 0.;
                for (int q = 0; q < count; ++q) {
                    int const index = q + start;
                    amrex::Real const i = index;
                    amrex::Real lower, upper, diagonal;
                    if (component == 0) {
                        // grad_r div_r on radial cell centers. The regular scalar
                        // axis has measure 1/8; the outer scalar PEC trace is fixed.
                        lower = index == 0 ? 0. : -(i - .5) / i;
                        upper = index == nr - 1 ? 0. : -(i + 1.5) / (i + 1.);
                        diagonal = (index == 0 ? 4. : (i + .5) / i) +
                            (index == nr - 1 ? 0. : (i + .5) / (i + 1.));
                    } else if (component == 1) {
                        // Native Yee curl of the radial face B_z curl. This is
                        // not the nodal scalar Laplacian minus a sampled 1/r^2.
                        lower = -(i - 1.) / (i - .5);
                        upper = -(i + 1.) / (i + .5);
                        diagonal = i * (1. / (i - .5) + 1. / (i + .5));
                    } else {
                        lower = index == 0 ? 0. : -(1. - .5 / i);
                        upper = index == 0 ? -4. : -(1. + .5 / i);
                        diagonal = index == 0 ? 4. : 2.;
                    }
                    lower /= dr * dr;
                    upper /= dr * dr;
                    diagonal = diagonal / (dr * dr) + axial;
                    amrex::Real const multiplier = lower * previous_inverse;
                    diagonal -= multiplier * previous_upper;
                    factors(mode, q, 0, 0) = 1. / diagonal;
                    factors(mode, q, 0, 1) = multiplier;
                    factors(mode, q, 0, 2) = upper;
                    previous_inverse = 1. / diagonal;
                    previous_upper = upper;
                }
            });
        }
    }

    void Solve (amrex::MultiFab& solution, amrex::MultiFab const& rhs)
    {
        ABLASTR_PROFILE("DarwinRZYeeGreenSolver::Solve()");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            rhs.ixType().toIntVect() == m_type && solution.ixType().toIntVect() == m_type &&
                rhs.nComp() == 1 && solution.nComp() == 1,
            "RZ Yee Green inverse requires the native one-component electric staggering");
        m_pencils.ParallelCopy(rhs, 0, 0, 1);
        int const start = m_radial_start, nz = m_nz, count = m_radial_count;
        bool const axial_component = m_component == 2;
        for (amrex::MFIter it(m_extended); it.isValid(); ++it) {
            auto const rhs_pencil = m_pencils.const_array(it.index());
            auto const extended = m_extended.array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int z, int r, int k) {
                int const source = axial_component ?
                    (z < nz ? z : 2 * nz - 1 - z) : (z <= nz ? z : 2 * nz - z);
                amrex::Real const sign = axial_component && z >= nz ? -1. : 1.;
                extended(z, r, k) = sign * rhs_pencil(r + start, source, k);
            });
        }
        m_fft->forward(m_extended, m_modes);
        for (amrex::MFIter it(m_modes); it.isValid(); ++it) {
            auto const u = m_modes.array(it);
            auto const f = m_factors.const_array(it);
            amrex::Box lines = it.validbox();
            lines.setBig(1, 0);
            amrex::ParallelFor(lines, [=] AMREX_GPU_DEVICE (int mode, int, int) {
                // Odd cell-centered reflection removes only the constant mode.
                // The Nyquist mode is a physical alternating Yee mode and stays.
                if (axial_component && mode == 0) {
                    for (int r = 0; r < count; ++r) {
                        u(mode, r, 0) = Complex(0., 0.);
                    }
                    return;
                }
                for (int r = 1; r < count; ++r) {
                    u(mode, r, 0) -= f(mode, r, 0, 1) * u(mode, r - 1, 0);
                }
                u(mode, count - 1, 0) *= f(mode, count - 1, 0, 0);
                for (int r = count - 2; r >= 0; --r) {
                    u(mode, r, 0) = (u(mode, r, 0) -
                        f(mode, r, 0, 2) * u(mode, r + 1, 0)) * f(mode, r, 0, 0);
                }
            });
        }
        m_fft->backward(m_modes, m_extended);
        amrex::Real const normalization = m_fft->scalingFactor();
        for (amrex::MFIter it(m_pencils); it.isValid(); ++it) {
            auto const out = m_pencils.array(it);
            auto const extended = m_extended.const_array(it.index());
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int r, int z, int k) {
                out(r, z, k) = normalization * extended(z, r - start, k);
            });
        }
        solution.setVal(0.);
        solution.ParallelCopy(m_pencils, 0, 0, 1);
    }
};
#else
struct DarwinRZYeeGreenSolver::Impl {};
#endif

DarwinRZYeeGreenSolver::DarwinRZYeeGreenSolver (
    amrex::Geometry const& geometry, bool pmc_zlo, bool pmc_zhi)
{
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.ProbLo(0) == 0. && geometry.Domain().smallEnd() == amrex::IntVect(0) &&
            !geometry.isPeriodic(0) && !geometry.isPeriodic(1) &&
            geometry.Domain().length(0) > 1 && geometry.Domain().length(1) > 1 &&
            pmc_zlo && pmc_zhi,
        "RZ Yee Green inverse currently requires a regular axis and two PMC axial caps");
#ifdef AMREX_USE_GPU
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::ParallelContext::NProcsSub() == 1 ||
            amrex::ParallelDescriptor::UseGpuAwareMpi(),
        "Distributed RZ Yee Green inverse requires GPU-aware MPI");
#endif
    for (int component = 0; component < 3; ++component) {
        m_components[component] = std::make_unique<Impl>(geometry, component);
    }
#else
    amrex::ignore_unused(geometry, pmc_zlo, pmc_zhi);
    WARPX_ABORT_WITH_MESSAGE("RZ Yee Green inverse requires RZ and AMReX FFT support");
#endif
}

DarwinRZYeeGreenSolver::~DarwinRZYeeGreenSolver () = default;

void DarwinRZYeeGreenSolver::SolveComponent (
    amrex::MultiFab& solution, amrex::MultiFab const& rhs, int component)
{
#if defined(WARPX_DIM_RZ) && defined(AMREX_USE_FFT)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(component >= 0 && component < 3,
                                     "Invalid RZ Yee Green component");
    m_components[component]->Solve(solution, rhs);
#else
    amrex::ignore_unused(solution, rhs, component);
    WARPX_ABORT_WITH_MESSAGE("RZ Yee Green inverse requires RZ and AMReX FFT support");
#endif
}
