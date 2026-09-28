/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/DarwinInitialRateSchur.H"

#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using Real = amrex::Real;
        using BC = warpx::darwin::InitialRateBoundary;
        using Vector = warpx::darwin::DarwinInitialRateSchur::ConstVector;
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
        amrex::RealBox physical({AMREX_D_DECL(0., 0., 0.)},
                               {AMREX_D_DECL(1., 1., 1.)});
        std::array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 1, 1)};
#if defined(WARPX_DIM_RZ)
        int const coordinate = 1;
#else
        int const coordinate = 0;
#endif
        amrex::Geometry geometry(domain, &physical, coordinate, periodic.data());
        amrex::BoxArray cells(domain);
        cells.maxSize(4);
        amrex::DistributionMapping distribution(cells);
        warpx::darwin::InitialRateSchurOptions options;
        options.compatible_yee = true;
        options.relative_tolerance = 1.e-12;
        options.lower.fill(BC::Periodic);
        options.upper.fill(BC::Periodic);
        options.upper[0] = BC::PEC;
#if defined(WARPX_DIM_RZ)
        options.lower[0] = BC::Axis;
#else
        options.lower[0] = BC::PEC;
#endif
        warpx::darwin::DarwinInitialRateSchur projection(
            geometry, cells, distribution, options);
        amrex::MultiFab unit(amrex::convert(cells, amrex::IntVect(1)),
                            distribution, 1, 1);
        unit.setVal(1.);
        AMREX_ALWAYS_ASSERT(projection.Freeze(unit));
        std::array<amrex::MultiFab, 3> raw, held, zero;
        for (int c = 0; c < 3; ++c) {
            amrex::IntVect type(1);
#if defined(WARPX_DIM_RZ)
            if (c != 1) { type[c / 2] = 0; }
#else
            type[c] = 0;
#endif
            auto const edges = amrex::convert(cells, type);
            for (auto* field : {&raw[c], &held[c], &zero[c]}) {
                field->define(edges, distribution, 1, 1);
                field->setVal(0.);
            }
        }
        // This uniform axial current has exactly zero discrete divergence.
        // Every held value is below half an ulp of it: forming raw-held loses
        // the entire correction. The divergence of the held field is nonzero.
        raw[2].setVal(1.);
        for (amrex::MFIter it(held[2]); it.isValid(); ++it) {
            auto const field = held[2].array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
#if defined(WARPX_DIM_RZ)
                int const axial = j;
#else
                int const axial = k;
#endif
                field(i,j,k) = (axial % 2 == 0 ? Real(1) : Real(-1)) * 0x1p-60;
            });
        }
        Vector const raw_view{&raw[0], &raw[1], &raw[2]};
        Vector const held_view{&held[0], &held[1], &held[2]};
        Vector const zero_view{&zero[0], &zero[1], &zero[2]};
        Real const reference = projection.DivergenceResidualNorm(zero_view, held_view);
        Real const observed = projection.DivergenceResidualNorm(raw_view, held_view);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reference > 0. && observed == reference,
            "A solenoidal current must not erase the longitudinal correction defect");
        auto const result = projection.CorrectWithIterationBudget(
            raw_view, held_view, options.max_iterations);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(result.converged &&
            result.initial_residual == reference && result.residual <= result.target,
            "Refinement must solve the same nonzero divergence defect it checks");
        auto const correction = projection.CorrectionField();
        for (int c = 0; c < 3; ++c) {
            amrex::MultiFab::Add(held[c], *correction[c], 0, 0, 1, 0);
        }
        Real const final = projection.DivergenceResidualNorm(raw_view, held_view);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(final <= result.target,
            "The returned correction must satisfy the original projection tolerance");
        // Finite vector entries can overflow the divergence stencil. Such a
        // failure must not become norm zero through max(0, NaN).
        for (auto& field : held) { field.setVal(0.); }
        Real const large = std::numeric_limits<Real>::max() / 2.;
        for (amrex::MFIter it(raw[2]); it.isValid(); ++it) {
            auto const field = raw[2].array(it);
            amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
#if defined(WARPX_DIM_RZ)
                int const axial = j;
#else
                int const axial = k;
#endif
                field(i,j,k) = axial % 2 == 0 ? large : -large;
            });
        }
        AMREX_ALWAYS_ASSERT(raw[2].is_finite(0, 1, 0));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !std::isfinite(projection.DivergenceResidualNorm(raw_view, held_view)),
            "Nonfinite stencil arithmetic must reject the action check");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !projection.CorrectWithIterationBudget(raw_view, held_view,
                options.max_iterations).converged,
            "Nonfinite divergence must not be reported as a converged projection");
        amrex::Print() << "Source projection defect: initial=" << reference
                       << " final=" << final << " target=" << result.target << '\n';
    }
    amrex::Finalize();
}
