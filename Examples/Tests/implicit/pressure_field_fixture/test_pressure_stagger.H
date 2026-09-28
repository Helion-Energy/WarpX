/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_TEST_PRESSURE_STAGGER_H_
#define WARPX_TEST_PRESSURE_STAGGER_H_
#include "FieldSolver/ImplicitSolvers/FrozenPressureFieldCoupling.H"
#include <ablastr/coarsen/sample.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <array>
#include <cmath>

// Padded storage makes the historical extra-plane interpolation measurable
// without an invalid access. Actual RZ scalar MultiFabs only have k=0.
inline void
TestPressureResponseStagger (std::array<amrex::IntVect, 3> const& types)
{
    using amrex::Real;
    amrex::Gpu::DeviceVector<Real> storage(64), measured(6);
    auto* data = storage.data();
    auto* output = measured.data();
    amrex::ParallelFor(64, [=] AMREX_GPU_DEVICE(int index) {
        data[index] = (index % 4) + 10 * ((index / 4) % 4) + 100 * (index / 16);
    });
    amrex::Array4<const Real> const input(
        data, amrex::Dim3{0, 0, 0}, amrex::Dim3{4, 4, 4}, 1);
    amrex::GpuArray<int, 3> const nodal{1, 1, 1}, ratio{1, 1, 1};
    for (int c = 0; c < 3; ++c) {
        auto const stagger = warpx::thermal::PressureResponseStagger(types[c]);
        for (int d = AMREX_SPACEDIM; d < 3; ++d) {
            AMREX_ALWAYS_ASSERT(stagger[d] == 1);
        }
        auto omitted = stagger;
#if AMREX_SPACEDIM == 2
        omitted[2] = 0;
#endif
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) {
            output[2*c] = ablastr::coarsen::sample::Interp(
                input, nodal, stagger, ratio, 1, 1, 0, 0);
            output[2*c+1] = ablastr::coarsen::sample::Interp(
                input, nodal, omitted, ratio, 1, 1, 0, 0);
        });
    }
    amrex::Gpu::HostVector<Real> values(6);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, measured.begin(), measured.end(), values.begin());
    Real omission = 0;
    for (int c = 0; c < 3; ++c) {
        Real expected = 11, factor = .5;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            expected += factor * (1 - types[c][d]);
            factor *= 10;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(values[2*c] == expected,
            "Pressure PC must interpolate only the active native Yee dimensions");
        omission = std::max(omission, std::abs(values[2*c+1] - expected));
    }
#if AMREX_SPACEDIM == 2
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(omission == 50.,
        "Padded plane must discriminate the RZ inactive-dimension error");
#else
    AMREX_ALWAYS_ASSERT(omission == 0.);
#endif
    amrex::Print() << "PRESSURE_STAGGER dim=" << AMREX_SPACEDIM
                   << " error=0 omitted_inactive_plane_error=" << omission << "\n";
}
#endif
