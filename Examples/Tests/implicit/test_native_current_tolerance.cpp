/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/NativeCurrentTolerance.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Print.H>

#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using Real = amrex::Real;
        struct Case { Real absolute, error, roundoff; bool accepted; };
        Real const inf = std::numeric_limits<Real>::infinity();
        Real const nan = std::numeric_limits<Real>::quiet_NaN();
        // Recorded rejected-source operands, plus errors on either side of
        // the physical floor and cancellation allowance. Both signs matter.
        amrex::GpuArray<Case, 15> const cases{{
            {2.2430472876e-8, -2.4800526906e-29, 7.3261728417e-43, true},
            {2.2430472876e-8, -6.8776247267e-25, 2.9140056452e-28, true},
            {2.2430472876e-8, -7.2759576142e-12, 5.1984239549e-10, true},
            {1.e-8, 0., 0., true},
            {1.e-8, 0.5e-8, 0., true},
            {1.e-8, 2.e-8, 0., false},
            {1.e-8, -2.e-8, 0., false},
            {1.e-8, 0.5e-5, 1.e-5, true},
            {0., -2.4800526906e-29, 7.3261728417e-43, false},
            {0., 0., 0., true},
            {1.e-8, nan, 0., false},
            {1.e-8, inf, 0., false},
            {-1.e-8, 0., 0., false},
            {1.e-8, 0., inf, false},
            {inf, 0., 0., false}
        }};
        amrex::Gpu::DeviceVector<int> device(cases.size());
        auto* result = device.data();
        amrex::ParallelFor(static_cast<int>(cases.size()), [=] AMREX_GPU_DEVICE (int i) {
            auto const& c = cases[i];
            result[i] = NativeCurrentTolerance{c.absolute}.Accepts(c.error, c.roundoff);
        });
        amrex::Gpu::HostVector<int> host(cases.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
        for (std::size_t i = 0; i < cases.size(); ++i) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(host[i] == int(cases[i].accepted),
                "Native current tolerance must accept small components and reject finite large or invalid defects");
        }
        NativeCurrentTolerance const disabled{};
        AMREX_ALWAYS_ASSERT(disabled.Bound(5.e-10) == 5.e-10);
        Real const large = std::numeric_limits<Real>::max();
        AMREX_ALWAYS_ASSERT(!NativeCurrentTolerance{large}.Accepts(0., large));
        amrex::Print() << "Native current tolerance: recorded rows, threshold rejection and nonfinite checks pass\n";
    }
    amrex::Finalize();
}
