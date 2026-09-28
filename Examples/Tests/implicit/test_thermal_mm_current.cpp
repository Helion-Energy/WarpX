/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ThermalMassMatrixResponse.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Print.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        namespace mm = warpx::thermal::mm_thermal_detail;
        using Real = amrex::Real;
        struct Case {
            mm::Pair plasma, base;
            Real delta_ion, expected_published, expected_vacuum;
        };
        // Exactly representable inputs and results: the first three changes
        // disappear if the retained current is omitted from the vacuum row.
        amrex::GpuArray<Case, 5> const cases{{
            {{1., 0x1.8p-54}, {1., 0x1p-54}, 0., 0., 0x1p-55},
            {{1., 0x1p-54}, {1., 0x1.8p-54}, 0., 0., -0x1p-55},
            {{2., 0x1.8p-54}, {1., 0x1p-54}, 1., 0., 0x1p-55},
            {{1.25, 0.}, {1., 0.}, 0.125, 0.125, 0.125},
            {{1., 0x1p-54}, {1., 0x1p-54}, 0., 0., 0.}
        }};
        amrex::Gpu::DeviceVector<mm::CurrentIncrements> device(cases.size());
        auto* result = device.data();
        amrex::ParallelFor(static_cast<int>(cases.size()), [=] AMREX_GPU_DEVICE (int i) {
            auto const& c = cases[i];
            result[i] = mm::CurrentIncrement(c.plasma, c.base, c.delta_ion);
        });
        amrex::Gpu::HostVector<mm::CurrentIncrements> host(cases.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
        for (std::size_t i = 0; i < cases.size(); ++i) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                host[i].published.hi == cases[i].expected_published && host[i].published.lo == 0.,
                "The thermal number flux must retain the published-high current convention");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                host[i].vacuum.hi == cases[i].expected_vacuum && host[i].vacuum.lo == 0.,
                "The vacuum thermal response must include the retained current increment");
        }
        amrex::Print() << "Thermal MM current: five published/vacuum cases passed\n";
    }
    amrex::Finalize();
}
