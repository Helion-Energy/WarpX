/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "Circuit/Coils/DeviceReduction.H"
#include <AMReX.H>
#include <AMReX_Print.H>

int main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    {
        auto const rank = amrex::ParallelDescriptor::MyProc();
        auto const ranks = amrex::ParallelDescriptor::NProcs();
        for (int count : {0, 1, 3, 103, 262144}) {
            amrex::Gpu::DeviceVector<double> values(count), scratch(count);
            amrex::Gpu::PinnedVector<double> host(count);
            for (int epoch = 0; epoch < 7; ++epoch) {
                amrex::Gpu::Device::setStreamIndex(epoch % 2);
                auto* data = values.data();
                amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    data[i] = (rank + 1) * (i + 1) + epoch;
                });
                warpx::circuit::SumDeviceCoils(values, scratch);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                    values.begin(), values.end(), host.begin());
                for (int i = 0; i < count; ++i) {
                    double const expected = ranks*(ranks+1)/2*(i+1) + ranks*epoch;
                    AMREX_ALWAYS_ASSERT(host[i] == expected);
                }
            }
        }
        amrex::Print() << "CIRCUIT_DEVICE_SUM_PASS ranks=" << ranks << "\n";
    }
    amrex::Finalize();
}
