/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_DEVICE_REDUCTION_H_
#define WARPX_CIRCUIT_DEVICE_REDUCTION_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>

#include <limits>

namespace warpx::circuit
{
    /** Sum small vectors using device-buffer MPI transfers and device
     * arithmetic. GPU-aware point-to-point MPI is sufficient: no assumption
     * is made about accelerator support in MPI's reduction implementation.
     * Scratch is persistent and distinct from values. All ranks participate
     * in the same order with identical vector lengths. This helper uses the
     * base AMReX communicator. Producer kernels must use the current stream or have
     * already joined it. A blocking exchange completes before its GPU consumer.
     * Recursive doubling takes O(log P) exchanges, including non-power-of-two
     * groups through an initial/final pairing. No vector is staged on host. */
    template <typename Scalar>
    inline void SumDeviceVector (amrex::Gpu::DeviceVector<Scalar>& values,
                                 amrex::Gpu::DeviceVector<Scalar>& scratch)
    {
        namespace pd = amrex::ParallelDescriptor;
        if (pd::NProcs() == 1 || values.empty()) { return; }
        AMREX_ALWAYS_ASSERT(values.size() == scratch.size() && &values != &scratch);
#ifdef AMREX_USE_MPI
#ifdef AMREX_USE_GPU
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pd::UseGpuAwareMpi(),
            "Device circuit probes require GPU-aware MPI; host staging is disabled");
#endif
        AMREX_ALWAYS_ASSERT(values.size() <= static_cast<std::size_t>(
            std::numeric_limits<int>::max()));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::ParallelContext::CommunicatorSub() == pd::Communicator(),
            "Device circuit reduction requires the base AMReX communicator");
        auto const comm = pd::Communicator();
        int const rank = pd::MyProc();
        int const count = static_cast<int>(values.size());
        auto const type = pd::Mpi_typemap<Scalar>::type();
        int const tag = pd::SeqNum();
        int power = 1;
        while (power <= pd::NProcs()/2) { power *= 2; }
        int const extra = pd::NProcs() - power;
        Scalar* const value = values.data();
        Scalar* const other = scratch.data();
        auto add = [=] () {
            amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE (int i) noexcept {
                value[i] += other[i];
            });
            // MPI can read the vector next; GPU-aware does not mean stream-aware.
            amrex::Gpu::streamSynchronize();
        };
        amrex::Gpu::streamSynchronize();
        int reduced_rank = rank - extra;
        if (rank < 2*extra) {
            if (rank%2 == 0) {
                BL_MPI_REQUIRE(MPI_Send(value, count, type, rank+1, tag, comm));
                reduced_rank = -1;
            } else {
                BL_MPI_REQUIRE(MPI_Recv(other, count, type, rank-1, tag,
                                        comm, MPI_STATUS_IGNORE));
                add();
                reduced_rank = rank/2;
            }
        }
        if (reduced_rank >= 0) {
            for (int mask = 1; mask < power; mask *= 2) {
                int const peer_reduced = reduced_rank ^ mask;
                int const peer = peer_reduced < extra ? 2*peer_reduced+1 : peer_reduced+extra;
                BL_MPI_REQUIRE(MPI_Sendrecv(value, count, type, peer, tag,
                                            other, count, type, peer, tag,
                                            comm, MPI_STATUS_IGNORE));
                add();
            }
        }
        if (rank < 2*extra) {
            if (rank%2 == 0) {
                BL_MPI_REQUIRE(MPI_Recv(value, count, type, rank+1, tag,
                                        comm, MPI_STATUS_IGNORE));
            } else {
                BL_MPI_REQUIRE(MPI_Send(value, count, type, rank-1, tag, comm));
            }
        }
#endif
    }

    // Preserve the circuit API and its binary64 arithmetic exactly.
    inline void SumDeviceCoils (amrex::Gpu::DeviceVector<double>& values,
                               amrex::Gpu::DeviceVector<double>& scratch)
    {
        SumDeviceVector(values, scratch);
    }
}
#endif
