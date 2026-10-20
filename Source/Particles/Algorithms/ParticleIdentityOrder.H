/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_PARTICLE_IDENTITY_ORDER_H_
#define WARPX_PARTICLE_IDENTITY_ORDER_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#if defined(AMREX_USE_CUDA)
#   include <cub/cub.cuh>
#elif defined(AMREX_USE_HIP)
#   include <rocprim/rocprim.hpp>
#endif

#include <algorithm>
#include <climits>
#include <cstdint>
#include <numeric>
#include <vector>

namespace warpx::particles {

/** Map draw indices to full particle identities without rearranging storage.
 *
 * GPU cell sorting and redistribution may permute a particle tile. A stochastic
 * update must associate each draw with the same identity in a same-layout
 * replay. The full idcpu word includes the birth CPU. This function consumes no
 * randomness and changes neither particle attributes nor the number of draws.
 * Tile traversal and ownership remain the caller's responsibility.
 */
inline void ParticleIdentityOrder (
    int const np, std::uint64_t const* idcpu, int* ordered)
{
    if (np == 0) { return; }
    amrex::Gpu::DeviceVector<std::uint64_t> keys(np), sorted_keys(np);
    amrex::Gpu::DeviceVector<int> indices(np);
    auto* key = keys.data();
    auto* index = indices.data();
    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) {
        key[i] = idcpu[i];
        index[i] = i;
    });

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    std::size_t bytes = 0;
    auto const stream = amrex::Gpu::gpuStream();
    AMREX_GPU_SAFE_CALL(AMREX_HIP_OR_CUDA(
        rocprim::radix_sort_pairs, cub::DeviceRadixSort::SortPairs)
        (nullptr, bytes, key, sorted_keys.data(), index, ordered, np,
         0, int(sizeof(std::uint64_t)*CHAR_BIT), stream));
    amrex::Gpu::DeviceVector<char> scratch(bytes);
    AMREX_GPU_SAFE_CALL(AMREX_HIP_OR_CUDA(
        rocprim::radix_sort_pairs, cub::DeviceRadixSort::SortPairs)
        (scratch.data(), bytes, key, sorted_keys.data(), index, ordered, np,
         0, int(sizeof(std::uint64_t)*CHAR_BIT), stream));
#else
    std::vector<std::uint64_t> host_keys(np);
    std::vector<int> order(np);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, keys.begin(), keys.end(), host_keys.begin());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return host_keys[a] < host_keys[b]; });
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, order.begin(), order.end(), ordered);
#endif
    amrex::Gpu::streamSynchronize();
}

}
#endif
