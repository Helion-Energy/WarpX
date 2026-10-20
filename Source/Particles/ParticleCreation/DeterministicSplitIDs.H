/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_DETERMINISTIC_SPLIT_IDS_H_
#define WARPX_DETERMINISTIC_SPLIT_IDS_H_

#include <AMReX_Gpu.H>
#include <AMReX_Scan.H>

#if defined(AMREX_USE_CUDA)
#   include <cub/cub.cuh>
#endif

#include <algorithm>
#include <climits>
#include <cstdint>
#include <numeric>
#include <vector>

namespace warpx::particles {

/** Assign identity offsets in parent-ID order without changing particle storage.
 *
 * Only selected parents are sorted. The existing physical output offsets remain
 * untouched, so this does not reorder deposition or the random stream assigned
 * to a stored particle. Tile traversal and the rank-local NextID reservation are
 * unchanged. The complete idcpu key distinguishes parents born on other ranks.
 */
inline amrex::Long SplitIdentityOffsets (
    int np, int nparents, std::uint64_t const* idcpu, int const* factors,
    amrex::Long* identity_offsets)
{
    if (nparents == 0) { return 0; }
    amrex::Gpu::DeviceVector<std::uint64_t> keys(nparents), sorted_keys(nparents);
    amrex::Gpu::DeviceVector<int> indices(nparents), sorted_indices(nparents);
    auto* key = keys.data();
    auto* index = indices.data();
    auto* ordered = sorted_indices.data();
    auto const selected = amrex::Scan::PrefixSum<int>(np,
        [=] AMREX_GPU_DEVICE (int i) -> int { return factors[i] > 0 ? 1 : 0; },
        [=] AMREX_GPU_DEVICE (int i, int offset) {
            if (factors[i] > 0) { key[offset] = idcpu[i]; index[offset] = i; }
        }, amrex::Scan::Type::exclusive, amrex::Scan::retSum);
    AMREX_ALWAYS_ASSERT(selected == nparents);

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    std::size_t bytes = 0;
    auto const stream = amrex::Gpu::gpuStream();
    AMREX_GPU_SAFE_CALL(AMREX_HIP_OR_CUDA(
        rocprim::radix_sort_pairs, cub::DeviceRadixSort::SortPairs)
        (nullptr, bytes, key, sorted_keys.data(), index, ordered, nparents,
         0, int(sizeof(std::uint64_t)*CHAR_BIT), stream));
    amrex::Gpu::DeviceVector<char> scratch(bytes);
    AMREX_GPU_SAFE_CALL(AMREX_HIP_OR_CUDA(
        rocprim::radix_sort_pairs, cub::DeviceRadixSort::SortPairs)
        (scratch.data(), bytes, key, sorted_keys.data(), index, ordered, nparents,
         0, int(sizeof(std::uint64_t)*CHAR_BIT), stream));
#else
    // CPU and portable SYCL fallback, matching AMReX's CSR sorting policy.
    // Transfer selected parents only; no physical particle arrays leave device.
    std::vector<std::uint64_t> host_keys(nparents);
    std::vector<int> host_indices(nparents), order(nparents), sorted(nparents);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, keys.begin(), keys.end(), host_keys.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, indices.begin(), indices.end(),
                     host_indices.begin());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return host_keys[a] < host_keys[b]; });
    for (int i = 0; i < nparents; ++i) { sorted[i] = host_indices[order[i]]; }
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, sorted.begin(), sorted.end(),
                     sorted_indices.begin());
#endif
    auto const total = amrex::Scan::PrefixSum<amrex::Long>(nparents,
        [=] AMREX_GPU_DEVICE (int i) -> amrex::Long { return factors[ordered[i]]; },
        [=] AMREX_GPU_DEVICE (int i, amrex::Long offset) {
            identity_offsets[ordered[i]] = offset;
        }, amrex::Scan::Type::exclusive, amrex::Scan::retSum);
    amrex::Gpu::streamSynchronize();
    return total;
}

}
#endif
