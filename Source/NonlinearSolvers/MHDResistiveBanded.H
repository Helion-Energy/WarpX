/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_RESISTIVE_BANDED_H_
#define MHD_RESISTIVE_BANDED_H_

#include "MHDResistiveStencil.H"
#include "Utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>

namespace warpx::mhd_pc
{

/**
 * \brief Canonical block indexing of the banded solver: the face
 * unknowns are grouped into blocks along the z axis (one block per z
 * index, dense over all components and, in RZ, the full radial line),
 * with periodic-z nodal images identified with their low-side image.
 */
struct BandedIndexMap
{
    int num_blocks = 0;
    int block_size = 0;
    int bandwidth = 0;
    int banded_low = 0;
    int banded_cells = 0;
    bool banded_periodic = false;
    amrex::GpuArray<int, 3> component_active = {0, 0, 0};
    amrex::GpuArray<int, 3> local_offset = {0, 0, 0};
    int transverse_low = 0;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
    BandedIndex (const int i, const int j) const
    {
#if defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(j);
        return i;
#else
        amrex::ignore_unused(i);
        return j;
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
    WrapBanded (const int banded) const
    {
        if (!banded_periodic)
        {
            return banded;
        }
        int offset = (banded - banded_low) % banded_cells;
        if (offset < 0)
        {
            offset += banded_cells;
        }
        return banded_low + offset;
    }

    /** \brief Elimination position of a banded index (0-based). Periodic
     * rings are FOLDED (0, nb-1, 1, nb-2, ...): ring distance d maps to
     * elimination distance at most 2d, so the wrap coupling becomes
     * ordinary in-band coupling at twice the stencil bandwidth instead
     * of a dropped defect (a cut diffusion ring measurably poisons
     * restarted GMRES at production stiffness). */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
    BlockOrder (const int banded) const
    {
        const int offset = WrapBanded(banded) - banded_low;
        if (!banded_periodic)
        {
            return offset;
        }
        const int mirrored = 2 * (banded_cells - offset) - 1;
        return (2 * offset < mirrored) ? 2 * offset : mirrored;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
    LocalIndex (const int a_component, const int i, const int j) const
    {
#if defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(i, j);
        return local_offset[a_component];
#else
        amrex::ignore_unused(j);
        return local_offset[a_component] + (i - transverse_low);
#endif
    }
};

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

namespace banded_detail
{

/** \brief In-place Gauss-Jordan inverse of one dense block (one thread
 * block; dynamic shared memory caches the pivot column). Pivots are
 * floored, never searched: the blocks are diagonally similar to SPD
 * matrices (curl-curl with positive coefficients under the cylindrical
 * metric), for which unpivoted elimination is stable. */
template <typename TS>
__global__ void
GaussJordanInvert (TS* const matrix, const int size, const TS pivot_floor)
{
    extern __shared__ char gj_shared_bytes[];
    auto* const column = reinterpret_cast<TS*>(gj_shared_bytes);
    __shared__ double pivot_inverse_shared;
    for (int t = 0; t < size; ++t)
    {
        if (threadIdx.x == 0)
        {
            TS pivot = matrix[t * size + t];
            if (std::abs(pivot) < pivot_floor)
            {
                pivot = pivot >= TS(0.0) ? pivot_floor : -pivot_floor;
            }
            pivot_inverse_shared = static_cast<double>(TS(1.0) / pivot);
        }
        __syncthreads();
        const auto pivot_inverse = static_cast<TS>(pivot_inverse_shared);
        // Cache column t, then scale row t and put 1/p on the diagonal.
        for (int r = static_cast<int>(threadIdx.x); r < size;
             r += static_cast<int>(blockDim.x))
        {
            column[r] = matrix[r * size + t];
        }
        __syncthreads();
        for (int c = static_cast<int>(threadIdx.x); c < size;
             c += static_cast<int>(blockDim.x))
        {
            matrix[t * size + c] =
                (c == t) ? pivot_inverse : matrix[t * size + c] * pivot_inverse;
        }
        __syncthreads();
        // Eliminate column t from every other row (in-place inverse:
        // the eliminated column receives -f/p).
        for (int flat = static_cast<int>(threadIdx.x); flat < size * size;
             flat += static_cast<int>(blockDim.x))
        {
            const int r = flat / size;
            const int c = flat - r * size;
            if (r == t)
            {
                continue;
            }
            matrix[flat] = (c == t)
                               ? -column[r] * pivot_inverse
                               : matrix[flat] - column[r] * matrix[t * size + c];
        }
        __syncthreads();
    }
}

/** \brief C (m x n, ld_c) = beta C + alpha A (m x k, ld_a) B (k x n, ld_b)
 * on row-major panels (16 x 16 tiles); rows in [skip_low, skip_high) are
 * left untouched (the in-place blocked Gauss-Jordan row panel). */
template <typename TS>
__global__ void
PanelGemm (TS* const c_panel, const TS* const a_panel, const TS* const b_panel,
           const int m, const int n, const int k_extent, const int ld_c,
           const int ld_a, const int ld_b, const TS alpha, const TS beta,
           const int skip_low, const int skip_high)
{
    constexpr int tile = 16;
    __shared__ TS a_tile[tile][tile];
    __shared__ TS b_tile[tile][tile];
    const int row = static_cast<int>(blockIdx.y) * tile + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * tile + static_cast<int>(threadIdx.x);
    TS accumulator = TS(0.0);
    const int num_tiles = (k_extent + tile - 1) / tile;
    for (int t = 0; t < num_tiles; ++t)
    {
        const int a_col = t * tile + static_cast<int>(threadIdx.x);
        const int b_row = t * tile + static_cast<int>(threadIdx.y);
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k_extent) ? a_panel[row * ld_a + a_col] : TS(0.0);
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k_extent && col < n) ? b_panel[b_row * ld_b + col] : TS(0.0);
        __syncthreads();
        for (int kk = 0; kk < tile; ++kk)
        {
            accumulator += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < n && (row < skip_low || row >= skip_high))
    {
        const int flat = row * ld_c + col;
        // BLAS beta = 0 contract: C is write-only (see BlockGemm).
        c_panel[flat] = (beta == TS(0.0)) ? alpha * accumulator
                                          : beta * c_panel[flat] + alpha * accumulator;
    }
}

/** \brief Strided row-major copy: dst (m x n, ld_dst) = src (ld_src). */
template <typename TS>
__global__ void
CopyPanel (TS* const dst, const TS* const src, const int m, const int n,
           const int ld_dst, const int ld_src)
{
    const int flat = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
    if (flat >= m * n)
    {
        return;
    }
    const int r = flat / n;
    const int c = flat - r * n;
    dst[r * ld_dst + c] = src[r * ld_src + c];
}

/** \brief C = beta C + alpha A B on size x size blocks (16 x 16 tiles). */
template <typename TS>
__global__ void
BlockGemm (TS* const c_block, const TS* const a_block, const TS* const b_block,
           const int size, const TS alpha, const TS beta)
{
    constexpr int tile = 16;
    __shared__ TS a_tile[tile][tile];
    __shared__ TS b_tile[tile][tile];
    const int row = static_cast<int>(blockIdx.y) * tile + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * tile + static_cast<int>(threadIdx.x);
    TS accumulator = TS(0.0);
    const int num_tiles = (size + tile - 1) / tile;
    for (int t = 0; t < num_tiles; ++t)
    {
        const int a_col = t * tile + static_cast<int>(threadIdx.x);
        const int b_row = t * tile + static_cast<int>(threadIdx.y);
        a_tile[threadIdx.y][threadIdx.x] =
            (row < size && a_col < size) ? a_block[row * size + a_col] : TS(0.0);
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < size && col < size) ? b_block[b_row * size + col] : TS(0.0);
        __syncthreads();
        for (int k = 0; k < tile; ++k)
        {
            accumulator += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < size && col < size)
    {
        const int flat = row * size + col;
        // BLAS beta = 0 contract: C is write-only (never read), so a
        // freshly allocated (uninitialized) output block cannot poison
        // the product through 0 * garbage.
        c_block[flat] = (beta == TS(0.0)) ? alpha * accumulator
                                          : beta * c_block[flat] + alpha * accumulator;
    }
}

/** \brief Forward-sweep step: y_{i+dj} -= L_{i+dj,i} y_i, dj = 1..p. */
template <typename TS>
__global__ void
ForwardStep (TS* const y, const TS* const bands, const int step, const int size,
             const int bandwidth, const int num_blocks)
{
    const int flat = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
    const int dj = flat / size + 1;
    const int r = flat - (dj - 1) * size;
    const int j = step + dj;
    if (dj > bandwidth || j >= num_blocks)
    {
        return;
    }
    const std::size_t stride = static_cast<std::size_t>(size) * size;
    const TS* const lower =
        bands + (static_cast<std::size_t>(j) * (2 * bandwidth + 1) +
                 (bandwidth - dj)) * stride + static_cast<std::size_t>(r) * size;
    const TS* const y_step = y + static_cast<std::size_t>(step) * size;
    TS accumulator = TS(0.0);
    for (int c = 0; c < size; ++c)
    {
        accumulator += lower[c] * y_step[c];
    }
    y[static_cast<std::size_t>(j) * size + r] -= accumulator;
}

/** \brief Backward-sweep defect: t = y_i - sum_dj U_{i,i+dj} x_{i+dj}
 * (x is zero-padded past the last block, and the unfilled band storage
 * there is zero, so no block-range branch is needed). */
template <typename TS>
__global__ void
BackwardDefect (TS* const defect, const TS* const y, const TS* const x,
                const TS* const bands, const int step, const int size,
                const int bandwidth)
{
    const int r = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (r >= size)
    {
        return;
    }
    const std::size_t stride = static_cast<std::size_t>(size) * size;
    TS accumulator = y[static_cast<std::size_t>(step) * size + r];
    for (int dj = 1; dj <= bandwidth; ++dj)
    {
        const TS* const upper =
            bands + (static_cast<std::size_t>(step) * (2 * bandwidth + 1) +
                     (bandwidth + dj)) * stride + static_cast<std::size_t>(r) * size;
        const TS* const x_next = x + static_cast<std::size_t>(step + dj) * size;
        for (int c = 0; c < size; ++c)
        {
            accumulator -= upper[c] * x_next[c];
        }
    }
    defect[r] = accumulator;
}

/** \brief Backward-sweep solve: x_i = invD_i t. */
template <typename TS>
__global__ void
BackwardSolve (TS* const x, const TS* const defect, const TS* const inverse_diagonal,
               const int step, const int size)
{
    const int r = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (r >= size)
    {
        return;
    }
    const TS* const row = inverse_diagonal +
                          static_cast<std::size_t>(step) * size * size +
                          static_cast<std::size_t>(r) * size;
    TS accumulator = TS(0.0);
    for (int c = 0; c < size; ++c)
    {
        accumulator += row[c] * defect[c];
    }
    x[static_cast<std::size_t>(step) * size + r] = accumulator;
}

} // namespace banded_detail

#endif // CUDA || HIP

/**
 * \brief Precision-templated storage and dense kernels of the banded
 * factorization (see ResistiveBandedSolver). Single precision is the
 * default: a preconditioner-grade truncation (like the out-of-band
 * dropping) that buys the large FP32/FP64 throughput ratio of
 * workstation GPUs; blocks are floored/regularized, so the reduced
 * precision only perturbs the preconditioner, never the answer.
 */
template <typename TS>
struct ResistiveBandedEngine
{
    int m_num_blocks = 0;
    int m_block_size = 0;
    int m_bandwidth = 0;
    bool m_factored = false;

    amrex::Gpu::DeviceVector<TS> m_bands;
    amrex::Gpu::DeviceVector<TS> m_inverse_diagonal;
    amrex::Gpu::DeviceVector<TS> m_scratch;
    amrex::Gpu::DeviceVector<TS> m_rhs;
    amrex::Gpu::DeviceVector<TS> m_solution;
    amrex::Gpu::DeviceVector<TS> m_defect;

    void
    Allocate (const int a_num_blocks, const int a_block_size, const int a_bandwidth)
    {
        m_num_blocks = a_num_blocks;
        m_block_size = a_block_size;
        m_bandwidth = a_bandwidth;
        const auto stride = static_cast<std::size_t>(a_block_size) * a_block_size;
        m_bands.resize(static_cast<std::size_t>(a_num_blocks) *
                       (2 * a_bandwidth + 1) * stride);
        m_inverse_diagonal.resize(static_cast<std::size_t>(a_num_blocks) * stride);
        m_scratch.resize(stride);
        const auto padded =
            static_cast<std::size_t>(a_num_blocks + a_bandwidth) * a_block_size;
        m_rhs.resize(padded);
        m_solution.resize(padded);
        m_defect.resize(static_cast<std::size_t>(a_block_size));
        // Zero-fill every engine buffer once: the arena does not zero,
        // and slots no gather/scatter ever touches (rhs padding, the
        // identity rows of block-local slots with no stored face) must
        // read as zero, never as reused-heap garbage -- 0 * NaN = NaN
        // would poison the sweep (measured as a heap-history-dependent
        // NaN factorization on the mirror-plane test).
        for (auto* vec : {&m_bands, &m_inverse_diagonal, &m_scratch, &m_rhs,
                          &m_solution, &m_defect})
        {
            TS* const data = vec->data();
            amrex::ParallelFor(static_cast<amrex::Long>(vec->size()),
                               [=] AMREX_GPU_DEVICE (amrex::Long flat)
                               {
                                   data[flat] = TS(0.0);
                               });
        }
        amrex::Gpu::streamSynchronize();
    }

    [[nodiscard]] static constexpr TS
    PivotFloor ()
    {
        return std::is_same_v<TS, float> ? TS(1.0e-30) : TS(1.0e-280);
    }

    /** \brief Reset the band tensor to the identity of I + A (every
     * stored diagonal slot, canonical or padding, carries 1). */
    void
    ResetToIdentity ()
    {
        m_factored = false;
        TS* const bands = m_bands.data();
        amrex::ParallelFor(static_cast<amrex::Long>(m_bands.size()),
                           [=] AMREX_GPU_DEVICE (amrex::Long flat)
                           {
                               bands[flat] = TS(0.0);
                           });
        const int size = m_block_size;
        const int bandwidth = m_bandwidth;
        amrex::ParallelFor(
            static_cast<amrex::Long>(m_num_blocks) * size,
            [=] AMREX_GPU_DEVICE (amrex::Long flat)
            {
                const auto block = static_cast<int>(flat / size);
                const auto local =
                    static_cast<int>(flat - static_cast<amrex::Long>(block) * size);
                const std::size_t stride = static_cast<std::size_t>(size) * size;
                bands[(static_cast<std::size_t>(block) * (2 * bandwidth + 1) +
                       bandwidth) * stride +
                      static_cast<std::size_t>(local) * size + local] = TS(1.0);
            });
    }

    [[nodiscard]] TS*
    BandBlock (const int a_block, const int a_band)
    {
        const std::size_t stride = static_cast<std::size_t>(m_block_size) * m_block_size;
        return m_bands.data() +
               (static_cast<std::size_t>(a_block) * (2 * m_bandwidth + 1) +
                (m_bandwidth + a_band)) * stride;
    }

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    /** \brief In-place dense inverse on the device. Small blocks use the
     * one-thread-block Gauss-Jordan directly; larger blocks use blocked
     * Gauss-Jordan with 64-wide panels, so the O(S^3) work runs in
     * grid-wide GEMM updates instead of a single bandwidth-starved
     * thread block (measured ~100x on the production block sizes). */
    void
    DeviceInvert (TS* const matrix, const int size, TS* const scratch)
    {
        const auto stream = amrex::Gpu::gpuStream();
        constexpr int invert_threads = 256;
        constexpr int panel_width = 64;
        // The blocked path needs 2 S w + w^2 scratch entries inside the
        // S^2 scratch block; below that size the plain kernel is fast.
        if (static_cast<std::size_t>(size) * size <
            2 * static_cast<std::size_t>(panel_width) * size +
                static_cast<std::size_t>(panel_width) * panel_width)
        {
            banded_detail::GaussJordanInvert<TS>
                <<<1, invert_threads, static_cast<int>(sizeof(TS)) * size, stream>>>(
                    matrix, size, PivotFloor());
            return;
        }
        constexpr int tile = 16;
        constexpr int copy_threads = 256;
        TS* const column_panel = scratch;                       // S x w
        TS* const row_panel = scratch + static_cast<std::size_t>(size) * panel_width;
        TS* const pivot_inverse =                               // w x w
            scratch + 2 * static_cast<std::size_t>(size) * panel_width;
        for (int start = 0; start < size; start += panel_width)
        {
            const int width = amrex::min(panel_width, size - start);
            const dim3 gemm_block(tile, tile);
            const auto blocks_for = [] (const int extent)
            {
                return (extent + tile - 1) / tile;
            };
            const auto copy_blocks = [] (const int count)
            {
                return (count + copy_threads - 1) / copy_threads;
            };
            // Pinv = inverse of the panel diagonal block.
            banded_detail::CopyPanel<TS>
                <<<copy_blocks(width * width), copy_threads, 0, stream>>>(
                    pivot_inverse, matrix + start * size + start, width, width, width,
                    size);
            banded_detail::GaussJordanInvert<TS>
                <<<1, invert_threads, static_cast<int>(sizeof(TS)) * width, stream>>>(
                    pivot_inverse, width, PivotFloor());
            // Stage the pre-update column and row panels.
            banded_detail::CopyPanel<TS>
                <<<copy_blocks(size * width), copy_threads, 0, stream>>>(
                    column_panel, matrix + start, size, width, panel_width, size);
            banded_detail::CopyPanel<TS>
                <<<copy_blocks(width * size), copy_threads, 0, stream>>>(
                    row_panel, matrix + start * size, width, size, size, size);
            // Row panel: M[panel, :] = Pinv R (its panel columns become I).
            {
                const dim3 grid(blocks_for(size), blocks_for(width));
                banded_detail::PanelGemm<TS><<<grid, gemm_block, 0, stream>>>(
                    matrix + start * size, pivot_inverse, row_panel, width, size,
                    width, size, width, size, TS(1.0), TS(0.0), size, size);
            }
            // Trailing update: M[out, :] -= C M[panel, :].
            {
                const dim3 grid(blocks_for(size), blocks_for(size));
                banded_detail::PanelGemm<TS><<<grid, gemm_block, 0, stream>>>(
                    matrix, column_panel, matrix + start * size, size, size, width,
                    size, panel_width, size, TS(-1.0), TS(1.0), start, start + width);
            }
            // Column fix: M[out, panel] = -C Pinv; M[panel, panel] = Pinv.
            {
                const dim3 grid(blocks_for(width), blocks_for(size));
                banded_detail::PanelGemm<TS><<<grid, gemm_block, 0, stream>>>(
                    matrix + start, column_panel, pivot_inverse, size, width, width,
                    size, panel_width, width, TS(-1.0), TS(0.0), start, start + width);
            }
            banded_detail::CopyPanel<TS>
                <<<copy_blocks(width * width), copy_threads, 0, stream>>>(
                    matrix + start * size + start, pivot_inverse, width, width, size,
                    width);
        }
    }
#endif

    /** \brief Sequential block-banded LU along the banded axis: invert
     * each diagonal block, scale the sub-diagonal blocks, update the
     * trailing band (the reference algorithm's BlockBandedLU factorization). */
    void
    Factorize ()
    {
        const int nb = m_num_blocks;
        const int size = m_block_size;
        const int p = m_bandwidth;
        const std::size_t stride = static_cast<std::size_t>(size) * size;
        TS* const inverse_diagonal = m_inverse_diagonal.data();
        TS* const scratch = m_scratch.data();

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
        const auto stream = amrex::Gpu::gpuStream();
        constexpr int tile = 16;
        const dim3 gemm_block(tile, tile);
        const dim3 gemm_grid((size + tile - 1) / tile, (size + tile - 1) / tile);
        for (int i = 0; i < nb; ++i)
        {
            TS* const diagonal_inverse =
                inverse_diagonal + static_cast<std::size_t>(i) * stride;
            amrex::Gpu::dtod_memcpy_async(diagonal_inverse, BandBlock(i, 0),
                                          stride * sizeof(TS));
            DeviceInvert(diagonal_inverse, size, scratch);
            for (int dj = 1; dj <= p; ++dj)
            {
                const int j = i + dj;
                if (j >= nb)
                {
                    continue;
                }
                // L_{j,i} <- A_{j,i} invD_i, staged through the scratch
                // block (the GEMM output must not alias an input).
                banded_detail::BlockGemm<TS><<<gemm_grid, gemm_block, 0, stream>>>(
                    scratch, BandBlock(j, -dj), diagonal_inverse, size, TS(1.0),
                    TS(0.0));
                amrex::Gpu::dtod_memcpy_async(BandBlock(j, -dj), scratch,
                                              stride * sizeof(TS));
                for (int dk = 1; dk <= p; ++dk)
                {
                    if (i + dk >= nb)
                    {
                        continue;
                    }
                    banded_detail::BlockGemm<TS><<<gemm_grid, gemm_block, 0, stream>>>(
                        BandBlock(j, dk - dj), scratch, BandBlock(i, dk), size,
                        TS(-1.0), TS(1.0));
                }
            }
        }
        amrex::Gpu::streamSynchronize();
#elif defined(AMREX_USE_SYCL)
        amrex::ignore_unused(nb, size, p, stride, inverse_diagonal, scratch);
        WARPX_ABORT_WITH_MESSAGE(
            "pc_mhd_block.resistive_solver = banded is not implemented for SYCL builds");
#else
        // Portable host path (CPU builds: the device vectors live in host
        // memory). Serial Gauss-Jordan and naive GEMMs; CI-scale grids.
        for (int i = 0; i < nb; ++i)
        {
            TS* const diagonal_inverse =
                inverse_diagonal + static_cast<std::size_t>(i) * stride;
            std::memcpy(diagonal_inverse, BandBlock(i, 0), stride * sizeof(TS));
            HostInvert(diagonal_inverse, size);
            for (int dj = 1; dj <= p; ++dj)
            {
                const int j = i + dj;
                if (j >= nb)
                {
                    continue;
                }
                HostGemm(scratch, BandBlock(j, -dj), diagonal_inverse, size, TS(1.0),
                         TS(0.0));
                std::memcpy(BandBlock(j, -dj), scratch, stride * sizeof(TS));
                for (int dk = 1; dk <= p; ++dk)
                {
                    if (i + dk >= nb)
                    {
                        continue;
                    }
                    HostGemm(BandBlock(j, dk - dj), scratch, BandBlock(i, dk), size,
                             TS(-1.0), TS(1.0));
                }
            }
        }
#endif
        m_factored = true;
    }

    /** \brief Solve (I + A) x = rhs through the factored bands: forward
     * elimination then back substitution (m_rhs in, m_solution out; the
     * forward sweep consumes m_rhs in place). */
    void
    Sweep ()
    {
        const int nb = m_num_blocks;
        const int size = m_block_size;
        const int p = m_bandwidth;
        TS* const rhs = m_rhs.data();
        TS* const solution = m_solution.data();
        amrex::ParallelFor(static_cast<amrex::Long>(m_solution.size()),
                           [=] AMREX_GPU_DEVICE (amrex::Long flat)
                           {
                               solution[flat] = TS(0.0);
                           });

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
        const auto stream = amrex::Gpu::gpuStream();
        constexpr int threads = 256;
        const int forward_blocks = (p * size + threads - 1) / threads;
        const int row_blocks = (size + threads - 1) / threads;
        for (int i = 0; i < nb - 1; ++i)
        {
            banded_detail::ForwardStep<TS><<<forward_blocks, threads, 0, stream>>>(
                rhs, m_bands.data(), i, size, p, nb);
        }
        for (int i = nb - 1; i >= 0; --i)
        {
            banded_detail::BackwardDefect<TS><<<row_blocks, threads, 0, stream>>>(
                m_defect.data(), rhs, solution, m_bands.data(), i, size, p);
            banded_detail::BackwardSolve<TS><<<row_blocks, threads, 0, stream>>>(
                solution, m_defect.data(), m_inverse_diagonal.data(), i, size);
        }
        amrex::Gpu::streamSynchronize();
#elif defined(AMREX_USE_SYCL)
        amrex::ignore_unused(nb, size, p, rhs, solution);
        WARPX_ABORT_WITH_MESSAGE(
            "pc_mhd_block.resistive_solver = banded is not implemented for SYCL builds");
#else
        const std::size_t stride = static_cast<std::size_t>(size) * size;
        for (int i = 0; i < nb - 1; ++i)
        {
            for (int dj = 1; dj <= p && i + dj < nb; ++dj)
            {
                const TS* const lower = BandBlock(i + dj, -dj);
                for (int r = 0; r < size; ++r)
                {
                    TS accumulator = TS(0.0);
                    for (int c = 0; c < size; ++c)
                    {
                        accumulator += lower[static_cast<std::size_t>(r) * size + c] *
                                       rhs[static_cast<std::size_t>(i) * size + c];
                    }
                    rhs[static_cast<std::size_t>(i + dj) * size + r] -= accumulator;
                }
            }
        }
        TS* const defect = m_defect.data();
        const TS* const inverse_diagonal = m_inverse_diagonal.data();
        for (int i = nb - 1; i >= 0; --i)
        {
            for (int r = 0; r < size; ++r)
            {
                TS accumulator = rhs[static_cast<std::size_t>(i) * size + r];
                for (int dj = 1; dj <= p; ++dj)
                {
                    const TS* const upper = BandBlock(i, dj);
                    for (int c = 0; c < size; ++c)
                    {
                        accumulator -=
                            upper[static_cast<std::size_t>(r) * size + c] *
                            solution[static_cast<std::size_t>(i + dj) * size + c];
                    }
                }
                defect[r] = accumulator;
            }
            for (int r = 0; r < size; ++r)
            {
                TS accumulator = TS(0.0);
                const TS* const row = inverse_diagonal +
                                      static_cast<std::size_t>(i) * stride +
                                      static_cast<std::size_t>(r) * size;
                for (int c = 0; c < size; ++c)
                {
                    accumulator += row[c] * defect[c];
                }
                solution[static_cast<std::size_t>(i) * size + r] = accumulator;
            }
        }
#endif
    }

#if !defined(AMREX_USE_CUDA) && !defined(AMREX_USE_HIP) && !defined(AMREX_USE_SYCL)
    static void
    HostInvert (TS* const matrix, const int size)
    {
        // In-place Gauss-Jordan inverse with floored pivots (see the
        // device kernel for the stability rationale).
        amrex::Vector<TS> column(size);
        for (int t = 0; t < size; ++t)
        {
            TS pivot = matrix[static_cast<std::size_t>(t) * size + t];
            if (std::abs(pivot) < PivotFloor())
            {
                pivot = pivot >= TS(0.0) ? PivotFloor() : -PivotFloor();
            }
            const TS pivot_inverse = TS(1.0) / pivot;
            for (int r = 0; r < size; ++r)
            {
                column[r] = matrix[static_cast<std::size_t>(r) * size + t];
            }
            for (int c = 0; c < size; ++c)
            {
                matrix[static_cast<std::size_t>(t) * size + c] =
                    (c == t)
                        ? pivot_inverse
                        : matrix[static_cast<std::size_t>(t) * size + c] * pivot_inverse;
            }
            for (int r = 0; r < size; ++r)
            {
                if (r == t)
                {
                    continue;
                }
                const TS factor = column[r];
                for (int c = 0; c < size; ++c)
                {
                    matrix[static_cast<std::size_t>(r) * size + c] =
                        (c == t)
                            ? -factor * pivot_inverse
                            : matrix[static_cast<std::size_t>(r) * size + c] -
                                  factor *
                                      matrix[static_cast<std::size_t>(t) * size + c];
                }
            }
        }
    }

    static void
    HostGemm (TS* const c_block, const TS* const a_block, const TS* const b_block,
              const int size, const TS alpha, const TS beta)
    {
        for (int r = 0; r < size; ++r)
        {
            for (int c = 0; c < size; ++c)
            {
                TS accumulator = TS(0.0);
                for (int k = 0; k < size; ++k)
                {
                    accumulator += a_block[static_cast<std::size_t>(r) * size + k] *
                                   b_block[static_cast<std::size_t>(k) * size + c];
                }
                const std::size_t flat = static_cast<std::size_t>(r) * size + c;
                // BLAS beta = 0 contract: C is write-only (never read),
                // so the uninitialized scratch block cannot poison the
                // product through 0 * garbage (the arena does not zero;
                // measured as a heap-history-dependent NaN factorization
                // on the mirror-plane test).
                c_block[flat] = (beta == TS(0.0))
                                    ? alpha * accumulator
                                    : beta * c_block[flat] + alpha * accumulator;
            }
        }
    }
#endif
};

namespace banded_detail
{

/** \brief Scatter the emission of the owned rows of one component into
 * the band tensor; out-of-band couplings are accounted rather than
 * scattered (a_accounting[0] accumulates their |value|; the folded
 * periodic ordering keeps every coupling in band, so the caller treats
 * any accumulation as an assembly bug). */
template <typename TS>
void
ScatterComponent (const ResistiveStencilConstants& a_constants,
                  const BandedIndexMap& a_map, const amrex::iMultiFab& a_owner,
                  const int a_component,
                  const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
                  const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
                  const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge,
                  const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge,
                  TS* const a_bands, amrex::Real* const a_accounting)
{
    for (amrex::MFIter mfi(a_owner); mfi.isValid(); ++mfi)
    {
        const auto owner = a_owner.const_array(mfi);
        ResistiveStencilArrays arrays;
        for (int field = 0; field < 3; ++field)
        {
            if (a_eta_edge[field] != nullptr)
            {
                arrays.eta[field] = a_eta_edge[field]->const_array(mfi);
            }
            if (a_eta_h_edge[field] != nullptr)
            {
                arrays.eta_h[field] = a_eta_h_edge[field]->const_array(mfi);
            }
            if (a_hall_edge[field] != nullptr)
            {
                arrays.hall[field] = a_hall_edge[field]->const_array(mfi);
            }
            if (a_inertia_edge[field] != nullptr)
            {
                arrays.inertia[field] = a_inertia_edge[field]->const_array(mfi);
            }
        }
        const BandedIndexMap map = a_map;
        const ResistiveStencilConstants constants = a_constants;
        const int row_component = a_component;
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (!owner(i, j, k))
                {
                    return;
                }
                const int row_order = map.BlockOrder(map.BandedIndex(i, j));
                const int row_local = map.LocalIndex(row_component, i, j);
                const int size = map.block_size;
                const int bandwidth = map.bandwidth;
                const std::size_t stride = static_cast<std::size_t>(size) * size;
                warpx::mhd_pc::EmitResistiveRow(
                    constants, arrays, row_component, i, j, k,
                    [&] (const int column_component, const int ic, const int jc,
                         const int kc, const amrex::Real coefficient)
                    {
                        amrex::ignore_unused(kc);
                        const int column_order =
                            map.BlockOrder(map.BandedIndex(ic, jc));
                        const int band = column_order - row_order;
                        if (band < -bandwidth || band > bandwidth)
                        {
                            // Never taken with the folded ordering; the
                            // caller asserts on any accumulation.
                            amrex::Gpu::Atomic::AddNoRet(&a_accounting[0],
                                                         std::abs(coefficient));
                            return;
                        }
                        const int column_local =
                            map.LocalIndex(column_component, ic, jc);
                        TS* const slot =
                            a_bands +
                            (static_cast<std::size_t>(row_order) *
                                 (2 * bandwidth + 1) +
                             (bandwidth + band)) * stride +
                            static_cast<std::size_t>(row_local) * size + column_local;
                        amrex::Gpu::Atomic::AddNoRet(slot,
                                                     static_cast<TS>(coefficient));
                    });
            });
    }
}

/** \brief Gather the owned stored faces of one component into the
 * canonical (zero-padded) right-hand side. */
template <typename TS>
void
GatherComponent (const BandedIndexMap& a_map, const amrex::iMultiFab& a_owner,
                 const int a_component, const amrex::MultiFab& a_faces,
                 TS* const a_rhs)
{
    for (amrex::MFIter mfi(a_owner); mfi.isValid(); ++mfi)
    {
        const auto owner = a_owner.const_array(mfi);
        const auto faces = a_faces.const_array(mfi);
        const BandedIndexMap map = a_map;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               if (!owner(i, j, k))
                               {
                                   return;
                               }
                               const int block =
                                   map.BlockOrder(map.BandedIndex(i, j));
                               const int local = map.LocalIndex(component, i, j);
                               a_rhs[static_cast<std::size_t>(block) * map.block_size +
                                     local] = static_cast<TS>(faces(i, j, k));
                           });
    }
}

/** \brief Scatter the canonical solution to every stored face of one
 * component (duplicated faces read the same canonical unknown). */
template <typename TS>
void
ScatterSolution (const BandedIndexMap& a_map, const int a_component,
                 amrex::MultiFab& a_faces, const TS* const a_solution)
{
    for (amrex::MFIter mfi(a_faces); mfi.isValid(); ++mfi)
    {
        const auto faces = a_faces.array(mfi);
        const BandedIndexMap map = a_map;
        const int component = a_component;
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const int block = map.BlockOrder(map.BandedIndex(i, j));
                const int local = map.LocalIndex(component, i, j);
                faces(i, j, k) = static_cast<amrex::Real>(
                    a_solution[static_cast<std::size_t>(block) * map.block_size +
                               local]);
            });
    }
}

/** \brief Zero a device vector (free helper: device lambdas may not be
 * defined inside generic lambdas under NVCC). */
template <typename TS>
void
ZeroVector (TS* const a_data, const std::size_t a_size)
{
    amrex::ParallelFor(static_cast<amrex::Long>(a_size),
                       [=] AMREX_GPU_DEVICE (amrex::Long flat)
                       {
                           a_data[flat] = TS(0.0);
                       });
}

} // namespace banded_detail

/**
 * \brief Block-banded LU inverse of the resistive(+hyper-resistive)
 * block operator I + A (see MHDResistiveStencil.H) for the MHD block
 * preconditioner: the Chacon block-banded direct preconditioner
 * reproduced on the WarpX face-staggered state.
 *
 * The canonical face unknowns are grouped into blocks along the z axis
 * and eliminated sequentially along z with block bandwidth p (1
 * resistive, 2 with the wider hyper-resistive chain); each block is
 * dense over all components and, in RZ, the full radial line, so the
 * axis/wall/vacuum-halo radial coupling is treated exactly inside the
 * dense diagonal blocks. The matrix values come from the SAME stencil
 * emission the matrix-free and direct paths consume, scattered directly
 * into the padded band tensor on the device; the factorization and the
 * sweeps are dense kernels that stay resident on the device (no rank-0
 * gather; multi-rank runs are asserted unsupported for now).
 *
 * A periodic banded axis is handled EXACTLY by eliminating the blocks
 * in folded ring order (0, nb-1, 1, nb-2, ...) at twice the bandwidth
 * (see BandedIndexMap::BlockOrder): dropping the wrap coupling instead
 * (the reference algorithm's out-of-band truncation) cuts the diffusion ring and was
 * measured to poison restarted GMRES at production stiffness.
 */
class ResistiveBandedSolver
{
    public:
        ResistiveBandedSolver () = default;

        /** \brief Whether the banded factorization runs on this build
         * (CUDA/HIP device path or the portable CPU path; SYCL is not
         * supported yet). */
        [[nodiscard]] static constexpr bool
        HaveBackend ()
        {
#if defined(AMREX_USE_SYCL)
            return false;
#else
            return true;
#endif
        }

        /**
         * \brief Build the canonical block indexing and the ownership
         * masks. a_faces holds the participating magnetic-field face
         * components (null outside the resistive curl-curl), a_bandwidth
         * the block bandwidth of the assembled operator (1 resistive, 2
         * with the hyper-resistive chain), a_single_precision the
         * factorization storage.
         */
        void
        Define (const amrex::Geometry& a_geometry,
                const amrex::Array<const amrex::MultiFab*, 3>& a_faces,
                const int a_bandwidth, const bool a_single_precision)
        {
            BL_PROFILE("ResistiveBanded::Define()");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !m_is_defined,
                "ResistiveBandedSolver::Define() called on a defined object");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                HaveBackend(),
                "pc_mhd_block.resistive_solver = banded is not implemented for "
                "SYCL builds");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                amrex::ParallelDescriptor::NProcs() == 1,
                "pc_mhd_block.resistive_solver = banded currently supports "
                "single-rank runs only (the block-banded elimination is "
                "sequential along z)");

            m_geometry = a_geometry;
            m_single_precision = a_single_precision;
            const amrex::Box domain = m_geometry.Domain();
#if defined(WARPX_DIM_1D_Z)
            constexpr int banded_direction = 0;
#else
            constexpr int banded_direction = AMREX_SPACEDIM - 1;
#endif
            m_map.banded_low = domain.smallEnd(banded_direction);
            m_map.banded_cells = domain.length(banded_direction);
            m_map.banded_periodic = m_geometry.isPeriodic(banded_direction);
            // The folded ring ordering of a periodic banded axis (see
            // BlockOrder) doubles the elimination bandwidth.
            m_map.bandwidth = m_map.banded_periodic ? 2 * a_bandwidth : a_bandwidth;
            m_map.transverse_low = domain.smallEnd(0);

            int block_size = 0;
            int num_blocks = 0;
            amrex::Long canonical_rows = 0;
            for (int component = 0; component < 3; ++component)
            {
                if (a_faces[component] == nullptr)
                {
                    continue;
                }
                m_map.component_active[component] = 1;
                m_map.local_offset[component] = block_size;
                const amrex::IntVect stagger =
                    a_faces[component]->ixType().toIntVect();
                int transverse_width = 1;
#if !defined(WARPX_DIM_1D_Z)
                transverse_width = domain.length(0) + stagger[0];
#endif
                const int banded_extent =
                    m_map.banded_cells +
                    ((stagger[banded_direction] == 1 && !m_map.banded_periodic) ? 1
                                                                                : 0);
                block_size += transverse_width;
                num_blocks = amrex::max(num_blocks, banded_extent);
                canonical_rows +=
                    static_cast<amrex::Long>(transverse_width) * banded_extent;

                m_owner[component] = amrex::OwnerMask(*a_faces[component],
                                                      m_geometry.periodicity());
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                block_size > 0,
                "ResistiveBandedSolver found no active face components");
            m_map.block_size = block_size;
            m_map.num_blocks = num_blocks;

            // Every canonical row must be owned exactly once (the owner
            // masks and the periodic-image wrap must tile the unknowns).
            amrex::Long owned_rows = 0;
            for (int component = 0; component < 3; ++component)
            {
                if (m_owner[component])
                {
                    owned_rows += m_owner[component]->sum(0);
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                owned_rows == canonical_rows,
                "ResistiveBandedSolver ownership does not cover the canonical "
                "unknowns exactly once");

            if (m_single_precision)
            {
                m_engine_single = std::make_unique<ResistiveBandedEngine<float>>();
                m_engine_single->Allocate(num_blocks, block_size, m_map.bandwidth);
            }
            else
            {
                m_engine_double = std::make_unique<ResistiveBandedEngine<double>>();
                m_engine_double->Allocate(num_blocks, block_size, m_map.bandwidth);
            }
            m_is_defined = true;
        }

        [[nodiscard]] bool IsDefined () const { return m_is_defined; }

        /** \brief Scatter the emission values into the band tensor. */
        void
        AssembleValues (const ResistiveStencilConstants& a_constants,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge = {
                            nullptr, nullptr, nullptr},
                        const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge = {
                            nullptr, nullptr, nullptr})
        {
            BL_PROFILE("ResistiveBanded::AssembleValues()");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                IsDefined(),
                "ResistiveBandedSolver::AssembleValues() called before Define()");
            amrex::Gpu::DeviceVector<amrex::Real> accounting(2, 0.0);
            for (int component = 0; component < 3; ++component)
            {
                if (!m_map.component_active[component])
                {
                    continue;
                }
                if (m_engine_single)
                {
                    if (component == FirstActiveComponent())
                    {
                        m_engine_single->ResetToIdentity();
                    }
                    banded_detail::ScatterComponent(
                        a_constants, m_map, *m_owner[component], component,
                        a_eta_edge, a_eta_h_edge, a_hall_edge, a_inertia_edge,
                        m_engine_single->m_bands.data(), accounting.data());
                }
                else
                {
                    if (component == FirstActiveComponent())
                    {
                        m_engine_double->ResetToIdentity();
                    }
                    banded_detail::ScatterComponent(
                        a_constants, m_map, *m_owner[component], component,
                        a_eta_edge, a_eta_h_edge, a_hall_edge, a_inertia_edge,
                        m_engine_double->m_bands.data(), accounting.data());
                }
            }

            amrex::Array<amrex::Real, 2> totals = {0.0, 0.0};
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, accounting.begin(),
                             accounting.end(), totals.begin());
            m_dropped_fraction = totals[0];
            // The folded periodic ordering keeps every coupling in band,
            // so any drop is an assembly bug, never PC-grade truncation.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_dropped_fraction == 0.0,
                "ResistiveBandedSolver assembled out-of-band entries (stencil "
                "reach exceeds the declared bandwidth)");
        }

        /** \brief In-place block-banded LU of the assembled tensor. */
        void
        Factorize ()
        {
            BL_PROFILE("ResistiveBanded::Factorize()");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                IsDefined(),
                "ResistiveBandedSolver::Factorize() called before Define()");
            if (m_engine_single)
            {
                m_engine_single->Factorize();
            }
            else
            {
                m_engine_double->Factorize();
            }
        }

        /** \brief In-place x := (I + A)^{-1} x on the face components. */
        void
        Solve (const amrex::Array<amrex::MultiFab*, 3>& a_faces)
        {
            BL_PROFILE("ResistiveBanded::Solve()");

            if (m_engine_single)
            {
                SolveWith(*m_engine_single, a_faces);
            }
            else
            {
                SolveWith(*m_engine_double, a_faces);
            }
        }

        /** \brief Summed |value| of out-of-band entries of the last
         * assembly (asserted zero; kept for reporting). */
        [[nodiscard]] amrex::Real
        DroppedFraction () const
        {
            return m_dropped_fraction;
        }

    private:
        [[nodiscard]] int
        FirstActiveComponent () const
        {
            for (int component = 0; component < 3; ++component)
            {
                if (m_map.component_active[component])
                {
                    return component;
                }
            }
            return -1;
        }

        template <typename TS>
        void
        SolveWith (ResistiveBandedEngine<TS>& a_engine,
                   const amrex::Array<amrex::MultiFab*, 3>& a_faces)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_engine.m_factored,
                "ResistiveBandedSolver::Solve() called before Factorize()");
            banded_detail::ZeroVector(a_engine.m_rhs.data(), a_engine.m_rhs.size());
            for (int component = 0; component < 3; ++component)
            {
                if (m_map.component_active[component])
                {
                    banded_detail::GatherComponent(m_map, *m_owner[component],
                                                   component, *a_faces[component],
                                                   a_engine.m_rhs.data());
                }
            }
            a_engine.Sweep();
            for (int component = 0; component < 3; ++component)
            {
                if (m_map.component_active[component])
                {
                    banded_detail::ScatterSolution(m_map, component,
                                                   *a_faces[component],
                                                   a_engine.m_solution.data());
                }
            }
            amrex::Gpu::streamSynchronize();
        }

        bool m_is_defined = false;
        bool m_single_precision = true;
        amrex::Real m_dropped_fraction = 0.0;

        amrex::Geometry m_geometry;
        BandedIndexMap m_map;
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> m_owner;

        std::unique_ptr<ResistiveBandedEngine<float>> m_engine_single;
        std::unique_ptr<ResistiveBandedEngine<double>> m_engine_double;
};

} // namespace warpx::mhd_pc

#endif // MHD_RESISTIVE_BANDED_H_
