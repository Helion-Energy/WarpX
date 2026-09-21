/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FluxProbes.H"
#include "DeviceReduction.H"

#include "YeeLoopKernel.H"

#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuReduce.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using namespace amrex;

namespace warpx::circuit
{

void
LinkageBatch::Measure (const CoilSet& coils,
                       const std::vector<ProbeKind>& probes,
                       const std::vector<double>& exclusion_radius,
                       const std::vector<VectorFieldPtrs>& a_ext,
                       const amrex::MultiFab* bz, const VectorFieldPtrs& j,
                       std::vector<amrex::Real>& lambda)
{
    BL_PROFILE("warpx::circuit::LinkageBatch::Measure");
    MeasureImpl(coils, probes, exclusion_radius, a_ext, bz, j, lambda, false);
}

const amrex::Gpu::DeviceVector<double>&
LinkageBatch::MeasureDevice (const CoilSet& coils,
                       const std::vector<ProbeKind>& probes,
                       const std::vector<double>& exclusion_radius,
                       const std::vector<VectorFieldPtrs>& a_ext,
                       const amrex::MultiFab* bz, const VectorFieldPtrs& j)
{
    BL_PROFILE("warpx::circuit::LinkageBatch::MeasureDevice");
    std::vector<amrex::Real> unused;
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    MeasureImpl(coils, probes, exclusion_radius, a_ext, bz, j, unused, true);
#elif !defined(AMREX_USE_GPU)
    // CPU oracle for exactly the same device response/scale algorithm.
    MeasureImpl(coils, probes, exclusion_radius, a_ext, bz, j, unused, false);
    m_lambda_device.resize(unused.size());
    std::copy(unused.begin(), unused.end(), m_lambda_device.begin());
#else
    WARPX_ABORT_WITH_MESSAGE("Device circuit probes require CUDA, HIP, or CPU");
#endif
    return m_lambda_device;
}

void
LinkageBatch::MeasureImpl (const CoilSet& coils,
                       const std::vector<ProbeKind>& probes,
                       const std::vector<double>& exclusion_radius,
                       const std::vector<VectorFieldPtrs>& a_ext,
                       const amrex::MultiFab* bz,
                       const VectorFieldPtrs& j,
                       std::vector<amrex::Real>& lambda, bool device_only)
{
#if !defined(AMREX_USE_CUDA) && !defined(AMREX_USE_HIP)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!device_only,
        "Device circuit probes currently require CUDA or HIP");
#endif
    // Both interfaces share the same local reduction tree. The host reference
    // copies the coil vector before MPI; device callers use GPU-aware MPI.
    BL_PROFILE("warpx::circuit::LinkageBatch::MeasureImpl");
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
    amrex::ignore_unused(coils, probes, exclusion_radius, a_ext, bz, j);
    lambda.assign(coils.size(), 0.0);
    WARPX_ABORT_WITH_MESSAGE(
        "LinkageBatch is implemented in RZ (m = 0) and 3D geometry");
#else
    if (!device_only) { lambda.assign(coils.size(), 0.0); }
    if (coils.size() == 0) { m_lambda_device.clear(); return; }

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(0);
    bool key_hit = m_built && m_key_domain == geom.Domain() &&
        (bz == nullptr ||
         (m_key_ba_bz == bz->boxArray() &&
          m_key_dm_bz == bz->DistributionMap()));
    for (int c = 0; c < 3; ++c) {
        key_hit = key_hit &&
            (j[c] == nullptr ||
             (m_key_ba_j[c] == j[c]->boxArray() &&
              m_key_dm_j[c] == j[c]->DistributionMap()));
    }
    if (!key_hit) {
        BuildPack(coils, probes, exclusion_radius, a_ext, bz, j);
    }

    const double* const AMREX_RESTRICT weights = m_weights.data();

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    constexpr int threads_per_block = 256;
    double* const AMREX_RESTRICT partials = m_partials.data();
    auto issue_jobs = [&](const amrex::MultiFab& src,
                          const std::map<int, std::vector<Job>>& jobs)
    {
        for (amrex::MFIter mfi(src); mfi.isValid(); ++mfi) {
            const auto it = jobs.find(mfi.index());
            if (it == jobs.end()) { continue; }
            const auto field = src.const_array(mfi);
            for (const Job& job : it->second) {
                const amrex::Box box = job.box;
                const long ncells = static_cast<long>(box.numPts());
                const long w_offset = job.weight_offset;
                const long p_offset = job.partial_offset;
                const int nx = box.length(0);
                const int ny = box.length(1);
                const auto lo = amrex::lbound(box);
                amrex::launch(job.nblocks, threads_per_block,
                              amrex::Gpu::gpuStream(),
                    [=] AMREX_GPU_DEVICE () noexcept
                    {
                        double acc = 0.0;
                        const long stride =
                            static_cast<long>(blockDim.x) * gridDim.x;
                        const long nxy = static_cast<long>(nx) * ny;
                        for (long lin = static_cast<long>(blockIdx.x)
                                            * blockDim.x + threadIdx.x;
                             lin < ncells; lin += stride) {
                            // Fortran-order (i, j, k) of the box's linear
                            // index (Box::index); k is 0 in 2D/RZ.
                            const int i = lo.x + static_cast<int>(lin % nx);
                            const int jj = lo.y + static_cast<int>((lin / nx) % ny);
                            const int k = lo.z + static_cast<int>(lin / nxy);
                            acc += weights[w_offset + lin]
                                * static_cast<double>(field(i, jj, k, 0));
                        }
                        const double block_sum =
                            amrex::Gpu::blockReduceSum(acc);
                        if (threadIdx.x == 0) {
                            partials[p_offset + blockIdx.x] = block_sum;
                        }
                    });
            }
        }
    };
    for (int c = 0; c < 3; ++c) {
        if (j[c] != nullptr && !m_jobs_j[c].empty()) {
            issue_jobs(*j[c], m_jobs_j[c]);
        }
    }
    if (bz != nullptr) { issue_jobs(*bz, m_jobs_bz); }

    // Fixed-order combine into the coil vector (one thread per row).
    {
        const int ncoils = m_ncoils;
        const int* const AMREX_RESTRICT seg_begin = m_seg_begin.data();
        const long* const AMREX_RESTRICT seg_offset =
            m_seg_partial_offset.data();
        const int* const AMREX_RESTRICT seg_nblocks = m_seg_nblocks.data();
        const double* const AMREX_RESTRICT partials_c = m_partials.data();
        double* const AMREX_RESTRICT lambda_device = m_lambda_device.data();
        amrex::launch(1, threads_per_block, amrex::Gpu::gpuStream(),
            [=] AMREX_GPU_DEVICE () noexcept
            {
                for (int row = static_cast<int>(threadIdx.x); row < ncoils;
                     row += blockDim.x) {
                    double acc = 0.0;
                    for (int k = seg_begin[row]; k < seg_begin[row + 1];
                         ++k) {
                        for (int b = 0; b < seg_nblocks[k]; ++b) {
                            acc += partials_c[seg_offset[k] + b];
                        }
                    }
                    lambda_device[row] = acc;
                }
            });
    }

    if (device_only) {
        SumDeviceCoils(m_lambda_device, m_exchange_device);
        return;
    }

    // THE one host synchronization of the measurement.
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, m_lambda_device.begin(),
                          m_lambda_device.end(), m_lambda_host.begin());
    amrex::Gpu::streamSynchronize();
    for (int ic = 0; ic < m_ncoils; ++ic) {
        lambda[ic] = static_cast<amrex::Real>(m_lambda_host[ic]);
    }
#else
    // Host path (also the SYCL fallback via the host arena): ordered
    // per-job accumulation over the same weight tables -- deterministic,
    // no reduction framework involved.
    auto run_jobs = [&](const amrex::MultiFab& src,
                        const std::map<int, std::vector<Job>>& jobs)
    {
        for (amrex::MFIter mfi(src); mfi.isValid(); ++mfi) {
            const auto it = jobs.find(mfi.index());
            if (it == jobs.end()) { continue; }
            const auto field = src.const_array(mfi);
            for (const Job& job : it->second) {
                const amrex::Box box = job.box;
                const long ncells = static_cast<long>(box.numPts());
                const int nx = box.length(0);
                const int ny = box.length(1);
                const long nxy = static_cast<long>(nx) * ny;
                const auto lo = amrex::lbound(box);
                double acc = 0.0;
                for (long lin = 0; lin < ncells; ++lin) {
                    const int i = lo.x + static_cast<int>(lin % nx);
                    const int jj = lo.y + static_cast<int>((lin / nx) % ny);
                    const int k = lo.z + static_cast<int>(lin / nxy);
                    acc += weights[job.weight_offset + lin]
                        * static_cast<double>(field(i, jj, k, 0));
                }
                lambda[job.row] += static_cast<amrex::Real>(acc);
            }
        }
    };
    for (int c = 0; c < 3; ++c) {
        if (j[c] != nullptr && !m_jobs_j[c].empty()) {
            run_jobs(*j[c], m_jobs_j[c]);
        }
    }
    if (bz != nullptr) { run_jobs(*bz, m_jobs_bz); }
#endif

    ParallelDescriptor::ReduceRealSum(lambda.data(),
                                      static_cast<int>(lambda.size()));
#endif
}

} // namespace warpx::circuit
