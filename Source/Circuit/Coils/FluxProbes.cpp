/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FluxProbes.H"

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

namespace
{
    /** One geometry-keyed owner-mask cache entry. The stored
     * BoxArray/DistributionMapping copies keep the layout refs alive, so
     * the pointer-identity hit below can never alias a rebuilt layout
     * that happened to reuse the address of a destroyed one. */
    struct OwnerMaskEntry
    {
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::IndexType typ;
        amrex::IntVect period;
        std::unique_ptr<amrex::iMultiFab> mask;
    };
    std::vector<OwnerMaskEntry> owner_mask_cache;
}

namespace warpx::circuit
{

const amrex::iMultiFab&
CachedOwnerMask (const amrex::MultiFab& mf, const amrex::Periodicity& period)
{
    // The cache outlives every consumer but must not outlive AMReX (the
    // masks' arena is torn down at Finalize; releasing them from a
    // static destructor would abort).
    static bool finalize_registered = false;
    if (!finalize_registered) {
        amrex::ExecOnFinalize([] () { owner_mask_cache.clear(); });
        finalize_registered = true;
    }
    const amrex::IntVect period_vect = period.intVect();
    for (const OwnerMaskEntry& entry : owner_mask_cache) {
        if (entry.typ == mf.ixType() && entry.period == period_vect &&
            entry.ba == mf.boxArray() &&
            entry.dm == mf.DistributionMap()) {
            return *entry.mask;
        }
    }
    // A regrid retires the old BoxArray/DistributionMapping refs for the
    // lifetime of the run; drop stale entries rather than accumulating
    // them (the cache holds at most a handful of live staggerings).
    constexpr std::size_t max_entries = 16;
    if (owner_mask_cache.size() >= max_entries) {
        owner_mask_cache.clear();
    }
    owner_mask_cache.push_back(OwnerMaskEntry{
        mf.boxArray(), mf.DistributionMap(), mf.ixType(), period_vect,
        mf.OwnerMask(period)});
    return *owner_mask_cache.back().mask;
}

amrex::Real
DiskFluxLinkage (const Coil& coil, const amrex::MultiFab& Bz)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(coil, Bz);
    WARPX_ABORT_WITH_MESSAGE(
        "DiskFluxLinkage is an RZ (m = 0) measurement");
    return 0.0_rt;
#else
    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(0);
    const double dr = geom.CellSize(0);
    const double dz = geom.CellSize(1);
    const double plo_r = geom.ProbLo(0);
    const double plo_z = geom.ProbLo(1);
    const int nz = geom.Domain().length(1);

    double r_off, z_off;
    QuarterOffset(coil.r, coil.z, dr, dz, r_off, z_off);

    // The two bracketing node planes and the (unclamped) linear weight --
    // identical rules to the discrete self-inductance.
    int j0 = static_cast<int>(std::floor((z_off - plo_z) / dz));
    if (j0 < 0) { j0 = 0; }
    if (j0 > nz - 1) { j0 = nz - 1; }
    const double w = (z_off - (plo_z + j0 * dz)) / dz;

    // Bz is nodal in z: boxes share the bracketing node planes, so count
    // each shared point once.
    const amrex::iMultiFab& owner = CachedOwnerMask(Bz, geom.periodicity());

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<double> reduce_data(reduce_op);
    for (MFIter mfi(Bz, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box tb = mfi.tilebox();
        const auto bz = Bz.const_array(mfi);
        const auto msk = owner.const_array(mfi);
        const int jj0 = j0;
        const double ww = w;
        const double roff = r_off;
        reduce_op.eval(tb, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> GpuTuple<double>
            {
                if (j != jj0 && j != jj0 + 1) { return 0.0; }
                if (msk(i, j, 0) == 0) { return 0.0; }
                const double r_c = plo_r + (i + 0.5) * dr;
                if (!(r_c < roff)) { return 0.0; }
                const double weight = (j == jj0) ? (1.0 - ww) : ww;
                return weight * static_cast<double>(bz(i, j, 0, 0)) * r_c;
            });
    }
    double flux = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealSum(flux);
    flux *= 2.0 * ablastr::coils::pi_ring * dr;

    return static_cast<amrex::Real>(flux * coil.I_ref * coil.n_turns);
#endif
}

amrex::Real
ReciprocityLinkage (const amrex::MultiFab& A_theta,
                    const amrex::MultiFab& J_theta)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(A_theta, J_theta);
    WARPX_ABORT_WITH_MESSAGE(
        "ReciprocityLinkage is an RZ (m = 0) measurement");
    return 0.0_rt;
#else
    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(0);
    const double dr = geom.CellSize(0);
    const double dz = geom.CellSize(1);
    const double plo_r = geom.ProbLo(0);
    const int nr = geom.Domain().length(0);

    // Both fields are fully nodal: shared nodes (box faces, periodic
    // images) must be counted once.
    const amrex::iMultiFab& owner =
        CachedOwnerMask(J_theta, geom.periodicity());

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<double> reduce_data(reduce_op);
    for (MFIter mfi(J_theta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box tb = mfi.tilebox(amrex::IntVect(1));
        const auto a = A_theta.const_array(mfi);
        const auto jt = J_theta.const_array(mfi);
        const auto msk = owner.const_array(mfi);
        const int nr_l = nr;
        reduce_op.eval(tb, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> GpuTuple<double>
            {
                if (msk(i, j, 0) == 0) { return 0.0; }
                const double r = plo_r + i * dr;
                // trapezoid end-weights in r only
                const double w_r = (i == 0 || i == nr_l) ? 0.5 : 1.0;
                return w_r * r * static_cast<double>(a(i, j, 0, 0))
                       * static_cast<double>(jt(i, j, 0, 0));
            });
    }
    double lam = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealSum(lam);
    lam *= 2.0 * ablastr::coils::pi_ring * dr * dz;

    return static_cast<amrex::Real>(lam);
#endif
}

amrex::Real
CouplingPowerIntegral (const std::array<const amrex::MultiFab*, 3>& J,
                       const std::array<const amrex::MultiFab*, 3>& E)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(J, E);
    WARPX_ABORT_WITH_MESSAGE(
        "CouplingPowerIntegral is an RZ (m = 0) measurement");
    return 0.0_rt;
#else
    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(0);
    const double dr = geom.CellSize(0);
    const double dz = geom.CellSize(1);
    const double plo_r = geom.ProbLo(0);

    const int nr = geom.Domain().length(0);
    double total = 0.0;
    for (int idir = 0; idir < 3; ++idir) {
        const amrex::MultiFab& jmf = *J[idir];
        const amrex::MultiFab& emf = *E[idir];
        const amrex::iMultiFab& owner =
            CachedOwnerMask(jmf, geom.periodicity());
        const bool r_nodal = jmf.ixType().nodeCentered(0);

        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<double> reduce_data(reduce_op);
        for (MFIter mfi(jmf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box tb = mfi.tilebox();
            const auto ja = jmf.const_array(mfi);
            const auto ea = emf.const_array(mfi);
            const auto msk = owner.const_array(mfi);
            const int nr_l = nr;
            reduce_op.eval(tb, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) -> GpuTuple<double>
                {
                    if (msk(i, j, 0) == 0) { return 0.0; }
                    const double r = plo_r + (r_nodal ? i : (i + 0.5)) * dr;
                    // the same trapezoid end-weights in r as the
                    // reciprocity linkage, so the coupling-power double
                    // entry closes exactly
                    const double w_r =
                        (r_nodal && (i == 0 || i == nr_l)) ? 0.5 : 1.0;
                    return w_r * r * static_cast<double>(ja(i, j, 0, 0))
                               * static_cast<double>(ea(i, j, 0, 0));
                });
        }
        total += amrex::get<0>(reduce_data.value(reduce_op));
    }
    ParallelDescriptor::ReduceRealSum(total);
    total *= 2.0 * ablastr::coils::pi_ring * dr * dz;
    return static_cast<amrex::Real>(total);
#endif
}

void
LinkageBatch::BuildPack (const CoilSet& coils,
                         const std::vector<ProbeKind>& probes,
                         const std::vector<const amrex::MultiFab*>& a_theta,
                         const amrex::MultiFab* bz,
                         const amrex::MultiFab* j_theta)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(coils, probes, a_theta, bz, j_theta);
    WARPX_ABORT_WITH_MESSAGE(
        "LinkageBatch is an RZ (m = 0) measurement");
#else
    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(0);
    const double dr = geom.CellSize(0);
    const double dz = geom.CellSize(1);
    const double plo_r = geom.ProbLo(0);
    const double plo_z = geom.ProbLo(1);
    const int nr = geom.Domain().length(0);
    const int nz = geom.Domain().length(1);

    m_ncoils = coils.size();
    m_jobs_j.clear();
    m_jobs_bz.clear();

    constexpr int threads_per_block = 256;
    constexpr int max_blocks_per_job = 64;

    long weight_total = 0;
    long partial_total = 0;
    std::vector<std::vector<std::pair<long, int>>> row_segments(m_ncoils);

    auto append_job = [&](std::map<int, std::vector<Job>>& jobs,
                          const int box_index, const int row,
                          const amrex::Box& box)
    {
        const long ncells = static_cast<long>(box.numPts());
        const int nblocks = static_cast<int>(std::min<long>(
            max_blocks_per_job,
            (ncells + threads_per_block - 1) / threads_per_block));
        jobs[box_index].push_back(
            Job{row, box, weight_total, partial_total, nblocks});
        row_segments[row].emplace_back(partial_total, nblocks);
        weight_total += ncells;
        partial_total += nblocks;
        return jobs[box_index].back();
    };

    // Enumerate the jobs (host) in a fixed order: reciprocity rows over
    // the J boxes, then disk rows over the Bz boxes.
    if (j_theta != nullptr) {
        for (amrex::MFIter mfi(*j_theta); mfi.isValid(); ++mfi) {
            const amrex::Box vb = mfi.validbox();
            for (int ic = 0; ic < m_ncoils; ++ic) {
                if (probes[ic] == ProbeKind::reciprocity) {
                    append_job(m_jobs_j, mfi.index(), ic, vb);
                }
            }
        }
    }
    if (bz != nullptr) {
        for (amrex::MFIter mfi(*bz); mfi.isValid(); ++mfi) {
            const amrex::Box vb = mfi.validbox();
            for (int ic = 0; ic < m_ncoils; ++ic) {
                if (probes[ic] != ProbeKind::disk) { continue; }
                const Coil& c = coils.coil(ic);
                double r_off, z_off;
                QuarterOffset(c.r, c.z, dr, dz, r_off, z_off);
                int j0 = static_cast<int>(std::floor((z_off - plo_z) / dz));
                if (j0 < 0) { j0 = 0; }
                if (j0 > nz - 1) { j0 = nz - 1; }
                amrex::Box planes = vb;
                planes.setSmall(1, std::max(vb.smallEnd(1), j0));
                planes.setBig(1, std::min(vb.bigEnd(1), j0 + 1));
                if (planes.isEmpty()) { continue; }
                append_job(m_jobs_bz, mfi.index(), ic, planes);
            }
        }
    }

    m_weights.resize(weight_total);
    m_partials.resize(partial_total);
    m_lambda_device.resize(m_ncoils);
    m_lambda_host.resize(m_ncoils);

    // Fixed-order combine segments per row.
    std::vector<int> seg_begin(m_ncoils + 1, 0);
    std::vector<long> seg_partial_offset;
    std::vector<int> seg_nblocks;
    for (int ic = 0; ic < m_ncoils; ++ic) {
        seg_begin[ic + 1] = seg_begin[ic]
            + static_cast<int>(row_segments[ic].size());
        for (const auto& [offset, nblocks] : row_segments[ic]) {
            seg_partial_offset.push_back(offset);
            seg_nblocks.push_back(nblocks);
        }
    }
    m_seg_begin.resize(seg_begin.size());
    m_seg_partial_offset.resize(std::max<std::size_t>(1, seg_partial_offset.size()));
    m_seg_nblocks.resize(std::max<std::size_t>(1, seg_nblocks.size()));
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, seg_begin.begin(),
                          seg_begin.end(), m_seg_begin.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, seg_partial_offset.begin(),
                          seg_partial_offset.end(), m_seg_partial_offset.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, seg_nblocks.begin(),
                          seg_nblocks.end(), m_seg_nblocks.begin());

    // Fill the weight tables device-side with the single-coil probes'
    // exact integrand factors (owner masks, radius/trapezoid/plane
    // weights, volume factors, the disk probe's reference amp-turns).
    double* const AMREX_RESTRICT weights = m_weights.data();
    if (j_theta != nullptr) {
        const amrex::iMultiFab& owner =
            CachedOwnerMask(*j_theta, geom.periodicity());
        for (amrex::MFIter mfi(*j_theta); mfi.isValid(); ++mfi) {
            const auto it = m_jobs_j.find(mfi.index());
            if (it == m_jobs_j.end()) { continue; }
            const auto msk = owner.const_array(mfi);
            for (const Job& job : it->second) {
                const auto a = a_theta[job.row]->const_array(mfi);
                const amrex::Box box = job.box;
                const long w_offset = job.weight_offset;
                const int nr_l = nr;
                const double factor =
                    2.0 * ablastr::coils::pi_ring * dr * dz;
                const double plo_r_l = plo_r;
                const double dr_l = dr;
                amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
                    {
                        const amrex::Box b = box;
                        const long lin = b.index(amrex::IntVect(i, j));
                        const double r = plo_r_l + i * dr_l;
                        const double w_r =
                            (i == 0 || i == nr_l) ? 0.5 : 1.0;
                        weights[w_offset + lin] =
                            (msk(i, j, 0) == 0)
                                ? 0.0
                                : w_r * r * factor
                                      * static_cast<double>(a(i, j, 0, 0));
                    });
            }
        }
    }
    if (bz != nullptr) {
        const amrex::iMultiFab& owner =
            CachedOwnerMask(*bz, geom.periodicity());
        for (amrex::MFIter mfi(*bz); mfi.isValid(); ++mfi) {
            const auto it = m_jobs_bz.find(mfi.index());
            if (it == m_jobs_bz.end()) { continue; }
            const auto msk = owner.const_array(mfi);
            for (const Job& job : it->second) {
                const Coil& c = coils.coil(job.row);
                double r_off, z_off;
                QuarterOffset(c.r, c.z, dr, dz, r_off, z_off);
                int j0 = static_cast<int>(std::floor((z_off - plo_z) / dz));
                if (j0 < 0) { j0 = 0; }
                if (j0 > nz - 1) { j0 = nz - 1; }
                const double w = (z_off - (plo_z + j0 * dz)) / dz;
                const amrex::Box box = job.box;
                const long w_offset = job.weight_offset;
                const double factor = 2.0 * ablastr::coils::pi_ring * dr
                    * c.I_ref * c.n_turns;
                const double plo_r_l = plo_r;
                const double dr_l = dr;
                const double roff_l = r_off;
                const int jj0 = j0;
                amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
                    {
                        const amrex::Box b = box;
                        const long lin = b.index(amrex::IntVect(i, j));
                        const double r_c = plo_r_l + (i + 0.5) * dr_l;
                        const double weight_j = (j == jj0) ? (1.0 - w) : w;
                        weights[w_offset + lin] =
                            (msk(i, j, 0) == 0 || !(r_c < roff_l))
                                ? 0.0
                                : weight_j * r_c * factor;
                    });
            }
        }
    }
    amrex::Gpu::streamSynchronize();

    m_key_ba_j = (j_theta != nullptr) ? j_theta->boxArray() : amrex::BoxArray{};
    m_key_dm_j = (j_theta != nullptr) ? j_theta->DistributionMap()
                                      : amrex::DistributionMapping{};
    m_key_ba_bz = (bz != nullptr) ? bz->boxArray() : amrex::BoxArray{};
    m_key_dm_bz = (bz != nullptr) ? bz->DistributionMap()
                                  : amrex::DistributionMapping{};
    m_key_domain = geom.Domain();
    m_built = true;
#endif
}

void
LinkageBatch::Measure (const CoilSet& coils,
                       const std::vector<ProbeKind>& probes,
                       const std::vector<const amrex::MultiFab*>& a_theta,
                       const amrex::MultiFab* bz,
                       const amrex::MultiFab* j_theta,
                       std::vector<amrex::Real>& lambda)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(coils, probes, a_theta, bz, j_theta);
    lambda.assign(coils.size(), 0.0);
    WARPX_ABORT_WITH_MESSAGE(
        "LinkageBatch is an RZ (m = 0) measurement");
#else
    lambda.assign(coils.size(), 0.0);
    if (coils.size() == 0) { return; }

    auto& warpx = WarpX::GetInstance();
    const auto& geom = warpx.Geom(0);
    const bool key_hit = m_built && m_key_domain == geom.Domain() &&
        (j_theta == nullptr ||
         (m_key_ba_j == j_theta->boxArray() &&
          m_key_dm_j == j_theta->DistributionMap())) &&
        (bz == nullptr ||
         (m_key_ba_bz == bz->boxArray() &&
          m_key_dm_bz == bz->DistributionMap()));
    if (!key_hit) {
        BuildPack(coils, probes, a_theta, bz, j_theta);
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
                const auto lo = amrex::lbound(box);
                amrex::launch(job.nblocks, threads_per_block,
                              amrex::Gpu::gpuStream(),
                    [=] AMREX_GPU_DEVICE () noexcept
                    {
                        double acc = 0.0;
                        const long stride =
                            static_cast<long>(blockDim.x) * gridDim.x;
                        for (long lin = static_cast<long>(blockIdx.x)
                                            * blockDim.x + threadIdx.x;
                             lin < ncells; lin += stride) {
                            const int i = lo.x + static_cast<int>(lin % nx);
                            const int j = lo.y + static_cast<int>(lin / nx);
                            acc += weights[w_offset + lin]
                                * static_cast<double>(field(i, j, 0, 0));
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
    if (j_theta != nullptr) { issue_jobs(*j_theta, m_jobs_j); }
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
                const auto lo = amrex::lbound(box);
                double acc = 0.0;
                for (long lin = 0; lin < ncells; ++lin) {
                    const int i = lo.x + static_cast<int>(lin % nx);
                    const int j = lo.y + static_cast<int>(lin / nx);
                    acc += weights[job.weight_offset + lin]
                        * static_cast<double>(field(i, j, 0, 0));
                }
                lambda[job.row] += static_cast<amrex::Real>(acc);
            }
        }
    };
    if (j_theta != nullptr) { run_jobs(*j_theta, m_jobs_j); }
    if (bz != nullptr) { run_jobs(*bz, m_jobs_bz); }
#endif

    ParallelDescriptor::ReduceRealSum(lambda.data(),
                                      static_cast<int>(lambda.size()));
#endif
}

} // namespace warpx::circuit
