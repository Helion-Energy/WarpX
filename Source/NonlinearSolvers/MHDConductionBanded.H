/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_CONDUCTION_BANDED_H_
#define MHD_CONDUCTION_BANDED_H_

#include "MHDConductionDirect.H"
#include "MHDConductionStencil.H"
#include "MHDResistiveBanded.H"
#include "Utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <memory>

namespace warpx::mhd_pc
{

namespace conduction_banded_detail
{

/**
 * \brief Scatter the rows of one conduction channel into a channel's band
 * tensor (see ResistiveBandedEngine): block = the folded elimination
 * position of the cell's banded (z) index, local = the radial index in the
 * block. Every (row, column) slot is written by the row's own thread, so
 * no atomics are needed; out-of-band couplings are accounted (never taken
 * with the folded periodic ordering; the caller asserts).
 */
template <typename TS>
void
ScatterChannel (const ConductionStencilConstants& a_constants, const BandedIndexMap& a_map,
                const int a_channel, const amrex::GpuArray<int, 3>& a_component,
                const amrex::MultiFab& a_acoef,
                const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef,
                TS* const a_bands, amrex::Real* const a_accounting)
{
    for (amrex::MFIter mfi(a_acoef); mfi.isValid(); ++mfi)
    {
        const ConductionStencilArrays arrays =
            conduction_detail::MakeStencilArrays(mfi, a_acoef, a_bcoef, a_component);
        const BandedIndexMap map = a_map;
        const ConductionStencilConstants constants = a_constants;
        const int channel = a_channel;
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const int row_order = map.BlockOrder(map.BandedIndex(i, j));
                const int row_local = map.LocalIndex(0, i, j);
                const int size = map.block_size;
                const int bandwidth = map.bandwidth;
                const std::size_t stride = static_cast<std::size_t>(size) * size;
                EmitConductionRow(
                    constants, arrays, channel, i, j, k,
                    [&] (const int ic, const int jc, const int kc, const amrex::Real a_value)
                    {
                        amrex::ignore_unused(kc);
                        const int column_order = map.BlockOrder(map.BandedIndex(ic, jc));
                        const int band = column_order - row_order;
                        if (band < -bandwidth || band > bandwidth)
                        {
                            amrex::Gpu::Atomic::AddNoRet(&a_accounting[0], std::abs(a_value));
                            return;
                        }
                        const int column_local = map.LocalIndex(0, ic, jc);
                        a_bands[(static_cast<std::size_t>(row_order) * (2 * bandwidth + 1) +
                                 (bandwidth + band)) *
                                    stride +
                                static_cast<std::size_t>(row_local) * size + column_local] +=
                            static_cast<TS>(a_value);
                    });
            });
    }
}

/** \brief Gather one stacked component into a channel engine's canonical
 * (zero-padded) right-hand side. */
template <typename TS>
void
GatherChannel (const BandedIndexMap& a_map, const int a_component,
               const amrex::MultiFab& a_stacked, TS* const a_rhs)
{
    for (amrex::MFIter mfi(a_stacked); mfi.isValid(); ++mfi)
    {
        const auto values = a_stacked.const_array(mfi);
        const BandedIndexMap map = a_map;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               const int block = map.BlockOrder(map.BandedIndex(i, j));
                               const int local = map.LocalIndex(0, i, j);
                               a_rhs[static_cast<std::size_t>(block) * map.block_size +
                                     local] = static_cast<TS>(values(i, j, k, component));
                           });
    }
}

/** \brief Scatter a channel engine's solution into one stacked component. */
template <typename TS>
void
ScatterChannelSolution (const BandedIndexMap& a_map, const int a_component,
                        const TS* const a_solution, amrex::MultiFab& a_stacked)
{
    for (amrex::MFIter mfi(a_stacked); mfi.isValid(); ++mfi)
    {
        const auto values = a_stacked.array(mfi);
        const BandedIndexMap map = a_map;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               const int block = map.BlockOrder(map.BandedIndex(i, j));
                               const int local = map.LocalIndex(0, i, j);
                               values(i, j, k, component) = static_cast<amrex::Real>(
                                   a_solution[static_cast<std::size_t>(block) *
                                                  map.block_size +
                                              local]);
                           });
    }
}

} // namespace conduction_banded_detail

/**
 * \brief Block-banded LU inverse of the frozen conduction Helmholtz block
 * of the MHD block preconditioner (pc_mhd_block.conduction_solver =
 * banded): the portable exact inverse, with no cuDSS dependency.
 *
 * Each conduction channel is an independent scalar Helmholtz operator
 * (the same rows the direct solver assembles, see MHDConductionStencil.H),
 * so each channel gets its own block-banded factorization along z with
 * blocks dense over the radial line (block size = the number of radial
 * cells in RZ, 1 in 1D; bandwidth 1, doubled by the folded ring ordering
 * of a periodic z, see BandedIndexMap::BlockOrder) -- the resistive
 * banded engine's dense device kernels (ResistiveBandedEngine) with a
 * cell-centered scatter. Factorization storage is double precision (the
 * conduction rows reach grid numbers of 1e5-1e6 at the production clamp,
 * beyond FP32 headroom). Single-rank runs only, like the resistive banded
 * solver.
 */
class ConductionBandedSolver
{
    public:
        ConductionBandedSolver () = default;

        [[nodiscard]] static constexpr bool
        HaveBackend ()
        {
            return ResistiveBandedSolver::HaveBackend();
        }

        /**
         * \brief Build the block indexing and the per-channel engines.
         * a_component holds the stacked component of each conduction channel
         * (-1 for an absent channel).
         */
        void
        Define (const amrex::Geometry& a_geometry, const ConductionStencilConstants& a_constants,
                const amrex::GpuArray<int, 3>& a_component)
        {
            BL_PROFILE("ConductionBanded::Define()");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !m_is_defined, "ConductionBandedSolver::Define() called on a defined object");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                HaveBackend(),
                "pc_mhd_block.conduction_solver = banded is not implemented for SYCL builds");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                amrex::ParallelDescriptor::NProcs() == 1,
                "pc_mhd_block.conduction_solver = banded currently supports single-rank runs "
                "only (the block-banded elimination is sequential along z); use direct "
                "(cuDSS) or mlmg on multi-rank runs");

            m_geometry = a_geometry;
            m_constants = a_constants;
            m_component = a_component;
            const amrex::Box domain = m_geometry.Domain();
#if defined(WARPX_DIM_1D_Z)
            constexpr int banded_direction = 0;
            const int block_size = 1;
#else
            constexpr int banded_direction = AMREX_SPACEDIM - 1;
            const int block_size = domain.length(0);
#endif
            m_map.banded_low = domain.smallEnd(banded_direction);
            m_map.banded_cells = domain.length(banded_direction);
            m_map.banded_periodic = m_geometry.isPeriodic(banded_direction);
            m_map.bandwidth = m_map.banded_periodic ? 2 : 1;
            m_map.transverse_low = domain.smallEnd(0);
            m_map.component_active = {1, 0, 0};
            m_map.local_offset = {0, 0, 0};
            m_map.block_size = block_size;
            m_map.num_blocks = m_map.banded_cells;
#if (AMREX_SPACEDIM == 3)
            WARPX_ABORT_WITH_MESSAGE(
                "pc_mhd_block.conduction_solver = banded supports 1D and RZ only");
#endif

            int active_channels = 0;
            for (int channel = 0; channel < 3; ++channel)
            {
                if (m_component[channel] < 0)
                {
                    continue;
                }
                m_engine[channel] = std::make_unique<ResistiveBandedEngine<double>>();
                m_engine[channel]->Allocate(m_map.num_blocks, m_map.block_size,
                                            m_map.bandwidth);
                ++active_channels;
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                active_channels > 0, "ConductionBandedSolver found no conduction channels");
            m_is_defined = true;
        }

        [[nodiscard]] bool IsDefined () const { return m_is_defined; }

        /** \brief Scatter the rows of every channel into its band tensor. */
        void
        AssembleValues (const amrex::MultiFab& a_acoef,
                        const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef)
        {
            BL_PROFILE("ConductionBanded::AssembleValues()");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                IsDefined(), "ConductionBandedSolver::AssembleValues() called before Define()");
            amrex::Gpu::DeviceVector<amrex::Real> accounting(1, 0.0);
            for (int channel = 0; channel < 3; ++channel)
            {
                if (!m_engine[channel])
                {
                    continue;
                }
                ResistiveBandedEngine<double>& engine = *m_engine[channel];
                engine.m_factored = false;
                banded_detail::ZeroVector(engine.m_bands.data(), engine.m_bands.size());
                conduction_banded_detail::ScatterChannel(m_constants, m_map, channel,
                                                         m_component, a_acoef, a_bcoef,
                                                         engine.m_bands.data(),
                                                         accounting.data());
            }
            amrex::Real dropped = 0.0;
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, accounting.begin(), accounting.end(),
                             &dropped);
            m_dropped_fraction = dropped;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_dropped_fraction == 0.0,
                "ConductionBandedSolver assembled out-of-band entries (stencil reach exceeds "
                "the declared bandwidth)");
        }

        /** \brief In-place block-banded LU of every channel. */
        void
        Factorize ()
        {
            BL_PROFILE("ConductionBanded::Factorize()");

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                IsDefined(), "ConductionBandedSolver::Factorize() called before Define()");
            for (int channel = 0; channel < 3; ++channel)
            {
                if (m_engine[channel])
                {
                    m_engine[channel]->Factorize();
                }
            }
        }

        /** \brief solution[channels] := L^{-1} rhs[channels] on the stacked
         * cell MultiFabs (valid cells; other components untouched). */
        void
        Solve (const amrex::MultiFab& a_rhs, amrex::MultiFab& a_solution)
        {
            BL_PROFILE("ConductionBanded::Solve()");

            for (int channel = 0; channel < 3; ++channel)
            {
                if (!m_engine[channel])
                {
                    continue;
                }
                ResistiveBandedEngine<double>& engine = *m_engine[channel];
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    engine.m_factored,
                    "ConductionBandedSolver::Solve() called before Factorize()");
                banded_detail::ZeroVector(engine.m_rhs.data(), engine.m_rhs.size());
                conduction_banded_detail::GatherChannel(m_map, m_component[channel], a_rhs,
                                                        engine.m_rhs.data());
                engine.Sweep();
                conduction_banded_detail::ScatterChannelSolution(
                    m_map, m_component[channel], engine.m_solution.data(), a_solution);
            }
            amrex::Gpu::streamSynchronize();
        }

        [[nodiscard]] amrex::Real DroppedFraction () const { return m_dropped_fraction; }

    private:
        bool m_is_defined = false;
        amrex::Real m_dropped_fraction = 0.0;
        amrex::Geometry m_geometry;
        ConductionStencilConstants m_constants;
        amrex::GpuArray<int, 3> m_component = {-1, -1, -1};
        BandedIndexMap m_map;
        amrex::Array<std::unique_ptr<ResistiveBandedEngine<double>>, 3> m_engine;
};

} // namespace warpx::mhd_pc

#endif // MHD_CONDUCTION_BANDED_H_
