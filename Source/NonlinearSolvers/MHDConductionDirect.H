/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_CONDUCTION_DIRECT_H_
#define MHD_CONDUCTION_DIRECT_H_

#include "MHDConductionStencil.H"
#include "MHDResistiveDirect.H"
#include "Utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#if defined(WARPX_USE_CUDSS)
#include <cudss.h>
#endif

#if defined(AMREX_USE_MPI)
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

namespace warpx::mhd_pc
{

/**
 * \brief Device-copyable canonical unknown indexing of the conduction
 * direct solver: one unknown per cell per conduction channel, row-major
 * over the level's cell domain with the first direction fastest (the
 * radial line of a z block in RZ), periodic directions wrapped to the
 * domain. Cell-centered unknowns are never duplicated across boxes, so
 * every canonical row is owned by exactly one stored cell.
 */
struct ConductionIndexView
{
    amrex::GpuArray<amrex::Long, 3> channel_offset = {0, 0, 0};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_low = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_length = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_periodic = {};

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Long
    Index (const int a_channel, const int i, const int j, const int k) const
    {
        const int point[3] = {i, j, k};
        amrex::Long index = 0;
        for (int direction = AMREX_SPACEDIM - 1; direction >= 0; --direction)
        {
            int offset = point[direction] - domain_low[direction];
            if (domain_periodic[direction])
            {
                offset %= domain_length[direction];
                if (offset < 0)
                {
                    offset += domain_length[direction];
                }
            }
            index = index * domain_length[direction] + offset;
        }
        return channel_offset[a_channel] + index;
    }
};

namespace conduction_detail
{

/** \brief Fill the per-box stencil views of the frozen coefficients. */
inline ConductionStencilArrays
MakeStencilArrays (const amrex::MFIter& a_mfi, const amrex::MultiFab& a_acoef,
                   const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef,
                   const amrex::GpuArray<int, 3>& a_component)
{
    ConductionStencilArrays arrays;
    arrays.acoef = a_acoef.const_array(a_mfi);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        arrays.bcoef[direction] = a_bcoef[direction]->const_array(a_mfi);
        arrays.box_length[direction] = a_mfi.validbox().length(direction);
    }
    arrays.component = a_component;
    return arrays;
}

/**
 * \brief Recompute the rows of one channel on the device: re-run the
 * shared emission on the coefficient MultiFabs' device arrays and
 * accumulate into the frozen CSR slots through the entry-to-slot map in
 * the exact per-row order of the host pattern build. Every row belongs
 * to one thread, so no atomics are needed; emissions beyond the frozen
 * per-row count are dropped and counted (a structure change is an
 * assembly bug the caller asserts on).
 */
inline void
RefreshChannelValues (const ConductionStencilConstants& a_constants, const int a_channel,
                      const amrex::GpuArray<int, 3>& a_component,
                      const amrex::iMultiFab& a_entry_base, const amrex::MultiFab& a_acoef,
                      const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef,
                      const int* const a_entry_slot, amrex::Real* const a_values,
                      int* const a_structure_errors)
{
    for (amrex::MFIter mfi(a_entry_base); mfi.isValid(); ++mfi)
    {
        const ConductionStencilArrays arrays =
            MakeStencilArrays(mfi, a_acoef, a_bcoef, a_component);
        const auto entry_base = a_entry_base.const_array(mfi);
        const ConductionStencilConstants constants = a_constants;
        const int channel = a_channel;
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const int base = entry_base(i, j, k, 0);
                const int count = entry_base(i, j, k, 1);
                int emitted = 0;
                EmitConductionRow(
                    constants, arrays, channel, i, j, k,
                    [&] (const int ic, const int jc, const int kc, const amrex::Real a_value)
                    {
                        amrex::ignore_unused(ic, jc, kc); // columns are frozen in the map
                        if (emitted < count)
                        {
                            a_values[a_entry_slot[base + emitted]] += a_value;
                        }
                        ++emitted;
                    });
                if (emitted != count)
                {
                    amrex::Gpu::Atomic::AddNoRet(a_structure_errors, 1);
                }
            });
    }
}

/** \brief Gather one stacked component into the canonical device
 * right-hand side (every canonical element written exactly once). */
inline void
GatherChannelRhs (const ConductionIndexView& a_view, const int a_channel,
                  const int a_component, const amrex::MultiFab& a_stacked,
                  amrex::Real* const a_rhs)
{
    for (amrex::MFIter mfi(a_stacked); mfi.isValid(); ++mfi)
    {
        const auto values = a_stacked.const_array(mfi);
        const ConductionIndexView view = a_view;
        const int channel = a_channel;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               a_rhs[view.Index(channel, i, j, k)] =
                                   values(i, j, k, component);
                           });
    }
}

/** \brief Scatter the canonical device solution into one stacked
 * component (valid cells). */
inline void
ScatterChannelSolution (const ConductionIndexView& a_view, const int a_channel,
                        const int a_component, const amrex::Real* const a_solution,
                        amrex::MultiFab& a_stacked)
{
    for (amrex::MFIter mfi(a_stacked); mfi.isValid(); ++mfi)
    {
        const auto values = a_stacked.array(mfi);
        const ConductionIndexView view = a_view;
        const int channel = a_channel;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               values(i, j, k, component) =
                                   a_solution[view.Index(channel, i, j, k)];
                           });
    }
}

} // namespace conduction_detail

/**
 * \brief Exact sparse inverse of the frozen conduction Helmholtz block of
 * the MHD block preconditioner (see MHDBlockPC, conduction block and
 * pc_mhd_block.conduction_solver = direct).
 *
 * The matrix rows are the energy channels of the stacked cell-centered
 * Helmholtz the MLMG path applies -- the same cell and per-component face
 * coefficients (MHDBlockPC::m_signal_acoef / m_signal_bcoef, i.e. the
 * solver's frozen conduction diffusivities as theta dt chi, the RZ metric
 * folded in) with the domain boundaries eliminated exactly as AMReX's
 * MLABecLaplacian does (see MHDConductionStencil.H). The channels are
 * independent, so the assembled matrix is block diagonal over the
 * channels and one factorization serves all of them.
 *
 * The sparsity pattern is geometric and frozen at the first assembly;
 * each preconditioner update recomputes only the values from the
 * refrozen coefficients and refactorizes, and every application is a
 * single forward/backward substitution -- the exact block inverse, in
 * place of the two fixed V-cycles with a point smoother of the MLMG
 * path (which is known to degrade for per-direction anisotropy above
 * ~1e2, the regime of the Braginskii wall corner).
 *
 * Single-rank runs refresh the values device-resident through the frozen
 * entry-to-slot map and pack/unpack the right-hand side on the device;
 * multi-rank runs assemble through the host on the I/O rank (which
 * factorizes and solves for all ranks, the solution broadcast). In
 * validation mode (pc_mhd_block.conduction_validate_assembly) both value
 * paths run and are compared bitwise (<= 4 ULP), and MHDBlockPC compares
 * the assembled product against the MLMG operator's own matrix-free
 * application (CompareMatrixFreeAction). The numeric backend is NVIDIA
 * cuDSS (CUDA builds configured with WarpX_CUDSS=ON); assembly and the
 * validation are portable.
 */
class ConductionDirectSolver
{
    public:
        ConductionDirectSolver () = default;
        ~ConductionDirectSolver ();

        ConductionDirectSolver (const ConductionDirectSolver&) = delete;
        ConductionDirectSolver& operator= (const ConductionDirectSolver&) = delete;
        ConductionDirectSolver (ConductionDirectSolver&&) noexcept = delete;
        ConductionDirectSolver& operator= (ConductionDirectSolver&&) noexcept = delete;

        /** \brief Whether a direct factorization backend was compiled in. */
        [[nodiscard]] static constexpr bool
        HaveFactorizationBackend ()
        {
            return ResistiveDirectSolver::HaveFactorizationBackend();
        }

        /**
         * \brief Build the canonical indexing and the host mirrors.
         * a_component holds the stacked component index of each
         * conduction channel (-1 for an absent channel); a_acoef and
         * a_bcoef are the stacked operator's coefficient MultiFabs whose
         * layouts the mirrors copy.
         */
        void Define (const amrex::Geometry& a_geometry,
                     const ConductionStencilConstants& a_constants,
                     const amrex::GpuArray<int, 3>& a_component,
                     const amrex::MultiFab& a_acoef,
                     const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef);

        [[nodiscard]] bool IsDefined () const { return m_is_defined; }

        /** \brief Run the host assembly next to the device refresh at every
         * update so the two CSR value arrays can be compared bitwise. */
        void SetValidationMode (const bool a_validate) { m_validate_parity = a_validate; }

        [[nodiscard]] bool UsesDeviceAssembly () const { return m_use_device_assembly; }

        [[nodiscard]] amrex::Long NumRows () const { return m_num_rows; }

        /** \brief Number of stored nonzeros of the frozen pattern (I/O rank). */
        [[nodiscard]] amrex::Long NumValues () const
        {
            return static_cast<amrex::Long>(m_column_indices.size());
        }

        /** \brief Recompute the matrix values from the frozen coefficients
         * (the first call also freezes the pattern). */
        void AssembleValues (const amrex::MultiFab& a_acoef,
                             const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef);

        /** \brief Numeric factorization of the assembled matrix (cuDSS). */
        void Factorize ();

        /** \brief solution[channels] := L^{-1} rhs[channels] on the stacked
         * cell MultiFabs (valid cells; other components untouched). */
        void Solve (const amrex::MultiFab& a_rhs, amrex::MultiFab& a_solution);

        /** \brief Load the deterministic validation vector into the channel
         * components of a stacked MultiFab (valid cells). */
        void FillValidationVector (amrex::MultiFab& a_stacked);

        /**
         * \brief Compare a matrix-free action L x on the validation vector
         * (the channel components of a_action) against the assembled
         * product: returns the maximum absolute entry difference and the
         * maximum absolute entry of the assembled product as the scale
         * (both reduced over ranks).
         */
        [[nodiscard]] std::pair<amrex::Real, amrex::Real>
        CompareMatrixFreeAction (const amrex::MultiFab& a_action);

    private:
        [[nodiscard]] static amrex::Real
        ValidationValue (const amrex::Long a_index)
        {
            auto z = static_cast<std::uint64_t>(a_index) + 0x9E3779B97F4A7C15ULL;
            z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31U);
            return amrex::Real(2.0) *
                       (static_cast<amrex::Real>(z >> 11U) *
                        amrex::Real(1.1102230246251565e-16)) -
                   amrex::Real(1.0);
        }

        [[nodiscard]] static std::uint64_t
        OrderedBits (const amrex::Real a_value)
        {
            static_assert(sizeof(amrex::Real) == sizeof(std::uint64_t),
                          "the assembly parity gate expects FP64 amrex::Real");
            std::uint64_t bits = 0;
            std::memcpy(&bits, &a_value, sizeof(bits));
            return (bits >> 63U) != 0U ? ~bits : (bits | 0x8000000000000000ULL);
        }

        void EmitLocalRows ();
        void BuildGlobalPattern ();
        void FoldGlobalValues ();
        void UploadFrozenMaps ();
        void RefreshValuesOnDevice (
            const amrex::MultiFab& a_acoef,
            const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef);
        void CheckAssemblyParity () const;

        /** \brief Stage one stacked component through the pinned channel
         * mirror (device -> host). */
        void StageChannelToHost (const amrex::MultiFab& a_stacked, const int a_component);
        /** \brief Push the pinned channel mirror into one stacked component
         * (host -> device, valid cells). */
        void PushChannelToDevice (amrex::MultiFab& a_stacked, const int a_component);

        bool m_is_defined = false;
        bool m_pattern_built = false;
        bool m_use_device_assembly = false;
        bool m_validate_parity = false;

        amrex::Geometry m_geometry;
        ConductionStencilConstants m_constants;
        amrex::GpuArray<int, 3> m_component = {-1, -1, -1};
        amrex::Long m_rows_per_channel = 0;
        amrex::Long m_num_rows = 0;
        ConductionIndexView m_view;

        // Host mirrors (pinned in GPU builds) of the frozen coefficients and
        // a one-component cell scratch pair for the host transport paths.
        const amrex::MultiFab* m_acoef_source = nullptr;
        amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> m_bcoef_source = {};
        std::unique_ptr<amrex::MultiFab> m_acoef_host;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_bcoef_host;
        std::unique_ptr<amrex::MultiFab> m_channel_host;
        std::unique_ptr<amrex::MultiFab> m_channel_device;

        // Locally owned rows in fixed emission order.
        amrex::Long m_local_owned_rows = 0;
        amrex::Vector<int> m_local_entry_rows;
        amrex::Vector<int> m_local_entry_columns;
        amrex::Vector<amrex::Real> m_local_entry_values;

        // I/O-rank global matrix (CSR through the frozen entry-to-slot map).
        amrex::Vector<int> m_gather_counts;
        amrex::Vector<int> m_gather_offsets;
        amrex::Vector<amrex::Real> m_gathered_values;
        amrex::Vector<int> m_entry_slot;
        amrex::Vector<int> m_row_offsets;
        amrex::Vector<int> m_column_indices;
        amrex::Vector<amrex::Real> m_matrix_values;

        // Host solve vectors (full canonical length on every rank).
        amrex::Vector<amrex::Real> m_rhs;
        amrex::Vector<amrex::Real> m_solution;

        // Single-rank device assembly: per-channel frozen per-row emission
        // ranges (component 0 the first local entry, 1 the count), the
        // device entry-to-slot map, the device CSR values refreshed in place
        // and the structure-error counter.
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> m_entry_base_host;
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> m_entry_base_device;
        amrex::Gpu::DeviceVector<int> m_entry_slot_device;
        amrex::Gpu::DeviceVector<amrex::Real> m_values_device;
        amrex::Gpu::DeviceVector<int> m_structure_errors_device;

        amrex::Gpu::DeviceVector<amrex::Real> m_device_rhs;
        amrex::Gpu::DeviceVector<amrex::Real> m_device_solution;

#if defined(WARPX_USE_CUDSS)
        bool m_cudss_ready = false;
        cudssHandle_t m_cudss_handle = nullptr;
        cudssConfig_t m_cudss_config = nullptr;
        cudssData_t m_cudss_data = nullptr;
        cudssMatrix_t m_cudss_matrix = nullptr;
        cudssMatrix_t m_cudss_rhs = nullptr;
        cudssMatrix_t m_cudss_solution = nullptr;
        amrex::Gpu::DeviceVector<int> m_device_row_offsets;
        amrex::Gpu::DeviceVector<int> m_device_column_indices;
        amrex::Gpu::DeviceVector<amrex::Real> m_device_values;
#endif
};

#if defined(WARPX_USE_CUDSS)
#define WARPX_CUDSS_CHECK_CONDUCTION(call)                                        \
    {                                                                             \
        const cudssStatus_t warpx_cudss_status = (call);                          \
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(                                         \
            warpx_cudss_status == CUDSS_STATUS_SUCCESS,                           \
            "cuDSS call failed in the MHD conduction direct solver: " #call);     \
    }
#endif

inline
ConductionDirectSolver::~ConductionDirectSolver ()
{
#if defined(WARPX_USE_CUDSS)
    if (m_cudss_ready)
    {
        cudssMatrixDestroy(m_cudss_solution);
        cudssMatrixDestroy(m_cudss_rhs);
        cudssMatrixDestroy(m_cudss_matrix);
        cudssDataDestroy(m_cudss_handle, m_cudss_data);
        cudssConfigDestroy(m_cudss_config);
        cudssDestroy(m_cudss_handle);
        m_cudss_ready = false;
    }
#endif
}

inline void
ConductionDirectSolver::Define (
    const amrex::Geometry& a_geometry, const ConductionStencilConstants& a_constants,
    const amrex::GpuArray<int, 3>& a_component, const amrex::MultiFab& a_acoef,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef)
{
    BL_PROFILE("ConductionDirect::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined, "ConductionDirectSolver::Define() called on a defined object");
    m_geometry = a_geometry;
    m_constants = a_constants;
    m_component = a_component;
    const amrex::Box domain = m_geometry.Domain();

    m_rows_per_channel = domain.numPts();
    amrex::Long offset = 0;
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_component[channel] < 0)
        {
            continue;
        }
        m_view.channel_offset[channel] = offset;
        offset += m_rows_per_channel;
    }
    m_num_rows = offset;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_rows > 0 &&
            m_num_rows < static_cast<amrex::Long>(std::numeric_limits<int>::max()),
        "ConductionDirectSolver requires at least one conduction channel and fewer "
        "than 2^31 cell unknowns");
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_view.domain_low[direction] = domain.smallEnd(direction);
        m_view.domain_length[direction] = domain.length(direction);
        m_view.domain_periodic[direction] = m_geometry.isPeriodic(direction) ? 1 : 0;
    }

    // Single-rank runs refresh the values and transport the vectors on the
    // device; multi-rank runs assemble through the host gather.
    m_use_device_assembly = amrex::ParallelDescriptor::NProcs() == 1;

    m_acoef_source = &a_acoef;
    m_acoef_host = std::make_unique<amrex::MultiFab>(
        a_acoef.boxArray(), a_acoef.DistributionMap(), a_acoef.nComp(), a_acoef.nGrowVect(),
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_bcoef[direction] != nullptr,
            "ConductionDirectSolver requires the face coefficients of every direction");
        m_bcoef_source[direction] = a_bcoef[direction];
        const amrex::MultiFab& bcoef = *a_bcoef[direction];
        m_bcoef_host[direction] = std::make_unique<amrex::MultiFab>(
            bcoef.boxArray(), bcoef.DistributionMap(), bcoef.nComp(), bcoef.nGrowVect(),
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    }
    m_channel_host = std::make_unique<amrex::MultiFab>(
        a_acoef.boxArray(), a_acoef.DistributionMap(), 1, 0,
        amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    m_channel_device =
        std::make_unique<amrex::MultiFab>(a_acoef.boxArray(), a_acoef.DistributionMap(), 1, 0);

    if (m_use_device_assembly)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            if (m_component[channel] < 0)
            {
                continue;
            }
            m_entry_base_host[channel] = std::make_unique<amrex::iMultiFab>(
                a_acoef.boxArray(), a_acoef.DistributionMap(), 2, 0,
                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
            m_entry_base_host[channel]->setVal(-1, 0, 1, 0);
            m_entry_base_host[channel]->setVal(0, 1, 1, 0);
            m_entry_base_device[channel] = std::make_unique<amrex::iMultiFab>(
                a_acoef.boxArray(), a_acoef.DistributionMap(), 2, 0);
        }
    }

    m_rhs.resize(m_num_rows);
    m_solution.resize(m_num_rows);
    m_device_rhs.resize(m_num_rows);
    m_device_solution.resize(m_num_rows);
    m_is_defined = true;
}

inline void
ConductionDirectSolver::EmitLocalRows ()
{
    const bool build_pattern = !m_pattern_built;
    std::size_t entry = 0;
    amrex::Long owned_rows = 0;
    const auto record = [&] (const int a_row, const int a_column, const amrex::Real a_value)
    {
        if (build_pattern)
        {
            m_local_entry_rows.push_back(a_row);
            m_local_entry_columns.push_back(a_column);
            m_local_entry_values.push_back(a_value);
        }
        else
        {
            // A structure change between updates would otherwise write past
            // the frozen entry list before the structure assertion below
            // fires; the emission structure is geometric (Dirichlet order
            // from box length and max_order), so this never trips.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                entry < static_cast<std::size_t>(m_local_entry_values.size()),
                "ConductionDirectSolver stencil emission grew between updates (the frozen "
                "sparsity pattern was violated)");
            m_local_entry_values[entry] = a_value;
        }
        ++entry;
    };

    amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> bcoef_host;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        bcoef_host[direction] = m_bcoef_host[direction].get();
    }
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_component[channel] < 0)
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_acoef_host); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const ConductionStencilArrays arrays =
                conduction_detail::MakeStencilArrays(mfi, *m_acoef_host, bcoef_host,
                                                     m_component);
            amrex::Array4<int> entry_base;
            if (build_pattern && m_entry_base_host[channel])
            {
                entry_base = m_entry_base_host[channel]->array(mfi);
            }
            amrex::LoopOnCpu(
                box,
                [&] (int i, int j, int k)
                {
                    const amrex::Long row = m_view.Index(channel, i, j, k);
                    const auto row_index = static_cast<int>(row);
                    const std::size_t row_entry_start = entry;
                    ++owned_rows;
                    EmitConductionRow(
                        m_constants, arrays, channel, i, j, k,
                        [&] (const int ic, const int jc, const int kc,
                             const amrex::Real a_value)
                        {
                            const amrex::Long column = m_view.Index(channel, ic, jc, kc);
                            record(row_index, static_cast<int>(column), a_value);
                        });
                    if (entry_base)
                    {
                        entry_base(i, j, k, 0) = static_cast<int>(row_entry_start);
                        entry_base(i, j, k, 1) = static_cast<int>(entry - row_entry_start);
                    }
                });
        }
    }

    if (build_pattern)
    {
        m_local_owned_rows = owned_rows;
    }
    else
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            entry == static_cast<std::size_t>(m_local_entry_values.size()) &&
                owned_rows == m_local_owned_rows,
            "ConductionDirectSolver stencil emission changed structure between updates");
    }
}

inline void
ConductionDirectSolver::BuildGlobalPattern ()
{
    amrex::Long total_owned_rows = m_local_owned_rows;
    amrex::ParallelDescriptor::ReduceLongSum(total_owned_rows);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        total_owned_rows == m_num_rows,
        "ConductionDirectSolver ownership does not cover the canonical unknowns exactly "
        "once");

    const int io_rank = amrex::ParallelDescriptor::IOProcessorNumber();
    const int num_ranks = amrex::ParallelDescriptor::NProcs();
    const auto local_count = static_cast<int>(m_local_entry_rows.size());

    amrex::Vector<int> rows;
    amrex::Vector<int> columns;
    if (num_ranks == 1)
    {
        rows = m_local_entry_rows;
        columns = m_local_entry_columns;
        m_gather_counts = {local_count};
        m_gather_offsets = {0};
    }
#if defined(AMREX_USE_MPI)
    else
    {
        m_gather_counts.resize(num_ranks, 0);
        MPI_Gather(&local_count, 1, MPI_INT, m_gather_counts.data(), 1, MPI_INT, io_rank,
                   amrex::ParallelDescriptor::Communicator());
        m_gather_offsets.resize(num_ranks, 0);
        amrex::Long total = 0;
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            for (int rank = 0; rank < num_ranks; ++rank)
            {
                m_gather_offsets[rank] = static_cast<int>(total);
                total += m_gather_counts[rank];
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                total < static_cast<amrex::Long>(std::numeric_limits<int>::max()),
                "ConductionDirectSolver matrix exceeds 2^31 assembled entries");
            rows.resize(total);
            columns.resize(total);
        }
        MPI_Gatherv(m_local_entry_rows.data(), local_count, MPI_INT, rows.data(),
                    m_gather_counts.data(), m_gather_offsets.data(), MPI_INT, io_rank,
                    amrex::ParallelDescriptor::Communicator());
        MPI_Gatherv(m_local_entry_columns.data(), local_count, MPI_INT, columns.data(),
                    m_gather_counts.data(), m_gather_offsets.data(), MPI_INT, io_rank,
                    amrex::ParallelDescriptor::Communicator());
    }
#endif

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        const auto num_entries = static_cast<amrex::Long>(rows.size());
        amrex::Vector<amrex::Long> permutation(num_entries);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(),
                  [&] (const amrex::Long a, const amrex::Long b)
                  {
                      return rows[a] != rows[b] ? rows[a] < rows[b] : columns[a] < columns[b];
                  });

        m_entry_slot.resize(num_entries);
        m_row_offsets.assign(m_num_rows + 1, 0);
        m_column_indices.clear();
        int previous_row = -1;
        int previous_column = -1;
        for (const amrex::Long entry : permutation)
        {
            const int row = rows[entry];
            const int column = columns[entry];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                row >= 0 && row < m_num_rows && column >= 0 && column < m_num_rows,
                "ConductionDirectSolver assembled an out-of-range matrix entry");
            if (row != previous_row || column != previous_column)
            {
                m_column_indices.push_back(column);
                ++m_row_offsets[row + 1];
                previous_row = row;
                previous_column = column;
            }
            m_entry_slot[entry] = static_cast<int>(m_column_indices.size()) - 1;
        }
        for (amrex::Long row = 0; row < m_num_rows; ++row)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_row_offsets[row + 1] > 0,
                "ConductionDirectSolver left a canonical row unassembled");
            m_row_offsets[row + 1] += m_row_offsets[row];
        }
        m_matrix_values.assign(m_column_indices.size(), amrex::Real(0.0));
        m_gathered_values.resize(num_entries);
    }
    m_pattern_built = true;
}

inline void
ConductionDirectSolver::FoldGlobalValues ()
{
    const int io_rank = amrex::ParallelDescriptor::IOProcessorNumber();
    const int num_ranks = amrex::ParallelDescriptor::NProcs();
    const auto local_count = static_cast<int>(m_local_entry_values.size());
    amrex::ignore_unused(io_rank, local_count);

    const amrex::Real* gathered_values = m_local_entry_values.data();
    if (num_ranks > 1)
    {
#if defined(AMREX_USE_MPI)
        MPI_Gatherv(m_local_entry_values.data(), local_count,
                    amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                    m_gathered_values.data(), m_gather_counts.data(), m_gather_offsets.data(),
                    amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(), io_rank,
                    amrex::ParallelDescriptor::Communicator());
        gathered_values = m_gathered_values.data();
#endif
    }

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::fill(m_matrix_values.begin(), m_matrix_values.end(), amrex::Real(0.0));
        const amrex::Long num_entries = m_entry_slot.size();
        for (amrex::Long entry = 0; entry < num_entries; ++entry)
        {
            m_matrix_values[m_entry_slot[entry]] += gathered_values[entry];
        }
    }
}

inline void
ConductionDirectSolver::AssembleValues (
    const amrex::MultiFab& a_acoef,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef)
{
    BL_PROFILE("ConductionDirect::AssembleValues()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "ConductionDirectSolver::AssembleValues() called before Define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        &a_acoef == m_acoef_source,
        "ConductionDirectSolver cell coefficients changed after Define()");
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_bcoef[direction] == m_bcoef_source[direction],
            "ConductionDirectSolver face coefficients changed after Define()");
    }
    const bool first_assembly = !m_pattern_built;
    // The one-time pattern freeze runs on the host; afterwards the host
    // emission survives as the multi-rank path and the validation
    // reference. The single-rank production refreeze is the device
    // refresh alone.
    const bool host_assembly = first_assembly || !m_use_device_assembly || m_validate_parity;

    if (host_assembly)
    {
        amrex::dtoh_memcpy(*m_acoef_host, a_acoef);
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            amrex::dtoh_memcpy(*m_bcoef_host[direction], *a_bcoef[direction]);
        }
        amrex::Gpu::streamSynchronize();
        EmitLocalRows();
    }
    if (first_assembly)
    {
        BuildGlobalPattern();
        if (m_use_device_assembly)
        {
            UploadFrozenMaps();
        }
    }
    if (host_assembly)
    {
        FoldGlobalValues();
    }
    if (m_use_device_assembly)
    {
        RefreshValuesOnDevice(a_acoef, a_bcoef);
        if (host_assembly)
        {
            CheckAssemblyParity();
        }
    }
}

inline void
ConductionDirectSolver::UploadFrozenMaps ()
{
    // Single rank: the global entry order IS the local emission order, so
    // the entry-to-slot map and the per-row ranges index the same entries.
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_entry_base_host[channel])
        {
            amrex::htod_memcpy(*m_entry_base_device[channel], *m_entry_base_host[channel]);
        }
    }
    m_entry_slot_device.resize(m_entry_slot.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_entry_slot.begin(), m_entry_slot.end(),
                          m_entry_slot_device.begin());
    m_values_device.resize(m_matrix_values.size());
    m_structure_errors_device.resize(1);
    amrex::Gpu::streamSynchronize();
}

inline void
ConductionDirectSolver::RefreshValuesOnDevice (
    const amrex::MultiFab& a_acoef,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bcoef)
{
    BL_PROFILE("ConductionDirect::RefreshValuesOnDevice()");

    direct_detail::ZeroRealVector(m_values_device.data(), m_values_device.size());
    direct_detail::ZeroIntScalar(m_structure_errors_device.data());
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_component[channel] < 0)
        {
            continue;
        }
        conduction_detail::RefreshChannelValues(
            m_constants, channel, m_component, *m_entry_base_device[channel], a_acoef,
            a_bcoef, m_entry_slot_device.data(), m_values_device.data(),
            m_structure_errors_device.data());
    }
    int structure_errors = 0;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_structure_errors_device.begin(),
                     m_structure_errors_device.end(), &structure_errors);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        structure_errors == 0,
        "ConductionDirectSolver device value refresh changed the emission structure "
        "(the frozen sparsity pattern was violated)");
}

inline void
ConductionDirectSolver::CheckAssemblyParity () const
{
    const std::size_t num_values = m_matrix_values.size();
    amrex::Vector<amrex::Real> device_values(num_values);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_values_device.begin(), m_values_device.end(),
                     device_values.begin());
    std::uint64_t max_ulp = 0;
    amrex::Real max_difference = 0.0;
    for (std::size_t slot = 0; slot < num_values; ++slot)
    {
        const std::uint64_t host_bits = OrderedBits(m_matrix_values[slot]);
        const std::uint64_t device_bits = OrderedBits(device_values[slot]);
        max_ulp = std::max(max_ulp, host_bits > device_bits ? host_bits - device_bits
                                                            : device_bits - host_bits);
        max_difference = amrex::max(
            max_difference, std::abs(m_matrix_values[slot] - device_values[slot]));
    }
    amrex::Print() << "pc_mhd_block conduction direct assembly parity: max ULP = " << max_ulp
                   << " (max |device - host| = " << max_difference << " over " << num_values
                   << " values)\n";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        max_ulp <= 4U,
        "ConductionDirectSolver device-assembled values differ from the host assembly "
        "beyond FMA contraction (> 4 ULP)");
}

inline void
ConductionDirectSolver::Factorize ()
{
    BL_PROFILE("ConductionDirect::Factorize()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_pattern_built, "ConductionDirectSolver::Factorize() called before AssembleValues()");
#if defined(WARPX_USE_CUDSS)
    static_assert(std::is_same_v<amrex::Real, double>,
                  "The cuDSS conduction direct solver requires double-precision amrex::Real");
    if (!amrex::ParallelDescriptor::IOProcessor())
    {
        return;
    }
    const auto num_rows = static_cast<std::int64_t>(m_num_rows);
    const auto num_values = static_cast<std::int64_t>(m_matrix_values.size());
    if (!m_cudss_ready)
    {
        WARPX_CUDSS_CHECK_CONDUCTION(cudssCreate(&m_cudss_handle));
        WARPX_CUDSS_CHECK_CONDUCTION(cudssSetStream(m_cudss_handle, amrex::Gpu::gpuStream()));
        WARPX_CUDSS_CHECK_CONDUCTION(cudssConfigCreate(&m_cudss_config));
        WARPX_CUDSS_CHECK_CONDUCTION(cudssDataCreate(m_cudss_handle, &m_cudss_data));

        m_device_row_offsets.resize(m_row_offsets.size());
        m_device_column_indices.resize(m_column_indices.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_row_offsets.begin(),
                              m_row_offsets.end(), m_device_row_offsets.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_column_indices.begin(),
                              m_column_indices.end(), m_device_column_indices.begin());
        amrex::Real* matrix_values = nullptr;
        if (m_use_device_assembly)
        {
            matrix_values = m_values_device.data();
        }
        else
        {
            m_device_values.resize(m_matrix_values.size());
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_matrix_values.begin(),
                                  m_matrix_values.end(), m_device_values.begin());
            matrix_values = m_device_values.data();
        }
        amrex::Gpu::streamSynchronize();

        // The Dirichlet end rows (three-point ghost extrapolation) break the
        // symmetry of the interior five-point stencil, so the matrix is
        // declared general.
        WARPX_CUDSS_CHECK_CONDUCTION(cudssMatrixCreateCsr(
            &m_cudss_matrix, num_rows, num_rows, num_values, m_device_row_offsets.data(),
            nullptr, m_device_column_indices.data(), matrix_values, CUDSS_R_32I, CUDSS_R_32I,
            CUDSS_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        WARPX_CUDSS_CHECK_CONDUCTION(cudssMatrixCreateDn(&m_cudss_rhs, num_rows, 1, num_rows,
                                                         m_device_rhs.data(), CUDSS_R_64F,
                                                         CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK_CONDUCTION(cudssMatrixCreateDn(
            &m_cudss_solution, num_rows, 1, num_rows, m_device_solution.data(), CUDSS_R_64F,
            CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK_CONDUCTION(cudssExecute(m_cudss_handle, CUDSS_PHASE_ANALYSIS,
                                                  m_cudss_config, m_cudss_data, m_cudss_matrix,
                                                  m_cudss_solution, m_cudss_rhs));
        m_cudss_ready = true;
    }
    else if (m_use_device_assembly)
    {
        WARPX_CUDSS_CHECK_CONDUCTION(
            cudssMatrixSetValues(m_cudss_matrix, m_values_device.data()));
    }
    else
    {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_matrix_values.begin(),
                              m_matrix_values.end(), m_device_values.begin());
        amrex::Gpu::streamSynchronize();
        WARPX_CUDSS_CHECK_CONDUCTION(
            cudssMatrixSetValues(m_cudss_matrix, m_device_values.data()));
    }
    WARPX_CUDSS_CHECK_CONDUCTION(cudssExecute(m_cudss_handle, CUDSS_PHASE_FACTORIZATION,
                                              m_cudss_config, m_cudss_data, m_cudss_matrix,
                                              m_cudss_solution, m_cudss_rhs));
    amrex::Gpu::streamSynchronize();
#else
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block.conduction_solver = direct requires a CUDA build configured with "
        "WarpX_CUDSS=ON (NVIDIA cuDSS); mlmg is the portable default and banded the "
        "portable exact inverse");
#endif
}

inline void
ConductionDirectSolver::StageChannelToHost (const amrex::MultiFab& a_stacked,
                                            const int a_component)
{
    amrex::MultiFab::Copy(*m_channel_device, a_stacked, a_component, 0, 1, 0);
    amrex::dtoh_memcpy(*m_channel_host, *m_channel_device);
    amrex::Gpu::streamSynchronize();
}

inline void
ConductionDirectSolver::PushChannelToDevice (amrex::MultiFab& a_stacked, const int a_component)
{
    amrex::htod_memcpy(*m_channel_device, *m_channel_host);
    amrex::Gpu::streamSynchronize();
    amrex::MultiFab::Copy(a_stacked, *m_channel_device, 0, a_component, 1, 0);
}

inline void
ConductionDirectSolver::Solve (const amrex::MultiFab& a_rhs, amrex::MultiFab& a_solution)
{
    BL_PROFILE("ConductionDirect::Solve()");

    if (m_use_device_assembly)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            if (m_component[channel] < 0)
            {
                continue;
            }
            conduction_detail::GatherChannelRhs(m_view, channel, m_component[channel], a_rhs,
                                                m_device_rhs.data());
        }
    }
    else
    {
        std::fill(m_rhs.begin(), m_rhs.end(), amrex::Real(0.0));
        for (int channel = 0; channel < 3; ++channel)
        {
            if (m_component[channel] < 0)
            {
                continue;
            }
            StageChannelToHost(a_rhs, m_component[channel]);
            for (amrex::MFIter mfi(*m_channel_host); mfi.isValid(); ++mfi)
            {
                const auto values = m_channel_host->const_array(mfi);
                amrex::LoopOnCpu(mfi.validbox(),
                                 [&] (int i, int j, int k)
                                 {
                                     m_rhs[m_view.Index(channel, i, j, k)] = values(i, j, k);
                                 });
            }
        }
        if (amrex::ParallelDescriptor::NProcs() > 1)
        {
            amrex::ParallelDescriptor::ReduceRealSum(
                m_rhs.data(), static_cast<int>(m_num_rows),
                amrex::ParallelDescriptor::IOProcessorNumber());
        }
    }

#if defined(WARPX_USE_CUDSS)
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cudss_ready, "ConductionDirectSolver::Solve() called before Factorize()");
        if (!m_use_device_assembly)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_rhs.begin(), m_rhs.end(),
                                  m_device_rhs.begin());
        }
        WARPX_CUDSS_CHECK_CONDUCTION(cudssExecute(m_cudss_handle, CUDSS_PHASE_SOLVE,
                                                  m_cudss_config, m_cudss_data, m_cudss_matrix,
                                                  m_cudss_solution, m_cudss_rhs));
        if (!m_use_device_assembly)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, m_device_solution.begin(),
                                  m_device_solution.end(), m_solution.begin());
            amrex::Gpu::streamSynchronize();
        }
    }
#else
    amrex::ignore_unused(a_solution);
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block.conduction_solver = direct requires a CUDA build configured with "
        "WarpX_CUDSS=ON (NVIDIA cuDSS); mlmg is the portable default and banded the "
        "portable exact inverse");
#endif

    if (m_use_device_assembly)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            if (m_component[channel] < 0)
            {
                continue;
            }
            conduction_detail::ScatterChannelSolution(m_view, channel, m_component[channel],
                                                      m_device_solution.data(), a_solution);
        }
        amrex::Gpu::streamSynchronize();
        return;
    }

    if (amrex::ParallelDescriptor::NProcs() > 1)
    {
        amrex::ParallelDescriptor::Bcast(m_solution.data(),
                                         static_cast<std::size_t>(m_num_rows),
                                         amrex::ParallelDescriptor::IOProcessorNumber());
    }
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_component[channel] < 0)
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_channel_host); mfi.isValid(); ++mfi)
        {
            const auto values = m_channel_host->array(mfi);
            amrex::LoopOnCpu(mfi.validbox(),
                             [&] (int i, int j, int k)
                             {
                                 values(i, j, k) = m_solution[m_view.Index(channel, i, j, k)];
                             });
        }
        PushChannelToDevice(a_solution, m_component[channel]);
    }
    amrex::Gpu::streamSynchronize();
}

inline void
ConductionDirectSolver::FillValidationVector (amrex::MultiFab& a_stacked)
{
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_component[channel] < 0)
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_channel_host); mfi.isValid(); ++mfi)
        {
            const auto values = m_channel_host->array(mfi);
            amrex::LoopOnCpu(mfi.validbox(),
                             [&] (int i, int j, int k)
                             {
                                 values(i, j, k) =
                                     ValidationValue(m_view.Index(channel, i, j, k));
                             });
        }
        PushChannelToDevice(a_stacked, m_component[channel]);
    }
    amrex::Gpu::streamSynchronize();
}

inline std::pair<amrex::Real, amrex::Real>
ConductionDirectSolver::CompareMatrixFreeAction (const amrex::MultiFab& a_action)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_pattern_built,
        "ConductionDirectSolver::CompareMatrixFreeAction() called before AssembleValues()");

    amrex::Real reference_scale = 0.0;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        for (amrex::Long row = 0; row < m_num_rows; ++row)
        {
            amrex::Real sum = 0.0;
            for (int slot = m_row_offsets[row]; slot < m_row_offsets[row + 1]; ++slot)
            {
                sum += m_matrix_values[slot] * ValidationValue(m_column_indices[slot]);
            }
            m_solution[row] = sum;
            reference_scale = amrex::max(reference_scale, std::abs(sum));
        }
    }
    if (amrex::ParallelDescriptor::NProcs() > 1)
    {
        amrex::ParallelDescriptor::Bcast(m_solution.data(),
                                         static_cast<std::size_t>(m_num_rows),
                                         amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealMax(reference_scale);
    }

    amrex::Real max_difference = 0.0;
    for (int channel = 0; channel < 3; ++channel)
    {
        if (m_component[channel] < 0)
        {
            continue;
        }
        StageChannelToHost(a_action, m_component[channel]);
        for (amrex::MFIter mfi(*m_channel_host); mfi.isValid(); ++mfi)
        {
            const auto action = m_channel_host->const_array(mfi);
            amrex::LoopOnCpu(mfi.validbox(),
                             [&] (int i, int j, int k)
                             {
                                 const amrex::Long row = m_view.Index(channel, i, j, k);
                                 max_difference = amrex::max(
                                     max_difference, std::abs(action(i, j, k) - m_solution[row]));
                             });
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(max_difference);
    return {max_difference, reference_scale};
}

} // namespace warpx::mhd_pc

#endif // MHD_CONDUCTION_DIRECT_H_
