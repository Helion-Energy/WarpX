/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_RESISTIVE_DIRECT_H_
#define MHD_RESISTIVE_DIRECT_H_

#include "MHDResistiveStencil.H"
#include "Utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
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
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

namespace warpx::mhd_pc
{

/**
 * \brief Exact sparse inverse of the resistive block operator I + A,
 * A = theta_r dt curl((eta_field/mu0) curl .), for the MHD block
 * preconditioner (see MHDBlockPC, resistive block).
 *
 * The matrix rows are assembled from the SAME stencil emission the
 * matrix-free Chebyshev path applies (EmitResistiveRow), so the factored
 * matrix is the applied operator by construction, including the
 * PEC-wall/axis/open boundary row conventions. The unknowns are the
 * participating magnetic-field face components on the level, indexed
 * canonically: one unknown per physical face, with nodal points on
 * periodic boundaries identified with their low-side image and stored
 * duplicates (shared box-interface nodes) resolved through the AMReX
 * owner mask so every canonical row is assembled exactly once.
 *
 * The sparsity pattern is geometric and frozen at Define(); each
 * preconditioner update recomputes only the matrix values from the
 * refrozen eta_field and refactorizes. Applications are then single
 * forward/backward substitutions. Multi-rank runs assemble the global
 * matrix on the I/O rank (the level-wide system is small: ~2 10^5 rows
 * with <= 9 entries each at the largest production resolution), which
 * factorizes and solves for all ranks; the solution is broadcast and
 * scattered, which also keeps duplicated stored faces bitwise
 * consistent across ranks.
 *
 * The numeric backend is NVIDIA cuDSS (CUDA builds configured with
 * WarpX_CUDSS=ON). Assembly and the roundoff matvec validation against
 * the matrix-free operator are portable and available on every build.
 */
class ResistiveDirectSolver
{
    public:
        ResistiveDirectSolver () = default;
        ~ResistiveDirectSolver ();

        ResistiveDirectSolver (const ResistiveDirectSolver&) = delete;
        ResistiveDirectSolver& operator= (const ResistiveDirectSolver&) = delete;
        ResistiveDirectSolver (ResistiveDirectSolver&&) noexcept = delete;
        ResistiveDirectSolver& operator= (ResistiveDirectSolver&&) noexcept = delete;

        /** \brief Whether a direct factorization backend was compiled in. */
        [[nodiscard]] static constexpr bool
        HaveFactorizationBackend ()
        {
#if defined(WARPX_USE_CUDSS)
            return true;
#else
            return false;
#endif
        }

        /**
         * \brief Build the canonical unknown indexing, the ownership
         * masks, the host mirrors, and (lazily, at the first assembly)
         * the frozen sparsity pattern. a_faces holds the participating
         * magnetic-field face components (null for components outside
         * the resistive curl-curl); a_eta_edge the matching eta_field
         * staggerings.
         */
        void Define (const amrex::Geometry& a_geometry,
                     const amrex::Array<const amrex::MultiFab*, 3>& a_faces,
                     const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
                     const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge = {
                         nullptr, nullptr, nullptr},
                     const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge = {
                         nullptr, nullptr, nullptr},
                     const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge = {
                         nullptr, nullptr, nullptr});

        [[nodiscard]] bool IsDefined () const { return m_is_defined; }

        /**
         * \brief Recompute the matrix values from the frozen eta_field
         * with the shared stencil emission (the first call also freezes
         * the sparsity pattern) and gather them into the I/O rank's
         * global matrix.
         */
        void AssembleValues (const ResistiveStencilConstants& a_constants,
                             const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
                             const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge = {
                                 nullptr, nullptr, nullptr},
                             const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge = {
                                 nullptr, nullptr, nullptr},
                             const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge = {
                                 nullptr, nullptr, nullptr});

        /** \brief Numeric factorization of the assembled matrix (cuDSS). */
        void Factorize ();

        /** \brief In-place x := (I + A)^{-1} x on the face components. */
        void Solve (const amrex::Array<amrex::MultiFab*, 3>& a_faces);

        /**
         * \brief Load the deterministic assembly-validation vector into
         * the face components (every stored point reads its canonical
         * unknown, so duplicated faces are consistent by construction).
         */
        void FillValidationVector (const amrex::Array<amrex::MultiFab*, 3>& a_faces);

        /**
         * \brief Compare the matrix-free action A x on the validation
         * vector against the assembled-matrix product: returns the
         * maximum absolute entry difference of (x + A x) and the
         * maximum absolute entry of the assembled-matrix product as the
         * reference scale (both reduced over ranks).
         */
        [[nodiscard]] std::pair<amrex::Real, amrex::Real>
        CompareMatrixFreeAction (const amrex::Array<const amrex::MultiFab*, 3>& a_actions);

    private:
        struct ComponentMap
        {
            bool active = false;
            amrex::IntVect low;
            amrex::IntVect high; // canonical range, inclusive
            amrex::Long offset = 0;
        };

        /** \brief Deterministic pseudo-random unit value keyed by the
         * canonical unknown index (reproducible on every rank without
         * communication). */
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

        [[nodiscard]] amrex::IntVect
        WrapPoint (amrex::IntVect a_point) const
        {
            const amrex::Box domain = m_geometry.Domain();
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
            {
                if (m_geometry.isPeriodic(direction))
                {
                    const int extent = domain.length(direction);
                    int index = a_point[direction] - domain.smallEnd(direction);
                    index %= extent;
                    if (index < 0)
                    {
                        index += extent;
                    }
                    a_point[direction] = domain.smallEnd(direction) + index;
                }
            }
            return a_point;
        }

        [[nodiscard]] amrex::Long
        GlobalIndex (const int a_component, const amrex::IntVect& a_wrapped) const
        {
            const ComponentMap& map = m_component_map[a_component];
            amrex::Long index = 0;
            for (int direction = AMREX_SPACEDIM - 1; direction >= 0; --direction)
            {
                AMREX_ALWAYS_ASSERT(a_wrapped[direction] >= map.low[direction] &&
                                    a_wrapped[direction] <= map.high[direction]);
                index = index * (map.high[direction] - map.low[direction] + 1) +
                        (a_wrapped[direction] - map.low[direction]);
            }
            return map.offset + index;
        }

        /**
         * \brief Run the shared stencil emission over the locally owned
         * rows: the first call records the (row, column) pattern, every
         * call records the values in the same deterministic order.
         */
        void EmitLocalRows (const ResistiveStencilConstants& a_constants);

        /** \brief Freeze the global sparsity pattern on the I/O rank. */
        void BuildGlobalPattern ();

        /** \brief Gather the local values and fold them into the I/O
         * rank's global matrix. */
        void FoldGlobalValues ();

        bool m_is_defined = false;
        bool m_pattern_built = false;

        amrex::Geometry m_geometry;
        amrex::Array<ComponentMap, 3> m_component_map;
        amrex::Long m_num_rows = 0;

        // Host mirrors (pinned in GPU builds): ownership masks, frozen
        // eta_field staggerings, and face-component transfer buffers.
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> m_owner_host;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_eta_host;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_eta_h_host;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_hall_host;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_inertia_host;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_face_host;
        amrex::Array<const amrex::MultiFab*, 3> m_eta_source = {nullptr, nullptr, nullptr};
        amrex::Array<const amrex::MultiFab*, 3> m_eta_h_source = {nullptr, nullptr, nullptr};
        amrex::Array<const amrex::MultiFab*, 3> m_hall_source = {nullptr, nullptr, nullptr};
        amrex::Array<const amrex::MultiFab*, 3> m_inertia_source = {nullptr, nullptr, nullptr};

        // Locally owned rows in fixed emission order.
        amrex::Long m_local_owned_rows = 0;
        amrex::Vector<int> m_local_entry_rows;
        amrex::Vector<int> m_local_entry_columns;
        amrex::Vector<amrex::Real> m_local_entry_values;

        // I/O-rank global matrix: gathered emission triplets folded into
        // CSR through the frozen entry-to-slot map.
        amrex::Vector<int> m_gather_counts;
        amrex::Vector<int> m_gather_offsets;
        amrex::Vector<amrex::Real> m_gathered_values;
        amrex::Vector<int> m_entry_slot;
        amrex::Vector<int> m_row_offsets;
        amrex::Vector<int> m_column_indices;
        amrex::Vector<amrex::Real> m_matrix_values;

        // Host solve vectors (full canonical length on every rank; the
        // broadcast solution keeps duplicated stored faces consistent).
        amrex::Vector<amrex::Real> m_rhs;
        amrex::Vector<amrex::Real> m_solution;

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
        amrex::Gpu::DeviceVector<amrex::Real> m_device_rhs;
        amrex::Gpu::DeviceVector<amrex::Real> m_device_solution;
#endif
};

#if defined(WARPX_USE_CUDSS)
#define WARPX_CUDSS_CHECK(call)                                                \
    {                                                                          \
        const cudssStatus_t warpx_cudss_status = (call);                       \
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(                                      \
            warpx_cudss_status == CUDSS_STATUS_SUCCESS,                        \
            "cuDSS call failed in the MHD resistive direct solver: " #call);   \
    }
#endif

inline
ResistiveDirectSolver::~ResistiveDirectSolver ()
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
ResistiveDirectSolver::Define (const amrex::Geometry& a_geometry,
                               const amrex::Array<const amrex::MultiFab*, 3>& a_faces,
                               const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
                               const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
                               const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge,
                               const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge)
{
    BL_PROFILE("ResistiveDirect::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined, "ResistiveDirectSolver::Define() called on a defined object");
    m_geometry = a_geometry;
    const amrex::Box domain = m_geometry.Domain();

    amrex::Long offset = 0;
    for (int component = 0; component < 3; ++component)
    {
        if (a_faces[component] == nullptr)
        {
            continue;
        }
        ComponentMap& map = m_component_map[component];
        map.active = true;
        const amrex::IntVect stagger = a_faces[component]->ixType().toIntVect();
        amrex::Long points = 1;
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            map.low[direction] = domain.smallEnd(direction);
            // Nodal points on periodic boundaries are identified with
            // their low-side image; non-periodic nodal directions keep
            // the extra boundary plane.
            map.high[direction] =
                domain.bigEnd(direction) +
                ((stagger[direction] == 1 && !m_geometry.isPeriodic(direction)) ? 1 : 0);
            points *= map.high[direction] - map.low[direction] + 1;
        }
        map.offset = offset;
        offset += points;
    }
    m_num_rows = offset;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_num_rows > 0 &&
            m_num_rows < static_cast<amrex::Long>(std::numeric_limits<int>::max()),
        "ResistiveDirectSolver requires a nonempty level with fewer than 2^31 face "
        "unknowns");

    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_eta_edge[component] != nullptr,
            "ResistiveDirectSolver requires the eta_field staggering of every "
            "participating face component");
        m_eta_source[component] = a_eta_edge[component];

        const amrex::MultiFab& faces = *a_faces[component];
        const auto owner = amrex::OwnerMask(faces, m_geometry.periodicity());
        m_owner_host[component] = std::make_unique<amrex::iMultiFab>(
            faces.boxArray(), faces.DistributionMap(), 1, 0,
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        amrex::dtoh_memcpy(*m_owner_host[component], *owner);
        amrex::Gpu::streamSynchronize();

        m_face_host[component] = std::make_unique<amrex::MultiFab>(
            faces.boxArray(), faces.DistributionMap(), 1, 0,
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));

        // The device-to-host mirror copies require matching ghost layouts.
        const amrex::MultiFab& eta = *a_eta_edge[component];
        m_eta_host[component] = std::make_unique<amrex::MultiFab>(
            eta.boxArray(), eta.DistributionMap(), 1, eta.nGrowVect(),
            amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        if (a_eta_h_edge[component] != nullptr)
        {
            m_eta_h_source[component] = a_eta_h_edge[component];
            const amrex::MultiFab& eta_h = *a_eta_h_edge[component];
            m_eta_h_host[component] = std::make_unique<amrex::MultiFab>(
                eta_h.boxArray(), eta_h.DistributionMap(), 1, eta_h.nGrowVect(),
                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        }
        if (a_hall_edge[component] != nullptr)
        {
            // Frozen Hall coefficient vector Hb = B/rho_q (three
            // components) on this edge staggering.
            m_hall_source[component] = a_hall_edge[component];
            const amrex::MultiFab& hall = *a_hall_edge[component];
            m_hall_host[component] = std::make_unique<amrex::MultiFab>(
                hall.boxArray(), hall.DistributionMap(), hall.nComp(),
                hall.nGrowVect(),
                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        }
        if (a_inertia_edge[component] != nullptr)
        {
            // Frozen electron-inertia coefficient C_i = m_e_eff/(e rho_q)
            // on this edge staggering.
            m_inertia_source[component] = a_inertia_edge[component];
            const amrex::MultiFab& inertia = *a_inertia_edge[component];
            m_inertia_host[component] = std::make_unique<amrex::MultiFab>(
                inertia.boxArray(), inertia.DistributionMap(), 1,
                inertia.nGrowVect(),
                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
        }
    }

    m_rhs.resize(m_num_rows);
    m_solution.resize(m_num_rows);
    m_is_defined = true;
}

inline void
ResistiveDirectSolver::EmitLocalRows (const ResistiveStencilConstants& a_constants)
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
            m_local_entry_values[entry] = a_value;
        }
        ++entry;
    };

    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_owner_host[component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto owner = m_owner_host[component]->const_array(mfi);
            ResistiveStencilArrays arrays;
            for (int eta_component = 0; eta_component < 3; ++eta_component)
            {
                if (m_eta_host[eta_component])
                {
                    arrays.eta[eta_component] =
                        m_eta_host[eta_component]->const_array(mfi);
                }
                if (m_eta_h_host[eta_component])
                {
                    arrays.eta_h[eta_component] =
                        m_eta_h_host[eta_component]->const_array(mfi);
                }
                if (m_hall_host[eta_component])
                {
                    arrays.hall[eta_component] =
                        m_hall_host[eta_component]->const_array(mfi);
                }
                if (m_inertia_host[eta_component])
                {
                    arrays.inertia[eta_component] =
                        m_inertia_host[eta_component]->const_array(mfi);
                }
            }
            amrex::LoopOnCpu(
                box,
                [&] (int i, int j, int k)
                {
                    if (!owner(i, j, k))
                    {
                        return;
                    }
                    const amrex::IntVect point(AMREX_D_DECL(i, j, k));
                    const amrex::Long row = GlobalIndex(component, WrapPoint(point));
                    const auto row_index = static_cast<int>(row);
                    ++owned_rows;
                    // The identity of I + A; the stencil emits only A.
                    record(row_index, row_index, amrex::Real(1.0));
                    EmitResistiveRow(
                        a_constants, arrays, component, i, j, k,
                        [&] (const int a_column_component, const int ic, const int jc,
                             const int kc, const amrex::Real a_value)
                        {
                            amrex::ignore_unused(jc, kc);
                            const amrex::IntVect column_point(AMREX_D_DECL(ic, jc, kc));
                            const amrex::Long column = GlobalIndex(
                                a_column_component, WrapPoint(column_point));
                            record(row_index, static_cast<int>(column), a_value);
                        });
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
            "ResistiveDirectSolver stencil emission changed structure between updates");
    }
}

inline void
ResistiveDirectSolver::BuildGlobalPattern ()
{
    // Every canonical row must be owned exactly once across ranks.
    amrex::Long total_owned_rows = m_local_owned_rows;
    amrex::ParallelDescriptor::ReduceLongSum(total_owned_rows);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        total_owned_rows == m_num_rows,
        "ResistiveDirectSolver ownership does not cover the canonical unknowns "
        "exactly once");

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
        MPI_Gather(&local_count, 1, MPI_INT, m_gather_counts.data(), 1, MPI_INT,
                   io_rank, amrex::ParallelDescriptor::Communicator());
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
                "ResistiveDirectSolver matrix exceeds 2^31 assembled entries");
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
                      return rows[a] != rows[b] ? rows[a] < rows[b]
                                                : columns[a] < columns[b];
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
                "ResistiveDirectSolver assembled an out-of-range matrix entry");
            if (row != previous_row || column != previous_column)
            {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    row >= previous_row,
                    "ResistiveDirectSolver entry ordering is inconsistent");
                m_column_indices.push_back(column);
                ++m_row_offsets[row + 1];
                previous_row = row;
                previous_column = column;
            }
            m_entry_slot[entry] = static_cast<int>(m_column_indices.size()) - 1;
        }
        for (amrex::Long row = 0; row < m_num_rows; ++row)
        {
            // Every canonical row carries at least its identity diagonal.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_row_offsets[row + 1] > 0,
                "ResistiveDirectSolver left a canonical row unassembled");
            m_row_offsets[row + 1] += m_row_offsets[row];
        }
        m_matrix_values.assign(m_column_indices.size(), amrex::Real(0.0));
        m_gathered_values.resize(num_entries);
    }
    m_pattern_built = true;
}

inline void
ResistiveDirectSolver::FoldGlobalValues ()
{
    const int io_rank = amrex::ParallelDescriptor::IOProcessorNumber();
    const int num_ranks = amrex::ParallelDescriptor::NProcs();
    const auto local_count = static_cast<int>(m_local_entry_values.size());
    amrex::ignore_unused(io_rank, local_count); // only the MPI gather reads them

    const amrex::Real* gathered_values = m_local_entry_values.data();
    if (num_ranks > 1)
    {
#if defined(AMREX_USE_MPI)
        MPI_Gatherv(m_local_entry_values.data(), local_count,
                    amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                    m_gathered_values.data(), m_gather_counts.data(),
                    m_gather_offsets.data(),
                    amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type(),
                    io_rank, amrex::ParallelDescriptor::Communicator());
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
ResistiveDirectSolver::AssembleValues (
    const ResistiveStencilConstants& a_constants,
    const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
    const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
    const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge,
    const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge)
{
    BL_PROFILE("ResistiveDirect::AssembleValues()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "ResistiveDirectSolver::AssembleValues() called before Define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !a_constants.include_hyper ||
            (m_eta_h_host[0] != nullptr || m_eta_h_host[1] != nullptr ||
             m_eta_h_host[2] != nullptr),
        "ResistiveDirectSolver hyper-resistive assembly requires the eta_H "
        "staggerings at Define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !a_constants.include_hall ||
            (m_hall_host[0] != nullptr || m_hall_host[1] != nullptr ||
             m_hall_host[2] != nullptr),
        "ResistiveDirectSolver Hall assembly requires the Hall-coefficient "
        "staggerings at Define()");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !a_constants.include_inertia ||
            (m_inertia_host[0] != nullptr || m_inertia_host[1] != nullptr ||
             m_inertia_host[2] != nullptr),
        "ResistiveDirectSolver electron-inertia assembly requires the "
        "inertia-coefficient staggerings at Define()");
    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_eta_edge[component] == m_eta_source[component],
            "ResistiveDirectSolver eta_field staggerings changed after Define()");
        amrex::dtoh_memcpy(*m_eta_host[component], *a_eta_edge[component]);
        if (m_eta_h_host[component])
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_eta_h_edge[component] == m_eta_h_source[component],
                "ResistiveDirectSolver eta_H staggerings changed after Define()");
            amrex::dtoh_memcpy(*m_eta_h_host[component], *a_eta_h_edge[component]);
        }
        if (m_hall_host[component])
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_hall_edge[component] == m_hall_source[component],
                "ResistiveDirectSolver Hall staggerings changed after Define()");
            amrex::dtoh_memcpy(*m_hall_host[component], *a_hall_edge[component]);
        }
        if (m_inertia_host[component])
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_inertia_edge[component] == m_inertia_source[component],
                "ResistiveDirectSolver inertia staggerings changed after Define()");
            amrex::dtoh_memcpy(*m_inertia_host[component],
                               *a_inertia_edge[component]);
        }
    }
    amrex::Gpu::streamSynchronize();

    const bool first_assembly = !m_pattern_built;
    EmitLocalRows(a_constants);
    if (first_assembly)
    {
        BuildGlobalPattern();
    }
    FoldGlobalValues();
}

inline void
ResistiveDirectSolver::Factorize ()
{
    BL_PROFILE("ResistiveDirect::Factorize()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_pattern_built,
        "ResistiveDirectSolver::Factorize() called before AssembleValues()");
#if defined(WARPX_USE_CUDSS)
    static_assert(std::is_same_v<amrex::Real, double>,
                  "The cuDSS resistive direct solver requires double-precision "
                  "amrex::Real");
    if (!amrex::ParallelDescriptor::IOProcessor())
    {
        return;
    }
    const auto num_rows = static_cast<std::int64_t>(m_num_rows);
    const auto num_values = static_cast<std::int64_t>(m_matrix_values.size());
    if (!m_cudss_ready)
    {
        WARPX_CUDSS_CHECK(cudssCreate(&m_cudss_handle));
        WARPX_CUDSS_CHECK(cudssSetStream(m_cudss_handle, amrex::Gpu::gpuStream()));
        WARPX_CUDSS_CHECK(cudssConfigCreate(&m_cudss_config));
        WARPX_CUDSS_CHECK(cudssDataCreate(m_cudss_handle, &m_cudss_data));

        m_device_row_offsets.resize(m_row_offsets.size());
        m_device_column_indices.resize(m_column_indices.size());
        m_device_values.resize(m_matrix_values.size());
        m_device_rhs.resize(m_num_rows);
        m_device_solution.resize(m_num_rows);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_row_offsets.begin(),
                              m_row_offsets.end(), m_device_row_offsets.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_column_indices.begin(),
                              m_column_indices.end(), m_device_column_indices.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_matrix_values.begin(),
                              m_matrix_values.end(), m_device_values.begin());
        amrex::Gpu::streamSynchronize();

        WARPX_CUDSS_CHECK(cudssMatrixCreateCsr(
            &m_cudss_matrix, num_rows, num_rows, num_values,
            m_device_row_offsets.data(), nullptr, m_device_column_indices.data(),
            m_device_values.data(), CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_rhs, num_rows, 1, num_rows,
                                              m_device_rhs.data(), CUDSS_R_64F,
                                              CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_solution, num_rows, 1, num_rows,
                                              m_device_solution.data(), CUDSS_R_64F,
                                              CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_ANALYSIS,
                                       m_cudss_config, m_cudss_data, m_cudss_matrix,
                                       m_cudss_solution, m_cudss_rhs));
        m_cudss_ready = true;
    }
    else
    {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_matrix_values.begin(),
                              m_matrix_values.end(), m_device_values.begin());
        amrex::Gpu::streamSynchronize();
        WARPX_CUDSS_CHECK(cudssMatrixSetValues(m_cudss_matrix, m_device_values.data()));
    }
    WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_FACTORIZATION,
                                   m_cudss_config, m_cudss_data, m_cudss_matrix,
                                   m_cudss_solution, m_cudss_rhs));
    amrex::Gpu::streamSynchronize();
#else
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block.resistive_solver = direct requires a CUDA build configured "
        "with WarpX_CUDSS=ON (NVIDIA cuDSS); chebyshev is the portable default");
#endif
}

inline void
ResistiveDirectSolver::Solve (const amrex::Array<amrex::MultiFab*, 3>& a_faces)
{
    BL_PROFILE("ResistiveDirect::Solve()");

    // Gather the owned stored faces into the canonical right-hand side.
    std::fill(m_rhs.begin(), m_rhs.end(), amrex::Real(0.0));
    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        amrex::dtoh_memcpy(*m_face_host[component], *a_faces[component]);
        amrex::Gpu::streamSynchronize();
        for (amrex::MFIter mfi(*m_face_host[component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto owner = m_owner_host[component]->const_array(mfi);
            const auto values = m_face_host[component]->const_array(mfi);
            amrex::LoopOnCpu(box,
                             [&] (int i, int j, int k)
                             {
                                 if (!owner(i, j, k))
                                 {
                                     return;
                                 }
                                 const amrex::IntVect point(AMREX_D_DECL(i, j, k));
                                 m_rhs[GlobalIndex(component, WrapPoint(point))] =
                                     values(i, j, k);
                             });
        }
    }
    if (amrex::ParallelDescriptor::NProcs() > 1)
    {
        // The owned faces partition the unknowns, so the sum assembles
        // the global right-hand side on the factorizing rank.
        amrex::ParallelDescriptor::ReduceRealSum(
            m_rhs.data(), static_cast<int>(m_num_rows),
            amrex::ParallelDescriptor::IOProcessorNumber());
    }

#if defined(WARPX_USE_CUDSS)
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cudss_ready, "ResistiveDirectSolver::Solve() called before Factorize()");
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_rhs.begin(), m_rhs.end(),
                              m_device_rhs.begin());
        WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_SOLVE,
                                       m_cudss_config, m_cudss_data, m_cudss_matrix,
                                       m_cudss_solution, m_cudss_rhs));
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, m_device_solution.begin(),
                              m_device_solution.end(), m_solution.begin());
        amrex::Gpu::streamSynchronize();
    }
#else
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block.resistive_solver = direct requires a CUDA build configured "
        "with WarpX_CUDSS=ON (NVIDIA cuDSS); chebyshev is the portable default");
#endif

    if (amrex::ParallelDescriptor::NProcs() > 1)
    {
        amrex::ParallelDescriptor::Bcast(
            m_solution.data(), static_cast<std::size_t>(m_num_rows),
            amrex::ParallelDescriptor::IOProcessorNumber());
    }

    // Scatter to every stored face (duplicated faces read the same
    // canonical unknown, so they stay bitwise consistent).
    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_face_host[component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto values = m_face_host[component]->array(mfi);
            amrex::LoopOnCpu(box,
                             [&] (int i, int j, int k)
                             {
                                 const amrex::IntVect point(AMREX_D_DECL(i, j, k));
                                 values(i, j, k) =
                                     m_solution[GlobalIndex(component, WrapPoint(point))];
                             });
        }
        amrex::htod_memcpy(*a_faces[component], *m_face_host[component]);
    }
    amrex::Gpu::streamSynchronize();
}

inline void
ResistiveDirectSolver::FillValidationVector (
    const amrex::Array<amrex::MultiFab*, 3>& a_faces)
{
    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_face_host[component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto values = m_face_host[component]->array(mfi);
            amrex::LoopOnCpu(
                box,
                [&] (int i, int j, int k)
                {
                    const amrex::IntVect point(AMREX_D_DECL(i, j, k));
                    values(i, j, k) =
                        ValidationValue(GlobalIndex(component, WrapPoint(point)));
                });
        }
        amrex::htod_memcpy(*a_faces[component], *m_face_host[component]);
    }
    amrex::Gpu::streamSynchronize();
}

inline std::pair<amrex::Real, amrex::Real>
ResistiveDirectSolver::CompareMatrixFreeAction (
    const amrex::Array<const amrex::MultiFab*, 3>& a_actions)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_pattern_built,
        "ResistiveDirectSolver::CompareMatrixFreeAction() called before "
        "AssembleValues()");

    // Assembled-matrix product on the validation vector (I/O rank holds
    // the matrix; the broadcast makes the reference available on every
    // rank's stored points, duplicated faces included).
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
        amrex::ParallelDescriptor::Bcast(
            m_solution.data(), static_cast<std::size_t>(m_num_rows),
            amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::ParallelDescriptor::ReduceRealMax(reference_scale);
    }

    // Matrix-free comparison at every stored face: x + (A x) against the
    // assembled product of I + A.
    amrex::Real max_difference = 0.0;
    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        amrex::dtoh_memcpy(*m_face_host[component], *a_actions[component]);
        amrex::Gpu::streamSynchronize();
        for (amrex::MFIter mfi(*m_face_host[component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto action = m_face_host[component]->const_array(mfi);
            amrex::LoopOnCpu(
                box,
                [&] (int i, int j, int k)
                {
                    const amrex::IntVect point(AMREX_D_DECL(i, j, k));
                    const amrex::Long row = GlobalIndex(component, WrapPoint(point));
                    const amrex::Real matrix_free_value =
                        ValidationValue(row) + action(i, j, k);
                    max_difference = amrex::max(
                        max_difference, std::abs(matrix_free_value - m_solution[row]));
                });
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(max_difference);
    return {max_difference, reference_scale};
}

} // namespace warpx::mhd_pc

#endif // MHD_RESISTIVE_DIRECT_H_
