/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_RESISTIVE_DIRECT_H_
#define MHD_RESISTIVE_DIRECT_H_

#include "FieldSolver/ImplicitSolvers/FusedResidualOps.H"
#include "MHDResistiveStencil.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
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
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
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
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace warpx::mhd_pc
{

/**
 * \brief Device-copyable view of the direct solver's canonical unknown
 * indexing: the exact integer arithmetic of the host-side
 * WrapPoint/GlobalIndex pair (periodic directions wrap to the low-side
 * image, row-major over the canonical component ranges), usable inside
 * kernels.
 */
struct DirectIndexView
{
    amrex::GpuArray<amrex::Long, 3> component_offset = {0, 0, 0};
    amrex::GpuArray<amrex::GpuArray<int, AMREX_SPACEDIM>, 3> component_low = {};
    amrex::GpuArray<amrex::GpuArray<int, AMREX_SPACEDIM>, 3> component_high = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_low = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_length = {};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_periodic = {};

    /** \brief Canonical unknown index of a stored face point. */
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Long
    Index (const int a_component, const int i, const int j, const int k) const
    {
        const int point[3] = {i, j, k};
        amrex::Long index = 0;
        for (int direction = AMREX_SPACEDIM - 1; direction >= 0; --direction)
        {
            int wrapped = point[direction];
            if (domain_periodic[direction])
            {
                int offset =
                    (wrapped - domain_low[direction]) % domain_length[direction];
                if (offset < 0)
                {
                    offset += domain_length[direction];
                }
                wrapped = domain_low[direction] + offset;
            }
            index = index * (component_high[a_component][direction] -
                             component_low[a_component][direction] + 1) +
                    (wrapped - component_low[a_component][direction]);
        }
        return component_offset[a_component] + index;
    }
};

namespace direct_detail
{

/**
 * \brief Recompute the owned rows of one face component directly on the
 * device: re-run the shared stencil emission on the coefficient
 * MultiFabs' device arrays and accumulate the values into the frozen CSR
 * slots through the entry-to-slot map, in the exact per-row order of the
 * host pattern build (the identity diagonal first, then the emission).
 * Each canonical row is owned by exactly one stored point, so its slots
 * are touched by a single thread and the accumulation needs no atomics.
 * Emissions beyond the frozen per-row count are dropped and counted in
 * a_structure_errors (a structure change is an assembly bug; the caller
 * asserts the count is zero).
 */
inline void
RefreshComponentValues (const ResistiveStencilConstants& a_constants,
                        const int a_component,
                        const amrex::iMultiFab& a_entry_base,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge,
                        const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge,
                        const int* const a_entry_slot, amrex::Real* const a_values,
                        int* const a_structure_errors)
{
    for (amrex::MFIter mfi(a_entry_base); mfi.isValid(); ++mfi)
    {
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
        const auto entry_base = a_entry_base.const_array(mfi);
        const ResistiveStencilConstants constants = a_constants;
        const int row_component = a_component;
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const int base = entry_base(i, j, k, 0);
                if (base < 0)
                {
                    return; // a stored duplicate, not the owned image
                }
                const int count = entry_base(i, j, k, 1);
                int emitted = 0;
                const auto record = [&] (const amrex::Real a_value)
                {
                    if (emitted < count)
                    {
                        a_values[a_entry_slot[base + emitted]] += a_value;
                    }
                    ++emitted;
                };
                // The identity of I + A; the stencil emits only A.
                record(amrex::Real(1.0));
                EmitResistiveRow(
                    constants, arrays, row_component, i, j, k,
                    [&] (const int a_column_component, const int ic, const int jc,
                         const int kc, const amrex::Real a_value)
                    {
                        // The columns are frozen in the entry-to-slot map.
                        amrex::ignore_unused(a_column_component, ic, jc, kc);
                        record(a_value);
                    });
                if (emitted != count)
                {
                    amrex::Gpu::Atomic::AddNoRet(a_structure_errors, 1);
                }
            });
    }
}

/** \brief Gather the owned stored faces of one component into the
 * canonical device right-hand side (single-rank: the owned faces tile
 * the unknowns, so every element is written exactly once). */
inline void
GatherComponentRhs (const DirectIndexView& a_view, const int a_component,
                    const amrex::iMultiFab& a_entry_base,
                    const amrex::MultiFab& a_faces, amrex::Real* const a_rhs)
{
    for (amrex::MFIter mfi(a_entry_base); mfi.isValid(); ++mfi)
    {
        const auto entry_base = a_entry_base.const_array(mfi);
        const auto faces = a_faces.const_array(mfi);
        const DirectIndexView view = a_view;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               if (entry_base(i, j, k, 0) < 0)
                               {
                                   return;
                               }
                               a_rhs[view.Index(component, i, j, k)] =
                                   faces(i, j, k);
                           });
    }
}

/** \brief Scatter the canonical device solution to every stored face of
 * one component (duplicated faces read the same canonical unknown, so
 * they stay bitwise consistent). */
inline void
ScatterComponentSolution (const DirectIndexView& a_view, const int a_component,
                          const amrex::Real* const a_solution,
                          amrex::MultiFab& a_faces)
{
    for (amrex::MFIter mfi(a_faces); mfi.isValid(); ++mfi)
    {
        const auto faces = a_faces.array(mfi);
        const DirectIndexView view = a_view;
        const int component = a_component;
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               faces(i, j, k) =
                                   a_solution[view.Index(component, i, j, k)];
                           });
    }
}

/** \brief Gather the reduced sub-system's values from the global CSR
 * values through the frozen slot map (reduced solve). */
inline void
GatherReducedValues (const amrex::Real* const a_values, const int* const a_slots,
                     amrex::Real* const a_reduced, const amrex::Long a_count)
{
    amrex::ParallelFor(a_count,
                       [=] AMREX_GPU_DEVICE (amrex::Long entry)
                       {
                           a_reduced[entry] = a_values[a_slots[entry]];
                       });
}

/** \brief Maximum over the dropped rows' couplings of |a_ij| / |a_ii|
 * (slots whose row or column is a dropped row; reduced solve). */
inline amrex::Real
DroppedCouplingMax (const amrex::Real* const a_values, const int* const a_slot_row,
                    const int* const a_slot_column, const int* const a_diag_slot,
                    const int* const a_keep, const amrex::Long a_num_slots)
{
    amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    reduce_op.eval(a_num_slots, reduce_data,
                   [=] AMREX_GPU_DEVICE (amrex::Long slot) -> ReduceTuple
                   {
                       const int row = a_slot_row[slot];
                       const int column = a_slot_column[slot];
                       amrex::Real ratio = 0.0;
                       if (row != column)
                       {
                           const amrex::Real magnitude = std::abs(a_values[slot]);
                           if (a_keep[row] == 0)
                           {
                               ratio = amrex::max(
                                   ratio, magnitude / std::abs(a_values[a_diag_slot[row]]));
                           }
                           if (a_keep[column] == 0)
                           {
                               ratio = amrex::max(
                                   ratio,
                                   magnitude / std::abs(a_values[a_diag_slot[column]]));
                           }
                       }
                       return {ratio};
                   });
    return amrex::get<0>(reduce_data.value(reduce_op));
}

/** \brief FP64 -> FP32 device conversion (single-precision solve). */
inline void
ConvertToSingle (const amrex::Real* const a_source, float* const a_target,
                 const amrex::Long a_count)
{
    amrex::ParallelFor(a_count,
                       [=] AMREX_GPU_DEVICE (amrex::Long index)
                       {
                           a_target[index] = static_cast<float>(a_source[index]);
                       });
}

/** \brief FP32 -> FP64 device conversion (single-precision solve). */
inline void
ConvertToDouble (const float* const a_source, amrex::Real* const a_target,
                 const amrex::Long a_count)
{
    amrex::ParallelFor(a_count,
                       [=] AMREX_GPU_DEVICE (amrex::Long index)
                       {
                           a_target[index] = static_cast<amrex::Real>(a_source[index]);
                       });
}

/** \brief Zero a device vector of matrix values. */
inline void
ZeroRealVector (amrex::Real* const a_data, const std::size_t a_size)
{
    amrex::ParallelFor(static_cast<amrex::Long>(a_size),
                       [=] AMREX_GPU_DEVICE (amrex::Long flat)
                       {
                           a_data[flat] = amrex::Real(0.0);
                       });
}

/** \brief Zero a single device counter. */
inline void
ZeroIntScalar (int* const a_data)
{
    amrex::single_task([=] AMREX_GPU_DEVICE ()
                       {
                           *a_data = 0;
                       });
}

} // namespace direct_detail

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
 * Single-rank runs (the production configuration) refresh the values
 * DEVICE-RESIDENT after the one-time host pattern freeze: the shared
 * stencil emission is re-run in kernels on the coefficient MultiFabs'
 * device arrays and scattered through a device copy of the frozen
 * entry-to-slot map, and each application packs the right-hand side and
 * unpacks the solution on the device -- no per-refreeze coefficient
 * mirrors and no per-apply host round trip
 * (implicit_mhd.resistive_direct_device_assembly = auto/on/off, default
 * auto = on for single-rank runs). The host gather path remains the
 * multi-rank fallback, and pc_mhd_block.resistive_validate_assembly runs
 * both paths and asserts the two CSR value arrays agree bitwise (<= 4
 * ULP; exact equality expected and observed on CPU builds, the allowance
 * absorbs device FMA contraction only).
 *
 * The numeric backend is NVIDIA cuDSS (CUDA builds configured with
 * WarpX_CUDSS=ON). Assembly and the roundoff matvec validation against
 * the matrix-free operator are portable and available on every build.
 *
 * Measurement and tuning knobs (implicit_mhd.resistive_direct_*, every
 * default = the behaviour above):
 *  - cudss_options: "name=value" pairs forwarded to cudssConfigSet
 *    before the analysis (reordering_alg, factorization_alg, solve_alg,
 *    matching_alg, pivot_type, ir_n_steps, nd_nlevels, nd_ubfactor,
 *    use_superpanels, deterministic_mode, host_nthreads);
 *  - solve_host_sync: the host synchronization that closes every
 *    application (0 lets the host run ahead; the results are stream
 *    ordered either way);
 *  - dump_prefix / dump_assembly: write the assembled CSR matrix of the
 *    given assembly (1-based count) and the right-hand side and
 *    solution of the first application after it, as binary files for an
 *    offline solver benchmark (I/O rank; see DumpMatrix for the format).
 *  - row_threshold (>= 0 engages; default -1 = off): the REDUCED solve.
 *    The cuDSS triangular solves cost time in proportion to the number
 *    of rows they visit (measured ~5 ns per row on the production
 *    matrix, independent of how trivial the row is), and a large part of
 *    the level's face unknowns are identity rows (frozen exterior
 *    faces, axis and boundary faces) or nearly so (hot plasma, where the
 *    resistive diffusion number is 1e-4). A row whose off-diagonal
 *    couplings, in its row AND in its column, are all <= threshold *
 *    diagonal is solved as x_i = b_i / a_ii; the remaining rows form the
 *    factorized sub-system. threshold = 0 drops exactly the rows without
 *    any coupling (exact inverse, roundoff-different factorization);
 *    threshold > 0 is an approximate inverse with a relative truncation
 *    error bounded by the threshold. The row set is built from the
 *    frozen values with threshold / row_threshold_margin (default 10) and
 *    rebuilt -- a new pattern analysis -- only when a dropped row's
 *    coupling exceeds the threshold at a later refreeze (checked on the
 *    device at every refreeze), so the bound holds at every application.
 *    Single-rank device assembly only.
 *  - precision (double|single, default double): factorize and solve in
 *    FP32 (values, right-hand side and solution converted on the device
 *    around the cuDSS calls; the assembled FP64 values stay the reference
 *    for the assembly checks). The block inverse is then accurate to
 *    ~3e-5 relative -- ample for a preconditioner -- and the solve is
 *    measured ~8 % cheaper on the production matrix. Single-rank device
 *    assembly only.
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
         * \brief Run the host assembly alongside the device value
         * refresh at every update so the two CSR value arrays can be
         * compared bitwise (pc_mhd_block.resistive_validate_assembly).
         */
        void SetValidationMode (const bool a_validate)
        {
            m_validate_parity = a_validate;
        }

        /** \brief Whether the values are refreshed device-resident
         * (single-rank; implicit_mhd.resistive_direct_device_assembly).
         */
        [[nodiscard]] bool UsesDeviceAssembly () const
        {
            return m_use_device_assembly;
        }

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

        /** \brief One-time htod of the frozen entry-to-slot map and the
         * per-row emission ranges; allocates the device CSR values
         * buffer the refreshes update in place (single-rank device
         * assembly only). */
        void UploadFrozenMaps ();

        /** \brief Device value refresh through the frozen maps (see
         * direct_detail::RefreshComponentValues); asserts the emission
         * structure is unchanged. */
        void RefreshValuesOnDevice (
            const ResistiveStencilConstants& a_constants,
            const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
            const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
            const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge,
            const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge);

        /** \brief Bitwise gate of the device-refreshed CSR values
         * against the host emission+fold on the same frozen state
         * (validation mode only; <= 4 ULP asserted, see the class
         * comment). */
        void CheckAssemblyParity () const;

        /** \brief Monotonic unsigned image of the FP64 total order (the
         * standard sign-magnitude flip), for ULP distances. */
        [[nodiscard]] static std::uint64_t
        OrderedBits (const amrex::Real a_value)
        {
            static_assert(sizeof(amrex::Real) == sizeof(std::uint64_t),
                          "the assembly parity gate expects FP64 amrex::Real");
            std::uint64_t bits = 0;
            std::memcpy(&bits, &a_value, sizeof(bits));
            return (bits >> 63U) != 0U ? ~bits
                                       : (bits | 0x8000000000000000ULL);
        }

        /** \brief Write the assembled CSR matrix (I/O rank) for the
         * offline benchmark; binary little-endian: magic "WXCSR001",
         * int64 rows, int64 nonzeros, int32 space dimension, int32 3
         * (components), per component {int32 active, int64 offset,
         * int32 low[3], int32 high[3]}, int32 row_offsets[rows + 1],
         * int32 columns[nonzeros], float64 values[nonzeros]; a text
         * twin "<prefix>_meta.txt" repeats the header. */
        void DumpMatrix () const;

        /** \brief Write one canonical-length vector (magic "WXVEC001",
         * int64 length, float64 values). */
        static void DumpVector (const std::string& a_path,
                                const amrex::Real* a_values, amrex::Long a_size);

#if defined(WARPX_USE_CUDSS)
        /** \brief Forward implicit_mhd.resistive_direct_cudss_options to
         * cudssConfigSet (unknown names or values abort). */
        void ApplyCudssOptions () const;

        /** \brief Factorization of the reduced sub-system (see the class
         * comment): (re)creates the cuDSS objects and runs the analysis
         * after a row-set (re)build, then gathers the sub-system values
         * and refactorizes. */
        void FactorizeReduced ();
#endif

        /** \brief Current CSR values on the host (the device copy on the
         * device-assembly path, the folded host values otherwise). */
        [[nodiscard]] amrex::Vector<amrex::Real> HostValues () const;

        /** \brief Build the reduced row set and its sub-system pattern from
         * the given values (portable, host): rows kept iff some
         * off-diagonal coupling in the row or the column exceeds
         * (threshold / margin) times the diagonal. Uploads the device
         * copies. */
        void BuildReducedSystem (const amrex::Vector<amrex::Real>& a_values);

        /** \brief Device reduction over the dropped rows' couplings: the
         * maximum of |a_ij| / |a_ii| over slots whose row or column is a
         * dropped row (the truncation the reduced solve makes). */
        [[nodiscard]] amrex::Real DroppedCouplingMax () const;

        /** \brief Write the reduced sub-system next to the full dump. */
        void DumpReducedMatrix (const amrex::Vector<amrex::Real>& a_values) const;

        bool m_is_defined = false;
        bool m_pattern_built = false;

        // Measurement / tuning knobs (implicit_mhd.resistive_direct_*, see
        // the class comment).
        std::vector<std::string> m_cudss_options;
        bool m_host_sync_after_solve = true;
        std::string m_dump_prefix;
        int m_dump_assembly = 1;
        int m_assembly_count = 0;
        bool m_dump_solve_pending = false;

        // Reduced solve (implicit_mhd.resistive_direct_row_threshold, see
        // the class comment). Host products of BuildReducedSystem and
        // their device copies; the global-slot helpers (row of every
        // slot, diagonal slot of every row) are built once at the freeze.
        // Single-precision factorization/solve (implicit_mhd.resistive_direct_precision):
        // FP32 twins of the values, right-hand side and solution buffers that
        // the cuDSS objects wrap; conversions run on the device around the calls.
        bool m_single_precision = false;
        amrex::Gpu::DeviceVector<float> m_values_single;
        amrex::Gpu::DeviceVector<float> m_rhs_single;
        amrex::Gpu::DeviceVector<float> m_solution_single;

        amrex::Real m_row_threshold = -1.0;
        amrex::Real m_row_threshold_margin = 10.0;
        bool m_reduced_active = false;
        bool m_reduced_pattern_new = false;
        int m_reduced_rebuilds = 0;
        amrex::Long m_reduced_rows = 0;
        amrex::Vector<int> m_reduced_row_list;
        amrex::Vector<int> m_reduced_row_offsets;
        amrex::Vector<int> m_reduced_columns;
        amrex::Vector<int> m_reduced_slots;
        amrex::Gpu::DeviceVector<int> m_slot_row_device;
        amrex::Gpu::DeviceVector<int> m_slot_column_device;
        amrex::Gpu::DeviceVector<int> m_diag_slot_device;
        amrex::Gpu::DeviceVector<int> m_keep_device;
        amrex::Gpu::DeviceVector<int> m_reduced_row_list_device;
        amrex::Gpu::DeviceVector<int> m_reduced_row_offsets_device;
        amrex::Gpu::DeviceVector<int> m_reduced_columns_device;
        amrex::Gpu::DeviceVector<int> m_reduced_slots_device;
        amrex::Gpu::DeviceVector<amrex::Real> m_reduced_values_device;
        amrex::Gpu::DeviceVector<amrex::Real> m_reduced_rhs_device;
        amrex::Gpu::DeviceVector<amrex::Real> m_reduced_solution_device;

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

        // Single-rank device assembly (see the class comment): the frozen
        // per-row emission ranges (component 0 the first local entry of
        // the row, component 1 the entry count, -1 marking stored
        // duplicates), the device entry-to-slot map, the device CSR
        // values the refreshes update in place, and the structure-change
        // counter the refresh kernels flag into.
        bool m_use_device_assembly = false;
        bool m_validate_parity = false;
        DirectIndexView m_index_view;
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> m_entry_base_host;
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> m_entry_base_device;
        amrex::Gpu::DeviceVector<int> m_entry_slot_device;
        amrex::Gpu::DeviceVector<amrex::Real> m_values_device;
        amrex::Gpu::DeviceVector<int> m_structure_errors_device;

        // Device solve vectors (device-filled in the single-rank device
        // path, staged from the host vectors otherwise).
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
        // Reduced sub-system objects (same handle and config; the data
        // object and the matrix wrappers are recreated at every rebuild).
        bool m_cudss_reduced_ready = false;
        cudssData_t m_cudss_reduced_data = nullptr;
        cudssMatrix_t m_cudss_reduced_matrix = nullptr;
        cudssMatrix_t m_cudss_reduced_rhs = nullptr;
        cudssMatrix_t m_cudss_reduced_solution = nullptr;
        amrex::Gpu::DeviceVector<int> m_device_row_offsets;
        amrex::Gpu::DeviceVector<int> m_device_column_indices;
        amrex::Gpu::DeviceVector<amrex::Real> m_device_values;
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
    if (m_cudss_reduced_ready)
    {
        cudssMatrixDestroy(m_cudss_reduced_solution);
        cudssMatrixDestroy(m_cudss_reduced_rhs);
        cudssMatrixDestroy(m_cudss_reduced_matrix);
        cudssDataDestroy(m_cudss_handle, m_cudss_reduced_data);
        m_cudss_reduced_ready = false;
    }
    if (m_cudss_ready)
    {
        if (m_cudss_matrix != nullptr)
        {
            cudssMatrixDestroy(m_cudss_solution);
            cudssMatrixDestroy(m_cudss_rhs);
            cudssMatrixDestroy(m_cudss_matrix);
            cudssDataDestroy(m_cudss_handle, m_cudss_data);
        }
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

    // Device value refresh and device RHS/solution transport (see the
    // class comment): auto engages on single-rank runs; multi-rank runs
    // keep the host gather path.
    {
        std::string device_assembly = "auto";
        const amrex::ParmParse pp_implicit_mhd("implicit_mhd");
        pp_implicit_mhd.query("resistive_direct_device_assembly", device_assembly);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            device_assembly == "auto" || device_assembly == "on" ||
                device_assembly == "off",
            "implicit_mhd.resistive_direct_device_assembly must be auto, on, or "
            "off");
        const bool single_rank = amrex::ParallelDescriptor::NProcs() == 1;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            device_assembly != "on" || single_rank,
            "implicit_mhd.resistive_direct_device_assembly = on requires a "
            "single-rank run (multi-rank assembles through the host gather)");
        m_use_device_assembly =
            device_assembly == "on" || (device_assembly == "auto" && single_rank);

        // Measurement / tuning knobs (defaults: today's behaviour).
        pp_implicit_mhd.queryarr("resistive_direct_cudss_options", m_cudss_options);
        pp_implicit_mhd.query("resistive_direct_solve_host_sync",
                              m_host_sync_after_solve);
        pp_implicit_mhd.query("resistive_direct_dump_prefix", m_dump_prefix);
        pp_implicit_mhd.query("resistive_direct_dump_assembly", m_dump_assembly);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_dump_assembly >= 1,
            "implicit_mhd.resistive_direct_dump_assembly must be >= 1 (the "
            "1-based count of the assembly whose matrix is written)");
        utils::parser::queryWithParser(pp_implicit_mhd, "resistive_direct_row_threshold",
                                       m_row_threshold);
        utils::parser::queryWithParser(pp_implicit_mhd,
                                       "resistive_direct_row_threshold_margin",
                                       m_row_threshold_margin);
        m_reduced_active = m_row_threshold >= 0.0;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_reduced_active || m_use_device_assembly,
            "implicit_mhd.resistive_direct_row_threshold requires the single-rank "
            "device assembly path (implicit_mhd.resistive_direct_device_assembly)");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_row_threshold_margin >= 1.0,
            "implicit_mhd.resistive_direct_row_threshold_margin must be >= 1");
        std::string precision = "double";
        pp_implicit_mhd.query("resistive_direct_precision", precision);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            precision == "double" || precision == "single",
            "implicit_mhd.resistive_direct_precision must be double or single");
        m_single_precision = precision == "single";
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_single_precision || m_use_device_assembly,
            "implicit_mhd.resistive_direct_precision = single requires the single-rank "
            "device assembly path (implicit_mhd.resistive_direct_device_assembly)");
    }

    // Device-copyable twin of WrapPoint/GlobalIndex for the kernels.
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_index_view.domain_low[direction] = domain.smallEnd(direction);
        m_index_view.domain_length[direction] = domain.length(direction);
        m_index_view.domain_periodic[direction] =
            m_geometry.isPeriodic(direction) ? 1 : 0;
    }
    for (int component = 0; component < 3; ++component)
    {
        const ComponentMap& map = m_component_map[component];
        if (!map.active)
        {
            continue;
        }
        m_index_view.component_offset[component] = map.offset;
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            m_index_view.component_low[component][direction] = map.low[direction];
            m_index_view.component_high[component][direction] = map.high[direction];
        }
    }

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

        if (m_use_device_assembly)
        {
            // Frozen per-row emission ranges, filled at the pattern
            // freeze and uploaded once (see UploadFrozenMaps).
            m_entry_base_host[component] = std::make_unique<amrex::iMultiFab>(
                faces.boxArray(), faces.DistributionMap(), 2, 0,
                amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
            m_entry_base_host[component]->setVal(-1, 0, 1, 0);
            m_entry_base_host[component]->setVal(0, 1, 1, 0);
            m_entry_base_device[component] = std::make_unique<amrex::iMultiFab>(
                faces.boxArray(), faces.DistributionMap(), 2, 0);
        }

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
    m_device_rhs.resize(m_num_rows);
    m_device_solution.resize(m_num_rows);
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
            amrex::Array4<int> entry_base;
            if (build_pattern && m_entry_base_host[component])
            {
                entry_base = m_entry_base_host[component]->array(mfi);
            }
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
                    const std::size_t row_entry_start = entry;
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
                    if (entry_base)
                    {
                        // Frozen per-row emission range of the device
                        // value refresh (local entry order).
                        entry_base(i, j, k, 0) = static_cast<int>(row_entry_start);
                        entry_base(i, j, k, 1) =
                            static_cast<int>(entry - row_entry_start);
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
    const bool first_assembly = !m_pattern_built;
    // The one-time pattern freeze runs on the host; afterwards the host
    // emission+fold survives only as the multi-rank fallback and as the
    // parity reference of the validation mode -- the production
    // single-rank refreeze is the device refresh alone, with no
    // coefficient mirror traffic.
    const bool host_assembly =
        first_assembly || !m_use_device_assembly || m_validate_parity;

    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            a_eta_edge[component] == m_eta_source[component],
            "ResistiveDirectSolver eta_field staggerings changed after Define()");
        if (host_assembly)
        {
            amrex::dtoh_memcpy(*m_eta_host[component], *a_eta_edge[component]);
        }
        if (m_eta_h_host[component])
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_eta_h_edge[component] == m_eta_h_source[component],
                "ResistiveDirectSolver eta_H staggerings changed after Define()");
            if (host_assembly)
            {
                amrex::dtoh_memcpy(*m_eta_h_host[component],
                                   *a_eta_h_edge[component]);
            }
        }
        if (m_hall_host[component])
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_hall_edge[component] == m_hall_source[component],
                "ResistiveDirectSolver Hall staggerings changed after Define()");
            if (host_assembly)
            {
                amrex::dtoh_memcpy(*m_hall_host[component], *a_hall_edge[component]);
            }
        }
        if (m_inertia_host[component])
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_inertia_edge[component] == m_inertia_source[component],
                "ResistiveDirectSolver inertia staggerings changed after Define()");
            if (host_assembly)
            {
                amrex::dtoh_memcpy(*m_inertia_host[component],
                                   *a_inertia_edge[component]);
            }
        }
    }

    if (host_assembly)
    {
        amrex::Gpu::streamSynchronize();
        EmitLocalRows(a_constants);
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
        RefreshValuesOnDevice(a_constants, a_eta_edge, a_eta_h_edge, a_hall_edge,
                              a_inertia_edge);
        if (host_assembly)
        {
            CheckAssemblyParity();
        }
    }

    ++m_assembly_count;
    if (m_reduced_active)
    {
        // First assembly: build the row set. Later ones: rebuild only when a
        // dropped row's coupling has grown past the threshold (the bound of
        // the reduced solve would otherwise be violated).
        bool rebuild = first_assembly;
        if (!rebuild)
        {
            const amrex::Real drift = DroppedCouplingMax();
            rebuild = drift > m_row_threshold;
            if (rebuild)
            {
                amrex::Print() << "pc_mhd_block resistive direct reduced solve: a dropped "
                                  "row's coupling reached "
                               << drift << " > " << m_row_threshold
                               << ", rebuilding the row set\n";
            }
        }
        if (rebuild)
        {
            BuildReducedSystem(HostValues());
        }
    }
    if (!m_dump_prefix.empty() && m_assembly_count == m_dump_assembly)
    {
        DumpMatrix();
        if (m_reduced_active)
        {
            DumpReducedMatrix(HostValues());
        }
        m_dump_solve_pending = true;
    }
}

inline amrex::Vector<amrex::Real>
ResistiveDirectSolver::HostValues () const
{
    amrex::Vector<amrex::Real> values;
    if (m_use_device_assembly)
    {
        values.resize(m_values_device.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_values_device.begin(),
                         m_values_device.end(), values.begin());
    }
    else
    {
        values = m_matrix_values;
    }
    return values;
}

inline void
ResistiveDirectSolver::BuildReducedSystem (const amrex::Vector<amrex::Real>& a_values)
{
    BL_PROFILE("ResistiveDirect::BuildReducedSystem()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_pattern_built && static_cast<amrex::Long>(a_values.size()) ==
                               static_cast<amrex::Long>(m_column_indices.size()),
        "ResistiveDirectSolver::BuildReducedSystem() needs the frozen pattern and its "
        "values");
    const amrex::Long num_rows = m_num_rows;
    const amrex::Long num_slots = m_column_indices.size();

    // Global-slot helpers (once): the row of every slot and the diagonal
    // slot of every row.
    if (m_slot_row_device.empty())
    {
        amrex::Vector<int> slot_row(num_slots);
        amrex::Vector<int> diag_slot(num_rows, -1);
        for (amrex::Long row = 0; row < num_rows; ++row)
        {
            for (int slot = m_row_offsets[row]; slot < m_row_offsets[row + 1]; ++slot)
            {
                slot_row[slot] = static_cast<int>(row);
                if (m_column_indices[slot] == row)
                {
                    diag_slot[row] = slot;
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                diag_slot[row] >= 0,
                "ResistiveDirectSolver reduced solve: a canonical row has no diagonal");
        }
        m_slot_row_device.resize(num_slots);
        m_slot_column_device.resize(num_slots);
        m_diag_slot_device.resize(num_rows);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, slot_row.begin(), slot_row.end(),
                              m_slot_row_device.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_column_indices.begin(),
                              m_column_indices.end(), m_slot_column_device.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, diag_slot.begin(), diag_slot.end(),
                              m_diag_slot_device.begin());
        m_keep_device.resize(num_rows);
    }

    // Keep a row when any off-diagonal coupling in its row or its column
    // exceeds (threshold / margin) x the diagonal of that row/column.
    const amrex::Real keep_threshold = m_row_threshold / m_row_threshold_margin;
    amrex::Vector<amrex::Real> diagonal(num_rows, 0.0);
    for (amrex::Long row = 0; row < num_rows; ++row)
    {
        for (int slot = m_row_offsets[row]; slot < m_row_offsets[row + 1]; ++slot)
        {
            if (m_column_indices[slot] == row)
            {
                diagonal[row] = std::abs(a_values[slot]);
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagonal[row] > 0.0,
            "ResistiveDirectSolver reduced solve: a canonical row has a zero diagonal");
    }
    amrex::Vector<int> keep(num_rows, 0);
    for (amrex::Long row = 0; row < num_rows; ++row)
    {
        for (int slot = m_row_offsets[row]; slot < m_row_offsets[row + 1]; ++slot)
        {
            const int column = m_column_indices[slot];
            if (column == row)
            {
                continue;
            }
            const amrex::Real magnitude = std::abs(a_values[slot]);
            if (magnitude > keep_threshold * diagonal[row])
            {
                keep[row] = 1;
            }
            if (magnitude > keep_threshold * diagonal[column])
            {
                keep[column] = 1;
            }
        }
    }

    // The sub-system: kept rows in canonical order, columns restricted to
    // kept rows (the dropped couplings are the bounded truncation).
    m_reduced_row_list.clear();
    amrex::Vector<int> local_index(num_rows, -1);
    for (amrex::Long row = 0; row < num_rows; ++row)
    {
        if (keep[row] != 0)
        {
            local_index[row] = static_cast<int>(m_reduced_row_list.size());
            m_reduced_row_list.push_back(static_cast<int>(row));
        }
    }
    m_reduced_rows = m_reduced_row_list.size();
    m_reduced_row_offsets.assign(m_reduced_rows + 1, 0);
    m_reduced_columns.clear();
    m_reduced_slots.clear();
    for (amrex::Long local = 0; local < m_reduced_rows; ++local)
    {
        const int row = m_reduced_row_list[local];
        for (int slot = m_row_offsets[row]; slot < m_row_offsets[row + 1]; ++slot)
        {
            const int column_local = local_index[m_column_indices[slot]];
            if (column_local >= 0)
            {
                m_reduced_columns.push_back(column_local);
                m_reduced_slots.push_back(slot);
            }
        }
        m_reduced_row_offsets[local + 1] = static_cast<int>(m_reduced_columns.size());
    }

    // Device copies.
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, keep.begin(), keep.end(),
                          m_keep_device.begin());
    m_reduced_row_list_device.resize(m_reduced_row_list.size());
    m_reduced_row_offsets_device.resize(m_reduced_row_offsets.size());
    m_reduced_columns_device.resize(m_reduced_columns.size());
    m_reduced_slots_device.resize(m_reduced_slots.size());
    m_reduced_values_device.resize(m_reduced_slots.size());
    m_reduced_rhs_device.resize(std::max<amrex::Long>(m_reduced_rows, 1));
    m_reduced_solution_device.resize(std::max<amrex::Long>(m_reduced_rows, 1));
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_reduced_row_list.begin(),
                          m_reduced_row_list.end(), m_reduced_row_list_device.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_reduced_row_offsets.begin(),
                          m_reduced_row_offsets.end(), m_reduced_row_offsets_device.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_reduced_columns.begin(),
                          m_reduced_columns.end(), m_reduced_columns_device.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_reduced_slots.begin(),
                          m_reduced_slots.end(), m_reduced_slots_device.begin());
    amrex::Gpu::streamSynchronize();
    m_reduced_pattern_new = true;
    ++m_reduced_rebuilds;
    amrex::Print() << "pc_mhd_block resistive direct reduced solve: " << m_reduced_rows
                   << " of " << num_rows << " rows kept (" << m_reduced_columns.size()
                   << " of " << num_slots << " nonzeros; threshold " << m_row_threshold
                   << ", build threshold " << keep_threshold << "; build #"
                   << m_reduced_rebuilds << " at assembly " << m_assembly_count << ")\n";
}

inline amrex::Real
ResistiveDirectSolver::DroppedCouplingMax () const
{
    if (m_keep_device.empty())
    {
        return 0.0;
    }
    return direct_detail::DroppedCouplingMax(
        m_values_device.data(), m_slot_row_device.data(), m_slot_column_device.data(),
        m_diag_slot_device.data(), m_keep_device.data(),
        static_cast<amrex::Long>(m_values_device.size()));
}

inline void
ResistiveDirectSolver::DumpReducedMatrix (const amrex::Vector<amrex::Real>& a_values) const
{
    if (!amrex::ParallelDescriptor::IOProcessor() || !m_reduced_active)
    {
        return;
    }
    // Same layout as the full dump (component records left inactive: the
    // rows are the listed subset of the canonical unknowns), plus the row
    // list as a vector file.
    const auto num_rows = static_cast<std::int64_t>(m_reduced_rows);
    const auto num_values = static_cast<std::int64_t>(m_reduced_columns.size());
    const std::string path = m_dump_prefix + "_reduced_matrix.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        out.good(), "ResistiveDirectSolver could not open the dump file " + path);
    const char magic[9] = "WXCSR001";
    out.write(magic, 8);
    out.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
    out.write(reinterpret_cast<const char*>(&num_values), sizeof(num_values));
    const std::int32_t spacedim = AMREX_SPACEDIM;
    const std::int32_t num_components = 3;
    out.write(reinterpret_cast<const char*>(&spacedim), sizeof(spacedim));
    out.write(reinterpret_cast<const char*>(&num_components), sizeof(num_components));
    for (int component = 0; component < 3; ++component)
    {
        const std::int32_t active = 0;
        const std::int64_t offset = 0;
        const std::int32_t zeros[3] = {0, 0, 0};
        out.write(reinterpret_cast<const char*>(&active), sizeof(active));
        out.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        out.write(reinterpret_cast<const char*>(zeros), sizeof(zeros));
        out.write(reinterpret_cast<const char*>(zeros), sizeof(zeros));
    }
    out.write(reinterpret_cast<const char*>(m_reduced_row_offsets.data()),
              static_cast<std::streamsize>(sizeof(int) * m_reduced_row_offsets.size()));
    out.write(reinterpret_cast<const char*>(m_reduced_columns.data()),
              static_cast<std::streamsize>(sizeof(int) * m_reduced_columns.size()));
    amrex::Vector<amrex::Real> values(m_reduced_slots.size());
    for (std::size_t entry = 0; entry < values.size(); ++entry)
    {
        values[entry] = a_values[m_reduced_slots[entry]];
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(sizeof(amrex::Real) * values.size()));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        out.good(), "ResistiveDirectSolver failed writing the dump file " + path);
    amrex::Vector<amrex::Real> rows(m_reduced_row_list.size());
    for (std::size_t entry = 0; entry < rows.size(); ++entry)
    {
        rows[entry] = static_cast<amrex::Real>(m_reduced_row_list[entry]);
    }
    DumpVector(m_dump_prefix + "_reduced_rows.bin", rows.data(),
               static_cast<amrex::Long>(rows.size()));
    amrex::Print() << "pc_mhd_block resistive direct: wrote the reduced sub-system ("
                   << num_rows << " rows, " << num_values << " nonzeros) to " << path
                   << "\n";
}

inline void
ResistiveDirectSolver::DumpVector (const std::string& a_path,
                                   const amrex::Real* const a_values,
                                   const amrex::Long a_size)
{
    static_assert(sizeof(amrex::Real) == 8, "the dump format stores float64 values");
    std::ofstream out(a_path, std::ios::binary | std::ios::trunc);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        out.good(), "ResistiveDirectSolver could not open the dump file " + a_path);
    const char magic[9] = "WXVEC001";
    out.write(magic, 8);
    const auto size = static_cast<std::int64_t>(a_size);
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(a_values),
              static_cast<std::streamsize>(sizeof(amrex::Real) * a_size));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        out.good(), "ResistiveDirectSolver failed writing the dump file " + a_path);
}

inline void
ResistiveDirectSolver::DumpMatrix () const
{
    if (!amrex::ParallelDescriptor::IOProcessor())
    {
        return;
    }
    // The current values: the device refresh updates m_values_device in
    // place (single-rank device path); the host fold fills
    // m_matrix_values otherwise.
    amrex::Vector<amrex::Real> values;
    const amrex::Real* value_data = m_matrix_values.data();
    if (m_use_device_assembly)
    {
        values.resize(m_values_device.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_values_device.begin(),
                         m_values_device.end(), values.begin());
        value_data = values.data();
    }
    const auto num_rows = static_cast<std::int64_t>(m_num_rows);
    const auto num_values = static_cast<std::int64_t>(m_column_indices.size());
    const std::string path = m_dump_prefix + "_matrix.bin";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            out.good(), "ResistiveDirectSolver could not open the dump file " + path);
        const char magic[9] = "WXCSR001";
        out.write(magic, 8);
        out.write(reinterpret_cast<const char*>(&num_rows), sizeof(num_rows));
        out.write(reinterpret_cast<const char*>(&num_values), sizeof(num_values));
        const std::int32_t spacedim = AMREX_SPACEDIM;
        const std::int32_t num_components = 3;
        out.write(reinterpret_cast<const char*>(&spacedim), sizeof(spacedim));
        out.write(reinterpret_cast<const char*>(&num_components),
                  sizeof(num_components));
        for (int component = 0; component < 3; ++component)
        {
            const ComponentMap& map = m_component_map[component];
            const std::int32_t active = map.active ? 1 : 0;
            const auto offset = static_cast<std::int64_t>(map.offset);
            std::int32_t low[3] = {0, 0, 0};
            std::int32_t high[3] = {0, 0, 0};
            for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
            {
                low[direction] = map.active ? map.low[direction] : 0;
                high[direction] = map.active ? map.high[direction] : 0;
            }
            out.write(reinterpret_cast<const char*>(&active), sizeof(active));
            out.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
            out.write(reinterpret_cast<const char*>(low), sizeof(low));
            out.write(reinterpret_cast<const char*>(high), sizeof(high));
        }
        static_assert(sizeof(int) == 4, "the dump format stores int32 indices");
        out.write(reinterpret_cast<const char*>(m_row_offsets.data()),
                  static_cast<std::streamsize>(sizeof(int) * m_row_offsets.size()));
        out.write(reinterpret_cast<const char*>(m_column_indices.data()),
                  static_cast<std::streamsize>(sizeof(int) * m_column_indices.size()));
        out.write(reinterpret_cast<const char*>(value_data),
                  static_cast<std::streamsize>(sizeof(amrex::Real) * num_values));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            out.good(), "ResistiveDirectSolver failed writing the dump file " + path);
    }
    {
        std::ofstream meta(m_dump_prefix + "_meta.txt", std::ios::trunc);
        meta << "rows " << num_rows << "\nnonzeros " << num_values << "\nspacedim "
             << AMREX_SPACEDIM << "\nassembly " << m_assembly_count << "\nvalues "
             << (m_use_device_assembly ? "device" : "host") << "\n";
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            meta << "periodic" << direction << " "
                 << (m_geometry.isPeriodic(direction) ? 1 : 0) << "\n";
        }
        for (int component = 0; component < 3; ++component)
        {
            const ComponentMap& map = m_component_map[component];
            meta << "component" << component << " active " << (map.active ? 1 : 0)
                 << " offset " << map.offset;
            if (map.active)
            {
                for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
                {
                    meta << " low" << direction << " " << map.low[direction] << " high"
                         << direction << " " << map.high[direction];
                }
            }
            meta << "\n";
        }
    }
    amrex::Print() << "pc_mhd_block resistive direct: wrote the assembled matrix ("
                   << num_rows << " rows, " << num_values << " nonzeros, assembly "
                   << m_assembly_count << ") to " << path << "\n";
}

inline void
ResistiveDirectSolver::UploadFrozenMaps ()
{
    // Single-rank device assembly: the gathered global entry order IS the
    // local emission order (BuildGlobalPattern's single-rank branch), so
    // the entry-to-slot map and the per-row emission ranges recorded at
    // the pattern freeze index the same entries. One-time transfers.
    for (int component = 0; component < 3; ++component)
    {
        if (m_entry_base_host[component])
        {
            amrex::htod_memcpy(*m_entry_base_device[component],
                               *m_entry_base_host[component]);
        }
    }
    m_entry_slot_device.resize(m_entry_slot.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_entry_slot.begin(),
                          m_entry_slot.end(), m_entry_slot_device.begin());
    m_values_device.resize(m_matrix_values.size());
    m_structure_errors_device.resize(1);
    amrex::Gpu::streamSynchronize();
}

inline void
ResistiveDirectSolver::RefreshValuesOnDevice (
    const ResistiveStencilConstants& a_constants,
    const amrex::Array<const amrex::MultiFab*, 3>& a_eta_edge,
    const amrex::Array<const amrex::MultiFab*, 3>& a_eta_h_edge,
    const amrex::Array<const amrex::MultiFab*, 3>& a_hall_edge,
    const amrex::Array<const amrex::MultiFab*, 3>& a_inertia_edge)
{
    BL_PROFILE("ResistiveDirect::RefreshValuesOnDevice()");

    direct_detail::ZeroRealVector(m_values_device.data(), m_values_device.size());
    direct_detail::ZeroIntScalar(m_structure_errors_device.data());
    for (int component = 0; component < 3; ++component)
    {
        if (!m_component_map[component].active)
        {
            continue;
        }
        direct_detail::RefreshComponentValues(
            a_constants, component, *m_entry_base_device[component], a_eta_edge,
            a_eta_h_edge, a_hall_edge, a_inertia_edge, m_entry_slot_device.data(),
            m_values_device.data(), m_structure_errors_device.data());
    }
    int structure_errors = 0;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_structure_errors_device.begin(),
                     m_structure_errors_device.end(), &structure_errors);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        structure_errors == 0,
        "ResistiveDirectSolver device value refresh changed the emission "
        "structure (the frozen sparsity pattern was violated)");
}

inline void
ResistiveDirectSolver::CheckAssemblyParity () const
{
    // Bitwise gate of the device value refresh against the host
    // emission+fold on the same frozen state: identical EmitResistiveRow
    // arithmetic accumulated in the identical per-slot order, so exact
    // FP64 equality is expected (and observed on CPU builds, where both
    // paths run the same compiled code); the <= 4 ULP allowance absorbs
    // device FMA contraction of the emission arithmetic only.
    const std::size_t num_values = m_matrix_values.size();
    amrex::Vector<amrex::Real> device_values(num_values);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_values_device.begin(),
                     m_values_device.end(), device_values.begin());
    std::uint64_t max_ulp = 0;
    amrex::Real max_difference = 0.0;
    for (std::size_t slot = 0; slot < num_values; ++slot)
    {
        const std::uint64_t host_bits = OrderedBits(m_matrix_values[slot]);
        const std::uint64_t device_bits = OrderedBits(device_values[slot]);
        max_ulp = std::max(max_ulp, host_bits > device_bits
                                        ? host_bits - device_bits
                                        : device_bits - host_bits);
        max_difference =
            amrex::max(max_difference,
                       std::abs(m_matrix_values[slot] - device_values[slot]));
    }
    amrex::Print() << "pc_mhd_block resistive direct assembly parity: max ULP = "
                   << max_ulp << " (max |device - host| = " << max_difference
                   << " over " << num_values << " values)\n";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        max_ulp <= 4U,
        "ResistiveDirectSolver device-assembled values differ from the host "
        "assembly beyond FMA contraction (> 4 ULP)");
}

#if defined(WARPX_USE_CUDSS)
inline void
ResistiveDirectSolver::ApplyCudssOptions () const
{
    // "name=value" pairs from implicit_mhd.resistive_direct_cudss_options,
    // applied once before the analysis (the phase that consumes the
    // reordering/pivoting/algorithm choices).
    const auto set_enum = [&] (const cudssConfigParam_t a_param, const int a_value)
    {
        WARPX_CUDSS_CHECK(cudssConfigSet(m_cudss_config, a_param, &a_value,
                                         sizeof(a_value)));
    };
    const auto parse_int = [] (const std::string& a_name, const std::string& a_text)
    {
        std::size_t consumed = 0;
        int value = 0;
        try
        {
            value = std::stoi(a_text, &consumed);
        }
        catch (const std::exception&)
        {
            consumed = 0;
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            consumed == a_text.size() && !a_text.empty(),
            "implicit_mhd.resistive_direct_cudss_options: '" + a_name +
                "' expects an integer, got '" + a_text + "'");
        return value;
    };
    const auto match = [] (const std::string& a_name, const std::string& a_text,
                           const std::vector<std::pair<std::string, int>>& a_table)
    {
        for (const auto& entry : a_table)
        {
            if (entry.first == a_text)
            {
                return entry.second;
            }
        }
        std::string allowed;
        for (const auto& entry : a_table)
        {
            allowed += (allowed.empty() ? "" : ", ") + entry.first;
        }
        WARPX_ABORT_WITH_MESSAGE("implicit_mhd.resistive_direct_cudss_options: '" +
                                 a_name + "' does not accept '" + a_text +
                                 "' (allowed: " + allowed + ")");
        return 0;
    };
    for (const std::string& option : m_cudss_options)
    {
        const std::size_t separator = option.find('=');
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            separator != std::string::npos && separator > 0 &&
                separator + 1 < option.size(),
            "implicit_mhd.resistive_direct_cudss_options entries are name=value "
            "pairs, got '" + option + "'");
        const std::string name = option.substr(0, separator);
        const std::string text = option.substr(separator + 1);
        if (name == "reordering_alg")
        {
            set_enum(CUDSS_CONFIG_REORDERING_ALG,
                     match(name, text,
                           {{"default", CUDSS_REORDERING_ALG_DEFAULT},
                            {"btf_colamd", CUDSS_REORDERING_ALG_BTF_COLAMD},
                            {"colamd", CUDSS_REORDERING_ALG_COLAMD},
                            {"amd", CUDSS_REORDERING_ALG_AMD},
                            {"nested_dissection", CUDSS_REORDERING_ALG_NESTED_DISSECTION},
                            {"none", CUDSS_REORDERING_ALG_NONE}}));
        }
        else if (name == "factorization_alg")
        {
            set_enum(CUDSS_CONFIG_FACTORIZATION_ALG,
                     match(name, text,
                           {{"default", CUDSS_FACTORIZATION_ALG_DEFAULT},
                            {"multiblock", CUDSS_FACTORIZATION_ALG_MULTIBLOCK},
                            {"general", CUDSS_FACTORIZATION_ALG_GENERAL}}));
        }
        else if (name == "solve_alg")
        {
            set_enum(CUDSS_CONFIG_SOLVE_ALG,
                     match(name, text,
                           {{"default", CUDSS_SOLVE_ALG_DEFAULT},
                            {"general", CUDSS_SOLVE_ALG_GENERAL}}));
        }
        else if (name == "matching_alg")
        {
            set_enum(CUDSS_CONFIG_MATCHING_ALG,
                     match(name, text,
                           {{"none", CUDSS_MATCHING_ALG_NONE},
                            {"max_diag_count", CUDSS_MATCHING_ALG_MAX_DIAG_COUNT},
                            {"max_min_diag", CUDSS_MATCHING_ALG_MAX_MIN_DIAG},
                            {"max_min_diag_alt", CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT},
                            {"max_diag_sum", CUDSS_MATCHING_ALG_MAX_DIAG_SUM},
                            {"max_diag_product", CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT},
                            {"auto", CUDSS_MATCHING_ALG_AUTO}}));
        }
        else if (name == "pivot_type")
        {
            set_enum(CUDSS_CONFIG_PIVOT_TYPE,
                     match(name, text,
                           {{"auto", CUDSS_PIVOT_AUTO},
                            {"none", CUDSS_PIVOT_NONE},
                            {"global_col", CUDSS_PIVOT_GLOBAL_COL},
                            {"global_row", CUDSS_PIVOT_GLOBAL_ROW},
                            {"diagonal", CUDSS_PIVOT_DIAGONAL},
                            {"local_block", CUDSS_PIVOT_LOCAL_BLOCK}}));
        }
        else if (name == "ir_n_steps")
        {
            set_enum(CUDSS_CONFIG_IR_N_STEPS, parse_int(name, text));
        }
        else if (name == "nd_nlevels")
        {
            set_enum(CUDSS_CONFIG_ND_NLEVELS, parse_int(name, text));
        }
        else if (name == "nd_ubfactor")
        {
            set_enum(CUDSS_CONFIG_ND_UBFACTOR, parse_int(name, text));
        }
        else if (name == "use_superpanels")
        {
            set_enum(CUDSS_CONFIG_USE_SUPERPANELS, parse_int(name, text));
        }
        else if (name == "deterministic_mode")
        {
            set_enum(CUDSS_CONFIG_DETERMINISTIC_MODE, parse_int(name, text));
        }
        else if (name == "host_nthreads")
        {
            set_enum(CUDSS_CONFIG_HOST_NTHREADS, parse_int(name, text));
        }
        else
        {
            WARPX_ABORT_WITH_MESSAGE(
                "implicit_mhd.resistive_direct_cudss_options: unknown option '" + name +
                "' (known: reordering_alg, factorization_alg, solve_alg, "
                "matching_alg, pivot_type, ir_n_steps, nd_nlevels, nd_ubfactor, "
                "use_superpanels, deterministic_mode, host_nthreads)");
        }
        amrex::Print() << "pc_mhd_block resistive direct: cuDSS option " << name << " = "
                       << text << "\n";
    }
}
#endif

#if defined(WARPX_USE_CUDSS)
inline void
ResistiveDirectSolver::FactorizeReduced ()
{
    BL_PROFILE("ResistiveDirect::FactorizeReduced()");

    // Sub-system values gathered from the refreshed global values (stream
    // ordered after the refresh).
    direct_detail::GatherReducedValues(
        m_values_device.data(), m_reduced_slots_device.data(),
        m_reduced_values_device.data(),
        static_cast<amrex::Long>(m_reduced_slots_device.size()));
    if (m_reduced_rows == 0)
    {
        // Every row is trivial: nothing to factorize.
        m_reduced_pattern_new = false;
        return;
    }
    const auto rows = static_cast<std::int64_t>(m_reduced_rows);
    const auto nonzeros = static_cast<std::int64_t>(m_reduced_columns.size());
    const bool pattern_new = m_reduced_pattern_new;
    if (m_reduced_pattern_new)
    {
        if (m_cudss_reduced_ready)
        {
            WARPX_CUDSS_CHECK(cudssMatrixDestroy(m_cudss_reduced_solution));
            WARPX_CUDSS_CHECK(cudssMatrixDestroy(m_cudss_reduced_rhs));
            WARPX_CUDSS_CHECK(cudssMatrixDestroy(m_cudss_reduced_matrix));
            WARPX_CUDSS_CHECK(cudssDataDestroy(m_cudss_handle, m_cudss_reduced_data));
            m_cudss_reduced_ready = false;
        }
        WARPX_CUDSS_CHECK(cudssDataCreate(m_cudss_handle, &m_cudss_reduced_data));
        const void* wrapped_values = m_reduced_values_device.data();
        void* wrapped_rhs = m_reduced_rhs_device.data();
        void* wrapped_solution = m_reduced_solution_device.data();
        cudssDataType_t value_type = CUDSS_R_64F;
        if (m_single_precision)
        {
            m_values_single.resize(m_reduced_columns.size());
            m_rhs_single.resize(std::max<amrex::Long>(m_reduced_rows, 1));
            m_solution_single.resize(std::max<amrex::Long>(m_reduced_rows, 1));
            wrapped_values = m_values_single.data();
            wrapped_rhs = m_rhs_single.data();
            wrapped_solution = m_solution_single.data();
            value_type = CUDSS_R_32F;
        }
        WARPX_CUDSS_CHECK(cudssMatrixCreateCsr(
            &m_cudss_reduced_matrix, rows, rows, nonzeros,
            m_reduced_row_offsets_device.data(), nullptr, m_reduced_columns_device.data(),
            wrapped_values, CUDSS_R_32I, CUDSS_R_32I, value_type,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_reduced_rhs, rows, 1, rows,
                                              wrapped_rhs, value_type,
                                              CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_reduced_solution, rows, 1, rows,
                                              wrapped_solution, value_type,
                                              CUDSS_LAYOUT_COL_MAJOR));
        amrex::Gpu::streamSynchronize();
        const amrex::Real analysis_start = amrex::ParallelDescriptor::second();
        WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_ANALYSIS,
                                       m_cudss_config, m_cudss_reduced_data,
                                       m_cudss_reduced_matrix, m_cudss_reduced_solution,
                                       m_cudss_reduced_rhs));
        amrex::Gpu::streamSynchronize();
        amrex::Print() << "pc_mhd_block resistive direct reduced solve: analysis of the "
                       << rows << "-row sub-system "
                       << amrex::ParallelDescriptor::second() - analysis_start << " s\n";
        m_cudss_reduced_ready = true;
        m_reduced_pattern_new = false;
    }
    if (m_single_precision)
    {
        direct_detail::ConvertToSingle(m_reduced_values_device.data(),
                                       m_values_single.data(), nonzeros);
        WARPX_CUDSS_CHECK(cudssMatrixSetValues(m_cudss_reduced_matrix,
                                               m_values_single.data()));
    }
    else if (!pattern_new)
    {
        WARPX_CUDSS_CHECK(cudssMatrixSetValues(m_cudss_reduced_matrix,
                                               m_reduced_values_device.data()));
    }
    WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_FACTORIZATION,
                                   m_cudss_config, m_cudss_reduced_data,
                                   m_cudss_reduced_matrix, m_cudss_reduced_solution,
                                   m_cudss_reduced_rhs));
    amrex::Gpu::streamSynchronize();
}
#endif

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
        ApplyCudssOptions();
        m_cudss_ready = true;
    }
    if (m_reduced_active)
    {
        // The reduced sub-system is the only factorized object (the full
        // matrix is never analysed).
        FactorizeReduced();
        return;
    }
    if (m_cudss_matrix == nullptr)
    {
        WARPX_CUDSS_CHECK(cudssDataCreate(m_cudss_handle, &m_cudss_data));

        // Device assembly refreshes m_values_device in place, so the CSR
        // wraps it directly; the host path stages through m_device_values.
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

        const void* wrapped_values = matrix_values;
        void* wrapped_rhs = m_device_rhs.data();
        void* wrapped_solution = m_device_solution.data();
        cudssDataType_t value_type = CUDSS_R_64F;
        if (m_single_precision)
        {
            m_values_single.resize(m_matrix_values.size());
            m_rhs_single.resize(m_num_rows);
            m_solution_single.resize(m_num_rows);
            direct_detail::ConvertToSingle(matrix_values, m_values_single.data(),
                                           num_values);
            wrapped_values = m_values_single.data();
            wrapped_rhs = m_rhs_single.data();
            wrapped_solution = m_solution_single.data();
            value_type = CUDSS_R_32F;
            amrex::Gpu::streamSynchronize();
        }
        WARPX_CUDSS_CHECK(cudssMatrixCreateCsr(
            &m_cudss_matrix, num_rows, num_rows, num_values,
            m_device_row_offsets.data(), nullptr, m_device_column_indices.data(),
            wrapped_values, CUDSS_R_32I, CUDSS_R_32I, value_type,
            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_rhs, num_rows, 1, num_rows,
                                              wrapped_rhs, value_type,
                                              CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssMatrixCreateDn(&m_cudss_solution, num_rows, 1, num_rows,
                                              wrapped_solution, value_type,
                                              CUDSS_LAYOUT_COL_MAJOR));
        WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_ANALYSIS,
                                       m_cudss_config, m_cudss_data, m_cudss_matrix,
                                       m_cudss_solution, m_cudss_rhs));
    }
    else if (m_use_device_assembly)
    {
        // The refresh already updated the wrapped device buffer in place
        // (same stream: ordered before the factorization); the FP32 twin
        // is refreshed from it.
        if (m_single_precision)
        {
            direct_detail::ConvertToSingle(m_values_device.data(), m_values_single.data(),
                                           num_values);
            WARPX_CUDSS_CHECK(cudssMatrixSetValues(m_cudss_matrix, m_values_single.data()));
        }
        else
        {
            WARPX_CUDSS_CHECK(cudssMatrixSetValues(m_cudss_matrix, m_values_device.data()));
        }
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
    if (m_use_device_assembly)
    {
        // Single-rank device path: pack the right-hand side on the
        // device; every canonical element is owned locally, so the
        // gather writes each element exactly once (no zero fill).
        if (warpx::fused_residual::Enabled())
        {
            // the three component gathers as one launch (inactive
            // components contribute an empty box)
            amrex::Real* const rhs_data = m_device_rhs.data();
            const DirectIndexView view = m_index_view;
            for (amrex::MFIter mfi(*a_faces[0]); mfi.isValid(); ++mfi)
            {
                amrex::GpuArray<amrex::Box, 3> boxes;
                amrex::GpuArray<amrex::Array4<int const>, 3> entry;
                amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> faces;
                for (int component = 0; component < 3; ++component)
                {
                    const bool active = m_component_map[component].active;
                    boxes[component] = active ? m_entry_base_device[component]->box(mfi.index())
                                              : amrex::Box();
                    entry[component] = active ? m_entry_base_device[component]->const_array(mfi)
                                              : amrex::Array4<int const>{};
                    faces[component] = active ? a_faces[component]->const_array(mfi)
                                              : amrex::Array4<amrex::Real const>{};
                }
                amrex::ParallelFor(
                    boxes[0], boxes[1], boxes[2],
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (entry[0](i, j, k, 0) >= 0) { rhs_data[view.Index(0, i, j, k)] = faces[0](i, j, k); }
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (entry[1](i, j, k, 0) >= 0) { rhs_data[view.Index(1, i, j, k)] = faces[1](i, j, k); }
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (entry[2](i, j, k, 0) >= 0) { rhs_data[view.Index(2, i, j, k)] = faces[2](i, j, k); }
                    });
            }
        }
        for (int component = 0; component < 3 && !warpx::fused_residual::Enabled(); ++component)
        {
            if (!m_component_map[component].active)
            {
                continue;
            }
            direct_detail::GatherComponentRhs(m_index_view, component,
                                              *m_entry_base_device[component],
                                              *a_faces[component],
                                              m_device_rhs.data());
        }
    }
    else
    {
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
    }

#if defined(WARPX_USE_CUDSS)
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_cudss_ready, "ResistiveDirectSolver::Solve() called before Factorize()");
        if (!m_use_device_assembly)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_rhs.begin(),
                                  m_rhs.end(), m_device_rhs.begin());
        }
        if (m_reduced_active)
        {
            // Trivial rows: x_i = b_i / a_ii everywhere (the kept rows are
            // overwritten by the sub-system solve below).
            {
                const amrex::Real* const rhs = m_device_rhs.data();
                const amrex::Real* const values = m_values_device.data();
                const int* const diag_slot = m_diag_slot_device.data();
                amrex::Real* const solution = m_device_solution.data();
                amrex::ParallelFor(m_num_rows,
                                   [=] AMREX_GPU_DEVICE (amrex::Long row)
                                   {
                                       solution[row] = rhs[row] / values[diag_slot[row]];
                                   });
            }
            if (m_reduced_rows > 0)
            {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    m_cudss_reduced_ready,
                    "ResistiveDirectSolver::Solve() reduced solve called before its "
                    "factorization");
                {
                    const amrex::Real* const rhs = m_device_rhs.data();
                    const int* const row_list = m_reduced_row_list_device.data();
                    amrex::Real* const reduced_rhs = m_reduced_rhs_device.data();
                    amrex::ParallelFor(m_reduced_rows,
                                       [=] AMREX_GPU_DEVICE (amrex::Long local)
                                       {
                                           reduced_rhs[local] = rhs[row_list[local]];
                                       });
                }
                if (m_single_precision)
                {
                    direct_detail::ConvertToSingle(m_reduced_rhs_device.data(),
                                                   m_rhs_single.data(), m_reduced_rows);
                }
                WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_SOLVE,
                                               m_cudss_config, m_cudss_reduced_data,
                                               m_cudss_reduced_matrix,
                                               m_cudss_reduced_solution,
                                               m_cudss_reduced_rhs));
                if (m_single_precision)
                {
                    direct_detail::ConvertToDouble(m_solution_single.data(),
                                                   m_reduced_solution_device.data(),
                                                   m_reduced_rows);
                }
                {
                    const amrex::Real* const reduced_solution =
                        m_reduced_solution_device.data();
                    const int* const row_list = m_reduced_row_list_device.data();
                    amrex::Real* const solution = m_device_solution.data();
                    amrex::ParallelFor(m_reduced_rows,
                                       [=] AMREX_GPU_DEVICE (amrex::Long local)
                                       {
                                           solution[row_list[local]] =
                                               reduced_solution[local];
                                       });
                }
            }
        }
        else
        {
            if (m_single_precision)
            {
                direct_detail::ConvertToSingle(m_device_rhs.data(), m_rhs_single.data(),
                                               m_num_rows);
            }
            WARPX_CUDSS_CHECK(cudssExecute(m_cudss_handle, CUDSS_PHASE_SOLVE,
                                           m_cudss_config, m_cudss_data, m_cudss_matrix,
                                           m_cudss_solution, m_cudss_rhs));
            if (m_single_precision)
            {
                direct_detail::ConvertToDouble(m_solution_single.data(),
                                               m_device_solution.data(), m_num_rows);
            }
        }
        if (!m_use_device_assembly)
        {
            amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, m_device_solution.begin(),
                                  m_device_solution.end(), m_solution.begin());
            amrex::Gpu::streamSynchronize();
        }
        if (m_dump_solve_pending)
        {
            // The right-hand side and solution of the first application
            // after the dumped assembly (the benchmark's check vectors).
            amrex::Vector<amrex::Real> rhs(m_num_rows);
            amrex::Vector<amrex::Real> solution(m_num_rows);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_device_rhs.begin(),
                             m_device_rhs.end(), rhs.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_device_solution.begin(),
                             m_device_solution.end(), solution.begin());
            DumpVector(m_dump_prefix + "_rhs.bin", rhs.data(), m_num_rows);
            DumpVector(m_dump_prefix + "_solution.bin", solution.data(), m_num_rows);
            amrex::Print() << "pc_mhd_block resistive direct: wrote the right-hand "
                              "side and solution of one application to "
                           << m_dump_prefix << "_{rhs,solution}.bin\n";
            m_dump_solve_pending = false;
        }
    }
#else
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block.resistive_solver = direct requires a CUDA build configured "
        "with WarpX_CUDSS=ON (NVIDIA cuDSS); chebyshev is the portable default");
#endif

    if (m_use_device_assembly)
    {
        // Unpack the device solution to every stored face (duplicated
        // faces read the same canonical unknown).
        if (warpx::fused_residual::Enabled())
        {
            // the three component scatters as one launch
            const amrex::Real* const solution_data = m_device_solution.data();
            const DirectIndexView view = m_index_view;
            for (amrex::MFIter mfi(*a_faces[0]); mfi.isValid(); ++mfi)
            {
                amrex::GpuArray<amrex::Box, 3> boxes;
                amrex::GpuArray<amrex::Array4<amrex::Real>, 3> faces;
                for (int component = 0; component < 3; ++component)
                {
                    const bool active = m_component_map[component].active;
                    boxes[component] = active ? a_faces[component]->box(mfi.index()) : amrex::Box();
                    faces[component] = active ? a_faces[component]->array(mfi)
                                              : amrex::Array4<amrex::Real>{};
                }
                amrex::ParallelFor(
                    boxes[0], boxes[1], boxes[2],
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) { faces[0](i, j, k) = solution_data[view.Index(0, i, j, k)]; },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) { faces[1](i, j, k) = solution_data[view.Index(1, i, j, k)]; },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) { faces[2](i, j, k) = solution_data[view.Index(2, i, j, k)]; });
            }
        }
        for (int component = 0; component < 3 && !warpx::fused_residual::Enabled(); ++component)
        {
            if (!m_component_map[component].active)
            {
                continue;
            }
            direct_detail::ScatterComponentSolution(m_index_view, component,
                                                    m_device_solution.data(),
                                                    *a_faces[component]);
        }
        if (m_host_sync_after_solve)
        {
            // Legacy close of the application; the scatter is stream
            // ordered before every consumer, so the synchronization only
            // keeps the host from running ahead
            // (implicit_mhd.resistive_direct_solve_host_sync = 0 skips it).
            amrex::Gpu::streamSynchronize();
        }
        return;
    }

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
