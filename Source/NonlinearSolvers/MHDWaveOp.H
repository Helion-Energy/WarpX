/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_WAVE_OP_H_
#define MHD_WAVE_OP_H_

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <cmath>

/**
 * \brief Constant-coefficient cell-centered ideal-MHD wave operator.
 *
 * This operator acts on a three-component physical Cartesian momentum vector M:
 *
 *     L M = M - acoustic_scale grad(div(M)) + magnetic_scale W_B M,
 *
 * where B is constant and
 *
 *     W_B M = B x curl(curl(M x B))
 *           = -(B dot grad)^2 M - |B|^2 grad(div(M))
 *             + B (B dot grad) div(M)
 *             + grad((B dot grad)(B dot M)).
 *
 * The implementation rediscretizes this expression on every multigrid level with
 * centered pure and mixed second derivatives. In three dimensions, its stencil is
 * contained in the 27-cell box around a cell. The inherited scalar AB coefficients
 * are identity coefficients used only by AMReX's hierarchy and boundary machinery.
 */
class MHDWaveOp : public amrex::MLABecLaplacian
{
    public:
        using RT = amrex::Real;

        MHDWaveOp (const amrex::Vector<amrex::Geometry>& geometry,
                   const amrex::Vector<amrex::BoxArray>& grids,
                   const amrex::Vector<amrex::DistributionMapping>& distribution_maps,
                   const amrex::LPInfo& a_info = amrex::LPInfo())
            : amrex::MLABecLaplacian(geometry, grids, distribution_maps, a_info, {}, 3)
        {
            setScalars(1.0, 0.0);
            for (int amr_level = 0; amr_level < static_cast<int>(geometry.size()); ++amr_level)
            {
                setACoeffs(amr_level, 1.0);
                setBCoeffs(amr_level, 0.0);
            }
            setGaussSeidel(false);
        }

        /**
         * \brief Set the fixed physical coefficients used on every AMR and MG level.
         */
        void
        setCoefficients (const RT acoustic_scale, const RT magnetic_scale,
                         const amrex::GpuArray<RT, 3>& magnetic_field)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(acoustic_scale) && acoustic_scale >= 0.0,
                "MHDWaveOp acoustic_scale must be finite and non-negative");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(magnetic_scale) && magnetic_scale >= 0.0,
                "MHDWaveOp magnetic_scale must be finite and non-negative");
            for (int component = 0; component < 3; ++component)
            {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::isfinite(magnetic_field[component]),
                    "MHDWaveOp magnetic field must be finite");
            }

            m_acoustic_scale = acoustic_scale;
            m_magnetic_scale = magnetic_scale;
            m_magnetic_field = magnetic_field;
        }

        /**
         * \brief Set the weighted block-Jacobi relaxation factor.
         */
        void
        setRelaxation (const RT relaxation)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(relaxation) && relaxation > 0.0 && relaxation <= 0.5,
                "MHDWaveOp relaxation must be finite and in (0, 0.5]");
            m_relaxation = relaxation;
        }

        [[nodiscard]] bool
        isCrossStencil () const override
        {
            return false;
        }

        [[nodiscard]] bool
        isTensorOp () const override
        {
            return true;
        }

        // These methods contain extended __device__ lambdas and must therefore
        // be public when compiled with NVCC.
        void
        Fapply (int amr_level, int mg_level, amrex::MultiFab& output,
                const amrex::MultiFab& input) const override
        {
            BL_PROFILE("MHDWaveOp::Fapply()");

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                input.nComp() >= 3 && output.nComp() >= 3,
                "MHDWaveOp requires three input and output components");

            const auto inverse_cell_size = PhysicalInverseCellSize(Geom(amr_level, mg_level));
            const auto magnetic_field = m_magnetic_field;
            const RT magnetic_field_squared = MagneticFieldSquared();
            const RT acoustic_scale = m_acoustic_scale;
            const RT magnetic_scale = m_magnetic_scale;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& box = mfi.tilebox();
                const auto in = input.const_array(mfi);
                const auto out = output.array(mfi);

                amrex::ParallelFor(
                    box, 3,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept
                    {
                        RT gradient_divergence = 0.0;
                        for (int direction = 0; direction < 3; ++direction)
                        {
                            gradient_divergence +=
                                SecondDerivative(in, i, j, k, direction, component, direction,
                                                 inverse_cell_size);
                        }

                        RT field_directional_second_derivative = 0.0;
                        RT field_directional_divergence = 0.0;
                        RT gradient_field_directional_field_dot_momentum = 0.0;
                        for (int first_direction = 0; first_direction < 3; ++first_direction)
                        {
                            for (int second_direction = 0; second_direction < 3;
                                 ++second_direction)
                            {
                                field_directional_second_derivative +=
                                    magnetic_field[first_direction] *
                                    magnetic_field[second_direction] *
                                    SecondDerivative(in, i, j, k, component, first_direction,
                                                     second_direction, inverse_cell_size);
                                field_directional_divergence +=
                                    magnetic_field[first_direction] *
                                    SecondDerivative(in, i, j, k, second_direction,
                                                     first_direction, second_direction,
                                                     inverse_cell_size);
                                gradient_field_directional_field_dot_momentum +=
                                    magnetic_field[first_direction] *
                                    magnetic_field[second_direction] *
                                    SecondDerivative(in, i, j, k, second_direction, component,
                                                     first_direction, inverse_cell_size);
                            }
                        }

                        const RT magnetic_wave =
                            -field_directional_second_derivative -
                            magnetic_field_squared * gradient_divergence +
                            magnetic_field[component] * field_directional_divergence +
                            gradient_field_directional_field_dot_momentum;
                        out(i, j, k, component) =
                            in(i, j, k, component) -
                            acoustic_scale * gradient_divergence +
                            magnetic_scale * magnetic_wave;
                    });
            }
        }

        void
        Fsmooth (int amr_level, int mg_level, amrex::MultiFab& solution,
                 const amrex::MultiFab& right_hand_side, int red_black) const override
        {
            BL_PROFILE("MHDWaveOp::Fsmooth()");
            amrex::ignore_unused(red_black);

            amrex::MultiFab operator_action(solution.boxArray(), solution.DistributionMap(), 3, 0);
            Fapply(amr_level, mg_level, operator_action, solution);

            const auto inverse_cell_block = InverseCellBlock(amr_level, mg_level);
            const RT relaxation = m_relaxation;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(solution, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& box = mfi.tilebox();
                const auto sol = solution.array(mfi);
                const auto rhs = right_hand_side.const_array(mfi);
                const auto action = operator_action.const_array(mfi);

                amrex::ParallelFor(
                    box,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        RT residual[3];
                        for (int component = 0; component < 3; ++component)
                        {
                            residual[component] =
                                rhs(i, j, k, component) - action(i, j, k, component);
                        }
                        for (int component = 0; component < 3; ++component)
                        {
                            RT correction = 0.0;
                            for (int coupled_component = 0; coupled_component < 3;
                                 ++coupled_component)
                            {
                                correction +=
                                    inverse_cell_block[3 * component + coupled_component] *
                                    residual[coupled_component];
                            }
                            sol(i, j, k, component) += relaxation * correction;
                        }
                    });
            }
        }

        void
        normalize (int amr_level, int mg_level, amrex::MultiFab& multifab) const override
        {
            BL_PROFILE("MHDWaveOp::normalize()");

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                multifab.nComp() >= 3, "MHDWaveOp normalize requires three components");
            const auto inverse_cell_block = InverseCellBlock(amr_level, mg_level);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(multifab, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& box = mfi.tilebox();
                const auto values = multifab.array(mfi);
                amrex::ParallelFor(
                    box,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        const RT original[3] = {values(i, j, k, 0), values(i, j, k, 1),
                                                values(i, j, k, 2)};
                        for (int component = 0; component < 3; ++component)
                        {
                            RT normalized = 0.0;
                            for (int coupled_component = 0; coupled_component < 3;
                                 ++coupled_component)
                            {
                                normalized +=
                                    inverse_cell_block[3 * component + coupled_component] *
                                    original[coupled_component];
                            }
                            values(i, j, k, component) = normalized;
                        }
                    });
            }
        }

    private:
        RT m_acoustic_scale = 0.0;
        RT m_magnetic_scale = 0.0;
        RT m_relaxation = 0.5;
        amrex::GpuArray<RT, 3> m_magnetic_field = {0.0, 0.0, 0.0};

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static int
        PhysicalToLogicalDirection (const int physical_direction) noexcept
        {
#if defined(WARPX_DIM_3D)
            return physical_direction;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return physical_direction == 0 ? 0 : (physical_direction == 2 ? 1 : -1);
#elif defined(WARPX_DIM_1D_Z)
            return physical_direction == 2 ? 0 : -1;
#else
            amrex::ignore_unused(physical_direction);
            return -1;
#endif
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
        ShiftLogicalIndex (int& i, int& j, int& k, const int logical_direction,
                           const int offset) noexcept
        {
#if (AMREX_SPACEDIM == 1)
            amrex::ignore_unused(j, k);
#elif (AMREX_SPACEDIM == 2)
            amrex::ignore_unused(k);
#endif
            if (logical_direction == 0)
            {
                i += offset;
            }
#if (AMREX_SPACEDIM >= 2)
            else if (logical_direction == 1)
            {
                j += offset;
            }
#endif
#if (AMREX_SPACEDIM == 3)
            else if (logical_direction == 2)
            {
                k += offset;
            }
#endif
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static RT
        SecondDerivative (const amrex::Array4<const RT>& field, const int i, const int j,
                          const int k, const int component, const int first_direction,
                          const int second_direction,
                          const amrex::GpuArray<RT, 3>& inverse_cell_size) noexcept
        {
            const int first_logical_direction =
                PhysicalToLogicalDirection(first_direction);
            const int second_logical_direction =
                PhysicalToLogicalDirection(second_direction);
            if (first_logical_direction < 0 || second_logical_direction < 0)
            {
                return 0.0;
            }

            if (first_direction == second_direction)
            {
                int i_low = i;
                int j_low = j;
                int k_low = k;
                int i_high = i;
                int j_high = j;
                int k_high = k;
                ShiftLogicalIndex(
                    i_low, j_low, k_low, first_logical_direction, -1);
                ShiftLogicalIndex(
                    i_high, j_high, k_high, first_logical_direction, 1);
                const RT inverse_spacing = inverse_cell_size[first_direction];
                return (field(i_high, j_high, k_high, component) -
                        2.0 * field(i, j, k, component) +
                        field(i_low, j_low, k_low, component)) *
                       inverse_spacing * inverse_spacing;
            }

            int i_pp = i;
            int j_pp = j;
            int k_pp = k;
            int i_pm = i;
            int j_pm = j;
            int k_pm = k;
            int i_mp = i;
            int j_mp = j;
            int k_mp = k;
            int i_mm = i;
            int j_mm = j;
            int k_mm = k;
            ShiftLogicalIndex(i_pp, j_pp, k_pp, first_logical_direction, 1);
            ShiftLogicalIndex(i_pp, j_pp, k_pp, second_logical_direction, 1);
            ShiftLogicalIndex(i_pm, j_pm, k_pm, first_logical_direction, 1);
            ShiftLogicalIndex(i_pm, j_pm, k_pm, second_logical_direction, -1);
            ShiftLogicalIndex(i_mp, j_mp, k_mp, first_logical_direction, -1);
            ShiftLogicalIndex(i_mp, j_mp, k_mp, second_logical_direction, 1);
            ShiftLogicalIndex(i_mm, j_mm, k_mm, first_logical_direction, -1);
            ShiftLogicalIndex(i_mm, j_mm, k_mm, second_logical_direction, -1);

            return 0.25 * inverse_cell_size[first_direction] *
                   inverse_cell_size[second_direction] *
                   (field(i_pp, j_pp, k_pp, component) -
                    field(i_pm, j_pm, k_pm, component) -
                    field(i_mp, j_mp, k_mp, component) +
                    field(i_mm, j_mm, k_mm, component));
        }

        [[nodiscard]] static amrex::GpuArray<RT, 3>
        PhysicalInverseCellSize (const amrex::Geometry& geometry) noexcept
        {
            const auto inverse_cell_size = geometry.InvCellSizeArray();
#if defined(WARPX_DIM_3D)
            return {inverse_cell_size[0], inverse_cell_size[1], inverse_cell_size[2]};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return {inverse_cell_size[0], 0.0, inverse_cell_size[1]};
#elif defined(WARPX_DIM_1D_Z)
            return {0.0, 0.0, inverse_cell_size[0]};
#else
            amrex::ignore_unused(inverse_cell_size);
            return {0.0, 0.0, 0.0};
#endif
        }

        [[nodiscard]] RT
        MagneticFieldSquared () const noexcept
        {
            RT result = 0.0;
            for (int component = 0; component < 3; ++component)
            {
                result += m_magnetic_field[component] * m_magnetic_field[component];
            }
            return result;
        }

        [[nodiscard]] amrex::GpuArray<RT, 9>
        InverseCellBlock (const int amr_level, const int mg_level) const noexcept
        {
            const auto inverse_cell_size =
                PhysicalInverseCellSize(Geom(amr_level, mg_level));
            amrex::GpuArray<RT, 3> inverse_spacing_squared;
            RT field_directional_inverse_spacing_squared = 0.0;
            for (int direction = 0; direction < 3; ++direction)
            {
                inverse_spacing_squared[direction] =
                    inverse_cell_size[direction] * inverse_cell_size[direction];
                const RT field_over_spacing =
                    m_magnetic_field[direction] * inverse_cell_size[direction];
                field_directional_inverse_spacing_squared +=
                    field_over_spacing * field_over_spacing;
            }

            const RT magnetic_field_squared = MagneticFieldSquared();
            amrex::GpuArray<RT, 9> cell_block;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const RT identity = row == column ? 1.0 : 0.0;
                    const RT acoustic =
                        row == column
                            ? 2.0 * m_acoustic_scale * inverse_spacing_squared[row]
                            : 0.0;
                    const RT magnetic =
                        2.0 * m_magnetic_scale *
                        (identity *
                             (field_directional_inverse_spacing_squared +
                              magnetic_field_squared * inverse_spacing_squared[row]) -
                         m_magnetic_field[row] * m_magnetic_field[column] *
                             (inverse_spacing_squared[row] +
                              inverse_spacing_squared[column]));
                    cell_block[3 * row + column] = identity + acoustic + magnetic;
                }
            }

            const RT cofactor_00 =
                cell_block[4] * cell_block[8] - cell_block[5] * cell_block[7];
            const RT cofactor_01 =
                cell_block[2] * cell_block[7] - cell_block[1] * cell_block[8];
            const RT cofactor_02 =
                cell_block[1] * cell_block[5] - cell_block[2] * cell_block[4];
            const RT cofactor_11 =
                cell_block[0] * cell_block[8] - cell_block[2] * cell_block[6];
            const RT cofactor_12 =
                cell_block[2] * cell_block[3] - cell_block[0] * cell_block[5];
            const RT cofactor_22 =
                cell_block[0] * cell_block[4] - cell_block[1] * cell_block[3];
            const RT determinant =
                cell_block[0] * cofactor_00 + cell_block[1] * cofactor_01 +
                cell_block[2] * cofactor_02;
            AMREX_ALWAYS_ASSERT(std::isfinite(determinant) && determinant > 0.0);

            const RT inverse_determinant = 1.0 / determinant;
            return {inverse_determinant * cofactor_00,
                    inverse_determinant * cofactor_01,
                    inverse_determinant * cofactor_02,
                    inverse_determinant * cofactor_01,
                    inverse_determinant * cofactor_11,
                    inverse_determinant * cofactor_12,
                    inverse_determinant * cofactor_02,
                    inverse_determinant * cofactor_12,
                    inverse_determinant * cofactor_22};
        }
};

#endif
