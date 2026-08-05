/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_WAVE_OP_RZ_H_
#define MHD_WAVE_OP_RZ_H_

#include <AMReX_Array.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <cmath>

#if defined(WARPX_DIM_RZ)

/**
 * \brief Constant-coefficient cell-centered ideal-MHD wave operator in
 * cylindrical (m = 0) coordinates with an axial reference field.
 *
 * This is the cylindrical port of MHDWaveOp for the conservative-form
 * theta-implicit MHD recast. Acting on the physical momentum components
 * M = (M_r, M_t, M_z) it applies
 *
 *     L M = M - acoustic_scale grad(div(M)) + magnetic_scale W_B M,
 *
 * with the constant reference field B = B0 zhat. Evaluating the
 * coordinate-free wave term W_B M = B x curl(curl(M x B)) in cylindrical
 * m = 0 components and writing a_s = acoustic_scale,
 * w = magnetic_scale B0^2 (the axial-field wave scale), and the
 * perpendicular fast scale f = a_s + w gives
 *
 *     (L M)_r = M_r - f [ (1/r) d_r(r d_r M_r) - M_r/r^2 ] - w d_z^2 M_r
 *                   - a_s d_r d_z M_z,
 *     (L M)_t = M_t - w d_z^2 M_t,
 *     (L M)_z = M_z - a_s d_z[ (1/r) d_r(r M_r) ] - a_s d_z^2 M_z.
 *
 * The discrete forms follow the residual's cylindrical stencil conventions
 * in ThetaImplicitMHD.cpp: r-weighted face-flux differences for the
 * (1/r) d_r(r .) divergences, so the axis face (r = 0) drops from the
 * radial flux exactly, plus the cell-centered geometric hoop term
 * -M_r/r^2. The mixed second derivatives are centered and clamp their
 * reads to the domain: at the axis the clamped read combined with the |r|
 * weight reproduces the odd m = 0 parity of M_r exactly, while at the
 * remaining physical boundaries the clamp is a zero-gradient,
 * preconditioner-only approximation (the nonlinear residual keeps the true
 * boundary coupling). Everything is rediscretized on every multigrid
 * level. The inherited scalar AB coefficients are identity coefficients
 * used only by AMReX's hierarchy and boundary machinery.
 *
 * The smoother keeps MHDWaveOp's weighted per-cell block-Jacobi pattern;
 * because the mixed second derivatives have no same-cell entry and the
 * reference field is axial, the per-cell 3x3 block is diagonal and its
 * inverse is applied elementwise.
 */
class MHDWaveOpRZ : public amrex::MLABecLaplacian
{
    public:
        using RT = amrex::Real;

        MHDWaveOpRZ (const amrex::Vector<amrex::Geometry>& geometry,
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
                         const RT axial_magnetic_field)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(acoustic_scale) && acoustic_scale >= 0.0,
                "MHDWaveOpRZ acoustic_scale must be finite and non-negative");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(magnetic_scale) && magnetic_scale >= 0.0,
                "MHDWaveOpRZ magnetic_scale must be finite and non-negative");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(axial_magnetic_field),
                "MHDWaveOpRZ axial magnetic field must be finite");

            m_acoustic_scale = acoustic_scale;
            m_axial_wave_scale =
                magnetic_scale * axial_magnetic_field * axial_magnetic_field;
        }

        /**
         * \brief Set the weighted block-Jacobi relaxation factor.
         */
        void
        setRelaxation (const RT relaxation)
        {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(relaxation) && relaxation > 0.0 && relaxation <= 0.5,
                "MHDWaveOpRZ relaxation must be finite and in (0, 0.5]");
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
            BL_PROFILE("MHDWaveOpRZ::Fapply()");

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                input.nComp() >= 3 && output.nComp() >= 3,
                "MHDWaveOpRZ requires three input and output components");

            const amrex::Geometry& geometry = Geom(amr_level, mg_level);
            const amrex::Box domain = geometry.Domain();
            const int domain_radial_low = domain.smallEnd(0);
            const int domain_radial_high = domain.bigEnd(0);
            const int domain_axial_low = domain.smallEnd(1);
            const int domain_axial_high = domain.bigEnd(1);
            const bool axial_periodic = geometry.isPeriodic(1);
            const RT radial_lower = static_cast<RT>(geometry.ProbLo(0));
            const RT radial_cell_size = static_cast<RT>(geometry.CellSize(0));
            const RT axial_cell_size = static_cast<RT>(geometry.CellSize(1));
            const RT inverse_dr2 = 1.0 / (radial_cell_size * radial_cell_size);
            const RT inverse_dz2 = 1.0 / (axial_cell_size * axial_cell_size);
            const RT inverse_4drdz = 0.25 / (radial_cell_size * axial_cell_size);
            const RT acoustic_scale = m_acoustic_scale;
            const RT axial_wave_scale = m_axial_wave_scale;
            const RT fast_scale = acoustic_scale + axial_wave_scale;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(output, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& box = mfi.tilebox();
                const auto in = input.const_array(mfi);
                const auto out = output.array(mfi);

                amrex::ParallelFor(
                    box,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        const RT radius_center =
                            radial_lower + (static_cast<RT>(i) + RT(0.5)) * radial_cell_size;
                        const RT radius_face_low =
                            radial_lower + static_cast<RT>(i) * radial_cell_size;
                        const RT radius_face_high =
                            radial_lower + (static_cast<RT>(i) + RT(1.0)) * radial_cell_size;

                        // Mixed-derivative reads clamp to the domain (see the
                        // class documentation); pure second derivatives use
                        // the boundary-condition-filled ghosts instead.
                        const int i_low = amrex::max(i - 1, domain_radial_low);
                        const int i_high = amrex::min(i + 1, domain_radial_high);
                        int j_low = j - 1;
                        int j_high = j + 1;
                        if (!axial_periodic)
                        {
                            j_low = amrex::max(j_low, domain_axial_low);
                            j_high = amrex::min(j_high, domain_axial_high);
                        }

                        // (1/r) d_r(r d_r M_r) - M_r/r^2: r-weighted face
                        // fluxes (the axis face has r = 0 and drops) plus the
                        // cell-centered hoop term.
                        const RT radial_flux =
                            (radius_face_high * (in(i + 1, j, k, 0) - in(i, j, k, 0)) -
                             radius_face_low * (in(i, j, k, 0) - in(i - 1, j, k, 0))) *
                                inverse_dr2 / radius_center -
                            in(i, j, k, 0) / (radius_center * radius_center);
                        const RT axial_second_r =
                            (in(i, j + 1, k, 0) - 2.0 * in(i, j, k, 0) + in(i, j - 1, k, 0)) *
                            inverse_dz2;
                        // d_r d_z M_z (the gradient of a scalar carries no
                        // cylindrical metric).
                        const RT mixed_axial_gradient =
                            ((in(i_high, j_high, k, 2) - in(i_high, j_low, k, 2)) -
                             (in(i_low, j_high, k, 2) - in(i_low, j_low, k, 2))) *
                            inverse_4drdz;
                        out(i, j, k, 0) = in(i, j, k, 0) - fast_scale * radial_flux -
                                          axial_wave_scale * axial_second_r -
                                          acoustic_scale * mixed_axial_gradient;

                        const RT axial_second_t =
                            (in(i, j + 1, k, 1) - 2.0 * in(i, j, k, 1) + in(i, j - 1, k, 1)) *
                            inverse_dz2;
                        out(i, j, k, 1) = in(i, j, k, 1) - axial_wave_scale * axial_second_t;

                        // d_z[(1/r) d_r(r M_r)]: |r|-weighted cell values so
                        // the clamped axis read reproduces the odd parity of
                        // M_r exactly.
                        const RT weight_high = std::abs(
                            radial_lower + (static_cast<RT>(i) + RT(1.5)) * radial_cell_size);
                        const RT weight_low = std::abs(
                            radial_lower + (static_cast<RT>(i) - RT(0.5)) * radial_cell_size);
                        const RT mixed_radial_divergence =
                            (weight_high * (in(i_high, j_high, k, 0) - in(i_high, j_low, k, 0)) -
                             weight_low * (in(i_low, j_high, k, 0) - in(i_low, j_low, k, 0))) *
                            inverse_4drdz / radius_center;
                        const RT axial_second_z =
                            (in(i, j + 1, k, 2) - 2.0 * in(i, j, k, 2) + in(i, j - 1, k, 2)) *
                            inverse_dz2;
                        out(i, j, k, 2) =
                            in(i, j, k, 2) -
                            acoustic_scale * (mixed_radial_divergence + axial_second_z);
                    });
            }
        }

        void
        Fsmooth (int amr_level, int mg_level, amrex::MultiFab& solution,
                 const amrex::MultiFab& right_hand_side, int red_black) const override
        {
            BL_PROFILE("MHDWaveOpRZ::Fsmooth()");
            amrex::ignore_unused(red_black);

            amrex::MultiFab operator_action(solution.boxArray(), solution.DistributionMap(), 3, 0);
            Fapply(amr_level, mg_level, operator_action, solution);

            const amrex::Geometry& geometry = Geom(amr_level, mg_level);
            const RT radial_lower = static_cast<RT>(geometry.ProbLo(0));
            const RT radial_cell_size = static_cast<RT>(geometry.CellSize(0));
            const RT inverse_dr2 = 1.0 / (radial_cell_size * radial_cell_size);
            const RT inverse_dz2 =
                1.0 / (static_cast<RT>(geometry.CellSize(1)) * geometry.CellSize(1));
            const RT acoustic_scale = m_acoustic_scale;
            const RT axial_wave_scale = m_axial_wave_scale;
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
                        const auto diagonal =
                            DiagonalBlock(i, radial_lower, radial_cell_size, inverse_dr2,
                                          inverse_dz2, acoustic_scale, axial_wave_scale);
                        for (int component = 0; component < 3; ++component)
                        {
                            const RT residual =
                                rhs(i, j, k, component) - action(i, j, k, component);
                            sol(i, j, k, component) +=
                                relaxation * residual / diagonal[component];
                        }
                    });
            }
        }

        void
        normalize (int amr_level, int mg_level, amrex::MultiFab& multifab) const override
        {
            BL_PROFILE("MHDWaveOpRZ::normalize()");

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                multifab.nComp() >= 3, "MHDWaveOpRZ normalize requires three components");

            const amrex::Geometry& geometry = Geom(amr_level, mg_level);
            const RT radial_lower = static_cast<RT>(geometry.ProbLo(0));
            const RT radial_cell_size = static_cast<RT>(geometry.CellSize(0));
            const RT inverse_dr2 = 1.0 / (radial_cell_size * radial_cell_size);
            const RT inverse_dz2 =
                1.0 / (static_cast<RT>(geometry.CellSize(1)) * geometry.CellSize(1));
            const RT acoustic_scale = m_acoustic_scale;
            const RT axial_wave_scale = m_axial_wave_scale;

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
                        const auto diagonal =
                            DiagonalBlock(i, radial_lower, radial_cell_size, inverse_dr2,
                                          inverse_dz2, acoustic_scale, axial_wave_scale);
                        for (int component = 0; component < 3; ++component)
                        {
                            values(i, j, k, component) /= diagonal[component];
                        }
                    });
            }
        }

    private:
        RT m_acoustic_scale = 0.0;
        RT m_axial_wave_scale = 0.0;
        RT m_relaxation = 0.5;

        /**
         * \brief Same-cell diagonal of the operator at radial index i.
         *
         * The block-Jacobi 3x3 cell block of MHDWaveOp degenerates to this
         * diagonal for an axial reference field: the mixed second derivatives
         * contribute no same-cell entry.
         */
        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static amrex::GpuArray<RT, 3>
        DiagonalBlock (const int i, const RT radial_lower, const RT radial_cell_size,
                       const RT inverse_dr2, const RT inverse_dz2, const RT acoustic_scale,
                       const RT axial_wave_scale) noexcept
        {
            const RT radius_center =
                radial_lower + (static_cast<RT>(i) + RT(0.5)) * radial_cell_size;
            const RT radius_face_low = radial_lower + static_cast<RT>(i) * radial_cell_size;
            const RT radius_face_high =
                radial_lower + (static_cast<RT>(i) + RT(1.0)) * radial_cell_size;
            const RT fast_scale = acoustic_scale + axial_wave_scale;
            return {1.0 +
                        fast_scale * ((radius_face_high + radius_face_low) * inverse_dr2 /
                                          radius_center +
                                      1.0 / (radius_center * radius_center)) +
                        2.0 * axial_wave_scale * inverse_dz2,
                    1.0 + 2.0 * axial_wave_scale * inverse_dz2,
                    1.0 + 2.0 * acoustic_scale * inverse_dz2};
        }
};

#endif // WARPX_DIM_RZ

#endif // MHD_WAVE_OP_RZ_H_
