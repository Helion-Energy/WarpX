/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MCCBackgroundField.H"

#include "Utils/TextMsg.H"

#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

std::string
MCCBackgroundField::fieldName (std::string const& background_name)
{
    return "n_background_" + background_name;
}

void
MCCBackgroundField::allocInit (
    ablastr::fields::MultiFabRegister& fields,
    amrex::Vector<DepletableBackgroundSpec> const& backgrounds,
    int const lev,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    amrex::IntVect const& nodal_flag,
    amrex::IntVect const& ngrow,
    amrex::Geometry const& geom
)
{
    using namespace amrex::literals;

    for (auto const& background : backgrounds)
    {
        // The depletion deposit spreads over shape/2+1 cells either side of a
        // particle and relies on those landing in allocated guard cells, so
        // reject a shape the level's rho guard cells cannot hold rather than
        // failing later inside the deposition.
        const int shape_extent = background.m_shape/2 + 1;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            ngrow.min() >= shape_extent,
            "background_shape " + std::to_string(background.m_shape) + " for background '"
            + background.m_background_name + "' needs at least " + std::to_string(shape_extent)
            + " rho guard cells, but only " + std::to_string(ngrow.min()) + " are allocated. "
            "Raise algo.particle_shape or lower background_shape.");

        auto * const mf = fields.alloc_init(
            MCCBackgroundField::fieldName(background.m_background_name),
            lev, amrex::convert(ba, nodal_flag), dm,
            /*ncomp=*/1, ngrow, 0.0_rt);

        auto const dx_lev = geom.CellSizeArray();
        const amrex::RealBox& real_box = geom.ProbDomain();
        auto const n_func = background.m_density_func;

        // The guard cells are filled as well as the valid region: the initial
        // state has to be consistent there too, since the depletion deposit
        // sums guard-cell contributions back in.
        for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& n_arr = mf->array(mfi);
            const amrex::Box& tb = mfi.growntilebox(ngrow);

            amrex::ParallelFor(tb,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    amrex::ignore_unused(i, j, k);
#if defined(WARPX_DIM_1D_Z)
                    const amrex::Real x = 0._rt;
                    const amrex::Real y = 0._rt;
                    const amrex::Real fac_z = (1._rt - nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                    const amrex::Real z = i*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ)
                    const amrex::Real fac_x = (1._rt - nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                    const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                    const amrex::Real y = 0._rt;
                    const amrex::Real fac_z = (1._rt - nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                    const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif defined(WARPX_DIM_3D)
                    const amrex::Real fac_x = (1._rt - nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                    const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                    const amrex::Real fac_y = (1._rt - nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                    const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                    const amrex::Real fac_z = (1._rt - nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                    const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#else
                    // Not reachable: the cylindrical and spherical geometries
                    // are rejected where depletion is requested.
                    const amrex::Real x = 0._rt;
                    const amrex::Real y = 0._rt;
                    const amrex::Real z = 0._rt;
#endif
                    n_arr(i,j,k) = n_func(x, y, z, 0._rt);
                });
        }
    }
}
