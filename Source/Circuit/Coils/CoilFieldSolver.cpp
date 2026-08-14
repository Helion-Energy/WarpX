/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CoilFieldSolver.H"

#include "YeeLoopKernel.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_MultiFab.H>

using namespace amrex;
using namespace amrex::literals;

namespace warpx::circuit
{

void
FillCoilUnitField (const Coil& coil)
{
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

#if defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.n_rz_azimuthal_modes == 1,
        "circuit coil fields are implemented for the m = 0 azimuthal mode only");
#elif !defined(WARPX_DIM_3D)
    WARPX_ABORT_WITH_MESSAGE(
        "circuit coil fields are implemented in RZ and 3D geometry only");
#endif

    const std::string Aext_field = coil.field_name + std::string{"_Aext"};

    const double coil_r = coil.r;
    const double coil_z = coil.z;
    const double amps = coil.I_ref * coil.n_turns;

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        const auto& geom = warpx.Geom(lev);
        const auto plo = geom.ProbLoArray();
        const auto dx = geom.CellSizeArray();

        // Same domain-ghost coverage policy as the external-field
        // accumulation: grow into the domain ghosts (they feed the exact
        // ghost curls), except below the axis and with embedded boundaries.
        for (int idir = 0; idir < 3; ++idir) {
            amrex::MultiFab& mf = *warpx.m_fields.get(Aext_field, Direction{idir}, lev);
            const amrex::IntVect ngrow =
                EB::enabled() ? amrex::IntVect(0) : mf.nGrowVect();
            amrex::Box grow_region = geom.Domain();
            grow_region.grow(ngrow);
#if defined(WARPX_DIM_RZ)
            if (geom.ProbLo(0) == 0.0_rt) {
                grow_region.setSmall(0, geom.Domain().smallEnd(0));
            }
#endif
            const amrex::Box allowed = amrex::convert(grow_region, mf.ixType());
            const auto type = mf.ixType();
            amrex::GpuArray<amrex::Real, 3> shift{0.5_rt, 0.5_rt, 0.5_rt};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                shift[d] = type.nodeCentered(d) ? 0.0_rt : 0.5_rt;
            }

#if defined(WARPX_DIM_RZ)
            double r_off, z_off;
            QuarterOffset(coil_r, coil_z, dx[0], dx[1], r_off, z_off);
            const bool is_theta = (idir == 1);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box tb =
                    mfi.tilebox(mf.ixType().toIntVect(), ngrow) & allowed;
                const Array4<Real> arr = mf.array(mfi);
#if defined(WARPX_DIM_RZ)
                amrex::ParallelFor(tb,
                    [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
                    {
                        if (!is_theta) { arr(i, j, 0, 0) = 0.0_rt; return; }
                        const double r = plo[0] + (i + shift[0]) * dx[0];
                        const double z = plo[1] + (j + shift[1]) * dx[1];
                        arr(i, j, 0, 0) = static_cast<Real>(
                            amps * YeeLoopATheta(r, z, r_off, z_off));
                    });
#else
                amrex::ParallelFor(tb,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if (idir == 2) { arr(i, j, k) = 0.0_rt; return; }
                        const double x = plo[0] + (i + shift[0]) * dx[0];
                        const double y = plo[1] + (j + shift[1]) * dx[1];
                        const double z = plo[2] + (k + shift[2]) * dx[2];
                        const double rho = std::sqrt(x * x + y * y);
                        if (rho <= 0.0) { arr(i, j, k) = 0.0_rt; return; }
                        const double a_theta =
                            amps * YeeLoopATheta(rho, z, coil_r, coil_z);
                        arr(i, j, k) = static_cast<Real>(
                            (idir == 0) ? -a_theta * y / rho
                                        :  a_theta * x / rho);
                    });
#endif
            }
        }
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace warpx::circuit
