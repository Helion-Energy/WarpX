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

#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>

using namespace amrex;

namespace warpx::circuit
{

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
    const auto owner = Bz.OwnerMask(geom.periodicity());

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<double> reduce_data(reduce_op);
    for (MFIter mfi(Bz, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box tb = mfi.tilebox();
        const auto bz = Bz.const_array(mfi);
        const auto msk = owner->const_array(mfi);
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
    const auto owner = J_theta.OwnerMask(geom.periodicity());

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<double> reduce_data(reduce_op);
    for (MFIter mfi(J_theta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box tb = mfi.tilebox(amrex::IntVect(1));
        const auto a = A_theta.const_array(mfi);
        const auto jt = J_theta.const_array(mfi);
        const auto msk = owner->const_array(mfi);
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

} // namespace warpx::circuit
