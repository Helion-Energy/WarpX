/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_GATHER_SCALARFIELDGATHER_H_
#define WARPX_PARTICLES_GATHER_SCALARFIELDGATHER_H_

#include "Particles/ShapeFactors.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

/** @file
 *
 * Shape-factor gather of a cell- or node-centered scalar mesh field onto a
 * particle position.
 *
 * The index arithmetic and shape-factor evaluation here deliberately mirror
 * `doChargeDepositionShapeN` in `Particles/Deposition/ChargeDeposition.H`. Any
 * quantity that is both gathered by this routine and deposited by that one is
 * then interpolated with the same weights, which is what makes the pair
 * conservative: the sum over the mesh of a deposited quantity is recovered
 * exactly by the gather. Do not "improve" one without the other.
 *
 * Cartesian geometries only. The cylindrical and spherical geometries need the
 * radial volume factors that `ApplyInverseVolumeScalingToChargeDensity` applies
 * to rho, so callers must reject them up front; this routine returns 0 there
 * rather than silently interpolating an unscaled field.
 */

/**
 * \brief Gather a scalar mesh field at a particle position (compile-time order).
 *
 * \tparam depos_order particle shape order
 * \param xp,yp,zp   particle position coordinates
 * \param arr        Array4 of the scalar field, either full array or tile
 * \param arr_type   IndexType (NODE/CELL per direction) of the scalar field
 * \param dinv       3D cell size inverse
 * \param xyzmin     lower bounds of the domain
 * \param lo         index lower bounds of the domain
 * \return the field interpolated to the particle position
 */
template <int depos_order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real doGatherScalarShapeN (
    [[maybe_unused]] const amrex::ParticleReal xp,
    [[maybe_unused]] const amrex::ParticleReal yp,
    [[maybe_unused]] const amrex::ParticleReal zp,
    amrex::Array4<amrex::Real const> const& arr,
    [[maybe_unused]] const amrex::IntVect arr_type,
    [[maybe_unused]] const amrex::XDim3 & dinv,
    [[maybe_unused]] const amrex::XDim3 & xyzmin,
    [[maybe_unused]] const amrex::Dim3 lo)
{
    using namespace amrex::literals;

    [[maybe_unused]] constexpr int NODE = amrex::IndexType::NODE;
    [[maybe_unused]] constexpr int CELL = amrex::IndexType::CELL;

    [[maybe_unused]] Compute_shape_factor< depos_order > const compute_shape_factor;

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
    // x direction: particle position in grid coordinates
    const amrex::Real x = (xp - xyzmin.x)*dinv.x;
    amrex::Real sx[depos_order + 1] = {0._rt};
    int i = 0;
    if (arr_type[0] == NODE) {
        i = compute_shape_factor(sx, x);
    } else if (arr_type[0] == CELL) {
        i = compute_shape_factor(sx, x - 0.5_rt);
    }
#endif

#if defined(WARPX_DIM_3D)
    // y direction
    const amrex::Real y = (yp - xyzmin.y)*dinv.y;
    amrex::Real sy[depos_order + 1] = {0._rt};
    int j = 0;
    if (arr_type[1] == NODE) {
        j = compute_shape_factor(sy, y);
    } else if (arr_type[1] == CELL) {
        j = compute_shape_factor(sy, y - 0.5_rt);
    }
#endif

#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z)
    // z direction
    const amrex::Real z = (zp - xyzmin.z)*dinv.z;
    amrex::Real sz[depos_order + 1] = {0._rt};
    int k = 0;
    if (arr_type[WARPX_ZINDEX] == NODE) {
        k = compute_shape_factor(sz, z);
    } else if (arr_type[WARPX_ZINDEX] == CELL) {
        k = compute_shape_factor(sz, z - 0.5_rt);
    }
#endif

    amrex::Real val = 0._rt;

#if defined(WARPX_DIM_1D_Z)
    for (int iz=0; iz<=depos_order; iz++){
        val += sz[iz]*arr(lo.x+k+iz, 0, 0, 0);
    }
#elif defined(WARPX_DIM_XZ)
    for (int iz=0; iz<=depos_order; iz++){
        for (int ix=0; ix<=depos_order; ix++){
            val += sx[ix]*sz[iz]*arr(lo.x+i+ix, lo.y+k+iz, 0, 0);
        }
    }
#elif defined(WARPX_DIM_3D)
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                val += sx[ix]*sy[iy]*sz[iz]*arr(lo.x+i+ix, lo.y+j+iy, lo.z+k+iz);
            }
        }
    }
#else
    // Cylindrical and spherical geometries: see the file-level note. Callers
    // must reject these geometries before reaching this point.
    amrex::ignore_unused(arr, lo);
#endif

    return val;
}

/**
 * \brief Gather a scalar mesh field at a particle position (runtime order).
 *
 * Dispatches on the runtime shape order, mirroring the dispatch that
 * `ablastr::particles::deposit_charge` performs for the deposition side.
 *
 * \param nox particle shape order, 1 to 4
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real doGatherScalarShapeN (
    const amrex::ParticleReal xp,
    const amrex::ParticleReal yp,
    const amrex::ParticleReal zp,
    amrex::Array4<amrex::Real const> const& arr,
    const amrex::IntVect arr_type,
    const amrex::XDim3 & dinv,
    const amrex::XDim3 & xyzmin,
    const amrex::Dim3 lo,
    const int nox)
{
    using namespace amrex::literals;

    if (nox == 1) {
        return doGatherScalarShapeN<1>(xp, yp, zp, arr, arr_type, dinv, xyzmin, lo);
    } else if (nox == 2) {
        return doGatherScalarShapeN<2>(xp, yp, zp, arr, arr_type, dinv, xyzmin, lo);
    } else if (nox == 3) {
        return doGatherScalarShapeN<3>(xp, yp, zp, arr, arr_type, dinv, xyzmin, lo);
    } else if (nox == 4) {
        return doGatherScalarShapeN<4>(xp, yp, zp, arr, arr_type, dinv, xyzmin, lo);
    }
    return 0._rt;
}

#endif // WARPX_PARTICLES_GATHER_SCALARFIELDGATHER_H_
