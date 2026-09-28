/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_TRANSPORT_STENCIL_H_
#define WARPX_THERMAL_TRANSPORT_STENCIL_H_
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace warpx::thermal {
// Low-order frozen central surrogate. This is the exact U derivative for
// reconstruction=none, entropy penalty=0, fixed physical n/u/inflow T. Higher
// reconstruction, entropy penalty and coupled d(u)/d(U) remain outer-Jv terms.
struct FrozenTransportGeometry {
    amrex::GpuArray<int, AMREX_SPACEDIM> low{}, high{}, periodic{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
    amrex::Real rlo = 0;
    amrex::Real h = 0;
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
FrozenCentralFaceWeights (FrozenTransportGeometry const& g, int d, int index,
                          amrex::Real u, amrex::Real& left,
                          amrex::Real& right) {
    left = right = amrex::Real(.5) * u;
    if (!g.periodic[d] && index == g.low[d]) {
        left = 0;
        right = u < 0 ? u : 0; // fixed inflow has zero U derivative
    } else if (!g.periodic[d] && index == g.high[d] + 1) {
        left = u > 0 ? u : 0;
        right = 0;
    }
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
FrozenCentralFaceFlux (FrozenTransportGeometry const& g, int d, int i, int j,
                       int k, amrex::Real velocity,
                       amrex::Array4<const amrex::Real> const& U) {
    amrex::Real left, right;
    FrozenCentralFaceWeights(g, d, d == 0 ? i : (d == 1 ? j : k), velocity,
                             left, right);
    // Do not read physical ghosts for an eliminated boundary coefficient.
    return (left == 0 ? 0
                      : left * U(i - (d == 0), j - (d == 1), k - (d == 2))) +
           (right == 0 ? 0 : right * U(i, j, k));
}
template <typename Emit>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EmitFrozenTransportRow (FrozenTransportGeometry const& g,
                        amrex::Array4<const amrex::Real> const& ur,
                        amrex::Array4<const amrex::Real> const& uz,
                        amrex::Real compression, int i, int j, int k,
                        Emit const& emit) {
    amrex::Real const r = g.rlo + (i + .5) * g.dx[0];
    emit(i, j, k, compression);
    for (int d = 0; d < 2; ++d) {
        for (int side = 0; side < 2; ++side) {
            int const fi = i + (d == 0 ? side : 0),
                      fj = j + (d == 1 ? side : 0);
            amrex::Real const u = d == 0 ? ur(fi, fj, k) : uz(fi, fj, k);
            amrex::Real l, rr;
            FrozenCentralFaceWeights(g, d, d == 0 ? fi : fj, u, l, rr);
            amrex::Real const area_over_volume =
                d == 0 ? (g.rlo + fi * g.dx[0]) / (r * g.dx[0]) : 1 / g.dx[1];
            amrex::Real const factor =
                (side == 0 ? -g.h : g.h) * area_over_volume;
            if (l != 0)
                emit(fi - (d == 0), fj - (d == 1), k, factor * l);
            if (rr != 0)
                emit(fi, fj, k, factor * rr);
        }
    }
}
#if defined(WARPX_DIM_3D)
// Cartesian rows include every x/y/z face with area/volume = 1/dx_d.
template <typename Emit>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
EmitFrozenTransportRow3D (
    FrozenTransportGeometry const& g,
    amrex::GpuArray<amrex::Array4<const amrex::Real>, 3> const& velocity,
    amrex::Real compression, int i, int j, int k, Emit const& emit) {
    emit(i, j, k, compression);
    for (int d = 0; d < 3; ++d)
        for (int side = 0; side < 2; ++side) {
            int face[3] = {i, j, k};
            face[d] += side;
            amrex::Real left, right;
            FrozenCentralFaceWeights(g, d, face[d],
                                     velocity[d](face[0], face[1], face[2]),
                                     left, right);
            amrex::Real const factor = (side == 0 ? -g.h : g.h) / g.dx[d];
            if (right != 0) {
                emit(face[0], face[1], face[2], factor * right);
            }
            face[d] -= 1;
            if (left != 0) {
                emit(face[0], face[1], face[2], factor * left);
            }
        }
}
#endif
} // namespace warpx::thermal
#endif
