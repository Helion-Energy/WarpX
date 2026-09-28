/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_BORIS_GATHER_FEEDBACK_H_
#define WARPX_IMPLICIT_BORIS_GATHER_FEEDBACK_H_

#include "ImplicitBorisLocalResponse.H"
#include "Particles/ShapeFactors.H"
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <cstdint>
#include <limits>

namespace warpx::particles
{
using ImplicitGatherGradient = amrex::GpuArray<amrex::GpuArray<double, 3>, 6>;

/** Spatial derivative of the actual nodal cubic MC gather, RZ m=0.
 * The six supplied Cartesian fields are the already consumed native gather.
 * Reads the same allocated support; does not manufacture physical images.
 */
AMREX_GPU_HOST_DEVICE inline bool NativeRZCubicGatherGradient (
    amrex::GpuArray<double, 3> const& point,
    amrex::GpuArray<amrex::Array4<amrex::Real const>, 6> const& fields,
    amrex::GpuArray<double, 6> const& gathered,
    amrex::XDim3 const inverse, amrex::XDim3 const origin, amrex::Dim3 const lower,
    ImplicitGatherGradient& output) noexcept
{
#if defined(WARPX_DIM_RZ)
    double const r = std::sqrt(point[0]*point[0] + point[1]*point[1]);
    if (!(r > 0.) || !std::isfinite(r) || !std::isfinite(point[2]) ||
        !(inverse.x > 0.) || !(inverse.z > 0.) ||
        !std::isfinite(inverse.x) || !std::isfinite(inverse.z)) { return false; }
    double const qr = (r-origin.x)*inverse.x, qz = (point[2]-origin.z)*inverse.z;
    double const limit = double(std::numeric_limits<int>::max()/8);
    if (!(qr >= 1.) || !(qz >= 1.) || !(qr <= limit) || !(qz <= limit)) { return false; }
    double sr[4]{}, sz[4]{};
    int const ir = Compute_shape_factor<3>{}(sr,qr);
    int const iz = Compute_shape_factor<3>{}(sz,qz);
    std::int64_t const ii = std::int64_t(lower.x)+ir, jj = std::int64_t(lower.y)+iz;
    if (ii < std::numeric_limits<int>::min() || jj < std::numeric_limits<int>::min() ||
        ii+3 > std::numeric_limits<int>::max() || jj+3 > std::numeric_limits<int>::max()) { return false; }
    int const i0 = int(ii), j0 = int(jj);
    // Cardinal derivative S3'(q-j) = S2(q+1/2-j)-S2(q-1/2-j).
    double rp[3]{}, rm[3]{}, zp[3]{}, zm[3]{};
    int const rp0 = Compute_shape_factor<2>{}(rp,qr+.5);
    int const rm0 = Compute_shape_factor<2>{}(rm,qr-.5);
    int const zp0 = Compute_shape_factor<2>{}(zp,qz+.5);
    int const zm0 = Compute_shape_factor<2>{}(zm,qz-.5);
    double dr[4]{}, dz[4]{};
    for (int k=0;k<4;++k) {
        int const a=ir+k-rp0, b=ir+k-rm0, c=iz+k-zp0, d=iz+k-zm0;
        dr[k]=(a>=0&&a<3?rp[a]:0.)-(b>=0&&b<3?rm[b]:0.);
        dz[k]=(c>=0&&c<3?zp[c]:0.)-(d>=0&&d<3?zm[d]:0.);
    }
    double radial[6]{}, axial[6]{};
    for (int c=0;c<6;++c) {
        if (!std::isfinite(gathered[c]) || !fields[c].contains(i0,j0,0) ||
            !fields[c].contains(i0+3,j0+3,0)) { return false; }
        // Sum derivative weights is zero. Anchoring avoids cancellation from
        // a uniform imposed field while preserving every spatial increment.
        double const reference=fields[c](i0,j0,0);
        if (!std::isfinite(reference)) { return false; }
        for (int j=0;j<4;++j) for (int i=0;i<4;++i) {
            double const delta=fields[c](i0+i,j0+j,0)-reference;
            if (!std::isfinite(delta)) { return false; }
            radial[c] += dr[i]*sz[j]*delta;
            axial[c] += sr[i]*dz[j]*delta;
        }
        radial[c] *= inverse.x; axial[c] *= inverse.z;
    }
    double const co=point[0]/r, si=point[1]/r;
    ImplicitGatherGradient result{};
    for (int c=0;c<6;c+=3) {
        double const dx=co*radial[c]-si*radial[c+1];
        double const dy=si*radial[c]+co*radial[c+1];
        // dtheta = (-sin(theta) dx + cos(theta) dy)/r.
        result[c]   = {co*dx+si*gathered[c+1]/r,
                       si*dx-co*gathered[c+1]/r,
                       co*axial[c]-si*axial[c+1]};
        result[c+1] = {co*dy-si*gathered[c]/r,
                       si*dy+co*gathered[c]/r,
                       si*axial[c]+co*axial[c+1]};
        result[c+2] = {co*radial[c+2],si*radial[c+2],axial[c+2]};
    }
    for (auto const& row:result) for (double v:row) if (!std::isfinite(v)) { return false; }
    output=result;return true;
#else
    amrex::ignore_unused(point,fields,gathered,inverse,origin,lower,output);
    return false;
#endif
}

/** Differentiate the converged midpoint, retaining the local Boris/gamma law.
 * Vnew=(I-h V G)^-1 V, with h=dt/2. Both E and B columns use the same solve.
 * In/out is committed only after every column and the solve residual pass.
 */
AMREX_GPU_HOST_DEVICE inline bool ApplyImplicitBorisGatherFeedback (
    double dt, ImplicitGatherGradient const& gradient,
    ImplicitBorisLocalResponse& response) noexcept
{
    if (!(dt>0.) || !std::isfinite(dt)) { return false; }
    for (auto const& row:gradient) for (double v:row) if (!std::isfinite(v)) { return false; }
    for (int c=0;c<6;++c) for (int i=0;i<3;++i) {
        if (!std::isfinite(response.midpoint_momentum[c][i]) ||
            !std::isfinite(response.midpoint_velocity[c][i]) ||
            !std::isfinite(response.endpoint_velocity[c][i])) { return false; }
    }
    bool uniform=true;
    for (auto const& row:gradient) for (double v:row) uniform=uniform && v==0.;
    if (uniform) { return true; } // Preserve the literal held response exactly.
    double const h=.5*dt;
    double original[3][3]{}, matrix[3][3]{}, right[3][6]{};
    for (int i=0;i<3;++i) {
        for (int j=0;j<3;++j) {
            double value=0.;
            for (int c=0;c<6;++c) value += response.midpoint_velocity[c][i]*gradient[c][j];
            original[i][j]=matrix[i][j]=(i==j?1.:0.)-h*value;
            if (!std::isfinite(matrix[i][j])) { return false; }
        }
        for (int c=0;c<6;++c) right[i][c]=response.midpoint_velocity[c][i];
    }
    for (int k=0;k<3;++k) {
        int pivot=k;
        for (int i=k+1;i<3;++i) if (std::abs(matrix[i][k])>std::abs(matrix[pivot][k])) { pivot=i; }
        if (matrix[pivot][k]==0. || !std::isfinite(matrix[pivot][k])) { return false; }
        if (pivot!=k) {
            for (int j=0;j<3;++j) { double const v=matrix[k][j];matrix[k][j]=matrix[pivot][j];matrix[pivot][j]=v; }
            for (int c=0;c<6;++c) { double const v=right[k][c];right[k][c]=right[pivot][c];right[pivot][c]=v; }
        }
        for (int i=k+1;i<3;++i) {
            double const factor=matrix[i][k]/matrix[k][k];
            if (!std::isfinite(factor)) { return false; }
            for (int j=k+1;j<3;++j) matrix[i][j]-=factor*matrix[k][j];
            for (int c=0;c<6;++c) right[i][c]-=factor*right[k][c];
            matrix[i][k]=0.;
        }
    }
    ImplicitBorisLocalResponse result{};
    for (int c=0;c<6;++c) for (int i=2;i>=0;--i) {
        double v=right[i][c];
        for (int j=i+1;j<3;++j) v-=matrix[i][j]*result.midpoint_velocity[c][j];
        v/=matrix[i][i];
        if (!std::isfinite(v)) { return false; }
        result.midpoint_velocity[c][i]=v;
    }
    // Fewer than 128 scalar operations per solved column, including formation
    // and residual reconstruction. This validates arithmetic, not physics.
    double constexpr u=std::numeric_limits<double>::epsilon()/2.;
    double constexpr gamma=128.*u/(1.-128.*u);
    for (int c=0;c<6;++c) {
        for (int i=0;i<3;++i) {
            double residual=-response.midpoint_velocity[c][i];
            double scale=std::abs(response.midpoint_velocity[c][i]);
            for (int j=0;j<3;++j) {
                double const term=original[i][j]*result.midpoint_velocity[c][j];
                residual+=term;scale+=std::abs(term);
            }
            double const bound=gamma*scale+128.*std::numeric_limits<double>::denorm_min();
            if (!std::isfinite(residual) || !std::isfinite(bound) || std::abs(residual)>bound) { return false; }
        }
        double effective[6]{};
        for (int f=0;f<6;++f) {
            double value=0.;
            for (int i=0;i<3;++i) value+=gradient[f][i]*result.midpoint_velocity[c][i];
            effective[f]=(f==c?1.:0.)+h*value;
        }
        for (int i=0;i<3;++i) for (int f=0;f<6;++f) {
            result.midpoint_momentum[c][i]+=response.midpoint_momentum[f][i]*effective[f];
            result.endpoint_velocity[c][i]+=response.endpoint_velocity[f][i]*effective[f];
        }
    }
    for (int c=0;c<6;++c) for (int i=0;i<3;++i) {
        if (!std::isfinite(result.midpoint_momentum[c][i]) ||
            !std::isfinite(result.endpoint_velocity[c][i])) { return false; }
    }
    response=result;return true;
}
} // namespace warpx::particles
#endif
