/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_RZ_CURRENT_FILTER_H_
#define WARPX_RZ_CURRENT_FILTER_H_
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <AMReX_Array.H>

/** Three coefficients (left, center, right) of a continuity-compatible
 * radial current-filter row. Shared by deposition and its work transpose.
 * The normal-current stencil is I-H D; transverse currents use I-D H.
 * Units pi*dr are factored out of the cylindrical deposition volumes.
 */
struct RZCurrentFilter
{
    int axis, last_cell;
    amrex::Real axis_volume;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::GpuArray<amrex::Real,3> row (int i, int component) const
    {
        using R = amrex::Real;
        if (component == 0) {
            if (i < axis || i > last_cell) { return {R(0),R(1),R(0)}; }
            R const x=i-axis, w0=i==axis ? axis_volume : R(2)*x;
            R const w1=R(2)*(x+R(1)), a=R(2)*x+R(1);
            R const conductance=i==axis ? amrex::min(R(0.125)*(w0+w1),w0)
                                       : R(0.125)*(w0+w1);
            return {i==axis ? R(0) : conductance*(a-R(2))/(a*w0),
                    R(1)-conductance*(R(1)/w0+R(1)/w1),
                    conductance*(a+R(2))/(a*w1)};
        }
        if (i < axis || i > last_cell+1) { return {R(0),R(1),R(0)}; }
        R const x=i-axis, w=i==axis ? axis_volume : R(2)*x;
        R left=i==axis ? R(0) : R(0.125)*(w+(i==axis+1 ? axis_volume : R(2)*(x-R(1))));
        R right=i==last_cell+1 ? R(0) : R(0.125)*(w+R(2)*(x+R(1)));
        if (i==axis) { right=amrex::min(right,axis_volume); }
        if (i==axis+1) { left=amrex::min(left,axis_volume); }
        return {left/w,R(1)-(left+right)/w,right/w};
    }
};
#endif
