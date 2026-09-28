/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_SHAPE_DIFFERENCE_H_
#define WARPX_ESIRKEPOV_SHAPE_DIFFERENCE_H_
#define WARPX_ESIRKEPOV_PRECISE_CHORD 1

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#include <cmath>

namespace warpx::particles {

/** Exact polynomial difference within one maintained B-spline support cell.
 * old_minus_new is evaluated as displacement times a divided difference.
 * No position, support index, transverse shape average, or current stencil is
 * changed. Coordinates refer to the same center used by Compute_shape_factor.
 */
template<int Order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double EsirkepovPolynomialDifferenceFromDisplacement(int slot,double a,double b,double d) noexcept
{
    if(slot<0 || slot>Order) return 0.;
    double const s=a+b,q=a*a+a*b+b*b;
    if constexpr(Order==1) return slot==0?d:-d;
    if constexpr(Order==2) {
        if(slot==0) return d*(.5-.5*s);
        if(slot==1) return d*s;
        return -d*(.5+.5*s);
    }
    if constexpr(Order==3) {
        if(slot==0) {
            double const u=1.-a,v=1.-b;
            return d*(u*u+u*v+v*v)/6.;
        }
        if(slot==1) return d*(s-.5*q);
        if(slot==2) return -d*(.5+.5*s-.5*q);
        return -d*q/6.;
    }
    if constexpr(Order==4) {
        double const t=(a+b)*(a*a+b*b);
        if(slot==0 || slot==4) {
            double const u=.5+(slot==0?-a:a),v=.5+(slot==0?-b:b);
            return (slot==0?d:-d)*(u+v)*(u*u+v*v)/24.;
        }
        if(slot==1) return d*(11.-6.*s-4.*q+4.*t)/24.;
        if(slot==2) return d*(15.*s-6.*t)/24.;
        return d*(-11.-6.*s+4.*q+4.*t)/24.;
    }
    return 0.;
}

template<int Order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double EsirkepovPolynomialDifference(int slot,double a,double b) noexcept
{
    return EsirkepovPolynomialDifferenceFromDisplacement<Order>(slot,a,b,b-a);
}

/** Stable replacement of one old_shape-new_shape subtraction.
 * This adapter accepts the caller's original support indices. Same-cell and
 * adjacent nonnegative support crossings use exact polynomial differences.
 * Other cases retain the original subtraction, including the maintained
 * negative-coordinate truncation semantics and arbitrarily long trajectories.
 * Thus this helper isolates shape-value cancellation; it does not repair
 * quantization already present in the supplied old/new coordinates.
 */
template<int Order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double EsirkepovShapeDifferenceFromDisplacement(double old_x,double new_x,int old_first,
    int new_first,int node,double old_value,double new_value,double displacement) noexcept
{
    if constexpr(Order<1 || Order>4) return old_value-new_value;
    int const old_center=old_first+Order/2,new_center=new_first+Order/2;
    if(old_first==new_first) {
        return EsirkepovPolynomialDifferenceFromDisplacement<Order>(node-old_first,
            old_x-old_center,new_x-old_center,displacement);
    }
    if(old_x>=0. && new_x>=0. && std::abs(new_center-old_center)==1) {
        double const half=(Order%2==0)?.5:0.;
        double const split=double(old_center>new_center?old_center:new_center)-half;
        double const first_displacement=split-old_x;
        return EsirkepovPolynomialDifferenceFromDisplacement<Order>(node-old_first,
                old_x-old_center,split-old_center,first_displacement)
            +EsirkepovPolynomialDifferenceFromDisplacement<Order>(node-new_first,
                split-new_center,new_x-new_center,displacement-first_displacement);
    }
    return old_value-new_value;
}
/** Original represented-endpoint API: all support and orbit semantics retained. */
template<int Order>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double EsirkepovShapeDifference(double old_x,double new_x,int old_first,
    int new_first,int node,double old_value,double new_value) noexcept
{
    return EsirkepovShapeDifferenceFromDisplacement<Order>(old_x,new_x,old_first,
        new_first,node,old_value,new_value,new_x-old_x);
}
}
#endif
