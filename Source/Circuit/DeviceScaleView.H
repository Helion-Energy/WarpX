/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DEVICE_SCALE_VIEW_H_
#define WARPX_DEVICE_SCALE_VIEW_H_
#include <AMReX_GpuQualifiers.H>
namespace warpx::circuit
{
/** Borrowed device arrays indexed by external-field number. The owner keeps
 * them alive for the entire nonlinear step and clears this view at acceptance. */
struct DeviceScaleView
{
    double const* start = nullptr;
    double const* end = nullptr;
    double t0 = 0., dt = 0.;
    AMREX_GPU_HOST_DEVICE double Slope (int field) const noexcept
    {
        return (end[field]-start[field])/dt;
    }
    AMREX_GPU_HOST_DEVICE double Value (int field, double time) const noexcept
    {
        volatile double change = Slope(field)*(time-t0);
        return start[field]+change;
    }
};
}
#endif
