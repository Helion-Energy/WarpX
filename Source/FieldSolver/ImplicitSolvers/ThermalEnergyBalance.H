/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_ENERGY_BALANCE_H_
#define WARPX_THERMAL_ENERGY_BALANCE_H_
#include "EulerianThermalStage.H"
namespace warpx::thermal {
// Physical joules over the full step, positive into electron internal energy.
// Caller must pass the SAME U/context as a fresh successful residual, with
// no intervening residual/probe or context refresh. The stage scratch is not
// a protected successful-state snapshot. This receipt reads those emitted
// fluxes/source; it never evaluates callbacks or commits a diagnostic counter.
struct ThermalEnergyBalance {
    amrex::Real initial=0, final=0, source=0, conduction=0, advection=0,
        compression=0, defect=0, absolute_defect=0;
};
ThermalEnergyBalance MeasureThermalEnergyBalance(
    const EulerianThermalStage& stage, const amrex::MultiFab& stage_energy);
} // namespace warpx::thermal
#endif
