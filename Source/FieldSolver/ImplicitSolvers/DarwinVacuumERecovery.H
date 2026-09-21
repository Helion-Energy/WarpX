/* This file is part of WarpX.
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DARWIN_VACUUM_E_RECOVERY_H_
#define WARPX_DARWIN_VACUUM_E_RECOVERY_H_

#include <AMReX_REAL.H>
#include <ablastr/fields/MultiFabRegister.H>

class WarpX;
class HybridPICModel;

/** Recover endpoint vacuum E_T with the native spatial curl-curl equation.
 *
 * The plasma trace and external circuit slope are fixed at the endpoint.
 * A zero-start correction solve preserves the accepted electric null modes,
 * including local charge and enclosed flux; no A derivative history is used.
 * Updates both the solver E_T and physical E_T + E_L registers and guards.
 */
void RecoverDarwinVacuumE (WarpX& warpx, HybridPICModel& hybrid,
                           ablastr::fields::VectorField const& endpoint,
                           ablastr::fields::VectorField const& accepted, amrex::Real time,
                           amrex::Real dt);

#endif
