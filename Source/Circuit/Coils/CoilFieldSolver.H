/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_COILFIELDSOLVER_H_
#define WARPX_CIRCUIT_COILFIELDSOLVER_H_

#include "CoilSet.H"

namespace warpx::circuit
{
    /** Fill the registered external vector potential <field_name>_Aext of a
     * coil with the coil's own field at I_ref, evaluated from the ring
     * kernel on the grid (the "magnetostatic external Green's solver" for
     * filamentary coils: a direct analytic evaluation -- free space needs
     * no elliptic solve and no FFT).
     *
     * RZ (m = 0): A_theta on its (node r, node z) staggering from the
     * quarter-offset filament with the Yee parameter clip -- the discrete
     * convention whose disk-flux self-inductance the circuit port must use;
     * A_r = A_z = 0. Covers the valid region plus the domain-ghost growth
     * (clipped below the axis) so the discrete curl and the split-field
     * registers inherit exact wall-ring ghosts.
     *
     * 3D: A_x = -A_theta * y/rho, A_y = A_theta * x/rho (rho = sqrt(x^2+y^2))
     * at each component's own staggering; the filament is used at its
     * nominal position (the quarter offset is an RZ-mesh policy) with the
     * same parameter clip keeping near-filament values finite.
     *
     * The caller re-runs CalculateExternalCurlA(field_name) and
     * UpdateHybridExternalFields afterwards. */
    void FillCoilUnitField (const Coil& coil);

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_COILFIELDSOLVER_H_
