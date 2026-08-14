/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_FLUXPROBES_H_
#define WARPX_CIRCUIT_FLUXPROBES_H_

#include "CoilSet.H"

#include <AMReX_MultiFab.H>

#include <array>

namespace warpx::circuit
{
    /** Which linkage measurement a coil's back-EMF port uses. */
    enum class ProbeKind
    {
        none,        //!< coil is not measured (drive-only)
        disk,        //!< plasma-frame disk flux through the coil circle
        reciprocity  //!< free-space reciprocity dot with the plasma current
    };

    /** Plasma flux linkage of a coil by the DISK rule: the plasma-frame
     * B_z integrated over the coil's quarter-offset circle,
     *     lambda_phys = 2 pi Int_0^{r_off} B_z(r, z_off) r dr,
     * with the identical staircase conventions as the discrete
     * self-inductance (STRICT r_cell < r_off cell test, unclamped linear z
     * interpolation between the two bracketing node planes) -- probe and
     * port self-inductance must share the rule or the coupled system's
     * passivity is violated. Returns lambda_phys * I_ref * n_turns (the
     * linkage of the coil's reference drive, whose time derivative divided
     * by I_ref * n_turns is the port EMF in volts). Collective
     * (all-reduces the partial sums); RZ (m = 0), level 0.
     *
     * Valid with conducting walls (it measures the model's own flux
     * through the disk); use it whenever the reciprocity probe is not
     * valid.
     *
     * \param[in] coil     the measured coil
     * \param[in] Bz       plasma-frame B_z (cell r, node z), level 0
     */
    amrex::Real DiskFluxLinkage (const Coil& coil, const amrex::MultiFab& Bz);

    /** Plasma flux linkage of a coil by FREE-SPACE RECIPROCITY: the dot
     * product of the coil's unit vector potential with the plasma current,
     *     lambda = Int A_theta^(I_ref n) J_theta,plasma dV
     *            = 2 pi Sum w_r r A_theta J_theta dr dz,
     * over the nodal (node r, node z) J_theta, with trapezoid end-weights
     * in r (w = 1/2 at i = 0 and i = nr) and each shared nodal point
     * counted once (owner-masked). A carries the coil's reference
     * amp-turns, so the result is lambda_phys * I_ref * n_turns directly.
     * Exact linkage in free space; INVALID with a conducting wall between
     * plasma and coil (the image response is not in A^unit) -- pair it
     * with the Green's-function open boundary, or use the disk probe.
     * Collective; RZ (m = 0), level 0.
     *
     * \param[in] A_theta  the coil's <field_name>_Aext theta component
     *                     (node r, node z), holding the field at
     *                     I_ref * n_turns, level 0
     * \param[in] J_theta  plasma-frame J_theta (node r, node z), level 0
     */
    amrex::Real ReciprocityLinkage (const amrex::MultiFab& A_theta,
                                    const amrex::MultiFab& J_theta);

    /** The coupling power Int J . E dV over the RZ (m = 0) domain, summed
     * component-wise on matched (E/J) staggerings with owner-masked nodal
     * points and volume weights 2 pi r dr dz (r at each sample position).
     * Used for the field side of the coupling-power double entry against
     * -sum_k (ds_k/dt) lambda_k. Collective; level 0.
     *
     * \param[in] J  the three plasma-current components (E/J staggering)
     * \param[in] E  the three field components on the same staggerings
     */
    amrex::Real CouplingPowerIntegral (
        const std::array<const amrex::MultiFab*, 3>& J,
        const std::array<const amrex::MultiFab*, 3>& E);

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_FLUXPROBES_H_
