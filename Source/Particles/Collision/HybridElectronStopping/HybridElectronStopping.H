/* Copyright 2026 The WarpX Community
 * Authors: Eric Clark (Helion Energy)
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_HYBRIDELECTRONSTOPPING_H_
#define WARPX_PARTICLES_COLLISION_HYBRIDELECTRONSTOPPING_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"

#include <AMReX_REAL.H>

#include <string>

/** Fast-ion slowing down on the self-consistent hybrid electron fluid.
 *
 * Acts on one fast ion species at the Goldston & Rutherford ion-on-electron
 * slowing-down rate (Introduction to Plasma Physics, Eq. 14.12; the same
 * expression as the background_stopping electron branch, with the Coulomb
 * logarithm supplied as an input),
 *
 *     nu_s = sqrt(2) n_e q_b^2 e^2 sqrt(m_e) ln(Lambda)
 *            / (12 pi^{3/2} eps0^2 m_b (k_B T_e)^{3/2}),
 *
 * evaluated per particle from the hybrid solver's own fields: n_e from
 * rho_fp / e and T_e from the electron-energy-equation temperature field
 * (both gathered at the particle, as in hybrid_resistive_drag).
 *
 * The drag is exponential, drag-only (no diffusion), toward the local
 * electron fluid velocity u_e (the Ve_fp field the resistive-drag operator
 * also reads), applied relativistically in velocity space:
 *
 *     v -> u_e + (v - u_e) exp(-nu_s dt),    v = u / gamma.
 *
 * Energy conservation: each particle's weighted kinetic-energy loss
 * w dKE (relativistic KE = (gamma - 1) m c^2) is scatter-added, with the
 * gather-conjugate linear nodal weights and the true node control volumes,
 * into the HybridPICModel fast-ion heating staging field, which the
 * electron-energy source stage converts into a T_e increment in the SAME
 * step (drag in the collision stage, heat landed in the source stage:
 * one-step staging, no energy lost).
 *
 * Requires hybrid_pic_model.solve_electron_energy_equation = 1 (checked at
 * construction): without the T_e field there is nothing to heat --
 * stopping on a prescribed background remains background_stopping's job.
 */
class HybridElectronStopping final
    : public CollisionBase
{
public:
    explicit HybridElectronStopping (const std::string& collision_name);

    ~HybridElectronStopping () override = default;

    HybridElectronStopping ( HybridElectronStopping const &)             = delete;
    HybridElectronStopping& operator= ( HybridElectronStopping const & ) = delete;
    HybridElectronStopping ( HybridElectronStopping&& )                  = delete;
    HybridElectronStopping& operator= ( HybridElectronStopping&& )       = delete;

    /** Apply the slowing-down drag and stage the conjugate electron heating
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

private:

    /** Coulomb logarithm for the slowing-down rate (input, required > 0). */
    amrex::ParticleReal m_coulomb_log;
};

#endif // WARPX_PARTICLES_COLLISION_HYBRIDELECTRONSTOPPING_H_
