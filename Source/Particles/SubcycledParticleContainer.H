/* Copyright 2026 S. Eric Clark (Helion Energy)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SubcycledParticleContainer_H_
#define WARPX_SubcycledParticleContainer_H_

#include "PhysicalParticleContainer.H"

#include "WarpXParticleContainer_fwd.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>
#include <AMReX_AmrCoreFwd.H>

#include <array>
#include <memory>
#include <string>

/**
 * SubcycledParticleContainer advances a species on its own subcycle time step,
 * smaller than the global step, and (optionally) deposits orbit-averaged
 * charge and current densities.
 *
 * This is intended for species whose characteristic time scales (gyration,
 * transit over a cell) are much shorter than the global time step, e.g.
 * fusion-born alpha particles in a hybrid (Ohm's law) simulation, or
 * strongly-magnetized species in electrostatic simulations. The species is
 * pushed n_subcycles times per global step against the (frozen) fields of the
 * current step, with
 *
 *   dt_sub = min( cfl_cyclotron / omega_c(B_max), cfl_grid * dx_min / v_ref )
 *
 * and the final subcycle truncated so the subcycles sum exactly to the global
 * dt. When orbit-averaged deposition is enabled, each subcycle deposits
 * charge and current scaled by lambda_i = dt_sub_i/dt into accumulators, so
 * that the species' contribution to rho and J is the time average over the
 * global step:
 *
 *   rho_avg = sum_i lambda_i rho(t_i),   J_avg = sum_i lambda_i J(t_i)
 *
 * with sum_i lambda_i = 1 exactly. The DepositCharge/DepositCurrent overrides
 * then serve these accumulated moments to any caller (hybrid Ohm's-law solve,
 * electrostatic solve, diagnostics) in place of an instantaneous deposition
 * at the end-of-step particle positions.
 *
 * Particles are scraped at the embedded boundary and at domain boundaries
 * every subcycle, so wall-loss diagnostics (ParticleBoundaryBuffer) record
 * the sub-step absorption time, and absorbed particles stop contributing to
 * the orbit-averaged moments from the subcycle of their absorption.
 *
 * SubcycledParticleContainer derives from PhysicalParticleContainer and is
 * selected with `<species>.do_subcycled_push = 1`.
 */
class SubcycledParticleContainer
    : public PhysicalParticleContainer
{
public:
    SubcycledParticleContainer (amrex::AmrCore* amr_core,
                                int ispecies,
                                const std::string& name);

    ~SubcycledParticleContainer () override = default;

    SubcycledParticleContainer ( SubcycledParticleContainer const &)             = delete;
    SubcycledParticleContainer& operator= ( SubcycledParticleContainer const & ) = delete;
    SubcycledParticleContainer ( SubcycledParticleContainer&& )                  = default;
    SubcycledParticleContainer& operator= ( SubcycledParticleContainer&& )       = default;

    /** Subcycled particle advance.
     *
     * Pushes the species n_subcycles times with dt_sub against the frozen
     * fields of this step (Efield_aux/Bfield_aux), applying boundary
     * conditions and embedded-boundary scraping every subcycle, and
     * accumulating orbit-averaged moments when enabled.
     */
    void Evolve (ablastr::fields::MultiFabRegister& fields,
                 int lev,
                 const std::string& current_fp_string,
                 amrex::Real t,
                 amrex::Real dt,
                 SubcyclingHalf subcycling_half=SubcyclingHalf::None,
                 bool skip_deposition=false,
                 PositionPushType position_push_type=PositionPushType::Full,
                 MomentumPushType momentum_push_type=MomentumPushType::Full,
                 ImplicitOptions const * implicit_options = nullptr) override;

    /** Serve the orbit-averaged current instead of an instantaneous deposit.
     *
     * If a valid orbit average is available (i.e. Evolve ran this step with
     * orbit-averaged deposition enabled), the accumulated J_avg is added to
     * J. Otherwise falls back to the standard instantaneous deposition.
     */
    void DepositCurrent (ablastr::fields::MultiLevelVectorField const & J,
                         amrex::Real dt, amrex::Real relative_time) override;

    /** Serve the orbit-averaged charge density instead of an instantaneous
     * deposit. Falls back to the standard deposition when no valid orbit
     * average is available.
     */
    void DepositCharge (const ablastr::fields::MultiLevelScalarField& rho,
                        bool local = false, bool reset = false,
                        bool apply_boundary_and_scale_volume = false,
                        bool interpolate_across_levels = true,
                        int icomp = 0) override;

    /** Half-global-step momentum shifts (leapfrog desynchronization at a
     * synchronized start, velocity synchronization around diagnostics) are
     * meaningless for a species whose momentum lives on the subcycle
     * stagger: a half-global-step Boris rotation can exceed a full gyration.
     * The momenta are instead kept at their subcycle stagger, synchronized
     * with the positions to within half a subcycle. This override is a
     * no-op.
     */
    void PushP (int lev, amrex::Real dt,
                const amrex::MultiFab& Ex,
                const amrex::MultiFab& Ey,
                const amrex::MultiFab& Ez,
                const amrex::MultiFab& Bx,
                const amrex::MultiFab& By,
                const amrex::MultiFab& Bz,
                MomentumPushType momentum_push_type = MomentumPushType::Full) override;

    /** Number of subcycles used in the most recent Evolve call */
    [[nodiscard]] int getNSubcycles () const { return m_n_subcycles; }

    /** Subcycle time step used in the most recent Evolve call */
    [[nodiscard]] amrex::Real getSubcycleDt () const { return m_dt_subcycle; }

protected:

    /** Compute m_dt_subcycle and m_n_subcycles for this step from the
     * cyclotron CFL (using max |B| over Bfield_aux) and the grid-crossing CFL
     * (using v_ref, or the current max particle speed when v_ref <= 0).
     */
    void ComputeSubcycleCount (ablastr::fields::MultiFabRegister& fields,
                               int lev, amrex::Real dt);

    /** Global maximum particle speed (m/s) of this species */
    [[nodiscard]] amrex::Real MaxParticleSpeed ();

    /** Apply domain boundary conditions, gather scraped particles into the
     * boundary buffer, redistribute, and scrape at the embedded boundary —
     * for this species only. Mirrors WarpX::HandleParticlesAtBoundaries.
     *
     * \param cur_time time at the end of the current subcycle
     * \param dt_sub subcycle time step (used for sub-step scrape timestamps)
     */
    void HandleBoundariesSubcycle (amrex::Real cur_time, amrex::Real dt_sub);

    /** Allocate/zero the moment accumulators using the layout of the
     * destination fabs in the field register.
     */
    void PrepareMomentAccumulators (ablastr::fields::MultiFabRegister& fields,
                                    int lev,
                                    const std::string& current_fp_string);

    /** Maximum gyration angle per subcycle (radians) */
    amrex::Real m_cfl_cyclotron = 0.1;
    /** CFL fraction of a cell crossing per subcycle */
    amrex::Real m_cfl_grid = 0.4;
    /** Reference speed for the grid CFL; <= 0 selects the per-step max
     * particle speed */
    amrex::Real m_v_ref = -1.0;
    /** Cap on the number of subcycles per step (safety) */
    int m_max_subcycles = 10000;
    /** Accumulate time-averaged rho/J over the subcycles and serve them from
     * the DepositCharge/DepositCurrent overrides */
    bool m_do_orbit_averaged_deposition = true;

    /** Subcycle count and time step of the most recent Evolve call */
    int m_n_subcycles = 1;
    amrex::Real m_dt_subcycle = 0.0;

    /** Orbit-averaged moment accumulators and per-subcycle scratch (level 0
     * only; multi-level runs are not supported) */
    std::unique_ptr<amrex::MultiFab> m_rho_avg;
    std::unique_ptr<amrex::MultiFab> m_rho_tmp;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_j_avg;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_j_tmp;

    /** True when the accumulators hold a valid average for the current step */
    bool m_moments_valid = false;
    /** True when rho accumulation is active (requires rho_fp to exist) */
    bool m_average_rho = false;
    /** True when J accumulation is active (requires the destination current
     * field to exist) */
    bool m_average_j = false;
};

#endif // WARPX_SubcycledParticleContainer_H_
