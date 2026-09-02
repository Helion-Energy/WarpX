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

    /** Global maximum particle speed (m/s) of this species.
     * Public because the reduction uses an extended device lambda, which
     * CUDA does not allow inside private/protected member functions.
     */
    [[nodiscard]] amrex::Real MaxParticleSpeed ();

    /** Component-resolved global maximum speeds (m/s) of this species.
     *
     * \c combined is the maximum over particles of the PER-PARTICLE quantity
     * v_meridional + margin*|v_theta|, and is the quantity the grid CFL is
     * sized on. It is deliberately NOT the sum of \c meridional and
     * \c azimuthal: those are independent reductions, attained in general by
     * DIFFERENT particles, so their sum can exceed the speed of every
     * particle present -- and in particular exceed \c full, which would make
     * sizing on it strictly worse than sizing on the full speed. Reduced per
     * particle, the combination is bounded by
     * max_a (cos a + margin*sin a) = sqrt(1 + margin^2) times that particle's
     * own speed, i.e. 1.0198*|v| at the default margin.
     *
     * \c full, \c meridional and \c azimuthal are kept separately because they
     * are what make a mis-sized reference speed diagnosable from the witness
     * line.
     */
    struct MaxSpeeds
    {
        amrex::Real full = 0.0;        //!< max |v|
        amrex::Real meridional = 0.0;  //!< max sqrt(v_r^2 + v_z^2)
        amrex::Real azimuthal = 0.0;   //!< max |v_theta|
        amrex::Real combined = 0.0;    //!< max of v_mer + margin*|v_theta|
    };

    /** Global maxima of the full, meridional, azimuthal and combined speeds,
     * in a single pass.
     *
     * In cylindrical geometry the momenta are stored as lab Cartesian
     * components alongside the azimuthal position angle, so the decomposition
     * uses that stored angle,
     *
     *   v_r     =  v_x cos(theta) + v_y sin(theta)
     *   v_theta = -v_x sin(theta) + v_y cos(theta)
     *
     * which is exact and free of the r -> 0 singularity that dividing by the
     * radius would introduce. Outside cylindrical geometry there is no
     * distinct azimuthal direction: the meridional entry is set to the full
     * speed and the azimuthal entry to zero, so that sizing on the meridional
     * speed degenerates to sizing on the full speed.
     *
     * Public because the reduction uses an extended device lambda, which
     * CUDA does not allow inside private/protected member functions.
     */
    [[nodiscard]] MaxSpeeds MaxParticleSpeedsByComponent ();

    /** Fraction of |v_theta| folded into the meridional reference speed.
     * Returns the explicit input when one was given, otherwise the value
     * derived from the cyclotron CFL (see m_v_theta_margin). */
    [[nodiscard]] amrex::Real EffectiveVThetaMargin () const
    {
        return (m_v_theta_margin >= 0.0) ? m_v_theta_margin
                                         : 0.5*m_cfl_cyclotron;
    }

    /** True when the v_theta margin was derived rather than set in the deck */
    [[nodiscard]] bool VThetaMarginIsDerived () const
    {
        return m_v_theta_margin < 0.0;
    }

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
     * particle speed (the measured path) */
    amrex::Real m_v_ref = -1.0;
    /** Size the grid-crossing CFL on the meridional speed instead of the full
     * speed (measured path only; default 0 = full speed, i.e. the behavior of
     * the code before this option existed).
     *
     * Motivation: a species accelerated by a rising axial flux (betatron
     * pumping) conserves the canonical angular momentum
     * m r v_theta + q psi/(2 pi), so the gained speed is almost entirely
     * AZIMUTHAL. In cylindrical geometry with a single azimuthal mode, motion
     * in theta crosses no (r,z) cell, so the deposition guard band -- the
     * constraint the grid CFL exists to satisfy -- is set by the meridional
     * speed alone. Sizing on the full speed then over-resolves by the ratio
     * of the two, which for a strongly pumped population is a factor of a
     * few. The constraint is per subcycle, not per step, because boundary
     * handling redistributes the particles every subcycle.
     *
     * The maxima are re-measured every evaluation and never inferred from a
     * cached azimuthal fraction: that fraction is a property of the current
     * regime, and any pitch-angle scattering or loss of axisymmetry converts
     * v_theta into meridional speed with no warning. */
    int m_v_ref_meridional = 0;
    /** Fraction of |v_theta| added to the meridional speed when sizing on the
     * meridional leg. Negative (the default) means DERIVE it from the
     * cyclotron CFL; a non-negative input is used as given.
     *
     * The derivation follows from the code rather than from taste. The
     * cyclotron leg of ComputeSubcycleCount enforces
     * dt_sub <= cfl_cyclotron/omega_c, so the turning angle within a subcycle
     * is phi = omega_c*dt_sub <= cfl_cyclotron radians. A particle starting
     * purely azimuthal accrues meridional displacement
     * v_theta*[1 - cos(phi)]/omega_c over that subcycle, which as a fraction
     * of the budgeted displacement v_ref*dt_sub is exactly
     *
     *     [1 - cos(phi)] / phi
     *
     * whose small-angle form is phi/2. At the endpoint phi = cfl_cyclotron
     * that is 0.0500 at 0.1, 0.0997 at 0.2 and 0.1973 at 0.4 -- i.e.
     * cfl_cyclotron/2 to better than 1.5% across the useful range, which is
     * the value used when the knob is left unset.
     *
     * Deriving it makes the coupling automatic: a deck that raises the
     * cyclotron CFL then gets the right margin without having to know that a
     * second knob has to move with it.
     *
     * \see EffectiveVThetaMargin */
    amrex::Real m_v_theta_margin = -1.0;
    /** Fractional change in the implied subcycle count below which the count
     * is left alone (meridional sizing only). The measured speeds move a
     * little every step, so without this the count chatters between
     * neighbouring values and rescales the deposition cadence for no benefit.
     * Only the count is held: no measurement is skipped, so a genuine
     * excursion is still caught on the step it happens. Holding a count up to
     * this fraction below the newly implied one lets the effective grid CFL
     * overshoot by the same fraction, which the default leaves far below
     * unity. */
    amrex::Real m_v_ref_hysteresis = 0.2;
    /** Cap on the number of subcycles per step (safety) */
    int m_max_subcycles = 10000;
    /** Treat a binding subcycle cap as an error rather than silently clamping
     * (measured path only; the fixed-v_ref path keeps its historical
     * clamping behavior). A silent clamp under-resolves the grid crossing
     * exactly as a stale reference speed would, but leaves no trace, so the
     * measured path refuses to continue instead. */
    int m_cap_is_error = 1;
    /** Accumulate time-averaged rho/J over the subcycles and serve them from
     * the DepositCharge/DepositCurrent overrides */
    bool m_do_orbit_averaged_deposition = true;

    /** Subcycle count and time step of the most recent Evolve call */
    int m_n_subcycles = 1;
    amrex::Real m_dt_subcycle = 0.0;

    /** One-time boot print of the subcycle configuration (first Evolve) */
    bool m_boot_printed = false;
    /** Subcycle count at the last witness line; -1 until the boot print seeds
     * it. Reporting is throttled to changes above 10% so a count hunting
     * between neighbouring values stays quiet. */
    int m_n_subcycles_last_reported = -1;
    /** True once the meridional-sizing gate has been reported, so the warning
     * about falling back to the full speed is issued only once. */
    bool m_meridional_gate_reported = false;

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
