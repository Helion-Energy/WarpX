/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *          Prabhat Kumar (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_HYBRIDPICMODEL_H_
#define WARPX_HYBRIDPICMODEL_H_

#include "HybridPICModel_fwd.H"

#include "Fields.H"
#include "Fluids/QdsmcParticleContainer_fwd.H"

#include "ExternalVectorPotential.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/text/IntervalsParser.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class PoissonBoundaryHandler;

/**
 * \brief This class contains the parameters needed to evaluate hybrid field
 * solutions (kinetic ions with fluid electrons).
 */
class HybridPICModel
{
public:
    HybridPICModel ();
    // Defined out-of-line in the .cpp so the unique_ptr<QdsmcParticleContainer>
    // member can destroy a forward-declared particle container.
    ~HybridPICModel ();

    HybridPICModel ( HybridPICModel const & )             = delete;
    HybridPICModel& operator= ( HybridPICModel const & )  = delete;
    HybridPICModel ( HybridPICModel && )                  = delete;
    HybridPICModel& operator= ( HybridPICModel && )       = delete;

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters ();

    /** Allocate hybrid-PIC specific multifabs. Called in constructor. */
    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister & fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        int ncomps,
        const amrex::IntVect& ngJ,
        const amrex::IntVect& ngRho,
        const amrex::IntVect& ngEB,
        const amrex::IntVect& jx_nodal_flag,
        const amrex::IntVect& jy_nodal_flag,
        const amrex::IntVect& jz_nodal_flag,
        const amrex::IntVect& rho_nodal_flag,
        const amrex::IntVect& Ex_nodal_flag,
        const amrex::IntVect& Ey_nodal_flag,
        const amrex::IntVect& Ez_nodal_flag,
        const amrex::IntVect& Bx_nodal_flag,
        const amrex::IntVect& By_nodal_flag,
        const amrex::IntVect& Bz_nodal_flag
    ) const;

    void InitData (const ablastr::fields::MultiFabRegister& fields);

    /**
     * \brief
     * Function to evaluate the external current expressions and populate the
     * external current multifab. Note the external current can be a function
     * of time and therefore this should be re-evaluated at every step.
     */
    void GetCurrentExternal ();

    /**
     * \brief
     * Function to calculate the total plasma current based on Ampere's law while
     * neglecting displacement current (J = curl x B). Any external current is
     * subtracted as well. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     *
     * \param[in] Bfield       Magnetic field from which the current is calculated.
     * \param[in] eb_update_E  Indicate in which cell J should be calculated (related to embedded boundaries).
     */
    void CalculatePlasmaCurrent (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E
    ) const;
    void CalculatePlasmaCurrent (
        ablastr::fields::VectorField const& Bfield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev
    ) const;

    /**
     * \brief
     * Function to update the E-field using Ohm's law (hybrid-PIC model).
     */
    void HybridPICSolveE (
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        bool solve_for_Faraday,
        std::optional<bool> include_resistivity = std::nullopt) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, bool solve_for_Faraday,
        std::optional<bool> include_resistivity = std::nullopt) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, PatchType patch_type, bool solve_for_Faraday,
        std::optional<bool> include_resistivity = std::nullopt) const;

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /** Topanga-style sponge caps for the e-form (Yee) substep loop,
     *  B-SIDE ONLY (Belyaev, Larson, Cohen & Clark, Phys. Plasmas 31,
     *  012902 (2024), Sec. II B — Topanga itself is e-form-class in
     *  its hybrid region and damps only the evolved B there; the
     *  algebraic E regenerates every solve and carries no wave state).
     *  Inside a layer of width je_sponge_width at each z end, damp the
     *  in-loop PLASMA-frame B (B_ext is subtracted at substep-loop
     *  entry and re-added at exit, so the drive never sees the layer
     *  and no external-change tracking is needed) toward its frozen
     *  first-call reference, with the rate ramped linearly from zero
     *  at the layer's inner edge (the paper's anti-reflection
     *  requirement) to 1/tau_max = v_w ln(1/f_s)/L_s at the wall.
     *
     *  WHY A SINGLE v_w CANNOT WORK (the two-tier motivation): a wave of
     *  speed v crossing the layer accumulates attenuation ~ ln(1/f_s)
     *  (v_w/v) <xi>, i.e. it scales as v_w/v, so one scalar matches ONE
     *  branch and mistreats every other by that ratio. Measured on the
     *  production anchor, cap band: v_slow (Alfven) 1.18e7 m/s, v_fast
     *  (whistler at k_max = pi/dz) 2.73e8 m/s, ratio 23.2 — and the
     *  legacy deck constant 1e8 sits BETWEEN them, 8.5x too fast for the
     *  Alfven branch (over-damped: a stiff pin that reflects) and 2.7x
     *  too slow for the whistler branch (under-damped: leaks through to
     *  the hard wall behind the layer). A single-speed layer therefore
     *  WALLS the slow branch and LEAKS the fast one simultaneously,
     *  which is the quantitative content of the "impedance geometry"
     *  failure that lost to bare ends on the full-stroke panel.
     *
     *  TWO-TIER PROFILE (je_sponge_grade = 1, adaptive path): the layer
     *  is split at je_sponge_tier_frac into an INNER slow-matched tier
     *  of width L1 and an OUTER fast-matched tier of width L2, with the
     *  rate ramped linearly 0 -> R1 = ln(1/f_s) v_slow/L1 across tier 1
     *  and R1 -> R2 = ln(1/f_s) v_fast/L2 across tier 2. The profile is
     *  piecewise-linear and CONTINUOUS at the interface: a rate step is
     *  itself an impedance discontinuity, so the anti-reflection
     *  requirement is applied per tier rather than once. Slow content
     *  meets its match first and is absorbed in tier 1; fast content
     *  crosses tier 1 nearly unattenuated (it accumulates only
     *  R1 L1/v_fast = ln(1/f_s) v_slow/v_fast << ln(1/f_s)) and is
     *  absorbed in tier 2 at its own match. Tiering rather than a single
     *  graded rate is what lets each branch's WIDTH be sized
     *  independently — absorption quality is set by attenuation per cell
     *  (ln(1/f_s) dz/L for that tier), so a branch needing gentler
     *  per-cell treatment simply gets a wider tier.
     *  Implicit per-direction update on each component's own
     *  staggering (the ramp coordinate offsets half a cell where B_d
     *  is cell-centered in z); unconditionally stable at any
     *  dt/tau and operator-split from the curl advance, so the
     *  substep error control never sees it. Applied once per ACCEPTED
     *  substep, BEFORE any coupling-interval close and BEFORE the
     *  B_old latch. Restart note: the references re-seed from the
     *  restored state on a mid-stroke restart (unround-tripped
     *  companion state; sidecar round-trip is a follow-up). */
    void ApplySpongeLayerBYee (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        int lev, amrex::Real dt_sub);

    /** \brief Measure the two wave-branch speeds in one z-cap band.
     *
     * Returns the xi-weighted means over the cap-band cells of
     *   v_slow = |B| / sqrt(mu0 rho_m)          (Alfven branch)
     *   v_fast = k_max |B| / (mu0 n_e e)        (grid-scale whistler,
     *                                            k_max = pi/dz)
     * i.e. the slowest and fastest signals the scheme retains at the
     * caps. Both are evaluated on CELL CENTRES from cell-centred
     * averages of the Yee B components and the solver's own charge
     * density, floored at n_floor exactly as the Ohm solve floors it,
     * so an evacuated cap cannot mint a divergent speed. The weight is
     * the ramp coordinate xi, so the deep layer — where the damping
     * rate actually matters — dominates the mean.
     *
     * \param[in] lev        refinement level
     * \param[in] cap        0 = lo-z cap, 1 = hi-z cap
     * \param[out] v_slow    xi-weighted mean Alfven speed [m/s]
     * \param[out] v_fast    xi-weighted mean whistler speed [m/s]
     * \return true if the band held any weighted cells on any rank
     */
    bool MeasureSpongeCapSpeeds (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        int lev, int cap, amrex::Real& v_slow, amrex::Real& v_fast);

    /** Frozen first-call plasma-frame B references of the Yee sponge,
     *  one per component staggering (valid region only; ghosts are
     *  refreshed after each application); lazily seeded by
     *  ApplySpongeLayerBYee on its first call. */
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_sponge_bref_yee;
    /** Seed latch of the Yee sponge references. */
    bool m_sponge_yee_init = false;
    /** Time-averaged (EMA) branch speeds per cap [m/s], index 0 = lo-z,
     *  1 = hi-z; only used when the adaptive path is armed
     *  (je_sponge_vw_tau > 0). Seeded from the first measurement so the
     *  layer never rides a ramp-in transient. NOT round-tripped through
     *  the checkpoint: on restart they re-seed from the first
     *  post-restart measurement, which is self-healing because seeding
     *  IS the measurement (unlike the B references, whose restart wart
     *  is a genuine state loss). */
    std::array<amrex::Real, 2> m_sponge_vslow_ema = {0.0, 0.0};
    std::array<amrex::Real, 2> m_sponge_vfast_ema = {0.0, 0.0};
    /** Seed latch of the adaptive-speed EMAs. */
    bool m_sponge_vw_ema_init = false;
    /** Last reported adaptive speeds [m/s]; the witness print is
     *  throttled to >= 10% changes, same philosophy as the substep
     *  controller witness. */
    std::array<amrex::Real, 2> m_sponge_vslow_reported = {0.0, 0.0};
    std::array<amrex::Real, 2> m_sponge_vfast_reported = {0.0, 0.0};
    /** Ion mass per unit charge m_i/q_i [kg/C], resolved once from the
     *  charged species on the first adaptive call and used to convert
     *  the solver's charge density to a mass density
     *  (rho_m = rho_charge * m_i/q_i). Exact for a single ion species
     *  (the production case); for several species it is the unweighted
     *  mean over charged species, which is a representative scalar for
     *  impedance matching rather than a conserved quantity — the layer
     *  tolerates a modest mis-set speed (it shifts where in the layer a
     *  branch is absorbed, not whether it is). */
    amrex::Real m_sponge_m_over_q = 0.0;

    void BfieldEvolveRK4 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    amrex::Real BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void FieldPush (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /**
     * \brief
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     * The two-argument overload evaluates the closure against an explicitly
     * provided charge density instead of rho_fp.
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;
    void CalculateElectronPressure (int lev, amrex::MultiFab const& rho) const;

    /**
     * \brief Re-evaluate the algebraic-closure electron pressure (and the
     * diagnostic electron-temperature mirror) from a fresh end-of-step
     * charge-density deposit. Only meaningful under the theta-implicit
     * scheme, where every in-solve closure evaluation consumes
     * midpoint-position deposits: without this refresh the dumped Pe/Te
     * carry a half-step-stale density label, a first-order error in any
     * Te-based convergence metric even though the field/ion dynamics are
     * second order. Deposits from the as-left (un-redistributed) step-end
     * particle state so the particle ordering -- and with it the bit-level
     * trajectory of the run -- is untouched; see the implementation note.
     */
    void CalculateElectronPressureAtStepEnd () const;

    /**
     * \brief Compute the electron fluid velocity V_e on the grid from the
     * deposited ion current and the plasma current: V_e = (J_i - J_plasma)/rho,
     * with rho floored at m_n_floor * q_e. Used by the resistive-drag operator.
     */
    void CalculateElectronFluidVelocity () const;
    void CalculateElectronFluidVelocity (int lev) const;

    /**
     * \brief Compute the per-species ion bulk velocity Vs = Js/rhos on the
     * grid for every charged species. Output is stored in Vs_fp_{species}
     * (face-staggered like J). Used by the resistive-drag operator to
     * apply a per-cell uniform shift (V_s - V_e)(1 - exp(-nu dt)) to every
     * particle, leaving the thermal moment intact.
     */
    void CalculateIonFluidVelocity () const;
    void CalculateIonFluidVelocity (int lev) const;

    /**
     * \brief Fill the electron pressure multifab given the kinetic particle
     * charge density (and assumption of quasi-neutrality) using the user
     * specified electron equation of state.
     *
     * \param[out] Pe_field scalar electron pressure MultiFab at a given level
     * \param[in] rho_field scalar ion charge density Multifab at a given level
     */
    void FillElectronPressureMF (
        amrex::MultiFab& Pe_field,
        amrex::MultiFab const& rho_field ) const;

    /** Which J_i / rho time levels QDSMCInitializeUe combines into V_e.
     *  The QDSMC advance is called at a point where hybrid_current_fp_temp
     *  holds J_i^{n-1/2}, current_fp holds J_i^{n+1/2}, hybrid_rho_fp_temp
     *  holds rho^n and rho_fp holds rho^{n+1} (the euler/leapfrog call site);
     *  the pc corrector runs after the J_i^n LinComb, where
     *  hybrid_current_fp_temp already holds J_i^n. */
    enum struct QdsmcUeMode {
        JiOld,       /**< euler (#6982): J_i^{n-1/2}, rho^n, J_plasma(B^n) */
        JiAvg,       /**< leapfrog: J_i^n = avg(temp, fp), rho^n, J_plasma(B^n) */
        JiNewRhoHalf /**< pc corrector: J_i^{n+1/2} (= current_fp),
                          rho^{n+1/2} = avg(rho_fp_temp, rho_fp),
                          J_plasma(B^{n+1/2}) */
    };

    /**
     * \brief Fill the nodal V_e = -(J_plasma - J_i) / (q_e n_e) MultiFabs by
     *        interpolating the Yee-staggered total plasma current and ion
     *        current to the nodal grid (using ablastr::coarsen::sample::Interp)
     *        and dividing by rho. Cells with rho <= rho_floor are left at 0.
     *
     *        Reads hybrid_current_fp_plasma (J_plasma) plus the J_i / rho
     *        fields selected by \c mode (see QdsmcUeMode). Writes
     *        hybrid_electron_velocity_fp (3 directions, nodal).
     */
    void QDSMCInitializeUe (int lev, QdsmcUeMode mode = QdsmcUeMode::JiOld) const;

    /**
     * \brief Parameterized variant of QDSMCInitializeUe: fill the nodal V_e
     *        from the given charge density and currents,
     *          V_e = -(J_plasma_mid - J_i) / rho,
     *        with the plasma current interpolated in time per cell as
     *          J_plasma_mid = Jp_old + jp_interp * (Jp_new - Jp_old).
     *        Pass the same field for Jp_new and Jp_old (any jp_interp) to use
     *        it directly, which reproduces the explicit-path behavior. The
     *        theta-implicit path passes Jp_new = J_plasma(B^{n+theta}),
     *        Jp_old = J_plasma(B^n) and jp_interp = 0.5/theta to sample the
     *        temporal midpoint.
     *
     *        The optional averaging partners serve the QdsmcUeMode wrapper:
     *        when rho_avg_with is non-null the denominator density is
     *        0.5*(rho + rho_avg_with) per node (pc corrector: rho^{n+1/2});
     *        when J_i_avg_with is non-null the ion current is
     *        0.5*(J_i + J_i_avg_with) (leapfrog: J_i^n from the bracketing
     *        half-integer deposits).
     */
    void QDSMCInitializeUe (int lev,
        amrex::MultiFab const & rho,
        ablastr::fields::VectorField const & J_i,
        ablastr::fields::VectorField const & Jp_new,
        ablastr::fields::VectorField const & Jp_old,
        amrex::Real jp_interp,
        amrex::MultiFab const * rho_avg_with = nullptr,
        ablastr::fields::VectorField const * J_i_avg_with = nullptr) const;

    /**
     * \brief Fill the nodal K_e = T_e * n_e^(1-gamma) * (k_B / q_e) MultiFab
     *        from the current T_e and rho_fp_temp (= rho at n+1/2), using the
     *        floored density max(n_e, n_floor) so below-floor (halo) cells
     *        carry a valid K_e instead of acting as an absorbing K_e = 0
     *        boundary.
     *
     *        Reads hybrid_electron_temperature_fp (T_e in K) and
     *        hybrid_rho_fp_temp; writes hybrid_entropy_fp.
     */
    void QDSMCInitializeKe (int lev) const;

    /** \brief Parameterized variant of QDSMCInitializeKe: seed K_e from the
     *         current T_e and the given charge density (the state whose
     *         entropy the markers carry — rho^n on the theta-implicit path).
     */
    void QDSMCInitializeKe (int lev, amrex::MultiFab const & rho) const;

    /**
     * \brief After the QDSMC scatter, recover T_e^{n+1} from
     *           T_e = (deposited K*N) / (deposited N) / n_e^(1-gamma)
     *                 / (k_B / q_e)
     *        using the floored density max(n_e, n_floor) from rho at n+1
     *        (rho_fp), the deposited entropy field (hybrid_entropy_fp), and
     *        the deposited weight field (hybrid_qdsmc_weights_fp). Cells
     *        whose deposited weight is at or below m_qdsmc_n_floor keep
     *        their previous T_e.
     *
     *        Writes hybrid_electron_temperature_fp.
     */
    void QDSMCUpdateTe (int lev) const;

    /** \brief Parameterized variant of QDSMCUpdateTe: recover T_e from the
     *         deposited entropy/weights and the given new-time charge density
     *         (the extrapolated rho^{n+1} iterate on the theta-implicit path).
     */
    void QDSMCUpdateTe (int lev, amrex::MultiFab const & rho_new) const;

    /** \brief Per-phase minimum-T_e tracer (WARPX_QDSMC_PHASE_MINTE=N):
     *         when the env var is set, print the global min of
     *         hybrid_electron_temperature_fp and its cell after the labeled
     *         step-phase. Locates which phase first writes a given T_e
     *         extremum when per-subsystem toggles cannot separate the
     *         writers (phases probed but not toggleable individually).
     *         The value is a step cadence: full probes run on steps with
     *         step % N == 0; the other steps carry a min()-only
     *         first-negative latch (one reduction) that prints the first
     *         negative it sees (suffix LATCHED) and then disarms, so no
     *         negative passes silently at any step while steady-state cost
     *         drops by the cadence factor. Unparsable values act as N = 1;
     *         no work unset.
     */
    void QdsmcPhaseMinTe (int lev, char const * phase) const;

    /**
     * \brief Insulating EB wall: after the QDSMC scatter and before
     *        QDSMCUpdateTe, fold the entropy deposit (K*N) that landed on
     *        below-floor nodes back onto the open nodes next to them.
     *
     *        Entropy deposited on below-floor nodes is dead weight:
     *        QDSMCUpdateTe's floor guard never recovers it and the next
     *        QDSMCInitializeKe re-zeroes K_e there, so entropy carried into
     *        the floored halo by the plasma-edge markers is dropped -- a
     *        one-way energy drain out of the open set at every density-floor
     *        boundary (measured on the standoff-disc instrument: -7.3e-2
     *        relative over 16 steps AT REST). The fold moves each
     *        below-floor node's deposited K*N equally onto the open nodes of
     *        its 3^D-1 neighborhood (the reach of the order-1 deposit) and
     *        clears the node, conserving the deposited entropy over the open
     *        set (same instrument: -1.1e-4). The deposited weight N is NOT
     *        folded: it is a marker-density estimate that the recovery
     *        discards on below-floor nodes anyway, and importing it without
     *        matching entropy dilutes the edge temperature (measured to make
     *        the drift worse than no fold at all).
     *
     *        Below-floor nodes with no open neighbor are cleared without a
     *        destination (their content is dropped, as before the fold);
     *        covered nodes that sit above the floor (only reachable through
     *        a pathological initial load) are left untouched.
     *
     *        Reads hybrid_rho_fp_temp (rho^n, the mask) and distance_to_eb;
     *        rewrites hybrid_entropy_fp in place.
     */
    void QDSMCFoldInsulatingDeposits (int lev) const;

    /**
     * \brief Insulating EB wall: after QDSMCUpdateTe, fill T_e with zero
     *        normal gradient into the standoff band and the EB-covered
     *        region, so the P_e emitted downstream drives no spurious
     *        grad-P_e electric field at the plasma edge.
     *
     *        The fill mask keys off the transport's own boundary set -- a
     *        node is filled only if it is below the density floor
     *        (m_qdsmc_n_floor on rho_fp, the QDSMCUpdateTe guard) or
     *        EB-covered (distance_to_eb <= 0), AND within
     *        (eb_standoff_cells + 1) cells of the surface. Live plasma nodes
     *        are never touched (a geometry-based mask would rewrite
     *        above-floor plasma-edge nodes each step and act as an energy
     *        source/sink). Filling proceeds by onion-peel sweeps: each
     *        masked node takes the average of its already-valid neighbors,
     *        so the first band ring copies the plasma-edge T_e (the
     *        zero-normal-gradient condition) and deeper rings follow.
     *
     *        Reads rho_fp and distance_to_eb; writes
     *        hybrid_electron_temperature_fp (masked nodes only, ghosts
     *        refreshed).
     */
    void QDSMCApplyInsulatingEBFill (int lev) const;

    /**
     * \brief Resistive electron-heating source (see Phys. Plasmas 31, 012902 (2024), Eq. 12).
     *        Adds, per cell and per charged ion species,
     *           T_e += dt * (gamma-1) * Z_s e^2 eta n_s |dV|^2 / k_B,
     *        with the e-i relative drift dV = J_plasma/(e n_e) and eta from the
     *        Ohm's-law parser. Summed over species,
     *           S_e = e^2 eta n_e * Sigma_s Z_s n_s |dV|^2,
     *        which reduces to eta J^2 for a single species. n_s is recovered from
     *        the species charge fraction f_s = rho_fp_s / Sigma_t rho_fp_t
     *        = Z_s n_s/n_e (2pi*r-cancelling) -> n_s = f_s n_e/Z_s, with n_e from
     *        the total rho_fp. Deterministic, no per-particle scatter. Called
     *        when m_include_joule_heating is true. When the Te-threshold redirect
     *        is on (\c redirect_E non-null), the contribution from cells with
     *        Te >= m_joule_redirect_Te_eV is written into \c redirect_E for
     *        QDSMCApplyIonHeating to deposit on the ions instead of T_e.
     *
     * \param lev AMR refinement level.
     * \param dt  PIC timestep [s].
     * \param redirect_E nodal output field for the redirected heat, one component
     *        per charged ion species (ordered as in GetSpeciesNames, charged
     *        species only). Component c holds the m_i-independent energy
     *        E_s = (2/3) n_e Z_s e^2 eta |dV|^2 dt [J] (so sig_redir^2 = E_s/m_i).
     *        nullptr when the Joule redirect is off.
     */
    void QDSMCAddJouleHeating (int lev, amrex::Real dt,
                               amrex::MultiFab * redirect_E = nullptr) const;

    /** \brief Parameterized variant of QDSMCAddJouleHeating using the given
     *         charge density for the coefficient state (rho^{n+1/2} on the
     *         theta-implicit path; the plasma current is read from
     *         hybrid_current_fp_plasma at its current state in both paths).
     */
    void QDSMCAddJouleHeating (int lev, amrex::Real dt,
                               amrex::MultiFab const & rho,
                               amrex::MultiFab * redirect_E) const;

    /**
     * \brief PHYSICAL Braginskii electron viscous heating source (Thrust C
     *        companion to the tensor conduction operator).
     *
     * Adds the anisotropic viscous dissipation of the electron fluid to T_e,
     *   T_e += dt (gamma-1) Q_nu / (n_e k_B),
     *   Q_nu = (3/4) mu_par S^2 + (1/2) mu_perp (W:W - (3/2) S^2)  [W/m^3],
     * with the Braginskii coefficients and the tensor algebra in
     * BraginskiiViscosity.H. Both terms are separately non-negative, so the
     * source is pointwise positive-definite and cannot cool a cell.
     *
     * The electron fluid velocity is rebuilt on the nodal grid exactly as
     * CalculateElectronFluidVelocity does it, u_e = (J_i - J_plasma)/rho,
     * so the operator does not depend on Ve_fp being allocated or on where
     * in the step it was last refreshed.
     *
     * Q_nu is also written (overwritten, not accumulated -- the theta path
     * re-enters this on every Newton residual evaluation) to the nodal
     * diagnostic field hybrid_qdsmc_visc_heating_fp, which is what the
     * HybridDissipation reduced diag integrates into its P_nu column. That
     * field is the only sound proof the term is live: a nonzero value in
     * this channel's own sink column, not a knob in warpx_used_inputs.
     *
     * NOT the hyper-resistive term: see the m_eta_h declaration and the
     * BraginskiiViscosity.H file comment for why the two are kept apart.
     *
     * \param lev AMR refinement level.
     * \param dt  source-stage timestep [s] (the Strang callers pass dt/2).
     */
    void QDSMCAddViscousHeating (int lev, amrex::Real dt) const;

    /** \brief Parameterized variant of QDSMCAddViscousHeating using the given
     *         charge density for the coefficient state (rho^{n+1/2} on the
     *         theta-implicit path), mirroring QDSMCAddJouleHeating.
     */
    void QDSMCAddViscousHeating (int lev, amrex::Real dt,
                                 amrex::MultiFab const & rho) const;

    /**
     * \brief Add the electron-ion thermal-equilibration sink Q_ei to T_e.
     *
     * Per ion species, cools T_e toward that species' (deposited) T_i:
     *   dT_e = -dt (gamma-1) * 3 * (n_s/n_e) * nu_ei * (T_e - T_i_s),
     * with nu_ei from the electron_ion_relaxation_rate parser. This is the
     * ELECTRON-SIDE sink; QDSMCApplyIonHeating deposits the conjugate ion gain
     * so the pair conserves energy. Called when m_include_temperature_relaxation
     * is true.
     *
     * \param lev AMR refinement level.
     * \param dt  PIC timestep [s].
     * \param Ti_dep_by_species per-charged-species ion temperature [eV], deposited
     *        once by the caller (AdvanceElectronEnergyQDSMC) and shared with
     *        QDSMCApplyIonHeating, which runs immediately after with no intervening
     *        ion motion -- so the deposit is identical for both and is done only once.
     */
    void QDSMCAddTemperatureRelaxation (int lev, amrex::Real dt,
        std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const;

    /** \brief Parameterized variant of QDSMCAddTemperatureRelaxation using the
     *         given charge density for the coefficient state (rho^{n+1/2} on
     *         the theta-implicit path).
     */
    void QDSMCAddTemperatureRelaxation (int lev, amrex::Real dt,
        amrex::MultiFab const & rho,
        std::map<std::string, amrex::MultiFab*> const & Ti_dep_by_species) const;

    /** \brief Build the per-charged-species cell-centered ion temperature [eV]
     *         from the shape-aware T_<species> deposits (collapse the
     *         Yee-staggered 3-component field to a cell-centered scalar).
     *         Shared by the explicit orchestrator (per step) and the
     *         theta-implicit path (frozen at t^n across the step).
     *
     * \param[in]  lev AMR level
     * \param[out] owned   storage for the built MultiFabs (must outlive users)
     * \param[out] by_name raw-pointer view keyed by species name
     */
    void QDSMCBuildTiDeposits (int lev,
        std::map<std::string, std::unique_ptr<amrex::MultiFab>> & owned,
        std::map<std::string, amrex::MultiFab*> & by_name) const;

    /**
     * \brief Ion-heating operator: a stochastic drag-diffusion applied to
     *        every ion. It delivers both electron-ion energy channels:
     *
     *   dv_p = -nu_ei (v_p - u_e) dt + sqrt(2 D dt) R,   D = D_qei + D_redir,
     *
     * with R a unit-variance Gaussian vector. The Q_ei conjugate (when
     * m_include_temperature_relaxation) supplies the drag toward the electron
     * fluid u_e and the thermal diffusion D_qei = nu_ei k_B T_e / m_i, relaxing
     * the ions toward a Maxwellian at T_e (T_i -> T_e at rate 2 nu_ei). The
     * Te-threshold Joule redirection (when \c redirect_E is non-null) supplies
     * an additional pure-diffusion heating sig_redir^2 = E_s/m_i that injects,
     * in expectation, the redirected electron energy per cell. Either or both
     * channels may be active. Both are per-species correct (the redirect reads
     * each species' own component of \c redirect_E).
     *
     * \param lev AMR refinement level.
     * \param dt  PIC timestep [s].
     * \param redirect_E per-cell redirected energy [J], one component per charged
     *        ion species (ordered as in QDSMCAddJouleHeating); nullptr when off.
     * \param Ti_dep_by_species per-charged-species ion temperature [eV], deposited
     *        once by the caller and shared with QDSMCAddTemperatureRelaxation
     *        (see that method); nullptr when the Q_ei relaxation channel is off.
     */
    void QDSMCApplyIonHeating (int lev, amrex::Real dt,
                               amrex::MultiFab const * redirect_E = nullptr,
                               std::map<std::string, amrex::MultiFab*> const * Ti_dep_by_species = nullptr) const;

    /**
     * \brief Fill hybrid_electron_pressure_fp from T_e and n_e using the
     *        ideal-gas relation Pe = n_e * k_B * T_e. Called at the end of
     *        each QDSMC step so the existing Ohm's-law E-solver consumes
     *        the QDSMC-updated pressure without modification.
     */
    void QDSMCFillElectronPressureFromTe (int lev) const;

    /** \brief Parameterized variant of QDSMCFillElectronPressureFromTe using
     *         the given charge density. */
    void QDSMCFillElectronPressureFromTe (int lev, amrex::MultiFab const & rho) const;

    /** \brief Theta-centered pressure emit for the implicit hybrid path:
     *           Pe^{n+theta} = n_e^{n+theta} k_B [(1-theta) T_e^n + theta T_e^{n+1}]
     *         with T_e^n from hybrid_electron_temperature_old_fp, T_e^{n+1}
     *         the current hybrid_electron_temperature_fp iterate, and
     *         n_e^{n+theta} extrapolated from rho^n (saved at step start in
     *         hybrid_rho_fp_temp) and rho^{n+1/2} (rho_fp comp 1):
     *           rho^{n+theta} = (1-2 theta) rho^n + 2 theta rho^{n+1/2}.
     */
    void QDSMCFillElectronPressureTheta (int lev, amrex::Real theta) const;

    /**
     * \brief Seed T_e = T_e0 * (max(n_e, n_floor)/n0)^(gamma-1): the
     *        algebraic adiabat evaluated on the floored density.
     *
     * This is the QDSMC transport's zero-gradient state (K_e uniform
     * everywhere, floored halo included), so an energy-equation run starts
     * free of spurious entropy mixing and from the same electron pressure as
     * the algebraic closure wherever n_e >= n_floor (below the floor the
     * emitted pressure is the floored-adiabat value, not the closure's
     * raw-density one). Called from HybridPICInitializeRhoJandB when
     * m_solve_electron_energy_equation is true (a uniform-T_e seed is NOT
     * usable: its K_e contrast across a plasma-vacuum edge is mixed by the
     * remap within ~100 steps, artificially heating the dense regions).
     */
    void SeedTeAdiabat (int lev) const;

    /**
     * \brief Top-level orchestrator for the QDSMC electron-energy step.
     *
     * Performs the full entropy transport over one PIC step (see Phys. Plasmas 31, 012902 (2024)):
     *   1. CalculatePlasmaCurrent so J_plasma is current
     *   2. QDSMCInitializeUe  (V_e from J_plasma and J_i)
     *   3. QDSMCInitializeKe  (K_e from T_e^n and rho^n)
     *   4. QDSMC particle SetV / SetK / PushX / DepositK / DepositField
     *      (4b, insulating EB wall: QDSMCFoldInsulatingDeposits)
     *   5. QDSMCUpdateTe      (recover T_e^{n+1} from K_arr / N_arr / rho^{n+1})
     *      (5b, insulating EB wall: QDSMCApplyInsulatingEBFill)
     *   6. QDSMCAddJouleHeating (if m_include_joule_heating is on):
     *      Joule source (Phys. Plasmas 31, 012902 (2024), Eq. 12) Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2 -> T_e
     *   7. QDSMCFillElectronPressureFromTe (Pe = n_e * k_B * T_e)
     *   8. Reset QDSMC particles to home positions for the next step
     *
     * Called from HybridPICEvolveFields when m_solve_electron_energy_equation is true.
     *
     * Scheme dispatch (qdsmc_time_advance, Thrust-A bake-off):
     *   - euler (default, #6982-identical): forward-Euler push with
     *     V_e(J_i^{n-1/2}, B^n, rho^n), Lie sources after transport.
     *   - leapfrog: T_e lives at HALF-INTEGER times; this call advances
     *     T_e^{n-1/2} -> T_e^{n+1/2} with the time-centered
     *     V_e^n(J_i^n, B^n, rho^n), a two-stage midpoint push, and
     *     Strang-split sources. The emitted Pe^{n+1/2} is time-centered for
     *     both B half-pushes; ApplyQdsmcPeExtrapolation swaps in the
     *     linearly extrapolated Pe^{n+1} before the final E-solve. The first
     *     call self-starts with a dt/2 advance (T_e^0 -> T_e^{1/2}).
     *   - pc: nothing happens at this call site; the transport runs in
     *     AdvanceElectronEnergyQDSMC_PC between the two B half-pushes.
     */
    void AdvanceElectronEnergyQDSMC (amrex::Real dt) const;

    /**
     * \brief Save the electron-energy state needed by the theta-implicit
     *        hybrid scheme at the start of a step: T_e^n (into
     *        hybrid_electron_temperature_old_fp), J_plasma(B^n) (into
     *        hybrid_current_fp_plasma_old; the caller must have refreshed
     *        hybrid_current_fp_plasma from B^n immediately before), and,
     *        when Q_ei relaxation is on, the per-species ion-temperature
     *        deposits (frozen at t^n for the whole step).
     *
     *        Called from ThetaImplicitHybrid::OneStep before the nonlinear
     *        solve.
     */
    void QDSMCSaveImplicitStepStart () const;

    /**
     * \brief Theta-centered QDSMC electron-energy stage for the implicit
     *        hybrid solver, run inside EVERY nonlinear residual evaluation
     *        (in place of CalculateElectronPressure). Re-entrant: always
     *        restarts from the saved T_e^n. Per level:
     *
     *        1. restore T_e <- T_e^n
     *        2. V_e^{n+1/2} from J_i^{n+1/2} (current_fp), the plasma current
     *           interpolated to the midpoint, and rho^{n+1/2} (rho_fp comp 1)
     *        3. seed K_e from (T_e^n, rho^n) (rho_fp comp 0)
     *        4. midpoint marker push: PushX(dt, 1/2) -> gather V_e at the
     *           characteristic midpoint -> PushX(dt, 1) [second order]
     *        5. deposit entropy/weights; recover T_e^{n+1} with the
     *           extrapolated rho^{n+1} = 2 rho^{n+1/2} - rho^n
     *        6. deterministic theta-state sources: Joule heating and the
     *           exact-exponential Q_ei electron sink (coefficients at
     *           rho^{n+1/2}; T_i frozen at t^n). The stochastic ion kick is
     *           deferred to QDSMCFinishImplicitStep.
     *        7. emit Pe^{n+theta} for the Ohm's-law E-solve.
     *
     *        Deterministic (quiet) throughout, so finite-difference
     *        Jacobian-vector products across this stage remain valid.
     *
     * \param dt    full step size
     * \param theta implicit time-biasing parameter
     * \param refresh_species_deposits re-deposit the per-species charge
     *        densities used by the multi-species sources (true once per
     *        Newton iteration, false during Jacobian evaluations so the
     *        linearization sees frozen species fractions)
     */
    void AdvanceElectronEnergyQDSMCTheta (amrex::Real dt, amrex::Real theta,
                                          bool refresh_species_deposits) const;


    /**
     * \brief Post-solve completion of the theta-implicit electron-energy
     *        step: apply the stochastic ion-heating realization (Q_ei
     *        conjugate) once with converged states, refresh Pe to
     *        Pe^{n+1} = n^{n+1} k_B T_e^{n+1} for diagnostics, reset the
     *        QDSMC markers, and release the frozen T_i deposits.
     *
     *        Called from ThetaImplicitHybrid::OneStep after
     *        FinishImplicitParticleUpdate.
     */
    void QDSMCFinishImplicitStep (amrex::Real dt, amrex::Real theta) const;

    /**
     * \brief Compute the per-species resistive overlay vector field added to
     *        Ohm's-law E alongside the existing +eta_global * J term:
     *
     *           overlay_d(i,j,k) = Sigma_s [ eta_s_per * rho_s * rho / rho_sum
     *                                       * (V_s_d - V_e_d) ]
     *
     *        summed over charged species that have a per-species
     *        resistivity parser registered. The total friction term that
     *        Ohm's law sees becomes
     *
     *           E_friction = +eta_global * J + overlay
     *                      = +e * Sigma_s Z_s * (eta_global + eta_s_per) * n_s
     *                                       * (V_s - V_e)
     *
     *        i.e. effectively eta_s_eff = eta_global + eta_s_per per species.
     *
     *        Output multifabs are at the Jx/Jy/Jz staggering (== E
     *        component staggering). When m_has_per_species_eta is false
     *        the overlay is set to zero and the function returns early
     *        (single-eta behaviour is exactly recovered when callers add
     *        a field of zeros to E).
     *
     *        Computed once per step (from HybridPICEvolveFields, into the
     *        registered hybrid_eta_overlay_fp fields) and held frozen across
     *        the subcycled E-solves, which only read it. Its inputs (Vs_fp,
     *        Ve_fp, rho_fp_{spec}, T_e) are per-step quantities anyway; only
     *        the |J| and |B| parser arguments would vary per substep, and
     *        those are frozen along with the rest. Caller passes
     *        pre-allocated Yee-staggered MultiFabs (one per direction) sized
     *        to match Jx/Jy/Jz at this level.
     */
    void ComputeResistiveOverlay (
        int lev,
        amrex::MultiFab & overlay_x,
        amrex::MultiFab & overlay_y,
        amrex::MultiFab & overlay_z) const;

    /**
     * \brief pc-mode corrector: single transport T_e^n -> T_e^{n+1} using the
     *        mid-step velocity V_e^{n+1/2}(J_i^{n+1/2}, B^{n+1/2}, rho^{n+1/2}),
     *        a two-stage midpoint push and Strang-split sources.
     *
     * Must be called from HybridPICEvolveFields AFTER the first B half-push
     * (so B^{n+1/2} is in the register) and BEFORE the rho^{n+1/2} LinComb
     * (so hybrid_rho_fp_temp still holds rho^n for the K_e / N_e load).
     * Recomputes J_plasma from B^{n+1/2}. The first B half-push runs with the
     * previous step's Pe^n (already in the register); the emitted Pe^{n+1}
     * serves the second half-push and the final E-solve.
     */
    void AdvanceElectronEnergyQDSMC_PC (amrex::Real dt) const;

    /**
     * \brief leapfrog only: overwrite the Pe register with the linearly
     *        extrapolated integer-time value
     *        Pe^{n+1} = 3/2 Pe^{n+1/2} - 1/2 Pe^{n-1/2} (first step:
     *        2 Pe^{1/2} - Pe^0), staged by AdvanceElectronEnergyQDSMC into
     *        hybrid_qdsmc_pe_ext_fp. Call right before the final E-solve;
     *        mirrors the existing J_i^{n+1} extrapolation.
     */
    void ApplyQdsmcPeExtrapolation () const;

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active). This is live
     *  controller state: the end-of-half update ratchets it and, under pure
     *  RK4 operation, never relaxes it (attempts per half always equal the
     *  current count, so the relax blend is a fixed point). Checkpoints
     *  persist the live value (hybrid_substeps.dat) and restarts restore it;
     *  re-seeding from the inputs value against a mid-stroke field state
     *  manufactures a NaN-recovery on the first post-restart substep and
     *  permanently over-ratchets the count (measured 2.8x B-field cost). */
    int m_substeps = 10;

    /** Restore m_substeps from the checkpoint on restart (default). Set to 0
     *  to keep the inputs value instead (controller experiments, e.g. seed
     *  probes). Legacy checkpoints without hybrid_substeps.dat always keep
     *  the inputs value. */
    bool m_substeps_from_checkpoint = true;

    /** Last B-substep count reported to stdout by the controller witness
     *  line; -1 until the first end-of-half adjustment seeds it. Reporting
     *  is throttled to >= 15% changes since the last report, so the RK4
     *  fixed point (count never moves) stays silent while an RKF45 decay
     *  or a rescue jump is witnessed in the log. */
    int m_substeps_last_reported = -1;

    /** Sponge-cap layer width [m] at each z end (see ApplySpongeLayerBYee;
     *  0 = off, bit-identical legacy path). Knob names keep the je_sponge_*
     *  spelling of the je-form lineage so matched decks drive both forms
     *  with identical inputs. Input: je_sponge_width. */
    amrex::Real m_je_sponge_width = 0.0;
    /** Sponge damping exponent ln(1/f_s) (default 4; validated range 3-6:
     *  significantly higher reflects off the layer's inner edge,
     *  significantly lower under-absorbs). Input: je_sponge_lnf. */
    amrex::Real m_je_sponge_lnf = 4.0;
    /** Sponge matching speed v_w [m/s]. REQUIRED > 0 when the layer is
     *  active on this form (no publishable fallback exists: the fastest
     *  retained signal at the caps is the whistler branch at cap-band
     *  parameters — deck-set from cap parameters; over-setting is the safe
     *  direction, slower content just sees more damping per crossing,
     *  which the ramp tolerates). With the adaptive path armed
     *  (je_sponge_vw_tau > 0) this becomes the SEED/fallback value only:
     *  the operating speeds are measured per cap every accepted substep.
     *  Input: je_sponge_vw. */
    amrex::Real m_je_sponge_vw = 0.0;

    /** EMA time constant [s] of the adaptive branch-speed measurement.
     *  0 (default) = LEGACY path: the single fixed je_sponge_vw with a
     *  linear ramp, bit-identical to the pre-adaptive code. > 0 arms the
     *  adaptive path, where both branch speeds are measured per cap on
     *  every accepted substep and low-passed with this constant so the
     *  layer's impedance does not chatter (a time-varying impedance
     *  scatters in its own right). Input: je_sponge_vw_tau. */
    amrex::Real m_je_sponge_vw_tau = 0.0;
    /** Profile selector within the adaptive path: 1 (default) = TWO-TIER
     *  (see ApplySpongeLayerBYee), 0 = single-speed linear ramp matched
     *  to the fast branch, retained as an experimental arm. Ignored on
     *  the legacy path. Input: je_sponge_grade. */
    int m_je_sponge_grade = 1;
    /** Inner (slow-matched) tier width as a fraction of je_sponge_width,
     *  default 0.5. 1.0 degenerates EXACTLY to a single slow-matched
     *  linear ramp and 0.0 to a single fast-matched one — both are free
     *  validation cases. Input: je_sponge_tier_frac. */
    amrex::Real m_je_sponge_tier_frac = 0.5;
    /** Multiplier applied to BOTH measured branch speeds (default 1.0 =
     *  pure measured matching), so the operating point can be moved off
     *  the raw measurement without a rebuild. Input: je_sponge_vw_scale. */
    amrex::Real m_je_sponge_vw_scale = 1.0;
    /** Optional clamps [m/s] on the EMA'd speeds (0 = unset). Safety
     *  rails against a transient measurement excursion; they do not
     *  participate in normal operation. Inputs: je_sponge_vw_min,
     *  je_sponge_vw_max. */
    amrex::Real m_je_sponge_vw_min = 0.0;
    amrex::Real m_je_sponge_vw_max = 0.0;
    /** Ceiling on the measured v_fast/v_slow ratio (default 100), so a
     *  pathological measurement cannot mint an absurd tier dynamic
     *  range. The physical ratio is a stable number wherever the density
     *  floor governs the cap band (see ApplySpongeLayerBYee): it reduces
     *  to k_max sqrt(m_i/(mu0 n_floor))/e, independent of local B and
     *  local density — measured ~23 on the production anchor. A ratio
     *  seen DRIFTING in the witness print is therefore a red flag that
     *  the floor assumption has broken, not routine variation.
     *  Input: je_sponge_grade_max_ratio. */
    amrex::Real m_je_sponge_grade_max_ratio = 100.0;

    /** Sponge reference-tracking time constant [s] (0 = frozen reference,
     *  bit-identical to the original behavior). A frozen reference under
     *  an evolving background becomes a stale pin: as the drive carries
     *  the cap-region state away from the seed, the layer forces against
     *  the growing difference and radiates the mismatch inward (measured
     *  on the je full-stroke panel: Alfvenic inward fronts from
     *  mid-stroke, spurious cap-region wave field at 0.4-0.65 of the
     *  interior level). With tau > 0 the reference low-passes toward the
     *  damped state (alpha = dt_sub/tau per accepted substep), following
     *  the slow background while still absorbing wave-timescale content;
     *  it also re-converges a restart-reseeded reference within tau.
     *  Input: je_sponge_track_tau. */
    amrex::Real m_je_sponge_track_tau = 0.0;

    /** Interval (in steps) of the per-species temperature moment deposit
     *  (DepositTemperatures — measured ~20% of a production step). Its
     *  consumers are collision-rate-class quantities (Q_ei relaxation,
     *  the nu_ei parser, ion heating) whose inputs evolve on collisional
     *  timescales, orders of magnitude slower than dt — a deposit held
     *  for N steps changes those rates at the 1e-4-class level while
     *  removing (N-1)/N of the deposit cost. The T_<species> fields
     *  persist between deposits; the deposit always runs on the FIRST
     *  call after boot or restart regardless of the interval (the fields
     *  are otherwise unseeded — restart does not round-trip them).
     *  Diagnostics requesting T_<species> read values up to N-1 steps
     *  stale. 1 = every step (bit-identical legacy path). Input:
     *  temperature_deposition_interval. */
    int m_t_deposit_interval = 1;
    /** First-deposit latch of the interval gate above (set by the
     *  HybridPICDepositRhoAndJ call site). */
    bool m_t_deposited_once = false;

    /** Intervals to use RKF45 integrator and to update substeps parameter */
    ablastr::utils::text::IntervalsParser m_rkf45_intervals;
    /** Relative tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_rtol = 1.e-4;
    /** Absolute tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_atol = 1.e-8;
    /** Safety factor for RKF45 step-size adjustment */
    amrex::Real m_substep_safety = 0.9;
    /** Maximum step-size growth factor per accepted step */
    amrex::Real m_substep_max_growth = 5.0;
    /** Maximum number of substep attempts (accepted + rejected) before abort */
    int m_max_substep_attempts = 250;
    /** Run the per-substep B-field finiteness check (RK4 path) only on every
     *  N-th substep attempt, plus always on the final substep of each
     *  half-step and on the attempt immediately following a failed attempt.
     *  Default 1 checks every attempt (the historical behavior).
     *  CAVEAT: with values > 1, a substep whose check is skipped can be
     *  consumed by any same-step corrector-style consumer before the next
     *  check runs. On a plain run the eventual check aborts (terminal
     *  either way); under an external corrector loop that treats a check
     *  failure as recoverable (e.g. a circuit-coupling engine), the skip
     *  forfeits the retry and accepts an unverified push -- keep the
     *  default 1 on coupled runs. */
    int m_substep_finite_check_interval = 1;

    bool m_holmstrom_vacuum_region = false;

    /** Smooth transition width of the Holmstrom vacuum gate, as a fraction
     * of n_floor. With width w > 0 the Hall + grad-Pe part of the Ohm's-law
     * E is multiplied by 0.5*(1 + tanh((rho - rho_floor)/(w*rho_floor)))
     * instead of switching off discontinuously below the floor: the binary
     * branch makes the implicit residual discontinuous in the state where
     * cells straddle the gate (Newton limit cycles, grid-scale E jumps at
     * the plasma edge). 0 (default) keeps the legacy hard branch. */
    amrex::Real m_holmstrom_transition_width = 0.0;

    /** Radial confinement of the Holmstrom vacuum gate (RZ only). With
     * radius R > 0 the gate is ELIGIBLE only at r < R: sub-floor cells
     * beyond R keep the ungated legacy Ohm's-law E exactly. Motivation:
     * a choked mirror-throat axis channel can sit below n_floor while
     * carrying peak |B|, where the floor-clamped Hall term produces
     * unphysical E spikes that feed grid-band B noise via Faraday and
     * kick through-axis particles; confining the gate to that channel
     * treats the measured pathology while leaving the outer halo -- and
     * the gate's historically problematic plasma/vacuum seam there --
     * untouched. 0 (default) keeps the gate global when armed. */
    amrex::Real m_holmstrom_axis_radius = 0.0;

    /** Radial rolloff width [m] of the axis confinement mask. 0 (default)
     * is a hard radial edge at the confinement radius; w > 0 rolls the
     * mask off as 0.5*(1 - tanh((r - R)/w)) so the gate's own radial
     * boundary does not introduce a new hard seam in the sub-floor halo. */
    amrex::Real m_holmstrom_axis_rolloff = 0.0;

    /** Independent Ohm's-law term controls: whether the Hall contribution
     * (the curl-B total current in the (J - J_i) x B / (e n) term; with it
     * off the term reduces to the ideal -u_i x B motional field) and the
     * electron-pressure gradient enter the E-field solve. Both default on;
     * turning them off makes e.g. a species-free resistive-vacuum deck
     * exactly linear (used by the open-boundary Green's-function tests). */
    bool m_include_hall_term = true;
    bool m_include_electron_pressure_term = true;

    /** Electron temperature: read from the input in eV, converted to
     * k_B T_e in J at the end of ReadParameters. */
    amrex::Real m_elec_temp;
    /** Reference electron density */
    amrex::Real m_n0_ref = 1.0;
    /** Electron pressure scaling exponent */
    amrex::Real m_gamma = 5.0/3.0;

    /** Master gate for the electron-energy equation. When true, K_e is
     * transported each step by QDSMC fictitious particles advecting with
     * V_e, then T_e is recovered from K_e and n_e via the polytropic
     * relation, then the Joule-heating source (if enabled) is added,
     * then Pe = n_e k_B T_e is emitted for the Ohm's-law E-solve. When
     * false (default), Pe is computed from the algebraic adiabatic closure
     * (FillElectronPressureMF) -- no temperature evolution. */
    bool m_solve_electron_energy_equation = false;

    /** Darwin (magnetoinductive) field split for the theta-implicit hybrid
     *  solver: the electric field is decomposed as E = E_T + E_L with the
     *  longitudinal part E_L = grad(phi) obtained from the instantaneous
     *  elliptic constraint
     *      laplacian(phi) = div( -grad(Pe) / (q_e n_e) ),
     *  i.e. the Helmholtz-longitudinal projection of the ambipolar pressure
     *  field, and the transverse part advanced through the magnetic vector
     *  potential, B = B_static + curl(A), A^{n+1} = A^n - dt E_T^{n+theta}.
     *  This removes the radiative light-wave branch while retaining
     *  Hall/whistler physics (Hewett & Nielson, J. Comput. Phys. 29, 219
     *  (1978); Hewett, Comput. Phys. Comm. 84, 243 (1994)). Consumed by
     *  ThetaImplicitHybrid. */
    bool m_darwin = false;
    amrex::Real m_darwin_poisson_rtol = 1.0e-10;
    amrex::Real m_darwin_poisson_atol = 0.0;
    int m_darwin_poisson_max_iters = 200;
    int m_darwin_poisson_verbosity = 0;

    /** Electron inertia in the generalized Ohm's law (per the implicit-PIC
     *  formulation of Angus et al.): the Je-form of the material derivative
     *  -(m_e_eff/e) D u_e/Dt of the electron fluid velocity u_e = -Je/rho,
     *      E_inertial = +(m_e_eff/(e rho)) [ dJe/dt - (Je/rho) drho/dt
     *                                        - (Je.grad)(Je/rho) ]
     *  (cold-electron limit: E = (m_e/(e^2 n_e)) dJe/dt, which fixes the
     *  sign and the 1/rho), assembled at nodes each residual evaluation and
     *  added to the Ohm electric field. Rolls the whistler branch over at
     *  the (effective) electron skin depth d_e = c/omega_pe(m_e_eff): with
     *  reduced_electron_mass_ratio = R, m_e_eff = m_lightest_ion / R
     *  (R = 0 or unset: the physical electron mass), so R tunes where the
     *  electron-scale dispersion and damping act relative to the grid.
     *  dJe/dt uses the second-order three-point stencil centered at the
     *  theta stage where Ohm's law is imposed (classic BDF2 at theta = 1,
     *  exactly the two-point midpoint form at theta = 1/2; two-point until
     *  two history levels exist or always when electron_inertia_bdf2 = 0)
     *  with the theta-extrapolated iterate Je^{n+1}; the term is zeroed
     *  only where rho <= 0 (density-floored cells keep their inertia --
     *  that is what regularizes the floored-halo response). */
    bool m_include_electron_inertia = false;
    amrex::Real m_reduced_electron_mass_ratio = 0.0;
    bool m_electron_inertia_bdf2 = true;
    /** Effective electron mass, resolved lazily on first use (needs the
     *  ion species list). */
    amrex::Real m_electron_inertia_mass = 0.0;
    /** NOTE on the theta-stage save feeding the history rotation: the
     *  last non-Jacobian residual evaluation is the accepted iterate for
     *  every converged Newton/Picard exit (convergence is checked on a
     *  fresh residual). A rolled-back final line-search iteration or a
     *  max-iteration exit can leave the save at a nearby trial state
     *  (error O(alpha dU)); reassembling at the accepted state in the
     *  rotation is a possible refinement.
     *
     *  One-time Je-history seeding from the first evaluated state: a
     *  quiescent plasma's pre-history is its initial (noisy) electron
     *  current, NOT zero -- seeding both levels with Je(t=0) makes the
     *  first BDF2 derivative exactly zero instead of a deposit-noise
     *  impulse of order Je/dt. */
    bool m_inertia_history_initialized = false;
    /** Number of genuine per-step history levels accumulated (seeding
     *  counts as one). BDF2 engages only at >= 2 -- the first step after
     *  seeding uses the two-point form (a seeded-equal history would
     *  overweight a driven first step by 1.5x under BDF2). */
    int m_inertia_history_levels = 0;
    /** Per-step latch for the frozen rho^n capture (see
     *  "hybrid_rho_n_frozen" in ComputeElectronInertiaNodal): set on the
     *  first non-Jacobian evaluation of each step -- the only evaluation
     *  whose rho_fp component 0 is a true step-start-position deposit --
     *  and cleared by RotateElectronInertiaHistory at the end of the
     *  step, alongside the equally per-step-frozen Je histories. */
    bool m_inertia_rho_n_captured = false;

    /** Assemble E_inertial on the nodal scratch "hybrid_E_inertial_nodal"
     *  from the current theta-stage state (see m_include_electron_inertia).
     */
    void ComputeElectronInertiaNodal (amrex::Real a_theta, amrex::Real a_dt,
                                      bool a_from_jacobian);

    /** Rotate the per-step nodal Je history: nm1 <- n, then
     *  n <- (Je^theta - (1-theta) Je^n)/theta from the converged
     *  theta-stage assembly saved by the last non-Jacobian evaluation --
     *  the same extrapolation family as every other end-of-step field, so
     *  the derivative never differences across deposit-noise realizations
     *  from different assembly conventions. Called from the implicit
     *  solver's end-of-step update. */
    void RotateElectronInertiaHistory (amrex::Real a_theta);

    /** Vacuum vector-potential recovery (Darwin path only): every application
     *  replaces the evolved A in low-density cells with the magnetostatic
     *  solution driven by the retained (plasma) currents and the boundary
     *  values of A, making the vacuum-region field response instantaneous
     *  instead of limited by the density-floored halo's Ohm dynamics
     *  (the vacuum-field limit of Hewett, J. Comput. Phys. 38, 378 (1980)).
     *  Implemented in correction form: with J_imp = curl(curl A)/mu0 the
     *  field-implied current, solve
     *      laplacian(dA) = +mu0 * (J_imp restricted to masked cells),
     *  with homogeneous Dirichlet values on non-periodic faces and inside
     *  embedded conductors (the evolved A already carries the external-A
     *  pin and the conductor gauge zero, so the recovered field shares
     *  them), then A += dA in masked cells. An empty mask gives dA = 0
     *  identically: the operation is exactly the identity there.
     *  Masks (from the step-entry rho^n, component 0 of rho_fp):
     *  "vacuum" = rho < q_e n_floor (includes true vacuum, default);
     *  "transition" = 0 < rho < q_e n_floor; "global" = everywhere.
     *  Cadence: "half" applies inside every residual evaluation at the
     *  theta-stage field (nonlinear elimination of the vacuum dynamics)
     *  plus once at the end-of-step state; "full" applies at the
     *  end-of-step state only. */
    bool m_darwin_vacuum_recovery = false;
    std::string m_darwin_vacuum_recovery_mask = "vacuum";
    std::string m_darwin_vacuum_recovery_cadence = "half";
    /** Which components are recovered. "flux" (default) recovers only the
     *  out-of-plane component (A_theta in RZ, A_y in 2D) that carries the
     *  boundary-driven flux; in 3D it is equivalent to "all". "all"
     *  recovers every component — CAUTION: in RZ the in-plane (A_r, A_z)
     *  corrections are discretely inconsistent with the edge curl-curl
     *  source near the axis (the 1/r stencil factors differ at O(1) in the
     *  first rows), which pumps an axis-localized instability in the
     *  B_theta-carrying content. */
    std::string m_darwin_vacuum_recovery_components = "flux";
    /** Density fraction (of q_e*n_floor) below which the recovery masks
     *  apply. The Ohm's-law density floor can sit well inside real plasma
     *  (it regularizes the stiff whistler response at the plasma edge);
     *  the vacuum treatment must not: with the default 1 the masks follow
     *  the floor, smaller values restrict the recovery (and especially the
     *  Faraday band-E overwrite, which fights valid Ohm physics if applied
     *  to genuine plasma) to the true low-density region. */
    amrex::Real m_darwin_vacuum_recovery_density_fraction = 1.0;
    /** Finite response time of the vacuum recovery (seconds; 0 = replace
     *  instantly). The band A relaxes toward the magnetostatic solution,
     *  A += (1 - exp(-dt_eff/tau)) * dA, capping the reconstructed band
     *  EMF at the correction rate dA/tau instead of the one-interval jump
     *  dA/(theta dt) -- the physical analog of the halo's finite (whistler)
     *  response time that the instantaneous elliptic limit idealizes away.
     *  Slow drives (ramp time >> tau) are delivered unchanged. */
    amrex::Real m_darwin_vacrec_relax_time = 0.0;
    amrex::Real m_darwin_vacrec_rtol = 1.0e-8;
    amrex::Real m_darwin_vacrec_atol = 0.0;
    int m_darwin_vacrec_max_iters = 200;
    int m_darwin_vacrec_verbosity = 0;
    /** Recompute the recovery (and evaluate the band field from the live
     *  recovered A) inside matrix-free Jacobian probe evaluations, instead
     *  of reusing the frozen correction and stored target. Initialized
     *  from the WARPX_VACREC_LIVE_PROBES environment variable; forced on
     *  by solvers that iterate the external drive inside the residual
     *  (with per-evaluation scale changes a frozen recovery makes the
     *  probed map inconsistent with the iterate map). Run with a tight
     *  darwin_vacuum_recovery_relative_tolerance (<= 1e-10). */
    bool m_vacuum_recovery_live_probes = false;

    /** Apply the vacuum vector-potential recovery to "hybrid_A_fp" in place
     *  (see m_darwin_vacuum_recovery). Uses Bfield_fp as curl scratch: the
     *  caller must rebuild B = B_static + curl A afterwards (both call
     *  sites do so anyway) and re-apply the A boundary values.
     */
    void ComputeVacuumARecovery (bool a_from_jacobian,
                                 amrex::Real a_dt_eff);

    /** Overwrite the recovered components of Efield_fp in masked (vacuum)
     *  cells with the Faraday-consistent field of the recovered vector
     *  potential history, E = -(A - A_old)/a_dt_eff (+ E_L when
     *  a_add_E_long, for call sites where Efield_fp holds the full field).
     *  The generalized Ohm's law has no validity in vacuum: without this,
     *  the flux delivered through A never materializes as the inductive
     *  electric field (particles miss the drive) and the stored E carries
     *  floored Hall noise; the resulting dA/dt != -E_T inconsistency also
     *  drives a secular instability of the band content. Only recovered
     *  components are overwritten -- for raw components the relation holds
     *  identically and the overwrite would degenerate the residual.
     */
    void ApplyVacuumFaradayE (amrex::Real a_dt_eff, bool a_add_E_long,
                              bool a_from_jacobian, bool a_bdf2);

    /** Solve the Darwin longitudinal-field constraint with the given charge
     *  density and the current hybrid_electron_pressure_fp, writing
     *  E_L = grad(phi) into the "hybrid_E_long_fp" fields (E staggering).
     */
    void ComputeDarwinELong (ablastr::fields::MultiLevelScalarField const& rho,
                             amrex::Real t);
    /** Time-advance scheme for the QDSMC energy step (bake-off switch,
     *  hybrid_pic_model.qdsmc_time_advance = euler | leapfrog | pc).
     *  See AdvanceElectronEnergyQDSMC for the scheme definitions.
     *  Default: pc — selected 2026-08-04 after the bake-off (second order
     *  in dt, best stiff-Joule energy budget, integer-time state). euler is
     *  the #6982 first-order control. */
    enum struct QdsmcTimeAdvance { Euler, Leapfrog, PC };
    QdsmcTimeAdvance m_qdsmc_time_advance = QdsmcTimeAdvance::PC;

    /** Half-gradient-corrected QDSMC deposit (Thrust B / B1): markers carry
     *  limited MC slopes of their content and the scatter deposits
     *  w * (A + 1/2 G . (x_dest - x_p)). Conservation is exact (the
     *  correction sums to zero by the hat's first-moment property); the
     *  remap becomes exact through quadratics, breaking the first-order
     *  remap-diffusion cap. Disable (with qdsmc_time_advance = euler) to
     *  recover #6982 bit-for-bit. Input: hybrid_pic_model.qdsmc_gradient_deposit. */
    bool m_qdsmc_gradient_deposit = true;

    /** One entropy-transport pass on level lev over dt_adv: K_e init ->
     *  SetV -> SetK [-> midpoint velocity re-gather] -> PushX -> deposits ->
     *  T_e recovery. V_e must already be filled (QDSMCInitializeUe) and
     *  sources are NOT applied here. */
    void QdsmcTransportOnce (int lev, amrex::Real dt_adv, bool midpoint) const;

    /** Operator-split energy sources over dt_src: Joule heating (+ the
     *  Te-threshold redirect staging), the per-species T_i collapse, the
     *  Q_ei electron sink and the conjugate/redirected stochastic ion
     *  heating. Factored out of the #6982 advance so Strang halves can call
     *  it twice. When fill_te_ghosts is true (the Strang paths), T_e's
     *  ghost cells are refreshed afterward (the source kernels write valid
     *  cells only and the subsequent K_e init reads ghosts). */
    void ApplyQdsmcEnergySources (int lev, amrex::Real dt_src,
                                  bool fill_te_ghosts) const;

    /** One Ito tensor-conduction substep on level lev over dt_c (Thrust C),
     *  operating on the energy density u = 3/2 n_e k_B T_e (NOT on the
     *  entropy K_e: conduction is not adiabatic, and mapping the Fourier
     *  flux into K-space drags in n-gradient corrections). Each owned node
     *  spawns ephemeral quiet Gauss--Hermite daughters along the
     *  field-aligned eigendirections of D = chi_par b b + chi_perp (I - bb),
     *  displaced by the Ito drift [div D + D . grad ln n_e] dt_c plus the
     *  quadrature offsets x_q sqrt(2 chi dt_c), and hat-deposits w_q * u
     *  with the same half-gradient correction as the advection markers
     *  (source-node MC slopes of u; without it the hat remap adds a
     *  chi_num ~ dx^2/(12 dt) floor -- the conduction analogue of the
     *  cell-center-home diffusion). T_e is then recovered as
     *  u_dep / (3/2 k_B n_e) and its ghosts refreshed. Positivity is
     *  automatic (positive quadrature weights, positive u); Sigma(u) is
     *  conserved to round-off (the correction sums to zero per daughter).
     *  No-op unless m_include_thermal_conduction.
     *
     *  use_rho_new selects rho_fp (post-transport Strang half) vs
     *  hybrid_rho_fp_temp (pre-transport half) for the u build, the
     *  grad ln n_e drift and the recovery, so u <-> T_e stays consistent
     *  within the substep. b and chi are evaluated from whatever B is in
     *  the register (B^{n+1/2} in the pc driver). */
    void QdsmcConductionOnce (int lev, amrex::Real dt_c, bool use_rho_new) const;

    /** Grid finite-difference conduction operator (qdsmc_conduction_operator
     *  = fd), dispatched from QdsmcConductionOnce in place of the SDE
     *  daughter/remap forms. Conservative face-flux discretization of
     *  du/dt = div(kappa_tensor grad T_e), u = (3/2) n_e k_B T_e, after
     *  Chacon, Hamilton & Krasheninnikova, CPC 313 (2025) 109646:
     *  co-derivative fluxes are centered (2nd order compact / 4th order
     *  C-interpolated A-stencil node fluxes); cross-derivative fluxes are
     *  recast as advective fluxes F = v* T_face with the fictitious face
     *  velocity v* = Xi_gh (d_h T)/T_face (Du Toit et al., CPC 228 (2018)
     *  61) and T_face reconstructed by the SMART monotonicity limiter --
     *  the limited operator satisfies a discrete maximum principle, so
     *  conduction cannot create new T_e extrema even where b-hat is noise
     *  (the reconnection-null blowup class). Subcycled with SSP-RK2 stages
     *  (a convex combination of forward-Euler stages -- monotone at the
     *  same CFL, time error O(dt_sub^2) ~ O(dx^4)) to the Gershgorin
     *  stability bound inside the same Strang half-steps; the
     *  kappa parsers, free-streaming limiter, iso_B blend, closed floor
     *  faces and wall BCs/tallies are shared with the SDE path. No hop
     *  cap and no vacuum fast front (neither concept applies).
     *
     *  In RZ geometry the operator carries the paper's curvilinear
     *  J Xi^ij form with J = r: radial node fluxes are J-folded before
     *  the face composites (preserving their design order) and the
     *  radial divergence divides by the dual-cell measure
     *  Vr = int r dr, which is regular at the axis (q_r(0) = 0 by
     *  symmetry, Vr = dr^2/8 for the axis node); z faces are metric-
     *  free. m = 0 fields only (b-hat is built from the m = 0 mode). */
    void QdsmcConductionOnceFD (int lev, amrex::Real dt_c, bool use_rho_new) const;

    /** Grid-transport arm of the energy-equation bake-off jig
     *  (qdsmc_transport_operator = grid): advances the entropy pair
     *  y = {n_e, K n_e} in conservative flux form dy/dt = -div(y V_e)
     *  with SMART-limited upwind face values and the shared adaptive RK
     *  integrator (V_e frozen over the transport slot, like the marker
     *  push). Writes the marker-deposit convention (entropy_fp = K n_e,
     *  weights = n_e) and calls QDSMCUpdateTe -- identical recovery, so
     *  the bake-off isolates the transport operator alone.
     *
     *  In RZ geometry the operator carries the same curvilinear form as
     *  QdsmcConductionOnceFD: radial face fluxes are J = r folded
     *  (analytic face radius), the radial divergence divides by the
     *  dual-cell measure Vr = int r dr (dr^2/8 at the regular axis node,
     *  whose inner face is absent), z faces are metric-free, and the
     *  advective CFL carries the largest existing face's r_face/Vr
     *  amplification. Faces beyond non-periodic walls do not exist and
     *  the CFL reduce counts open nodes only (the bound mirrors the flux
     *  structure). */
    void QdsmcTransportOnceGrid (int lev, amrex::Real dt_adv) const;

    /** leapfrog self-start: false until the first advance has run (which
     *  advances by dt/2 to move T_e^0 to the half level). */
    mutable bool m_qdsmc_leapfrog_started = false;
    /** leapfrog: false until hybrid_qdsmc_pe_prev_fp holds a valid previous
     *  Pe level (Pe^0 on the first step, Pe^{n-1/2} afterward). */
    mutable bool m_qdsmc_pe_prev_valid = false;

    /** Resistive electron-heating source on T_e (see Phys. Plasmas 31, 012902 (2024), Eq. 12),
     * computed per cell as
     *     S_e = Sigma_s nu_{s,e} n_s m_s |V_s - V_e|^2,
     *         nu_{s,e} = Z_s e^2 eta n_e / m_s,
     * from the gridded Vs_fp_s, Ve_fp, rho_fp(_s) fields and the
     * Ohm's-law eta parser. Single bool: on or off. Reduces exactly to
     * eta J^2 in single species; resolves inter-species drift-dispersion
     * heating in multispecies. Orthogonal to whether HybridResistiveDrag
     * is registered on ions (that controls particle dynamics, not the
     * T_e source). Only consulted when m_solve_electron_energy_equation
     * is also true. */
    bool m_include_joule_heating = false;

    /** Keep -grad Pe/(e n) in the Faraday-solve E (the Biermann battery
     *  term (grad Pe x grad n)/(e n^2), which the closure-era grad-Pe drop
     *  discards; nonzero whenever Te decouples from n, i.e. with the
     *  electron energy equation). Input include_biermann_battery,
     *  default off. */
    bool m_include_biermann_battery = false;

    /** Te-threshold Joule redirection. When active, the eta*J^2 Joule source in
     * QDSMCAddJouleHeating heats electrons in cells where Te < m_joule_redirect_Te_eV
     * and is instead deposited to the ions (as stochastic pure-diffusion heating in
     * QDSMCApplyIonHeating) where Te >= m_joule_redirect_Te_eV. This caps electron
     * heating at the threshold and routes the rest to the ions, so Ti can exceed Te.
     * The redirected energy is bookkept per charged ion species. Requires
     * m_include_joule_heating. Default off. */
    bool m_joule_redirect_to_ions = false;
    amrex::Real m_joule_redirect_Te_eV = -1.0;   // threshold [eV]; < 0 -> off

    /** Guard-rail escape hatch (2026-08 liftoff runaway forensics): without a
     * relaxation channel the redirect's OU ion-heating operator has no drag
     * leg and runs as pure undamped diffusion, measured anti-stabilizing on
     * the liftoff slab (arm SR: ion-kick feedback -- unbounded Gaussian kicks
     * on the few macro-ions in near-floor cells spray current noise into the
     * eta_vac band, raising eta J^2, 10x hotter than its bit-identical
     * control within 600 steps of engaging). Default off: the redirect
     * ABORTS at init when electron_ion_relaxation_rate is not set. Input
     * joule_redirect_allow_undamped = 1 restores the old (undamped) behavior
     * for legacy control-arm reproduction only. */
    bool m_joule_redirect_allow_undamped = false;

    /** Per-particle redirect kick cap: clamp the redirect's per-cell OU kick
     * variance so the per-application redirected energy per ion satisfies
     * E_s <= f^2 kB T (T = the cell's deposited T_i when relaxation is on,
     * else the local T_e as the velocity scale), i.e.
     * sigma_redirect <= f * v_th = f * sqrt(kB T / m_i). The clipped
     * remainder is NOT re-deposited; it accumulates in the dropped-energy
     * tally (channel kick_cap). Input joule_redirect_kick_cap_vth_frac;
     * default < 0 -> off (back-compat). */
    amrex::Real m_joule_redirect_kick_cap_vth_frac = -1.0;

    /** Density gate on the redirect deposit: only cells with
     * n >= joule_redirect_n_min_factor * n_floor stage redirected energy
     * (matching the Joule kernel's own floor-gate philosophy); below the
     * gate the above-threshold source is dropped to the tally (channel
     * redirect_gate) instead of heating anything. Input
     * joule_redirect_n_min_factor (g ~ 2-4); default 0 -> off. */
    amrex::Real m_joule_redirect_n_min_factor = 0.0;

    /** Separate resistivity for the Joule HEATING source,
     * joule_heating_resistivity(rho,J,Te,t) [Ohm m] (Te in eV, so Spitzer
     * eta(Te) is expressible): the E-solve keeps plasma_resistivity
     * (including any numerical eta_vac vacuum-regularizer ramp needed for
     * stability), while the heating (and redirect) source is evaluated with
     * this physical eta -- removing the numerical band heating at its root
     * (the liftoff runaway ignited where the density-power-law eta was still
     * ~eta_vac at n just above n_floor). Unset -> the heating uses the
     * E-solve eta (back-compat). */
    bool m_has_heating_resistivity = false;
    std::string m_eta_heating_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_heating_resistivity_parser;
    amrex::ParserExecutor<4> m_eta_heating;

    /** Volumetric electron-energy sink, input qdsmc_energy_sink(rho,Te,B,t)
     * [W/m^3] (rho = total charge density [C/m^3], Te in eV, B = |B| in T,
     * t in s; armed by presence). S > 0 REMOVES energy: each source
     * application subtracts dTe = dt (gamma-1) S / (n_e kB) per cell,
     * mirroring the Joule source's dTe conversion with the opposite sign
     * and the same solver-floor density gate (no sink below the floor).
     * The per-cell decrement is clamped so Te lands no lower than the
     * conduction Te floor (m_cond_te_floor; plain positivity when unset);
     * the clamped-away energy accumulates in m_energy_sink_declined_J and
     * is printed on the joule_dropped_energy_print_interval cadence.
     * Intended for radiative-loss closures (e.g. n^2 sqrt(Te) or
     * n Te B^2 forms) expressed through rho, Te and |B|. */
    bool m_has_energy_sink = false;
    std::string m_energy_sink_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_energy_sink_parser;
    amrex::ParserExecutor<4> m_energy_sink;

    /** Cumulative energy declined by the sink's Te-floor clamp [J]
     * (globally reduced; mutable: written by the const source kernel). */
    mutable amrex::Real m_energy_sink_declined_J = 0.0;

    /** Apply the volumetric electron-energy sink to T_e over dt_src (see
     * m_has_energy_sink). Runs inside ApplyQdsmcEnergySources between the
     * Q_ei relaxation (6b) and the ion-heating operator (6c), so one call
     * site covers every explicit time-advance scheme. */
    void QDSMCApplyEnergySink (int lev, amrex::Real dt_src) const;

    /** True when a hybrid_electron_stopping collision is registered: fast
     * ions drag against the electron fluid (Goldston & Rutherford
     * slowing-down on the self-consistent n_e, T_e, u_e) and their
     * kinetic-energy loss heats T_e through the staging field below. Also
     * arms m_need_fluid_velocities (the drag target is Ve_fp). */
    bool m_has_electron_stopping = false;

    /** Per-cell staged fast-ion heating energy density [J/m^3], nodal like
     * T_e (one MultiFab per level, allocated lazily on first access). The
     * hybrid_electron_stopping collision scatter-adds each particle's
     * weighted kinetic-energy loss here during the collision stage of the
     * step; QDSMCApplyFastIonHeating consumes (and zeroes) it in the source
     * stage of the SAME step -- one-step staging, no energy lost. Mutable:
     * written through the const model pointer the collision holds. */
    mutable amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_fast_ion_heating_mf;

    /** Lazily allocate (zero-initialized, nodal on the T_e layout) and
     * return the fast-ion heating staging field for this level. */
    amrex::MultiFab & GetFastIonHeatingStaging (int lev) const;

    /** Cumulative fast-ion heating energy declined by the solver-floor gate
     * or the Te-floor clamp [J] (globally reduced; printed with the
     * dropped-energy channels as stopping_floor). */
    mutable amrex::Real m_stopping_declined_J = 0.0;

    /** Consume the staged fast-ion heating: SumBoundary the staging field
     * (the deposit writes each box's own copy of shared seam nodes, so the
     * nodal redundant copies hold partial sums that must be folded into
     * consistent totals first), add dTe = (gamma-1) E / (n_e kB) per cell
     * (the energy sink's conversion with the opposite sign, same
     * solver-floor gate and Te-floor clamp discipline), then zero the
     * staging field. Runs inside ApplyQdsmcEnergySources after the energy
     * sink; on the twice-per-step source schemes the second call finds the
     * staging already zeroed and no-ops. */
    void QDSMCApplyFastIonHeating (int lev) const;

    /** Independent density gate for the Joule heating source [m^-3]: cells
     * with n_e <= max(joule_heating_n_min, n_floor) receive no Joule heat
     * (the declined source energy goes to the dropped tally, channel
     * heat_gate), so heating can be restricted to the eta ~ eta_plasma
     * region without moving the solver floor. Default < 0 -> n_floor
     * (back-compat: gate inactive). */
    amrex::Real m_joule_heating_n_min = -1.0;

    /** Graceful failure mode for Te runaways: abort with a clear message
     * when max(Te) exceeds this ceiling [eV] after the energy sources
     * (>700 keV runaways otherwise end in SEGV). Checked at the end of
     * ApplyQdsmcEnergySources. Input Te_abort_threshold; default < 0 -> off. */
    amrex::Real m_te_abort_threshold_eV = -1.0;

    /** General Te limiter with ion shunt (the any-channel generalization of
     * #6982's Joule-scoped redirect): open-set cells with Te above this
     * threshold [eV] are capped to it and the excess energy density
     * 3/2 n_e kB (Te - Tcap) is staged into the per-species redirect_E for
     * the stochastic ion-heating operator -- per-ion energy
     * E_s = Z_s kB (Te - Tcap), which delivers the excess exactly (the
     * density-fraction split collapses to a Z-weighted per-ion energy).
     * Unlike the Joule redirect, this catches energy from EVERY channel
     * (transport/recovery, conduction, reconnection exhaust), which the
     * 2026-08 forensics showed is where the cascade energy actually
     * arrives. The existing protections apply downstream: the undamped
     * guard rail (requires the relaxation channel or
     * joule_redirect_allow_undamped), the per-particle kick cap, and the
     * redirect density gate (below the gate the excess is dropped to the
     * tally -- quarantined cells are never shunted). Cumulative staged
     * energy tallied in m_te_shunt_J. Input Te_shunt_threshold;
     * default < 0 -> off. */
    amrex::Real m_te_shunt_eV = -1.0;
    mutable amrex::Real m_te_shunt_J = 0.0;

    /** Cap open-set Te at m_te_shunt_eV and stage the excess into
     * redirect_E (per charged species, ordered as in QDSMCAddJouleHeating).
     * Runs inside ApplyQdsmcEnergySources between the Joule source and the
     * ion-heating operator. */
    void QDSMCShuntTeExcess (int lev, amrex::MultiFab * redirect_E) const;

    /** Cumulative dropped-Joule-energy tallies [J] (globally reduced), one
     * per decline channel: the joule_heating_n_min gate, the redirect
     * density gate, and the per-particle kick cap. Only accumulated when the
     * corresponding (opt-in) feature is armed; printed on rank 0 every
     * joule_dropped_energy_print_interval steps (0 disables the print).
     * Mutable: written by the const source kernels. */
    mutable amrex::Real m_joule_dropped_heat_gate_J = 0.0;
    mutable amrex::Real m_joule_dropped_redirect_gate_J = 0.0;
    mutable amrex::Real m_joule_dropped_kick_cap_J = 0.0;
    int m_joule_dropped_print_interval = 200;

    /** Quarantine/contamination tally (QUARANTINE_INSTRUMENT_PLAN.md).
     * Armed by qdsmc_contamination_n_boundary > 0 [m^-3]: nodes with
     * n <= boundary form the QUAR class, n > boundary the OPEN class (set
     * the boundary above n_floor to watch the eta_vac ignition band; at
     * n_floor the production closed-floor-faces rule makes the conduction
     * channels structurally zero, which is itself the verification). Every
     * conduction split-sweep face flux crossing QUAR -> OPEN accumulates
     * into a cumulative per-grid-axis tally [J] (axis attribution is done
     * deck-side: in a Bz slab the axial axis is kappa_par, the in-plane
     * axes kappa_perp); a fourth channel tallies the subset whose QUAR-side
     * node is FLOORED while vacuum_fast_front is on (the S0->SK conduit
     * hypothesis). The ion-kick channel tallies redirected energy staged in
     * QUAR cells by QDSMCApplyIonHeating. Split-fluxform path only: the
     * scatter/layer/unsplit control arms are not tapped (warned at init).
     * Printed with the joule_dropped_energy_print_interval cadence. */
    amrex::Real m_contam_n_boundary = -1.0;
    mutable amrex::Real m_contam_axis_J[3] = {0.0, 0.0, 0.0};
    mutable amrex::Real m_contam_fast_front_J = 0.0;
    mutable amrex::Real m_contam_kicks_J = 0.0;

    /** Cliff-limited entropy deposit (the option-b fix for the K-diffusion
     * heat pump; see QdsmcParticleContainer::CliffDeposit for the mechanism
     * and blend definition). Off by default; requires
     * qdsmc_gradient_deposit = 1. r1/r2 are |ln(n_dest/n_home)| blend
     * bounds: with the defaults the correction only engages across >~1.4x
     * single-hat density jumps and is fully isothermal beyond ~4x. */
    bool m_cliff_limited_deposit = false;
    amrex::Real m_cliff_deposit_r1 = 0.35;
    amrex::Real m_cliff_deposit_r2 = 1.4;

    /** Per-stage open-set energy budget (instrument;
     * qdsmc_energy_budget = 1, pc scheme only): brackets each stage of the
     * Strang advance C(dt/2) S(dt/2) A(dt) S(dt/2) C(dt/2) with class-summed
     * thermal energy U = 3/2 n_e k_B T_e (fixed rho^{n+1} weights, so each
     * stage's dU attributes that stage's T_e change), split into BULK
     * (n > max(qdsmc_contamination_n_boundary, n_floor)) and BAND
     * (n_floor < n <= boundary). The transport stage's dU contains the
     * advection AND the polytropic compression (pdV) signal -- the channel
     * the 2026-08 cascade forensics could not see (arm SH2 cascaded to keV
     * with its Joule source redirected and gated, implicating a non-Joule
     * channel). Cumulative [J], printed on the shared tally cadence. */
    bool m_energy_budget = false;
    mutable amrex::Real m_ebud_cond_bulk = 0.0, m_ebud_cond_band = 0.0;
    mutable amrex::Real m_ebud_src_bulk  = 0.0, m_ebud_src_band  = 0.0;
    mutable amrex::Real m_ebud_adv_bulk  = 0.0, m_ebud_adv_band  = 0.0;
    /** Compression part of the transport-stage dU: the exact polytropic
     * recovery term 3/2 kB n^{n+1} Te_pre [(n^{n+1}/n^n)^{gamma-1} - 1]
     * (both densities floored as in the K <-> Te conversions), summed per
     * class. The remainder adv - comp is the pure advection/remap residual
     * -- the physical-vs-numerical discriminator for the cascade (marker
     * CFL measured 0.017 cells/step at ignition, so motion is small). */
    mutable amrex::Real m_ebud_comp_bulk = 0.0, m_ebud_comp_band = 0.0;
    /** Energy-sink part of the source-stage dU (a sub-account of src: the
     * sink runs inside ApplyQdsmcEnergySources, whose brackets set the src
     * channel), bracketed separately when the instrument is armed. */
    mutable amrex::Real m_ebud_sink_bulk = 0.0, m_ebud_sink_band = 0.0;
    /** Fast-ion stopping heat part of the source-stage dU (a sub-account of
     * src, like the sink), bracketed separately when the instrument is
     * armed. */
    mutable amrex::Real m_ebud_stopping_bulk = 0.0, m_ebud_stopping_band = 0.0;

    /** Class-summed thermal energy {bulk, band} [J] for the energy budget
     * (nodal sum_unique; classes per the m_energy_budget doc). */
    std::array<amrex::Real, 2> QDSMCClassEnergy (int lev) const;

    /** Class-summed polytropic compression energy {bulk, band} [J] of the
     * current step's transport stage, from the pre-transport Te and the
     * rho^n -> rho^{n+1} density change (see m_ebud_comp_* doc). */
    std::array<amrex::Real, 2> QDSMCCompressionEnergy (
        int lev, amrex::MultiFab const & te_pre) const;

    /** Electron-ion thermal equilibration (Q_ei) on T_e:
     *   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),   dU_e/dt += -Q_ei,
     * cooling T_e toward each ion species' T_i at the rate given by the
     * electron_ion_relaxation_rate(rho,Te,Ti,t) parser, with the matching ion
     * heating deposited conservatively (QDSMCApplyIonHeating) so the exchange
     * conserves energy. Default off; only consulted when
     * m_solve_electron_energy_equation is on. */
    bool m_include_temperature_relaxation = false;

    /** Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) [1/s] used by
     * the Q_ei term (rho = charge density [C/m^3]; Te, Ti in eV; t in s). */
    std::string m_nu_ei_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_nu_ei_parser;
    amrex::ParserExecutor<4> m_nu_ei;

    /** Species excluded from the Q_ei exchange, input
     * electron_ion_relaxation_excluded_species (space-separated species
     * names, validated at input parsing). A fast non-thermal species'
     * deposited 'temperature' is a drift artifact, not a temperature, and
     * the relaxation would drag T_e toward it. Excluded species are skipped
     * by the electron-side sink, the T_i deposit build, the ion-side OU
     * heating operator (their redirect/shunt component is declined, not
     * delivered) and the automatic do_temperature_deposition arming at
     * species construction. Their charge still counts in the species-
     * fraction denominators: the excluded share simply does not relax. */
    std::vector<std::string> m_relaxation_excluded_species;

    /** True when species_name is listed in
     * m_relaxation_excluded_species. */
    [[nodiscard]] bool IsRelaxationExcluded (std::string const & species_name) const;

    /** Ito-process tensor thermal conduction on T_e (Thrust C):
     *   dU_e/dt += div(kappa grad T_e),
     *   kappa = kappa_par b b + kappa_perp (I - b b),  b = B/|B|,
     * applied as a split substep bracketing the entropy transport
     * (QdsmcConductionOnce). Enabled by specifying the parallel-conductivity
     * parser qdsmc_kappa_par(n,Te,t) (n [m^-3], Te [eV], t [s], kappa in
     * W/(m K)); kappa_perp has its own optional parser and defaults to 0.
     * Only consulted when m_solve_electron_energy_equation is on. */
    bool m_include_thermal_conduction = false;
    /** Isotropic-conduction options (b-hat noise at small |B|; see the
     * QdsmcConductionOnce kernel comment): qdsmc_conduction_isotropic
     * conducts at the cross-field rate everywhere (chi_par := chi_perp, the
     * field-aligned frame drops out exactly); qdsmc_conduction_iso_B [T]
     * blends chi_par -> chi_perp smoothly below that |B|
     * (s = B^2/(B^2+B_iso^2)) -- the surgical treatment of the
     * reconnection-null region. Defaults off (anisotropic). */
    bool m_cond_isotropic = false;
    amrex::Real m_cond_iso_B = -1.0;
    /** Conduction operator selector: 0 = sde (the QDSMC daughter/remap
     *  forms, production default), 1 = fd (grid finite-difference operator,
     *  see QdsmcConductionOnceFD). Input: qdsmc_conduction_operator =
     *  sde | fd. */
    int m_cond_operator = 0;
    /** FD operator spatial order: 2 (compact star co-derivative + SMART
     *  cross-advective fluxes; strict positivity) or 4 (Chacon Appendix-A
     *  C/A/D stencils; needed for physical-Braginskii anisotropy,
     *  effective eps ~ 1e-7 at liftoff). Interior stencils only at 4th
     *  order for now -- faces within 3 nodes of a non-periodic wall
     *  degrade to the 2nd-order compact form. Input:
     *  qdsmc_conduction_fd_order. */
    int m_cond_fd_order = 2;
    /** Global conduction diffusivity ceiling: when chi_max > 0 both
     *  kappa parsers are folded through a hard min at parser-build
     *  time with cap = 1.5 kB chi_max n, i.e. chi = kappa/(1.5 kB n)
     *  never exceeds chi_max at any (n, Te) the subcycle bound reads.
     *  Hard min rather than a soft-min: harmonic soft-mins read low
     *  near the cap (20% at cap/4), and the explicit path needs no
     *  C-infinity here. Off (0) is bit-identical to unset. Input:
     *  qdsmc_conduction_chi_max [m^2/s]. */
    amrex::Real m_cond_chi_max = 0.0;
    /** FD cross-flux transport reconstruction: 0 = none (unlimited QUICK,
     *  attribution control -- can overshoot), 1 = upwind1 (1st-order
     *  upwind, most diffusive), 2 = smart (Gaskell--Lau SMART, default:
     *  up to 3rd order, locally monotone face VALUES -- which does NOT
     *  bound the flux divergence under sharp field-line rotation, so
     *  composites of NVD-bounded faces can still mint new extrema),
     *  3 = sh (Sharma--Hammett, JCP 227 (2007): nested-limited
     *  transverse face gradient, MC inner / minmod outer -- the cross
     *  flux through a face vanishes when either adjacent node is a
     *  transverse local extremum, so cross transport cannot amplify
     *  extrema; 2nd-order in smooth regions), 4 = sh_minmod (minmod
     *  inner, maximally diffusive bracket arm). The sh modes replace
     *  the pseudo-advection recast of the cross terms entirely (see
     *  QdsmcFluxLimiters.H) and are conduction-only: combining them
     *  with qdsmc_transport_operator = grid aborts at parse. Input:
     *  qdsmc_conduction_fd_limiter = none | upwind1 | smart | sh |
     *  sh_minmod. */
    int m_cond_fd_limiter = 2;
    /** FD subcycle CFL fraction of the Gershgorin stability bound
     *  (sum_gh 2 |Xi_gh| dxi_g dxi_h, n-ratio weighted). Default 0.4:
     *  under SSP-RK2 the Nyquist amplification G = 1 + z + z^2/2 never
     *  flips sign but goes NEUTRAL (G -> 1, zero damping) at the
     *  stability edge, so deposit-noise checkerboards persist instead of
     *  damping -- the ledger-40 lesson from Helion's internal hybrid python prototype (alpha_sub <= 0.25, i.e.
     *  cfl <= 0.5 in this normalization) plus margin for Gershgorin
     *  being an upper bound. Do not run near 1. Input:
     *  qdsmc_conduction_fd_cfl. */
    amrex::Real m_cond_fd_cfl = 0.4;
    /** FD subcycle count ceiling per conduction call (attempts budget of
     *  the adaptive integrator); on hitting it the remaining fraction of
     *  dt_c is dropped with a warning (visible, not silent). Input:
     *  qdsmc_conduction_fd_max_subcycles. */
    int m_cond_fd_max_subcycles = 10000;
    /** FD subcycle time integrator (QdsmcRKIntegrator scheme): 0 = ssprk2
     *  (embedded 2(1) Heun/Euler pair -- monotone, keeps the operator's
     *  maximum principle; default), 1 = rkf45 (Fehlberg 4(5), the
     *  whistler-validated pair shared with BfieldEvolve; NOT SSP -- gate
     *  via the X-point stress arm before production use on monotone
     *  operators), 2 = rkl2 (Runge--Kutta--Legendre super-time-stepping:
     *  covers a parabolic subcycle count N in ~2 sqrt(N) stages instead
     *  of ~2 N RHS evaluations; stability-driven, NOT error-adaptive --
     *  rtol/atol unused; NOT SSP, same gating rule as rkf45; conduction
     *  only, the grid transport operator aborts on it). ssprk2/rkf45 are
     *  error-adaptive on top of the stability ceiling. Input:
     *  qdsmc_conduction_fd_time = ssprk2 | rkf45 | rkl2. */
    int m_cond_fd_time = 0;
    /** FD adaptive-integrator error tolerances (weighted max-norm,
     *  accept at err <= 1): relative and absolute [K]. Inputs:
     *  qdsmc_conduction_fd_rtol / _atol. */
    amrex::Real m_cond_fd_rtol = 1.0e-4;
    amrex::Real m_cond_fd_atol = 1.0e-3;
    /** Energy-equation TRANSPORT operator (bake-off jig): 0 = markers
     *  (the QDSMC particle transport, production default), 1 = grid
     *  (prototype: the entropy pair {n_e, K n_e} advanced in conservative
     *  SMART-limited flux form through QdsmcRKIntegrator -- "integrate
     *  the energy equation directly like the field equation"). The grid
     *  arm shares QDSMCInitializeKe and QDSMCUpdateTe, so ONLY the
     *  transport operator differs; it reuses the fd_time/fd_cfl/fd_rtol/
     *  fd_atol integrator knobs. EB + RZ abort. Input:
     *  qdsmc_transport_operator = markers | grid. */
    int m_qdsmc_transport_operator = 0;
    std::string m_kappa_par_expression = "0.0";
    std::string m_kappa_perp_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_kappa_par_parser;
    std::unique_ptr<amrex::Parser> m_kappa_perp_parser;
    amrex::ParserExecutor<3> m_kappa_par;
    amrex::ParserExecutor<3> m_kappa_perp;

    /* ------------------------------------------------------------------
     * PHYSICAL electron viscosity (Braginskii 1965) -- energy-BOOKED.
     *
     * This block is the physical Braginskii electron stress. Its
     * dissipation IS booked into the electron energy equation and IS
     * reported in its own reduced-diagnostic column (P_nu). It is NOT the
     * hyper-resistivity m_eta_h, which is a numerical stabiliser whose
     * dissipation is deliberately left unbooked -- the two share a grad^2
     * form on the current and differ entirely in the coefficient. See the
     * m_eta_h declaration and BraginskiiViscosity.H before touching either.
     * ------------------------------------------------------------------ */

    /** Electron viscosity model: 0 = none (default, term absent and
     *  bit-identical to unset), 1 = braginskii (built-in coefficients from
     *  n_e, T_e, |B| -- the intended production setting), 2 = parser (user
     *  nu_par/nu_perp expressions, for manufactured solutions and
     *  sensitivity scans). ANISOTROPIC in every case: the dissipative
     *  cross-field coefficient is ~(Omega_ce tau_e)^2 below the
     *  field-aligned one, so there is deliberately no isotropic mode --
     *  an isotropic electron viscosity is physically meaningless at these
     *  magnetisations. Input:
     *  qdsmc_viscosity_model = none | braginskii | parser. */
    int m_visc_model = 0;
    /** True when m_visc_model != 0; the single gate every call site tests. */
    bool m_include_electron_viscosity = false;
    /** Kinematic viscosity parsers nu(n [m^-3], Te [eV], B [T]) in m^2/s,
     *  consulted only under qdsmc_viscosity_model = parser. Inputs:
     *  qdsmc_nu_par(n,Te,B) / qdsmc_nu_perp(n,Te,B). */
    std::string m_visc_nu_par_expression = "0.0";
    std::string m_visc_nu_perp_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_visc_nu_par_parser;
    std::unique_ptr<amrex::Parser> m_visc_nu_perp_parser;
    amrex::ParserExecutor<3> m_visc_nu_par;
    amrex::ParserExecutor<3> m_visc_nu_perp;
    /** Coulomb logarithm and effective ion charge entering the Braginskii
     *  collision time tau_e. Inputs: qdsmc_viscosity_coulomb_log,
     *  qdsmc_viscosity_Z_eff. */
    amrex::Real m_visc_coulomb_log = 10.0;
    amrex::Real m_visc_Z_eff = 1.0;
    /** Free-streaming flux-limit factor f on the PARALLEL channel, the
     *  momentum analogue of qdsmc_conduction_flux_limit_factor and set to
     *  the same default for the same reason: unbounded Braginskii parallel
     *  viscosity is as unphysically large as unbounded Spitzer parallel
     *  conduction. Values <= 0 disable it (not recommended -- the limiter
     *  is also what bounds the heating rate at dTe/dt <= f |S| Te). Input:
     *  qdsmc_viscosity_flux_limit_factor. */
    amrex::Real m_visc_flux_limit_factor = 0.1;
    /** Global kinematic-viscosity ceiling [m^2/s], applied as a hard min to
     *  both channels (the viscosity analogue of qdsmc_conduction_chi_max).
     *  Off (0) is bit-identical to unset. Input:
     *  qdsmc_viscosity_nu_max. */
    amrex::Real m_visc_nu_max = 0.0;
    /** Velocity-gradient reconstruction: 0 = none (centred differences,
     *  2nd order, unbounded), 1 = mc (MC-limited slopes, default). The
     *  heating is a positive-definite QUADRATIC in grad u_e, so grid-scale
     *  deposition noise in u_e does not cancel -- it rectifies into a
     *  spurious heat source everywhere. The MC slope equals the centred
     *  derivative in smooth regions (retaining 2nd order) and vanishes at a
     *  local extremum, which is exactly the checkerboard mode.
     *
     *  NOTE the deliberate difference from qdsmc_conduction_fd_limiter:
     *  'smart' and 'sh' are FLUX reconstructions (an NVD face value and a
     *  limited transverse face gradient) and have no meaning for an
     *  algebraic source with no flux divergence, so they are not offered
     *  here. mc is the inner limiter of the sh family, i.e. the same
     *  primitive from QdsmcFluxLimiters.H. Input:
     *  qdsmc_viscosity_limiter = none | mc. */
    int m_visc_limiter = 1;

    /** Quiet Gauss--Hermite quadrature points per field-aligned direction
     *  (parallel, perpendicular), each in [2, 7]. npts = 2 is variance-exact
     *  but has a per-step 4th-moment defect (conduction weak order 1 in dt);
     *  npts = 3 matches through the 5th moment (weak order 2). The counts
     *  are per-direction so npts_par can be raised alone for fast parallel
     *  tails at large Spitzer chi_par. Default 3 (Eric, 2026-08-05): the
     *  curvature leak is NON-monotone in npts and npts = 3 is
     *  leak-optimal — its weight-2/3 zero-offset center branch pays no
     *  fractional-redistribution cost, 5x below npts = 2 at the liftoff
     *  point (plan doc C.7d closeout). Input:
     *  qdsmc_conduction_quadrature_points = <npts_par> [<npts_perp>]
     *  (a single value broadcasts to both). */
    int m_cond_npts_par = 3;
    int m_cond_npts_perp = 3;

    /** Free-streaming flux-limit factor f (longitudinal channel only):
     *  kappa_eff = kappa / (1 + |q_Sp| / (f q_fs)) with
     *  q_Sp = kappa |grad_par T_e| and q_fs = n_e k_B T_e v_te. Values
     *  <= 0 disable the limiter. Input:
     *  qdsmc_conduction_flux_limit_factor. */
    amrex::Real m_cond_flux_limit_factor = 0.1;

    /** Numerical hop cap in cells: chi is soft-min limited so the largest
     *  quadrature offset obeys x_max sqrt(2 chi dt_c) <= m_cond_max_hop *
     *  dx_min. Where the cap engages, transport is slower than physical --
     *  accepted by design, like the vacuum-resistivity ceiling turning
     *  vacuum into a fast-but-finite diffusion front. Also bounds the
     *  deposit's guard-cell reach. Input: qdsmc_conduction_max_hop. */
    amrex::Real m_cond_max_hop = 2.0;

    /** In floored cells (n_e <= qdsmc_n_floor) set chi (both channels) to
     *  the hop-cap ceiling instead of the parser value, so conduction
     *  fronts propagate through vacuum/low-density regions at the capped
     *  rate instead of stalling. Input:
     *  qdsmc_conduction_vacuum_fast_front. */
    bool m_cond_vacuum_fast_front = true;

    /** Conduction grid-transfer form. 0 = scatter (spawn daughters at each
     *  node, deposit u forward with the B1 half-gradient hat; exactly
     *  conservative). 1 = layer (Milstein layer/gather form: T^{n+1}(node)
     *  = sum_q w_q Interp(T^n)(foot_q), the adjoint transfer; pointwise
     *  interpolation has no deposit-moment frame seam, so the curvature-
     *  activated remap leak of the scatter form is absent by construction,
     *  at the cost of exact conservation -- see
     *  qdsmc_conduction_conserve_fixup). 2 = fluxform (Esirkepov flux-form
     *  conservative semi-Lagrangian remap, the production path -- plan doc
     *  C.7d: per GH branch, dimensionally split 1D remaps whose face
     *  fluxes integrate an MC-limited PLM reconstruction of u over the
     *  swept interval of the backward-traced displacement field.
     *  Conservation is telescoping-exact, monotonicity is never-add
     *  (limited reconstruction), and the displacement field interpolates
     *  continuously between nodes so the source-evaluated pushforward
     *  error of the deposit kernels is absent; wall faces carry zero flux
     *  = exact adiabatic default). Input: qdsmc_advection_form =
     *  scatter | layer | fluxform. DEFAULT fluxform (the gate-passed
     *  production form; the only one with the closed floor faces and the
     *  per-line EB machinery -- next to a maintained wall band the
     *  mask-less control arms act as a perpetual bath donor). */
    int m_adv_form = 2;

    /** Layer-form interpolant: 0 = linear (bilinear; the exact adjoint of
     *  the uncorrected hat scatter -- control arm, carries the full
     *  dx^2/(4 dt) remap floor), 1 = monotone cubic Hermite (MC-limited
     *  slopes; positive, overshoot-free, ~3rd order in smooth regions),
     *  2 = Keys cubic convolution (a = -1/2; higher order, NOT monotone).
     *  All three are interpolating (exact at integer feet), so the at-rest
     *  identity is machine-exact. Input: qdsmc_conduction_interp =
     *  linear | monocubic | keys. */
    int m_cond_interp = 1;

    /** Layer form: trace the quadrature feet along the curved field line
     *  (midpoint/RK2 rotation: re-gather b at the half-displacement point
     *  and rebuild the quadrature frame there) instead of the straight
     *  tangent chord at the node. MEASURED HARMFUL as implemented
     *  (2026-08-05): the div-D Ito drift already IS the mean-curvature
     *  correction for straight chords (the consistent Euler--Maruyama
     *  pairing, validated by the ring test), so rotating the feet
     *  double-counts the curvature -- systematic cross-field drift plus
     *  10-80x more leak than straight feet on the wiggle instrument.
     *  Default OFF; kept for future consistently-paired weak-2 work.
     *  Input: qdsmc_conduction_curved_feet. */
    bool m_cond_curved_feet = false;

    /** Mehrstellen-style isotropized perpendicular launch
     *  (qdsmc_conduction_isotropic_launch, default off): split chi_perp
     *  equally between the (e1, e2) quadrature lattice and its 45-degree
     *  rotated pair. The equal variance split cancels the leading
     *  cos(4 theta) fourth-moment anisotropy of the axis-pair launch (the
     *  star-shaped point-source diffusion pattern) and gives diagonal hops
     *  first-class per-axis clamping. Doubles the perpendicular branch
     *  count when on. */
    bool m_cond_iso_launch = false;

    /** Layer form: restore Sigma(u) after each gather with a global
     *  proportional rescale of T_e (the gather transfer is not exactly
     *  conservative; the per-call deficit is what this hides -- keep OFF
     *  while measuring, ON for production once the deficit is known to be
     *  diffuse). Input: qdsmc_conduction_conserve_fixup. */
    bool m_cond_conserve_fixup = false;

    /** Scatter-form compensated deposit (Boris--Book FCT architecture,
     *  JCP 11, 38 (1973)): after the hat deposit, run one limited
     *  antidiffusive flux sweep per axis whose coefficient is the
     *  EXACTLY BOOKKEPT deposit covariance (a second tally accumulates
     *  u-weighted fr(1-fr)dx^2 per axis during the deposit), evaluated
     *  with destination-local gradients. In the unlimited smooth limit
     *  the composition hat (x) filter cancels the kernel's second moment
     *  -- equivalent to the Keys deposit through quadratics -- while the
     *  Boris--Book limiter clips the antidiffusive fluxes at extrema:
     *  keys-class interior accuracy with hat positivity/monotonicity and
     *  exact (flux-form) conservation. Requires the hat kernel and
     *  qdsmc_gradient_deposit = 0 (B1 would partially cancel the variance
     *  the tally then over-compensates). Input:
     *  qdsmc_advection_compensate. */
    bool m_adv_compensate = false;

    /** FCT limiter for the compensated deposit. 0 = bb (Boris--Book,
     *  bounds from post-deposit neighbors only: provably clips the
     *  antidiffusion at cross-field filament crests -- a confined
     *  filament IS an extremum -- re-admitting the leak there).
     *  1 = zalesak (Zalesak 1979 flux budgeting with bounds that INCLUDE
     *  the pre-deposit field: restoring a crest toward its pre-substep
     *  height is not a new extremum, so crest compensation is allowed
     *  while ringing stays forbidden -- values remain in the hull of the
     *  old and transported states). 2 = none (unlimited: composes with
     *  the hat to the Keys-equivalent kernel; control arm, rings).
     *  Input: qdsmc_conduction_fct_limiter = bb | zalesak | none. */
    int m_cond_fct_limiter = 0;

    /** Fluxform reconstruction slope limiter. 0 = mc (MC-limited PLM:
     *  never-add monotone, production). 1 = none (unlimited central
     *  slopes -- DIAGNOSTIC control arm: separates limiter-clipping at
     *  filament crests from splitting error in the curvature-leak
     *  metric; can ring at fronts). Input:
     *  qdsmc_advection_slope_limiter = mc | none. */
    int m_adv_slope_limiter = 0;

    /** Fluxform reconstruction order. 0 = plm (MC-limited linear, the
     *  original production reconstruction). 1 = ppm (Colella--Woodward
     *  1984 piecewise parabolic: 4th-order face values from the limited
     *  cell differences + parabola monotonization; ~3rd-order transport,
     *  the same accuracy class as the layer form's monocubic/Keys
     *  interpolant). Motivated by the GF3 curvature-leak attribution
     *  (2026-08-05): the unsplit-donor experiment left the leak
     *  unchanged, falsifying the splitting-error hypothesis -- the
     *  residual is the per-remap truncation error of the 1D PLM
     *  transport itself, differentially modulated across rows by the
     *  row-dependent fractional displacement (npts-sensitivity and the
     *  clean eps=0 floor corroborate; the leak itself is a feature-
     *  resolution property, C.7d closeout). Default ppm (Eric,
     *  2026-08-05): full gauntlet green, aligned constant 6-7x below
     *  plm, new campaign-best ZK front, no undershoot. Split sweeps
     *  only; with qdsmc_advection_fluxform_unsplit the default
     *  silently reverts to plm (the unsplit piece bookkeeping is
     *  PLM-exact) — explicitly requesting ppm + unsplit aborts. Input:
     *  qdsmc_advection_reconstruction = plm | ppm. */
    int m_adv_reconstruction = 1;

    /** Fluxform unsplit donor transport (plan doc C.7d GF3 rung, fork (b):
     *  CSLAM-style flux form). The dimensionally split sweeps remap the
     *  INTERMEDIATE field, which loses donor identity between sweeps --
     *  the residual splitting error measured as the curvature leak
     *  (liftoff point 1.13e-1 vs layer 2.56e-4). Unsplit instead
     *  transports each donor dual cell through all sweep legs with its
     *  OWN material-evaluated displacement (edge displacements read at
     *  the donor's material centers of the earlier legs -- no pull-back,
     *  no intermediate reconstruction): the composed per-donor map is the
     *  bilinear corner image, so cross/corner transport is carried within
     *  a single branch remap. Fluxes remain face-local first-class
     *  objects (Thrust D) computed by deterministic recomputation from
     *  both neighbors (race-free, no atomics), so conservation stays
     *  telescoping-exact; each leg keeps the forward-donor fold rule
     *  (degenerate image = point mass, positivity unconditional). The
     *  split form remains as the cheap arm for gently-curved fields.
     *  Input: qdsmc_advection_fluxform_unsplit. */
    bool m_adv_ff_unsplit = false;

    /** Domain-face conduction BCs on T_e (Thrust D). Per grid dim and
     *  side: 0 = adiabatic (default; the fluxform wall faces already
     *  carry F = 0 = exact zero heat flux), 1 = isothermal (the
     *  boundary node row is a thermal bath pinned to the wall
     *  temperature after every conduction substep — the grid form of
     *  bath re-emission; the exchanged energy is tallied), 2 = flux
     *  (a prescribed heat flux q_wall [W/m^2, positive = into the
     *  plasma] is injected into the boundary row each substep;
     *  tallied). Only meaningful on non-periodic dims (asserted at
     *  first use). Inputs (entries per grid dim):
     *  qdsmc_conduction_bc_lo / _hi = adiabatic | isothermal | flux,
     *  qdsmc_conduction_bc_Te_lo / _hi = <eV>,
     *  qdsmc_conduction_bc_q_lo / _hi = <W/m^2>. */
    int m_cond_bc[AMREX_SPACEDIM][2] = {};
    amrex::Real m_cond_bc_Te[AMREX_SPACEDIM][2] = {};
    amrex::Real m_cond_bc_q[AMREX_SPACEDIM][2] = {};

    /** Cumulative wall-BC energy tallies per (grid dim, side 0=lo/
     *  1=hi), in node-u units [J/m^3] summed over boundary nodes
     *  (multiply by the node dual-cell volume for Joules; positive =
     *  energy added to the plasma). In RZ each node's du is weighted
     *  by 2 pi Vr/dr [J/m^2], the r-dependent part of the dual-cell
     *  volume, so multiplying by dr dz gives Joules. The G4
     *  budget-closure instrument; exposed to Python as
     *  warpx.get_qdsmc_wall_tally(dim, side). */
    mutable amrex::Real m_cond_wall_tally[AMREX_SPACEDIM][2] = {};

    amrex::Real GetQdsmcWallTally (int const dim, int const side) const
    {
        return m_cond_wall_tally[dim][side];
    }

    /** Fluxform only: faces touching a floored node (n_e <= n_floor)
     *  carry zero flux — conduction does not exchange energy with
     *  effectively-electron-free regions. Closes the measured vacuum
     *  deletion class (PEC rho=0 rows + vacuum_fast_front drift cliff
     *  pushed u into floored nodes whose recovery skip then DROPPED
     *  it: Te(row+-1) -> 0 in one substep, -6% Sigma even at chi = 0 —
     *  ring-test record). No-op on decks with no floored nodes (all
     *  gauntlet arms bit-identical). Input:
     *  qdsmc_conduction_closed_floor_faces. */
    bool m_cond_closed_floor_faces = true;

    /** Apply the Thrust-D domain-face conduction BCs (isothermal pin /
     *  prescribed-flux injection) to the boundary node rows and
     *  accumulate the wall tallies. Called at the end of every
     *  conduction substep, before the final T_e ghost fill, for all
     *  conduction forms. rho must be the same density paired with T_e
     *  in that substep. */
    void ApplyQdsmcConductionWallBCs (int lev, amrex::Real dt_c,
                                      amrex::MultiFab & Te,
                                      amrex::MultiFab const & rho) const;

    /** Embedded-boundary conduction BC (Thrust D EB leg, staircase
     *  per-line form — deliberately NO boundary normals: level-set
     *  mirror normals are ill-posed at staircase corners (the covered-B
     *  fill campaign's div instability), while per-sweep-line 1D folds
     *  have no corners at all). Covered nodes (nodal distance_to_eb
     *  <= 0) join the sweep mask. 0 = adiabatic (default): each sweep
     *  line folds donor images specularly at its local mask-boundary
     *  faces (method of images, same machinery as the domain walls;
     *  mask-boundary faces carry zero flux) — exactly conservative.
     *  1 = isothermal: the adiabatic fold PLUS the wall-adjacent fluid
     *  ring (every fluid node with a covered face neighbor) pinned to
     *  qdsmc_conduction_eb_Te(x,y,z) [eV] after each substep — the
     *  per-line form of thermal-bath re-emission, the slab-validated
     *  domain-pin semantics — with the exchange tallied exactly
     *  (positive = energy into the plasma; Python:
     *  warpx.get_qdsmc_eb_tally()). (A bath-DONOR formulation was
     *  measured broken: masked donors carry zero displacement, so a
     *  bath that never hops only absorbs — one-way drain.) Fluxform split
     *  sweeps only; scatter/layer keep pre-EB behavior (control arms)
     *  and the unsplit arm aborts when an EB is present. Prescribed-
     *  flux EB and marker/advection EB reflection are recorded
     *  follow-ups. Inputs: qdsmc_conduction_eb_bc = adiabatic |
     *  isothermal, qdsmc_conduction_eb_Te(x,y,z). */
    int m_cond_eb_bc = 0;

    /** Advection-marker EB handling (Thrust D, the E7-replacement at
     *  embedded boundaries; dsmc-eb-reflection semantics at sub-cell
     *  hop scale): markers homed at covered nodes (level set <= 0)
     *  never move — V_e inside the conductor is not physical — and
     *  fluid markers pushed into the conductor are specularly mirrored
     *  back across the level set (x -> x - 2 phi n; a mirror that
     *  lands covered again keeps the marker home). The midpoint
     *  velocity sample is mirror-guarded the same way. Default ON;
     *  only active when an embedded boundary exists. Input:
     *  qdsmc_eb_marker_reflect. */
    bool m_qdsmc_eb_marker_reflect = true;

    /** Isothermal-EB bath ring depth (Chebyshev distance from a covered
     *  node, [1, 3]). Depth 1 leaves the deposition density ramp at the
     *  wall OUTSIDE the bath: the flux form transports u = n T, and
     *  across an unresolved n cliff the discrete Ito drift
     *  under-corrects — u leaks toward the low-density band and the
     *  pins eat it (measured: annulus interior settled BELOW both bath
     *  temperatures, resolution-independent). Default 2 puts the ramp
     *  inside the bath at the price of an O(dx)-convergent geometric
     *  radius shift. Input: qdsmc_conduction_eb_ring. */
    int m_cond_eb_ring = 2;

    std::string m_cond_eb_Te_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_cond_eb_Te_parser;
    amrex::ParserExecutor<3> m_cond_eb_Te;

    /** Cumulative EB conduction tally (isothermal EB only), node-u
     *  units [J/m^3] summed over fluid nodes, positive = energy added
     *  to the plasma. In RZ each node is weighted by 2 pi Vr/dr
     *  [J/m^2]; multiply by dr dz for Joules. */
    mutable amrex::Real m_cond_eb_tally = 0.0;

    amrex::Real GetQdsmcEbTally () const { return m_cond_eb_tally; }

    /** Positivity floor on T_e, applied after every ACCEPTED conduction
     *  substep of the FD operator, in eV; 0 (the default) disables it
     *  and is bit-identical to the unfloored update. The anisotropic
     *  tensor update is not a monotone scheme: its off-diagonal
     *  cross-term fluxes can drive T_e below zero at grid-sharp
     *  rotations of the field direction, and the embedded temporal
     *  error estimate accepts the undershoot because both stages share
     *  it (a spatial-discretization property, not a time-step one).
     *  The floor clamps such nodes to the given value and books the
     *  injected energy so conservation diagnostics see it.
     *  Input: qdsmc_conduction_Te_floor. */
    amrex::Real m_cond_te_floor = 0.0;

    /** Exclude floor-clamped nodes from the conduction error norm
     *  (meaningful only with the floor armed; silently inert
     *  otherwise). A projection inside an error-controlled loop pins
     *  the embedded error at its own kinks: under sustained undershoot
     *  pressure the clamp re-fires every accepted substep, the
     *  controller collapses dt to resolve dynamics the projection
     *  immediately overwrites, and the attempts budget burns marching
     *  (a clamp/controller limit cycle). Masking the last accepted
     *  substep's clamp set restores error control over the nodes the
     *  integrator actually owns; the floor itself bounds the local
     *  error at the clamped nodes. Off (the default) is bit-identical.
     *  Input: qdsmc_conduction_Te_floor_mask_err. */
    bool m_cond_te_floor_mask = false;

    /** Cumulative conduction-floor tally, same units and weighting as
     *  m_cond_eb_tally (node-u [J/m^3] summed over fluid nodes; in RZ
     *  each node weighted by 2 pi Vr/dr [J/m^2] -- multiply by dr dz
     *  for Joules). Positive = energy added by the floor. */
    mutable amrex::Real m_cond_floor_tally = 0.0;

    amrex::Real GetQdsmcCondFloorTally () const { return m_cond_floor_tally; }

    /** Scatter-form deposit kernel. 0 = hat (2^d CIC + the B1
     *  half-gradient correction). 1 = Keys cubic convolution (a = -1/2,
     *  4^d): partition of unity is exact (conservation stays exact) and
     *  the kernel reproduces quadratics, so its first AND second deposit
     *  moments vanish -- the remap diffusion the B1 correction fights,
     *  including the curvature-rotated cross part the per-axis B1 cannot
     *  cancel, is absent by construction (no correction applied). The
     *  price is Godunov's: negative lobes, NOT monotone -- sharp fronts
     *  can ring/undershoot. Interpolating at zero displacement, so the
     *  at-rest identity stays machine-exact. Input:
     *  qdsmc_conduction_deposit_kernel = hat | keys. */
    int m_cond_deposit_kernel = 0;

    /** Deposited-weight threshold (density units, m^-3) for the QDSMC T_e
     * update: cells whose deposited marker weight is at or below this value
     * are skipped and keep their previous T_e, guarding the division by the
     * deposited weight in cells no QDSMC marker reached. Defaults to the
     * same floor as the rest of the hybrid solver. */
    amrex::Real m_qdsmc_n_floor = 1.0;
    /** False until the adiabat T_e seed (SeedTeAdiabat, PR #7128) has run
     *  once. HybridPICInitializeRhoJandB executes at step == step_begin of
     *  EVERY Evolve() entry (each segmented PICMI sim.step()), and an
     *  unconditional re-seed there would wipe the evolved T_e — the same
     *  re-entry class as the Pe closure reset fixed earlier on this
     *  branch. Mutable: lazily-set cache flag inside a const call chain. */
    mutable bool m_qdsmc_te_seeded = false;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** False until the first AdvanceElectronEnergyQDSMC of the run has
     *  filled hybrid_current_fp_plasma. On later steps J_plasma is already
     *  valid from the previous step's final E-solve (B is unchanged in
     *  between), so the QDSMC entry-point recompute is skipped. Mutable
     *  because it is a lazily-set cache flag inside a const call chain. */
    mutable bool m_qdsmc_J_plasma_valid = false;

    /** Opt-in momentum-consistent particle push field for the
     *  theta-implicit hybrid solver: subtract the resistive part of the
     *  Ohm field from the field gathered by the ions (the explicit scheme
     *  pushes ions with the no-resistivity Ohm field; ion-side resistive
     *  friction is the separate drag operator). Off by default: the
     *  correction couples the resistive field into the particle response
     *  and can destabilize the Newton iteration in whistler-marginal
     *  configurations. */
    bool m_implicit_push_excludes_resistive_field = false;

    /** True when any resistive term is configured: a non-trivial global
     *  resistivity parser, per-species overlays, or hyper-resistivity. */
    [[nodiscard]] bool HasResistivity () const {
        return (m_eta_expression != "0.0") || m_has_per_species_eta
            || m_include_hyper_resistivity_term;
    }

    /** Per-species resistivity overlay (see Phys. Plasmas 31, 012902 (2024), Eq. 10).
     *  Optional. For any charged species {spec} for which the user provides
     *
     *      hybrid_pic_model.plasma_resistivity_{spec}(rho_s,rho,Te,J,J_s,B,t) = "..."
     *
     *  the parser is compiled and stored here. The total per-species
     *  effective resistivity used by Ohm's law, the Joule-heating source,
     *  and the resistive-drag operator is
     *
     *      eta_s_eff = eta_global(rho, |J|, t)              [the global m_eta]
     *                + eta_per_species_s(rho_s, rho, Te,    [this map]
     *                                    |J|, |J_s|, |B|, t)
     *
     *  so the global parser remains the shared anomalous (e.g. Chodura)
     *  layer and the per-species parser overlays species-resolved physics
     *  (e.g. classical Spitzer with Z^2 scaling). Species without a
     *  per-species parser registered fall back to the global parser only.
     *  Parser variables, in order, are:
     *    rho_s : per-species charge density        (C/m^3, unscaled in RZ)
     *    rho   : total plasma charge density       (C/m^3, volume-scaled)
     *    Te    : electron temperature              (Kelvin)
     *    J     : |J_plasma|                        (A/m^2)
     *    J_s   : |J_{species}|                     (A/m^2, unscaled in RZ)
     *    B     : |B|                               (Tesla)
     *    t     : time                              (seconds)
     */
    std::map<std::string, std::unique_ptr<amrex::Parser>> m_per_species_resistivity_parser;
    std::map<std::string, amrex::ParserExecutor<7>>       m_eta_per_species;
    /** Fast-path flag: true if at least one charged species has a
     *  registered per-species resistivity parser. When false the
     *  per-species code paths short-circuit to the global parser only,
     *  preserving the existing single-eta behaviour with zero overhead.
     *  Set from the input deck in ReadParameters (so AllocateLevelMFs can
     *  gate the overlay fields) and confirmed in InitData when the parsers
     *  are compiled. */
    bool m_has_per_species_eta = false;

    /** True when the gridded fluid velocities Ve_fp / Vs_fp_{spec} are
     *  needed, i.e. when a per-species resistivity parser is registered or
     *  a hybrid_resistive_drag collision is present in the input deck.
     *  Gates their allocation and the once-per-step
     *  CalculateElectronFluidVelocity / CalculateIonFluidVelocity calls.
     *  Set in ReadParameters. */
    bool m_need_fluid_velocities = false;

    /** True when the per-species deposited fields current_fp_{spec} /
     *  rho_fp_{spec} are needed, i.e. when m_need_fluid_velocities is true
     *  or the electron-energy equation is solved (its Joule and Q_ei
     *  sources read the per-species charge densities). Gates their
     *  allocation and the per-species deposition path in
     *  HybridPICDepositRhoAndJ; when false, the deposition falls back to
     *  the single-pass MultiParticleContainer path with zero extra cost.
     *  Set in ReadParameters. */
    bool m_need_per_species_fields = false;

    /** Plasma hyper-resistivity eta_H [Ohm m^3].
     *
     *  NUMERICAL STABILISER -- ITS DISSIPATION IS DELIBERATELY NOT BOOKED
     *  AS HEATING. This is a standing decision, not an oversight, and it is
     *  the reason the physical electron viscosity exists as a SEPARATE term
     *  (m_visc_model, BraginskiiViscosity.H) rather than as a relabelling of
     *  this one:
     *
     *   - eta_H is the sledgehammer for unphysical modes. Its magnitude is
     *     chosen for stability, not from any transport coefficient, so the
     *     energy it removes is a numerical artefact with no claim on the
     *     electron internal energy. Booking it would credit T_e for the
     *     removal of modes the model never resolved.
     *   - The physical Braginskii electron viscosity has the SAME grad^2
     *     form on the current (u_e ~ -J/(e n_e)) and differs entirely in
     *     the coefficient. That shared form is exactly why the two must
     *     stay explicitly apart -- a reader who sees only the form will
     *     conflate them.
     *
     *  If you are here because P_etaH looks like uncounted energy: it is,
     *  by design. Add or resize the physical viscosity instead; do not
     *  route eta_H into the electron energy equation.
     *
     *  Its power is still REPORTED (never booked) in the HybridDissipation
     *  reduced diag as the P_etaH column, kept separate from the physical
     *  viscous column P_nu precisely so the two can never be confused.
     */
    std::string m_eta_h_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_hyper_resistivity_parser;
    amrex::ParserExecutor<2> m_eta_h;
    bool m_include_hyper_resistivity_term = false;
    bool m_hyper_resistivity_has_B_dependence = false;
    /** Apply the hyper-resistivity as the adjoint-paired curl-curl form
     *  E_H = +curl(eta_H curl J) instead of the component-wise vector
     *  Laplacian -eta_H lap J (cross-lane operator ruling: the two differ
     *  by eta_H grad(div J), which is NOT Faraday-invisible under a
     *  spatially varying eta_H — grad(eta_H) x grad(div J) is a spurious
     *  B source at coefficient ramps, driven by deposition-noise div J at
     *  ~70% of ||curlcurl J|| — and the component Laplacian is not the
     *  discrete grad-div - curl-curl split, leaking curl at truncation
     *  order). The composition uses the NATIVE Yee curls (J on edges ->
     *  curl -> faces, eta_H applied at faces where |B| is native -> curl
     *  back to edges), which are mutually adjoint by construction, so
     *  the dissipation -int eta_H |curl J|^2 is sign-definite discretely
     *  (the collocated-curl failure mode: checkerboard-blind stencils
     *  without adjoint pairing admit positive-real grid-scale
     *  eigenvalues). Boundary statement: the operator ends one node
     *  inside non-periodic domain faces (r_max wall, non-periodic z) and
     *  carries no wall flux — no pass reads a domain ghost. RZ only.
     *  Default off (the Laplacian path is bit-identical legacy).
     *  Input: hyper_resistivity_curl_curl. */
    bool m_hyper_res_curl_curl = false;

    /** External current */
    std::string m_Jx_ext_grid_function = "0.0";
    std::string m_Jy_ext_grid_function = "0.0";
    std::string m_Jz_ext_grid_function = "0.0";
    std::array< std::unique_ptr<amrex::Parser>, 3> m_J_external_parser;
    std::array< amrex::ParserExecutor<4>, 3> m_J_external;
    bool m_has_external_current = true;
    bool m_external_current_has_time_dependence = false;

    /** External E/B fields */
    bool m_add_external_fields = false;
    /** Which magnetic field the Ohm kernels receive when external
     * vector-potential fields are enabled. Both schemes advance the
     * external flux analytically from A(t) (wall boundary conditions act
     * on the plasma fields only and cannot exclude the programmed coil
     * flux) and both use the same stored-E convention (the kernels
     * subtract the inductive E_ext from plasma cells; the push field
     * re-adds it).
     *
     * true (explicit scheme): Bfield_fp is plasma-only during the field
     * advance, so the kernels add hybrid_B_fp_external for the Hall term.
     *
     * false (theta-implicit scheme): the field assembly hands the kernels
     * the total B already, so they must not add it again. */
    bool m_external_split = true;
    /** Unified-drive mode (Darwin path): the external vector potential
     * enters through the boundary values of the evolved A instead of the
     * split external fields, and the Ohm kernels must not touch the
     * external-field arrays at all. */
    bool m_external_unified = false;
    /** Subtract the external E field from the Ohm's-law E unconditionally
     * instead of gating the subtraction on rho >= rho_floor. The per-cell
     * gate is a state-dependent discontinuity that breaks matrix-free
     * Jacobian probes, so a solver iterating the external fields inside
     * its residual can enable this in its Define(). The default (false)
     * preserves the split-field behavior, where external fields penetrate
     * sub-floor vacuum regions freely. Not a user input. */
    bool m_external_e_subtraction_unconditional = false;
    std::unique_ptr<ExternalVectorPotential> m_external_vector_potential;

    /** Gpu Vector with index type of the Jx multifab */
    amrex::GpuArray<int, 3> Jx_IndexType;
    /** Gpu Vector with index type of the Jy multifab */
    amrex::GpuArray<int, 3> Jy_IndexType;
    /** Gpu Vector with index type of the Jz multifab */
    amrex::GpuArray<int, 3> Jz_IndexType;
    /** Gpu Vector with index type of the Bx multifab */
    amrex::GpuArray<int, 3> Bx_IndexType;
    /** Gpu Vector with index type of the By multifab */
    amrex::GpuArray<int, 3> By_IndexType;
    /** Gpu Vector with index type of the Bz multifab */
    amrex::GpuArray<int, 3> Bz_IndexType;
    /** Gpu Vector with index type of the Ex multifab */
    amrex::GpuArray<int, 3> Ex_IndexType;
    /** Gpu Vector with index type of the Ey multifab */
    amrex::GpuArray<int, 3> Ey_IndexType;
    /** Gpu Vector with index type of the Ez multifab */
    amrex::GpuArray<int, 3> Ez_IndexType;

    /** QDSMC fictitious-particle container used to transport the electron
     *  entropy K_e by V_e over one PIC step. Lazily constructed in InitData
     *  when m_solve_electron_energy_equation is true; otherwise nullptr.
     *  Owned by HybridPICModel
     *  because its lifetime is tied to the electron-energy equation that
     *  the hybrid model is responsible for. */
    std::unique_ptr<QdsmcParticleContainer> m_qdsmc_pc;

    /** Dedicated container for the one-pass conduction nodes (Phase 3a).
     *  Kept separate from m_qdsmc_pc: the advection markers are persistent
     *  (spawned once per run), while conduction nodes are spawned, kicked,
     *  deposited and destroyed every pass. */

    /** Per-charged-species ion-temperature deposits frozen at t^n for the
     *  duration of one theta-implicit step (built by
     *  QDSMCSaveImplicitStepStart, read by the in-residual Q_ei sink and the
     *  post-solve ion-heating kick, released by QDSMCFinishImplicitStep).
     *  Keyed by level, then species name. Mutable because the QDSMC stage
     *  runs inside const residual evaluations. */
    mutable std::map<int, std::map<std::string, std::unique_ptr<amrex::MultiFab>>> m_qdsmc_Ti_owned;
    mutable std::map<int, std::map<std::string, amrex::MultiFab*>> m_qdsmc_Ti_by_name;

    /** Scratch midpoint charge-density deposit used by the theta-implicit
     *  QDSMC stage. The K_e seed / T_e recovery algebra must pair densities
     *  of a single deposit convention: in the radial geometries the
     *  boundary-node values of the nonlinear-solver deposit (rho_fp) and of
     *  MultiParticleContainer::DepositCharge differ (no wall fold, different
     *  axis handling), and mixing them turns the intensive T_e recovery into
     *  a spurious boundary heat source. The stage therefore deposits its own
     *  midpoint density through the same MultiParticleContainer path used
     *  for rho^n and rho^{n+1}. Lazily allocated per level. */
    mutable std::map<int, std::unique_ptr<amrex::MultiFab>> m_qdsmc_rho_mid;

    /** Boundary-condition handler for the Darwin phi solve (same BC mapping
     *  as the electrostatic solver). Lazily constructed on first use. */
    mutable std::unique_ptr<PoissonBoundaryHandler> m_darwin_bc_handler;
};

/**
 * \brief
 * This struct contains only static functions to compute the electron pressure
 * using the particle density at a given point and the user provided reference
 * density and temperatures.
 */
struct ElectronPressure {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real get_pressure (amrex::Real const n0,
                                     amrex::Real const T0,
                                     amrex::Real const gamma,
                                     amrex::Real const rho) {
        return n0 * T0 * std::pow((rho/PhysConst::q_e)/n0, gamma);
    }
};

#endif // WARPX_HYBRIDPICMODEL_H_
