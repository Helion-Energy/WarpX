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

#include <map>
#include <memory>
#include <optional>

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
        bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, PatchType patch_type, bool solve_for_Faraday) const;

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
     * Function to calculate the electron pressure, and the closure's electron
     * temperature alongside it, from the simulation charge density. Used in the
     * Ohm's law solver (kinetic-fluid hybrid model).
     *
     * \param[in] floor_density if true, evaluate the closure on the floored
     *            density max(n_e, n_floor) instead of the raw n_e, so T_e
     *            flattens to T_e0 (n_floor/n0)^(gamma-1) below the floor rather
     *            than decaying towards zero with n_e. The QDSMC electron-energy
     *            equation needs that flattened state from its seed: it is the
     *            transport's zero-gradient state (K_e uniform, floored halo
     *            included), whereas a halo whose T_e decays to ~0 acts as an
     *            absorbing K_e = 0 boundary. Off by default, leaving the
     *            algebraic closure path evaluated on the raw density.
     */
    void CalculateElectronPressure (bool floor_density = false) const;
    void CalculateElectronPressure (int lev, bool floor_density = false) const;

    /**
     * \brief Fill the electron pressure multifab given the kinetic particle
     * charge density (and assumption of quasi-neutrality) using the user
     * specified electron equation of state. The closure's electron temperature,
     * T_e = T0 (n_e/n0)^(gamma-1), is filled at the same time (P_e = n_e T_e
     * is formed from it). That T_e is diagnostic-only on this path: with
     * solve_electron_energy_equation on, this function is not called and
     * T_e is owned by the QDSMC entropy transport, which fills Te/Pe at
     * this same point in the field loop.
     *
     * \param[out] Pe_field scalar electron pressure MultiFab at a given level
     * \param[out] Te_field scalar electron temperature MultiFab (in Kelvin) at a given level
     * \param[in] rho_field scalar ion charge density Multifab at a given level
     * \param[in] floor_density evaluate the closure at max(rho, q_e n_floor)
     *            instead of rho
     */
    void FillElectronPressureMF (
        amrex::MultiFab& Pe_field,
        amrex::MultiFab& Te_field,
        amrex::MultiFab const& rho_field,
        bool floor_density = false ) const;

    /**
     * \brief Fill the nodal V_e = -(J_plasma - J_i) / (q_e n_e) MultiFabs by
     *        interpolating the Yee-staggered total plasma current and ion
     *        current to the nodal grid (using ablastr::coarsen::sample::Interp)
     *        and dividing by rho. Cells with rho <= rho_floor are left at 0.
     *
     *        Reads hybrid_current_fp_plasma (J_plasma), hybrid_current_fp_temp
     *        (J_i at n+1/2), hybrid_rho_fp_temp (rho at n+1/2). Writes
     *        hybrid_electron_velocity_fp (3 directions, nodal).
     */
    void QDSMCInitializeUe (int lev) const;

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

    /**
     * \brief After the QDSMC scatter, recover T_e^{n+1} from
     *           T_e = (deposited K*N) / (deposited N) / n_e^(1-gamma)
     *                 / (k_B / q_e)
     *        using the floored density max(n_e, n_floor) from rho at n+1
     *        (rho_fp), the deposited entropy field (hybrid_entropy_fp), and
     *        the deposited weight field (hybrid_qdsmc_weights_fp). Cells that
     *        received no deposited weight keep their previous T_e.
     *
     *        Writes hybrid_electron_temperature_fp.
     */
    void QDSMCUpdateTe (int lev) const;

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
     *        (m_n_floor on rho_fp, the QDSMCUpdateTe guard) or
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
     */
    void AdvanceElectronEnergyQDSMC (amrex::Real dt) const;

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active). May increase under
     *  stress and slowly decay toward 2*n_attempts (absolute minimum 2). */
    int m_substeps = 10;

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

    bool m_holmstrom_vacuum_region = false;

    /** Generalized-Ohm's-law (GOL) E-solve FORM. Terminology: the
     *  hybrid solver always solves the generalized Ohm's law; what
     *  this selects is the form. E-FORM (esolve = e_form, alias ohm,
     *  the default): the legacy algebraic solve for E with its density
     *  division. AMANO-FORM (esolve = amano_form, alias gol): the
     *  inertia-screened division-free form (Amano, Higashimori &
     *  Shirakawa, JCP 275 (2014) 197; the elliptic version is the
     *  Hewett-Nielson Darwin transverse-field solve, JCP 29 (1978)
     *  219, in generalized-Ohm clothing): instead of the algebraic
     *  solve, treat the screened equation
     *      (rho_e - eps grad^2) E = F,
     *      F = enE - grad Pe + (rho_e eta)_eff J,   eps = m_e'/(mu0 e),
     *  whose RHS is division-free (every term ~ density moments), so the
     *  vacuum limit is Laplace's equation -- no floor, no eta_vac ramp,
     *  no pedestal needed for the FIELD solve. Solved with
     *  warm-started Jacobi sweeps per B-substep stage (lock-step;
     *  the operator is diagonally dominant where d_e < dx). SECTOR
     *  CAVEAT: the true inertia screening operator is PURE CURL-CURL
     *  (dJe/dt = -curlcurl E/mu0 - dJi/dt) and annihilates the
     *  longitudinal field identically; this component-wise grad^2 form
     *  (Amano Eq (14), div E ~ 0) spuriously screens E_L -- invisible
     *  for 1D transverse modes, but in 2D/3D it damps the
     *  grad-Pe-driven longitudinal field at density-contrast edges.
     *  Multi-D users of the jacobi form should prefer a curl-curl
     *  operator (which passes E_L through untouched); the relax form
     *  below has no grad^2 and is unaffected. m_e' is the Eq-(27)
     *  variable electron mass evaluated at the CURRENT substep dt
     *  (published in m_gol_dt_sub by the substep loop), so subcycling
     *  directly reduces the artificial inertia. gol_vacuum_gamma_frac
     *  arms the Eq-(30) vacuum-resistivity-as-LOCAL-DAMPING term
     *  (gamma_max = frac/dt): in the GOL form vacuum current decays
     *  exponentially with NO diffusion CFL. v1: collocated Cartesian
     *  only. Inputs: hybrid_pic_model.esolve = ohm|gol, gol_sweeps,
     *  gol_cfl_alpha, gol_n_min [m^-3, the one-count level],
     *  gol_vacuum_gamma_frac. */
    bool m_esolve_gol = false;
    int  m_gol_sweeps = 4;
    amrex::Real m_gol_alpha = 0.5;
    amrex::Real m_gol_n_min = 0.0;
    amrex::Real m_gol_vac_gamma_frac = 0.0;
    /** Current B-substep dt, published for the Eq-(27) mass evaluation.
     *  Mutable: set inside the const substep loop; 0 -> full-step dt. */
    mutable amrex::Real m_gol_dt_sub = 0.0;

    /** Amano-form advance (how the screened equation is treated).
     *  0 = jacobi: the screened elliptic solve above,
     *  warm-started stationary sweeps -- converges only where plasma
     *  makes the operator diagonally dominant; across wide vacuum gaps
     *  the Laplace limit is irreducibly elliptic (measured: per-sweep
     *  contraction 0.991 over an 85-cell gap, error ~ sweeps^-1.3).
     *  1 = relax: the semi-implicit E/J_e relaxation of Angus, Dorf &
     *  Geyko (PoP 26, 072505 (2019) Sec. V.A; Harned & Mikic JCP 83, 1
     *  (1989) lineage): J_e is kept as a persistent dynamical field
     *  ("hybrid_gol_je_fp", division-free electron momentum with the
     *  stiff local terms -- E coupling, gyration, friction -- solved
     *  pointwise implicitly, closed form) and E advances by
     *  Ampere-Maxwell with displacement current at the CFL-set
     *  artificial light speed c_art = min(c, gol_c_frac/(dt_sub
     *  sqrt(sum dx_i^-2))). The elliptic vacuum continuation is then
     *  paid in RETARDATION (waves cross the gap at c_art, ~1 cell per
     *  substep) instead of iterations: ballistic O(N_gap) information
     *  transport per step vs the diffusive O(N_gap^2) of stationary
     *  sweeps. Steady state recovers generalized Ohm + Ampere exactly.
     *  State advances ONLY on solve_for_Faraday calls (once per
     *  substep); non-Faraday calls return the current state. Inputs:
     *  gol_form = jacobi|relax, gol_c_frac (default 0.5).
     *  THIS FORK: only the relax form is ported (default 1 = relax,
     *  a deliberate deviation from the source default); gol_form =
     *  jacobi aborts at parse. */
    int m_gol_form = 1;
    amrex::Real m_gol_c_frac = 0.5;
    /** Absolute ceiling on the relax form's artificial light speed
     *  [m/s]. The CFL formula alone saturates at the REAL c when
     *  dt_sub is small relative to dx (measured on a compression
     *  slab: 256 substeps put the CFL value above c), and at real c
     *  the artificial permittivity is the real eps0 -- every
     *  substep's deposition shot noise then kicks an undamped
     *  light-branch bath that grows exponentially-slowly (~11x per
     *  400 steps measured, backend-independent) until particles
     *  over-jump cells and deposition faults. c_art only needs to
     *  exceed the fastest RESOLVED signal (electron-Alfven /
     *  grid-whistler scale) with margin; the E-per-noise response
     *  falls as c_art^2. 0 = no ceiling (legacy min(c, CFL)).
     *  Input: gol_c_max. */
    amrex::Real m_gol_c_max = 0.0;
    /** Quasineutral relaxation rate of the relax form, as a fraction
     *  of 1/dt (gamma_qn = gol_qn_frac/dt, applied implicitly per
     *  substep, plasma-blended: full above the one-count density,
     *  off below it). WHY: the E-form hybrid never evolves the
     *  longitudinal field -- quasineutrality is imposed
     *  algebraically -- but the relax form integrates charge
     *  separation honestly against kinetic ions pushed explicitly at
     *  an unresolvable omega_pi dt >> 1: the ion-Langmuir loop pumps
     *  E_L into a slowly growing bath (measured ~10x per 400 steps,
     *  c_art-INDEPENDENT, with B and Te cold -- the longitudinal
     *  signature). Fix per the Hewett-Nielson split: transverse
     *  dynamics ride free (the inertia-limited whistler ceiling
     *  omega_ce dt >> gamma_qn dt = O(1) is far above the relaxation)
     *  while E relaxes toward the one-count-guarded generalized-Ohm
     *  value, pinning the slow/longitudinal sector algebraically
     *  without any projection or elliptic machinery. In vacuum the
     *  blend is zero and the wave physics is untouched. Default 1;
     *  0 = off (the unstable honest-E_L dynamics, for study).
     *  NOTE: the blended generalized-Ohm target retains a
     *  one-count-floored density division -- the core Ampere-Maxwell
     *  update is division-free, the OPTIONAL qn pin is not (its tanh
     *  weight is ~0.018 at rho = 0, not 0). Set gol_qn_frac = 0 for
     *  a strictly division-free solve, at the documented cost of the
     *  longitudinal bath. Input: gol_qn_frac. */
    amrex::Real m_gol_qn_frac = 1.0;
    /** Marder divergence cleaning of the relax form's E, restricted to
     *  the vacuum side of the one-count blend (complement of the qn
     *  blend): dE/dt += d_M grad(div E), d_M = frac/(2 dt_sub
     *  sum(dx_i^-2)) (the explicit diffusion CFL). WHY: the reproducer
     *  localized the residual E bath to the LONGITUDINAL field in the
     *  vacuum gap -- electrostatic noise from unresolved edge
     *  charge-separation accumulating where both the qn relaxation
     *  (plasma-only) and the Eq-30 current damping (acts on J_e, not
     *  E) are silent. Marder cleaning diffuses divergence noise while
     *  the PHYSICAL vacuum continuation is exactly in its kernel
     *  (harmonic fields have div E = 0) and transverse waves are
     *  untouched -- gap tracking is preserved by construction. Local
     *  stencil, no projection, no elliptic machinery. Source default
     *  0.5; 0 = off. THIS FORK: the Marder pass still needs its RZ
     *  forms ((1/r) d(r Er)/dr divergence and the vector-Laplacian
     *  components) -- default 0 and a nonzero selection under
     *  esolve = gol aborts at parse until it lands.
     *  Input: gol_div_clean_frac. */
    amrex::Real m_gol_div_clean_frac = 0.0;
    /** Amano Eq-(27) variable electron mass in the RELAX advance
     *  (default on). The relax form's true-m_e J_e dynamics carry the
     *  grid-whistler branch honestly, and the frozen-field split has
     *  an explicit stability edge in the control variable B^2/n at a
     *  dt_sub-scaled bound -- measured on the compression slab as a
     *  deterministic threshold (three arms; ignition at the
     *  floor-density band, correlate on the near-vacuum reservoir)
     *  with a finite-time blow-up past it. Per-cell inflation
     *  m_e' = max(m_e, B^2 q_e/(mu0 rho_g) (dt_sub/(2 alpha dx))^2)
     *  (the shared GolEffectiveElectronMass helper, alpha =
     *  gol_cfl_alpha, rho_g one-count-guarded) slows exactly the
     *  cells whose grid-whistler speed would violate the substep CFL
     *  and is INERT wherever the physics is resolved (max with the
     *  true m_e -- the dispersion gates measure the physical branch
     *  unchanged). Applies to every 1/m_e in the J_e update: E
     *  coupling, gyration, friction, grad-Pe. 0 = true m_e (the
     *  measured-unstable configuration, for study). KNOWN LIMIT
     *  (separate from the whistler edge, alpha- and mass-cap-
     *  insensitive): at very coarse substep counts the split itself
     *  destabilizes -- measured onset on the 1D slab-gap deck at
     *  substeps 8/16/24 dying in 17/221/382 steps, while 40+ (every
     *  production configuration) is stable across the full campaign.
     *  Keep substeps at the whistler-resolving counts the adaptive
     *  E-form runs already selected (O(100+) on compression decks).
     *  Input: gol_var_mass. */
    bool m_gol_var_mass = true;
    /** Lazy one-time J_e seed from the Ampere closure (J_plasma - J_i)
     *  on the first relax call. Mutable: set inside the const solve.
     *  NOTE: single flag -- multi-level relax untested in v1. */
    mutable bool m_gol_je_init = false;
    /** Relax-form advance gate. The RK substep STAGES call the E solve
     *  4-6x per substep with intermediate B states; a stateful update
     *  there would over-advance (and a rejected adaptive substep could
     *  not roll the E/J_e state back). So under gol_form = relax every
     *  ordinary solve call returns immediately (the stages read the
     *  frozen E state -- frozen-E/frozen-B splitting, FDTD-leapfrog
     *  class) and BfieldEvolve raises this gate to advance (E, J_e)
     *  exactly once per ACCEPTED substep against the updated B.
     *  Mutable: toggled inside the const substep loop. */
    mutable bool m_gol_relax_advance = false;
    /** Time centering of the relax form's frozen-E/frozen-B split
     *  (default on). The legacy (Lie) sequencing per substep is
     *  B^{n+1} = B^n - dt curl E^n followed by the (E, J_e) advance
     *  against B^{n+1}: each operator samples the other a full
     *  substep off-center (B sees stale E, the electron advance sees
     *  future B), the first-order splitting error class the
     *  Harned-Mikic stability analysis flags for the whistler branch
     *  at coarse omega dt. Centered (Strang) sequencing symmetrizes
     *  both couplings at zero extra solves: a half Faraday push with
     *  E^n, the full-dt (E, J_e) advance against the HALF-LEVEL B
     *  and its Ampere current, then a half Faraday push with the
     *  updated E -- B integrates the time-centered average of E and
     *  the electron advance reads time-centered B. MEASURED: the
     *  dispersion gates are unchanged (whistler inertia branch to
     *  the same 0.7%), and the coarse-substep floor (see the
     *  gol_var_mass known limit) is INVARIANT under centering --
     *  slab-gap deaths at substeps 8/16/24 land within 3 steps of
     *  the Lie split's at every rung, so that floor is a property of
     *  the coarse-dt_sub (E, J_e) relax dynamics themselves, not of
     *  the split's time centering. The DEATH MODE at that floor does
     *  differ: the Lie split diverges through NaN and takes the
     *  designed substep assert, while the centered split can blow up
     *  huge-but-finite -- particles then leave the domain by O(100)
     *  cells in one step and the crash surfaces as an out-of-bounds
     *  gather/deposition access (bounds-check verified) instead of
     *  the assert. Both are the same upstream instability in an
     *  already-fatal regime. 0 = the legacy Lie split.
     *  THIS FORK: the centered split still needs the half-dt B push
     *  plumbing in the substep loop -- default 0 (Lie) and selecting
     *  it under esolve = gol aborts at parse until it lands (the
     *  dispersion gates measured Lie and centered identical, so Lie
     *  is a valid production configuration, not a downgrade).
     *  Input: gol_centered_split. */
    bool m_gol_centered_split = false;

    /** Amano Eq (27) effective electron mass, multiplied through by m_i
     *  (the ion mass cancels): m_e' = max(m_e, B^2 q_e/(mu0 rho_e) *
     *  (dt/(2 alpha dx))^2), which pins the electron-Alfven phase
     *  velocity at the local CFL. Shared helper by design -- the
     *  implicit path consumes the same formula (keep callable, do not
     *  inline into loops). rho_e must be guarded positive by the caller
     *  (one-count level). */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real GolEffectiveElectronMass (
        amrex::Real const B2, amrex::Real const rho_e_guarded,
        amrex::Real const dt, amrex::Real const dx_min,
        amrex::Real const alpha)
    {
        amrex::Real const va2_me =
            B2 * PhysConst::q_e / (PhysConst::mu0 * rho_e_guarded);
        amrex::Real const s = dt / (amrex::Real(2.0) * alpha * dx_min);
        // local copy: amrex::max binds const references and the
        // namespace constexpr has no device definition (nvcc ODR-use)
        amrex::Real const m_e_v = PhysConst::m_e;
        return amrex::max(m_e_v, va2_me * s * s);
    }

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
     * false (default), Pe is computed from the algebraic polytropic closure
     * (FillElectronPressureMF): T_e is not an evolved state variable, but
     * the implied temperature Pe/(n_e k_B) still varies with density
     * through the closure (with the chosen gamma, which need not be
     * adiabatic). */
    bool m_solve_electron_energy_equation = false;

    /** Resistive electron-heating source on T_e (see Phys. Plasmas 31, 012902 (2024), Eq. 12),
     * computed per cell as
     *     S_e = Sigma_s nu_{s,e} n_s m_s |dV|^2,
     *         nu_{s,e} = Z_s e^2 eta n_e / m_s,   dV = J_plasma / (e n_e),
     * from the nodal plasma current, the per-species charge densities and
     * the Ohm's-law eta parser. Single bool: on or off. Reduces exactly to
     * eta J^2 in single species. Only consulted when
     * m_solve_electron_energy_equation is also true. */
    bool m_include_joule_heating = false;

    /** Te-threshold Joule redirection. When active, the eta*J^2 Joule source in
     * QDSMCAddJouleHeating heats electrons in cells where Te < m_joule_redirect_Te_eV
     * and is instead deposited to the ions (as stochastic pure-diffusion heating in
     * QDSMCApplyIonHeating) where Te >= m_joule_redirect_Te_eV. This caps electron
     * heating at the threshold and routes the rest to the ions, so Ti can exceed Te.
     * The redirected energy is bookkept per charged ion species. Requires
     * m_include_joule_heating. Enabled by specifying a threshold >= 0 via
     * joule_redirect_Te_threshold; default off (threshold < 0). The bool is
     * derived from the threshold in ReadParameters, not a user input. */
    bool m_joule_redirect_to_ions = false;
    amrex::Real m_joule_redirect_Te_eV = -1.0;   // threshold [eV]; < 0 -> off

    /** Electron-ion thermal equilibration (Q_ei) on T_e:
     *   Q_ei = 3 n_e k_B nu_ei (T_e - T_i),   dU_e/dt += -Q_ei,
     * cooling T_e toward each ion species' T_i at the rate given by the
     * electron_ion_relaxation_rate(rho,Te,Ti,t) parser, with the matching ion
     * heating deposited conservatively (QDSMCApplyIonHeating) so the exchange
     * conserves energy. Enabled by specifying the
     * electron_ion_relaxation_rate expression (the bool is derived in
     * ReadParameters, not a separate user input); only consulted when
     * m_solve_electron_energy_equation is on. */
    bool m_include_temperature_relaxation = false;

    /** Electron-ion energy-equilibration rate nu_ei(rho,Te,Ti,t) [1/s] used by
     * the Q_ei term (rho = charge density [C/m^3]; Te, Ti in eV; t in s). */
    std::string m_nu_ei_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_nu_ei_parser;
    amrex::ParserExecutor<4> m_nu_ei;

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

    /** True when the per-species deposited charge densities rho_fp_{spec}
     *  are needed, i.e. when the electron-energy equation is solved (its
     *  Joule and Q_ei sources read the per-species charge densities).
     *  Gates their allocation and the per-species deposition path in
     *  HybridPICDepositRhoAndJ; when false, the deposition falls back to
     *  the single-pass MultiParticleContainer path with zero extra cost.
     *  Set in ReadParameters. */
    bool m_need_per_species_fields = false;

    /** Plasma hyper-resisitivity */
    std::string m_eta_h_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_hyper_resistivity_parser;
    amrex::ParserExecutor<2> m_eta_h;
    bool m_include_hyper_resistivity_term = false;
    bool m_hyper_resistivity_has_B_dependence = false;

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
    std::unique_ptr<ExternalVectorPotential> m_external_vector_potential;

    /** How external fields enter the Ohm's-law kernels.
     *
     * Split (default, the explicit hybrid advance): the evolved fields hold
     * the plasma response inside the field advance and the kernels add
     * B_ext for the J x B term (and subtract E_ext from the solved E)
     * themselves. TotalAssembled: the caller hands the kernels total fields
     * already and only the E_ext subtraction applies (theta-implicit
     * consumers). UnifiedA: external fields enter through boundary values
     * of an evolved vector potential and the kernels must not touch the
     * external-field registers at all. Solvers select their convention in
     * their initialization; there is no user-facing input.
     */
    enum class ExternalFieldMode { Split, TotalAssembled, UnifiedA };
    ExternalFieldMode m_external_field_mode = ExternalFieldMode::Split;

    /** Subtract E_ext from the solved E everywhere instead of only where
     * rho >= rho_floor. The per-cell density gate is a state-dependent
     * discontinuity that breaks matrix-free Jacobian probes, so implicit
     * consumers enable the unconditional form; the explicit advance keeps
     * the gated default. */
    bool m_subtract_E_ext_everywhere = false;

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
};

#endif // WARPX_HYBRIDPICMODEL_H_
