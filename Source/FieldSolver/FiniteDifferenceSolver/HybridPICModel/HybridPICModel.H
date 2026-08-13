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
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;

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
     *        the deposited weight field (hybrid_qdsmc_weights_fp). Cells
     *        whose deposited weight is at or below m_qdsmc_n_floor keep
     *        their previous T_e.
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
     *  cap and no vacuum fast front (neither concept applies). */
    void QdsmcConductionOnceFD (int lev, amrex::Real dt_c, bool use_rho_new) const;

    /** Grid-transport arm of the energy-equation bake-off jig
     *  (qdsmc_transport_operator = grid): advances the entropy pair
     *  y = {n_e, K n_e} in conservative flux form dy/dt = -div(y V_e)
     *  with SMART-limited upwind face values and the shared adaptive RK
     *  integrator (V_e frozen over the transport slot, like the marker
     *  push). Writes the marker-deposit convention (entropy_fp = K n_e,
     *  weights = n_e) and calls QDSMCUpdateTe -- identical recovery, so
     *  the bake-off isolates the transport operator alone. */
    void QdsmcTransportOnceGrid (int lev, amrex::Real dt_adv) const;

    /** leapfrog self-start: false until the first advance has run (which
     *  advances by dt/2 to move T_e^0 to the half level). */
    mutable bool m_qdsmc_leapfrog_started = false;
    /** leapfrog: false until hybrid_qdsmc_pe_prev_fp holds a valid previous
     *  Pe level (Pe^0 on the first step, Pe^{n-1/2} afterward). */
    mutable bool m_qdsmc_pe_prev_valid = false;

    /** Resistive electron-heating source on T_e (see Phys. Plasmas 31, 012902 (2024), Eq. 12),
     * computed per cell as
     *     S_e = Sigma_s nu_{s,e} n_s m_s |dV|^2,
     *         nu_{s,e} = Z_s e^2 eta n_e / m_s,   dV = J_plasma / (e n_e),
     * from the nodal plasma current, the per-species charge densities and
     * the Ohm's-law eta parser. Single bool: on or off. Reduces exactly to
     * eta J^2 in single species. Only consulted when
     * m_solve_electron_energy_equation is also true. */
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
     * m_include_joule_heating. Enabled by specifying a threshold >= 0 via
     * joule_redirect_Te_threshold; default off (threshold < 0). The bool is
     * derived from the threshold in ReadParameters, not a user input. */
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
    /** FD cross-flux transport reconstruction: 0 = none (unlimited QUICK,
     *  attribution control -- can overshoot), 1 = upwind1 (1st-order
     *  upwind, most diffusive), 2 = smart (Gaskell--Lau SMART, default:
     *  up to 3rd order, locally monotone). Input:
     *  qdsmc_conduction_fd_limiter = none | upwind1 | smart. */
    int m_cond_fd_limiter = 2;
    /** FD subcycle CFL fraction of the Gershgorin stability bound
     *  (sum_gh 2 |Xi_gh| dxi_g dxi_h, n-ratio weighted). Default 0.4:
     *  under SSP-RK2 the Nyquist amplification G = 1 + z + z^2/2 never
     *  flips sign but goes NEUTRAL (G -> 1, zero damping) at the
     *  stability edge, so deposit-noise checkerboards persist instead of
     *  damping -- the reference algorithm's ledger-40 lesson (alpha_sub <= 0.25, i.e.
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
     *  operators). Both are error-adaptive on top of the stability
     *  ceiling. Input: qdsmc_conduction_fd_time = ssprk2 | rkf45. */
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
     *  = exact adiabatic default). Input: qdsmc_conduction_form =
     *  scatter | layer | fluxform. DEFAULT fluxform (the gate-passed
     *  production form; the only one with the closed floor faces and the
     *  per-line EB machinery -- next to a maintained wall band the
     *  mask-less control arms act as a perpetual bath donor). */
    int m_cond_form = 2;

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
     *  qdsmc_conduction_compensate. */
    bool m_cond_compensate = false;

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
     *  qdsmc_conduction_slope_limiter = mc | none. */
    int m_cond_slope_limiter = 0;

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
     *  only; with qdsmc_conduction_fluxform_unsplit the default
     *  silently reverts to plm (the unsplit piece bookkeeping is
     *  PLM-exact) — explicitly requesting ppm + unsplit aborts. Input:
     *  qdsmc_conduction_reconstruction = plm | ppm. */
    int m_cond_reconstruction = 1;

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
     *  Input: qdsmc_conduction_fluxform_unsplit. */
    bool m_cond_ff_unsplit = false;

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
     *  energy added to the plasma). The G4 budget-closure instrument;
     *  exposed to Python as warpx.get_qdsmc_wall_tally(dim, side). */
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
     *  to the plasma. */
    mutable amrex::Real m_cond_eb_tally = 0.0;

    amrex::Real GetQdsmcEbTally () const { return m_cond_eb_tally; }

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
