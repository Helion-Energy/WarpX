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
     *        from the current T_e and rho_fp_temp (= rho at n+1/2). Cells with
     *        rho <= rho_floor are left at 0.
     *
     *        Reads hybrid_electron_temperature_fp (T_e in K) and
     *        hybrid_rho_fp_temp; writes hybrid_entropy_fp.
     */
    void QDSMCInitializeKe (int lev) const;

    /**
     * \brief After the QDSMC scatter, recover T_e^{n+1} from
     *           T_e = (deposited K*N) / (deposited N) / n_e^(1-gamma)
     *                 / (k_B / q_e)
     *        using rho at n+1 (rho_fp), the deposited entropy field
     *        (hybrid_entropy_fp), and the deposited weight field
     *        (hybrid_qdsmc_weights_fp).
     *
     *        Writes hybrid_electron_temperature_fp.
     */
    void QDSMCUpdateTe (int lev) const;

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
     *   5. QDSMCUpdateTe      (recover T_e^{n+1} from K_arr / N_arr / rho^{n+1})
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

    /** Electron temperature in eV */
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

    /** Ito-process tensor thermal conduction on T_e (Thrust C):
     *   dU_e/dt += div(kappa grad T_e),
     *   kappa = kappa_par b b + kappa_perp (I - b b),  b = B/|B|,
     * applied as a split substep bracketing the entropy transport
     * (QdsmcConductionOnce). Enabled by specifying the parallel-conductivity
     * parser qdsmc_kappa_par(n,Te,t) (n [m^-3], Te [eV], t [s], kappa in
     * W/(m K)); kappa_perp has its own optional parser and defaults to 0.
     * Only consulted when m_solve_electron_energy_equation is on. */
    bool m_include_thermal_conduction = false;
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
     *  tails at large Spitzer chi_par. Input:
     *  qdsmc_conduction_quadrature_points = <npts_par> [<npts_perp>]
     *  (a single value broadcasts to both). */
    int m_cond_npts_par = 2;
    int m_cond_npts_perp = 2;

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
     *  qdsmc_conduction_conserve_fixup). Input: qdsmc_conduction_form =
     *  scatter | layer. */
    int m_cond_form = 0;

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

    /** Density floor used when dividing by the deposited QDSMC weight to
     * recover K_e^{n+1} from (K*N) / N. Avoids divide-by-zero in cells
     * that no QDSMC particle reached during the push. Defaults to the
     * same floor as the rest of the hybrid solver. */
    amrex::Real m_qdsmc_n_floor = 1.0;

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
