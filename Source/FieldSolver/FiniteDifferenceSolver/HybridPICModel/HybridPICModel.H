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

#include "EBJBoundary.H"
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

#include <AMReX_GpuQualifiers.H>

#include <cmath>
#include <map>
#include <memory>
#include <optional>

/** C1 smooth-max blend of the density-floor guard: exact max(rho,
 *  rho_floor) at w_abs = 0 (legacy hard floor, bit-compatible), the
 *  smooth-max 0.5*(rho + rho_floor + sqrt((rho-rho_floor)^2 + w_abs^2))
 *  otherwise. Bounded below by rho_floor everywhere like the hard max
 *  (the halo response stays floored -- this is not a response mask),
 *  but infinitely differentiable, so deposit noise chattering across
 *  the threshold no longer kinks the Ohm residual. w_abs is the
 *  absolute (charge-density) blend width. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real HybridSmoothFloor (amrex::Real rho, amrex::Real rho_floor,
                               amrex::Real w_abs)
{
    if (w_abs <= amrex::Real(0.)) { return amrex::max(rho, rho_floor); }
    const amrex::Real d = rho - rho_floor;
    return amrex::Real(0.5)*(rho + rho_floor + std::sqrt(d*d + w_abs*w_abs));
}

/** Weight of the conditional external-E subtraction across the density
 *  floor: the legacy step (rho >= rho_floor) at w_abs = 0, a tanh blend
 *  of the same width as the smooth floor otherwise (the two share the
 *  hybrid_pic_model.n_floor_smooth_width knob -- they are the same
 *  crossing and must smooth together). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real HybridExtSubWeight (amrex::Real rho, amrex::Real rho_floor,
                                amrex::Real w_abs)
{
    if (w_abs <= amrex::Real(0.)) {
        return (rho >= rho_floor) ? amrex::Real(1.) : amrex::Real(0.);
    }
    return amrex::Real(0.5)*(amrex::Real(1.)
        + std::tanh((rho - rho_floor)/w_abs));
}

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

    /** Constitutive-PEC zeroing of an edge-staggered field on the conformal
     *  (ECT) wall: zero every masked (fully covered) edge and every cut edge.
     *  Applied to the Ohm's-law E and the Ampere current. */
    void ZeroConductorEdges (
        ablastr::fields::VectorField const& field,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
        int lev) const;

    /** Zero fully covered faces (open area = 0) of a face-staggered field on
     *  the conformal (ECT) wall; cut faces keep their open-part flux. Applied
     *  to the external B so the conductor excludes the time-varying flux. */
    void ZeroCoveredFaces (
        ablastr::fields::VectorField const& field,
        int lev) const;

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
     */
    void QDSMCInitializeUe (int lev,
        amrex::MultiFab const & rho,
        ablastr::fields::VectorField const & J_i,
        ablastr::fields::VectorField const & Jp_new,
        ablastr::fields::VectorField const & Jp_old,
        amrex::Real jp_interp) const;

    /**
     * \brief Fill the nodal K_e = T_e * n_e^(1-gamma) * (k_B / q_e) MultiFab
     *        from the current T_e and rho_fp_temp (= rho at n+1/2). Cells with
     *        rho <= rho_floor are left at 0.
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
     *        using rho at n+1 (rho_fp), the deposited entropy field
     *        (hybrid_entropy_fp), and the deposited weight field
     *        (hybrid_qdsmc_weights_fp).
     *
     *        Writes hybrid_electron_temperature_fp.
     */
    void QDSMCUpdateTe (int lev) const;

    /** \brief Parameterized variant of QDSMCUpdateTe: recover T_e from the
     *         deposited entropy/weights and the given new-time charge density
     *         (the extrapolated rho^{n+1} iterate on the theta-implicit path).
     */
    void QDSMCUpdateTe (int lev, amrex::MultiFab const & rho_new) const;

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
     * \brief One isotropic QDSMC conduction half-pass over the interval h
     *        (Phase 3a; Strang-wrapped around the entropy-advection stage
     *        by the callers).
     *
     *        Per level: build the nodal internal-energy density
     *        U = (3/2) n_e k_B T_e and diffusivity
     *        D = 2 kappa(T_e, n_e) / (3 n_e k_B) (kappa from the
     *        qdsmc_conduction_kappa parser) plus grad(D) by central
     *        differences; spawn-kick the Gauss-Hermite conduction nodes
     *        (see QdsmcParticleContainer::SpawnConductionNodes); deposit
     *        the transported energy; recover
     *        T_e = (2/3) U_new / (n_e k_B) into
     *        hybrid_electron_temperature_fp. Unconditionally stable and
     *        positivity-preserving (positive weights, conservative
     *        deposit); no conduction CFL.
     *
     * \param h        pass interval [s] (dt/2 per Strang half)
     * \param rho_type field register holding the charge density that
     *                 defines n_e for this pass
     * \param rho_comp component of that field to read
     */
    void QdsmcConductionPass (amrex::Real h,
                              warpx::fields::FieldType rho_type,
                              int rho_comp) const;

    /**
     * \brief PROTOTYPE (WARPX_COND_FUSED): fused composition of the
     *        conduction increment with the advection stage. The
     *        kicked-minus-unkicked energy-deposit difference (in the
     *        stage's K*N units) is added into the stage's deposited
     *        entropy BEFORE its single N-ratio recovery, so conduction
     *        and advection share ONE gather-deposit remap instead of
     *        exchanging spectral power across two.
     *
     * \param dt        full step (the increment is applied once per step)
     * \param Karr      the stage's deposited K*N field, incremented
     * \param rho_type  density register defining n_e for the kernel
     * \param rho_comp  component of that register
     */
    void QdsmcConductionFusedIncrement (amrex::Real dt,
                                        amrex::MultiFab & Karr,
                                        warpx::fields::FieldType rho_type,
                                        int rho_comp) const;

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

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
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

    /** Smooth transition width of the Holmstrom vacuum gate, as a fraction
     * of n_floor. With width w > 0 the Hall + grad-Pe part of the Ohm's-law
     * E is multiplied by 0.5*(1 + tanh((rho - rho_floor)/(w*rho_floor)))
     * instead of switching off discontinuously below the floor: the binary
     * branch makes the implicit residual discontinuous in the state where
     * cells straddle the gate (Newton limit cycles, grid-scale E jumps at
     * the plasma edge). 0 (default) keeps the legacy hard branch. */
    amrex::Real m_holmstrom_transition_width = 0.0;

    /** Independent Ohm's-law term controls: whether the Hall contribution
     * (the curl-B total current in the (J - J_i) x B / (e n) term; with it
     * off the term reduces to the ideal -u_i x B motional field) and the
     * electron-pressure gradient enter the E-field solve. Both default on;
     * turning them off makes e.g. a species-free resistive-vacuum deck
     * exactly linear (used by the open-boundary Green's-function tests). */
    bool m_include_hall_term = true;
    bool m_include_electron_pressure_term = true;

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

    /** Drop the advective (Je.grad)u and u drho/dt legs of the inertia
     *  term, keeping only the dJe/dt leg. With electron_inertia_bdf2 = 0
     *  and the measured history store, the remaining term is algebraically
     *  the operator form -d_e^2 curl curl E (the two-point dJe/dt with the
     *  measured Je^n equals (Je^theta - Je^n)/(theta dt), which is exactly
     *  the theta-stage Faraday-Ampere identity): the generalized Ohm's law
     *  then rolls the whistler over at the electron skin depth with no
     *  bare 1/n on a deposit [Amano, Higashimori & Shirakawa, J. Comput.
     *  Phys. 275, 197 (2014)]. */
    bool m_electron_inertia_djedt_only = false;
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

    /** Te-threshold Joule redirection. When true, the eta*J^2 Joule source in
     * QDSMCAddJouleHeating heats electrons in cells where Te < m_joule_redirect_Te_eV
     * and is instead deposited to the ions (as stochastic pure-diffusion heating in
     * QDSMCApplyIonHeating) where Te >= m_joule_redirect_Te_eV. This caps electron
     * heating at the threshold and routes the rest to the ions, so Ti can exceed Te.
     * The redirected energy is bookkept per charged ion species. Requires
     * m_include_joule_heating. Default off. */
    bool m_joule_redirect_to_ions = false;
    amrex::Real m_joule_redirect_Te_eV = 100.0;   // threshold [eV]

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

    /** Density floor used when dividing by the deposited QDSMC weight to
     * recover K_e^{n+1} from (K*N) / N. Avoids divide-by-zero in cells
     * that no QDSMC particle reached during the push. Defaults to the
     * same floor as the rest of the hybrid solver. */
    amrex::Real m_qdsmc_n_floor = 1.0;

    /** QDSMC electron thermal conduction (Phase 3a: isotropic scaffold).
     *  "off" (default) or "isotropic". Requires
     *  solve_electron_energy_equation. Strang-wrapped around the
     *  entropy-advection stage; see QdsmcConductionPass. */
    std::string m_qdsmc_conduction = "off";

    /** Electron thermal conductivity kappa(T, n) [W/(m K)] for the QDSMC
     *  conduction pass (T in Kelvin, n in 1/m^3). */
    std::string m_qdsmc_kappa_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_qdsmc_kappa_parser;
    amrex::ParserExecutor<2> m_qdsmc_kappa;

    /** Substeps per conduction half-pass. The Gaussian kernel is exact in
     *  time for frozen coefficients; substeps only refine the coefficient
     *  variation of a strongly nonlinear kappa(T). */
    int m_qdsmc_conduction_substeps = 1;

    /** Free-streaming flux-limiter coefficient alpha (0 = off, default).
     *  The parallel/isotropic diffusivity is blended harmonically against
     *  the free-streaming bound q_fs = alpha n k_B T v_th,e:
     *  D_eff = D / (1 + (3/2) D |grad T| / (alpha T v_th,e)), evaluated
     *  per node at the pass state. */
    amrex::Real m_qdsmc_conduction_flux_limiter = 0.0;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;

    /** Width of the C1 smooth-max blend of the density floor, in units of
     *  n_floor (0 = exact hard max, bit-compatible legacy behavior). The
     *  hard max(rho, rho_floor) is a C0 kink in the Ohm residual at the
     *  floor crossing (and the conditional external-E subtraction a jump);
     *  cells whose deposit chatters across the threshold stall a sharp
     *  Newton iteration on it. The blend keeps the halo response FLOORED
     *  (bounded exactly like the hard max -- this is NOT the Holmstrom
     *  vacuum mask, which suppresses the halo response) but makes the
     *  residual smooth. */
    amrex::Real m_n_floor_smooth_width = 0.0;

    /** Opt-in legacy behavior: store the theta-extrapolated Je^{n+1} in
     *  the inertia history instead of the measured delivered-state
     *  assembly. The extrapolated store is a -(1-theta)/theta recursion
     *  on the stored values -- marginal (-1) at theta = 1/2 -- and rings
     *  at period 2 with the E_inertial feedback. Comparison knob only. */
    bool m_electron_inertia_extrapolated_history = false;

    /** Width (in units of n_floor) of the electron-inertia fluid-validity
     *  taper: the inertia term fades to zero through the floor band, so
     *  density-floored cells holding O(1) particles cannot self-force
     *  runaway through (m_e/e) dJe/dt / rho_lim. 0 disables (legacy). */
    amrex::Real m_electron_inertia_floor_taper = 0.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** Joule-heating resistivity split. The QDSMC electron Joule source
     *  evaluates joule_heating_resistivity(rho,J,t) instead of the
     *  E-solve resistivity; default (unset) aliases plasma_resistivity,
     *  bit-identical. Decks whose plasma_resistivity carries a
     *  density-ramped vacuum regularization should set this to the
     *  physical (un-boosted) eta: the ramp exists to relax the vacuum
     *  FIELD, and heating with it deposits eta_vac*J^2 into just-above-
     *  floor cells whose heat capacity scales with n -- a numerical
     *  electron-temperature source at the plasma edge. */
    std::string m_eta_heating_expression;
    std::unique_ptr<amrex::Parser> m_heating_resistivity_parser;
    amrex::ParserExecutor<3> m_eta_heating;

    /** Density gate (m^-3) below which the Joule source is dropped;
     *  default (negative) = n_floor (the historical gate). Raise it to
     *  keep heating out of the resistivity-ramp band without moving the
     *  Ohm's-law floor. */
    amrex::Real m_joule_heating_n_min = -1.0;

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

    /** Opt-in conformal embedded-boundary wall: the enlarged-cell (ECT)
     *  Faraday update on cut faces with a constitutive-PEC closure (E and J
     *  zeroed on covered and cut edges). Requires embedded boundaries, a
     *  staggered (Yee) grid, and 3D or 2D (XZ) Cartesian geometry. */
    bool m_use_conformal_eb = false;

    /** Wall constitutive model for the conformal EB wall: "conductor"
     *  (default; E and J zeroed on covered and cut edges, Dirichlet Pe)
     *  or "transparent" (EM-transparent standoff: the fields ignore the
     *  wall entirely while the EB geometry and particle handling stay
     *  identical -- a validation mode that separates "wall eats plasma"
     *  from "wall shorts fields"). */
    std::string m_conformal_wall_model = "conductor";
    bool m_conformal_wall_conductor = true;

    /** Cached EB fill classification for the deposited edge fields (ion
     *  current fill + deposit fold), one entry per level; reset on regrid.
     *  Mutable: (re)built inside const solver calls. */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_E;

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
    std::unique_ptr<QdsmcParticleContainer> m_qdsmc_cond_pc;

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
