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

    /** Thin resistive-shell wall (RZ r-max, see m_resistive_wall): overwrite
     *  the tangential wall-node rows of Efield with the shell value computed
     *  from the CURRENT Bfield iterate,
     *      E_theta(wall) = (eta_w/delta) * K_theta
     *                    = (eta_w/delta) * (B_z(inside) - B_z,ext) / mu0,
     *      E_z(wall)     = (eta_w/delta) * K_z
     *                    = -(eta_w/delta) * B_theta(inside) / mu0,
     *  with B(inside) read at the last interior cell-centered-in-r row, and
     *  image the r-hi tangential ghosts odd ABOUT that value
     *  (ghost = 2*E_wall - interior mirror: the nonzero-Dirichlet image).
     *  E_r (normal) is left with its PEC treatment. Must run AFTER
     *  ApplyEfieldBoundary (it owns the wall row + tangential r-hi ghosts;
     *  composition order is stated at the call sites). No-op unless
     *  m_resistive_wall is set and the r-hi field boundary is PEC. */
    void ApplyResistiveWallE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Bfield,
        int lev) const;

    /** Thin resistive-shell wall, iterate-conditioning leg (RZ r-max, see
     *  m_resistive_wall): restore the tangential wall-node rows of Efield
     *  from the unconditioned source Esrc (the nonlinear-solver iterate,
     *  which ApplyEfieldBoundary's PEC kernel has just zeroed) and image the
     *  r-hi tangential ghosts odd about that value. Keeping the ITERATE's
     *  wall value -- rather than recomputing the shell value from the stale
     *  Bfield_fp of the previous residual evaluation -- makes the residual
     *  map a deterministic function of the iterate (no hidden one-evaluation
     *  B lag in Jacobian probes); the wall row converges to the shell value
     *  through the residual, whose Ohm output imposes it via
     *  ApplyResistiveWallE. E_r (normal) unchanged. */
    void ImposeResistiveWallIterateRowsE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Esrc,
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

    /** Treat the r-max PEC wall node row as a conductor row in the Ohm
     *  solve (RZ): the conductor row carries no plasma moments -- ne,
     *  tangential J, and the Hall force are all zero there. The full
     *  wall-moment closure stack applied under this knob is
     *   (a) the nodal (J - J_i) x B force is zeroed on the wall node plane
     *       (HybridPICSolveECylindrical pass 1), so the last interior
     *       half-cell of E_r receives no Hall/motional contribution
     *       interpolated from the conductor surface;
     *   (b) the tangential (theta, z) components of BOTH the Ampere and
     *       ion currents are zeroed on the wall node plane and its ghost
     *       rows (HybridPICSolveE wrapper, after the SyncCurrentAndRho
     *       guard folds), removing the spurious electron current sheet
     *       J_e = J - J_i built from the deposit-fold / one-sided-curl
     *       parity mismatch;
     *   (c) ne-Dirichlet wall moment: the deposited charge density the
     *       Ohm divide consumes is zeroed on the wall node plane with
     *       odd-imaged ghost rows (ghost = -interior mirror), in the same
     *       wrapper, immediately before the Ohm solve;
     *   (d) gate-on-raw / divide-by-floored Hall (HybridPICSolveE Yee
     *       pass): where the raw (unfloored) rho at the E stencil location
     *       is not positive, the Hall/motional term is identically zero --
     *       where(rho_raw > 0, enE/max(rho, rho_floor), 0) -- so with (c)
     *       the wall row's zeroed numerator is NOT amplified into an
     *       enE/rho_floor response by the floored divide, at and beyond
     *       the wall row.
     *  Without the stack, the tangential Ampere current's cylindrical
     *  curvature remnant at the wall (J_z = B_theta/(mu0 r), a genuine
     *  surface current) and the reflecting-fold ion-deposit parity combine
     *  into a spurious electron-current sheet whose Hall response the
     *  theta = 1/2 advance integrates without damping - a secularly
     *  growing whistler-polarized wall layer whenever plasma touches the
     *  wall. Conductor-surface rows are not plasma Ohm unknowns and carry
     *  no plasma moments or Hall force. */
    bool m_pec_conductor_wall_rows = false;

    /** Thin resistive-shell wall at the RZ r-max PEC boundary. The hard PEC
     *  identity row E_t(wall) = 0 responds instantaneously within each
     *  implicit solve; this knob replaces it with a finite-response-time
     *  conducting shell of resistivity eta_w (m_wall_resistivity) and
     *  thickness delta (m_wall_thickness), carrying surface current K
     *  determined by the tangential-B jump across the shell,
     *      r_hat x (B_out - B_in) = mu0 K, with
     *      B_out = B_z,ext z_hat (the shell is the flux boundary:
     *              B_theta(outside) = 0, B_z(outside) = the applied uniform
     *              field, m_wall_bz_ext), giving
     *      K_z     = -B_theta(inside) / mu0,
     *      K_theta = +(B_z(inside) - B_z,ext) / mu0,
     *  and Ohm's law in the shell sets the tangential electric field there,
     *      E_t(wall) = (eta_w / delta) * K_t.
     *  This is a relaxation (Robin) row computed from the B iterate inside
     *  every residual evaluation of the theta-implicit hybrid -- in-residual
     *  by construction -- instead of the hard Dirichlet identity row. The
     *  shell L/R decay time for a mode of wavenumber k is
     *      tau = mu0 * delta / (2 * k * eta_w);
     *  at copper resistivity (the default, 1.68e-8 Ohm m) and delta = dr the
     *  grid-scale-mode tau is ~1e4 ion-cyclotron periods on typical hybrid
     *  decks: physically near-PEC but structurally relaxational. Ghost rows
     *  beyond the wall are imaged odd about the wall value
     *  (ghost = 2*E_wall - interior mirror). E_r (normal) keeps the PEC
     *  treatment. Independent of (and composable with)
     *  m_pec_conductor_wall_rows, which edits the Ohm-solve INPUT moments,
     *  not the wall E row. Only implemented for RZ r-max with a PEC r-hi
     *  field boundary; applied at both E boundary-application sites of the
     *  theta-implicit residual (see ApplyResistiveWallE /
     *  ImposeResistiveWallIterateRowsE). */
    bool m_resistive_wall = false;
    /** Shell resistivity eta_w [Ohm m]; default = copper (1.68e-8). */
    amrex::Real m_wall_resistivity = 1.68e-8;
    /** Shell thickness delta [m]; <= 0 (default) means use the radial cell
     *  size dr of the level the wall row is applied on. */
    amrex::Real m_wall_thickness = -1.0;
    /** Applied uniform external B_z [T] outside the shell, subtracted in
     *  K_theta = (B_z(inside) - B_z,ext)/mu0. The applied field of this
     *  class of decks is loaded directly into Bfield_fp (grid init style),
     *  not into the hybrid external-field registry, so it is not
     *  recoverable from m_add_external_fields here; decks must pass their
     *  applied B0 explicitly (default 0: the shell then relaxes the TOTAL
     *  enclosed B_z flux). */
    amrex::Real m_wall_bz_ext = 0.0;

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
    /** E-solve form: false = legacy algebraic solve with the floored
     *  density division ("e_form"); true = the division-free
     *  multiplied-through solve ("curlcurl_form"): the dJe/dt leg's
     *  E-dependence becomes the elliptic operator (beta + curl curl) with
     *  beta = mu0 e^2 n / m_e_eff unfloored, solved per residual
     *  evaluation (theta-implicit only). Toroidal RZ m=0 sector first. */
    bool m_esolve_curlcurl = false;

    /** \brief Division-free (curlcurl_form) toroidal E-solve, RZ m = 0.
     *
     * Solves the multiplied-through generalized Ohm's law for E_theta:
     *
     *     (beta(x) + curlcurl_theta) E_theta = (mu0 e / m_e_eff) num_theta,
     *
     * with beta = mu0 e^2 n / m_e_eff evaluated from the UNfloored
     * midpoint density and curlcurl_theta = -(1/r) d_r (r d_r .) - d_z^2 .
     * + 1/r^2 (the theta component of the vector curl-curl). The RZ
     * metric is absorbed into a Cartesian-form nodal ABec operator by
     * multiplying the equation by r:
     *
     *     [r beta + 1/r] E - div( r grad E ) = r * rhs,
     *
     * i.e. MLNodeABecLaplacian with nodal acoef = r beta + 1/r (>= 1/r,
     * never singular; the axis row is a Dirichlet regularity boundary)
     * and cell-centered bcoef = r. E_theta's RZ Yee staggering is nodal
     * in (r, z), so the solve operates on the field's native staggering.
     *
     * \param[in,out] Etheta   E_theta MultiFab (nodal); solution in place
     * \param[in]     num      multiplied-through numerator (nodal, no
     *                         density division anywhere)
     * \param[in]     rho      charge density (cell-centered) for beta
     * \param[in]     rho_comp component of rho to use (midpoint)
     * \param[in]     lev      level (must be 0)
     */
    void SolveEThetaCurlCurlRZ (amrex::MultiFab& Etheta,
                                amrex::MultiFab const& num,
                                amrex::MultiFab const& rho,
                                int rho_comp, int lev) const;

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

    /** Resolve (once) the effective electron inertia mass m_e_eff from the
     *  lightest charged ion species and reduced_electron_mass_ratio
     *  (physical m_e when the ratio is unset); no-op once resolved. */
    void ResolveElectronInertiaMass ();

    /** Allocate (once) the curlcurl_form scratch MultiFabs consumed by the
     *  Ohm kernel and the elliptic solves; no-op on e_form. Must run before
     *  ANY HybridPICSolveE call on the curlcurl path: the resistive
     *  push-field correction solves in every residual evaluation BEFORE the
     *  inertia assembly that used to allocate these lazily (stale scratch
     *  content cancels exactly in the correction's two-pass difference;
     *  unallocated arrays do not). */
    void EnsureCurlCurlScratch ();

    /** curlcurl_form scratch (see m_esolve_curlcurl): the step-start plasma
     *  current (staggered, from curl B_old) and its nodal theta component
     *  -- the "measured" reference the dJe/dt leg differences against so
     *  that the current iterate's curl-curl-E content stays on the
     *  operator side of the solve (differencing against the iterate's own
     *  B(theta) would cancel the operator at the fixed point and
     *  degenerate the solve to an unfloored pointwise division). Captured
     *  once per step (same latch discipline as the frozen rho^n). */
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_jp_old_stag;
    std::unique_ptr<amrex::MultiFab> m_jp_old_nodal;
    /** curlcurl_form: the division-free measured inertia numerator for the
     *  toroidal solve, (m_e/e) dJe_meas/dt (nodal, one component). */
    std::unique_ptr<amrex::MultiFab> m_ei_curlcurl_theta;
    /** curlcurl_form: the multiplied-through toroidal numerator captured by
     *  the Ohm kernel (nodal, one component; the elliptic solve's RHS). */
    std::unique_ptr<amrex::MultiFab> m_num_theta;
    /** curlcurl_form stage 2: the poloidal numerators on the E_r and E_z
     *  staggerings (the poloidal curl-curl system's RHS). */
    std::array<std::unique_ptr<amrex::MultiFab>, 2> m_num_pol;

    /** \brief curlcurl_form stage 2: coupled poloidal (E_r, E_z) solve,
     *  RZ m = 0, on the fields' native staggerings (E_r: cc in r / nodal
     *  in z; E_z: nodal in r / cc in z).
     *
     * Solves the multiplied-through system
     *
     *     beta(x) E + (curlcurl E)|_pol = (mu0 e/m_e_eff) num
     *
     * with the TRUE poloidal curl-curl composed from the discrete curls:
     * W = (curl E)_theta = d_z E_r - d_r E_z lands on cell centers, then
     * (curlcurl E)_r = -d_z W and (curlcurl E)_z = (1/r) d_r (r W). A
     * component Laplacian equals this only where the discrete div E
     * vanishes -- at machine-level nonzero div it spuriously screens the
     * longitudinal field at density-contrast edges (curlcurl = grad div
     * - lap_vec: the old component solves are the div-free limit of this
     * operator).
     *
     * Rows are scaled by the control volumes V_r = r_c dr and V_z = r_n
     * dr (E_z axis node: dr^2/8): the curl-curl block is then the exact
     * gradient of (1/2) sum_cells r_c dr dz W^2 -- symmetric positive
     * semi-definite by construction -- and beta >= 0 stays unfloored (no
     * pedestal). Where the unfloored density vanishes exactly, the
     * multiplied-through equation is 0 = 0: those RHS rows are zeroed (a
     * nonzero numerator there is stencil-support bleed from the wider
     * nodal interpolations) and the operator is closed with the vacuum
     * Gauss constraint div E = 0, imposed as the energy-form
     * -grad(gamma div .) completion with gamma = 1 on strictly-vacuum
     * NODES only (nodal unfloored density <= 0; axis, wall, and
     * non-periodic cap nodes stay ungated). Deep vacuum then carries the
     * full vector Laplacian (curlcurl - grad div), eliminating the bare
     * curl-curl's gradient null space (which otherwise makes CG grind),
     * and the system is SPD on the free rows. Every row that carries
     * plasma keeps the pure identity form: the closure lives on
     * strictly-vacuum nodes, so density-sheet rows -- whose spurious E_L
     * screening under the component Laplacian motivated this operator --
     * couple to it only as boundary data of the vacuum region.
     *
     * Boundary rows follow the toroidal solve's conventions: the E_r
     * z-cap nodes and the E_z wall node (r_hi) are Dirichlet identity
     * rows (PEC tangential E); the E_r wall row and the E_z axis row are
     * natural energy-form rows (the W cell beyond the wall / inside the
     * axis does not exist); z periodic through ghosts. Solved by
     * Jacobi-preconditioned CG on the two-field block system.
     *
     * hybrid_pic_model.esolve_pol_verify = 1 runs a one-shot discrete
     * verification on the first call: the fused operator applied to a
     * deterministic pseudo-random poloidal field is compared against the
     * directly composed curl(curl .) through a stored W field, printing
     * the max relative row error per sector (roundoff on interior and
     * natural-boundary rows; Dirichlet rows identity by construction).
     */
    void SolveEPolCurlCurlRZ (amrex::MultiFab& Er,
                              amrex::MultiFab& Ez,
                              amrex::MultiFab const& num_r,
                              amrex::MultiFab const& num_z,
                              amrex::MultiFab const& rho,
                              int rho_comp, int lev) const;

    /** One-shot discrete verification of the poloidal curl-curl assembly
     *  (see SolveEPolCurlCurlRZ). */
    bool m_esolve_pol_verify = false;
    mutable bool m_esolve_pol_verified = false;
    bool m_inertia_jpold_captured = false;

    /** curlcurl_form stage 2: evaluate the poloidal stage's density-derived
     *  operator content -- the beta rows, the zero-density RHS membership
     *  test, the vacuum-closure gamma gate, and the numerator fold weights
     *  -- from the per-step-frozen rho^n snapshot ("hybrid_rho_n_frozen",
     *  captured on the first non-Jacobian evaluation of each step, the
     *  same latch the inertia drho/dt leg uses) instead of the live
     *  per-evaluation midpoint deposit. The binary gates are otherwise
     *  re-decided from a fresh deposit in EVERY residual evaluation, so
     *  near-threshold cells flip membership under finite-difference
     *  Jacobian probe deposits (an O(1) operator-row change at probe
     *  amplitude poisons the JVP secants) and between Newton/line-search
     *  evaluations (compounding near-tolerance misses on sharp-edge cold
     *  starts). The beta rows and fold weights freeze TOGETHER with the
     *  gates: a frozen-plasma gate over a live-zero beta row would reopen
     *  the curl-curl gradient null family (gamma stays off exactly where
     *  beta no longer covers the row) on edge cells whose shape-function
     *  tail vacates mid-step. Consumers fall back to the live density
     *  until the first capture of a run (initialization solves) and when
     *  the knob is off (= 0, the legacy per-evaluation coupling, A/B
     *  only). */
    bool m_curlcurl_pol_frozen_rho = true;

    /** E-solve form "tensor_form": the stateless Je elimination on the
     *  curl-curl base. The theta-level electron momentum equation
     *  (backward Euler over dt_eff = theta dt, division-free, Amano,
     *  J. Comput. Phys. 275, 197 (2014) class) is solved for Je' in
     *  closed form, (a I - [beta]x) Je' = r with a = 1 + dt_eff kE eta,
     *  beta = dt_eff qm' B_tot, kE = qm' rho (qm' = e/m_e' with the
     *  Eq-(27)-class effective mass), and Je' is eliminated through the
     *  theta-stage Ampere/Faraday identity into the E-only operator
     *
     *      (1/mu0) curlcurl E + kE M^{-1} E
     *          = M^{-1} [ kE eta Jp^n - qm' B x Je_meas - qm' grad Pe ],
     *
     *  M^{-1} = (a^2 I + a [beta]x + beta beta^T) / (a (a^2 + |beta|^2)),
     *  Jp^n the step-start plasma current (frozen per step), Je_meas =
     *  Jp^n - Ji the measured electron current. T(x) = kE M^{-1} is the
     *  pointwise 3x3 screening tensor: |beta|-magnitude = the d_e^-2
     *  coefficient, the [beta]x part the (now point-implicit) gyration
     *  that e_form/curlcurl_form carry as the lagged Hall numerator, the
     *  diagonal the resistive friction. Every coefficient scales with
     *  density moments: T -> 0 smoothly in vacuum and the operator tends
     *  to pure curl-curl (Laplace-class continuation) -- no floor, no
     *  eta_vac, no pedestal in the E assembly. Stateless by construction:
     *  Je is derived from the iterate, never stored (nothing to roll back
     *  in rejected iterates/line search, nothing new to checkpoint).
     *  v1: 1D_Z transverse sector (the RZ m = 0 assembly on the landed
     *  curlcurl_form elliptic base follows). Theta-implicit hybrid only. */
    bool m_esolve_tensor = false;
    /** Eq-(27)-class CFL fraction of the tensor_form effective electron
     *  mass (alpha in m_e' = max(m_e, B^2 q_e/(mu0 rho_g)
     *  (dt_eff/(2 alpha dx_min))^2)). */
    amrex::Real m_tensor_alpha = 0.5;
    /** One-count density anchor [m^-3] of the effective-mass guard (a
     *  STATISTICS anchor: it guards the rho division in the Eq-27 formula
     *  at deposit-starved cells and must be the one-macroparticle-per-cell
     *  density; physics features must never anchor here). 0 falls back to
     *  n_floor. */
    amrex::Real m_tensor_n_min = 0.0;
    /** Effective theta-interval dt_eff = theta dt of the tensor
     *  elimination; published once per step by the theta-implicit solver
     *  (a committed per-step scalar, constant through all residual
     *  evaluations). */
    amrex::Real m_tensor_dt_eff = 0.0;
    /** tensor_form scratch: the step-start plasma current Jp^n on the E
     *  staggerings, captured once per step from the entry B^n (frozen --
     *  committed pre-step state, per-step constant through all residual
     *  evaluations). Zero init = the exact quiescent pre-history for
     *  solves that run before the first capture. Mutable: allocated
     *  lazily inside the const solve path. */
    mutable std::array<std::unique_ptr<amrex::MultiFab>, 3> m_tensor_jp_old;
    /** tensor_form (RZ) scratch: the per-step-frozen density snapshot
     *  (rho_fp component 0 at step entry -- the committed-entry family the
     *  darwin vacuum-recovery mask freezes). The RZ assembly derives EVERY
     *  rho-dependent quantity from this one bit-frozen field -- the kE
     *  screening rows, the Eq-(27) effective-mass guard, eta(rho), the
     *  vacuum-closure gamma gate, and the RHS membership test -- so gate
     *  membership and screening coefficients cannot flip between residual
     *  evaluations (the frozen-rho-gates iron rule; a frozen gate over a
     *  live-zero screening row would reopen the poloidal curl-curl
     *  gradient null family). The 1D solve keeps its validated live
     *  comp-0 convention. */
    mutable std::unique_ptr<amrex::MultiFab> m_tensor_rho_old;
    /** Set by the first CaptureTensorStepStart of the run; solves before
     *  it (initialization) fall back to the live density. */
    bool m_tensor_rho_captured = false;

    /** One-shot discrete verification of the tensor-Helmholtz assembly
     *  (hybrid_pic_model.tensor_verify, the esolve_pol_verify pattern):
     *  the fused operator on a deterministic multi-mode field vs the
     *  independently composed (1/mu0) curl(curl F) (two staggered curls
     *  through a stored cell-centered intermediate) plus kE M^{-1} F with
     *  M inverted by a general 3x3 adjugate; prints the max row error per
     *  component normalized by the row magnitude (roundoff interior). */
    bool m_esolve_tensor_verify = false;
    mutable bool m_esolve_tensor_verified = false;

    /** Allocate the tensor_form scratch (lazy, first use). */
    void EnsureTensorScratch () const;
    /** Capture the step-start plasma current Jp^n = curl B^n / mu0 - J_ext
     *  into m_tensor_jp_old (called once per step by the theta-implicit
     *  solver at step entry, when Bfield_fp still holds B^n). */
    void CaptureTensorStepStart ();
    /** \brief tensor_form E-solve, 1D_Z (see m_esolve_tensor). Transverse
     *  (E_x, E_y) rows carry the staggered curl-curl (-d2/dz2 on the
     *  shared z-nodal stagger, the discrete curl composed with itself)
     *  plus the pointwise tensor; the E_z row is pointwise (curl-curl
     *  annihilates the 1D longitudinal component) with the tensor
     *  couplings interpolated across staggers. Point-block-Jacobi
     *  preconditioned BiCGStab (the gyration part of T is antisymmetric,
     *  so the operator is non-symmetric). Periodic z only (v1). */
    void SolveETensor1D (ablastr::fields::VectorField const& Efield,
                         ablastr::fields::VectorField const& Jifield,
                         ablastr::fields::VectorField const& Bfield,
                         amrex::MultiFab const& rho,
                         amrex::MultiFab const& Pe,
                         int lev, bool solve_for_Faraday,
                         bool include_resistivity) const;

    /** \brief tensor_form E-solve, RZ m = 0 (see m_esolve_tensor): the
     *  stateless Je elimination on the RZ curl-curl base, solved as one
     *  coupled 3-component system on the native Yee staggerings.
     *
     * Rows are scaled by the poloidal identity form's control volumes
     * (V_r = r_c dr, V_z = r_n dr with the exact axis half cell, V_t =
     * r_n dr), making the curl-curl block the energy-form gradient:
     *
     *   V_c [ (1/mu0) (curlcurl E)_c + kE (M^{-1} E)_c ] = V_c (M^{-1} w)_c
     *
     * with w = kE eta Jp^n - qm B x Je_meas - qm grad Pe (all division-
     * free). The poloidal (E_r, E_z) curl-curl rows are the composed
     * identity-form rows of SolveEPolCurlCurlRZ (W = d_z E_r - d_r E_z
     * through cell centers), completed in vacuum by the same energy-form
     * -grad(gamma div .) closure; the toroidal row composes the native
     * curls w_r = -d_z E_t (B_r stagger) and w_z = (1/r_c) d_r (r E_t)
     * (B_z stagger), whose energy gradient carries the axis regularity
     * naturally (r_0 = 0 removes the axis flux -- no hand-built branch).
     * Dirichlet identity rows: E_theta axis (regularity) and wall (PEC),
     * E_z wall (PEC). Periodic z only (v1); non-darwin only (the
     * transverse-state split makes curlcurl E_L != 0 discretely in RZ --
     * the sector caveat).
     *
     * EVERY rho-derived coefficient and binary gate (kE, effective mass,
     * eta(rho), gamma, RHS membership) reads the per-step-frozen
     * m_tensor_rho_old snapshot -- nodal rho consumed by proper staggered
     * Interp (nodal source), unlike the identity form's cell-centered-
     * convention averages. B, Ji, and E follow the iterate (smooth).
     * Membership (gamma and the RHS test) thresholds at the ONE-COUNT
     * anchor rho_1c = q_e (tensor_n_min | n_floor) -- a statistics guard,
     * never a density modification: strictly-zero gating leaves epsilon
     * deposit-tail rows as near-null gradient directions driven by
     * non-rho-scaled numerator bleed, i.e. the unfloored division
     * reappears as a solver conditioning floor. Sub-one-count rows keep
     * their (tiny) screening in the operator and take the div E = 0
     * continuation as their content.
     * Point-Jacobi preconditioned BiCGStab on the composite (the gyration
     * part of the tensor is antisymmetric). hybrid_pic_model.tensor_verify
     * runs the one-shot fused-vs-composed row verification. */
    void SolveETensorRZ (ablastr::fields::VectorField const& Efield,
                         ablastr::fields::VectorField const& Jifield,
                         ablastr::fields::VectorField const& Bfield,
                         amrex::MultiFab const& rho,
                         amrex::MultiFab const& Pe,
                         int lev, bool solve_for_Faraday,
                         bool include_resistivity) const;

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
    /** Freeze the recovery/Faraday-overwrite mask density per step. The
     *  masks were written against "component 0 of rho_fp = the step-entry
     *  rho^n", but under the implicit scheme that component is a PRE-push
     *  deposit: only the first evaluation of a step sees rho(x^n) there,
     *  and afterwards it holds the previous evaluation's midpoint
     *  positions (see hybrid_rho_n_frozen, which exists for the same
     *  reason). A live read therefore lets mask MEMBERSHIP follow the
     *  iterate: near-threshold cells swap between Ohm rows and
     *  band-overwrite rows across residual evaluations -- a non-smooth
     *  hidden state in the residual map that floors Newton at the flicker
     *  amplitude independently of tolerance. It also splits within one
     *  evaluation (the A recovery runs before the deposit, the Faraday
     *  overwrite after it). With this on (default), all masks read a
     *  per-step copy of the committed entry density, captured in the
     *  implicit step entry alongside the E_L^n entry solve (the same
     *  density snapshot that solve consumes); the partition is constant
     *  through every base/probe/line-search evaluation of the step and
     *  refreshes between steps. 0 restores the legacy live-density
     *  masks. */
    bool m_darwin_vacuum_recovery_frozen_mask = true;
    amrex::Real m_darwin_vacrec_rtol = 1.0e-8;
    amrex::Real m_darwin_vacrec_atol = 0.0;
    int m_darwin_vacrec_max_iters = 200;
    /** Iteration operator of the recovery correction solve.
     *  "poisson" (default): the nodal FD vector-Poisson map (truncated
     *  identity -lap dA = mu0 J_imp; the defect J_imp is still measured
     *  by composed curls, so the recovery FIXED POINT -- masked
     *  curl(curl A) = 0 -- is exact either way; a nonzero div dA along
     *  the iteration never reaches B, since B = curl A).
     *  "curlcurl": the native amrex::MLCurlCurl operator on the edge (A)
     *  staggering -- (curlcurl + beta) dA = -mu0 J_imp -- which cancels
     *  the measured defect in a single solve instead of asymptotically
     *  and needs no nodal-edge interpolation of dA. beta is a small
     *  Tikhonov regularization of the curl-curl null space (the discrete
     *  gradient family; the masked rhs is not orthogonal to it): it
     *  shapes the iteration only, not the fixed point, and its gradient-
     *  family content in dA is invisible to B. Cartesian (3D / 2D-XZ)
     *  only: amrex::MLCurlCurl carries no RZ metric (the in-tree fused RZ
     *  curl-curl of SolveEPolCurlCurlRZ is the future native-RZ path);
     *  RZ and embedded boundaries stay on "poisson". */
    std::string m_darwin_vacuum_recovery_operator = "poisson";
    /** Tikhonov beta for the "curlcurl" recovery operator, relative to
     *  the grid-scale curl-curl diagonal sum_d 4/dx_d^2. */
    amrex::Real m_darwin_vacrec_cc_beta_rel = 1.0e-8;
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

    /** "curlcurl" backend of the recovery correction solve (see
     *  m_darwin_vacuum_recovery_operator): edge-staggered
     *  (curlcurl + beta) dA = -mu0 J_imp through amrex::MLCurlCurl, then
     *  A += omega dA in the mask. Cartesian-only (parse-guarded), no EB.
     */
    void VacuumRecoveryCurlCurlSolve (bool a_frozen_probe,
                                      amrex::Real a_omega,
                                      amrex::MultiFab const& a_rho,
                                      amrex::Real a_rho_floor,
                                      int a_mask_mode,
                                      bool a_flux_only);

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

    /** Momentum-consistent particle push field for the theta-implicit
     *  hybrid solver: subtract the resistive part of the Ohm field from
     *  the field gathered by the ions (the explicit scheme pushes ions
     *  with the no-resistivity Ohm field; ion-side resistive friction is
     *  the separate drag operator). Default ON: with the full Ohm field
     *  in the push, eta*J does net positive work on the deposited ion
     *  current and the resistive term becomes an energy source of the
     *  advance rather than a sink (1D periodic energy-ledger campaign,
     *  2026-08-25). Default ON: the correction is assembled per residual
     *  evaluation with both Ohm passes through the identical boundary
     *  stack (see ThetaImplicitHybrid::ComputeRHS); the old opt-in
     *  default's Newton-stability caution traced to a boundary-asymmetric,
     *  lagged assembly, fixed alongside this flip. Set false to reproduce
     *  pre-fix trajectories. */
    bool m_implicit_push_excludes_resistive_field = true;

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
    /** Exact-operator hyper-resistivity: apply E_H = +curl(eta_H curl J)
     *  with the curls composed discretely (edge -> face -> edge, the
     *  Ampere-closure stencils) and the coefficient INSIDE the outer curl,
     *  face-averaged -- the energy-sign-definite form for spatially
     *  varying eta_H (E_H.J integrates to eta_H |curl J|^2 >= 0) --
     *  instead of the truncated -eta_H lap J, which equals it only at
     *  zero discrete div J. Deposited and Ampere-closure currents carry
     *  machine-level-plus divergence, and the truncated form's
     *  grad(div J) remainder is non-dissipative. Default off to keep
     *  existing trajectories bit-identical; intended to become the
     *  default with the next validated chain. */
    bool m_hyper_resistivity_curlcurl = false;

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
