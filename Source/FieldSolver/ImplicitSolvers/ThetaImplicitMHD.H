/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef THETA_IMPLICIT_MHD_H_
#define THETA_IMPLICIT_MHD_H_

#include "FieldSolver/ImplicitSolvers/ImplicitMHDWallMask.H"
#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"

#include <AMReX_Array.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <memory>
#include <string>

class CircuitCoupler;
class HybridPICModel;
namespace theta_implicit_mhd {
struct FluxParameters;
}

/**
 * \brief Theta-implicit single-ion-fluid MHD solver.
 *
 * The JFNK state is
 *
 *     (E^{n+theta}, rho_i^{n+theta}, (rho_i u_i)^{n+theta},
 *      U_e^{n+theta}),
 *
 * where the fluid variables are cell centered. Magnetic field evolution and
 * generalized Ohm's law reuse the hybrid-PIC field operators. Unlike
 * ThetaImplicitHybrid, the nonlinear residual advances ion continuity and
 * momentum directly and does not push or redeposit ion particles.
 *
 * With implicit_mhd.ion_closure = total_energy the state gains the
 * conservative ion total energy E_i = U_i + rho |u|^2 / 2 as an additional
 * block, with p_i = (gamma_i - 1) max(E_i - KE, U_i_floor); the default
 * barotropic closure evaluates p_i from the polytropic law and leaves the
 * (always allocated) ion-energy MultiFab zeroed and unused.
 *
 * With implicit_mhd.fluid_flux = hlld the solver switches to the
 * conservative-form recast: the JFNK state carries B^{n+theta} (the Yee
 * face-staggered Bfield_fp) instead of E, one HLLD Riemann solution per
 * cell face supplies the fluid fluxes (with the Maxwell stress replacing
 * the pointwise J x B force), the ideal EMF (upwinded constrained
 * transport, so div B = 0 is preserved exactly by the Yee curl), and the
 * ion-energy magnetic work; E becomes a derived Ohm's-law quantity
 * (Riemann EMF + eta J) assembled by the solver itself rather than by
 * HybridPICSolveE. The B-block residual is the theta-implicit Faraday
 * update, B - B^n + theta dt curl E(B, fluid) = 0.
 */
class ThetaImplicitMHD final : public ImplicitSolver
{
  public:
    ThetaImplicitMHD ();
    ~ThetaImplicitMHD () override = default;

    ThetaImplicitMHD (const ThetaImplicitMHD&) = delete;
    ThetaImplicitMHD& operator=(const ThetaImplicitMHD&) = delete;
    ThetaImplicitMHD (ThetaImplicitMHD&&) = delete;
    ThetaImplicitMHD& operator=(ThetaImplicitMHD&&) = delete;

    void AllocateLevelMFs (ablastr::fields::MultiFabRegister& fields, int lev,
                           const amrex::BoxArray& ba,
                           const amrex::DistributionMapping& dm) const override;

    void Define (WarpX* a_WarpX, bool from_restart) override;
    void PrintParameters () const override;

    int OneStep (amrex::Real start_time, amrex::Real dt, int step) override;

    void ComputeRHS (WarpXSolverVec& rhs, const WarpXSolverVec& state, amrex::Real start_time,
                     int nonlinear_iteration, bool from_jacobian) override;

    // These implementation methods contain extended __device__ lambdas and must
    // therefore be public when compiled with NVCC.
    [[nodiscard]] amrex::Real
    LimitSolverStep (const WarpXSolverVec& state,
                     const WarpXSolverVec& direction,
                     amrex::Real requested_step) const override;

    [[nodiscard]] amrex::Real
    ProjectAndLimitSolverStep (const WarpXSolverVec& state,
                               WarpXSolverVec& direction,
                               amrex::Real requested_step) const override;

    [[nodiscard]] bool
    UseNewtonLineSearch () const override
    {
        return true;
    }

    /** Record whether the next non-Jacobian ComputeRHS call evaluates
     * the residual at an accepted Newton iterate (see the base
     * declaration); read by the implicit_mhd.circuit_hook_scope =
     * newton gating of the external-coil circuit hook. */
    void SetResidualIsNewtonIterate (bool is_iterate) override
    {
        m_residual_is_newton_iterate = is_iterate;
    }

    /** Free-subspace residual norm over the complement of the clamp
     * masks recorded by the last ProjectAndLimitSolverStep call, with
     * the pinned-defect norm and clamped-component count (see the base
     * declaration). Public for the NVCC extended-lambda requirement. */
    [[nodiscard]] amrex::Real
    FreeResidualNorm (const WarpXSolverVec& residual,
                      amrex::Real& pinned_defect,
                      amrex::Long& num_pinned) const override;

    [[nodiscard]] std::string PinnedComponentReport () const override;

    /** List up to implicit_mhd.pinned_cell_report_max clamped cells from
     * the last projection's active-set masks -- block name, (i,j[,k])
     * index, and physical cell-center coordinates -- one line per cell
     * via amrex::AllPrint() (rank-local scan: each rank prints its own
     * cells; the counts in PinnedComponentReport remain global). Called
     * from FreeResidualNorm, i.e. only on the line-search failure path,
     * rate-limited there so a long frozen plateau does not flood the
     * log. Public for the NVCC extended-lambda requirement. */
    void PrintPinnedCells () const;

    /** Localize a non-converged Newton residual: per-block L2 norms
     * (fluid blocks and B faces, unscaled MultiFab norms) plus the
     * cells within 2x of each block's max |F|, with physical
     * coordinates (bounded harvest, same pattern as PrintPinnedCells).
     * Public for the NVCC extended-lambda requirement. */
    void ResidualHotspotReport (const WarpXSolverVec& residual) const override;

    /**
     * \brief The MHD residual is evaluable for ARBITRARY real states by
     * construction: densities are max-floored, pressures and energies go
     * through C-infinity smooth floors, the barotropic pow() falls back
     * to the floor ratio below the density floor, and every user-parser
     * input (resistivity) receives a floored density. Jacobian probes
     * therefore need no admissibility limiting -- and skipping it keeps
     * the Jacobian exact for states resting on their bounds, where a
     * limited probe underflows to the eps -> 0 abort whenever a Krylov
     * vector concentrates on a bound-resident cell (observed at the
     * 54.8 t_ci CGL reorganization event of the FRC free hold). The
     * Newton UPDATE path keeps its projection and limiting unchanged;
     * only probes go unlimited.
     */
    [[nodiscard]] bool
    ProbePerturbationsNeedLimiting () const override
    {
        return false;
    }

    void InitializeFluidState ();
    void SanitizeLoadedState ();
    /** Rigid-vacuum exterior clamp of the shaped-wall band (see the
     * band-hygiene note in ImplicitMHDWallMask): in every
     * wall_live-freezing mode (an active wall_thermal_bc), SET every
     * masked fluid cell to the fixed clamp image -- mass density at the
     * positivity floor, zero momentum, floor-consistent energies at
     * the background temperature (T_wall under the temperature-mode
     * thermal wall, so the interface drain starts at zero gradient) --
     * once at first-step sanitize time (after SanitizeLoadedState, so
     * Python-loaded exteriors are scraped too). Bit-exactly idempotent
     * by construction: restarted states that already carry the clamp
     * are unchanged. Reports clamped-cell count and net scraped mass
     * (rank-0 banner + a '#'-comment header row of wall_ledger_file).
     * No-op without the freeze. */
    void ClampWallExteriorState ();
    /** Recompute the dynamic halo pedestal density once per step (frozen
     * for the whole nonlinear solve) and raise sub-pedestal cells onto
     * it. No-op when halo_pedestal_fraction = 0. See
     * m_halo_pedestal_fraction for the offset-density rationale. */
    void RefreshHaloPedestal ();
    /** Recompute the dynamic vacuum-reference density once per step (the
     * The reference code's en0 update, step.f90 ~218: en0 = max(en00, 0.1 max(en))):
     * m_vacuum_reference_mass_density = vacuum_reference_peak_fraction
     * times the global step-old density peak, FROZEN for the whole
     * nonlinear solve so every residual/Jacobian evaluation keys the
     * density-keyed boosts to the same reference (Newton-consistent).
     * Rank 0 prints the effective reference on >1% change. No-op when
     * vacuum_reference_peak_fraction = 0. */
    void RefreshVacuumReferenceDensity (int step);
    /** Effective reference mass density of the density-keyed boosts
     * (the field-eta vacuum boost, the Ohm-current Joule quench
     * criterion, and the conduction halo boost): the static Ohm guard
     * by default, raised to fraction * rho_peak when the dynamic
     * Reference is active (see m_vacuum_reference_peak_fraction). */
    [[nodiscard]] amrex::Real VacuumReferenceMassDensity () const;
    /** Mass-density equivalent of the Ohm's-law division guard
     * (hybrid_pic_model.n_floor). Deliberately independent of the fluid
     * admissibility floor m_mass_density_floor: eta/T_e evaluations and
     * the Ohm -(u x B) velocity reconstruction floor their inputs here,
     * while the fluid floor stays a pure positivity guard. Computed from
     * the live hybrid n_floor (Python can update it at runtime). */
    [[nodiscard]] amrex::Real OhmMassDensityFloor () const;
    void FinishStateUpdate (amrex::Real end_time, int step);
    void FillFluidSources (const WarpXSolverVec& state);
    void ApplyFluidDomainBoundaries (amrex::MultiFab& density, amrex::MultiFab& momentum,
                                     amrex::MultiFab& electron_energy,
                                     amrex::MultiFab& ion_energy,
                                     amrex::MultiFab& ion_parallel_energy,
                                     amrex::MultiFab& ion_perp_energy,
                                     amrex::MultiFab& ion_internal_energy) const;
    /** Zero-gradient (Neumann) fill of the axial domain ghosts,
     * staggering aware: cell-centered-in-z data clamps to the boundary
     * cell (zero gradient across the boundary face) while NODAL-in-z
     * data mirrors evenly across the boundary node itself (a clamp
     * would leave a spurious half-gradient at the end node: the
     * centered derivative there reads (f_1 - f_0)/(2 dz), not 0). On an
     * OPEN z face (Green's-function boundary), the outermost
     * a_open_face_keep_rows ghost rows are left untouched -- they carry
     * free-space values written by the Green's fill (B) or computed from
     * it (curl-B currents, cell-centered interpolants) -- and any deeper
     * ghost rows clamp/mirror about the outermost kept row instead of
     * the interior plane. The default (0) anchors every ghost row to the
     * boundary plane on every face, which is the unconditional
     * fluid-moment behavior. */
    void ApplyNeumannZDomainGhosts (amrex::MultiFab& mf,
                                    int a_open_face_keep_rows = 0) const;
    /** Mirror-parity fill of the z_lo domain ghosts (the PMC/symmetry
     * face, m_z_lo_pmc): every ghost row is the exact linear reflection
     * of the interior with the per-component parity a_component_parity
     * (+1 even, -1 odd; entries beyond nComp are ignored), staggering
     * aware -- cell-centered-in-z data reflects across the boundary
     * FACE (ghost z_lo - 1 - g takes +/- the value at z_lo + g) and
     * nodal-in-z data reflects across the boundary NODE (ghost z_lo - g
     * takes +/- the value at z_lo + g, and an odd component is zeroed
     * ON the plane node). Runs after the Neumann pass of the same field
     * (which owns z_hi) and overwrites its z_lo image; even nodal-in-z
     * data is unchanged by construction (the Neumann nodal fill is
     * already the even mirror). */
    void ApplyMirrorZLoDomainGhosts (amrex::MultiFab& mf,
                                     const amrex::GpuArray<int, 3>& a_component_parity) const;
    void ApplyScalarRadialDomainGhosts (amrex::MultiFab& mf) const;
    /** Copy the three momentum-density components into their
     * single-component diagnostic views (MomentumDiag*Name). */
    void PublishMomentumComponents () const;
    void FillCellCenteredElectromagneticFields ();
    /** Gather the assembled stage E (Efield_fp; plus the external E under
     * the split-field external-A scheme) to cell centers into
     * OhmElectricFieldCCName for the Ohm-current Joule quench (see
     * m_joule_ohm_current). Called by ComputeRHS after the stage E
     * assembly on every residual evaluation, so matrix-free Jacobian
     * probes see the quenched deposit exactly. */
    void FillCellCenteredOhmElectricField ();
    void ComputeFluidRHS (WarpXSolverVec& rhs, amrex::Real time) const;
    void ComputeFluidRHSFromFaceFluxes (WarpXSolverVec& rhs, amrex::Real time) const;
    /** \param a_time the residual (theta-stage) time, forwarded to the
     * parser thermal diffusivities' t symbol. */
    void ComputeFaceFluxes (amrex::Real a_time);
    void ComputeDirectionalFaceFluxes (amrex::MultiFab& face_flux_mf,
                                       int normal_direction,
                                       amrex::Real a_time);
    /** Absorbing-wall bookkeeping (r_open_fluid = absorb, RZ hlld): the
     * wall is a physical sink by design, so the conservation ledger
     * tracks it explicitly. Recomputes the radial face fluxes of the
     * ACCEPTED theta state (the same fluxes the step update integrated,
     * so a converged solve books the export exactly), integrates the
     * r-weighted wall-face mass and fluid-energy fluxes over the step,
     * accumulates them into the cumulative absorbed-mass/energy
     * counters, and prints (and optionally appends to
     * implicit_mhd.absorb_ledger_file) every absorb_ledger_interval
     * steps. Must be called after UpdateWarpXFields(m_state, ...) and
     * before FinishStateUpdate extrapolates the state to t^{n+1}. */
    void AccumulateAbsorbedWallLedger (amrex::Real dt, int step,
                                       amrex::Real a_time);
    /** Floor-consistency supply bookkeeping (floor_consistency_rate > 0):
     * the relaxation source creates mass/energy at bound-riding cells by
     * design, so the conservation ledger tracks it explicitly. Evaluates
     * the source at the ACCEPTED theta state (m_state vs m_state_old --
     * for a converged solve U^{n+1} - U^n = dt * RHS(theta state), so the
     * booked amounts match the state change to the nonlinear solver
     * tolerance, exactly the wall-ledger contract), integrates
     * dt * rate_eff * deficit over the domain per bounded block with the
     * geometry's cell measure (2 pi r dr dz in RZ), accumulates the
     * cumulative supplied-mass/energy counters, and appends a
     * "step mass energy" row to implicit_mhd.floor_ledger_file when set.
     * Wall-masked (rigid-conductor) cells are excluded, mirroring the
     * residual kernels. Must be called after the nonlinear solve and
     * before FinishStateUpdate extrapolates the state to t^{n+1}. */
    void AccumulateFloorConsistencySupplyLedger (amrex::Real dt, int step);
    /** The reference code's density eater (ntb.f90:773-807, trigger eaten_type > 0 at
     * ntb.f90:202): a one-sided per-step relaxation of the mass density
     * toward a low target in a fixed axial band,
     *     rho_lim = (1 - rate) * max(rho, target) + rate * target,
     *     rho    := min(rho, rho_lim),
     * i.e. the excess above target decays by the factor (1 - rate) per
     * step; at-or-below-target cells are never touched (the MIN).
     * The reference code hard-codes rate = 0.2 and target = en0/100 with the
     * step-start dynamic reference en0 = max(en0_input,
     * 0.1 max(en)) (step.f90:218-225, en0_upd = 1 default). The eater
     * modifies ONLY the density state variable: velocities and the ion
     * energy DENSITIES (the reference shot's closure mix = -1: wio and wik both
     * state variables, both untouched) are preserved, so momentum
     * density scales with the mass, E_i/U_i (and the CGL pair) are
     * invariant (the per-particle ion temperature RISES), and the
     * electron temperature is preserved (te is the reference code's electron
     * state under flg_wie = false, so U_e scales with the mass).
     * Applied to the committed t^{n+1} state at the top of
     * FinishStateUpdate (before the floor restorations, matching the
     * The reference code's order eater -> mixmaster within step_en/ntbstep), outside
     * every implicit solve like the reference code's call site. The removed mass
     * and (electron) energy are booked cumulatively and optionally
     * appended to density_eater_ledger_file as "step mass energy" rows
     * (the floor-ledger conventions). Wall-masked (rigid-conductor)
     * cells are excluded, the x_kind < 1 analog. */
    void ApplyDensityEater (int step);
    /** Assemble the solver-owned Ohm's law E = -u x B + eta J (hlld path).
     * With at_resistive_stage = true (residual evaluations) the
     * dissipative terms (eta J, the vacuum-eta boost, and the
     * hyper-resistive term) are evaluated at the resistive-stage current
     * J^{n+theta_r} (see m_resistive_theta); with false (the end-of-step
     * recompute) they use the instantaneous current, so the published E
     * satisfies Ohm's law at t^{n+1}. */
    void AssembleOhmElectricField (amrex::Real time, bool at_resistive_stage) const;
    /** When the resistive-stage weighting is active (at_resistive_stage
     * and resistive_theta != theta), fill the resistive-stage current
     * scratch J^{n+theta_r} = (theta_r/theta) J^{n+theta}
     * + (1 - theta_r/theta) J^n and return true; return false otherwise
     * (the caller then reads the plasma current directly). */
    [[nodiscard]] bool PrepareResistiveStageCurrents (bool at_resistive_stage) const;
    void ApplyMagneticCCDomainGhosts (amrex::MultiFab& mf) const;

    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0; }

    [[nodiscard]] amrex::Real GetMHDThetaForPC () const override { return m_theta; }

    [[nodiscard]] amrex::Real
    GetMHDReferenceResistivityForPC (amrex::Real time) const override;

    [[nodiscard]] amrex::Real
    GetMHDResistiveThetaForPC () const override
    {
        return m_resistive_theta;
    }

    [[nodiscard]] const amrex::MultiFab*
    GetMHDFieldResistivityCCForPC (amrex::Real time) const override;

    [[nodiscard]] amrex::Array<const amrex::MultiFab*, 3>
    GetMHDFieldResistivityEdgeForPC (amrex::Real time) const override;

    [[nodiscard]] bool GetMHDIncludeHyperResistivityForPC () const override;

    [[nodiscard]] amrex::Array<const amrex::MultiFab*, 3>
    GetMHDHyperResistivityEdgeForPC (amrex::Real time) const override;

    [[nodiscard]] amrex::GpuArray<amrex::Real, 3>
    GetMHDReferenceMagneticFieldForPC () const override;

    [[nodiscard]] bool GetMHDIncludeHallTermForPC () const override;

    [[nodiscard]] amrex::Array<const amrex::MultiFab*, 3>
    GetMHDHallCoefficientEdgeForPC () const override;

    [[nodiscard]] bool GetMHDIncludeElectronInertiaForPC () const override;

    [[nodiscard]] amrex::Real
    GetMHDElectronInertiaStencilScaleForPC () const override;

    [[nodiscard]] amrex::Array<const amrex::MultiFab*, 3>
    GetMHDElectronInertiaCoefficientEdgeForPC () const override;

    [[nodiscard]] bool
    GetMHDEvolveIonFluidForPC () const override
    {
        return m_evolve_ion_fluid;
    }

    [[nodiscard]] amrex::Real
    GetMHDReferenceChargeDensityForPC () const override
    {
        return m_ion_charge_to_mass * m_reference_mass_density;
    }

    [[nodiscard]] const amrex::MultiFab*
    GetMHDMagneticFieldCCForPC () const override;

    [[nodiscard]] warpx::mhd_pc::WallMaskView
    GetMHDWallMaskForPC () const override
    {
        return m_wall_mask.View();
    }

    [[nodiscard]] warpx::mhd_pc::WallFluidFreezeView
    GetMHDWallFluidFreezeForPC () const override
    {
        return m_wall_mask.FluidFreezeView();
    }

    [[nodiscard]] warpx::mhd_pc::WallFieldFreezeView
    GetMHDWallFieldFreezeForPC () const override
    {
        return m_wall_mask.FieldFreezeView();
    }

    [[nodiscard]] bool GetMHDZLoMirrorForPC () const override
    {
        return m_z_lo_pmc;
    }

    inline static constexpr char MassDensityName[] = "implicit_mhd_mass_density";
    inline static constexpr char MomentumDensityName[] = "implicit_mhd_momentum_density";
    inline static constexpr char ElectronEnergyName[] = "implicit_mhd_electron_energy";
    inline static constexpr char IonEnergyName[] = "implicit_mhd_ion_energy";
    // CGL (bi-Maxwellian) ion closure: parallel internal energy
    // U_par = p_par/2 (one degree of freedom along b) and perpendicular
    // internal energy U_perp = p_perp (two degrees of freedom).
    inline static constexpr char IonParallelEnergyName[] =
        "implicit_mhd_ion_parallel_energy";
    inline static constexpr char IonPerpEnergyName[] =
        "implicit_mhd_ion_perp_energy";
    // Dual-energy (the reference code's mixmaster) ion closure: the auxiliary internal
    // ion energy U_i evolved alongside the conservative total E_i; the
    // ion pressure every consumer sees is the kinetic-fraction blend
    // p_i = fk p_i(E_i) + (1 - fk)(gamma_i - 1) U_i (see
    // FluxParameters::dual_energy_closure in ThetaImplicitMHD_K.H).
    inline static constexpr char IonInternalEnergyName[] =
        "implicit_mhd_ion_internal_energy";
    // Single-component cell-centered views of the momentum density for
    // field diagnostics (the 3-component register block cannot pass
    // through the generic fields_to_plot path; exact T_i reconstruction
    // from dumps needs the kinetic energy). Refreshed at initialization
    // and at every accepted step end; never read by the solver.
#if defined(WARPX_DIM_RZ)
    inline static constexpr char MomentumDiag0Name[] = "implicit_mhd_momentum_r";
    inline static constexpr char MomentumDiag1Name[] = "implicit_mhd_momentum_t";
#else
    inline static constexpr char MomentumDiag0Name[] = "implicit_mhd_momentum_x";
    inline static constexpr char MomentumDiag1Name[] = "implicit_mhd_momentum_y";
#endif
    inline static constexpr char MomentumDiag2Name[] = "implicit_mhd_momentum_z";

  private:
    inline static constexpr char TotalCurrentCCName[] = "implicit_mhd_total_current_cc";
    /** Cell-centered electron temperature scratch of the temperature-primary
     * nodal closure fill (see FillFluidSources): T_e = p_e/(n_f kB) is formed
     * NATIVELY on the cell grid (the same floored pressure recovery as the
     * flux kernels over the Ohm-floored density) and then interpolated, so
     * the node interpolant acts on the smooth bounded ratio rather than the
     * n T product. */
    inline static constexpr char ElectronTemperatureCCName[] =
        "implicit_mhd_electron_temperature_cc";
    inline static constexpr char MagneticFieldCCName[] = "implicit_mhd_magnetic_field_cc";
    /** Cell-centered gather of the SOLVED stage electric field for the
     * Ohm-current Joule quench (see m_joule_ohm_current): refreshed by
     * FillCellCenteredOhmElectricField on every residual evaluation from
     * the just-assembled Efield_fp (plus the external E under the
     * split-field external-A scheme, whose subtraction is Faraday
     * bookkeeping only -- Ohm's law holds for the total field). */
    inline static constexpr char OhmElectricFieldCCName[] =
        "implicit_mhd_ohm_electric_field_cc";
    // Field resistivity eta_field (user eta through the density-keyed
    // vacuum boost) scratch, filled on demand for the MHD block
    // preconditioner's resistive block: cell-centered (the activation
    // gate) and on the electric-field staggerings (the discrete resistive
    // curl-curl the block iterates on; per physical E component, with the
    // residual's own density interpolation).
    inline static constexpr char FieldResistivityCCName[] =
        "implicit_mhd_field_resistivity_cc";
    inline static constexpr char FieldResistivityE0Name[] =
        "implicit_mhd_field_resistivity_e0";
    inline static constexpr char FieldResistivityE1Name[] =
        "implicit_mhd_field_resistivity_e1";
    inline static constexpr char FieldResistivityE2Name[] =
        "implicit_mhd_field_resistivity_e2";
    // eta_H on the same electric-field staggerings (the preconditioner's
    // frozen hyper-resistivity; zeroed and unused when the Ohm law runs
    // without the hyper-resistive term).
    inline static constexpr char HyperResistivityE0Name[] =
        "implicit_mhd_hyper_resistivity_e0";
    inline static constexpr char HyperResistivityE1Name[] =
        "implicit_mhd_hyper_resistivity_e1";
    inline static constexpr char HyperResistivityE2Name[] =
        "implicit_mhd_hyper_resistivity_e2";
    // Frozen Hall coefficient vector Hb = B/rho_q (three physical
    // components) on the same electric-field staggerings: the
    // preconditioner's whistler-rotation coefficients (zeroed and unused
    // when the Ohm law runs without the Hall term). In 1D the transverse
    // components share the z-nodal register.
    inline static constexpr char HallCoefficientE0Name[] =
        "implicit_mhd_hall_coefficient_e0";
    inline static constexpr char HallCoefficientE1Name[] =
        "implicit_mhd_hall_coefficient_e1";
    inline static constexpr char HallCoefficientE2Name[] =
        "implicit_mhd_hall_coefficient_e2";
    // Frozen electron-inertia coefficient C_i = m_e_eff/(e rho_q) on the
    // same electric-field staggerings: the preconditioner's inertia-mass
    // coefficients (zeroed and unused when the Ohm law runs without the
    // electron-inertia term). In 1D the transverse components share the
    // z-nodal register.
    inline static constexpr char InertiaCoefficientE0Name[] =
        "implicit_mhd_inertia_coefficient_e0";
    inline static constexpr char InertiaCoefficientE1Name[] =
        "implicit_mhd_inertia_coefficient_e1";
    inline static constexpr char InertiaCoefficientE2Name[] =
        "implicit_mhd_inertia_coefficient_e2";
    // Per-face-direction Riemann flux registers for the conservative-form
    // hlld path: one HLLD solution per face feeds the fluid divergences,
    // the Maxwell stress, the ion-energy magnetic work, and the ideal EMF.
    // Channel layout is FaceFluxComponent below. 1D has z-faces only; RZ
    // has r-faces (normal = physical direction 0) and z-faces (normal =
    // physical direction 2).
    inline static constexpr char FaceFluxRName[] = "implicit_mhd_face_flux_r";
    inline static constexpr char FaceFluxZName[] = "implicit_mhd_face_flux_z";

    // Component indices of the face-flux MultiFabs (hlld path).
    struct FaceFluxComponent {
        enum : int {
            mass = 0,
            momentum_x,   // total momentum flux, including Maxwell stress
            momentum_y,
            momentum_z,
            magnetic_x,   // Maxwell-stress part of the momentum flux
            magnetic_y,
            magnetic_z,
            electron_energy,
            ion_energy,
            ion_parallel_energy, // CGL closure: U_par advective flux
            ion_perp_energy,     // CGL closure: U_perp advective flux
            electron_velocity,
            induction_t1, // flux of B_t1 through the face = -EMF_t2
            induction_t2, // flux of B_t2 through the face = +EMF_t1
            signal_left,  // smoothed Davis bounds: corner-EMF upwind weights
            signal_right,
            signal_alfven_left,  // fan rotational bounds S_M -+ c_An*:
            signal_alfven_right, // corner-EMF dissipation coefficients
            ion_internal_energy, // dual_energy closure: U_i advective flux
            count
        };
    };
    // Ghosted beginning-of-step fluid state (the solver-vec old-state
    // blocks are zero-ghost): filled once per step in OneStep, read by
    // the flux kernels' floor limiters, which gate on theta-extrapolated
    // end-of-step DONOR values and therefore need old neighbors.
    inline static constexpr char OldMassDensityName[] = "implicit_mhd_old_mass_density";
    inline static constexpr char OldMomentumDensityName[] = "implicit_mhd_old_momentum_density";
    inline static constexpr char OldElectronEnergyName[] = "implicit_mhd_old_electron_energy";
    inline static constexpr char OldIonEnergyName[] = "implicit_mhd_old_ion_energy";
    inline static constexpr char OldIonParallelEnergyName[] =
        "implicit_mhd_old_ion_parallel_energy";
    inline static constexpr char OldIonPerpEnergyName[] =
        "implicit_mhd_old_ion_perp_energy";
    inline static constexpr char OldIonInternalEnergyName[] =
        "implicit_mhd_old_ion_internal_energy";
    // Resistive backward-Euler option (m_resistive_theta): the
    // beginning-of-step plasma current J^n (captured once per step) and
    // the per-residual resistive-stage scratch
    // J^{n+theta_r} = (theta_r/theta) J^{n+theta}
    //                 + (1 - theta_r/theta) J^n,
    // both allocated (per direction, at the Yee current staggering) only
    // when resistive_theta differs from the global theta.
    inline static constexpr char OldPlasmaCurrentName[] =
        "implicit_mhd_old_plasma_current";
    inline static constexpr char ResistiveStageCurrentName[] =
        "implicit_mhd_resistive_stage_current";

    HybridPICModel* m_hybrid_pic_model = nullptr;

    WarpXSolverVec m_state;
    WarpXSolverVec m_state_old;

    // Newton active set of the most recent ProjectAndLimitSolverStep
    // call, one cell-centered mask per fluid block in the block order
    // (rho, U_e, E_i or U_par, U_perp or U_i): 1 = this solve's Newton
    // direction component was clamped onto its admissibility bound,
    // 0 = free. Allocated once on first use (the box layout is fixed
    // after Define), overwritten by every projection call, and consumed
    // by FreeResidualNorm / PinnedComponentReport on the line-search
    // failure path. The counts are the GLOBAL (all-rank) sums. Mutable:
    // the projection and the norms are const observers of the solve.
    mutable std::array<amrex::iMultiFab, 4> m_projection_masks;
    mutable std::array<amrex::Long, 4> m_projected_per_block = {0, 0, 0, 0};
    mutable amrex::Long m_projected_components = 0;
    // Pinned-cell coordinate dump on the line-search failure path:
    // largest number of cells listed per report (0 disables), and the
    // failure-path call counter driving the rate limit.
    int m_pinned_cell_report_max = 16;
    mutable amrex::Long m_free_residual_norm_calls = 0;

    amrex::Real m_ion_charge_to_mass;
    amrex::Real m_gamma_e = 5.0 / 3.0;
    amrex::Real m_gamma_i = 5.0 / 3.0;
    amrex::Real m_reference_mass_density = 0.0;
    amrex::Real m_reference_velocity = 0.0;
    amrex::Real m_reference_magnetic_field = 0.0;
    amrex::Real m_reference_ion_pressure = 0.0;
    amrex::Real m_mass_density_floor = 0.0;
    amrex::Real m_electron_pressure_floor = 0.0;
    // One-time admissibility raise of the beginning-of-run fluid state
    // (Python fluid loaders bypass InitializeFluidState's floor clamp);
    // performed at the first OneStep, see SanitizeLoadedState.
    bool m_loaded_state_sanitized = false;
    // Temperature floors [K] on the evolved energy blocks (default: room
    // temperature; 0 disables). Each energy block's admissibility lower
    // bound is the MAX of its absolute pressure/energy floor and the
    // density-dependent n kB T_floor equivalent, with n evaluated from
    // the STEP-OLD mass density (m_state_old), frozen for the whole
    // nonlinear solve so the per-solve admissible set stays well-defined
    // and the bounds stay linear in the unknowns. The end-of-step floor
    // restoration in FinishStateUpdate instead uses the END-of-step
    // density -- which becomes the next solve's frozen step-old density,
    // so every solve starts admissible. The temperature bound is a
    // one-way RATCHET, engaged per cell only where the step-old value
    // already satisfied it: states at or above the floor can never cool
    // through it, while colder-than-floor states (e.g. cold initial
    // data) are never lifted and keep the absolute floors alone. The
    // kernels' smooth positivity anchors (pressure recovery, CGL rates)
    // keep their absolute pressure values.
    amrex::Real m_ion_temperature_floor = 0.0;
    amrex::Real m_electron_temperature_floor = 0.0;
    // Ion thermodynamic closure: "barotropic" (default; p_i from the
    // polytropic law), "total_energy" (E_i evolved as a JFNK block),
    // "dual_energy" (the reference code's mixmaster mix = -1 split-energy form:
    // E_i evolved exactly as under total_energy PLUS the auxiliary
    // internal energy U_i as an extra JFNK block; the ion pressure every
    // consumer sees is the kinetic-fraction blend
    // p_i = fk p_i(E_i) + (1 - fk)(gamma_i - 1) U_i with
    // fk = 1 - KE/(gamma_i E_i), killing the total-minus-kinetic
    // catastrophic-cancellation heating of KE-dominated cells; recast
    // fluxes hlld/central only), or "cgl" (bi-Maxwellian:
    // U_par = p_par/2 and U_perp = p_perp evolved as JFNK blocks with
    // CGL work terms and ion-ion isotropization; hlld only). Under cgl
    // the fan sees the effective isotropic pressure
    // p_eff = (p_par + 2 p_perp)/3 and the pressure recovery is LINEAR
    // in the unknowns (no kinetic-energy subtraction).
    std::string m_ion_closure = "barotropic";
    amrex::Real m_ion_pressure_floor = 0.0;
    // Dual-energy low-internal cutoff fraction (the reference code's mix = -2
    // variant; 0 = off): cells whose STEP-OLD U_i sits below
    // fraction * max_domain(U_i_old) get fk = 0 exactly (pure internal
    // pressure in the halo/dust). The max is a once-per-step step-old
    // global reduction, frozen for the whole nonlinear solve.
    amrex::Real m_dual_energy_internal_cutoff = 0.0;
    // Dual-energy end-of-step re-sync (the reference code's mixmaster temperature
    // update, ntb.f90): at every accepted step end E_i's internal
    // content is rewritten from the blended pressure,
    // E_i := KE + p_blend/(gamma_i - 1) -- the identity in thermal
    // cells, while in KE-dominated cells it drains the step's
    // cancellation drift into the U_i anchor (without it the drift
    // accumulates in the conservative E_i and the blend re-imports
    // fk >= 1 - 1/gamma_i of the accumulated total) -- and cells with
    // fk > this threshold (thermal, well-conditioned recovery)
    // additionally overwrite U_i := E_i - KE (the standard Enzo-style
    // sync), so the two energy variables cannot drift apart in thermal
    // regions.
    amrex::Real m_dual_energy_sync_threshold = 0.99;
    // Per-step frozen max_domain(U_i_old) of the cutoff mask.
    amrex::Real m_dual_energy_internal_old_max = 0.0;
    // CGL isotropization: Braginskii ion-ion collision rate times this
    // scale (0 disables relaxation), with the given Coulomb logarithm.
    amrex::Real m_cgl_relaxation_scale = 1.0;
    amrex::Real m_cgl_coulomb_log = 10.0;
    // Instability-bounded relaxation (Stage B): where the state crosses
    // the firehose (p_par - p_perp > B^2/mu0) or mirror
    // (beta_perp (p_perp/p_par - 1) > 1) thresholds, kinetic
    // instabilities isotropize on cyclotron timescales; the relaxation
    // rate gains cgl_instability_scale * Omega_ci through a C-infinity
    // switch of relative width cgl_instability_width in the threshold
    // measures. 0 disables the bounds (pure collisional Stage A).
    amrex::Real m_cgl_instability_scale = 0.1;
    amrex::Real m_cgl_instability_width = 0.1;
    // Magnetization-weighted null blend (Stage B+): the gyrotropic CGL
    // closure loses meaning where the ion gyroradius exceeds the grid
    // scale (field nulls: Omega_ci -> 0 while the instability bounds,
    // being Omega_ci-scaled, vanish with it). With the C-infinity
    // weight w = Omega_ci^2 dx^2 / (Omega_ci^2 dx^2 + v_thi^2)
    // (w = 1/(1 + (r_L/dx)^2), dx the smallest cell size), the
    // deviation stress carries w (p_par - p_perp), the work terms blend
    // w * CGL + (1 - w) * isotropic split -- total internal work stays
    // exactly -P_blend : grad u with P_blend = p_eff I
    // + w (p_par - p_perp)(bhat bhat - I/3) -- and the relaxation rate
    // gains cgl_null_scale * (v_thi/dx) * (1 - w), isotropizing at the
    // cell-transit rate of unmagnetized ions. 0 disables the blend
    // rate (the stress/work weighting still applies whenever the
    // closure is cgl).
    amrex::Real m_cgl_null_scale = 1.0;
    // ---- Electron-ion temperature equilibration (0 = OFF, no
    // arithmetic at all; 1 = the reference code's rate; other values scale it).
    // The reference code's eq_brate collisional exchange (ntb.f90 658-700, called
    // from the step_tm P/C loop at 322 whenever two_t = 1 -- the reference shot's
    // configuration -- with NO deck knob of its own): a time-implicit,
    // energy-conserving Spitzer pair relaxation between the electron
    // and ion temperatures,
    //     dTi/dt = -(gamma_i - 1) z nu (Ti - Te),
    //     dTe/dt = +(gamma_e - 1) nu (Ti - Te),
    //     nu = 4.75e-15 lnLambda (m_p/m_i) n_i z^2 / Te[eV]^{3/2},
    // which in this solver's energy variables (z_eff = 1) is the single
    // linear pair term
    //     dU_e/dt = +nu (p_i - p_e) = -dU_i/dt,
    // conserving U_e + U_i(internal) by construction. Placement: in the
    // Newton residual with the rate FROZEN from the step-old density
    // and electron pressure (the conduction_coefficient_state =
    // step_old precedent -- Newton-linear, JFNK-exact, and equivalent
    // to the reference code's coefficient-at-the-iterate backward Euler to first
    // order), the live pressures floored like every recovered pressure,
    // and the rate capped at the theta-scheme monotonicity bound
    // 1/((gamma_e + gamma_i - 2)(1 - theta) dt) so the pair difference
    // decays monotonically like the reference code's backward Euler. Deposits: +Q
    // to U_e; -Q to the ion channel of every closure (E_i under
    // total_energy; BOTH E_i and the auxiliary U_i under dual_energy,
    // keeping the pair consistent; the isotropic-temperature split
    // -Q/3 to U_par and -2Q/3 to U_perp under cgl, the same (1/3, 2/3)
    // convention as the w = 0 isotropic work split). Same envelope as
    // the other reactive/relaxation sources (plasma_weight, wall_live,
    // halo_source_taper), applied SYMMETRICALLY to both rows so the
    // pair sum is exact under every gate. Requires an evolved ion
    // energy block. Rationale: without it Joule-heated electrons never
    // warm the bulk ions -- the reference code's 152 eV live-median total
    // temperature vs our 44/12 eV Te/Ti split at matched formation
    // time, with the e-i relaxation time microsecond-class at
    // n ~ 1e20-3e20, Te ~ 40 eV.
    amrex::Real m_electron_ion_equilibration = 0.0;
    // Vacuum cell-switch threshold for the ion fluid (0 disables): cells
    // below it are passive dust with frozen momentum (Holmstrom-style).
    amrex::Real m_vacuum_mass_density = 0.0;
    // Velocity relaxation rate [1/s] applied to vacuum dust (0 disables).
    amrex::Real m_vacuum_drag_rate = 0.0;
    // ---- Offset-density (pedestal) halo formulation (0 disables). ----
    //
    // Fraction f of the instantaneous density peak defining the dynamic
    // halo pedestal rho_ped = f * max(rho_peak, rho_ref), recomputed once
    // per step and FROZEN for the whole nonlinear solve. Each step starts
    // by raising every sub-pedestal cell onto rho_ped, and the mass drain
    // gates of the Riemann fluxes anchor at rho_ped instead of the
    // positivity floor, so the pedestal band {rho >= rho_ped} is
    // dynamically invariant.
    //
    // EQUIVALENCE ARGUMENT (why this realizes the offset-density variable
    // change without changing the unknown): write rho = rho_ped + delta.
    // In the offset formulation the mass unknown is the deviation delta,
    // the continuity system advects the (gated) deviation, and every
    // derived quantity (velocity, pressures, Ohm inputs) restores the
    // pedestal arithmetically -- so the positivity bound disappears from
    // the neighborhood of the halo operating point: delta = 0 is a
    // REGULAR interior point of the flow (all derived quantities are
    // evaluated at rho = rho_ped > 0), not a constraint boundary, and no
    // cell can sit ON a bound. The formulation here achieves the
    // identical geometry in the original variable: (i) the halo is
    // loaded/raised AT rho_ped, so derived quantities see rho >= rho_ped
    // exactly as the offset form's arithmetic restoration would; (ii) the
    // donor drain gate closes the mass outflow C^1-smoothly at rho_ped --
    // the same gated deviation flux, since rho u = (rho_ped + delta) u
    // gated to zero as delta -> 0 transports no mass out of an
    // exhausted-deviation cell; and (iii) the Newton admissibility bound
    // stays at the ORIGINAL positivity floor, orders of magnitude below
    // rho_ped, so it is strictly inactive on the halo -- the same
    // "no bound near the operating point" property the variable change
    // provides (there the bound is displaced to delta = -(rho_ped -
    // floor), equally unreachable). Both formulations therefore admit
    // the same invariant set, the same derived-quantity evaluations, and
    // a bound-free neighborhood of the halo band; they differ only in
    // which variable carries the Newton update, which is a linear
    // (constant-shift) reparameterization with identical Jacobians.
    //
    // MEASURED MOTIVATION: on the FRC free-boundary benchmark ladder
    // (494/257/172/177/150 steps across five floor/dust configurations,
    // all ending in frozen-solve aborts) the projection reports showed
    // 40-60k Newton-direction components clamped per solve across the
    // floor-riding halo, and the theta = 1 + vacuum-resistivity variant
    // relaxed the halo residual two orders of magnitude in one step and
    // then stagnated SOLELY on the admissibility-bound-resident
    // population -- proving a state with NO bound-resident cells is the
    // fix. The pedestal absorbs the bound into the operating point.
    // THE PEDESTAL IS A STATE, NOT A DENSITY: the raise and the drain
    // gates cover the ENERGY blocks as well, each at f times its own
    // instantaneous peak (U_e, the ion internal energy, and the CGL
    // pair), so the pedestal band is an f-scaled image of the peak
    // state. Measured motivation for the extension: with a mass-only
    // pedestal the first FRC benchmark attempt still froze from step 2
    // with ~19k projected direction components, and the projection
    // report located them almost entirely in the energy blocks (16k U_e
    // + 3k E_i components resting on the sanitize floors, vs 169 mass
    // components -- the mass pedestal had collapsed the 40-60k mass
    // population). A pedestal whose energies sit on their bounds is the
    // same bound-resident population in different variables; the
    // f-scaled-image rule displaces every block's operating point into
    // the interior by the same argument as the density.
    amrex::Real m_halo_pedestal_fraction = 0.0;
    // Velocity relaxation rate [1/s] of the pedestal band (0 disables).
    // The drag engages C^1-smoothly below TWICE the density pedestal
    // (weight 1 - floor_outflow_limiter(rho_old, rho_ped): full rate at
    // the pedestal, exactly zero at/above 2 rho_ped), keyed to the
    // step-old density like the halo source taper, so it is a per-solve
    // constant mask. Rationale: the pedestal's donor drain gates hold
    // the band's MASS flux at zero, but the band's MOMENTUM rows still
    // integrate the discrete-stress truncation forcing of the
    // surrounding plasma -- a residual demand the gated mass can never
    // supply, whose least-squares compromise is an irreducible Newton
    // plateau (measured on the FRC benchmark ladder: iteration-0
    // line-search death at a small constant residual). The diagonal,
    // JFNK-friendly drag (same pattern as vacuum_drag_rate) bounds the
    // band at the terminal velocity F/(rho nu) instead, so band momentum
    // cannot demand mass the gates refuse. Under the total_energy
    // closure the matched kinetic-energy drain keeps the ion INTERNAL
    // energy invariant (exactly so at theta = 1/2).
    amrex::Real m_halo_pedestal_drag_rate = 0.0;
    // Ion-energy relaxation rate [1/s] of the pedestal band (0
    // disables). Engages with a C^1 density mask of the drag's form
    // but an OCTAVE NARROWER window (full rate at/below 1.125
    // rho_ped, exactly zero at/above 1.25 rho_ped, a per-solve
    // constant keyed to the step-old density): the treatable
    // population is the gate-PINNED band (measured at 1.0-1.2
    // rho_ped), while the wall's live hot boundary layer -- E_i stock
    // 20-400x the pedestal image -- cycles through 1.3-1.8 rho_ped
    // and must not be drained (a 2x-window drain was measured to rip
    // it down to the image cell-by-cell, a realization-dependent
    // secular residual growth; the drag keeps its 2x window because
    // velocity relaxation is inert on the quasi-static layer).
    // Rationale (the last free band
    // channel, after the mass gates and the momentum drag): under a
    // ramping drive the band's rho sits gate-pinned at the pedestal
    // while the E_i rows keep integrating the drive forcing (measured
    // on the FRC benchmark ladder: wall-band scrape-off accretion grew
    // E_i x350 and pinned an O(1)-abs irreducible ion-energy residual
    // -- Newton's only admissible move, draining mass, is refused by
    // the gates). The diagonal relaxation drains the band's ion energy
    // toward its pedestal-consistent image instead, so the accretion
    // never builds into an inadmissible demand. Coupling to the
    // momentum drag (total_energy closure): E_i = e_int + |m|^2/(2rho)
    // and the drag already OWNS the kinetic channel (it decays m and
    // drains E_i by exactly the induced kinetic decay), so this term
    // acts ONLY on the internal part e_int = E_i - |m|^2/(2 rho), the
    // composition is triangular in (KE, e_int), and neither term
    // double-counts the other: d(KE)/dt = -2 nu_drag KE, d(e_int)/dt =
    // -nu_E max(e_int - e_ped, 0)-like. ONE-SIDED (rectified): a C^1
    // gate (floor_outflow_limiter(e_int, e_ped): zero at/below the
    // pedestal image, full exact rate at/above twice it) closes the
    // term from below, so it is a pure DRAIN and never a source --
    // a two-sided KE-following form was measured to pump the
    // KE-dominated wall-band cells (theta-stage E_i < KE) toward their
    // growing kinetic target, a secular runaway of the band's
    // conservative E_i stock. Under the cgl closure U_par and U_perp
    // (purely internal) drain toward their pedestal images directly,
    // with the same one-sided gates. Targets are clamped from below at
    // the corresponding positivity floors so the drain can never
    // demand an inadmissible state.
    amrex::Real m_halo_pedestal_energy_rate = 0.0;
    // ---- Floor-consistency relaxation rate [1/s] (0 = off, bit-identical
    // when off): the consistency completion of the admissibility floors.
    // Measured escalation trigger (production rr13e_C1w, the annihilation
    // endgame of a weak-anomalous-eta formation arm): a persistent active
    // set of 240-241 pure ion-energy components -- the live-plasma /
    // floor-sea interface ring -- was projected onto its admissibility
    // bounds every solve; the free-subspace rescue engaged (7 accepted
    // rescue steps) but the pinned-row defect GREW monotonically
    // 0.468 -> 1.004 (count 157 -> 241) until no admissible step reduced
    // even the free subspace and the freeze guard fired at 43.97 us. The
    // discrete equations there CONTINUOUSLY demand sub-floor evolution: a
    // consistency gap between the residual and the admissible set that no
    // line-search policy can close. This source closes it: every bounded
    // fluid block (the blocks ProjectAndLimitSolverStep clamps) gains the
    // one-sided per-cell residual source
    //     S = rate_eff * floor_consistency_deficit(u, ...),
    //     rate_eff = min(rate, 1/(theta dt)),
    // which supplies (never drains) exactly the demanded sub-bound drain
    // at bound-riding cells -- see floor_consistency_deficit in
    // ThetaImplicitMHD_K.H for the rectifier construction and the
    // one-sidedness / JFNK-exactness / exact-zero-above-bound /
    // no-overshoot guarantees. The reservoir's supplied mass and energy
    // are BOOKED per step (AccumulateFloorConsistencySupplyLedger) so the
    // non-conservation is tracked, like the wall ledgers. Wall-masked
    // (rigid-conductor) cells are excluded like every other source. ----
    amrex::Real m_floor_consistency_rate = 0.0;
    // Must match theta_implicit_mhd::floor_consistency_width_fraction
    // (ThetaImplicitMHD_K.H) so the default stays bit-identical.
    amrex::Real m_floor_consistency_width_fraction = 0.1;
    // Floor-consistency supply ledger: cumulative supplied mass [kg] and
    // fluid energy [J] ([kg/m^2], [J/m^2] in 1D: the cell measure is the
    // geometry's), evaluated at the accepted theta state each step and
    // optionally appended to floor_ledger_file as "step mass energy"
    // rows (every step; first write of a run truncates a stale file).
    // Counters restart at zero on a simulation restart.
    amrex::Real m_floor_supplied_mass = 0.0;
    amrex::Real m_floor_supplied_energy = 0.0;
    std::string m_floor_ledger_file;
    bool m_floor_ledger_started = false;
    // ---- the reference code's density eater (see ApplyDensityEater): per-STEP
    // fraction of the density excess above the target removed each step
    // (0 = off, bit-identical when off; the reference code hard-codes 0.2 --
    // dt-independent, like the source). ----
    amrex::Real m_density_eater_rate = 0.0;
    // Target = target_fraction x reference (the reference code's en0/100).
    amrex::Real m_density_eater_target_fraction = 0.01;
    // Reference base density [kg/m^3] (the reference code's en0 input, a NUMBER
    // density there -- multiply by the ion mass). Sentinel <= 0 defaults
    // to implicit_mhd.reference_mass_density.
    amrex::Real m_density_eater_reference_density = -1.0;
    // Dynamic reference (the reference code's en0_upd = 1, step.f90:218-225):
    // Reference = max(base, fraction x step-old global density peak);
    // 0 freezes the reference at the base (en0_upd = 0).
    amrex::Real m_density_eater_reference_peak_fraction = 0.1;
    // Axial band: "z_lo" (the reference code's sym_bc = 1: planes at the mirror
    // plane) or "z_center" (planes about the domain z-center); empty =
    // auto (z_lo when the z_lo pmc mirror is active, else z_center) --
    // The reference code's sym_bc dispatch (ntb.f90:783-792). Resolved in Define.
    std::string m_density_eater_band;
    // Band size in cells; <= 0 = the reference code's grid rule (z_lo: 2 + nz/500
    // planes from the boundary; z_center: half-width 1 + nz/1000 about
    // the center). An explicit value is the plane count (z_lo) or the
    // half-width (z_center).
    int m_density_eater_band_cells = -1;
    // Closed-flux gate (the reference code's psi(k) > 0, ntb.f90:799): +1 eats only
    // where sign * psi > 0 with psi(r, z) the poloidal flux integral of
    // the theta-stage TOTAL Bz (plasma + external under the split-field
    // drive; the reference code gates on the step-old psi, so any intra-step stage
    // is faithful), -1 flips the convention, 0 disables the gate. RZ
    // only: 1D has no poloidal flux, so the default (and only legal
    // value) there is 0.
    int m_density_eater_flux_sign = 0;
    // Eater removal ledger: cumulative removed mass [kg] and electron
    // energy [J] ([kg/m^2], [J/m^2] in 1D), booked at the committed
    // t^{n+1} state each step and optionally appended to
    // density_eater_ledger_file as "step mass energy" rows (every step;
    // first write of a run truncates a stale file). The ion energy
    // DENSITY is invariant under the eater by construction (the reference code's
    // wio/wik preservation), so the mass and electron columns close the
    // accounting. Counters restart at zero on a simulation restart.
    amrex::Real m_eater_removed_mass = 0.0;
    amrex::Real m_eater_removed_energy = 0.0;
    std::string m_density_eater_ledger_file;
    bool m_density_eater_ledger_started = false;
    // The frozen per-step pedestal state (0 while inactive): density,
    // electron energy, ion internal energy (total_energy closure), and
    // the CGL pair.
    amrex::Real m_halo_pedestal_density = 0.0;
    amrex::Real m_halo_pedestal_electron_energy = 0.0;
    amrex::Real m_halo_pedestal_ion_internal = 0.0;
    amrex::Real m_halo_pedestal_ion_parallel = 0.0;
    amrex::Real m_halo_pedestal_ion_perp = 0.0;
    // ---- Density-keyed vacuum resistivity D_vac [m^2/s] (0 disables):
    // the FIELD advance sees eta >= mu0 * D_vac * (rho_ref_ohm/rho)^2
    // with rho_ref_ohm the Ohm density guard; Joule heating keeps the
    // un-boosted user eta. See vacuum_keyed_resistivity in
    // ThetaImplicitMHD_K.H for the full rationale. hlld path only. ----
    amrex::Real m_vacuum_resistivity_diffusivity = 0.0;
    // ---- reference-code-style DYNAMIC reference density (en0_upd = 1,
    // step.f90 ~218-224: en0 = max(en00, 0.1 * MAXVAL(en)), refreshed
    // every step): fraction of the instantaneous (step-old) global
    // density peak keying the density-keyed halo mechanisms -- the
    // field-eta vacuum boost (vacuum_keyed_resistivity), the
    // joule_ohm_current diffusion-dominance criterion (through the
    // boosted eta_field it tests), and the conduction halo boost
    // (conduction_halo_boost). The effective reference is
    // max(Ohm guard, fraction * rho_peak) -- see
    // VacuumReferenceMassDensity() -- so the static Ohm-guard keying is
    // the exact fraction = 0 limit (bit-identical default). Rationale:
    // the static guard protects only near-vacuum cells, leaving the
    // 1e18..6e19 halo band taking full Joule/conduction; the reference code keys
    // the same three mechanisms on the tenth-of-peak en0, which tracks
    // compression. The Joule COEFFICIENT keeps the un-boosted user eta
    // (the eta_jh/eta split of the reference code's step.f90) and the eta-parser
    // density arguments keep the static Ohm-guard floor. ----
    amrex::Real m_vacuum_reference_peak_fraction = 0.0;
    // The frozen per-step dynamic reference (fraction * step-old global
    // density peak; 0 while inactive), refreshed once per step by
    // RefreshVacuumReferenceDensity, a per-solve constant like the halo
    // pedestal.
    amrex::Real m_vacuum_reference_mass_density = 0.0;
    // Last rank-0-printed effective reference (print on >1% change).
    amrex::Real m_vacuum_reference_printed = -1.0;
    // ---- Resistive theta (default: the global implicit_evolve.theta).
    // The dissipative Ohm terms (eta J including the vacuum boost, and
    // the hyper-resistive term) are evaluated at the theta_r-weighted
    // current J^{n+theta_r} while the ideal EMF keeps the global theta.
    // Rationale from the theta-scheme amplification factor: for a stiff
    // resistive mode with z = eta dt k^2/mu0 >> 1 the trapezoidal update
    // has g = (1 - z/2)/(1 + z/2) -> -1, i.e. the mode is marginally
    // damped and flips sign every step -- measured on the FRC halo as a
    // never-decaying residual plateau under theta = 1/2 that a single
    // theta = 1 step removed. Backward Euler on the resistive terms
    // alone (theta_r = 1) gives g = 1/(1 + z) -> 0, damping the stiff
    // halo modes in one step while the ideal dynamics keeps its
    // second-order centering. ----
    amrex::Real m_resistive_theta = -1.0;
    // ---- Conduction theta (default: the global implicit_evolve.theta).
    // The conductive-flux ENERGY arguments -- the per-side specific
    // internal energies whose normal/tangential differences drive the
    // flux, the wall thermal-drain interior energy, and the
    // free-streaming-cap temperatures -- are evaluated at the theta_c
    // stage
    //     e^{n+theta_c} = (theta_c/theta) e^{n+theta}
    //                     + (1 - theta_c/theta) e^n,
    // the conduction twin of m_resistive_theta: the extrapolation is
    // linear in the theta-stage values, so matrix-free Jacobian probes
    // see the shifted centering exactly. Rationale from the theta-scheme
    // amplification factor: at halo diffusion numbers
    // theta dt chi / dx^2 >> 1 the trapezoidal update leaves the
    // grid-Nyquist conduction modes at g -> -1 (amplitude ~1, sign
    // flipped every step) -- measured on production arms as a halo
    // temperature checkerboard (Nyquist fluctuation-power fraction 0.6,
    // unphysical Ti spikes) preceding Newton freeze-guard aborts.
    // conduction_theta = 1 damps them as g = 1/(1 + z) -> 0 while the
    // wave/advective dynamics keep the global centering. The conduction
    // COEFFICIENTS keep their own conduction_coefficient_state rule. ----
    amrex::Real m_conduction_theta = -1.0;
    // Non-periodic z (RZ only): the solver fills axial domain ghosts with
    // zero-gradient (Neumann) values, giving outflow ends instead of the
    // periodic seam. Set automatically when the z geometry is non-periodic.
    bool m_z_neumann = false;
    // Opt-in no-reflux variant of the Neumann-z outflow: ghost axial
    // momentum clamped to the outward sign (drainage allowed, reflux
    // forbidden). Default keeps the passive zero-gradient behavior.
    bool m_z_outflow_no_reflux = false;
    // Fluid-moment ghost treatment at the non-periodic z ends (recast
    // path, RZ): "neumann" (default; the passive zero-gradient clamp,
    // which is a zero-flux end for the energies -- device-generated heat
    // cannot escape through the exhaust), "wall_temperature" (ghosts
    // copy the end plane's density and momentum but their ENERGIES are
    // set to z_wall_temperature at the ghost density, ion internal
    // energy plus the kinetic energy of the copied momentum, so the end
    // faces exchange advectively against a T_wall reservoir -- the z
    // analog of the r-wall temperature anchoring), or "outflow" (the
    // ghost AXIAL momentum is C-infinity rectified to its OUTGOING
    // part, the exact recipe of the r_max absorbing wall: the ends
    // advect mass/energy out at interior values but never feed plasma
    // back).
    std::string m_z_boundary_fluid = "neumann";
    // Wall temperature [eV] of z_boundary_fluid = wall_temperature.
    amrex::Real m_z_wall_temperature = 0.0;
    // Fluid-moment ghost treatment at the z_lo end ONLY (RZ, recast
    // path): empty (default) inherits z_boundary_fluid; "symmetry" makes
    // z = z_lo a mirror-symmetry plane -- the z_lo ghost rows are the
    // exact linear reflection of the interior (even scalars and
    // tangential momentum, odd axial momentum), which yields zero mass/
    // energy/normal-momentum advective flux and zero conductive heat
    // flux through the plane. Requires (and is required by)
    // boundary.field_lo = pmc on z: the PMC field parities (Br, B_theta
    // odd; Bz even; E_r, E_theta even; E_z odd) are the mirror parities
    // of the electromagnetic subsystem. z_hi keeps z_boundary_fluid.
    std::string m_z_lo_boundary_fluid;
    // Open upper radial boundary (RZ only): boundary.field_hi = open on r
    // selects the Green's-function free-space fill for the fields; the
    // fluid then gets zero-gradient (outflow) radial ghosts at r_max
    // instead of the reflecting conducting-wall mirror.
    bool m_r_open = false;
    // Open axial cap faces (RZ only): boundary.field_lo/hi = open on z
    // selects the Green's-function free-space fill for the field z ghosts
    // of that cap; the fluid moments keep the Neumann outflow ghosts
    // (m_z_neumann), but the field/current z-ghost fills defer to the
    // Green's values instead of re-imposing the z-invariant clamp.
    bool m_z_lo_open = false;
    bool m_z_hi_open = false;
    // Mirror-symmetry z_lo face (RZ only): boundary.field_lo = pmc on z,
    // paired 1:1 with implicit_mhd.z_lo_boundary_fluid = symmetry (both
    // asserted at Define). Every solver z-ghost fill then replaces the
    // zero-gradient image at z_lo with the PARITY image: even fields
    // keep the staggering-aware mirror (identical to the Neumann fill
    // for nodal-in-z data), odd fields get the sign-flipped mirror, and
    // the z_hi face keeps its configured behavior.
    bool m_z_lo_pmc = false;
    // Fluid ghost treatment at an OPEN r_max: "outflow" (default;
    // zero-gradient), "reflect" (keep the no-normal-flow mirror while
    // the field is free-space -- removes the wall-cell drainage channel
    // during violent boundary relaxations; the wall face is then a true
    // zero-flux wall for every advective fluid channel), or "absorb"
    // (solid-conductor wall that swallows incident plasma, not stores
    // it: outflow's impedance-matched clamp ghosts with the ghost's
    // NORMAL momentum C-infinity rectified to its outward part, the
    // smoothed radial twin of the z-outflow no-reflux clamp -- plasma
    // leaves at its incident, signal-limited rate and the wall never
    // feeds plasma back; hlld only, tallied by the absorbed-mass/energy
    // ledger).
    std::string m_r_open_fluid = "outflow";
    // Absorbing-wall ledger (r_open_fluid = absorb): cumulative mass [kg]
    // and fluid energy [J] exported through the r_max wall face
    // (r-weighted face-flux sums of the accepted theta-state fluxes,
    // integrated per step), printed every absorb_ledger_interval steps
    // (0 disables printing; accumulation always runs) and optionally
    // appended to absorb_ledger_file as "step mass energy" rows.
    // The counters restart at zero on a simulation restart.
    amrex::Real m_absorbed_wall_mass = 0.0;
    amrex::Real m_absorbed_wall_energy = 0.0;
    int m_absorb_ledger_interval = 1;
    std::string m_absorb_ledger_file;
    // First ledger-file write of this run truncates a stale file.
    bool m_absorb_ledger_started = false;
    // Shaped-wall deposition ledger (wall_thermal_bc != none): cumulative
    // mass [kg] and fluid energy [J] deposited through the stair-step
    // interface faces into the rigid-conductor band (conductive drain +
    // advective capture, r-weighted accepted theta-state face fluxes,
    // both r- and z-normal interfaces), printed on the absorb ledger's
    // cadence and optionally appended to wall_ledger_file as
    // "step mass energy" rows. Counters restart at zero on restart.
    amrex::Real m_shaped_wall_mass = 0.0;
    amrex::Real m_shaped_wall_energy = 0.0;
    std::string m_wall_ledger_file;
    bool m_wall_ledger_started = false;
    amrex::Real m_positivity_safety = 0.99;
    // Production default: the recast central/FD flux.
    // The legacy "legacy_e_centered" (non-recast, E-based) form and every Riemann flux
    // (rusanov/hllc/hlld) remain explicit opt-ins. Note central requires
    // a positive implicit_mhd.viscosity (asserted with a clear message):
    // the flux carries no Riemann dissipation by design.
    std::string m_fluid_flux = "central";
    // Conservative-form recast (fluid_flux = hlld): B replaces E as the
    // JFNK array block and E is derived from the Riemann EMF + eta J.
    bool m_use_hlld = false;
    // hlld is non-production (kernel regression coverage only) and must
    // be opted into explicitly via implicit_mhd.allow_hlld.
    bool m_allow_hlld = false;
    // fluid_flux = central: the SAME recast with the Chacon-style central
    // conservative flux in place of the HLLD fan (explicit viscosity
    // replaces the Riemann dissipation).
    bool m_use_central = false;
    // Either recast flux (hlld or central): gates all recast plumbing.
    bool m_use_recast = false;
    // Explicit ion kinematic viscosity nu_i [m^2/s] of the recast face
    // fluxes (0 = off; required positive for fluid_flux = central). The
    // normal-gradient viscous stress and its velocity-weighted work enter
    // the face registers as an exactly conservative pair.
    amrex::Real m_viscosity = 0.0;
    // ---- Wall-row viscosity mask (implicit_mhd.wall_viscosity_mask,
    // default off = bit-identical; requires an active wall mask): the
    // The reference code's slip wall rows (step.f90 disip: CALL set(vis_*, 'skin'/
    // 'bndy'/'subr'/'subz', small_vis) -- viscosity vanishes on the
    // wall-adjacent rows). Zeroes the viscous face COEFFICIENT -- the
    // momentum stress AND its velocity-weighted heating work, which
    // share the same coefficient assembly (an exactly conservative
    // pair, so both must go together) -- at every face either of whose
    // adjacent cells lies within wall_viscosity_mask_width cells
    // (Chebyshev distance over the stair-step first-masked tables) of
    // the masked wall contour. Kills the coil-footprint viscous-heating
    // bead row a uniform viscosity deposits under every wall ripple
    // (heating the reference code does not have). The mask is static geometry after
    // Define, so JFNK probes see constant structure. ----
    bool m_wall_viscosity_mask = false;
    int m_wall_viscosity_mask_width = 2;
    // Thermal diffusivities chi_i/chi_e [m^2/s] of the recast face fluxes
    // (0 = off): conservative internal-energy diffusion
    // -chi rho_f d(e_spec)/dn (q = -kappa grad T, Chacon JCP 526 (2025))
    // on the ion total-energy and electron energy face channels.
    //
    // Each channel takes EITHER the legacy numeric key
    // (implicit_mhd.thermal_diffusivity_ion/electron, bit-identical
    // constant fast path) OR a parser expression
    // implicit_mhd.thermal_diffusivity_*(rho,Te,Ti,J,t), evaluated at the
    // e_spec-gradient FACES (z-faces in 1D; r- and z-faces in RZ -- the
    // same normal-gradient faces the conductive flux lives on) from the
    // donor-averaged face state: rho = the Ohm-floored face charge
    // density max((q/m) rho_f, rho_q_floor); Te [K] and Ti [K] the
    // temperature-primary face ratios p_face/(n_f kB) built from the SAME
    // recovered/floored CellState pressures the physical fluxes use (Ti
    // is 0 outside the total_energy closure, where no recovered ion
    // pressure exists); J the |.| of the face-averaged cell-centered
    // total current; t the residual (theta-stage) time. Expressions must
    // be C-infinity in the state symbols (smooth floors, no max()), like
    // the resistivity parser.
    amrex::Real m_thermal_diffusivity_ion = 0.0;
    amrex::Real m_thermal_diffusivity_electron = 0.0;
    std::unique_ptr<amrex::Parser> m_chi_ion_parser;
    std::unique_ptr<amrex::Parser> m_chi_electron_parser;
    std::string m_chi_ion_expression;
    std::string m_chi_electron_expression;
    amrex::ParserExecutor<5> m_chi_ion;
    amrex::ParserExecutor<5> m_chi_electron;
    bool m_chi_ion_is_parser = false;
    bool m_chi_electron_is_parser = false;
    // Free-streaming limiter of the conductive face fluxes (BOTH
    // channels): chi_eff = chi / (1 + |q|/(f q_fs)) with q the unlimited
    // conductive face flux and q_fs = n kB T v_th the free-streaming flux
    // of the channel's species (v_th = sqrt(kB T/m), evaluated from the
    // same donor-averaged face state as the parser symbols). f = 0
    // (default) disables the limiter EXACTLY (bit-identical); f > 0
    // applies the smooth harmonic cap with no branches. NEVER silent:
    // the factor is printed in the solver banner.
    amrex::Real m_conduction_flux_limit_factor = 0.0;
    /** conduction_coefficient_state = step_old: the conduction
     * COEFFICIENT inputs (face rho_f, charge density, temperatures --
     * feeding the parser chi, the Braginskii coefficients, and the
     * free-streaming caps) are evaluated from the step-old fields,
     * per-solve constants, so Newton sees linear diffusion in the
     * energies. The flux keeps the live theta-state gradient. */
    bool m_conduction_coefficient_step_old = false;
    // Thermal-conduction closure of the conductive face fluxes
    // (implicit_mhd.thermal_conduction_model): "isotropic" (default;
    // bit-identical to the scalar thermal_diffusivity_* path) or
    // "braginskii" -- the anisotropic tensor flux
    //   q_n = -rho_f [chi_perp de/dn
    //                 + (chi_par - chi_perp) bhat_n (bhat . grad e)]
    // per channel, with e the SAME specific internal energies the
    // isotropic path diffuses, bhat from the face-averaged TOTAL B, and
    // chi_par/chi_perp the Braginskii (1965) Z = 1 parallel/perpendicular
    // thermal diffusivities evaluated from the donor-averaged face state
    // (electron channel always; ion channel under ion_closure =
    // total_energy). Mutually exclusive with the constant/parser
    // diffusivities above: braginskii computes its own coefficients.
    std::string m_thermal_conduction_model = "isotropic";
    bool m_conduction_braginskii = false;
    // Braginskii cross-term tangential-gradient limiter: "none" keeps
    // the legacy centered corner average (bit-identical); "minmod" is
    // the Sharma-Hammett monotone form (see minmod_pair in _K.H) that
    // kills the cross-term flux at tangential extrema -- the fix for
    // the plasma-edge electron_energy direction pinning.
    // Default minmod (2026-08-29 parity audit): the centered corner
    // stencil of the Braginskii cross term is non-monotone where
    // The reference code's FEM form is SPD; at clamp-forced anisotropy it carried a
    // measured wall-ledger runaway (3.3e11 J -> 3.5e4 J under minmod in
    // the E4 A/B). "none" remains selectable for the centered reference.
    std::string m_braginskii_tangential_limiter = "minmod";
    // Coulomb logarithm of the Braginskii collision times tau_e/tau_i.
    amrex::Real m_conduction_coulomb_log = 10.0;
    // Optional absolute clamps [m^2/s] on the Braginskii chi_par and
    // chi_perp (0 = off): C-infinity smooth-max floor at chi_min and C^2
    // soft cap at chi_max (exact pass-through below 0.9 chi_max), applied
    // per channel BEFORE the free-streaming limiter.
    amrex::Real m_conduction_chi_min = 0.0;
    // Reference-matched per-component clamps (xil*_mn/mx vs xip*_mn/mx):
    // -1 (default) inherits the shared chi_min/chi_max above.
    amrex::Real m_conduction_chi_par_min = -1.0;
    amrex::Real m_conduction_chi_par_max = -1.0;
    amrex::Real m_conduction_chi_perp_min = -1.0;
    amrex::Real m_conduction_chi_perp_max = -1.0;
    amrex::Real m_conduction_chi_max = 0.0;
    // Quasi-shorting cross-field boost of the Braginskii chi_perp
    // (implicit_mhd.conduction_qs_*, braginskii only; qs_chi = 0
    // disables): subgrid turbulence on broken flux surfaces shorts the
    // channel across surfaces, modeled as an ADDITIVE cross-field
    // diffusivity keyed to the pseudo-entropy excess
    //     s = (T/T0) (rho0 / rho_guarded)^{2/3},
    //     rho_guarded = sqrt(rho^2 + rho_guard^2),
    // above the load envelope (T0 = conduction_qs_reference_temperature
    // [eV], rho0 = reference_mass_density, rho_guard the Ohm density
    // guard), ramped by the C-infinity smooth-max
    //     chi_add = qs_chi * 0.5 ((s - onset)
    //                             + sqrt((s - onset)^2 + w^2)),
    //     w = 0.3 (onset - 1),
    // CENTERED ABOVE the envelope (onset > 1 enforced): a ramp centered
    // ON the envelope leaks w/2 of the amplitude onto every on-adiabat
    // cell (measured fatal in production). Electron channel always; ion
    // channel under total_energy, keyed on the ion temperature with the
    // same T0. The s inputs follow conduction_coefficient_state like
    // every other Braginskii coefficient input; the chi_min/max perp
    // clamp applies AFTER the addition.
    amrex::Real m_conduction_qs_chi = 0.0;
    amrex::Real m_conduction_qs_onset = 1.5;
    amrex::Real m_conduction_qs_reference_temperature = 0.0;
    // Density-keyed halo boost of the Braginskii chi_perp
    // (implicit_mhd.conduction_halo_boost [m^2/s], braginskii only,
    // 0 = off): the reference code's low-density perp-chi boost
    // (ntb.f90 t_cond ~584: xip = MAX(xip, xip*(en0/en)^2*dp_mn)),
    // draining residual halo heat to the wall bath while the bulk keeps
    // physical Braginskii. chi_perp is composed with the boost
    // diffusivity chi_halo = D_boost * (rho_ref/rho)^2 -- reaching
    // D_boost AT the reference density, quadratic in reference/n below
    // it, vanishing quadratically above -- through the same C-infinity
    // quadrature smooth max as vacuum_keyed_resistivity (whose kernel is
    // reused verbatim), with rho_ref the mechanism's shared dynamic
    // Reference (VacuumReferenceMassDensity: the reference code's en0) and the
    // division guarded at the positivity floor. ORDER OF COMPOSITION
    // (matches the reference code's ntb.f90 t_cond): the boost applies to the raw
    // Braginskii perp value (after the additive quasi-shorting term,
    // which is our own knob) and BEFORE the chi_min/chi_max clamps --
    // in the reference code the boost at line ~584 precedes the
    // xipi_mn/xipi_mx clamps at ~592-593, so the clamps cap the BOOSTED
    // value (chi_max still bounds the halo diffusivity). Perp only:
    // The reference code does not boost xil (it only re-enforces xil >= xip for its
    // (xil-xip)/B^2 operator form, which our SPD tensor split does not
    // need). Both species channels, with the SAME
    // conduction_coefficient_state face inputs as the base chi (the
    // boost is evaluated at the coefficient-state face density, so
    // step_old keeps Newton-linear diffusion).
    amrex::Real m_conduction_halo_boost = 0.0;
    // Floor-independent corner width of the C^1 smooth-max pressure
    // recovery (see FluxParameters::pressure_corner_width_fraction;
    // 0 = legacy width = internal-energy floor).
    amrex::Real m_pressure_corner_width_fraction = 0.0;
    // C-infinity smoothing widths of the HLLD wave fan (see FluxParameters
    // in ThetaImplicitMHD_K.H); inputs implicit_mhd.hlld_kappa_*.
    amrex::Real m_hlld_kappa_signal = 0.05;
    amrex::Real m_hlld_kappa_contact = 0.05;
    amrex::Real m_hlld_kappa_bn = 0.05;
    amrex::Real m_hlld_kappa_denominator = 0.05;
    // HLLD fan closure: "consistent" (default; the wave fan sees the
    // recovered p_i(E_i)) or "barotropic" (the fan structure sees the
    // polytropic p_i(rho); physical fluxes stay consistent). The
    // barotropic fan, with a widened hlld_kappa_bn, is the robustness
    // configuration for violent rotational/compound structures.
    std::string m_hlld_fan_closure = "consistent";
    // Ion-energy channel: "star" (default; fan-consistent star energy,
    // contact-scale dissipation) or "llf" (the fast-alpha enthalpy
    // fallback, which diffuses every E_i gradient at the fast speed).
    std::string m_hlld_ion_energy_flux = "star";
    // LHLLD all-speed scaling of the star-pressure velocity-difference
    // term (Minoshima & Miyoshi, JCP 446 (2021) 110639).
    bool m_hlld_all_speed = true;
    // HLLC signal-speed closure: "consistent" (default; wave-speed
    // estimates from the same p_i as the physical flux) or "barotropic"
    // (wave-speed estimates from the polytropic p_i(rho), decoupling the
    // HLLC wave structure from the E_i unknown; total_energy only).
    std::string m_hllc_signal_closure = "consistent";
    // Smooth contact-side blending width kappa for the HLLC flux
    // (0 disables the blend and keeps the hard upwind-side switch).
    amrex::Real m_hllc_contact_blend = 0.0;
    bool m_evolve_ion_fluid = true;
    bool m_include_joule_heating = true;
    // The reference code's f_ohmi: direct ion share of the Joule power (0 = legacy
    // all-electron; (0,1] constant; -1 = the Te-keyed tanh split
    // 0.1 + 0.4 (1 + tanh((Te[eV] - 60)/20)) the reference shot flies).
    amrex::Real m_joule_ion_fraction = 0.0;
    // ---- reference-code-style Ohm-current Joule quench (joule_ohm_current,
    // default off), ported from the reference code's "fancy current" (psi.f90): at
    // Joule-deposition cells where the FIELD advance is diffusion
    // dominated, dt eta_field / mu0 > min(dx)^2, the |J|^2 of the
    // electron-energy Joule deposit is evaluated from Ohm's law as
    // |E|^2 / eta_field^2 -- E the SOLVED stage electric field (see
    // OhmElectricFieldCCName) and eta_field the SAME floored/boosted
    // field-advance resistivity the field solve uses (the user eta
    // through vacuum_keyed_resistivity plus the wall-band override,
    // composed like GetMHDFieldResistivityCCForPC) -- instead of the
    // curl-B current. The heating COEFFICIENT keeps the un-boosted user
    // eta (the eta_jh/eta split of the reference code's step.f90). Where eta_field is
    // vacuum-boosted this quenches the halo Joule deposit quadratically:
    // the stiff implicit field solve relaxes E while the pointwise
    // curl-B current stays gradient-noise scale. Binary per-cell switch,
    // exactly as the reference code (no smooth blend). ----
    bool m_joule_ohm_current = false;
    // Circuit-in-the-residual coupling: every residual evaluation executes
    // the "externalcoiltheta" python callback (and FinishStateUpdate the
    // "externalcoilfinish" one) and then re-refreshes the external fields,
    // so python can re-advance a coupled circuit against the current
    // iterate's plasma flux linkage and push updated coil scales. The
    // segregated one-step-lagged alternative is unstable whenever the
    // coil-plasma mutual coupling is strong.
    bool m_external_field_iteration = false;
    // implicit_mhd.circuit_hook_scope = newton: fire the external-coil
    // hook ONLY on the residual evaluations at accepted Newton iterates
    // -- never inside matrix-free Jacobian probes or line-search trials,
    // which reuse the coil scales cached from the last iterate
    // evaluation (a lagged-circuit quasi-Newton: the Jacobian and the
    // Armijo grades see a per-iterate frozen circuit, possibly costing a
    // few extra Newton iterations, while the converged answer is
    // unchanged -- convergence is still tested on the live-coupled
    // iterate residual, and the end-of-step "externalcoilfinish" commit
    // is untouched). Removes the per-evaluation python round-trip and
    // its host-device flux-probe reductions from the GMRES/line-search
    // inner loops. Default false (scope = residual, bit-identical).
    bool m_circuit_hook_newton_scope = false;
    // Whether the next non-Jacobian ComputeRHS call evaluates the
    // residual at an accepted Newton iterate (maintained by the Newton
    // solver through SetResidualIsNewtonIterate; true by default so
    // solvers that never notify keep residual-scope semantics).
    bool m_residual_is_newton_iterate = true;
    // implicit_mhd.circuit_driver = native: drive the circuit through the
    // C++ circuit-coupling engine (circuit.coils + circuit.engine,
    // typically a compiled ExternalCircuit plugin) instead of the python
    // "externalcoiltheta"/"externalcoilfinish" hooks. Same firing points
    // and circuit_hook_scope semantics as the python driver, with the
    // python round-trip and its per-evaluation device->host flux pulls
    // replaced by the engine's batched device probes (one device pass,
    // one sync, one all-reduce per measurement) and an in-process
    // AdvanceInterval: residual scope restores EXACT circuit-in-residual
    // coupling at native cost. In-residual (accept = false) advances
    // target the theta-stage time and replay from the committed interval
    // entry; the single accept = true advance and FinishStep commit fire
    // from FinishStateUpdate on the accepted end-of-step state, so
    // discontinuous engine transitions latch in committed time only.
    // Default false (driver = python, bit-identical).
    bool m_circuit_native = false;
    // Whether the native circuit step is open: BeginStep is fired lazily
    // at the FIRST qualifying residual evaluation of the step, where the
    // just-computed plasma current is exactly the committed t^n state
    // (the Newton initial iterate is the old state), so the interval
    // entry linkages are seeded without an extra full-grid curl.
    bool m_circuit_step_open = false;
    // Per-step "externalcoiltheta" firing count, printed at the end of
    // OneStep when the circuit coupling is active (the sync-tax
    // instrument for circuit_hook_scope).
    amrex::Long m_circuit_hook_calls = 0;
    // Stair-step conducting shaped-wall mask (implicit_mhd.wall_model,
    // RZ recast path only; default none = bit-identical): projects the
    // tangential E to the conductor condition on/outside the revolved
    // wall polyline in every Ohm assembly -- "pec" pins the TOTAL field
    // (freezes total flux inside the metal: the honest perfect-conductor
    // response, for raw waveforms), "pec_response" pins the PLASMA-
    // RESPONSE field only (plasma flux cannot leak, the prescribed drive
    // is transparent: run32 EB parity for fitted waveform composites).
    // The preconditioner sees the same mask through GetMHDWallMaskForPC
    // (identical row drops in both contracts). The fluid is untouched
    // (the dust/vacuum fill is the fluid-side wall analog).
    ImplicitMHDWallMask m_wall_mask;

    std::unique_ptr<amrex::Parser> m_mass_density_parser;
    std::unique_ptr<amrex::Parser> m_velocity_x_parser;
    std::unique_ptr<amrex::Parser> m_velocity_y_parser;
    std::unique_ptr<amrex::Parser> m_velocity_z_parser;
    std::unique_ptr<amrex::Parser> m_electron_pressure_parser;
    std::unique_ptr<amrex::Parser> m_ion_pressure_parser;
    std::unique_ptr<amrex::Parser> m_ion_anisotropy_parser;
    amrex::ParserExecutor<3> m_mass_density;
    amrex::ParserExecutor<3> m_velocity_x;
    amrex::ParserExecutor<3> m_velocity_y;
    amrex::ParserExecutor<3> m_velocity_z;
    amrex::ParserExecutor<3> m_electron_pressure;
    amrex::ParserExecutor<3> m_ion_pressure;
    amrex::ParserExecutor<3> m_ion_anisotropy;

    void SaveMagneticField ();
    void UpdateWarpXFields (const WarpXSolverVec& state, amrex::Real start_time);

    /** The C++ circuit-coupling engine's coupler for the native circuit
     * driver; aborts with a configuration message when circuit.coils /
     * circuit.engine are not set up. */
    [[nodiscard]] CircuitCoupler& NativeCircuitCoupler () const;
    [[nodiscard]] theta_implicit_mhd::FluxParameters MakeFluxParameters () const;

    /** Per-block admissibility lower bounds of the fluid blocks, in the
     * order (rho, U_e, E_i or U_par, U_perp or U_i) used by LimitSolverStep and
     * ProjectAndLimitSolverStep: the bound of block b at a cell with
     * step-old density rho_old and step-old value u_old is
     *
     *     max(floors[b], temperature_coefficients[b] * rho_old)
     *         where u_old >= temperature_coefficients[b] * rho_old
     *     floors[b] otherwise (temperature ratchet not engaged),
     *
     * i.e. the absolute pressure/energy floors extended by the
     * temperature floors' n kB T_floor equivalents (n = rho (q/m)/e for
     * the quasi-neutral single-ion fluid). */
    struct AdmissibilityBounds {
        std::array<amrex::Real, 4> floors;
        std::array<amrex::Real, 4> temperature_coefficients;
    };
    [[nodiscard]] AdmissibilityBounds MakeAdmissibilityBounds () const;

    /** Split-field external-A bookkeeping: add (sign = +1) or subtract
     * (sign = -1) the stored hybrid external E/B fields to/from
     * Efield_fp/Bfield_fp. Between steps the WarpX fields hold total
     * fields; inside the nonlinear solve they hold the plasma response
     * only. Using the stored multifabs (rather than re-evaluating the
     * time function) makes each subtract/add cycle exact. */
    void AddExternalFieldsToTotals (amrex::Real sign) const;
};

#endif
