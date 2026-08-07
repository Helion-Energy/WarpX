/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef THETA_IMPLICIT_MHD_H_
#define THETA_IMPLICIT_MHD_H_

#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <array>
#include <memory>
#include <string>

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
    /** Recompute the dynamic halo pedestal density once per step (frozen
     * for the whole nonlinear solve) and raise sub-pedestal cells onto
     * it. No-op when halo_pedestal_fraction = 0. See
     * m_halo_pedestal_fraction for the offset-density rationale. */
    void RefreshHaloPedestal ();
    /** Mass-density equivalent of the Ohm's-law division guard
     * (hybrid_pic_model.n_floor). Deliberately independent of the fluid
     * admissibility floor m_mass_density_floor: eta/T_e evaluations and
     * the Ohm -(u x B) velocity reconstruction floor their inputs here,
     * while the fluid floor stays a pure positivity guard. Computed from
     * the live hybrid n_floor (Python can update it at runtime). */
    [[nodiscard]] amrex::Real OhmMassDensityFloor () const;
    void FinishStateUpdate (amrex::Real end_time);
    void FillFluidSources (const WarpXSolverVec& state);
    void ApplyFluidDomainBoundaries (amrex::MultiFab& density, amrex::MultiFab& momentum,
                                     amrex::MultiFab& electron_energy,
                                     amrex::MultiFab& ion_energy,
                                     amrex::MultiFab& ion_parallel_energy,
                                     amrex::MultiFab& ion_perp_energy) const;
    void ApplyNeumannZDomainGhosts (amrex::MultiFab& mf) const;
    void ApplyScalarRadialDomainGhosts (amrex::MultiFab& mf) const;
    void FillCellCenteredElectromagneticFields ();
    void ComputeFluidRHS (WarpXSolverVec& rhs, amrex::Real time) const;
    void ComputeFluidRHSFromFaceFluxes (WarpXSolverVec& rhs, amrex::Real time) const;
    void ComputeFaceFluxes ();
    void ComputeDirectionalFaceFluxes (amrex::MultiFab& face_flux_mf,
                                       int normal_direction);
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

    [[nodiscard]] amrex::GpuArray<amrex::Real, 3>
    GetMHDReferenceMagneticFieldForPC () const override;

    [[nodiscard]] bool GetMHDIncludeHallTermForPC () const override;

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

  private:
    inline static constexpr char TotalCurrentCCName[] = "implicit_mhd_total_current_cc";
    inline static constexpr char MagneticFieldCCName[] = "implicit_mhd_magnetic_field_cc";
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
    // polytropic law), "total_energy" (E_i evolved as a JFNK block), or
    // "cgl" (bi-Maxwellian: U_par = p_par/2 and U_perp = p_perp evolved
    // as JFNK blocks with CGL work terms and ion-ion isotropization;
    // hlld only). Under cgl the fan sees the effective isotropic
    // pressure p_eff = (p_par + 2 p_perp)/3 and the pressure recovery
    // is LINEAR in the unknowns (no kinetic-energy subtraction).
    std::string m_ion_closure = "barotropic";
    amrex::Real m_ion_pressure_floor = 0.0;
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
    // Non-periodic z (RZ only): the solver fills axial domain ghosts with
    // zero-gradient (Neumann) values, giving outflow ends instead of the
    // periodic seam. Set automatically when the z geometry is non-periodic.
    bool m_z_neumann = false;
    // Opt-in no-reflux variant of the Neumann-z outflow: ghost axial
    // momentum clamped to the outward sign (drainage allowed, reflux
    // forbidden). Default keeps the passive zero-gradient behavior.
    bool m_z_outflow_no_reflux = false;
    // Open upper radial boundary (RZ only): boundary.field_hi = open on r
    // selects the Green's-function free-space fill for the fields; the
    // fluid then gets zero-gradient (outflow) radial ghosts at r_max
    // instead of the reflecting conducting-wall mirror.
    bool m_r_open = false;
    // Fluid ghost treatment at an OPEN r_max: "outflow" (default;
    // zero-gradient) or "reflect" (keep the no-normal-flow mirror while
    // the field is free-space -- removes the wall-cell drainage channel
    // during violent boundary relaxations).
    std::string m_r_open_fluid = "outflow";
    amrex::Real m_positivity_safety = 0.99;
    std::string m_fluid_flux = "centered";
    // Conservative-form recast (fluid_flux = hlld): B replaces E as the
    // JFNK array block and E is derived from the Riemann EMF + eta J.
    bool m_use_hlld = false;
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
    // Circuit-in-the-residual coupling: every residual evaluation executes
    // the "externalcoiltheta" python callback (and FinishStateUpdate the
    // "externalcoilfinish" one) and then re-refreshes the external fields,
    // so python can re-advance a coupled circuit against the current
    // iterate's plasma flux linkage and push updated coil scales. The
    // segregated one-step-lagged alternative is unstable whenever the
    // coil-plasma mutual coupling is strong.
    bool m_external_field_iteration = false;

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
    [[nodiscard]] theta_implicit_mhd::FluxParameters MakeFluxParameters () const;

    /** Per-block admissibility lower bounds of the fluid blocks, in the
     * order (rho, U_e, E_i or U_par, U_perp) used by LimitSolverStep and
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
