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

    // NOTE: Jacobian probe perturbations stay LIMITED for this solver
    // (base-class default). Although every floored quantity is clamped
    // pointwise inside the residual kernels (so inadmissible probe states
    // would be evaluable), an unlimited probe crossing a hard clamp kink
    // measures the flat exterior side of the floor instead of the
    // interior slope, and GMRES then confidently steers into the bounds
    // (measured: Newton stalls at the first FRC step with hundreds of
    // projected components). The componentwise projection keeps a small
    // slack margin above each bound precisely so limited probes always
    // retain an admissible perturbation. Caveat for the future: the
    // limiter is GLOBAL (one bound-resting cell shrinks every probe's
    // epsilon toward that slack margin), so a large floored population
    // can degrade the finite-difference Krylov space; with the hlld
    // path's C-infinity floor smoothing this trade-off could be
    // revisited, but flipping it showed no benefit on the FRC
    // free-boundary hold (2026-08-04).

    void InitializeFluidState ();
    void FinishStateUpdate (amrex::Real end_time);
    void FillFluidSources (const WarpXSolverVec& state);
    void ApplyFluidDomainBoundaries (amrex::MultiFab& density, amrex::MultiFab& momentum,
                                     amrex::MultiFab& electron_energy,
                                     amrex::MultiFab& ion_energy,
                                     amrex::MultiFab& ion_parallel_energy,
                                     amrex::MultiFab& ion_perp_energy) const;
    void ApplyNeumannZDomainGhosts (amrex::MultiFab& mf) const;
    void FillCellCenteredElectromagneticFields ();
    void ComputeFluidRHS (WarpXSolverVec& rhs, amrex::Real time) const;
    void ComputeFluidRHSFromFaceFluxes (WarpXSolverVec& rhs, amrex::Real time) const;
    void ComputeFaceFluxes ();
    void ComputeDirectionalFaceFluxes (amrex::MultiFab& face_flux_mf,
                                       int normal_direction);
    void AssembleOhmElectricField (amrex::Real time) const;
    void ApplyMagneticCCDomainGhosts (amrex::MultiFab& mf) const;

    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0; }

    [[nodiscard]] amrex::Real GetMHDThetaForPC () const override { return m_theta; }

    [[nodiscard]] amrex::Real
    GetMHDReferenceResistivityForPC (amrex::Real time) const override;

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
    // Vacuum cell-switch threshold for the ion fluid (0 disables): cells
    // below it are passive dust with frozen momentum (Holmstrom-style).
    amrex::Real m_vacuum_mass_density = 0.0;
    // Velocity relaxation rate [1/s] applied to vacuum dust (0 disables).
    amrex::Real m_vacuum_drag_rate = 0.0;
    // Non-periodic z (RZ only): the solver fills axial domain ghosts with
    // zero-gradient (Neumann) values, giving outflow ends instead of the
    // periodic seam. Set automatically when the z geometry is non-periodic.
    bool m_z_neumann = false;
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

    /** Split-field external-A bookkeeping: add (sign = +1) or subtract
     * (sign = -1) the stored hybrid external E/B fields to/from
     * Efield_fp/Bfield_fp. Between steps the WarpX fields hold total
     * fields; inside the nonlinear solve they hold the plasma response
     * only. Using the stored multifabs (rather than re-evaluating the
     * time function) makes each subtract/add cycle exact. */
    void AddExternalFieldsToTotals (amrex::Real sign) const;
};

#endif
