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
 * The implementation is restricted to a single Cartesian, periodic AMR level.
 * Fluid transport can use centered or conservative Rusanov face fluxes. The
 * current state omits ion total energy and a compatible magnetic-energy flux,
 * so the Rusanov option regularizes compression but is not yet a complete
 * conservative shock-MHD scheme.
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

    [[nodiscard]] bool
    UseNewtonLineSearch () const override
    {
        return true;
    }

    void InitializeFluidState ();
    void FillFluidSources (const WarpXSolverVec& state);
    void FillCellCenteredElectromagneticFields ();
    void ComputeFluidRHS (WarpXSolverVec& rhs, amrex::Real time) const;

    [[nodiscard]] amrex::Real GetThetaForPC () const override { return 0.0; }

    inline static constexpr char MassDensityName[] = "implicit_mhd_mass_density";
    inline static constexpr char MomentumDensityName[] = "implicit_mhd_momentum_density";
    inline static constexpr char ElectronEnergyName[] = "implicit_mhd_electron_energy";

  private:
    inline static constexpr char TotalCurrentCCName[] = "implicit_mhd_total_current_cc";
    inline static constexpr char MagneticFieldCCName[] = "implicit_mhd_magnetic_field_cc";

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
    amrex::Real m_positivity_safety = 0.99;
    std::string m_fluid_flux = "centered";
    bool m_evolve_ion_fluid = true;
    bool m_include_joule_heating = true;

    std::unique_ptr<amrex::Parser> m_mass_density_parser;
    std::unique_ptr<amrex::Parser> m_velocity_x_parser;
    std::unique_ptr<amrex::Parser> m_velocity_y_parser;
    std::unique_ptr<amrex::Parser> m_velocity_z_parser;
    std::unique_ptr<amrex::Parser> m_electron_pressure_parser;
    amrex::ParserExecutor<3> m_mass_density;
    amrex::ParserExecutor<3> m_velocity_x;
    amrex::ParserExecutor<3> m_velocity_y;
    amrex::ParserExecutor<3> m_velocity_z;
    amrex::ParserExecutor<3> m_electron_pressure;

    void SaveMagneticField ();
    void UpdateWarpXFields (const WarpXSolverVec& state, amrex::Real start_time);
    void FinishStateUpdate (amrex::Real end_time);
};

#endif
