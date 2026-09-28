/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_STAGE_H_
#define WARPX_EULERIAN_THERMAL_STAGE_H_

#include "ThermalStageReceipt.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConductivity.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalTransport.H"
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace warpx::thermal {

// This bounded API implements one RZ or Cartesian 3D physical level without EB,
// with prescribed fields/moments/source. It does not allocate application state
// or select an advance. The two Darwin modes must eventually share this stage
// operator.
enum class BoundaryKind {
    Adiabatic,
    PrescribedFlux,
    Reservoir,
    Leg,
    LegUnsupported // legacy mutating node-map coupling remains unsupported
};
enum class WallCapForm { FreeStreaming, Sonic };

struct ThermalBoundary {
    BoundaryKind kind = BoundaryKind::Adiabatic;
    // PrescribedFlux: outward W/m^2 (positive removes heat). Reservoir: Te [eV]
    // on the physical face; its distance from the cell centre is dx/2.
    amrex::Real value = 0.0;
    // Leg: eliminated physical interface with a parallel-conducting leg.
    // Set value=0.5 eV and leg_length=6 m for the maintained stroke.
    amrex::Real leg_length = 6.0;
    amrex::Real flux_limit = 0.0; // wall's own hard cap fraction; 0 disables
    WallCapForm cap_form = WallCapForm::FreeStreaming;
    amrex::Real ion_mass = 0.0; // required explicitly for a sonic cap [kg]
    bool drain_only = false;    // Leg is always drain-only
    // Inflow must be supplied explicitly; negative rejects inward advection.
    amrex::Real inflow_temperature_ev = -1.0;
};

struct ThermalStageOptions {
    std::shared_ptr<ThermalStageLifetime> lifetime; // optional DTA-owned lifetime
    bool retain_current_remainder=false; // optional source-free transport arithmetic
    amrex::Real dt = 1.0;
    amrex::Real theta = 0.5;
    amrex::Real conduction_theta = 0.5;
    amrex::Real time = 0.0;
    amrex::Real gamma = 5.0 / 3.0;
    // Zero disables the positive floor; stage and endpoint temperatures must
    // still be strictly positive. No artificial floor is inserted.
    amrex::Real temperature_floor_ev = 1.e-6;
    amrex::Real magnetic_floor = 1.e-12; // strictly positive |B| floor [T]
    amrex::Real limiter_width = 0.01;
    amrex::Real free_streaming_fraction = 0.0; // zero disables the cap
    int order = 4;
    int cross_mode = 1; // corrected smooth SMART; 0 is centered control
    // Native physical conductivity laws: n [m^-3], Te [eV], t [s] -> W/(m K).
    // Parsed/owned here, evaluated from every trial U, never frozen in Jv.
    std::string kappa_parallel = "0";
    std::string kappa_perpendicular = "0";
    // Runtime model owns these parsers and their constants/caps. Borrow only
    // while that owner outlives the stage; fixture strings remain the default.
    bool use_compiled_conductivity = false;
    amrex::ParserExecutor<3> parallel_executor{}, perpendicular_executor{};
    // Retain the exact compiled Data allocation. AMReX rejects setConstant/
    // registerVariables after compile; define() creates a new allocation.
    std::shared_ptr<amrex::Parser const> parallel_owner,perpendicular_owner;
    ThermalConductivityOptions conductivity{};
    std::array<std::array<ThermalBoundary, 2>, AMREX_SPACEDIM> boundary{};
    mhd_transport::TransportParameters transport{};
};

// One physical velocity per coordinate-normal face, positive along +r/+z (RZ)
// or +x/+y/+z (3D). Optional dissipation speed is a caller-derived
// MHD-compatible alpha [m/s]. If absent, alpha=abs(u_e), the MHD flow-entropy
// choice at zero smooth width. All directions must be provided together;
// all-null disables transport.
struct ThermalFaceContext {
    std::array<const amrex::MultiFab*, AMREX_SPACEDIM> velocity{};
    std::array<const amrex::MultiFab*, AMREX_SPACEDIM> dissipation_speed{};
    // Optional conservative electron number flux = -J_e/q_e [m^-2 s^-1].
    // Takes precedence over velocity. This native kinetic path requires
    // reconstruction=none and entropy penalty=0 until their continuity/work
    // adaptation is separately qualified. u = F_n / arithmetic n_cc face,
    // the same denominator as its central U flux and the moments adapter.
    std::array<const amrex::MultiFab*, AMREX_SPACEDIM> number_flux{};
    // Optional companion to number_flux, consumed with the SAME Stage n_face.
    // Source-free, conduction-free central transport only. All or none.
    std::array<const amrex::MultiFab*, AMREX_SPACEDIM> number_flux_remainder{};
    // Optional CELL mask: exactly 0 (inactive) or 1 (active). An inactive
    // cell closes all incident conductive faces, including physical walls;
    // advection/source/capacity are unaffected. Null means all active.
    // Mapping native nodal activity to this FV mesh belongs to the caller;
    // averaging density then applying its floor is not an equivalent map.
    const amrex::MultiFab* conduction_active = nullptr;
};

class EulerianThermalStage {
  public:
    ~EulerianThermalStage();
    EulerianThermalStage(EulerianThermalStage const&)=delete;
    EulerianThermalStage& operator=(EulerianThermalStage const&)=delete;
    ThermalStageReceipt CompletedReceipt() const noexcept;
    EulerianThermalStage (const amrex::Geometry& geometry,
                          const amrex::MultiFab& old_energy,
                          const amrex::MultiFab& number_density,
                          const amrex::MultiFab& magnetic_field,
                          const amrex::MultiFab& source,
                          ThermalStageOptions options);
    EulerianThermalStage (const amrex::Geometry& geometry,
                          const amrex::MultiFab& old_energy,
                          const amrex::MultiFab& old_density,
                          const amrex::MultiFab& stage_density,
                          const amrex::MultiFab& endpoint_density,
                          const amrex::MultiFab& magnetic_field,
                          const amrex::MultiFab& source,
                          ThermalFaceContext faces,
                          ThermalStageOptions options);

    // Refresh only trial context, transactionally after validation. Old U/n
    // remain immutable. A coupled caller must refresh this on EVERY trial,
    // including base restoration after Jv; a segregated inner solve freezes it.
    bool RefreshContext (const amrex::MultiFab& stage_density,
                         const amrex::MultiFab& endpoint_density,
                         const amrex::MultiFab& magnetic_field,
                         const amrex::MultiFab& source,
                         ThermalFaceContext faces = {});
    using SourceEvaluator = std::function<bool(
        const amrex::MultiFab&, const amrex::MultiFab&, const amrex::MultiFab&,
        amrex::MultiFab&, amrex::MultiFab&)>;
    // Pure callback(U_s,n_s,B_s,Q,dQ/dU_local), Q in W/m^3 and derivative in
    // s^-1. REPLACES prescribed source; private outputs are zeroed each call.
    // A provider may leave the local derivative zero (explicit PC surrogate).
    // Captured dependencies must be trial-safe; no accepted writes/tallies.
    void
    SetSourceEvaluator (SourceEvaluator evaluator, bool verified_zero_provider=false) {
        InvalidateEvaluation();
        m_zero_source_provider=!evaluator || verified_zero_provider;
        m_source_evaluator = std::move(evaluator);
    }
    const amrex::MultiFab&
    SourceDerivative () const {
        return m_source_derivative;
    }
    const amrex::MultiFab&
    EvaluatedSource () const {
        return m_live_source;
    }
    const amrex::MultiFab&
    OldDensity () const {
        return m_old_density;
    }
    const amrex::MultiFab&
    EndpointDensity () const {
        return m_endpoint_density;
    }

    // U_s-U_n-theta dt [source-div(F_adv+q)-(gamma-1)U_s div(u_e)].
    // Deep copies of U_n, old/stage/endpoint n, B and prescribed source protect
    // caller/accepted data and ghosts during probes. Scratch and output only
    // are changed. false means inadmissible/nonfinite trial or conductivity; no
    // clipping/energy repair is performed.
    bool Residual (const amrex::MultiFab& stage, amrex::MultiFab& residual);
    bool HasResidualRemainder() const noexcept { return m_remainder_context; }
    bool CopyResidualRemainder(amrex::MultiFab& output) const;
    void InvalidateResidualRemainder() noexcept { m_remainder_ready=false; }
    // CUDA-enclosing helper. Ordinary residual and physical gates stay high-only.
    bool CompleteResidualRemainder(amrex::MultiFab const& represented);
    bool Admissible (const amrex::MultiFab& stage) const;
    // Upper bound in [0,1] for stage + alpha direction, using the true affine
    // endpoint. A floor-bound state can legitimately return zero.
    amrex::Real StepBound (const amrex::MultiFab& stage,
                           const amrex::MultiFab& direction) const;
    void Endpoint (const amrex::MultiFab& stage,
                   amrex::MultiFab& endpoint) const;

    const amrex::Geometry&
    Geometry () const {
        return m_geometry;
    }
    const amrex::MultiFab&
    Density () const {
        return m_density;
    }
    const amrex::MultiFab&
    ConductionActive () const {
        return m_conduction_active;
    }
    const amrex::MultiFab&
    OldEnergy () const {
        return m_old;
    }
    const ThermalStageOptions&
    Options () const {
        return m_options;
    }
    // Most recently evaluated conductive HEAT flux [W/m^2], excluding
    // advection, positive in coordinate direction, and n_face * chi_nn *
    // (theta_c/theta)/cap^2 [m^-1 s^-1]. The latter freezes coefficients, SMART
    // and cap temperature, following the MHD PC.
    const amrex::MultiFab&
    FaceFlux (int d) const {
        return *m_flux[d];
    }
    const amrex::MultiFab&
    FaceCoefficient (int d) const {
        return *m_coefficient[d];
    }

    const amrex::MultiFab&
    FaceAdvectionFlux (int d) const {
        return *m_advection[d];
    }
    const amrex::MultiFab&
    FaceVelocity (int d) const {
        return *m_velocity[d];
    }
    // Physical wall dq_out/d(U_s/n_s), including kappa and hard cap response.
    // These faces have FaceCoefficient=0: PC adds the local wall row once.
    const amrex::MultiFab&
    FaceWallDerivative (int d) const {
        return *m_wall_derivative[d];
    }

  private:
    friend class ThermalMassMatrixResponse;
    friend class ThermalFiniteFluxResponse;
    bool InvalidateEvaluation() noexcept;
    bool CompleteEvaluation() noexcept;
    std::shared_ptr<ThermalStageLifetime> m_lifetime;
    std::uint64_t m_instance=0;
    bool m_zero_source_provider=true;
    amrex::Geometry m_geometry;
    ThermalStageOptions m_options;
    amrex::MultiFab m_old, m_old_density, m_density, m_endpoint_density,
        m_magnetic, m_source, m_stage, m_kappa, m_live_source,
        m_source_derivative, m_conduction_active;
    SourceEvaluator m_source_evaluator;
    amrex::Parser m_parallel_parser, m_perpendicular_parser;
    amrex::ParserExecutor<3> m_parallel, m_perpendicular;
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_flux,
        m_coefficient, m_velocity, m_advection, m_speed, m_wall_derivative;
    bool m_transport = false;
    bool m_remainder_context=false,m_remainder_ready=false;
    std::array<std::unique_ptr<amrex::MultiFab>,AMREX_SPACEDIM> m_velocity_remainder,m_advection_remainder;
    std::unique_ptr<amrex::MultiFab> m_residual_remainder;
};

} // namespace warpx::thermal
#endif
