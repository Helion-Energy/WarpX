/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_CARRY_CERTIFICATE_H_
#define WARPX_NATIVE_STOPPING_CARRY_CERTIFICATE_H_
#include <AMReX_REAL.H>
#include <AMReX_INT.H>
#include <AMReX_iMultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <cstdint>
#include <memory>
class WarpX;
namespace warpx::implicit { class NativeAcceptedStepCandidate; }
namespace warpx::thermal {
class DarwinVacuumJointSolve;
class DarwinThermalAdvance;
class NativeStoppingMaterialOwner;
class NativeStoppingMaterialSupport;
// Explicit factory opt-in before the field producer; no runtime flag or late
// reconstruction can supply a missing root/stage/rotation history.
enum class NativeStoppingCarryMode { VacuumResidualOnly, FieldTransition, AcceptedOrigin };
struct NativeStoppingCarryRequest final {
    NativeStoppingCarryMode mode=NativeStoppingCarryMode::VacuumResidualOnly;
};
enum class StoppingCarryStatus { Unavailable, Root, Stage, Rotated, Ready,
    Stale, DeclinedTransition, DeclinedIdentity, Unsupported, TransitionReady, AcceptedReady };
/** F's density-dependent kinetic metric account; never an S heat source.
 * Native phase metrics and currents are measured before source mutation.
 * The last two terms retain represented rotation and summation remainders. */
struct StoppingMovingMetricReport {
    bool available=false;
    amrex::Long rows=0;
    amrex::Real initial_kinetic=0., final_kinetic=0., endpoint_mean_work=0.;
    amrex::Real stage_work=0., metric_work=0., transition_metric_work=0.;
    amrex::Real rotation_work=0., identity_remainder=0., absolute_scale=0.;
    amrex::Real arithmetic_bound=0.;
};
// A finite C-I-Je-D defect can survive spatial endpoint A recovery. This
// receipt attests its field history; it is not an Ampere-closure tolerance,
// a free-vacuum residual bound or a heat reservoir.
struct StoppingAmpereHistoryReport {
    bool available=false;
    amrex::Long rows=0;
    amrex::Real initial_defect=0.,endpoint_defect=0.,curl_centering=0.;
    amrex::Real stage_quadrature=0.,current_rotation=0.,displacement_rotation=0.;
    amrex::Real identity_error=0.,identity_bound=0.,physical_scale=0.;
};
struct StoppingCarryReport {
    StoppingCarryStatus status=StoppingCarryStatus::Unavailable;
    std::uint64_t root_generation=0;
    amrex::Long norm_rows=0,free_rows=0,fixed_rows=0,axis_rows=0,transition_rows=0;
    amrex::Real native_norm=0.,native_target=0.,array_scale=0.,electric_scale=0.,alpha=0.;
    amrex::Real norm_arithmetic_bound=0.,rotation_coefficient=0.;
    NativeStoppingCarryMode mode=NativeStoppingCarryMode::VacuumResidualOnly;
    StoppingMovingMetricReport moving_metric;
    StoppingAmpereHistoryReport ampere_history;
    // All signed work is separate from thermal heat and the Ampere residual.
    amrex::Real free_work=0.,fixed_work=0.,root_work=0.,history_work=0.,
        represented_work_remainder=0.,absolute_work=0.,dual_norm=0.,root_dual_bound=0.;
    // Finite T=P_F intersect V_S has field-history provenance, not the V norm
    // certificate. Never include this work in root_dual_bound or thermal heat.
    amrex::Real transition_work=0.,transition_absolute_work=0.;
    // S1 carries an actual accepted current, with no F root/rotation claim.
    amrex::Real accepted_work=0.,accepted_absolute_work=0.;
    std::uint64_t accepted_generation=0;
};
struct NativeStoppingTransitionView {
    bool ready=false;
    ablastr::fields::ConstVectorField old_current{}, stage_current{}, endpoint_current{};
    ablastr::fields::ConstVectorField old_mass{}, stage_mass{}, endpoint_mass{}, root_residual{};
    std::array<amrex::iMultiFab const*,3> field_physical{}, field_vacuum{}, transition{};
    std::array<amrex::MultiFab const*,3> phase_density{}, phase_mask_density{};
    // Actual native curl(B)/mu0, instantaneous Ji and D from owned phases.
    // No stored W exists on this joined-vacuum path.
    std::array<ablastr::fields::ConstVectorField,3> phase_curl_current{},phase_ion{},phase_displacement{};
};
struct NativeStoppingSourceOriginView {
    bool ready=false,field_transition=false;
    ablastr::fields::ConstVectorField current{},ion{},electron{},displacement{},mass{};
};
class NativeStoppingCarryCertificate final {
public:
    using CV=ablastr::fields::ConstVectorField;
    ~NativeStoppingCarryCertificate();
    NativeStoppingCarryCertificate(NativeStoppingCarryCertificate const&)=delete;
    NativeStoppingCarryCertificate& operator=(NativeStoppingCarryCertificate const&)=delete;
    StoppingCarryReport const& Report() const noexcept;
    void Invalidate() noexcept;
    // Collective; all ranks call, including a rank whose local state is stale.
    bool ValidateSource(WarpX const&,amrex::Real,std::uint64_t) const;
    bool ValidateTrial(CV const&) const;
    bool MeasureWork(CV const&,StoppingCarryReport&) const;
    CV MaterialCurrent() const;
    CV CarriedCurrent() const;
    // Historical device views; a consumer must collectively ValidateSource
    // before every action. No borrowed view grants source publication.
    NativeStoppingTransitionView TransitionView() const;
    NativeStoppingSourceOriginView SourceOrigin() const;
    // CUDA-enclosing methods are public, but construction and mutable ownership
    // stay producer-only. Consumers only receive shared_ptr<const ...>.
    bool CaptureFieldOrigin();
    bool ArmAtVerifiedRoot(DarwinVacuumJointSolve const&);
    bool BindCapturedStage();
    bool BindRotatedEndpoint(amrex::Real theta);
    bool BindSourceEndpoint(amrex::Real,std::uint64_t);
    // CUDA requires an accessible enclosing type for extended device lambdas.
    // Only the opaque name is public; construction and storage remain private.
    struct Impl;
private:
    static std::shared_ptr<NativeStoppingCarryCertificate const> CaptureAcceptedOrigin(
        WarpX&,amrex::Real,std::uint64_t,std::uint64_t);
    explicit NativeStoppingCarryCertificate(WarpX&,NativeStoppingCarryMode);
    std::unique_ptr<Impl> m_impl;
    friend class warpx::implicit::NativeAcceptedStepCandidate;
    friend class DarwinThermalAdvance;
    friend class DarwinVacuumJointSolve;
};
}
#endif
