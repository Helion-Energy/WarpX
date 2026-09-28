/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_DARWIN_VACUUM_JOINT_SOLVE_H_
#define WARPX_DARWIN_VACUUM_JOINT_SOLVE_H_
#include "ThermalStageSolver.H"
#include "DarwinVacuumHodgePC.H"
#include "DarwinLongitudinalSchur.H"
#include "NativeInertiaSupport.H"
#include "NativeVacuumParticleSupport.H"
#include <cstdint>
#include <AMReX_GpuContainers.H>
#include <chrono>
#include <map>
#include <memory>
class WarpX;
class ThetaImplicitHybrid;
class MassMatrixDensityProjection;
class NativeOwnedEndpointAmpere;
class NativeEndpointAmpereOrigin;
namespace warpx::particles { class NativeEndpointCurrentResponse; }
namespace warpx::thermal {
class DarwinThermalAdvance;
class ThermalMassMatrixResponse;
class NativeJointCurrentArithmetic;
/** Attempt-owned joint E/phi/U equations. The verified-stage lease, including
 * particle receipt and actual push field, survives solve return. It is never
 * an accepted state, endpoint publisher, or ambient option-based bypass. */
class DarwinVacuumJointSolve final : public ThermalNonlinearOperator {
public:
    using MF=amrex::MultiFab;
    using Real=amrex::Real;
    using Vec=WarpXSolverVec;
    using Field=amrex::Array<MF,3>;
    using View=ablastr::fields::VectorField;
    explicit DarwinVacuumJointSolve(DarwinThermalAdvance&);
    ~DarwinVacuumJointSolve() override;
    int Solve(int step);
    bool Active() const noexcept { return m_active; }
    void Invalidate() noexcept;
    bool PublishVerifiedStage();
    // Same verified midpoint after publication, before the native finisher.
    bool CapturePublishedEndpointStage();
    bool FinishParticles(Real endpoint_time);
    bool CheckFinishedSupport();
    bool PrepareEndpointAmpere(Real endpoint_time);
    bool VerifyEndpointAmpere(Real endpoint_time);
    std::shared_ptr<NativeEndpointAmpereOrigin const> RetainEndpointAmpereOrigin(Real endpoint_time);
    struct AxialProducerLease {
        DarwinVacuumJointSolve const* owner=nullptr;
        std::shared_ptr<unsigned char const> lifetime;
        std::uint64_t generation=0;
        bool valid=false,enabled=false;
    };
    // Current typed borrower only; local lifetime identities are never
    // compared across ranks. Strong token ownership prevents address ABA.
    static AxialProducerLease CaptureAxialProducerLease(WarpX&);
    static bool AxialProducerLeaseCurrentLocal(WarpX&,AxialProducerLease const&);
    bool AllowsAxialReflection() const noexcept;
    bool AxialReflectionPrepared() const noexcept;
    bool AxialReflectionFinished() const noexcept;
    // Actual retained owner only; these CUDA-enclosing methods remain public.
    bool PrepareAxialReflection(amrex::Long reflected);
    bool CompleteAxialReflection();
    // Called only by the shared native map under the typed borrowed context.
    void CaptureTransverse();
    void CapturePushField();
    bool Residual(Vec&,Vec const&,int,bool) override;
    Real StepBound(Vec const&,Vec const&) const override;
    bool Freeze(Vec const&,int,bool,ThermalSolveResult&) override;
    bool Precondition(Vec&,Vec const&) override;
    void RestoreInput(Vec const&) override;
    bool PhysicalConverged(Vec const&) const override;
    bool BeginInputSolve(Vec const&) override;
    bool FormInput(Vec&,Vec const&,Real,Vec const&,bool) override;
    bool AcceptInput(Vec const&) override;
    void EndInputSolve(bool) noexcept override;
    bool CurrentArithmeticActive() const noexcept {return m_active&&bool(m_arithmetic);}
    void BuildArithmeticFields(Real,amrex::IntVect const&);
    void PublishArithmeticCurrent();
    void SubtractArithmeticDisplacement(Real,View const*);
    void AssembleArithmeticElectric();
    void CaptureArithmeticOhm();
    bool HasResidualRemainder() const noexcept override;
    bool CopyResidualRemainder(Vec&) const override;
    bool DifferenceResidualRemainder(Vec&,Vec const&,Vec const&,Vec const&,Vec const&,Real) const override;
    bool CompleteResidualRemainder(Vec const&);
    bool CompleteMassMatrixThermalIncrement(Vec const&,Vec const&,bool probe);
    // Public CUDA-enclosing methods; private owner supplies all arguments.
    bool Prepare();
    bool CheckSupport(bool prepare,bool finished=false);
    bool CheckParticleSupport();
    void CanonicalScalar(MF&);
    void Gradient(View const&,MF const&);
    void Materialize(Vec const&);
    bool Project(Vec&);
    bool Transform(MF&,View const&,MF const&,MF const&);
    bool RawAccepted() const;
    void CaptureReceipt(bool include_fields=true);
    void VisitReceipt(std::function<void(void const*,std::size_t)> const&,std::vector<std::size_t>&) const;
    bool CheckReceipt() const;
private:
    static int AxialProducerState(WarpX&,DarwinVacuumJointSolve const*&);
    std::shared_ptr<unsigned char const> m_axial_lifetime=std::make_shared<unsigned char const>(0);
    void InvalidateConvergenceReceipt() noexcept;
    void InvalidateAxialReflection() noexcept;
    bool MassMatrixScopeSupported() const;
    bool MassMatrixBaseMatches(Vec const&) const;
    bool MassMatrixCacheReady() const;
    bool MassMatrixLeaseReady() const;
    bool MassMatrixThermalCompanionReady() const;
    void InvalidateMassMatrixLease() noexcept;
    friend class NativeStoppingCarryCertificate;
    static constexpr char const* U="hybrid_electron_energy_fp";
    static constexpr char const* Phi="hybrid_phi_darwin_fp";
    static View V(Field& f) { return {&f[0],&f[1],&f[2]}; }
    DarwinThermalAdvance& a;
    ThetaImplicitHybrid& s;
    WarpX& w;
    bool m_active=false,m_verified=false,m_support_ready=false;
    std::unique_ptr<NativeOwnedEndpointAmpere> m_endpoint_ampere;
    std::unique_ptr<NativeJointCurrentArithmetic> m_arithmetic;
    std::uint64_t m_endpoint_ampere_generation=0;
    bool EndpointAmpereOwnerCurrent(Real endpoint_time) const;
    enum class ParticlePhase { Inactive, Trial, Published, Finishing, Finished };
    ParticlePhase m_particle_phase=ParticlePhase::Inactive;
    std::uint64_t m_generation=0,m_verified_generation=0;
    // Local input identity and current FULL map generation bind this query.
    // Jv, PC, materialization and invalidation revoke it. Only a successful
    // nonprobe Residual produces a receipt, before any input copy/mutation.
    Vec const* m_convergence_input=nullptr;
    std::uint64_t m_convergence_generation=0;
    int m_convergence_part=-1;
    // Full particle bases have their own monotone generation. A signed MM
    // sample advances m_generation, but cannot become a new native base.
    bool m_mm_requested=false,m_mm_full_base_current=false,m_mm_linear_ready=false;
    std::uint64_t m_mm_base_generation=0,m_mm_lease_base_generation=0,m_mm_endpoint_epoch=0;
    warpx::particles::NativeEndpointCurrentResponse const* m_mm_response=nullptr;
    MassMatrixDensityProjection const* m_mm_density=nullptr;
    int m_mm_interval_width=0;
    Vec m_mm_base;
    bool m_mm_thermal_requested=false;
    std::unique_ptr<ThermalMassMatrixResponse> m_mm_thermal;
    std::uint64_t m_mm_thermal_density_epoch=0,m_mm_thermal_publication=0;
    // The ordinary PLUS/MINUS vectors own their bytes. These receipts bind the
    // copy destinations to the native base/sample that produced each copy.
    mutable std::map<Vec const*,std::array<std::uint64_t,3>> m_mm_thermal_copies;
    int m_part=0; // 0 coupled; 1 E/phi at held U. Scalar thermal uses native stage.
    int m_maps=0,m_step=0;
    Real m_h=0,m_alpha=0,m_b=0,m_length=0,m_field_target=0,m_energy_target=0,m_h_target=0;
    Real m_raw_e=0,m_raw_u=0,m_raw_h=0,m_inverse_error=0;
    std::chrono::steady_clock::time_point m_started;
    Vec m_state,m_residual,m_raw,m_last,m_pc_rhs,m_pc_out;
    Vec m_residual_remainder;
    std::uint64_t m_remainder_generation=0;
    std::unique_ptr<Field> m_current_remainder;
    std::array<std::unique_ptr<MF>,AMREX_SPACEDIM> m_face_remainder,m_velocity_remainder,m_advection_remainder;
    std::unique_ptr<MF> m_correction_remainder;
    MF m_origin_phi,m_phi,m_held_phi,m_held_phi_low,m_phi_a,m_phi_w,m_scalar_work;
    MF m_raw_u_field,m_correction,m_U_images,m_frozen_U,m_frozen_n,m_support_rho;
    Field m_origin_EL,m_delta,m_transverse,m_push,m_held_EL,m_held_low,m_saved_E;
    Field m_current,m_pc_current,m_pc_d,m_pc_h,m_pc_e,m_pc_k,m_kappa,m_zero;
    std::unique_ptr<warpx::darwin::NativeVacuumParticleSupport> m_particle_support;
    Field m_verified_A,m_verified_B,m_verified_EL,m_verified_Et,m_verified_push;
    amrex::Array<amrex::iMultiFab,3> m_P,m_V;
    std::array<MF,AMREX_SPACEDIM> m_face,m_velocity,m_advection;
    std::unique_ptr<warpx::darwin::NativeInertiaSupport> m_support;
    std::unique_ptr<warpx::darwin::DarwinVacuumHodgePC> m_hodge;
    std::unique_ptr<warpx::darwin::DarwinLongitudinalSchur> m_schur;
    struct Bytes { void const* pointer=nullptr; std::size_t size=0; amrex::Gpu::DeviceVector<unsigned char> saved; };
    std::vector<Bytes> m_particles;
    // Private expected full receipt, derived before publication from the
    // verified native midpoint. Only six position/momentum arrays may change.
    std::vector<Bytes> m_axial_particles;
    std::vector<void const*> m_axial_changed;
    bool m_axial_scope=false,m_axial_prepared=false,m_axial_finished=false;
    std::uint64_t m_axial_generation=0;
    amrex::Long m_axial_count=0;
    std::vector<std::size_t> m_particle_schema;
    std::vector<MF*> m_receipt_fields;
    std::string m_host_rng;
    long m_next_id=0;
    Real m_time=0,m_dt=0;
    int m_physical_step=0;
    long m_pc_calls=0,m_auxiliary_iterations=0,m_saddle_iterations=0;
};
} // namespace warpx::thermal
#endif
