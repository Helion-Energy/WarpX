#ifndef WARPX_NATIVE_ENDPOINT_FIELD_H_
#define WARPX_NATIVE_ENDPOINT_FIELD_H_
#include <AMReX_REAL.H>
#include <AMReX_MultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <functional>
#include <string>
class WarpX;
class ThetaImplicitHybrid;
namespace warpx::implicit { class NativeAcceptedStepCandidate; }
namespace warpx::darwin { class NativeCoilField; }
#include <optional>
#include <memory>
#include <AMReX_iMultiFab.H>
#include <cstdint>
namespace warpx::particles { class NativeEndpointCurrentResponse; }
bool NativeEndpointEnabled();
// Fixed SI current floor; source defaults follow the source nonlinear scaling.
amrex::Real NativeEndpointCurrentAbsoluteTolerance(WarpX&);
bool NativeIonQuadratureEnabled();
bool NativeCorrelatedIncrementEnabled();
// Captured CPU-RZ accepted-pair/MM qualification. Pure local metadata only;
// callers must collectively validate before branching or mutating state.
// This is not a current tensor/publication lease or a source capability.
ThetaImplicitHybrid const* NativeEndpointPairMassMatrixOwnerLocal(WarpX&) noexcept;
// Collective, default-off adjacent-cell current policy. All ranks must call,
// including unsupported/local-false ranks. Invalid or differing selectors
// return false with enabled=false, before any field/particle mutation.
// direct_deposit=false requires the separately captured paired-MM owner;
// the response/publication lease is validated by its actual consumer.
bool TryNativeAdjacentInstantaneousPolicy(WarpX&,bool& enabled,
                                         bool direct_deposit=true);
// Guarded native full-Ohm projection experiment; checkpoint-bound, default off.
bool NativeFullOhmLongitudinalEnabled();
// Versioned accepted-state support; trial companions remain rebuildable scratch.
bool NativeEndpointRestartSupported();
void AllocateNativeEndpointStorage(WarpX&);
void ValidateNativeEndpointRestart(WarpX&);
// Validate saved descriptors before VisMF reads into allocated fields: the
// reader can otherwise silently retain ghosts absent from the saved file.
// Invoke the returned finite/allocation check immediately after registry read.
std::function<void()> NativeEndpointRestartStorageValidator(
    WarpX&,std::string const& checkpoint_prefix);
void ValidateNativeEndpointCapabilities(WarpX&);
// The bounded Joule-only exception uses one positive species and a constant
// global eta. It does not supply per-species or temperature-dependent tangents.
bool NativeConstantEtaMassMatrixSupported(WarpX&);
// The approximate-MM envelope excludes temperature-dependent push sources.
void ValidateNativeEndpointMassMatrixCapabilities(WarpX&);
std::string NativeEndpointCheckpointConfiguration(WarpX&);
bool NativePrescribedDriveEnabled();
// Accepted instantaneous circuit rate is distinct from active interval secants.
bool NativeCircuitDriveEnabled();
// Runs after CircuitCoupling::InitData restores its closed native state.
// Checks immutable binding and prepares a fresh accepted rate at the restored
// time; it never solves, reprojects or reseeds accepted fields/companions.
void ValidateNativeCircuitEndpointRestart(WarpX&,amrex::Real time);
// t0 and h identify the physical interval; a static control ignores them.
void PrepareNativeCurlRate(WarpX&,amrex::Real t0=0.,amrex::Real h=0.,
    warpx::darwin::NativeCoilField* coil=nullptr);
// Pure native instantaneous curl response; affine=false is the homogeneous
// linear operator, affine=true includes the known prescribed boundary rate.
void NativeInstantaneousCurlRate(WarpX&,amrex::Real time,bool affine);
// The finite-dt outer correction reuses the physical coefficient builder;
// interval=theta*dt, output kappa=epsilon0*alpha_node/interval^2.
bool FillNativeYeeSchurKappa(WarpX&,amrex::MultiFab const& rho,
                            amrex::MultiFab& kappa,amrex::Real interval);
// false returns no borrowed view and marks the residual invalid; never falls
// back to an interval-current derivative. Scratch is owned by the registry.
bool FreezeNativeInertiaResponse(WarpX&,warpx::particles::NativeEndpointCurrentResponse&,
                                amrex::Real dt,int interval_width);
bool PrepareNativeInertiaStage(WarpX&,amrex::Real dt,amrex::MultiFab const*& current,
                               amrex::MultiFab const*& rate,
                               warpx::particles::NativeEndpointCurrentResponse* response=nullptr,
    warpx::darwin::NativeCoilField* coil=nullptr,
    std::optional<amrex::Real> stage_time=std::nullopt);
// Diagnostic nodal mirror only; native Ohm/EL consume the Yee field directly.
void PublishNativeYeeInertiaMirror(WarpX&);
// Reversible owner path: false is collective, but scratch may have changed.
// The caller must restore its complete step transaction on failure.
bool TryCaptureNativeEndpointStage(WarpX&,amrex::Real time,amrex::Real dt,
    warpx::darwin::NativeCoilField* coil=nullptr);
void CaptureNativeEndpointStage(WarpX&,amrex::Real time,amrex::Real dt,
    warpx::darwin::NativeCoilField* coil=nullptr);
void RotateNativeEndpointCurrent(WarpX&,amrex::Real theta);
// Immutable, owned export from an actually verified F endpoint. This is
// provenance, not permission to reuse F's residual after a source mutation.
struct NativeEndpointAmpereCheck {
    bool valid=false;
    amrex::Long rows=0;
    amrex::Real maximum_error=0.,maximum_bound=0.,maximum_ratio=0.;
    amrex::Real absolute_tolerance=0.;
};
struct NativeEndpointMagneticWork {
    bool valid=false;
    amrex::Long rows=0;
    amrex::Real initial=0.,final=0.,change=0.,curl_work=0.,curl_transfer=0.;
    amrex::Real represented_faraday_work=0.,represented_faraday_bound=0.;
    amrex::Real faraday_error=0.,faraday_bound=0.,identity_error=0.,arithmetic_bound=0.;
};
struct NativeEndpointAmpereOriginView {
    bool available=false;
    std::uint64_t generation=0,epoch=0;
    amrex::Real time=0.;
    ablastr::fields::ConstVectorField current{},ion{},electron{},displacement{},bound{};
    std::array<amrex::iMultiFab const*,3> free_rows{};
};
class NativeEndpointAmpereOrigin final {
public:
    ~NativeEndpointAmpereOrigin();
    NativeEndpointAmpereOrigin(NativeEndpointAmpereOrigin const&)=delete;
    NativeEndpointAmpereOrigin& operator=(NativeEndpointAmpereOrigin const&)=delete;
    NativeEndpointAmpereOriginView View() const;
    bool Matches(WarpX&,amrex::Real endpoint_time,std::uint64_t epoch) const;
    // Fresh actual split-C and endpoint Ji/Je/D; exact next-F entry predicate,
    // with a fixed physical absolute floor plus the gamma(128) operand bound.
    NativeEndpointAmpereCheck CheckCurrent(WarpX&) const;
    // Pure predictor on caller-owned completed candidate images/current. This
    // grants no endpoint lease; publication still requires CheckCurrent.
    NativeEndpointAmpereCheck CheckCandidateCurrent(WarpX&,
        ablastr::fields::VectorField const& potential,
        ablastr::fields::ConstVectorField const& ion,
        ablastr::fields::ConstVectorField const& electron,
        ablastr::fields::ConstVectorField const& displacement) const;
    // Total B at macrostep entry comes from the original rollback owner, not
    // the potentially response-only B_old. No output aliases are accepted.
    NativeEndpointMagneticWork FieldMagneticWork(WarpX&,
        ablastr::fields::ConstVectorField const& original_total_B) const;
    struct Impl; // Public CUDA-enclosing name; all ownership remains private.
private:
    // S1 has an accepted state, not a completed F producer. The same free-row
    // predicate is checked freshly; no field-stage history is fabricated.
    static std::shared_ptr<NativeEndpointAmpereOrigin const> CaptureAccepted(
        WarpX&,amrex::Real,std::uint64_t,std::uint64_t);
    explicit NativeEndpointAmpereOrigin(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> m_impl;
    friend class NativeOwnedEndpointAmpere;
    friend class warpx::implicit::NativeAcceptedStepCandidate;
};
// One attempt's physical current certificate. The actual finished joint owner
// supplies its fixed P/V rows; this object grants no ambient recovery bypass.
// Capture/Verify write private scratch only and return collective failure.
class NativeOwnedEndpointAmpere {
public:
    using Masks=std::array<amrex::iMultiFab const*,3>;
    explicit NativeOwnedEndpointAmpere(WarpX&);
    ~NativeOwnedEndpointAmpere();
    bool Capture(amrex::Real dt,Masks const& P,Masks const& V);
    bool Verify(Masks const& P,Masks const& V);
    // Called only by the live finished Joint owner. Rechecks the original F
    // verification before copying; it does not change the field equations.
    std::shared_ptr<NativeEndpointAmpereOrigin const> RetainVerifiedOrigin(
        amrex::Real endpoint_time,std::uint64_t epoch,std::uint64_t generation,
        Masks const& P,Masks const& V);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
enum class NativeEndpointFailure {
    None, UnsupportedScope, InvalidDensity, Projection, LinearSolve,
    PhysicalConstraint, Companion, CircuitRate
};
struct NativeEndpointResult {
    NativeEndpointFailure failure = NativeEndpointFailure::None;
    const char* reason = "accepted";
    amrex::Real linear_residual = 0., linear_bound = 0.;
    amrex::Real ohm_error = 0., longitudinal_error = 0., companion_error = 0.;
    explicit operator bool() const { return failure == NativeEndpointFailure::None; }
};
// Positive-density accepted or explicitly retained native circuit candidate.
// Numerical projection/linear/physical
// failures restore every registered field and return; programmer/layout and
// existing unsupported-capability failures remain fatal. No particle/time/RNG
// publication occurs. The caller still owns complete step rollback.
// Optional output receives the native completed transverse root only after
// every final gate succeeds. It must be private, correctly staggered storage
// that aliases neither a registered field nor another output component.
// Rejection leaves it unchanged; null preserves the original arithmetic path.
enum class NativeEndpointContext { RetainedFieldEnd, PreSource, PostSource };
NativeEndpointResult TryConstrainNativeEndpointField(
    WarpX&, amrex::Real time, std::function<void()> const& pure_curl_rate,
    ablastr::fields::VectorField const* exact_transverse=nullptr,
    NativeEndpointContext context=NativeEndpointContext::RetainedFieldEnd);
void ConstrainNativeEndpointField(WarpX&,amrex::Real time,bool initial,std::function<void()>const& pure_curl_rate,
    ablastr::fields::VectorField const* exact_transverse=nullptr);
#endif
