/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_ACCEPTED_STEP_CANDIDATE_H_
#define WARPX_NATIVE_ACCEPTED_STEP_CANDIDATE_H_
#include <AMReX_REAL.H>
#include <cstdint>
#include <memory>
#include <string>
class WarpX;
class ThetaImplicitHybrid;
class WarpXSolverVec;
namespace warpx::darwin { class NativeLongitudinalProducerCertificate; }
namespace warpx::thermal {
struct NativeStoppingThermalOptions;
struct NativeStoppingOwnerOptions;
struct NativeStoppingSourceOptions;
struct NativeStoppingSourceView;
struct NativeStoppingSourcePublication;
struct NativeStoppingOwnerTrial;
struct NativeStoppingOwnerResult;
struct NativeStoppingOwnerView;
class NativeStoppingOwnerLease;
struct NativeStoppingCarryRequest;
class NativeStoppingCarryCertificate;
}
namespace warpx::implicit {
enum class NativeCandidateStatus { Success, Declined, Unsupported, Terminal };
struct NativeCandidateStoppingOptions;
struct NativeSymmetricStoppingOptions;
struct NativeCandidateStoppingResult;

class NativeEndpointCandidateLease {
public:
    NativeEndpointCandidateLease() noexcept = default;
    // Local lifetime/generation check only; use the owner's collective
    // ValidateLease before a collective consumer or physical mutation.
    bool Valid() const noexcept;
    amrex::Real EndpointTime() const noexcept;
    std::uint64_t EndpointEpoch() const noexcept;
private:
    struct Token;
    std::shared_ptr<Token const> m_token;
    std::uint64_t m_generation = 0;
    friend class NativeAcceptedStepCandidate;
};

class NativeAcceptedStepCandidate {
public:
    NativeAcceptedStepCandidate(NativeAcceptedStepCandidate&&) noexcept;
    NativeAcceptedStepCandidate& operator=(NativeAcceptedStepCandidate&&) = delete;
    NativeAcceptedStepCandidate(NativeAcceptedStepCandidate const&) = delete;
    NativeAcceptedStepCandidate& operator=(NativeAcceptedStepCandidate const&) = delete;
    ~NativeAcceptedStepCandidate(); // Never performs collective rollback.
    bool Active() const noexcept;
    NativeEndpointCandidateLease EndpointLease() const;
    // All ranks must call each method below, including a rank with local false.
    // No short-circuiting these calls in a compound logical expression.
    bool ValidateLease(NativeEndpointCandidateLease const&) const;
    // Collective immutable producer view, bound to this exact live lease.
    // Null when disabled, stale, unsupported or missing any native producer.
    std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate const>
        StoppingProducerCertificate(NativeEndpointCandidateLease const&) const;
    std::shared_ptr<warpx::thermal::NativeStoppingCarryCertificate const>
        StoppingCarryCertificate(NativeEndpointCandidateLease const&) const;
    // Pure material prerequisite only; no accepted source publication.
    // Every method is collective, including local invalid leases. Failures
    // leave the whole-step candidate active; resolve it with Cancel/Finalize.
    warpx::thermal::NativeStoppingOwnerResult CaptureStoppingMaterial(
        NativeEndpointCandidateLease const&, warpx::thermal::NativeStoppingOwnerOptions const&);
    warpx::thermal::NativeStoppingOwnerResult EvaluateStoppingMaterial(
        warpx::thermal::NativeStoppingOwnerLease const&, warpx::thermal::NativeStoppingOwnerTrial const&);
    warpx::thermal::NativeStoppingOwnerView StoppingMaterialView(
        warpx::thermal::NativeStoppingOwnerLease const&);
    // Additive producer-bound T source. Numerical actions remain pure. Source
    // publication and optional acceptance use separate collective owner methods;
    // capture alone does not grant an accepted endpoint.
    bool CaptureStoppingMaterialSource(NativeEndpointCandidateLease const&,
        warpx::thermal::NativeStoppingSourceOptions const&);
    bool DefineStoppingMaterialSourceState(NativeEndpointCandidateLease const&,WarpXSolverVec&) const;
    bool StoppingMaterialSourceResidual(NativeEndpointCandidateLease const&,WarpXSolverVec&,WarpXSolverVec const&);
    bool SolveStoppingMaterialSource(NativeEndpointCandidateLease const&);
    warpx::thermal::NativeStoppingSourceView StoppingMaterialSourceView(NativeEndpointCandidateLease const&);
    // Physical source mutation and fresh CHIEF paired reclosure. The default
    // result remains provisional. Explicit acceptance also requires the fresh
    // original next-F current predicate and the actual F/source work receipts.
    warpx::thermal::NativeStoppingSourcePublication PublishStoppingMaterialSource(NativeEndpointCandidateLease const&);
    // Fresh value receipt only; stale leases return no approval and do not
    // revoke a different current lease. Finalize repeats the complete check.
    warpx::thermal::NativeStoppingSourcePublication StoppingMaterialAcceptance(NativeEndpointCandidateLease const&) const;
    bool CanCancel() const;
    NativeCandidateStatus Cancel();
    bool CanFinalize() const;
    NativeCandidateStatus Finalize();
    // Invalidates every outstanding lease before an explicitly authorized
    // event mutation. This alone neither mutates fields nor admits an event.
    // CanFinalize remains false until a future qualified reclosure republishes
    // a lease. Initial event-off implementation tests invalidation + Cancel.
    NativeCandidateStatus InvalidateEndpointLease();
    // Accepted-only fixed-density stopping; see NativeCandidateStopping.H.
    // Success reissues a physically validated lease. A numerical decline
    // cancels the whole step; Unsupported leaves the candidate unchanged.
    NativeCandidateStoppingResult ApplyStopping(NativeEndpointCandidateLease const&,
                                                NativeCandidateStoppingOptions const&);
private:
    enum class SourcePhase { FixedTemperaturePost, MidpointPre, MidpointPost };
    NativeCandidateStoppingResult ApplySource(SourcePhase,
        NativeEndpointCandidateLease const*, NativeCandidateStoppingOptions const&,
        warpx::thermal::NativeStoppingThermalOptions const*);
    NativeCandidateStatus ApplyMaterialHalf(bool pre,NativeSymmetricStoppingOptions const&);
    bool CheckMaterialSymmetricWork() const;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    explicit NativeAcceptedStepCandidate(std::unique_ptr<Impl>);
    friend class ::ThetaImplicitHybrid;
};

struct NativeStepCandidateResult {
    NativeCandidateStatus status = NativeCandidateStatus::Unsupported;
    int implicit_exit_status = 0;
    std::string reason;
    std::unique_ptr<NativeAcceptedStepCandidate> candidate;
};
// Explicit deferred path; ordinary ImplicitSolver::OneStep still calls the same
// private implementation with immediate finalization. No base virtual change.
NativeStepCandidateResult AdvanceNativeStepCandidate(
    WarpX&, amrex::Real start_time, amrex::Real dt, int step,
    NativeSymmetricStoppingOptions const* symmetric_source=nullptr,
    warpx::thermal::NativeStoppingCarryRequest const* stopping_carry=nullptr);
}
#endif
