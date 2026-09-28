/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_DARWIN_VACUUM_AFFINE_RESPONSE_H_
#define WARPX_DARWIN_VACUUM_AFFINE_RESPONSE_H_
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <memory>
class WarpX;
namespace warpx::darwin
{
/** Frozen spatial recovery. No particle, density, accepted-history or field-register writes.
 * Geometry/native curl solver must remain fixed for this object's lifetime. The two baselines
 * are deliberately caller selected: accepted vacuum E_T, or incoming endpoint A.
 */
class DarwinVacuumAffineResponse
{
  public:
    using View = ablastr::fields::VectorField;
    using Mask = amrex::Array<amrex::iMultiFab const *, 3>;
    struct Options
    {
        amrex::Real relative_tolerance = 1.e-12;
        // Normalized native curl-curl units. Magnetic caller divides its raw atol by diagonal.
        amrex::Real absolute_tolerance = 1.e-10;
        int max_iterations = 2000;
        bool magnetic = false;
        bool flux_only = false;
        bool check_operator = false;
        amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo{}, pmc_hi{};
    };
    enum class Failure
    {
        None,
        NotFrozen,
        InvalidLayout,
        InvalidTrace,
        Nonfinite,
        Curvature,
        IterationLimit,
        UnsupportedRetained
    };
    struct Result
    {
        Failure failure = Failure::None;
        int iterations = 0;
        amrex::Real initial = 0., residual = 0., target = 0.;
        explicit operator bool() const noexcept { return failure == Failure::None; }
    };
    struct Impl;
    /** A private two-part solve is deliberately distinct from a publishable View.
     * There is no implicit bool conversion: ready does not authorize discarding low.
     */
    enum class AffineTraceInput : int { None, High, Pair };
    struct RetainedReceipt
    {
        Result solve;
        AffineTraceInput affine_trace=AffineTraceInput::None;
        bool homogeneous=false;
        bool ready = false, single_field_within_target = false, fixed_trace_exact = false;
        amrex::Real assembled_residual = 0., materialized_compensated = 0.,
                    materialized_ordinary = 0., discarded_action = 0.,
                    norm_roundoff_factor = 0.;
    };
    class RetainedRecovery
    {
      public:
        RetainedRecovery();
        ~RetainedRecovery();
        RetainedRecovery(RetainedRecovery&&) noexcept;
        RetainedRecovery& operator=(RetainedRecovery&&) noexcept;
        RetainedRecovery(RetainedRecovery const&) = delete;
        RetainedRecovery& operator=(RetainedRecovery const&) = delete;
        /** Local introspection. Collective action methods perform global preflight. */
        bool Ready() const noexcept;
        amrex::MultiFab const& High(int component) const;
        amrex::MultiFab const& Low(int component) const;
        RetainedReceipt const& Receipt() const;
        struct Storage;
      private:
        std::unique_ptr<Storage> m_storage;
        friend class DarwinVacuumAffineResponse;
        friend struct DarwinVacuumAffineResponse::Impl;
    };
    /** Additive precise-CPU electric RZ prototype. Preserves represented fixed
     * trace and zero-start range correction; no production caller is enabled.
     * All ranks call. Failure leaves the old owned output unchanged. Every
     * Freeze attempt invalidates prior outputs; high/low access is const and generation-bound.
     * No single-field publication is performed, even when its receipt passes.
     */
    RetainedReceipt RecoverRetained(RetainedRecovery&,
                                     View const* trace_increment = nullptr);
    /** Same owned representation and frozen masks, with a zero homogeneous
     * baseline. All ranks must select the same affine/homogeneous method. */
    RetainedReceipt ResponseRetained(RetainedRecovery&, View const& trace_increment);
    /** Additive accepted-origin consumers. Both trace parts have the same
     * support contract. The original overloads retain their exact arithmetic
     * when no paired origin/trace was supplied. No single-field publication. */
    RetainedReceipt RecoverRetainedPair(RetainedRecovery&, View const& trace_high,
                                        View const& trace_low);
    RetainedReceipt ResponseRetainedPair(RetainedRecovery&, View const& trace_high,
                                         View const& trace_low);
    /** Reconstruct an affine recovery from checkpoint numeric values without
     * a solve. Requires a freshly frozen identical operator and baseline plus
     * the original numeric trace increment(s), if used. Their signed-zero
     * values matter to the original high-only affine-base arithmetic.
     * The original target is recomputed and every fixed/trace row and actual
     * retained masked curl is checked. Only a fresh process-local recovery
     * binding is produced. The caller still owes the joined plasma/divergence/
     * null and physical checks. Failure preserves output and input bytes;
     * no public fields are written. No serialized convergence flag is trusted.
     */
    RetainedReceipt ReconstructAffineRetained(RetainedRecovery&,
        View const& solution_high, View const& solution_low,
        View const* original_trace_high=nullptr, View const* original_trace_low=nullptr);
    bool MatchesRetained(RetainedRecovery const&) const noexcept;
    /** Transactional paired action of the identical frozen masked operator.
     * All ranks call, including ranks whose local output/binding is invalid.
     * Output FAB storage is disjoint pairwise and from both owned input parts.
     * Only valid rows are written; the caller owns output guard completion.
     */
    bool ApplyRetainedMaskedCurl(View const& high, View const& low,
                                 RetainedRecovery const&);
    /** Same-core paired curl on the frozen correction-or-trace union. At the
     * joined endpoint this is exactly its global free mask; fixed PEC/axis rows
     * remain excluded. Binding, actual storage-alias and collective no-write
     * preflight are identical to ApplyRetainedMaskedCurl. */
    bool ApplyRetainedGlobalCurl(View const& high, View const& low,
                                RetainedRecovery const&);
    DarwinVacuumAffineResponse(WarpX &, View const &electric_layout, View const &magnetic_layout,
                               Options);
    ~DarwinVacuumAffineResponse();
    DarwinVacuumAffineResponse(DarwinVacuumAffineResponse const &) = delete;
    DarwinVacuumAffineResponse &operator=(DarwinVacuumAffineResponse const &) = delete;
    /** Transactional. Masks must be disjoint 0/1 arrays. All baseline/reference values finite.
     * Copies the baseline, masks and entire reference allocation. Null reference is homogeneous.
     * The trace mask excludes vacuum correction rows, walls and conductors.
     */
    bool Freeze(Mask const &correction, Mask const &trace, View const &baseline,
                View const &imposed_reference);
    /** Opt-in copied two-part origin for the same retained Core. No support
     * clipping: old/new recovery-mask transitions keep the complete origin.
     * All ranks call; every attempt revokes old retained outputs. Ordinary
     * Recover/Response decline while this paired origin is bound. */
    bool FreezeRetained(Mask const& correction, Mask const& trace,
                        View const& baseline_high, View const& baseline_low,
                        View const& imposed_reference);
    /** Recover the frozen origin plus an optional homogeneous plasma-trace increment.
     * Trace must vanish outside the frozen trace mask. Output components must be distinct and
     * may not alias trace components. On failure output, frozen data and accepted state remain
     * unchanged. Only output valid rows are written; the caller owns publication/guard filling.
     */
    Result Recover(View const &output, View const *trace_increment = nullptr);
    /** The linear response uses zero vacuum baseline and homogeneous boundary data; it retains
     * no correction history. Repeated applications always start CG from zero.
     */
    Result Response(View const &output, View const &trace_increment);
    /** Native masked curl-curl action, with the unchanged axis/PMC boundary completion.
     * No input/output aliasing. Writes valid output only. Does not mask the input.
     */
    void ApplyMaskedCurl(View const &output, View const &input, bool affine = false);
    /** Additive homogeneous private-PC action, same native mask/normalization.
     * False before output writes if the compensated stencil is unsupported.
     * Existing ApplyMaskedCurl and all production callers remain unchanged. */
    bool ApplyCompensatedMaskedCurl(View const &output, View const &input);
    amrex::Real MetricDot(View const &, View const &) const;
    amrex::Real Diagonal() const;
  private:
    bool ApplyRetainedCurl(View const&, View const&, RetainedRecovery const&, bool);
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::darwin
#endif
