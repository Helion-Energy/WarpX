/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_INERTIA_SUPPORT_H_
#define WARPX_NATIVE_INERTIA_SUPPORT_H_

#include "DarwinInitialRateSchur.H"

#include <AMReX_iMultiFab.H>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace warpx::darwin
{

/** Explicit experimental interface choices; neither changes a runtime consumer. */
enum class InertiaEdgePolicy
{
    Unspecified,
    ExistingNodalGate,
    NativeEdgeCandidate
};
enum class InertiaRecoveryMask
{
    Unspecified,
    None,
    Vacuum,
    Transition,
    Global
};
enum class InertiaRecoveryComponents
{
    Unspecified,
    Flux,
    All
};

struct NativeInertiaSupportOptions
{
    std::array<InitialRateBoundary, AMREX_SPACEDIM> lower{};
    std::array<InitialRateBoundary, AMREX_SPACEDIM> upper{};
    InertiaEdgePolicy coefficient = InertiaEdgePolicy::Unspecified;
    InertiaRecoveryMask recovery = InertiaRecoveryMask::Unspecified;
    InertiaRecoveryComponents components = InertiaRecoveryComponents::Unspecified;
    amrex::Real charge_floor = 0.;             // Existing q_e*n_floor [C/m^3].
    amrex::Real reference_charge_density = 0.; // q_e*n_reference [C/m^3].
    amrex::Real recovery_density_fraction = 1.;
};

struct InertiaRangeMoment
{
    int component = -1;
    amrex::Real volume = 0.;
    amrex::Real integral = 0.;
    amrex::Real mean = 0.;
};
struct InertiaRangeResult
{
    bool compatible = false;
    bool finite = false;
    amrex::Real rhs_norm = 0.;
    amrex::Real target = 0.;
    amrex::Real isolated_defect = 0.;
    amrex::Real max_null_mean = 0.;
    std::vector<InertiaRangeMoment> floating;
};
struct InertiaRangeTarget
{
    const amrex::MultiFab *rhs = nullptr;
    amrex::Real absolute_tolerance = 0.;
    amrex::Real relative_tolerance = 0.;
};
struct InertiaSupportStats
{
    int canonical_nodes = 0;
    int anchored_components = 0;
    int floating_components = 0;
    int isolated_nodes = 0;
    int dirichlet_nodes = 0;
    // Flat setup-array payload; excludes map nodes, MPI buffers and local FABs.
    std::size_t replicated_graph_bytes = 0;
};
enum class InertiaSupportFailure
{
    None,
    InvalidDensity,
    IncompatibleRange
};

/** Pure frozen support/range adapter for native Yee inertia qualification.
 *
 * No runtime solver uses this helper. No particle, accepted field, density,
 * thermal capacity, or history is written. Input rho is positive-species charge
 * density [C/m^3], not mass density. Pedestal, taper and EB are outside this slice.
 * The raw endpoint density and the frozen recovery-mask density are separate.
 * The candidate edge policy differs from the nodal control at empty/subfloor
 * interfaces, but preserves its exact arithmetic when both nodes exceed floor.
 *
 * Freeze builds a replicated global graph on each rank. This is intentionally a
 * bounded fixture implementation, with O(global nodes) setup memory per rank,
 * not a scalable production topology service. Only Freeze copies topology data
 * to host. ApplyMass and CheckRange use device kernels/reductions; CheckRange
 * visits the grid once per nontrivial floating component and once for isolated
 * nodes. A distributed topology/reduction design is required before large runs.
 */
class NativeInertiaSupport
{
  public:
    struct State;
    using ConstVector = std::array<const amrex::MultiFab *, 3>;
    using Vector = std::array<amrex::MultiFab *, 3>;
    NativeInertiaSupport(const amrex::Geometry &, const amrex::BoxArray &,
                         const amrex::DistributionMapping &, const NativeInertiaSupportOptions &);
    ~NativeInertiaSupport();
    NativeInertiaSupport(const NativeInertiaSupport &) = delete;
    NativeInertiaSupport &operator=(const NativeInertiaSupport &) = delete;

    /** Transactional: invalid densities or an optional incompatible RHS leave
     * the previous frozen state intact. RHS includes its affine BC contribution;
     * it is checked unchanged, never projected. Caller ghosts are not consumed.
     */
    bool Freeze(const amrex::MultiFab &endpoint_raw_rho, const amrex::MultiFab &recovery_raw_rho,
                const InertiaRangeTarget *required_range = nullptr);
    /** Coefficients, masks and metric only, with no replicated graph or host
     * topology copy. Range/topology accessors are unavailable in this mode. */
    bool FreezeEdges(const amrex::MultiFab &endpoint_raw_rho,
                     const amrex::MultiFab &recovery_raw_rho);
    bool IsFrozen() const noexcept;
    InertiaSupportFailure LastFailure() const noexcept;
    const InertiaRangeResult &LastFreezeRange() const noexcept;
    const InertiaSupportStats &Stats() const;

    const amrex::MultiFab &RawEdge(int component) const;
    const amrex::MultiFab &CapacityEdge(int component) const;
    const amrex::MultiFab &KappaEdge(int component) const;
    const amrex::iMultiFab &PhysicalSupport(int component) const;
    const amrex::iMultiFab &RecoveryMask(int component) const;
    /** -1 anchored, -2 isolated free node, >=0 floating component identifier. */
    const amrex::iMultiFab &ComponentLabels() const;
    /** One selected scalar gauge per floating component, all isolated nodes. */
    const amrex::iMultiFab &GaugeNodes() const;
    /** Physical native nodal dual volume, zero on PEC rows/duplicate owners. */
    const amrex::MultiFab &NodeWeights() const;

    /** out=scale*K_edge*in. Native zero/periodic/PMC response images. */
    void ApplyMass(const ConstVector &in, const Vector &out, amrex::Real scale = 1.) const;
    /** Checks all metric left-null moments; does not change the RHS or state. */
    InertiaRangeResult CheckRange(const amrex::MultiFab &rhs, amrex::Real absolute_tolerance,
                                  amrex::Real relative_tolerance) const;

    // Public only because CUDA extended lambdas require a public enclosing
    // member. Call Freeze or FreezeEdges for the supported API contracts.
    bool FreezeImpl(const amrex::MultiFab &, const amrex::MultiFab &,
                    const InertiaRangeTarget *, bool build_topology);

  private:
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells;
    amrex::DistributionMapping m_distribution;
    NativeInertiaSupportOptions m_options;
    std::unique_ptr<State> m_state;
    InertiaSupportFailure m_failure = InertiaSupportFailure::None;
    InertiaRangeResult m_last_range;
};
} // namespace warpx::darwin
#endif
