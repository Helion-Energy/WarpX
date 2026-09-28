/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_MATERIAL_SUPPORT_H_
#define WARPX_NATIVE_STOPPING_MATERIAL_SUPPORT_H_

#include "NativeAcceptedStoppingContext.H"
#include <AMReX_iMultiFab.H>
#include <array>
#include <memory>

class WarpX;
namespace warpx::thermal {

class NativeStoppingMaterialSupport;
class NativeStoppingMaterialOwner;
class NativeStoppingCarryCertificate;
/** Local opaque token. Every collective action checks the token on every rank. */
class StoppingMaterialLease {
    friend class NativeStoppingMaterialSupport;
    void const* m_owner = nullptr;
    std::uint64_t m_generation = 0;
    amrex::Real m_time = 0.;
    std::uint64_t m_epoch = 0;
};

struct StoppingMaterialPoint {
    amrex::GpuArray<amrex::ParticleReal,3> position{}; // Cartesian, including RZ.
    amrex::Real charge = 0.; // This list contains charged participants only.
    amrex::Real weight = 0.;
};
struct StoppingMaterialFootprint {
    // points, invalid input, rate density, current V, reaction V,
    // uncovered allocation, branch, malformed local tile call.
    std::array<amrex::Long,8> counts{};
    bool valid = false;
};
struct StoppingMaterialWork {
    amrex::Real kinetic_before = 0., kinetic_after = 0.;
    amrex::Real kinetic_increment = 0., constitutive_work = 0.;
    amrex::Real endpoint_rounding = 0.; // increment - constitutive work
    amrex::Real velocity_reaction = 0., mean_reaction = 0.;
    amrex::Real reaction_rounding = 0.; // <ve,Pi> + <Jbar,b>
    amrex::Real absolute_kinetic = 0., absolute_increment = 0.;
    amrex::Real absolute_constitutive = 0., absolute_reaction = 0.;
    amrex::Long physical_rows = 0, empty_rows = 0;
};
enum class StoppingMaterialFailure {
    None, Unsupported, InvalidBinding, InvalidDensity, EmptyCurrent,
    StaleMaterial, Footprint, InvalidAction, Unprepared
};

/** Pure fixed-position material prerequisite; no event/solver publication API.
 * Uses NativeInertiaSupport::FreezeEdges with the explicit candidate edge law.
 * The old context and all event guards retain their positive-density contract.
 *
 * Prepare, Matches, BeginFootprint, FinishFootprint and Apply are collective.
 * All ranks must enter even with a locally bad token/input. AddFootprintTile
 * and const accessors are local. Prepared/ActionReady are local diagnostics,
 * never a reason to skip a collective action. A successful preparation/action
 * invalidates old views. A failed action preserves all previously published
 * output bytes but revokes ActionReady. Invalidate performs no field writes.
 *
 * Footprint coverage proves only the supplied point list; a later event owner
 * must bind complete charged population coverage to its particle transaction.
 * No inverse mass is evaluated on V. Trial Jbar_V is allowed, with residual
 * Jbar_V; Psi_V is unrestricted. Accepted Je_V and deposited Pi_V must be zero.
 */
class NativeStoppingMaterialSupport {
public:
    struct Impl; // Public enclosing type for native CUDA lambdas, not an API.
    using ConstVector = ablastr::fields::ConstVectorField;
    NativeStoppingMaterialSupport(WarpX&, AcceptedStoppingOptions const&);
    ~NativeStoppingMaterialSupport();
    NativeStoppingMaterialSupport(NativeStoppingMaterialSupport const&) = delete;
    NativeStoppingMaterialSupport& operator=(NativeStoppingMaterialSupport const&) = delete;

    bool Prepare(AcceptedStoppingBinding const&);
    StoppingMaterialLease Lease() const;
    bool Matches(StoppingMaterialLease const&);
    void Invalidate() noexcept;
    bool Prepared() const noexcept { return m_prepared; }
    bool ActionReady() const noexcept { return m_action; }
    bool FootprintReady() const noexcept { return m_footprint; }
    StoppingMaterialFailure Failure() const noexcept { return m_failure; }
    bool LegacyPositiveBranch() const;
    std::uint64_t Generation() const noexcept { return m_generation; }

    bool BeginFootprint(StoppingMaterialLease const&);
    void AddFootprintTile(amrex::MFIter const&, amrex::Box const& particle_tile,
                          StoppingMaterialPoint const* device_points, amrex::Long count);
    bool FinishFootprint(StoppingMaterialLease const&);
    StoppingMaterialFootprint const& Footprint() const noexcept { return m_report; }

    bool Apply(StoppingMaterialLease const&, ConstVector const& psi,
               ConstVector const& conjugate_momentum, ConstVector const& mean_current);
    ConstVector CapturedCurrent() const;
    amrex::MultiFab const& CapturedCharge() const;
    amrex::iMultiFab const& PhysicalSupport(int component) const;
    amrex::iMultiFab const& CurrentPullback(int component) const;
    ConstVector Mass() const;
    ConstVector Velocity() const;
    ConstVector InertiaImpulse() const;
    ConstVector EndpointCurrent() const;
    ConstVector MeanResidual() const;
    StoppingMaterialWork const& Work() const;
    AcceptedStoppingGather GatherView(amrex::MFIter const&, ConstVector const& magnetic) const;

private:
    friend class NativeStoppingMaterialOwner;
    bool PrepareCarried(AcceptedStoppingBinding const&,NativeStoppingCarryCertificate const&);
    bool PrepareImpl(AcceptedStoppingBinding const&,ConstVector const* material_current);
    bool LocalMatches(StoppingMaterialLease const&) const;
    bool Fail(StoppingMaterialFailure, bool invalidate_material = false) noexcept;
    WarpX& m_sim;
    AcceptedStoppingOptions m_options;
    std::unique_ptr<Impl> m_impl;
    std::uint64_t m_generation = 0;
    bool m_prepared = false, m_action = false, m_open = false, m_footprint = false;
    bool m_local_tile_bad = false;
    StoppingMaterialFailure m_failure = StoppingMaterialFailure::Unprepared;
    StoppingMaterialFootprint m_report;
};
} // namespace warpx::thermal
#endif
