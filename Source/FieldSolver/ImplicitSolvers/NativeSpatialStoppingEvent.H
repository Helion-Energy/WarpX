/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_SPATIAL_STOPPING_EVENT_H_
#define WARPX_NATIVE_SPATIAL_STOPPING_EVENT_H_
#include "NativeAcceptedStoppingContext.H"
#include <memory>
#include <string>
class WarpX;
namespace warpx::thermal {
struct SpatialStoppingFields {
    ablastr::fields::VectorField potential{}, magnetic{}, displacement{};
};
struct SpatialStoppingOptions {
    std::string fast_species;
    amrex::Real interval=0.;
    amrex::Real coulomb_log=0.;
    amrex::Real proper_speed_cap=0.;
    amrex::Real relative_convention_budget=0.;
};
struct SpatialStoppingLedger {
    amrex::GpuArray<amrex::Real,3> electric_impulse{}, mean_current_increment{};
    amrex::GpuArray<amrex::Real,3> ion_momentum_change{}, reaction_momentum{}, electron_momentum_change{};
    amrex::Real ion_initial_energy=0., electron_initial_energy=0.;
    amrex::Real ion_energy_change=0., electron_energy_change=0., heat=0.;
    amrex::Real drag_energy_change=0., drag_bulk_work=0., electric_particle_work=0.;
    amrex::Real endpoint_particle_work=0., relativistic_defect=0., relativistic_bound=0.;
    amrex::Real spatial_transfer=0., constraint_work=0., mean_current_work=0.;
    amrex::Real transpose_work=0., heat_transfer_error=0., accounting_defect=0.;
    amrex::Real actual_energy_defect=0., positive_energy_scale=0., arithmetic_bound=0.;
    amrex::Real current_scale=0., spatial_residual=0., grid_residual=0., grid_bound=0.;
    amrex::Real uniform_error=0., uniform_bound=0., current_initial_error=0.;
    amrex::Real absolute_work=0., contribution_count=0., operations=0.;
    amrex::Real positive_momentum_scale=0., momentum_error=0., momentum_bound=0.;
    amrex::Real magnetic_energy_change=0., magnetic_initial_energy=0., curl_work=0.;
    amrex::Real displacement_work=0., gauge_error=0., electric_momentum_transfer=0.;
    amrex::Real magnetic_norm=0., projection_residual=0.;
    amrex::Real faraday_error=0.,faraday_bound=0.,displacement_curl=0.,displacement_curl_bound=0.;
    amrex::Real current_range_mean=0.,current_range_bound=0.,impulse_range_mean=0.,impulse_range_bound=0.,range_ratio=0.;
    amrex::Real current_range_scale=0.,impulse_range_scale=0.;
    int range_calls=0;
    int linear_iterations=0, evaluations=0;
    int iterations=0;
};
/** Accepted-only fixed-density periodic Cartesian3D spatial event.
 * Private preparation uses the native stopping law/gather/reaction transpose
 * and instantaneous Esirkepov current. No field residual, RNG or event guards.
 * This does NOT implement RZ/walls, variable density, DT or circuit events.
 *
 * Constructor borrows immutable context/temperature and the accepted cell-U
 * destination. Prepare copies state; CommitOnce collectively checks raw bytes
 * and publishes u, accepted Je/U and the coupled A/B/D impulse. Rollback restores these bytes until
 * Finalize. Required nonthrowing hook invalidates caller caches after either
 * mutation. No topology or external state change is allowed in that window.
 * Caller owns thermodynamic mirrors and accepted event epoch publication.
 */
class NativeSpatialStoppingEvent {
public:
    struct InvalidationHook { void (*function)(void*) noexcept=nullptr; void* context=nullptr; };
    NativeSpatialStoppingEvent (WarpX&, NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin, amrex::MultiFab& cell_energy,
        SpatialStoppingOptions const&, SpatialStoppingFields const&);
    ~NativeSpatialStoppingEvent ();
    NativeSpatialStoppingEvent (NativeSpatialStoppingEvent const&)=delete;
    bool Prepare ();
    bool CommitOnce (ablastr::fields::VectorField const& accepted_current, InvalidationHook);
    bool Rollback (InvalidationHook);
    void Finalize ();
    bool Prepared () const;
    char const* Failure () const;
    SpatialStoppingLedger const& Ledger () const;
    ablastr::fields::ConstVectorField CandidateElectronCurrent () const;
    amrex::MultiFab const& CandidateEnergy () const;
    ablastr::fields::ConstVectorField CandidateMagnetic () const;
    ablastr::fields::ConstVectorField CandidatePotential () const;
    ablastr::fields::ConstVectorField CandidateDisplacement () const;
    ablastr::fields::ConstVectorField ElectricImpulse () const;
    // Independent receipt validation also supports deliberate fixture corruptions.
    static bool ValidateLedger (SpatialStoppingLedger const&, SpatialStoppingOptions const&);
public:
    // Public opaque name is required by NVCC extended-lambda access rules.
    struct Impl;
private:
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::thermal
#endif
