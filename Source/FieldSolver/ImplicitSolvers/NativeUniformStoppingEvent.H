/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_UNIFORM_STOPPING_EVENT_H_
#define WARPX_NATIVE_UNIFORM_STOPPING_EVENT_H_
#include "NativeAcceptedStoppingContext.H"
#include <memory>
#include <string>
class WarpX;
namespace warpx::thermal {
struct UniformStoppingOptions {
    std::string fast_species;
    amrex::Real interval=0.;
    amrex::Real coulomb_log=0.;
    amrex::Real proper_speed_cap=0.;
    amrex::Real relative_convention_budget=0.;
};
struct UniformStoppingLedger {
    amrex::GpuArray<amrex::Real,3> electric_impulse{}, mean_current_increment{};
    amrex::GpuArray<amrex::Real,3> ion_momentum_change{}, reaction_momentum{}, electron_momentum_change{};
    amrex::Real ion_initial_energy=0., electron_initial_energy=0.;
    amrex::Real ion_energy_change=0., electron_energy_change=0., heat=0.;
    amrex::Real drag_energy_change=0., drag_bulk_work=0., electric_particle_work=0.;
    amrex::Real endpoint_particle_work=0., relativistic_defect=0., relativistic_bound=0.;
    amrex::Real spatial_transfer=0., constraint_work=0., mean_current_work=0.;
    amrex::Real transpose_work=0., heat_transfer_error=0., accounting_defect=0.;
    amrex::Real actual_energy_defect=0., positive_energy_scale=0., arithmetic_bound=0.;
    amrex::Real current_scale=0., harmonic_residual=0., grid_residual=0., grid_bound=0.;
    amrex::Real uniform_error=0., uniform_bound=0., current_initial_error=0.;
    amrex::Real absolute_work=0., contribution_count=0., operations=0.;
    amrex::Real positive_momentum_scale=0., momentum_error=0., momentum_bound=0.;
    int iterations=0;
};
/** Accepted-only fixed-density periodic Cartesian3D harmonic event.
 * Private preparation uses the native stopping law/gather/reaction transpose
 * and instantaneous Esirkepov current. No field residual, RNG or event guards.
 * This does NOT implement the general spatial/DT/circuit stopping event.
 *
 * Constructor borrows immutable context/temperature and the accepted cell-U
 * destination. Prepare copies state; CommitOnce collectively checks raw bytes
 * and publishes only u, accepted Je and U. Rollback restores these bytes until
 * Finalize. Required nonthrowing hook invalidates caller caches after either
 * mutation. No topology or external state change is allowed in that window.
 * Caller owns thermodynamic mirrors and accepted event epoch publication.
 */
class NativeUniformStoppingEvent {
public:
    struct InvalidationHook { void (*function)(void*) noexcept=nullptr; void* context=nullptr; };
    NativeUniformStoppingEvent (WarpX&, NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin, amrex::MultiFab& cell_energy,
        UniformStoppingOptions const&);
    ~NativeUniformStoppingEvent ();
    NativeUniformStoppingEvent (NativeUniformStoppingEvent const&)=delete;
    bool Prepare ();
    bool CommitOnce (ablastr::fields::VectorField const& accepted_current, InvalidationHook);
    bool Rollback (InvalidationHook);
    void Finalize ();
    bool Prepared () const;
    char const* Failure () const;
    UniformStoppingLedger const& Ledger () const;
    ablastr::fields::ConstVectorField CandidateElectronCurrent () const;
    amrex::MultiFab const& CandidateEnergy () const;
    // Independent receipt validation also supports deliberate fixture corruptions.
    static bool ValidateLedger (UniformStoppingLedger const&, UniformStoppingOptions const&);
public:
    // Public opaque name is required by NVCC extended-lambda access rules.
    struct Impl;
private:
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::thermal
#endif
