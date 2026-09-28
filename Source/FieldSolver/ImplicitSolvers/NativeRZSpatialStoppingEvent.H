/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_RZ_SPATIAL_STOPPING_EVENT_H_
#define WARPX_NATIVE_RZ_SPATIAL_STOPPING_EVENT_H_
#include "NativeSpatialStoppingEvent.H"
#include <memory>
#include <string>
class WarpX;
namespace warpx::darwin { class NativeLongitudinalProducerCertificate; }
namespace warpx::thermal {
// Inherited electron_momentum_change[0,1] are cylindrical component
// integrals, not Cartesian net momentum. Only component2 is represented
// Cartesian axial momentum; the explicit m0 omission is reported below.
struct RZSpatialStoppingLedger : SpatialStoppingLedger {
    amrex::Real conductor_current_norm=0., conductor_work=0.;
    amrex::Real axis_reaction_conjugate=0., axis_reaction_work=0.;
    amrex::Real axial_momentum_defect=0., axial_electric_transfer=0., axial_image_transfer=0.;
    amrex::GpuArray<amrex::Real,3> unrepresented_particle_momentum{};
};
/** Explicit accepted-background binding. The conductor current is an
 * independent accepted companion, supported only on rmax tangential rows.
 * All A/B/D/Je/W fields are borrowed destinations; no companion is reseeded.
 */
struct RZStoppingBackground {
    ablastr::fields::VectorField conductor_current{};
    // Default-null retains the original D-only gate. The native certificate
    // has no public constructor from a caller-selected numerical allowance.
    std::shared_ptr<warpx::darwin::NativeLongitudinalProducerCertificate const> longitudinal_certificate;
};
struct RZStoppingBackgroundLedger {
    amrex::Real current_residual=0., current_bound=0.;
    amrex::Real magnetic_residual=0., magnetic_bound=0.;
    amrex::Real displacement_residual=0., displacement_bound=0.;
    amrex::Real displacement_certificate_bound=0.;
    bool displacement_certified=false;
    amrex::Real magnetic_cross_work=0., magnetic_self_work=0.;
    amrex::Real magnetic_cross_scale=0., initial_constraint_work=0.;
    amrex::Real initial_displacement_work=0., initial_conductor_work=0.;
};
/** Standalone accepted-only fixed-density RZm0 spatial event prerequisite.
 * Private preparation uses the native stopping law/gather/reaction transpose
 * and instantaneous Esirkepov current. No field residual, RNG or event guards.
 * Radial PEC conductor current is explicit and is NOT plasma Je. No live
 * endpoint publication, DT/topology/circuit/drive, vacuum or alpha qualification.
 *
 * Constructor borrows immutable context/temperature and the accepted cell-U
 * destination. Prepare copies state; CommitOnce collectively checks raw bytes
 * and publishes u, accepted Je/U and the coupled A/B/D impulse. Rollback restores these bytes until
 * Finalize. Required nonthrowing hook invalidates caller caches after either
 * mutation. No topology or external state change is allowed in that window.
 * Caller owns thermodynamic mirrors and accepted event epoch publication.
 */
class NativeRZSpatialStoppingEvent {
public:
    struct InvalidationHook { void (*function)(void*) noexcept=nullptr; void* context=nullptr; };
    NativeRZSpatialStoppingEvent (WarpX&, NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin, amrex::MultiFab& cell_energy,
        SpatialStoppingOptions const&, SpatialStoppingFields const&);
    // Increment form: preserve accepted backgrounds, including independent
    // Je/D/W. No finite magnetic rotation is added to this instantaneous
    // fixed-position source split. It does not open live event capability.
    NativeRZSpatialStoppingEvent (WarpX&, NativeAcceptedStoppingContext const&,
        amrex::MultiFab const& temperature_kelvin, amrex::MultiFab& cell_energy,
        SpatialStoppingOptions const&, SpatialStoppingFields const&,
        RZStoppingBackground const&);
    RZStoppingBackgroundLedger const& BackgroundLedger () const;
    // Advance only the opaque trace from the prepared native projection.
    // Requires CommitOnce; it never changes a physical field or source account.
    bool AppendDisplacementCertificate(warpx::darwin::NativeLongitudinalProducerCertificate const& old,
                                      warpx::darwin::NativeLongitudinalProducerCertificate& updated) const;
    ablastr::fields::ConstVectorField PotentialIncrement () const;
    ablastr::fields::ConstVectorField MagneticIncrement () const;
    ablastr::fields::ConstVectorField DisplacementIncrement () const;
    ablastr::fields::ConstVectorField ConductorIncrement () const;
    ~NativeRZSpatialStoppingEvent ();
    NativeRZSpatialStoppingEvent (NativeRZSpatialStoppingEvent const&)=delete;
    bool Prepare ();
    bool CommitOnce (ablastr::fields::VectorField const& accepted_current, InvalidationHook);
    bool Rollback (InvalidationHook);
    void Finalize ();
    bool Prepared () const;
    char const* Failure () const;
    RZSpatialStoppingLedger const& Ledger () const;
    ablastr::fields::ConstVectorField CandidateElectronCurrent () const;
    amrex::MultiFab const& CandidateEnergy () const;
    ablastr::fields::ConstVectorField CandidateMagnetic () const;
    ablastr::fields::ConstVectorField CandidatePotential () const;
    ablastr::fields::ConstVectorField CandidateDisplacement () const;
    ablastr::fields::ConstVectorField ElectricImpulse () const;
    ablastr::fields::ConstVectorField CandidateConductorCurrent () const;
    // Read-only prepared operands for independent accepted-event inventories.
    ablastr::fields::ConstVectorField MeanElectronCurrent () const;
    ablastr::fields::ConstVectorField StoppingVelocity () const;
    ablastr::fields::ConstVectorField StoppingImpulseDensity () const;
    ablastr::fields::ConstVectorField ElectricAuxiliary () const;
    // Independent receipt validation also supports deliberate fixture corruptions.
    static bool ValidateLedger (RZSpatialStoppingLedger const&, SpatialStoppingOptions const&);
public:
    // Public opaque name is required by NVCC extended-lambda access rules.
    struct Impl;
private:
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::thermal
#endif
