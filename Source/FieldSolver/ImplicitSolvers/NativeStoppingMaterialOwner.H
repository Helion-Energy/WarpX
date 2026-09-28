/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_MATERIAL_OWNER_H_
#define WARPX_NATIVE_STOPPING_MATERIAL_OWNER_H_
#include "NativeStoppingMaterialSupport.H"
#include "NativeStoppingCarryCertificate.H"
#include "NativeStoppingReactionTranspose.H"
#include <string>
#include <vector>
namespace warpx::implicit { class NativeAcceptedStepCandidate; }
namespace warpx::thermal {
struct NativeStoppingOwnerToken;
class NativeStoppingMaterialOwner;
class NativeStoppingMaterialSource;
/** Local lifetime token, never an MPI readiness predicate. */
class NativeStoppingOwnerLease {
public:
    bool Valid() const noexcept;
private:
    std::shared_ptr<NativeStoppingOwnerToken const> m_token;
    std::uint64_t m_generation=0;
    friend class NativeStoppingMaterialOwner;
};
enum class NativeStoppingOwnerStatus { Ready, Unsupported, Stale, Invalid, Terminal };
struct NativeStoppingOwnerOptions {
    std::string fast_species;
    amrex::Real coulomb_log=1.;
    amrex::Real proper_speed_cap=0.;
};
struct NativeStoppingOwnerTrial {
    ablastr::fields::ConstVectorField impulse{}, mean_current{};
    amrex::MultiFab const* energy_increment=nullptr;
};
struct NativeStoppingOwnerCensus {
    amrex::Long particles=0, charged=0, neutral=0, charged_tiles=0;
    amrex::Long footprint_points=0;
    amrex::Long empty_free_rows=0, empty_fixed_rows=0;
    amrex::Long nonzero_free_rows=0, nonzero_fixed_rows=0;
    amrex::Long empty_axis_rows=0, nonzero_axis_rows=0;
    amrex::Real axis_current_max=0.;
    amrex::Real free_current_max=0., fixed_current_max=0.;
    StoppingMaterialFootprint footprint;
    StoppingMaterialFailure material_failure=StoppingMaterialFailure::Unprepared;
    amrex::Real density_difference=0., density_bound=0.;
};
struct NativeStoppingOwnerWork {
    StoppingMaterialWork material;
    StoppingCarryReport carry;
    StoppingReactionBalance reaction;
    amrex::Real ion_before=0., ion_increment=0., drag=0., bulk=0.;
    amrex::Real electric=0., endpoint_particle_work=0., heat_nominal=0.;
    amrex::Real heat_delivered=0., heat_transfer=0., convention=0.;
    amrex::Real convention_bound=0., absolute_particle_work=0.;
    amrex::Real current_grid_work=0., spatial_transfer=0.;
    amrex::Real signed_thermal_residual=0., axis_reaction_work=0.;
    amrex::Long actions=0, material_actions=0, tuple_validation_preflights=0;
};
struct NativeStoppingOwnerPoint {
    amrex::GpuArray<amrex::Real,3> position{}, old{}, minus{}, drag{}, next{}, ve{}, psi{};
    amrex::Real weight=0., mass=0., charge=0., rate=0., heat=0.;
};
struct NativeStoppingOwnerParticleView {
    std::string species;
    int grid=-1, tile=-1;
    amrex::Long count=0;
    std::uint64_t const* device_idcpu=nullptr;
    NativeStoppingOwnerPoint const* device_points=nullptr;
};
struct NativeStoppingOwnerView {
    bool ready=false;
    ablastr::fields::ConstVectorField impulse{}, mean_current{}, velocity{};
    // Read-only native constitutive operands for the owned source residual.
    ablastr::fields::ConstVectorField initial_electron_current{}, mass{}, inertia_impulse{};
    std::array<amrex::iMultiFab const*,3> physical_support{};
    ablastr::fields::ConstVectorField momentum{}, raw_momentum{}, electron_current{}, mean_residual{};
    ablastr::fields::ConstVectorField ion_initial{}, ion_endpoint{};
    // Valid-row stored current includes the separately certified V carry.
    // electron_current above remains the physical material constituent.
    ablastr::fields::ConstVectorField stored_electron_current{}, carried_current{};
    amrex::MultiFab const* midpoint_energy=nullptr;
    amrex::MultiFab const* nodal_temperature=nullptr;
    amrex::MultiFab const* nodal_heat=nullptr;
    amrex::MultiFab const* cell_heat=nullptr;
    amrex::MultiFab const* thermal_residual=nullptr;
    std::vector<NativeStoppingOwnerParticleView> particles;
    NativeStoppingOwnerWork work;
};
struct NativeStoppingOwnerResult {
    NativeStoppingOwnerStatus status=NativeStoppingOwnerStatus::Unsupported;
    std::string reason;
    NativeStoppingOwnerLease lease;
    NativeStoppingOwnerCensus census;
};
/** Complete real-population capture and pure fixed-position source action.
 * Only the retained candidate constructs this owner. No public publication/root API.
 * Entry methods are collective, including invalid-token calls. Stale failure
 * precedes private action/output writes. Actions invalidate prior views.
 * Cancel destroys the owner before replacing the whole particle map.
 * Invalidate/destruction never communicate. Je_V==0 remains strict.
 */
class NativeStoppingMaterialOwner {
public:
    struct Impl;
    ~NativeStoppingMaterialOwner();
    NativeStoppingMaterialOwner(NativeStoppingMaterialOwner const&)=delete;
    NativeStoppingMaterialOwner& operator=(NativeStoppingMaterialOwner const&)=delete;
    void Invalidate() noexcept;
    NativeStoppingOwnerResult Evaluate(NativeStoppingOwnerLease const&,
                                      NativeStoppingOwnerTrial const&);
    NativeStoppingOwnerView View(NativeStoppingOwnerLease const&);
    NativeStoppingOwnerResult State() const;
private:
    NativeStoppingMaterialOwner(WarpX&, AcceptedStoppingOptions const&,
        NativeStoppingOwnerOptions const&, amrex::Real source_time,
        std::uint64_t source_epoch, amrex::Real half_interval,
        std::shared_ptr<NativeStoppingCarryCertificate const> carry={},
        bool stable_drag_increment=false);
    NativeStoppingOwnerResult Capture();
    bool Matches(NativeStoppingOwnerLease const&);
    bool PublishOnce(NativeStoppingOwnerLease const&);
    bool PublishedPopulationMatches();
    NativeStoppingOwnerLease Lease() const;
    std::unique_ptr<Impl> m_impl;
    std::shared_ptr<NativeStoppingOwnerToken> m_token;
    friend class warpx::implicit::NativeAcceptedStepCandidate;
    friend class NativeStoppingMaterialSource;
};
}
#endif
