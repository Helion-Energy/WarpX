/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_ION_ENDPOINT_TRIAL_H_
#define WARPX_IMPLICIT_ION_ENDPOINT_TRIAL_H_
#include "ExpectedIonExchangeSource.H"
#include "ImplicitAbsorbingEndpoint.H"
#include <AMReX_Particles.H>
namespace warpx::thermal {
// Read-only native midpoint and immutable Cartesian step-start arrays. Native
// coordinates are (r,z) in RZ and (x,y,z) in Cartesian. Proper u is Cartesian.
struct ImplicitIonMidpointTile {
    std::size_t species = 0;
    int grid = 0, tile = 0;
    long count = 0;
    const std::uint64_t* idcpu = nullptr;
    bool allow_deterministic_absorption = false;
    warpx::implicit::AbsorbingEndpointBoundary absorption;
    std::array<const amrex::ParticleReal*, AMREX_SPACEDIM> position{};
    const amrex::ParticleReal* theta = nullptr;
    const amrex::ParticleReal* weight = nullptr;
    std::array<const amrex::ParticleReal*, 3> momentum{}, old_position{},
        old_momentum{};
};
// Private native device particles, with private redistribution. No boundary
// operator, global NextID, RNG, accepted particles or registry is invoked.
// Qualified deterministic absorption is removed in private storage. The
// native RZ endpoint-only reflection image is applied to private r/theta/u;
// the runtime audit still rejects unresolved boundary actions.
class ImplicitIonEndpointTrial {
  public:
    enum : int { Weight = AMREX_SPACEDIM, Ux, Uy, Uz, Theta, Count };
    using Container =
        amrex::ParticleContainerPureSoA<Count, 0,
                                        amrex::PolymorphicArenaAllocator>;
    explicit ImplicitIonEndpointTrial (const AcceptedIonExchange&);
    bool Refresh (const std::vector<ImplicitIonMidpointTile>&);
    const std::vector<IonExchangeParticleView>& Views () const;
    const AcceptedIonExchange&
    Layout () const {
        return m_layout;
    }
    bool
    Valid () const {
        return m_valid;
    }

  private:
    const AcceptedIonExchange& m_layout;
    std::vector<std::unique_ptr<Container>> m_particles;
    std::vector<amrex::Gpu::DeviceVector<amrex::IntVect>> m_indices;
    std::vector<IonExchangeParticleView> m_views;
    bool m_valid = false;
};
// Caller must supply a full materialized implicit midpoint for EVERY probe.
// This never substitutes an aggregate MM response for the actual particles.
bool RefreshNativeImplicitIonEndpointTrial (ImplicitIonEndpointTrial&,
                                            MultiParticleContainer&);
} // namespace warpx::thermal
#endif
