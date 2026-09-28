/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXPECTED_ION_EXCHANGE_SOURCE_H_
#define WARPX_EXPECTED_ION_EXCHANGE_SOURCE_H_
#include "AcceptedIonExchange.H"
#include "ExpectedIonExchangeSource_K.H"
namespace warpx::thermal {
enum class ExpectedIonEndpointContract {
    Unspecified,
    CompleteRedistributedPreCollisionEndpoint
};
struct ExpectedIonWorkComponent {
    enum : int { Relaxation, Redirect, NonrelativisticRelaxation,
        NonrelativisticRedirect, NonrelativisticDriftWork,
        ConventionLower, ConventionUpper, QuadratureError, Count };
};
using ExpectedIonWorkLedger = std::array<amrex::Real, ExpectedIonWorkComponent::Count>;
// Opt-in scratch source. No legacy or accepted momentum law is changed.
// The caller explicitly owns the channel work budget and its force-work
// interpretation; this helper is not a whole-field energy conservation proof.
class ExpectedIonExchangeSource {
  public:
    ExpectedIonExchangeSource (const AcceptedIonExchange& layout,
                               ExpectedIonEnergyOptions options);
    // Collective, PURE, repeatable; no RNG. The prepared coefficients must be
    // precisely those used by the accepted endpoint kick. Views cover all live
    // charged native species in order, each physical particle exactly once.
    // budget: scalar CELL-EXTENSIVE joules for the whole physical step and the
    // exact selected source subsystem. It is REQUIRED; no Joule/drift work is
    // guessed. Qe=(budget-sum conditional ion work)/(physical cell volume*dt).
    bool Evaluate (const AcceptedIonExchange& prepared,
                   const std::vector<IonExchangeParticleView>& endpoints,
                   const amrex::MultiFab& available_cell_work,
                   ExpectedIonEndpointContract contract);
    // Valid only after successful Evaluate. SpeciesWork has interleaved Count
    // extensive-joule channels per charged species, including excluded zeros.
    const amrex::MultiFab& Source () const;
    const amrex::MultiFab& SpeciesWork () const;
    // Stateless MPI-collective acceptance diagnostic; not called by Evaluate.
    std::vector<ExpectedIonWorkLedger> Ledger () const;
    bool Valid () const { return m_valid; }
    const ExpectedIonEnergyOptions& Options () const { return m_options; }
  private:
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells;
    amrex::DistributionMapping m_distribution;
    std::vector<KineticSpeciesDescriptor> m_descriptors;
    ExpectedIonEnergyOptions m_options;
    amrex::MultiFab m_source, m_work;
    bool m_valid = false;
};
// The container must hold the explicitly acknowledged materialized endpoint,
// already redistributed. This bridge only reads it; it neither constructs an
// endpoint from midpoint/saved state nor substitutes aggregate MM moments.
bool EvaluateNativeExpectedIonExchangeSource (
    ExpectedIonExchangeSource&, const AcceptedIonExchange&,
    MultiParticleContainer&, const amrex::MultiFab& available_cell_work,
    ExpectedIonEndpointContract);
} // namespace warpx::thermal
#endif
