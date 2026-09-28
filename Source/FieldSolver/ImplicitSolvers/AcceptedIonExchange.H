/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ACCEPTED_ION_EXCHANGE_H_
#define WARPX_ACCEPTED_ION_EXCHANGE_H_
#include "KineticThermalSpecies.H"
#include <AMReX_GpuContainers.H>
#include <cstdint>
namespace warpx::thermal {
struct IonExchangeOptions {
    amrex::Real dt = 1, raw_density_floor = 0, time = 0;
    bool relaxation = false, redirect = false;
    bool embedded_boundary = false;
    int physical_levels = 1, azimuthal_modes = 1;
    // These transforms remain outside the qualified volumetric source path.
    bool redirect_kick_cap = false, temperature_shunt = false;
};
struct IonExchangeCoefficient {
    enum : int {
        Nu,
        DriftR,
        DriftTheta,
        DriftZ,
        TemperatureKelvin,
        RedirectEnergy,
        ReferenceNu,
        Count
    };
};
struct IonExchangeSpeciesLedger {
    // Joules. Requested values integrate converged physical nodal rates * dt.
    amrex::Real requested_relaxation = 0, requested_redirect = 0;
    // Finite-step NONRELATIVISTIC OU expectation for actual endpoint particles.
    amrex::Real expected_relaxation = 0, expected_redirect = 0;
    // Relativistic native kinetic energy, weighted by actual macro weights.
    // Relaxation is a same-draw relaxation-only counterfactual; redirect is
    // K(combined variance)-K(relaxation-only), so the two telescope. This is an
    // explicit attribution convention, not two independent random operators.
    amrex::Real realized_relaxation = 0, realized_redirect = 0,
                realized_total = 0;
    // Expected-requested and realized-expected. The latter includes the
    // relativistic-vs-nonrelativistic energy convention as well as sampling.
    amrex::Real expectation_defect = 0, realization_defect = 0;
    amrex::Real reference_nu_max_difference = 0; // 1/s; optional legacy oracle
};
enum class IonExchangeStatus {
    Unprepared,
    Prepared,
    Committed,
    Rejected,
    FailedAfterDraw
};
// Endpoint view: device-readable NGP cell indices and azimuth [radians],
// physical macro weight and native proper momenta u=gamma*v [m/s]. Distinct
// views must refer to disjoint particle momentum storage. Grid index is the
// global BoxArray index. Native bridge constructs these after redistribution.
struct IonExchangeParticleView {
    std::size_t species = 0;
    int grid_index = 0;
    long count = 0;
    const amrex::IntVect* cell = nullptr;
    const amrex::ParticleReal* theta = nullptr;
    const amrex::ParticleReal* weight = nullptr;
    std::array<amrex::ParticleReal*, 3> momentum{};
};
// All methods that prepare or commit are MPI collective; every rank,
// including an empty rank, must participate in the same transaction.
class AcceptedIonExchange {
  public:
    AcceptedIonExchange (const amrex::Geometry&, const amrex::BoxArray&,
                         const amrex::DistributionMapping&,
                         std::vector<KineticSpeciesDescriptor>,
                         IonExchangeOptions);
    AcceptedIonExchange (const AcceptedIonExchange&) = delete;
    AcceptedIonExchange& operator=(const AcceptedIonExchange&) = delete;
    // PURE and repeatable before CommitOnce. Drift is a required NODAL
    // (Vr,Vtheta,Vz) vector [m/s] from the consistent native midpoint state.
    // Nu, redirect variance rate, Te and drift use ordinary corner averages
    // to CC, then endpoint NGP lookup. No independent parser sets the new nu.
    // Optional legacy oracle evaluates parser(rho_CC,Te_CC[eV],old_Ti_CC[eV],t)
    // for diagnostics only. Copies all needed data; no borrowed input survives.
    bool Prepare (const ThermalSourceState&, const amrex::MultiFab& ion_rates,
                  const std::array<const amrex::MultiFab*, 3>& nodal_drift,
                  const std::vector<const amrex::MultiFab*>& old_cell_ti = {},
                  const amrex::ParserExecutor<4>* legacy_nu_oracle = nullptr);
    // ACCEPTED ENDPOINT ONLY. Preflight consumes no RNG. All stochastic
    // candidates are staged before any particle write. Failure after drawing
    // consumes RNG but writes no momenta and is terminal: no automatic retry.
    // Every first attempt is terminal; any subsequent attempt returns false.
    bool CommitOnce (const std::vector<IonExchangeParticleView>&);
    IonExchangeStatus
    Status () const {
        return m_status;
    }
    const std::vector<IonExchangeSpeciesLedger>& Ledger () const;
    const amrex::MultiFab& Coefficients (std::size_t) const;
    const amrex::Geometry&
    Geometry () const {
        return m_geometry;
    }
    const amrex::BoxArray&
    Cells () const {
        return m_cells;
    }
    const amrex::DistributionMapping&
    Distribution () const {
        return m_distribution;
    }
    const std::vector<KineticSpeciesDescriptor>&
    Descriptors () const {
        return m_descriptors;
    }
    const IonExchangeOptions&
    Options () const {
        return m_options;
    }

  private:
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells, m_nodes;
    amrex::DistributionMapping m_distribution;
    std::vector<KineticSpeciesDescriptor> m_descriptors;
    IonExchangeOptions m_options;
    std::vector<std::unique_ptr<amrex::MultiFab>> m_coefficients;
    std::vector<IonExchangeSpeciesLedger> m_ledger;
    std::unique_ptr<amrex::iMultiFab> m_owner;
    IonExchangeStatus m_status = IonExchangeStatus::Unprepared;
};
// Checks actual native geometry/species/layout and constructs endpoint NGP
// indices read-only. Caller must first materialize AND redistribute endpoint
// particles. No saved xn/un, charge/current registry, RNG in Prepare, or U
// writes.
bool CommitNativeIonExchangeOnce (AcceptedIonExchange&,
                                  MultiParticleContainer&);
} // namespace warpx::thermal
#endif
