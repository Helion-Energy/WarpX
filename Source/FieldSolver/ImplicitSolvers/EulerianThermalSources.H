/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_SOURCES_H_
#define WARPX_EULERIAN_THERMAL_SOURCES_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_iMultiFab.H>
#include <array>
#include <memory>
#include <vector>

namespace warpx::thermal {

// Every component is a nodal physical power density [W/m^3]. Positive
// ElectronTotal heats U_e. Diagnostic components are NOT added a second time.
struct SourceComponent {
    enum : int {
        ElectronTotal,
        Joule,
        Relaxation,
        Viscous,
        HyperResistive,
        ExternalSink,
        Stopping,
        DeclinedJouleGate,
        DeclinedRedirect,
        DeclinedStopping,
        ViscousWork,
        ViscousStrain,
        DeclinedViscousWork,
        DeclinedHyperResistive,
        // Absolute unassignable work cannot cancel between cooling/heating
        // edges. The Declined channels above retain SIGNED work identities.
        UnassignableViscousWorkAbs,
        UnassignableHyperResistiveAbs,
        DeclinedJouleWork,
        UnassignableJouleWorkAbs,
        Count
    };
};

// Interleaved per charged species. Powers are W/m^3, CollisionFrequency is
// 1/s and RedirectVarianceRate is J/s per physical ion: over dt it supplies
// sigma_v^2 = dt * RedirectVarianceRate / m_s. No random draw is made here.
struct IonSourceComponent {
    enum : int {
        RelaxationPower,
        RedirectPower,
        RedirectVarianceRate,
        CollisionFrequency,
        Count
    };
};

enum class ViscousSourceKind { None, AppliedWork, Strain };

struct ThermalSourceOptions {
    amrex::Real density_floor = 0.0; // raw electron number density [m^-3]
    amrex::Real time = 0.0;
    // Full physical step, NEVER theta*dt. A supplied stopping impulse is
    // represented by impulse/dt in the stage equation; it is not re-consumed.
    amrex::Real physical_dt = 1.0;
    bool joule = false;
    // Opt-in actual applied edge Jp.Eeta -> physical nodal extensive work.
    // The caller supplies the SAME homogeneous component used by full Ohm.
    // No independent heating eta, overlay, gate/taper or redirect is admitted.
    bool joule_applied_edge_work = false;
    bool relaxation = false;
    bool external_sink = false;
    bool hyperresistive_work = false;
    ViscousSourceKind viscosity = ViscousSourceKind::None;
    bool field_eta_uses_kelvin = false;
    bool use_heating_eta = false;
    amrex::Real joule_density_gate = 0.0;
    bool joule_taper = false;
    amrex::Real source_taper_density = 0.0;
    bool redirect_joule = false;
    amrex::Real redirect_temperature_ev = 0.0;
    amrex::Real redirect_density_floor_factor = 0.0;
    // Parsers must outlive this object. Retain the current unit conventions:
    // field_eta(rho[C/m^3], |J|[A/m^2], t[s]); field_eta_kelvin adds Te[K];
    // heating_eta(rho,|J|,Te[eV],t); relaxation_rate(rho,Te[eV],Ti[eV],t);
    // sink(rho,Te[eV],|B|[T],t) [W/m^3], positive removes electron energy.
    amrex::ParserExecutor<3> field_eta{};
    amrex::ParserExecutor<4> field_eta_kelvin{}, heating_eta{};
    amrex::ParserExecutor<4> relaxation_rate{}, sink{};
    // A configured legacy Te_shunt_threshold alone is not an active shunt:
    // the existing implicit advance never called that post-source transform.
    // Set temperature_shunt only for an explicitly active request.
    // Explicit capability guards. These are accepted state transforms in the
    // legacy lane, not volumetric laws. They require a separate derived stage.
    bool temperature_shunt = false;
    bool pedestal_temperature_cap = false;
    bool redirect_kick_cap = false;
    bool embedded_boundary = false;
    int physical_levels = 1;
};

struct ThermalSourceSpecies {
    amrex::Real charge_number = 1.0; // Z_s > 0; neutral species are omitted
    bool relaxation_excluded = false;
    // rho_s is the RAW deposit, before RZ volume scaling (C/m^2 in RZ).
    // Its dimensionless fraction is rho_s/sum_charged rho_s for a positive
    // sum, zero for an empty sum. The physical density floor acts only on
    // ThermalSourceState::raw_charge, never on this unscaled denominator.
    // Excluded charged species still contribute to the sum. Keep the native
    // materialized-midpoint measure and raw overlay-parser arguments.
    const amrex::MultiFab* raw_charge = nullptr;
    // Pre-gathered fixed old/species Ti [eV] on nodes. Required for relaxation
    // unless excluded. The caller owns its deposit and boundary convention.
    const amrex::MultiFab* ion_temperature_ev = nullptr;
    bool has_resistivity_overlay = false;
    // Only required for a species overlay, on the same Yee grid as total J.
    std::array<const amrex::MultiFab*, 3> raw_current{};
    // eta_s(rho_s RAW, rho physical, Te[K], |J|, |J_s RAW|, |B|, t).
    amrex::ParserExecutor<7> resistivity_overlay{};
};

struct ThermalSourceState {
    const amrex::MultiFab* raw_charge = nullptr;         // physical nodal C/m^3
    const amrex::MultiFab* temperature_kelvin = nullptr; // LIVE nodal trial
    // Optional charge-equivalent heat capacity [C/m^3] for work eligibility.
    // Defaults to raw_charge; supply the current thermal state's capacity
    // when it differs. The raw-density availability gate is never replaced.
    const amrex::MultiFab* heat_capacity_charge = nullptr;
    std::array<const amrex::MultiFab*, 3> plasma_current{}, magnetic_field{};
    // Actual applied Yee fields from HybridPICSolveE's EV_out/EH_out mirrors
    // at THIS trial. Their stencil/coefficients/masks MUST match the force.
    // Never replace these by eta*J or by a scalar Laplacian approximation.
    std::array<const amrex::MultiFab*, 3> viscous_electric{}, hyper_electric{};
    std::array<const amrex::MultiFab*, 3> joule_electric{}; // actual full-Ohm Eeta
    const amrex::MultiFab* viscous_strain_power = nullptr; // live Q_nu [W/m^3]
    const amrex::MultiFab* additive_resistivity = nullptr; // end-region eta
    // Consistent nodal J/m^3 sums, consolidated ONCE outside residuals from
    // collision partial deposits. Evaluation never SumBoundary's or clears it.
    const amrex::MultiFab* stopping_impulse = nullptr;
    std::vector<ThermalSourceSpecies> species;
};

struct ThermalSourceLedger {
    std::array<amrex::Real, SourceComponent::Count> energy{}; // J over dt
    std::vector<amrex::Real> ion_relaxation, ion_redirect;    // J over dt
};

// Pure source assembly on one RZ physical level without EB. Inputs are borrowed
// const views, explicitly refreshed per coupled trial. In a segregated thermal
// solve the caller may freeze species/B/J ONLY for that subsolve; trial T and
// T-dependent applied viscous work must remain live. All owned storage is
// scratch.
class EulerianThermalSources {
  public:
    EulerianThermalSources (const amrex::Geometry& geometry,
                            const amrex::MultiFab& nodal_layout,
                            ThermalSourceOptions options);

    // Outputs are scratch, overwritten including ghosts. Inputs, accepted U/T,
    // particles, RNG and diagnostic sums are never written. false indicates an
    // invalid coefficient/state; discard output. Signed cooling is not clipped.
    // ion_rates needs max(1, species.size()*IonSourceComponent::Count)
    // components.
    bool Evaluate (const ThermalSourceState& state,
                   amrex::MultiFab& nodal_rates, amrex::MultiFab& ion_rates);

    // Stateless acceptance primitive: owner-mask and clipped physical dual
    // volumes match QDSMC's work ledger. Calling this does NOT commit anything.
    // Invoke only for the converged accepted state, then add each returned
    // channel once in the parent's transaction. Eulerian cell rates are
    // obtained with KineticThermalMoments::RestrictNodalScalar; no
    // heat-capacity scaling.
    ThermalSourceLedger Ledger (const amrex::MultiFab& nodal_rates,
                                const amrex::MultiFab& ion_rates,
                                int species_count) const;

    // Borrow the last successful applied-Joule redistribution's exact edge
    // diagnostic. Other matched-work channels reuse this scratch and therefore
    // make this accessor unavailable; no inferred eligibility is exposed.
    std::array<const amrex::MultiFab*,3> JouleWorkScratch() const {
        AMREX_ALWAYS_ASSERT(m_options.joule && m_options.joule_applied_edge_work &&
            m_options.viscosity==ViscousSourceKind::None && !m_options.hyperresistive_work);
        return {m_work[0].get(),m_work[1].get(),m_work[2].get()};
    }

  private:
    void CheckNodal (const amrex::MultiFab* input, int components) const;
    void CheckYee (const std::array<const amrex::MultiFab*, 3>& input) const;
    void MatchedWork (const ThermalSourceState& state,
                      const std::array<const amrex::MultiFab*, 3>& electric,
                      amrex::MultiFab& rates, int component);
    amrex::Real Integral (const amrex::MultiFab& field, int component) const;
    amrex::Geometry m_geometry;
    amrex::BoxArray m_nodes;
    amrex::DistributionMapping m_distribution;
    ThermalSourceOptions m_options;
    amrex::MultiFab m_species_sum, m_work_eligible;
    amrex::MultiFab m_joule_capacity, m_joule_work;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_work;
    std::unique_ptr<amrex::iMultiFab> m_owner;
};

} // namespace warpx::thermal
#endif
