/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_KINETIC_THERMAL_SPECIES_H_
#define WARPX_KINETIC_THERMAL_SPECIES_H_
#include "EulerianThermalSources.H"
#include <functional>
#include <string>

class MultiParticleContainer;
namespace warpx::thermal {
struct KineticSpeciesDescriptor {
    std::string name;
    amrex::Real charge_number = 1;
    amrex::Real mass = 1; // kg, for the native old-temperature variance
    bool relaxation_excluded = false;
    bool has_resistivity_overlay = false;
    amrex::ParserExecutor<7> resistivity_overlay{};
};
struct KineticSpeciesOptions {
    bool relaxation = false;
    bool single_precision_comms = false;
    bool embedded_boundary = false;
    int physical_levels = 1;
    int azimuthal_modes = 1;
};
enum class MaterializedSpeciesTrial {
    FullParticleState,
    AggregateMassMatrixOnly
};
enum class SpeciesVariancePass { First, Second };
using SpeciesMutableVector = std::array<amrex::MultiFab*, 3>;
struct SpeciesVarianceFields {
    SpeciesMutableVector temperature, weight, variance, velocity_sum;
    std::array<amrex::iMultiFab*, 3> count;
};

/** Owns the source's native raw species context, separate from physical rho.
 * No volume scaling, boundary reflection or density-floor reconstruction is
 * applied to raw deposits. All charged species participate, in native order.
 * A full materialized particle state is required at EVERY residual/probe.
 * An aggregate MM current cannot recover per-species density/current response.
 *
 * Old Ti uses private native double-pass variance, then the legacy Yee K ->
 * CC trace/3 eV -> nodal eV map. CC exterior ghosts are zero, as in the old
 * source. FreezeOldTemperature is once per step, before trial pushes. It does
 * not invoke the registry-backed native variance accumulator.
 *
 * Callbacks receive ONLY owned scratch. Temperature callback scatters one
 * native variance pass; optional filter acts on owned Kelvin Yee fields.
 * Trial callback scatters LOCAL raw midpoint rho and, if requested, native
 * implicit current using saved xn/un. This class sums all guards exactly
 * once after each fresh deposit. Inputs/particles/registry/RNG/tallies must
 * not be modified by callbacks; the production bridge obeys this contract.
 * Failed calls invalidate views. A source without relaxation need not freeze.
 */
class KineticThermalSpecies {
  public:
    using TemperatureDeposit = std::function<bool(
        std::size_t, SpeciesVariancePass, const SpeciesVarianceFields&)>;
    using TemperatureFilter =
        std::function<void(std::size_t, const SpeciesMutableVector&)>;
    using TrialDeposit = std::function<bool(std::size_t, amrex::MultiFab&,
                                            const SpeciesMutableVector&)>;
    KineticThermalSpecies (const amrex::Geometry&, const amrex::BoxArray&,
                           const amrex::DistributionMapping&,
                           amrex::IntVect ghosts,
                           std::vector<KineticSpeciesDescriptor>,
                           KineticSpeciesOptions);
    bool FreezeOldTemperature (const TemperatureDeposit&,
                               const TemperatureFilter& = {});
    bool RefreshMaterializedTrial (MaterializedSpeciesTrial,
                                   const TrialDeposit&);
    const std::vector<ThermalSourceSpecies>& Species () const;
    const std::vector<KineticSpeciesDescriptor>&
    Descriptors () const {
        return m_descriptors;
    }
    const KineticSpeciesOptions&
    Options () const {
        return m_options;
    }
    const amrex::Geometry&
    Geometry () const {
        return m_geometry;
    }
    amrex::IntVect
    Ghosts () const {
        return m_ghosts;
    }
    const amrex::BoxArray&
    Cells () const {
        return m_cells;
    }
    const amrex::DistributionMapping&
    Distribution () const {
        return m_distribution;
    }
    const amrex::MultiFab& OldCellTemperature (std::size_t) const;
    const amrex::MultiFab& OldNodalTemperature (std::size_t) const;

  private:
    struct OwnedSpecies {
        std::unique_ptr<amrex::MultiFab> rho, ti_cell, ti_node;
        std::array<std::unique_ptr<amrex::MultiFab>, 3> current;
    };
    void Sum (amrex::MultiFab&) const;
    void ConvertTemperature (std::size_t);
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells, m_nodes;
    amrex::DistributionMapping m_distribution;
    amrex::IntVect m_ghosts;
    std::vector<KineticSpeciesDescriptor> m_descriptors;
    KineticSpeciesOptions m_options;
    std::vector<OwnedSpecies> m_owned;
    std::vector<ThermalSourceSpecies> m_views;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_temperature, m_weight,
        m_variance, m_velocity_sum;
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> m_count;
    bool m_old_ready = false, m_trial_ready = false;
};

// Native production bridge. Describe returns EVERY charged species, omitting
// neutrals; caller then sets exclusions and overlay parsers before
// construction.
std::vector<KineticSpeciesDescriptor>
DescribeNativeThermalSpecies (const MultiParticleContainer&);
bool FreezeNativeOldSpeciesTemperature (KineticThermalSpecies&,
                                        MultiParticleContainer&);
// implicit_deposition_dt is EXACTLY the dt supplied to
// PhysicalParticleContainer Evolve/DepositCurrent for this stage:
// WarpX::getdt(0) in the current path. The bridge never guesses it or uses the
// public explicit-current overload.
bool RefreshNativeThermalSpecies (KineticThermalSpecies&,
                                  MultiParticleContainer&,
                                  amrex::Real implicit_deposition_dt,
                                  MaterializedSpeciesTrial);
} // namespace warpx::thermal
#endif
