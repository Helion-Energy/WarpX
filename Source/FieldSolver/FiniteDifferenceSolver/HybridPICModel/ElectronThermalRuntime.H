/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ELECTRON_THERMAL_RUNTIME_H_
#define WARPX_ELECTRON_THERMAL_RUNTIME_H_
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "FieldSolver/ImplicitSolvers/ExpectedIonExchangeSource_K.H"
#include <iosfwd>
#include <AMReX_ParmParse.H>
#include <map>
#include <string>

namespace warpx::thermal {
enum class ElectronEnergyMode { Legacy, DecoupledJFNK, CoupledJFNK };
ElectronEnergyMode ParseElectronEnergyMode (const amrex::ParmParse& pp,
                                            bool& solve);
const char* ElectronEnergyModeName (ElectronEnergyMode mode);
struct ElectronEnergyRuntimeConfig {
    bool darwin = false;
    std::string evolve_scheme;
    int max_level = 0;
    bool embedded_boundary = false;
    bool corrected_axis = true;
    amrex::Real gamma = 5.0 / 3.0;
    amrex::Real number_floor = 1.;
    amrex::Real representation_floor = 1.;
    amrex::Real smooth_floor_width = 0.;
};
void ValidateElectronEnergyRuntime (ElectronEnergyMode mode,
                                    const ElectronEnergyRuntimeConfig& config);

// Borrowed immutable solver views. The owner must clear before their lifetime
// ends. Accepted fields are only layout/ghost contracts, never mutated here.
class ElectronThermalTrialViews {
  public:
    void Set (int lev, const amrex::MultiFab& temperature,
              const amrex::MultiFab& pressure,
              const amrex::MultiFab& accepted_temperature,
              const amrex::MultiFab& accepted_pressure);
    void
    Clear (int lev) {
        m_views.erase(lev);
    }
    void
    Clear () {
        m_views.clear();
    }
    bool
    Has (int lev) const {
        return m_views.count(lev) != 0;
    }
    const amrex::MultiFab& Temperature (int lev,
                                        const amrex::MultiFab& accepted) const;
    const amrex::MultiFab& Pressure (int lev,
                                     const amrex::MultiFab& accepted) const;

  private:
    struct View {
        const amrex::MultiFab* temperature;
        const amrex::MultiFab* pressure;
    };
    std::map<int, View> m_views;
};

// Resolved once before restart validation and shared with the lazy stage driver.
// For active Eulerian relaxation, Resolve defaults to population_bounded_nr;
// absent relaxation remains off. Explicit off is a diagnostic/reference law.
// The per-ion NR accuracy budget is unchanged and remains a live rejection gate.
// This is the source-partner/endpoint contract, not a fingerprint of every
// transport, parser or collision setting. Inactive accuracy inputs canonicalize
// to the helper defaults. The optional reference only changes diagnostics and
// may reject an inaccurate reference; it never changes an accepted Q or kick.
struct ElectronThermalModelConfig {
    std::string relaxation_partner = "off";
    std::string expected_ou_convention = "off";
    ExpectedIonEnergyOptions expected_options;
    bool relativistic_reference = false;
    // Raw species moments share their deposition units; positive-sum
    // fractions never divide by a physical density floor. Bind this change
    // so an old checkpoint cannot silently resume with different source power.
    std::string source_semantics = "midpoint_raw_fraction_positive_sum_fixed_old_ti_endpoint_ou_v2";
    std::string endpoint_absorption_policy = "endpoint_absorption_frozen_cell_temperature";
    // Bump when the currently unqualified boundary-history split is changed.
    std::string boundary_history_semantics = "unqualified_endpoint_absorption_split_v1";

    static ElectronThermalModelConfig Resolve (const amrex::ParmParse& pp,
                                                bool relaxation,
                                                bool joule_redirect);
    bool PopulationPartner () const { return relaxation_partner == "population_bounded_nr"; }
    bool ExpectedAudit () const { return expected_ou_convention != "off"; }
    void Validate () const;
    std::string EncodePhysicalContract () const;
    static ElectronThermalModelConfig DecodePhysicalContract (std::istream& in);
    void ValidateRestart (const ElectronThermalModelConfig& requested) const;
};

struct ElectronEnergyMetadata {
    ElectronEnergyMode mode = ElectronEnergyMode::Legacy;
    amrex::Real gamma = 5.0 / 3.0;
    amrex::Real number_floor = 1.0;
    bool corrected_axis = true;
    ElectronThermalModelConfig thermal_model;
    std::string Encode () const;
    static ElectronEnergyMetadata Decode (const std::string& text);
    void ValidateRestart (const ElectronEnergyMetadata& requested) const;
};

struct InitialElectronProfile {
    amrex::Real temperature_kelvin = 0.;
    amrex::Real reference_density = 1.;
    amrex::Real gamma = 1.;
    amrex::Real number_floor = 1.;
    bool include_pedestal = true;
};

// Physical nodal product first, then the conservative restriction. In general
// R(n*T) != R(n)*R(T). No accepted field except the explicit U output is
// written.
bool SeedCellElectronEnergy (KineticThermalMoments& moments,
                             KineticThermalStateView density,
                             const InitialElectronProfile& profile,
                             amrex::Real gamma, amrex::MultiFab& energy);
} // namespace warpx::thermal
#endif
