/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EULERIAN_SOURCE_DIAGONAL_H_
#define WARPX_EULERIAN_SOURCE_DIAGONAL_H_
#include "EulerianThermalSources.H"
namespace warpx::thermal {
struct SourceDiagonalOptions {
    amrex::Real relative_temperature_step = 2.e-5;
    amrex::Real absolute_temperature_step_kelvin = 1.e-6;
};
// A frozen PARTIAL derivative: density, species, old Ti, currents and B are
// fixed. Only actual pointwise Joule+relaxation laws enter this PC surrogate.
// Other source derivatives and all field/species responses remain in the full
// physical residual/Jv. Parser owners must outlive this object.
class EulerianSourceDiagonal {
  public:
    EulerianSourceDiagonal (const amrex::Geometry&,
                            const amrex::MultiFab& nodal_layout,
                            ThermalSourceOptions, SourceDiagonalOptions = {});
    // Pure scratch finite differences of the existing source evaluator. No
    // model temperature registry, RNG, particles, accepted fields or tallies.
    // n is the CELL number capacity including pedestal, gamma>1. This matches
    // current KineticThermalMoments' Tcell=(gamma-1)*U/(kb*n), ordinary
    // cell->node average with even/periodic images, and physical Q restriction.
    // Known redirect and nu-argument-floor kinks zero ONLY the affected
    // channel's surrogate slope. Arbitrary parser nonsmoothness is not
    // inferred. false means unavailable PC data, NOT a changed/invalid physical
    // source; caller may use a zero source derivative or decline PC
    // construction.
    bool Freeze (const ThermalSourceState&,
                 const amrex::MultiFab& cell_capacity, amrex::Real gamma);
    // Components Joule,Relaxation,Total [W/(m^3 K)].
    const amrex::MultiFab& NodalDerivative () const;
    // Components Joule,Relaxation; 1 denotes omitted known kink neighborhood.
    const amrex::MultiFab& KinkMask () const;
    // The same-cell diagonal dQ_cell/dU_cell [1/s] for the frozen local map.
    const amrex::MultiFab& RawDiagonal () const;
    // min(0,RawDiagonal). Positive feedback is deliberately omitted so the
    // existing PC n*(1-h*d_pc) >= n for h>=0. Physical Q/Jv is NEVER clipped.
    const amrex::MultiFab& PCDiagonal () const;

  private:
    amrex::Geometry m_geometry;
    amrex::BoxArray m_nodes, m_cells;
    amrex::DistributionMapping m_distribution;
    ThermalSourceOptions m_source_options;
    SourceDiagonalOptions m_options;
    EulerianThermalSources m_source;
    amrex::MultiFab m_plus_temperature, m_minus_temperature, m_span;
    amrex::MultiFab m_plus_rates, m_minus_rates, m_derivative, m_kinks, m_raw,
        m_pc;
    std::unique_ptr<amrex::MultiFab> m_ion_rates;
    bool m_valid = false;
};
} // namespace warpx::thermal
#endif
