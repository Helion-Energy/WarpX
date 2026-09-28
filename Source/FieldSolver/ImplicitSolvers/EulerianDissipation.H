/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_DISSIPATION_H_
#define WARPX_EULERIAN_DISSIPATION_H_

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronViscosityPoint.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>

namespace warpx::thermal {
using DissipationVector = std::array<const amrex::MultiFab*, 3>;
// Open means the legacy FieldBoundaryType::None (no electric projection).
// PMC is the homogeneous normal-odd/tangential-even electric response.
enum class DissipationBoundary { Axis, PEC, Periodic, Open, PMC };
enum class HyperMode { Off, Laplacian, InteriorCurlCurl, AmpereCurlCurl };
struct TrialDissipationOptions {
    bool viscosity = false;
    ElectronViscosityParams viscous;
    HyperMode hyper = HyperMode::Off;
    // Same production convention: physical UNFLOORED rho [C/m3], |B| [T].
    amrex::ParserExecutor<2> eta_h{};
    bool eta_h_depends_on_B = false;
    std::array<std::array<DissipationBoundary, 2>, AMREX_SPACEDIM> boundary{};
    bool embedded_boundary = false;
    int physical_levels = 1;
    int azimuthal_modes = 1;
    // Raw additive fields are not their applied response after either solve.
    bool transformed_electric_solve = false;
};
struct TrialDissipationState {
    const amrex::MultiFab* charge = nullptr; // nodal physical rho, comp 0
    const amrex::MultiFab* temperature_kelvin = nullptr; // LIVE nodal Te
    DissipationVector plasma_current{}, ion_current{}, magnetic_field{};
};

/** Pure current-trial viscosity/hyperresistivity preparation.
 * One zero-based RZ Yee level, m=0, no EB. Existing 3D paths are untouched.
 * All inputs are borrowed const with one already-physical ghost layer; their
 * boundary moment closures must be the SAME ones consumed by the Ohm solve.
 * Copies synchronize internal seams without changing any input or registry.
 * Parser owners must outlive this object. No accepted fields, RNG or tallies.
 *
 * ViscousField/HyperField are homogeneous electric-boundary responses of
 * additive terms, in V/m. PEC tangential increments vanish, normal ghosts
 * reflect evenly. PMC normal ghosts reflect oddly and tangential ghosts
 * evenly; unconstrained tangential cap values are retained. Axis parity is
 * enforced. The current/deposition images are supplied by the caller and
 * need not equal the electric images (e.g. reflecting ions at a PEC wall).
 * A resistive-shell affine wall
 * value cancels between full/no-term evaluations at the SAME B; its increment
 * has this same homogeneous PEC response. Its physical wall work belongs to
 * the shell ledger, not to viscosity/hyperresistivity. RawHyperField exposes
 * the pre-projection kernel value for audit; never book it after a projection.
 *
 * Apply these EXACT fields in the new-mode Ohm residual, and pass the SAME
 * fields to EulerianThermalSources. Do not recompute a separate force using
 * legacy frozen temperature/B/current. Tensor/curlcurl electric solves and
 * other boundary types are guarded; their response must first be derived.
 */
class EulerianDissipation {
  public:
    EulerianDissipation (const amrex::Geometry& geometry,
                         const amrex::BoxArray& cells,
                         const amrex::DistributionMapping& distribution,
                         TrialDissipationOptions options);
    // Scratch is overwritten. false means invalid state/coefficient: discard
    // it. A/B/A evaluations are deterministic; no accumulated clamp counts.
    bool Evaluate (const TrialDissipationState& state);
    const amrex::MultiFab&
    NodalVelocity () const {
        return m_velocity;
    }
    const amrex::MultiFab&
    Stress () const {
        return m_stress;
    } // xx,yy,zz,xy,xz,yz [Pa]
    const amrex::MultiFab&
    NodalDrag () const {
        return m_drag;
    } // V/m
    const amrex::MultiFab&
    StrainPower () const {
        return m_strain;
    } // W/m3
    const amrex::MultiFab&
    CoefficientClampMask () const {
        return m_clamps;
    }
    DissipationVector ViscousField () const;
    DissipationVector HyperField () const;
    DissipationVector RawHyperField () const;
    // Pure homogeneous projection, also useful to verify the caller's field
    // composition. Only matching owned/caller SCRATCH arrays may be passed.
    void ProjectElectricIncrement (std::array<amrex::MultiFab*, 3> field) const;

  private:
    void CheckInput (const amrex::MultiFab* field,
                     const amrex::BoxArray& expected, int components) const;
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells, m_nodes;
    amrex::DistributionMapping m_distribution;
    TrialDissipationOptions m_options;
    amrex::MultiFab m_rho, m_temperature, m_velocity, m_stress, m_drag,
        m_strain, m_clamps, m_valid;
    std::array<amrex::MultiFab, 3> m_current, m_ion, m_magnetic, m_viscous,
        m_hyper, m_hyper_raw, m_curl;
};
} // namespace warpx::thermal
#endif
