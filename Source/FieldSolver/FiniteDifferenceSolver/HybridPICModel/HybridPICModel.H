/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_HYBRIDPICMODEL_H_
#define WARPX_HYBRIDPICMODEL_H_

#include "HybridPICModel_fwd.H"

#include "Fields.H"

#include "EBJBoundary.H"
#include "ExternalVectorPotential.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>
#include <utility>

/**
 * \brief This class contains the parameters needed to evaluate hybrid field
 * solutions (kinetic ions with fluid electrons).
 */
class HybridPICModel
{
public:
    HybridPICModel ();

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters ();

    /** Allocate hybrid-PIC specific multifabs. Called in constructor. */
    void AllocateLevelMFs (
        ablastr::fields::MultiFabRegister & fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        int ncomps,
        const amrex::IntVect& ngJ,
        const amrex::IntVect& ngRho,
        const amrex::IntVect& ngEB,
        const amrex::IntVect& jx_nodal_flag,
        const amrex::IntVect& jy_nodal_flag,
        const amrex::IntVect& jz_nodal_flag,
        const amrex::IntVect& rho_nodal_flag,
        const amrex::IntVect& Ex_nodal_flag,
        const amrex::IntVect& Ey_nodal_flag,
        const amrex::IntVect& Ez_nodal_flag,
        const amrex::IntVect& Bx_nodal_flag,
        const amrex::IntVect& By_nodal_flag,
        const amrex::IntVect& Bz_nodal_flag
    ) const;

    void InitData (const ablastr::fields::MultiFabRegister& fields);

    /**
     * \brief Apply the collocated conformal magnetic-field wall treatment to the
     * INITIAL B field (call once after the initial grid B has been set). The
     * direct mirror fill sets the covered band to its Neumann value so the first
     * field-push substep does not see it jump from the staircase-masked zero
     * (which stalls the RKF45 substepping). A no-op when the wall fill is
     * disabled (conformal_b_off) or off the collocated embedded-boundary path.
     */
    void InitialBEBFill ();

    /**
     * \brief
     * Function to evaluate the external current expressions and populate the
     * external current multifab. Note the external current can be a function
     * of time and therefore this should be re-evaluated at every step.
     */
    void GetCurrentExternal ();

    /**
     * \brief
     * Function to calculate the total plasma current based on Ampere's law while
     * neglecting displacement current (J = curl x B). Any external current is
     * subtracted as well. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     *
     * \param[in] Bfield       Magnetic field from which the current is calculated.
     * \param[in] eb_update_E  Indicate in which cell J should be calculated (related to embedded boundaries).
     */
    void CalculatePlasmaCurrent (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E
    ) const;
    void CalculatePlasmaCurrent (
        ablastr::fields::VectorField const& Bfield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev
    ) const;

    /**
     * \brief
     * Function to update the E-field using Ohm's law (hybrid-PIC model).
     */
    void HybridPICSolveE (
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, bool solve_for_Faraday) const;

    void HybridPICSolveE (
        ablastr::fields::VectorField const& Efield,
        ablastr::fields::VectorField const& Jfield,
        ablastr::fields::VectorField const& Bfield,
        amrex::MultiFab const& rhofield,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update_E,
        int lev, PatchType patch_type, bool solve_for_Faraday) const;

    /**
     * \brief Optional Marder-like diffusive divergence clean of a nodal vector
     * field. Iterates field += alpha_scaled * grad(div(field)) restricted to
     * a near-wall geometric band (distance_to_eb <= band_cells*h), re-applying
     * the embedded-boundary PEC fill (with the field's own parity) each sweep.
     * The correction is a pure gradient, so curl(field) -- and hence J=curl(B) /
     * the physics it feeds -- is unchanged to round-off away from the band
     * cutoffs; only the spurious curved-wall divergence injected by the mirror
     * fill is dissipated. With the inner cutoffs at 0 (the wall-layer mode) the
     * clean reaches the first fluid layers, where a sharp re-entrant corner
     * (e.g. a wall radius step) otherwise pumps an exponentially growing
     * div(B). No-op when alpha<=0 or no EB. 3D/XZ Cartesian only (the gradient
     * is the adjoint of the Cartesian divergence; ReadParameters aborts if the
     * knobs are set on other geometries).
     */
    void MarderCleanDivergence (
        ablastr::fields::VectorField const& field,
        std::array< std::unique_ptr<amrex::iMultiFab>,3 >& eb_update,
        warpx::hybrid::EBFillStatus* status_cache,
        bool normal_odd,
        bool fill_covered_centers,
        amrex::Real alpha,
        int max_iters,
        amrex::Real clean_band_cells,
        amrex::Real fill_band_cells,
        int lev) const;

    /** Once-per-step driver for the div(B)/div(J_total) clean: cleans Bfield_fp
     *  (magnetic parity) and the total Ampere current hybrid_current_fp_plasma
     *  (electric parity) on every level via MarderCleanDivergence. Called from
     *  the hybrid field advance after the plasma-current update and before the
     *  external-field add-back, so it acts on the plasma field only. No-op
     *  unless the respective alpha>0. */
    void MarderCleanFieldsPerStep () const;

    void BfieldEvolveRK (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRK (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt_half, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void FieldPush (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /**
     * \brief
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;

    /**
     * \brief Warn (once) if plasma sits within the particle shape order of the
     * embedded-boundary wall. Deposition reaches nox cells, so plasma that close
     * lays current onto wall-adjacent edges and drives near-wall current spikes
     * (and an unmodelled sub-grid sheath). Recommends scraping >= nox*dx off the
     * wall. Cheap: runs a single masked reduction on the first call, then latches.
     */
    void WarnIfPlasmaAgainstWall (int lev) const;

    /**
     * \brief Fill the electron pressure multifab given the kinetic particle
     * charge density (and assumption of quasi-neutrality) using the user
     * specified electron equation of state.
     *
     * \param[out] Pe_field scalar electron pressure MultiFab at a given level
     * \param[in] rho_field scalar ion charge density Multifab at a given level
     */
    void FillElectronPressureMF (
        amrex::MultiFab& Pe_field,
        amrex::MultiFab const& rho_field ) const;

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
    int m_substeps = 10;

    /** Relative tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_rtol = 1.e-4;
    /** Absolute tolerance for RKF45 adaptive substep error control */
    amrex::Real m_substep_atol = 1.e-8;
    /** Safety factor for RKF45 step-size adjustment */
    amrex::Real m_substep_safety = 0.9;
    /** Maximum step-size growth factor per accepted step */
    amrex::Real m_substep_max_growth = 5.0;
    /** Maximum number of substep attempts (accepted + rejected) before abort */
    int m_max_substep_attempts = 250;
    /** If true, use the adaptive RKF45 integrator; if false, use fixed-step RK4 */
    bool m_use_rkf45 = false;

    bool m_holmstrom_vacuum_region = false;


    /** Electron temperature in eV */
    amrex::Real m_elec_temp;
    /** Reference electron density */
    amrex::Real m_n0_ref = 1.0;
    /** Electron pressure scaling exponent */
    amrex::Real m_gamma = 5.0/3.0;

    /** Plasma density floor - if n < n_floor it will be set to n_floor */
    amrex::Real m_n_floor = 1.0;

    /** Plasma resistivity */
    std::string m_eta_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_resistivity_parser;
    amrex::ParserExecutor<3> m_eta;
    bool m_resistivity_has_J_dependence = false;

    /** Plasma hyper-resisitivity */
    std::string m_eta_h_expression = "0.0";
    std::unique_ptr<amrex::Parser> m_hyper_resistivity_parser;
    amrex::ParserExecutor<2> m_eta_h;
    bool m_include_hyper_resistivity_term = false;
    bool m_hyper_resistivity_has_B_dependence = false;
    /** If true (opt-in), evaluate the hyper-resistivity Laplacian with the
     *  isotropic Mehrstellen (2D) / Patra-Karttunen (3D) stencils instead of
     *  the cross stencil, removing the cos(4*theta) anisotropy of its
     *  damping rate (fully isotropic on cubic cells; Cartesian only). */
    bool m_isotropic_hyper_resistivity = false;
    /** If true (opt-in), add an isotropizing corner-curl correction to the
     *  resistive electric field so the emergent in-plane B diffusion uses the
     *  isotropic (Mehrstellen) Laplacian instead of the cross stencil. This
     *  removes the cos(4*theta) modulation of the resistive damping rate that
     *  drives a grid m=4 mode. Applied through E, so the standard Faraday
     *  curl is unchanged and div(B) stays exactly zero. Cartesian only (the
     *  RZ resistive operator is axisymmetric and has no such anisotropy). */
    bool m_isotropic_resistivity = false;

    /** External current */
    std::string m_Jx_ext_grid_function = "0.0";
    std::string m_Jy_ext_grid_function = "0.0";
    std::string m_Jz_ext_grid_function = "0.0";
    std::array< std::unique_ptr<amrex::Parser>, 3> m_J_external_parser;
    std::array< amrex::ParserExecutor<4>, 3> m_J_external;
    bool m_has_external_current = true;
    bool m_external_current_has_time_dependence = false;

    /** External E/B fields */
    bool m_add_external_fields = false;
    std::unique_ptr<ExternalVectorPotential> m_external_vector_potential;

    /** If true, use the conformal embedded-boundary Faraday update (enlarged
     *  cell technique, Xiao and Liu, IEEE MWCL 2004) for the B-field push
     *  instead of the stair-step approximation. Requires embedded boundaries
     *  and a 3D Cartesian staggered grid. */
    bool m_use_conformal_eb = false;
    /** If true (default), the hybrid Hall term's nodal J x B interpolation excludes
     *  covered edges/faces (EB-aware, mask-renormalized) so covered-cell field values do
     *  not pollute the near-wall J x B. Set false to recover the plain (polluting)
     *  averaging for A/B. Only affects conformal-EB staggered runs. */
    bool m_eb_hall_mask = true;
    /** If true, skip the collocated conformal magnetic-field wall treatment
     *  entirely (no EB B fill): the unmasked nodal Faraday curl is left as-is.
     *  This reproduces the pre-treatment baseline and isolates the wall fill as
     *  the source of the grid-locked m = 4 perturbation. Diagnostic only -- it
     *  leaves spurious fields at the wall, so it is not for production runs. */
    bool m_conformal_b_off = false;
    /** If true, apply the collocated-style direct level-set mirror wall fill to the
     *  STAGGERED (Yee) Bfield_fp after each Faraday push, in place of the ECT
     *  enlarged-cell face borrowing. Run with the standard masked Yee Faraday
     *  (use_conformal_eb off) + this straight mirror to get the collocated wall
     *  treatment on a staggered grid, avoiding the ECT face-extension / cross-box
     *  seam. Opt-in (default off -> byte-identical); intended for staggered runs
     *  with the plasma held off the wall (a standoff), which removes the dense
     *  near-wall radial B that made the pointwise Yee mirror spike. */
    bool m_eb_b_straight_mirror = false;
    /** If true, apply the cylindrical (surface-of-revolution) radial metric
     *  correction to the embedded-boundary mirror fill of E, J and B: the
     *  radial and azimuthal parts of the reflected field are scaled by the
     *  radial Jacobian r_image/r_fill -- the leading O(h/R) curved-wall
     *  correction the planar mirror misses. Opt-in; requires use_conformal_eb
     *  and a 3D Cartesian grid; the wall must be a surface of revolution about
     *  m_eb_cyl_axis, centered on the transverse origin. */
    bool m_eb_cylindrical_correction = false;
    /** Symmetry axis (0=x, 1=y, 2=z) for m_eb_cylindrical_correction. */
    int m_eb_cyl_axis = 2;

    /** Relative residual tolerance of the Jacobi band relaxation of the
     *  embedded-boundary PEC field fills (only used when eb_bc_direct_fill
     *  is false) */
    amrex::Real m_eb_bc_rtol = 1.e-4;
    /** Sweep cap of the embedded-boundary PEC field fills: maximum Jacobi
     *  iterations of the band relaxation, and maximum cascade sweeps over
     *  ill-posed targets in the direct fill */
    int m_eb_bc_max_iters = 10;
    /** If true (default), the embedded-boundary PEC field fills use a
     *  single-pass mirrored interpolation over unmasked (solution-domain)
     *  values; if false, the iterative Jacobi band relaxation */
    bool m_eb_bc_direct_fill = true;
    /** Image parity of the embedded-boundary deposit fold: true (default)
     *  folds with the PEC image signs (charge and tangential current
     *  subtracted, normal current added); false uses the reflecting-wall
     *  parities (mass-conserving for a wall-supported column) */
    bool m_eb_fold_pec = true;
    /** Parity of the embedded-boundary rho mirror fill: true (default) is
     *  the odd reflection (Dirichlet 0 at the wall); false is the even
     *  reflection (Neumann: the wall supports the column) */
    bool m_eb_rho_dirichlet = true;
    /** Parity of the electron-pressure embedded-boundary fill at a PEC wall:
     *  true (default) is the odd reflection (Dirichlet 0 -> Pe vanishes at the
     *  wall, so grad(Pe) supports the allowable radial E a PEC sustains via
     *  surface charge); false is the even reflection (Neumann: zero normal
     *  gradient). A physical sheath/wall-function is not modelled here. */
    bool m_eb_pe_dirichlet = true;
    /** Latch so the near-wall plasma warning is emitted at most once. */
    mutable bool m_checked_plasma_at_wall = false;

    /** Cached embedded-boundary fill classification for the deposited edge
     *  fields (ion current fill + deposit fold), one entry per level; reset on
     *  regrid. Built at the one-cell band: the fold's covered gather set must
     *  equal this fill's write set, and both use the fixed one-cell geometry.
     *  Mutable: the cache is (re)built inside const solver calls. */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_E;
    /** Cached EB fill classification for the Ohm's-law electric field, one
     *  entry per level; reset on regrid. Separate from m_eb_bc_status_E
     *  because the E fill band is pinned to the (possibly widened) plasma
     *  current band m_eb_fill_band_cells, while the deposited-current cache
     *  stays at the one-cell band -- an EBFillStatus cache must never be
     *  shared between calls with different band widths (the first caller
     *  would pin the classification). */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_Eohm;
    /** Cached embedded-boundary fill classification for the recovered
     *  magnetic field, one entry per level; reset on regrid */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_B;
    /** Cached EB fill classification for the Ampere plasma current, one entry
     *  per level; reset on regrid. Separate from m_eb_bc_status_E because the
     *  isotropic hyper-resistivity Laplacian reads the plasma current at its
     *  diagonal/corner neighbors, so this fill widens the mirror band
     *  (m_eb_fill_band_cells) -- a wider classification than the edge-field
     *  cache, which must stay at the one-cell band shared with the deposit
     *  fold. Only used when isotropic hyper-resistivity is enabled. */
    mutable amrex::Vector<warpx::hybrid::EBFillStatus> m_eb_bc_status_Jplasma;
    /** Mirror-fill band width (in cells) for the plasma-current EB fill, and --
     *  PINNED to it -- the electric-field (Efield_fp) EB fill. Value:
     *  sqrt(AMREX_SPACEDIM) when the isotropic hyper-resistivity Laplacian is
     *  enabled (so its diagonal/corner cut edges are mirror-filled, not zeroed),
     *  else 1. The E fill is tied to this same band because in the cut/covered
     *  region E = eta*J: filling E in cells BEYOND where J is filled leaves E != 0
     *  where J = 0, an inconsistent source that the curl(E)->B->curl(B)->J->E loop
     *  amplifies into a non-finite blow-up. So the E fill band must not exceed the
     *  J fill band; pinning them equal keeps E and J consistent at the wall. */
    amrex::Real m_eb_fill_band_cells = 1.0;
    /** Mirror-fill band width (in cells) for the magnetic-field (Bfield_fp) EB
     *  fill. The filled B couples back into the solution only through curl/B
     *  reads; with the isotropic corner-curl E correction on, that composite
     *  reach is 2 cells, so widening this band to >=3 pushes the level-set
     *  mirror's div(B) jump out of the solution-coupling zone (into the zeroed
     *  deep interior, where it is causally inert). Runtime knob
     *  hybrid_pic_model.eb_b_fill_band_cells; default 1 (legacy behavior),
     *  auto-widened to sqrt(2) on 3D/XZ when isotropic_resistivity is enabled
     *  so the corner-curl's in-plane diagonal B taps read mirror values rather
     *  than the zeroed deep interior (user query still overrides). */
    amrex::Real m_eb_b_fill_band_cells = 1.0;

    /** Marder-like diffusive div(B) clean coefficient (fraction of the explicit
     *  grad(div) CFL cap; stable below ~1/6 in 3D, PoC-validated sweet spot
     *  ~0.15). 0 (default) disables. Runtime knob hybrid_pic_model.divb_clean_alpha. */
    amrex::Real m_divb_clean_alpha = 0.0;
    /** Same for the total Ampere current hybrid_current_fp_plasma; dissipates the
     *  spurious near-wall div(J_total) the pointwise mirror injects (see
     *  MarderCleanDivergence). 0 (default) disables. */
    amrex::Real m_divj_clean_alpha = 0.0;
    /** Number of grad(div) sweeps per application of the divergence clean. */
    int m_divb_clean_iters = 5;
    /** Outer cutoff of the divergence-clean band, in cells from the wall. */
    amrex::Real m_divb_clean_band_cells = 4.0;
    /** Inner cutoff (in cells) below which the computed divergence is dropped.
     *  Default 1 trusts the divergence only where its own +/-1 stencil is fully
     *  in the fluid; 0 keeps it on every uncovered node (wall-layer mode). */
    amrex::Real m_divb_clean_inner_div_cells = 1.0;
    /** Inner cutoff (in cells) below which no correction is applied. Default 2
     *  keeps the full +/-2 grad(div) stencil in the fluid (order-preserving on a
     *  smooth wall); 0 corrects every uncovered band node (wall-layer mode, for
     *  a re-entrant-corner divergence instability). */
    amrex::Real m_divb_clean_inner_corr_cells = 2.0;

    /** Gpu Vector with index type of the Jx multifab */
    amrex::GpuArray<int, 3> Jx_IndexType;
    /** Gpu Vector with index type of the Jy multifab */
    amrex::GpuArray<int, 3> Jy_IndexType;
    /** Gpu Vector with index type of the Jz multifab */
    amrex::GpuArray<int, 3> Jz_IndexType;
    /** Gpu Vector with index type of the Bx multifab */
    amrex::GpuArray<int, 3> Bx_IndexType;
    /** Gpu Vector with index type of the By multifab */
    amrex::GpuArray<int, 3> By_IndexType;
    /** Gpu Vector with index type of the Bz multifab */
    amrex::GpuArray<int, 3> Bz_IndexType;
    /** Gpu Vector with index type of the Ex multifab */
    amrex::GpuArray<int, 3> Ex_IndexType;
    /** Gpu Vector with index type of the Ey multifab */
    amrex::GpuArray<int, 3> Ey_IndexType;
    /** Gpu Vector with index type of the Ez multifab */
    amrex::GpuArray<int, 3> Ez_IndexType;
};

/**
 * \brief
 * This struct contains only static functions to compute the electron pressure
 * using the particle density at a given point and the user provided reference
 * density and temperatures.
 */
struct ElectronPressure {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real get_pressure (amrex::Real const n0,
                                     amrex::Real const T0,
                                     amrex::Real const gamma,
                                     amrex::Real const rho) {
        return n0 * T0 * std::pow((rho/PhysConst::q_e)/n0, gamma);
    }
};

#endif // WARPX_HYBRIDPICMODEL_H_
