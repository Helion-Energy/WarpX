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

#include "ExternalVectorPotential.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/text/IntervalsParser.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_EdgeFluxRegister.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>

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
     * \brief Advance B (and the intermediate E) from t to t + dt_half on all
     * levels in lockstep: every RK stage advances level 0 through
     * finest_level (coarse first) before the next stage runs. With a single
     * level this reduces exactly to the historic single-level substep loop.
     */
    void BfieldEvolve (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        int step, amrex::Real dt_half, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    void BfieldEvolveRK4 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Vector<std::array<amrex::MultiFab, 3>>& B_old,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    amrex::Real BfieldEvolveRKF45 (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        std::array<amrex::MultiFab, 3>& B_old,
        amrex::Real dt, int lev, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /**
     * \brief One RK stage advance of all levels (E solve then Faraday push).
     *
     * The E phase runs coarse first: a fine level's coarse-fine B ghost fill
     * always reads the coarse solution of the current stage (B is untouched
     * during the E phase). The afterEpush/afterBpush Python callbacks fire
     * once per stage, after the corresponding level loop.
     */
    void FieldPushStage (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        ablastr::fields::MultiLevelVectorField const& Efield,
        ablastr::fields::MultiLevelVectorField const& Jfield,
        ablastr::fields::MultiLevelScalarField const& rhofield,
        amrex::Vector<std::array< std::unique_ptr<amrex::iMultiFab>,3 > >& eb_update_E,
        amrex::Real dt, SubcyclingHalf subcycling_half,
        amrex::IntVect ng, std::optional<bool> nodal_sync);

    /**
     * \brief Fill the fine-level (lev >= 1) B-field ghost cells at the
     * coarse-fine boundary by divergence-free face prolongation
     * (amrex::FaceDivFree) of the lev-1 solution. Interior valid data is not
     * touched; ghost cells covered by same-level valid data are restored by a
     * FillBoundary afterwards; ghost cells outside a non-periodic domain
     * boundary are left to the domain boundary conditions.
     * Cartesian staggered 2D/3D only.
     */
    void FillBfieldCoarseFineGhosts (
        int lev, amrex::IntVect const& ng, std::optional<bool> nodal_sync);

    /**
     * \brief One-time end-of-init overwrite of every fine level's B (valid
     * region AND ghosts) by divergence-free face prolongation from the
     * parent level (active when m_mr_prolong_b_init is set; fresh starts
     * only -- on restart the fine B is the evolved checkpoint state). Both
     * levels initialize B independently (per-level analytic evaluation), so
     * without this the FIRST restriction commit mixes avg(fine analytic)
     * with coarse analytic -- an O(dx^2) static div(B) imprint on the seam
     * ring. FaceDivFree preserves the coarse face fluxes, so after the
     * prolongation avg(fine) == coarse to roundoff and the imprint
     * vanishes. EB masks are respected on both levels: coarse staircase-
     * frozen faces are not sources (the fine init is retained above them)
     * and fine staircase-frozen faces are never overwritten (each level
     * keeps its own frozen staircase solution). Called by WarpX::InitData
     * after all B initialization (analytic init + external-field adds).
     */
    void ProlongBfieldInitFromCoarse ();

    /**
     * \brief Restrict B fine->coarse (face average, cell average for the 2D
     * out-of-plane component) on every coarse level, but only on the
     * coarsened fine region shrunk by m_mr_restrict_setback coarse cells: the
     * fine edge band is sacrificial (its moments are partial by design since
     * buffer particles deposit to the coarse level).
     */
    void RestrictBfieldFineToCoarse (
        amrex::IntVect const& ng, std::optional<bool> nodal_sync);

    /**
     * \brief Fill the fine-level (lev >= 1) ghost cells of rho_fp and
     * current_fp at the coarse-fine boundary from the lev-1 valid data, and
     * overwrite the sacrificial fine-edge band's valid moments with the same
     * coarse-interpolated data (fine deposits there are partial by design --
     * buffer particles deposit to the coarse level, whose moments are
     * complete everywhere -- and evolving the Ohm's-law E with the partial
     * band moments is nonlinearly unstable). Interpolation:
     * bilinear nodal interpolation for rho, a hand-rolled staggering-aware
     * linear interpolation for the edge-staggered current (AMReX offers no
     * edge interpolater through FillPatch). Called once per step after
     * SyncCurrentAndRho and the post-sync FillBoundary sweep, so the
     * time-interpolated moment scratch fields inherit filled ghosts through
     * their LinComb/Copy bookkeeping.
     */
    void FillMomentsCoarseFineGhosts ();

    /**
     * \brief Mesh refinement + embedded boundary clearance guard: for every
     * fine level, no EB cut (partially covered) cell of the parent level may
     * lie within N_clear coarse cells (Linf metric, both sides) of the
     * coarse-fine patch boundary. The coarse-fine interlevel operators
     * (divergence-free B ghost fill, restriction, moment band fill, particle
     * buffers) are EB-blind; a cut face inside their footprint is silent
     * corruption, so a violation aborts with the offending cell and the
     * measured clearance. Fully covered shell cells are legal: both levels
     * freeze E/J there and curl E vanishes, so the transfers are trivially
     * consistent (this makes an EB fully interior to the patch legal).
     * N_clear is derived from the operator footprints (floor 4) or taken
     * from hybrid_pic_model.mr_eb_clearance_cells if >= 0 (0 disables).
     * No-op unless EB is enabled and a fine level exists.
     *
     * \param[in] verbose print the computed clearance and its components
     */
    void CheckMREBClearance (bool verbose) const;

    /** \brief Warn-only check (run once after the initial deposition): flag
     * fine-level cells whose initial ion density is at/below 1.05x the
     * density floor. Fine cells at the floor host a noise-seeded
     * physical-rate instability that mesh refinement amplifies (T1.7). */
    void CheckFineLevelDensityFloor () const;

    /**
     * \brief Seed the full B field of level lev (valid cells and all ghost
     * cells inside the periodically grown domain) by divergence-free face
     * prolongation of the lev-1 solution -- the same operator as
     * FillBfieldCoarseFineGhosts, applied to the whole level instead of the
     * ghost region only. Used when a dynamic regrid has just created (or
     * relocated) level lev. Ghost cells outside a non-periodic domain
     * boundary are left at zero for the domain boundary conditions to fill.
     */
    void SeedBfieldFromCoarse (
        int lev, amrex::IntVect const& ng, std::optional<bool> nodal_sync);

    /**
     * \brief Seed the full E field of level lev (valid + ghost) by
     * staggering-aware linear interpolation of the lev-1 solution. This is a
     * placeholder initialization: E is recomputed from the moments and B in
     * the first Ohm's-law solve after the regrid, so the seeded values are
     * only gathered by particles for the first half-push on the new level.
     */
    void SeedEfieldFromCoarse (int lev);

    /**
     * \brief Final fine->coarse B restriction of level flev before it is
     * removed by a dynamic regrid: the same face-averaged restriction as
     * RestrictBfieldFineToCoarse but WITHOUT the sacrificial setback erosion,
     * so the coarse cells under the (formerly sacrificial) fine edge band are
     * not left holding never-restricted data. Only the outermost shared
     * faces of the patch boundary itself keep the coarse solution.
     */
    void RestrictBfieldBeforeRemoval (
        int flev, amrex::IntVect const& ng, std::optional<bool> nodal_sync);

    /** \brief Drop the cached coarse-fine masks (restriction keep-masks and
     * fine-edge band masks). They are rebuilt on demand and keyed to the
     * BoxArray/DistributionMapping they were built from; dropping them at
     * every hierarchy change keeps a dynamic regrid from ever consulting a
     * stale layout. */
    void ClearMRMaskCache ();

    /** \brief Re-initialize the level-resident external-source data of level
     * lev after a dynamic regrid created or relocated it: re-evaluate the
     * external current (if any) on the level, re-seed the electron
     * temperature fill value, and re-read/re-evaluate the external vector
     * potential and its curl. */
    void ReinitLevelData (int lev);

    /** \brief div(B) audit after the coarse-fine ghost fill of level lev
     * (active when m_mr_check_div_b is set). */
    void CheckDivBAfterGhostFill (int lev, amrex::IntVect const& ng);

    /** \brief div(B) audit of coarse level clev after restriction, split into
     * the restricted interior and the seam/exterior region. */
    void CheckDivBAfterRestriction (int clev);

    /** \brief Print (and reset) the running div(B) audit maxima. */
    void PrintDivBDiagnostics ();

    /**
     * \brief
     * Function to calculate the electron pressure using the simulation charge
     * density. Used in the Ohm's law solver (kinetic-fluid hybrid model).
     */
    void CalculateElectronPressure () const;
    void CalculateElectronPressure (int lev) const;

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

    /** Check if rkf45 should be used */
    [[nodiscard]] bool DoRKF45(int step) const { return m_rkf45_intervals.contains(step); }

    // Declare variables to hold hybrid-PIC model parameters
    /** Number of substeps to take when evolving B (also used as initial substep
     *  count guess when RKF45 adaptive stepping is active) */
    int m_substeps = 10;

    /** Intervals to use RKF45 integrator and to update substeps parameter */
    ablastr::utils::text::IntervalsParser m_rkf45_intervals;
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

    bool m_holmstrom_vacuum_region = false;

    /** Mesh refinement: number of coarse cells by which the fine->coarse B
     *  restriction region is shrunk from the coarse-fine boundary (the fine
     *  edge band is sacrificial). */
    int m_mr_restrict_setback = 2;
    /** Mesh refinement: restrict B fine->coarse after every accepted substep
     *  (true, default) or only once per half-step (false). */
    bool m_mr_restrict_every_substep = true;
    /** Mesh refinement: runtime div(B) audit of the coarse-fine ghost fill
     *  and the restriction (debug diagnostic, off by default). */
    bool m_mr_check_div_b = false;
    /** Mesh refinement + embedded boundary: required clearance (coarse
     *  cells) between any fine-patch boundary and the nearest EB cut cell
     *  (-1 = derive from the coarse-fine operator footprints, 0 = disable
     *  the guard). See CheckMREBClearance(). */
    int m_mr_eb_clearance_cells = -1;
    /** Mesh refinement + embedded boundary: per-dimension clearance waiver
     *  (1 = crossings through patch faces normal to this dimension do not
     *  trigger the clearance guard; the other faces stay guarded). */
    amrex::IntVect m_mr_eb_clearance_waive = amrex::IntVect(0);
    /** Mesh refinement + embedded boundary: gate the divergence-free B ghost
     *  fill by the coarse staircase masks (frozen coarse faces are never a
     *  source; the scratch is pre-seeded with live fine data). Debug knob:
     *  disabling restores the EB-blind prolongation. */
    bool m_mr_eb_gate_prolong = true;
    /** Mesh refinement + embedded boundary: gate the fine->coarse B
     *  restriction commit by the coarse staircase masks (frozen coarse faces
     *  are never a destination). Debug knob: disabling restores the EB-blind
     *  restriction. */
    bool m_mr_eb_gate_restrict = true;
    /** Mesh refinement + embedded boundary: gate the moment band fill by the
     *  fine staircase masks (J writes skipped on frozen fine edges, rho
     *  writes skipped on nodes with no uncovered adjacent cell), so coarse
     *  in-wall moments never overwrite the fine level's own frozen/starved
     *  values. Debug knob: disabling restores the EB-blind band fill. */
    bool m_mr_eb_gate_moments = true;
    /** Mesh refinement: seam EMF matching (Balsara-style edge-EMF flux
     *  register on the restriction commit boundary). Coarse faces sharing a
     *  commit-boundary edge with the restricted region integrate the fine
     *  level's edge-EMF history, so coarse div(B) at the seam is preserved to
     *  machine precision by construction. Default on; disabling restores the
     *  bare (unmatched) seam treatment. */
    bool m_mr_emf_matching = true;
    /** Waived-crossing follow-up to the seam EMF matching, candidate (a),
     *  DEBUG toggle (default off; measured HARMFUL as a default -- see
     *  phase2c notes): skip the restriction commit of keep-faces bounded by
     *  a keep-adjacent coarse edge that is staircase-frozen on the coarse
     *  level but live on the fine level. The skipped faces stay
     *  coarse-evolved; the coarse-vs-fine wall-band solution difference they
     *  then carry against their committed neighbors grows with the drive and
     *  measured LARGER than the frozen/live violation it replaces. Only
     *  active when EMF matching is on, EB is enabled and
     *  mr_eb_clearance_waive_dims is set. */
    bool m_mr_emf_xing_commit_skip = false;
    /** Waived-crossing follow-up, candidate (b), WINNER (default on): freeze
     *  the fine edges exactly overlying keep-adjacent frozen-coarse/live-fine
     *  coarse edges on the fine staircase (fine eb_update_E cleared there;
     *  the fine E and plasma-current values on those edges are zeroed at
     *  every substep start). The committed fine face averages then integrate
     *  a zero EMF history through frozen coarse edges, matching the frozen
     *  faces and the gated register: the wall-band coarse div(B) growth at
     *  waived crossings is eliminated (static init-imprint residual only).
     *  Cost: the fine staircase thickens by the overlying-edge ring. Only
     *  active when EMF matching is on, EB is enabled and
     *  mr_eb_clearance_waive_dims is set (no-op elsewhere: the
     *  frozen-coarse/live-fine edge set is empty without a waived
     *  crossing). */
    bool m_mr_emf_xing_fine_freeze = true;
    /** Mesh refinement: overwrite the fine levels' independently initialized
     *  B by divergence-free prolongation from the coarse level at the end of
     *  init (fresh starts only), so the first restriction commit is exact
     *  and the O(dx^2) static seam-ring div(B) init imprint vanishes. See
     *  ProlongBfieldInitFromCoarse(). */
    bool m_mr_prolong_b_init = true;

    /** Running maxima of the div(B) audit (raw value and value relative to
     *  max|B|/dx), accumulated while m_mr_check_div_b is set. */
    struct DivBStats {
        amrex::Real raw = 0.;
        amrex::Real rel = 0.;
        void update (amrex::Real a_raw, amrex::Real a_rel) {
            raw = std::max(raw, a_raw);
            rel = std::max(rel, a_rel);
        }
    };
    amrex::Vector<DivBStats> m_divb_fine_valid;    /**< fine valid region */
    amrex::Vector<DivBStats> m_divb_fine_ring;     /**< first C-F ghost ring */
    amrex::Vector<DivBStats> m_divb_fine_band;     /**< outer C-F ghost band */
    amrex::Vector<DivBStats> m_divb_fine_wall;     /**< ghost cells in the EB wall band */
    amrex::Vector<DivBStats> m_divb_crse_interior; /**< restricted interior */
    amrex::Vector<DivBStats> m_divb_crse_seam;     /**< seam ring (mixed faces) */
    amrex::Vector<DivBStats> m_divb_crse_exterior; /**< freely evolved exterior */
    amrex::Vector<DivBStats> m_divb_crse_wall;     /**< wall seam (frozen faces mixed in) */
    amrex::Vector<DivBStats> m_divb_crse_wall_all; /**< all wall cells (incl. exterior) */

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

private:
    /** \brief Prolong the lev-1 B solution onto a scratch copy of level lev
     * by divergence-free face interpolation (amrex::FaceDivFree;
     * conservative-linear for the 2D out-of-plane cell-centered component).
     * The scratch MultiFabs are created on the level's layout with ng padded
     * up to coarse alignment and are filled over each fine box grown by that
     * padding; anything beyond a non-periodic domain boundary stays zero.
     * Shared by FillBfieldCoarseFineGhosts (ghost-only copy-back) and
     * SeedBfieldFromCoarse (whole-level copy-back). */
    void ProlongBfieldFromCoarse (
        int lev, amrex::IntVect const& ng, std::array<amrex::MultiFab, 3>& Btmp);

    /** \brief (Re)build the cached cell-centered fine-edge mask of refined
     * level lev used by FillMomentsCoarseFineGhosts: 0 on cells within the
     * sacrificial band (within max(mr_restrict_setback * ratio,
     * n_current_deposition_buffer + 2) cells of the coarse-fine edge), 1 on
     * the trusted fine interior. Cached until the box layout changes. */
    void EnsureFineEdgeMask (int lev, amrex::IntVect const& ratio);

    /** \brief (Re)build the cached cell-centered keep-mask of coarse level
     * clev used by RestrictBfieldFineToCoarse: 1 on the coarsened fine region
     * eroded by m_mr_restrict_setback cells (Linf ball; non-periodic domain
     * boundaries do not erode), 0 elsewhere. Cached until the box layout of
     * either level changes. */
    void EnsureRestrictKeepMask (int clev, amrex::IntVect const& ratio);

    /** \brief (Re)build the cached seam EMF-matching edge-flux register of
     * the (flev-1, flev) level pair. The register is defined on a SYNTHETIC
     * fine BoxArray whose cell union equals the restriction keep region, so
     * its registered edges are exactly the edges of the restriction commit
     * boundary (where committed and freely evolved coarse faces meet). The
     * keep region is derived from the authoritative eroded keep mask
     * (EnsureRestrictKeepMask) as the bounding box of its covered cells and
     * verified bitwise against the mask (single-rectangle keep regions only;
     * any other geometry aborts with instructions to disable
     * hybrid_pic_model.mr_emf_matching). With embedded boundaries the
     * per-direction edge gates and E scratch space for the source gating are
     * (re)built as well. Cached until any box layout or distribution map of
     * the pair changes. */
    void EnsureEmfMatchingRegister (int flev);

    /** \brief Substep-start hook of the seam EMF matching: (re)build the
     * registers if needed and zero their accumulators. */
    void EmfMatchingReset ();

    /** \brief Per-RK-stage hook of the seam EMF matching: accumulate the
     * current stage EMFs (Efield_fp on every level, exactly the fields the
     * stage's Faraday push consumed) into the edge-flux registers with the
     * stage weight wdt = b_s * dt_sub. In 2D the accumulation weight is
     * negated: the out-of-plane WarpX component is Ey while the register
     * models dB/dt = -curl(Ez zhat) with its y axis equal to the WarpX z
     * axis. With embedded boundaries both sources are gated by the coarse
     * staircase edge mask (eb_update_E of the coarse level, valid region
     * only), so the applied delta vanishes identically on frozen edges. */
    void EmfMatchingAccumulate (
        ablastr::fields::MultiLevelVectorField const& Efield, amrex::Real wdt);

    /** \brief Substep-end hook of the seam EMF matching: apply the delta-form
     * circulation correction to the coarse B faces around the registered
     * edges. Must run before the masked restriction commit of the same
     * substep (the restriction overwrites the covered-side keep faces; the
     * correction owns the live-side faces). */
    void EmfMatchingReflux (
        ablastr::fields::MultiLevelVectorField const& Bfield,
        amrex::IntVect const& ng, std::optional<bool> nodal_sync);

    /** Seam EMF-matching state of one coarse-fine level pair (indexed by the
     *  fine level). */
    struct EmfMatchingLevel {
        /** Edge-flux register on the synthetic (keep-region) fine layout. */
        std::unique_ptr<amrex::EdgeFluxRegister> reg;
        /** Synthetic box -> fine-level grid index (same rank by construction). */
        amrex::Vector<int> orig;
        /** Unallocated iteration helper on the synthetic layout (MFIter anchor
         *  for FineAdd). */
        std::unique_ptr<amrex::iMultiFab> iter;
        /** True once the EB source gating of this pair is active. */
        bool gated = false;
        /** Fine-edge gates on the synthetic layout: parent coarse edge's
         *  eb_update_E value (EB source gating). */
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> gate_f;
        /** Gated fine-E stage scratch on the synthetic layout. */
        std::array<std::unique_ptr<amrex::MultiFab>, 3> scratch_f;
        /** Gated coarse-E stage scratch on the coarse layout. */
        std::array<std::unique_ptr<amrex::MultiFab>, 3> scratch_c;
        /** Waived-crossing state: number of registered commit-boundary edges
         *  frozen on the coarse staircase but live on the fine one (box-shared
         *  edges may be counted more than once; only == 0, meaning the
         *  candidates are inert, is load-bearing). */
        amrex::Long n_bad_edges = 0;
        /** Candidate (a): per-direction coarse face masks on the coarse
         *  layout (1 = skip the restriction commit of this face). */
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> commit_skip;
        /** Candidate (b): per-direction fine-edge freeze indicators on the
         *  fine layout (1 = frozen by the candidate). */
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> fine_freeze;
        /** Cache keys: layouts the register was built from. */
        amrex::BoxArray fba, cba;
        amrex::DistributionMapping fdm, cdm;
    };
    /** Per-fine-level seam EMF-matching state. */
    amrex::Vector<EmfMatchingLevel> m_mr_emf;

    /** \brief (Re)build the cached per-direction coarse-face staircase masks
     * of refined level lev on the per-box coarse patch layout used by
     * FillBfieldCoarseFineGhosts (1 = live, 0 = frozen on the coarse
     * staircase, from m_eb_update_B[lev-1]). Cached until the fine box layout
     * changes; only built when EB is enabled and the prolongation gate is on.
     *
     * \param[in] lev fine level
     * \param[in] cpatch_ba cell-centered per-box coarse patch BoxArray
     * \param[in] fdm fine-level DistributionMapping */
    void EnsureProlongGateMask (int lev, amrex::BoxArray const& cpatch_ba,
                                amrex::DistributionMapping const& fdm);

    /** Cached restriction keep-masks, indexed by coarse level. */
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_mr_keep_mask;
    /** Fine-level cell BoxArray each cached keep-mask was built from. */
    amrex::Vector<amrex::BoxArray> m_mr_keep_mask_fba;

    /** Cached prolongation gate masks (per direction, on the coarse patch
     *  layout of FillBfieldCoarseFineGhosts), indexed by fine level. */
    amrex::Vector<std::array<std::unique_ptr<amrex::iMultiFab>, 3>> m_mr_prolong_gate;
    /** Cell BoxArray (coarse patch layout) each cached gate was built from. */
    amrex::Vector<amrex::BoxArray> m_mr_prolong_gate_ba;
    /** Per fine level: true right after a gate rebuild. The first fill after
     *  a (re)build runs unmasked: the fine coarse-fine ghosts hold no live
     *  data yet at init/restart, and a masked fill would freeze zeros into
     *  the retained faces. */
    amrex::Vector<char> m_mr_prolong_gate_fresh;
    /** Cached frozen ghost-fill state (per direction, fine layout, padded
     *  ghosts), captured at the first unmasked fill after a gate rebuild.
     *  Later masked fills pre-seed their scratch from this cache: faces
     *  overlying frozen coarse faces keep exactly this state (each level
     *  owns its own frozen staircase solution), independent of transient
     *  ghost values in Bfield_fp (the RK stage arithmetic manufactures
     *  stale-B_old mixtures on ghost faces between fills). */
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> m_mr_prolong_frozen;

    /** Cached fine-edge masks (0 = sacrificial band), indexed by fine level. */
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> m_mr_fine_edge_mask;
    /** Cell BoxArray each cached fine-edge mask was built from. */
    amrex::Vector<amrex::BoxArray> m_mr_fine_edge_mask_ba;
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
