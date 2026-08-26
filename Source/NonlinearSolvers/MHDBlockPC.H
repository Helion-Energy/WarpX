/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_BLOCK_PC_H_
#define MHD_BLOCK_PC_H_

#include "Fields.H"
#include "MHDResistiveBanded.H"
#include "MHDResistiveDirect.H"
#include "MHDResistiveStencil.H"
#include "MHDWaveOp.H"
#if defined(WARPX_DIM_RZ)
#include "MHDWaveOpRZ.H"
#endif
#include "Preconditioner.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <limits>
#include <memory>
#include <string>

/**
 * \brief Physics-based block preconditioner for the theta-implicit MHD state.
 *
 * The electric predictor and corrector approximate resistive and Hall
 * magnetic stiffness with
 *
 *     [I + d_* curl curl] dE = bE,
 *
 *     d_* = hypot(h eta_ref / mu0,
 *                 h |B_ref| / (mu0 rho_q,ref)).
 *
 * The second term is enabled only when Hall physics is active. It is the
 * spectral-magnitude approximation to the skew Hall/whistler block: for a
 * homogeneous one-dimensional circular mode the exact transverse field
 * symbol is I +/- i d_H k^2, while this fixed-work scalar approximation is
 * I + d_H k^2. It bounds the preconditioned high-k spectrum without changing
 * the nonlinear residual.
 *
 * For a nonzero domain-mean magnetic field with Hall physics disabled, the
 * fluid block applies the constant-coefficient Chacon small-flow wave Schur
 * approximation
 *
 *     [I - h^2 c_s^2 grad div
 *        + h^2/(mu0 rho_ref) W(B_ref)] dM = r_M,
 *
 * where W(B_ref) is the conservative ideal-MHD magnetic wave operator.
 * Density and electron energy are then recovered from continuity and
 * compression, and the electric field is corrected by dM/rho_ref x B_ref.
 * When Hall physics is active with evolving ions, the acoustic fluid block is
 * followed by the momentum-to-field correction instead: the current wave
 * Schur approximates the inverse Hall field block by the identity and is not
 * robust in that regime. The acoustic pressure Schur block also remains the
 * zero-background path. All MLMG applications use zero initial guesses, fixed
 * cycle counts, and fixed bottom smoothing.
 *
 * For the conservative-form recast (implicit_mhd.fluid_flux = hlld) the state
 * carries face-staggered B^{n+theta} instead of E, and the preconditioner
 * composes three pieces, each of which engages only where it measurably
 * pays (on resolved states the residuals are smooth, and any preconditioner
 * content merely injects structure that GMRES must then resolve against the
 * Jacobian's rough grid-scale HLLD response):
 *
 * 1. Per-direction signal diffusion (OPT-IN via
 *    pc_mhd_block.signal_diffusion_scale; it degraded plain GMRES on every
 *    benchmarked regime because HLLD dissipates per characteristic channel
 *    while one scalar coefficient per block diffuses them all): every
 *    cell-centered block sees
 *
 *        [I + theta dt S_dir (h_dir/2) (-lap_dir)] dU = bU,
 *
 *    with S_dir = |u_dir|_ref + c_fast,ref, |u_dir|_ref the domain RMS of
 *    the instantaneous velocity component (an isotropic |u|_ref would add
 *    the axial flow to the radial faces' diffusion on axially-flowing
 *    states), reduced by pc_mhd_block.signal_cfl_floor resolved cells per
 *    implicit interval. In RZ the Helmholtz is assembled in cylindrical
 *    form by multiplying the equation through by r (a = r, b_r = r_face
 *    D_r, b_z = r_cell D_z).
 *
 * 2. RZ (m = 0) momentum wave Schur: the momentum block is composed with
 *    the cylindrical wave operator MHDWaveOpRZ
 *    (I - h^2 c_s^2 grad div + h^2/(mu0 rho_ref) W_B with the INPUT
 *    Reference field magnitude along zhat; the domain mean is near zero
 *    for reversed-field states, and W_B is quadratic in B so the sign
 *    structure is immaterial), applied after the scalar diffusion with a
 *    fixed-cycle MLMG and zero initial guess, engaged when the reference
 *    fast CFL reaches pc_mhd_block.wave_cfl_threshold.
 *
 * 3. Triangular ideal-induction (Faraday) corrector on the B block:
 *
 *        dB = bB + theta dt curl(dM/rho_ref x B_cc),
 *
 *    evaluated with the solver's staggered curl conventions from the
 *    cell-centered magnetic field frozen at the preconditioner update (the
 *    frozen field carries the reversed-core sign structure and any
 *    external contribution, and vanishes on unmagnetized states; the
 *    coupling is linear in B, so unlike W_B the sign matters). The sign
 *    follows from the residual convention F_B = B - b + theta dt curl E
 *    with E = -u x B, whose momentum coupling is
 *    J_BM dM = -theta dt curl(dM/rho x B). The corrector requires either
 *    an active wave Schur or a sub-unity Alfven CFL: correcting from an
 *    identity momentum block at stiff coupling amplifies the neglected
 *    J_MB J_BM product quadratically.
 *
 * 4. Resistive block: the theta_r-implicit resistive term of the B
 *    residual, dB + theta_r dt curl(eta_field/mu0 curl dB) with eta_field
 *    the density-keyed field resistivity (including the vacuum boost)
 *    frozen at the preconditioner update, is approximately inverted by a
 *    FIXED-COUNT CHEBYSHEV iteration on the operator
 *
 *        I + A_cc,    A_cc = theta_r dt curl((eta_field/mu0) curl .),
 *
 *    applied matrix-free with the residual's exact Yee staggering,
 *    eta_field evaluated AT the electric-field locations from the
 *    residual's own interpolated density (the vacuum boost is quadratic
 *    in 1/rho: averaging cell etas instead misrepresents the operator by
 *    order unity across halo gradients), and tangential E zeroed on
 *    non-periodic domain boundary planes (verified exact against the
 *    residual's one-step map at PEC walls and the axis to roundoff on the
 *    stiff z-modulated vacuum-eta deck). The eigenvalue interval is
 *    [1, 1 + 1.5 theta_r dt (eta_max/mu0) sum_d 4/dx_d^2].
 *
 *    Iterating on the EXACT block operator is the load-bearing choice.
 *    Every per-component scalar-Helmholtz approximation tried first
 *    (isotropic, transverse-masked, r-weighted, radius-floored, with and
 *    without cylindrical curvature terms, solved exactly or by fixed
 *    V-cycles, applied directly or in the projector form
 *    I - S A_cc) DEGRADED restarted GMRES on stiff RZ vacuum-eta states
 *    by 10-100x relative to no preconditioner at all, both in situ and in
 *    an offline replica with exact sparse solves: the resistive curl-curl
 *    is spectrally dominated by its curl-free kernel (identity rows) and
 *    strong cross-component coupling, and any per-component surrogate
 *    misassigns order-one fractions of the stiff spectrum, filling the
 *    preconditioned spectrum with slow clusters that restarted GMRES
 *    grinds against (1D masks this: its compressive component Bz drops
 *    out of the curl entirely). The same offline replica shows the exact
 *    coupled inverse converging in one outer iteration, and k Chebyshev
 *    applications of the exact operator converging the outer solve in
 *    ~30/16/11 iterations for k = 10/20/30 against 306 unpreconditioned
 *    on rough right-hand sides. Each Chebyshev application costs one
 *    curl-curl stencil sweep -- far cheaper than the full HLLD residual
 *    evaluation an outer Krylov iteration costs. The block engages only
 *    when the grid-scale resistive diffusion number theta_r dt
 *    max(eta_field)/mu0 / min(dx)^2 reaches
 *    pc_mhd_block.resistive_threshold: below it the unpreconditioned
 *    rows are benign and the block is the exact identity at zero cost.
 *    It is composed AFTER the Faraday corrector, i.e.
 *    dB = J_BB^{-1} (bB - J_BM dM), the consistent triangular order.
 *
 *    pc_mhd_block.resistive_solver selects the inner solver.
 *    "chebyshev" is the portable fixed-count iteration above; its
 *    automatic application count grows as ceil(2 sqrt(lambda_max))
 *    (~50-67 stencil sweeps per application at the measured stiff
 *    regimes, z up to ~1e3 with eta contrasts up to 1e4), and its
 *    polynomial interval degrades as the contrast and diffusion number
 *    grow. "direct" instead assembles the frozen I + A_cc in sparse-row
 *    form ONCE per preconditioner update -- rows emitted by the SAME
 *    stencil routine the matrix-free path applies (MHDResistiveStencil.H,
 *    validated to roundoff by pc_mhd_block.resistive_validate_assembly)
 *    -- and factorizes it exactly (NVIDIA cuDSS; CUDA builds configured
 *    with WarpX_CUDSS=ON), so every application is a single
 *    forward/backward substitution: the exact block inverse, with the
 *    factorization amortized over all applications in the update.
 *    "direct" is the default wherever it is available (cuDSS compiled
 *    and the CUDA backend active); every other build defaults to
 *    "chebyshev", and an explicit "direct" request there aborts with a
 *    descriptive message (no heavy portable dependency is worth the
 *    default path). "banded" is the Chacon block-banded LU
 *    (MHDResistiveBanded.H): the same emission scattered into a
 *    block-banded tensor along z (blocks dense over components and the
 *    radial line; periodic z eliminated in folded ring order) and
 *    factored/solved by dense kernels resident on the device -- an exact
 *    inverse up to the selectable factorization precision, with no cuDSS
 *    dependency and no rank-0 gather (single-rank only for now).
 *
 *    When the residual's Ohm law carries the hyper-resistive term, the
 *    frozen chain -theta_r dt curl(eta_H lap(curl .))/mu0 joins the block
 *    operator through the same emission (eta_H frozen at the residual's
 *    interpolated arguments), the activation gate and the Chebyshev
 *    interval gaining the corresponding kmax^4-scale bounds.
 *
 *    When the residual's Ohm law carries the Hall term
 *    (pc_mhd_block.include_hall_mhd_coupling, default on), the whistler
 *    rows + theta dt curl((curl . / mu0) x Hb) join the same emission,
 *    linearized about the frozen Hall coefficient vector Hb = B/rho_q on
 *    the electric-field staggerings (Chacon, JCP 526 (2025) 113789, in
 *    B-form) at the REACTIVE theta dt staging of the ideal EMF. The
 *    coupling is first-order and skew (whistler): its complex spectrum
 *    sits outside any real Chebyshev interval, so the Hall rows require
 *    resistive_solver = banded or direct (asserted), whose exact LU
 *    inverts the non-symmetric frozen block directly. The grid whistler
 *    number theta dt max|Hb|/mu0 / min(dx)^2 joins the activation gate.
 *
 * Boundary types the residual handles itself are mapped per component to
 * preconditioner-only linear-operator types matching the residual's fluid
 * ghosts: PEC (reflecting wall) -> Neumann for scalars and tangential
 * momentum, homogeneous Dirichlet for the normal momentum; Open (fluid
 * outflow) and None (solver-managed zero-gradient ghosts) -> Neumann.
 */
template <class T, class Ops>
class MHDBlockPC : public Preconditioner<T, Ops>
{
    public:
        using RT = typename T::value_type;
        using VectorMultiFab = amrex::Array<amrex::MultiFab, 3>;

        MHDBlockPC () = default;
        ~MHDBlockPC () override = default;

        MHDBlockPC (const MHDBlockPC&) = delete;
        MHDBlockPC& operator=(const MHDBlockPC&) = delete;
        MHDBlockPC (MHDBlockPC&&) noexcept = delete;
        MHDBlockPC& operator=(MHDBlockPC&&) noexcept = delete;

        void Define (const T& state, Ops* ops) override;
        void Update (const T& state) override;
        void Apply (T& solution, const T& rhs) override;
        void printParameters () const override;

        // These methods contain extended __device__ lambdas and must therefore
        // be public when compiled with NVCC.
        void ApplyFluidBlock (T& solution, const T& rhs);
        void ApplyFrozenIonFluidBlock (T& solution, const T& rhs);
        void BuildFieldRHS (const T& rhs, const amrex::MultiFab* momentum);
        void AddIdealLorentzForceToMomentumRHS (const T& field_predictor, const T& rhs);
        void ApplyIdealWaveFluidBlock (T& solution, const T& rhs);
        void UpdateSignalDiffusion (const T& state);
        void ApplySignalDiffusionBlocks (T& solution, const T& rhs);
        void ApplyFaradayCorrector (T& solution);
        void UpdateResistiveBlock ();
        void ApplyResistiveBlock (T& solution);
        void ApplyResistiveCurlCurl ();
        void ValidateResistiveAssembly ();

        [[nodiscard]] bool
        IsDefined () const override
        {
            return m_is_defined;
        }

    private:
        inline static constexpr char MassDensityName[] = "implicit_mhd_mass_density";
        inline static constexpr char MomentumDensityName[] = "implicit_mhd_momentum_density";
        inline static constexpr char ElectronEnergyName[] = "implicit_mhd_electron_energy";

        bool m_is_defined = false;
        bool m_hlld_state = false;
        bool m_verbose = false;
        bool m_bottom_verbose = false;
        bool m_agglomeration = true;
        bool m_consolidation = true;
        bool m_acoustic_active = false;
        bool m_include_ideal_mhd_coupling = true;
        bool m_include_hall_mhd_coupling = true;
        bool m_ideal_mhd_active = false;
        bool m_ideal_mhd_wave_active = false;
        bool m_hall_mhd_active = false;
        bool m_evolve_ion_fluid = true;

        bool m_include_wave_schur = true;
        bool m_wave_schur_active = false;
        bool m_signal_diffusion_active = false;
        bool m_faraday_active = false;
        bool m_resistive_active = false;
        // Signal diffusion is opt-in (see UpdateSignalDiffusion): on every
        // benchmarked regime of the conservative-form recast it degraded
        // plain GMRES, because HLLD dissipates per characteristic channel
        // while one scalar coefficient per block diffuses them all.
        RT m_signal_diffusion_scale = 0.0;
        RT m_signal_cfl_floor = 1.0;
        RT m_wave_cfl_threshold = 1.0;

        int m_field_iterations = 2;
        int m_fluid_iterations = 2;
        // Chebyshev applications of the exact resistive curl-curl per
        // preconditioner application; 0 (the default) selects
        // ceil(2 sqrt(lambda_max)) automatically, the count at which the
        // fixed Chebyshev contraction reaches ~e^-4 independent of the
        // stiffness (measured on the z ~ 150 z-modulated vacuum-eta RZ
        // deck: 16/32/64/96 applications give 862/55/28/20 outer
        // iterations against 1570 unpreconditioned).
        int m_resistive_iterations = 0;
        int m_resistive_iteration_count = 0;
        int m_max_coarsening_level = 30;
        int m_signal_components = 0;
        int m_momentum_component = -1;

        // Resistive-block inner solver: "chebyshev" (the portable
        // fixed-count iteration) or "direct" (exact sparse
        // factorization of the frozen operator, one factorization per
        // preconditioner update amortized over all applications; requires
        // a CUDA build configured with WarpX_CUDSS=ON). Where the exact
        // factorization is available -- the cuDSS path compiled AND the
        // CUDA backend active -- it is the default (strictly stronger
        // and cheaper per application than the Chebyshev iteration at
        // every measured stiff regime); all other builds default to the
        // portable Chebyshev iteration.
        // "banded" is the Chacon block-banded LU along z
        // (MHDResistiveBanded.H): a portable exact-inverse alternative to
        // "direct" (no cuDSS dependency, no rank-0 gather; single-rank
        // only for now). pc_mhd_block.banded_precision selects the
        // factorization storage; "double" is the default because the
        // unpivoted elimination at production stiffness (block condition
        // numbers ~1e4-1e5 over hundreds of pivots) exhausts FP32
        // headroom and was measured to break the Newton line search.
#if defined(WARPX_USE_CUDSS) && defined(AMREX_USE_CUDA)
        std::string m_resistive_solver = "direct";
#else
        std::string m_resistive_solver = "chebyshev";
#endif
        bool m_resistive_direct_enabled = false;
        bool m_resistive_banded_enabled = false;
        std::string m_banded_precision = "double";
        // Refreeze policy of the banded factorization: "step" (default)
        // factors once per time step and reuses it across the step's
        // Newton iterations (the standard frozen-coefficient PC lag;
        // the factorization is the dominant banded cost), "newton"
        // refactors at every preconditioner update.
        std::string m_banded_refreeze = "step";
        RT m_banded_frozen_time = std::numeric_limits<RT>::lowest();
        // Whether the residual's Ohm law carries the hyper-resistive term
        // (deck-static: sizes the stencil scratch and the banded
        // bandwidth at Define) and whether the frozen eta_H content is in
        // the current update's operator.
        bool m_resistive_hyper = false;
        bool m_resistive_hyper_active = false;
        // Whether the recast B block carries the Hall/whistler rows
        // (deck-static: the Ohm law's Hall term gated by
        // include_hall_mhd_coupling) and their reactive scale
        // theta dt / mu0 (the resistive rows keep theta_r dt / mu0).
        bool m_resistive_hall_active = false;
        RT m_resistive_hall_scale = 0.0;
        // Whether the recast B block carries the electron-inertia rows
        // (the Ohm law's inertia term gated by
        // include_electron_inertia_coupling) and their dt-independent
        // scale s/mu0 (s = the residual's active dJe/dt stencil factor;
        // see MHDResistiveStencil.H).
        bool m_include_electron_inertia_coupling = true;
        bool m_resistive_inertia_active = false;
        RT m_resistive_inertia_scale = 0.0;
        // Roundoff matvec check of the assembled sparse rows against the
        // matrix-free operator at every active update (any build).
        bool m_resistive_validate_assembly = false;

        // Resistive-block gate: minimum resistive diffusion number
        // theta_r dt max(eta_field)/mu0 / min(dx)^2 at which the resistive
        // Helmholtz engages (below it the block is the exact identity).
        RT m_resistive_threshold = 1.0;
        RT m_resistive_h = 0.0;
        RT m_resistive_diffusion_number = 0.0;
        // Upper spectral bound of I + A_cc for the Chebyshev interval.
        RT m_resistive_lambda_max = 1.0;

        RT m_gamma_e = RT(5.0 / 3.0);
        RT m_gamma_i = RT(5.0 / 3.0);
        RT m_reference_mass_density = 0.0;
        RT m_reference_magnetic_field_scale = 0.0;
        RT m_reference_ion_pressure = 0.0;
        RT m_reference_charge_density = 0.0;
        RT m_reference_velocity = 0.0;
        RT m_wave_relaxation = 0.5;

        RT m_h = 0.0;
        RT m_eta_reference = 0.0;
        RT m_field_diffusion_coefficient = 0.0;
        RT m_hall_diffusion_coefficient = 0.0;
        RT m_field_preconditioning_coefficient = 0.0;
        RT m_electron_enthalpy = 0.0;
        RT m_ion_sound_speed_squared = 0.0;
        RT m_sound_speed_squared = 0.0;
        RT m_fast_speed = 0.0;
        amrex::GpuArray<RT, AMREX_SPACEDIM> m_signal_speed = {};
        amrex::GpuArray<RT, AMREX_SPACEDIM> m_signal_diffusion = {};
        amrex::GpuArray<RT, 3> m_reference_magnetic_field = {0.0, 0.0, 0.0};

        Ops* m_ops = nullptr;
        amrex::Geometry m_geometry;
        amrex::BoxArray m_cell_grids;
        amrex::DistributionMapping m_distribution_map;

        std::unique_ptr<amrex::LPInfo> m_info;
        std::unique_ptr<amrex::MLCurlCurl> m_curl_curl;
        std::unique_ptr<amrex::MLMGT<VectorMultiFab>> m_field_solver;
        std::unique_ptr<amrex::MLCurlCurl> m_unit_curl_curl;
        std::unique_ptr<amrex::MLMGT<VectorMultiFab>> m_unit_curl_curl_apply;
        std::unique_ptr<amrex::MLABecLaplacian> m_acoustic_operator;
        std::unique_ptr<amrex::MLMG> m_fluid_solver;
        std::unique_ptr<MHDWaveOp> m_wave_operator;
        std::unique_ptr<amrex::MLMG> m_wave_solver;
        std::unique_ptr<amrex::MLABecLaplacian> m_signal_operator;
        std::unique_ptr<amrex::MLMG> m_signal_solver;
#if defined(WARPX_DIM_RZ)
        std::unique_ptr<MHDWaveOpRZ> m_wave_operator_rz;
        std::unique_ptr<amrex::MLMG> m_wave_solver_rz;
#endif

        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_field_rhs;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_field_operator_action;
        std::unique_ptr<amrex::MultiFab> m_momentum_rhs;
        std::unique_ptr<amrex::MultiFab> m_momentum_solution;
        std::unique_ptr<amrex::MultiFab> m_pressure_rhs;
        std::unique_ptr<amrex::MultiFab> m_pressure;
        // All cell-centered blocks stacked as components of one Helmholtz
        // solve (they share the same reference operator).
        std::unique_ptr<amrex::MultiFab> m_signal_rhs;
        std::unique_ptr<amrex::MultiFab> m_signal_solution;
        // Per-direction face coefficients; in RZ they and the cell
        // coefficient also carry the cylindrical metric factor r.
        std::unique_ptr<amrex::MultiFab> m_signal_acoef;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_signal_bcoef;
        // Scratch for the per-direction velocity RMS reductions.
        std::unique_ptr<amrex::MultiFab> m_signal_scratch;
        // Cell-centered magnetic field frozen at the preconditioner update,
        // read by the triangular Faraday corrector.
        std::unique_ptr<amrex::MultiFab> m_faraday_field;

        // Resistive block scratch (hlld state): the Chebyshev iterate,
        // residual, direction (ghosted: the curl-curl stencils read one
        // interior/periodic ghost), and operator action, one per
        // participating magnetic-field component.
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_resistive_iterate;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_resistive_residual;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_resistive_direction;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, 3> m_resistive_action;
        // eta_field (and, with the hyper-resistive term, eta_H) on the
        // electric-field staggerings, frozen at the preconditioner update
        // (solver-owned scratch).
        amrex::Array<const amrex::MultiFab*, 3> m_resistive_eta_edge = {nullptr, nullptr,
                                                                        nullptr};
        amrex::Array<const amrex::MultiFab*, 3> m_resistive_eta_h_edge = {nullptr, nullptr,
                                                                          nullptr};
        // Frozen Hall coefficient vector Hb = B/rho_q on the
        // electric-field staggerings (three components per register),
        // read by the Hall/whistler rows of the emission.
        amrex::Array<const amrex::MultiFab*, 3> m_resistive_hall_edge = {nullptr, nullptr,
                                                                         nullptr};
        // Frozen electron-inertia coefficient C_i = m_e_eff/(e rho_q) on
        // the electric-field staggerings, read by the inertia rows of the
        // emission.
        amrex::Array<const amrex::MultiFab*, 3> m_resistive_inertia_edge = {nullptr, nullptr,
                                                                            nullptr};
        // Exact sparse inverse of the frozen resistive block (direct
        // solver and assembly validation), built lazily at the first
        // active update that needs it.
        std::unique_ptr<warpx::mhd_pc::ResistiveDirectSolver> m_resistive_direct;
        // Block-banded LU of the frozen resistive block (banded solver),
        // built lazily at the first active update that needs it.
        std::unique_ptr<warpx::mhd_pc::ResistiveBandedSolver> m_resistive_banded;

        void ReadParameters ();
        void ApplyFieldBlock (T& solution, const T& rhs);
        void SolveFieldBlock (T& solution);
        void ComputeUnitCurlCurlAction (const T& field);
        void DefineSignalDiffusion (const T& state);
        void DefineResistiveBlock (const T& state);

        /**
         * \brief Whether this magnetic-field component participates in the
         * resistive block (in 1D the curl reaches only the transverse
         * components: the Bz row of the resistive curl-curl is exactly
         * zero, so the Chebyshev iteration leaves it at its identity
         * content).
         */
        [[nodiscard]] static constexpr bool
        ResistiveComponentActive (const int a_component)
        {
#if defined(WARPX_DIM_1D_Z)
            return a_component < 2;
#else
            return a_component < 3;
#endif
        }
        [[nodiscard]] amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
        GetCellCenteredLinOpBC (bool a_high_side, int a_momentum_component) const;
        [[nodiscard]] amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>>
        GetMomentumLinOpBC (bool a_high_side) const;

        /**
         * \brief Physical vector component normal to the given logical
         * direction (the same map orders the per-direction velocity RMS).
         */
        [[nodiscard]] static int
        PhysicalNormalComponent (const int a_direction)
        {
#if defined(WARPX_DIM_1D_Z)
            constexpr amrex::GpuArray<int, AMREX_SPACEDIM> normal_component = {2};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            constexpr amrex::GpuArray<int, AMREX_SPACEDIM> normal_component = {0, 2};
#else
            constexpr amrex::GpuArray<int, AMREX_SPACEDIM> normal_component = {0, 1, 2};
#endif
            return normal_component[a_direction];
        }
};

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ReadParameters ()
{
    const amrex::ParmParse pp(amrex::getEnumNameString(PreconditionerType::pc_mhd_block));
    pp.query("verbose", m_verbose);
    pp.query("bottom_verbose", m_bottom_verbose);
    pp.query("agglomeration", m_agglomeration);
    pp.query("consolidation", m_consolidation);
    pp.query("include_ideal_mhd_coupling", m_include_ideal_mhd_coupling);
    pp.query("include_hall_mhd_coupling", m_include_hall_mhd_coupling);
    pp.query("include_electron_inertia_coupling",
             m_include_electron_inertia_coupling);
    pp.query("include_wave_schur", m_include_wave_schur);
    utils::parser::queryWithParser(pp, "field_iterations", m_field_iterations);
    utils::parser::queryWithParser(pp, "fluid_iterations", m_fluid_iterations);
    utils::parser::queryWithParser(pp, "resistive_iterations", m_resistive_iterations);
    utils::parser::queryWithParser(pp, "resistive_threshold", m_resistive_threshold);
    pp.query("resistive_solver", m_resistive_solver);
    pp.query("banded_precision", m_banded_precision);
    pp.query("banded_refreeze", m_banded_refreeze);
    pp.query("resistive_validate_assembly", m_resistive_validate_assembly);
    utils::parser::queryWithParser(pp, "max_coarsening_level", m_max_coarsening_level);
    utils::parser::queryWithParser(pp, "wave_relaxation", m_wave_relaxation);
    utils::parser::queryWithParser(pp, "signal_diffusion_scale", m_signal_diffusion_scale);
    utils::parser::queryWithParser(pp, "signal_cfl_floor", m_signal_cfl_floor);
    utils::parser::queryWithParser(pp, "wave_cfl_threshold", m_wave_cfl_threshold);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_signal_diffusion_scale >= 0.0,
        "pc_mhd_block.signal_diffusion_scale cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_signal_cfl_floor >= 0.0,
                                     "pc_mhd_block.signal_cfl_floor cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_wave_cfl_threshold >= 0.0,
                                     "pc_mhd_block.wave_cfl_threshold cannot be negative");

    const amrex::ParmParse pp_mhd("implicit_mhd");
    utils::parser::getWithParser(pp_mhd, "reference_mass_density", m_reference_mass_density);
    utils::parser::getWithParser(pp_mhd, "reference_magnetic_field",
                                m_reference_magnetic_field_scale);
    utils::parser::queryWithParser(pp_mhd, "reference_ion_pressure", m_reference_ion_pressure);
    utils::parser::queryWithParser(pp_mhd, "gamma_e", m_gamma_e);
    utils::parser::queryWithParser(pp_mhd, "gamma_i", m_gamma_i);
    // Same default as the ThetaImplicitMHD constructor: the reference Alfven
    // speed when implicit_mhd.reference_velocity is not provided.
    const bool has_reference_velocity =
        utils::parser::queryWithParser(pp_mhd, "reference_velocity", m_reference_velocity);
    if (!has_reference_velocity && m_reference_mass_density > 0.0)
    {
        m_reference_velocity = m_reference_magnetic_field_scale /
                               std::sqrt(PhysConst::mu0 * m_reference_mass_density);
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_field_iterations > 0,
                                     "pc_mhd_block.field_iterations must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_fluid_iterations > 0,
                                     "pc_mhd_block.fluid_iterations must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resistive_iterations >= 0,
        "pc_mhd_block.resistive_iterations cannot be negative (0 = automatic)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_resistive_threshold > 0.0,
                                     "pc_mhd_block.resistive_threshold must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_resistive_solver == "chebyshev" || m_resistive_solver == "direct" ||
            m_resistive_solver == "banded",
        "pc_mhd_block.resistive_solver must be chebyshev, direct, or banded");
    m_resistive_direct_enabled = m_resistive_solver == "direct";
    m_resistive_banded_enabled = m_resistive_solver == "banded";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_resistive_direct_enabled ||
            warpx::mhd_pc::ResistiveDirectSolver::HaveFactorizationBackend(),
        "pc_mhd_block.resistive_solver = direct requires a CUDA build configured "
        "with WarpX_CUDSS=ON (NVIDIA cuDSS); chebyshev is the portable default");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_resistive_banded_enabled ||
            warpx::mhd_pc::ResistiveBandedSolver::HaveBackend(),
        "pc_mhd_block.resistive_solver = banded is not implemented for SYCL builds");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_banded_precision == "single" || m_banded_precision == "double",
        "pc_mhd_block.banded_precision must be single or double");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_banded_refreeze == "step" || m_banded_refreeze == "newton",
        "pc_mhd_block.banded_refreeze must be step or newton");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_max_coarsening_level >= 0,
                                     "pc_mhd_block.max_coarsening_level cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_mass_density > 0.0,
                                     "implicit_mhd.reference_mass_density must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_magnetic_field_scale > 0.0,
                                     "implicit_mhd.reference_magnetic_field must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_ion_pressure >= 0.0,
                                     "implicit_mhd.reference_ion_pressure cannot be negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_gamma_e > 1.0,
                                     "implicit_mhd.gamma_e must be greater than one");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_gamma_i > 0.0, "implicit_mhd.gamma_i must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_reference_velocity > 0.0,
                                     "implicit_mhd.reference_velocity must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_wave_relaxation > 0.0 && m_wave_relaxation <= 0.5,
                                     "pc_mhd_block.wave_relaxation must be in (0, 0.5]");
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::Define (const T& state, Ops* const ops)
{
    BL_PROFILE("MHDBlockPC::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!IsDefined(),
                                     "MHDBlockPC::Define() called on a defined object");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ops != nullptr,
                                     "MHDBlockPC::Define() received a null operator");
    const warpx::fields::FieldType array_type = state.getArrayVecType();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        array_type == warpx::fields::FieldType::Efield_fp ||
            array_type == warpx::fields::FieldType::Bfield_fp,
        "pc_mhd_block requires Efield_fp or Bfield_fp as its vector block");
    m_hlld_state = (array_type == warpx::fields::FieldType::Bfield_fp);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(state.numAMRLevels() == 1 && ops->numAMRLevels() == 1,
                                     "pc_mhd_block currently supports one AMR level");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        state.hasMultiFabBlock(MassDensityName) && state.hasMultiFabBlock(MomentumDensityName) &&
            state.hasMultiFabBlock(ElectronEnergyName),
        "pc_mhd_block requires the theta-implicit MHD composite state");

    m_ops = ops;
    ReadParameters();

    auto const& state_fields = state.getArrayVec();
    m_geometry = m_ops->GetGeometry(0);
    m_distribution_map = state_fields[0][0]->DistributionMap();
    m_cell_grids = state_fields[0][0]->boxArray();
    m_cell_grids.enclosedCells();

    m_info = std::make_unique<amrex::LPInfo>();
    m_info->setAgglomeration(m_agglomeration);
    m_info->setConsolidation(m_consolidation);
    m_info->setMaxCoarseningLevel(m_max_coarsening_level);

    if (m_hlld_state)
    {
        // Conservative-form recast: the Faraday block is the identity (up
        // to the triangular Faraday corrector and the gated resistive
        // Helmholtz) and the cell-centered blocks share one
        // signal-diffusion Helmholtz, so none of the E-based operators
        // below are constructed (their domain boundary conversion also
        // rejects Open/None boundaries, which the hlld residual supports).
        DefineSignalDiffusion(state);
        DefineResistiveBlock(state);
        m_is_defined = true;
        return;
    }

    m_curl_curl = std::make_unique<amrex::MLCurlCurl>(
        amrex::Vector<amrex::Geometry>{m_geometry}, amrex::Vector<amrex::BoxArray>{m_cell_grids},
        amrex::Vector<amrex::DistributionMapping>{m_distribution_map}, *m_info);
    m_curl_curl->setDomainBC(m_ops->GetLinOpBCLo(), m_ops->GetLinOpBCHi());
    m_curl_curl->setScalars(1.0, 1.0);

    m_unit_curl_curl = std::make_unique<amrex::MLCurlCurl>(
        amrex::Vector<amrex::Geometry>{m_geometry}, amrex::Vector<amrex::BoxArray>{m_cell_grids},
        amrex::Vector<amrex::DistributionMapping>{m_distribution_map}, *m_info);
    m_unit_curl_curl->setDomainBC(m_ops->GetLinOpBCLo(), m_ops->GetLinOpBCHi());
    m_unit_curl_curl->setScalars(1.0, 1.0);

    m_field_solver = std::make_unique<amrex::MLMGT<VectorMultiFab>>(*m_curl_curl);
    m_field_solver->setMaxIter(m_field_iterations);
    m_field_solver->setFixedIter(m_field_iterations);
    m_field_solver->setMaxFmgIter(0);
    m_field_solver->setBottomSolver(amrex::BottomSolver::smoother);
    m_field_solver->setVerbose(static_cast<int>(m_verbose));
    m_field_solver->setBottomVerbose(static_cast<int>(m_bottom_verbose));

    m_unit_curl_curl_apply =
        std::make_unique<amrex::MLMGT<VectorMultiFab>>(*m_unit_curl_curl);

    for (int component = 0; component < 3; ++component)
    {
        m_field_rhs[component] =
            std::make_unique<amrex::MultiFab>(state_fields[0][component]->boxArray(),
                                              state_fields[0][component]->DistributionMap(), 1, 0);
        m_field_operator_action[component] =
            std::make_unique<amrex::MultiFab>(state_fields[0][component]->boxArray(),
                                              state_fields[0][component]->DistributionMap(), 1, 0);
    }

    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);
    m_momentum_rhs =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 3, 1);
    m_momentum_solution =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 3, 1);
    m_pressure_rhs =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 1, 1);
    m_pressure =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 1, 1);

    m_acoustic_operator = std::make_unique<amrex::MLABecLaplacian>(
        amrex::Vector<amrex::Geometry>{m_geometry},
        amrex::Vector<amrex::BoxArray>{density.boxArray()},
        amrex::Vector<amrex::DistributionMapping>{density.DistributionMap()}, *m_info);
    m_acoustic_operator->setDomainBC(m_ops->GetLinOpBCLo(), m_ops->GetLinOpBCHi());
    m_acoustic_operator->setLevelBC(0, m_pressure.get());
    m_acoustic_operator->setScalars(1.0, 1.0);
    m_acoustic_operator->setACoeffs(0, 1.0);
    m_acoustic_operator->setBCoeffs(0, 1.0);

    m_fluid_solver = std::make_unique<amrex::MLMG>(*m_acoustic_operator);
    m_fluid_solver->setMaxIter(m_fluid_iterations);
    m_fluid_solver->setFixedIter(m_fluid_iterations);
    m_fluid_solver->setMaxFmgIter(0);
    m_fluid_solver->setBottomSolver(amrex::BottomSolver::smoother);
    m_fluid_solver->setVerbose(static_cast<int>(m_verbose));
    m_fluid_solver->setBottomVerbose(static_cast<int>(m_bottom_verbose));

    m_wave_operator = std::make_unique<MHDWaveOp>(
        amrex::Vector<amrex::Geometry>{m_geometry},
        amrex::Vector<amrex::BoxArray>{density.boxArray()},
        amrex::Vector<amrex::DistributionMapping>{density.DistributionMap()}, *m_info);
    m_wave_operator->setDomainBC(m_ops->GetLinOpBCLo(), m_ops->GetLinOpBCHi());
    m_wave_operator->setLevelBC(0, m_momentum_solution.get());
    m_wave_operator->setRelaxation(m_wave_relaxation);

    m_wave_solver = std::make_unique<amrex::MLMG>(*m_wave_operator);
    m_wave_solver->setMaxIter(m_fluid_iterations);
    m_wave_solver->setFixedIter(m_fluid_iterations);
    m_wave_solver->setMaxFmgIter(0);
    m_wave_solver->setBottomSolver(amrex::BottomSolver::smoother);
    m_wave_solver->setVerbose(static_cast<int>(m_verbose));
    m_wave_solver->setBottomVerbose(static_cast<int>(m_bottom_verbose));

    m_is_defined = true;
}

template <class T, class Ops>
amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
MHDBlockPC<T, Ops>::GetCellCenteredLinOpBC (const bool a_high_side,
                                            const int a_momentum_component) const
{
    // Match the residual's fluid domain ghosts (ApplyFluidDomainBoundaries):
    // a PEC wall is a reflecting no-normal-flow wall (even scalars and
    // tangential momentum -> Neumann; odd normal momentum -> homogeneous
    // Dirichlet), and a PMC face -- admitted by theta_implicit_mhd only as
    // the z_lo mirror-symmetry plane, paired with
    // implicit_mhd.z_lo_boundary_fluid = symmetry -- carries the identical
    // fluid parity contract, so it maps like PEC. Open is zero-gradient
    // fluid outflow (-> Neumann; the Green's-function coupling lives on
    // the face-staggered field block, not here), and None marks
    // solver-managed zero-gradient ghosts (the RZ axis and outflow z ends
    // -> Neumann; the axis parity of the odd components is inert because
    // the r-weighted radial fluxes vanish on the r = 0 face).
    // a_momentum_component is the physical component a momentum row
    // carries, or -1 for scalar blocks.
    const amrex::Array<FieldBoundaryType, AMREX_SPACEDIM>& field_bc =
        a_high_side ? m_ops->GetFieldBoundaryHi() : m_ops->GetFieldBoundaryLo();
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> linop_bc;
    for (auto& bc : linop_bc)
    {
        bc = amrex::LinOpBCType::bogus;
    }
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        if (field_bc[direction] == FieldBoundaryType::Periodic)
        {
            linop_bc[direction] = amrex::LinOpBCType::Periodic;
        }
        else if (field_bc[direction] == FieldBoundaryType::PEC ||
                 field_bc[direction] == FieldBoundaryType::PMC)
        {
            // FieldBoundaryType::Neumann aliases PMC and lands here too;
            // theta_implicit_mhd rejects it everywhere except the z_lo
            // mirror plane, where the PEC-like parity map is exact.
            linop_bc[direction] =
                a_momentum_component == PhysicalNormalComponent(direction)
                    ? amrex::LinOpBCType::Dirichlet
                    : amrex::LinOpBCType::Neumann;
        }
        else if (field_bc[direction] == FieldBoundaryType::Open ||
                 field_bc[direction] == FieldBoundaryType::None)
        {
            linop_bc[direction] = amrex::LinOpBCType::Neumann;
        }
        else
        {
            WARPX_ABORT_WITH_MESSAGE(
                "pc_mhd_block has no cell-centered linear-operator boundary "
                "for this field boundary type");
        }
    }
    return linop_bc;
}

template <class T, class Ops>
amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>>
MHDBlockPC<T, Ops>::GetMomentumLinOpBC (const bool a_high_side) const
{
    amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>> linop_bc;
    linop_bc.reserve(3);
    for (int component = 0; component < 3; ++component)
    {
        linop_bc.push_back(GetCellCenteredLinOpBC(a_high_side, component));
    }
    return linop_bc;
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::DefineSignalDiffusion (const T& state)
{
    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);

    m_signal_components = 0;
    m_momentum_component = -1;
    for (const auto& spec : state.getMultiFabBlockSpecs())
    {
        if (spec.name == MomentumDensityName)
        {
            m_momentum_component = m_signal_components;
        }
        m_signal_components += state.getMultiFabBlock(spec.name, 0).nComp();
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_signal_components > 0 && m_momentum_component >= 0,
                                     "pc_mhd_block found no cell-centered state blocks");

    m_signal_rhs = std::make_unique<amrex::MultiFab>(density.boxArray(),
                                                     density.DistributionMap(),
                                                     m_signal_components, 0);
    m_signal_solution = std::make_unique<amrex::MultiFab>(density.boxArray(),
                                                          density.DistributionMap(),
                                                          m_signal_components, 1);

    m_signal_acoef =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 1, 0);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_signal_bcoef[direction] = std::make_unique<amrex::MultiFab>(
            amrex::convert(density.boxArray(), amrex::IntVect::TheDimensionVector(direction)),
            density.DistributionMap(), 1, 0);
    }
    m_signal_scratch =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 1, 0);
    m_momentum_solution =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 3, 1);
    m_faraday_field =
        std::make_unique<amrex::MultiFab>(density.boxArray(), density.DistributionMap(), 3, 1);

    m_signal_operator = std::make_unique<amrex::MLABecLaplacian>(
        amrex::Vector<amrex::Geometry>{m_geometry},
        amrex::Vector<amrex::BoxArray>{density.boxArray()},
        amrex::Vector<amrex::DistributionMapping>{density.DistributionMap()}, *m_info,
        amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, m_signal_components);
    // Per-component boundaries: the momentum components carry the residual's
    // reflecting-wall parity; every other stacked component is a scalar.
    {
        amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>> low_bc;
        amrex::Vector<amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>> high_bc;
        low_bc.reserve(m_signal_components);
        high_bc.reserve(m_signal_components);
        for (const auto& spec : state.getMultiFabBlockSpecs())
        {
            const bool is_momentum = spec.name == MomentumDensityName;
            const int block_components = state.getMultiFabBlock(spec.name, 0).nComp();
            for (int component = 0; component < block_components; ++component)
            {
                const int momentum_component = is_momentum ? component : -1;
                low_bc.push_back(GetCellCenteredLinOpBC(false, momentum_component));
                high_bc.push_back(GetCellCenteredLinOpBC(true, momentum_component));
            }
        }
        m_signal_operator->setDomainBC(low_bc, high_bc);
    }
    // Homogeneous boundary values on every side the map produces.
    m_signal_operator->setLevelBC(0, nullptr);
    m_signal_operator->setScalars(1.0, 1.0);
    m_signal_operator->setACoeffs(0, 1.0);
    m_signal_operator->setBCoeffs(0, 0.0);

    m_signal_solver = std::make_unique<amrex::MLMG>(*m_signal_operator);
    m_signal_solver->setMaxIter(m_fluid_iterations);
    m_signal_solver->setFixedIter(m_fluid_iterations);
    m_signal_solver->setMaxFmgIter(0);
    m_signal_solver->setBottomSolver(amrex::BottomSolver::smoother);
    m_signal_solver->setVerbose(static_cast<int>(m_verbose));
    m_signal_solver->setBottomVerbose(static_cast<int>(m_bottom_verbose));

#if defined(WARPX_DIM_RZ)
    // Cylindrical m = 0 momentum wave Schur, composed after the scalar
    // signal diffusion (see the class documentation).
    m_wave_operator_rz = std::make_unique<MHDWaveOpRZ>(
        amrex::Vector<amrex::Geometry>{m_geometry},
        amrex::Vector<amrex::BoxArray>{density.boxArray()},
        amrex::Vector<amrex::DistributionMapping>{density.DistributionMap()}, *m_info);
    m_wave_operator_rz->setDomainBC(GetMomentumLinOpBC(false), GetMomentumLinOpBC(true));
    m_wave_operator_rz->setLevelBC(0, nullptr);
    m_wave_operator_rz->setRelaxation(m_wave_relaxation);

    m_wave_solver_rz = std::make_unique<amrex::MLMG>(*m_wave_operator_rz);
    m_wave_solver_rz->setMaxIter(m_fluid_iterations);
    m_wave_solver_rz->setFixedIter(m_fluid_iterations);
    m_wave_solver_rz->setMaxFmgIter(0);
    m_wave_solver_rz->setBottomSolver(amrex::BottomSolver::smoother);
    m_wave_solver_rz->setVerbose(static_cast<int>(m_verbose));
    m_wave_solver_rz->setBottomVerbose(static_cast<int>(m_bottom_verbose));
#endif
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::DefineResistiveBlock (const T& state)
{
    // Resistive-block scratch (see the class documentation, point 4): the
    // Chebyshev iteration is matrix-free, so only per-component face
    // vectors are needed. Two ghosts unconditionally: the hyper-resistive
    // chain reaches two cells (curl of lap of curl), and whether the Ohm
    // law carries it is only knowable after the hybrid model's InitData
    // (the preconditioner is defined earlier).
    constexpr int stencil_ghosts = 2;
    auto const& state_fields = state.getArrayVec();
    for (int component = 0; component < 3; ++component)
    {
        if (!ResistiveComponentActive(component))
        {
            continue;
        }
        const amrex::BoxArray& grids = state_fields[0][component]->boxArray();
        const amrex::DistributionMapping& dmap =
            state_fields[0][component]->DistributionMap();
        m_resistive_iterate[component] =
            std::make_unique<amrex::MultiFab>(grids, dmap, 1, 0);
        m_resistive_residual[component] =
            std::make_unique<amrex::MultiFab>(grids, dmap, 1, 0);
        m_resistive_direction[component] =
            std::make_unique<amrex::MultiFab>(grids, dmap, 1, stencil_ghosts);
        m_resistive_direction[component]->setVal(0.0);
        m_resistive_action[component] =
            std::make_unique<amrex::MultiFab>(grids, dmap, 1, 0);
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::UpdateResistiveBlock ()
{
    m_resistive_active = false;
    const RT evaluation_time = this->m_time + m_h;
    const amrex::MultiFab* const eta_field =
        m_ops->GetMHDFieldResistivityCCForPC(evaluation_time);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        eta_field != nullptr,
        "pc_mhd_block requires the solver's cell-centered field resistivity "
        "for the resistive block");
    const RT resistive_theta = m_ops->GetMHDResistiveThetaForPC();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        resistive_theta > 0.0,
        "pc_mhd_block requires a positive resistive theta");
    m_resistive_h = resistive_theta * this->m_dt;

    // The frozen eta_H staggerings (null when the Ohm law runs without
    // the hyper-resistive term); their maximum feeds the gate and the
    // Chebyshev interval below. The flag is queried per update: it is
    // only valid after the hybrid model's InitData, which runs after the
    // preconditioner is defined.
    m_resistive_hyper = m_ops->GetMHDIncludeHyperResistivityForPC();
    m_resistive_hyper_active = false;
    RT eta_h_maximum = 0.0;
    if (m_resistive_hyper)
    {
        m_resistive_eta_h_edge = m_ops->GetMHDHyperResistivityEdgeForPC(evaluation_time);
        for (int component = 0; component < 3; ++component)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !ResistiveComponentActive(component) ||
                    m_resistive_eta_h_edge[component] != nullptr,
                "pc_mhd_block requires the solver's edge hyper-resistivity "
                "for the resistive block");
            if (m_resistive_eta_h_edge[component] != nullptr)
            {
                eta_h_maximum =
                    amrex::max(eta_h_maximum, m_resistive_eta_h_edge[component]->max(0));
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(eta_h_maximum) && eta_h_maximum >= 0.0,
            "pc_mhd_block requires a finite non-negative hyper-resistivity");
        m_resistive_hyper_active = true;
    }

    // Hall/whistler rows of the recast B block (Chacon, JCP 526 (2025)
    // 113789, in B-form -- see MHDResistiveStencil.H): the whistler
    // rotation linearized about the frozen Hb = B/rho_q joins the same
    // emission as the resistive term, at the REACTIVE theta dt / mu0
    // scale of the ideal EMF. Requested per update: the Hall flag is
    // only valid after the hybrid model's InitData.
    m_resistive_hall_active = false;
    m_resistive_hall_scale = 0.0;
    RT hall_maximum = 0.0;
    if (m_include_hall_mhd_coupling && m_ops->GetMHDIncludeHallTermForPC())
    {
        // The whistler block is first-order and SKEW: its spectrum is
        // the complex conjugate pairs 1 +/- i h D_H k^2, outside the
        // real interval the fixed-count Chebyshev iteration assumes
        // (it would amplify, not contract). Only the exact inverses may
        // consume the Hall rows.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_resistive_direct_enabled || m_resistive_banded_enabled,
            "pc_mhd_block.include_hall_mhd_coupling with the "
            "conservative-form recast requires pc_mhd_block.resistive_solver "
            "= banded or direct: the Hall/whistler B block is first-order "
            "and skew (complex eigenvalue pairs), outside the real Chebyshev "
            "interval. Set pc_mhd_block.include_hall_mhd_coupling = false to "
            "run the B block without the whistler rows.");
        m_resistive_hall_edge = m_ops->GetMHDHallCoefficientEdgeForPC();
        for (int component = 0; component < 3; ++component)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !ResistiveComponentActive(component) ||
                    m_resistive_hall_edge[component] != nullptr,
                "pc_mhd_block requires the solver's edge Hall coefficients "
                "for the recast Hall coupling");
            if (m_resistive_hall_edge[component] != nullptr)
            {
                for (int direction = 0; direction < 3; ++direction)
                {
                    hall_maximum = amrex::max(
                        hall_maximum,
                        m_resistive_hall_edge[component]->norm0(direction));
                }
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(hall_maximum) && hall_maximum >= 0.0,
            "pc_mhd_block requires finite Hall coefficients");
        m_resistive_hall_active = true;
        m_resistive_hall_scale = m_h / RT(PhysConst::mu0);
    }

    // Electron-inertia rows of the recast B block (see
    // MHDResistiveStencil.H): the frozen theta-stage dJe/dt response
    // folded into the resistive emission at the dt-INDEPENDENT scale
    // s/mu0 (s from the residual's active stencil), with the frozen
    // coefficient C_i = m_e_eff/(e rho_q) on the electric-field
    // staggerings. The block is SPD and BOUNDS the skew whistler part
    // (capped branch 1 + s k^2 d_e^2 +/- i h D_H k^2), but it joins the
    // Hall rows' consumers: the residual's inertia term requires the
    // Hall term, whose rows already require the exact inverses.
    m_resistive_inertia_active = false;
    m_resistive_inertia_scale = 0.0;
    RT inertia_maximum = 0.0;
    if (m_include_electron_inertia_coupling &&
        m_ops->GetMHDIncludeElectronInertiaForPC())
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_resistive_direct_enabled || m_resistive_banded_enabled,
            "pc_mhd_block.include_electron_inertia_coupling with the "
            "conservative-form recast requires pc_mhd_block.resistive_solver "
            "= banded or direct (the inertia rows join the Hall/whistler "
            "block, which the fixed-count Chebyshev iteration cannot "
            "invert). Set pc_mhd_block.include_electron_inertia_coupling = "
            "false to run the B block without them.");
        m_resistive_inertia_edge =
            m_ops->GetMHDElectronInertiaCoefficientEdgeForPC();
        for (int component = 0; component < 3; ++component)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !ResistiveComponentActive(component) ||
                    m_resistive_inertia_edge[component] != nullptr,
                "pc_mhd_block requires the solver's edge electron-inertia "
                "coefficients for the recast inertia coupling");
            if (m_resistive_inertia_edge[component] != nullptr)
            {
                inertia_maximum = amrex::max(
                    inertia_maximum,
                    m_resistive_inertia_edge[component]->max(0));
            }
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(inertia_maximum) && inertia_maximum >= 0.0,
            "pc_mhd_block requires finite electron-inertia coefficients");
        const RT stencil_scale = m_ops->GetMHDElectronInertiaStencilScaleForPC();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            stencil_scale > 0.0,
            "pc_mhd_block requires a positive electron-inertia stencil "
            "scale");
        m_resistive_inertia_active = true;
        m_resistive_inertia_scale = stencil_scale / RT(PhysConst::mu0);
    }

    // Gate on the grid-scale resistive diffusion number of the stiffest
    // cell: below the threshold the unpreconditioned rows are benign and
    // the block stands down to the exact identity at zero cost. The
    // cell-centered maximum also bounds the edge values (eta_field is
    // monotone decreasing in the interpolated density).
    const RT eta_maximum = eta_field->max(0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(eta_maximum) && eta_maximum >= 0.0,
        "pc_mhd_block requires a finite non-negative field resistivity");
    const auto inverse_cell_size = m_geometry.InvCellSizeArray();
    m_resistive_diffusion_number = 0.0;
    RT laplacian_bound = 0.0;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_resistive_diffusion_number = amrex::max(
            m_resistive_diffusion_number,
            m_resistive_h * eta_maximum / RT(PhysConst::mu0) *
                inverse_cell_size[direction] * inverse_cell_size[direction]);
        // Grid whistler number theta dt D_H / dx^2 with D_H = |Hb|/mu0:
        // the same first-order-in-k^2 stiffness scale as the resistive
        // number, at the reactive theta staging of the Hall rows.
        m_resistive_diffusion_number = amrex::max(
            m_resistive_diffusion_number,
            m_h * hall_maximum / RT(PhysConst::mu0) *
                inverse_cell_size[direction] * inverse_cell_size[direction]);
        // Grid electron-inertia number s C_i/mu0 / dx^2 = s (d_e/dx)^2:
        // dt-independent, O(1) exactly where the inertial cap matters.
        m_resistive_diffusion_number = amrex::max(
            m_resistive_diffusion_number,
            m_resistive_inertia_scale * inertia_maximum *
                inverse_cell_size[direction] * inverse_cell_size[direction]);
        laplacian_bound += RT(4.0) * inverse_cell_size[direction] *
                           inverse_cell_size[direction];
    }
    // The hyper-resistive grid number theta_r dt (eta_H/mu0) kmax^4-like
    // scale joins the gate (kmax^4 ~ laplacian_bound^2 / 16 per
    // direction-summed bound).
    m_resistive_diffusion_number = amrex::max(
        m_resistive_diffusion_number, m_resistive_h * eta_h_maximum /
                                          RT(PhysConst::mu0) * laplacian_bound *
                                          laplacian_bound / RT(16.0));
    m_resistive_active = m_resistive_diffusion_number >= m_resistive_threshold;
    if (m_verbose)
    {
        amrex::Print() << "Updating pc_mhd_block (recast B block): "
                       << "theta_r dt = " << m_resistive_h
                       << ", grid number = " << m_resistive_diffusion_number
                       << " (threshold " << m_resistive_threshold << "), block = "
                       << (m_resistive_active ? m_resistive_solver : std::string("identity"))
                       << ", Hall rows = "
                       << (m_resistive_hall_active ? "active" : "inactive")
                       << ", inertia rows = "
                       << (m_resistive_inertia_active ? "active" : "inactive")
                       << ", hyper-resistive rows = "
                       << (m_resistive_hyper_active ? "active" : "inactive") << "\n";
    }
    if (!m_resistive_active)
    {
        return;
    }

    // Chebyshev interval: [1, 1 + safety * theta_r dt (eta_max/mu0)
    // sum_d 4/dx_d^2]. The 1.5 safety absorbs the cylindrical metric
    // ratios; overestimating the top of the spectrum only slows the
    // fixed-count contraction, never destabilizes it. The eta_H chain
    // adds its curl lap curl bound, laplacian_bound^2.
    m_resistive_lambda_max = RT(1.0) + RT(1.5) * m_resistive_h * eta_maximum /
                                           RT(PhysConst::mu0) * laplacian_bound +
                             RT(1.5) * m_resistive_h * eta_h_maximum /
                                 RT(PhysConst::mu0) * laplacian_bound *
                                 laplacian_bound;
    // Automatic application count: ceil(2 sqrt(lambda_max)) contracts the
    // Chebyshev residual to ~e^-4 independent of the stiffness (bounded
    // for pathological states; each application costs one stencil sweep,
    // far below the full residual evaluation an outer Krylov iteration
    // costs).
    m_resistive_iteration_count =
        m_resistive_iterations > 0
            ? m_resistive_iterations
            : amrex::min(512, static_cast<int>(std::ceil(
                                  RT(2.0) * std::sqrt(m_resistive_lambda_max))));

    m_resistive_eta_edge = m_ops->GetMHDFieldResistivityEdgeForPC(evaluation_time);
    for (int component = 0; component < 3; ++component)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !ResistiveComponentActive(component) ||
                m_resistive_eta_edge[component] != nullptr,
            "pc_mhd_block requires the solver's edge field resistivity "
            "for the resistive block");
    }

    warpx::mhd_pc::ResistiveStencilConstants stencil_constants =
        warpx::mhd_pc::MakeResistiveStencilConstants(m_geometry, m_resistive_h);
    stencil_constants.include_hyper = m_resistive_hyper_active;
    stencil_constants.include_hall = m_resistive_hall_active;
    stencil_constants.hall_scale = m_resistive_hall_scale;
    stencil_constants.include_inertia = m_resistive_inertia_active;
    stencil_constants.inertia_scale = m_resistive_inertia_scale;
#if defined(WARPX_DIM_RZ)
    // Stair-step PEC wall mask (implicit_mhd.wall_model = pec): the
    // assemblies drop the same masked tangential-E rows the residual
    // projects (inactive default leaves the emission unchanged).
    stencil_constants.wall = m_ops->GetMHDWallMaskForPC();
    // Shaped-wall field freeze (implicit_mhd.wall_field_freeze): frozen
    // magnetic-field faces assemble as pure identity rows, matching the
    // residual's F = B - B^n rows exactly (their columns stay in the
    // live rows: the exact block-triangular Jacobian structure).
    stencil_constants.field_freeze = m_ops->GetMHDWallFieldFreezeForPC();
    // z_lo mirror plane (pmc + fluid symmetry): the end rows fold their
    // ghost columns with the mirror parities instead of keeping the
    // tangential-E-zero Neumann-end convention.
    stencil_constants.axial_lo_mirror = m_ops->GetMHDZLoMirrorForPC();
#endif
    amrex::Array<const amrex::MultiFab*, 3> face_layout = {nullptr, nullptr, nullptr};
    for (int component = 0; component < 3; ++component)
    {
        if (ResistiveComponentActive(component))
        {
            face_layout[component] = m_resistive_iterate[component].get();
        }
    }

    if (m_resistive_direct_enabled || m_resistive_validate_assembly)
    {
        // The frozen operator's sparse rows come from the SAME stencil
        // emission the matrix-free path applies; the pattern is geometric
        // and built once, the values are refrozen at every update.
        if (m_resistive_direct == nullptr)
        {
            m_resistive_direct =
                std::make_unique<warpx::mhd_pc::ResistiveDirectSolver>();
        }
        if (!m_resistive_direct->IsDefined())
        {
            m_resistive_direct->Define(m_geometry, face_layout, m_resistive_eta_edge,
                                       m_resistive_eta_h_edge, m_resistive_hall_edge,
                                       m_resistive_inertia_edge);
        }
        m_resistive_direct->AssembleValues(stencil_constants, m_resistive_eta_edge,
                                           m_resistive_eta_h_edge, m_resistive_hall_edge,
                                           m_resistive_inertia_edge);
        if (m_resistive_validate_assembly)
        {
            ValidateResistiveAssembly();
        }
        if (m_resistive_direct_enabled)
        {
            m_resistive_direct->Factorize();
        }
    }

    if (m_resistive_banded_enabled)
    {
        // The same emission scattered into the block-banded tensor and
        // factored on the device (see MHDResistiveBanded.H). Under the
        // "step" refreeze policy the factorization is reused across the
        // step's Newton iterations (frozen-coefficient PC lag).
        if (m_resistive_banded == nullptr)
        {
            m_resistive_banded =
                std::make_unique<warpx::mhd_pc::ResistiveBandedSolver>();
        }
        const bool first_factorization = !m_resistive_banded->IsDefined();
        if (first_factorization)
        {
            m_resistive_banded->Define(m_geometry, face_layout,
                                       m_resistive_hyper ? 2 : 1,
                                       m_banded_precision == "single");
        }
        if (first_factorization || m_banded_refreeze == "newton" ||
            this->m_time != m_banded_frozen_time)
        {
            m_resistive_banded->AssembleValues(stencil_constants,
                                               m_resistive_eta_edge,
                                               m_resistive_eta_h_edge,
                                               m_resistive_hall_edge,
                                               m_resistive_inertia_edge);
            m_resistive_banded->Factorize();
            m_banded_frozen_time = this->m_time;
        }
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ValidateResistiveAssembly ()
{
    // Roundoff matvec check of the assembled sparse rows against the
    // matrix-free operator on a deterministic pseudo-random vector: load
    // the vector through the canonical scatter (duplicated stored faces
    // consistent by construction), apply the matrix-free curl-curl, and
    // compare x + A x against the assembled-matrix product.
    amrex::Array<amrex::MultiFab*, 3> test_vector = {nullptr, nullptr, nullptr};
    amrex::Array<const amrex::MultiFab*, 3> action = {nullptr, nullptr, nullptr};
    for (int component = 0; component < 3; ++component)
    {
        if (!ResistiveComponentActive(component))
        {
            continue;
        }
        test_vector[component] = m_resistive_iterate[component].get();
        action[component] = m_resistive_action[component].get();
    }
    m_resistive_direct->FillValidationVector(test_vector);
    for (int component = 0; component < 3; ++component)
    {
        if (!ResistiveComponentActive(component))
        {
            continue;
        }
        amrex::MultiFab::Copy(*m_resistive_direction[component],
                              *m_resistive_iterate[component], 0, 0, 1, 0);
        m_resistive_direction[component]->FillBoundary(m_geometry.periodicity());
    }
    ApplyResistiveCurlCurl();
    const auto [max_difference, reference_scale] =
        m_resistive_direct->CompareMatrixFreeAction(action);
    const RT tolerance = RT(1.0e-12) * amrex::max(RT(1.0), reference_scale);
    amrex::Print() << "pc_mhd_block resistive assembly check: max |assembled - "
                      "matrix-free| = "
                   << max_difference << " (scale " << reference_scale
                   << ", tolerance " << tolerance << ")\n";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        max_difference <= tolerance,
        "pc_mhd_block resistive assembly does not match the matrix-free operator "
        "to roundoff");
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::UpdateSignalDiffusion (const T& state)
{
    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& momentum = state.getMultiFabBlock(MomentumDensityName, 0);
    const amrex::MultiFab& electron_energy = state.getMultiFabBlock(ElectronEnergyName, 0);
    const RT density_sum = density.sum(0, false);
    const RT energy_sum = electron_energy.sum(0, false);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(density_sum > 0.0,
                                     "pc_mhd_block encountered non-positive total mass density");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(energy_sum >= 0.0,
                                     "pc_mhd_block encountered negative total electron energy");

    const RT chi = m_gamma_e - 1.0;
    m_electron_enthalpy = m_gamma_e * energy_sum / density_sum;
    m_ion_sound_speed_squared = m_gamma_i * m_reference_ion_pressure / m_reference_mass_density;
    m_sound_speed_squared = m_ion_sound_speed_squared + chi * m_electron_enthalpy;

    // Domain-reference fast speed from the INPUT reference field scale: the
    // domain-mean B is near zero for reversed-field states, which would
    // deactivate the Alfven contribution exactly where it is stiff.
    const RT alfven_speed_squared = m_reference_magnetic_field_scale *
                                    m_reference_magnetic_field_scale /
                                    (PhysConst::mu0 * m_reference_mass_density);
    m_fast_speed = std::sqrt(m_sound_speed_squared + alfven_speed_squared);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m_fast_speed) && m_fast_speed > 0.0,
        "pc_mhd_block requires a positive finite reference fast speed");

    // Per-direction reference signal speeds S_dir = |u_dir|_ref + c_fast,ref
    // with |u_dir|_ref the domain RMS of the instantaneous velocity
    // component: an isotropic |u|_ref would add the axial flow to the radial
    // faces' diffusion, over-preconditioning r on axially-flowing states and
    // distorting the Krylov space.
    // Divide-by-zero guard only; admissible states stay far above this.
    const RT density_guard = RT(1.0e-12) * m_reference_mass_density;
    const RT inverse_number_of_cells =
        RT(1.0) / static_cast<RT>(density.boxArray().numPts());
    const auto cell_size = m_geometry.CellSizeArray();
    const RT fast_speed = m_fast_speed;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        const int component = PhysicalNormalComponent(direction);
        for (amrex::MFIter mfi(*m_signal_scratch); mfi.isValid(); ++mfi)
        {
            const auto scratch = m_signal_scratch->array(mfi);
            const auto rho = density.const_array(mfi);
            const auto mom = momentum.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   const RT velocity =
                                       mom(i, j, k, component) /
                                       amrex::max(rho(i, j, k), density_guard);
                                   scratch(i, j, k) = velocity * velocity;
                               });
        }
        const RT velocity_rms =
            std::sqrt(m_signal_scratch->sum(0, false) * inverse_number_of_cells);
        m_signal_speed[direction] = velocity_rms + fast_speed;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_signal_speed[direction]) && m_signal_speed[direction] > 0.0,
            "pc_mhd_block requires positive finite reference signal speeds");
        // Precondition only the UNRESOLVED part of the upwind dissipation:
        // subtract signal_cfl_floor cells per implicit interval from the
        // Reference speed. Below the floor the diffusion switches off
        // exactly: on resolved states the residuals are smooth and any
        // preconditioner content only injects structure that GMRES must
        // then resolve against the Jacobian's rough (grid-scale HLLD)
        // response, degrading iteration counts. Asymptotically (stiff CFL)
        // the subtraction is negligible and the full dissipation is
        // preconditioned.
        const RT excess_speed =
            amrex::max(RT(0.0), m_signal_speed[direction] -
                                    m_signal_cfl_floor * cell_size[direction] / m_h);
        m_signal_diffusion[direction] = m_signal_diffusion_scale * m_h * excess_speed *
                                        RT(0.5) * cell_size[direction];
    }
    m_signal_diffusion_active = false;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_signal_diffusion_active =
            m_signal_diffusion_active || m_signal_diffusion[direction] > 0.0;
    }

    m_evolve_ion_fluid = m_ops->GetMHDEvolveIonFluidForPC();

#if defined(WARPX_DIM_RZ)
    // Engage the momentum wave Schur only when the reference fast wave is
    // actually stiff on the grid (same rationale as the signal-CFL floor: a
    // fixed-cycle multigrid application of a near-identity wave operator is
    // pure spectral pollution on resolved states).
    RT fast_wave_cfl = 0.0;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        fast_wave_cfl = amrex::max(fast_wave_cfl, m_h * fast_speed / cell_size[direction]);
    }
    m_wave_schur_active = m_evolve_ion_fluid && m_include_wave_schur &&
                          fast_wave_cfl >= m_wave_cfl_threshold;
    if (m_wave_schur_active)
    {
        m_wave_operator_rz->setCoefficients(
            m_h * m_h * m_sound_speed_squared,
            m_h * m_h / (PhysConst::mu0 * m_reference_mass_density),
            m_reference_magnetic_field_scale);
    }
#endif

    // The triangular Faraday corrector is exact only when the momentum
    // block it corrects from has been solved; with an identity momentum
    // block and a stiff ideal coupling (Alfven CFL >= 1) the neglected
    // J_MB J_BM product amplifies quadratically and measurably saturates
    // GMRES, so the corrector then stands down.
    RT alfven_wave_cfl = 0.0;
    const RT alfven_speed = m_reference_magnetic_field_scale /
                            std::sqrt(PhysConst::mu0 * m_reference_mass_density);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        alfven_wave_cfl = amrex::max(alfven_wave_cfl, m_h * alfven_speed / cell_size[direction]);
    }
    m_faraday_active = m_evolve_ion_fluid && m_include_ideal_mhd_coupling &&
                       (m_wave_schur_active || alfven_wave_cfl < 1.0);
    if (m_faraday_active)
    {
        // Freeze the cell-centered magnetic field (including any external
        // contribution) at the update state for the triangular Faraday
        // corrector; domain ghosts are never read (the corrector clamps its
        // reads to the domain).
        const amrex::MultiFab* const magnetic_field_cc =
            m_ops->GetMHDMagneticFieldCCForPC();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            magnetic_field_cc != nullptr,
            "pc_mhd_block requires the solver's cell-centered magnetic field "
            "for the Faraday corrector");
        amrex::MultiFab::Copy(*m_faraday_field, *magnetic_field_cc, 0, 0, 3, 0);
        m_faraday_field->FillBoundary(m_geometry.periodicity());
    }

    m_signal_operator->setScalars(1.0, 1.0);
#if defined(WARPX_DIM_RZ)
    // Cylindrical assembly: multiply [I + D (-lap_cyl)] u = b through by r,
    // i.e. a = r_cell, b_r = r_face D_r, b_z = r_cell D_z (and r b on the
    // right-hand side, applied when the RHS is loaded). The axis face has
    // r = 0, which enforces the regularity (zero-flux) condition there.
    const RT radial_lo = static_cast<RT>(m_geometry.ProbLo(0));
    const RT radial_cell_size = cell_size[0];
    const RT radial_diffusion = m_signal_diffusion[0];
    const RT axial_diffusion = m_signal_diffusion[1];
    for (amrex::MFIter mfi(*m_signal_acoef); mfi.isValid(); ++mfi)
    {
        const auto acoef = m_signal_acoef->array(mfi);
        const auto bcoef_r = m_signal_bcoef[0]->array(mfi);
        const auto bcoef_z = m_signal_bcoef[1]->array(mfi);
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::ignore_unused(j, k);
                acoef(i, j, k) =
                    radial_lo + (static_cast<RT>(i) + RT(0.5)) * radial_cell_size;
            });
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect::TheDimensionVector(0)),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::ignore_unused(j, k);
                bcoef_r(i, j, k) =
                    radial_diffusion * (radial_lo + static_cast<RT>(i) * radial_cell_size);
            });
        amrex::ParallelFor(
            amrex::convert(mfi.validbox(), amrex::IntVect::TheDimensionVector(1)),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::ignore_unused(j, k);
                bcoef_z(i, j, k) =
                    axial_diffusion *
                    (radial_lo + (static_cast<RT>(i) + RT(0.5)) * radial_cell_size);
            });
    }
    m_signal_operator->setACoeffs(0, *m_signal_acoef);
#else
    m_signal_operator->setACoeffs(0, 1.0);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_signal_bcoef[direction]->setVal(m_signal_diffusion[direction]);
    }
#endif
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> face_coefficients;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        face_coefficients[direction] = m_signal_bcoef[direction].get();
    }
    // Single-component coefficients broadcast across the stacked block
    // components.
    m_signal_operator->setBCoeffs(0, face_coefficients);

    if (m_verbose)
    {
        amrex::Print() << "Updating pc_mhd_block (hlld state): h = " << m_h
                       << ", a_e = " << m_electron_enthalpy
                       << ", c^2 = " << m_sound_speed_squared
                       << ", c_fast = " << m_fast_speed << ", S_dir = (";
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            amrex::Print() << m_signal_speed[direction]
                           << (direction + 1 < AMREX_SPACEDIM ? ", " : ")");
        }
        amrex::Print() << ", D_dir = (";
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            amrex::Print() << m_signal_diffusion[direction]
                           << (direction + 1 < AMREX_SPACEDIM ? ", " : ")");
        }
        amrex::Print() << ", Faraday = " << (m_faraday_active ? "active" : "inactive")
                       << ", wave Schur = " << (m_wave_schur_active ? "active" : "inactive")
                       << ", resistive = " << (m_resistive_active ? "active" : "inactive")
                       << " (solver = " << m_resistive_solver
                       << ", eta_H = " << (m_resistive_hyper_active ? "on" : "off")
                       << ", z = " << m_resistive_diffusion_number
                       << ", lambda_max = " << m_resistive_lambda_max
                       << ", chebyshev = " << m_resistive_iteration_count;
        if (m_resistive_banded_enabled && m_resistive_banded != nullptr)
        {
            amrex::Print() << ", banded dropped = "
                           << m_resistive_banded->DroppedFraction();
        }
        amrex::Print() << ")\n";
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplySignalDiffusionBlocks (T& solution, const T& rhs)
{
    // Faraday block: identity here; the triangular ideal-induction
    // corrector needs the momentum solution and is added at the end, and
    // the gated resistive Helmholtz then smooths the assembled B content.
    auto const& rhs_fields = rhs.getArrayVec();
    auto const& solution_fields = solution.getArrayVec();
    for (int component = 0; component < 3; ++component)
    {
        amrex::MultiFab::Copy(*solution_fields[0][component], *rhs_fields[0][component], 0, 0, 1,
                              0);
    }

    // Every cell-centered block (rho, M components, U_e, and the
    // closure-dependent ion energies) sees the same reference Helmholtz:
    // stack the block components and solve once.
    int stacked_component = 0;
    for (const auto& spec : rhs.getMultiFabBlockSpecs())
    {
        const amrex::MultiFab& block_rhs = rhs.getMultiFabBlock(spec.name, 0);
        amrex::MultiFab::Copy(*m_signal_rhs, block_rhs, 0, stacked_component,
                              block_rhs.nComp(), 0);
        stacked_component += block_rhs.nComp();
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(stacked_component == m_signal_components,
                                     "pc_mhd_block state block layout changed after Define()");

    // Rigid-conductor fluid freeze of the shaped wall
    // (implicit_mhd.wall_thermal_bc, every wall_model including
    // dielectric): the residual's masked-band fluid rows are exact
    // identities (wall_live) with no fluid coupling across the stair
    // interface (the interface faces present the interior's absorb
    // image), so the band must pass through this preconditioner as
    // identity rows too. Feeding the band into the sweeps instead
    // divides its identity rows by the signal/wave inverses, collapsing
    // their preconditioned eigenvalues by the squared wave CFL --
    // measured as a hard GMRES stall (iteration-capped with no residual
    // reduction) and a 20-50x Newton degradation on every shaped-wall
    // live-fluid solve, whose stagnated updates imprint the leftover
    // residual as a drive-scale B seam along the wall contour. Mask the
    // band OUT of the sweep inputs here and restore the identity rows
    // after the copy-back below.
    const warpx::mhd_pc::WallFluidFreezeView wall_freeze =
        m_ops->GetMHDWallFluidFreezeForPC();
    if (wall_freeze.active)
    {
        const int* const AMREX_RESTRICT first_masked = wall_freeze.first_masked_cc;
        const int mask_z_lo = wall_freeze.z_lo;
        const int mask_z_hi = wall_freeze.z_hi;
        const int number_of_components = m_signal_components;
        for (amrex::MFIter mfi(*m_signal_rhs); mfi.isValid(); ++mfi)
        {
            const auto signal_rhs = m_signal_rhs->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), number_of_components,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    const int jc = amrex::max(mask_z_lo, amrex::min(mask_z_hi, j));
                    if (i >= first_masked[jc])
                    {
                        signal_rhs(i, j, k, n) = RT(0.0);
                    }
                });
        }
    }

    if (m_signal_diffusion_active)
    {
#if defined(WARPX_DIM_RZ)
        // The cylindrical operator assembly multiplied the equation by r.
        const RT radial_lo = static_cast<RT>(m_geometry.ProbLo(0));
        const RT radial_cell_size = static_cast<RT>(m_geometry.CellSize(0));
        const int number_of_components = m_signal_components;
        for (amrex::MFIter mfi(*m_signal_rhs); mfi.isValid(); ++mfi)
        {
            const auto signal_rhs = m_signal_rhs->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), number_of_components,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                {
                    const RT radius =
                        radial_lo + (static_cast<RT>(i) + RT(0.5)) * radial_cell_size;
                    signal_rhs(i, j, k, n) *= radius;
                });
        }
#endif

        m_signal_solution->setVal(0.0);
        m_signal_solver->solve({m_signal_solution.get()}, {m_signal_rhs.get()}, 0.0, 0.0);
    }
    else
    {
        // Every direction is below the signal-CFL floor: the diffusion block
        // is exactly the identity (skipping the fixed-cycle solve keeps it
        // exact rather than V-cycle-approximate).
        amrex::MultiFab::Copy(*m_signal_solution, *m_signal_rhs, 0, 0, m_signal_components,
                              0);
    }

#if defined(WARPX_DIM_RZ)
    if (m_wave_schur_active)
    {
        // Cylindrical momentum wave Schur composed after the scalar signal
        // diffusion: the Helmholtz output is the wave RHS (fixed-cycle MLMG,
        // zero initial guess), so the momentum block sees the product of
        // both approximate inverses.
        const amrex::MultiFab wave_rhs(*m_signal_solution, amrex::make_alias,
                                       m_momentum_component, 3);
        m_momentum_solution->setVal(0.0);
        m_wave_solver_rz->solve({m_momentum_solution.get()}, {&wave_rhs}, 0.0, 0.0);
        amrex::MultiFab::Copy(*m_signal_solution, *m_momentum_solution, 0,
                              m_momentum_component, 3, 0);
    }
#endif

    stacked_component = 0;
    for (const auto& spec : rhs.getMultiFabBlockSpecs())
    {
        amrex::MultiFab& block_solution = solution.getMultiFabBlock(spec.name, 0);
        amrex::MultiFab::Copy(block_solution, *m_signal_solution, stacked_component, 0,
                              block_solution.nComp(), 0);
        if (wall_freeze.active)
        {
            // Identity restore of the masked band (see the wall_freeze
            // comment above): the residual's frozen rows are exact
            // identities, so their preconditioned update is the RHS
            // itself. The Faraday corrector below re-reads the momentum
            // block, so the ideal-induction coupling sees the corrected
            // band values consistently.
            const amrex::MultiFab& block_rhs = rhs.getMultiFabBlock(spec.name, 0);
            const int* const AMREX_RESTRICT first_masked = wall_freeze.first_masked_cc;
            const int mask_z_lo = wall_freeze.z_lo;
            const int mask_z_hi = wall_freeze.z_hi;
            for (amrex::MFIter mfi(block_solution); mfi.isValid(); ++mfi)
            {
                const auto block_out = block_solution.array(mfi);
                const auto block_in = block_rhs.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), block_solution.nComp(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        const int jc = amrex::max(mask_z_lo, amrex::min(mask_z_hi, j));
                        if (i >= first_masked[jc])
                        {
                            block_out(i, j, k, n) = block_in(i, j, k, n);
                        }
                    });
            }
        }
        stacked_component += block_solution.nComp();
    }

    // Shaped-wall FIELD freeze (implicit_mhd.wall_field_freeze): the
    // residual's frozen magnetic-field faces are exact identity rows
    // with no momentum coupling, so their preconditioned update is the
    // RHS itself -- the field twin of the fluid identity restore above.
    // Restored after the Faraday corrector (whose curl writes every
    // face, including frozen ones) so the field-block inverses receive
    // the exact identity RHS there, and again after the resistive block
    // (the banded/direct LU propagates identity rows exactly, but the
    // approximate Chebyshev application only does so to its polynomial
    // accuracy).
    const warpx::mhd_pc::WallFieldFreezeView field_freeze =
        m_ops->GetMHDWallFieldFreezeForPC();
    const auto restore_frozen_faces = [&] ()
    {
        if (!field_freeze.active)
        {
            return;
        }
        const amrex::Array<const int*, 3> first_frozen = {
            field_freeze.first_frozen_br, field_freeze.first_frozen_bt,
            field_freeze.first_frozen_bz};
        for (int component = 0; component < 3; ++component)
        {
            const int* const AMREX_RESTRICT first = first_frozen[component];
            for (amrex::MFIter mfi(*solution_fields[0][component]); mfi.isValid(); ++mfi)
            {
                const auto face = solution_fields[0][component]->array(mfi);
                const auto face_rhs = rhs_fields[0][component]->const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        if (i >= first[j])
                        {
                            face(i, j, k) = face_rhs(i, j, k);
                        }
                    });
            }
        }
    };

    if (m_faraday_active)
    {
        ApplyFaradayCorrector(solution);
        restore_frozen_faces();
    }

    if (m_resistive_active)
    {
        // Composed after the Faraday corrector: the B block then applies
        // the consistent triangular product J_BB^{-1} (bB - J_BM dM).
        ApplyResistiveBlock(solution);
        restore_frozen_faces();
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyFaradayCorrector (T& solution)
{
    // Triangular ideal-induction (Faraday) corrector on the face-staggered
    // B block: dB += theta dt curl(dM/rho_ref x B_cc), with dM the momentum
    // solution and B_cc the cell-centered field frozen at the update (see
    // the class documentation for the sign convention). The curl uses the
    // solver's staggered conventions: (dM x B) is built on the edge/corner
    // locations of E and differenced onto the B faces, with the RZ
    // r-weighted corner-EMF divergence for Bz and a zero azimuthal edge
    // value on the axis corner (odd m = 0 parity, matching the residual's
    // zero axis EMF). Cell-centered reads clamp to the domain: a
    // zero-gradient preconditioner-only approximation at the physical
    // boundaries (interior and periodic ghosts are filled).
    amrex::MultiFab::Copy(*m_momentum_solution,
                          solution.getMultiFabBlock(MomentumDensityName, 0), 0, 0, 3, 0);
    m_momentum_solution->FillBoundary(m_geometry.periodicity());

    auto const& solution_fields = solution.getArrayVec();
    const RT coefficient = m_h / m_reference_mass_density;
    const amrex::Box domain = m_geometry.Domain();
    const auto inverse_cell_size = m_geometry.InvCellSizeArray();

#if defined(WARPX_DIM_1D_Z)
    const int domain_axial_low = domain.smallEnd(0);
    const int domain_axial_high = domain.bigEnd(0);
    const bool axial_periodic = m_geometry.isPeriodic(0);
    const RT inverse_dz = inverse_cell_size[0];

    // Bx and By are z-cell faces; (curl V)_x = -d_z V_y and
    // (curl V)_y = d_z V_x with V on the z-nodal transverse edges. Bz is
    // z-nodal and (curl V)_z vanishes in 1D, so it keeps the identity.
    for (int component = 0; component < 2; ++component)
    {
        const int transverse_component = component == 0 ? 1 : 0;
        const RT sign = component == 0 ? RT(-1.0) : RT(1.0);
        for (amrex::MFIter mfi(*solution_fields[0][component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto face = solution_fields[0][component]->array(mfi);
            const auto dm = m_momentum_solution->const_array(mfi);
            const auto bcc = m_faraday_field->const_array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // (dM x B) component at the z-node between cells
                    // node - 1 and node, averaged from the two cells.
                    const auto edge_value = [&] (const int node) -> RT
                    {
                        int cell_low = node - 1;
                        int cell_high = node;
                        if (!axial_periodic)
                        {
                            cell_low = amrex::max(cell_low, domain_axial_low);
                            cell_high = amrex::min(cell_high, domain_axial_high);
                        }
                        RT momentum_average[3];
                        RT field_average[3];
                        for (int direction = 0; direction < 3; ++direction)
                        {
                            momentum_average[direction] =
                                RT(0.5) * (dm(cell_low, j, k, direction) +
                                           dm(cell_high, j, k, direction));
                            field_average[direction] =
                                RT(0.5) * (bcc(cell_low, j, k, direction) +
                                           bcc(cell_high, j, k, direction));
                        }
                        const int first = (transverse_component + 1) % 3;
                        const int second = (transverse_component + 2) % 3;
                        return momentum_average[first] * field_average[second] -
                               momentum_average[second] * field_average[first];
                    };
                    face(i, j, k) += coefficient * sign * inverse_dz *
                                     (edge_value(i + 1) - edge_value(i));
                });
        }
    }
#elif defined(WARPX_DIM_RZ)
    const int domain_radial_low = domain.smallEnd(0);
    const int domain_radial_high = domain.bigEnd(0);
    const int domain_axial_low = domain.smallEnd(1);
    const int domain_axial_high = domain.bigEnd(1);
    const bool axial_periodic = m_geometry.isPeriodic(1);
    const RT inverse_dr = inverse_cell_size[0];
    const RT inverse_dz = inverse_cell_size[1];
    const RT radial_lower = static_cast<RT>(m_geometry.ProbLo(0));
    const RT radial_cell_size = static_cast<RT>(m_geometry.CellSize(0));

    for (int component = 0; component < 3; ++component)
    {
        for (amrex::MFIter mfi(*solution_fields[0][component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto face = solution_fields[0][component]->array(mfi);
            const auto dm = m_momentum_solution->const_array(mfi);
            const auto bcc = m_faraday_field->const_array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    const auto clamp_radial = [&] (const int cell) -> int
                    {
                        return amrex::max(domain_radial_low,
                                          amrex::min(cell, domain_radial_high));
                    };
                    const auto clamp_axial = [&] (const int cell) -> int
                    {
                        return axial_periodic
                                   ? cell
                                   : amrex::max(domain_axial_low,
                                                amrex::min(cell, domain_axial_high));
                    };
                    // V_t = (dM x B)_t on the (r-node, z-node) corner
                    // (node_r, node_z); zero on the axis corner.
                    const auto corner_azimuthal = [&] (const int node_r,
                                                       const int node_z) -> RT
                    {
                        const RT corner_radius =
                            radial_lower + static_cast<RT>(node_r) * radial_cell_size;
                        if (corner_radius == RT(0.0))
                        {
                            return RT(0.0);
                        }
                        const int cell_r[2] = {clamp_radial(node_r - 1),
                                               clamp_radial(node_r)};
                        const int cell_z[2] = {clamp_axial(node_z - 1),
                                               clamp_axial(node_z)};
                        RT momentum_r = RT(0.0);
                        RT momentum_z = RT(0.0);
                        RT field_r = RT(0.0);
                        RT field_z = RT(0.0);
                        for (int radial_side = 0; radial_side < 2; ++radial_side)
                        {
                            for (int axial_side = 0; axial_side < 2; ++axial_side)
                            {
                                momentum_r += dm(cell_r[radial_side], cell_z[axial_side], k, 0);
                                momentum_z += dm(cell_r[radial_side], cell_z[axial_side], k, 2);
                                field_r += bcc(cell_r[radial_side], cell_z[axial_side], k, 0);
                                field_z += bcc(cell_r[radial_side], cell_z[axial_side], k, 2);
                            }
                        }
                        return RT(0.0625) *
                               (momentum_z * field_r - momentum_r * field_z);
                    };
                    // V_r = (dM x B)_r on the (r-cell, z-node) edge.
                    const auto edge_radial = [&] (const int cell_r,
                                                  const int node_z) -> RT
                    {
                        const int cell_z_low = clamp_axial(node_z - 1);
                        const int cell_z_high = clamp_axial(node_z);
                        const RT momentum_t =
                            RT(0.5) * (dm(cell_r, cell_z_low, k, 1) +
                                       dm(cell_r, cell_z_high, k, 1));
                        const RT momentum_z =
                            RT(0.5) * (dm(cell_r, cell_z_low, k, 2) +
                                       dm(cell_r, cell_z_high, k, 2));
                        const RT field_t = RT(0.5) * (bcc(cell_r, cell_z_low, k, 1) +
                                                      bcc(cell_r, cell_z_high, k, 1));
                        const RT field_z = RT(0.5) * (bcc(cell_r, cell_z_low, k, 2) +
                                                      bcc(cell_r, cell_z_high, k, 2));
                        return momentum_t * field_z - momentum_z * field_t;
                    };
                    // V_z = (dM x B)_z on the (r-node, z-cell) edge.
                    const auto edge_axial = [&] (const int node_r,
                                                 const int cell_z) -> RT
                    {
                        const int cell_r_low = clamp_radial(node_r - 1);
                        const int cell_r_high = clamp_radial(node_r);
                        const RT momentum_r =
                            RT(0.5) * (dm(cell_r_low, cell_z, k, 0) +
                                       dm(cell_r_high, cell_z, k, 0));
                        const RT momentum_t =
                            RT(0.5) * (dm(cell_r_low, cell_z, k, 1) +
                                       dm(cell_r_high, cell_z, k, 1));
                        const RT field_r = RT(0.5) * (bcc(cell_r_low, cell_z, k, 0) +
                                                      bcc(cell_r_high, cell_z, k, 0));
                        const RT field_t = RT(0.5) * (bcc(cell_r_low, cell_z, k, 1) +
                                                      bcc(cell_r_high, cell_z, k, 1));
                        return momentum_r * field_t - momentum_t * field_r;
                    };

                    if (component == 0)
                    {
                        // Br on (r-node, z-cell): (curl V)_r = -d_z V_t.
                        face(i, j, k) -= coefficient * inverse_dz *
                                         (corner_azimuthal(i, j + 1) -
                                          corner_azimuthal(i, j));
                    }
                    else if (component == 1)
                    {
                        // Bt on (r-cell, z-cell):
                        // (curl V)_t = d_z V_r - d_r V_z.
                        face(i, j, k) +=
                            coefficient *
                            (inverse_dz * (edge_radial(i, j + 1) - edge_radial(i, j)) -
                             inverse_dr * (edge_axial(i + 1, j) - edge_axial(i, j)));
                    }
                    else
                    {
                        // Bz on (r-cell, z-node):
                        // (curl V)_z = (1/r) d_r(r V_t), the residual's
                        // r-weighted corner-EMF divergence.
                        const RT radius_center =
                            radial_lower +
                            (static_cast<RT>(i) + RT(0.5)) * radial_cell_size;
                        const RT radius_node_low =
                            radial_lower + static_cast<RT>(i) * radial_cell_size;
                        const RT radius_node_high =
                            radial_lower + (static_cast<RT>(i) + RT(1.0)) * radial_cell_size;
                        face(i, j, k) +=
                            coefficient * inverse_dr / radius_center *
                            (radius_node_high * corner_azimuthal(i + 1, j) -
                             radius_node_low * corner_azimuthal(i, j));
                    }
                });
        }
    }
#else
    amrex::ignore_unused(solution_fields, coefficient, domain, inverse_cell_size);
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block: the hlld Faraday corrector supports 1D and RZ only");
#endif
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyResistiveBlock (T& solution)
{
    BL_PROFILE("MHDBlockPC::ApplyResistiveBlock()");

    // Approximate inverse of I + A_cc (see the class documentation,
    // point 4) applied to the CURRENT B content of the solution (the rhs
    // plus any Faraday correction, so the block composes triangularly
    // with the corrector). Both inner solvers are stationary linear
    // operations, standard right-preconditioned GMRES safe.
    auto const& solution_fields = solution.getArrayVec();

    if (m_resistive_direct_enabled || m_resistive_banded_enabled)
    {
        // Exact block inverse: one forward/backward substitution of the
        // factorization frozen at the preconditioner update.
        amrex::Array<amrex::MultiFab*, 3> faces = {nullptr, nullptr, nullptr};
        for (int component = 0; component < 3; ++component)
        {
            if (ResistiveComponentActive(component))
            {
                faces[component] = solution_fields[0][component];
            }
        }
        if (m_resistive_banded_enabled)
        {
            m_resistive_banded->Solve(faces);
        }
        else
        {
            m_resistive_direct->Solve(faces);
        }
        return;
    }

    // Fixed-count Chebyshev iteration on the interval
    // [1, m_resistive_lambda_max] with zero initial iterate.

    const RT lambda_low = RT(1.0);
    const RT lambda_high = m_resistive_lambda_max;
    const RT center = RT(0.5) * (lambda_high + lambda_low);
    const RT half_width = RT(0.5) * (lambda_high - lambda_low);
    const RT sigma_one = center / half_width;
    RT rho = RT(1.0) / sigma_one;

    for (int component = 0; component < 3; ++component)
    {
        if (!ResistiveComponentActive(component))
        {
            continue;
        }
        // x = 0, r = b, d = r / center.
        m_resistive_iterate[component]->setVal(0.0);
        amrex::MultiFab::Copy(*m_resistive_residual[component],
                              *solution_fields[0][component], 0, 0, 1, 0);
        amrex::MultiFab::LinComb(*m_resistive_direction[component],
                                 RT(1.0) / center, *m_resistive_residual[component], 0,
                                 RT(0.0), *m_resistive_residual[component], 0, 0, 1, 0);
    }

    for (int iteration = 0; iteration < m_resistive_iteration_count; ++iteration)
    {
        // x += d; r -= (I + A_cc) d.
        for (int component = 0; component < 3; ++component)
        {
            if (!ResistiveComponentActive(component))
            {
                continue;
            }
            amrex::MultiFab::Saxpy(*m_resistive_iterate[component], RT(1.0),
                                   *m_resistive_direction[component], 0, 0, 1, 0);
            m_resistive_direction[component]->FillBoundary(m_geometry.periodicity());
        }
        ApplyResistiveCurlCurl();
        const RT rho_next = RT(1.0) / (RT(2.0) * sigma_one - rho);
        for (int component = 0; component < 3; ++component)
        {
            if (!ResistiveComponentActive(component))
            {
                continue;
            }
            amrex::MultiFab::Saxpy(*m_resistive_action[component], RT(1.0),
                                   *m_resistive_direction[component], 0, 0, 1, 0);
            amrex::MultiFab::Saxpy(*m_resistive_residual[component], RT(-1.0),
                                   *m_resistive_action[component], 0, 0, 1, 0);
            // d = rho_next rho d + (2 rho_next / half_width) r.
            amrex::MultiFab::LinComb(*m_resistive_direction[component],
                                     rho_next * rho, *m_resistive_direction[component],
                                     0, RT(2.0) * rho_next / half_width,
                                     *m_resistive_residual[component], 0, 0, 1, 0);
        }
        rho = rho_next;
    }

    for (int component = 0; component < 3; ++component)
    {
        if (!ResistiveComponentActive(component))
        {
            continue;
        }
        amrex::MultiFab::Copy(*solution_fields[0][component],
                              *m_resistive_iterate[component], 0, 0, 1, 0);
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyResistiveCurlCurl ()
{
    // A_cc d = theta_r dt curl((eta_field/mu0) curl d) on the residual's
    // exact Yee staggering, reading the Chebyshev direction (ghosts
    // filled by the caller) and the frozen eta at the electric-field
    // locations. The rows come from the shared stencil emission
    // (MHDResistiveStencil.H) so the matrix-free application and the
    // direct-solver assembly are the same operator by construction,
    // including the boundary row conventions: tangential E on every
    // non-periodic domain boundary plane is zeroed (exact for PEC walls
    // and the residual's zero axis EMF, a benign identity-leaning
    // approximation for Open/None ends), and every remaining read stays
    // within one filled ghost.
#if defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RZ)
    warpx::mhd_pc::ResistiveStencilConstants stencil_constants =
        warpx::mhd_pc::MakeResistiveStencilConstants(m_geometry, m_resistive_h);
    stencil_constants.include_hyper = m_resistive_hyper_active;
    stencil_constants.include_hall = m_resistive_hall_active;
    stencil_constants.hall_scale = m_resistive_hall_scale;
    stencil_constants.include_inertia = m_resistive_inertia_active;
    stencil_constants.inertia_scale = m_resistive_inertia_scale;
#if defined(WARPX_DIM_RZ)
    // Same wall-mask row drops, field-freeze identity rows and z_lo
    // mirror fold as the frozen assemblies (see UpdateResistiveBlock):
    // matrix-free application == assembled operator.
    stencil_constants.wall = m_ops->GetMHDWallMaskForPC();
    stencil_constants.field_freeze = m_ops->GetMHDWallFieldFreezeForPC();
    stencil_constants.axial_lo_mirror = m_ops->GetMHDZLoMirrorForPC();
#endif

    for (int component = 0; component < 3; ++component)
    {
        if (!ResistiveComponentActive(component))
        {
            continue;
        }
        for (amrex::MFIter mfi(*m_resistive_action[component]); mfi.isValid(); ++mfi)
        {
            const auto action = m_resistive_action[component]->array(mfi);
            warpx::mhd_pc::ResistiveStencilArrays stencil_arrays;
            amrex::GpuArray<amrex::Array4<const amrex::Real>, 3> direction;
            for (int field_component = 0; field_component < 3; ++field_component)
            {
                if (m_resistive_eta_edge[field_component] != nullptr)
                {
                    stencil_arrays.eta[field_component] =
                        m_resistive_eta_edge[field_component]->const_array(mfi);
                }
                if (m_resistive_eta_h_edge[field_component] != nullptr)
                {
                    stencil_arrays.eta_h[field_component] =
                        m_resistive_eta_h_edge[field_component]->const_array(mfi);
                }
                if (m_resistive_hall_active &&
                    m_resistive_hall_edge[field_component] != nullptr)
                {
                    stencil_arrays.hall[field_component] =
                        m_resistive_hall_edge[field_component]->const_array(mfi);
                }
                if (m_resistive_inertia_active &&
                    m_resistive_inertia_edge[field_component] != nullptr)
                {
                    stencil_arrays.inertia[field_component] =
                        m_resistive_inertia_edge[field_component]->const_array(mfi);
                }
                if (ResistiveComponentActive(field_component))
                {
                    direction[field_component] =
                        m_resistive_direction[field_component]->const_array(mfi);
                }
            }
            const int row_component = component;
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real row_action = 0.0;
                    warpx::mhd_pc::EmitResistiveRow(
                        stencil_constants, stencil_arrays, row_component, i, j, k,
                        [&] (const int column_component, const int ic, const int jc,
                             const int kc, const amrex::Real coefficient)
                        {
                            row_action +=
                                coefficient * direction[column_component](ic, jc, kc);
                        });
                    action(i, j, k) = row_action;
                });
        }
    }
#else
    WARPX_ABORT_WITH_MESSAGE(
        "pc_mhd_block: the hlld resistive block supports 1D and RZ only");
#endif
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::Update (const T& state)
{
    BL_PROFILE("MHDBlockPC::Update()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(IsDefined(),
                                     "MHDBlockPC::Update() called on an undefined object");

    const RT theta = m_ops->GetMHDThetaForPC();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(theta > 0.0,
                                     "pc_mhd_block requires a positive implicit theta");
    m_h = theta * this->m_dt;
    if (m_hlld_state)
    {
        UpdateResistiveBlock();
        UpdateSignalDiffusion(state);
        return;
    }

    const RT evaluation_time = this->m_time + m_h;
    m_eta_reference = m_ops->GetMHDReferenceResistivityForPC(evaluation_time);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_eta_reference >= 0.0,
                                     "pc_mhd_block requires a non-negative reference resistivity");
    m_field_diffusion_coefficient = m_h * m_eta_reference / PhysConst::mu0;

    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& electron_energy = state.getMultiFabBlock(ElectronEnergyName, 0);
    const RT density_sum = density.sum(0, false);
    const RT energy_sum = electron_energy.sum(0, false);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(density_sum > 0.0,
                                     "pc_mhd_block encountered non-positive total mass density");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(energy_sum >= 0.0,
                                     "pc_mhd_block encountered negative total electron energy");

    const RT chi = m_gamma_e - 1.0;
    m_electron_enthalpy = m_gamma_e * energy_sum / density_sum;
    m_ion_sound_speed_squared = m_gamma_i * m_reference_ion_pressure / m_reference_mass_density;
    m_sound_speed_squared = m_ion_sound_speed_squared + chi * m_electron_enthalpy;
    m_acoustic_active = m_sound_speed_squared > 0.0;
    if (m_acoustic_active)
    {
        m_acoustic_operator->setScalars(1.0 / m_sound_speed_squared, m_h * m_h);
    }

    m_reference_magnetic_field = m_ops->GetMHDReferenceMagneticFieldForPC();
    RT magnetic_field_squared = 0.0;
    for (int component = 0; component < 3; ++component)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(m_reference_magnetic_field[component]),
            "pc_mhd_block encountered a non-finite reference magnetic field");
        magnetic_field_squared += m_reference_magnetic_field[component] *
                                  m_reference_magnetic_field[component];
    }
    const RT activation_threshold = 1.0e-12 * m_reference_magnetic_field_scale;
    const bool magnetic_field_active =
        magnetic_field_squared > activation_threshold * activation_threshold;
    m_evolve_ion_fluid = m_ops->GetMHDEvolveIonFluidForPC();

    const bool include_hall_term = m_ops->GetMHDIncludeHallTermForPC();
    m_hall_mhd_active =
        m_include_hall_mhd_coupling && include_hall_term && magnetic_field_active;
    m_hall_diffusion_coefficient = 0.0;
    if (m_hall_mhd_active)
    {
        m_reference_charge_density = m_ops->GetMHDReferenceChargeDensityForPC();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_reference_charge_density > 0.0,
            "pc_mhd_block requires a positive reference charge density for Hall physics");
        m_hall_diffusion_coefficient =
            m_h * std::sqrt(magnetic_field_squared) /
            (PhysConst::mu0 * m_reference_charge_density);
    }
    m_field_preconditioning_coefficient =
        std::hypot(m_field_diffusion_coefficient, m_hall_diffusion_coefficient);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m_field_preconditioning_coefficient),
        "pc_mhd_block encountered a non-finite field preconditioning coefficient");
    m_curl_curl->setScalars(m_field_preconditioning_coefficient, 1.0);

    m_ideal_mhd_active =
        m_evolve_ion_fluid && m_include_ideal_mhd_coupling && magnetic_field_active;
    // The ideal wave Schur approximates the inverse field block by the
    // identity. That approximation is not robust once the Hall field block is
    // stiff, so Hall-active solves retain only the triangular momentum-to-field
    // correction until a Hall-aware ion Schur operator is available.
    m_ideal_mhd_wave_active = m_ideal_mhd_active && !include_hall_term;

    m_wave_operator->setCoefficients(
        m_h * m_h * m_sound_speed_squared,
        m_h * m_h / (PhysConst::mu0 * m_reference_mass_density),
        m_reference_magnetic_field);

    if (m_verbose)
    {
        amrex::Print() << "Updating pc_mhd_block: h = " << m_h << ", eta_ref = " << m_eta_reference
                       << ", d_eta = " << m_field_diffusion_coefficient
                       << ", d_H = " << m_hall_diffusion_coefficient
                       << ", a_e = " << m_electron_enthalpy
                       << ", ci^2 = " << m_ion_sound_speed_squared
                       << ", c^2 = " << m_sound_speed_squared
                       << ", B_ref = (" << m_reference_magnetic_field[0] << ", "
                       << m_reference_magnetic_field[1] << ", "
                       << m_reference_magnetic_field[2] << "), ideal MHD = "
                       << (m_ideal_mhd_wave_active
                               ? "wave"
                               : (m_ideal_mhd_active ? "triangular" : "inactive"))
                       << ", Hall MHD = " << (m_hall_mhd_active ? "active" : "inactive")
                       << ", ion fluid = " << (m_evolve_ion_fluid ? "evolving" : "frozen")
                       << "\n";
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::Apply (T& solution, const T& rhs)
{
    BL_PROFILE("MHDBlockPC::Apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(IsDefined(),
                                     "MHDBlockPC::Apply() called on an undefined object");
    solution.zero();
    if (m_hlld_state)
    {
        // Every block is written (identity or Helmholtz-preconditioned):
        // a zero write on an unknown block would make the operator singular.
        ApplySignalDiffusionBlocks(solution, rhs);
        return;
    }

    if (!m_evolve_ion_fluid)
    {
        ApplyFrozenIonFluidBlock(solution, rhs);
        BuildFieldRHS(
            rhs, m_include_ideal_mhd_coupling ? m_momentum_solution.get() : nullptr);
        SolveFieldBlock(solution);
        return;
    }

    if (m_ideal_mhd_wave_active)
    {
        ApplyFieldBlock(solution, rhs);
        AddIdealLorentzForceToMomentumRHS(solution, rhs);
        ApplyIdealWaveFluidBlock(solution, rhs);
        BuildFieldRHS(rhs, m_momentum_solution.get());
        SolveFieldBlock(solution);
        return;
    }

    ApplyFluidBlock(solution, rhs);
    BuildFieldRHS(rhs, m_ideal_mhd_active ? m_momentum_solution.get() : nullptr);
    SolveFieldBlock(solution);
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyFrozenIonFluidBlock (T& solution, const T& rhs)
{
    const amrex::MultiFab& density_rhs = rhs.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& momentum_rhs = rhs.getMultiFabBlock(MomentumDensityName, 0);
    const amrex::MultiFab& energy_rhs = rhs.getMultiFabBlock(ElectronEnergyName, 0);
    amrex::MultiFab& density_solution = solution.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_solution = solution.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& energy_solution = solution.getMultiFabBlock(ElectronEnergyName, 0);

    amrex::MultiFab::Copy(density_solution, density_rhs, 0, 0, 1, 0);
    amrex::MultiFab::Copy(momentum_solution, momentum_rhs, 0, 0, 3, 0);
    amrex::MultiFab::Copy(energy_solution, energy_rhs, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*m_momentum_solution, momentum_rhs, 0, 0, 3, 0);
    m_momentum_solution->FillBoundary(m_geometry.periodicity());
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyFieldBlock (T& solution, const T& rhs)
{
    BuildFieldRHS(rhs, nullptr);
    SolveFieldBlock(solution);
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::BuildFieldRHS (const T& rhs, const amrex::MultiFab* const momentum)
{
    auto const& rhs_fields = rhs.getArrayVec();
    if (momentum == nullptr)
    {
        for (int component = 0; component < 3; ++component)
        {
            amrex::MultiFab::Copy(*m_field_rhs[component], *rhs_fields[0][component], 0, 0, 1, 0);
        }
        return;
    }

    amrex::GpuArray<int, 3> cell_staggering = {1, 1, 1};
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        cell_staggering[direction] = 0;
    }
    const amrex::GpuArray<int, 3> coarsening = {1, 1, 1};
    const auto magnetic_field = m_reference_magnetic_field;
    const RT inverse_density = 1.0 / m_reference_mass_density;

    for (int component = 0; component < 3; ++component)
    {
        amrex::GpuArray<int, 3> field_staggering = {1, 1, 1};
        const amrex::IntVect index_type = m_field_rhs[component]->ixType().toIntVect();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            field_staggering[direction] = index_type[direction];
        }
        for (amrex::MFIter mfi(*m_field_rhs[component]); mfi.isValid(); ++mfi)
        {
            const amrex::Box box = mfi.validbox();
            const auto output = m_field_rhs[component]->array(mfi);
            const auto b_electric = rhs_fields[0][component]->const_array(mfi);
            const auto b_momentum = momentum->const_array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    RT velocity[3];
                    for (int direction = 0; direction < 3; ++direction)
                    {
                        velocity[direction] =
                            inverse_density * ablastr::coarsen::sample::Interp(
                                                  b_momentum, cell_staggering, field_staggering,
                                                  coarsening, i, j, k, direction);
                    }
                    const RT velocity_cross_magnetic_field[3] = {
                        velocity[1] * magnetic_field[2] - velocity[2] * magnetic_field[1],
                        velocity[2] * magnetic_field[0] - velocity[0] * magnetic_field[2],
                        velocity[0] * magnetic_field[1] - velocity[1] * magnetic_field[0]};
                    output(i, j, k) =
                        b_electric(i, j, k) - velocity_cross_magnetic_field[component];
                });
        }
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::SolveFieldBlock (T& solution)
{
    auto const& solution_fields = solution.getArrayVec();
    for (int component = 0; component < 3; ++component)
    {
        solution_fields[0][component]->setVal(0.0);
    }

    if (m_field_preconditioning_coefficient == 0.0)
    {
        for (int component = 0; component < 3; ++component)
        {
            amrex::MultiFab::Copy(*solution_fields[0][component], *m_field_rhs[component], 0, 0, 1,
                                  0);
        }
        return;
    }

#if defined(WARPX_DIM_1D_Z)
    VectorMultiFab field_solution{amrex::MultiFab(*solution_fields[0][2], amrex::make_alias, 0, 1),
                                  amrex::MultiFab(*solution_fields[0][1], amrex::make_alias, 0, 1),
                                  amrex::MultiFab(*solution_fields[0][0], amrex::make_alias, 0, 1)};
    VectorMultiFab field_rhs{amrex::MultiFab(*m_field_rhs[2], amrex::make_alias, 0, 1),
                             amrex::MultiFab(*m_field_rhs[1], amrex::make_alias, 0, 1),
                             amrex::MultiFab(*m_field_rhs[0], amrex::make_alias, 0, 1)};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    VectorMultiFab field_solution{amrex::MultiFab(*solution_fields[0][0], amrex::make_alias, 0, 1),
                                  amrex::MultiFab(*solution_fields[0][2], amrex::make_alias, 0, 1),
                                  amrex::MultiFab(*solution_fields[0][1], amrex::make_alias, 0, 1)};
    VectorMultiFab field_rhs{amrex::MultiFab(*m_field_rhs[0], amrex::make_alias, 0, 1),
                             amrex::MultiFab(*m_field_rhs[2], amrex::make_alias, 0, 1),
                             amrex::MultiFab(*m_field_rhs[1], amrex::make_alias, 0, 1)};
#else
    VectorMultiFab field_solution{amrex::MultiFab(*solution_fields[0][0], amrex::make_alias, 0, 1),
                                  amrex::MultiFab(*solution_fields[0][1], amrex::make_alias, 0, 1),
                                  amrex::MultiFab(*solution_fields[0][2], amrex::make_alias, 0, 1)};
    VectorMultiFab field_rhs{amrex::MultiFab(*m_field_rhs[0], amrex::make_alias, 0, 1),
                             amrex::MultiFab(*m_field_rhs[1], amrex::make_alias, 0, 1),
                             amrex::MultiFab(*m_field_rhs[2], amrex::make_alias, 0, 1)};
#endif

    m_curl_curl->prepareRHS({&field_rhs});
    m_field_solver->solve({&field_solution}, {&field_rhs}, 0.0, 0.0);
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ComputeUnitCurlCurlAction (const T& field)
{
    auto const& input_fields = field.getArrayVec();

#if defined(WARPX_DIM_1D_Z)
    VectorMultiFab field_input{amrex::MultiFab(*input_fields[0][2], amrex::make_alias, 0, 1),
                               amrex::MultiFab(*input_fields[0][1], amrex::make_alias, 0, 1),
                               amrex::MultiFab(*input_fields[0][0], amrex::make_alias, 0, 1)};
    VectorMultiFab field_action{
        amrex::MultiFab(*m_field_operator_action[2], amrex::make_alias, 0, 1),
        amrex::MultiFab(*m_field_operator_action[1], amrex::make_alias, 0, 1),
        amrex::MultiFab(*m_field_operator_action[0], amrex::make_alias, 0, 1)};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    VectorMultiFab field_input{amrex::MultiFab(*input_fields[0][0], amrex::make_alias, 0, 1),
                               amrex::MultiFab(*input_fields[0][2], amrex::make_alias, 0, 1),
                               amrex::MultiFab(*input_fields[0][1], amrex::make_alias, 0, 1)};
    VectorMultiFab field_action{
        amrex::MultiFab(*m_field_operator_action[0], amrex::make_alias, 0, 1),
        amrex::MultiFab(*m_field_operator_action[2], amrex::make_alias, 0, 1),
        amrex::MultiFab(*m_field_operator_action[1], amrex::make_alias, 0, 1)};
#else
    VectorMultiFab field_input{amrex::MultiFab(*input_fields[0][0], amrex::make_alias, 0, 1),
                               amrex::MultiFab(*input_fields[0][1], amrex::make_alias, 0, 1),
                               amrex::MultiFab(*input_fields[0][2], amrex::make_alias, 0, 1)};
    VectorMultiFab field_action{
        amrex::MultiFab(*m_field_operator_action[0], amrex::make_alias, 0, 1),
        amrex::MultiFab(*m_field_operator_action[1], amrex::make_alias, 0, 1),
        amrex::MultiFab(*m_field_operator_action[2], amrex::make_alias, 0, 1)};
#endif

    m_unit_curl_curl_apply->apply({&field_action}, {&field_input});
    for (int component = 0; component < 3; ++component)
    {
        m_field_operator_action[component]->minus(*input_fields[0][component], 0, 1, 0);
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::AddIdealLorentzForceToMomentumRHS (const T& field_predictor, const T& rhs)
{
    ComputeUnitCurlCurlAction(field_predictor);

    const amrex::MultiFab& original_momentum_rhs = rhs.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab::Copy(*m_momentum_rhs, original_momentum_rhs, 0, 0, 3, 0);

    amrex::GpuArray<int, 3> cell_staggering = {1, 1, 1};
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        cell_staggering[direction] = 0;
    }
    amrex::GpuArray<amrex::GpuArray<int, 3>, 3> field_staggering;
    for (int component = 0; component < 3; ++component)
    {
        field_staggering[component] = {1, 1, 1};
        const amrex::IntVect index_type =
            m_field_operator_action[component]->ixType().toIntVect();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
        {
            field_staggering[component][direction] = index_type[direction];
        }
    }

    const amrex::GpuArray<int, 3> coarsening = {1, 1, 1};
    const auto magnetic_field = m_reference_magnetic_field;
    const RT force_scale = m_h * m_h / PhysConst::mu0;
    for (amrex::MFIter mfi(*m_momentum_rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto output = m_momentum_rhs->array(mfi);
        const auto action_x = m_field_operator_action[0]->const_array(mfi);
        const auto action_y = m_field_operator_action[1]->const_array(mfi);
        const auto action_z = m_field_operator_action[2]->const_array(mfi);
        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const RT curl_curl[3] = {
                    ablastr::coarsen::sample::Interp(
                        action_x, field_staggering[0], cell_staggering, coarsening, i, j, k, 0),
                    ablastr::coarsen::sample::Interp(
                        action_y, field_staggering[1], cell_staggering, coarsening, i, j, k, 0),
                    ablastr::coarsen::sample::Interp(
                        action_z, field_staggering[2], cell_staggering, coarsening, i, j, k, 0)};
                const RT curl_curl_cross_magnetic_field[3] = {
                    curl_curl[1] * magnetic_field[2] - curl_curl[2] * magnetic_field[1],
                    curl_curl[2] * magnetic_field[0] - curl_curl[0] * magnetic_field[2],
                    curl_curl[0] * magnetic_field[1] - curl_curl[1] * magnetic_field[0]};
                for (int component = 0; component < 3; ++component)
                {
                    output(i, j, k, component) -=
                        force_scale * curl_curl_cross_magnetic_field[component];
                }
            });
    }
    m_momentum_rhs->FillBoundary(m_geometry.periodicity());
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyFluidBlock (T& solution, const T& rhs)
{
    const amrex::MultiFab& density_rhs = rhs.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& momentum_rhs = rhs.getMultiFabBlock(MomentumDensityName, 0);
    const amrex::MultiFab& energy_rhs = rhs.getMultiFabBlock(ElectronEnergyName, 0);
    amrex::MultiFab& density_solution = solution.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_solution = solution.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& energy_solution = solution.getMultiFabBlock(ElectronEnergyName, 0);

    amrex::MultiFab::Copy(*m_momentum_rhs, momentum_rhs, 0, 0, 3, 0);
    m_momentum_rhs->FillBoundary(m_geometry.periodicity());
    m_pressure->setVal(0.0);
    m_pressure_rhs->setVal(0.0);

    const auto inverse_cell_size = m_geometry.InvCellSizeArray();
    const RT h = m_h;
    const RT ci_squared = m_ion_sound_speed_squared;
    const RT chi = m_gamma_e - 1.0;
    const RT c_squared = m_sound_speed_squared;
    const bool acoustic_active = m_acoustic_active;

    for (amrex::MFIter mfi(*m_pressure_rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto b_rho = density_rhs.const_array(mfi);
        const auto b_momentum = m_momentum_rhs->const_array(mfi);
        const auto b_energy = energy_rhs.const_array(mfi);
        const auto pressure_rhs = m_pressure_rhs->array(mfi);
        amrex::ParallelFor(box,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               RT divergence = 0.0;
#if defined(WARPX_DIM_3D)
                               divergence =
                                   0.5 * (b_momentum(i + 1, j, k, 0) - b_momentum(i - 1, j, k, 0)) *
                                       inverse_cell_size[0] +
                                   0.5 * (b_momentum(i, j + 1, k, 1) - b_momentum(i, j - 1, k, 1)) *
                                       inverse_cell_size[1] +
                                   0.5 * (b_momentum(i, j, k + 1, 2) - b_momentum(i, j, k - 1, 2)) *
                                       inverse_cell_size[2];
#elif defined(WARPX_DIM_XZ)
                               divergence =
                                   0.5 * (b_momentum(i + 1, j, k, 0) -
                                          b_momentum(i - 1, j, k, 0)) *
                                       inverse_cell_size[0] +
                                   0.5 * (b_momentum(i, j + 1, k, 2) -
                                          b_momentum(i, j - 1, k, 2)) *
                                       inverse_cell_size[1];
#elif defined(WARPX_DIM_1D_Z)
                               divergence =
                                   0.5 * (b_momentum(i + 1, j, k, 2) -
                                          b_momentum(i - 1, j, k, 2)) *
                                   inverse_cell_size[0];
#else
                               amrex::ignore_unused(i, j, k, b_momentum, inverse_cell_size);
#endif
                               if (acoustic_active)
                               {
                                   const RT pressure_predictor =
                                       ci_squared * b_rho(i, j, k) + chi * b_energy(i, j, k);
                                   pressure_rhs(i, j, k) =
                                       pressure_predictor / c_squared - h * divergence;
                               }
                           });
    }

    if (m_acoustic_active)
    {
        m_fluid_solver->solve({m_pressure.get()}, {m_pressure_rhs.get()}, 0.0, 0.0);
        m_pressure->FillBoundary(m_geometry.periodicity());
    }

    m_momentum_solution->setVal(0.0);
    for (amrex::MFIter mfi(*m_momentum_solution); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto b_momentum = m_momentum_rhs->const_array(mfi);
        const auto pressure = m_pressure->const_array(mfi);
        const auto x_momentum = m_momentum_solution->array(mfi);
        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                RT pressure_gradient[3] = {0.0, 0.0, 0.0};
#if defined(WARPX_DIM_3D)
                pressure_gradient[0] =
                    0.5 * (pressure(i + 1, j, k) - pressure(i - 1, j, k)) * inverse_cell_size[0];
                pressure_gradient[1] =
                    0.5 * (pressure(i, j + 1, k) - pressure(i, j - 1, k)) * inverse_cell_size[1];
                pressure_gradient[2] =
                    0.5 * (pressure(i, j, k + 1) - pressure(i, j, k - 1)) * inverse_cell_size[2];
#elif defined(WARPX_DIM_XZ)
                pressure_gradient[0] =
                    0.5 * (pressure(i + 1, j, k) - pressure(i - 1, j, k)) * inverse_cell_size[0];
                pressure_gradient[2] =
                    0.5 * (pressure(i, j + 1, k) - pressure(i, j - 1, k)) * inverse_cell_size[1];
#elif defined(WARPX_DIM_1D_Z)
                pressure_gradient[2] =
                    0.5 * (pressure(i + 1, j, k) - pressure(i - 1, j, k)) * inverse_cell_size[0];
#else
                amrex::ignore_unused(i, j, k, pressure, inverse_cell_size);
#endif
                for (int component = 0; component < 3; ++component)
                {
                    x_momentum(i, j, k, component) =
                        b_momentum(i, j, k, component) - h * pressure_gradient[component];
                }
            });
    }
    m_momentum_solution->FillBoundary(m_geometry.periodicity());

    const RT electron_enthalpy = m_electron_enthalpy;
    for (amrex::MFIter mfi(density_solution); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto b_rho = density_rhs.const_array(mfi);
        const auto b_energy = energy_rhs.const_array(mfi);
        const auto x_momentum = m_momentum_solution->const_array(mfi);
        const auto x_rho = density_solution.array(mfi);
        const auto x_energy = energy_solution.array(mfi);
        const auto output_momentum = momentum_solution.array(mfi);
        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                RT divergence = 0.0;
#if defined(WARPX_DIM_3D)
                divergence = 0.5 * (x_momentum(i + 1, j, k, 0) - x_momentum(i - 1, j, k, 0)) *
                                 inverse_cell_size[0] +
                             0.5 * (x_momentum(i, j + 1, k, 1) - x_momentum(i, j - 1, k, 1)) *
                                 inverse_cell_size[1] +
                             0.5 * (x_momentum(i, j, k + 1, 2) - x_momentum(i, j, k - 1, 2)) *
                                 inverse_cell_size[2];
#elif defined(WARPX_DIM_XZ)
                divergence = 0.5 * (x_momentum(i + 1, j, k, 0) - x_momentum(i - 1, j, k, 0)) *
                                 inverse_cell_size[0] +
                             0.5 * (x_momentum(i, j + 1, k, 2) - x_momentum(i, j - 1, k, 2)) *
                                 inverse_cell_size[1];
#elif defined(WARPX_DIM_1D_Z)
                divergence = 0.5 * (x_momentum(i + 1, j, k, 2) - x_momentum(i - 1, j, k, 2)) *
                             inverse_cell_size[0];
#else
                amrex::ignore_unused(i, j, k, x_momentum, inverse_cell_size);
#endif
                x_rho(i, j, k) = b_rho(i, j, k) - h * divergence;
                x_energy(i, j, k) = b_energy(i, j, k) - h * electron_enthalpy * divergence;
                for (int component = 0; component < 3; ++component)
                {
                    output_momentum(i, j, k, component) = x_momentum(i, j, k, component);
                }
            });
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplyIdealWaveFluidBlock (T& solution, const T& rhs)
{
    const amrex::MultiFab& density_rhs = rhs.getMultiFabBlock(MassDensityName, 0);
    const amrex::MultiFab& energy_rhs = rhs.getMultiFabBlock(ElectronEnergyName, 0);
    amrex::MultiFab& density_solution = solution.getMultiFabBlock(MassDensityName, 0);
    amrex::MultiFab& momentum_solution = solution.getMultiFabBlock(MomentumDensityName, 0);
    amrex::MultiFab& energy_solution = solution.getMultiFabBlock(ElectronEnergyName, 0);

    const auto inverse_cell_size = m_geometry.InvCellSizeArray();
    const RT h = m_h;
    const RT ci_squared = m_ion_sound_speed_squared;
    const RT chi = m_gamma_e - 1.0;

    m_pressure_rhs->setVal(0.0);
    for (amrex::MFIter mfi(*m_pressure_rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto b_rho = density_rhs.const_array(mfi);
        const auto b_energy = energy_rhs.const_array(mfi);
        const auto pressure_predictor = m_pressure_rhs->array(mfi);
        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                pressure_predictor(i, j, k) =
                    ci_squared * b_rho(i, j, k) + chi * b_energy(i, j, k);
            });
    }
    m_pressure_rhs->FillBoundary(m_geometry.periodicity());

    for (amrex::MFIter mfi(*m_momentum_rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto pressure_predictor = m_pressure_rhs->const_array(mfi);
        const auto wave_rhs = m_momentum_rhs->array(mfi);
        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                RT pressure_gradient[3] = {0.0, 0.0, 0.0};
#if defined(WARPX_DIM_3D)
                pressure_gradient[0] =
                    0.5 * (pressure_predictor(i + 1, j, k) -
                           pressure_predictor(i - 1, j, k)) *
                    inverse_cell_size[0];
                pressure_gradient[1] =
                    0.5 * (pressure_predictor(i, j + 1, k) -
                           pressure_predictor(i, j - 1, k)) *
                    inverse_cell_size[1];
                pressure_gradient[2] =
                    0.5 * (pressure_predictor(i, j, k + 1) -
                           pressure_predictor(i, j, k - 1)) *
                    inverse_cell_size[2];
#elif defined(WARPX_DIM_XZ)
                pressure_gradient[0] =
                    0.5 * (pressure_predictor(i + 1, j, k) -
                           pressure_predictor(i - 1, j, k)) *
                    inverse_cell_size[0];
                pressure_gradient[2] =
                    0.5 * (pressure_predictor(i, j + 1, k) -
                           pressure_predictor(i, j - 1, k)) *
                    inverse_cell_size[1];
#elif defined(WARPX_DIM_1D_Z)
                pressure_gradient[2] =
                    0.5 * (pressure_predictor(i + 1, j, k) -
                           pressure_predictor(i - 1, j, k)) *
                    inverse_cell_size[0];
#else
                amrex::ignore_unused(i, j, k, pressure_predictor, inverse_cell_size);
#endif
                for (int component = 0; component < 3; ++component)
                {
                    wave_rhs(i, j, k, component) -= h * pressure_gradient[component];
                }
            });
    }
    m_momentum_rhs->FillBoundary(m_geometry.periodicity());

    m_momentum_solution->setVal(0.0);
    m_wave_solver->solve({m_momentum_solution.get()}, {m_momentum_rhs.get()}, 0.0, 0.0);
    m_momentum_solution->FillBoundary(m_geometry.periodicity());

    const RT electron_enthalpy = m_electron_enthalpy;
    for (amrex::MFIter mfi(density_solution); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.validbox();
        const auto b_rho = density_rhs.const_array(mfi);
        const auto b_energy = energy_rhs.const_array(mfi);
        const auto x_momentum = m_momentum_solution->const_array(mfi);
        const auto x_rho = density_solution.array(mfi);
        const auto x_energy = energy_solution.array(mfi);
        const auto output_momentum = momentum_solution.array(mfi);
        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                RT divergence = 0.0;
#if defined(WARPX_DIM_3D)
                divergence = 0.5 * (x_momentum(i + 1, j, k, 0) -
                                    x_momentum(i - 1, j, k, 0)) *
                                 inverse_cell_size[0] +
                             0.5 * (x_momentum(i, j + 1, k, 1) -
                                    x_momentum(i, j - 1, k, 1)) *
                                 inverse_cell_size[1] +
                             0.5 * (x_momentum(i, j, k + 1, 2) -
                                    x_momentum(i, j, k - 1, 2)) *
                                 inverse_cell_size[2];
#elif defined(WARPX_DIM_XZ)
                divergence = 0.5 * (x_momentum(i + 1, j, k, 0) -
                                    x_momentum(i - 1, j, k, 0)) *
                                 inverse_cell_size[0] +
                             0.5 * (x_momentum(i, j + 1, k, 2) -
                                    x_momentum(i, j - 1, k, 2)) *
                                 inverse_cell_size[1];
#elif defined(WARPX_DIM_1D_Z)
                divergence = 0.5 * (x_momentum(i + 1, j, k, 2) -
                                    x_momentum(i - 1, j, k, 2)) *
                             inverse_cell_size[0];
#else
                amrex::ignore_unused(i, j, k, x_momentum, inverse_cell_size);
#endif
                x_rho(i, j, k) = b_rho(i, j, k) - h * divergence;
                x_energy(i, j, k) = b_energy(i, j, k) - h * electron_enthalpy * divergence;
                for (int component = 0; component < 3; ++component)
                {
                    output_momentum(i, j, k, component) =
                        x_momentum(i, j, k, component);
                }
            });
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::printParameters () const
{
    const std::string name = amrex::getEnumNameString(PreconditionerType::pc_mhd_block);
    amrex::Print() << name << " verbose:              " << (m_verbose ? "true" : "false") << "\n";
    amrex::Print() << name << " bottom verbose:       " << (m_bottom_verbose ? "true" : "false")
                   << "\n";
    amrex::Print() << name << " ideal MHD coupling:   "
                   << (m_include_ideal_mhd_coupling ? "true" : "false") << "\n";
    amrex::Print() << name << " Hall MHD coupling:    "
                   << (m_include_hall_mhd_coupling ? "true" : "false") << "\n";
    amrex::Print() << name << " field iterations:     " << m_field_iterations << "\n";
    amrex::Print() << name << " fluid iterations:     " << m_fluid_iterations << "\n";
    amrex::Print() << name << " resistive iterations: " << m_resistive_iterations << "\n";
    amrex::Print() << name << " resistive threshold:  " << m_resistive_threshold << "\n";
    amrex::Print() << name << " resistive solver:     " << m_resistive_solver << "\n";
    amrex::Print() << name << " banded precision:     " << m_banded_precision << "\n";
    amrex::Print() << name << " banded refreeze:      " << m_banded_refreeze << "\n";
    amrex::Print() << name << " wave relaxation:      " << m_wave_relaxation << "\n";
    amrex::Print() << name << " agglomeration:        " << (m_agglomeration ? "true" : "false")
                   << "\n";
    amrex::Print() << name << " consolidation:        " << (m_consolidation ? "true" : "false")
                   << "\n";
    amrex::Print() << name << " max coarsening level: " << m_max_coarsening_level << "\n";
}

#endif
