/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MHD_BLOCK_PC_H_
#define MHD_BLOCK_PC_H_

#include "Fields.H"
#include "MHDWaveOp.H"
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
 * carries face-staggered B^{n+theta} instead of E. Stage-1 treatment: the
 * Faraday block is the identity, and every cell-centered block (rho, M,
 * U_e, and the ion-closure energies) is preconditioned by the scalar
 * signal-diffusion Helmholtz
 *
 *     [I + theta dt S_ref (h_dir/2) (-lap)] dU = bU,
 *
 * where S_ref = |u|_ref + sqrt(c_s,ref^2 + v_A,ref^2) is the domain-reference
 * signal speed and h_dir the cell size. This stands in for the
 * O(theta dt |S|/h) HLLD upwind dissipation, the stiffest near-wall Jacobian
 * content. In RZ the Helmholtz is assembled in cylindrical form by
 * multiplying the equation through by r (a = r, b_r = r_face D_r,
 * b_z = r_cell D_z). Boundary types the residual handles itself are mapped
 * to preconditioner-only linear-operator types: Open -> homogeneous
 * Dirichlet and None (solver-managed zero-gradient ghosts) -> Neumann.
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

        int m_field_iterations = 2;
        int m_fluid_iterations = 2;
        int m_max_coarsening_level = 30;
        int m_signal_components = 0;

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
        RT m_signal_speed = 0.0;
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

        void ReadParameters ();
        void ApplyFieldBlock (T& solution, const T& rhs);
        void SolveFieldBlock (T& solution);
        void ComputeUnitCurlCurlAction (const T& field);
        void DefineSignalDiffusion (const T& state);
        [[nodiscard]] amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
        GetCellCenteredLinOpBC (bool a_high_side) const;
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
    utils::parser::queryWithParser(pp, "field_iterations", m_field_iterations);
    utils::parser::queryWithParser(pp, "fluid_iterations", m_fluid_iterations);
    utils::parser::queryWithParser(pp, "max_coarsening_level", m_max_coarsening_level);
    utils::parser::queryWithParser(pp, "wave_relaxation", m_wave_relaxation);

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
        // Conservative-form recast: the Faraday block is the identity and
        // the cell-centered blocks share one signal-diffusion Helmholtz, so
        // none of the E-based operators below are constructed (their domain
        // boundary conversion also rejects Open/None boundaries, which the
        // hlld residual supports).
        DefineSignalDiffusion(state);
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
MHDBlockPC<T, Ops>::GetCellCenteredLinOpBC (const bool a_high_side) const
{
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
                 field_bc[direction] == FieldBoundaryType::Open)
        {
            // Open is a preconditioner-only approximation: homogeneous
            // Dirichlet stands in for the Green's-function coupling, which
            // the nonlinear residual retains exactly.
            linop_bc[direction] = amrex::LinOpBCType::Dirichlet;
        }
        else if (field_bc[direction] == FieldBoundaryType::Neumann ||
                 field_bc[direction] == FieldBoundaryType::None)
        {
            // None marks solver-managed zero-gradient ghosts (the RZ axis
            // and outflow z ends); the preconditioner treats them as
            // homogeneous Neumann.
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
void
MHDBlockPC<T, Ops>::DefineSignalDiffusion (const T& state)
{
    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);

    m_signal_components = 0;
    for (const auto& spec : state.getMultiFabBlockSpecs())
    {
        m_signal_components += state.getMultiFabBlock(spec.name, 0).nComp();
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_signal_components > 0,
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

    m_signal_operator = std::make_unique<amrex::MLABecLaplacian>(
        amrex::Vector<amrex::Geometry>{m_geometry},
        amrex::Vector<amrex::BoxArray>{density.boxArray()},
        amrex::Vector<amrex::DistributionMapping>{density.DistributionMap()}, *m_info,
        amrex::Vector<amrex::FabFactory<amrex::FArrayBox> const*>{}, m_signal_components);
    m_signal_operator->setDomainBC(GetCellCenteredLinOpBC(false), GetCellCenteredLinOpBC(true));
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
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::UpdateSignalDiffusion (const T& state)
{
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

    // Domain-reference fast speed from the INPUT reference field scale: the
    // domain-mean B is near zero for reversed-field states, which would
    // deactivate the Alfven contribution exactly where it is stiff.
    const RT alfven_speed_squared = m_reference_magnetic_field_scale *
                                    m_reference_magnetic_field_scale /
                                    (PhysConst::mu0 * m_reference_mass_density);
    m_signal_speed =
        m_reference_velocity + std::sqrt(m_sound_speed_squared + alfven_speed_squared);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m_signal_speed) && m_signal_speed > 0.0,
        "pc_mhd_block requires a positive finite reference signal speed");

    const auto cell_size = m_geometry.CellSizeArray();
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction)
    {
        m_signal_diffusion[direction] = m_h * m_signal_speed * RT(0.5) * cell_size[direction];
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
                       << ", S_ref = " << m_signal_speed << "\n";
    }
}

template <class T, class Ops>
void
MHDBlockPC<T, Ops>::ApplySignalDiffusionBlocks (T& solution, const T& rhs)
{
    // Faraday block: identity. TODO(Stage 2): triangular corrector
    // bB - theta dt curl(dM/rho_ref x B_ref) on faces, which needs the
    // solver's staggered curl and the RZ corner-EMF conventions.
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

    stacked_component = 0;
    for (const auto& spec : rhs.getMultiFabBlockSpecs())
    {
        amrex::MultiFab& block_solution = solution.getMultiFabBlock(spec.name, 0);
        amrex::MultiFab::Copy(block_solution, *m_signal_solution, stacked_component, 0,
                              block_solution.nComp(), 0);
        stacked_component += block_solution.nComp();
    }
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
    amrex::Print() << name << " wave relaxation:      " << m_wave_relaxation << "\n";
    amrex::Print() << name << " agglomeration:        " << (m_agglomeration ? "true" : "false")
                   << "\n";
    amrex::Print() << name << " consolidation:        " << (m_consolidation ? "true" : "false")
                   << "\n";
    amrex::Print() << name << " max coarsening level: " << m_max_coarsening_level << "\n";
}

#endif
