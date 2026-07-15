/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Hybrid implicit magnetic diffusion (FLASH MagDiff-inspired)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "HybridMagDiffusion.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GMRES.H>
#include <AMReX_MLCurlCurl.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <map>
#include <memory>

using namespace amrex;

void
HybridMagDiffusion::ReadParameters ()
{
    const ParmParse pp("hybrid_pic_model");

    pp.query("implicit_mag_diffusion", m_enabled);
    utils::parser::queryWithParser(pp, "mag_diff_theta", m_theta);
    utils::parser::queryWithParser(pp, "mag_diff_rtol", m_rtol);
    utils::parser::queryWithParser(pp, "mag_diff_atol", m_atol);
    pp.query("mag_diff_max_iter", m_max_iter);
    pp.query("mag_diff_verbose", m_verbose);
    utils::parser::queryWithParser(pp, "mag_diff_eta_explicit_max", m_eta_explicit_max);
    pp.query("mag_diff_use_variable_eta", m_use_variable_eta);

    if (utils::parser::queryWithParser(pp, "mag_diff_constant_eta", m_constant_eta)) {
        m_has_constant_eta = true;
    }

    if (!m_enabled) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta > 0.0_rt && m_theta <= 1.0_rt,
        "hybrid_pic_model.mag_diff_theta must be in (0,1] for the MVP "
        "(theta=0 explicit is not supported)");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_rtol > 0.0_rt && m_atol >= 0.0_rt,
        "hybrid_pic_model.mag_diff_rtol/atol must be non-negative (rtol > 0)");

    // MVP: only backward Euler is fully exercised. Crank–Nicolson (theta=0.5)
    // uses the same matrix with a modified RHS assembled via an extra apply.
    amrex::Print() << "HybridMagDiffusion: enabled"
                   << "  theta=" << m_theta
                   << "  rtol=" << m_rtol
                   << "  atol=" << m_atol
                   << "  eta_explicit_max=" << m_eta_explicit_max
                   << "  use_variable_eta=" << m_use_variable_eta;
    if (m_has_constant_eta) {
        amrex::Print() << "  constant_eta=" << m_constant_eta << " Ohm m";
    }
    amrex::Print() << "\n";
}

void
HybridMagDiffusion::GetLinOpBCs (
    Array<LinOpBCType, AMREX_SPACEDIM>& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM>& hibc)
{
    // Map mirrors ProjectionDivCleaner; unmapped types fall back to Neumann
    // (FLASH MagDiff default homogeneous Neumann).
    const std::map<FieldBoundaryType, LinOpBCType> bcmap{
        {FieldBoundaryType::PEC, LinOpBCType::Dirichlet},
        {FieldBoundaryType::Neumann, LinOpBCType::Neumann},
        {FieldBoundaryType::PMC, LinOpBCType::Neumann},
        {FieldBoundaryType::Periodic, LinOpBCType::Periodic},
        {FieldBoundaryType::None, LinOpBCType::Neumann}
    };

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        auto itlo = bcmap.find(WarpX::field_boundary_lo[idim]);
        auto ithi = bcmap.find(WarpX::field_boundary_hi[idim]);
        lobc[idim] = (itlo != bcmap.end()) ? itlo->second : LinOpBCType::Neumann;
        hibc[idim] = (ithi != bcmap.end()) ? ithi->second : LinOpBCType::Neumann;
    }
}

namespace {

/** Alias WarpX (Bx,By,Bz) MultiFabs into AMReX MLCurlCurl component order. */
Array<MultiFab,3>
MakeCurlCurlAliases (Array<MultiFab,3>& mf)
{
#if defined(WARPX_DIM_1D_Z)
    // Missing dimensions are x,y in WarpX and y,z in AMReX
    return {
        MultiFab(mf[2], make_alias, 0, 1),
        MultiFab(mf[1], make_alias, 0, 1),
        MultiFab(mf[0], make_alias, 0, 1)
    };
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    // Missing dimension is y in WarpX and z in AMReX
    return {
        MultiFab(mf[0], make_alias, 0, 1),
        MultiFab(mf[2], make_alias, 0, 1),
        MultiFab(mf[1], make_alias, 0, 1)
    };
#else
    // 3D and RCYLINDER / RSPHERE
    return {
        MultiFab(mf[0], make_alias, 0, 1),
        MultiFab(mf[1], make_alias, 0, 1),
        MultiFab(mf[2], make_alias, 0, 1)
    };
#endif
}

class MagDiffVector
{
public:
    using value_type = Real;

    MagDiffVector () = default;
    MagDiffVector (MagDiffVector const&) = delete;
    MagDiffVector& operator= (MagDiffVector const&) = delete;
    MagDiffVector (MagDiffVector&&) noexcept = default;
    MagDiffVector& operator= (MagDiffVector&&) noexcept = default;

    void Define (ablastr::fields::VectorField const& source)
    {
        for (int idim = 0; idim < 3; ++idim) {
            m_fields[idim].define(
                source[idim]->boxArray(), source[idim]->DistributionMap(),
                source[idim]->nComp(), 0);
        }
        m_is_defined = true;
    }

    void Copy (MagDiffVector const& source)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_fields[idim], source.m_fields[idim], 0, 0,
                           m_fields[idim].nComp(), 0);
        }
    }

    void CopyFrom (ablastr::fields::VectorField const& source)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_fields[idim], *source[idim], 0, 0,
                           m_fields[idim].nComp(), 0);
        }
    }

    void setVal (Real value)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        for (auto& field : m_fields) {
            field.setVal(value);
        }
    }

    void increment (MagDiffVector const& source, Real scale)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Saxpy(m_fields[idim], scale, source.m_fields[idim],
                            0, 0, m_fields[idim].nComp(), 0);
        }
    }

    void scale (Real scale)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined);
        for (auto& field : m_fields) {
            field.mult(scale, 0, field.nComp(), 0);
        }
    }

    void linComb (Real a, MagDiffVector const& x, Real b, MagDiffVector const& y)
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && x.m_is_defined && y.m_is_defined);
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::LinComb(m_fields[idim], a, x.m_fields[idim], 0,
                               b, y.m_fields[idim], 0, 0,
                               m_fields[idim].nComp(), 0);
        }
    }

    [[nodiscard]] Real dotProduct (MagDiffVector const& source) const
    {
        AMREX_ALWAYS_ASSERT(m_is_defined && source.m_is_defined);
        Real result = 0.0_rt;
        for (int idim = 0; idim < 3; ++idim) {
            result += MultiFab::Dot(m_fields[idim], 0, source.m_fields[idim],
                                    0, m_fields[idim].nComp(), 0);
        }
        return result;
    }

    [[nodiscard]] Array<MultiFab,3>& fields () { return m_fields; }
    [[nodiscard]] Array<MultiFab,3> const& fields () const { return m_fields; }

private:
    Array<MultiFab,3> m_fields;
    bool m_is_defined = false;
};

class VariableCoeffMagDiffusionOp
{
public:
    using RT = Real;

    VariableCoeffMagDiffusionOp (
        ablastr::fields::VectorField const& Bfield,
        ablastr::fields::VectorField const& eta,
        Real theta_dt, int lev)
        : m_theta_dt(theta_dt), m_lev(lev),
          m_source{Bfield[0], Bfield[1], Bfield[2]},
          m_eta{eta[0], eta[1], eta[2]},
          m_geom(WarpX::GetInstance().Geom(lev))
    {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        WARPX_ABORT_WITH_MESSAGE(
            "Variable-coefficient hybrid magnetic diffusion is not yet supported "
            "in cylindrical or spherical geometries");
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_geom.isAllPeriodic(),
            "Variable-coefficient hybrid magnetic diffusion currently requires "
            "periodic field boundaries");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !EB::enabled(),
            "Variable-coefficient hybrid magnetic diffusion does not support "
            "embedded boundaries yet");

        auto& warpx = WarpX::GetInstance();
        m_fdtd = warpx.get_pointer_fdtd_solver_fp(lev);
        m_eb_update_E = &warpx.GetEBUpdateEFlag()[lev];
        m_eb_update_B = &warpx.GetEBUpdateBFlag()[lev];

        for (int idim = 0; idim < 3; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                Bfield[idim]->nComp() == 1 && eta[idim]->nComp() == 1,
                "Variable-coefficient hybrid magnetic diffusion supports one "
                "field component per Yee direction");
            m_Bwork[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(),
                                 1, Bfield[idim]->nGrowVect());
            m_Jwork[idim].define(eta[idim]->boxArray(), eta[idim]->DistributionMap(),
                                 1, eta[idim]->nGrowVect());
            m_etaJ[idim].define(eta[idim]->boxArray(), eta[idim]->DistributionMap(),
                                1, eta[idim]->nGrowVect());
            m_curl_etaJ[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(),
                                     1, Bfield[idim]->nGrowVect());
        }
    }

    [[nodiscard]] MagDiffVector makeVecRHS () const
    {
        MagDiffVector result;
        result.Define({m_source[0], m_source[1], m_source[2]});
        return result;
    }

    [[nodiscard]] MagDiffVector makeVecLHS () const
    {
        return makeVecRHS();
    }

    void assign (MagDiffVector& destination, MagDiffVector const& source)
    {
        destination.Copy(source);
    }

    void setToZero (MagDiffVector& vector)
    {
        vector.setVal(0.0_rt);
    }

    void increment (MagDiffVector& destination, MagDiffVector const& source, Real scale)
    {
        destination.increment(source, scale);
    }

    void scale (MagDiffVector& vector, Real scale_factor)
    {
        vector.scale(scale_factor);
    }

    void linComb (MagDiffVector& destination, Real a, MagDiffVector const& x,
                  Real b, MagDiffVector const& y)
    {
        destination.linComb(a, x, b, y);
    }

    [[nodiscard]] Real dotProduct (MagDiffVector const& x, MagDiffVector const& y) const
    {
        return x.dotProduct(y);
    }

    [[nodiscard]] Real norm2 (MagDiffVector const& vector) const
    {
        return std::sqrt(vector.dotProduct(vector));
    }

    void precond (MagDiffVector& destination, MagDiffVector const& source)
    {
        destination.Copy(source);
    }

    void apply (MagDiffVector& output, MagDiffVector const& input)
    {
        auto const& input_fields = input.fields();
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(m_Bwork[idim], input_fields[idim], 0, 0, 1, 0);
            m_Bwork[idim].FillBoundaryAndSync(m_geom.periodicity());
        }

        ablastr::fields::VectorField Bwork = {
            &m_Bwork[0], &m_Bwork[1], &m_Bwork[2]};
        ablastr::fields::VectorField Jwork = {
            &m_Jwork[0], &m_Jwork[1], &m_Jwork[2]};
        m_fdtd->CalculateCurrentAmpere(Jwork, Bwork, *m_eb_update_E, m_lev);

        for (int idim = 0; idim < 3; ++idim) {
            for (MFIter mfi(m_etaJ[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const etaJ = m_etaJ[idim].array(mfi);
                auto const eta = m_eta[idim]->const_array(mfi);
                auto const J = m_Jwork[idim].const_array(mfi);
                ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    etaJ(i, j, k) = eta(i, j, k) * J(i, j, k);
                });
            }
            m_etaJ[idim].FillBoundaryAndSync(m_geom.periodicity());
        }

        ablastr::fields::VectorField etaJ = {
            &m_etaJ[0], &m_etaJ[1], &m_etaJ[2]};
        ablastr::fields::VectorField curl_etaJ = {
            &m_curl_etaJ[0], &m_curl_etaJ[1], &m_curl_etaJ[2]};
        m_fdtd->ComputeCurlA(curl_etaJ, etaJ, *m_eb_update_B, m_lev);

        auto& output_fields = output.fields();
        for (int idim = 0; idim < 3; ++idim) {
            m_curl_etaJ[idim].OverrideSync(m_geom.periodicity());
            MultiFab::Copy(output_fields[idim], input_fields[idim], 0, 0, 1, 0);
            MultiFab::Saxpy(output_fields[idim], m_theta_dt, m_curl_etaJ[idim],
                            0, 0, 1, 0);
        }
    }

private:
    Real m_theta_dt;
    int m_lev;
    Array<MultiFab*,3> m_source;
    Array<MultiFab const*,3> m_eta;
    Geometry const& m_geom;
    FiniteDifferenceSolver* m_fdtd = nullptr;
    std::array<std::unique_ptr<iMultiFab>,3> const* m_eb_update_E = nullptr;
    std::array<std::unique_ptr<iMultiFab>,3> const* m_eb_update_B = nullptr;
    Array<MultiFab,3> m_Bwork;
    Array<MultiFab,3> m_Jwork;
    Array<MultiFab,3> m_etaJ;
    Array<MultiFab,3> m_curl_etaJ;
};

} // namespace

void
HybridMagDiffusion::Advance (
    ablastr::fields::VectorField const& Bfield,
    Real eta_SI,
    Real dt,
    int lev,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc) const
{
    ABLASTR_PROFILE("HybridMagDiffusion::Advance()");

    if (!m_enabled) { return; }
    if (dt <= 0.0_rt) { return; }
    if (eta_SI <= 0.0_rt) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "HybridMagDiffusion only supports single-level hybrid runs");

    // Magnetic diffusivity chi = eta / mu0  [m^2/s]
    const Real chi = eta_SI / PhysConst::mu0;
    // MLCurlCurl: curl(alpha curl B) + beta B = rhs
    // with alpha = theta * dt * chi, beta = 1  =>  (I + theta dt chi curlcurl) B
    const Real alpha = m_theta * dt * chi;

    if (m_verbose > 0) {
        amrex::Print() << "HybridMagDiffusion::Advance: eta=" << eta_SI
                       << " chi=" << chi
                       << " dt=" << dt
                       << " alpha=theta*dt*chi=" << alpha << "\n";
    }

    auto& warpx = WarpX::GetInstance();
    Geometry const& geom = warpx.Geom(lev);

    // MLCurlCurl expects cell-centered BoxArray (enclosedCells of edge BA)
    BoxArray ba = Bfield[0]->boxArray();
    ba.enclosedCells();
    DistributionMapping const& dm = Bfield[0]->DistributionMap();

    LPInfo info;
    info.setMaxCoarseningLevel(30);

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    constexpr int coord = 1;
#else
    constexpr int coord = 0;
#endif

    Vector<Geometry> geom_v{geom};
    Vector<BoxArray> grids_v{ba};
    Vector<DistributionMapping> dmap_v{dm};

    MLCurlCurl linop(geom_v, grids_v, dmap_v, info, coord);
    linop.setDomainBC(lobc, hibc);
    linop.setScalars(alpha, Real(1.0));

    // Working storage in WarpX component order
    Array<MultiFab, 3> sol;
    Array<MultiFab, 3> rhs;
    for (int idim = 0; idim < 3; ++idim) {
        const IntVect ng = Bfield[idim]->nGrowVect();
        sol[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(), 1, ng);
        rhs[idim].define(Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(), 1, ng);
        MultiFab::Copy(sol[idim], *Bfield[idim], 0, 0, 1, ng);
        MultiFab::Copy(rhs[idim], *Bfield[idim], 0, 0, 1, ng);
    }

    // For 0 < theta < 1 (e.g. Crank–Nicolson):
    //   rhs = B - (1-theta) dt chi curlcurl B
    //       = 2 B - A_e(B)
    // where A_e uses alpha_e = (1-theta) dt chi and beta = 1.
    if (m_theta < 1.0_rt) {
        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                rhs[idim],
                WarpX::do_single_precision_comms,
                geom.periodicity(),
                true);
        }

        const Real alpha_e = (1.0_rt - m_theta) * dt * chi;
        MLCurlCurl linop_e(geom_v, grids_v, dmap_v, info, coord);
        linop_e.setDomainBC(lobc, hibc);
        linop_e.setScalars(alpha_e, Real(1.0));
        linop_e.setLevelBC(0, nullptr);
        linop_e.prepareForSolve();

        Array<MultiFab, 3> Ae_out;
        for (int idim = 0; idim < 3; ++idim) {
            Ae_out[idim].define(
                Bfield[idim]->boxArray(), Bfield[idim]->DistributionMap(), 1, 0);
            Ae_out[idim].setVal(0.0_rt);
        }

        Array<MultiFab, 3> in_arr = MakeCurlCurlAliases(rhs);
        Array<MultiFab, 3> out_arr = MakeCurlCurlAliases(Ae_out);

        using Op = MLLinOpT<Array<MultiFab,3>>;
        linop_e.apply(0, 0, out_arr, in_arr, Op::BCMode::Homogeneous, Op::StateMode::Solution);

        for (int idim = 0; idim < 3; ++idim) {
            // rhs := 2*B - A_e(B)
            MultiFab::LinComb(
                rhs[idim],
                2.0_rt, sol[idim], 0,
                -1.0_rt, Ae_out[idim], 0,
                0, 1, 0);
        }
    }

    Array<MultiFab, 3> solution = MakeCurlCurlAliases(sol);
    Array<MultiFab, 3> rhs_arr = MakeCurlCurlAliases(rhs);

    linop.setLevelBC(0, nullptr);
    linop.prepareRHS({&rhs_arr});

    using MFArr = Array<MultiFab, 3>;
    MLMGT<MFArr> mlmg(linop);
    mlmg.setMaxIter(m_max_iter);
    mlmg.setVerbose(m_verbose);
    mlmg.setBottomVerbose(0);
    const Real residual = mlmg.solve({&solution}, {&rhs_arr}, m_rtol, m_atol);

    if (m_verbose > 0) {
        amrex::Print() << "HybridMagDiffusion MLMG residual=" << residual << "\n";
    }

    for (int idim = 0; idim < 3; ++idim) {
        MultiFab::Copy(*Bfield[idim], sol[idim], 0, 0, 1, 0);
        ablastr::utils::communication::FillBoundary(
            *Bfield[idim],
            WarpX::do_single_precision_comms,
            geom.periodicity(),
            true);
    }
}

void
HybridMagDiffusion::AdvanceVariable (
    ablastr::fields::VectorField const& Bfield,
    ablastr::fields::VectorField const& eta_SI,
    Real dt,
    int lev,
    Array<LinOpBCType, AMREX_SPACEDIM> const& lobc,
    Array<LinOpBCType, AMREX_SPACEDIM> const& hibc) const
{
    ABLASTR_PROFILE("HybridMagDiffusion::AdvanceVariable()");

    amrex::ignore_unused(lobc, hibc);
    if (!m_enabled || dt <= 0.0_rt) { return; }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0,
        "HybridMagDiffusion only supports single-level hybrid runs");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta == 1.0_rt,
        "Variable-coefficient hybrid magnetic diffusion currently supports "
        "only backward Euler (mag_diff_theta = 1)");

    VariableCoeffMagDiffusionOp linop(Bfield, eta_SI, m_theta * dt, lev);
    MagDiffVector solution;
    MagDiffVector rhs;
    solution.Define(Bfield);
    rhs.Define(Bfield);
    solution.CopyFrom(Bfield);
    rhs.CopyFrom(Bfield);

    amrex::GMRES<MagDiffVector, VariableCoeffMagDiffusionOp> solver;
    solver.define(linop);
    solver.setMaxIters(m_max_iter);
    solver.setVerbose(m_verbose);
    solver.solve(solution, rhs, m_rtol, m_atol);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        solver.getStatus() == 0,
        "Variable-coefficient hybrid magnetic diffusion GMRES did not converge");

    if (m_verbose > 0) {
        amrex::Print() << "HybridMagDiffusion variable-eta GMRES iterations="
                       << solver.getNumIters()
                       << " residual=" << solver.getResidualNorm() << "\n";
    }

    auto const& solution_fields = solution.fields();
    auto& warpx = WarpX::GetInstance();
    Geometry const& geom = warpx.Geom(lev);
    for (int idim = 0; idim < 3; ++idim) {
        MultiFab::Copy(*Bfield[idim], solution_fields[idim], 0, 0, 1, 0);
        ablastr::utils::communication::FillBoundary(
            *Bfield[idim],
            WarpX::do_single_precision_comms,
            geom.periodicity(),
            true);
    }
}
