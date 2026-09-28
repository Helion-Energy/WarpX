/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <memory>
#include <string>

using Real = amrex::Real;
using MF = amrex::MultiFab;
using ablastr::fields::Direction;
using warpx::fields::FieldType;
using Owned = std::array<std::unique_ptr<MF>, 3>;

// Only selects the native physical-current units in PreLinearSolve. No test
// solver/inverse replaces the matrix allocation, deposition, reduction or push.
class PhysicalPCType final
    : public NonlinearSolver<WarpXSolverVec, ImplicitSolver> {
  public:
    void
    Define (const WarpXSolverVec&, ImplicitSolver*) override {}
    void
    Solve (WarpXSolverVec&, const WarpXSolverVec&, Real, Real,
           int) const override {
        amrex::Abort("The fixture never invokes a nonlinear solver");
    }
    void
    PrintParams () const override {}
    void
    GetSolverParams (Real& r, Real& a, int& n) override {
        r = 0;
        a = 0;
        n = 0;
    }
    PreconditionerType
    GetPreconditionerType () const override {
        return PreconditionerType::pc_hybrid_pic;
    }
};
class ParticlePCProbe final : public ThetaImplicitHybrid {
  public:
    void
    Configure (WarpX& sim, bool jacobian) {
        m_WarpX = &sim;
        m_num_amr_levels = 1;
        m_dt = sim.getdt(0);
        m_theta = .5;
        m_nlsolver_type = NonlinearSolverType::newton;
        m_nlsolver = std::make_unique<PhysicalPCType>();
        m_max_particle_iterations = 100;
        m_particle_tolerance = 1.e-12;
        m_use_mass_matrices = true;
        m_use_mass_matrices_jacobian = jacobian;
        m_use_mass_matrices_pc = true;
        m_esirkepov_mass_matrices = true;
        m_mass_matrices_pc_width = 0;
        auto const count =
            sim.m_fields.get(FieldType::MassMatrices_X, Direction{0}, 0)
                ->nComp();
        int const width = static_cast<int>(std::sqrt(count));
        AMREX_ALWAYS_ASSERT(width >= 2 * WarpX::nox + 1 &&
                            width * width == count);
        m_ncomp_xx = m_ncomp_xy = m_ncomp_xz = amrex::IntVect(width);
        m_ncomp_yx = m_ncomp_yy = m_ncomp_yz = amrex::IntVect(width);
        m_ncomp_zx = m_ncomp_zy = m_ncomp_zz = amrex::IntVect(width);
        m_ncomp_pc_xx = m_ncomp_pc_yy = m_ncomp_pc_zz = amrex::IntVect(1);
        AMREX_ALWAYS_ASSERT(NeedFullCurrentResponse());
    }
    void
    FullParticleProbes () {
        m_use_mass_matrices_jacobian = false;
    }
    void
    Evaluate (bool signed_probe) {
        PreRHSOp(.5 * m_dt, 1, signed_probe);
    }
    void
    Diagonal () {
        SyncMassMatricesPCAndApplyBCs();
    }
    void
    CorruptWidth () {
        m_ncomp_xy = amrex::IntVect(m_ncomp_xx[0] + 2);
    }
    void
    MatrixAction () {
        auto out =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, 0);
        auto const in =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_aux, 0);
        ApplyMassMatrices(out, in, nullptr, nullptr, 1., true);
        m_WarpX->ApplyInverseVolumeScalingToCurrentDensity(out[0][0], out[0][1],
                                                           out[0][2], 0);
        m_WarpX->SyncCurrent("current_fp");
        m_WarpX->ApplyJfieldBoundary(0, out[0][0], out[0][1], out[0][2],
                                     PatchType::fine);
    }
};
MF
CloneBand (const MF& f) {
    MF result(f.boxArray(), f.DistributionMap(), f.nComp(), f.nGrowVect());
    MF::Copy(result, f, 0, 0, f.nComp(), f.nGrowVect());
    return result;
}
Real
BandError (const MF& a, const MF& b) {
    auto diff = CloneBand(a);
    MF::Subtract(diff, b, 0, 0, a.nComp(), a.nGrowVect());
    return diff.norminf(0, a.nComp(), a.nGrowVect());
}
MF
Clone (const MF& f, int component = 0) {
    MF result(f.boxArray(), f.DistributionMap(), 1, f.nGrowVect());
    MF::Copy(result, f, component, 0, 1, f.nGrowVect());
    return result;
}
Real
Error (const MF& a, const MF& b, bool ghosts = false) {
    auto diff = Clone(a);
    MF::Subtract(diff, b, 0, 0, 1, ghosts ? a.nGrowVect() : amrex::IntVect(0));
    return diff.norminf(0, 1, ghosts ? a.nGrowVect() : amrex::IntVect(0));
}
void FillGather (WarpX& sim);
void
FillElectric (WarpX& sim, Real amplitude) {
    auto const electric = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const dx = sim.Geom(0).CellSizeArray(), lo = sim.Geom(0).ProbLoArray();
    Real const kr = .5 * std::acos(-1.) / sim.Geom(0).ProbLength(0);
    Real const kz = 2 * std::acos(-1.) / sim.Geom(0).ProbLength(1);
    for (int c = 0; c < 3; ++c) {
        auto const nodal = electric[c]->ixType().toIntVect();
        for (amrex::MFIter mfi(*electric[c]); mfi.isValid(); ++mfi) {
            auto const e = electric[c]->array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                  int k) {
                Real const r = (i + .5 * (1 - nodal[0])) * dx[0],
                           z = lo[1] + (j + .5 * (1 - nodal[1])) * dx[1];
                e(i, j, k) = amplitude *
                             (c == 2 ? std::cos(kr * r) : std::sin(kr * r)) *
                             (1 + .2 * std::sin(kz * z));
            });
        }
    }
    FillGather(sim);
}
void
FillGather (WarpX& sim) {
    auto const e = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    sim.FillBoundaryE(0, sim.get_ng_fieldgather());
    sim.ApplyEfieldBoundary(0, PatchType::fine, 0.);
    sim.ApplyFieldBoundaryOnAxis(e[0], e[1], e[2], 0);
    sim.UpdateAuxiliaryData();
    sim.FillBoundaryAux(sim.getngUpdateAux());
}
void
Run (WarpX& sim) {
    std::string mode = "pc_only";
    amrex::ParmParse("mm_probe").query("mode", mode);
    bool const none = mode == "none" || mode == "effective_none";
    if (none) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !sim.m_fields.has(FieldType::MassMatrices_X, Direction{0}, 0),
            "No effective MM consumer must allocate no current-response bands");
        amrex::Print() << "PASS MM_PC " << mode << " no response allocation\n";
        return;
    }
    AMREX_ALWAYS_ASSERT(mode == "pc_only" || mode == "fallback" ||
                        mode == "matrix" || mode == "bad_width" ||
                        mode == "bad_band");
    auto const current = sim.m_fields.get_alldirs(FieldType::current_fp, 0);
    auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
    int const mid = rho.nComp() / 2;
    auto old = Clone(rho);
    ablastr::fields::MultiLevelScalarField old_refs{&old};
    sim.GetPartContainer().DepositCharge(old_refs, 0.);
    sim.SyncRho(old_refs, {}, {});
    sim.ApplyRhofieldBoundary(0, &old, PatchType::fine);
    old.FillBoundary(sim.Geom(0).periodicity());
    for (int c = 0; c < 3; ++c)
        sim.m_fields.get(FieldType::Bfield_fp, Direction{c}, 0)
            ->setVal(c == 2 ? .2 : .03);
    sim.SaveParticlesAtImplicitStepStart();
    MassMatrixDensityProjection density;
    density.CaptureStepStart(old, 0);
    ParticlePCProbe probe;
    probe.Configure(sim, mode == "matrix");
    FillElectric(sim, 0.);
    probe.Evaluate(false);
    density.DepositEndpointAverage(sim, 0, rho, mid);
    probe.PreLinearSolve(0);
    if (mode == "bad_width") {
        probe.CorruptWidth();
        probe.Diagonal();
        amrex::Abort("width rejection missing");
    }
    if (mode == "bad_band") {
        auto& band =
            *sim.m_fields.get(FieldType::MassMatrices_X, Direction{1}, 0);
        band = MF(band.boxArray(), band.DistributionMap(), 1, band.nGrowVect());
        probe.Diagonal();
        amrex::Abort("band rejection missing");
    }
    auto const diagonal =
        sim.m_fields.get_alldirs(FieldType::MassMatrices_PC, 0);
    Owned frozen;
    std::array<std::unique_ptr<MF>, 9> frozen_bands;
    FieldType const types[3] = {FieldType::MassMatrices_X,
                                FieldType::MassMatrices_Y,
                                FieldType::MassMatrices_Z};
    for (int c = 0; c < 3; ++c) {
        frozen[c] = std::make_unique<MF>(Clone(*diagonal[c]));
        for (int d = 0; d < 3; ++d)
            frozen_bands[3 * c + d] = std::make_unique<MF>(
                CloneBand(*sim.m_fields.get(types[c], Direction{d}, 0)));
    }
    auto const electric = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const cells = sim.Geom(0).Domain().length();
    Real worst = 0;
    for (int c = 0; c < 3; ++c) {
        for (auto point :
             {amrex::IntVect(0, 0), amrex::IntVect(0, 5), amrex::IntVect(1, 1),
              amrex::IntVect(7, 15), amrex::IntVect(cells[0] - 2, 1),
              amrex::IntVect(cells[0] - 1, cells[1] - 1),
              amrex::IntVect(cells[0] - (c == 0), cells[1] - (c == 2))}) {
            if (c == 1 && point[0] == 0)
                continue;
            for (int d = 0; d < 3; ++d) {
                electric[d]->setVal(0.);
                if (c != d)
                    continue;
                int const ii = point[0], jj = point[1], nz = cells[1];
                bool const periodic = sim.Geom(0).isPeriodic(1);
                for (amrex::MFIter mfi(*electric[d]); mfi.isValid(); ++mfi) {
                    auto const e = electric[d]->array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            e(i, j, k) =
                                i == ii &&
                                        (periodic ? j % nz == jj % nz : j == jj)
                                    ? 1.
                                    : 0.;
                        });
                }
            }
            FillGather(sim);
            probe.MatrixAction();
            amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce;
            amrex::ReduceData<Real, Real> data(reduce);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(*current[c]); mfi.isValid(); ++mfi) {
                auto const j = current[c]->const_array(mfi),
                           d = diagonal[c]->const_array(mfi);
                int const ii = point[0], jj = point[1];
                reduce.eval(
                    mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int y, int k) -> Tuple {
                        return i == ii && y == jj
                                   ? Tuple{std::abs(j(i, y, k) - d(i, y, k)),
                                           std::abs(j(i, y, k))}
                                   : Tuple{0., 0.};
                    });
            }
            auto value = data.value();
            Real err = amrex::get<0>(value), scale = amrex::get<1>(value);
            amrex::ParallelDescriptor::ReduceRealMax(err);
            amrex::ParallelDescriptor::ReduceRealMax(scale);
            worst = std::max(worst, err / std::max(scale, Real(1.e-20)));
        }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(worst < 1.e-8,
                                     "native unit-basis MM diagonal mismatch");
    // Real full-particle signed probes, even in the originally matrix-enabled
    // arm. Neither their current nor their physical density can be a stale
    // anchor.
    probe.FullParticleProbes();
    FillElectric(sim, 0.);
    probe.Evaluate(true);
    density.DepositEndpointAverage(sim, 0, rho, mid);
    Owned base;
    for (int c = 0; c < 3; ++c)
        base[c] = std::make_unique<MF>(Clone(*current[c]));
    auto base_rho = Clone(rho, mid);
    Real jchange = 0, rhochange = 0;
    for (Real sign : {1., -1., 0.}) {
        FillElectric(sim, sign * 1.e4);
        probe.Evaluate(true);
        density.DepositEndpointAverage(sim, 0, rho, mid);
        for (int c = 0; c < 3; ++c) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                Error(*diagonal[c], *frozen[c], true) == 0.,
                "full particle probe mutated frozen MM-PC");
            for (int d = 0; d < 3; ++d)
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    BandError(*sim.m_fields.get(types[c], Direction{d}, 0),
                              *frozen_bands[3 * c + d]) == 0.,
                    "full particle probe mutated a frozen current-response "
                    "band");
            if (sign != 0)
                jchange = std::max(jchange, Error(*current[c], *base[c]));
            else
                AMREX_ALWAYS_ASSERT(Error(*current[c], *base[c]) <
                                    1.e-10 *
                                        std::max(Real(1), base[c]->norminf()));
        }
        if (sign != 0)
            rhochange = std::max(rhochange, Error(Clone(rho, mid), base_rho));
        else
            AMREX_ALWAYS_ASSERT(Error(Clone(rho, mid), base_rho) <
                                1.e-10 *
                                    std::max(Real(1.e-30), base_rho.norminf()));
    }
    AMREX_ALWAYS_ASSERT(jchange > 0 && rhochange > 0);
    probe.Diagonal();
    for (int c = 0; c < 3; ++c)
        AMREX_ALWAYS_ASSERT(Error(*diagonal[c], *frozen[c], true) == 0.);
    // A new nonlinear anchor must rebuild the response; only signed Jv
    // probes are required to leave the previous coefficients frozen.
    FillElectric(sim, 1.e4);
    probe.Evaluate(false);
    density.DepositEndpointAverage(sim, 0, rho, mid);
    probe.PreLinearSolve(-1);
    Real rebuild_change = 0;
    for (int c = 0; c < 3; ++c)
        rebuild_change =
            std::max(rebuild_change, Error(*diagonal[c], *frozen[c]));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        rebuild_change > 0., "new particle anchor did not refresh MM-PC");
    amrex::Print() << "PASS MM_PC " << mode
                   << " diagonal_relative_error=" << worst
                   << " full_particle_J_change=" << jchange
                   << " endpoint_average_rho_change=" << rhochange
                   << " frozen_coefficient_error=0 frozen_nine_band_error=0"
                   << " rebuilt_diagonal_change=" << rebuild_change
                   << " boxes=" << current[0]->size() << "\n";
}
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        Run(sim);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
