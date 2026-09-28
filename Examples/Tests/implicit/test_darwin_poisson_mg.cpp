/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/DarwinPoissonMG.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <iomanip>

namespace
{
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
potential (int i, int j, int k, amrex::GpuArray<int, 3> const& n)
{
    constexpr amrex::Real pi = 3.1415926535897932384626433832795;
    amrex::Real const x = amrex::Real(i) / n[0];
    amrex::Real value = (1. - x * x) * (1. - x * x) * (.7 + .3 * std::cos(2. * pi * j / n[1]));
#if AMREX_SPACEDIM == 3
    value *= .8 + .2 * std::cos(4. * pi * k / n[2]);
#else
    amrex::ignore_unused(k);
#endif
    return value;
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("poisson_test");
        amrex::Real aspect = 5.;
        int grid = 64, expected_levels = 2;
        pp.query("aspect", aspect);
        pp.query("grid", grid);
        pp.query("expected_levels", expected_levels);
        amrex::GpuArray<int, 3> const n{64, AMREX_D_PICK(1, 128, 64), 128};
        amrex::IntVect const hi(AMREX_D_DECL(n[0] - 1, n[1] - 1, n[2] - 1));
        amrex::Box const domain(amrex::IntVect(0), hi);
        amrex::RealBox physical(AMREX_D_DECL(0., 0., 0.),
                                AMREX_D_DECL(1., AMREX_D_PICK(1., 2. * aspect, 1.), 2. * aspect));
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
#ifdef DARWIN_POISSON_TEST_RZ
        constexpr int coord = 1;
#else
        constexpr int coord = 0;
#endif
        amrex::Geometry geom(domain, &physical, coord, periodic.data());
        auto const dx = geom.CellSizeArray();
        amrex::BoxArray cells(domain);
        cells.maxSize(grid);
        amrex::DistributionMapping dm(cells);
        amrex::BoxArray const nodes = amrex::convert(cells, amrex::IntVect(1));
        amrex::MultiFab exact(nodes, dm, 1, 1), rhs(nodes, dm, 1, 0);
        amrex::MultiFab solution(nodes, dm, 1, 1), action(nodes, dm, 1, 0);
        amrex::MultiFab error(nodes, dm, 1, 0), reference(nodes, dm, 1, 0);
        for (amrex::MFIter mfi(exact); mfi.isValid(); ++mfi)
        {
            auto a = exact.array(mfi), b = rhs.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { a(i, j, k) = potential(i, j, k, n); });
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real v = 0.;
                    if (i != n[0])
                    {
                        amrex::Real const u = potential(i, j, k, n);
#ifdef DARWIN_POISSON_TEST_RZ
                        v = i == 0 ? 4. * (potential(1, j, k, n) - u) / (dx[0] * dx[0])
                                   : ((1. + .5 / i) * potential(i + 1, j, k, n) +
                                      (1. - .5 / i) * potential(i - 1, j, k, n) - 2. * u) /
                                         (dx[0] * dx[0]);
#else
                        v = (potential(i + 1, j, k, n) + potential(i - 1, j, k, n) - 2. * u) /
                            (dx[0] * dx[0]);
#endif
                        v += (potential(i, j + 1, k, n) + potential(i, j - 1, k, n) - 2. * u) /
                             (dx[1] * dx[1]);
#if AMREX_SPACEDIM == 3
                        v += (potential(i, j, k + 1, n) + potential(i, j, k - 1, n) - 2. * u) /
                             (dx[2] * dx[2]);
#endif
                    }
                    b(i, j, k) = v;
                });
        }
        exact.OverrideSync(geom.periodicity());
        rhs.OverrideSync(geom.periodicity());
        exact.FillBoundary(geom.periodicity());
        for (int const cap : {0, 2})
        {
            amrex::LPInfo info;
            int const levels = ConfigureDarwinPoissonSemicoarsening(info, geom, cap);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(levels == (cap == 0 ? 0 : expected_levels),
                                             "Unexpected anisotropy-dependent hierarchy");
            amrex::MLEBNodeFDLaplacian op({geom}, {cells}, {dm}, info);
#ifdef DARWIN_POISSON_TEST_RZ
            op.setRZ(true);
            op.setSigma({0., 1.});
#else
            op.setSigma({AMREX_D_DECL(1., 1., 1.)});
#endif
            amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lo, upper;
            lo.fill(amrex::LinOpBCType::Neumann);
            upper.fill(amrex::LinOpBCType::Neumann);
            upper[0] = amrex::LinOpBCType::Dirichlet;
            op.setDomainBC(lo, upper);
            op.prepareForSolve();
            op.apply(0, 0, action, exact, amrex::MLLinOp::BCMode::Homogeneous,
                     amrex::MLLinOp::StateMode::Solution);
            amrex::MultiFab::LinComb(error, 1., action, 0, -1., rhs, 0, 0, 1, 0);
            amrex::Real const stencil_error = error.norminf() / rhs.norminf();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                stencil_error < 5.e-12,
                "E_L finest operator disagrees with independent cylindrical/Cartesian stencil");
            solution.setVal(0.);
            amrex::MLMG mg(op);
            mg.setVerbose(0);
            mg.setMaxIter(2000);
            mg.setThrowException(true);
            amrex::Gpu::streamSynchronize();
            amrex::Real const start = amrex::second();
            mg.solve({&solution}, {&rhs}, 1.e-12, 0.);
            amrex::Gpu::streamSynchronize();
            amrex::Real const seconds = amrex::second() - start;
            op.apply(0, 0, action, solution, amrex::MLLinOp::BCMode::Homogeneous,
                     amrex::MLLinOp::StateMode::Solution);
            amrex::MultiFab::LinComb(error, 1., action, 0, -1., rhs, 0, 0, 1, 0);
            amrex::Real const residual = error.norminf() / rhs.norminf();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(residual < 1.1e-12, "E_L true residual gate failed");
            amrex::MultiFab::LinComb(error, 1., solution, 0, -1., exact, 0, 0, 1, 0);
            amrex::Real const solution_error = error.norminf() / exact.norminf();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(solution_error < 1.e-9,
                                             "E_L manufactured solution mismatch");
            amrex::Real difference = 0.;
            if (cap == 0)
            {
                amrex::MultiFab::Copy(reference, solution, 0, 0, 1, 0);
            }
            else
            {
                amrex::MultiFab::LinComb(error, 1., solution, 0, -1., reference, 0, 0, 1, 0);
                difference = error.norminf() / reference.norminf();
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(difference < 1.e-9,
                                                 "Semicoarsening changes E_L solution");
            }
            amrex::Print() << std::setprecision(12) << "EL_MG_TEST aspect=" << aspect
                           << " cap=" << cap << " levels=" << levels
                           << " retained_direction=" << info.semicoarsening_direction
                           << " iterations=" << mg.getNumIters() << " seconds=" << seconds
                           << " stencil_error=" << stencil_error << " residual=" << residual
                           << " solution_error=" << solution_error
                           << " relative_difference=" << difference << "\n";
        }
    }
    amrex::Finalize();
}
