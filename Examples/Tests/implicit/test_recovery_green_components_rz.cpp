/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/DarwinRZGreenSolver.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <iomanip>

namespace {
AMREX_GPU_HOST_DEVICE amrex::Real field (int i, int j, int nr, int nz, amrex::Real amplitude,
                                         bool pmc_lo, bool pmc_hi, int component)
{
    if ((component < 2 && i <= 0) || i >= nr ||
        (j <= 0 && !pmc_lo) || (j >= nz && !pmc_hi))
    {
        return 0.;
    }
    amrex::Real const r = static_cast<amrex::Real>(i) / nr;
    amrex::Real const z = static_cast<amrex::Real>(j) / nz;
    constexpr amrex::Real pi = 3.14159265358979323846;
    amrex::Real const axial =
        pmc_lo && pmc_hi ? 1. + .3 * std::cos(pi * z) + .1 * std::cos(pi * j)
                         : (pmc_lo ? std::cos(.5 * pi * z) : std::sin((pmc_hi ? .5 : 1.) * pi * z));
    return amplitude * (component < 2 ? r : 1.) * (1. - r * r) * axial *
           (1. + .2 * std::cos(8. * pi * r) * std::cos(6. * pi * z));
}
} // namespace

int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        int nr = 32, nz = 160, grid = 32, repeats = 3, max_semi = 2;
        amrex::Real aspect = 5.;
        bool greens = true, pmc_lo = false, pmc_hi = false;
        int component = 2;
        amrex::ParmParse pp("mg_test");
        pp.query("nr", nr);
        pp.query("nz", nz);
        pp.query("grid", grid);
        pp.query("repeats", repeats);
        pp.query("max_semi", max_semi);
        pp.query("aspect", aspect);
        pp.query("greens", greens);
        pp.query("pmc_lo", pmc_lo);
        pp.query("pmc_hi", pmc_hi);
        pp.query("component", component);
        AMREX_ALWAYS_ASSERT(component >= 0 && component < 3);
        bool const field_pmc_lo = pmc_lo, field_pmc_hi = pmc_hi;
        // A_z is normal at the caps, so its correction is Dirichlet even
        // when the physical field end is PMC. Its radial axis remains free.
        if (component == 2) { pmc_lo = false; pmc_hi = false; }
        AMREX_ALWAYS_ASSERT(nr >= 8 && nz >= 8 && repeats > 0);
        amrex::Box domain(amrex::IntVect(0, 0), amrex::IntVect(nr - 1, nz - 1));
        amrex::Real const dr = .2 / nr, dz = aspect * dr;
        amrex::RealBox physical({0., 0.}, {.2, nz * dz});
        int periodic[2] = {0, 0};
        amrex::Geometry geom(domain, &physical, 1, periodic);
        amrex::BoxArray cells(domain);
        cells.maxSize(grid);
        amrex::DistributionMapping dm(cells);
        auto nodes = amrex::convert(cells, amrex::IntVect::TheNodeVector());
        amrex::MultiFab exact(nodes, dm, 1, 1), rhs(nodes, dm, 1, 0);
        amrex::MultiFab axial(nodes, dm, 1, 0), actual(nodes, dm, 1, 0);
        amrex::MultiFab solution(nodes, dm, 1, 1), error(nodes, dm, 1, 0);
        amrex::MultiFab reference(nodes, dm, 1, 0);
        amrex::MultiFab green_solution(nodes, dm, 1, 1);
        std::unique_ptr<DarwinRZGreenSolver> green;
        if (greens)
        {
            green = std::make_unique<DarwinRZGreenSolver>(geom, field_pmc_lo, field_pmc_hi, true);
        }
        for (int masked = 0; masked <= 1; ++masked) {
            // This source uses an independently assembled five-point stencil.
            // The vacuum mask limits the source, not the inverse domain.
            for (amrex::MFIter mfi(exact); mfi.isValid(); ++mfi) {
                auto a = exact.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   { a(i, j, k) = field(i, j, nr, nz, 1., pmc_lo, pmc_hi, component); });
                auto b = rhs.array(mfi), zonly = axial.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    amrex::Real value = 0., without_radial = 0.;
                    if ((i > 0 || (component == 2 && i == 0)) && i < nr &&
                        (j > 0 || pmc_lo) && (j < nz || pmc_hi))
                    {
                        amrex::Real const r = i * dr, u = field(i, j, nr, nz, 1., pmc_lo, pmc_hi, component);
                        amrex::Real const radial = i == 0
                            ? 4. * (field(1, j, nr, nz, 1., pmc_lo, pmc_hi, component) - u) / (dr * dr)
                            : ((r + .5 * dr) * field(i + 1, j, nr, nz, 1., pmc_lo, pmc_hi, component) +
                             (r - .5 * dr) * field(i - 1, j, nr, nz, 1., pmc_lo, pmc_hi, component) -
                             2. * r * u) /
                            (r * dr * dr);
                        without_radial = (field(i, j + 1, nr, nz, 1., pmc_lo, pmc_hi, component) - 2. * u +
                                          field(i, j - 1, nr, nz, 1., pmc_lo, pmc_hi, component)) /
                                             (dz * dz) -
                                         (component < 2 ? u / (r * r) : 0.);
                        value = radial + without_radial;
                    }
                    b(i, j, k) = value;
                    zonly(i, j, k) = without_radial;
                });
            }
            exact.OverrideSync(geom.periodicity());
            exact.FillBoundary(geom.periodicity());
            for (int semi = 0; semi <= max_semi; ++semi) {
                amrex::Gpu::streamSynchronize();
                amrex::Real const setup_start = amrex::second();
                amrex::LPInfo info;
                if (semi > 0) {
                    info.setSemicoarsening(true)
                        .setSemicoarseningDirection(1)
                        .setMaxSemicoarseningLevel(semi);
                }
                amrex::MLEBNodeFDLaplacian linop({geom}, {cells}, {dm}, info);
                linop.setRZ(true);
                // Preparation restores the legacy zero radial sigma to one.
                linop.setSigma({0., 1.});
                linop.setAlpha(component < 2 ? 1. : 0.);
                linop.setDomainBC(
                    {amrex::LinOpBCType::Neumann,
                     pmc_lo ? amrex::LinOpBCType::Neumann : amrex::LinOpBCType::Dirichlet},
                    {amrex::LinOpBCType::Dirichlet,
                     pmc_hi ? amrex::LinOpBCType::Neumann : amrex::LinOpBCType::Dirichlet});
                linop.prepareForSolve();
                linop.apply(0, 0, actual, exact,
                            amrex::MLLinOp::BCMode::Homogeneous,
                            amrex::MLLinOp::StateMode::Solution);
                amrex::MultiFab::LinComb(error, 1., actual, 0, -1., rhs, 0, 0,
                                         1, 0);
                amrex::Real const op_error = error.norminf() / rhs.norminf();
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    op_error < 2.e-13, "prepared radial stencil mismatch");
                amrex::MultiFab::LinComb(error, 1., actual, 0, -1., axial, 0, 0,
                                         1, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    error.norminf() > 0.1 * rhs.norminf(),
                    "radial-coupling negative control failed");
                amrex::MultiFab source(nodes, dm, 1, 0);
                amrex::MultiFab::Copy(source, rhs, 0, 0, 1, 0);
                if (masked) {
                    for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) {
                        auto a = source.array(mfi);
                        amrex::ParallelFor(
                            mfi.validbox(),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                if (i < nr / 2) {
                                    a(i, j, k) = 0.;
                                }
                            });
                    }
                }
                amrex::MLMG mg(linop);
                mg.setVerbose(0);
                mg.setMaxIter(2000);
                mg.setThrowException(true);
                mg.setBottomSolver(amrex::BottomSolver::smoother);
                amrex::Gpu::streamSynchronize();
                amrex::Real const setup = amrex::second() - setup_start;
                solution.setVal(0.);
                for (int rep = 0; rep < repeats; ++rep) {
                    // The same scaled RHS sequence tests cold and warm starts.
                    if (rep > 0) {
                        source.mult(1.001);
                    }
                    amrex::Gpu::streamSynchronize();
                    amrex::Real const start = amrex::second();
                    mg.solve({&solution}, {&source}, 1.e-12, 0.);
                    amrex::Gpu::streamSynchronize();
                    amrex::Real const solve = amrex::second() - start;
                    linop.apply(0, 0, actual, solution,
                                amrex::MLLinOp::BCMode::Homogeneous,
                                amrex::MLLinOp::StateMode::Solution);
                    amrex::MultiFab::LinComb(error, 1., actual, 0, -1., source,
                                             0, 0, 1, 0);
                    amrex::Real const residual =
                        error.norminf() / source.norminf();
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(residual < 1.1e-12,
                                                     "true residual failed");
                    if (green) {
                        amrex::Gpu::streamSynchronize();
                        amrex::Real const green_start = amrex::second();
                        green->SolveComponent(green_solution, source, component);
                        amrex::Gpu::streamSynchronize();
                        amrex::Real const green_time = amrex::second() - green_start;
                        linop.apply(0, 0, actual, green_solution,
                            amrex::MLLinOp::BCMode::Homogeneous,
                            amrex::MLLinOp::StateMode::Solution);
                        amrex::MultiFab::LinComb(error, 1., actual, 0, -1., source,
                                                0, 0, 1, 0);
                        amrex::Real const green_residual = error.norminf() / source.norminf();
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(green_residual < 1.1e-12,
                                                        "Green inverse true residual failed");
                        amrex::MultiFab::LinComb(error, 1., green_solution, 0, -1., solution,
                                                0, 0, 1, 0);
                        amrex::Real const green_error = error.norminf() / solution.norminf();
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(green_error < 1.e-9,
                                                        "Green and multigrid inverses differ");
                        amrex::Print() << std::setprecision(12)
                            << "GREEN_COMPONENT_BENCH component=" << component << " nr=" << nr << " nz=" << nz << " grid=" << grid
                            << " masked=" << masked << " semi=" << semi << " repeat=" << rep
                            << " solve_seconds=" << green_time << " residual=" << green_residual
                            << " relative_map_error=" << green_error << "\n";
                        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                            auto const e = error.array(mfi);
                            auto const u = green_solution.const_array(mfi);
                            amrex::ParallelFor(mfi.validbox(),
                                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                               {
                                                   e(i, j, k) = (i == 0 && component < 2) || i == nr ||
                                                                        (j == 0 && !pmc_lo) ||
                                                                        (j == nz && !pmc_hi)
                                                                    ? u(i, j, k)
                                                                    : 0.;
                                               });
                        }
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norminf() == 0.,
                                                        "Green inverse changes fixed trace");
                    }
                    amrex::Real solution_error = 0.;
                    if (!masked) {
                        amrex::Real const scale = std::pow(1.001, rep);
                        amrex::MultiFab::LinComb(error, 1., solution, 0, -scale,
                                                 exact, 0, 0, 1, 0);
                        solution_error =
                            error.norminf() / (scale * exact.norminf());
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                            solution_error < 1.e-9,
                            "manufactured solution mismatch");
                    }
                    if (masked) {
                        if (semi == 0 && rep == 0) {
                            amrex::MultiFab::Copy(reference, solution, 0, 0, 1,
                                                  0);
                        } else {
                            amrex::Real const scale = std::pow(1.001, rep);
                            amrex::MultiFab::LinComb(error, 1., solution, 0,
                                                     -scale, reference, 0, 0, 1,
                                                     0);
                            solution_error =
                                error.norminf() / (scale * reference.norminf());
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                                solution_error < 1.e-9,
                                "masked inverse differs");
                        }
                    }
                    for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                        auto const e = error.array(mfi);
                        auto const u = solution.const_array(mfi);
                        amrex::ParallelFor(mfi.validbox(),
                                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                           {
                                               e(i, j, k) =
                                                   ((i == 0 && component < 2) || i == nr || (j == 0 && !pmc_lo) ||
                                                    (j == nz && !pmc_hi))
                                                       ? u(i, j, k)
                                                       : 0.;
                                           });
                    }
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norminf() == 0.,
                                                     "fixed boundary changed");
                    amrex::Print()
                        << std::setprecision(12) << "MG_COMPONENT_BENCH component=" << component << " nr=" << nr
                        << " nz=" << nz << " grid=" << grid
                        << " aspect=" << aspect << " masked=" << masked
                        << " semi=" << semi << " repeat=" << rep
                        << " levels=" << linop.NMGLevels(0)
                        << " iterations=" << mg.getNumIters()
                        << " setup_seconds=" << setup
                        << " solve_seconds=" << solve
                        << " op_error=" << op_error << " residual=" << residual
                        << " solution_error=" << solution_error << "\n";
                }
            }
        }
        if (green) {
            rhs.setVal(0.);
            green->SolveComponent(green_solution, rhs, component);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(green_solution.norminf() == 0.,
                                             "Green inverse reuses a stale source");
        }
    }
    amrex::Finalize();
}
