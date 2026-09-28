/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/DarwinABoundary.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <cmath>

// Independent analytic continuation checks every ghost layer and corner,
// and includes both Yee E and B staggers. No component uses an all-even
// scalar Neumann image: normal E / tangential B must be odd at PMC.
void check_pmc (amrex::Geometry const &geom, amrex::BoxArray const &cells,
                amrex::DistributionMapping const &dm)
{
    AMREX_ALWAYS_ASSERT(FieldBoundaryType::Neumann == FieldBoundaryType::PMC);
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc{};
    for (auto &value : pmc)
    {
        value = 1;
    }
    for (int electric = 0; electric <= 1; ++electric)
    {
        int const count = AMREX_SPACEDIM == 2 && electric ? 3 : AMREX_SPACEDIM;
        for (int component = 0; component < count; ++component)
        {
            auto stag = amrex::IntVect(electric ? 1 : 0);
            if (component < AMREX_SPACEDIM)
            {
                stag[component] = electric ? 0 : 1;
            }
            auto ba = amrex::convert(cells, stag);
            amrex::MultiFab value(ba, dm, 1, 4), drive(ba, dm, 1, 4), exact(ba, dm, 1, 4),
                error(ba, dm, 1, 4);
            amrex::GpuArray<int, AMREX_SPACEDIM> node{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                node[d] = stag[d];
            }
            for (amrex::MFIter mfi(value); mfi.isValid(); ++mfi)
            {
                auto a = value.array(mfi), ref = drive.array(mfi), ex = exact.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       int const p[3] = {i, j, k};
                                       amrex::Real response = 1., imposed = .17;
                                       bool outside = false;
                                       for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                       {
                                           amrex::Real const x = (p[d] + (node[d] ? 0. : .5)) / 32.;
                                           response *= node[d] ? std::cos(6.283185307179586 * x)
                                                               : std::sin(6.283185307179586 * x);
                                           imposed += .03 * x;
                                           outside = outside || p[d] < 0 || p[d] > 31 + node[d];
                                       }
                                       ref(i, j, k) = imposed;
                                       ex(i, j, k) = imposed + response;
                                       a(i, j, k) = outside ? -9876. : ex(i, j, k);
                                   });
            }
            value.FillBoundary(geom.periodicity());
            ApplyDarwinPMCVectorBoundary(value, geom, pmc, pmc, &drive);
            amrex::MultiFab::LinComb(error, 1., value, 0, -1., exact, 0, 0, 1, 4);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                error.norminf(0, 4) < 3.e-14,
                "PMC vector parity or imposed-response corner is incorrect");
            if (electric)
            {
                // The A-specific path must agree with the same analytic
                // response, including its tangential endpoint degrees of
                // freedom.
                ApplyDarwinCellCenteredABoundary(value, drive, geom, false, nullptr, pmc, pmc);
                amrex::MultiFab::LinComb(error, 1., value, 0, -1., exact, 0, 0, 1, 4);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    error.norminf(0, 4) < 3.e-14,
                    "PMC A boundary pinned a free node or lost vector parity");
            }
        }
    }
    amrex::Print() << "PMC_BOUNDARY_PASS even_tangential_E odd_normal_E "
                      "analytic_drive=PASS\n";
}

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        int grid = 32;
        amrex::ParmParse("boundary_test").query("grid", grid);
        amrex::Box domain(amrex::IntVect(0), amrex::IntVect(31));
        amrex::RealBox physical(AMREX_D_DECL(0., 0., 0.), AMREX_D_DECL(1., 1., 1.));
        amrex::Array<int, AMREX_SPACEDIM> periodic{};
        amrex::Geometry geom(domain, &physical, 0, periodic.data());
        amrex::BoxArray cells(domain);
        cells.maxSize(grid);
        amrex::DistributionMapping dm(cells);
        check_pmc(geom, cells, dm);
        for (int normal = 0; normal < AMREX_SPACEDIM; ++normal)
        {
            auto stag = amrex::IntVect(1);
            stag[normal] = 0;
            amrex::BoxArray edges = amrex::convert(cells, stag);
            amrex::MultiFab a(edges, dm, 1, 4), bc(edges, dm, 1, 4);
            amrex::MultiFab expected(edges, dm, 1, 4), error(edges, dm, 1, 4);
            for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi)
            {
                auto av = a.array(mfi), bv = bc.array(mfi), ev = expected.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       int const p[3] = {i, j, k};
                                       amrex::Real shape = .3 + std::pow((p[normal] + .5) / 32., 2);
                                       amrex::Real imposed = .17;
                                       bool node_wall = false;
                                       for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                       {
                                           imposed += .01 * (p[d] + (d == normal ? .5 : 0.));
                                           if (d != normal)
                                           {
                                               shape *=
                                                   std::sin(3.14159265358979323846 * p[d] / 32.);
                                               node_wall = node_wall || p[d] <= 0 || p[d] >= 32;
                                           }
                                       }
                                       bv(i, j, k) = imposed;
                                       // Nonzero boundary-adjacent A change: pinning the final
                                       // valid half-cell would violate this known update.
                                       av(i, j, k) = imposed + (node_wall ? 0. : .071 * shape);
                                       ev(i, j, k) = av(i, j, k);
                                   });
            }
            a.FillBoundary(geom.periodicity());
            ApplyDarwinCellCenteredABoundary(a, bc, geom, false);
            amrex::MultiFab::LinComb(error, 1., a, 0, -1., expected, 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                error.norminf() == 0.,
                "Prescribed-A wall changed a valid half-cell (or nodal boundary)");
            // Check a nonzero prescribed trace, at the true wall between
            // centers, and the odd continuation for every physical ghost.
            for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi)
            {
                auto const av = a.const_array(mfi), bv = bc.const_array(mfi);
                auto const ev = error.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       int const p[3] = {i, j, k};
                                       int mirror[3] = {i, j, k};
                                       ev(i, j, k) = 0.;
                                       for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                       {
                                           if (d != normal && (p[d] <= 0 || p[d] >= 32))
                                           {
                                               return;
                                           }
                                       }
                                       if (p[normal] < 0)
                                       {
                                           mirror[normal] = -1 - p[normal];
                                       }
                                       else if (p[normal] >= 32)
                                       {
                                           mirror[normal] = 63 - p[normal];
                                       }
                                       else
                                       {
                                           return;
                                       }
                                       ev(i, j, k) = (av(i, j, k) - bv(i, j, k)) +
                                                     (av(mirror[0], mirror[1], mirror[2]) -
                                                      bv(mirror[0], mirror[1], mirror[2]));
                                   });
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                error.norminf(0, 4) < 2.e-15,
                "Prescribed-A trace is not centered on the physical wall");
            // A conductor crossing the physical face keeps its pre-existing
            // zero pin; the normal mirror must not replace covered guards.
            amrex::iMultiFab eb(edges, dm, 1, 4);
            eb.setVal(1);
            for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi)
            {
                auto av = a.array(mfi);
                auto flag = eb.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       int const p[3] = {i, j, k};
                                       if (p[normal] >= 32)
                                       {
                                           flag(i, j, k) = 0;
                                           av(i, j, k) = 0.;
                                       }
                                   });
            }
            ApplyDarwinCellCenteredABoundary(a, bc, geom, false, &eb);
            for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi)
            {
                auto const av = a.const_array(mfi);
                auto const flag = eb.const_array(mfi);
                auto const ev = error.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   { ev(i, j, k) = flag(i, j, k) == 0 ? av(i, j, k) : 0.; });
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error.norminf(0, 4) == 0.,
                                             "Normal-A mirror replaced a covered conductor pin");
            amrex::Print() << "A_BOUNDARY_TEST normal=" << normal << " grid=" << grid
                           << " valid_change=0 wall_trace=PASS conductor_pin=PASS\n";
        }
    }
    amrex::Finalize();
}
