/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/DarwinVacuumERecovery.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>

int main (int argc, char **argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto &simulation = WarpX::GetInstance();
        simulation.InitData();
        auto &hybrid = *simulation.get_pointer_HybridPICModel();
        auto const &geom = simulation.Geom(0);
        constexpr int axial = AMREX_SPACEDIM - 1;
        amrex::Real drive_slope = 0.;
        amrex::ParmParse("test").query("pmc_drive_slope", drive_slope);
        amrex::Real const dr = geom.CellSize(0);
        amrex::Real const dz = geom.CellSize(axial);
        amrex::Real const zlo = geom.ProbLo(axial);
        amrex::ignore_unused(dr, dz, zlo);
        AMREX_ALWAYS_ASSERT(WarpX::field_boundary_lo[axial] == FieldBoundaryType::PMC &&
                            WarpX::field_boundary_hi[axial] == FieldBoundaryType::PMC);
        auto e = simulation.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0);
        auto el = simulation.m_fields.get_alldirs("hybrid_E_long_fp", 0);
        auto &rho = *simulation.m_fields.get("hybrid_rho_vacmask_fp", 0);
        amrex::Array<amrex::MultiFab, 3> accepted, endpoint;
        for (int d = 0; d < 3; ++d)
        {
            accepted[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, e[d]->nGrowVect());
            endpoint[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, e[d]->nGrowVect());
            accepted[d].setVal(0.);
            endpoint[d].setVal(0.);
            el[d]->setVal(0.);
        }
        rho.setVal(0.);
        for (amrex::MFIter mfi(accepted[1]); mfi.isValid(); ++mfi)
        {
            auto a = accepted[1].array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               { a(i, j, k) = 1.e4 * std::sin(.37 * (i + 3 * j)); });
        }
        // In an empty cylinder with uniform axial drive, the unique flux
        // recovery is E_theta=-r/2 * dB_external/dt, also at the free PMC ends.
        RecoverDarwinVacuumE(simulation, hybrid, {&endpoint[0], &endpoint[1], &endpoint[2]},
                             {&accepted[0], &accepted[1], &accepted[2]}, 1.e-9, 1.e-9);
        amrex::Real error = 0.;
#if defined(WARPX_DIM_RZ)
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<amrex::Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(endpoint[1]); mfi.isValid(); ++mfi)
        {
            auto a = endpoint[1].const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                    {
                        return {std::abs(a(i, j, k) +
                            250000. * i * dr * (1. + drive_slope * (zlo + j * dz)))};
                    });
        }
        error = amrex::get<0>(data.value(op));
        amrex::ParallelDescriptor::ReduceRealMax(error);
        amrex::Print() << "PMC_VACUUM_FLUX error=" << error << "\n";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error < 1.e-5, "PMC vacuum flux fails analytic drive");
#endif

        // With no vacuum rows, tangential plasma nodes at both PMC ends
        // must remain genuine endpoint unknowns. A Dirichlet end pin fails.
        rho.setVal(1.e8);
        for (int d = 0; d < 3; ++d)
        {
            accepted[d].setVal(31. + d);
            endpoint[d].setVal(17. + d);
        }
        RecoverDarwinVacuumE(simulation, hybrid, {&endpoint[0], &endpoint[1], &endpoint[2]},
                             {&accepted[0], &accepted[1], &accepted[2]}, 1.e-9, 1.e-9);
        int const nz = geom.Domain().length(axial);
        for (int d = 0; d < 3; ++d)
        {
            amrex::ReduceOps<amrex::ReduceOpMax> check;
            amrex::ReduceData<amrex::Real> result(check);
            using Result = decltype(result)::Type;
            bool const znode = e[d]->ixType().nodeCentered(axial);
            for (amrex::MFIter mfi(*e[d]); mfi.isValid(); ++mfi)
            {
                auto a = e[d]->const_array(mfi);
                auto box = mfi.fabbox();
                for (int c = 0; c < axial; ++c)
                {
                    box.setSmall(c, amrex::max(1, box.smallEnd(c)));
                    box.setBig(c, amrex::min(geom.Domain().length(c) - 1, box.bigEnd(c)));
                }
                check.eval(box, result,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Result
                           {
                               int const index = axial == 1 ? j : k;
                               amrex::Real const sign =
                                   !znode && (index < 0 || index >= nz) ? -1. : 1.;
                               amrex::Real reference_difference = 0.;
#if defined(WARPX_DIM_RZ)
                               if (d == 1)
                               {
                                   int const mirror = index < 0 ? -index
                                       : (index > nz ? 2 * nz - index : index);
                                   reference_difference =
                                       -250000. * i * dr * drive_slope * dz * (index - mirror);
                               }
#endif
                               return {std::abs(a(i, j, k) - sign * (17. + d) -
                                               reference_difference)};
                           });
            }
            error = amrex::get<0>(result.value(check));
            amrex::ParallelDescriptor::ReduceRealMax(error);
            amrex::Print() << "PMC_DENSE_ENDPOINT dir=" << d << " error=" << error << "\n";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error < 1.e-10, "PMC endpoint pin/parity is wrong");
        }
        amrex::Print() << "PMC_NATIVE_RECOVERY_PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
