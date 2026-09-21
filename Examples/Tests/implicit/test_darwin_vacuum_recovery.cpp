/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/DarwinVacuumERecovery.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <cmath>

// A nodal plateau encloses the plasma and conductor. Its edge gradient has
// support only in vacuum: its pairing with E measures enclosed electric flux.
AMREX_GPU_HOST_DEVICE amrex::Real
potential (int i, int j, int, int mode)
{
    if (mode == 0)
    {
        return (i >= 4 && i <= 12 && j >= 4 && j <= 12) ? 1. : 0.;
    }
    return (i == 3 && j == 3) ? 1. : 0.;
}

int
main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        auto& hybrid = *simulation.get_pointer_HybridPICModel();
        auto const& geom = simulation.Geom(0);
        auto const periodic = geom.periodicity();
        auto const dx = geom.CellSizeArray();
        auto const fields = simulation.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0);
        auto const longitudinal = simulation.m_fields.get_alldirs("hybrid_E_long_fp", 0);
        auto& rho = *simulation.m_fields.get("hybrid_rho_vacmask_fp", 0);
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi)
        {
            auto const density = rho.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { density(i, j, k) = (i >= 6 && i <= 10 && j >= 6 && j <= 10) ? 1.e8 : 0.; });
        }
        amrex::Array<amrex::MultiFab, 3> accepted, endpoint, difference, nullmode;
        for (int d = 0; d < 3; ++d)
        {
            for (auto* array : {&accepted, &endpoint, &difference, &nullmode})
            {
                (*array)[d].define(fields[d]->boxArray(), fields[d]->DistributionMap(), 1,
                                   fields[d]->nGrowVect());
            }
            longitudinal[d]->setVal(.37 * (d + 1));
            for (amrex::MFIter mfi(accepted[d]); mfi.isValid(); ++mfi)
            {
                auto const a = accepted[d].array(mfi), e = endpoint[d].array(mfi);
                amrex::ParallelFor(mfi.fabbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                   {
                                       a(i, j, k) =
                                           1.e4 * std::sin(.37 * (i + 3 * j + 7 * k + 11 * d));
                                       e(i, j, k) =
                                           2.e3 * std::cos(.23 * (2 * i + 5 * j + 3 * k + 7 * d));
                                   });
            }
            accepted[d].OverrideSync(periodic);
            endpoint[d].OverrideSync(periodic);
        }
        RecoverDarwinVacuumE(simulation, hybrid, {&endpoint[0], &endpoint[1], &endpoint[2]},
                             {&accepted[0], &accepted[1], &accepted[2]}, 1.e-9, 1.e-9);
        for (int d = 0; d < 3; ++d)
        {
            amrex::MultiFab::LinComb(difference[d], 1., endpoint[d], 0, -1., accepted[d], 0, 0, 1,
                                     0);
            endpoint[d].FillBoundary(periodic);
        }
        // Two independent null tests: a component flux and a local divergence.
        // A converged arbitrary-PC solve can violate these even if curl curl
        // E=0.
        for (int mode = 0; mode < 2; ++mode)
        {
            amrex::Real charge_change = 0., reference = 0.;
            for (int d = 0; d < 3; ++d)
            {
                for (amrex::MFIter mfi(nullmode[d]); mfi.isValid(); ++mfi)
                {
                    auto const n = nullmode[d].array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                                       {
                                           int q[3]{i, j, k};
                                           ++q[d];
                                           n(i, j, k) = (potential(q[0], q[1], q[2], mode) -
                                                         potential(i, j, k, mode)) /
                                                        dx[d];
                                       });
                }
                auto const owner = fields[d]->OwnerMask(periodic);
                charge_change +=
                    amrex::MultiFab::Dot(*owner, nullmode[d], 0, difference[d], 0, 1, 0, true);
                reference +=
                    amrex::MultiFab::Dot(*owner, nullmode[d], 0, accepted[d], 0, 1, 0, true);
            }
            amrex::ParallelDescriptor::ReduceRealSum(charge_change);
            amrex::ParallelDescriptor::ReduceRealSum(reference);
            amrex::Print() << "RECOVERY_CHARGE mode=" << mode << " change=" << charge_change
                           << " reference=" << reference << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(reference) > 1.);
            AMREX_ALWAYS_ASSERT(std::abs(charge_change) < 1.e-8 * std::abs(reference));
        }
        if (geom.isAllPeriodic())
        {
            for (int d = 0; d < 3; ++d)
            {
                nullmode[d].setVal(1.);
                auto const owner = fields[d]->OwnerMask(periodic);
                amrex::Real change =
                    amrex::MultiFab::Dot(*owner, nullmode[d], 0, difference[d], 0, 1, 0, false);
                amrex::Print() << "RECOVERY_PERIODIC_MEAN dir=" << d << " change=" << change
                               << "\n";
                AMREX_ALWAYS_ASSERT(std::abs(change) < 1.e-6);
            }
        }
        // Guard values must use the clamped endpoint even at intersections of
        // physical walls and transverse box boundaries. E_L is added once.
        for (int d = 0; d < 3; ++d)
        {
            auto const domain = amrex::convert(geom.Domain(), fields[d]->ixType());
            auto const lo = domain.smallEnd(), hi = domain.bigEnd();
            amrex::GpuArray<int, 3> per{geom.isPeriodic(0), geom.isPeriodic(1), geom.isPeriodic(2)};
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<amrex::Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(*fields[d]); mfi.isValid(); ++mfi)
            {
                auto const e = fields[d]->const_array(mfi), t = endpoint[d].const_array(mfi);
                op.eval(mfi.fabbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                        {
                            int q[3]{i, j, k};
                            for (int c = 0; c < 3; ++c)
                            {
                                if (!per[c])
                                {
                                    q[c] = amrex::Clamp(q[c], lo[c], hi[c]);
                                }
                            }
                            return {std::abs(e(i, j, k) - t(q[0], q[1], q[2]) - .37 * (d + 1))};
                        });
            }
            amrex::Real error = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealMax(error);
            amrex::Print() << "RECOVERY_REGISTER dir=" << d << " error=" << error << "\n";
            AMREX_ALWAYS_ASSERT(error < 1.e-9);
        }
        amrex::Print() << "SPATIAL_RECOVERY_TEST_PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
