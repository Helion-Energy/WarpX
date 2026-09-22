/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/DarwinVacuumERecovery.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_Reduce.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE amrex::Real
potential (int i, int j, int mode)
{
    return mode == 0 ? (i <= 12 ? 1. : 0.) : (i == 0 && j == 5 ? 1. : 0.);
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
            auto r = rho.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { r(i, j, k) = i >= 6 && i <= 10 ? 1.e8 : 0.; });
        }
        amrex::Array<amrex::MultiFab, 3> accepted, endpoint, difference;
        for (int d = 0; d < 3; ++d)
        {
            for (auto* a : {&accepted, &endpoint, &difference})
            {
                (*a)[d].define(fields[d]->boxArray(), fields[d]->DistributionMap(), 1,
                               fields[d]->nGrowVect());
            }
            bool const radial_node = fields[d]->ixType().nodeCentered(0);
            for (amrex::MFIter mfi(accepted[d]); mfi.isValid(); ++mfi)
            {
                auto a = accepted[d].array(mfi), e = endpoint[d].array(mfi),
                     l = longitudinal[d]->array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        a(i, j, k) = 1.e4 * std::sin(.37 * (i + 3 * j + 11 * d));
                        e(i, j, k) = 2.e3 * std::cos(.23 * (2 * i + 5 * j + 7 * d));
                        l(i, j, k) = d == 2 ? .73 : .37 * (i + (radial_node ? 0. : .5)) * dx[0];
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
        for (int mode = 0; mode < 2; ++mode)
        {
            amrex::Real change = 0., reference = 0.;
            for (int d : {0, 2})
            {
                auto const owner = fields[d]->OwnerMask(periodic);
                amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> op;
                amrex::ReduceData<amrex::Real, amrex::Real> data(op);
                using Tuple = decltype(data)::Type;
                for (amrex::MFIter mfi(difference[d]); mfi.isValid(); ++mfi)
                {
                    auto a = difference[d].const_array(mfi), old = accepted[d].const_array(mfi);
                    auto o = owner->const_array(mfi);
                    op.eval(mfi.validbox(), data,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                            {
                                amrex::Real const weight = d == 0 ? i + .5 : (i == 0 ? .125 : i);
                                amrex::Real const grad =
                                    (potential(i + (d == 0), j + (d == 2), mode) -
                                     potential(i, j, mode)) /
                                    dx[d == 0 ? 0 : 1];
                                return {o(i, j, k) ? weight * grad * a(i, j, k) : 0.,
                                        o(i, j, k) ? weight * grad * old(i, j, k) : 0.};
                            });
                }
                auto value = data.value(op);
                change += amrex::get<0>(value);
                reference += amrex::get<1>(value);
            }
            amrex::ParallelDescriptor::ReduceRealSum(change);
            amrex::ParallelDescriptor::ReduceRealSum(reference);
            amrex::Print() << "RZ_CHARGE mode=" << mode << " change=" << change
                           << " reference=" << reference << "\n";
            AMREX_ALWAYS_ASSERT(std::abs(reference) > 1. &&
                                std::abs(change) < 1.e-8 * std::abs(reference));
        }
        for (int d = 0; d < 3; ++d)
        {
            auto domain = amrex::convert(geom.Domain(), fields[d]->ixType());
            int const hi = domain.bigEnd(0);
            bool const node = fields[d]->ixType().nodeCentered(0);
            amrex::Real const sign = d == 2 ? 1. : -1.;
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<amrex::Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(*fields[d]); mfi.isValid(); ++mfi)
            {
                auto e = fields[d]->const_array(mfi), t = endpoint[d].const_array(mfi),
                     l = longitudinal[d]->const_array(mfi);
                op.eval(mfi.fabbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                        {
                            int const ii = i < 0 ? -i - (node ? 0 : 1) : amrex::min(i, hi);
                            amrex::Real const expected =
                                (i < 0 ? sign : 1.) * t(ii, j, k) + l(i, j, k);
                            return {std::abs(e(i, j, k) - expected)};
                        });
            }
            amrex::Real error = amrex::get<0>(data.value(op));
            amrex::ParallelDescriptor::ReduceRealMax(error);
            amrex::Print() << "RZ_REGISTER dir=" << d << " error=" << error << "\n";
            AMREX_ALWAYS_ASSERT(error < 1.e-9);
        }
        amrex::Print() << "RZ_SPATIAL_RECOVERY_PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
