/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Test the applied native Ohm operator, with no reconstructed force formula.
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <iomanip>

using namespace amrex::literals;
using warpx::fields::FieldType;

int main (int argc, char **argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto &w = WarpX::GetInstance();
        w.InitData();
        auto &hp = *w.get_pointer_HybridPICModel();
        auto const &geom = w.Geom(0);
        auto const dx = geom.CellSizeArray(), lo = geom.ProbLoArray();
        auto const hi = geom.ProbHiArray();
        auto &rho = *w.m_fields.get(FieldType::rho_fp, 0);
        auto *ped = const_cast<amrex::MultiFab *>(hp.DensityPedestal(0));
        auto J = w.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto const Ji = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto const B = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        amrex::MultiFab pressure(rho.boxArray(), rho.DistributionMap(), 1, rho.nGrowVect());
        pressure.setVal(0.0_rt);
        hp.m_include_electron_pressure_term = false;
        hp.m_add_external_fields = false;
        bool hall = true;
        amrex::ParmParse("motional_test").query("hall", hall);
        hp.m_include_hall_term = hall;
        amrex::Real largest_legacy = 0.0_rt;
        amrex::Print() << std::setprecision(17);
        for (amrex::Real phase : {0.0_rt, 0.25_rt, 0.5_rt, 0.75_rt})
        {
            for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi)
            {
                auto const out = rho.array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), rho.nComp(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        auto const x = (i * dx[0]) / (hi[0] - lo[0]);
                        auto const y = (j * dx[1]) / (hi[1] - lo[1]);
                        auto const z = (k * dx[2]) / (hi[2] - lo[2]);
                        out(i, j, k, n) =
                            PhysConst::q_e *
                            (1.e16_rt +
                             (std::cos(2.0_rt * MathConst::pi * (x + y + phase * 0.13_rt)) > 0.1_rt
                                  ? 3.e19_rt *
                                        (1.0_rt + 0.2_rt * std::cos(2.0_rt * MathConst::pi * z))
                                  : 4.8e17_rt *
                                        (1.0_rt + 0.03_rt * std::cos(2.0_rt * MathConst::pi * z))));
                    });
            }
            if (ped)
            {
                ped->setVal(5.e17_rt * PhysConst::q_e);
            }
            for (int c = 0; c < 3; ++c)
            {
                auto const it = Ji[c]->ixType().toIntVect(), bt = B[c]->ixType().toIntVect();
                for (amrex::MFIter mfi(*Ji[c]); mfi.isValid(); ++mfi)
                {
                    auto const ji = Ji[c]->array(mfi), total = J[c]->array(mfi);
                    amrex::ParallelFor(
                        mfi.fabbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            auto const x =
                                (i + (it[0] ? 0.0_rt : 0.5_rt)) * dx[0] / (hi[0] - lo[0]);
                            auto const y =
                                (j + (it[1] ? 0.0_rt : 0.5_rt)) * dx[1] / (hi[1] - lo[1]);
                            auto const z =
                                (k + (it[2] ? 0.0_rt : 0.5_rt)) * dx[2] / (hi[2] - lo[2]);
                            auto const q =
                                1.e5_rt * (0.4_rt +
                                           std::sin(2.0_rt * MathConst::pi *
                                                    (x + (c + 1) * y + z + 0.17_rt * c)) +
                                           0.3_rt * std::cos(2.0_rt * MathConst::pi *
                                                             ((c + 1) * x - y + z + phase)));
                            total(i, j, k) = 0.3_rt * q;
                            ji(i, j, k) = 1.3_rt * q;
                        });
                }
                for (amrex::MFIter mfi(*B[c]); mfi.isValid(); ++mfi)
                {
                    auto const b = B[c]->array(mfi);
                    amrex::ParallelFor(
                        mfi.fabbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            auto const x =
                                (i + (bt[0] ? 0.0_rt : 0.5_rt)) * dx[0] / (hi[0] - lo[0]);
                            auto const y =
                                (j + (bt[1] ? 0.0_rt : 0.5_rt)) * dx[1] / (hi[1] - lo[1]);
                            auto const z =
                                (k + (bt[2] ? 0.0_rt : 0.5_rt)) * dx[2] / (hi[2] - lo[2]);
                            b(i, j, k) =
                                0.3_rt *
                                (1.0_rt + 0.2_rt * std::sin(2.0_rt * MathConst::pi *
                                                            (x + y * (c + 1) + z + 0.11_rt * c)));
                        });
                }
            }
            rho.FillBoundary(geom.periodicity());
            for (int c = 0; c < 3; ++c)
            {
                Ji[c]->FillBoundary(geom.periodicity());
                J[c]->FillBoundary(geom.periodicity());
                B[c]->FillBoundary(geom.periodicity());
            }
            for (bool compatible : {false, true})
            {
                hp.m_energy_conserving_motional = compatible;
                for (int c = 0; c < 3; ++c)
                {
                    E[c]->setVal(0.0_rt);
                }
                w.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(
                    E, J, Ji, B, rho, pressure, w.GetEBUpdateEFlag()[0], 0, &hp, false, false);
                amrex::Real work = 0.0_rt, scale = 0.0_rt;
                for (int c = 0; c < 3; ++c)
                {
                    amrex::MultiFab measured(E[c]->boxArray(), E[c]->DistributionMap(), 2, 0);
                    auto const vol = MakeQdsmcVolumeElement(geom, E[c]->ixType());
                    for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi)
                    {
                        auto const out = measured.array(mfi);
                        auto const e = E[c]->const_array(mfi);
                        auto const ji = Ji[c]->const_array(mfi), total = J[c]->const_array(mfi);
                        amrex::ParallelFor(
                            mfi.validbox(),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k)
                            {
                                out(i, j, k, 0) = vol(i, j, k) *
                                                  (ji(i, j, k) - (hall ? total(i, j, k) : 0.0_rt)) *
                                                  e(i, j, k);
                                out(i, j, k, 1) = std::abs(out(i, j, k, 0));
                            });
                    }
                    work += measured.sum_unique(0, false, geom.periodicity());
                    scale += measured.sum_unique(1, false, geom.periodicity());
                }
                auto const relative = std::abs(work) / scale;
                amrex::Print() << "MOTIONAL_WORK {\"phase\":" << phase
                               << ",\"compatible\":" << compatible << ",\"work\":" << work
                               << ",\"absolute_work\":" << scale << ",\"relative\":" << relative
                               << "}\n";
                AMREX_ALWAYS_ASSERT(std::isfinite(relative));
                if (compatible)
                {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                        relative < 2.e-13_rt, "Native motional force creates electron energy");
                }
                else
                {
                    largest_legacy = amrex::max(largest_legacy, relative);
                }
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            largest_legacy > 1.e-5_rt, "Negative control must expose the legacy Hall work defect");
    }
    warpx::initialization::finalize_external_libraries();
}
