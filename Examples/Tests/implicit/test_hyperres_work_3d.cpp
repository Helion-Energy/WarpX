/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Certify the volume-adjoint Cartesian curl-curl stabilizer, including walls.
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/MotionalFieldCartesian.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <ablastr/coarsen/sample.H>
#include <cmath>
#include <iomanip>

using namespace amrex::literals;
using warpx::fields::FieldType;

int main(int argc, char **argv)
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
            // Apply the same conductor projection used for the plasma current.
            MotionalFieldCartesian projection;
            projection.conformal = hp.m_use_conformal_eb;
            auto lengths = EB::enabled() && projection.conformal
                                         ? w.m_fields.get_alldirs(FieldType::edge_lengths, 0)
                                         : ablastr::fields::VectorField{nullptr, nullptr, nullptr};
            for (int d = 0; d < 3; ++d)
            {
                projection.lo[d] = geom.Domain().smallEnd(d);
                projection.hi[d] = geom.Domain().bigEnd(d);
                projection.periodic[d] = geom.isPeriodic(d);
                projection.dx[d] = dx[d];
                projection.pec_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PEC;
                projection.pec_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PEC;
                Ji[d]->setVal(0.0_rt);
                if (EB::enabled())
                {
                    if (lengths[d]) { lengths[d]->FillBoundary(geom.periodicity()); }
                    w.GetEBUpdateEFlag()[0][d]->FillBoundary(geom.periodicity());
                }
            }
            for (int c = 0; c < 3; ++c)
            {
                for (amrex::MFIter mfi(*J[c]); mfi.isValid(); ++mfi)
                {
                    auto mask = projection;
                    if (EB::enabled())
                    {
                        for (int d = 0; d < 3; ++d)
                        {
                            mask.update[d] = w.GetEBUpdateEFlag()[0][d]->const_array(mfi);
                            if (lengths[d]) { mask.length[d] = lengths[d]->const_array(mfi); }
                        }
                    }
                    auto j = J[c]->array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int jj, int k)
                                       {
                                           if (!mask.active(i, jj, k, c))
                                           {
                                               j(i, jj, k) = 0.0_rt;
                                           }
                                       });
                }
                J[c]->FillBoundary(geom.periodicity());
                E[c]->setVal(0.0_rt);
            }
            hp.m_include_hall_term = false;
            hp.HybridPICSolveE(E, Ji, B, rho, w.GetEBUpdateEFlag()[0], 0, true, true);
            amrex::Real work = 0.0_rt, dissipation = 0.0_rt;
            auto K = w.m_fields.get_alldirs("hybrid_hyperres_curlJ_fp", 0);
            auto const eta = hp.m_eta_h;
            for (int c = 0; c < 3; ++c)
            {
                amrex::MultiFab measured(E[c]->boxArray(), E[c]->DistributionMap(), 1, 0);
                auto const vol = MakeQdsmcVolumeElement(geom, E[c]->ixType());
                for (amrex::MFIter mfi(measured); mfi.isValid(); ++mfi)
                {
                    auto const out = measured.array(mfi);
                    auto const e = E[c]->const_array(mfi), j = J[c]->const_array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int jj, int k)
                        { out(i, jj, k) = vol(i, jj, k) * j(i, jj, k) * e(i, jj, k); });
                }
                work += measured.sum_unique(0, false, geom.periodicity());
                amrex::MultiFab expected(K[c]->boxArray(), K[c]->DistributionMap(), 1, 0);
                auto const vf = MakeQdsmcVolumeElement(geom, K[c]->ixType());
                for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi)
                {
                    auto const out = expected.array(mfi);
                    auto const kval = K[c]->const_array(mfi), r = rho.const_array(mfi);
                    amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> b;
                    amrex::GpuArray<amrex::GpuArray<int, 3>, 3> bs;
                    for (int d = 0; d < 3; ++d)
                    {
                        b[d] = B[d]->const_array(mfi);
                        auto const iv = B[d]->ixType().toIntVect();
                        bs[d] = {iv[0], iv[1], iv[2]};
                    }
                    amrex::ParallelFor(mfi.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                       {
                                           amrex::Real rr = 0.0_rt;
                                           for (int di = 0; di <= (c != 0); ++di)
                                           {
                                               for (int dj = 0; dj <= (c != 1); ++dj)
                                               {
                                                   for (int dk = 0; dk <= (c != 2); ++dk)
                                                   {
                                                       rr += 0.25_rt * r(i + di, j + dj, k + dk);
                                                   }
                                               }
                                           }
                                           amrex::Real b2 = 0.0_rt;
                                           amrex::GpuArray<int, 3> const unit{1, 1, 1};
                                           for (int d = 0; d < 3; ++d)
                                           {
                                               auto const value = ablastr::coarsen::sample::Interp(
                                                   b[d], bs[d], bs[c], unit, i, j, k, 0);
                                               b2 += value * value;
                                           }
                                           out(i, j, k) = vf(i, j, k) * kval(i, j, k) *
                                                          kval(i, j, k) / eta(rr, std::sqrt(b2));
                                       });
                }
                dissipation += expected.sum_unique(0, false, geom.periodicity());
            }
            auto const relative = std::abs(work - dissipation) / dissipation;
            amrex::Print() << "HYPER_WORK {\"phase\":" << phase << ",\"work\":" << work
                           << ",\"dissipation\":" << dissipation << ",\"relative\":" << relative
                           << "}\n";
            AMREX_ALWAYS_ASSERT(std::isfinite(relative));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                relative < 2.e-13_rt,
                "Cartesian hyper-resistive work is not the volume-adjoint curl dissipation");
            // The energy-booking mirror is the field actually applied. The
            // particle-push call excludes the stabilizer and must not reuse K.
            std::array<std::unique_ptr<amrex::MultiFab>, 3> mirror_storage;
            ablastr::fields::VectorField mirror;
            for (int c = 0; c < 3; ++c)
            {
                mirror_storage[c] = std::make_unique<amrex::MultiFab>(
                    E[c]->boxArray(), E[c]->DistributionMap(), 1, 0);
                mirror[c] = mirror_storage[c].get();
                mirror[c]->setVal(123.0_rt);
            }
            auto *solver = w.get_pointer_fdtd_solver_fp(0);
            solver->HybridPICSolveE(E, J, Ji, B, rho, pressure, w.GetEBUpdateEFlag()[0], 0, &hp,
                                    true, true, &mirror);
            for (int c = 0; c < 3; ++c)
            {
                amrex::MultiFab::Subtract(*mirror[c], *E[c], 0, 0, 1, 0);
                AMREX_ALWAYS_ASSERT(mirror[c]->norm0() == 0.0_rt);
            }
            solver->HybridPICSolveE(E, J, Ji, B, rho, pressure, w.GetEBUpdateEFlag()[0], 0, &hp,
                                    false, false, &mirror);
            for (int c = 0; c < 3; ++c)
            {
                AMREX_ALWAYS_ASSERT(E[c]->norm0() == 0.0_rt);
                AMREX_ALWAYS_ASSERT(mirror[c]->norm0() == 0.0_rt);
            }
        }
        // Periodic gradient null family, including mixed grid-scale modes.
        // A component Laplacian would damp this longitudinal current.
        if (geom.isAllPeriodic())
        {
            for (int c = 0; c < 3; ++c)
            {
                for (amrex::MFIter mfi(*J[c]); mfi.isValid(); ++mfi)
                {
                    auto out = J[c]->array(mfi);
                    amrex::ParallelFor(
                        mfi.fabbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        {
                            auto phi = [] AMREX_GPU_DEVICE(int ii, int jj, int kk)
                            {
                                auto const a = (ii % 2 == 0 ? 1.0_rt : -1.0_rt);
                                auto const b = (jj % 2 == 0 ? 1.0_rt : -1.0_rt);
                                auto const d = (kk % 2 == 0 ? 1.0_rt : -1.0_rt);
                                return 1000.0_rt * (a * b + 2.0_rt * b * d + 3.0_rt * d * a);
                            };
                            out(i, j, k) =
                                (phi(i + (c == 0), j + (c == 1), k + (c == 2)) - phi(i, j, k)) /
                                dx[c];
                        });
                }
                J[c]->FillBoundary(geom.periodicity());
            }
            hp.HybridPICSolveE(E, Ji, B, rho, w.GetEBUpdateEFlag()[0], 0, true, true);
            for (int c = 0; c < 3; ++c)
            {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    E[c]->norm0() < 1.e-10_rt,
                    "Curl-curl stabilizer must preserve the discrete gradient null family");
            }
            amrex::Print() << "HYPER_GRADIENT_NULL PASS\n";
        }
    }
    warpx::initialization::finalize_external_libraries();
}
