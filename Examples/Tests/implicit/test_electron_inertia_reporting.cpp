/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaElliptic.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_MFIter.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace
{
    using namespace amrex::literals;
    using Field = ablastr::fields::VectorField;
    using amrex::MultiFab;
    using warpx::fields::FieldType;

    struct Scratch
    {
        std::array<MultiFab, 3> values;
        Field field;
        explicit Scratch (Field const& like)
        {
            for (int d = 0; d < 3; ++d) {
                values[d].define(like[d]->boxArray(), like[d]->DistributionMap(),
                                 1, like[d]->nGrowVect());
                values[d].setVal(0.0_rt);
                field[d] = &values[d];
            }
        }
    };

    void manufacture (Field const& field, amrex::Geometry const& geom, int stage)
    {
        auto const lo = geom.ProbLoArray();
        auto const dx = geom.CellSizeArray();
        auto const hi = geom.ProbHiArray();
        for (int d = 0; d < 3; ++d) {
            auto const nodal = field[d]->ixType().toIntVect();
            for (amrex::MFIter mfi(*field[d]); mfi.isValid(); ++mfi) {
                auto const a = field[d]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        amrex::Real const x = (lo[0] + (i + (nodal[0] ? 0.0_rt : 0.5_rt))
                            * dx[0]) / (hi[0] - lo[0]);
                        amrex::Real const y = (lo[1] + (j + (nodal[1] ? 0.0_rt : 0.5_rt))
                            * dx[1]) / (hi[1] - lo[1]);
                        amrex::Real const z = (lo[2] + (k + (nodal[2] ? 0.0_rt : 0.5_rt))
                            * dx[2]) / (hi[2] - lo[2]);
                        amrex::Real const phase = 0.017_rt * stage;
                        a(i,j,k) = (1.0_rt + 0.01_rt * stage) * (
                            std::sin(2.0_rt * MathConst::pi * (x + (d+1)*y) + phase)
                            + 0.3_rt * std::cos(4.0_rt * MathConst::pi * (y+z) - phase)
                            + 0.2_rt * std::sin(6.0_rt * MathConst::pi * (z+x)));
                    });
            }
        }
    }

    amrex::Real compare (Field const& a, Field const& b)
    {
        Scratch delta(a);
        amrex::Real error = 0.0_rt;
        amrex::Real scale = 0.0_rt;
        for (int d = 0; d < 3; ++d) {
            MultiFab::Copy(delta.values[d], *a[d], 0, 0, 1, 0);
            MultiFab::Subtract(delta.values[d], *b[d], 0, 0, 1, 0);
            error = std::max(error, delta.values[d].norminf());
            scale = std::max(scale, b[d]->norminf());
        }
        AMREX_ALWAYS_ASSERT(std::isfinite(error) && scale > 0.0_rt);
        amrex::Real const relative = error / scale;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(relative <= 1.e-13_rt,
            "Reporting must not change the numerical inertia solution");
        return relative;
    }
}

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        rho.setVal(1.e18_rt * PhysConst::q_e);
        amrex::Print() << std::setprecision(17);
        for (bool const warm : {true, false}) {
            Scratch quiet_field(E), reported_field(E);
            ElectronInertiaElliptic quiet, reported;
            quiet.m_warm_start = reported.m_warm_start = warm;
            quiet.m_rtol = reported.m_rtol = 1.e-8_rt;
            quiet.m_max_iter = reported.m_max_iter = 500;
            quiet.m_verbose = 0;
            reported.m_verbose = 1;
            quiet.Define(quiet_field.field, 0);
            reported.Define(reported_field.field, 0);
            for (int stage = 0; stage < 4; ++stage) {
                manufacture(quiet_field.field, w.Geom(0), stage);
                manufacture(reported_field.field, w.Geom(0), stage);
                quiet.PrepareCoefficients(rho, 1.e16_rt * PhysConst::q_e, 0);
                reported.PrepareCoefficients(rho, 1.e16_rt * PhysConst::q_e, 0);
                long const old_iters = quiet.m_n_iters;
                quiet.Solve(quiet_field.field, 0);
                reported.Solve(reported_field.field, 0);
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    reported.m_n_cold_solves == ((stage == 0 || !warm) ? 1 : 0),
                    "Resetting statistics must not invalidate the previous correction");
                AMREX_ALWAYS_ASSERT(quiet.m_n_iters - old_iters == reported.m_n_iters);
                auto const relative = compare(quiet_field.field, reported_field.field);
                amrex::Print() << "INERTIA_REPORT warm=" << warm << " stage=" << stage
                               << " relative=" << relative << '\n';
                quiet.ReportAndResetStats(2);
                reported.ReportAndResetStats(2);
                AMREX_ALWAYS_ASSERT(reported.m_n_solves == 0);
            }

            // A zero RHS is a valid previous zero correction. Reporting after
            // this early return must leave the next solve's warm state intact.
            for (auto& a : quiet_field.values) { a.setVal(0.0_rt); }
            for (auto& a : reported_field.values) { a.setVal(0.0_rt); }
            quiet.Solve(quiet_field.field, 0);
            reported.Solve(reported_field.field, 0);
            reported.ReportAndResetStats(2);
            manufacture(quiet_field.field, w.Geom(0), 5);
            manufacture(reported_field.field, w.Geom(0), 5);
            quiet.Solve(quiet_field.field, 0);
            reported.Solve(reported_field.field, 0);
            AMREX_ALWAYS_ASSERT(reported.m_n_cold_solves == (warm ? 0 : 1));
            compare(quiet_field.field, reported_field.field);

            // A new layout really does invalidate the previous correction.
            long const previous_cold = quiet.m_n_cold_solves;
            quiet.Define(quiet_field.field, 0);
            quiet.PrepareCoefficients(rho, 1.e16_rt * PhysConst::q_e, 0);
            manufacture(quiet_field.field, w.Geom(0), 6);
            quiet.Solve(quiet_field.field, 0);
            AMREX_ALWAYS_ASSERT(quiet.m_n_cold_solves == previous_cold + 1);
        }
        amrex::Print() << "INERTIA_REPORTING_PASS\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
