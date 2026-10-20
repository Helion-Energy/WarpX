/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaElliptic.H"
#include "Initialization/WarpXInit.H"
#include "Utils/HybridInertiaState.H"
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <filesystem>
#include <fstream>
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
        auto& hp = *w.get_pointer_HybridPICModel();
        std::string mode = "warm";
        amrex::ParmParse("test").query("case", mode);
        bool const initialized = mode != "uninitialized" && mode != "disabled";
        hp.m_include_electron_inertia_elliptic = mode != "disabled";
        hp.m_electron_inertia_warm_start = mode != "cold";
        hp.m_electron_inertia_rtol = 1.e-8_rt;
        hp.m_electron_inertia_max_iters = 500;
        hp.m_inertia_elliptic.reset();
        auto const E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto& rho = *w.m_fields.get(FieldType::rho_fp, 0);
        rho.setVal(1.e18_rt * PhysConst::q_e);
        Scratch next(E), expected(E);
        if (initialized) {
            hp.m_inertia_elliptic = std::make_unique<ElectronInertiaElliptic>();
            auto& solver = *hp.m_inertia_elliptic;
            solver.m_rtol = hp.m_electron_inertia_rtol;
            solver.m_max_iter = hp.m_electron_inertia_max_iters;
            solver.m_warm_start = hp.m_electron_inertia_warm_start;
            solver.m_verbose = 1;
            solver.Define(E, 0);
            solver.PrepareCoefficients(rho, 1.e16_rt * PhysConst::q_e, 0);
            manufacture(E, w.Geom(0), 0);
            solver.Solve(E, 0);
            if (mode == "zero") {
                for (int d = 0; d < 3; ++d) { E[d]->setVal(0.0_rt); }
                solver.Solve(E, 0);
            }
            solver.ReportAndResetStats(2);
        }
        std::string const directory = "checkpoint-test";
        warpx::hybrid_inertia_state::Write(w, directory);
        auto original = std::move(hp.m_inertia_elliptic);
        std::string const folder = directory + "/HybridInertiaState";
        if (mode == "config") { hp.m_electron_inertia_rtol *= 0.1_rt; }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (mode == "missing") { std::filesystem::remove(folder + "/Header"); }
            if (mode == "component") { std::filesystem::remove(folder + "/correction-0_H"); }
            if (mode == "layout") {
                std::ifstream input(folder + "/StateHeader");
                std::string magic, layout;
                int version = 0, max_iter = 0;
                amrex::Real rtol = 0.0;
                bool warm = false, valid = false;
                input >> magic >> version >> rtol >> max_iter >> warm >> valid
                      >> std::quoted(layout);
                AMREX_ALWAYS_ASSERT(input.good());
                input.close();
                std::ofstream output(folder + "/StateHeader");
                output << std::setprecision(17) << magic << ' ' << version << '\n'
                       << rtol << ' ' << max_iter << ' ' << warm << ' ' << valid << '\n'
                       << std::quoted(layout + "changed-rank-layout") << '\n';
            }
            if (mode == "payload") {
                std::ifstream input(folder + "/correction-0_H");
                amrex::VisMF::Header header;
                input >> header;
                AMREX_ALWAYS_ASSERT(input.good());
                input.close();
                header.m_ngrow = amrex::IntVect(0);
                std::ofstream output(folder + "/correction-0_H");
                output << header;
            }
        }
        amrex::ParallelDescriptor::Barrier();
        warpx::hybrid_inertia_state::Read(w, directory);
        if (!initialized) {
            AMREX_ALWAYS_ASSERT(hp.m_inertia_elliptic == nullptr);
        } else {
            AMREX_ALWAYS_ASSERT(hp.m_inertia_elliptic != nullptr);
            auto& restored = *hp.m_inertia_elliptic;
            manufacture(next.field, w.Geom(0), 1);
            manufacture(expected.field, w.Geom(0), 1);
            original->PrepareCoefficients(rho, 1.e16_rt * PhysConst::q_e, 0);
            restored.PrepareCoefficients(rho, 1.e16_rt * PhysConst::q_e, 0);
            original->Solve(expected.field, 0);
            restored.Solve(next.field, 0);
            AMREX_ALWAYS_ASSERT(original->m_n_cold_solves == restored.m_n_cold_solves);
            AMREX_ALWAYS_ASSERT(restored.m_n_cold_solves == (mode == "cold" ? 1 : 0));
            AMREX_ALWAYS_ASSERT(original->m_n_iters == restored.m_n_iters);
            auto const relative = compare(next.field, expected.field);
            amrex::Print() << std::setprecision(17)
                           << "INERTIA_RESTART relative=" << relative << '\n';
        }
        amrex::Print() << "INERTIA_RESTART_PASS case=" << mode << '\n';
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
