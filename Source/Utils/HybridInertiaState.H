/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_INERTIA_STATE_H_
#define WARPX_HYBRID_INERTIA_STATE_H_
#if defined(WARPX_DIM_3D)
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronInertiaElliptic.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>

namespace warpx::hybrid_inertia_state
{
    inline void Write (WarpX& sim, std::string const& checkpoint)
    {
        auto const* model = sim.get_pointer_HybridPICModel();
        if (model == nullptr || sim.evolve_scheme != EvolveScheme::Explicit) { return; }
        std::string const folder = checkpoint + "/HybridInertiaState";
        bool const enabled = model->m_include_electron_inertia_elliptic;
        bool const initialized = enabled && model->m_inertia_elliptic != nullptr;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            AMREX_ALWAYS_ASSERT(amrex::UtilCreateDirectory(folder, 0755));
            std::ofstream output(folder + "/Header");
            output << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
                   << "HybridInertiaState 1\n"
                   << sim.getistep(0) << ' ' << sim.gett_new(0) << ' '
                   << sim.finestLevel() + 1 << ' '
                   << enabled << ' ' << initialized << '\n';
            output.close();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(output.good(),
                "Failed to write native elliptic inertia state manifest");
        }
        amrex::ParallelDescriptor::Barrier();
        if (initialized) { model->m_inertia_elliptic->WriteWarmStart(folder); }
    }

    // Called after model initialization, before restart callbacks or evolution.
    inline void Read (WarpX& sim, std::string const& checkpoint)
    {
        auto* model = sim.get_pointer_HybridPICModel();
        if (model == nullptr || sim.evolve_scheme != EvolveScheme::Explicit) { return; }
        bool required = false;
        amrex::ParmParse("warpx").query("require_restart_rng_state", required);
        std::string const folder = checkpoint + "/HybridInertiaState";
        if (!amrex::FileExists(folder + "/Header")) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                !(required && model->m_include_electron_inertia_elliptic),
                "Exact native restart requires elliptic inertia warm-start state");
            if (model->m_include_electron_inertia_elliptic) {
                amrex::Print() << "restart: legacy checkpoint has no elliptic warm start; "
                                 "the first solve will use a zero correction\n";
            }
            model->m_inertia_elliptic.reset();
            return;
        }
        std::ifstream input(folder + "/Header");
        std::string magic;
        int version = 0, step = -1, levels = 0;
        amrex::Real time = 0.0;
        bool enabled = false, initialized = false;
        input >> magic >> version >> step >> time >> levels >> enabled >> initialized;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            input && magic == "HybridInertiaState" && version == 1
            && step == sim.getistep(0) && time == sim.gett_new(0)
            && levels == sim.finestLevel() + 1
            && enabled == model->m_include_electron_inertia_elliptic
            && (!initialized || enabled),
            "Invalid or incompatible native elliptic inertia state manifest");
        std::string extra;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(input >> extra),
            "Unexpected native elliptic inertia manifest content");
        model->m_inertia_elliptic.reset();
        if (initialized) {
            auto solver = std::make_unique<ElectronInertiaElliptic>();
            solver->m_rtol = model->m_electron_inertia_rtol;
            solver->m_max_iter = model->m_electron_inertia_max_iters;
            solver->m_verbose = model->m_electron_inertia_verbose;
            solver->m_warm_start = model->m_electron_inertia_warm_start;
            solver->Define(sim.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, 0), 0);
            solver->ReadWarmStart(folder);
            model->m_inertia_elliptic = std::move(solver);
        }
        amrex::Print() << "restart: native elliptic correction and warm-start validity restored\n";
    }
}
#endif
#endif
