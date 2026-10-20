/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CARTESIAN_HYBRID_HISTORY_H_
#define WARPX_CARTESIAN_HYBRID_HISTORY_H_
#if defined(WARPX_DIM_3D)
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

namespace warpx::cartesian_history
{
    // Deposits belong to the accepted field update. Splitting, loss, and
    // velocity synchronization can subsequently change the particles.
    inline std::map<std::string, amrex::MultiFab*> Fields (WarpX& sim, int lev, bool temperatures = true)
    {
        using warpx::fields::FieldType;
        using ablastr::fields::Direction;
        std::map<std::string, amrex::MultiFab*> result;
        result["rho"] = sim.m_fields.get(FieldType::rho_fp, lev);
        result["rho_old"] = sim.m_fields.get(FieldType::hybrid_rho_fp_temp, lev);
        result["Pe"] = sim.m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
        for (int d = 0; d < 3; ++d) {
            result["J" + std::to_string(d)] =
                sim.m_fields.get(FieldType::current_fp, Direction{d}, lev);
            result["Jold" + std::to_string(d)] =
                sim.m_fields.get(FieldType::hybrid_current_fp_temp, Direction{d}, lev);
        }
        if (sim.get_pointer_HybridPICModel()->m_need_per_species_fields) {
            auto& particles = sim.GetPartContainer();
            for (auto const& species : particles.GetSpeciesNames()) {
                if (particles.GetParticleContainerFromName(species).getCharge() == 0.0) {
                    continue;
                }
                result["rho_" + species] = sim.m_fields.get("rho_fp_" + species, lev);
                for (int d = 0; d < 3; ++d) {
                    result["J" + std::to_string(d) + "_" + species] =
                        sim.m_fields.get("current_fp_" + species, Direction{d}, lev);
                }
            }
        }
        if (temperatures) {
            auto& particles = sim.GetPartContainer();
            for (auto const& species : particles.GetSpeciesNames()) {
                if (!particles.GetParticleContainerFromName(species).getTemperatureDepositionFlag()) {
                    continue;
                }
                for (int d = 0; d < 3; ++d) {
                    result["T" + std::to_string(d) + "_" + species] =
                        sim.m_fields.get("T_" + species, Direction{d}, lev);
                }
            }
        }
        return result;
    }

    inline void Write (WarpX& sim, std::string const& checkpoint)
    {
        auto const* model = sim.get_pointer_HybridPICModel();
        if (model == nullptr || sim.evolve_scheme != EvolveScheme::Explicit) { return; }
        std::string const folder = checkpoint + "/CartesianHybridHistory";
        if (amrex::ParallelDescriptor::IOProcessor()) {
            AMREX_ALWAYS_ASSERT(amrex::UtilCreateDirectory(folder, 0750));
        }
        amrex::ParallelDescriptor::Barrier();
        std::ostringstream metadata;
        metadata << std::setprecision(17) << "CartesianHybridHistory 2\n"
                 << sim.getistep(0) << ' ' << sim.gett_new(0) << ' '
                 << sim.finestLevel() + 1 << ' ' << model->m_evolve_initialized << ' '
                 << model->m_t_deposited_once << ' ' << model->m_t_deposit_interval << '\n';
        if (model->m_evolve_initialized) {
            for (int lev = 0; lev <= sim.finestLevel(); ++lev) {
                auto const fields = Fields(sim, lev);
                metadata << lev << ' ' << fields.size() << '\n';
                for (auto const& [name, field] : fields) {
                    metadata << name << ' ' << field->nComp() << '\n';
                    amrex::VisMF::Write(*field,
                        folder + "/level-" + std::to_string(lev) + "-" + name);
                }
            }
        }
        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream output(folder + "/Header");
            output << metadata.str();
            output.close();
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(output.good(),
                "Cannot write Cartesian hybrid moment-history manifest");
        }
        amrex::ParallelDescriptor::Barrier();
    }

    inline void Read (WarpX& sim, std::string const& checkpoint)
    {
        auto* model = sim.get_pointer_HybridPICModel();
        if (model == nullptr || sim.evolve_scheme != EvolveScheme::Explicit) { return; }
        std::string const folder = checkpoint + "/CartesianHybridHistory";
        if (!amrex::FileExists(folder + "/Header")) {
            amrex::Print() << "restart: Cartesian hybrid moment history absent; "
                "legacy checkpoint will rebuild moments from particles\n";
            return;
        }
        std::ifstream input(folder + "/Header");
        std::string magic;
        int version = 0, step = -1, levels = 0;
        amrex::Real time = 0.0;
        bool initialized = false;
        input >> magic >> version >> step >> time >> levels >> initialized;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input && magic == "CartesianHybridHistory"
            && (version == 1 || version == 2) && step == sim.getistep(0) && time == sim.gett_new(0)
            && levels == sim.finestLevel() + 1,
            "Invalid Cartesian hybrid moment-history manifest");
        bool temperatures_deposited = false;
        if (version == 2) {
            int temperature_interval = 0;
            input >> temperatures_deposited >> temperature_interval;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input &&
                temperature_interval == model->m_t_deposit_interval,
                "Cartesian temperature-history deposition interval mismatch");
        } else {
            bool required = false;
            amrex::ParmParse("warpx").query("require_restart_rng_state", required);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!required,
                "Exact native restart requires cached temperature history (version 2)");
            amrex::Print() << "restart: legacy Cartesian history lacks cached ion temperatures\n";
        }
        if (!initialized) {
            // A legacy donor may be checkpointed before its first Evolve.
            // Its native record explicitly declares that moments need initialization.
            return;
        }
        for (int lev = 0; lev < levels; ++lev) {
            auto const fields = Fields(sim, lev, version >= 2);
            int saved_level = -1;
            std::size_t count = 0;
            input >> saved_level >> count;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input && saved_level == lev
                && count == fields.size(), "Cartesian hybrid moment-history layout mismatch");
            for (auto const& [name, field] : fields) {
                std::string saved_name;
                int components = 0;
                input >> saved_name >> components;
                std::string const path = folder + "/level-" + std::to_string(lev) + "-" + name;
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input && saved_name == name
                    && components == field->nComp() && amrex::FileExists(path + "_H"),
                    "Missing or incompatible Cartesian hybrid carried moment " + name);
                amrex::VisMF::Read(*field, path);
            }
        }
        std::string extra;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(input >> extra),
            "Unexpected Cartesian hybrid moment-history manifest content");
        model->m_t_deposited_once = temperatures_deposited;
        model->m_restart_moments_restored = true;
        amrex::Print() << "restart: Cartesian hybrid carried moments and pressure restored\n";
    }
}
#endif
#endif
