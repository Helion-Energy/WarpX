/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Python/callbacks.H"
#include "Utils/CartesianHybridHistory.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto const verify = [&] (bool only_temperature = false) {
            int index = 0;
            int temperatures = 0;
            AMREX_ALWAYS_ASSERT(sim.get_pointer_HybridPICModel()->m_t_deposited_once);
            AMREX_ALWAYS_ASSERT(sim.get_pointer_HybridPICModel()->m_t_deposit_interval == 10);
            for (auto const& [name, field] : warpx::cartesian_history::Fields(sim, 0)) {
                temperatures += name[0] == 'T';
                amrex::Real const value = 1.25 * (++index);
                if (only_temperature && name[0] != 'T') { continue; }
                for (int c = 0; c < field->nComp(); ++c) {
                    AMREX_ALWAYS_ASSERT(field->min(c, field->nGrow()) == value);
                    AMREX_ALWAYS_ASSERT(field->max(c, field->nGrow()) == value);
                }
            }
            AMREX_ALWAYS_ASSERT(temperatures == 6);
        };
        if (sim.getistep(0) == 0) {
            InstallPythonCallback("afterstep", [&] () {
                int index = 0;
                for (auto const& [name, field] : warpx::cartesian_history::Fields(sim, 0)) {
                    field->setVal(1.25 * (++index));
                }
            });
            sim.Evolve();
            AMREX_ALWAYS_ASSERT(sim.getistep(0) == 1);
        } else {
            AMREX_ALWAYS_ASSERT(sim.getistep(0) == 1);
            AMREX_ALWAYS_ASSERT(sim.get_pointer_HybridPICModel()->m_restart_moments_restored);
            verify();
            // The actual first-Evolve initialization must preserve these
            // accepted deposits, including their ghost cells and pressure.
            sim.HybridPICInitializeRhoJandB();
            verify();
            // Step 1 is between deposits at 0 and 10. The first resumed
            // charge/current update must retain the saved temperature cache.
            sim.HybridPICDepositRhoAndJ();
            verify(true);
            amrex::Print() << "CARTESIAN_CARRIED_HISTORY_RESTORE_EXACT\n";
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
