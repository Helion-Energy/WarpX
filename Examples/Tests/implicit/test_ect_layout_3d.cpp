/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_MultiFab.H>
#include <algorithm>
#include <limits>

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        auto const fields = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        bool const expected =
            fields[0]->boxArray().size() > 1 || sim.Geom(0).isAnyPeriodic();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            sim.ECTNeedsSeamSync() == expected,
            "ECT seam synchronization must follow the allocated field layout "
            "on fresh initialization");
        amrex::Real worst = 0;
        for (int d = 0; d < 3; ++d) {
            auto const& area = *sim.m_fields.get(
                FieldType::area_mod, ablastr::fields::Direction{d}, 0);
            amrex::MultiFab synced(area.boxArray(), area.DistributionMap(), 1,
                                   0);
            amrex::MultiFab::Copy(synced, area, 0, 0, 1, 0);
            synced.OverrideSync(sim.Geom(0).periodicity());
            amrex::MultiFab::Subtract(synced, area, 0, 0, 1, 0);
            amrex::Real const scale =
                std::max(area.norminf(), amrex::Real(1.e-300));
            amrex::Real const relative = synced.norminf() / scale;
            worst = std::max(worst, relative);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                relative <= 64 * std::numeric_limits<amrex::Real>::epsilon(),
                "ECT modified face areas disagree across copies of the same "
                "face");
        }
        amrex::Print() << "ECT_LAYOUT_PASS seam_sync=" << expected
                       << " relative_area_error=" << worst << "\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
