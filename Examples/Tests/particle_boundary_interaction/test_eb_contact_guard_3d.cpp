/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <exception>
#include <string>

int main(int argc, char **argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto &w = WarpX::GetInstance();
        w.InitData();
        auto &pc = w.GetPartContainer().GetParticleContainer(0);
        AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 0);
        amrex::Vector<amrex::ParticleReal> x{.16}, zero{0.};
        amrex::Vector<amrex::Vector<amrex::ParticleReal>> weight{{1.}};
        pc.AddNParticles(0, 1, x, zero, zero, zero, zero, zero, 1, weight, 0, {}, 0, -1);
        AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 1);
        // AddNParticles intentionally absorbs initial solid-side markers.
        // Move a valid marker within the same cell to manufacture the invalid
        // contact after insertion, as a malformed restart/trajectory would.
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti)
        {
            auto const data = pti.GetParticleTile().getParticleTileData();
            amrex::ParallelFor(pti.numParticles(), [=] AMREX_GPU_DEVICE(int p)
                               { data.m_rdata[PIdx::x][p] = .18; });
        }
        int caught = 0;
        try
        {
            w.GetPartContainer().ScrapeParticlesAtEB(
                w.m_fields.get_mr_levels(warpx::fields::FieldType::distance_to_eb, 0));
        }
        catch (std::exception const &error)
        {
            if (std::string(error.what()).find("EB particle contact could not be bracketed") ==
                std::string::npos)
            {
                throw;
            }
            caught = 1;
        }
        amrex::ParallelDescriptor::ReduceIntSum(caught);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            caught > 0, "Invalid contact must reach the host guard, including release GPU builds");
        AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 1);
        amrex::Print() << "CONTACT_HOST_GUARD_PASS ranks_reporting=" << caught << "\n";
        WarpX::ResetInstance();
    }
    warpx::initialization::finalize_external_libraries();
}
