/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "EmbeddedBoundary/Enabled.H"
#include "ImplicitIonEndpointTrial.H"
#include "ImplicitSolver.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "WarpX.H"
namespace warpx::thermal {
bool
RefreshNativeImplicitIonEndpointTrial (ImplicitIonEndpointTrial& trial,
                                       MultiParticleContainer& particles) {
    auto const& w = WarpX::GetInstance();
    auto const& exchange = trial.Layout();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !EB::enabled() && w.finestLevel() == 0 && !w.getdo_moving_window(),
        "Private endpoint trial requires static single-level/no-EB");
    auto const& g = w.Geom(0);
    auto const& e = exchange.Geometry();
    AMREX_ALWAYS_ASSERT(g.Domain() == e.Domain());
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        AMREX_ALWAYS_ASSERT(g.ProbLo(d) == e.ProbLo(d) &&
                            g.ProbHi(d) == e.ProbHi(d) &&
                            g.isPeriodic(d) == e.isPeriodic(d));
    std::vector<ImplicitIonMidpointTile> views;
    std::size_t s = 0;
    for (auto const& name : particles.GetSpeciesNames()) {
        auto& pc = particles.GetParticleContainerFromName(name);
        if (pc.getCharge() == 0)
            continue;
        AMREX_ALWAYS_ASSERT(s < exchange.Descriptors().size());
        auto const& descriptor = exchange.Descriptors()[s];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            name == descriptor.name && pc.getMass() == descriptor.mass &&
                pc.getCharge() / PhysConst::q_e == descriptor.charge_number &&
                dynamic_cast<PhysicalParticleContainer*>(&pc) &&
                !dynamic_cast<SubcycledParticleContainer*>(&pc) &&
                !pc.DoFieldIonization(),
            "Private endpoint trial charged species metadata/container "
            "mismatch");
        AMREX_ALWAYS_ASSERT(pc.ParticleBoxArray(0) == exchange.Cells() &&
                            pc.ParticleDistributionMap(0) ==
                                exchange.Distribution());
        if (!descriptor.relaxation_excluded)
            for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                ImplicitIonMidpointTile t;
                t.species = s;
                t.allow_deterministic_absorption=true;
                t.absorption=warpx::implicit::MakeAbsorbingEndpointBoundary(g,
                    pc.GetParticleBoundaryData(),WarpX::field_boundary_lo,WarpX::field_boundary_hi,
                    w.get_pointer_ImplicitSolver() &&
                    !w.get_pointer_ImplicitSolver()->ReflectsParticlesInsidePush());
                t.grid = pti.index();
                t.tile = pti.LocalTileIndex();
                t.count = pti.numParticles();
                t.idcpu = pti.GetStructOfArrays().GetIdCPUData().data();
                t.weight = pti.GetAttribs(PIdx::w).dataPtr();
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    t.position[d] =
                        pti.GetStructOfArrays().GetRealData(d).data();
#ifdef WARPX_DIM_RZ
                t.theta = pti.GetAttribs(PIdx::theta).dataPtr();
#endif
                t.old_position = {pti.GetAttribs("x_n").dataPtr(),
                                  pti.GetAttribs("y_n").dataPtr(),
                                  pti.GetAttribs("z_n").dataPtr()};
                t.old_momentum = {pti.GetAttribs("ux_n").dataPtr(),
                                  pti.GetAttribs("uy_n").dataPtr(),
                                  pti.GetAttribs("uz_n").dataPtr()};
                t.momentum = {pti.GetAttribs(PIdx::ux).dataPtr(),
                              pti.GetAttribs(PIdx::uy).dataPtr(),
                              pti.GetAttribs(PIdx::uz).dataPtr()};
                views.push_back(t);
            }
        ++s;
    }
    AMREX_ALWAYS_ASSERT(s == exchange.Descriptors().size());
    return trial.Refresh(views);
}
} // namespace warpx::thermal
