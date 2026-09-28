/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ExpectedIonExchangeSource.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "WarpX.H"
#include <cmath>
namespace warpx::thermal {
bool
EvaluateNativeExpectedIonExchangeSource (ExpectedIonExchangeSource& source,
    const AcceptedIonExchange& exchange, MultiParticleContainer& particles,
    const amrex::MultiFab& budget, ExpectedIonEndpointContract contract) {
    int ready = contract == ExpectedIonEndpointContract::CompleteRedistributedPreCollisionEndpoint
        && exchange.Status() == IonExchangeStatus::Prepared;
    amrex::ParallelDescriptor::ReduceIntMin(ready);
    if (!ready) return false;
    auto const& w = WarpX::GetInstance();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !EB::enabled() && w.finestLevel() == 0 &&
            WarpX::n_rz_azimuthal_modes == 1,
        "ExpectedIonExchangeSource native bridge requires single-level m0/no-EB");
    auto const& geometry = w.Geom(0);
    auto const& expected = exchange.Geometry();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Domain() == expected.Domain(),
        "ExpectedIonExchangeSource native domain mismatch");
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            geometry.ProbLo(d) == expected.ProbLo(d) &&
                geometry.ProbHi(d) == expected.ProbHi(d) &&
                geometry.isPeriodic(d) == expected.isPeriodic(d),
            "ExpectedIonExchangeSource native geometry mismatch");
    struct Indices {
        amrex::Gpu::DeviceVector<amrex::IntVect> cell;
#ifndef WARPX_DIM_RZ
        amrex::Gpu::DeviceVector<amrex::ParticleReal> theta;
#endif
    };
    std::vector<std::unique_ptr<Indices>> storage;
    std::vector<IonExchangeParticleView> views;
    auto const plo = geometry.ProbLoArray(), dxi = geometry.InvCellSizeArray();
    std::size_t species = 0;
    for (auto const& name : particles.GetSpeciesNames()) {
        auto& pc = particles.GetParticleContainerFromName(name);
        if (pc.getCharge() == 0)
            continue;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            species < exchange.Descriptors().size(),
            "AcceptedIonExchange missing charged species");
        auto const& descriptor = exchange.Descriptors()[species];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            name == descriptor.name && pc.getMass() == descriptor.mass &&
                pc.getCharge() / PhysConst::q_e == descriptor.charge_number,
            "ExpectedIonExchangeSource native charged species order/metadata "
            "mismatch");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            dynamic_cast<PhysicalParticleContainer*>(&pc) &&
                !dynamic_cast<SubcycledParticleContainer*>(&pc) &&
                !pc.DoFieldIonization(),
            "AcceptedIonExchange variable-charge/explicit subcycled species "
            "unsupported");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc.ParticleBoxArray(0) == exchange.Cells() &&
                pc.ParticleDistributionMap(0) == exchange.Distribution(),
            "ExpectedIonExchangeSource native particle layout mismatch");
        if (!descriptor.relaxation_excluded)
            for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                long const count = pti.numParticles();
                if (count == 0)
                    continue;
                auto owned = std::make_unique<Indices>();
                owned->cell.resize(count);
                auto index = owned->cell.data();
                auto data = pti.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(long p) {
                    auto const particle =
                        WarpXParticleContainer::ParticleType(data, p);
                    // Do not convert NaN/Inf positions to integer cell indices.
                    // Endpoint redistribution must remove invalid/deleted ids.
                    bool valid = particle.id() > 0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        valid = valid && std::isfinite(particle.pos(d));
                    index[p] = valid
                                   ? amrex::getParticleCell(particle, plo, dxi)
                                   : amrex::IntVect(-1);
                });
                IonExchangeParticleView view;
                view.species = species;
                view.grid_index = pti.index();
                view.count = count;
                view.cell = index;
#ifdef WARPX_DIM_RZ
                view.theta = pti.GetAttribs(PIdx::theta).dataPtr();
#else
                owned->theta.resize(count, 0);
                view.theta = owned->theta.data();
#endif
                view.weight = pti.GetAttribs(PIdx::w).dataPtr();
                view.momentum = {pti.GetAttribs(PIdx::ux).dataPtr(),
                                 pti.GetAttribs(PIdx::uy).dataPtr(),
                                 pti.GetAttribs(PIdx::uz).dataPtr()};
                views.push_back(view);
                storage.push_back(std::move(owned));
            }
        ++species;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        species == exchange.Descriptors().size(),
        "AcceptedIonExchange unexpected charged species metadata");
    return source.Evaluate(exchange,views,budget,contract);
}
} // namespace warpx::thermal
