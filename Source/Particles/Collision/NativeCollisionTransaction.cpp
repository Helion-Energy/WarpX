/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeCollisionTransaction.H"

#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "FieldSolver/ImplicitSolvers/ThermalRandomCheckpoint.H"
#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_LayoutData.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace warpx::particles {
namespace {
using Status = NativeCollisionTransaction::Status;
using PC = WarpXParticleContainer;

Status Collective (Status status)
{
    int value = static_cast<int>(status);
    amrex::ParallelDescriptor::ReduceIntMax(value);
    return static_cast<Status>(value);
}

bool Supported (WarpX& simulation)
{
#if defined(WARPX_QED) || defined(AMREX_SINGLE_PRECISION_PARTICLES) || \
    (!defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)) || \
    (defined(AMREX_USE_GPU) && !defined(AMREX_USE_CUDA) && !defined(AMREX_USE_HIP))
    amrex::ignore_unused(simulation);
    return false;
#else
    if (amrex::OpenMP::in_parallel()) { return false; }
    amrex::Vector<std::string> names;
    amrex::ParmParse("collisions").queryarr("collision_names", names);
    for (auto const& name : names) {
        std::string type;
        amrex::ParmParse(name).query("type", type);
        std::transform(type.begin(), type.end(), type.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (type != "nuclearfusion" ||
            BinaryCollisionUtils::get_nuclear_fusion_type(name, &simulation.GetPartContainer()) !=
                NuclearFusionType::DeuteriumTritiumToNeutronHelium) {
            return false;
        }
    }
    for (auto const& pc : simulation.GetPartContainer()) {
        if (!dynamic_cast<PhysicalParticleContainer const*>(pc.get())) { return false; }
    }
    return true;
#endif
}
} // namespace

struct NativeCollisionTransaction::Impl
{
    struct Species {
        PC* container = nullptr;
        std::vector<std::string> real_names, int_names;
        amrex::Vector<PC::ParticleLevel> particles;
        amrex::Real charge = 0, mass = 0;
    };
    struct Grid {
        amrex::BoxArray boxes;
        amrex::DistributionMapping distribution;
        amrex::Geometry geometry;
        amrex::Real time = 0;
        int step = 0;
        std::unique_ptr<amrex::LayoutData<amrex::Real>> costs;
    };
    WarpX* simulation = nullptr;
    std::vector<Species> species;
    std::vector<Grid> grids;
    std::vector<std::string> species_names;
    amrex::Long next_id = 0;
    std::string random_compatibility, host_random;
    amrex::Gpu::DeviceVector<char> device_random;
    Memory memory;
    bool active = false;
};

NativeCollisionTransaction::NativeCollisionTransaction () : m_impl(std::make_unique<Impl>()) {}
NativeCollisionTransaction::~NativeCollisionTransaction () = default;

NativeCollisionTransaction::Status NativeCollisionTransaction::Capture (WarpX& simulation)
{
    auto status = Collective(m_impl->active ? Status::AlreadyActive :
                             Supported(simulation) ? Status::Success : Status::UnsupportedScope);
    if (status != Status::Success) { return status; }
    amrex::Gpu::synchronize();
    auto candidate = std::make_unique<Impl>();
    candidate->simulation = &simulation;
    candidate->species_names = simulation.GetPartContainer().GetSpeciesNames();
    candidate->species.reserve(candidate->species_names.size());
    // The public pinned AMReX counter is read without calling NextID(), whose
    // no-argument overload INCREMENTS it. Capture is quiescent and has no ID side effect.
    candidate->next_id = PC::ParticleType::the_next_id;
    for (auto const& pc : simulation.GetPartContainer()) {
        candidate->species.emplace_back();
        auto& saved = candidate->species.back();
        saved.container = pc.get();
        saved.real_names = pc->GetRealSoANames();
        saved.int_names = pc->GetIntSoANames();
        saved.mass = pc->getMass();
        saved.charge = pc->getCharge();
        auto const& source = pc->GetParticles();
        saved.particles.resize(source.size());
        for (amrex::Long lev = 0; lev < source.size(); ++lev) {
            for (auto const& [key, tile] : source[lev]) {
                auto& copy = saved.particles[lev][key];
                // Preserve the tile's actual runtime schema, including native
                // lazy empty product tiles. SoA assignment deep-copies every
                // PODVector on its original arena and preserves the container's
                // name-table pointers plus neighbor metadata. PODVector uses
                // device-to-device memcpy for device allocations. No host data
                // staging or container mutation is required.
                copy.define(tile.NumRealComps() - PIdx::nattribs,
                            tile.NumIntComps() - IntIdx::nattribs,
                            nullptr, nullptr, pc->arena());
                copy.GetStructOfArrays() = tile.GetStructOfArrays();
                auto const& soa = copy.GetStructOfArrays();
                auto& memory = candidate->memory;
                ++memory.tiles;
                memory.particles += tile.numTotalParticles();
                auto const& ids = soa.GetIdCPUData();
                memory.particle_bytes += ids.size() * sizeof(std::uint64_t);
                memory.particle_capacity_bytes += ids.capacity() * sizeof(std::uint64_t);
                for (int c = 0; c < copy.NumRealComps(); ++c) {
                    auto const& a = soa.GetRealData(c);
                    memory.particle_bytes += a.size() * sizeof(amrex::ParticleReal);
                    memory.particle_capacity_bytes += a.capacity() * sizeof(amrex::ParticleReal);
                }
                for (int c = 0; c < copy.NumIntComps(); ++c) {
                    auto const& a = soa.GetIntData(c);
                    memory.particle_bytes += a.size() * sizeof(int);
                    memory.particle_capacity_bytes += a.capacity() * sizeof(int);
                }
            }
        }
    }
    for (int lev = 0; lev <= simulation.finestLevel(); ++lev) {
        candidate->grids.emplace_back();
        auto& saved = candidate->grids.back();
        saved.boxes = simulation.boxArray(lev);
        saved.distribution = simulation.DistributionMap(lev);
        saved.geometry = simulation.Geom(lev);
        saved.time = simulation.gett_new(lev);
        saved.step = simulation.getistep(lev);
        if (auto const* cost = WarpX::getCosts(lev)) {
            saved.costs = std::make_unique<amrex::LayoutData<amrex::Real>>(*cost);
            candidate->memory.cost_bytes += cost->local_size() * sizeof(amrex::Real);
        }
    }
    candidate->random_compatibility = warpx::thermal::ThermalRandomCompatibility();
    std::ostringstream random;
    amrex::SaveRandomState(random);
    candidate->host_random = random.str();
    candidate->memory.host_rng_bytes = candidate->host_random.size();
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    candidate->device_random.resize(warpx::thermal::ThermalRandomDeviceBytes());
    auto const* source = reinterpret_cast<char const*>(amrex::getRandState());
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, source,
                    source + candidate->device_random.size(), candidate->device_random.begin());
#endif
    candidate->memory.device_rng_bytes = candidate->device_random.size();
    amrex::Gpu::synchronize();
    candidate->active = true;
    m_impl = std::move(candidate);
    return Status::Success;
}

NativeCollisionTransaction::Status NativeCollisionTransaction::CanRestore (WarpX& simulation) const
{
    auto const& saved = *m_impl;
    Status status = Status::Success;
    if (!saved.active) { status = Status::Inactive; }
    else if (&simulation != saved.simulation || !Supported(simulation) ||
             saved.random_compatibility != warpx::thermal::ThermalRandomCompatibility() ||
             saved.species_names != simulation.GetPartContainer().GetSpeciesNames() ||
             saved.grids.size() != static_cast<std::size_t>(simulation.finestLevel() + 1)) {
        status = Status::ConfigurationChanged;
    } else {
        for (std::size_t i = 0; i < saved.species.size(); ++i) {
            auto const& species = saved.species[i];
            auto const& pc = simulation.GetPartContainer().GetParticleContainerFromName(
                saved.species_names[i]);
            if (&pc != species.container) {
                status = Status::ConfigurationChanged;
                continue;
            }
            if (pc.GetRealSoANames() != species.real_names ||
                pc.GetIntSoANames() != species.int_names || pc.getMass() != species.mass ||
                pc.getCharge() != species.charge ||
                pc.GetParticles().size() != species.particles.size()) {
                status = Status::ConfigurationChanged;
            }
        }
        for (int lev = 0; lev <= simulation.finestLevel(); ++lev) {
            auto const& g = saved.grids[lev];
            auto const& geom = simulation.Geom(lev);
            if (simulation.boxArray(lev) != g.boxes ||
                simulation.DistributionMap(lev) != g.distribution ||
                simulation.gett_new(lev) != g.time || simulation.getistep(lev) != g.step ||
                geom.Domain() != g.geometry.Domain() || geom.Coord() != g.geometry.Coord()) {
                status = Status::ConfigurationChanged;
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (geom.ProbLo(d) != g.geometry.ProbLo(d) ||
                    geom.ProbHi(d) != g.geometry.ProbHi(d) ||
                    geom.isPeriodic(d) != g.geometry.isPeriodic(d)) {
                    status = Status::ConfigurationChanged;
                }
            }
            auto const* cost = WarpX::getCosts(lev);
            if (bool(cost) != bool(g.costs) ||
                (cost && (cost->boxArray() != g.costs->boxArray() ||
                          cost->DistributionMap() != g.costs->DistributionMap()))) {
                status = Status::ConfigurationChanged;
            }
        }
    }
    return Collective(status);
}

NativeCollisionTransaction::Status NativeCollisionTransaction::Restore (
    WarpX& simulation, InvalidationHook invalidate_borrowed_views)
{
    auto status = CanRestore(simulation);
    if (status != Status::Success) { return status; }
    status = Collective(invalidate_borrowed_views.invalidate ? Status::Success :
                        Status::MissingInvalidationHook);
    if (status != Status::Success) { return status; }
    // Every preflight/reduction finishes before ANY accepted state is touched.
    amrex::Gpu::synchronize();
    simulation.DiscardSavedImplicitParticleState();
    if (auto* solver = simulation.get_pointer_ImplicitSolver()) {
        solver->InvalidateMassMatrices();
    }
    invalidate_borrowed_views.invalidate(invalidate_borrowed_views.context);
    for (auto& species : m_impl->species) {
        species.container->GetParticles().swap(species.particles);
    }
    PC::ParticleType::NextID(m_impl->next_id);
    for (int lev = 0; lev <= simulation.finestLevel(); ++lev) {
        if (auto* cost = WarpX::getCosts(lev)) {
            auto const& saved = *m_impl->grids[lev].costs;
            std::copy(saved.data(), saved.data() + saved.local_size(), cost->data());
        }
    }
    std::istringstream random(m_impl->host_random);
    amrex::RestoreRandomState(random, amrex::OpenMP::get_max_threads(), 0);
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    auto* destination = reinterpret_cast<char*>(amrex::getRandState());
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, m_impl->device_random.begin(),
                    m_impl->device_random.end(), destination);
#endif
    amrex::Gpu::synchronize();
    Commit();
    return Status::Success;
}

void NativeCollisionTransaction::Commit ()
{
    amrex::Gpu::synchronize();
    m_impl = std::make_unique<Impl>();
}
bool NativeCollisionTransaction::Active () const { return m_impl->active; }
NativeCollisionTransaction::Memory NativeCollisionTransaction::LocalMemory () const
{
    return m_impl->memory;
}
} // namespace warpx::particles
