/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ImplicitIonElectricWork.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/ImplicitFinalGather.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "ImplicitSolver.H"
#include "WarpX.H"
#include <algorithm>
#include <memory>
namespace warpx::thermal {
namespace {
IonElectricWorkResult
MeasureNativeIonElectricWork (
    WarpX& simulation, int lev, amrex::Real dt, IonElectricWorkFields const& fields,
    NativeIonWorkContract contract, bool recorded)
{
    auto const gather_contract = recorded ? IonElectricGatherContract::RecordedFinalGather
        : IonElectricGatherContract::FinalMidpointReconstruction;
    IonElectricWorkResult failed;
    failed.gather_contract=gather_contract;
    bool ok=contract==NativeIonWorkContract::FullBorisGatherWithoutAdditionalMomentumChanges &&
        lev==0 && simulation.finestLevel()==0 && !EB::enabled() &&
        !simulation.getdo_moving_window() && WarpX::nox==3 &&
        WarpX::particle_pusher_algo==ParticlePusherAlgo::Boris &&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving &&
        WarpX::current_deposition_algo==CurrentDepositionAlgo::Esirkepov &&
        simulation.m_v_galilean[0]==0 && simulation.m_v_galilean[1]==0 &&
        simulation.m_v_galilean[2]==0 &&
        simulation.get_ng_fieldgather().allGE(fields.filled_ghosts);
#ifdef WARPX_DIM_RZ
    ok=ok && WarpX::n_rz_azimuthal_modes==1;
#endif
    if (lev==0) {
        ok=ok && dt==simulation.getdt(lev);
        auto const ba=amrex::convert(simulation.boxArray(lev),amrex::IntVect::TheNodeVector());
        for (auto f:fields.component) {
            ok=ok && f && f->boxArray()==ba &&
                f->DistributionMap()==simulation.DistributionMap(lev);
        }
    }
    if (recorded) {
        auto const* model=simulation.get_pointer_HybridPICModel();
        auto const* implicit=simulation.get_pointer_ImplicitSolver();
        ok=ok && model && model->UsesEulerianElectronEnergy() && implicit &&
            !implicit->DoParticleSuborbits();
    }
    auto& particles=simulation.GetPartContainer();
    std::vector<IonElectricWorkSpecies> species;
    for (auto const& name:particles.GetSpeciesNames()) {
        auto& pc=particles.GetParticleContainerFromName(name);
        if (pc.getCharge()==0) { continue; }
        ok=ok && dynamic_cast<PhysicalParticleContainer*>(&pc) &&
            !dynamic_cast<SubcycledParticleContainer*>(&pc) && !pc.DoFieldIonization() &&
            pc.ParticleBoxArray(0)==simulation.boxArray(0) &&
            pc.ParticleDistributionMap(0)==simulation.DistributionMap(0);
        auto const& names=pc.GetRealSoANames();
        for (auto const* attribute:{"ux_n","uy_n","uz_n"}) {
            ok=ok && std::find(names.begin(),names.end(),attribute)!=names.end();
        }
        if (recorded) {
            for (auto const* attribute:warpx::particles::FinalGatherAttributeNames) {
                ok=ok && std::find(names.begin(),names.end(),attribute)!=names.end();
            }
            if (auto const* physical=dynamic_cast<PhysicalParticleContainer const*>(&pc)) {
                physical->ValidateImplicitIonElectricWorkCapture();
            }
        }
        species.push_back({name,pc.getMass(),pc.getCharge()});
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok) { return failed; }
    struct PositionStorage {
        std::array<amrex::Gpu::DeviceVector<amrex::ParticleReal>,3> x;
    };
    std::vector<std::unique_ptr<PositionStorage>> storage;
    std::vector<IonElectricWorkParticles> views;
    std::size_t s=0;
    for (auto const& name:particles.GetSpeciesNames()) {
        auto& pc=particles.GetParticleContainerFromName(name);
        if (pc.getCharge()==0) { continue; }
        for (WarpXParIter pti(pc,lev);pti.isValid();++pti) {
            auto const count=pti.numParticles();
            if (count==0) { continue; }
            IonElectricWorkParticles v;
            v.species=s; v.grid=pti.index(); v.tile=pti.LocalTileIndex(); v.count=count;
            v.idcpu=pti.GetStructOfArrays().GetIdCPUData().data();
            v.weight=pti.GetAttribs(PIdx::w).dataPtr();
            if (recorded) {
                for (int d=0;d<3;++d) {
                    v.gather_position[d]=pti.GetAttribs(warpx::particles::FinalGatherAttributeNames[d]).dataPtr();
                }
            } else {
                auto copy=std::make_unique<PositionStorage>();
                for (auto& x:copy->x) { x.resize(count); }
                auto x=copy->x[0].data(), y=copy->x[1].data(), z=copy->x[2].data();
                auto const position=GetParticlePosition<PIdx>(pti);
                amrex::ParallelFor(count,[=] AMREX_GPU_DEVICE(long p) {
                    position(p,x[p],y[p],z[p]);
                });
                v.gather_position={x,y,z};
                storage.push_back(std::move(copy));
            }
            v.old_momentum={pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),
                            pti.GetAttribs("uz_n").dataPtr()};
            v.midpoint_momentum={pti.GetAttribs(PIdx::ux).dataPtr(),pti.GetAttribs(PIdx::uy).dataPtr(),
                                 pti.GetAttribs(PIdx::uz).dataPtr()};
            views.push_back(v);
        }
        ++s;
    }
    // No parser recomputation, native BC, redistribution, RNG or registry write.
    return MeasureImplicitIonElectricWork(simulation.Geom(lev),fields,species,views,dt,
        gather_contract);
}
} // namespace
IonElectricWorkResult MeasureNativeImplicitIonElectricWork (
    WarpX& simulation, int lev, amrex::Real dt, IonElectricWorkFields const& fields,
    NativeIonWorkContract contract)
{
    return MeasureNativeIonElectricWork(simulation,lev,dt,fields,contract,true);
}
IonElectricWorkResult MeasureNativeReconstructedIonElectricWork (
    WarpX& simulation, int lev, amrex::Real dt, IonElectricWorkFields const& fields,
    NativeIonWorkContract contract)
{
    return MeasureNativeIonElectricWork(simulation,lev,dt,fields,contract,false);
}
} // namespace warpx::thermal
