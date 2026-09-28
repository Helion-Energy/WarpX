/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_KINETIC_THERMAL_SPECIES_DEPOSITION_H_
#define WARPX_KINETIC_THERMAL_SPECIES_DEPOSITION_H_
#include "KineticThermalSpecies.H"
#include "Particles/Deposition/TemperatureDeposition.H"
namespace warpx::thermal {
template <int Order>
void
DepositSpeciesTemperatureShapeN (
    const SpeciesVarianceFields& f, const amrex::MFIter& mfi,
    const GetParticlePosition<PIdx>& position,
    const amrex::ParticleReal* weight, const amrex::ParticleReal* ux,
    const amrex::ParticleReal* uy, const amrex::ParticleReal* uz, long count,
    const amrex::XDim3& inverse, const amrex::XDim3& origin, amrex::Dim3 lower,
    SpeciesVariancePass pass) {
    using namespace warpx::particles::deposition;
    doVarianceDepositionShapeN<Order>(
        position, weight, ux, uy, uz, (*f.temperature[0])[mfi],
        (*f.temperature[1])[mfi], (*f.temperature[2])[mfi], (*f.count[0])[mfi],
        (*f.count[1])[mfi], (*f.count[2])[mfi], (*f.weight[0])[mfi],
        (*f.weight[1])[mfi], (*f.weight[2])[mfi], (*f.variance[0])[mfi],
        (*f.variance[1])[mfi], (*f.variance[2])[mfi], (*f.velocity_sum[0])[mfi],
        (*f.velocity_sum[1])[mfi], (*f.velocity_sum[2])[mfi],
        TemperatureDepositionType::DOUBLE_PASS,
        pass == SpeciesVariancePass::First ? TemperatureDepositionPass::FIRST
                                           : TemperatureDepositionPass::SECOND,
        count, 0., inverse, origin, lower, 1);
}
} // namespace warpx::thermal
#endif
