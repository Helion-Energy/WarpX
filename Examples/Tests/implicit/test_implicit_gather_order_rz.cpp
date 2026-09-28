/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_Reduce.H>

#include <cmath>

class GatherOrderProbe : public ThetaImplicitHybrid
{
public:
    void Run (WarpX& sim)
    {
        m_WarpX = &sim;
        m_dt = sim.getdt(0);
        m_nlsolver_type = NonlinearSolverType::newton;
        m_max_particle_iterations = 100;
        m_particle_tolerance = 1.e-13;
        PreRHSOp(0.5*m_dt, 0, false);
    }
};

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        AMREX_ALWAYS_ASSERT(WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving);
        AMREX_ALWAYS_ASSERT(!WarpX::galerkin_interpolation);
        auto const fields = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto const aux = sim.m_fields.get_alldirs(FieldType::Efield_aux, 0);
        for (auto* field : fields) { field->setVal(0.0); }
        for (auto* field : sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0)) {
            field->setVal(0.0);
        }
        for (auto* field : aux) { AMREX_ALWAYS_ASSERT(field->ixType().nodeCentered()); }
        auto& ez = *fields[2];
        auto const dz = sim.Geom(0).CellSize(1);
        auto const zmin = sim.Geom(0).ProbLo(1);
        auto const wave = 2.0*std::acos(-1.0)/sim.Geom(0).ProbLength(1);
        amrex::Real const amplitude = 1.e4;
        for (amrex::MFIter mfi(ez); mfi.isValid(); ++mfi) {
            auto const e = ez.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                e(i,j,k) = amplitude*std::sin(wave*(j+0.5)*dz);
            });
        }
        auto& pc = sim.GetPartContainer().GetParticleContainer(0);
        sim.SaveParticlesAtImplicitStepStart();
        GatherOrderProbe probe;
        // Poisoning proves the real residual refreshes its nodal gather data.
        for (auto* field : aux) { field->setVal(-7.e4); }
        probe.Run(sim);
        auto const scale = 0.5*sim.getdt(0)*pc.getCharge()*amplitude/pc.getMass();
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<amrex::Real> data(op);
        using Tuple = decltype(data)::Type;
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            auto const* uz = pti.GetAttribs()[PIdx::uz].data();
            auto const position = GetParticlePosition<PIdx>(pti);
            op.eval(pti.numParticles(), data, [=] AMREX_GPU_DEVICE(int i) -> Tuple {
                amrex::ParticleReal x, y, z;
                position(i,x,y,z);
                return {std::abs(uz[i]/scale - std::sin(wave*(z-zmin)))};
            });
        }
        auto error = amrex::get<0>(data.value(op));
        amrex::ParallelDescriptor::ReduceRealMax(error);
        amrex::Print() << "GATHER_ORDER shape=" << WarpX::nox
                       << " nz=" << sim.Geom(0).Domain().length(1)
                       << " relative_error=" << error
                       << " error_over_h2=" << error/(wave*wave*dz*dz) << "\n";
        // MC uses full order in all directions. Centering and B-spline
        // smoothing both have O(h^2) error: bound each independently rather
        // than checking an implementation-derived expected sample value.
        AMREX_ALWAYS_ASSERT(error < 0.4*wave*wave*dz*dz);
        AMREX_ALWAYS_ASSERT(error > 0.1*wave*wave*dz*dz);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
