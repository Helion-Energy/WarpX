/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

class GatherProbe : public ThetaImplicitHybrid {
  public:
    void
    Run (WarpX& sim, bool jac) {
        m_WarpX = &sim;
        m_dt = sim.getdt(0);
        m_nlsolver_type = NonlinearSolverType::newton;
        PreRHSOp(0.5 * m_dt, 0, jac);
    }
};

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        using warpx::fields::FieldType;
        auto const E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto const B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto const Ea = sim.m_fields.get_alldirs(FieldType::Efield_aux, 0);
        auto const Ba = sim.m_fields.get_alldirs(FieldType::Bfield_aux, 0);
        AMREX_ALWAYS_ASSERT(E[2]->ixType() != Ea[2]->ixType());

        bool endpoint = false;
        amrex::ParmParse("gather_test").query("endpoint", endpoint);
        if (endpoint) {
            // No full diagnostic is enabled. The accepted endpoint must already
            // be published before any diagnostic can refresh auxiliary fields.
            sim.Evolve(2);
            amrex::Real signal = 0.0;
            std::array<std::unique_ptr<amrex::MultiFab>, 6> saved;
            std::array<amrex::MultiFab*, 6> aux{Ea[0], Ea[1], Ea[2],
                                                Ba[0], Ba[1], Ba[2]};
            for (int d = 0; d < 6; ++d) {
                saved[d] = std::make_unique<amrex::MultiFab>(
                    aux[d]->boxArray(), aux[d]->DistributionMap(),
                    aux[d]->nComp(), 0);
                amrex::MultiFab::Copy(*saved[d], *aux[d], 0, 0, aux[d]->nComp(),
                                      0);
            }
            sim.UpdateAuxiliaryData();
            for (int d = 0; d < 6; ++d) {
                amrex::Real const scale = aux[d]->norminf();
                signal = std::max(signal, scale);
                amrex::MultiFab::Subtract(*saved[d], *aux[d], 0, 0,
                                          aux[d]->nComp(), 0);
                AMREX_ALWAYS_ASSERT(saved[d]->norminf() <
                                    1.e-12 * std::max(scale, 1.e-20));
            }
            AMREX_ALWAYS_ASSERT(signal > 1.e-6);
            amrex::Print() << "IMPLICIT_GATHER endpoint PASS\n";
        } else {
            for (int d = 0; d < 3; ++d) {
                E[d]->setVal(0.0);
                B[d]->setVal(0.0);
            }
            auto& pc = sim.GetPartContainer().GetParticleContainer(0);
            sim.SaveParticlesAtImplicitStepStart();
            GatherProbe solver;
            for (int trial = 0; trial < 3; ++trial) {
                amrex::Real const amplitude = trial == 1 ? 2.e4 : 1.e4;
                E[2]->setVal(amplitude);
                for (int d = 0; d < 3; ++d) {
                    Ea[d]->setVal(-7.e4);
                    Ba[d]->setVal(0.0);
                }
                solver.Run(sim, trial == 1);
                // Uniform electric force: PreRHSOp leaves the exact midpoint
                // proper velocity, independent of the previous gather buffers.
                amrex::Real const expected = 0.5 * sim.getdt(0) *
                                             pc.getCharge() * amplitude /
                                             pc.getMass();
                amrex::ReduceOps<amrex::ReduceOpMax> op;
                amrex::ReduceData<amrex::Real> data(op);
                using Tuple = decltype(data)::Type;
                for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                    auto const* uz = pti.GetAttribs()[PIdx::uz].data();
                    op.eval(pti.numParticles(), data,
                            [=] AMREX_GPU_DEVICE(int i) -> Tuple {
                                return {std::abs(uz[i] - expected)};
                            });
                }
                amrex::Real error = amrex::get<0>(data.value(op));
                amrex::ParallelDescriptor::ReduceRealMax(error);
                amrex::Print()
                    << "IMPLICIT_GATHER trial=" << trial
                    << " expected=" << expected << " error=" << error << "\n";
                AMREX_ALWAYS_ASSERT(error < 1.e-10 * std::abs(expected));
            }
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
