/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <AMReX_Reduce.H>
#include <cmath>

// Exercise the actual implicit push/deposit/synchronization path without
// a field solve changing the prescribed zero force during this comparison.
class DepositProbe : public ThetaImplicitHybrid {
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
        for (auto type : {FieldType::Efield_fp, FieldType::Bfield_fp}) {
            for (auto* field : sim.m_fields.get_alldirs(type, 0)) {
                field->setVal(0.0);
            }
        }
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        amrex::MultiFab reference(rho.boxArray(), rho.DistributionMap(), 1,
                                  rho.nGrowVect());
        ablastr::fields::MultiLevelScalarField refs{&reference};
        sim.SaveParticlesAtImplicitStepStart();
        DepositProbe solver;
        for (bool jac : {false, true, false}) {
            solver.Run(sim, jac);
            // Particles now sit at the same midpoint positions used by rho's
            // second slot. This independent deposit owns its volume scaling.
            sim.GetPartContainer().DepositCharge(refs, 0.0);
            sim.SyncRho(refs, {}, {});
            sim.ApplyRhofieldBoundary(0, &reference, PatchType::fine);
            auto const domain =
                amrex::convert(sim.Geom(0).Domain(), rho.ixType());
            int const irmax = domain.bigEnd(0);
            for (int region = 0; region < 3; ++region) {
                amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> op;
                amrex::ReduceData<amrex::Real, amrex::Real> data(op);
                using Tuple = decltype(data)::Type;
                for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                    auto const r = rho.const_array(mfi),
                               ref = reference.const_array(mfi);
                    int const mid = rho.nComp() / 2;
                    op.eval(mfi.validbox(), data,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                                bool const use =
                                    region == 0
                                        ? i == 0
                                        : (region == 1 ? i > 0 && i < irmax
                                                       : i == irmax);
                                return {use ? std::abs(r(i, j, k, mid) -
                                                       ref(i, j, k))
                                            : 0.0,
                                        use ? std::abs(ref(i, j, k)) : 0.0};
                            });
                }
                auto const value = data.value(op);
                amrex::Real v[2] = {amrex::get<0>(value), amrex::get<1>(value)};
                amrex::ParallelDescriptor::ReduceRealMax(v, 2);
                amrex::Print()
                    << "IMPLICIT_RHO jac=" << jac << " region=" << region
                    << " error=" << v[0] << " reference=" << v[1] << "\n";
                AMREX_ALWAYS_ASSERT(
                    v[0] < 1.e-12 * std::max(reference.norminf(), 1.e-100));
            }
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
