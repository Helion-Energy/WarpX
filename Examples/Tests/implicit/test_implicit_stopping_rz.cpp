/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <cmath>
#include <limits>

int
main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        te.setVal(100.0 * PhysConst::q_e / PhysConst::kb);
        for (auto* j : sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0))
        {
            j->setVal(0.0);
        }
        // Poison registered midpoint inputs: endpoint coefficients must be
        // freshly deposited scratch, without relabeling these solver arrays.
        auto& registered_rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        registered_rho.setVal(-17.0);
        for (auto* j : sim.m_fields.get_alldirs(FieldType::current_fp, 0))
        {
            j->setVal(12345.0);
        }
        // A nonzero manufactured plasma current makes velocity gather active.
        // Previously stored physical ghosts must not affect its next filtered
        // value. Opposite poison signs expose filtering before image filling.
        auto const plasma_current =
            sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        auto const velocity = sim.m_fields.get_alldirs("Ve_fp", 0);
        std::array<amrex::MultiFab, 3> reference;
        for (int d = 0; d < 3; ++d)
        {
            for (amrex::MFIter mfi(*plasma_current[d]); mfi.isValid(); ++mfi)
            {
                auto const current = plasma_current[d]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   { current(i, j, k) = (d + 1.0) * (1.0 + 0.1 * i + 0.01 * j); });
            }
            velocity[d]->setVal(1.e20);
        }
        hp.PrepareImplicitStopping(sim.getdt(0));
        for (int d = 0; d < 3; ++d)
        {
            reference[d].define(velocity[d]->boxArray(), velocity[d]->DistributionMap(),
                                velocity[d]->nComp(), velocity[d]->nGrowVect());
            amrex::MultiFab::Copy(reference[d], *velocity[d], 0, 0, velocity[d]->nComp(),
                                  velocity[d]->nGrowVect());
            AMREX_ALWAYS_ASSERT(velocity[d]->norminf() > 1.e-3);
        }
        hp.FinishImplicitStopping();
        for (auto* v : velocity)
        {
            v->setVal(-1.e20);
        }
        hp.PrepareImplicitStopping(sim.getdt(0));
        for (int d = 0; d < 3; ++d)
        {
            amrex::MultiFab::Subtract(reference[d], *velocity[d], 0, 0, velocity[d]->nComp(),
                                      velocity[d]->nGrowVect());
            auto const error =
                reference[d].norminf(0, reference[d].nComp(), reference[d].nGrowVect());
            amrex::Print() << "VE_POISON_RESULT component=" << d << " error=" << error
                           << " scale=" << velocity[d]->norminf() << "\n";
            // Fresh GPU charge deposition changes atomic summation order;
            // require roundoff-scale agreement, not bitwise reproducibility.
            auto const roundoff =
                256 * std::numeric_limits<amrex::Real>::epsilon() *
                velocity[d]->norminf(0, velocity[d]->nComp(), velocity[d]->nGrowVect());
            AMREX_ALWAYS_ASSERT(error <= roundoff);
        }
        hp.FinishImplicitStopping();
        for (auto* current : plasma_current)
        {
            current->setVal(0.0);
        }
        hp.PrepareImplicitStopping(sim.getdt(0));
        auto const& rho = hp.GetStoppingChargeDensity(0);
        auto energy = [&] ()
        {
            auto const e = hp.QDSMCClassEnergy(0, &rho);
            return e[0] + e[1];
        };
        auto kinetic = [&] ()
        {
            amrex::Real e = 0.0;
            for (auto const* name : {"alpha_axis", "alpha_seam", "alpha_wall", "alpha_zface"})
            {
                e += sim.GetPartContainer().GetParticleContainerFromName(name).sumParticleEnergy();
            }
            return e;
        };
        // Independent constant-density integral: both periodic and physical
        // axial faces must partition exactly the analytic cylinder volume.
        amrex::MultiFab measure(te.boxArray(), te.DistributionMap(), 1, 0);
        measure.setVal(1.0);
        auto const& geom = sim.Geom(0);
        double const analytic_volume =
            MathConst::pi * (geom.ProbHi(0) * geom.ProbHi(0) - geom.ProbLo(0) * geom.ProbLo(0)) *
            (geom.ProbHi(1) - geom.ProbLo(1));
        AMREX_ALWAYS_ASSERT(std::abs(hp.EnergyVolumeIntegral(measure, 0, 0) - analytic_volume) <
                            1.e-13 * analytic_volume);
        auto const u0 = energy();
        auto const k0 = kinetic();
        sim.GetPartContainer().doCollisions(0, sim.getdt(0), sim.getdt(0));
        auto const k1 = kinetic();
        auto& staged = hp.GetFastIonHeatingStaging(0);
        amrex::MultiFab summed(staged.boxArray(), staged.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(summed, staged, 0, 0, 1, 0);
        summed.SumBoundary(sim.Geom(0).periodicity());
        auto const deposited = hp.EnergyVolumeIntegral(summed, 0, 0);
        hp.FinishImplicitStopping();
        auto const u1 = energy();
        auto const dx = sim.Geom(0).CellSizeArray();
        auto const wall = hp.GetQdsmcWallTally(0, 1) * dx[0] * dx[1];
        auto const loss = k0 - k1;
        amrex::Print() << "STOPPING_RESULT loss=" << loss << " staged=" << deposited
                       << " electron_gain=" << u1 - u0 << " wall=" << wall
                       << " balance=" << (u1 - u0 - wall - loss) / loss << "\n";
        AMREX_ALWAYS_ASSERT(loss > 0.0);
        AMREX_ALWAYS_ASSERT(std::abs(deposited - loss) < 1.e-9 * loss);
        AMREX_ALWAYS_ASSERT(std::abs(u1 - u0 - wall - loss) < 2.e-7 * loss);
        AMREX_ALWAYS_ASSERT(staged.norminf() == 0.0);
        // Check every shared valid wall copy, not merely the unique tally owner.
        auto const radial_wall = geom.Domain().bigEnd(0) + 1;
        auto const wall_temperature = 100.0 * PhysConst::q_e / PhysConst::kb;
        for (amrex::MFIter mfi(measure); mfi.isValid(); ++mfi)
        {
            auto const error = measure.array(mfi);
            auto const temperature = te.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   error(i, j, k) =
                                       i == radial_wall
                                           ? std::abs(temperature(i, j, k) - wall_temperature)
                                           : 0.0;
                               });
        }
        AMREX_ALWAYS_ASSERT(measure.norminf() < 1.e-8);
        AMREX_ALWAYS_ASSERT(registered_rho.min(0) == -17.0 && registered_rho.max(0) == -17.0);
        // A second consume must not apply the energy twice.
        hp.QDSMCApplyFastIonHeating(0, &rho);
        AMREX_ALWAYS_ASSERT(energy() == u1);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
