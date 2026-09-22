/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>

int
main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto& Te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        int const nz = geom.Domain().length(1);
        amrex::Real const kelvin = PhysConst::q_e / PhysConst::kb;
        amrex::Real const pi = std::acos(-1.0);
        amrex::ParmParse pp("thermal_test");
        bool conduction = true;
        pp.query("conduction", conduction);
        hp.m_include_thermal_conduction = conduction;
        int steps = 1;
        amrex::Real duration = 1.e-5;
        pp.query("steps", steps);
        pp.query("duration", duration);
        for (amrex::MFIter mfi(Te); mfi.isValid(); ++mfi)
        {
            auto const te = Te.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { te(i, j, k) = kelvin * (100.0 + 10.0 * std::cos(2.0 * pi * j / nz)); });
        }
        auto amplitude = [&] ()
        {
            auto owner = Te.OwnerMask(geom.periodicity());
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> op;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter mfi(Te); mfi.isValid(); ++mfi)
            {
                auto const te = Te.const_array(mfi);
                auto const own = owner->const_array(mfi);
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                        {
                            amrex::Real const w = own(i, j, k) ? (i == 0 ? 0.125 : i) : 0.0;
                            amrex::Real const c = std::cos(2.0 * pi * j / nz);
                            return {w * te(i, j, k) * c / kelvin, w * c * c,
                                    w * te(i, j, k) / kelvin};
                        });
            }
            auto const val = data.value(op);
            amrex::Real a[3] = {amrex::get<0>(val), amrex::get<1>(val), amrex::get<2>(val)};
            amrex::ParallelDescriptor::ReduceRealSum(a, 3);
            return std::array<amrex::Real, 2>{a[0] / a[1], a[2]};
        };
        auto const initial = amplitude();
        amrex::Real const dt = duration / steps;
        for (int step = 0; step < steps; ++step)
        {
            // Stationary ions and zero electron current isolate the real
            // accepted-step thermal driver from field acceleration.
            for (auto type : {FieldType::current_fp, FieldType::hybrid_current_fp_plasma})
            {
                for (auto* field : sim.m_fields.get_alldirs(type, 0))
                {
                    field->setVal(0.0);
                }
            }
            hp.QDSMCSaveImplicitStepStart(dt, step * dt);
            auto const& rho_n = *sim.m_fields.get(FieldType::hybrid_rho_fp_temp, 0);
            for (int c = 0; c < rho.nComp(); ++c)
            {
                amrex::MultiFab::Copy(rho, rho_n, 0, c, 1, rho.nGrowVect());
            }
            hp.AdvanceElectronEnergyQDSMCTheta(dt, 0.5, true);
            hp.QDSMCFinishImplicitStep(dt, 0.5, (step + 1) * dt);
        }
        auto const final = amplitude();
        amrex::Print() << "THERMAL_RESULT steps=" << steps << " duration=" << duration
                       << " initial_amplitude=" << initial[0] << " final_amplitude=" << final[0]
                       << " mean_change=" << (final[1] - initial[1]) / initial[1] << "\n";
        AMREX_ALWAYS_ASSERT(Te.is_finite());
        if (hp.m_include_thermal_conduction)
        {
            AMREX_ALWAYS_ASSERT(final[0] < 0.999 * initial[0] && final[0] > 0.0);
        }
        else
        {
            AMREX_ALWAYS_ASSERT(std::abs(final[0] - initial[0]) < 1.e-8);
        }
        AMREX_ALWAYS_ASSERT(std::abs(final[1] - initial[1]) < 1.e-8 * initial[1]);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
