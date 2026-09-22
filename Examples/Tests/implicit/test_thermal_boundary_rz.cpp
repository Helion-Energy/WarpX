/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
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
        auto& te = *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        auto const& geom = sim.Geom(0);
        auto const dx = geom.CellSizeArray();
        int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
        amrex::Real const kelvin = PhysConst::q_e / PhysConst::kb;
        rho.setVal(2.e18 * PhysConst::q_e); // Above-floor capacity throughout the closed domain.
        auto energy = [&] ()
        {
            auto const e = hp.QDSMCClassEnergy(0, &rho);
            return e[0] + e[1];
        };
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            for (int side = 0; side < 2; ++side)
            {
                hp.m_cond_bc[d][side] = 0;
            }
        }
        // Nonzero gradients at physical faces expose incorrect half-cell
        // divergence normalization; the closed domain must retain its heat.
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi)
        {
            auto const t = te.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::Real const r = amrex::Real(i) / nr,
                                                     z = amrex::Real(j) / nz;
                                   t(i, j, k) = kelvin * (100.0 + 10.0 * r * r + 5.0 * z);
                               });
        }
        auto const closed_before = energy();
        hp.QdsmcConductionOnceFDAtState(0, 1.e-5, false, rho, 0.0);
        auto const closed_after = energy();
        auto const closed_error = std::abs(closed_after - closed_before) / closed_before;
        AMREX_ALWAYS_ASSERT(closed_error < 1.e-11);
        AMREX_ALWAYS_ASSERT(te.is_finite());

        // Known heat flux through the cylinder wall and, when nonperiodic,
        // both end disks. Check absolute heat against analytic surface areas.
        te.setVal(100.0 * kelvin);
        hp.m_cond_bc[0][1] = 2;
        hp.m_cond_bc_q[0][1] = 1.e4;
        amrex::Real area = 2.0 * MathConst::pi * geom.ProbHi(0) * (geom.ProbHi(1) - geom.ProbLo(1));
        if (!geom.isPeriodic(1))
        {
            for (int side = 0; side < 2; ++side)
            {
                hp.m_cond_bc[1][side] = 2;
                hp.m_cond_bc_q[1][side] = 1.e4;
            }
            area += 2.0 * MathConst::pi * geom.ProbHi(0) * geom.ProbHi(0);
        }
        auto const flux_before = energy();
        hp.ApplyQdsmcConductionWallBCs(0, 1.e-6, te, rho, 0.0);
        auto const injected = energy() - flux_before;
        auto const expected = 1.e4 * 1.e-6 * area;
        AMREX_ALWAYS_ASSERT(std::abs(injected - expected) < 1.e-9 * expected);

        // Constant-coefficient row ODE: compare the exact relaxation rate,
        // physical V/A capacity, curvature and every shared wall copy.
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            if (geom.isPeriodic(d))
            {
                continue;
            }
            for (amrex::Real const duration : {1.e-10, 1.e-5})
            {
                te.setVal(100.0 * kelvin);
                auto const leg_before = energy();
                auto const tally_before = hp.GetQdsmcLegTally(d, 1);
                hp.ApplyQdsmcConductionLegBC(0, d, 1, duration, te, rho, 0.0);
                auto const leg_change = energy() - leg_before;
                auto const leg_tally = (hp.GetQdsmcLegTally(d, 1) - tally_before) * dx[0] * dx[1];
                amrex::Print() << "LEG_BUDGET_RESULT dim=" << d << " dt=" << duration
                               << " before=" << leg_before << " heat=" << leg_change
                               << " tally=" << leg_tally << "\n";
                AMREX_ALWAYS_ASSERT(leg_change < 0.0);
                AMREX_ALWAYS_ASSERT(std::abs(leg_change - leg_tally) < 1.e-10 * leg_before);
                amrex::Real const radius = geom.ProbHi(0);
                amrex::Real const area_ratio = d == 0 ? (radius - 0.5 * dx[0]) / radius : 1.0;
                amrex::Real const area_over_volume =
                    d == 0 ? radius / (radius * dx[0] / 2.0 - dx[0] * dx[0] / 8.0) : 2.0 / dx[d];
                amrex::Real const gi = area_ratio / dx[d], gl = 1.0 / hp.m_cond_leg_length;
                amrex::Real const target =
                    kelvin * (gi * 100.0 + gl * hp.m_cond_leg_Te_wall) / (gi + gl);
                amrex::Real const exact =
                    target + (100.0 * kelvin - target) *
                                 std::exp(-duration * 100.0 * (gi + gl) * area_over_volume);
                int const wall = geom.Domain().bigEnd(d) + 1;
                amrex::MultiFab error(te.boxArray(), te.DistributionMap(), 1, 0);
                for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi)
                {
                    auto const out = error.array(mfi);
                    auto const t = te.const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(),
                                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                       {
                                           int const index[3] = {i, j, k};
                                           out(i, j, k) = index[d] == wall
                                                              ? std::abs(t(i, j, k) - exact)
                                                              : 0.0;
                                       });
                }
                AMREX_ALWAYS_ASSERT(error.norminf() < 1.e-8);
                amrex::Print() << "LEG_RESULT dim=" << d << " dt=" << duration
                               << " temperature_error=" << error.norminf() << " heat=" << leg_change
                               << " tally=" << leg_tally << "\n";
            }
        }
        amrex::Print() << "BOUNDARY_THERMAL_RESULT closed_relative=" << closed_error
                       << " flux=" << injected << " expected=" << expected << "\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
