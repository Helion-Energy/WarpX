/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"

#include <AMReX_Reduce.H>
#include <cmath>

int main (int argc, char **argv)
{
    warpx::initialization::initialize_external_libraries (argc, argv);
    {
        auto &sim = WarpX::GetInstance ();
        sim.InitData ();
        using warpx::fields::FieldType;
        auto &model = *sim.get_pointer_HybridPICModel ();
        auto const &geom = sim.Geom (0);
        auto const dx = geom.CellSizeArray ();
        auto const lo = geom.ProbLoArray ();
        auto const domain = geom.Domain ();
        constexpr int axial = AMREX_SPACEDIM - 1;
        amrex::Real const length = geom.ProbHi (axial) - geom.ProbLo (axial);
        auto const E = sim.m_fields.get_alldirs (FieldType::Efield_fp, 0);
        auto J = sim.m_fields.get_alldirs (FieldType::hybrid_current_fp_plasma, 0);
        auto const Ji = sim.m_fields.get_alldirs (FieldType::current_fp, 0);
        auto const B = sim.m_fields.get_alldirs (FieldType::Bfield_fp, 0);
        auto &rho = *sim.m_fields.get (FieldType::rho_fp, 0);
        auto &Pe = *sim.m_fields.get (FieldType::hybrid_electron_pressure_fp, 0);
        auto &Te = *sim.m_fields.get (FieldType::hybrid_electron_temperature_fp, 0);
        rho.setVal (1000.);
        model.m_add_external_fields = false;
        model.m_holmstrom_vacuum_region = false;
        model.m_include_electron_pressure_term = true;
        model.m_include_biermann_battery = true;
        model.m_end_region = {};
        for (int d = 0; d < 3; ++d)
        {
            J[d]->setVal (2. + d);
            Ji[d]->setVal (0.);
            B[d]->setVal (d == 2 ? 2. : 0.);
        }
        // Nonzero Hall and pressure drive plus a separately known inertia term.
        for (amrex::MFIter mfi (Pe); mfi.isValid (); ++mfi)
        {
            auto const p = Pe.array (mfi);
            amrex::ParallelFor (mfi.fabbox (),
                                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                                {
                                    int const iv[3]{i, j, k};
                                    p (i, j, k) = 100. + 10. * (lo[0] + i * dx[0]) +
                                                  20. * (lo[axial] + iv[axial] * dx[axial]);
                                });
        }
        sim.m_fields.get ("hybrid_E_inertial_nodal", 0)->setVal (7.);
        auto solve = [&] (bool resistivity)
        {
            sim.get_pointer_fdtd_solver_fp (0)->HybridPICSolveE (
                E, J, Ji, B, rho, Pe, sim.GetEBUpdateEFlag ()[0], 0, &model, true, resistivity);
        };
        solve (false);
        amrex::Array<amrex::MultiFab, 3> reference;
        for (int d = 0; d < 3; ++d)
        {
            reference[d].define (E[d]->boxArray (), E[d]->DistributionMap (), 1, 0);
            amrex::MultiFab::Copy (reference[d], *E[d], 0, 0, 1, 0);
        }
        model.m_end_region.width = {.17 * length, .27 * length};
        model.m_end_region.rolloff = {.025 * length, .04 * length};
        // Asymmetric end layers and several decompositions catch a tile-local
        // coordinate or a nodal/cell-center mixup. Expected values are computed
        // independently of the production Weight helper.
        for (bool physical_relaxation : {false, true})
        {
            model.m_include_temperature_relaxation = physical_relaxation;
            for (int mode = 0; mode < 4; ++mode)
            {
                bool const hall = (mode & 1) != 0;
                bool const eta = (mode & 2) != 0;
                model.m_end_region.holmstrom = hall;
                model.m_end_region.resistivity = eta ? .125 : 0.;
                for (bool resistivity : {false, true})
                {
                    solve (resistivity);
                    amrex::Real error = 0.;
                    for (int d = 0; d < 3; ++d)
                    {
                        int const nodal = E[d]->ixType ().nodeCentered (axial);
                        amrex::ReduceOps<amrex::ReduceOpMax> op;
                        amrex::ReduceData<amrex::Real> data (op);
                        using Tuple = decltype (data)::Type;
                        for (amrex::MFIter mfi (*E[d]); mfi.isValid (); ++mfi)
                        {
                            auto const e = E[d]->const_array (mfi),
                                       base = reference[d].const_array (mfi);
                            op.eval (mfi.validbox (), data,
                                     [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
                                     {
                                         if (i < 2 || i > domain.bigEnd (0) - 2)
                                         {
                                             return {0.};
                                         }
#if defined(WARPX_DIM_3D)
                                         if (j < 2 || j > domain.bigEnd (1) - 2)
                                         {
                                             return {0.};
                                         }
#endif
                                         int const iv[3]{i, j, k};
                                         amrex::Real const z =
                                             (iv[axial] - domain.smallEnd (axial) +
                                              .5 * (1 - nodal)) *
                                             dx[axial] / length;
                                         amrex::Real const w =
                                             amrex::max (.5 * (1. + std::tanh ((.17 - z) / .025)),
                                                         .5 * (1. + std::tanh ((z - .73) / .04)));
                                         amrex::Real const expected =
                                             (base (i, j, k) - 7.) * (hall ? 1. - w : 1.) + 7. +
                                             (eta && resistivity ? .125 * w * (2. + d) : 0.);
                                         return {std::abs (e (i, j, k) - expected)};
                                     });
                        }
                        error = std::max (error, amrex::get<0> (data.value (op)));
                    }
                    amrex::ParallelDescriptor::ReduceRealMax (error);
                    amrex::Print () << "END_OHM mode=" << mode << " eta_path=" << resistivity
                                    << " error=" << error << "\n";
                    AMREX_ALWAYS_ASSERT (error < 2.e-12);
                }
            }
        }
        AMREX_ALWAYS_ASSERT (model.HasResistivity ());
        // Independent projection check: divergence of E_L must equal the
        // analytic divergence of the same end-weighted pressure drive. Remove
        // inertia here to resolve the small pressure term without cancellation.
        sim.m_fields.get ("hybrid_E_inertial_nodal", 0)->setVal (0.);
        model.ComputeDarwinELong (sim.m_fields.get_mr_levels (FieldType::rho_fp, 0), 0., true);
        auto const longitudinal = sim.m_fields.get_alldirs ("hybrid_E_long_fp", 0);
        amrex::MultiFab divergence (Pe.boxArray (), Pe.DistributionMap (), 1, 0);
        sim.get_pointer_fdtd_solver_fp (0)->ComputeDivE (longitudinal, divergence);
        amrex::ReduceOps<amrex::ReduceOpMax> projection_op;
        amrex::ReduceData<amrex::Real> projection_data (projection_op);
        using ProjectionTuple = decltype (projection_data)::Type;
        for (amrex::MFIter mfi (divergence); mfi.isValid (); ++mfi)
        {
            auto const div = divergence.const_array (mfi);
            projection_op.eval (
                mfi.validbox (), projection_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ProjectionTuple
                {
                    int const iv[3]{i, j, k};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        if (iv[d] < domain.smallEnd (d) + 2 || iv[d] > domain.bigEnd (d) - 2)
                        {
                            return {0.};
                        }
                    }
                    amrex::Real const z =
                        (iv[axial] - domain.smallEnd (axial)) * dx[axial] / length;
                    amrex::Real const half = .5 * dx[axial] / length;
                    amrex::Real const wp =
                        amrex::max (.5 * (1. + std::tanh ((.17 - z - half) / .025)),
                                    .5 * (1. + std::tanh ((z + half - .73) / .04)));
                    amrex::Real const wm =
                        amrex::max (.5 * (1. + std::tanh ((.17 - z + half) / .025)),
                                    .5 * (1. + std::tanh ((z - half - .73) / .04)));
                    amrex::Real expected = .02 * (wp - wm) / dx[axial];
#if defined(WARPX_DIM_RZ)
                    amrex::Real const w = amrex::max (.5 * (1. + std::tanh ((.17 - z) / .025)),
                                                      .5 * (1. + std::tanh ((z - .73) / .04)));
                    expected -= .01 * (1. - w) / (lo[0] + i * dx[0]);
#endif
                    return {std::abs (div (i, j, k) - expected)};
                });
        }
        amrex::Real projection_error = amrex::get<0> (projection_data.value (projection_op));
        amrex::ParallelDescriptor::ReduceRealMax (projection_error);
        amrex::Print () << "END_LONGITUDINAL error=" << projection_error << "\n";
        AMREX_ALWAYS_ASSERT (projection_error < 1.e-8);

        // Single-species Joule heating must receive the same added eta even
        // when the deck supplies a distinct zero physical-heating parser.
        sim.m_fields.get ("rho_fp_ions", 0)->setVal (1000.);
        sim.m_fields.get_alldirs ("current_fp_ions", 0)[0]->setVal (0.);
        sim.m_fields.get_alldirs ("current_fp_ions", 0)[1]->setVal (0.);
        sim.m_fields.get_alldirs ("current_fp_ions", 0)[2]->setVal (0.);
        model.m_joule_heating_taper = false;
        model.m_qdsmc_source_taper_n = -1.;
        Te.setVal (1000.);
        amrex::Real const dt = .1;
        amrex::Real const factor =
            dt * (model.m_gamma - 1.) * .125 * 29. * PhysConst::q_e / (1000. * PhysConst::kb);
        model.QDSMCAddJouleHeating (0, dt, rho, nullptr);
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<amrex::Real> data (op);
        using Tuple = decltype (data)::Type;
        for (amrex::MFIter mfi (Te); mfi.isValid (); ++mfi)
        {
            auto const te = Te.const_array (mfi);
            op.eval (mfi.validbox (), data,
                     [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
                     {
                         int const iv[3]{i, j, k};
                         amrex::Real const z =
                             (iv[axial] - domain.smallEnd (axial)) * dx[axial] / length;
                         amrex::Real const w = amrex::max (.5 * (1. + std::tanh ((.17 - z) / .025)),
                                                           .5 * (1. + std::tanh ((z - .73) / .04)));
                         return {std::abs (te (i, j, k) - (1000. + factor * w))};
                     });
        }
        amrex::Real heat_error = amrex::get<0> (data.value (op));
        amrex::ParallelDescriptor::ReduceRealMax (heat_error);
        amrex::Print () << "END_JOULE error=" << heat_error << "\n";
        AMREX_ALWAYS_ASSERT (heat_error < 2.e-10);
        amrex::Print () << "HYBRID_END_REGION_PASS\n";
        WarpX::Finalize ();
    }
    warpx::initialization::finalize_external_libraries ();
}
