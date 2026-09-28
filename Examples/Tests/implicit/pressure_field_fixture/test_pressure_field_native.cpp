/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "BoundaryConditions/WarpX_PEC.H"
#include "test_pressure_stagger.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/FrozenPressureFieldCoupling.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <string>

using namespace warpx::thermal;
using amrex::Real;
using MF = amrex::MultiFab;
using Owned = std::array<std::unique_ptr<MF>, 3>;

namespace {
MomentBoundary
boundary (FieldBoundaryType b) {
    if (b == FieldBoundaryType::Periodic) {
        return MomentBoundary::Periodic;
    }
    if (b == FieldBoundaryType::PEC) {
        return MomentBoundary::PEC;
    }
    AMREX_ALWAYS_ASSERT(b == FieldBoundaryType::PMC);
    return MomentBoundary::PMC;
}
amrex::Array<MF*, 3>
view (Owned& f) {
    return {f[0].get(), f[1].get(), f[2].get()};
}
FrozenPressureFieldCoupling::ConstFieldView
cview (const Owned& f) {
    return {f[0].get(), f[1].get(), f[2].get()};
}
Real
maxnorm (const Owned& f) {
    Real x = 0.;
    for (int c = 0; c < 3; ++c) {
        x = std::max(x, f[c]->norm0());
    }
    return x;
}

void
run (WarpX& sim) {
    using warpx::fields::FieldType;
    auto& model = *sim.get_pointer_HybridPICModel();
    model.m_include_electron_inertia = false;
    model.m_include_hyper_resistivity_term = false;
    model.m_visc_in_ohms_law = false;
    model.m_add_external_fields = false;
    model.m_esolve_curlcurl = false;
    model.m_esolve_tensor = false;
    model.m_pec_conductor_wall_rows = false;
    model.m_include_temperature_relaxation = false;
    auto const& appgeom = sim.Geom(0);
    int periodic[AMREX_SPACEDIM] = {
        AMREX_D_DECL(appgeom.isPeriodic(0), appgeom.isPeriodic(1), 0)};
    amrex::RealBox rb(appgeom.ProbLo(), appgeom.ProbHi());
    amrex::Geometry geom(appgeom.Domain(), &rb, 1, periodic);
    auto const er = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const br = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    TestPressureResponseStagger({er[0]->ixType().toIntVect(),
        er[1]->ixType().toIntVect(), er[2]->ixType().toIntVect()});
    auto const cells =
        amrex::convert(er[0]->boxArray(), amrex::IntVect::TheZeroVector());
    auto const& dm = er[0]->DistributionMap();
    bool unequal_density = false, recovery_rows = false, expect_empty_rank = false;
    amrex::ParmParse pressure_pp("pressure_oracle");
    pressure_pp.query("unequal_density_components", unequal_density);
    pressure_pp.query("recovery_rows", recovery_rows);
    pressure_pp.query("expect_empty_rank", expect_empty_rank);
    // Mirror the runtime register: component zero is accepted old charge,
    // component one is the live midpoint charge consumed by native Ohm.
    MF density_register(amrex::convert(cells, amrex::IntVect::TheNodeVector()),
                        dm, 2, 2);
    MF rho(density_register, amrex::make_alias, 1, 1);
    auto const dx = geom.CellSizeArray();
    auto const lo = geom.ProbLoArray();
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const raw = density_register.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                              int k) {
            Real const r = i * dx[0], z = lo[1] + j * dx[1];
            raw(i, j, k, 1) = PhysConst::q_e * 8.e17 *
                              (1. + .65 * std::cos(8. * r) *
                                        std::cos(2. * 3.141592653589793 * z));
            raw(i, j, k, 0) = unequal_density
                                  ? PhysConst::q_e * 2.e17 *
                                        (1. + .2 * std::cos(3. * r + 4. * z))
                                  : raw(i, j, k, 1);
        });
    }
    ThermalMomentOptions mo;
    mo.number_density_floor = 8.e17;
    mo.nodal_ghosts = 2;
    mo.boundary[0] = {MomentBoundary::Axis,
                      boundary(WarpX::field_boundary_hi[0])};
    mo.boundary[1] = {boundary(WarpX::field_boundary_lo[1]),
                      boundary(WarpX::field_boundary_hi[1])};
    FrozenPressureFieldOptions opts;
    opts.gamma = mo.gamma;
    opts.boundary = mo.boundary;
    FrozenPressureFieldCoupling coupling(geom, cells, dm, opts);
    KineticThermalMoments moments(geom, cells, dm, mo);
    auto const* pedestal = model.DensityPedestal(0);
    AMREX_ALWAYS_ASSERT(pedestal);
    AMREX_ALWAYS_ASSERT(moments.Evaluate({rho, 0, pedestal}));
    MF u(cells, dm, 1, 0), direction(cells, dm, 1, 0), trial(cells, dm, 1, 0);
    for (amrex::MFIter mfi(u); mfi.isValid(); ++mfi) {
        auto const out = u.array(mfi);
        auto const du = direction.array(mfi);
        auto const n = moments.NumberDensity().const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real const r = (i + .5) * dx[0], z = lo[1] + (j + .5) * dx[1];
            out(i, j, k) = PhysConst::kb * n(i, j, k) * 2.e4 / (5. / 3. - 1.);
            du(i, j, k) = .013 * (std::cos(8. * r) +
                                  .7 * std::sin(2. * 3.141592653589793 * z));
        });
    }
    MF plus(rho.boxArray(), dm, 1, 2), minus(rho.boxArray(), dm, 1, 2);
    Real const eps = .125;
    MF::LinComb(trial, 1., u, 0, eps, direction, 0, 0, 1, 0);
    AMREX_ALWAYS_ASSERT(moments.Evaluate({rho, 0, pedestal, &trial}));
    MF::Copy(plus, moments.OhmPressure(), 0, 0, 1, 2);
    MF::LinComb(trial, 1., u, 0, -eps, direction, 0, 0, 1, 0);
    AMREX_ALWAYS_ASSERT(moments.Evaluate({rho, 0, pedestal, &trial}));
    MF::Copy(minus, moments.OhmPressure(), 0, 0, 1, 2);
    Owned B, J, Ji, Eplus, Eminus, weights, predicted, oracle, rowmask,
        old_weights, old_predicted;
    for (int c = 0; c < 3; ++c) {
        B[c] = std::make_unique<MF>(br[c]->boxArray(), dm, 1, 2);
        B[c]->setVal(0.);
        for (auto* fields : {&J, &Ji, &Eplus, &Eminus, &weights, &predicted,
                             &oracle, &rowmask, &old_weights, &old_predicted}) {
            (*fields)[c] = std::make_unique<MF>(er[c]->boxArray(), dm, 1, 2);
            (*fields)[c]->setVal(0.);
        }
    }
    auto apply_boundary = [&] (Owned& f) {
        using ablastr::utils::enums::PatchType;
        amrex::Vector<amrex::IntVect> const ratios;
        PEC::ApplyPECtoEfield(view(f), WarpX::field_boundary_lo,
                              WarpX::field_boundary_hi, FieldBoundaryType::PEC,
                              amrex::IntVect(0), appgeom, 0, PatchType::fine,
                              ratios);
        PEC::ApplyPECtoBfield(view(f), WarpX::field_boundary_lo,
                              WarpX::field_boundary_hi, FieldBoundaryType::PMC,
                              amrex::IntVect(0), appgeom, 0, PatchType::fine,
                              ratios);
    };
    for (int c = 0; c < 3; ++c) {
        rowmask[c]->setVal(1.);
    }
    apply_boundary(rowmask);
    for (std::string const name :
         {"ungated", "hard", "smooth", "confined", "ends", "masked",
          "boundary_rows", "disabled", "conductor_raw"}) {
        if (name == "ends" && geom.isPeriodic(1)) {
            continue;
        }
        model.m_n_floor = 8.e17;
        model.m_n_floor_smooth_width = name == "ungated" ? 0. : .15;
        model.m_holmstrom_vacuum_region =
            name == "hard" || name == "smooth" || name == "confined";
        model.m_holmstrom_transition_width = name == "hard" ? 0. : .2;
        model.m_holmstrom_axis_radius = name == "confined" ? .1 : 0.;
        model.m_holmstrom_axis_rolloff = .025;
        model.m_include_electron_pressure_term = name != "disabled";
        model.m_pec_conductor_wall_rows = name == "conductor_raw";
        // The raw gate is tested here without invoking the known mutating
        // conductor-wall wrapper. New runtime modes guard that wrapper option.
        model.m_end_region.holmstrom = name == "ends";
        model.m_end_region.width = {.125, .125};
        model.m_end_region.rolloff = {.02, .02};
        if (name == "conductor_raw") {
            // Gate negative raw charge while preserving a positive pedestal
            // denominator. These are coefficient-only oracle values; thermal
            // capacity remains the preceding physically positive frozen
            // context.
            rho.mult(-.1, 0, 1, 2);
        }
        auto const end = model.EndRegion(0);
        PressureResponseParameters p;
        p.charge_floor = PhysConst::q_e * model.m_n_floor;
        p.floor_width = model.m_n_floor_smooth_width * p.charge_floor;
        p.holmstrom_width = model.m_holmstrom_transition_width * p.charge_floor;
        p.axis_radius = model.m_holmstrom_axis_radius;
        p.axis_rolloff = model.m_holmstrom_axis_rolloff;
        p.holmstrom = model.m_holmstrom_vacuum_region;
        p.pressure_enabled = model.m_include_electron_pressure_term;
        p.conductor_raw_gate = model.m_pec_conductor_wall_rows;
        bool const mask = name == "masked";
        bool const boundary_rows = name == "boundary_rows";
        for (int c = 0; c < 3; ++c) {
            auto const iv = er[c]->ixType().toIntVect();
            auto const stagger = PressureResponseStagger(iv);
            amrex::GpuArray<int, 3> const nodal{1,1,1}, ratio{1,1,1};
            for (amrex::MFIter mfi(*weights[c]); mfi.isValid(); ++mfi) {
                auto const w = weights[c]->array(mfi);
                auto const old_w = old_weights[c]->array(mfi);
                auto const old_rho = density_register.const_array(mfi);
                auto const raw = rho.const_array(mfi);
                auto const ped = pedestal->const_array(mfi);
                auto const rows = rowmask[c]->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    int const di = c == 0, dj = c == 2;
                    // Same production helper and three-index Interp as DTA.
                    // The independent native Ohm secant below is the oracle.
                    Real const rawface = ablastr::coarsen::sample::Interp(
                        raw, nodal, stagger, ratio, i, j, k, 0);
                    Real const pedface = ablastr::coarsen::sample::Interp(
                        ped, nodal, stagger, ratio, i, j, k, 0);
                    Real response =
                        end.holmstrom ? 1. - end.Weight(i, j, k, stagger) : 1.;
                    if (mask) {
                        response *= j % 5 == 0 ? 0. : (j % 5 == 1 ? .3 : 1.);
                    }
                    if (boundary_rows) {
                        response *= rows(i, j, k);
                    }
                    w(i, j, k) = PressureResponseWeight(
                        rawface, pedface, (i + (c == 0 ? .5 : 0.)) * dx[0], p,
                        response);
                    // Omission twin: reproduce the reviewed wrong-register-
                    // component coefficient with exactly the same trial Pe.
                    Real const oldface = .5 * (old_rho(i, j, k, 0) +
                                               old_rho(i + di, j + dj, k, 0));
                    old_w(i, j, k) = PressureResponseWeight(
                        oldface, pedface, (i + (c == 0 ? .5 : 0.)) * dx[0], p,
                        response);
                });
            }
            weights[c]->OverrideSync(geom.periodicity());
        }
        AMREX_ALWAYS_ASSERT(coupling.Freeze(moments.NumberDensity(),
                                            moments.NodalNumberDensity(),
                                            cview(weights)));
        coupling.Apply(view(predicted), direction);
        auto current_view = view(J);
        sim.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(
            view(Eplus), current_view, view(Ji), view(B), rho, plus,
            sim.GetEBUpdateEFlag()[0], 0, &model, false, false);
        sim.get_pointer_fdtd_solver_fp(0)->HybridPICSolveE(
            view(Eminus), current_view, view(Ji), view(B), rho, minus,
            sim.GetEBUpdateEFlag()[0], 0, &model, false, false);
        if (boundary_rows) {
            apply_boundary(Eplus);
            apply_boundary(Eminus);
        }
        Real err = 0.;
        for (int c = 0; c < 3; ++c) {
            MF::LinComb(*oracle[c], -.5 / eps, *Eplus[c], 0, .5 / eps,
                        *Eminus[c], 0, 0, 1, 0);
            if (mask) {
                for (amrex::MFIter mfi(*oracle[c]); mfi.isValid(); ++mfi) {
                    auto const a = oracle[c]->array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            a(i, j, k) *=
                                j % 5 == 0 ? 0. : (j % 5 == 1 ? .3 : 1.);
                        });
                }
            }
            oracle[c]->OverrideSync(geom.periodicity());
            MF::Subtract(*oracle[c], *predicted[c], 0, 0, 1, 0);
            err = std::max(err, oracle[c]->norm0());
        }
        Real const scale = std::max(Real(1.), maxnorm(predicted));
        amrex::Print() << "NATIVE_PRESSURE " << name << " error=" << err
                       << " scale=" << scale << " relative=" << err / scale
                       << "\n";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            err < 2.e-11 * scale,
            "native HybridPICSolveE pressure derivative/sign/units");
        if (unequal_density && name == "ungated") {
            AMREX_ALWAYS_ASSERT(coupling.Freeze(moments.NumberDensity(),
                                                moments.NodalNumberDensity(),
                                                cview(old_weights)));
            coupling.Apply(view(old_predicted), direction);
            for (int c = 0; c < 3; ++c) {
                MF::Subtract(*old_predicted[c], *predicted[c], 0, 0, 1, 0);
            }
            Real const old_gap = maxnorm(old_predicted);
            amrex::Print() << "UNEQUAL_DENSITY midpoint_native_error=" << err
                           << " old_component_gap=" << old_gap << " V/m\n";
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                old_gap > 1.e-4 * scale, "unequal old/midpoint density must "
                                         "expose native coefficient omission");
        }
        if (recovery_rows && name == "ungated") {
            // Qualify the caller's two distinct density contracts against the
            // actual native replacement kernel. The Ohm secants above retain
            // midpoint charge; only the recovery mask reads old component zero.
            // An omission twin reproduces the former midpoint-mask mismatch.
            struct RecoveryCase {
                const char* label;
                const char* mode;
                bool frozen, flux_only;
                Real fraction, old_ratio, frozen_ratio;
            };
            RecoveryCase const cases[] = {
                {"live_old_low", "vacuum", false, false, .1, .5, 2.},
                {"live_old_high", "vacuum", false, false, 3., 2., .5},
                {"live_transition_low", "transition", false, false, .1, .5, 2.},
                {"live_transition_high", "transition", false, false, 3., 2., .5},
                {"frozen_low", "vacuum", true, false, .1, 2., .5},
                {"frozen_high", "vacuum", true, false, 3., .5, 2.},
                {"live_flux", "vacuum", false, true, .1, .5, 2.},
                {"live_global", "global", false, false, .1, 2., .5}};
            auto& native_rho = *sim.m_fields.get(FieldType::rho_fp, 0);
            auto& frozen_rho = *sim.m_fields.get("hybrid_rho_vacmask_fp", 0);
            AMREX_ALWAYS_ASSERT(native_rho.nComp() == 2);
            Owned native_plus, native_minus, recovered_weights, omission_weights;
            for (int c = 0; c < 3; ++c) {
                for (auto* fields : {&native_plus, &native_minus,
                                     &recovered_weights, &omission_weights}) {
                    (*fields)[c] = std::make_unique<MF>(er[c]->boxArray(), dm, 1, 0);
                }
                sim.m_fields.get("hybrid_E_vac_target_fp",
                    ablastr::fields::Direction{c}, 0)->setVal(0.);
            }
            bool const saved_frozen = model.m_darwin_vacuum_recovery_frozen_mask;
            auto const saved_components = model.m_darwin_vacuum_recovery_components;
            auto const saved_mode = model.m_darwin_vacuum_recovery_mask;
            Real const saved_fraction = model.m_darwin_vacuum_recovery_density_fraction;
            for (auto const& test : cases) {
                model.m_darwin_vacuum_recovery_frozen_mask = test.frozen;
                model.m_darwin_vacuum_recovery_components = test.flux_only ? "flux" : "all";
                model.m_darwin_vacuum_recovery_mask = test.mode;
                model.m_darwin_vacuum_recovery_density_fraction = test.fraction;
                Real const threshold = p.charge_floor * test.fraction;
                native_rho.setVal(test.old_ratio * threshold);
                MF::Copy(native_rho, rho, 0, 1, 1,
                         amrex::min(native_rho.nGrowVect(), rho.nGrowVect()));
                frozen_rho.setVal(test.frozen_ratio * threshold);
                bool const global = std::string(test.mode) == "global";
                bool const transition = std::string(test.mode) == "transition";
                bool const frozen = test.frozen;
                bool const flux_only = test.flux_only;
                for (int c = 0; c < 3; ++c) {
                    for (amrex::MFIter mfi(*weights[c]); mfi.isValid(); ++mfi) {
                        auto const native_charge = native_rho.const_array(mfi);
                        auto const fixed_charge = frozen_rho.const_array(mfi);
                        auto const base_weight = weights[c]->const_array(mfi);
                        auto const correct = recovered_weights[c]->array(mfi);
                        auto const wrong = omission_weights[c]->array(mfi);
                        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                              int i, int j, int k) {
                            int const di = c == 0, dj = c == 2;
                            Real const charge = frozen
                                ? .5 * (fixed_charge(i,j,k) + fixed_charge(i+di,j+dj,k))
                                : .5 * (native_charge(i,j,k,0) + native_charge(i+di,j+dj,k,0));
                            Real const midpoint = frozen ? charge
                                : .5 * (native_charge(i,j,k,1) + native_charge(i+di,j+dj,k,1));
                            bool const replaces = !flux_only || c == 1;
                            bool const mask = replaces && (global ||
                                (charge < threshold && (!transition || charge > 0.)));
                            bool const old_mask = replaces && (global ||
                                (midpoint < threshold && (!transition || midpoint > 0.)));
                            correct(i,j,k) = mask ? Real(0) : base_weight(i,j,k);
                            wrong(i,j,k) = old_mask ? Real(0) : base_weight(i,j,k);
                        });
                    }
                }
                apply_boundary(recovered_weights);
                apply_boundary(omission_weights);
                AMREX_ALWAYS_ASSERT(coupling.Freeze(moments.NumberDensity(),
                    moments.NodalNumberDensity(), cview(recovered_weights)));
                coupling.Apply(view(predicted), direction);
                AMREX_ALWAYS_ASSERT(coupling.Freeze(moments.NumberDensity(),
                    moments.NodalNumberDensity(), cview(omission_weights)));
                coupling.Apply(view(old_predicted), direction);
                // The target is held on finite-difference probes. These are
                // actual production registry fields and the native overwrite,
                // not a synthetic multiplication of the expected derivative.
                for (int sign = 0; sign < 2; ++sign) {
                    auto const& input = sign == 0 ? Eplus : Eminus;
                    auto& output = sign == 0 ? native_plus : native_minus;
                    for (int c = 0; c < 3; ++c) {
                        MF::Copy(*output[c], *input[c], 0, 0, 1, 0);
                    }
                    apply_boundary(output);
                    for (int c = 0; c < 3; ++c) {
                        MF::Copy(*er[c], *output[c], 0, 0, 1, 0);
                    }
                    model.ApplyVacuumFaradayE(1., false, true, false);
                    for (int c = 0; c < 3; ++c) {
                        MF::Copy(*output[c], *er[c], 0, 0, 1, 0);
                    }
                }
                Real defect = 0., omission = 0.;
                for (int c = 0; c < 3; ++c) {
                    MF::LinComb(*oracle[c], -.5/eps, *native_plus[c], 0,
                                .5/eps, *native_minus[c], 0, 0, 1, 0);
                    MF::Subtract(*old_predicted[c], *oracle[c], 0, 0, 1, 0);
                    MF::Subtract(*oracle[c], *predicted[c], 0, 0, 1, 0);
                    defect = std::max(defect, oracle[c]->norm0());
                    omission = std::max(omission, old_predicted[c]->norm0());
                }
                Real const response_scale = std::max(Real(1), maxnorm(predicted));
                bool const discriminates = !frozen && !flux_only && !global;
                amrex::Print() << "NATIVE_RECOVERY_PRESSURE " << test.label
                    << " defect=" << defect << " midpoint_mask_defect=" << omission
                    << " scale=" << response_scale << "\n";
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(defect < 2.e-11 * response_scale,
                    "pressure block must match actual native vacuum replacement rows");
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(discriminates
                    ? omission > 1.e-4 * response_scale
                    : omission < 2.e-11 * response_scale,
                    "native live-mask omission twin and unchanged frozen/flux controls");
            }
            model.m_darwin_vacuum_recovery_frozen_mask = saved_frozen;
            model.m_darwin_vacuum_recovery_components = saved_components;
            model.m_darwin_vacuum_recovery_mask = saved_mode;
            model.m_darwin_vacuum_recovery_density_fraction = saved_fraction;
        }
        if (name == "disabled" || name == "conductor_raw") {
            AMREX_ALWAYS_ASSERT(maxnorm(predicted) == 0.);
        }
    }
    int minimum_local_boxes = 0;
    for (int owner : dm.ProcessorMap()) {
        minimum_local_boxes += owner == amrex::ParallelDescriptor::MyProc();
    }
    amrex::ParallelDescriptor::ReduceIntMin(minimum_local_boxes);
    AMREX_ALWAYS_ASSERT(!expect_empty_rank || minimum_local_boxes == 0);
    amrex::Print() << "PASS native pressure Ohm oracle boxes=" << cells.size()
                   << " periodic_z=" << geom.isPeriodic(1)
                   << " minimum_local_boxes=" << minimum_local_boxes << "\n";
}
} // namespace
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        amrex::ParmParse pp("hybrid_pic_model");
        pp.add("density_pedestal", 1);
        pp.add("density_pedestal_profile(x,y,z)", std::string("2.e17"));
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        run(sim);
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
