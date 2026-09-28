/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionFDOperator.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionWall.H"
#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <iomanip>

using namespace amrex::literals;
namespace thermal = warpx::thermal;

namespace {
struct Kappa {
    amrex::Real chi, temperature_slope;
    AMREX_GPU_HOST_DEVICE amrex::Real
    operator()(amrex::Real n, amrex::Real eV, amrex::Real) const {
        return 1.5_rt * PhysConst::kb * n * chi *
               (1.0_rt + temperature_slope * eV);
    }
};

void
check (bool condition, char const* message) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(condition, message);
}

amrex::Real
max_difference (amrex::MultiFab const& a, amrex::MultiFab const& b,
                int ng = 0) {
    amrex::MultiFab diff(a.boxArray(), a.DistributionMap(), a.nComp(), ng);
    amrex::MultiFab::Copy(diff, a, 0, 0, a.nComp(), ng);
    amrex::MultiFab::Subtract(diff, b, 0, 0, a.nComp(), ng);
    amrex::Real out = 0.0_rt;
    for (int n = 0; n < a.nComp(); ++n) {
        out = std::max(out, diff.norm0(n, ng));
    }
    return out;
}

// Physical closed-box heat integral, with exact clipped dual volumes and
// unique node ownership. Every internal face must cancel with its neighbor.
amrex::Real
heat_rate (amrex::MultiFab const& rhs, amrex::MultiFab const& bne,
           thermal::ConductionFDGeometry const& geometry, bool absolute) {
    amrex::MultiFab weighted(rhs.boxArray(), rhs.DistributionMap(), 1, 0);
    auto const volume = geometry.dual_volume;
    for (amrex::MFIter mfi(weighted); mfi.isValid(); ++mfi) {
        auto const out = weighted.array(mfi);
        auto const in = rhs.const_array(mfi);
        auto const n = bne.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const value = in(i, j, k) * 1.5_rt * PhysConst::kb *
                                   n(i, j, k, thermal::ConductionState::b_ne) *
                                   volume(i, j, k);
                out(i, j, k) = absolute ? std::abs(value) : value;
            });
    }
    return weighted.sum_unique(0, false, geometry.period);
}

void
wall_tests () {
    amrex::Real const T = 20000.0_rt, inner = 30000.0_rt, Tw = 5000.0_rt;
    amrex::Real const n = 2.e18_rt, gamma = 5.0_rt / 3.0_rt,
                      mi = PhysConst::m_p;
    check(thermal::IsothermalWallTemperature(T, Tw, 0.0_rt, false, gamma, mi) ==
              Tw,
          "uncapped bath is a prescribed temperature");
    auto const speed = std::sqrt(PhysConst::kb * T / PhysConst::m_e);
    auto const cap = 1.e-8_rt;
    auto const capped =
        thermal::IsothermalWallTemperature(T, Tw, cap, false, gamma, mi);
    check(capped == std::max(Tw, T - cap * speed * T / 1.5_rt),
          "free-streaming bath cap");
    check(thermal::IsothermalWallTemperature(100.0_rt, Tw, cap, true, gamma,
                                             mi) == Tw,
          "bath heating remains an uncapped constraint in the legacy map");
    auto const capacity = 1.5_rt * PhysConst::kb * n;
    check(std::abs(thermal::PrescribedWallTemperatureChange(
                       T, n, capacity * 12.0_rt) -
                   12.0_rt) < 1.e-12_rt,
          "prescribed inward flux sign and capacity");
    check(thermal::PrescribedWallTemperatureChange(
              T, n, -2.0_rt * capacity * T) == -T,
          "legacy cooling is bounded at zero temperature");
    Kappa const kappa{3.0_rt, 0.0_rt};
    auto const target =
        thermal::EvaluateLegTarget(T, inner, n, n, n, true, 0.1_rt, 1.0_rt,
                                   2.0_rt, Tw, Tw, 3, kappa, 0.0_rt);
    auto const gi = capacity * 3.0_rt / 0.1_rt, gl = capacity * 3.0_rt / 2.0_rt;
    auto const exact_target = (gi * inner + gl * Tw) / (gi + gl);
    check(std::abs(target.temperature - exact_target) < 1.e-10_rt,
          "series-conductance target");
    check(thermal::LegWallTemperature(T, target, n, 20.0_rt, 1.e-2_rt, 0.0_rt,
                                      false, gamma, mi) == T,
          "leg map is drain-only when row is below series temperature");
    auto const closed =
        thermal::EvaluateLegTarget(T, inner, n, n, n, false, 0.1_rt, 1.0_rt,
                                   2.0_rt, Tw, Tw, 3, kappa, 0.0_rt);
    check(closed.interior_conductance == 0.0_rt && closed.temperature == Tw,
          "closed-face leg target");
    auto const rate =
        thermal::LegWallRate(T, closed, n, 20.0_rt, 0.0_rt, false, gamma, mi);
    auto const exact_rate = (Tw - T) * gl / (capacity / 20.0_rt);
    check(std::abs(rate - exact_rate) < 1.e-9_rt,
          "leg generator capacity uses physical V/A");
    amrex::Real last_error = 1.0_rt;
    for (auto const dt : {1.e-5_rt, 1.e-6_rt, 1.e-7_rt}) {
        auto const out = thermal::LegWallTemperature(T, closed, n, 20.0_rt, dt,
                                                     0.0_rt, false, gamma, mi);
        auto const fd = (out - T) / dt;
        auto const err = std::abs((fd - rate) / rate);
        check(err < last_error,
              "finite-dt map derivative approaches pure leg rate");
        last_error = err;
    }
    check(last_error < 2.e-6_rt, "leg dt->0 rate");
    // A radial/axial corner receives additive prescribed energy using each
    // face's own A/V. The extensive heat is independent of the order of fluxes.
    auto const dT1 =
        thermal::PrescribedWallTemperatureChange(T, n, capacity * 4.0_rt);
    auto const dT2 =
        thermal::PrescribedWallTemperatureChange(T + dT1, n, capacity * 7.0_rt);
    check(std::abs(capacity * (dT1 + dT2) - capacity * 11.0_rt) < 1.e-13_rt,
          "corner flux heat sum");
}

amrex::Real
smooth_axial_error (int cells) {
    amrex::Box domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
    amrex::RealBox physical({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 1.0, 1.0)});
    int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom(domain, &physical,
#ifdef WARPX_DIM_RZ
                         1,
#else
                         0,
#endif
                         periodic);
    amrex::BoxArray ba(domain);
    ba.maxSize(16);
    ba.surroundingNodes();
    amrex::DistributionMapping dm(ba);
    using State = thermal::ConductionState;
    amrex::MultiFab T(ba, dm, 1, 3), bne(ba, dm, State::b_ncomp, 3),
        xi(ba, dm, AMREX_SPACEDIM * (AMREX_SPACEDIM + 1) / 2, 2),
        rhs(ba, dm, 1, 0), fc, rb;
    bne.setVal(0.0_rt);
    bne.setVal(2.e18_rt, State::b_ne, 1, 3);
    bne.setVal(1.0_rt, State::b_open, 1, 3);
    bne.setVal(1.0_rt, State::b_bz, 1, 3);
    bne.setVal(1.0_rt, State::b_B2, 1, 3);
    auto const wave = 2.0_rt * MathConst::pi;
    auto const spacing = 1.0_rt / cells;
    for (amrex::MFIter mfi(T); mfi.isValid(); ++mfi) {
        auto const t = T.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                              int k) {
            int const node[3] = {i, j, k};
            t(i, j, k) =
                20000.0_rt +
                1000.0_rt * std::cos(wave * spacing * node[AMREX_SPACEDIM - 1]);
        });
    }
    thermal::ConductionFDGeometry geometry(geom);
    thermal::ConductionFDOptions options;
    options.limiter = 5;
    options.fourth_order = true;
    thermal::BuildConductionTensor(T, bne, xi, geometry, options,
                                   Kappa{40.0_rt, 0.0_rt},
                                   Kappa{0.0_rt, 0.0_rt}, 0.0_rt);
    thermal::EvaluateConductionFDRHS(T, bne, xi, rhs, fc, rb, geometry, options,
                                     false);
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto const result = rhs.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int const node[3] = {i, j, k};
                bool interior = true;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    interior = interior && node[d] >= 4 && node[d] <= cells - 4;
                }
                auto const exact =
                    -40.0_rt * wave * wave * 1000.0_rt *
                    std::cos(wave * spacing * node[AMREX_SPACEDIM - 1]);
                result(i, j, k) =
                    interior ? std::abs(result(i, j, k) - exact) : 0.0_rt;
            });
    }
    return rhs.norm0();
}

void
operator_tests (int max_grid) {
    int const cells = 32;
    amrex::Box domain(amrex::IntVect(0), amrex::IntVect(cells - 1));
    amrex::RealBox physical({AMREX_D_DECL(0.0, 0.0, 0.0)},
                            {AMREX_D_DECL(1.0, 2.0, 1.5)});
    int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom(domain, &physical,
#ifdef WARPX_DIM_RZ
                         1,
#else
                         0,
#endif
                         periodic);
    amrex::BoxArray ba(domain);
    ba.maxSize(max_grid);
    ba.surroundingNodes();
    amrex::DistributionMapping dm(ba);
    thermal::ConductionFDGeometry geometry(geom);
    thermal::ConductionFDOptions options;
    options.limiter = 5;
    using State = thermal::ConductionState;
    int constexpr tensor_components = AMREX_SPACEDIM * (AMREX_SPACEDIM + 1) / 2;
    amrex::MultiFab T(ba, dm, 1, 3), saved(ba, dm, 1, 3),
        context(ba, dm, State::b_ncomp, 3),
        context_saved(ba, dm, State::b_ncomp, 3),
        tensor(ba, dm, tensor_components, 2), rhs(ba, dm, 1, 0),
        previous(ba, dm, 1, 0), fcache(ba, dm, AMREX_SPACEDIM, 1),
        budget(ba, dm, 1, 1);
    context.setVal(0.0_rt);
    context.setVal(1.0_rt, State::b_open, 1, 3);
    context.setVal(1.0_rt, State::b_ebm, 1, 3);
    auto const dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(T); mfi.isValid(); ++mfi) {
        auto const t = T.array(mfi), b = context.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto const x = i * dx[0], y = j * dx[1];
                t(i, j, k) = 20000.0_rt + 2000.0_rt * x * x + 3000.0_rt * y +
                             1000.0_rt * std::sin(6.0_rt * x * y);
                b(i, j, k, State::b_ne) =
                    2.e18_rt * (1.1_rt + 0.1_rt * std::cos(x + y));
                b(i, j, k, State::b_bx) = 0.6_rt;
                b(i, j, k, State::b_by) = 0.0_rt;
                b(i, j, k, State::b_bz) = 0.8_rt;
                b(i, j, k, State::b_B2) = 1.0_rt;
            });
    }
    amrex::MultiFab::Copy(saved, T, 0, 0, 1, 3);
    amrex::MultiFab::Copy(context_saved, context, 0, 0, State::b_ncomp, 3);
    Kappa const kp{40.0_rt, 0.03_rt}, kt{0.5_rt, 0.01_rt};
    for (int limiter : {0, 1, 2, 3, 4, 5})
        for (bool fourth : {false, true})
            for (bool bounded : {false, true}) {
                options.limiter = limiter;
                options.fourth_order = fourth;
                options.flux_budget = bounded;
                options.budget_floor_K = 5000.0_rt;
                options.budget_dt = 1.0_rt;
                fcache.setVal(0.0_rt);
                budget.setVal(1.0_rt);
                thermal::BuildConductionTensor(T, context, tensor, geometry,
                                               options, kp, kt, 0.0_rt);
                thermal::EvaluateConductionFDRHS(T, context, tensor, rhs,
                                                 fcache, budget, geometry,
                                                 options, false);
                check(rhs.is_finite(), "finite anisotropic FD RHS");

                auto const heat = heat_rate(rhs, context, geometry, false);
                auto const l1 = heat_rate(rhs, context, geometry, true);
                check(std::abs(heat) < 2.e-13_rt * l1,
                      "closed-domain physical heat balance at axis, walls and "
                      "seams");
                amrex::MultiFab::Copy(previous, rhs, 0, 0, 1, 0);
                thermal::EvaluateConductionFDRHS(T, context, tensor, rhs,
                                                 fcache, budget, geometry,
                                                 options, false);
                check(max_difference(previous, rhs) == 0.0_rt,
                      "same-state repeatability");
                check(max_difference(saved, T, 3) == 0.0_rt,
                      "trial temperature including ghosts is read-only");
                check(max_difference(context_saved, context, 3) == 0.0_rt,
                      "frozen B/density context is read-only");
                amrex::Print()
                    << std::setprecision(17) << "RHS_CASE limiter=" << limiter
                    << " fd4=" << fourth << " budget=" << bounded
                    << " norm=" << rhs.norm0() << " heat=" << heat
                    << " l1=" << l1 << "\n";
            }
    // The coefficient parser and tensor must see each trial temperature;
    // evaluate a different T, then restore A and reproduce its exact RHS.
    options.limiter = 5;
    options.fourth_order = true;
    options.flux_budget = false;
    options.flux_limit = 0.05_rt;
    thermal::BuildConductionTensor(T, context, tensor, geometry, options, kp,
                                   kt, 0.0_rt);
    thermal::EvaluateConductionFDRHS(T, context, tensor, rhs, fcache, budget,
                                     geometry, options, false);
    amrex::MultiFab::Copy(previous, rhs, 0, 0, 1, 0);
    T.mult(1.01_rt, 0, 1, 3);
    thermal::BuildConductionTensor(T, context, tensor, geometry, options, kp,
                                   kt, 0.0_rt);
    thermal::EvaluateConductionFDRHS(T, context, tensor, rhs, fcache, budget,
                                     geometry, options, false);
    check(max_difference(previous, rhs) > 1.0_rt,
          "live thermal coefficients and RHS respond to changed trial");
    amrex::MultiFab::Copy(T, saved, 0, 0, 1, 3);
    thermal::BuildConductionTensor(T, context, tensor, geometry, options, kp,
                                   kt, 0.0_rt);
    thermal::EvaluateConductionFDRHS(T, context, tensor, rhs, fcache, budget,
                                     geometry, options, false);
    check(max_difference(previous, rhs) == 0.0_rt,
          "A/B/A evaluation has no thermal history");
    // Exact constant-temperature null response is tested with FD2; FD4's
    // derivative sum may leave roundoff scaled by its cancellation condition.
    T.setVal(20000.0_rt);
    options.flux_limit = 0.0_rt;
    options.fourth_order = false;
    thermal::BuildConductionTensor(T, context, tensor, geometry, options, kp,
                                   kt, 0.0_rt);
    thermal::EvaluateConductionFDRHS(T, context, tensor, rhs, fcache, budget,
                                     geometry, options, false);
    check(rhs.norm0() == 0.0_rt, "constant temperature adiabatic nullspace");
    context.setVal(0.0_rt, State::b_open, 1, 3);
    thermal::BuildConductionTensor(T, context, tensor, geometry, options, kp,
                                   kt, 0.0_rt);
    thermal::EvaluateConductionFDRHS(T, context, tensor, rhs, fcache, budget,
                                     geometry, options, false);
    check(rhs.norm0() == 0.0_rt,
          "closed density/covered mask gives zero interior RHS");
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        int max_grid = 8;
        amrex::ParmParse pp("test");
        pp.query("max_grid_size", max_grid);
        wall_tests();
        operator_tests(max_grid);
        auto const coarse = smooth_axial_error(16),
                   fine = smooth_axial_error(32);
        auto const order = std::log(coarse / fine) / std::log(2.0_rt);
        check(order >= 3.5_rt, "smooth interior FD4 spatial order");
        amrex::Print() << "SPATIAL_ORDER " << order << " coarse=" << coarse
                       << " fine=" << fine << "\n";
        amrex::Print() << "THERMAL_OPERATOR_PASS dim=" << AMREX_SPACEDIM
                       << " max_grid=" << max_grid << "\n";
    }
    amrex::Finalize();
}
