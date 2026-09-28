/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConduction.H"
#include <AMReX.H>
#include <AMReX_Print.H>
#include <array>
#include <cmath>
#include <limits>

using namespace amrex::literals;
namespace mhd = warpx::thermal::mhd;
namespace {
void
check (bool condition, char const* message) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(condition, message);
}

// Two periodic cells, constant capacity/density, no source: the difference
// mode has eigenvalue -4*chi/h^2. Verify both conservation and the exact
// theta_c amplification by evaluating the real face composition and residual.
void
stage_amplification (amrex::Real theta, amrex::Real theta_c, amrex::Real z) {
    auto const dt = z / 4.0_rt;
    auto const w = theta_c / theta;
    auto const old_amplitude = 0.125_rt;
    auto const stage_amplitude = old_amplitude *
                                 (1.0_rt - theta * z * (1.0_rt - w)) /
                                 (1.0_rt + theta * z * w);
    std::array<amrex::Real, 2> old{20.0_rt + old_amplitude,
                                   20.0_rt - old_amplitude};
    std::array<amrex::Real, 2> stage{20.0_rt + stage_amplitude,
                                     20.0_rt - stage_amplitude};
    std::array<amrex::Real, 2> flux{};
    for (int i = 0; i < 2; ++i) {
        int const j = 1 - i;
        flux[i] =
            mhd::ComposeConductionFlux(stage[j] - stage[i], old[j] - old[i],
                                       theta, theta_c, 1.0_rt, 0.0_rt)
                .flux;
    }
    std::array<amrex::Real, 2> endpoint{};
    for (int i = 0; i < 2; ++i) {
        auto const rhs = -theta * dt * (flux[i] - flux[1 - i]);
        check(
            std::abs((stage[i] - rhs) - old[i]) < 1.e-10_rt,
            "mixed-stage cell energy satisfies the implicit finite-volume row");
        endpoint[i] = stage[i] / theta + (1.0_rt - 1.0_rt / theta) * old[i];
    }
    auto const measured =
        (endpoint[0] - endpoint[1]) / (2.0_rt * old_amplitude);
    auto const exact =
        (1.0_rt - (1.0_rt - theta_c) * z) / (1.0_rt + theta_c * z);
    check(std::abs(measured - exact) < 1.e-12_rt,
          "thermal theta amplification");
    check(std::abs(endpoint[0] + endpoint[1] - 40.0_rt) < 1.e-13_rt,
          "closed energy conservation");
    if (z > 1000.0_rt && theta_c == 0.5_rt) {
        check(measured < -0.99_rt,
              "midpoint retains stiff sign-alternating mode");
    }
    if (z > 1000.0_rt && theta_c == 1.0_rt) {
        check(measured > 0.0_rt && measured < 2.e-4_rt,
              "thermal backward Euler damps stiff mode");
    }
    amrex::Print() << "MHD_STAGE theta=" << theta << " theta_c=" << theta_c
                   << " z=" << z << " amplification=" << measured << "\n";
}

void
cap_staging () {
    auto const composed = mhd::ComposeConductionFlux(12.0_rt, -4.0_rt, 0.5_rt,
                                                     1.0_rt, 1.0_rt, 5.0_rt);
    check(composed.flux == -28.0_rt / (1.0_rt + 28.0_rt / 5.0_rt),
          "mixed uncapped flux is capped once");
    auto const separate =
        2.0_rt * mhd::ComposeConductionFlux(12.0_rt, 0.0_rt, 0.5_rt, 0.5_rt,
                                            1.0_rt, 5.0_rt)
                     .flux -
        mhd::ComposeConductionFlux(-4.0_rt, 0.0_rt, 0.5_rt, 0.5_rt, 1.0_rt,
                                   5.0_rt)
            .flux;
    check(std::abs(composed.flux - separate) > 1.0_rt,
          "test distinguishes wrong mix-after-cap ordering");
    auto const no_old = mhd::ComposeConductionFlux(
        12.0_rt, std::numeric_limits<amrex::Real>::quiet_NaN(), 0.5_rt, 0.5_rt,
        1.0_rt, 5.0_rt);
    check(std::isfinite(no_old.flux),
          "equal theta branch does not consume old flux");
    auto const eps = 1.e-5_rt;
    auto const plus = mhd::ComposeConductionFlux(12.0_rt + eps, -4.0_rt, 0.5_rt,
                                                 1.0_rt, 1.0_rt, 5.0_rt)
                          .flux;
    auto const minus =
        mhd::ComposeConductionFlux(12.0_rt - eps, -4.0_rt, 0.5_rt, 1.0_rt,
                                   1.0_rt, 5.0_rt)
            .flux;
    auto const derivative = (plus - minus) / (2.0_rt * eps);
    check(std::abs(derivative + composed.pc_stage_cap_factor) < 1.e-9_rt,
          "PC cap/stage factor is the fixed-q_limit face derivative");
}

#if defined(WARPX_DIM_RZ)
void
corrected_mhd_face () {
    // Host fixture for the pure GPU-portable face kernel. The production
    // kernel uses the same Array4 accessor; no MultiFab state is mutated.
    std::array<amrex::Real, 16 * 16 * 3> b_storage{};
    amrex::Array4<amrex::Real> b(b_storage.data(), {-4, -4, 0}, {12, 12, 1}, 3);
    for (int j = -4; j < 12; ++j)
        for (int i = -4; i < 12; ++i) {
            b(i, j, 0, 0) = 1.0_rt;
            b(i, j, 0, 2) = 1.0_rt;
        }
    amrex::Array4<amrex::Real const> bc(b);
    mhd::ChaconFDParams p;
    p.normal = 2;
    p.tangential = 0;
    p.order = 2;
    p.domain_lo_normal = -4;
    p.domain_hi_normal = 11;
    p.domain_lo_tangential = -4;
    p.domain_hi_tangential = 11;
    p.limiter_width = 0.001_rt;
    auto const masked = [] (int, int, int) { return false; };
    for (auto const slope : {-1.0_rt, 1.0_rt}) {
        auto const e = [=] (int i, int j, int) {
            return 20.0_rt + (j < 2 ? 1.0_rt : 3.0_rt) + slope * i;
        };
        amrex::Real pc = 0.0_rt;
        auto const full = mhd::chacon_fd_face_flux(e, masked, bc, 4, 1, 0, 4, 2,
                                                   0, 0.0_rt, 2.0_rt, p, pc);
        // With B_r=B_z=1 and chi_par-chi_perp=2, Xi_nn=Xi_nt=1.
        // F_co=2, F_cross=slope*e_donor/e_mean. Smooth SMART at either
        // plateau returns nearly its donor value; the stale MHD sign
        // would reverse the choice and fail this interval gate.
        auto const cross = full - 2.0_rt;
        auto const mean = 0.5_rt * (e(4, 1, 0) + e(4, 2, 0));
        auto const donor = slope > 0.0_rt ? e(4, 2, 0) : e(4, 1, 0);
        auto const wrong = slope > 0.0_rt ? e(4, 1, 0) : e(4, 2, 0);
        check(std::abs(cross - slope * donor / mean) < 0.002_rt,
              "gradient-flux donor follows physical -v direction");
        check(std::abs(cross - slope * wrong / mean) > 0.05_rt,
              "regression excludes stale MHD donor sign");
        check(pc == 1.0_rt,
              "same compact normal PC coefficient after donor correction");
    }
    // Nonlinear smooth SMART: the pinned algorithm combines F(e_stage) and
    // F(e_old), not F(2e_stage-e_old). These profiles make the distinction
    // measurable without negative energy or changing face coefficients.
    std::array<amrex::Real, 4> now{8.0_rt, 8.0_rt, 10.0_rt, 11.0_rt};
    std::array<amrex::Real, 4> old{9.0_rt, 11.0_rt, 12.0_rt, 13.0_rt};
    auto const en = [=] (int i, int j, int) {
        return now[std::clamp(j, 0, 3)] + 0.2_rt * i;
    };
    auto const eo = [=] (int i, int j, int) {
        return old[std::clamp(j, 0, 3)] + 0.1_rt * i;
    };
    auto const ex = [=] (int i, int j, int k) {
        return 2.0_rt * en(i, j, k) - eo(i, j, k);
    };
    amrex::Real pc = 0.0_rt;
    auto const fn = mhd::chacon_fd_face_flux(en, masked, bc, 4, 1, 0, 4, 2, 0,
                                             0.0_rt, 2.0_rt, p, pc);
    auto const fo = mhd::chacon_fd_face_flux(eo, masked, bc, 4, 1, 0, 4, 2, 0,
                                             0.0_rt, 2.0_rt, p, pc);
    auto const fx = mhd::chacon_fd_face_flux(ex, masked, bc, 4, 1, 0, 4, 2, 0,
                                             0.0_rt, 2.0_rt, p, pc);
    auto const exact =
        mhd::ComposeConductionFlux(fn, fo, 0.5_rt, 1.0_rt, 1.0_rt, 3.0_rt);
    auto const alternate =
        mhd::ComposeConductionFlux(fx, 0.0_rt, 0.5_rt, 0.5_rt, 1.0_rt, 3.0_rt);
    check(std::abs(exact.flux - alternate.flux) > 1.e-5_rt,
          "limited mixed-stage flux differs from limit-after-extrapolation");
    amrex::Print() << "MHD_NONLINEAR_STAGE flux=" << exact.flux
                   << " extrapolated=" << alternate.flux << "\n";
    // Smooth FD4 normal flux must reproduce the second derivative of x^4
    // exactly in the telescoped divergence (with a constant tensor).
    p.order = 4;
    auto const quartic = [] (int, int j, int) {
        return 100.0_rt + amrex::Real(j * j * j * j);
    };
    auto const left = mhd::chacon_fd_face_flux(quartic, masked, bc, 4, 0, 0, 4,
                                               1, 0, 2.0_rt, 0.0_rt, p, pc);
    auto const right = mhd::chacon_fd_face_flux(quartic, masked, bc, 4, 1, 0, 4,
                                                2, 0, 2.0_rt, 0.0_rt, p, pc);
    check(std::abs((right - left) - 24.0_rt) < 1.e-10_rt,
          "MHD FD4 conservative divergence on quartic profile");
}
#endif
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        for (auto const theta_c : {0.5_rt, 1.0_rt})
            for (auto const z : {0.2_rt, 8000.0_rt})
                stage_amplification(0.5_rt, theta_c, z);
        stage_amplification(1.0_rt, 1.0_rt, 8000.0_rt);
        cap_staging();
#if defined(WARPX_DIM_RZ)
        corrected_mhd_face();
#endif
        amrex::Print() << "EULERIAN_THERMAL_STAGE_PASS dim=" << AMREX_SPACEDIM
                       << "\n";
    }
    amrex::Finalize();
}
