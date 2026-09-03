/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

/* Standalone unit test of the TVD reconstruction in
 * FieldSolver/ImplicitSolvers/ThetaImplicitMHD_K.H
 * (implicit_mhd.fluid_reconstruction). Host-only, runs in milliseconds.
 *
 * Two halves.
 *
 * A. LIMITER MATH -- the closed forms the kernel's block comment
 *    documents, checked rather than asserted:
 *      * smooth_median3(a, a, a, w) == a EXACTLY in floating point for
 *        every width (the reason the identity is collapsed analytically
 *        before evaluation: a uniform region must reconstruct to donor
 *        cell, or the scheme manufactures an O(w) face jump on every
 *        face of every quiescent region),
 *      * deviation from the hard median bounded by w/2,
 *      * the median's smooth-flow slope defect d (1 - kappa/sqrt(2)),
 *        which is where its numerical diffusivity 0.354 kappa |u| dx
 *        comes from,
 *      * van Albada exact on the symmetric branch for ANY guard and
 *        exactly zero at an extremum,
 *      * the new-extremum bounds of each mode, and
 *      * Lipschitz smoothness across the minmod SELECTION kink
 *        d_up = d_down, where a HARD minmod's derivative jumps 1 -> 0.
 *
 * B. FACE-STATE ADMISSIBILITY -- the production constraint. The ion
 *    energy channel is the fragile one in the formation campaign (arms
 *    have died on ion_energy going negative in the interior at first
 *    order), so a reconstruction that can hand the Riemann solver an
 *    inadmissible interface state is unusable no matter what it does to
 *    the order. reconstruct_face_states is therefore driven with
 *    ADVERSARIAL randomized four-cell stencils -- 1e4:1 density
 *    contrasts across one cell, counter-streaming shear, internal
 *    energies riding their floors, tangential fields through zero -- in
 *    every mode and every ion closure, and every reconstructed face
 *    state is required to satisfy
 *
 *      rho_face      >= mass_density_floor
 *      E_i - KE      >= ion_pressure_floor/(gamma_i - 1)   [total/dual]
 *      U_e           >= electron_pressure_floor/(gamma_e - 1)
 *      U_i           >= ion_pressure_floor/(gamma_i - 1)   [dual]
 *      U_par, U_perp >= p_i_floor/2, p_i_floor             [cgl]
 *
 *    with every derived member finite. The E_i row is the one that
 *    matters: it holds because the reconstruction never touches the
 *    conserved ion energy -- it reconstructs the SPECIFIC internal
 *    energy (E_i - KE)/rho, floors the resulting internal energy
 *    density, and rebuilds the kinetic part from the reconstructed
 *    momentum with the same denominator rebuild_cell_state uses.
 */

#include <FieldSolver/ImplicitSolvers/ThetaImplicitMHD_K.H>

#include <cmath>
#include <cstdio>
#include <random>

namespace
{
using theta_implicit_mhd::CellState;
using theta_implicit_mhd::FluxParameters;

int s_failures = 0;

void check (const bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++s_failures;
    }
}

double hard_median (const double a, const double b, const double c)
{
    return std::min(std::max(a, b), std::max(c, std::min(a, b)));
}

// The deck-scale floors of the RZ formation production deck, so the
// adversarial sweep exercises the real ratios (rho_floor/rho_peak ~ 4e-4,
// p_floor/p ~ 1e-9) rather than pretty round ones.
constexpr double density_floor = 2.5089e-10;
constexpr double ion_pressure_floor = 3.785e-5;
constexpr double electron_pressure_floor = 6.008e-3;
constexpr double gamma_gas = 5.0 / 3.0;

FluxParameters make_parameters (const int mode, const int closure)
{
    FluxParameters parameters{};
    parameters.density_floor = density_floor;
    parameters.charge_to_mass = 9.578833e7;
    parameters.gamma_e = gamma_gas;
    parameters.gamma_i = gamma_gas;
    parameters.reference_density = 1.6726e-7;
    parameters.reference_ion_pressure = 0.0;
    parameters.electron_pressure_floor = electron_pressure_floor;
    parameters.ion_pressure_floor = ion_pressure_floor;
    parameters.include_hall_term = false;
    parameters.use_rusanov = false;
    parameters.use_hllc = false;
    parameters.barotropic_signal_speeds = false;
    parameters.contact_blend = 0.0;
    parameters.theta = 0.5;
    parameters.mu0 = 1.2566370612685e-06;
    parameters.fluid_reconstruction = mode;
    parameters.reconstruction_kappa = 0.01;
    // closure: 0 = barotropic, 1 = total_energy, 2 = dual_energy, 3 = cgl
    parameters.total_energy_closure = (closure == 1 || closure == 2);
    parameters.dual_energy_closure = (closure == 2);
    parameters.cgl_closure = (closure == 3);
    if (closure == 0) {
        // The barotropic branch needs a positive reference pressure or
        // ion_pressure() short-circuits to zero.
        parameters.reference_ion_pressure = 1.0e2;
    }
    return parameters;
}

// A deliberately hostile cell: log-uniform density over four decades
// (so a single face can straddle a 1e4:1 step), counter-streaming
// velocities, internal energies that reach down to a few times their
// floors, and tangential fields through zero.
CellState make_cell (std::mt19937& rng, const FluxParameters& parameters)
{
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> signed_unit(-1.0, 1.0);

    CellState state{};
    state.density = density_floor * std::pow(10.0, 4.0 * unit(rng));
    state.safe_density = std::max(state.density, parameters.density_floor);
    for (int component = 0; component < 3; ++component) {
        state.ion_velocity[component] = 6.0e5 * signed_unit(rng);
        state.momentum[component] =
            state.safe_density * state.ion_velocity[component];
    }
    const double electron_floor =
        electron_pressure_floor / (gamma_gas - 1.0);
    const double ion_floor = ion_pressure_floor / (gamma_gas - 1.0);
    state.electron_energy = electron_floor * std::pow(10.0, 6.0 * unit(rng));
    const double internal = ion_floor * std::pow(10.0, 6.0 * unit(rng));
    double kinetic = 0.0;
    for (int component = 0; component < 3; ++component) {
        kinetic += state.momentum[component] * state.momentum[component];
    }
    kinetic *= 0.5 / state.safe_density;
    state.ion_energy = internal + kinetic;
    state.ion_internal_energy = ion_floor * std::pow(10.0, 6.0 * unit(rng));
    state.old_ion_internal_energy = state.ion_internal_energy;
    state.ion_parallel_energy =
        0.5 * ion_pressure_floor * std::pow(10.0, 6.0 * unit(rng));
    state.ion_perp_energy =
        ion_pressure_floor * std::pow(10.0, 6.0 * unit(rng));
    for (int component = 0; component < 3; ++component) {
        state.magnetic[component] = 20.0 * signed_unit(rng);
    }
    state.electron_velocity_normal = state.ion_velocity[0];
    return state;
}

bool finite_state (const CellState& state)
{
    bool ok = std::isfinite(state.density) &&
              std::isfinite(state.safe_density) &&
              std::isfinite(state.electron_energy) &&
              std::isfinite(state.ion_energy) &&
              std::isfinite(state.ion_internal) &&
              std::isfinite(state.ion_pressure) &&
              std::isfinite(state.electron_pressure) &&
              std::isfinite(state.electron_velocity_normal) &&
              std::isfinite(state.sound_speed) &&
              std::isfinite(state.wave_speed) &&
              std::isfinite(state.fast_speed) &&
              std::isfinite(state.fast_wave_speed) &&
              std::isfinite(state.ion_parallel_energy) &&
              std::isfinite(state.ion_perp_energy) &&
              std::isfinite(state.ion_internal_energy);
    for (int component = 0; component < 3; ++component) {
        ok = ok && std::isfinite(state.momentum[component]) &&
             std::isfinite(state.ion_velocity[component]) &&
             std::isfinite(state.magnetic[component]);
    }
    return ok;
}

double internal_ion_energy (const CellState& state)
{
    double kinetic = 0.0;
    for (int component = 0; component < 3; ++component) {
        kinetic += state.momentum[component] * state.momentum[component];
    }
    kinetic *= 0.5 / state.safe_density;
    return state.ion_energy - kinetic;
}
} // namespace

int main ()
{
    // ------------------------------------------------------------------
    // A. LIMITER MATH
    // ------------------------------------------------------------------
    for (const double width : {0.0, 1.0e-8, 1.0e-3, 0.1, 1.0, 17.0}) {
        for (const double value : {-3.5, 0.0, 1.0, 1.0e6}) {
            check(theta_implicit_mhd::smooth_median3(value, value, value,
                                                     width) == value,
                  "smooth_median3(a, a, a, w) is not exactly a");
        }
    }

    std::mt19937 rng(20260902);
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);
    double worst_median_deviation = 0.0;
    for (int trial = 0; trial < 200000; ++trial) {
        const double a = uniform(rng);
        const double b = uniform(rng);
        const double c = uniform(rng);
        constexpr double width = 0.05;
        worst_median_deviation = std::max(
            worst_median_deviation,
            std::abs(theta_implicit_mhd::smooth_median3(a, b, c, width) -
                     hard_median(a, b, c)) /
                width);
        check(std::abs(theta_implicit_mhd::smooth_median3(a, b, c, 0.0) -
                       hard_median(a, b, c)) < 1.0e-14,
              "smooth_median3 at w = 0 is not the hard median");
    }
    std::printf("smooth_median3 worst deviation / w = %.4f (bound 0.5)\n",
                worst_median_deviation);
    check(worst_median_deviation < 0.51, "median deviation exceeds w/2");

    const FluxParameters median = make_parameters(
        theta_implicit_mhd::reconstruction_median, 1);
    const FluxParameters albada = make_parameters(
        theta_implicit_mhd::reconstruction_vanalbada, 1);
    const FluxParameters unlimited = make_parameters(
        theta_implicit_mhd::reconstruction_unlimited, 1);

    const double predicted_defect =
        1.0 - median.reconstruction_kappa / std::sqrt(2.0);
    for (const double slope : {1.0e-3, 1.0, 1.0e5}) {
        const double sigma = theta_implicit_mhd::limited_slope(
            slope, slope, 1.0e-6 * slope, median);
        std::printf("median sigma(d, d)/d = %.8f (predicted %.8f)\n",
                    sigma / slope, predicted_defect);
        check(std::abs(sigma / slope - predicted_defect) < 3.0e-3,
              "median smooth-flow slope defect is not kappa/sqrt(2)");
        for (const double guard : {0.0, 1.0e-6 * slope, 0.3 * slope,
                                   10.0 * slope}) {
            check(std::abs(theta_implicit_mhd::limited_slope(
                      slope, slope, guard, albada) -
                           slope) <= 1.0e-15 * slope,
                  "van Albada sigma(d, d) != d");
        }
        check(theta_implicit_mhd::limited_slope(slope, -slope,
                                                1.0e-6 * slope, albada) == 0.0,
              "van Albada does not vanish at an extremum");
        check(std::abs(theta_implicit_mhd::limited_slope(
                  slope, -slope, 1.0e-6 * slope, median)) < 0.03 * slope,
              "median does not clip a smooth extremum");
    }

    // New-extremum bounds. The smoothed median's excursion is unbounded
    // relative to a VANISHING local jump (its kinks lie on rays through
    // the origin, so a homogeneous smoothing rounds the d_down = 0 ray
    // transversally) but bounded by kappa/4 of the stencil scale, which
    // is the measure that matters at a plateau edge. van Albada is
    // exactly zero on that ray and pays 0.10355 of the local jump on the
    // opposite-sign branch instead.
    double local_bound[3] = {0.0, 0.0, 0.0};
    double stencil_bound[3] = {0.0, 0.0, 0.0};
    const FluxParameters* modes[3] = {&median, &albada, &unlimited};
    for (int trial = 0; trial < 500000; ++trial) {
        const double q[4] = {uniform(rng), uniform(rng), uniform(rng),
                             uniform(rng)};
        const double low = std::min(q[1], q[2]);
        const double high = std::max(q[1], q[2]);
        const double reach = std::max(
            {std::abs(q[1] - q[0]), std::abs(q[2] - q[1]),
             std::abs(q[3] - q[2])});
        if (reach < 1.0e-12) {
            continue;
        }
        for (int mode = 0; mode < 3; ++mode) {
            double face_left = 0.0;
            double face_right = 0.0;
            theta_implicit_mhd::reconstruct_face_pair(
                q[0], q[1], q[2], q[3], *modes[mode], face_left, face_right);
            const double outside =
                std::max({0.0, low - face_left, face_left - high,
                          low - face_right, face_right - high});
            stencil_bound[mode] =
                std::max(stencil_bound[mode], outside / reach);
            if (high - low > 1.0e-3) {
                local_bound[mode] =
                    std::max(local_bound[mode], outside / (high - low));
            }
        }
    }
    const char* mode_names[3] = {"median   ", "vanalbada", "unlimited"};
    for (int mode = 0; mode < 3; ++mode) {
        std::printf(
            "%s new extrema: %10.6f of the local jump, %10.6f of the "
            "stencil scale\n",
            mode_names[mode], local_bound[mode], stencil_bound[mode]);
    }
    check(stencil_bound[0] < 0.5 * median.reconstruction_kappa,
          "median excursion exceeds kappa/2 of the stencil scale");
    check(local_bound[1] < 0.12 && stencil_bound[1] < 0.12,
          "van Albada excursion exceeds the 0.10355 bound");
    check(local_bound[2] > 0.4,
          "the unlimited reconstruction does not overshoot");

    // MIRROR SYMMETRY. Reversing the stencil must swap the two face
    // states and nothing else:
    //     reconstruct(q3, q2, q1, q0) = (face_right, face_left).
    // This is what makes the discrete operator commute with a mirror
    // plane -- the property test_rz_theta_implicit_mhd_mirror_full_
    // reconstruction checks on the real RZ solver, and the reason the
    // z_lo symmetry boundary still holds with the widened stencil. It
    // requires the limited slope to be ODD and SYMMETRIC in its two
    // one-sided differences and the guard to be a symmetric stencil
    // norm; agreement is to round-off rather than bitwise only because
    // the guard's sum of squares is accumulated in the opposite order.
    {
        double worst_mirror = 0.0;
        for (int trial = 0; trial < 200000; ++trial) {
            const double q[4] = {uniform(rng), uniform(rng), uniform(rng),
                                 uniform(rng)};
            const double scale =
                std::max({std::abs(q[0]), std::abs(q[1]), std::abs(q[2]),
                          std::abs(q[3]), 1.0e-300});
            for (int mode = 0; mode < 3; ++mode) {
                double face_left = 0.0;
                double face_right = 0.0;
                double mirror_left = 0.0;
                double mirror_right = 0.0;
                theta_implicit_mhd::reconstruct_face_pair(
                    q[0], q[1], q[2], q[3], *modes[mode], face_left,
                    face_right);
                theta_implicit_mhd::reconstruct_face_pair(
                    q[3], q[2], q[1], q[0], *modes[mode], mirror_left,
                    mirror_right);
                worst_mirror = std::max(
                    worst_mirror,
                    std::max(std::abs(mirror_left - face_right),
                             std::abs(mirror_right - face_left)) /
                        scale);
            }
        }
        std::printf("mirror symmetry: worst relative asymmetry %.3e\n",
                    worst_mirror);
        check(worst_mirror < 1.0e-15,
              "the reconstruction is not mirror symmetric");
    }

    // Smoothness across the minmod SELECTION kink. A hard minmod's
    // derivative jumps 1 -> 0 there; the smoothed median must not.
    {
        constexpr double slope = 1.0;
        constexpr double guard = 1.0e-6;
        constexpr double step = 1.0e-7;
        const auto derivative = [&] (const FluxParameters& parameters,
                                     const double offset) {
            const double base = slope + offset;
            return (theta_implicit_mhd::limited_slope(base + step, slope,
                                                      guard, parameters) -
                    theta_implicit_mhd::limited_slope(base - step, slope,
                                                      guard, parameters)) /
                   (2.0 * step);
        };
        const double median_below = derivative(median, -3.0e-3);
        const double median_above = derivative(median, 3.0e-3);
        const double albada_below = derivative(albada, -3.0e-3);
        const double albada_above = derivative(albada, 3.0e-3);
        std::printf(
            "d(sigma)/d(d_up) across d_up = d_down: median %.4f -> %.4f, "
            "vanalbada %.4f -> %.4f (hard minmod: 1 -> 0)\n",
            median_below, median_above, albada_below, albada_above);
        check(std::abs(median_below - median_above) < 0.85,
              "the median derivative jump is (near) the hard minmod kink");
        check(std::abs(albada_below - albada_above) < 0.05,
              "van Albada is not smooth at d_up = d_down");
    }

    // ------------------------------------------------------------------
    // B. FACE-STATE ADMISSIBILITY (the production constraint)
    // ------------------------------------------------------------------
    const double electron_energy_floor =
        electron_pressure_floor / (gamma_gas - 1.0);
    const double ion_internal_floor = ion_pressure_floor / (gamma_gas - 1.0);
    const int limiter_modes[3] = {
        theta_implicit_mhd::reconstruction_median,
        theta_implicit_mhd::reconstruction_vanalbada,
        theta_implicit_mhd::reconstruction_unlimited};
    const char* closure_names[4] = {"barotropic", "total_energy",
                                    "dual_energy", "cgl"};
    for (int closure = 0; closure < 4; ++closure) {
        for (const int mode : limiter_modes) {
            const FluxParameters parameters = make_parameters(mode, closure);
            double worst_density = 1.0e300;
            double worst_internal = 1.0e300;
            double worst_electron = 1.0e300;
            double worst_dual = 1.0e300;
            double worst_parallel = 1.0e300;
            double worst_perp = 1.0e300;
            double worst_recovered = 1.0e300;
            double worst_cancellation_ulps = 0.0;
            double worst_ion_energy = 1.0e300;
            for (int trial = 0; trial < 60000; ++trial) {
                for (int normal = 0; normal < 3; ++normal) {
                    CellState cells[4];
                    for (auto& cell : cells) {
                        cell = make_cell(rng, parameters);
                        // Give every cell the derived members the loaders
                        // would have built, so the reconstruction sees a
                        // consistent input state.
                        theta_implicit_mhd::rebuild_cell_state(
                            cell, 0.0, normal, parameters);
                    }
                    CellState left = cells[1];
                    CellState right = cells[2];
                    theta_implicit_mhd::reconstruct_face_states(
                        cells[0], cells[3], normal, parameters, left, right);
                    for (const CellState& face : {left, right}) {
                        check(finite_state(face),
                              "a reconstructed face state is not finite");
                        worst_density =
                            std::min(worst_density, face.density);
                        worst_electron = std::min(worst_electron,
                                                  face.electron_energy);
                        if (parameters.total_energy_closure) {
                            // Three separate statements, because they
                            // have three different strengths.
                            //  (i)  E_i itself: the CONSERVED variable
                            //       handed to the flux. Strictly
                            //       positive by construction (it is a
                            //       floored internal energy PLUS a
                            //       non-negative kinetic energy).
                            worst_ion_energy =
                                std::min(worst_ion_energy, face.ion_energy);
                            //  (ii) the RECOVERED internal energy every
                            //       pressure consumer reads, floored by
                            //       the same C^1 smooth max the loaders
                            //       use: exact.
                            worst_recovered =
                                std::min(worst_recovered, face.ion_internal);
                            // (iii) E_i - KE recomputed. This is the
                            //       construction identity, and it holds
                            //       only to the precision of the
                            //       cancellation: at the KE/internal
                            //       ratios this sweep reaches (up to
                            //       ~1e10, a 600 km/s cell riding its
                            //       ion pressure floor) the difference
                            //       carries ~1e-16 * E_i of absolute
                            //       noise, which is what the total-energy
                            //       closure is catastrophic about and
                            //       exactly why dual_energy exists. It is
                            //       measured in ulps of E_i rather than
                            //       asserted against the floor.
                            const double difference =
                                internal_ion_energy(face);
                            worst_internal =
                                std::min(worst_internal, difference);
                            if (difference < ion_internal_floor) {
                                worst_cancellation_ulps = std::max(
                                    worst_cancellation_ulps,
                                    (ion_internal_floor - difference) /
                                        (2.220446049250313e-16 *
                                         std::abs(face.ion_energy)));
                            }
                        }
                        if (parameters.dual_energy_closure) {
                            worst_dual = std::min(worst_dual,
                                                  face.ion_internal_energy);
                        }
                        if (parameters.cgl_closure) {
                            worst_parallel =
                                std::min(worst_parallel,
                                         face.ion_parallel_energy);
                            worst_perp =
                                std::min(worst_perp, face.ion_perp_energy);
                        }
                        // The pressures every flux consumer reads must be
                        // strictly positive whatever the closure.
                        check(face.ion_pressure > 0.0,
                              "a reconstructed ion pressure is not positive");
                        check(face.electron_pressure > 0.0,
                              "a reconstructed electron pressure is not "
                              "positive");
                    }
                }
            }
            std::printf(
                "closure %-12s mode %d: min rho/floor %.6f  min U_e/floor "
                "%.6f",
                closure_names[closure], mode, worst_density / density_floor,
                worst_electron / electron_energy_floor);
            if (parameters.total_energy_closure) {
                std::printf(
                    "  min E_i %.6e  min recovered U_i/floor %.6f"
                    "  worst (E_i - KE) shortfall %.1f ulp(E_i)",
                    worst_ion_energy, worst_recovered / ion_internal_floor,
                    worst_cancellation_ulps);
            }
            std::printf("\n");
            check(worst_density >= density_floor,
                  "a reconstructed density fell below mass_density_floor");
            check(worst_electron >= electron_energy_floor,
                  "a reconstructed U_e fell below its floor");
            if (parameters.total_energy_closure) {
                // THE production constraint: the reconstruction may never
                // hand the Riemann solver an inadmissible ion energy.
                check(worst_ion_energy > 0.0,
                      "a reconstructed ion energy is not strictly positive");
                check(worst_recovered >= ion_internal_floor,
                      "a reconstructed internal ion energy fell below the "
                      "ion internal-energy floor");
                // The construction identity, to the precision the
                // cancellation allows. A few ulps of E_i is round-off; a
                // large multiple would mean the floor is not actually
                // being applied where it is claimed.
                check(worst_cancellation_ulps < 64.0,
                      "the reconstructed E_i - KE identity is broken by "
                      "more than round-off");
            }
            if (parameters.dual_energy_closure) {
                check(worst_dual >= ion_internal_floor,
                      "a reconstructed U_i fell below its floor");
            }
            if (parameters.cgl_closure) {
                check(worst_parallel >= 0.5 * ion_pressure_floor,
                      "a reconstructed U_par fell below its floor");
                check(worst_perp >= ion_pressure_floor,
                      "a reconstructed U_perp fell below its floor");
            }
        }
    }

    // QUIESCENCE. Two statements of different strength, both needed.
    //
    // (i) The limited SLOPE of a uniform stencil is EXACTLY zero, so the
    //     scheme manufactures no face jump anywhere in a quiescent region
    //     -- the property the collapsed smooth_median3 form buys, and the
    //     one the literal nested expression of the reference code would lose (it returns
    //     -0.104 w there, an O(w) jump on every face of every halo
    //     pedestal, vacuum band, and held equilibrium).
    for (const int mode : limiter_modes) {
        const FluxParameters parameters = make_parameters(mode, 1);
        for (const double value : {-3.0e7, 0.0, 1.0e-30, 6.6e-7, 3.7e12}) {
            double face_left = 0.0;
            double face_right = 0.0;
            theta_implicit_mhd::reconstruct_face_pair(
                value, value, value, value, parameters, face_left, face_right);
            check(face_left == value && face_right == value,
                  "a uniform stencil does not reconstruct its own value "
                  "bitwise");
        }
    }
    // (ii) The full uniform-stencil face STATE is not bitwise identical to
    //      the donor -- it cannot be: the reconstruction rebuilds momentum
    //      as rho_face * u and passes every density and energy through the
    //      positivity floor, so a HEALTHY cell agrees to round-off and a
    //      floor-riding cell picks up the floor's documented inflation
    //      (width^2/4)/(value - floor). This checks the healthy case, which
    //      is what "reconstruction does not disturb a quiescent region"
    //      actually means.
    for (const int mode : limiter_modes) {
        for (int closure = 1; closure <= 2; ++closure) {
            const FluxParameters parameters = make_parameters(mode, closure);
            CellState cell{};
            cell.density = 1.6726e-7;
            cell.safe_density = cell.density;
            for (int component = 0; component < 3; ++component) {
                cell.ion_velocity[component] = 1.0e5 * (component + 1);
                cell.momentum[component] =
                    cell.density * cell.ion_velocity[component];
                cell.magnetic[component] = 2.0 * (component + 1);
            }
            cell.electron_energy = 1.0e4 * electron_energy_floor;
            cell.ion_internal_energy = 1.0e4 * ion_internal_floor;
            cell.old_ion_internal_energy = cell.ion_internal_energy;
            double kinetic = 0.0;
            for (int component = 0; component < 3; ++component) {
                kinetic += cell.momentum[component] * cell.momentum[component];
            }
            kinetic *= 0.5 / cell.safe_density;
            cell.ion_energy = 1.0e4 * ion_internal_floor + kinetic;
            cell.electron_velocity_normal = cell.ion_velocity[2];
            theta_implicit_mhd::rebuild_cell_state(cell, 0.0, 2, parameters);
            CellState left = cell;
            CellState right = cell;
            theta_implicit_mhd::reconstruct_face_states(cell, cell, 2,
                                                        parameters, left,
                                                        right);
            // Every deviation here traces to the positivity floor's
            // documented inflation, 2.5e-7 (floor/value)^2 relative --
            // which for the DENSITY (666 floors up in this cell) is
            // 2.2e-12, and which then propagates proportionally into
            // every specific-energy channel, since those are rebuilt as
            // rho_face * (energy/rho). 1e-11 is therefore a real bound
            // on "a quiescent region is not disturbed", not a fudge: a
            // genuine limiter bug shows up at 1e-3 or worse.
            double worst_relative = 0.0;
            for (const CellState& face : {left, right}) {
                const auto relative = [&] (const double got,
                                           const double want) {
                    const double value =
                        std::abs(got - want) / std::abs(want);
                    worst_relative = std::max(worst_relative, value);
                    return value;
                };
                check(relative(face.density, cell.density) <= 1.0e-11,
                      "a uniform healthy stencil moves the density");
                check(relative(face.ion_energy, cell.ion_energy) <= 1.0e-11,
                      "a uniform healthy stencil moves E_i");
                check(relative(face.electron_energy, cell.electron_energy) <=
                          1.0e-11,
                      "a uniform healthy stencil moves U_e");
                check(relative(face.ion_pressure, cell.ion_pressure) <=
                          1.0e-9,
                      "a uniform healthy stencil moves the ion pressure");
            }
            if (mode == limiter_modes[0] && closure == 1) {
                std::printf(
                    "uniform healthy stencil: worst relative face-state "
                    "deviation %.3e (floor inflation bound %.3e)\n",
                    worst_relative,
                    0.25 *
                        theta_implicit_mhd::
                            reconstruction_floor_width_fraction *
                        theta_implicit_mhd::
                            reconstruction_floor_width_fraction *
                        (density_floor / cell.density) *
                        (density_floor / cell.density));
            }
        }
    }

    std::printf(s_failures ? "\n%d CHECK(S) FAILED\n" : "\nALL CHECKS PASSED\n",
                s_failures);
    return s_failures ? 1 : 0;
}
