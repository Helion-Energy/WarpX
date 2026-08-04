/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
/*
 * Standalone host-side verification driver for the HLLD Riemann kernel in
 * ThetaImplicitMHD_K.H (theta_implicit_mhd::hlld_flux). It is deliberately
 * NOT part of the production CMake build: compile and run it directly with
 * any AMReX header set whose generated AMReX_Config.H is available. With
 * the repository's CPU test build (build-tests, OpenMP + MPI enabled):
 *
 *   g++ -std=c++20 -O2 -fopenmp \
 *       -I<repo>/build-tests/_deps/fetchedamrex-src/Src/Base \
 *       -I<repo>/build-tests/_deps/fetchedamrex-build \
 *       -I/usr/local/openmpi5/include \
 *       <repo>/Source/FieldSolver/ImplicitSolvers/ThetaImplicitMHDKernelTest.cpp \
 *       -o theta_implicit_mhd_kernel_test
 *   ./theta_implicit_mhd_kernel_test
 *
 * (-fopenmp and the MPI include dir are only needed because that AMReX
 * config was built with OpenMP and MPI; only AMReX headers are used, no
 * AMReX library is linked.)
 *
 * Verified invariant groups (all must PASS):
 *  1. Consistency: F(U, U) equals the exact physical flux of U on every
 *     channel to machine precision (random states, mu0 = 1 and SI mu0).
 *  2. Hydro limit: with B = 0 and hard switches (kappa_signal =
 *     kappa_contact = kappa_denominator = 0) hlld_flux reproduces
 *     hllc_flux to roundoff.
 *  3. B_n -> 0 limit: at bn_face = 0 with finite B_t the flux equals the
 *     HLLC-like two-state fan with magnetic pressure in p_T and B_t
 *     advected with the contact; continuity across bn_face = +-1e-12.
 *  4. Exact preservation of a static, total-pressure-balanced tangential
 *     discontinuity (the FRC-separatrix invariant).
 *  5. Rankine-Hugoniot: a fast MHD shock built from the MK05 jump
 *     relations satisfies F(U_L) - F(U_R) = s (U_L - U_R) channel-wise.
 *  6. Smoothness: central finite differences of every output channel with
 *     respect to every input agree between 1e-6 and 1e-8 relative
 *     perturbations, including across S_M = 0 and B_n = 0.
 *  7. Positivity: p_T* > 0 and star densities > 0 over a randomized sweep
 *     of physically bounded states; worst margins reported.
 */

#include "ThetaImplicitMHD_K.H"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace timhd = theta_implicit_mhd;
using amrex::Real;

namespace {

constexpr Real pi = 3.14159265358979323846;
constexpr Real mu0_si = 4.0e-7 * pi; // SI vacuum permeability (pre-2019)

int g_failures = 0;

void report (const std::string& name, const bool pass,
             const std::string& detail)
{
    std::printf("%-52s %s  %s\n", name.c_str(), pass ? "PASS" : "FAIL",
                detail.c_str());
    if (!pass) {
        ++g_failures;
    }
}

timhd::FluxParameters make_params (const Real mu0 = 1.0)
{
    timhd::FluxParameters parameters{};
    parameters.density_floor = 1.0e-8;
    parameters.charge_to_mass = 1.0;
    parameters.gamma_e = 5.0 / 3.0;
    parameters.gamma_i = 5.0 / 3.0;
    parameters.reference_density = 1.0;
    parameters.reference_ion_pressure = 0.0;
    parameters.electron_pressure_floor = 1.0e-12;
    parameters.ion_pressure_floor = 1.0e-12;
    parameters.include_hall_term = false;
    parameters.use_rusanov = false;
    parameters.use_hllc = false;
    parameters.total_energy_closure = true;
    parameters.barotropic_signal_speeds = false;
    parameters.contact_blend = 0.0;
    parameters.theta = 1.0;
    parameters.mu0 = mu0;
    // hlld_kappa_* keep their in-struct defaults (0.05).
    return parameters;
}

timhd::FluxParameters make_hard_params (const Real mu0 = 1.0)
{
    // Hard-switch configuration: recovers exact min/max signal bounds,
    // hard upwind region selection, and unguarded star denominators.
    // kappa_bn must stay positive (chi and sign(B_n) need eps_B > 0).
    timhd::FluxParameters parameters = make_params(mu0);
    parameters.hlld_kappa_signal = 0.0;
    parameters.hlld_kappa_contact = 0.0;
    parameters.hlld_kappa_denominator = 0.0;
    return parameters;
}

// One cell's conserved inputs; primitive helpers fill the energies.
struct HostCell {
    Real rho = 1.0;
    Real mom[3] = {0.0, 0.0, 0.0};
    Real ue = 0.0;
    Real ei = 0.0;
    Real current[3] = {0.0, 0.0, 0.0};
    Real b[3] = {0.0, 0.0, 0.0};
};

HostCell make_cell (const Real rho, const Real ux, const Real uy,
                    const Real uz, const Real p_ion, const Real p_electron,
                    const Real bx, const Real by, const Real bz,
                    const timhd::FluxParameters& parameters)
{
    HostCell cell;
    cell.rho = rho;
    cell.mom[0] = rho * ux;
    cell.mom[1] = rho * uy;
    cell.mom[2] = rho * uz;
    cell.ue = p_electron / (parameters.gamma_e - 1.0);
    cell.ei = p_ion / (parameters.gamma_i - 1.0) +
              0.5 * rho * (ux * ux + uy * uy + uz * uz);
    cell.b[0] = bx;
    cell.b[1] = by;
    cell.b[2] = bz;
    return cell;
}

timhd::CellState load_state (const HostCell& cell, const int direction,
                             const timhd::FluxParameters& parameters)
{
    const amrex::Dim3 lo{0, 0, 0};
    const amrex::Dim3 hi{1, 1, 1};
    const amrex::Array4<const Real> rho(&cell.rho, lo, hi, 1);
    const amrex::Array4<const Real> mom(cell.mom, lo, hi, 3);
    const amrex::Array4<const Real> ue(&cell.ue, lo, hi, 1);
    const amrex::Array4<const Real> ei(&cell.ei, lo, hi, 1);
    const amrex::Array4<const Real> current(cell.current, lo, hi, 3);
    const amrex::Array4<const Real> magnetic(cell.b, lo, hi, 3);
    return timhd::load_cell_state_hlld(rho, mom, ue, ei, current, magnetic,
                                       0, 0, 0, direction, parameters);
}

// Flat channel vector for comparisons; layout documented by names().
constexpr int n_channels = 14;

using Channels = std::array<Real, n_channels>;

const char* channel_name (const int channel)
{
    static const char* names[n_channels] = {
        "mass",       "momentum_x", "momentum_y",  "momentum_z",
        "elec_energy", "ion_energy", "elec_velocity", "mom_mag_x",
        "mom_mag_y",  "mom_mag_z",  "induction_t1", "induction_t2",
        "signal_left", "signal_right"};
    return names[channel];
}

Channels flatten (const timhd::FaceFlux& flux)
{
    return {flux.mass,          flux.momentum[0],
            flux.momentum[1],   flux.momentum[2],
            flux.electron_energy, flux.ion_energy,
            flux.electron_velocity, flux.momentum_magnetic[0],
            flux.momentum_magnetic[1], flux.momentum_magnetic[2],
            flux.induction_t1,  flux.induction_t2,
            flux.signal_left,   flux.signal_right};
}

Channels hlld_channels (const HostCell& left_cell, const HostCell& right_cell,
                        const Real bn_face, const int direction,
                        const timhd::FluxParameters& parameters)
{
    const timhd::CellState left = load_state(left_cell, direction, parameters);
    const timhd::CellState right =
        load_state(right_cell, direction, parameters);
    return flatten(
        timhd::hlld_flux(left, right, bn_face, direction, parameters));
}

// Exact physical flux of a single loaded state (independent formulas).
Channels physical_flux (const timhd::CellState& state, const Real bn_face,
                        const int direction,
                        const timhd::FluxParameters& parameters)
{
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const Real inv_mu0 = 1.0 / parameters.mu0;
    Real field[3];
    field[direction] = bn_face;
    field[tangent1] = state.magnetic[tangent1];
    field[tangent2] = state.magnetic[tangent2];
    const Real magnetic_pressure =
        0.5 * (field[0] * field[0] + field[1] * field[1] +
               field[2] * field[2]) * inv_mu0;
    const Real p_total =
        state.ion_pressure + state.electron_pressure + magnetic_pressure;
    const Real u_normal = state.ion_velocity[direction];

    Channels flux{};
    flux[0] = state.momentum[direction];
    for (int component = 0; component < 3; ++component) {
        flux[1 + component] =
            state.momentum[direction] * state.ion_velocity[component] +
            (component == direction ? p_total : 0.0) -
            bn_face * field[component] * inv_mu0;
        flux[7 + component] =
            (component == direction ? magnetic_pressure : 0.0) -
            bn_face * field[component] * inv_mu0;
    }
    flux[4] = state.electron_energy * u_normal;
    flux[5] = u_normal * (state.ion_energy + state.ion_pressure +
                          state.electron_pressure);
    flux[6] = state.electron_velocity_normal;
    flux[10] = u_normal * state.magnetic[tangent1] -
               state.ion_velocity[tangent1] * bn_face;
    flux[11] = u_normal * state.magnetic[tangent2] -
               state.ion_velocity[tangent2] * bn_face;
    flux[12] = 0.0; // signal bounds have no single-state counterpart
    flux[13] = 0.0;
    return flux;
}

// ---------------------------------------------------------------------
// 1. Consistency: F(U, U) = exact physical flux, all channels.
// ---------------------------------------------------------------------
void test_consistency ()
{
    std::mt19937 rng(20260803);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);
    Real worst = 0.0;
    int worst_channel = 0;

    for (const Real mu0 : {1.0, mu0_si}) {
        const timhd::FluxParameters parameters = make_params(mu0);
        const Real sqrt_mu0 = std::sqrt(mu0);
        for (int sample = 0; sample < 200; ++sample) {
            const Real rho = 0.1 + 4.9 * uniform(rng);
            const Real p_ion = 0.1 + 4.9 * uniform(rng);
            const Real p_electron = 0.05 + 2.95 * uniform(rng);
            const Real ux = 4.0 * (uniform(rng) - 0.5);
            const Real uy = 4.0 * (uniform(rng) - 0.5);
            const Real uz = 4.0 * (uniform(rng) - 0.5);
            // b-unit field components O(1), converted to physical B.
            const Real bx = 4.0 * (uniform(rng) - 0.5) * sqrt_mu0;
            const Real by = 4.0 * (uniform(rng) - 0.5) * sqrt_mu0;
            const Real bz = 4.0 * (uniform(rng) - 0.5) * sqrt_mu0;
            const HostCell cell = make_cell(rho, ux, uy, uz, p_ion,
                                            p_electron, bx, by, bz,
                                            parameters);
            for (int direction = 0; direction < 3; ++direction) {
                const timhd::CellState state =
                    load_state(cell, direction, parameters);
                const Real bn_face = cell.b[direction];
                const Channels flux =
                    hlld_channels(cell, cell, bn_face, direction, parameters);
                const Channels expected =
                    physical_flux(state, bn_face, direction, parameters);
                for (int channel = 0; channel < 12; ++channel) {
                    // Induction channels are O(sqrt(mu0)); compare in
                    // b-units so the relative scale stays O(1).
                    const Real unit =
                        (channel >= 10 && channel <= 11) ? sqrt_mu0 : 1.0;
                    const Real error =
                        std::abs(flux[channel] - expected[channel]) /
                        (unit * (1.0 + std::abs(expected[channel] / unit)));
                    if (error > worst) {
                        worst = error;
                        worst_channel = channel;
                    }
                }
                // Signal-bound sanity: the fan must bracket the flow.
                if (!(flux[12] < state.ion_velocity[direction] &&
                      flux[13] > state.ion_velocity[direction])) {
                    worst = 1.0;
                }
            }
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail),
                  "max rel error %.3e (channel %s), tol 1e-12", worst,
                  channel_name(worst_channel));
    report("1. consistency F(U,U) = physical flux", worst < 1.0e-12, detail);
}

// ---------------------------------------------------------------------
// 2. Hydro limit: B = 0 with hard switches reproduces hllc_flux.
// ---------------------------------------------------------------------
void test_hydro_limit ()
{
    std::mt19937 rng(20260804);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);
    const timhd::FluxParameters parameters = make_hard_params(1.0);
    Real worst = 0.0;
    int worst_channel = 0;

    for (int sample = 0; sample < 200; ++sample) {
        HostCell cells[2];
        for (HostCell& cell : cells) {
            cell = make_cell(0.1 + 4.9 * uniform(rng),
                             4.0 * (uniform(rng) - 0.5),
                             4.0 * (uniform(rng) - 0.5),
                             4.0 * (uniform(rng) - 0.5),
                             0.1 + 4.9 * uniform(rng),
                             0.05 + 2.95 * uniform(rng), 0.0, 0.0, 0.0,
                             parameters);
        }
        for (int direction = 0; direction < 3; ++direction) {
            const timhd::CellState left =
                load_state(cells[0], direction, parameters);
            const timhd::CellState right =
                load_state(cells[1], direction, parameters);
            const timhd::FaceFlux hlld =
                timhd::hlld_flux(left, right, 0.0, direction, parameters);
            const timhd::FaceFlux hllc =
                timhd::hllc_flux(left, right, direction, parameters);
            const Real errors[6] = {
                std::abs(hlld.mass - hllc.mass),
                std::abs(hlld.momentum[0] - hllc.momentum[0]),
                std::abs(hlld.momentum[1] - hllc.momentum[1]),
                std::abs(hlld.momentum[2] - hllc.momentum[2]),
                std::abs(hlld.electron_energy - hllc.electron_energy),
                std::abs(hlld.ion_energy - hllc.ion_energy)};
            const Real scales[6] = {
                std::abs(hllc.mass),          std::abs(hllc.momentum[0]),
                std::abs(hllc.momentum[1]),   std::abs(hllc.momentum[2]),
                std::abs(hllc.electron_energy), std::abs(hllc.ion_energy)};
            for (int channel = 0; channel < 6; ++channel) {
                const Real error = errors[channel] / (1.0 + scales[channel]);
                if (error > worst) {
                    worst = error;
                    worst_channel = channel;
                }
            }
        }
    }
    char detail[160];
    std::snprintf(
        detail, sizeof(detail),
        "max rel error %.3e vs hllc_flux (channel %d), tol 1e-12", worst,
        worst_channel);
    report("2. hydro limit B=0 reproduces hllc_flux", worst < 1.0e-12,
           detail);
}

// ---------------------------------------------------------------------
// 3. B_n -> 0: HLLC-like two-state fan with magnetic pressure in p_T and
//    B_t advected with the contact (side-complete reference), plus
//    continuity across bn_face = +-1e-12 at production smoothing.
// ---------------------------------------------------------------------
Channels reference_bn0 (const timhd::CellState& left,
                        const timhd::CellState& right, const int direction)
{
    // Expected form at B_n = 0 (mu0 = 1): hard HLLC fan {S_L, S_M, S_R}
    // built on p_T = p_i + p_e + |B_t|^2/2 with signal speeds from the
    // fast magnetosonic speed; B_t and U_e advected with the contact
    // (compression factor identical to the density's).
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const Real u_left = left.ion_velocity[direction];
    const Real u_right = right.ion_velocity[direction];
    const Real bt1_left = left.magnetic[tangent1];
    const Real bt2_left = left.magnetic[tangent2];
    const Real bt1_right = right.magnetic[tangent1];
    const Real bt2_right = right.magnetic[tangent2];
    const Real p_total_left = left.ion_pressure + left.electron_pressure +
                              0.5 * (bt1_left * bt1_left +
                                     bt2_left * bt2_left);
    const Real p_total_right = right.ion_pressure +
                               right.electron_pressure +
                               0.5 * (bt1_right * bt1_right +
                                      bt2_right * bt2_right);
    const Real s_left = std::min(u_left - left.fast_speed,
                                 u_right - right.fast_speed);
    const Real s_right = std::max(u_left + left.fast_speed,
                                  u_right + right.fast_speed);
    const Real m_left = left.safe_density * (s_left - u_left);
    const Real m_right = right.safe_density * (s_right - u_right);
    const Real s_mid = (p_total_right - p_total_left + m_left * u_left -
                        m_right * u_right) /
                       (m_left - m_right);

    const bool from_left = (s_mid >= 0.0);
    const timhd::CellState& up = from_left ? left : right;
    const Real s_outer = from_left ? s_left : s_right;
    const bool subsonic = from_left ? (s_left < 0.0) : (s_right > 0.0);
    const Real u_up = up.ion_velocity[direction];
    const Real bt1_up = up.magnetic[tangent1];
    const Real bt2_up = up.magnetic[tangent2];
    const Real p_total_up = from_left ? p_total_left : p_total_right;

    Channels flux{};
    flux[0] = up.momentum[direction];
    flux[1 + direction] = up.momentum[direction] * u_up + p_total_up;
    flux[1 + tangent1] =
        up.momentum[direction] * up.ion_velocity[tangent1];
    flux[1 + tangent2] =
        up.momentum[direction] * up.ion_velocity[tangent2];
    flux[4] = up.electron_energy * u_up;
    flux[10] = u_up * bt1_up;
    flux[11] = u_up * bt2_up;
    Real bt1_face = bt1_up;
    Real bt2_face = bt2_up;
    if (subsonic) {
        const Real fraction = (s_outer - u_up) / (s_outer - s_mid);
        flux[0] += s_outer * (fraction - 1.0) * up.density;
        flux[1 + direction] +=
            s_outer * (fraction * up.density * s_mid -
                       up.momentum[direction]);
        flux[1 + tangent1] +=
            s_outer * (fraction - 1.0) * up.momentum[tangent1];
        flux[1 + tangent2] +=
            s_outer * (fraction - 1.0) * up.momentum[tangent2];
        flux[4] += s_outer * (fraction - 1.0) * up.electron_energy;
        flux[10] += s_outer * (fraction - 1.0) * bt1_up;
        flux[11] += s_outer * (fraction - 1.0) * bt2_up;
        bt1_face = fraction * bt1_up;
        bt2_face = fraction * bt2_up;
    }
    // Ion energy: LLF enthalpy channel with the fast-speed alpha.
    const Real alpha =
        std::max(left.fast_wave_speed, right.fast_wave_speed);
    flux[5] = 0.5 * (u_left * (left.ion_energy + left.ion_pressure +
                               left.electron_pressure) +
                     u_right * (right.ion_energy + right.ion_pressure +
                                right.electron_pressure) -
                     alpha * (right.ion_energy - left.ion_energy));
    flux[6] = 0.5 * (left.electron_velocity_normal +
                     right.electron_velocity_normal);
    // Maxwell stress of the region containing the interface.
    flux[7 + direction] =
        0.5 * (bt1_face * bt1_face + bt2_face * bt2_face);
    flux[7 + tangent1] = 0.0;
    flux[7 + tangent2] = 0.0;
    return flux;
}

void test_bn_zero_limit ()
{
    std::mt19937 rng(20260805);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);
    const timhd::FluxParameters hard = make_hard_params(1.0);
    const timhd::FluxParameters smooth = make_params(1.0);
    Real worst_exact = 0.0;
    int worst_channel = 0;
    Real worst_continuity = 0.0;

    for (int sample = 0; sample < 200; ++sample) {
        HostCell cells[2];
        for (HostCell& cell : cells) {
            cell = make_cell(0.1 + 4.9 * uniform(rng),
                             4.0 * (uniform(rng) - 0.5),
                             4.0 * (uniform(rng) - 0.5),
                             4.0 * (uniform(rng) - 0.5),
                             0.1 + 4.9 * uniform(rng),
                             0.05 + 2.95 * uniform(rng),
                             3.0 * (uniform(rng) - 0.5),
                             3.0 * (uniform(rng) - 0.5),
                             3.0 * (uniform(rng) - 0.5), hard);
        }
        for (int direction = 0; direction < 3; ++direction) {
            cells[0].b[direction] = 0.0;
            cells[1].b[direction] = 0.0;
            const timhd::CellState left =
                load_state(cells[0], direction, hard);
            const timhd::CellState right =
                load_state(cells[1], direction, hard);
            const Channels flux = flatten(
                timhd::hlld_flux(left, right, 0.0, direction, hard));
            const Channels expected = reference_bn0(left, right, direction);
            for (int channel = 0; channel < 12; ++channel) {
                const Real error =
                    std::abs(flux[channel] - expected[channel]) /
                    (1.0 + std::abs(expected[channel]));
                if (error > worst_exact) {
                    worst_exact = error;
                    worst_channel = channel;
                }
            }
            // Continuity at production smoothing across bn_face = 0.
            const Channels center =
                hlld_channels(cells[0], cells[1], 0.0, direction, smooth);
            for (const Real bn_face : {1.0e-12, -1.0e-12}) {
                const Channels shifted = hlld_channels(
                    cells[0], cells[1], bn_face, direction, smooth);
                for (int channel = 0; channel < 12; ++channel) {
                    const Real jump =
                        std::abs(shifted[channel] - center[channel]) /
                        (1.0 + std::abs(center[channel]));
                    worst_continuity = std::max(worst_continuity, jump);
                }
            }
        }
    }
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "max rel error %.3e vs two-state fan (channel %s), "
                  "continuity %.3e across bn=+-1e-12",
                  worst_exact, channel_name(worst_channel),
                  worst_continuity);
    report("3. B_n -> 0 limit and continuity",
           worst_exact < 1.0e-11 && worst_continuity < 1.0e-9, detail);
}

// ---------------------------------------------------------------------
// 4. Static, total-pressure-balanced tangential discontinuity.
// ---------------------------------------------------------------------
void test_static_tangential_discontinuity ()
{
    const Real mu0 = mu0_si;
    const timhd::FluxParameters parameters = make_params(mu0);
    const Real sqrt_mu0 = std::sqrt(mu0);

    // u = 0, B_n = 0, rho and B_t jump; p_i continuous (so E_i is too);
    // p_e balances the magnetic pressure jump:
    // p_i + p_e + |B_t|^2 / (2 mu0) continuous.
    const Real p_ion = 1.0;
    const Real bt_left[2] = {0.8, -0.4};   // b-units
    const Real bt_right[2] = {0.3, 0.55};
    const Real p_electron_left = 0.5;
    const Real p_electron_right =
        p_electron_left +
        0.5 * (bt_left[0] * bt_left[0] + bt_left[1] * bt_left[1] -
               bt_right[0] * bt_right[0] - bt_right[1] * bt_right[1]);
    const Real p_total = p_ion + p_electron_left +
                         0.5 * (bt_left[0] * bt_left[0] +
                                bt_left[1] * bt_left[1]);

    Real worst = 0.0;
    Real worst_pressure = 0.0;
    for (int direction = 0; direction < 3; ++direction) {
        const int tangent1 = (direction + 1) % 3;
        const int tangent2 = (direction + 2) % 3;
        Real b_left[3] = {0.0, 0.0, 0.0};
        Real b_right[3] = {0.0, 0.0, 0.0};
        b_left[tangent1] = bt_left[0] * sqrt_mu0;
        b_left[tangent2] = bt_left[1] * sqrt_mu0;
        b_right[tangent1] = bt_right[0] * sqrt_mu0;
        b_right[tangent2] = bt_right[1] * sqrt_mu0;
        const HostCell left_cell =
            make_cell(1.0, 0.0, 0.0, 0.0, p_ion, p_electron_left, b_left[0],
                      b_left[1], b_left[2], parameters);
        const HostCell right_cell =
            make_cell(2.5, 0.0, 0.0, 0.0, p_ion, p_electron_right,
                      b_right[0], b_right[1], b_right[2], parameters);
        const Channels flux =
            hlld_channels(left_cell, right_cell, 0.0, direction, parameters);

        // Every channel except the trivial normal pressure-momentum flux
        // (and its Maxwell part) must vanish to machine precision.
        for (int channel = 0; channel < 12; ++channel) {
            if (channel == 1 + direction || channel == 7 + direction) {
                continue;
            }
            worst = std::max(worst, std::abs(flux[channel]) / p_total);
        }
        worst_pressure = std::max(
            worst_pressure,
            std::abs(flux[1 + direction] - p_total) / p_total);
    }
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "max |flux|/p_T %.3e (tol 1e-13), "
                  "|F_mom_n - p_T|/p_T %.3e",
                  worst, worst_pressure);
    report("4. static tangential discontinuity exact",
           worst < 1.0e-13 && worst_pressure < 1.0e-13, detail);
}

// ---------------------------------------------------------------------
// 5. Rankine-Hugoniot for a fast MHD shock from the MK05 jump relations.
// ---------------------------------------------------------------------
struct ShockSide {
    Real rho, u, vt, p, bt; // shock-frame normal u, tangential magnitude
};

// Downstream state for compression ratio X (b-units, mu0 = 1, gamma law).
ShockSide downstream_of (const ShockSide& up, const Real bn, const Real X)
{
    ShockSide down;
    down.rho = up.rho * X;
    const Real mass_flux = up.rho * up.u;
    down.u = mass_flux / down.rho;
    // Tangential momentum and induction jump conditions:
    //   j v - bn bt = const, u bt - v bn = const.
    const Real momentum_const = mass_flux * up.vt - bn * up.bt;
    const Real induction_const = up.u * up.bt - up.vt * bn;
    const Real det = mass_flux * down.u - bn * bn;
    down.vt = (down.u * momentum_const + bn * induction_const) / det;
    down.bt = (bn * momentum_const + mass_flux * induction_const) / det;
    // Normal momentum: j u + p + bt^2/2 = const (bn^2 terms cancel).
    down.p = up.p + mass_flux * (up.u - down.u) +
             0.5 * (up.bt * up.bt - down.bt * down.bt);
    return down;
}

Real energy_flux (const ShockSide& side, const Real bn, const Real gamma)
{
    const Real total_energy =
        side.p / (gamma - 1.0) +
        0.5 * side.rho * (side.u * side.u + side.vt * side.vt) +
        0.5 * (bn * bn + side.bt * side.bt);
    const Real p_total = side.p + 0.5 * (bn * bn + side.bt * side.bt);
    return side.u * (total_energy + p_total) -
           bn * (side.u * bn + side.vt * side.bt);
}

void test_rankine_hugoniot ()
{
    const Real mu0 = mu0_si;
    const timhd::FluxParameters parameters = make_params(mu0);
    const Real sqrt_mu0 = std::sqrt(mu0);
    const Real gamma = parameters.gamma_i;

    // Upstream state (b-units, shock frame).
    ShockSide up;
    up.rho = 1.0;
    up.p = 1.0;
    up.vt = 0.0;
    const Real bn = 0.7;
    up.bt = 0.6;
    const Real sound2 = gamma * up.p / up.rho;
    const Real alfven2 = (bn * bn + up.bt * up.bt) / up.rho;
    const Real fast2 =
        0.5 * (sound2 + alfven2 +
               std::sqrt((sound2 + alfven2) * (sound2 + alfven2) -
                         4.0 * sound2 * bn * bn / up.rho));
    up.u = 2.0 * std::sqrt(fast2); // fast Mach 2 inflow

    // Solve the energy jump condition for the compression ratio X by
    // bracketing the nontrivial root (X = 1 is the trivial solution).
    const auto residual = [&] (const Real X) {
        const ShockSide down = downstream_of(up, bn, X);
        return energy_flux(up, bn, gamma) - energy_flux(down, bn, gamma);
    };
    Real x_low = 1.01;
    Real x_high = x_low;
    const Real x_max = (gamma + 1.0) / (gamma - 1.0) - 1.0e-3;
    Real r_low = residual(x_low);
    bool bracketed = false;
    for (int step = 1; step <= 4000; ++step) {
        x_high = 1.01 + (x_max - 1.01) * step / 4000.0;
        const Real r_high = residual(x_high);
        if (r_low * r_high <= 0.0) {
            bracketed = true;
            break;
        }
        x_low = x_high;
        r_low = r_high;
    }
    for (int iteration = 0; iteration < 200 && bracketed; ++iteration) {
        const Real x_mid = 0.5 * (x_low + x_high);
        if (r_low * residual(x_mid) <= 0.0) {
            x_high = x_mid;
        } else {
            x_low = x_mid;
        }
    }
    const Real X = 0.5 * (x_low + x_high);
    const ShockSide down = downstream_of(up, bn, X);

    // Lab frame: shock speed s; rotate the tangential direction so both
    // t1 and t2 channels are exercised. Check RH channel-wise on the
    // conservative kernel channels (mass, momentum, induction; E_i is
    // the non-conservative ion enthalpy channel and U_e is zero here).
    const Real shock_speed = 0.35;
    const Real phi = 30.0 * pi / 180.0;
    Real worst = 0.0;
    bool finite_solution = bracketed && X > 1.05 && down.p > 0.0;
    for (const int direction : {0, 2}) {
        const int tangent1 = (direction + 1) % 3;
        const int tangent2 = (direction + 2) % 3;
        HostCell cells[2];
        Real u_states[2][3];
        Real b_states[2][3];
        const ShockSide* sides[2] = {&up, &down};
        for (int index = 0; index < 2; ++index) {
            const ShockSide& side = *sides[index];
            u_states[index][direction] = side.u + shock_speed;
            u_states[index][tangent1] = side.vt * std::cos(phi);
            u_states[index][tangent2] = side.vt * std::sin(phi);
            b_states[index][direction] = bn * sqrt_mu0;
            b_states[index][tangent1] = side.bt * std::cos(phi) * sqrt_mu0;
            b_states[index][tangent2] = side.bt * std::sin(phi) * sqrt_mu0;
            cells[index] = make_cell(
                side.rho, u_states[index][0], u_states[index][1],
                u_states[index][2], side.p, 0.0, b_states[index][0],
                b_states[index][1], b_states[index][2], parameters);
        }
        const Channels flux_left = hlld_channels(
            cells[0], cells[0], bn * sqrt_mu0, direction, parameters);
        const Channels flux_right = hlld_channels(
            cells[1], cells[1], bn * sqrt_mu0, direction, parameters);

        // Conserved densities (physical units).
        const Real state_left[6] = {
            up.rho, up.rho * u_states[0][0], up.rho * u_states[0][1],
            up.rho * u_states[0][2], b_states[0][tangent1],
            b_states[0][tangent2]};
        const Real state_right[6] = {
            down.rho, down.rho * u_states[1][0], down.rho * u_states[1][1],
            down.rho * u_states[1][2], b_states[1][tangent1],
            b_states[1][tangent2]};
        const int flux_index[6] = {0, 1, 2, 3, 10, 11};
        for (int channel = 0; channel < 6; ++channel) {
            const Real unit = (channel >= 4) ? sqrt_mu0 : 1.0;
            const Real defect =
                (flux_left[flux_index[channel]] -
                 flux_right[flux_index[channel]]) -
                shock_speed * (state_left[channel] - state_right[channel]);
            worst = std::max(worst, std::abs(defect) / unit);
        }
    }
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "X = %.6f, max |dF - s dU| %.3e (tol 1e-10)", X, worst);
    report("5. Rankine-Hugoniot fast shock", finite_solution &&
           worst < 1.0e-10, detail);
}

// ---------------------------------------------------------------------
// 6. Smoothness: FD derivatives at 1e-6 and 1e-8 relative steps agree.
// ---------------------------------------------------------------------
void test_smoothness ()
{
    const timhd::FluxParameters parameters = make_params(1.0);
    const int direction = 2;
    Real worst = 0.0;
    char worst_label[96] = "none";

    for (const Real bn_center : {0.0, 0.03, 0.3}) {
        // Near-stagnant, near-balanced pair; the two sides differ so the
        // remaining hard max() in the LLF alpha is not probed at a tie.
        HostCell base_left =
            make_cell(1.0, 1.0e-3, 2.0e-3, -1.5e-3, 1.0, 0.5, 0.5, -0.3,
                      bn_center, parameters);
        HostCell base_right =
            make_cell(1.15, -2.0e-3, 1.0e-3, 2.5e-3, 0.95, 0.55, 0.45,
                      -0.35, bn_center, parameters);

        // Inputs: 9 per side (rho, mom[3], ue, ei, B[3]) plus bn_face.
        for (int input = 0; input < 19; ++input) {
            const auto evaluate = [&] (const Real delta) {
                HostCell left = base_left;
                HostCell right = base_right;
                Real bn_face = bn_center;
                const int side_input = input % 9;
                HostCell& cell = (input < 9) ? left : right;
                if (input == 18) {
                    bn_face += delta;
                } else if (side_input == 0) {
                    cell.rho += delta;
                } else if (side_input <= 3) {
                    cell.mom[side_input - 1] += delta;
                } else if (side_input == 4) {
                    cell.ue += delta;
                } else if (side_input == 5) {
                    cell.ei += delta;
                } else {
                    cell.b[side_input - 6] += delta;
                }
                return hlld_channels(left, right, bn_face, direction,
                                     parameters);
            };
            const Real scale = 1.0; // all inputs are O(1) here
            for (int channel = 0; channel < n_channels; ++channel) {
                Real derivative[2];
                const Real steps[2] = {1.0e-6, 1.0e-8};
                for (int level = 0; level < 2; ++level) {
                    const Real delta = steps[level] * scale;
                    const Channels plus = evaluate(delta);
                    const Channels minus = evaluate(-delta);
                    derivative[level] =
                        (plus[channel] - minus[channel]) / (2.0 * delta);
                }
                const Real mismatch =
                    std::abs(derivative[0] - derivative[1]) /
                    (1.0e-4 * std::max(std::abs(derivative[0]),
                                       std::abs(derivative[1])) +
                     1.0e-6);
                if (mismatch > worst) {
                    worst = mismatch;
                    std::snprintf(worst_label, sizeof(worst_label),
                                  "bn=%.2f input %d channel %s", bn_center,
                                  input, channel_name(channel));
                }
            }
        }
    }
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "worst FD(1e-6) vs FD(1e-8) mismatch %.3f x tol (%s)",
                  worst, worst_label);
    report("6. smoothness across S_M = 0 and B_n = 0", worst < 1.0,
           detail);
}

// ---------------------------------------------------------------------
// 7. Positivity of p_T* and the star densities over a randomized sweep.
// ---------------------------------------------------------------------
struct FanDiagnostics {
    Real p_total_star;
    Real rho_star_left;
    Real rho_star_right;
};

FanDiagnostics wave_fan (const timhd::CellState& left,
                         const timhd::CellState& right, const Real bn_face,
                         const int direction,
                         const timhd::FluxParameters& parameters)
{
    // Independent recomputation of the kernel's wave structure, including
    // the MK05 eq. 41 star total pressure that the flux assembly itself
    // never needs to form.
    const int tangent1 = (direction + 1) % 3;
    const int tangent2 = (direction + 2) % 3;
    const Real inv_sqrt_mu0 = 1.0 / std::sqrt(parameters.mu0);
    const Real bn = bn_face * inv_sqrt_mu0;
    const Real u_left = left.ion_velocity[direction];
    const Real u_right = right.ion_velocity[direction];
    const auto p_total = [&] (const timhd::CellState& state) {
        const Real bt1 = state.magnetic[tangent1] * inv_sqrt_mu0;
        const Real bt2 = state.magnetic[tangent2] * inv_sqrt_mu0;
        return state.ion_pressure + state.electron_pressure +
               0.5 * (bn * bn + bt1 * bt1 + bt2 * bt2);
    };
    const Real p_total_left = p_total(left);
    const Real p_total_right = p_total(right);
    const Real width = parameters.hlld_kappa_signal *
                       (left.fast_speed + right.fast_speed);
    const Real s_left = timhd::smooth_min(
        u_left - left.fast_speed, u_right - right.fast_speed, width);
    const Real s_right = timhd::smooth_max(
        u_left + left.fast_speed, u_right + right.fast_speed, width);
    const Real m_left = left.safe_density * (s_left - u_left);
    const Real m_right = right.safe_density * (s_right - u_right);
    const Real s_mid = (p_total_right - p_total_left + m_left * u_left -
                        m_right * u_right) /
                       (m_left - m_right);
    FanDiagnostics diagnostics;
    diagnostics.p_total_star =
        (m_right * p_total_left - m_left * p_total_right +
         m_left * m_right * (u_right - u_left)) /
        (m_right - m_left);
    diagnostics.rho_star_left =
        left.density * (s_left - u_left) / (s_left - s_mid);
    diagnostics.rho_star_right =
        right.density * (s_right - u_right) / (s_right - s_mid);
    return diagnostics;
}

void test_positivity ()
{
    std::mt19937 rng(20260806);
    std::uniform_real_distribution<Real> uniform(0.0, 1.0);
    const timhd::FluxParameters parameters = make_params(1.0);

    // Physically bounded sweep: log-uniform densities and pressures over
    // several decades and |b| up to Alfven-dominated. Two boundedness
    // constraints define the physical regime:
    //  - div(B)-consistent normal field: B_n is shared across the face
    //    (up to 10 percent cell-centering noise), as the recast's
    //    staggered field guarantees; independent per-side normal
    //    components of opposite sign would remove the normal magnetic
    //    pressure from the single-valued face p_T while keeping its
    //    contribution in the signal speeds.
    //  - per-component velocities bounded by mach_limit times the JOINT
    //    (smaller) fast-speed scale of the two sides: once the face
    //    velocity divergence is comparable to the weaker side's signal
    //    speed the exact solution approaches vacuum formation and any
    //    linearized star pressure (MK05's included) legitimately goes
    //    negative -- the recast's Newton admissibility bound owns that
    //    regime. A wider sweep is reported for information only.
    const auto sweep = [&] (const Real mach_limit, const int samples,
                            Real& min_pressure, Real& min_density,
                            bool& finite) {
        min_pressure = 1.0e30;
        min_density = 1.0e30;
        finite = true;
        for (int sample = 0; sample < samples; ++sample) {
            const int direction = sample % 3;
            const Real bn_face = 8.0 * (uniform(rng) - 0.5);
            Real rho[2];
            Real p_ion[2];
            Real p_electron[2];
            Real field[2][3];
            Real fast_scale = 1.0e30;
            for (int index = 0; index < 2; ++index) {
                rho[index] = 0.05 * std::pow(400.0, uniform(rng));
                p_ion[index] = 3.0e-3 * std::pow(1.0e4 / 3.0, uniform(rng));
                p_electron[index] =
                    3.0e-3 * std::pow(1.0e4 / 3.0, uniform(rng));
                Real field2 = 0.0;
                for (int component = 0; component < 3; ++component) {
                    field[index][component] =
                        (component == direction)
                            ? bn_face * (1.0 + 0.1 * (uniform(rng) - 0.5))
                            : 8.0 * (uniform(rng) - 0.5);
                    field2 +=
                        field[index][component] * field[index][component];
                }
                fast_scale = std::min(
                    fast_scale,
                    std::sqrt((parameters.gamma_i * p_ion[index] +
                               parameters.gamma_e * p_electron[index] +
                               field2) /
                              rho[index]));
            }
            HostCell cells[2];
            for (int index = 0; index < 2; ++index) {
                const Real ux =
                    2.0 * mach_limit * (uniform(rng) - 0.5) * fast_scale;
                const Real uy =
                    2.0 * mach_limit * (uniform(rng) - 0.5) * fast_scale;
                const Real uz =
                    2.0 * mach_limit * (uniform(rng) - 0.5) * fast_scale;
                cells[index] = make_cell(
                    rho[index], ux, uy, uz, p_ion[index],
                    p_electron[index], field[index][0], field[index][1],
                    field[index][2], parameters);
            }
            const timhd::CellState left =
                load_state(cells[0], direction, parameters);
            const timhd::CellState right =
                load_state(cells[1], direction, parameters);
            const FanDiagnostics fan =
                wave_fan(left, right, bn_face, direction, parameters);
            min_pressure = std::min(min_pressure, fan.p_total_star);
            min_density = std::min(
                min_density,
                std::min(fan.rho_star_left, fan.rho_star_right));
            const Channels flux = flatten(timhd::hlld_flux(
                left, right, bn_face, direction, parameters));
            for (const Real value : flux) {
                if (!std::isfinite(value)) {
                    finite = false;
                }
            }
        }
    };

    Real min_pressure = 0.0;
    Real min_density = 0.0;
    bool finite = true;
    sweep(0.2, 100000, min_pressure, min_density, finite);
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "min p_T* %.6e, min rho* %.6e over 1e5 states, all "
                  "outputs finite: %s",
                  min_pressure, min_density, finite ? "yes" : "no");
    report("7. positivity sweep (|u| <= 0.2 c_f,joint)",
           min_pressure > 0.0 && min_density > 0.0 && finite, detail);

    Real info_pressure = 0.0;
    Real info_density = 0.0;
    bool info_finite = true;
    sweep(0.5, 100000, info_pressure, info_density, info_finite);
    std::printf("   [info] |u| <= 0.5 c_f,joint sweep: min p_T* %.6e, "
                "min rho* %.6e, finite: %s\n",
                info_pressure, info_density, info_finite ? "yes" : "no");
}

} // namespace

int main ()
{
    std::printf("HLLD kernel verification (ThetaImplicitMHD_K.H)\n");
    std::printf("------------------------------------------------\n");
    test_consistency();
    test_hydro_limit();
    test_bn_zero_limit();
    test_static_tangential_discontinuity();
    test_rankine_hugoniot();
    test_smoothness();
    test_positivity();
    std::printf("------------------------------------------------\n");
    if (g_failures == 0) {
        std::printf("ALL INVARIANT GROUPS PASS\n");
        return 0;
    }
    std::printf("%d INVARIANT GROUP(S) FAILED\n", g_failures);
    return 1;
}
