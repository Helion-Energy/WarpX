/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_CONDUCTION_3D_H_
#define WARPX_EULERIAN_THERMAL_CONDUCTION_3D_H_
#include "EulerianThermalConduction.H"
#include <AMReX_Array.H>
#if defined(WARPX_DIM_3D)
namespace warpx::thermal::mhd {
// Cartesian counterpart of the pinned MHD FD2/FD4 cell face stencil. The
// native QdsmcConductionFDOperator provides the full 3D sum over BOTH tensor
// cross directions; this FV adapter retains MHD coefficient staging, smooth
// SMART with corrected donor, mask downgrade and exact constant-null rows.
struct CartesianFDParams {
    int normal = 0, order = 2, cross_mode = 1;
    bool periodic_normal = false;
    int domain_lo_normal = 0, domain_hi_normal = 0;
    amrex::GpuArray<int, 3> low{}, high{}, periodic{};
    amrex::GpuArray<amrex::Real, 3> inverse_size{};
    amrex::Real limiter_width = .01, b2_floor = 0, cross_scale = 1;
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cartesian_shift_index (int& i, int& j, int& k, int d, int offset) noexcept {
    if (d == 0) {
        i += offset;
    } else if (d == 1) {
        j += offset;
    } else {
        k += offset;
    }
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
cartesian_index_along (int i, int j, int k, int d) noexcept {
    return d == 0 ? i : (d == 1 ? j : k);
}
template <class Sampler, class Masked>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
cartesian_chacon_fd_face_flux (const Sampler& e, const Masked& masked,
                               const amrex::Array4<const amrex::Real>& b_cc,
                               const int il, const int jl, const int kl,
                               const int i, const int j, const int k,
                               const amrex::Real chi_perp,
                               const amrex::Real dchi,
                               const CartesianFDParams& p,
                               amrex::Real& chi_nn_compact) noexcept {
    // the cell at offset m along the normal from the RIGHT cell (m = -1
    // is the left cell)
    const auto cell_n = [&] (const int m, int& ic, int& jc, int& kc) {
        ic = i;
        jc = j;
        kc = k;
        cartesian_shift_index(ic, jc, kc, p.normal, m);
    };
    const auto weight = [] (int) { return 1.0_rt; };
    const amrex::Real inverse_face_weight = 1.0_rt;
    // Full Cartesian tensor: all three B components enter the norm and each
    // directional face receives its two transverse projections.
    int tangent = (p.normal + 1) % 3;
    const auto tensor = [&] (const int ic, const int jc, const int kc,
                             amrex::Real& xi_nn, amrex::Real& xi_nt) {
        const amrex::Real b_x = b_cc(ic, jc, kc, 0);
        const amrex::Real b_y = b_cc(ic, jc, kc, 1);
        const amrex::Real b_z = b_cc(ic, jc, kc, 2);
        const amrex::Real b2 = smooth_positive_floor(
            b_x * b_x + b_y * b_y + b_z * b_z, p.b2_floor);
        const amrex::Real b_n = b_cc(ic, jc, kc, p.normal);
        xi_nn = chi_perp + dchi * b_n * b_n / b2;
        xi_nt = dchi * b_n * b_cc(ic, jc, kc, tangent) / b2;
    };
    // On a physical face, only the interior normal tensor enters. The stage
    // handles the actual wall law; cross flux never leaks through its image.
    const int n_face = cartesian_index_along(i, j, k, p.normal);
    const bool domain_lo_face =
        !p.periodic_normal && (n_face == p.domain_lo_normal);
    const bool domain_hi_face =
        !p.periodic_normal && (n_face == p.domain_hi_normal + 1);
    const bool domain_face = domain_lo_face || domain_hi_face;
    // order-4 window test (interior only; masked cells count as walls)
    bool use4 = (p.order == 4);
    if (use4) {
        if (!p.periodic_normal && (n_face - 3 < p.domain_lo_normal ||
                                   n_face + 2 > p.domain_hi_normal)) {
            use4 = false;
        }
        for (int tdir = 0; use4 && tdir < 3; ++tdir) {
            if (tdir == p.normal) {
                continue;
            }
            int const t_face = cartesian_index_along(i, j, k, tdir);
            if (!p.periodic[tdir] &&
                (t_face - 2 < p.low[tdir] || t_face + 2 > p.high[tdir])) {
                use4 = false;
            }
        }
        for (int m = -3; use4 && m <= 2; ++m) {
            int ic, jc, kc;
            cell_n(m, ic, jc, kc);
            if (masked(ic, jc, kc)) {
                use4 = false;
            }
            if (use4 && m >= -2 && m <= 1) {
                for (int tdir = 0; use4 && tdir < 3; ++tdir) {
                    if (tdir == p.normal) {
                        continue;
                    }
                    for (int t = -2; t <= 2; ++t) {
                        if (t == 0) {
                            continue;
                        }
                        int it = ic, jt = jc, kt = kc;
                        cartesian_shift_index(it, jt, kt, tdir, t);
                        if (masked(it, jt, kt)) {
                            use4 = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    // compact-star (co-derivative) part and the PC coefficient
    amrex::Real xi_nn_left, xi_nt_left, xi_nn_right, xi_nt_right;
    tensor(il, jl, kl, xi_nn_left, xi_nt_left);
    tensor(i, j, k, xi_nn_right, xi_nt_right);
    if (domain_lo_face) {
        xi_nn_left = xi_nn_right;
        xi_nt_left = xi_nt_right;
    } else if (domain_hi_face) {
        xi_nn_right = xi_nn_left;
        xi_nt_right = xi_nt_left;
    }
    const amrex::Real weight_left = weight(il);
    const amrex::Real weight_right = weight(i);
    chi_nn_compact = 0.5_rt *
                     (weight_left * xi_nn_left + weight_right * xi_nn_right) *
                     inverse_face_weight;
    const amrex::Real e_left = e(il, jl, kl);
    const amrex::Real e_right = e(i, j, k);
    const amrex::Real c0[4] = {-1.0_rt / 12.0_rt, 7.0_rt / 12.0_rt,
                               7.0_rt / 12.0_rt, -1.0_rt / 12.0_rt};
    const amrex::Real a6[4][6] = {
        {-12.0_rt, -65.0_rt, 120.0_rt, -60.0_rt, 20.0_rt, -3.0_rt},
        {3.0_rt, -30.0_rt, -20.0_rt, 60.0_rt, -15.0_rt, 2.0_rt},
        {-2.0_rt, 15.0_rt, -60.0_rt, 20.0_rt, 30.0_rt, -3.0_rt},
        {3.0_rt, -20.0_rt, 60.0_rt, -120.0_rt, 65.0_rt, 12.0_rt}};
    amrex::Real flux;
    if (!use4) {
        flux = chi_nn_compact * (e_right - e_left) * p.inverse_size[p.normal];
    } else {
        amrex::Real window[6];
        for (int m = 0; m < 6; ++m) {
            int ic, jc, kc;
            cell_n(m - 3, ic, jc, kc);
            window[m] = e(ic, jc, kc);
        }
        flux = 0.0_rt;
        for (int l = 0; l < 4; ++l) {
            int ic, jc, kc;
            cell_n(l - 2, ic, jc, kc);
            amrex::Real xi_nn, xi_nt;
            tensor(ic, jc, kc, xi_nn, xi_nt);
            amrex::Real derivative = 0.0_rt;
            for (int m = 0; m < 6; ++m) {
                // Each derivative row sums to zero. Remove a common value
                // before accumulation so constant T remains exactly null even
                // across a density jump and at very stiff diffusion steps.
                derivative += a6[l][m] * (window[m] - window[2]);
            }
            flux += c0[l] * weight(ic) * xi_nn * derivative / 60.0_rt;
        }
        flux *= p.inverse_size[p.normal] * inverse_face_weight;
    }
    // Sum both native Cartesian transverse directions independently. The
    // corrected smooth SMART donor and flux-before-cap staging are retained.
    for (tangent = 0; tangent < 3; ++tangent) {
        if (tangent != p.normal && dchi != 0.0_rt && p.cross_scale != 0.0_rt &&
            !domain_face) {
            tensor(il, jl, kl, xi_nn_left, xi_nt_left);
            tensor(i, j, k, xi_nn_right, xi_nt_right);
            // transverse difference at a cell: centered; one-sided against
            // a masked neighbour (a wall to the stencil); order 4 uses the
            // five-point d0 row (never beside a masked cell: use4)
            const auto transverse = [&] (const int ic, const int jc,
                                         const int kc) {
                if (use4) {
                    const amrex::Real d0[5] = {1.0_rt, -8.0_rt, 0.0_rt, 8.0_rt,
                                               -1.0_rt};
                    amrex::Real derivative = 0.0_rt;
                    for (int t = -2; t <= 2; ++t) {
                        if (t == 0) {
                            continue;
                        }
                        int it = ic, jt = jc, kt = kc;
                        cartesian_shift_index(it, jt, kt, tangent, t);
                        derivative +=
                            d0[t + 2] * (e(it, jt, kt) - e(ic, jc, kc));
                    }
                    return derivative / 12.0_rt * p.inverse_size[tangent];
                }
                int ip = ic, jp = jc, kp = kc;
                int im = ic, jm = jc, km = kc;
                cartesian_shift_index(ip, jp, kp, tangent, 1);
                cartesian_shift_index(im, jm, km, tangent, -1);
                const bool plus_open = !masked(ip, jp, kp);
                const bool minus_open = !masked(im, jm, km);
                const amrex::Real center = e(ic, jc, kc);
                const amrex::Real plus = plus_open ? e(ip, jp, kp) : center;
                const amrex::Real minus = minus_open ? e(im, jm, km) : center;
                const int span = (plus_open ? 1 : 0) + (minus_open ? 1 : 0);
                return (span == 0)
                           ? 0.0_rt
                           : (plus - minus) * p.inverse_size[tangent] / span;
            };
            amrex::Real raw;
            if (!use4) {
                raw = 0.5_rt *
                      (weight_left * xi_nt_left * transverse(il, jl, kl) +
                       weight_right * xi_nt_right * transverse(i, j, k));
            } else {
                raw = 0.0_rt;
                for (int l = 0; l < 4; ++l) {
                    int ic, jc, kc;
                    cell_n(l - 2, ic, jc, kc);
                    amrex::Real xi_nn, xi_nt;
                    tensor(ic, jc, kc, xi_nn, xi_nt);
                    raw += c0[l] * weight(ic) * xi_nt * transverse(ic, jc, kc);
                }
            }
            raw *= p.cross_scale * inverse_face_weight;
            if (p.cross_mode == 0) {
                // centered cross flux: the unlimited control
                flux += raw;
            } else {
                // advective recast: F_cross = v* e_face with v* = raw / e_f and
                // e_face the C-infinity SMART face value from the upwind side
                // of -v*: q = -rho F and the equation is dU/dt = +div(rho F).
                // Positive gradient flux therefore takes its donor from the
                // RIGHT. This carries Darwin d861b2c4d forward over the pinned
                // MHD sign. The upwind switch is the smooth sign of -v* at the
                // width a transverse contrast of limiter_width per cell would
                // give (per-solve constant under the frozen coefficients)
                const amrex::Real e_face_mean = 0.5_rt * (e_left + e_right);
                const amrex::Real velocity = raw / e_face_mean;
                int ill = il, jll = jl, kll = kl;
                cartesian_shift_index(ill, jll, kll, p.normal, -1);
                int irr = i, jrr = j, krr = k;
                cartesian_shift_index(irr, jrr, krr, p.normal, 1);
                const amrex::Real e_left_left =
                    masked(ill, jll, kll) ? e_left : e(ill, jll, kll);
                const amrex::Real e_right_right =
                    masked(irr, jrr, krr) ? e_right : e(irr, jrr, krr);
                const amrex::Real width_e = p.limiter_width * e_face_mean;
                const amrex::Real from_left = smart_face_value_smooth(
                    e_left_left, e_left, e_right, width_e);
                const amrex::Real from_right = smart_face_value_smooth(
                    e_right_right, e_right, e_left, width_e);
                const amrex::Real xi_reference =
                    0.5_rt * (std::abs(xi_nt_left) + std::abs(xi_nt_right)) *
                        std::abs(p.cross_scale) +
                    p.limiter_width * chi_perp;
                const amrex::Real width_v =
                    p.limiter_width * xi_reference * p.inverse_size[tangent];
                const amrex::Real upwind_left =
                    0.5_rt * (1.0_rt - smooth_sign(velocity, width_v));
                flux += velocity * (upwind_left * from_left +
                                    (1.0_rt - upwind_left) * from_right);
            }
        }
    }
    return flux;
}

} // namespace warpx::thermal::mhd
#endif
#endif
