/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_THERMAL_CONDUCTION_H_
#define WARPX_EULERIAN_THERMAL_CONDUCTION_H_

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>
#include <algorithm>
#include <cmath>

// Extracted from Helion-Energy/WarpX resistive_mhd at
// 592e2b3dfe9469eabab7bd8c6696a07ce1d68277, ThetaImplicitMHD{,_K}.{cpp,H}.
// Pure cell-centered face kernels: no solver state, accepted writes or tallies.
// The SMART donor sign carries the validated Darwin d861b2c4d correction.
// These kernels are not yet connected to a Darwin time-advance mode.
namespace warpx::thermal::mhd {
using namespace amrex::literals;
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_min (const amrex::Real a, const amrex::Real b,
            const amrex::Real width) noexcept {
    return 0.5 * (a + b - std::sqrt((a - b) * (a - b) + width * width));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_max (const amrex::Real a, const amrex::Real b,
            const amrex::Real width) noexcept {
    return 0.5 * (a + b + std::sqrt((a - b) * (a - b) + width * width));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_sign (const amrex::Real value, const amrex::Real width) noexcept {
    const amrex::Real magnitude = std::sqrt(value * value + width * width);
    return magnitude > 0.0 ? value / magnitude : 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smooth_positive_floor (const amrex::Real value,
                       const amrex::Real floor) noexcept {
    const amrex::Real excess = value - floor;
    return floor + 0.5 * (excess + std::sqrt(excess * excess + floor * floor));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
smart_face_value_smooth (const amrex::Real e_uu, const amrex::Real e_u,
                         const amrex::Real e_d,
                         const amrex::Real width) noexcept {
    const amrex::Real d_u = e_u - e_uu;
    const amrex::Real d_d = e_d - e_u;
    const amrex::Real s = smooth_sign(d_u + d_d, width);
    const amrex::Real a = 2.0 * s * d_u;
    const amrex::Real b = 0.125 * s * (d_u + 3.0 * d_d);
    const amrex::Real c = s * d_d;
    const amrex::Real correction =
        smooth_max(0.0, smooth_min(smooth_min(a, b, width), c, width), width);
    return e_u + s * correction;
}

// The pinned MHD stencil has a single transverse direction and is qualified
// only in RZ/1D. Do not silently expose that incomplete tensor as a 3D
// operator. Darwin's existing full 3D nodal operator remains independent and
// intact.
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
shift_index (int& i, int& j, int& k, const int physical_direction,
             const int offset) noexcept {
#if defined(WARPX_DIM_3D)
    if (physical_direction == 0) {
        i += offset;
    } else if (physical_direction == 1) {
        j += offset;
    } else {
        k += offset;
    }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::ignore_unused(k);
    if (physical_direction == 0) {
        i += offset;
    } else if (physical_direction == 2) {
        j += offset;
    }
#elif defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(j, k);
    if (physical_direction == 2) {
        i += offset;
    }
#else
    amrex::ignore_unused(i, j, k, physical_direction, offset);
#endif
}

// chacon_fd conduction operator (implicit_mhd.conduction_operator =
// chacon_fd; the ThetaImplicitMHD.H member comment carries the grid
// mapping, the smoothness class and the references): per-solve constants
// of the stencil, filled beside the other host constants of the face
// kernel.
struct ChaconFDParams {
    int normal = 2;     // physical direction of the face normal
    int tangential = 0; // in-plane transverse direction (RZ only)
    int order = 2;      // 2 = compact star, 4 = c0/A6/d0 composite
    int cross_mode = 1; // 1 = SMART advective recast, 0 = centered
    bool radial_faces = false;
    bool periodic_normal = false;
    bool periodic_tangential = false;
    int domain_lo_normal = 0;
    int domain_hi_normal = 0;
    int domain_lo_tangential = 0;
    int domain_hi_tangential = 0;
    amrex::Real radial_lower = 0.0;
    amrex::Real radial_cell_size = 1.0;
    amrex::Real inverse_normal_size = 1.0;
    amrex::Real inverse_tangential_size = 1.0;
    amrex::Real limiter_width = 0.01;
    amrex::Real b2_floor = 0.0;
    amrex::Real cross_scale = 1.0;
};

// Cell index along a physical direction (the bookkeeping shift_index
// inverts): RZ/XZ (i, j) = (r|x, z); 1D i = z.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
index_along (const int i, const int j, const int k,
             const int physical_direction) noexcept {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
    amrex::ignore_unused(k);
    return (physical_direction == 0) ? i : j;
#elif defined(WARPX_DIM_1D_Z)
    amrex::ignore_unused(j, k, physical_direction);
    return i;
#else
    return (physical_direction == 0) ? i : ((physical_direction == 1) ? j : k);
#endif
}

// The chacon_fd flux bracket F through the face between the left cell
// (il, jl, kl) and the right cell (i, j, k), positive along +n, in
// chi x e / length units: the caller books q_n = -rho_f F on the face
// registers exactly as for the other assembly. chi_perp and dchi =
// chi_par - chi_perp are the FACE coefficients of the pipeline (dchi = 0
// on the scalar channel, where the operator reduces to the compact-star
// or composite normal flux); chi_nn_compact receives the compact-star nn
// coefficient <J Xi_nn>_face / J_face for the preconditioner (both
// orders). e(ic, jc, kc) samples a cell's specific internal energy
// (floored, positive); masked(ic, jc, kc) is the static shaped-wall mask;
// b_cc the cell-centred total B (3 components, indexed by physical
// direction). The face at r = 0 never reaches this function (the kernel
// zeroes it before the conduction block).
//
// Order 4 (Chacon et al. 2025): the face flux is the c0 = (-1, 7, 7, -1)/12
// composite over the cells L-1, L, R, R+1 of the six-point A6 derivative
// rows on the window L-2 .. R+2 (co-part) and of the five-point d0
// transverse derivative (cross part). c0 is NOT a face interpolant: the
// face DIFFERENCE of the c0-weighted cell fluxes reproduces the
// fourth-order five-point divergence exactly, so the design order lives
// in the telescoped divergence. Interior only: a face whose normal window
// leaves a non-periodic domain, whose +-2 transverse window does, or
// whose windows touch a masked cell takes the compact second-order form.
template <class Sampler, class Masked>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
chacon_fd_face_flux (const Sampler& e, const Masked& masked,
                     const amrex::Array4<const amrex::Real>& b_cc, const int il,
                     const int jl, const int kl, const int i, const int j,
                     const int k, const amrex::Real chi_perp,
                     const amrex::Real dchi, const ChaconFDParams& p,
                     amrex::Real& chi_nn_compact) noexcept {
    // the cell at offset m along the normal from the RIGHT cell (m = -1
    // is the left cell)
    const auto cell_n = [&] (const int m, int& ic, int& jc, int& kc) {
        ic = i;
        jc = j;
        kc = k;
        shift_index(ic, jc, kc, p.normal, m);
    };
    // metric weight of a cell's product: J = r_cell on r-faces (the
    // nodal J of the hybrid operator folded into the cell products), 1
    // otherwise; the register value divides the face weight out again
    // (the divergence multiplies by r_face).
    const auto weight = [&] (const int ic) {
        return p.radial_faces
                   ? p.radial_lower + (ic + 0.5_rt) * p.radial_cell_size
                   : 1.0_rt;
    };
    const amrex::Real inverse_face_weight =
        p.radial_faces ? 1.0_rt / (p.radial_lower + i * p.radial_cell_size)
                       : 1.0_rt;
    // cell tensor components from the cell unit field (all three B
    // components in the norm, the in-plane components in the tensor;
    // the smooth |B|^2 floor of the pipeline regularizes B -> 0)
    const auto tensor = [&] (const int ic, const int jc, const int kc,
                             amrex::Real& xi_nn, amrex::Real& xi_nt) {
        const amrex::Real b_x = b_cc(ic, jc, kc, 0);
        const amrex::Real b_y = b_cc(ic, jc, kc, 1);
        const amrex::Real b_z = b_cc(ic, jc, kc, 2);
        const amrex::Real b2 = smooth_positive_floor(
            b_x * b_x + b_y * b_y + b_z * b_z, p.b2_floor);
        const amrex::Real b_n = b_cc(ic, jc, kc, p.normal);
        xi_nn = chi_perp + dchi * b_n * b_n / b2;
#if defined(WARPX_DIM_RZ)
        xi_nt = dchi * b_n * b_cc(ic, jc, kc, p.tangential) / b2;
#else
        xi_nt = 0.0_rt;
#endif
    };
    // Non-periodic domain-boundary face: the ghost cell is a boundary
    // image, not a field sample -- its field direction and the recast's
    // transverse derivative in it would leak the parallel channel through
    // the boundary face (measured: 1e-7 relative energy drift on the
    // oblique-field test whose FACE B_r vanishes at r_max while the last
    // cell's does not). Such a face takes the INTERIOR cell's tensor on
    // both sides and carries no cross flux: the compact normal flux
    // against the ghost image is then the boundary condition itself --
    // zero for the mirror/Neumann images, the reservoir exchange for a
    // wall-temperature image -- exactly the one-sided rule of the
    // conductive z-end branch.
    const int n_face = index_along(i, j, k, p.normal);
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
#if defined(WARPX_DIM_RZ)
        const int t_face = index_along(i, j, k, p.tangential);
        if (use4 && !p.periodic_tangential &&
            (t_face - 2 < p.domain_lo_tangential ||
             t_face + 2 > p.domain_hi_tangential)) {
            use4 = false;
        }
#endif
        for (int m = -3; use4 && m <= 2; ++m) {
            int ic, jc, kc;
            cell_n(m, ic, jc, kc);
            if (masked(ic, jc, kc)) {
                use4 = false;
            }
#if defined(WARPX_DIM_RZ)
            if (use4 && m >= -2 && m <= 1) {
                for (int t = -2; t <= 2; ++t) {
                    if (t == 0) {
                        continue;
                    }
                    int it = ic, jt = jc, kt = kc;
                    shift_index(it, jt, kt, p.tangential, t);
                    if (masked(it, jt, kt)) {
                        use4 = false;
                        break;
                    }
                }
            }
#endif
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
        flux = chi_nn_compact * (e_right - e_left) * p.inverse_normal_size;
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
        flux *= p.inverse_normal_size * inverse_face_weight;
    }
#if defined(WARPX_DIM_RZ)
    // cross-derivative part (the in-plane tangent; absent on the scalar
    // channel where dchi = 0)
    if (dchi != 0.0_rt && p.cross_scale != 0.0_rt && !domain_face) {
        // transverse difference at a cell: centered; one-sided against
        // a masked neighbour (a wall to the stencil); order 4 uses the
        // five-point d0 row (never beside a masked cell: use4)
        const auto transverse = [&] (const int ic, const int jc, const int kc) {
            if (use4) {
                const amrex::Real d0[5] = {1.0_rt, -8.0_rt, 0.0_rt, 8.0_rt,
                                           -1.0_rt};
                amrex::Real derivative = 0.0_rt;
                for (int t = -2; t <= 2; ++t) {
                    if (t == 0) {
                        continue;
                    }
                    int it = ic, jt = jc, kt = kc;
                    shift_index(it, jt, kt, p.tangential, t);
                    derivative += d0[t + 2] * (e(it, jt, kt) - e(ic, jc, kc));
                }
                return derivative / 12.0_rt * p.inverse_tangential_size;
            }
            int ip = ic, jp = jc, kp = kc;
            int im = ic, jm = jc, km = kc;
            shift_index(ip, jp, kp, p.tangential, 1);
            shift_index(im, jm, km, p.tangential, -1);
            const bool plus_open = !masked(ip, jp, kp);
            const bool minus_open = !masked(im, jm, km);
            const amrex::Real center = e(ic, jc, kc);
            const amrex::Real plus = plus_open ? e(ip, jp, kp) : center;
            const amrex::Real minus = minus_open ? e(im, jm, km) : center;
            const int span = (plus_open ? 1 : 0) + (minus_open ? 1 : 0);
            return (span == 0)
                       ? 0.0_rt
                       : (plus - minus) * p.inverse_tangential_size / span;
        };
        amrex::Real raw;
        if (!use4) {
            raw = 0.5_rt * (weight_left * xi_nt_left * transverse(il, jl, kl) +
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
            // Positive gradient flux therefore takes its donor from the RIGHT.
            // This carries Darwin d861b2c4d forward over the pinned MHD sign.
            // The upwind switch is the smooth sign of -v* at the
            // width a transverse contrast of limiter_width per cell would
            // give (per-solve constant under the frozen coefficients)
            const amrex::Real e_face_mean = 0.5_rt * (e_left + e_right);
            const amrex::Real velocity = raw / e_face_mean;
            int ill = il, jll = jl, kll = kl;
            shift_index(ill, jll, kll, p.normal, -1);
            int irr = i, jrr = j, krr = k;
            shift_index(irr, jrr, krr, p.normal, 1);
            const amrex::Real e_left_left =
                masked(ill, jll, kll) ? e_left : e(ill, jll, kll);
            const amrex::Real e_right_right =
                masked(irr, jrr, krr) ? e_right : e(irr, jrr, krr);
            const amrex::Real width_e = p.limiter_width * e_face_mean;
            const amrex::Real from_left =
                smart_face_value_smooth(e_left_left, e_left, e_right, width_e);
            const amrex::Real from_right = smart_face_value_smooth(
                e_right_right, e_right, e_left, width_e);
            const amrex::Real xi_reference =
                0.5_rt * (std::abs(xi_nt_left) + std::abs(xi_nt_right)) *
                    std::abs(p.cross_scale) +
                p.limiter_width * chi_perp;
            const amrex::Real width_v =
                p.limiter_width * xi_reference * p.inverse_tangential_size;
            const amrex::Real upwind_left =
                0.5_rt * (1.0_rt - smooth_sign(velocity, width_v));
            flux += velocity * (upwind_left * from_left +
                                (1.0_rt - upwind_left) * from_right);
        }
    }
#else
    amrex::ignore_unused(xi_nt_left, xi_nt_right, domain_face);
#endif
    return flux;
}

#endif

struct ConductionFaceResult {
    amrex::Real flux;
    amrex::Real pc_stage_cap_factor;
};

/** The exact MHD chacon_fd mixed-stage order: combine uncapped, limited
 * gradient-flux brackets with the SAME face coefficients, convert to heat
 * flux, then cap once. It differs from evaluating a nonlinear limiter on the
 * extrapolated energy and from mixing two already capped fluxes. q_limit=0
 * disables the cap; otherwise q_limit must be positive. The returned PC factor
 * freezes the limiter/coefficient/q_limit dependence, as the MHD surrogate
 * does.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ConductionFaceResult
ComposeConductionFlux (amrex::Real const stage_bracket,
                       amrex::Real const old_bracket, amrex::Real const theta,
                       amrex::Real const conduction_theta,
                       amrex::Real const face_density,
                       amrex::Real const q_limit) noexcept {
    auto bracket = stage_bracket;
    auto const weight = conduction_theta / theta;
    if (conduction_theta != theta) {
        bracket = weight * bracket + (1.0_rt - weight) * old_bracket;
    }
    auto flux = -face_density * bracket;
    auto cap = 1.0_rt;
    if (q_limit > 0.0_rt) {
        cap = 1.0_rt + std::abs(flux) / q_limit;
        flux /= cap;
    }
    return {flux, weight / (cap * cap)};
}
} // namespace warpx::thermal::mhd
#endif
