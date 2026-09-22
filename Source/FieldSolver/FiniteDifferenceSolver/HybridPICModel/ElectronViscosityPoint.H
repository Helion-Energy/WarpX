/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ELECTRON_VISCOSITY_POINT_H_
#define WARPX_ELECTRON_VISCOSITY_POINT_H_

#include "BraginskiiViscosity.H"
#include "QdsmcFluxLimiters.H"
#include "Utils/WarpXConst.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuUtility.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

/** \file ElectronViscosityPoint.H
 *
 * ONE evaluation of the electron viscous state at a node, shared by the
 * two consumers of the Braginskii electron viscosity:
 *
 *   - the HEATING kernel (HybridPICModel::QDSMCAddViscousHeating), which
 *     books Q_nu = -Pi : grad u_e into T_e, and
 *   - the STRESS precompute of the E-solve (HybridPICModel::
 *     ComputeViscousStressNodal), which builds Pi_e so that Ohm's law can
 *     carry the drag E_visc = -div(Pi_e)/(e n_e)
 *     (hybrid_pic_model.qdsmc_viscosity_in_ohms_law).
 *
 * The energy balance between the two -- the field loses Int J . E_visc dV,
 * the electrons gain Int Q_nu dV -- holds only if BOTH are built from the
 * same velocity gradient, the same field direction and the same capped and
 * limited coefficients at every node. Factoring that evaluation here is
 * what makes it impossible for a later edit to change one and not the
 * other. The arithmetic below is the heating kernel's original Stage 2,
 * operation for operation, so the heating with the drag off is
 * bit-identical to the pre-factoring code.
 *
 * The caller owns two things this header deliberately does not: the nodal
 * electron velocity u_e (HybridPICModel::FillNodalElectronVelocity, one
 * ghost layer filled) and the domain gate (one node inside every
 * non-periodic face, where the centred stencil has no neighbour).
 */

#if defined(WARPX_DIM_RZ)
/** Projection onto the free physical Yee edges of an axisymmetric cylinder.
 * Used on both sides of the viscous adjoint pair. Exterior ghost values and
 * fixed tangential PEC rows are not degrees of freedom of that pair. */
struct ElectronViscosityRZEdges {
    amrex::GpuArray<int, 2> lo{}, hi{}, periodic{}, pec_lo{}, pec_hi{};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
    active (int const i, int const j, int const component) const {
        int const ir_hi = hi[0] + (component == 0 ? 0 : 1);
        int const iz_hi = hi[1] + (component == 2 ? 0 : 1);
        if (i < lo[0] || i > ir_hi ||
            (!periodic[1] && (j < lo[1] || j > iz_hi))) {
            return false;
        }
        if (component == 1 && i == lo[0]) {
            return false;
        }
        if (component != 0 && pec_hi[0] && i == hi[0] + 1) {
            return false;
        }
        if (component != 2 &&
            ((pec_lo[1] && j == lo[1]) || (pec_hi[1] && j == hi[1] + 1))) {
            return false;
        }
        return true;
    }
};
#endif

/** Kernel-side parameters of the evaluation (trivially copyable; captured
 *  by value into ParallelFor lambdas). Filled once per call by the host
 *  from the HybridPICModel inputs. */
struct ElectronViscosityParams {
    /** qdsmc_viscosity_model: 1 = braginskii, 2 = parser. */
    int model = 1;
    /** qdsmc_viscosity_limiter = mc (true) or none (centred). */
    bool mc_limited = true;
    /** qdsmc_nu_par / qdsmc_nu_perp (n [m^-3], Te [eV], B [T]) -> m^2/s;
     *  consulted under model == 2 only. */
    amrex::ParserExecutor<3> nu_par_pars{};
    amrex::ParserExecutor<3> nu_perp_pars{};
    /** Braginskii collision-time inputs (model == 1). */
    amrex::Real coulomb_log = 10.0;
    amrex::Real Z_eff = 1.0;
    /** Free-streaming factor on the parallel channel (<= 0 disables). */
    amrex::Real flux_limit_f = 0.1;
    /** Hard kinematic ceiling on both channels (0 = off). */
    amrex::Real nu_max = 0.0;
    /** Solver density floor as a charge density [C/m^3]. */
    amrex::Real rho_floor = 0.0;
    /** Optional C^1 taper of BOTH coefficients to zero across the plasma
     *  edge (smoothstep, 0 at n <= taper_n, 1 at n >= 2 taper_n; 0 = off).
     *  Applied to mu_par and mu_perp alike, so Q_nu and Pi scale together
     *  and the energy identity is untouched; it removes the O(1) jump of
     *  Pi at the density gate that otherwise makes div(Pi)/rho_floor the
     *  largest drag in the run. Input: qdsmc_viscosity_taper_n [m^-3]. */
    amrex::Real taper_n = 0.0;
    /** Cell sizes and the physical direction each grid axis differentiates
     *  along (components are always (x, y, z) / (r, theta, z)). */
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
    amrex::GpuArray<int, AMREX_SPACEDIM> dim_to_dir{};
    /** Radial origin of the grid (RZ metric terms; unused elsewhere). */
    amrex::Real problo_r = 0.0;
};

/** The evaluated state at one node. */
struct ElectronViscosityPoint {
    /** Velocity gradient G[dir][comp] = d u_comp / d x_dir [1/s]. */
    amrex::Real G[3][3];
    /** Unit vector along B (z-hat at a field null; immaterial there). */
    amrex::Real b[3];
    /** Field-aligned strain S = b.W.b [1/s]. */
    amrex::Real S;
    /** Dynamic viscosities [Pa s] AFTER the nu_max cap and the
     *  free-streaming limit on the parallel channel. */
    amrex::Real mu_par;
    amrex::Real mu_perp;
    /** Electron density [m^-3] and temperature [J] at the node. */
    amrex::Real n_e;
    amrex::Real Te_J;
    /** Parser model only: the coefficient evaluated negative and was
     *  clamped to zero (the caller marks and reports it). */
    bool nu_par_clamped;
    bool nu_perp_clamped;
};

/** Evaluate the electron viscous state at node (i, j, k).
 *
 *  Returns false, leaving \c out untouched, where the source is gated off:
 *  below the solver density floor (u_e is the floored-rho quotient there,
 *  not a fluid velocity, and nu_par ~ Te^{5/2}/n_e diverges) or at a
 *  non-positive temperature. Both consumers must treat "false" as
 *  Q_nu = 0 and Pi = 0.
 *
 * @param ue      nodal electron velocity, 3 components, one ghost layer
 * @param rho_arr charge density on the nodal grid [C/m^3]
 * @param Te_arr  electron temperature on the nodal grid [K]
 * @param B*_arr  magnetic field on its Yee staggering (B*_stag)
 * @param p       parameters
 * @param out     result */
// Device-only: it calls ablastr::coarsen::sample::Interp, which is
// AMREX_GPU_DEVICE; both call sites are ParallelFor device lambdas (a
// __host__ __device__ qualifier here fails to compile under nvcc).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool
electron_viscosity_point (int const i, int const j, int const k,
                          amrex::Array4<amrex::Real const> const& ue,
                          amrex::Array4<amrex::Real const> const& rho_arr,
                          amrex::Array4<amrex::Real const> const& Te_arr,
                          amrex::Array4<amrex::Real const> const& Bx_arr,
                          amrex::Array4<amrex::Real const> const& By_arr,
                          amrex::Array4<amrex::Real const> const& Bz_arr,
                          amrex::GpuArray<int, 3> const& Bx_stag,
                          amrex::GpuArray<int, 3> const& By_stag,
                          amrex::GpuArray<int, 3> const& Bz_stag,
                          ElectronViscosityParams const& p,
                          ElectronViscosityPoint& out) {
    using namespace amrex::literals;
    using ablastr::coarsen::sample::Interp;

    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    // Vacuum gate: below the solver density floor u_e is the floored-rho
    // quotient, not a fluid velocity, and nu_par ~ Te^{5/2}/n_e diverges
    // there. Heating (or stress) outside the plasma is a runaway vector,
    // not physics.
    amrex::Real const rho_val = rho_arr(i, j, k);
    if (rho_val <= p.rho_floor) {
        return false;
    }

    amrex::Real const n_e = rho_val / PhysConst::q_e;
    amrex::Real const Te_K = Te_arr(i, j, k);
    if (Te_K <= 0.0_rt) {
        return false;
    }
    // A non-finite state (a NaN/inf another operator let through) must not
    // become a NaN stress that the E-solve broadcasts through Faraday into
    // every field and, via the push, into the particle positions -- the
    // CUDA 700 / segfault route. Gate the node off; the originating
    // operator is the bug and stays visible in Te.
    if (amrex::isnan(rho_val) || amrex::isinf(rho_val) || amrex::isnan(Te_K) ||
        amrex::isinf(Te_K)) {
        return false;
    }
    amrex::Real const Te_J = Te_K * PhysConst::kb;

    // Velocity gradient tensor G[dir][comp] = d u_comp / d x_dir.
    amrex::Real G[3][3] = {{0.0_rt, 0.0_rt, 0.0_rt},
                           {0.0_rt, 0.0_rt, 0.0_rt},
                           {0.0_rt, 0.0_rt, 0.0_rt}};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        int const di = (d == 0) ? 1 : 0;
        int const dj = (d == 1) ? 1 : 0;
        int const dk = (d == 2) ? 1 : 0;
        amrex::Real const inv_dx = 1.0_rt / p.dx[d];
        for (int c = 0; c < 3; ++c) {
            amrex::Real const up = ue(i + di, j + dj, k + dk, c);
            amrex::Real const u0 = ue(i, j, k, c);
            amrex::Real const um = ue(i - di, j - dj, k - dk, c);
            G[p.dim_to_dir[d]][c] =
                p.mc_limited ? qdsmc_mc_slope(up - u0, u0 - um) * inv_dx
                             : 0.5_rt * (up - um) * inv_dx;
        }
    }
#if defined(WARPX_DIM_RZ)
    // Cylindrical metric terms of grad u at m = 0. The caller excludes the
    // axis node, so r > 0 here.
    {
        amrex::Real const r = p.problo_r + i * p.dx[0];
        G[1][0] = -ue(i, j, k, 1) / r;
        G[1][1] = ue(i, j, k, 0) / r;
        G[1][2] = 0.0_rt;
    }
#endif

    // Field direction. |B| = 0 leaves b-hat undefined; the regularised
    // nu_perp then equals nu_par up to the 0.51/0.73 ratio, so the choice
    // of axis is immaterial there.
    amrex::Real const bx = Interp(Bx_arr, Bx_stag, nodal, coarsen, i, j, k, 0);
    amrex::Real const by = Interp(By_arr, By_stag, nodal, coarsen, i, j, k, 0);
    amrex::Real const bz = Interp(Bz_arr, Bz_stag, nodal, coarsen, i, j, k, 0);
    amrex::Real const Bmag = std::sqrt(bx * bx + by * by + bz * bz);
    amrex::Real b[3] = {0.0_rt, 0.0_rt, 1.0_rt};
    if (Bmag > 0.0_rt) {
        b[0] = bx / Bmag;
        b[1] = by / Bmag;
        b[2] = bz / Bmag;
    }

    // Kinematic viscosities [m^2/s].
    amrex::Real nu_par;
    amrex::Real nu_perp;
    bool par_clamped = false;
    bool perp_clamped = false;
    if (p.model == 1) {
        amrex::Real const tau_e =
            braginskii_tau_e(n_e, Te_J, p.Z_eff, p.coulomb_log);
        nu_par = braginskii_nu_par(Te_J, tau_e);
        nu_perp = braginskii_nu_perp(nu_par, PhysConst::q_e * Bmag * tau_e /
                                                 PhysConst::m_e);
    } else {
        amrex::Real const K_per_eV = PhysConst::q_e / PhysConst::kb;
        amrex::Real const Te_eV = Te_K / K_per_eV;
        nu_par = p.nu_par_pars(n_e, Te_eV, Bmag);
        nu_perp = p.nu_perp_pars(n_e, Te_eV, Bmag);
        // A user expression can go negative in some cells at some times,
        // which no parse-time check can see. Clamp to zero -- a negative
        // coefficient flips the sign of a positive-definite dissipation --
        // and report it to the caller, which marks and counts it.
        if (nu_par < 0.0_rt) {
            nu_par = 0.0_rt;
            par_clamped = true;
        }
        if (nu_perp < 0.0_rt) {
            nu_perp = 0.0_rt;
            perp_clamped = true;
        }
    }
    if (p.nu_max > 0.0_rt) {
        nu_par = std::min(nu_par, p.nu_max);
        nu_perp = std::min(nu_perp, p.nu_max);
    }
    // Edge taper (ElectronViscosityParams::taper_n): the same weight on
    // both channels keeps -Pi:G == Q_nu exact. Off (0) leaves them alone.
    if (p.taper_n > 0.0_rt) {
        amrex::Real x = (n_e - p.taper_n) / p.taper_n;
        x = amrex::min(amrex::max(x, 0.0_rt), 1.0_rt);
        amrex::Real const w = x * x * (3.0_rt - 2.0_rt * x);
        nu_par *= w;
        nu_perp *= w;
    }

    // Dynamic viscosities [Pa s], with the free-streaming cap on the
    // parallel channel only (the cross-field one is ~12 orders smaller and
    // never approaches the bound).
    amrex::Real const S = braginskii_strain_par(G, b);
    amrex::Real const mu_par = braginskii_visc_flux_limit(
        PhysConst::m_e * n_e * nu_par, std::abs(S), n_e * Te_J, p.flux_limit_f);
    amrex::Real const mu_perp = PhysConst::m_e * n_e * nu_perp;
    // Non-finite gradients (a NaN u_e neighbour) gate the node off for the
    // same reason as the state check above.
    if (amrex::isnan(S) || amrex::isinf(S) || amrex::isnan(mu_par) ||
        amrex::isinf(mu_par) || amrex::isnan(mu_perp) ||
        amrex::isinf(mu_perp)) {
        return false;
    }

    for (int a = 0; a < 3; ++a) {
        out.b[a] = b[a];
        for (int c = 0; c < 3; ++c) {
            out.G[a][c] = G[a][c];
        }
    }
    out.S = S;
    out.mu_par = mu_par;
    out.mu_perp = mu_perp;
    out.n_e = n_e;
    out.Te_J = Te_J;
    out.nu_par_clamped = par_clamped;
    out.nu_perp_clamped = perp_clamped;
    return true;
}

#endif // WARPX_ELECTRON_VISCOSITY_POINT_H_
