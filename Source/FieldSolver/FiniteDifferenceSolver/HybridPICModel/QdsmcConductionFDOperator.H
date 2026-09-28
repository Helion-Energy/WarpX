/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_CONDUCTION_FD_OPERATOR_H_
#define WARPX_QDSMC_CONDUCTION_FD_OPERATOR_H_

#include "QdsmcFluxLimiters.H"
#include "QdsmcVolumeElement.H"
#include "Utils/WarpXConst.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <ablastr/utils/Communication.H>
#include <cmath>

namespace warpx::thermal {
using namespace amrex::literals;

/** Frozen B/density context of one thermal subsolve, rebuilt after field/moment
 * changes. The caller owns the MultiFab with these components and valid ghosts.
 * This is distinct from the T-dependent tensor, which is rebuilt at every
 * probe.
 */
struct ConductionState {
    enum : int { b_bx = 0, b_by, b_bz, b_B2, b_ne, b_open, b_ebm, b_ncomp };
};

/** Spatial parameters only. flux_budget is the optional legacy donor-energy
 * bound: it additionally depends on budget_dt and is not a geometric cache.
 * The implicit path must deliberately select/derive that bound, never inherit
 * an explicit substep size by accident. No integration or clipping occurs here.
 */
struct ConductionFDOptions {
    amrex::Real flux_limit = 0.0_rt;
    bool isotropic = false;
    amrex::Real isotropic_B = 0.0_rt;
    bool fourth_order = true;
    int limiter = 2;
    amrex::Real limiter_width = 1.0e-3_rt;
    bool flux_budget = false;
    amrex::Real budget_floor_K = 0.0_rt;
    amrex::Real budget_dt = 1.0_rt;
};

/** Geometry shared by coefficient and flux evaluations. Nodal domain faces own
 * clipped dual cells; no outward face beyond a nonperiodic wall is introduced.
 */
struct ConductionFDGeometry {
    QdsmcVolumeElement dual_volume;
    amrex::Periodicity period;
    amrex::GpuArray<int, AMREX_SPACEDIM> gd2ax{}, is_per{}, dom_lo{}, dom_hi{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi_g{};
    amrex::Real r_edge0 = 0.0_rt, dr_rz = 1.0_rt;

    explicit ConductionFDGeometry (amrex::Geometry const& geom)
        : dual_volume(
              MakeQdsmcVolumeElement(geom, amrex::IndexType::TheNodeType())),
          period(geom.periodicity()) {
        auto const nodes = amrex::surroundingNodes(geom.Domain());
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#if defined(WARPX_DIM_3D)
            gd2ax[d] = d;
#elif (AMREX_SPACEDIM == 2)
            gd2ax[d] = d == 0 ? 0 : 2;
#else
            gd2ax[d] = 2;
#endif
            is_per[d] = geom.isPeriodic(d) ? 1 : 0;
            dom_lo[d] = nodes.smallEnd(d);
            dom_hi[d] = nodes.bigEnd(d);
            dxi_g[d] = geom.InvCellSize(d);
        }
        r_edge0 = geom.ProbLo(0);
        dr_rz = geom.CellSize(0);
    }
};

/** Rebuild the nonlinear conductivity tensor from one immutable trial T.
 * Tin requires three valid ghosts; bne requires three and xi two. Only xi is
 * written. Conductivity parsers use n [m^-3], T [eV], t [s]. No wall map,
 * positivity floor, integration, physical diagnostic or RNG update occurs.
 */
template <typename ParallelKappa, typename PerpendicularKappa>
void
BuildConductionTensor (amrex::MultiFab const& Tin, amrex::MultiFab const& bne,
                       amrex::MultiFab& xi,
                       ConductionFDGeometry const& geometry,
                       ConductionFDOptions const& options,
                       ParallelKappa const& kappa_par_ex,
                       PerpendicularKappa const& kappa_perp_ex,
                       amrex::Real const t_now) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        Tin.ixType().nodeCentered() &&
            Tin.nGrowVect().allGE(amrex::IntVect(3)) &&
            bne.nGrowVect().allGE(amrex::IntVect(3)) &&
            xi.nGrowVect().allGE(amrex::IntVect(2)),
        "Conduction tensor needs nodal T with three ghosts, context with "
        "three, tensor with two");
    using BNE = ConductionState;
    int constexpr NXI = AMREX_SPACEDIM * (AMREX_SPACEDIM + 1) / 2;
    auto const gd2ax = geometry.gd2ax;
    auto const dxi_g = geometry.dxi_g;
    auto const is_per = geometry.is_per;
    auto const dom_lo = geometry.dom_lo;
    auto const dom_hi = geometry.dom_hi;
    auto const f_lim = options.flux_limit;
    auto const iso_full = options.isotropic;
    auto const iso_B = options.isotropic_B;
    bool const iso_any = iso_full || (iso_B > 0.0_rt);
    amrex::Real const kb = PhysConst::kb, qe = PhysConst::q_e,
                      me = PhysConst::m_e;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(xi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const box = mfi.growntilebox(2);
        amrex::Array4<amrex::Real> const& x_arr = xi.array(mfi);
        amrex::Array4<amrex::Real const> const& b_arr = bne.const_array(mfi);
        amrex::Array4<amrex::Real const> const& T_arr = Tin.const_array(mfi);

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real const ne = b_arr(i, j, k, BNE::b_ne);
            // Closed (floored/covered) nodes carry a ZERO tensor: their
            // faces carry no flux, so no consumer may ever see a value
            // there -- and this is also the guard that keeps unfilled
            // non-periodic domain ghosts (b_ne = 0 setVal default, T
            // ghosts unspecified on integrator stage arrays) from
            // minting inf/nan chi that the stability bound could read.
            // With a constant-kappa parser over a floored region the
            // old build produced chi = kappa/(1.5 n_floor kB) there
            // (huge) and chi = kappa/0 = inf at unfilled ghosts, which
            // pinned the Gershgorin cap to zero: the integrator then
            // accepted its whole attempts budget in zero-length
            // substeps and dropped the entire conduction step
            // (measured on the RZ cylinder-compression EE deck;
            // kappa ~ n parsers masked the same defect as 0/0 = nan).
            if (ne <= 0.0_rt || b_arr(i, j, k, BNE::b_open) <= 0.5_rt) {
                for (int c = 0; c < NXI; ++c) {
                    x_arr(i, j, k, c) = 0.0_rt;
                }
                return;
            }
            amrex::Real const TeK = amrex::max(T_arr(i, j, k), 0.0_rt);
            amrex::Real const Te_eV = TeK * kb / qe;
            amrex::Real const B2 = b_arr(i, j, k, BNE::b_B2);
            bool const unmag = (B2 <= 0.0_rt);
            amrex::Real const ubx = b_arr(i, j, k, BNE::b_bx);
            amrex::Real const uby = b_arr(i, j, k, BNE::b_by);
            amrex::Real const ubz = b_arr(i, j, k, BNE::b_bz);

            amrex::Real chi_par =
                kappa_par_ex(ne, Te_eV, t_now) / (1.5_rt * ne * kb);
            amrex::Real chi_perp =
                (unmag && !iso_any)
                    ? chi_par
                    : kappa_perp_ex(ne, Te_eV, t_now) / (1.5_rt * ne * kb);
            chi_par = amrex::max(chi_par, 0.0_rt);
            chi_perp = amrex::max(chi_perp, 0.0_rt);

            // iso options: same semantics as the SDE path
            if (iso_full) {
                chi_par = chi_perp;
            } else if (iso_B > 0.0_rt) {
                amrex::Real const s = B2 / (B2 + iso_B * iso_B);
                chi_par = chi_perp + (chi_par - chi_perp) * s;
            }

            // Free-streaming limiter (longitudinal), same form as the
            // SDE path but on the subcycle-current T_e.
            bool const open = (b_arr(i, j, k, BNE::b_open) > 0.5_rt);
            if (f_lim > 0.0_rt && TeK > 0.0_rt && open && chi_par > 0.0_rt) {
                amrex::Real gT[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                int const node[3] = {i, j, k};
                for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                    int cp = node[g] + 1, cm = node[g] - 1;
                    if (!is_per[g]) {
                        cp = amrex::min(cp, dom_hi[g]);
                        cm = amrex::max(cm, dom_lo[g]);
                    }
                    if (cp == cm) {
                        continue;
                    }
                    int p[3] = {i, j, k}, m[3] = {i, j, k};
                    p[g] = cp;
                    m[g] = cm;
                    gT[gd2ax[g]] =
                        (T_arr(p[0], p[1], p[2]) - T_arr(m[0], m[1], m[2])) *
                        dxi_g[g] / amrex::Real(cp - cm);
                }
                amrex::Real const gparT =
                    std::abs(ubx * gT[0] + uby * gT[1] + ubz * gT[2]);
                amrex::Real const q_sp = 1.5_rt * ne * kb * chi_par * gparT;
                amrex::Real const q_fs =
                    ne * kb * TeK * std::sqrt(kb * TeK / me);
                chi_par /= (1.0_rt + q_sp / (f_lim * q_fs));
            }
            // No hop cap and no vacuum fast front here: subcycling
            // handles stability and closed floor faces handle vacuum.

            amrex::Real const dchi = chi_par - chi_perp;
            amrex::Real const ub3[3] = {ubx, uby, ubz};
            for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                for (int h = g; h < AMREX_SPACEDIM; ++h) {
                    int const sidx =
                        g * AMREX_SPACEDIM - g * (g - 1) / 2 + (h - g);
                    amrex::Real const del =
                        (gd2ax[g] == gd2ax[h]) ? 1.0_rt : 0.0_rt;
                    x_arr(i, j, k, sidx) =
                        chi_perp * del + dchi * ub3[gd2ax[g]] * ub3[gd2ax[h]];
                }
            }
        });
    }
}

/** Instantaneous temperature RHS L(T)=+div(K grad T)/(1.5 n kB).
 * Input trial T, frozen B/density context and live tensor are read-only. The
 * caller fills their ghosts and rebuilds xi before every evaluation. Writes
 * only Kout and optional budget scratch; preserves both valid and ghost T.
 * This function is shared by legacy explicit integration and future implicit
 * residuals. It does not advance time, impose physical baths, or tally heat.
 */
inline void
EvaluateConductionFDRHS (amrex::MultiFab const& Tin, amrex::MultiFab const& bne,
                         amrex::MultiFab const& xi, amrex::MultiFab& Kout,
                         amrex::MultiFab& fcache, amrex::MultiFab& rbud,
                         ConductionFDGeometry const& geometry,
                         ConductionFDOptions const& options,
                         bool const single_precision_comms) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        &Tin != &Kout && &bne != &Kout && &xi != &Kout,
        "Conduction RHS requires separate read-only input and output fields");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !options.flux_budget ||
            (options.budget_dt > 0.0_rt && fcache.nComp() >= AMREX_SPACEDIM &&
             rbud.nComp() >= 1 && fcache.nGrowVect().allGE(amrex::IntVect(1)) &&
             rbud.nGrowVect().allGE(amrex::IntVect(1))),
        "Conduction donor budget needs a positive interval and one-ghost "
        "scratch fields");
    using BNE = ConductionState;
    auto const dual_volume = geometry.dual_volume;
    auto const dxi_g = geometry.dxi_g;
    auto const is_per = geometry.is_per;
    auto const dom_lo = geometry.dom_lo;
    auto const dom_hi = geometry.dom_hi;
    auto const& period = geometry.period;
#ifdef WARPX_DIM_RZ
    auto const r_edge0 = geometry.r_edge0;
    auto const dr_rz = geometry.dr_rz;
#endif
    bool const fd4 = options.fourth_order;
    int const fd_limiter = options.limiter;
    auto const fd_lim_width = options.limiter_width;
    bool const flux_budget = options.flux_budget;
    auto const budget_floor_K = options.budget_floor_K;
    auto const budget_dt = options.budget_dt;
    amrex::Real const kb = PhysConst::kb;
    // 4th-order coefficient tables (Chacon et al. Appendix A): face
    // interpolation C0 over the middle-4 nodes of the 6-point window, and
    // the A-matrix derivative rows at window positions 1..4.
    amrex::GpuArray<amrex::Real, 4> const c0 = {
        -1.0_rt / 12.0_rt, 7.0_rt / 12.0_rt, 7.0_rt / 12.0_rt,
        -1.0_rt / 12.0_rt};
    amrex::GpuArray<amrex::Real, 24> A6{}; // rows 1..4 of A_6x6, flattened
    {
        amrex::Real const a[4][6] = {
            {-12.0_rt, -65.0_rt, 120.0_rt, -60.0_rt, 20.0_rt, -3.0_rt},
            {3.0_rt, -30.0_rt, -20.0_rt, 60.0_rt, -15.0_rt, 2.0_rt},
            {-2.0_rt, 15.0_rt, -60.0_rt, 20.0_rt, 30.0_rt, -3.0_rt},
            {3.0_rt, -20.0_rt, 60.0_rt, -120.0_rt, 65.0_rt, 12.0_rt}};
        for (int r = 0; r < 4; ++r) {
            for (int m = 0; m < 6; ++m) {
                A6[6 * r + m] = a[r][m] / 60.0_rt;
            }
        }
    }
    // D0 5-point 4th-order central first derivative (transverse cross term)
    amrex::GpuArray<amrex::Real, 5> const d0 = {
        1.0_rt / 12.0_rt, -8.0_rt / 12.0_rt, 0.0_rt, 8.0_rt / 12.0_rt,
        -1.0_rt / 12.0_rt};

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Kout, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
        amrex::Box const box = mfi.tilebox();
        amrex::Array4<amrex::Real> const& tn = Kout.array(mfi);
        amrex::Array4<amrex::Real const> const& tc = Tin.const_array(mfi);
        amrex::Array4<amrex::Real const> const& x_arr = xi.const_array(mfi);
        amrex::Array4<amrex::Real const> const& b_arr = bne.const_array(mfi);
        // Null (falsy) Array4s when the budget bound is off, so the
        // kernel below compiles to the unbounded path exactly.
        amrex::Array4<amrex::Real> const fc_arr =
            flux_budget ? fcache.array(mfi) : amrex::Array4<amrex::Real>{};
        amrex::Array4<amrex::Real> const rb_arr =
            flux_budget ? rbud.array(mfi) : amrex::Array4<amrex::Real>{};

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (b_arr(i, j, k, BNE::b_open) <= 0.5_rt) {
                tn(i, j, k) = 0.0_rt; // floored nodes keep their T_e
                // A closed node is held fixed, so it has no budget to
                // exceed: R = 1 leaves its faces unthrottled.
                if (rb_arr) {
                    rb_arr(i, j, k) = 1.0_rt;
                }
                if (fc_arr) {
                    for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                        fc_arr(i, j, k, g) = 0.0_rt;
                    }
                }
                return;
            }

            auto Tat = [&] (int const p[3]) { return tc(p[0], p[1], p[2]); };
            // clamped shift along grid dim g (non-periodic walls)
            auto shift = [&] (int const p[3], int const g, int const off,
                              int q[3]) {
                q[0] = p[0];
                q[1] = p[1];
                q[2] = p[2];
                int idx = p[g] + off;
                if (!is_per[g]) {
                    idx = amrex::min(amrex::max(idx, dom_lo[g]), dom_hi[g]);
                }
                q[g] = idx;
            };
            // central transverse derivative of T at node p along h
            auto dTh2 = [&] (int const p[3], int const h) {
                int qp[3], qm[3];
                shift(p, h, +1, qp);
                shift(p, h, -1, qm);
                if (qp[h] == qm[h]) {
                    return 0.0_rt;
                }
                return (Tat(qp) - Tat(qm)) * dxi_g[h] /
                       amrex::Real(qp[h] - qm[h]);
            };
            // 4th-order D0 transverse derivative (interior only)
            auto dTh4 = [&] (int const p[3], int const h) {
                amrex::Real der = 0.0_rt;
                for (int m = 0; m < 5; ++m) {
                    if (m == 2) {
                        continue;
                    }
                    int q[3] = {p[0], p[1], p[2]};
                    q[h] += (m - 2);
                    der += d0[m] * tc(q[0], q[1], q[2]);
                }
                return der * dxi_g[h];
            };

#ifdef WARPX_DIM_RZ
            // J = r at a node index along grid dim g: folded into
            // radial (g = 0) node fluxes; z fluxes carry J = 1
            auto Jn = [&] (int const p[3], int const g) {
                return (g == 0)
                           ? r_edge0 + amrex::Real(p[0] - dom_lo[0]) * dr_rz
                           : 1.0_rt;
            };
#else
                auto Jn = [] (int const [3], int const) {
                    return 1.0_rt;
                };
#endif

            // flux through the face between node m0 and m0 + e_g,
            // positive along +g, in u units [W/m^2]
            auto face_flux = [&] (int const m0[3], int const g) {
                // outside a non-periodic wall: no face (adiabatic)
                if (!is_per[g] &&
                    (m0[g] < dom_lo[g] || m0[g] + 1 > dom_hi[g])) {
                    return 0.0_rt;
                }
                int m1[3] = {m0[0], m0[1], m0[2]};
                m1[g] += 1;
                if (b_arr(m0[0], m0[1], m0[2], BNE::b_open) <= 0.5_rt ||
                    b_arr(m1[0], m1[1], m1[2], BNE::b_open) <= 0.5_rt) {
                    return 0.0_rt; // closed floor faces
                }
                // Harmonic (series-conductance) face density: with the
                // n-proportional kappa cap the face conductance then
                // scales with the SMALL side across a density cliff,
                // capping the tail-node rate at 2 chi/dx^2 -- the
                // arithmetic mean carries the dense side's kappa into
                // the tiny heat capacity (rate ~ nrat chi/dx^2, an
                // explicit-stability wall at deposit-tail nodes).
                amrex::Real const ne_0 = b_arr(m0[0], m0[1], m0[2], BNE::b_ne);
                amrex::Real const ne_1 = b_arr(m1[0], m1[1], m1[2], BNE::b_ne);
                amrex::Real const ne_f =
                    (ne_0 + ne_1 > 0.0_rt)
                        ? 2.0_rt * ne_0 * ne_1 / (ne_0 + ne_1)
                        : 0.0_rt;
                amrex::Real const kfac = 1.5_rt * kb * ne_f;
                int const sgg = g * AMREX_SPACEDIM - g * (g - 1) / 2;

                // 4th-order stencils need the full interior window;
                // near non-periodic walls fall back to 2nd order.
                bool use4 = fd4;
                if (use4 && !is_per[g]) {
                    use4 = (m0[g] - 2 >= dom_lo[g]) && (m0[g] + 3 <= dom_hi[g]);
                }
                if (use4) {
                    for (int h = 0; h < AMREX_SPACEDIM; ++h) {
                        if (h == g || is_per[h]) {
                            continue;
                        }
                        use4 = use4 && (m0[h] - 2 >= dom_lo[h]) &&
                               (m0[h] + 2 <= dom_hi[h]);
                    }
                }

                amrex::Real F = 0.0_rt; // chi-units flux [K m/s]
                if (use4) {
                    // co-derivative: C0 over A-row node fluxes
                    for (int l = 0; l < 4; ++l) {
                        int q[3] = {m0[0], m0[1], m0[2]};
                        q[g] += (l - 1);
                        amrex::Real der = 0.0_rt;
                        for (int m = 0; m < 6; ++m) {
                            int w[3] = {m0[0], m0[1], m0[2]};
                            w[g] += (m - 2);
                            der += A6[6 * (l) + m] * tc(w[0], w[1], w[2]);
                        }
                        F += c0[l] * Jn(q, g) * x_arr(q[0], q[1], q[2], sgg) *
                             der * dxi_g[g];
                    }
                } else {
                    F = 0.5_rt *
                        (Jn(m0, g) * x_arr(m0[0], m0[1], m0[2], sgg) +
                         Jn(m1, g) * x_arr(m1[0], m1[1], m1[2], sgg)) *
                        (Tat(m1) - Tat(m0)) * dxi_g[g];
                }

                // cross-derivative fluxes: Sharma-Hammett limited
                // transverse face gradient (sh modes) or the
                // SMART-limited pseudo-advection recast
                for (int h = 0; h < AMREX_SPACEDIM; ++h) {
                    if (h == g) {
                        continue;
                    }
                    int const gg = amrex::min(g, h);
                    int const hh = amrex::max(g, h);
                    int const sgh =
                        gg * AMREX_SPACEDIM - gg * (gg - 1) / 2 + (hh - gg);
                    if (fd_limiter == 3 || fd_limiter == 4) {
                        // Sharma-Hammett (JCP 227, 2007) flux-level
                        // extrema bound: nested-limited combination
                        // of the one-sided transverse differences at
                        // the two face nodes (2nd-order form
                        // regardless of fd_order). A shift clamped
                        // at a non-periodic wall contributes a zero
                        // one-sided difference, which zeroes the
                        // face's cross flux (adiabatic-consistent).
                        // Reads only face-adjacent values, so both
                        // neighbors compute the identical flux and
                        // the telescoping sum is preserved.
                        auto osd = [&] (int const p[3],
                                        int const sgn) -> amrex::Real {
                            int q[3];
                            shift(p, h, sgn, q);
                            if (q[h] == p[h]) {
                                return 0.0_rt;
                            }
                            return amrex::Real(sgn) * (Tat(q) - Tat(p)) *
                                   dxi_g[h];
                        };
                        amrex::Real const gf = qdsmc_sh_face_gradient(
                            osd(m0, +1), osd(m0, -1), osd(m1, +1), osd(m1, -1),
                            fd_limiter == 3);
                        F += 0.5_rt *
                             (Jn(m0, g) * x_arr(m0[0], m0[1], m0[2], sgh) +
                              Jn(m1, g) * x_arr(m1[0], m1[1], m1[2], sgh)) *
                             gf;
                        continue;
                    }
                    amrex::Real raw;
                    if (use4) {
                        raw = 0.0_rt;
                        for (int l = 0; l < 4; ++l) {
                            int q[3] = {m0[0], m0[1], m0[2]};
                            q[g] += (l - 1);
                            raw += c0[l] * Jn(q, g) *
                                   x_arr(q[0], q[1], q[2], sgh) * dTh4(q, h);
                        }
                    } else {
                        raw = 0.5_rt *
                              (Jn(m0, g) * x_arr(m0[0], m0[1], m0[2], sgh) *
                                   dTh2(m0, h) +
                               Jn(m1, g) * x_arr(m1[0], m1[1], m1[2], sgh) *
                                   dTh2(m1, h));
                    }
                    if (raw == 0.0_rt) {
                        continue;
                    }
                    amrex::Real Tf = 0.5_rt * (Tat(m0) + Tat(m1));
                    if (Tf <= 0.0_rt) {
                        continue;
                    }
                    Tf = amrex::max(Tf,
                                    1.0e-4_rt * amrex::max(Tat(m0), Tat(m1)));
                    amrex::Real const v = raw / Tf;
                    int uu[3];
                    amrex::Real tU, tD;
                    // F is +Xi*grad(T), and the temperature RHS is
                    // +div(F): the equivalent advective velocity is
                    // -v. Thus positive F takes its donor from m1.
                    // Selecting along +v would downwind this term.
                    if (v < 0.0_rt) {
                        tU = Tat(m0);
                        tD = Tat(m1);
                        shift(m0, g, -1, uu);
                    } else {
                        tU = Tat(m1);
                        tD = Tat(m0);
                        shift(m1, g, +1, uu);
                    }
                    F += v * qdsmc_fd_face_value(Tat(uu), tU, tD, fd_limiter,
                                                 fd_lim_width);
                }
                return kfac * F;
            };

            int const node[3] = {i, j, k};
            amrex::Real div = 0.0_rt;
            // Energy density per unit time LEAVING this node, summed
            // over its faces (budget path only). The hi face enters
            // div as +F and the lo face as -F, so a negative hi
            // contribution and a positive lo contribution are both
            // outflow.
            amrex::Real out_rate = 0.0_rt;
            for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                int lo[3] = {i, j, k};
                lo[g] -= 1;
                amrex::Real const f_hi = face_flux(node, g);
                amrex::Real const f_lo = face_flux(lo, g);
                if (fc_arr) {
                    fc_arr(i, j, k, g) = f_hi;
                }
#ifdef WARPX_DIM_RZ
                if (g == 0) {
                    // radial: J-folded face fluxes over the dual-
                    // cell measure Vr (finite-volume form; regular
                    // at the axis, whose inner face is absent)
                    amrex::Real const Vr = dual_volume.radial_measure(i);
                    div += (f_hi - f_lo) / Vr;
                    out_rate +=
                        (amrex::max(-f_hi, 0.0_rt) + amrex::max(f_lo, 0.0_rt)) /
                        Vr;
                    continue;
                }
#endif
                div += (f_hi - f_lo) / dual_volume.width(g, node[g]);
                out_rate +=
                    (amrex::max(-f_hi, 0.0_rt) + amrex::max(f_lo, 0.0_rt)) /
                    dual_volume.width(g, node[g]);
            }
            amrex::Real const ne0 = b_arr(i, j, k, BNE::b_ne);
            tn(i, j, k) = div / (1.5_rt * kb * ne0);

            if (rb_arr) {
                // Energy density this node holds ABOVE the floor,
                // spread over the substep: the most it may shed.
                amrex::Real const avail =
                    1.5_rt * kb * ne0 *
                    amrex::max(tc(i, j, k) - budget_floor_K, 0.0_rt) /
                    budget_dt;
                rb_arr(i, j, k) = (out_rate > avail && out_rate > 0.0_rt)
                                      ? avail / out_rate
                                      : 1.0_rt;
            }
        });
    }

    // ---- budget pass: rescale each face by its DONOR's ratio -------
    // Face F(m,g) enters node m as +F and node m+e_g as -F. So the
    // donor -- the node energy leaves -- is m when F < 0 and m+e_g
    // when F > 0. Scaling the face ONCE by that node's R keeps both
    // neighbours reading the identical value, which is what preserves
    // the telescoping sum. Nothing is injected; the heat the bound
    // withholds simply stays in the donor.
    if (flux_budget) {
        ablastr::utils::communication::FillBoundary(
            fcache, single_precision_comms, period, true);
        ablastr::utils::communication::FillBoundary(
            rbud, single_precision_comms, period, true);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(Kout, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            amrex::Box const box = mfi.tilebox();
            amrex::Array4<amrex::Real> const& tn = Kout.array(mfi);
            amrex::Array4<amrex::Real const> const& fc =
                fcache.const_array(mfi);
            amrex::Array4<amrex::Real const> const& rb = rbud.const_array(mfi);
            amrex::Array4<amrex::Real const> const& b_arr =
                bne.const_array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (b_arr(i, j, k, BNE::b_open) <= 0.5_rt) {
                    return;
                }
                amrex::Real div = 0.0_rt;
                for (int g = 0; g < AMREX_SPACEDIM; ++g) {
                    int hi[3] = {i, j, k};
                    hi[g] += 1;
                    int lo[3] = {i, j, k};
                    lo[g] -= 1;
                    amrex::Real const f_hi = fc(i, j, k, g);
                    amrex::Real const f_lo = fc(lo[0], lo[1], lo[2], g);
                    // donor of the hi face: this node if F<0, else hi
                    amrex::Real const r_hi =
                        (f_hi < 0.0_rt) ? rb(i, j, k) : rb(hi[0], hi[1], hi[2]);
                    // donor of the lo face: lo node if F<0, else this
                    amrex::Real const r_lo =
                        (f_lo < 0.0_rt) ? rb(lo[0], lo[1], lo[2]) : rb(i, j, k);
                    amrex::Real const fh = f_hi * r_hi;
                    amrex::Real const fl = f_lo * r_lo;
#ifdef WARPX_DIM_RZ
                    if (g == 0) {
                        amrex::Real const Vr = dual_volume.radial_measure(i);
                        div += (fh - fl) / Vr;
                        continue;
                    }
#endif
                    int const node[3] = {i, j, k};
                    div += (fh - fl) / dual_volume.width(g, node[g]);
                }
                amrex::Real const ne0 = b_arr(i, j, k, BNE::b_ne);
                tn(i, j, k) = div / (1.5_rt * kb * ne0);
            });
        }
    }
}
} // namespace warpx::thermal
#endif
