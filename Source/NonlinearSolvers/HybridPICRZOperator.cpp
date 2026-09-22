/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "HybridPICRZOperator.H"
#ifdef WARPX_DIM_RZ
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <cmath>

using namespace amrex::literals;
namespace hybrid_pc_rz {
amrex::IntVect
type (int c, bool magnetic = false) {
    amrex::IntVect t(1, 1);
    if (c == 0) {
        t[0] = 0;
    }
    if (c == 2) {
        t[1] = 0;
    }
    if (magnetic) {
        t = amrex::IntVect(1, 1) - t;
    }
    return t;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
to_node (amrex::Array4<amrex::Real const> const& a, int i, int j, int c,
         int n = 0) {
    if (c == 0) {
        return .5_rt * (a(i - 1, j, 0, n) + a(i, j, 0, n));
    }
    if (c == 2) {
        return .5_rt * (a(i, j - 1, 0, n) + a(i, j, 0, n));
    }
    return a(i, j, 0, n);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
to_edge (amrex::Array4<amrex::Real const> const& a, int i, int j, int c,
         int n) {
    if (c == 0) {
        return .5_rt * (a(i, j, 0, n) + a(i + 1, j, 0, n));
    }
    if (c == 2) {
        return .5_rt * (a(i, j, 0, n) + a(i, j + 1, 0, n));
    }
    return a(i, j, 0, n);
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
lap (amrex::Array4<amrex::Real const> const& a, int i, int j, int c,
     amrex::Real dr, amrex::Real dz) {
    amrex::Real const r = (i + (c == 0 ? .5_rt : 0._rt)) * dr;
    amrex::Real v =
        (a(i, j - 1, 0) - 2._rt * a(i, j, 0) + a(i, j + 1, 0)) / (dz * dz);
    if (r > 0._rt) {
        v += ((r + .5_rt * dr) * (a(i + 1, j, 0) - a(i, j, 0)) -
              (r - .5_rt * dr) * (a(i, j, 0) - a(i - 1, j, 0))) /
             (r * dr * dr);
        if (c != 2) {
            v -= a(i, j, 0) / (r * r);
        }
    } else if (c == 2) {
        // Match the native Ohm hyper-resistive axial branch at r=0.
        v += 2._rt * (a(i - 1, j, 0) - 2._rt * a(i, j, 0) + a(i + 1, j, 0)) /
             (dr * dr);
    } else {
        v = 0._rt;
    }
    return v;
}
} // namespace hybrid_pc_rz

void
HybridPICRZOperator::Define (amrex::Geometry const& geom,
                             amrex::BoxArray const& cells,
                             amrex::DistributionMapping const& dm,
                             BC const& bc_lo, BC const& bc_hi, int mg_floor) {
    AMREX_ALWAYS_ASSERT(geom.ProbLo(0) == 0. && !geom.isPeriodic(0) &&
                        geom.Domain().smallEnd() == amrex::IntVect(0, 0));
    lo = bc_lo;
    hi = bc_hi;
    amrex::BoxArray ba = cells;
    amrex::Geometry g = geom;
    while (true) {
        auto l = std::make_unique<Level>();
        l->geom = g;
        l->cells = ba;
        auto const nd = amrex::convert(ba, amrex::IntVect(1, 1));
        l->nodes = std::make_unique<MF>(nd, dm, NN, 2);
        l->cross = std::make_unique<MF>(nd, dm, 12, 1);
        l->smoother = std::make_unique<MF>(nd, dm, 9, 1);
        l->nodes->setVal(0.);
        for (int c = 0; c < 3; ++c) {
            auto const eb = amrex::convert(ba, hybrid_pc_rz::type(c));
            auto const bb = amrex::convert(ba, hybrid_pc_rz::type(c, true));
            for (auto* v :
                 {&l->x, &l->b, &l->residual, &l->action, &l->correction}) {
                (*v)[c] = std::make_unique<MF>(eb, dm, 2, 2);
                (*v)[c]->setVal(0.);
            }
            l->coeff[c] = std::make_unique<MF>(eb, dm, NC, 2);
            l->coeff[c]->setVal(0.);
            l->coeff[c]->setVal(1., Scale, 1, 2);
            l->coeff[c]->setVal(1., RowWeight, 1, 2);
            for (auto* v : {&l->physical_w, &l->electric, &l->curl_source,
                            &l->curlcurl_e, &l->ion_current}) {
                (*v)[c] = std::make_unique<MF>(eb, dm, 1, 2);
                (*v)[c]->setVal(0.);
            }
            l->curl_e[c] = std::make_unique<MF>(bb, dm, 1, 1);
            l->curl_e[c]->setVal(0.);
            l->owner[c] = l->x[c]->OwnerMask(g.periodicity());
        }
        levels.push_back(std::move(l));
        if (!ba.coarsenable(2, mg_floor) || !g.Domain().coarsenable(2)) {
            break;
        }
        ba.coarsen(2);
        g.coarsen(amrex::IntVect(2, 2));
    }
}

void
HybridPICRZOperator::Fill (MF& a, int lev, int component, bool magnetic,
                           bool coefficient) const {
    auto const& g = levels[lev]->geom;
    a.OverrideSync(g.periodicity());
    a.FillBoundary(g.periodicity());
    auto const domain = amrex::convert(g.Domain(), a.ixType());
    auto const low = amrex::lbound(domain), high = amrex::ubound(domain);
    auto const typ = a.ixType().toIntVect();
    auto const blo = lo, bhi = hi;
    bool const periodic_z = g.isPeriodic(1);
    for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto v = a.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), a.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                int ii = i, jj = j;
                amrex::Real sign = 1.;
                bool zero = false;
                int idx[2] = {i, j}, limlo[2] = {low.x, low.y},
                    limhi[2] = {high.x, high.y};
                for (int d = 0; d < 2; ++d) {
                    if (d == 1 && periodic_z) {
                        continue;
                    }
                    int const normal = d == 0 ? 0 : 2;
                    for (int side = 0; side < 2; ++side) {
                        int const wall = side == 0 ? limlo[d] : limhi[d];
                        bool const outside =
                            side == 0 ? idx[d] < wall : idx[d] > wall;
                        bool const axis = d == 0 && side == 0;
                        auto const bc = side == 0 ? blo[d] : bhi[d];
                        bool const pec = bc == FieldBoundaryType::PEC ||
                                         bc == FieldBoundaryType::PEC_Insulator;
                        bool const pmc = bc == FieldBoundaryType::PMC;
                        bool const odd =
                            !coefficient &&
                            (axis ? component != 2
                                  : ((magnetic != pmc) ? component == normal
                                                       : component != normal));
                        if (!coefficient && (axis || pec || pmc) && odd &&
                            typ[d] && idx[d] == wall) {
                            zero = true;
                        }
                        if (!outside) {
                            continue;
                        }
                        if (coefficient || axis || pec || pmc) {
                            int const reflected =
                                2 * wall - idx[d] +
                                (typ[d] ? 0 : (side == 0 ? -1 : 1));
                            if (d == 0) {
                                ii = reflected;
                            } else {
                                jj = reflected;
                            }
                            if (odd) {
                                sign = -sign;
                            }
                        } else {
                            // Other field BCs leave the physical E ghosts held.
                            // Their increments have homogeneous zero extension.
                            zero = true;
                        }
                    }
                }
                if (zero) {
                    v(i, j, k, n) = 0.;
                } else if (ii != i || jj != j) {
                    v(i, j, k, n) = sign * v(ii, jj, k, n);
                }
            });
    }
}

void
HybridPICRZOperator::FillNodes (MF& a, int lev) const {
    auto const& g = levels[lev]->geom;
    a.OverrideSync(g.periodicity());
    a.FillBoundary(g.periodicity());
    auto const hi_nd = amrex::ubound(amrex::surroundingNodes(g.Domain()));
    bool const per = g.isPeriodic(1);
    for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto v = a.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), a.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                int const ii = i < 0 ? -i : amrex::min(i, hi_nd.x);
                int const jj = per ? j : amrex::max(0, amrex::min(j, hi_nd.y));
                amrex::Real const s =
                    (i < 0 && n < 6 && n % 3 != 2) ? -1._rt : 1._rt;
                if (i == 0 && n < 6 && n % 3 != 2) {
                    v(i, j, k, n) = 0.;
                } else if (i != ii || j != jj) {
                    v(i, j, k, n) = s * v(ii, jj, k, n);
                }
            });
    }
}

void
HybridPICRZOperator::PrepareCoefficients () {
    for (int lev = 0; lev < static_cast<int>(levels.size()); ++lev) {
        auto& l = *levels[lev];
        if (lev > 0) {
            auto& f = *levels[lev - 1];
            for (amrex::MFIter mfi(*l.nodes); mfi.isValid(); ++mfi) {
                auto dst = l.nodes->array(mfi);
                auto in = f.nodes->const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), NN,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                        dst(i, j, k, n) = in(2 * i, 2 * j, k, n);
                    });
            }
            for (int c = 0; c < 3; ++c) {
                for (amrex::MFIter mfi(*l.coeff[c]); mfi.isValid(); ++mfi) {
                    auto dst = l.coeff[c]->array(mfi);
                    auto in = f.coeff[c]->const_array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(), NC,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                            dst(i, j, k, n) =
                                c == 0   ? .5_rt * (in(2 * i, 2 * j, k, n) +
                                                    in(2 * i + 1, 2 * j, k, n))
                                : c == 2 ? .5_rt * (in(2 * i, 2 * j, k, n) +
                                                    in(2 * i, 2 * j + 1, k, n))
                                         : in(2 * i, 2 * j, k, n);
                        });
                }
            }
        }
        FillNodes(*l.nodes, lev);
        for (int c = 0; c < 3; ++c) {
            Fill(*l.coeff[c], lev, c, false, true);
        }
    }
}

void
HybridPICRZOperator::Curl (Field const& out, Field const& in, int lev,
                           bool upward) const {
    auto const& g = levels[lev]->geom;
    auto const dr = g.CellSize(0), dz = g.CellSize(1);
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*out[c]); mfi.isValid(); ++mfi) {
            auto a = in[0]->const_array(mfi), b = in[1]->const_array(mfi),
                 d = in[2]->const_array(mfi);
            auto v = out[c]->array(mfi);
            // Upward curl also computes its physical ghosts from E ghosts;
            // no independent magnetic BC changes this derived field.
            auto const bx = upward ? mfi.growntilebox(1) : mfi.tilebox();
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real val = 0.;
                if (upward) {
                    if (c == 0) {
                        val = i == 0 ? 0. : -(b(i, j + 1, k) - b(i, j, k)) / dz;
                    } else if (c == 1) {
                        val = (a(i, j + 1, k) - a(i, j, k)) / dz -
                              (d(i + 1, j, k) - d(i, j, k)) / dr;
                    } else {
                        amrex::Real const r = (i + .5_rt) * dr;
                        val = ((r + .5_rt * dr) * b(i + 1, j, k) -
                               (r - .5_rt * dr) * b(i, j, k)) /
                              (r * dr);
                    }
                } else {
                    if (c == 0) {
                        val = -(b(i, j, k) - b(i, j - 1, k)) / dz;
                    } else if (c == 1) {
                        val = i == 0 ? 0.
                                     : (a(i, j, k) - a(i, j - 1, k)) / dz -
                                           (d(i, j, k) - d(i - 1, j, k)) / dr;
                    } else if (i == 0) {
                        val = 4._rt * b(i, j, k) / dr;
                    } else {
                        val = ((i + .5_rt) * b(i, j, k) -
                               (i - .5_rt) * b(i - 1, j, k)) /
                              (i * dr);
                    }
                }
                v(i, j, k) = val;
            });
        }
        // The upward curl is computed on its full one-cell halo from the
        // synchronized two-cell E halo. The downward curl consumes only
        // those local values; its output is read only on valid cells. No
        // exchange of either derived field is needed, including across MPI
        // partitions and nodal overlaps.
    }
}

void
HybridPICRZOperator::PrepareCurlSource (int lev) {
    auto& l = *levels[lev];
    bool const periodic_z = l.geom.isPeriodic(1);
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*l.curl_source[c], *l.electric[c], 0, 0, 1, 2);
        if (!pin_vector_potential) {
            continue;
        }
        auto const high = amrex::ubound(
            amrex::convert(l.geom.Domain(), l.curl_source[c]->ixType()));
        for (amrex::MFIter mfi(*l.curl_source[c]); mfi.isValid(); ++mfi) {
            auto a = l.curl_source[c]->array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                  int k) {
                // Prescribed-drive derivative of DarwinApplyABoundary.
                // These are A pins, never electric identity rows.
                if (i >= high.x || (!periodic_z && (j <= 0 || j >= high.y)) ||
                    (c == 1 && i == 0)) {
                    a(i, j, k) = 0.;
                }
            });
        }
    }
}

void
HybridPICRZOperator::ApplyPair (int lev, Pair& out, Pair const& in) {
    BL_PROFILE("HybridPICRZOperator::ApplyPair");
    auto& l = *levels[lev];
    auto const dr = l.geom.CellSize(0), dz = l.geom.CellSize(1);
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*l.electric[c], *in[c], 0, 0, 1, 0);
        for (amrex::MFIter mfi(*in[c]); mfi.isValid(); ++mfi) {
            auto q = in[c]->const_array(mfi), cf = l.coeff[c]->const_array(mfi);
            auto w = l.physical_w[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    w(i, j, k) = q(i, j, k, 1) / cf(i, j, k, Scale);
                });
        }
        Fill(*l.electric[c], lev, c, false);
        Fill(*l.physical_w[c], lev, c, false);
    }
    PrepareCurlSource(lev);
    Curl(l.curl_e, l.curl_source, lev, true);
    Curl(l.curlcurl_e, l.curl_e, lev, false);
    // Local mass response sees E* = E + D W, retaining the paper's S D term.
    if (ion_response) {
        for (int c = 0; c < 3; ++c) {
            for (amrex::MFIter mfi(*l.ion_current[c]); mfi.isValid(); ++mfi) {
                auto w = l.physical_w[c]->const_array(mfi),
                     e = l.electric[c]->const_array(mfi),
                     cf = l.coeff[c]->const_array(mfi);
                auto ji = l.ion_current[c]->array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    ji(i, j, k) = cf(i, j, k, Ion) *
                                  (e(i, j, k) + cf(i, j, k, Eta) * w(i, j, k) -
                                   cf(i, j, k, Hyper) *
                                       hybrid_pc_rz::lap(w, i, j, c, dr, dz));
                });
            }
            Fill(*l.ion_current[c], lev, c, false);
        }
    }
    bool const use_ion_response = ion_response;
    for (amrex::MFIter mfi(*l.cross); mfi.isValid(); ++mfi) {
        auto wr = l.physical_w[0]->const_array(mfi),
             wt = l.physical_w[1]->const_array(mfi),
             wz = l.physical_w[2]->const_array(mfi);
        auto fr = l.curl_e[0]->const_array(mfi),
             ft = l.curl_e[1]->const_array(mfi),
             fz = l.curl_e[2]->const_array(mfi);
        auto jr = l.ion_current[0]->const_array(mfi),
             jt = l.ion_current[1]->const_array(mfi),
             jz = l.ion_current[2]->const_array(mfi);
        auto n = l.nodes->const_array(mfi);
        auto v = l.cross->array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real const w[3] = {hybrid_pc_rz::to_node(wr, i, j, 0),
                                          wt(i, j, k),
                                          hybrid_pc_rz::to_node(wz, i, j, 2)};
                amrex::Real const f[3] = {
                    .5_rt * (fr(i, j - 1, k) + fr(i, j, k)),
                    .25_rt * (ft(i - 1, j - 1, k) + ft(i, j - 1, k) +
                              ft(i - 1, j, k) + ft(i, j, k)),
                    .5_rt * (fz(i - 1, j, k) + fz(i, j, k))};
                amrex::Real ji[3] = {0., 0., 0.};
                if (use_ion_response) {
                    ji[0] = hybrid_pc_rz::to_node(jr, i, j, 0);
                    ji[1] = jt(i, j, k);
                    ji[2] = hybrid_pc_rz::to_node(jz, i, j, 2);
                }
                for (int c = 0; c < 3; ++c) {
                    int const a = (c + 1) % 3, b = (c + 2) % 3;
                    v(i, j, k, c) = w[a] * n(i, j, k, b) - w[b] * n(i, j, k, a);
                    v(i, j, k, c + 3) = n(i, j, k, 6) * w[c];
                    v(i, j, k, c + 6) =
                        n(i, j, k, a + 3) * f[b] - n(i, j, k, b + 3) * f[a];
                    v(i, j, k, c + 9) =
                        ji[a] * n(i, j, k, b) - ji[b] * n(i, j, k, a);
                }
            });
    }
    // Each edge reads the two enclosing nodes in this same box. Their
    // values already use synchronized input halos, so cross needs no halo
    // exchange or duplicate-node reduction.
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*out[c]); mfi.isValid(); ++mfi) {
            auto v = out[c]->array(mfi);
            auto q = in[c]->const_array(mfi),
                 w = l.physical_w[c]->const_array(mfi),
                 ke = l.curlcurl_e[c]->const_array(mfi);
            auto cf = l.coeff[c]->const_array(mfi),
                 cross = l.cross->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                v(i, j, k, 0) =
                    q(i, j, k, 0) +
                    cf(i, j, k, RowWeight) *
                        (cf(i, j, k, Hall) *
                             hybrid_pc_rz::to_edge(cross, i, j, c, c) +
                         hybrid_pc_rz::to_edge(cross, i, j, c, c + 3) +
                         cf(i, j, k, Eta) * w(i, j, k) -
                         cf(i, j, k, Hyper) *
                             hybrid_pc_rz::lap(w, i, j, c, dr, dz) +
                         cf(i, j, k, InvRho) *
                             (hybrid_pc_rz::to_edge(cross, i, j, c, c + 9) -
                              hybrid_pc_rz::to_edge(cross, i, j, c, c + 6)));
                v(i, j, k, 1) =
                    q(i, j, k, 1) - cf(i, j, k, Scale) * ke(i, j, k);
            });
        }
        RestoreRows(*out[c], *in[c], lev, c);
    }
}

void
HybridPICRZOperator::RestoreRows (MF& out, MF const& rhs, int lev,
                                  int c) const {
    auto const dom = amrex::convert(levels[lev]->geom.Domain(), out.ixType());
    auto const high = amrex::ubound(dom);
    auto const typ = out.ixType().toIntVect();
    auto const blo = lo, bhi = hi;
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        auto o = out.array(mfi);
        auto r = rhs.const_array(mfi);
        auto cf = levels[lev]->coeff[c]->const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), out.nComp(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                bool pin = (c == 1 && i == 0);
                int const idx[2] = {i, j}, h[2] = {high.x, high.y};
                for (int d = 0; d < 2; ++d) {
                    bool const tangential = c != (d == 0 ? 0 : 2);
                    if (tangential && typ[d]) {
                        if (idx[d] == 0 &&
                            (blo[d] == FieldBoundaryType::PEC ||
                             blo[d] == FieldBoundaryType::PEC_Insulator)) {
                            pin = true;
                        }
                        if (idx[d] == h[d] &&
                            (bhi[d] == FieldBoundaryType::PEC ||
                             bhi[d] == FieldBoundaryType::PEC_Insulator)) {
                            pin = true;
                        }
                    }
                }
                if (pin || (n == 0 && cf(i, j, k, RowWeight) == 0.)) {
                    o(i, j, k, n) = r(i, j, k, n);
                }
            });
    }
}

HybridPICRZOperator::Pair
HybridPICRZOperator::makeVecRHS () const {
    Pair out;
    for (int c = 0; c < 3; ++c) {
        auto const& x = *levels[0]->x[c];
        out[c] = std::make_unique<MF>(x.boxArray(), x.DistributionMap(), 2, 2);
        out[c]->setVal(0.);
    }
    return out;
}
void
HybridPICRZOperator::assign (Pair& a, Pair const& b) const {
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*a[c], *b[c], 0, 0, 2, 0);
    }
}
void
HybridPICRZOperator::setToZero (Pair& a) const {
    for (auto& x : a) {
        x->setVal(0.);
    }
}
void
HybridPICRZOperator::scale (Pair& a, RT s) const {
    for (auto& x : a) {
        x->mult(s, 0, 2, 0);
    }
}
void
HybridPICRZOperator::increment (Pair& a, Pair const& b, RT s) const {
    for (int c = 0; c < 3; ++c) {
        MF::Saxpy(*a[c], s, *b[c], 0, 0, 2, 0);
    }
}
void
HybridPICRZOperator::linComb (Pair& a, RT s, Pair const& b, RT t,
                              Pair const& d) const {
    for (int c = 0; c < 3; ++c) {
        MF::LinComb(*a[c], s, *b[c], 0, t, *d[c], 0, 0, 2, 0);
    }
}
HybridPICRZOperator::RT
HybridPICRZOperator::dotProduct (Pair const& x, Pair const& y) const {
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<RT> data(op);
    auto const dr = levels[0]->geom.CellSize(0),
               rmax = levels[0]->geom.ProbHi(0);
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*x[c]); mfi.isValid(); ++mfi) {
            auto a = x[c]->const_array(mfi), b = y[c]->const_array(mfi);
            auto mask = levels[0]->owner[c]->const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j,
                                         int k) -> amrex::GpuTuple<RT> {
                        RT const r = (i + (c == 0 ? .5_rt : 0._rt)) * dr;
                        RT const rl = amrex::max(0._rt, r - .5_rt * dr),
                                 rh = amrex::min(rmax, r + .5_rt * dr);
                        return {mask(i, j, k) *
                                (.5_rt * (rh * rh - rl * rl) / dr) *
                                (a(i, j, k, 0) * b(i, j, k, 0) +
                                 a(i, j, k, 1) * b(i, j, k, 1))};
                    });
        }
    }
    RT result = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(result);
    return result;
}
HybridPICRZOperator::RT
HybridPICRZOperator::norm2 (Pair const& a) const {
    return std::sqrt(dotProduct(a, a));
}

void
HybridPICRZOperator::Smooth (int lev, int sweeps) {
    BL_PROFILE("HybridPICRZOperator::Smooth");
    auto& l = *levels[lev];
    auto const dr = l.geom.CellSize(0), dz = l.geom.CellSize(1);
    RT const l2 = 4._rt / (dr * dr) + 4._rt / (dz * dz),
             l1 = 2._rt / dr + 2._rt / dz;
    RT const chi = whistler_defect, damp = sigma, dampw = sigma_w;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        ApplyPair(lev, l.action, l.x);
        linComb(l.residual, 1., l.b, -1., l.action);
        for (int c = 0; c < 3; ++c) {
            Fill(*l.residual[c], lev, c, false);
        }
        for (amrex::MFIter mfi(*l.smoother); mfi.isValid(); ++mfi) {
            auto r0 = l.residual[0]->const_array(mfi),
                 r1 = l.residual[1]->const_array(mfi),
                 r2 = l.residual[2]->const_array(mfi);
            auto c0 = l.coeff[0]->const_array(mfi),
                 c1 = l.coeff[1]->const_array(mfi),
                 c2 = l.coeff[2]->const_array(mfi);
            auto n = l.nodes->const_array(mfi);
            auto v = l.smoother->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    RT const bm = std::sqrt(n(i, j, k, 0) * n(i, j, k, 0) +
                                            n(i, j, k, 1) * n(i, j, k, 1) +
                                            n(i, j, k, 2) * n(i, j, k, 2));
                    RT b[3] = {0., 0., 0.};
                    if (bm > 0.) {
                        for (int c = 0; c < 3; ++c) {
                            b[c] = n(i, j, k, c) / bm;
                        }
                    }
                    RT const aeta = (hybrid_pc_rz::to_node(c0, i, j, 0, Eta) +
                                     c1(i, j, k, Eta) +
                                     hybrid_pc_rz::to_node(c2, i, j, 2, Eta)) /
                                    3._rt;
                    RT const ah = (hybrid_pc_rz::to_node(c0, i, j, 0, Hyper) +
                                   c1(i, j, k, Hyper) +
                                   hybrid_pc_rz::to_node(c2, i, j, 2, Hyper)) /
                                  3._rt;
                    RT const h = (hybrid_pc_rz::to_node(c0, i, j, 0, Hall) +
                                  c1(i, j, k, Hall) +
                                  hybrid_pc_rz::to_node(c2, i, j, 2, Hall)) /
                                 3._rt;
                    RT const ir = (hybrid_pc_rz::to_node(c0, i, j, 0, InvRho) +
                                   c1(i, j, k, InvRho) +
                                   hybrid_pc_rz::to_node(c2, i, j, 2, InvRho)) /
                                  3._rt;
                    RT const mm = (hybrid_pc_rz::to_node(c0, i, j, 0, Ion) +
                                   c1(i, j, k, Ion) +
                                   hybrid_pc_rz::to_node(c2, i, j, 2, Ion)) /
                                  3._rt;
                    RT const u = ir * std::sqrt(n(i, j, k, 3) * n(i, j, k, 3) +
                                                n(i, j, k, 4) * n(i, j, k, 4) +
                                                n(i, j, k, 5) * n(i, j, k, 5));
                    RT const weight[3] = {
                        hybrid_pc_rz::to_node(c0, i, j, 0, RowWeight),
                        c1(i, j, k, RowWeight),
                        hybrid_pc_rz::to_node(c2, i, j, 2, RowWeight)};
                    RT const au = aeta + n(i, j, k, 6) + ah * l2;
                    RT const diagonal =
                        l1 * u + l2 * (aeta + n(i, j, k, 6) + 2._rt * ah * l2);
                    RT const rotation =
                        bm *
                        (chi * l2 * (h + mm * ir * (aeta + ah * l2)) + mm * ir);
                    RT alpha[3], beta[3];
                    for (int c = 0; c < 3; ++c) {
                        alpha[c] = 1._rt + weight[c] * diagonal;
                        beta[c] = weight[c] * rotation;
                    }
                    RT const rw[3] = {
                        .5_rt * (r0(i - 1, j, k, 1) / c0(i - 1, j, k, Scale) +
                                 r0(i, j, k, 1) / c0(i, j, k, Scale)),
                        r1(i, j, k, 1) / c1(i, j, k, Scale),
                        .5_rt * (r2(i, j - 1, k, 1) / c2(i, j - 1, k, Scale) +
                                 r2(i, j, k, 1) / c2(i, j, k, Scale))};
                    RT const re[3] = {hybrid_pc_rz::to_node(r0, i, j, 0),
                                      r1(i, j, k, 0),
                                      hybrid_pc_rz::to_node(r2, i, j, 2)};
                    RT s[3], diag[3];
                    for (int c = 0; c < 3; ++c) {
                        int const a = (c + 1) % 3, d = (c + 2) % 3;
                        diag[c] = re[c] - weight[c] * au * rw[c];
                        s[c] = diag[c] - weight[c] * bm *
                                             (h + mm * ir * (aeta + ah * l2)) *
                                             (rw[a] * b[d] - rw[d] * b[a]);
                    }
                    // Invert diag(alpha)+diag(beta)*(v cross b). A
                    // recovered component has its own identity row, even
                    // when the other two components retain Hall coupling.
                    // The positive determinant avoids subtracting large
                    // terms at high whistler CFL.
                    RT const det = alpha[0] * alpha[1] * alpha[2] +
                                   alpha[0] * beta[1] * beta[2] * b[0] * b[0] +
                                   alpha[1] * beta[0] * beta[2] * b[1] * b[1] +
                                   alpha[2] * beta[0] * beta[1] * b[2] * b[2];
                    for (int c = 0; c < 3; ++c) {
                        int const a = (c + 1) % 3, d = (c + 2) % 3;
                        RT const inverse = ((alpha[a] * alpha[d] +
                                             beta[a] * beta[d] * b[c] * b[c]) *
                                                s[c] +
                                            (beta[c] * beta[d] * b[c] * b[a] -
                                             beta[c] * alpha[d] * b[d]) *
                                                s[a] +
                                            (beta[c] * beta[a] * b[c] * b[d] +
                                             beta[c] * alpha[a] * b[a]) *
                                                s[d]) /
                                           det;
                        v(i, j, k, c) = inverse - diag[c] / alpha[c];
                        v(i, j, k, c + 3) = alpha[c];
                        v(i, j, k, c + 6) = weight[c] * au;
                    }
                });
        }
        // The following edge scatter reads only this box's valid nodes.
        for (int c = 0; c < 3; ++c) {
            for (amrex::MFIter mfi(*l.x[c]); mfi.isValid(); ++mfi) {
                auto x = l.x[c]->array(mfi);
                auto r = l.residual[c]->const_array(mfi),
                     cf = l.coeff[c]->const_array(mfi),
                     sm = l.smoother->const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    RT const alpha = hybrid_pc_rz::to_edge(sm, i, j, c, c + 3),
                             au = hybrid_pc_rz::to_edge(sm, i, j, c, c + 6);
                    RT const de = (r(i, j, k, 0) -
                                   au * r(i, j, k, 1) / cf(i, j, k, Scale)) /
                                      alpha +
                                  hybrid_pc_rz::to_edge(sm, i, j, c, c);
                    x(i, j, k, 0) += damp * de;
                    x(i, j, k, 1) +=
                        dampw *
                        (r(i, j, k, 1) + cf(i, j, k, Scale) * chi * l2 * de);
                });
            }
            RestoreRows(*l.x[c], *l.b[c], lev, c);
        }
    }
}

void
HybridPICRZOperator::Restrict (int lev) {
    auto& f = *levels[lev];
    auto& c = *levels[lev + 1];
    ApplyPair(lev, f.action, f.x);
    linComb(f.residual, 1., f.b, -1., f.action);
    for (int d = 0; d < 3; ++d) {
        Fill(*f.residual[d], lev, d, false);
        bool const nr = d != 0, nz = d != 2;
        for (amrex::MFIter mfi(*c.b[d]); mfi.isValid(); ++mfi) {
            auto out = c.b[d]->array(mfi);
            auto in = f.residual[d]->const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), 2,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    RT value = 0.;
                    for (int a = nr ? -1 : 0; a <= 1; ++a) {
                        for (int b = nz ? -1 : 0; b <= 1; ++b) {
                            RT const wr =
                                         nr ? (a == 0 ? .5_rt : .25_rt) : .5_rt,
                                     wz =
                                         nz ? (b == 0 ? .5_rt : .25_rt) : .5_rt;
                            value += wr * wz * in(2 * i + a, 2 * j + b, k, n);
                        }
                    }
                    out(i, j, k, n) = value;
                });
        }
    }
    setToZero(c.x);
}
void
HybridPICRZOperator::Prolong (int lev) {
    auto& f = *levels[lev];
    auto& c = *levels[lev + 1];
    for (int d = 0; d < 3; ++d) {
        Fill(*c.x[d], lev + 1, d, false);
        bool const nr = d != 0, nz = d != 2;
        for (amrex::MFIter mfi(*f.x[d]); mfi.isValid(); ++mfi) {
            auto out = f.x[d]->array(mfi);
            auto in = c.x[d]->const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), 2,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    RT const x = .5_rt * i + (nr ? 0._rt : -.25_rt),
                             z = .5_rt * j + (nz ? 0._rt : -.25_rt);
                    int const ii = static_cast<int>(std::floor(x)),
                              jj = static_cast<int>(std::floor(z));
                    RT const a = x - ii, b = z - jj;
                    out(i, j, k, n) +=
                        (1._rt - a) * (1._rt - b) * in(ii, jj, k, n) +
                        a * (1._rt - b) * in(ii + 1, jj, k, n) +
                        (1._rt - a) * b * in(ii, jj + 1, k, n) +
                        a * b * in(ii + 1, jj + 1, k, n);
                });
        }
        RestoreRows(*f.x[d], *f.b[d], lev, d);
    }
}
void
HybridPICRZOperator::Vcycle (int lev) {
    if (lev + 1 == static_cast<int>(levels.size())) {
        Smooth(lev, levels.size() == 1 ? 24 : 8);
        return;
    }
    Smooth(lev, 3);
    Restrict(lev);
    Vcycle(lev + 1);
    Prolong(lev);
    Smooth(lev, 3);
}
void
HybridPICRZOperator::precond (Pair& out, Pair const& in) {
    assign(levels[0]->b, in);
    setToZero(levels[0]->x);
    Vcycle(0);
    assign(out, levels[0]->x);
}
void
HybridPICRZOperator::Solve (Field const& out, Field const& rhs, int inner_max,
                            RT inner_rtol, int vcycles) {
    BL_PROFILE("HybridPICRZOperator::Solve");
    auto b = makeVecRHS(), x = makeVecLHS();
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*b[c], *rhs[c], 0, 0, 1, 0);
    }
    if (inner_max > 0) {
        FlexibleGMRES<Pair, HybridPICRZOperator> solver;
        solver.define(*this);
        solver.setRestartLength(inner_max);
        solver.setVerbose(verbose ? 1 : 0);
        solver.solve(x, b, inner_rtol, 0., inner_max);
        last_iterations = solver.getNumIters();
        last_residual = solver.getResidualNorm();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(last_residual),
            "Non-finite native RZ Hall preconditioner residual");
    } else {
        assign(levels[0]->b, b);
        setToZero(levels[0]->x);
        for (int n = 0; n < vcycles; ++n) {
            Vcycle(0);
        }
        assign(x, levels[0]->x);
        last_iterations = vcycles;
    }
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*out[c], *x[c], 0, 0, 1, 0);
        RestoreRows(*out[c], *rhs[c], 0, c);
    }
}
void
HybridPICRZOperator::ApplyField (Field const& out, Field const& in) {
    auto q = makeVecRHS(), v = makeVecRHS();
    auto& l = *levels[0];
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*q[c], *in[c], 0, 0, 1, 0);
        MF::Copy(*l.electric[c], *in[c], 0, 0, 1, 0);
        Fill(*l.electric[c], 0, c, false);
    }
    PrepareCurlSource(0);
    Curl(l.curl_e, l.curl_source, 0, true);
    Curl(l.curlcurl_e, l.curl_e, 0, false);
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*q[c]); mfi.isValid(); ++mfi) {
            auto x = q[c]->array(mfi);
            auto k = l.curlcurl_e[c]->const_array(mfi),
                 cf = l.coeff[c]->const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int z) {
                    x(i, j, z, 1) = cf(i, j, z, Scale) * k(i, j, z);
                });
        }
    }
    ApplyPair(0, v, q);
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*out[c], *v[c], 0, 0, 1, 0);
    }
}
#endif
