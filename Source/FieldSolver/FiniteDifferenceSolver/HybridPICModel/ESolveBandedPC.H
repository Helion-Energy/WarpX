/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef ESOLVE_BANDED_PC_H_
#define ESOLVE_BANDED_PC_H_

#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

#if defined(WARPX_DIM_RZ)

/**
 * \brief Z-line block-banded direct preconditioner for the inner elliptic
 *  E-solves of the theta-implicit hybrid model (RZ m = 0):
 *  SolveEThetaCurlCurlRZ (scalar ABec), SolveEPolCurlCurlRZ (coupled
 *  Er/Ez energy-form curl-curl), SolveETensorRZ (coupled 3-component
 *  tensor-Helmholtz). Selected by hybrid_pic_model.esolve_pc =
 *  block_banded (default jacobi = the legacy point-diagonal path,
 *  bit-identical).
 *
 *  The preconditioner MODEL is the screened, metric-aware COMPONENT
 *  LAPLACIAN (the operator form of Amano, J. Comput. Phys. 275, 197
 *  (2014) restricted to its per-component elliptic part), assembled
 *  ANALYTICALLY by the caller from the same frozen coefficient snapshot
 *  as the residual operator (a-coefficients, anchored gamma gates,
 *  Dirichlet rows, volume scalings, RZ metric with the axis regularity
 *  limits). In RZ m = 0 the poloidal vector Laplacian is per-component
 *  decoupled ((lap_vec E)_r = lap E_r - E_r/r^2, (lap_vec E)_z =
 *  lap E_z), the anchored vacuum-Gauss-completed residual operator IS
 *  curlcurl - grad div = -lap_vec on deep-vacuum rows (where the model is
 *  therefore exact up to discretization commutators), and on plasma rows
 *  the grad-div difference is dominated by the beta screening -- the
 *  model is spectrally equivalent and SPD with the same volume scalings,
 *  keeping preconditioned CG legal.
 *
 *  Rows are packed with one dense block per radial index (each
 *  component's z-line at that index; the periodic-z wrap and the
 *  pointwise inter-component coupling of the tensor form live inside the
 *  blocks) and factored by block-banded LU across r, once per time step
 *  (the caller passes a per-step stamp). A factor/solve round-trip check
 *  runs at every rebuild and aborts on disagreement. Applied strictly as
 *  the Krylov preconditioner: staleness or model error can only cost
 *  iterations, never correctness.
 *
 *  v1 scope: single MPI rank (any box/tile layout on that rank); the
 *  caller falls back to the Jacobi path otherwise. Host-side factor and
 *  solve; on GPU builds field data is staged through host copies.
 */
class ESolveBandedPC
{
public:

    using RT = amrex::Real;

    ESolveBandedPC () = default;

    [[nodiscard]] bool Ready (int a_stamp) const
    {
        return m_built && (m_stamp == a_stamp);
    }

    /** A factorization exists (possibly from an earlier step; using a
     *  stale one only costs Krylov iterations). */
    [[nodiscard]] bool Built () const { return m_built; }

    /** Row-entry sink handed to the row callback: add(c2, i2, j2, val)
     *  accumulates one stencil entry of the current row. Entries pointing
     *  outside the DOF windows are dropped by the assembler (the callback
     *  is responsible for Dirichlet-column elimination: keep the diagonal
     *  contribution of a pinned neighbor, emit no off-diagonal). */
    using AddFn = std::function<void(int, int, int, RT)>;
    /** Row callback: emit the stencil of row (c, i, j) -- unique DOFs
     *  only; duplicated periodic z-nodes are handled by the assembler. */
    using RowFn = std::function<void(int, int, int, AddFn const&)>;

    /** Assemble (analytically, via a_rows) + factor. a_protos provide the
     *  component staggerings/BoxArrays only. */
    void BuildRows (int a_stamp,
                    const amrex::Geometry& a_geom,
                    amrex::Vector<amrex::MultiFab const*> const& a_protos,
                    RowFn const& a_rows);

    /** z = P^{-1} r. r is read on valid boxes; z is written on valid
     *  boxes. Duplicated DOFs (periodic nodal-z pairs, nodal box seams)
     *  receive the representative solution plus the Jacobi-scaled
     *  local-vs-representative residual difference,
     *  (r_local - r_rep)/diag: P must stay nonsingular AND correctly
     *  scaled on the copy-difference subspace. Collapsing the copies
     *  makes P singular there (CG stalls exactly flat, measured); an
     *  UNSCALED identity pass-through mis-scales the difference by the
     *  raw row diagonal (~beta V) relative to the preconditioned bulk
     *  (P^{-1}A ~ 1), and the Krylov alpha ~ 1 then explodes it
     *  (measured churn at the duplicated node). Jacobi scaling contracts
     *  it at the bulk rate and keeps P SPD (a_diag = the solves' own
     *  point-Jacobi diagonals, strictly positive). */
    void Apply (amrex::Vector<amrex::MultiFab const*> const& a_r,
                amrex::Vector<amrex::MultiFab const*> const& a_diag,
                amrex::Vector<amrex::MultiFab*> const& a_z) const;

private:

    void FactorBlocks ();
    void SolvePacked (std::vector<RT>& a_x) const;
    void RoundTripCheck () const;

    // dense kernels (row-major), block-banded LU
    static void lu_factor (int n, RT* A, int* piv);
    static void lu_solve (int n, const RT* A, const int* piv, RT* b);
    static void lu_solve_T (int n, const RT* A, const int* piv, RT* b);
    static void solve_right (int m, int n, RT* X, const RT* Df, const int* piv);
    static void gemm_minus (int m, int k, int n, const RT* A, const RT* B, RT* C);
    static void gemv_minus (int m, int n, const RT* A, const RT* x, RT* y);

    template <typename F>
    static void ParFor (int n, int grain, F&& f)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(static) if (n > grain)
        for (int i = 0; i < n; i++) { f(i); }
#else
        const int nt = (n > grain)
            ? std::min(16, std::max(1, n/std::max(grain,1))) : 1;
        if (nt <= 1) {
            for (int i = 0; i < n; i++) { f(i); }
            return;
        }
        std::vector<std::thread> ts;
        ts.reserve(nt - 1);
        const int chunk = (n + nt - 1)/nt;
        for (int t = 1; t < nt; t++) {
            const int lo = t*chunk;
            const int hi = std::min(n, lo + chunk);
            if (lo >= hi) { break; }
            ts.emplace_back([lo, hi, &f] () {
                for (int i = lo; i < hi; i++) { f(i); }
            });
        }
        for (int i = 0; i < std::min(chunk, n); i++) { f(i); }
        for (auto& th : ts) { th.join(); }
#endif
    }

    // packed indexing ------------------------------------------------
    // block b holds, for each component c present at r-index rlo+b, that
    // component's unique z-line DOFs, in component order.
    [[nodiscard]] bool compInBlock (int c, int b) const
    {
        const int i = m_rlo + b;
        return (i >= m_cilo[c]) && (i <= m_cihi[c]);
    }
    [[nodiscard]] int locOf (int c, int b, int j) const
    {
        int off = 0;
        for (int cc = 0; cc < c; cc++) {
            if (compInBlock(cc, b)) { off += m_nzu[cc]; }
        }
        return off + zmap(c, j);
    }
    [[nodiscard]] int zmap (int c, int j) const
    {
        if (m_zwrap) {
            int t = (j - m_cjlo[c]) % m_nzc;
            if (t < 0) { t += m_nzc; }
            return t;
        }
        return j - m_cjlo[c];
    }

    bool m_built = false;
    int m_stamp = -1;
    int m_ncomp = 0;
    int m_rlo = 0;
    int m_nb = 0;
    bool m_zwrap = false;
    int m_nzc = 1;                    // z cells in the domain
    std::vector<int> m_cilo, m_cihi;  // per-comp r DOF range
    std::vector<int> m_cjlo, m_cjhi;  // per-comp unique z DOF range
    std::vector<int> m_nzu;           // per-comp unique z DOF count
    std::vector<amrex::Box> m_dombx;  // per-comp converted domain box

    int m_bw = 2;   // assembly half-bandwidth budget
    int m_bwf = 1;  // effective half-bandwidth (trimmed)
    std::vector<int> m_bsz;
    std::vector<long> m_boff;
    long m_ntot = 0;
    // blk[b][d]: row-major bsz[b] x bsz[b+d-bw], column block b+d-bw
    std::vector<std::vector<std::vector<RT>>> m_blk;
    std::vector<std::vector<int>> m_piv;
    mutable std::vector<std::vector<RT>> m_blk_save;  // pre-factor copy (round-trip check)
    mutable std::vector<RT> m_x;      // packed solve vector
};

inline void
ESolveBandedPC::BuildRows (int a_stamp,
                           const amrex::Geometry& a_geom,
                           amrex::Vector<amrex::MultiFab const*> const& a_protos,
                           RowFn const& a_rows)
{
    BL_PROFILE("ESolveBandedPC::BuildRows()");
    using namespace amrex;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        ParallelDescriptor::NProcs() == 1,
        "hybrid_pic_model.esolve_pc = block_banded supports a single MPI "
        "rank in v1 (use esolve_pc = jacobi on multi-rank runs)");

    m_ncomp = int(a_protos.size());
    const Box dom = a_geom.Domain();
    m_zwrap = a_geom.isPeriodic(1);
    m_nzc = dom.length(1);
    m_rlo = dom.smallEnd(0);

    m_cilo.assign(m_ncomp, 0); m_cihi.assign(m_ncomp, 0);
    m_cjlo.assign(m_ncomp, 0); m_cjhi.assign(m_ncomp, 0);
    m_nzu.assign(m_ncomp, 0);
    m_dombx.assign(m_ncomp, Box());
    int nbmax = 0;
    for (int c = 0; c < m_ncomp; c++) {
        const Box dc = amrex::convert(dom, a_protos[c]->ixType());
        m_dombx[c] = dc;
        m_cilo[c] = dc.smallEnd(0);
        m_cihi[c] = dc.bigEnd(0);
        m_cjlo[c] = dc.smallEnd(1);
        // duplicated periodic z-node folds onto the wrapped column
        m_cjhi[c] = m_zwrap ? (dc.smallEnd(1) + m_nzc - 1) : dc.bigEnd(1);
        m_nzu[c] = m_cjhi[c] - m_cjlo[c] + 1;
        nbmax = std::max(nbmax, m_cihi[c] - m_rlo + 1);
    }
    m_nb = nbmax;

    m_bsz.assign(m_nb, 0);
    m_boff.assign(m_nb, 0);
    m_ntot = 0;
    for (int b = 0; b < m_nb; b++) {
        int s = 0;
        for (int c = 0; c < m_ncomp; c++) {
            if (compInBlock(c, b)) { s += m_nzu[c]; }
        }
        m_bsz[b] = s;
        m_boff[b] = m_ntot;
        m_ntot += s;
    }
    m_x.assign(m_ntot, RT(0.0));

    // ---- allocate blocks (band budget 2; trimmed after assembly) ----
    m_bw = 2;
    m_blk.assign(m_nb, {});
    m_piv.assign(m_nb, {});
    for (int b = 0; b < m_nb; b++) {
        m_blk[b].resize(2*m_bw + 1);
        for (int d = 0; d <= 2*m_bw; d++) {
            const int bc = b + d - m_bw;
            if (bc < 0 || bc >= m_nb) { continue; }
            m_blk[b][d].assign(std::size_t(m_bsz[b])*m_bsz[bc], RT(0.0));
        }
        m_piv[b].assign(m_bsz[b], 0);
    }

    // ---- analytic assembly: one row callback per unique DOF ----
    for (int c = 0; c < m_ncomp; c++) {
        for (int b = 0; b < m_nb; b++) {
            if (!compInBlock(c, b)) { continue; }
            const int i = m_rlo + b;
            const int ncomp = m_ncomp;
            for (int j = m_cjlo[c]; j <= m_cjhi[c]; j++) {
                const int lrow = locOf(c, b, j);
                AddFn add = [&](int c2, int i2, int j2, RT v)
                {
                    if (v == RT(0.0)) { return; }
                    AMREX_ALWAYS_ASSERT(c2 >= 0 && c2 < ncomp);
                    if (m_zwrap) {
                        j2 = m_cjlo[c2]
                            + (((j2 - m_cjlo[c2]) % m_nzc) + m_nzc) % m_nzc;
                    } else if (j2 < m_cjlo[c2] || j2 > m_cjhi[c2]) {
                        return;   // outside (non-periodic caps): dropped
                    }
                    if (i2 < m_cilo[c2] || i2 > m_cihi[c2]) { return; }
                    const int bcol = i2 - m_rlo;
                    const int d = bcol - b + m_bw;
                    if (d < 0 || d > 2*m_bw) { return; }
                    auto& blkv = m_blk[b][d];
                    if (blkv.empty()) { return; }
                    blkv[std::size_t(lrow)*m_bsz[bcol] + locOf(c2, bcol, j2)]
                        += v;
                };
                a_rows(c, i, j, add);
            }
        }
    }

    // trim the effective bandwidth
    int bwf = 1;
    for (int b = 0; b < m_nb; b++) {
        for (int d = 0; d <= 2*m_bw; d++) {
            const int reach = std::abs(d - m_bw);
            if (reach <= bwf) { continue; }
            for (const RT v : m_blk[b][d]) {
                if (v != RT(0.0)) { bwf = reach; break; }
            }
        }
    }
    m_bwf = bwf;

    // pre-factor snapshot of the block diagonals actually used, for the
    // factor/solve round-trip gate
    m_blk_save.assign(std::size_t(m_nb)*(2*m_bw + 1), {});
    for (int b = 0; b < m_nb; b++) {
        for (int d = 0; d <= 2*m_bw; d++) {
            m_blk_save[std::size_t(b)*(2*m_bw + 1) + d] = m_blk[b][d];
        }
    }

    FactorBlocks();
    RoundTripCheck();

    m_stamp = a_stamp;
    m_built = true;
}

inline void ESolveBandedPC::RoundTripCheck () const
{
    // random packed vector v; w = A_model v (saved blocks); solve w; must
    // reproduce v (factor/solve defect gate -- the assembly itself is
    // analytic and visible, unlike a probed extraction)
    std::vector<RT> v(m_ntot);
    for (long i = 0; i < m_ntot; i++) {
        unsigned int h = static_cast<unsigned int>(i*2654435761u) + 977u;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        v[i] = RT(h % 20011u)/RT(20011.0) - RT(0.5);
    }
    std::vector<RT> w(m_ntot, RT(0.0));
    for (int b = 0; b < m_nb; b++) {
        for (int d = 0; d <= 2*m_bw; d++) {
            const int bc = b + d - m_bw;
            if (bc < 0 || bc >= m_nb) { continue; }
            const auto& A = m_blk_save[std::size_t(b)*(2*m_bw + 1) + d];
            if (A.empty()) { continue; }
            const int m = m_bsz[b], n = m_bsz[bc];
            for (int i = 0; i < m; i++) {
                RT s = RT(0.0);
                const RT* ai = &A[std::size_t(i)*n];
                const RT* xv = &v[m_boff[bc]];
                for (int jj = 0; jj < n; jj++) { s += ai[jj]*xv[jj]; }
                w[m_boff[b] + i] += s;
            }
        }
    }
    SolvePacked(w);
    RT num = RT(0.0), den = RT(0.0);
    for (long i = 0; i < m_ntot; i++) {
        const RT d2 = w[i] - v[i];
        num += d2*d2; den += v[i]*v[i];
    }
    const RT rel = std::sqrt(num/den);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rel <= RT(1.0e-8),
        "esolve_pc block_banded: LU round-trip failed (singular or "
        "mis-assembled preconditioner model)");
}

// ---------------- dense kernels (row-major) ----------------

inline void ESolveBandedPC::lu_factor (int n, RT* A, int* piv)
{
    constexpr int NB = 64;
    for (int kp = 0; kp < n; kp += NB) {
        const int ke = std::min(kp + NB, n);
        for (int k = kp; k < ke; k++) {
            int p = k;
            RT amax = std::abs(A[std::size_t(k)*n + k]);
            for (int i = k+1; i < n; i++) {
                const RT vv = std::abs(A[std::size_t(i)*n + k]);
                if (vv > amax) { amax = vv; p = i; }
            }
            piv[k] = p;
            if (p != k) {
                for (int j = 0; j < n; j++) {
                    std::swap(A[std::size_t(k)*n + j], A[std::size_t(p)*n + j]);
                }
            }
            const RT dkk = A[std::size_t(k)*n + k];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(dkk != RT(0.0),
                "esolve_pc block_banded: singular diagonal block in LU");
            const RT inv = RT(1.0)/dkk;
            const RT* rowk = &A[std::size_t(k)*n];
            for (int i = k+1; i < n; i++) {
                RT* rowi = &A[std::size_t(i)*n];
                rowi[k] *= inv;
                const RT lik = rowi[k];
                if (lik == RT(0.0)) { continue; }
                for (int j = k+1; j < ke; j++) { rowi[j] -= lik*rowk[j]; }
            }
        }
        if (ke >= n) { break; }
        ParFor(n - ke, 256, [&](int jj) {
            const int j = ke + jj;
            for (int i = kp+1; i < ke; i++) {
                const RT* rowi = &A[std::size_t(i)*n];
                RT s = rowi[j];
                for (int k2 = kp; k2 < i; k2++) {
                    s -= rowi[k2]*A[std::size_t(k2)*n + j];
                }
                A[std::size_t(i)*n + j] = s;
            }
        });
        ParFor(n - ke, 16, [&](int ii) {
            const int i = ke + ii;
            RT* ci = &A[std::size_t(i)*n];
            for (int k2 = kp; k2 < ke; k2++) {
                const RT a = ci[k2];
                if (a == RT(0.0)) { continue; }
                const RT* bk = &A[std::size_t(k2)*n];
                for (int j = ke; j < n; j++) { ci[j] -= a*bk[j]; }
            }
        });
    }
}

inline void ESolveBandedPC::lu_solve (int n, const RT* A, const int* piv, RT* b)
{
    for (int k = 0; k < n; k++) {
        if (piv[k] != k) { std::swap(b[k], b[piv[k]]); }
    }
    for (int i = 1; i < n; i++) {
        const RT* rowi = &A[std::size_t(i)*n];
        RT s = b[i];
        for (int j = 0; j < i; j++) { s -= rowi[j]*b[j]; }
        b[i] = s;
    }
    for (int i = n-1; i >= 0; i--) {
        const RT* rowi = &A[std::size_t(i)*n];
        RT s = b[i];
        for (int j = i+1; j < n; j++) { s -= rowi[j]*b[j]; }
        b[i] = s/rowi[i];
    }
}

inline void ESolveBandedPC::lu_solve_T (int n, const RT* A, const int* piv, RT* b)
{
    for (int i = 0; i < n; i++) {
        RT s = b[i];
        for (int j = 0; j < i; j++) { s -= A[std::size_t(j)*n + i]*b[j]; }
        b[i] = s/A[std::size_t(i)*n + i];
    }
    for (int i = n-1; i >= 0; i--) {
        RT s = b[i];
        for (int j = i+1; j < n; j++) { s -= A[std::size_t(j)*n + i]*b[j]; }
        b[i] = s;
    }
    for (int k = n-1; k >= 0; k--) {
        if (piv[k] != k) { std::swap(b[k], b[piv[k]]); }
    }
}

inline void ESolveBandedPC::solve_right (int m, int n, RT* X, const RT* Df,
                                         const int* piv)
{
    ParFor(m, 4, [&](int i) {
        lu_solve_T(n, Df, piv, &X[std::size_t(i)*n]);
    });
}

inline void ESolveBandedPC::gemm_minus (int m, int k, int n, const RT* A,
                                        const RT* B, RT* C)
{
    ParFor(m, 8, [&](int i) {
        RT* ci = &C[std::size_t(i)*n];
        const RT* ai = &A[std::size_t(i)*k];
        for (int p = 0; p < k; p++) {
            const RT a = ai[p];
            if (a == RT(0.0)) { continue; }
            const RT* bp = &B[std::size_t(p)*n];
            for (int j = 0; j < n; j++) { ci[j] -= a*bp[j]; }
        }
    });
}

inline void ESolveBandedPC::gemv_minus (int m, int n, const RT* A, const RT* x,
                                        RT* y)
{
    ParFor(m, 128, [&](int i) {
        const RT* ai = &A[std::size_t(i)*n];
        RT s = RT(0.0);
        for (int j = 0; j < n; j++) { s += ai[j]*x[j]; }
        y[i] -= s;
    });
}

inline void ESolveBandedPC::FactorBlocks ()
{
    for (int b = 0; b < m_nb; b++) {
        auto& D = m_blk[b][m_bw];
        lu_factor(m_bsz[b], D.data(), m_piv[b].data());
        for (int l = 1; l <= m_bwf && b + l < m_nb; l++) {
            auto& L = m_blk[b+l][m_bw - l];
            if (L.empty()) { continue; }
            solve_right(m_bsz[b+l], m_bsz[b], L.data(), D.data(), m_piv[b].data());
            for (int m2 = 1; m2 <= m_bwf && b + m2 < m_nb; m2++) {
                auto& U = m_blk[b][m_bw + m2];
                if (U.empty()) { continue; }
                auto& C = m_blk[b+l][m_bw - l + m2];
                if (C.empty()) { continue; }
                gemm_minus(m_bsz[b+l], m_bsz[b], m_bsz[b+m2],
                           L.data(), U.data(), C.data());
            }
        }
    }
}

inline void ESolveBandedPC::SolvePacked (std::vector<RT>& a_x) const
{
    RT* x = a_x.data();
    for (int b = 0; b < m_nb; b++) {
        for (int l = 1; l <= m_bwf && b - l >= 0; l++) {
            const auto& L = m_blk[b][m_bw - l];
            if (L.empty()) { continue; }
            gemv_minus(m_bsz[b], m_bsz[b-l], L.data(), &x[m_boff[b-l]], &x[m_boff[b]]);
        }
    }
    for (int b = m_nb - 1; b >= 0; b--) {
        for (int m2 = 1; m2 <= m_bwf && b + m2 < m_nb; m2++) {
            const auto& U = m_blk[b][m_bw + m2];
            if (U.empty()) { continue; }
            gemv_minus(m_bsz[b], m_bsz[b+m2], U.data(), &x[m_boff[b+m2]], &x[m_boff[b]]);
        }
        lu_solve(m_bsz[b], m_blk[b][m_bw].data(), m_piv[b].data(), &x[m_boff[b]]);
    }
}

inline void
ESolveBandedPC::Apply (amrex::Vector<amrex::MultiFab const*> const& a_r,
                       amrex::Vector<amrex::MultiFab const*> const& a_diag,
                       amrex::Vector<amrex::MultiFab*> const& a_z) const
{
    BL_PROFILE("ESolveBandedPC::Apply()");
    using namespace amrex;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_built,
        "ESolveBandedPC::Apply() called before Build()");

    // stage the residual to host (full valid regions; at duplicated DOFs
    // -- periodic nodal-z pairs, nodal box seams -- the last writer is the
    // packed system's representative copy)
    std::vector<std::vector<RT>> hR(m_ncomp);
    for (int c = 0; c < m_ncomp; c++) {
        hR[c].assign(std::size_t(m_dombx[c].numPts()), RT(0.0));
        const auto dlo = m_dombx[c].smallEnd();
        const auto dlen = m_dombx[c].size();
        for (MFIter mfi(*a_r[c]); mfi.isValid(); ++mfi) {
            const Box vb = mfi.validbox();
            const auto& fab = (*a_r[c])[mfi];
            const Box fb = fab.box();
            const auto flo = fb.smallEnd();
            const auto flen = fb.size();
            std::vector<RT> tmp(std::size_t(fb.numPts()));
#ifdef AMREX_USE_GPU
            amrex::Gpu::dtoh_memcpy(tmp.data(), fab.dataPtr(),
                                    tmp.size()*sizeof(RT));
#else
            std::copy_n(fab.dataPtr(), tmp.size(), tmp.data());
#endif
            const auto lo = vb.smallEnd();
            const auto hi = vb.bigEnd();
            for (int j = lo[1]; j <= hi[1]; j++) {
                for (int i = lo[0]; i <= hi[0]; i++) {
                    hR[c][std::size_t(i-dlo[0])
                          + std::size_t(dlen[0])*std::size_t(j-dlo[1])]
                        = tmp[std::size_t(i-flo[0])
                              + std::size_t(flen[0])*std::size_t(j-flo[1])];
                }
            }
        }
    }

    // pack the unique DOFs
    std::fill(m_x.begin(), m_x.end(), RT(0.0));
    for (int c = 0; c < m_ncomp; c++) {
        const auto dlo = m_dombx[c].smallEnd();
        const auto dlen = m_dombx[c].size();
        for (int b = 0; b < m_nb; b++) {
            if (!compInBlock(c, b)) { continue; }
            const int i = m_rlo + b;
            for (int j = m_cjlo[c]; j <= m_cjhi[c]; j++) {
                m_x[m_boff[b] + locOf(c, b, j)]
                    = hR[c][std::size_t(i-dlo[0])
                            + std::size_t(dlen[0])*std::size_t(j-dlo[1])];
            }
        }
    }

    SolvePacked(m_x);

    // unpack: duplicated DOFs (periodic nodal-z pairs, box seams) receive
    // the representative solution PLUS the local-vs-representative residual
    // difference. The Krylov residual's copies can disagree at roundoff/
    // deposit-noise level; a P that collapses the copies is singular on the
    // difference subspace and stalls CG exactly flat there (measured:
    // |r| pinned at the seam-inconsistency norm). The identity pass-through
    // keeps P SPD on the extended space (block-diagonal: A^{-1} on the
    // unique subspace, I on the difference subspace) -- the pc_block_banded
    // global-mode (b_local - b_gathered) rule. The term is exactly zero at
    // representative copies.
    for (int c = 0; c < m_ncomp; c++) {
        const auto dlo = m_dombx[c].smallEnd();
        const auto dlen = m_dombx[c].size();
        const int zwrap = m_zwrap ? 1 : 0;
        for (MFIter mfi(*a_z[c]); mfi.isValid(); ++mfi) {
            const Box vb = mfi.validbox();
            auto& fab = (*a_z[c])[mfi];
            const Box fb = fab.box();
            const auto flo = fb.smallEnd();
            const auto flen = fb.size();
            std::vector<RT> tmp(std::size_t(fb.numPts()), RT(0.0));
            // local residual + Jacobi-diagonal copies of this fab (for
            // the scaled difference term). NOTE: the residual and diag
            // fabs can carry different ghost depths (the tensor path hands
            // 1-ghost Krylov vectors with 0-ghost diagonals), so each
            // buffer is indexed from its OWN fab box.
            std::vector<RT> rloc(std::size_t(fb.numPts()));
            const auto& rfab = (*a_r[c])[mfi];
            const amrex::Box rfb = rfab.box();
            const auto rflo = rfb.smallEnd();
            const auto rflen = rfb.size();
            rloc.resize(std::size_t(rfb.numPts()));
#ifdef AMREX_USE_GPU
            amrex::Gpu::dtoh_memcpy(rloc.data(), rfab.dataPtr(),
                                    rloc.size()*sizeof(RT));
#else
            std::copy_n(rfab.dataPtr(), rloc.size(), rloc.data());
#endif
            std::vector<RT> dloc;
            amrex::Box dfb;
            if (a_diag[c] != nullptr) {
                const auto& dfab = (*a_diag[c])[mfi];
                dfb = dfab.box();
                dloc.resize(std::size_t(dfb.numPts()));
#ifdef AMREX_USE_GPU
                amrex::Gpu::dtoh_memcpy(dloc.data(), dfab.dataPtr(),
                                        dloc.size()*sizeof(RT));
#else
                std::copy_n(dfab.dataPtr(), dloc.size(), dloc.data());
#endif
            }
            const auto dflo = dfb.smallEnd();
            const auto dflen = dfb.size();
            const auto lo = vb.smallEnd();
            const auto hi = vb.bigEnd();
            for (int j = lo[1]; j <= hi[1]; j++) {
                int jw = j;
                if (zwrap) {
                    jw = m_cjlo[c]
                        + (((j - m_cjlo[c]) % m_nzc) + m_nzc) % m_nzc;
                }
                for (int i = lo[0]; i <= hi[0]; i++) {
                    const int b = i - m_rlo;
                    const std::size_t pf = std::size_t(i-flo[0])
                        + std::size_t(flen[0])*std::size_t(j-flo[1]);
                    const std::size_t pr2 = std::size_t(i-rflo[0])
                        + std::size_t(rflen[0])*std::size_t(j-rflo[1]);
                    const RT dj = dloc.empty() ? RT(1.0)
                        : dloc[std::size_t(i-dflo[0])
                               + std::size_t(dflen[0])*std::size_t(j-dflo[1])];
                    const RT rrep =
                        hR[c][std::size_t(i-dlo[0])
                              + std::size_t(dlen[0])*std::size_t(jw-dlo[1])];
                    tmp[pf] = m_x[m_boff[b] + locOf(c, b, j)]
                            + (rloc[pr2] - rrep)/dj;
                }
            }
#ifdef AMREX_USE_GPU
            amrex::Gpu::htod_memcpy(fab.dataPtr(), tmp.data(),
                                    tmp.size()*sizeof(RT));
#else
            std::copy_n(tmp.data(), tmp.size(), fab.dataPtr());
#endif
        }
    }
}

#endif // WARPX_DIM_RZ

#endif // ESOLVE_BANDED_PC_H_
