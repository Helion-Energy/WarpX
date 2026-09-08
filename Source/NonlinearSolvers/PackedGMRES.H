/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PACKED_GMRES_H_
#define WARPX_PACKED_GMRES_H_

#include "LinearSolver.H"
#include "Utils/TextMsg.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_Config.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuReduce.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_TableData.H>
#include <AMReX_Vector.H>

#if defined(AMREX_USE_CUDA) && defined(WARPX_USE_CUBLAS)
#include <cublas_v2.h>
#define WARPX_PACKED_GMRES_HAS_BLAS 1
#else
#define WARPX_PACKED_GMRES_HAS_BLAS 0
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

/**
 * \brief Restarted GMRES on a packed (serialized) copy of the solver vector.
 *
 * Algorithmically this is AMReX's GMRES<V,M> (AMReX_GMRES.H: restarted
 * GMRES, right preconditioning, two passes of classical Gram-Schmidt,
 * Givens rotations on the Hessenberg matrix, the same convergence test and
 * restart semantics), with one difference in the data layout: the Krylov
 * basis is kept as a single contiguous device array of restart_length+1
 * packed vectors instead of restart_length+1 solver vectors of many
 * MultiFabs. The orthogonalization then costs two matrix-vector products
 * per Gram-Schmidt pass on that array (a GEMV against the basis for the
 * it+1 inner products, a GEMV for the it+1 updates) instead of 2*(it+1)
 * inner products and 2*(it+1) Saxpys of one kernel launch per MultiFab
 * each plus one host synchronization per inner product. On a GPU this
 * removes the launch latency and synchronization cost that dominates the
 * orthogonalization of a multi-block state (measured 75% of the wall time
 * of a production theta-implicit MHD run: one kernel per MultiFab of the
 * state per retained Krylov vector per iteration).
 *
 * The packing is the serialized degree-of-freedom layout the vector class
 * already provides (V::copyTo / V::copyFrom, V::nDOF_local): every grid
 * value whose ownership mask is set is written once, divided by its block
 * scale. Because V::dotProduct sums mask-selected products divided by the
 * squared block scales, the plain Euclidean inner product of two packed
 * vectors equals V::dotProduct of the corresponding solver vectors (the
 * summation order differs, nothing else), so the Krylov process sees the
 * same inner product as the unpacked solver. The operator (matrix-free
 * Jacobian) and the preconditioner are applied on solver vectors: each
 * iteration unpacks the current basis vector, applies P^-1 and A through
 * the operator class, and packs the result (two copies per iteration).
 *
 * The template parameters follow LinearSolver<Vec,Ops>: V is the solver
 * vector class (WarpXSolverVec), M the linear operator (JacobianFunctionMF)
 * with the AMReX GMRES interface (apply, precond, assign, increment,
 * setToZero, makeVecLHS/RHS, norm2). In addition V must provide
 * nDOF_local(), copyTo(RT*) and copyFrom(const RT*).
 *
 * Inputs (ParmParse prefix "packed_gmres"):
 *   gemv = auto | blas | native   orthogonalization kernels: cuBLAS GEMV
 *                                 (CUDA builds configured with cuBLAS) or
 *                                 the portable fused kernels; auto picks
 *                                 blas when available.
 *   self_test = 0 | 1             run the packed-versus-solver-vector
 *                                 consistency checks at the first solve
 *                                 (inner products, norms, axpy, lincomb,
 *                                 the pack/unpack round trip and the GEMV
 *                                 kernels) and abort on a mismatch.
 * The restart length, iteration cap and tolerances come from the Newton
 * solver (newton.linear_solver = packed_gmres reads amrex_gmres.*, then
 * packed_gmres.*, then the legacy gmres.* keys).
 */
template <typename V, typename M>
class PackedGMRES : public LinearSolver<V,M>
{
public:

    using RT = typename M::RT; // double or float

    PackedGMRES ();

    ~PackedGMRES () override;

    // Prohibit Move and Copy operations
    PackedGMRES (const PackedGMRES&) = delete;
    PackedGMRES& operator= (const PackedGMRES&) = delete;
    PackedGMRES (PackedGMRES&&) noexcept = delete;
    PackedGMRES& operator= (PackedGMRES&&) noexcept = delete;

    //! Defines with a reference to Ops. It's the user's responsibility to
    //! keep the Ops object alive for the solver to be functional. This
    //! function must be called before solve() can be called.
    void define (M& linop) override;

    /**
     * \brief Solve the linear system
     *
     * \param a_sol     unknowns, i.e., x in A x = b.
     * \param a_rhs     RHS, i.e., b in A x = b.
     * \param a_tol_rel relative tolerance.
     * \param a_tol_abs absolute tolerance.
     * \param a_its     optional argument specifying the maximum number of iterations.
     */
    void solve (V& a_sol,
                V const& a_rhs,
                RT a_tol_rel,
                RT a_tol_abs,
                int a_its=-1) override;

    //! Sets verbosity.
    void setVerbose (int v) override { m_verbose = v; }

    //! Sets restart length. The default is 30.
    void setRestartLength (int rl) override;

    //! Sets the max number of iterations
    void setMaxIters (int niters) override { m_maxiter = niters; }

    //! Gets the number of iterations.
    [[nodiscard]] int getNumIters () const override { return m_its; }

    //! Gets the solver status.
    [[nodiscard]] int getStatus () const override { return m_status; }

    //! Gets the 2-norm of the residual.
    [[nodiscard]] RT getResidualNorm () const override { return m_res; }

    //! Name of the orthogonalization kernels in use ("blas" or "native").
    [[nodiscard]] std::string gemvBackend () const { return m_use_blas ? "blas" : "native"; }

    // --- packed kernels (implementation detail; public because the CUDA
    // extended device lambdas they launch may not live in private members)
    //! a_out[j] = sum_i basis_j[i] * a_w[i] for j < a_ncols (global sum over ranks)
    void dot_many (const RT* a_basis, int a_ncols, const RT* a_w, RT* a_out);
    //! a_w += a_sign * sum_j a_coef[j] basis_j (a_coef on the host)
    void axpy_many (const RT* a_basis, int a_ncols, const RT* a_coef, RT a_sign, RT* a_w);
    void scale (RT* a_w, RT a_fac);
    //! a_out = a*x + b*y
    void axpby (RT* a_out, RT a, const RT* a_x, RT b, const RT* a_y);
    void set_zero (RT* a_w);

private:

    // --- GMRES bookkeeping (ported from AMReX_GMRES.H) ---------------------
    void allocate_scratch ();
    void allocate_packed (V const& a_model);
    void cycle (V& a_xx, int& a_status, int& a_itcount, RT& a_rnorm0);
    void build_solution (V& a_xx, int it);
    void compute_residual (V const& a_xx);
    [[nodiscard]] bool converged (RT r0, RT r) const;
    void gram_schmidt_orthogonalization (int it);
    void update_hessenberg (int it, bool happyend, RT& res);

    //! column j of the packed Krylov basis
    [[nodiscard]] RT* basis (int j) { return m_basis.data() + std::size_t(j) * std::size_t(m_n); }
    [[nodiscard]] const RT* basis (int j) const { return m_basis.data() + std::size_t(j) * std::size_t(m_n); }
    //! global squared 2-norm of a packed vector
    [[nodiscard]] RT norm2_squared (const RT* a_w);
    void copy_packed (RT* a_dst, const RT* a_src);

    void pack (V const& a_v, RT* a_dst) { BL_PROFILE("PackedGMRES::pack"); a_v.copyTo(a_dst); }
    void unpack (const RT* a_src, V& a_v) { BL_PROFILE("PackedGMRES::unpack"); a_v.copyFrom(a_src); }

    void run_self_test (V const& a_model);

    int m_verbose = 0;
    int m_maxiter = 2000;
    int m_its = 0;
    int m_status = -1;
    int m_restrtlen = 30;
    RT m_res = std::numeric_limits<RT>::max();
    RT m_rtol = RT(0);
    RT m_atol = RT(0);
    amrex::Vector<RT> m_hh_1d;
    amrex::Vector<RT> m_hes_1d;
    amrex::Table2D<RT> m_hh;
    amrex::Table2D<RT> m_hes;
    amrex::Vector<RT> m_grs;
    amrex::Vector<RT> m_cc;
    amrex::Vector<RT> m_ss;
    amrex::Vector<RT> m_lhh;

    std::unique_ptr<V> m_v_tmp_rhs;
    std::unique_ptr<V> m_v_tmp_lhs;
    M* m_linop = nullptr;

    // packed storage (device memory on GPU builds)
    int m_n = 0;                                //!< local packed length
    bool m_packed_allocated = false;
    amrex::Gpu::DeviceVector<RT> m_basis;       //!< (restart_length+1) x m_n
    amrex::Gpu::DeviceVector<RT> m_b;           //!< packed right-hand side
    amrex::Gpu::DeviceVector<RT> m_tmp;         //!< packed scratch
    amrex::Gpu::DeviceVector<RT> m_coef_d;      //!< coefficients for axpy_many
    amrex::Gpu::DeviceVector<RT> m_dots_d;      //!< GEMV output (blas path)
    amrex::Gpu::DeviceVector<RT> m_partials_d;  //!< per-block partial sums (native GPU path)
    amrex::Gpu::PinnedVector<RT> m_partials_h;
    int m_nblocks = 1;

    bool m_use_blas = false;
    std::string m_gemv_request = "auto";
    bool m_self_test = false;
    bool m_self_test_done = false;

#if WARPX_PACKED_GMRES_HAS_BLAS
    cublasHandle_t m_blas_handle = nullptr;
    void blas_check (cublasStatus_t status, const char* what) const;
#endif
};

// ---------------------------------------------------------------------------

template <typename V, typename M>
PackedGMRES<V,M>::PackedGMRES ()
{
    const amrex::ParmParse pp("packed_gmres");
    pp.query("gemv", m_gemv_request);
    pp.query("self_test", m_self_test);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_gemv_request == "auto" || m_gemv_request == "blas" || m_gemv_request == "native",
        "packed_gmres.gemv must be auto, blas or native");
    if (m_gemv_request == "blas") {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WARPX_PACKED_GMRES_HAS_BLAS,
            "packed_gmres.gemv = blas requires a CUDA build configured with WarpX_CUBLAS=ON");
    }
    m_use_blas = (WARPX_PACKED_GMRES_HAS_BLAS != 0) && (m_gemv_request != "native");
#if WARPX_PACKED_GMRES_HAS_BLAS
    if (m_use_blas) {
        blas_check(cublasCreate(&m_blas_handle), "cublasCreate");
        blas_check(cublasSetPointerMode(m_blas_handle, CUBLAS_POINTER_MODE_HOST),
                   "cublasSetPointerMode");
    }
#endif
    allocate_scratch();
}

template <typename V, typename M>
PackedGMRES<V,M>::~PackedGMRES ()
{
#if WARPX_PACKED_GMRES_HAS_BLAS
    if (m_blas_handle != nullptr) { cublasDestroy(m_blas_handle); }
#endif
}

#if WARPX_PACKED_GMRES_HAS_BLAS
template <typename V, typename M>
void PackedGMRES<V,M>::blas_check (cublasStatus_t status, const char* what) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        status == CUBLAS_STATUS_SUCCESS,
        std::string("PackedGMRES: ") + what + " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status)));
}
#endif

template <typename V, typename M>
void PackedGMRES<V,M>::allocate_scratch ()
{
    const int rs = m_restrtlen;

    m_hh_1d.resize(std::size_t(rs + 2) * (rs + 1));
    m_hh = amrex::Table2D<RT>(m_hh_1d.data(), {0,0}, {rs+1,rs}); // (0:rs+1,0:rs)

    m_hes_1d.resize(std::size_t(rs + 2) * (rs + 1));
    m_hes = amrex::Table2D<RT>(m_hes_1d.data(), {0,0}, {rs+1,rs}); // (0:rs+1,0:rs)

    m_grs.resize(rs + 2);
    m_cc.resize(rs + 1);
    m_ss.resize(rs + 1);
    m_lhh.resize(rs + 1);
}

template <typename V, typename M>
void PackedGMRES<V,M>::setRestartLength (int rl)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rl > 0, "PackedGMRES: restart length must be positive");
    if (m_restrtlen != rl) {
        m_restrtlen = rl;
        allocate_scratch();
        m_packed_allocated = false; // the basis is re-sized at the next solve
    }
}

template <typename V, typename M>
void PackedGMRES<V,M>::define (M& linop)
{
    m_its = 0;
    m_status = -1;
    m_res = std::numeric_limits<RT>::max();
    m_v_tmp_rhs.reset();
    m_v_tmp_lhs.reset();
    m_packed_allocated = false;
    m_linop = &linop;
    amrex::Print() << "PackedGMRES: packed Krylov basis, CGS2 orthogonalization through "
                   << gemvBackend() << " kernels"
                   << (m_self_test ? ", self test at the first solve" : "") << "\n";
}

template <typename V, typename M>
void PackedGMRES<V,M>::allocate_packed (V const& a_model)
{
    BL_PROFILE("PackedGMRES::allocate_packed()");
    const amrex::Long n_long = a_model.nDOF_local();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_long >= 0 && n_long < static_cast<amrex::Long>(std::numeric_limits<int>::max()),
        "PackedGMRES: the local packed vector length must fit in an int");
    m_n = static_cast<int>(n_long);
    const auto n = std::size_t(m_n);
    m_basis.resize(std::size_t(m_restrtlen + 1) * n);
    m_b.resize(n);
    m_tmp.resize(n);
    m_coef_d.resize(std::size_t(m_restrtlen + 1));
    m_dots_d.resize(std::size_t(m_restrtlen + 1));
    // native GPU reduction: grid of at most 1024 blocks of 256 threads
    m_nblocks = std::max(1, std::min(1024, (m_n + 255) / 256));
    m_partials_d.resize(std::size_t(m_restrtlen + 1) * std::size_t(m_nblocks));
    m_partials_h.resize(std::size_t(m_restrtlen + 1) * std::size_t(m_nblocks));
    m_packed_allocated = true;
    if (m_verbose > 0) {
        amrex::Print() << "PackedGMRES: allocated basis of " << m_restrtlen + 1
                       << " x " << m_n << " local values ("
                       << double(std::size_t(m_restrtlen + 1) * n * sizeof(RT)) / 1.0e9
                       << " GB per rank)\n";
    }
}

template <typename V, typename M>
bool PackedGMRES<V,M>::converged (RT r0, RT r) const
{
    return (r < r0*m_rtol) || (r < m_atol);
}

template <typename V, typename M>
void PackedGMRES<V,M>::solve (V& a_sol, V const& a_rhs, RT a_tol_rel, RT a_tol_abs, int a_its)
{
    BL_PROFILE("PackedGMRES::solve()");

    AMREX_ALWAYS_ASSERT(m_linop != nullptr);

    auto t0 = amrex::second();

    if (m_v_tmp_rhs == nullptr) {
        m_v_tmp_rhs = std::make_unique<V>(m_linop->makeVecRHS());
    }
    if (m_v_tmp_lhs == nullptr) {
        m_v_tmp_lhs = std::make_unique<V>(m_linop->makeVecLHS());
    }
    if (!m_packed_allocated) {
        allocate_packed(a_rhs);
    }
    if (m_self_test && !m_self_test_done) {
        run_self_test(a_rhs);
        m_self_test_done = true;
    }

    m_rtol = a_tol_rel;
    m_atol = a_tol_abs;

    if (a_its < 0) { a_its = m_maxiter; }

    auto rnorm0 = RT(0);

    pack(a_rhs, m_b.data());
    copy_packed(basis(0), m_b.data());
    m_linop->setToZero(a_sol);

    m_its = 0;
    m_status = -1;
    cycle(a_sol, m_status, m_its, rnorm0);

    while (m_status == -1 && m_its < a_its) {
        compute_residual(a_sol);
        cycle(a_sol, m_status, m_its, rnorm0);
    }

    if (m_status == -1 && m_its >= a_its) { m_status = 1; }

    auto t1 = amrex::second();
    if (m_verbose > 0) {
        amrex::Print() << "GMRES: Solve Time = " << t1-t0 << '\n';
    }
}

template <typename V, typename M>
void PackedGMRES<V,M>::cycle (V& a_xx, int& a_status, int& a_itcount, RT& a_rnorm0)
{
    BL_PROFILE("PackedGMRES::cycle()");

    m_res = std::sqrt(norm2_squared(basis(0)));
    m_grs[0] = m_res;

    if (m_res == RT(0.0)) {
        a_status = 0;
        return;
    }

    scale(basis(0), RT(1.0)/m_res);

    if (a_itcount == 0) { a_rnorm0 = m_res; }

    a_status = converged(a_rnorm0,m_res) ? 0 : -1;

    int it = 0;
    while (it < m_restrtlen && a_itcount < m_maxiter)
    {
        if (m_verbose > 1) {
            amrex::Print() << "GMRES: iter = " << a_itcount
                           << ", residual = " << m_res << ", " << m_res/a_rnorm0
                           << " (rel.)\n";
        }

        if (a_status == 0) { break; }

        // w = A P^-1 v_it, through the operator class on solver vectors
        {
            BL_PROFILE("PackedGMRES::apply");
            unpack(basis(it), *m_v_tmp_rhs);
            m_linop->precond(*m_v_tmp_lhs, *m_v_tmp_rhs);
            m_linop->apply(*m_v_tmp_rhs, *m_v_tmp_lhs);
            pack(*m_v_tmp_rhs, basis(it+1));
        }

        gram_schmidt_orthogonalization(it);

        auto tt = std::sqrt(norm2_squared(basis(it+1)));

        auto const sml = RT((sizeof(RT) == 8) ? 1.e-99 : 1.e-30);
        bool happyend = (tt < sml);
        if (!happyend) {
            scale(basis(it+1), RT(1.0)/tt);
        }

        m_hh (it+1,it) = tt;
        m_hes(it+1,it) = tt;

        update_hessenberg(it, happyend, m_res);

        ++it;
        ++a_itcount;
        a_status = converged(a_rnorm0, m_res) ? 0 : -1;
        if (happyend) { break; }
    }

    if ((m_verbose > 1) && (a_status != 0 || a_itcount >= m_maxiter)) {
        amrex::Print() << "GMRES: iter = " << a_itcount
                       << ", residual = " << m_res << ", " << m_res/a_rnorm0
                       << " (rel.)\n";
    }

    build_solution(a_xx, it-1);
}

template <typename V, typename M>
void PackedGMRES<V,M>::gram_schmidt_orthogonalization (int const it)
{
    // Two passes of classical Gram-Schmidt (as AMReX's GMRES): each pass is
    // one GEMV for the it+1 inner products and one GEMV for the update.

    BL_PROFILE("PackedGMRES::GramSchmidt");

    RT* const w = basis(it+1);

    for (int j = 0; j <= it; ++j) {
        m_hh (j,it) = RT(0.0);
        m_hes(j,it) = RT(0.0);
    }

    for (int ncnt = 0; ncnt < 2 ; ++ncnt)
    {
        dot_many(basis(0), it+1, w, m_lhh.data());
        axpy_many(basis(0), it+1, m_lhh.data(), RT(-1.0), w);

        for (int j = 0; j <= it; ++j) {
            m_hh (j,it) += m_lhh[j];
            m_hes(j,it) -= m_lhh[j];
        }
    }
}

template <typename V, typename M>
void PackedGMRES<V,M>::update_hessenberg (int it, bool happyend, RT& res)
{
    BL_PROFILE("PackedGMRES::update_hessenberg()");

    for (int j = 1; j <= it; ++j) {
        auto tt = m_hh(j-1,it);
        m_hh(j-1,it) = m_cc[j-1] * tt         + m_ss[j-1] * m_hh(j,it);
        m_hh(j  ,it) = m_cc[j-1] * m_hh(j,it) - m_ss[j-1] * tt;
    }

    if (!happyend)
    {
        auto tt = std::sqrt(m_hh(it,it)*m_hh(it,it) + m_hh(it+1,it)*m_hh(it+1,it));
        m_cc[it] = m_hh(it  ,it) / tt;
        m_ss[it] = m_hh(it+1,it) / tt;
        m_grs[it+1] = - (m_ss[it] * m_grs[it]);
        m_grs[it  ] =    m_cc[it] * m_grs[it];
        m_hh(it,it) = m_cc[it] * m_hh(it,it) + m_ss[it] * m_hh(it+1,it);
        res = std::abs(m_grs[it+1]);
    }
    else
    {
        res = RT(0.0);
    }
}

template <typename V, typename M>
void PackedGMRES<V,M>::build_solution (V& a_xx, int const it)
{
    BL_PROFILE("PackedGMRES:build_solution()");

    if (it < 0) { return; }

    if (m_hh(it,it) != RT(0.0)) {
        m_grs[it] /= m_hh(it,it);
    } else {
        m_grs[it] = RT(0.0);
    }

    for (int ii = 1; ii <= it; ++ii) {
        int k = it - ii;
        auto tt = m_grs[k];
        for (int j = k+1; j <= it; ++j) {
            tt -= m_hh(k,j) * m_grs[j];
        }
        m_grs[k] = tt / m_hh(k,k);
    }

    // y = sum_i grs_i v_i (packed), then x += P^-1 y on solver vectors
    set_zero(m_tmp.data());
    axpy_many(basis(0), it+1, m_grs.data(), RT(1.0), m_tmp.data());
    unpack(m_tmp.data(), *m_v_tmp_rhs);

    m_linop->precond(*m_v_tmp_lhs, *m_v_tmp_rhs);
    m_linop->increment(a_xx, *m_v_tmp_lhs, RT(1.0));
}

template <typename V, typename M>
void PackedGMRES<V,M>::compute_residual (V const& a_xx)
{
    BL_PROFILE("PackedGMRES::compute_residual()");
    m_linop->assign(*m_v_tmp_lhs, a_xx);
    m_linop->apply(*m_v_tmp_rhs, *m_v_tmp_lhs);
    pack(*m_v_tmp_rhs, m_tmp.data());
    axpby(basis(0), RT(1.0), m_b.data(), RT(-1.0), m_tmp.data()); // r = b - A x
}

// ---------------------------------------------------------------------------
// packed kernels

template <typename V, typename M>
void PackedGMRES<V,M>::copy_packed (RT* a_dst, const RT* a_src)
{
    amrex::Gpu::copy(amrex::Gpu::deviceToDevice, a_src, a_src + m_n, a_dst);
}

template <typename V, typename M>
void PackedGMRES<V,M>::set_zero (RT* a_w)
{
    const int n = m_n;
    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) noexcept { a_w[i] = RT(0.0); });
}

template <typename V, typename M>
void PackedGMRES<V,M>::scale (RT* a_w, RT a_fac)
{
    const int n = m_n;
#if WARPX_PACKED_GMRES_HAS_BLAS
    if (m_use_blas) {
        blas_check(cublasSetStream(m_blas_handle, amrex::Gpu::gpuStream()), "cublasSetStream");
        if constexpr (std::is_same_v<RT,double>) {
            blas_check(cublasDscal(m_blas_handle, n, &a_fac, a_w, 1), "cublasDscal");
        } else {
            blas_check(cublasSscal(m_blas_handle, n, &a_fac, a_w, 1), "cublasSscal");
        }
        return;
    }
#endif
    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) noexcept { a_w[i] *= a_fac; });
}

template <typename V, typename M>
void PackedGMRES<V,M>::axpby (RT* a_out, RT a, const RT* a_x, RT b, const RT* a_y)
{
    const int n = m_n;
    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        a_out[i] = a * a_x[i] + b * a_y[i];
    });
}

template <typename V, typename M>
void PackedGMRES<V,M>::dot_many (const RT* a_basis, int a_ncols, const RT* a_w, RT* a_out)
{
    BL_PROFILE("PackedGMRES::dot_many");
    const int n = m_n;
    const auto nl = static_cast<amrex::Long>(n);
    amrex::ignore_unused(nl);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(a_ncols >= 1 && a_ncols <= m_restrtlen + 1,
                                     "PackedGMRES::dot_many: bad column count");
#if WARPX_PACKED_GMRES_HAS_BLAS
    if (m_use_blas) {
        blas_check(cublasSetStream(m_blas_handle, amrex::Gpu::gpuStream()), "cublasSetStream");
        const RT one = RT(1.0);
        const RT zero = RT(0.0);
        if constexpr (std::is_same_v<RT,double>) {
            blas_check(cublasDgemv(m_blas_handle, CUBLAS_OP_T, n, a_ncols, &one, a_basis, n,
                                   a_w, 1, &zero, m_dots_d.data(), 1), "cublasDgemv");
        } else {
            blas_check(cublasSgemv(m_blas_handle, CUBLAS_OP_T, n, a_ncols, &one, a_basis, n,
                                   a_w, 1, &zero, m_dots_d.data(), 1), "cublasSgemv");
        }
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_dots_d.begin(),
                         m_dots_d.begin() + a_ncols, a_out);
        amrex::ParallelAllReduce::Sum(a_out, a_ncols, amrex::ParallelContext::CommunicatorSub());
        return;
    }
#endif
#if defined(AMREX_USE_GPU)
#if defined(AMREX_USE_SYCL)
    WARPX_ABORT_WITH_MESSAGE("PackedGMRES: the native GEMV kernels are not implemented for SYCL");
#else
    // Deterministic two-stage reduction: every block writes one partial sum
    // per column (grid-stride loop, warp-shuffle block reduction), the host
    // sums the partials in a fixed order. Up to KC columns per launch.
    constexpr int KC = 8;
    constexpr int NT = 256;
    const int nblocks = m_nblocks;
    for (int j0 = 0; j0 < a_ncols; j0 += KC) {
        const int kc = std::min(KC, a_ncols - j0);
        RT* const part = m_partials_d.data() + std::size_t(j0) * std::size_t(nblocks);
        const RT* const bcol = a_basis + static_cast<std::size_t>(j0) * static_cast<std::size_t>(n);
        amrex::launch(nblocks, NT, amrex::Gpu::gpuStream(), [=] AMREX_GPU_DEVICE () noexcept
        {
            RT acc[KC];
            for (int k = 0; k < KC; ++k) { acc[k] = RT(0.0); }
            const int stride = static_cast<int>(blockDim.x * gridDim.x);
            for (int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x); i < n; i += stride) {
                const RT wi = a_w[i];
                for (int k = 0; k < KC; ++k) {
                    if (k < kc) { acc[k] += bcol[static_cast<amrex::Long>(k) * nl + i] * wi; }
                }
            }
            for (int k = 0; k < KC; ++k) {
                if (k < kc) {
                    const RT s = amrex::Gpu::blockReduceSum<RT>(acc[k]);
                    if (threadIdx.x == 0) {
                        part[static_cast<std::size_t>(k) * std::size_t(nblocks) + blockIdx.x] = s;
                    }
                }
            }
        });
    }
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_partials_d.begin(),
                     m_partials_d.begin() + std::size_t(a_ncols) * std::size_t(nblocks),
                     m_partials_h.begin());
    for (int j = 0; j < a_ncols; ++j) {
        RT s = RT(0.0);
        const RT* const pj = m_partials_h.data() + std::size_t(j) * std::size_t(nblocks);
        for (int b = 0; b < nblocks; ++b) { s += pj[b]; }
        a_out[j] = s;
    }
#endif
#else
    // host: static chunks per thread, partials summed in thread order
    int nthreads = 1;
#ifdef AMREX_USE_OMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<RT> part(std::size_t(nthreads) * std::size_t(a_ncols), RT(0.0));
#ifdef AMREX_USE_OMP
#pragma omp parallel num_threads(nthreads)
#endif
    {
        int tid = 0;
#ifdef AMREX_USE_OMP
        tid = omp_get_thread_num();
#endif
        const int chunk = (n + nthreads - 1) / nthreads;
        const int i0 = std::min(n, tid * chunk);
        const int i1 = std::min(n, i0 + chunk);
        for (int j = 0; j < a_ncols; ++j) {
            const RT* const bcol = a_basis + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
            RT s = RT(0.0);
            for (int i = i0; i < i1; ++i) { s += bcol[i] * a_w[i]; }
            part[std::size_t(tid) * std::size_t(a_ncols) + std::size_t(j)] = s;
        }
    }
    for (int j = 0; j < a_ncols; ++j) {
        RT s = RT(0.0);
        for (int t = 0; t < nthreads; ++t) { s += part[std::size_t(t) * std::size_t(a_ncols) + std::size_t(j)]; }
        a_out[j] = s;
    }
#endif
    amrex::ParallelAllReduce::Sum(a_out, a_ncols, amrex::ParallelContext::CommunicatorSub());
}

template <typename V, typename M>
void PackedGMRES<V,M>::axpy_many (const RT* a_basis, int a_ncols, const RT* a_coef, RT a_sign, RT* a_w)
{
    BL_PROFILE("PackedGMRES::axpy_many");
    const int n = m_n;
    const auto nl = static_cast<amrex::Long>(n);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(a_ncols >= 1 && a_ncols <= m_restrtlen + 1,
                                     "PackedGMRES::axpy_many: bad column count");
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, a_coef, a_coef + a_ncols, m_coef_d.begin());
    const RT* const coef = m_coef_d.data();
#if WARPX_PACKED_GMRES_HAS_BLAS
    if (m_use_blas) {
        blas_check(cublasSetStream(m_blas_handle, amrex::Gpu::gpuStream()), "cublasSetStream");
        const RT one = RT(1.0);
        if constexpr (std::is_same_v<RT,double>) {
            blas_check(cublasDgemv(m_blas_handle, CUBLAS_OP_N, n, a_ncols, &a_sign, a_basis, n,
                                   coef, 1, &one, a_w, 1), "cublasDgemv");
        } else {
            blas_check(cublasSgemv(m_blas_handle, CUBLAS_OP_N, n, a_ncols, &a_sign, a_basis, n,
                                   coef, 1, &one, a_w, 1), "cublasSgemv");
        }
        return;
    }
#endif
    // native: one pass over the basis; the update order over j matches the
    // sequence of Saxpys of the unpacked solver.
    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        RT acc = a_w[i];
        for (int j = 0; j < a_ncols; ++j) {
            acc += (a_sign * coef[j]) * a_basis[static_cast<amrex::Long>(j) * nl + i];
        }
        a_w[i] = acc;
    });
}

template <typename V, typename M>
typename PackedGMRES<V,M>::RT PackedGMRES<V,M>::norm2_squared (const RT* a_w)
{
    RT out = RT(0.0);
    dot_many(a_w, 1, a_w, &out);
    return out;
}

// ---------------------------------------------------------------------------
// self test: the packed operations against the solver-vector operations

template <typename V, typename M>
void PackedGMRES<V,M>::run_self_test (V const& a_model)
{
    BL_PROFILE("PackedGMRES::run_self_test()");
    using namespace amrex::literals;
    const int n = m_n;
    const auto nsz = std::size_t(n);
    amrex::Print() << "PackedGMRES self test: local packed length " << n << ", "
                   << (m_use_blas ? "blas" : "native") << " kernels\n";

    // deterministic pseudo-random packed data in [-1, 1) (per local index;
    // the comparisons are between two evaluations of the same data)
    auto fill = [] (std::vector<RT>& v, unsigned int seed)
    {
        for (std::size_t i = 0; i < v.size(); ++i) {
            unsigned int h = static_cast<unsigned int>(i) * 2654435761u + seed * 0x9E3779B9u;
            h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
            v[i] = RT(2.0) * (RT(h) / RT(4294967296.0)) - RT(1.0);
        }
    };
    std::vector<RT> px(nsz), py(nsz), q(nsz);
    fill(px, 11u);
    fill(py, 29u);

    V x, y, z;
    x.Define(a_model);
    y.Define(a_model);
    z.Define(a_model);

    // packed -> solver vector -> packed round trip (owned values only pass
    // through the mask; duplicates are synchronized by copyFrom)
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, px.begin(), px.end(), m_tmp.begin());
    unpack(m_tmp.data(), x);
    pack(x, basis(0));
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, basis(0), basis(0) + n, q.begin());
    RT max_rt = 0._rt, max_abs = 0._rt;
    for (std::size_t i = 0; i < nsz; ++i) {
        max_rt = std::max(max_rt, std::abs(q[i] - px[i]));
        max_abs = std::max(max_abs, std::abs(px[i]));
    }
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, py.begin(), py.end(), m_tmp.begin());
    unpack(m_tmp.data(), y);
    pack(y, basis(1));
    // from here on px/py are the packed images of x/y as the solver holds them
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, basis(0), basis(0) + n, px.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, basis(1), basis(1) + n, py.begin());
    amrex::ParallelAllReduce::Max(max_rt, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Max(max_abs, amrex::ParallelContext::CommunicatorSub());
    const RT round_trip = (max_abs > 0._rt) ? max_rt / max_abs : max_rt;

    // inner products: solver vector versus a plain host loop over the packed
    // images versus the packed kernel
    auto host_dot = [&] (const std::vector<RT>& a, const std::vector<RT>& b, RT& sum_abs) -> RT
    {
        RT s = 0._rt; sum_abs = 0._rt;
        for (std::size_t i = 0; i < nsz; ++i) { s += a[i]*b[i]; sum_abs += std::abs(a[i]*b[i]); }
        amrex::ParallelAllReduce::Sum(s, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(sum_abs, amrex::ParallelContext::CommunicatorSub());
        return s;
    };
    RT sabs_xy = 0._rt, sabs_xx = 0._rt;
    const RT dot_host = host_dot(px, py, sabs_xy);
    const RT dot_vec = x.dotProduct(y);
    RT dot_kernel = 0._rt;
    dot_many(basis(1), 1, basis(0), &dot_kernel);
    const RT dot_err_vec = std::abs(dot_vec - dot_host) / sabs_xy;
    const RT dot_err_kernel = std::abs(dot_kernel - dot_host) / sabs_xy;
    const RT nrm_host = std::sqrt(host_dot(px, px, sabs_xx));
    const RT nrm_vec = x.norm2();
    const RT nrm_kernel = std::sqrt(norm2_squared(basis(0)));
    const RT nrm_err_vec = std::abs(nrm_vec - nrm_host) / nrm_host;
    const RT nrm_err_kernel = std::abs(nrm_kernel - nrm_host) / nrm_host;

    // axpy and lincomb: solver-vector ops versus the packed images
    const RT a = 0.37_rt, b = -1.91_rt;
    z.Copy(x);
    z.increment(y, a);
    pack(z, m_tmp.data());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_tmp.begin(), m_tmp.end(), q.begin());
    RT axpy_err = 0._rt, axpy_scale = 0._rt;
    for (std::size_t i = 0; i < nsz; ++i) {
        const RT ref = px[i] + a*py[i];
        axpy_err = std::max(axpy_err, std::abs(q[i] - ref));
        axpy_scale = std::max(axpy_scale, std::abs(ref));
    }
    z.linComb(a, x, b, y);
    pack(z, m_tmp.data());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_tmp.begin(), m_tmp.end(), q.begin());
    RT lin_err = 0._rt, lin_scale = 0._rt;
    for (std::size_t i = 0; i < nsz; ++i) {
        const RT ref = a*px[i] + b*py[i];
        lin_err = std::max(lin_err, std::abs(q[i] - ref));
        lin_scale = std::max(lin_scale, std::abs(ref));
    }
    amrex::ParallelAllReduce::Max(axpy_err, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Max(axpy_scale, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Max(lin_err, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Max(lin_scale, amrex::ParallelContext::CommunicatorSub());
    axpy_err = (axpy_scale > 0._rt) ? axpy_err / axpy_scale : axpy_err;
    lin_err = (lin_scale > 0._rt) ? lin_err / lin_scale : lin_err;

    // the GEMV kernels on a small basis: columns c_j = px + j*py/(j+1)
    const int ncols = std::min(m_restrtlen + 1, 5);
    std::vector<std::vector<RT>> cols(ncols, std::vector<RT>(nsz));
    for (int j = 0; j < ncols; ++j) {
        for (std::size_t i = 0; i < nsz; ++i) { cols[j][i] = px[i] + RT(j) * py[i] / RT(j + 1); }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, cols[j].begin(), cols[j].end(), basis(j));
    }
    std::vector<RT> w(nsz);
    fill(w, 71u);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, w.begin(), w.end(), m_tmp.begin());
    std::vector<RT> dots(ncols), dots_ref(ncols), coef(ncols);
    dot_many(basis(0), ncols, m_tmp.data(), dots.data());
    RT gemv_dot_err = 0._rt;
    for (int j = 0; j < ncols; ++j) {
        RT sabs = 0._rt;
        dots_ref[j] = host_dot(cols[j], w, sabs);
        gemv_dot_err = std::max(gemv_dot_err, std::abs(dots[j] - dots_ref[j]) / sabs);
        coef[j] = RT(0.5) - RT(0.25) * RT(j);
    }
    axpy_many(basis(0), ncols, coef.data(), RT(-1.0), m_tmp.data());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, m_tmp.begin(), m_tmp.end(), q.begin());
    RT gemv_axpy_err = 0._rt, gemv_axpy_scale = 0._rt;
    for (std::size_t i = 0; i < nsz; ++i) {
        RT ref = w[i];
        for (int j = 0; j < ncols; ++j) { ref -= coef[j] * cols[j][i]; }
        gemv_axpy_err = std::max(gemv_axpy_err, std::abs(q[i] - ref));
        gemv_axpy_scale = std::max(gemv_axpy_scale, std::abs(ref));
    }
    amrex::ParallelAllReduce::Max(gemv_axpy_err, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Max(gemv_axpy_scale, amrex::ParallelContext::CommunicatorSub());
    gemv_axpy_err = (gemv_axpy_scale > 0._rt) ? gemv_axpy_err / gemv_axpy_scale : gemv_axpy_err;

    int failures = 0;
    auto report = [&] (const char* name, RT value, RT bound)
    {
        const bool pass = (value <= bound) && !std::isnan(value);
        amrex::Print() << "PackedGMRES self test: " << name << " = " << value
                       << " (bound " << bound << ") " << (pass ? "PASS" : "FAIL") << "\n";
        if (!pass) { ++failures; }
    };
    // elementwise operations agree to a few ulps; reductions to the
    // round-off of the differently ordered sums (relative to sum |x_i y_i|)
    const RT ulps = RT(8) * std::numeric_limits<RT>::epsilon();
    const RT reduce_bound = RT(1.0e-12);
    report("pack/unpack round trip rel. max error", round_trip, ulps);
    report("dotProduct(vec) vs packed host sum, rel. to sum|xy|", dot_err_vec, reduce_bound);
    report("dot kernel vs packed host sum, rel. to sum|xy|", dot_err_kernel, reduce_bound);
    report("norm2(vec) vs packed host norm, rel.", nrm_err_vec, reduce_bound);
    report("norm kernel vs packed host norm, rel.", nrm_err_kernel, reduce_bound);
    report("increment(vec) vs packed axpy, rel. max error", axpy_err, ulps);
    report("linComb(vec) vs packed lincomb, rel. max error", lin_err, ulps);
    report("dot_many kernel vs host, rel. to sum|xy| (max over columns)", gemv_dot_err, reduce_bound);
    report("axpy_many kernel vs host, rel. max error", gemv_axpy_err, RT(64) * ulps);
    amrex::Print() << "PackedGMRES self test: dot vec/host/kernel = " << dot_vec << " " << dot_host
                   << " " << dot_kernel << "; norm vec/host/kernel = " << nrm_vec << " "
                   << nrm_host << " " << nrm_kernel << "\n";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(failures == 0,
        "PackedGMRES self test FAILED (" + std::to_string(failures) + " checks)");
    amrex::Print() << "PackedGMRES self test: all checks PASS\n";
}

#endif
