/* Copyright 2024 Justin Angus
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef JacobianFunctionMF_H_
#define JacobianFunctionMF_H_

#include "LinearFunction.H"
#include "CurlCurlMLMGPC.H"
#include "MHDBlockPC.H"
#include "MatrixPC.H"
#include "JacobiPC.H"
#include "Utils/TextMsg.H"
#include <AMReX_Config.H>
#include <AMReX_Enum.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

/**
 * \brief Sizing rule of the matrix-free Jacobian probe (newton.jfnk_epsilon_mode).
 *
 * global:    eps = jfnk_epsilon ||U|| / ||v|| with the solver norm (the per-block
 *            reference scales): one relative perturbation of the GLOBAL state along
 *            the Krylov direction v (Pernice-Walker). The historical behaviour.
 * component: eps = jfnk_epsilon ||D^{-1} U|| / ||D^{-1} v|| with the diagonal
 *            component scale D = diag(|U_i| + floor_c), floor_c a per-block
 *            additive floor: ONE scalar perturbation per Jacobian application,
 *            sized in the component-scaled ("typical value") variables D^{-1} U
 *            (the scaling of the unknowns of Dennis & Schnabel, Numerical
 *            Methods for Unconstrained Optimization and Nonlinear Equations,
 *            1983, ch. 7), so that every component is perturbed by jfnk_epsilon
 *            of its OWN magnitude in the root-mean-square sense, whatever the
 *            block scales and however the state varies within a block. It is
 *            NOT a per-component epsilon: the direction stays the Krylov
 *            direction. The scalar-epsilon sizing rules themselves are those of
 *            Knoll & Keyes, JCP 193 (2004) 357, sec. 2.3.1, eqs. (11)-(14). See
 *            JacobianFunctionMF::apply for the column-scaling derivation, which
 *            rests on the right-preconditioned finite-difference product of the
 *            same paper, eqs. (25)-(28).
 *
 * The component scale D is rebuilt from the base state at every
 * setBaseSolution() (once per Newton iteration); nothing in the probe itself
 * can detect a stale D, so the optional probe report (setProbeReportFile)
 * recomputes |U| + floor from the base state and asserts it equals D.
 */
AMREX_ENUM(JFNKEpsilonMode, global, component);

/**
 * \brief This is a linear function class for computing the action of a
 *  Jacobian on a vector using a matrix-free finite-difference method.
 *  This class has all of the required functions to be used as the
 *  linear operator template parameter in AMReX_GMRES.
 */

template <class T, class Ops>
class JacobianFunctionMF : public LinearFunction<T,Ops>
{
    public:

    using RT = typename T::value_type;

    JacobianFunctionMF() = default;
    ~JacobianFunctionMF() override = default;

    // Default move and copy operations
    JacobianFunctionMF(const JacobianFunctionMF&) = default;
    JacobianFunctionMF& operator=(const JacobianFunctionMF&) = default;
    JacobianFunctionMF(JacobianFunctionMF&&) noexcept = default;
    JacobianFunctionMF& operator=(JacobianFunctionMF&&) noexcept = default;

    void apply ( T& a_dF, const T& a_dU ) override;

    inline
    void precond ( T& a_U, const T& a_X ) override
    {
        if (m_usePreCond) {
            if (m_active_set_masking) {
                // Reduced-space preconditioner P_F M^{-1} P_F + P_A: the
                // block solves see a zero right-hand side on the pinned
                // components and their output there is replaced by the
                // input (identity on the active set, consistent with the
                // masked operator in apply()).
                m_X_masked.Copy(a_X);
                m_ops->ZeroActiveSetComponents(m_X_masked);
                a_U.zero();
                m_preCond->Apply(a_U, m_X_masked);
                m_ops->CopyActiveSetComponents(a_U, a_X);
            } else {
                a_U.zero();
                m_preCond->Apply(a_U, a_X);
            }
        } else {
            a_U.Copy(a_X);
        }
    }

    inline
    void updatePreCondMat ( const T&  a_X ) override
    {
        if (m_usePreCond) { m_preCond->Update(a_X); }
    }

    inline
    void getPCMatrix( amrex::Gpu::DeviceVector<int>& a_ridx_g,
                      amrex::Gpu::DeviceVector<int>& a_nnz,
                      amrex::Gpu::DeviceVector<int>& a_cidx_g,
                      amrex::Gpu::DeviceVector<RT>&  a_aij,
                      int& a_n, int& a_ncols_max ) override
    {
        AMREX_ALWAYS_ASSERT(m_usePreCond);
        m_preCond->getPCMatrix(a_ridx_g, a_nnz, a_cidx_g, a_aij, a_n, a_ncols_max);
    }


    T makeVecLHS () const override;
    T makeVecRHS () const override;

    [[nodiscard]] inline
    bool isDefined() const { return m_is_defined;  }

    [[nodiscard]] inline
    bool usePreconditioner() const { return m_usePreCond;  }

    inline
    void setBaseSolution ( const T&  a_U )
    {
        m_Y0.Copy(a_U);
        m_normY0 = this->norm2(m_Y0);
        m_probe_report_pending = !m_probe_report_file.empty();
        if (m_eps_mode == JFNKEpsilonMode::component || m_probe_report_pending) {
            // Component probe scale of this Newton iteration,
            // D = |U| + floor (per block), and the scaled base-state norm
            // ||D^{-1} U|| (the count of components above their floor, in
            // effect). Both are frozen for the whole linear solve so the
            // probe operator stays linear in the Krylov vector.
            m_D.Copy(a_U);
            m_D.abs();
            m_D.addBlockConstants(m_component_floors);
            m_W.Copy(a_U);
            m_W.divideBy(m_D);
            m_normY0_scaled = std::sqrt(m_W.dotProduct(m_W, false));
            // A non-finite scaled norm (a NaN/inf in the state, or a zero
            // component scale) must not silently demote the component mode
            // to the global branch in apply(): abort with a message.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(m_normY0_scaled),
                "JacobianFunctionMF: the component probe scale ||D^{-1} U|| is "
                "not finite (non-finite state or zero component scale)");
            if (m_eps_mode == JFNKEpsilonMode::component &&
                m_normY0_scaled == 0.0 && !m_zero_scale_warned) {
                m_zero_scale_warned = true;
                amrex::Print() << "WARNING: JacobianFunctionMF: the base state "
                                  "is identically zero; the component probe "
                                  "falls back to the global sizing rule for "
                                  "this solve (reported once)\n";
            }
        }
    }

    inline
    void setBaseRHS ( const T&  a_R )
    {
        m_R0.Copy(a_R);
    }

    inline
    void setJFNKEps ( RT a_eps )
    {
        m_epsJFNK = a_eps;
    }

    /**
     * \brief Reduced-space (active-set) Newton mode: apply() computes
     * (P_F J P_F + P_A) dU -- the Jacobian-vector product with the
     * operator's active-set components unperturbed (pinned columns) and
     * the result on them replaced by the input (pinned rows = identity)
     * -- and precond() the correspondingly masked preconditioner. The
     * masks are the operator's (ImplicitSolver::Zero/CopyActiveSetComponents);
     * with an empty active set both reduce to the unmasked operations.
     */
    inline
    void setActiveSetMasking ( bool a_on )
    {
        m_active_set_masking = a_on;
    }

    /**
     * \brief Select the probe sizing rule; a_component_floors are the
     * per-block additive floors of the component scale, in block units,
     * ordered as T::blockNames() (used by the component mode only).
     * Must be called after define().
     */
    inline
    void setJFNKEpsMode ( JFNKEpsilonMode a_mode,
                          const std::vector<RT>& a_component_floors )
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isDefined(),
            "JacobianFunctionMF::setJFNKEpsMode() called before define()");
        m_eps_mode = a_mode;
        m_component_floors = a_component_floors;
        if (m_eps_mode == JFNKEpsilonMode::component) {
            m_D.Define(m_Y0);
            m_W.Define(m_Y0);
        }
    }

    [[nodiscard]] inline
    JFNKEpsilonMode jfnkEpsMode () const { return m_eps_mode; }

    /**
     * \brief Diagnostic: append, once per Newton iteration (the first
     * Jacobian application of each linear solve), the statistics of the
     * effective relative perturbation eps |v_i| / (|U_i| + floor_c) of the
     * probe to a_file, split into small components (|U_i| below 1e-2 of
     * their block's reference scale) and large ones. Empty = off.
     * Must be called after define().
     */
    inline
    void setProbeReportFile ( const std::string& a_file )
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            isDefined(),
            "JacobianFunctionMF::setProbeReportFile() called before define()");
        m_probe_report_file = a_file;
        if (!m_probe_report_file.empty()) {
            if (!m_D.IsDefined()) { m_D.Define(m_Y0); }
            if (!m_W.IsDefined()) { m_W.Define(m_Y0); }
        }
    }

    inline
    void setIsLinear ( bool a_isLinear )
    {
        m_is_linear = a_isLinear;
    }

    inline
    void curTime ( RT a_time )
    {
        m_cur_time = a_time;
        if (m_usePreCond) { m_preCond->CurTime(a_time); }
    }

    inline
    void curTimeStep ( RT a_dt )
    {
        m_dt = a_dt;
        if (m_usePreCond) { m_preCond->CurTimeStep(a_dt); }
    }

    inline
    void printParams () const
    {
        if (m_pc_type != PreconditionerType::none) {
            m_preCond->printParameters();
        }
    }

    void define( const T&, Ops*, const PreconditionerType& ) override;

    [[nodiscard]] inline
    PreconditionerType pcType () const override { return m_pc_type; }

    private:

    bool m_is_defined = false;
    bool m_is_linear = false;
    bool m_usePreCond = false;
    bool m_active_set_masking = false;
    RT m_epsJFNK = RT(1.0e-6);
    RT m_normY0;
    RT m_cur_time, m_dt;

    JFNKEpsilonMode m_eps_mode = JFNKEpsilonMode::global;
    std::vector<RT> m_component_floors;
    RT m_normY0_scaled = RT(0.0);
    std::string m_probe_report_file;
    bool m_probe_report_pending = false;
    bool m_probe_report_header_written = false;
    bool m_zero_scale_warned = false;

    void reportProbe ( RT a_eps, const T& a_dU );
    // component mode work vectors: the probe scale D of the current
    // Newton iteration and a scratch vector for D^{-1} x
    T m_D, m_W;

    PreconditionerType m_pc_type = PreconditionerType::none;

    T m_Z, m_Y0, m_R0, m_R;
    // Scratch for the masked preconditioner input (active-set mode).
    T m_X_masked;
    Ops* m_ops = nullptr;
    std::unique_ptr<Preconditioner<T,Ops>> m_preCond = nullptr;
};

template <class T, class Ops>
void JacobianFunctionMF<T,Ops>::define ( const T& a_U,
                                         Ops* a_ops,
                                         const PreconditionerType& a_pc_type )
{
    BL_PROFILE("JacobianFunctionMF::::define()");
    m_Z.Define(a_U);
    m_Y0.Define(a_U);
    m_R0.Define(a_U);
    m_R.Define(a_U);
    m_X_masked.Define(a_U);

    m_ops = a_ops;

    m_usePreCond = (a_pc_type != PreconditionerType::none);
    if (m_usePreCond) {
        m_pc_type = a_pc_type;
        if (m_pc_type == PreconditionerType::pc_curl_curl_mlmg) {
            m_preCond = std::make_unique<CurlCurlMLMGPC<T,Ops>>();
        } else if (m_pc_type == PreconditionerType::pc_jacobi) {
            m_preCond = std::make_unique<JacobiPC<T,Ops>>();
        } else if (m_pc_type == PreconditionerType::pc_mhd_block) {
            m_preCond = std::make_unique<MHDBlockPC<T,Ops>>();
        } else {
            m_preCond = std::make_unique<MatrixPC<T,Ops>>();
            m_preCond->setName(amrex::getEnumNameString(m_pc_type));
        }
        m_preCond->Define(a_U, a_ops);
    }

    m_is_defined = true;
}

template <class T, class Ops>
auto JacobianFunctionMF<T,Ops>::makeVecRHS () const -> T
{
    BL_PROFILE("JacobianFunctionMF::::makeVecRHS()");
    T x;
    x.Define(m_R);
    return x;
}

template <class T, class Ops>
auto JacobianFunctionMF<T,Ops>::makeVecLHS () const -> T
{
    BL_PROFILE("JacobianFunctionMF::::makeVecLHS()");
    T x;
    x.Define(m_R);
    return x;
}

template <class T, class Ops>
void JacobianFunctionMF<T,Ops>::apply (T& a_dF, const T& a_dU)
{
    BL_PROFILE("JacobianFunctionMF::apply()");
    using namespace amrex::literals;
    RT const normY = this->norm2(a_dU); // always 1 when called from GMRES

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        isDefined(),
        "JacobianFunction::apply() called on undefined JacobianFunction");

    if (normY < 1.0e-15) { a_dF.zero(); }
    else {

        RT eps;
        if (m_is_linear) {
            eps = 1.0_rt;
        } else if (m_eps_mode == JFNKEpsilonMode::component &&
                   m_normY0_scaled > 0.0) {
            /* Component-scaled probe: a scalar perturbation sized in the
             * component-scaled (typical-value) variables, Dennis & Schnabel
             * 1983 ch. 7, so that the perturbation is relative to each
             * component's own magnitude instead of the global state norm.
             * The scalar-epsilon sizing follows Knoll & Keyes, JCP 193 (2004)
             * 357, sec. 2.3.1 (eqs. 11-14); the realization below is their
             * right-preconditioned finite-difference product, eqs. (25)-(28).
             *
             * Realized as a right (column) scaling of the Jacobian. With
             * D = diag(|U_i| + floor_c) the linear system J dU = F is solved
             * for the scaled unknown w = D^{-1} dU, i.e. (J D) w = F, and
             * one residual evaluation gives the operator's action,
             *   (J D) w ~ [F(U + eps D w) - F(U)] / eps,
             * with eps sized in the scaled variables (Pernice-Walker on
             * D^{-1} U): eps = jfnk_epsilon ||D^{-1} U|| / ||w||. AMReX GMRES
             * is right-preconditioned; the preconditioner of the scaled
             * system is D^{-1} M^{-1}, so the preconditioned operator
             * (J D)(D^{-1} M^{-1}) = J M^{-1} is the one of the unscaled
             * system: the Krylov space, the iterates dU = D w and the GMRES
             * residual norms are unchanged (exact arithmetic), and only the
             * probe direction's SIZE differs. The scaling therefore never
             * leaves this routine: GMRES hands in v = M^{-1} q = D w as
             * before, w = D^{-1} v is formed here, and the state is perturbed
             * along v by eps = jfnk_epsilon ||D^{-1} U|| / ||D^{-1} v||,
             * whose relative perturbation eps v_i / (|U_i| + floor_c) has
             * root-mean-square jfnk_epsilon over the components.
             */
            m_W.Copy(a_dU);
            m_W.divideBy(m_D);
            const RT normW = std::sqrt(m_W.dotProduct(m_W, false));
            eps = m_epsJFNK * m_normY0_scaled / normW;
        } else {
            /* eps = error_rel * sqrt(1 + ||Y0||) / ||dU||
             * M. Pernice and H. F. Walker, "NITSOL: A Newton Iterative Solver for
             * Nonlinear Systems", SIAM J. Sci. Stat. Comput., 1998, vol 19,
             * pp. 302--318. */
            if (m_normY0==0.0) { eps = m_epsJFNK * this->norm2(m_R0) / normY; }
            else {
                // m_epsJFNK * sqrt(1.0 + m_normY0) / normY
                // above commonly used form not recommend for poorly scaled Y0
                eps = m_epsJFNK * m_normY0 / normY;
            }
        }
        // Operators whose residual evaluation clamps every floored
        // quantity pointwise can evaluate transiently inadmissible probe
        // states; skipping the limiter for them keeps the probe exact at
        // states resting on their admissibility bounds (where neither
        // perturbation sign would otherwise stay admissible and the
        // assert below fires).
        if (m_ops->ProbePerturbationsNeedLimiting()) {
            const RT requested_eps = eps;
            eps = m_ops->LimitSolverStep(m_Y0, a_dU, requested_eps);
            if (std::abs(eps) < std::abs(requested_eps)) {
                const RT negative_eps =
                    m_ops->LimitSolverStep(m_Y0, a_dU, -requested_eps);
                if (std::abs(negative_eps) > std::abs(eps)) {
                    eps = negative_eps;
                }
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(eps) > std::numeric_limits<RT>::epsilon(),
                "The matrix-free Jacobian perturbation cannot remain in the "
                "operator's admissible state domain");
        }
        if (m_probe_report_pending) {
            m_probe_report_pending = false;
            reportProbe(eps, a_dU);
        }
        const RT eps_inv = 1.0_rt/eps;

        m_Z.linComb( 1.0, m_Y0, eps, a_dU ); // Z = Y0 + eps*dU
        if (m_active_set_masking) {
            // Pinned columns: the probe state keeps the base values on
            // the active set, so R(Z) - R(Y0) responds to the FREE
            // perturbation only.
            m_ops->CopyActiveSetComponents(m_Z, m_Y0);
        }
        m_ops->ComputeRHS(m_R, m_Z, m_cur_time, -1, true );

        // F(Y) = Y - b - R(Y) ==> dF = dF/dY*dU = [1 - dR/dY]*dU
        //                            = dU - (R(Z)-R(Y0))/eps
        a_dF.linComb( 1.0, a_dU, eps_inv, m_R0 );
        a_dF.increment(m_R,-eps_inv);
        if (m_active_set_masking) {
            // Pinned rows: identity on the active set (replaces the
            // coupling J_AF dU_F the free perturbation induced there).
            m_ops->CopyActiveSetComponents(a_dF, a_dU);
        }

    }

}

template <class T, class Ops>
void JacobianFunctionMF<T,Ops>::reportProbe ( RT a_eps, const T& a_dU )
{
    BL_PROFILE("JacobianFunctionMF::reportProbe()");
    // Staleness guard: the component scale must be the one of the current
    // base state. Rebuild |U| + floor into the scratch vector (the same
    // operations that built m_D, so the two agree bit for bit when D is
    // fresh) and require an exactly zero difference.
    m_W.Copy(m_Y0);
    m_W.abs();
    m_W.addBlockConstants(m_component_floors);
    m_W.increment(m_D, RT(-1.0));
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_W.dotProduct(m_W, false) == RT(0.0),
        "JacobianFunctionMF::reportProbe(): the component probe scale D is "
        "stale (not rebuilt from the current base state)");
    // Serialize the base state, the probe direction and the component scale
    // (all in block-reference units: copyTo divides by the block scales, so
    // D_i / s_c = |U_i| / s_c + f_c and the ratios below are unchanged).
    const amrex::Long n = m_Y0.nDOF_local();
    amrex::Gpu::DeviceVector<RT> d_u(n), d_v(n), d_d(n);
    m_Y0.copyTo(d_u.data());
    a_dU.copyTo(d_v.data());
    m_D.copyTo(d_d.data());
    amrex::Gpu::streamSynchronize();
    std::vector<RT> u(n), v(n), d(n);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_u.begin(), d_u.end(), u.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_v.begin(), d_v.end(), v.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_d.begin(), d_d.end(), d.begin());

    // active components: the direction is nonzero there (1e-14 of its max)
    RT vmax = 0.0;
    for (amrex::Long i = 0; i < n; ++i) { vmax = std::max(vmax, std::abs(v[i])); }
    amrex::ParallelAllReduce::Max(vmax, amrex::ParallelContext::CommunicatorSub());
    const RT v_cut = 1.0e-14 * vmax;
    constexpr RT small_cut = 1.0e-2; // |U_i| below 1e-2 of the block reference scale
    // [0] n_active [1] sumsq_all [2] n_small [3] sumsq_small [4] n_large [5] sumsq_large
    RT sums[6] = {0., 0., 0., 0., 0., 0.};
    RT maxes[3] = {0., 0., 0.}; // all, small, large
    for (amrex::Long i = 0; i < n; ++i) {
        if (std::abs(v[i]) <= v_cut) { continue; }
        const RT rel = std::abs(a_eps * v[i] / d[i]);
        sums[0] += 1.0; sums[1] += rel * rel; maxes[0] = std::max(maxes[0], rel);
        if (std::abs(u[i]) < small_cut) {
            sums[2] += 1.0; sums[3] += rel * rel; maxes[1] = std::max(maxes[1], rel);
        } else {
            sums[4] += 1.0; sums[5] += rel * rel; maxes[2] = std::max(maxes[2], rel);
        }
    }
    amrex::ParallelAllReduce::Sum(sums, 6, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Max(maxes, 3, amrex::ParallelContext::CommunicatorSub());

    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }
    auto rms = [](RT sumsq, RT count) { return count > 0. ? std::sqrt(sumsq / count) : 0.; };
    if (!m_probe_report_header_written) {
        m_probe_report_header_written = true;
        if (!amrex::FileExists(m_probe_report_file)) {
            std::filesystem::path const path(m_probe_report_file);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream header{m_probe_report_file, std::ofstream::out | std::ofstream::trunc};
            header << "#[0]time(s) [1]eps [2]n_active [3]rms_all [4]max_all"
                   << " [5]n_small [6]rms_small [7]max_small"
                   << " [8]n_large [9]rms_large [10]max_large"
                   << "   (relative perturbation eps|v_i|/(|U_i|+floor); small: |U_i| < 1e-2 block scale)\n";
        }
    }
    std::ofstream out{m_probe_report_file, std::ofstream::out | std::ofstream::app};
    out << std::setprecision(8) << std::scientific
        << m_cur_time << " " << a_eps << " "
        << static_cast<amrex::Long>(sums[0]) << " " << rms(sums[1], sums[0]) << " " << maxes[0] << " "
        << static_cast<amrex::Long>(sums[2]) << " " << rms(sums[3], sums[2]) << " " << maxes[1] << " "
        << static_cast<amrex::Long>(sums[4]) << " " << rms(sums[5], sums[4]) << " " << maxes[2] << "\n";
}

#endif
