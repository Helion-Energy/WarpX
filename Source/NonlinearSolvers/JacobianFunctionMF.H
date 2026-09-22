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
#include "MatrixPC.H"
#include "JacobiPC.H"
#include "HallJacobiPC.H"
#include "HybridPICPC.H"
#ifdef WARPX_DIM_RZ
#include "HybridPICRZPC.H"
#endif
#include "BlockBandedPC.H"
#include "Utils/TextMsg.H"
#include <AMReX_Config.H>
#include <AMReX_ParmParse.H>
#include <algorithm>
#include <cmath>
#include <string>

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

    /** Apply the matrix-free operator with an explicit relative JFNK scale.
     *  The saved full-residual base is unchanged, so this is the exact B_g
     *  action seen by GMRES, evaluated at a diagnostic epsilon. */
    void applyWithJFNKEps (T& a_dF, const T& a_dU, RT a_relative_eps);

    /** Return the absolute perturbation used by applyWithJFNKEps. */
    [[nodiscard]] RT perturbationScale (const T& a_dU, RT a_relative_eps);

    /** Diagnostics-only frozen preconditioner operator C*v (not C^{-1}*v). */
    bool applyFrozenPreconditionerOperator (T& a_out, const T& a_in)
    {
        return m_usePreCond && m_preCond->ApplyFrozenOperator(a_out, a_in);
    }

    inline
    void precond ( T& a_U, const T& a_X ) override
    {
        if (m_usePreCond) {
            a_U.zero();
            m_preCond->Apply(a_U, a_X);
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
    }

    inline
    void setBaseRHS ( const T&  a_R )
    {
        m_R0.Copy(a_R);
        if (m_rhs_scale_floor) { m_normR0 = this->norm2(m_R0); }
    }

    inline
    void setJFNKEps ( RT a_eps )
    {
        m_epsJFNK = a_eps;
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
    RT m_epsJFNK = RT(1.0e-6);
    RT m_normY0;
    bool m_rhs_scale_floor = false;
    RT m_normR0 = RT(0.0);
    RT m_cur_time, m_dt;

    PreconditionerType m_pc_type = PreconditionerType::none;

    T m_Z, m_Y0, m_R0, m_R;
    Ops* m_ops = nullptr;
    std::unique_ptr<Preconditioner<T,Ops>> m_preCond = nullptr;

    [[nodiscard]] RT perturbationScaleFromNorm (RT a_norm_dU,
                                                RT a_relative_eps);
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

    m_ops = a_ops;
    amrex::ParmParse("jacobian").query("rhs_scale_floor", m_rhs_scale_floor);

    m_usePreCond = (a_pc_type != PreconditionerType::none);
    if (m_usePreCond) {
        m_pc_type = a_pc_type;
        if (m_pc_type == PreconditionerType::pc_curl_curl_mlmg) {
            m_preCond = std::make_unique<CurlCurlMLMGPC<T,Ops>>();
        } else if (m_pc_type == PreconditionerType::pc_jacobi) {
            m_preCond = std::make_unique<JacobiPC<T,Ops>>();
        } else if (m_pc_type == PreconditionerType::pc_hybrid_pic) {
#ifdef WARPX_DIM_RZ
            if (WarpX::grid_type == ablastr::utils::enums::GridType::Staggered) {
                m_preCond = std::make_unique<HybridPICRZPC<T,Ops>>();
            } else
#endif
            {
                m_preCond = std::make_unique<HybridPICPC<T,Ops>>();
            }
        } else if (m_pc_type == PreconditionerType::pc_hall_jacobi) {
            m_preCond = std::make_unique<HallJacobiPC<T,Ops>>();
        } else if (m_pc_type == PreconditionerType::pc_block_banded) {
            m_preCond = std::make_unique<BlockBandedPC<T,Ops>>();
        } else if (m_pc_type == PreconditionerType::pc_curlcurl_banded) {
            // the form-aware BlockBandedPC: assembles the Jacobian of the
            // configured hybrid_pic_model.esolve form (e_form rows are the
            // pc_block_banded assembly; curlcurl_form composes the inner
            // elliptic operator with the numerator response)
            auto pc = std::make_unique<BlockBandedPC<T,Ops>>();
            pc->SetFormAware(true);
            m_preCond = std::move(pc);
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
    applyWithJFNKEps(a_dF, a_dU, m_epsJFNK);
}

template <class T, class Ops>
auto JacobianFunctionMF<T,Ops>::perturbationScale (
    const T& a_dU, RT a_relative_eps) -> RT
{
    const RT normY = this->norm2(a_dU);
    return perturbationScaleFromNorm(normY, a_relative_eps);
}

template <class T, class Ops>
auto JacobianFunctionMF<T,Ops>::perturbationScaleFromNorm (
    RT normY, RT a_relative_eps) -> RT
{
    if (normY < RT(1.0e-15)) { return RT(0.0); }
    if (m_is_linear) { return RT(1.0); }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(a_relative_eps) && a_relative_eps > RT(0.0),
        "JacobianFunctionMF diagnostic JFNK epsilon must be positive and finite");
    if (m_rhs_scale_floor) {
        // R has the same units as Y in F(Y)=Y-b-R(Y). Extend the legacy
        // EXACT-zero fallback continuously to near-zero base states: a tiny
        // projection remainder must not collapse a Jv probe below roundoff.
        RT const scale = std::max(m_normY0, m_normR0);
        if (scale > RT(0.0)) { return a_relative_eps * scale / normY; }
    }
    if (m_normY0 == RT(0.0)) {
        return a_relative_eps * this->norm2(m_R0) / normY;
    }
    return a_relative_eps * m_normY0 / normY;
}

template <class T, class Ops>
void JacobianFunctionMF<T,Ops>::applyWithJFNKEps (
    T& a_dF, const T& a_dU, RT a_relative_eps)
{
    BL_PROFILE("JacobianFunctionMF::apply()");
    using namespace amrex::literals;
    RT const normY = this->norm2(a_dU); // always 1 when called from GMRES

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        isDefined(),
        "JacobianFunction::apply() called on undefined JacobianFunction");

    if (normY < 1.0e-15) { a_dF.zero(); }
    else {

        /* eps = error_rel * ||Y0|| / ||dU||.  The more common
         * error_rel*sqrt(1+||Y0||) form is not recommended for poorly scaled
         * Y0.  The zero-base fallback uses ||R0||. */
        // Reuse the norm already computed above: the production apply path
        // must not acquire a second global reduction when D1 is disabled.
        const RT eps = perturbationScaleFromNorm(normY, a_relative_eps);
        const RT eps_inv = 1.0_rt/eps;

        m_Z.linComb( 1.0, m_Y0, eps, a_dU ); // Z = Y0 + eps*dU
        m_ops->ComputeRHS(m_R, m_Z, m_cur_time, -1, true );

        // F(Y) = Y - b - R(Y) ==> dF = dF/dY*dU = [1 - dR/dY]*dU
        //                            = dU - (R(Z)-R(Y0))/eps
        a_dF.linComb( 1.0, a_dU, eps_inv, m_R0 );
        a_dF.increment(m_R,-eps_inv);

    }

}

#endif
