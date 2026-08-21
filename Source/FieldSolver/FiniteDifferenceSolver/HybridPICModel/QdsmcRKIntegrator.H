/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMCRKINTEGRATOR_H_
#define WARPX_QDSMCRKINTEGRATOR_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <functional>
#include <utility>

/** Diagnostics returned by QdsmcRKIntegrator::Advance. t_done < t_end
 *  means the attempts budget ran out -- the caller decides warn/abort. */
struct QdsmcRKStats
{
    int n_attempts = 0;
    int n_accepted = 0;
    amrex::Real t_done = 0.0;
    amrex::Real dt_last = 0.0;
};

/** Adaptive explicit Runge--Kutta stepper over a single-component
 *  MultiFab state, factored from the hybrid solver's BfieldEvolve
 *  RKF45 controller so the field push and the QDSMC FD conduction
 *  subcycles can share one integration scheme (the B push migration is
 *  a follow-up; this class is consumer-agnostic).
 *
 *  The RHS is a host functor rhs(y, k) that owns its kernel launches
 *  and refreshes y's ghosts before evaluating; k is written on the
 *  valid region only. An optional stability-ceiling functor cap() is
 *  consulted after each fresh RHS evaluation (any state the ceiling
 *  needs -- e.g. a coefficient field -- is a side effect of rhs) and is
 *  always honored on top of the error controller: explicit RK on a
 *  diffusive spectrum is stability-limited, and the embedded error
 *  alone detects the instability edge only after noise has grown.
 *
 *  Schemes:
 *   - SSPRK2: Heun with the forward-Euler predictor as the embedded
 *     order-1 error estimate. Convex combination of Euler stages =
 *     monotone at the stability CFL -> preserves the discrete maximum
 *     principle of limited flux-form operators (the property the FD
 *     conduction operator exists for).
 *   - RKF45: the classic Fehlberg 4(5) pair, coefficient-identical to
 *     BfieldEvolveRKF45. NOT SSP (negative tableau weights): no
 *     monotonicity guarantee -- gate against SSPRK2 before making it a
 *     production default on monotone operators.
 *
 *  Controller (matches BfieldEvolve): accept when the weighted max-norm
 *  error ratio err = max |y_hi - y_lo| / (atol + rtol |y_old|) <= 1;
 *  dt *= clamp(safety * err^(-1/(p+1))) with growth factor max_growth
 *  on accept and a floor of 0.1 on reject. */
class QdsmcRKIntegrator
{
public:
    enum class Scheme : int { SSPRK2 = 0, RKF45 = 1 };

    using Rhs = std::function<void (amrex::MultiFab&, amrex::MultiFab&)>;
    using Cap = std::function<amrex::Real ()>;
    /** Optional hook run on the state after every ACCEPTED step (before
     *  the accepted state is latched for the next attempt), with the
     *  accepted dt. Used for constraint enforcement that must track the
     *  subcycle cadence -- e.g. Dirichlet bath pins, whose once-per-call
     *  application is a contact resistance that GROWS under refinement
     *  (bath-row heat capacity ~ dx; measured anti-convergent wall flux
     *  0.986 -> 0.967 from N=64 -> 128 on the conduction slab). */
    using PostStep = std::function<void (amrex::MultiFab&, amrex::Real)>;

    QdsmcRKIntegrator (Scheme scheme, Rhs rhs, Cap cap,
                       amrex::Real rtol, amrex::Real atol,
                       amrex::Real safety, amrex::Real max_growth,
                       int max_attempts, PostStep post_step = {})
        : m_scheme(scheme), m_rhs(std::move(rhs)), m_cap(std::move(cap)),
          m_post_step(std::move(post_step)),
          m_rtol(rtol), m_atol(atol), m_safety(safety),
          m_max_growth(max_growth), m_max_attempts(max_attempts)
    {}

    /** Advance y in place from t = 0 to t_end. y's ghost refresh is the
     *  RHS functor's job; on return y's ghosts are stale. */
    QdsmcRKStats Advance (amrex::MultiFab & y, amrex::Real const t_end) const
    {
        using namespace amrex::literals;
        QdsmcRKStats st;
        if (t_end <= 0.0_rt) { return st; }

        int const nk = (m_scheme == Scheme::RKF45) ? 6 : 2;
        int const nc = y.nComp();
        amrex::MultiFab y0(y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::Vector<amrex::MultiFab> k(nk);
        for (int s = 0; s < nk; ++s) {
            k[s].define(y.boxArray(), y.DistributionMap(), nc, 0);
        }
        amrex::MultiFab::Copy(y0, y, 0, 0, nc, 0);

        amrex::Real const expo =
            (m_scheme == Scheme::RKF45) ? -0.2_rt : -0.5_rt;
        amrex::Real dt = t_end;

        while (st.t_done < t_end*(1.0_rt - 1.0e-12_rt))
        {
            if (st.n_attempts >= m_max_attempts) { break; }
            ++st.n_attempts;

            // k1 (side effect: whatever state the ceiling functor reads)
            m_rhs(y, k[0]);
            if (m_cap) { dt = amrex::min(dt, m_cap()); }
            dt = amrex::min(dt, t_end - st.t_done);

            amrex::Real err;
            if (m_scheme == Scheme::RKF45) {
                err = StepRKF45(y, y0, k, dt);
            } else {
                err = StepSSPRK2(y, y0, k, dt);
            }

            amrex::Real const fac =
                m_safety * std::pow(err + 1.0e-10_rt, expo);
            if (err <= 1.0_rt) {
                if (m_post_step) { m_post_step(y, dt); }
                st.t_done += dt;
                ++st.n_accepted;
                st.dt_last = dt;
                amrex::MultiFab::Copy(y0, y, 0, 0, nc, 0);
                dt *= amrex::min(m_max_growth, fac);
            } else {
                amrex::MultiFab::Copy(y, y0, 0, 0, nc, 0);
                dt *= amrex::max(0.1_rt, fac);
            }
        }
        return st;
    }

private:
    /** Weighted max-norm error ratio of e against y0 (globally reduced). */
    public:
    amrex::Real ErrNorm (amrex::MultiFab const & e,
                         amrex::MultiFab const & y0) const
    {
        using namespace amrex::literals;
        amrex::Real const rtol = m_rtol, atol = m_atol;
        int const nc = e.nComp();
        amrex::ReduceOps<amrex::ReduceOpMax> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(e); mfi.isValid(); ++mfi) {
            amrex::Box const box = mfi.validbox();
            auto const & e_arr = e.const_array(mfi);
            auto const & y_arr = y0.const_array(mfi);
            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) -> ReduceTuple
            {
                amrex::Real r = 0.0_rt;
                for (int c = 0; c < nc; ++c) {
                    r = amrex::max(r, std::abs(e_arr(i,j,kk,c)) /
                        (atol + rtol*std::abs(y_arr(i,j,kk,c))));
                }
                return {r};
            });
        }
        auto tup = reduce_data.value(reduce_op);
        amrex::Real err = amrex::get<0>(tup);
        amrex::ParallelDescriptor::ReduceRealMax(err);
        return err;
    }
    private:

    /** One SSPRK2 attempt: y <- candidate; returns the error ratio from
     *  the embedded Euler solution (e = dt/2 (k2 - k1)). */
    amrex::Real StepSSPRK2 (amrex::MultiFab & y, amrex::MultiFab const & y0,
                            amrex::Vector<amrex::MultiFab> & k,
                            amrex::Real const dt) const
    {
        using namespace amrex::literals;
        int const nc = y.nComp();
        // predictor y = y0 + dt k1 (also the embedded order-1 solution)
        amrex::MultiFab::LinComb(y, 1.0_rt, y0, 0, dt, k[0], 0, 0, nc, 0);
        m_rhs(y, k[1]);
        // corrector y = y0 + dt/2 (k1 + k2); error e = dt/2 (k2 - k1)
        amrex::MultiFab e(y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::MultiFab::LinComb(e, -0.5_rt*dt, k[0], 0, 0.5_rt*dt, k[1],
                                 0, 0, nc, 0);
        amrex::MultiFab::LinComb(y, 1.0_rt, y0, 0, 0.5_rt*dt, k[0],
                                 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(y, 0.5_rt*dt, k[1], 0, 0, nc, 0);
        return ErrNorm(e, y0);
    }

    /** One Fehlberg 4(5) attempt (coefficients identical to
     *  BfieldEvolveRKF45): y <- 5th-order candidate; returns the error
     *  ratio from the 4th/5th difference. */
    amrex::Real StepRKF45 (amrex::MultiFab & y, amrex::MultiFab const & y0,
                           amrex::Vector<amrex::MultiFab> & k,
                           amrex::Real const dt) const
    {
        using namespace amrex::literals;
        constexpr amrex::Real a21 = 1._rt/4._rt;
        constexpr amrex::Real a31 = 3._rt/32._rt,      a32 = 9._rt/32._rt;
        constexpr amrex::Real a41 = 1932._rt/2197._rt, a42 = -7200._rt/2197._rt,
                              a43 = 7296._rt/2197._rt;
        constexpr amrex::Real a51 = 439._rt/216._rt,   a52 = -8._rt,
                              a53 = 3680._rt/513._rt,  a54 = -845._rt/4104._rt;
        constexpr amrex::Real a61 = -8._rt/27._rt,     a62 = 2._rt,
                              a63 = -3544._rt/2565._rt, a64 = 1859._rt/4104._rt,
                              a65 = -11._rt/40._rt;
        // 5th-order weights (solution) and 4th-order weights (embedded)
        constexpr amrex::Real b51 = 16._rt/135._rt,    b53 = 6656._rt/12825._rt,
                              b54 = 28561._rt/56430._rt, b55 = -9._rt/50._rt,
                              b56 = 2._rt/55._rt;
        constexpr amrex::Real b41 = 25._rt/216._rt,    b43 = 1408._rt/2565._rt,
                              b44 = 2197._rt/4104._rt, b45 = -1._rt/5._rt;

        int const nc = y.nComp();
        auto stage = [&] (std::initializer_list<std::pair<amrex::Real, int>>
                              terms, int const kout)
        {
            amrex::MultiFab::Copy(y, y0, 0, 0, nc, 0);
            for (auto const & t : terms) {
                amrex::MultiFab::Saxpy(y, dt*t.first, k[t.second], 0, 0, nc, 0);
            }
            m_rhs(y, k[kout]);
        };
        stage({{a21, 0}},                                     1);
        stage({{a31, 0}, {a32, 1}},                           2);
        stage({{a41, 0}, {a42, 1}, {a43, 2}},                 3);
        stage({{a51, 0}, {a52, 1}, {a53, 2}, {a54, 3}},       4);
        stage({{a61, 0}, {a62, 1}, {a63, 2}, {a64, 3}, {a65, 4}}, 5);

        // 5th-order candidate and 4th/5th error in one pass each
        amrex::MultiFab e(y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::MultiFab::LinComb(y, 1.0_rt, y0, 0, dt*b51, k[0], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(y, dt*b53, k[2], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(y, dt*b54, k[3], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(y, dt*b55, k[4], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(y, dt*b56, k[5], 0, 0, nc, 0);
        amrex::MultiFab::LinComb(e, dt*(b51 - b41), k[0], 0,
                                 dt*(b53 - b43), k[2], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(e, dt*(b54 - b44), k[3], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(e, dt*(b55 - b45), k[4], 0, 0, nc, 0);
        amrex::MultiFab::Saxpy(e, dt*b56, k[5], 0, 0, nc, 0);
        return ErrNorm(e, y0);
    }

    Scheme m_scheme;
    Rhs m_rhs;
    Cap m_cap;
    PostStep m_post_step;
    amrex::Real m_rtol, m_atol, m_safety, m_max_growth;
    int m_max_attempts;
};

#endif // WARPX_QDSMCRKINTEGRATOR_H_
