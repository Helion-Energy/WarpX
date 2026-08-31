/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMCRKINTEGRATOR_H_
#define WARPX_QDSMCRKINTEGRATOR_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <cstdlib>
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
    // dt trajectory (accepted steps): discriminates a cap/physics-bound
    // crawl (flat at the cap) from controller decay or a frozen-growth
    // defect (see the attempts-budget warnings that print these).
    amrex::Real dt_first = 0.0;
    amrex::Real dt_min = 0.0;
    amrex::Real dt_max = 0.0;
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
 *   - RKL2: Runge--Kutta--Legendre super-time-stepping (Meyer, Balsara
 *     & Aslam, JCP 257 (2014)): one super-step of s stages covers the
 *     stability window of ~s^2/4 explicit steps, so a parabolic
 *     subcycle count N costs ~2 sqrt(N) RHS evaluations instead of
 *     ~2 N. PARABOLIC OPERATORS ONLY (the stability region hugs the
 *     negative real axis -- useless and unsafe on advection). NOT SSP
 *     and not error-adaptive: the safety factor scales the super-step
 *     against the stability ceiling, rtol/atol are unused, and the
 *     monotone/maximum-principle guarantee of SSPRK2 is demoted to a
 *     gated measurement (same production rule as RKF45 above).
 *
 *  Controller (SSPRK2/RKF45, matches BfieldEvolve): accept when the
 *  weighted max-norm error ratio
 *  err = max |y_hi - y_lo| / (atol + rtol |y_old|) <= 1;
 *  dt *= clamp(safety * err^(-1/(p+1))) with growth factor max_growth
 *  on accept and a floor of 0.1 on reject. */
class QdsmcRKIntegrator
{
public:
    enum class Scheme : int { SSPRK2 = 0, RKF45 = 1, RKL2 = 2 };

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

    /** err_mask (optional): cells with a nonzero mask value are excluded
     *  from the error-norm reduction. For states owned by a post-step
     *  projection (e.g. a positivity floor): the projection re-creates a
     *  grid-scale kink at its cells every accepted step, the embedded
     *  pair correctly flags it, and the controller collapses dt to
     *  resolve dynamics the projection immediately overwrites -- a
     *  clamp/controller limit cycle. Masking the projected cells
     *  restores error control over the cells the integrator actually
     *  owns. The pointee must outlive the integrator; the caller updates
     *  it (post-step) so each attempt's norm sees the LAST accepted
     *  step's projection set. nullptr = today's behavior exactly. */
    QdsmcRKIntegrator (Scheme scheme, Rhs rhs, Cap cap,
                       amrex::Real rtol, amrex::Real atol,
                       amrex::Real safety, amrex::Real max_growth,
                       int max_attempts, PostStep post_step = {},
                       amrex::iMultiFab const * err_mask = nullptr)
        : m_scheme(scheme), m_rhs(std::move(rhs)), m_cap(std::move(cap)),
          m_post_step(std::move(post_step)),
          m_rtol(rtol), m_atol(atol), m_safety(safety),
          m_max_growth(max_growth), m_max_attempts(max_attempts),
          m_err_mask(err_mask)
    {}

    /** Advance y in place from t = 0 to t_end. y's ghost refresh is the
     *  RHS functor's job; on return y's ghosts are stale. */
    QdsmcRKStats Advance (amrex::MultiFab & y, amrex::Real const t_end) const
    {
        using namespace amrex::literals;
        QdsmcRKStats st;
        if (t_end <= 0.0_rt) { return st; }

        if (m_scheme == Scheme::RKL2) { return AdvanceRKL2(y, t_end); }

        int const nk = (m_scheme == Scheme::RKF45) ? 6 : 2;
        int const nc = y.nComp();
        amrex::MultiFab y0(y.boxArray(), y.DistributionMap(), nc, 0);
        // Error scratch is hoisted out of the attempt loop: allocating it
        // per attempt (as a Step-local) costs one arena round trip per
        // substep, which at production substep counts is tens of millions
        // of allocations per stroke.
        amrex::MultiFab e_scratch(y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::Vector<amrex::MultiFab> k(nk);
        for (int s = 0; s < nk; ++s) {
            k[s].define(y.boxArray(), y.DistributionMap(), nc, 0);
        }
        amrex::MultiFab::Copy(y0, y, 0, 0, nc, 0);

        amrex::Real const expo =
            (m_scheme == Scheme::RKF45) ? -0.2_rt : -0.5_rt;
        amrex::Real dt = t_end;

        // SSPRK2 archive invariant: y == y0 (bitwise) at every attempt
        // entry -- established by the copy above, re-established by the
        // fused predictor's archive on accepted attempts and by the
        // rewind copy on rejected ones. The predictor reads y in place of
        // y0 and archives it, which retires the per-accept y0 <- y copy.
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
                err = StepRKF45(y, y0, k, e_scratch, dt);
            } else {
                err = StepSSPRK2(y, y0, k, e_scratch, dt);
            }

            amrex::Real const fac =
                m_safety * std::pow(err + 1.0e-10_rt, expo);
            if (err <= 1.0_rt) {
                if (m_post_step) { m_post_step(y, dt); }
                st.t_done += dt;
                ++st.n_accepted;
                st.dt_last = dt;
                if (st.n_accepted == 1) {
                    st.dt_first = dt; st.dt_min = dt; st.dt_max = dt;
                } else {
                    st.dt_min = amrex::min(st.dt_min, dt);
                    st.dt_max = amrex::max(st.dt_max, dt);
                }
                if (m_scheme == Scheme::RKF45) {
                    amrex::MultiFab::Copy(y0, y, 0, 0, nc, 0);
                }
                dt *= amrex::min(m_max_growth, fac);
            } else {
                amrex::MultiFab::Copy(y, y0, 0, 0, nc, 0);
                dt *= amrex::max(0.1_rt, fac);
            }
        }
        return st;
    }

    // nvcc (CUDA >= 13) requires that member functions whose bodies
    // contain extended __device__ lambdas (every ParallelFor) be public.
public:
    /** RKL2 super-time-stepping advance. Per super-step tau, the stage
     *  count s satisfies tau <= safety * dt_expl * (s^2+s-2)/4 with
     *  dt_expl from the ceiling functor (evaluated once at super-step
     *  entry; the coefficient state is that of the entry RHS call).
     *  Recurrence (Meyer, Balsara & Aslam 2014):
     *      Y_1 = Y_0 + mu_t1 tau M(Y_0),          mu_t1 = b_1 w_1
     *      Y_j = mu_j Y_{j-1} + nu_j Y_{j-2} + (1 - mu_j - nu_j) Y_0
     *            + mu_tj tau M(Y_{j-1}) + ga_tj tau M(Y_0),  j = 2..s
     *  with b_0 = b_1 = 1/3, b_j = (j^2+j-2)/(2j(j+1)),
     *  w_1 = 4/(s^2+s-2), mu_j = ((2j-1)/j) b_j/b_{j-1},
     *  nu_j = -((j-1)/j) b_j/b_{j-2}, mu_tj = mu_j w_1,
     *  ga_tj = -(1 - b_{j-1}) mu_tj.
     *  post_step runs after EVERY stage with the stage's effective time
     *  increment (c_j - c_{j-1}) tau, c_j = (j^2+j-2)/(s^2+s-2) and
     *  c_1 = mu_t1; the increments sum to tau exactly, so BC flux
     *  injections scaled by the post-step dt still integrate to the
     *  full interval. Stage states are built in the caller's y (the
     *  ghosted buffer the RHS functor refreshes); Y_{j-1}, Y_{j-2},
     *  M(Y_0) are archived in ghostless temporaries. Stats: n_attempts
     *  and n_accepted count STAGES (one RHS evaluation each, the work
     *  unit the attempts budget bounds); the dt trajectory books the
     *  per-stage increments. The stage count of a super-step is sized
     *  to the remaining budget BEFORE the recurrence starts -- the
     *  recurrence cannot stop midway (intermediate stages are not
     *  consistent solutions), so a budget-limited super-step covers a
     *  shorter tau and the while loop continues (or exits with
     *  t_done < t_end, which the caller's budget warning reports). */
    QdsmcRKStats AdvanceRKL2 (amrex::MultiFab & y, amrex::Real const t_end) const
    {
        using namespace amrex::literals;
        QdsmcRKStats st;
        int const nc = y.nComp();
        // Y_{j-1} needs no archive of its own: the caller's y holds it at
        // every stage entry (the fused stage kernel reads it there before
        // overwriting), so only Y_{j-2}, Y_0, and the two RHS archives
        // are kept.
        amrex::MultiFab y0  (y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::MultiFab yjm2(y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::MultiFab my0 (y.boxArray(), y.DistributionMap(), nc, 0);
        amrex::MultiFab mjm1(y.boxArray(), y.DistributionMap(), nc, 0);

        auto book = [&] (amrex::Real const dts)
        {
            ++st.n_attempts;
            ++st.n_accepted;
            st.dt_last = dts;
            if (st.n_accepted == 1) {
                st.dt_first = dts; st.dt_min = dts; st.dt_max = dts;
            } else {
                st.dt_min = amrex::min(st.dt_min, dts);
                st.dt_max = amrex::max(st.dt_max, dts);
            }
        };

        while (st.t_done < t_end*(1.0_rt - 1.0e-12_rt))
        {
            if (st.n_attempts >= m_max_attempts) { break; }

            // Entry evaluation M(Y_0) (side effect: the ceiling functor's
            // coefficient state) and the stability-sized stage count.
            m_rhs(y, my0);
            amrex::Real const dt_expl = m_cap ? m_cap()
                : std::numeric_limits<amrex::Real>::max();
            amrex::Real tau = t_end - st.t_done;
            int s = 2;
            if (tau > m_safety*dt_expl) {
                amrex::Real const nn = tau/(m_safety*dt_expl);
                s = static_cast<int>(std::ceil(
                        0.5_rt*(std::sqrt(9.0_rt + 16.0_rt*nn) - 1.0_rt)));
                s = std::max(s, 2);
            }
            int const s_budget = m_max_attempts - st.n_attempts;
            if (s > s_budget) {
                s = std::max(2, s_budget);
                tau = amrex::min(tau, m_safety*dt_expl
                    * 0.25_rt*(amrex::Real(s)*(amrex::Real(s) + 1.0_rt)
                               - 2.0_rt));
            }

            amrex::Real const ssq =
                amrex::Real(s)*(amrex::Real(s) + 1.0_rt) - 2.0_rt;
            amrex::Real const w1 = 4.0_rt/ssq;

            // Stage 1 (into y; archives y0 = yjm2 = Y_0 and applies the
            // Euler update in one fused launch, in place of a copy, a
            // LinComb, and two archive copies)
            amrex::Real const mu_t1 = w1/3.0_rt;   // b_1 w_1
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(y0, amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                amrex::Box const bx = mfi.tilebox();
                auto const & y_arr    = y.array(mfi);
                auto const & y0_arr   = y0.array(mfi);
                auto const & yjm2_arr = yjm2.array(mfi);
                auto const & my0_arr  = my0.const_array(mfi);
                amrex::Real const c1 = mu_t1*tau;
                amrex::ParallelFor(bx, nc,
                    [=] AMREX_GPU_DEVICE (int i, int j, int kk, int c)
                {
                    amrex::Real const yv = y_arr(i,j,kk,c);
                    y0_arr(i,j,kk,c) = yv;
                    yjm2_arr(i,j,kk,c) = yv;
                    y_arr(i,j,kk,c) = 1.0_rt*yv + c1*my0_arr(i,j,kk,c);
                });
            }
            if (m_post_step) { m_post_step(y, mu_t1*tau); }
            book(mu_t1*tau);
            amrex::Real c_prev = mu_t1;
            amrex::Real b_jm1 = 1.0_rt/3.0_rt;     // b_1
            amrex::Real b_jm2 = 1.0_rt/3.0_rt;     // b_0

            for (int j = 2; j <= s; ++j) {
                // y holds Y_{j-1} (with its post-step mutation): evaluate
                // M(Y_{j-1}) (ghost refresh is the RHS functor's job),
                // then one fused launch computes Y_j in the exact
                // per-element accumulation order of the original
                // LinComb + 3 Saxpy sequence (bit-identical), reading
                // Y_{j-1} from y before overwriting and re-archiving it
                // as the next stage's Y_{j-2}.
                m_rhs(y, mjm1);
                amrex::Real const rj = amrex::Real(j);
                amrex::Real const bj =
                    (rj*rj + rj - 2.0_rt)/(2.0_rt*rj*(rj + 1.0_rt));
                amrex::Real const mu   = ((2.0_rt*rj - 1.0_rt)/rj)*bj/b_jm1;
                amrex::Real const nu   = -((rj - 1.0_rt)/rj)*bj/b_jm2;
                amrex::Real const mu_t = mu*w1;
                amrex::Real const ga_t = -(1.0_rt - b_jm1)*mu_t;
                amrex::Real const w0   = 1.0_rt - mu - nu;
                amrex::Real const ct_m = mu_t*tau;
                amrex::Real const ct_g = ga_t*tau;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(y0, amrex::TilingIfNotGPU());
                     mfi.isValid(); ++mfi) {
                    amrex::Box const bx = mfi.tilebox();
                    auto const & y_arr    = y.array(mfi);
                    auto const & yjm2_arr = yjm2.array(mfi);
                    auto const & y0_arr   = y0.const_array(mfi);
                    auto const & mjm1_arr = mjm1.const_array(mfi);
                    auto const & my0_arr  = my0.const_array(mfi);
                    amrex::ParallelFor(bx, nc,
                        [=] AMREX_GPU_DEVICE (int i, int jj, int kk, int c)
                    {
                        amrex::Real const yv = y_arr(i,jj,kk,c);
                        amrex::Real r = mu*yv + nu*yjm2_arr(i,jj,kk,c);
                        r = r + w0*y0_arr(i,jj,kk,c);
                        r = r + ct_m*mjm1_arr(i,jj,kk,c);
                        r = r + ct_g*my0_arr(i,jj,kk,c);
                        yjm2_arr(i,jj,kk,c) = yv;
                        y_arr(i,jj,kk,c) = r;
                    });
                }
                amrex::Real const cj = (rj*rj + rj - 2.0_rt)/ssq;
                if (m_post_step) { m_post_step(y, (cj - c_prev)*tau); }
                book((cj - c_prev)*tau);
                c_prev = cj;
                b_jm2 = b_jm1;
                b_jm1 = bj;
            }
            st.t_done += tau;
        }
        return st;
    }

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
        bool const has_mask = (m_err_mask != nullptr);
        for (amrex::MFIter mfi(e); mfi.isValid(); ++mfi) {
            amrex::Box const box = mfi.validbox();
            auto const & e_arr = e.const_array(mfi);
            auto const & y_arr = y0.const_array(mfi);
            amrex::Array4<int const> const m_arr = has_mask
                ? m_err_mask->const_array(mfi) : amrex::Array4<int const>{};
            reduce_op.eval(box, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int kk) -> ReduceTuple
            {
                if (has_mask && m_arr(i,j,kk) != 0) { return {0.0_rt}; }
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
        // WARPX_QDSMC_ERR_DEBUG: on rejecting evaluations, locate and
        // print the error-ratio argmax cell with its ingredients
        // (device-safe single-cell gather; the caller's attempts-budget
        // warning supplies the dt context). Zero cost when unset.
        static bool const err_debug =
            (std::getenv("WARPX_QDSMC_ERR_DEBUG") != nullptr);
        if (err_debug && !(err <= 1.0_rt)) {
            amrex::MultiFab ratio(e.boxArray(), e.DistributionMap(), 1, 0);
            for (amrex::MFIter mfi(ratio); mfi.isValid(); ++mfi) {
                amrex::Box const box = mfi.validbox();
                auto const & r_arr = ratio.array(mfi);
                auto const & e_arr = e.const_array(mfi);
                auto const & y_arr = y0.const_array(mfi);
                amrex::Array4<int const> const m_arr = has_mask
                    ? m_err_mask->const_array(mfi)
                    : amrex::Array4<int const>{};
                amrex::ParallelFor(box,
                    [=] AMREX_GPU_DEVICE (int i, int j, int kk)
                {
                    if (has_mask && m_arr(i,j,kk) != 0) {
                        r_arr(i,j,kk) = 0.0_rt;   // excluded from the norm
                        return;
                    }
                    amrex::Real r = 0.0_rt;
                    for (int c = 0; c < nc; ++c) {
                        r = amrex::max(r, std::abs(e_arr(i,j,kk,c)) /
                            (atol + rtol*std::abs(y_arr(i,j,kk,c))));
                    }
                    r_arr(i,j,kk) = r;
                });
            }
            amrex::IntVect const loc = ratio.maxIndex(0);
            amrex::Real ev = 0.0_rt, yv = 0.0_rt;
            for (amrex::MFIter mfi(ratio); mfi.isValid(); ++mfi) {
                if (mfi.validbox().contains(loc)) {
                    amrex::Gpu::DeviceVector<amrex::Real> dv(2, 0.0_rt);
                    amrex::Real * const dvp = dv.data();
                    auto const & e_arr = e.const_array(mfi);
                    auto const & y_arr = y0.const_array(mfi);
                    amrex::Box const one(loc, loc);
                    amrex::ParallelFor(one,
                        [=] AMREX_GPU_DEVICE (int i, int j, int kk)
                    {
                        dvp[0] = e_arr(i,j,kk,0);
                        dvp[1] = y_arr(i,j,kk,0);
                    });
                    amrex::Gpu::streamSynchronize();
                    amrex::Vector<amrex::Real> hv(2, 0.0_rt);
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                        dv.begin(), dv.end(), hv.begin());
                    ev = hv[0]; yv = hv[1];
                }
            }
            amrex::ParallelDescriptor::ReduceRealSum(ev);
            amrex::ParallelDescriptor::ReduceRealSum(yv);
            amrex::AllPrint() << "[qdsmc-err-debug] err=" << err
                << " argmax=(" << loc[0] << ","
                << ((AMREX_SPACEDIM > 1) ? loc[1] : 0) << ","
                << ((AMREX_SPACEDIM > 2) ? loc[2] : 0)
                << ") e=" << ev << " y0=" << yv << "\n";
        }
        return err;
    }
    // Public for the same nvcc-13 extended-__device__-lambda rule.
    public:

    /** One SSPRK2 attempt: y <- candidate; returns the error ratio from
     *  the embedded Euler solution (e = dt/2 (k2 - k1)).
     *
     *  Fused-glue form. The predictor requires (and the accept path of
     *  Advance maintains) y == y0 bitwise at entry: it reads y in place
     *  of y0, archives it into y0 (retiring the per-accept y0 <- y
     *  copy), and applies the Euler update -- one launch instead of a
     *  copy plus a LinComb. The corrector fuses the error assembly and
     *  the two-pass solution update into a single launch, accumulating
     *  in the exact per-element order of the original LinComb + Saxpy
     *  sequence, so all results are bit-identical to the unfused form. */
    amrex::Real StepSSPRK2 (amrex::MultiFab & y, amrex::MultiFab & y0,
                            amrex::Vector<amrex::MultiFab> & k,
                            amrex::MultiFab & e,
                            amrex::Real const dt) const
    {
        using namespace amrex::literals;
        int const nc = y.nComp();
        // predictor: y0 = y (archive); y = y0 + dt k1 (embedded order-1)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(y0, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            amrex::Box const bx = mfi.tilebox();
            auto const & y_arr  = y.array(mfi);
            auto const & y0_arr = y0.array(mfi);
            auto const & k1_arr = k[0].const_array(mfi);
            amrex::ParallelFor(bx, nc,
                [=] AMREX_GPU_DEVICE (int i, int j, int kk, int c)
            {
                amrex::Real const yv = y_arr(i,j,kk,c);
                y0_arr(i,j,kk,c) = yv;
                y_arr(i,j,kk,c) = yv + dt*k1_arr(i,j,kk,c);
            });
        }
        m_rhs(y, k[1]);
        // corrector y = y0 + dt/2 (k1 + k2); error e = dt/2 (k2 - k1)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(y0, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            amrex::Box const bx = mfi.tilebox();
            auto const & y_arr  = y.array(mfi);
            auto const & e_arr  = e.array(mfi);
            auto const & y0_arr = y0.const_array(mfi);
            auto const & k1_arr = k[0].const_array(mfi);
            auto const & k2_arr = k[1].const_array(mfi);
            amrex::ParallelFor(bx, nc,
                [=] AMREX_GPU_DEVICE (int i, int j, int kk, int c)
            {
                amrex::Real const k1v = k1_arr(i,j,kk,c);
                amrex::Real const k2v = k2_arr(i,j,kk,c);
                e_arr(i,j,kk,c) = (-0.5_rt*dt)*k1v + (0.5_rt*dt)*k2v;
                amrex::Real r = 1.0_rt*y0_arr(i,j,kk,c) + (0.5_rt*dt)*k1v;
                r = r + (0.5_rt*dt)*k2v;
                y_arr(i,j,kk,c) = r;
            });
        }
        return ErrNorm(e, y0);
    }

    private:

    /** One Fehlberg 4(5) attempt (coefficients identical to
     *  BfieldEvolveRKF45): y <- 5th-order candidate; returns the error
     *  ratio from the 4th/5th difference. */
    amrex::Real StepRKF45 (amrex::MultiFab & y, amrex::MultiFab const & y0,
                           amrex::Vector<amrex::MultiFab> & k,
                           amrex::MultiFab & e,
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
    /** Optional error-norm exclusion set (see the constructor doc);
     *  not owned. */
    amrex::iMultiFab const * m_err_mask = nullptr;
};

#endif // WARPX_QDSMCRKINTEGRATOR_H_
