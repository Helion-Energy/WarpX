/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_QDSMC_CONDUCTION_BOUNDARY_H_
#define WARPX_QDSMC_CONDUCTION_BOUNDARY_H_
#include "Utils/WarpXConst.H"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>

/** External leg law only: the native tensor operator owns the inward face.
 * Parser time is frozen for the conduction half. All temperatures here are K.
 */
struct QdsmcConductionLeg {
    amrex::ParserExecutor<3> kappa;
    amrex::Real length, reservoir, eta, speed_coefficient, time;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    flux (amrex::Real const tb, amrex::Real const neff,
          amrex::Real const ncap) const {
        amrex::Real const t = amrex::max(tb, amrex::Real(0.0));
        amrex::Real const conductivity =
            amrex::max(kappa(ncap, t * PhysConst::kb / PhysConst::q_e, time),
                       amrex::Real(0.0));
        amrex::Real const q =
            conductivity * amrex::max(t - reservoir, amrex::Real(0.0)) / length;
        amrex::Real const cap =
            eta * neff * PhysConst::kb * t * std::sqrt(speed_coefficient * t);
        return eta > 0.0 ? amrex::min(q, cap) : q;
    }

    /** Local branch stiffness estimate. The two one-sided differences also
     * cover a cap kink; q/(T-Tres) retains the linear rate near equilibrium.
     * Factor two is a local margin, not a rigorous bound for arbitrary parser
     * expressions. The caller revalidates every stage and qualifies configured
     * nonlinear laws against time refinement. No general RKL stability claim.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    slope (amrex::Real const tb, amrex::Real const neff, amrex::Real const ncap,
           amrex::Real const q) const {
        amrex::Real const scale = amrex::max(std::abs(tb), amrex::Real(1.0));
        amrex::Real const h = amrex::Real(1.e-5) * scale;
        amrex::Real const qp = flux(tb + h, neff, ncap);
        amrex::Real const qm = flux(tb - h, neff, ncap);
        amrex::Real const secant = tb > reservoir ? q / (tb - reservoir) : 0.0;
        return 2.0 *
               amrex::max(secant,
                          amrex::max(std::abs(qp - q), std::abs(q - qm)) / h);
    }
};
/** Reuse call-owned sum/nonnegative-max reduction scratch across stages.
 * AMReX ReduceData has no reset operation; its public per-stream block counts
 * (GPU) or per-thread tuples (CPU) define the accumulated data. A fresh empty
 * ReduceOps object also invalidates its cached value. Streams stay unchanged.
 */
template <typename... Ops, typename... Values>
void
ResetConductionReduction (amrex::ReduceOps<Ops...>& op,
                          amrex::ReduceData<Values...>& data) {
    op = amrex::ReduceOps<Ops...>{};
#ifdef AMREX_USE_GPU
    data.nBlocks().fill(0);
#else
    for (auto& value : data.reference()) {
        value = typename amrex::ReduceData<Values...>::Type{};
    }
#endif
}
#endif
