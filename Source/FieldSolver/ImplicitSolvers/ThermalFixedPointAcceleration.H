/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_FIXED_POINT_ACCELERATION_H_
#define WARPX_THERMAL_FIXED_POINT_ACCELERATION_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <memory>

/** Scalar secant inverse for the segregated thermal fixed point.
 * The map being accelerated is the complete existing entropy/FD4/RKL2 stage.
 * Only its next guess changes; callers must grade the unaccelerated map and
 * re-evaluate it before acceptance. No particle or field vector visits the host.
 * Pressure follows the same temperature change at the emitted stage density.
 */
class ThermalFixedPointAcceleration
{
public:
    using MF = amrex::MultiFab;
    using RT = amrex::Real;
    void Capture (MF const& temperature)
    {
        if (!m_old) {
            m_old = std::make_unique<MF>(temperature.boxArray(),temperature.DistributionMap(),
                                         1,temperature.nGrowVect());
            m_previous = std::make_unique<MF>(temperature.boxArray(),temperature.DistributionMap(),1,0);
            m_previous->setVal(0.);
        }
        MF::Copy(*m_old,temperature,0,0,1,temperature.nGrowVect());
    }
    RT Apply (MF& temperature, MF& pressure)
    {
        AMREX_ALWAYS_ASSERT(m_old && temperature.nComp() == 1 && pressure.nComp() == 1);
        AMREX_ALWAYS_ASSERT(temperature.boxArray() == pressure.boxArray() &&
            temperature.DistributionMap() == pressure.DistributionMap() &&
            temperature.nGrowVect() == pressure.nGrowVect());
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum> ops;
        amrex::ReduceData<RT,RT> data(ops);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
            auto const now = temperature.const_array(mfi), old = m_old->const_array(mfi);
            auto const prev = m_previous->const_array(mfi);
            ops.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) -> Tuple {
                RT const r = now(i,j,k)-old(i,j,k), dr = r-prev(i,j,k);
                return {prev(i,j,k)*dr,dr*dr};
            });
        }
        auto const sums = data.value();
        RT dots[2] = {amrex::get<0>(sums),amrex::get<1>(sums)};
        amrex::ParallelDescriptor::ReduceRealSum(dots,2);
        RT omega = 1.;
        if (m_have_previous && dots[1] > 0.) {
            omega = -m_omega*dots[0]/dots[1];
            if (!std::isfinite(omega)) { omega = 1.; }
            omega = amrex::max(.1,amrex::min(2.,omega));
        }
        // Positivity safeguard: revert the entire extrapolation if any valid
        // temperature would cross zero. Never clip the residual to fake convergence.
        amrex::ReduceOps<amrex::ReduceOpMin> min_op;
        amrex::ReduceData<RT> min_data(min_op);
        using MinTuple = decltype(min_data)::Type;
        for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
            auto const now = temperature.const_array(mfi), old = m_old->const_array(mfi);
            min_op.eval(mfi.validbox(),min_data,[=] AMREX_GPU_DEVICE(int i,int j,int k) -> MinTuple {
                return {old(i,j,k)+omega*(now(i,j,k)-old(i,j,k))};
            });
        }
        RT minimum = amrex::get<0>(min_data.value());
        amrex::ParallelDescriptor::ReduceRealMin(minimum);
        if (minimum < 0.) { omega = 1.; }
        for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
            auto now = temperature.array(mfi), p = pressure.array(mfi);
            auto const old = m_old->const_array(mfi);
            auto prev = m_previous->array(mfi);
            auto const valid = mfi.validbox();
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                RT const raw = now(i,j,k), r = raw-old(i,j,k);
                if (valid.contains(amrex::IntVect(AMREX_D_DECL(i,j,k)))) { prev(i,j,k) = r; }
                RT const next = old(i,j,k)+omega*r;
                if (raw > 0. && next >= 0.) {
                    p(i,j,k) *= next/raw;
                    now(i,j,k) = next;
                }
            });
        }
        m_omega = omega;
        m_have_previous = true;
        return omega;
    }
private:
    std::unique_ptr<MF> m_old, m_previous;
    RT m_omega = 1.;
    bool m_have_previous = false;
};
#endif
