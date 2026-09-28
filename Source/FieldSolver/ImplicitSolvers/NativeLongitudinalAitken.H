/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_LONGITUDINAL_AITKEN_H_
#define WARPX_NATIVE_LONGITUDINAL_AITKEN_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>

namespace warpx::darwin {
namespace detail {
// Non-constructor kernel enclosure is required by CUDA extended lambdas.
inline void SetNativeLongitudinalMetric(amrex::MultiFab& weights,
    amrex::MultiFab const* shape,amrex::Geometry const& geom,
    amrex::GpuArray<int,AMREX_SPACEDIM> const& pmc_lo,
    amrex::GpuArray<int,AMREX_SPACEDIM> const& pmc_hi)
{
    auto owner=shape->OwnerMask(geom.periodicity());
    auto const domain=amrex::convert(geom.Domain(),shape->ixType());
    auto const node=shape->ixType().toIntVect();
    auto const low=domain.smallEnd(),high=domain.bigEnd();
#if defined(WARPX_DIM_RZ)
    double const r0=geom.ProbLo(0)/geom.CellSize(0);
    bool const axis=geom.ProbLo(0)==0.;
#endif
    for(amrex::MFIter mfi(weights);mfi.isValid();++mfi) {
        auto a=weights.array(mfi);auto o=owner->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            double v=1.;
#if defined(WARPX_DIM_RZ)
            v=r0+i+(node[0]?0.:.5);
            if(axis&&node[0]&&i==0)v=.125;
#endif
            amrex::IntVect p(AMREX_D_DECL(i,j,k));
            for(int d=0;d<AMREX_SPACEDIM;++d)
                if(node[d]&&((pmc_lo[d]&&p[d]==low[d])||
                              (pmc_hi[d]&&p[d]==high[d])))v*=.5;
            a(i,j,k)=o(i,j,k)?v:0.;
        });
    }
}
} // namespace detail
/** Solver-only inverse secant on native Schur corrections. The caller grades
 * the raw physical residual and applies the returned scalar identically to
 * its potential and field correction. This owns no accepted/time history. */
class NativeLongitudinalAitken {
public:
    using MF=amrex::MultiFab;
    using View=amrex::Array<MF*,3>;
    struct Result { double omega=1.; bool valid=true, secant=false; };
    NativeLongitudinalAitken(View const& shape,amrex::Geometry const& geom,
        amrex::GpuArray<int,AMREX_SPACEDIM> const& pmc_lo,
        amrex::GpuArray<int,AMREX_SPACEDIM> const& pmc_hi)
    {
        for(int c=0;c<3;++c) {
            AMREX_ALWAYS_ASSERT(shape[c] && shape[c]->nComp()==1);
            m_previous[c].define(shape[c]->boxArray(),shape[c]->DistributionMap(),1,0);
            m_weights[c].define(shape[c]->boxArray(),shape[c]->DistributionMap(),1,0);
            detail::SetNativeLongitudinalMetric(m_weights[c],shape[c],geom,pmc_lo,pmc_hi);
        }
    }
    void Reset() noexcept { m_have_previous=false; m_omega=1.; }
    double WeightedDot(View const& a,View const& b) const {
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<double> data(op);using T=decltype(data)::Type;
        for(int c=0;c<3;++c) {
            Check(c,*a[c]);Check(c,*b[c]);
            for(amrex::MFIter mfi(*a[c]);mfi.isValid();++mfi) {
                auto x=a[c]->const_array(mfi),y=b[c]->const_array(mfi),w=m_weights[c].const_array(mfi);
                op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                    return {w(i,j,k)*x(i,j,k)*y(i,j,k)};
                });
            }
        }
        double value=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(value);return value;
    }
    Result Select(View const& correction,double fallback,bool accelerate,
                  bool clamp_positive_secant=false) {
        AMREX_ALWAYS_ASSERT(std::isfinite(fallback)&&fallback>0.&&fallback<=1.);
        amrex::ReduceOps<amrex::ReduceOpSum,amrex::ReduceOpSum,amrex::ReduceOpSum> op;
        amrex::ReduceData<double,double,double> data(op);using T=decltype(data)::Type;
        bool const have=m_have_previous;
        for(int c=0;c<3;++c) {
            Check(c,*correction[c]);
            for(amrex::MFIter mfi(*correction[c]);mfi.isValid();++mfi) {
                auto x=correction[c]->const_array(mfi),p=m_previous[c].const_array(mfi),w=m_weights[c].const_array(mfi);
                op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                    double const now=x(i,j,k),old=have?p(i,j,k):0.,d=now-old,v=w(i,j,k);
                    return {v*old*d,v*d*d,v*(now*now+old*old)};
                });
            }
        }
        auto t=data.value();double dots[]{amrex::get<0>(t),amrex::get<1>(t),amrex::get<2>(t)};
        amrex::ParallelDescriptor::ReduceRealSum(dots,3);
        Result result;result.omega=fallback;
        for(double x:dots)if(!std::isfinite(x)){result.valid=false;return result;}
        double const eps=std::numeric_limits<double>::epsilon();
        if(accelerate&&have&&dots[1]>64.*eps*dots[2]) {
            double const candidate=-m_omega*dots[0]/dots[1];
            if(std::isfinite(candidate)&&candidate>=std::sqrt(eps)&&
               (candidate<=1.||clamp_positive_secant)) {
                // The constrained minimum of the frozen secant model is at
                // the upper bound when its positive stationary point is >1.
                result.omega=std::min(candidate,1.);result.secant=true;
            }
        }
        for(int c=0;c<3;++c)MF::Copy(m_previous[c],*correction[c],0,0,1,0);
        m_have_previous=true;m_omega=result.omega;return result;
    }
private:
    void Check(int c,MF const& f)const {
        AMREX_ALWAYS_ASSERT(f.nComp()==1&&f.boxArray()==m_previous[c].boxArray()&&
            f.DistributionMap()==m_previous[c].DistributionMap());
    }
    amrex::Array<MF,3> m_previous,m_weights;
    double m_omega=1.;bool m_have_previous=false;
};
} // namespace warpx::darwin
#endif
