/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_FINITE_WALL_H_
#define WARPX_THERMAL_FINITE_WALL_H_
#include "ThermalFiniteFluxResponse.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalWall.H"
namespace warpx::thermal::finite_flux {
struct LegRoot {Ball temperature;Real width=0.,residual=0.;};
// The quadratic law has a strictly decreasing interface equation on the
// drain branch. Keep 56 bisections. If an operation enclosure cannot decide a
// branch, retain the current bracket; no old branch is silently reused.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE LegRoot
LegInterface(Ball tc,LegFaceParameters const& p) noexcept {
    Ball lo{{p.bath_ev,0.},0.},hi=tc;
    for(int it=0;it<56;++it) {
        auto const tf=Add(lo,Scale(Add(hi,Neg(lo)),.5));
        auto const tc2=Mul(tc,tc),tf2=Mul(tf,tf);
        auto const f=Add(Scale(Mul(Add(tc2,tf2),Add(tc,Neg(tf))),p.length),
                         Neg(Scale(Mul(tf2,Add(tf,Neg({{p.bath_ev,0.},0.}))),2*p.half_cell)));
        int const sign=Sign(f);
        if(sign==1)lo=tf;else if(sign==-1)hi=tf;
        // Exact zero or uncertain sign: the root lies in this operation
        // interval. Continue the fixed count without narrowing unsafely.
    }
    auto center=Add(lo,Scale(Add(hi,Neg(lo)),.5));
    Real const width=Upper(Mag(Add(hi,Neg(lo)).value)+hi.radius+lo.radius);
    auto const tf2=Mul(center,center);
    auto const f=Add(Scale(Mul(Add(Mul(tc,tc),tf2),Add(tc,Neg(center))),p.length),
                     Neg(Scale(Mul(tf2,Add(center,Neg({{p.bath_ev,0.},0.}))),2*p.half_cell)));
    // Positive lower bound of -dF/dTf: the first term alone suffices because
    // 3*Tf-2*Tb >= Tf >= Tb on the qualified nonnegative drain branch.
    Real const lower=lo.value.hi-std::abs(lo.value.lo)-lo.radius;
    Real const slope=lower>0.?2*p.length*lower*lower:0.;
    Real const residual=Upper(Mag(f.value)+f.radius);
    Real radius=.5*width+center.radius;
    if(slope>0.)radius=amrex::min(radius,Upper(2*residual/slope));
    center.radius=Upper(radius);return {center,width,residual};
}
// The native capped power is nonnegative and nondecreasing in Te at fixed n.
// Divide its positive-conductance balance by (kc+kf)/(2h):
// F(Tf)=Tc-Tf-(2h/L)*(kf/(kc+kf))*(Tf-Tbath).
// On the drain branch kf/(kc+kf) is nondecreasing, so F' <= -1,
// including one-sided cap derivatives. The residual is therefore already a
// temperature-error bound. The native high producer keeps its own 56 steps.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball
LegEquation(Ball tc,Ball n,Ball tf,LegFaceParameters const& p,MaterialLaw const& law) noexcept {
    int bad=0;
    auto const kc=law.Evaluate(Value(n,{},&bad),Value(tc,{},&bad)).base;
    auto const kf=law.Evaluate(Value(n,{},&bad),Value(tf,{},&bad)).base;
    auto const fraction=Div(kf,Add(kc,kf));
    auto const ratio=Div(Scale({{p.half_cell,0.},0.},2.),{{p.length,0.},0.});
    auto const result=Add(Add(tc,Neg(tf)),Neg(Mul(Mul(ratio,fraction),Add(tf,Neg({{p.bath_ev,0.},0.})))));
    return bad?Invalid():result;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE LegRoot
LegInterface(Ball tc,Ball n,LegFaceParameters const& p,MaterialLaw const& law) noexcept {
    if(law.family==MaterialLaw::Quadratic)return LegInterface(tc,p);
    if(Sign(n)!=1||Sign(Add(tc,Neg({{p.bath_ev,0.},0.})))!=1||
       !(std::isfinite(p.bath_ev)&&p.bath_ev>=0.&&std::isfinite(p.length)&&p.length>0.&&
         std::isfinite(p.half_cell)&&p.half_cell>0.))return {Invalid(),0.,0.};
    Ball lo{{p.bath_ev,0.},0.},hi=tc;
    for(int it=0;it<56;++it){
        auto const tf=Add(lo,Scale(Add(hi,Neg(lo)),.5));
        auto const f=LegEquation(tc,n,tf,p,law);int const sign=Sign(f);
        if(!Valid(f))return {Invalid(),0.,0.};
        if(sign==1)lo=tf;else if(sign==-1)hi=tf;
    }
    auto center=Add(lo,Scale(Add(hi,Neg(lo)),.5));
    auto const span=Add(hi,Neg(lo));
    Real const width=Upper(Mag(span.value)+span.radius);
    auto const f=LegEquation(tc,n,{center.value,0.},p,law);
    if(!Valid(f))return {Invalid(),0.,0.};
    Real const residual=Upper(Mag(f.value)+f.radius);
    center.radius=amrex::min(Upper(.5*width+center.radius),residual);
    return {center,width,residual};
}
struct LegIncrement {Value flux;Real base_width=0.,trial_width=0.,base_residual=0.,trial_residual=0.;};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE LegIncrement
Leg(Value tc,Value n,LegFaceParameters const& p,MaterialLaw const& law) noexcept {
    int* bad=Context(tc,n);LegIncrement result;
    Value const threshold(amrex::max(p.bath_ev,p.temperature_floor_ev));
    auto const difference=tc-threshold;
    int const sign0=Sign(difference.base),sign1=Sign(Add(difference.base,difference.delta));
    if(sign0==2||sign1==2||((sign0>0)!=(sign1>0))){if(bad)*bad=1;result.flux=Check({Invalid(),Invalid(),bad});return result;}
    if(sign0<=0||law.zero){result.flux=Value({{0.,0.},0.},{},bad);return result;}
    auto const r0=LegInterface(tc.base,n.base,p,law);
    auto const r1=LegInterface(Add(tc.base,tc.delta),Add(n.base,n.delta),p,law);
    result.base_width=r0.width;result.trial_width=r1.width;
    result.base_residual=r0.residual;result.trial_residual=r1.residual;
    // Zero retained input is exactly anchored: do not subtract two independent
    // root enclosures and invent a finite change or a zero-input uncertainty.
    bool const same_root=ExactZero(tc.delta)&&(law.family!=MaterialLaw::CappedPowerFiveHalves||ExactZero(n.delta));
    Ball const dtf=same_root?Ball{}:Add(r1.temperature,Neg(r0.temperature));
    Value const tf{r0.temperature,dtf,bad};
    auto const kc=law.Evaluate(n,tc),kf=law.Evaluate(n,tf);
    auto const gh=Value(.5)*(kc+kf)/Value(p.half_cell),gl=kf/Value(p.length);
    result.flux=gh/(gh+gl)*gl*(tc-Value(p.bath_ev))*Value(PhysConst::q_e)/Value(PhysConst::kb);
    return result;
}
}
#endif
