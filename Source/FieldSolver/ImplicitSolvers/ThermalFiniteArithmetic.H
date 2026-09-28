/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_FINITE_ARITHMETIC_H_
#define WARPX_THERMAL_FINITE_ARITHMETIC_H_
#include "ThermalCurrentRemainder.H"
#include <cmath>
#include <limits>
namespace warpx::thermal::finite_flux {
using remainder::Pair;
using Real=amrex::Real;
namespace dd=warpx::thermal::remainder;
// Enclosures describe the expression at the represented input pairs. They do
// not include particle, density projection, field solve or trajectory error.
// RN binary64, gradual underflow and explicit correctly-rounded fma/sqrt are
// required by ArithmeticSupported. No fast-math/device admission is inferred.
struct Ball { Pair value{0.,0.}; Real radius=0.; };
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real Upper(Real x) noexcept {
    return x==0. ? 0. : std::nextafter(x,std::numeric_limits<Real>::infinity());
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real Mag(Pair x) noexcept {
    return Upper(std::abs(x.hi)+std::abs(x.lo));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool ExactZero(Ball a) noexcept {
    return a.value.hi==0. && a.value.lo==0. && a.radius==0.;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Invalid() noexcept {
    Real const nan=std::numeric_limits<Real>::quiet_NaN();return {{nan,nan},nan};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool Valid(Ball x) noexcept {
    return std::isfinite(x.value.hi)&&std::isfinite(x.value.lo)&&std::isfinite(x.radius)&&x.radius>=0.;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Neg(Ball a) noexcept {
    return {dd::Negate(a.value),a.radius};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Add(Ball a,Ball b) noexcept {
    if(ExactZero(a))return b;
    if(ExactZero(b))return a;
    auto const s=dd::Sum(a.value.hi,b.value.hi);
    // TwoSum is exact in this environment barring overflow. Only the two
    // low-term additions before the final TwoSum round: bound their absolute
    // operands, with an outward factor covering bound arithmetic/subnormals.
    constexpr Real u=0x1p-53, tiny=std::numeric_limits<Real>::denorm_min();
    Real const round=4*u*(std::abs(s.lo)+std::abs(a.value.lo)+std::abs(b.value.lo))+8*tiny;
    return {dd::Add(a.value,b.value),Upper(Upper(a.radius+b.radius)+round)};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Scale(Ball a,Real b) noexcept {
    if(b==0.)return {};
    if(b==1.)return a;
    if(b==-1.)return Neg(a);
    if(ExactZero(a))return {};
    constexpr Real u=0x1p-53,tiny=std::numeric_limits<Real>::denorm_min();
    Real const main=std::abs(a.value.hi*b),low=std::abs(a.value.lo*b);
    // fma product residual + low multiplication + their addition. Explicit
    // low-part terms are retained even when a is anchored to native high.
    Real const round=16*u*u*main+16*u*low+16*tiny;
    return {dd::Multiply(a.value,b),Upper(Upper(std::abs(b)*a.radius)+round)};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Mul(Ball a,Ball b) noexcept {
    if(ExactZero(a)||ExactZero(b))return {};
    auto r=Add(Scale(a,b.value.hi),Scale(a,b.value.lo));
    r.radius=Upper(r.radius+Upper(Upper(Mag(a.value)+a.radius)*b.radius));return r;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Div(Ball a,Ball b) noexcept {
    Real const denominator=std::nextafter(std::nextafter(std::abs(b.value.hi)-std::abs(b.value.lo),-std::numeric_limits<Real>::infinity())-b.radius,-std::numeric_limits<Real>::infinity());
    if(!(denominator>0.) || !Valid(a)||!Valid(b))return Invalid();
    if(ExactZero(a))return {};
    Real const q=a.value.hi/b.value.hi;
    auto const residual=dd::Add(a.value,dd::Negate(dd::Multiply(b.value,q)));
    Pair const result=dd::Add({q,0.},dd::Divide(residual,b.value.hi));
    // Residual certification bounds the represented quotient against a/b.
    // This is a scalar operation bound, not an observed native/oracle gap.
    auto const defect=Add(a,Neg(Mul({result,0.},b)));
    Real const radius=Upper(2*(Mag(defect.value)+defect.radius)/denominator);
    return {result,radius};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball Sqrt(Ball a) noexcept {
    if(ExactZero(a))return {};
    Real const lower=a.value.hi-std::abs(a.value.lo)-a.radius;
    if(!(lower>0.) || !Valid(a))return Invalid();
    Real const first=std::sqrt(a.value.hi);
    if(!(first>0.)||!std::isfinite(first))return Invalid();
    auto const residue=dd::Add(a.value,dd::Negate(dd::Multiply({first,0.},first)));
    auto const root=dd::Add({first,0.},dd::Divide(residue,2*first));
    Real const positive=root.hi-std::abs(root.lo);
    if(!(positive>0.))return Invalid();
    auto const defect=Add(a,Neg(Mul({root,0.},{root,0.})));
    // |sqrt(a)-r| = |a-r*r|/(sqrt(a)+r) <= |defect|/r.
    return {root,Upper(2*(Mag(defect.value)+defect.radius)/positive)};
}
// Finite pair (base, increment): multiplication retains the delta*delta term.
// A pointer to the face-local failure flag makes every branch fail closed.
struct Value {
    Ball base{},delta{};
    int* bad=nullptr;
    AMREX_GPU_HOST_DEVICE Value() noexcept=default;
    AMREX_GPU_HOST_DEVICE Value(Real a) noexcept:base{{a,0.},0.}{}
    AMREX_GPU_HOST_DEVICE Value(Ball b,Ball d,int* failure) noexcept:base(b),delta(d),bad(failure){}
    AMREX_GPU_HOST_DEVICE Value& operator+=(Value const&);
    AMREX_GPU_HOST_DEVICE Value& operator*=(Value const&);
};
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int* Context(Value a,Value b) noexcept {
    return a.bad?a.bad:b.bad;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value Check(Value a) noexcept {
    if(a.bad && (!Valid(a.base)||!Valid(a.delta)))*a.bad=1;
    return a;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value operator-(Value a) noexcept {
    return {Neg(a.base),Neg(a.delta),a.bad};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value operator+(Value a,Value b) noexcept {
    return Check({Add(a.base,b.base),Add(a.delta,b.delta),Context(a,b)});
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value operator-(Value a,Value b) noexcept {return a+(-b);}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value operator*(Value a,Value b) noexcept {
    return Check({Mul(a.base,b.base),Add(Add(Mul(a.base,b.delta),Mul(a.delta,b.base)),Mul(a.delta,b.delta)),Context(a,b)});
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value operator/(Value a,Value b) noexcept {
    auto const base=Div(a.base,b.base);
    return Check({base,Div(Add(a.delta,Neg(Mul(base,b.delta))),Add(b.base,b.delta)),Context(a,b)});
}
AMREX_GPU_HOST_DEVICE inline Value& Value::operator+=(Value const& b){*this=*this+b;return *this;}
AMREX_GPU_HOST_DEVICE inline Value& Value::operator*=(Value const& b){*this=*this*b;return *this;}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value sqrt(Value a) noexcept {
    auto const base=Sqrt(a.base),trial=Sqrt(Add(a.base,a.delta));
    if(ExactZero(a.delta))return Check({base,{},a.bad});
    return Check({base,Div(a.delta,Add(base,trial)),a.bad});
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int Sign(Ball a) noexcept {
    if(ExactZero(a))return 0;
    if(a.value.hi-std::abs(a.value.lo)-a.radius>0.)return 1;
    if(a.value.hi+std::abs(a.value.lo)+a.radius<0.)return -1;
    return 2;
}
// Constitutive min/absolute value are continuous at their kinks. Their
// finite actions can enclose a crossing; this does NOT change the separate
// fixed-branch SMART comparison/abs contract below. Normalize the center with
// exact TwoSum before ordering so a retained low part decides equal highs.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball AbsBall(Ball a) noexcept {
    if(!Valid(a))return Invalid();
    auto const center=dd::Sum(a.value.hi,a.value.lo);
    bool const negative=center.hi<0.||(center.hi==0.&&center.lo<0.);
    return {negative?dd::Negate(center):center,a.radius};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Ball MinBall(Ball a,Ball b) noexcept {
    if(!Valid(a)||!Valid(b))return Invalid();
    auto const ac=dd::Sum(a.value.hi,a.value.lo),bc=dd::Sum(b.value.hi,b.value.lo);
    bool const first=ac.hi<bc.hi||(ac.hi==bc.hi&&ac.lo<=bc.lo);
    // min is 1-Lipschitz in the maximum norm, including either cap branch.
    return {first?ac:bc,amrex::max(a.radius,b.radius)};
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value ConstitutiveAbs(Value a) noexcept {
    auto const base=AbsBall(a.base);
    if(ExactZero(a.delta))return Check({base,{},a.bad});
    int const s0=Sign(a.base),s1=Sign(Add(a.base,a.delta));
    if(s0==1&&s1==1)return a;
    if(s0==-1&&s1==-1)return -a;
    return Check({base,Add(AbsBall(Add(a.base,a.delta)),Neg(base)),a.bad});
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value ConstitutiveMin(Value a,Value b) noexcept {
    int* bad=Context(a,b);
    auto const gap=Add(a.base,Neg(b.base));
    auto const change=Add(a.delta,Neg(b.delta));
    int const s0=Sign(gap),s1=Sign(Add(gap,change));
    if(s0==-1&&s1==-1)return Check({a.base,a.delta,bad});
    if(s0==1&&s1==1)return Check({b.base,b.delta,bad});
    auto const base=MinBall(a.base,b.base);
    if(ExactZero(a.delta)&&ExactZero(b.delta))return Check({base,{},bad});
    if(s0==-1&&s1==1)return Check({a.base,Add(Neg(gap),b.delta),bad});
    if(s0==1&&s1==-1)return Check({b.base,Add(gap,a.delta),bad});
    // At equality or overlapping operation enclosures do not freeze a branch:
    // evaluate both values and retain an enclosure of the whole finite change.
    auto const trial=MinBall(Add(a.base,a.delta),Add(b.base,b.delta));
    return Check({base,Add(trial,Neg(base)),bad});
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool Compare(Value a,Value b,int comparison) noexcept {
    auto const d=a-b;int const s0=Sign(d.base),s1=Sign(Add(d.base,d.delta));
    bool const uncertain=s0==2||s1==2;
    bool const r0=comparison==0?s0!=0:s0>0;
    bool const r1=comparison==0?s1!=0:s1>0;
    if(d.bad&&(uncertain||r0!=r1))*d.bad=1;
    return r0;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool operator!=(Value a,Value b) noexcept {return Compare(a,b,0);}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool operator>(Value a,Value b) noexcept {return Compare(a,b,1);}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Value abs(Value a) noexcept {
    int const s0=Sign(a.base),s1=Sign(Add(a.base,a.delta));
    if(s0==2||s1==2||s0!=s1){if(a.bad)*a.bad=1;return Check({Invalid(),Invalid(),a.bad});}
    return s0<0?-a:a;
}
}
#endif
