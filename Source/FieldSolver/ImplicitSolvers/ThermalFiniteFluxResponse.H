/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_FINITE_FLUX_RESPONSE_H_
#define WARPX_THERMAL_FINITE_FLUX_RESPONSE_H_
#include "ThermalFiniteArithmetic.H"
#include "ThermalStageReceipt.H"
#include <AMReX_Geometry.H>
#include <array>
#include <string>
namespace warpx::thermal {
class EulerianThermalStage;
namespace finite_flux {
// Bounded owned-expression grammar: literal zero, C*n*Te^2, C*Te^2.5,
// and the native outer min(C*Te^2.5,Ccap*n). Nonnegative constants and their
// original product/division order are retained. No arbitrary parser functions,
// time/halo law, fitted coefficients or parser Jv is admitted.
struct MaterialLaw {
    enum Op:int { Constant,Density,Temperature,Product,Quotient,Square,FiveHalves,Minimum };
    enum Family:int { Zero,Quadratic,PowerFiveHalves,CappedPowerFiveHalves };
    int family=Zero;
    static constexpr int capacity=64;
    int size=0;
    amrex::GpuArray<int,capacity> operation{};
    amrex::GpuArray<Real,capacity> constant{};
    bool zero=false;
    AMREX_GPU_HOST_DEVICE Value Evaluate(Value n,Value te) const noexcept {
        Value stack[capacity];int count=0;
        for(int k=0;k<size;++k) {
            switch(operation[k]) {
            case Constant: stack[count++]=Value(constant[k]);break;
            case Density: stack[count++]=n;break;
            case Temperature: stack[count++]=te;break;
            case Square: stack[count-1]=stack[count-1]*stack[count-1];break;
            case FiveHalves: {
                auto const t=stack[count-1];
                stack[count-1]=(t*t)*sqrt(t);break;
            }
            case Minimum: stack[count-2]=ConstitutiveMin(stack[count-2],stack[count-1]);--count;break;
            case Product: stack[count-2]=stack[count-2]*stack[count-1];--count;break;
            case Quotient: stack[count-2]=stack[count-2]/stack[count-1];--count;break;
            }
        }
        return count==1?stack[0]:Check({Invalid(),Invalid(),n.bad});
    }
};
bool ParseMaterial(std::string const&,MaterialLaw&);
}
// Conditional finite constitutive action; it does not admit an application.
// Retained input increments are exact represented pairs, not a statement that
// the upstream physical particle/field action has been resolved.
class ThermalFiniteFluxResponse {
public:
    using MF=amrex::MultiFab;using Real=amrex::Real;
    struct PairView {MF const& high;MF const& low;};
    struct Field {MF high,low,radius;};
    struct Binding {
        std::uint64_t field_base=0,particle_response=0,density_base=0;
        bool operator==(Binding const&) const=default;
    };
    bool Capture(EulerianThermalStage const&,Binding);
    bool Apply(EulerianThermalStage const&,Binding,std::uint64_t publication,
               std::uint64_t sample,PairView energy,PairView density);
    bool Ready(Binding,std::uint64_t publication,std::uint64_t sample) const noexcept;
    void Invalidate() noexcept;
    // Collective validation occurs before writes; only private raw U receives
    // h*D(delta q)-h*delta Q. Current-row correction remains independent.
    bool AddResidual(MF& high,MF& low,Binding,std::uint64_t publication,std::uint64_t sample) const;
    Field const& FluxIncrement(int d) const {return m_dq[d];}
    Field const& ResidualIncrement() const {return m_dr;}
    Field const& SourceIncrement() const {return m_dsource;}
    Field const& SpecificEnergyIncrement() const {return m_de;}
    Field const& TemperatureEVIncrement() const {return m_dte;}
    Field const& ConductivityIncrement() const {return m_dkappa;}
    Field const& MagneticIncrement() const {return m_db;}
    MF const& BaseEnergy() const {return m_u0;}
    MF const& BaseDensity() const {return m_n0;}
    MF const& BaseMagnetic() const {return m_b0;}
    MF const& BaseActivity() const {return m_activity0;}
    MF const& BaseFlux(int d) const {return m_q0[d];}
    MF const& LegCertificate(int d) const {return m_leg[d];}
    ThermalStageReceipt const& Receipt() const {return m_current;}
    // Device-enclosing helpers; public for NVCC, called only after preflight.
    bool CellActions(EulerianThermalStage const&);
    bool FaceActions(EulerianThermalStage const&);
    bool Divergence(EulerianThermalStage const&);
private:
    bool Scope(EulerianThermalStage const&) const;
    bool m_captured=false,m_ready=false;
    Binding m_binding;
    std::uint64_t m_publication=0,m_sample=0;
    ThermalStageReceipt m_base,m_current;
    amrex::Geometry m_geometry;
    finite_flux::MaterialLaw m_parallel,m_perpendicular;
    MF m_u0,m_n0,m_b0,m_kappa0,m_activity0,m_e0,m_te0;
    Field m_ebase,m_tebase,m_kbase;
    Field m_du,m_dn,m_db,m_de,m_dte,m_dkappa,m_dsource,m_dr;
    std::array<MF,AMREX_SPACEDIM> m_q0,m_leg;
    std::array<Field,AMREX_SPACEDIM> m_dq;
};
}
#endif
