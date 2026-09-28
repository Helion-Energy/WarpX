/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_THERMAL_MASS_MATRIX_RESPONSE_H_
#define WARPX_THERMAL_MASS_MATRIX_RESPONSE_H_
#include "KineticThermalMoments.H"
#include "ThermalCurrentRemainder.H"
#include <AMReX_iMultiFab.H>
#include <cstdint>
class MassMatrixDensityProjection;
namespace warpx::thermal {
class EulerianThermalStage;
class ThermalFiniteFluxResponse;
/** Increment-coordinate thermal consumer of one frozen native MM response.
 * Ordinary field, density, thermal residual and acceptance are never changed.
 * Every output is private device storage; a copied PLUS survives the next map.
 */
class ThermalMassMatrixResponse {
public:
    using MF=amrex::MultiFab;using Real=amrex::Real;
    using ConstFaces=std::array<MF const*,AMREX_SPACEDIM>;
    struct PairField {MF high,low;};
    struct Binding {std::uint64_t field_base=0,particle_response=0,density_base=0;};
    struct Base {
        Binding binding;
        MF const& rho;MF const& energy;MF const& raw;MF const& correction;
        YeeCurrentView ion,plasma;
        std::array<amrex::iMultiFab const*,3> vacuum;
        ConstFaces gamma;
        KineticThermalMoments const& moments;
        EulerianThermalStage const& stage;
        MassMatrixDensityProjection const& projection;
        // Only the vacuum row uses the retained plasma-current low part.
        // Gamma keeps the ordinary published-high current convention.
        YeeCurrentView plasma_low{};
        YeeCurrentView vacuum_current{};
    };
    explicit ThermalMassMatrixResponse(KineticThermalMoments const&,bool finite_conduction=false);
    ~ThermalMassMatrixResponse();
    bool Capture(Base const&);
    bool Apply(Binding,std::uint64_t publication,std::uint64_t sample,
        MassMatrixDensityProjection const&,MF const& energy,YeeCurrentView const& plasma,
        EulerianThermalStage const&,YeeCurrentView const& plasma_low={});
    bool CopyCompanion(MF& output,MF const& represented,Binding,
        std::uint64_t publication,std::uint64_t sample) const;
    void Invalidate() noexcept;
    bool Ready() const noexcept;
    bool CanSample(Binding,std::uint64_t sample) const;
    // Read-only represented-operand certificates; readiness/epoch remains
    // required at the consuming copy interface, not inferred from allocation.
    PairField const& NodeDensityIncrement() const {return m_dn_node;}
    PairField const& DensityIncrement() const {return m_dn;}
    PairField const& EnergyIncrement() const {return m_du;}
    PairField const& TemperatureIncrement() const {return m_dt;}
    PairField const& PressureIncrement() const {return m_dp;}
    PairField const& OhmPressureIncrement() const {return m_dp_ohm;}
    PairField const& NumberFluxIncrement(int d) const {return m_dgamma[d];}
    PairField const& FaceDensityIncrement(int d) const {return m_dnf[d];}
    PairField const& BaseVelocity(int d) const {return m_v0[d];}
    PairField const& VelocityIncrement(int d) const {return m_dv[d];}
    PairField const& FluxIncrement(int d) const {return m_df[d];}
    PairField const& RawIncrement() const {return m_dru;}
    PairField const& CorrectionIncrement() const {return m_dcorrection;}
    // Public CUDA-enclosing methods; Apply supplies validated private data.
    bool DensityAndEOS(MF const& energy);
    bool FaceActions();
    bool ResidualActions();
private:
    bool SameBinding(Binding) const;
    bool ContextSupported(EulerianThermalStage const&) const;
    bool Inputs(MF const&,YeeCurrentView const&,YeeCurrentView const&) const;
    std::unique_ptr<ThermalFiniteFluxResponse> m_finite_flux;
    bool m_finite_selector_valid=true;
    KineticThermalMoments m_map;
    Binding m_binding;
    MassMatrixDensityProjection const* m_projection=nullptr;
    bool m_captured=false,m_ready=false,m_retained_current=false;
    std::uint64_t m_publication=0,m_sample=0,m_generation=0;
    Real m_h=0.,m_gm=0.;
    MF m_rho0,m_node0,m_n0,m_u0,m_raw0,m_correction0,m_delta_rho;
    std::array<MF,3> m_ji0,m_jp0,m_jp0_low,m_delta_ji;
    std::array<amrex::iMultiFab,3> m_vacuum;
    std::array<MF,AMREX_SPACEDIM> m_gamma0,m_nf0,m_jv_face0;
    std::array<PairField,3> m_dj,m_djv;
    std::array<PairField,AMREX_SPACEDIM> m_v0,m_w0,m_dgamma,m_djv_face,m_dnf,m_dv,m_dw,m_df,m_dtransform_flux;
    PairField m_dn_node,m_dn,m_du,m_dt,m_dp,m_dp_ohm,m_dru,m_dcorrection;
};
namespace mm_thermal_detail {
using namespace remainder;
struct CurrentIncrements { Pair published, vacuum; };
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
CurrentIncrements CurrentIncrement(Pair plasma,Pair base,Real delta_ion) noexcept {
    auto const published=Add(Sum(plasma.hi,-base.hi),{-delta_ion,0.});
    return {published,Add(published,Sum(plasma.lo,-base.lo))};
}
// Both numerator and denominator retain their represented low part. This
// quotient is used only after the complete finite branch/positive-n preflight.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair DividePair(Pair a,Pair b) noexcept {
    Real const q=a.hi/b.hi;
    auto rem=Add(a,Negate(Multiply(b,q)));
    return Add({q,0.},Divide(rem,b.hi));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair QuotientIncrement(Pair dn,Pair df,Pair base,Real n0) noexcept {
    return DividePair(Add(df,Negate(Product(base,dn))),Add({n0,0.},dn));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair FluxIncrement(Pair du,Pair dv,Pair u0,Pair v0) noexcept {
    return Add(Add(Product(u0,dv),Product(du,v0)),Product(du,dv));
}
}
}
#endif
