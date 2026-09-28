/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_LONGITUDINAL_PRODUCER_CERTIFICATE_H_
#define WARPX_NATIVE_LONGITUDINAL_PRODUCER_CERTIFICATE_H_
#include <ablastr/fields/MultiFabRegister.H>
#include <AMReX_REAL.H>
#include <cstdint>
#include <memory>
class WarpX;
class ThetaImplicitHybrid;
namespace warpx::thermal { class NativeRZSpatialStoppingEvent; class NativeRZMidpointStoppingEvent; }
namespace warpx::implicit { class NativeAcceptedStepCandidate; }
namespace warpx::darwin {
/** Arithmetic provenance of the actual longitudinal field producer.
 * No constructor accepts a tolerance. The private native callers record only
 * represented scalar gradients and spatially constant affine operations.
 * This certificate never changes a physical field or a native solve gate.
 */
class NativeLongitudinalProducerCertificate {
public:
    using ConstVector=ablastr::fields::ConstVectorField;
    using Vector=ablastr::fields::VectorField;
    struct Stats {
        amrex::Real electric_radius=0., displacement_radius=0., curl_bound=0.;
        std::uint64_t operations=0,local_bytes=0;
    };
    ~NativeLongitudinalProducerCertificate();
    NativeLongitudinalProducerCertificate(NativeLongitudinalProducerCertificate const&)=delete;
    bool Valid() const noexcept;
    bool Matches(WarpX&,amrex::Real endpoint_time,std::uint64_t epoch) const;
    bool ValidateCurl(ConstVector const& actual_curl,amrex::Real& maximum_bound) const;
    Stats Statistics() const;
    // Immutable operands for independent accepted-only proof audits.
    ConstVector ElectricRadii() const;
    ConstVector DisplacementRadii() const;
    ConstVector ElectricIncrementRadii() const;
    // Public opaque name is needed by NVCC extended device lambdas.
    struct Impl;
private:
    friend class ::ThetaImplicitHybrid;
    friend class warpx::implicit::NativeAcceptedStepCandidate;
    friend class warpx::thermal::NativeRZSpatialStoppingEvent;
    friend class warpx::thermal::NativeRZMidpointStoppingEvent;
    explicit NativeLongitudinalProducerCertificate(WarpX&);
    static std::shared_ptr<NativeLongitudinalProducerCertificate> Seed(WarpX&);
    bool BindInitialDisplacement(amrex::Real,std::uint64_t);
    std::shared_ptr<NativeLongitudinalProducerCertificate> Fork(amrex::Real,std::uint64_t) const;
    void BeginOuter();
    bool MatchesElectric() const;
    void RawGradient(amrex::MultiFab const&,Vector const&);
    void SchurAddition(amrex::MultiFab const&,ConstVector const&,Vector const&);
    void Relax(Vector const&,amrex::Real);
    void CaptureStage(amrex::Real dt,amrex::Real theta);
    void PairedBeginField();
    void PairedSchurAddition(amrex::MultiFab const&,ConstVector const&,Vector const&,amrex::Real);
    void PairedCaptureStage(amrex::Real dt,amrex::Real theta);
    void PairedExtrapolateElectric(Vector const&,amrex::Real theta);
    void MaterializeElectric();
    void ExtrapolateElectric(Vector const&,amrex::Real theta);
    void RotateDisplacement(amrex::Real theta);
    bool AppendEvent(amrex::MultiFab const& potential,ConstVector const& increment);
    void Invalidate() noexcept;
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::darwin
#endif
