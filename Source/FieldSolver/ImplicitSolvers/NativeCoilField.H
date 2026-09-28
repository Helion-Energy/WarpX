/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_COIL_FIELD_H_
#define WARPX_NATIVE_COIL_FIELD_H_
#include <AMReX_MultiFab.H>
#include <AMReX_GpuContainers.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <array>
class WarpX;
namespace warpx::darwin {
/** Private qualification switch. This does not admit a source/circuit event. */
bool NativeCoilCurrentEnabled();
/** Actual full imposed coil field and its native Ampere current.
 * Every sample takes an explicit physical clock or a prepared device rate.
 * No last-sampled value is used implicitly. All scratch is private and
 * allocated once; residual applications never download circuit vectors.
 * Initial-field exclusion is deliberately inapplicable to the full coil sum.
 */
class NativeCoilField {
public:
    explicit NativeCoilField(WarpX&);
    void SampleAtTime(amrex::Real time);
    void SampleSecant(amrex::Real start_time,amrex::Real duration);
    void SampleRates(amrex::Gpu::DeviceVector<double> const& rates);
    void SubtractCurrent(ablastr::fields::VectorField const& fresh_total) const;
    void CalculatePlasmaCurrent(ablastr::fields::VectorField const& total_B,
                                amrex::Real time);
    [[nodiscard]] amrex::MultiFab const& MagneticField(int c) const;
    [[nodiscard]] amrex::MultiFab const& Current(int c) const;
private:
    enum class Sample { Value, Secant, Rate };
    void Apply(Sample,amrex::Real time,amrex::Real duration,
               amrex::Gpu::DeviceVector<double> const* rates);
    WarpX& m_warp;
    std::array<amrex::MultiFab,3> m_b,m_j;
    bool m_sampled=false;
};
}
#endif
