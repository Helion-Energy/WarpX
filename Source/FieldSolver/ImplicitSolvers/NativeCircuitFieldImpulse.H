/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_CIRCUIT_FIELD_IMPULSE_H_
#define WARPX_NATIVE_CIRCUIT_FIELD_IMPULSE_H_
#include "Circuit/CircuitCoupler.H"
#include "NativeCoilField.H"
#include <AMReX_MultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <cstdint>
#include <string>
#include <memory>
#include <utility>
#include <vector>
class WarpX;
namespace warpx::darwin {
/** Retained source-only field/circuit map. Prepare copies the provider's
 * homogeneous impulse response. Apply is a linear device action: actual native
 * boundary images, disk flux, geometric H-Q Schur, and final native curl/current.
 * Registered fields, particles, provider state and scale segments are unchanged.
 * In particular TotalImpulse includes the actual driven tangential trace; the
 * free input alone is not the impulse to gather or use for work.
 */
class NativeCircuitFieldImpulse {
public:
    NativeCircuitFieldImpulse(WarpX&,CircuitCoupler&);
    bool Prepare(amrex::Real,CircuitCoupler::SourceImpulsePhase,std::string& error);
    /** Collective metadata-only validation precedes every write. Input must
     * match the prepared native layout and not overlap any owned allocation,
     * including through a distinct MultiFab alias. Failure is a pure decline. */
    bool TryApply(ablastr::fields::VectorField const& free_impulse,std::string& error);
    void Apply(ablastr::fields::VectorField const& free_impulse);
    [[nodiscard]] bool Ready() const noexcept;
    /** Owned field and coupler outputs still belong to the same application.
     * A later direct coupler action also invalidates this identity. */
    [[nodiscard]] bool Applied() const noexcept;
    [[nodiscard]] std::uint64_t Application() const noexcept { return m_application; }
    [[nodiscard]] bool MatchesSourceContext(WarpX const&,amrex::Real,
        CircuitCoupler::SourceImpulsePhase) const noexcept;
    [[nodiscard]] std::uint64_t Generation() const noexcept { return m_generation; }
    /** Original full coil inventory captured during successful Prepare.
     * Immutable across Apply/Jv and readable through provisional commit. */
    [[nodiscard]] amrex::MultiFab const& InitialCoilMagneticField(int c) const
    { AMREX_ALWAYS_ASSERT(m_initial_ready); return m_initial_coil->MagneticField(c); }
    [[nodiscard]] amrex::MultiFab const& InitialCoilCurrent(int c) const
    { AMREX_ALWAYS_ASSERT(m_initial_ready); return m_initial_coil->Current(c); }
    [[nodiscard]] std::vector<double> const& InitialPortCurrents() const
    { AMREX_ALWAYS_ASSERT(m_initial_ready); return m_initial_port_current; }

    [[nodiscard]] amrex::MultiFab const& PotentialIncrement(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_a[c]; }
    [[nodiscard]] amrex::MultiFab const& MagneticIncrement(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_b[c]; }
    [[nodiscard]] amrex::MultiFab const& TotalCurrentIncrement(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_j[c]; }
    [[nodiscard]] amrex::MultiFab const& TotalImpulse(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_impulse[c]; }
    [[nodiscard]] amrex::MultiFab const& CoilPotentialIncrement(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_reference[c]; }
    [[nodiscard]] amrex::MultiFab const& CoilMagneticIncrement(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_coil_b[c]; }
    [[nodiscard]] amrex::MultiFab const& CoilCurrentIncrement(int c) const { AMREX_ALWAYS_ASSERT(m_applied); return m_coil_j[c]; }
private:
    using Field=amrex::Array<amrex::MultiFab,3>;
    WarpX& m_warp;
    CircuitCoupler& m_coupler;
    Field m_zero,m_reference,m_a,m_b,m_j,m_impulse,m_coil_b,m_coil_j;
    std::unique_ptr<NativeCoilField> m_initial_coil;
    bool m_initial_ready=false;
    amrex::Real m_time=0.;
    CircuitCoupler::SourceImpulsePhase m_phase=CircuitCoupler::SourceImpulsePhase::PreField;
    std::vector<double> m_initial_port_current;
    bool m_ready=false,m_applied=false;
    std::vector<std::pair<std::uintptr_t,std::uintptr_t>> m_owned_storage,m_initial_storage;
    std::uint64_t m_generation=0, m_application=0;
};
}
#endif
