/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_CIRCUIT_FIELD_RATE_H_
#define WARPX_NATIVE_CIRCUIT_FIELD_RATE_H_
#include <AMReX_MultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <string>
#include <cstdint>
#include <vector>
class WarpX;
class CircuitCoupler;
namespace warpx::darwin {
/** Accepted-state native circuit/field response with private field scratch.
 * Prepare samples immutable geometry only once per coupler and prepares the
 * accepted circuit packet outside residual/Jv. Apply reads the trial E_T and
 * evaluates both its homogeneous coil response and the requested affine part.
 * No registered field, circuit storage, scale segment or particle is advanced.
 */
class NativeCircuitFieldRate {
public:
    NativeCircuitFieldRate(WarpX&,CircuitCoupler&);
    bool Prepare(amrex::Real time,std::string& error);
    /** Same accepted-time preparation with a caller-owned frozen response.
     * The selected matrix is copied, never retained or installed as standard H. */
    bool Prepare(amrex::Real time,std::string& error,
                 std::vector<double> const* boundary_response_override);
    void Apply(ablastr::fields::VectorField const& transverse_electric,bool affine);
    [[nodiscard]] bool Ready() const noexcept;
    /** Completed native A derivative for the exact last Apply input. */
    [[nodiscard]] amrex::MultiFab const& PotentialRate(int c) const { AMREX_ALWAYS_ASSERT(Ready()); return m_a[c]; }
    [[nodiscard]] amrex::MultiFab const& MagneticRate(int c) const { return m_b[c]; }
    [[nodiscard]] amrex::MultiFab const& CurrentRate(int c) const { return m_j[c]; }
    /** −A_dot_external, including the E-dependent homogeneous coil rate. */
    [[nodiscard]] amrex::MultiFab const& ElectricReference(int c) const { return m_e_reference[c]; }
private:
    using Field=amrex::Array<amrex::MultiFab,3>;
    WarpX& m_warp;
    CircuitCoupler& m_coupler;
    Field m_zero,m_reference,m_e_reference,m_a,m_b,m_j;
    bool m_ready=false;
    std::uint64_t m_generation=0;
    void BuildReference();
};
}
#endif
