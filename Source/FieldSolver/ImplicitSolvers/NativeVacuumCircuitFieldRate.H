/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#ifndef WARPX_NATIVE_VACUUM_CIRCUIT_FIELD_RATE_H_
#define WARPX_NATIVE_VACUUM_CIRCUIT_FIELD_RATE_H_
#include "DarwinVacuumAffineResponse.H"
#include "NativeCircuitFieldRate.H"
#include <memory>
#include <string>
#include <vector>
class WarpX;
class CircuitCoupler;
namespace warpx::darwin {
/** Low-rank elimination of the vacuum field's native circuit feedback.
 * Prepare freezes geometry and density support outside residual/Jv. Apply
 * receives the base vacuum completion with ZERO external reference, including
 * its chosen null baseline for affine actions. It returns the completed field,
 * its dynamic electric wall reference, and native Bdot/Cdot. No registry,
 * particle, plugin state, or circuit history is advanced.
 */
class NativeVacuumCircuitFieldRate {
public:
    using View=ablastr::fields::VectorField;
    using Mask=DarwinVacuumAffineResponse::Mask;
    using Options=DarwinVacuumAffineResponse::Options;
    NativeVacuumCircuitFieldRate(WarpX&,CircuitCoupler&);
    bool Prepare(amrex::Real time,Mask const& correction,Mask const& trace,
                 Options const&,std::string& error);
    void Apply(View const& base_electric,bool affine);
    [[nodiscard]] bool Ready() const noexcept;
    [[nodiscard]] amrex::MultiFab const& ElectricField(int c) const { return m_e[c]; }
    [[nodiscard]] amrex::MultiFab const& ElectricReference(int c) const
    { return m_native.ElectricReference(c); }
    [[nodiscard]] amrex::MultiFab const& MagneticRate(int c) const { return m_b[c]; }
    [[nodiscard]] amrex::MultiFab const& CurrentRate(int c) const { return m_j[c]; }
    [[nodiscard]] std::vector<double> const& BoundaryResponse() const { return m_h; }
    [[nodiscard]] amrex::MultiFab const& UnitElectricResponse(int field,int component) const
    { return (*m_columns.at(field))[component]; }
private:
    using Field=amrex::Array<amrex::MultiFab,3>;
    WarpX& m_warp;
    CircuitCoupler& m_coupler;
    NativeCircuitFieldRate m_native;
    Field m_zero,m_e,m_reference,m_e_reference,m_a,m_b,m_j;
    std::vector<std::unique_ptr<Field>> m_columns;
    std::vector<double> m_h;
    bool m_ready=false;
};
}
#endif
