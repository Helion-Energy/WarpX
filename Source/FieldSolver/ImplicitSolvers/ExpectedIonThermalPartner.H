/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_EXPECTED_ION_THERMAL_PARTNER_H_
#define WARPX_EXPECTED_ION_THERMAL_PARTNER_H_
#include "IonExchangeMomentAudit.H"
namespace warpx::thermal
{
// Conditional population variance about the deterministic post-OU cell mean.
// N is physical-ion weight sum; S=sum w*|u-mean(u)|^2, without a Bessel factor.
// This supplies the full mean random-kick energy. It does not remove the
// finite-sample mean-motion noise a second time. Inputs obey prepared-map
// gates.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
PopulationThermalOUIncrement (amrex::Real number, amrex::Real variance_sum, amrex::Real mass,
                              amrex::Real temperature, amrex::Real nu, amrex::Real dt)
{
    return .5 * (-std::expm1(-2 * nu * dt)) *
           (3 * number * PhysConst::kb * temperature - mass * variance_sum);
}
// Replaces ONLY the electron relaxation partner. Joule/redirect and every
// mechanical channel remain separately named. No accepted particle/RNG writes.
class ExpectedIonThermalPartner
{
  public:
    explicit ExpectedIonThermalPartner (const AcceptedIonExchange&);
    // Both inputs must be from the SAME current endpoint/prepared evaluation.
    // Requires successful NonrelativisticBounded work evaluation, whose
    // per-particle bound controls total NR-vs-relativistic accepted-map work.
    // There is no asserted unique relativistic thermal/bulk decomposition.
    bool Evaluate (const IonExchangeMomentAudit&, const ExpectedIonExchangeSource& bounded_work);
    const amrex::MultiFab& Source () const;      // -thermal/(physical volume*dt), W/m3
    const amrex::MultiFab& ThermalWork () const; // extensive J per cell
    amrex::Real ThermalEnergy () const;          // stateless collective acceptance only
    amrex::Real ElectronEnergy () const;         // measured integral of Source()*V*dt
    bool Valid () const { return m_valid; }

  private:
    amrex::Geometry m_geometry;
    amrex::Real m_dt;
    int m_species;
    amrex::MultiFab m_source, m_thermal_work;
    bool m_valid = false;
};
} // namespace warpx::thermal
#endif
