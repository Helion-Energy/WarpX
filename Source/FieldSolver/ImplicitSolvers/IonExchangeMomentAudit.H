/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ION_EXCHANGE_MOMENT_AUDIT_H_
#define WARPX_ION_EXCHANGE_MOMENT_AUDIT_H_
#include "ImplicitIonEndpointTrial.H"
#include <algorithm>
namespace warpx::thermal {
struct IonExchangeMomentComponent {
    enum : int {
        Number,
        WeightSquare,
        MeanR,
        MeanTheta,
        MeanZ,
        VarianceSum,
        DeterministicBulk,
        ThermalRelaxation,
        Redirect,
        FiniteSampleBulkNoise,
        Count
    };
};
// All energies are NONRELATIVISTIC proper-u quadratic moments. This does not
// define relativistic thermodynamic temperature: selected relativistic total
// work/convention bounds remain in ExpectedIonExchangeSource. Two passes avoid
// subtracting a large bulk square from a small thermal variance.
class IonExchangeMomentAudit {
  public:
    explicit IonExchangeMomentAudit (const AcceptedIonExchange&);
    bool Evaluate (const ImplicitIonEndpointTrial&, const AcceptedIonExchange&);
    const amrex::MultiFab& Moments () const;
    bool Valid () const { return m_valid; }
    std::vector<std::array<amrex::Real, IonExchangeMomentComponent::Count>>
    Ledger () const;

  private:
    amrex::MultiFab m_moments;
    bool m_valid = false;
};
// An additional eta-derived uniform bulk shift overlaps the active OU drift.
// Excluded/non-relaxed species do not overlap. No parser equality is assumed.
inline bool
IonExchangeDragOverlaps (bool relaxation, bool excluded,
                         std::string const& species,
                         std::vector<std::string> const& drag_species) {
    return relaxation && !excluded &&
           std::find(drag_species.begin(), drag_species.end(), species) !=
               drag_species.end();
}
} // namespace warpx::thermal
#endif
