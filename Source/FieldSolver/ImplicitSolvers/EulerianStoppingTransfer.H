/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EULERIAN_STOPPING_TRANSFER_H_
#define WARPX_EULERIAN_STOPPING_TRANSFER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <memory>

namespace warpx::thermal {
class KineticThermalMoments;

struct StoppingTransferOptions {
    amrex::Real gamma = 5.0 / 3.0;
    amrex::Real raw_density_floor = 0; // m^-3; model.m_n_floor
    amrex::Real temperature_floor_ev = 0;
    bool embedded_boundary = false;
    int physical_levels = 1;
    int azimuthal_modes = 1;
};

struct StoppingTransferLedger {
    // All values are J. Impulse and raw decline are signed. A positive
    // requested impulse is the measured ion kinetic loss; delta_Ki=-requested.
    amrex::Real requested = 0;
    amrex::Real density_eligible = 0;
    amrex::Real density_declined = 0;
    amrex::Real mapped = 0;
    amrex::Real delivered = 0;
    amrex::Real floor_declined =
        0;                      // nonnegative cooling declined by cell floor
    amrex::Real map_defect = 0; // mapped-density_eligible
    // Actual stored delta_U - requested + density_declined - floor_declined.
    // Includes map/arithmetic roundoff. A nonzero declined account is NOT
    // exact conservation: delta_U + delta_Ki = floor_declined-density_declined.
    amrex::Real exchange_defect = 0;
};

/** Pure accepted-endpoint stopping transfer into primary cell energy.
 * Input impulse is ALREADY CONSOLIDATED signed physical nodal J/m3, not a
 * rate. No dt/theta multiplier. The caller owns copy+SumBoundary exactly once,
 * actual ion collisions, candidate commit, accounting and staging clear.
 * This helper never touches particles, RNG, registry Te/Pe, markers or walls.
 * The collision split remains after the accepted field/thermal advance; this
 * is not a time-centered implicit predictor of stopping inside that solve.
 *
 * Physical source restriction is independent of the native particle density
 * quadrature. map must have the SAME RZ geometry/layout/periodicity; its
 * RestrictNodalScalar call changes only its private scalar scratch. Capacity
 * is the supplied endpoint cell n including any pedestal. Eligibility uses
 * raw rho alone. The primary-cell floor never lifts an already subfloor U.
 *
 * Outputs are owned scratch; consume them only after Evaluate returns true.
 * A failed evaluation invalidates outputs. A/B/A and repeated evaluations
 * have no cumulative side effects. Capability/layout/alias errors assert;
 * nonfinite, inconsistent seam copies beyond floating-point sum roundoff, or
 * unphysical capacity/U return false.
 */
class EulerianStoppingTransfer {
  public:
    EulerianStoppingTransfer (const amrex::Geometry& geometry,
                              const amrex::BoxArray& cells,
                              const amrex::DistributionMapping& distribution,
                              StoppingTransferOptions options);
    bool Evaluate (KineticThermalMoments& map,
                   const amrex::MultiFab& consolidated_nodal_impulse,
                   const amrex::MultiFab& raw_endpoint_charge,
                   const amrex::MultiFab& endpoint_number_capacity,
                   const amrex::MultiFab& before_energy);
    const amrex::MultiFab& CandidateEnergy () const;
    const amrex::MultiFab& CellImpulse () const; // actual stored delta_U
    const amrex::MultiFab& RequestedCellImpulse () const; // before cell floor
    const amrex::MultiFab& NodalEligibleImpulse () const;
    const amrex::MultiFab& NodalDeclinedImpulse () const;
    const amrex::MultiFab& CellFloorDeclinedImpulse () const;
    const StoppingTransferLedger& Ledger () const;

  private:
    void CheckInput (const amrex::MultiFab& field, bool nodal) const;
    bool CopyConsolidated (amrex::MultiFab& output,
                           const amrex::MultiFab& input);
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells, m_nodes;
    amrex::DistributionMapping m_distribution;
    StoppingTransferOptions m_options;
    amrex::MultiFab m_impulse, m_charge, m_eligible, m_declined,
        m_seam_difference;
    amrex::MultiFab m_requested, m_candidate, m_delivered, m_floor_declined;
    std::unique_ptr<amrex::iMultiFab> m_owner;
    StoppingTransferLedger m_ledger;
    bool m_valid = false;
};
} // namespace warpx::thermal
#endif
