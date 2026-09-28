/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_JOULE_WORK_H_
#define WARPX_NATIVE_JOULE_WORK_H_
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>
#include <memory>
#include <array>
class WarpX;
namespace warpx::thermal {
class EulerianDissipation;
// Default off preserves the inherited nodal-current-square law. The opt-in
// physical spatial transfer is bound by NativeEndpointCheckpointConfiguration.
bool NativeJouleWorkEnabled ();
void ValidateNativeJouleWork (WarpX&);

/** Pure preallocated capture of the applied global full-Ohm resistive field.
 * Caller installs the actual Ji/Jp/B, charge and borrowed trial Te/OhmPe and
 * inertia state. We call the native algebraic wrapper with include_eta=true,
 * then restore registered E INCLUDING guards before return. Eeta comes from
 * that SAME kernel expression, not from a separately evaluated parser.
 * Native homogeneous PEC/PMC/axis images and owner/periodic sync are applied
 * to the component only; no affine shell/drive contribution is added.
 *
 * No particle, RNG, accepted thermal, ledger or diagnostic mutation. All
 * field storage is allocated in the constructor, never during Evaluate.
 * Geometric end eta and Holmstrom change have separately captured signed
 * sink accounts; neither is deposited as heat in this opt-in policy. Power
 * scratch uses LAST-evaluation plasma and interval ion currents. Reductions
 * and receipt printing occur only through explicit acceptance methods.
 * Only constant common physical eta is admitted: frozen kinetic dQ/dU=0,
 * while the full residual keeps the actual field/current response live.
 */
class NativeJouleWork {
  public:
    explicit NativeJouleWork (WarpX&);
    bool Evaluate (WarpX&, const amrex::MultiFab& charge, EulerianDissipation*);
    std::array<const amrex::MultiFab*,3> Electric () const;
    // Numerical components are distinct from physical electron heat.
    bool HasEndSinks () const noexcept { return m_end_active; }
    std::array<const amrex::MultiFab*,3> EndResistivity () const;
    std::array<const amrex::MultiFab*,3> EndHolmstrom () const;
    // Reduce cached LAST-evaluation powers only at converged acceptance.
    // Order: signed eta, absolute eta, signed Holmstrom, absolute Holmstrom.
    bool PrepareEndReceipt (amrex::Real dt);
    void CommitEndReceipt ();
    void DiscardEndReceipt () noexcept { m_end_prepared=false; }
    const std::array<amrex::Real,4>& EndReceipt () const { return m_end_energy; }
  private:
    amrex::Geometry m_geometry;
    std::array<amrex::MultiFab,3> m_saved_electric, m_electric;
    std::array<amrex::MultiFab,3> m_end_resistivity, m_end_holmstrom, m_end_power;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_owner;
    std::array<amrex::Real,4> m_end_energy{};
    bool m_end_active=false, m_valid=false, m_end_prepared=false;
};
}
#endif
