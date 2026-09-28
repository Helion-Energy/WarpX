/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_REACTION_TRANSPOSE_H_
#define WARPX_NATIVE_STOPPING_REACTION_TRANSPOSE_H_
#include "NativeAcceptedStoppingContext.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>

namespace warpx::thermal {
/** Per physical ion impulse, not acceleration/proper-velocity increment.
 * position and impulse are Cartesian even in RZ; weight is macro multiplicity.
 * Producer owns phase/species/charge provenance; no stochastic draw occurs here.
 */
struct StoppingIonImpulse {
    amrex::GpuArray<amrex::ParticleReal,3> position{};
    amrex::GpuArray<amrex::Real,3> impulse{}; // kg m/s per physical particle
    amrex::Real weight=0.;
};
struct StoppingReactionBalance {
    amrex::GpuArray<amrex::Real,3> particle_cartesian{};
    amrex::GpuArray<amrex::Real,3> electron_cartesian{};
    amrex::GpuArray<amrex::Real,3> unrepresented_cartesian{};
    amrex::GpuArray<amrex::Real,3> conjugate_integral{};
    amrex::Real absolute_particle_impulse=0.;
};
/** Exact real-arithmetic transpose of the accepted stopping Cartesian velocity
 * gather, including its physical images. Floating arithmetic has independently
 * tested signed-sum bounds. No event law, heat, field publication or RNG calls.
 *
 * Pi=V^-1 G^T w dpi [kg/(m^2 s)], se=(qe/me)Pi [A/m^2], b=M se [V s/m].
 * Electron momentum density is -Pi. Outputs are valid physical DOFs, not already
 * constrained endpoint fields; nonphysical ghosts are zero. Internal/periodic
 * ghosts of b follow the native mass map only. Unique owner masks are required
 * for integrals. In RZ m0 only the axial integral is Cartesian net momentum;
 * discarded transverse and wall-image impulse is exposed, not booked as heat.
 *
 * Begin/Finish are collective. DepositTile may be called only serially on host,
 * on this context's MFIter layout, once per disjoint particle subset. The const
 * tuple span must stay alive until GPU completion. This object and context must
 * outlive the transaction; re-Prepare/Invalidate invalidates the transaction.
 * Failed Finish clears all private outputs. No rollback of particles is needed.
 */
class NativeStoppingReactionTranspose {
public:
    explicit NativeStoppingReactionTranspose (NativeAcceptedStoppingContext const&);
    bool Begin ();
    void DepositTile (amrex::MFIter const&, StoppingIonImpulse const* device_tuples,
                      amrex::Long count);
    bool Finish ();
    void Invalidate ();
    bool Valid () const { return m_valid && ContextUnchanged(); }
    char const* Failure () const { return m_failure; }
    ablastr::fields::ConstVectorField ConjugateMomentumDensity () const;
    ablastr::fields::ConstVectorField ElectronCurrentImpulse () const;
    ablastr::fields::ConstVectorField InertiaImpulse () const;
    StoppingReactionBalance const& Balance () const;
private:
    bool ContextUnchanged () const;
    bool Reject (char const*);
    NativeAcceptedStoppingContext const& m_context;
    std::uint64_t m_generation=0;
    std::array<amrex::MultiFab,3> m_extensive, m_pi, m_current, m_inertia;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_owner;
    amrex::Gpu::DeviceVector<amrex::Real> m_totals;
    amrex::Gpu::DeviceVector<int> m_bad;
    StoppingReactionBalance m_balance;
    bool m_open=false, m_valid=false, m_host_bad=false;
    char const* m_failure="unprepared";
};
} // namespace warpx::thermal
#endif
