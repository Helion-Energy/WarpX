/* Copyright 2026 The WarpX Community. BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_STOPPING_MATERIAL_REACTION_H_
#define WARPX_NATIVE_STOPPING_MATERIAL_REACTION_H_
#include "NativeStoppingMaterialSupport.H"
#include "NativeStoppingReactionTranspose.H"
namespace warpx::thermal {
class NativeStoppingMaterialOwner;
/** Pure material-bound cubic transpose. No accepted-context surrogate and no
 * inverse mass. It returns Pi only; the strict material applies its own P/V
 * mass. Old positive transpose unchanged. Begin/Finish are collective. */
class NativeStoppingMaterialReaction {
public:
    NativeStoppingMaterialReaction(NativeStoppingMaterialSupport&, StoppingMaterialLease,
        amrex::Geometry const&, amrex::BoxArray const&, amrex::DistributionMapping const&,
        AcceptedStoppingOptions const&);
    bool Begin();
    void DepositTile(amrex::MFIter const&, StoppingIonImpulse const*, amrex::Long);
    bool Finish();
    ablastr::fields::ConstVectorField Momentum() const;
    StoppingReactionBalance const& Balance() const;
private:
    // Owner checks FAB addresses before an action, including invalid scratch.
    friend class NativeStoppingMaterialOwner;
    NativeStoppingMaterialSupport& m_material;
    StoppingMaterialLease m_lease;
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells;
    amrex::DistributionMapping m_distribution;
    AcceptedStoppingOptions m_options;
    std::array<amrex::MultiFab,3> m_extensive, m_momentum;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> m_owner;
    amrex::Gpu::DeviceVector<amrex::Real> m_totals;
    amrex::Gpu::DeviceVector<int> m_bad;
    StoppingReactionBalance m_balance;
    bool m_open=false, m_valid=false, m_local_bad=false;
};
}
#endif
