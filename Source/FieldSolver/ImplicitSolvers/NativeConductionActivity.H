/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_CONDUCTION_ACTIVITY_H_
#define WARPX_NATIVE_CONDUCTION_ACTIVITY_H_
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>
#include <memory>

namespace warpx::thermal {
enum class ConductionCellClosure { Unspecified, AllContributors };
struct ConductionActivityOptions {
    // No implicit choice: callers must explicitly select the conservative FV
    // adaptation. It is an O(h) insulating erosion, not the native nodal graph.
    ConductionCellClosure closure = ConductionCellClosure::Unspecified;
    amrex::Real legacy_density_floor = 1.0; // qdsmc_n_floor [m^-3]
    amrex::Real common_density_floor = 1.0; // common Ohm/thermal n_floor [m^-3]
    bool halo_unfreeze = false;
};
struct ConductionActivityMeasure {
    amrex::Real native_open_volume =
        0; // physical overlap of open nodal duals [m^3]
    amrex::Real retained_open_volume = 0; // full open FV cells [m^3]
    amrex::Real mixed_cell_volume =
        0;                            // full measure of partly open cells [m^3]
    amrex::Real lost_open_volume = 0; // native_open-retained [m^3]
    amrex::Real max_capacity_relative_gap = 0;        // includes inactive nodes
    amrex::Real max_active_capacity_relative_gap = 0; // |common-legacy|/legacy
};

// Pure adapter for one RZ or Cartesian 3D physical level, no EB/AMR. Gate
// native nodes BEFORE any restriction. A pedestal POINTER (even zero valued)
// selects the n>0 gate. Capacity floors are exposed separately and never
// overwrite thermal density.
class NativeConductionActivity {
  public:
    enum Component {
        Open = 0,
        LegacyCapacity,
        CommonCapacity,
        RelativeCapacityGap,
        ActiveRelativeCapacityGap,
        Count
    };
    NativeConductionActivity (const amrex::Geometry&, const amrex::BoxArray&,
                              const amrex::DistributionMapping&,
                              ConductionActivityOptions);
    // Charge and pedestal in C/m^3. Reads valid inputs only, validates before
    // publishing; false leaves all previous outputs unchanged. Negative raw
    // charge may close a node; a negative pedestal is invalid.
    bool Evaluate (const amrex::MultiFab& charge, int charge_component = 0,
                   const amrex::MultiFab* pedestal = nullptr);
    const amrex::MultiFab&
    Nodes () const {
        return m_nodes;
    }
    // Single-component 0/1 mask, ready for
    // ThermalFaceContext.conduction_active.
    const amrex::MultiFab&
    CellOpen () const {
        return m_cell_open;
    }
    const amrex::MultiFab&
    OpenFraction () const {
        return m_fraction;
    }
    // These are DIFFERENT staggerings: native edges connect two nodes;
    // thermal faces connect two FV cells. Both return shared exact 0/1 values.
    const amrex::MultiFab&
    NativeEdgeOpen (int d) const {
        return *m_edges[d];
    }
    const amrex::MultiFab&
    FaceOpen (int d) const {
        return *m_faces[d];
    }
    // Optional global diagnostics; Evaluate itself does not tally accepted
    // heat.
    ConductionActivityMeasure Measure () const;

  private:
    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells;
    amrex::DistributionMapping m_distribution;
    ConductionActivityOptions m_options;
    amrex::MultiFab m_candidate, m_nodes, m_cell_open, m_fraction;
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_edges,
        m_faces;
    bool m_valid = false;
};
} // namespace warpx::thermal
#endif
