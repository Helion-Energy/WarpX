/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_THERMAL_CONDUCTION_PC_H_
#define WARPX_THERMAL_CONDUCTION_PC_H_
#include "EulerianThermalStage.H"
#include "NonlinearSolvers/MHDConductionStencil.H"
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <cstdint>

namespace warpx::thermal {

struct ThermalPCOptions {
    int cycles = 2;
    // The frozen target is nonsymmetric: central advection plus signed
    // compression are handled by a bounded inner FGMRES. Its elliptic inverse
    // alone receives flow-speed diffusion; the physical residual is untouched.
    bool transport = true;
    int transport_inner_max_iterations = 32;
    amrex::Real transport_inner_relative_tolerance = 0.1;
    // Final MHD keeps flow diffusion opt-in; measured central cases agree.
    amrex::Real transport_upwind_stabilization = 0.0;
    bool agglomeration = true;
    bool consolidation = true;
    // AMReX CPU 2D nonregular coarsening uses its axial line smoother (<=32
    // cells/line, inverse dz^2 > inverse dr^2). Caller must respect this
    // backend restriction; GPU runtime semicoarsening is not qualified here.
    bool semicoarsening = false;
    int max_coarsening_level = 30;
    int max_semicoarsening_level = 0;
    int semicoarsening_direction =
        -1; // dimension held fixed; -1 automatic, RZ: r/z, 3D: x/y/z
};

struct ThermalPCDiagnostics {
    std::uint64_t setups = 0;
    std::uint64_t applies = 0;
    std::uint64_t operator_applies = 0;
    std::uint64_t mlmg_solves = 0;
    std::uint64_t inner_iterations = 0;
    std::uint64_t reductions = 0;
    std::uint64_t breakdowns = 0;
    amrex::Real setup_seconds = 0;
    amrex::Real freeze_seconds = 0;
    amrex::Real apply_seconds = 0;
    amrex::Real last_inner_relative_residual = 0;
    amrex::Real minimum_transport_reaction = 1; // min(1+h(gamma-1)div u)
    // -1 elliptic-only; 0 inner converged; 1 bounded iteration limit (normal
    // approximate inverse); 2 breakdown/nonfinite. Never claims SPD or an
    // exact inverse. A breakdown returns zero so the outer solve must handle
    // it.
    int last_inner_status = -1;
};

// Electron-only extraction of resistive_mhd MHDBlockPC's density-weighted
// distributed MLMG path. No ion-fluid/acoustic block or host gather is used.
class ThermalConductionPC {
  public:
    explicit ThermalConductionPC (EulerianThermalStage& stage, int cycles = 2);
    ThermalConductionPC (EulerianThermalStage& stage, ThermalPCOptions options);
    // Exactly once at a Newton base, never during Jv/line search/precond Apply.
    bool Freeze (const amrex::MultiFab& state);
    void Apply (amrex::MultiFab& correction, const amrex::MultiFab& rhs);
    void ApplyOperator (amrex::MultiFab& out,
                        const amrex::MultiFab& correction);
    // Same frozen operator via the unmodified MHD conduction row emitter and
    // the central transport row emitter. No host gather or dense global solve.
    void ApplyRows (amrex::MultiFab& out, const amrex::MultiFab& correction);
    int
    Updates () const {
        return m_updates;
    }

    const ThermalPCDiagnostics&
    Diagnostics () const {
        return m_diagnostics;
    }
    bool
    TransportActive () const {
        return m_transport_active;
    }

  private:
    struct TransportOps;
    void DefineElliptic (std::unique_ptr<amrex::MLABecLaplacian>& op,
                         std::unique_ptr<amrex::MLMG>& solver);
    void FreezeTransport ();
    void ApplyElliptic (amrex::MultiFab& correction,
                        const amrex::MultiFab& rhs);
    void AddTransport (amrex::MultiFab& out, bool rows);
    void LoadSpecific (const amrex::MultiFab& correction);
    EulerianThermalStage& m_stage;
    ThermalPCOptions m_options;
    ThermalPCDiagnostics m_diagnostics;
    bool m_transport_active = false;
    amrex::MultiFab m_a, m_rhs, m_solution, m_scratch, m_density;
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_b;
    // Independently owned snapshots: residual/Jv/context refresh never changes
    // the frozen target or either inverse's density conversion.
    amrex::MultiFab m_compression, m_transport_state, m_inverse_a;
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_velocity,
        m_inverse_b;
    std::unique_ptr<amrex::MLABecLaplacian> m_inverse_operator;
    std::unique_ptr<amrex::MLMG> m_inverse_solver;
    std::unique_ptr<amrex::MLABecLaplacian> m_operator;
    std::unique_ptr<amrex::MLMG> m_solver;
    warpx::mhd_pc::ConductionStencilConstants m_rows;
    int m_updates = 0;
};
} // namespace warpx::thermal
#endif
