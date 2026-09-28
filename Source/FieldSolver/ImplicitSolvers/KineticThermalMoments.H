/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_KINETIC_THERMAL_MOMENTS_H_
#define WARPX_KINETIC_THERMAL_MOMENTS_H_

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <array>
#include <memory>

namespace warpx::thermal {

// Pressure and post-deposition current images are separate. PEC current
// permits normal charge exchange; PMC current reflects it, including the
// particle-reflecting override at a PEC wall. RZ current images preserve r*J.
enum class MomentBoundary { Unspecified, Periodic, Axis, PEC, PMC };
enum class ThermalCurrentInput { SignedElectron, TotalPlasma };
using YeeCurrentView = std::array<const amrex::MultiFab*, 3>;
using ThermalFaceView = std::array<amrex::MultiFab*, AMREX_SPACEDIM>;

struct ThermalMomentOptions {
    bool retain_current_remainder=false; // optional arithmetic only; OFF allocates/communicates nothing
    amrex::Real number_density_floor = 1.0; // m^-3, nodal Ohm/capacity floor
    amrex::Real active_density_floor = 1.0; // m^-3, raw physical density gate
    amrex::Real gamma = 5.0 / 3.0;
    amrex::Real temperature_floor_kelvin = 0.0; // admissibility; never clips U
    bool halo_unfreeze = false;
    // Pass the actual runtime flag: native deposited moments use the matching
    // axis3 (true) or axis4 (false) particle-continuity quadrature.
    bool verboncoeur_axis_correction = true;
    int azimuthal_modes = 1;
    int nodal_ghosts =
        1; // fully filled trial Te/Pe; match consuming solver width
    // Physical pressure images, distinct from particle/current reflection.
    std::array<std::array<MomentBoundary, 2>, AMREX_SPACEDIM> boundary{};
    // Unspecified inherits the pressure image. Pass PMC explicitly for a
    // reflecting-particle override on a PEC field boundary.
    std::array<std::array<MomentBoundary, 2>, AMREX_SPACEDIM>
        current_boundary{};
};

struct KineticThermalStateView {
    // Positive ion charge density in C/m^3, after deposition/MM projection,
    // synchronization and physical deposition images. No mass-density input.
    const amrex::MultiFab& charge;
    int charge_component = 0;
    const amrex::MultiFab* pedestal = nullptr; // nodal C/m^3; nonnegative
    const amrex::MultiFab* energy = nullptr;   // cell U_e in J/m^3, optional
    YeeCurrentView ion_current{}; // native Yee A/m^2, optional as a whole
    YeeCurrentView current{}; // signed Je or total plasma Jp, NOT nodal history
    ThermalCurrentInput current_input = ThermalCurrentInput::SignedElectron;
    YeeCurrentView current_remainder{}; // optional companion to current, all three or none
};

// One full physical RZ m=0 or Cartesian 3D mesh, without EB/AMR. Owns only
// geometry and derived scratch, never accepted state or a frozen trial density.
// Use distinct instances for old, stage and true particle endpoint contexts.
// Evaluation reads valid inputs only: private copies are synced and ghosted.
// The outputs remain valid until the next Evaluate on THIS instance.
class KineticThermalMoments {
  public:
    KineticThermalMoments (const amrex::Geometry& geometry,
                           const amrex::BoxArray& cells,
                           const amrex::DistributionMapping& distribution,
                           ThermalMomentOptions options);

    // false marks invalid/nonfinite density/pedestal/U/current or temperature
    // below the requested floor. Outputs are scratch on failure; no input,
    // energy, particle, RNG, diagnostic or accepted-state mutation occurs.
    bool Evaluate (const KineticThermalStateView& state);

    // Physical-volume LINEAR scalar restriction, preserving input units and
    // constant fields. Use for nodal work/source rates, not particle density.
    void RestrictNodalScalar (const amrex::MultiFab& input, int component,
                              amrex::MultiFab& cell_output);
    // Particle deposition quadrature -> physical cell volume. Unlike the
    // physical source map, native moments rescale axis/wall extensive measures.
    // D_cc RestrictCurrent = RestrictNativeMoment D_particle, with the actual
    // configured axis3/4 divergence. Use for rho and initial nodal n*T. Native
    // quadrature conversion need not preserve constant nodal density at axis
    // or wall; mapping n and n*T together preserves constant temperature.
    void RestrictNativeMoment (const amrex::MultiFab& input, int component,
                               amrex::MultiFab& cell_output);
    void RestrictCurrent (const YeeCurrentView& input,
                          const ThermalFaceView& face_output);

    // Compensated linear increment maps. These use the same native metric,
    // slot coefficients and image kernels as the ordinary maps. Both parts
    // are caller-owned, independent outputs; no high base is subtracted.
    bool RestrictNativeMomentIncrement(const amrex::MultiFab& high,
        const amrex::MultiFab& low,amrex::MultiFab& cell_high,amrex::MultiFab& cell_low);
    bool RestrictCurrentIncrement(YeeCurrentView const& high,YeeCurrentView const& low,
        ThermalFaceView const& face_high,ThermalFaceView const& face_low);
    // CUDA enclosing implementation; inputs are preflighted by the public adapters.
    bool RestrictCurrentRemainder(KineticThermalStateView const&,ThermalFaceView const&,ThermalFaceView const&,bool publish_pair=false);
    // Companion of an already computed ordinary RestrictCurrent result.
    // All face outputs must have independent storage; inputs read valid only.
    bool RestrictCurrentPair(YeeCurrentView const& input,YeeCurrentView const& low,
        ThermalFaceView const& represented,ThermalFaceView const& output);
    const amrex::MultiFab&
    RawNumberDensity () const {
        return m_raw_density;
    }
    // R_rho max((rho + rho_ped)/q_e, n_floor), not max(R_rho rho/q_e,n_floor).
    const amrex::MultiFab&
    NumberDensity () const {
        return m_density;
    }
    const amrex::MultiFab&
    NodalNumberDensity () const {
        return m_node_density;
    }
    const amrex::MultiFab&
    ActiveCells () const {
        return m_active;
    }
    const amrex::MultiFab&
    FaceNumberDensity (int d) const {
        return *m_face_density[d];
    }
    const amrex::MultiFab&
    FaceIonCurrent (int d) const {
        return *m_face_ion[d];
    }
    const amrex::MultiFab&
    FaceElectronCurrent (int d) const {
        return *m_face_electron[d];
    }
    const amrex::MultiFab& FaceElectronCurrentRemainder(int d) const {
        AMREX_ALWAYS_ASSERT(m_has_current_remainder); return *m_face_remainder[d];
    }
    bool HasCurrentRemainder() const noexcept { return m_has_current_remainder; }
    // u_e = -Je_face/(q_e n_face); positive Je is negative electron flow.
    // One common normal velocity, shared by central flux and compression.
    const amrex::MultiFab&
    FaceElectronVelocity (int d) const {
        return *m_face_velocity[d];
    }
    const amrex::MultiFab&
    CellTemperature () const {
        return m_temperature;
    } // K
    const amrex::MultiFab&
    NodalTemperature () const {
        return m_node_temperature;
    } // K
    // Thermodynamic product n_node kB T_node, including PEC boundary nodes.
    const amrex::MultiFab&
    NodalPressure () const {
        return m_pressure;
    } // Pa
    // Actual field closure: PEC boundary nodes copy the adjacent interior
    // pressure, and scalar ghosts are even at axis/PEC/PMC. Kept separate from
    // the thermodynamic product because the PEC constraint overrides it.
    const amrex::MultiFab&
    OhmPressure () const {
        return m_ohm_pressure;
    }
    bool
    HasPressure () const {
        return m_has_pressure;
    }
    bool
    HasCurrents () const {
        return m_has_currents;
    }

  private:
    friend class ThermalMassMatrixResponse;
    void CheckLayout (const amrex::MultiFab& field, const amrex::IntVect& type,
                      int component) const;
    void CopyNodal (const amrex::MultiFab& input, int component,
                    amrex::MultiFab& output) const;
    void FillScalarImages (amrex::MultiFab& data) const;
    void FillCurrentImages (amrex::MultiFab& data,
                            int physical_component) const;
    void ApplyPressureImages ();
    void ApplyPressureIncrementImages(amrex::MultiFab& high,amrex::MultiFab& low,
        amrex::MultiFab& output_high,amrex::MultiFab& output_low);
    bool ValidateCurrentRemainder(KineticThermalStateView const&) const;


    amrex::Geometry m_geometry;
    amrex::BoxArray m_cells;
    amrex::DistributionMapping m_distribution;
    ThermalMomentOptions m_options;
    amrex::MultiFab m_scalar, m_charge, m_pedestal, m_node_density;
    amrex::MultiFab m_raw_density, m_density, m_active, m_temperature;
    amrex::MultiFab m_node_temperature, m_pressure, m_ohm_pressure;
    std::array<std::unique_ptr<amrex::MultiFab>, 3> m_current;
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> m_face_density,
        m_face_ion, m_face_electron, m_face_velocity;
    bool m_has_pressure = false;
    bool m_has_currents = false,m_has_current_remainder=false;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_current_pair,m_current_remainder;
    std::array<std::unique_ptr<amrex::MultiFab>,AMREX_SPACEDIM> m_face_remainder;
};

} // namespace warpx::thermal
#endif
