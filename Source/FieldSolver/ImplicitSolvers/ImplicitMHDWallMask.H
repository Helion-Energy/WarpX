/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IMPLICIT_MHD_WALL_MASK_H_
#define WARPX_IMPLICIT_MHD_WALL_MASK_H_

#include "NonlinearSolvers/MHDResistiveStencil.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

/**
 * \brief Stair-step PEC (perfectly conducting) shaped-wall mask for the
 * theta-implicit MHD solver (RZ only).
 *
 * The machine wall is a revolved poloidal polyline r_wall(z) (CSV file,
 * "z, r" rows, single-valued in z; near-vertical sections are expressed
 * with epsilon-offset duplicate z values, as in the run32 embedded-
 * boundary CSVs). Everything on or outside the polyline (r >= r_wall(z),
 * the run32 "recessed EB" convention: the conductor surface sits AT the
 * polyline) is treated as solid perfect conductor:
 *
 * - Every electric-field component located in the masked region is a
 *   tangential-E degree of freedom of the stair-step conductor surface or
 *   an interior-metal value (by the Yee staggering, an E component that
 *   lies exactly ON a stair face is always tangential to it), and is
 *   projected to the conductor condition at the END of every Ohm's-law
 *   assembly, i.e. inside every residual evaluation. TWO conductor
 *   contracts are provided:
 *
 *   * wall_model = pec: the condition acts on the TOTAL field. Under the
 *     split-field external-A scheme the evolved fields are the plasma
 *     response, so the projection writes E_plasma = -E_external (total
 *     tangential E = 0); without external fields it writes zero. Cells
 *     whose complete edge set is masked keep their TOTAL field frozen at
 *     its initial value (dB_plasma/dt = +curl E_ext = -dB_ext/dt exactly,
 *     discrete-curl identical): the physically honest perfect-conductor
 *     eddy response, which also SHIELDS programmed coil ramps whose
 *     filaments sit on/outside the polyline. Use with RAW coil waveforms.
 *
 *   * wall_model = pec_response: the condition acts on the PLASMA-RESPONSE
 *     field only (E_plasma = 0 at masked locations; the external drive
 *     passes through the wall untouched). This is the run32 embedded-
 *     boundary parity contract for FITTED waveform composites that
 *     already embed the real machine's wall response (one-way coupling):
 *     plasma-generated flux cannot leak through the wall (the plasma
 *     flux through every contour inside the metal is conserved), while
 *     the prescribed drive acts everywhere as measured. Requires the
 *     split external fields (hybrid_pic_model.add_external_fields);
 *     without them the mode is meaningless and Define aborts.
 *
 *   Both projections are affine in the state with the same sparsity as a
 *   zeroed row, so matrix-free Jacobian probes see the wall exactly
 *   (JFNK-exact), and both share the identical preconditioner treatment
 *   (the masked E values are state-independent constants either way).
 *
 * - The fluid is deliberately untouched by default: the wall region is
 *   expected to sit in the dust/vacuum fill (frozen momentum,
 *   E -> eta J). The optional THERMAL wall boundary (wall_thermal_bc)
 *   upgrades the band to an actual rigid conductor: it closes the
 *   spurious heat channels the default leaves open (the conduction
 *   operator exchanging blindly across the stair-step interface, and
 *   the cell-centered Joule heating eta J^2 of the PEC surface current
 *   meeting the -- possibly anomalous -- plasma resistivity at the
 *   near-floor wall-band density, an unbounded heat source inside the
 *   metal):
 *
 *   * In BOTH modes every fluid increment inside a masked cell is
 *     zero: the metal is a rigid absorbing body, not fluid. The band
 *     keeps its bounded load state forever (no Joule pile-up, no
 *     drained-heat accumulation, no runaway signal speeds feeding the
 *     interface fans), and the stair interface faces drain the
 *     INTERIOR one-sidedly -- what crosses is gone (mass, momentum,
 *     enthalpy), the fluid analog of run32's wall particle scraper.
 *     Faces between two masked cells carry no conduction (interior
 *     metal).
 *   * wall_thermal_bc = zero_flux: additionally, the interface faces
 *     carry exactly zero conductive flux (adiabatic wall).
 *   * wall_thermal_bc = temperature: the interface faces DRAIN
 *     conductively toward a T_wall reservoir (wall_temperature, eV).
 *     The drain is one-sided (C^1 gated off at/below the wall value:
 *     the reservoir cools the interior toward T_wall, never heats it)
 *     and always free-streaming limited at the wall face -- a
 *     two-sided, un-capped exchange heats the machine's entire
 *     near-floor wall rim on the sub-step cell-diffusion time and was
 *     measured fatal on the formation ladder (rr12_S0w).
 *
 *   The default none keeps wall_model bit-identical to the
 *   electromagnetic-only wall.
 *
 * The preconditioner sees the identical mask: the resistive stencil
 * emission (MHDResistiveStencil.H) drops the same masked E rows through
 * the WallMaskView, so the matrix-free Chebyshev application and the
 * banded/direct assemblies stay consistent with the residual.
 *
 * The wall polyline must stay clear of the axis (first masked radial
 * index >= 1 everywhere): the r = 0 reflecting boundary is handled by
 * the m = 0 parity ghosts and the axis-corner EMF parities, which
 * assume plasma at the axis; a polyline pinching to r = 0 would fight
 * those fills, so Define aborts loudly instead.
 *
 * Inputs (all under implicit_mhd.):
 * - wall_model = none (default; bit-identical to no wall) | pec |
 *   pec_response
 * - wall_polyline_file = <csv path> ("z, r" rows; resolved against
 *   AMREX_INPUTS_FILE_PREFIX like the inputs file itself when relative)
 * - wall_thermal_bc = none (default) | zero_flux | temperature
 * - wall_temperature = <eV> (temperature mode only)
 */
class ImplicitMHDWallMask
{
public:
    /** Thermal wall boundary at the stair-step fluid interface
     * (implicit_mhd.wall_thermal_bc); see the class comment. */
    enum class ThermalBC { none, zero_flux, temperature };

    /** Parse the wall inputs and, when active, build the static
     * first-masked-radial-index tables per electric-field staggering over
     * the ghost-extended axial range. Aborts outside RZ, on periodic z,
     * on a malformed polyline, or on a polyline that reaches the axis. */
    void Define (const amrex::Geometry& geom, const amrex::IntVect& ngrow);

    [[nodiscard]] bool IsActive () const { return m_active; }

    [[nodiscard]] ThermalBC GetThermalBC () const
    {
        return m_active ? m_thermal_bc : ThermalBC::none;
    }

    /** Human-readable thermal mode for banners. */
    [[nodiscard]] const char* ThermalBCName () const;

    /** Wall reservoir temperature [eV] (temperature mode). */
    [[nodiscard]] amrex::Real WallTemperature_eV () const
    {
        return m_wall_temperature;
    }

    /** Cell-centered (r-cell, z-cell) first-masked-radial-index table,
     * pre-offset like the View() pointers (valid over the z-cell range
     * [-GhostCells(), AxialCells() - 1 + GhostCells()]); consumers must
     * clamp their axial index to that range (constant continuation,
     * like the polyline itself). Non-null only when a thermal wall BC
     * is active. */
    [[nodiscard]] const int* FirstMaskedCellCentered () const;

    [[nodiscard]] int GhostCells () const { return m_ng; }
    [[nodiscard]] int AxialCells () const { return m_nz; }

    /** Human-readable mode for banners: "none", "pec" or "pec_response". */
    [[nodiscard]] const char* ModeName () const
    {
        return m_active ? (m_total_field ? "pec" : "pec_response") : "none";
    }

    /** Device view consumed by the preconditioner stencil emission; the
     * pointers are pre-offset so the axial index of the residual grid is
     * used directly (valid over [-ng, nz+ng] nodal, [-ng, nz-1+ng] cell).
     * Mode-independent: the masked E values are state-independent
     * constants in both conductor contracts, so the Jacobian row drops
     * are identical. */
    [[nodiscard]] warpx::mhd_pc::WallMaskView View () const;

    /** Apply the conductor projection to the assembled electric field.
     * wall_model = pec (total-field contract): at every masked location
     * set E = -E_external (when the split external fields are present)
     * or E = 0. wall_model = pec_response (plasma-response contract):
     * set E = 0 (the external drive is transparent). Covers the domain
     * ghosts (the external registers carry trusted domain-ghost values). */
    void ProjectElectricField (ablastr::fields::VectorField const& efield,
                               ablastr::fields::VectorField const* efield_external,
                               const amrex::Geometry& geom) const;

private:
    bool m_active = false;
    /** true: pec (conductor condition on the TOTAL field); false:
     * pec_response (plasma-response field only, run32 EB parity). */
    bool m_total_field = true;
    ThermalBC m_thermal_bc = ThermalBC::none;
    amrex::Real m_wall_temperature = 0.0; //!< [eV], temperature mode
    int m_ng = 0;   //!< axial ghost extension of the index tables
    int m_nz = 0;   //!< axial cells

    /** First masked radial index per axial index (sentinel: never masked),
     * one table per electric-field staggering; storage index j + m_ng.
     * MANAGED memory: the preconditioner's stencil emission runs on the
     * device (Chebyshev, banded) AND on the host (the direct solver's
     * sparse-row assembly), and both read these tables through the same
     * WallMaskView pointers. */
    amrex::Gpu::ManagedVector<int> m_first_masked_er; //!< (r-cell, z-node)
    amrex::Gpu::ManagedVector<int> m_first_masked_et; //!< (r-node, z-node)
    amrex::Gpu::ManagedVector<int> m_first_masked_ez; //!< (r-node, z-cell)
    amrex::Gpu::ManagedVector<int> m_first_masked_cc; //!< (r-cell, z-cell)
};

#endif // WARPX_IMPLICIT_MHD_WALL_MASK_H_
