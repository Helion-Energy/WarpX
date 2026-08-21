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

#include <AMReX_Array.H>
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
 *   * wall_model = dielectric: the formation-tube standoff. The shaped
 *     wall between the plasma and the drive coils in the real machine
 *     is a quartz-class DIELECTRIC tube: it absorbs plasma and holds a
 *     temperature boundary, but it is electromagnetically TRANSPARENT
 *     (no eddy screening, no response-field pinning; represent the
 *     machine's actual metal structures separately, e.g. as
 *     circuit-side eddy rings). The FLUID contract is bit-identical to
 *     pec_response -- the same mask tables, rigid freeze, no-injection
 *     absorb image, thermal BC/drains/ledger, pedestal exclusion and
 *     axis guard, and the same split-field-drive requirement; an
 *     active wall_thermal_bc is REQUIRED (the standoff's entire wall
 *     action is the fluid contract, so without it the mode would be a
 *     silent no-op) -- while
 *     the FIELD side imposes NO constraint on the plasma-response
 *     field: ProjectElectricField performs nothing, the preconditioner
 *     drops no rows (View() reports an inactive mask), and the
 *     wall-seam guard is inactive (nothing to guard: the Ohm stencils
 *     see a continuous live field -- the seam-contamination mechanism
 *     was specific to the pinned-response contract's drive-scale
 *     surface current at the stair seam). The band's frozen
 *     floor-density dust plus the vacuum-keyed resistivity already
 *     make the region current-free in the response solve: the
 *     physically correct insulator interior. The semantics follow
 *     WarpX's mixed PEC/insulator domain boundary
 *     (Source/BoundaryConditions/PEC_Insulator.{H,cpp}): on the
 *     insulator side, tangential fields are left UNCHANGED unless a
 *     value is prescribed and the normal B stays evolved -- nothing is
 *     prescribed here, so the response field lives everywhere.
 *     pec_response is one-way glass by comparison (the drive passes,
 *     but the pinned response makes the plasma flux linkage of every
 *     in-band coil contour exactly zero, killing two-way circuit
 *     coupling); the dielectric mode restores the finite, physical
 *     response linkage at coil radii inside the (former) metal band.
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
 * SEAM GUARD on the non-ideal Ohm edge terms: the Hall EMF, the
 * electron-inertia field contribution and the hyper-resistive term of
 * the Ohm assembly read currents and cell-centered fields ACROSS the
 * stair seam -- at live rows adjacent to the masked band those stencils
 * ingest the conducting wall's SURFACE current (curl B across the seam
 * is drive-scale, not plasma) and masked-band field values, divided by
 * the Ohm-floored near-wall density: a spurious, drive-powered EMF pump
 * at the seam (measured as the global E_theta maximum at the first live
 * corner row inside a formation-section step corner, growing with the
 * drive). The eta J term at the same rows is protected by the
 * vacuum-keyed resistivity; the reactive terms have no such key.
 * Physically, electron-frame boundary-layer physics at a rigid conductor
 * is below grid resolution, so the ideal + resistive Ohm law is the
 * correct wall-adjacent contract (the flux kernels' donor-redirection
 * discipline, applied to the Ohm edge terms). Define therefore also
 * builds per-family FIRST-GUARDED radial index tables marking every E
 * row whose Hall/inertia/hyper stencil footprint (radial reach two
 * cells outward -- the widest chain: the nodal inertia advection and the
 * hyper Laplacian-of-current both reach two cells -- and the matching
 * axial windows: cells [j-2, j+1] for the z-nodal E_r/E_theta rows,
 * [j-2, j+2] for the z-cell E_z rows) touches a masked cell-centered
 * cell; the assembly zeroes the three terms at those rows and the
 * preconditioner's stencil emission drops the same contributions through
 * the same tables (residual/PC exact twins). The tables are
 * geometry-static (computed once here), so JFNK probes see constant
 * structure; first_guarded <= first_masked always, and with
 * wall_model = none nothing changes anywhere.
 *
 * The wall polyline must stay clear of the axis (first masked radial
 * index >= 1 everywhere): the r = 0 reflecting boundary is handled by
 * the m = 0 parity ghosts and the axis-corner EMF parities, which
 * assume plasma at the axis; a polyline pinching to r = 0 would fight
 * those fills, so Define aborts loudly instead.
 *
 * Inputs (all under implicit_mhd.):
 * - wall_model = none (default; bit-identical to no wall) | pec |
 *   pec_response | dielectric
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

    /** Dielectric standoff (wall_model = dielectric): the fluid contract
     * of pec_response with an EM-transparent field side; see the class
     * comment. Gates the field-only machinery (projection, PC row
     * drops, seam guard, seam banner) off. */
    [[nodiscard]] bool IsDielectric () const
    {
        return m_active && m_dielectric;
    }

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

    /** Human-readable mode for banners: "none", "pec", "pec_response"
     * or "dielectric". */
    [[nodiscard]] const char* ModeName () const
    {
        if (!m_active) { return "none"; }
        if (m_dielectric) { return "dielectric"; }
        return m_total_field ? "pec" : "pec_response";
    }

    /** Device view consumed by the preconditioner stencil emission AND
     * by the residual's seam guard on the Hall/inertia/hyper Ohm terms;
     * the pointers are pre-offset so the axial index of the residual
     * grid is used directly (valid over [-ng, nz+ng] nodal,
     * [-ng, nz-1+ng] cell). Identical in both CONDUCTOR contracts (the
     * masked E values are state-independent constants either way, so
     * the Jacobian row drops match); the DIELECTRIC standoff reports an
     * inactive view instead -- the band imposes no field constraint, so
     * the preconditioner drops no rows and the seam guard never
     * engages (the stencils see a continuous live field). */
    [[nodiscard]] warpx::mhd_pc::WallMaskView View () const;

    /** Device view of the rigid-conductor FLUID freeze for the
     * preconditioner's cell-centered fluid sweeps: active whenever a
     * thermal wall BC is active (the same gate as the residual's
     * wall_live masked-row identities), in EVERY wall_model -- in
     * particular under dielectric, whose FIELD-side View() is inactive
     * by design. */
    [[nodiscard]] warpx::mhd_pc::WallFluidFreezeView FluidFreezeView () const;

    /** Count of LIVE (unmasked) seam-guarded rows per electric-field
     * family in the valid domain, for banners: [E_r, E_theta, E_z].
     * Zero everywhere when the wall is inactive or dielectric (the
     * EM-transparent standoff guards nothing). */
    [[nodiscard]] amrex::Array<long, 3> SeamGuardedRowCounts () const
    {
        return m_dielectric ? amrex::Array<long, 3>{0, 0, 0}
                            : m_seam_guarded_rows;
    }

    /** Apply the conductor projection to the assembled electric field.
     * wall_model = pec (total-field contract): at every masked location
     * set E = -E_external (when the split external fields are present)
     * or E = 0. wall_model = pec_response (plasma-response contract):
     * set E = 0 (the external drive is transparent). Covers the domain
     * ghosts (the external registers carry trusted domain-ghost values).
     * wall_model = dielectric: performs NOTHING -- the EM-transparent
     * standoff imposes no condition on the response field (the
     * PEC_Insulator insulator-side convention; see the class comment). */
    void ProjectElectricField (ablastr::fields::VectorField const& efield,
                               ablastr::fields::VectorField const* efield_external,
                               const amrex::Geometry& geom) const;

private:
    bool m_active = false;
    /** true: pec (conductor condition on the TOTAL field); false:
     * pec_response (plasma-response field only, run32 EB parity) or
     * dielectric (no field condition at all; see m_dielectric). */
    bool m_total_field = true;
    /** wall_model = dielectric: EM-transparent standoff -- pec_response's
     * fluid contract with NO field-side machinery (see the class
     * comment). */
    bool m_dielectric = false;
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
    /** First SEAM-GUARDED radial index per axial index (see the class
     * comment: rows whose Hall/inertia/hyper stencil footprint touches a
     * masked cell; <= the masked index everywhere), same layout and
     * memory contract as the masked tables. */
    amrex::Gpu::ManagedVector<int> m_first_guarded_er; //!< (r-cell, z-node)
    amrex::Gpu::ManagedVector<int> m_first_guarded_et; //!< (r-node, z-node)
    amrex::Gpu::ManagedVector<int> m_first_guarded_ez; //!< (r-node, z-cell)
    /** Live seam-guarded rows per family in the valid domain
     * [E_r, E_theta, E_z] (banner reporting). */
    amrex::Array<long, 3> m_seam_guarded_rows = {0, 0, 0};
};

#endif // WARPX_IMPLICIT_MHD_WALL_MASK_H_
