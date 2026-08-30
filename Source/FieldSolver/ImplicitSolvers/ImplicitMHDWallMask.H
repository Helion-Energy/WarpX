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
 *     BAND HYGIENE. Because the dielectric band imposes no field
 *     constraint, whatever fluid the mask freezes outside the contour
 *     stays electrically live in the response Ohm's law: a loaded IC
 *     that leaves above-vacuum-threshold density there produces a
 *     frozen CONDUCTING shell (measured on the formation ladder:
 *     band |J_theta| 20x the interior median, current-keyed anomalous
 *     eta blow-up, Newton soft-caps, partial screening of the coil
 *     drive). Two solver-side guarantees close this:
 *
 *     - EXTERIOR CLAMP (automatic in every wall_live-freezing mode,
 *       i.e. whenever a thermal wall BC is active): at first-step
 *       sanitize time every masked fluid cell is SET to the rigid
 *       vacuum image -- mass density at the solver's positivity floor,
 *       zero momentum, electron/ion energies at their floor-consistent
 *       values at the configured background temperature (T_wall itself
 *       under wall_thermal_bc = temperature, so the one-sided
 *       interface drain starts at exactly zero gradient; else the
 *       temperature floors when set, else the pressure floors), each
 *       with the standard admissibility slack margin. The clamp is a
 *       fixed image (idempotent bit-exactly),
 *       so restarted states that already carry it are unchanged; the
 *       removed cell count and net scraped mass are reported in a
 *       rank-0 banner and as a '#'-comment header row of
 *       wall_ledger_file. The pedestal refresh and every end-of-step
 *       floor restoration skip the masked band, so nothing ever
 *       re-creates mass/energy outside the wall.
 *
 *     - wall_band_eta_override (input, default 0 = off): when > 0, the
 *       resistivity of the FIELD-advance Ohm's law is set to EXACTLY
 *       this constant at every BAND-INTERIOR electric-field location
 *       -- one whose complete interpolation neighborhood of
 *       cell-centered cells is masked; E locations on the stair
 *       interface itself keep the composed physical eta, so the
 *       live-side field evolution and the interface scraper are
 *       untouched. The override REPLACES the composed eta
 *       (Spitzer/user parser + current-keyed anomalous terms +
 *       density-keyed vacuum ramp), it does not compose with it:
 *       eta_band = const, dEta/dState = 0 there. Both halves matter
 *       (measured on the resistive-band physics of the production-formation
 *       lane):
 *       (a) INTERMEDIATE-ETA SHELL: a density-keyed eta ramp that
 *           lands mid-band leaves a resistive shell whose L/R time
 *           mu0 L^2 / eta can sit ON the drive timescale and screen
 *           the coil drive (band L ~ 0.08 m: mu0 L^2 ~ 8e-9 Ohm m s,
 *           so a ~10 us drive needs eta_band >> ~1e-3 Ohm m -- choose
 *           the override at or above the deck's vacuum-eta SATURATION
 *           value, uniform across the band, never mid-ramp);
 *       (b) STATE-DEPENDENT STIFFNESS: the current-keyed Chodura term
 *           scales ~1/n and explodes at the clamped floor density --
 *           a huge eta DOES insulate, but its Jacobian stiffness is
 *           what froze the production arm's Newton (run_v8_wall,
 *           step 415: 118 + 36 admissibility-pinned energy rows). A
 *           constant override removes the band from the Jacobian's
 *           eta chain entirely.
 *       The preconditioner ingests the identically overridden eta
 *       (per-staggering fills; the cell-centered interval-bound fill
 *       takes max(composed, override) so it keeps bounding the edge
 *       values), keeping residual/PC exact twins. Joule heating keeps
 *       the composed eta (masked cells receive no fluid increments
 *       anyway).
 *
 *     WALL FLUID BC vs the reference code (the `orange` source tree, surveyed 2026-08):
 *     The reference code treats the shaped wall as a hard, IMPERMEABLE, NO-SLIP
 *     solid -- all three velocity components held at zero on the wall
 *     nodes and the adjacent cut-cell skin layer (rows omitted from
 *     the implicit momentum solve, vp.f90/aniso_visc.f90), an explicit
 *     zero-mass-flux advection condition at the contour (adv.f90; no
 *     recycling/sputtering/absorption model anywhere), and a
 *     fixed-temperature Dirichlet bath at the background fill
 *     temperature (t_bc = 'd': T_i = tm0, T_e = f_te*tm0; anything
 *     else leaves the wall adiabatic). Particles and momentum are
 *     perfectly confined; only heat leaves, through conduction. OUR
 *     interface is deliberately different -- the "dielectric scraper":
 *     absorb-image Riemann faces (mass, momentum and enthalpy that
 *     cross are gone, the fluid analog of the run32 wall particle
 *     scraper) with C-inf rectified normal momentum, composed with the
 *     one-sided wall_thermal_bc drain. If a reference-parity no-normal-
 *     flow reflecting wall is ever needed, add it as a future
 *     wall_fluid_bc knob (absorb | reflect) rather than changing this
 *     default.
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
 *   * wall_thermal_bc = dirichlet: the interface faces PIN the wall
 *     temperature (the reference code's t_bc = 'd' contract): a TWO-SIDED
 *     conductive exchange against T_wall at the half-cell Dirichlet
 *     distance, free-streaming limited in BOTH directions (this is
 *     what distinguishes it from the rr12-fatal unlimited exchange).
 *     The pin target is anchored at max(T_wall, temperature-floor
 *     image), so the bath restores floor-adjacent cells instead of
 *     demanding states the admissibility projection forbids.
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
 * - wall_thermal_bc = none (default) | zero_flux | outflow_limited |
 *   dirichlet | dirichlet_limited ('temperature' was RENAMED
 *   outflow_limited: a capped one-sided drain is not a temperature pin)
 * - wall_temperature = <eV> (temperature/dirichlet modes)
 * - wall_band_eta_override = <ohm m> (default 0 = off): constant,
 *   state-independent field-advance eta at band-interior electric-
 *   field locations; see the band-hygiene note above. Aimed at the
 *   dielectric standoff (the conductor modes pin the masked E rows
 *   outright, making the override redundant there).
 * - wall_field_freeze = <bool> (default 0 = off, bit-identical):
 *   FIELD-side freeze of the masked band (work item B, v1). When on,
 *   every EVOLVED magnetic-field face whose COMPLETE Faraday update
 *   stencil -- its curl-E edges -- lies in the masked E region becomes
 *   an exact identity row of the residual (F = B - B^n, unit Jacobian
 *   diagonal; the field twin of the fluid wall_live identities): the
 *   plasma-response B on/outside the wall contour stays bit-frozen at
 *   its boot/restart value (physically ~0: the response starts zero
 *   and the exterior carries no fluid), so the discrete curl of the
 *   evolved B -- the band current Eric's directive targets -- is
 *   exactly zero at every strictly-exterior location from step 1 on.
 *   The painted EXTERNAL fields (split registers) are untouched: the
 *   coil drive still penetrates by prescription.
 *
 *   FREEZE-FACE CONVENTION (which faces count as "on the contour"):
 *   frozen iff ALL curl-E edges of the face are masked --
 *   B_r (r-node i, z-cell j) reads E_theta (i, j) and (i, j+1), so
 *   first_frozen_br[j] = max(first_masked_et[j], first_masked_et[j+1]);
 *   B_theta (r-cell i, z-cell j) reads E_r (i, j), (i, j+1) and
 *   E_z nodes (i, j), (i+1, j), so first_frozen_bt[j] =
 *   max(first_masked_er[j], first_masked_er[j+1], first_masked_ez[j]);
 *   B_z (r-cell i, z-node j) reads E_theta nodes (i, j) and (i+1, j),
 *   so first_frozen_bz[j] = first_masked_et[j]. The max() composition
 *   is INT_MAX-safe (a never-masked table entry keeps the whole row
 *   never-frozen), the same all-adjacent-masked convention as the
 *   band-interior eta tables. At a stair corner this is the
 *   conservative OUTER envelope: a face with even one live E edge
 *   stays LIVE, so the last live Faraday row still closes against the
 *   surface E and interior induction is untouched. The one-cell shell
 *   of cells straddling the freeze boundary loses the exact
 *   discrete-curl structure of its B update (its div B can drift by
 *   the dropped surface-E increments); the shell lies inside the
 *   floor-clamped band, so the drift is bounded and never touches
 *   live fluid.
 *
 *   CONSISTENCY TRIANGLE: the scraped fluid (exterior clamp, rho at
 *   the floor and u = 0 outside the contour), the frozen exterior
 *   response field (this knob) and J -> 0 at the boundary close each
 *   other -- with no exterior current there is nothing for the
 *   exterior field to evolve toward, and with the exterior field
 *   frozen the discrete curl feeding the band current is static at
 *   its (zero) boot value, consistent with the zero-density band.
 *   With the freeze on, the wall_band_eta_override becomes moot for
 *   the FIELD advance (the band-interior E rows no longer influence
 *   any live B row): the freeze supersedes the override's job; both
 *   knobs stay independent and composable.
 *
 *   Interactions: requires an active wall_model. Under pec_response
 *   it is an exact no-op (the pinned E already makes fully-masked
 *   faces identity rows); under pec it OVERRIDES the eddy-response
 *   closure of fully-masked faces (the response is pinned outright,
 *   so the metal no longer shields the drive there -- not recommended);
 *   it is aimed at dielectric, whose EM-transparent band otherwise
 *   lets the response solve move flux through unresolved band
 *   structure. Domain-ghost fills (ApplyBfieldBoundary, the
 *   Green's-function open-boundary r_hi/z-cap fills) act on DOMAIN
 *   ghosts outside the valid box and never overlap the interior
 *   frozen faces (the polyline is interior by the axis-clearance
 *   guard); checkpoint/restart carries the frozen faces like any
 *   state, and the freeze then holds the restart value. A
 *   Green's-contour exterior FILL (response-consistent exterior field
 *   instead of a frozen one) is a documented FUTURE upgrade of this
 *   hook. The preconditioner sees the identical freeze through
 *   GetMHDWallFieldFreezeForPC(): the resistive stencil emission
 *   drops frozen rows to identities (keeping their columns in live
 *   rows -- the exact block-triangular structure) and the block PC
 *   restores dB = r_B on frozen faces after the field-block inverses
 *   (residual/PC exact twins, validated in-run by
 *   pc_mhd_block.resistive_validate_assembly).
 */
/**
 * \brief Device view of the wall-band resistivity override
 * (implicit_mhd.wall_band_eta_override; see the ImplicitMHDWallMask
 * class comment). A location is BAND-INTERIOR -- and its field-advance
 * eta is REPLACED by the constant eta_override (dEta/dState = 0) -- iff
 * i >= first_band_*[j] for its staggering: every cell-centered cell of
 * its density/temperature interpolation neighborhood is masked, so E
 * locations on the stair interface itself (which bound live cells)
 * keep the composed eta by construction. The pointers are pre-offset
 * like the mask tables (nodal j valid over [-ng, nz+ng], cell j over
 * [-ng, nz-1+ng]); first_band_cc is the plain masked cell-centered
 * table, consumed by the preconditioner's cell-centered interval-bound
 * fill (max(composed, override) there, so its maximum keeps bounding
 * the overridden edge values). Consumed by the residual's Ohm assembly
 * and by the solver-side PC eta fills only -- the preconditioner
 * itself ingests the overridden MultiFabs, so residual and PC stay
 * exact twins with no extra machinery.
 */
struct WallBandEtaOverrideView
{
    bool active = false;
    amrex::Real eta_override = 0.0;
    const int* first_band_er = nullptr; //!< E_r rows: (r-cell, z-node)
    const int* first_band_et = nullptr; //!< E_theta rows: (r-node, z-node)
    const int* first_band_ez = nullptr; //!< E_z rows: (r-node, z-cell)
    const int* first_band_cc = nullptr; //!< cc fill: (r-cell, z-cell)
};

class ImplicitMHDWallMask
{
public:
    /** Thermal wall boundary at the stair-step fluid interface
     * (implicit_mhd.wall_thermal_bc); see the class comment. */
    enum class ThermalBC {
        none,
        zero_flux,
        outflow_limited, // one-sided capped drain (was 'temperature')
        dirichlet,       // HARD pin: uncapped two-sided half-cell exchange
        dirichlet_limited // the free-streaming-capped two-sided variant
    };

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
     * like the polyline itself). Non-null in every active wall_model
     * (thermal-BC consumers gate themselves on GetThermalBC(); the
     * wall-row viscosity mask reads it with or without a thermal
     * wall). */
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

    /** Device view of the wall-band resistivity override
     * (implicit_mhd.wall_band_eta_override): active only when the wall
     * is active AND the override is positive, in every wall_model
     * (under the conductor contracts the overridden rows are
     * subsequently pinned by the projection, so the override is
     * redundant there by construction). See WallBandEtaOverrideView. */
    [[nodiscard]] WallBandEtaOverrideView BandEtaOverrideView () const;

    /** Device view of the rigid-conductor FLUID freeze for the
     * preconditioner's cell-centered fluid sweeps: active whenever a
     * thermal wall BC is active (the same gate as the residual's
     * wall_live masked-row identities), in EVERY wall_model -- in
     * particular under dielectric, whose FIELD-side View() is inactive
     * by design. */
    [[nodiscard]] warpx::mhd_pc::WallFluidFreezeView FluidFreezeView () const;

    /** Device view of the shaped wall's FIELD freeze
     * (implicit_mhd.wall_field_freeze; see the class comment): the
     * per-B-staggering first-frozen tables consumed by the residual's
     * frozen-row rhs zeroing, by FinishStateUpdate's bit-exact carry,
     * and by the preconditioner (identity rows + post-inverse restore).
     * Active only when the wall is active AND the freeze is requested. */
    [[nodiscard]] warpx::mhd_pc::WallFieldFreezeView FieldFreezeView () const;

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
    /** implicit_mhd.wall_band_eta_override [ohm m]: constant
     * field-advance eta at band-interior electric-field locations
     * (0 = off). */
    amrex::Real m_band_eta_override = 0.0;
    /** implicit_mhd.wall_field_freeze: identity rows for every evolved
     * magnetic-field face whose complete Faraday stencil is masked (see
     * the class comment). */
    bool m_field_freeze = false;
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
    /** First BAND-INTERIOR radial index per axial index (see
     * WallBandEtaFloorView: every cell of the location's interpolation
     * neighborhood masked; >= the masked index everywhere), same layout
     * and memory contract as the masked tables. The cell-centered band
     * table is m_first_masked_cc itself. */
    amrex::Gpu::ManagedVector<int> m_first_band_er; //!< (r-cell, z-node)
    amrex::Gpu::ManagedVector<int> m_first_band_et; //!< (r-node, z-node)
    amrex::Gpu::ManagedVector<int> m_first_band_ez; //!< (r-node, z-cell)
    /** First FROZEN radial index per axial index for the FIELD freeze
     * (see WallFieldFreezeView and the freeze-face convention in the
     * class comment: all curl-E edges masked), same layout and memory
     * contract as the masked tables. Built in every active wall_model
     * (geometry-static); consumed only when m_field_freeze is set. */
    amrex::Gpu::ManagedVector<int> m_first_frozen_br; //!< (r-node, z-cell)
    amrex::Gpu::ManagedVector<int> m_first_frozen_bt; //!< (r-cell, z-cell)
    amrex::Gpu::ManagedVector<int> m_first_frozen_bz; //!< (r-cell, z-node)
    /** Live seam-guarded rows per family in the valid domain
     * [E_r, E_theta, E_z] (banner reporting). */
    amrex::Array<long, 3> m_seam_guarded_rows = {0, 0, 0};
    /** Frozen magnetic-field faces per family in the valid domain
     * [B_r, B_theta, B_z] (banner reporting). */
    amrex::Array<long, 3> m_frozen_faces = {0, 0, 0};
};

#endif // WARPX_IMPLICIT_MHD_WALL_MASK_H_
