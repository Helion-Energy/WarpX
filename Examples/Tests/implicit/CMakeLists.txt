# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_theta_implicit_planar_pinch  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_planar_pinch  # inputs
    "analysis_pinch.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_uniform  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_uniform  # inputs
    "analysis_mhd_uniform.py diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_alfven  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_alfven  # inputs
    "analysis_mhd_alfven.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_alfven_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_alfven  # inputs
    "analysis_mhd_alfven.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_alfven_preconditioner  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_alfven_preconditioner  # inputs
    "analysis_mhd_stiff_alfven.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_alfven_preconditioner_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_alfven_preconditioner  # inputs
    "analysis_mhd_stiff_alfven.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_evolving_ions  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_stiff_whistler_preconditioner implicit_mhd.evolve_ion_fluid=true"  # inputs
    "analysis_mhd_stiff_whistler_evolving_ions.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_preconditioner  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_whistler_preconditioner  # inputs
    "analysis_mhd_stiff_whistler.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Whistler under the conservative-form recast: Hall EMF in the solver
# Ohm assembly + block-banded Hall/whistler preconditioner rows
# (single-rank: the banded factorization is sequential along z).
add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_central  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_whistler_central  # inputs
    "analysis_mhd_stiff_whistler_central.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Electron-inertia whistler cap at k_eff d_e ~ 1 under the recast: the
# dispersion test that discriminates the capped branch from the
# wrong-sign spurious resonance (nothing else can), plus the SPD inertia
# mass in the banded preconditioner rows (validated to roundoff in-run).
add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_inertia  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_whistler_inertia  # inputs
    "analysis_mhd_stiff_whistler_inertia.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Advective electron-inertia coverage: a frozen z-drift Dopplers the
# capped whistler by (1+kappa) k_c u0 (ideal + kappa-weighted advective
# piece) -- the only 1D state where -(Je.grad)(Je/rho) is nonzero. Also
# the ungated baseline the armed-but-inactive dust-gate twin must
# reproduce bit for bit.
add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_inertia_drift  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_whistler_inertia_drift  # inputs
    "analysis_mhd_stiff_whistler_inertia_drift.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Dust-gate bit-identity (electron_inertia_linear_below armed BELOW the
# uniform density, rho0/4): the blend weight is exactly 1.0 in double
# precision above ~1.77x the gate, so this run must match the ungated
# drift baseline BIT FOR BIT -- and must still land on the tight FULL
# drift-Doppler dispersion (a flipped blend would run the linear form
# and fail that gate by ~7e-5 T against a ~4e-15 T pass residual).
add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_inertia_drift_gated  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_stiff_whistler_inertia_drift my_constants.rho_gate=rho0/4"  # inputs
    "analysis_mhd_stiff_whistler_inertia_drift_gated.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_stiff_whistler_inertia_drift  # dependency
)

# Dust-band effectiveness baseline (UNGATED): a kappa ~ 1 whistler packet
# in a band at 1.5x the Ohm floor (grid-scale d_e -- the production dust
# regime) with electrons streaming through; must follow the FULL
# drift-Doppler capped advance of the band parameters.
add_warpx_test(
    test_1d_theta_implicit_mhd_inertia_dustband  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_inertia_dustband  # inputs
    "analysis_mhd_inertia_dustband.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Dust-gate effectiveness: gate at 5x the band density (weight exactly 0
# in the band, exactly 1 in the 2x-gate background) -- the band packet
# must follow the LINEAR capped advance instead (the PC-unrepresentable
# advective Doppler removed) and the GMRES count must not exceed the
# ungated baseline's.
add_warpx_test(
    test_1d_theta_implicit_mhd_inertia_dustband_gated  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_inertia_dustband my_constants.rho_gate=1.0e21*m_p"  # inputs
    "analysis_mhd_inertia_dustband_gated.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_inertia_dustband  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_stiff_whistler_preconditioner_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_stiff_whistler_preconditioner  # inputs
    "analysis_mhd_stiff_whistler.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_contact  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_contact  # inputs
    "analysis_mhd_contact.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_static_contact_hllc  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_static_contact  # inputs
    "analysis_mhd_static_contact.py diags/diag000000 diags/diag000100 hllc"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# calibration variant: the SAME static contact must diffuse under the
# Rusanov flux, proving the hllc preservation assert is discriminating
add_warpx_test(
    test_1d_theta_implicit_mhd_static_contact_rusanov  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_static_contact implicit_mhd.fluid_flux=rusanov"  # inputs
    "analysis_mhd_static_contact.py diags/diag000000 diags/diag000100 rusanov"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_sod  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_sod  # inputs
    "analysis_mhd_sod.py diags/diag000000 diags/diag000200 total_energy"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# calibration variant: the SAME shock tube with the barotropic closure has
# the wrong (isentropic) jump conditions and must MISS the exact Riemann
# plateaus, proving the total_energy assert above is discriminating
add_warpx_test(
    test_1d_theta_implicit_mhd_sod_barotropic  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod implicit_mhd.ion_closure=barotropic"  # inputs
    "analysis_mhd_sod.py diags/diag000000 diags/diag000200 barotropic"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# JFNK-smoothing options: barotropic signal speeds + the C-infinity
# contact-side blend must not degrade the shock solution -- the same
# exact-Riemann plateau asserts as the reference sod test above
add_warpx_test(
    test_1d_theta_implicit_mhd_sod_smoothed  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod implicit_mhd.hllc_signal_closure=barotropic implicit_mhd.hllc_contact_blend=0.05"  # inputs
    "analysis_mhd_sod.py diags/diag000000 diags/diag000200 total_energy"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# electron-pressure-dominated regime (FRC-halo-like): the total-pressure
# enthalpy form of the E_i flux must stay consistent with the HLLC wave
# structure, or the Newton line search locks here
add_warpx_test(
    test_1d_theta_implicit_mhd_sod_pe_dominated  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_sod_pe_dominated  # inputs
    "analysis_mhd_sod_pe_dominated.py diags/diag000000 diags/diag000200"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_energy_conservation  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_energy_conservation  # inputs
    "analysis_mhd_energy_conservation.py diags/diag000000 diags/diag000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_hllc_floor  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_hllc_floor  # inputs
    "analysis_mhd_hllc_floor.py diags/diag000000 diags/diag000250"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# dual-energy (the reference code's mixmaster) ion closure: cold supersonic advection
# stays bit-cold under the fk-blended pressure, while the total_energy
# twin on the identical flow shows the total-minus-kinetic cancellation
# heating the closure exists to remove
add_warpx_test(
    test_1d_theta_implicit_mhd_dual_energy_advection  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_dual_energy_advection  # inputs
    "analysis_mhd_dual_energy_advection.py diags/diag000032"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_dual_energy_advection_total  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_dual_energy_advection implicit_mhd.ion_closure=total_energy"  # inputs
    "analysis_mhd_dual_energy_advection_total.py diags/diag000032 ../test_1d_theta_implicit_mhd_dual_energy_advection/diags/diag000032"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_dual_energy_advection  # dependency
)

# dual-energy converging-flow stagnation: heating comes ONLY from the
# physical blended PdV work, so the blended pressure holds the analytic
# adiabat; the total_energy twin overshoots it on the identical flow
add_warpx_test(
    test_1d_theta_implicit_mhd_dual_energy_stagnation  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_dual_energy_stagnation  # inputs
    "analysis_mhd_dual_energy_stagnation.py diags/diag000032"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_dual_energy_stagnation_total  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_dual_energy_stagnation implicit_mhd.ion_closure=total_energy"  # inputs
    "analysis_mhd_dual_energy_stagnation_total.py diags/diag000032 ../test_1d_theta_implicit_mhd_dual_energy_stagnation/diags/diag000032"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_dual_energy_stagnation  # dependency
)

# dual-energy restart round-trip: the auxiliary U_i block is checkpointed
# like every other fluid block, so a run restarted from the mid-run
# checkpoint reproduces the uninterrupted final state to round-off
add_warpx_test(
    test_1d_theta_implicit_mhd_dual_energy_restart  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_dual_energy_stagnation amr.restart='../test_1d_theta_implicit_mhd_dual_energy_stagnation/diags/chk000016'"  # inputs
    "analysis_mhd_dual_energy_restart.py diags/diag000032 ../test_1d_theta_implicit_mhd_dual_energy_stagnation/diags/diag000032"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_dual_energy_stagnation  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_temperature_floor  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_temperature_floor  # inputs
    "analysis_mhd_temperature_floor.py diags/diag000000 diags/diag000200"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# offset-density halo pedestal: the sub-pedestal halo of a static contact
# is raised exactly onto the dynamic (peak-keyed) pedestal and rides
# there, while the bulk state is exactly invariant
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_pedestal  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_pedestal  # inputs
    "analysis_mhd_halo_pedestal.py diags/diag000000 diags/diag000005"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# the same pedestal invariance under the CENTRAL recast flux: the donor
# drain gates are shared by every recast face flux, so the pedestal band
# must be held identically (gates the lifted Riemann-only assert)
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_pedestal_central  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_halo_pedestal implicit_mhd.fluid_flux=central implicit_mhd.viscosity=1.0"  # inputs
    "analysis_mhd_halo_pedestal.py diags/diag000000 diags/diag000005"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Calibrated pedestal-band velocity relaxation: on a static tangential-
# slip contact (identically zero HLLD fluxes) the drag is the only
# dynamics -- the band decays at the exact theta-discretized rate with
# a matched E_i kinetic drain, while the bulk (>= 2x pedestal) is
# exactly invariant
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_pedestal_drag  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_pedestal_drag  # inputs
    "analysis_mhd_halo_pedestal_drag.py diags/diag000000 diags/diag000008"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Calibrated pedestal-band ion-energy relaxation: on the same static
# tangential-slip contact (identically zero HLLD fluxes at t = 0, drag
# OFF) the relaxation is the only dynamics -- the band's INTERNAL
# energy decays toward the frozen pedestal image at the exact
# theta-discretized rate while momentum (and with it the kinetic part
# of E_i) stays invariant, and the bulk (>= 2x pedestal) is exactly
# invariant
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_pedestal_energy  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_pedestal_energy  # inputs
    "analysis_mhd_halo_pedestal_energy.py diags/diag000000 diags/diag000008"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Rectifier companion: with the band's internal energy loaded BELOW
# the pedestal image (and E_i ~99.9% kinetic, mirroring the wall-band
# population), the one-sided drain must do exactly nothing -- a
# two-sided form would pump the band toward its kinetic-following
# target at the exact theta rate (the measured FRC-ladder runaway)
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_pedestal_energy_one_sided  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_pedestal_energy_one_sided  # inputs
    "analysis_mhd_halo_pedestal_energy_one_sided.py diags/diag000000 diags/diag000008"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Calibrated windowed halo temperature-relaxation outlet
# (implicit_mhd.halo_relaxation_*): on a three-region static column the
# hot sub-window band drains BOTH species toward the cold-medium
# targets at the exact theta-discretized rate (measured nu to < 2%),
# the above-window band (window weight exactly zero) and the sub-target
# cold band (one-sided rectifier exactly closed) are untouched, and the
# removal ledger closes against the domain energy loss to < 1e-6
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_relaxation  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_relaxation  # inputs
    "analysis_mhd_halo_relaxation.py on diags/diag000000 diags/diag000008"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Knob-off NULL TWIN of the same column: the legacy path must be
# untouched (the outlet would have drained the hot band by ~4%) and no
# ledger file may be produced -- the bit-identical-when-off contract
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_relaxation_off  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_relaxation_off  # inputs
    "analysis_mhd_halo_relaxation.py off diags/diag000000 diags/diag000008"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Ion-target valve mode (halo_relaxation_target = ion, the reference code's
# te <= tm cap vp.f90:731 in relaxed JFNK-safe form): on a uniform
# sub-window column the electron energy decays toward the LOCAL frozen
# ion-temperature image at the exact theta rate while the ion channel,
# momentum, and density stay untouched; the ledger closes against the
# electron-energy loss
add_warpx_test(
    test_1d_theta_implicit_mhd_halo_relaxation_ion_target  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_halo_relaxation_ion_target  # inputs
    "analysis_mhd_halo_relaxation.py ion_target diags/diag000000 diags/diag000008"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Engineered Newton active-set deadlock (the rr13d_C1 production-corpse
# mechanism): a partnerless ion-energy sink pinned at a temperature-floor
# admissibility bound dominates the residual norm and stagnates the
# full-norm Armijo search at iteration 0 from ~step 15 -- without the
# free-subspace line-search rescue the run aborts on the frozen-step
# guard at ~step 33; with it every pinned solve accepts a free-subspace
# step, the ratchet releases, and the run completes with the defect
# bounded and finally resolved
add_warpx_test(
    test_1d_theta_implicit_mhd_active_set_rescue  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_active_set_rescue  # inputs
    "analysis_mhd_active_set_rescue.py diags/diag000000 diags/diag000060 diags/diag000120"  # analysis
    OFF  # checksum
    OFF  # dependency
)
if(TEST test_1d_theta_implicit_mhd_active_set_rescue.run)
    # the rescue MUST fire: this line only prints when the full-norm
    # Armijo search exhausted its backtracks and the free-subspace
    # ladder accepted a step (the unfixed solver aborts instead)
    set_property(
        TEST test_1d_theta_implicit_mhd_active_set_rescue.run
        PROPERTY PASS_REGULAR_EXPRESSION
        "Newton: accepted free-subspace step"
    )
endif()

# Floor-consistency completion of the engineered deadlock (the designed
# escalation of the rescue, motivated by rr13e_C1w where the pinned
# defect grew 0.468 -> 1.004 until the freeze guard fired): the SAME
# deck with floor_consistency_rate armed at the 1/(theta dt) cap. The
# reservoir supplies the partnerless sink's demand at the bound, the
# band rides the admissibility image instead of wedging (no persistent
# projections, no rescues, no frozen steps), and the supplied energy is
# booked and closed against the analytic sink demand by the analysis.
# Depends on the rescue baseline for its newton.txt defect plateau and
# its drained-through terminal state.
add_warpx_test(
    test_1d_theta_implicit_mhd_floor_consistency  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_floor_consistency  # inputs
    "analysis_mhd_floor_consistency.py diags/diag000000 diags/diag000060 diags/diag000120"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_active_set_rescue  # dependency
)
if(TEST test_1d_theta_implicit_mhd_floor_consistency.run)
    # ZERO frozen-step warnings and ZERO pinned components: the supply
    # keeps the equations consistent at the bound, so the line search
    # never freezes ("(frozen step" only prints in the iteration-0
    # zero-progress warning; the banner's "max frozen steps" must not
    # trip this) and the projection never clamps a direction component
    set_property(
        TEST test_1d_theta_implicit_mhd_floor_consistency.run
        PROPERTY FAIL_REGULAR_EXPRESSION
        "[(]frozen step;Newton: projected"
    )
endif()

# Do-no-harm twin: the quiet static contact with the source armed at the
# rate cap and a ledger file must be BIT-IDENTICAL to the rate-off twin
# (fields and newton.txt) with an all-zero ledger -- the rectifier's
# exact-zero gate above 1.2x the bound, proven on a live deck
add_warpx_test(
    test_1d_theta_implicit_mhd_floor_consistency_noharm  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_static_contact implicit_mhd.floor_consistency_rate=1.0e10 implicit_mhd.floor_ledger_file=diags/floor_ledger.txt"  # inputs
    "analysis_mhd_floor_consistency_noharm.py diags/diag000000 diags/diag000100"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_static_contact_hllc  # dependency
)

# The reference code's density eater (ntb.f90 density_eater, eaten_type > 0): on a
# static contact with an over-dense zone in the z_lo band, the band
# excess above the 0.01 rho0 target must decay by the analytic factor
# (1 - rate) = 0.8 per step with velocity and electron temperature
# preserved, the ion energy DENSITY invariant (the reference shot wio/wik
# preservation -- per-particle T_i rises), a far out-of-band bump
# untouched, and the removal ledger closing the domain mass deficit.
# OFF-state bit-identity is by construction (rate = 0 returns before
# touching any state).
add_warpx_test(
    test_1d_theta_implicit_mhd_density_eater  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_density_eater  # inputs
    "analysis_mhd_density_eater.py diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_positivity  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_positivity  # inputs
    "analysis_mhd_positivity.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)
if(TEST test_1d_theta_implicit_mhd_positivity.run)
    # the positivity mechanism is componentwise projection of the Newton
    # direction onto the admissibility bounds (was: a global step clamp)
    set_property(
        TEST test_1d_theta_implicit_mhd_positivity.run
        PROPERTY PASS_REGULAR_EXPRESSION
        "Newton: projected"
    )
endif()

add_warpx_test(
    test_1d_theta_implicit_mhd_preconditioner  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_preconditioner  # inputs
    "analysis_mhd_preconditioner.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_preconditioner_mpi  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_preconditioner  # inputs
    "analysis_mhd_preconditioner.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_resistive  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_resistive  # inputs
    "analysis_mhd_resistive.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# live Spitzer sweep on the parser Te symbol: circularly-polarized mode
# decay slows by (Te/Te0)^-1.5 as its own Joule heat raises Te
add_warpx_test(
    test_1d_theta_implicit_mhd_spitzer_decay  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_spitzer_decay  # inputs
    "analysis_mhd_spitzer_decay.py diags"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# run32-form Chodura anomalous resistivity (J/rho-keyed, smooth
# saturation and guards) on the tangential-discontinuity current sheets
add_warpx_test(
    test_1d_theta_implicit_mhd_chodura_sheet  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_chodura_sheet  # inputs
    "analysis_mhd_chodura_sheet.py diags/diag000000 diags/diag000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# live Spitzer chi_e(Te) on the parser Te symbol: a temperature ripple's
# diffusive relaxation speeds up by (Te/Te0)^2.5 as uniform Joule
# heating raises the base Te (the conduction analog of spitzer_decay)
# the same Spitzer conduction physics with the coefficient inputs
# FROZEN at the step-old state (conduction_coefficient_state =
# step_old): the chi(T) coefficient lags one step, which the same
# analysis tolerances must absorb -- gates the frozen-coefficient
# option's physics fidelity
add_warpx_test(
    test_1d_theta_implicit_mhd_spitzer_conduction_frozen_coeff  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_spitzer_conduction implicit_mhd.conduction_coefficient_state=step_old"  # inputs
    "analysis_mhd_spitzer_conduction.py diags"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_spitzer_conduction  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_spitzer_conduction  # inputs
    "analysis_mhd_spitzer_conduction.py diags"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# free-streaming conduction limiter: the cliff face flux saturates at
# the analytic harmonic cap q_unl/(1 + q_unl/(f q_fs))
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_limiter  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_conduction_limiter  # inputs
    "analysis_mhd_conduction_limiter.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Braginskii conduction in the unmagnetized limit (B = 0): both channels
# must decay at the first-principles chi_par k^2 rates (chi_perp =
# chi_par exactly at x = 0), with the conduction_chi_max soft cap armed
# BETWEEN the two channel values -- the ion chi lands on the cap (-11%
# shift the 5% gate must see applied) while the electron chi passes the
# knee untouched
add_warpx_test(
    test_1d_theta_implicit_mhd_braginskii_isolimit  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_braginskii_isolimit  # inputs
    "analysis_mhd_braginskii_isolimit.py diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Braginskii quasi-shorting twin (conduction_qs_chi = 0): the no-QS
# Reference the braginskii_qs gates difference against
add_warpx_test(
    test_1d_theta_implicit_mhd_braginskii_qs_twin  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_braginskii_qs my_constants.chi_qs=0.0"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Braginskii quasi-shorting cross-field boost: an entropy-excess pocket
# in a rarefied magnetized band must drain across the flux surfaces at
# the additive chi_add rate (>1e3 x the no-QS twin, matching the
# initial-profile face-flux prediction), while an ON-ADIABAT (s = 1)
# background ripple sees the shifted-ramp leak only -- measured
# < 1% of the amplitude (a ramp centered on the envelope leaks 3.75%
# and fails)
add_warpx_test(
    test_1d_theta_implicit_mhd_braginskii_qs  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_braginskii_qs  # inputs
    "analysis_mhd_braginskii_qs.py diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_braginskii_qs_twin  # dependency
)

# Reference-code-style density-keyed halo boost of the Braginskii ION chi_perp
# (implicit_mhd.conduction_halo_boost, the dimensionless dp_mn of the
# exact multiplicative the reference code's form, keyed to the shared reference of
# vacuum_reference_base_density -- the reference code's en00): a single-zone
# entropy-mode magnetized column whose cross-field ION e_spec ripple
# must relax at raw chi_perp_i MULTIPLIED by exactly dp (n_ref/n)^2 =
# 100 while the electron channel stays frozen (no electron boost)
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_halo_boost  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_conduction_halo_boost  # inputs
    "analysis_mhd_conduction_halo_boost.py boost diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# The sub-reference twin: dp = 1 with the reference BELOW the density
# (boost = 0.09 < 1) is an EXACT no-op of the multiplicative MAX form --
# both channels must decay at their raw Braginskii chi_perp (an
# additive/quadrature floor composition would still add here)
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_halo_boost_noop  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_conduction_halo_boost_noop  # inputs
    "analysis_mhd_conduction_halo_boost.py noop diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# The ion-only discriminator and clamp-order twin: dp = 1e6 rides the
# boosted ion value onto the DEFINED-convention perp cap (the clamps
# cap the BOOSTED value, the reference code's t_cond order): the measured ion chi
# must equal the cap itself while the electron channel stays on its
# RAW value -- an electron leak of the boost (the pre-2026-08-31
# both-channels form) would drag it onto the same cap rate; also
# exercises the dynamic peak-fraction reference path winning the max
# composition over the static base
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_halo_boost_ion_only  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_conduction_halo_boost_ion_only  # inputs
    "analysis_mhd_conduction_halo_boost.py ion_only diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# The production-stack routing gate (the run_h1_halodrain masking hunt,
# 2026-08-31): the "boost" arm's exact x100 gate re-run through every
# routing knob of the RZ production conduction stack -- explicit
# conduction_theta (== theta), step-old frozen coefficients, the
# free-streaming limiter (at an amplitude that keeps it inert), armed
# per-component + legacy clamps with the target interior, and the hlld
# fan; none of them may re-route or re-scale the boost. The actual
# production masker (conduction_qs_chi saturating every face onto the
# perp cap) is excluded against the dp boost by the parse guard.
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_halo_boost_production_stack  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_conduction_halo_boost_production_stack  # inputs
    "analysis_mhd_conduction_halo_boost.py production_stack diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# conduction-stage L-stability (implicit_mhd.conduction_theta):
# backward-Euler conduction (theta_c = 1) must obliterate a seeded
# grid-Nyquist Te ripple at theta dt chi/dx^2 = 1e3 (one-step factor
# 1/(1 + 8e3)), land the analytic fully-diffused profile, and close
# sum(U_e) to round-off
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_theta  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_conduction_theta  # inputs
    "analysis_mhd_conduction_theta.py diags/diag000000 diags/diag000010 backward_euler"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# the same deck at conduction_theta = theta = 0.5 (the bit-identical
# default Crank-Nicolson path): documents the L-stability failure the
# stage shift exists for -- the Nyquist checkerboard survives with
# |amplification| ~ 1 and its fluctuation-power share grows (the
# production halo signature)
add_warpx_test(
    test_1d_theta_implicit_mhd_conduction_theta_cn  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_conduction_theta implicit_mhd.conduction_theta=0.5"  # inputs
    "analysis_mhd_conduction_theta.py diags/diag000000 diags/diag000010 crank_nicolson"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# density-keyed vacuum resistivity + resistive backward-Euler centering:
# one-step transverse-mode decay matches the theta_r-discrete prediction
# 1/(1+z) (not the trapezoidal value), with no electron heating from the
# boosted field-advance eta
add_warpx_test(
    test_1d_theta_implicit_mhd_vacuum_resistive  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_vacuum_resistive  # inputs
    "analysis_mhd_vacuum_resistive.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# preconditioned twin of the calibrated vacuum-resistive decay: identical
# physics assertions, and the pc_mhd_block Chebyshev resistive block must
# cut the cumulative GMRES iterations strictly (and by at least 3x; the
# measured reduction is 90 -> 8) below the baseline run's
add_warpx_test(
    test_1d_theta_implicit_mhd_vacuum_resistive_pc  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_vacuum_resistive_pc  # inputs
    "analysis_mhd_vacuum_resistive_pc.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_vacuum_resistive  # dependency
)

# assembly-validation twin: the direct resistive solver's sparse rows are
# checked to roundoff against the matrix-free operator at every active
# preconditioner update (in-run abort on mismatch); physics assertions
# are the calibrated baseline's
add_warpx_test(
    test_1d_theta_implicit_mhd_resistive_pc_assembly  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_resistive_pc_assembly  # inputs
    "analysis_mhd_vacuum_resistive.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# banded twin: the same preconditioned physics with the resistive block
# inverted by the block-banded LU along z (Chacon direct
# preconditioner; periodic-z wraps PC-grade dropped); same GMRES margin
# as the Chebyshev twin
add_warpx_test(
    test_1d_theta_implicit_mhd_vacuum_resistive_pc_banded  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_vacuum_resistive_pc_banded  # inputs
    "analysis_mhd_vacuum_resistive_pc.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_vacuum_resistive  # dependency
)

# Reference-code-style Ohm-current Joule quench (implicit_mhd.joule_ohm_current):
# calibrated one-step two-zone check -- the vacuum-boosted halo's Joule
# deposit is quenched by orders of magnitude relative to the exact
# OFF-knob integral eta0 |J^theta|^2 dt, while the bulk (criterion not
# met) matches that integral tightly, i.e. is untouched by the knob
add_warpx_test(
    test_1d_theta_implicit_mhd_joule_ohm_quench  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_joule_ohm_quench  # inputs
    "analysis_mhd_joule_ohm_quench.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Reference-code-style dynamic reference density
# (implicit_mhd.vacuum_reference_peak_fraction): the joule-ohm-quench
# sibling with a MID-DENSITY zone (2% of the peak) whose Joule deposit
# is quenched ONLY because the dynamic reference max(n_floor,
# 0.1 n_peak) re-keys the vacuum boost and the dominance criterion --
# the analysis re-derives the static-reference margin showing the
# knob-OFF run would leave the zone unquenched, and the bulk (still
# non-dominated under the dynamic reference) must match the exact
# OFF integral
add_warpx_test(
    test_1d_theta_implicit_mhd_dynamic_reference  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_mhd_dynamic_reference  # inputs
    "analysis_mhd_dynamic_reference.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_sound  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_sound  # inputs
    "analysis_mhd_sound.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_emhd  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_emhd  # inputs
    "analysis_mhd_emhd.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_semi_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_semi_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

if(AMReX_PETSC)
    add_warpx_test(
        test_2d_curl_curl_petsc_pc  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_curl_curl_petsc_pc  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1001000"  # analysis
        "analysis_default_regression.py --path diags/diag1001000"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_curl_curl_petsc_pc  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_theta_implicit_planar_pinch  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_theta_implicit_planar_pinch  # inputs
        "analysis_pinch.py diags/diag1000020"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_theta_implicit_dynamic_pinch  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_theta_implicit_dynamic_pinch  # inputs
        "analysis_pinch.py diags/diag1000020"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_theta_implicit_dynamic_pinch  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_theta_implicit_dynamic_pinch  # inputs
        "analysis_pinch.py diags/diag1000010"  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        OFF  # dependency
    )

endif()

add_warpx_test(
    test_rz_theta_implicit_mhd_z_outflow  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_outflow  # inputs
    "analysis_mhd_z_outflow.py diags/diag000000 diags/diag000240"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# z_lo mirror-symmetry boundary (boundary.field_lo = pmc on z +
# implicit_mhd.z_lo_boundary_fluid = symmetry), full-domain reference: a
# z-symmetric blob on z in [-L, L] whose evolution must PRESERVE the
# mirror parity of every field to solver roundoff (the operator commutes
# with the mirror), with an odd Btheta seed and an even azimuthal flow
# so every parity channel is live. Single rank: the half-domain twin
# must match it to the nonlinear-solver tolerance, so the runs keep the
# reduction order deterministic.
add_warpx_test(
    test_rz_theta_implicit_mhd_mirror_full  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_mirror_full  # inputs
    "analysis_mhd_mirror_full.py diags/diag000000 diags/diag000032"  # analysis
    "analysis_default_regression.py --path diags/diag000032"  # checksum
    OFF  # dependency
)

# half-domain twin on z in [0, L] with the mirror at z_lo: the gold
# standard -- fields and moments must match the full run's upper half to
# solver roundoff, the half-domain mass change must be roundoff before
# anything reaches the z_hi outflow end, tangential B on the plane must
# vanish, and the first-row axial momentum must be
# antisymmetry-consistent with the full run
add_warpx_test(
    test_rz_theta_implicit_mhd_mirror_half  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_mirror_half  # inputs
    "analysis_mhd_mirror_half.py diags/diag000000 diags/diag000032"  # analysis
    "analysis_default_regression.py --path diags/diag000032"  # checksum
    test_rz_theta_implicit_mhd_mirror_full  # dependency
)

# preconditioned twin of the half-domain mirror test: the MHD block PC's
# z_lo end rows must carry the mirror fold (ghost B columns folded with
# the tangential-odd/normal-even parities) instead of the Neumann-end
# tangential-E-zero convention -- resistive_validate_assembly checks the
# folded sparse assembly against the matrix-free application to
# roundoff, the banded scatter asserts the folded columns stay in-band,
# and right preconditioning must leave the converged answer on the full
# run's upper half (the same mirror-equivalence analysis runs unchanged)
add_warpx_test(
    test_rz_theta_implicit_mhd_mirror_half_pc  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_theta_implicit_mhd_mirror_half jacobian.pc_type=pc_mhd_block pc_mhd_block.resistive_solver=banded pc_mhd_block.resistive_validate_assembly=true pc_mhd_block.resistive_threshold=1.0e-6"  # inputs
    "analysis_mhd_mirror_half.py diags/diag000000 diags/diag000032"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_mirror_full  # dependency
)

# RZ coverage of the TVD reconstruction (implicit_mhd.fluid_reconstruction)
# on the PRODUCTION flux (central + viscosity): the same mirror-parity
# Reference, which asserts that the discrete operator commutes with the
# z-mirror to solver roundoff for every field of the parity table. That is
# a structural check the 1D gates cannot make -- it exercises the widened
# stencil on radial faces, across the axis (whose cell-centered current
# ghosts the reconstruction newly fills with the m = 0 vector parity), and
# on z faces at once, and it fails if the limiter is not symmetric under
# (far_left, left | right, far_right) -> (far_right, right | left,
# far_left). The limited slope is odd and symmetric in its two one-sided
# differences and the guard is built from a symmetric stencil norm, so it
# is; this is the gate that keeps it that way.
add_warpx_test(
    test_rz_theta_implicit_mhd_mirror_full_reconstruction  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_theta_implicit_mhd_mirror_full implicit_mhd.fluid_reconstruction=median"  # inputs
    "analysis_mhd_mirror_full.py diags/diag000000 diags/diag000032"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_mirror_full  # dependency
)

# Green's-function open r_hi boundary through the implicit residual: the
# same outflow physics with the uniform Bz supplied by the split external
# vector potential; the ghost fill (a linear map of the interior J_theta)
# runs inside every JFNK residual evaluation, on 2 ranks so the
# MPI-distributed kernel GEMV and psi reduction are exercised.
add_warpx_test(
    test_rz_theta_implicit_mhd_z_outflow_open  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_outflow_open  # inputs
    "analysis_mhd_z_outflow.py diags/diag000000 diags/diag000240 open"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# open (Green's-function) z caps: the axial FIELD ghosts defer to the
# free-space fill while the fluid keeps the Neumann outflow ghosts; the
# pulse must exit exactly as through the Neumann ends
add_warpx_test(
    test_rz_theta_implicit_mhd_z_open_caps  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_open_caps  # inputs
    "analysis_mhd_z_outflow.py diags/diag000000 diags/diag000240 open"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# ---------------------------------------------------------------------
# Conservative-form recast (fluid_flux = hlld): B is the JFNK array block
# and one HLLD Riemann solution per face feeds the fluid divergences, the
# Maxwell stress, and the ideal EMF.
# ---------------------------------------------------------------------

# circularly polarized Alfven wave: exact nonlinear solution; pins the
# EMF signs, the propagation speed, the first-order dissipation scale,
# and the absence of magnetosonic contamination
add_warpx_test(
    test_1d_theta_implicit_mhd_cpaw  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_cpaw  # inputs
    "analysis_mhd_cpaw.py diags/diag000000 diags/diag000080 diags/diag000160"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Brio & Wu shock tube in the robustness configuration (barotropic fan
# closure + widened kappa_bn): with the consistent fan at default
# kappa_bn this deterministically locks the Newton line search at step
# 18, so completion is the Newton-lock regression gate
add_warpx_test(
    test_1d_theta_implicit_mhd_briowu  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_briowu  # inputs
    "analysis_mhd_briowu.py diags/diag000000 diags/diag000200"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# static magnetized tangential discontinuity (the FRC-separatrix
# invariant) must be machine-preserved through the full assembly
add_warpx_test(
    test_1d_theta_implicit_mhd_tangential_discontinuity  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_tangential_discontinuity  # inputs
    "analysis_mhd_tangential_discontinuity.py diags/diag000000 diags/diag000100 hlld"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# calibration variant: the SAME state must diffuse under Rusanov,
# proving the exactness assert is discriminating
add_warpx_test(
    test_1d_theta_implicit_mhd_tangential_discontinuity_rusanov  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_tangential_discontinuity implicit_mhd.fluid_flux=rusanov"  # inputs
    "analysis_mhd_tangential_discontinuity.py diags/diag000000 diags/diag000100 rusanov"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# hyper-resistivity in the solver-assembled Ohm's law: a contact-inert
# transverse mode must decay at exactly eta_H kd^4 / mu0 through the
# theta-discrete amplification factor
add_warpx_test(
    test_1d_theta_implicit_mhd_hyper_resistive  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_hyper_resistive  # inputs
    "analysis_mhd_hyper_resistive.py diags/diag000000 diags/diag000040"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# preconditioned banded twin: the pc_mhd_block resistive block carries the
# frozen eta_H chain (banded LU at bandwidth 2), the calibrated decay is
# unchanged, the cumulative GMRES count falls strictly (with margin) below
# the baseline's, and every active update roundoff-validates the assembled
# eta_H rows against the matrix-free operator
add_warpx_test(
    test_1d_theta_implicit_mhd_hyper_resistive_pc_banded  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_hyper_resistive_pc_banded  # inputs
    "analysis_mhd_hyper_resistive_pc.py diags/diag000000 diags/diag000040"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_hyper_resistive  # dependency
)

# CGL bi-Maxwellian isotropization: a uniform static anisotropic state
# must relax p_perp - p_par at exactly the Braginskii nu_ii (re-derived
# from first principles in the analysis) through the theta-discrete
# factor, conserving U_par + U_perp to round-off
add_warpx_test(
    test_1d_theta_implicit_mhd_cgl_relaxation  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_cgl_relaxation  # inputs
    "analysis_mhd_cgl_relaxation.py diags/diag000000 diags/diag000040"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# instability-bounded variant: the same mirror-unstable state must be
# pulled to marginal stability at the cyclotron-scale enhanced rate
add_warpx_test(
    test_1d_theta_implicit_mhd_cgl_relaxation_bounded  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_cgl_relaxation implicit_mhd.cgl_instability_scale=0.1"  # inputs
    "analysis_mhd_cgl_relaxation.py diags/diag000000 diags/diag000040 bounded"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Stage B+ null blend, calibrated: at B = 0 the magnetization weight is
# exactly 0 and the same bi-Maxwellian decays at nu_ii + v_thi/dz
# (~116x collisional), against the theta-discrete amplification
add_warpx_test(
    test_1d_theta_implicit_mhd_cgl_null  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_cgl_null  # inputs
    "analysis_mhd_cgl_relaxation.py diags/diag000000 diags/diag000040 null"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# The reference code's electron-ion temperature equilibration (eq_brate): a uniform
# static two-temperature state must relax p_i - p_e through the exact
# per-step frozen-coefficient theta recurrence of the reference code's Spitzer
# rate (branches, refresh, antisymmetry), conserving U_e + U_i to
# round-off (recast/central path)
add_warpx_test(
    test_1d_theta_implicit_mhd_ei_equilibration  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_ei_equilibration  # inputs
    "analysis_mhd_ei_equilibration.py diags/diag000000 diags/diag000030 total"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# reversed variant: cold electrons against warm ions (the opposite
# relaxation direction AND the low-Te Coulomb-logarithm branch)
add_warpx_test(
    test_1d_theta_implicit_mhd_ei_equilibration_reverse  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_ei_equilibration my_constants.Te0=5.0 my_constants.Ti0=15.0"  # inputs
    "analysis_mhd_ei_equilibration.py diags/diag000000 diags/diag000030 total"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# non-recast (centered-flux) path of the same exchange: the residual
# must be one map regardless of the flux path
add_warpx_test(
    test_1d_theta_implicit_mhd_ei_equilibration_centered  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_ei_equilibration implicit_mhd.fluid_flux=legacy_e_centered implicit_mhd.viscosity=0"  # inputs
    "analysis_mhd_ei_equilibration.py diags/diag000000 diags/diag000030 total"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# dual-energy closure: the exchange must deposit into BOTH E_i and the
# auxiliary U_i (the analysis asserts the pair stays pinned)
add_warpx_test(
    test_1d_theta_implicit_mhd_ei_equilibration_dual  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_ei_equilibration_dual  # inputs
    "analysis_mhd_ei_equilibration.py diags/diag000000 diags/diag000030 dual"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# cgl closure: the isotropic -Q/3 / -2Q/3 pair deposit must keep an
# isotropic state exactly isotropic while p_eff follows the recurrence
add_warpx_test(
    test_1d_theta_implicit_mhd_ei_equilibration_cgl  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_ei_equilibration_cgl  # inputs
    "analysis_mhd_ei_equilibration.py diags/diag000000 diags/diag000030 cgl"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# hlld variants of the closure/positivity suite: the recast must
# reproduce the exact-Riemann plateaus, the exact static contact, and
# the exact fluid-energy pairing of the hllc/rusanov paths
add_warpx_test(
    test_1d_theta_implicit_mhd_sod_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true"  # inputs
    "analysis_mhd_sod.py diags/diag000000 diags/diag000200 total_energy"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# paired pc/no-pc gate for the Stage-1 hlld block preconditioner: the SAME
# shock tube with jacobian.pc_type = pc_mhd_block must land on the same
# exact-Riemann plateaus (identical-physics contract) with a bounded
# cumulative GMRES count (stationary fixed-cycle contract for plain GMRES)
add_warpx_test(
    test_1d_theta_implicit_mhd_sod_hlld_pc  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true jacobian.pc_type=pc_mhd_block"  # inputs
    "analysis_mhd_hlld_preconditioner.py diags/diag000000 diags/diag000200 total_energy"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_sod_pe_dominated_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod_pe_dominated implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true"  # inputs
    "analysis_mhd_sod_pe_dominated.py diags/diag000000 diags/diag000200"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_static_contact_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_static_contact implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true"  # inputs
    "analysis_mhd_static_contact.py diags/diag000000 diags/diag000100 hlld"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_energy_conservation_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_energy_conservation implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true"  # inputs
    "analysis_mhd_energy_conservation.py diags/diag000000 diags/diag000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# central-flux recast (Chacon-style central conservative flux + explicit
# viscosity in place of the Riemann fan): the sound-wave gate must land
# on the same centered discrete acoustic phase (nu = 0.5 m^2/s damps the
# mode by ~3e-7 over the one-step horizon, well inside the existing
# tolerance), and a uniform magnetized state must stay uniform to
# round-off (the viscous stress vanishes identically at u = 0)
add_warpx_test(
    test_1d_theta_implicit_mhd_central_sound  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sound implicit_mhd.fluid_flux=central implicit_mhd.viscosity=0.5 hybrid_pic_model.include_electron_pressure_term=false"  # inputs
    "analysis_mhd_sound.py diags/diag000000 diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_central_uniform  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_uniform implicit_mhd.fluid_flux=central implicit_mhd.viscosity=0.5 hybrid_pic_model.include_hall_term=false hybrid_pic_model.include_electron_pressure_term=false"  # inputs
    "analysis_mhd_uniform.py diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# ion thermal conduction on the central-flux face registers: a 1% sinusoidal
# ion-temperature mode on a uniform static background, with chi_i stiff
# against the acoustic response (chi k / c_s ~ 57), must decay at the
# analytic q = -kappa grad T rate chi k^2 to 5% (the known discretization
# corrections total ~0.3%) while the conservative conductive flux keeps the
# fluid energy ledger closed to roundoff
add_warpx_test(
    test_1d_theta_implicit_mhd_central_conduction  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_theta_implicit_mhd_central_conduction  # inputs
    "analysis_mhd_central_conduction.py diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# pressure-recovery corner width smoke check: a uniform magnetized state
# under the total_energy closure with a uniform axial flow (so the local
# kinetic energy is nonzero and pressure_corner_width_fraction genuinely
# inflates the smooth-max corner width) must stay uniform to round-off --
# the state never approaches the corner, and a uniform width inflation has
# zero gradient
add_warpx_test(
    test_1d_theta_implicit_mhd_central_uniform_corner_width  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_uniform implicit_mhd.fluid_flux=central implicit_mhd.viscosity=0.5 hybrid_pic_model.include_hall_term=false hybrid_pic_model.include_electron_pressure_term=false implicit_mhd.ion_closure=total_energy implicit_mhd.ion_pressure(x,y,z)=Pe implicit_mhd.ion_pressure_floor=1.0e-6*Pe implicit_mhd.velocity_z(x,y,z)=1.0e4 implicit_mhd.pressure_corner_width_fraction=0.01"  # inputs
    "analysis_mhd_uniform.py diags/diag000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# RZ recast invariants: div B at round-off and B_theta EXACTLY zero
# through 60 steps of corner-UCT-driven poloidal dynamics, on 2 ranks so
# the corner EMF is exercised across MPI seams
add_warpx_test(
    test_rz_theta_implicit_mhd_poloidal_divb  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_poloidal_divb  # inputs
    "analysis_mhd_poloidal_divb.py diags/diag000000 diags/diag000060"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# static radial contact: exact cancellation of the cylindrical
# face-weighted divergence against the geometric sources (hydro and
# Maxwell-stress parts)
add_warpx_test(
    test_rz_theta_implicit_mhd_radial_contact  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_radial_contact  # inputs
    "analysis_mhd_radial_contact.py diags/diag000000 diags/diag000100 hlld"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# anisotropic Braginskii spreading: a hot Gaussian electron spot in a
# strongly magnetized uniform Bz (Omega_e tau_e ~ 9.6e3) spreads along
# the field at the Braginskii parallel rate (z-variance growth
# 2 chi_par t) while the r-variance stays frozen (chi_perp/chi_par ~
# 1.6e-8); 2 ranks split z through the spot so the tangential corner
# stencil crosses the MPI seam at peak gradient
add_warpx_test(
    test_rz_theta_implicit_mhd_braginskii_spreading  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_braginskii_spreading  # inputs
    "analysis_mhd_braginskii_anisotropy.py diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_theta_implicit_mhd_radial_contact_rusanov  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_theta_implicit_mhd_radial_contact implicit_mhd.fluid_flux=rusanov"  # inputs
    "analysis_mhd_radial_contact.py diags/diag000000 diags/diag000100 rusanov"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# stiff RZ vacuum-resistive diffusion (grid diffusion number 150 at the
# halo eta maximum): corner-EMF structural invariants (exactly zero
# B_theta, conserved r-weighted axial flux, per-radius vanishing axial Br
# sums) plus the z-modulated eta exciting Br
add_warpx_test(
    test_rz_theta_implicit_mhd_vacuum_resistive  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_vacuum_resistive  # inputs
    "analysis_mhd_vacuum_resistive_rz.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# preconditioned twin: identical structural physics, final fields matching
# the baseline run to solver roundoff, and the pc_mhd_block Chebyshev
# resistive block cutting the cumulative GMRES iterations strictly (and by
# at least 10x; the measured reduction is ~1570 -> ~28) below the baseline
add_warpx_test(
    test_rz_theta_implicit_mhd_vacuum_resistive_pc  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_vacuum_resistive_pc  # inputs
    "analysis_mhd_vacuum_resistive_rz_pc.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_vacuum_resistive  # dependency
)

# assembly-validation twin: the direct resistive solver's sparse rows are
# checked to roundoff against the matrix-free operator at every active
# preconditioner update (in-run abort on mismatch); two ranks exercise the
# owner-mask deduplication of shared box-interface faces, the periodic
# z-nodal identification, and the gathered global pattern
add_warpx_test(
    test_rz_theta_implicit_mhd_resistive_pc_assembly  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_resistive_pc_assembly  # inputs
    "analysis_mhd_vacuum_resistive_rz.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# device-assembly parity twin (single rank, so the device value refresh
# engages): the device-refreshed CSR values are gated bitwise against the
# host emission+fold at every preconditioner update, on top of the
# roundoff matvec check; physics assertions are the baseline's
add_warpx_test(
    test_rz_theta_implicit_mhd_resistive_pc_assembly_device  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_resistive_pc_assembly_device  # inputs
    "analysis_mhd_vacuum_resistive_rz.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# banded twin: the same preconditioned physics with the resistive block
# inverted by the block-banded LU along z (dense radial blocks treat the
# axis/wall/halo coupling exactly; periodic-z wraps PC-grade dropped); one
# rank (the banded solver asserts single-rank), fields at solver tolerance
# against the two-rank baseline, same 10x GMRES margin
add_warpx_test(
    test_rz_theta_implicit_mhd_vacuum_resistive_pc_banded  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_vacuum_resistive_pc_banded  # inputs
    "analysis_mhd_vacuum_resistive_rz_pc_banded.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_vacuum_resistive  # dependency
)

# stiff RZ density-keyed hyper-resistive diffusion (grid hyper number 100
# at the halo minimum): the corner/face eta_H stencils with the axis
# regularizations drive the only dynamics; corner-EMF structural
# invariants plus the z-modulated eta_H exciting Br
add_warpx_test(
    test_rz_theta_implicit_mhd_hyper_resistive  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_hyper_resistive  # inputs
    "analysis_mhd_hyper_resistive_rz.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# preconditioned banded twin of the RZ hyper-resistive diffusion: fields
# at solver tolerance against the baseline, cumulative GMRES strictly
# (with margin) below it, and in-run roundoff validation of the assembled
# cylindrical eta_H rows against the matrix-free operator
add_warpx_test(
    test_rz_theta_implicit_mhd_hyper_resistive_pc_banded  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_hyper_resistive_pc_banded  # inputs
    "analysis_mhd_hyper_resistive_rz_pc.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_hyper_resistive  # dependency
)

# static theta-pinch column (pe-balanced Bz(r), u = 0, B_n = 0): the
# UCT-HLLD corner E_theta must be exactly inert -- the equilibrium-hold
# gate for the corner-EMF dissipation (fast-bound coefficients diffuse
# this profile within the run)
add_warpx_test(
    test_rz_theta_implicit_mhd_theta_pinch  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_theta_pinch  # inputs
    "analysis_mhd_theta_pinch.py diags/diag000000 diags/diag000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# hlld variant of the RZ outflow test (Neumann-z ends, PEC wall, axis)
add_warpx_test(
    test_rz_theta_implicit_mhd_z_outflow_hlld  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_z_outflow implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true"  # inputs
    "analysis_mhd_z_outflow.py diags/diag000000 diags/diag000240 pec"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# paired pc/no-pc gate for the Stage-2 hlld block preconditioner in RZ:
# the SAME advected pulse with jacobian.pc_type = pc_mhd_block must leave
# the domain identically (the z-outflow analysis runs unchanged) and,
# because every reference CFL is below one here, the stiffness-gated
# preconditioner must stay iteration-neutral against the pc-off twin
add_warpx_test(
    test_rz_theta_implicit_mhd_z_outflow_hlld_pc  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_z_outflow implicit_mhd.fluid_flux=hlld implicit_mhd.allow_hlld=true jacobian.pc_type=pc_mhd_block"  # inputs
    "analysis_mhd_z_outflow_pc.py diags/diag000000 diags/diag000240 pec"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# zero-flux reflecting wall at an OPEN radial field boundary (hlld +
# r_open_fluid = reflect): a rotating column driven into the wall with a
# nonzero external B_r through the open face must conserve total mass to
# round-off and keep the wall-ring ion energy healthy (the unfixed wall
# face leaks mass at the mirror-jump dissipation scale and drains E_i
# through the open-B_n Maxwell-stress work channel)
add_warpx_test(
    test_rz_theta_implicit_mhd_r_open_reflect_wall  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_r_open_reflect_wall  # inputs
    "analysis_mhd_r_open_reflect_wall.py diags/diag000000 diags/diag000080"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_theta_implicit_mhd_r_open_absorb_wall  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_r_open_absorb_wall  # inputs
    "analysis_mhd_r_open_absorb_wall.py diags/diag000000 diags/diag000080 diags/absorb_ledger.txt"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_theta_implicit_mhd_absorb_scrape_off  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_absorb_scrape_off  # inputs
    "analysis_mhd_absorb_scrape_off.py diags/diag000000 diags/diag000120 diags/absorb_ledger.txt"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_theta_implicit_mhd_external_wall_ghost  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_external_wall_ghost  # inputs
    "analysis_mhd_external_wall_ghost.py diags/diag000000 diags/diag000002"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# stair-step PEC shaped wall (implicit_mhd.wall_model = pec) under a
# programmed external uniform-Bz ramp: total flux inside the metal frozen
# to solver precision, interior ramp exclusion to the analytic
# infinite-cylinder-shell image solution; the resistive-block assembly
# validation asserts the wall-masked PC rows against the matrix-free
# operator at every update
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_pec_ramp  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_pec_ramp  # inputs
    "analysis_mhd_wall_pec_ramp.py diags/diag000000 diags/diag000040 pec"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# plasma-response conductor contract (run32 EB parity): the same mask
# pins the plasma field only -- the programmed drives are transparent
# (pocket ratio 1 vs < 0.2 for pec) while the plasma flux through every
# masked corner ring is pinned to solver precision against the analytic
# external r*A_theta reference
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_pec_response_ramp  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_pec_ramp implicit_mhd.wall_model=pec_response"  # inputs
    "analysis_mhd_wall_pec_ramp.py diags/diag000000 diags/diag000040 pec_response"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# stair-step wall THERMAL boundary, mode "zero_flux": conduction must
# pass exactly nothing through the shaped-wall interface (interior
# unmasked energy conserved) while every masked cell stays bit-static
# (the rigid-conductor fluid freeze) and interior conduction stays alive
# (the Gaussian core measurably flattens); the STEPPED polyline
# exercises r- AND z-normal interface faces plus the interior-metal
# face drop, and the axis column gates the r = 0 parity origin
# composing with the shaped wall
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_thermal_zero_flux  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_thermal  # inputs
    "analysis_mhd_wall_thermal.py diags zero_flux"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# mode "outflow_limited" (RENAMED from temperature -- a capped one-sided
# drain is not a temperature pin): the interface faces exchange conductively against
# the T_wall reservoir, draining the interior strictly monotonically
# through a one-sided sink (the scraper-analog wall) -- the
# wall-adjacent ring hugs T_wall without undershoot, the step ledge
# drains through its z-normal interface faces too, the masked band
# stays bit-static, and the axis column stays clean
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_thermal_outflow_limited  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_thermal implicit_mhd.wall_thermal_bc=outflow_limited implicit_mhd.wall_temperature=2.0"  # inputs
    "analysis_mhd_wall_thermal.py diags outflow_limited"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# mode "dirichlet": the interface faces PIN the wall temperature (the
# The reference code's t_bc='d' contract) -- a two-sided exchange at the half-cell
# Dirichlet distance, free-streaming capped in both directions; T_wall
# = 4 eV (distinct from the temperature twin's 2 eV, proving the mode
# plumbing end-to-end) under the same rigidity/ledger-closure/axis gates
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_thermal_dirichlet  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_thermal implicit_mhd.wall_thermal_bc=dirichlet implicit_mhd.wall_temperature=4.0"  # inputs
    "analysis_mhd_wall_thermal.py diags dirichlet"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# dielectric standoff, FLUID-contract regression: the temperature-mode
# thermal wall run again under wall_model=dielectric (which requires the
# split external fields; the drive is zero) must satisfy the identical
# assertions -- rigid bit-static masked band, strictly monotonic
# one-sided interior drain, ledger closure, T_wall anchoring, z-normal
# ledge drain, axis health -- proving the pec_response fluid contract
# carried over unchanged while the field side went EM-transparent
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_thermal_dielectric  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_thermal implicit_mhd.wall_model=dielectric implicit_mhd.wall_thermal_bc=outflow_limited implicit_mhd.wall_temperature=2.0 hybrid_pic_model.add_external_fields=1 external_vector_potential.fields=hold external_vector_potential.do_diva_cleaning=0"  # inputs
    "analysis_mhd_wall_thermal.py diags outflow_limited"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Braginskii clamp CLASS of the shaped-wall interface conductance
# (implicit_mhd.wall_conduction_scale), DEFAULT/legacy twin: a hot
# uniform column inside a straight cylinder wall under the HARD
# dirichlet pin (no free-streaming cap, no one-sided gate -- the only
# thing the knob touches is the chi that multiplies the exchange), with
# B TANGENTIAL to the wall so every radial face has b_n = 0 and the
# interior tensor nn projection is chi_perp exactly. Both per-component
# clamps ride their (production-ratio, 1e4 apart) caps over the whole
# run, so with theta = 1 each mode's conduction subsystem is EXACTLY a
# linear backward-Euler cylindrical Dirichlet column and the analysis
# integrates the identical discrete operator. "perp" is the default:
# the wall face conducts at the interior coefficient
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_conduction_scale_perp  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_conduction_scale  # inputs
    "analysis_mhd_wall_conduction_scale.py diags perp"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# ...and "parallel" (the reference-matched wall conductance G, whose
# implicit temperature solve builds G from the PARALLEL conductivity
# clamp maximum xile_mx = xili_mx at every cut-cell node): BOTH species
# channels must land on the 1e4-stronger column, and the measured
# first-step parallel/perp drain separation must be the analytic clamp
# ratio -- the quantitative gate on the reference-parity fix
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_conduction_scale_parallel  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_conduction_scale implicit_mhd.wall_conduction_scale=parallel"  # inputs
    "analysis_mhd_wall_conduction_scale.py diags parallel ../test_rz_theta_implicit_mhd_wall_conduction_scale_perp/diags"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_conduction_scale_perp  # dependency
)

# PRECONDITIONER ROWS for the wall thermal terms, ROW CORRECTNESS 1/2 --
# the shaped-wall interface drain. The same saturated-clamp uniform
# column as the wall_conduction_scale pair (whose conduction subsystem is
# an exactly known backward-Euler cylindrical Dirichlet column), now with
# pc_mhd_block on and the emitted rows graded IN-RUN against the analytic
# half-cell rate: theta dt D = 2 chi_op dt/dr^2 * r_face/r_cell
# = 0.25 * (0.0625/0.05625) * 1e-4 = 2.777777778e-05 at the deck's
# default perp clamp class, the cylindrical metric
# included. wall_conduction_validate_rows separately aborts if the
# emitted conductance stops factoring the residual's own drain to 1e-12.
# The physics assertions are the perp arm's, unchanged: a preconditioner
# may not move the converged answer.
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_conduction_rows  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_conduction_scale jacobian.pc_type=pc_mhd_block implicit_mhd.wall_conduction_validate_rows=1 implicit_mhd.wall_conduction_validate_rows_reference=2.777777778e-05"  # inputs
    "analysis_mhd_wall_conduction_scale.py diags perp"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# preconditioner rows, the ITERATION-COUNT gate -- BASELINE twin
# (implicit_mhd.wall_conduction_pc_rows = 0 on the CLI): the row-less
# preconditioner whose cumulative GMRES count the headline is graded
# against
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_conduction_pc_off  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_conduction_pc implicit_mhd.wall_conduction_pc_rows=0"  # inputs
    "analysis_mhd_wall_conduction_pc.py diags/diag000006 diags/newton.txt off"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# preconditioner rows, the HEADLINE: a z-shaped temperature spreads the
# wall-face Braginskii conductance over decades along the contour (the
# production band, not one degenerate ring), both wall terms are live
# (the dirichlet interface drain at the PARALLEL clamp class and the
# z-end conductive exchange), and the emitted rows must cut the
# cumulative GMRES count against the row-less twin at IDENTICAL Newton
# work and an unmoved converged state
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_conduction_pc  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_conduction_pc  # inputs
    "analysis_mhd_wall_conduction_pc.py diags/diag000006 diags/newton.txt on ../test_rz_theta_implicit_mhd_wall_conduction_pc_off/diags/diag000006 ../test_rz_theta_implicit_mhd_wall_conduction_pc_off/diags/newton.txt"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_conduction_pc_off  # dependency
)

# wall-drain gate anchored at the reachable set: with wall_temperature
# far BELOW the electron/ion temperature floors (the ratchet engaged
# from step 0) and dt far above the wall-face free-streaming time, a
# reservoir-anchored drain gate never closes while the projection pins
# the same components at the floor bound -- iteration-0 frozen steps
# within a handful of steps and the max_frozen_steps abort (the
# reproducer contract, documented in the deck). With the gate anchored
# at max(reservoir, floor image) the run completes with zero frozen
# solves and the wall-adjacent ring settles AT the floor from above, in
# the electron channel and the dual_energy twin pair alike
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_floor_gate  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_floor_gate  # inputs
    "analysis_mhd_wall_floor_gate.py diags/diag000000 diags/diag000016"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Reference-code-style wall viscosity band, default (zero) coefficient
# (implicit_mhd.wall_viscosity_mask): a sheared column u_z = A r inside
# a cylinder wall must deposit the discretely exact uniform shear
# heating 2 rho nu A^2 in every live-viscosity cell while every face
# within wall_viscosity_mask_width of the masked contour carries zero
# viscous stress AND zero stress work -- zero viscous deposit at the
# masked-adjacent rows, unchanged elsewhere
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_viscosity_mask  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_viscosity_mask  # inputs
    "analysis_mhd_wall_viscosity_mask.py diags/diag000000 diags/diag000003"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# wall viscosity BAND VALUE, regime 1/3 -- UNMASKED reference
# (wall_viscosity_mask = 0 on the CLI): the band cells must heat at the
# same rho nu deposit as the interior, fixing the "no band at all"
# analytic coefficient the other two regimes are measured against
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_viscosity_band_unmasked  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_viscosity_band implicit_mhd.wall_viscosity_mask=0 implicit_mhd.wall_viscosity_band_value=0"  # inputs
    "analysis_mhd_wall_viscosity_band.py diags/diag000000 diags/diag000003 unmasked"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# wall viscosity BAND VALUE, regime 2/3 -- ZERO band (band value 0 on
# the CLI, the DEFAULT): the legacy exact-zero band must be preserved
# unchanged by the coefficient-substitution rework -- no viscous
# deposit at all on the band rows, full deposit in the interior
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_viscosity_band_zero  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_viscosity_band implicit_mhd.wall_viscosity_band_value=0"  # inputs
    "analysis_mhd_wall_viscosity_band.py diags/diag000000 diags/diag000003 zero"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# wall viscosity BAND VALUE, regime 3/3 -- the reference code's PEDESTAL (band value
# 1e-4 Pa s = small_vis, step.f90:90): the band faces must carry that
# ABSOLUTE dynamic viscosity, not zero and not rho nu, so cells 13-14
# heat at the exact 2 mu_band A^2 rate (29.9x the interior here) while
# the live interior rows keep their own rho nu deposit
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_viscosity_band  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_viscosity_band  # inputs
    "analysis_mhd_wall_viscosity_band.py diags/diag000000 diags/diag000003 band"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Reference-parity NO-SLIP PIN, NULL TWIN (implicit_mhd.wall_no_slip=0 on
# the CLI): today's free-slip stair wall -- the masked-side image copies
# the tangential momentum verbatim, so the wall-adjacent rows slide at
# rho A r with no pin anywhere
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_no_slip_off  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_no_slip implicit_mhd.wall_no_slip=0"  # inputs
    "analysis_mhd_wall_no_slip.py diags/diag000003 off"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Reference-parity NO-SLIP PIN, knob ON (width 2 = their wall + skin
# depth): the first two LIVE cell rows adjacent to the contour must
# carry EXACTLY zero momentum in all three components, the pin must
# deposit real wall drag into the first unpinned row (the transport
# channel free slip does not have), and the far interior must match the
# null twin to solver precision
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_no_slip  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_no_slip  # inputs
    "analysis_mhd_wall_no_slip.py diags/diag000003 on ../test_rz_theta_implicit_mhd_wall_no_slip_off/diags/diag000003"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_no_slip_off  # dependency
)

# no-slip pin preconditioner parity, NULL TWIN (wall_no_slip=0 on the
# CLI): the healthy pc_mhd_block baseline on the live-fluid shaped-wall
# deck, whose GMRES cost per Newton iteration the pinned run is graded
# against
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_no_slip_pc_off  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_no_slip_pc implicit_mhd.wall_no_slip=0"  # inputs
    "analysis_mhd_wall_no_slip_pc.py diags/newton.txt off"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# no-slip pin preconditioner parity, the headline: pc_mhd_block must
# emit identity rows for exactly the pinned MOMENTUM components (mask
# them out of the cell-centered sweeps, restore them from the RHS), so
# the pinned solve stays within a small factor of the null twin's GMRES
# cost -- a residual-only wall term is a measured 7-21x stagnation
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_no_slip_pc  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_no_slip_pc  # inputs
    "analysis_mhd_wall_no_slip_pc.py diags/newton.txt on ../test_rz_theta_implicit_mhd_wall_no_slip_pc_off/diags/newton.txt"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_no_slip_pc_off  # dependency
)

# wall-seam guard on the Hall/inertia/hyper Ohm terms: a ramped coil
# blob against the STEP corner of the pec stair wall forces the image
# current sheet along the seam; the seam band (rows whose non-ideal Ohm
# stencils read across the stair) must stay at the eta J /
# ideal-dissipation scale instead of the measured drive-powered Hall
# pump (unguarded: seam E_theta 33 V/m = the global max, seam E_r
# 69 V/m; guarded: 2.2 / 0.09 V/m), the interior Hall response must
# stay alive (over-covering guard fails), the masked band stays frozen
# to solver precision, and the in-run resistive-block assembly
# validation asserts the seam-guard row drops of the banded PC against
# the matrix-free operator to roundoff at every update
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_hall_seam  # name
    RZ  # dims
    1  # nprocs (the banded resistive solver is single-rank)
    inputs_test_rz_theta_implicit_mhd_wall_hall_seam  # inputs
    "analysis_mhd_wall_hall_seam.py diags/diag000000 diags/diag000010"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# dielectric standoff, transparency reference twin: the seeded-loop deck
# with NO wall at all -- the probe-ring flux linkage deep inside the
# (would-be) band must come out finite, and the band region's fluid must
# evolve (the teeth of the dielectric run's bit-static check)
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_transparency_none  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_dielectric implicit_mhd.wall_model=none implicit_mhd.wall_thermal_bc=none"  # inputs
    "analysis_mhd_wall_dielectric.py diags/diag000000 diags/diag000030 none"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# dielectric standoff (EM-transparent formation tube), the headline
# transparency test: the seeded plasma-response loop must link the probe
# rings deep inside the masked band FINITELY and identically to the
# wall_model=none twin (no response pinning, no PC row drops -- asserted
# in-run by the resistive assembly validation -- and no wall-seam guard:
# Hall is on, so a wrongly-engaged guard breaks the twin match), while
# the masked-band FLUID state stays bit-static (the pec_response rigid
# freeze, zero_flux thermal wall)
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_dielectric  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_dielectric  # inputs
    "analysis_mhd_wall_dielectric.py diags/diag000000 diags/diag000030 dielectric ../test_rz_theta_implicit_mhd_wall_transparency_none/diags/diag000030"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_transparency_none  # dependency
)

# shaped-wall LIVE-fluid preconditioner parity, the unpreconditioned
# Reference twin: healthy Newton on the freeze contract without a PC
# (every other wall test runs frozen ions; the freeze/PC interaction
# only exists with a live fluid)
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_live_fluid_nopc  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_live_fluid_pc jacobian.pc_type=none"  # inputs
    "analysis_mhd_wall_live_fluid_pc.py diags/newton.txt diags/diag000005 reference"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# shaped-wall LIVE-fluid preconditioner parity, the headline:
# pc_mhd_block must mirror the rigid-conductor fluid freeze (masked-band
# identity rows) -- a PC that feeds the band into its fluid sweeps
# stalls GMRES at the iteration cap on every wall solve and the
# stagnated updates imprint a drive-scale B seam along the wall contour
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_live_fluid_pc  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_live_fluid_pc  # inputs
    "analysis_mhd_wall_live_fluid_pc.py diags/newton.txt diags/diag000005 pc ../test_rz_theta_implicit_mhd_wall_live_fluid_nopc/diags/diag000005"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_live_fluid_nopc  # dependency
)

# pec_response contrast on the same deck (the adversarial twin): the
# one-way-glass conductor pins the identical probe-ring linkage at
# zero/roundoff -- the measured coil-contour back-EMF kill the
# dielectric mode exists to remove
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_dielectric_pinned_contrast  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_dielectric implicit_mhd.wall_model=pec_response"  # inputs
    "analysis_mhd_wall_dielectric.py diags/diag000000 diags/diag000030 pec_response"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# shaped-wall EXTERIOR CLAMP (the band-hygiene contract, T5 in
# miniature): a deliberately contaminated IC -- uniform rho0 (2000x the
# mass floor) plus painted band momentum -- must be scraped to the
# rigid vacuum image at first-step sanitize time (ledger header closes
# against the measured removal), stay BIT-IDENTICAL there with the
# halo pedestal, the temperature floors and the floor-consistency
# supply all armed (every repopulation path must skip the band), and
# leave the quiet interior equilibrium untouched
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_dielectric_clamp  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_dielectric_clamp  # inputs
    "analysis_mhd_wall_dielectric_clamp.py diags"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# wall-band eta override, the no-knob CONTRAST (the T5 repro in
# miniature): with a density-independent constant eta the clamped
# frozen band stays electrically conductive -- the uniform-Bz ramp's
# shielding response books a deep-band |j_theta| DOMINATING the
# interior response (the screening teeth of the override run's
# collapse check)
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_band_eta_conductive  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_band_eta implicit_mhd.wall_band_eta_override=0.0"  # inputs
    "analysis_mhd_wall_band_eta.py diags/diag000006 reference"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# wall-band eta override (implicit_mhd.wall_band_eta_override), the
# headline: the constant state-independent band eta at mu0*Dwall (500x
# the plasma eta; bit-equal across the band, asserted from the plotted
# PC eta register) must collapse the deep-band eddy |j_theta| below
# 1e-2 of the no-knob twin (the production bar from the run_v8_wall
# frozen-step abort) while the live interior RECOVERS the shielding
# (interface rows keep the composed eta) and the driven window stays
# Newton-grind-free; the in-run resistive assembly validation asserts
# the overridden residual/PC row twin-ness at every update
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_band_eta_override  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_wall_band_eta  # inputs
    "analysis_mhd_wall_band_eta.py diags/diag000006 override ../test_rz_theta_implicit_mhd_wall_band_eta_conductive/diags/diag000006"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_band_eta_conductive  # dependency
)

# shaped-wall FIELD freeze (implicit_mhd.wall_field_freeze, work item
# B), the headline T-B1: the conductive contrast deck (band-eta
# override OFF -- band |j_theta| ~3.2e4 dominating the interior) with
# the freeze armed must book a deep-band current that is exactly zero
# (round-off; bit-frozen Br/Bt at the zero boot response) at every
# post-boot snapshot while the interior response RECOVERS the
# shielding; the in-run resistive assembly validation gates the
# frozen-identity-row residual/PC twin-ness at every update
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_field_freeze  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_band_eta implicit_mhd.wall_field_freeze=1 implicit_mhd.wall_band_eta_override=0.0 diag1.intervals=2"  # inputs
    "analysis_mhd_wall_field_freeze.py diags ../test_rz_theta_implicit_mhd_wall_band_eta_conductive/diags/diag000006"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_band_eta_conductive  # dependency
)

# shaped-wall FIELD freeze no-op twin (T-B2): quiescent no-drive
# conduction-drain run (the pec thermal deck -- every masked E row is
# already projected to zero, so every frozen face's Faraday rhs is
# exactly zero before the freeze zeroes it) with the freeze armed must
# stay BIT-IDENTICAL to the freeze-off reference at every snapshot and
# every plotted field: the freeze must never touch a live row
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_field_freeze_noop  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_wall_thermal implicit_mhd.wall_field_freeze=1"  # inputs
    "analysis_mhd_wall_field_freeze_noop.py diags ../test_rz_theta_implicit_mhd_wall_thermal_zero_flux/diags"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_thermal_zero_flux  # dependency
)

# shaped-wall FIELD freeze live-fluid PC parity, the unpreconditioned
# truth (T-B3 reference): the live-fluid wall deck with the freeze
# armed and no preconditioner
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_field_freeze_live_nopc  # name
    RZ  # dims
    1  # nprocs (matches the banded single-rank twin)
    "inputs_test_rz_theta_implicit_mhd_wall_live_fluid_pc jacobian.pc_type=none implicit_mhd.wall_field_freeze=1"  # inputs
    "analysis_mhd_wall_live_fluid_pc.py diags/newton.txt diags/diag000005 reference"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# shaped-wall FIELD freeze live-fluid PC parity (T-B3): pc_mhd_block
# with the BANDED resistive block and in-run assembly validation must
# mirror the residual's frozen-face identity rows (identity stencil
# rows + the post-inverse dB = r_B restore, the field twin of the
# fluid wall_live restore) -- healthy GMRES, and the converged answer
# matches the unpreconditioned truth pointwise
add_warpx_test(
    test_rz_theta_implicit_mhd_wall_field_freeze_live_pc  # name
    RZ  # dims
    1  # nprocs (the banded elimination is sequential along z: single rank)
    "inputs_test_rz_theta_implicit_mhd_wall_live_fluid_pc implicit_mhd.wall_field_freeze=1 pc_mhd_block.resistive_solver=banded pc_mhd_block.resistive_validate_assembly=1"  # inputs
    "analysis_mhd_wall_live_fluid_pc.py diags/newton.txt diags/diag000005 pc ../test_rz_theta_implicit_mhd_wall_field_freeze_live_nopc/diags/diag000005"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_wall_field_freeze_live_nopc  # dependency
)

# selectable z-end fluid boundary, mode "outflow" (rectified outgoing-only
# end ghosts): a hot slab with outward end flow drains mass and total
# fluid energy strictly monotonically through the z faces
add_warpx_test(
    test_rz_theta_implicit_mhd_z_end_outflow  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_end_outflow  # inputs
    "analysis_mhd_z_end_outflow.py diags drain"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# same configuration with the end flow reversed (interior pulls away from
# the ends): the rectified ghosts must not inject -- the plain Neumann
# clamp would feed plasma back in at interior values
add_warpx_test(
    test_rz_theta_implicit_mhd_z_end_outflow_noinject  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_z_end_outflow my_constants.u_dir=-1.0"  # inputs
    "analysis_mhd_z_end_outflow.py diags noinject"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# selectable z-end fluid boundary, mode "wall_temperature": a hot static
# slab against cold end walls loses energy monotonically through the ends
# at an advective-exchange rate while the midplane stays hot
add_warpx_test(
    test_rz_theta_implicit_mhd_z_wall_temperature  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_wall_temperature  # inputs
    "analysis_mhd_z_wall_temperature.py diags cold"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# equilibrium gate of the wall-temperature ends: a slab AT the wall
# temperature has ghosts equal to the interior and must stay unchanged
# to solver roundoff
add_warpx_test(
    test_rz_theta_implicit_mhd_z_wall_temperature_equilibrium  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_z_wall_temperature my_constants.Twall_ev=100.0"  # inputs
    "analysis_mhd_z_wall_temperature.py diags equilibrium"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# conductive z-end exchange, NULL TWIN (implicit_mhd.z_wall_conduction=0
# on the CLI): the legacy wall_temperature end faces -- the plain
# Braginskii tensor branch against the wall-image ghost at the ONE-CELL
# Neumann-ghost distance. The twin must match the one-cell Dirichlet
# diffusion column exactly (the analysis integrates the identical
# backward-Euler operator), proving the knob's default preserves the
# legacy end-face behavior quantitatively
add_warpx_test(
    test_rz_theta_implicit_mhd_z_wall_conduction_off  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_z_wall_conduction implicit_mhd.z_wall_conduction=0"  # inputs
    "analysis_mhd_z_wall_conduction.py diags off"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# conductive z-end exchange, knob ON: both end faces REPLACE the plain
# branch with the hard half-cell Dirichlet exchange
# chi_nn rho (e - e_wall) 2/dz at the cap-pinned chi_par (bn = 1 -> the
# PARALLEL channel), in both energy channels; the run must match the
# half-cell pin column (2x the legacy end-face exchange) and its end
# cells must land materially below the null twin's
add_warpx_test(
    test_rz_theta_implicit_mhd_z_wall_conduction  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_wall_conduction  # inputs
    "analysis_mhd_z_wall_conduction.py diags on ../test_rz_theta_implicit_mhd_z_wall_conduction_off/diags"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_z_wall_conduction_off  # dependency
)

# PRECONDITIONER ROWS for the wall thermal terms, ROW CORRECTNESS 2/2 --
# the z-end conductive exchange. The same saturated-clamp uniform column
# as the pair above, whose end-face Dirichlet rate the deck fixes at
# 2 chi_op dt/dz^2 = 1 exactly, now with pc_mhd_block on: the emitted row
# must be theta dt D = 1 (z faces carry no cylindrical weight -- the
# other half of the geometry the factorization check cannot see), and
# wall_conduction_validate_rows aborts if the emitted conductance stops
# factoring the residual's own end-face drain. Physics assertions are the
# knob-ON arm's, unchanged.
add_warpx_test(
    test_rz_theta_implicit_mhd_z_wall_conduction_rows  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_z_wall_conduction jacobian.pc_type=pc_mhd_block implicit_mhd.wall_conduction_validate_rows=1 implicit_mhd.wall_conduction_validate_rows_reference=1.0"  # inputs
    "analysis_mhd_z_wall_conduction.py diags on ../test_rz_theta_implicit_mhd_z_wall_conduction_off/diags"  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_z_wall_conduction_off  # dependency
)

# staggering-aware Neumann z-ghost fill: nodal-in-z fields must mirror
# evenly about the boundary node (zero centered end-node gradient under a
# z-linear ramp) while cell-centered fields keep the bit-identical clamp;
# the ghost rows are read back directly through the MultiFab register
add_warpx_test(
    test_rz_theta_implicit_mhd_z_nodal_ghost_picmi  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_z_nodal_ghost_picmi.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_theta_implicit_mhd_te_primary_cliff_picmi  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_theta_implicit_mhd_te_primary_cliff_picmi.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

# circuit-in-the-residual coupling, hook-scope reference: a toy driven
# RL loop rides the "externalcoiltheta" hook on EVERY residual
# evaluation (implicit_mhd.circuit_hook_scope = residual, the
# bit-compatible default) and records the per-step hook counts and
# committed coil scales
add_warpx_test(
    test_rz_theta_implicit_mhd_circuit_hook_residual  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_theta_implicit_mhd_circuit_hook_picmi.py --scope residual"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

# newton-scope twin: the hook fires ONLY at accepted Newton iterates
# (Jacobian probes and line-search trials reuse the cached coil
# scales); the analysis asserts the committed per-step coil scales
# match the residual-scope run to solver tolerance while the hook-call
# counts collapse
add_warpx_test(
    test_rz_theta_implicit_mhd_circuit_hook_newton  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_theta_implicit_mhd_circuit_hook_picmi.py --scope newton"  # inputs
    analysis_mhd_circuit_hook_scope.py  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_circuit_hook_residual  # dependency
)

# ---------------------------------------------------------------------
# TVD reconstruction of the recast face states
# (implicit_mhd.fluid_reconstruction; the reference code's adv_type = 4 median
# limiter). Three gate families sharing two base decks.
# ---------------------------------------------------------------------

# ORDER OF ACCURACY. A sine density perturbation advected as an exact
# contact on 32/64/128 cells, with dt held FIXED across the family (keyed
# to the finest grid) so the theta = 0.5 temporal error is a common
# constant and the measured order is purely spatial. Five twins carry the
# refinement family; the sixth arm's analysis reads all six and asserts
# first order OFF (donor cell) against second order ON, and reports the
# effective diffusivity of each arm against the audit's c_f dx / 2.
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_order_none_n32  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_order my_constants.nz=32 implicit_mhd.fluid_reconstruction=none"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_order_none_n64  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_order my_constants.nz=64 implicit_mhd.fluid_reconstruction=none"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_order_none_n32  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_order_none_n128  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_order my_constants.nz=128 implicit_mhd.fluid_reconstruction=none"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_order_none_n64  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_order_median_n32  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_order my_constants.nz=32 implicit_mhd.fluid_reconstruction=median"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_order_none_n128  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_order_median_n64  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_order my_constants.nz=64 implicit_mhd.fluid_reconstruction=median"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_order_median_n32  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_order  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_order my_constants.nz=128 implicit_mhd.fluid_reconstruction=median"  # inputs
    "analysis_mhd_reconstruction_order.py diags/diag000000 diags/diag000100"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_order_median_n64  # dependency
)

# TVD / NO NEW EXTREMA. A 4:1 square contact advected 0.4 domain lengths
# with hlld. The median limiter must hold the invariant interval while
# the UNLIMITED twin (the same reconstruction with the limiter removed)
# rings -- without that twin the gate would be vacuous, since donor cell
# is monotone by construction too. Also gates that the reconstruction
# never manufactures an inadmissible ion internal energy.
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_tvd_none  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_tvd implicit_mhd.fluid_reconstruction=none"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_tvd_vanalbada  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_tvd implicit_mhd.fluid_reconstruction=vanalbada"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_tvd_none  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_tvd_unlimited  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_tvd implicit_mhd.fluid_reconstruction=unlimited"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_tvd_vanalbada  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_tvd  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_tvd implicit_mhd.fluid_reconstruction=median"  # inputs
    "analysis_mhd_reconstruction_tvd.py diags/diag000000 diags/diag000120"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_tvd_unlimited  # dependency
)

# DEFAULT-PATH IDENTITY. The same deck with the knob absent and with
# fluid_reconstruction = none must agree BITWISE: the new branch sits on
# the JFNK residual hot path of every production deck, and the whole
# committed checksum suite depends on the default not having moved.
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_identity_default  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_tvd my_constants.nz=32 max_step=50 diag1.intervals=50"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
add_warpx_test(
    test_1d_theta_implicit_mhd_reconstruction_identity  # name
    1  # dims
    1  # nprocs
    "inputs_base_1d_theta_implicit_mhd_reconstruction_tvd my_constants.nz=32 max_step=50 diag1.intervals=50 implicit_mhd.fluid_reconstruction=none"  # inputs
    "analysis_mhd_reconstruction_identity.py diags/diag000050"  # analysis
    OFF  # checksum
    test_1d_theta_implicit_mhd_reconstruction_identity_default  # dependency
)

# Test-fixture ExternalCircuit plugin (independent driven RL loops, the
# exact backward-Euler map of the python-hook reference) for the native
# circuit-driver tests: a self-contained MODULE library dlopen'd through
# circuit.plugin_library. Links nothing (the ABI header pulls only the
# amrex::Real typedef); ablastr provides the include paths.
if(RZ IN_LIST WarpX_DIMS)
    warpx_set_suffix_dims(_rl_plugin_sd RZ)
    add_library(rl_test_circuit MODULE rl_test_circuit.cpp)
    target_link_libraries(rl_test_circuit PRIVATE ablastr_${_rl_plugin_sd})
    target_include_directories(rl_test_circuit
        PRIVATE ${WarpX_SOURCE_DIR}/Source)
    set_target_properties(rl_test_circuit PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
        PREFIX "lib"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

# native-driver twin of the residual-scope reference: the SAME coupled
# RL map advanced by the compiled plugin behind the C++ circuit-coupling
# engine (implicit_mhd.circuit_driver = native), measuring with the
# engine's batched device disk probe instead of the python hook; the
# analysis asserts the committed per-step coil scales and linkages match
# the python driver to solver tolerance
add_warpx_test(
    test_rz_theta_implicit_mhd_circuit_native  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_theta_implicit_mhd_circuit_hook_picmi.py --scope residual --driver native"  # inputs
    analysis_mhd_circuit_native_parity.py  # analysis
    OFF  # checksum
    test_rz_theta_implicit_mhd_circuit_hook_residual  # dependency
)
# multi-coil batched-probe equivalence (5 disk rows, 2 ranks, multi-box
# grids): circuit.probe_crosscheck re-measures every batched linkage
# through the single-coil reference probes and aborts on disagreement
add_warpx_test(
    test_rz_theta_implicit_mhd_circuit_crosscheck  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_circuit_hook_picmi.py --scope residual --driver native --extra-coils 4 --crosscheck"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
# reciprocity-probe twin of the crosscheck under the Green's-function
# open r_hi boundary (ring-kernel unit fields on coils outside the
# wall): exercises the batched reciprocity path
add_warpx_test(
    test_rz_theta_implicit_mhd_circuit_crosscheck_reciprocity  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_theta_implicit_mhd_circuit_hook_picmi.py --scope residual --driver native --extra-coils 2 --crosscheck --r-open"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)


add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_cropping  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_cropping  # inputs
    "analysis_vandb_jfnk_2d_cropping.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_filtered  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_filtered  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_picmi.py  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_symmetry  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_symmetry  # inputs
    "analysis_implicit.py diags/diag1000400"  # analysis
    "analysis_default_regression.py --path diags/diag1000400"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_2d_theta_implicit_strang_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_theta_implicit_strang_psatd  # inputs
        "analysis_2d_psatd.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )
endif()
