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

# hlld variants of the closure/positivity suite: the recast must
# reproduce the exact-Riemann plateaus, the exact static contact, and
# the exact fluid-energy pairing of the hllc/rusanov paths
add_warpx_test(
    test_1d_theta_implicit_mhd_sod_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod implicit_mhd.fluid_flux=hlld"  # inputs
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
    "inputs_test_1d_theta_implicit_mhd_sod implicit_mhd.fluid_flux=hlld jacobian.pc_type=pc_mhd_block"  # inputs
    "analysis_mhd_hlld_preconditioner.py diags/diag000000 diags/diag000200 total_energy"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_sod_pe_dominated_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_sod_pe_dominated implicit_mhd.fluid_flux=hlld"  # inputs
    "analysis_mhd_sod_pe_dominated.py diags/diag000000 diags/diag000200"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_static_contact_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_static_contact implicit_mhd.fluid_flux=hlld"  # inputs
    "analysis_mhd_static_contact.py diags/diag000000 diags/diag000100 hlld"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_mhd_energy_conservation_hlld  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_theta_implicit_mhd_energy_conservation implicit_mhd.fluid_flux=hlld"  # inputs
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
    "inputs_test_rz_theta_implicit_mhd_z_outflow implicit_mhd.fluid_flux=hlld"  # inputs
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
    "inputs_test_rz_theta_implicit_mhd_z_outflow implicit_mhd.fluid_flux=hlld jacobian.pc_type=pc_mhd_block"  # inputs
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
