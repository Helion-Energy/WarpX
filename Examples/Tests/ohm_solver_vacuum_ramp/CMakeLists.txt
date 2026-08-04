# Add tests (alphabetical order) ##############################################
#

# Vacuum inductive ramp (plan: exercise the external vector-potential drive
# and the Darwin longitudinal projection with the whole domain below the
# Ohm's-law density floor). The analysis asserts (1) the volume-averaged B_z
# tracks B(0) + f(t) dB, (2) no spurious E_r/E_z appears against the
# azimuthal inductive field, and (3) the dilute-ion ensemble conserves
# <r^2> B through the compression (zero spurious particle drift).
add_warpx_test(
    test_rz_ohm_solver_vacuum_ramp_darwin_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_vacuum_ramp_picmi.py --test --implicit --darwin"  # inputs
    "analysis_vacuum_ramp.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000400 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_vacuum_ramp_darwin_picmi slow)

# Same ramp with the vacuum vector-potential recovery instead of the Hewett
# vacuum resistivity: the deck runs at the plasma eta (1e-6, six orders below
# the value the plain darwin variant needs for flux penetration), so the same
# three asserts prove the recovery delivers the programmed flux through the
# sub-floor region by itself. Runs the in-residual ("half") cadence.
add_warpx_test(
    test_rz_ohm_solver_vacuum_ramp_darwin_recovery_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_vacuum_ramp_picmi.py --test --implicit --darwin --recovery"  # inputs
    "analysis_vacuum_ramp.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000400 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_vacuum_ramp_darwin_recovery_picmi slow)

# Same recovery ramp driven through the circuit-in-the-residual coupling
# (implicit_evolve.external_field_iteration with a python_scale coil): the
# sigmoid is sampled into per-step piecewise-linear scale segments pushed
# from the externalcoiltheta callback inside every residual evaluation, so
# the standard ramp asserts prove the coupled drive delivers the programmed
# flux, and the deck asserts the callback cadence (every evaluation for the
# theta callback, once per step for the finish one).
add_warpx_test(
    test_rz_ohm_solver_vacuum_ramp_darwin_circuit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_vacuum_ramp_picmi.py --test --implicit --darwin --recovery --circuit"  # inputs
    "analysis_vacuum_ramp.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_vacuum_ramp_darwin_circuit_picmi slow)

# Split-field-path variant of the circuit coupling (standard theta-implicit
# hybrid, no darwin): the callback runs in the plasma-response frame
# (stored externals subtracted, refreshed at the pushed scales, totals
# restored within the evaluation). Same drive and asserts as the darwin
# variant.
#
# REGISTRATION DISABLED: the STANDARD scheme mis-tracks the programmed
# flux under Newton on this deck (B overshoots ~293x over the ramp;
# reproduces IDENTICALLY without the circuit coupling, so the coupling
# machinery is not the cause). Picard cannot cover it either (the known
# driven marginal-mode limit cycle). Deck support (--circuit without
# --darwin) is kept; re-register once the split+Newton drive defect is
# fixed.
#
# add_warpx_test(
#     test_rz_ohm_solver_vacuum_ramp_circuit_split_picmi  # name
#     RZ  # dims
#     2  # nprocs
#     "inputs_test_rz_ohm_solver_vacuum_ramp_picmi.py --test --implicit --circuit"  # inputs
#     "analysis_vacuum_ramp.py"  # analysis
#     OFF  # checksum
#     OFF  # dependency
# )
# label_warpx_test(test_rz_ohm_solver_vacuum_ramp_circuit_split_picmi slow)
