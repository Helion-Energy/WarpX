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
