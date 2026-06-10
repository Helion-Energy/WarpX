# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_ohm_solver_global_A_recovery_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_global_A_recovery_picmi.py --test"  # inputs
    "analysis.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
