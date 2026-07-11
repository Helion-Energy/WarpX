# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ohm_solver_enE_interpolation  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ohm_solver_enE_interpolation  # inputs
    "analysis_enE.py --case o2 diags/diag1000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_enE_interpolation_order4  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_ohm_solver_enE_interpolation_order4  # inputs
    "analysis_enE.py --case o4 diags/diag1000001"  # analysis
    OFF  # checksum
    OFF  # dependency
)
