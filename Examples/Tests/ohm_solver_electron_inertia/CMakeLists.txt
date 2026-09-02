# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ohm_solver_electron_inertia_elliptic  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_ohm_solver_electron_inertia_elliptic"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
