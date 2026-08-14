# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_ohm_solver_external_wall_ghost_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_external_wall_ghost_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
