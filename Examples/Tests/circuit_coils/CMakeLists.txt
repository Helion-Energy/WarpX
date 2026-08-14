# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_circuit_coil_unit_field_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_circuit_coil_unit_field_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_circuit_coil_unit_field_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_circuit_coil_unit_field_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
