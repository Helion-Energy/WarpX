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

add_warpx_test(
    test_rz_circuit_lenz_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_circuit_lenz_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_circuit_restart_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_circuit_restart_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_circuit_restart_picmi_restart  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_circuit_restart_picmi.py amr.restart='../test_rz_circuit_restart_picmi/diags/chk000004'"  # inputs
    OFF  # analysis
    OFF  # checksum
    test_rz_circuit_restart_picmi  # dependency
)

add_warpx_test(
    test_rz_circuit_rl_discharge_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_circuit_rl_discharge_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
