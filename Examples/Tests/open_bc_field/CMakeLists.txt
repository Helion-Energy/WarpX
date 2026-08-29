# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_open_bc_field_hybrid_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_open_bc_field_hybrid_picmi.py"  # inputs
    "analysis_open_bc_field.py diags/diag1000150"  # analysis
    "analysis_default_regression.py --path diags/diag1000150"  # checksum
    OFF  # dependency
)
