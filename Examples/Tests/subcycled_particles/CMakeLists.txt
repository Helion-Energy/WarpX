# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_subcycled_gyro_ring  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_subcycled_gyro_ring  # inputs
    "analysis_gyro_ring.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
