# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_resample_velocity_coincidence_thinning  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_resample_velocity_coincidence_thinning  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000004"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_resample_velocity_coincidence_thinning_cartesian  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_resample_velocity_coincidence_thinning_cartesian  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000004"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_leveling_thinning  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_leveling_thinning  # inputs
    "analysis.py diags/diag1000008"  # analysis
    "analysis_default_regression.py --path diags/diag1000008"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_resampling_reflecting_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_resampling_reflecting_picmi.py  # inputs
    OFF  # analysis
    # No checksum: the script asserts that the total weight is conserved to
    # round-off with reflecting faces, which is the contract under test.
    OFF  # checksum
    OFF  # dependency
)
