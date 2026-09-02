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

add_warpx_test(
    test_2d_subcycled_gyro_ring_esirkepov  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_subcycled_gyro_ring_esirkepov  # inputs
    "analysis_gyro_ring.py diags/diag1000010 --continuity"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_subcycled_gyro_ring  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_subcycled_gyro_ring  # inputs
    "analysis_gyro_ring_rz.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

# Run-only: exercises the meridional sizing path end-to-end (including the
# azimuthal rotation and the cap-as-error branch being armed), so a crash or
# an abort regression is caught in CI. The quantitative gate lives in
# analysis_vref_meridional.py, which parses the "[subcycle]" witness line and
# must therefore be run by hand against a captured log -- the test harness
# does not preserve the run's stdout in the working directory.
add_warpx_test(
    test_rz_subcycled_vref_meridional  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_subcycled_vref_meridional  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
