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

# Regression guard against building the reference speed by summing the
# meridional and azimuthal maxima. Those are independent reductions, attained
# in general by different particles, so their sum can exceed the speed of
# every particle present -- and hence exceed the full-speed answer the option
# exists to improve on. The two particles here carry equal speeds, one purely
# meridional and one purely azimuthal, which is the configuration that
# separates the two forms most sharply. Quantitative check in
# analysis_vref_crossparticle.py, run by hand for the reason noted above.
add_warpx_test(
    test_rz_subcycled_vref_crossparticle  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_subcycled_vref_crossparticle  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)
