# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_mcc_collision_rate_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_mcc_collision_rate_picmi.py  # inputs
    "analysis_mcc_collision_rate.py"  # analysis
    # No checksum: the survivor count is driven by the collision RNG, which
    # WarpX does not reproduce run to run under GPU threading. The analysis
    # compares the decay rate against n*sigma*v_rel instead, which holds for
    # any random stream.
    OFF  # checksum
    OFF  # dependency
)
