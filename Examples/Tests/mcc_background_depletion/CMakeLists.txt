# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_mcc_background_depletion  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_mcc_background_depletion  # inputs
    "analysis_mcc_background_depletion.py"  # analysis
    # No checksum: the ionization events are driven by the collision RNG, which
    # WarpX does not reproduce run to run under GPU threading, so a benchmark
    # would be platform-specific. The analysis asserts a conservation identity
    # that holds for any random stream instead, which is the actual contract.
    OFF  # checksum
    OFF  # dependency
)
