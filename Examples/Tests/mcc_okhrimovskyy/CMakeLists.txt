# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_mcc_okhrimovskyy  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_mcc_okhrimovskyy  # inputs
    "analysis_mcc_okhrimovskyy.py"  # analysis
    # No checksum: the scattering is driven by the collision RNG, which WarpX
    # does not reproduce run to run under GPU threading. The analysis compares
    # the scattering angles against the distribution they are drawn from.
    OFF  # checksum
    OFF  # dependency
)
