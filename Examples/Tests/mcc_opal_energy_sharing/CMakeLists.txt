# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_mcc_opal_energy_sharing  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_mcc_opal_energy_sharing  # inputs
    "analysis_mcc_opal_energy_sharing.py"  # analysis
    # No checksum: the ionization events are driven by the collision RNG, which
    # WarpX does not reproduce run to run under GPU threading, so a benchmark
    # would be platform-specific. The analysis compares the outgoing energies
    # against the distribution they are drawn from instead, which holds for any
    # random stream.
    OFF  # checksum
    OFF  # dependency
)
