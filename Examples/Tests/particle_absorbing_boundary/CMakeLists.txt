# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_particle_absorbing_boundary  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_particle_absorbing_boundary  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diagInst008000/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_particle_absorbing_boundary_direction_dependent_thresholds  # name
    1  # dims
    1  # nprocs
    inputs_test_1d_particle_absorbing_boundary_direction_dependent_thresholds  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diagInst008000/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_particle_boundary_flux  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_particle_boundary_flux  # inputs
    "analysis_boundary_flux.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
