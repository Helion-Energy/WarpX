# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ohm_solver_eb_diffusion_picmi  # name
    3  # dims
    1  # nprocs
    "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32"  # inputs
    "analysis.py diags"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_eb_diffusion_conformal_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal --split-z"  # inputs
    "analysis.py ../test_3d_ohm_solver_eb_diffusion_picmi/diags diags"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    test_3d_ohm_solver_eb_diffusion_picmi  # dependency
)

# Theta-implicit arm of the conformal wall: the projection runs inside every
# residual evaluation of the JFNK solve (newton.require_convergence is the
# wall-contract assertion) and the decayed eigenmode is gated against the
# analytic solution in single-run mode.
add_warpx_test(
    test_3d_ohm_solver_eb_diffusion_conformal_implicit_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal --split-z --implicit"  # inputs
    "analysis.py diags"  # analysis
    "analysis_default_regression.py --path diags/diag1000200"  # checksum
    OFF  # dependency
)
