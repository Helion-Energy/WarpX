# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_ohm_solver_cylinder_compression_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_cylinder_compression_picmi.py --test"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010 --rtol 5e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_3d_ohm_solver_cylinder_compression_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_cylinder_compression_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_cylinder_compression_picmi.py --test"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020 --rtol 1e-6"  # output
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_cylinder_compression_picmi slow)

# Theta-implicit variants of the RZ compression test. These exercise the
# external vector-potential drive in total-field form (the step-averaged
# inductive E_ext enters the Ohm residual and the theta-Faraday advance
# carries the compression flux), plus, for the darwin variant, the ramp
# entering through the evolved vector potential.
#
# --dt-mult 0.25: Picard contracts only while the whistler phase advance
# theta*(k*l_i)**2*omega_ci*dt stays below ~2-3, and the binding k here is
# in the density-floored region OUTSIDE the plasma (n_floor = 0.05*n0
# raises the local l_i by sqrt(20)), giving ~5 at the base dt on the test
# grid. Newton converges at the base dt; Picard needs dt/4.
add_warpx_test(
    test_rz_ohm_solver_cylinder_compression_implicit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_cylinder_compression_picmi.py --test --implicit --dt-mult 0.25"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020 --rtol 2e-4"  # output
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_cylinder_compression_implicit_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_cylinder_compression_darwin_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_cylinder_compression_picmi.py --test --implicit --darwin --dt-mult 0.25"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020 --rtol 2e-4"  # output
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_cylinder_compression_darwin_picmi slow)
