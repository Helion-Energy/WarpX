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

add_warpx_test(
    test_rtz_ohm_solver_cylinder_compression_picmi  # name
    RTZ  # dims
    2  # nprocs
    "inputs_test_rtz_ohm_solver_cylinder_compression_picmi.py --test"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000020 --rtol 1e-6"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rtz_ohm_solver_cylinder_compression_picmi slow)

# GPU-only: this m=1 theta-mode case exercises the RTZ theta-curl operators
# through the full hybrid solver and is validated on the GPU CI / developer
# workflow. It is intentionally NOT registered for CPU builds, so it never runs
# on the CPU regression runner (the Azure ctest lane applies no label filter, so
# a label alone would not exclude it -- gating the registration is what does).
if(WarpX_COMPUTE STREQUAL "CUDA" OR WarpX_COMPUTE STREQUAL "HIP" OR WarpX_COMPUTE STREQUAL "SYCL")
    add_warpx_test(
        test_rtz_ohm_solver_cylinder_compression_theta_mode_picmi  # name
        RTZ  # dims
        2  # nprocs
        "inputs_test_rtz_ohm_solver_cylinder_compression_theta_mode_picmi.py --test"  # inputs
        "analysis_rtz_theta_mode.py"  # analysis (m=1 theta-operator exercise)
        OFF  # checksum
        OFF  # dependency
    )
    label_warpx_test(test_rtz_ohm_solver_cylinder_compression_theta_mode_picmi slow)
    label_warpx_test(test_rtz_ohm_solver_cylinder_compression_theta_mode_picmi gpu)
endif()
