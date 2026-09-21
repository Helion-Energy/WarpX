# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_theta_implicit_planar_pinch  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_planar_pinch  # inputs
    "analysis_pinch.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_semi_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_semi_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_theta_implicit_picard  # name
    1  # dims
    2  # nprocs
    inputs_test_1d_theta_implicit_picard  # inputs
    "analysis_1d.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

if(AMReX_PETSC)
    add_warpx_test(
        test_2d_curl_curl_petsc_pc  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_curl_curl_petsc_pc  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1001000"  # analysis
        "analysis_default_regression.py --path diags/diag1001000"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_curl_curl_petsc_pc  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_curl_curl_petsc_pc  # inputs
        "analysis_petsc_matrix.py diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_theta_implicit_planar_pinch  # name
        2  # dims
        2  # nprocs
        inputs_test_2d_theta_implicit_planar_pinch  # inputs
        "analysis_pinch.py diags/diag1000020"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rcylinder_theta_implicit_dynamic_pinch  # name
        RCYLINDER  # dims
        2  # nprocs
        inputs_test_rcylinder_theta_implicit_dynamic_pinch  # inputs
        "analysis_pinch.py diags/diag1000020"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_theta_implicit_dynamic_pinch  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_theta_implicit_dynamic_pinch  # inputs
        "analysis_pinch.py diags/diag1000010"  # analysis
        "analysis_default_regression.py --path diags/diag1000010"  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_cropping  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_cropping  # inputs
    "analysis_vandb_jfnk_2d_cropping.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_filtered  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_filtered  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_picmi.py  # inputs
    "analysis_vandb_jfnk_2d.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_symmetry  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_symmetry  # inputs
    "analysis_implicit.py diags/diag1000400"  # analysis
    "analysis_default_regression.py --path diags/diag1000400"  # checksum
    OFF  # dependency
)

if(WarpX_FFT)
    add_warpx_test(
        test_2d_theta_implicit_strang_psatd  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_theta_implicit_strang_psatd  # inputs
        "analysis_2d_psatd.py"  # analysis
        "analysis_default_regression.py --path diags/diag1000020"  # checksum
        OFF  # dependency
    )
endif()

# Opt-in Darwin split: regular nullspace, coupled energy and vacuum induction.
foreach(case IN ITEMS energy vacuum vacuum_energy)
    add_warpx_test(
        test_3d_darwin_split_${case}
        3 1
        inputs_test_3d_darwin_split_${case}
        "analysis_darwin_split.py ${case}"
        OFF OFF
    )
endforeach()
if(WarpX_EB)
    add_warpx_test(
        test_3d_darwin_split_eb
        3 1
        inputs_test_3d_darwin_split_eb
        "analysis_darwin_split.py eb"
        OFF OFF
    )
    add_warpx_test(
        test_3d_darwin_split_eb_vacuum
        3 1
        inputs_test_3d_darwin_split_eb_vacuum
        "analysis_darwin_split.py eb_vacuum"
        OFF OFF
    )
endif()

# Manufactured pressure BC catches GPU read/write races at PEC intersections.
if(TARGET lib_3d)
    add_executable(test_3d_pressure_pec test_pressure_pec.cpp)
    target_link_libraries(test_3d_pressure_pec PRIVATE lib_3d)
    target_include_directories(test_3d_pressure_pec PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_3d_pressure_pec)
    endif()
    add_test(NAME test_3d_darwin_pressure_pec COMMAND test_3d_pressure_pec)
    set_tests_properties(test_3d_darwin_pressure_pec PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

# Vacuum PC tests check electric gauge as well as magnetic/flux/temperature gates.
add_warpx_test(
    test_3d_darwin_split_pc_vacuum_energy
    3 1
    inputs_test_3d_darwin_split_pc_vacuum_energy
    "analysis_darwin_split.py pc_vacuum_energy"
    OFF OFF
)
if(WarpX_EB)
    add_warpx_test(
        test_3d_darwin_split_pc_eb_vacuum
        3 1
        inputs_test_3d_darwin_split_pc_eb_vacuum
        "analysis_darwin_split.py pc_eb_vacuum"
        OFF OFF
    )
endif()


# Exercise ghost exchange and duplicate ownership in the masked projection.
add_warpx_test(
    test_3d_darwin_split_pc_vacuum_mpi
    3 2
    inputs_test_3d_darwin_split_pc_vacuum_energy
    "analysis_darwin_split.py pc_vacuum_energy"
    OFF OFF
)

# A cheaper PC projection must retain strict electric-field and boundary gates.
add_warpx_test(
    test_3d_darwin_split_pc_gauge_tolerance
    3 2
    "inputs_test_3d_darwin_split_pc_vacuum_energy implicit_evolve.darwin_vacuum_gauge_pc_relative_tolerance=1e-6"
    "analysis_darwin_split.py pc_vacuum_energy"
    OFF OFF
)
if(WarpX_EB)
    add_warpx_test(
        test_3d_darwin_split_pc_gauge_tolerance_eb
        3 1
        "inputs_test_3d_darwin_split_pc_eb_vacuum implicit_evolve.darwin_vacuum_gauge_pc_relative_tolerance=1e-6"
        "analysis_darwin_split.py pc_eb_vacuum"
        OFF OFF
    )
endif()

# Mobile ions develop E_L; the diagnosed E - E_L must retain the inductive limit.
add_warpx_test(
    test_3d_darwin_split_pc_mobile
    3 1
    inputs_test_3d_darwin_split_pc_mobile
    "analysis_darwin_split.py pc_vacuum_energy 20 1e-7"
    OFF OFF
)

# Changing E_L after a near-steady field guess must not trigger divergence at
# Newton iteration zero merely because the shared step reference is tiny.
# The reproduced false-divergence case is a million-cell CUDA run. Keep
# its mesh/backend rather than claiming the small CPU case detects it.
if(WarpX_COMPUTE STREQUAL "CUDA")
    add_warpx_test(
        test_3d_darwin_split_pc_vacuum_long
        3 1
        inputs_test_3d_darwin_split_pc_vacuum_long
        "analysis_darwin_split.py pc_vacuum_energy 5 1e-7"
        OFF OFF
    )
endif()

# A residual must be a function of the current iterate, independent of the
# preceding particle push or Krylov probe. Both electron closures are covered.
if(TARGET app_3d)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_test(NAME test_3d_darwin_residual_freshness
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_darwin_residual_freshness.py
            $<TARGET_FILE:app_3d>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_residual_freshness
            ${CMAKE_CURRENT_BINARY_DIR}/residual_freshness)
    set_tests_properties(test_3d_darwin_residual_freshness PROPERTIES TIMEOUT 200)
endif()

# Algebraic variable-PC negative control and flexible reconstruction contract.
if(TARGET app_3d)
    add_executable(test_flexible_gmres test_flexible_gmres.cpp)
    target_include_directories(test_flexible_gmres PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_link_libraries(test_flexible_gmres PRIVATE AMReX::amrex_3d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_flexible_gmres)
    endif()
    add_test(NAME test_flexible_gmres COMMAND test_flexible_gmres)
    set_tests_properties(test_flexible_gmres PROPERTIES
        TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

# Nonzero installed circuit feedback, including a legacy-order negative control.
if(TARGET lib_3d)
    add_executable(test_darwin_circuit_stage test_darwin_circuit_stage.cpp)
    target_link_libraries(test_darwin_circuit_stage PRIVATE lib_3d)
    target_include_directories(test_darwin_circuit_stage PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_darwin_circuit_stage)
    endif()
    add_test(NAME test_3d_darwin_circuit_stage
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_darwin_circuit_stage.py
            $<TARGET_FILE:test_darwin_circuit_stage>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_circuit_stage
            ${CMAKE_CURRENT_BINARY_DIR}/circuit_stage)
    set_tests_properties(test_3d_darwin_circuit_stage PROPERTIES TIMEOUT 280)
endif()

# Compiled plugin oracle shared with the implicit MHD coupling tests.
if(TARGET app_3d)
    add_library(darwin_rl_test_circuit MODULE rl_test_circuit.cpp)
    target_include_directories(darwin_rl_test_circuit PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_link_libraries(darwin_rl_test_circuit PRIVATE AMReX::amrex_3d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(darwin_rl_test_circuit)
    endif()
    add_test(NAME test_3d_darwin_native_circuit
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_darwin_native_circuit.py
            $<TARGET_FILE:app_3d> $<TARGET_FILE:darwin_rl_test_circuit>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_native_circuit
            ${CMAKE_CURRENT_BINARY_DIR}/native_circuit ${WarpX_EB})
    set_tests_properties(test_3d_darwin_native_circuit PROPERTIES TIMEOUT 400)
endif()

# Device communication uses GPU-aware transfers and GPU arithmetic, including
# MPI builds without accelerator-aware reduction collectives.
if(TARGET app_3d)
    add_executable(test_circuit_device_sum test_circuit_device_sum.cpp)
    target_include_directories(test_circuit_device_sum PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_link_libraries(test_circuit_device_sum PRIVATE AMReX::amrex_3d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_circuit_device_sum)
    endif()
    add_test(NAME test_circuit_device_sum COMMAND test_circuit_device_sum)
    set_tests_properties(test_circuit_device_sum PROPERTIES
        TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
    # GPU MPI is exercised by the hardware validation runner: ordinary CI
    # MPI installations need not support device buffers.
    if(WarpX_MPI AND AMReX_GPU_BACKEND STREQUAL "NONE")
        add_test(NAME test_circuit_device_sum_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_circuit_device_sum>
                ${MPIEXEC_POSTFLAGS})
        set_tests_properties(test_circuit_device_sum_mpi PROPERTIES
            TIMEOUT 30 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endif()

# Immutable circuit response and release-build refusal tests.
if(TARGET lib_3d)
    add_executable(test_device_circuit test_device_circuit.cpp)
    target_include_directories(test_device_circuit PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_link_libraries(test_device_circuit PRIVATE lib_3d)
    add_library(darwin_rl_test_no_affine MODULE rl_test_circuit.cpp)
    target_include_directories(darwin_rl_test_no_affine PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_link_libraries(darwin_rl_test_no_affine PRIVATE AMReX::amrex_3d)
    target_compile_definitions(darwin_rl_test_no_affine PRIVATE WARPX_TEST_WITHOUT_AFFINE)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_device_circuit)
        setup_target_for_cuda_compilation(darwin_rl_test_no_affine)
    endif()
    add_test(NAME test_device_circuit
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_device_circuit.py
            $<TARGET_FILE:test_device_circuit> ${CMAKE_CURRENT_BINARY_DIR}/device_circuit)
    set_tests_properties(test_device_circuit PROPERTIES TIMEOUT 300)
    add_test(NAME test_3d_darwin_device_circuit
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_darwin_device_circuit.py
            $<TARGET_FILE:app_3d> $<TARGET_FILE:darwin_rl_test_circuit>
            $<TARGET_FILE:darwin_rl_test_no_affine>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_native_circuit
            ${CMAKE_CURRENT_BINARY_DIR}/darwin_device ${WarpX_EB})
    set_tests_properties(test_3d_darwin_device_circuit PROPERTIES TIMEOUT 1000)
endif()

if(TARGET lib_3d)
    add_executable(test_device_external_fields test_device_external_fields.cpp)
    target_include_directories(test_device_external_fields PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_link_libraries(test_device_external_fields PRIVATE lib_3d)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_device_external_fields)
    endif()
endif()
if(TARGET test_device_external_fields)
    add_test(NAME test_device_external_fields
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_device_external_fields.py
            $<TARGET_FILE:test_device_external_fields> $<TARGET_FILE:darwin_rl_test_circuit>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_native_circuit
            ${CMAKE_CURRENT_BINARY_DIR}/device_external_fields ${WarpX_EB})
    set_tests_properties(test_device_external_fields PROPERTIES TIMEOUT 150)
endif()

# Direct endpoint test: preserve nonzero enclosed charge and local divergence,
# and reconstruct physical E at wall/box intersections without field history.
if(TARGET lib_3d)
    add_executable(test_darwin_vacuum_recovery test_darwin_vacuum_recovery.cpp)
    target_link_libraries(test_darwin_vacuum_recovery PRIVATE lib_3d)
    target_include_directories(test_darwin_vacuum_recovery PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_darwin_vacuum_recovery)
    endif()
    foreach(grid IN ITEMS 8 16)
        add_test(NAME test_3d_darwin_vacuum_recovery_${grid}
            COMMAND test_darwin_vacuum_recovery
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_split_vacuum_energy
                amr.max_grid_size=${grid} amr.blocking_factor=8
                max_step=0 diagnostics.enable=0
                hybrid_pic_model.darwin_vacuum_e_check_operator=1)
        set_tests_properties(test_3d_darwin_vacuum_recovery_${grid} PROPERTIES
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET test_darwin_vacuum_recovery)
    add_test(NAME test_3d_darwin_vacuum_recovery_periodic
        COMMAND test_darwin_vacuum_recovery
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_split_vacuum_energy
            amr.max_grid_size=8 amr.blocking_factor=8 max_step=0 diagnostics.enable=0
            "boundary.field_lo=periodic periodic periodic"
            "boundary.field_hi=periodic periodic periodic"
            "boundary.particle_lo=periodic periodic periodic"
            "boundary.particle_hi=periodic periodic periodic"
            hybrid_pic_model.add_external_fields=0
            hybrid_pic_model.darwin_vacuum_recovery_mask=global
            hybrid_pic_model.darwin_vacuum_e_check_operator=1)
    set_tests_properties(test_3d_darwin_vacuum_recovery_periodic PROPERTIES
        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    if(WarpX_EB)
        add_test(NAME test_3d_darwin_vacuum_recovery_eb
            COMMAND test_darwin_vacuum_recovery
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_split_vacuum_energy
                amr.max_grid_size=8 amr.blocking_factor=8 max_step=0 diagnostics.enable=0
                "warpx.eb_implicit_function=0.06**2-x*x-y*y"
                hybrid_pic_model.use_conformal_eb=0
                hybrid_pic_model.darwin_vacuum_e_check_operator=1)
        set_tests_properties(test_3d_darwin_vacuum_recovery_eb PROPERTIES
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endif()
