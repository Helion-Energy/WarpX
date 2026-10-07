# Add tests (alphabetical order) ##############################################
#

# Custom Darwin keeps its own guard allocation when implicit-EM guards change.
foreach(shape IN ITEMS 2 3 4)
    add_warpx_test(
        test_1d_darwin_villasenor_guard_shape${shape}
        1 1
        "inputs_test_1d_darwin_villasenor_guards algo.particle_shape=${shape}"
        OFF OFF OFF
    )
endforeach()
add_warpx_test(
    test_1d_darwin_villasenor_guard_mpi
    1 2
    "inputs_test_1d_darwin_villasenor_guards algo.particle_shape=3"
    OFF OFF OFF
)

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
    "analysis_implicit_jfnk.py diags/diag1000020"  # analysis
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
    "analysis_implicit_jfnk.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_theta_implicit_jfnk_vandb_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_theta_implicit_jfnk_vandb_picmi.py  # inputs
    "analysis_implicit_jfnk.py diags/diag1000020"  # analysis
    "analysis_default_regression.py --path diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_theta_implicit_jfnk_direct  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_theta_implicit_jfnk_direct  # inputs
    "analysis_implicit_jfnk.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

# Exercise the stationary-particle deposition guard floor with cubic shapes.
add_warpx_test(
    test_3d_theta_implicit_jfnk_direct_zero_crossing  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_theta_implicit_jfnk_direct algo.particle_shape=3 particles.max_grid_crossings=0 max_step=1 diag1.intervals=1 newton.diagnostic_interval=1"  # inputs
    "analysis_implicit_jfnk.py diags/diag1000001"  # analysis
    OFF  # checksum
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

if(TARGET lib_rz)
    add_executable(test_darwin_vacuum_recovery_rz test_darwin_vacuum_recovery_rz.cpp)
    target_link_libraries(test_darwin_vacuum_recovery_rz PRIVATE lib_rz)
    target_include_directories(test_darwin_vacuum_recovery_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_darwin_vacuum_recovery_rz)
    endif()
    foreach(grid IN ITEMS 8 32)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_${grid})
        add_test(NAME test_rz_darwin_vacuum_recovery_${grid}
            COMMAND test_darwin_vacuum_recovery_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_darwin_vacuum_recovery
                amr.max_grid_size=${grid})
        set_tests_properties(test_rz_darwin_vacuum_recovery_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_${grid}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_implicit_thermal_rz test_implicit_thermal_rz.cpp)
    target_link_libraries(test_implicit_thermal_rz PRIVATE lib_rz)
    target_include_directories(test_implicit_thermal_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_implicit_thermal_rz)
    endif()
    foreach(conduction IN ITEMS 0 1)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_thermal_${conduction})
        add_test(NAME test_rz_implicit_thermal_${conduction}
            COMMAND test_implicit_thermal_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_implicit_thermal
                thermal_test.conduction=${conduction})
        set_tests_properties(test_rz_implicit_thermal_${conduction} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_thermal_${conduction}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;WARPX_QDSMC_COND_STATS=1")
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_implicit_stopping_rz test_implicit_stopping_rz.cpp)
    target_link_libraries(test_implicit_stopping_rz PRIVATE lib_rz)
    target_include_directories(test_implicit_stopping_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_implicit_stopping_rz)
    endif()
    foreach(filtered IN ITEMS 0 1)
    foreach(pedestal IN ITEMS 0 1)
        foreach(axial IN ITEMS periodic pec)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_stopping_${pedestal}_${axial}_${filtered})
            if(axial STREQUAL "periodic")
                set(particle_axial periodic)
            else()
                set(particle_axial reflecting)
            endif()
            add_test(NAME test_rz_implicit_stopping_${pedestal}_${axial}_${filtered}
                COMMAND test_implicit_stopping_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_implicit_stopping
                    hybrid_pic_model.density_pedestal=${pedestal}
                    warpx.use_filter=${filtered}
                    "boundary.field_lo=none ${axial}" "boundary.field_hi=pec ${axial}"
                    "boundary.particle_lo=none ${particle_axial}"
                    "boundary.particle_hi=reflecting ${particle_axial}")
            set_tests_properties(test_rz_implicit_stopping_${pedestal}_${axial}_${filtered} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_stopping_${pedestal}_${axial}_${filtered}
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_thermal_boundary_rz test_thermal_boundary_rz.cpp)
    target_link_libraries(test_thermal_boundary_rz PRIVATE lib_rz)
    target_include_directories(test_thermal_boundary_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_thermal_boundary_rz)
    endif()
    foreach(axial IN ITEMS periodic pec)
        foreach(integrator IN ITEMS ssprk2 rkl2)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_thermal_boundary_${axial}_${integrator})
            if(axial STREQUAL "periodic")
                set(particle_axial periodic)
            else()
                set(particle_axial reflecting)
            endif()
            add_test(NAME test_rz_thermal_boundary_${axial}_${integrator}
                COMMAND test_thermal_boundary_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_implicit_thermal
                    "boundary.field_lo=none ${axial}" "boundary.field_hi=pec ${axial}"
                    "boundary.particle_lo=none ${particle_axial}"
                    "boundary.particle_hi=reflecting ${particle_axial}"
                    hybrid_pic_model.qdsmc_conduction_fd_time=${integrator})
            set_tests_properties(test_rz_thermal_boundary_${axial}_${integrator} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_thermal_boundary_${axial}_${integrator}
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

# The default prescribed-drive recovery must expose the same map to full
# residuals and Jacobian probes. The frozen arm verifies test sensitivity.
if(TARGET lib_rz)
    foreach(fixture IN ITEMS recovery_probe implicit_density recovery_projection)
        add_executable(test_${fixture}_rz test_${fixture}_rz.cpp)
        target_link_libraries(test_${fixture}_rz PRIVATE lib_rz)
        target_include_directories(test_${fixture}_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_${fixture}_rz)
        endif()
        foreach(grid IN ITEMS 8 32)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_${fixture}_${grid})
            add_test(NAME test_rz_${fixture}_${grid}
                COMMAND test_${fixture}_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                    amr.max_grid_size=${grid})
            set_tests_properties(test_rz_${fixture}_${grid} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_${fixture}_${grid}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
    foreach(control IN ITEMS large_poloidal axial_wall)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_projection_${control})
        if(control STREQUAL "large_poloidal")
            set(projection_options projection_test.poloidal_scale=1.e20)
        else()
            set(projection_options "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
        endif()
        add_test(NAME test_rz_recovery_projection_${control}
            COMMAND test_recovery_projection_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe ${projection_options})
        set_tests_properties(test_rz_recovery_projection_${control} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_projection_${control}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_projection_legacy)
    add_test(NAME test_rz_recovery_projection_legacy
        COMMAND test_recovery_projection_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
            projection_test.legacy=1)
    set_tests_properties(test_rz_recovery_projection_legacy PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_projection_legacy
        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_frozen)
    add_test(NAME test_rz_recovery_probe_frozen
        COMMAND test_recovery_probe_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
            hybrid_pic_model.darwin_vacuum_recovery_live_probes=0
            probe_test.expect_frozen=1)
    set_tests_properties(test_rz_recovery_probe_frozen PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_frozen
        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

if(TARGET lib_rz)
    add_executable(test_recovery_multigrid_rz test_recovery_multigrid_rz.cpp)
    target_link_libraries(test_recovery_multigrid_rz PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_recovery_multigrid_rz)
    endif()
    foreach(grid IN ITEMS 8 32)
        add_test(NAME test_rz_recovery_multigrid_${grid}
            COMMAND test_recovery_multigrid_rz mg_test.grid=${grid})
        set_tests_properties(test_rz_recovery_multigrid_${grid} PROPERTIES
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET test_recovery_probe_rz)
    foreach(semi IN ITEMS 0 1)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_semi_${semi})
        add_test(NAME test_rz_recovery_probe_semi_${semi}
            COMMAND test_recovery_probe_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting"
                "geometry.prob_hi=0.25 2.0"
                hybrid_pic_model.darwin_vacuum_recovery_semicoarsening=${semi})
        set_tests_properties(test_rz_recovery_probe_semi_${semi} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_semi_${semi}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_hybrid_pc_rz test_hybrid_pc_rz.cpp)
    target_link_libraries(test_hybrid_pc_rz PRIVATE lib_rz)
    target_include_directories(test_hybrid_pc_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_hybrid_pc_rz)
    endif()
    add_test(NAME test_rz_hybrid_pc_outer
        COMMAND test_hybrid_pc_rz hall_test.grid=8 hall_test.hall=0.05 hall_test.outer=1)
    set_tests_properties(test_rz_hybrid_pc_outer PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    foreach(grid IN ITEMS 8 32)
        add_test(NAME test_rz_hybrid_pc_${grid}
            COMMAND test_hybrid_pc_rz hall_test.grid=${grid})
        set_tests_properties(test_rz_hybrid_pc_${grid} PROPERTIES
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_hybrid_pc_coefficients_rz test_hybrid_pc_coefficients_rz.cpp)
    target_link_libraries(test_hybrid_pc_coefficients_rz PRIVATE lib_rz)
    target_include_directories(test_hybrid_pc_coefficients_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_hybrid_pc_coefficients_rz)
    endif()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_hybrid_pc_coefficients)
    add_test(NAME test_rz_hybrid_pc_coefficients
        COMMAND test_hybrid_pc_coefficients_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe)
    set_tests_properties(test_rz_hybrid_pc_coefficients PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_hybrid_pc_coefficients
        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

if(TARGET lib_3d)
    add_executable(test_hybrid_pc_cartesian test_hybrid_pc_cartesian.cpp)
    target_link_libraries(test_hybrid_pc_cartesian PRIVATE lib_3d)
    target_include_directories(test_hybrid_pc_cartesian PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_hybrid_pc_cartesian)
    endif()
    foreach(grid_type IN ITEMS staggered collocated)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hybrid_pc_3d_${grid_type})
        add_test(NAME test_3d_hybrid_pc_${grid_type}
            COMMAND test_hybrid_pc_cartesian
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_hybrid_pc
                warpx.grid_type=${grid_type})
        set_tests_properties(test_3d_hybrid_pc_${grid_type} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hybrid_pc_3d_${grid_type}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_implicit_gather_rz test_implicit_gather_rz.cpp)
    target_link_libraries(test_implicit_gather_rz PRIVATE lib_rz)
    target_include_directories(test_implicit_gather_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_implicit_gather_rz)
    endif()
    foreach(endpoint IN ITEMS 0 1)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_implicit_gather_${endpoint})
        add_test(NAME test_rz_implicit_gather_${endpoint}
            COMMAND test_implicit_gather_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                algo.field_gathering=momentum-conserving gather_test.endpoint=${endpoint})
        set_tests_properties(test_rz_implicit_gather_${endpoint} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_implicit_gather_${endpoint}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

foreach(dimension IN ITEMS rz 3d)
    if(TARGET lib_${dimension})
        add_executable(test_viscous_work_${dimension} test_viscous_work.cpp)
        target_link_libraries(test_viscous_work_${dimension} PRIVATE lib_${dimension})
        target_include_directories(test_viscous_work_${dimension} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_viscous_work_${dimension})
        endif()
        if(dimension STREQUAL "rz")
            set(viscous_input inputs_test_rz_implicit_thermal)
        else()
            set(viscous_input inputs_test_3d_hybrid_pc)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/viscous_work_${dimension})
        add_test(NAME test_${dimension}_viscous_work
            COMMAND test_viscous_work_${dimension} ${CMAKE_CURRENT_SOURCE_DIR}/${viscous_input})
        set_tests_properties(test_${dimension}_viscous_work PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/viscous_work_${dimension}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()

if(TARGET test_hybrid_pc_cartesian)
    foreach(hall_case IN ITEMS nyquist oblique gradient)
        if(hall_case STREQUAL "nyquist")
            set(hall_mode "4 0 2")
        else()
            set(hall_mode "2 0 1")
        endif()
        if(hall_case STREQUAL "gradient")
            set(hall_gradient 1)
        else()
            set(hall_gradient 0)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hall_mode_${hall_case})
        add_test(NAME test_3d_hybrid_pc_${hall_case}
            COMMAND test_hybrid_pc_cartesian ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_hybrid_pc
                warpx.grid_type=staggered "hall_test.mode=${hall_mode}" hall_test.gradient=${hall_gradient})
        set_tests_properties(test_3d_hybrid_pc_${hall_case} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hall_mode_${hall_case}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET test_viscous_work_rz)
    foreach(viscous_boundary IN ITEMS periodic pec pmc)
        if(viscous_boundary STREQUAL "pec")
            set(viscous_particle_boundary reflecting)
        elseif(viscous_boundary STREQUAL "pmc")
            set(viscous_particle_boundary absorbing)
        else()
            set(viscous_particle_boundary periodic)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/viscous_rz_${viscous_boundary})
        add_test(NAME test_rz_viscous_work_boundary_${viscous_boundary}
            COMMAND test_viscous_work_rz ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_implicit_thermal
                viscous_test.boundary_mode=1
                "boundary.field_lo=none ${viscous_boundary}"
                "boundary.field_hi=pec ${viscous_boundary}"
                "boundary.particle_lo=none ${viscous_particle_boundary}"
                "boundary.particle_hi=reflecting ${viscous_particle_boundary}")
        set_tests_properties(test_rz_viscous_work_boundary_${viscous_boundary} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/viscous_rz_${viscous_boundary}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Exercise each axial PMC projection independently, with the other end PEC.
if(TARGET test_viscous_work_rz)
    foreach(pmc_end IN ITEMS lo hi)
        if(pmc_end STREQUAL "lo")
            set(viscous_field_lo pmc)
            set(viscous_field_hi pec)
        else()
            set(viscous_field_lo pec)
            set(viscous_field_hi pmc)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/viscous_rz_pmc_${pmc_end})
        add_test(NAME test_rz_viscous_work_boundary_pmc_${pmc_end}
            COMMAND test_viscous_work_rz ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_implicit_thermal
                viscous_test.boundary_mode=1
                "boundary.field_lo=none ${viscous_field_lo}"
                "boundary.field_hi=pec ${viscous_field_hi}"
                "boundary.particle_lo=none absorbing"
                "boundary.particle_hi=reflecting absorbing")
        set_tests_properties(test_rz_viscous_work_boundary_pmc_${pmc_end} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/viscous_rz_pmc_${pmc_end}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Closed-field energy balance exercises every cylindrical current component.
if(TARGET lib_rz)
    add_executable(test_viscous_decay_rz test_viscous_decay_rz.cpp)
    target_link_libraries(test_viscous_decay_rz PRIVATE lib_rz)
    target_include_directories(test_viscous_decay_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_viscous_decay_rz)
    endif()
    foreach(boundary IN ITEMS periodic pmc)
        if(boundary STREQUAL "periodic")
            set(particle_boundary periodic)
        else()
            set(particle_boundary absorbing)
        endif()
        foreach(refinement IN ITEMS 1 2)
            if(refinement EQUAL 1)
                set(decay_steps 20)
                set(decay_dt 1.e-10)
            else()
                set(decay_steps 40)
                set(decay_dt 5.e-11)
            endif()
            set(test_name test_rz_viscous_decay_${boundary}_${refinement})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
            add_test(NAME ${test_name}
                COMMAND test_viscous_decay_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_viscous_decay
                    viscous_decay.steps=${decay_steps} warpx.const_dt=${decay_dt}
                    "boundary.field_lo=none ${boundary}"
                    "boundary.field_hi=pec ${boundary}"
                    "boundary.particle_lo=none ${particle_boundary}"
                    "boundary.particle_hi=reflecting ${particle_boundary}")
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

# A density trough crosses the source floor at a decomposition seam.
if(TARGET test_viscous_decay_rz)
    foreach(boundary IN ITEMS periodic pmc)
        if(boundary STREQUAL "periodic")
            set(particle_boundary periodic)
        else()
            set(particle_boundary absorbing)
        endif()
        foreach(refinement IN ITEMS 1 2)
            if(refinement EQUAL 1)
                set(decay_steps 20)
                set(decay_dt 1.e-10)
            else()
                set(decay_steps 40)
                set(decay_dt 5.e-11)
            endif()
            set(test_name test_rz_viscous_decay_mask_${boundary}_${refinement})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
            add_test(NAME ${test_name}
                COMMAND test_viscous_decay_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_viscous_decay
                    viscous_decay.steps=${decay_steps} warpx.const_dt=${decay_dt}
                    "boundary.field_lo=none ${boundary}"
                    "boundary.field_hi=pec ${boundary}"
                    "boundary.particle_lo=none ${particle_boundary}"
                    "boundary.particle_hi=reflecting ${particle_boundary}"
                    ions.profile=parse_density_function
                    "ions.density_function(x,y,z)=1.e19*(0.01+0.99*0.5*(1+cos(20.943951023931955*x)))")
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

# Preserve the legacy temporal-inertia/drag certificate explicitly. With
# convection on, use a profile with no electron kinetic-energy wall flux.
if(TARGET test_viscous_decay_rz)
    foreach(boundary IN ITEMS periodic pmc)
        if(boundary STREQUAL "periodic")
            set(particle_boundary periodic)
        else()
            set(particle_boundary absorbing)
        endif()
        foreach(convection IN ITEMS 0 1)
            foreach(refinement IN ITEMS 1 2)
                if(refinement EQUAL 1)
                    set(decay_steps 20)
                    set(decay_dt 1.e-10)
                else()
                    set(decay_steps 40)
                    set(decay_dt 5.e-11)
                endif()
                if(convection)
                    set(test_name test_rz_electron_inertia_convection_energy_${boundary}_${refinement})
                else()
                    set(test_name test_rz_viscous_decay_inertia_${boundary}_${refinement})
                endif()
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
                add_test(NAME ${test_name}
                    COMMAND test_viscous_decay_rz
                        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_viscous_decay
                        viscous_decay.steps=${decay_steps} warpx.const_dt=${decay_dt}
                        "boundary.field_lo=none ${boundary}"
                        "boundary.field_hi=pec ${boundary}"
                        "boundary.particle_lo=none ${particle_boundary}"
                        "boundary.particle_hi=reflecting ${particle_boundary}"
                        viscous_decay.include_inertia_energy=1
                        viscous_decay.closed_electron_flow=${convection}
                        hybrid_pic_model.electron_inertia_convection=${convection}
                        hybrid_pic_model.include_electron_inertia_elliptic=1
                        hybrid_pic_model.electron_inertia_relative_tolerance=1.e-10
                        ions.profile=parse_density_function
                        "ions.density_function(x,y,z)=1.e19*(0.01+0.99*0.5*(1+cos(20.943951023931955*x)))")
                set_tests_properties(${test_name} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
                    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endforeach()
endif()

# Independent physical-energy regression for the additive-state Joule receiver.
if(TARGET lib_rz)
    add_executable(test_joule_pedestal_rz test_joule_pedestal.cpp)
    target_link_libraries(test_joule_pedestal_rz PRIVATE lib_rz)
    target_include_directories(test_joule_pedestal_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_joule_pedestal_rz)
    endif()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_test(NAME test_rz_joule_pedestal_capacity
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_joule_pedestal.py
            --executable $<TARGET_FILE:test_joule_pedestal_rz>
            --output ${CMAKE_CURRENT_BINARY_DIR}/joule_pedestal_capacity)
    set_tests_properties(test_rz_joule_pedestal_capacity PROPERTIES
        TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

# At-rest entropy transport and particle-precision boundary ownership.
include(${CMAKE_CURRENT_SOURCE_DIR}/qdsmc_marker_identity.cmake)

# Common particles/markers and independent viscous source bookkeeping.
include(${CMAKE_CURRENT_SOURCE_DIR}/viscous_common_state.cmake)

# Native nodal conduction probe. Numerical convergence is checked by the Python
# driver, which runs fixed physical profiles at multiple resolutions.
if(TARGET lib_rz)
    add_executable(test_conduction_order_rz test_conduction_order_rz.cpp)
    target_link_libraries(test_conduction_order_rz PRIVATE lib_rz)
    target_include_directories(test_conduction_order_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_conduction_order_rz)
    endif()
endif()

if(TARGET lib_3d)
    add_executable(test_conduction_order_3d test_conduction_order_rz.cpp)
    target_link_libraries(test_conduction_order_3d PRIVATE lib_3d)
    target_include_directories(test_conduction_order_3d PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_conduction_order_3d)
    endif()
endif()

# This gate is intentionally the normal-eligibility correction only. The full
# axis/wall upgrade is not certified by an axial-mode test.
if(TARGET test_conduction_order_rz OR TARGET test_conduction_order_3d)
    find_package(Python REQUIRED COMPONENTS Interpreter)
endif()
foreach(dim IN ITEMS rz 3d)
    if(TARGET test_conduction_order_${dim})
        if(dim STREQUAL "rz")
            set(conduction_order_input inputs_test_rz_implicit_thermal)
        else()
            set(conduction_order_input inputs_test_3d_conduction_order)
        endif()
        add_test(NAME test_${dim}_conduction_fd4_normal_edge_order
            COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_conduction_order_rz.py
                --executable $<TARGET_FILE:test_conduction_order_${dim}>
                --inputs ${CMAKE_CURRENT_SOURCE_DIR}/${conduction_order_input}
                --geometry ${dim} --profiles axial --sizes 32 64 128
                --output-parent ${CMAKE_CURRENT_BINARY_DIR}/conduction_order_${dim})
        set_tests_properties(test_${dim}_conduction_fd4_normal_edge_order PROPERTIES
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "conduction;fd4;native")
    endif()
endforeach()

if(TARGET lib_rz)
    add_executable(test_holmstrom_axis_rz test_holmstrom_axis.cpp)
    target_link_libraries(test_holmstrom_axis_rz PRIVATE lib_rz)
    target_include_directories(test_holmstrom_axis_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_holmstrom_axis_rz)
    endif()
    add_test(NAME test_rz_holmstrom_axis_attribution
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_holmstrom_axis.py
            --baseline $<TARGET_FILE:test_holmstrom_axis_rz>
            --candidate $<TARGET_FILE:test_holmstrom_axis_rz>
            --size 32
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/holmstrom_axis)
    set_tests_properties(test_rz_holmstrom_axis_attribution PROPERTIES
        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "conduction;ohm;native")
endif()

# Native shape-3 deposition and cylindrical filter continuity.
if(TARGET lib_rz)
    add_executable(test_rz_continuity test_rz_continuity.cpp)
    target_link_libraries(test_rz_continuity PRIVATE lib_rz)
    target_include_directories(test_rz_continuity PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_rz_continuity)
    endif()
    # Fixed roundoff bounds and bitwise diagnostic-isolation comparisons below
    # qualify CPU double precision. GPU/float qualification is separate.
    if((WarpX_COMPUTE STREQUAL "NOACC" OR WarpX_COMPUTE STREQUAL "OMP")
       AND WarpX_PRECISION STREQUAL "DOUBLE" AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        find_package(Python REQUIRED COMPONENTS Interpreter)
        add_test(NAME test_rz_continuity
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_continuity.py
                --executable $<TARGET_FILE:test_rz_continuity>
                --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_continuity
                --output-parent ${CMAKE_CURRENT_BINARY_DIR}/continuity)
        add_test(NAME test_rz_continuity_evolve
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_continuity_evolve.py
                --executable $<TARGET_FILE:test_rz_continuity>
                --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_continuity
                --output-parent ${CMAKE_CURRENT_BINARY_DIR}/continuity_evolve)
        set_tests_properties(test_rz_continuity test_rz_continuity_evolve PROPERTIES TIMEOUT 120
            ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "continuity;native")
    endif()
endif()

# Source capacities must follow the temperature's PC stage density.
if(TARGET lib_rz)
    foreach(fixture IN ITEMS order capacity)
        add_executable(test_qdsmc_source_stage_${fixture}_rz
            test_qdsmc_source_stage_${fixture}.cpp)
        target_link_libraries(test_qdsmc_source_stage_${fixture}_rz PRIVATE lib_rz)
        target_include_directories(test_qdsmc_source_stage_${fixture}_rz
            PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_qdsmc_source_stage_${fixture}_rz)
        endif()
    endforeach()
    add_test(NAME test_rz_qdsmc_source_stage
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_qdsmc_source_stage.py
            --order-exe $<TARGET_FILE:test_qdsmc_source_stage_order_rz>
            --capacity-exe $<TARGET_FILE:test_qdsmc_source_stage_capacity_rz>
            --output ${CMAKE_CURRENT_BINARY_DIR}/qdsmc_source_stage)
    set_tests_properties(test_rz_qdsmc_source_stage PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "HYBRID;RZ")
endif()

# Composed native conduction boundary counterexamples and regression gates.
if(TARGET lib_rz)
    add_executable(test_conduction_boundary_native_rz test_conduction_boundary_native_rz.cpp)
    target_link_libraries(test_conduction_boundary_native_rz PRIVATE lib_rz)
    target_include_directories(test_conduction_boundary_native_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_conduction_boundary_native_rz)
    endif()
    foreach(mode IN ITEMS flux robin closed)
        set(boundary_dt 1.e-7)
        set(boundary_integrator rkl2)
        if(mode STREQUAL "robin")
            set(boundary_dt 1.e-10)
            set(boundary_integrator ssprk2)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_native_${mode})
        add_test(NAME test_rz_conduction_boundary_native_${mode}
            COMMAND test_conduction_boundary_native_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/../ohm_solver_electron_energy_eq/inputs_test_rz_conduction_boundary
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting"
                hybrid_pic_model.qdsmc_conduction_flux_limit_factor=0
                hybrid_pic_model.qdsmc_conduction_fd_time=${boundary_integrator}
                boundary_test.mode=${mode} boundary_test.dt=${boundary_dt})
        set_tests_properties(test_rz_conduction_boundary_native_${mode} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_native_${mode}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;WARPX_QDSMC_COND_STATS=1")
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_qdsmc_rk_accounting test_qdsmc_rk_accounting.cpp)
    target_link_libraries(test_qdsmc_rk_accounting PRIVATE lib_rz)
    target_include_directories(test_qdsmc_rk_accounting PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_qdsmc_rk_accounting)
    endif()
    add_test(NAME test_rz_qdsmc_rk_accounting COMMAND test_qdsmc_rk_accounting)
    set_tests_properties(test_rz_qdsmc_rk_accounting PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    foreach(mode IN ITEMS pinned entry floor_equal floor_invalid conflict budget corner floor
                          gate_failure growth_failure growth_capped nonfinite_rhs density_cap density_closed capped_wall ghost eb_invalid eb_equal eb_capped pedestal)
        set(extra_options)
        set(boundary_dt 1.e-7)
        if(mode STREQUAL "corner" OR mode MATCHES "^density_")
            list(APPEND extra_options hybrid_pic_model.density_pedestal=0
                hybrid_pic_model.density_pedestal_track_floor=0)
        endif()
        if(mode MATCHES "^density_")
            set(boundary_dt 1.e-10)
            list(APPEND extra_options boundary_test.density=5.e13 hybrid_pic_model.qdsmc_n_floor=1.e14)
        elseif(mode STREQUAL "floor")
            set(boundary_dt 1.e-5)
        elseif(mode STREQUAL "growth_failure" OR mode STREQUAL "growth_capped")
            list(APPEND extra_options
                "hybrid_pic_model.qdsmc_kappa_par(n,Te,t)=1.5*1.380649e-23*100*n*(Te/2)^2.5"
                "hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*100*n*(Te/2)^2.5"
                hybrid_pic_model.qdsmc_conduction_chi_max=0)
            if(mode STREQUAL "growth_capped")
                list(REMOVE_ITEM extra_options hybrid_pic_model.qdsmc_conduction_chi_max=0)
                list(APPEND extra_options hybrid_pic_model.qdsmc_conduction_chi_max=400)
            endif()
        endif()
        if(mode STREQUAL "pedestal")
            list(APPEND extra_options hybrid_pic_model.density_pedestal=1
                hybrid_pic_model.density_pedestal_track_floor=0
                "hybrid_pic_model.density_pedestal_profile(x,y,z)=1.e14*(1+2*x+z)"
                boundary_test.density=2.e13 hybrid_pic_model.qdsmc_n_floor=1.e14)
        endif()
        if(mode MATCHES "^eb_")
            set(eb_temperature 0.25)
            set(eb_cap 0)
            if(mode STREQUAL "eb_invalid")
                set(eb_temperature 0.125)
            elseif(mode STREQUAL "eb_capped")
                set(eb_cap 1.e-6)
            endif()
            list(APPEND extra_options "warpx.eb_implicit_function=x*x-0.2*0.2"
                hybrid_pic_model.qdsmc_conduction_eb_bc=isothermal
                "hybrid_pic_model.qdsmc_conduction_eb_Te(x,y,z)=${eb_temperature}"
                hybrid_pic_model.qdsmc_conduction_eb_flux_limit=${eb_cap})
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_native_${mode})
        add_test(NAME test_rz_conduction_boundary_native_${mode}
            COMMAND test_conduction_boundary_native_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/../ohm_solver_electron_energy_eq/inputs_test_rz_conduction_boundary
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting"
                hybrid_pic_model.qdsmc_conduction_flux_limit_factor=0
                hybrid_pic_model.qdsmc_conduction_Te_floor=0.25
                hybrid_pic_model.qdsmc_conduction_fd_time=rkl2
                boundary_test.mode=${mode} boundary_test.dt=${boundary_dt} ${extra_options})
        set_tests_properties(test_rz_conduction_boundary_native_${mode} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_native_${mode}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;WARPX_QDSMC_COND_STATS=1")
    endforeach()
endif()

# Instrument only this translation unit in test builds; production builds with
# BUILD_TESTING=OFF contain no environment-controlled ghost mutation.
if(TARGET lib_rz)
    set_property(SOURCE ${WarpX_SOURCE_DIR}/Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.cpp
        TARGET_DIRECTORY lib_rz APPEND PROPERTY COMPILE_DEFINITIONS WARPX_CONDUCTION_TEST_HOOKS)
endif()

if(TARGET lib_rz)
    add_executable(test_conduction_boundary_evolve_rz test_conduction_boundary_evolve_rz.cpp)
    target_link_libraries(test_conduction_boundary_evolve_rz PRIVATE lib_rz)
    target_include_directories(test_conduction_boundary_evolve_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_compile_definitions(test_conduction_boundary_evolve_rz PRIVATE BOUNDARY_CANDIDATE_REPORT)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_conduction_boundary_evolve_rz)
    endif()
endif()

# Direct external-leg law and full-tensor inward-face oracles.
if(TARGET test_conduction_boundary_native_rz)
    foreach(case IN ITEMS eta0 eta1e6 eta1e5 eta1e4 sonic gate_off cold empty tensor)
        set(opts boundary_test.mode=leg_law)
        if(case STREQUAL "eta1e6" OR case STREQUAL "sonic")
            list(APPEND opts boundary_test.eta=1.e-6)
        elseif(case STREQUAL "eta1e5")
            list(APPEND opts boundary_test.eta=1.e-5)
        elseif(case STREQUAL "eta1e4")
            list(APPEND opts boundary_test.eta=1.e-4)
        elseif(case STREQUAL "gate_off")
            list(APPEND opts boundary_test.ti=0.4)
        elseif(case STREQUAL "cold")
            list(APPEND opts boundary_test.tb=0.4)
        elseif(case STREQUAL "empty")
            list(APPEND opts boundary_test.density=0)
        elseif(case STREQUAL "tensor")
            set(opts boundary_test.mode=tensor
                "hybrid_pic_model.qdsmc_kappa_perp(n,Te,t)=1.5*1.380649e-23*0.01*n")
        endif()
        if(case STREQUAL "sonic")
            list(APPEND opts hybrid_pic_model.qdsmc_conduction_wall_flux_cap_form=sonic)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_law_${case})
        add_test(NAME test_rz_conduction_boundary_law_${case}
            COMMAND test_conduction_boundary_native_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/../ohm_solver_electron_energy_eq/inputs_test_rz_conduction_boundary
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting"
                hybrid_pic_model.density_pedestal=0 hybrid_pic_model.density_pedestal_track_floor=0
                hybrid_pic_model.qdsmc_conduction_flux_limit_factor=0
                hybrid_pic_model.qdsmc_conduction_Te_floor=0.25
                hybrid_pic_model.qdsmc_conduction_fd_time=rkl2 boundary_test.dt=1.e-10 ${opts})
        set_tests_properties(test_rz_conduction_boundary_law_${case} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_law_${case}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET test_conduction_boundary_evolve_rz)
    foreach(suite IN ITEMS time restart)
        add_test(NAME test_rz_conduction_boundary_${suite}
            COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_conduction_boundary_evolve.py
                --executable $<TARGET_FILE:test_conduction_boundary_evolve_rz>
                --inputs ${CMAKE_CURRENT_SOURCE_DIR}/../ohm_solver_electron_energy_eq/inputs_test_rz_conduction_boundary
                --suite ${suite}
                --output-parent ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_${suite})
        set_tests_properties(test_rz_conduction_boundary_${suite} PROPERTIES
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
            LABELS "conduction;native")
    endforeach()
endif()

if(TARGET app_rz)
    add_test(NAME test_rz_conduction_boundary_explicit_pc
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/../ohm_solver_electron_energy_eq/analysis_conduction_boundary_explicit.py
            --executable $<TARGET_FILE:app_rz>
            --inputs ${CMAKE_CURRENT_SOURCE_DIR}/../ohm_solver_electron_energy_eq/inputs_test_rz_conduction_boundary
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_explicit_pc)
    set_tests_properties(test_rz_conduction_boundary_explicit_pc PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
        LABELS "conduction;native;explicit")
endif()

# Explicit PC stopping packets must deliver half on each side of transport.
if(TARGET lib_rz)
    add_executable(test_qdsmc_stopping_split_rz test_qdsmc_stopping_split.cpp)
    target_link_libraries(test_qdsmc_stopping_split_rz PRIVATE lib_rz)
    target_include_directories(test_qdsmc_stopping_split_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_qdsmc_stopping_split_rz)
    endif()
    set(stopping_ranks 1)
    if(WarpX_MPI)
        list(APPEND stopping_ranks 2)
    endif()
    foreach(ranks IN LISTS stopping_ranks)
        add_test(NAME test_rz_qdsmc_stopping_split_mpi${ranks}
            COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_qdsmc_stopping_split.py
                --split-exe $<TARGET_FILE:test_qdsmc_stopping_split_rz>
                --order-exe $<TARGET_FILE:test_qdsmc_source_stage_order_rz>
                --output ${CMAKE_CURRENT_BINARY_DIR}/stopping_split_mpi${ranks}
                --ranks ${ranks})
        set_tests_properties(test_rz_qdsmc_stopping_split_mpi${ranks} PROPERTIES
            TIMEOUT 180 PROCESSORS ${ranks} ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "HYBRID;RZ")
    endforeach()
endif()

# A controller floor change must reach every explicit thermal operator together.
if(TARGET lib_rz)
    add_executable(test_qdsmc_density_floor_sync_rz test_qdsmc_density_floor_sync.cpp)
    target_link_libraries(test_qdsmc_density_floor_sync_rz PRIVATE lib_rz)
    target_include_directories(test_qdsmc_density_floor_sync_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_qdsmc_density_floor_sync_rz)
    endif()
    set(floor_sync_ranks 1)
    if(WarpX_MPI)
        list(APPEND floor_sync_ranks 2)
    endif()
    foreach(ranks IN LISTS floor_sync_ranks)
        add_test(NAME test_rz_qdsmc_density_floor_sync_mpi${ranks}
            COMMAND ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/analysis_qdsmc_density_floor_sync.py
                --executable $<TARGET_FILE:test_qdsmc_density_floor_sync_rz>
                --output ${CMAKE_CURRENT_BINARY_DIR}/density_floor_sync_mpi${ranks}
                --ranks ${ranks})
        set_tests_properties(test_rz_qdsmc_density_floor_sync_mpi${ranks} PROPERTIES
            TIMEOUT 120 PROCESSORS ${ranks} ENVIRONMENT "OMP_NUM_THREADS=1"
            LABELS "HYBRID;RZ;explicit")
    endforeach()
endif()


# Nonlinear convective electron inertia: exact RZ acceleration, moving ions,
# density floor/pedestal, axis, fresh stages and full elliptic Ohm ordering.
if(TARGET lib_rz)
    add_executable(test_electron_inertia_convection_rz test_electron_inertia_convection.cpp)
    target_link_libraries(test_electron_inertia_convection_rz PRIVATE lib_rz)
    target_include_directories(test_electron_inertia_convection_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_electron_inertia_convection_rz)
    endif()
    foreach(box_size IN ITEMS 8 16)
        set(test_name test_rz_electron_inertia_convection_grid${box_size})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name}
            COMMAND test_electron_inertia_convection_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_convection
                amr.max_grid_size=${box_size})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Refine a nonlinear analytic electron flow independently of box decomposition.
if(TARGET test_electron_inertia_convection_rz)
    foreach(nz IN ITEMS 32 64)
        set(test_name test_rz_electron_inertia_convection_smooth_z${nz})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name}
            COMMAND test_electron_inertia_convection_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_convection
                "amr.n_cell=32 ${nz}" amr.max_grid_size=16)
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Continuity-form electron inertia: ion-current slope, compression and both
# production magnetic halves, with multiple box layouts and substep counts.
if(TARGET lib_rz)
    add_executable(test_electron_inertia_moments_rz test_electron_inertia_moments.cpp)
    target_link_libraries(test_electron_inertia_moments_rz PRIVATE lib_rz)
    target_include_directories(test_electron_inertia_moments_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_electron_inertia_moments_rz)
    endif()
    foreach(box_size IN ITEMS 8 16)
        foreach(substeps IN ITEMS 2 8)
            set(test_name test_rz_electron_inertia_moments_grid${box_size}_sub${substeps})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
            add_test(NAME ${test_name}
                COMMAND test_electron_inertia_moments_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_convection
                    hybrid_pic_model.electron_inertia_moment_terms=1
                    amr.max_grid_size=${box_size} hybrid_pic_model.substeps=${substeps})
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

# Actual charge-conserving moving edge and driven compression energy diagnostics.
if(TARGET lib_rz)
    add_executable(test_electron_inertia_transport_rz test_electron_inertia_transport.cpp)
    target_link_libraries(test_electron_inertia_transport_rz PRIVATE lib_rz)
    target_include_directories(test_electron_inertia_transport_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_electron_inertia_transport_rz)
    endif()
endif()

if(TARGET lib_rz)
    add_executable(test_electron_inertia_flux_profile_rz test_electron_inertia_flux_profile.cpp)
    target_link_libraries(test_electron_inertia_flux_profile_rz PRIVATE lib_rz)
    target_include_directories(test_electron_inertia_flux_profile_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_electron_inertia_flux_profile_rz)
    endif()
    foreach(nr IN ITEMS 32 64 128)
        set(test_name test_rz_electron_inertia_flux_profile_${nr})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name} COMMAND test_electron_inertia_flux_profile_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_convection
            "amr.n_cell=${nr} 16" amr.max_grid_size=16
            "boundary.field_lo=none periodic" "boundary.field_hi=pec periodic"
            "boundary.particle_lo=none periodic" "boundary.particle_hi=reflecting periodic"
            boundary.verboncoeur_axis_correction=0)
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET test_electron_inertia_transport_rz)
    foreach(nr IN ITEMS 32 64 128)
        math(EXPR edge_steps "100 * ${nr} / 32")
        if(nr EQUAL 32)
            set(edge_dt 2.5e-8)
        elseif(nr EQUAL 64)
            set(edge_dt 1.25e-8)
        else()
            set(edge_dt 6.25e-9)
        endif()
        set(test_name test_rz_electron_inertia_moving_edge_${nr})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name} COMMAND test_electron_inertia_transport_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_transport
            "amr.n_cell=${nr} 16" warpx.const_dt=${edge_dt} edge_transport.steps=${edge_steps})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
    find_package(Python COMPONENTS Interpreter REQUIRED)
    set(test_name test_rz_electron_inertia_moving_edge_order)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    add_test(NAME ${test_name} COMMAND ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis_electron_inertia_transport.py
        $<TARGET_FILE:test_electron_inertia_transport_rz>
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_transport)
    set_tests_properties(${test_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
        TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

if(TARGET lib_rz)
    add_executable(test_motional_work_rz test_motional_work_rz.cpp)
    target_link_libraries(test_motional_work_rz PRIVATE lib_rz)
    target_include_directories(test_motional_work_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_motional_work_rz)
    endif()
    foreach(box IN ITEMS 16 64)
        foreach(bc IN ITEMS periodic pec pmc)
            set(test_name test_rz_motional_work_${bc}_${box})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
            if(bc STREQUAL "periodic")
                set(particle_bc periodic)
            else()
                set(particle_bc reflecting)
            endif()
            add_test(NAME ${test_name} COMMAND test_motional_work_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_convection
                "amr.n_cell=64 32" amr.max_grid_size=${box}
                "boundary.field_lo=none ${bc}" "boundary.field_hi=pec ${bc}"
                "boundary.particle_lo=none ${particle_bc}"
                "boundary.particle_hi=reflecting ${particle_bc}")
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_rz_filter_work test_rz_filter_work.cpp)
    target_link_libraries(test_rz_filter_work PRIVATE lib_rz)
    target_include_directories(test_rz_filter_work PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_rz_filter_work)
    endif()
    foreach(box IN ITEMS 16 64)
        foreach(passes IN ITEMS 0 1 4)
            foreach(axis IN ITEMS 0 1)
                set(test_name test_rz_filter_work_${box}_${passes}_${axis})
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
                add_test(NAME ${test_name} COMMAND test_rz_filter_work
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_transport
                    "amr.n_cell=64 32" amr.max_grid_size=${box}
                    "warpx.filter_npass_each_dir=${passes} ${passes}"
                    warpx.rz_adjoint_gather=1 boundary.verboncoeur_axis_correction=${axis})
                set_tests_properties(${test_name} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
                    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endforeach()
endif()

# Complete energy exchange: actual charge-conserving ions, kinetic electron
# inertia, magnetic energy, and full represented thermal energy. A sharp edge
# must move at least two cells; numerical Hall work is never an energy source.
if(TARGET test_electron_inertia_transport_rz)
    set(energy_pair_options
        hybrid_pic_model.energy_conserving_motional=1
        hybrid_pic_model.center_field_moments=1
        warpx.rz_adjoint_gather=1)
    set(warm_energy_options
        hybrid_pic_model.elec_temp=5
        hybrid_pic_model.include_electron_pressure_term=1
        hybrid_pic_model.solve_electron_energy_equation=1
        hybrid_pic_model.qdsmc_time_advance=pc
        hybrid_pic_model.qdsmc_full_pedestal_transport=1)
    foreach(phase IN ITEMS 0 0.25 0.75)
        set(test_name test_rz_electron_energy_closed_phase${phase})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name} COMMAND test_electron_inertia_transport_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_closed_compression
            "amr.n_cell=64 16" warpx.const_dt=1.e-9 edge_transport.steps=300
            "ions.density_function(x,y,z)=if(x*x+y*y<(0.015+${phase}*0.03/64)^2,1.e21,0)"
            ${energy_pair_options})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 600 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
    endforeach()
    foreach(kind IN ITEMS closed_warm driven)
        set(test_name test_rz_electron_energy_${kind})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        if(kind STREQUAL "closed_warm")
            set(energy_input inputs_test_rz_electron_inertia_closed_compression)
            set(energy_options warpx.const_dt=1.e-9 edge_transport.steps=300 ${warm_energy_options})
        else()
            set(energy_input inputs_test_rz_electron_inertia_compression)
            set(energy_options hybrid_pic_model.substeps=128 edge_transport.steps=300)
        endif()
        add_test(NAME ${test_name} COMMAND test_electron_inertia_transport_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/${energy_input} "amr.n_cell=64 16"
            ${energy_options} ${energy_pair_options})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 600 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
    endforeach()
    # Dissipation must return magnetic work as thermal energy while the same
    # sharp pedestal edge moves. Keep the existing 1% energy and two-cell gates.
    foreach(kind IN ITEMS resistive resistive_halfdt hyper)
        set(test_name test_rz_electron_energy_${kind})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        set(dissipation_options
            "hybrid_pic_model.plasma_resistivity(rho,J,t)=if(rho>8.01088317,1.e-5,0)"
            warpx.const_dt=1.e-9 edge_transport.steps=300)
        if(kind STREQUAL "resistive_halfdt")
            list(APPEND dissipation_options warpx.const_dt=5.e-10 edge_transport.steps=600)
        elseif(kind STREQUAL "hyper")
            set(dissipation_options
                "hybrid_pic_model.plasma_hyper_resistivity(rho,B)=if(rho>8.01088317,1.e-12,0)"
                hybrid_pic_model.hyper_resistivity_curl_curl=1
                hybrid_pic_model.hyper_resistivity_heating=1
                warpx.const_dt=1.e-9 edge_transport.steps=300)
        endif()
        add_test(NAME ${test_name} COMMAND test_electron_inertia_transport_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_closed_compression
            "amr.n_cell=64 16" ${energy_pair_options} ${warm_energy_options}
            hybrid_pic_model.qdsmc_source_stage_density=1
            hybrid_pic_model.include_joule_heating=1 ${dissipation_options})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 900 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
    endforeach()
    foreach(transport IN ITEMS markers grid)
        set(test_name test_rz_electron_energy_warm_edge_${transport})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name} COMMAND test_electron_inertia_transport_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_transport
            ${warm_energy_options} ${energy_pair_options}
            hybrid_pic_model.qdsmc_transport_operator=${transport})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
    endforeach()
endif()

# The same independently integrated heat packets must survive full-population
# pedestal entropy transport, including empty ion cells and density jumps.
if(TARGET test_qdsmc_source_stage_capacity_rz)
    add_test(NAME test_rz_qdsmc_source_stage_full_pedestal
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_qdsmc_source_stage.py
            --order-exe $<TARGET_FILE:test_qdsmc_source_stage_order_rz>
            --capacity-exe $<TARGET_FILE:test_qdsmc_source_stage_capacity_rz>
            --output ${CMAKE_CURRENT_BINARY_DIR}/qdsmc_source_stage_full_pedestal
            --full-pedestal)
    set_tests_properties(test_rz_qdsmc_source_stage_full_pedestal PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics;HYBRID;RZ")
endif()

# Reject incompatible gather settings before a large particle load.
if(TARGET lib_rz)
    add_executable(test_rz_gather_configuration test_rz_gather_configuration.cpp)
    target_link_libraries(test_rz_gather_configuration PRIVATE lib_rz)
    target_include_directories(test_rz_gather_configuration PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_rz_gather_configuration)
    endif()
endif()

if(TARGET lib_rz)
    add_executable(test_rz_mc_boundary test_rz_mc_boundary.cpp)
    target_link_libraries(test_rz_mc_boundary PRIVATE lib_rz)
    target_include_directories(test_rz_mc_boundary PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_rz_mc_boundary)
    endif()
endif()

if(TARGET test_rz_gather_configuration)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_rz_gather_configuration_early)
    add_test(NAME test_rz_gather_configuration_early
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_gather_configuration.py
        $<TARGET_FILE:test_rz_gather_configuration>
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_electron_inertia_transport)
    set_tests_properties(test_rz_gather_configuration_early PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_rz_gather_configuration_early
        TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

if(TARGET test_rz_mc_boundary)
    foreach(absorption IN ITEMS 0 0.05 1)
        set(test_name test_rz_mc_boundary_continuity_${absorption})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        if(absorption STREQUAL "0")
            set(require_escape 0)
        else()
            set(require_escape 1)
        endif()
        add_test(NAME ${test_name} COMMAND test_rz_mc_boundary
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_mc_boundary
            boundary.particle_absorption_fraction=${absorption}
            mc_boundary.require_escape=${require_escape})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
    endforeach()
endif()

if(TARGET test_rz_mc_boundary)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_rz_mc_boundary_restart)
    add_test(NAME test_rz_mc_boundary_restart
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis_rz_boundary_restart.py
        $<TARGET_FILE:test_rz_mc_boundary>
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_mc_boundary)
    set_tests_properties(test_rz_mc_boundary_restart PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_rz_mc_boundary_restart
        TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
endif()

if(TARGET test_rz_mc_boundary)
    foreach(layout IN ITEMS filter4 multibox)
        set(test_name test_rz_mc_boundary_${layout})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        if(layout STREQUAL "multibox")
            set(boundary_grid 16)
        else()
            set(boundary_grid 32)
        endif()
        add_test(NAME ${test_name} COMMAND test_rz_mc_boundary
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_mc_boundary
            "warpx.filter_npass_each_dir=4 4" amr.max_grid_size=${boundary_grid}
            boundary.particle_absorption_fraction=1)
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "physics")
    endforeach()
endif()

# Default-off native source eligibility, including paired source legs and RZ support.
if(TARGET lib_rz)
    add_executable(test_source_guard_rz test_source_guard_rz.cpp)
    target_link_libraries(test_source_guard_rz PRIVATE lib_rz)
    target_include_directories(test_source_guard_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_source_guard_rz)
    endif()
    foreach(mode IN ITEMS dense dust vacuum redirect drift mask_change
            configured_floor ohmic_floor nonperiodic_low nonperiodic_high periodic_wrap moving_epoch)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/source_guard_${mode})
        set(extra_args "")
        if(mode STREQUAL "dust")
            set(extra_args hybrid_pic_model.source_guard.minimum_resident_markers=17)
        elseif(mode STREQUAL "vacuum")
            set(extra_args source_test.n=0 source_test.defect=0)
        elseif(mode STREQUAL "redirect")
            set(extra_args source_test.redirect=1 source_test.thermal=0 source_test.stopping=0)
        elseif(mode STREQUAL "drift")
            set(extra_args source_test.electron_drift=6.e6)
        elseif(mode STREQUAL "mask_change")
            set(extra_args source_test.changed_mask=1)
        elseif(mode STREQUAL "configured_floor")
            set(extra_args hybrid_pic_model.source_guard.n_min=2.e17 source_test.n=2.e17 source_test.defect=0)
        elseif(mode STREQUAL "ohmic_floor")
            set(extra_args hybrid_pic_model.joule_heating_n_min=2.e17 source_test.n=2.e17 source_test.defect=0)
        elseif(mode MATCHES "^nonperiodic_")
            set(extra_args "boundary.field_lo=none pec" "boundary.field_hi=none pec"
                "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
            if(mode STREQUAL "nonperiodic_low")
                list(APPEND extra_args source_test.defect_j=0)
            else()
                list(APPEND extra_args source_test.defect_j=16)
            endif()
        elseif(mode STREQUAL "periodic_wrap")
            set(extra_args source_test.defect_j=0)
        elseif(mode STREQUAL "moving_epoch")
            set(extra_args source_test.move_epoch=1 source_test.defect=0)
        endif()
        add_test(NAME test_rz_source_guard_${mode} COMMAND test_source_guard_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_source_guard_rz ${extra_args})
        set_tests_properties(test_rz_source_guard_${mode} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/source_guard_${mode}
            TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "HYBRID;RZ;source_guard")
    endforeach()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    foreach(fault IN ITEMS thermal_nan thermal_negative_rate ohmic_nan stopping_nan_te stopping_nan_velocity)
        add_test(NAME test_rz_source_guard_reject_${fault}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_source_guard.py
                --executable $<TARGET_FILE:test_source_guard_rz>
                --output ${CMAKE_CURRENT_BINARY_DIR}/source_guard_reject_${fault}
                --case eligible_${fault} --new-attempt)
        set_tests_properties(test_rz_source_guard_reject_${fault} PROPERTIES
            TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "HYBRID;RZ;source_guard")
    endforeach()
endif()
