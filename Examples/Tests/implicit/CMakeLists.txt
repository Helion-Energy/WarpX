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
    foreach(grid IN ITEMS 8 32)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_edge_${grid})
        add_test(NAME test_rz_recovery_probe_edge_${grid}
            COMMAND test_recovery_probe_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                amr.max_grid_size=${grid}
                hybrid_pic_model.darwin_vacuum_recovery_operator=edge_relaxation)
        set_tests_properties(test_rz_recovery_probe_edge_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_edge_${grid}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
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
    target_include_directories(test_recovery_multigrid_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
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
    foreach(viscous_boundary IN ITEMS periodic pec)
        if(viscous_boundary STREQUAL "pec")
            set(viscous_particle_boundary reflecting)
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

if(TARGET lib_rz AND (WarpX_FFT OR ABLASTR_FFT))
    foreach(grid IN ITEMS 8 32)
        add_test(NAME test_rz_cached_green_${grid}
            COMMAND test_recovery_multigrid_rz mg_test.grid=${grid}
                mg_test.greens=1 mg_test.max_semi=1)
        set_tests_properties(test_rz_cached_green_${grid} PROPERTIES
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_greens_${grid})
        add_test(NAME test_rz_recovery_probe_greens_${grid}
            COMMAND test_recovery_probe_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                amr.max_grid_size=${grid}
                "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
                "boundary.particle_lo=none reflecting"
                "boundary.particle_hi=reflecting reflecting"
                hybrid_pic_model.darwin_vacuum_recovery_operator=greens)
        set_tests_properties(test_rz_recovery_probe_greens_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_recovery_probe_greens_${grid}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Anchored continuity density response and independent endpoint-charge audit.
if(TARGET lib_rz)
    add_executable(test_mm_density_rz test_mm_density_rz.cpp)
    target_link_libraries(test_mm_density_rz PRIVATE lib_rz)
    target_include_directories(test_mm_density_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_mm_density_rz)
    endif()
    foreach(grid IN ITEMS 8 32)
        foreach(nz IN ITEMS 32 64)
            foreach(deposition IN ITEMS direct esirkepov)
                set(case_name rz_mm_density_${grid}_${nz}_${deposition})
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
                add_test(NAME test_${case_name} COMMAND test_mm_density_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                    amr.max_grid_size=${grid} "amr.n_cell=16 ${nz}"
                    hybrid_pic_model.n_floor=1.e14 algo.current_deposition=${deposition}
                    algo.particle_shape=2 algo.field_gathering=momentum-conserving)
                set_tests_properties(test_${case_name} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
                    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_implicit_gather_order_rz test_implicit_gather_order_rz.cpp)
    target_link_libraries(test_implicit_gather_order_rz PRIVATE lib_rz)
    target_include_directories(test_implicit_gather_order_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_implicit_gather_order_rz)
    endif()
    foreach(shape IN ITEMS 2 3)
        foreach(nz IN ITEMS 32 64 128)
            set(case_name rz_gather_order_${shape}_${nz})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
            add_test(NAME test_${case_name} COMMAND test_implicit_gather_order_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                "amr.n_cell=16 ${nz}" algo.particle_shape=${shape}
                algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving)
            set_tests_properties(test_${case_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

# Compare all nine Esirkepov current / momentum-gather electric response blocks
# with independent particle pushes and conservative endpoint-averaged charge.
if(TARGET lib_rz)
    add_executable(test_esirkepov_mm_rz test_esirkepov_mm_rz.cpp)
    target_link_libraries(test_esirkepov_mm_rz PRIVATE lib_rz)
    target_include_directories(test_esirkepov_mm_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_esirkepov_mm_rz)
    endif()
    foreach(shape IN ITEMS 2 3 4)
        foreach(grid IN ITEMS 8 32)
            foreach(boundary IN ITEMS periodic finite absorbing)
                set(extra_args)
                if(boundary STREQUAL "finite")
                    list(APPEND extra_args "boundary.field_lo=none neumann"
                        "boundary.field_hi=pec neumann" "boundary.particle_lo=none reflecting"
                        "boundary.particle_hi=reflecting reflecting")
                elseif(boundary STREQUAL "absorbing")
                    list(APPEND extra_args "boundary.field_lo=none neumann"
                        "boundary.field_hi=pec neumann" "boundary.particle_lo=none absorbing"
                        "boundary.particle_hi=reflecting absorbing")
                endif()
                set(case_name rz_esirkepov_mm_${shape}_${grid}_${boundary})
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
                add_test(NAME test_${case_name} COMMAND test_esirkepov_mm_rz
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                    algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                    algo.particle_shape=${shape} amr.max_grid_size=${grid}
                    implicit_evolve.use_mass_matrices_jacobian=1 hybrid_pic_model.n_floor=1.e14
                    ${extra_args})
                set_tests_properties(test_${case_name} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
                    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
        foreach(drift IN ITEMS -0.0003 0.0003)
            set(case_name rz_esirkepov_mm_multicell_${shape}_${drift})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
            add_test(NAME test_${case_name} COMMAND test_esirkepov_mm_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                algo.particle_shape=${shape} amr.max_grid_size=32 "amr.n_cell=16 64"
                implicit_evolve.use_mass_matrices_jacobian=1 hybrid_pic_model.n_floor=1.e14
                warpx.const_dt=1.e-6 ions.uz=${drift} particles.max_grid_crossings=8
                qualification.oblique_b=0 qualification.perturbation=1.e-6)
            set_tests_properties(test_${case_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
            set(absorbing_case ${case_name}_absorbing)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${absorbing_case})
            add_test(NAME test_${absorbing_case} COMMAND test_esirkepov_mm_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                algo.particle_shape=${shape} amr.max_grid_size=32 "amr.n_cell=16 64"
                implicit_evolve.use_mass_matrices_jacobian=1 hybrid_pic_model.n_floor=1.e14
                warpx.const_dt=1.e-6 ions.uz=${drift} particles.max_grid_crossings=8
                qualification.oblique_b=0 qualification.perturbation=1.e-6
                "boundary.field_lo=none neumann" "boundary.field_hi=pec neumann"
                "boundary.particle_lo=none absorbing" "boundary.particle_hi=reflecting absorbing")
            set_tests_properties(test_${absorbing_case} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${absorbing_case}
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

# The E_L hierarchy must preserve the native gradient/divergence operator on
# anisotropic cells, including the RZ axis and finite Neumann end caps.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_darwin_poisson_mg_${dim} test_darwin_poisson_mg.cpp)
        if(dim STREQUAL "rz")
            target_link_libraries(test_darwin_poisson_mg_${dim} PRIVATE AMReX::amrex_2d)
            target_compile_definitions(test_darwin_poisson_mg_${dim} PRIVATE DARWIN_POISSON_TEST_RZ)
        else()
            target_link_libraries(test_darwin_poisson_mg_${dim} PRIVATE AMReX::amrex_3d)
        endif()
        target_include_directories(test_darwin_poisson_mg_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_darwin_poisson_mg_${dim})
        endif()
        foreach(aspect IN ITEMS 1 5 16)
            if(aspect EQUAL 1)
                set(el_semi_levels 0)
            else()
                set(el_semi_levels 2)
            endif()
            add_test(NAME test_${dim}_darwin_poisson_mg_${aspect}
                COMMAND test_darwin_poisson_mg_${dim} poisson_test.aspect=${aspect}
                    poisson_test.expected_levels=${el_semi_levels})
            set_tests_properties(test_${dim}_darwin_poisson_mg_${aspect} PROPERTIES
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
        if(dim STREQUAL "rz")
            add_test(NAME test_rz_darwin_poisson_mg_radial_wide
                COMMAND test_darwin_poisson_mg_rz poisson_test.aspect=0.2
                    poisson_test.expected_levels=2 poisson_test.grid=32)
            set_tests_properties(test_rz_darwin_poisson_mg_radial_wide PROPERTIES
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endif()
    endif()
endforeach()

# A normal to a wall is half-cell centered: the physical Dirichlet trace
# belongs between the valid and ghost centers, in every spatial dimension.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_darwin_a_boundary_${dim} test_darwin_a_boundary.cpp)
        if(dim STREQUAL "rz")
            target_link_libraries(test_darwin_a_boundary_${dim} PRIVATE AMReX::amrex_2d)
        else()
            target_link_libraries(test_darwin_a_boundary_${dim} PRIVATE AMReX::amrex_3d)
        endif()
        target_include_directories(test_darwin_a_boundary_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_darwin_a_boundary_${dim})
        endif()
        foreach(grid IN ITEMS 8 32)
            add_test(NAME test_${dim}_darwin_a_boundary_${grid}
                COMMAND test_darwin_a_boundary_${dim} boundary_test.grid=${grid})
            set_tests_properties(test_${dim}_darwin_a_boundary_${grid} PROPERTIES
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()

# The physical t0 field and current history precede every implicit particle trial.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_darwin_initial_state_${dim} test_darwin_initial_state.cpp)
        target_link_libraries(test_darwin_initial_state_${dim} PRIVATE lib_${dim})
        target_include_directories(test_darwin_initial_state_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_darwin_initial_state_${dim})
        endif()
        if(dim STREQUAL "rz")
            set(startup_input inputs_test_rz_darwin_vacuum_recovery)
            set(startup_boundaries)
        else()
            set(startup_input inputs_test_3d_darwin_split_vacuum_energy)
            set(startup_boundaries "boundary.particle_lo=reflecting reflecting periodic"
                "boundary.particle_hi=reflecting reflecting periodic")
        endif()
        foreach(grid IN ITEMS 8 32)
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/startup_${dim}_${grid})
            add_test(NAME test_${dim}_darwin_initial_state_${grid}
                COMMAND test_darwin_initial_state_${dim}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${startup_input}
                    amr.max_grid_size=${grid} amr.blocking_factor=8
                    max_step=0 diagnostics.enable=0 ${startup_boundaries}
                    hybrid_pic_model.darwin_vacuum_recovery=0
                    hybrid_pic_model.solve_electron_energy_equation=0
                    hybrid_pic_model.n_floor=1.e12 hybrid_pic_model.n0_ref=1.e18
                    hybrid_pic_model.elec_temp=0 hybrid_pic_model.gamma=1
                    ions.density=1.e18
                    algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                    algo.particle_shape=3 particles.max_grid_crossings=4
                    jacobian.pc_type=none
                    "warpx.Bz_external_grid_function(x,y,z)=0.1+0.2*x"
                    "external_vector_potential.uniform_analytical.A_time_external_function(t)=1+3*t"
                    "external_vector_potential.uniform_analytical.Ay_external_grid_function(x,y,z)=0.015*x"
                    "external_vector_potential.uniform_analytical.Ax_external_grid_function(x,y,z)=-0.015*y")
            set_tests_properties(test_${dim}_darwin_initial_state_${grid} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/startup_${dim}_${grid}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/startup_pressure_${dim}_${grid})
            add_test(NAME test_${dim}_darwin_initial_pressure_${grid}
                COMMAND test_darwin_initial_state_${dim}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${startup_input}
                    amr.max_grid_size=${grid} amr.blocking_factor=8
                    max_step=0 diagnostics.enable=0 ${startup_boundaries}
                    hybrid_pic_model.darwin_vacuum_recovery=0
                    hybrid_pic_model.solve_electron_energy_equation=0
                    hybrid_pic_model.n_floor=1.e12 hybrid_pic_model.n0_ref=1.e18
                    hybrid_pic_model.elec_temp=10 hybrid_pic_model.gamma=1
                    ions.profile=parse_density_function
                    "ions.density_function(x,y,z)=1.e18*(1+0.2*cos(50.26548245743669*z))"
                    ions.uz=0.01 startup_test.pressure=1
                    algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                    algo.particle_shape=3 particles.max_grid_crossings=4
                    jacobian.pc_type=none
                    "warpx.Bz_external_grid_function(x,y,z)=0.1+0.2*x"
                    "external_vector_potential.uniform_analytical.A_time_external_function(t)=1+3*t"
                    "external_vector_potential.uniform_analytical.Ay_external_grid_function(x,y,z)=0.015*x"
                    "external_vector_potential.uniform_analytical.Ax_external_grid_function(x,y,z)=-0.015*y")
            set_tests_properties(test_${dim}_darwin_initial_pressure_${grid} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/startup_pressure_${dim}_${grid}
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
            foreach(seed IN ITEMS default fitted pedestal fitted_deposited)
                set(seed_args)
                if(seed STREQUAL "fitted" OR seed STREQUAL "fitted_deposited")
                    set(seed_args
                        hybrid_pic_model.initial_electron_temperature_gamma=1.426
                        hybrid_pic_model.initial_electron_temperature_n_floor=9.5e17
                        startup_test.gamma_init=1.426 startup_test.floor_init=9.5e17)
                endif()
                if(seed STREQUAL "pedestal" OR seed STREQUAL "fitted_deposited")
                    list(APPEND seed_args hybrid_pic_model.density_pedestal=1
                        "hybrid_pic_model.density_pedestal_profile(x,y,z)=1.e17"
                        startup_test.density_pedestal=1.e17)
                endif()
                if(seed STREQUAL "fitted_deposited")
                    list(APPEND seed_args
                        hybrid_pic_model.initial_electron_temperature_include_pedestal=0
                        startup_test.include_pedestal=0)
                endif()
                set(seed_dir ${CMAKE_CURRENT_BINARY_DIR}/startup_Te_${dim}_${grid}_${seed})
                file(MAKE_DIRECTORY ${seed_dir})
                add_test(NAME test_${dim}_darwin_initial_temperature_${grid}_${seed}
                    COMMAND test_darwin_initial_state_${dim}
                        ${CMAKE_CURRENT_SOURCE_DIR}/${startup_input}
                        amr.max_grid_size=${grid} amr.blocking_factor=8
                        max_step=0 diagnostics.enable=0 ${startup_boundaries}
                        hybrid_pic_model.darwin_vacuum_recovery=0
                        hybrid_pic_model.solve_electron_energy_equation=1
                        hybrid_pic_model.n_floor=1.e12 hybrid_pic_model.n0_ref=1.e18
                        hybrid_pic_model.elec_temp=10 hybrid_pic_model.gamma=1.6666666666666667
                        hybrid_pic_model.qdsmc_te_n_floor=1.e14
                        ions.profile=parse_density_function
                        "ions.density_function(x,y,z)=1.e18*(1+0.2*cos(50.26548245743669*z))"
                        ions.uz=0.01 startup_test.pressure=1 startup_test.temperature=1
                        ${seed_args}
                        algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                        algo.particle_shape=3 particles.max_grid_crossings=4
                        jacobian.pc_type=none
                        "warpx.Bz_external_grid_function(x,y,z)=0.1+0.2*x"
                        "external_vector_potential.uniform_analytical.A_time_external_function(t)=1+3*t"
                        "external_vector_potential.uniform_analytical.Ay_external_grid_function(x,y,z)=0.015*x"
                        "external_vector_potential.uniform_analytical.Ax_external_grid_function(x,y,z)=-0.015*y")
                set_tests_properties(test_${dim}_darwin_initial_temperature_${grid}_${seed}
                    PROPERTIES WORKING_DIRECTORY ${seed_dir} TIMEOUT 60
                    ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endif()
endforeach()

# Finite-z PMC response: compare the cached inverse to an independently
# assembled stencil and native MLMG for both equal and mixed end types.
if(TARGET test_recovery_multigrid_rz)
    foreach(pair IN ITEMS 01 10 11)
        string(SUBSTRING "${pair}" 0 1 pmc_lo)
        string(SUBSTRING "${pair}" 1 1 pmc_hi)
        foreach(grid IN ITEMS 8 32)
            add_test(NAME test_rz_recovery_greens_pmc_${pair}_${grid}
                COMMAND test_recovery_multigrid_rz mg_test.grid=${grid}
                    mg_test.greens=1 mg_test.max_semi=1
                    mg_test.pmc_lo=${pmc_lo} mg_test.pmc_hi=${pmc_hi})
            set_tests_properties(test_rz_recovery_greens_pmc_${pair}_${grid}
                PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_darwin_pmc_recovery_rz test_darwin_pmc_recovery.cpp)
    target_link_libraries(test_darwin_pmc_recovery_rz PRIVATE lib_rz)
    target_include_directories(test_darwin_pmc_recovery_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_darwin_pmc_recovery_rz)
    endif()
    foreach(grid IN ITEMS 8 32)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_pmc_recovery_${grid})
        add_test(NAME test_rz_darwin_pmc_recovery_${grid}
            COMMAND test_darwin_pmc_recovery_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_darwin_vacuum_recovery
                amr.max_grid_size=${grid}
                "boundary.field_lo=none neumann" "boundary.field_hi=pec neumann"
                "boundary.particle_lo=none absorbing" "boundary.particle_hi=reflecting absorbing"
                hybrid_pic_model.darwin_vacuum_recovery_components=flux)
        set_tests_properties(test_rz_darwin_pmc_recovery_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_pmc_recovery_${grid}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_pmc_drive_${grid})
        add_test(NAME test_rz_darwin_pmc_drive_${grid}
            COMMAND test_darwin_pmc_recovery_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_darwin_vacuum_recovery
                amr.max_grid_size=${grid}
                "boundary.field_lo=none neumann" "boundary.field_hi=pec neumann"
                "boundary.particle_lo=none absorbing" "boundary.particle_hi=reflecting absorbing"
                hybrid_pic_model.darwin_vacuum_recovery_components=flux
                test.pmc_drive_slope=0.25
                "external_vector_potential.uniform_analytical.Ay_external_grid_function(x,y,z)=0.5*x*0.01*(1+0.25*z)")
        set_tests_properties(test_rz_darwin_pmc_drive_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rz_pmc_drive_${grid}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET lib_3d)
    add_executable(test_darwin_pmc_recovery_3d test_darwin_pmc_recovery.cpp)
    target_link_libraries(test_darwin_pmc_recovery_3d PRIVATE lib_3d)
    target_include_directories(test_darwin_pmc_recovery_3d PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_darwin_pmc_recovery_3d)
    endif()
    foreach(grid IN ITEMS 8 16)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pmc_recovery_3d_${grid})
        add_test(NAME test_3d_darwin_pmc_recovery_${grid}
            COMMAND test_darwin_pmc_recovery_3d
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_darwin_split_vacuum_energy
                amr.max_grid_size=${grid} amr.blocking_factor=8 max_step=0 diagnostics.enable=0
                "boundary.field_lo=pec pec neumann" "boundary.field_hi=pec pec neumann"
                "boundary.particle_lo=reflecting reflecting absorbing"
                "boundary.particle_hi=reflecting reflecting absorbing"
                hybrid_pic_model.darwin_vacuum_e_check_operator=1
                hybrid_pic_model.darwin_vacuum_recovery_components=all)
        set_tests_properties(test_3d_darwin_pmc_recovery_${grid} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pmc_recovery_3d_${grid}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Native Ohm/Joule end-buffer coupling, including the no-resistivity push path.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_hybrid_end_region_${dim} test_hybrid_end_region.cpp)
        target_link_libraries(test_hybrid_end_region_${dim} PRIVATE lib_${dim})
        target_include_directories(test_hybrid_end_region_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_hybrid_end_region_${dim})
        endif()
        if(dim STREQUAL "rz")
            set(end_input inputs_test_rz_darwin_vacuum_recovery)
            set(end_boundaries "boundary.field_lo=none neumann" "boundary.field_hi=pec neumann"
                "boundary.particle_lo=none absorbing" "boundary.particle_hi=reflecting absorbing")
        else()
            set(end_input inputs_test_3d_darwin_split_vacuum_energy)
            set(end_boundaries "boundary.field_lo=pec pec neumann" "boundary.field_hi=pec pec neumann"
                "boundary.particle_lo=reflecting reflecting absorbing"
                "boundary.particle_hi=reflecting reflecting absorbing")
        endif()
        foreach(grid IN ITEMS 8 32)
            set(end_dir ${CMAKE_CURRENT_BINARY_DIR}/end_region_${dim}_${grid})
            file(MAKE_DIRECTORY ${end_dir})
            add_test(NAME test_${dim}_hybrid_end_region_${grid}
                COMMAND test_hybrid_end_region_${dim} ${CMAKE_CURRENT_SOURCE_DIR}/${end_input}
                    amr.max_grid_size=${grid} amr.blocking_factor=8 max_step=0 diagnostics.enable=0
                    ${end_boundaries} hybrid_pic_model.darwin_vacuum_recovery=0
                    hybrid_pic_model.add_external_fields=0 jacobian.pc_type=none
                    "hybrid_pic_model.joule_heating_resistivity(rho,J,Te,t)=0.0")
            set_tests_properties(test_${dim}_hybrid_end_region_${grid} PROPERTIES
                WORKING_DIRECTORY ${end_dir} TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()

if(TARGET test_hybrid_pc_coefficients_rz)
    set(case_name rz_hybrid_pc_coefficients_end)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
    add_test(NAME test_${case_name} COMMAND test_hybrid_pc_coefficients_rz
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
        "boundary.field_lo=none neumann" "boundary.field_hi=pec neumann"
        "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
    set_tests_properties(test_${case_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

if(TARGET test_esirkepov_mm_rz)
    foreach(boundary IN ITEMS periodic finite absorbing)
        set(extra_args)
        if(NOT boundary STREQUAL "periodic")
            list(APPEND extra_args "boundary.field_lo=none neumann"
                "boundary.field_hi=pec neumann" "boundary.particle_lo=none ${boundary}")
            if(boundary STREQUAL "finite")
                set(extra_args "boundary.field_lo=none neumann" "boundary.field_hi=pec neumann"
                    "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
            else()
                list(APPEND extra_args "boundary.particle_hi=reflecting absorbing")
            endif()
        endif()
        foreach(grid IN ITEMS 8 32)
            set(case_name rz_esirkepov_mm_pc_${boundary}_${grid})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
            add_test(NAME test_${case_name} COMMAND test_esirkepov_mm_rz
                ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_recovery_probe
                algo.current_deposition=esirkepov algo.field_gathering=momentum-conserving
                algo.particle_shape=3 amr.max_grid_size=${grid} qualification.mm_pc=1
                implicit_evolve.use_mass_matrices_jacobian=1 implicit_evolve.use_mass_matrices_pc=1
                jacobian.pc_type=pc_hybrid_pic newton.linear_solver=amrex_fgmres
                hybrid_pic_model.n_floor=1.e14 ${extra_args})
            set_tests_properties(test_${case_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

if(TARGET lib_rz)
    add_executable(test_thermal_fixed_point test_thermal_fixed_point.cpp)
    target_link_libraries(test_thermal_fixed_point PRIVATE lib_rz)
    target_include_directories(test_thermal_fixed_point PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_thermal_fixed_point)
    endif()
    add_test(NAME test_rz_thermal_fixed_point COMMAND test_thermal_fixed_point)
    set_tests_properties(test_rz_thermal_fixed_point PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

if(TARGET test_hybrid_pc_cartesian)
    foreach(case IN ITEMS ion_sd pmc)
        set(extra_args)
        if(case STREQUAL "ion_sd")
            set(extra_args hall_test.ion=1)
        else()
            set(extra_args hall_test.pmc=1 "boundary.field_lo=periodic periodic neumann"
                "boundary.field_hi=periodic periodic neumann"
                "boundary.particle_lo=periodic periodic reflecting"
                "boundary.particle_hi=periodic periodic reflecting")
        endif()
        set(case_name 3d_hybrid_pc_${case})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
        add_test(NAME test_${case_name} COMMAND test_hybrid_pc_cartesian
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_hybrid_pc
            warpx.grid_type=collocated ${extra_args})
        set_tests_properties(test_${case_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

if(TARGET test_hybrid_pc_rz)
    foreach(case IN ITEMS stiff_stencil stiff_outer moderate_inverse)
        set(outer 0)
        set(extra_args hall_test.hall=0.05)
        if(case STREQUAL "stiff_stencil")
            list(APPEND extra_args hall_test.solve=0)
        elseif(case STREQUAL "stiff_outer")
            set(outer 1)
        else()
            set(extra_args) # same Hall coefficient as the established inverse test
        endif()
        set(case_name rz_hybrid_pc_paired_symbol_${case})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name})
        add_test(NAME test_${case_name} COMMAND test_hybrid_pc_rz
            hall_test.grid=8 hall_test.outer=${outer} hall_test.rz_block_symbol=1
            hall_test.chi=0.4 ${extra_args})
        set_tests_properties(test_${case_name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${case_name}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Pure spatial conduction and legacy wall-law contracts for both geometries.
foreach(dimension IN ITEMS rz 3d)
    if(TARGET lib_${dimension})
        add_executable(test_thermal_conduction_operator_${dimension}
            test_thermal_conduction_operator.cpp)
        target_link_libraries(test_thermal_conduction_operator_${dimension} PRIVATE lib_${dimension})
        target_include_directories(test_thermal_conduction_operator_${dimension}
            PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_thermal_conduction_operator_${dimension})
        endif()
        foreach(grid IN ITEMS 8 64)
            add_test(NAME test_${dimension}_thermal_conduction_operator_${grid}
                COMMAND test_thermal_conduction_operator_${dimension} test.max_grid_size=${grid})
            set_tests_properties(test_${dimension}_thermal_conduction_operator_${grid} PROPERTIES
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()

# Pure MHD-derived thermal stage and corrected Chacon donor regression.
foreach(thermal_dim IN ITEMS RZ 3D)
    string(TOLOWER "${thermal_dim}" thermal_suffix)
    if(TARGET lib_${thermal_suffix})
        set(thermal_stage_target test_eulerian_thermal_stage_${thermal_suffix})
        add_executable(${thermal_stage_target} test_eulerian_thermal_stage.cpp)
        target_link_libraries(${thermal_stage_target} PRIVATE lib_${thermal_suffix})
        target_include_directories(${thermal_stage_target} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(${thermal_stage_target})
        endif()
        add_test(NAME ${thermal_stage_target} COMMAND ${thermal_stage_target})
        set_tests_properties(${thermal_stage_target} PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()

# Conservative signed work: independent physical-edge scatter reference, masks,
# corners, periodic images and an empty rank when run with MPI and one box.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_thermal_work_redistribution_${dim}
            test_thermal_work_redistribution.cpp)
        target_link_libraries(test_thermal_work_redistribution_${dim} PRIVATE lib_${dim})
        target_include_directories(test_thermal_work_redistribution_${dim}
            PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_thermal_work_redistribution_${dim})
        endif()
        foreach(scenario RANGE 0 5)
            foreach(box IN ITEMS 4 64)
                foreach(periodic IN ITEMS 0 1)
                    set(work_test test_${dim}_thermal_work_s${scenario}_b${box}_p${periodic})
                    add_test(NAME ${work_test} COMMAND test_thermal_work_redistribution_${dim}
                        test.scenario=${scenario} test.box=${box} test.periodic=${periodic})
                    set_tests_properties(${work_test} PROPERTIES
                        TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
                endforeach()
            endforeach()
        endforeach()
    endif()
endforeach()


# Accepted density-control remap and checkpoint invariants.
find_package(Python COMPONENTS Interpreter REQUIRED)
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_electron_density_control_runtime_${dim} test_electron_density_control_runtime.cpp)
        target_link_libraries(test_electron_density_control_runtime_${dim} PRIVATE lib_${dim})
        target_include_directories(test_electron_density_control_runtime_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_electron_density_control_runtime_${dim})
        endif()
        add_executable(test_electron_density_control_${dim} test_electron_density_control.cpp)
        target_link_libraries(test_electron_density_control_${dim} PRIVATE lib_${dim})
        target_include_directories(test_electron_density_control_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_electron_density_control_${dim})
        endif()
        foreach(grid IN ITEMS 4 64)
            add_test(NAME test_${dim}_electron_density_control_${grid}
                COMMAND test_electron_density_control_${dim} test.grid=${grid})
            set_tests_properties(test_${dim}_electron_density_control_${grid} PROPERTIES
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
        foreach(mode IN ITEMS decoupled_jfnk coupled_jfnk)
            foreach(pedestal IN ITEMS tracked static none)
                add_test(NAME test_${dim}_electron_density_control_${mode}_${pedestal}
                    COMMAND ${Python_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/analysis_density_control.py
                        $<TARGET_FILE:test_electron_density_control_runtime_${dim}>
                        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_${dim}_density_control
                        ${mode} ${pedestal})
                set_tests_properties(test_${dim}_electron_density_control_${mode}_${pedestal}
                    PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endif()
endforeach()

# Native curl -> Ampere -> nodal-current cap/seam consistency. The independent
# single-box reference includes physical guards, including on empty MPI ranks.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_darwin_curl_guards_${dim} test_darwin_curl_guards.cpp)
        target_link_libraries(test_darwin_curl_guards_${dim} PRIVATE lib_${dim})
        target_include_directories(test_darwin_curl_guards_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_darwin_curl_guards_${dim})
        endif()
        foreach(case IN ITEMS pmc periodic)
            if(dim STREQUAL "rz")
                set(curl_input inputs_test_rz_darwin_vacuum_recovery)
                set(curl_cells "amr.n_cell=16 32")
                if(case STREQUAL "pmc")
                    set(curl_boundaries "boundary.field_lo=none pmc" "boundary.field_hi=pec pmc"
                        "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
                else()
                    set(curl_boundaries "boundary.field_lo=none periodic" "boundary.field_hi=pec periodic"
                        "boundary.particle_lo=none periodic" "boundary.particle_hi=reflecting periodic")
                endif()
            else()
                set(curl_input inputs_test_3d_darwin_split_vacuum_energy)
                set(curl_cells "amr.n_cell=16 16 32")
                if(case STREQUAL "pmc")
                    set(curl_boundaries "boundary.field_lo=pec pec pmc" "boundary.field_hi=pec pec pmc"
                        "boundary.particle_lo=reflecting reflecting reflecting"
                        "boundary.particle_hi=reflecting reflecting reflecting")
                else()
                    set(curl_boundaries "boundary.field_lo=periodic periodic pmc"
                        "boundary.field_hi=periodic periodic pmc"
                        "boundary.particle_lo=periodic periodic reflecting"
                        "boundary.particle_hi=periodic periodic reflecting")
                endif()
            endif()
            foreach(grid IN ITEMS 8 16 64)
                file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/curl_guards_${dim}_${case}_${grid})
                add_test(NAME test_${dim}_darwin_curl_guards_${case}_${grid}
                    COMMAND test_darwin_curl_guards_${dim}
                        ${CMAKE_CURRENT_SOURCE_DIR}/${curl_input} ${curl_boundaries} ${curl_cells}
                        amr.max_grid_size=${grid} amr.blocking_factor=8 amr.refine_grid_layout=0
                        max_step=0 diagnostics.enable=0
                        hybrid_pic_model.electron_energy_mode=legacy
                        hybrid_pic_model.solve_electron_energy_equation=0
                        hybrid_pic_model.include_electron_inertia=0 hybrid_pic_model.elec_temp=0
                        hybrid_pic_model.darwin_vacuum_recovery=0 hybrid_pic_model.add_external_fields=0
                        hybrid_pic_model.darwin=0 implicit_evolve.darwin_segregated_solve=0)
                set_tests_properties(test_${dim}_darwin_curl_guards_${case}_${grid} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/curl_guards_${dim}_${case}_${grid}
                    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endif()
endforeach()

add_subdirectory(random_checkpoint_fixture)
add_subdirectory(final_gather_fixture)

# Native PMC trajectory/survivor continuity and physical absorption ledgers.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_absorbing_native_${dim} test_absorbing_native.cpp)
        target_link_libraries(test_absorbing_native_${dim} PRIVATE lib_${dim})
        target_include_directories(test_absorbing_native_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_absorbing_native_${dim})
        endif()
    endif()
endforeach()

# Optional diagnostic's native field-kernel oracle (run_checks.py owns arms).
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_eta_component_native_${dim}
            eta_component_fixture/test_eta_component_native.cpp)
        target_link_libraries(test_eta_component_native_${dim} PRIVATE lib_${dim})
        target_include_directories(test_eta_component_native_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_eta_component_native_${dim})
        endif()
    endif()
endforeach()

# Native alpha gyro-phase, radial-E orbit and crossed-field drift qualification.
add_subdirectory(alpha_orbit_fixture)

# Native raw eta capture across physical and decomposition corners.
add_subdirectory(source_gate_fixture)

# Completed implicit-orbit chord capture and cancellation-safe spline differences.
add_subdirectory(esirkepov_chord_fixture)

# Native Ohm pressure secants and actual vacuum-replacement row oracle.
if(TARGET lib_rz)
    add_executable(test_pressure_field_native_rz
        pressure_field_fixture/test_pressure_field_native.cpp)
    target_link_libraries(test_pressure_field_native_rz PRIVATE lib_rz)
    target_include_directories(test_pressure_field_native_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_pressure_field_native_rz)
    endif()
    foreach(grid IN ITEMS 8 64)
        set(name test_rz_pressure_recovery_rows_${grid})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name})
        add_test(NAME ${name} COMMAND test_pressure_field_native_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/pressure_field_fixture/inputs_native_pressure
            amr.max_grid_size=${grid} pressure_oracle.unequal_density_components=1
            pressure_oracle.recovery_rows=1 algo.current_deposition=esirkepov
            algo.particle_shape=3 algo.field_gathering=momentum-conserving)
        set_tests_properties(${name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
            TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# Pressure interpolation must retain nodal centering in inactive dimensions.
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_pressure_stagger_${dim}
            pressure_field_fixture/test_pressure_stagger.cpp)
        target_link_libraries(test_pressure_stagger_${dim} PRIVATE lib_${dim})
        target_include_directories(test_pressure_stagger_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_pressure_stagger_${dim})
        endif()
        add_test(NAME test_pressure_stagger_${dim} COMMAND test_pressure_stagger_${dim})
        set_tests_properties(test_pressure_stagger_${dim} PROPERTIES
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()

# Native endpoint-only RZ reflection, composed with existing axial absorption.
# This exercises the production audit, private OU endpoint and actual native
# finish/boundary/deposit kernels; it does not enable segmented wall orbits.
if(TARGET lib_rz)
    add_executable(test_reflecting_endpoint_native_rz test_reflecting_endpoint_native.cpp)
    target_link_libraries(test_reflecting_endpoint_native_rz PRIVATE lib_rz)
    target_include_directories(test_reflecting_endpoint_native_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_reflecting_endpoint_native_rz)
    endif()
    foreach(case IN ITEMS reflection corner axis4 all_velocities)
        set(extra_args)
        if(NOT case STREQUAL "reflection")
            list(APPEND extra_args radial_test.axial_displacement_cells=.4)
        endif()
        if(case STREQUAL "axis4")
            list(APPEND extra_args boundary.verboncoeur_axis_correction=0)
        endif()
        if(case STREQUAL "all_velocities")
            list(APPEND extra_args boundary.reflect_all_velocities=1)
        endif()
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/native_radial_${case})
        add_test(NAME native_radial_${case}_rz COMMAND test_reflecting_endpoint_native_rz
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_rz_absorbing_native ${extra_args})
        set_tests_properties(native_radial_${case}_rz PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/native_radial_${case}
            TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()

# All-component bounded RZ inverse: independent vector/scalar stencils,
# scalar-axis regularity and native MLMG for every nonperiodic cap parity.
if(TARGET lib_rz AND (WarpX_FFT OR ABLASTR_FFT))
    add_executable(test_recovery_green_components_rz test_recovery_green_components_rz.cpp)
    target_link_libraries(test_recovery_green_components_rz PRIVATE lib_rz)
    target_include_directories(test_recovery_green_components_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_recovery_green_components_rz)
    endif()
    foreach(component IN ITEMS 0 1 2)
        foreach(pair IN ITEMS 00 01 10 11)
            string(SUBSTRING "${pair}" 0 1 pmc_lo)
            string(SUBSTRING "${pair}" 1 1 pmc_hi)
            foreach(grid IN ITEMS 8 32)
                set(green_case test_rz_green_component_${component}_${pair}_${grid})
                add_test(NAME ${green_case} COMMAND test_recovery_green_components_rz
                    mg_test.component=${component} mg_test.grid=${grid}
                    mg_test.max_semi=1 mg_test.pmc_lo=${pmc_lo} mg_test.pmc_hi=${pmc_hi})
                set_tests_properties(${green_case} PROPERTIES
                    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
            endforeach()
        endforeach()
    endforeach()
endif()

# The current convention differs between the number flux and vacuum row.
foreach(D IN LISTS WarpX_DIMS)
    if(D STREQUAL "RZ" OR D STREQUAL "3")
        warpx_set_suffix_dims(SD ${D})
        add_executable(test_thermal_mm_current_${SD} test_thermal_mm_current.cpp)
        target_link_libraries(test_thermal_mm_current_${SD} PRIVATE lib_${SD})
        target_include_directories(test_thermal_mm_current_${SD} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_thermal_mm_current_${SD})
        endif()
        add_test(NAME thermal_mm_current_${SD} COMMAND test_thermal_mm_current_${SD})
        set_tests_properties(thermal_mm_current_${SD} PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()

# Resolve longitudinal corrections below an ulp of the raw current.
foreach(D IN LISTS WarpX_DIMS)
    if(D STREQUAL "RZ" OR D STREQUAL "3")
        warpx_set_suffix_dims(SD ${D})
        add_executable(test_source_projection_defect_${SD} test_source_projection_defect.cpp)
        target_link_libraries(test_source_projection_defect_${SD} PRIVATE lib_${SD})
        target_include_directories(test_source_projection_defect_${SD} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_source_projection_defect_${SD})
        endif()
        add_test(NAME source_projection_defect_${SD} COMMAND test_source_projection_defect_${SD})
        set_tests_properties(source_projection_defect_${SD} PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()

# Absolute current accuracy must remain meaningful for zero components.
foreach(D IN LISTS WarpX_DIMS)
    if(D STREQUAL "RZ" OR D STREQUAL "3")
        warpx_set_suffix_dims(SD ${D})
        add_executable(test_native_current_tolerance_${SD} test_native_current_tolerance.cpp)
        target_link_libraries(test_native_current_tolerance_${SD} PRIVATE lib_${SD})
        target_include_directories(test_native_current_tolerance_${SD} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_native_current_tolerance_${SD})
        endif()
        add_test(NAME native_current_tolerance_${SD} COMMAND test_native_current_tolerance_${SD})
        set_tests_properties(native_current_tolerance_${SD} PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()

# The direct longitudinal row preserves the native Yee metric and boundaries.
foreach(D IN LISTS WarpX_DIMS)
    if(D STREQUAL "RZ" OR D STREQUAL "3")
        warpx_set_suffix_dims(SD ${D})
        add_executable(test_direct_constraint_${SD}
            longitudinal_schur_fixture/test_longitudinal_schur.cpp)
        target_link_libraries(test_direct_constraint_${SD} PRIVATE lib_${SD})
        target_include_directories(test_direct_constraint_${SD} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_direct_constraint_${SD})
        endif()
        foreach(wall IN ITEMS pec pmc)
            foreach(cap IN ITEMS periodic pmc pec)
                foreach(box IN ITEMS 4 64)
                    set(test_name direct_constraint_${SD}_${wall}_${cap}_box${box})
                    add_test(NAME ${test_name} COMMAND test_direct_constraint_${SD}
                        test.wall=${wall} test.cap=${cap} test.box=${box}
                        amrex.verbose=0 tiny_profiler.enabled=0)
                    set_tests_properties(${test_name} PROPERTIES TIMEOUT 30
                        ENVIRONMENT "OMP_NUM_THREADS=1")
                endforeach()
            endforeach()
        endforeach()
    endif()
endforeach()
