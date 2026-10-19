# Native Cartesian thermal qualification, separate from the RZ metric oracles.
if(TARGET lib_3d)
    foreach(fixture IN ITEMS boundary_native cartesian_order qdsmc_rkl_retry)
        if(fixture STREQUAL "boundary_native")
            set(test_target test_conduction_boundary_native_3d)
            set(test_source test_conduction_boundary_native_3d.cpp)
        elseif(fixture STREQUAL "cartesian_order")
            set(test_target test_conduction_cartesian_order_3d)
            set(test_source test_conduction_cartesian_order.cpp)
        else()
            set(test_target test_qdsmc_rkl_retry_3d)
            set(test_source test_qdsmc_rkl_retry.cpp)
        endif()
        add_executable(${test_target} ${test_source})
        target_link_libraries(${test_target} PRIVATE lib_3d)
        target_include_directories(${test_target} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(${test_target})
        endif()
    endforeach()
    find_package(Python REQUIRED COMPONENTS Interpreter)
    add_test(NAME test_3d_conduction_boundary
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_conduction_boundary_3d.py
            --executable $<TARGET_FILE:test_conduction_boundary_native_3d>
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/conduction_boundary_3d)
    add_test(NAME test_3d_conduction_cartesian_order
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_conduction_cartesian_order.py
            --executable $<TARGET_FILE:test_conduction_cartesian_order_3d>
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/conduction_cartesian_order)
    set_tests_properties(test_3d_conduction_boundary test_3d_conduction_cartesian_order PROPERTIES
        TIMEOUT 1200 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "conduction;3d;native")
    add_test(NAME test_3d_qdsmc_rkl_retry COMMAND test_qdsmc_rkl_retry_3d)
    set_tests_properties(test_3d_qdsmc_rkl_retry PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "conduction;3d;native")
    if(WarpX_MPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME test_3d_qdsmc_rkl_retry_mpi
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_qdsmc_rkl_retry_3d> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(test_3d_qdsmc_rkl_retry_mpi PROPERTIES
            TIMEOUT 120 PROCESSORS 4 ENVIRONMENT "OMP_NUM_THREADS=1"
            LABELS "conduction;3d;native;mpi")
    endif()
    foreach(fixture IN ITEMS qdsmc_source_stage_order qdsmc_source_stage_capacity
                             qdsmc_density_floor_sync braginskii_conduction)
        set(test_target test_${fixture}_3d)
        if(fixture STREQUAL "braginskii_conduction")
            set(test_source test_braginskii_conduction_3d.cpp)
        else()
            set(test_source test_${fixture}.cpp)
        endif()
        add_executable(${test_target} ${test_source})
        target_link_libraries(${test_target} PRIVATE lib_3d)
        target_include_directories(${test_target} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(${test_target})
        endif()
    endforeach()
    add_test(NAME test_3d_eee_source_stage
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_eee_source_3d.py
            --order-executable $<TARGET_FILE:test_qdsmc_source_stage_order_3d>
            --capacity-executable $<TARGET_FILE:test_qdsmc_source_stage_capacity_3d>
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/eee_source_3d)
    add_test(NAME test_3d_thermal_floor_sync
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_thermal_matrix_3d.py
            --executable $<TARGET_FILE:test_qdsmc_density_floor_sync_3d>
            --manifest ${CMAKE_CURRENT_SOURCE_DIR}/thermal_floor_3d_cases.json
            --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs_3d_floor_sync
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/thermal_floor_3d)
    add_test(NAME test_3d_thermal_physical
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_thermal_matrix_3d.py
            --executable $<TARGET_FILE:test_braginskii_conduction_3d>
            --manifest ${CMAKE_CURRENT_SOURCE_DIR}/thermal_physical_3d_cases.json
            --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs_3d_braginskii
            --output-parent ${CMAKE_CURRENT_BINARY_DIR}/thermal_physical_3d)
    set_tests_properties(test_3d_eee_source_stage test_3d_thermal_floor_sync
                        test_3d_thermal_physical PROPERTIES
        TIMEOUT 3600 ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "conduction;3d;native;slow")
endif()

# Fresh AmrCore level creation installs the hierarchy BoxArray after this
# geometry initialization. Exercise single- and multi-box layouts explicitly.
if(TARGET lib_3d AND WarpX_EB)
    add_executable(test_ect_layout_3d test_ect_layout_3d.cpp)
    target_link_libraries(test_ect_layout_3d PRIVATE lib_3d)
    target_include_directories(test_ect_layout_3d PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_ect_layout_3d)
    endif()
    foreach(grid IN ITEMS 8 16)
        set(name test_3d_ect_fresh_layout_${grid})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name})
        add_test(NAME ${name} COMMAND test_ect_layout_3d
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_coupled_port
            amr.max_grid_size=${grid} max_step=0 diagnostics.enable=0
            hybrid_pic_model.use_conformal_eb=1
            amrex.the_arena_init_size=0 amrex.the_arena_is_managed=0)
        set_tests_properties(${name} PROPERTIES TIMEOUT 120
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
            ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "3d;native;ect")
    endforeach()
endif()
