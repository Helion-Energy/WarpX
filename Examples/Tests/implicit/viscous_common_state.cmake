# The serialized-state reader is deliberately restricted to native double
# precision. CUDA compilation is supported; GPU A/A determinism is unqualified.
if(TARGET lib_rz AND WarpX_PRECISION STREQUAL "DOUBLE"
   AND WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    add_executable(test_viscous_common_state_rz
        "${CMAKE_CURRENT_LIST_DIR}/test_viscous_common_state.cpp")
    target_link_libraries(test_viscous_common_state_rz PRIVATE lib_rz)
    target_include_directories(test_viscous_common_state_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_viscous_common_state_rz)
    endif()
    if(WarpX_COMPUTE STREQUAL "OMP" OR WarpX_COMPUTE STREQUAL "NOACC")
        find_package(Python COMPONENTS Interpreter REQUIRED)
        add_test(NAME test_rz_viscous_common_state
            COMMAND ${Python_EXECUTABLE}
                "${CMAKE_CURRENT_LIST_DIR}/analysis_viscous_common_state.py"
                --executable $<TARGET_FILE:test_viscous_common_state_rz>
                --output "${CMAKE_CURRENT_BINARY_DIR}/viscous_common_state"
                --new-run)
        set_tests_properties(test_rz_viscous_common_state PROPERTIES
            TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    endif()
endif()
