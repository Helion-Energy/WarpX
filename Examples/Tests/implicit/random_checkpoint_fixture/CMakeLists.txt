find_package(Python3 REQUIRED COMPONENTS Interpreter)
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_thermal_random_checkpoint_${dim} test_random_checkpoint.cpp)
        target_link_libraries(test_thermal_random_checkpoint_${dim} PRIVATE lib_${dim})
        target_include_directories(test_thermal_random_checkpoint_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_thermal_random_checkpoint_${dim})
        endif()
        foreach(case IN ITEMS replay metadata truncated checksum missing)
            add_test(NAME test_${dim}_thermal_random_${case}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_case.py
                    $<TARGET_FILE:test_thermal_random_checkpoint_${dim}> ${case})
            set_tests_properties(test_${dim}_thermal_random_${case} PROPERTIES
                TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1" RUN_SERIAL TRUE)
        endforeach()
    endif()
endforeach()
