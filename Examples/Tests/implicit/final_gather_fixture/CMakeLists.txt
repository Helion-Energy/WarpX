find_package(Python COMPONENTS Interpreter REQUIRED)
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_final_gather_capture_${dim} ../test_final_gather_capture.cpp)
        target_link_libraries(test_final_gather_capture_${dim} PRIVATE lib_${dim})
        target_include_directories(test_final_gather_capture_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_final_gather_capture_${dim})
        endif()
        foreach(case IN ITEMS pair crop iterated negative)
            set(directory ${CMAKE_CURRENT_BINARY_DIR}/${dim}_${case})
            file(MAKE_DIRECTORY ${directory})
            add_test(NAME test_${dim}_final_gather_${case}
                COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_final_gather.py
                    $<TARGET_FILE:test_final_gather_capture_${dim}>
                    ${CMAKE_CURRENT_SOURCE_DIR}/../inputs_test_${dim}_density_control ${case})
            set_tests_properties(test_${dim}_final_gather_${case} PROPERTIES
                WORKING_DIRECTORY ${directory} TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()
