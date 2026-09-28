# In a full WarpX build, link the production libraries. The standalone mode
# registers the same tests for immutable, independently compiled native probes.
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NativeAlphaOrbitQualification LANGUAGES NONE)
    enable_testing()
endif()
find_package(Python COMPONENTS Interpreter REQUIRED)
foreach(dim IN ITEMS rz 3d)
    if(TARGET lib_${dim})
        add_executable(test_alpha_orbits_${dim} test_alpha_orbits.cpp)
        target_link_libraries(test_alpha_orbits_${dim} PRIVATE lib_${dim})
        target_include_directories(test_alpha_orbits_${dim} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_alpha_orbits_${dim})
        endif()
    elseif(DEFINED ALPHA_ORBIT_${dim}_BINARY)
        add_executable(test_alpha_orbits_${dim} IMPORTED)
        set_target_properties(test_alpha_orbits_${dim} PROPERTIES
            IMPORTED_LOCATION "${ALPHA_ORBIT_${dim}_BINARY}")
    endif()
    if(TARGET test_alpha_orbits_${dim})
        set(cases gyro circular)
        if(dim STREQUAL "3d")
            list(APPEND cases drift)
        endif()
        foreach(case IN LISTS cases)
            set(directory ${CMAKE_CURRENT_BINARY_DIR}/${dim}_${case})
            file(MAKE_DIRECTORY ${directory})
            add_test(NAME test_${dim}_alpha_orbit_${case}
                COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_alpha_orbits.py
                    $<TARGET_FILE:test_alpha_orbits_${dim}>
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_${dim} ${case})
            set_tests_properties(test_${dim}_alpha_orbit_${case} PROPERTIES
                WORKING_DIRECTORY ${directory} TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()
