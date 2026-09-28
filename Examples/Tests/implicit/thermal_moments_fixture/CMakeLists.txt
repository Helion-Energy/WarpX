cmake_minimum_required(VERSION 3.24)
project(WarpXThermalMomentsFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(MOMENT_GEOMETRY RZ CACHE STRING "RZ or 3D")
enable_testing()
add_executable(test_thermal_moments test_thermal_moments.cpp test_moment_particles.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/KineticThermalMoments.cpp
    ${WARPX_SOURCE}/Source/BoundaryConditions/WarpX_PEC.cpp)
target_link_libraries(test_thermal_moments PRIVATE AMReX::amrex)
target_compile_features(test_thermal_moments PRIVATE cxx_std_20)
target_include_directories(test_thermal_moments PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_thermal_moments PRIVATE WARPX_DIM_${MOMENT_GEOMETRY})
if(MOMENT_GEOMETRY STREQUAL RZ)
    target_compile_definitions(test_thermal_moments PRIVATE WARPX_ZINDEX=1)
else()
    target_compile_definitions(test_thermal_moments PRIVATE WARPX_ZINDEX=2)
endif()
target_compile_options(test_thermal_moments PRIVATE -Wall -Wextra -Werror=return-type)
foreach(case IN ITEMS continuity pressure derivatives purity particles closed_faces)
    foreach(box IN ITEMS 4 64)
        foreach(periodic IN ITEMS 0 1)
            add_test(NAME moments_${case}_box${box}_periodic${periodic}
                COMMAND test_thermal_moments test.case=${case} test.max_grid_size=${box}
                    test.periodic=${periodic})
            set_tests_properties(moments_${case}_box${box}_periodic${periodic}
                PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endforeach()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(rejected_cases floor boundary layout alias)
if(MOMENT_GEOMETRY STREQUAL RZ)
    list(APPEND rejected_cases current_boundary modes)
endif()
foreach(case IN LISTS rejected_cases)
    add_test(NAME moments_reject_${case}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
            $<TARGET_FILE:test_thermal_moments> ${case})
endforeach()

if(MOMENT_GEOMETRY STREQUAL RZ)
    foreach(case IN ITEMS continuity pressure derivatives purity particles closed_faces)
        foreach(box IN ITEMS 4 64)
            foreach(periodic IN ITEMS 0 1)
                foreach(axis IN ITEMS 0 1)
                    add_test(NAME moments_native_${case}_box${box}_periodic${periodic}_axis${axis}
                        COMMAND test_thermal_moments test.case=${case} test.max_grid_size=${box}
                            test.periodic=${periodic} test.corrected_axis=${axis} test.reflecting=1)
                    set_tests_properties(moments_native_${case}_box${box}_periodic${periodic}_axis${axis}
                        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endif()
