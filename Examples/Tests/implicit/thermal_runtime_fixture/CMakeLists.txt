cmake_minimum_required(VERSION 3.24)
project(WarpXThermalRuntimeFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(THERMAL_RUNTIME_GEOMETRY RZ CACHE STRING "RZ or 3D")
enable_testing()
add_executable(test_thermal_runtime test_thermal_runtime.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/KineticThermalMoments.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermalRuntime.cpp
    ${WARPX_SOURCE}/Source/ablastr/fields/MultiFabRegister.cpp)
target_link_libraries(test_thermal_runtime PRIVATE AMReX::amrex)
target_compile_features(test_thermal_runtime PRIVATE cxx_std_20)
target_include_directories(test_thermal_runtime PRIVATE ${WARPX_SOURCE}/Source
    ${CMAKE_CURRENT_SOURCE_DIR}/../thermal_moments_fixture)
target_compile_definitions(test_thermal_runtime PRIVATE WARPX_DIM_${THERMAL_RUNTIME_GEOMETRY})
foreach(case IN ITEMS selector seed views serialization)
    foreach(box IN ITEMS 4 64)
        foreach(periodic IN ITEMS 0 1)
            add_test(NAME runtime_${case}_box${box}_periodic${periodic}
                COMMAND test_thermal_runtime test.case=${case} test.max_grid_size=${box}
                    test.periodic=${periodic})
            set_tests_properties(runtime_${case}_box${box}_periodic${periodic}
                PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endforeach()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(case IN ITEMS mode conflict eb driver amr floor gamma metadata ghost alias)
    add_test(NAME runtime_reject_${case}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
            $<TARGET_FILE:test_thermal_runtime> ${case} ${THERMAL_RUNTIME_GEOMETRY})
endforeach()

if(THERMAL_RUNTIME_GEOMETRY STREQUAL RZ)
    foreach(box IN ITEMS 4 64)
        foreach(periodic IN ITEMS 0 1)
            add_test(NAME runtime_seed_native_axis3_box${box}_periodic${periodic}
                COMMAND test_thermal_runtime test.case=seed test.max_grid_size=${box}
                    test.periodic=${periodic} test.corrected_axis=1)
            set_tests_properties(runtime_seed_native_axis3_box${box}_periodic${periodic}
                PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()
