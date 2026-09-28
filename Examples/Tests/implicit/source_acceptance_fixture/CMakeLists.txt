cmake_minimum_required(VERSION 3.24)
project(SourceAcceptanceFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(test_source_acceptance test_source_acceptance.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalSources.cpp)
target_link_libraries(test_source_acceptance PRIVATE AMReX::amrex)
target_compile_features(test_source_acceptance PRIVATE cxx_std_20)
target_include_directories(test_source_acceptance PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_source_acceptance PRIVATE WARPX_DIM_RZ WARPX_ZINDEX=1)
enable_testing()
foreach(case IN ITEMS smooth cooling orphan_viscous orphan_hyper cancellation invalid)
    foreach(box IN ITEMS 4 64)
        add_test(NAME source_acceptance_${case}_box${box}
            COMMAND test_source_acceptance test.case=${case} test.box=${box})
        set_tests_properties(source_acceptance_${case}_box${box}
            PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()
