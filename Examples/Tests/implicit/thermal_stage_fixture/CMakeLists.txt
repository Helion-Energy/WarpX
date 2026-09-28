cmake_minimum_required(VERSION 3.24)
project(WarpXThermalStageFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
enable_testing()
add_library(thermal_fixture_core STATIC
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalEnergyBalance.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalStage.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalResidual.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalConductionPC.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalStageSolver.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverVec.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverDOF.cpp
    ${WARPX_SOURCE}/Source/ablastr/fields/MultiFabRegister.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/TextMsg.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/text/StringUtils.cpp)
target_link_libraries(thermal_fixture_core PUBLIC AMReX::amrex)
target_compile_features(thermal_fixture_core PUBLIC cxx_std_20)
target_include_directories(thermal_fixture_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../solver_layout_fixture ${WARPX_SOURCE}/Source)
target_compile_definitions(thermal_fixture_core PUBLIC WARPX_DIM_RZ)
target_compile_options(thermal_fixture_core PUBLIC -Wall -Wextra -Werror=return-type)
add_executable(test_thermal_stage test_thermal_stage.cpp)
target_link_libraries(test_thermal_stage PRIVATE thermal_fixture_core)
add_executable(test_coupled_driver test_coupled_driver.cpp)
target_link_libraries(test_coupled_driver PRIVATE thermal_fixture_core)
foreach(case IN ITEMS coupled pc_off reference weak initial probe nonfinite freeze pc final limit)
    foreach(box IN ITEMS 4 32)
        add_test(NAME driver_${case}_box${box}
            COMMAND test_coupled_driver test.case=${case} test.max_grid_size=${box})
        set_tests_properties(driver_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()
foreach(case IN ITEMS coupled pc_off reference weak initial probe nonfinite freeze pc final limit)
    foreach(box IN ITEMS 4 32)
        add_test(NAME driver_adaptive_${case}_box${box}
            COMMAND test_coupled_driver test.case=${case} test.max_grid_size=${box}
                test.adaptive_forcing=1)
        set_tests_properties(driver_adaptive_${case}_box${box} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()
foreach(case IN ITEMS eigen constant nonlinear cap density boundary rows rollback source invalid transport compression density_context source_callback leg_law leg_root manufactured mg_controls)
    foreach(box IN ITEMS 8 64)
        add_test(NAME thermal_${case}_box${box}
            COMMAND test_thermal_stage test.case=${case} test.max_grid_size=${box})
        set_tests_properties(thermal_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(case IN ITEMS leg bfloor density periodic_wall)
    add_test(NAME thermal_reject_${case}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
            $<TARGET_FILE:test_thermal_stage> ${case})
    set_tests_properties(thermal_reject_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()

# Re-run the preceding extracted-kernel donor/stage/FD4 regression because this
# slice improves cancellation in that shared derivative accumulation.
add_executable(test_eulerian_thermal_stage ../test_eulerian_thermal_stage.cpp)
target_link_libraries(test_eulerian_thermal_stage PRIVATE AMReX::amrex)
target_compile_features(test_eulerian_thermal_stage PRIVATE cxx_std_20)
target_include_directories(test_eulerian_thermal_stage PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_eulerian_thermal_stage PRIVATE WARPX_DIM_RZ)
add_test(NAME thermal_donor_stage_regression COMMAND test_eulerian_thermal_stage)
set_tests_properties(thermal_donor_stage_regression PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

# Optional exact oracle from the audited MHD checkout. Never copies or changes it.
set(MHD_REFERENCE_DIR "" CACHE PATH "Pinned resistive_mhd source for exact transport oracle")
if(MHD_REFERENCE_DIR)
    add_executable(test_thermal_transport_mhd test_thermal_transport_mhd.cpp)
    target_link_libraries(test_thermal_transport_mhd PRIVATE AMReX::amrex)
    target_compile_features(test_thermal_transport_mhd PRIVATE cxx_std_20)
    target_include_directories(test_thermal_transport_mhd PRIVATE ${WARPX_SOURCE}/Source
        ${MHD_REFERENCE_DIR}/Source/FieldSolver/ImplicitSolvers)
    target_compile_definitions(test_thermal_transport_mhd PRIVATE WARPX_DIM_RZ)
    add_test(NAME thermal_transport_pinned_mhd COMMAND test_thermal_transport_mhd)
    set_tests_properties(thermal_transport_pinned_mhd PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

add_executable(test_thermal_balance test_thermal_balance.cpp)
target_link_libraries(test_thermal_balance PRIVATE thermal_fixture_core)
foreach(box IN ITEMS 4 32)
    add_test(NAME thermal_energy_receipt_box${box}
        COMMAND test_thermal_balance test.max_grid_size=${box})
    set_tests_properties(thermal_energy_receipt_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()

add_executable(test_field_rollback test_field_rollback.cpp)
target_link_libraries(test_field_rollback PRIVATE thermal_fixture_core)
foreach(box IN ITEMS 4 32)
    add_test(NAME field_rollback_box${box}
        COMMAND test_field_rollback test.max_grid_size=${box})
    set_tests_properties(field_rollback_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()
