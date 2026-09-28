cmake_minimum_required(VERSION 3.24)
project(WarpXThermalTrialFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(test_thermal_trial test_thermal_trial.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/NativeConductionActivity.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalStageInitialGuess.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalStage.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalResidual.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalConductionPC.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalStageSolver.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverVec.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverDOF.cpp
    ${WARPX_SOURCE}/Source/ablastr/fields/MultiFabRegister.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/TextMsg.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/text/StringUtils.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/Communication.cpp)
target_link_libraries(test_thermal_trial PRIVATE AMReX::amrex)
target_compile_features(test_thermal_trial PRIVATE cxx_std_20)
target_compile_definitions(test_thermal_trial PRIVATE WARPX_DIM_RZ)
target_include_directories(test_thermal_trial PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../solver_layout_fixture ${WARPX_SOURCE}/Source)
target_compile_options(test_thermal_trial PRIVATE -Wall -Wextra -Werror=return-type)
enable_testing()
foreach(case IN ITEMS activity floors refinement channels native_edges initializer compression masked_balance)
    foreach(box IN ITEMS 8 128)
        add_test(NAME thermal_trial_${case}_box${box} COMMAND test_thermal_trial test.case=${case} test.box=${box})
        set_tests_properties(thermal_trial_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME thermal_trial_reject_implicit_closure COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py $<TARGET_FILE:test_thermal_trial>)
set_tests_properties(thermal_trial_reject_implicit_closure PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
