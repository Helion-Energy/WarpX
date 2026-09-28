cmake_minimum_required(VERSION 3.24)
project(WarpXThermalCartesianFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(THERMAL_SOURCE "${WARPX_SOURCE}" CACHE PATH "Thermal Stage/PC source snapshot; defaults to current source")
set(THERMAL_GEOMETRY "3D" CACHE STRING "3D or RZ preservation snapshot")
if(THERMAL_GEOMETRY STREQUAL "3D")
    set(FIXTURE test_thermal_cartesian.cpp)
else()
    set(FIXTURE test_rz_snapshot.cpp)
endif()
add_executable(test_thermal_cartesian ${FIXTURE}
    ${THERMAL_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalStage.cpp
    ${THERMAL_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalResidual.cpp
    ${THERMAL_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalConductionPC.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalStageSolver.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ThermalStageInitialGuess.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverVec.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverDOF.cpp
    ${WARPX_SOURCE}/Source/ablastr/fields/MultiFabRegister.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/TextMsg.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/text/StringUtils.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/Communication.cpp)
target_link_libraries(test_thermal_cartesian PRIVATE AMReX::amrex)
target_compile_features(test_thermal_cartesian PRIVATE cxx_std_20)
target_compile_definitions(test_thermal_cartesian PRIVATE WARPX_DIM_${THERMAL_GEOMETRY})
target_include_directories(test_thermal_cartesian PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../solver_layout_fixture ${THERMAL_SOURCE}/Source ${WARPX_SOURCE}/Source ${WARPX_SOURCE}/Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel)
target_compile_options(test_thermal_cartesian PRIVATE -Wall -Wextra -Werror=return-type)
enable_testing()
if(THERMAL_GEOMETRY STREQUAL "3D")
    foreach(case IN ITEMS constant native_tensor spatial temporal root nonlinear contact rows boundary compression)
        foreach(box IN ITEMS 6 64)
            add_test(NAME thermal_cartesian_${case}_box${box} COMMAND test_thermal_cartesian test.case=${case} test.box=${box})
            set_tests_properties(thermal_cartesian_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()

if(THERMAL_GEOMETRY STREQUAL "3D")
    add_executable(test_activity_cartesian test_activity_cartesian.cpp
        ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/NativeConductionActivity.cpp
        ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalStage.cpp
        ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalResidual.cpp)
    target_link_libraries(test_activity_cartesian PRIVATE AMReX::amrex)
    target_compile_features(test_activity_cartesian PRIVATE cxx_std_20)
    target_compile_definitions(test_activity_cartesian PRIVATE WARPX_DIM_3D)
    target_include_directories(test_activity_cartesian PRIVATE ${WARPX_SOURCE}/Source)
    foreach(case IN ITEMS gate refinement masked)
        foreach(box IN ITEMS 4 64)
            add_test(NAME thermal_activity3d_${case}_box${box} COMMAND test_activity_cartesian test.case=${case} test.box=${box})
            set_tests_properties(thermal_activity3d_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endif()
