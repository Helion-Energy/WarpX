cmake_minimum_required(VERSION 3.24)
project(WarpXThermalSourcesFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
enable_testing()
add_executable(test_thermal_sources test_thermal_sources.cpp
  ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalSources.cpp)
target_link_libraries(test_thermal_sources PRIVATE AMReX::amrex)
target_compile_features(test_thermal_sources PRIVATE cxx_std_20)
target_include_directories(test_thermal_sources PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_thermal_sources PRIVATE WARPX_DIM_RZ)
target_compile_options(test_thermal_sources PRIVATE -Wall -Wextra -Werror=return-type)
foreach(case IN ITEMS rates work derivative gates stopping purity invalid)
  foreach(box IN ITEMS 4 64)
    add_test(NAME thermal_sources_${case}_box${box}
      COMMAND test_thermal_sources test.case=${case} test.max_grid_size=${box})
    set_tests_properties(thermal_sources_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
  endforeach()
endforeach()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(case IN ITEMS eb amr shunt pedestal kickcap parser alias)
  add_test(NAME thermal_sources_reject_${case}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
      $<TARGET_FILE:test_thermal_sources> ${case})
  set_tests_properties(thermal_sources_reject_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()

# Optional composition with the sibling's production conservative scalar map.
# The default is useful after both scoped commits are integrated by the parent.
set(THERMAL_MOMENTS_SOURCE_DIR "${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers" CACHE PATH
    "Directory containing the separately delivered KineticThermalMoments sources")
if(EXISTS "${THERMAL_MOMENTS_SOURCE_DIR}/KineticThermalMoments.cpp")
  target_sources(test_thermal_sources PRIVATE ${THERMAL_MOMENTS_SOURCE_DIR}/KineticThermalMoments.cpp)
  target_include_directories(test_thermal_sources PRIVATE ${THERMAL_MOMENTS_SOURCE_DIR})
  target_compile_definitions(test_thermal_sources PRIVATE TEST_THERMAL_MOMENTS=1)
endif()

add_executable(test_eulerian_dissipation test_eulerian_dissipation.cpp
  ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianDissipation.cpp
  ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalSources.cpp
  ${WARPX_SOURCE}/Source/BoundaryConditions/WarpX_PEC.cpp)
target_link_libraries(test_eulerian_dissipation PRIVATE AMReX::amrex)
target_compile_features(test_eulerian_dissipation PRIVATE cxx_std_20)
target_include_directories(test_eulerian_dissipation PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_eulerian_dissipation PRIVATE WARPX_DIM_RZ)
target_compile_options(test_eulerian_dissipation PRIVATE -Wall -Wextra -Werror=return-type)
foreach(case IN ITEMS viscosity braginskii derivative hyper curl_adjoint purity invalid pmc_projection pmc_sources pmc_guards)
  foreach(box IN ITEMS 4 64)
    add_test(NAME dissipation_${case}_box${box}
      COMMAND test_eulerian_dissipation test.case=${case} test.max_grid_size=${box})
    set_tests_properties(dissipation_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
  endforeach()
endforeach()

foreach(case IN ITEMS eb amr mode transform boundary limited parser alias)
  add_test(NAME dissipation_reject_${case}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_dissipation_reject.py
      $<TARGET_FILE:test_eulerian_dissipation> ${case})
  set_tests_properties(dissipation_reject_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()

if(EXISTS "${THERMAL_MOMENTS_SOURCE_DIR}/KineticThermalMoments.cpp")
  add_executable(test_stopping_transfer test_stopping_transfer.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianStoppingTransfer.cpp
    ${THERMAL_MOMENTS_SOURCE_DIR}/KineticThermalMoments.cpp)
  target_link_libraries(test_stopping_transfer PRIVATE AMReX::amrex)
  target_compile_features(test_stopping_transfer PRIVATE cxx_std_20)
  target_include_directories(test_stopping_transfer PRIVATE ${WARPX_SOURCE}/Source ${THERMAL_MOMENTS_SOURCE_DIR})
  target_compile_definitions(test_stopping_transfer PRIVATE WARPX_DIM_RZ)
  target_compile_options(test_stopping_transfer PRIVATE -Wall -Wextra -Werror=return-type)
  foreach(case IN ITEMS conservation gates floor counterexample purity invalid)
    foreach(box IN ITEMS 4 64)
      add_test(NAME stopping_${case}_box${box}
        COMMAND test_stopping_transfer test.case=${case} test.max_grid_size=${box})
      set_tests_properties(stopping_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
  endforeach()
  foreach(case IN ITEMS eb amr mode gamma alias)
    add_test(NAME stopping_reject_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_stopping_reject.py
        $<TARGET_FILE:test_stopping_transfer> ${case})
    set_tests_properties(stopping_reject_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
  endforeach()
endif()
