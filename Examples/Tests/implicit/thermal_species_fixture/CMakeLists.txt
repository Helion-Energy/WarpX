cmake_minimum_required(VERSION 3.24)
project(WarpXThermalSpeciesFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
enable_testing()
add_executable(test_thermal_species test_thermal_species.cpp
  ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/KineticThermalSpecies.cpp
  ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/EulerianThermalSources.cpp
  ${WARPX_SOURCE}/Source/ablastr/utils/Communication.cpp
  ${WARPX_SOURCE}/Source/ablastr/utils/TextMsg.cpp
  ${WARPX_SOURCE}/Source/ablastr/utils/text/StringUtils.cpp)
target_link_libraries(test_thermal_species PRIVATE AMReX::amrex)
target_compile_features(test_thermal_species PRIVATE cxx_std_20)
target_include_directories(test_thermal_species PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_thermal_species PRIVATE WARPX_DIM_RZ WARPX_ZINDEX=1)
target_compile_options(test_thermal_species PRIVATE -Wall -Wextra -Werror=return-type)
foreach(case IN ITEMS native direct temperature source derivative overlay_derivative aggregate purity invalid)
 foreach(box IN ITEMS 4 64)
  foreach(order IN ITEMS 1 3)
   add_test(NAME species_${case}_box${box}_shape${order}
    COMMAND test_thermal_species test.case=${case} test.max_grid_size=${box} test.shape=${order})
   set_tests_properties(species_${case}_box${box}_shape${order} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
  endforeach()
 endforeach()
endforeach()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(case IN ITEMS mm old descriptor eb)
 add_test(NAME species_reject_${case}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
  $<TARGET_FILE:test_thermal_species> ${case})
 set_tests_properties(species_reject_${case} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()
