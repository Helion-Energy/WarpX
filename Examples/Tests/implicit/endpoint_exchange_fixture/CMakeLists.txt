cmake_minimum_required(VERSION 3.24)
project(EndpointExchangeFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(test_endpoint_exchange test_endpoint_exchange.cpp
 ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ImplicitIonEndpointTrial.cpp
 ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/IonExchangeMomentAudit.cpp
 ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ExpectedIonExchangeSource.cpp
 ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/AcceptedIonExchange.cpp
 ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ExpectedIonExchangeChannel.cpp
 ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/ExpectedIonThermalPartner.cpp)
target_link_libraries(test_endpoint_exchange PRIVATE AMReX::amrex)
target_compile_features(test_endpoint_exchange PRIVATE cxx_std_20)
target_include_directories(test_endpoint_exchange PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_endpoint_exchange PRIVATE WARPX_DIM_RZ WARPX_ZINDEX=1)
enable_testing()
foreach(case IN ITEMS endpoint moments purity invalid guard drift partner finite_step finite_step_species derivative noise)
 foreach(box IN ITEMS 4 64)
  add_test(NAME endpoint_${case}_box${box} COMMAND test_endpoint_exchange test.case=${case} test.max_grid_size=${box})
  set_tests_properties(endpoint_${case}_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
 endforeach()
endforeach()
