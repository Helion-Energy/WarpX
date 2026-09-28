cmake_minimum_required(VERSION 3.24)
project(WarpXParticleRollbackFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(ROLLBACK_GEOMETRY RZ CACHE STRING "RZ or 3D")
enable_testing()
add_executable(test_particle_rollback test_particle_rollback.cpp
    ../thermal_moments_fixture/test_moment_particles.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/KineticThermalMoments.cpp
    ${WARPX_SOURCE}/Source/BoundaryConditions/WarpX_PEC.cpp)
target_link_libraries(test_particle_rollback PRIVATE AMReX::amrex)
target_compile_features(test_particle_rollback PRIVATE cxx_std_20)
target_include_directories(test_particle_rollback PRIVATE ${WARPX_SOURCE}/Source
    ${CMAKE_CURRENT_SOURCE_DIR}/../thermal_moments_fixture)
target_compile_definitions(test_particle_rollback PRIVATE WARPX_DIM_${ROLLBACK_GEOMETRY})
if(ROLLBACK_GEOMETRY STREQUAL RZ)
    target_compile_definitions(test_particle_rollback PRIVATE WARPX_ZINDEX=1)
else()
    target_compile_definitions(test_particle_rollback PRIVATE WARPX_ZINDEX=2)
endif()
foreach(case IN ITEMS restore guards endpoint absorption migration)
    foreach(box IN ITEMS 4 64)
        foreach(sub IN ITEMS 0 1)
            add_test(NAME rollback_${case}_b${box}_sub${sub} COMMAND test_particle_rollback
                test.case=${case} test.max_grid_size=${box} test.suborbits=${sub})
            set_tests_properties(rollback_${case}_b${box}_sub${sub} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endforeach()
