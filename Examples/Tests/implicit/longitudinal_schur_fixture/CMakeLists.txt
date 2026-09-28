cmake_minimum_required(VERSION 3.24)
project(WarpXLongitudinalSchurFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(SCHUR_GEOMETRY RZ CACHE STRING "RZ or 3D")
add_executable(test_longitudinal_schur test_longitudinal_schur.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/DarwinLongitudinalSchur.cpp)
target_link_libraries(test_longitudinal_schur PRIVATE AMReX::amrex)
target_compile_features(test_longitudinal_schur PRIVATE cxx_std_20)
target_include_directories(test_longitudinal_schur PRIVATE ${WARPX_SOURCE}/Source)
target_compile_definitions(test_longitudinal_schur PRIVATE WARPX_DIM_${SCHUR_GEOMETRY})
if(SCHUR_GEOMETRY STREQUAL RZ)
    target_compile_definitions(test_longitudinal_schur PRIVATE WARPX_ZINDEX=1)
else()
    target_compile_definitions(test_longitudinal_schur PRIVATE WARPX_ZINDEX=2)
endif()
target_compile_options(test_longitudinal_schur PRIVATE -Wall -Wextra -Werror=return-type)
enable_testing()
foreach(wall IN ITEMS pec pmc)
    foreach(cap IN ITEMS periodic pmc pec)
        foreach(box IN ITEMS 4 64)
            add_test(NAME schur_${wall}_${cap}_box${box}
                COMMAND test_longitudinal_schur test.wall=${wall} test.cap=${cap}
                    test.box=${box} test.stiffness=600 amrex.verbose=0 tiny_profiler.enabled=0)
            set_tests_properties(schur_${wall}_${cap}_box${box}
                PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
        endforeach()
    endforeach()
endforeach()
foreach(stiffness IN ITEMS 0 .06 6 60000)
    foreach(variable IN ITEMS 0 1)
        add_test(NAME schur_ladder_${stiffness}_variable${variable}
            COMMAND test_longitudinal_schur test.stiffness=${stiffness}
                test.variable=${variable} test.box=4 amrex.verbose=0 tiny_profiler.enabled=0)
        set_tests_properties(schur_ladder_${stiffness}_variable${variable}
            PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(form IN ITEMS theta bdf advective eb modes)
    add_test(NAME schur_reject_${form}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
            $<TARGET_FILE:test_longitudinal_schur> ${form})
endforeach()
add_test(NAME schur_inactive COMMAND test_longitudinal_schur test.variable=2 test.box=4
    amrex.verbose=0 tiny_profiler.enabled=0)
set_tests_properties(schur_inactive PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
if(SCHUR_GEOMETRY STREQUAL RZ)
    add_test(NAME schur_coord1 COMMAND test_longitudinal_schur test.coord=1 test.box=4
        amrex.verbose=0 tiny_profiler.enabled=0)
    set_tests_properties(schur_coord1 PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
else()
    foreach(box IN ITEMS 4 64)
        add_test(NAME schur_allperiodic_box${box} COMMAND test_longitudinal_schur
            test.all_periodic=1 test.box=${box} amrex.verbose=0 tiny_profiler.enabled=0)
        set_tests_properties(schur_allperiodic_box${box} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endif()
