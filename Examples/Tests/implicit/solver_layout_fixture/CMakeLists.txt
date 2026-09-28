cmake_minimum_required(VERSION 3.24)
project(WarpXSolverLayoutFixture LANGUAGES C CXX)
find_package(AMReX CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
get_filename_component(WARPX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
enable_testing()
add_executable(test_solver_layout
    test_solver_layout.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverVec.cpp
    ${WARPX_SOURCE}/Source/FieldSolver/ImplicitSolvers/WarpXSolverDOF.cpp
    ${WARPX_SOURCE}/Source/ablastr/fields/MultiFabRegister.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/TextMsg.cpp
    ${WARPX_SOURCE}/Source/ablastr/utils/text/StringUtils.cpp)
target_link_libraries(test_solver_layout PRIVATE AMReX::amrex)
target_compile_features(test_solver_layout PRIVATE cxx_std_20)
target_include_directories(test_solver_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${WARPX_SOURCE}/Source)
target_compile_definitions(test_solver_layout PRIVATE WARPX_DIM_${LAYOUT_GEOMETRY})
target_compile_options(test_solver_layout PRIVATE -Wall -Wextra -Werror=return-type)
foreach(box IN ITEMS 4 64)
    foreach(periodic IN ITEMS 0 1)
        add_test(NAME solver_layout_box${box}_periodic${periodic}
            COMMAND test_solver_layout test.max_grid_size=${box} test.periodic=${periodic})
        set_tests_properties(solver_layout_box${box}_periodic${periodic}
            PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
    endforeach()
endforeach()
foreach(case IN ITEMS empty duplicate scalar_alias nan_scale inf_scale zero_scale bad_layout)
    add_test(NAME solver_layout_reject_${case}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/expect_reject.py
            $<TARGET_FILE:test_solver_layout> ${case})
    set_tests_properties(solver_layout_reject_${case}
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach()

# Optional exact legacy oracle: pass the read-only baseline source to compare
# bit representations of single-layout E/phi arithmetic and packed values.
if(LAYOUT_REFERENCE_SOURCE)
    foreach(variant IN ITEMS candidate reference)
        if(variant STREQUAL reference)
            set(layout_source ${LAYOUT_REFERENCE_SOURCE})
        else()
            set(layout_source ${WARPX_SOURCE})
        endif()
        add_executable(legacy_${variant}
            test_legacy_compat.cpp
            ${layout_source}/Source/FieldSolver/ImplicitSolvers/WarpXSolverVec.cpp
            ${layout_source}/Source/FieldSolver/ImplicitSolvers/WarpXSolverDOF.cpp
            ${layout_source}/Source/ablastr/fields/MultiFabRegister.cpp
            ${layout_source}/Source/ablastr/utils/TextMsg.cpp
            ${layout_source}/Source/ablastr/utils/text/StringUtils.cpp)
        target_link_libraries(legacy_${variant} PRIVATE AMReX::amrex)
        target_compile_features(legacy_${variant} PRIVATE cxx_std_20)
        target_include_directories(legacy_${variant} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${layout_source}/Source)
        target_compile_definitions(legacy_${variant} PRIVATE WARPX_DIM_${LAYOUT_GEOMETRY})
    endforeach()
endif()
