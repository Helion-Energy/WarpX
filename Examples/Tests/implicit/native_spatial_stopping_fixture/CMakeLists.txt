cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(WarpXNativeSpatialStoppingFixture LANGUAGES CXX)
    enable_testing()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NATIVE_SPATIAL_STOPPING_EXECUTABLE)
    add_executable(test_native_spatial_stopping IMPORTED GLOBAL)
    set_target_properties(test_native_spatial_stopping PROPERTIES
        IMPORTED_LOCATION "${NATIVE_SPATIAL_STOPPING_EXECUTABLE}")
elseif(TARGET lib_3d)
    set(helper_dir ${WarpX_SOURCE_DIR}/Source/FieldSolver/ImplicitSolvers)
    add_executable(test_native_spatial_stopping main.cpp)
    target_link_libraries(test_native_spatial_stopping PRIVATE lib_3d)
    target_include_directories(test_native_spatial_stopping PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_spatial_stopping)
    endif()
    # Compile-only fault copy; no production runtime knob or shared source edit.
    file(READ ${helper_dir}/NativeSpatialStoppingEvent.cpp wrong_source)
    string(REGEX MATCHALL "Sync\\(b1,g\\);" fault_sites "${wrong_source}")
    list(LENGTH fault_sites fault_count)
    if(NOT fault_count EQUAL 1)
        message(FATAL_ERROR "Magnetic-sign fixture requires exactly one expected source expression")
    endif()
    string(REPLACE "Sync(b1,g);"
        "for(auto& field:b1){field.mult(-1.,0,1,field.nGrow());}Sync(b1,g);"
        wrong_source "${wrong_source}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/WrongSpatialMagnetic.cpp "${wrong_source}")
    add_executable(test_native_spatial_wrong_magnetic main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/WrongSpatialMagnetic.cpp)
    target_link_libraries(test_native_spatial_wrong_magnetic PRIVATE lib_3d)
    target_include_directories(test_native_spatial_wrong_magnetic PRIVATE
        ${WarpX_SOURCE_DIR}/Source ${helper_dir})
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_spatial_wrong_magnetic)
    endif()
else()
    message(FATAL_ERROR "Requires lib_3d or NATIVE_SPATIAL_STOPPING_EXECUTABLE")
endif()
if(NATIVE_SPATIAL_WRONG_MAGNETIC_EXECUTABLE)
    add_executable(test_native_spatial_wrong_magnetic IMPORTED GLOBAL)
    set_target_properties(test_native_spatial_wrong_magnetic PROPERTIES
        IMPORTED_LOCATION "${NATIVE_SPATIAL_WRONG_MAGNETIC_EXECUTABLE}")
endif()
set(common --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs_3d)
function(stopping_case name)
    add_test(NAME native_spatial_stopping_${name} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_and_analyze.py
        --executable $<TARGET_FILE:test_native_spatial_stopping> ${common}
        --case ${name} ${ARGN})
    set_tests_properties(native_spatial_stopping_${name} PROPERTIES TIMEOUT 240
        ENVIRONMENT "OMP_NUM_THREADS=1")
endfunction()
stopping_case(transverse)
stopping_case(seam --override "amr.n_cell=16 16 16")
stopping_case(mixed --override fixture.mixed=1)
stopping_case(mixed_seam --override fixture.mixed=1 --override "amr.n_cell=16 16 16")
stopping_case(harmonic --override fixture.harmonic=1)
stopping_case(zero --override fixture.zero=1)
foreach(negative IN ITEMS budget speed nonuniform external_current external_particle)
    stopping_case(${negative} --negative ${negative})
endforeach()
if(MPIEXEC_EXECUTABLE)
    stopping_case(mpi_seam --ranks 2 --mpi ${MPIEXEC_EXECUTABLE}
        --override "amr.n_cell=16 16 16")
    stopping_case(mpi_mixed_seam --ranks 2 --mpi ${MPIEXEC_EXECUTABLE}
        --override fixture.mixed=1 --override "amr.n_cell=16 16 16")
    stopping_case(mpi_mixed_empty --ranks 2 --mpi ${MPIEXEC_EXECUTABLE}
        --override fixture.mixed=1)
    set_tests_properties(native_spatial_stopping_mpi_seam native_spatial_stopping_mpi_mixed_seam
        native_spatial_stopping_mpi_mixed_empty PROPERTIES PROCESSORS 2)
endif()
if(TARGET test_native_spatial_wrong_magnetic)
    add_test(NAME native_spatial_stopping_faraday_sign COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_and_analyze.py
        --executable $<TARGET_FILE:test_native_spatial_wrong_magnetic> ${common}
        --case faraday_sign --negative faraday)
    set_tests_properties(native_spatial_stopping_faraday_sign PROPERTIES TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
