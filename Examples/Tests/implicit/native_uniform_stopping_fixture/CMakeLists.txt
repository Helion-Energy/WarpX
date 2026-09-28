cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(WarpXNativeUniformStoppingFixture LANGUAGES CXX)
    enable_testing()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NATIVE_UNIFORM_STOPPING_EXECUTABLE)
    add_executable(test_native_uniform_stopping IMPORTED GLOBAL)
    set_target_properties(test_native_uniform_stopping PROPERTIES
        IMPORTED_LOCATION "${NATIVE_UNIFORM_STOPPING_EXECUTABLE}")
elseif(TARGET lib_3d)
    set(helper_dir ${WarpX_SOURCE_DIR}/Source/FieldSolver/ImplicitSolvers)
    add_executable(test_native_uniform_stopping main.cpp)
    target_link_libraries(test_native_uniform_stopping PRIVATE lib_3d)
    target_include_directories(test_native_uniform_stopping PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_uniform_stopping)
    endif()
    # Isolated test-only sign fault: no runtime production knob or shared-source edit.
    file(READ ${helper_dir}/NativeUniformStoppingEvent.cpp wrong_source)
    string(REGEX MATCHALL "old\\(i,j,k\\)\\+\\(value\\+ri\\(i,j,k\\)\\)/coefficient\\(i,j,k\\)" fault_sites "${wrong_source}")
    list(LENGTH fault_sites fault_count)
    if(NOT fault_count EQUAL 1)
        message(FATAL_ERROR "Reaction-sign fixture requires exactly one expected source expression")
    endif()
    string(REPLACE "old(i,j,k)+(value+ri(i,j,k))/coefficient(i,j,k)"
        "old(i,j,k)+(value-ri(i,j,k))/coefficient(i,j,k)" wrong_source "${wrong_source}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/WrongStoppingReaction.cpp "${wrong_source}")
    add_executable(test_native_stopping_wrong_reaction main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/WrongStoppingReaction.cpp)
    target_link_libraries(test_native_stopping_wrong_reaction PRIVATE lib_3d)
    target_include_directories(test_native_stopping_wrong_reaction PRIVATE
        ${WarpX_SOURCE_DIR}/Source ${helper_dir})
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_stopping_wrong_reaction)
    endif()
else()
    message(FATAL_ERROR "Requires lib_3d or NATIVE_UNIFORM_STOPPING_EXECUTABLE")
endif()
if(NATIVE_STOPPING_WRONG_REACTION_EXECUTABLE)
    add_executable(test_native_stopping_wrong_reaction IMPORTED GLOBAL)
    set_target_properties(test_native_stopping_wrong_reaction PROPERTIES
        IMPORTED_LOCATION "${NATIVE_STOPPING_WRONG_REACTION_EXECUTABLE}")
endif()
set(common --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs_3d)
function(stopping_case name)
    add_test(NAME native_stopping_${name} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_and_analyze.py
        --executable $<TARGET_FILE:test_native_uniform_stopping> ${common}
        --case ${name} ${ARGN})
    set_tests_properties(native_stopping_${name} PROPERTIES TIMEOUT 60
        ENVIRONMENT "OMP_NUM_THREADS=1")
endfunction()
stopping_case(single)
stopping_case(seam --override "amr.n_cell=16 16 16")
stopping_case(zero --override fixture.zero=1)
foreach(negative IN ITEMS budget speed nonuniform)
    stopping_case(${negative} --negative ${negative})
endforeach()
if(MPIEXEC_EXECUTABLE)
    stopping_case(mpi_seam --ranks 2 --mpi ${MPIEXEC_EXECUTABLE}
        --override "amr.n_cell=16 16 16")
    stopping_case(mpi_empty --ranks 2 --mpi ${MPIEXEC_EXECUTABLE})
    set_tests_properties(native_stopping_mpi_seam native_stopping_mpi_empty PROPERTIES PROCESSORS 2)
endif()
if(TARGET test_native_stopping_wrong_reaction)
    add_test(NAME native_stopping_reaction_sign COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_and_analyze.py
        --executable $<TARGET_FILE:test_native_stopping_wrong_reaction> ${common}
        --case reaction_sign --negative reaction)
    set_tests_properties(native_stopping_reaction_sign PROPERTIES TIMEOUT 60
        ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
