cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(WarpXNativeRZSpatialStoppingFixture LANGUAGES CXX)
    enable_testing()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NATIVE_RZ_STOPPING_EXECUTABLE)
    add_executable(test_native_rz_stopping IMPORTED GLOBAL)
    set_target_properties(test_native_rz_stopping PROPERTIES
        IMPORTED_LOCATION "${NATIVE_RZ_STOPPING_EXECUTABLE}")
elseif(TARGET lib_rz)
    set(helper_dir ${WarpX_SOURCE_DIR}/Source/FieldSolver/ImplicitSolvers)
    add_executable(test_native_rz_stopping main.cpp)
    target_link_libraries(test_native_rz_stopping PRIVATE lib_rz)
    target_include_directories(test_native_rz_stopping PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_native_rz_stopping)
    endif()
    # Compile-only copies challenge the actual physical candidate, not just
    # printed receipts. Production source and runtime inputs stay unchanged.
    file(READ ${helper_dir}/NativeRZSpatialStoppingEvent.cpp pristine)
    set(axis_old "if(i==0){a(i,j,k)=0.;}});}\n    Sync(reaction_b,g);")
    set(axis_new "if(i==0){a(i,j,k)=a(i,j,k);}});}\n    Sync(reaction_b,g);")
    set(wall_old "w(i,j,k)=conductor?dc(i,j,k)-di(i,j,k)-dd(i,j,k):0.;")
    set(wall_new "w(i,j,k)=conductor?-dc(i,j,k)+di(i,j,k)+dd(i,j,k):0.;")
    foreach(fault IN ITEMS axis wall)
        string(FIND "${pristine}" "${${fault}_old}" location)
        if(location LESS 0)
            message(FATAL_ERROR "RZ event fault fixture source expression missing")
        endif()
        string(REPLACE "${${fault}_old}" "${${fault}_new}" wrong "${pristine}")
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/WrongRZ_${fault}.cpp "${wrong}")
        add_executable(test_native_rz_wrong_${fault} main.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/WrongRZ_${fault}.cpp)
        target_link_libraries(test_native_rz_wrong_${fault} PRIVATE lib_rz)
        target_include_directories(test_native_rz_wrong_${fault} PRIVATE
            ${WarpX_SOURCE_DIR}/Source ${helper_dir})
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(test_native_rz_wrong_${fault})
        endif()
    endforeach()
else()
    message(FATAL_ERROR "Requires lib_rz or NATIVE_RZ_STOPPING_EXECUTABLE")
endif()
foreach(fault IN ITEMS AXIS WALL)
    if(NATIVE_RZ_WRONG_${fault}_EXECUTABLE)
        string(TOLOWER ${fault} lower)
        add_executable(test_native_rz_wrong_${lower} IMPORTED GLOBAL)
        set_target_properties(test_native_rz_wrong_${lower} PROPERTIES
            IMPORTED_LOCATION "${NATIVE_RZ_WRONG_${fault}_EXECUTABLE}")
    endif()
endforeach()
# The strict retry byte gate below is qualified on deterministic CPU only.
# CUDA compilation is covered; no GPU scatter/output byte identity is claimed.
if(WarpX_COMPUTE STREQUAL "CUDA")
    return()
endif()
set(common --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs_event_rz)
set(pmc --override "boundary.field_lo=none pmc" --override "boundary.field_hi=pec pmc"
    --override "boundary.particle_lo=none reflecting" --override "boundary.particle_hi=reflecting reflecting")
function(rz_case name)
    add_test(NAME native_rz_stopping_${name} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/run_and_analyze.py
        --executable $<TARGET_FILE:test_native_rz_stopping> ${common}
        --case ${name} ${ARGN})
    set_tests_properties(native_rz_stopping_${name} PROPERTIES TIMEOUT 240
        ENVIRONMENT "OMP_NUM_THREADS=1")
endfunction()
rz_case(periodic)
rz_case(pmc ${pmc})
rz_case(pmc_axis4 ${pmc} --override boundary.verboncoeur_axis_correction=0)
rz_case(zero --override fixture.zero=1)
foreach(negative IN ITEMS budget speed nonuniform external_current external_particle)
    rz_case(${negative} --negative ${negative})
endforeach()
if(MPIEXEC_EXECUTABLE)
    rz_case(mpi_seam --ranks 2 --mpi ${MPIEXEC_EXECUTABLE} --override "amr.n_cell=16 16")
    rz_case(mpi_empty ${pmc} --ranks 2 --mpi ${MPIEXEC_EXECUTABLE})
    set_tests_properties(native_rz_stopping_mpi_seam native_rz_stopping_mpi_empty
        PROPERTIES PROCESSORS 2)
endif()
foreach(fault IN ITEMS axis wall)
    if(TARGET test_native_rz_wrong_${fault})
        add_test(NAME native_rz_stopping_wrong_${fault} COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_and_analyze.py
            --executable $<TARGET_FILE:test_native_rz_wrong_${fault}> ${common}
            --case wrong_${fault} --negative wrong_${fault})
        set_tests_properties(native_rz_stopping_wrong_${fault} PROPERTIES TIMEOUT 240
            ENVIRONMENT "OMP_NUM_THREADS=1")
    endif()
endforeach()
