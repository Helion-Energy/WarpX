include_guard(GLOBAL)

# A build capability, not a claim of physical CUDA qualification. OFF changes
# no compiler flags, generated headers or target properties.
option(WarpX_NATIVE_PAIRED_PRECISE_CUDA
    "Build experimental paired CUDA arithmetic with an audited precise closure" OFF)

function(warpx_prepare_paired_precise_cuda)
    if(NOT WarpX_NATIVE_PAIRED_PRECISE_CUDA)
        return()
    endif()
    if(NOT WarpX_COMPUTE STREQUAL "CUDA" OR NOT WarpX_DIMS STREQUAL "RZ" OR
       NOT WarpX_PRECISION STREQUAL "DOUBLE" OR
       NOT WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
        message(FATAL_ERROR "Paired precise CUDA requires RZ double-precision CUDA")
    endif()
    if(NOT WarpX_amrex_src OR NOT IS_DIRECTORY "${WarpX_amrex_src}")
        message(FATAL_ERROR "Paired precise CUDA requires a local AMReX source build")
    endif()
    foreach(option WarpX_FASTMATH ABLASTR_FASTMATH AMReX_FASTMATH
                   AMReX_CUDA_FASTMATH WarpX_IPO AMReX_IPO AMReX_CUDA_LTO
                   CMAKE_INTERPROCEDURAL_OPTIMIZATION WarpX_CCACHE WarpX_UNITY_BUILD)
        if(${option})
            message(FATAL_ERROR "Paired precise CUDA does not support ${option}=ON")
        endif()
    endforeach()
    if(NOT DEFINED AMReX_CUDA_FASTMATH)
        message(FATAL_ERROR "Paired precise CUDA requires explicit AMReX_CUDA_FASTMATH=OFF")
    endif()
    foreach(name NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS NVCC_CCBIN CUDAFLAGS CXXFLAGS CFLAGS)
        if(DEFINED ENV{${name}} AND NOT "$ENV{${name}}" STREQUAL "")
            message(FATAL_ERROR "Paired precise CUDA requires empty ${name}; use audited CMake flags")
        endif()
    endforeach()
    foreach(lang C CXX CUDA)
        if(CMAKE_${lang}_COMPILER_LAUNCHER)
            message(FATAL_ERROR "Paired precise CUDA does not admit compiler launchers")
        endif()
    endforeach()
    if(NOT CMAKE_GENERATOR STREQUAL "Unix Makefiles" AND NOT CMAKE_GENERATOR STREQUAL "Ninja")
        message(FATAL_ERROR "Paired precise CUDA requires a single-configuration command database")
    endif()
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(FATAL_ERROR "Paired precise CUDA first capability requires Release")
    endif()
    # Installed before every dependency. Explicit fma is retained; implicit
    # multiply-add contraction and reassociation are disabled.
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-fast-math;-fno-associative-math;-fno-finite-math-only;-ffp-contract=off>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-fno-associative-math;-Xcompiler=-fno-finite-math-only;-Xcompiler=-ffp-contract=off>"
    )
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Audited command database" FORCE)
endfunction()

function(warpx_paired_collect_targets directory result)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        warpx_paired_collect_targets("${child}" child_targets)
        list(APPEND targets ${child_targets})
    endforeach()
    set(${result} "${targets}" PARENT_SCOPE)
endfunction()

function(warpx_finalize_paired_precise_cuda)
    if(NOT WarpX_NATIVE_PAIRED_PRECISE_CUDA)
        return()
    endif()
    file(REAL_PATH "${CMAKE_CUDA_HOST_COMPILER}" cuda_host_real)
    file(REAL_PATH "${CMAKE_CXX_COMPILER}" cxx_real)
    if(NOT cuda_host_real STREQUAL cxx_real)
        message(FATAL_ERROR "Paired precise CUDA requires the pinned C++ host compiler")
    endif()
    get_target_property(amrex_arch amrex_2d CUDA_ARCHITECTURES)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR
       NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL "12.9.86" OR
       NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
       NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "13.3.0" OR
       NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86" OR NOT amrex_arch STREQUAL "86")
        message(FATAL_ERROR "Paired precise CUDA first capability requires NVCC 12.9.86, GCC 13.3.0, SM86")
    endif()
    foreach(option WarpX_FASTMATH ABLASTR_FASTMATH AMReX_FASTMATH AMReX_CUDA_FASTMATH
                   WarpX_IPO AMReX_IPO AMReX_CUDA_LTO)
        if(${option})
            message(FATAL_ERROR "Paired precise CUDA dependency enabled ${option}")
        endif()
    endforeach()
    foreach(target amrex_2d ablastr_rz lib_rz)
        if(NOT TARGET ${target})
            message(FATAL_ERROR "Paired precise CUDA requires local target ${target}")
        endif()
        get_target_property(imported ${target} IMPORTED)
        if(imported)
            message(FATAL_ERROR "Paired precise CUDA rejects prebuilt target ${target}")
        endif()
    endforeach()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    file(MAKE_DIRECTORY "${WarpX_BINARY_DIR}/paired-precise")
    file(SHA256 "${CMAKE_CUDA_COMPILER}" cuda_hash)
    file(SHA256 "${CMAKE_CXX_COMPILER}" host_hash)
    set(context "${WarpX_BINARY_DIR}/paired-precise/context.json")
    file(WRITE "${context}" "{\n  \"source\": \"${WarpX_SOURCE_DIR}\",\n  \"binary\": \"${WarpX_BINARY_DIR}\",\n  \"amrex\": \"${WarpX_amrex_src}\",\n  \"cuda_compiler\": \"${CMAKE_CUDA_COMPILER}\",\n  \"cuda_sha256\": \"${cuda_hash}\",\n  \"host_compiler\": \"${CMAKE_CXX_COMPILER}\",\n  \"host_sha256\": \"${host_hash}\",\n  \"cuda_version\": \"${CMAKE_CUDA_COMPILER_VERSION}\",\n  \"host_version\": \"${CMAKE_CXX_COMPILER_VERSION}\",\n  \"architecture\": \"86\"\n}\n")
    set(header "${WarpX_BINARY_DIR}/Source/FieldSolver/ImplicitSolvers/NativePairedCudaBuild.H")
    set(receipt "${WarpX_BINARY_DIR}/paired-precise/COMMAND_AUDIT.json")
    warpx_paired_collect_targets("${WarpX_SOURCE_DIR}" targets)
    add_custom_target(warpx_paired_precise_command_audit
        COMMAND "${Python3_EXECUTABLE}" -B
            "${WarpX_SOURCE_DIR}/cmake/audit_paired_precise_cuda.py"
            --context "${context}"
            --commands "${WarpX_BINARY_DIR}/compile_commands.json"
            --header "${header}" --receipt "${receipt}"
        BYPRODUCTS "${header}" "${receipt}"
        COMMENT "Checking every command before paired precise CUDA compilation"
        VERBATIM)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(type STREQUAL "STATIC_LIBRARY" OR type STREQUAL "SHARED_LIBRARY" OR
           type STREQUAL "MODULE_LIBRARY" OR type STREQUAL "OBJECT_LIBRARY" OR
           type STREQUAL "EXECUTABLE")
            add_dependencies(${target} warpx_paired_precise_command_audit)
        endif()
    endforeach()
    target_compile_definitions(lib_rz PRIVATE WARPX_NATIVE_PAIRED_PRECISE_CUDA_BUILD=1)
endfunction()
