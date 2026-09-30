# Native FAB reader and qualification are currently double precision only.
if(NOT WarpX_PRECISION STREQUAL "DOUBLE" OR
   NOT WarpX_PARTICLE_PRECISION STREQUAL "DOUBLE")
    return()
endif()

# Private native qualification of unit conversion on existing ghost support.
set(_temperature_ghost_dir "${CMAKE_CURRENT_LIST_DIR}")
foreach(_dim IN ITEMS rz 3d)
    if(TARGET lib_${_dim})
        set(_target "test_temperature_ghost_units_${_dim}")
        add_executable(${_target} "${_temperature_ghost_dir}/test_temperature_ghost_units.cpp")
        target_link_libraries(${_target} PRIVATE lib_${_dim})
        target_include_directories(${_target} PRIVATE ${WarpX_SOURCE_DIR}/Source)
        if(WarpX_COMPUTE STREQUAL "CUDA")
            setup_target_for_cuda_compilation(${_target})
        endif()
        if(_dim STREQUAL "rz")
            set(_cases
                rz_constant_p0
                rz_constant_p1
                rz_constant_p2
                rz_profile_tiled
                rz_profile_mpi
                rz_mixed_mpi
                rz_filter_off
                reject_nan_ghost_gate
                rz_particles_p0
                rz_particles_p1
                rz_particles_p2
                rz_particles_mixed
                rz_particles_onebox
                rz_particles_filter_off
            )
        endif()
        if(_dim STREQUAL "3d")
            set(_cases
                3d_constant_p0
                3d_periodic_p0
                3d_constant_p1
                3d_periodic_p1
                3d_constant_p2
                3d_periodic_p2
                3d_physical_mpi
                3d_mixed_mpi
                3d_filter_off
                3d_particles_physical
                3d_particles_periodic
                3d_particles_mixed
            )
        endif()
        foreach(_case IN LISTS _cases)
            set(_name "test_${_dim}_temperature_ghost_units_${_case}")
            add_test(NAME ${_name}
                COMMAND ${Python_EXECUTABLE} "${_temperature_ghost_dir}/analysis_ghost_units.py"
                    --executable $<TARGET_FILE:${_target}>
                    --case "${_temperature_ghost_dir}/ghost_units/${_case}.json"
                    --output "${CMAKE_CURRENT_BINARY_DIR}/${_name}")
            set_tests_properties(${_name} PROPERTIES TIMEOUT 90
                ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
        endforeach()
    endif()
endforeach()
