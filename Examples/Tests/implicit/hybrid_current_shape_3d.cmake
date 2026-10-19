# Native continuity of charge and current across an EB shape-mask interface.
if(TARGET lib_3d AND WarpX_EB)
    add_executable(test_hybrid_current_shape_3d test_hybrid_current_shape_3d.cpp)
    target_link_libraries(test_hybrid_current_shape_3d PRIVATE lib_3d)
    target_include_directories(test_hybrid_current_shape_3d PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_hybrid_current_shape_3d)
    endif()
    foreach(axis RANGE 0 2)
        set(momentum "0 0 0")
        if(axis EQUAL 0)
            set(momentum "0.001 0 0")
        elseif(axis EQUAL 1)
            set(momentum "0 0.001 0")
        else()
            set(momentum "0 0 0.001")
        endif()
        foreach(shape RANGE 1 4)
            set(test_name test_3d_hybrid_current_shape_axis${axis}_order${shape})
            add_test(NAME ${test_name}
                COMMAND test_hybrid_current_shape_3d
                    ${CMAKE_CURRENT_SOURCE_DIR}/inputs_test_3d_hybrid_current_shape
                    audit.axis=${axis} algo.particle_shape=${shape}
                    "ions.single_particle_u=${momentum}")
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 120
                ENVIRONMENT "OMP_NUM_THREADS=1" LABELS "hybrid;eb;current;continuity;native")
        endforeach()
    endforeach()
endif()
