foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    set(split_test test_particle_splitting_${SD})
    add_executable(${split_test} test_particle_splitting.cpp)
    target_link_libraries(${split_test} PRIVATE lib_${SD})
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(${split_test})
    endif()
    foreach(split_type IN ITEMS 0 1)
        foreach(attributes IN ITEMS 0 1)
            set(test_name ${split_test}_type${split_type}_attrs${attributes})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
            add_test(NAME ${test_name}
                COMMAND ${split_test} test.split_type=${split_type}
                    test.add_attributes=${attributes})
            set_tests_properties(${test_name} PROPERTIES
                LABELS "particle_splitting" TIMEOUT 60
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        endforeach()
    endforeach()
endforeach()

if("RZ" IN_LIST WarpX_DIMS)
    add_executable(test_depleted_band_rz test_depleted_band.cpp)
    target_link_libraries(test_depleted_band_rz PRIVATE lib_rz)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_depleted_band_rz)
    endif()
    foreach(ppc IN ITEMS 5 10 20)
        set(test_name test_depleted_band_rz_ppc${ppc})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
        add_test(NAME ${test_name} COMMAND test_depleted_band_rz
            ions_D.hybrid_split_target_ppc=${ppc} ions_T.hybrid_split_target_ppc=${ppc})
        set_tests_properties(${test_name} PROPERTIES LABELS "particle_splitting" TIMEOUT 60
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    endforeach()
endif()
