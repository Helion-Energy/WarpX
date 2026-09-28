# Full native advance: regression for eta capture at physical/box corners.
if(TARGET lib_rz)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    add_executable(test_thermal_source_gate_rz main.cpp)
    target_link_libraries(test_thermal_source_gate_rz PRIVATE lib_rz)
    target_include_directories(test_thermal_source_gate_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_thermal_source_gate_rz)
    endif()
    foreach(mode IN ITEMS decoupled_jfnk coupled_jfnk)
        foreach(ranks IN ITEMS 1 2)
            if(ranks EQUAL 1 OR WarpX_MPI)
                set(directory ${CMAKE_CURRENT_BINARY_DIR}/${mode}_${ranks})
                file(MAKE_DIRECTORY ${directory})
                add_test(NAME test_rz_thermal_source_corner_${mode}_mpi${ranks}
                    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_source_gate.py
                        $<TARGET_FILE:test_thermal_source_gate_rz>
                        ${CMAKE_CURRENT_SOURCE_DIR}/inputs_rz
                        --mode ${mode} --ranks ${ranks})
                set_tests_properties(test_rz_thermal_source_corner_${mode}_mpi${ranks}
                    PROPERTIES WORKING_DIRECTORY ${directory} TIMEOUT 180
                    ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS ${ranks})
            endif()
        endforeach()
    endforeach()
    set(energy_directory ${CMAKE_CURRENT_BINARY_DIR}/native_energy_inventory)
    file(MAKE_DIRECTORY ${energy_directory})
    set(energy_mpi_args)
    if(WarpX_MPI)
        list(APPEND energy_mpi_args --with-mpi)
    endif()
    set(energy_receipt_check exact)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        set(energy_receipt_check roundoff)
    endif()
    add_test(NAME test_rz_thermal_source_energy_inventory
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_native_energy.py
            $<TARGET_FILE:test_thermal_source_gate_rz>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_rz ${energy_mpi_args}
            --receipt-check=${energy_receipt_check})
    set(inertia_directory ${CMAKE_CURRENT_BINARY_DIR}/native_inertia_inventory)
    file(MAKE_DIRECTORY ${inertia_directory})
    add_test(NAME test_rz_thermal_source_inertia_inventory
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/analysis_native_inertia.py
            $<TARGET_FILE:test_thermal_source_gate_rz>
            ${CMAKE_CURRENT_SOURCE_DIR}/inputs_rz ${energy_mpi_args}
            --receipt-check=${energy_receipt_check})
    set_tests_properties(test_rz_thermal_source_inertia_inventory PROPERTIES
        WORKING_DIRECTORY ${inertia_directory} TIMEOUT 480
        ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2)
    set_tests_properties(test_rz_thermal_source_energy_inventory PROPERTIES
        WORKING_DIRECTORY ${energy_directory} TIMEOUT 240
        ENVIRONMENT "OMP_NUM_THREADS=1" PROCESSORS 2)
endif()
