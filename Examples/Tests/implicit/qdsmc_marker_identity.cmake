# Native raw-marker null and boundary ownership controls. The production
# marker representation and all physical source/conduction settings are intact.
if(TARGET lib_rz)
    set(_qdsmc_marker_fixture_dir "${CMAKE_CURRENT_LIST_DIR}")
    add_executable(test_qdsmc_marker_identity_rz
        "${_qdsmc_marker_fixture_dir}/test_qdsmc_marker_identity.cpp")
    target_link_libraries(test_qdsmc_marker_identity_rz PRIVATE lib_rz)
    target_include_directories(test_qdsmc_marker_identity_rz PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        setup_target_for_cuda_compilation(test_qdsmc_marker_identity_rz)
    endif()

    function(add_qdsmc_marker_case suffix)
        set(test_name "test_rz_qdsmc_marker_identity_${suffix}")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
        add_test(NAME ${test_name}
            COMMAND test_qdsmc_marker_identity_rz
                "${_qdsmc_marker_fixture_dir}/inputs_qdsmc_marker_identity"
                identity.require_identity=1 ${ARGN})
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${test_name}"
            TIMEOUT 90 ENVIRONMENT "OMP_NUM_THREADS=1")
    endfunction()

    add_qdsmc_marker_case(null identity.profile=0)
    add_qdsmc_marker_case(temperature identity.profile=1)
    add_qdsmc_marker_case(density identity.profile=2)
    add_qdsmc_marker_case(temperature_density identity.profile=3)
    add_qdsmc_marker_case(periodic identity.profile=4)
    add_qdsmc_marker_case(no_push identity.profile=1 identity.mode=1)
    add_qdsmc_marker_case(zero_push identity.profile=1 identity.mode=2)
    add_qdsmc_marker_case(shifted identity.profile=4
        "geometry.prob_lo=0.0 0.123" "geometry.prob_hi=0.04 0.203")
    add_qdsmc_marker_case(closed identity.profile=5
        "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
        "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
    foreach(sign IN ITEMS -1 1)
        add_qdsmc_marker_case(push_${sign}_periodic identity.profile=4
            identity.iterations=0 identity.ownership_push=${sign})
        add_qdsmc_marker_case(push_${sign}_closed identity.profile=5
            identity.iterations=0 identity.ownership_push=${sign}
            "boundary.field_lo=none pec" "boundary.field_hi=pec pec"
            "boundary.particle_lo=none reflecting" "boundary.particle_hi=reflecting reflecting")
    endforeach()
endif()
