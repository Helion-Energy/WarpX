# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB AND "3" IN_LIST WarpX_DIMS)
    add_executable(test_eb_contact_3d test_eb_contact_3d.cpp)
    target_link_libraries(test_eb_contact_3d PRIVATE lib_3d)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_eb_contact_3d)
    endif()
    add_test(NAME test_3d_eb_contact COMMAND test_eb_contact_3d)
    set_tests_properties(test_3d_eb_contact PROPERTIES TIMEOUT 60 LABELS "particle_boundary")
    add_test(NAME test_3d_eb_contact_plane COMMAND test_eb_contact_3d wall_test.plane=1)
    add_test(NAME test_3d_eb_contact_reject COMMAND test_eb_contact_3d wall_test.reject_invalid=1)
    set_tests_properties(test_3d_eb_contact_plane test_3d_eb_contact_reject
        PROPERTIES TIMEOUT 60 LABELS "particle_boundary")
    add_executable(test_eb_contact_guard_3d test_eb_contact_guard_3d.cpp)
    target_link_libraries(test_eb_contact_guard_3d PRIVATE lib_3d)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_eb_contact_guard_3d)
    endif()
    add_test(NAME test_3d_eb_contact_host_guard COMMAND test_eb_contact_guard_3d
        ${CMAKE_CURRENT_SOURCE_DIR}/../implicit/inputs_test_3d_motional_work
        "boundary.field_lo=pec pec pec" "boundary.field_hi=pec pec pec"
        "boundary.particle_lo=reflecting reflecting reflecting"
        "boundary.particle_hi=reflecting reflecting reflecting"
        boundary.particle_eb=reflecting "warpx.eb_implicit_function=x-0.17"
        ions.density=0 amrex.throw_exception=1 amrex.signal_handling=0)
    set_tests_properties(test_3d_eb_contact_host_guard PROPERTIES
        TIMEOUT 60 LABELS "particle_boundary")
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_particle_boundary_interaction_thermal  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_particle_boundary_interaction_thermal  # inputs
        "analysis_thermal.py"  # analysis
        "analysis_default_regression.py --path diags/diag1004000"  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_particle_boundary_interaction_picmi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_particle_boundary_interaction_picmi.py  # inputs
        "analysis.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_particle_boundary_interaction_reflecting  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_particle_boundary_interaction_reflecting  # inputs
        "analysis.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()
