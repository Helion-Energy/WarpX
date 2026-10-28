# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB AND "3" IN_LIST WarpX_DIMS)
    add_executable(test_eb_standoff_3d test_eb_standoff_3d.cpp)
    target_link_libraries(test_eb_standoff_3d PRIVATE lib_3d)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_eb_standoff_3d)
    endif()
    # Cubic particles plus one filter pass require four hybrid E/B geometry guards.
    # eb_distance_guard_cells is a minimum request, not an exact factory size.
    foreach(padding IN ITEMS 2 4 8 12)
        add_test(NAME test_3d_eb_standoff_geometry_${padding}
            COMMAND test_eb_standoff_3d
            ${CMAKE_CURRENT_SOURCE_DIR}/../implicit/inputs_test_3d_motional_work
            "amr.n_cell=160 160 16" "amr.max_grid_size=80 40 16" "amr.blocking_factor=8"
            "geometry.prob_lo=-0.22222222222222222 -0.22222222222222222 -0.109375"
            "geometry.prob_hi=0.22222222222222222 0.22222222222222222 0.109375"
            "boundary.field_lo=pec pec neumann" "boundary.field_hi=pec pec neumann"
            "boundary.particle_lo=absorbing absorbing absorbing"
            "boundary.particle_hi=absorbing absorbing absorbing"
            "warpx.eb_implicit_function=sqrt(x*x+y*y)-0.2"
            boundary.eb_distance_guard_cells=${padding}
            "standoff_test.expected_roof=(max(${padding},4)+1)*0.2/72"
            algo.particle_shape=3 algo.current_deposition=esirkepov
            algo.field_gathering=momentum-conserving
            warpx.use_filter=1 "warpx.filter_npass_each_dir=1 1 1"
            hybrid_pic_model.use_conformal_eb=1)
        set_tests_properties(test_3d_eb_standoff_geometry_${padding}
            PROPERTIES TIMEOUT 180 LABELS "particle_boundary")
    endforeach()
    foreach(mode IN ITEMS startup reject footprint subcycle split restart shape2)
        if(NOT "${mode}" STREQUAL "reject")
            set(padding 12)
            set(reject 0)
        else()
            set(padding 2)
            set(reject 1)
        endif()
        set(standoff_extra "")
        if(mode STREQUAL footprint)
            list(APPEND standoff_extra standoff_test.reject_footprint=1)
        elseif(mode STREQUAL restart)
            list(APPEND standoff_extra standoff_test.restart=1
                diagnostics.diags_names=check check.diag_type=Full check.format=checkpoint
                check.intervals=-1 check.file_prefix=standoff_chk check.file_min_digits=6)
        elseif(mode STREQUAL split)
            list(APPEND standoff_extra standoff_test.split=1 ions.hybrid_split_target_ppc=30
                ions.ux=0.00001 ions.uy=0.00002 ions.uz=0.00003)
        elseif(mode STREQUAL subcycle)
            list(APPEND standoff_extra standoff_test.advance=2
                ions.do_subcycled_push=1 ions.do_orbit_averaged_deposition=1
                ions.do_not_push=0 ions.ux=0.00333564095198152 warpx.const_dt=2.e-9 max_step=2)
        endif()
        if(mode STREQUAL shape2)
            set(shape 2)
        else()
            set(shape 3)
        endif()
        add_test(NAME test_3d_eb_standoff_${mode}
            COMMAND test_eb_standoff_3d
            ${CMAKE_CURRENT_SOURCE_DIR}/../implicit/inputs_test_3d_motional_work
            "amr.n_cell=160 160 16" "amr.max_grid_size=80 40 16" "amr.blocking_factor=8"
            "geometry.prob_lo=-0.22222222222222222 -0.22222222222222222 -0.109375"
            "geometry.prob_hi=0.22222222222222222 0.22222222222222222 0.109375"
            "boundary.field_lo=pec pec neumann" "boundary.field_hi=pec pec neumann"
            "boundary.particle_lo=absorbing absorbing absorbing"
            "boundary.particle_hi=absorbing absorbing absorbing"
            "warpx.eb_implicit_function=sqrt(x*x+y*y)-0.2"
            boundary.eb_distance_guard_cells=${padding}
            boundary.eb_type=insulating boundary.eb_standoff_cells=1.462857142857143
            "standoff_test.expected_roof=(${padding}+1)*0.2/72"
            standoff_test.reject=${reject} amrex.throw_exception=1 amrex.signal_handling=0
            boundary.eb_footprint_cylinder_radius=0.195 ${standoff_extra}
            algo.particle_shape=${shape} algo.current_deposition=esirkepov
            algo.field_gathering=momentum-conserving
            warpx.use_filter=1 "warpx.filter_npass_each_dir=1 1 1"
            hybrid_pic_model.use_conformal_eb=1)
        set_tests_properties(test_3d_eb_standoff_${mode}
            PROPERTIES TIMEOUT 180 LABELS "particle_boundary")
    endforeach()
    add_test(NAME test_3d_eb_standoff_coupled
        COMMAND $<TARGET_FILE:app_3d>
        ${CMAKE_CURRENT_SOURCE_DIR}/../implicit/inputs_test_3d_coupled_port
        max_step=1 warpx.const_dt=1.e-12
        "amr.n_cell=80 80 16" "amr.max_grid_size=40" "amr.blocking_factor=8"
        "geometry.prob_lo=-0.11111111111111111 -0.11111111111111111 -0.109375"
        "geometry.prob_hi=0.11111111111111111 0.11111111111111111 0.109375"
        "warpx.eb_implicit_function=sqrt(x*x+y*y)-0.09"
        boundary.eb_type=insulating boundary.particle_eb=absorbing
        boundary.eb_standoff_cells=1.462857142857143 boundary.eb_distance_guard_cells=12
        boundary.eb_footprint_cylinder_radius=0.085
        hybrid_pic_model.qdsmc_conduction_rkl_max_stages=5000
        hybrid_pic_model.qdsmc_conduction_fd_max_subcycles=250000)
    set_tests_properties(test_3d_eb_standoff_coupled
        PROPERTIES TIMEOUT 300 LABELS "particle_boundary")
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

# A direct native lineage test: permute storage without changing physical parents.
if("3" IN_LIST WarpX_DIMS)
    add_executable(test_split_lineage_3d test_split_lineage_3d.cpp)
    target_link_libraries(test_split_lineage_3d PRIVATE lib_3d)
    target_include_directories(test_split_lineage_3d PRIVATE ${WarpX_SOURCE_DIR}/Source)
    if(WarpX_COMPUTE STREQUAL CUDA)
        setup_target_for_cuda_compilation(test_split_lineage_3d)
    endif()
    foreach(grid IN ITEMS 8 16)
        add_test(NAME test_3d_split_lineage_grid${grid}
            COMMAND $<TARGET_FILE:test_split_lineage_3d>
                ${CMAKE_CURRENT_SOURCE_DIR}/../implicit/inputs_electron_inertia_restart_3d
                amr.max_grid_size=${grid} ions.hybrid_split_target_ppc=30
                diagnostics.enable=0)
        set_tests_properties(test_3d_split_lineage_grid${grid}
            PROPERTIES TIMEOUT 120 LABELS "particle_boundary;restart")
    endforeach()
endif()
