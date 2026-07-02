# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_2d_eb_diffuse_reflection  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_eb_diffuse_reflection  # inputs
        "analysis_eb_diffuse_reflection.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_2d_eb_specular_reflection  # name
        2  # dims
        1  # nprocs
        inputs_test_2d_eb_specular_reflection  # inputs
        "analysis_eb_specular_reflection.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()
