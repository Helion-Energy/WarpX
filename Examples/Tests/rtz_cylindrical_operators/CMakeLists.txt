# RTZ cylindrical finite-difference operator unit tests (C++, method of
# manufactured solutions). This is a standalone host executable that only *reads*
# the header-only operators from Source/ -- no test code is added to the source
# tree. Built only when RTZ is in WarpX_DIMS and WarpX_TEST_CXX is ON.

if(WarpX_TEST_CXX AND ("RTZ" IN_LIST WarpX_DIMS))
    add_executable(test_rtz_cylindrical_yee_operators
        test_cylindrical_yee_operators.cpp)

    # RTZ maps to AMReX 3D; the operators need only AMReX + the header-only
    # Utils/WarpXConst.H, called on the host.
    target_link_libraries(test_rtz_cylindrical_yee_operators PRIVATE AMReX::amrex_3d)
    target_include_directories(test_rtz_cylindrical_yee_operators
        PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_compile_definitions(test_rtz_cylindrical_yee_operators
        PRIVATE WARPX_DIM_RTZ WARPX_ZINDEX=2)
    target_compile_features(test_rtz_cylindrical_yee_operators PRIVATE cxx_std_17)

    add_test(
        NAME test_rtz_cylindrical_yee_operators
        COMMAND $<TARGET_FILE:test_rtz_cylindrical_yee_operators>
    )
    set_tests_properties(test_rtz_cylindrical_yee_operators PROPERTIES
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()

# RTZ azimuthal band-limit filter unit test (MultiFab-level: exercises the
# full-ring gather on a theta-split BoxArray, exact keep/kill per ring,
# idempotence, ring-average conservation, and the periodic ghost fill).
if(WarpX_TEST_CXX AND ("RTZ" IN_LIST WarpX_DIMS))
    add_executable(test_rtz_azimuthal_filter
        test_rtz_azimuthal_filter.cpp
        ${WarpX_SOURCE_DIR}/Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/AzimuthalFilter.cpp
        ${WarpX_SOURCE_DIR}/Source/ablastr/utils/TextMsg.cpp
        ${WarpX_SOURCE_DIR}/Source/ablastr/utils/text/StringUtils.cpp)
    if(WarpX_COMPUTE STREQUAL CUDA)
        set_source_files_properties(
            test_rtz_azimuthal_filter.cpp
            ${WarpX_SOURCE_DIR}/Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/AzimuthalFilter.cpp
            ${WarpX_SOURCE_DIR}/Source/ablastr/utils/TextMsg.cpp
            ${WarpX_SOURCE_DIR}/Source/ablastr/utils/text/StringUtils.cpp
            PROPERTIES LANGUAGE CUDA)
    endif()
    target_link_libraries(test_rtz_azimuthal_filter PRIVATE AMReX::amrex_3d)
    target_include_directories(test_rtz_azimuthal_filter
        PRIVATE ${WarpX_SOURCE_DIR}/Source)
    target_compile_definitions(test_rtz_azimuthal_filter
        PRIVATE WARPX_DIM_RTZ WARPX_ZINDEX=2)
    target_compile_features(test_rtz_azimuthal_filter PRIVATE cxx_std_17)

    add_test(
        NAME test_rtz_azimuthal_filter
        COMMAND $<TARGET_FILE:test_rtz_azimuthal_filter>
    )
    set_tests_properties(test_rtz_azimuthal_filter PROPERTIES
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
