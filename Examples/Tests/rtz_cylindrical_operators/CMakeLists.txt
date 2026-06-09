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
