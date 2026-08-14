# Standalone unit tests of the continuum ring kernels (header-only
# ablastr/coils/RingKernel.H). Dimension independent: built once against the
# first configured dimension's ablastr target for the AMReX include paths.

list(GET WarpX_DIMS 0 _coils_unit_dim)
warpx_set_suffix_dims(_coils_unit_sd ${_coils_unit_dim})

add_executable(coils_unit_kernel coils_unit_kernel.cpp)
target_link_libraries(coils_unit_kernel PRIVATE ablastr_${_coils_unit_sd})

add_test(NAME coils_unit_kernel COMMAND coils_unit_kernel)
set_tests_properties(coils_unit_kernel PROPERTIES LABELS "unit")

add_executable(coils_unit_L_disc coils_unit_L_disc.cpp)
target_link_libraries(coils_unit_L_disc PRIVATE ablastr_${_coils_unit_sd})
target_include_directories(coils_unit_L_disc PRIVATE ${WarpX_SOURCE_DIR}/Source)

add_test(NAME coils_unit_L_disc COMMAND coils_unit_L_disc)
set_tests_properties(coils_unit_L_disc PROPERTIES LABELS "unit")
