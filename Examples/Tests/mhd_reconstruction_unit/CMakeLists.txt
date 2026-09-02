# Standalone unit test of the TVD reconstruction kernels (header-only
# FieldSolver/ImplicitSolvers/ThetaImplicitMHD_K.H). Dimension
# independent: built once against the first configured dimension's
# ablastr target for the AMReX include paths.
#
# It carries the FACE-STATE admissibility gate of
# implicit_mhd.fluid_reconstruction -- adversarial four-cell stencils
# (1e4:1 density steps, counter-streaming shear, floor-riding energies)
# driven through every limiter mode and every ion closure, asserting
# that no reconstructed interface state falls below its positivity
# floor. That is the property the formation campaign's ion-energy
# failures make non-negotiable, and it cannot be observed from a
# plotfile: the face states never reach one.

list(GET WarpX_DIMS 0 _mhd_reconstruction_unit_dim)
warpx_set_suffix_dims(_mhd_reconstruction_unit_sd
    ${_mhd_reconstruction_unit_dim})

# On GPU builds the AMREX_GPU_HOST_DEVICE markers in ThetaImplicitMHD_K.H
# expand to CUDA/HIP qualifiers, so this host test source must be
# compiled by the device-capable compiler.
if(WarpX_COMPUTE STREQUAL CUDA)
    set_source_files_properties(mhd_reconstruction_unit.cpp
        PROPERTIES LANGUAGE CUDA)
elseif(WarpX_COMPUTE STREQUAL HIP)
    set_source_files_properties(mhd_reconstruction_unit.cpp
        PROPERTIES LANGUAGE HIP)
endif()

add_executable(mhd_reconstruction_unit mhd_reconstruction_unit.cpp)
target_link_libraries(mhd_reconstruction_unit
    PRIVATE ablastr_${_mhd_reconstruction_unit_sd})
target_include_directories(mhd_reconstruction_unit
    PRIVATE ${WarpX_SOURCE_DIR}/Source)

add_test(NAME mhd_reconstruction_unit COMMAND mhd_reconstruction_unit)
set_tests_properties(mhd_reconstruction_unit PROPERTIES LABELS "unit")
