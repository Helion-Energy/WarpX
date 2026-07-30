# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_open_bc_greens_btheta  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_btheta  # inputs
    "analysis_btheta.py diags/diag1000170"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_open_bc_greens_ring_pec  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_ring_pec  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_open_bc_greens_ring  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_ring  # inputs
    "analysis_ring.py diags/diag1000170 ../test_rz_open_bc_greens_ring_pec/diags/diag1000170"  # analysis
    OFF  # checksum
    test_rz_open_bc_greens_ring_pec  # dependency
)
