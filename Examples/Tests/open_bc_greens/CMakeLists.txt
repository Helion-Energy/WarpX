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
    test_rz_open_bc_greens_btheta_isoz  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_btheta_isoz  # inputs
    "analysis_btheta.py diags/diag1000170"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# declared before test_rz_open_bc_greens_ring, which depends on it
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
    "analysis_ring.py diags/diag1000170 centered ../test_rz_open_bc_greens_ring_pec/diags/diag1000170"  # analysis
    OFF  # checksum
    test_rz_open_bc_greens_ring_pec  # dependency
)

add_warpx_test(
    test_rz_open_bc_greens_ring_nearwall  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_ring_nearwall  # inputs
    "analysis_ring.py diags/diag1000170 nearwall"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_open_bc_greens_ring_offcenter  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_ring_offcenter  # inputs
    "analysis_ring.py diags/diag1000170 offcenter"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_open_bc_greens_transient_sub20  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_transient_sub20  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_open_bc_greens_transient_sub80  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_transient_sub80  # inputs
    "analysis_transient.py diags/diag1000030 ../test_rz_open_bc_greens_transient_sub20/diags/diag1000030"  # analysis
    OFF  # checksum
    test_rz_open_bc_greens_transient_sub20  # dependency
)
