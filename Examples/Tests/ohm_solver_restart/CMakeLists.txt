# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_checkpoint_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_checkpoint_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_checkpoint_picmi_restart  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_checkpoint_picmi.py amr.restart='../test_2d_ohm_solver_checkpoint_picmi/diags/chk000005'"  # inputs
    "analysis_default_restart.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_checkpoint_picmi  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_ee_checkpoint_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_ee_checkpoint_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

# Restarted arm of the EE pair. The historical ~1e-1 gained-Te restart
# deficit ("nearly independent of whether T_e was restored or wiped")
# was the restored T_e being WIPED back to the uniform seed constant by
# HybridPICModel::InitData's unconditional fill, which runs after
# InitFromCheckpoint in WarpX::InitData — the restore was dead code
# until the fill was latch-guarded. With the guard, this pair is
# bit-exact (measured on this deck with direct AND Esirkepov
# deposition, and on the RZ FD-conduction deck explicit and
# theta-implicit).
add_warpx_test(
    test_2d_ohm_solver_ee_checkpoint_picmi_restart  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_ee_checkpoint_picmi.py amr.restart='../test_2d_ohm_solver_ee_checkpoint_picmi/diags/chk000005'"  # inputs
    "analysis_default_restart.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_ee_checkpoint_picmi  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_external_field_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_ohm_solver_external_field_segmented_picmi.py  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_external_field_picmi_segmented  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_external_field_segmented_picmi.py --segments 3,3,4"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)
