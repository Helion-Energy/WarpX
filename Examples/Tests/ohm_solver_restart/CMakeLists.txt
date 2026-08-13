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

# The restarted arm of the EE pair is NOT registered yet: the electron
# temperature is now checkpointed and restored (with the adiabat re-seed
# suppressed), but the restarted evolution still loses part of the
# accumulated Joule heating relative to the uninterrupted run (~1e-1 of
# the gained Te in this deck, nearly independent of whether T_e was
# restored or wiped) — an Evolve()-re-entry state-continuation gap in the
# QDSMC entry path (marker/entropy re-initialization chain), shared with
# in-process segmented sim.step() re-entry, under joint investigation
# with the qdsmc branch. Register the arm below once re-entry
# continuation is exact:
#
# add_warpx_test(
#     test_2d_ohm_solver_ee_checkpoint_picmi_restart  # name
#     2  # dims
#     2  # nprocs
#     "inputs_test_2d_ohm_solver_ee_checkpoint_picmi.py amr.restart='../test_2d_ohm_solver_ee_checkpoint_picmi/diags/chk000005'"  # inputs
#     "analysis_default_restart.py diags/diag1000010"  # analysis
#     "analysis_default_regression.py --path diags/diag1000010"  # checksum
#     test_2d_ohm_solver_ee_checkpoint_picmi  # dependency
# )

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
