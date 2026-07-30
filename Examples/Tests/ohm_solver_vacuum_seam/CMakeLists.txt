# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_cell  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_ohm_solver_vacuum_seam hybrid_pic_model.holmstrom_switch_mode=cell"  # inputs
    "analysis_seam.py ../test_2d_ohm_solver_vacuum_seam_edge ../test_2d_ohm_solver_vacuum_seam_node ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_vacuum_seam_node  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_edge  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_ohm_solver_vacuum_seam hybrid_pic_model.holmstrom_switch_mode=edge"  # inputs
    "analysis_seam.py ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_vacuum_seam_node  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_ohm_solver_vacuum_seam hybrid_pic_model.holmstrom_switch_mode=node"  # inputs
    "analysis_seam.py ."  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_2d_ohm_solver_vacuum_seam_edge  # dependency
)
