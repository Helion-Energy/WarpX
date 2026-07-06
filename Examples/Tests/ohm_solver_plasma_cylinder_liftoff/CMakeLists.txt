# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_plasma_cylinder_liftoff_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_plasma_cylinder_liftoff_picmi.py --test"  # inputs
        "analysis.py --path diags/field_diag000010"  # analysis
        "analysis_default_regression.py --path diags/field_diag000010"  # checksum
        OFF  # dependency
    )
    label_warpx_test(test_3d_ohm_solver_plasma_cylinder_liftoff_picmi slow)
endif()

# Staggered (Yee) grid counterpart with the direct level-set mirror wall fill:
# same physics checks as the collocated test, in particular that the external
# drive field fills through the conducting wall (guards the staggered
# curl(A_ext) covered-cell masking regression) and that the early-time
# azimuthal spectrum stays at the particle-noise floor on both grid types.
# The particle stack is pinned to the legacy shape-1/direct/Galerkin combo so
# the checksums are independent of the deck's staggered defaults (which are
# the charge-conserving shape-2/esirkepov/mc-gather stack).
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_plasma_cylinder_liftoff_yee_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_plasma_cylinder_liftoff_picmi.py --test --grid-type staggered --eb-b-straight-mirror --particle-shape 1 --deposition direct --no-mc-gather"  # inputs
        "analysis.py --path diags/field_diag000010"  # analysis
        "analysis_default_regression.py --path diags/field_diag000010"  # checksum
        test_3d_ohm_solver_plasma_cylinder_liftoff_picmi  # dependency
    )
    label_warpx_test(test_3d_ohm_solver_plasma_cylinder_liftoff_yee_picmi slow)
endif()
