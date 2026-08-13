# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ohm_solver_em_modes_darwin_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z --darwin"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000050 --rtol 2e-4"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000050 --rtol 2e-4"  # checksum
    OFF  # dependency
)

# Implicit-scheme python-callback contract: the step-family callbacks fire
# exactly once per step (never inside residual evaluations), and afterEpush
# observes hybrid_current_fp_plasma at the delivered end-of-step state
# curl(B^{n+1})/mu0 -- the read contract of a segregated circuit coupler.
# Assertions run inside the input script, so no analysis/checksum is needed.
add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_callbacks_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z --callback-gating"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Solver-knob smoke: Eisenstat-Walker adaptive forcing (newton.adaptive_forcing)
# on a 10-step slice of the seeded-inertia configuration. Convergence is the
# assertion (newton.require_convergence aborts on failure); no physics
# analysis is meaningful at this length.
add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_forcing_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z --inertia-seeded --steps 10 --adaptive-forcing"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_inertia_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z --inertia-seeded"  # inputs
    "analysis_inertia.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000100 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_1d_ohm_solver_em_modes_implicit_inertia_picmi slow)

add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_inertia_mm_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z --inertia-seeded --mass-matrices"  # inputs
    "analysis_inertia.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000100 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_1d_ohm_solver_em_modes_implicit_inertia_mm_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test"  # inputs
    "analysis_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000050 --rtol 1e-6"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_picmi slow)

# 3D Cartesian coverage of the theta-implicit hybrid solver and its Darwin
# variant (CI-sized column: Nz = 128, NPPC = 32 in test mode; the physics
# dispersion analyses live in the 1D tests).
add_warpx_test(
    test_3d_ohm_solver_em_modes_implicit_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 3 --bdir z"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/field_diag000050 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_3d_ohm_solver_em_modes_implicit_picmi slow)

add_warpx_test(
    test_3d_ohm_solver_em_modes_darwin_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 3 --bdir z --darwin"  # inputs
    OFF  # analysis
    "analysis_default_regression.py --path diags/field_diag000050 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_3d_ohm_solver_em_modes_darwin_picmi slow)
