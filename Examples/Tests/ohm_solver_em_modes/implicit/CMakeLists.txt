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

# Curl-curl MLMG preconditioner in its hybrid electron-inertia mode (the
# curl-curl operator form of Amano et al., J. Comput. Phys. 275, 197
# (2014)): runs the seeded-inertia dispersion variant at the
# DEFAULT Krylov budget with newton.require_convergence -- the plain
# inertia variant needs a boosted GMRES restart length to converge, so the
# preconditioner replacing that crutch is itself the solver assertion, and
# the two-branch dispersion fit is the physics assertion.
add_warpx_test(
    test_1d_ohm_solver_em_modes_implicit_inertia_ccpc_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 1 --bdir z --inertia-seeded --pc-curl-curl"  # inputs
    "analysis_inertia.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000100 --rtol 2e-4"  # checksum
    OFF  # dependency
)
label_warpx_test(test_1d_ohm_solver_em_modes_implicit_inertia_ccpc_picmi slow)

# 3D smoke of the same arm: exercises the 3D MLCurlCurl path and the
# 3D edge-staggered beta assembly (convergence is the assertion; no
# physics analysis is meaningful at this length).
add_warpx_test(
    test_3d_ohm_solver_em_modes_implicit_inertia_ccpc_picmi  # name
    3  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_implicit_picmi.py --test --dim 3 --bdir z --inertia-seeded --pc-curl-curl --steps 10"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

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

# Block-banded direct preconditioner (pc_block_banded) arm of the RZ deck
# (declared after its dependency, which must already be registered). Two
# ranks split the BoxArray, which routes the preconditioner through the
# replicated-global solve with the duplicate-DOF pass-through in the
# write-back. The solver-internal verification gates (extraction self-check
# against the frozen apply, LU and Apply round-trips, FD Jacobian-vector
# product comparison against the true residual) run at every rebuild via
# precond.bb_verify and abort on failure, and newton.require_convergence
# aborts on any unconverged step. The analysis then asserts physics
# agreement with the unpreconditioned parent test and iteration-count
# regression bounds from the Newton diagnostic files of both arms.
add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --pc-bb --steps 30"  # inputs
    "analysis_rz_bb.py"  # analysis
    OFF  # checksum
    test_rz_ohm_solver_em_modes_implicit_picmi  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_picmi slow)

# Whistler-stiff smoke of the same preconditioner arm: dt x8 puts the
# grid-scale whistler CFL where the unpreconditioned solve costs ~44-60
# GMRES iterations/step, so the analysis iteration ceiling is a sharp
# tripwire for preconditioner quality (at dt x1 the linear solve is cheap
# either way and the bound discriminates weakly). Ten steps, Newton line
# search on (rescues the large-dt basin independently of the PC),
# convergence required, verification gates active on the rebuild.
add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_stiff_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --pc-bb --dt-mult 8 --steps 10"  # inputs
    "analysis_rz_bb_stiff.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_stiff_picmi slow)

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
