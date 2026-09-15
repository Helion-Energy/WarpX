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
# (declared after its dependency, which must already be registered). In test
# mode, max_grid_size=128 plus refine_grid_layout=false keeps the 64x128
# domain in one BoxArray box. Two ranks therefore exercise auto-global
# resolution to the local path while leaving one rank with no local box. The
# solver-internal extraction check runs at every rebuild via precond.bb_verify
# and aborts on failure. LU/Apply round trips and the FD Jacobian-vector
# comparison are printed for evidence review; newton.require_convergence
# aborts on any unconverged step. The analysis asserts the one-box layout and
# then checks physics
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

# One-shot D1 common-direction operator partition.  The grid remains one
# BoxArray box in both arms, so the two-rank arm has one empty rank and must
# reproduce the one-rank A/B_g/B_s/C metrics and worst-row locations.
add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_d1_default_off_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --pc-bb --steps 1"  # inputs
    "analysis_rz_d1.py --assert-default-off"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_d1_default_off_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_d1_1rank_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --d1 --steps 1"  # inputs
    "analysis_rz_d1.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_d1_1rank_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_d1_2rank_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --d1 --steps 1"  # inputs
    "analysis_rz_d1.py --compare ../test_rz_ohm_solver_em_modes_implicit_bb_d1_1rank_picmi/diags/d1_operator_partition.txt"  # analysis
    OFF  # checksum
    test_rz_ohm_solver_em_modes_implicit_bb_d1_1rank_picmi  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_d1_2rank_picmi slow)

# Default-preserving P1 attribution pair. The mass-matrix Jacobian and
# block-banded preconditioner are identical in both arms; only the independent
# mass-matrix-PC request changes. BlockBandedPC does not consume those PC
# matrices, so the dependent on-arm requires byte-identical solver records,
# serialized parameters, and normalized used inputs before a production twin
# can be considered. Timing is deliberately outside this CI gate.
add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_mm_pc_off_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --pc-bb --mm-jacobian --mm-pc off --steps 1"  # inputs
    "analysis_rz_mm_pc.py --expect off"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_mm_pc_off_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_implicit_bb_mm_pc_on_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_implicit_picmi.py --test --pc-bb --mm-jacobian --mm-pc on --steps 1"  # inputs
    "analysis_rz_mm_pc.py --expect on --reference ../test_rz_ohm_solver_em_modes_implicit_bb_mm_pc_off_picmi"  # analysis
    OFF  # checksum
    test_rz_ohm_solver_em_modes_implicit_bb_mm_pc_off_picmi  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_implicit_bb_mm_pc_on_picmi slow)

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
