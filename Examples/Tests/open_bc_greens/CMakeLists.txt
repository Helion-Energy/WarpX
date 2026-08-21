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

# Theta-implicit arm: the Green's ghost fill runs inside every residual
# evaluation (design Sec. 6), so the implicit solve must relax to the same
# analytic free-space loop field as the explicit arm, with convergence
# required (a lagged fill stagnates Newton on the boundary rows).
add_warpx_test(
    test_rz_open_bc_greens_ring_implicit  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_ring_implicit  # inputs
    "analysis_ring.py diags/diag1000170 centered ../test_rz_open_bc_greens_ring_pec/diags/diag1000170"  # analysis
    OFF  # checksum
    test_rz_open_bc_greens_ring_pec  # dependency
)

# Theta-implicit arm of the temporal-coupling guard: with the ghost fill
# inside every residual evaluation, the theta-midpoint transient lands on
# the explicit subcycled trajectory at truncation level (measured bulk
# 4.3e-5 / wall 6.8e-6; gated at 5e-4, 1-2 orders below the lagged-fill
# failure class). A delivered-state-only fill fails the wall rows by
# orders of magnitude.
add_warpx_test(
    test_rz_open_bc_greens_transient_implicit  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_open_bc_greens_transient_implicit  # inputs
    "analysis_transient.py diags/diag1000030 ../test_rz_open_bc_greens_transient_sub80/diags/diag1000030 5.0e-4"  # analysis
    OFF  # checksum
    test_rz_open_bc_greens_transient_sub80  # dependency
)

# Coil-set drive against the open boundary: an out-of-domain filament pair
# ramps through the split-field external vector potential (parser A_time
# path, Hann engagement). Speciesless vacuum ride-through: the total field
# must equal s(t) * (discrete Yee curl of the sampled analytic loop A) to
# roundoff at every dump -- the analytic external advance must pass the
# Green's face untouched while the ghost fill acts on the (zero) response.
add_warpx_test(
    test_rz_open_bc_greens_coil_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_open_bc_greens_coil_picmi.py --test"  # inputs
    "analysis_coil.py vacuum diags/diag1000025 diags/diag1000050 diags/diag1000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Theta-implicit arm of the ride-through gate. Guards the response-frame
# plasma-current contract: J_plasma computed from the total field would
# deposit the coil field's O(h^2) discrete curl as a spurious near-face
# current whose Ohm response integrates secularly (measured 4e-7 and
# growing linearly before the UpdateWarpXFields response-frame fix;
# 9e-10 flat after).
add_warpx_test(
    test_rz_open_bc_greens_coil_implicit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_open_bc_greens_coil_picmi.py --test --implicit"  # inputs
    "analysis_coil.py vacuum diags/diag1000025 diags/diag1000050 diags/diag1000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Lenz arm: a super-floor plasma column (standoff from the open face)
# excludes the ramped coil flux (skin time >> ramp). Gates the response
# sign and the exclusion fraction of the eddy shield.
add_warpx_test(
    test_rz_open_bc_greens_coil_lenz_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_open_bc_greens_coil_picmi.py --test --plasma"  # inputs
    "analysis_coil.py lenz diags/diag1000025 diags/diag1000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Theta-implicit Lenz arm: gates the drive contract (Lenz sign and the
# mid-ramp exclusion window) only. The PLATEAU evolution of this deck is a
# KNOWN scheme-level divergence and is deliberately not gated: the
# theta-coupled ion/field system deterministically ejects marginal edge
# ions outward (a super-floor shell whose screening boundary marches
# toward the face and collapses the pile-up ring), where the explicit
# staggering sweeps them inward and quenches. Established by discriminator
# ladder: dt/4-converged (not truncation), PEC-vs-open bit-identical (not
# the boundary), heavy-ion (m x1000) arms agree to ~1% (the ion channel is
# the driver), density_min at the floor does not remove it (the marginal
# band n ~ n_floor suffices). Do not add an implicit-vs-explicit end-state
# cross-check here until that physics call is resolved.
add_warpx_test(
    test_rz_open_bc_greens_coil_lenz_implicit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_open_bc_greens_coil_picmi.py --test --plasma --implicit"  # inputs
    "analysis_coil.py lenz diags/diag1000025 diags/diag1000100"  # analysis
    OFF  # checksum
    OFF  # dependency
)
