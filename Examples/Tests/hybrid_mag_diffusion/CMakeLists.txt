# Add tests (alphabetical order) ##############################################
#

# Analytic constant-η magnetic diffusion (Fourier By mode) with
# hybrid_pic_model.implicit_mag_diffusion. Uses the C++ executable directly
# (no PICMI) so pywarpx is not required.
add_warpx_test(
    test_2d_hybrid_mag_diffusion  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic analysis is the oracle)
    OFF  # dependency
)

add_warpx_test(
    test_2d_hybrid_mag_diffusion_cn  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion_cn"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic analysis is the oracle)
    OFF  # dependency
)

add_warpx_test(
    test_2d_hybrid_mag_diffusion_variable_eta  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion_variable_eta"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic analysis is the oracle)
    OFF  # dependency
)

add_warpx_test(
    test_rz_hybrid_mag_diffusion_smoke  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_smoke"  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum (RZ vacuum smoke analysis is the oracle)
    OFF  # dependency
)

# Parser-driven eta on the matrix-free RZ path (uniform 129 Ohm m smoke).
# analysis_rz.py is a finite/damping stability smoke, not a Fourier analytic oracle.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_variable_eta  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_variable_eta"  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum (RZ vacuum smoke analysis is the oracle)
    OFF  # dependency
)

# Dirichlet B_t feed at the upper-r face via pec_insulator (FLASH CIRCUIT
# B_phi = f(I(t)) analogue). The feed is baked into the matrix-free operator
# (homogeneous) + RHS (affine offset). analysis_rz_pec_feed.py is a finite
# feed-held smoke, not an analytic oracle.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_pec_feed  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_pec_feed"  # inputs
    "analysis_rz_pec_feed.py"  # analysis
    OFF  # checksum (feed smoke analysis is the oracle)
    OFF  # dependency
)

# P1b: non-periodic axial (z) boundary via PEC endwalls (B_normal=0, tangential
# mirror). Lifts the former periodic-z-only restriction on the z-PEC wall path.
# analysis_rz_z_pec_wall.py is a finite/damping smoke, not an analytic oracle.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_z_pec_wall  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_z_pec_wall"  # inputs
    "analysis_rz_z_pec_wall.py"  # analysis
    OFF  # checksum (z-PEC smoke analysis is the oracle)
    OFF  # dependency
)

# P1b-zfeed: axial z_lo pec_insulator B_t feed with matching r_hi Dirichlet so
# the B_t∝r null mode fills the domain in z (proves axial curl-chain coupling).
# A pure PEC tube instead yields a physical O(R) skin under stiff BE — not a
# missing d^2/dz^2. analysis_rz_zfeed.py checks deep axial fill, not Fourier.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_zfeed  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_zfeed"  # inputs
    "analysis_rz_zfeed.py"  # analysis
    OFF  # checksum (z-feed deep-fill analysis is the oracle)
    OFF  # dependency
)

# Stair-case embedded boundary on the RZ matrix-free variable-eta path (theta=1):
# outer-wall solid for r > r_wall, fluid on axis. Covered Bt stays inert (0),
# fluid Bt persists (mild eta). analysis_rz_eb.py checks finite Bt, fluid Bt
# nonzero, damping, and covered-region Bt ~ 0 (global min). EB is a build-time
# option, so only register when WarpX_EB is ON (EB-off CI skips cleanly).
# Ported from public feature/hybrid-mag-diffusion-wip (EB mag-diff MVP).
if(WarpX_EB)
    add_warpx_test(
        test_rz_hybrid_mag_diffusion_variable_eta_eb  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_hybrid_mag_diffusion_variable_eta_eb"  # inputs
        "analysis_rz_eb.py"  # analysis
        OFF  # checksum (finite/covered-inert oracle in analysis_rz_eb.py)
        OFF  # dependency
    )
endif()
