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
