# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_hr_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 64"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_conformal_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal --grid-type collocated"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_hr_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_seam_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal --grid-type collocated --split-z"  # inputs
        "analysis_seam.py --path diags/diag1000200 --ref ../test_3d_ohm_solver_eb_diffusion_conformal_picmi/diags/diag1000200"  # analysis
        OFF  # checksum
        test_3d_ohm_solver_eb_diffusion_conformal_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_conformal_hr_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 64 --conformal --grid-type collocated"  # inputs
        "analysis.py --stair-lo ../test_3d_ohm_solver_eb_diffusion_picmi/diags/diag1000200 --stair-hi ../test_3d_ohm_solver_eb_diffusion_hr_picmi/diags/diag1000200 --conformal-lo ../test_3d_ohm_solver_eb_diffusion_conformal_picmi/diags/diag1000200 --conformal-hi diags/diag1000200"  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_conformal_picmi  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_pec_j_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --pec-j"  # inputs
        "analysis_pec_j.py --path diags/diag1000024"  # analysis
        "analysis_default_regression.py --path diags/diag1000024"  # checksum
        test_3d_ohm_solver_eb_diffusion_conformal_hr_picmi  # dependency
    )
endif()

# Yee (staggered) wall treatment for B: the opt-in direct level-set mirror fill
# (hybrid_pic_model.eb_b_straight_mirror) applied to the staggered Bfield_fp
# after each Faraday push -- smoke + checksum coverage of the staggered-grid
# counterpart of use_conformal_eb.
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_yee_mirror_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --eb-b-straight-mirror"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_pec_j_picmi  # dependency
    )
endif()

# Staggered (Yee) conformal wall treatment: the enlarged-cell (ECT) Faraday
# update on cut faces/edges (hybrid_pic_model.use_conformal_eb on the default
# staggered grid) -- smoke + checksum coverage of the staggered conformal path.
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_conformal_yee_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal"  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000200"  # checksum
        test_3d_ohm_solver_eb_diffusion_yee_mirror_picmi  # dependency
    )
endif()

# Cross-box seam equivalence of the staggered ECT face borrowing: the z-split
# decomposition forces box seams across the borrowing planes and must
# reproduce the y-split run to communication roundoff (guards the
# owner-unique cross-box face-extension reduction in the hybrid solver).
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_eb_diffusion_conformal_yee_seam_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_eb_diffusion_picmi.py --test -n 32 --conformal --split-z"  # inputs
        "analysis_seam.py --path diags/diag1000200 --ref ../test_3d_ohm_solver_eb_diffusion_conformal_yee_picmi/diags/diag1000200"  # analysis
        OFF  # checksum
        test_3d_ohm_solver_eb_diffusion_conformal_yee_picmi  # dependency
    )
endif()
