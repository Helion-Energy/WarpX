# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_1d_ohm_solver_em_modes_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_picmi.py --test --dim 1 --bdir z"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000250"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_ohm_solver_em_modes_rkf45_picmi  # name
    1  # dims
    2  # nprocs
    "inputs_test_1d_ohm_solver_em_modes_picmi.py --test --dim 1 --bdir z --use_rkf45"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/field_diag000250"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_je_whistler_gate_eform  # name
    1  # dims
    2  # nprocs
    "je_whistler_gate.py --esolve e_form"  # inputs
    OFF  # analysis (consumed by the je arms' branch discrimination)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_1d_je_whistler_gate  # name
    1  # dims
    2  # nprocs
    "je_whistler_gate.py --esolve je_form"  # inputs
    "je_whistler_analysis.py ../test_1d_je_whistler_gate_eform ."  # analysis
    OFF  # checksum
    test_1d_je_whistler_gate_eform  # dependency
)

add_warpx_test(
    test_1d_je_whistler_gate_be  # name (the legacy backward-Euler advance)
    1  # dims
    2  # nprocs
    "je_whistler_gate.py --esolve je_form --je-advance be"  # inputs
    "je_whistler_analysis.py ../test_1d_je_whistler_gate_eform ."  # analysis
    OFF  # checksum
    test_1d_je_whistler_gate_eform  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_em_modes_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_picmi.py --test"  # inputs
    "analysis_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100 --rtol 1e-6"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_je_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_picmi.py --test --esolve je_form"  # inputs
    "analysis_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100 --rtol 1e-6"  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_je_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_annulus_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_annulus_picmi.py --test"  # inputs
    "analysis_annulus.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_annulus_picmi slow)

add_warpx_test(
    test_rz_ohm_solver_em_modes_annulus_je_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_em_modes_annulus_picmi.py --test --esolve je_form"  # inputs
    "analysis_annulus.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
label_warpx_test(test_rz_ohm_solver_em_modes_annulus_je_picmi slow)

add_subdirectory(implicit)
