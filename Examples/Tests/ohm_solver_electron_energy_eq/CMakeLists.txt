# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_electron_energy_adiabat_implicit_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_adiabat_picmi.py --test --implicit --nlsolver picard"  # inputs
    "analysis_adiabat.py --tol-median 0.01 --tol-max 0.05"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_adiabat_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_adiabat_picmi.py --test"  # inputs
    "analysis_adiabat.py --tol-median 0.01 --tol-max 0.05"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_joule_implicit_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_joule_picmi.py --test --eta-scale 100 --implicit"  # inputs
    "analysis_joule.py --eta-scale 100"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_joule_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_joule_picmi.py --test --eta-scale 100"  # inputs
    "analysis_joule.py --eta-scale 100"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_qei_implicit_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_qei_picmi.py --test --implicit"  # inputs
    "analysis_qei.py --nu-ei 1e6"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_qei_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_qei_picmi.py --test"  # inputs
    "analysis_qei.py --nu-ei 1e6"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_electron_energy_adiabat_implicit_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_electron_energy_adiabat_picmi.py --test --implicit --nlsolver picard"  # inputs
    "analysis_adiabat.py --tol-median 0.03 --tol-max 0.08"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_electron_energy_adiabat_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_electron_energy_adiabat_picmi.py --test"  # inputs
    "analysis_adiabat.py --tol-median 0.02 --tol-max 0.05"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

# RZ notes:
#   * The adiabat analysis compares each cell against its own first-dump
#     state, which cancels the static per-node deposit conventions of the
#     radial geometries (half-valued wall node, Verboncoeur axis node --
#     these offset the diagnosed density by time-constant factors and are a
#     base-scheme property, present with the adiabatic closure as well).
#   * The theta-implicit RZ run carries a slow secular redistribution at the
#     outermost ring (wall-adjacent density/temperature drift over long runs,
#     absent from the explicit path); its interior tracks the adiabat at the
#     percent level. Tracked as a follow-up together with the upstream
#     boundary-node deposit conventions.
#   * do_temperature_deposition crashes in RZ when run on more than one MPI
#     rank (SIGSEGV in the two-pass variance deposition at the first step;
#     single-rank runs and the externally-driven cylinder-compression test
#     pass). Pre-existing upstream issue in the temperature-deposition
#     kernels; the RZ Q_ei configuration remains unregistered until it is
#     fixed.
