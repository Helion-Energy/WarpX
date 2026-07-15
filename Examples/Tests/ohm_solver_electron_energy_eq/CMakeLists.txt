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

# The RZ adiabat decks (inputs_test_rz_ohm_solver_electron_energy_adiabat_picmi.py)
# are not registered yet. Investigation status (reproduce with the decks +
# hybrid_pic_model.qdsmc_allow_unvalidated_rz = 1):
#   * The QDSMC entropy transport itself is CLEAN in RZ: the median interior
#     entropy K_e is conserved to +0.4% over the 60-step test and T_e follows
#     its adiabat; the apparent ~5.7% adiabat error is dominated by a BASE
#     RZ-hybrid ion-density redistribution (interior median n drops ~4%, wall
#     ring to 0.66 n0, axis to 0.90 n0) that is IDENTICAL with the energy
#     equation off (adiabatic closure) -- i.e., a pre-existing RZ hybrid
#     wall/axis artifact, not a QDSMC one.
#   * Remaining QDSMC-specific RZ items: the outermost ring shows a
#     recovery-vs-diagnosed density inconsistency (adiabat ratio ~1.3 in that
#     single ring; marker wall clamp + wall-node density under scrutiny), and
#     the theta-implicit RZ run deviates strongly (needs the same
#     energy-on/off bisect against the implicit closure run).
#   * do_temperature_deposition segfaults in RZ (blocks RZ Q_ei runs).
# Guards remain until the wall-ring and implicit items are resolved
# (HybridPICModel::ReadParameters refuses solve_electron_energy_equation in
# RZ without the override; PhysicalParticleContainer refuses
# do_temperature_deposition in radial geometries). Register the tests here
# once RZ validation lands.
