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
    test_2d_ohm_solver_electron_energy_adiabat_implicit_segregated_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_adiabat_picmi.py --test --implicit --nlsolver picard --segregated"  # inputs
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
    test_2d_ohm_solver_electron_energy_conduction_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_conduction_picmi.py --test"  # inputs
    "analysis_conduction.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

# Same Gaussian-bump fixture on the grid-FD conduction operator. Unlike
# the kernel path (whose absolute rate competes with remap scales, hence
# the sanity window in the analysis), the FD operator admits a tight
# rate assert: the analysis detects the operator from warpx_used_inputs
# and pins the variance slope to 15%. Guards the kappa -> chi ->
# divergence-assembly calibration chain end to end (a heat-capacity
# class 3/2 slip passes the kernel test's sanity window but fails this).
add_warpx_test(
    test_2d_ohm_solver_electron_energy_conduction_fd_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_conduction_picmi.py --test --fd"  # inputs
    "analysis_conduction.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

# RZ radial through-axis mode of the grid-FD operator (curvilinear
# J Xi^ij form). The committed tight-rate analysis (10% slope window,
# r-weighted ledger) previously had no CI consumer.
add_warpx_test(
    test_rz_ohm_solver_electron_energy_conduction_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_conduction_picmi.py --test --rz"  # inputs
    "analysis_conduction_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

# Grid-FD conduction on circular field lines with the Sharma-Hammett
# cross-flux limiter (qdsmc_conduction_fd_limiter = sh): the anisotropic
# cross-term stress case, b-hat rotating against the grid everywhere.
# Gates the flux-level extrema bound (global Te max non-increasing, no
# background undershoot) plus a parallel-spreading sanity check.
add_warpx_test(
    test_2d_ohm_solver_electron_energy_conduction_circle_sh_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_conduction_picmi.py --test --circle --fd --fd-limiter sh"  # inputs
    "analysis_conduction_circle_fd.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_energy_conduction_front_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_conduction_picmi.py --test --front"  # inputs
    "analysis_conduction_front.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000300"  # checksum
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
    test_2d_ohm_solver_electron_energy_joule_implicit_segregated_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_joule_picmi.py --test --eta-scale 100 --implicit --segregated"  # inputs
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

# Calibrated heating-resistivity split: the Joule source runs at half the
# E-solve eta (hybrid_pic_model.joule_heating_resistivity), so the
# analysis' independent field-decay and Te-ramp eta fits must disagree by
# exactly 2x. Guards the split against silently heating with the E-solve
# (possibly vacuum-regularized) resistivity.
add_warpx_test(
    test_2d_ohm_solver_electron_energy_joule_split_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_joule_picmi.py --test --eta-scale 100 --heating-eta-scale 0.5"  # inputs
    "analysis_joule.py --eta-scale 100 --heating-eta-scale 0.5"  # analysis
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
    test_2d_ohm_solver_electron_energy_vacuum_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_picmi.py --case vacuum --test"  # inputs
    "analysis_vacuum.py"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

# PHYSICAL anisotropic electron viscosity (Braginskii 1965), the energy-BOOKED
# companion to the numerical hyper-resistivity. Manufactured solution: a
# tilted guide field with a small y-perturbation makes the electron fluid
# carry curl(B), so grad u_e is analytic and the Braginskii heating has a
# closed form. The active arm pins the absolute rate (coefficients, tensor
# algebra, energy booking and units end to end); the null arm is the
# anisotropy gate -- at zero tilt b_x vanishes everywhere, so the
# field-aligned strain b.W.b is identically zero and a correct PARALLEL
# viscosity must deposit no heat, while an isotropic one heats at full rate.
add_warpx_test(
    test_2d_ohm_solver_electron_viscosity_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_viscosity_picmi.py --test"  # inputs
    "analysis_viscosity.py"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_electron_viscosity_null_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_viscosity_picmi.py --test --tilt-deg 0"  # inputs
    "analysis_viscosity.py --null"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

# BOOKED hyper-resistive dissipation (hybrid_pic_model.hyper_resistivity_heating):
# a single current mode in a 3D periodic box decays under eta_H alone, and the
# magnetic energy Faraday removes must reappear as the hyp_bulk/hyp_band ledger
# columns (and as the time integral of the P_etaH_booked reduced-diag column).
# This Cartesian arm uses the component Laplacian form.
# The analysis reads the run's stdout, so the test captures it to run.log.
add_warpx_test(
    test_3d_ohm_solver_hyper_res_heating  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_hyper_res_heating amrex.omp_threads=1 > run.log"  # inputs
    "analysis_hyper_res_heating.py run.log"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# The RZ curl-curl extension is exercised below. The Cartesian
# curl-curl implementation is not part of this change.

# RZ hyper-resistive vacuum relaxation (curl-curl form, PEC wall): exercises
# the cylindrical hyper-resistive kernels with the EH_out mirror absent. Its
# value is the checksum -- the fixture is the bit-identity arm of the booking
# change (compare against a pre-change binary by hand; see the 2026-09-22
# evisc-drag report).
add_warpx_test(
    test_rz_ohm_solver_hyper_res_vacuum  # name
    RZ  # dims
    2  # nprocs
    inputs_test_rz_ohm_solver_hyper_res_vacuum  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Braginskii electron viscous DRAG in Ohm's law (qdsmc_viscosity_in_ohms_law),
# energy-balanced with the booked heating. Isotropic arm: the drag on a single
# divergence-free current mode reduces exactly to a hyper-resistivity with
# eta_eff = m_e nu/(e^2 n0); the mode decays at the discrete-symbol rate
# eta_eff kc^2 ky^2/mu0 (T1/T3) and the booked Q_nu, the drag work J.E_visc
# and the magnetic energy lost agree (T2). Second arm books the work itself.
# Third arm tilts B 45 deg with nu_perp = 1e-3 nu_par (anisotropic closure).
add_warpx_test(
    test_3d_ohm_solver_viscous_drag  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_viscous_drag > run.log"  # inputs
    "analysis_viscous_drag.py run.log"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_viscous_drag_work  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_viscous_drag hybrid_pic_model.qdsmc_viscosity_heating=work > run.log"  # inputs
    "analysis_viscous_drag.py run.log"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_viscous_drag_aniso  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_viscous_drag warpx.Bx_external_grid_function(x,y,z)=0.0707106781186548 warpx.By_external_grid_function(x,y,z)=0.005*sin(41.887902047863905*x) warpx.Bz_external_grid_function(x,y,z)=0.0707106781186548 hybrid_pic_model.qdsmc_nu_perp(n,Te,B)=0.2817 > run.log"  # inputs
    "analysis_viscous_drag.py run.log --aniso"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Density-pedestal arms: uniform pedestal 5 pct of n0 with the ion density
# dipping to 0.04 n0 (below the pedestal, above n_floor 0.02 n0). The ledger
# state is U_e = 1.5 (n_e + n_ped) kB Te, so a deposit converted with n_e alone
# would raise U by n_eff/n_e; with the state's
# heat capacity the bracketed visc_bulk equals the deposited visc_work_bulk to
# round-off -- the gate the orchestrator reads.
add_warpx_test(
    test_3d_ohm_solver_viscous_drag_pedestal  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_viscous_drag hybrid_pic_model.qdsmc_viscosity_heating=work hybrid_pic_model.density_pedestal=1 hybrid_pic_model.density_pedestal_profile(x,y,z)=5.0e17 hybrid_pic_model.n_floor=2.0e17 ions.profile=parse_density_function ions.density_function(x,y,z)=1.0e19*(0.52+0.48*cos(20.943951023931955*x)) > run.log"  # inputs
    "analysis_viscous_drag.py run.log --aniso"  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_ohm_solver_viscous_drag_pedestal_strain  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_viscous_drag hybrid_pic_model.qdsmc_viscosity_heating=strain hybrid_pic_model.density_pedestal=1 hybrid_pic_model.density_pedestal_profile(x,y,z)=5.0e17 hybrid_pic_model.n_floor=2.0e17 ions.profile=parse_density_function ions.density_function(x,y,z)=1.0e19*(0.52+0.48*cos(20.943951023931955*x)) > run.log"  # inputs
    "analysis_viscous_drag.py run.log --aniso"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Compressive / 2D arm: B_z(x,y) gives J_x and J_y with G_xx = -G_yy != 0, so
# the diagonal legs of div Pi and the 2D transposition are exercised (the
# arms above vary along x only). Strain and work must still agree to
# round-off -- this is the arm that exposes a non-adjoint diagonal stencil.
add_warpx_test(
    test_3d_ohm_solver_viscous_drag_compressive  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_ohm_solver_viscous_drag warpx.Bz_external_grid_function(x,y,z)=0.1+0.005*cos(41.887902047863905*x)*cos(83.77580409572781*y) warpx.const_dt=4.0e-9 > run.log"  # inputs
    "analysis_viscous_drag.py run.log --aniso"  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Insulating EB wall (boundary.eb_type = insulating) with the electron energy
# equation. Entropy-drift tolerances are set from measured values with margin:
# at rest the open-set drift is a saturating step-1 transient (-1.1e-4 at both
# 16 and 48 steps; the un-folded baseline leaks -7.3e-2 per 16 steps), under
# the compression drive it is the marker/ion transport-mismatch class
# (-1.5e-2 per 48 steps; un-folded baseline -1.9e-1).
add_warpx_test(
    test_2d_ohm_solver_insulating_eb_at_rest_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_insulating_eb_picmi.py --case at-rest --test"  # inputs
    "analysis_insulating_eb.py --tol-entropy 5.0e-4"  # analysis
    "analysis_default_regression.py --path diags/field_diags"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_ohm_solver_insulating_eb_compress_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_insulating_eb_picmi.py --case compress --test"  # inputs
    "analysis_insulating_eb.py --tol-entropy 5.0e-2"  # analysis
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

# Grid-FD conduction on the RZ curvilinear (J Xi^ij) form: a radial
# Gaussian THROUGH THE AXIS spreads as planar 2D diffusion (exact peak
# decay), with the r-weighted energy ledger, the discrete maximum
# principle, and z-uniformity (metric-leak guard) as gates. The SDE
# conduction forms remain Cartesian-only (parse-guarded in RZ).
add_warpx_test(
    test_rz_ohm_solver_electron_energy_conduction_fd_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_electron_energy_conduction_picmi.py --test --rz"  # inputs
    "analysis_conduction_rz.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000040"  # checksum
    OFF  # dependency
)

# Segregated-variant notes:
#   * The *_implicit_segregated_* tests exercise
#     implicit_evolve.qdsmc_segregated_solve: the QDSMC stage runs once per
#     outer iteration against inner-converged fields (frozen electron
#     pressure inside the residual) instead of inside every residual
#     evaluation. Converges to the same coupled state as the monolithic
#     form up to the solver tolerances, so the physics analyses apply
#     unchanged; the checksums are legitimately distinct.
#   * Both variants run the Picard inner solver, like their monolithic
#     implicit counterparts: the joule deck's Newton configuration diverges
#     at step 1 on this branch in BOTH monolithic and segregated forms
#     (pre-existing weakly-magnetized JFNK fragility, see the deck comment).
#
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
