#!/usr/bin/env python3
#
# --- Theta-pinch liftoff of an annular plasma column in RTZ (3D cylindrical,
# --- hybrid-PIC solver). RTZ port of the 3D Cartesian liftoff deck.
# ---
# --- A thin slab (periodic in z) is cut around the formation-section midplane:
# --- an annular deuterium column rests inside a conducting cylindrical wall,
# --- threaded by a -0.01 T bias field. A uniform external axial field, applied
# --- through the external vector potential machinery with a Hermite-smoothstep
# --- time ramp, reverses the field up to +0.5 T: the annulus lifts off the wall
# --- region and implodes (theta-pinch formation with a reversed bias core).
# ---
# --- In RTZ the conducting wall is the grid-aligned r = R_WALL domain boundary
# --- (dirichlet fields + absorbing particles), so the baseline needs no
# --- embedded boundary. Unlike the Cartesian grid (which breaks rotational
# --- symmetry and numerically seeds an m=4 mode), the cylindrical grid is
# --- rotationally symmetric: a theta-uniform implosion should stay uniform to
# --- particle noise. Controlled azimuthal structure is seeded explicitly:
# ---   --perturb-m M --perturb-eps EPS   modal perturbation of the external
# ---                                     drive field, Bz ~ f(t)*(1+EPS*cos(M*theta))
# ---   --perturb-density-eps EPS        matching modal seed on the loaded
# ---                                     annulus density
# ---
# --- Not ported from the 3D deck (feature lives on the conformal-EB branch):
# --- conformal-EB wall knobs, holmstrom blend shaping, esirkepov deposition
# --- (RTZ hybrid uses direct deposition), collocated grid (RTZ is Yee-only).

import argparse
import sys

import numpy as np

from pywarpx import picmi

constants = picmi.constants

# ----------------------------------------------------------------------------
# Physics parameters: slab at the z = 5 m midplane of the formation section
# ----------------------------------------------------------------------------
M_AMU = 2  # deuterium
N_I = 1.5e20  # reference ion density (m^-3)
N_FLOOR_FRAC = 0.05
T_I0 = 5.0  # ion temperature (eV)
T_E0 = 5.0  # electron temperature (eV)

# Annular column (radii in m) and low-density interior fill
R_INNER = 0.65
R_OUTER = 0.75
R_PART = 0.385  # the annulus carries the inventory of a full column of this radius
ANNULUS_SMOOTH_CELLS = 2.0
R_WALL = 0.8  # conducting wall radius = r domain boundary

# External field ramp: Hermite smoothstep from the bias to the reversed field
BZ_BIAS = -0.01  # T
BZ_REV = 0.5  # T
TAU_RAMP = 10.0e-6  # s

# Resistivity (density-scaled power law) and hyper-resistivity
ETA_PLASMA = 1.0e-6  # Ohm m, bulk plasma
ETA_VAC_FRAC = 5.0e-2  # vacuum resistivity as a fraction of the CFL limit
ETA_POWER = 3.0
N_TRANSITION_FRAC = 0.4

# Time stepping
F_T_CI = 0.01  # dt as a fraction of the (reversed-field) ion cyclotron period
SUBSTEPS = 256  # initial RKF45 substep count


def hermite_ramp_expression(b0, b1, tau):
    """Time scaling of the unit-Bz external vector potential: Hermite
    smoothstep h(s) = s^2*(3-2*s) from b0 (bias) to b1 (reversed field)."""
    s = f"min(max(t/{tau:.9e},0),1)"
    return f"({b0:.9e} + ({b1 - b0:.9e})*({s})*({s})*(3-2*({s})))"


def power_law_resistivity(
    eta_plasma, eta_vac, power, n_floor_frac, n_transition_frac, n0
):
    """Density-scaled resistivity interpolating from eta_plasma in the bulk
    (n >= n_transition) to eta_vac at the density floor, as a power law of
    the density (see the 3D liftoff deck this case is derived from)."""
    a = (n_floor_frac / n_transition_frac) ** power
    res_str = (
        f"eta_plasma + (eta_vac - eta_plasma)"
        f"*max(0.0, (rho_f/max(rho,rho_f))**({power:.6g}) - ({a:.12g}))"
        f"/({1.0 - a:.12g})"
    )
    return {
        "plasma_resistivity": res_str,
        "eta_plasma": eta_plasma,
        "eta_vac": eta_vac,
        "rho_f": constants.q_e * n0 * n_floor_frac,
    }


def setup_simulation(
    nr,
    ntheta,
    nz,
    nppc,
    max_steps,
    diag_period,
    verbose,
    holmstrom=False,
    holmstrom_switch_mode="edge",
    holmstrom_blend_pow=0.0,
    holmstrom_blend_width=2.0,
    eta_power=ETA_POWER,
    eta_ntrans=N_TRANSITION_FRAC,
    eta_vac_frac=ETA_VAC_FRAC,
    f_t_ci=F_T_CI,
    r_outer=R_OUTER,
    n_floor_frac=N_FLOOR_FRAC,
    substeps=SUBSTEPS,
    substep_rtol=1.0e-3,
    te=T_E0,
    bz_rev=BZ_REV,
    bz_bias=BZ_BIAS,
    annulus_smooth_cells=ANNULUS_SMOOTH_CELLS,
    equilibrium_b=False,
    vacuum=False,
    fill_frac=None,
    collisions=True,
    particle_shape=1,
    perturb_m=0,
    perturb_eps=0.0,
    perturb_phase=0.0,
    perturb_density_eps=0.0,
    eb_wall=False,
    openpmd=False,
):
    """Create the PICMI simulation object.

    RTZ parser conventions (they differ!):
      * grid-field parsers (external A, initial B seed, EB implicit function):
        x = r, y = theta, z = z (the grid axes);
      * particle-injection parsers (density/momentum): Cartesian x, y, z
        (AddPlasma converts positions before evaluating), so radius is
        sqrt(x^2+y^2) as in the 3D Cartesian deck.
    """
    m_i = M_AMU * constants.m_p
    n_floor = n_floor_frac * N_I
    # Interior fill density above the holmstrom window (see the 3D deck).
    n_fill = fill_frac * N_I if fill_frac is not None else 2.0 * n_floor
    vth = np.sqrt(constants.q_e * T_I0 / m_i)

    # cell sizes; z slab of nz cells at dz = dr, centered on the z = 5 m
    # midplane. With --eb-wall the domain extends past the conducting wall
    # (embedded boundary at R_WALL, like the 3D Cartesian deck); otherwise the
    # wall IS the r domain boundary.
    r_domain = 1.0 if eb_wall else R_WALL
    dr = r_domain / nr
    annulus_w = annulus_smooth_cells * dr
    lz = nz * dr
    zmin, zmax = 5.0 - lz / 2.0, 5.0 + lz / 2.0

    # time step from the reversed-field ion cyclotron period
    w_ci = constants.q_e * abs(bz_rev) / m_i
    t_ci = 2.0 * np.pi / w_ci
    dt = f_t_ci * t_ci

    # ion skin depth and Alfven speed at the reversed field, for the
    # Chacon-style hyper-resistivity floor; CFL-limited vacuum resistivity.
    # The grid-scale length uses the mid-annulus azimuthal arc as the
    # effective "dy" (the cylindrical cell aspect varies with r).
    w_pi = np.sqrt(constants.q_e**2 * N_I / (constants.ep0 * m_i))
    l_i = constants.c / w_pi
    vA = abs(bz_rev) / np.sqrt(constants.mu0 * N_I * m_i)
    d_arc = 0.5 * (R_INNER + R_WALL) * (2.0 * np.pi / ntheta)
    dL2 = 1.0 / (1.0 / dr**2 + 1.0 / d_arc**2 + 1.0 / dr**2)
    eta_max = constants.mu0 * dL2 / (2.0 * dt)
    eta_hyper = constants.mu0 * 0.2 * l_i * vA * dL2

    grid = picmi.CylindricalGrid3D(
        number_of_cells=[nr, ntheta, nz],
        lower_bound=[0.0, -np.pi, zmin],
        upper_bound=[r_domain, np.pi, zmax],
        lower_boundary_conditions=["none", "periodic", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
        lower_boundary_conditions_particles=["none", "periodic", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        warpx_max_grid_size=2048,
        warpx_max_grid_size_x=max(nr // 2, 8),
        warpx_blocking_factor=min(8, nr, ntheta, nz),
    )

    sim = picmi.Simulation(
        time_step_size=dt,
        max_steps=max_steps,
        particle_shape=particle_shape,
        verbose=verbose,
    )
    sim.current_deposition_algo = "direct"  # RTZ hybrid: direct deposition only
    sim.grid_type = "staggered"  # RTZ is Yee-only

    # External reversal field: uniform Bz through A_theta = 0.5*r (x = r), with
    # the Hermite ramp from the bias to the reversed field in f(t).
    # Optional modal drive perturbation: A_theta *= (1 + eps*cos(m*(theta-phase)))
    # gives Bz = f(t)*(1 + eps*cos(m*theta_shifted)) through the cylindrical curl
    # (Bz = (1/r) d(r A_theta)/dr; Br = -dz A_theta = 0, so the perturbed drive
    # stays divergence-free).
    if perturb_m > 0 and perturb_eps != 0.0:
        a_theta = (
            f"0.5*x*(1.0+{perturb_eps:.9e}*cos({perturb_m:d}*(y-{perturb_phase:.9e})))"
        )
    else:
        a_theta = "0.5*x"
    A_ext = {
        "uniform_reversal": {
            "Ax_external_function": "0",
            "Ay_external_function": a_theta,
            "Az_external_function": "0",
            "A_time_external_function": hermite_ramp_expression(
                bz_bias, bz_rev, TAU_RAMP
            ),
        },
    }

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=te,
        n0=N_I,
        n_floor=n_floor,
        plasma_hyper_resistivity=eta_hyper,
        substeps=substeps,
        holmstrom_vacuum_region=True if holmstrom else None,
        # cell-sampled vacuum switch + above-floor blend (the conformal-EB
        # branch's validated staggered config) regularize the plasma/vacuum
        # seam: one decision per cell instead of per-component half-cell
        # offsets, and the Hall physics fades in over the low-density band
        holmstrom_switch_mode=holmstrom_switch_mode
        if (holmstrom and holmstrom_switch_mode != "edge") else None,
        holmstrom_blend_pow=holmstrom_blend_pow
        if (holmstrom and holmstrom_blend_pow > 0) else None,
        holmstrom_blend_width=holmstrom_blend_width
        if (holmstrom and holmstrom_blend_pow > 0) else None,
        use_rkf45=True,
        substep_rtol=substep_rtol,
        substep_atol=1.0e-8,
        max_substep_attempts=1000,
        # The uniform (and modally perturbed) drive A is divergence-free where
        # it matters (B = curl A regardless); the projection div cleaner is not
        # implemented for RTZ and would abort.
        do_external_diva_cleaning=False,
        A_external=A_ext,
        **power_law_resistivity(
            ETA_PLASMA,
            eta_vac_frac * eta_max,
            eta_power,
            n_floor_frac,
            eta_ntrans,
            N_I,
        ),
    )

    # Seed the initial grid B with the PLASMA self-field only: the split-field
    # external-A machinery adds B_ext(t=0) = bias at PIC-loop start (see the 3D
    # deck for the double-count analysis). Seeds are divergence-free, so the
    # initial projection div cleaner stays off (it is not implemented for RTZ
    # anyway; ProjectionDivCleaner aborts).
    if equilibrium_b and not vacuum:
        # Theta-pinch radial force balance d/dr[P + Bz^2/(2 mu0)] = 0 referenced
        # to the bias field at the annulus/wall; seed only the diamagnetic
        # deviation B_eq(r) - bz_bias (x = r in the parser).
        n_ann_eq = N_I * R_PART**2 / (r_outer**2 - R_INNER**2)
        k_eq = 2.0 * constants.mu0 * (T_I0 + te) * constants.q_e
        sgn = "" if bz_bias >= 0 else "-"
        if annulus_w > 0.0:
            e_in = f"0.5*(1.0+tanh((x-{R_INNER})/{annulus_w}))"
            e_out = f"0.5*(1.0+tanh(({r_outer}-x)/{annulus_w}))"
            n_of_r = f"({n_fill}*(1.0-{e_in})+{n_ann_eq}*{e_in}*{e_out})"
        else:
            n_of_r = (
                f"({n_ann_eq}*((x>={R_INNER})*(x<={r_outer}))+{n_fill}*(x<{R_INNER}))"
            )
        bz_seed = f"{sgn}sqrt({bz_bias**2}+{k_eq}*({n_ann_eq}-{n_of_r}))-({bz_bias})"
    else:
        bz_seed = "0"
    sim.add_applied_field(
        picmi.AnalyticInitialField(
            Bx_expression="0",
            By_expression="0",
            Bz_expression=f"{bz_seed}",
            warpx_do_initial_div_cleaning=False,
        )
    )

    if eb_wall:
        # Conducting cylindrical wall as an embedded boundary inside the
        # domain (staircase masking in the hybrid kernels). The implicit
        # function is evaluated in grid coordinates: x = r, so the conductor
        # region r > R_WALL is simply x - R_w > 0.
        sim.embedded_boundary = picmi.EmbeddedBoundary(
            implicit_function="x-R_w", R_w=R_WALL
        )

    # Annular column + low-density interior fill; tanh-softened radial edges so
    # the density gradient (and its diamagnetic current) is grid-resolved.
    # NOTE: unlike the grid-field parsers (which see x = r, y = theta), the
    # INJECTION density/momentum parsers receive Cartesian positions in RTZ
    # (AddPlasma converts before evaluating), so radius is sqrt(x^2+y^2) and
    # theta is atan2(y,x) here — exactly as in the 3D Cartesian deck.
    if vacuum:
        density_expression = "0"
        dist_kwargs = {}
    else:
        n_annulus = N_I * R_PART**2 / (r_outer**2 - R_INNER**2)
        rr = "sqrt(x*x+y*y)"
        dist_kwargs = dict(n_a=n_annulus, n_f=n_fill, R_in=R_INNER, R_out=r_outer)
        if annulus_w > 0.0:
            edge_in = f"0.5*(1.0+tanh(({rr}-R_in)/sw))"
            edge_out = f"0.5*(1.0+tanh((R_out-{rr})/sw))"
            annulus_term = f"n_a*{edge_in}*{edge_out}"
            dist_kwargs["sw"] = annulus_w
        else:
            edge_in = f"({rr}>=R_in)"
            annulus_term = f"n_a*(({rr}>=R_in)*({rr}<=R_out))"
        if perturb_density_eps != 0.0 and perturb_m > 0:
            # Modal seed on the annulus only (the fill stays uniform)
            annulus_term = (
                f"({annulus_term})"
                f"*(1.0+{perturb_density_eps:.9e}"
                f"*cos({perturb_m:d}*(atan2(y,x)-{perturb_phase:.9e})))"
            )
        density_expression = f"n_f*(1.0-{edge_in})+{annulus_term}"
    ions = picmi.Species(
        name="ions",
        mass=m_i,
        charge="q_e",
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=density_expression,
            momentum_expressions=["0", "0", "0"],
            warpx_momentum_spread_expressions=[f"{vth}"] * 3,
            warpx_density_min=0.05 * n_fill,
            **dist_kwargs,
        ),
    )
    sim.add_species(
        ions,
        layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=nppc),
    )

    if collisions and not vacuum:
        ion_ion_coulomb = picmi.CoulombCollisions(
            name="ion_ion_Coulomb",
            species=[ions, ions],
            CoulombLog=12,
        )
        sim.collisions = [ion_ion_coulomb]

    field_diag = picmi.FieldDiagnostic(
        name="field_diag",
        grid=grid,
        period=diag_period,
        data_list=["B", "E", "rho", "J"],
        write_dir="diags",
        warpx_format="openpmd" if openpmd else "plotfile",
        warpx_openpmd_backend="h5" if openpmd else None,
    )
    sim.add_diagnostic(field_diag)

    return sim


def install_wall_scraper(species_name, r_standoff, verbose=False):
    """Hold the plasma off the conducting wall with a Python particle scraper.

    RTZ port of the 3D scraper: the radius is the STORED first position
    component (r), so the test is a direct comparison, no sqrt. Particles with
    r > r_standoff are flagged invalid (bit 63 of idcpu cleared) so the
    same-step Redistribute removes them. Models a dielectric (quartz liner)
    standoff so the plasma never contacts the wall boundary.
    """
    from pywarpx import callbacks
    from pywarpx.LoadThirdParty import load_cupy
    from pywarpx.particle_containers import ParticleContainerWrapper

    ions = ParticleContainerWrapper(species_name)
    keep_valid_bits = 0x7FFFFFFFFFFFFFFF
    scraped_total = [0]

    def scrape():
        xp, _ = load_cupy()
        r_arrays = ions.get_particle_r(level=0)
        id_arrays = ions.get_particle_idcpu_arrays(level=0)
        n_scraped = 0
        for rr, idc in zip(r_arrays, id_arrays):
            mask = rr > r_standoff
            n = int(xp.count_nonzero(mask))
            if n > 0:
                idc[mask] = idc[mask] & keep_valid_bits
                n_scraped += n
        if n_scraped and verbose:
            scraped_total[0] += n_scraped
            print(f"wall scraper: removed {n_scraped} ions "
                  f"(total {scraped_total[0]})", flush=True)

    callbacks.installparticlescraper(scrape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true",
                        help="CI-sized quick run")
    parser.add_argument("--nr", type=int, default=64,
                        help="radial cells across [0, R_WALL] (default 64)")
    parser.add_argument("--ntheta", type=int, default=64,
                        help="azimuthal cells across 2*pi (default 64)")
    parser.add_argument("--nz", type=int, default=8,
                        help="axial cells in the periodic slab (default 8)")
    parser.add_argument("--nppc", type=int, default=100,
                        help="macroparticles per cell in the loaded region")
    parser.add_argument("--steps", type=int, default=None,
                        help="override the step count (default: ramp + 1 t_ci)")
    parser.add_argument("--diag-steps", type=int, default=None,
                        help="field diagnostic period (default: steps/100)")
    parser.add_argument("--f-tci", type=float, default=F_T_CI,
                        help="dt as a fraction of the reversed-field t_ci")
    parser.add_argument("--bz-rev", type=float, default=BZ_REV)
    parser.add_argument("--bz-bias", type=float, default=BZ_BIAS)
    parser.add_argument("--te", type=float, default=T_E0)
    parser.add_argument("--eta-power", type=float, default=ETA_POWER)
    parser.add_argument("--eta-ntrans", type=float, default=N_TRANSITION_FRAC)
    parser.add_argument("--eta-vac-frac", type=float, default=ETA_VAC_FRAC)
    parser.add_argument("--n-floor-frac", type=float, default=N_FLOOR_FRAC)
    parser.add_argument("--fill-frac", type=float, default=None,
                        help="interior fill density as a fraction of N_I")
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--substep-rtol", type=float, default=1.0e-3)
    parser.add_argument("--holmstrom", action="store_true",
                        help="Holmstrom vacuum-region Ohm's law")
    parser.add_argument("--holmstrom-switch-mode", default="cell",
                        choices=["edge", "node", "cell"],
                        help="vacuum-switch decision-density sampling (with "
                        "--holmstrom; default cell = the EB branch's "
                        "staggered config)")
    parser.add_argument("--holmstrom-blend-pow", type=float, default=1.0,
                        help="above-floor blend power (0 disables; with "
                        "--holmstrom; default 1 = p1w4 config)")
    parser.add_argument("--holmstrom-blend-width", type=float, default=4.0,
                        help="blend window upper edge in units of the floor")
    parser.add_argument("--equilibrium-b", action="store_true",
                        help="seed the diamagnetic equilibrium Bz(r) profile")
    parser.add_argument("--vacuum", action="store_true",
                        help="no plasma: resistive-vacuum drive only")
    parser.add_argument("--no-collisions", action="store_true",
                        help="disable ion-ion Coulomb collisions")
    parser.add_argument("--annulus-smooth-cells", type=float,
                        default=ANNULUS_SMOOTH_CELLS)
    parser.add_argument("--standoff-cells", type=float, default=3.0,
                        help="dielectric standoff from the wall in cells; sets "
                        "the annulus outer radius and the Python scraper radius "
                        "(0 disables the scraper and uses R_OUTER)")
    parser.add_argument("--r-outer", type=float, default=R_OUTER,
                        help="annulus outer radius when --standoff-cells 0")
    parser.add_argument("--particle-shape", type=int, default=1)
    parser.add_argument("--perturb-m", type=int, default=0,
                        help="azimuthal mode number of the seeded perturbation")
    parser.add_argument("--perturb-eps", type=float, default=0.0,
                        help="relative amplitude of the modal drive-field "
                        "perturbation (external A_theta)")
    parser.add_argument("--perturb-phase", type=float, default=0.0,
                        help="phase (rad) of the modal perturbation")
    parser.add_argument("--perturb-density-eps", type=float, default=0.0,
                        help="relative amplitude of the modal seed on the "
                        "loaded annulus density")
    parser.add_argument("--eb-wall", action="store_true",
                        help="model the conducting wall as an embedded "
                        "boundary at R_WALL inside a larger domain (r up to "
                        "1.0 m) instead of the r domain edge")
    parser.add_argument("--openpmd", action="store_true",
                        help="openPMD (h5) field output instead of plotfiles")
    parser.add_argument("-v", "--verbose", type=int, default=1)
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if args.test:
        nr, ntheta, nz = 16, 16, 4
        nppc = 4
        max_steps = 10
        diag_period = 10
        # at nr=16 the default 3-cell standoff would push the annulus outer
        # radius to R_INNER; keep a 1-cell standoff so the scraper still runs
        args.standoff_cells = min(args.standoff_cells, 1.0)
    else:
        nr, ntheta, nz = args.nr, args.ntheta, args.nz
        nppc = args.nppc
        m_i = M_AMU * constants.m_p
        t_ci = 2.0 * np.pi * m_i / (constants.q_e * abs(args.bz_rev))
        dt = args.f_tci * t_ci
        max_steps = (
            args.steps if args.steps is not None else int((TAU_RAMP + t_ci) / dt)
        )
        diag_period = (
            args.diag_steps if args.diag_steps is not None else max(max_steps // 100, 1)
        )

    # Tie the annulus outer radius to the dielectric standoff (see the 3D deck:
    # keeps the ion inventory identical across standoff choices).
    dr = (1.0 if args.eb_wall else R_WALL) / nr
    if args.standoff_cells > 0.0:
        r_outer_eff = R_WALL - args.standoff_cells * dr
    else:
        r_outer_eff = args.r_outer
    if not args.vacuum and r_outer_eff <= R_INNER:
        parser.error(
            f"effective annulus outer radius {r_outer_eff:.4f} m does not "
            f"exceed R_INNER={R_INNER} m; increase --nr or reduce "
            "--standoff-cells."
        )

    sim = setup_simulation(
        nr,
        ntheta,
        nz,
        nppc,
        max_steps,
        diag_period,
        args.verbose,
        holmstrom=args.holmstrom,
        holmstrom_switch_mode=args.holmstrom_switch_mode,
        holmstrom_blend_pow=args.holmstrom_blend_pow,
        holmstrom_blend_width=args.holmstrom_blend_width,
        eta_power=args.eta_power,
        eta_ntrans=args.eta_ntrans,
        eta_vac_frac=args.eta_vac_frac,
        f_t_ci=args.f_tci,
        r_outer=r_outer_eff,
        n_floor_frac=args.n_floor_frac,
        substeps=args.substeps,
        substep_rtol=args.substep_rtol,
        te=args.te,
        bz_rev=args.bz_rev,
        bz_bias=args.bz_bias,
        annulus_smooth_cells=args.annulus_smooth_cells,
        equilibrium_b=args.equilibrium_b,
        vacuum=args.vacuum,
        fill_frac=args.fill_frac,
        collisions=not args.no_collisions,
        particle_shape=args.particle_shape,
        perturb_m=args.perturb_m,
        perturb_eps=args.perturb_eps,
        perturb_phase=args.perturb_phase,
        perturb_density_eps=args.perturb_density_eps,
        eb_wall=args.eb_wall,
        openpmd=args.openpmd,
    )

    if args.standoff_cells > 0.0 and not args.vacuum:
        install_wall_scraper("ions", r_outer_eff, verbose=args.verbose > 1)

    sim.step()


if __name__ == "__main__":
    main()
