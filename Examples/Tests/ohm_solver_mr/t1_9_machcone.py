#!/usr/bin/env python3
"""T1.9 -- Mach-cone angle test for hybrid-PIC mesh refinement.

A.I.K.E.F. (Mueller et al. 2011) Sec. 7.4 design, FRAME-FLIPPED: a heavy-ion
cloud obstacle moves super-magnetosonically through a plasma AT REST on the
grid, inside a long static refinement patch.  The obstacle trails a standing
(in its own frame) Mach cone with half-angle

    theta = asin(v_ms / v_rel),

and v_rel is chosen so the analytic target equals the A.I.K.E.F. template
value of 15.6 deg.

WHY the frame flip (measured, 2026-08-07; runs t19_null_*): the textbook
frame -- bulk plasma E x B-drifting across the grid at M_ms = 3.7 -- is
numerically unstable in this solver at any tested beta: a drift-resonant
mode where the dispersive magnetosonic-whistler branch crosses the beam
line (kz l_i ~ 2.9 at beta 0.05, oblique |k| l_i ~ 2.7 at beta 1, offsets
= cyclotron harmonics) grows at gamma = 0.56 w_ci (beta 0.05) ... 0.07 w_ci
(beta 1) INDEPENDENT of the obstacle, and neither eta_h (mode slides
down-branch; tested up to 48 eta_h*), binomial filtering, nor beta = 1 +
5 eta_h* suppresses it.  With the plasma at rest the channel is absent (the
quiet-plasma T1 batteries are flat-stable); the heavy (1e4 M_i), weak
(n_pk = 0.05 n0) moving cloud is numerically benign, and physics is
identical by Galilean invariance.

Geometry (genuinely 2D):
  * (x, z) periodic, 256 x 512 coarse cells (dx = dz = 0.5 l_i,
    128 x 256 l_i); B0 = 0.25 T along +y (out of plane) -> every in-plane
    wavevector is perpendicular to B0, the in-plane fast magnetosonic speed
    is isotropic, the cone symmetric.  beta_i = beta_e = 0.05:
    v_ms = vA sqrt(1 + [g_e beta_e + g_i beta_i]/2) = 1.0368 vA (g_e = 1
    isothermal electrons, g_i = 2 CGL-perpendicular); v_rel = v_ms /
    sin(15.6 deg) = 3.855 vA.
  * Obstacle: Gaussian heavy-ion cloud (mass 1e4 M_i, charge e,
    sigma = 3 l_i, n_pk = 0.05 n0), all particles at uz = -v_rel, starting
    at z = 244 l_i.  Mass loading: inside the cloud the rho-weighted bulk
    velocity (and with it the motional E in Ohm's law) is eps*v_obs,
    eps = n_pk/(n0+n_pk) -- a moving weak obstacle.  Pickup displacement
    of the cloud trajectory over the horizon is < 0.1 l_i.
  * MR arm: ratio-2 patch x in [-12,12], z in [32,224] l_i
    (+ warpx.refine_plasma_init = 1).  The trailing wings cross the two
    SIDE seams at the comoving-stationary station zeta = 12/tan(15.6 deg)
    = 43 l_i behind the obstacle (while the obstacle is inside the patch),
    and sweep the downstream z-seam (z = 224) continuously after entry at
    t = 0.66 t_ci.  Control arm: identical deck at max_level = 0.
  * Horizon 1750 steps x 0.005 t_ci = 8.75 t_ci: the obstacle travels
    212 l_i (no wrap); comoving standing-state analysis window
    t in [5.8, 8.2] t_ci (obstacle z in [104, 44] l_i, wing developed to
    >= 140 l_i, obstacle >= 12 l_i inside the patch).

Analysis (t1_9_analyze.py): maps are shifted to the obstacle frame (exact
periodic FFT phase shift) and time-averaged; wing ridges are traced and fit
in comoving coordinates zeta = z - z_obs(t).

Example:
  python t1_9_machcone.py --name t19_mr --max-level 1 --out t1_runs/t19_mr
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_whistler import (  # noqa: E402  (shared physics, identical by import)
    B0,
    ETA_H_STAR,
    M_ION,
    c_light,
    l_i,
    n0,
    run_warpx,
    t_ci,
    vA,
    w_ci,
)

# ----------------------------------------------------------------------
# grid: dx = dz = 0.5 l_i (coarse); genuinely 2D
# ----------------------------------------------------------------------
DXY_LI = 0.5
NX_COARSE = 256  # x in [-64, 64] l_i
NZ_COARSE = 512  # z in [0, 256] l_i
dz_c = DXY_LI * l_i
Lx = NX_COARSE * dz_c
Lz = NZ_COARSE * dz_c

# patch (multiples of blocking_factor*dz from the domain edges)
PATCH_XHW_LI = 12.0  # patch |x| <= 12 l_i
PATCH_ZLO_LI, PATCH_ZHI_LI = 32.0, 224.0  # long: obstacle stays inside
# while its trailing wing develops (frame-flipped design, see below)

# obstacle: STARTS at Z_OBS_LI and moves in -z at v_flow (frame-flipped:
# plasma at rest on the grid -- no bulk grid-drift -- obstacle supplies
# the relative super-magnetosonic motion; physics identical by Galilean
# invariance, numerics benign)
Z_OBS_LI = 244.0
SIGMA_OBS_LI = 3.0
M_OBS_FAC = 1.0e4  # obstacle mass / M_ION
# Peak obstacle density / n0.  MEASURED (smoke t19_smoke_mr / t19_smoke_ctl16):
# n_pk = 1.0 n0 is an impenetrable magnetosphere-like obstacle at M_A = 3.9
# (stagnation pileup B/B0 ~ sqrt(1+2 M_A^2) ~ 5.5): |dBy| reached 2.9 B0 by
# t = 0.5 t_ci and both MR and uniform-coarse arms NaN'd (whistler CFL under
# pileup; obstacle-generic, not a seam effect).  The linear Mach-cone regime
# needs a WEAK mass-loading cloud.  The switch-on transient scales as
# eps*M_ms*B0 (eps = n_pk/(n0+n_pk)): at n_pk = 0.1 the ~0.3 B0 transient
# still steepened into a dispersive shock train (soliton spikes ~2 B0, rho
# dips 0.14 n0 at the wing front) and NaN'd both arms -> default 0.05 plus
# the eta_h = 0.2 eta_h* absorber (T1.2 floor) in ALL arms identically.
N_PK_FAC = 0.05
OBS_BOUND_LI = 10.0  # injection bounds around the center (> 3.33 sigma)

# plasma betas (per species)
BETA_I = 0.05
BETA_E = 0.05
GAMMA_E = 1.0  # deck uses hybrid_pic_model.gamma = 1 (isothermal electrons)
GAMMA_I = 2.0  # CGL-perpendicular kinetic ion response (fast wave k _|_ B)

THETA_TARGET_DEG = 15.6  # A.I.K.E.F. template cone half-angle

INPUTS_TEMPLATE = """# auto-generated by t1_9_machcone.py -- do not edit ({case_name})
max_step = {steps}
warpx.const_dt = {dt:.10e}
warpx.verbose = 1

amr.n_cell = {nx} {nz}
amr.max_level = {max_level}
amr.blocking_factor = 4
amr.max_grid_size = {max_grid_size}
amr.n_error_buf = 0
{fine_tag_block}
geometry.dims = 2
geometry.prob_lo = {xlo:.10e} 0.0
geometry.prob_hi = {xhi:.10e} {Lz:.10e}

boundary.field_lo = periodic periodic
boundary.field_hi = periodic periodic
boundary.particle_lo = periodic periodic
boundary.particle_hi = periodic periodic

algo.maxwell_solver = hybrid
algo.current_deposition = direct
algo.particle_shape = {shape}

hybrid_pic_model.elec_temp = {Te:.10e}
hybrid_pic_model.gamma = 1.0
hybrid_pic_model.n0_ref = {n0:.10e}
hybrid_pic_model.plasma_resistivity(rho,J,t) = {eta}
hybrid_pic_model.plasma_hyper_resistivity(rho,B) = {eta_h}
hybrid_pic_model.substeps = {substeps}

# uniform B0 OUT of the (x,z) plane
warpx.B_ext_grid_init_style = parse_b_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "0.0"
warpx.By_external_grid_function(x,y,z) = "{B0:.10e}"
warpx.Bz_external_grid_function(x,y,z) = "0.0"

particles.species_names = ions obstacle

ions.charge = q_e
ions.mass = {M_ion:.10e}
ions.injection_style = nrandompercell
ions.num_particles_per_cell = {ppc}
ions.profile = constant
ions.density = {n0:.10e}
ions.momentum_distribution_type = gaussian
ions.ux_th = {uth:.10e}
ions.uy_th = {uth:.10e}
ions.uz_th = {uth:.10e}

obstacle.charge = q_e
obstacle.mass = {M_obs:.10e}
obstacle.injection_style = nrandompercell
obstacle.num_particles_per_cell = {ppc}
obstacle.profile = parse_density_function
obstacle.density_function(x,y,z) = "{n_pk:.10e}*exp(-((x-0.0)^2+(z-{z_obs:.10e})^2)/(2*{sig_obs:.10e}^2))"
obstacle.density_min = {obs_dens_min:.10e}
obstacle.xmin = {obs_xmin:.10e}
obstacle.xmax = {obs_xmax:.10e}
obstacle.zmin = {obs_zmin:.10e}
obstacle.zmax = {obs_zmax:.10e}
obstacle.momentum_distribution_type = constant
obstacle.uz = -{u_flow:.10e}

diagnostics.diags_names = diag1
diag1.diag_type = Full
diag1.format = plotfile
diag1.intervals = {plot_int}
diag1.fields_to_plot = Bx By Bz Ex Ey Ez rho
diag1.write_species = 0

warpx.reduced_diags_names = field_energy
field_energy.type = FieldEnergy
field_energy.intervals = 10

warpx.serialize_initial_conditions = 1
"""


def build_case(args):
    beta_i = beta_e = args.beta
    v_ms = vA * np.sqrt(1.0 + 0.5 * (GAMMA_E * beta_e + GAMMA_I * beta_i))
    theta = np.deg2rad(THETA_TARGET_DEG)
    v_flow = v_ms / np.sin(theta)

    if args.resolution == "coarse":
        nx, nz = NX_COARSE, NZ_COARSE
    else:  # uniform-fine reference (not used by the battery; kept for parity)
        nx, nz = 2 * NX_COARSE, 2 * NZ_COARSE
        assert args.max_level == 0, "fine resolution is a max_level=0 reference"

    v_ti = np.sqrt(beta_i / 2.0) * vA
    u_th = v_ti / c_light
    Te_eV = M_ION * (beta_e / 2.0) * vA**2 / 1.602176634e-19

    if args.max_level > 0:
        fine_tag_block = (
            f"warpx.fine_tag_lo = {-PATCH_XHW_LI * l_i:.10e} "
            f"{PATCH_ZLO_LI * l_i:.10e}\n"
            f"warpx.fine_tag_hi = {+PATCH_XHW_LI * l_i:.10e} "
            f"{PATCH_ZHI_LI * l_i:.10e}\n"
            "warpx.refine_plasma_init = 1\n"
        )
    else:
        fine_tag_block = ""

    dt = args.dt_fac * t_ci
    n_pk = args.npk_fac * n0
    eta_h = args.eta_h_fac * ETA_H_STAR

    case = dict(
        case_name=args.name,
        steps=args.steps,
        dt=dt,
        dt_fac=args.dt_fac,
        nx=nx,
        nz=nz,
        max_grid_size=args.max_grid_size,
        max_level=args.max_level,
        resolution=args.resolution,
        xlo=-0.5 * Lx,
        xhi=+0.5 * Lx,
        Lz=Lz,
        Lx=Lx,
        fine_tag_block=fine_tag_block,
        Te=Te_eV,
        n0=n0,
        eta=args.eta,
        eta_h=f"{eta_h:.10e}" if args.eta_h_fac > 0 else "0.0",
        eta_h_fac=args.eta_h_fac,
        substeps=args.substeps,
        B0=B0,
        M_ion=M_ION,
        M_obs=M_OBS_FAC * M_ION,
        ppc=args.ppc,
        uth=u_th,
        u_flow=v_flow / c_light,
        n_pk=n_pk,
        z_obs=Z_OBS_LI * l_i,
        sig_obs=SIGMA_OBS_LI * l_i,
        obs_dens_min=1.0e-2 * n_pk,
        obs_xmin=-OBS_BOUND_LI * l_i,
        obs_xmax=+OBS_BOUND_LI * l_i,
        obs_zmin=(Z_OBS_LI - OBS_BOUND_LI) * l_i,
        obs_zmax=(Z_OBS_LI + OBS_BOUND_LI) * l_i,
        plot_int=args.plot_int,
        # bookkeeping for the analysis
        mode="machcone",
        theta_target_deg=THETA_TARGET_DEG,
        v_ms=v_ms,
        v_ms_va=v_ms / vA,
        v_flow=v_flow,
        v_flow_va=v_flow / vA,
        mach_ms=v_flow / v_ms,
        beta_i=beta_i,
        beta_e=beta_e,
        shape=args.shape,
        gamma_i=GAMMA_I,
        gamma_e=GAMMA_E,
        n_pk_fac=args.npk_fac,
        m_obs_fac=M_OBS_FAC,
        sigma_obs_li=SIGMA_OBS_LI,
        z_obs_li=Z_OBS_LI,
        patch_xhw_li=PATCH_XHW_LI,
        patch_zlo_li=PATCH_ZLO_LI,
        patch_zhi_li=PATCH_ZHI_LI,
        patch_zlo=PATCH_ZLO_LI * l_i,
        patch_zhi=PATCH_ZHI_LI * l_i,
        w_ci=w_ci,
        t_ci=t_ci,
        l_i=l_i,
        vA=vA,
        dz_coarse=dz_c,
        extra_inputs=args.extra_inputs,
    )
    return case


def write_run(case, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    inputs_path = os.path.join(out_dir, "inputs_t1")
    text = INPUTS_TEMPLATE.format(**case)
    if case["n_pk"] <= 0.0:  # null run: no obstacle species at all
        text = text.replace(
            "particles.species_names = ions obstacle", "particles.species_names = ions"
        )
        text = (
            "\n".join(ln for ln in text.splitlines() if not ln.startswith("obstacle."))
            + "\n"
        )
    with open(inputs_path, "w") as f:
        f.write(text)
        for line in case.get("extra_inputs") or []:
            f.write(line + "\n")
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(
            {k: v for k, v in case.items() if not k.endswith("_block")},
            f,
            indent=1,
            default=float,
        )
    return inputs_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="t19_run")
    p.add_argument("--max-level", type=int, default=1, choices=[0, 1])
    p.add_argument("--resolution", choices=["coarse", "fine"], default="coarse")
    p.add_argument("--eta", default="1.0e-7", help="plasma_resistivity (Ohm m)")
    p.add_argument(
        "--npk-fac", type=float, default=N_PK_FAC, help="peak obstacle density / n0"
    )
    p.add_argument(
        "--beta",
        type=float,
        default=BETA_I,
        help="per-species plasma beta (beta_i = beta_e)",
    )
    p.add_argument("--shape", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--eta-h-fac",
        type=float,
        default=0.2,
        help="hyper-resistivity / eta_h* (T1.2 absorber floor; "
        "scale-selective: damps the dispersive-shock spikes at "
        "k ~ k_Nyq, cone band k/k_Nc ~ 0.05 damped by (k/k_Nc)^4)",
    )
    p.add_argument("--substeps", type=int, default=24)
    p.add_argument("--max-grid-size", type=int, default=64)
    p.add_argument("--extra-inputs", action="append", default=None)
    p.add_argument("--ppc", type=int, default=64)
    p.add_argument("--dt-fac", type=float, default=0.005, help="dt / t_ci")
    p.add_argument("--steps", type=int, default=1750)
    p.add_argument("--plot-int", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--exe", default=None)
    p.add_argument("--nprocs", type=int, default=24)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.exe is None:
        args.exe = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "build",
            "bin",
            "warpx.2d",
        )

    case = build_case(args)
    write_run(case, args.out)
    tt = case["Lz"] / case["v_flow"] / t_ci  # obstacle lap time
    print(
        f"[t1_9_machcone] {args.name}: v_ms = {case['v_ms_va']:.4f} vA, "
        f"v_flow = {case['v_flow_va']:.4f} vA (M_ms = {case['mach_ms']:.4f}), "
        f"analytic cone half-angle = {THETA_TARGET_DEG} deg"
    )
    print(
        f"[t1_9_machcone] domain {Lx / l_i:.0f} x {Lz / l_i:.0f} l_i, "
        f"obstacle at z = {Z_OBS_LI} l_i (sigma {SIGMA_OBS_LI} l_i, "
        f"n_pk {case['n_pk_fac']:g} n0, mass {M_OBS_FAC:g} M_i); "
        f"flow domain-crossing = {tt:.2f} t_ci; "
        f"horizon = {args.steps * args.dt_fac:.1f} t_ci"
    )
    if args.dry_run:
        return 0
    rc = run_warpx(case, args.out, args.exe, args.nprocs)
    if rc != 0:
        print(f"[t1_9_machcone] WARNING: {args.name} exited with code {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
