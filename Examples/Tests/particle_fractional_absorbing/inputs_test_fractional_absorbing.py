#!/usr/bin/env python3
"""Ballistic wall encounters through native domain/EB and subcycled paths.

Every tagged particle strikes the selected wall once. Check inventory, exact
survivor identity/weight, specular momentum, and absorbed-only loss accounting.
Statistics, rather than an RNG-dependent checksum, are the intermediate oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from pywarpx import fields, particle_containers, picmi
from pywarpx._libwarpx import libwarpx

parser = argparse.ArgumentParser()
parser.add_argument("--dim", choices=["RZ", "3"], default="RZ")
parser.add_argument(
    "--wall", choices=["rhi", "zlo", "zhi", "eb", "curved", "corner"], default="zhi"
)
parser.add_argument("--fraction", type=float, default=0.5)
parser.add_argument(
    "--mode",
    choices=["fractional_absorbing", "absorbing", "reflecting", "legacy"],
    default="fractional_absorbing",
)
parser.add_argument("--n", type=int, default=100000)
parser.add_argument("--subcycled", action="store_true")
parser.add_argument("--coupled", action="store_true")
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--repeat", action="store_true")
parser.add_argument("--checkpoint", action="store_true")
parser.add_argument("--restart")
parser.add_argument("--write-inputs")
args = parser.parse_args()
comm = MPI.COMM_WORLD
is_rz = args.dim == "RZ"
is_eb = args.wall in ("eb", "curved")
assert is_rz or args.wall not in ("rhi", "corner")
assert not is_eb or not is_rz
assert args.mode != "legacy" or not is_eb
assert not args.repeat or (args.wall in ("zlo", "zhi") and not args.coupled)
dt = 1.0e-9
u_normal = 3e7
u_tangent = (4e6, 7e6)
gamma = np.sqrt(1 + (u_normal**2 + sum(u**2 for u in u_tangent)) / picmi.constants.c**2)
# A thin axial box gives exactly one hit per full step for repeated encounters.
length_z = u_normal * dt / gamma if args.repeat else 1.0
mode = "absorbing" if args.mode == "legacy" else args.mode
fraction = (
    args.fraction
    if args.mode in ("fractional_absorbing", "legacy")
    else float(mode == "absorbing")
)

lo = ["none", "reflecting"] if is_rz else ["reflecting"] * 3
hi = ["reflecting"] * (2 if is_rz else 3)
face_dim = 0 if args.wall == "rhi" else (1 if is_rz else 2)
face_side = 0 if args.wall == "zlo" else 1
if not is_eb:
    (lo if face_side == 0 else hi)[face_dim] = mode
    if args.repeat:
        lo[face_dim] = hi[face_dim] = mode
    if args.wall == "corner":
        hi[0] = hi[1] = mode
bounds_lo = [0.0, 0.0] if is_rz else [-1.0, -1.0, 0.0]
bounds_hi = [1.0, length_z] if is_rz else [1.0, 1.0, length_z]
grid = (picmi.CylindricalGrid if is_rz else picmi.Cartesian3DGrid)(
    number_of_cells=[32] * len(lo),
    lower_bound=bounds_lo,
    upper_bound=bounds_hi,
    lower_boundary_conditions=["none", "dirichlet"] if is_rz else ["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * len(lo),
    lower_boundary_conditions_particles=lo,
    upper_boundary_conditions_particles=hi,
    warpx_max_grid_size=16,
    warpx_blocking_factor=8,
)
eb = None
if is_eb:
    eb = picmi.EmbeddedBoundary(
        implicit_function="z-0.5" if args.wall == "eb" else "x*x+y*y-0.25",
        particle_boundary=mode,
    )
if args.coupled:
    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=0.0,
        n0=1.0e15,
        n_floor=1.0e14,
        plasma_resistivity=0.0,
        plasma_hyper_resistivity=0.0,
        substeps=2,
    )
else:
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=None,
    particle_shape=3,
    warpx_current_deposition_algo="esirkepov",
    warpx_particle_pusher_algo="boris",
    warpx_use_filter=False,
    warpx_serialize_initial_conditions=True,
    warpx_embedded_boundary=eb,
    warpx_random_seed=47,
    warpx_particle_absorption_fraction=args.fraction
    if args.mode == "fractional_absorbing"
    else None,
    verbose=0,
)
# Ordinary ions, a charged alpha-like species and a neutral product share f.
# Use unequal marker weights to check actual weighted losses as well as counts.
species = []
for name, mass, charge in [
    ("ions", picmi.constants.m_p, picmi.constants.q_e),
    ("alpha", 4 * picmi.constants.m_p, 2 * picmi.constants.q_e),
    ("neutral", picmi.constants.m_p, 0.0),
]:
    kw = (
        {"warpx_save_particles_at_eb": True}
        if is_eb
        else {
            "warpx_save_particles_at_"
            + ("xhi" if args.wall == "rhi" else args.wall): True
        }
    )
    if args.wall == "corner":
        kw = {"warpx_save_particles_at_xhi": True, "warpx_save_particles_at_zhi": True}
    if args.repeat:
        kw["warpx_save_particles_at_zlo"] = True
        kw["warpx_save_particles_at_zhi"] = True
    if args.mode == "legacy":
        kw["warpx_reflection_model_" + ("xhi" if args.wall == "rhi" else args.wall)] = (
            str(1.0 - fraction)
        )
        if args.repeat:
            kw["warpx_reflection_model_zlo"] = str(1.0 - fraction)
            kw["warpx_reflection_model_zhi"] = str(1.0 - fraction)
    sp = picmi.Species(
        name=name,
        mass=mass,
        charge=charge,
        warpx_do_not_deposit=not args.coupled,
        warpx_do_not_gather=True,
        warpx_add_real_attributes={"tag": "0"},
        **kw,
    )
    sim.add_species(sp, layout=None)
    species.append(sp)
if args.checkpoint:
    sim.add_diagnostic(
        picmi.Checkpoint(period=args.steps, write_dir="checkpoints", name="chk")
    )
sim.initialize_inputs()
if args.subcycled:
    species[1].species.do_subcycled_push = 1
    species[1].species.subcycling_cfl_grid = 0.15
    species[1].species.subcycling_v_ref = 3.2e7
    species[1].species.subcycling_max_subcycles = 128
import pywarpx

pywarpx.amrex.the_arena_init_size = 0
if not args.coupled:
    pywarpx.algo.maxwell_solver = "none"
else:
    pywarpx.warpx.B_ext_grid_init_style = "constant"
    pywarpx.warpx.B_external_grid = [0.0, 0.0, 0.01]
if args.restart:
    pywarpx.amr.restart = args.restart
if args.write_inputs:
    sim.write_input_file(args.write_inputs)
    raise SystemExit(0)
sim.initialize_warpx()

# Proper velocities give a 3 cm step. Tangential velocity tests specularity.
local_n = args.n if comm.rank == 0 else 0
tag = np.arange(local_n, dtype=float)
angle = 2 * np.pi * (tag + 0.5) / args.n
rad = (
    0.99 if args.wall in ("rhi", "corner") else (0.49 if args.wall == "curved" else 0.3)
)
x, y = rad * np.cos(angle), rad * np.sin(angle)
z = np.full(
    local_n, 0.49 if args.wall == "eb" else (0.01 if args.wall == "zlo" else 0.99)
)
if args.repeat:
    z[:] = length_z / 2
if args.wall in ("rhi", "curved", "corner"):
    if args.wall != "corner":
        z[:] = 0.25 + 0.5 * (tag + 0.5) / args.n
    ux = 3e7 * np.cos(angle) - 7e6 * np.sin(angle)
    uy = 3e7 * np.sin(angle) + 7e6 * np.cos(angle)
    uz = np.full(local_n, 3e7 if args.wall == "corner" else 4e6)
else:
    ux, uy = np.full(local_n, 4e6), np.full(local_n, 7e6)
    uz = np.full(local_n, -3e7 if args.wall == "zlo" else 3e7)
w = 1.0 + 0.25 * (tag % 5)
for sp in [] if args.restart else species:
    sim.particles.get(sp.name).add_particles(
        x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w, tag=tag, unique_particles=True
    )


def host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def collect(arrays, dtype=float):
    local = (
        np.concatenate([host(a) for a in arrays])
        if arrays
        else np.empty(0, dtype=dtype)
    )
    chunks = comm.allgather(local)
    return np.concatenate(chunks)


def snapshot(name):
    pc = particle_containers.ParticleContainerWrapper(name)
    data = {
        k: collect(pc.get_particle_real_arrays(k, 0, copy_to_host=True))
        for k in (["r", "theta", "z"] if is_rz else ["x", "y", "z"])
        + ["ux", "uy", "uz", "w", "tag"]
    }
    data["idcpu"] = collect(
        pc.get_particle_idcpu_arrays(0, copy_to_host=True), dtype=np.uint64
    )
    if is_rz:
        radius = data["r"].copy()
        data["x"], data["y"] = (
            radius * np.cos(data["theta"]),
            radius * np.sin(data["theta"]),
        )
    order = np.argsort(data["tag"])
    return {k: v[order] for k, v in data.items()}


initial = {sp.name: snapshot(sp.name) for sp in species}
sim.step(args.steps)
results = {}
buffer = particle_containers.ParticleBoundaryBufferWrapper()
for sp in species:
    before, after = initial[sp.name], snapshot(sp.name)
    n_ref = len(after["tag"])
    selected = np.searchsorted(before["tag"], after["tag"])
    n_incident = len(before["tag"])
    if not args.restart:
        assert n_incident == args.n
    assert len(np.unique(selected)) == n_ref
    assert np.array_equal(after["idcpu"], before["idcpu"][selected]), (
        "survivor identity changed"
    )
    assert np.array_equal(after["w"], before["w"][selected]), "survivor weight changed"
    n_abs = n_incident - n_ref
    probability = 1 - (1 - fraction) ** (
        args.steps if args.repeat else (2 if args.wall == "corner" else 1)
    )
    sigma = np.sqrt(n_incident * probability * (1 - probability))
    assert abs(n_abs - n_incident * probability) <= 6 * sigma + 1, (
        sp.name,
        n_abs,
        probability,
    )
    if probability in (0, 1):
        assert n_abs == int(n_incident * probability)
    u0 = np.array([before[k][selected] for k in ["ux", "uy", "uz"]])
    u1 = np.array([after[k] for k in ["ux", "uy", "uz"]])
    np.testing.assert_allclose(
        np.sum(u1 * u1, axis=0), np.sum(u0 * u0, axis=0), rtol=2e-12
    )
    if args.wall in ("zhi", "zlo", "eb"):
        np.testing.assert_allclose(u1[:2], u0[:2], rtol=2e-12)
        np.testing.assert_allclose(
            u1[2], (-1) ** (args.steps if args.repeat else 1) * u0[2], rtol=2e-12
        )
        wall = 0.5 if is_eb else float(face_side)
        speed = u0 / np.sqrt(1 + np.sum(u0 * u0, axis=0) / picmi.constants.c**2)
        expected = (
            before["z"][selected]
            if args.repeat
            else 2 * wall - (before["z"][selected] + speed[2] * dt * args.steps)
        )
        np.testing.assert_allclose(after["z"], expected, atol=2e-12, rtol=2e-12)
    else:
        np.testing.assert_allclose(
            u1[2], (-1 if args.wall == "corner" else 1) * u0[2], rtol=2e-12
        )
        radial = after["x"] * after["ux"] + after["y"] * after["uy"]
        assert np.all(radial < 0), "reflected radial momentum is not inward"
    radius = np.hypot(after["x"], after["y"])
    assert np.all(radius <= (0.5 if args.wall == "curved" else 1.0))
    assert np.all(after["z"] >= 0)
    assert np.all(after["z"] <= (0.5 if args.wall == "eb" else length_z))
    bname = (
        "eb"
        if is_eb
        else (
            "x_hi" if args.wall == "rhi" else "z_" + ("lo" if face_side == 0 else "hi")
        )
    )
    bnames = (
        ["z_lo", "z_hi"]
        if args.repeat
        else (["x_hi", "z_hi"] if args.wall == "corner" else [bname])
    )
    lost_tag = collect(
        [
            a
            for bn in bnames
            for a in buffer.get_particle_boundary_buffer(sp.name, bn, "tag", 0)
        ]
    ).astype(int)
    lost_w = collect(
        [
            a
            for bn in bnames
            for a in buffer.get_particle_boundary_buffer(sp.name, bn, "w", 0)
        ]
    )
    # Existing scraping buffers are per-face geometric records: a corner loss
    # can appear on both faces. The cumulative loss tally must still count once.
    if args.wall == "corner":
        unique, index = np.unique(lost_tag, return_index=True)
        lost_tag, lost_w = unique, lost_w[index]
    lost_index = np.searchsorted(before["tag"], lost_tag)
    assert len(lost_tag) == n_abs, "buffer contains survivors or misses losses"
    assert len(np.unique(lost_tag)) == n_abs, "duplicate absorption"
    assert np.array_equal(
        np.sort(np.concatenate([after["tag"], lost_tag])), before["tag"]
    )
    np.testing.assert_allclose(lost_w, before["w"][lost_index], rtol=0, atol=0)
    absorbed_w = float(np.sum(lost_w))
    np.testing.assert_allclose(
        absorbed_w + np.sum(after["w"]), np.sum(before["w"]), rtol=2e-13
    )
    weight_sigma = np.sqrt(probability * (1 - probability) * np.sum(before["w"] ** 2))
    assert abs(absorbed_w - probability * np.sum(before["w"])) <= 6 * weight_sigma + 1
    if not is_eb:
        sides = [0, 1] if args.repeat else [face_side]
        face_pairs = (
            [(0, 1), (1, 1)]
            if args.wall == "corner"
            else [(face_dim, side) for side in sides]
        )
        charge = sum(
            libwarpx.warpx.get_boundary_absorbed_charge(sp.name, dim, side)
            for dim, side in face_pairs
        )
        energy = sum(
            libwarpx.warpx.get_boundary_absorbed_energy(sp.name, dim, side)
            for dim, side in face_pairs
        )
        u2 = sum(before[k][lost_index] ** 2 for k in ["ux", "uy", "uz"])
        lost_ke = float(
            np.sum(lost_w * sp.mass * u2 / (np.sqrt(1 + u2 / picmi.constants.c**2) + 1))
        )
        np.testing.assert_allclose(
            charge, sp.charge * absorbed_w, rtol=2e-11, atol=1e-30
        )
        np.testing.assert_allclose(energy, lost_ke, rtol=2e-11, atol=1e-30)
    results[sp.name] = {
        "incident": n_incident,
        "absorbed": n_abs,
        "reflected": n_ref,
        "absorbed_weight": absorbed_w,
        "fraction": fraction,
        "absorption_probability_over_run": probability,
    }
    if comm.rank == 0:
        np.savez(sp.name + "_particles.npz", **after)
# Exercise actual fields, including guard fills, for coupled fixtures.
if args.coupled:
    for label, wrapper in [
        ("Ex", fields.ExWrapper),
        ("Ey", fields.EyWrapper),
        ("Ez", fields.EzWrapper),
        ("Bx", fields.BxWrapper),
        ("By", fields.ByWrapper),
        ("Bz", fields.BzWrapper),
        ("Jx", fields.JxFPWrapper),
        ("Jy", fields.JyFPWrapper),
        ("Jz", fields.JzFPWrapper),
        ("rho", fields.RhoFPWrapper),
    ]:
        value = host(wrapper()[...])
        assert np.all(np.isfinite(value)), label
        if comm.rank == 0:
            np.save(label + ".npy", value)
if comm.rank == 0:
    Path("result.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "species": results,
                "runtime": {
                    "pywarpx": pywarpx.__file__,
                    "native": libwarpx.libwarpx_so.__file__,
                    "native_sha256": hashlib.sha256(
                        Path(libwarpx.libwarpx_so.__file__).read_bytes()
                    ).hexdigest(),
                    "mpi": MPI.Get_library_version(),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS fractional_absorbing: native trajectories, statistics and loss accounting",
        flush=True,
    )
# Release wrappers before WarpX finalization.
del buffer
