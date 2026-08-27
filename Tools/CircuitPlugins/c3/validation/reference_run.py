#!/usr/bin/env python3
"""Python reference trajectories for the c3 plugin validation cases.

Drives the REAL hetools machinery -- c3_ref_circuit / c3_stepper /
c3_bank_source (imported from their home trees) -- on the exported case
inputs (mcomb.npy through a stub probe_fn, so both sides consume
bit-identical matrices), with the single-pass WarpX cadence: one
advance per coupling step to the interval end, arm_locks fired between
steps at the boundary nearest machine 0. Output rows match the C++
harness CSV layout (hexfloat scales per accepted step).

Run (deck venv):
    HETOOLS_USE_GPU=0 ~/.env/warpx-mhd/bin/python3 reference_run.py \
        --case <dir> --t-end 14.5e-6 --dt 5e-7 --out ref.csv
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("HETOOLS_USE_GPU", "0")

import numpy as np
import yaml

DECK = (pathlib.Path.home() / "src/hetools/Examples/warpx/Profiles/"
        "production/formation/RZ-mhd-formation")
GREENS = pathlib.Path.home() / "src/hetools-greens/Machines/<machine>"
sys.path.insert(0, str(DECK))
sys.path.insert(0, str(GREENS))

import c3_bank_source as c3src  # noqa: E402
import c3_ref_circuit as c3  # noqa: E402
import c3_stepper  # noqa: E402


def load_manifest(path):
    names, i_ref = [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        nm, ir = line.split()
        names.append(nm)
        i_ref.append(float.fromhex(ir))
    return names, np.array(i_ref)


def load_eps_table(path, port_names):
    lines = [ln for ln in pathlib.Path(path).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    head = lines[0].split()
    assert head[0] == "ports", head
    ports = [port_names.index(nm) for nm in head[1:]]
    rows = [[float.fromhex(t) for t in ln.split()] for ln in lines[1:]]
    return ports, np.array(rows)


def read_matrix_blocks(path):
    toks = pathlib.Path(path).read_text().split()
    assert toks[0] == "c3-mel-blocks" and toks[1] == "v1", toks[:2]
    assert toks[2] == "n"
    n = int(toks[3])
    names = toks[4:4 + n]
    i = 4 + n
    assert toks[i] == "M_EE"
    ee = np.array([float.fromhex(t)
                   for t in toks[i + 1:i + 1 + n * n]]).reshape(n, n)
    i += 1 + n * n
    assert toks[i] == "M_EW"
    ew = np.array([float.fromhex(t)
                   for t in toks[i + 1:i + 1 + n * n]]).reshape(n, n)
    return names, ee, ew


def build_adapter(case, meta, names48, dt_warpx):
    """The mirror-machine path: the REAL c3_bank_source.build() on the
    exported probe matrix."""
    m_comb = np.load(case / "mcomb.npy")
    n = m_comb.shape[0] // 2

    def probe_stub(rz, cur):
        assert len(rz) == 2 * n and len(cur) == 2 * n, (len(rz), n)
        return m_comb, None

    lock_rings = None
    if meta["lock_rings"]:
        lock_rings = [dict(name=rg["name"], rz=tuple(rg["rz"]),
                           R_ohm=rg["R_ohm"], seg=tuple(rg["seg"]))
                      for rg in meta["lock_rings"]]
    coils_rz48 = [(1.0, 1.0)] * len(names48)   # probe is stubbed
    stepper, twin_of, d_full, bmeta = c3src.build(
        meta["yaml"], names48, coils_rz48, probe_stub, meta["dr"],
        meta["dz"], meta["dt"], meta["t0"], meta["shift"],
        log=lambda *a: print(*a, flush=True),
        lock_rings=lock_rings, pre_roll=bool(lock_rings))
    if bmeta.get("lock_swap"):
        swap_step = max(1, int(round(meta["shift"] / dt_warpx)))
        bmeta["lock_swap"]["t_machine"] = (swap_step * dt_warpx
                                           - meta["shift"])
    adapter = c3src.C3CircuitAdapter(stepper, d_full, bmeta["n_e"],
                                     twin_of,
                                     lock_swap=bmeta.get("lock_swap"))
    if not lock_rings:
        n_adv = stepper.advance(-meta["shift"])   # the bias soak
        assert n_adv == bmeta["n_pre"], (n_adv, bmeta["n_pre"])
    return adapter


def build_adapter_alt(case, meta, names):
    """The mirror=0 path (case_e): C3Stepper on the file M_EE, folded
    with the plugin's expressions (symmetrize = exact no-op, then
    M * outer(d, d))."""
    fnames, m_ee, _ = read_matrix_blocks(case / "matrix.dat")
    assert fnames == names, "matrix/manifest name mismatch"
    doc = yaml.safe_load(pathlib.Path(meta["yaml"]).read_text())
    by_coil = {e["coil"]: e for e in doc["coils"] if e.get("driven_parts")}
    circuits, d = [], []
    for nm in names:
        ent = by_coil[nm[1:]]
        circuits.append((nm, ent["parameters"]))
        d.append(float(ent["part_turns"][0]))
    d = np.array(d)
    m_sym = 0.5 * (m_ee + m_ee.T)             # the plugin's fold pass
    mel = m_sym * np.outer(d, d)
    n_pre = int(round((-meta["shift"] - meta["t0"]) / meta["dt"]))
    t0_real = -meta["shift"] - n_pre * meta["dt"]
    c3.DT = meta["dt"]
    stepper = c3_stepper.C3Stepper(circuits, mel, t0=t0_real,
                                   dt=meta["dt"])
    adapter = c3src.C3CircuitAdapter(stepper, d, len(names), {})
    n_adv = stepper.advance(-meta["shift"])
    assert n_adv == n_pre, (n_adv, n_pre)
    return adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--t-end", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps")
    ap.add_argument("--lock-out")
    ap.add_argument("--arming-out")
    args = ap.parse_args()
    case = pathlib.Path(args.case)
    meta = json.loads((case / "meta.json").read_text())
    names, i_ref = load_manifest(case / "manifest.txt")

    if meta["mirror"]:
        adapter = build_adapter(case, meta, names, args.dt)
    else:
        adapter = build_adapter_alt(case, meta, names)

    ports, eps_rows = ([], np.zeros((0, 0)))
    if args.eps:
        ports, eps_rows = load_eps_table(args.eps, names)

    n_steps = int(round(args.t_end / args.dt))
    swap_step = None
    if getattr(adapter, "lock_phase", 1) == 0:
        swap_step = max(1, int(round(meta["shift"] / args.dt)))
    n_lock = len(meta["lock_rings"])
    lock_idx = list(range(adapter.n_east - n_lock, adapter.n_east))

    def lock_flux(k):
        if adapter.lock_phase == 0:
            return 0.0
        st = adapter.stepper
        return float(st.mel[k] @ st.x[st.illd])

    out = open(args.out, "w")
    out.write("# t_sim " + " ".join(names) + "\n")
    lock_out = open(args.lock_out, "w") if args.lock_out else None
    for i in range(n_steps):
        if swap_step is not None and i == swap_step:
            t_machine = swap_step * args.dt - meta["shift"]
            locked = adapter.arm_locks(t_machine,
                                       log=lambda m: print(m, flush=True))
            if args.arming_out:
                st_tot = {}
                for nm, p in locked.items():
                    st_tot[nm[3]] = st_tot.get(nm[3], 0.0) + p
                rec = dict(t_machine=t_machine,
                           locked={k: v.hex() for k, v in locked.items()},
                           stations={k: v.hex()
                                     for k, v in st_tot.items()})
                pathlib.Path(args.arming_out).write_text(
                    json.dumps(rec, indent=1))
            swap_step = None
        # per-step held eps (the coupler surface), then the single-pass
        # advance to the interval end -- adapter.step()'s mirror + slice
        if ports and i < eps_rows.shape[0]:
            for j, k in enumerate(ports):
                adapter.set_plasma_emf(k, eps_rows[i][j])
        for w, e in adapter.twin_of.items():
            adapter._eps[w] = adapter._eps[e]
        eps = (adapter._eps if adapter.lock_phase
               else adapter._eps[adapter._keep])
        adapter.stepper.advance((i + 1) * args.dt - meta["shift"], eps=eps)
        t1 = (i + 1) * args.dt
        scales = [adapter.channel_current(j) / i_ref[j]
                  for j in range(len(names))]
        out.write(t1.hex() + " " + " ".join(s.hex() for s in scales)
                  + "\n")
        if lock_out is not None:
            row = [t1.hex()]
            for k in lock_idx:
                row.append(float(adapter.channel_current(k)).hex())
                row.append(float(lock_flux(k)).hex())
            lock_out.write(" ".join(row) + "\n")
    out.close()
    if lock_out is not None:
        lock_out.close()
    print(f"reference done: {n_steps} steps -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
