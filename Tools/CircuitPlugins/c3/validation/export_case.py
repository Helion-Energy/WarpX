#!/usr/bin/env python3
"""Export the validation cases for the c3 external-circuit plugin.

Each case directory holds bit-identical inputs for BOTH sides of the
parity gate -- the C++ plugin (matrix.dat hexfloat blocks + config.ini)
and the Python reference (mcomb.npy, the raw deck-probe output that
c3_bank_source.build() consumes through a stub probe_fn):

  case_a/  the reference shot FULL mirror machine, no rings (the --bank-source c3
           default; also drives case b, the eps-driven arm)
  case_c/  case_a + the deck's --lock-rings flux-lock stations (two-stage
           machine-0 swap; also drives case d, the checkpoint round trip)
  case_e/  alternate-case bank circuits on the cached c2 M_ff (mirror=0):
           covers the reverse-recovery companion, segment 3/5 devices and
           the reversed formation banks that the reference shot never exercises

The matrix file carries the amp-turn M_EE/M_EW blocks exactly as
c3_bank_source.build() derives them (post symmetrization, post
ring-kernel overrides); the exporter cross-checks its replica of that
matrix section against the real build() output bit for bit.

Run (deck venv, no WarpX boot):
    HETOOLS_USE_GPU=0 ~/.env/warpx-mhd/bin/python3 export_case.py \
        --workdir /path/to/cases
"""

import argparse
import json
import os
import pathlib
import sys
import types

os.environ.setdefault("HETOOLS_USE_GPU", "0")

import numpy as np
import yaml

DECK = (pathlib.Path.home() / "src/hetools/Examples/warpx/Profiles/"
        "production/formation/RZ-mhd-formation")
GREENS = pathlib.Path.home() / "src/hetools-greens/Machines/<machine>"
sys.path.insert(0, str(DECK))
sys.path.insert(0, str(GREENS))

import c3_bank_source as c3src  # noqa: E402
import coil_set  # noqa: E402
import PICMI_inputs_RZ as deckmod  # noqa: E402

# case-a/c clock conventions (c3_openloop_ref_shot / the deck defaults)
DT_CIRC = 5.0e-8
T0_MACHINE = -200.0e-6
BANK_T_SHIFT = 2.5e-6
# case-e clock conventions (c3_ref_circuit deck globals)
T0_ALT = -170.0e-6


def hexfloat(v):
    return float(v).hex()


def write_matrix(path, names, m_ee, m_ew):
    with open(path, "w") as f:
        f.write("c3-mel-blocks v1\n")
        f.write(f"n {len(names)}\n")
        for nm in names:
            f.write(nm + "\n")
        for tag, m in (("M_EE", m_ee), ("M_EW", m_ew)):
            f.write(tag + "\n")
            for row in np.asarray(m):
                f.write(" ".join(hexfloat(v) for v in row) + "\n")


def build_ref_shot_case(out, lock=False):
    """Export case_a (lock=False) or case_c (lock=True)."""
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = DECK / "ref_shot_circuit.yaml"
    doc = yaml.safe_load(yaml_path.read_text())
    by_coil = {e["coil"]: e for e in doc["coils"]}

    # grid shim: the M probe only needs the class grid constants
    cls = deckmod.FormationMHD
    deck = types.SimpleNamespace(R_DOM=cls.R_DOM, NR=cls.NR,
                                 Z_MIN=cls.Z_MIN, Z_MAX=cls.Z_MAX,
                                 NZ=cls.NZ)
    dr = deck.R_DOM / deck.NR
    dz = (deck.Z_MAX - deck.Z_MIN) / deck.NZ

    deck_names = sorted(coil_set.COILS)
    coils_rz = [(coil_set.COILS[nm]["Rc"] + 0.25 * dr,
                 coil_set.COILS[nm]["zc"] + 0.25 * dz)
                for nm in deck_names]
    i_ref = [coil_set.COILS[nm]["I"] for nm in deck_names]

    # ---- replicate build()'s filament list: coils + EXTRA + rings ----
    names_e = [nm.replace("coil_", "") for nm in deck_names] + \
        list(c3src.EXTRA)
    full_rz = list(coils_rz)
    for cn in c3src.EXTRA:
        ent = by_coil[cn]
        full_rz.append(
            (0.5 * (ent["r_extent_m"][0] + ent["r_extent_m"][1]) + 0.25 * dr,
             0.5 * (ent["z_extent_m"][0] + ent["z_extent_m"][1]) + 0.25 * dz))
    n_c = len(names_e)
    rings = deckmod._lock_ring_specs(dr, dz) if lock else []
    for rg in rings:
        full_rz.append(tuple(rg["rz"]))
        names_e.append(rg["name"])
    n = len(full_rz)

    # ---- the deck probe (the exact call build() makes) ----
    comb = full_rz + [(r, -z) for (r, z) in full_rz]
    m_comb, _ = deckmod.FormationMHD._bank_discrete_inductance_matrices(
        deck, comb, [1.0] * (2 * n))

    # ---- replicate build()'s matrix section (cross-checked below) ----
    m_ee = 0.5 * (m_comb[:n, :n] + m_comb[:n, :n].T)
    m_ew = 0.5 * (m_comb[:n, n:] + m_comb[:n, n:].T)
    segs = [rg.get("seg") for rg in rings]
    if rings and all(s is not None for s in segs):
        for i in range(len(rings)):
            for j in range(i, len(rings)):
                m_ij = c3src.ref_seg_mutual(segs[i], segs[j])
                m_ee[n_c + i, n_c + j] = m_ij
                m_ee[n_c + j, n_c + i] = m_ij
            for j in range(len(rings)):
                sjm = (segs[j][0], -segs[j][1], segs[j][2], -segs[j][3])
                m_ew[n_c + i, n_c + j] = c3src.ref_seg_mutual(
                    segs[i], sjm)
    for j, rg in enumerate(rings):
        if rg.get("L_self") is not None:
            m_ee[n_c + j, n_c + j] = float(rg["L_self"])

    # ---- bit-exact cross-check against the REAL build() ----
    def probe_stub(rz, cur):
        assert len(rz) == 2 * n and len(cur) == 2 * n, (len(rz), n)
        return m_comb, None

    _, _, _, meta = c3src.build(
        str(yaml_path), deck_names, coils_rz, probe_stub, dr, dz,
        DT_CIRC, T0_MACHINE, BANK_T_SHIFT, log=lambda *a: None,
        lock_rings=(rings if lock else None), pre_roll=False)
    assert np.array_equal(meta["M_EE"], m_ee), \
        "exporter replica of build()'s M_EE is not bit-identical"

    # ---- write the case ----
    write_matrix(out / "matrix.dat", names_e, m_ee, m_ew)
    np.save(out / "mcomb.npy", m_comb)
    with open(out / "manifest.txt", "w") as f:
        f.write("# name i_ref\n")
        for nm, ir in zip(deck_names, i_ref):
            f.write(f"{nm} {hexfloat(ir)}\n")
    if lock:
        with open(out / "rings.txt", "w") as f:
            f.write("# name R_ohm lock\n")
            for rg in rings:
                f.write(f"{rg['name']} {rg['R_ohm']} 1\n")
    cfg = {
        "yaml": str(yaml_path), "matrix": str(out / "matrix.dat"),
        "dt": DT_CIRC, "t0": T0_MACHINE, "shift": BANK_T_SHIFT,
        "extra": "+".join(c3src.EXTRA), "mirror": 1,
    }
    if lock:
        cfg["rings"] = str(out / "rings.txt")
    with open(out / "config.ini", "w") as f:
        for k, v in cfg.items():
            f.write(f"{k} = {v}\n")
    meta_out = {
        "yaml": str(yaml_path), "dt": DT_CIRC, "t0": T0_MACHINE,
        "shift": BANK_T_SHIFT, "extra": list(c3src.EXTRA), "mirror": True,
        "dr": dr, "dz": dz, "names": deck_names,
        "lock_rings": [{k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in rg.items()} for rg in rings],
    }
    (out / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print(f"exported {out.name}: {n} east channels "
          f"({len(rings)} lock rings), M blocks {m_ee.shape}")


def build_alt_case(out):
    """Export case_e: the alternate-case bank circuits on the cached c2 M_ff."""
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = GREENS / "circuits" / "alt_case_circuit.yaml"
    cache = GREENS / "c2_alt_case" / "c2_data.npz"
    d = np.load(cache, allow_pickle=True)
    pack_names = [str(s) for s in d["pack_names"]]
    m_ff = d["m_ff"]
    doc = yaml.safe_load(yaml_path.read_text())
    by_coil = {e["coil"]: e for e in doc["coils"]
               if e.get("driven_parts")}
    fixed = {e["coil"] for e in doc["fixed_circuits"]}
    dyn_idx, names, i_ref = [], [], []
    for i, name in enumerate(pack_names):
        coil = name[1:]
        if coil in fixed:
            continue
        ent = by_coil[coil]
        dyn_idx.append(i)
        names.append(name)
        # the plugin folds d = part_turns[0] raw; load_alt folds
        # 1/round(1/pt) -- assert the two are bit-identical here
        pt = float(ent["part_turns"][0])
        assert pt == 1.0 / round(1.0 / pt), (name, pt)
        i_ref.append(1.0e5)   # scale normalization only (no deck I_ref)
    dyn_idx = np.array(dyn_idx)
    m_block = m_ff[np.ix_(dyn_idx, dyn_idx)]
    m_ee = 0.5 * (m_block + m_block.T)   # plugin re-symmetrizes: exact no-op
    m_ew = np.zeros_like(m_ee)           # mirror=0: block unused
    write_matrix(out / "matrix.dat", names, m_ee, m_ew)
    with open(out / "manifest.txt", "w") as f:
        f.write("# name i_ref\n")
        for nm, ir in zip(names, i_ref):
            f.write(f"{nm} {hexfloat(ir)}\n")
    cfg = {
        "yaml": str(yaml_path), "matrix": str(out / "matrix.dat"),
        "dt": DT_CIRC, "t0": T0_ALT, "shift": 0.0,
        "extra": "", "mirror": 0,
    }
    with open(out / "config.ini", "w") as f:
        for k, v in cfg.items():
            f.write(f"{k} = {v}\n")
    meta_out = {
        "yaml": str(yaml_path), "dt": DT_CIRC, "t0": T0_ALT,
        "shift": 0.0, "extra": [], "mirror": False, "names": names,
        "lock_rings": [],
    }
    (out / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print(f"exported {out.name}: {len(names)} circuits from the c2 cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cases", default="a,c,e")
    args = ap.parse_args()
    wd = pathlib.Path(args.workdir)
    cases = set(args.cases.split(","))
    if "a" in cases:
        build_ref_shot_case(wd / "case_a", lock=False)
    if "c" in cases:
        build_ref_shot_case(wd / "case_c", lock=True)
    if "e" in cases:
        build_alt_case(wd / "case_e")


if __name__ == "__main__":
    main()
