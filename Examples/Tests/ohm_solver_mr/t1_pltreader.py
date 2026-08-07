#!/usr/bin/env python3
"""Minimal AMReX plotfile reader and T1 run-to-npz converter.

Purpose-built for the t1_whistler.py decks (2D, double precision, native
little-endian, cell-centered plotfile fields). yt handles these files fine
but is ~30x too slow for the ~1000 per-step plotfiles a T1 run produces;
the FieldProbe reduced diagnostic cannot be used instead because it drops
all level-0 points when amr.max_level > 0.

convert_run() reduces every plotfile of a run to x-averaged complex-field
lineouts per level plus per-step artifact monitors, stored in a single
compressed npz:
    t        (NT,)        output times (s)
    z0       (nz0,)       level-0 cell centers (m)
    bx0, by0 (NT, nz0)    x-averaged Bx, By on level 0 (float32)
    amax0    (NT, 4)      max over level 0 of |Ex|, |Ey|, |Ez|, |Bz - B0|
    and, when a level 1 exists:
    z1, bx1, by1, amax1   same for the level-1 (patch) region
    patch_ilo, patch_ihi  level-1 z-index range (fine cells)
"""

import glob
import os
import re
import sys

import numpy as np

_BOX_RE = re.compile(r"\(\((-?\d+),(-?\d+)\) \((-?\d+),(-?\d+)\) \(-?\d+,-?\d+\)\)")
_FAB_RE = re.compile(r"FabOnDisk: (\S+) (\d+)")


def _read_cell_header(cell_h_path):
    """Parse Level_*/Cell_H: returns (boxes, fabs, nvar) where boxes is a
    list of ((ilo,jlo),(ihi,jhi)) and fabs a list of (filename, offset)."""
    with open(cell_h_path) as f:
        txt = f.read()
    nvar = int(txt.splitlines()[2])
    fabs = [(m.group(1), int(m.group(2))) for m in _FAB_RE.finditer(txt)]
    boxes = []
    for m in _BOX_RE.finditer(txt):
        ilo, jlo, ihi, jhi = (int(m.group(i)) for i in range(1, 5))
        boxes.append(((ilo, jlo), (ihi, jhi)))
        if len(boxes) == len(fabs):
            break  # ignore any box-shaped noise in the min/max tail
    assert len(boxes) == len(fabs), f"{cell_h_path}: {len(boxes)} vs {len(fabs)}"
    return boxes, fabs, nvar


def read_plotfile(pltdir, fields):
    """Read the requested cell-centered fields of a 2D plotfile.

    Returns (time, levels) with levels a list of dicts:
      {"lo": (ilo, jlo), "arr": {field: 2D array indexed [j (z), i (x)]}}
    where lo is the level-wide index of arr[0, 0].
    """
    with open(os.path.join(pltdir, "Header")) as f:
        lines = f.read().splitlines()
    nf = int(lines[1])
    names = lines[2 : 2 + nf]
    ptr = 2 + nf
    assert int(lines[ptr]) == 2, "reader is 2D-only"
    time = float(lines[ptr + 1])
    finest = int(lines[ptr + 2])
    fidx = {name: i for i, name in enumerate(names)}
    for f_ in fields:
        assert f_ in fidx, f"{pltdir}: field {f_} not in plotfile"

    levels = []
    for lev in range(finest + 1):
        boxes, fabs, nvar = _read_cell_header(
            os.path.join(pltdir, f"Level_{lev}", "Cell_H")
        )
        ilo = min(b[0][0] for b in boxes)
        jlo = min(b[0][1] for b in boxes)
        ihi = max(b[1][0] for b in boxes)
        jhi = max(b[1][1] for b in boxes)
        nx, nz = ihi - ilo + 1, jhi - jlo + 1
        arr = {f_: np.empty((nz, nx)) for f_ in fields}
        handles = {}
        try:
            for (blo, bhi), (fname, off) in zip(boxes, fabs):
                if fname not in handles:
                    handles[fname] = open(
                        os.path.join(pltdir, f"Level_{lev}", fname), "rb"
                    )
                fh = handles[fname]
                fh.seek(off)
                hdr = fh.readline().decode()  # FAB ((8, (...)),(..)) box nvar
                assert hdr.startswith("FAB ((8,"), f"unexpected FAB: {hdr[:40]}"
                bnx = bhi[0] - blo[0] + 1
                bnz = bhi[1] - blo[1] + 1
                raw = np.frombuffer(fh.read(8 * bnx * bnz * nvar), dtype="<f8")
                fab = raw.reshape(nvar, bnz, bnx)  # x fastest, then z
                sl = np.s_[
                    blo[1] - jlo : bhi[1] - jlo + 1, blo[0] - ilo : bhi[0] - ilo + 1
                ]
                for f_ in fields:
                    arr[f_][sl] = fab[fidx[f_]]
        finally:
            for fh in handles.values():
                fh.close()
        levels.append({"lo": (ilo, jlo), "arr": arr})
    return time, levels


FIELDS = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]


def convert_run(run_dir, b0=0.25, out_name="t1_lineouts.npz"):
    """Reduce all diags/diag1?????? plotfiles of a run to the npz described
    in the module docstring. Returns the npz path."""
    plts = sorted(glob.glob(os.path.join(run_dir, "diags", "diag1??????")))
    assert plts, f"no plotfiles under {run_dir}"
    t_list = []
    lev_data = [dict(bx=[], by=[], amax=[]) for _ in range(2)]
    nlev = None
    patch_lo = patch_hi = None
    dz = [None, None]
    zlo_dom = None
    for p in plts:
        time, levels = read_plotfile(p, FIELDS)
        if nlev is None:
            nlev = len(levels)
            with open(os.path.join(p, "Header")) as f:
                lines = f.read().splitlines()
            nf = int(lines[1])
            zlo_dom = float(lines[2 + nf + 3].split()[1])
            # per-level dz: lines after coord data; recompute from prob size
            # via the level boxes instead (robust): dz = Lz_dom / nz_lev0
            zhi_dom = float(lines[2 + nf + 4].split()[1])
            nz0 = levels[0]["arr"]["Bx"].shape[0]
            dz[0] = (zhi_dom - zlo_dom) / nz0
            if nlev > 1:
                dz[1] = 0.5 * dz[0]
                patch_lo = levels[1]["lo"][1]
                patch_hi = patch_lo + levels[1]["arr"]["Bx"].shape[0]
        t_list.append(time)
        for lev, ld in enumerate(levels):
            a = ld["arr"]
            lev_data[lev]["bx"].append(a["Bx"].mean(axis=1))
            lev_data[lev]["by"].append(a["By"].mean(axis=1))
            lev_data[lev]["amax"].append(
                [
                    np.abs(a["Ex"]).max(),
                    np.abs(a["Ey"]).max(),
                    np.abs(a["Ez"]).max(),
                    np.abs(a["Bz"] - b0).max(),
                ]
            )

    nz0 = len(lev_data[0]["bx"][0])
    out = dict(
        t=np.array(t_list),
        z0=zlo_dom + (np.arange(nz0) + 0.5) * dz[0],
        bx0=np.array(lev_data[0]["bx"], dtype=np.float32),
        by0=np.array(lev_data[0]["by"], dtype=np.float32),
        amax0=np.array(lev_data[0]["amax"], dtype=np.float32),
    )
    if nlev > 1:
        nz1 = len(lev_data[1]["bx"][0])
        out.update(
            z1=zlo_dom + (patch_lo + np.arange(nz1) + 0.5) * dz[1],
            bx1=np.array(lev_data[1]["bx"], dtype=np.float32),
            by1=np.array(lev_data[1]["by"], dtype=np.float32),
            amax1=np.array(lev_data[1]["amax"], dtype=np.float32),
            patch_ilo=patch_lo,
            patch_ihi=patch_hi,
        )
    npz_path = os.path.join(run_dir, out_name)
    np.savez_compressed(npz_path, **out)
    return npz_path


if __name__ == "__main__":
    print(convert_run(sys.argv[1]))
