# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Read the native double-precision fixture snapshots without a plotfile regrid."""
import hashlib
from pathlib import Path
import re
import struct
import numpy as np

BOX = re.compile(r"^\(\(([-0-9,]+)\) \(([-0-9,]+)\) \(([-0-9,]+)\)\)$")
FAB = re.compile(rb"\(\(([-0-9,]+)\) \(([-0-9,]+)\) \(([-0-9,]+)\)\) (\d+)\s*$")


def ints(v):
    return tuple(map(int, v.split(b"," if isinstance(v, bytes) else ",")))


def mf(path):
    h = Path(str(path) + "_H").read_text().splitlines()
    boxes = [tuple(ints(v) for v in m.groups()) for x in h if (m := BOX.match(x))]
    files = [x.split()[1:] for x in h if x.startswith("FabOnDisk:")]
    assert len(boxes) == len(files)
    result = {}
    for box, (name, offset) in zip(boxes, files):
        with (path.parent / name).open("rb") as f:
            f.seek(int(offset))
            line = f.readline()
            m = FAB.search(line)
            assert m, line
            lo, hi, typ = map(ints, m.groups()[:3])
            nc = int(m.group(4))
            nx = hi[0] - lo[0] + 1
            nz = hi[1] - lo[1] + 1
            data = f.read(nc * nx * nz * 8)
            assert len(data) == nc * nx * nz * 8
        result[box] = {
            "fab_lo": lo,
            "fab_hi": hi,
            "data": data,
            "array": np.frombuffer(data, dtype="<f8").reshape(nc, nz, nx),
        }
    return result


def fields_compare(a, b):
    names = (a / "field_names.txt").read_text().splitlines()
    assert names == (b / "field_names.txt").read_text().splitlines()
    out = []
    for name in names:
        x = mf(a / "fields" / name)
        y = mf(b / "fields" / name)
        assert x.keys() == y.keys()
        different = []
        for box in x:
            xx, yy = x[box], y[box]
            assert xx["fab_lo"] == yy["fab_lo"] and xx["fab_hi"] == yy["fab_hi"]
            if xx["data"] != yy["data"]:
                diff = xx["array"] - yy["array"]
                mask = diff != 0
                different.append(
                    {
                        "box": box,
                        "n_diff": int(mask.sum()),
                        "max_abs": float(np.nanmax(abs(diff))),
                    }
                )
        out.append(
            {
                "field": name,
                "exact_all_fabs_with_ghosts": not different,
                "different_fabs": different,
                "finite_all_fabs": bool(
                    all(
                        np.isfinite(f["array"]).all()
                        for f in [*x.values(), *y.values()]
                    )
                ),
            }
        )
    return out


def particles(root, prefix):
    ids = []
    rr = []
    ii = []
    for path in sorted(root.glob(prefix + "_rank*.bin")):
        with path.open("rb") as f:
            nr, ni, size, nt = struct.unpack("<4Q", f.read(32))
            assert size == 8
            for _ in range(nt):
                grid, tile, n = struct.unpack("<3q", f.read(24))
                ids.append(np.frombuffer(f.read(n * 8), dtype="<u8"))
                rr.append(np.frombuffer(f.read(n * nr * 8), dtype="<f8").reshape(nr, n))
                ii.append(np.frombuffer(f.read(n * ni * 4), dtype="<i4").reshape(ni, n))
            assert not f.read(1)
    ids = np.concatenate(ids)
    real = np.concatenate(rr, axis=1)
    integer = np.concatenate(ii, axis=1)
    order = np.argsort(ids)
    ids = ids[order]
    real = real[:, order]
    integer = integer[:, order]
    assert len(np.unique(ids)) == len(ids)
    return ids, real, integer


def particle_compare(a, b, prefix):
    aa = particles(a, prefix)
    bb = particles(b, prefix)
    return {
        "kind": prefix,
        "counts": [len(x[0]) for x in [aa, bb]],
        "component_counts": [aa[1].shape[0], aa[2].shape[0]],
        "exact_ids": aa[0].tobytes() == bb[0].tobytes(),
        "exact_all_real": aa[1].tobytes() == bb[1].tobytes(),
        "exact_all_int": aa[2].tobytes() == bb[2].tobytes(),
        "canonical_sha256": [
            hashlib.sha256(b"".join(x.tobytes() for x in data)).hexdigest()
            for data in [aa, bb]
        ],
        "finite": bool(np.isfinite(aa[1]).all() and np.isfinite(bb[1]).all()),
    }
