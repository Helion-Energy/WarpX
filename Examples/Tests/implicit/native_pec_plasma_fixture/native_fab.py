import re
from pathlib import Path

import numpy as np

box_re = re.compile(r"\(\(([-\d, ]+)\) \(([-\d, ]+)\) \(([-\d, ]+)\)\)")


def box(text):
    m = box_re.search(text)
    assert m, text
    return tuple(tuple(int(v) for v in group.split(",")) for group in m.groups())


def read(prefix):
    prefix = Path(prefix)
    lines = prefix.with_name(prefix.name + "_H").read_text().splitlines()
    count = int(lines[2])
    valid = [box(x) for x in lines if box_re.fullmatch(x)]
    parts = []
    for line in lines:
        if not line.startswith("FabOnDisk:"):
            continue
        _, file, offset = line.split()
        idx = len(parts)
        with (prefix.parent / file).open("rb") as f:
            f.seek(int(offset))
            header = f.readline().decode()
            bounds = box(header)
            assert int(header.split()[-1]) == count
            lo, hi, st = bounds
            shape = tuple(b - a + 1 for a, b in zip(lo, hi)) + (count,)
            a = np.fromfile(f, "<f8", int(np.prod(shape))).reshape(shape, order="F")
        parts.append(
            {"valid": valid[idx], "lo": lo, "hi": hi, "stagger": st, "array": a}
        )
    assert len(parts) == len(valid)
    return parts


def values(prefix, valid=True, component=0):
    out = {}
    for f, p in enumerate(read(prefix)):
        lo, hi, _ = p["valid"] if valid else (p["lo"], p["hi"], p["stagger"])
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                out[f, i, j] = float(
                    p["array"][i - p["lo"][0], j - p["lo"][1], component]
                )
    return out


def name(root, field, c=None):
    return Path(root) / (
        field
        + (
            f"[dir={chr(114) if c == 0 else 'theta' if c == 1 else 'z'}]"
            if c is not None
            else ""
        )
        + "[level=0]"
    )
