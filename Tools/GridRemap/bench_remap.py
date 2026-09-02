#!/usr/bin/env python3
"""Cost and memory of a remap at production scale.

Answers the practical question: can the extract/interpolate step be run
on one node for a large state, or does it need streaming?
"""

from __future__ import annotations

import resource
import time

import numpy as np

from grid_remap import CylindricalMesh, divergence, prolong_b


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def bench_fields(nr, nz, ratio):
    mesh = CylindricalMesh(nr, nz, 0.0, 0.2, -2.0, 2.0)
    rng = np.random.default_rng(0)
    b_r = rng.standard_normal((nr + 1, nz))
    b_z = rng.standard_normal((nr, nz + 1))
    b_t = rng.standard_normal((nr, nz))
    t0 = time.perf_counter()
    fine, br_f, bz_f, bt_f = prolong_b(mesh, b_r, b_z, b_t, ratio=ratio)
    t1 = time.perf_counter()
    _ = divergence(fine, br_f, bz_f)
    t2 = time.perf_counter()
    nbytes = (br_f.nbytes + bz_f.nbytes + bt_f.nbytes) / 1024**3
    print(f"  fields {nr}x{nz} ratio {ratio}: prolong {t1-t0:.2f} s, "
          f"div check {t2-t1:.2f} s, fine arrays {nbytes:.3f} GB")


def bench_particles(npart):
    """A remap carries 7 doubles per particle (x,y,z,ux,uy,uz,w)."""
    per_particle = 7 * 8
    gb = npart * per_particle / 1024**3
    chunk = 8_000_000
    t0 = time.perf_counter()
    acc_w = 0.0
    rng = np.random.default_rng(1)
    done = 0
    while done < min(npart, 32_000_000):
        n = min(chunk, npart - done)
        block = rng.standard_normal((n,))
        acc_w += float(block.sum())
        done += n
    t1 = time.perf_counter()
    rate = done / max(t1 - t0, 1e-9)
    print(f"  particles: {npart/1e6:.0f}M x 7 doubles = {gb:.2f} GB resident")
    print(f"    streaming touch rate {rate/1e6:.1f} M/s -> "
          f"{npart/rate:.1f} s for one pass over {npart/1e6:.0f}M")
    print(f"    a serial single-rank read of the full list needs "
          f"~{2*gb:.1f} GB (openPMD chunk + host vectors)")


def main():
    print("Field prolongation cost:")
    for nr, nz, ratio in ((256, 1024, 2), (256, 1024, 4), (512, 2048, 2)):
        bench_fields(nr, nz, ratio)
    print("\nParticle carry cost:")
    for npart in (32_000_000, 128_000_000):
        bench_particles(npart)
    print(f"\npeak RSS of this benchmark: {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
