#!/usr/bin/env python3
"""Particle-count matching for resolution studies.

Refining a mesh at fixed particle count lowers the particles-per-cell,
so a bare coarse-vs-fine comparison confounds resolution with sampling
noise.  This module provides the weight-preserving decimation used to
build the matched-statistics arm.

The trap worth stating explicitly: SPLITTING each macro-particle into
several coincident copies restores the particles-per-cell *count* but
does NOT restore the statistics.  Copies at identical positions carry
identical deposition error, so the noise in every moment is unchanged --
the arm looks better resolved and samples exactly as badly.  Only
decimation (throwing samples away) moves noise in a direction you can
actually reason about, which is why the matched arm is built by
*removing* particles from the coarse run rather than adding them to the
fine one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decimate"]


def decimate(arrays, weights, keep_fraction, seed):
    """Weight-preserving random decimation.

    Keeps ``keep_fraction`` of the particles, chosen without replacement
    with a stated seed, and scales the surviving weights by
    ``1 / keep_fraction`` so total weight -- and hence every weighted
    moment, in expectation -- is preserved.  Sampling noise grows as
    ``1 / sqrt(keep_fraction)``.

    Parameters
    ----------
    arrays : sequence of ndarray
        Per-particle quantities to carry through, each shape ``(n,)``.
    weights : ndarray
        Per-particle weights, shape ``(n,)``.
    keep_fraction : float
        In ``(0, 1]``.
    seed : int
        Explicit, so the arm is reproducible; record it with the run.

    Returns
    -------
    (list of ndarray, ndarray)
        Decimated arrays and rescaled weights.
    """
    if not 0.0 < keep_fraction <= 1.0:
        msg = "keep_fraction must lie in (0, 1]"
        raise ValueError(msg)
    n = int(weights.shape[0])
    for a in arrays:
        if a.shape[0] != n:
            msg = "all arrays must have the same length as weights"
            raise ValueError(msg)
    if keep_fraction == 1.0:
        return [np.array(a, copy=True) for a in arrays], np.array(weights, copy=True)

    rng = np.random.default_rng(seed)
    n_keep = int(round(n * keep_fraction))
    idx = rng.choice(n, size=n_keep, replace=False)
    idx.sort()
    scale = n / float(n_keep)      # exact weight preservation in expectation
    return [a[idx] for a in arrays], weights[idx] * scale


def _selftest():
    rng = np.random.default_rng(7)
    n = 2_000_000
    w = rng.random(n) + 0.5
    x = rng.standard_normal(n)
    v = rng.standard_normal(n) * 3.0

    for frac in (1.0, 0.5, 0.25):
        (xd, vd), wd = decimate([x, v], w, frac, seed=11)
        tot = wd.sum() / w.sum()
        mean_v = (wd * vd).sum() / wd.sum()
        mean_v0 = (w * v).sum() / w.sum()
        energy = (wd * vd**2).sum() / (w * v**2).sum()
        print(f"  keep {frac:4.2f}: n {len(wd):>9d}  "
              f"weight ratio {tot:.6f}  "
              f"<v> {mean_v0:+.5f} -> {mean_v:+.5f}  "
              f"energy ratio {energy:.6f}  "
              f"expected noise x{1/np.sqrt(frac):.3f}")


if __name__ == "__main__":
    print("weight-preserving decimation self-test:")
    _selftest()
