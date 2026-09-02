#!/usr/bin/env python3
"""Validation for the divergence-preserving cylindrical remap.

Test fields are built as the *discrete* curl of a vector potential on the
staggered mesh, which makes them divergence-free to roundoff by
construction -- so any divergence measured afterwards is attributable to
the prolongation and nothing else.

Run: python3 test_grid_remap.py
"""

from __future__ import annotations

import numpy as np

from grid_remap import CylindricalMesh, divergence, prolong_b


def b_from_potential(mesh, a_theta):
    """Discrete curl of A_theta on the staggered mesh.

    ``a_theta`` is given on ``(r-node, z-node)``, shape ``(nr+1, nz+1)``.
    The result is divergence-free to roundoff by construction.
    """
    r_n = mesh.r_nodes
    b_r = -(a_theta[:, 1:] - a_theta[:, :-1]) / mesh.dz
    r_in = r_n[:-1][:, None]
    r_out = r_n[1:][:, None]
    b_z = 2.0 * (r_out * a_theta[1:, :] - r_in * a_theta[:-1, :]) / (
        r_out**2 - r_in**2
    )
    return b_r, b_z


def make_case(nr, nz, r_min, r_max, z_min, z_max, seed):
    mesh = CylindricalMesh(nr, nz, r_min, r_max, z_min, z_max)
    rng = np.random.default_rng(seed)
    r_n = mesh.r_nodes[:, None]
    z_n = (mesh.z_min + mesh.dz * np.arange(nz + 1))[None, :]
    # a smooth part plus a rough part, so the test is not only easy modes
    a = (
        0.5 * r_n**2 * np.exp(-((z_n / 0.6) ** 2))
        + 0.2 * r_n * np.sin(3.0 * z_n) * np.cos(5.0 * r_n)
        + 0.05 * rng.standard_normal((nr + 1, nz + 1)) * r_n
    )
    b_r, b_z = b_from_potential(mesh, a)
    b_t = rng.standard_normal((nr, nz)) * 0.3 + np.cos(2.0 * r_n[:-1])
    return mesh, b_r, b_z, b_t


def report(name, mesh, b_r, b_z, b_t, ratio):
    div_c = divergence(mesh, b_r, b_z)
    scale = max(np.abs(b_r).max(), np.abs(b_z).max())
    fine, br_f, bz_f, bt_f = prolong_b(mesh, b_r, b_z, b_t, ratio=ratio)
    div_f = divergence(fine, br_f, bz_f)

    # normalise div by (field scale / cell size) so meshes are comparable
    norm_c = scale / mesh.dr
    norm_f = scale / fine.dr
    print(f"\n--- {name}  (ratio {ratio}, {mesh.nr}x{mesh.nz} -> "
          f"{fine.nr}x{fine.nz}) ---")
    print(f"  coarse  max|div B| = {np.abs(div_c).max():.4e} T/m"
          f"   L2 = {np.sqrt((div_c**2).mean()):.4e} T/m"
          f"   (relative {np.abs(div_c).max()/norm_c:.3e})")
    print(f"  fine    max|div B| = {np.abs(div_f).max():.4e} T/m"
          f"   L2 = {np.sqrt((div_f**2).mean()):.4e} T/m"
          f"   (relative {np.abs(div_f).max()/norm_f:.3e})")

    # naive component-wise prolongation, for contrast
    br_n = np.repeat(b_r[:-1, :], ratio, axis=0)
    br_n = np.vstack([br_n, b_r[-1:, :]])
    bz_n = np.repeat(b_z, ratio, axis=0)
    div_n = divergence(fine, br_n, bz_n)
    print(f"  naive   max|div B| = {np.abs(div_n).max():.4e} T/m"
          f"   L2 = {np.sqrt((div_n**2).mean()):.4e} T/m"
          f"   <-- component-wise, for contrast")

    # flux conservation through the outer boundary and total |B| energy
    r_n_c, r_n_f = mesh.r_nodes, fine.r_nodes
    flux_c = 2 * np.pi * r_n_c[-1] * mesh.dz * b_r[-1, :].sum()
    flux_f = 2 * np.pi * r_n_f[-1] * fine.dz * br_f[-1, :].sum()
    print(f"  outer radial flux: coarse {flux_c:.10e}  fine {flux_f:.10e}"
          f"   rel diff {abs(flux_f-flux_c)/max(abs(flux_c),1e-30):.2e}")

    vol_c = np.pi * (r_n_c[1:]**2 - r_n_c[:-1]**2)[:, None] * mesh.dz
    vol_f = np.pi * (r_n_f[1:]**2 - r_n_f[:-1]**2)[:, None] * fine.dz
    bz_cc_c = 0.5 * (b_z[:, 1:] + b_z[:, :-1])
    bz_cc_f = 0.5 * (bz_f[:, 1:] + bz_f[:, :-1])
    e_c = (bz_cc_c**2 * vol_c).sum()
    e_f = (bz_cc_f**2 * vol_f).sum()
    print(f"  axial field energy (arb): coarse {e_c:.8e}  fine {e_f:.8e}"
          f"   rel diff {abs(e_f-e_c)/max(abs(e_c),1e-30):.2e}")
    return np.abs(div_f).max(), np.abs(div_c).max(), np.abs(div_n).max()


def main():
    cases = [
        ("off-axis annulus", dict(nr=32, nz=48, r_min=0.05, r_max=0.25,
                                  z_min=-1.0, z_max=1.0, seed=1)),
        ("axis-touching", dict(nr=48, nz=64, r_min=0.0, r_max=0.20,
                               z_min=-2.0, z_max=2.0, seed=2)),
        ("production-like aspect", dict(nr=256, nz=1024, r_min=0.0,
                                        r_max=0.20, z_min=-2.0, z_max=2.0,
                                        seed=3)),
    ]
    worst = 0.0
    for name, kw in cases:
        mesh, b_r, b_z, b_t = make_case(**kw)
        for ratio in (1, 2, 4):
            d_f, d_c, d_n = report(name, mesh, b_r, b_z, b_t, ratio)
            # the prolonged divergence must not exceed the coarse one by
            # more than roundoff growth
            tol = max(10.0 * d_c, 1e-8 * np.abs(b_z).max() / mesh.dr)
            status = "PASS" if d_f <= tol else "FAIL"
            gain = d_n / max(d_f, 1e-300)
            print(f"  => {status}   (tol {tol:.3e};"
                  f" naive is {gain:.3e}x worse)")
            if status == "FAIL":
                worst = max(worst, d_f)
    print("\nALL CASES PASS" if worst == 0.0 else "\nSOME CASES FAILED")


if __name__ == "__main__":
    main()
