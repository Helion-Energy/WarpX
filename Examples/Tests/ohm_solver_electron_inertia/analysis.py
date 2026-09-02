#!/usr/bin/env python3
"""Whistler dispersion check for the elliptic electron-inertia term.

Reads the field-probe line, forms the right-hand circular combination
B+ = Bx + i By, and takes a 2D FFT over (z, t). For each seeded box mode the
peak positive frequency is compared with

    omega = Omega_ce (k_d d_e)^2 / (1 + (k_d d_e)^2),

where k_d = (2/dz) sin(k dz/2) is the wavenumber of the Yee curl pair -- the
one the discrete operator actually sees, which differs from k by a factor of
1.6 at the top mode here.

The inertialess Ohm's law would give omega = Omega_ce (k_d d_e)^2 instead:
2.0 rather than 0.667 Omega_ce at mode 16, and 3.96 rather than 0.798 at mode
30. The tolerance below is far tighter than that gap, so this fails loudly if
the term is dropped, and equally if it is applied twice or with the wrong
coefficient.
"""

import numpy as np

# Deck parameters (keep in sync with the inputs file).
QE = 1.602176634e-19
ME = 9.1093837015e-31
B0 = 0.25
DZ = 6.852041e-06
DE = 6.852041e-06  # electron skin depth, one cell by construction
DT = 7.144774e-12
NZ = 64
SEED_MODES = (8, 16, 24, 30)
RTOL = 0.12

WCE = QE * B0 / ME

d = np.loadtxt("diags/reducedfiles/probe.txt", skiprows=1)
steps = d[:, 0].astype(np.int64)
n_out = len(np.unique(steps))
npts = len(d) // n_out
bx = d[: n_out * npts, 8].reshape(n_out, npts)
by = d[: n_out * npts, 9].reshape(n_out, npts)
assert npts == NZ, f"probe returned {npts} points, expected {NZ}"

# Right-hand branch: B+ = Bx + i By puts the R mode at positive omega for +k.
bp = bx + 1j * by
bp -= bp.mean(axis=0, keepdims=True)
spec = np.abs(np.fft.fft(np.fft.fft(bp * np.hanning(n_out)[:, None], axis=0),
                         axis=1)) ** 2
omega = np.fft.fftfreq(n_out, d=DT) * 2.0 * np.pi
d_omega = omega[1] - omega[0]

print(f"{n_out} time samples, d(omega) = {d_omega / WCE:.4f} Omega_ce")
print(f"{'mode':>5} {'k*d_e':>7} {'measured':>10} {'expected':>10} {'rel err':>9}")

worst = 0.0
for n in SEED_MODES:
    k = 2.0 * np.pi * n / (NZ * DZ)
    kd_de = 2.0 / DZ * np.sin(k * DZ / 2.0) * DE
    expected = WCE * kd_de**2 / (1.0 + kd_de**2)

    column = spec[:, n]
    positive = np.where(omega > 0.05 * WCE)[0]
    j = positive[np.argmax(column[positive])]
    # Parabolic refinement of the peak in log power.
    y0, y1, y2 = np.log(column[j - 1: j + 2] + 1e-300)
    denom = y0 - 2.0 * y1 + y2
    measured = omega[j] + (0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0) * d_omega

    err = abs(measured - expected) / expected
    worst = max(worst, err)
    print(f"{n:5d} {k * DE:7.3f} {measured / WCE:10.4f} {expected / WCE:10.4f} "
          f"{err:9.4f}")

    assert err < RTOL, (
        f"mode {n}: measured omega = {measured / WCE:.4f} Omega_ce but the "
        f"inertial dispersion relation gives {expected / WCE:.4f} "
        f"(relative error {err:.4f} > {RTOL}). The inertialess value would be "
        f"{kd_de**2:.4f} Omega_ce."
    )

# The whole point of the term: the highest mode the grid carries must not
# oscillate faster than the electron cyclotron frequency.
k_top = 2.0 * np.pi * SEED_MODES[-1] / (NZ * DZ)
kd_top = 2.0 / DZ * np.sin(k_top * DZ / 2.0) * DE
column = spec[:, SEED_MODES[-1]]
positive = np.where(omega > 0.05 * WCE)[0]
top = omega[positive[np.argmax(column[positive])]] / WCE
assert top < 1.0, (
    f"the top seeded mode oscillates at {top:.3f} Omega_ce, above the "
    f"electron cyclotron frequency; the inertia term is not bounding the "
    f"whistler branch (inertialess prediction here is {kd_top**2:.3f})"
)

print(f"\nworst relative error {worst:.4f} (tolerance {RTOL}); "
      f"top mode at {top:.3f} Omega_ce, bounded by the cyclotron frequency")
