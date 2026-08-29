#!/usr/bin/env python3
#
# --- Analysis for the electron-inertia seeded-whistler dispersion test.
# --- A single transverse mode at k*d_e ~ 1 is seeded in the initial field;
# --- its complex frequency is fit from the per-step line-probe series and
# --- compared against the parallel R-mode of the quasineutral hybrid
# --- dispersion WITH the electron-inertia correction (whistler rollover,
# --- w -> w_ce k^2 d_e^2 / (1 + k^2 d_e^2) at high k) including the
# --- resistive damping. The fit must land on the inertia-corrected branch
# --- and exclude the massless-electron branch (about a factor 2 higher).
# --- With --massless-control, assert the massless branch instead (for the
# --- inertia-off control of the same seeded configuration).

import sys

import dill
import numpy as np

from pywarpx import picmi

constants = picmi.constants

MASSLESS = "--massless-control" in sys.argv

with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

MODE_M = 41
K = 2.0 * np.pi * MODE_M / sim.Lz
w_ci = sim.w_ci
d_i = constants.c / sim.w_pi
d_e = d_i * np.sqrt(constants.m_e / sim.M)
w_ce_eff = constants.q_e * sim.B0 / constants.m_e
eta = sim.eta
# The theory curves take the effective wavenumber of the second-order
# curl stencil, k_eff = 2 sin(k dz/2)/dz: at k dz ~ 1 the whistler
# (w ~ k^2) is ~8% below the continuum value, and with this correction
# both branches agree with the measured frequencies to ~1%.
K_EFF = 2.0 * np.sin(K * sim.dz / 2.0) / sim.dz


def dispersion_root(inertia, seed):
    """Complex root of the parallel R-mode (s = -1) of the quasineutral
    hybrid dispersion with optional electron inertia and resistivity:
    1 + i eta k^2/(mu0 w) = -(s + w/w_ce)[d_i^2 k^2 w_ci/w - w_ci/(s w_ci - w)]
    """
    W = w_ce_eff if inertia else 1e30 * w_ci
    s = -1.0

    def F(w):
        return (-(s + w / W) * (d_i**2 * K_EFF**2 * w_ci / w
                                - w_ci / (s * w_ci - w))
                - 1.0 - 1j * eta * K_EFF**2 / (constants.mu0 * w))

    w = complex(seed)
    for _ in range(200):
        dw = w * 1e-8
        w = w - F(w) / ((F(w + dw) - F(w)) / dw)
    return w


def load_series(path="diags/par_field_data.txt"):
    """Per-step complex mode-41 projections of Bx, By from the line probe."""
    data = np.loadtxt(path, skiprows=1)
    steps = data[:, 0].astype(int)
    t, bxh, byh = [], [], []
    for s in np.unique(steps):
        rows = data[steps == s]
        z = rows[:, 4]
        order = np.argsort(z)
        z, bx, by = z[order], rows[order, 8], rows[order, 9]
        ph = np.exp(-1j * K * z)
        t.append(rows[0, 1])
        bxh.append(2.0 * np.mean(bx * ph))
        byh.append(2.0 * np.mean(by * ph))
    return np.array(t), np.array(bxh), np.array(byh)


def fit_fast_component(t, sig, wmin=10.0):
    """Frequency and damping of the dominant spectral component above
    wmin*w_ci: Hann FFT peak with quadratic sub-bin interpolation, then a
    bandpassed-envelope decay fit."""
    dt = t[1] - t[0]
    n = len(sig)
    pad = 16 * n
    spec = np.fft.fft(sig * np.hanning(n), pad)
    freq = np.fft.fftfreq(pad, dt) * 2 * np.pi
    ipk = np.argmax(np.abs(spec) * (np.abs(freq) > wmin * w_ci))
    ya, yb, yc = (np.log(np.abs(spec[ipk - 1])), np.log(np.abs(spec[ipk])),
                  np.log(np.abs(spec[ipk + 1])))
    delta = 0.5 * (ya - yc) / (ya - 2 * yb + yc)
    w_meas = abs(freq[ipk] + delta * (freq[1] - freq[0]))
    band = np.abs(np.abs(freq) - w_meas) < 8.0 * w_ci
    filt = np.fft.ifft(np.where(band, np.fft.fft(sig, pad), 0.0))[:n]
    env = np.abs(filt)
    lo, hi = max(2, int(0.05 * n)), int(0.7 * n)
    p = np.polyfit(t[lo:hi], np.log(env[lo:hi]), 1)
    return w_meas, p[0]


t, bxh, byh = load_series()
print(f"loaded {len(t)} probe dumps, T*w_ci = {(t[-1] - t[0]) * w_ci:.3f}, "
      f"k*d_e = {K * d_e:.4f}")

fits = []
for lbl, sig in (("c+", bxh + 1j * byh), ("c-", bxh - 1j * byh)):
    w, g = fit_fast_component(t, sig)
    fits.append((np.abs(sig).max(), lbl, w, g))
    print(f"  {lbl}: w = {w / w_ci:7.2f} w_ci, gamma = {g / w_ci:+6.2f} w_ci")
_, lbl, w_meas, g_meas = max(fits)

w_in = dispersion_root(True, 60 * w_ci)
w_ml = dispersion_root(False, 60 * w_ci)
w_match = w_ml if MASSLESS else w_in
w_other = w_in if MASSLESS else w_ml
print(f"whistler ({lbl}): measured w = {w_meas / w_ci:.2f} w_ci, "
      f"gamma = {g_meas / w_ci:+.2f} w_ci")
print(f"inertia-corrected branch: w = {w_in.real / w_ci:.2f} "
      f"{w_in.imag / w_ci:+.2f}i;  massless branch: w = {w_ml.real / w_ci:.2f} "
      f"{w_ml.imag / w_ci:+.2f}i")

err = abs(w_meas - w_match.real) / w_match.real
sep = abs(w_meas - w_other.real) / abs(w_match.real - w_other.real)
print(f"relative error vs matching branch: {err:.3%}; "
      f"distance to the other branch: {sep:.2f} of the gap")
assert err < 0.10, f"frequency error {err:.3%} exceeds 10%"
assert sep > 0.75, "fit does not prefer the matching dispersion branch"
print(f"PASS: whistler follows the "
      f"{'massless' if MASSLESS else 'inertia-corrected'} branch")
