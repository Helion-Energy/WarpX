#!/usr/bin/env python3
#
# --- Analysis for the Je-form whistler-dispersion gate
# --- (je_whistler_gate.py). Fits the complex frequency of the seeded
# --- k*d_e ~ 1 transverse mode from the per-step line probe and
# --- discriminates between BOTH analytic branches of the parallel
# --- R-mode of the quasineutral hybrid dispersion: the
# --- electron-inertia-corrected branch (whistler rollover) and the
# --- massless-electron branch (about a factor 2 higher). The gate: the
# --- e_form arm must land on the massless branch and the je_form arm on
# --- the inertia branch, each within tolerance AND at > 0.9 of the
# --- branch gap from the wrong branch (an absolute tolerance alone can
# --- pass while sitting on the wrong branch).
# ---
# --- Discretization: the theory takes an effective wavenumber per
# --- operator. The curl k_eff is read from the E-FORM arm (self-
# --- calibration among the continuum/staggered/centered candidates).
# --- The relax advance carries TRUE dJe/dt dynamics with no spatial
# --- screening stencil (r = 1); an elliptic (jacobi) advance would
# --- instead screen through the grad^2 stencil (r = k_scr^2/k_curl^2).
# ---
# --- usage: je_whistler_analysis.py <e_form_run_dir> <je_form_run_dir>

import json
import sys

import numpy as np

M_E = 9.1093837015e-31
Q_E = 1.602176634e-19
MU0 = 4e-7 * np.pi


class Arm:
    def __init__(self, run_dir):
        with open(f"{run_dir}/gate_params.json") as f:
            p = json.load(f)
        self.p = p
        self.M = p["m_ion"] * M_E
        self.w_ci = Q_E * p["B0"] / self.M
        self.w_ce = Q_E * p["B0"] / M_E
        self.d_i = np.sqrt(self.M / (MU0 * p["n_plasma"] * Q_E**2))
        self.d_e = self.d_i * np.sqrt(M_E / self.M)
        self.K = 2.0 * np.pi * p["mode_m"] / p["Lz"]
        dz = p["dz"]
        self.k_cands = {
            "continuum": self.K,
            "staggered": 2.0 * np.sin(self.K * dz / 2.0) / dz,
            "centered": np.sin(self.K * dz) / dz,
        }
        self.k_scr = self.k_cands["staggered"]  # elliptic grad^2 stencil
        self.t, self.bxh, self.byh = self.load_series(run_dir)

    def load_series(self, run_dir):
        """Per-step complex mode projections of Bx, By from the probe."""
        data = np.loadtxt(f"{run_dir}/diags/par_field_data.txt", skiprows=1)
        steps = data[:, 0].astype(int)
        t, bxh, byh = [], [], []
        for s in np.unique(steps):
            rows = data[steps == s]
            z = rows[:, 4]
            order = np.argsort(z)
            z, bx, by = z[order], rows[order, 8], rows[order, 9]
            ph = np.exp(-1j * self.K * z)
            t.append(rows[0, 1])
            bxh.append(2.0 * np.mean(bx * ph))
            byh.append(2.0 * np.mean(by * ph))
        return np.array(t), np.array(bxh), np.array(byh)

    def fit_fast_component(self, sig, wmin=10.0):
        """Hann-FFT peak with quadratic sub-bin interpolation above
        wmin*w_ci, then a bandpassed-envelope decay fit."""
        t = self.t
        dt = t[1] - t[0]
        n = len(sig)
        pad = 16 * n
        spec = np.fft.fft(sig * np.hanning(n), pad)
        freq = np.fft.fftfreq(pad, dt) * 2 * np.pi
        ipk = np.argmax(np.abs(spec) * (np.abs(freq) > wmin * self.w_ci))
        ya, yb, yc = (
            np.log(np.abs(spec[ipk - 1])),
            np.log(np.abs(spec[ipk])),
            np.log(np.abs(spec[ipk + 1])),
        )
        delta = 0.5 * (ya - yc) / (ya - 2 * yb + yc)
        w_meas = abs(freq[ipk] + delta * (freq[1] - freq[0]))
        band = np.abs(np.abs(freq) - w_meas) < 8.0 * self.w_ci
        filt = np.fft.ifft(np.where(band, np.fft.fft(sig, pad), 0.0))[:n]
        env = np.abs(filt)
        lo, hi = max(2, int(0.05 * n)), int(0.7 * n)
        p = np.polyfit(t[lo:hi], np.log(env[lo:hi]), 1)
        return w_meas, p[0]

    def measure(self):
        fits = []
        for lbl, sig in (
            ("c+", self.bxh + 1j * self.byh),
            ("c-", self.bxh - 1j * self.byh),
        ):
            w, g = self.fit_fast_component(sig)
            fits.append((np.abs(sig).max(), lbl, w, g))
        _, lbl, w, g = max(fits)
        return lbl, w, g

    def dispersion_root(self, inertia, k_curl, seed):
        """Complex root of the parallel R-mode (s = -1) of the
        quasineutral hybrid dispersion with resistivity. The inertia
        screening is advance-dependent: an elliptic (jacobi) advance
        screens through the grad^2 stencil (r = k_scr^2/k_curl^2),
        while the relax advance carries TRUE dJe/dt dynamics with no
        spatial screening stencil (r = 1)."""
        if self.p.get("je_advance", "relax") == "relax":
            r = 1.0 if inertia else 0.0
        else:
            r = (self.k_scr / k_curl) ** 2 if inertia else 0.0
        W = self.w_ce / r if inertia else 1e30 * self.w_ci
        s = -1.0

        def F(w):
            return (
                -(s + w / W)
                * (
                    self.d_i**2 * k_curl**2 * self.w_ci / w
                    - self.w_ci / (s * self.w_ci - w)
                )
                - 1.0
                - 1j * self.p["eta"] * k_curl**2 / (MU0 * w)
            )

        w = complex(seed)
        for _ in range(200):
            dw = w * 1e-8
            w = w - F(w) / ((F(w + dw) - F(w)) / dw)
        return w


eform = Arm(sys.argv[1])
jeform = Arm(sys.argv[2])
assert eform.p["esolve"] == "e_form" and jeform.p["esolve"] == "je_form", (
    "pass the run dirs as <e_form> <je_form>"
)

print(
    f"loaded: e_form {len(eform.t)} dumps, je_form {len(jeform.t)} dumps, "
    f"k*d_e = {eform.K * eform.d_e:.4f}"
)

lbl_e, w_ef, g_ef = eform.measure()
lbl_j, w_jf, g_jf = jeform.measure()
print(
    f"measured: e_form ({lbl_e}) w = {w_ef / eform.w_ci:.3f} w_ci, "
    f"je_form ({lbl_j}) w = {w_jf / jeform.w_ci:.3f} w_ci"
)

# --- curl k_eff self-calibration on the e_form arm --------------------------
print("e_form arm vs massless branch per curl-k candidate:")
best = None
for name, kc in eform.k_cands.items():
    w_ml = eform.dispersion_root(False, kc, 60 * eform.w_ci)
    err = abs(w_ef - w_ml.real) / w_ml.real
    print(f"  {name:10s}: w_ml = {w_ml.real / eform.w_ci:7.2f} w_ci, rel err = {err:.3%}")
    if best is None or err < best[2]:
        best = (name, kc, err)
k_name, k_curl, err_cal = best
print(f"calibrated curl k_eff = {k_name} (rel err {err_cal:.3%})")

# --- dual-branch gate on both arms -----------------------------------------
failures = []
for arm, w_meas, want_inertia in ((eform, w_ef, False), (jeform, w_jf, True)):
    w_in = arm.dispersion_root(True, k_curl, 60 * arm.w_ci)
    w_ml = arm.dispersion_root(False, k_curl, 60 * arm.w_ci)
    w_match = w_in if want_inertia else w_ml
    w_other = w_ml if want_inertia else w_in
    err = abs(w_meas - w_match.real) / w_match.real
    sep = abs(w_meas - w_other.real) / abs(w_match.real - w_other.real)
    tag = arm.p["esolve"]
    print(
        f"{tag}: measured {w_meas / arm.w_ci:7.2f} w_ci | inertia branch "
        f"{w_in.real / arm.w_ci:7.2f} | massless branch "
        f"{w_ml.real / arm.w_ci:7.2f} | rel err vs matching = {err:.3%} | "
        f"gap fraction = {sep:.2f}"
    )
    if err > 0.02:
        failures.append(f"{tag}: rel err {err:.3%} > 2%")
    if sep < 0.9:
        failures.append(f"{tag}: gap fraction {sep:.2f} < 0.9")

if failures:
    print("FAIL:\n  " + "\n  ".join(failures))
    sys.exit(1)
print(
    "PASS: e_form arm on the massless branch, je_form arm on the "
    "inertia-corrected branch, both > 0.9 of the branch gap from the "
    "wrong branch"
)
