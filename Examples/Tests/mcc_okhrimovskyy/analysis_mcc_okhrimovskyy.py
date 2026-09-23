#!/usr/bin/env python3
"""Check Okhrimovskyy anisotropic scattering for background MCC.

All beams start along +z, and every collided electron was scattered exactly
once, so the cosine of its scattering angle is uz/|u|. The Okhrimovskyy
distribution of mu = cos(chi) has density proportional to 1/(1 - xi mu)^2 and
its sampling inverts R -> mu = 1 - 2R(1 - xi)/(1 + xi(1 - 2R)), whence the CDF

    F(mu) = 1 - a(1 + xi)/(1 - xi + 2 xi a),    a = (1 - mu)/2,

and the mean

    <mu> = 1/xi + (1 - xi^2)/(2 xi^2) ln((1 - xi)/(1 + xi)).

Each sample is compared with its distribution by a Kolmogorov-Smirnov test
and by its mean. Isotropic scattering (xi = 0) would give <mu> = 0 and fails
both outright for the xi used here.

For the free-kinematics beam the ejected electron's direction is set from the
incident one's: opposite azimuth, p_s sin(theta_s) = p_1 sin(theta_1) with
sin(theta_s) clamped at 1, and the same hemisphere. The two electrons of each
event are paired through their energies, which sum to E0 - E_loss, and the
balance is checked event by event.
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy import constants, stats

# must match the input file
E_elec = 200.0  # eV
E_loss = 20.0  # eV, for both the excitation and the ionization


def xi_excitation(E):
    """The tabulated excitation anisotropy: 0.2 at 0 eV to 0.8 at 400 eV."""
    return np.interp(E, [0.0, 400.0], [0.2, 0.8])


xi_ionization = -0.5

mc2 = constants.m_e * constants.c**2 / constants.e  # eV
u0 = np.sqrt(2.0 * E_elec * constants.e / constants.m_e) / constants.c
E0 = (np.sqrt(1.0 + u0**2) - 1.0) * mc2


def okhrimovskyy_cdf(mu, xi):
    a = 0.5 * (1.0 - mu)
    return 1.0 - a * (1.0 + xi) / (1.0 - xi + 2.0 * xi * a)


def okhrimovskyy_mean(xi):
    return 1.0 / xi + (1.0 - xi**2) / (2.0 * xi**2) * np.log((1.0 - xi) / (1.0 + xi))


def check(name, mu, cdf, mean_theory):
    """KS and mean test of the sample mu against a distribution."""
    n = mu.size
    assert n > 1000, f"{name}: only {n} samples, too few for the check"
    ks = stats.kstest(mu, cdf)
    ks_tolerance = 4.0 / np.sqrt(n)
    # The standard deviation of mu is at most 1, so this is a >4 sigma band.
    mean_tolerance = 4.0 / np.sqrt(n)
    print(
        f"{name:28s} n = {n:6d}  <cos> = {mu.mean():+.4f} "
        f"(expected {mean_theory:+.4f})  KS = {ks.statistic:.4f} "
        f"(tolerance {ks_tolerance:.4f})"
    )
    assert ks.statistic < ks_tolerance, (
        f"{name}: the scattering cosines deviate from the expected distribution by "
        f"a KS statistic of {ks.statistic:.4f}, above {ks_tolerance:.4f}"
    )
    assert abs(mu.mean() - mean_theory) < mean_tolerance, (
        f"{name}: the mean scattering cosine is {mu.mean():.4f} against an "
        f"expected {mean_theory:.4f}"
    )


ts = OpenPMDTimeSeries("diags/diag1/")
it = ts.iterations[-1]


def momenta(species):
    ux, uy, uz = ts.get_particle(var_list=["ux", "uy", "uz"], species=species, iteration=it)
    u = np.sqrt(ux**2 + uy**2 + uz**2)
    return (np.sqrt(1.0 + u**2) - 1.0) * mc2, ux, uy, uz, u


def energy_and_cosine(species):
    E, _, _, uz, u = momenta(species)
    return E, uz / u


# Excitation: the collided electrons carry E0 - E_loss, less a recoil of order
# (m_e/M) E0 ~ 1e-4 eV off the 1000 m_p background.
E, mu = energy_and_cosine("beam_exc")
collided = E < E0 - 1.0
assert np.allclose(E[~collided], E0, rtol=1.0e-12, atol=0.0)
assert np.allclose(E[collided], E0 - E_loss, rtol=0.0, atol=1.0e-2), (
    "collided excitation electrons are not at E0 - E_loss"
)
# The anisotropy is looked up at the collision energy, E0 to within m_e/M.
xi = xi_excitation(E0)
check(
    f"excitation (xi = {xi:.3f})",
    mu[collided],
    lambda m: okhrimovskyy_cdf(m, xi),
    okhrimovskyy_mean(xi),
)

# Ionization: with Opal sharing the ejected electron takes W <= dE/2, so the
# incident one leaves with dE - W >= dE/2.
E, mu = energy_and_cosine("beam_ion")
n_events = ts.get_particle(var_list=["w"], species="ions", iteration=it)[0].size
dE = E0 - E_loss
collided = E < E0 - 1.0
assert np.allclose(E[~collided], E0, rtol=1.0e-12, atol=0.0)
incident = collided & (E > 0.5 * dE)
ejected = collided & (E < 0.5 * dE)
assert incident.sum() == n_events and ejected.sum() == n_events, (
    f"{incident.sum()} incident and {ejected.sum()} ejected electrons against "
    f"{n_events} ionization events"
)
check(
    f"ionization incident (xi = {xi_ionization:+.1f})",
    mu[incident],
    lambda m: okhrimovskyy_cdf(m, xi_ionization),
    okhrimovskyy_mean(xi_ionization),
)
check("ionization ejected (isotropic)", mu[ejected], lambda m: 0.5 * (1.0 + m), 0.0)

# Free kinematics. u = gamma v is the momentum per unit mass, so transverse
# momentum balance between the two electrons is a balance of (ux, uy).
E, ux, uy, uz, u = momenta("beam_fk")
n_events = ts.get_particle(var_list=["w"], species="ions_fk", iteration=it)[0].size
collided = E < E0 - 1.0
incident = np.flatnonzero(collided & (E > 0.5 * dE))
ejected = np.flatnonzero(collided & (E < 0.5 * dE))
assert incident.size == n_events and ejected.size == n_events

xi_fk = xi_excitation(E0)  # the same linear table
check(
    f"free-kinematics incident (xi = {xi_fk:.3f})",
    uz[incident] / u[incident],
    lambda m: okhrimovskyy_cdf(m, xi_fk),
    okhrimovskyy_mean(xi_fk),
)

# Pair each incident electron with the one it ejected: E_1 + E_s = dE.
i1 = incident[np.argsort(dE - E[incident])]
i2 = ejected[np.argsort(E[ejected])]
pair_mismatch = np.abs(E[i2] - (dE - E[i1])).max()
assert pair_mismatch < 1.0e-6, f"events do not pair by energy ({pair_mismatch:.2e} eV)"

p1 = np.stack([ux[i1], uy[i1]], axis=1)
p2 = np.stack([ux[i2], uy[i2]], axis=1)
p1_perp = np.linalg.norm(p1, axis=1)
p2_perp = np.linalg.norm(p2, axis=1)
scale = u[i1]

# Opposite azimuths: the transverse momenta are antiparallel in every event.
cross = (p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]) / (scale * u[i2])
dot = np.sum(p1 * p2, axis=1)
assert np.abs(cross).max() < 1.0e-9, "ejected azimuth is not opposite the incident one"
assert np.all(dot[(p1_perp > 1.0e-9 * scale) & (p2_perp > 1.0e-9 * u[i2])] < 0.0)

# Clamped events: sin(theta_s) = 1, so the ejected electron is transverse.
clamped = np.abs(uz[i2]) < 1.0e-12 * u[i2]
free = ~clamped
imbalance = np.linalg.norm(p1 + p2, axis=1) / scale
print(
    f"{'free-kinematics pairs':28s} n = {n_events:6d}  unclamped = {free.sum()}  "
    f"max transverse imbalance (unclamped) = {imbalance[free].max():.2e}"
)
assert free.sum() > 200, f"only {free.sum()} unclamped events, too few for the check"
assert imbalance[free].max() < 1.0e-9, (
    "the transverse momenta of the two electrons do not cancel in unclamped events"
)
assert np.all(p1_perp[clamped] >= u[i2][clamped] * (1.0 - 1.0e-9)), (
    "an event was clamped although p_1 sin(theta_1) < p_s"
)
assert np.all(np.sign(uz[i2][free]) == np.sign(uz[i1][free])), (
    "the ejected electron is not in the incident electron's hemisphere"
)
