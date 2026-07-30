# Literature review: open / free-space boundary conditions for gridded axisymmetric
# MHD / hybrid-PIC field advances via Green's-function and boundary-integral closures

*(2026-07-30. Companion to OPEN_BC_GREENS_DESIGN.md and INTERNAL-REF_WALL_TREATMENT.md.
Purpose: ground the WarpX RZ hybrid-PIC open-BC implementation in prior art;
confirm or challenge the four load-bearing design decisions. References at the
end; bracketed numbers cite that list.)*

---

## 1. Problem class

Our task — filling boundary ghost values of B for an RZ (m=0) Faraday/Ohm's-law
field advance with the free-space field of the interior sources — sits at the
intersection of four literatures:

1. **Free-boundary Grad–Shafranov equilibrium**: an elliptic solve for
   psi = r A_theta on a finite grid, closed by boundary values obtained from the
   axisymmetric ring-current Green's function (von Hagenow & Lackner [1,2],
   Jardin's textbook treatment [5]). The kernel is *identical* to ours.
2. **Time-dependent free-boundary transport / positional-control codes**
   (TSC [6,7], DINA-class solvers, TokaMaker [17], FGE [18]): the same elliptic
   closure applied every timestep of an evolving system, plus circuit equations
   for external conductors.
3. **Nonlinear MHD initial-value codes with vacuum/resistive-wall closures**
   (NIMROD+GRIN [8,9], M3D-C1 [10], JOREK+STARWALL [11,12,13], PIXIE3D [14],
   SPECYL [15], and the thin-wall eddy-current literature CARIDDI /
   Albanese–Rubinacci [19,20], Chance's VACUUM [16]): here the *hyperbolic /
   parabolic* field advance is closed by a boundary-integral vacuum response,
   which is the closest structural analog to our per-substep ghost fill.
4. **Open-boundary Poisson solvers in particle codes** (James [21], Hockney &
   Eastwood [22], Vico–Greengard–Ferrando [23]) and **fast-summation error
   analysis** (Barnes–Hut / Salmon–Warren [24,25,26]): the algorithmic
   machinery — screening charges, kernel convolutions, and the error laws
   governing source coarse-graining.

The distinguishing feature of our setting: the field is *advanced* (Faraday
update of B from a curl of the Ohm E), not *solved*; there is no elliptic
operator to attach a boundary condition to. The boundary values must therefore
be produced by direct source integration each time the BC is applied, exactly
as the design doc states (Sec. 1). Precedents for this "integral closure of a
time advance" exist and are reviewed in Secs. 4–6.

---

## 2. von Hagenow & Lackner: the canonical free-boundary elliptic closure

**What is gridded / what is integral.** The Grad–Shafranov operator
Delta* psi = -mu0 r J_theta is discretized on a rectangular (r,z) grid; the
vacuum outside the rectangle is never gridded. The free-space solution is

    psi(x_b) = Int_domain G0(x_b; x') J_theta(x') dV',

with the ring-current kernel

    G0 = (mu0/4pi) sqrt((z-z')^2 + (r+r')^2) [(2-k^2) K(k) - 2 E(k)],
    k^2 = 4 r r' / ((z-z')^2 + (r+r')^2),

— i.e., precisely the kernel in OPEN_BC_GREENS_DESIGN.md and internal-ref's `G0`.
Jardin [5, Ch. 4] gives this kernel in Sec. 4.6.3 and von Hagenow's method in
Sec. 4.6.4.

**The von Hagenow trick** [1,2,5]: rather than evaluating the volume integral
directly for every boundary node (cost O(N_b × N_vol) = O(N^3) per iteration in
2D), one performs one *interior* solve with homogeneous Dirichlet data,
psi_D, and then converts the volume integral into a *surface* integral over the
boundary of the normal derivative of psi_D:

    psi(x_b) = - Oint_Gamma G0(x_b; x') (1/(mu0 r')) d(psi_D)/dn' dGamma',

reducing the boundary fill to O(N_b^2) ~ O(N^2) (often quoted as O(N^2 ln N)
including the fast interior solve). This is the standard method in essentially
every free-boundary equilibrium code (EFIT, CEDRES++ [27], FreeGSNKE/Fiesta
[28], TokaMaker [17], CRATOS-GS [29]). Recent numerical-analysis work
(Faugeras et al. FEM-BEM coupling study [3]) formalizes von Hagenow–Lackner as
one of several FEM-BEM coupling schemes and notes it has *non-optimal
convergence* compared with Johnson–Nédélec or Albanese–Blum–de Barbieri
couplings for nonlinear problems — but it remains the community default because
it is simple, robust, and cheap.

**Coupling into the solve**: iterative (Picard/Newton on the nonlinear GS
problem); the boundary values are lagged one nonlinear iteration. That is an
artifact of the elliptic problem's nonlinearity, not of the boundary closure —
the closure itself is a *linear* map from interior source to boundary value,
which is the property our design leans on for JFNK smoothness.

**Relevance to us**: von Hagenow's surface-integral speedup is only available
when you *have* an interior elliptic solve whose homogeneous-Dirichlet solution
you can differentiate. Our Faraday advance has none. The volume-integral form
(the "slow" direct method von Hagenow was accelerating) is exactly our
psi_ghost integral — so the correct reading of this literature for us is: the
*kernel and the closure are standard*; the *acceleration* must come from
elsewhere (coarse-graining / precomputed GEMV, Sec. 7 below).

---

## 3. Jardin's textbook and TSC: time-dependent free-boundary evolution

**Jardin, Computational Methods in Plasma Physics (CRC, 2010)** [5] treats the
free-boundary problem in Ch. 4 (Green's function for Delta*, Sec. 4.6.3;
von Hagenow, Sec. 4.6.4) and time-dependent axisymmetric evolution with
external conductors in the transport chapters. The key structural lesson: the
vacuum/conductor system is *linear*, so it can always be reduced to
precomputed geometry matrices (mutual inductances, Green's tables) applied per
step — the expensive elliptic-integral evaluations happen once at setup.

**TSC (Jardin, Pomphrey & DeLucia, J. Comput. Phys. 66, 481 (1986))** [6,7]
evolves a free-boundary axisymmetric plasma on a uniform Cartesian (r,z) grid
that is divided into three regions: plasma, *gridded vacuum* (modeled as a
highly resistive, force-free medium so the same field equations apply
everywhere), and solid conductors. External-to-grid conductors (PF coils,
vessel structures) are advanced by *circuit equations* coupled through
free-space mutual inductances (ring-current Green's functions); grid-boundary
flux is set from the Green's-function contributions of all currents (plasma +
conductor). So TSC is a *mixed* strategy: grid the near vacuum with a large
resistivity, integral-couple the far region and conductors. Cost: the
boundary/conductor coupling is a dense but small precomputed matrix; the
per-step cost is dominated by the interior solve. Coupling into the advance:
implicit for the field/circuit system (the conductor circuit equations are
folded into the same time level), which is essentially internal-ref's monolithic
scheme (INTERNAL-REF_WALL_TREATMENT.md Sec. 3) — unsurprising, as internal-ref's
technical notes cite the same lineage.

**Relevance**: TSC legitimizes both halves of our roadmap — Green's-table
boundary values recomputed every step from the evolving interior current, and
(later) conductors as circuit unknowns coupled through the same kernel. TSC's
gridded-resistive-vacuum region is the alternative we are *rejecting* (it
requires the fluid solver to own the vacuum, which the hybrid-PIC Ohm's law
cannot do gracefully at near-zero density; the dust/vacuum floor issue).

---

## 4. Initial-value MHD codes: integral closures of a time advance

These are the closest analogs: a time-advanced field system whose boundary is
closed each step (or each linear solve) by a boundary-integral vacuum response.

### 4.1 NIMROD + GRIN / resistive wall

NIMROD's resistive-wall capability [8,9] uses the **GRIN** code to construct
*vacuum response matrices*: dense matrices, computed once from geometry, that
map boundary values of the normal field (or scalar potential) to the tangential
field just outside, i.e., a discrete Dirichlet-to-Neumann (DtN) operator for
the exterior Laplace problem. The thin resistive wall then advances
B_normal at the wall by the jump condition. Two implementations have been
compared in toroidal geometry: (a) the Green's-function (GRIN-style) approach
with *no gridded exterior*, and (b) a Lagrange-multiplier approach with a
meshed external vacuum; the Green's-function version is noted as "more
straightforwardly compatible" with a boundary placed inside the coils [9] —
directly analogous to our in-domain-coil motivation.

- Gridded: plasma region (high-order finite elements).
- Integral: exterior vacuum, as a *precomputed dense DtN matrix* applied to
  boundary data each step.
- Coupling: enters the (implicit) time advance as a boundary term; because the
  map is linear and precomputed, it does not perturb solver structure.
- Cost: dense N_b × N_b matrix application per step — a GEMV. Setup is the
  expensive part (Green's-table assembly with singular quadrature).

### 4.2 M3D-C1 multi-region resistive wall

Ferraro, Jardin, Lao, Shephard & Zhang, Phys. Plasmas 23, 056114 (2016) [10]
took the *opposite* route in M3D-C1: the resistive wall **and a surrounding
vacuum region are gridded inside the computational domain** (vacuum modeled,
TSC-like, as a resistive medium), preserving the sparse local structure of the
implicit solve. Motivations stated: avoiding dense boundary matrices keeps the
implicit solver's scalability; the price is meshing to a far outer boundary
where a crude (conducting) BC is finally imposed. This is the reference point
for "what if we just moved the PEC wall far away" — it works, but the domain
(and cost) grows, and the far boundary is still an image-current wall, just a
weaker one (the design doc's |c_wall| ~ (R_c/R_w)^2.6 scaling estimates the
residual).

### 4.3 JOREK + STARWALL

Hoelzl, Merkel, Huysmans, Nardon, Strumberger et al. [11,12] couple the
nonlinear MHD code JOREK to **STARWALL** [13], which discretizes conducting
structures into triangles (thin-wall approximation) and represents the
un-gridded vacuum by surface Green's functions. The coupling replaces the
Dirichlet BC of the JOREK domain with a **natural (weak) boundary condition**:
a boundary integral expressing the tangential magnetic field at the domain edge
in terms of the boundary poloidal-flux values and the conductor currents. The
conductor currents are extra unknowns advanced with an *implicit* (in the
free-boundary extension [12], fully monolithic) scheme. Cost structure:
STARWALL precomputes large dense response matrices (this is the memory
bottleneck — the JORSTAR parallelization reports [30] exist because those
matrices reach hundreds of GB for fine 3D walls); per-step application is dense
matvecs.

### 4.4 PIXIE3D / SPECYL

Bonofiglo, Chacón et al. (arXiv:2606.05446, 2026) [14] give the most recent and
most detailed write-up: resistive-wall BCs for the PIXIE3D extended-MHD code in
axisymmetric toroidal geometry via a **thin wall + boundary integral method**
for the magnetic scalar potential in the surrounding vacuum, with specialized
quadrature for the singular and hypersingular kernels (Laplace Green's function
and derivatives), an extension to a second, outer perfectly conducting wall,
and *analytic verification suites demonstrating second-order accuracy*. Earlier
SPECYL/PIXIE3D cross-verification [15] establishes the nonlinear correctness
methodology (code-to-code benchmarks of resistive-wall modes). Chance's VACUUM
[16] is the classic stability-code version of the same construction (Green's
second identity + collocation for the exterior scalar potential in general
axisymmetric geometry).

**Common structure across 4.1–4.4** (and the internal-ref closure): what is gridded
is the plasma; what is integral is the exterior Laplace/vacuum response,
reduced at setup to dense boundary matrices; the coupling enters the time
advance linearly through boundary data, so implicit solvers see a smooth,
constant linear operator; per-step cost is a dense matvec of modest size, setup
cost is kernel assembly with singular quadrature. Our design matches this
pattern, with one deliberate simplification: because we need *ghost values of B
produced by interior sources* rather than a self-consistent
plasma–vacuum–conductor eigenresponse, we can skip the DtN construction and
its singular quadrature entirely — our kernel is evaluated at ghost points
*separated from the source cells*, so all integrands are regular (nearest
source cell is >= 1 cell away from the boundary psi-points; only the
coarse-graining, not singularity handling, controls accuracy).

---

## 5. Boundary-element eddy-current coupling (CARIDDI lineage)

Albanese & Rubinacci's integral eddy-current formulation (IEE Proc. A 135, 457
(1988); the CARIDDI code) [19,20] solves for volumetric conductor eddy currents
with edge elements whose shape functions impose div J = 0 exactly, coupled
through free-space Green's functions — no vacuum mesh, unbounded exterior
handled exactly. CARIDDI has been coupled to MHD codes in the same role as
STARWALL (e.g., in the JOREK ecosystem and CarMa0NL). Two lessons transfer:

1. **Enforce the discrete constraint in the basis, not by penalty.** CARIDDI
   builds div J = 0 into the shape functions; our analog is building
   div B = 0 into the ghost fill by *differencing psi exactly as the Yee
   stencil does* rather than evaluating B_r, B_z from analytic derivatives of
   G0 (which would satisfy div B = 0 only to truncation error).
2. **Fast matvecs for the dense kernel.** The follow-on "fast 3D eddy current
   integral formulation" work [20] splits short-range (exact) from long-range
   (compressed) interactions — the same structure as our Phase-2 FMM option.

---

## 6. Open-boundary Poisson solvers in particle codes

**James (J. Comput. Phys. 25, 71 (1977))** [21]: solve the interior problem
with homogeneous Dirichlet data, compute the induced *screening surface
charge* on the boundary, convolve it with the free-space Green's function to
get the boundary potential, and re-solve. This is von Hagenow's identity in
Cartesian particle-code clothing, and it is the standard "isolated source" BC
in gravity/beam PIC codes to this day [31]. **Hockney–Eastwood zero-padded
convolution** [22] and its modern replacement **Vico–Greengard–Ferrando** [23]
solve the same problem spectrally by domain doubling. All three are again
*elliptic-solve piggyback* methods: they need a fast solver to exist. For our
Faraday advance the analog of James's screening-charge step does not apply,
but James's key economy — *the expensive free-space convolution is needed on
the boundary only, with the interior handled by the existing machinery* — is
exactly the economy of our design: the Green's integral is evaluated only at
O(N_b) ghost psi-points, never in the volume.

**Hybrid-PIC prior art.** Published hybrid codes (Winske-lineage, Hybrid-VPIC
[32]) handle "open" field boundaries by splitting B = B0 + B1 with a
prescribed curl-free vacuum B0 held fixed in ghost cells and zero-gradient
extrapolation of B1 and moments — i.e., the plasma-response field still sees a
reflective/extrapolative boundary; no published hybrid-PIC code we found closes
the ghost B with a source-integral free-space response. A 2026 dense-plasma-
focus hybrid PIC-fluid paper [33] evaluates fields inside the domain with the
free-space Green's function for external structures, which is the nearest
relative. **The proposed WarpX BC is, to the best of this survey, novel in the
hybrid-PIC context** — its components (kernel, psi-differencing, per-step dense
matvec closure) are each individually standard in the MHD literature of Sec. 4.

---

## 7. Fast summation and coarse-graining error

The design coarse-grains J_theta onto a 4–8x coarser source grid before the
kernel GEMV, claiming boundary error ~ (d_cell/d_boundary)^2. The relevant
literature is multipole error analysis:

- Replacing a source cell of size h at distance d by its monopole (total
  current at centroid) incurs a relative far-field error whose leading term is
  the *quadrupole* moment, O((h/d)^2), because centroid placement cancels the
  dipole term. Barnes–Hut [24] with opening angle theta = h/d has exactly this
  error law; Salmon & Warren [25] give rigorous bounds and the pathological
  cases (source near cell edge — avoided by centroid weighting, which our
  conservative r-weighted binning implements); Grama, Kumar & Sameh [26]
  sharpen the bounds. Note the axisymmetric refinement: our "monopole" should
  conserve Sum J dV and place the coarse ring at the *current-weighted
  centroid* in (r,z), which is what conservative binning at the centroid does;
  strict moment cancellation to O((h/d)^2) requires the centroid, not the
  geometric cell center — worth an implementation note and a convergence test
  (design doc test 2 covers this).
- The design's fixed coarsening factor corresponds to a Barnes–Hut MAC with
  theta ~ h_coarse/d_min evaluated at the *worst-case* (nearest-source)
  distance. Since sources adjacent to the open face have small d_boundary, a
  uniform 8x coarsening is only safe if the current density near the boundary
  is small (true for our dust-annulus FRC configurations) — otherwise a
  distance-graded coarsening (finer bins near the face, i.e., exactly a
  treecode) is the robust generalization. Phase-2 FMM/H-matrix (ACA)
  compression of the kernel matrix is standard practice in the eddy-current
  BEM literature [20].
- **Periodic images**: for periodic z with period L_z, the image sum of the
  ring kernel converges *geometrically* because the m=0 exterior harmonics of
  a ring decay as exp(-|k_n| Delta z) in a Fourier-z representation — a
  spectral (Fourier-in-z, modified-Bessel-in-r) form of the periodic kernel
  converges much faster than the real-space image sum when many images are
  needed. Ewald splitting [34,35] is the general tool but is overkill at m=0:
  with ~10 real-space images at |Delta z| >= L_z (design doc estimate) the
  direct sum is fine *because the kernel is precomputed once*; convergence of
  the image sum affects only setup cost, not per-step cost. The real-space
  tail of the axisymmetric kernel decays as the dipole field ~ 1/Delta z^2 per
  image at fixed evaluation radius — the *geometric* claim in the design doc is
  not right for the raw kernel sum (it is algebraic, ~ sum 1/n^2 per-image for
  psi), but since the *summed tail* enters a precomputed matrix, simply summing
  images until the increment is below a set tolerance (or Richardson-
  extrapolating the 1/n tail) at init is cheap and removes the concern.
  Recommendation below.

---

## 8. Method comparison table

| Method | Gridded | Integral | Coupling into advance | Per-step cost | Setup cost |
|---|---|---|---|---|---|
| von Hagenow–Lackner [1,2,5] | GS domain | boundary surface integral of dpsi_D/dn | inside nonlinear elliptic iteration | O(N_b^2) surface GEMV + 1 extra Dirichlet solve | small |
| TSC [6,7] | plasma + resistive vacuum + in-grid conductors | ring-current mutuals to external circuits | implicit, monolithic with circuits | small dense matvec | Green tables |
| NIMROD+GRIN [8,9] | plasma | precomputed DtN (vacuum response matrix) | boundary term in implicit solve | dense N_b GEMV | singular-quadrature assembly |
| M3D-C1 [10] | plasma + wall + vacuum (all gridded) | none | fully implicit, sparse | sparse solve on larger domain | mesh generation |
| JOREK+STARWALL [11–13] | plasma | triangulated thin-wall + vacuum Green matrices, natural BC | implicit; wall currents are unknowns | dense GEMVs (large) | very large dense assembly |
| PIXIE3D [14] | plasma | thin wall + BIM scalar potential, singular quadrature | implicit BC rows | dense boundary op | quadrature assembly |
| CARIDDI/BEM [19,20] | conductors (volumetric) | free-space kernel everywhere | implicit L/R dynamics | dense (or compressed) matvec | dense assembly |
| James / H-E / VGF [21–23] | Poisson domain | boundary screening-charge convolution | inside each elliptic solve | FFT / boundary convolution | none |
| internal-ref (internal ref) | plasma flux | segment L, K matrices; S = sym(K L^-1) | monolithic BE, unconditionally passive | small dense solve | dense assembly |
| **This design** | **plasma (RZ Yee)** | **ring kernel: coarse J_theta -> ghost psi** | **explicit per substep; linear map, JFNK-safe** | **GEMV, few MB** | **kernel + coarsening op at init/regrid** |

---

## 9. Recommendations for OPEN_BC_GREENS_DESIGN.md

**(a) Direct kernel GEMV with conservative source coarsening vs von Hagenow
surface integrals — CONFIRMED, with a caveat.** Von Hagenow/James-type surface
methods require an interior elliptic solve to differentiate; a Faraday advance
has none, and *manufacturing* one (an auxiliary Delta* psi = -mu0 r J_theta
Dirichlet solve per application, then the surface integral) would cost an
elliptic solve per substep — strictly worse than a precomputed few-MB GEMV.
Every initial-value analog (GRIN, STARWALL, PIXIE3D, internal-ref) likewise reduces
the vacuum response to a precomputed dense matrix applied per step; ours
differs only in mapping *volume sources* rather than *boundary traces*, which
is what removes all singular quadrature (source cells are separated from ghost
evaluation points). Caveat from the treecode error law: a *uniform* 4–8x
coarsening implies a Barnes–Hut opening ratio evaluated at the nearest source;
add (i) the conservative binning at the current-weighted *centroid* (not
geometric center — this is what cancels the dipole term and delivers the
(h/d)^2 law), and (ii) either verify in tests that near-face currents are
negligible in the target configurations, or grade the coarsening with distance
to the face (a two-level kernel: fine strip near the face, coarse elsewhere)
if test 2's convergence study shows the near-face error dominating.

**(b) psi-differenced ghost fill for div-B consistency — CONFIRMED; this is
the right and non-negotiable choice.** It is the exact analog of the
constraint-in-the-basis principle that the eddy-current BEM literature settled
on (CARIDDI's div-free edge elements) and of constrained-transport practice:
the ghost B must be the Yee difference of a scalar psi extension so that the
discrete div B built from ghost values vanishes identically, not to truncation
order. Evaluating B_r = -(1/r) dpsi/dz analytically from the kernel instead
would inject O(h^2) divergence at the boundary every substep, which a Faraday
advance preserves and accumulates. Also fill *corner* ghost values from the
same psi table so the two faces sharing a corner agree by construction.

**(c) Per-substep application in an explicit advance; smoothness for JFNK —
CONFIRMED.** All implicit MHD analogs apply the (linear, precomputed) vacuum
response inside every linear/nonlinear iteration precisely because it is a
constant linear operator — the Jacobian contribution is exact and the map is
C-infinity in the state. Two implementation cautions from that literature:
(i) apply the fill from the *same time-level* J that the substep's curl uses
(a lagged fill degrades the subcycle order and can drive a weak boundary
instability analogous to loosely-coupled wall schemes; STARWALL's move from
explicit to implicit wall-current coupling [12] was motivated by exactly this
class of instability at strong coupling); since our map is one GEMV, there is
no cost reason to lag it. (ii) There is no CFL-type penalty: the closure adds
no new wave speed — it is an instantaneous magnetostatic response, consistent
with the hybrid model's neglect of displacement current (the same
quasi-static approximation the whole Ohm's-law advance already makes).

**(d) Periodic-z image sums — CONFIRMED as Phase-1 mechanism, with one
correction and one option.** Correction: the per-image tail of the
axisymmetric psi kernel decays *algebraically* (dipole-like, ~1/(n L_z)^2 per
image, worse near k^2 -> 0 cancellation), not geometrically; the design doc's
"~10 images" is still a fine starting point, but make the image count a
tolerance-driven loop at kernel-assembly time (sum until relative increment <
1e-8, with optional Richardson extrapolation of the 1/n^2 tail) rather than a
hard-coded 10. Since the sum lands in a precomputed matrix, this costs setup
time only. Option, if the tolerance loop is ever slow: the m=0
Fourier-in-z representation (modified Bessel I_0/K_0 products per k_n =
2 pi n/L_z) converges exponentially in n at any r_ghost > r_source and is the
standard fast periodic form [34,35]; keep it in a back pocket, do not build it
in Phase 1. Finally, note B_theta on the open r-face under periodic z:
Ampere closure B_theta = mu0 I_enc(z)/(2 pi r) is exact and needs no images at
all — only the poloidal-flux kernel needs the image treatment.

**One addition the survey argues for**: the PIXIE3D paper [14] ships an
*analytic verification suite* (ring/wire fields, second-order convergence) that
maps one-to-one onto our planned tests 1–3; mirror their structure (analytic
free-space field of a prescribed ring; convergence order of the boundary
error; image-suppression factor vs PEC) so the WarpX test suite is directly
comparable to published verification practice.

---

## References

1. K. von Hagenow and K. Lackner, "Computation of axisymmetric MHD equilibria,"
   Proc. 7th Conf. on Numerical Simulation of Plasmas, New York (1975).
   (Method description as standardly cited; see also [5] Sec. 4.6.4.)
2. K. Lackner, "Computation of ideal MHD equilibria," Comput. Phys. Commun.
   12, 33–44 (1976). https://www.sciencedirect.com/science/article/abs/pii/0010465576900084
3. B. Faugeras et al., "FEM-BEM coupling methods for Tokamak plasma
   axisymmetric free-boundary equilibrium computations in unbounded domains,"
   J. Comput. Phys. 343, 201 (2017).
   https://www.sciencedirect.com/science/article/abs/pii/S0021999117303261
4. S. A. Sabbagh et al., "Development of an auto-convergent free-boundary
   axisymmetric equilibrium solver," (2012). https://www.osti.gov/biblio/1051805
5. S. C. Jardin, *Computational Methods in Plasma Physics*, Chapman &
   Hall/CRC (2010), Ch. 4 (Green's function for Delta*, Sec. 4.6.3; von
   Hagenow's method, Sec. 4.6.4).
   https://www.routledge.com/Computational-Methods-in-Plasma-Physics/Jardin/p/book/9781439810217
6. S. C. Jardin, N. Pomphrey, and J. DeLucia, "Dynamic modeling of transport
   and positional control of tokamaks," J. Comput. Phys. 66, 481–507 (1986).
   https://sciencedirect.com/science/article/abs/pii/002199918690077X
7. TSC users manual / fact sheet, PPPL. https://w3.pppl.gov/topdac/tscman.pdf ,
   https://w3.pppl.gov/topdac/tsc.htm
8. Center for Extended MHD Modeling final report (NIMROD/GRIN vacuum response
   matrices). https://www.osti.gov/servlets/purl/1406519/
9. "Boundary condition effects on runaway electron mitigation coil modeling
   for the SPARC and DIII-D tokamaks," Nucl. Fusion 64 (2024) — comparison of
   Green's-function vs meshed-vacuum Lagrange-multiplier resistive-wall
   implementations in NIMROD.
   https://iopscience.iop.org/article/10.1088/1741-4326/ad3c52
10. N. M. Ferraro, S. C. Jardin, L. L. Lao, M. S. Shephard, and F. Zhang,
    "Multi-region approach to free-boundary three-dimensional tokamak
    equilibria and resistive wall instabilities," Phys. Plasmas 23, 056114
    (2016). https://pubs.aip.org/aip/pop/article/23/5/056114/966095/
11. M. Hoelzl, P. Merkel, G. T. A. Huysmans, E. Nardon, E. Strumberger,
    R. McAdams, I. Chapman, S. Guenter, and K. Lackner, "Coupling JOREK and
    STARWALL codes for non-linear resistive-wall simulations," J. Phys.: Conf.
    Ser. 401, 012010 (2012). https://arxiv.org/abs/1206.2748
12. "Self-consistent full MHD coupling of JOREK and STARWALL for advanced
    plasma free boundary simulation," (2025). https://arxiv.org/html/2501.05956
13. P. Merkel and E. Strumberger, "Linear MHD stability studies with the
    STARWALL code," arXiv:1508.04911 (2015). https://arxiv.org/abs/1508.04911
14. P. J. Bonofiglo, L. Chacón, et al., "Implementation and verification of
    toroidal resistive wall boundary conditions in the PIXIE3D MHD code using
    a boundary integral method," arXiv:2606.05446 (2026).
    https://arxiv.org/abs/2606.05446
15. L. Spinicci et al., "Nonlinear verification of the resistive-wall boundary
    modules in the SPECYL and PIXIE3D magneto-hydrodynamic codes for fusion
    plasmas," AIP Advances 13, 095111 (2023).
    https://pubs.aip.org/aip/adv/article/13/9/095111/2910703/
16. M. S. Chance, "Vacuum calculations in azimuthally symmetric geometry,"
    Phys. Plasmas 4, 2161 (1997).
    https://pubs.aip.org/aip/pop/article/4/6/2161/263192/
17. C. Hansen et al., "TokaMaker: an open-source time-dependent Grad-Shafranov
    tool for the design and modeling of axisymmetric fusion devices,"
    arXiv:2311.07719 (2023). https://arxiv.org/pdf/2311.07719
18. "FGE: a fast free-boundary Grad–Shafranov evolutive solver," Plasma Phys.
    Control. Fusion (2026). https://iopscience.iop.org/article/10.1088/1361-6587/ae56b7
19. R. Albanese and G. Rubinacci, "Integral formulation for 3D eddy-current
    computation using edge elements," IEE Proc. A 135, 457–462 (1988).
    https://digital-library.theiet.org/doi/10.1049/ip-a-1.1988.0072
20. R. Albanese, G. Rubinacci et al., "A fast 3D eddy current integral
    formulation," COMPEL 20 (2001).
    https://www.emerald.com/insight/content/doi/10.1108/03321640110383221/full/pdf
21. R. A. James, "The solution of Poisson's equation for isolated source
    distributions," J. Comput. Phys. 25, 71–93 (1977).
    https://www.sciencedirect.com/science/article/abs/pii/0021999177900134
22. R. W. Hockney and J. W. Eastwood, *Computer Simulation Using Particles*,
    McGraw-Hill (1981) — zero-padded convolution for isolated-source BCs.
23. "FFT-based free space Poisson solvers: why Vico-Greengard-Ferrando should
    replace Hockney-Eastwood," arXiv:2103.08531 (2021).
    https://arxiv.org/pdf/2103.08531
24. J. Barnes and P. Hut, "A hierarchical O(N log N) force-calculation
    algorithm," Nature 324, 446 (1986).
25. J. K. Salmon and M. S. Warren, "Skeletons from the treecode closet,"
    J. Comput. Phys. 111, 136 (1994) — rigorous multipole-acceptance error
    bounds; pathological near-edge cases.
26. A. Grama, V. Kumar, and A. Sameh, "Analyzing the error bounds of
    multipole-based treecodes," Proc. SC'98.
    https://people.engr.tamu.edu/sarin/Publications/SC98-87070019.pdf
27. "Quasi-static free-boundary equilibrium of toroidal plasma with CEDRES++,"
    J. Plasma Phys. 81 (2015).
    https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/abs/quasistatic-freeboundary-equilibrium-of-toroidal-plasma-with-cedres-computational-methods-and-applications/4358A3CED81708EBBF4BD64FC8D290A1
28. "Validation of the static forward Grad–Shafranov equilibrium solvers in
    FreeGSNKE and Fiesta using EFIT++ reconstructions from MAST-U," Phys. Scr.
    (2025). https://iopscience.iop.org/article/10.1088/1402-4896/ada192
29. "CRATOS-GS: a free-boundary, hierarchical adaptive mesh refinement
    Grad–Shafranov solver," (2025).
    https://www.researchgate.net/publication/395801680
30. M. Merkel et al., "Parallelization of JOREK-STARWALL for non-linear MHD
    simulations including resistive walls," arXiv:1609.07441 (2016).
    https://arxiv.org/pdf/1609.07441
31. "A fast Poisson solver of second-order accuracy for isolated systems in
    three-dimensional Cartesian and cylindrical coordinates," ApJS (2019) —
    modern James-method implementation including cylindrical coordinates.
    https://iopscience.iop.org/article/10.3847/1538-4365/ab09e9
32. A. Le et al., "Hybrid-VPIC: an open-source kinetic/fluid hybrid
    particle-in-cell code," Phys. Plasmas 30 (2023) — B = B0 + B1 splitting,
    fixed vacuum B0 in ghost cells for open boundaries.
    https://arxiv.org/pdf/2305.05600
33. "A fully electromagnetic hybrid PIC-fluid model for predictive fusion
    neutron yield in dense plasma focus," arXiv:2604.09032 (2026).
    https://arxiv.org/pdf/2604.09032
34. F. Capolino, D. R. Wilton, W. A. Johnson, "Efficient computation of the 2D
    periodic Green's function using the Ewald method," J. Comput. Phys. 219,
    899 (2006). https://www.sciencedirect.com/science/article/abs/pii/S0021999106003354
35. F. T. Celepcikay et al., "Choosing splitting parameters and summation
    limits in the numerical evaluation of 1D and 2D periodic Green's functions
    using the Ewald method," Radio Sci. 43 (2008).
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007rs003820
