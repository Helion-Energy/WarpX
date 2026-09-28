# Frozen longitudinal displacement-inertia correction

`DarwinLongitudinalSchur` solves the frozen longitudinal block that is stiff at
short time steps. It does not replace the full Darwin/particle/thermal residual,
accept an implicit step, or change boundary conditions. The new-mode runtime
remains RZ-only; the generic Cartesian helper is separately tested in 3D.

At midpoint let `h=theta*dt`, `L=E_L`, `N` be the production Yee-to-node
interpolation, `M` its production node-to-Yee counterpart, and `P=G(DG)^-1 D`
the longitudinal projection. Displacement current gives

```
delta Je_node = -epsilon0 N(delta L) / h
kappa_node = w_taper epsilon0 m_e / (q_e rho_lim h^2)
delta E_inertial_node = -kappa_node N(delta L)
delta L_raw = -P M kappa N(delta L).
```

For `n_eff=2.1e18 m^-3`, physical electron mass and `dt=1e-12 s`, `kappa` is
about 598.49. The raw longitudinal outer iteration is then noncontractive.
The helper treats this frozen response implicitly:

```
(DG + DM kappa NG) delta_phi = D(L_raw - L_held)
delta_L = G delta_phi.
```

The operator uses native `ablastr::coarsen::sample::Interp`, native
cylindrical/Cartesian Yee divergence kernels, and the exact existing E_L
source images. RZ divergence uses the Maxwell/field axis factor **4**; this is
independent of the native deposited-charge axis3/axis4 quadrature. The two
operators must not be conflated. RZ storage uses compressed grid dimensions
`(r,z)` even though there are three physical vector components `(r,theta,z)`.

Physical edge ghosts are zero-extended except for the existing PMC vector
reflection. PEC uses homogeneous Dirichlet potential; PMC and the radial axis
use Neumann potential. These are the field/constraint images, not ion boundary
conditions or charge-deposition images. The nodal inertial response is
zero-extended before its native interpolation back to edges. Axis, wall, cap,
periodic seam and corner guards are covered by the fixture.

Only the preconditioner approximates the operator: nodal `kappa` is averaged
to cell centers and `1+kappa_cc` supplies an ordinary FD nodal elliptic MLMG
preconditioner. Flexible GMRES operates on the exact wider operator and
recomputes the true residual. No pointwise division replaces the Schur solve.
All field operations stay on AMReX MultiFabs; only scalar reductions cross MPI.
The Krylov norm counts each node once using the natural FD nodal quadrature.
An all-Neumann/periodic potential is solved modulo its constant using that
quadrature (`1/8` at the RZ axis, interior radial index, `(N-1/2)/2` at
a PMC wall, and half weights at finite axial caps).

## Caller contract

1. Construct from the actual single-level rectangular cell layout, distribution,
   geometry and explicit field BCs. Both actual WarpX RZ `Coord()==0` and an
   equivalent thermal `Coord()==1` are accepted. Set `output_ghosts` to the
   required phi/EL width. Set the actual physics flags: only `theta==0.5`,
   `djedt_only=true`, `bdf2=false`, no embedded boundaries and one RZ mode are
   qualified; unsupported forms fail explicitly.
2. After a successful inner field/thermal solve, freeze a nodal scalar `kappa`
   from **that current midpoint density**. Form `rho_lim` with the same native
   deposited charge, pedestal, smooth floor and taper as
   `ComputeElectronInertiaNodal`. Set `kappa=0` where raw charge is nonpositive,
   just as that routine does. Use effective electron mass when enabled.
   `Freeze` copies the valid coefficient into independent storage and does not
   consume caller ghosts. It returns false for negative/nonfinite values.
3. Save held phi and held EL before the ordinary raw longitudinal refresh.
   Grade its unchanged full physical constraint defect. If that gate passes,
   retain/restore the pair actually solved by the inner solver.
4. Otherwise call `Correct(raw_EL,held_EL)` and require `result.converged`.
   Update `phi=held_phi+CorrectionPotential()` and
   `EL=held_EL+CorrectionField()` together. The correction has an **additive**
   sign. It owns fully filled output guards. Inputs must be independent of
   these borrowed output views and remain unchanged through evaluation.
5. Preserve total E in the transverse warm start, and repeat the complete
   nonlinear solve and raw physical gate. Re-freeze kappa from each new
   accepted inner context. A successful frozen Schur solve alone is not a
   converged physical step. Do not silently reuse it as a frozen density
   denominator in the combined physical Jv.

`ApplyPotential` and `SolvePotential` expose the same nodal operator for
manufactured tests. They eliminate Dirichlet rows; the reported residual is the
weighted nodal L2 norm, without an overall mesh-volume normalization. A failed
solve does not commit anything; its private outputs must not be applied.

## Validation and replay

Configure this directory with `AMReX_DIR` and `SCHUR_GEOMETRY=RZ` or `3D`, then
build and run CTest. The AMReX package must match the selected dimensionality.
Native tests use independent analytic edge/node/metric compositions, two
manufactured directional fields, and a projected physical frozen source to
verify the same fixed point. They check owned-node norms, variable and inactive
coefficients, stiffness 0/0.06/6/600/60000, full guard images, exact A/B/A,
unchanged caller fields/poisoned guards, and explicit unsupported-form guards.
Box size4 exercises seams; size64 produces a single box and an empty second MPI
rank. The non-rejection test commands also run under MPI2. CUDA qualification
is compilation of the real production helper and fixture; no GPU execution.

The all-Neumann RZ reference projection needs explicit constant-quotient Krylov:
this pinned AMReX `MLEBNodeFDLaplacian` declares `isSingular=false`, and its raw
standalone MLMG solve can fail for radial-PMC plus periodic-z. The fixture uses
a zero-kappa quotient solve in that case, then independently applies the native
FD Poisson operator and checks its true residual. Ordinary fully nonlinear
`ComputeDarwinELong` retains its existing projection; this helper does not claim
to repair that separate all-Neumann application limitation. PEC-containing
reference projections use native MLMG directly.
