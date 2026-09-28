# Kinetic-to-Eulerian thermal moment adapter

`KineticThermalMoments` provides independent GPU-native scratch for a single
physical RZ m=0 or Cartesian 3D mesh. It does not change particles, accepted
fields, diagnostics, RNG state, thermal transport, or the implicit advance.
The caller supplies old, nonlinear-stage, and true particle-endpoint inputs
separately. Do not extrapolate temperature or reuse an old density denominator.

## Inputs and outputs

| Quantity | Layout | Units / convention |
|---|---|---|
| `charge` + selected component | Fully nodal | Positive ion charge, C/m^3, after deposition/MM response and synchronization |
| Optional `pedestal` | Fully nodal | Nonnegative charge, C/m^3 |
| Optional `energy` | Primal cell | Electron internal energy U, J/m^3 |
| `ion_current` | Native Yee vector | Signed conventional Ji, A/m^2 |
| `current` | Native Yee vector | Signed Je, or total plasma Jp; not nodal Je history |
| `RawNumberDensity()` | Cell | Rrho(charge)/qe, m^-3 |
| `NodalNumberDensity()` | Node | max((charge + pedestal)/qe, n_floor) |
| `NumberDensity()` | Cell | Rrho of the effective nodal density |
| `FaceIonCurrent(d)`, `FaceElectronCurrent(d)` | Primal face normal to d | Conservative transferred current, A/m^2 |
| `FaceNumberDensity(d)` | Same face | Arithmetic mean of adjacent effective cell densities |
| `FaceElectronVelocity(d)` | Same face | -Je_face/(qe n_face), m/s |
| `CellTemperature()`, `NodalTemperature()` | Cell / node | Kelvin |
| `NodalPressure()` | Node | n_node kB T_node, Pa |
| `OhmPressure()` | Node | The product followed by deterministic physical pressure images |

RZ native rho/Jtheta are `(1,1)`, Jr is `(0,1)`, Jz is `(1,0)`;
target r/z faces are `(1,0)` and `(0,1)`. Here 1 means nodal.
In 3D, a native current is cell-centered in its vector direction and nodal
transversely; a target face has the opposite staggering.

`Evaluate` derives every density, denominator, temperature, and pressure from
its current arguments. A combined Jv therefore retains both U and rho
responses. No frozen coefficient is hidden in this class. Distinct instances
hold independent old/stage/endpoint scratch. Getter references last until the
next evaluation on that instance. Invalid finite-state/admissibility input
returns false without changing inputs; outputs after false are only scratch.
Layout, unsupported geometry and boundary mistakes fail explicitly.

Density regularization follows the nodal Ohm convention: add pedestal and
floor **before** restriction. The physical active gate uses raw cell density:
raw n > active_density_floor, or raw n > 0 when a pedestal pointer is supplied
or halo_unfreeze is enabled. Active status does not erase current. Negative
raw charge can occur in a trial and is regularized for capacity, not silently
changed in `RawNumberDensity`. The linear continuity identity below applies
to raw charge, not to a nonlinear floored density or a prescribed pedestal.

For advection without primitive reconstruction, the supplied face density is
exactly the arithmetic density pair used by central thermal advection. For a
stage with reconstructed primitives, use the conserved number flux
`-FaceElectronCurrent(d)/qe`; let that stage divide by **its same reconstructed
face density** and use the resulting velocity for advection and compression.
An independent nodal-face average in that denominator would spoil isothermal
contact consistency. The electron continuity context also requires consistent
ion response and solenoidal total plasma current; the adapter does not impose
those global equations.

## Commuting map in qualified RZ geometry

Let r_i = i dr, z_j = j dz and C_(i,j) be the primal cell. Its physical volume
is `pi (r_(i+1)^2 - r_i^2) dz`. A nodal dual volume is clipped to the physical
domain, including the axis disk `pi (dr/2)^2` and half axial caps. All
restrictions below act on already synchronized, post-image fields.

Define

```
a_i = (r_i + dr/4)/(2 r_i + dr)
(Rphysical q)_(i,j) = 1/2 [a_i (q_(i,j) + q_(i,j+1))
                       + (1-a_i) (q_(i+1,j) + q_(i+1,j+1))].
```

These are the exact overlaps of clipped nodal dual volumes with the cell,
divided by cell volume. They sum to one, are positive, and give a_axis = 1/4.
Consequently the physical integral and constants are preserved. The bare
`RestrictNodalScalar` also applies to a selected source component in W/m^3;
it has no charge conversion, floor, active mask or pedestal.

For r_i > 0 let s_minus = (r_i - dr/4)/(2 r_i), s_plus = 1-s_minus. Define

```
(RJ Jr)_(i,j+1/2) = 1/2 sum_(b=0,1) [
    s_plus  (r_i-dr/2)/r_i Jr_(i-1/2,j+b)
  + s_minus (r_i+dr/2)/r_i Jr_(i+1/2,j+b) ].
(RJ Jr)_axis = 0.

(RJ Jz)_(i+1/2,j) = 1/2 [
    a_i     (Jz_(i,j-1/2)   + Jz_(i,j+1/2))
  + (1-a_i) (Jz_(i+1,j-1/2) + Jz_(i+1,j+1/2)) ].
```

The displayed current formulas apply at interior radial faces. At the wall,
replace both s coefficients by 1/2. In the Jz formula multiply the coefficients
of node i by C_i and node i+1 by C_(i+1), where

```
C_0 = 4/3 for Verboncoeur=true, otherwise 1
C_N = N/(N-1/4) at the outer radial wall
C_i = 1 elsewhere
Rnative q = Rphysical(C q).
```

`RestrictNativeMoment` is this explicit native deposition-quadrature map. It
preserves deposited extensive charge, whereas `RestrictNodalScalar` preserves
physical clipped-dual source integrals. The two maps have deliberately distinct
names. Density and the initial nodal product n*kB*T/(gamma-1) use Rnative.

These are flux restrictions, not point averages of vector components. The
actual particle continuity divergence is

```
(Dparticle J)_(i,j) = ((r_i+dr/2) Jr_(i+1/2,j)
                       -(r_i-dr/2) Jr_(i-1/2,j))/(r_i dr)
                     + (Jz_(i,j+1/2)-Jz_(i,j-1/2))/dz,   i > 0
(Dparticle J)_(0,j) = (Verboncoeur ? 3 : 4) Jr_(1/2,j)/dr + Dz Jz.
(Dcc F)_(i,j)       = (r_(i+1) Fr_(i+1,j+1/2)
                       -r_i Fr_(i,j+1/2))/((r_i+dr/2) dr)
                     + (Fz_(i+1/2,j+1)-Fz_(i+1/2,j))/dz.
```

Matching the three radial edge coefficients gives
`Dcc RJ = Rnative Dparticle`; transverse half averages commute. For the axis cell,
both sides have radial coefficients `(5/8 Jr_(1/2) + 9/8 Jr_(3/2))/dr`.

Production PEC normal-current images preserve rJ at the radial wall, so at
r_N, `(RJ Jr)_wall = (r_N-dr/2)/r_N Jr_(N-1/2)`. This is generally nonzero:
particle/cloud charge exchange at PEC remains in the surface ledger. Axial
PEC/PMC images give their normal even/odd parity, respectively. Exact scalar
physical-volume accounting uses owner masks for shared/periodic nodal points;
MPI seams never get counted twice.

The constructor requires axis at r=0, radial PEC/PMC, explicit effective
post-deposition axial PEC/PMC/periodic images, one azimuthal mode, a complete
single mesh, matching BoxArray order/DM, and no EB. Pressure and current images
are separate options: reflecting or thermal particles impose the PMC current
image even at a PEC field wall, without changing the pressure constraint.
WarpX's `Neumann` field enum is an alias for PMC. The maintained configuration
is radial PEC with reflecting particles and axial PMC with absorbing particles.
Annular geometry and other dimensions fail explicitly. Cartesian 3D is the
direct tensor-product half-weight counterpart and has an independent build;
this does not qualify the separate Eulerian transport runtime in 3D.

At a reflecting radial wall the actual metric image is
`Jr_guard = -(R-dr/2)/(R+dr/2) Jr_inner`. The symmetric integrated wall map gives
exactly zero. The PEC current image has the positive sign and gives nonzero
`(R-dr/2)/R Jr_inner`; no boundary exchange is artificially removed. The native
wall quadrature is `pi R dr dz`, whereas the clipped physical wall dual volume
is `pi (R dr-dr^2/4) dz`. Multiplying by C_N converts these extensive measures.

## Both actual axis conventions

The existing Darwin particle operator already supports both axis volume
conventions. `MassMatrixDensityProjection::ComputeDivergence`, lines18–49 at
the base, selects axis factor3 when `UseVerboncoeurAxisCorrection()` is true
and4 when false. `ComputeDivE.cpp:190` has Maxwell factor4 and is not the
particle continuity operator for the corrected convention.

`WarpXPushFieldsEM.cpp:1414,1574,1776,1840` divides nodal axis rho/Jz by
`pi dr/3` or `pi dr/4` after Cartesian deposition. Jr is unchanged. Thus the
native quadrature at the axis is `pi dr^2/3 dz` or `pi dr^2/4 dz`. C_0 converts
that native extensive moment into the physical dual measure before overlap
restriction. This does not alter the configured axis flag or any deposit.

A constant nodal density maps to 13/12 of that density in the first cell for
corrected axis3 and `1+1/[4(2N-1)]` in the final radial cell for either convention.
Interior cells preserve constants. This is an explicit quadrature conversion,
not a pointwise density interpolant. The wall bias is O(dr/R); the axis bias is
localized to a volume O(dr^2). Both integrated biases are O(dr^2). For constant
nodal density, total native quadrature exceeds physical volume by
`pi dr^2 Lz [1/4 + (Verboncoeur ? 1/12 : 0)]`. Tests independently check the
predicted local biases and the extensive native integral. Mapping initial n*T
with exactly the same map ensures uniform T remains uniform despite these
boundary density differences. Nodal pressure uses the original nodal density,
not the quadrature-rescaled node value.

The real-particle fixture tests shape3 endpoints/current under both configured
axis factors and both radial current images, then verifies mapped continuity
and particle-charge accounting. For corrected axis3 it retains Maxwell4 only
as a negative control; the mismatch equals Jr_axis/dr. It must not be reported
as a production particle/MM defect. This implementation supersedes the first
bounded physical-overlap adapter, which explicitly rejected corrected axis3
and radial reflection. The physical source restriction remains unchanged.

## Temperature and pressure

Cell T = `(gamma-1) U/(kB n_cell)` is formed before interpolation, following
the MHD temperature-primary pattern. A convex arithmetic cell-to-node map
uses periodic communication and even physical scalar images. It preserves
constant T even across density/pedestal/floor jumps, and bounds nodal T by its
neighboring cell values. Nodal P is then `n_node kB T_node`, using the **same
trial** effective density. Density, energy, temperature, and pressure must be
finite; temperatures below the requested floor reject without clipping U.

At a PEC boundary, the existing field-pressure constraint copies adjacent
interior P onto a boundary node. It need not equal that boundary node's local
n kB T. Thus `NodalPressure` and `OhmPressure` are separate, explicit outputs;
the latter applies a deterministic gather image from immutable product
pressure and is checked against production `WarpX_PEC.cpp`. Axis/PMC scalar
ghosts are even. `nodal_ghosts` requests a fully filled width for trial Te/Pe,
including corners; production chooses at least the accepted registry width.
All field data stay on device; reductions only return scalar
admissibility checks.

## Native validation and replay

The standalone CMake fixture links the production adapter, production PEC
boundary implementation, production AMReX, real shape-3 charge and implicit
Esirkepov current deposition kernels, and production Yee derivative headers.
It does not link or instantiate the full WarpX singleton. Its scalar
inverse-volume folding reproduces the production expressions above, and its
particle divergence uses the exact RZ expression from
`MassMatrixDensityProjection::ComputeDivergence` with either configured axis
factor. This distinction is explicit; tests are not a full-application run.

The five native cases cover arbitrary-current commutation and metric surface
flux, real shape-3 trajectories near axis/wall/caps/seams, charge-volume
accounting, constant T at sharp density jumps, current charge signs, true
endpoint inputs, bounded reconstruction, independent analytic U/rho/mixed
pressure directional derivatives, live velocity denominators, inactive
floors and live U response, and exact A/B/A/all-input/ghost purity. RZ includes
a failing Cartesian point-average negative control. Rejection cases validate
unsupported geometry/mode options, missing boundary qualification, invalid
floor/layout and accidental reuse of owned outputs as inputs.

For each dimension the matrix uses maximum box sizes 4 (many boxes) and 64
(one box), physical or periodic transverse caps, serial and MPI2. MPI2 with
one box gives an empty rank. RZ also tests both actual axis conventions and
reflecting particle overrides at the radial PEC wall. Configure with the corresponding dimension's
CPU AMReX package, including particles:

```sh
cmake -S Examples/Tests/implicit/thermal_moments_fixture -B build-moments-rz \
  -DAMReX_DIR=/path/to/cpu-rz-AMReX/lib/cmake/AMReX -DMOMENT_GEOMETRY=RZ
cmake --build build-moments-rz -j4
ctest --test-dir build-moments-rz --output-on-failure
mpiexec -n 2 build-moments-rz/test_thermal_moments \
  test.case=particles test.max_grid_size=4 test.periodic=0 \
  test.corrected_axis=1 test.reflecting=1
```

Use `MOMENT_GEOMETRY=3D` and a 3D AMReX package for the Cartesian counterpart.
The companion hub report records exact local package paths, every command,
source/build hashes, MPI matrix, and CPU/CUDA compile-only evidence. CUDA
compilation is qualification of the real adapter TU, not a claim of GPU runs.
