# PR: QDSMC cylindrical volume weighting for RZ / RCYLINDER

## Summary

QDSMC electron-energy markers now carry **extensive** electron count
`N_e = n_e * V_phys(r)` with the cylindrical metric

\[
V_\mathrm{phys} =
\begin{cases}
(\prod_d \Delta x_d)\, 2\pi r & r > 0 \\
(\prod_d \Delta x_d)\, \pi\cdot f_\mathrm{axis} & r = 0
\end{cases}
\]

in **RZ** and **RCYLINDER** (Verboncoeur axis factor \(f_\mathrm{axis}=1/3\),
matching `ApplyInverseVolumeScalingToChargeDensity` / FieldEnergy).

Previously markers used Cartesian cell volume \(\prod \Delta x\) only, so
entropy transport across radii did not conserve extensive electron number
under the cylindrical measure.

## Code changes

| File | Change |
|---|---|
| `Source/Fluids/QdsmcParticleContainer.cpp` | `qdsmc_physical_volume()`; `SetK` uses `N=n*V_phys`; `DepositField` deposits extensive `N` |
| `Source/Fluids/QdsmcParticleContainer.H` | Doc update for `DepositField` |
| `Source/FieldSolver/.../HybridPICModel.cpp` | `QDSMCUpdateTe`: weights are extensive `N` (no extra `* cell_volume`) |
| `Examples/Tests/ohm_solver_electron_energy_eq/` | New RCYLINDER adiabat CI inputs + analysis |

Cartesian behavior is equivalent (constant \(V\)).

## Geometry policy

- **RCYLINDER**: energy equation allowed (already unblocked on this branch; still blocked on stock upstream — this PR is the physics fix that justifies unblocking).
- **RZ**: same volume path; still no hard refuse on BLAST upstream (Helion had a separate “not validated” guard).
- **RSPHERE**: still refused.

## Tests

1. **CI:** `test_rcylinder_ohm_solver_electron_energy_adiabat`  
   - Radial velocity perturbation, sources off, interior adiabat  
   - Manual result: `median_rel_err ~ 2.5e-5`, `max_rel_err ~ 6e-4` (PASS)

2. **Stationary smoke:** uniform RCYL plasma, 50 steps, Te stays \(50\pm0.2\,\mathrm{eV}\).

3. **2D Cartesian regression:** existing `test_2d_ohm_solver_electron_energy_*` should still pass (constant volume path).

4. **Use-case (z-pinch 40 ns):** production hybrid deck.  
   - Volume fix is **correct** for conservation.  
   - The warm Te foot (4–10 mm) is **not removed** — it is high-entropy heat
     from QDSMC \(V_e\) transport relative to kinetic electrons, not the old
     missing-\(2\pi r\) factor alone. Foot mean Te: baseline ~5.7 eV → fix ~8.0 eV
     (outer sheath markers carry more physical weight when \(V\propto r\)).

## How to run CI locally

```bash
# RCYLINDER build
exe=build_rcyl_magdiff/bin/warpx.rcylinder
cd /tmp && mkdir -p qdsmc_rcyl && cd qdsmc_rcyl
mpirun -np 2 $exe \
  $WARPX_DIR/Examples/Tests/ohm_solver_electron_energy_eq/inputs_test_rcylinder_ohm_solver_electron_energy_adiabat
python3 $WARPX_DIR/Examples/Tests/ohm_solver_electron_energy_eq/analysis_adiabat_rcylinder.py \
  --diag-dir diags --tol-median 0.08 --tol-max 0.35
```

## Related prior work

- Per-species \(\rho\) volume scaling for Joule / \(Q_{ei}\) (tomzhu / Prabhat)
- Ion temperature deposition in cylindrical coordinates
- Unblock RCYL energy equation (`ac9d7df`) — incomplete without this volume fix
