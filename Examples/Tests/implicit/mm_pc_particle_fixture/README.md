# Native Esirkepov MM-PC with particle Jacobian probes

This isolated RZ fixture links to a completed native WarpX build containing the
`NeedFullCurrentResponse()` allocation/build versus Jacobian-use split. It does
not modify production source or the application CMake targets. Use matching
headers and libraries when another worker is rebuilding.

```sh
python3 Examples/Tests/implicit/mm_pc_particle_fixture/build_native.py \
  --native-source /path/to/snapshot/source \
  --native-build /path/to/snapshot/cpu \
  --output build-mm-pc-native
python3 Examples/Tests/implicit/mm_pc_particle_fixture/run_suite.py \
  --binary build-mm-pc-native/test_mm_pc_particle \
  --output build-mm-pc-native/cpu-results
python3 Examples/Tests/implicit/mm_pc_particle_fixture/run_suite.py \
  --binary build-mm-pc-native/test_mm_pc_particle \
  --output build-mm-pc-native/mpi2-results --ranks 2
```

The native build snapshot needs `CMakeFiles/lib_rz.dir/flags.make`,
`CMakeFiles/app_rz.dir/link.txt`, generated `Source` headers and the referenced
static libraries. Dependency include paths in that build remain read-only.
Outputs must be new directories. `--case NAME` selects a single case. The MPI
runner uses two ranks and Open MPI's local TCP transport; all runs use one OpenMP
thread per rank.

`WarpX::InitData()` exercises the actual runtime allocation for direct
Jacobian-off/PC-on, both-on and effective-neither configurations. The derived
probe then calls native `PreRHSOp`, `PreLinearSolve`, `ApplyMassMatrices`,
`SyncMassMatricesPCAndApplyBCs`, and the native physical endpoint charge deposit.
The only nonlinear-solver stub supplies `pc_hybrid_pic` to select physical
`dJ_i/dE` units. No solver or preconditioner inverse is substituted or tested.

For each current component, a unit electric input at interior, axis, box seam,
wall and corner locations goes through native field boundaries and Yee-to-node
centering, the complete deposited response matrix, radial volume scaling,
communication and current boundary folding. Its value at the selected output
location independently checks the projected PC diagonal. The oracle does not
call the diagonal index/parity helper. The diagonal has SI units
`(A/m²)/(V/m) = S/m`; the particle time response is already in the deposit.
There is no additional electromagnetic `c² mu0 theta dt` factor for this Hall-PC
contract.

After freezing the full matrix and projected diagonal, signed **particle** Jv
probes must change both current and physical endpoint-average charge. Every
component and guard of all nine response bands and all three PC diagonals must
remain exactly unchanged. Returning to the original field checks A/B/A current
and density purity. Reprojection from the held bands must reproduce the held
PC exactly. A subsequent nonlinear push plus forced rebuild must change the
PC, proving that the freeze does not mute future anchor refreshes.

The 13 cases per rank count cover the three allocation modes on periodic one-
and multiple-box layouts and PMC ends, effective PC-none, no MM consumers, and
rejections for inconsistent width metadata and a truncated off-diagonal band.
AMReX may split the nominal one-box layout across MPI ranks; the executable
reports the actual box count. Reflecting particles and radial PEC are retained.
`fallback` models the post-allocation switch to full particle probes; it does
not call the thermal source fallback or qualify its source physics. `matrix`
starts with the Jacobian-use flag on, then explicitly turns it off for the
signed particle probes. These gates do not compare a complete coupled Jacobian,
solve with a Hall-PC inverse, or claim 3D/GPU runtime qualification.
