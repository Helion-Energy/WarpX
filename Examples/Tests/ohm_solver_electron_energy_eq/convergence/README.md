# QDSMC energy-equation convergence harness (research tooling)

Campaign plan: `Docs/qdsmc_energy/QDSMC_ENERGY_2ND_ORDER_PLAN.md` (this
branch). Handoff: `Docs/qdsmc_energy/QDSMC_HANDOFF.md`. Branch:
`qdsmc_energy_leapfrog`, worktree `~/src/WarpX-qdsmc`.

Environment: dedicated venv `~/.env/warpx-qdsmc` (pywarpx pip-installed from
this worktree's `build/`; keeps the main `~/.env/warpx` + `build_eb` stack
untouched). Build: 2D, OMP, **MPI=ON (openmpi5) + openPMD (parallel HDF5)**,
EB=ON, QED=OFF — MPI/openPMD are required by the CI-matrix decks.

Every run needs:
```
LD_LIBRARY_PATH=/usr/local/openmpi5/lib:/usr/local/hdf5/lib OMP_NUM_THREADS=2
```

Rebuild:
```
cd ~/src/WarpX-qdsmc && PATH=/usr/local/openmpi5/bin:$PATH \
    /usr/bin/cmake --build build -j 8 --target pip_install
```

## Scripts

- `qdsmc_advection_test.py` — single-run advection deck: at_rest / translate
  / rotate modes; V_e controlled exactly via heavy drifting ions with B=0;
  Te blob poked via the `particleinjection` callback (fires after the step-0
  Pe fill in `HybridPICInitializeRhoJandB`, which would otherwise overwrite
  it). Flags: `--advance euler|leapfrog|pc`, `--grad-deposit 0|1`.
- `run_baseline.py` — advection sweep driver: at-rest smoothing (E5/E6),
  translate dx order vs exact, rotate dt self-convergence, bake-off
  same-discretization diffs. Writes `baseline_results.md`.
- `qdsmc_conduction_test.py` — single-run conduction deck (Thrust C): uniform
  n, ions at rest, uniform B (keep B0 whistler-stable: `omega_wh(k_max) *
  dt_sub <~ 0.1`; default B0=2e-5 T is invariant under the parabolic sweep);
  aligned (1D Gaussian spread along B || z) and tilted (30 deg, 2D blob,
  chi_perp=0) modes vs exact wrapped solutions; measures rel L2, sigma^2
  growth, spurious perpendicular chi, conservation. Flags: `--npts <par>
  [<perp>]`, `--flux-limit`, `--max-hop`, `--grad-deposit`.
- `run_conduction.py` — G3 sweep driver: `aligned2` (npts=3), `aligned2_np2`
  (npts=2), `nograd` (correction-off control), `tilted`; parabolic
  refinement (dt ~ dx^2, nsteps = 32(N/32)^2). Caches npz per case.
- `run_ci_matrix.py` — 3 CI cases (adiabat / joule / qei) x 3 schemes via a
  scheme-injection wrapper; runs each test's analysis with CI tolerances.
- `regime_survey.py` — C.7 stiffness table (no runs). PLACEHOLDER deck
  numbers; replace with real liftoff/annulus/compression values before
  gating G3a.

Output dirs (`baseline_out/`, `cond_out/`, `ci_matrix*/`, `*.log`) are
regenerable and stay untracked.
