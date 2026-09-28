# Accepted density-control state

The `decoupled_jfnk` and `coupled_jfnk` modes preserve accepted cell electron temperature when the common density floor changes. The canonical capacity is the native deposition-quadrature restriction of nodal `max((rho+rho_ped)/q_e,n_floor)`. With old/new capacities `n0,n1`, the accepted update is `U1=U0*n1/n0`. This is an external control inventory adjustment, not physical heating in the nonlinear residual.

The update records two signed physical-volume integrals, first changing the common floor with the old pedestal and then changing the tracked pedestal. Their sum equals the actual change of accepted thermal inventory. An unchanged floor does not write U, Te or Pe or increment the epoch. New modes keep `qdsmc_te_n_floor==n_floor` and leave `qdsmc_n_floor` independent. Legacy setters retain their existing conversion/gate behavior. An untracked uniform pedestal remains at its original floor on restart.

`SetHybridDensityFloor` and `SetQdsmcDensityFloor` are accepted-boundary APIs. BeginStep freezes the density-control epoch and refreshes all moment, conduction-activity and viscous contexts. Setters reject while a thermal step is open. They do not alter force/deposition denominators inside a Jv evaluation.

`density_control.dat` stores original configuration separately from live floors, monotone epoch, cumulative floor/pedestal inventory entries and controller configuration/last/EMA/sample-step. `electron_energy.dat` compares the original common floor. Restart restores live control values before recovering Te from U. The original boot arguments remain the restart inputs; changing them, losing the state file, or changing a registered controller is an explicit error. Older experimental Eulerian checkpoints without this control file are not silently reseeded.

The maintained stroke adapter uses `configure_hybrid_pic_density_controller` after initialization and `commit_hybrid_pic_density_controller` on every sample, even when the deadband suppresses a floor move. It preserves the default absence of gate tracking and the peak EMA arithmetic. Its patch is delivered separately against the pinned maintained hetools deck. The legacy controller path remains unchanged.

Native tests:

- `test_electron_density_control_{rz,3d}` checks pure remap temperature invariance, exact no-change identity, signed inventory, invalid trial rejection, and strict metadata roundtrip/configuration.
- `test_electron_density_control_runtime_{rz,3d}` runs accepted particle/field steps, updates control state, checks temperature and inventory immediately at each boundary, and can write/replay full native checkpoints with original input arguments. MPI2 cases include seams and empty ranks.

Checkpoint validation distinguishes exact serialized restoration and exact controller/epoch continuation from the physical inventory accumulated afterward. Particle redistribution can change deposited fields at floating-point roundoff; subsequent inventory integrals are therefore compared at the same numerical scale, while the controller state remains exact. The registered tests include a deadband checkpoint where EMA and the last applied floor differ.
