# Native random checkpoint regression

Draws from the actual AMReX host engines and ParallelForRNG, saves their state,
draws and disturbs both streams, then restores and requires byte-identical replay.
Separate cases reject missing, malformed, truncated and checksum-corrupt files.
The component supports CPU and CUDA; MPI launches run independent rank files.
CPU replay requires matching thread count and repeatable particle traversal.
CUDA state replay requires matching backend/library/device architecture.

The production hook runs only at accepted step boundaries and at the end of
restart initialization. It copies no random state inside residuals or Jv. This
format covers the particle Random/RandomNormal/ParallelForRNG path; AMReX's
separate FillRandom generator is not exposed and is not used by native WarpX.
Actual stochastic application and circuit mode-switch replay are separate gates.
