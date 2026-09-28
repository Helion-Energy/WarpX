# Accepted source-work gate

This RZ fixture sends actual EulerianThermalSources applied work through the same pure
acceptance policy used by DarwinThermalAdvance::PrepareAcceptance. Smooth heating and
signed cooling preserve the physical integral; isolated viscous and hyperresistive orphan
work rejects. The cancellation case has equal opposite physical theta-edge work, a zero
signed orphan integral, and a positive absolute orphan integral: acceptance still rejects.
MPI split boxes and a single box/empty rank use the same cases. Policy inspection preserves
the source ledger and RNG state. Nonfinite and invalid absolute ledgers also reject.

Build against an installed double-precision RZ AMReX package:

    cmake -S . -B build -DAMReX_DIR=<package>
    cmake --build build -j 2
    ctest --test-dir build --output-on-failure

The fixture is a source/policy integration test. Native two-mode application evidence and
DTA acceptance ordering are recorded in the worker handoff, not inferred from this test.
