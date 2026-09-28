# Eulerian checkpoint physical-model contract

This fixture compiles the production resolver and metadata reader/writer. It
checks lossless v2 round trips, same-mode and coupled/decoupled compatibility,
effective bounded-NR resolution, inactive accuracy canonicalization, optional
reference independence, and RNG purity. Negative processes prove rejection of
changed physical partners/active conventions/accuracy budgets, missing or old
metadata, malformed/duplicate/trailing fields, unsupported source/absorption/
history semantics, and invalid parser combinations. Existing gamma, floor,
axis and legacy-mode incompatibilities remain checked.

The history tag explicitly says the endpoint absorption history is unqualified;
recording a version does not qualify its physics. This is a component fixture,
not a full application restart replay. The schema covers the new source-partner
and endpoint policies plus the existing U contract; it does not fingerprint every
resistivity, conduction or collision parser.

Configure with `-DAMReX_DIR=<native AMReX config>` and optionally
`-DTHERMAL_MODEL_GEOMETRY=3D`, build, then run `ctest --output-on-failure`.
Both MPI ranks independently validate the broadcast-equivalent metadata. Only
positive cases should be launched under MPI; rejection cases deliberately abort.
