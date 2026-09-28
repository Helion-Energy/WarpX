# Eulerian thermal runtime prerequisites

`hybrid_pic_model.electron_energy_mode` is `legacy` by default and also accepts
`decoupled_jfnk` and `coupled_jfnk`. New modes enable electron energy and reject
an explicit `solve_electron_energy_equation=0`. They require the Darwin theta
hybrid driver, RZ single level without EB, gamma>1 and a hard common
Ohm/capacity floor (`qdsmc_te_n_floor=n_floor`, `n_floor_smooth_width=0`).
The kinetic moment adapter accepts both actual RZ axis quadratures and uses
separate pressure versus deposited-current boundary images.

HybridPICModel owns accepted cell U_e in J/m^3 in the existing field registry.
It constructs no QDSMC marker container in either new mode. After actual charge
deposition and before initial E/Je, the startup hook seeds U by native
restriction of the nodal physical product n_eff*kB*T_initial/(gamma-1).
The initial polytropic profile honors its own density floor, gamma, reference
and pedestal option. Re-entry after restoration/initialization refreshes Te/Pe
from accepted U and current deposited rho without thermal reseeding.

SetElectronThermalTrial borrows const independent Te and Ohm-pressure MultiFabs;
ClearElectronThermalTrial(s) releases them. ElectronTemperatureForSolve and
ElectronPressureForSolve select these views for resistivity/E/longitudinal-E.
They neither overwrite accepted fields nor own the supplied scratch. Every
required ghost layer must be supplied, and the moment adapter fills those
layers and corners. The outer driver owns scratch lifetimes and must clear
views before acceptance, rollback or destruction. Accepted refresh rejects an
active trial view.

Checkpoints serialize only accepted thermal U plus explicit metadata for mode,
coordinate, units, gamma, common density floor and actual axis convention.
Missing/mismatched U metadata refuses silent reseeding; decoupled/coupled
switches with matching state conventions are permitted. A checkpoint refuses
active thermal trial views before writing any data. The pre-existing complete
Darwin/inertia restart rejection remains: U serialization alone does not
qualify A/B/Je/circuit restart state.

The standalone CPU fixture links the production runtime helper, moment map,
MultiFabRegister and VisMF. It checks real parameter parsing/guards, conservative
product seeding, four complete ghost layers, accepted/trial pointer and value
purity, mode metadata, and real registered-U checkpoint round trips. Tests cover
multi-box and single-box MPI2 layouts (the latter has an empty rank), physical
caps and periodic z, and both native axis quadratures. Cartesian compilation
and helper tests preserve existing support; new 3D runtime startup explicitly
rejects until transport is implemented and qualified.

Configure inside the source tree with the matching CPU AMReX package:

```sh
cmake -S Examples/Tests/implicit/thermal_runtime_fixture -B build-runtime-rz \
  -DAMReX_DIR=/path/to/rz/AMReX -DTHERMAL_RUNTIME_GEOMETRY=RZ
cmake --build build-runtime-rz -j2
ctest --test-dir build-runtime-rz --output-on-failure
```

This slice does not wire either implicit advance. Full application integration,
combined Jv, convergence, particle rollback and restart acceptance are driver
qualification gates owned by the integrating caller.
