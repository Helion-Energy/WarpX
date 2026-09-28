# Native implicit endpoint exchange audit

The fixture tests the private native endpoint bridge, two-pass species moment decomposition, pure midpoint electron drift, and overlapping drag guard. Inputs are actual native SoA tile arrays with immutable saved Cartesian x_n/u_n. It exercises two charged species and excluded alpha, unequal particle weights, twelve ring angles, MPI seams, periodic endpoints and empty ranks. Endpoint positions/momenta use the actual implicit 2*midpoint-old formula before private native redistribution and NGP indexing. Invalid views are rejected collectively without input or RNG mutation.

Build with a matching RZ MPI AMReX package:

```sh
cmake -S Examples/Tests/implicit/endpoint_exchange_fixture -B build-endpoint-exchange -DAMReX_DIR=/path/to/AMReX -DCMAKE_BUILD_TYPE=Release
cmake --build build-endpoint-exchange -j4
ctest --test-dir build-endpoint-exchange --output-on-failure
mpiexec -n 2 build-endpoint-exchange/test_endpoint_exchange test.case=moments test.max_grid_size=4
```

Cases are endpoint, moments, purity, invalid, guard and drift. The two layouts use max_grid_size=4 and64. Physical geometry is single-level RZ without EB. Existing Cartesian code is unchanged; native bridge translation units must also compile in3D/CUDA, which is not a claim of full3D source qualification.

The audit is enabled only by implicit_evolve.thermal.expected_ou_audit=relativistic or bounded_nr. It leaves electron Q and its frozen source diagonal unchanged. It evaluates prepared source rates against privately reconstructed endpoint particles on every residual/Jv, then checks the result against the actual accepted endpoint before the one stochastic kick. No random draw is made in the audit. A configured Te_shunt_threshold is historically inactive in this implicit lane; explicitly active shunt transforms remain rejected by AcceptedIonExchange. A separately registered hybrid_resistive_drag on an active, nonexcluded OU species is rejected because it duplicates an unqualified bulk momentum exchange.

All reported moment energies use the quadratic proper-u NR convention. Conditional population thermal work is measured about the deterministic post-kick mean. Finite-sample mean-motion energy shifts the partition of measured thermal/bulk work; it is not an additional heat debit. Full lab energy is not subtracted from electron heat. Actual projected eta-electric work remains a separate integration obligation; full-minus-nores does not isolate it while relaxation is active. See ENDPOINT_FORCE_WORK_AUDIT.md in the worker artifact package.


The optional physical variant is `implicit_evolve.thermal.expected_ou_thermal_partner=population_bounded_nr`. It replaces only electron relaxation. Additional cases partner, finite_step, finite_step_species, derivative and noise test its physical volume integral, explicit convention rejection, one/two-species completed-time second-order frozen-rate limit, live nonlinear-nu derivative and finite-sample partition. Twenty-two serial cases plus matching MPI2 coverage are qualified. The full force/collision split and Ti-dependent fixed-old-rate temporal order remain separate obligations.

`implicit_evolve.thermal.expected_ou_relativistic_reference=1` optionally evaluates one accepted relativistic reference pass before RNG; it defaults off, reports particle count/time, and never changes Q or momentum. Population partner production residuals remain bounded NR. Only double field/particle precision is currently qualified and enforced for this variant.
