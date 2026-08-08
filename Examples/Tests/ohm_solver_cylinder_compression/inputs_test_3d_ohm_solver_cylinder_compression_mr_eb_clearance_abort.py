#!/usr/bin/env python3
#
# --- Negative test for the hybrid-PIC mesh-refinement + embedded-boundary
# --- clearance guard: run the cylinder-compression deck with a refinement
# --- patch deliberately grazing the flux-conserver wall (--patch-mode
# --- grazing) and assert that WarpX aborts at initialization with the
# --- actionable clearance message instead of running EB-blind coarse-fine
# --- operators through the wall cut band.
#
# The deck runs in a subprocess whose environment is scrubbed of MPI job
# variables, so the child initializes as an MPI singleton even when this
# driver itself is launched under mpiexec by CTest.

import os
import subprocess
import sys
from pathlib import Path

deck = Path(__file__).parent / "inputs_test_3d_ohm_solver_cylinder_compression_picmi.py"

env = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("OMPI_", "PMIX_", "PMI_", "HYDRA_", "SLURM_", "PALS_"))
}

result = subprocess.run(
    [
        sys.executable,
        str(deck),
        "--test",
        "--refined-core",
        "--grid-type",
        "staggered",
        "--patch-mode",
        "grazing",
    ],
    capture_output=True,
    text=True,
    env=env,
    timeout=600,
    check=False,
)

output = result.stdout + result.stderr
print(output)

assert result.returncode != 0, (
    "the grazing-patch run completed instead of aborting: the MR+EB "
    "clearance guard did not trip"
)
assert "clearance violation" in output, (
    "the run failed without the expected clearance-violation message"
)
assert "mr_eb_clearance_cells" in output, (
    "the abort message does not mention the override knob"
)
print("MR+EB clearance-guard abort test PASSED")
