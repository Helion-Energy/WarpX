#!/usr/bin/env python3
#
# --- Abort-gate test: the hybrid conformal embedded-boundary wall treatment
# --- (hybrid_pic_model.use_conformal_eb) is COLLOCATED-ONLY, and a staggered
# --- (Yee) deck that requests it must ABORT in the WarpX constructor with a
# --- clear message rather than silently degrade to the staircase wall.
# ---
# --- The parent process (this test) launches a subprocess running a minimal
# --- staggered-grid hybrid+EB simulation with use_conformal_eb=True and
# --- asserts that the child exits NONZERO and that its output contains the
# --- "collocated-only" abort message. The parent exits 0 when the abort
# --- fires correctly.

import argparse
import glob
import os
import subprocess
import sys

N_CELL = 16
ABORT_MARKER = "collocated-only"

# Environment prefixes injected by MPI launchers (hydra/PMI, Open MPI, Slurm).
# The child is a fresh singleton MPI process: if it inherits these from the
# mpiexec that launched the parent, its MPI_Init tries to join the parent's
# job and fails/hangs, so they are scrubbed from the inherited environment.
MPI_LAUNCHER_PREFIXES = (
    "PMI_",
    "PMIX_",
    "HYDRA_",
    "HYDI_",
    "OMPI_",
    "PRTE_",
    "PRRTE_",
    "SLURM_",
    "MPIEXEC_",
)


def run_child_sim():
    """Minimal STAGGERED hybrid+EB sim requesting the conformal wall treatment.

    This must die inside the WarpX constructor (during initialize_warpx),
    before any stepping -- if it survives to the end, exit 0 and let the
    parent flag the missing abort as the failure.
    """
    from pywarpx import picmi

    grid = picmi.Cartesian3DGrid(
        number_of_cells=[N_CELL, N_CELL, N_CELL],
        lower_bound=[-1.0, -1.0, -1.0],
        upper_bound=[1.0, 1.0, 1.0],
        lower_boundary_conditions=["dirichlet", "periodic", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic", "periodic"],
        lower_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic", "periodic"],
    )

    sim = picmi.Simulation(
        time_step_size=1.0e-9,
        max_steps=1,
        particle_shape=1,
        verbose=0,
    )
    sim.grid_type = "staggered"  # the default, spelled out: this is the gate

    sim.solver = picmi.HybridPICSolver(
        grid=grid,
        gamma=5.0 / 3.0,
        Te=0.0,
        n0=1.0e18,
        n_floor=1.0e16,
        plasma_resistivity=1.0e-6,
        substeps=4,
        use_conformal_eb=True,  # collocated-only: must abort on staggered
    )
    sim.embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="(x-0.5)",
    )

    sim.initialize_inputs()
    sim.initialize_warpx()  # expected to abort in the WarpX constructor
    sim.step(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--child",
        action="store_true",
        help="run the (aborting) staggered use_conformal_eb simulation",
    )
    args = parser.parse_args()

    if args.child:
        run_child_sim()
        return

    # Inherit the parent environment (PYTHONPATH, LD_LIBRARY_PATH, ...) minus
    # the MPI launcher bookkeeping, so the child MPI-initializes standalone.
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(MPI_LAUNCHER_PREFIXES)
    }

    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child"],
        env=env,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    output = result.stdout or ""

    # amrex::Abort routes the message to stderr and/or the Backtrace files
    # (depending on signal handling), so scan those too.
    for bt in glob.glob(os.path.join(os.getcwd(), "Backtrace*")):
        try:
            with open(bt, errors="replace") as f:
                output += "\n" + f.read()
        except OSError:
            # a Backtrace file may be unreadable or disappear between the
            # glob and the open; the abort marker is still searched for in
            # the child's captured stdout/stderr, so just skip it
            pass

    print("---- child output ----")
    print(output)
    print("----------------------")
    print(f"child exit code: {result.returncode}")

    assert result.returncode != 0, (
        "the staggered use_conformal_eb child completed without aborting: "
        "the collocated-only gate in the WarpX constructor did not fire"
    )
    assert ABORT_MARKER in output, (
        f"the child aborted (exit {result.returncode}) but its output does not "
        f"contain '{ABORT_MARKER}': it died for a different reason than the "
        "use_conformal_eb collocated-only gate"
    )
    print(
        "abort gate OK: staggered use_conformal_eb aborted with the "
        "collocated-only message"
    )


if __name__ == "__main__":
    main()
