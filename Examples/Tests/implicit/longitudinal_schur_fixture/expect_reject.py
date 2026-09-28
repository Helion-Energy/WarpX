#!/usr/bin/env python3
"""Require the explicit unsupported-inertia-form guard, not any crash."""

import os
import resource
import subprocess
import sys

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
env = dict(os.environ, OMP_NUM_THREADS="1")
r = subprocess.run(
    [
        sys.argv[1],
        "test.reject=" + sys.argv[2],
        "amrex.verbose=0",
        "tiny_profiler.enabled=0",
    ],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    timeout=30,
)
assert (
    r.returncode != 0
    and "Longitudinal Schur qualifies midpoint dJe/dt-only" in r.stdout
), r.stdout
print("SCHUR_EXPECTED_REJECTION", sys.argv[2])
