import os
import resource
import subprocess
import sys

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
exe, case = sys.argv[1:]
p = subprocess.run(
    [exe, "test.case=" + case, "amrex.signal_handling=0"],
    capture_output=True,
    text=True,
    env=dict(os.environ, OMP_NUM_THREADS="1"),
)
text = p.stdout + p.stderr
expected = {
    "partner": "changes the physical electron relaxation partner",
    "convention": "changes the expected OU convention or accuracy budget",
    "relative": "changes the expected OU convention or accuracy budget",
    "absolute": "changes the expected OU convention or accuracy budget",
    "version": "requires complete version 2 metadata",
    "missing": "Invalid or missing Eulerian thermal model checkpoint contract",
    "model_version": "Invalid or missing Eulerian thermal model checkpoint contract",
    "duplicate": "Invalid or missing Eulerian thermal model checkpoint contract",
    "malformed": "Invalid or missing Eulerian thermal model checkpoint contract",
    "policy": "unsupported source/boundary semantics",
    "history": "unsupported source/boundary semantics",
    "source": "unsupported source/boundary semantics",
    "negative_tolerance": "Invalid Eulerian thermal model contract",
    "trailing": "invalid Eulerian electron energy checkpoint metadata",
    "gamma": "requires matching U coordinate",
    "floor": "requires matching U coordinate",
    "axis": "requires matching U coordinate",
    "legacy": "requires matching U coordinate",
    "no_channel": "requires an active accepted exchange channel",
    "bad_partner": "must be off or population_bounded_nr",
    "partner_relativistic": "requires relaxation and bounded_nr audit convention",
    "reference_without_partner": "requires the population thermal partner",
    "default_restart_off": "changes the physical electron relaxation partner",
    "default_relativistic": "requires relaxation and bounded_nr audit convention",
    "partner_no_relaxation": "requires relaxation and bounded_nr audit convention",
}[case]
print(text)
if p.returncode == 0 or expected not in text:
    raise SystemExit("FAIL unexpected checkpoint rejection result")
print("PASS expected rejection", case)
