"""Verify the unsafe-footprint guard across serial, OpenMP and GPU builds."""

import subprocess
import sys

result = subprocess.run(
    sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
print(result.stdout, end="")
if result.returncode == 0:
    assert "STANDOFF_FOOTPRINT_REJECTION_PASS" in result.stdout
else:
    assert (
        "Insulating EB particle/current footprint leaves the validated" in result.stdout
    )
print("STANDOFF_FOOTPRINT_REJECTION_DRIVER_PASS")
