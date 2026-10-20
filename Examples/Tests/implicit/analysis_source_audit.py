"""Compare raw particle states for source audit off/on; no conservation claim."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--off", type=Path)
parser.add_argument("--on", type=Path)
parser.add_argument("--reject-executable", type=Path)
parser.add_argument("--inputs", type=Path)
args = parser.parse_args()
if args.reject_executable:
    result = subprocess.run([str(args.reject_executable), str(args.inputs),
                             "hybrid_pic_model.source_guard.enabled=1",
                             "hybrid_pic_model.source_guard.audit_only=1"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode != 0, result.stdout
    assert "source_guard.audit_only requires source_guard.enabled = 0" in result.stdout, result.stdout
    print("SOURCE_AUDIT_EXCLUSIVE_MODE_REJECTION_PASS")
else:
    off = json.loads((args.off / "EXCHANGE.json").read_text())
    on = json.loads((args.on / "EXCHANGE.json").read_text())
    assert off["audit_enabled"] is False and on["audit_enabled"] is True
    assert off["source_guard_enabled"] is False and on["source_guard_enabled"] is False
    for row in (off, on):
        assert row["operator_oracle_pass"] is True
        assert row["exact_pair_conservation_claim"] is False
    for key in ("dt_s", "density_m3", "pedestal_m3", "Te0_eV", "Ti0_eV", "Te1_eV",
                "electron_rate_s1", "ion_rate_s1", "ranks"):
        assert off[key] == on[key], (key, off[key], on[key])
    names = sorted(f.name for f in args.off.glob("particles-rank*.bin"))
    assert names == sorted(f.name for f in args.on.glob("particles-rank*.bin"))
    assert len(names) == off["ranks"]
    hashes = {}
    for name in names:
        left = (args.off / name).read_bytes()
        right = (args.on / name).read_bytes()
        assert left == right, name
        hashes[name] = hashlib.sha256(left).hexdigest()
    scale = max(on["electron_energy_initial_J"], 1.)
    assert abs(on["audited_electron_J"] - on["electron_change_J"]) <= 1.e-12 * scale
    assert abs(on["audited_ion_J"] - on["actual_relativistic_ion_change_J"]) <= 1.e-12 * scale
    receipt = {"passed": True, "scope": "diagnostic on/off and independent energy oracle",
               "raw_particle_arrays_bitwise_equal": True, "particle_sha256": hashes,
               "source_conservation_qualified": False}
    (args.on / "AUDIT_INVARIANCE_PASS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("SOURCE_AUDIT_ON_OFF_AND_ENERGY_ORACLE_PASS")
