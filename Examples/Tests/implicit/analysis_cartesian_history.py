"""Native checkpoint writer/reader regression, including incomplete histories."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--executable', required=True)
parser.add_argument('--inputs', required=True)
parser.add_argument('--output-parent', required=True)
args = parser.parse_args()
parent = Path(args.output_parent).resolve()
parent.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix='native-', dir=parent))
exe = str(Path(args.executable).resolve())
inputs = str(Path(args.inputs).resolve())

def run(name, extra=(), expected=None):
    folder = root / name
    folder.mkdir()
    with (folder / 'stdout.log').open('w') as output:
        result = subprocess.run([exe, inputs, *extra], cwd=folder,
                                stdout=output, stderr=subprocess.STDOUT)
    raw = (folder / 'stdout.log').read_text()
    if expected is None:
        assert result.returncode == 0, (name, result.returncode, raw[-3000:])
    else:
        assert result.returncode != 0 and expected in raw, (name, result.returncode, raw[-3000:])
    return folder, raw

fresh, _ = run('fresh')
checkpoint = fresh / 'diags/chk000001'
assert (checkpoint / 'CartesianHybridHistory/Header').is_file()
_, raw = run('restored', ['amr.restart=' + str(checkpoint)])
assert 'CARTESIAN_CARRIED_HISTORY_RESTORE_EXACT' in raw
missing = root / 'missing'
shutil.copytree(checkpoint, missing)
(missing / 'CartesianHybridHistory/level-0-rho_old_H').unlink()
run('reject_missing', ['amr.restart=' + str(missing)],
    'Missing or incompatible Cartesian hybrid carried moment rho_old')
bad = root / 'bad_manifest'
shutil.copytree(checkpoint, bad)
header = bad / 'CartesianHybridHistory/Header'
lines = header.read_text().splitlines()
lines[0] = 'CartesianHybridHistory 999'
header.write_text('\n'.join(lines) + '\n')
run('reject_version', ['amr.restart=' + str(bad)],
    'Invalid Cartesian hybrid moment-history manifest')
(root / 'PASS.json').write_text(json.dumps(dict(passed=True,
    native_restore_exact=True, first_evolve_initialization_preserves_history=True,
    missing_payload_rejected=True, invalid_manifest_rejected=True)) + '\n')
print('NATIVE_CARTESIAN_HISTORY_PASS', root)
