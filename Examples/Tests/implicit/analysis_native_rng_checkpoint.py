"""Exercise the actual native checkpoint writer and late InitData RNG restore."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--executable', required=True)
parser.add_argument('--inputs', required=True)
parser.add_argument('--output-parent', required=True)
parser.add_argument('--launcher-json', default='[]')
args = parser.parse_args()
parent = Path(args.output_parent)
parent.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
launcher = json.loads(args.launcher_json)
rows = []
for label in ['fresh', 'restart']:
    folder = out / label
    folder.mkdir()
    command = launcher + [str(Path(args.executable).resolve()), str(Path(args.inputs).resolve()),
                          'amrex.the_arena_init_size=0', 'amrex.the_arena_is_managed=0']
    if label == 'restart':
        command += ['amr.restart=' + str(out / 'fresh/diags/chk000010'),
                    'warpx.require_restart_rng_state=1']
    (folder / 'COMMAND.json').write_text(json.dumps(command, indent=2) + '\n')
    with (folder / 'stdout.log').open('x') as log:
        result = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'CB_WRAPPER_VERBOSE': '1'})
    assert result.returncode == 0, (label, result.returncode)
    text = (folder / 'stdout.log').read_text()
    assert 'NATIVE_RNG_CHECKPOINT_EVOLUTION_COMPLETE' in text
    if label == 'restart':
        assert 'NATIVE_RNG_RESTART restored_host_and_cuda_pool' in text
    rows.append({'label': label, 'returncode': result.returncode})
files = sorted((out / 'fresh').glob('samples-rank-*.jsonl'))
assert files
for file in files:
    fresh = [json.loads(line) for line in file.read_text().splitlines()]
    restart = [json.loads(line) for line in (out / 'restart' / file.name).read_text().splitlines()]
    assert len(fresh) == 20 and len(restart) == 10
    assert fresh[10:] == restart, file.name
    rank = file.stem.removeprefix('samples-rank-')
    name = 'final-rng-rank-' + rank + '.bin'
    assert (out / 'fresh' / name).read_bytes() == (out / 'restart' / name).read_bytes()
receipt = {'passed': True, 'ranks': len(files), 'steps': 20, 'restart_step': 10,
           'host_and_cuda_draws_exact': True, 'final_engine_bytes_exact': True,
           'actual_native_writer_and_initdata_restore': True,
           'scope': 'same MPI and box ownership layout; isolated stochastic probe in a coupled EEE evolution',
           'processes': rows}
(out / 'PASS.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('NATIVE_RNG_CHECKPOINT_REPLAY_PASS', len(files), flush=True)
