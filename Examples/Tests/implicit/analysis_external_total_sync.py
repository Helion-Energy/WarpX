"""Check ideal frozen-ion total E with a ramped imposed field in each orientation."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--executable', required=True)
p.add_argument('--inputs', required=True)
p.add_argument('--axis', type=int, choices=range(3), required=True)
p.add_argument('--output-parent', required=True)
a = p.parse_args()
parent = Path(a.output_parent)
parent.mkdir(parents=True, exist_ok=True)
out = Path(tempfile.mkdtemp(prefix=f'axis{a.axis}-', dir=parent))
potential = [('0', '-0.5*z', '0.5*y'),
             ('0.5*z', '0', '-0.5*x'),
             ('-0.5*y', '0.5*x', '0')][a.axis]
internal = [0, 0, 0]
internal[(a.axis + 1) % 3] = 1
cmd = [a.executable, a.inputs, 'probe.check_total_E_zero=1',
       'ions.density=1e22', 'hybrid_pic_model.n0_ref=1e22',
       'hybrid_pic_model.add_external_fields=1',
       'warpx.B_external_grid=' + ' '.join(map(str, internal)),
       'external_vector_potential.uniform.A_time_external_function(t)=1+1e4*t',
       'amrex.the_arena_init_size=0']
for axis, expression in zip('xyz', potential):
    cmd.append(f'external_vector_potential.uniform.A{axis}_external_grid_function(x,y,z)={expression}')
(out / 'COMMAND.json').write_text(json.dumps(cmd, indent=2) + '\n')
with (out / 'stdout.log').open('x') as log:
    process = subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
assert process.returncode == 0, (out, process.returncode)
assert 'RESTORED_TOTAL_E_ZERO_WITH_GHOSTS' in (out / 'stdout.log').read_text()
result = json.loads((out / 'RESULT.json').read_text())
assert result['completed'] and result['step'] == 1
(out / 'PASS.json').write_text(json.dumps({'passed': True, 'axis': a.axis,
    'criterion': 'All native total-E components, valid and ghost nodes, <=1e-12 V/m',
    'result': result}, indent=2) + '\n')
print('EXTERNAL_TOTAL_E_SYNC_PASS', a.axis, out)
