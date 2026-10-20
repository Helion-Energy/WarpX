"""Check native transport execution and strict JSON Hall-overflow reporting."""
import argparse, json, math, os, subprocess, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--executable', required=True)
p.add_argument('--inputs', required=True)
p.add_argument('--overflow', type=int, choices=(0, 1), required=True)
a = p.parse_args()
command = [a.executable, a.inputs, 'amr.n_cell=16 16 16', 'amr.max_grid_size=8',
           'amr.blocking_factor=8', 'physical_test.mode=4', 'physical_test.guard_only=1',
           'physical_test.temperature_eV=8000', 'physical_test.unfreeze=1',
           'physical_test.density=' + ('1e-280' if a.overflow else '3e21'),
           'physical_test.expect_hall_overflow=' + str(a.overflow),
           'hybrid_pic_model.qdsmc_conduction_coefficient_interval=1']
env = dict(os.environ, WARPX_QDSMC_COND_REPORT='1')
with Path('native.log').open('x') as stream:
    result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
assert result.returncode == 0, result.returncode
def invalid(value):
    raise ValueError('nonstandard JSON constant: ' + value)
def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)
records = [json.loads(s.split('CONDUCTION_ACCOUNT ', 1)[1], parse_constant=invalid)
           for s in Path('native.log').read_text().splitlines() if 'CONDUCTION_ACCOUNT ' in s]
assert len(records) == 1, len(records)
r = records[0]
assert finite(r)
for name in ['min', 'max']:
    flag = r['hall_' + name + '_overflow']
    assert type(flag) is bool and flag == bool(a.overflow)
    assert len(r['coefficient_' + name]) == 6
    if flag: assert r['coefficient_' + name][5] == sys.float_info.max
assert r['coefficient_nodes'] > 0
assert r['completed_time_s'] == r['requested_time_s'] > 0
assert all(0 <= x <= y for x, y in zip(r['coefficient_min'], r['coefficient_max']))
scale = max(abs(r['energy_before_J'][0]), abs(r['energy_after_J'][0]), 1e-300)
residual = math.fsum([r['energy_after_J'][0], -r['energy_before_J'][0], -r['floor_heat_J'],
                      *r['outward_heat_J'], *[-v for v in r['entry_heat_J']]])
assert abs(residual) / scale < 1e-11
Path('REPORT.json').write_text(json.dumps(dict(passed=True, overflow=bool(a.overflow),
    native_record=r, independent_relative_residual=abs(residual)/scale), indent=2)+'\n')
print('HALL_REPORT_NATIVE_JSON PASS')
