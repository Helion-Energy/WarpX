#!/usr/bin/env python3
"""Preserved boundary-current history must survive reentry and restart."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

exe, inputs = map(lambda p: str(Path(p).resolve()), sys.argv[1:3])


def run(root, name, *options, expect_failure=False):
    directory = root / name
    directory.mkdir()
    command = [exe, inputs, 'boundary.particle_absorption_fraction=1', *options]
    result = subprocess.run(command, cwd=directory, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / 'stdout.log').write_text(result.stdout)
    if expect_failure:
        assert result.returncode != 0 and 'missing carried boundary moment' in result.stdout
        return directory, []
    assert result.returncode == 0, result.stdout
    rows = [json.loads(line.split(' ', 1)[1]) for line in result.stdout.splitlines()
            if line.startswith('MC_BOUNDARY ')]
    assert rows and 'MC_BOUNDARY_COMPLETE ' in result.stdout
    return directory, rows


with tempfile.TemporaryDirectory(prefix='rz-boundary-history-', dir='.') as tmp:
    root = Path(tmp).resolve()
    checkpoint, reference = run(root, 'reference', 'diagnostics.enable=1',
                                'diagnostics.diags_names=chk', 'chk.diag_type=Full',
                                'chk.format=checkpoint', 'chk.intervals=12',
                                'chk.file_prefix=chk')
    _, segmented = run(root, 'segmented', 'mc_boundary.segment_steps=6')
    middle = next(p for p in checkpoint.glob('chk*')
                  if p.is_dir() and int(p.name[3:]) == 12)
    _, restarted = run(root, 'restarted', 'mc_boundary.steps=12', f'amr.restart={middle}')
    expected = {row['step']: row for row in reference}
    errors = {}
    for name, rows in [('segmented', segmented), ('restarted', restarted)]:
        for key in ['particle_charge_C', 'deposition_grid_charge_C', 'ion_J',
                    'electron_bulk_J', 'magnetic_J', 'thermal_J', 'Te_max_K']:
            scale = max(max(abs(row[key]) for row in reference), 1.e-30)
            error = max(abs(row[key]-expected[row['step']][key]) for row in rows)/scale
            assert error < 1.e-12, (name, key, error)
            errors[f'{name}:{key}'] = error
    incomplete = root / 'incomplete-checkpoint'
    shutil.copytree(middle, incomplete)
    header = next(incomplete.rglob('*hybrid_current_fp_temp*_H'))
    header.unlink()  # Only this test-created copy is intentionally incomplete.
    run(root, 'missing-history', 'mc_boundary.steps=12', f'amr.restart={incomplete}',
        expect_failure=True)
    print(json.dumps({'boundary_history': 'PASS', 'relative_errors': errors,
                      'missing_history_rejected': True}, indent=2))
