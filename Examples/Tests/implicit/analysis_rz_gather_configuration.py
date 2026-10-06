#!/usr/bin/env python3
"""Require early constructor rejection, before InitData or particle allocation."""
import argparse
import json
import re
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('executable')
parser.add_argument('inputs')
args = parser.parse_args()
executable = str(Path(args.executable).resolve())
inputs = str(Path(args.inputs).resolve())
open_ends = ['boundary.field_lo=none neumann', 'boundary.field_hi=pec neumann',
             'boundary.particle_lo=none fractional_absorbing',
             'boundary.particle_hi=reflecting fractional_absorbing',
             'boundary.particle_absorption_fraction=0.05']
cases = [
    ('periodic-adjoint', ['warpx.rz_adjoint_gather=1',
                          'algo.field_gathering=energy-conserving'], None),
    ('periodic-mc', ['warpx.rz_adjoint_gather=0',
                     'algo.field_gathering=momentum-conserving'], None),
    ('open-mc', open_ends + ['warpx.rz_adjoint_gather=0',
                             'algo.field_gathering=momentum-conserving'], None),
    ('reject-mc-adjoint', ['warpx.rz_adjoint_gather=1',
                           'algo.field_gathering=momentum-conserving'],
     'requires algo.field_gathering=energy-conserving'),
    ('reject-open-adjoint', open_ends + ['warpx.rz_adjoint_gather=1',
                                        'algo.field_gathering=energy-conserving'],
     'requires periodic z; its axial endcap transpose is not implemented'),
    ('reject-radial-absorber-adjoint', ['warpx.rz_adjoint_gather=1',
                                       'algo.field_gathering=energy-conserving',
                                       'boundary.particle_hi=absorbing periodic'],
     'a PEC radial wall with reflecting particles'),
]
results = []
for name, options, rejection in cases:
    directory = Path(name)
    directory.mkdir(exist_ok=True)
    command = [executable, inputs, *options]
    result = subprocess.run(command, cwd=directory, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60, check=False)
    (directory/'run.log').write_text(result.stdout)
    # WarpX wraps diagnostic text with a '# ' continuation prefix.
    diagnostic = ' '.join(re.sub(r'(?m)^#\s*', '', result.stdout).split())
    accepted = 'GATHER_CONFIGURATION_ACCEPTED_BEFORE_INITDATA' in result.stdout
    passed = (result.returncode == 0 and accepted) if rejection is None else (
        result.returncode != 0 and not accepted and rejection in diagnostic)
    results.append({'name': name, 'command': command, 'exit': result.returncode,
                    'expected_rejection': rejection, 'passed': passed})
Path('RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
assert all(row['passed'] for row in results), results
print('PASS: six gather configurations checked before InitData/particle loading')
