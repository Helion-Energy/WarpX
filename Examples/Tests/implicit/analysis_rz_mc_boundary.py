#!/usr/bin/env python3
"""Explicit continuity gate for the native MC boundary probe's measured output.

The native CTests assert continuity and inventory; coupled energy is separate.
This standalone gate checks every reported continuity and surface-flux measure.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('log', type=Path)
parser.add_argument('--continuity-tolerance', type=float, required=True)
args = parser.parse_args()
if not 0 < args.continuity_tolerance < 1:
    parser.error('continuity tolerance must lie strictly between zero and one')
rows = []
complete = []
for line in args.log.read_text().splitlines():
    if line.startswith('MC_BOUNDARY '):
        rows.append(json.loads(line.split(' ', 1)[1]))
    elif line.startswith('MC_BOUNDARY_COMPLETE '):
        complete.append(json.loads(line.split(' ', 1)[1]))
passed = (len(complete) == 1 and len(rows) == complete[0]['steps']
          and all(row['finite'] and all(
              row[key] < args.continuity_tolerance for key in [
                  'continuity_relative', 'species_continuity_relative',
                  'grid_particle_charge_relative', 'integrated_continuity_relative',
                  'zlo_escape_current_relative', 'zhi_escape_current_relative'])
                  for row in rows)
          and complete[0]['max_charge_inventory_relative'] < 2e-12
          and complete[0]['max_continuity_relative'] < args.continuity_tolerance)
print(json.dumps({'continuity_gate': 'PASS' if passed else 'FAIL',
                  'continuity_tolerance': args.continuity_tolerance,
                  'measured': complete, 'energy_closure': 'NOT ASSERTED'}, indent=2))
raise SystemExit(0 if passed else 2)
