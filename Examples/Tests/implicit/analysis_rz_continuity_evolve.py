#!/usr/bin/env python3
"""Short native PC-marker/pedestal evolution, Holmstrom on/off and audit isolation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--mpi-exec")
    args = parser.parse_args()
    args.output_parent.mkdir(exist_ok=True, parents=True)
    root = Path(tempfile.mkdtemp(prefix="evolve-", dir=args.output_parent))
    summary = []
    for holmstrom in ((1,) if args.mpi_exec else (0, 1)):
        for mode in ((2,) if args.mpi_exec else (0, 1, 2)):
            reference = None
            for audit in ((0, 2) if args.mpi_exec else (0, 1, 2)):
                path = root / f"holm{holmstrom}-mode{mode}-audit{audit}"
                path.mkdir()
                command = [str(args.executable.resolve()), str(args.inputs.resolve()),
                    "max_step=4", "amr.max_grid_size=8", "warpx.const_dt=1.e-10",
                    "continuity_test.evolve_steps=4", "continuity_test.write_maps=1",
                    f"warpx.use_filter={int(mode != 0)}",
                    f"warpx.rz_continuity_filter={int(mode == 2)}",
                    f"warpx.rz_continuity_audit_interval={audit}",
                    "hybrid_pic_model.solve_electron_energy_equation=1",
                    "hybrid_pic_model.elec_temp=10", "hybrid_pic_model.gamma=1.6666666666666667",
                    "hybrid_pic_model.qdsmc_time_advance=pc", "hybrid_pic_model.n_floor=1.e17",
                    "hybrid_pic_model.n0_ref=1.e18",
                    "hybrid_pic_model.density_pedestal=1",
                    "hybrid_pic_model.density_pedestal_track_floor=1",
                    f"hybrid_pic_model.holmstrom_vacuum_region={holmstrom}",
                    f"hybrid_pic_model.holmstrom_transition_width={.5 if holmstrom else 0}",
                    f"hybrid_pic_model.holmstrom_axis_radius={.006 if holmstrom else 0}",
                    f"hybrid_pic_model.holmstrom_axis_rolloff={.003 if holmstrom else 0}",
                    "warpx.B_ext_grid_init_style=constant", "warpx.B_external_grid=0 0 .0001"]
                if args.mpi_exec:
                    command = [args.mpi_exec, "-n", "2"] + command
                (path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                with (path / "run.log").open("w") as log:
                    proc = subprocess.run(command, cwd=path, env=dict(os.environ, OMP_NUM_THREADS="1"),
                                          stdout=log, stderr=subprocess.STDOUT, timeout=90)
                lines = (path / "run.log").read_text().splitlines()
                rows = [json.loads(x.split(" ", 1)[1]) for x in lines
                        if x.startswith("RZ_CONTINUITY ")]
                evolved = [json.loads(x.split(" ", 1)[1]) for x in lines
                           if x.startswith("CONTINUITY_EVOLVE ")]
                assert proc.returncode == 0 and len(evolved) == 1, (path, proc.returncode)
                assert evolved[0]["steps"] == 4 and evolved[0]["finite"]
                # Preserve every native box copy. Some thermal/velocity fields
                # are not synchronized at this endpoint; averaging duplicates
                # would hide a diagnostic-induced change. Require identical
                # bytes for all copies in the audit-off/audit-on comparison.
                data = {p.name: p.read_bytes() for p in path.glob("*_rank*.csv")}
                if audit == 0:
                    assert not rows
                    reference = data
                else:
                    assert len(rows) == 8 // audit, (path, rows)
                    assert {r["step"] for r in rows} == set(range(audit, 5, audit)), rows
                    assert data == reference, (path, "audit changed the physical state")
                    raw = [r["relative_step_residual"] for r in rows if r["stage"] == "raw"]
                    post = [r["relative_step_residual"] for r in rows if r["stage"] == "post_sync"]
                    assert max(raw) < 2.e-12, (path, "raw continuity", raw)
                    if mode != 1:
                        assert max(post) < 2.e-12, (path, "filtered continuity", post)
                    else:
                        assert max(post) > 1.e-4, (path, "missing historical failure", post)
                summary.append(dict(case=path.name, evolved=evolved[0], residuals=rows))
                print(path.name, evolved[0], flush=True)
    (root / "RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {len(summary)} evolved cases, audit on/off identical; evidence {root}")


if __name__ == "__main__":
    main()
