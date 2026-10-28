"""Certify the explicit pre-advance occupied-cell conditioning Python API.

Run with inputs_test_3d_coupled_port on one or more CPU/GPU MPI ranks. This
invokes the same configured splitter as accepted-step evolution, without
advancing time or changing carried fields.
"""

import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

import pywarpx
from pywarpx import fields, libwarpx
from pywarpx.particle_containers import ParticleContainerWrapper

comm = MPI.COMM_WORLD
pywarpx.geometry.dims = 3
libwarpx.initialize(
    argv=[
        sys.executable,
        sys.argv[1],
        "max_step=0",
        "amrex.the_arena_init_size=0",
        "amrex.the_arena_is_managed=0",
    ]
)
wx = libwarpx.libwarpx_so.get_instance()
rho = fields.MultiFabWrapper(mf_name="hybrid_rho_fp_temp", level=0).mf
field_images = {
    name: np.asarray(fields.MultiFabWrapper(mf_name=name, level=0)[...]).copy()
    for name in ("hybrid_rho_fp_temp", "hybrid_electron_temperature_fp")
}
before_clock = (wx.getistep(0), wx.gett_new(0), wx.getdt(0))


def state(pc):
    arrays = [
        pc.get_particle_real_arrays(name, 0, copy_to_host=True)
        for name in ("x", "y", "z", "w", "ux", "uy", "uz")
    ]
    moments = np.zeros(8)
    counts = np.zeros(16**3, dtype=np.int64)
    for x, y, z, w, ux, uy, uz in zip(*arrays):
        assert np.isfinite(np.concatenate((x, y, z, w, ux, uy, uz))).all()
        cells = np.floor((np.column_stack((x, y, z)) + 0.3) / (0.6 / 16)).astype(int)
        assert ((cells >= 0) & (cells < 16)).all()
        index = (cells[:, 0] * 16 + cells[:, 1]) * 16 + cells[:, 2]
        counts += np.bincount(index, minlength=16**3)
        moments += np.array(
            [
                w.sum(),
                (w * x).sum(),
                (w * y).sum(),
                (w * z).sum(),
                (w * ux).sum(),
                (w * uy).sum(),
                (w * uz).sum(),
                (w * (ux * ux + uy * uy + uz * uz)).sum(),
            ]
        )
    return comm.allreduce(moments), comm.allreduce(counts)


results = {}
for species in ("ions_D", "ions_T"):
    pc = ParticleContainerWrapper(species)
    before, counts0 = state(pc)
    assert np.any((counts0 > 0) & (counts0 < 30))
    for low, high, lev in ((-1.0, 1.0, 0), (1.0, 0.0, 0), (0.0, 1.0, 1)):
        try:
            pc.particle_container.split_depleted_band(rho, low, high, lev)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid conditioning request was not rejected")
    pc.particle_container.split_depleted_band(rho, 0.0, 1.0e100, 0)
    after, counts1 = state(pc)
    np.testing.assert_array_equal(counts0 > 0, counts1 > 0)
    assert np.all(counts1[counts1 > 0] >= 30)
    assert np.all(counts1[counts0 >= 30] == counts0[counts0 >= 30])
    scales = np.array(
        [before[0]]
        + [0.3 * before[0]] * 3
        + [np.sqrt(before[0] * before[7])] * 3
        + [before[7]]
    )
    error = np.abs(after - before) / scales
    assert np.max(error) < 3.0e-13, error
    pc.particle_container.split_depleted_band(rho, 0.0, 1.0e100, 0)
    again, counts2 = state(pc)
    np.testing.assert_array_equal(counts1, counts2)
    np.testing.assert_array_equal(after, again)
    results[species] = dict(
        before=int(counts0.sum()),
        after=int(counts1.sum()),
        min_occupied_ppc=int(counts1[counts1 > 0].min()),
        max_invariant_scaled_error=float(error.max()),
    )

assert before_clock == (wx.getistep(0), wx.gett_new(0), wx.getdt(0))
for name, saved in field_images.items():
    np.testing.assert_array_equal(
        saved, np.asarray(fields.MultiFabWrapper(mf_name=name, level=0)[...])
    )
libwarpx.finalize()
if comm.rank == 0:
    with Path("CONDITIONING_PASS.json").open("x") as f:
        json.dump(
            dict(
                passed=True,
                ranks=comm.size,
                species=results,
                clock_unchanged=True,
                carried_fields_unchanged=True,
            ),
            f,
            indent=2,
        )
