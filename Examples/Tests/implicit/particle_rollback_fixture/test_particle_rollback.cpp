/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ImplicitParticleRollback.H"
#include "MomentFixture.H"
#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/ParticleBoundaries_K.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include <map>

namespace moment_test {
void inverse_rz_volume (MultiFab&, bool);
}
using namespace moment_test;
using warpx::implicit::ParticleRollbackTile;
using warpx::implicit::ParticleStartView;
using PC = amrex::ParticleContainerPureSoA<PIdx::nattribs, 0>;
using Iter = amrex::ParIterSoA<PIdx::nattribs, 0>;
using PR = amrex::ParticleReal;
using Key = std::pair<int, int>;
ParticleStartView
start (Iter const& it) {
    auto const& a = it.GetStructOfArrays();
    ParticleStartView s;
    s.position = {a.GetRealData("x_n").data(), a.GetRealData("y_n").data(),
                  a.GetRealData("z_n").data()};
    s.momentum = {a.GetRealData("ux_n").data(), a.GetRealData("uy_n").data(),
                  a.GetRealData("uz_n").data()};
    return s;
}
struct State {
    Context c;
    PC pc;
    bool sub;
    std::map<Key, ParticleRollbackTile> saved;
    std::map<Key, amrex::Gpu::DeviceVector<PR>> reference;
    State (int box, bool suborbits, bool endpoint = false)
        : c(box, false, true, true), pc(c.geom, c.dm, c.ba), sub(suborbits) {
        amrex::Vector<std::string> names;
        for (auto const* s : PIdx::names) {
            names.push_back(s);
        }
        pc.SetSoACompileTimeNames(names, {});
        for (auto const* s : {"x_n", "y_n", "z_n", "ux_n", "uy_n", "uz_n"}) {
            pc.AddRealComp(s, 0);
        }
        if (sub) {
            pc.AddIntComp("nsuborbits", 0);
        }
        auto dx = c.geom.CellSizeArray();
        for (amrex::MFIter it(c.u); it.isValid(); ++it) {
            if (endpoint) {
                amrex::IntVect const target(AMREX_D_DECL(7, 15, 7));
                if (!it.validbox().contains(target)) {
                    continue;
                }
            }
            auto& tile = pc.DefineAndReturnParticleTile(0, it.index(), 0);
            tile.resize(endpoint ? 1 : 5);
            auto* subcounter =
                sub ? tile.GetStructOfArrays().GetIntData("nsuborbits").data()
                    : nullptr;
            auto data = tile.getParticleTileData();
            auto const valid = it.validbox();
            auto lo = valid.smallEnd();
            int const grid = it.index();
            int const rank = amrex::ParallelDescriptor::MyProc();
            amrex::ParallelFor(tile.numParticles(), [=] AMREX_GPU_DEVICE(
                                                        amrex::Long i) {
                auto p = data[i];
                p.id() = 100 * grid + i + 1;
                p.cpu() = rank;
                for (int d = 0; d < PIdx::nattribs; ++d) {
                    data.rdata(d)[i] = 1. + .13 * d + .01 * i;
                }
#ifdef WARPX_DIM_RZ
                p.pos(0) = (endpoint ? 7.4 : lo[0] + 1.5 + .03 * i) * dx[0];
                p.pos(1) = (endpoint ? 15.75 : lo[1] + 1.5 + .04 * i) * dx[1];
                data.rdata(PIdx::theta)[i] =
                    7.2 + .123 * i; // cannot recover this branch from atan2
#else
                p.pos(0)=(endpoint?7.4:lo[0]+1.5+.03*i)*dx[0];
                p.pos(1)=(endpoint?15.75:lo[1]+1.5+.04*i)*dx[1];
                p.pos(2)=(endpoint?7.4:lo[2]+1.5+.02*i)*dx[2];
#endif
                data.rdata(PIdx::w)[i] = 1.;
                if (subcounter)
                    subcounter[i] = 7 + int(i);
            });
        }
        amrex::Gpu::streamSynchronize();
    }
    void
    capture () {
        for (Iter it(pc, 0); it.isValid(); ++it) {
            Key const key{it.index(), it.LocalTileIndex()};
            auto& a = it.GetStructOfArrays();
            int* nsub = sub ? a.GetIntData("nsuborbits").data() : nullptr;
            auto get = GetParticlePosition<PIdx>(it);
            auto const s = start(it);
            saved[key].Capture(it.numParticles(), a.GetIdCPUData().data(), get,
                               s, nsub);
            auto& ref = reference[key];
            ref.resize(PIdx::nattribs * it.numParticles());
            auto* snap = ref.data();
            auto data = it.GetParticleTile().getParticleTileData();
            auto* xn = a.GetRealData("x_n").data();
            auto* yn = a.GetRealData("y_n").data();
            auto* zn = a.GetRealData("z_n").data();
            auto* un = a.GetRealData("ux_n").data();
            auto* vn = a.GetRealData("uy_n").data();
            auto* wn = a.GetRealData("uz_n").data();
            amrex::ParallelFor(
                it.numParticles(), [=] AMREX_GPU_DEVICE(amrex::Long i) {
                    get(i, xn[i], yn[i], zn[i]);
                    un[i] = data.rdata(PIdx::ux)[i];
                    vn[i] = data.rdata(PIdx::uy)[i];
                    wn[i] = data.rdata(PIdx::uz)[i];
                    for (int d = 0; d < PIdx::nattribs; ++d) {
                        snap[PIdx::nattribs * i + d] = data.rdata(d)[i];
                    }
                    if (nsub) {
                        nsub[i] = 1;
                    }
                });
        }
        amrex::Gpu::streamSynchronize();
    }
    bool
    matches () {
        int valid = 1;
        std::size_t count = 0;
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto pos = saved.find({it.index(), it.LocalTileIndex()});
            auto const& a = it.GetStructOfArrays();
            if (pos == saved.end() ||
                !pos->second.Matches(
                    it.numParticles(), a.GetIdCPUData().data(), start(it),
                    sub ? a.GetIntData("nsuborbits").data() : nullptr)) {
                valid = 0;
            }
            ++count;
        }
        if (count != saved.size()) {
            valid = 0;
        }
        amrex::ParallelDescriptor::ReduceIntMin(valid);
        return valid != 0;
    }
    void
    perturb (int trial) {
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto data = it.GetParticleTile().getParticleTileData();
            auto set = SetParticlePosition<PIdx>(it);
            auto get = GetParticlePosition<PIdx>(it);
            int* nsub =
                sub ? it.GetStructOfArrays().GetIntData("nsuborbits").data()
                    : nullptr;
            amrex::ParallelFor(it.numParticles(), [=] AMREX_GPU_DEVICE(
                                                      amrex::Long i) {
                PR x, y, z;
                get(i, x, y, z);
                set(i, x + .021 * trial, y - .033 * trial, z + .018 * trial);
                for (int d : {PIdx::ux, PIdx::uy, PIdx::uz}) {
                    data.rdata(d)[i] *= 1. + .125 * trial;
                }
                if (nsub) {
                    nsub[i] = 11 + trial;
                }
            });
        }
    }
    void
    restore () {
        check(matches(), "all topology guards precede restore");
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto& a = it.GetStructOfArrays();
            saved.at({it.index(), it.LocalTileIndex()})
                .Restore(SetParticlePosition<PIdx>(it),
                         {a.GetRealData(PIdx::ux).data(),
                          a.GetRealData(PIdx::uy).data(),
                          a.GetRealData(PIdx::uz).data()},
                         sub ? a.GetIntData("nsuborbits").data() : nullptr);
        }
        amrex::Gpu::streamSynchronize();
    }
    void
    exact () {
        int bad = 0;
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto data = it.GetParticleTile().getParticleTileData();
            auto const* ref =
                reference.at({it.index(), it.LocalTileIndex()}).data();
            auto const* nsub =
                sub ? it.GetStructOfArrays().GetIntData("nsuborbits").data()
                    : nullptr;
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<int> rd(op);
            op.eval(
                it.numParticles(), rd,
                [=] AMREX_GPU_DEVICE(amrex::Long i) -> amrex::GpuTuple<int> {
                    int mismatch = 0;
                    for (int d = 0; d < PIdx::nattribs; ++d) {
                        auto const a = data.rdata(d)[i],
                                   b = ref[PIdx::nattribs * i + d];
                        mismatch |=
                            a != b || std::signbit(a) != std::signbit(b);
                    }
                    if (nsub) {
                        mismatch |= nsub[i] != 7 + int(i);
                    }
                    return {mismatch};
                });
            bad |= amrex::get<0>(rd.value());
        }
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        check(bad == 0, "all accepted native attributes restored bitwise");
    }
    MultiFab
    deposit (bool virtual_endpoint = false) {
        auto rho = c.node(4);
        rho.setVal(0.);
        auto dx = c.geom.CellSizeArray();
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto get = GetParticlePosition<PIdx>(it);
            auto const s = start(it);
            auto endpoint = [=] AMREX_GPU_HOST_DEVICE(long i, PR& x, PR& y,
                                                      PR& z) {
                get(i, x, y, z);
                if (virtual_endpoint) {
                    x = 2 * x - s.position[0][i];
                    y = 2 * y - s.position[1][i];
                    z = 2 * z - s.position[2][i];
                }
            };
#ifdef WARPX_DIM_RZ
            amrex::XDim3 inv{1 / dx[0], 1., 1 / dx[1]};
#else
            amrex::XDim3 inv{1 / dx[0], 1 / dx[1], 1 / dx[2]};
#endif
            doChargeDepositionShapeN<3>(
                endpoint, it.GetStructOfArrays().GetRealData(PIdx::w).data(),
                nullptr, rho[it], it.numParticles(), inv,
                amrex::XDim3{0., 0., 0.}, amrex::Dim3{0, 0, 0}, 1., 1);
        }
        inverse_rz_volume(rho, c.opts.verboncoeur_axis_correction);
        rho.SumBoundary(0, 1, rho.nGrowVect(), rho.nGrowVect(),
                        c.geom.periodicity());
        PEC::ApplyReflectiveBoundarytoRhofield(&rho, c.blo, c.bhi, c.plo, c.phi,
                                               c.geom, 0, PatchType::fine, {});
        rho.OverrideSync(c.geom.periodicity());
        rho.FillBoundary(c.geom.periodicity());
        return rho;
    }
};
void
endpoint (State& s, bool crossing) {
    s.capture();
    auto dx = s.c.geom.CellSizeArray();
    for (Iter it(s.pc, 0); it.isValid(); ++it) {
        auto get = GetParticlePosition<PIdx>(it);
        auto set = SetParticlePosition<PIdx>(it);
        amrex::ParallelFor(it.numParticles(),
                           [=] AMREX_GPU_DEVICE(amrex::Long i) {
                               PR x, y, z;
                               get(i, x, y, z);
#ifdef WARPX_DIM_RZ
                               z = .5 * (z + (crossing ? 16.2 : 15.9) * dx[1]);
#else
            y=.5*(y+(crossing?16.2:15.9)*dx[1]);
#endif
                               set(i, x, y, z);
                           });
    }
    auto predicted = s.deposit(true);
    for (Iter it(s.pc, 0); it.isValid(); ++it) {
        auto get = GetParticlePosition<PIdx>(it);
        auto set = SetParticlePosition<PIdx>(it);
        auto const old = start(it);
        amrex::ParallelFor(
            it.numParticles(), [=] AMREX_GPU_DEVICE(amrex::Long i) {
                PR x, y, z;
                get(i, x, y, z);
                set(i, 2 * x - old.position[0][i], 2 * y - old.position[1][i],
                    2 * z - old.position[2][i]);
            });
    }
    auto finished = s.deposit();
    near(difference(predicted, finished), 0.,
         2.e-12 * std::max(1., predicted.norm0()),
         "virtual and finished pre-boundary shape3 endpoints match");
    for (Iter it(s.pc, 0); it.isValid(); ++it) {
        auto get = GetParticlePosition<PIdx>(it);
        auto set = SetParticlePosition<PIdx>(it);
        auto* ids = it.GetStructOfArrays().GetIdCPUData().data();
        amrex::ParallelForRNG(
            it.numParticles(),
            [=] AMREX_GPU_DEVICE(amrex::Long i,
                                 amrex::RandomEngine const& engine) {
                PR x, y, z;
                get.AsStored(i, x, y, z);
                bool flip = false, thermal = false, lost = false;
#ifdef WARPX_DIM_RZ
                ApplyParticleBoundaries::apply_boundary(
                    z, 0., 3., flip, thermal, lost,
                    ParticleBoundaryType::Absorbing,
                    ParticleBoundaryType::Absorbing, 0., 0., engine);
#else
                ApplyParticleBoundaries::apply_boundary(
                    y, 0., 3., flip, thermal, lost,
                    ParticleBoundaryType::Absorbing,
                    ParticleBoundaryType::Absorbing, 0., 0., engine);
#endif
                if (lost)
                    amrex::ParticleIDWrapper{ids[i]}.make_invalid();
                else
                    set.AsStored(i, x, y, z);
            });
    }
    s.pc.Redistribute();
    auto actual = s.deposit();
    if (crossing) {
        near(actual.norm0(), 0., 0., "absorbed endpoint has no charge");
        near(integral(s.c, predicted, true, 0, true), 1., 2.e-12,
             "virtual PMC endpoint still contains lost charge");
        check(!s.matches(),
              "post-absorption topology cannot roll back from xn/un");
    } else {
        near(difference(predicted, actual), 0.,
             2.e-12 * std::max(1., predicted.norm0()),
             "actual no-crossing endpoint deposit after redistribution");
    }
    amrex::Print() << "endpoint crossing=" << crossing << " virtual_charge="
                   << integral(s.c, predicted, true, 0, true)
                   << " actual_charge=" << integral(s.c, actual, true, 0, true)
                   << '\n';
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int box = 4, sub = 0;
        std::string name = "restore";
        pp.query("case", name);
        pp.query("max_grid_size", box);
        pp.query("suborbits", sub);
        State s(box, sub != 0, name == "endpoint" || name == "absorption");
        if (name == "endpoint" || name == "absorption") {
            endpoint(s, name == "absorption");
        } else {
            s.capture();
            auto old = s.deposit();
            s.perturb(1);
            if (name == "restore") {
                s.restore();
                s.exact();
                auto restored = s.deposit();
                near(difference(old, restored, true), 0., 0.,
                     "restored deposited rho including images bitwise");
                s.perturb(2);
                s.restore();
                s.exact();
                s.restore();
                s.exact();
            } else if (name == "guards") {
                for (Iter it(s.pc, 0); it.isValid(); ++it) {
                    auto* id = it.GetStructOfArrays().GetIdCPUData().data();
                    bool const invalidate =
                        amrex::ParallelDescriptor::MyProc() == 0;
                    amrex::ParallelFor(
                        it.numParticles(), [=] AMREX_GPU_DEVICE(amrex::Long i) {
                            if (invalidate && i == 0) {
                                amrex::ParticleIDWrapper{id[i]}.make_invalid();
                            }
                        });
                }
                check(!s.matches(), "invalidated particle identity guard");
            } else if (name == "migration") {
                bool const one_box = s.c.ba.size() == 1;
                for (Iter it(s.pc, 0); it.isValid(); ++it) {
                    auto set = SetParticlePosition<PIdx>(it);
                    amrex::ParallelFor(it.numParticles(),
                                       [=] AMREX_GPU_DEVICE(amrex::Long i) {
                                           set(i, one_box ? 3. : .4, .3, .6);
                                       });
                }
                s.pc.Redistribute();
                check(!s.matches(),
                      "real AMReX redistribution invalidates saved topology");
            } else {
                amrex::Abort("unknown fixture");
            }
        }
        amrex::Print() << "PASS " << name << " boxes=" << s.c.ba.size()
                       << " ranks=" << amrex::ParallelDescriptor::NProcs()
                       << " rollback_bytes_per_particle="
                       << ParticleRollbackTile::BytesPerParticle(sub != 0)
                       << '\n';
    }
    amrex::Finalize();
}
