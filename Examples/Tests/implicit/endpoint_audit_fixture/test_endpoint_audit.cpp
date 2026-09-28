/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ImplicitParticleEndpointAudit.H"
#include "MomentFixture.H"
#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/Deposition/CurrentDeposition.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include <map>
using namespace moment_test;
using namespace warpx::implicit;
using PR = amrex::ParticleReal;
using PC = amrex::ParticleContainerPureSoA<PIdx::nattribs, 0>;
using Iter = amrex::ParIterSoA<PIdx::nattribs, 0>;
using Point = endpoint_detail::Point;

ParticleStartView
start (Iter const& it) {
    auto const& a = it.GetStructOfArrays();
    return {{a.GetRealData("x_n").data(), a.GetRealData("y_n").data(),
             a.GetRealData("z_n").data()},
            {a.GetRealData("ux_n").data(), a.GetRealData("uy_n").data(),
             a.GetRealData("uz_n").data()}};
}
struct Test {
    Context c;
    PC pc;
    MultiFab rho;
    Current eg, bg;
    EndpointAuditFields fields;
    std::map<int, amrex::Gpu::DeviceVector<PR>> snapshot;
    std::map<int, amrex::Gpu::DeviceVector<std::uint64_t>> ids;
    std::string name;
    Test (int box, std::string label)
        : c(box, label == "periodic" || label == "projection_reach", true,
            true),
          pc(c.geom, c.dm, c.ba),
          rho(amrex::convert(c.ba, amrex::IntVect::TheNodeVector()), c.dm, 1,
              label == "endpoint_reach"     ? 0
              : label == "projection_reach" ? 2
                                            : 4),
          name(std::move(label)) {
        // Exercise real WarpX RZ convention: Coord0 with compile-time RZ
        // mapping.
        auto const physical = c.geom.ProbDomain();
        int periodic[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            periodic[d] = c.geom.isPeriodic(d);
        }
        c.geom = amrex::Geometry(c.geom.Domain(), &physical, 0, periodic);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (periodic[d]) {
                c.plo[d] = c.phi[d] = ParticleBoundaryType::Periodic;
            }
        }
#ifdef WARPX_DIM_RZ
        c.plo[0] = ParticleBoundaryType::None;
#endif
        if (name == "thermal") {
            c.phi[1] = ParticleBoundaryType::Thermal;
        }
        if (name == "reflecting") {
            c.phi[1] = ParticleBoundaryType::Reflecting;
        }
        fields.endpoint_density = &rho;
        fields.current = view(c.ji);
        fields.current_deposit_ghosts =
            amrex::IntVect(name == "current_reach" ? 0 : 4);
        fields.gather_filled_ghosts =
            amrex::IntVect(name == "gather_reach" ? 0 : 4);
        for (int d = 0; d < 3; ++d) {
            eg[d] = std::make_unique<MultiFab>(rho.boxArray(), c.dm, 1, 4);
            bg[d] = std::make_unique<MultiFab>(rho.boxArray(), c.dm, 1, 4);
            eg[d]->setVal(1);
            bg[d]->setVal(1);
            fields.gather_e[d] = eg[d].get();
            fields.gather_b[d] = bg[d].get();
        }
        amrex::Vector<std::string> names;
        for (auto const* s : PIdx::names) {
            names.push_back(s);
        }
        pc.SetSoACompileTimeNames(names, {});
        for (auto const* s : {"x_n", "y_n", "z_n", "ux_n", "uy_n", "uz_n"}) {
            pc.AddRealComp(s, 0);
        }
        pc.AddIntComp("nsuborbits", 0);
        auto const dx = c.geom.CellSizeArray();
        for (amrex::MFIter it(c.u); it.isValid(); ++it) {
            auto const box0 = it.validbox();
            auto const lo = box0.smallEnd();
            auto& tile = pc.DefineAndReturnParticleTile(0, it.index(), 0);
            tile.resize(1);
            auto data = tile.getParticleTileData();
            auto* counter =
                tile.GetStructOfArrays().GetIntData("nsuborbits").data();
            Point x0{}, x1{};
#ifdef WARPX_DIM_RZ
            x0 = {PR((lo[0] + 1.3) * dx[0]), 0, PR((lo[1] + 1.3) * dx[1])};
            x1 = {PR((lo[0] + 1.6) * dx[0]), 0, PR((lo[1] + 1.6) * dx[1])};
            constexpr int axial = 2;
#else
            x0 = {PR((lo[0] + 1.3) * dx[0]), PR((lo[1] + 1.3) * dx[1]),
                  PR((lo[2] + 1.3) * dx[2])};
            x1 = {PR((lo[0] + 1.6) * dx[0]), PR((lo[1] + 1.6) * dx[1]),
                  PR((lo[2] + 1.6) * dx[2])};
            constexpr int axial = 1;
#endif
            if (name == "axis" && lo[0] == 0) {
                x0[0] = .1 * dx[0];
                x1[0] = .3 * dx[0];
            }
            if ((name == "periodic" || name == "absorbing" ||
                 name == "thermal" || name == "reflecting") &&
                box0.bigEnd(1) == 15) {
                x0[axial] = 15.7 * dx[1];
                x1[axial] = 16.2 * dx[1];
            }
            if ((name == "wall" || name == "near_wall" ||
                 name == "resolved_wall") &&
                box0.bigEnd(0) == 15) {
                x0[0] = 15.7 * dx[0];
                x1[0] = name == "resolved_wall" ? 15.2 * dx[0]
                        : name == "near_wall"   ? std::nextafter(PR(2), PR(0))
                                                : 2;
            }
            if (name == "gather_reach" || name == "current_reach" ||
                name == "aba") {
                x0[0] = (lo[0] + .1) * dx[0];
                x1[0] = (lo[0] + .2) * dx[0];
            }
            if (name == "endpoint_reach") {
                x1[0] = (box0.bigEnd(0) + .7) * dx[0];
            }
            if (name == "projection_reach") {
                x0[axial] = (lo[1] + 1.3) * dx[1];
                x1[axial] = (lo[1] - .5) * dx[1];
            }
            if (name == "old_outside" && lo[1] == 0) {
                x0[axial] = -.25 * dx[1];
                x1[axial] = .5 * dx[1];
            }
            auto& a = tile.GetStructOfArrays();
            auto xn = a.GetRealData("x_n").data(),
                 yn = a.GetRealData("y_n").data(),
                 zn = a.GetRealData("z_n").data();
            auto un = a.GetRealData("ux_n").data(),
                 vn = a.GetRealData("uy_n").data(),
                 wn = a.GetRealData("uz_n").data();
            bool const bad = name == "nonfinite";
            bool const invalid = name == "identity" && it.index() == 0;
            int const grid = it.index(),
                      rank = amrex::ParallelDescriptor::MyProc();
            amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int i) {
                auto particle = data[i];
                counter[i] = 777;
                particle.id() = grid + 1;
                particle.cpu() = rank;

                xn[i] = x0[0];
                yn[i] = x0[1];
                zn[i] = x0[2];
                un[i] = .1;
                vn[i] = .2;
                wn[i] = .3;
                for (int d = 0; d < PIdx::nattribs; ++d) {
                    data.rdata(d)[i] = .1 + d;
                }
                data.rdata(PIdx::ux)[i] =
                    bad ? std::numeric_limits<PR>::quiet_NaN() : .1;
                data.rdata(PIdx::uy)[i] = .2;
                data.rdata(PIdx::uz)[i] = .3;
                data.rdata(PIdx::w)[i] = 1;
#ifdef WARPX_DIM_RZ
                particle.pos(0) = .5 * (x0[0] + x1[0]);
                particle.pos(1) = .5 * (x0[2] + x1[2]);
#else
                for(int d=0;d<3;++d) { particle.pos(d)=.5*(x0[d]+x1[d]); }
#endif
#ifdef WARPX_DIM_RZ
                data.rdata(PIdx::theta)[i] = 0;
#endif
                if (invalid) {
                    amrex::ParticleIDWrapper{data.m_idcpu[i]}.make_invalid();
                }
            });
        }
        amrex::Gpu::synchronize();
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto& a = it.GetStructOfArrays();
            int const nr = a.NumRealComps();
            auto& snap = snapshot[it.index()];
            snap.resize(nr);
            for (int j = 0; j < nr; ++j) {
                amrex::Gpu::copy(amrex::Gpu::deviceToDevice,
                                 a.GetRealData(j).begin(),
                                 a.GetRealData(j).end(), snap.begin() + j);
            }
            auto& id = ids[it.index()];
            id.resize(1);
            amrex::Gpu::copy(amrex::Gpu::deviceToDevice,
                             a.GetIdCPUData().begin(), a.GetIdCPUData().end(),
                             id.begin());
        }
    }
    EndpointAuditResult
    audit () {
        EndpointAuditResult total;
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto const& a = it.GetStructOfArrays();
            auto const contract = MakeEndpointTileContract(
                c.geom, it.tilebox(), it.index(), fields, c.plo, c.phi);
            auto const r = AuditEndpointTile(
                it.numParticles(), GetParticlePosition<PIdx>(it), start(it),
                {a.GetRealData(PIdx::ux).data(), a.GetRealData(PIdx::uy).data(),
                 a.GetRealData(PIdx::uz).data()},
                a.GetIdCPUData().data(), contract);
            total.particles += r.particles;
            for (int j = 0; j < endpoint_issue_count; ++j) {
                total.counts[j] += r.counts[j];
            }
        }
        total.reduceAcrossRanks();
        return total;
    }
    void
    purity () {
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto const& a = it.GetStructOfArrays();
            auto const* snap = snapshot.at(it.index()).data();
            for (int j = 0; j < a.NumRealComps(); ++j) {
                auto const* ptr = a.GetRealData(j).data();
                amrex::ReduceOps<amrex::ReduceOpMax> op;
                amrex::ReduceData<int> rd(op);
                op.eval(
                    1, rd, [=] AMREX_GPU_DEVICE(int) -> amrex::GpuTuple<int> {
                        // Compare object bytes so the NaN adversary is tested
                        // too.
                        auto const* lhs =
                            reinterpret_cast<const unsigned char*>(ptr);
                        auto const* rhs =
                            reinterpret_cast<const unsigned char*>(snap + j);
                        int bad = 0;
                        for (std::size_t k = 0; k < sizeof(PR); ++k) {
                            bad |= lhs[k] != rhs[k];
                        }
                        return {bad};
                    });
                check(amrex::get<0>(rd.value()) == 0,
                      "audit preserved every particle real byte");
            }
            auto const* counter = a.GetIntData("nsuborbits").data();
            auto const* saved = ids.at(it.index()).data();
            auto const* id = a.GetIdCPUData().data();
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<int> rd(op);
            op.eval(1, rd, [=] AMREX_GPU_DEVICE(int) -> amrex::GpuTuple<int> {
                return {saved[0] != id[0] || counter[0] != 777};
            });
            check(amrex::get<0>(rd.value()) == 0,
                  "audit preserved IDs and suborbit counters");
        }
    }
    void
    actual_kernels () {
        rho.setVal(0);
        for (auto& j : c.ji) {
            j->setVal(0);
        }
        auto dx = c.geom.CellSizeArray();
#ifdef WARPX_DIM_RZ
        amrex::XDim3 inv{1 / dx[0], 1, 1 / dx[1]};
#else
        amrex::XDim3 inv{1 / dx[0], 1 / dx[1], 1 / dx[2]};
#endif
        for (Iter it(pc, 0); it.isValid(); ++it) {
            auto const contract = MakeEndpointTileContract(
                c.geom, it.tilebox(), it.index(), fields, c.plo, c.phi);
            // Poison outside explicitly qualified gather ghosts; actual native
            // MC gather may read only finite values when the audit accepts.
            for (int f = 0; f < 6; ++f) {
                auto& mf = f < 3 ? *eg[f] : *bg[f - 3];
                auto a = mf.array(it);
                auto const r = contract.gather[f];
                amrex::ParallelFor(
                    mf[it].box(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        amrex::GpuArray<int, AMREX_SPACEDIM> q{
                            AMREX_D_DECL(i, j, k)};
                        bool valid = true;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            valid &= q[d] >= r.low[d] && q[d] <= r.high[d];
                        }
                        a(i, j, k) =
                            valid ? 1 : std::numeric_limits<Real>::quiet_NaN();
                    });
            }
            auto get = GetParticlePosition<PIdx>(it);
            auto s = start(it);
            auto const& a = it.GetStructOfArrays();
            auto end = [=] AMREX_GPU_HOST_DEVICE(long i, PR& x, PR& y, PR& z) {
                get(i, x, y, z);
                x = 2 * x - s.position[0][i];
                y = 2 * y - s.position[1][i];
                z = 2 * z - s.position[2][i];
            };
            amrex::GpuArray<amrex::GpuArray<double, 2>, AMREX_SPACEDIM>
                domain{};
            amrex::GpuArray<amrex::GpuArray<bool, 2>, AMREX_SPACEDIM> crop{};
            auto origin = amrex::XDim3{0, 0, 0};
            auto low = amrex::Dim3{0, 0, 0};
            // Native kernels normally use a ghost-box origin to keep shape
            // coordinates nonnegative; use this exact common origin below.
            auto const box0 = it.tilebox();
            auto scratch = box0;
            scratch.grow(4);
            auto lo = scratch.smallEnd();
#ifdef WARPX_DIM_RZ
            origin = {lo[0] * dx[0], 0, lo[1] * dx[1]};
            low = {lo[0], lo[1], 0};
#else
            origin = {lo[0] * dx[0], lo[1] * dx[1], lo[2] * dx[2]};
            low = {lo[0], lo[1], lo[2]};
#endif
            doChargeDepositionShapeN<3>(end, a.GetRealData(PIdx::w).data(),
                                        nullptr, rho[it], 1, inv, origin, low,
                                        1, 1);
            doChargeConservingDepositionShapeNImplicit<3>(
                s.position[0], s.position[1], s.position[2], get,
                a.GetRealData(PIdx::w).data(), s.momentum[0], s.momentum[1],
                s.momentum[2], a.GetRealData(PIdx::ux).data(),
                a.GetRealData(PIdx::uy).data(), a.GetRealData(PIdx::uz).data(),
                nullptr, c.ji[0]->array(it), c.ji[1]->array(it),
                c.ji[2]->array(it), 1, .1, inv, origin, domain, crop, low, 1,
                1);
            auto e0 = eg[0]->const_array(it), e1 = eg[1]->const_array(it),
                 e2 = eg[2]->const_array(it);
            auto b0 = bg[0]->const_array(it), b1 = bg[1]->const_array(it),
                 b2 = bg[2]->const_array(it);
            auto type = eg[0]->ixType();
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<int> rd(op);
            op.eval(1, rd, [=] AMREX_GPU_DEVICE(int i) -> amrex::GpuTuple<int> {
                PR x, y, z;
                get(i, x, y, z);
                PR ex = 0, ey = 0, ez = 0, bx = 0, by = 0, bz = 0;
                doGatherShapeN<3, 0>(x, y, z, ex, ey, ez, bx, by, bz, e0, e1,
                                     e2, b0, b1, b2, type, type, type, type,
                                     type, type, inv, origin, low, 1);
                return {!std::isfinite(ex + ey + ez + bx + by + bz)};
            });
            check(amrex::get<0>(rd.value()) == 0,
                  "actual shape3 MC gather stays inside filled range");
            for (int f = 0; f < 4; ++f) {
                auto& mf = f == 3 ? rho : *c.ji[f];
                auto arr = mf.const_array(it);
                auto r = f == 3 ? contract.rho : contract.current[f];
                amrex::ReduceOps<amrex::ReduceOpMax> ro;
                amrex::ReduceData<Real> rr(ro);
                ro.eval(mf[it].box(), rr,
                        [=] AMREX_GPU_DEVICE(int i, int j,
                                             int k) -> amrex::GpuTuple<Real> {
                            amrex::GpuArray<int, AMREX_SPACEDIM> q{
                                AMREX_D_DECL(i, j, k)};
                            bool valid = true;
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                                valid &= q[d] >= r.low[d] && q[d] <= r.high[d];
                            }
                            return {valid ? 0 : std::abs(arr(i, j, k))};
                        });
                check(amrex::get<0>(rr.value()) == 0,
                      "actual Esirkepov/endpoint deposit stays inside "
                      "qualified range");
            }
        }
    }
};
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int box = 4;
        std::string name = "interior";
        pp.query("case", name);
        pp.query("max_grid_size", box);
        Test t(box, name);
        auto const seed =
            amrex::ULong(20260925 + amrex::ParallelDescriptor::MyProc());
        amrex::ResetRandomSeed(seed);
        auto const next_random = amrex::Random();
        amrex::ResetRandomSeed(seed);
        auto a = t.audit();
        check(amrex::Random() == next_random, "audit preserved RNG sequence");
        amrex::Print() << "audit counts=";
        for (auto n : a.counts) {
            amrex::Print() << n << ",";
        }
        amrex::Print() << "\n";
        t.purity();
        auto again = t.audit();
        check(a.counts == again.counts, "A/A audit deterministic");
        if (name == "aba") {
            auto wide = t.fields.gather_filled_ghosts;
            t.fields.gather_filled_ghosts = amrex::IntVect(0);
            auto b = t.audit();
            check(!b.ok(), "B narrow range rejects");
            t.fields.gather_filled_ghosts = wide;
            check(t.audit().counts == a.counts,
                  "A/B/A independent range evaluation");
        }
        if (name == "interior" || name == "axis" || name == "periodic" ||
            name == "resolved_wall" || name == "aba") {
            check(a.ok(), "qualified endpoint accepts");
            t.actual_kernels();
        } else {
            EndpointIssue expect = EndpointIssue::NonFinite;
            if (name == "absorbing") {
                expect = EndpointIssue::AbsorbingBoundary;
            }
            if (name == "thermal") {
                expect = EndpointIssue::ThermalBoundary;
            }
            if (name == "reflecting") {
                expect = EndpointIssue::OtherBoundaryAction;
            }
            if (name == "wall" || name == "near_wall") {
#ifdef WARPX_DIM_RZ
                expect = EndpointIssue::UnresolvedRadialWall;
#else
                expect = EndpointIssue::OtherBoundaryAction;
#endif
            }
            if (name == "gather_reach") {
                expect = EndpointIssue::GatherReach;
            }
            if (name == "current_reach") {
                expect = EndpointIssue::CurrentReach;
            }
            if (name == "endpoint_reach") {
                expect = EndpointIssue::EndpointReach;
            }
            if (name == "projection_reach") {
                expect = EndpointIssue::ProjectionReach;
                check(a.count(EndpointIssue::EndpointReach) == 0,
                      "projection guard stricter than actual shape reach");
            }
            if (name == "identity") {
                expect = EndpointIssue::InvalidIdentity;
            }
            if (name == "old_outside") {
                expect = EndpointIssue::OldOutsideDomain;
            }
            check(a.count(expect) > 0, "expected typed rejection");
        }
        t.purity();
        amrex::Print() << "PASS " << name << " boxes=" << t.c.ba.size()
                       << " ranks=" << amrex::ParallelDescriptor::NProcs()
                       << " particles=" << a.particles << " counts=";
        for (auto n : a.counts) {
            amrex::Print() << n << ",";
        }
        amrex::Print() << '\n';
    }
    amrex::Finalize();
}
