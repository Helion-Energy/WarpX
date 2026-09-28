/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/ThermalConductionPC.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <fstream>
using namespace warpx::thermal;
using Real = amrex::Real;
using MF = amrex::MultiFab;
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        AMREX_ALWAYS_ASSERT(amrex::ParallelDescriptor::NProcs() == 1);
        std::string path = "snapshot.bin";
        amrex::ParmParse("test").query("output", path);
        std::ofstream stream(path, std::ios::binary);
        AMREX_ALWAYS_ASSERT(stream.good());
        amrex::Box box(amrex::IntVect(0), amrex::IntVect(11, 15));
        amrex::BoxArray ba(box);
        amrex::DistributionMapping dm(ba);
        amrex::RealBox real({0., 0.}, {1., 2.});
        int per[2] = {0, 1};
        amrex::Geometry geom(box, &real, 1, per);
        auto dx = geom.CellSizeArray();
        MF old(ba, dm, 1, 0), n(ba, dm, 1, 0), b(ba, dm, 3, 0), q(ba, dm, 1, 0),
            u(ba, dm, 1, 0), direction(ba, dm, 1, 0), mask(ba, dm, 1, 0),
            out(ba, dm, 1, 0);
        for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
            auto un = old.array(mfi), nn = n.array(mfi), bb = b.array(mfi),
                 qq = q.array(mfi), us = u.array(mfi), v = direction.array(mfi),
                 active = mask.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                Real r = (i + .5) * dx[0], z = (j + .5) * dx[1];
                nn(i, j, k) =
                    1.e18 * (1. + .2 * r + .1 * std::sin(MathConst::pi * z));
                un(i, j, k) = nn(i, j, k) * PhysConst::q_e / (2. / 3.) *
                              (2. + .1 * r + .1 * std::cos(MathConst::pi * z));
                us(i, j, k) = 1.03 * un(i, j, k);
                v(i, j, k) = .07 * (1. + std::sin(.7 * i + 1.2 * j));
                bb(i, j, k, 0) = .2 * r;
                bb(i, j, k, 1) = .4;
                bb(i, j, k, 2) = 1. + .05 * std::cos(MathConst::pi * z);
                qq(i, j, k) = .003 * r;
                active(i, j, k) = i == 4 && j > 4 && j < 9 ? 0. : 1.;
            });
        }
        std::array<std::unique_ptr<MF>, 2> velocity;
        ThermalFaceContext faces;
        faces.conduction_active = &mask;
        for (int d = 0; d < 2; ++d) {
            velocity[d] = std::make_unique<MF>(
                amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)), dm,
                1, 0);
            faces.velocity[d] = velocity[d].get();
            for (amrex::MFIter mfi(*velocity[d]); mfi.isValid(); ++mfi) {
                auto v = velocity[d]->array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        v(i, j, k) =
                            d == 0 ? .04 * i * dx[0]
                                   : .03 * std::sin(MathConst::pi * j * dx[1]);
                    });
            }
        }
        auto record = [&] (MF const& f) {
            for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
                auto a = f.const_array(mfi);
                auto bx = mfi.validbox();
                auto lo = bx.smallEnd(), hi = bx.bigEnd();
                for (int j = lo[1]; j <= hi[1]; ++j)
                    for (int i = lo[0]; i <= hi[0]; ++i)
                        for (int c = 0; c < f.nComp(); ++c) {
                            Real value = a(i, j, 0, c);
                            stream.write(reinterpret_cast<char const*>(&value),
                                         sizeof(value));
                        }
            }
        };
        for (int kind : {0, 1}) {
            ThermalStageOptions o;
            o.dt = .03;
            o.order = 4;
            o.cross_mode = 1;
            o.conduction_theta = 1.;
            o.free_streaming_fraction = .02;
            o.kappa_parallel = "0.000003*Te^1.5";
            o.kappa_perpendicular = "0.0000002";
            o.boundary[0][1].kind =
                kind ? BoundaryKind::Leg : BoundaryKind::Reservoir;
            o.boundary[0][1].value = .5;
            o.boundary[0][1].flux_limit = kind ? .03 : 0.;
            EulerianThermalStage stage(geom, old, n, n, n, b, q, faces, o);
            AMREX_ALWAYS_ASSERT(stage.Residual(u, out));
            record(out);
            for (int d = 0; d < 2; ++d) {
                record(stage.FaceFlux(d));
                record(stage.FaceCoefficient(d));
                record(stage.FaceWallDerivative(d));
                record(stage.FaceAdvectionFlux(d));
            }
            ThermalConductionPC pc(stage);
            AMREX_ALWAYS_ASSERT(pc.Freeze(u));
            pc.ApplyOperator(out, direction);
            record(out);
            pc.ApplyRows(out, direction);
            record(out);
            pc.Apply(out, direction);
            record(out);
        }
        stream.close();
        amrex::Print() << "PASS RZ byte snapshot " << path << "\n";
    }
    amrex::Finalize();
}
