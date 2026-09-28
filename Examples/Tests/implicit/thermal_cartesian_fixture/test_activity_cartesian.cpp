/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/EulerianThermalStage.H"
#include "FieldSolver/ImplicitSolvers/NativeConductionActivity.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
using namespace warpx::thermal;
using MF = amrex::MultiFab;
using Real = amrex::Real;
namespace {
constexpr Real density = 1.e18;
void
require (bool value, const char* text) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(value, text);
}
void
near (Real error, Real tolerance, const char* text) {
    amrex::Print() << "ACTIVITY3D " << text << " error=" << error
                   << " tolerance=" << tolerance << "\n";
    require(std::isfinite(error) && std::abs(error) <= tolerance, text);
}
struct Context {
    amrex::Geometry geometry;
    amrex::BoxArray boxes;
    amrex::DistributionMapping dm;
    MF rho, ped;
    Context (int box, int n = 16)
        : boxes(amrex::Box(amrex::IntVect(0), amrex::IntVect(n - 1))) {
        amrex::RealBox real({0., 0., 0.}, {1., 1., 1.});
        int periodic[3] = {1, 1, 1};
        geometry = amrex::Geometry(boxes.minimalBox(), &real, 0, periodic);
        boxes.maxSize(box);
        dm = amrex::DistributionMapping(boxes);
        rho.define(amrex::convert(boxes, amrex::IntVect::TheNodeVector()), dm,
                   1, 1);
        ped.define(rho.boxArray(), dm, 1, 1);
        rho.setVal(2. * density * PhysConst::q_e);
        ped.setVal(0.);
    }
    ConductionActivityOptions
    options () const {
        ConductionActivityOptions o;
        o.closure = ConductionCellClosure::AllContributors;
        o.legacy_density_floor = density;
        o.common_density_floor = 2. * density;
        return o;
    }
    void
    plane () {
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            auto r = rho.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    r(i, j, k) = density * PhysConst::q_e * (k == 4 ? .5 : 2.);
                });
        }
    }
};
Real
difference (MF const& a, MF const& b, int ghosts = 0) {
    MF tmp(a.boxArray(), a.DistributionMap(), 1, ghosts);
    MF::LinComb(tmp, 1, a, 0, -1, b, 0, 0, 1, ghosts);
    return tmp.norm0(0, ghosts);
}
void
gate (Context& c) {
    c.plane();
    NativeConductionActivity map(c.geometry, c.boxes, c.dm, c.options());
    MF saved(c.rho.boxArray(), c.dm, 1, 1);
    MF::Copy(saved, c.rho, 0, 0, 1, 1);
    require(map.Evaluate(c.rho), "native plane evaluates");
    near(difference(saved, c.rho, 1), 0, "native caller guards unchanged");
    MF error(c.boxes, c.dm, 1, 0);
    for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
        auto e = error.array(mfi);
        auto open = map.CellOpen().const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                e(i, j, k) = open(i, j, k) - (k == 3 || k == 4 ? 0. : 1.);
            });
    }
    near(error.norm0(), 0, "all eight native contributors");
    for (int d = 0; d < 3; ++d) {
        MF err(map.NativeEdgeOpen(d).boxArray(), c.dm, 1, 0);
        for (amrex::MFIter mfi(err); mfi.isValid(); ++mfi) {
            auto e = err.array(mfi);
            auto open = map.NativeEdgeOpen(d).const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    e(i, j, k) = open(i, j, k) -
                                 ((k == 4 || (d == 2 && k == 3)) ? 0. : 1.);
                });
        }
        near(err.norm0(), 0, "native edge endpoint graph");
    }
    auto m = map.Measure();
    Real const h = c.geometry.CellSize(2);
    near(m.native_open_volume - (1. - h), 1.e-14,
         "physical Cartesian native-dual volume");
    near(m.retained_open_volume - (1. - 2. * h), 1.e-14,
         "physical all-eight-cell volume");
    near(m.mixed_cell_volume - 2. * h, 1.e-14, "mixed-cell Cartesian volume");
    near(m.lost_open_volume - h, 1.e-14, "insulating erosion measured");
    c.rho.setVal(.5 * density * PhysConst::q_e);
    require(map.Evaluate(c.rho), "low native density");
    near(map.CellOpen().norm0(), 0, "native floor closes");
    require(map.Evaluate(c.rho, 0, &c.ped), "zero pedestal pointer");
    near(map.CellOpen().min(0) - 1., 0,
         "pedestal pointer opens positive native values");
    near(map.Measure().max_active_capacity_relative_gap - 1., 1.e-15,
         "unequal capacities remain explicit");
    auto old = map.CellOpen().sum(0);
    c.ped.setVal(-1.);
    require(!map.Evaluate(c.rho, 0, &c.ped), "invalid pedestal rejects");
    near(map.CellOpen().sum(0) - old, 0, "invalid evaluation keeps old mask");
}
void
refinement (int box) {
    for (int n : {8, 16, 32}) {
        Context c(box, n);
        auto dx = c.geometry.CellSizeArray();
        for (amrex::MFIter mfi(c.rho); mfi.isValid(); ++mfi) {
            auto rho = c.rho.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   rho(i, j, k) = density * PhysConst::q_e *
                                                  (1.6 - (i % n) * dx[0]);
                               });
        }
        NativeConductionActivity map(c.geometry, c.boxes, c.dm, c.options());
        require(map.Evaluate(c.rho), "plane refinement");
        auto m = map.Measure();
        Real const retained =
            Real(int(.6 * n)) / n; // compatible periodic endpoints; seam is a
                                   // second material interface
        // The material law has a jump across the periodic seam: all-contributor
        // mixed-cell measure includes that interface too.
        near(m.retained_open_volume - retained, 2.e-14,
             "Cartesian threshold retained volume");
        require(.6 - m.retained_open_volume >= 0 &&
                    .6 - m.retained_open_volume <= 1. / n,
                "Cartesian O(h) boundary erosion");
        require(m.lost_open_volume >= 0 &&
                    m.lost_open_volume <= m.mixed_cell_volume + 1.e-14,
                "native lost volume bounded by mixed cells");
        amrex::Print() << "REFINEMENT3D n=" << n
                       << " native=" << m.native_open_volume
                       << " retained=" << m.retained_open_volume
                       << " mixed=" << m.mixed_cell_volume
                       << " lost=" << m.lost_open_volume << "\n";
    }
}
void
masked (Context& c) {
    c.plane();
    NativeConductionActivity map(c.geometry, c.boxes, c.dm, c.options());
    require(map.Evaluate(c.rho), "3D mapped mask");
    MF old(c.boxes, c.dm, 1, 0), n(c.boxes, c.dm, 1, 0), b(c.boxes, c.dm, 3, 0),
        q(c.boxes, c.dm, 1, 0), r(c.boxes, c.dm, 1, 0);
    n.setVal(2. * density);
    b.setVal(0);
    q.setVal(0);
    auto dx = c.geometry.CellSizeArray();
    Real const scale = 2. * density * PhysConst::q_e / (2. / 3.);
    for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
        auto u = old.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                u(i, j, k) =
                    scale *
                    (2. + .3 * std::cos(2. * MathConst::pi * (k + .5) * dx[2]));
            });
    }
    ThermalStageOptions opts;
    opts.dt = .01;
    opts.kappa_parallel = "0.000001";
    opts.kappa_perpendicular = "0.000001";
    ThermalFaceContext faces;
    faces.conduction_active = &map.CellOpen();
    EulerianThermalStage stage(c.geometry, old, n, n, n, b, q, faces, opts);
    require(stage.Residual(old, r),
            "production residual with mapped 3D activity");
    near(r.sum(0) * dx[0] * dx[1] * dx[2] / scale, 1.e-14,
         "Cartesian extensive mask energy balance");
    for (int d = 0; d < 3; ++d) {
        MF error(stage.FaceFlux(d).boxArray(), c.dm, 1, 0);
        for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
            auto e = error.array(mfi);
            auto f = stage.FaceFlux(d).const_array(mfi),
                 open = map.FaceOpen(d).const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    e(i, j, k) = open(i, j, k) == 0 ? std::abs(f(i, j, k)) : 0.;
                });
        }
        near(error.norm0(), 0, "actual Cartesian closed-face heat flux");
    }
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse p("test");
        std::string test = "gate";
        int box = 4;
        p.query("case", test);
        p.query("box", box);
        if (test == "refinement")
            refinement(box);
        else {
            Context c(box);
            if (test == "gate")
                gate(c);
            else if (test == "masked")
                masked(c);
            else
                amrex::Abort("unknown Cartesian activity case");
        }
        amrex::Print() << "PASS activity_cartesian " << test << "\n";
    }
    amrex::Finalize();
}
