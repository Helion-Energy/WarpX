/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/AbsorbingElectronInventory.H"
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>
#include <string>
using namespace warpx::thermal;
using amrex::MultiFab;
using amrex::Real;
constexpr Real eps = std::numeric_limits<Real>::epsilon();

MultiFab
copy (const MultiFab& a) {
    MultiFab b(a.boxArray(), a.DistributionMap(), 1, 0);
    MultiFab::Copy(b, a, 0, 0, 1, 0);
    return b;
}
Real
difference (const MultiFab& a, const MultiFab& b) {
    auto c = copy(a);
    MultiFab::Subtract(c, b, 0, 0, 1, 0);
    return c.norm0();
}
Real
integral (const MultiFab& a, const amrex::Geometry& g) {
    auto dx = g.CellSizeArray(), lo = g.ProbLoArray();
    auto first = g.Domain().smallEnd();
    Real vol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    bool rz = g.IsRZ();
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<Real> data(op);
    for (amrex::MFIter it(a); it.isValid(); ++it) {
        auto f = a.const_array(it);
        op.eval(
            it.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> decltype(data)::Type {
                // Annulus area is an independent physical-volume oracle.
                Real r0 = lo[0] + (i - first[0]) * dx[0], r1 = r0 + dx[0];
                Real v = rz ? 3.1415926535897932384626433832795 *
                                  (r1 * r1 - r0 * r0) * dx[1]
                            : vol;
                return {v * f(i, j, k)};
            });
    }
    Real sum = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}
void
ledger (const AbsorbingElectronInventoryResult& r, const MultiFab& uv,
        const MultiFab& us, const amrex::Geometry& g) {
    AMREX_ALWAYS_ASSERT(r.valid && r.max_temperature_relative_error < 16 * eps);
    auto loss = copy(uv);
    MultiFab::Subtract(loss, us, 0, 0, 1, 0);
    Real physical = integral(loss, g);
    Real scale = std::max({Real(1), r.absolute_raw_requested_joule,
                           r.absolute_represented_joule,
                           r.absolute_floor_replacement_joule});
    AMREX_ALWAYS_ASSERT(std::abs(physical - r.represented_joule) <
                        128 * eps * scale);
    AMREX_ALWAYS_ASSERT(std::abs(r.raw_requested_joule - r.represented_joule -
                                 r.floor_replacement_joule) <
                        128 * eps * scale);
}
void
algebra (const amrex::Geometry& g, const amrex::BoxArray& ba,
         const amrex::DistributionMapping& dm) {
    MultiFab uv(ba, dm, 1, 1), nv(ba, dm, 1, 1), ns(ba, dm, 1, 1),
        lost(ba, dm, 1, 1), us(ba, dm, 1, 1);
    uv.setVal(std::numeric_limits<Real>::quiet_NaN());
    nv.setVal(std::numeric_limits<Real>::quiet_NaN());
    ns.setVal(std::numeric_limits<Real>::quiet_NaN());
    lost.setVal(std::numeric_limits<Real>::quiet_NaN());
    auto first = g.Domain().smallEnd();
    for (int scenario = 0; scenario < 6; ++scenario) {
        for (amrex::MFIter it(uv); it.isValid(); ++it) {
            auto u = uv.array(it), v = nv.array(it), s = ns.array(it),
                 l = lost.array(it);
            amrex::ParallelFor(
                it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real q = 1. + .01 * (i - first[0] + j - first[1] + k);
                    v(i, j, k) = 2.e18 * q;
                    u(i, j, k) = 10. * q;
                    s(i, j, k) = v(i, j, k);
                    l(i, j, k) = 0.;
                    if (scenario == 1) {
                        s(i, j, k) = .75 * v(i, j, k);
                        l(i, j, k) = .25 * v(i, j, k);
                    }
                    if (scenario == 2) {
                        l(i, j, k) = .4 * v(i, j, k);
                    }
                    if (scenario == 3) {
                        s(i, j, k) = .75 * v(i, j, k);
                        l(i, j, k) = .5 * v(i, j, k);
                    }
                    if (scenario ==
                        4) { // signed cancellation with equal-volume z/y pairs
                        v(i, j, k) = 2.e18;
                        u(i, j, k) = 10.;
                        s(i, j, k) = v(i, j, k);
                        l(i, j, k) = ((j - first[1]) % 2 ? -1. : 1.) * 1.e18;
                    }
                    if (scenario == 5) {
                        s(i, j, k) = v(i, j, k) * (1. + 4 * eps);
                        l(i, j, k) = -4 * eps * v(i, j, k);
                    }
                });
        }
        auto old_u = copy(uv), old_v = copy(nv), old_s = copy(ns),
             old_l = copy(lost);
        us.setVal(-719.);
        auto r = RemapAbsorbingElectronInventory(g, uv, nv, ns, lost, us);
        ledger(r, uv, us, g);
        AMREX_ALWAYS_ASSERT(
            difference(uv, old_u) == 0 && difference(nv, old_v) == 0 &&
            difference(ns, old_s) == 0 && difference(lost, old_l) == 0);
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<Real> data(op);
        for (amrex::MFIter it(us); it.isValid(); ++it) {
            auto u = uv.const_array(it), s = us.const_array(it);
            auto valid = it.validbox();
            op.eval(
                it.fabbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j,
                                     int k) -> decltype(data)::Type {
                    if (!valid.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                        return {std::abs(s(i, j, k) + 719.)};
                    if (scenario == 0 || scenario == 2 || scenario == 4)
                        return {std::abs(s(i, j, k) - u(i, j, k))};
                    return {0.};
                });
        }
        Real e = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(e);
        AMREX_ALWAYS_ASSERT(e == 0.);
        if (scenario == 0)
            AMREX_ALWAYS_ASSERT(r.raw_requested_joule == 0 &&
                                r.represented_joule == 0 &&
                                r.floor_replacement_joule == 0);
        if (scenario == 2)
            AMREX_ALWAYS_ASSERT(
                r.raw_requested_joule > 0 && r.represented_joule == 0 &&
                r.floor_replacement_joule == r.raw_requested_joule);
        if (scenario == 4)
            AMREX_ALWAYS_ASSERT(std::abs(r.raw_requested_joule) <
                                    64 * eps * r.absolute_raw_requested_joule &&
                                r.absolute_raw_requested_joule > 0 &&
                                r.represented_joule == 0);
        if (scenario == 5)
            AMREX_ALWAYS_ASSERT(r.max_survivor_density_relative_increase > 0 &&
                                r.represented_joule < 0 &&
                                r.raw_requested_joule < 0);
        auto a = copy(us);
        auto prior = r;
        ns.setVal(1.e18, 0, 1, 0);
        AMREX_ALWAYS_ASSERT(
            RemapAbsorbingElectronInventory(g, uv, nv, ns, lost, us).valid);
        MultiFab::Copy(ns, old_s, 0, 0, 1, 0);
        r = RemapAbsorbingElectronInventory(g, uv, nv, ns, lost, us);
        AMREX_ALWAYS_ASSERT(
            difference(a, us) == 0 &&
            r.raw_requested_joule == prior.raw_requested_joule &&
            r.represented_joule == prior.represented_joule &&
            r.floor_replacement_joule == prior.floor_replacement_joule);
    }
    for (int bad = 0; bad < 7; ++bad) {
        uv.setVal(10.);
        nv.setVal(2.e18);
        ns.setVal(1.e18);
        lost.setVal(1.e18);
        if (bad == 0)
            nv.setVal(0.);
        if (bad == 1)
            ns.setVal(0.);
        if (bad == 2)
            uv.setVal(-1.);
        if (bad == 3)
            lost.setVal(std::numeric_limits<Real>::quiet_NaN());
        if (bad == 4)
            uv.setVal(std::numeric_limits<Real>::infinity());
        if (bad == 5) {
            uv.setVal(std::numeric_limits<Real>::max());
            nv.setVal(1.);
            ns.setVal(2.);
        }
        if (bad == 6) {
            uv.setVal(std::numeric_limits<Real>::max());
            nv.setVal(1.);
            ns.setVal(1.);
            lost.setVal(std::numeric_limits<Real>::max());
        }
        AMREX_ALWAYS_ASSERT(
            !RemapAbsorbingElectronInventory(g, uv, nv, ns, lost, us).valid);
    }
    uv.setVal(-0.);
    nv.setVal(1.);
    ns.setVal(1.);
    lost.setVal(0.);
    AMREX_ALWAYS_ASSERT(
        RemapAbsorbingElectronInventory(g, uv, nv, ns, lost, us).valid);
    amrex::ReduceOps<amrex::ReduceOpMin> op;
    amrex::ReduceData<int> data(op);
    for (amrex::MFIter it(us); it.isValid(); ++it) {
        auto a = us.const_array(it);
        op.eval(
            it.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> decltype(data)::Type {
                return {int(std::signbit(a(i, j, k)))};
            });
    }
    int sign = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceIntMin(sign);
    AMREX_ALWAYS_ASSERT(sign != 0);
}
void
moments (const amrex::Geometry& g, const amrex::BoxArray& ba,
         const amrex::DistributionMapping& dm, bool corrected) {
    auto nodes = amrex::convert(ba, amrex::IntVect::TheNodeVector());
    MultiFab rv(nodes, dm, 1, 0), rs(nodes, dm, 1, 0), rl(nodes, dm, 1, 0),
        ped(nodes, dm, 1, 0);
    MultiFab uv(ba, dm, 1, 0), us(ba, dm, 1, 0), lost(ba, dm, 1, 0);
    ThermalMomentOptions options;
    options.number_density_floor = 2.e18;
    options.active_density_floor = 1.e17;
    options.verboncoeur_axis_correction = corrected;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        for (int side = 0; side < 2; ++side)
            options.boundary[d][side] = g.isPeriodic(d)
                                            ? MomentBoundary::Periodic
                                            : MomentBoundary::PMC;
#ifdef WARPX_DIM_RZ
    options.boundary[0][0] = MomentBoundary::Axis;
    options.boundary[0][1] = MomentBoundary::PEC;
    options.current_boundary[0][1] = MomentBoundary::PMC;
#endif
    for (int scenario = 0; scenario < 4; ++scenario) {
        rv.setVal(.7e18 * PhysConst::q_e);
        rs.setVal(.2e18 * PhysConst::q_e);
        ped.setVal((scenario == 0   ? 0.
                    : scenario == 1 ? 1.5e18
                                    : 2.e18) *
                   PhysConst::q_e);
        if (scenario == 3) {
            for (amrex::MFIter it(rv); it.isValid(); ++it) {
                auto v = rv.array(it), s = rs.array(it);
                amrex::ParallelFor(
                    it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        v(i, j, k) = (.1 + .07 * i + .03 * j + .02 * k) *
                                     1.e18 * PhysConst::q_e;
                        s(i, j, k) = .3 * v(i, j, k);
                    });
            }
        }
        MultiFab::LinComb(rl, 1., rv, 0, -1., rs, 0, 0, 1, 0);
        KineticThermalMoments mv(g, ba, dm, options), ms(g, ba, dm, options);
        KineticThermalStateView v{rv}, s{rs};
        v.pedestal = &ped;
        s.pedestal = &ped;
        AMREX_ALWAYS_ASSERT(mv.Evaluate(v) && ms.Evaluate(s));
        MultiFab::Copy(uv, mv.NumberDensity(), 0, 0, 1, 0);
        uv.mult(2.e-17);
        v.energy = &uv;
        AMREX_ALWAYS_ASSERT(mv.Evaluate(v));
        mv.RestrictNativeMoment(rl, 0, lost);
        lost.mult(1. / PhysConst::q_e);
        auto r = RemapAbsorbingElectronInventory(g, uv, mv.NumberDensity(),
                                                 ms.NumberDensity(), lost, us);
        ledger(r, uv, us, g);
        s.energy = &us;
        AMREX_ALWAYS_ASSERT(ms.Evaluate(s));
        AMREX_ALWAYS_ASSERT(
            difference(mv.NodalTemperature(), ms.NodalTemperature()) <
            64 * eps * mv.NodalTemperature().norm0());
        if (scenario == 0)
            AMREX_ALWAYS_ASSERT(r.represented_joule == 0 &&
                                r.floor_replacement_joule ==
                                    r.raw_requested_joule &&
                                r.raw_requested_joule > 0);
        if (scenario == 1)
            AMREX_ALWAYS_ASSERT(r.represented_joule > 0 &&
                                r.floor_replacement_joule > 0);
        if (scenario >= 2)
            AMREX_ALWAYS_ASSERT(std::abs(r.floor_replacement_joule) <
                                128 * eps * r.absolute_raw_requested_joule);
        AMREX_ALWAYS_ASSERT(r.max_survivor_density_relative_increase == 0.);
    }
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int grid = 4, periodic = 0, axis = 1;
        std::string which = "algebra";
        pp.query("grid", grid);
        pp.query("periodic", periodic);
        pp.query("axis", axis);
        pp.query("case", which);
        int shift = which == "algebra" ? 3 : 0;
        amrex::Box domain(amrex::IntVect(shift), amrex::IntVect(shift + 7));
        amrex::RealBox real({AMREX_D_DECL(0., 0., 0.)},
                            {AMREX_D_DECL(2., 3., 4.)});
        amrex::Array<int, AMREX_SPACEDIM> periods{};
        for (int d = 1; d < AMREX_SPACEDIM; ++d)
            periods[d] = periodic;
#ifdef WARPX_DIM_RZ
        int coord = 1;
#else
        int coord = 0;
#endif
        amrex::Geometry geom(domain, &real, coord, periods.data());
        amrex::BoxArray ba(domain);
        ba.maxSize(grid);
        amrex::DistributionMapping dm(ba);
        if (which == "algebra")
            algebra(geom, ba, dm);
        else
            moments(geom, ba, dm, axis != 0);
        amrex::Print() << "ABSORBING_INVENTORY_PASS case=" << which
                       << " grid=" << grid << " periodic=" << periodic
                       << " axis=" << axis << "\n";
    }
    amrex::Finalize();
}
