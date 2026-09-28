/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/FrozenPressureFieldCoupling.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <limits>
#include <string>

using namespace warpx::thermal;
using amrex::Real;
using MF = amrex::MultiFab;
using Fields = std::array<std::unique_ptr<MF>, 3>;

namespace {
amrex::IntVect
yee (int c) {
    auto v = amrex::IntVect::TheNodeVector();
    if (c != 1) {
        v[c / 2] = 0;
    }
    return v;
}
Fields
fields (const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
        int ng = 0) {
    Fields result;
    for (int c = 0; c < 3; ++c) {
        result[c] = std::make_unique<MF>(amrex::convert(ba, yee(c)), dm, 1, ng);
    }
    return result;
}
FrozenPressureFieldCoupling::FieldView
view (Fields& f) {
    return {f[0].get(), f[1].get(), f[2].get()};
}
FrozenPressureFieldCoupling::ConstFieldView
cview (const Fields& f) {
    return {f[0].get(), f[1].get(), f[2].get()};
}
Real
norm (const Fields& a) {
    Real r = 0;
    for (int c = 0; c < 3; ++c) {
        r = std::max(r, a[c]->norm0(0, 0));
    }
    return r;
}
Real
error (const MF& a, const MF& b, Real alpha = 1.0, int ng = 0) {
    MF tmp(a.boxArray(), a.DistributionMap(), 1, ng);
    MF::LinComb(tmp, 1., a, 0, -alpha, b, 0, 0, 1, ng);
    return tmp.norm0(0, ng);
}
Real
error (const Fields& a, const Fields& b, Real alpha = 1.0) {
    Real r = 0;
    for (int c = 0; c < 3; ++c) {
        r = std::max(r, error(*a[c], *b[c], alpha));
    }
    return r;
}
void
copy (Fields& to, const Fields& from) {
    for (int c = 0; c < 3; ++c) {
        MF::Copy(*to[c], *from[c], 0, 0, 1, 0);
    }
}
void
check (Real err, Real scale, Real tol, const std::string& label) {
    amrex::Print() << "PRESSURE " << label << " error=" << err
                   << " scale=" << scale
                   << " relative=" << err / std::max(scale, Real(1.e-100))
                   << "\n";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(err) && err <= tol * std::max(scale, Real(1.e-100)),
        label.c_str());
}
void
exact (bool okay, const std::string& label) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(okay, label.c_str());
}

void
weights (Fields& w, const MF& rho, const MF& ped, const amrex::Geometry& geom) {
    PressureResponseParameters p;
    p.charge_floor = 0.2;
    p.floor_width = 0.03;
    p.holmstrom = true;
    p.holmstrom_width = 0.05;
    p.axis_radius = 0.45;
    p.axis_rolloff = 0.1;
    auto const dx = geom.CellSizeArray();
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(*w[c]); mfi.isValid(); ++mfi) {
            auto const out = w[c]->array(mfi);
            auto const raw = rho.const_array(mfi);
            auto const rp = ped.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int const di = c == 0, dj = c == 2;
                    Real const r = .5 * (raw(i, j, k) + raw(i + di, j + dj, k));
                    Real const pedestal =
                        .5 * (rp(i, j, k) + rp(i + di, j + dj, k));
                    Real const radius = (i + (c == 0 ? .5 : 0.)) * dx[0];
                    // A frozen recovery mask with both exact replacement and
                    // blend.
                    Real const response =
                        j % 7 == 0 ? 0.0 : (j % 7 == 1 ? .25 : 1.0);
                    out(i, j, k) = PressureResponseWeight(r, pedestal, radius,
                                                          p, response);
                });
        }
        w[c]->OverrideSync(geom.periodicity());
    }
}

void
run (const std::string& name, int box, const std::string& boundary) {
    amrex::Box domain(amrex::IntVect(0),
                      amrex::IntVect(AMREX_D_DECL(15, 31, 0)));
    amrex::RealBox rb({AMREX_D_DECL(0., -1., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
    int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, boundary == "periodic", 0)};
    amrex::Geometry geom(domain, &rb, 1, periodic);
    amrex::BoxArray ba(domain);
    ba.maxSize(box);
    amrex::DistributionMapping dm(ba);
    ThermalMomentOptions mo;
    mo.number_density_floor = 4.e17;
    mo.active_density_floor = 0.;
    mo.boundary[0] = {MomentBoundary::Axis, boundary == "pmc"
                                                ? MomentBoundary::PMC
                                                : MomentBoundary::PEC};
    mo.boundary[1] = {boundary == "periodic" ? MomentBoundary::Periodic
                                             : MomentBoundary::PMC,
                      boundary == "periodic" ? MomentBoundary::Periodic
                                             : MomentBoundary::PEC};
    FrozenPressureFieldOptions options;
    options.gamma = mo.gamma;
    options.boundary = mo.boundary;
    options.transformed_ohm_form = name == "reject";
    FrozenPressureFieldCoupling coupling(geom, ba, dm, options);
    KineticThermalMoments moments(geom, ba, dm, mo);
    MF rho(amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm, 1, 1);
    MF ped(rho.boxArray(), dm, 1, 1), nc(ba, dm, 1, 1),
        nn(rho.boxArray(), dm, 1, 1);
    MF energy(ba, dm, 1, 1), direction(ba, dm, 1, 1), second(ba, dm, 1, 1),
        trial(ba, dm, 1, 1);
    rho.setVal(-9876.);
    ped.setVal(-9876.);
    direction.setVal(-9876.);
    auto const dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const a = rho.array(mfi);
        auto const p = ped.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const r = i * dx[0], z = j * dx[1];
                a(i, j, k) = PhysConst::q_e * 1.e18 *
                             (1. + .35 * std::cos(3.141592653589793 * r) *
                                       std::cos(3.141592653589793 * z));
                p(i, j, k) = PhysConst::q_e * 2.e17 *
                             (1. + .2 * std::cos(3.141592653589793 * z));
            });
    }
    exact(moments.Evaluate({rho, 0, &ped}), "moment density");
    MF::Copy(nc, moments.NumberDensity(), 0, 0, 1, 0);
    MF::Copy(nn, moments.NodalNumberDensity(), 0, 0, 1, 0);
    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
        auto const n = nc.const_array(mfi);
        auto const u = energy.array(mfi);
        auto const v = direction.array(mfi);
        auto const b = second.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const r = (i + .5) * dx[0], z = (j + .5) * dx[1];
                u(i, j, k) = PhysConst::kb * n(i, j, k) * 2.e4 / (5. / 3. - 1.);
                v(i, j, k) = .013 * (std::cos(2. * 3.141592653589793 * r) +
                                     .7 * std::sin(3.141592653589793 * z));
                b(i, j, k) = .009 * (std::sin(3.141592653589793 * r) +
                                     .4 * std::cos(2. * 3.141592653589793 * z));
            });
    }
    Fields w = fields(ba, dm), a = fields(ba, dm), b = fields(ba, dm),
           saved = fields(ba, dm), expected = fields(ba, dm);
    weights(w, rho, ped, geom);
    exact(coupling.Freeze(nc, nn, cview(w)), "initial freeze");
    coupling.Apply(view(a), direction);
    copy(saved, a);
    if (name == "transfer") {
        MF plus(rho.boxArray(), dm, 1, 1), minus(rho.boxArray(), dm, 1, 1),
            derivative(rho.boxArray(), dm, 1, 1);
        Real const eps = .125;
        MF::LinComb(trial, 1., energy, 0, eps, direction, 0, 0, 1, 0);
        exact(moments.Evaluate({rho, 0, &ped, &trial}), "plus pressure");
        MF::Copy(plus, moments.OhmPressure(), 0, 0, 1, 1);
        MF::LinComb(trial, 1., energy, 0, -eps, direction, 0, 0, 1, 0);
        exact(moments.Evaluate({rho, 0, &ped, &trial}), "minus pressure");
        MF::Copy(minus, moments.OhmPressure(), 0, 0, 1, 1);
        MF::LinComb(derivative, .5 / eps, plus, 0, -.5 / eps, minus, 0, 0, 1,
                    1);
        check(error(derivative, coupling.PressureIncrement(), 1., 1),
              derivative.norm0(0, 1), 2.e-12,
              "native_moment_directional_pressure");
        // Independent difference of actual native moment pressure values; no
        // shared pressure map or stencil helper in this expected expression.
        for (int c = 0; c < 3; ++c) {
            for (amrex::MFIter mfi(*expected[c]); mfi.isValid(); ++mfi) {
                auto const out = expected[c]->array(mfi);
                auto const p = derivative.const_array(mfi);
                auto const coeff = w[c]->const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        out(i, j, k) =
                            c == 1 ? 0.
                                   : coeff(i, j, k) *
                                         (p(i + (c == 0), j + (c == 2), k) -
                                          p(i, j, k)) /
                                         dx[c / 2];
                    });
            }
        }
        check(error(a, expected), norm(expected), 5.e-12,
              "physical_positive_field_response");
        exact(a[1]->norm0() == 0., "axisymmetric toroidal response zero");
    } else if (name == "linear") {
        coupling.Apply(view(b), second);
        MF::LinComb(trial, -1.3, direction, 0, .7, second, 0, 0, 1, 0);
        coupling.Apply(view(a), trial);
        for (int c = 0; c < 3; ++c) {
            MF::LinComb(*expected[c], -1.3, *saved[c], 0, .7, *b[c], 0, 0, 1,
                        0);
        }
        check(error(a, expected), norm(expected), 2.e-14, "signed_linearity");
        trial.setVal(0.);
        coupling.Apply(view(a), trial);
        exact(norm(a) == 0., "zero response");
        // Actual zero weights must be structural zeros, including negative
        // probes.
        for (int c = 0; c < 3; ++c) {
            w[c]->setVal(0.);
        }
        exact(coupling.Freeze(nc, nn, cview(w)), "masked freeze");
        coupling.Apply(view(a), direction);
        exact(norm(a) == 0., "recovered rows exact zero");
    } else if (name == "uniform") {
        nc.setVal(1.e18);
        nn.setVal(1.e18);
        direction.setVal(.137);
        exact(coupling.Freeze(nc, nn, cview(w)), "constant density freeze");
        coupling.Apply(view(a), direction);
        exact(norm(a) == 0., "constant U uniform n nullspace");
        // Constant U is not a null mode when native cell/node densities differ.
        MF::Copy(nc, moments.NumberDensity(), 0, 0, 1, 0);
        MF::Copy(nn, moments.NodalNumberDensity(), 0, 0, 1, 0);
        exact(coupling.Freeze(nc, nn, cview(w)), "nonuniform density freeze");
        coupling.Apply(view(a), direction);
        exact(norm(a) > 1.e-4, "variable n constant U response");
    } else if (name == "freeze") {
        nc.mult(7.);
        nn.mult(.25);
        for (int c = 0; c < 3; ++c) {
            w[c]->mult(3.);
        }
        coupling.Apply(view(a), direction);
        check(error(a, saved), norm(saved), 0.,
              "A_after_live_density_weight_mutation");
        exact(coupling.Freeze(nc, nn, cview(w)), "new B freeze");
        coupling.Apply(view(b), direction);
        check(error(b, saved, 3. / 28.), norm(b), 4.e-14,
              "new_frozen_density_response");
        nc.setVal(-1.);
        exact(!coupling.Freeze(nc, nn, cview(w)), "negative density rejects");
        coupling.Apply(view(a), direction);
        check(error(a, b), norm(b), 0., "failed_freeze_preserves_B");
        MF::Copy(nc, moments.NumberDensity(), 0, 0, 1, 0);
        MF::Copy(nn, moments.NodalNumberDensity(), 0, 0, 1, 0);
        weights(w, rho, ped, geom);
        w[0]->setVal(std::numeric_limits<Real>::quiet_NaN());
        exact(!coupling.Freeze(nc, nn, cview(w)), "nonfinite weight rejects");
        coupling.Apply(view(a), direction);
        check(error(a, b), norm(b), 0., "failed_weight_preserves_B");
        weights(w, rho, ped, geom);
        exact(coupling.Freeze(nc, nn, cview(w)), "restore A freeze");
        coupling.Apply(view(a), direction);
        check(error(a, saved), norm(saved), 0., "A_B_A_freeze");
        exact(coupling.FreezeCount() == 3, "successful freeze count");
    } else if (name == "weight") {
        PressureResponseParameters p;
        p.charge_floor = 2.;
        exact(PressureResponseWeight(1., 0., 0., p) == .5, "hard floor");
        exact(PressureResponseWeight(1., 3., 0., p) == .25,
              "pedestal only divisor");
        p.holmstrom = true;
        exact(PressureResponseWeight(1., 3., 0., p) == 0.,
              "raw gate ignores pedestal");
        p.axis_radius = 1.;
        exact(PressureResponseWeight(1., 3., 2., p) == .25,
              "hard gate axis confined");
        p.holmstrom_width = .5;
        p.axis_rolloff = .2;
        p.floor_width = .3;
        Real const div = .5 * (4. + 2. + std::sqrt(4. + .09));
        Real const g = .5 * (1. + std::tanh(-2.));
        check(std::abs(PressureResponseWeight(1., 3., 1., p, .4) -
                       (.5 + .5 * g) * .4 / div),
              1., 2.e-16, "smooth gate floor and row blend");
        p.conductor_raw_gate = true;
        exact(PressureResponseWeight(0., 3., 0., p) == 0.,
              "conductor raw gate");
        p.pressure_enabled = false;
        exact(PressureResponseWeight(4., 3., 0., p) == 0., "pressure disabled");
    } else {
        amrex::Abort("unknown pressure fixture case");
    }
    amrex::Print() << "PASS pressure " << name << " boundary=" << boundary
                   << " boxes=" << ba.size()
                   << " freeze=" << coupling.FreezeCount()
                   << " apply=" << coupling.ApplyCount() << "\n";
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        std::string name = "transfer", boundary = "periodic";
        int box = 8;
        pp.query("case", name);
        pp.query("boundary", boundary);
        pp.query("box", box);
        run(name, box, boundary);
    }
    amrex::Finalize();
}
