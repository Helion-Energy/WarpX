/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "BoundaryConditions/WarpX_PEC.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "FieldSolver/ImplicitSolvers/EulerianDissipation.H"
#include "FieldSolver/ImplicitSolvers/EulerianThermalSources.H"
#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>

namespace {
using namespace warpx::thermal;
using amrex::Real;
using MF = amrex::MultiFab;
constexpr Real ne = 1.e19, K = PhysConst::q_e / PhysConst::kb;
void
require (bool p, const char* why) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p, why);
}
void
near (Real a, Real b, Real tol, const char* why) {
    if (!(std::isfinite(a) && std::isfinite(b) &&
          std::abs(a - b) <=
              tol * std::max({Real(1.e-20), std::abs(a), std::abs(b)}))) {
        amrex::Print() << std::setprecision(17) << why << ": " << a << " vs "
                       << b << '\n';
        amrex::Abort(why);
    }
}
MF
clone (const MF& x) {
    MF y(x.boxArray(), x.DistributionMap(), x.nComp(), x.nGrowVect());
    MF::Copy(y, x, 0, 0, x.nComp(), x.nGrowVect());
    return y;
}
Real
difference (const MF& x, const MF& y, int ng = 0) {
    MF d(x.boxArray(), x.DistributionMap(), x.nComp(), ng);
    MF::LinComb(d, 1, x, 0, -1, y, 0, 0, x.nComp(), ng);
    Real value = 0;
    for (int c = 0; c < x.nComp(); ++c) {
        value = std::max(value, d.norm0(c, ng));
    }
    return value;
}
Real
integral (const MF& f, const amrex::Geometry& g, int c = 0) {
    auto mask = f.OwnerMask(g.periodicity());
    auto volume = MakeQdsmcVolumeElement(g, f.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a = f.const_array(mfi);
        auto own = mask->const_array(mfi);
        ops.eval(mfi.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                     return {own(i, j, k) ? volume(i, j, k) * a(i, j, k, c)
                                          : Real(0)};
                 });
    }
    Real value = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(value);
    return value;
}
Real
work (const std::array<MF, 3>& J, DissipationVector E,
      const amrex::Geometry& g) {
    Real sum = 0;
    for (int c = 0; c < 3; ++c) {
        auto product = clone(J[c]);
        // Independent Contexts may choose different processor maps (notably
        // two single-box contexts on MPI2). Redistribute before pointwise
        // multiplication; MultiFab::Multiply requires identical local maps.
        MF aligned(J[c].boxArray(), J[c].DistributionMap(), 1, 0);
        aligned.ParallelCopy(*E[c], 0, 0, 1, 0, 0, g.periodicity());
        MF::Multiply(product, aligned, 0, 0, 1, 0);
        sum += integral(product, g);
    }
    return sum;
}
amrex::Parser
parser (std::string expr, amrex::Vector<std::string> vars) {
    amrex::Parser p(expr);
    p.registerVariables(vars);
    return p;
}
struct Context {
    amrex::Geometry g;
    amrex::BoxArray cells;
    amrex::DistributionMapping dm;
    MF rho, te;
    std::array<MF, 3> J, Ji, B;
    TrialDissipationState state;
    Context (int box, bool periodic) {
        amrex::Box dom(amrex::IntVect(0, 0), amrex::IntVect(15, 15));
        amrex::RealBox physical({0., 0.}, {1., 2.});
        int periods[2] = {0, int(periodic)};
        g = amrex::Geometry(dom, &physical, 1, periods);
        cells = amrex::BoxArray(dom);
        cells.maxSize(box);
        dm = amrex::DistributionMapping(cells);
        auto nodes = amrex::convert(cells, amrex::IntVect(1));
        rho.define(nodes, dm, 1, 1);
        te.define(nodes, dm, 1, 1);
        state.charge = &rho;
        state.temperature_kelvin = &te;
        auto dx = g.CellSizeArray();
        for (auto* f : {&rho, &te}) {
            bool const temp = f == &te;
            for (amrex::MFIter mfi(*f); mfi.isValid(); ++mfi) {
                auto a = f->array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Real r = i * dx[0], z = j * dx[1];
                        a(i, j, k) =
                            temp ? 50 * K *
                                       (1 + .2 * std::cos(MathConst::pi * z) +
                                        .1 * r * r)
                                 : ne * PhysConst::q_e *
                                       (1 + .2 * r * r +
                                        .1 * std::cos(MathConst::pi * z));
                    });
            }
        }
        for (int c = 0; c < 3; ++c) {
            amrex::IntVect e(1), b(0);
            if (c == 0) {
                e[0] = 0;
                b[0] = 1;
            }
            if (c == 2) {
                e[1] = 0;
                b[1] = 1;
            }
            J[c].define(amrex::convert(cells, e), dm, 1, 1);
            Ji[c].define(J[c].boxArray(), dm, 1, 1);
            B[c].define(amrex::convert(cells, b), dm, 1, 1);
            for (amrex::MFIter mfi(J[c]); mfi.isValid(); ++mfi) {
                auto a = J[c].array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    Real const r = (i + (1 - e[0]) * .5) * dx[0],
                               z = (j + (1 - e[1]) * .5) * dx[1];
                    a(i, j, k) =
                        c == 0 ? 1.e4 * std::sin(MathConst::pi * r) *
                                     std::sin(MathConst::pi * z)
                        : c == 1
                            ? 1.e4 * r * (1 - r) * (1 - r) *
                                  (1 - std::cos(MathConst::pi * z))
                            : 1.e4 * (1 - r * r) * std::cos(MathConst::pi * z);
                });
            }
            Ji[c].setVal(0);
            B[c].setVal(c == 2 ? .2 : 0);
            state.plasma_current[c] = &J[c];
            state.ion_current[c] = &Ji[c];
            state.magnetic_field[c] = &B[c];
        }
    }
    TrialDissipationOptions
    options () const {
        TrialDissipationOptions o;
        o.boundary[0] = {DissipationBoundary::Axis, DissipationBoundary::PEC};
        o.boundary[1] = g.isPeriodic(1)
                            ? std::array{DissipationBoundary::Periodic,
                                         DissipationBoundary::Periodic}
                            : std::array{DissipationBoundary::PEC,
                                         DissipationBoundary::PEC};
        o.viscous.mc_limited = false;
        o.viscous.rho_floor = 1.e17 * PhysConst::q_e;
        o.viscous.flux_limit_f = 0;
        return o;
    }
};
void
viscosity (int box) {
    for (bool periodic : {false, true}) {
        for (Real fraction : {Real(0), Real(.3)}) {
            Context c(box, periodic);
            auto p = parser("2e4*(Te/50)*(1+B)", {"n", "Te", "B"});
            auto pp = parser("700*(Te/50)*(1+.01*n/1e19)", {"n", "Te", "B"});
            auto o = c.options();
            o.viscosity = true;
            o.viscous.model = 2;
            o.viscous.nu_par_pars = p.compile<3>();
            o.viscous.nu_perp_pars = pp.compile<3>();
            for (int d = 0; d < 3; ++d) {
                MF::Copy(c.Ji[d], c.J[d], 0, 0, 1, 1);
                c.Ji[d].mult(fraction, 1);
            }
            EulerianDissipation model(c.g, c.cells, c.dm, o);
            require(model.Evaluate(c.state), "viscosity state");
            Real const q = integral(model.StrainPower(), c.g),
                       power = work(c.J, model.ViscousField(), c.g);
            near((1 - fraction) * power, q, 3.e-13,
                 "exact variable-density anisotropic adjoint balance including "
                 "ion-strain contribution");
            require(q > 0, "nonzero strain work");
            // Compose the actual applied field through the previously delivered
            // physical source rate and dual-volume energy ledger.
            ThermalSourceOptions so;
            so.density_floor = 1.e17;
            so.physical_dt = 1;
            so.viscosity = ViscousSourceKind::AppliedWork;
            EulerianThermalSources source(c.g, c.rho, so);
            ThermalSourceState s;
            s.raw_charge = &c.rho;
            s.temperature_kelvin = &c.te;
            s.plasma_current = c.state.plasma_current;
            s.magnetic_field = c.state.magnetic_field;
            s.viscous_electric = model.ViscousField();
            s.viscous_strain_power = &model.StrainPower();
            MF rates(c.rho.boxArray(), c.dm, SourceComponent::Count, 1),
                ions(c.rho.boxArray(), c.dm, 1, 1);
            require(source.Evaluate(s, rates, ions), "source composition");
            auto ledger = source.Ledger(rates, ions, 0);
            near(ledger.energy[SourceComponent::ViscousWork], power, 3.e-13,
                 "actual EV work delivered to source ledger");
            near(ledger.energy[SourceComponent::ViscousStrain], q, 3.e-13,
                 "strain remains distinct diagnostic");
            amrex::Print() << std::setprecision(12)
                           << "ADJOINT periodic=" << periodic
                           << " ion_fraction=" << fraction << " Q=" << q
                           << " JEV=" << power
                           << " defect=" << (1 - fraction) * power - q << '\n';
        }
    }
}
void
braginskii (int box) {
    Context c(box, true);
    auto o = c.options();
    o.viscosity = true;
    for (int variant = 0; variant < 2; ++variant) {
        o.viscous.model = 1;
        o.viscous.flux_limit_f = variant == 0 ? .07 : 0;
        o.viscous.nu_max = variant == 0 ? 0 : 2.e4;
        o.viscous.taper_n = .6 * ne;
        EulerianDissipation m(c.g, c.cells, c.dm, o);
        require(m.Evaluate(c.state), "Braginskii live coefficients");
        Real const q = integral(m.StrainPower(), c.g);
        require(q > 0, "Braginskii capped/tapered strain is exercised");
        near(work(c.J, m.ViscousField(), c.g), q, 4.e-13,
             "Braginskii capped/tapered anisotropic adjoint");
    }
}
void
derivative (int box) {
    Context c(box, true);
    auto nu = parser("1000*Te", {"n", "Te", "B"});
    auto o = c.options();
    o.viscosity = true;
    o.viscous.model = 2;
    o.viscous.nu_par_pars = nu.compile<3>();
    o.viscous.nu_perp_pars = nu.compile<3>();
    EulerianDissipation m(c.g, c.cells, c.dm, o);
    require(m.Evaluate(c.state), "base derivative");
    auto base = clone(m.NodalDrag()), q = clone(m.StrainPower());
    auto original = clone(c.te);
    Real const eps = 2.e-5;
    c.te.mult(1 + eps, 1);
    require(m.Evaluate(c.state), "plus derivative");
    auto plus = clone(m.NodalDrag()), qplus = clone(m.StrainPower());
    MF::Copy(c.te, original, 0, 0, 1, 1);
    c.te.mult(1 - eps, 1);
    require(m.Evaluate(c.state), "minus derivative");
    MF::Subtract(plus, m.NodalDrag(), 0, 0, 3, 0);
    plus.mult(1 / (2 * eps));
    MF::Subtract(qplus, m.StrainPower(), 0, 0, 1, 0);
    qplus.mult(1 / (2 * eps));
    Real error = difference(plus, base) /
                 std::max({base.norm0(0), base.norm0(1), base.norm0(2)});
    require(error < 2.e-10, "live trial Te must differentiate viscosity force");
    require(difference(qplus, q) / q.norm0(0) < 2.e-10,
            "live trial Te must differentiate strain source");
    MF::Copy(c.te, original, 0, 0, 1, 1);
    require(m.Evaluate(c.state), "A/B/A derivative base");
    near(difference(m.NodalDrag(), base, 1), 0, 0,
         "A/B/A identical force including ghosts");
    near(difference(m.StrainPower(), q, 1), 0, 0,
         "A/B/A identical strain including ghosts");
    amrex::Print() << "DISSIPATION_DERIVATIVE relative=" << error << '\n';
}
void
hyper (int box) {
    for (bool periodic : {false, true}) {
        Context c(box, periodic);
        auto eta = parser("0.0003", {"rho", "B"});
        auto o = c.options();
        o.eta_h = eta.compile<2>();
        for (int d = 0; d < 3; ++d) {
            c.J[d].setVal(0);
        }
        auto dx = c.g.CellSizeArray();
        for (amrex::MFIter mfi(c.J[2]); mfi.isValid(); ++mfi) {
            auto a = c.J[2].array(mfi);
            amrex::ParallelFor(mfi.fabbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   a(i, j, k) = 1000 * i * i * dx[0] * dx[0];
                               });
        }
        for (HyperMode mode :
             {HyperMode::Laplacian, HyperMode::InteriorCurlCurl,
              HyperMode::AmpereCurlCurl}) {
            o.hyper = mode;
            EulerianDissipation m(c.g, c.cells, c.dm, o);
            require(m.Evaluate(c.state), "hyper state");
            auto const& raw = *m.RawHyperField()[2];
            Real err = 0;
            for (amrex::MFIter mfi(raw); mfi.isValid(); ++mfi) {
                amrex::Box b = mfi.validbox();
                b &= amrex::Box(amrex::IntVect(0, 1), amrex::IntVect(14, 14));
                if (!b.ok()) {
                    continue;
                }
                auto a = raw.const_array(mfi);
                amrex::ReduceOps<amrex::ReduceOpMax> ops;
                amrex::ReduceData<Real> data(ops);
                using T = typename decltype(data)::Type;
                ops.eval(b, data,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T {
                             return {std::abs(a(i, j, k) + 1.2)};
                         });
                err = std::max(err, amrex::get<0>(data.value()));
            }
            amrex::ParallelDescriptor::ReduceRealMax(err);
            require(err < 2.e-13,
                    "native radial quadratic and axis4 hyper stencil");
            // Distinguish the cropped interior and full Ampere wall closures.
            if (periodic) {
                for (amrex::MFIter mfi(raw); mfi.isValid(); ++mfi) {
                    auto b = mfi.validbox();
                    if (b.contains(amrex::IntVect(16, 5))) {
                        Real const wall = raw[mfi](amrex::IntVect(16, 5));
                        require(mode != HyperMode::InteriorCurlCurl ||
                                    wall == 0,
                                "interior curlcurl excludes r wall");
                        require(mode != HyperMode::AmpereCurlCurl || wall > 0,
                                "Ampere curlcurl keeps zero-extension wall "
                                "closure");
                        require(mode != HyperMode::Laplacian ||
                                    std::abs(wall + 1.2) < 2.e-13,
                                "Laplacian raw boundary value");
                        require((*m.HyperField()[2])[mfi](
                                    amrex::IntVect(16, 5)) == 0,
                                "applied PEC tangential field vanishes");
                    }
                }
            }
            // This polynomial has external-boundary work: a negative local
            // applied J.EH is retained, never clipped into positive heating.
            if (mode == HyperMode::Laplacian) {
                require(work(c.J, m.HyperField(), c.g) < 0,
                        "signed hyper work retained");
            }
        }
    }
}
void
curl_adjoint (int box) {
    Context a(box, true), b(box, true);
    auto eta = parser("0.001*(1+rho)*(1+B)", {"rho", "B"});
    auto o = a.options();
    o.eta_h = eta.compile<2>();
    o.eta_h_depends_on_B = true;
    // Compact support removes the deliberately cropped mode's boundary ring.
    for (int c = 0; c < 3; ++c) {
        for (auto* f : {&a.J[c], &b.J[c]}) {
            bool const second = f == &b.J[c];
            auto ix = f->ixType().toIntVect();
            for (amrex::MFIter mfi(*f); mfi.isValid(); ++mfi) {
                auto ar = f->array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                     int i, int j, int k) {
                    Real r = (i + (1 - ix[0]) * .5) / 16.,
                         z = (j + (1 - ix[1]) * .5) / 16.;
                    ar(i, j, k) =
                        i < 3 || i > 12
                            ? 0
                            : (second
                                   ? std::cos(2 * MathConst::pi * z + .3 * c) *
                                         r * r
                                   : std::sin(2 * MathConst::pi * z + .4 * c) *
                                         r);
                });
            }
        }
    }
    for (auto mode : {HyperMode::InteriorCurlCurl, HyperMode::AmpereCurlCurl}) {
        o.hyper = mode;
        EulerianDissipation ma(a.g, a.cells, a.dm, o),
            mb(b.g, b.cells, b.dm, o);
        require(ma.Evaluate(a.state) && mb.Evaluate(b.state),
                "curl adjoint states");
        near(work(a.J, mb.HyperField(), a.g), work(b.J, ma.HyperField(), a.g),
             2.e-12, "variable coefficient native weighted curl adjoint");
        require(work(a.J, ma.HyperField(), a.g) > 0,
                "positive curlcurl dissipation on closed free rows");
    }
}
void
purity (int box) {
    Context c(box, false);
    auto nu = parser("2000", {"n", "Te", "B"});
    auto eta = parser(".001*(1+B)", {"rho", "B"});
    auto o = c.options();
    o.viscosity = true;
    o.viscous.model = 2;
    o.viscous.nu_par_pars = nu.compile<3>();
    o.viscous.nu_perp_pars = nu.compile<3>();
    o.hyper = HyperMode::AmpereCurlCurl;
    o.eta_h = eta.compile<2>();
    o.eta_h_depends_on_B = true;
    auto rho = clone(c.rho), te = clone(c.te);
    std::array<MF, 3> j, ji, b;
    for (int d = 0; d < 3; ++d) {
        j[d] = clone(c.J[d]);
        ji[d] = clone(c.Ji[d]);
        b[d] = clone(c.B[d]);
    }
    EulerianDissipation m(c.g, c.cells, c.dm, o);
    require(m.Evaluate(c.state), "purity base");
    auto ev0 = clone(*m.ViscousField()[0]), eh0 = clone(*m.HyperField()[2]);
    c.te.mult(1.2, 1);
    c.B[2].mult(3, 1);
    require(m.Evaluate(c.state), "purity B");
    MF::Copy(c.te, te, 0, 0, 1, 1);
    MF::Copy(c.B[2], b[2], 0, 0, 1, 1);
    require(m.Evaluate(c.state), "purity A/B/A");
    near(difference(ev0, *m.ViscousField()[0], 1), 0, 0, "A/B/A applied EV");
    near(difference(eh0, *m.HyperField()[2], 1), 0, 0, "A/B/A applied EH");
    require(difference(c.rho, rho, 1) == 0 && difference(c.te, te, 1) == 0,
            "accepted scalar storage untouched");
    for (int d = 0; d < 3; ++d) {
        require(difference(c.J[d], j[d], 1) == 0 &&
                    difference(c.Ji[d], ji[d], 1) == 0 &&
                    difference(c.B[d], b[d], 1) == 0,
                "input Yee storage/ghosts untouched");
    }
    // Check homogeneous shell/PEC increment images: full affine shell value
    // cancels from full-minus-no-term, leaving negative interior increment.
    auto const field = m.HyperField();
    for (int d = 1; d < 3; ++d) {
        auto const& f = *field[d];
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto const& fab = f[mfi];
            if (!fab.box().contains(amrex::IntVect(17, 5))) {
                continue;
            }
            Real const wall = 7.2, base = 3.1,
                       increment = fab(amrex::IntVect(15, 5));
            Real const full = 2 * wall - (base + increment),
                       without = 2 * wall - base;
            near(fab(amrex::IntVect(17, 5)), full - without, 3.e-13,
                 "affine shell cancellation equals homogeneous PEC ghost");
        }
    }
}

// Use the actual production boundary routines as an independent reference.
// Neumann is an enum alias of PMC; absorbing particles still use its current
// images. Radial PEC + reflecting particles instead has PMC current parity
// and the production r*J metric image, distinct from the electric projection.
void
prepare_pmc_current (Context& c) {
    amrex::Array<FieldBoundaryType, 2> lo{FieldBoundaryType::None,
                                          FieldBoundaryType::Neumann};
    amrex::Array<FieldBoundaryType, 2> hi{FieldBoundaryType::PEC,
                                          FieldBoundaryType::Neumann};
    amrex::Array<ParticleBoundaryType, 2> plo{ParticleBoundaryType::Absorbing,
                                              ParticleBoundaryType::Absorbing};
    amrex::Array<ParticleBoundaryType, 2> phi{ParticleBoundaryType::Reflecting,
                                              ParticleBoundaryType::Absorbing};
    for (auto& f : c.J) {
        auto const dom = amrex::convert(c.g.Domain(), f.ixType());
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto a = f.array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (i > dom.bigEnd(0) || j < 0 || j > dom.bigEnd(1)) {
                        a(i, j, k) = 0;
                    }
                });
        }
    }
    PEC::ApplyReflectiveBoundarytoJfield(&c.J[0], &c.J[1], &c.J[2], lo, hi, plo,
                                         phi, c.g, 0, PatchType::fine, {});
    for (auto& f : c.J) {
        f.OverrideSync(c.g.periodicity());
        f.FillBoundary(c.g.periodicity());
    }
}
void
pmc_projection (int box) {
    for (bool mixed : {false, true}) {
        Context c(box, false);
        auto o = c.options();
        o.boundary[1] = {DissipationBoundary::PMC,
                         mixed ? DissipationBoundary::PEC
                               : DissipationBoundary::PMC};
        EulerianDissipation model(c.g, c.cells, c.dm, o);
        std::array<MF, 3> ref, projected;
        auto dx = c.g.CellSizeArray();
        for (int d = 0; d < 3; ++d) {
            ref[d] = clone(c.J[d]);
            auto ix = ref[d].ixType().toIntVect();
            for (amrex::MFIter mfi(ref[d]); mfi.isValid(); ++mfi) {
                auto a = ref[d].array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Real r = (i + .5 * (1 - ix[0])) * dx[0],
                             z = (j + .5 * (1 - ix[1])) * dx[1];
                        // Already regular at the axis; production PEC/PMC does
                        // not own the independent RZ-axis conditioning.
                        a(i, j, k) = (d == 2 ? Real(1) : r) * (1 + .2 * r * r) *
                                     (1 + d + .3 * z + .1 * z * z);
                    });
            }
            projected[d] = clone(ref[d]);
        }
        amrex::Array<FieldBoundaryType, 2> lo{FieldBoundaryType::None,
                                              FieldBoundaryType::PMC};
        amrex::Array<FieldBoundaryType, 2> hi{FieldBoundaryType::PEC,
                                              mixed ? FieldBoundaryType::PEC
                                                    : FieldBoundaryType::PMC};
        std::array<MF*, 3> reference{&ref[0], &ref[1], &ref[2]};
        PEC::ApplyPECtoEfield(reference, lo, hi, FieldBoundaryType::PEC,
                              amrex::IntVect(1), c.g, 0, PatchType::fine, {});
        PEC::ApplyPECtoBfield(reference, lo, hi, FieldBoundaryType::PMC,
                              amrex::IntVect(1), c.g, 0, PatchType::fine, {});
        model.ProjectElectricIncrement(
            {&projected[0], &projected[1], &projected[2]});
        for (int d = 0; d < 3; ++d) {
            ref[d].OverrideSync(c.g.periodicity());
            ref[d].FillBoundary(c.g.periodicity());
            require(difference(ref[d], projected[d], 1) < 3.e-15,
                    "PMC/PEC/corner increment equals production electric "
                    "boundary stack");
        }
        auto original = clone(projected[2]);
        model.ProjectElectricIncrement(
            {&projected[0], &projected[1], &projected[2]});
        require(difference(original, projected[2], 1) == 0,
                "PMC projection is idempotent");
    }
}
void
pmc_sources (int box) {
    Context c(box, false);
    prepare_pmc_current(c);
    auto rho = clone(c.rho), te = clone(c.te);
    std::array<MF, 3> saved;
    for (int d = 0; d < 3; ++d) {
        saved[d] = clone(c.J[d]);
    }
    auto nu = parser("1000*Te", {"n", "Te", "B"});
    auto eta = parser("0.001*(1+rho)*(1+B)", {"rho", "B"});
    auto o = c.options();
    o.boundary[1] = {DissipationBoundary::PMC, DissipationBoundary::PMC};
    o.viscosity = true;
    o.viscous.model = 2;
    o.viscous.nu_par_pars = nu.compile<3>();
    o.viscous.nu_perp_pars = nu.compile<3>();
    o.eta_h = eta.compile<2>();
    o.eta_h_depends_on_B = true;
    for (auto mode : {HyperMode::Laplacian, HyperMode::InteriorCurlCurl,
                      HyperMode::AmpereCurlCurl}) {
        o.hyper = mode;
        EulerianDissipation m(c.g, c.cells, c.dm, o);
        require(m.Evaluate(c.state), "PMC applied dissipation state");
        Real const viscous = work(c.J, m.ViscousField(), c.g),
                   hyper = work(c.J, m.HyperField(), c.g);
        near(viscous, integral(m.StrainPower(), c.g), 4.e-13,
             "PMC physical current and matched viscosity adjoint");
        ThermalSourceOptions so;
        so.density_floor = 1.e17;
        so.physical_dt = .3;
        so.viscosity = ViscousSourceKind::AppliedWork;
        so.hyperresistive_work = true;
        EulerianThermalSources source(c.g, c.rho, so);
        ThermalSourceState s;
        s.raw_charge = &c.rho;
        s.temperature_kelvin = &c.te;
        s.plasma_current = c.state.plasma_current;
        s.magnetic_field = c.state.magnetic_field;
        s.viscous_electric = m.ViscousField();
        s.hyper_electric = m.HyperField();
        s.viscous_strain_power = &m.StrainPower();
        MF rates(c.rho.boxArray(), c.dm, SourceComponent::Count, 1),
            ions(c.rho.boxArray(), c.dm, 1, 1);
        require(source.Evaluate(s, rates, ions),
                "PMC applied force/source composition");
        auto ledger = source.Ledger(rates, ions, 0);
        near(ledger.energy[SourceComponent::ViscousWork], .3 * viscous, 5.e-13,
             "PMC EV physical work reaches nodal source ledger");
        near(ledger.energy[SourceComponent::HyperResistive], .3 * hyper, 5.e-13,
             "PMC signed EH work reaches nodal source ledger");
        auto ev = clone(*m.ViscousField()[0]), eh = clone(*m.HyperField()[2]);
        c.te.mult(1.1, 1);
        require(m.Evaluate(c.state), "PMC perturbed temperature");
        require(difference(ev, *m.ViscousField()[0]) > 0,
                "PMC live T force response");
        MF::Copy(c.te, te, 0, 0, 1, 1);
        require(m.Evaluate(c.state), "PMC restored temperature");
        require(difference(ev, *m.ViscousField()[0], 1) == 0 &&
                    difference(eh, *m.HyperField()[2], 1) == 0,
                "PMC A/B/A applied fields including guards");
        amrex::Print() << std::setprecision(15)
                       << "PMC_SOURCE mode=" << int(mode) << " JEV=" << viscous
                       << " JEH=" << hyper << " viscous_defect="
                       << viscous - integral(m.StrainPower(), c.g) << '\n';
    }
    require(difference(rho, c.rho, 1) == 0 && difference(te, c.te, 1) == 0,
            "PMC scalar inputs untouched");
    for (int d = 0; d < 3; ++d) {
        require(difference(saved[d], c.J[d], 1) == 0,
                "PMC supplied current guards untouched");
    }
}
void
pmc_guards (int box) {
    Context c(box, false);
    prepare_pmc_current(c);
    for (int d = 0; d < 3; ++d) {
        auto const& f = c.J[d];
        int const last = 15 + f.ixType().nodeCentered(1);
        Real defect = 0;
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            for (int j : {-1, last + 1}) {
                int const mirror =
                    j < 0 ? (f.ixType().nodeCentered(1) ? 1 : 0)
                          : (f.ixType().nodeCentered(1) ? last - 1 : last);
                for (int i = 1; i < 16; ++i) {
                    amrex::IntVect dst(i, j), src(i, mirror);
                    if (mfi.fabbox().contains(dst) &&
                        mfi.fabbox().contains(src)) {
                        defect = std::max(
                            defect, std::abs(f[mfi](dst) -
                                             (d == 2 ? -1 : 1) * f[mfi](src)));
                    }
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(defect);
        require(defect == 0,
                "actual PMC absorbing-particle current image parity");
    }
    auto eta = parser("0.001", {"rho", "B"});
    auto o = c.options();
    o.boundary[1] = {DissipationBoundary::PMC, DissipationBoundary::PMC};
    o.hyper = HyperMode::Laplacian;
    o.eta_h = eta.compile<2>();
    EulerianDissipation m(c.g, c.cells, c.dm, o);
    require(m.Evaluate(c.state), "PMC ghost sensitivity base");
    auto base = clone(*m.RawHyperField()[2]);
    // Deliberately perturb ONLY a physical input ghost. The helper must use
    // the caller-supplied moment closure rather than silently replacing it
    // with an electric image. Expected native stencil response is exact.
    for (amrex::MFIter mfi(c.J[2]); mfi.isValid(); ++mfi) {
        auto a = c.J[2].array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               if (j == -1 && i >= 3 && i <= 12) {
                                   a(i, j, k) += 7.;
                               }
                           });
    }
    require(m.Evaluate(c.state), "PMC supplied ghost perturbation");
    Real error = 0;
    auto const& raw = *m.RawHyperField()[2];
    Real const expected = -.001 * 7 / (c.g.CellSize(1) * c.g.CellSize(1));
    for (amrex::MFIter mfi(raw); mfi.isValid(); ++mfi) {
        for (int i = 3; i <= 12; ++i) {
            amrex::IntVect iv(i, 0);
            if (mfi.validbox().contains(iv)) {
                error = std::max(
                    error, std::abs(raw[mfi](iv) - base[mfi](iv) - expected));
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(error);
    require(
        error < 2.e-11,
        "native hyper stencil consumes actual supplied physical current ghost");
    amrex::Print() << "PMC_GUARD_RESPONSE expected=" << expected
                   << " max_error=" << error << '\n';
}

void
invalid (int box) {
    Context c(box, true);
    auto nu = parser("-2", {"n", "Te", "B"});
    auto o = c.options();
    o.viscosity = true;
    o.viscous.model = 2;
    o.viscous.nu_par_pars = nu.compile<3>();
    o.viscous.nu_perp_pars = nu.compile<3>();
    EulerianDissipation m(c.g, c.cells, c.dm, o);
    require(m.Evaluate(c.state), "legacy negative nu is audited clamp");
    require(m.CoefficientClampMask().max(0) == 1 &&
                m.CoefficientClampMask().max(1) == 1,
            "both viscosity clamp channels visible");
    require(m.StrainPower().norm0(0) == 0,
            "clamped coefficients produce no heating");
    c.te.setVal(std::numeric_limits<Real>::quiet_NaN());
    require(!m.Evaluate(c.state),
            "nonfinite trial rejects without silent gating");
    c.te.setVal(50 * K);
    auto bad = parser("-1", {"rho", "B"});
    o.viscosity = false;
    o.hyper = HyperMode::AmpereCurlCurl;
    o.eta_h = bad.compile<2>();
    EulerianDissipation h(c.g, c.cells, c.dm, o);
    require(!h.Evaluate(c.state), "negative eta_H rejects even at zero curl");
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int box = 4;
        std::string which = "viscosity";
        pp.query("case", which);
        pp.query("max_grid_size", box);
        std::string reject;
        pp.query("reject", reject);
        if (!reject.empty()) {
            Context c(box, true);
            auto o = c.options();
            if (reject == "eb") {
                o.embedded_boundary = true;
            }
            if (reject == "amr") {
                o.physical_levels = 2;
            }
            if (reject == "mode") {
                o.azimuthal_modes = 2;
            }
            if (reject == "transform") {
                o.transformed_electric_solve = true;
            }
            if (reject == "boundary") {
                o.boundary[1][0] = DissipationBoundary::PEC;
            }
            if (reject == "limited") {
                o.viscosity = true;
                o.viscous.mc_limited = true;
            }
            if (reject == "parser") {
                o.hyper = HyperMode::Laplacian;
            }
            EulerianDissipation m(c.g, c.cells, c.dm, o);
            if (reject == "alias") {
                c.state.charge = &m.StrainPower();
            }
            m.Evaluate(c.state);
            amrex::Abort("REJECTION DID NOT OCCUR");
        }
        if (which == "braginskii") {
            braginskii(box);
        } else if (which == "viscosity") {
            viscosity(box);
        } else if (which == "derivative") {
            derivative(box);
        } else if (which == "hyper") {
            hyper(box);
        } else if (which == "curl_adjoint") {
            curl_adjoint(box);
        } else if (which == "purity") {
            purity(box);
        } else if (which == "pmc_projection") {
            pmc_projection(box);
        } else if (which == "pmc_sources") {
            pmc_sources(box);
        } else if (which == "pmc_guards") {
            pmc_guards(box);
        } else if (which == "invalid") {
            invalid(box);
        } else {
            amrex::Abort("Unknown dissipation fixture case");
        }
        amrex::Print() << "DISSIPATION_PASS case=" << which << " box=" << box
                       << '\n';
    }
    amrex::Finalize();
}
