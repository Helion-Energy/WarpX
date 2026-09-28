/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "FieldSolver/ImplicitSolvers/EulerianThermalSources.H"
#include "Utils/WarpXConst.H"
#include "ablastr/coarsen/sample.H"
#ifdef TEST_THERMAL_MOMENTS
#include "KineticThermalMoments.H"
#endif

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
using SC = SourceComponent;
using IC = IonSourceComponent;
constexpr Real ne = 1.e19;
constexpr Real kelvin = PhysConst::q_e / PhysConst::kb;
void
near (Real x, Real y, Real relative, const char* reason) {
    if (!(std::isfinite(x) &&
          std::abs(x - y) <= relative * std::max(Real(1), std::abs(y)))) {
        amrex::Print() << std::setprecision(17) << reason << ": " << x << " vs "
                       << y << '\n';
        amrex::Abort(reason);
    }
}
void
require (bool ok, const char* reason) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok, reason);
}
Real
difference (const amrex::MultiFab& a, const amrex::MultiFab& b,
            int ghosts = 0) {
    amrex::MultiFab d(a.boxArray(), a.DistributionMap(), a.nComp(), ghosts);
    amrex::MultiFab::LinComb(d, 1, a, 0, -1, b, 0, 0, a.nComp(), ghosts);
    Real error = 0;
    for (int n = 0; n < a.nComp(); ++n) {
        error = std::max(error, d.norm0(n, ghosts));
    }
    return error;
}
Real
integrate (const amrex::MultiFab& f, const amrex::Geometry& geometry,
           int component) {
    auto const owner = f.OwnerMask(geometry.periodicity());
    auto const volume = MakeQdsmcVolumeElement(geometry, f.ixType());
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<Real> data(ops);
    using Tuple = typename decltype(data)::Type;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto const a = f.const_array(mfi);
        auto const mask = owner->const_array(mfi);
        ops.eval(mfi.validbox(), data,
                 [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                     return {mask(i, j, k)
                                 ? volume(i, j, k) * a(i, j, k, component)
                                 : Real(0)};
                 });
    }
    Real value = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(value);
    return value;
}
struct Fixture {
    amrex::Geometry geometry;
    amrex::MultiFab rho, te, rho1, rho2, ti1, ti2, add, stop, strain, rates,
        ions;
    std::array<amrex::MultiFab, 3> J, B, EV, EH, JS;
    ThermalSourceState state;
    Fixture (int box, bool periodic = false) {
        amrex::Box domain(amrex::IntVect(0, 0), amrex::IntVect(7, 11));
        amrex::RealBox physical({0., 0.}, {1., 2.});
        int periods[2] = {0, int(periodic)};
        geometry = amrex::Geometry(domain, &physical, 1, periods);
        amrex::BoxArray cells(domain);
        cells.maxSize(box);
        amrex::DistributionMapping dm(cells);
        auto nodes = amrex::convert(cells, amrex::IntVect::TheNodeVector());
        for (auto* f :
             {&rho, &te, &rho1, &rho2, &ti1, &ti2, &add, &stop, &strain}) {
            f->define(nodes, dm, 1, 1);
        }
        rates.define(nodes, dm, SC::Count, 1);
        ions.define(nodes, dm, 2 * IC::Count, 1);
        rho.setVal(ne * PhysConst::q_e);
        te.setVal(100 * kelvin);
        // Deliberately RAW RZ species density weights, not physical n_s.
        rho1.setVal(2.1);
        rho2.setVal(4.9);
        ti1.setVal(20);
        ti2.setVal(150);
        add.setVal(0);
        stop.setVal(0);
        strain.setVal(0);
        for (int c = 0; c < 3; ++c) {
            amrex::IntVect type(1);
            if (c == 0) {
                type[0] = 0;
            }
            if (c == 2) {
                type[1] = 0;
            }
            auto ba = amrex::convert(cells, type);
            for (auto* f : {&J[c], &EV[c], &EH[c], &JS[c]}) {
                f->define(ba, dm, 1, 1);
                f->setVal(0);
            }
            // Test gather also with complementary B staggering.
            amrex::IntVect bt(0);
            if (c == 0) {
                bt[0] = 1;
            }
            if (c == 2) {
                bt[1] = 1;
            }
            B[c].define(amrex::convert(cells, bt), dm, 1, 1);
            B[c].setVal(c == 2 ? 2 : 0);
            J[c].setVal(c == 0 ? 3.e4 : (c == 1 ? 4.e4 : 0));
            JS[c].setVal(7);
            state.plasma_current[c] = &J[c];
            state.magnetic_field[c] = &B[c];
        }
        state.raw_charge = &rho;
        state.temperature_kelvin = &te;
        state.additive_resistivity = &add;
        ThermalSourceSpecies a, b;
        a.charge_number = 1;
        a.raw_charge = &rho1;
        a.ion_temperature_ev = &ti1;
        b.charge_number = 2;
        b.raw_charge = &rho2;
        b.ion_temperature_ev = &ti2;
        for (int c = 0; c < 3; ++c) {
            a.raw_current[c] = &JS[c];
            b.raw_current[c] = &JS[c];
        }
        state.species = {a, b};
    }
    void
    uniform (const amrex::MultiFab& x, int c, Real expected, const char* reason,
             Real tolerance = 2.e-13) {
        near(x.min(c), expected, tolerance, reason);
        near(x.max(c), expected, tolerance, reason);
    }
    ThermalSourceOptions
    options () const {
        ThermalSourceOptions o;
        o.density_floor = 1.e17;
        o.physical_dt = 2.e-6;
        return o;
    }
};
amrex::Parser
parser (std::string const& expression, amrex::Vector<std::string> const& vars) {
    amrex::Parser p(expression);
    p.registerVariables(vars);
    return p;
}
void
rates_test (int box) {
    Fixture f(box);
    auto o = f.options();
    o.joule = true;
    o.relaxation = true;
    o.external_sink = true;
    auto eta = parser("2.e-6", {"rho", "J", "t"});
    o.field_eta = eta.compile<3>();
    auto overlay =
        parser("1.e-6 + Te*1.e-12", {"rs", "rho", "Te", "J", "Js", "B", "t"});
    f.state.species[1].has_resistivity_overlay = true;
    f.state.species[1].resistivity_overlay = overlay.compile<7>();
    auto nu = parser("3+Te/100", {"rho", "Te", "Ti", "t"});
    o.relaxation_rate = nu.compile<4>();
    auto sink = parser("10+Te+2*B", {"rho", "Te", "B", "t"});
    o.sink = sink.compile<4>();
    f.add.setVal(0.5e-6);
    EulerianThermalSources source(f.geometry, f.rho, o);
    require(source.Evaluate(f.state, f.rates, f.ions), "valid rates");
    Real const joule = 2.5e9 * (2.5e-6 + 0.7 * (1.e-6 + 100 * kelvin * 1.e-12));
    Real const ion1 = 3 * (0.3 * ne) * PhysConst::q_e * 4 * 80;
    Real const ion2 = 3 * (0.7 * ne / 2) * PhysConst::q_e * 4 * (-50);
    f.uniform(f.rates, SC::Joule, joule,
              "species Joule with raw charge fractions and overlay");
    f.uniform(f.ions, IC::RelaxationPower, ion1, "Z1 exchange");
    f.uniform(f.ions, IC::Count + IC::RelaxationPower, ion2, "Z2 exchange");
    f.uniform(f.rates, SC::Relaxation, -ion1 - ion2,
              "opposite electron exchange");
    f.uniform(f.rates, SC::ExternalSink, -114, "sink eV and B convention");
    f.uniform(f.rates, SC::ElectronTotal, joule - ion1 - ion2 - 114,
              "total physical source");
    auto ledger = source.Ledger(f.rates, f.ions, 2);
    Real const volume = 2 * MathConst::pi;
    near(ledger.energy[SC::ElectronTotal],
         o.physical_dt * volume * (joule - ion1 - ion2 - 114), 2.e-13,
         "physical clipped volume ledger");
    near(ledger.energy[SC::Relaxation] + ledger.ion_relaxation[0] +
             ledger.ion_relaxation[1],
         0, 2.e-13, "paired source exchange conservation");
    // Exclusion keeps the denominator, without reassigning its 70% charge.
    f.state.species[1].relaxation_excluded = true;
    require(source.Evaluate(f.state, f.rates, f.ions), "excluded species");
    f.uniform(f.rates, SC::Relaxation, -ion1,
              "excluded share not redistributed");
    f.uniform(f.ions, IC::Count + IC::RelaxationPower, 0,
              "excluded ion unchanged");
}
void
work_test (int box, bool periodic) {
    Fixture f(box, periodic);
    auto o = f.options();
    o.viscosity = ViscousSourceKind::AppliedWork;
    o.hyperresistive_work = true;
    for (int c = 0; c < 3; ++c) {
        for (amrex::MFIter mfi(f.J[c]); mfi.isValid(); ++mfi) {
            auto j = f.J[c].array(mfi), ev = f.EV[c].array(mfi),
                 eh = f.EH[c].array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int z, int k) {
                    Real const wave = std::cos(2 * MathConst::pi * z / 12);
                    j(i, z, k) = (2 + c) * (1 + 0.05 * i + 0.3 * wave);
                    ev(i, z, k) = (-0.2 + 0.04 * i + 0.7 * wave) * (1 + c);
                    eh(i, z, k) = (0.1 + 0.08 * i - 0.4 * wave) * (1 + c);
                });
        }
        f.J[c].FillBoundary(f.geometry.periodicity());
        f.EV[c].FillBoundary(f.geometry.periodicity());
        f.EH[c].FillBoundary(f.geometry.periodicity());
        f.state.viscous_electric[c] = &f.EV[c];
        f.state.hyper_electric[c] = &f.EH[c];
    }
    f.strain.setVal(12345);
    f.state.viscous_strain_power = &f.strain;
    EulerianThermalSources source(f.geometry, f.rho, o);
    require(source.Evaluate(f.state, f.rates, f.ions), "signed work valid");
    auto const ledger = source.Ledger(f.rates, f.ions, 2);
    Real visc = 0, hyper = 0;
    for (int c = 0; c < 3; ++c) {
        amrex::MultiFab product(f.J[c].boxArray(), f.J[c].DistributionMap(), 1,
                                0);
        for (auto const kind : {0, 1}) {
            for (amrex::MFIter mfi(product); mfi.isValid(); ++mfi) {
                auto p = product.array(mfi);
                auto j = f.J[c].const_array(mfi),
                     e = (kind == 0 ? f.EV[c] : f.EH[c]).const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int z, int k) {
                                       p(i, z, k) = j(i, z, k) * e(i, z, k);
                                   });
            }
            (kind == 0 ? visc : hyper) += integrate(product, f.geometry, 0);
        }
    }
    near(ledger.energy[SC::Viscous], o.physical_dt * visc, 2.e-13,
         "actual edge J.Evisc work");
    near(ledger.energy[SC::HyperResistive], o.physical_dt * hyper, 2.e-13,
         "actual edge J.Ehyper work");
    near(ledger.energy[SC::ElectronTotal], o.physical_dt * (visc + hyper),
         2.e-13, "work sources once; no strain double count");
    require(f.rates.min(SC::Viscous) < 0,
            "signed cooling retained without clamp");
    require(std::abs(ledger.energy[SC::ViscousStrain] -
                     ledger.energy[SC::Viscous]) > 1.e-3,
            "strain and matched work remain distinct");
#ifdef TEST_THERMAL_MOMENTS
    // Compose the independently delivered production scalar map, preserving
    // source units. This tests the complete edge -> node -> cell work chain.
    auto cells =
        amrex::convert(f.rho.boxArray(), amrex::IntVect::TheCellVector());
    ThermalMomentOptions mo;
    mo.verboncoeur_axis_correction = false;
    mo.boundary[0] = {MomentBoundary::Axis, MomentBoundary::PEC};
    mo.boundary[1] =
        periodic
            ? std::array{MomentBoundary::Periodic, MomentBoundary::Periodic}
            : std::array{MomentBoundary::PMC, MomentBoundary::PMC};
    KineticThermalMoments moments(f.geometry, cells, f.rho.DistributionMap(),
                                  mo);
    amrex::MultiFab cell_rate(cells, f.rho.DistributionMap(), 1, 1);
    Real map_error = 0;
    for (int component = 0; component < SC::Count; ++component) {
        moments.RestrictNodalScalar(f.rates, component, cell_rate);
        Real const cell_energy =
            o.physical_dt * integrate(cell_rate, f.geometry, 0);
        near(cell_energy, ledger.energy[component], 2.e-13,
             "edge/node/cell physical source work conservation");
        map_error = std::max(map_error,
                             std::abs(cell_energy - ledger.energy[component]));
    }
    amrex::Print() << "source cell-map absolute energy defect=" << map_error
                   << '\n';
#endif
    // Deliberately perform the WRONG product-after-gather operation; the
    // fixture must distinguish it from the actual applied work partition.
    amrex::MultiFab wrong(f.rho.boxArray(), f.rho.DistributionMap(), 1, 0);
    wrong.setVal(0);
    for (int c = 0; c < 3; ++c) {
        amrex::GpuArray<int, 3> st{1, 1, 1}, node{1, 1, 1}, coarsen{1, 1, 1};
        for (int d = 0; d < 2; ++d) {
            st[d] = f.J[c].ixType().nodeCentered(d);
        }
        for (amrex::MFIter mfi(wrong); mfi.isValid(); ++mfi) {
            auto a = wrong.array(mfi);
            auto j = f.J[c].const_array(mfi);
            auto e = f.EV[c].const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int z, int k) {
                    a(i, z, k) += ablastr::coarsen::sample::Interp(
                                      j, st, node, coarsen, i, z, k, 0) *
                                  ablastr::coarsen::sample::Interp(
                                      e, st, node, coarsen, i, z, k, 0);
                });
        }
    }
    require(
        std::abs(integrate(wrong, f.geometry, 0) - visc) > 1.e-3,
        "fixture distinguishes interpolation-before-product from matched work");
    // Masked applied work remains visible as a signed declined channel.
    for (amrex::MFIter mfi(f.rho); mfi.isValid(); ++mfi) {
        auto a = f.rho.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               if (i < 3) {
                                   a(i, j, k) = 0;
                               }
                           });
    }
    require(source.Evaluate(f.state, f.rates, f.ions), "masked work source");
    auto const masked = source.Ledger(f.rates, f.ions, 2);
    near(masked.energy[SC::ViscousWork] +
             masked.energy[SC::DeclinedViscousWork],
         o.physical_dt * visc, 2.e-13,
         "accepted plus declined viscous applied work");
    near(masked.energy[SC::HyperResistive] +
             masked.energy[SC::DeclinedHyperResistive],
         o.physical_dt * hyper, 2.e-13,
         "accepted plus declined hyper applied work");
}
void
derivative_test (int box) {
    Fixture f(box);
    auto o = f.options();
    o.joule = true;
    o.relaxation = true;
    o.external_sink = true;
    o.field_eta_uses_kelvin = true;
    auto eta = parser("1.e-6*(Te/K)^(-1.5)", {"rho", "J", "Te", "t"});
    eta.setConstant("K", kelvin);
    o.field_eta_kelvin = eta.compile<4>();
    auto nu = parser("0.3+0.01*Te^2", {"rho", "Te", "Ti", "t"});
    o.relaxation_rate = nu.compile<4>();
    auto sink = parser("0.7*Te^2", {"rho", "Te", "B", "t"});
    o.sink = sink.compile<4>();
    EulerianThermalSources source(f.geometry, f.rho, o);
    Real const T = 100, direction = 13;
    Real const ns1 = 0.3 * ne, ns2 = 0.7 * ne / 2;
    Real const derivative =
        -1.5 * 2500 * std::pow(T, -2.5) - 1.4 * T -
        3 * PhysConst::q_e *
            (ns1 * (0.02 * T * (T - 20) + (0.3 + 0.01 * T * T)) +
             ns2 * (0.02 * T * (T - 150) + (0.3 + 0.01 * T * T)));
    Real previous = std::numeric_limits<Real>::infinity(), last = 0;
    for (Real const eps : {1.e-1, 1.e-2, 1.e-3, 1.e-4}) {
        f.te.setVal((T + eps * direction) * kelvin);
        require(source.Evaluate(f.state, f.rates, f.ions), "plus direction");
        Real plus = f.rates.max(SC::ElectronTotal);
        f.te.setVal((T - eps * direction) * kelvin);
        require(source.Evaluate(f.state, f.rates, f.ions), "minus direction");
        Real minus = f.rates.max(SC::ElectronTotal);
        Real fd = (plus - minus) / (2 * eps), exact = direction * derivative;
        last = std::abs(fd - exact) / std::abs(exact);
        require(last < previous * 0.02,
                "centered live thermal derivative convergence");
        previous = last;
    }
    require(last < 1.e-8, "smooth TT directional derivative gate");
    // Heating eta is eV, independently of the Kelvin field-eta contract.
    o.use_heating_eta = true;
    auto heating = parser("1.e-6*Te", {"rho", "J", "Te", "t"});
    o.heating_eta = heating.compile<4>();
    EulerianThermalSources ev_source(f.geometry, f.rho, o);
    f.te.setVal(T * kelvin);
    require(ev_source.Evaluate(f.state, f.rates, f.ions),
            "live separate heating eta");
    f.uniform(f.rates, SC::Joule, 2500 * T, "heating eta receives eV");
    amrex::Print() << "directional relative error=" << last << '\n';
}
void
gates_test (int box) {
    Fixture f(box);
    auto o = f.options();
    o.joule = true;
    o.joule_density_gate = 0.75 * ne;
    o.joule_taper = true;
    auto eta = parser("2.e-6", {"rho", "J", "t"});
    o.field_eta = eta.compile<3>();
    EulerianThermalSources source(f.geometry, f.rho, o);
    require(source.Evaluate(f.state, f.rates, f.ions), "taper");
    Real const w = (1.0 / 3) * (1.0 / 3) * (3 - 2.0 / 3);
    f.uniform(f.rates, SC::Joule, 5000 * w, "same C1 halo valve");
    f.uniform(f.rates, SC::DeclinedJouleGate, 5000 * (1 - w),
              "withheld power exposed");
    o.redirect_joule = true;
    o.redirect_temperature_ev = 50;
    EulerianThermalSources redir(f.geometry, f.rho, o);
    require(redir.Evaluate(f.state, f.rates, f.ions), "redirect");
    f.uniform(f.rates, SC::Joule, 0, "redirect removes electron delivery");
    f.uniform(f.ions, IC::RedirectPower, 1500 * w, "Z1 redirect");
    f.uniform(f.ions, IC::Count + IC::RedirectPower, 3500 * w, "Z2 redirect");
    auto l = redir.Ledger(f.rates, f.ions, 2);
    near(l.energy[SC::DeclinedJouleGate] + l.ion_redirect[0] +
             l.ion_redirect[1],
         o.physical_dt * 2 * MathConst::pi * 5000, 2.e-13,
         "redirect plus withheld conservation");
    f.state.species[1].relaxation_excluded = true;
    require(redir.Evaluate(f.state, f.rates, f.ions), "excluded redirect");
    f.uniform(f.rates, SC::DeclinedRedirect, 3500 * w,
              "excluded redirect explicitly declined");
    f.rho.setVal(0.5 * ne * PhysConst::q_e);
    require(source.Evaluate(f.state, f.rates, f.ions), "gate");
    f.uniform(f.rates, SC::Joule, 0, "hard heat gate");
    f.uniform(f.rates, SC::DeclinedJouleGate, 5000,
              "gate retains declined physical rate");
}
void
stopping_test (int box) {
    Fixture f(box);
    auto o = f.options();
    f.stop.setVal(-3);
    f.state.stopping_impulse = &f.stop;
    EulerianThermalSources source(f.geometry, f.rho, o);
    require(source.Evaluate(f.state, f.rates, f.ions),
            "signed stopping impulse");
    f.uniform(f.rates, SC::Stopping, -3 / o.physical_dt,
              "physical full dt impulse rate");
    auto l = source.Ledger(f.rates, f.ions, 2);
    near(l.energy[SC::Stopping], -6 * MathConst::pi, 2.e-13,
         "full accepted impulse exactly once");
    require(source.Evaluate(f.state, f.rates, f.ions),
            "repeat impulse is pure");
    f.uniform(f.stop, 0, -3, "input impulse not consumed");
    f.rho.setVal(o.density_floor * PhysConst::q_e);
    require(source.Evaluate(f.state, f.rates, f.ions), "below-floor impulse");
    f.uniform(f.rates, SC::ElectronTotal, 0, "no fluid below raw floor");
    f.uniform(f.rates, SC::DeclinedStopping, -3 / o.physical_dt,
              "signed declined stopping work");
}
void
purity_test (int box) {
    Fixture f(box);
    auto o = f.options();
    o.joule = true;
    auto eta = parser("1.e-6+Te*1.e-13", {"rho", "J", "Te", "t"});
    o.field_eta_uses_kelvin = true;
    o.field_eta_kelvin = eta.compile<4>();
    EulerianThermalSources source(f.geometry, f.rho, o);
    std::vector<const amrex::MultiFab*> fields{&f.rho, &f.te,  &f.rho1, &f.rho2,
                                               &f.ti1, &f.ti2, &f.add};
    for (int c = 0; c < 3; ++c) {
        fields.push_back(&f.J[c]);
        fields.push_back(&f.B[c]);
    }
    std::vector<std::unique_ptr<amrex::MultiFab>> copy;
    for (auto* v : fields) {
        auto b = std::make_unique<amrex::MultiFab>(
            v->boxArray(), v->DistributionMap(), v->nComp(), v->nGrowVect());
        amrex::MultiFab::Copy(*b, *v, 0, 0, v->nComp(), v->nGrowVect());
        copy.push_back(std::move(b));
    }
    require(source.Evaluate(f.state, f.rates, f.ions), "A");
    amrex::MultiFab baseline(f.rates.boxArray(), f.rates.DistributionMap(),
                             f.rates.nComp(), 1);
    amrex::MultiFab::Copy(baseline, f.rates, 0, 0, f.rates.nComp(), 1);
    f.te.mult(1.1, 0, 1, 1);
    require(source.Evaluate(f.state, f.rates, f.ions), "B");
    amrex::MultiFab::Copy(f.te, *copy[1], 0, 0, 1, 1);
    require(source.Evaluate(f.state, f.rates, f.ions), "A again");
    near(difference(f.rates, baseline, 1), 0, 0,
         "A/B/A bitwise deterministic including output ghosts");
    for (std::size_t i = 0; i < fields.size(); ++i) {
        near(difference(*fields[i], *copy[i], 1), 0, 0,
             "no accepted or context writes including ghosts");
    }
}
void
invalid_test (int box) {
    Fixture f(box);
    auto o = f.options();
    o.joule = true;
    auto eta = parser("-1.e-6", {"rho", "J", "t"});
    o.field_eta = eta.compile<3>();
    EulerianThermalSources bad_eta(f.geometry, f.rho, o);
    require(!bad_eta.Evaluate(f.state, f.rates, f.ions),
            "negative resistivity rejected");
    o.joule = false;
    o.relaxation = true;
    auto nu = parser("-1", {"rho", "Te", "Ti", "t"});
    o.relaxation_rate = nu.compile<4>();
    EulerianThermalSources bad_nu(f.geometry, f.rho, o);
    require(!bad_nu.Evaluate(f.state, f.rates, f.ions),
            "negative frequency rejected");
    o.relaxation = false;
    o.viscosity = ViscousSourceKind::Strain;
    f.state.viscous_strain_power = &f.strain;
    f.strain.setVal(-1);
    EulerianThermalSources bad_strain(f.geometry, f.rho, o);
    require(!bad_strain.Evaluate(f.state, f.rates, f.ions),
            "negative strain heating rejected");
    o.viscosity = ViscousSourceKind::None;
    f.state.viscous_strain_power = nullptr;
    EulerianThermalSources finite(f.geometry, f.rho, o);
    f.te.setVal(std::numeric_limits<Real>::quiet_NaN());
    require(!finite.Evaluate(f.state, f.rates, f.ions),
            "nonfinite active temperature rejected");
    f.te.setVal(10 * kelvin);
    f.state.stopping_impulse = &f.stop;
    f.stop.setVal(-1.e20);
    require(finite.Evaluate(f.state, f.rates, f.ions),
            "large finite cooling stays physical; stage admissibility decides");
    f.uniform(f.rates, SC::Stopping, -1.e20 / o.physical_dt,
              "no hidden floor repair");
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        std::string which = "rates", reject;
        int box = 4;
        pp.query("case", which);
        pp.query("max_grid_size", box);
        pp.query("reject", reject);
        if (!reject.empty()) {
            Fixture f(box);
            auto o = f.options();
            o.embedded_boundary = reject == "eb";
            o.physical_levels = reject == "amr" ? 2 : 1;
            o.temperature_shunt = reject == "shunt";
            o.pedestal_temperature_cap = reject == "pedestal";
            o.redirect_kick_cap = reject == "kickcap";
            o.joule = reject == "parser";
            EulerianThermalSources source(f.geometry, f.rho, o);
            if (reject == "alias") {
                amrex::MultiFab aliased(f.rates, amrex::make_alias, 0, 1);
                f.state.temperature_kelvin = &aliased;
                source.Evaluate(f.state, f.rates, f.ions);
            }
            amrex::Abort("missing source capability guard");
        }
        if (which == "rates") {
            rates_test(box);
        } else if (which == "work") {
            work_test(box, false);
            work_test(box, true);
        } else if (which == "derivative") {
            derivative_test(box);
        } else if (which == "gates") {
            gates_test(box);
        } else if (which == "stopping") {
            stopping_test(box);
        } else if (which == "purity") {
            purity_test(box);
        } else if (which == "invalid") {
            invalid_test(box);
        } else {
            amrex::Abort("unknown source test");
        }
        amrex::Print() << "PASS source case " << which << " box " << box
                       << '\n';
    }
    amrex::Finalize();
}
