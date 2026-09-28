/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/EulerianSourceDiagonal.H"
#include "KineticThermalMoments.H"
#include "Utils/WarpXConst.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <cmath>
#include <limits>
using namespace warpx::thermal;
using amrex::Real;
using MF = amrex::MultiFab;
constexpr Real kelvin = PhysConst::q_e / PhysConst::kb;
void
require (bool ok, const char* why) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok, why);
}
void
near (Real a, Real b, Real tolerance, const char* why) {
    if (!(std::abs(a - b) <=
          tolerance * std::max({Real(1e-100), std::abs(a), std::abs(b)}))) {
        amrex::Print() << why << " " << a << " " << b << "\n";
        amrex::Abort(why);
    }
}
Real
difference (const MF& a, const MF& b) {
    MF diff(a.boxArray(), a.DistributionMap(), a.nComp(), 0);
    MF::LinComb(diff, 1, a, 0, -1, b, 0, 0, a.nComp(), 0);
    Real error = 0;
    for (int n = 0; n < a.nComp(); ++n)
        error = std::max(error, diff.norminf(n));
    return error;
}
MF
clone (const MF& a) {
    MF out(a.boxArray(), a.DistributionMap(), a.nComp(), 0);
    MF::Copy(out, a, 0, 0, a.nComp(), 0);
    return out;
}
Real
value (const MF& a, int i, int j, int component = 0) {
    Real out = -std::numeric_limits<Real>::max();
    for (amrex::MFIter mfi(a); mfi.isValid(); ++mfi)
        if (mfi.validbox().contains(amrex::IntVect(i, j)))
            out = std::max(out, a[mfi](amrex::IntVect(i, j), component));
    amrex::ParallelDescriptor::ReduceRealMax(out);
    return out;
}
struct Fixture {
    amrex::Geometry g;
    amrex::BoxArray cells, nodes;
    amrex::DistributionMapping dm;
    MF rho, pedestal, energy, old, te, capacity, rates, ions, qcell;
    std::array<MF, 3> charge, ti, current, magnetic;
    ThermalSourceState state;
    ThermalSourceOptions options;
    Real gamma = 1.6;
    amrex::Parser eta{"1.e-7*(Te/3)^(-1.2)"},
        eta_kelvin{"1.e-7*(Te/34813.554364650) ^ (-1.2)"},
        nu{"1.e4*(Te/3)^(-1.5)"}, grow{"1.e-7*Te^2"};
    std::unique_ptr<KineticThermalMoments> map;
    Fixture (int box, bool periodic) {
        amrex::Box domain(amrex::IntVect(0, 0), amrex::IntVect(15, 15));
        amrex::RealBox physical({0., 0.}, {1.6, 1.6});
        int periods[2] = {0, int(periodic)};
        g.define(domain, &physical, 1, periods);
        cells.define(domain);
        cells.maxSize(box);
        nodes = amrex::convert(cells, amrex::IntVect(1));
        dm = amrex::DistributionMapping(cells);
        for (auto* mf : {&rho, &pedestal, &te})
            mf->define(nodes, dm, 1, 1);
        for (auto* mf : {&energy, &old, &capacity, &qcell})
            mf->define(cells, dm, 1, 0);
        rates.define(nodes, dm, SourceComponent::Count, 0);
        ions.define(nodes, dm, 3 * IonSourceComponent::Count, 0);
        rho.setVal(3e19 * PhysConst::q_e);
        pedestal.setVal(1e19 * PhysConst::q_e);
        eta.registerVariables({"rho", "J", "Te", "t"});
        eta_kelvin.registerVariables({"rho", "J", "Te", "t"});
        nu.registerVariables({"rho", "Te", "Ti", "t"});
        grow.registerVariables({"rho", "J", "Te", "t"});
        options.joule = true;
        options.relaxation = true;
        options.use_heating_eta = true;
        options.heating_eta = eta.compile<4>();
        options.relaxation_rate = nu.compile<4>();
        options.density_floor = 1e16;
        state.raw_charge = &rho;
        state.temperature_kelvin = &te;
        for (int s = 0; s < 3; ++s) {
            charge[s].define(nodes, dm, 1, 0);
            charge[s].setVal(s + 1);
            ti[s].define(nodes, dm, 1, 0);
            ti[s].setVal(1);
            ThermalSourceSpecies species;
            species.charge_number = s == 2 ? 2 : 1;
            species.relaxation_excluded = s == 2;
            species.raw_charge = &charge[s];
            species.ion_temperature_ev = &ti[s];
            state.species.push_back(species);
            amrex::IntVect jtype(1);
            if (s == 0)
                jtype[0] = 0;
            if (s == 2)
                jtype[1] = 0;
            current[s].define(amrex::convert(cells, jtype), dm, 1, 1);
            current[s].setVal((s + 1) * 2e4);
            magnetic[s].define(amrex::convert(cells, amrex::IntVect(1) - jtype),
                               dm, 1, 1);
            magnetic[s].setVal(s == 2 ? 1. : 0.);
            state.plasma_current[s] = &current[s];
            state.magnetic_field[s] = &magnetic[s];
        }
        ThermalMomentOptions m;
        m.number_density_floor = options.density_floor;
        m.active_density_floor = options.density_floor;
        m.gamma = gamma;
        m.verboncoeur_axis_correction = true;
        m.boundary[0] = {MomentBoundary::Axis, MomentBoundary::PEC};
        m.boundary[1] =
            periodic
                ? std::array{MomentBoundary::Periodic, MomentBoundary::Periodic}
                : std::array{MomentBoundary::PMC, MomentBoundary::PMC};
        map = std::make_unique<KineticThermalMoments>(g, cells, dm, m);
        KineticThermalStateView input{rho};
        input.pedestal = &pedestal;
        require(map->Evaluate(input), "density map failed");
        MF::Copy(capacity, map->NumberDensity(), 0, 0, 1, 0);
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto n = capacity.const_array(mfi);
            auto u = energy.array(mfi);
            Real factor = PhysConst::q_e / (gamma - 1);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    u(i, j, k) = factor * n(i, j, k) * (2 + .04 * i + .02 * j);
                });
        }
        MF::Copy(old, energy, 0, 0, 1, 0);
        update();
    }
    void
    update () {
        KineticThermalStateView input{rho};
        input.pedestal = &pedestal;
        input.energy = &energy;
        require(map->Evaluate(input), "temperature map failed");
        MF::Copy(te, map->NodalTemperature(), 0, 0, 1, 1);
    }
    void
    constant_temperature (Real ev) {
        for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi) {
            auto n = capacity.const_array(mfi);
            auto u = energy.array(mfi);
            Real f = ev * PhysConst::q_e / (gamma - 1);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   u(i, j, k) = f * n(i, j, k);
                               });
        }
        update();
    }
    std::unique_ptr<EulerianSourceDiagonal>
    diagonal (Real step = 2e-5) {
        SourceDiagonalOptions d;
        d.relative_temperature_step = step;
        return std::make_unique<EulerianSourceDiagonal>(g, rho, options, d);
    }
    void
    physical () {
        update();
        EulerianThermalSources source(g, rho, options);
        require(source.Evaluate(state, rates, ions), "physical source failed");
        map->RestrictNodalScalar(rates, SourceComponent::ElectronTotal, qcell);
    }
    void
    analytic () {
        auto d = diagonal();
        require(d->Freeze(state, capacity, gamma), "analytic freeze failed");
        EulerianThermalSources src(g, rho, options);
        require(src.Evaluate(state, rates, ions), "reference source failed");
        Real error = 0;
        for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
            auto t = te.const_array(mfi),
                 derivative = d->NodalDerivative().const_array(mfi),
                 power = rates.const_array(mfi);
            for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it) {
                auto p = it();
                Real T = t(p), ev = T / kelvin, n = 3e19, sum = 0;
                for (int species = 0; species < 2; ++species) {
                    Real raw = charge[species][mfi](p),
                         denom =
                             std::max(charge[0][mfi](p) + charge[1][mfi](p) +
                                          charge[2][mfi](p),
                                      PhysConst::q_e * options.density_floor);
                    Real ns = n * raw / denom;
                    Real frequency = 1e4 * std::pow(ev / 3, -1.5);
                    sum += -3 * ns * PhysConst::q_e * frequency *
                           (1 - 1.5 * (ev - 1) / ev) / kelvin;
                }
                Real expectJ = -1.2 * power(p, SourceComponent::Joule) / T;
                error = std::max(error,
                                 std::abs(derivative(p, 0) - expectJ) /
                                     std::max(Real(1e-20), std::abs(expectJ)));
                error =
                    std::max(error, std::abs(derivative(p, 1) - sum) /
                                        std::max(Real(1e-20), std::abs(sum)));
            }
        }
        amrex::ParallelDescriptor::ReduceRealMax(error);
        require(error < 2e-8, "analytic local law derivative mismatch");
        amrex::Print() << "ANALYTIC relative_error=" << error << "\n";
    }
    void
    basis () {
        auto d = diagonal(), fine = diagonal(1e-5);
        require(d->Freeze(state, capacity, gamma) &&
                    fine->Freeze(state, capacity, gamma),
                "basis freeze failed");
        auto raw = clone(d->RawDiagonal());
        Real epsilon_error =
            difference(raw, fine->RawDiagonal()) / raw.norminf();
        require(epsilon_error < 3e-8, "source step refinement unstable");
        Real max_error = 0, refinement = 0;
        for (auto p : {amrex::IntVect(0, 0), amrex::IntVect(0, 7),
                       amrex::IntVect(15, 15), amrex::IntVect(4, 7),
                       amrex::IntVect(7, 7)}) {
            Real const step = value(old, p[0], p[1]) * 2e-5;
            Real fd[2];
            for (int level = 0; level < 2; ++level) {
                Real delta = step / std::pow(2., level);
                Real sides[2];
                for (int side = 0; side < 2; ++side) {
                    MF::Copy(energy, old, 0, 0, 1, 0);
                    for (amrex::MFIter mfi(energy); mfi.isValid(); ++mfi)
                        if (mfi.validbox().contains(p))
                            energy[mfi](p) += (side == 0 ? delta : -delta);
                    physical();
                    sides[side] = value(qcell, p[0], p[1]);
                }
                fd[level] = (sides[0] - sides[1]) / (2 * delta);
            }
            Real exact = value(raw, p[0], p[1]);
            max_error =
                std::max(max_error, std::abs(fd[1] - exact) /
                                        std::max(std::abs(exact), Real(1e-20)));
            refinement = std::max(refinement,
                                  std::abs(fd[1] - fd[0]) /
                                      std::max(std::abs(exact), Real(1e-20)));
        }
        require(max_error < 4e-8 && refinement < 4e-8,
                "same-cell moment/source diagonal differs from full "
                "directional probe");
        amrex::Print() << "BASIS relative_error=" << max_error
                       << " epsilon_refinement=" << epsilon_error
                       << " probe_refinement=" << refinement << "\n";
    }
};
void
run (std::string const& name, int box, bool periodic) {
    Fixture f(box, periodic);
    if (name == "kelvin") {
        f.options.use_heating_eta = false;
        f.options.field_eta_uses_kelvin = true;
        f.options.field_eta_kelvin = f.eta_kelvin.compile<4>();
        f.analytic();
        return;
    }
    if (name == "rawfloor") {
        for (auto& raw : f.charge)
            raw.setVal(.1 * PhysConst::q_e * f.options.density_floor);
        f.analytic();
        f.basis();
        return;
    }
    if (name == "analytic") {
        f.analytic();
        return;
    }
    if (name == "basis") {
        f.basis();
        return;
    }
    if (name == "clip") {
        f.options.relaxation = false;
        f.options.heating_eta = f.grow.compile<4>();
        auto d = f.diagonal();
        require(d->Freeze(f.state, f.capacity, f.gamma),
                "growth freeze failed");
        require(d->RawDiagonal().min(0) > 0 && d->PCDiagonal().norminf() == 0,
                "positive feedback not clipped in PC");
        f.options.heating_eta = f.eta.compile<4>();
        auto negative = f.diagonal();
        require(negative->Freeze(f.state, f.capacity, f.gamma),
                "damping freeze failed");
        require(negative->PCDiagonal().max(0) < 0 &&
                    difference(negative->RawDiagonal(),
                               negative->PCDiagonal()) == 0,
                "negative damping lost");
        for (auto h : {1e-9, 1., 1e6})
            require(1 - h * negative->PCDiagonal().max(0) >= 1,
                    "PC mass positivity lost");
        return;
    }
    if (name == "gates") {
        f.rho.setVal(PhysConst::q_e * f.options.density_floor);
        auto d = f.diagonal();
        require(d->Freeze(f.state, f.capacity, f.gamma) &&
                    d->RawDiagonal().norminf() == 0,
                "density gate derivative nonzero");
        f.rho.setVal(3e19 * PhysConst::q_e);
        f.options.joule_density_gate = 4e19;
        f.options.joule_taper = true;
        auto gate = f.diagonal();
        require(gate->Freeze(f.state, f.capacity, f.gamma) &&
                    gate->NodalDerivative().norminf(0) == 0 &&
                    gate->NodalDerivative().norminf(1) > 0,
                "Joule gate changed relaxation");
        return;
    }
    if (name == "kink") {
        f.options.redirect_joule = true;
        f.options.redirect_temperature_ev = 2;
        f.constant_temperature(2);
        auto d = f.diagonal();
        require(d->Freeze(f.state, f.capacity, f.gamma),
                "redirect kink freeze failed");
        require(d->NodalDerivative().norminf(0) == 0 &&
                    d->KinkMask().min(0) == 1 &&
                    d->NodalDerivative().norminf(1) > 0,
                "redirect kink not channel-local");
        f.options.redirect_joule = false;
        f.constant_temperature(1e-3);
        auto floor = f.diagonal();
        require(floor->Freeze(f.state, f.capacity, f.gamma),
                "nu floor freeze failed");
        require(floor->KinkMask().min(1) == 1 &&
                    floor->NodalDerivative().norminf(1) == 0 &&
                    floor->NodalDerivative().norminf(0) > 0,
                "nu floor kink not channel-local");
        return;
    }
    auto d = f.diagonal();
    require(d->Freeze(f.state, f.capacity, f.gamma), "freeze failed");
    auto original = clone(d->RawDiagonal()), inputs = clone(f.te);
    if (name == "purity") {
        auto saved_capacity = clone(f.capacity);
        amrex::ResetRandomSeed(681);
        Real expected = amrex::Random();
        amrex::ResetRandomSeed(681);
        require(d->Freeze(f.state, f.capacity, f.gamma),
                "repeat freeze failed");
        require(amrex::Random() == expected && difference(inputs, f.te) == 0,
                "PC probe mutated RNG/Te");
        f.te.mult(1.1, 0, 1, 0);
        f.capacity.mult(1.2, 0, 1, 0);
        require(difference(original, d->RawDiagonal()) == 0,
                "frozen derivative borrowed live inputs");
        require(d->Freeze(f.state, f.capacity, f.gamma), "B freeze failed");
        MF::Copy(f.te, inputs, 0, 0, 1, 0);
        MF::Copy(f.capacity, saved_capacity, 0, 0, 1, 0);
        require(d->Freeze(f.state, f.capacity, f.gamma),
                "A replay freeze failed");
        near(difference(original, d->RawDiagonal()), 0, 1e-15,
             "A/B/A source derivative differs");
        return;
    }
    if (name == "invalid") {
        auto saved = clone(f.capacity);
        f.capacity.setVal(-1);
        require(!d->Freeze(f.state, f.capacity, f.gamma),
                "negative capacity accepted");
        MF::Copy(f.capacity, saved, 0, 0, 1, 0);
        f.te.setVal(std::numeric_limits<Real>::quiet_NaN());
        require(!d->Freeze(f.state, f.capacity, f.gamma),
                "nonfinite thermal input accepted");
        MF::Copy(f.te, inputs, 0, 0, 1, 0);
        require(d->Freeze(f.state, f.capacity, f.gamma),
                "invalid recovery failed");
        require(difference(original, d->RawDiagonal()) == 0,
                "invalid retry retains stale slope");
        require(!d->Freeze(f.state, d->RawDiagonal(), f.gamma),
                "aliased output/capacity accepted");
        return;
    }
    amrex::Abort("unknown source diagonal case");
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        std::string name = "basis";
        int box = 4, periodic = 0;
        amrex::ParmParse pp("test");
        pp.query("case", name);
        pp.query("max_grid_size", box);
        pp.query("periodic", periodic);
        run(name, box, periodic);
        amrex::Print() << "PASS source_diagonal " << name << " box=" << box
                       << " periodic=" << periodic << "\n";
    }
    amrex::Finalize();
}
