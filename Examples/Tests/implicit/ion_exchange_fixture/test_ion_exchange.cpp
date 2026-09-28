/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/AcceptedIonExchange.H"
#include "FieldSolver/ImplicitSolvers/AcceptedIonExchange_K.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <cmath>
#include <limits>
#include <numbers>
using namespace warpx::thermal;
using amrex::Real;
using P = amrex::ParticleReal;
using MF = amrex::MultiFab;
void
require (bool ok, const char* text) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok, text);
}
void
near (Real a, Real b, Real rel, const char* text) {
    require(std::abs(a - b) <=
                rel * std::max({std::abs(a), std::abs(b), Real(1e-100)}),
            text);
}
struct Fixture {
    amrex::Geometry g;
    amrex::BoxArray cells, nodes;
    amrex::DistributionMapping dm;
    std::vector<KineticSpeciesDescriptor> descriptors;
    IonExchangeOptions options;
    std::unique_ptr<MF> rho, te, rates;
    std::array<std::unique_ptr<MF>, 3> drift, raw, ti;
    ThermalSourceState state;
    Fixture (int maxbox) {
        amrex::Box box(amrex::IntVect(0, 0), amrex::IntVect(15, 15));
        amrex::RealBox physical({0., 0.}, {1.6, 1.6});
        int periodic[2] = {0, 1};
        g.define(box, &physical, 1, periodic);
        cells.define(box);
        cells.maxSize(maxbox);
        nodes = amrex::convert(cells, amrex::IntVect(1));
        dm = amrex::DistributionMapping(cells);
        rho = std::make_unique<MF>(nodes, dm, 1, 0);
        te = std::make_unique<MF>(nodes, dm, 1, 0);
        rates =
            std::make_unique<MF>(nodes, dm, 3 * IonSourceComponent::Count, 0);
        rho->setVal(2 * PhysConst::q_e);
        te->setVal(3 * PhysConst::q_e / PhysConst::kb);
        rates->setVal(0);
        options.dt = .07;
        options.relaxation = true;
        options.redirect = true;
        options.raw_density_floor = 1;
        for (int s = 0; s < 3; ++s) {
            KineticSpeciesDescriptor descriptor;
            descriptor.name = std::array<std::string, 3>{"D", "T", "alpha"}[s];
            descriptor.mass = (s + 2) * PhysConst::m_p;
            descriptor.charge_number = s == 2 ? 2 : 1;
            descriptor.relaxation_excluded = s == 2;
            descriptors.push_back(descriptor);
            drift[s] = std::make_unique<MF>(nodes, dm, 1, 0);
            drift[s]->setVal((s + 1) * 100.);
            raw[s] = std::make_unique<MF>(nodes, dm, 1, 0);
            raw[s]->setVal(PhysConst::q_e);
            ti[s] = std::make_unique<MF>(cells, dm, 1, 0);
            ti[s]->setVal(.5 + s);
            ThermalSourceSpecies species;
            species.charge_number = descriptor.charge_number;
            species.relaxation_excluded = descriptor.relaxation_excluded;
            species.raw_charge = raw[s].get();
            state.species.push_back(species);
            if (s == 2)
                continue;
            rates->setVal(2 + s, s * 4 + IonSourceComponent::CollisionFrequency,
                          1, 0);
            rates->setVal((4 + s) * 1e-19,
                          s * 4 + IonSourceComponent::RedirectVarianceRate, 1,
                          0);
            rates->setVal((5 + s) * 1e-18,
                          s * 4 + IonSourceComponent::RelaxationPower, 1, 0);
            rates->setVal((6 + s) * 1e-18,
                          s * 4 + IonSourceComponent::RedirectPower, 1, 0);
        }
        state.raw_charge = rho.get();
        state.temperature_kelvin = te.get();
    }
    std::array<const MF*, 3>
    velocity () {
        return {drift[0].get(), drift[1].get(), drift[2].get()};
    }
    std::unique_ptr<AcceptedIonExchange>
    exchange () {
        return std::make_unique<AcceptedIonExchange>(g, cells, dm, descriptors,
                                                     options);
    }
    struct Cloud {
        std::vector<amrex::Gpu::DeviceVector<amrex::IntVect>> indices;
        std::vector<amrex::Gpu::DeviceVector<P>> theta, weight;
        std::vector<std::array<amrex::Gpu::DeviceVector<P>, 3>> u;
        std::vector<IonExchangeParticleView> views;
    };
    Cloud
    cloud (bool invalid = false, bool fail = false) {
        Cloud cloud;
        // Pair of cells with three species, distributed by actual owning grids.
        for (int s = 0; s < 3; ++s)
            for (amrex::MFIter mfi(cells, dm); mfi.isValid(); ++mfi) {
                std::vector<amrex::IntVect> indices;
                for (auto cell : {amrex::IntVect(0, 0), amrex::IntVect(1, 1),
                                  amrex::IntVect(9, 7), amrex::IntVect(15, 15)})
                    if (mfi.validbox().contains(cell))
                        indices.push_back(cell);
                if (indices.empty())
                    continue;
                if (invalid && s == 0)
                    indices[0] = amrex::IntVect(-1, -1);
                cloud.indices.emplace_back(indices.size());
                auto& index = cloud.indices.back();
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, indices.begin(),
                                 indices.end(), index.begin());
                cloud.theta.emplace_back(indices.size(), .7);
                cloud.weight.emplace_back(indices.size(), fail ? 1e300 : 2e10);
                cloud.u.emplace_back();
                for (int d = 0; d < 3; ++d)
                    cloud.u.back()[d].resize(indices.size(),
                                             fail ? 0 : (d + 1) * 1e3);
                IonExchangeParticleView view;
                view.species = s;
                view.grid_index = mfi.index();
                view.count = indices.size();
                view.cell = index.data();
                view.theta = cloud.theta.back().data();
                view.weight = cloud.weight.back().data();
                for (int d = 0; d < 3; ++d)
                    view.momentum[d] = cloud.u.back()[d].data();
                cloud.views.push_back(view);
            }
        return cloud;
    }
};
std::vector<P>
snapshot (Fixture::Cloud const& cloud) {
    std::vector<P> out;
    for (auto const& u : cloud.u)
        for (auto const& v : u) {
            std::vector<P> host(v.size());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.end(),
                             host.begin());
            out.insert(out.end(), host.begin(), host.end());
        }
    return out;
}
void
kernel_test (std::string const& name) {
    amrex::GpuArray<P, 3> u{2e4, -1e4, 3e4}, v{400, 700, -900}, r{.2, -.7, 1.3};
    P mass = 2 * PhysConst::m_p, temperature = 2e5, nu = 2.3, energy = 8e-20;
    Real dt = .07;
    auto result = ApplyIonExchangeKick(u, v, 0, nu, temperature, energy, mass,
                                       2e10, dt, r);
    P drag = -std::expm1(-nu * dt),
      sigma = std::sqrt(
          (-PhysConst::kb * temperature * std::expm1(-2 * nu * dt) + energy) /
          mass);
    for (int d = 0; d < 3; ++d)
        require(result.momentum[d] ==
                    u[d] + (-drag * (u[d] - v[d]) + sigma * r[d]),
                "native OU arithmetic differs");
    near(result.realized_total,
         result.realized_relaxation + result.realized_redirect, 8e-15,
         "counterfactual energies do not telescope");
    if (name == "rotation")
        for (int point = 0; point < 12; ++point) {
            // An azimuthal ring with identical cylindrical state; rotate each
            // supplied Gaussian realization along with the native momentum.
            Real angle = .61 + point * 2 * std::numbers::pi / 12,
                 c = std::cos(angle), s = std::sin(angle);
            auto rotate = [&] (auto a) {
                return amrex::GpuArray<P, 3>{c * a[0] - s * a[1],
                                             s * a[0] + c * a[1], a[2]};
            };
            auto rotated =
                ApplyIonExchangeKick(rotate(u), v, angle, nu, temperature,
                                     energy, mass, 2e10, dt, rotate(r));
            auto expected = rotate(result.momentum);
            for (int d = 0; d < 3; ++d)
                near(rotated.momentum[d], expected[d], 4e-15,
                     "ring rotation breaks covariance");
            near(rotated.realized_total, result.realized_total, 2e-14,
                 "ring rotation breaks energy invariance");
        }
    if (name == "expectation") {
        Real er = 0, ed = 0; // Exact cubature E[R_i]=0,E[R_i R_j]=delta_ij.
        for (int axis = 0; axis < 3; ++axis)
            for (int sign : {-1, 1}) {
                amrex::GpuArray<P, 3> draw{0, 0, 0};
                draw[axis] = sign * std::sqrt(3.);
                auto k = ApplyIonExchangeKick(u, v, 0, nu, temperature, energy,
                                              mass, 2e10, dt, draw);
                Real before = 0, after = 0;
                for (int d = 0; d < 3; ++d) {
                    before += u[d] * u[d];
                    after += k.momentum[d] * k.momentum[d];
                }
                er += .5 * mass * 2e10 * (after - before) / 6;
                ed = k.expected_relaxation + k.expected_redirect;
            }
        near(er, ed, 8e-15, "finite-step OU expectation mismatch");
    }
}
void
finite_step_test () {
    Real const mass = 2 * PhysConst::m_p, te = 3., ti = 1., nu = 2.;
    for (Real x : {.001, .01, .1, 1.}) {
        Real const dt = x / nu,
                   speed = std::sqrt(3 * PhysConst::q_e * ti / mass);
        Real measured_expectation = 0;
        for (int axis = 0; axis < 3; ++axis)
            for (int sign : {-1, 1}) {
                amrex::GpuArray<P, 3> u{0, 0, 0}, zero{0, 0, 0};
                u[axis] = sign * speed;
                auto kick = ApplyIonExchangeKick(
                    u, zero, 0, nu, te * PhysConst::q_e / PhysConst::kb, 0,
                    mass, 1. / 6., dt, zero);
                measured_expectation += kick.expected_relaxation;
            }
        Real const requested = 3 * PhysConst::q_e * (te - ti) * x;
        Real const expected =
            1.5 * PhysConst::q_e * (te - ti) * (-std::expm1(-2 * x));
        near(measured_expectation, expected, 2e-13,
             "thermal OU mean disagrees with finite-step analytic result");
        require(measured_expectation < requested,
                "finite-step electron/ion discrepancy hidden");
        amrex::Print() << "FINITE_STEP nu_dt=" << x
                       << " expected_over_requested="
                       << measured_expectation / requested
                       << " defect=" << measured_expectation - requested
                       << " J_per_ion\n";
    }
}
void
run (std::string const& name, int maxbox) {
    if (name == "finite_step") {
        finite_step_test();
        return;
    }
    if (name == "kernel" || name == "rotation" || name == "expectation") {
        kernel_test(name);
        return;
    }
    Fixture f(maxbox);
    if (name == "guard_cap")
        f.options.redirect_kick_cap = true;
    if (name == "guard_shunt")
        f.options.temperature_shunt = true;
    if (name == "guard_eb")
        f.options.embedded_boundary = true;
    if (name == "guard_mode")
        f.options.azimuthal_modes = 2;
    auto exchange = f.exchange();
    if (name == "relax_only") {
        for (int species = 0; species < 2; ++species) {
            f.rates->setVal(0, species * 4 + IonSourceComponent::RedirectPower,
                            1, 0);
            f.rates->setVal(
                0, species * 4 + IonSourceComponent::RedirectVarianceRate, 1,
                0);
        }
    }
    if (name == "redirect_only") {
        for (int species = 0; species < 2; ++species) {
            f.rates->setVal(
                0, species * 4 + IonSourceComponent::RelaxationPower, 1, 0);
            f.rates->setVal(
                0, species * 4 + IonSourceComponent::CollisionFrequency, 1, 0);
        }
    }
    if (name == "inactive") {
        f.rates->setVal(0);
        f.rho->setVal(0);
    }
    if (name == "failed") {
        // Finite coefficients, but enormous weighted random energy overflows
        // only AFTER drawing; tests terminal failure with no particle writes.
        f.rates->setVal(1e200, IonSourceComponent::RedirectVarianceRate, 1, 0);
    }
    if (name == "reference") {
        // Same nonlinear law sampled on nodes vs after cell gathering. These
        // differ by the local Te variance, even with the same frozen old Ti.
        for (amrex::MFIter mfi(*f.te); mfi.isValid(); ++mfi) {
            auto temperature = f.te->array(mfi), q = f.rates->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real const ev = 3 + .1 * i;
                    temperature(i, j, k) = ev * PhysConst::q_e / PhysConst::kb;
                    q(i, j, k, IonSourceComponent::CollisionFrequency) =
                        ev * ev + .5;
                });
        }
        amrex::Parser parser("rho*rho+Te*Te+Ti");
        parser.registerVariables({"rho", "Te", "Ti", "t"});
        auto ref = parser.compile<4>();
        require(exchange->Prepare(f.state, *f.rates, f.velocity(),
                                  {f.ti[0].get(), f.ti[1].get(), nullptr},
                                  &ref),
                "reference preparation failed");
        near(exchange->Coefficients(0).min(IonExchangeCoefficient::Nu), 9.805,
             3e-15, "reference replaced converged rate");
        near(exchange->Ledger()[0].reference_nu_max_difference, .0025, 1e-11,
             "legacy rate comparison wrong");
        return;
    }
    if (name == "invalid_prepare") {
        f.rates->setVal(-1, IonSourceComponent::CollisionFrequency, 1, 0);
        amrex::ResetRandomSeed(331);
        Real expected = amrex::Random();
        amrex::ResetRandomSeed(331);
        require(!exchange->Prepare(f.state, *f.rates, f.velocity()),
                "negative nu accepted");
        require(amrex::Random() == expected, "failed Prepare consumed RNG");
        f.rates->setVal(2, IonSourceComponent::CollisionFrequency, 1, 0);
        require(exchange->Prepare(f.state, *f.rates, f.velocity()),
                "recovery after bad rates failed");
        return;
    }
    require(exchange->Prepare(f.state, *f.rates, f.velocity()),
            "preparation failed");
    if (name == "alias_prepare") {
        amrex::Parser parser("Ti");
        parser.registerVariables({"rho", "Te", "Ti", "t"});
        auto ref = parser.compile<4>();
        auto const* owned = &exchange->Coefficients(0);
        require(!exchange->Prepare(f.state, *f.rates, f.velocity(),
                                   {owned, f.ti[1].get(), nullptr}, &ref),
                "aliased preparation input accepted");
        require(exchange->Prepare(f.state, *f.rates, f.velocity()),
                "private alias rejection poisoned clean preparation");
        return;
    }
    if (name == "prepare") {
        Real const request = exchange->Ledger()[0].requested_relaxation;
        Real const volume = std::numbers::pi * 1.6 * 1.6 * 1.6;
        near(request, volume * .07 * 5e-18, 5e-15,
             "physical requested integral wrong");
        // Repeat pure preparation without advancing CPU RNG.
        amrex::ResetRandomSeed(931);
        Real expect = amrex::Random();
        amrex::ResetRandomSeed(931);
        require(exchange->Prepare(f.state, *f.rates, f.velocity()),
                "repeat preparation failed");
        require(amrex::Random() == expect, "Prepare consumed RNG");
        f.te->setVal(999);
        f.rates->setVal(999);
        for (auto& v : f.drift)
            v->setVal(999);
        near(exchange->Coefficients(0).min(
                 IonExchangeCoefficient::TemperatureKelvin),
             3 * PhysConst::q_e / PhysConst::kb, 1e-15,
             "prepared coefficient borrowed inputs");
        require(exchange->Ledger()[0].requested_relaxation == request,
                "prepared ledger borrowed rates");
        return;
    }
    auto cloud = f.cloud(name == "invalid", name == "failed");
    auto before = snapshot(cloud);
    if (name == "alias" && !cloud.views.empty())
        cloud.views.push_back(cloud.views.front());
    amrex::ResetRandomSeed(773 + amrex::ParallelDescriptor::MyProc());
    Real const untouched_rng = amrex::Random();
    amrex::ResetRandomSeed(773 + amrex::ParallelDescriptor::MyProc());
    bool const committed = exchange->CommitOnce(cloud.views);
    if (name == "invalid" || name == "failed" || name == "alias") {
        require(!committed, "invalid endpoint/stochastic candidate accepted");
        if (name == "invalid" || name == "alias")
            require(amrex::Random() == untouched_rng,
                    "preflight rejection consumed RNG");
        require(snapshot(cloud) == before, "failed commit mutated particles");
        require(!exchange->CommitOnce(cloud.views), "failed commit retried");
        require(exchange->Status() ==
                    (name != "failed" ? IonExchangeStatus::Rejected
                                      : IonExchangeStatus::FailedAfterDraw),
                "wrong failure phase");
        return;
    }
    require(committed, "commit failed");
    auto after = snapshot(cloud);
    for (std::size_t t = 0; t < cloud.views.size(); ++t)
        if (cloud.views[t].species == 2) {
            for (int d = 0; d < 3; ++d) {
                std::vector<P> values(cloud.u[t][d].size());
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                                 cloud.u[t][d].begin(), cloud.u[t][d].end(),
                                 values.begin());
                for (auto value : values)
                    require(value == (d + 1) * 1e3, "excluded alpha changed");
            }
        }
    if (name == "inactive") {
        require(after == before, "inactive operator changed momenta");
        require(amrex::Random() == untouched_rng,
                "inactive operator consumed RNG");
    }
    for (auto const& l : exchange->Ledger()) {
        near(l.realized_total, l.realized_relaxation + l.realized_redirect,
             2e-14, "global measured split mismatch");
        if (name == "relax_only")
            require(l.realized_redirect == 0 && l.expected_redirect == 0,
                    "absent redirect delivered energy");
        if (name == "redirect_only")
            require(l.realized_relaxation == 0 && l.expected_relaxation == 0,
                    "absent relaxation delivered energy");
    }
    if (name == "once") {
        require(!exchange->CommitOnce(cloud.views),
                "duplicate accepted kick allowed");
        require(!exchange->Prepare(f.state, *f.rates, f.velocity()),
                "consumed transaction rearmed");
        require(snapshot(cloud) == after, "second attempt changed momenta");
    }
    if (name == "commit") {
        Real actual = 0;
        std::size_t offset = 0;
        for (auto const& view : cloud.views) {
            auto mass = f.descriptors[view.species].mass;
            for (long p = 0; p < view.count; ++p)
                actual +=
                    2e10 *
                    (Algorithms::KineticEnergy<Real>(
                         after[offset + p], after[offset + view.count + p],
                         after[offset + 2 * view.count + p], mass) -
                     Algorithms::KineticEnergy<Real>(
                         before[offset + p], before[offset + view.count + p],
                         before[offset + 2 * view.count + p], mass));
            offset += 3 * view.count;
        }
        amrex::ParallelDescriptor::ReduceRealSum(actual);
        Real ledger = 0;
        for (auto const& l : exchange->Ledger())
            ledger += l.realized_total;
        near(actual, ledger, 3e-14,
             "ledger differs from actual stored native energy");
        amrex::Print() << "ENERGY requested="
                       << exchange->Ledger()[0].requested_relaxation +
                              exchange->Ledger()[0].requested_redirect
                       << " expected="
                       << exchange->Ledger()[0].expected_relaxation +
                              exchange->Ledger()[0].expected_redirect
                       << " measured=" << exchange->Ledger()[0].realized_total
                       << "\n";
    }
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        std::string name = "commit";
        int maxbox = 4;
        amrex::ParmParse pp("test");
        pp.query("case", name);
        pp.query("max_grid_size", maxbox);
        run(name, maxbox);
        amrex::Print() << "PASS ion_exchange " << name << " box=" << maxbox
                       << "\n";
    }
    amrex::Finalize();
}
