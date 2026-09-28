/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ExpectedIonExchangeChannel.H"
#include "FieldSolver/ImplicitSolvers/ExpectedIonThermalPartner.H"
#include "FieldSolver/ImplicitSolvers/ImplicitIonEndpointTrial.H"
#include "FieldSolver/ImplicitSolvers/IonExchangeMomentAudit.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <cmath>
using namespace warpx::thermal;
using R = amrex::Real;
using P = amrex::ParticleReal;
using MF = amrex::MultiFab;
using PC = amrex::ParticleContainerPureSoA<PIdx::nattribs, 0>;
void
check (bool ok, char const* why) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ok, why);
}
void
near (R a, R b, R rel, char const* why) {
    check(std::abs(a - b) <=
              rel * std::max({std::abs(a), std::abs(b), R(1.e-90)}),
          why);
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        int box = 4;
        std::string test = "endpoint";
        amrex::ParmParse pp("test");
        pp.query("max_grid_size", box);
        pp.query("case", test);
        amrex::Box domain(amrex::IntVect(0, 0), amrex::IntVect(15, 15));
        amrex::RealBox rb({0., 0.}, {1.6, 1.6});
        int periodic[2]{0, 1};
        amrex::Geometry g(domain, &rb, 1, periodic);
        amrex::BoxArray ba(domain);
        ba.maxSize(box);
        amrex::DistributionMapping dm(ba);
        auto nodes = amrex::convert(ba, amrex::IntVect(1));
        MF rho(nodes, dm, 1, 1), te(nodes, dm, 1, 1), rates(nodes, dm, 12, 1);
        rho.setVal(2 * PhysConst::q_e);
        te.setVal(3 * PhysConst::q_e / PhysConst::kb);
        rates.setVal(0.);
        std::array<MF, 3> drift;
        std::array<const MF*, 3> vel;
        for (int d = 0; d < 3; ++d) {
            drift[d].define(nodes, dm, 1, 1);
            drift[d].setVal((d + 1) * 100.);
            vel[d] = &drift[d];
        }
        std::vector<KineticSpeciesDescriptor> desc;
        ThermalSourceState state;
        state.raw_charge = &rho;
        state.temperature_kelvin = &te;
        for (int s = 0; s < 3; ++s) {
            KineticSpeciesDescriptor d;
            d.name = std::to_string(s);
            d.mass = (2 + s) * PhysConst::m_p;
            d.charge_number = s == 2 ? 2 : 1;
            d.relaxation_excluded = s == 2;
            desc.push_back(d);
            ThermalSourceSpecies st;
            st.charge_number = d.charge_number;
            st.relaxation_excluded = d.relaxation_excluded;
            st.raw_charge = &rho;
            state.species.push_back(st);
            if (s < 2) {
                rates.setVal(2 + s,
                             s * 4 + IonSourceComponent::CollisionFrequency, 1,
                             1);
                rates.setVal(5.e-19,
                             s * 4 + IonSourceComponent::RedirectVarianceRate,
                             1, 1);
            }
        }
        IonExchangeOptions options;
        options.dt = .07;
        options.raw_density_floor = 1;
        options.relaxation = true;
        options.redirect = true;
        AcceptedIonExchange ex(g, ba, dm, desc, options);
        check(ex.Prepare(state, rates, vel), "prepare");
        std::vector<std::unique_ptr<PC>> input;
        std::vector<ImplicitIonMidpointTile> views;
        for (int s = 0; s < 3; ++s) {
            auto pc = std::make_unique<PC>(g, dm, ba);
            amrex::Vector<std::string> names;
            for (auto n : PIdx::names)
                names.push_back(n);
            pc->SetSoACompileTimeNames(names, {});
            for (auto n : {"x_n", "y_n", "z_n", "ux_n", "uy_n", "uz_n"})
                pc->AddRealComp(n, 0);
            for (amrex::MFIter it(ba, dm); it.isValid(); ++it)
                for (int group = 0; group < 2; ++group) {
                    auto cell = amrex::IntVect(4, group ? 15 : 7);
                    if (!it.validbox().contains(cell))
                        continue;
                    auto& tile =
                        pc->DefineAndReturnParticleTile(0, it.index(), 0);
                    long const begin = tile.numParticles();
                    tile.resize(begin + 72);
                    auto a = tile.getParticleTileData();
                    amrex::ParallelFor(72, [=] AMREX_GPU_DEVICE(long n) {
                        long const k = begin + n;
                        auto p = a[k];
                        p.id() = 1 + s * 10000 + group * 100 + n;
                        p.cpu() = 0;
                        R const angle = (n / 6) * 2 * MathConst::pi / 12 + .2,
                                cs = std::cos(angle), sn = std::sin(angle);
                        R const ur = 1000 + (n % 6 == 0   ? 400
                                             : n % 6 == 1 ? -400
                                                          : 0),
                                ut = 600 + (n % 6 == 2   ? 400
                                            : n % 6 == 3 ? -400
                                                         : 0),
                                uz = -500 + (n % 6 == 4   ? 400
                                             : n % 6 == 5 ? -400
                                                          : 0);
                        R const ux = cs * ur - sn * ut, uy = sn * ur + cs * ut;
                        a.rdata(PIdx::r)[k] = .45;
                        a.rdata(PIdx::z)[k] = group ? 1.58 : .78;
                        a.rdata(PIdx::theta)[k] = angle;
                        a.rdata(PIdx::w)[k] = (1 + n % 3) * 1.e10;
                        a.rdata(PIdx::ux)[k] = .5 * (ux + 200);
                        a.rdata(PIdx::uy)[k] = .5 * (uy + 300);
                        a.rdata(PIdx::uz)[k] = .5 * (uz + 400);
                        a.m_runtime_rdata[0][k] = .43 * cs;
                        a.m_runtime_rdata[1][k] = .43 * sn;
                        a.m_runtime_rdata[2][k] = group ? 1.53 : .73;
                        a.m_runtime_rdata[3][k] = 200;
                        a.m_runtime_rdata[4][k] = 300;
                        a.m_runtime_rdata[5][k] = 400;
                    });
                }
            for (PC::ParIterType it(*pc, 0); it.isValid(); ++it) {
                auto& a = it.GetStructOfArrays();
                ImplicitIonMidpointTile t;
                t.species = s;
                t.grid = it.index();
                t.tile = it.LocalTileIndex();
                t.count = it.numParticles();
                t.idcpu = a.GetIdCPUData().data();
                t.position = {a.GetRealData(PIdx::r).data(),
                              a.GetRealData(PIdx::z).data()};
                t.theta = a.GetRealData(PIdx::theta).data();
                t.weight = a.GetRealData(PIdx::w).data();
                t.momentum = {a.GetRealData(PIdx::ux).data(),
                              a.GetRealData(PIdx::uy).data(),
                              a.GetRealData(PIdx::uz).data()};
                t.old_position = {a.GetRealData("x_n").data(),
                                  a.GetRealData("y_n").data(),
                                  a.GetRealData("z_n").data()};
                t.old_momentum = {a.GetRealData("ux_n").data(),
                                  a.GetRealData("uy_n").data(),
                                  a.GetRealData("uz_n").data()};
                views.push_back(t);
            }
            input.push_back(std::move(pc));
        }
        ImplicitIonEndpointTrial trial(ex);
        amrex::ResetRandomSeed(777);
        R const ref = amrex::Random();
        amrex::ResetRandomSeed(777);
        check(trial.Refresh(views), "endpoint redistribution");
        near(amrex::Random(), ref, 0, "no RNG");
        IonExchangeMomentAudit moments(ex);
        check(moments.Evaluate(trial, ex), "moments");
        auto l = moments.Ledger();
        MF budget(ba, dm, 1, 0);
        budget.setVal(0);
        ExpectedIonEnergyOptions eo;
        eo.mode = ExpectedIonEnergyMode::RelativisticQuadrature;
        ExpectedIonExchangeSource energy(ex, eo);
        check(energy.Evaluate(ex, trial.Views(), budget,
                              ExpectedIonEndpointContract::
                                  CompleteRedistributedPreCollisionEndpoint),
              "expected work");
        auto work = energy.Ledger();
        if (test == "endpoint" || test == "moments" || test == "purity") {
            for (int s = 0; s < 2; ++s) {
                using C = IonExchangeMomentComponent;
                using E = ExpectedIonWorkComponent;
                near(l[s][C::Number], 288.e10, 2.e-15,
                     "physical count excludes alpha");
                near(l[s][C::DeterministicBulk] + l[s][C::ThermalRelaxation] +
                         l[s][C::Redirect],
                     work[s][E::NonrelativisticRelaxation] +
                         work[s][E::NonrelativisticRedirect],
                     2.e-14, "bulk/thermal identity");
                // Weighted +/- Cartesian-cylindrical shell: means differ by
                // -400/12 in each component.
                R const mean[3]{1000 - 400. / 12, 600 + 800. / 12,
                                -500 - 400. / 12};
                R variance = 160000 - 6 * std::pow(400. / 12, 2);
                R const n = 288.e10,
                        beta = -std::expm1(-2 * (s + 2) * options.dt),
                        b = -std::expm1(-(s + 2) * options.dt),
                        m = desc[s].mass;
                R bulk = 0;
                for (int d = 0; d < 3; ++d) {
                    R delta = -b * (mean[d] - (d + 1) * 100);
                    bulk += .5 * m * n * delta * (2 * mean[d] + delta);
                }
                near(l[s][C::DeterministicBulk], bulk, 2.e-13,
                     "bulk closed oracle");
                near(l[s][C::ThermalRelaxation],
                     .5 * beta * n * (9 * PhysConst::q_e - m * variance),
                     2.e-14, "thermal variance oracle");
                check(l[s][C::FiniteSampleBulkNoise] > 0,
                      "finite sample mean noise recorded");
            }
            check(l[2][IonExchangeMomentComponent::Number] == 0,
                  "excluded alpha");
            int bad = 0;
            for (auto const& v : trial.Views()) {
                std::vector<amrex::IntVect> ids(v.count);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.cell,
                                 v.cell + v.count, ids.begin());
                for (auto iv : ids)
                    bad += iv[0] != 4 || (iv[1] != 0 && iv[1] != 8);
            }
            amrex::ParallelDescriptor::ReduceIntSum(bad);
            check(bad == 0, "true endpoint NGP, seams/periodic wrap");
        }
        if (test == "partner") {
            ExpectedIonThermalPartner partner(ex);
            check(!partner.Evaluate(moments, energy),
                  "unbounded convention rejected");
            eo.mode = ExpectedIonEnergyMode::NonrelativisticBounded;
            eo.relative_component_tolerance = 1.e-6;
            ExpectedIonExchangeSource nr(ex, eo);
            check(nr.Evaluate(ex, trial.Views(), budget,
                              ExpectedIonEndpointContract::
                                  CompleteRedistributedPreCollisionEndpoint),
                  "bounded NR");
            amrex::ResetRandomSeed(987);
            R const draw = amrex::Random();
            amrex::ResetRandomSeed(987);
            check(partner.Evaluate(moments, nr), "population thermal partner");
            near(amrex::Random(), draw, 0, "partner no RNG");
            R thermal = 0, bulk = 0, redirect = 0;
            for (auto const& v : l) {
                thermal += v[IonExchangeMomentComponent::ThermalRelaxation];
                bulk += v[IonExchangeMomentComponent::DeterministicBulk];
                redirect += v[IonExchangeMomentComponent::Redirect];
            }
            near(partner.ThermalEnergy(), thermal, 1.e-15,
                 "same endpoint thermal work");
            near(partner.ElectronEnergy(), -thermal, 2.e-15,
                 "physical cell-volume partner");
            check(std::abs(bulk) > 0 && redirect > 0,
                  "mechanical/redirect work nonzero and separate");
            eo.relative_component_tolerance = 0;
            eo.absolute_tolerance_joule = 1.e-40;
            ExpectedIonExchangeSource tight(ex, eo);
            check(
                !tight.Evaluate(ex, trial.Views(), budget,
                                ExpectedIonEndpointContract::
                                    CompleteRedistributedPreCollisionEndpoint),
                "NR bound reject");
            check(!partner.Evaluate(moments, tight),
                  "failed bound prevents physical source");
        }
        if (test == "finite_step") {
            R const te0 = 1.e5, ti0 = 2.e4, nu = 2.3, end = .37, ni = 3, ne = 7;
            R const ci = 1.5 * ni * PhysConst::kb,
                    ce = 1.5 * ne * PhysConst::kb;
            R const equilibrium = (ce * te0 + ci * ti0) / (ce + ci);
            R const exact =
                equilibrium + ci / (ce + ci) * (te0 - ti0) *
                                  std::exp(-2 * nu * (1 + ci / ce) * end);
            R prior = 0;
            for (int steps : {16, 32, 64, 128}) {
                R const dt = end / steps, beta = -std::expm1(-2 * nu * dt);
                R te = te0, ti = ti0;
                for (int n = 0; n < steps; ++n) {
                    R const tm = (te + (.5 * ci / ce) * beta * ti) /
                                 (1 + (.5 * ci / ce) * beta);
                    R const heat = PopulationThermalOUIncrement(
                        ni, 3 * ni * PhysConst::kb * ti / PhysConst::m_p,
                        PhysConst::m_p, tm, nu, dt);
                    te -= heat / ce;
                    ti += heat / ci;
                }
                near(ce * te + ci * ti, ce * te0 + ci * ti0, 2.e-15,
                     "paired mean energy");
                R const error = std::abs(te - exact);
                if (prior > 0)
                    check(prior / error > 3.98 && prior / error < 4.02,
                          "completed mean second order");
                prior = error;
                amrex::Print() << "thermal partner refinement steps=" << steps
                               << " error=" << error << "\n";
            }
            R prior_defect = 0;
            for (R dt : {.02, .01, .005}) {
                R const heat = PopulationThermalOUIncrement(
                    ni, 3 * ni * PhysConst::kb * ti0 / PhysConst::m_p,
                    PhysConst::m_p, te0, nu, dt);
                R const old = 3 * ni * PhysConst::kb * nu * dt * (te0 - ti0);
                R const defect = old - heat;
                if (prior_defect > 0)
                    check(prior_defect / defect > 3.9 &&
                              prior_defect / defect < 4.1,
                          "legacy finite-step local quadratic mismatch");
                prior_defect = defect;
            }
        }
        if (test == "finite_step_species") {
            R const te0 = 1.e5, ti0[2]{2.e4, 5.5e4}, nu[2]{2.3, 3.7},
                    ni[2]{3, 2}, ne = 7, end = .37;
            R const ce = 1.5 * ne * PhysConst::kb,
                    ci[2]{1.5 * ni[0] * PhysConst::kb,
                          1.5 * ni[1] * PhysConst::kb};
            R const initial = ce * te0 + ci[0] * ti0[0] + ci[1] * ti0[1];
            // Independent exact exponential of the two temperature-difference
            // rows.
            R const aa = 2 * nu[0] * (1 + ci[0] / ce),
                    bb = 2 * nu[1] * ci[1] / ce;
            R const cc = 2 * nu[0] * ci[0] / ce,
                    dd = 2 * nu[1] * (1 + ci[1] / ce);
            R const center = .5 * (aa + dd),
                    root = std::sqrt(.25 * (aa - dd) * (aa - dd) + bb * cc);
            R const f = std::exp(-center * end), ch = std::cosh(root * end),
                    sh = std::sinh(root * end) / root;
            R const d0 = te0 - ti0[0], d1 = te0 - ti0[1];
            R const exact0 =
                f * ((ch - sh * (aa - center)) * d0 - sh * bb * d1);
            R const exact1 =
                f * (-sh * cc * d0 + (ch - sh * (dd - center)) * d1);
            R const exact_te = (initial + ci[0] * exact0 + ci[1] * exact1) /
                               (ce + ci[0] + ci[1]);
            R prior = 0;
            for (int steps : {16, 32, 64, 128}) {
                R const dt = end / steps, beta[2]{-std::expm1(-2 * nu[0] * dt),
                                                  -std::expm1(-2 * nu[1] * dt)};
                R te = te0, ti[2]{ti0[0], ti0[1]};
                for (int step = 0; step < steps; ++step) {
                    R const tm =
                        (ce * te + .5 * (ci[0] * beta[0] * ti[0] +
                                         ci[1] * beta[1] * ti[1])) /
                        (ce + .5 * (ci[0] * beta[0] + ci[1] * beta[1]));
                    for (int sp = 0; sp < 2; ++sp) {
                        R const mass = (sp + 1) * PhysConst::m_p;
                        R const heat = PopulationThermalOUIncrement(
                            ni[sp], 3 * ni[sp] * PhysConst::kb * ti[sp] / mass,
                            mass, tm, nu[sp], dt);
                        te -= heat / ce;
                        ti[sp] += heat / ci[sp];
                    }
                }
                near(ce * te + ci[0] * ti[0] + ci[1] * ti[1], initial, 3.e-15,
                     "multispecies paired mean energy");
                R const error = std::abs(te - exact_te);
                if (prior > 0)
                    check(prior / error > 3.95 && prior / error < 4.05,
                          "two different rate second order");
                prior = error;
                amrex::Print()
                    << "multispecies partner refinement steps=" << steps
                    << " error=" << error << "\n";
            }
        }
        if (test == "derivative") {
            R const n = 4.e12, m = 2 * PhysConst::m_p, t = 2.e4, s = n * 3.e8,
                    nu0 = 2.3, dt = .04;
            auto f = [&] (R temperature) {
                return PopulationThermalOUIncrement(
                    n, s, m, temperature, nu0 * std::sqrt(temperature / t), dt);
            };
            R const beta = -std::expm1(-2 * nu0 * dt),
                    dbeta = dt * nu0 / t * std::exp(-2 * nu0 * dt);
            R const analytic =
                .5 * ((3 * n * PhysConst::kb * t - m * s) * dbeta +
                      3 * n * PhysConst::kb * beta);
            R prior = 0;
            for (R eps : {1.e-2, 5.e-3, 2.5e-3}) {
                R const h = eps * t, fd = (f(t + h) - f(t - h)) / (2 * h);
                R const error = std::abs((fd - analytic) / analytic);
                if (prior > 0)
                    check(prior / error > 3.98 && prior / error < 4.02,
                          "nonlinear nu derivative refinement");
                prior = error;
            }
            check(prior < 2.e-6, "live thermal derivative");
            near(PopulationThermalOUIncrement(n, s, m, t, 0, dt), 0, 0,
                 "zero rate gate");
            check(PopulationThermalOUIncrement(n, s, m, 1., nu0, dt) < 0,
                  "signed ion cooling");
        }
        if (test == "noise") {
            // Exhaustive independent +/- unit draws have the exact second
            // moments of Gaussian noise for this quadratic observable.
            R const weights[2]{1, 2}, n = 3, w2 = 5, m = 2 * PhysConst::m_p,
                                      nu = 2.1, dt = .08;
            R const temp = 3 * PhysConst::q_e / PhysConst::kb,
                    H = .4 * PhysConst::q_e;
            R const u[2][3]{{1000, 300, -200}, {-200, 700, 900}},
                ve[3]{100, 200, 300};
            R mean[3]{}, postmean[3]{};
            R s = 0, k0 = 0;
            R const a = std::exp(-nu * dt), beta = -std::expm1(-2 * nu * dt);
            R const sigma = std::sqrt((PhysConst::kb * temp * beta + H) / m);
            for (int d = 0; d < 3; ++d) {
                mean[d] = (u[0][d] + 2 * u[1][d]) / n;
                postmean[d] = a * mean[d] + (1 - a) * ve[d];
            }
            for (int p = 0; p < 2; ++p)
                for (int d = 0; d < 3; ++d) {
                    s += weights[p] * std::pow(u[p][d] - mean[d], 2);
                    k0 += .5 * m * weights[p] * u[p][d] * u[p][d];
                }
            R expected_sample_thermal = 0, expected_sample_bulk = 0,
              expected_lab = 0;
            for (int draw = 0; draw < 64; ++draw) {
                R v[2][3], mu[3]{}, lab = 0;
                for (int p = 0; p < 2; ++p)
                    for (int d = 0; d < 3; ++d) {
                        v[p][d] = a * u[p][d] + (1 - a) * ve[d] +
                                  sigma * ((draw >> (3 * p + d)) & 1 ? 1 : -1);
                        mu[d] += weights[p] * v[p][d] / n;
                        lab += .5 * m * weights[p] * v[p][d] * v[p][d];
                    }
                R sampled = 0, bulk = 0;
                for (int p = 0; p < 2; ++p)
                    for (int d = 0; d < 3; ++d)
                        sampled +=
                            .5 * m * weights[p] * std::pow(v[p][d] - mu[d], 2);
                for (int d = 0; d < 3; ++d)
                    bulk += .5 * m * n * (mu[d] * mu[d] - mean[d] * mean[d]);
                expected_sample_thermal += (sampled - .5 * m * s) / 64;
                expected_sample_bulk += bulk / 64;
                expected_lab += (lab - k0) / 64;
            }
            R const population =
                PopulationThermalOUIncrement(n, s, m, temp, nu, dt);
            R const redirect = 1.5 * n * H,
                    mean_noise =
                        1.5 * (PhysConst::kb * temp * beta + H) * w2 / n;
            R deterministic_bulk = 0;
            for (int d = 0; d < 3; ++d)
                deterministic_bulk +=
                    .5 * m * n *
                    (postmean[d] * postmean[d] - mean[d] * mean[d]);
            near(expected_sample_thermal, population + redirect - mean_noise,
                 3.e-15, "sample variance convention");
            near(expected_sample_bulk, deterministic_bulk + mean_noise, 3.e-15,
                 "mean noise partition");
            near(expected_lab, population + redirect + deterministic_bulk,
                 3.e-15, "no double debit of sample mean noise");
        }
        if (test == "invalid") {
            auto bad = views;
            if (!bad.empty())
                bad.push_back(bad.front());
            check(!trial.Refresh(bad), "duplicate rejected collectively");
            check(trial.Refresh(views), "recover after invalid");
            bad = views;
            if (!bad.empty())
                bad.front().position[0] = nullptr;
            check(!trial.Refresh(bad), "null rejected");
            check(trial.Refresh(views), "recover null");
        }
        if (test == "purity") {
            for (auto& pc : input)
                for (PC::ParIterType it(*pc, 0); it.isValid(); ++it) {
                    auto& v = it.GetStructOfArrays().GetRealData(PIdx::r);
                    std::vector<P> host(v.size());
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(),
                                     v.end(), host.begin());
                    for (auto x : host)
                        near(x, .45, 0, "input positions unchanged");
                }
            check(trial.Refresh(views) && moments.Evaluate(trial, ex),
                  "repeat");
            auto repeat = moments.Ledger();
            for (int s = 0; s < 3; ++s)
                for (int c = 0; c < IonExchangeMomentComponent::Count; ++c)
                    near(repeat[s][c], l[s][c], 0, "A/A exact");
        }
        if (test == "drift") {
            std::array<MF, 3> ji, jp, out;
            std::array<const MF*, 3> jip, jpp;
            std::array<MF*, 3> op;
            for (int d = 0; d < 3; ++d) {
                auto type = amrex::IntVect(1);
                if (d != 1)
                    type[d / 2] = 0;
                auto edges = amrex::convert(ba, type);
                ji[d].define(edges, dm, 1, 1);
                jp[d].define(edges, dm, 1, 1);
                out[d].define(nodes, dm, 1, 1);
                ji[d].setVal((d + 1) * 100 * PhysConst::q_e);
                jp[d].setVal((d + 1) * 400 * PhysConst::q_e);
                jip[d] = &ji[d];
                jpp[d] = &jp[d];
                op[d] = &out[d];
            }
            check(BuildExpectedIonDrift(g, rho, nullptr, jip, jpp, 1, op),
                  "pure drift");
            for (int d = 0; d < 3; ++d) {
                near(out[d].min(0), -(d + 1) * 150, 2.e-15,
                     "native Yee/nodal drift");
                near(out[d].max(0), -(d + 1) * 150, 2.e-15, "uniform drift");
            }
            rho.setVal(.5 * PhysConst::q_e);
            check(BuildExpectedIonDrift(g, rho, nullptr, jip, jpp, 1, op),
                  "floor drift");
            for (auto& v : out)
                near(v.norminf(0), 0, 0, "floor stops drift");
            MF pedestal(nodes, dm, 1, 1);
            pedestal.setVal(PhysConst::q_e);
            check(BuildExpectedIonDrift(g, rho, &pedestal, jip, jpp, 1, op),
                  "pedestal drift");
            for (int d = 0; d < 3; ++d)
                near(out[d].min(0), -(d + 1) * 200, 2.e-15,
                     "pedestal capacity convention");
        }
        if (test == "guard") {
            check(IonExchangeDragOverlaps(true, false, "D", {"D"}),
                  "overlap rejected");
            check(!IonExchangeDragOverlaps(false, false, "D", {"D"}),
                  "drag only allowed");
            check(!IonExchangeDragOverlaps(true, true, "alpha", {"alpha"}),
                  "excluded no overlap");
            check(!IonExchangeDragOverlaps(true, false, "T", {"D"}),
                  "other species no overlap");
            R const n = 1.e19, Z = 1, m = 2 * PhysConst::m_p, eta = 1.e-5,
                    vi = 3.e4, ve = 1.e4,
                    nu = Z * PhysConst::q_e * PhysConst::q_e * n * eta / m;
            R const electric =
                Z * PhysConst::q_e * n * eta * (PhysConst::q_e * n * (vi - ve));
            near(electric, m * n * nu * (vi - ve), 2.e-15,
                 "matched eta/nu force");
            R previous = 0;
            for (R dt : {1.e-8, 5.e-9, 2.5e-9}) {
                R const velocity1 =
                    (vi + electric / (m * n) * dt - ve) * std::exp(-nu * dt) +
                    ve;
                R const defect = std::abs(velocity1 - vi);
                if (previous > 0)
                    check(previous / defect > 3.98 && previous / defect < 4.02,
                          "finite split local second-order defect");
                previous = defect;
            }
            check(std::abs(electric - m * n * (2 * nu) * (vi - ve)) >
                      .9 * std::abs(electric),
                  "independent rate mismatch");
        }
        amrex::Print() << "PASS endpoint_exchange " << test << " box=" << box
                       << "\n";
    }
    amrex::Finalize();
}
