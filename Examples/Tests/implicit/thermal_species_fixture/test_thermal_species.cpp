/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/KineticThermalSpeciesDeposition.H"
#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/Deposition/CurrentDeposition.H"
#include "Utils/WarpXConst.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <array>
#include <cmath>
#include <limits>
using namespace warpx::thermal;
using amrex::Real;
using MF = amrex::MultiFab;
using Device = amrex::Gpu::DeviceVector<amrex::ParticleReal>;
using Vector = std::array<std::unique_ptr<MF>, 3>;
void
require (bool x, const char* text) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(x, text);
}
Real
difference (const MF& a, const MF& b, int ng = 0) {
    MF delta(a.boxArray(), a.DistributionMap(), 1, ng);
    MF::LinComb(delta, 1, a, 0, -1, b, 0, 0, 1, ng);
    return delta.norminf(0, ng);
}
MF
copy (const MF& a) {
    MF b(a.boxArray(), a.DistributionMap(), a.nComp(), a.nGrowVect());
    MF::Copy(b, a, 0, 0, a.nComp(), a.nGrowVect());
    return b;
}
Real
value (const MF& f, int i, int j) {
    Real result = -1e100;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(amrex::IntVect(i, j))) {
            result = f[mfi](amrex::IntVect(i, j));
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(result);
    return result;
}
struct Fixture {
    amrex::Geometry g;
    amrex::BoxArray cells, nodes;
    amrex::DistributionMapping dm;
    int order = 3;
    bool direct = false;
    amrex::Parser zero, eta{"2.15"}, nu{"1"};
    Fixture (int maxbox, int shape, bool overlay = false,
             bool use_direct = false)
        : order(shape), direct(use_direct), zero(overlay ? "0.25*Js" : "0") {
        amrex::Box domain(amrex::IntVect(0, 0), amrex::IntVect(15, 15));
        amrex::RealBox real({0., 0.}, {1.6, 1.6});
        int periodic[2] = {0, 1};
        g.define(domain, &real, 1, periodic);
        cells.define(domain);
        cells.maxSize(maxbox);
        nodes = amrex::convert(cells, amrex::IntVect(1));
        dm = amrex::DistributionMapping(cells);
        zero.registerVariables({"rs", "rho", "Te", "J", "Js", "B", "t"});
        eta.registerVariables({"rho", "J", "t"});
        nu.registerVariables({"rho", "Te", "Ti", "t"});
    }
    std::unique_ptr<KineticThermalSpecies>
    context (bool relaxation = true) {
        std::vector<KineticSpeciesDescriptor> d(3);
        for (int s = 0; s < 3; ++s) {
            d[s].name = std::array<std::string, 3>{"D", "T", "alpha"}[s];
            d[s].charge_number = s == 2 ? 2 : 1;
            d[s].mass = (s + 2) * PhysConst::m_p;
            d[s].relaxation_excluded = s == 2;
            d[s].has_resistivity_overlay = true;
            d[s].resistivity_overlay = zero.compile<7>();
        }
        KineticSpeciesOptions o;
        o.relaxation = relaxation;
        return std::make_unique<KineticThermalSpecies>(g, cells, dm,
                                                       amrex::IntVect(4), d, o);
    }
    MF
    node (int comps = 1, int ng = 0) {
        return MF(nodes, dm, comps, ng);
    }
    Vector
    vector (Real initial = 0) {
        Vector out;
        for (int c = 0; c < 3; ++c) {
            amrex::IntVect type(1);
            if (c == 0)
                type[0] = 0;
            if (c == 2)
                type[1] = 0;
            out[c] =
                std::make_unique<MF>(amrex::convert(cells, type), dm, 1, 4);
            out[c]->setVal(initial + c);
        }
        return out;
    }
    // Native particle cloud; old positions/momenta immutable. Delta parameter
    // materializes different trial species responses without aggregate guesses.
    template <class F>
    void
    cloud (std::size_t species, Real delta, bool old, F&& f) {
        // Coincident pairs with opposite momenta give an analytic old variance.
        std::vector<std::array<Real, 3>> points = {{{.72, .76, 1.e11}},
                                                   {{.035, .035, 2.e10}},
                                                   {{.395, 1.565, 6.e10}},
                                                   {{1.565, .405, 4.e10}},
                                                   {{1.21, .79, 7.e10}}};
        for (amrex::MFIter mfi(cells, dm); mfi.isValid(); ++mfi) {
            std::vector<amrex::ParticleReal> x0, z0, xm, zm, theta, w, ux, uy,
                uz, unx, uny, unz, y;
            for (auto const& p : points) {
                Real r = p[0], z = p[1];
                // Ownership remains the original tile through a trial push.
                if (!mfi.validbox().contains(
                        amrex::IntVect(int(r / .1), int(z / .1))))
                    continue;
                for (int sign : {-1, 1}) {
                    x0.push_back(r);
                    z0.push_back(z);
                    y.push_back(0);
                    theta.push_back(0);
                    xm.push_back(r + (old ? 0 : delta * .013));
                    zm.push_back(z + (old ? 0 : delta * .019));
                    w.push_back(p[2] * (species == 2 ? .5 : 1));
                    unx.push_back(sign * 1.e3);
                    uny.push_back(sign * 2.e3);
                    unz.push_back(sign * 3.e3);
                    ux.push_back(unx.back() + (old ? 0 : delta * 2e2));
                    uy.push_back(uny.back() + (old ? 0 : delta * 1e2));
                    uz.push_back(unz.back() + (old ? 0 : delta * 3e2));
                }
            }
            if (w.empty())
                continue;
            auto device = [] (auto const& x) {
                Device d(x.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, x.begin(), x.end(),
                                 d.begin());
                return d;
            };
            auto X0 = device(x0), Z0 = device(z0), XM = device(xm),
                 ZM = device(zm), TH = device(theta), W = device(w),
                 Y = device(y);
            auto UX = device(ux), UY = device(uy), UZ = device(uz),
                 UNX = device(unx), UNY = device(uny), UNZ = device(unz);
            GetParticlePosition<PIdx> pos;
            pos.m_x = XM.data();
            pos.m_y = Y.data();
            pos.m_z = ZM.data();
            pos.m_theta = TH.data();
            f(mfi, pos, W, X0, Z0, Y, UX, UY, UZ, UNX, UNY, UNZ);
            amrex::Gpu::streamSynchronize();
        }
    }
    bool
    old_deposit (std::size_t s, SpeciesVariancePass pass,
                 const SpeciesVarianceFields& f) {
        cloud(s, 0, true,
              [&] (auto const& mfi, auto const& pos, auto const& w, auto const&,
                   auto const&, auto const&, auto const& ux, auto const& uy,
                   auto const& uz, auto const&, auto const&, auto const&) {
                  auto deposit = [&]<int N>() {
                      DepositSpeciesTemperatureShapeN<N>(
                          f, mfi, pos, w.data(), ux.data(), uy.data(),
                          uz.data(), w.size(), {10., 1., 10.}, {0., 0., 0.},
                          {0, 0, 0}, pass);
                  };
                  if (order == 1)
                      deposit.template operator()<1>();
                  else
                      deposit.template operator()<3>();
              });
        return true;
    }
    bool
    trial (std::size_t s, Real delta, MF& rho, const SpeciesMutableVector& j,
           MF* endpoint_average = nullptr) {
        cloud(s, delta, false,
              [&] (auto const& mfi, auto const& pos, auto const& w,
                   auto const& x0, auto const& z0, auto const& y,
                   auto const& ux, auto const& uy, auto const& uz,
                   auto const& unx, auto const& uny, auto const& unz) {
                  Real q = PhysConst::q_e * (s == 2 ? 2 : 1);
                  auto lower = amrex::lbound(amrex::grow(mfi.validbox(), 4));
                  amrex::XDim3 origin{lower.x * .1, 0., lower.y * .1};
                  amrex::GpuArray<amrex::GpuArray<double, 2>, 2> domain{
                      {{double(-lower.x), double(16 - lower.x)},
                       {double(-lower.y), double(16 - lower.y)}}};
                  amrex::GpuArray<amrex::GpuArray<bool, 2>, 2> crop{
                      {{false, false}, {false, false}}};
                  auto deposit = [&]<int N>() {
                      doChargeDepositionShapeN<N>(
                          pos, w.data(), nullptr, rho[mfi], w.size(),
                          {10., 1., 10.}, origin, lower, q, 1);
                      if (j[0] && !direct)
                          doChargeConservingDepositionShapeNImplicit<N>(
                              x0.data(), y.data(), z0.data(), pos, w.data(),
                              unx.data(), uny.data(), unz.data(), ux.data(),
                              uy.data(), uz.data(), nullptr, j[0]->array(mfi),
                              j[1]->array(mfi), j[2]->array(mfi), w.size(),
                              1.e-4, {10., 1., 10.}, origin, domain, crop,
                              lower, q, 1);
                      if (j[0] && direct) {
                          doDepositionShapeNImplicit<N>(
                              pos, w.data(), unx.data(), uny.data(), unz.data(),
                              ux.data(), uy.data(), uz.data(), nullptr,
                              (*j[0])[mfi], (*j[1])[mfi], (*j[2])[mfi],
                              w.size(), {10., 1., 10.}, origin, lower, q, 1);
                      }
                      if (endpoint_average) {
                          auto old = pos;
                          old.m_x = x0.data();
                          old.m_z = z0.data();
                          Device xend(w.size()), zend(w.size());
                          auto xe = xend.data(), ze = zend.data();
                          auto xp = x0.data(), zp = z0.data();
                          amrex::ParallelFor(w.size(),
                                             [=] AMREX_GPU_DEVICE(int p) {
                                                 amrex::ParticleReal r, a, z;
                                                 pos(p, r, a, z);
                                                 xe[p] = 2 * r - xp[p];
                                                 ze[p] = 2 * z - zp[p];
                                             });
                          auto end = pos;
                          end.m_x = xe;
                          end.m_z = ze;
                          doChargeDepositionShapeN<N>(old, w.data(), nullptr,
                                                      (*endpoint_average)[mfi],
                                                      w.size(), {10., 1., 10.},
                                                      origin, lower, .5 * q, 1);
                          doChargeDepositionShapeN<N>(end, w.data(), nullptr,
                                                      (*endpoint_average)[mfi],
                                                      w.size(), {10., 1., 10.},
                                                      origin, lower, .5 * q, 1);
                          amrex::Gpu::streamSynchronize();
                      }
                  };
                  if (order == 1)
                      deposit.template operator()<1>();
                  else
                      deposit.template operator()<3>();
              });
        return true;
    }
    void
    freeze (KineticThermalSpecies& k) {
        require(k.FreezeOldTemperature([&] (auto s, auto p, auto const& f) {
            return old_deposit(s, p, f);
        }),
                "native old Ti");
    }
    void
    refresh (KineticThermalSpecies& k, Real amplitude = 1, bool swap = false) {
        require(k.RefreshMaterializedTrial(
                    MaterializedSpeciesTrial::FullParticleState,
                    [&] (auto s, auto& rho, auto const& j) {
                        Real direction = s == 2 ? 0 : (s == 0 ? 1 : -1);
                        if (swap)
                            direction = -direction;
                        return trial(s, amplitude * direction, rho, j);
                    }),
                "native trial");
    }
    MF
    sources (KineticThermalSpecies& k, bool relaxation = true) {
        ThermalSourceOptions o;
        o.density_floor = 1.e16;
        o.joule = true;
        o.relaxation = relaxation;
        o.field_eta = eta.compile<3>();
        o.relaxation_rate = nu.compile<4>();
        auto rho = node(), te = node(), rates = node(SourceComponent::Count),
             ions = node(12);
        rho.setVal(2 * PhysConst::q_e * o.density_floor);
        te.setVal(15 * PhysConst::q_e / PhysConst::kb);
        auto j = vector(1), b = vector(0);
        ThermalSourceState state;
        state.raw_charge = &rho;
        state.temperature_kelvin = &te;
        for (int c = 0; c < 3; ++c) {
            state.plasma_current[c] = j[c].get();
            state.magnetic_field[c] = b[c].get();
        }
        state.species = k.Species();
        EulerianThermalSources source(g, rho, o);
        require(source.Evaluate(state, rates, ions),
                "native context/source composition");
        require(ions.norminf(2 * IonSourceComponent::Count +
                             IonSourceComponent::RelaxationPower) == 0,
                "alpha excluded from relaxation only");
        return rates;
    }
};
void
run (Fixture& f, const std::string& name) {
    auto k = f.context();
    f.freeze(*k);
    f.refresh(*k);
    if (name == "native" || name == "direct") {
        for (std::size_t s = 0; s < 3; ++s) {
            auto rho = f.node(1, 4), average = f.node(1, 4);
            rho.setVal(0);
            average.setVal(0);
            auto j = f.vector();
            for (auto& v : j)
                v->setVal(0);
            SpeciesMutableVector ptr{j[0].get(), j[1].get(), j[2].get()};
            f.trial(s, s == 2 ? 0 : (s == 0 ? 1 : -1), rho, ptr, &average);
            for (auto* p : {&rho, &average, j[0].get(), j[1].get(), j[2].get()})
                p->SumBoundary(0, 1, p->nGrowVect(), p->nGrowVect(),
                               f.g.periodicity());
            require(difference(rho, *k->Species()[s].raw_charge, 4) < 1e-18,
                    "raw native charge parity");
            for (int c = 0; c < 3; ++c)
                require(difference(*j[c], *k->Species()[s].raw_current[c], 4) <
                            1e-18,
                        "implicit native current parity");
            if (f.order == 3 && s < 2)
                require(difference(rho, average) > 1e-9,
                        "midpoint raw charge differs intentionally from "
                        "endpoint average");
        }
    } else if (name == "temperature") {
        Real variance = (1e6 + 4e6 + 9e6) / (1 + 14e6 * PhysConst::inv_c2);
        Real expected = 2 * variance / 3 * 2 * PhysConst::m_p / PhysConst::q_e;
        Real measured = value(k->OldCellTemperature(0), 7, 7);
        require(std::abs(measured - expected) < 1e-12 * expected,
                "native double-pass trace temperature and eV units");
        auto frozen = copy(k->OldNodalTemperature(0));
        f.refresh(*k, 2);
        require(difference(frozen, k->OldNodalTemperature(0)) == 0,
                "old Ti frozen across trial momentum changes");
        require(k->Species()[2].ion_temperature_ev == nullptr,
                "excluded alpha has no Ti requirement");
    } else if (name == "source") {
        auto q = f.sources(*k);
        auto sum = f.node();
        sum.setVal(0);
        for (auto const& s : k->Species())
            MF::Add(sum, *s.raw_charge, 0, 0, 1, 0);
        auto diff = f.node();
        Real floor = PhysConst::q_e * 1e16;
        for (amrex::MFIter mfi(diff); mfi.isValid(); ++mfi) {
            auto out = diff.array(mfi);
            auto total = sum.const_array(mfi);
            auto power = q.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int l) {
                    out(i, j, l) = power(i, j, l, SourceComponent::Joule) -
                                   2.15 * 14 * total(i, j, l) /
                                       amrex::max(total(i, j, l), floor);
                });
        }
        require(diff.norminf() < 1e-12,
                "all charged raw density fractions including alpha and "
                "denominator floor");
        require(q.norminf(SourceComponent::Joule) > 0,
                "nonzero Joule transfer");
    } else if (name == "derivative" || name == "overlay_derivative") {
        auto derivative = [&] (Real h) {
            f.refresh(*k, 1 + h);
            auto plus = f.sources(*k);
            f.refresh(*k, 1 - h);
            auto minus = f.sources(*k);
            auto d = f.node();
            MF::LinComb(d, 1 / (2 * h), plus, SourceComponent::ElectronTotal,
                        -1 / (2 * h), minus, SourceComponent::ElectronTotal, 0,
                        1, 0);
            return d;
        };
        auto d1 = derivative(1e-3), d2 = derivative(5e-4);
        Real norm = d2.norminf();
        require(norm > 1e-5, "source observes live species field response");
        Real error = difference(d1, d2) / norm;
        require(error < 2e-5,
                "native species source directional derivative convergence");
        amrex::Print() << "SPECIES_DERIVATIVE error=" << error
                       << " norm=" << norm << "\n";
    } else if (name == "aggregate") {
        auto a0 = copy(*k->Species()[0].raw_charge),
             a1 = copy(*k->Species()[1].raw_charge);
        auto total = copy(a0);
        MF::Add(total, a1, 0, 0, 1, 4);
        auto current = f.vector();
        for (int c = 0; c < 3; ++c)
            MF::LinComb(*current[c], 1, *k->Species()[0].raw_current[c], 0, 1,
                        *k->Species()[1].raw_current[c], 0, 0, 1, 4);
        auto before = f.sources(*k);
        f.refresh(*k, 1, true);
        auto after = f.sources(*k);
        auto next = copy(*k->Species()[0].raw_charge);
        MF::Add(next, *k->Species()[1].raw_charge, 0, 0, 1, 4);
        require(difference(next, total, 4) < 1e-18,
                "aggregate rho identical after species response swap");
        for (int c = 0; c < 3; ++c) {
            auto sum = copy(*k->Species()[0].raw_current[c]);
            MF::Add(sum, *k->Species()[1].raw_current[c], 0, 0, 1, 4);
            require(difference(sum, *current[c], 4) < 1e-18,
                    "aggregate J identical after species response swap");
        }
        require(difference(a0, *k->Species()[0].raw_charge) > 1e-9,
                "individual charge response differs");
        auto delta = f.node();
        MF::LinComb(delta, 1, before, SourceComponent::Relaxation, -1, after,
                    SourceComponent::Relaxation, 0, 1, 0);
        require(delta.norminf() > 1e-9,
                "aggregate moments cannot identify species relaxation source");
        amrex::Print() << "SPECIES_AGGREGATE_SOURCE_DIFFERENCE "
                       << delta.norminf() << "\n";
    } else if (name == "purity") {
        auto a = copy(*k->Species()[0].raw_charge),
             j = copy(*k->Species()[0].raw_current[2]),
             ti = copy(k->OldNodalTemperature(0));
        auto qa = f.sources(*k);
        f.refresh(*k, 2);
        f.sources(*k);
        f.refresh(*k, 1);
        auto again = f.sources(*k);
        require(difference(a, *k->Species()[0].raw_charge, 4) == 0 &&
                    difference(j, *k->Species()[0].raw_current[2], 4) == 0,
                "native A B A deposits");
        require(difference(ti, k->OldNodalTemperature(0)) == 0 &&
                    difference(qa, again) == 0,
                "fixed old Ti and source A B A");
    } else if (name == "invalid") {
        bool ok = k->RefreshMaterializedTrial(
            MaterializedSpeciesTrial::FullParticleState,
            [&] (auto, auto& rho, auto const&) {
                rho.setVal(std::numeric_limits<Real>::quiet_NaN());
                return true;
            });
        require(!ok, "nonfinite deposit rejects");
        f.refresh(*k);
        f.sources(*k);
        require(!k->FreezeOldTemperature(
                    [] (auto, auto, auto const&) { return false; }),
                "failed temperature provider rejects");
        f.freeze(*k);
        f.refresh(*k);
        f.sources(*k);
    } else
        require(false, "unknown case");
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int box = 4, shape = 3;
        pp.query("max_grid_size", box);
        pp.query("shape", shape);
        std::string name = "native", reject;
        pp.query("case", name);
        pp.query("reject", reject);
        Fixture f(box, shape, name == "overlay_derivative", name == "direct");
        if (!reject.empty()) {
            auto k = f.context();
            if (reject == "mm")
                k->RefreshMaterializedTrial(
                    MaterializedSpeciesTrial::AggregateMassMatrixOnly,
                    [] (auto, auto&, auto const&) { return true; });
            if (reject == "old")
                k->RefreshMaterializedTrial(
                    MaterializedSpeciesTrial::FullParticleState,
                    [] (auto, auto&, auto const&) { return true; });
            auto d = k->Descriptors();
            KineticSpeciesOptions o;
            if (reject == "descriptor")
                d[1].name = d[0].name;
            if (reject == "eb")
                o.embedded_boundary = true;
            KineticThermalSpecies bad(f.g, f.cells, f.dm, amrex::IntVect(4), d,
                                      o);
            require(false, "rejection failed");
        }
        run(f, name);
        amrex::Print() << "SPECIES_PASS case=" << name << " box=" << box
                       << " shape=" << shape << "\n";
    }
    amrex::Finalize();
}
