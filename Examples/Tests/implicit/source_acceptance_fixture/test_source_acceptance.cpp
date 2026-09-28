/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ThermalSourceAcceptance.H"
#include "Utils/WarpXConst.H"
#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <cmath>
#include <limits>
#include <string>

using namespace warpx::thermal;
using amrex::Real;
void
run (const std::string& name, int box) {
    bool const orphan = name == "orphan_viscous" || name == "orphan_hyper" ||
                        name == "cancellation";
    bool const cancellation = name == "cancellation";
    bool const cooling = name == "cooling";
    bool const hyper = name == "orphan_hyper";
    amrex::Box domain(amrex::IntVect(0), amrex::IntVect(7));
    amrex::RealBox rb({0., 0.}, {1., 1.});
    int periodic[2]{};
    amrex::Geometry g(domain, &rb, 1, periodic);
    amrex::BoxArray cells(domain);
    cells.maxSize(box);
    amrex::DistributionMapping dm(cells);
    auto nodes = amrex::convert(cells, amrex::IntVect(1));
    amrex::MultiFab rho(nodes, dm, 1, 2), te(nodes, dm, 1, 2),
        cap(nodes, dm, 1, 2);
    Real constexpr nfloor = 1.e17;
    rho.setVal((orphan ? 1 : 3) * PhysConst::q_e * nfloor);
    cap.setVal(4 * PhysConst::q_e * nfloor);
    te.setVal(1.e5);
    std::array<amrex::MultiFab, 3> current, electric;
    ThermalSourceState state;
    state.raw_charge = &rho;
    state.temperature_kelvin = &te;
    state.heat_capacity_charge = &cap;
    for (int c = 0; c < 3; ++c) {
        amrex::IntVect type(1);
        if (c == 0) {
            type[0] = 0;
        }
        if (c == 2) {
            type[1] = 0;
        }
        auto edges = amrex::convert(cells, type);
        current[c].define(edges, dm, 1, 2);
        electric[c].define(edges, dm, 1, 2);
        current[c].setVal(c == 1 ? 1. : 0.);
        for (amrex::MFIter it(electric[c]); it.isValid(); ++it) {
            auto e = electric[c].array(it);
            amrex::ParallelFor(it.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                 int k) {
                Real value = cooling ? -1. : 1.;
                if (cancellation) {
                    value =
                        i == 3 && j == 2 ? 1. : (i == 3 && j == 6 ? -1. : 0.);
                }
                e(i, j, k) = c == 1 ? value : 0.;
            });
        }
        state.plasma_current[c] = &current[c];
        state.magnetic_field[c] = &current[c];
        state.viscous_electric[c] = &electric[c];
        state.hyper_electric[c] = &electric[c];
    }
    ThermalSourceOptions options;
    options.density_floor = nfloor;
    options.physical_dt = .125;
    options.viscosity =
        hyper ? ViscousSourceKind::None : ViscousSourceKind::AppliedWork;
    options.hyperresistive_work = hyper;
    EulerianThermalSources source(g, rho, options);
    amrex::MultiFab rates(nodes, dm, SourceComponent::Count, 2),
        ions(nodes, dm, 1, 2);
    AMREX_ALWAYS_ASSERT(source.Evaluate(state, rates, ions));
    auto ledger = source.Ledger(rates, ions, 0);
    auto const before = ledger.energy;
    amrex::ResetRandomSeed(711);
    auto const random = amrex::Random();
    amrex::ResetRandomSeed(711);
    auto const status = InspectThermalSourceAcceptance(ledger);
    AMREX_ALWAYS_ASSERT(random == amrex::Random() && ledger.energy == before);
    int const signed_channel = hyper ? SourceComponent::DeclinedHyperResistive
                                     : SourceComponent::DeclinedViscousWork;
    int const absolute_channel =
        hyper ? SourceComponent::UnassignableHyperResistiveAbs
              : SourceComponent::UnassignableViscousWorkAbs;
    if (orphan) {
        AMREX_ALWAYS_ASSERT(
            status == ThermalSourceAcceptanceStatus::UnassignableAppliedWork);
        AMREX_ALWAYS_ASSERT(ledger.energy[absolute_channel] > 0);
        AMREX_ALWAYS_ASSERT(ledger.energy[SourceComponent::ElectronTotal] == 0);
        if (cancellation) {
            // Actual opposing physical edge work, not a synthetic edited
            // ledger.
            AMREX_ALWAYS_ASSERT(std::abs(ledger.energy[signed_channel]) <=
                                1.e-14 * ledger.energy[absolute_channel]);
        }
    } else {
        AMREX_ALWAYS_ASSERT(status == ThermalSourceAcceptanceStatus::Ready);
        Real const expected =
            (cooling ? -1. : 1.) * MathConst::pi * options.physical_dt;
        AMREX_ALWAYS_ASSERT(
            std::abs(ledger.energy[SourceComponent::ElectronTotal] - expected) <
            2.e-14 * std::abs(expected));
    }
    if (name == "invalid") {
        ledger.energy[SourceComponent::UnassignableViscousWorkAbs] = -1;
        AMREX_ALWAYS_ASSERT(InspectThermalSourceAcceptance(ledger) ==
                            ThermalSourceAcceptanceStatus::InvalidAbsoluteWork);
        ledger.energy = before;
        ledger.ion_relaxation.push_back(std::numeric_limits<Real>::quiet_NaN());
        AMREX_ALWAYS_ASSERT(InspectThermalSourceAcceptance(ledger) ==
                            ThermalSourceAcceptanceStatus::NonfiniteLedger);
    }
    amrex::Print() << "PASS source acceptance case=" << name
                   << " status=" << static_cast<int>(status)
                   << " signed=" << before[signed_channel]
                   << " absolute=" << before[absolute_channel]
                   << " delivered=" << before[SourceComponent::ElectronTotal]
                   << "\n";
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        std::string name = "smooth";
        int box = 4;
        pp.query("case", name);
        pp.query("box", box);
        run(name, box);
    }
    amrex::Finalize();
}
