/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/MultiDiagnostics.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/DarwinThermalAdvance.H"
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "FieldSolver/ImplicitSolvers/ThetaImplicitHybrid.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_VisMF.H>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
using warpx::fields::FieldType;
using namespace warpx::thermal;
amrex::Real
relative_difference (amrex::MultiFab const& a, amrex::MultiFab const& b) {
    amrex::MultiFab work(a.boxArray(), a.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(work, a, 0, 0, 1, 0);
    // VisMF may select a different rank distribution on read. Match that
    // layout explicitly before the local cellwise comparison, including
    // when one rank owns no boxes.
    amrex::MultiFab reference(a.boxArray(), a.DistributionMap(), 1, 0);
    reference.ParallelCopy(b, 0, 0, 1);
    amrex::MultiFab::Subtract(work, reference, 0, 0, 1, 0);
    return work.norm0() / std::max(a.norm0(), amrex::Real(1.e-200));
}
amrex::Real
integral (amrex::MultiFab const& f, amrex::Geometry const& geometry) {
    auto const dx = geometry.CellSizeArray(), lo = geometry.ProbLoArray();
    auto const index_lo = geometry.Domain().smallEnd();
    bool const rz = geometry.IsRZ();
    amrex::Real volume = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
    amrex::ReduceOps<amrex::ReduceOpSum> op;
    amrex::ReduceData<amrex::Real> data(op);
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a = f.const_array(mfi);
        op.eval(
            mfi.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> decltype(data)::Type {
                auto v = rz ? volume * 6.283185307179586476925 *
                                  (lo[0] + (i - index_lo[0] + .5) * dx[0])
                            : volume;
                return {a(i, j, k) * v};
            });
    }
    auto value = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealSum(value);
    return value;
}
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        int stop = 3, checkpoint = -1;
        bool controller = true, gate = false;
        amrex::Real factor = .25, deadband = 0.;
        std::string reference, configuration = "density-fixture-alpha0.2-v1";
        std::string negative;
        amrex::ParmParse pp("control_test");
        pp.query("stop", stop);
        pp.query("checkpoint", checkpoint);
        pp.query("controller", controller);
        pp.query("factor", factor);
        pp.query("deadband", deadband);
        pp.query("gate", gate);
        pp.query("reference", reference);
        pp.query("configuration", configuration);
        pp.query("negative", negative);
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& model = *w.get_pointer_HybridPICModel();
        if (!model.UsesEulerianElectronEnergy()) {
            auto const rep = model.m_qdsmc_te_n_floor,
                       oldgate = model.m_qdsmc_n_floor;
            model.SetHybridDensityFloor(model.m_n_floor * factor);
            AMREX_ALWAYS_ASSERT(model.m_qdsmc_te_n_floor == rep &&
                                model.m_qdsmc_n_floor == oldgate);
            amrex::Print() << "LEGACY_DENSITY_SETTER_PASS\n";
        } else {
            auto const geometry = model.ElectronThermalGeometry(0);
            auto const& u =
                *w.m_fields.get(FieldType::hybrid_electron_energy_fp, 0);
            auto const& rho = *w.m_fields.get(FieldType::rho_fp, 0);
            auto const& te =
                *w.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
            if (controller) {
                model.ConfigureDensityFloorController(configuration);
                std::string restart;
                amrex::ParmParse amr("amr");
                amr.query("restart", restart);
                if (!restart.empty()) {
                    amrex::Vector<char> saved;
                    amrex::ParallelDescriptor::ReadAndBcastFile(
                        restart + "/density_control.dat", saved);
                    AMREX_ALWAYS_ASSERT(
                        model.DensityControlCheckpointMetadata() ==
                        std::string(saved.dataPtr()));
                    amrex::Print() << "CONTROL_RESTORE exact=1\n";
                }
            }
            // A persistent second native stage exercises the very same cached
            // context lifecycle independently of the application-owned stage.
            // Snapshot/Cancel restore every borrowed field and solver flag.
            auto* theta = dynamic_cast<ThetaImplicitHybrid*>(
                w.get_pointer_ImplicitSolver());
            AMREX_ALWAYS_ASSERT(theta);
            DarwinThermalAdvance context_probe(*theta);
            if (negative == "during_step") {
                model.BeginDensityControlStep();
                model.SetHybridDensityFloor(model.m_n_floor * 2.);
            }
            if (negative == "nonfinite") {
                model.SetHybridDensityFloor(
                    std::numeric_limits<amrex::Real>::quiet_NaN());
            }
            while (w.getistep(0) < stop) {
                w.Evolve(1);
                auto const before_book = model.DensityControlState();
                auto const before = integral(u, geometry);
                amrex::MultiFab oldte(te.boxArray(), te.DistributionMap(), 1,
                                      0);
                amrex::MultiFab oldu(u.boxArray(), u.DistributionMap(), 1, 0);
                amrex::MultiFab::Copy(oldte, te, 0, 0, 1, 0);
                amrex::MultiFab::Copy(oldu, u, 0, 0, 1, 0);
                int const step = w.getistep(0);
                if (controller) {
                    // Prescribed measured-peak observations isolate controller
                    // continuation from the physical evolution being
                    // checkpointed.
                    amrex::Real const samples[4] = {4.e18, .5e18, 3.e18, 1.e18};
                    auto const& c = model.DensityControlState();
                    auto const raw = samples[(step - 1) % 4];
                    auto const ema =
                        c.controller_has_ema
                            ? c.controller_ema + .2 * (raw - c.controller_ema)
                            : raw;
                    auto target = std::max(ema, .05 * c.initial_number_floor);
                    if (std::abs(target - c.controller_last) <=
                        deadband * c.controller_last) {
                        target = c.controller_last;
                    }
                    model.CommitDensityFloorController(
                        configuration, target, ema, true, step,
                        gate ? target * c.initial_gate_floor /
                                   c.initial_number_floor
                             : -1.);
                } else if (step == 1) {
                    model.SetHybridDensityFloor(model.m_n_floor * factor);
                    if (gate) {
                        model.SetQdsmcDensityFloor(model.m_qdsmc_n_floor *
                                                   factor);
                    }
                }
                auto const after = integral(u, geometry);
                auto const& c = model.DensityControlState();
                auto const df =
                    c.floor_inventory_joule - before_book.floor_inventory_joule;
                auto const dp = c.pedestal_inventory_joule -
                                before_book.pedestal_inventory_joule;
                auto const err = relative_difference(te, oldte);
                AMREX_ALWAYS_ASSERT(
                    err < 128 * std::numeric_limits<amrex::Real>::epsilon());
                AMREX_ALWAYS_ASSERT(
                    std::abs((after - before) - df - dp) <
                    128 * std::numeric_limits<amrex::Real>::epsilon() *
                        std::max({1., std::abs(before), std::abs(after)}));
                if (c.number_floor == before_book.number_floor) {
                    AMREX_ALWAYS_ASSERT(relative_difference(u, oldu) == 0. &&
                                        df == 0. && dp == 0.);
                }
                AMREX_ALWAYS_ASSERT(model.m_qdsmc_te_n_floor ==
                                    model.m_n_floor);
                if (!gate) {
                    AMREX_ALWAYS_ASSERT(model.m_qdsmc_n_floor ==
                                        c.initial_gate_floor);
                }
                KineticThermalMoments fresh(geometry, u.boxArray(),
                                            u.DistributionMap(),
                                            model.EulerianMomentOptions());
                KineticThermalStateView state{rho};
                state.energy = &u;
                state.pedestal = model.DensityPedestal(0);
                AMREX_ALWAYS_ASSERT(fresh.Evaluate(state));
                AMREX_ALWAYS_ASSERT(
                    relative_difference(te, fresh.NodalTemperature()) == 0.);
                context_probe.SnapshotStepStart();
                context_probe.BeginStep(w.gett_new(0), w.getdt(0));
                context_probe.InitializePushContext();
                AMREX_ALWAYS_ASSERT(context_probe.SetPushThermodynamics());
                auto const trial_error =
                    relative_difference(model.ElectronTemperatureForSolve(0),
                                        fresh.NodalTemperature());
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    trial_error == 0.,
                    "Native step context retained a stale density floor");
                context_probe.Cancel();
                amrex::Print()
                    << std::setprecision(17) << "CONTROL_RUNTIME step=" << step
                    << " epoch=" << c.epoch << " T_change=" << err
                    << " trial_T_error=" << trial_error
                    << " floor=" << c.number_floor << " gate=" << c.gate_floor
                    << " U_J=" << after << " delta_U=" << after - before
                    << " floor_J=" << df << " pedestal_J=" << dp << "\n";
                if (step == checkpoint) {
                    w.GetMultiDiags().NewIteration();
                    w.GetMultiDiags().FilterComputePackFlush(step - 1, true);
                }
            }
            for (auto const& entry :
                 std::vector<std::pair<FieldType, std::string>>{
                     {FieldType::hybrid_electron_energy_fp, "U"},
                     {FieldType::hybrid_electron_temperature_fp, "Te"},
                     {FieldType::hybrid_electron_pressure_fp, "Pe"},
                     {FieldType::rho_fp, "rho"}}) {
                auto const& field = *w.m_fields.get(entry.first, 0);
                if (!reference.empty()) {
                    amrex::MultiFab expected;
                    amrex::VisMF::Read(expected,
                                       reference + "." + entry.second);
                    auto const error = relative_difference(field, expected);
                    amrex::Print() << "CONTROL_REPLAY " << entry.second
                                   << " relative=" << error << "\n";
                    AMREX_ALWAYS_ASSERT(
                        error <
                        256 * std::numeric_limits<amrex::Real>::epsilon());
                }
                amrex::VisMF::Write(field, "final." + entry.second);
            }
            auto const encoded = model.DensityControlCheckpointMetadata();
            if (!reference.empty()) {
                amrex::Vector<char> text;
                amrex::ParallelDescriptor::ReadAndBcastFile(
                    reference + ".control", text);
                auto expected =
                    ElectronDensityControl::Decode(std::string(text.dataPtr()));
                auto actual = ElectronDensityControl::Decode(encoded);
                // Restart can reorder deposited particles at roundoff. The
                // physical inventory accumulated after restart follows those
                // fields; the controller itself must still continue exactly.
                auto const scale =
                    std::max({amrex::Real(1.), std::abs(integral(u, geometry)),
                              std::abs(expected.floor_inventory_joule),
                              std::abs(expected.pedestal_inventory_joule)});
                auto const floor_error =
                    std::abs(actual.floor_inventory_joule -
                             expected.floor_inventory_joule);
                auto const pedestal_error =
                    std::abs(actual.pedestal_inventory_joule -
                             expected.pedestal_inventory_joule);
                AMREX_ALWAYS_ASSERT(
                    std::max(floor_error, pedestal_error) <
                    64 * std::numeric_limits<amrex::Real>::epsilon() * scale);
                actual.floor_inventory_joule = expected.floor_inventory_joule;
                actual.pedestal_inventory_joule =
                    expected.pedestal_inventory_joule;
                AMREX_ALWAYS_ASSERT(actual.Encode() == expected.Encode());
                amrex::Print()
                    << "CONTROL_REPLAY controller_exact=1 floor_J_error="
                    << floor_error << " pedestal_J_error=" << pedestal_error
                    << "\n";
            }
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream out("final.control");
                out << encoded;
            }
        }
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
