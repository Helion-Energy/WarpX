/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Circuit/CircuitCoupler.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "ReversibleRL.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <array>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace {
void
check (bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
using Segments = std::vector<std::array<double, 4>>;
Segments
segments (const ExternalVectorPotential& ext) {
    Segments out(ext.nFields());
    for (int f = 0; f < ext.nFields(); ++f)
        check(
            ext.GetScaleSegment(f, out[f][0], out[f][1], out[f][2], out[f][3]),
            "expected segment");
    return out;
}
void
parameters () {
    amrex::ParmParse external("external_vector_potential");
    external.addarr("fields", std::vector<std::string>{"c0", "c1", "passive"});
    for (auto name : {"c0", "c1", "passive"})
        external.add(std::string(name) + ".python_scale", 1);
    amrex::ParmParse pp("circuit");
    pp.addarr("coils", std::vector<std::string>{"c0", "c1"});
    for (auto name : {"c0", "c1"}) {
        pp.add(std::string(name) + ".r", .25);
        pp.add(std::string(name) + ".z", 0.);
    }
    pp.add("c0.I_ref", 2.);
    pp.add("c1.I_ref", -3.);
}
struct Fixture {
    ExternalVectorPotential external;
    warpx::circuit::CoilSet coils;
    ReversibleRL* engine = nullptr;
    std::unique_ptr<CircuitCoupler> coupler;
    explicit Fixture (const std::string& seed) {
        coils.ReadParameters();
        auto plugin = std::make_unique<ReversibleRL>();
        engine = plugin.get();
        plugin->Define({"c0", "c1"}, {2., -3.}, "");
        CircuitCoupler::Params params;
        params.eps_lowpass_tau = .2;
        params.linkage_reference_accepted = true;
        coupler = std::make_unique<CircuitCoupler>(
            coils,
            std::vector<warpx::circuit::ProbeKind>{
                warpx::circuit::ProbeKind::disk,
                warpx::circuit::ProbeKind::disk},
            std::vector<double>{0., 0.}, params, std::move(plugin));
        coupler->ReadMemoryCheckpoint(
            seed); // fixed initial fixture, never a cancellation mechanism
        external.SetScale("c0", .25, .25, -1., 0.);
        external.SetScale("c1", 1. / 12., 1. / 12., -1., 0.);
        external.SetScale("passive", .4, .7, -2., -1.);
    }
    void
    attach () {
        check(coupler->SetRejectionCapability(ReversibleRL::API()),
              "attach failed");
    }
    void
    begin (double dt = .1) {
        check(coupler->SnapshotNativeStep(external), "snapshot failed");
        coupler->BeginStepMeasured(.9, dt);
    }
    void
    trial (double dt = .1) {
        coupler->EvaluateInterval(.9, .9 + dt, false, .5 * dt);
    }
    void
    accept (double dt = .1) {
        coupler->EvaluateInterval(.9, .9 + dt, true, .5 * dt);
        check(!coupler->CancelNativeStep(),
              "accepted state claimed reversible");
        coupler->FinishStep();
    }
};
void
absent_library (const std::string& path) {
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    check(library, "cannot open native plugin");
    auto version = reinterpret_cast<warpx_external_circuit_abi_version_t>(
        dlsym(library, "warpx_external_circuit_abi_version"));
    check(version && version() == WARPX_EXTERNAL_CIRCUIT_ABI_VERSION,
          "not native ABI2 plugin");
    check(!dlsym(library, "warpx_external_circuit_rejection_api_v1"),
          "reference plugin unexpectedly supports cancel");
    dlclose(library);
}
void
run (const std::string& mode, const std::string& seed) {
    Fixture attempted(seed), clean(seed);
    auto& f = attempted;
    auto entry = f.engine->state;
    auto scales = segments(f.external);
    if (mode == "absent") {
        check(!f.coupler->SupportsNativeRejection() &&
                  !f.coupler->SnapshotNativeStep(f.external),
              "missing capability accepted");
        check(f.engine->snapshots == 0 && f.engine->state == entry &&
                  segments(f.external) == scales,
              "absent mutated state");
        return;
    }
    f.attach();
    clean.attach();
    if (mode == "bad_api") {
        auto api = *ReversibleRL::API();
        api.api_version = 99;
        check(!f.coupler->SetRejectionCapability(&api), "bad API accepted");
        api = *ReversibleRL::API();
        api.cancel = nullptr;
        check(!f.coupler->SetRejectionCapability(&api),
              "missing callback accepted");
        return;
    }
    if (mode == "snapshot_failure") {
        f.engine->fail_snapshot = true;
        check(!f.coupler->SnapshotNativeStep(f.external),
              "failed snapshot accepted");
        check(f.engine->state == entry && segments(f.external) == scales &&
                  !f.coupler->NativeStepCancelable(),
              "snapshot failure mutated input");
        return;
    }
    if (mode == "active_view") {
        std::array<double, 3> start{1., 2., 3.}, end{3., 2., 1.};
        f.external.SetDeviceScaleSegments(start.data(), end.data(), .9, .1,
                                          {0, 1});
        check(!f.coupler->SnapshotNativeStep(f.external) &&
                  f.engine->snapshots == 0,
              "snapshot read active view");
        f.external.ClearDeviceScaleSegments();
        return;
    }
    bool phase = mode == "phase";
    f.engine->mutate_phase = phase;
    clean.engine->mutate_phase = phase;
    double rejected_dt = (mode == "resize" || phase) ? .4 : .1;
    f.begin(rejected_dt);
    check(!f.coupler->SnapshotNativeStep(f.external),
          "second snapshot accepted");
    check(!f.coupler->SetRejectionCapability(nullptr),
          "changed API during attempt");
    if (mode == "finish_unaccepted") {
        f.coupler->FinishStep();
        throw std::runtime_error("missing FinishStep refusal");
    }
    if (phase)
        check(f.engine->state.phase == 1 &&
                  f.engine->state.accepted != entry.accepted,
              "phase mutation not exercised");
    f.trial(rejected_dt);
    auto one = segments(f.external);
    f.trial(rejected_dt);
    check(segments(f.external) == one, "native false replay not deterministic");
    if (mode == "cancel_failure") {
        f.engine->fail_cancel = true;
        check(!f.coupler->CancelNativeStep() &&
                  f.coupler->NativeRejectionFailed(),
              "failed cancellation not terminal");
        check(!f.coupler->SnapshotNativeStep(f.external),
              "rearmed poisoned cancellation");
        return;
    }
    if (mode == "accept_failure") {
        f.engine->throw_accept = true;
        bool caught = false;
        try {
            f.accept(rejected_dt);
        } catch (const std::runtime_error&) {
            caught = true;
        }
        check(caught && !f.coupler->CancelNativeStep() &&
                  f.engine->Token() == 0,
              "accept failure rollback barrier missing");
        return;
    }
    if (mode == "token") {
        char error[64]{};
        check(ReversibleRL::API()->cancel(
                  static_cast<ExternalCircuit*>(f.engine),
                  f.engine->Token() + 1, error,
                  sizeof(error)) == WARPX_REJECTION_BAD_REQUEST,
              "wrong token accepted");
    }
    std::array<double, 3> start{1., 2., 3.}, end{3., 2., 1.};
    if (mode == "device_view")
        f.external.SetDeviceScaleSegments(start.data(), end.data(), .9,
                                          rejected_dt, {0, 1});
    f.external.SetScale("passive", 9., 11., .9, 1.3);
    check(f.coupler->CancelNativeStep(), "cancel failed");
    check(f.engine->state == entry, "pre-BeginStep engine state not restored");
    check(!f.external.DeviceScales().start && !f.external.DeviceScales().end &&
              !f.coupler->DarwinDeviceStepReady(),
          "device view/readiness retained");
    check(segments(f.external) == scales, "exact segments not restored");
    check(f.coupler->CurrentInterval().t0 == 0. &&
              f.coupler->CurrentInterval().t1 == 0. &&
              f.coupler->CurrentInterval().iteration == 0,
          "coupler metadata not restored");
    check(!f.coupler->CancelNativeStep(), "second cancel succeeded");
    f.begin();
    clean.begin();
    f.trial();
    clean.trial();
    f.accept();
    clean.accept();
    check(f.engine->state == clean.engine->state &&
              segments(f.external) == segments(clean.external),
          "retry differs from clean native coupling");
    check(f.engine->state.accepts == 1 && f.engine->state.finishes == 1 &&
              f.engine->checkpoint_calls == 0,
          "accept/history/checkpoint contract failed");
    const std::array<double, 2> expected_eps{2.5 * (1. - .1 / (.1 + .2)),
                                             -1.5 * (1. - .1 / (.1 + .2))};
    for (int port = 0; port < 2; ++port) {
        check(std::abs(f.engine->state.eps[port] - expected_eps[port]) < 2e-15,
              "frozen filter not restored");
        double dt = 1. - .9;
        double expected =
            (entry.accepted[port] + dt * (3. - expected_eps[port])) / (1. + dt);
        check(std::abs(f.engine->state.accepted[port] - expected) < 2e-15,
              "accepted native RL oracle mismatch");
    }
    if (phase)
        check(f.engine->state.phase == 0,
              "rejected dt-dependent phase remained latched");
    // Prove new successful steps can arm again and canceled attempts do not
    // share stale tokens or leave the accepted protocol open.
    check(f.coupler->SnapshotNativeStep(f.external),
          "accepted step did not release readiness");
    check(f.coupler->CancelNativeStep(),
          "unused next attempt cancellation failed");
}
} // namespace
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    int result = 0;
    try {
        parameters();
        std::string mode = "replay", seed, library;
        amrex::ParmParse pp("test");
        pp.query("case", mode);
        pp.query("seed", seed);
        pp.query("library", library);
        if (mode == "legacy_library" || mode == "c3_library")
            absent_library(library);
        else
            run(mode, seed);
        amrex::Print() << "PASS circuit_rejection " << mode << "\n";
    } catch (const std::exception& error) {
        amrex::Print() << "FAIL circuit_rejection " << error.what() << "\n";
        result = 1;
    }
    amrex::ParallelDescriptor::ReduceIntMax(result);
    amrex::Finalize();
    return result;
}
