/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronThermalRuntime.H"
#include "Fields.H"
#include "MomentFixture.H"
#include <AMReX_VisMF.H>
#include <filesystem>

using namespace moment_test;
using namespace warpx::thermal;
using warpx::fields::FieldType;

ElectronEnergyRuntimeConfig
valid_config () {
    ElectronEnergyRuntimeConfig c;
    c.darwin = true;
    c.evolve_scheme = "theta_implicit_hybrid";
    c.corrected_axis = false;
    return c;
}
void
selector () {
    bool solve = false;
    amrex::ParmParse empty("native_legacy");
    check(ParseElectronEnergyMode(empty, solve) == ElectronEnergyMode::Legacy &&
              !solve,
          "legacy unset unchanged");
    empty.add("solve_electron_energy_equation", true);
    check(ParseElectronEnergyMode(empty, solve) == ElectronEnergyMode::Legacy &&
              solve,
          "legacy marker selection unchanged");
    for (auto mode :
         {ElectronEnergyMode::DecoupledJFNK, ElectronEnergyMode::CoupledJFNK}) {
        amrex::ParmParse pp(ElectronEnergyModeName(mode));
        pp.add("electron_energy_mode", ElectronEnergyModeName(mode));
        solve = false;
        check(ParseElectronEnergyMode(pp, solve) == mode && solve,
              "new mode enables energy");
#ifdef WARPX_DIM_RZ
        ValidateElectronEnergyRuntime(mode, valid_config());
#endif
    }
    ValidateElectronEnergyRuntime(ElectronEnergyMode::Legacy, {});
}
void
seed (Context& c) {
    c.opts.nodal_ghosts = 4;
    auto m = c.adapter();
    auto u = c.cell(1), exact = c.cell(), nodal = c.node();
    InitialElectronProfile p{t0, n0, 1.4, .06 * n0, true};
    KineticThermalStateView s{c.rho};
    s.pedestal = &c.ped;
    check(SeedCellElectronEnergy(m, s, p, gamma_e, u),
          "conservative physical product seed");
    for (amrex::MFIter it(nodal); it.isValid(); ++it) {
        auto rho = c.rho.const_array(it);
        auto ped = c.ped.const_array(it);
        auto out = nodal.array(it);
        amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                               int k) {
            Real const raw = (rho(i, j, k) + ped(i, j, k)) / PhysConst::q_e;
            Real const capacity = amrex::max(raw, .05 * n0);
            Real const t = t0 * std::pow(amrex::max(raw, .06 * n0) / n0, .4);
            out(i, j, k) = capacity * PhysConst::kb * t / (gamma_e - 1.);
        });
    }
    m.RestrictNativeMoment(nodal, 0, exact);
    near(difference(u, exact), 0., 3.e-15 * exact.norm0(),
         "U=R(n_eff kB T)/(gamma-1)");
    near(integral(c, u, false), integral(c, nodal, true, 0, true),
         1.e-12 * integral(c, u, false), "initial energy integral");
    s.energy = &u;
    check(m.Evaluate(s), "seed thermodynamics");
    check(m.NodalTemperature().nGrow() == 4 && m.OhmPressure().nGrow() == 4,
          "full solver ghosts allocated");
    check(!m.NodalTemperature().contains_nan(0, 1, 4), "full T images finite");
    check(!m.OhmPressure().contains_nan(0, 1, 4) &&
              !m.OhmPressure().contains_inf(0, 1, 4) &&
              m.OhmPressure().min(0, 4) > 0.,
          "all pressure ghost layers and corners are finite positive images");
    // Independently mirror every physical corner and use communicated interior
    // values, so a zero-filled extra layer cannot pass this test.
    auto const dom =
        amrex::convert(c.geom.Domain(), amrex::IntVect::TheNodeVector());
    auto const periodic = c.geom.isPeriodicArray();
    auto ghost_error = clone(m.NodalTemperature());
    for (amrex::MFIter it(ghost_error); it.isValid(); ++it) {
        auto a = m.NodalTemperature().const_array(it);
        auto out = ghost_error.array(it);
        amrex::ParallelFor(
            it.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int src[3] = {i, j, k};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (periodic[d])
                        continue;
                    if (src[d] < dom.smallEnd(d))
                        src[d] = 2 * dom.smallEnd(d) - src[d];
                    if (src[d] > dom.bigEnd(d))
                        src[d] = 2 * dom.bigEnd(d) - src[d];
                }
                out(i, j, k) = a(i, j, k) - a(src[0], src[1], src[2]);
            });
    }
    near(ghost_error.norm0(0, 4), 0., 0.,
         "all four thermal ghost layers and corners filled");
}
void
views (Context& c, std::string const& reject = "") {
    auto accepted_t = c.node(4), accepted_p = c.node(4);
    auto trial_t = c.node(reject == "ghost" ? 1 : 4), trial_p = c.node(4);
    accepted_t.setVal(31.);
    accepted_p.setVal(41.);
    trial_t.setVal(51.);
    trial_p.setVal(61.);
    ElectronThermalTrialViews v;
    if (reject == "alias")
        v.Set(0, accepted_t, trial_p, accepted_t, accepted_p);
    else
        v.Set(0, trial_t, trial_p, accepted_t, accepted_p);
    check(&v.Temperature(0, accepted_t) == &trial_t &&
              &v.Pressure(0, accepted_p) == &trial_p,
          "borrowed trial pointers");
    near(accepted_t.norm0(0, 4), 31., 0., "accepted T and ghosts untouched");
    near(accepted_p.norm0(0, 4), 41., 0., "accepted P and ghosts untouched");
    v.Clear(0);
    check(!v.Has(0) && &v.Temperature(0, accepted_t) == &accepted_t &&
              &v.Pressure(0, accepted_p) == &accepted_p,
          "explicit fallback after clear");
    v.Set(0, trial_t, trial_p, accepted_t, accepted_p);
    v.Clear();
    check(!v.Has(0), "all views clear");
}
void
serialization (Context& c) {
    ElectronEnergyMetadata before{ElectronEnergyMode::DecoupledJFNK, gamma_e,
                                  .05 * n0, false};
    auto after = ElectronEnergyMetadata::Decode(before.Encode());
    auto request = after;
    request.mode = ElectronEnergyMode::CoupledJFNK;
    after.ValidateRestart(request);
    check(after.Encode() == before.Encode(),
          "metadata lossless roundtrip and mode switch");
    ablastr::fields::MultiFabRegister writer, reader;
    for (auto* r : {&writer, &reader}) {
        r->alloc_init(FieldType::hybrid_electron_energy_fp, 0, c.ba, c.dm, 1,
                      amrex::IntVect(1), 0., true, true, true);
        r->alloc_init(FieldType::hybrid_electron_temperature_fp, 0,
                      c.rho.boxArray(), c.dm, 1, amrex::IntVect(4), -7., true,
                      true, false);
    }
    auto* u = writer.get(FieldType::hybrid_electron_energy_fp, 0);
    fill(*u, [] AMREX_GPU_DEVICE(int i, int j, int k) {
        return 3. + .1 * i + .2 * j + .3 * k;
    });
    std::string const dir = "runtime_checkpoint_" +
                            std::to_string(c.ba.size()) + "_" +
                            std::to_string(c.geom.isPeriodic(1)) + "/";
    if (amrex::ParallelDescriptor::IOProcessor())
        std::filesystem::create_directories(dir);
    amrex::ParallelDescriptor::Barrier();
    writer.write_checkpoints(0, dir);
    auto names = reader.read_restarts(0, dir);
    check(names.size() == 1 && names[0].find("hybrid_electron_energy_fp[") == 0,
          "only cell U is thermal checkpoint state");
    auto* restored = reader.get(FieldType::hybrid_electron_energy_fp, 0);
    near(difference(*u, *restored, true), 0., 0.,
         "real registry VisMF U roundtrip");
    near(reader.get(FieldType::hybrid_electron_temperature_fp, 0)->norm0(0, 4),
         7., 0., "derived Te is not restored");
    amrex::ParallelDescriptor::Barrier();
    if (amrex::ParallelDescriptor::IOProcessor())
        std::filesystem::remove_all(dir);
}
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        std::string name = "selector";
        int box = 4, periodic = 0, corrected = 0;
        pp.query("case", name);
        pp.query("max_grid_size", box);
        pp.query("periodic", periodic);
        pp.query("corrected_axis", corrected);
        Context c(box, periodic != 0, corrected != 0, true);
        if (name.rfind("reject_", 0) == 0) {
            auto which = name.substr(7);
            bool solve = false;
            amrex::ParmParse mode("bad");
            if (which == "mode" || which == "conflict") {
                mode.add("electron_energy_mode",
                         which == "mode" ? "bogus" : "coupled_jfnk");
                if (which == "conflict")
                    mode.add("solve_electron_energy_equation", false);
                ParseElectronEnergyMode(mode, solve);
            } else if (which == "metadata") {
                ElectronEnergyMetadata::Decode(
                    "WarpXElectronEnergy 999 mode coupled_jfnk coordinate "
                    "U_e_cell units J_per_m3 gamma 1.7 number_floor 1 "
                    "corrected_axis 0");
            } else if (which == "ghost" || which == "alias") {
                views(c, which);
            } else {
                auto config = valid_config();
                if (which == "eb")
                    config.embedded_boundary = true;
                if (which == "driver")
                    config.darwin = false;
                if (which == "amr")
                    config.max_level = 1;
                if (which == "floor")
                    config.representation_floor = 2.;
                if (which == "gamma")
                    config.gamma = 1.;
                ValidateElectronEnergyRuntime(ElectronEnergyMode::CoupledJFNK,
                                              config);
            }
            amrex::Abort("ERROR guard absent");
        }
        if (name == "selector")
            selector();
        else if (name == "seed")
            seed(c);
        else if (name == "views")
            views(c);
        else if (name == "serialization")
            serialization(c);
        else
            amrex::Abort("unknown runtime fixture");
        amrex::Print() << "PASS runtime " << name << " boxes=" << c.ba.size()
                       << " ranks=" << amrex::ParallelDescriptor::NProcs()
                       << '\n';
    }
    amrex::Finalize();
}
