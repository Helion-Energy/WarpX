/* Native production residual tests for conductivity compatibility. */
void
conductivity_point_laws () {
    auto const par = [] (Real n, Real t, Real time) {
        return n + 2 * t + time;
    };
    auto const perp = [] (Real n, Real t, Real time) {
        return .1 * (n + t + time);
    };
    for (Real b2 : {0., 1.e-20, .1, 1., 1.e20}) {
        for (int mode = 0; mode < 4; ++mode) {
            ThermalConductivityOptions o;
            o.isotropic = mode == 1;
            o.isotropic_B = mode == 2 ? .7 : -1.;
            o.unmagnetized_parallel = mode == 3;
            auto const value =
                EvaluateThermalConductivity(2., 3., .4, b2, o, par, perp);
            Real kp = 8.4, kt = .54;
            if (mode == 1)
                kp = kt;
            if (mode == 2)
                kp = kt + (kp - kt) * b2 / (b2 + .49);
            if (mode == 3 && b2 == 0)
                kt = kp;
            near(value.parallel, kp, 2.e-15, "parallel material law");
            near(value.perpendicular, kt, 2.e-15, "perpendicular material law");
        }
    }
    auto const negative = [] (Real, Real, Real) { return -1.; };
    require(!std::isfinite(
                EvaluateThermalConductivity(1., 1., 0., 1., {}, negative, perp)
                    .parallel),
            "negative law rejected");
    for (bool pedestal : {false, true}) {
        for (bool halo : {false, true}) {
            ThermalConductionDensityOptions o{10, halo, pedestal};
            for (Real raw : {-12., -1., 0., 1., 5., 10., 11., 30.}) {
                Real const ped = pedestal ? 5 : 0;
                auto const v = EvaluateThermalConductionDensity(raw, ped, o);
                Real const total = raw + ped;
                require(v.valid &&
                            v.active == (total > (halo || pedestal ? 0 : 10)),
                        "strict maintained open-set predicate");
                near(v.effective_number_density, std::max(total, 10.), 0,
                     "maintained floored capacity");
            }
        }
    }
    require(EvaluateThermalConductionDensity(1, 0, {10, false, true}).active,
            "zero-valued enabled pedestal retains zero gate");
    require(!EvaluateThermalConductionDensity(1, 0, {0, false, false}).valid,
            "positive effective floor required");
    require(!EvaluateThermalConductionDensity(1, -1, {10, false, true}).valid,
            "negative pedestal rejected");
    // floor(max)/restriction noncommutation is explicit, not a transfer rule.
    auto const a = EvaluateThermalConductionDensity(0, 0, {10, false, false});
    auto const b = EvaluateThermalConductionDensity(30, 0, {10, false, false});
    auto const c = EvaluateThermalConductionDensity(15, 0, {10, false, false});
    near((a.effective_number_density + b.effective_number_density) / 2 -
             c.effective_number_density,
         5, 0, "restriction and floor do not commute");
}

void
conductivity_laws (int box) {
    conductivity_point_laws();
    Real const k0 = density * PhysConst::kb / (gamma - 1);
    amrex::Parser parallel("min(A*(1+Te/100+t)*n/N0,C*n)");
    amrex::Parser perpendicular("min(A*(1+Te/100+t)*n/N0,C*n)");
    parallel.registerVariables({"n", "Te", "t"});
    perpendicular.registerVariables({"n", "Te", "t"});
    parallel.setConstant("A", k0);
    perpendicular.setConstant("A", .15 * k0);
    parallel.setConstant("N0", density);
    perpendicular.setConstant("N0", density);
    parallel.setConstant("C", 1.5 * PhysConst::kb * 2.);
    perpendicular.setConstant("C", 1.5 * PhysConst::kb * .3);
    auto const kp = parallel.compile<3>();
    auto const kt = perpendicular.compile<3>();
    for (Real n : {1.e10, 1.e18, 1.e24}) {
        for (Real t : {1.e-3, 50., 100., 1000.}) {
            auto const v =
                EvaluateThermalConductivity(n, t, .07, 1., {}, kp, kt);
            near(v.parallel / (1.5 * n * PhysConst::kb),
                 std::min(1 + t / 100 + .07, 2.), 8.e-16,
                 "borrowed parallel hard cap and constants");
            near(v.perpendicular / (1.5 * n * PhysConst::kb),
                 std::min(.15 * (1 + t / 100 + .07), .3), 2.e-16,
                 "borrowed perpendicular hard cap and constants");
        }
    }
    std::string const rawp = "min(" + number(k0) + "*(1+Te/100+t)*n/" +
                             number(density) + "," +
                             number(1.5 * PhysConst::kb * 2) + "*n)";
    std::string const rawt = "min(" + number(.15 * k0) + "*(1+Te/100+t)*n/" +
                             number(density) + "," +
                             number(1.5 * PhysConst::kb * .3) + "*n)";
    for (int order : {2, 4}) {
        for (int mode = 0; mode < 7; ++mode) {
            Context c(box, false);
            c.set_pattern(.2, true, true);
            Real const strength = mode >= 3 && mode <= 5 ? 0 : .7;
            c.b.setVal(.6 * strength, 0, 1, 3);
            c.b.setVal(.3 * strength, 1, 1, 3);
            c.b.setVal(.8 * strength, 2, 1, 3);
            auto o = options();
            o.dt = .003;
            o.time = .07;
            o.order = order;
            o.use_compiled_conductivity = true;
            o.parallel_executor = kp;
            o.perpendicular_executor = kt;
            // Undefined symbols would fail if the runtime borrowed path were
            // accidentally recompiled from uncapped fixture strings.
            o.kappa_parallel = "runtime_only_constant";
            o.kappa_perpendicular = "runtime_only_constant";
            o.conductivity.isotropic = mode == 1 || mode == 6;
            o.conductivity.isotropic_B =
                mode == 2 || mode == 5 || mode == 6 ? 2 : -1;
            o.conductivity.unmagnetized_parallel = mode == 3 || mode == 6;
            o.boundary[0][1].kind = BoundaryKind::Reservoir;
            o.boundary[0][1].value = 80;
            o.boundary[0][1].drain_only = true;
            o.boundary[0][1].flux_limit = 1.e-7;
            o.boundary[1][1].kind = BoundaryKind::Reservoir;
            o.boundary[1][1].value = 80; // ordinary emitted coefficient row
            auto expected = o;
            expected.use_compiled_conductivity = false;
            expected.conductivity = {};
            expected.kappa_parallel = rawp;
            expected.kappa_perpendicular = rawt;
            if (o.conductivity.isotropic) {
                expected.kappa_parallel = rawt;
            } else if (o.conductivity.isotropic_B > 0) {
                Real const b2 = 1.09 * strength * strength;
                Real const s = b2 / (b2 + 4);
                expected.kappa_parallel = "(" + rawt + ")+((" + rawp + ")-(" +
                                          rawt + "))*" + number(s);
            } else if (o.conductivity.unmagnetized_parallel && strength == 0) {
                expected.kappa_perpendicular = rawp;
            }
            EulerianThermalStage actual(c.sim.geometry, c.old, c.n, c.b,
                                        c.source, o);
            EulerianThermalStage control(c.sim.geometry, c.old, c.n, c.b,
                                         c.source, expected);
            auto ra = c.scalar(), re = c.scalar();
            require(actual.Residual(c.old, ra) && control.Residual(c.old, re),
                    "conductivity residuals valid");
            near(difference(ra, re) / energy, 0, 3.e-12,
                 "bulk/reservoir material oracle");
            for (int d = 0; d < 2; ++d) {
                near(difference(actual.FaceFlux(d), control.FaceFlux(d)) /
                         std::max(control.FaceFlux(d).norm0(0), Real(1)),
                     0, 3.e-13,
                     "unchanged MHD face stencil after material transform");
                near(difference(actual.FaceCoefficient(d),
                                control.FaceCoefficient(d)) /
                         std::max(control.FaceCoefficient(d).norm0(0), Real(1)),
                     0, 3.e-13,
                     "PC compact normal row uses transformed material");
                near(
                    difference(actual.FaceWallDerivative(d),
                               control.FaceWallDerivative(d)) /
                        std::max(control.FaceWallDerivative(d).norm0(0),
                                 Real(1)),
                    0, 3.e-10,
                    "nonlinear reservoir derivative uses transformed material");
            }
            near(integral(ra, c.sim.geometry) +
                     o.theta * o.dt * (-boundary_power(actual)),
                 0, 1.e-10 * energy, "conductive physical RZ balance");
            // A physical parallel leg is its own retained constitutive law.
            o.boundary[0][1].kind = BoundaryKind::Leg;
            o.boundary[0][1].value = .5;
            o.boundary[0][1].flux_limit = 0;
            expected = o;
            expected.conductivity = {};
            EulerianThermalStage leg(c.sim.geometry, c.old, c.n, c.b, c.source,
                                     o);
            EulerianThermalStage rawleg(c.sim.geometry, c.old, c.n, c.b,
                                        c.source, expected);
            require(leg.Residual(c.old, ra) && rawleg.Residual(c.old, re),
                    "parallel leg laws");
            for (amrex::MFIter mfi(leg.FaceFlux(0)); mfi.isValid(); ++mfi) {
                auto const a = leg.FaceFlux(0).const_array(mfi);
                auto const b = rawleg.FaceFlux(0).const_array(mfi);
                amrex::LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k) {
                    if (i == 16)
                        near(a(i, j, k), b(i, j, k), 0,
                             "leg retains raw parallel law");
                });
            }
        }
    }
}

void
conductivity_variable_field (int box) {
    Context c(box, false);
    Real const k0 = density * PhysConst::kb / (gamma - 1);
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto const u = c.old.array(mfi);
        auto const b = c.b.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               Real const z = (j + .5) / 32;
                               u(i, j, k) = density * PhysConst::q_e /
                                            (gamma - 1) * (100 + 20 * z);
                               b(i, j, k, 2) = .1 + z;
                           });
    }
    auto o = options();
    o.kappa_parallel = number(5 * k0);
    o.kappa_perpendicular = number(.2 * k0);
    o.conductivity.isotropic_B = .7;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto r = c.scalar();
    require(stage.Residual(c.old, r), "spatially varying weak-B law");
    for (amrex::MFIter mfi(stage.FaceFlux(1)); mfi.isValid(); ++mfi) {
        auto const q = stage.FaceFlux(1).const_array(mfi);
        auto const row = stage.FaceCoefficient(1).const_array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k) {
            if (j == 0 || j == 32)
                return;
            Real const bl = .1 + (j - .5) / 32, br = .1 + (j + .5) / 32;
            Real const kl = k0 * (.2 + 4.8 * bl * bl / (bl * bl + .49));
            Real const kr = k0 * (.2 + 4.8 * br * br / (br * br + .49));
            Real const kface = .5 * (kl + kr);
            near(q(i, j, k) / (-kface * 20 * PhysConst::q_e / PhysConst::kb), 1,
                 1.e-13, "analytic variable-B face heat flux");
            near(row(i, j, k) / (kface * (gamma - 1) / PhysConst::kb), 1,
                 8.e-16, "analytic variable-B compact PC row");
        });
    }
}

void
conductivity_active (int box) {
    Context c(box, false);
    c.set_pattern(.05, true);
    auto active = c.scalar();
    for (amrex::MFIter mfi(active); mfi.isValid(); ++mfi) {
        auto const a = active.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               a(i, j, k) = j >= 10 && j <= 13 ? 0 : 1;
                           });
    }
    auto o = options();
    o.order = 4;
    o.dt = .003;
    c.b.setVal(.6, 0, 1, 3);
    c.b.setVal(.8, 2, 1, 3);
    ThermalFaceContext faces;
    faces.conduction_active = &active;
    auto const before = snapshot(active);
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                               c.source, faces, o);
    auto residual = c.scalar(), v = c.scalar(), a = c.scalar(), b = c.scalar();
    require(stage.Residual(c.old, residual), "masked oblique residual");
    near(integral(residual, c.sim.geometry), 0, 3.e-13 * energy,
         "closed mask still conserves physical extensive energy");
    for (amrex::MFIter mfi(residual); mfi.isValid(); ++mfi) {
        auto const r = residual.const_array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k) {
            if (j >= 10 && j <= 13)
                near(r(i, j, k), 0, 0, "inactive cell has zero conduction");
        });
    }
    for (int d = 0; d < 2; ++d) {
        for (amrex::MFIter mfi(stage.FaceFlux(d)); mfi.isValid(); ++mfi) {
            auto const q = stage.FaceFlux(d).const_array(mfi);
            auto const row = stage.FaceCoefficient(d).const_array(mfi);
            amrex::LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k) {
                bool const shut = j >= 10 && j <= 13 + (d == 1);
                if (shut) {
                    near(q(i, j, k), 0, 0, "incident heat flux exactly closed");
                    near(row(i, j, k), 0, 0, "incident PC row exactly closed");
                }
            });
        }
    }
    require(snapshot(active) == before, "caller mask untouched by residual");
    active.setVal(0);
    require(stage.Residual(c.old, a), "owned mask survives caller mutation");
    near(difference(a, residual), 0, 0, "deep-copied mask");
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
        auto const x = v.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            x(i, j, k) = energy * (.1 + .03 * std::sin(Real(2 * i + 3 * j)));
        });
    }
    ThermalConductionPC pc(stage);
    require(pc.Freeze(c.old), "masked PC freeze");
    pc.ApplyOperator(a, v);
    require(stage.RefreshContext(c.n, c.n, c.b, c.source, faces),
            "refresh closed mask");
    pc.ApplyOperator(b, v);
    near(difference(a, b), 0, 0, "frozen operator mask immune to live refresh");
    pc.ApplyRows(b, v);
    near(difference(a, b) / energy, 0, 3.e-15,
         "frozen row emission mask immune to live refresh");
    ThermalConductionPC closed_pc(stage);
    require(closed_pc.Freeze(c.old), "fresh fully closed PC");
    closed_pc.ApplyOperator(b, v);
    Real const frozen_mask_gap = difference(a, b) / energy;
    require(frozen_mask_gap > .001,
            "nonuniform probe detects mask replacement");
    amrex::Print() << "MASK frozen_vs_refreshed_action_gap=" << frozen_mask_gap
                   << '\n';
    pc.Apply(a, v);
    active.setVal(1);
    require(stage.RefreshContext(c.n, c.n, c.b, c.source, faces),
            "refresh open mask");
    pc.Apply(b, v);
    near(difference(a, b), 0, 0, "frozen inverse mask immune to live refresh");
    require(stage.Residual(c.old, residual), "open-mask base");
    active.setVal(.5);
    require(!stage.RefreshContext(c.n, c.n, c.b, c.source, faces),
            "fractional mask rejected");
    require(stage.Residual(c.old, a), "rejected refresh leaves valid base");
    near(difference(a, residual), 0, 0, "transactional mask rejection");
    active.setVal(std::numeric_limits<Real>::quiet_NaN());
    require(!stage.RefreshContext(c.n, c.n, c.b, c.source, faces),
            "nonfinite mask rejected");
    ThermalFaceContext bad;
    bad.conduction_active = &c.b;
    require(!stage.RefreshContext(c.n, c.n, c.b, c.source, bad),
            "wrong mask layout rejected");
    active.setVal(0);
    FaceFlow flow(c, 0, .1);
    flow.context.conduction_active = &active;
    c.source.setVal(.03 * energy);
    o.boundary[1][0].inflow_temperature_ev = temperature;
    o.boundary[1][1].kind = BoundaryKind::PrescribedFlux;
    o.boundary[1][1].value = .3;
    o.kappa_parallel = "-1"; // dormant invalid material law is never evaluated
    auto zero = o;
    zero.kappa_parallel = "0";
    zero.boundary[1][1].kind = BoundaryKind::Adiabatic;
    ThermalFaceContext openflow = flow.context;
    openflow.conduction_active = nullptr;
    EulerianThermalStage closed(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                                c.source, flow.context, o);
    EulerianThermalStage control(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                                 c.source, openflow, zero);
    require(closed.Residual(c.old, a) && control.Residual(c.old, b),
            "closed-law purity");
    near(difference(a, b), 0, 0,
         "mask leaves capacity/source/advection unchanged");
    active.setVal(1);
    require(closed.RefreshContext(c.n, c.n, c.b, c.source, flow.context),
            "activate invalid law");
    require(!closed.Residual(c.old, a), "active invalid law rejected");
}

void
conductivity_roots (int box) {
    Context c(box, true);
    c.set_pattern(.03, false, true);
    auto active = c.scalar();
    active.setVal(1);
    for (amrex::MFIter mfi(active); mfi.isValid(); ++mfi) {
        auto const a = active.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               if (j == 8 || j == 24)
                                   a(i, j, k) = 0;
                           });
    }
    ThermalFaceContext faces;
    faces.conduction_active = &active;
    auto o = options();
    o.dt = .002;
    o.kappa_parallel = number(PhysConst::kb / (gamma - 1)) + "*n*(Te/100)^1.5";
    o.kappa_perpendicular =
        number(.2 * PhysConst::kb / (gamma - 1)) + "*n*(Te/100)^1.5";
    o.conductivity.isotropic_B = .7;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                               c.source, faces, o);
    auto end = c.scalar(), saved = c.scalar();
    ThermalSolveOptions solver;
    for (bool enabled : {true, false}) {
        solver.use_preconditioner = enabled;
        amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
        auto const result = SolveThermalStage(stage, c.u, end, name, solver);
        converged(enabled ? "conductivity_pc" : "conductivity_no_pc", result);
        if (enabled)
            amrex::MultiFab::Copy(saved, end, 0, 0, 1, 0);
        else
            near(difference(saved, end) / energy, 0, 2.e-9,
                 "masked nonlinear PC roots agree");
        near((integral(end, c.sim.geometry) - integral(c.old, c.sim.geometry)) /
                 integral(c.old, c.sim.geometry),
             0, 2.e-11, "closed nonlinear energy balance");
    }
}

void
conductivity_zero_floor (int box) {
    for (Real theta : {.5, 1.}) {
        Context c(box, true);
        auto o = options();
        o.theta = theta;
        o.conduction_theta = theta;
        o.temperature_floor_ev = 0;
        o.kappa_parallel = "0";
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source,
                                   o);
        auto trial = c.scalar(), residual = c.scalar(), direction = c.scalar();
        require(stage.Admissible(c.old) && stage.Residual(c.old, residual),
                "positive state with configured zero floor");
        residual.setVal(-321);
        auto const before = snapshot(residual);
        for (Real fraction : {0., -1.e-3}) {
            trial.setVal((1 - theta + theta * fraction) * energy);
            require(!stage.Admissible(trial),
                    "nonpositive true endpoint rejected");
            require(!stage.Residual(trial, residual),
                    "nonpositive endpoint residual rejected");
            require(snapshot(residual) == before,
                    "invalid zero-floor probe preserves output");
        }
        trial.setVal((1 - theta + theta * 1.e-10) * energy);
        require(stage.Admissible(trial) && stage.Residual(trial, residual),
                "small positive endpoint accepted without artificial floor");
        direction.setVal(-energy);
        Real const bound = stage.StepBound(c.old, direction);
        near(bound, theta, 2.e-16, "zero-floor step bound");
        amrex::MultiFab::LinComb(trial, 1, c.old, 0, .99 * bound, direction, 0,
                                 0, 1, 0);
        require(stage.Admissible(trial),
                "step-bound backoff preserves strict positivity");
        c.source.setVal(-.5 * energy / o.dt);
        EulerianThermalStage cooling(c.sim.geometry, c.old, c.n, c.b, c.source,
                                     o);
        auto end = c.scalar(), saved = c.scalar();
        for (bool enabled : {true, false}) {
            ThermalSolveOptions solver;
            solver.use_preconditioner = enabled;
            amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1,
                                  0);
            auto const result =
                SolveThermalStage(cooling, c.u, end, name, solver);
            converged(enabled ? "zero_floor_pc" : "zero_floor_no_pc", result);
            trial.setVal(.5 * energy);
            near(difference(end, trial) / energy, 0, 2.e-10,
                 "zero-floor physical cooling root");
            if (enabled)
                amrex::MultiFab::Copy(saved, end, 0, 0, 1, 0);
            else
                near(difference(saved, end) / energy, 0, 2.e-10,
                     "zero-floor PC roots agree");
        }
    }
}
