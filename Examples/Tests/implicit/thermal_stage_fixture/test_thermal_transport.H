/* Production residual transport, physical leg and live-context qualification.
 */
struct FaceFlow {
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> velocity;
    ThermalFaceContext context;
    explicit FaceFlow (Context const& c, Real radial = 0, Real axial = 0,
                       bool linear = false) {
        auto const dx = c.sim.geometry.CellSizeArray();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            velocity[d] = std::make_unique<amrex::MultiFab>(
                amrex::convert(c.old.boxArray(),
                               amrex::IntVect::TheDimensionVector(d)),
                c.old.DistributionMap(), 1, 0);
            context.velocity[d] = velocity[d].get();
            for (amrex::MFIter mfi(*velocity[d]); mfi.isValid(); ++mfi) {
                auto const out = velocity[d]->array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                       int i, int j, int k) {
                    out(i, j, k) = d == 0 ? radial * (linear ? i * dx[0] : 1)
                                          : axial * (linear ? j * dx[1] : 1);
                });
            }
        }
    }
};

void
transport_identity (int box) {
    Context c(box, true);
    FaceFlow flow(c, 0, 0.7);
    auto o = options();
    o.kappa_parallel = "0";
    o.dt = 0.01;
    EulerianThermalStage uniform(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                                 c.source, flow.context, o);
    auto residual = c.scalar();
    require(uniform.Residual(c.old, residual),
            "uniform moving state admissible");
    near(residual.norm0(0), 0, 0,
         "uniform divergence-free translation exactly null");
    // Translating isothermal contact solves the independently discretized
    // continuity equation at the same stage. Distinct n_old/n_s/n_end matter.
    auto old_n = c.scalar(), end_n = c.scalar(), stage_u = c.scalar();
    Real const h = o.dt * o.theta, theta = o.theta, v = 0.7, a = 0.4;
    Real const dz = c.sim.geometry.CellSize(1), wave = 2 * MathConst::pi;
    for (Real penalty : {Real(0), Real(0.25), Real(1)}) {
        o.transport.central_dissipation_entropy = penalty;
        for (amrex::MFIter mfi(c.n); mfi.isValid(); ++mfi) {
            auto const ns = c.n.array(mfi), nn = old_n.array(mfi),
                       ne = end_n.array(mfi);
            auto const us = stage_u.array(mfi), un = c.old.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real const z = (j + 0.5) * dz;
                    Real const n = density * (1 + a * std::cos(wave * z));
                    Real const divergence =
                        density * a *
                        (-v * std::sin(wave * z) * std::sin(wave * dz) / dz +
                         penalty * std::abs(v) * (1 - std::cos(wave * dz)) /
                             dz * std::cos(wave * z));
                    ns(i, j, k) = n;
                    nn(i, j, k) = n + h * divergence;
                    ne(i, j, k) = (n - (1 - theta) * nn(i, j, k)) / theta;
                    us(i, j, k) = specific * n;
                    un(i, j, k) = specific * nn(i, j, k);
                });
        }
        EulerianThermalStage contact(c.sim.geometry, c.old, old_n, c.n, end_n,
                                     c.b, c.source, flow.context, o);
        require(contact.Residual(stage_u, residual),
                "translating contact admissible");
        near(residual.norm0(0) / energy, 0, 2.e-14,
             "constant T moving contact continuity identity");
        if (penalty == 0) {
            FaceFlow number_flow(c);
            ThermalFaceContext number_context;
            c.n.FillBoundary(c.sim.geometry.periodicity());
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                number_context.number_flux[d] = number_flow.velocity[d].get();
                if (d == 0) {
                    continue;
                }
                for (amrex::MFIter mfi(*number_flow.velocity[d]); mfi.isValid();
                     ++mfi) {
                    auto const n = c.n.const_array(mfi);
                    auto const fn = number_flow.velocity[d]->array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            fn(i, j, k) =
                                0.5 * (n(i, j, k) + n(i, j - 1, k)) * v;
                        });
                }
            }
            EulerianThermalStage conserved(c.sim.geometry, c.old, old_n, c.n,
                                           end_n, c.b, c.source, number_context,
                                           o);
            require(conserved.Residual(stage_u, residual),
                    "conserved number-flux stage");
            near(residual.norm0(0) / energy, 0, 2.e-14,
                 "native number-flux contact and compression identity");
            auto bad = o;
            bad.transport.central_dissipation_entropy = 0.25;
            EulerianThermalStage penalty_control(c.sim.geometry, c.old, old_n,
                                                 c.n, end_n, c.b, c.source,
                                                 flow.context, bad);
            require(!penalty_control.RefreshContext(c.n, end_n, c.b, c.source,
                                                    number_context),
                    "kinetic number flux cannot silently acquire MHD density "
                    "diffusion");
            bad = o;
            bad.transport.fluid_reconstruction = 5;
            EulerianThermalStage reconstruction_control(
                c.sim.geometry, c.old, old_n, c.n, end_n, c.b, c.source,
                flow.context, bad);
            require(!reconstruction_control.RefreshContext(
                        c.n, end_n, c.b, c.source, number_context),
                    "kinetic reconstruction/compression requires separate "
                    "qualification");
        }
    }
    // Smooth reconstruction on stationary contacts cannot invent flow-speed
    // entropy diffusion, including a sharp 1024:1 density jump.
    flow.velocity[1]->setVal(0);
    c.set_pattern(0, false, true);
    for (int mode = 0; mode <= 5; ++mode) {
        o.transport.fluid_reconstruction = mode;
        EulerianThermalStage stationary(c.sim.geometry, c.old, c.n, c.n, c.n,
                                        c.b, c.source, flow.context, o);
        require(stationary.Residual(c.old, residual),
                "stationary contact reconstructed");
        near(residual.norm0(0), 0, 0,
             "reconstruction preserves static contact");
    }
}

void
compression_identity (int box) {
    for (Real rate : {Real(-0.2), Real(0.2)}) {
        Context c(box, false);
        FaceFlow flow(c, rate, rate, true);
        auto o = options();
        o.kappa_parallel = "0";
        o.dt = 0.05;
        Real const divergence = 3 * rate;
        Real const expected_stage =
            energy / (1 + o.theta * o.dt * gamma * divergence);
        if (rate < 0) {
            o.boundary[0][1].inflow_temperature_ev =
                temperature * expected_stage / energy;
            o.boundary[1][1].inflow_temperature_ev =
                temperature * expected_stage / energy;
        }
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                                   c.source, flow.context, o);
        auto residual = c.scalar(), expected = c.scalar(), end = c.scalar();
        expected.setVal(expected_stage);
        require(stage.Residual(expected, residual),
                "uniform compressive/expansive stage");
        near(residual.norm0(0) / energy, 0, 3.e-15,
             "paired central flux/compression identity");
        auto result = SolveThermalStage(stage, c.u, end, name);
        converged("compression", result);
        near(difference(c.u.getMultiFabBlock(name, 0), expected) / energy, 0,
             5.e-11, "compressive-flow root");
        amrex::MultiFab::Subtract(end, c.old, 0, 0, 1, 0);
        Real const total_change = integral(end, c.sim.geometry);
        // Volume-integrated advective outflow is U*div(u); pressure work is
        // (gamma-1)U*div(u), with the exact RZ axis/full-cell volume.
        near(total_change,
             -o.dt * gamma * divergence * expected_stage * MathConst::pi, 1.e-9,
             "physical extensive RZ transport/work balance");
    }
}

void
density_context (int box) {
    Context c(box, true);
    c.set_pattern(0, false, true);
    auto old_n = c.scalar(), end_n = c.scalar(), stage_u = c.scalar(),
         residual = c.scalar();
    amrex::MultiFab::Copy(old_n, c.n, 0, 0, 1, 0);
    amrex::MultiFab::Copy(end_n, c.n, 0, 0, 1, 0);
    end_n.mult(1.5, 0);
    c.n.mult(1.2, 0);
    amrex::MultiFab::Copy(stage_u, c.old, 0, 0, 1, 0);
    stage_u.mult(1.2, 0);
    auto o = options();
    o.conduction_theta = 1;
    amrex::MultiFab::LinComb(c.source, 1 / (o.theta * o.dt), stage_u, 0,
                             -1 / (o.theta * o.dt), c.old, 0, 0, 1, 0);
    EulerianThermalStage stage(c.sim.geometry, c.old, old_n, c.n, end_n, c.b,
                               c.source, {}, o);
    require(stage.Residual(stage_u, residual), "distinct density context");
    near(residual.norm0(0) / energy, 0, 3.e-12,
         "mixed conduction retains constant old/stage T");
    near(stage.FaceFlux(0).norm0(0) + stage.FaceFlux(1).norm0(0), 0, 0,
         "old bracket must use old density");
    ThermalConductionPC pc(stage);
    require(pc.Freeze(stage_u), "freeze context PC");
    auto direction = c.scalar(), before = c.scalar(), after = c.scalar();
    direction.setVal(1);
    pc.ApplyOperator(before, direction);
    auto inverse_before = c.scalar(), inverse_after = c.scalar();
    auto rows_before = c.scalar(), rows_after = c.scalar();
    pc.Apply(inverse_before, direction);
    pc.ApplyRows(rows_before, direction);
    end_n.setVal(1.e30);
    require(stage.RefreshContext(c.n, end_n, c.b, c.source),
            "valid context refresh");
    require(!stage.Admissible(stage_u), "endpoint uses true endpoint density");
    c.n.mult(2, 0);
    end_n.setVal(density);
    require(stage.RefreshContext(c.n, end_n, c.b, c.source),
            "second context refresh");
    pc.ApplyOperator(after, direction);
    near(difference(before, after), 0, 0,
         "frozen PC cannot read mutable density");
    pc.Apply(inverse_after, direction);
    pc.ApplyRows(rows_after, direction);
    near(difference(inverse_before, inverse_after), 0, 0,
         "frozen PC inverse ignores refreshed density");
    near(difference(rows_before, rows_after), 0, 0,
         "frozen PC emitted rows ignore refreshed density");
    c.n.setVal(-1);
    auto const saved = snapshot(stage.Density());
    require(!stage.RefreshContext(c.n, end_n, c.b, c.source),
            "negative density context rejects");
    require(snapshot(stage.Density()) == saved,
            "context refresh validates transactionally");
    Context cap_context(box, true);
    auto ns = cap_context.scalar(), ne = cap_context.scalar();
    ns.setVal(3 * density);
    ne.setVal(0.25 * density);
    o.free_streaming_fraction = 0.01;
    EulerianThermalStage cap_stage(cap_context.sim.geometry, cap_context.old,
                                   cap_context.n, ns, ne, cap_context.b,
                                   cap_context.source, {}, o);
    require(cap_stage.Admissible(cap_context.old),
            "endpoint density admits cap probe");
    auto cap_residual = cap_context.scalar();
    require(!cap_stage.Residual(cap_context.old, cap_residual),
            "nonphysical mixed cap temperature must not silently disable cap");
}

void
source_callback (int box) {
    Context c(box, true);
    c.source.setVal(1.e9);
    auto o = options();
    o.kappa_parallel = "0";
    o.dt = 0.2;
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    Real const cooling = 0.7;
    stage.SetSourceEvaluator([=] (auto const& u, auto const&, auto const&,
                                  auto& q, auto& derivative) {
        for (amrex::MFIter mfi(q); mfi.isValid(); ++mfi) {
            auto const in = u.const_array(mfi), out = q.array(mfi),
                       d = derivative.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   // Deliberately +=: the stage must clear
                                   // callback scratch.
                                   out(i, j, k) += -cooling * in(i, j, k);
                                   d(i, j, k) = -cooling;
                               });
        }
        return true;
    });
    auto r = c.scalar(), a = c.scalar(), b = c.scalar(), end = c.scalar();
    require(stage.Residual(c.old, a), "source callback A");
    b.setVal(1.1 * energy);
    require(stage.Residual(b, r), "source callback B");
    require(stage.Residual(c.old, r), "source callback A repeat");
    near(difference(a, r), 0, 0,
         "source replacement A/B/A and cleared scratch");
    ThermalConductionPC pc(stage);
    require(pc.Freeze(c.old), "source local PC freeze");
    b.setVal(1);
    pc.ApplyOperator(a, b);
    a.plus(-(1 + o.theta * o.dt * cooling), 0);
    near(a.norm0(0), 0, 5.e-15, "source derivative in PC row");
    auto result = SolveThermalStage(stage, c.u, end, name);
    converged("live_source", result);
    Real const expected =
        (energy / (1 + o.theta * o.dt * cooling) - (1 - o.theta) * energy) /
        o.theta;
    end.plus(-expected, 0);
    near(end.norm0(0) / energy, 0, 2.e-12,
         "live source root and replacement semantics");
    auto saved = snapshot(c.old);
    stage.SetSourceEvaluator([] (auto const&, auto const&, auto const&, auto&,
                                 auto&) { return false; });
    require(!stage.Residual(c.old, r), "callback rejects trial");
    require(snapshot(c.old) == saved,
            "callback leaves accepted input unchanged");
}

void
leg_law (int box) {
    using warpx::thermal::LegFaceParameters;
    Real const kappa = 2.e-5, tc = 100, ne = density;
    auto const constant = [=] AMREX_GPU_HOST_DEVICE(Real, Real, Real) {
        return kappa;
    };
    LegFaceParameters p{0.5, 6, 1. / 64, 0, false, gamma, 0, 1.e-6};
    auto const f = EvaluateLegFace(tc, ne, 0, p, constant);
    Real const expected = kappa / (p.length + p.half_cell) * (tc - p.bath_ev) *
                          PhysConst::q_e / PhysConst::kb;
    near(f.flux / expected, 1, 3.e-15,
         "halfcell+leg constant-conductivity series law");
    near(f.half_conductance * (tc - f.interface_ev),
         f.leg_conductance * (f.interface_ev - p.bath_ev), 1.e-14,
         "nonlinear interface flux balance");
    near(EvaluateLegFace(0.4, ne, 0, p, constant).flux, 0, 0, "leg drain gate");
    p.flux_limit = 1.e-8;
    p.sonic = true;
    p.ion_mass = 2 * PhysConst::m_p;
    auto const cap = EvaluateLegFace(tc, ne, 0, p, constant);
    require(cap.cap < cap.uncapped_flux, "sonic cap materially active");
    near(cap.flux, cap.cap, 0, "maintained leg uses hard cap");
    near(LegFaceTemperatureDerivative(tc, ne, 0, p, constant) /
             (1.5 * cap.flux / tc),
         1, 2.e-10, "sonic cap temperature derivative");
    near(EvaluateLegFace(tc, 1.2 * ne, 0, p, constant).flux / cap.flux, 1.2,
         4.e-15, "sonic cap physical density derivative");
    p.sonic = false;
    auto const fs = EvaluateLegFace(tc, ne, 0, p, constant);
    near(fs.cap / cap.cap,
         std::sqrt(2 * PhysConst::m_p / (gamma * PhysConst::m_e)), 1.e-12,
         "free-streaming vs sonic speed");
    // Legacy target with the matching access distance has the same STEADY
    // interface; its finite-dt node reset remains a different operator.
    Real const kelvin = PhysConst::q_e / PhysConst::kb;
    auto const target = EvaluateLegTarget(
        f.interface_ev * kelvin, tc * kelvin, ne, ne, ne, true, p.half_cell, 1,
        p.length, p.bath_ev * kelvin, p.bath_ev * kelvin, 4, constant, 0);
    near(target.temperature / kelvin, f.interface_ev, 1.e-12,
         "legacy steady series target comparison");
    auto const legacy = LegWallTemperature(tc * kelvin, target, ne, 32, 1.e-5,
                                           0, false, gamma, 0) /
                        kelvin;
    require(legacy > target.temperature / kelvin && legacy < tc,
            "legacy finite-dt node reset is not eliminated interface");

    Context c(box, false);
    auto o = options();
    o.kappa_parallel = number(kappa) + "*(Te/100)^2*(n/1e18)^0.4";
    o.dt = 0.2;
    o.conduction_theta = 1;
    o.boundary[1][0] = {BoundaryKind::Leg, 0.5};
    o.boundary[1][1] = {BoundaryKind::Leg, 0.5};
    auto trial = c.scalar(), rp = c.scalar(), rm = c.scalar(),
         delta = c.scalar(), action = c.scalar(), rows = c.scalar();
    for (Real limit : {Real(0), Real(1.e-8)}) {
        for (int side = 0; side < 2; ++side) {
            o.boundary[1][side].flux_limit = limit;
            o.boundary[1][side].cap_form = WallCapForm::Sonic;
            o.boundary[1][side].ion_mass = 2 * PhysConst::m_p;
        }
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source,
                                   o);
        ThermalConductionPC pc(stage);
        require(pc.Freeze(c.old), "nonlinear leg PC freeze");
        delta.setVal(energy);
        pc.ApplyOperator(action, delta);
        pc.ApplyRows(rows, delta);
        near(difference(action, rows) / energy, 0, 5.e-12,
             "leg local row matches MLMG");
        Real const eps = 2.e-5;
        trial.setVal((1 + eps) * energy);
        require(stage.Residual(trial, rp), "leg +probe");
        trial.setVal((1 - eps) * energy);
        require(stage.Residual(trial, rm), "leg -probe");
        amrex::MultiFab::LinComb(rows, 1 / (2 * eps), rp, 0, -1 / (2 * eps), rm,
                                 0, 0, 1, 0);
        near(difference(action, rows) / energy, 0, 2.e-7,
             "physical kappa/cap wall derivative matches PC local row");
        auto const old_snapshot = snapshot(c.old), n_snapshot = snapshot(c.n);
        require(stage.Residual(c.old, rp), "base wall density response");
        Real const base_flux = stage.FaceFlux(1).norm0(0);
        auto changed_n = c.scalar();
        Real const density_step = 1.e-4;
        changed_n.setVal(density * (1 + density_step));
        trial.setVal(energy * (1 + density_step));
        require(stage.RefreshContext(changed_n, c.n, c.b, c.source),
                "live wall density probe");
        require(stage.Residual(trial, rm),
                "density-dependent physical wall residual");
        near(stage.FaceFlux(1).norm0(0) / base_flux,
             std::pow(1 + density_step, limit > 0 ? 1 : 0.4), 2.e-12,
             "physical leg density/cap response in residual");
        require(stage.RefreshContext(c.n, c.n, c.b, c.source),
                "restore wall context");
        require(stage.Residual(c.old, rp), "leg A");
        trial.setVal(1.1 * energy);
        require(stage.Residual(trial, rm), "leg B");
        require(stage.Residual(c.old, rm), "leg A repeat");
        near(difference(rp, rm), 0, 0, "leg purity A/B/A");
        require(snapshot(c.old) == old_snapshot && snapshot(c.n) == n_snapshot,
                "wall inputs immutable");
    }
    // The separately configured hard sonic reservoir uses the same local-row
    // ownership; its capped derivative has an independent analytic oracle.
    for (int side = 0; side < 2; ++side) {
        o.boundary[1][side].kind = BoundaryKind::Reservoir;
    }
    EulerianThermalStage reservoir(c.sim.geometry, c.old, c.n, c.b, c.source,
                                   o);
    ThermalConductionPC reservoir_pc(reservoir);
    require(reservoir_pc.Freeze(c.old), "physical hard-capped reservoir PC");
    Real const qcap =
        1.e-8 * density * PhysConst::q_e * temperature *
        std::sqrt(gamma * PhysConst::q_e * temperature / (2 * PhysConst::m_p));
    near(reservoir.FaceFlux(1).norm0(0), qcap, 1.e-14,
         "explicit hard sonic reservoir cap law");
    delta.setVal(energy);
    reservoir_pc.ApplyOperator(action, delta);
    Real const correction =
        o.dt * o.conduction_theta * 1.5 * qcap / c.sim.geometry.CellSize(1);
    for (amrex::MFIter mfi(rows); mfi.isValid(); ++mfi) {
        auto const expected = rows.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            expected(i, j, k) = energy + ((j == 0 || j == 31) ? correction : 0);
        });
    }
    near(difference(action, rows) / energy, 0, 2.e-12,
         "hard reservoir cap analytic local derivative row");
}

void
leg_root (int box) {
    Context c(box, false);
    auto o = options();
    o.dt = 0.3;
    o.conduction_theta = 1;
    o.kappa_parallel =
        number(density * PhysConst::kb / (gamma - 1)) + "*(Te/100)^2";
    for (int side = 0; side < 2; ++side) {
        o.boundary[1][side] = {BoundaryKind::Leg, 0.5};
        o.boundary[1][side].flux_limit = 1.e-7;
        o.boundary[1][side].cap_form = WallCapForm::Sonic;
        o.boundary[1][side].ion_mass = 2 * PhysConst::m_p;
    }
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto end = c.scalar(), reference = c.scalar(), residual = c.scalar();
    auto result = SolveThermalStage(stage, c.u, end, name);
    converged("leg_pc", result);
    require(end.min(0) > energy * 0.005 && end.max(0) < energy,
            "bounded physical leg cooling");
    amrex::MultiFab::Copy(reference, end, 0, 0, 1, 0);
    require(stage.Residual(c.u.getMultiFabBlock(name, 0), residual),
            "accepted leg residual");
    amrex::MultiFab::Subtract(end, c.old, 0, 0, 1, 0);
    near(integral(end, c.sim.geometry), -o.dt * boundary_power(stage), 2.e-9,
         "leg extensive RZ heat balance and no interior double count");
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    ThermalSolveOptions solver;
    solver.use_preconditioner = false;
    result = SolveThermalStage(stage, c.u, end, name, solver);
    converged("leg_no_pc", result);
    near(difference(reference, end) / energy, 0, 5.e-10,
         "leg PC-on/off matching roots");
}

void
manufactured (int box) {
    Real previous = 0;
    for (int nz : {16, 32, 64}) {
        Context c(box, true, 16, nz);
        FaceFlow flow(c, 0, 0.7);
        auto o = options();
        o.dt = 1.e-4;
        Real const dz = 1. / nz, wave = 2 * MathConst::pi;
        for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
            auto const u = c.old.array(mfi), source = c.source.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Real const z = (j + 0.5) * dz;
                    u(i, j, k) = energy * (1 + 0.2 * std::cos(wave * z));
                    source(i, j, k) = 0.2 * energy *
                                      (wave * wave * std::cos(wave * z) -
                                       0.7 * wave * std::sin(wave * z));
                });
        }
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                                   c.source, flow.context, o);
        auto residual = c.scalar();
        require(stage.Residual(c.old, residual), "spatial manufactured stage");
        Real const defect = residual.norm0(0) / (o.theta * o.dt * energy);
        amrex::Print() << "MANUFACTURED spatial nz=" << nz
                       << " defect=" << defect << '\n';
        if (previous > 0) {
            require(previous / defect > 3.8,
                    "second-order central transport/conduction spatial "
                    "consistency");
        }
        previous = defect;
    }
    for (Real theta_c : {Real(0.5), Real(1)}) {
        previous = 0;
        for (Real dt : {Real(0.004), Real(0.002), Real(0.001)}) {
            Context c(box, true);
            FaceFlow flow(c, 0, 0.7);
            auto o = options();
            o.dt = dt;
            o.conduction_theta = theta_c;
            Real const dz = c.sim.geometry.CellSize(1),
                       wave = 2 * MathConst::pi;
            Real const lambda =
                4 * std::pow(std::sin(0.5 * wave * dz), 2) / (dz * dz);
            Real const omega = 0.7 * std::sin(wave * dz) / dz;
            auto trial = c.scalar(), residual = c.scalar();
            for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
                auto const old = c.old.array(mfi), u = trial.array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Real const phase = wave * (j + 0.5) * dz;
                        Real const un = energy * (1 + 0.2 * std::cos(phase));
                        Real const end =
                            energy * (1 + 0.2 * std::exp(-lambda * dt) *
                                              std::cos(phase - omega * dt));
                        old(i, j, k) = un;
                        u(i, j, k) = 0.5 * (un + end);
                    });
            }
            EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n,
                                       c.b, c.source, flow.context, o);
            require(stage.Residual(trial, residual),
                    "temporal manufactured stage");
            Real const defect = residual.norm0(0) / (energy * dt);
            amrex::Print() << "MANUFACTURED temporal theta_c=" << theta_c
                           << " dt=" << dt << " defect=" << defect << '\n';
            if (previous > 0) {
                require(previous / defect > (theta_c == 0.5 ? 3.6 : 1.8),
                        "midpoint second order / overcentered conduction first "
                        "order");
            }
            previous = defect;
        }
    }
}

void
mg_controls (int box) {
    Context c(box, true, 16, 32, 0.125);
    auto o = options();
    o.dt = 0.002;
    o.kappa_perpendicular = "1.e-8";
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.b, c.source, o);
    auto direction = c.scalar(), out = c.scalar(), rows = c.scalar(),
         solved = c.scalar();
    for (amrex::MFIter mfi(direction); mfi.isValid(); ++mfi) {
        auto const q = direction.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                q(i, j, k) = std::cos(2 * MathConst::pi * (j + 0.5) / 32.) +
                             0.2 * std::cos(MathConst::pi * (i + 0.5) / 16.);
            });
    }
    for (int mode = 0; mode < 3; ++mode) {
        ThermalPCOptions pc_options;
        pc_options.semicoarsening = mode > 0;
        pc_options.max_semicoarsening_level = mode > 0 ? 3 : 0;
        pc_options.agglomeration = mode != 2;
        pc_options.consolidation = mode != 2;
        ThermalConductionPC pc(stage, pc_options);
        require(pc.Freeze(c.old), "anisotropic mesh PC freeze");
        pc.ApplyOperator(out, direction);
        pc.ApplyRows(rows, direction);
        near(difference(out, rows), 0, 3.e-13,
             "semicoarsening preserves emitted fine-grid row");
        pc.Apply(solved, out);
        Real const error = difference(solved, direction);
        pc.ApplyOperator(rows, solved);
        Real const residual_ratio = difference(rows, out) / out.norm0(0);
        amrex::Print() << "MG_CONTROL mode=" << mode
                       << " two_cycle_error=" << error
                       << " residual_contraction=" << residual_ratio << '\n';
        require(
            std::isfinite(error) && residual_ratio < 1,
            "configured fixed-cycle MLMG reduces the actual operator residual");
    }
}
