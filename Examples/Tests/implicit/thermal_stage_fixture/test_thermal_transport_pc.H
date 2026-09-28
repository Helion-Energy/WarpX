/* Production frozen central transport PC qualification. */
void
transport_pc_omission (int box) {
    amrex::ParmParse pp("test");
    int nz = 128;
    Real cfl = 32;
    pp.query("nz", nz);
    pp.query("cfl", cfl);
    Context c(box, true, 4, nz);
    FaceFlow flow(c, 0, 1);
    auto o = options();
    o.dt = 2 * cfl / nz;
    o.kappa_parallel = number(1.e-4 * density * PhysConst::kb / (gamma - 1));
    for (amrex::MFIter mfi(c.old); mfi.isValid(); ++mfi) {
        auto const u = c.old.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const z = (j + .5) / nz;
                u(i, j, k) =
                    energy * (1 + .2 * std::exp(-800 * (z - .4) * (z - .4)));
            });
    }
    amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                               c.source, flow.context, o);
    auto end = c.scalar(), saved = c.scalar(), v = c.scalar(),
         action = c.scalar(), probe = c.scalar(), plus = c.scalar(),
         minus = c.scalar(), fp = c.scalar(), fm = c.scalar();
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
        auto const a = v.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                a(i, j, k) = energy * (.1 + std::sin(Real(2 * i + 3 * j)));
            });
    }
    ThermalConductionPC pc(stage);
    require(pc.Freeze(c.old), "omission base freeze");
    pc.ApplyOperator(action, v);
    Real const eps = 1.e-4;
    amrex::MultiFab::LinComb(plus, 1, c.old, 0, eps, v, 0, 0, 1, 0);
    amrex::MultiFab::LinComb(minus, 1, c.old, 0, -eps, v, 0, 0, 1, 0);
    require(stage.Residual(plus, fp) && stage.Residual(minus, fm),
            "omission Jacobian probes");
    amrex::MultiFab::LinComb(probe, .5 / eps, fp, 0, -.5 / eps, fm, 0, 0, 1, 0);
    amrex::Print() << "OMISSION nz=" << nz << " cfl=" << cfl
                   << " relative_action_gap="
                   << difference(action, probe) / probe.norm0(0) << '\n';
    ThermalSolveOptions solver;
    solver.max_linear_iterations = 1200;
    for (bool enabled : {true, false}) {
        solver.use_preconditioner = enabled;
        amrex::MultiFab::Copy(c.u.getMultiFabBlock(name, 0), c.old, 0, 0, 1, 0);
        auto const start = amrex::second();
        auto result = SolveThermalStage(stage, c.u, end, name, solver);
        converged(enabled ? "omission_pc" : "omission_no_pc", result);
        amrex::Print() << "OMISSION time=" << amrex::second() - start << '\n';
        if (enabled)
            amrex::MultiFab::Copy(saved, end, 0, 0, 1, 0);
        else
            near(difference(saved, end) / energy, 0, 2.e-9,
                 "transport roots agree at matched tolerances");
    }
}

struct FrozenThermalOps {
    using RT = Real;
    Context const& c;
    ThermalConductionPC& target;
    ThermalConductionPC* inverse;
    amrex::MultiFab
    makeVecRHS () const {
        return c.scalar();
    }
    amrex::MultiFab
    makeVecLHS () const {
        return c.scalar();
    }
    void
    setToZero (amrex::MultiFab& x) const {
        x.setVal(0);
    }
    void
    assign (amrex::MultiFab& x, amrex::MultiFab const& y) const {
        amrex::MultiFab::Copy(x, y, 0, 0, 1, 0);
    }
    void
    scale (amrex::MultiFab& x, Real a) const {
        x.mult(a, 0, 1, 0);
    }
    void
    increment (amrex::MultiFab& x, amrex::MultiFab const& y, Real a) const {
        amrex::MultiFab::Saxpy(x, a, y, 0, 0, 1, 0);
    }
    void
    linComb (amrex::MultiFab& x, Real a, amrex::MultiFab const& y, Real b,
             amrex::MultiFab const& z) const {
        amrex::MultiFab::LinComb(x, a, y, 0, b, z, 0, 0, 1, 0);
    }
    Real
    dotProduct (amrex::MultiFab const& x, amrex::MultiFab const& y) const {
        return amrex::MultiFab::Dot(x, 0, y, 0, 1, 0);
    }
    Real
    norm2 (amrex::MultiFab const& x) const {
        return std::sqrt(dotProduct(x, x));
    }
    void
    apply (amrex::MultiFab& x, amrex::MultiFab const& y) const {
        target.ApplyOperator(x, y);
    }
    void
    precond (amrex::MultiFab& x, amrex::MultiFab const& y) const {
        if (inverse)
            inverse->Apply(x, y);
        else
            assign(x, y);
    }
};
void
print_pc_diagnostics (ThermalConductionPC const& pc) {
    auto const& d = pc.Diagnostics();
    amrex::Print() << "PC_WORK setups=" << d.setups << " applies=" << d.applies
                   << " operator_applies=" << d.operator_applies
                   << " mlmg_solves=" << d.mlmg_solves
                   << " inner_iterations=" << d.inner_iterations
                   << " reductions=" << d.reductions
                   << " breakdowns=" << d.breakdowns
                   << " setup_seconds=" << d.setup_seconds
                   << " freeze_seconds=" << d.freeze_seconds
                   << " apply_seconds=" << d.apply_seconds
                   << " last_status=" << d.last_inner_status
                   << " last_relative_residual="
                   << d.last_inner_relative_residual
                   << " min_transport_reaction=" << d.minimum_transport_reaction
                   << '\n';
}

void
transport_pc_rows (int box) {
    for (bool periodic : {true, false}) {
        Context c(box, periodic);
        c.set_pattern(.02, false, true);
        FaceFlow flow(c, .03, periodic ? .7 : -.4, !periodic);
        if (periodic) {
            flow.velocity[0]->setVal(0);
            c.n.FillBoundary(c.sim.geometry.periodicity());
            // Exercise the native conservative number-flux path too. Freeze
            // must copy the emitted u, including its exact nface denominator.
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                flow.context.number_flux[d] = flow.velocity[d].get();
                for (amrex::MFIter mfi(*flow.velocity[d]); mfi.isValid();
                     ++mfi) {
                    auto const n = c.n.const_array(mfi);
                    auto const f = flow.velocity[d]->array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            if (d == 1)
                                f(i, j, k) *=
                                    .5 * (n(i, j, k) + n(i, j - 1, k));
                        });
                }
            }
        }
        auto o = options();
        o.dt = .2;
        o.boundary[0][1].inflow_temperature_ev = temperature;
        o.boundary[1][1].inflow_temperature_ev = temperature;
        EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                                   c.source, flow.context, o);
        stage.SetSourceEvaluator([] (auto const& u, auto const& n, auto const&,
                                     auto& q, auto& derivative) {
            for (amrex::MFIter mfi(q); mfi.isValid(); ++mfi) {
                auto const a = u.const_array(mfi), nn = n.const_array(mfi);
                auto const b = q.array(mfi), d = derivative.array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        d(i, j, k) = .001 * density / nn(i, j, k);
                        b(i, j, k) = d(i, j, k) * a(i, j, k);
                    });
            }
            return true;
        });
        ThermalConductionPC pc(stage);
        require(pc.Freeze(c.old), "transport rows freeze");
        auto v = c.scalar(), action = c.scalar(), rows = c.scalar(),
             plus = c.scalar(), minus = c.scalar(), fp = c.scalar(),
             fm = c.scalar(), inverse = c.scalar(), inverse_again = c.scalar(),
             changed_n = c.scalar();
        for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
            auto const a = v.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    a(i, j, k) = energy * (.1 + std::sin(Real(2 * i + 3 * j)));
                });
        }
        pc.ApplyOperator(action, v);
        pc.ApplyRows(rows, v);
        near(difference(action, rows) / action.norm0(0), 0, 3.e-13,
             "central emitted rows match face action");
        Real const eps = 1.e-4;
        amrex::MultiFab::LinComb(plus, 1, c.old, 0, eps, v, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(minus, 1, c.old, 0, -eps, v, 0, 0, 1, 0);
        require(stage.Residual(plus, fp) && stage.Residual(minus, fm),
                "frozen transport Jacobian probes");
        amrex::MultiFab::LinComb(rows, .5 / eps, fp, 0, -.5 / eps, fm, 0, 0, 1,
                                 0);
        near(difference(action, rows) / action.norm0(0), 0, 2.e-7,
             "central/compression/conduction match live Jv");
        amrex::MultiFab::Copy(rows, v, 0, 0, 1, 0);
        pc.ApplyOperator(rows, rows);
        near(difference(action, rows), 0, 0,
             "in-place frozen transport action");
        amrex::MultiFab::Copy(rows, v, 0, 0, 1, 0);
        pc.ApplyRows(rows, rows);
        near(difference(action, rows) / action.norm0(0), 0, 3.e-13,
             "in-place frozen row action");
        pc.Apply(inverse, v);
        auto const before = snapshot(inverse);
        amrex::MultiFab::Copy(rows, v, 0, 0, 1, 0);
        pc.Apply(rows, rows);
        near(difference(inverse, rows), 0, 0,
             "in-place bounded transport inverse");
        amrex::MultiFab::Copy(changed_n, c.n, 0, 0, 1, 0);
        changed_n.mult(2, 0, 1, 0);
        c.b.setVal(2);
        c.source.setVal(1);
        for (auto& u : flow.velocity)
            u->mult(-2, 0, 1, 0);
        // Both sides have explicit inflow values when the velocity reverses.
        if (!periodic) {
            flow.velocity[0]->mult(-1, 0, 1, 0);
            flow.velocity[1]->mult(-1, 0, 1, 0);
        }
        require(stage.RefreshContext(changed_n, changed_n, c.b, c.source,
                                     flow.context),
                "changed live transport context");
        require(stage.Residual(c.old, fp),
                "live context probe after PC freeze");
        pc.ApplyOperator(rows, v);
        near(difference(action, rows), 0, 0,
             "transport target freezes n/B/Q/u/divu");
        pc.ApplyRows(rows, v);
        near(difference(action, rows) / action.norm0(0), 0, 3.e-13,
             "frozen transport rows after context refresh");
        pc.Apply(inverse_again, v);
        require(snapshot(inverse_again) == before,
                "inner inverse unchanged by live context refresh");
        require(pc.Diagnostics().inner_iterations > 0,
                "production PC exercises inner FGMRES");
    }
}

void
transport_pc_roots (int box, bool compression) {
    int nz = compression ? 32 : 128;
    Context c(box, true, 16, nz);
    auto o = options();
    o.theta = 1;
    o.conduction_theta = 1;
    o.dt = compression ? 1 : .25;
    o.kappa_parallel = number(1.e-4 * density * PhysConst::kb / (gamma - 1));
    o.boundary[0][1].inflow_temperature_ev = temperature;
    FaceFlow flow(c, compression ? -1 : 0, compression ? 0 : 1, compression);
    auto expected = c.scalar(), residual = c.scalar(), rhs = c.scalar(),
         direction = c.scalar(), root = c.scalar(), saved = c.scalar();
    for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
        auto const a = expected.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real const r = (i + .5) / 16, z = (j + .5) / nz;
            a(i, j, k) = energy * (1 + .03 * std::cos(3 * MathConst::pi * r) +
                                   .03 * std::exp(-800 * (z - .4) * (z - .4)));
        });
    }
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                               c.source, flow.context, o);
    require(stage.Residual(expected, residual),
            "manufactured transport target");
    amrex::MultiFab::Copy(c.source, residual, 0, 0, 1, 0);
    c.source.mult(1 / o.dt, 0, 1, 0);
    require(stage.RefreshContext(c.n, c.n, c.b, c.source, flow.context),
            "manufactured forcing");
    require(stage.Residual(expected, residual), "manufactured forced residual");
    near(residual.norm0(0) / energy, 0, 1.e-13,
         "exact manufactured physical root");
    require(stage.Residual(c.old, rhs), "initial transport residual");
    rhs.mult(-1, 0, 1, 0);
    ThermalConductionPC target(stage);
    require(target.Freeze(c.old), "physical frozen target");
    amrex::ParmParse pp("test");
    // The deliberately indefinite compression control needs a larger outer
    // space for all three inverses. Tolerances and iteration caps stay equal.
    // Restart60/128 stagnation of the omission control is retained evidence.
    int inner = 32, restart = compression ? 512 : 60;
    Real stabilization = 0, tolerance = .1;
    pp.query("inner", inner);
    pp.query("outer_restart", restart);
    pp.query("stabilization", stabilization);
    pp.query("inner_tolerance", tolerance);
    for (int mode = 0; mode < 3; ++mode) {
        ThermalPCOptions options;
        options.transport = mode == 0;
        options.transport_inner_max_iterations = inner;
        options.transport_inner_relative_tolerance = tolerance;
        options.transport_upwind_stabilization = stabilization;
        ThermalConductionPC inverse(stage, options);
        require(inverse.Freeze(c.old), "comparison PC freeze");
        FrozenThermalOps ops{c, target, mode == 2 ? nullptr : &inverse};
        FlexibleGMRES<amrex::MultiFab, FrozenThermalOps> solve;
        solve.define(ops);
        solve.setMaxIters(2400);
        solve.setRestartLength(restart);
        auto const start = amrex::second();
        solve.solve(direction, rhs, 1.e-10, 1.e-12);
        amrex::Print() << "TRANSPORT_ROOT compression=" << compression
                       << " mode=" << mode << " restart=" << restart
                       << " status=" << solve.getStatus()
                       << " outer=" << solve.getNumIters()
                       << " residual=" << solve.getResidualNorm()
                       << " seconds=" << amrex::second() - start << '\n';
        print_pc_diagnostics(inverse);
        require(solve.getStatus() == 0,
                "same frozen target converges for all inverses");
        amrex::MultiFab::LinComb(root, 1, c.old, 0, 1, direction, 0, 0, 1, 0);
        require(stage.Residual(root, residual), "production root admissible");
        near(residual.norm0(0) / energy, 0, 1.e-9,
             "physical residual closes after transport inverse");
        near(difference(root, expected) / energy, 0, 1.e-8,
             "transport target root recovered");
        if (mode == 0)
            amrex::MultiFab::Copy(saved, root, 0, 0, 1, 0);
        else
            near(difference(root, saved) / energy, 0, 1.e-8,
                 "transport PC-on/off same root");
        if (compression && mode == 0)
            require(inverse.Diagnostics().minimum_transport_reaction < 0,
                    "negative signed compression reaction retained");
    }
}

void
transport_pc_breakdown (int box) {
    Context c(box, true);
    FaceFlow flow(c, 0, 1);
    auto o = options();
    o.theta = 1;
    o.conduction_theta = 1;
    o.dt = 1;
    o.kappa_parallel = "0";
    EulerianThermalStage stage(c.sim.geometry, c.old, c.n, c.n, c.n, c.b,
                               c.source, flow.context, o);
    stage.SetSourceEvaluator([] (auto const& u, auto const&, auto const&,
                                 auto& q, auto& derivative) {
        for (amrex::MFIter mfi(q); mfi.isValid(); ++mfi) {
            auto const a = u.const_array(mfi);
            auto const b = q.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   b(i, j, k) = a(i, j, k) - energy + 1;
                               });
        }
        derivative.setVal(1);
        return true;
    });
    ThermalConductionPC pc(stage);
    require(pc.Freeze(c.old), "singular transport target freeze");
    auto rhs = c.scalar(), out = c.scalar(), action = c.scalar();
    rhs.setVal(1);
    pc.ApplyOperator(action, rhs);
    near(action.norm0(0), 0, 0,
         "signed target retains true constant nullspace");
    pc.Apply(out, rhs);
    print_pc_diagnostics(pc);
    require(pc.Diagnostics().last_inner_status == 2 &&
                pc.Diagnostics().breakdowns == 1,
            "singular inner inverse reports breakdown");
    near(out.norm0(0), 0, 0,
         "breakdown yields zero correction for outer handling");
    // A one-cell radial inflow has fixed advective inflow data. With gamma=2,
    // u_r=-r/2, h=1 the compression row is exactly -I, so the true energy
    // Jacobian is singular independently of any source approximation.
    Context compressed(box, true, 1, 1);
    FaceFlow compressing(compressed, -.5, 0, true);
    o.gamma = 2;
    o.boundary[0][1].inflow_temperature_ev = temperature;
    EulerianThermalStage singular(
        compressed.sim.geometry, compressed.old, compressed.n, compressed.n,
        compressed.n, compressed.b, compressed.source, compressing.context, o);
    ThermalConductionPC compressive_pc(singular);
    require(compressive_pc.Freeze(compressed.old),
            "singular compression freeze");
    auto crhs = compressed.scalar(), cout = compressed.scalar(),
         caction = compressed.scalar();
    crhs.setVal(1);
    compressive_pc.ApplyOperator(caction, crhs);
    near(caction.norm0(0), 0, 0,
         "compression exactly cancels physical mass row");
    compressive_pc.Apply(cout, crhs);
    print_pc_diagnostics(compressive_pc);
    require(compressive_pc.Diagnostics().last_inner_status == 2,
            "compression singularity reports breakdown");
    near(cout.norm0(0), 0, 0,
         "singular compression returns no usable direction");
}
void
transport_pc (int box) {
    transport_pc_rows(box);
    transport_pc_roots(box, false);
    transport_pc_roots(box, true);
    transport_pc_breakdown(box);
}
