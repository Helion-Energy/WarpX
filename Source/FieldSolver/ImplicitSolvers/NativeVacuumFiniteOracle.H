/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Opt-in verification invoked only at completed constrained initialization.
bool VerifyNativeVacuumFiniteSchur(WarpX &w, MF const &density, int component, amrex::Real interval)
{
    MF physical(density, amrex::make_alias, component, 1);
    auto options = NativeVacuumSupportOptions(w);
    NativeInertiaSupport support(Geometry(w), w.boxArray(0), w.DistributionMap(0), options);
    AMREX_ALWAYS_ASSERT(support.FreezeEdges(physical, *w.m_fields.get("hybrid_rho_vacmask_fp", 0)));
    Core::Options co;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        co.pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
        co.pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
    }
    amrex::Array<amrex::iMultiFab, 3> trace, global, empty;
    DefineTrace(w, support, co, trace, global, empty);
    auto &model = *w.get_pointer_HybridPICModel();
    double const beta = PhysConst::epsilon_0 * model.m_electron_inertia_mass /
                        (PhysConst::q_e * PhysConst::q_e * model.m_n0_ref * interval * interval);
    VacuumFiniteSchur op(w, support, trace, beta);
    auto p = op.makeVecLHS(), a = op.makeVecLHS(), a2 = op.makeVecLHS(), b = op.makeVecLHS(),
         other = op.makeVecLHS();
    auto dx = w.Geom(0).CellSizeArray();
    auto lo = w.Geom(0).ProbLoArray();
    auto hi = w.Geom(0).ProbHiArray();
    double const pi = std::acos(-1.);
    for (amrex::MFIter mfi(p); mfi.isValid(); ++mfi)
    {
        auto x = p.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k)
                           {
                               amrex::IntVect q(AMREX_D_DECL(i, j, k));
                               double f = 1.;
                               for (int d = 0; d < AMREX_SPACEDIM; ++d)
                               {
                                   double t = (lo[d] + q[d] * dx[d] - lo[d]) / (hi[d] - lo[d]);
                                   f *= d == 0 ? std::cos(.5 * pi * t)
                                               : std::sin(2 * pi * t) + .25 * std::cos(4 * pi * t);
                               }
                               x(i, j, k) = f;
                           });
    }
    op.Canonical(p);
    op.apply(a, p);
    MF::Copy(other, p, 0, 0, 1, 0);
    other.mult(-.375, 0, 1, 0);
    op.apply(b, other);
    op.apply(a2, p);
    MF::Subtract(a2, a, 0, 0, 1, 0);
    double const aba = a2.norminf();
    auto g = Clone(V(op.gradient)), explicit_flux = Clone(V(g));
    Gradient(w, p, V(g));
    for (int c = 0; c < 3; ++c)
        for (amrex::MFIter mfi(explicit_flux[c]); mfi.isValid(); ++mfi)
        {
            auto out = explicit_flux[c].array(mfi);
            auto in = g[c].const_array(mfi);
            auto kap = support.KappaEdge(c).const_array(mfi);
            auto mask = trace[c].const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                { out(i, j, k) = mask(i, j, k) ? (1. + beta * kap(i, j, k)) * in(i, j, k) : 0.; });
        }
    VectorImages(w, V(explicit_flux), false);
    w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(explicit_flux), b);
    b.mult(-1., 0, 1, 0);
    double const manufactured_null = op.graph.Compatibility(b, support.NodeWeights());
    op.graph.Restrict(b);
    MF::Subtract(b, a, 0, 0, 1, 0);
    double const product_error = b.norminf() / std::max(1., a.norminf());
    // All-positive density reduction uses the existing frozen Schur operator.
    auto uniform = op.makeVecLHS();
    uniform.setVal(PhysConst::q_e * model.m_n0_ref);
    NativeInertiaSupport positive(Geometry(w), w.boxArray(0), w.DistributionMap(0), options);
    AMREX_ALWAYS_ASSERT(positive.FreezeEdges(uniform, uniform));
    amrex::Array<amrex::iMultiFab, 3> pt, pg, pe;
    DefineTrace(w, positive, co, pt, pg, pe);
    VacuumFiniteSchur pop(w, positive, pt, beta);
    auto pp = pop.makeVecLHS();
    MF::Copy(pp, p, 0, 0, 1, 0);
    pop.Canonical(pp);
    LongitudinalSchurOptions oldopts;
    oldopts.compatible_yee = true;
    auto translate = [](InitialRateBoundary type)
    {
        switch (type)
        {
        case InitialRateBoundary::Axis:
            return LongitudinalBoundary::Axis;
        case InitialRateBoundary::PEC:
            return LongitudinalBoundary::PEC;
        case InitialRateBoundary::PMC:
            return LongitudinalBoundary::PMC;
        default:
            return LongitudinalBoundary::Periodic;
        }
    };
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        oldopts.lower[d] = translate(options.lower[d]);
        oldopts.upper[d] = translate(options.upper[d]);
    }
    DarwinLongitudinalSchur old(w.Geom(0), w.boxArray(0), w.DistributionMap(0), oldopts);
    uniform.setVal(beta);
    AMREX_ALWAYS_ASSERT(old.Freeze(uniform));
    pop.apply(b, pp);
    old.ApplyPotential(a2, pp);
    MF::Add(a2, b, 0, 0, 1, 0);
    pop.graph.Restrict(a2);
    double const positive_error = a2.norminf() / std::max(1., b.norminf());
    // Auxiliary full-grid Schur is deliberately not the exact mixed derivative.
    // Its action uses the same masked edge coefficient plus a full DG solver term.
    auto auxiliary_coeff=Clone(V(g));
    for(int c=0;c<3;++c)for(amrex::MFIter mfi(auxiliary_coeff[c]);mfi.isValid();++mfi) {
        auto coeff=auxiliary_coeff[c].array(mfi);auto f=explicit_flux[c].array(mfi);
        auto grad=g[c].const_array(mfi),kap=support.KappaEdge(c).const_array(mfi);
        auto mask=trace[c].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            double const v=mask(i,j,k)?beta*kap(i,j,k):0.;
            coeff(i,j,k)=v;f(i,j,k)=(1.+v)*grad(i,j,k);
        });
    }
    DarwinLongitudinalSchur auxiliary_op(w.Geom(0),w.boxArray(0),w.DistributionMap(0),oldopts);
    AMREX_ALWAYS_ASSERT(auxiliary_op.FreezeEdges({&auxiliary_coeff[0],&auxiliary_coeff[1],&auxiliary_coeff[2]}));
    auxiliary_op.ApplyPotential(a2,p);
    VectorImages(w,V(explicit_flux),false);
    w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(explicit_flux),b);
    pop.graph.Restrict(b);MF::Subtract(b,a2,0,0,1,0);
    double const auxiliary_error=b.norminf()/std::max(1.,a2.norminf());
    for(auto& f:auxiliary_coeff)f.setVal(23.);
    auxiliary_op.ApplyPotential(other,p);MF::Subtract(other,a2,0,0,1,0);
    double freeze_error=other.norminf();
    auxiliary_coeff[0].setVal(-1.);
    bool const invalid_rejected=!auxiliary_op.FreezeEdges({&auxiliary_coeff[0],&auxiliary_coeff[1],&auxiliary_coeff[2]});
    auxiliary_op.ApplyPotential(other,p);MF::Subtract(other,a2,0,0,1,0);
    freeze_error=std::max(freeze_error,other.norminf());
    for(auto& f:auxiliary_coeff)f.setVal(beta);
    AMREX_ALWAYS_ASSERT(auxiliary_op.FreezeEdges({&auxiliary_coeff[0],&auxiliary_coeff[1],&auxiliary_coeff[2]}));
    auxiliary_op.ApplyPotential(a2,pp);old.ApplyPotential(b,pp);
    MF::Subtract(a2,b,0,0,1,0);
    double const positive_edge_error=a2.norminf()/std::max(1.,b.norminf());
    // Actual native full Ohm source + the actual vacuum Faraday completion.
    auto names = w.m_fields.list();
    std::vector<std::unique_ptr<MF>> copies;
    for (auto const &name : names)
    {
        auto const *f = w.m_fields.internal_get(name);
        auto x =
            std::make_unique<MF>(f->boxArray(), f->DistributionMap(), f->nComp(), f->nGrowVect());
        MF::Copy(*x, *f, 0, 0, f->nComp(), f->nGrowVect());
        copies.push_back(std::move(x));
    }
    warpx::implicit::FieldRollback saved;
    saved.Capture(w.m_fields);
    auto E = w.m_fields.get_alldirs(FT::Efield_fp, 0),
         EL = w.m_fields.get_alldirs("hybrid_E_long_fp", 0),
         Ei = w.m_fields.get_alldirs("hybrid_E_inertial_fp", 0);
    auto base_el = Clone(EL), base_ei = Clone(Ei), delta_ei = Clone(Ei), source = Clone(E);
    support.ApplyMass({&g[0], &g[1], &g[2]}, V(delta_ei), -beta);
    auto evaluate = [&](double amplitude, MF &out)
    {
        AMREX_ALWAYS_ASSERT(saved.Restore(w.m_fields));
        for (int c = 0; c < 3; ++c)
        {
            MF::LinComb(*EL[c], 1., base_el[c], 0, amplitude, g[c], 0, 0, 1, 0);
            MF::LinComb(*Ei[c], 1., base_ei[c], 0, amplitude, delta_ei[c], 0, 0, 1, 0);
        }
        VectorImages(w, EL, false);
        VectorImages(w, Ei, false);
        model.HybridPICSolveE(E, w.m_fields.get_alldirs(FT::current_fp, 0),
                              w.m_fields.get_alldirs(FT::Bfield_fp, 0), physical,
                              w.GetEBUpdateEFlag()[0], 0, false, true);
        CompleteNativeVacuumLongitudinalSource(w, interval);
        for (int c = 0; c < 3; ++c)
            MF::LinComb(source[c], 1., *E[c], 0, -1., *EL[c], 0, 0, 1, 0);
        VectorImages(w, V(source), false);
        w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(source), out);
        op.graph.Restrict(out);
    };
    std::vector<double> fd;
    for (double eps : {1.e-2, 1.e-3, 1.e-4})
    {
        evaluate(eps, a2);
        evaluate(-eps, b);
        MF::Subtract(a2, b, 0, 0, 1, 0);
        a2.mult(.5 / eps, 0, 1, 0);
        // d[D(S-EL)] = -D[P*(1+beta*kappa)*G] = op, with this sign.
        MF::Subtract(a2, a, 0, 0, 1, 0);
        fd.push_back(a2.norminf() / std::max(1., a.norminf()));
    }
    evaluate(.01, a2);
    evaluate(.0375, b);
    evaluate(.01, other);
    MF::Subtract(other, a2, 0, 0, 1, 0);
    double const source_aba = other.norminf();
    AMREX_ALWAYS_ASSERT(saved.Restore(w.m_fields));
    saved.Discard();
    double purity = 0.;
    bool same_names = names == w.m_fields.list();
    for (std::size_t n = 0; n < names.size(); ++n)
    {
        auto const *f = w.m_fields.internal_get(names[n]);
        MF::Subtract(*copies[n], *f, 0, 0, f->nComp(), f->nGrowVect());
        amrex::ReduceOps<amrex::ReduceOpMax> reduce;
        amrex::ReduceData<double> data(reduce);
        using Tuple=decltype(data)::Type;
        int const components=f->nComp();
        for(amrex::MFIter mfi(*copies[n]);mfi.isValid();++mfi) {
            auto values=copies[n]->const_array(mfi);
            reduce.eval(mfi.fabbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                double error=0.;
                for(int c=0;c<components;++c)error=amrex::max(error,std::abs(values(i,j,k,c)));
                return {error};
            });
        }
        double error=amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(error);
        purity=std::max(purity,error);
    }
    b.setVal(1.);
    double const incompatible = op.graph.Compatibility(b, support.NodeWeights());
    bool const pass = aba == 0. && source_aba == 0. && purity == 0. && same_names &&
                      product_error < 1.e-12 && positive_error < 1.e-12 && auxiliary_error < 1.e-12 && positive_edge_error < 1.e-12 &&
                      freeze_error == 0. && invalid_rejected &&
                      *std::max_element(fd.begin(), fd.end()) < 1.e-8 && incompatible > .5;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream f("NATIVE_VACUUM_FINITE_ORACLE.json");
        f << std::setprecision(17) << "{\"pass\":" << (pass ? "true" : "false")
          << ",\"beta\":" << beta << ",\"product_relative\":" << product_error
          << ",\"positive_control_relative\":" << positive_error << ",\"ABA\":" << aba
          << ",\"source_ABA\":" << source_aba << ",\"registered_purity\":" << purity
          << ",\"manufactured_null\":" << manufactured_null
          << ",\"incompatible_constant\":" << incompatible << ",\"auxiliary_relative\":" << auxiliary_error
          << ",\"positive_edge_relative\":" << positive_edge_error
          << ",\"frozen_edge_error\":" << freeze_error
          << ",\"invalid_edge_rejected\":" << (invalid_rejected?"true":"false")
          << ",\"fd_relative\":[" << fd[0] << ","
          << fd[1] << "," << fd[2] << "]}\n";
    }
    amrex::Print() << "Native finite Schur oracle pass=" << pass << " product=" << product_error
                   << " positive=" << positive_error << " fd=" << fd[0] << "/" << fd[1] << "/"
                   << fd[2] << " purity=" << purity << "\n";
    return pass;
}
