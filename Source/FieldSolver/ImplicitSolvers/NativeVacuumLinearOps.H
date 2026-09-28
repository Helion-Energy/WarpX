/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Distributed native-action solve. No assembled matrix or host field copies.
#include "NativeVacuumGraph.H"
struct JoinedVector
{
    Field edge;
    MF node;
    JoinedVector(View const &shape, MF const &scalar) : edge(Clone(shape))
    {
        node.define(scalar.boxArray(), scalar.DistributionMap(), 1, 1);
        Zero(V(edge));
        node.setVal(0.);
    }
};
struct JoinedMatrixFree
{
    using RT = amrex::Real;
    WarpX &w;
    warpx::darwin::NativeInertiaSupport &support;
    Core &core;
    Core &all;
    NativeVacuumCircuitFieldRate *circuit;
    amrex::Array<amrex::iMultiFab, 3> const &trace;
    JoinedGraph graph;
    Field shape, field, curl, total, rate, current, ea, ba, mass, gradient, origin, reference;
    // Owned frozen native force pair. No live parser/source or registry read
    // is performed by an affine or homogeneous electric response action.
    std::unique_ptr<Field> full_force, nores_force;
    MF potential, divergence;
    double scale_si = 0., length = 0., edge_measure = 1., volume = 1.;
    int actions = 0;
    bool failed = false;
    JoinedMatrixFree(WarpX &warp, warpx::darwin::NativeInertiaSupport &sup, Core &recovery,
                     Core &global, amrex::Array<amrex::iMultiFab, 3> const &mask, View const &ref,
                     View const *full = nullptr, View const *nores = nullptr,
                     NativeVacuumCircuitFieldRate *native_circuit = nullptr)
        : w(warp), support(sup), core(recovery), all(global), circuit(native_circuit), trace(mask),
          graph(w, trace, support.NodeWeights()),
          shape(Clone(w.m_fields.get_alldirs(FT::Efield_fp, 0))), field(Clone(V(shape))),
          curl(Clone(V(shape))), total(Clone(V(shape))),
          rate(Clone(w.m_fields.get_alldirs(FT::current_fp, 0))), current(Clone(V(rate))),
          ea(Clone(w.m_fields.get_alldirs(FT::Efield_aux, 0))),
          ba(Clone(w.m_fields.get_alldirs(FT::Bfield_aux, 0))), mass(Clone(V(shape))),
          gradient(Clone(V(shape))), origin(Clone(V(shape))), reference(Clone(ref))
    {
        AMREX_ALWAYS_ASSERT((full == nullptr) == (nores == nullptr));
        if (full)
        {
            full_force = std::make_unique<Field>(Clone(*full));
            nores_force = std::make_unique<Field>(Clone(*nores));
        }
        w.InterpolateLevelZeroFieldToAux(V(ba), w.m_fields.get_alldirs(FT::Bfield_fp, 0));
        auto const &weights = support.NodeWeights();
        potential.define(weights.boxArray(), weights.DistributionMap(), 1, 1);
        divergence.define(weights.boxArray(), weights.DistributionMap(), 1, 0);
        auto const &model = *w.get_pointer_HybridPICModel();
        scale_si =
            model.m_electron_inertia_mass / (PhysConst::q_e * PhysConst::q_e * model.m_n0_ref);
        auto const &g = w.Geom(0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            length = std::max(length, g.ProbHi(d) - g.ProbLo(d));
            volume *= g.ProbHi(d) - g.ProbLo(d);
            edge_measure *= g.CellSize(d);
        }
#if defined(WARPX_DIM_RZ)
        double const pi = std::acos(-1.);
        volume = pi * (g.ProbHi(0) * g.ProbHi(0) - g.ProbLo(0) * g.ProbLo(0)) *
                 (g.ProbHi(1) - g.ProbLo(1));
        edge_measure *= 2. * pi * g.CellSize(0);
#endif
    }
    JoinedVector makeVecLHS() { return JoinedVector(V(shape), support.NodeWeights()); }
    JoinedVector makeVecRHS() { return makeVecLHS(); }
    static View ViewOf(JoinedVector const &x)
    {
        return {const_cast<MF *>(&x.edge[0]), const_cast<MF *>(&x.edge[1]),
                const_cast<MF *>(&x.edge[2])};
    }
    void setToZero(JoinedVector &x)
    {
        Zero(V(x.edge));
        x.node.setVal(0.);
    }
    void assign(JoinedVector &a, JoinedVector const &b)
    {
        Copy(V(a.edge), ViewOf(b));
        MF::Copy(a.node, b.node, 0, 0, 1, 1);
    }
    void scale(JoinedVector &a, double f)
    {
        for (auto &x : a.edge)
            x.mult(f, 0, 1, 0);
        a.node.mult(f, 0, 1, 0);
    }
    void increment(JoinedVector &a, JoinedVector const &b, double f)
    {
        for (int c = 0; c < 3; ++c)
            MF::Saxpy(a.edge[c], f, b.edge[c], 0, 0, 1, 0);
        MF::Saxpy(a.node, f, b.node, 0, 0, 1, 0);
    }
    void linComb(JoinedVector &a, double f, JoinedVector const &b, double g, JoinedVector const &c)
    {
        for (int d = 0; d < 3; ++d)
            MF::LinComb(a.edge[d], f, b.edge[d], 0, g, c.edge[d], 0, 0, 1, 0);
        MF::LinComb(a.node, f, b.node, 0, g, c.node, 0, 0, 1, 0);
    }
    double dotProduct(JoinedVector const &a, JoinedVector const &b)
    {
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<double> data(op);
        using T = decltype(data)::Type;
        for (amrex::MFIter mfi(a.node); mfi.isValid(); ++mfi)
        {
            auto x = a.node.const_array(mfi), y = b.node.const_array(mfi),
                 v = support.NodeWeights().const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    { return {v(i, j, k) * x(i, j, k) * y(i, j, k)}; });
        }
        double scalar = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealSum(scalar);
        return (scalar + edge_measure * all.MetricDot(ViewOf(a), ViewOf(b))) / volume;
    }
    double norm2(JoinedVector const &a)
    {
        if (failed) return std::numeric_limits<double>::quiet_NaN();
        double const square = dotProduct(a, a);
        return std::isfinite(square) ? std::sqrt(std::max(0., square)) : square;
    }
    void precond(JoinedVector &a, JoinedVector const &b) { assign(a, b); }
    void RestrictEdges(View const &x) const
    {
        for (int c = 0; c < 3; ++c)
            for (amrex::MFIter mfi(*x[c]); mfi.isValid(); ++mfi)
            {
                auto a = x[c]->array(mfi);
                auto t = trace[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       if (!t(i, j, k))
                                           a(i, j, k) = 0.;
                                   });
            }
    }
    // The helper is applied to the base recovery exactly once. Its current
    // response stays paired with this field through gather, mass and divergence.
    bool Complete(View const &electric, bool affine)
    {
        if (!circuit) return true;
        circuit->Apply(electric, affine);
        for (int c = 0; c < 3; ++c)
        {
            MF::Copy(*electric[c], circuit->ElectricField(c), 0, 0, 1,
                     electric[c]->nGrowVect());
            MF::Copy(reference[c], circuit->ElectricReference(c), 0, 0, 1,
                     reference[c].nGrowVect());
        }
        for (int c = 0; c < 3; ++c)
        {
            bool const finite = electric[c]->is_finite(0, 1, electric[c]->nGrowVect()) &&
                reference[c].is_finite(0, 1, reference[c].nGrowVect()) &&
                circuit->CurrentRate(c).is_finite(0, 1, circuit->CurrentRate(c).nGrowVect());
            failed = !finite || failed;
        }
        return !failed;
    }
    void ElectricRate(View const &e, bool linear)
    {
        Copy(V(total), e);
        if (!linear && full_force)
            warpx::darwin::ApplyNativeInstantaneousForce(
                V(total), V(*full_force), V(*nores_force), true);
        auto ref = V(reference);
        VectorImages(w, V(total), true, (linear && !circuit) ? nullptr : &ref);
        w.InterpolateLevelZeroFieldToAux(V(ea), V(total));
        warpx::particles::EvaluateNativeInstantaneousIonRate(w, V(ea), V(ba), V(current), V(rate),
                                                             linear);
    }
    void ApplyRateMass(View const &e, bool affine)
    {
        if (circuit)
        {
            for (int c = 0; c < 3; ++c)
                MF::Copy(curl[c], circuit->CurrentRate(c), 0, 0, 1, 0);
        }
        else
        {
            all.ApplyMaskedCurl(V(curl), e, affine);
            for (int c = 0; c < 3; ++c)
                curl[c].mult(-all.Diagonal() / PhysConst::mu0, 0, 1, 0);
        }
        for (int c = 0; c < 3; ++c)
            MF::Subtract(curl[c], rate[c], 0, 0, 1, 0);
        support.ApplyMass({&curl[0], &curl[1], &curl[2]}, V(mass), scale_si);
    }
    void Div(View const &e, MF &out, bool affine)
    {
        Copy(V(total), e);
        auto ref = V(reference);
        VectorImages(w, V(total), false, (affine || circuit) ? &ref : nullptr);
        w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(total), out);
    }
    void apply(JoinedVector &out, JoinedVector const &in)
    {
        ++actions;
        if (!core.Response(V(field), ViewOf(in)) || !Complete(V(field), false))
        {
            failed = true;
            return;
        }
        ElectricRate(V(field), true);
        ApplyRateMass(V(field), false);
        MF::Copy(potential, in.node, 0, 0, 1, 0);
        potential.mult(length, 0, 1, 0);
        potential.OverrideSync(w.Geom(0).periodicity());
        potential.FillBoundary(w.Geom(0).periodicity());
        Gradient(w, potential, V(gradient));
        support.ApplyMass({&gradient[0], &gradient[1], &gradient[2]}, V(curl));
        for (int c = 0; c < 3; ++c)
        {
            MF::LinComb(out.edge[c], 1., field[c], 0, -1., mass[c], 0, 0, 1, 0);
            MF::Add(out.edge[c], curl[c], 0, 0, 1, 0);
        }
        RestrictEdges(V(out.edge));
        Div(V(field), out.node, false);
        out.node.mult(length, 0, 1, 0);
        if (!graph.ProjectAndRestrict(out.node, support.NodeWeights())) failed = true;
    }
    bool RightHandSide(JoinedVector &rhs, View const &f0, View const &ell, double &compatibility)
    {
        if (!core.Recover(V(origin)) || !Complete(V(origin), true))
            return false;
        for (int c = 0; c < 3; ++c)
            MF::LinComb(field[c], 1., origin[c], 0, 1., *ell[c], 0, 0, 1, 0);
        ElectricRate(V(field), false);
        ApplyRateMass(V(origin), true);
        for (int c = 0; c < 3; ++c)
        {
            MF::LinComb(rhs.edge[c], 1., *f0[c], 0, -1., *ell[c], 0, 0, 1, 0);
            MF::Subtract(rhs.edge[c], origin[c], 0, 0, 1, 0);
            MF::Add(rhs.edge[c], mass[c], 0, 0, 1, 0);
        }
        RestrictEdges(V(rhs.edge));
        Div(V(origin), rhs.node, true);
        rhs.node.mult(-length, 0, 1, 0);
        compatibility = graph.Compatibility(rhs.node, support.NodeWeights());
        if (!std::isfinite(compatibility) || compatibility > 1.e-9)
            return false;
        return graph.ProjectAndRestrict(rhs.node, support.NodeWeights());
    }
    double FullDivergenceError(View const &e)
    {
        Div(e, divergence, true);
        if (!divergence.is_finite(0, 1, 0))
        {
            failed = true;
            return std::numeric_limits<double>::quiet_NaN();
        }
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<double> data(op);
        using T = decltype(data)::Type;
        for (amrex::MFIter mfi(divergence); mfi.isValid(); ++mfi)
        {
            auto x = divergence.const_array(mfi), v = support.NodeWeights().const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    { return {v(i, j, k) > 0. ? std::abs(x(i, j, k)) : 0.}; });
        }
        double emax = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(emax);
        return length * emax;
    }
};
