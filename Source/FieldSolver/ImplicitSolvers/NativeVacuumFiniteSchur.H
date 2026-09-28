/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Private frozen displacement part of the mixed outer Jacobian.
struct VacuumFiniteSchur
{
    using RT = amrex::Real;
    WarpX &w;
    NativeInertiaSupport &support;
    amrex::Array<amrex::iMultiFab, 3> const &trace;
    JoinedGraph graph;
    MF potential;
    Field gradient, flux;
    double beta;
    int actions = 0;
    VacuumFiniteSchur(WarpX &warp, NativeInertiaSupport &sup,
                      amrex::Array<amrex::iMultiFab, 3> const &p, double coefficient)
        : w(warp), support(sup), trace(p), graph(w, p, sup.NodeWeights()),
          potential(sup.NodeWeights().boxArray(), sup.NodeWeights().DistributionMap(), 1, 1),
          gradient(Clone(w.m_fields.get_alldirs(FT::Efield_fp, 0))), flux(Clone(V(gradient))),
          beta(coefficient)
    {
    }
    MF makeVecLHS() const { return MF(potential.boxArray(), potential.DistributionMap(), 1, 1); }
    MF makeVecRHS() const { return makeVecLHS(); }
    void setToZero(MF &x) const { x.setVal(0.); }
    void assign(MF &x, MF const &y) const { MF::Copy(x, y, 0, 0, 1, 0); }
    void scale(MF &x, double a) const { x.mult(a, 0, 1, 0); }
    void increment(MF &x, MF const &y, double a) const { MF::Saxpy(x, a, y, 0, 0, 1, 0); }
    void linComb(MF &z, double a, MF const &x, double b, MF const &y) const
    {
        MF::LinComb(z, a, x, 0, b, y, 0, 0, 1, 0);
    }
    double dotProduct(MF const &x, MF const &y) const
    {
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<double> data(op);
        using T = decltype(data)::Type;
        for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi)
        {
            auto a = x.const_array(mfi), b = y.const_array(mfi),
                 v = support.NodeWeights().const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    { return {v(i, j, k) * a(i, j, k) * b(i, j, k)}; });
        }
        double value = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealSum(value);
        return value;
    }
    double norm2(MF const &x) const { return std::sqrt(std::max(0., dotProduct(x, x))); }
    void precond(MF &x, MF const &b) const { assign(x, b); }
    void Canonical(MF &x) const
    {
        graph.Restrict(x);
        x.setBndry(0.);
        x.OverrideSync(w.Geom(0).periodicity());
        x.FillBoundary(w.Geom(0).periodicity());
    }
    void Product(View const &g, View const &out)
    {
        support.ApplyMass({g[0], g[1], g[2]}, out, beta);
        for (int c = 0; c < 3; ++c)
            for (amrex::MFIter mfi(*out[c]); mfi.isValid(); ++mfi)
            {
                auto a = out[c]->array(mfi);
                auto b = g[c]->const_array(mfi);
                auto t = trace[c].const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   { a(i, j, k) = t(i, j, k) ? a(i, j, k) + b(i, j, k) : 0.; });
            }
        VectorImages(w, out, false);
    }
    void FullAction(MF &out, MF const &x)
    {
        MF::Copy(potential, x, 0, 0, 1, 0);
        Canonical(potential);
        Gradient(w, potential, V(gradient));
        Product(V(gradient), V(flux));
        w.get_pointer_fdtd_solver_fp(0)->ComputeDivE(V(flux), out);
        out.mult(-1., 0, 1, 0);
    }
    void apply(MF &out, MF const &x)
    {
        ++actions;
        FullAction(out, x);
        graph.Restrict(out);
    }
};
