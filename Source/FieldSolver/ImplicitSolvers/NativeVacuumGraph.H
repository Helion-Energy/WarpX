/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Distributed plasma-edge connectivity and exact gauge elimination.
// Private implementation: neighboring label exchanges and scalar reductions only.
struct JoinedGraph
{
    amrex::iMultiFab labels, next, gauge;
    std::vector<int> roots;
    int iterations = 0;
    int isolated = 0;
    amrex::Geometry geometry;

    JoinedGraph(WarpX &w, amrex::Array<amrex::iMultiFab, 3> const &trace, MF const &weights)
        : geometry(w.Geom(0))
    {
        Build(w, trace, weights);
    }
    void Build(WarpX &w, amrex::Array<amrex::iMultiFab, 3> const &trace, MF const &weights)
    {
        auto const ba = weights.boxArray();
        auto const dm = weights.DistributionMap();
        labels.define(ba, dm, 1, 1);
        next.define(ba, dm, 1, 1);
        gauge.define(ba, dm, 1, 0);
        amrex::Array<amrex::iMultiFab, 3> edges;
        for (int c = 0; c < 3; ++c)
        {
            edges[c].define(trace[c].boxArray(), dm, 1, 1);
            edges[c].setVal(0);
            amrex::iMultiFab::Copy(edges[c], trace[c], 0, 0, 1, 0);
            edges[c].FillBoundary(geometry.periodicity());
        }
        auto const low = geometry.Domain().smallEnd();
        auto const high = geometry.Domain().bigEnd() + amrex::IntVect(1);
        amrex::GpuArray<int, AMREX_SPACEDIM> periodic{}, fixedlo{}, fixedhi{}, extent{};
        long long count = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            periodic[d] = geometry.isPeriodic(d);
            fixedlo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PEC;
            fixedhi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PEC;
            extent[d] = high[d] - low[d] + 1 - periodic[d];
            count *= extent[d];
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(count < std::numeric_limits<int>::max(),
                                         "Native plasma graph canonical label overflows int");
        labels.setVal(-1);
        for (amrex::MFIter mfi(labels); mfi.isValid(); ++mfi)
        {
            auto a = labels.array(mfi);
            amrex::GpuArray<amrex::Array4<int const>, AMREX_SPACEDIM> e{};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
#if defined(WARPX_DIM_RZ)
                int const c = d == 0 ? 0 : 2;
#else
                int const c = d;
#endif
                e[d] = edges[c].const_array(mfi);
            }
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::IntVect q(AMREX_D_DECL(i, j, k)), canonical = q;
                                   bool fixed = false, active = false;
                                   for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                   {
                                       fixed = fixed || (q[d] == low[d] && fixedlo[d]) ||
                                               (q[d] == high[d] && fixedhi[d]);
                                       auto off = amrex::IntVect::TheDimensionVector(d);
                                       active = active || e[d](q) != 0 || e[d](q - off) != 0;
                                       if (periodic[d] && canonical[d] == high[d])
                                           canonical[d] = low[d];
                                   }
                                   int id = 0, stride = 1;
                                   for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                   {
                                       id += (canonical[d] - low[d]) * stride;
                                       stride *= extent[d];
                                   }
                                   a(q) = fixed ? 0 : (active ? id + 1 : -1);
                               });
        }
        // All duplicate nodes see identical edge images; FillBoundary exchanges
        // labels only, never a gathered full-grid topology array.
        bool changed = true;
        while (changed)
        {
            AMREX_ALWAYS_ASSERT(iterations++ <= count);
            labels.FillBoundary(geometry.periodicity());
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<int> data(op);
            using T = decltype(data)::Type;
            for (amrex::MFIter mfi(labels); mfi.isValid(); ++mfi)
            {
                auto a = labels.const_array(mfi);
                auto b = next.array(mfi);
                amrex::GpuArray<amrex::Array4<int const>, AMREX_SPACEDIM> e{};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
#if defined(WARPX_DIM_RZ)
                    int const c = d == 0 ? 0 : 2;
#else
                    int const c = d;
#endif
                    e[d] = edges[c].const_array(mfi);
                }
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                        {
                            amrex::IntVect q(AMREX_D_DECL(i, j, k));
                            int value = a(q);
                            if (value > 0)
                                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                {
                                    auto off = amrex::IntVect::TheDimensionVector(d);
                                    if (e[d](q) && a(q + off) >= 0)
                                        value = amrex::min(value, a(q + off));
                                    if (e[d](q - off) && a(q - off) >= 0)
                                        value = amrex::min(value, a(q - off));
                                }
                            b(q) = value;
                            return {int(value != a(q))};
                        });
            }
            int flag = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceIntMax(flag);
            changed = flag != 0;
            amrex::iMultiFab::Copy(labels, next, 0, 0, 1, 0);
        }
        // Enumerate one scalar root per floating component using reductions.
        // This is O(component count) metadata, not O(global nodes) per rank.
        int previous = 0;
        while (true)
        {
            amrex::ReduceOps<amrex::ReduceOpMin> op;
            amrex::ReduceData<int> data(op);
            using T = decltype(data)::Type;
            for (amrex::MFIter mfi(labels); mfi.isValid(); ++mfi)
            {
                auto a = labels.const_array(mfi);
                op.eval(mfi.validbox(), data,
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                        {
                            int n = a(i, j, k);
                            return {n > previous ? n : std::numeric_limits<int>::max()};
                        });
            }
            int root = amrex::get<0>(data.value());
            amrex::ParallelDescriptor::ReduceIntMin(root);
            if (root == std::numeric_limits<int>::max())
                break;
            roots.push_back(root);
            previous = root;
        }
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<int> data(op);
        using T = decltype(data)::Type;
        for (amrex::MFIter mfi(gauge); mfi.isValid(); ++mfi)
        {
            auto a = labels.const_array(mfi);
            auto g = gauge.array(mfi);
            auto weight = weights.const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    {
                        amrex::IntVect q(AMREX_D_DECL(i, j, k));
                        int id = 0, stride = 1;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            int p = periodic[d] && q[d] == high[d] ? low[d] : q[d];
                            id += (p - low[d]) * stride;
                            stride *= extent[d];
                        }
                        bool fixed = false;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            fixed = fixed || (q[d] == low[d] && fixedlo[d]) ||
                                    (q[d] == high[d] && fixedhi[d]);
                        g(q) = fixed || a(q) < 0 || a(q) == id + 1;
                        return {int(a(q) < 0 && weight(q) > 0.)};
                    });
        }
        isolated = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceIntSum(isolated);
    }
    double Compatibility(MF const &rhs, MF const &weights) const
    {
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<double> data(op);
        using T = decltype(data)::Type;
        for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
        {
            auto r = rhs.const_array(mfi), w = weights.const_array(mfi);
            auto l = labels.const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    { return {l(i, j, k) < 0 && w(i, j, k) > 0. ? std::abs(r(i, j, k)) : 0.}; });
        }
        double error = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(error);
        for (int root : roots)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> sum;
            amrex::ReduceData<double, double> pair(sum);
            using P = decltype(pair)::Type;
            for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
            {
                auto r = rhs.const_array(mfi), w = weights.const_array(mfi);
                auto l = labels.const_array(mfi);
                sum.eval(mfi.validbox(), pair,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) -> P
                         {
                             double v = l(i, j, k) == root ? w(i, j, k) : 0.;
                             return {v * r(i, j, k), v};
                         });
            }
            auto v = pair.value();
            double values[]{amrex::get<0>(v), amrex::get<1>(v)};
            amrex::ParallelDescriptor::ReduceRealSum(values, 2);
            AMREX_ALWAYS_ASSERT(values[1] > 0.);
            error = std::max(error, std::abs(values[0] / values[1]));
        }
        return error;
    }
    // Eliminate each floating scalar component's constant multiplier, then
    // retain the existing potential gauge. Row deletion alone concentrates
    // the tolerated integrated compatibility defect in that row's dual volume.
    // Only equation scratch is projected; physical acceptance remains full-row.
    bool ProjectAndRestrict(MF &x, MF const &weights) const
    {
        for (int root : roots)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> sum;
            amrex::ReduceData<double, double> pair(sum);
            using P = decltype(pair)::Type;
            for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi)
            {
                auto value = x.const_array(mfi), weight = weights.const_array(mfi);
                auto label = labels.const_array(mfi);
                sum.eval(mfi.validbox(), pair,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) -> P
                         {
                             double const v = label(i, j, k) == root ? weight(i, j, k) : 0.;
                             return {v * value(i, j, k), v};
                         });
            }
            auto const local = pair.value();
            double moments[]{amrex::get<0>(local), amrex::get<1>(local)};
            amrex::ParallelDescriptor::ReduceRealSum(moments, 2);
            if (!std::isfinite(moments[0]) || !std::isfinite(moments[1]) || !(moments[1] > 0.))
                return false;
            double const mean = moments[0] / moments[1];
            if (!std::isfinite(mean)) return false;
            for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi)
            {
                auto value = x.array(mfi);
                auto label = labels.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
                                       if (label(i, j, k) == root)
                                           value(i, j, k) -= mean;
                                   });
            }
        }
        Restrict(x);
        return true;
    }
    void Restrict(MF &x) const
    {
        for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi)
        {
            auto a = x.array(mfi);
            auto g = gauge.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   if (g(i, j, k))
                                       a(i, j, k) = 0.;
                               });
        }
    }
};
