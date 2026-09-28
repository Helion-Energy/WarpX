/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Private setup-only conversion of the joined weighted residual to every
// physical scalar row. The operator and potential gauge are unchanged.
struct NativeVacuumPhysicalStopping
{
    using Real = amrex::Real;
    static constexpr Real physical_limit = 1.e-8;
    static constexpr Real remainder_limit = 1.e-9;
    struct Floating { int label; Real gauge_weight; };
    std::vector<Floating> floating;
    Real kappa = 0., underflow_allowance = 0., sum_gamma = 0., sum_tiny = 0.;
    Real target = 0., remainder_bound = std::numeric_limits<Real>::infinity();
    AMREX_GPU_HOST_DEVICE static Real Up(Real x)
    { return std::nextafter(x, std::numeric_limits<Real>::infinity()); }
    AMREX_GPU_HOST_DEVICE static Real Down(Real x)
    { return std::nextafter(x, -std::numeric_limits<Real>::infinity()); }
    AMREX_GPU_HOST_DEVICE static Real Add(Real a, Real b) { return Up(a+b); }
    AMREX_GPU_HOST_DEVICE static Real Mul(Real a, Real b) { return Up(a*b); }
    AMREX_GPU_HOST_DEVICE static Real Div(Real a, Real b) { return Up(a/b); }
    AMREX_GPU_HOST_DEVICE static bool GradualUnderflow()
    {
        volatile Real smallest = std::numeric_limits<Real>::min();
        volatile Real two = 2.;
        Real const half = smallest/two;
        return half>0. && half+half==smallest;
    }
    static Real Gamma(Real n)
    {
        Real const e = Mul(n, std::numeric_limits<Real>::epsilon());
        return e < .25 ? Div(e, Down(1.-e)) : std::numeric_limits<Real>::infinity();
    }
    bool Prepare(JoinedMatrixFree const& op)
    {
        auto const& weights = op.support.NodeWeights();
        Real const volume = op.volume;
        Real const count = static_cast<Real>(weights.boxArray().numPts());
        Real const operations = Add(Mul(8., count), 64.);
        // Positive scalar-norm summands: two products per row, arbitrary
        // reduction tree, normalization and sqrt. Normal weights also bound
        // gradual-underflow errors without copying any field to the host.
        sum_gamma = Gamma(operations);
        sum_tiny = Mul(operations, std::numeric_limits<Real>::denorm_min());
        int invalid = !std::numeric_limits<Real>::is_iec559 ||
            std::numeric_limits<Real>::digits != 53 || std::fegetround() != FE_TONEAREST ||
            !GradualUnderflow() ||
            !(count > 0.) || !std::isfinite(sum_gamma) ||
            !(volume >= std::numeric_limits<Real>::min()) || !std::isfinite(volume) ||
            !(op.edge_measure > 0.) || !std::isfinite(op.edge_measure) ||
            !(op.length > 0.) || !std::isfinite(op.length);
#if defined(WARPX_DIM_RZ)
        invalid = invalid || op.w.Geom(0).ProbLo(0)<0.;
#endif
        amrex::ReduceOps<amrex::ReduceOpMin, amrex::ReduceOpMax,
                         amrex::ReduceOpSum> reduce;
        amrex::ReduceData<Real, int, amrex::Long> data(reduce);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(weights); mfi.isValid(); ++mfi)
        {
            auto w = weights.const_array(mfi);
            reduce.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                {
                    Real const v = w(i,j,k);
                    return {v > 0. ? v : std::numeric_limits<Real>::infinity(),
                        int(!GradualUnderflow() || !std::isfinite(v) || v < 0. ||
                            (v > 0. && v < std::numeric_limits<Real>::min())),
                        amrex::Long(v>0.)};
                });
        }
        auto pair = data.value();
        Real minimum = amrex::get<0>(pair);
        invalid = std::max(invalid, amrex::get<1>(pair));
        amrex::Long positive = amrex::get<2>(pair);
        amrex::ParallelDescriptor::ReduceLongSum(positive);
        amrex::ParallelDescriptor::ReduceRealMin(minimum);
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if (invalid || positive<=0 || !(minimum > 0.) || !std::isfinite(minimum)) return false;
        // Every retained positive-weight row, including grounded components.
        // Isolated rows have zero lift and are certified in the remainder.
        Real factor = Up(std::sqrt(Div(volume, minimum)));
        for (int root : op.graph.roots)
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpSum> sum;
            amrex::ReduceData<Real, Real, int> values(sum);
            using T = decltype(values)::Type;
            for (amrex::MFIter mfi(weights); mfi.isValid(); ++mfi)
            {
                auto w = weights.const_array(mfi);
                auto label = op.graph.labels.const_array(mfi);
                auto gauge = op.graph.gauge.const_array(mfi);
                sum.eval(mfi.validbox(), values,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    {
                        bool const active = label(i,j,k)==root && w(i,j,k)>0.;
                        bool const fixed = active && gauge(i,j,k);
                        return {active && !fixed ? w(i,j,k) : 0.,
                            fixed ? w(i,j,k) : 0., int(fixed)};
                    });
            }
            auto local = values.value();
            Real moments[]{amrex::get<0>(local), amrex::get<1>(local)};
            int gauges = amrex::get<2>(local);
            amrex::ParallelDescriptor::ReduceRealSum(moments, 2);
            amrex::ParallelDescriptor::ReduceIntSum(gauges);
            if (gauges!=1 || !(moments[1]>0.) || !std::isfinite(moments[1]) ||
                !(moments[0]>=0.) || !std::isfinite(moments[0])) return false;
            // One owned nonzero gauge weight: any reduction tree returns its
            // exact represented value, regardless of rank decomposition.
            Real const rest = Div(Add(moments[0], sum_tiny), Down(1.-sum_gamma));
            factor = std::max(factor, Div(Up(std::sqrt(Mul(volume, rest))), moments[1]));
            floating.push_back({root, moments[1]});
        }
        kappa = Mul(factor, Up(std::sqrt(Div(1., Down(1.-sum_gamma)))));
        underflow_allowance = Mul(factor, Up(std::sqrt(Div(sum_tiny, volume))));
        Real const budget = Down(Down(physical_limit-remainder_limit)-underflow_allowance);
        if (!(budget>0.) || !(kappa>0.) || !std::isfinite(kappa) ||
            !std::isfinite(underflow_allowance)) return false;
        target = Down(budget/kappa);
        return target>0. && std::isfinite(target);
    }
    // For r=rhs-Ax, Lift(r) is r on retained rows and
    // -sum_{i!=g}(w_i*r_i)/w_g at a deleted floating gauge. The actual
    // e=L*div(E_finished)+Lift(r) includes compatibility, inner affine/core
    // nonlinearity and action arithmetic. Its independent bound must fit the
    // fixed compatibility allowance; it never enlarges or tunes that budget.
    bool Certify(JoinedMatrixFree const& op, JoinedVector const& residual,
                 MF const* divergence_low = nullptr)
    {
        auto const& weights = op.support.NodeWeights();
        Real const length = op.length;
        Real maximum = 0.;
        int invalid = std::fegetround()!=FE_TONEAREST || !GradualUnderflow();
        Real const local_gamma = Gamma(8.);
        Real const tiny = Mul(8., std::numeric_limits<Real>::denorm_min());
        auto rows = [&](int component, Real lift, Real lift_error)
        {
            amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> reduce;
            amrex::ReduceData<Real, int> data(reduce);
            using T = decltype(data)::Type;
            for (amrex::MFIter mfi(weights); mfi.isValid(); ++mfi)
            {
                auto w = weights.const_array(mfi), d = op.divergence.const_array(mfi);
                auto r = residual.node.const_array(mfi);
                auto dl = divergence_low ? divergence_low->const_array(mfi)
                                         : amrex::Array4<Real const>{};
                auto label = op.graph.labels.const_array(mfi);
                auto gauge = op.graph.gauge.const_array(mfi);
                reduce.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    {
                        bool const floating_gauge = gauge(i,j,k) && label(i,j,k)>0;
                        bool const selected = w(i,j,k)>0. &&
                            (component<0 ? !floating_gauge :
                                floating_gauge && label(i,j,k)==component);
                        if (!selected) return {0., 0};
                        Real const term = component>=0 ? lift : (gauge(i,j,k) ? 0. : r(i,j,k));
                        Real const physical = length*d(i,j,k);
                        Real const expression = physical+term;
                        Real const scale = Add(Mul(std::abs(length),std::abs(d(i,j,k))),std::abs(term));
                        Real const radius = Add(Add(Mul(local_gamma,scale),tiny),lift_error);
                        Real bound = Add(std::abs(expression),radius);
                        // The pair's low divergence is physical state, not an
                        // optional publication error. Triangle enclosure adds
                        // its complete directed product to the same fixed B.
                        if (dl) bound = Add(bound,Mul(std::abs(length),std::abs(dl(i,j,k))));
                        return {bound, int(!GradualUnderflow() || !std::isfinite(bound) ||
                            !std::isfinite(r(i,j,k)) || (dl && !std::isfinite(dl(i,j,k))))};
                    });
            }
            auto local = data.value();
            maximum = std::max(maximum, amrex::get<0>(local));
            invalid = std::max(invalid, amrex::get<1>(local));
        };
        rows(-1,0.,0.);
        for (auto const& group : floating)
        {
            int const root = group.label;
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                             amrex::ReduceOpMax> sum;
            amrex::ReduceData<Real, Real, int> data(sum);
            using T = decltype(data)::Type;
            for (amrex::MFIter mfi(weights); mfi.isValid(); ++mfi)
            {
                auto w = weights.const_array(mfi), r = residual.node.const_array(mfi);
                auto label = op.graph.labels.const_array(mfi);
                auto gauge = op.graph.gauge.const_array(mfi);
                sum.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> T
                    {
                        if (!(w(i,j,k)>0.) || label(i,j,k)!=root || gauge(i,j,k))
                            return {0.,0.,0};
                        Real const term = w(i,j,k)*r(i,j,k);
                        return {term, std::abs(term), int(!std::isfinite(term))};
                    });
            }
            auto local = data.value();
            Real moments[]{amrex::get<0>(local), amrex::get<1>(local)};
            int bad = amrex::get<2>(local);
            amrex::ParallelDescriptor::ReduceRealSum(moments,2);
            amrex::ParallelDescriptor::ReduceIntMax(bad);
            invalid = std::max(invalid,bad);
            Real const scale = Div(Add(moments[1],sum_tiny),Down(1.-sum_gamma));
            Real const sum_error = Add(Mul(sum_gamma,scale),sum_tiny);
            Real const lift = -moments[0]/group.gauge_weight;
            Real const lift_error = Add(Div(sum_error,group.gauge_weight),
                                        Add(Mul(local_gamma,std::abs(lift)),tiny));
            rows(root,lift,lift_error);
        }
        amrex::ParallelDescriptor::ReduceRealMax(maximum);
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        remainder_bound = maximum;
        return !invalid && std::isfinite(maximum) && maximum<=remainder_limit;
    }
};
