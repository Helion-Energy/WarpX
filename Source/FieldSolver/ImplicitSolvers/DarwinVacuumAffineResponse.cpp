/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "DarwinVacuumAffineResponse.H"
#include "NativeEndpointArithmetic.H"
#include "DarwinABoundary.H"
#include "CompensatedNativeCurl.H"
#include <iomanip>
#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "WarpX.H"
#if defined(AMREX_USE_GPU)
#include "NativeEndpointArithmetic.H"
#include "NativePairedDarwinFields.H"
#endif
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <array>
#include <vector>
#include <utility>
#include <limits>
namespace warpx::darwin
{
namespace
{
using Field = amrex::Array<amrex::MultiFab, 3>;
using MF = amrex::MultiFab;
using View = ablastr::fields::VectorField;
View view(Field &f) { return {&f[0], &f[1], &f[2]}; }
View view(Field const &f)
{
    return {const_cast<MF *>(&f[0]), const_cast<MF *>(&f[1]), const_cast<MF *>(&f[2])};
}
// A distinct lifetime token replaces arithmetic generation counters. Revoking
// the token expires every old weak binding; no wraparound can revalidate it.
struct RetainedBinding {};
bool StorageOverlaps(MF const& left,MF const& right)
{
    using Interval=std::pair<std::uintptr_t,std::uintptr_t>;
    bool valid=true;
    auto intervals=[&valid](MF const& field)
    {
        std::vector<Interval> result;
        // AMReX forbids nested live MFIter instances by default. This scope
        // finishes completely before the next field's intervals are collected.
        for(amrex::MFIter it(field);it.isValid();++it)
        {
            auto const begin=reinterpret_cast<std::uintptr_t>(field[it].dataPtr());
            auto const count=static_cast<std::uint64_t>(field[it].size());
            if(count>std::numeric_limits<std::uintptr_t>::max()/sizeof(amrex::Real))
            {valid=false;continue;}
            auto const bytes=static_cast<std::uintptr_t>(count*sizeof(amrex::Real));
            if(begin>std::numeric_limits<std::uintptr_t>::max()-bytes || (!begin && bytes))
            {valid=false;continue;}
            result.emplace_back(begin,begin+bytes);
        }
        return result;
    };
    auto const a=intervals(left);
    auto const b=intervals(right);
    if(!valid)return true;
    for(auto const& x:a)for(auto const& y:b)
        if(x.first<y.second && y.first<x.second)return true;
    return false;
}
} // namespace
struct DarwinVacuumAffineResponse::RetainedRecovery::Storage
{
    Field high, low;
    std::weak_ptr<RetainedBinding> binding;
    RetainedReceipt receipt;
};
struct DarwinVacuumAffineResponse::Impl
{
    WarpX &warpx;
    amrex::Geometry geom;
    Options options;
    amrex::Periodicity periodic;
    bool axis, magnetic, flux_only, check_operator, frozen = false;
    amrex::Real diagonal = 0., scale, rtol, atol;
    int max_iterations;
    amrex::GpuArray<int, AMREX_SPACEDIM> pmc_lo, pmc_hi;
    amrex::IntVect curl_grow;
    Field origin, base, delta, delta_low, residual, direction, action, input, curl, current, reference;
    // Private operator scratch only; never published or retained between solves.
    Field curl_low, current_low;
    bool compensated_supported = false;
    compensated_curl::Coefficients curl_coefficients;
    amrex::Array<amrex::iMultiFab, 3> mask, trace_mask;
    amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> owner;
    // Allocated only by the additive retained API, never by legacy Recover/Response.
    struct RetainedScratch
    {
        Field input_low, base_action, base_action_low, action_low,
              assembled, assembled_low, residual_low, zero, base_low;
        std::shared_ptr<RetainedBinding> binding = std::make_shared<RetainedBinding>();
        amrex::Real norm_factor = 0.;
        explicit RetainedScratch(Impl const& p)
        {
            long double count=0.;
            for(int c=0;c<3;++c)
            {
                input_low[c].define(p.input[c].boxArray(),p.input[c].DistributionMap(),
                                    1,p.input[c].nGrowVect());
                for(auto* f:{&base_action,&base_action_low,&action_low,&assembled,
                              &assembled_low,&residual_low,&zero,&base_low})
                    (*f)[c].define(p.base[c].boxArray(),p.base[c].DistributionMap(),1,0);
                zero[c].setVal(0.);base_low[c].setVal(0.);
                count+=p.base[c].boxArray().numPts();
            }
            // Positive weighted sums only. Counting all valid replicas is a
            // conservative upper bound on owner terms and sequential reduction work.
            long double const u=std::numeric_limits<amrex::Real>::epsilon();
            long double const budget=(8.L*count+64.L*amrex::ParallelDescriptor::NProcs())*u;
            norm_factor=budget<.25L?static_cast<amrex::Real>(1.L/(1.L-budget)):
                std::numeric_limits<amrex::Real>::infinity();
            norm_factor=std::nextafter(norm_factor,std::numeric_limits<amrex::Real>::infinity());
        }
    };
    std::unique_ptr<RetainedScratch> retained;
    // Only the accepted-origin API allocates this part. The legacy Freeze
    // path owns no additional baseline field or per-action allocation.
    std::unique_ptr<Field> origin_low;
    bool retained_freeze_valid=false;
    Impl(WarpX &w, View const &e, View const &b, Options o)
        : warpx(w), geom(w.Geom(0)), options(o), periodic(geom.periodicity()), axis(false),
          magnetic(o.magnetic), flux_only(o.flux_only), check_operator(o.check_operator),
          rtol(o.relative_tolerance), atol(o.absolute_tolerance), max_iterations(o.max_iterations),
          pmc_lo(o.pmc_lo), pmc_hi(o.pmc_hi),
          curl_grow(DarwinPMCCurlGrow(WarpX::field_boundary_lo, WarpX::field_boundary_hi))
    {
#if defined(WARPX_DIM_RZ)
        axis = geom.ProbLo(0) == 0.;
#endif
        AMREX_ALWAYS_ASSERT(std::isfinite(rtol) && std::isfinite(atol) && rtol > 0. && atol >= 0. &&
                            max_iterations > 0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            diagonal += 4. / (geom.CellSize(d) * geom.CellSize(d));
        scale = PhysConst::mu0 / diagonal;
        compensated_supported = !magnetic && compensated_curl::Supported(warpx);
        if (compensated_supported)
            curl_coefficients = compensated_curl::NativeCoefficients(warpx);
        for (int d = 0; d < 3; ++d)
        {
            for (auto *f : {&origin, &base, &delta, &residual, &direction, &action})
                (*f)[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, 0);
            if (!magnetic)
                delta_low[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, 0);
            input[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, e[d]->nGrowVect());
            current[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, 1);
            curl[d].define(b[d]->boxArray(), b[d]->DistributionMap(), 1, b[d]->nGrowVect());
            if (compensated_supported)
            {
                curl_low[d].define(b[d]->boxArray(), b[d]->DistributionMap(), 1, b[d]->nGrowVect());
                current_low[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, 1);
            }
            mask[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, 0);
            trace_mask[d].define(e[d]->boxArray(), e[d]->DistributionMap(), 1, 0);
            owner[d] = e[d]->OwnerMask(periodic);
        }
    }
    bool layout(View const &fields) const
    {
        for (int c = 0; c < 3; ++c)
        {
            if (!fields[c] || fields[c]->boxArray() != origin[c].boxArray() ||
                fields[c]->DistributionMap() != origin[c].DistributionMap() ||
                fields[c]->nComp() != 1)
                return false;
            for (int d = 0; d < c; ++d)
                if (fields[c] == fields[d])
                    return false;
        }
        return true;
    }
    amrex::Real component_dot(amrex::MultiFab const &x, amrex::MultiFab const &y,
                              amrex::iMultiFab const &own) const
    {
        [[maybe_unused]] auto const has_axis = axis;
        auto const low_pmc = pmc_lo, high_pmc = pmc_hi;
        auto const domain = amrex::convert(geom.Domain(), x.ixType());
        amrex::GpuArray<int, AMREX_SPACEDIM> lo{}, hi{}, node{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            lo[d] = domain.smallEnd(d);
            hi[d] = domain.bigEnd(d);
            node[d] = x.ixType().nodeCentered(d);
        }
#if defined(WARPX_DIM_RZ)
        amrex::Real const r0 = geom.ProbLo(0) / geom.CellSize(0);
#endif
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(x); mfi.isValid(); ++mfi)
        {
            auto const a = x.const_array(mfi), other = y.const_array(mfi);
            auto const o = own.const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                    {
                        amrex::Real weight = 1.;
#if defined(WARPX_DIM_RZ)
                        weight = r0 + i + (node[0] ? 0. : .5);
                        if (has_axis && node[0] && i == 0)
                        {
                            weight = .125;
                        }
#endif
                        int const p[3] = {i, j, k};
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            if (node[d] &&
                                ((low_pmc[d] && p[d] == lo[d]) || (high_pmc[d] && p[d] == hi[d])))
                            {
                                weight *= .5;
                            }
                        }
                        return {o(i, j, k) ? weight * a(i, j, k) * other(i, j, k) : 0.};
                    });
        }
        return amrex::get<0>(data.value(op));
    }
    amrex::Real dot(View const &x, View const &y) const
    {
        amrex::Real v = 0.;
        for (int d = 0; d < 3; ++d)
        {
            v += component_dot(*x[d], *y[d], *owner[d]);
        }
        amrex::ParallelAllReduce::Sum(v, amrex::ParallelContext::CommunicatorSub());
        return v;
    }
    void apply(View const &out, View const &x, bool const with_boundary_data = false,
               bool const compensated = false)
    {
        for (int d = 0; d < 3; ++d)
        {
            input[d].setVal(0.);
            amrex::MultiFab::Copy(input[d], *x[d], 0, 0, 1, 0);
            input[d].OverrideSync(periodic);
            input[d].FillBoundary(periodic);
            curl[d].setVal(0.);
            current[d].setVal(0.);
        }
#if defined(WARPX_DIM_RZ)
        if (axis)
        {
            warpx.ApplyFieldBoundaryOnAxis(&input[0], &input[1], &input[2], 0);
        }
#endif
        for (int d = 0; d < 3; ++d)
        {
            // The total endpoint A already contains the prescribed external
            // trace. Krylov directions instead use homogeneous response BCs.
            auto const *imposed =
                with_boundary_data && reference[d].isDefined() ? &reference[d] : nullptr;
            ApplyDarwinPMCVectorBoundary(input[d], geom, pmc_lo, pmc_hi, imposed);
        }
        auto cv = view(curl), iv = view(input), jv = view(current);
        if (compensated)
        {
            AMREX_ALWAYS_ASSERT(compensated_supported && !with_boundary_data);
            compensated_curl::First(cv,view(curl_low),iv,curl_coefficients,geom,curl_grow);
        }
        else
            warpx.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(cv, iv, warpx.GetEBUpdateBFlag()[0], 0,
                                                              curl_grow);
        for (auto &f : curl)
        {
            f.OverrideSync(periodic);
            f.FillBoundary(periodic);
        }
#if defined(WARPX_DIM_RZ)
        if (axis)
        {
            warpx.ApplyFieldBoundaryOnAxis(&curl[0], &curl[1], &curl[2], 0);
        }
#endif
        if (compensated)
        {
            // The same owner/periodic/axis operations act on both parts. PMC
            // curl guards already came from the reflected input on curl_grow.
            for (auto &f : curl_low)
            {
                f.OverrideSync(periodic);
                f.FillBoundary(periodic);
            }
#if defined(WARPX_DIM_RZ)
            if (axis)
                warpx.ApplyFieldBoundaryOnAxis(&curl_low[0],&curl_low[1],&curl_low[2],0);
#endif
            compensated_curl::Second(jv,view(current_low),cv,view(curl_low),curl_coefficients);
        }
        else
            warpx.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(jv, cv,
                                                                        warpx.GetEBUpdateEFlag()[0], 0);
        auto const operator_scale = scale;
        for (int d = 0; d < 3; ++d)
        {
            for (amrex::MFIter mfi(*out[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto a = out[d]->array(mfi);
                auto c = current[d].const_array(mfi);
                auto m = mask[d].const_array(mfi);
                auto low = compensated ? current_low[d].const_array(mfi) : amrex::Array4<amrex::Real const>{};
                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    if (compensated)
                    {
                        auto const value=warpx::ohm::compensated::Multiply(
                            {c(i,j,k),low(i,j,k)},operator_scale);
                        a(i,j,k)=m(i,j,k)?value.hi:0.;
                    }
                    else
                        a(i,j,k)=m(i,j,k)?operator_scale*c(i,j,k):0.;
                });
            }
            out[d]->OverrideSync(periodic);
        }
    }

    bool retained_scope() const
    {
        bool valid=false;
#if (defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)) && !defined(AMREX_USE_GPU)
        // The retained representation changes arithmetic only. Geometry,
        // staggering, masks and the native zero-reference boundary operator
        // are those already supported by CompensatedNativeCurl.
        valid=compensated_supported && !magnetic && !flux_only &&
            NativeEndpointArithmeticSupported();
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            bool const low_pmc=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
            bool const high_pmc=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
            valid=valid && bool(pmc_lo[d])==low_pmc && bool(pmc_hi[d])==high_pmc;
            if(geom.isPeriodic(d)) valid=valid && !low_pmc && !high_pmc;
            else {
                bool const low_fixed=WarpX::field_boundary_lo[d]==FieldBoundaryType::PEC;
                bool const high_fixed=WarpX::field_boundary_hi[d]==FieldBoundaryType::PEC;
                bool axis_low=false;
#if defined(WARPX_DIM_RZ)
                axis_low=axis && d==0;
#endif
                valid=valid && (low_pmc || low_fixed || axis_low) &&
                    (high_pmc || high_fixed);
            }
        }
#elif defined(WARPX_DIM_RZ) && defined(AMREX_USE_CUDA)
        // The core is also called during constrained initialization, before
        // the endpoint owner exists. Require the actual solver's immutable
        // startup selection plus this TU's precise arithmetic attestation.
        valid=NativeEndpointArithmeticSupported() &&
            NativeEndpointCudaQualificationSelected(warpx) &&
            compensated_supported && sizeof(amrex::Real)==sizeof(double) &&
            !magnetic && !flux_only && axis && !geom.isPeriodic(0) && geom.isPeriodic(1) &&
            WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC;
        for(int d=0;d<AMREX_SPACEDIM;++d)
            valid=valid && !pmc_lo[d] && !pmc_hi[d];
#endif
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if(!valid) return false;
        // Reference presence need not agree locally: inspect locally, then agree
        // once. Do not condition a collective MultiFab norm on pointer presence.
        int bad=0;
        for(int c=0;c<3;++c) if(reference[c].isDefined())
        {
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<int> data(op);
            using T=decltype(data)::Type;
            for(amrex::MFIter mfi(reference[c]);mfi.isValid();++mfi)
            {
                auto x=reference[c].const_array(mfi);
                op.eval(reference[c][mfi].box(),data,
                    [=] AMREX_GPU_DEVICE(int i,int j,int k)->T
                    {return {!std::isfinite(x(i,j,k)) || x(i,j,k)!=0.};});
            }
            bad=std::max(bad,amrex::get<0>(data.value(op)));
        }
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        return bad==0;
    }
    void retained_images(Field& f)
    {
        for(auto& x:f){x.OverrideSync(periodic);x.FillBoundary(periodic);}
#if defined(WARPX_DIM_RZ)
        if(axis)warpx.ApplyFieldBoundaryOnAxis(&f[0],&f[1],&f[2],0);
#endif
        for(auto& x:f)ApplyDarwinPMCVectorBoundary(x,geom,pmc_lo,pmc_hi,nullptr);
    }
    void apply_pair(Field& out,Field& out_low,Field const& high,Field const& low,
                    bool include_trace=false)
    {
        auto& q=*retained;
        for(int c=0;c<3;++c)
        {
            input[c].setVal(0.);q.input_low[c].setVal(0.);
            MF::Copy(input[c],high[c],0,0,1,0);
            MF::Copy(q.input_low[c],low[c],0,0,1,0);
        }
        retained_images(input);retained_images(q.input_low);
        compensated_curl::FirstPaired(view(curl),view(curl_low),view(input),
            view(q.input_low),curl_coefficients,geom,curl_grow);
        for(auto* f:{&curl,&curl_low})
        {
            for(auto& x:*f){x.OverrideSync(periodic);x.FillBoundary(periodic);}
#if defined(WARPX_DIM_RZ)
            if(axis)warpx.ApplyFieldBoundaryOnAxis(&(*f)[0],&(*f)[1],&(*f)[2],0);
#endif
        }
        compensated_curl::Second(view(current),view(current_low),view(curl),
                                 view(curl_low),curl_coefficients);
        auto const operator_scale=scale;
        for(int c=0;c<3;++c)
        {
            for(amrex::MFIter mfi(out[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
            {
                auto h=out[c].array(mfi),l=out_low[c].array(mfi);
                auto x=current[c].const_array(mfi),y=current_low[c].const_array(mfi);
                auto m=mask[c].const_array(mfi),t=trace_mask[c].const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
                {
                    auto const a=warpx::ohm::compensated::Multiply({x(i,j,k),y(i,j,k)},operator_scale);
                    bool const active=m(i,j,k) || (include_trace && t(i,j,k));
                    h(i,j,k)=active?a.hi:0.;l(i,j,k)=active?a.lo:0.;
                });
            }
            out[c].OverrideSync(periodic);out_low[c].OverrideSync(periodic);
        }
    }
    bool finite_pair(Field const& hi,Field const& lo) const
    {
        bool valid=true;
        for(int c=0;c<3;++c)
        {
            bool const h=hi[c].is_finite(0,1,0);
            bool const l=lo[c].is_finite(0,1,0);
            valid=h && l && valid;
        }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        return valid;
    }
    amrex::Real retained_norm(Field const& hi,Field const& lo) const
    {
        amrex::Real const h=dot(view(hi),view(hi)),l=dot(view(lo),view(lo));
        if(!(h>=0.) || !(l>=0.) || !std::isfinite(h) || !std::isfinite(l))
            return std::numeric_limits<amrex::Real>::infinity();
        return std::nextafter((std::sqrt(h)+std::sqrt(l))*retained->norm_factor,
                             std::numeric_limits<amrex::Real>::infinity());
    }
    void retained_residual()
    {
        auto& q=*retained;
        apply_pair(action,q.action_low,delta,delta_low);
        for(int c=0;c<3;++c)
            for(amrex::MFIter mfi(residual[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
            {
                auto r=residual[c].array(mfi),rl=q.residual_low[c].array(mfi);
                auto b=q.base_action[c].const_array(mfi),bl=q.base_action_low[c].const_array(mfi);
                auto a=action[c].const_array(mfi),al=q.action_low[c].const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
                {
                    namespace arithmetic=warpx::ohm::compensated;
                    auto const x=arithmetic::Negate(arithmetic::Add(
                        {b(i,j,k),bl(i,j,k)},{a(i,j,k),al(i,j,k)}));
                    r(i,j,k)=x.hi;rl(i,j,k)=x.lo;
                });
            }
    }
    void retained_assemble()
    {
        auto& q=*retained;
        for(int c=0;c<3;++c)
        {
            for(amrex::MFIter mfi(base[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
            {
                auto b=base[c].const_array(mfi),d=delta[c].const_array(mfi),
                     dl=delta_low[c].const_array(mfi),bl=q.base_low[c].const_array(mfi);
                auto h=q.assembled[c].array(mfi),l=q.assembled_low[c].array(mfi);
                auto m=mask[c].const_array(mfi);
                amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
                {
                    auto const x=warpx::ohm::compensated::Add({b(i,j,k),bl(i,j,k)},{d(i,j,k),dl(i,j,k)});
                    h(i,j,k)=m(i,j,k)?x.hi:b(i,j,k);
                    l(i,j,k)=m(i,j,k)?x.lo:bl(i,j,k);
                });
            }
            q.assembled[c].OverrideSync(periodic);q.assembled_low[c].OverrideSync(periodic);
        }
    }
    RetainedReceipt solve_retained(RetainedRecovery& output,View const* increment,
                                   bool homogeneous=false,View const* increment_low=nullptr)
    {
        RetainedReceipt receipt;
        receipt.homogeneous=homogeneous;
        receipt.affine_trace=increment_low?AffineTraceInput::Pair:
            (increment?AffineTraceInput::High:AffineTraceInput::None);
        int mode_min=homogeneous?1:0,mode_max=mode_min;
        amrex::ParallelDescriptor::ReduceIntMin(mode_min);
        amrex::ParallelDescriptor::ReduceIntMax(mode_max);
        bool valid=frozen && retained_freeze_valid && mode_min==mode_max;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if(!valid){receipt.solve.failure=Failure::NotFrozen;return receipt;}
        if(!retained_scope()){receipt.solve.failure=Failure::UnsupportedRetained;return receipt;}
        int has_increment=(increment?1:0)+(increment_low?2:0),minimum=has_increment,maximum=has_increment;
        amrex::ParallelDescriptor::ReduceIntMin(minimum);
        amrex::ParallelDescriptor::ReduceIntMax(maximum);
        valid=minimum==maximum && (!increment_low || increment) &&
            (!increment || layout(*increment)) && (!increment_low || layout(*increment_low));
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if(!valid){receipt.solve.failure=Failure::InvalidLayout;return receipt;}
        int bad=0;
        for(auto const* part:{increment,increment_low})if(part)for(int c=0;c<3;++c)
        {
            amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);
            using T=decltype(data)::Type;
            for(amrex::MFIter mfi(*(*part)[c]);mfi.isValid();++mfi)
            {
                auto x=(*part)[c]->const_array(mfi);
                auto m=trace_mask[c].const_array(mfi);
                op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T
                    {return {!std::isfinite(x(i,j,k)) || (!m(i,j,k) && x(i,j,k)!=0.)};});
            }
            bad=std::max(bad,amrex::get<0>(data.value(op)));
        }
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        if(bad){receipt.solve.failure=Failure::InvalidTrace;return receipt;}
        if(!retained)retained=std::make_unique<RetainedScratch>(*this);
        auto& q=*retained;
        if(!q.binding)q.binding=std::make_shared<RetainedBinding>();
        receipt.norm_roundoff_factor=q.norm_factor;
        for(int c=0;c<3;++c)
        {
            delta[c].setVal(0.);delta_low[c].setVal(0.);q.base_low[c].setVal(0.);
            if(homogeneous)base[c].setVal(0.);
            else {
                MF::Copy(base[c],origin[c],0,0,1,0);
                if(origin_low)MF::Copy(q.base_low[c],(*origin_low)[c],0,0,1,0);
            }
            if(increment) {
                if(!origin_low && !increment_low) {
                    // Exact old single-trace arithmetic when this extension
                    // has never supplied a paired origin or trace.
                    MF::Add(base[c],*(*increment)[c],0,0,1,0);
                } else {
                    bool const low=increment_low!=nullptr;
                    for(amrex::MFIter mfi(base[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
                        auto h=base[c].array(mfi),l=q.base_low[c].array(mfi);
                        auto dh=(*increment)[c]->const_array(mfi);
                        auto dl=low?(*increment_low)[c]->const_array(mfi):amrex::Array4<amrex::Real const>{};
                        auto t=trace_mask[c].const_array(mfi);
                        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                            if(t(i,j,k)) {
                                auto const value=warpx::ohm::compensated::Add({h(i,j,k),l(i,j,k)},
                                    {dh(i,j,k),low?dl(i,j,k):0.});
                                h(i,j,k)=value.hi;l(i,j,k)=value.lo;
                            }
                        });
                    }
                }
            }
            base[c].OverrideSync(periodic);q.base_low[c].OverrideSync(periodic);
        }
        apply_pair(q.base_action,q.base_action_low,base,q.base_low);
        retained_residual();
        amrex::Real const initial=retained_norm(residual,q.residual_low);
        amrex::Real const target=std::max(atol,rtol*std::sqrt(dot(residual,residual)));
        receipt.solve.initial=initial;receipt.solve.target=target;
        if(!finite_pair(residual,q.residual_low) || !std::isfinite(initial) || !std::isfinite(target))
        {receipt.solve.failure=Failure::Nonfinite;return receipt;}
        for(int c=0;c<3;++c)MF::Copy(direction[c],residual[c],0,0,1,0);
        amrex::Real rr=dot(residual,residual),truth=initial;
        int iterations=0;
        while(truth>target && iterations<max_iterations)
        {
            apply_pair(action,q.action_low,direction,q.zero);
            amrex::Real const pap=dot(direction,action);
            if(!(pap>0.) || !std::isfinite(pap))
            {receipt.solve.failure=Failure::Curvature;break;}
            amrex::Real const alpha=rr/pap;
            for(int c=0;c<3;++c)
            {
                for(amrex::MFIter mfi(delta[c],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
                {
                    auto h=delta[c].array(mfi),l=delta_low[c].array(mfi);
                    auto d=direction[c].const_array(mfi);
                    amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
                    {
                        namespace arithmetic=warpx::ohm::compensated;
                        auto const x=arithmetic::Add({h(i,j,k),l(i,j,k)},
                            arithmetic::Multiply({d(i,j,k),0.},alpha));
                        h(i,j,k)=x.hi;l(i,j,k)=x.lo;
                    });
                }
                MF::Saxpy(residual[c],-alpha,action[c],0,0,1,0);
            }
            ++iterations;
            amrex::Real next=dot(residual,residual);
            truth=std::sqrt(next);
            bool restart=false;
            if(truth<=target)
            {
                retained_residual();
                truth=retained_norm(residual,q.residual_low);
                next=dot(residual,residual);restart=true;
            }
            if(!std::isfinite(truth)){receipt.solve.failure=Failure::Nonfinite;break;}
            if(truth<=target)break;
            for(int c=0;c<3;++c)
                MF::LinComb(direction[c],1.,residual[c],0,restart?0.:next/rr,
                            direction[c],0,0,1,0);
            rr=next;
        }
        receipt.solve.iterations=iterations;receipt.solve.residual=truth;
        if(!receipt.solve)return receipt;
        if(truth>target){receipt.solve.failure=Failure::IterationLimit;return receipt;}
        retained_assemble();
        bad=0;
        for(int c=0;c<3;++c)
        {
            amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);
            using T=decltype(data)::Type;
            for(amrex::MFIter mfi(base[c]);mfi.isValid();++mfi)
            {
                auto b=base[c].const_array(mfi),bl=q.base_low[c].const_array(mfi),
                     h=q.assembled[c].const_array(mfi),l=q.assembled_low[c].const_array(mfi);
                auto m=mask[c].const_array(mfi);
                op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T
                {return {!m(i,j,k) && (h(i,j,k)!=b(i,j,k) || l(i,j,k)!=bl(i,j,k) ||
                    std::signbit(h(i,j,k))!=std::signbit(b(i,j,k)) ||
                    std::signbit(l(i,j,k))!=std::signbit(bl(i,j,k)))};});
            }
            bad=std::max(bad,amrex::get<0>(data.value(op)));
        }
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        receipt.fixed_trace_exact=bad==0;
        if(bad){receipt.solve.failure=Failure::InvalidTrace;return receipt;}
        apply_pair(action,q.action_low,q.assembled,q.assembled_low);
        receipt.assembled_residual=retained_norm(action,q.action_low);
        if(!finite_pair(q.assembled,q.assembled_low) || !std::isfinite(receipt.assembled_residual))
        {receipt.solve.failure=Failure::Nonfinite;return receipt;}
        if(receipt.assembled_residual>target)
        {receipt.solve.failure=Failure::IterationLimit;return receipt;}
        apply_pair(action,q.action_low,q.assembled,q.zero);
        receipt.materialized_compensated=retained_norm(action,q.action_low);
        apply(action,q.assembled,false,false);
        receipt.materialized_ordinary=retained_norm(action,q.zero);
        apply_pair(action,q.action_low,q.assembled_low,q.zero);
        receipt.discarded_action=retained_norm(action,q.action_low);
        receipt.single_field_within_target=
            receipt.materialized_compensated<=target && receipt.materialized_ordinary<=target;
        if(!std::isfinite(receipt.materialized_compensated) ||
           !std::isfinite(receipt.materialized_ordinary) || !std::isfinite(receipt.discarded_action))
        {receipt.solve.failure=Failure::Nonfinite;return receipt;}
        auto completed=std::make_unique<RetainedRecovery::Storage>();
        for(int c=0;c<3;++c)
        {
            for(auto* f:{&completed->high,&completed->low})
                (*f)[c].define(base[c].boxArray(),base[c].DistributionMap(),1,0);
            MF::Copy(completed->high[c],q.assembled[c],0,0,1,0);
            MF::Copy(completed->low[c],q.assembled_low[c],0,0,1,0);
        }
        receipt.ready=true;completed->receipt=receipt;
        completed->binding=q.binding;
        output.m_storage=std::move(completed);
        return receipt;
    }

    void apply(Field &out, Field const &x, bool affine = false, bool compensated = false)
    {
        apply(view(out), view(x), affine, compensated);
    }
    amrex::Real dot(Field const &a, Field const &b) const { return dot(view(a), view(b)); }
    Result solve(View const &output, View const *increment, bool homogeneous)
    {
        Result result;
        if (origin_low) {result.failure=Failure::UnsupportedRetained;return result;}
        if (!frozen)
        {
            result.failure = Failure::NotFrozen;
            return result;
        }
        if (!layout(output) || (increment && !layout(*increment)))
        {
            result.failure = Failure::InvalidLayout;
            return result;
        }
        if (increment)
        {
            for (auto *out : output)
                for (auto *in : *increment)
                    if (out == in)
                    {
                        result.failure = Failure::InvalidLayout;
                        return result;
                    }
            int bad = 0;
            for (int c = 0; c < 3; ++c)
            {
                amrex::ReduceOps<amrex::ReduceOpMax> op;
                amrex::ReduceData<int> data(op);
                using Tuple = decltype(data)::Type;
                for (amrex::MFIter mfi(*(*increment)[c]); mfi.isValid(); ++mfi)
                {
                    auto x = (*increment)[c]->const_array(mfi);
                    auto m = trace_mask[c].const_array(mfi);
                    op.eval(mfi.validbox(), data,
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                            {
                                return {!std::isfinite(x(i, j, k)) ||
                                        (!m(i, j, k) && x(i, j, k) != 0.)};
                            });
                }
                bad = std::max(bad, amrex::get<0>(data.value(op)));
            }
            amrex::ParallelAllReduce::Max(bad, amrex::ParallelContext::CommunicatorSub());
            if (bad)
            {
                result.failure = Failure::InvalidTrace;
                return result;
            }
        }
        for (int c = 0; c < 3; ++c)
        {
            delta[c].setVal(0.);
            if (homogeneous && !magnetic)
                delta_low[c].setVal(0.);
            if (homogeneous)
                base[c].setVal(0.);
            else
                MF::Copy(base[c], origin[c], 0, 0, 1, 0);
            if (increment)
                MF::Add(base[c], *(*increment)[c], 0, 0, 1, 0);
            base[c].OverrideSync(periodic);
        }
        bool const compensated = homogeneous && compensated_supported;
        if (check_operator)
        {
            Field u, v, ku, kv;
            for (int d = 0; d < 3; ++d)
            {
                for (auto *f : {&u, &v, &ku, &kv})
                {
                    (*f)[d].define(base[d].boxArray(), base[d].DistributionMap(), 1, 0);
                }
                for (amrex::MFIter mfi(u[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    auto ua = u[d].array(mfi), va = v[d].array(mfi);
                    auto m = mask[d].const_array(mfi);
                    amrex::ParallelFor(
                        mfi.tilebox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                        {
                            ua(i, j, k) =
                                m(i, j, k) ? std::sin(.731 * (i + 3 * j + 7 * k + 11 * d)) : 0.;
                            va(i, j, k) =
                                m(i, j, k) ? std::cos(.413 * (5 * i + 2 * j + 3 * k + 13 * d)) : 0.;
                        });
                }
            }
            apply(ku, u, false, compensated);
            amrex::Real energy = 0.;
            for (int d = 0; d < 3; ++d)
            {
                auto const own = curl[d].OwnerMask(periodic);
                energy += component_dot(curl[d], curl[d], *own) / diagonal;
            }
            amrex::ParallelAllReduce::Sum(energy, amrex::ParallelContext::CommunicatorSub());
            apply(kv, v, false, compensated);
            amrex::Real const uku = dot(u, ku), ukv = dot(u, kv), vku = dot(v, ku);
            amrex::Real const energy_error =
                std::abs(uku - energy) / std::max(1., std::abs(energy));
            amrex::Real const symmetry_error =
                std::abs(ukv - vku) / std::max(1., std::sqrt(dot(u, u) * dot(v, v)));
            amrex::Print() << "Spatial recovery operator: symmetry=" << symmetry_error
                           << " energy=" << energy_error << "\n";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                symmetry_error < 1.e-12 && energy_error < 1.e-12,
                "Spatial recovery operator violates its symmetry or curl-energy "
                "identity");
        }
        apply(action, base, !homogeneous, compensated);
        for (int d = 0; d < 3; ++d)
        {
            amrex::MultiFab::Copy(residual[d], action[d], 0, 0, 1, 0);
            residual[d].mult(-1., 0, 1, 0);
            amrex::MultiFab::Copy(direction[d], residual[d], 0, 0, 1, 0);
        }
        // With K = I_V C^T C I_V, zero-start unpreconditioned CG keeps every
        // correction in range(K). It preserves the baseline's null modes (accepted
        // electric modes or extrapolated endpoint A modes). Fixed plasma and wall
        // values never enter the correction space. A global inverse followed by
        // masking is not this projection and can amplify the interface defect.
        // An arbitrary preconditioner would invalidate the null-mode selection.
        amrex::Real rr = dot(residual, residual), initial = std::sqrt(rr);
        // A has no universal dimensional absolute floor. Stop at the native
        // operator's roundoff scale when an already projected field is revisited;
        // otherwise a second projection would demand a tolerance below cancellation
        // error. This bound scales with the actual field, not a fixed unit value.
        amrex::Real reference_norm = 0.;
        if (magnetic)
        {
#if defined(WARPX_DIM_RZ)
            if (flux_only)
            {
                // The axisymmetric toroidal block does not depend on Ar or Az.
                // Large unchanged poloidal fields must not loosen this solve.
                reference_norm = component_dot(base[1], base[1], *owner[1]);
                amrex::ParallelAllReduce::Sum(reference_norm,
                                              amrex::ParallelContext::CommunicatorSub());
            }
            else
#endif
            {
                reference_norm = dot(base, base);
            }
        }
        amrex::Real const roundoff =
            64. * std::numeric_limits<amrex::Real>::epsilon() * std::sqrt(reference_norm);
        amrex::Real const target = std::max({atol, rtol * initial, roundoff});
        amrex::Real truth = initial;
        int iter = 0;
        while (truth > target && iter < max_iterations)
        {
            apply(action, direction, false, compensated);
            amrex::Real const pap = dot(direction, action);
            if (!(pap > 0.) || !std::isfinite(pap))
            {
                result.failure = Failure::Curvature;
                result.iterations = iter;
                result.initial = initial;
                result.residual = truth;
                result.target = target;
                return result;
            }
            amrex::Real const alpha = rr / pap;
            for (int d = 0; d < 3; ++d)
            {
                if (homogeneous && !magnetic)
                {
                    // Carry only the roundoff of the accumulated CG correction.
                    // Every operator/true-residual check uses the represented high
                    // field; only its two-curl evaluation is compensated. No correction
                    // survives between Response calls.
                    for (amrex::MFIter mfi(delta[d], amrex::TilingIfNotGPU());
                         mfi.isValid(); ++mfi)
                    {
                        auto high = delta[d].array(mfi);
                        auto low = delta_low[d].array(mfi);
                        auto step = direction[d].const_array(mfi);
                        amrex::ParallelFor(mfi.tilebox(),
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                            {
                                namespace arithmetic = warpx::ohm::compensated;
                                auto const update = arithmetic::Multiply(
                                    {step(i,j,k), 0.}, alpha);
                                auto const value = arithmetic::Add(
                                    {high(i,j,k), low(i,j,k)}, update);
                                high(i,j,k) = value.hi;
                                low(i,j,k) = value.lo;
                            });
                    }
                }
                else
                    amrex::MultiFab::Saxpy(delta[d], alpha, direction[d], 0, 0, 1, 0);
                amrex::MultiFab::Saxpy(residual[d], -alpha, action[d], 0, 0, 1, 0);
            }
            ++iter;
            amrex::Real next = dot(residual, residual);
            truth = std::sqrt(next);
            bool restart = false;
            if (truth <= target)
            {
                for (int d = 0; d < 3; ++d)
                {
                    amrex::MultiFab::LinComb(input[d], 1., base[d], 0, 1., delta[d], 0, 0, 1, 0);
                }
                // apply() owns input scratch: use direction only after its old
                // value is no longer needed.
                for (int d = 0; d < 3; ++d)
                {
                    amrex::MultiFab::Copy(direction[d], input[d], 0, 0, 1, 0);
                }
                apply(action, direction, !homogeneous, compensated);
                for (int d = 0; d < 3; ++d)
                {
                    amrex::MultiFab::Copy(residual[d], action[d], 0, 0, 1, 0);
                    residual[d].mult(-1., 0, 1, 0);
                }
                next = dot(residual, residual);
                truth = std::sqrt(next);
                restart = true;
            }
            if (truth <= target)
            {
                break;
            }
            for (int d = 0; d < 3; ++d)
            {
                amrex::MultiFab::LinComb(direction[d], 1., residual[d], 0, restart ? 0. : next / rr,
                                         direction[d], 0, 0, 1, 0);
            }
            rr = next;
        }

        result.iterations = iter;
        result.initial = initial;
        result.residual = truth;
        result.target = target;
        if (!std::isfinite(truth))
            result.failure = Failure::Nonfinite;
        else if (truth > target)
            result.failure = Failure::IterationLimit;
        if (compensated)
        {
            // Independent ordinary action on exactly the stored field. This is
            // diagnostic only; the caller's physical gates remain unchanged.
            for (int c=0;c<3;++c)
                MF::LinComb(direction[c],1.,base[c],0,1.,delta[c],0,0,1,0);
            apply(action,direction,false,true);
            amrex::Real const compensated_norm=std::sqrt(dot(action,action));
            apply(action,direction,false,false);
            amrex::Real const ordinary_norm=std::sqrt(dot(action,action));
            amrex::Print()<<std::setprecision(17)<<"COMPENSATED_CORE_RESULT failure="
                <<static_cast<int>(result.failure)<<" iterations="<<result.iterations
                <<" initial="<<result.initial<<" residual="<<result.residual
                <<" target="<<result.target<<" completed_compensated="<<compensated_norm
                <<" completed_ordinary="<<ordinary_norm<<"\n";
        }
        if (!result)
            return result;
        for (int c = 0; c < 3; ++c)
        {
            MF::LinComb(*output[c], 1., base[c], 0, 1., delta[c], 0, 0, 1, 0);
            output[c]->OverrideSync(periodic);
        }
        return result;
    }
};
DarwinVacuumAffineResponse::DarwinVacuumAffineResponse(WarpX &w, View const &e, View const &b,
                                                       Options o)
    : m_impl(std::make_unique<Impl>(w, e, b, o))
{
}
DarwinVacuumAffineResponse::~DarwinVacuumAffineResponse() = default;
bool DarwinVacuumAffineResponse::Freeze(Mask const &correction, Mask const &trace,
                                        View const &baseline, View const &imposed_reference)
{
    auto &p = *m_impl;
    // Revoke before any validation or possible mutation, including failed
    // Freeze attempts. Legacy arithmetic/state remains otherwise unchanged.
    p.retained_freeze_valid=false;
    if(p.retained)p.retained->binding.reset();
    // A failed legacy re-freeze must not expose the old paired origin as an
    // ordinary high-only solve after discarding its low storage.
    if(p.origin_low)p.frozen=false;
    p.origin_low.reset();
    if (!p.layout(baseline))
        return false;
    int bad = 0;
    for (int c = 0; c < 3; ++c)
    {
        for (auto *m : {correction[c], trace[c]})
            if (!m || m->boxArray() != p.origin[c].boxArray() ||
                m->DistributionMap() != p.origin[c].DistributionMap() || m->nComp() != 1)
                return false;
        if (imposed_reference[c] &&
            (imposed_reference[c]->boxArray() != p.origin[c].boxArray() ||
             imposed_reference[c]->DistributionMap() != p.origin[c].DistributionMap() ||
             imposed_reference[c]->nComp() != 1 ||
             !imposed_reference[c]->nGrowVect().allGE(p.input[c].nGrowVect())))
            return false;
        if (!baseline[c]->is_finite(0, 1, 0))
            bad = 1;
        if (imposed_reference[c] &&
            !imposed_reference[c]->is_finite(0, 1, imposed_reference[c]->nGrowVect()))
            bad = 1;
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<int> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(*correction[c]); mfi.isValid(); ++mfi)
        {
            auto m = correction[c]->const_array(mfi);
            auto t = trace[c]->const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                    {
                        return {(m(i, j, k) != 0 && m(i, j, k) != 1) ||
                                (t(i, j, k) != 0 && t(i, j, k) != 1) || (m(i, j, k) && t(i, j, k))};
                    });
        }
        bad = std::max(bad, amrex::get<0>(data.value(op)));
    }
    amrex::ParallelAllReduce::Max(bad, amrex::ParallelContext::CommunicatorSub());
    if (bad)
        return false;
    for (int c = 0; c < 3; ++c)
    {
        MF::Copy(p.origin[c], *baseline[c], 0, 0, 1, 0);
        p.origin[c].OverrideSync(p.periodic);
        amrex::iMultiFab::Copy(p.mask[c], *correction[c], 0, 0, 1, 0);
        amrex::iMultiFab::Copy(p.trace_mask[c], *trace[c], 0, 0, 1, 0);
        p.reference[c].clear();
        if (imposed_reference[c])
        {
            p.reference[c].define(imposed_reference[c]->boxArray(),
                                  imposed_reference[c]->DistributionMap(), 1,
                                  imposed_reference[c]->nGrowVect());
            MF::Copy(p.reference[c], *imposed_reference[c], 0, 0, 1, p.reference[c].nGrowVect());
        }
    }
    p.retained_freeze_valid=true;
    p.frozen = true;
    return true;
}
bool DarwinVacuumAffineResponse::FreezeRetained(Mask const& correction,Mask const& trace,
    View const& high,View const& low,View const& reference)
{
    auto& p=*m_impl;
    // Revoke even if a rank-local layout fails before the legacy Freeze call.
    p.retained_freeze_valid=false;p.frozen=false;p.origin_low.reset();
    if(p.retained)p.retained->binding.reset();
    bool valid=p.layout(high)&&p.layout(low);
    for(int c=0;c<3;++c) {
        for(auto const* mask:{correction[c],trace[c]})
            valid=mask&&mask->boxArray()==p.origin[c].boxArray()&&
                mask->DistributionMap()==p.origin[c].DistributionMap()&&mask->nComp()==1&&valid;
        if(reference[c])valid=reference[c]->boxArray()==p.origin[c].boxArray()&&
            reference[c]->DistributionMap()==p.origin[c].DistributionMap()&&reference[c]->nComp()==1&&
            reference[c]->nGrowVect().allGE(p.input[c].nGrowVect())&&valid;
    }
    // Legacy Freeze conditionally performs a finite reduction for each
    // reference. Agree presence before entering those collectives.
    int presence=0;for(int c=0;c<3;++c)if(reference[c])presence|=1<<c;
    int minimum=presence,maximum=presence;
    amrex::ParallelDescriptor::ReduceIntMin(minimum);
    amrex::ParallelDescriptor::ReduceIntMax(maximum);
    valid=minimum==maximum&&valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);if(!valid)return false;
    for(int c=0;c<3;++c)valid=low[c]->is_finite(0,1,0,true)&&valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);if(!valid)return false;
    bool const frozen=Freeze(correction,trace,high,reference);
    valid=frozen;amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid){p.retained_freeze_valid=false;p.frozen=false;return false;}
    if(!p.retained_scope()){p.retained_freeze_valid=false;p.frozen=false;return false;}
    auto copy=std::make_unique<Field>();
    for(int c=0;c<3;++c) {
        (*copy)[c].define(p.origin[c].boxArray(),p.origin[c].DistributionMap(),1,0);
        MF::Copy((*copy)[c],*low[c],0,0,1,0);(*copy)[c].OverrideSync(p.periodic);
    }
    p.origin_low=std::move(copy);return true;
}
DarwinVacuumAffineResponse::Result DarwinVacuumAffineResponse::Recover(View const &out,
                                                                       View const *inc)
{
    return m_impl->solve(out, inc, false);
}
DarwinVacuumAffineResponse::Result DarwinVacuumAffineResponse::Response(View const &out,
                                                                        View const &inc)
{
    return m_impl->solve(out, &inc, true);
}
void DarwinVacuumAffineResponse::ApplyMaskedCurl(View const &out, View const &in, bool affine)
{
    auto &p = *m_impl;
    AMREX_ALWAYS_ASSERT(p.frozen && p.layout(out) && p.layout(in));
    for (auto *a : out)
        for (auto *b : in)
            AMREX_ALWAYS_ASSERT(a != b);
    p.apply(out, in, affine);
}
bool DarwinVacuumAffineResponse::ApplyCompensatedMaskedCurl(View const &out, View const &in)
{
    auto &p = *m_impl;
    if (!p.frozen || !p.compensated_supported || !p.layout(out) || !p.layout(in)) return false;
    for (auto *a : out)
        for (auto *b : in)
            AMREX_ALWAYS_ASSERT(a != b);
    p.apply(out, in, false, true);
    return true;
}
DarwinVacuumAffineResponse::RetainedRecovery::RetainedRecovery()=default;
DarwinVacuumAffineResponse::RetainedRecovery::~RetainedRecovery()=default;
DarwinVacuumAffineResponse::RetainedRecovery::RetainedRecovery(RetainedRecovery&&) noexcept=default;
DarwinVacuumAffineResponse::RetainedRecovery&
DarwinVacuumAffineResponse::RetainedRecovery::operator=(RetainedRecovery&&) noexcept=default;
bool DarwinVacuumAffineResponse::RetainedRecovery::Ready()const noexcept
{
    if(!m_storage)return false;
    auto const binding=m_storage->binding.lock();
    return binding && m_storage->receipt.ready;
}
amrex::MultiFab const& DarwinVacuumAffineResponse::RetainedRecovery::High(int c)const
{AMREX_ALWAYS_ASSERT(Ready() && c>=0 && c<3);return m_storage->high[c];}
amrex::MultiFab const& DarwinVacuumAffineResponse::RetainedRecovery::Low(int c)const
{AMREX_ALWAYS_ASSERT(Ready() && c>=0 && c<3);return m_storage->low[c];}
DarwinVacuumAffineResponse::RetainedReceipt const&
DarwinVacuumAffineResponse::RetainedRecovery::Receipt()const
{AMREX_ALWAYS_ASSERT(Ready());return m_storage->receipt;}
DarwinVacuumAffineResponse::RetainedReceipt
DarwinVacuumAffineResponse::RecoverRetained(RetainedRecovery& out,View const* increment)
{return m_impl->solve_retained(out,increment);}
DarwinVacuumAffineResponse::RetainedReceipt
DarwinVacuumAffineResponse::ResponseRetained(RetainedRecovery& out,View const& increment)
{return m_impl->solve_retained(out,&increment,true);}
DarwinVacuumAffineResponse::RetainedReceipt
DarwinVacuumAffineResponse::RecoverRetainedPair(RetainedRecovery& out,View const& high,View const& low)
{return m_impl->solve_retained(out,&high,false,&low);}
DarwinVacuumAffineResponse::RetainedReceipt
DarwinVacuumAffineResponse::ResponseRetainedPair(RetainedRecovery& out,View const& high,View const& low)
{return m_impl->solve_retained(out,&high,true,&low);}
DarwinVacuumAffineResponse::RetainedReceipt
DarwinVacuumAffineResponse::ReconstructAffineRetained(RetainedRecovery& output,
    View const& high,View const& low,View const* increment,View const* increment_low)
{
    auto& p=*m_impl;
    RetainedReceipt receipt;
    int mode_min=(increment?1:0)+(increment_low?2:0),mode_max=mode_min;
    amrex::ParallelDescriptor::ReduceIntMin(mode_min);
    amrex::ParallelDescriptor::ReduceIntMax(mode_max);
    if(mode_min!=mode_max || (increment_low && !increment))
    {receipt.solve.failure=Failure::InvalidTrace;return receipt;}
    receipt.affine_trace=increment_low?AffineTraceInput::Pair:
        (increment?AffineTraceInput::High:AffineTraceInput::None);
    bool valid=p.frozen && p.retained_freeze_valid;
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid){receipt.solve.failure=Failure::NotFrozen;return receipt;}
    valid=NativeEndpointArithmeticSupported();
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid || !p.retained_scope())
    {receipt.solve.failure=Failure::UnsupportedRetained;return receipt;}
    valid=p.layout(high) && p.layout(low) &&
        (!increment || p.layout(*increment)) && (!increment_low || p.layout(*increment_low));
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid){receipt.solve.failure=Failure::InvalidLayout;return receipt;}
    int bad=0;
    for(auto const* part:{increment,increment_low})if(part)for(int c=0;c<3;++c) {
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);
        using T=decltype(data)::Type;
        for(amrex::MFIter it(*(*part)[c]);it.isValid();++it) {
            auto x=(*part)[c]->const_array(it);auto t=p.trace_mask[c].const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T
                {return {!std::isfinite(x(i,j,k)) || (!t(i,j,k) && x(i,j,k)!=0.)};});
        }
        bad=std::max(bad,amrex::get<0>(data.value(op)));
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    if(bad){receipt.solve.failure=Failure::InvalidTrace;return receipt;}
    if(!p.retained)p.retained=std::make_unique<Impl::RetainedScratch>(p);
    auto& q=*p.retained;
    if(!q.binding)q.binding=std::make_shared<RetainedBinding>();
    receipt.norm_roundoff_factor=q.norm_factor;

    // Snapshot every input before touching the owned result. A caller may
    // supply views of an older recovery, whose storage remains alive until
    // all checks complete. The frozen origin and masks are never changed.
    for(int c=0;c<3;++c){
        MF::Copy(q.assembled[c],*high[c],0,0,1,0);
        MF::Copy(q.assembled_low[c],*low[c],0,0,1,0);
        MF::Copy(p.base[c],p.origin[c],0,0,1,0);
        q.base_low[c].setVal(0.);
        if(p.origin_low)MF::Copy(q.base_low[c],(*p.origin_low)[c],0,0,1,0);
        // Reproduce the original affine base from the actual numeric trace
        // operands. Trace-kind metadata alone loses off-trace signed zeros.
        if(increment) {
            if(!p.origin_low && !increment_low) {
                MF::Add(p.base[c],*(*increment)[c],0,0,1,0);
            } else {
                bool const paired=increment_low!=nullptr;
                for(amrex::MFIter it(p.base[c],amrex::TilingIfNotGPU());it.isValid();++it) {
                    auto h=p.base[c].array(it),l=q.base_low[c].array(it);
                    auto dh=(*increment)[c]->const_array(it);
                    auto dl=paired?(*increment_low)[c]->const_array(it):amrex::Array4<amrex::Real const>{};
                    auto t=p.trace_mask[c].const_array(it);
                    amrex::ParallelFor(it.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                        if(t(i,j,k)) {
                            auto const value=warpx::ohm::compensated::Add({h(i,j,k),l(i,j,k)},
                                {dh(i,j,k),paired?dl(i,j,k):0.});
                            h(i,j,k)=value.hi;l(i,j,k)=value.lo;
                        }
                    });
                }
            }
        }
        p.base[c].OverrideSync(p.periodic);q.base_low[c].OverrideSync(p.periodic);
    }
    if(!p.finite_pair(q.assembled,q.assembled_low))
    {receipt.solve.failure=Failure::Nonfinite;return receipt;}
    bad=0;
    for(int c=0;c<3;++c){
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);
        using T=decltype(data)::Type;
        for(amrex::MFIter it(p.base[c]);it.isValid();++it){
            auto h=q.assembled[c].const_array(it),l=q.assembled_low[c].const_array(it);
            auto b=p.base[c].const_array(it),bl=q.base_low[c].const_array(it);
            auto m=p.mask[c].const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                // Every fixed row, including the supplied plasma trace,
                // must match the original affine-base arithmetic exactly.
                return {!m(i,j,k) && (h(i,j,k)!=b(i,j,k) || l(i,j,k)!=bl(i,j,k) ||
                    std::signbit(h(i,j,k))!=std::signbit(b(i,j,k)) ||
                    std::signbit(l(i,j,k))!=std::signbit(bl(i,j,k)))};
            });
        }
        bad=std::max(bad,amrex::get<0>(data.value(op)));
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    if(bad){receipt.solve.failure=Failure::InvalidTrace;return receipt;}

    // Importing a checkpoint must not silently repair inconsistent nodal or
    // periodic aliases. Sync private copies and require exact represented
    // values on every valid replica, including signed zero.
    for(int c=0;c<3;++c){
        q.assembled[c].OverrideSync(p.periodic);q.assembled_low[c].OverrideSync(p.periodic);
        amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<int> data(op);
        using T=decltype(data)::Type;
        for(amrex::MFIter it(q.assembled[c]);it.isValid();++it){
            auto h=q.assembled[c].const_array(it),l=q.assembled_low[c].const_array(it);
            auto a=high[c]->const_array(it),b=low[c]->const_array(it);
            op.eval(it.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->T {
                return {h(i,j,k)!=a(i,j,k) || l(i,j,k)!=b(i,j,k) ||
                    std::signbit(h(i,j,k))!=std::signbit(a(i,j,k)) ||
                    std::signbit(l(i,j,k))!=std::signbit(b(i,j,k))};
            });
        }
        bad=std::max(bad,amrex::get<0>(data.value(op)));
    }
    amrex::ParallelDescriptor::ReduceIntMax(bad);
    if(bad){receipt.solve.failure=Failure::InvalidTrace;return receipt;}
    receipt.fixed_trace_exact=true;

    // RecoverRetained starts its range correction at zero. Reproduce that
    // initial residual from the frozen origin plus the saved trace, rather
    // than trusting a serialized success flag, residual or tolerance.
    for(int c=0;c<3;++c){p.delta[c].setVal(0.);p.delta_low[c].setVal(0.);}
    p.apply_pair(q.base_action,q.base_action_low,p.base,q.base_low);
    p.retained_residual();
    receipt.solve.initial=p.retained_norm(p.residual,q.residual_low);
    receipt.solve.target=std::max(p.atol,p.rtol*std::sqrt(p.dot(p.residual,p.residual)));
    if(!p.finite_pair(p.residual,q.residual_low) ||
       !std::isfinite(receipt.solve.initial) || !std::isfinite(receipt.solve.target) ||
       receipt.solve.target<0.)
    {receipt.solve.failure=Failure::Nonfinite;return receipt;}
    p.apply_pair(p.action,q.action_low,q.assembled,q.assembled_low);
    receipt.assembled_residual=p.retained_norm(p.action,q.action_low);
    receipt.solve.residual=receipt.assembled_residual;
    if(!std::isfinite(receipt.assembled_residual))
    {receipt.solve.failure=Failure::Nonfinite;return receipt;}
    if(receipt.assembled_residual>receipt.solve.target)
    {receipt.solve.failure=Failure::IterationLimit;return receipt;}
    p.apply_pair(p.action,q.action_low,q.assembled,q.zero);
    receipt.materialized_compensated=p.retained_norm(p.action,q.action_low);
    p.apply(p.action,q.assembled,false,false);
    receipt.materialized_ordinary=p.retained_norm(p.action,q.zero);
    p.apply_pair(p.action,q.action_low,q.assembled_low,q.zero);
    receipt.discarded_action=p.retained_norm(p.action,q.action_low);
    if(!std::isfinite(receipt.materialized_compensated) ||
       !std::isfinite(receipt.materialized_ordinary) || !std::isfinite(receipt.discarded_action))
    {receipt.solve.failure=Failure::Nonfinite;return receipt;}
    receipt.single_field_within_target=receipt.materialized_compensated<=receipt.solve.target &&
        receipt.materialized_ordinary<=receipt.solve.target;
    auto completed=std::make_unique<RetainedRecovery::Storage>();
    for(int c=0;c<3;++c){
        for(auto* f:{&completed->high,&completed->low})
            (*f)[c].define(p.base[c].boxArray(),p.base[c].DistributionMap(),1,0);
        MF::Copy(completed->high[c],q.assembled[c],0,0,1,0);
        MF::Copy(completed->low[c],q.assembled_low[c],0,0,1,0);
    }
    receipt.ready=true;completed->receipt=receipt;completed->binding=q.binding;
    output.m_storage=std::move(completed);
    return receipt;
}
bool DarwinVacuumAffineResponse::MatchesRetained(RetainedRecovery const& value)const noexcept
{
    return m_impl->frozen && m_impl->retained_freeze_valid && m_impl->retained && value.Ready() &&
        value.m_storage->binding.lock()==m_impl->retained->binding;
}
bool DarwinVacuumAffineResponse::ApplyRetainedMaskedCurl(View const& high,View const& low,
                                                         RetainedRecovery const& value)
{return ApplyRetainedCurl(high,low,value,false);}
bool DarwinVacuumAffineResponse::ApplyRetainedGlobalCurl(View const& high,View const& low,
                                                         RetainedRecovery const& value)
{return ApplyRetainedCurl(high,low,value,true);}
bool DarwinVacuumAffineResponse::ApplyRetainedCurl(View const& high,View const& low,
                                                  RetainedRecovery const& value,bool global)
{
    auto& p=*m_impl;
    // Every consumer rechecks precise arithmetic and its actual frozen scope;
    // a prior receipt cannot authorize a changed host environment.
    if(!p.retained_scope())return false;
    int mode_min=global?1:0,mode_max=mode_min;
    amrex::ParallelDescriptor::ReduceIntMin(mode_min);
    amrex::ParallelDescriptor::ReduceIntMax(mode_max);
    bool valid=mode_min==mode_max && MatchesRetained(value) && p.layout(high) && p.layout(low);
    if(valid)
    {
        std::array<MF const*,6> const outputs{high[0],high[1],high[2],low[0],low[1],low[2]};
        std::array<MF const*,6> const inputs{&value.m_storage->high[0],&value.m_storage->high[1],
            &value.m_storage->high[2],&value.m_storage->low[0],&value.m_storage->low[1],
            &value.m_storage->low[2]};
        for(std::size_t i=0;i<outputs.size();++i)
        {
            for(std::size_t j=0;j<i;++j)
                valid=!StorageOverlaps(*outputs[i],*outputs[j]) && valid;
            for(auto const* input:inputs)
                valid=!StorageOverlaps(*outputs[i],*input) && valid;
        }
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(valid);
    if(!valid)return false;
    auto& q=*p.retained;
    p.apply_pair(p.action,q.action_low,value.m_storage->high,value.m_storage->low,global);
    if(!p.finite_pair(p.action,q.action_low))return false;
    for(int c=0;c<3;++c)
    {
        MF::Copy(*high[c],p.action[c],0,0,1,0);
        MF::Copy(*low[c],q.action_low[c],0,0,1,0);
    }
    return true;
}
amrex::Real DarwinVacuumAffineResponse::MetricDot(View const &a, View const &b) const
{
    AMREX_ALWAYS_ASSERT(m_impl->layout(a) && m_impl->layout(b));
    return m_impl->dot(a, b);
}
amrex::Real DarwinVacuumAffineResponse::Diagonal() const { return m_impl->diagonal; }
} // namespace warpx::darwin
