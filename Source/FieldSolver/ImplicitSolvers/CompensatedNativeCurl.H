/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_COMPENSATED_NATIVE_CURL_H_
#define WARPX_COMPENSATED_NATIVE_CURL_H_

#include "FieldSolver/FiniteDifferenceSolver/CompensatedTransverseOhm.H"
#include "EmbeddedBoundary/Enabled.H"
#include "WarpX.H"
#if defined(WARPX_DIM_RZ)
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#elif defined(WARPX_DIM_3D)
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

// Internal arithmetic companion of the native two-curl operator. No boundary
// map, mask, coefficient, input field, or null-space selection is changed here.
namespace warpx::darwin::compensated_curl
{
using Real = amrex::Real;
using Pair = warpx::ohm::compensated::Pair;
namespace arithmetic = warpx::ohm::compensated;
using View = ablastr::fields::VectorField;
struct Coefficients
{
    amrex::GpuArray<Real, 3> inverse{};
    Real dr = 0., rmin = 0.;
};
inline bool Supported(WarpX const& w)
{
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    bool valid = w.finestLevel() == 0 && !EB::enabled() &&
        WarpX::grid_type == ablastr::utils::enums::GridType::Staggered &&
        WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC;
#if defined(WARPX_DIM_RZ)
    valid = valid && WarpX::n_rz_azimuthal_modes == 1;
#endif
    return valid;
#else
    amrex::ignore_unused(w);
    return false;
#endif
}
inline Coefficients NativeCoefficients(WarpX const& w)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Supported(w),
        "Compensated native curl requires single-level non-EB staggered HybridPIC RZ m0 or 3D");
    auto cell_size = WarpX::CellSize(0); // exact FD constructor argument
    amrex::Vector<Real> x, y, z;
    Coefficients c;
#if defined(WARPX_DIM_RZ)
    CylindricalYeeAlgorithm::InitializeStencilCoefficients(cell_size, x, z);
    AMREX_ALWAYS_ASSERT(x.size() == 1 && z.size() == 1);
    c.inverse = {x[0], 0., z[0]};
    c.dr = cell_size[0];
    c.rmin = w.Geom(0).ProbLo(0);
#elif defined(WARPX_DIM_3D)
    CartesianYeeAlgorithm::InitializeStencilCoefficients(cell_size, x, y, z);
    AMREX_ALWAYS_ASSERT(x.size() == 1 && y.size() == 1 && z.size() == 1);
    c.inverse = {x[0], y[0], z[0]};
#endif
    return c;
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Subtract(Pair a, Pair b) noexcept
{
    return arithmetic::Add(a, arithmetic::Negate(b));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Product(Pair a, Pair b) noexcept
{
    return arithmetic::Add(arithmetic::Multiply(a, b.hi),
                           arithmetic::Multiply(a, b.lo));
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Quotient(Pair a, Real b) noexcept
{
    Real const q = a.hi / b;
    Real const r = std::fma(-q, b, a.hi) + a.lo;
    return arithmetic::Sum(q, r / b);
}
struct SingleRead
{
    amrex::GpuArray<amrex::Array4<Real const>, 3> value;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Pair operator()(int c, int i, int j, int k) const noexcept
    { return {value[c](i,j,k), 0.}; }
};
struct PairRead
{
    amrex::GpuArray<amrex::Array4<Real const>, 3> high, low;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Pair operator()(int c, int i, int j, int k) const noexcept
    { return {high[c](i,j,k), low[c](i,j,k)}; }
};
template<class Read>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair Difference(Read const& f, int c, int axis, int i, int j, int k,
                bool upward, Real inverse) noexcept
{
    int const direction = upward ? 1 : -1;
    Pair const here = f(c,i,j,k);
    Pair const other = f(c,i+direction*int(axis==0),
                         j+direction*int(axis==1), k+direction*int(axis==2));
    return arithmetic::Multiply(upward ? Subtract(other,here) : Subtract(here,other),
                                 inverse);
}
template<class Read>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair RadialWeighted(Read const& f, int component, int i, int j, int k,
                    bool upward, Real r, Coefficients const& c) noexcept
{
    // Native association: (1/r * inv_dr) * ((r+.5*dr)*Fplus
    //                                      - (r-.5*dr)*Fminus).
    // r is the exact same represented native coordinate; its reciprocal is
    // the same scalar operand. Products and sums retain their low terms.
    Pair const plus_radius = arithmetic::Sum(r, .5*c.dr);
    Pair const minus_radius = arithmetic::Sum(r, -.5*c.dr);
    Pair const weighted = Subtract(
        Product(plus_radius, f(component,i+int(upward),j,k)),
        Product(minus_radius, f(component,i-int(!upward),j,k)));
    Pair const factor = arithmetic::Multiply({1./r,0.},c.inverse[0]);
    return Product(factor, weighted);
}
template<class Read>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair CurlUp(Read const& f, int component, int i, int j, [[maybe_unused]] int k,
            Coefficients const& c) noexcept
{
#if defined(WARPX_DIM_RZ)
    if (component == 0)
    {
        Real const r = c.rmin + i*c.dr;
        return r != 0. ? arithmetic::Negate(
            Difference(f,1,1,i,j,0,true,c.inverse[2])) : Pair{0.,0.};
    }
    if (component == 1)
        return arithmetic::Negate(Subtract(
            Difference(f,2,0,i,j,0,true,c.inverse[0]),
            Difference(f,0,1,i,j,0,true,c.inverse[2])));
    Real const r = c.rmin + (i+.5)*c.dr;
    return RadialWeighted(f,1,i,j,0,true,r,c);
#elif defined(WARPX_DIM_3D)
    int const a = (component+1)%3, b = (component+2)%3;
    return arithmetic::Add(arithmetic::Negate(Difference(f,a,b,i,j,k,true,c.inverse[b])),
                             Difference(f,b,a,i,j,k,true,c.inverse[a]));
#else
    amrex::ignore_unused(f,component,i,j,k,c);
    return {0.,0.};
#endif
}
template<class Read>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Pair CurrentDown(Read const& f, int component, int i, int j, [[maybe_unused]] int k,
                 Coefficients const& c) noexcept
{
    Real const one_over_mu0 = 1. / PhysConst::mu0;
    Pair curl{0.,0.};
#if defined(WARPX_DIM_RZ)
    if (component == 0)
        curl = arithmetic::Negate(Difference(f,1,1,i,j,0,false,c.inverse[2]));
    else
    {
        Real const r = c.rmin + i*c.dr;
        if (component == 1)
        {
            if (r > .5*c.dr)
                curl = arithmetic::Add(
                    arithmetic::Negate(Difference(f,2,0,i,j,0,false,c.inverse[0])),
                    Difference(f,0,1,i,j,0,false,c.inverse[2]));
        }
        else if (r > .5*c.dr)
            curl = RadialWeighted(f,1,i,j,0,false,r,c);
        else
        {
            // Native axis association: (one_over_mu0 * 4) * Btheta / dr.
            return Quotient(arithmetic::Multiply(f(1,i,j,0),one_over_mu0*4.),c.dr);
        }
    }
#elif defined(WARPX_DIM_3D)
    int const a = (component+1)%3, b = (component+2)%3;
    curl = arithmetic::Add(arithmetic::Negate(Difference(f,a,b,i,j,k,false,c.inverse[b])),
                             Difference(f,b,a,i,j,k,false,c.inverse[a]));
#else
    amrex::ignore_unused(f,component,i,j,k,c);
#endif
    return arithmetic::Multiply(curl,one_over_mu0);
}

using namespace amrex;
    // A physical curl guard also needs its tangential box guards: the next
    // curl/interpolation can read their intersection at a physical boundary.
    // FillBoundary cannot obtain those corners from any valid cell. Complete
    // them from A without extending the requested physical-boundary region.
    [[maybe_unused]] inline Box CurlEvaluationBox (
        MFIter const& mfi, MultiFab const& field, IntVect const& physical_grow,
        Geometry const& geom)
    {
        if (physical_grow == IntVect::TheZeroVector()) {
            return mfi.tilebox(field.ixType().toIntVect());
        }
        IntVect const local_grow = amrex::max(physical_grow, IntVect(1));
        IntVect domain_grow = physical_grow;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (geom.isPeriodic(d)) {
                domain_grow[d] = local_grow[d];
            }
        }
        return mfi.tilebox(field.ixType().toIntVect(), local_grow)
            & amrex::grow(amrex::convert(geom.Domain(), field.ixType()), domain_grow);
    }

    // Check actual integer-index footprints before any grown curl launch. Yee
    // differences read p and p+e_d; collocated differences read p-e_d and p+e_d.
    // setType preserves the index bounds (convert would change them).
    [[maybe_unused]] inline void AssertCurlStorage (
        MFIter const& mfi, MultiFab const& output,
        ablastr::fields::VectorField const& input, int const component,
        Box const& region, bool const collocated, iMultiFab const* eb_flag,
        bool const axis_mode = false)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(output[mfi].box().contains(region),
            "ComputeCurlA: output allocation does not contain the completed curl guards");
        if (eb_flag) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE((*eb_flag)[mfi].box().contains(region),
                "ComputeCurlA: EB flag allocation does not contain the completed curl guards");
        }
        for (int c = 0; c < 3; ++c) {
            if (c == component) { continue; }
            int const physical_direction = 3 - component - c;
            int direction = physical_direction;
#if defined(WARPX_DIM_RCYLINDER)
            direction = physical_direction == 0 ? 0 : -1;
#elif AMREX_SPACEDIM == 2
            direction = physical_direction == 2 ? 1 : (physical_direction == 0 ? 0 : -1);
#elif AMREX_SPACEDIM == 1
            direction = physical_direction == 2 ? 0 : -1;
#endif
            Box required = region;
            required.setType(input[c]->ixType());
            if (direction >= 0) {
                required.growHi(direction, 1);
                if (collocated) { required.growLo(direction, 1); }
            }
            // The higher-mode cylindrical Br axis formula also samples Az at r=dr.
            if (axis_mode && component == 0 && c == 2 && required.smallEnd(0) == 0) {
                required.setBig(0, amrex::max(required.bigEnd(0), 1));
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE((*input[c])[mfi].box().contains(required),
                "ComputeCurlA: input allocation does not contain the completed curl stencil");
        }
    }

inline void First(View const& high, View const& low, View const& input,
                  Coefficients const& coeff, amrex::Geometry const& geom,
                  amrex::IntVect const& grow)
{
    for (int c=0;c<3;++c) { high[c]->setVal(0.); low[c]->setVal(0.); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*input[0],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
    {
        SingleRead const read{{input[0]->const_array(mfi),input[1]->const_array(mfi),
                              input[2]->const_array(mfi)}};
        for (int component=0;component<3;++component)
        {
            auto box = CurlEvaluationBox(mfi,*high[component],grow,geom);
#if defined(WARPX_DIM_RZ)
            if (grow != amrex::IntVect::TheZeroVector() && coeff.rmin == 0.)
                box.setSmall(0,amrex::max(box.smallEnd(0),0));
#endif
            if (grow != amrex::IntVect::TheZeroVector())
                AssertCurlStorage(mfi,*high[component],input,component,box,false,nullptr,false);
            AMREX_ALWAYS_ASSERT((*low[component])[mfi].box().contains(box));
            auto hi = high[component]->array(mfi), lo = low[component]->array(mfi);
            amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
            {
                Pair const value = CurlUp(read,component,i,j,k,coeff);
                hi(i,j,k)=value.hi; lo(i,j,k)=value.lo;
            });
        }
    }
}
// Additive paired-input action. First remains unchanged for existing callers.
inline void FirstPaired(View const& high, View const& low, View const& input_high,
                        View const& input_low, Coefficients const& coeff,
                        amrex::Geometry const& geom, amrex::IntVect const& grow)
{
    for (int c=0;c<3;++c) { high[c]->setVal(0.); low[c]->setVal(0.); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*input_high[0],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
    {
        PairRead const read{
            {input_high[0]->const_array(mfi),input_high[1]->const_array(mfi),
             input_high[2]->const_array(mfi)},
            {input_low[0]->const_array(mfi),input_low[1]->const_array(mfi),
             input_low[2]->const_array(mfi)}};
        for (int component=0;component<3;++component)
        {
            auto box=CurlEvaluationBox(mfi,*high[component],grow,geom);
#if defined(WARPX_DIM_RZ)
            if (grow!=amrex::IntVect::TheZeroVector() && coeff.rmin==0.)
                box.setSmall(0,amrex::max(box.smallEnd(0),0));
#endif
            AssertCurlStorage(mfi,*high[component],input_high,component,box,false,nullptr,false);
            AssertCurlStorage(mfi,*low[component],input_low,component,box,false,nullptr,false);
            auto hi=high[component]->array(mfi),lo=low[component]->array(mfi);
            amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
            {
                Pair const value=CurlUp(read,component,i,j,k,coeff);
                hi(i,j,k)=value.hi;lo(i,j,k)=value.lo;
            });
        }
    }
}
inline void Second(View const& high, View const& low, View const& curl_high,
                   View const& curl_low, Coefficients const& coeff)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*high[0],amrex::TilingIfNotGPU());mfi.isValid();++mfi)
    {
        PairRead const read{{curl_high[0]->const_array(mfi),curl_high[1]->const_array(mfi),
                             curl_high[2]->const_array(mfi)},
                            {curl_low[0]->const_array(mfi),curl_low[1]->const_array(mfi),
                             curl_low[2]->const_array(mfi)}};
        for (int component=0;component<3;++component)
        {
            auto const box=mfi.tilebox(high[component]->ixType().toIntVect(),amrex::IntVect(1));
            AMREX_ALWAYS_ASSERT((*high[component])[mfi].box().contains(box) &&
                                (*low[component])[mfi].box().contains(box));
            auto hi=high[component]->array(mfi),lo=low[component]->array(mfi);
            amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept
            {
                Pair const value=CurrentDown(read,component,i,j,k,coeff);
                hi(i,j,k)=value.hi;lo(i,j,k)=value.lo;
            });
        }
    }
}
} // namespace warpx::darwin::compensated_curl
#endif
