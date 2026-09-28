/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Particles/Gather/FieldGather.H"
#include <AMReX_Geometry.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <array>
#include <limits>
using namespace amrex;
using namespace warpx::particles;
using PR = ParticleReal;
using Point = GpuArray<PR, 3>;
using Arrays = GpuArray<Array4<Real const>, 6>;
using Types = GpuArray<IndexType, 6>;
using Values = GpuArray<PR, 6>;

// Actual unchecked native kernel is the numerical oracle for admissible inputs.
AMREX_GPU_DEVICE void
native (Point const& x, Values& v, Arrays const& a, Types const& t,
        XDim3 const& inv, XDim3 const& origin, Dim3 const& lo, int modes,
        int order) {
    doGatherShapeN(x[0], x[1], x[2], v[0], v[1], v[2], v[3], v[4], v[5], a[0],
                   a[1], a[2], a[3], a[4], a[5], t[0], t[1], t[2], t[3], t[4],
                   t[5], inv, origin, lo, modes, order, false);
}
AMREX_GPU_DEVICE ImplicitGatherResult
checked (Point const& old, Point const& x, Values& v, Arrays const& a,
         Types const& t, XDim3 const& inv, XDim3 const& origin, Dim3 const& lo,
         int modes, int order,
         GpuArray<GpuArray<double, 2>, AMREX_SPACEDIM> const& domain,
         GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> const& crop) {
    return doGatherShapeNImplicit(
        old[0], old[1], old[2], x[0], x[1], x[2], v[0], v[1], v[2], v[3], v[4],
        v[5], a[0], a[1], a[2], a[3], a[4], a[5], t[0], t[1], t[2], t[3], t[4],
        t[5], inv, origin, domain, crop, lo, modes, order,
        CurrentDepositionAlgo::Direct);
}
AMREX_GPU_DEVICE bool
same (Values const& a, Values const& b) {
    for (int f = 0; f < 6; ++f)
        if (a[f] != b[f])
            return false;
    return true;
}
AMREX_GPU_HOST_DEVICE Point
point (Real x, Real y, Real z) {
#ifdef WARPX_DIM_RZ
    amrex::ignore_unused(y);
    return {PR(x), PR(0), PR(z)};
#else
    return {PR(x), PR(y), PR(z)};
#endif
}
int
main (int argc, char** argv) {
    Initialize(argc, argv);
    {
        ParmParse pp("test");
        int order = 3, box = 4;
        std::string layout = "nodal";
        pp.query("order", order);
        pp.query("box", box);
        pp.query("layout", layout);
        std::string unsafe;
        pp.query("unguarded", unsafe);
        struct Context {
            Geometry geom;
            BoxArray ba;
            DistributionMapping dm;
        } context;
        Box const domain_box(IntVect(0), IntVect(15));
        RealBox const physical({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(2., 3., 2.5)});
        int const periods[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 1, 1)};
#ifdef WARPX_DIM_RZ
        constexpr int coord = 1;
#else
        constexpr int coord = 0;
#endif
        context.geom = Geometry(domain_box, &physical, coord, periods);
        context.ba = BoxArray(domain_box);
        context.ba.maxSize(box);
        context.dm = DistributionMapping(context.ba);
        std::array<std::unique_ptr<MultiFab>, 6> fields;
        Types types;
        int const modes = 2;
        for (int f = 0; f < 6; ++f) {
            IntVect ix = IntVect::TheNodeVector();
            if (layout == "yee") {
                int const component = f % 3;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#ifdef WARPX_DIM_RZ
                    int const physical = d == 0 ? 0 : 2;
#else
                    int const physical = d;
#endif
                    ix[d] =
                        f < 3 ? component != physical : component == physical;
                }
            }
            types[f] = IndexType(ix);
            fields[f] = std::make_unique<MultiFab>(convert(context.ba, ix),
                                                   context.dm, 3, 3);
            for (MFIter mfi(*fields[f]); mfi.isValid(); ++mfi) {
                auto a = fields[f]->array(mfi);
                auto const full = mfi.fabbox();
                ParallelFor(
                    full, 3, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                        a(i, j, k, n) = Real(f + 1) +
                                        Real(.001) * Real(i + 2 * j + 3 * k) +
                                        Real(.03) * n;
                    });
            }
        }
        std::array<std::unique_ptr<MultiFab>, 6> nonfinite;
        for (int f = 0; f < 6; ++f) {
            nonfinite[f] = std::make_unique<MultiFab>(fields[f]->boxArray(),
                                                      context.dm, 3, 3);
            nonfinite[f]->setVal(std::numeric_limits<Real>::quiet_NaN());
        }
        std::array<Real, 6> before{};
        for (int f = 0; f < 6; ++f)
            before[f] = fields[f]->norm1(0, 3);
        Gpu::Buffer<Long> tally({0, 0, 0});
        auto* counts = tally.data();
        auto const dx = context.geom.CellSizeArray(),
                   pl = context.geom.ProbLoArray();
        XDim3 inv{1 / dx[0], 1, 1 / dx[AMREX_SPACEDIM - 1]};
#ifdef WARPX_DIM_3D
        inv.y = 1 / dx[1];
#endif
        for (MFIter mfi(*fields[0]); mfi.isValid(); ++mfi) {
            Arrays arrays, nan_arrays;
            for (int f = 0; f < 6; ++f) {
                arrays[f] = fields[f]->const_array(mfi);
                nan_arrays[f] = nonfinite[f]->const_array(mfi);
            }
            Box tile = enclosedCells(mfi.validbox());
            // Keep the same cell-box reference origin the production pusher
            // uses.
            tile = context.ba[mfi.index()];
            Box grown = grow(tile, 3);
            Dim3 lo = lbound(grown);
            XDim3 origin{pl[0] + lo.x * dx[0], 0,
                         pl[AMREX_SPACEDIM - 1] +
                             lo.y * dx[AMREX_SPACEDIM - 1]};
#ifdef WARPX_DIM_3D
            origin.y = pl[1] + lo.y * dx[1];
            origin.z = pl[2] + lo.z * dx[2];
#endif
            auto const center = tile.smallEnd();
            Point x = point(pl[0] + (center[0] + Real(.25)) * dx[0],
#ifdef WARPX_DIM_3D
                            pl[1] + (center[1] + Real(.25)) * dx[1],
#else
                            0,
#endif
                            pl[AMREX_SPACEDIM - 1] +
                                (center[AMREX_SPACEDIM - 1] + Real(.25)) *
                                    dx[AMREX_SPACEDIM - 1]);
            GpuArray<GpuArray<double, 2>, AMREX_SPACEDIM> domain{};
            GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> crop{};
            auto const dim = tile.length();
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                domain[d] = {double(context.geom.Domain().smallEnd(d) -
                                    grown.smallEnd(d)),
                             double(context.geom.Domain().bigEnd(d) + 1 -
                                    grown.smallEnd(d))};
            bool const touches_cap =
                tile.bigEnd(AMREX_SPACEDIM - 1) ==
                context.geom.Domain().bigEnd(AMREX_SPACEDIM - 1);
            amrex::ignore_unused(dim);
            int const unsafe_case = unsafe == "nan"     ? 1
                                    : unsafe == "range" ? 2
                                                        : 0;
            amrex::For(1, [=] AMREX_GPU_DEVICE(int) {
                auto verify = [&] (bool ok) {
                    Gpu::Atomic::Add(counts, Long(1));
                    if (!ok)
                        Gpu::Atomic::Add(counts + 1, Long(1));
                };
                Values reference{}, actual{};
                if (unsafe_case) {
                    Point bad = x;
                    bad[2] = unsafe_case == 1
                                 ? std::numeric_limits<PR>::quiet_NaN()
                                 : PR(1.e6);
                    native(bad, actual, arrays, types, inv, origin, lo, modes,
                           order);
                    return;
                }
                native(x, reference, arrays, types, inv, origin, lo, modes,
                       order);
                auto ok = checked(x, x, actual, arrays, types, inv, origin, lo,
                                  modes, order, domain, crop);
                verify(bool(ok) && same(reference, actual));
                // A/B/A. Rejected finite/nonfinite positions must not read or
                // change the six accumulated output values.
                for (int which = 0; which < 4; ++which) {
                    Point bad = x;
                    bad[2] = which == 0   ? std::numeric_limits<PR>::quiet_NaN()
                             : which == 1 ? std::numeric_limits<PR>::infinity()
                             : which == 2 ? PR(1.e20)
                                          : PR(-1.e20);
                    Values untouched{1, 2, 3, 4, 5, 6}, output = untouched;
                    auto result =
                        checked(x, bad, output, arrays, types, inv, origin, lo,
                                modes, order, domain, crop);
                    verify(!result && same(output, untouched));
                }
                actual = {};
                ok = checked(x, x, actual, arrays, types, inv, origin, lo,
                             modes, order, domain, crop);
                verify(bool(ok) && same(reference, actual));
                // A separate truncated layout for each component proves all
                // six supports are checked, rather than just the E-r array.
                for (int f = 0; f < 6; ++f) {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        Arrays shifted = arrays;
                        auto first = lbound(arrays[f]),
                             last = ubound(arrays[f]);
                        if (d == 0)
                            first.x = last.x;
                        if (d == 1)
                            first.y = last.y;
                        if (d == 2)
                            first.z = last.z;
                        Dim3 end{last.x + 1, last.y + 1, last.z + 1};
                        shifted[f] = Array4<Real const>(arrays[f].dataPtr(),
                                                        first, end, 3);
                        auto result = CheckMomentumGather(
                            x[0], x[1], x[2], shifted, types, inv, origin, lo,
                            modes, order);
                        verify(result.issue ==
                                   ImplicitGatherIssue::StencilOutOfBounds &&
                               result.component == f && result.direction == d);
                    }
                    // Exact r20 trigger: valid support, but the actual grid
                    // values are NaN. Detect it before the momentum push.
                    Arrays poisoned = arrays;
                    poisoned[f] = nan_arrays[f];
                    actual = {};
                    auto bad_field =
                        checked(x, x, actual, poisoned, types, inv, origin, lo,
                                modes, order, domain, crop);
                    verify(bad_field.issue ==
                           ImplicitGatherIssue::NonFiniteField);
                }
                Arrays missing = arrays;
                missing[5] = Array4<Real const>();
                verify(CheckMomentumGather(x[0], x[1], x[2], missing, types,
                                           inv, origin, lo, modes, order)
                           .issue == ImplicitGatherIssue::InvalidLayout);
                verify(CheckMomentumGather(x[0], x[1], x[2], arrays, types, inv,
                                           origin, lo, modes, 0)
                           .issue == ImplicitGatherIssue::InvalidLayout);
#ifdef WARPX_DIM_RZ
                verify(CheckMomentumGather(x[0], x[1], x[2], arrays, types, inv,
                                           origin, lo, 3, order)
                           .issue == ImplicitGatherIssue::InvalidLayout);
                // Axis and a negative Cartesian x have correct radial support.
                if (center[0] == 0) {
                    Point axis = x;
                    axis[0] = 0;
                    actual = {};
                    reference = {};
                    native(axis, reference, arrays, types, inv, origin, lo,
                           modes, order);
                    ok = checked(axis, axis, actual, arrays, types, inv, origin,
                                 lo, modes, order, domain, crop);
                    verify(bool(ok) && same(reference, actual));
                }
                Point rotated = x;
                rotated[0] = -x[0];
                actual = {};
                reference = {};
                native(rotated, reference, arrays, types, inv, origin, lo,
                       modes, order);
                ok = checked(rotated, rotated, actual, arrays, types, inv,
                             origin, lo, modes, order, domain, crop);
                verify(bool(ok) && same(reference, actual));
#endif
                // Existing absorbing-PEC crop yields a supported cap point even
                // when the original trial point is far beyond the local FAB.
                if (touches_cap) {
                    auto clipped = crop;
                    clipped[AMREX_SPACEDIM - 1][1] = true;
                    Point outside = x;
                    outside[2] =
                        PR(origin.z +
                           (domain[AMREX_SPACEDIM - 1][1] + 100) / inv.z);
                    actual = {};
                    ok = checked(x, outside, actual, arrays, types, inv, origin,
                                 lo, modes, order, domain, clipped);
                    verify(bool(ok));
                }
                // Finite grid inputs plus nonfinite accumulated external field
                // are rejected before the particle momentum pusher can use it.
                actual = {};
                actual[0] = std::numeric_limits<PR>::infinity();
                ok = checked(x, x, actual, arrays, types, inv, origin, lo,
                             modes, order, domain, crop);
                verify(ok.issue == ImplicitGatherIssue::NonFiniteField);
                // Check native support thresholds, including exactly zero
                // endpoint weights and rounded-integer changes of shape index.
                for (int n = 0; n < 24; ++n) {
                    Point probe = x;
                    probe[2] = PR(origin.z + (Real(n) / 4) / inv.z);
                    auto result = CheckMomentumGather(
                        probe[0], probe[1], probe[2], arrays, types, inv,
                        origin, lo, modes, order);
                    if (result) {
                        actual = {};
                        reference = {};
                        native(probe, reference, arrays, types, inv, origin, lo,
                               modes, order);
                        ok = checked(probe, probe, actual, arrays, types, inv,
                                     origin, lo, modes, order, domain, crop);
                        verify(bool(ok) && same(actual, reference));
                    }
                }
                Gpu::Atomic::Add(counts + 2, Long(1));
            });
        }
        auto const* host = tally.copyToHost();
        Long summary[3] = {host[0], host[1], host[2]};
        ParallelDescriptor::ReduceLongSum(summary, 3);
        for (int f = 0; f < 6; ++f)
            AMREX_ALWAYS_ASSERT(before[f] == fields[f]->norm1(0, 3));
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(summary[1] == 0 && summary[0] > 0,
                                         "gather safety fixture failed");
        Print() << "GATHER_SAFETY_PASS order=" << order << " layout=" << layout
                << " box=" << box << " assertions=" << summary[0]
                << " tiles=" << summary[2]
                << " ranks=" << ParallelDescriptor::NProcs() << "\n";
    }
    Finalize();
}
