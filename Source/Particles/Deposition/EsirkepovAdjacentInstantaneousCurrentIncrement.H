/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_ESIRKEPOV_ADJACENT_INSTANTANEOUS_CURRENT_INCREMENT_H_
#define WARPX_ESIRKEPOV_ADJACENT_INSTANTANEOUS_CURRENT_INCREMENT_H_

#include "EsirkepovInstantaneousCurrentIncrement.H"

namespace warpx::particles {

/** A cubic point-pair footprint, not an orbit/history or material certificate.
 * upper is inclusive for nodal shapes; a normal face ends at upper[d]-1.
 * Endpoint coordinates only classify support. They never replace the captured
 * position increment used in the polynomial. Default value is invalid.
 */
struct CubicAdjacentPointSupport {
    amrex::GpuArray<int,3> old_first{}, new_first{}, lower{}, upper{};
    unsigned int crossed_axes = 0;
    bool valid = false;
};

template<int Dim> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool PrepareCubicAdjacentPointSupport (
    EsirkepovInstantaneousPointIncrement const& point,
    CubicAdjacentPointSupport& output) noexcept
{
    static_assert(Dim == 2 || Dim == 3);
    output = {};
    CubicAdjacentPointSupport candidate{};
    for (int d=0; d<3; ++d) {
        if constexpr (Dim == 2) { if (d == 1) { continue; } }
        auto const& x = point.position[d];
        if (!instantaneous_increment_detail::FinitePair(x)) { return false; }
        int old_center=0, new_center=0, old_first=0, new_first=0;
        // Reuse the unmodified margin/range predicate separately at each end.
        // Calling it on the complete pair would intentionally reject crossing.
        if (!instantaneous_increment_detail::CommonPointSupport<3>(
                {x.value,0.},old_center,old_first) ||
            !instantaneous_increment_detail::CommonPointSupport<3>(
                {x.value+x.delta,0.},new_center,new_first)) { return false; }
        int const shift = new_center-old_center;
        if (shift < -1 || shift > 1) { return false; }
        candidate.old_first[d] = old_first;
        candidate.new_first[d] = new_first;
        candidate.lower[d] = old_first < new_first ? old_first : new_first;
        candidate.upper[d] = (old_first > new_first ? old_first : new_first)+3;
        if (shift != 0) { candidate.crossed_axes |= 1u << d; }
    }
    candidate.valid = true;
    output = candidate;
    return true;
}

namespace adjacent_instantaneous_detail {
/** Lift the existing adjacent-knot split into finite-increment arithmetic.
 * N=3 is the nodal factor; N=2 is its normal face factor shifted by -1/2.
 * Each bracket is evaluated as Shape(...,{anchor,displacement}).delta, so
 * no separately rounded O(1) endpoint shapes are subtracted. At the shared
 * knot the two ideal polynomial pieces agree. Represented formation/summation
 * remainders (including gradual underflow) are not claimed to vanish.
 */
template<int N> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
EsirkepovIncrement ShapePair (
    EsirkepovIncrement const& x, int old_first, int new_first, int node) noexcept
{
    static_assert(N == 2 || N == 3);
    int const old_center=old_first+1, new_center=new_first+1;
    double const offset = N == 2 ? -.5 : 0.;
    EsirkepovIncrement const local{x.value-double(old_center)+offset,x.delta};
    int const old_slot=node-old_first;
    if (old_first == new_first) {
        return esirkepov_increment_detail::Shape<N>(old_slot,local);
    }
    double const knot=double(old_center > new_center ? old_center : new_center);
    double const first_displacement=knot-x.value;
    double const second_displacement=x.delta-first_displacement;
    auto const first=esirkepov_increment_detail::Shape<N>(
        old_slot,{local.value,first_displacement});
    auto const second=esirkepov_increment_detail::Shape<N>(
        node-new_first,{knot-double(new_center)+offset,second_displacement});
    return {first.value,first.delta+second.delta};
}
} // namespace adjacent_instantaneous_detail

/** Explicit cubic adjacent-cell extension; the old common-cell API is intact.
 * Same-cell inputs call the exact old function, including its invalid-input
 * and tiny-increment semantics. Crossing inputs retain the native old point
 * current and finite nonlinear product increments over the complete union.
 * More than one cell per active direction and near-knot endpoints decline.
 * No caller/current/particle/history state is mutated. On failure current is
 * zero. Native charge/metric/sums/images and endpoint representation belong to
 * the depositor and its independent arithmetic/continuity qualification.
 */
template<int Dim> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool EsirkepovAdjacentInstantaneousCurrentPair (
    EsirkepovInstantaneousPointIncrement const& point,
    amrex::GpuArray<int,3> const& node,
    amrex::GpuArray<EsirkepovIncrement,3>& current) noexcept
{
    static_assert(Dim == 2 || Dim == 3);
    current = {};
    CubicAdjacentPointSupport support{};
    if (!PrepareCubicAdjacentPointSupport<Dim>(point,support)) { return false; }
    if (support.crossed_axes == 0) {
        return EsirkepovInstantaneousCurrentPair<3,Dim>(point,node,current);
    }
    amrex::GpuArray<double,3> x{},v{},zero{},base{},unused_rate{};
    amrex::GpuArray<int,3> safe_node{};
    EsirkepovIncrement nodal[3]{{1.,0.},{1.,0.},{1.,0.}},face[3]{};
    for (int d=0; d<3; ++d) {
        if constexpr (Dim == 2) { if (d == 1) { continue; } }
        if (!instantaneous_increment_detail::FinitePair(point.velocity[d])) {
            return false;
        }
        x[d]=point.position[d].value;
        v[d]=point.velocity[d].value;
        // A distant query is identically zero for both pieces. Clamp only its
        // integer index, never a physical coordinate or captured increment.
        int const query=node[d] < support.lower[d] ? support.lower[d]-1 :
            node[d] > support.upper[d] ? support.upper[d]+1 : node[d];
        nodal[d]=adjacent_instantaneous_detail::ShapePair<3>(point.position[d],
            support.old_first[d],support.new_first[d],query);
        face[d]=adjacent_instantaneous_detail::ShapePair<2>(point.position[d],
            support.old_first[d],support.new_first[d],query);
        int const first=support.old_first[d];
        safe_node[d]=query < first ? first-1 : query > first+3 ? first+4 : query;
    }
    double transverse=0.;
    if constexpr (Dim == 2) {
        if (!instantaneous_increment_detail::FinitePair(point.azimuthal_velocity)) {
            return false;
        }
        transverse=point.azimuthal_velocity.value;
    }
    if (!EsirkepovCurrentRate<3,Dim>(x,v,zero,safe_node,transverse,0.,base,unused_rate)) {
        return false;
    }
    amrex::GpuArray<EsirkepovIncrement,3> candidate{};
    for (int d=0; d<3; ++d) {
        if constexpr (Dim == 2) {
            if (d == 1) {
                candidate[d]=point.azimuthal_velocity*nodal[0]*nodal[2];
                continue;
            }
        }
        int const e=(d+1)%3,f=(d+2)%3;
        candidate[d]=face[d]*point.velocity[d]*nodal[e]*nodal[f];
    }
    for (int d=0; d<3; ++d) {
        candidate[d].value=base[d];
        if (!std::isfinite(candidate[d].value) || !std::isfinite(candidate[d].delta) ||
            !std::isfinite(candidate[d].value+candidate[d].delta)) { return false; }
    }
    current=candidate;
    return true;
}
} // namespace warpx::particles
#endif
