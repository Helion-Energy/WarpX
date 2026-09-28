/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeEndpointCurrentResponse.H"
#include "NativeVacuumMassMatrixCapabilities.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Deposition/EsirkepovInstantaneousCurrentJacobian.H"
#include "Particles/Deposition/EsirkepovMassMatrices.H"
#include "Particles/Deposition/ImplicitBorisLocalResponse.H"
#include "Particles/Deposition/ImplicitBorisGatherFeedback.H"
#include "Particles/Deposition/FrozenImplicitEndpointResponse.H"
#include "Particles/Deposition/MassMatricesDeposition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "WarpX.H"

#include <ablastr/utils/Communication.H>
#include <AMReX_ConstexprFor.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace warpx::particles
{
namespace
{
using MF = amrex::MultiFab;
using Real = amrex::Real;
using P = amrex::ParticleReal;
using View = NativeEndpointCurrentResponse::View;
using warpx::fields::FieldType;
View Pointers (std::array<MF, 3> &f)
{
    return {&f[0], &f[1], &f[2]};
}
void Copy (MF &a, MF const &b)
{
    MF::Copy(a, b, 0, 0, 1, a.nGrowVect());
}
std::uint64_t Bytes (MF const &f)
{
    std::uint64_t n = 0;
    for (int i = 0; i < f.boxArray().size(); ++i)
    {
        n += amrex::grow(f.boxArray()[i], f.nGrowVect()).numPts();
    }
    return n * std::uint64_t(f.nComp()) * sizeof(Real);
}
void Allocate (MF &out, MF const &layout, int components)
{
    if (!out.isDefined() || out.boxArray() != layout.boxArray() ||
        out.DistributionMap() != layout.DistributionMap() ||
        out.nGrowVect() != layout.nGrowVect() || out.nComp() != components)
    {
        out = MF(layout.boxArray(), layout.DistributionMap(), components, layout.nGrowVect());
    }
}
void Finish (WarpX &sim, View const &current)
{
#if defined(WARPX_DIM_RZ)
    sim.ApplyInverseVolumeScalingToCurrentDensity(current[0], current[1], current[2], 0);
#endif
    auto const &period = sim.Geom(0).periodicity();
    for (auto *f : current)
    {
        ablastr::utils::communication::SumBoundary(*f, 0, 1, f->nGrowVect(), f->nGrowVect(),
                                                   WarpX::do_single_precision_comms, period);
    }
    sim.ApplyJfieldBoundary(0, current[0], current[1], current[2], PatchType::fine);
    for (auto *f : current)
    {
        f->OverrideSync(period);
        f->FillBoundary(period);
    }
}
} // namespace

bool NativeEndpointCurrentResponse::Freeze(WarpX &sim, Real dt, int interval_width,
                                           View const &gather_base, View const &old_current,
                                           View const &endpoint_current,
                                           View const &current_increment, bool local_boris_response)
{
    auto const* pmc_owner=NativePrivatePMCMassMatrixOwner(sim);
    m_private_pmc_collective_required=
        NativePrivatePMCMassMatrixSelectionOf(sim)!=NativePrivatePMCMassMatrixSelection::Off;
    m_private_pmc_owner_generation=NativePrivatePMCMassMatrixOwnerGeneration(sim);
    Invalidate();
    m_private_pmc_owner=pmc_owner;
    m_local_boris_response = local_boris_response;
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(sim, dt, interval_width, gather_base, old_current, endpoint_current,
                         current_increment, local_boris_response);
    amrex::Abort("Stored instantaneous endpoint MM response currently requires RZ");
    return false;
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        sim.maxLevel() == 0 && !EB::enabled() && !sim.getdo_moving_window() && WarpX::nox == 3 &&
            WarpX::ncomps == 1 && WarpX::n_rz_azimuthal_modes == 1 && !WarpX::use_filter &&
            WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov &&
            WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving &&
            (sim.Geom(0).isPeriodic(1) || (pmc_owner && local_boris_response)) &&
            dt > 0. && std::isfinite(dt),
        "Endpoint MM requires static single-level m0/order3 RZ Esirkepov/MC with periodic z "
        "or the private owner-bound PMC local-Boris capability");
    auto const layout = sim.m_fields.get_alldirs(FieldType::current_fp, 0);
    auto const magnetic = sim.m_fields.get_alldirs(FieldType::Bfield_aux, 0);
    FieldType const interval_types[3] = {FieldType::MassMatrices_X, FieldType::MassMatrices_Y,
                                         FieldType::MassMatrices_Z};
    if (local_boris_response)
    {
        AMREX_ALWAYS_ASSERT(interval_width > 0 && interval_width % 2 == 1);
        m_interval_width = interval_width;
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                auto const &raw =
                    *sim.m_fields.get(interval_types[a], ablastr::fields::Direction{b}, 0);
                AMREX_ALWAYS_ASSERT(raw.nComp() == interval_width * interval_width &&
                                    raw.boxArray() == layout[a]->boxArray() &&
                                    raw.nGrowVect() == layout[a]->nGrowVect());
            }
        }
    }
    auto const inverse = WarpX::InvCellSize(0);
    amrex::ReduceOps<amrex::ReduceOpMax> reach_op;
    amrex::ReduceData<int> reach_data(reach_op);
    using Reach = decltype(reach_data)::Type;
    for (auto const &species : sim.GetPartContainer())
    {
        if (species->getCharge() == 0. || species->do_not_deposit)
        {
            continue;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            dynamic_cast<PhysicalParticleContainer *>(species.get()) &&
                !dynamic_cast<SubcycledParticleContainer *>(species.get()) &&
                !species->DoFieldIonization(),
            "Endpoint MM requires pushed fixed-charge physical species without subcycling");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            species->getCharge() > 0. && std::isfinite(species->getCharge()) &&
                species->getMass() > 0. && std::isfinite(species->getMass()),
            "Endpoint MM requires positive finite charge/mass");
        dynamic_cast<PhysicalParticleContainer const *>(species.get())
            ->ValidateImplicitIonElectricWorkCapture();
        for (WarpXParIter pti(*species, 0); pti.isValid(); ++pti)
        {
            auto const position = GetParticlePosition<PIdx>(pti);
            auto const *xn = pti.GetAttribs("x_n").dataPtr();
            auto const *yn = pti.GetAttribs("y_n").dataPtr();
            auto const *zn = pti.GetAttribs("z_n").dataPtr();
            auto tile = pti.tilebox();
            tile.grow(sim.get_ng_depos_J());
            auto const origin = WarpX::LowerCorner(tile, 0, 0.);
            auto const lower = amrex::lbound(tile);
            amrex::GpuArray<amrex::Array4<Real const>, 3> rows{layout[0]->const_array(pti),
                                                               layout[1]->const_array(pti),
                                                               layout[2]->const_array(pti)};
            reach_op.eval(
                pti.numParticles(), reach_data,
                [=] AMREX_GPU_DEVICE(int p) -> Reach
                {
                    P x, y, z;
                    position(p, x, y, z);
                    P xe = 2. * x - xn[p], ye = 2. * y - yn[p];
                    P const radius = std::sqrt(xe * xe + ye * ye), angle = std::atan2(ye, xe);
                    xe = radius * std::cos(angle);
                    ye = radius * std::sin(angle);
                    P const endpoint_r = std::sqrt(xe * xe + ye * ye),
                            midpoint_r = std::hypot(x, y);
                    P const coordinates[4]{(endpoint_r - origin.x) * inverse.x,
                                           (2. * z - zn[p] - origin.z) * inverse.z,
                                           (midpoint_r - origin.x) * inverse.x,
                                           (z - origin.z) * inverse.z};
                    for (P coordinate : coordinates)
                    {
                        if (!std::isfinite(coordinate) || coordinate < 1. ||
                            coordinate > double(std::numeric_limits<int>::max() / 8))
                        {
                            return {std::numeric_limits<int>::max()};
                        }
                    }
                    // Same integer support convention as the order3 native point map.
                    int const ir = int(coordinates[0]) - 1, iz = int(coordinates[1]) - 1;
                    int const gr = int(coordinates[2]) - 1, gz = int(coordinates[3]) - 1;
                    for (int component = 0; component < 3; ++component)
                    {
                        int const ni = component == 0 ? 2 : 3;
                        int const nj = component == 2 ? 2 : 3;
                        if (!rows[component].contains(lower.x + ir, lower.y + iz, 0) ||
                            !rows[component].contains(lower.x + ir + ni, lower.y + iz + nj, 0))
                        {
                            return {std::numeric_limits<int>::max()};
                        }
                    }
                    return {amrex::max(std::abs(gr - ir), std::abs(gz - iz))};
                });
        }
    }
    int reach = std::max(0, amrex::get<0>(reach_data.value(reach_op)));
    amrex::ParallelDescriptor::ReduceIntMax(reach);
    if (reach == std::numeric_limits<int>::max())
    {
        return false;
    }
    int const needed_width = 2 * (3 + reach) + 1;
    if (std::int64_t(needed_width) * needed_width > std::numeric_limits<int>::max())
    {
        return false;
    }
    int const previous_width = m_width;
    m_width = std::max(m_width, needed_width);
    AMREX_ALWAYS_ASSERT(m_width > 0 && m_width % 2 == 1);
    m_matrix_bytes = 0;
    m_scratch_bytes = 0;
    for (int a = 0; a < 3; ++a)
    {
        AMREX_ALWAYS_ASSERT(gather_base[a]->ixType().nodeCentered());
        Allocate(m_gather_base[a], *gather_base[a], 1);
        Copy(m_gather_base[a], *gather_base[a]);
        Allocate(m_old[a], *old_current[a], 1);
        Copy(m_old[a], *old_current[a]);
        Allocate(m_endpoint[a], *endpoint_current[a], 1);
        Copy(m_endpoint[a], *endpoint_current[a]);
        Allocate(m_increment[a], *current_increment[a], 1);
        Copy(m_increment[a], *current_increment[a]);
        Allocate(m_direction[a], *layout[a], 1);
        if (local_boris_response)
        {
            AMREX_ALWAYS_ASSERT(magnetic[a]->ixType().nodeCentered());
            Allocate(m_magnetic_base[a], *magnetic[a], 1);
            Copy(m_magnetic_base[a], *magnetic[a]);
            m_scratch_bytes += Bytes(m_magnetic_base[a]);
        }
        m_scratch_bytes += Bytes(m_gather_base[a]) + Bytes(m_old[a]) + Bytes(m_endpoint[a]) +
                           Bytes(m_increment[a]) + Bytes(m_direction[a]);
        for (int b = 0; b < 3; ++b)
        {
            Allocate(m_matrix[3 * a + b], *layout[a], m_width * m_width);
            m_matrix[3 * a + b].setVal(0.);
            m_matrix_bytes += Bytes(m_matrix[3 * a + b]);
            if (local_boris_response)
            {
                Allocate(m_endpoint_magnetic[3 * a + b], *layout[a], m_width * m_width);
                Allocate(m_interval_magnetic[3 * a + b], *layout[a],
                         interval_width * interval_width);
                m_endpoint_magnetic[3 * a + b].setVal(0.);
                m_interval_magnetic[3 * a + b].setVal(0.);
                m_matrix_bytes +=
                    Bytes(m_endpoint_magnetic[3 * a + b]) + Bytes(m_interval_magnetic[3 * a + b]);
                // PreLinearSolve has already constructed independent PC fields.
                sim.m_fields.get(interval_types[a], ablastr::fields::Direction{b}, 0)->setVal(0.);
            }
        }
    }
    int const width = m_width, band_radius = (width - 1) / 2;
    int const zero = 0;
    amrex::Gpu::htod_memcpy(m_invalid.dataPtr(), &zero, sizeof(zero));
    int *bad = m_invalid.dataPtr();
    auto const plo = sim.Geom(0).ProbLoArray(), phi = sim.Geom(0).ProbHiArray();
    for (auto const &species : sim.GetPartContainer())
    {
        Real const charge = species->getCharge(), mass = species->getMass();
        if (charge == 0. || species->do_not_deposit)
        {
            continue;
        }
        for (WarpXParIter pti(*species, 0); pti.isValid(); ++pti)
        {
            auto const position = GetParticlePosition<PIdx>(pti);
            auto const *xn = pti.GetAttribs("x_n").dataPtr();
            auto const *yn = pti.GetAttribs("y_n").dataPtr();
            auto const *zn = pti.GetAttribs("z_n").dataPtr();
            auto const *un = pti.GetAttribs("ux_n").dataPtr();
            auto const *vn = pti.GetAttribs("uy_n").dataPtr();
            auto const *wn = pti.GetAttribs("uz_n").dataPtr();
            auto const *ux = pti.GetAttribs(PIdx::ux).dataPtr();
            auto const *uy = pti.GetAttribs(PIdx::uy).dataPtr();
            auto const *uz = pti.GetAttribs(PIdx::uz).dataPtr();
            auto const *weight = pti.GetAttribs(PIdx::w).dataPtr();
            int const *sub = species->HasiAttrib("nsuborbits")
                                 ? pti.GetiAttribs("nsuborbits").dataPtr()
                                 : nullptr;
            auto tile = pti.tilebox();
            tile.grow(sim.get_ng_depos_J());
            auto const origin = WarpX::LowerCorner(tile, 0, 0.);
            auto const lo = amrex::lbound(tile);
            amrex::GpuArray<amrex::Array4<Real>, 9> matrix, endpoint_b, interval_e, interval_b;
            for (int q = 0; q < 9; ++q)
            {
                matrix[q] = m_matrix[q].array(pti);
                if (local_boris_response)
                {
                    endpoint_b[q] = m_endpoint_magnetic[q].array(pti);
                    interval_b[q] = m_interval_magnetic[q].array(pti);
                    interval_e[q] =
                        sim.m_fields
                            .get(interval_types[q / 3], ablastr::fields::Direction{q % 3}, 0)
                            ->array(pti);
                }
            }
            amrex::GpuArray<amrex::Array4<Real const>, 3> E{gather_base[0]->const_array(pti),
                                                            gather_base[1]->const_array(pti),
                                                            gather_base[2]->const_array(pti)};
            amrex::GpuArray<amrex::Array4<Real const>, 3> B{magnetic[0]->const_array(pti),
                                                            magnetic[1]->const_array(pti),
                                                            magnetic[2]->const_array(pti)};
            amrex::GpuArray<amrex::IndexType, 3> types{magnetic[0]->ixType(), magnetic[1]->ixType(),
                                                       magnetic[2]->ixType()};
            amrex::For<128>(
                pti.numParticles(),
                [=] AMREX_GPU_DEVICE(long p)
                {
                    P x, y, z;
                    position(p, x, y, z);
                    P const r = std::hypot(x, y);
                    P xe = 2. * x - xn[p], ye = 2. * y - yn[p], ze = 2. * z - zn[p];
                    P const radius = std::sqrt(xe * xe + ye * ye), angle = std::atan2(ye, xe);
                    xe = radius * std::cos(angle);
                    ye = radius * std::sin(angle);
                    P const re = std::sqrt(xe * xe + ye * ye);
                    if (!(r > 0.) || !(re >= plo[0] && re < phi[0]) || !std::isfinite(ze) ||
                        !std::isfinite(weight[p]) || weight[p] < 0. || (sub && sub[p] > 1))
                    {
                        amrex::Gpu::Atomic::Max(bad, 1);
                        return;
                    }
                    if (!std::isfinite(un[p]) || !std::isfinite(vn[p]) || !std::isfinite(wn[p]) ||
                        !std::isfinite(ux[p]) || !std::isfinite(uy[p]) || !std::isfinite(uz[p]))
                    {
                        amrex::Gpu::Atomic::Max(bad, 1);
                        return;
                    }
                    int const gather_i = int((r - origin.x) * inverse.x) - 1 + lo.x;
                    int const gather_j = int((z - origin.z) * inverse.z) - 1 + lo.y;
                    for (int d = 0; d < 3; ++d)
                    {
                        if (!types[d].nodeCentered() || !B[d].contains(gather_i, gather_j, 0) ||
                            !B[d].contains(gather_i + 3, gather_j + 3, 0) ||
                            (local_boris_response &&
                             (!E[d].contains(gather_i, gather_j, 0) ||
                              !E[d].contains(gather_i + 3, gather_j + 3, 0))))
                        {
                            amrex::Gpu::Atomic::Max(bad, 1);
                            return;
                        }
                    }
                    P const cs = x / r, sn = y / r;
                    P bx = 0., by = 0., bz = 0.;
                    doDirectGatherVectorField<3, 3>(x, y, z, bx, by, bz, B[0], B[1], B[2], types[0],
                                                    types[1], types[2], inverse, origin, lo, 1);
                    P response[9];
                    P const gi = GetImplicitGammaInverse(un[p], vn[p], wn[p], ux[p], uy[p], uz[p]);
                    setMassMatricesKernels(charge, mass, dt, gi, cs, sn, ux[p], uy[p], uz[p], bx,
                                           by, bz, response[0], response[1], response[2],
                                           response[3], response[4], response[5], response[6],
                                           response[7], response[8]);
                    ImplicitBorisLocalResponse local_response;
                    if (local_boris_response)
                    {
                        P ex = 0., ey = 0., ez = 0.;
                        doDirectGatherVectorField<3, 3>(x, y, z, ex, ey, ez, E[0], E[1], E[2],
                                                        types[0], types[1], types[2], inverse,
                                                        origin, lo, 1);
                        if (!EvaluateImplicitBorisLocalResponse(
                                {un[p], vn[p], wn[p]}, {ux[p], uy[p], uz[p]}, {ex, ey, ez},
                                {bx, by, bz}, charge, mass, dt, PhysConst::inv_c2, local_response))
                        {
                            amrex::Gpu::Atomic::Max(bad, 1);
                            return;
                        }
                        ImplicitGatherGradient gradient{};
                        if (!NativeRZCubicGatherGradient({x,y,z},
                                {E[0],E[1],E[2],B[0],B[1],B[2]},
                                {ex,ey,ez,bx,by,bz},inverse,origin,lo,gradient) ||
                            !ApplyImplicitBorisGatherFeedback(dt,gradient,local_response))
                        {
                            amrex::Gpu::Atomic::Max(bad, 1);
                            return;
                        }
                    }
                    double gr[4], gz[4];
                    int const gr0 = Compute_shape_factor<3>{}(gr, (r - origin.x) * inverse.x);
                    int const gz0 = Compute_shape_factor<3>{}(gz, (z - origin.z) * inverse.z);
                    amrex::GpuArray<double, 3> const endpoint_x{xe, ye, ze};
                    amrex::GpuArray<double, 3> const endpoint_u{
                        2. * ux[p] - un[p], 2. * uy[p] - vn[p], 2. * uz[p] - wn[p]};
                    amrex::GpuArray<double, 3> const grid_x{(re - origin.x) * inverse.x, 0.,
                                                            (ze - origin.z) * inverse.z};
                    int const ir0 = int(grid_x[0]) - 1, iz0 = int(grid_x[2]) - 1;
                    amrex::GpuArray<double, 3> vel[6], dx[6], dv[6], dvbar_all[6];
                    double vt[6], dvt[6];
                    int const columns = local_boris_response ? 6 : 3;
                    double endpoint_norm = 0.;
                    for (int a = 0; a < 3; ++a)
                    {
                        endpoint_norm += endpoint_u[a] * endpoint_u[a];
                    }
                    double const end_gi = 1. / std::sqrt(1. + endpoint_norm * PhysConst::inv_c2);
                    for (int e = 0; e < columns; ++e)
                    {
                        amrex::GpuArray<double, 3> vcart{}, dcart{}, dvcart{};
                        if (local_boris_response)
                        {
                            int const component = e % 3, offset = e < 3 ? 0 : 3;
                            for (int a = 0; a < 3; ++a)
                            {
                                // Cylindrical nodal field components enter native Cartesian Boris.
                                double const l0 = component == 0 ? cs : component == 1 ? -sn : 0.;
                                double const l1 = component == 0 ? sn : component == 1 ? cs : 0.;
                                double const l2 = component == 2 ? 1. : 0.;
                                dvbar_all[e][a] =
                                    l0 * local_response.midpoint_velocity[offset][a] +
                                    l1 * local_response.midpoint_velocity[offset + 1][a] +
                                    l2 * local_response.midpoint_velocity[offset + 2][a];
                                dvcart[a] = l0 * local_response.endpoint_velocity[offset][a] +
                                            l1 * local_response.endpoint_velocity[offset + 1][a] +
                                            l2 * local_response.endpoint_velocity[offset + 2][a];
                                dcart[a] = dt * dvbar_all[e][a];
                                vcart[a] = end_gi * endpoint_u[a];
                            }
                        }
                        else
                        {
                            dvbar_all[e] = {cs * response[e] - sn * response[3 + e],
                                            sn * response[e] + cs * response[3 + e],
                                            response[6 + e]};
                            if (!FrozenImplicitEndpointResponse(endpoint_u, dvbar_all[e], gi, dt,
                                                                PhysConst::inv_c2, vcart, dcart,
                                                                dvcart))
                            {
                                amrex::Gpu::Atomic::Max(bad, 1);
                                return;
                            }
                        }
                        if (!CylindricalEndpointCurrentDirection(endpoint_x, vcart, dcart, dvcart,
                                                                 inverse.x, inverse.z, vel[e],
                                                                 dx[e], dv[e], vt[e], dvt[e]))
                        {
                            amrex::Gpu::Atomic::Max(bad, 1);
                            return;
                        }
                    }
                    Real const factors[3]{charge * weight[p] * inverse.y * inverse.z,
                                          charge * weight[p] * inverse.x * inverse.y * inverse.z,
                                          charge * weight[p] * inverse.x * inverse.y};
                    for (int j = iz0; j <= iz0 + 3; ++j)
                    {
                        for (int i = ir0; i <= ir0 + 3; ++i)
                        {
                            Real row[18]{};
                            for (int e = 0; e < columns; ++e)
                            {
                                amrex::GpuArray<double, 3> value{}, derivative{};
                                if (!EsirkepovInstantaneousCurrentJacobian<3, 2>(
                                        grid_x, vel[e], dx[e], dv[e], {i, 0, j}, vt[e], dvt[e],
                                        value, derivative))
                                {
                                    amrex::Gpu::Atomic::Max(bad, 1);
                                    return;
                                }
                                for (int a = 0; a < 3; ++a)
                                {
                                    row[3 * a + (e % 3) + 9 * (e / 3)] = factors[a] * derivative[a];
                                }
                            }
                            int const di0 = gr0 - i + band_radius, dj0 = gz0 - j + band_radius;
                            if (di0 < 0 || di0 + 3 >= width || dj0 < 0 || dj0 + 3 >= width)
                            {
                                amrex::Gpu::Atomic::Max(bad, 1);
                                return;
                            }
                            amrex::constexpr_for<0, 9>(
                                [&] (auto block)
                                {
                                    constexpr int q = block;
                                    if ((q < 3 && i == ir0 + 3) || (q >= 6 && j == iz0 + 3))
                                    {
                                        return;
                                    }
                                    if (!matrix[q].contains(lo.x + i, lo.y + j, 0))
                                    {
                                        amrex::Gpu::Atomic::Max(bad, 1);
                                        return;
                                    }
                                    for (int a = 0; a < 4; ++a)
                                    {
                                        for (int b = 0; b < 4; ++b)
                                        {
                                            int const comp = di0 + a + width * (dj0 + b);
                                            amrex::Gpu::Atomic::AddNoRet(
                                                &matrix[q](lo.x + i, lo.y + j, 0, comp),
                                                row[q] * gr[a] * gz[b]);
                                            if (local_boris_response)
                                            {
                                                amrex::Gpu::Atomic::AddNoRet(
                                                    &endpoint_b[q](lo.x + i, lo.y + j, 0, comp),
                                                    row[9 + q] * gr[a] * gz[b]);
                                            }
                                        }
                                    }
                                });
                        }
                    }
                    if (local_boris_response)
                    {
                        double const ro = std::hypot(xn[p], yn[p]);
                        // Interval deposition uses the virtual Cartesian endpoint before
                        // native Set/Get canonicalization, independently of endpoint I.
                        double const xi = 2. * x - xn[p], yi = 2. * y - yn[p];
                        double const ri = std::hypot(xi, yi), ra = .5 * (ro + ri);
                        esirkepov_mm::Shape<3> const sr0((ro - origin.x) * inverse.x),
                            sr1((ri - origin.x) * inverse.x), sz0((zn[p] - origin.z) * inverse.z),
                            sz1((ze - origin.z) * inverse.z);
                        esirkepov_mm::Derivative<3> const dr1((ri - origin.x) * inverse.x),
                            dz1((ze - origin.z) * inverse.z);
                        int const irlo = std::min(sr0.first, sr1.first),
                                  irhi = std::max(sr0.first, sr1.first) + 3;
                        int const izlo = std::min(sz0.first, sz1.first),
                                  izhi = std::max(sz0.first, sz1.first) + 3;
                        double const qvol = charge * weight[p] * inverse.x * inverse.y * inverse.z;
                        double const phi_velocity =
                            ra > 0. ? (x * uy[p] - y * ux[p]) * gi / ra : 0.;
                        double dre[6], dze[6], dphi[6];
                        for (int e = 0; e < 6; ++e)
                        {
                            auto const &v = dvbar_all[e];
                            dre[e] = ri > 0. ? dt * (xi * v[0] + yi * v[1]) / ri : 0.;
                            dze[e] = dt * v[2];
                            dphi[e] =
                                ra > 0.
                                    ? (xn[p] * v[1] - yn[p] * v[0] - .5 * phi_velocity * dre[e]) /
                                          ra
                                    : 0.;
                        }
                        int const interval_radius = (interval_width - 1) / 2;
                        double sum_r = 0., sum_dr = 0.;
                        for (int i = irlo; i <= irhi; ++i)
                        {
                            sum_r += sr0(i) - sr1(i);
                            sum_dr += dr1(i);
                            double sum_z = 0., sum_dz = 0.;
                            for (int j = izlo; j <= izhi; ++j)
                            {
                                sum_z += sz0(j) - sz1(j);
                                sum_dz += dz1(j);
                                double row[18]{};
                                double const wt = (sr1(i) * sz1(j) + sr0(i) * sz0(j)) / 3. +
                                                  (sr1(i) * sz0(j) + sr0(i) * sz1(j)) / 6.;
                                for (int e = 0; e < 6; ++e)
                                {
                                    int const q = e % 3 + 9 * (e / 3);
                                    if (i < irhi)
                                    {
                                        row[q] =
                                            qvol / (dt * inverse.x) *
                                            (-sum_dr * dre[e] * inverse.x * .5 * (sz0(j) + sz1(j)) +
                                             sum_r * .5 * dz1(j) * dze[e] * inverse.z);
                                    }
                                    row[q + 3] =
                                        qvol * (dphi[e] * wt +
                                                phi_velocity * (dr1(i) * dre[e] * inverse.x *
                                                                    (sz1(j) / 3. + sz0(j) / 6.) +
                                                                dz1(j) * dze[e] * inverse.z *
                                                                    (sr1(i) / 3. + sr0(i) / 6.)));
                                    if (j < izhi)
                                    {
                                        row[q + 6] =
                                            qvol / (dt * inverse.z) *
                                            (.5 * dr1(i) * dre[e] * inverse.x * sum_z -
                                             .5 * (sr0(i) + sr1(i)) * sum_dz * dze[e] * inverse.z);
                                    }
                                }
                                int const di = gr0 - i + interval_radius,
                                          dj = gz0 - j + interval_radius;
                                if (di < 0 || di + 3 >= interval_width || dj < 0 ||
                                    dj + 3 >= interval_width)
                                {
                                    amrex::Gpu::Atomic::Max(bad, 1);
                                    return;
                                }
                                amrex::constexpr_for<0, 9>(
                                    [&] (auto block)
                                    {
                                        constexpr int q = block;
                                        if ((q < 3 && i == irhi) || (q >= 6 && j == izhi))
                                        {
                                            return;
                                        }
                                        if (!interval_e[q].contains(lo.x + i, lo.y + j, 0) ||
                                            !interval_b[q].contains(lo.x + i, lo.y + j, 0))
                                        {
                                            amrex::Gpu::Atomic::Max(bad, 1);
                                            return;
                                        }
                                        for (int a = 0; a < 4; ++a)
                                        {
                                            for (int b = 0; b < 4; ++b)
                                            {
                                                int const component =
                                                    di + a + interval_width * (dj + b);
                                                double const weight_gather = gr[a] * gz[b];
                                                amrex::Gpu::Atomic::AddNoRet(
                                                    &interval_e[q](lo.x + i, lo.y + j, 0,
                                                                   component),
                                                    row[q] * weight_gather);
                                                amrex::Gpu::Atomic::AddNoRet(
                                                    &interval_b[q](lo.x + i, lo.y + j, 0,
                                                                   component),
                                                    row[q + 9] * weight_gather);
                                            }
                                        }
                                    });
                            }
                        }
                    }
                });
        }
    }
    int failed = m_invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed)
    {
        return false;
    }
    m_dt = dt;
    m_time = sim.gett_new(0);
    m_step = sim.getistep(0);
    m_valid = true;
    amrex::Print() << "Endpoint MM freeze epoch=" << m_epoch << " width=" << m_width
                   << " previous_width=" << previous_width << " interval_width=" << interval_width
                   << " matrix_bytes_global=" << m_matrix_bytes
                   << " scratch_bytes_global=" << m_scratch_bytes
                   << " local_boris_EB_response=" << m_local_boris_response
                   << " midpoint_gather_feedback=" << m_local_boris_response
                   << " frozen_gather_input_weights=1\n";
    return true;
#endif
}

bool NativeEndpointCurrentResponse::CollectiveContextRequired(WarpX const& sim) const noexcept
{
    return m_private_pmc_collective_required ||
        NativePrivatePMCMassMatrixSelectionOf(sim)!=NativePrivatePMCMassMatrixSelection::Off;
}

bool NativeEndpointCurrentResponse::ContextCurrentLocal(WarpX& sim,Real dt) const
{
    auto const* current=NativePrivatePMCMassMatrixOwnerLocal(sim);
    bool const selected=NativePrivatePMCMassMatrixSelectionOf(sim)==NativePrivatePMCMassMatrixSelection::On;
    return current==m_private_pmc_owner && (!selected || (current &&
        m_private_pmc_owner_generation==NativePrivatePMCMassMatrixOwnerGeneration(sim))) &&
        m_valid && (!current || m_local_boris_response) && dt==m_dt &&
        sim.gett_new(0)==m_time && sim.getistep(0)==m_step;
}

bool NativeEndpointCurrentResponse::Multiply(WarpX &sim, Real dt, View const &input,
                                             bool subtract_base, View const *magnetic_direction)
{
    bool ready=ContextCurrentLocal(sim,dt);
    bool const use_magnetic = m_local_boris_response && (subtract_base || magnetic_direction);
    View magnetic{};
    if (ready && use_magnetic)
    {
        if(magnetic_direction) magnetic=*magnetic_direction;
        else {
            for(int d=0;d<3;++d)
                ready=ready && sim.m_fields.has(FieldType::Bfield_aux,ablastr::fields::Direction{d},0);
            if(ready) magnetic=sim.m_fields.get_alldirs(FieldType::Bfield_aux,0);
        }
    }
    bool layout=ready;
    for (int d = 0; ready && d < 3; ++d) {
        layout=layout && input[d] && input[d]->boxArray()==m_gather_base[d].boxArray() &&
            input[d]->DistributionMap()==m_gather_base[d].DistributionMap() &&
            input[d]->nGrowVect()==m_gather_base[d].nGrowVect() &&
            (!use_magnetic || (magnetic[d] &&
             magnetic[d]->boxArray()==m_magnetic_base[d].boxArray() &&
             magnetic[d]->DistributionMap()==m_magnetic_base[d].DistributionMap() &&
             magnetic[d]->nGrowVect()==m_magnetic_base[d].nGrowVect()));
    }
    // No output or device scratch changes until the one uniform private AND.
    if(CollectiveContextRequired(sim))amrex::ParallelDescriptor::ReduceBoolAnd(layout);
    if(!layout)return false;
    int const width = m_width, radius = (width - 1) / 2, columns = use_magnetic ? 6 : 3;
    int const zero = 0;
    amrex::Gpu::htod_memcpy(m_invalid.dataPtr(), &zero, sizeof(zero));
    int *bad = m_invalid.dataPtr();
    for (int a = 0; a < 3; ++a)
    {
        m_direction[a].setVal(0.);
        for (amrex::MFIter mfi(m_direction[a]); mfi.isValid(); ++mfi)
        {
            auto const out = m_direction[a].array(mfi);
            amrex::GpuArray<amrex::Array4<Real const>, 6> matrix, in, ref;
            for (int b = 0; b < 3; ++b)
            {
                matrix[b] = m_matrix[3 * a + b].const_array(mfi);
                in[b] = input[b]->const_array(mfi);
                ref[b] = m_gather_base[b].const_array(mfi);
                if (use_magnetic)
                {
                    matrix[b + 3] = m_endpoint_magnetic[3 * a + b].const_array(mfi);
                    in[b + 3] = magnetic[b]->const_array(mfi);
                    ref[b + 3] = m_magnetic_base[b].const_array(mfi);
                }
            }
            amrex::For(mfi.fabbox(),
                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                       {
                           Real sum = 0.;
                           for (int b = 0; b < columns; ++b)
                           {
                               for (int v = 0; v < width; ++v)
                               {
                                   for (int u = 0; u < width; ++u)
                                   {
                                       Real const coefficient = matrix[b](i, j, k, u + width * v);
                                       if (coefficient == 0.)
                                       {
                                           continue;
                                       }
                                       int const ii = i + u - radius, jj = j + v - radius;
                                       if (!in[b].contains(ii, jj, k) ||
                                           !ref[b].contains(ii, jj, k))
                                       {
                                           amrex::Gpu::Atomic::Max(bad, 1);
                                           continue;
                                       }
                                       Real value = in[b](ii, jj, k);
                                       if (subtract_base)
                                       {
                                           value -= ref[b](ii, jj, k);
                                       }
                                       sum += coefficient * value;
                                   }
                               }
                           }
                           if (!std::isfinite(sum))
                           {
                               amrex::Gpu::Atomic::Max(bad, 1);
                           }
                           out(i, j, k) = sum;
                       });
        }
    }
    int failed = m_invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed)
    {
        return false;
    }
    Finish(sim, Pointers(m_direction));
    return true;
}

bool NativeEndpointCurrentResponse::ApplyIntervalMagnetic(WarpX &sim, Real dt,
                                                          View const &magnetic_trial,
                                                          View const &raw_interval_current)
{
    bool const ready=ContextCurrentLocal(sim,dt);
    bool layout=ready;
    for (int d = 0; ready && m_local_boris_response && d < 3; ++d) {
        layout=layout && magnetic_trial[d] && raw_interval_current[d] &&
            magnetic_trial[d]->boxArray()==m_magnetic_base[d].boxArray() &&
            magnetic_trial[d]->DistributionMap()==m_magnetic_base[d].DistributionMap() &&
            magnetic_trial[d]->nGrowVect()==m_magnetic_base[d].nGrowVect() &&
            raw_interval_current[d]->boxArray()==m_direction[d].boxArray() &&
            raw_interval_current[d]->DistributionMap()==m_direction[d].DistributionMap() &&
            raw_interval_current[d]->nGrowVect()==m_direction[d].nGrowVect();
    }
    if(CollectiveContextRequired(sim))amrex::ParallelDescriptor::ReduceBoolAnd(layout);
    if(!layout)return false;
    if(!m_local_boris_response)return true;
    int const width = m_interval_width, radius = (width - 1) / 2, zero = 0;
    amrex::Gpu::htod_memcpy(m_invalid.dataPtr(), &zero, sizeof(zero));
    int *bad = m_invalid.dataPtr();
    for (int a = 0; a < 3; ++a)
    {
        m_direction[a].setVal(0.);
        for (amrex::MFIter mfi(m_direction[a]); mfi.isValid(); ++mfi)
        {
            auto const out = m_direction[a].array(mfi);
            amrex::GpuArray<amrex::Array4<Real const>, 3> matrix, in, ref;
            for (int b = 0; b < 3; ++b)
            {
                matrix[b] = m_interval_magnetic[3 * a + b].const_array(mfi);
                in[b] = magnetic_trial[b]->const_array(mfi);
                ref[b] = m_magnetic_base[b].const_array(mfi);
            }
            amrex::For(mfi.fabbox(),
                       [=] AMREX_GPU_DEVICE(int i, int j, int k)
                       {
                           Real sum = 0.;
                           for (int b = 0; b < 3; ++b)
                           {
                               for (int v = 0; v < width; ++v)
                               {
                                   for (int u = 0; u < width; ++u)
                                   {
                                       Real const coefficient = matrix[b](i, j, k, u + width * v);
                                       if (coefficient == 0.)
                                       {
                                           continue;
                                       }
                                       int const ii = i + u - radius, jj = j + v - radius;
                                       if (!in[b].contains(ii, jj, k) ||
                                           !ref[b].contains(ii, jj, k))
                                       {
                                           amrex::Gpu::Atomic::Max(bad, 1);
                                           continue;
                                       }
                                       sum += coefficient * (in[b](ii, jj, k) - ref[b](ii, jj, k));
                                   }
                               }
                           }
                           if (!std::isfinite(sum))
                           {
                               amrex::Gpu::Atomic::Max(bad, 1);
                           }
                           out(i, j, k) = sum;
                       });
        }
    }
    int failed = m_invalid.dataValue();
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed)
    {
        return false;
    }
    for (int d = 0; d < 3; ++d)
    {
        MF::Add(*raw_interval_current[d], m_direction[d], 0, 0, 1,
                raw_interval_current[d]->nGrowVect());
    }
    return true;
}

bool NativeEndpointCurrentResponse::ApplyElectromagneticDirection(WarpX &sim, Real dt,
                                                                  View const &electric_direction,
                                                                  View const &magnetic_direction,
                                                                  View const &current_direction)
{
    if (!Multiply(sim, dt, electric_direction, false, &magnetic_direction))
    {
        return false;
    }
    for (int d = 0; d < 3; ++d)
    {
        Copy(*current_direction[d], m_direction[d]);
    }
    return true;
}

bool NativeEndpointCurrentResponse::Apply(WarpX &sim, Real dt, View const &gather_trial,
                                          View const &old_current, View const &endpoint_current,
                                          View const &current_increment)
{
    if (!Multiply(sim, dt, gather_trial, true))
    {
        return false;
    }
    for (int d = 0; d < 3; ++d)
    {
        Copy(*old_current[d], m_old[d]);
        Copy(*endpoint_current[d], m_endpoint[d]);
        Copy(*current_increment[d], m_increment[d]);
        MF::Add(*endpoint_current[d], m_direction[d], 0, 0, 1, endpoint_current[d]->nGrowVect());
        MF::Add(*current_increment[d], m_direction[d], 0, 0, 1, current_increment[d]->nGrowVect());
    }
    return true;
}

bool NativeEndpointCurrentResponse::ApplyDirection(WarpX &sim, Real dt, View const &direction,
                                                   View const &current_direction)
{
    if (!Multiply(sim, dt, direction, false))
    {
        return false;
    }
    for (int d = 0; d < 3; ++d)
    {
        Copy(*current_direction[d], m_direction[d]);
    }
    return true;
}
} // namespace warpx::particles
