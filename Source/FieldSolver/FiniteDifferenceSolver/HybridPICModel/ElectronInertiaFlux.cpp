/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ElectronInertiaMoments.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "QdsmcVolumeElement.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_MFIter.H>

using namespace amrex::literals;

namespace
{
    using Stagger = amrex::GpuArray<int, 3>;

    Stagger staggering (amrex::MultiFab const& field)
    {
        Stagger result{1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[d] = field.ixType().nodeCentered(d) ? 1 : 0;
        }
        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int component (int const d)
    {
#if defined(WARPX_DIM_3D)
        return d;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        return d == 0 ? 0 : 2;
#else
        amrex::ignore_unused(d);
        return 2;
#endif
    }

    // FillBoundary supplies inter-box and periodic ghosts. Continue physical
    // faces quadratically, using only interior/previously exchanged values.
    // At the RZ axis use vector parity, including the half-cell radial index.
    // No kernel reads a ghost written by another invocation of the kernel.
    void fill_physical_ghosts (amrex::MultiFab& field, amrex::Geometry const& geom,
                               int const c)
    {
        auto const domain = amrex::convert(geom.Domain(), field.ixType());
        auto const lo = amrex::lbound(domain), hi = amrex::ubound(domain);
        amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { periodic[d] = geom.isPeriodic(d); }
#if defined(WARPX_DIM_RZ)
        bool const axis = geom.ProbLo(0) == 0.0_rt;
        bool const radial_nodal = field.ixType().nodeCentered(0);
#else
        amrex::ignore_unused(c);
#endif
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            auto const a = field.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                int const pos[3] = {i,j,k};
                int const lower[3] = {lo.x,lo.y,lo.z}, upper[3] = {hi.x,hi.y,hi.z};
                int index[3][3] = {{i,i,i},{j,j,j},{k,k,k}};
                amrex::Real weight[3][3] = {{1,0,0},{1,0,0},{1,0,0}};
                bool outside = false;
                amrex::Real parity = 1.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (periodic[d] || (pos[d] >= lower[d] && pos[d] <= upper[d])) {
                        continue;
                    }
                    outside = true;
#if defined(WARPX_DIM_RZ)
                    if (axis && d == 0 && pos[d] < lower[d]) {
                        index[d][0] = 2*lower[d]-pos[d]-(radial_nodal ? 0 : 1);
                        if (c < 2) { parity = -parity; }
                        continue;
                    }
#endif
                    bool const low = pos[d] < lower[d];
                    int const origin = low ? lower[d] : upper[d];
                    int const step = low ? 1 : -1;
                    amrex::Real const t = step*(pos[d]-origin);
                    for (int q = 0; q < 3; ++q) { index[d][q] = origin+step*q; }
                    weight[d][0] = 0.5_rt*(t-1.0_rt)*(t-2.0_rt);
                    weight[d][1] = t*(2.0_rt-t);
                    weight[d][2] = 0.5_rt*t*(t-1.0_rt);
                }
                if (!outside) { return; }
                amrex::Real value = 0.0_rt;
                for (int x = 0; x < 3; ++x) {
                    for (int y = 0; y < 3; ++y) {
                        for (int z = 0; z < 3; ++z) {
                            auto const w = weight[0][x]*weight[1][y]*weight[2][z];
                            if (w != 0.0_rt) {
                                value += w*a(index[0][x],index[1][y],index[2][z]);
                            }
                        }
                    }
                }
                a(i,j,k) = parity*value;
            });
        }
    }
}

void ElectronInertiaMoments::AddMomentumFluxToRHS (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& J_plasma,
    ablastr::fields::VectorField const& J_ion,
    amrex::MultiFab const& rho_i, amrex::MultiFab const* pedestal,
    amrex::Real const rho_floor, amrex::Geometry const& geom,
    FiniteDifferenceSolver& solver,
    std::array<amrex::iMultiFab const*, 3> const& eb_mask)
{
    ABLASTR_PROFILE("ElectronInertiaMoments::AddMomentumFluxToRHS");
    AMREX_ALWAYS_ASSERT(rho_i.ixType().nodeCentered() && rho_floor > 0.0_rt);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(d) >= 3,
            "Electron momentum flux requires at least three cells per dimension");
    }
    if (!m_current_slope[0].isDefined()) { SetCurrentSlope(J_ion,J_ion,1.0_rt); }
    if (!m_div_ion_current.isDefined() || m_div_ion_current.boxArray() != rho_i.boxArray() ||
        m_div_ion_current.DistributionMap() != rho_i.DistributionMap())
    {
        m_div_ion_current.define(rho_i.boxArray(),rho_i.DistributionMap(),1,0);
    }
#if defined(WARPX_DIM_RZ)
    if (geom.ProbLo(0) == 0.0_rt) {
        // Identical endpoints leave only the native deposited-current divergence.
        // This uses the Verboncoeur axis volume when that correction is enabled;
        // Ampere/Faraday keep their separate Yee field divergence.
        WarpX::GetInstance().ComputeRZContinuityResidual(
            m_div_ion_current,rho_i,rho_i,J_ion,1.0_rt);
    } else
#endif
    {
        solver.ComputeDivE(J_ion,m_div_ion_current);
    }
    Stagger const nodal{1,1,1}, ratio{1,1,1};
    amrex::GpuArray<Stagger,3> stag{};
    bool const has_pedestal = pedestal != nullptr;
    auto const dx = geom.CellSizeArray(), plo = geom.ProbLoArray();
    auto const small = geom.Domain().smallEnd();
    amrex::GpuArray<int,AMREX_SPACEDIM> pmc_lo{},pmc_hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        pmc_lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
        pmc_hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
    }
#if defined(WARPX_DIM_RZ)
    bool const has_axis = geom.ProbLo(0) == 0.0_rt;
    int const axis = geom.Domain().smallEnd(0);
#endif
    for (int c = 0; c < 3; ++c) {
        AMREX_ALWAYS_ASSERT(J_ion[c]->ixType() == Efield[c]->ixType());
        AMREX_ALWAYS_ASSERT(J_plasma[c]->ixType() == Efield[c]->ixType());
        auto& current = m_flux_current[c];
        auto& velocity = m_flux_velocity[c];
        if (!current.isDefined() || current.boxArray() != Efield[c]->boxArray() ||
            current.DistributionMap() != Efield[c]->DistributionMap())
        {
            current.define(Efield[c]->boxArray(),Efield[c]->DistributionMap(),1,2);
            velocity.define(Efield[c]->boxArray(),Efield[c]->DistributionMap(),1,2);
        }
        stag[c] = staggering(*Efield[c]);
        auto const st = stag[c];
        bool const masked = eb_mask[c] != nullptr;
        for (amrex::MFIter mfi(current); mfi.isValid(); ++mfi) {
            auto const q = current.array(mfi), u = velocity.array(mfi);
            auto const ji = J_ion[c]->const_array(mfi);
            auto const jp = J_plasma[c]->const_array(mfi);
            auto const rho = rho_i.const_array(mfi);
            auto const ped = has_pedestal ? pedestal->const_array(mfi)
                : amrex::Array4<amrex::Real const>{};
            auto const mask = masked ? eb_mask[c]->const_array(mfi) : amrex::Array4<int const>{};
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                amrex::Real const raw = ablastr::coarsen::sample::Interp(
                    rho,nodal,st,ratio,i,j,k,0) + (has_pedestal ?
                    ablastr::coarsen::sample::Interp(ped,nodal,st,ratio,i,j,k,0) : 0.0_rt);
                q(i,j,k) = masked && mask(i,j,k) == 0 ? 0.0_rt : ji(i,j,k)-jp(i,j,k);
#if defined(WARPX_DIM_RZ)
                if (has_axis && c == 1 && i == axis) { q(i,j,k) = 0.0_rt; }
#endif
                u(i,j,k) = q(i,j,k)/amrex::max(rho_floor,raw);
            });
        }
        current.FillBoundary(geom.periodicity());
        velocity.FillBoundary(geom.periodicity());
        fill_physical_ghosts(current,geom,c);
        fill_physical_ghosts(velocity,geom,c);
    }
    constexpr amrex::Real minus_me_over_e = -PhysConst::m_e/PhysConst::q_e;
    for (int c = 0; c < 3; ++c) {
        auto const st = stag[c];
        auto const momentum_volume = MakeQdsmcVolumeElement(geom,Efield[c]->ixType());
#if defined(WARPX_DIM_RZ)
        auto const theta_volume = MakeQdsmcVolumeElement(geom,rho_i.ixType());
#endif
        bool const masked = eb_mask[c] != nullptr;
        for (amrex::MFIter mfi(*Efield[c],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const E = Efield[c]->array(mfi);
            auto const rate = m_current_slope[c].const_array(mfi);
            auto const rho = rho_i.const_array(mfi);
            auto const divji = m_div_ion_current.const_array(mfi);
            auto const ped = has_pedestal ? pedestal->const_array(mfi)
                : amrex::Array4<amrex::Real const>{};
            auto const mask = masked ? eb_mask[c]->const_array(mfi) : amrex::Array4<int const>{};
            amrex::GpuArray<amrex::Array4<amrex::Real const>,3> q{},u{},ji{};
            for (int d = 0; d < 3; ++d) {
                q[d] = m_flux_current[d].const_array(mfi);
                u[d] = m_flux_velocity[d].const_array(mfi);
                ji[d] = J_ion[d]->const_array(mfi);
            }
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                if (masked && mask(i,j,k) == 0) { return; }
#if defined(WARPX_DIM_RZ)
                if (has_axis && c == 1 && i == axis) { return; }
                amrex::Real const r = plo[0]+(i-small[0]+(st[0] ? 0.0_rt : 0.5_rt))*dx[0];
#else
                amrex::ignore_unused(plo,small);
#endif
                amrex::Real const raw = ablastr::coarsen::sample::Interp(
                    rho,nodal,st,ratio,i,j,k,0) + (has_pedestal ?
                    ablastr::coarsen::sample::Interp(ped,nodal,st,ratio,i,j,k,0) : 0.0_rt);
                amrex::Real const charge = amrex::max(rho_floor,raw);
                amrex::Real const density_rate = raw > rho_floor ?
                    -ablastr::coarsen::sample::Interp(divji,nodal,st,ratio,i,j,k,0) : 0.0_rt;
                amrex::Real div_momentum = 0.0_rt, div_mass = 0.0_rt;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int low[3] = {i,j,k}, high[3] = {i,j,k};
                    auto face_stag = st;
                    face_stag[d] = 1-st[d];
                    if (st[d]) { --low[d]; } else { ++high[d]; }
                    int const dc = component(d);
                    auto ql = ablastr::coarsen::sample::Interp(
                        q[dc],stag[dc],face_stag,ratio,low[0],low[1],low[2],0);
                    auto qr = ablastr::coarsen::sample::Interp(
                        q[dc],stag[dc],face_stag,ratio,high[0],high[1],high[2],0);
                    auto ul = ablastr::coarsen::sample::Interp(
                        u[c],st,face_stag,ratio,low[0],low[1],low[2],0);
                    auto ur = ablastr::coarsen::sample::Interp(
                        u[c],st,face_stag,ratio,high[0],high[1],high[2],0);
                    int const pos[3] = {i,j,k};
                    auto const lower = momentum_volume.lower(d,pos[d]);
                    auto const upper = momentum_volume.upper(d,pos[d]);
                    bool const low_wall = !momentum_volume.periodic[d] &&
                        lower == momentum_volume.lo[d];
                    bool const high_wall = !momentum_volume.periodic[d] &&
                        upper == momentum_volume.hi[d];
                    // Clip nodal dual cells at physical walls. At a PMC,
                    // curl(B).normal is zero, so the normal electron charge
                    // flux is precisely the deposited ion boundary flux.
                    // This preserves absorbing/outflow ions as well as closed
                    // flow; no electron reflection is inferred from particle BCs.
                    auto wall_stag = st;
                    wall_stag[d] = 1;
                    if (low_wall) {
                        int wall[3] = {i,j,k};
                        wall[d] = small[d];
                        if (pmc_lo[d]) {
                            ql = ablastr::coarsen::sample::Interp(ji[dc],stag[dc],
                                wall_stag,ratio,wall[0],wall[1],wall[2],0);
                        } else if (st[d]) { ql = 0.5_rt*(ql+qr); }
                        if (st[d]) { ul = u[c](i,j,k); }
                    }
                    if (high_wall) {
                        int wall[3] = {i,j,k};
                        wall[d] += st[d] ? 0 : 1;
                        if (pmc_hi[d]) {
                            qr = ablastr::coarsen::sample::Interp(ji[dc],stag[dc],
                                wall_stag,ratio,wall[0],wall[1],wall[2],0);
                        } else if (st[d]) { qr = 0.5_rt*(ql+qr); }
                        if (st[d]) { ur = u[c](i,j,k); }
                    }
#if defined(WARPX_DIM_RZ)
                    if (d == 0) {
                        auto const measure = momentum_volume.radial_measure(i);
                        div_mass += (upper*qr-lower*ql)/measure;
                        div_momentum += (upper*qr*ur-lower*ql*ul)/measure;
                    } else
#endif
                    {
                        div_mass += (qr-ql)/(upper-lower);
                        div_momentum += (qr*ur-ql*ul)/(upper-lower);
                    }
                }
#if defined(WARPX_DIM_RZ)
                if (c == 0) {
                    auto const rl = r-0.5_rt*dx[0], rr = r+0.5_rt*dx[0];
                    auto const left = rl > 0.0_rt ?
                        theta_volume(i,j,k)*u[1](i,j,k)*u[1](i,j,k)/rl : 0.0_rt;
                    auto const right = theta_volume(i+1,j,k)*u[1](i+1,j,k)*u[1](i+1,j,k)/rr;
                    div_momentum -= 0.5_rt*charge*(left+right)/momentum_volume(i,j,k);
                } else if (c == 1) {
                    // Adjoint radial/azimuthal metric forces: their work
                    // cancels using the same dual volumes as bulk energy.
                    div_momentum += u[1](i,j,k)*0.5_rt*(q[0](i-1,j,k)+q[0](i,j,k))/r;
                }
#endif
                amrex::Real const ion_divergence = ablastr::coarsen::sample::Interp(
                    divji,nodal,st,ratio,i,j,k,0);
                amrex::Real const floor_source = density_rate+ion_divergence;
                amrex::Real const metric_residual = div_mass-ion_divergence;
                // Central momentum flux has work sum(u div F)=sum(u^2 div Q)/2.
                // Half of the staggered metric residual gives its compatible
                // kinetic-energy split. This residual vanishes in the continuum
                // (and for unclipped Cartesian Yee data). The physical floor
                // source retains its full coefficient, as required by D_t u.
                // No extra thermal energy is deposited. partial_t(J_plasma)
                // belongs only to curl-curl recovery.
                E(i,j,k) += minus_me_over_e/charge * (rate(i,j,k)+div_momentum-
                    u[c](i,j,k)*(floor_source+0.5_rt*metric_residual));
            });
        }
    }
}
