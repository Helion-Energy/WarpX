/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ElectronInertiaConvection.H"

#include "Utils/WarpXConst.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>

using namespace amrex::literals;

namespace
{
    amrex::GpuArray<int, 3> staggering (amrex::MultiFab const& field)
    {
        // Collapsed array dimensions must remain nodal: Interp must never
        // read a second point in a dimension that does not exist.
        amrex::GpuArray<int, 3> result{1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[d] = field.ixType().nodeCentered(d) ? 1 : 0;
        }
        return result;
    }
}

void ElectronInertiaConvection::AddToRHS (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& J_plasma,
    ablastr::fields::VectorField const& J_ion,
    amrex::MultiFab const& rho, amrex::MultiFab const* pedestal,
    amrex::Real const rho_floor, amrex::Geometry const& geom,
    std::array<amrex::iMultiFab const*, 3> const& eb_mask)
{
    ABLASTR_PROFILE("ElectronInertiaConvection::AddToRHS");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rho.ixType().nodeCentered(),
        "Electron inertia convection requires nodal charge density");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rho_floor > 0.0_rt,
        "Electron inertia convection requires a positive density floor");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().length(d) >= 2,
            "Electron inertia convection requires at least two cells per dimension");
    }
    if (!m_velocity.isDefined() || m_velocity.boxArray() != rho.boxArray() ||
        m_velocity.DistributionMap() != rho.DistributionMap())
    {
        m_velocity.define(rho.boxArray(), rho.DistributionMap(), 4, 2);
        m_acceleration.define(rho.boxArray(), rho.DistributionMap(), 3, 0);
    }
    auto const dx = geom.CellSizeArray();
    auto const lo = amrex::lbound(amrex::surroundingNodes(geom.Domain()));
    auto const hi = amrex::ubound(amrex::surroundingNodes(geom.Domain()));
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { periodic[d] = geom.isPeriodic(d); }
    amrex::GpuArray<int, 3> const nodal{1, 1, 1}, ratio{1, 1, 1};
    amrex::GpuArray<amrex::GpuArray<int, 3>, 3> jp_stag{}, ji_stag{};
    for (int c = 0; c < 3; ++c) {
        jp_stag[c] = staggering(*J_plasma[c]);
        ji_stag[c] = staggering(*J_ion[c]);
    }
    bool const has_pedestal = pedestal != nullptr;
#if defined(WARPX_DIM_RZ)
    amrex::Real const rmin = geom.ProbLo(0);
#endif
    // The entire valid nodal field is rewritten, then exchanged, every
    // stage. No cached QDSMC velocity or preceding-stage ghosts are read.
    for (amrex::MFIter mfi(m_velocity, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_velocity.array(mfi);
        auto const den = rho.const_array(mfi);
        auto const ped = has_pedestal ? pedestal->const_array(mfi)
                                      : amrex::Array4<amrex::Real const>{};
        amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> jp{}, ji{};
        for (int c = 0; c < 3; ++c) {
            jp[c] = J_plasma[c]->const_array(mfi);
            ji[c] = J_ion[c]->const_array(mfi);
        }
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real const charge = amrex::max(rho_floor,
                den(i, j, k, 0) + (has_pedestal ? ped(i, j, k) : 0.0_rt));
            for (int c = 0; c < 3; ++c) {
                u(i, j, k, c) =
                    (ablastr::coarsen::sample::Interp(ji[c], ji_stag[c], nodal, ratio,
                                                      i, j, k, 0) -
                     ablastr::coarsen::sample::Interp(jp[c], jp_stag[c], nodal, ratio,
                                                      i, j, k, 0)) / charge;
            }
#if defined(WARPX_DIM_RZ)
            if (rmin == 0.0_rt && i == lo.x) {
                u(i, j, k, 0) = 0.0_rt;
                u(i, j, k, 1) = 0.0_rt;
            }
#endif
            u(i,j,k,3) = 0.5_rt*(u(i,j,k,0)*u(i,j,k,0) + u(i,j,k,1)*u(i,j,k,1)
                                + u(i,j,k,2)*u(i,j,k,2));
        });
    }
    m_velocity.FillBoundary(geom.periodicity());

    for (amrex::MFIter mfi(m_acceleration, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const a = m_acceleration.array(mfi);
        auto const u = m_velocity.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int const pos[3] = {i, j, k};
            int const lower[3] = {lo.x, lo.y, lo.z};
            int const upper[3] = {hi.x, hi.y, hi.z};
            amrex::Real du[3][3] = {};
            for (int c = 0; c < 3; ++c) {
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int p[3] = {i, j, k}, m[3] = {i, j, k};
                    amrex::Real derivative;
                    if (!periodic[d] && pos[d] == lower[d]) {
                        p[d] += 1; m[d] += 2;
                        derivative = (-3.0_rt*u(i,j,k,c) + 4.0_rt*u(p[0],p[1],p[2],c)
                                      - u(m[0],m[1],m[2],c)) / (2.0_rt*dx[d]);
                    } else if (!periodic[d] && pos[d] == upper[d]) {
                        p[d] -= 1; m[d] -= 2;
                        derivative = (3.0_rt*u(i,j,k,c) - 4.0_rt*u(p[0],p[1],p[2],c)
                                      + u(m[0],m[1],m[2],c)) / (2.0_rt*dx[d]);
                    } else {
                        p[d] += 1; m[d] -= 1;
                        derivative = (u(p[0],p[1],p[2],c) - u(m[0],m[1],m[2],c))
                                     / (2.0_rt*dx[d]);
                    }
#if defined(WARPX_DIM_3D)
                    int const component = d;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    int const component = d == 0 ? 0 : 2;
#else
                    int const component = 2;
#endif
                    du[c][component] = derivative;
                }
            }
            amrex::Real omega[3] = {du[2][1]-du[1][2], du[0][2]-du[2][0],
                                     du[1][0]-du[0][1]};
#if defined(WARPX_DIM_RZ)
            amrex::Real const r = rmin + (i - lo.x)*dx[0];
            omega[2] += r > 0.0_rt ? u(i,j,k,1)/r : du[1][0];
#endif
            // (u.grad)u = grad(|u|^2/2) - u cross curl(u). The scalar
            // gradient is added directly on Yee edges below, so its
            // discrete curl is exactly zero away from physical BCs.
            a(i,j,k,0) = u(i,j,k,2)*omega[1] - u(i,j,k,1)*omega[2];
            a(i,j,k,1) = u(i,j,k,0)*omega[2] - u(i,j,k,2)*omega[0];
            a(i,j,k,2) = u(i,j,k,1)*omega[0] - u(i,j,k,0)*omega[1];
        });
    }
    constexpr amrex::Real minus_me_over_e = -PhysConst::m_e / PhysConst::q_e;
    for (int c = 0; c < 3; ++c) {
        auto const dest = staggering(*Efield[c]);
        bool const masked = eb_mask[c] != nullptr;
#if defined(WARPX_DIM_3D)
        int const gradient_dir = c;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        int const gradient_dir = c == 0 ? 0 : c == 2 ? 1 : -1;
#else
        int const gradient_dir = c == 2 ? 0 : -1;
#endif
        for (amrex::MFIter mfi(*Efield[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const E = Efield[c]->array(mfi);
            auto const a = m_acceleration.const_array(mfi);
            auto const u = m_velocity.const_array(mfi);
            auto const mask = masked ? eb_mask[c]->const_array(mfi) : amrex::Array4<int const>{};
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (!masked || mask(i,j,k) != 0) {
                    amrex::Real gradient = 0.0_rt;
                    if (gradient_dir >= 0) {
                        int p[3] = {i,j,k};
                        p[gradient_dir] += 1;
                        gradient = (u(p[0],p[1],p[2],3)-u(i,j,k,3)) / dx[gradient_dir];
                    }
                    E(i,j,k) += minus_me_over_e * (gradient +
                        ablastr::coarsen::sample::Interp(a, nodal, dest, ratio, i,j,k,c));
                }
            });
        }
    }
}
