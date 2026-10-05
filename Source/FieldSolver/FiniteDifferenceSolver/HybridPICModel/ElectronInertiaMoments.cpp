/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ElectronInertiaMoments.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/WarpXConst.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_MFIter.H>
#include <cmath>

using namespace amrex::literals;

namespace
{
    amrex::GpuArray<int, 3> staggering (amrex::MultiFab const& field)
    {
        amrex::GpuArray<int, 3> result{1, 1, 1};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            result[d] = field.ixType().nodeCentered(d) ? 1 : 0;
        }
        return result;
    }
}

void ElectronInertiaMoments::SetCurrentSlope (
    ablastr::fields::VectorField const& old_current,
    ablastr::fields::VectorField const& new_current, amrex::Real const dt)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(dt) && dt > 0.0_rt,
        "Electron inertia current slope requires a positive finite particle timestep");
    for (int c = 0; c < 3; ++c) {
        auto& slope = m_current_slope[c];
        auto const& newer = *new_current[c];
        auto const& older = *old_current[c];
        auto const ng = amrex::min(newer.nGrowVect(), older.nGrowVect());
        if (!slope.isDefined() || slope.boxArray() != newer.boxArray() ||
            slope.DistributionMap() != newer.DistributionMap() || slope.nGrowVect() != ng)
        {
            slope.define(newer.boxArray(), newer.DistributionMap(), 1, ng);
        }
        // Subtract before dividing: identical deposits must give an exact zero.
        for (amrex::MFIter mfi(slope); mfi.isValid(); ++mfi) {
            auto const out = slope.array(mfi);
            auto const next = newer.const_array(mfi), prev = older.const_array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                out(i,j,k) = (next(i,j,k)-prev(i,j,k))/dt;
            });
        }
    }
}

void ElectronInertiaMoments::AddToRHS (
    ablastr::fields::VectorField const& Efield,
    ablastr::fields::VectorField const& J_plasma,
    ablastr::fields::VectorField const& J_ion,
    amrex::MultiFab const& rho_i, amrex::MultiFab const* pedestal,
    amrex::Real const rho_floor, amrex::Geometry const& geom,
    FiniteDifferenceSolver& solver,
    std::array<amrex::iMultiFab const*, 3> const& eb_mask)
{
    ABLASTR_PROFILE("ElectronInertiaMoments::AddToRHS");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rho_i.ixType().nodeCentered() && rho_floor > 0.0_rt,
        "Electron inertia moment terms require nodal ion density and a positive floor");
    // Standalone initial E evaluations have no deposited time interval yet.
    // The first particle advance prepares a measured slope on fresh starts
    // and restarts alike, before either magnetic half-step consumes it.
    if (!m_current_slope[0].isDefined()) { SetCurrentSlope(J_ion, J_ion, 1.0_rt); }
    if (!m_rhs.isDefined() || m_rhs.boxArray() != rho_i.boxArray() ||
        m_rhs.DistributionMap() != rho_i.DistributionMap())
    {
        m_rhs.define(rho_i.boxArray(), rho_i.DistributionMap(), 3, 0);
        m_div_ion_current.define(rho_i.boxArray(), rho_i.DistributionMap(), 1, 0);
    }
    // Native edge-to-node divergence, including the RZ axis. J_i is fixed
    // within each magnetic half, so this value is also fixed. Rebuilding
    // avoids stale caches when a moment buffer is overwritten in place.
    solver.ComputeDivE(J_ion, m_div_ion_current);
    amrex::GpuArray<int, 3> const nodal{1, 1, 1}, ratio{1, 1, 1};
    amrex::GpuArray<amrex::GpuArray<int, 3>, 3> jp_stag{}, ji_stag{};
    for (int c = 0; c < 3; ++c) {
        jp_stag[c] = staggering(*J_plasma[c]);
        ji_stag[c] = staggering(*J_ion[c]);
        AMREX_ALWAYS_ASSERT(m_current_slope[c].boxArray() == J_ion[c]->boxArray());
        AMREX_ALWAYS_ASSERT(m_current_slope[c].DistributionMap() == J_ion[c]->DistributionMap());
    }
    bool const has_pedestal = pedestal != nullptr;
#if defined(WARPX_DIM_RZ)
    bool const has_axis = geom.ProbLo(0) == 0.0_rt;
    int const axis = geom.Domain().smallEnd(0);
#else
    amrex::ignore_unused(geom);
#endif
    constexpr amrex::Real minus_me_over_e = -PhysConst::m_e / PhysConst::q_e;
    for (amrex::MFIter mfi(m_rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const rhs = m_rhs.array(mfi);
        auto const den = rho_i.const_array(mfi);
        auto const div = m_div_ion_current.const_array(mfi);
        auto const ped = has_pedestal ? pedestal->const_array(mfi)
                                      : amrex::Array4<amrex::Real const>{};
        amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> jp{}, ji{}, slope{};
        for (int c = 0; c < 3; ++c) {
            jp[c] = J_plasma[c]->const_array(mfi);
            ji[c] = J_ion[c]->const_array(mfi);
            slope[c] = m_current_slope[c].const_array(mfi);
        }
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real const raw = den(i,j,k,0) + (has_pedestal ? ped(i,j,k) : 0.0_rt);
            amrex::Real const charge = amrex::max(rho_floor, raw);
            amrex::Real const divergence = raw > rho_floor ? div(i,j,k) : 0.0_rt;
            for (int c = 0; c < 3; ++c) {
                amrex::Real const ue = (
                    ablastr::coarsen::sample::Interp(ji[c], ji_stag[c], nodal, ratio, i,j,k,0)
                    - ablastr::coarsen::sample::Interp(jp[c], jp_stag[c], nodal, ratio, i,j,k,0)
                    ) / charge;
                amrex::Real const rate = ablastr::coarsen::sample::Interp(
                    slope[c], ji_stag[c], nodal, ratio, i,j,k,0);
                rhs(i,j,k,c) = minus_me_over_e * (rate + ue*divergence) / charge;
#if defined(WARPX_DIM_RZ)
                if (has_axis && i == axis && c < 2) { rhs(i,j,k,c) = 0.0_rt; }
#endif
            }
        });
    }
    for (int c = 0; c < 3; ++c) {
        auto const dest = staggering(*Efield[c]);
        bool const masked = eb_mask[c] != nullptr;
        for (amrex::MFIter mfi(*Efield[c], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const E = Efield[c]->array(mfi);
            auto const rhs = m_rhs.const_array(mfi);
            auto const mask = masked ? eb_mask[c]->const_array(mfi) : amrex::Array4<int const>{};
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (!masked || mask(i,j,k) != 0) {
                    E(i,j,k) += ablastr::coarsen::sample::Interp(rhs, nodal, dest, ratio, i,j,k,c);
                }
            });
        }
    }
}
