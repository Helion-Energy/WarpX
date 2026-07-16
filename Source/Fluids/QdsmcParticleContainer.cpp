/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "QdsmcParticleContainer.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/particles/NodalFieldGather.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <AMReX.H>
#include <AMReX_Algorithm.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_Box.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBCellFlag.H>
#   include <AMReX_EBFabFactory.H>
#endif
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleContainer.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_ParticleUtil.H>
#include <AMReX_Scan.H>
#include <AMReX_Utility.H>

#include <cmath>
#include <cstdint>

using namespace amrex::literals;

// The QDSMC grid fields (K_e, the deposited weights and the nodal v_e) are
// stored with NODAL staggering; every gather and scatter below uses the
// matching order-1 (linear) nodal weights, so a marker at rest reproduces
// its cell values exactly.


QdsmcParticleContainer::QdsmcParticleContainer (amrex::AmrCore* amr_core)
    : amrex::ParticleContainerPureSoA<QdsmcPIdx::nattribs, 0>(amr_core->GetParGDB())
{
    SetParticleSize();
}


void QdsmcParticleContainer::InitParticles (int lev)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::InitParticles()");

    reserveData();
    resizeData();

    amrex::Geometry const & geom = Geom(lev);
    auto const dx_arr = geom.CellSizeArray();
    auto const plo    = geom.ProbLoArray();

    // Define particle tiles for every (grid, tile) pair on this level.
    for (auto mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
        DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
    }

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    amrex::MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel if (not WarpX::serialize_initial_conditions)
#endif
    for (amrex::MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        amrex::Box const & tile_box = mfi.tilebox();
        int const grid_id = mfi.index();
        int const tile_id = mfi.LocalTileIndex();

        // One particle per cell, except cells fully covered by an embedded
        // boundary: no electron fluid lives there, so no entropy marker is
        // created (deposits into covered cells are inert -- the recovery is
        // gated on the density floor -- and the EB clamp in PushX keeps
        // markers out of the body). Use exclusive scan to assign per-cell
        // offsets so the per-cell writes are race-free in parallel.
        amrex::Gpu::DeviceVector<amrex::Long> counts(tile_box.numPts(), 1);
#ifdef AMREX_USE_EB
        bool eb_on = false;
        amrex::Array4<amrex::EBCellFlag const> eb_flag_arr;
        if (EB::enabled()) {
            eb_on = true;
            auto const& eb_fact = WarpX::GetInstance().fieldEBFactory(lev);
            eb_flag_arr = eb_fact.getMultiEBCellFlagFab().const_array(mfi);
            auto * const pcounts = counts.data();
            auto const flag = eb_flag_arr;
            amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                if (flag(i, j, k).isCovered()) {
                    pcounts[tile_box.index(iv)] = 0;
                }
            });
        }
#endif
        amrex::Gpu::DeviceVector<amrex::Long> offset(tile_box.numPts());
        amrex::Long const max_new_particles = amrex::Scan::ExclusiveSum(
            counts.size(), counts.data(), offset.data());

        // Reserve a globally-unique ID range for the new particles.
        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (qdsmc_init_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid + max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < amrex::LongParticleIds::LastParticleID,
            "QdsmcParticleContainer::InitParticles: overflow on particle id numbers");

        int const cpuid = amrex::ParallelDescriptor::MyProc();

        auto & particle_tile =
            GetParticles(lev)[std::make_pair(grid_id, tile_id)];

        if ((NumRuntimeRealComps() > 0) || (NumRuntimeIntComps() > 0)) {
            DefineAndReturnParticleTile(lev, grid_id, tile_id);
        }

        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        auto const new_size = old_size + max_new_particles;
        particle_tile.resize(new_size);

        auto & soa = particle_tile.GetStructOfArrays();

        amrex::GpuArray<amrex::ParticleReal*, QdsmcPIdx::nattribs> pa;
        for (int ia = 0; ia < QdsmcPIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        std::uint64_t * AMREX_RESTRICT pa_idcpu =
            soa.GetIdCPUData().data() + old_size;

        auto * const poffset = offset.data();

#ifdef AMREX_USE_EB
        auto const eb_flag = eb_flag_arr;
        bool const skip_covered = eb_on;
#endif
        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::ignore_unused(j, k);  // unused below AMREX_SPACEDIM
            amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
#ifdef AMREX_USE_EB
            if (skip_covered && eb_flag(i, j, k).isCovered()) { return; }
#endif
            long const ip = poffset[tile_box.index(iv)];

            pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid + ip, cpuid);

            // Compute the cell-center position in physical units. The field
            // dimension determines which axis indices are physically meaningful;
            // missing axes are set to 0 on the particle's home record.
#if defined(WARPX_DIM_3D)
            amrex::Real const x_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            amrex::Real const y_pos = plo[1] + (iv[1] + amrex::Real(0.5)) * dx_arr[1];
            amrex::Real const z_pos = plo[2] + (iv[2] + amrex::Real(0.5)) * dx_arr[2];
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::y][ip] = y_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            // In 2D Cartesian and RZ the second in-plane coord is z; the y
            // axis is the unused out-of-plane direction.
            amrex::Real const x_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            auto const y_pos = amrex::Real(0);
            amrex::Real const z_pos = plo[1] + (iv[1] + amrex::Real(0.5)) * dx_arr[1];
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_1D_Z)
            auto const x_pos = amrex::Real(0);
            auto const y_pos = amrex::Real(0);
            amrex::Real const z_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            pa[QdsmcPIdx::z][ip] = z_pos;
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: 1D radial; the single
            // AMReX-tracked position slot is x (= r). QDSMC is not validated
            // in these geometries (no radial volume weighting yet) and
            // HybridPICModel::ReadParameters refuses to enable the energy
            // equation there -- this branch only needs to compile and be sane.
            amrex::Real const x_pos = plo[0] + (iv[0] + amrex::Real(0.5)) * dx_arr[0];
            auto const y_pos = amrex::Real(0);
            auto const z_pos = amrex::Real(0);
            pa[QdsmcPIdx::x][ip] = x_pos;
#endif

            // Home position is always stored as a 3D vector.
            pa[QdsmcPIdx::x_node][ip] = x_pos;
            pa[QdsmcPIdx::y_node][ip] = y_pos;
            pa[QdsmcPIdx::z_node][ip] = z_pos;

            // Velocity, entropy and weight are populated each step by SetV/SetK.
            pa[QdsmcPIdx::vx][ip] = amrex::Real(0);
            pa[QdsmcPIdx::vy][ip] = amrex::Real(0);
            pa[QdsmcPIdx::vz][ip] = amrex::Real(0);
            pa[QdsmcPIdx::entropy][ip] = amrex::Real(0);
            pa[QdsmcPIdx::np_real][ip] = amrex::Real(0);
        });

        amrex::Gpu::synchronize();

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add(&(*cost)[mfi.index()], wt);
        }
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::SetV (int lev,
                              const amrex::MultiFab & Ux,
                              const amrex::MultiFab & Uy,
                              const amrex::MultiFab & Uz)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::SetV()");

    auto & warpx = WarpX::GetInstance();
    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();

        auto const ux_arr = Ux.const_array(pti);
        auto const uy_arr = Uy.const_array(pti);
        auto const uz_arr = Uz.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Linear gather of the nodal field at the marker's home position.
            auto const v = ablastr::particles::doGatherVectorFieldNodal(
                x_node[ip], y_node[ip], z_node[ip],
                ux_arr, uy_arr, uz_arr, dxi, plo);

            vx[ip] = v[0];
            vy[ip] = v[1];
            vz[ip] = v[2];
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::GatherVAtCurrentPosition (int lev,
                                                  const amrex::MultiFab & Ux,
                                                  const amrex::MultiFab & Uy,
                                                  const amrex::MultiFab & Uz)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::GatherVAtCurrentPosition()");

    auto & warpx = WarpX::GetInstance();
    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        // Current position: the AMReX-tracked slots written by PushX. Axes
        // without a tracked slot use the (zero) home value, consistent with
        // the home-position convention in SetV.
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal const* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#else
        amrex::ParticleReal const* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x_node].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal const* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#else
        amrex::ParticleReal const* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y_node].dataPtr();
#endif
#if !defined(WARPX_DIM_RCYLINDER) && !defined(WARPX_DIM_RSPHERE)
        amrex::ParticleReal const* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();
#else
        amrex::ParticleReal const* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z_node].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();

        auto const ux_arr = Ux.const_array(pti);
        auto const uy_arr = Uy.const_array(pti);
        auto const uz_arr = Uz.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Linear gather of the nodal field at the marker's current position.
            auto const v = ablastr::particles::doGatherVectorFieldNodal(
                pa_x[ip], pa_y[ip], pa_z[ip],
                ux_arr, uy_arr, uz_arr, dxi, plo);

            vx[ip] = v[0];
            vy[ip] = v[1];
            vz[ip] = v[2];
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::SetK (int lev,
                              const amrex::MultiFab & Kfield,
                              const amrex::MultiFab & rhofield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::SetK()");

    auto & warpx = WarpX::GetInstance();
    auto const plo = warpx.Geom(lev).ProbLoArray();
    auto const dxi = warpx.Geom(lev).InvCellSizeArray();
    auto const * dx_arr = warpx.Geom(lev).CellSize();

    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT entropy =
            attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

        auto const K_arr   = Kfield.const_array(pti);
        auto const rho_arr = rhofield.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Linear gathers of the nodal charge density and entropy at the
            // marker's home position; the marker then carries the electron
            // count N of its cell and the matching entropy content K*N.
            amrex::Real const n_p = ablastr::particles::doGatherScalarFieldNodal(
                x_node[ip], y_node[ip], z_node[ip], rho_arr, dxi, plo)
                * cell_volume / PhysConst::q_e;
            amrex::Real const k_p = ablastr::particles::doGatherScalarFieldNodal(
                x_node[ip], y_node[ip], z_node[ip], K_arr, dxi, plo);

            np_real[ip] = n_p;
            entropy[ip] = k_p * n_p;
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::PushX (int lev, amrex::Real dt, amrex::Real frac)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::PushX()");

    amrex::Geometry const & geom = Geom(lev);
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx_arr = geom.CellSizeArray();

    // Effective displacement time: frac*dt of the full-step characteristic.
    amrex::Real const fdt = frac * dt;

    // Per-dimension domain clamp bounds. In non-periodic directions the
    // advected position is clamped just inside the domain (positions at or
    // beyond ProbHi count as outside) rather than handed to Redistribute,
    // which would DELETE the marker: since InitParticles runs only once, the
    // home cell would then have no QDSMC marker for the rest of the run and
    // its T_e could never be updated again. Clamping instead accumulates the
    // carried entropy at the boundary nodes and preserves the
    // one-marker-per-cell invariant (ResetParticles returns it home).
    // Periodic directions are left unclamped so Redistribute wraps them.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_bnd;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_bnd;
    amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bnd[d] = plo[d];
        hi_bnd[d] = phi[d] - 1.e-6_rt * dx_arr[d];
        is_periodic[d] = geom.isPeriodic(d);
    }

    // Embedded-boundary clamp: markers whose advected position lands inside
    // an EB body stay at home for this push. The entropy they carry then
    // remains at the (uncovered) home node -- the same conservative
    // treatment the domain-wall clamp gives markers pushed out of the
    // domain. The signed distance function is negative inside the body.
    auto const dxi = geom.InvCellSizeArray();
    amrex::MultiFab const* eb_phi = nullptr;
#ifdef AMREX_USE_EB
    if (EB::enabled()) {
        auto & warpx = WarpX::GetInstance();
        eb_phi = warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, lev);
    }
#endif

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        amrex::Array4<amrex::Real const> eb_phi_arr;
        if (eb_phi != nullptr) { eb_phi_arr = eb_phi->const_array(pti); }
        bool const have_eb = (eb_phi != nullptr);

        // Home and velocity components are only needed for the axes with an
        // AMReX-tracked position slot (x everywhere but 1D_Z, y only in 3D).
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

        // Position attributes (only the AMReX-tracked subset). For
        // dimensions that are not represented in the field (y in 2D,
        // x and y in 1D Z), the position attribute does not exist as
        // an enum value, so the corresponding update is omitted.
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Skip particles with no weight (e.g. just-reset particles
            // before a SetK call). They contribute nothing to the deposit.
            if (np_real[ip] <= amrex::Real(0)) { return; }

            // Push of one coordinate along the characteristic from home,
            // clamped just inside the domain in non-periodic directions (see
            // the bound setup above). Markers may cross multiple cells;
            // Redistribute() below reassigns tiles.
            auto const push_clamp = [&] (amrex::Real x0, amrex::Real v, int d)
            {
                amrex::Real const xnew = x0 + v * fdt;
                return is_periodic[d] ? xnew
                                      : amrex::Clamp(xnew, lo_bnd[d], hi_bnd[d]);
            };

            // Write the new position to the AMReX-tracked position slots.
            // Axes not represented in the field have no enum slot and are
            // simply not tracked (consistent with field dimensionality).
#if defined(WARPX_DIM_3D)
            amrex::Real const xn = push_clamp(x_node[ip], vx[ip], 0);
            amrex::Real const yn = push_clamp(y_node[ip], vy[ip], 1);
            amrex::Real const zn = push_clamp(z_node[ip], vz[ip], 2);
            if (have_eb) {
                amrex::Real const d = ablastr::particles::doGatherScalarFieldNodal(
                    xn, yn, zn, eb_phi_arr, dxi, plo);
                if (d < amrex::Real(0)) { return; }  // stay at home this push
            }
            pa_x[ip] = xn;
            pa_y[ip] = yn;
            pa_z[ip] = zn;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::Real const xn = push_clamp(x_node[ip], vx[ip], 0);
            amrex::Real const zn = push_clamp(z_node[ip], vz[ip], 1);
            if (have_eb) {
                amrex::Real const d = ablastr::particles::doGatherScalarFieldNodal(
                    xn, amrex::Real(0), zn, eb_phi_arr, dxi, plo);
                if (d < amrex::Real(0)) { return; }  // stay at home this push
            }
            pa_x[ip] = xn;
            pa_z[ip] = zn;
#elif defined(WARPX_DIM_1D_Z)
            amrex::ignore_unused(have_eb, eb_phi_arr, dxi);
            pa_z[ip] = push_clamp(z_node[ip], vz[ip], 0);
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: x (= r) is the single
            // tracked position (QDSMC is refused at runtime in these
            // geometries); z is a plain attribute, advanced unclamped.
            amrex::ignore_unused(have_eb, eb_phi_arr, dxi);
            pa_x[ip] = push_clamp(x_node[ip], vx[ip], 0);
            pa_z[ip] = z_node[ip] + vz[ip] * dt;
#endif
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::ResetParticles (int lev)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::ResetParticles()");

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        // The home components are only needed where a matching AMReX-tracked
        // position slot exists (x everywhere but 1D_Z, y only in 3D).
#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT x_node =
            attribs[QdsmcPIdx::x_node].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT y_node =
            attribs[QdsmcPIdx::y_node].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT z_node =
            attribs[QdsmcPIdx::z_node].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT entropy =
            attribs[QdsmcPIdx::entropy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT np_real =
            attribs[QdsmcPIdx::np_real].dataPtr();

#if !defined(WARPX_DIM_1D_Z)
        amrex::ParticleReal* const AMREX_RESTRICT pa_x =
            attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
        amrex::ParticleReal* const AMREX_RESTRICT pa_y =
            attribs[QdsmcPIdx::y].dataPtr();
#endif
        amrex::ParticleReal* const AMREX_RESTRICT pa_z =
            attribs[QdsmcPIdx::z].dataPtr();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
#if !defined(WARPX_DIM_1D_Z)
            pa_x[ip] = x_node[ip];
#endif
#if defined(WARPX_DIM_3D)
            pa_y[ip] = y_node[ip];
#endif
            pa_z[ip] = z_node[ip];

            vx[ip] = 0;
            vy[ip] = 0;
            vz[ip] = 0;
            entropy[ip] = 0;
            np_real[ip] = 0;
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::SpawnConductionNodes (int lev, amrex::Real const h,
                                              const amrex::MultiFab & Ufield,
                                              const amrex::MultiFab & Nfield,
                                              const amrex::MultiFab & Dfield,
                                              const amrex::MultiFab & gradDx,
                                              const amrex::MultiFab & gradDy,
                                              const amrex::MultiFab & gradDz,
                                              bool const apply_kicks)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::SpawnConductionNodes()");

    // Conduction nodes live for exactly one pass.
    clearParticles();
    reserveData();
    resizeData();

    amrex::Geometry const & geom = Geom(lev);
    auto const dx_arr = geom.CellSizeArray();
    auto const dxi    = geom.InvCellSizeArray();
    auto const plo    = geom.ProbLoArray();
    auto const phi    = geom.ProbHiArray();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_bnd;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_bnd;
    amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bnd[d] = plo[d];
        hi_bnd[d] = phi[d] - 1.e-6_rt * dx_arr[d];
        is_periodic[d] = geom.isPeriodic(d);
    }

    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }

    // Active kick axes and node count: the Gauss-Hermite tensor product runs
    // over every physical direction of the energy transport, independent of
    // the field dimensionality (RZ kicks off-plane and folds into r).
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
    constexpr int n_ax = 3;
#elif defined(WARPX_DIM_XZ)
    constexpr int n_ax = 2;
#else
    constexpr int n_ax = 1;
#endif
    constexpr int n_nodes = (n_ax == 3) ? 27 : ((n_ax == 2) ? 9 : 3);

    // 3-node Gauss-Hermite rule.
    const amrex::GpuArray<amrex::Real, 3> gh_xi
        {0.0_rt, 1.7320508075688772_rt, -1.7320508075688772_rt};
    const amrex::GpuArray<amrex::Real, 3> gh_w
        {2.0_rt/3.0_rt, 1.0_rt/6.0_rt, 1.0_rt/6.0_rt};

    amrex::MultiFab const* eb_phi = nullptr;
#ifdef AMREX_USE_EB
    if (EB::enabled()) {
        auto & warpx = WarpX::GetInstance();
        eb_phi = warpx.m_fields.get(warpx::fields::FieldType::distance_to_eb, lev);
    }
#endif

    // Define every (grid, tile) key with the SAME tiled iterator the
    // parallel loop below uses: the tile map must not be mutated
    // concurrently from OMP threads.
    amrex::MFItInfo info;
    if (do_tiling && amrex::Gpu::notInLaunchRegion()) {
        info.EnableTiling(tile_size);
    }
    for (amrex::MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi) {
        DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
    }

#ifdef AMREX_USE_OMP
    info.SetDynamic(true);
#pragma omp parallel
#endif
    for (amrex::MFIter mfi = MakeMFIter(lev, info); mfi.isValid(); ++mfi)
    {
        amrex::Box const & tile_box = mfi.tilebox();
        int const grid_id = mfi.index();
        int const tile_id = mfi.LocalTileIndex();

        // Per-cell node counts (0 in covered cells), scanned into offsets.
        amrex::Gpu::DeviceVector<amrex::Long> counts(tile_box.numPts(), n_nodes);
#ifdef AMREX_USE_EB
        if (EB::enabled()) {
            auto const& eb_fact = WarpX::GetInstance().fieldEBFactory(lev);
            auto const flag = eb_fact.getMultiEBCellFlagFab().const_array(mfi);
            auto * const pcounts = counts.data();
            amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
                if (flag(i, j, k).isCovered()) {
                    pcounts[tile_box.index(iv)] = 0;
                }
            });
        }
#endif
        amrex::Gpu::DeviceVector<amrex::Long> offset(tile_box.numPts());
        amrex::Long const max_new_particles = amrex::Scan::ExclusiveSum(
            counts.size(), counts.data(), offset.data());

        amrex::Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (qdsmc_cond_nextid)
#endif
        {
            pid = ParticleType::NextID();
            ParticleType::NextID(pid + max_new_particles);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pid + max_new_particles < amrex::LongParticleIds::LastParticleID,
            "QdsmcParticleContainer::SpawnConductionNodes: particle id overflow");

        int const cpuid = amrex::ParallelDescriptor::MyProc();

        auto & particle_tile =
            GetParticles(lev)[std::make_pair(grid_id, tile_id)];
        auto const old_size = static_cast<amrex::Long>(particle_tile.size());
        particle_tile.resize(old_size + max_new_particles);

        auto & soa = particle_tile.GetStructOfArrays();
        amrex::GpuArray<amrex::ParticleReal*, QdsmcPIdx::nattribs> pa;
        for (int ia = 0; ia < QdsmcPIdx::nattribs; ++ia) {
            pa[ia] = soa.GetRealData(ia).data() + old_size;
        }
        std::uint64_t * AMREX_RESTRICT pa_idcpu =
            soa.GetIdCPUData().data() + old_size;

        auto * const poffset = offset.data();
        auto const * pcounts = counts.data();

        auto const u_arr   = Ufield.const_array(mfi);
        auto const nn_arr  = Nfield.const_array(mfi);
        auto const d_arr   = Dfield.const_array(mfi);
        auto const gdx_arr = gradDx.const_array(mfi);
        auto const gdy_arr = gradDy.const_array(mfi);
        auto const gdz_arr = gradDz.const_array(mfi);

        amrex::Array4<amrex::Real const> eb_phi_arr;
        if (eb_phi != nullptr) { eb_phi_arr = eb_phi->const_array(mfi); }
        bool const have_eb = (eb_phi != nullptr);

        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::ignore_unused(j, k);
            amrex::IntVect const iv(AMREX_D_DECL(i, j, k));
            if (pcounts[tile_box.index(iv)] == 0) { return; }
            long const ip0 = poffset[tile_box.index(iv)];

            // Cell-center home position (3D record, missing axes zero).
#if defined(WARPX_DIM_3D)
            amrex::Real const xh = plo[0] + (iv[0] + 0.5_rt) * dx_arr[0];
            amrex::Real const yh = plo[1] + (iv[1] + 0.5_rt) * dx_arr[1];
            amrex::Real const zh = plo[2] + (iv[2] + 0.5_rt) * dx_arr[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::Real const xh = plo[0] + (iv[0] + 0.5_rt) * dx_arr[0];
            auto const yh = amrex::Real(0);
            amrex::Real const zh = plo[1] + (iv[1] + 0.5_rt) * dx_arr[1];
#else
            auto const xh = amrex::Real(0);
            auto const yh = amrex::Real(0);
            amrex::Real const zh = plo[0] + (iv[0] + 0.5_rt) * dx_arr[0];
#endif

            // Home-state gathers (nodal linear, exact cell-center average).
            amrex::Real const u_home = ablastr::particles::doGatherScalarFieldNodal(
                xh, yh, zh, u_arr, dxi, plo);
            amrex::Real const n_home = ablastr::particles::doGatherScalarFieldNodal(
                xh, yh, zh, nn_arr, dxi, plo);
            amrex::Real const d_home = amrex::max(0.0_rt,
                ablastr::particles::doGatherScalarFieldNodal(
                    xh, yh, zh, d_arr, dxi, plo));
            auto const gd = ablastr::particles::doGatherVectorFieldNodal(
                xh, yh, zh, gdx_arr, gdy_arr, gdz_arr, dxi, plo);

            amrex::Real const sig =
                apply_kicks ? std::sqrt(2.0_rt * d_home * h) : 0.0_rt;
            amrex::Real const hdrift = apply_kicks ? h : 0.0_rt;

            // Energy content of the home cell carried by this cell's nodes.
            // The deposition kernel normalizes by the true (cylindrical in
            // RZ) cell volume, so the content must be the true shell energy.
#if defined(WARPX_DIM_RZ)
            amrex::Real const vol = cell_volume * 2.0_rt * MathConst::pi * xh;
#else
            amrex::Real const vol = cell_volume;
#endif
            amrex::Real const content = u_home * vol;
            // Electron count of the home cell: the recovery ratio
            // (deposited energy)/(deposited count) gives the energy per
            // electron with the SAME noise realization in numerator and
            // denominator, so the PIC density noise cancels instead of
            // rectifying into a mean-temperature drift (the advection
            // stage's K*N / N construction).
            amrex::Real const count = n_home * vol;

            // Boundary fold: periodic directions wrap by fmod (the
            // wrapped-Gaussian quadrature, exact for periodic boxes and
            // O(1) for kernels wider than the box -- Redistribute's
            // one-period-per-iteration wrap must never see distant
            // positions); non-periodic walls mirror-reflect (insulating
            // image construction).
            auto const reflect = [&] (amrex::Real p, int d) {
                if (is_periodic[d]) {
                    amrex::Real const L = phi[d] - plo[d];
                    p = std::fmod(p - plo[d], L);
                    if (p < 0.0_rt) { p += L; }
                    p += plo[d];
                } else {
                    if (p < lo_bnd[d]) { p = 2.0_rt*lo_bnd[d] - p; }
                    if (p > hi_bnd[d]) { p = 2.0_rt*hi_bnd[d] - p; }
                    p = amrex::Clamp(p, lo_bnd[d], hi_bnd[d]);
                }
                return p;
            };

            for (int kn = 0; kn < n_nodes; ++kn) {
                long const ip = ip0 + kn;
                pa_idcpu[ip] = amrex::SetParticleIDandCPU(pid + ip, cpuid);

                int const kx =  kn % 3;
                int const ky = (n_ax >= 2) ? (kn / 3) % 3 : 0;
                int const kz = (n_ax == 3) ? (kn / 9)     : 0;

                // Tensor-product weight and per-axis kicks. Axis mapping per
                // dim: 3D/RZ kick (x, y, z); XZ kicks (x, z) via (kx, ky);
                // 1D kicks z via kx.
                amrex::Real w = gh_w[kx];
                if (n_ax >= 2) { w *= gh_w[ky]; }
                if (n_ax == 3) { w *= gh_w[kz]; }

                amrex::Real xn = xh, yn = yh, zn = zh;
#if defined(WARPX_DIM_3D)
                xn += gh_xi[kx] * sig + gd[0] * hdrift;
                yn += gh_xi[ky] * sig + gd[1] * hdrift;
                zn += gh_xi[kz] * sig + gd[2] * hdrift;
                xn = reflect(xn, 0); yn = reflect(yn, 1); zn = reflect(zn, 2);
#elif defined(WARPX_DIM_RZ)
                // In-plane radial and off-plane kicks fold into the radius:
                // exact cylindrical diffusion, no metric drift, axis-safe.
                xn += gh_xi[kx] * sig + gd[0] * hdrift;
                yn  = gh_xi[ky] * sig;
                zn += gh_xi[kz] * sig + gd[2] * hdrift;
                xn = std::sqrt(xn*xn + yn*yn);
                yn = 0.0_rt;
                if (xn > hi_bnd[0]) { xn = 2.0_rt*hi_bnd[0] - xn; }
                xn = amrex::Clamp(xn, lo_bnd[0], hi_bnd[0]);
                zn = reflect(zn, 1);
#elif defined(WARPX_DIM_XZ)
                xn += gh_xi[kx] * sig + gd[0] * hdrift;
                zn += gh_xi[ky] * sig + gd[2] * hdrift;
                xn = reflect(xn, 0); zn = reflect(zn, 1);
#else
                zn += gh_xi[kx] * sig + gd[2] * hdrift;
                zn = reflect(zn, 0);
#endif

                // Nodes kicked into an embedded body deposit at home
                // instead (energy stays out of conductors; 3a stub).
                if (have_eb) {
                    amrex::Real const dist =
                        ablastr::particles::doGatherScalarFieldNodal(
                            xn, yn, zn, eb_phi_arr, dxi, plo);
                    if (dist < 0.0_rt) { xn = xh; yn = yh; zn = zh; }
                }

                pa[QdsmcPIdx::x_node][ip] = xh;
                pa[QdsmcPIdx::y_node][ip] = yh;
                pa[QdsmcPIdx::z_node][ip] = zh;
                pa[QdsmcPIdx::vx][ip] = 0.0_rt;
                pa[QdsmcPIdx::vy][ip] = 0.0_rt;
                pa[QdsmcPIdx::vz][ip] = 0.0_rt;
                pa[QdsmcPIdx::entropy][ip] = w * content;
                // Volume measure paired with the energy content: the
                // recovery uses the deposited-energy / deposited-volume
                // ratio, so every kernel, staggering and periodic-seam
                // factor cancels exactly (the same construction as the
                // advection stage's K*N / N recovery).
                pa[QdsmcPIdx::np_real][ip] = w * count;

#if defined(WARPX_DIM_3D)
                pa[QdsmcPIdx::x][ip] = xn;
                pa[QdsmcPIdx::y][ip] = yn;
                pa[QdsmcPIdx::z][ip] = zn;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                pa[QdsmcPIdx::x][ip] = xn;
                pa[QdsmcPIdx::z][ip] = zn;
#elif defined(WARPX_DIM_1D_Z)
                pa[QdsmcPIdx::z][ip] = zn;
#else
                pa[QdsmcPIdx::x][ip] = xn;
#endif
            }
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::DepositConductionEnergy (int lev, amrex::MultiFab & Ufield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositConductionEnergy()");

    // The carried attribute is an energy content [J] (times 2 pi r_home in
    // RZ); the deposition kernel already normalizes by the true cell
    // volume (cylindrical in RZ), so the deposited field is an energy
    // density [J/m^3] with unit scale.
    DepositScalar(lev, QdsmcPIdx::entropy, 1.0_rt, Ufield);
}


void
QdsmcParticleContainer::DepositConductionCount (int lev, amrex::MultiFab & Nfield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositConductionCount()");

    // Companion electron-count measure for the ratio recovery (see
    // SpawnConductionNodes).
    DepositScalar(lev, QdsmcPIdx::np_real, 1.0_rt, Nfield);
}


void
QdsmcParticleContainer::DepositScalar (int lev, int const attr,
                                       amrex::Real const scale,
                                       amrex::MultiFab & field)
{
    auto & warpx = WarpX::GetInstance();
    amrex::Periodicity const & period = warpx.Geom(lev).periodicity();
    amrex::XDim3 const dinv = WarpX::InvCellSize(lev);

    field.setVal(0);

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

        // Assemble the position functor by hand: its constructor indexes the
        // SoA with the physical-particle PIdx layout, which does not match
        // QdsmcPIdx, so it must not be used with this container. The y_node
        // attribute (identically zero outside 3D) stands in for the angle
        // components, so the azimuthal geometries evaluate to (r, 0, z).
        GetParticlePosition<PIdx> GetPosition;
#if defined(WARPX_DIM_3D)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_y = attribs[QdsmcPIdx::y].dataPtr();
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
#elif defined(WARPX_DIM_XZ)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
#elif defined(WARPX_DIM_RZ)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
        GetPosition.m_theta = attribs[QdsmcPIdx::y_node].dataPtr();
#elif defined(WARPX_DIM_1D_Z)
        GetPosition.m_z = attribs[QdsmcPIdx::z].dataPtr();
#elif defined(WARPX_DIM_RCYLINDER)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_theta = attribs[QdsmcPIdx::y_node].dataPtr();
#elif defined(WARPX_DIM_RSPHERE)
        GetPosition.m_x = attribs[QdsmcPIdx::x].dataPtr();
        GetPosition.m_theta = attribs[QdsmcPIdx::y_node].dataPtr();
        GetPosition.m_phi = attribs[QdsmcPIdx::y_node].dataPtr();
#endif

        amrex::Box tilebox = pti.tilebox();
        tilebox.grow(field.nGrowVect());
        amrex::Dim3 const lo = amrex::lbound(tilebox);
        amrex::XDim3 const xyzmin = WarpX::LowerCorner(tilebox, lev, 0.0_rt);

        doChargeDepositionShapeN<1>(GetPosition, attribs[attr].dataPtr(),
                                    nullptr, field[pti], np, dinv, xyzmin, lo,
                                    scale, WarpX::n_rz_azimuthal_modes);
    }

    amrex::Gpu::synchronize();

    ablastr::utils::communication::SumBoundary(
        field, 0, field.nComp(), field.nGrowVect(), field.nGrowVect(),
        WarpX::do_single_precision_comms, period);
}


void
QdsmcParticleContainer::DepositK (int lev, amrex::MultiFab & Kfield)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositK()");

    DepositScalar(lev, QdsmcPIdx::entropy, 1.0_rt, Kfield);
}


void
QdsmcParticleContainer::DepositField (int lev, amrex::MultiFab & Field)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositField()");

    // np_real carries the electron count n_e * V_cell; the 1/V_cell scale
    // makes the deposited field an electron (number) density.
    auto const * dx_arr = WarpX::GetInstance().Geom(lev).CellSize();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }
    DepositScalar(lev, QdsmcPIdx::np_real, 1.0_rt / cell_volume, Field);
}
