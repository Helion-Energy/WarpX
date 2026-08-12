/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Acciarri, Prabhat Kumar (Helion Energy Inc.)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "QdsmcParticleContainer.H"

#include "EmbeddedBoundary/DistanceToEB.H"
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
// stored with NODAL staggering and the markers are homed AT the grid nodes,
// so with the order-1 (linear) nodal weights used by every gather and
// scatter below, a marker at rest reproduces its node values exactly: the
// at-rest gather/deposit round trip is the identity. (Cell-center homes on
// the nodal grid make that round trip a [1/2,1/2]^dim box filter instead,
// i.e. a numerical diffusion of dx^2/(4*dt) per direction regardless of dt
// -- measured to three digits before the switch to node homing.)


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
    auto const phi    = geom.ProbHiArray();

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

        // One marker per OWNED grid node. Nodal tileboxes are disjoint within
        // a box (AMReX gives the box's top node in each direction to the last
        // tile only), but ACROSS boxes the seam node appears in both boxes'
        // nodal boxes, and in periodic directions the domain's top node is
        // the image of the bottom one: trim the high side in those cases so
        // every physical node gets exactly one marker (Sigma(N) must not
        // double-count at seams).
        amrex::Box tile_box = mfi.tilebox(amrex::IntVect::TheNodeVector());
        {
            amrex::Box const box_nodes = amrex::surroundingNodes(mfi.validbox());
            amrex::Box const dom_nodes = amrex::surroundingNodes(geom.Domain());
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (tile_box.bigEnd(d) == box_nodes.bigEnd(d) &&
                    (box_nodes.bigEnd(d) != dom_nodes.bigEnd(d) ||
                     geom.isPeriodic(d))) {
                    tile_box.growHi(d, -1);
                }
            }
        }
        int const grid_id = mfi.index();
        int const tile_id = mfi.LocalTileIndex();

        // One entropy marker per grid node, except nodes fully covered by an
        // embedded boundary: no electron fluid lives there, so no entropy
        // marker is created (deposits into covered cells are inert -- the
        // recovery is gated on the density floor -- and the EB clamp in
        // PushX keeps markers out of the body). Use exclusive scan to assign
        // per-node offsets so the per-node writes are race-free in parallel.
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

            // Compute the node position in physical units. The field
            // dimension determines which axis indices are physically
            // meaningful; missing axes are set to 0 on the particle's home
            // record. At a non-periodic domain-top node the position is
            // pulled just inside the boundary (positions at or beyond
            // ProbHi count as outside and Redistribute would delete the
            // marker) -- same guard as the PushX clamp; interior nodes are
            // unaffected by the min().
            auto const node_pos = [&] (int d_field, int d_iv)
            {
                return amrex::min(
                    plo[d_field] + iv[d_iv] * dx_arr[d_field],
                    phi[d_field] - amrex::Real(1.e-6) * dx_arr[d_field]);
            };
#if defined(WARPX_DIM_3D)
            amrex::Real const x_pos = node_pos(0, 0);
            amrex::Real const y_pos = node_pos(1, 1);
            amrex::Real const z_pos = node_pos(2, 2);
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::y][ip] = y_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            // In 2D Cartesian and RZ the second in-plane coord is z; the y
            // axis is the unused out-of-plane direction.
            amrex::Real const x_pos = node_pos(0, 0);
            auto const y_pos = amrex::Real(0);
            amrex::Real const z_pos = node_pos(1, 1);
            pa[QdsmcPIdx::x][ip] = x_pos;
            pa[QdsmcPIdx::z][ip] = z_pos;
#elif defined(WARPX_DIM_1D_Z)
            auto const x_pos = amrex::Real(0);
            auto const y_pos = amrex::Real(0);
            amrex::Real const z_pos = node_pos(0, 0);
            pa[QdsmcPIdx::z][ip] = z_pos;
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: 1D radial; the single
            // AMReX-tracked position slot is x (= r). QDSMC is not validated
            // in these geometries (no radial volume weighting yet) and
            // HybridPICModel::ReadParameters refuses to enable the energy
            // equation there -- this branch only needs to compile and be sane.
            amrex::Real const x_pos = node_pos(0, 0);
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
            pa[QdsmcPIdx::entropy_g0][ip] = amrex::Real(0);
            pa[QdsmcPIdx::entropy_g1][ip] = amrex::Real(0);
            pa[QdsmcPIdx::entropy_g2][ip] = amrex::Real(0);
            pa[QdsmcPIdx::np_g0][ip] = amrex::Real(0);
            pa[QdsmcPIdx::np_g1][ip] = amrex::Real(0);
            pa[QdsmcPIdx::np_g2][ip] = amrex::Real(0);
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
    amrex::Geometry const & geom = warpx.Geom(lev);
    auto const plo = geom.ProbLoArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const dx  = geom.CellSizeArray();
    auto const * dx_arr = geom.CellSize();

    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }

    // Nodal domain bounds and periodicity for the boundary-safe slope
    // stencils: at a non-periodic domain edge the missing neighbor is
    // replaced by the node value itself, which zeroes the MC slope there.
    amrex::Box const dom_nodes = amrex::surroundingNodes(geom.Domain());
    amrex::GpuArray<int, AMREX_SPACEDIM> ndlo, ndhi, is_per;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        ndlo[d] = dom_nodes.smallEnd(d);
        ndhi[d] = dom_nodes.bigEnd(d);
        is_per[d] = geom.isPeriodic(d);
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
        amrex::GpuArray<amrex::ParticleReal*, 3> s_g = {
            attribs[QdsmcPIdx::entropy_g0].dataPtr(),
            attribs[QdsmcPIdx::entropy_g1].dataPtr(),
            attribs[QdsmcPIdx::entropy_g2].dataPtr()};
        amrex::GpuArray<amrex::ParticleReal*, 3> n_g = {
            attribs[QdsmcPIdx::np_g0].dataPtr(),
            attribs[QdsmcPIdx::np_g1].dataPtr(),
            attribs[QdsmcPIdx::np_g2].dataPtr()};

        auto const K_arr   = Kfield.const_array(pti);
        auto const rho_arr = rhofield.const_array(pti);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Linear gathers of the nodal charge density and entropy at the
            // marker's home position; the marker then carries the electron
            // count N of its dual cell and the matching entropy content K*N.
            // (Kept as gathers -- identical to node reads for node-homed
            // markers -- so the plain-deposit path stays bit-identical.)
            amrex::Real const n_p = ablastr::particles::doGatherScalarFieldNodal(
                x_node[ip], y_node[ip], z_node[ip], rho_arr, dxi, plo)
                * cell_volume / PhysConst::q_e;
            amrex::Real const k_p = ablastr::particles::doGatherScalarFieldNodal(
                x_node[ip], y_node[ip], z_node[ip], K_arr, dxi, plo);

            np_real[ip] = n_p;
            entropy[ip] = k_p * n_p;

            // Limited (MC) slopes of the transported node fields
            // s(x) = K*rho*V/q and n(x) = rho*V/q, per grid dimension, for
            // the half-gradient-corrected deposit. The marker sits on its
            // owning node: snap the compute_weights base index to it.
            int i = 0, j = 0, k = 0;
            amrex::Real W[AMREX_SPACEDIM][2];
            ablastr::particles::compute_weights<amrex::IndexType::CellIndex::NODE>(
                x_node[ip], y_node[ip], z_node[ip], plo, dxi, i, j, k, W);
            amrex::GpuArray<int, 3> iv = {i, j, k};
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (W[d][1] >= 0.5_rt) { iv[d] += 1; }
            }

            auto const node_val = [&] (int const i0, int const j0, int const k0,
                                       bool const with_k)
            {
                amrex::Real const v = rho_arr(i0, j0, k0) * cell_volume / PhysConst::q_e;
                return with_k ? K_arr(i0, j0, k0) * v : v;
            };
            auto const mc = [] (amrex::Real const dfp, amrex::Real const dfm)
            {
                if (dfp * dfm <= 0.0_rt) { return 0.0_rt; }
                amrex::Real const s = (dfp > 0.0_rt) ? 1.0_rt : -1.0_rt;
                return s * amrex::min(2.0_rt * amrex::Math::abs(dfp),
                                      2.0_rt * amrex::Math::abs(dfm),
                                      0.5_rt * amrex::Math::abs(dfp + dfm));
            };

            for (int comp = 0; comp < 2; ++comp) {
                bool const with_k = (comp == 0);
                amrex::ParticleReal* const* gout =
                    with_k ? s_g.data() : n_g.data();
                amrex::Real const f0 = node_val(iv[0], iv[1], iv[2], with_k);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    amrex::GpuArray<int, 3> ivp = iv, ivm = iv;
                    ivp[d] += 1; ivm[d] -= 1;
                    bool const has_p = is_per[d] || (iv[d] + 1 <= ndhi[d]);
                    bool const has_m = is_per[d] || (iv[d] - 1 >= ndlo[d]);
                    amrex::Real const fp = has_p ? node_val(ivp[0], ivp[1], ivp[2], with_k) : f0;
                    amrex::Real const fm = has_m ? node_val(ivm[0], ivm[1], ivm[2], with_k) : f0;
                    gout[d][ip] = mc(fp - f0, f0 - fm) * dxi[d];
                    amrex::ignore_unused(dx);
                }
                for (int d = AMREX_SPACEDIM; d < 3; ++d) {
                    gout[d][ip] = 0.0_rt;
                }
            }
        });
    }

    amrex::Gpu::synchronize();
}


namespace
{
    /** Nodal-interpolated signed distance to the embedded boundary at a
     *  physical point (< 0 inside the conductor). */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_eb_distance (
        amrex::Real const x, amrex::Real const y, amrex::Real const z,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dxi,
        amrex::Array4<amrex::Real const> const & phi_arr)
    {
        int i, j, k;
        amrex::Real W[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            x, y, z, plo, dxi, i, j, k, W);
        return ablastr::particles::interp_field_nodal(i, j, k, W, phi_arr);
    }

    /** Specular level-set mirror of a point inside the conductor:
     *  x -> x - 2 phi(x) n(x). Returns the post-mirror signed distance
     *  (callers fall back to the home position when it is still <= 0,
     *  e.g. deep in a staircase corner where the mirror overshoots). */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real qdsmc_eb_mirror (
        amrex::Real & x, amrex::Real & y, amrex::Real & z,
        amrex::Real const dist,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const & dxi,
        amrex::Array4<amrex::Real const> const & phi_arr)
    {
        auto const n3 = DistanceToEB::interp_normal(
            amrex::ParticleReal(x), amrex::ParticleReal(y),
            amrex::ParticleReal(z), plo, dxi, phi_arr);
        // interp_normal normalizes internally: a degenerate level-set
        // gradient (distance-function ridge) arrives as inf/NaN — treat
        // as unmirrorable so the caller keeps the marker home (a NaN
        // position would crash Redistribute via floor() -> wild index)
        if (!(std::isfinite(n3[0]) && std::isfinite(n3[1]) &&
              std::isfinite(n3[2]))) {
            return dist;
        }
        x -= amrex::Real(2.0) * dist * amrex::Real(n3[0]);
        y -= amrex::Real(2.0) * dist * amrex::Real(n3[1]);
        z -= amrex::Real(2.0) * dist * amrex::Real(n3[2]);
        return qdsmc_eb_distance(x, y, z, plo, dxi, phi_arr);
    }
}

void
QdsmcParticleContainer::GatherVAtMidpoint (int lev, amrex::Real dt,
                                           amrex::MultiFab const* eb_dist,
                                           const amrex::MultiFab & Ux,
                                           const amrex::MultiFab & Uy,
                                           const amrex::MultiFab & Uz)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::GatherVAtMidpoint()");

    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dxi = geom.InvCellSizeArray();
    auto const dx_arr = geom.CellSizeArray();

    // Same clamp policy as PushX: keep the sample point just inside the
    // domain in non-periodic directions; periodic directions gather through
    // the (FillBoundary-filled) ghost layer unchanged.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_bnd;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_bnd;
    amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bnd[d] = plo[d];
        hi_bnd[d] = phi[d] - 1.e-6_rt * dx_arr[d];
        is_periodic[d] = geom.isPeriodic(d);
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
        amrex::ParticleReal* const AMREX_RESTRICT vx =
            attribs[QdsmcPIdx::vx].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vy =
            attribs[QdsmcPIdx::vy].dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT vz =
            attribs[QdsmcPIdx::vz].dataPtr();

        auto const ux_arr = Ux.const_array(pti);
        auto const uy_arr = Uy.const_array(pti);
        auto const uz_arr = Uz.const_array(pti);

        bool const has_eb = (eb_dist != nullptr);
        amrex::Array4<amrex::Real const> const phi_ls =
            has_eb ? eb_dist->const_array(pti)
                   : amrex::Array4<amrex::Real const>{};
        auto const dxi_eb = dxi;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
        {
            // Midpoint of the would-be push, per the 3-vector home record.
            // The clamp applies only to the axes that map to a simulated
            // dimension; the remaining components (e.g. y in 2D) shift
            // freely -- they only matter through the gather's coordinate
            // routing (e.g. r = sqrt(x^2 + y^2) in RZ).
            auto const mid_clamp = [&] (amrex::Real x0, amrex::Real v, int d)
            {
                amrex::Real const xm = x0 + amrex::Real(0.5) * dt * v;
                return is_periodic[d] ? xm
                                      : amrex::Clamp(xm, lo_bnd[d], hi_bnd[d]);
            };

#if defined(WARPX_DIM_3D)
            amrex::Real const xm = mid_clamp(x_node[ip], vx[ip], 0);
            amrex::Real const ym = mid_clamp(y_node[ip], vy[ip], 1);
            amrex::Real const zm = mid_clamp(z_node[ip], vz[ip], 2);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::Real const xm = mid_clamp(x_node[ip], vx[ip], 0);
            amrex::Real const ym = y_node[ip] + amrex::Real(0.5) * dt * vy[ip];
            amrex::Real const zm = mid_clamp(z_node[ip], vz[ip], 1);
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real const xm = x_node[ip] + amrex::Real(0.5) * dt * vx[ip];
            amrex::Real const ym = y_node[ip] + amrex::Real(0.5) * dt * vy[ip];
            amrex::Real const zm = mid_clamp(z_node[ip], vz[ip], 0);
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: compile-and-be-sane
            // only (the energy equation is refused at runtime there).
            amrex::Real const xm = mid_clamp(x_node[ip], vx[ip], 0);
            amrex::Real const ym = y_node[ip] + amrex::Real(0.5) * dt * vy[ip];
            amrex::Real const zm = z_node[ip] + amrex::Real(0.5) * dt * vz[ip];
#endif

            amrex::Real xs = xm, ys = ym, zs = zm;
            if (has_eb) {
                // markers homed inside the conductor never move; a
                // midpoint that dips into the conductor is mirrored
                // back (still-covered mirror -> sample at home)
                if (qdsmc_eb_distance(x_node[ip], y_node[ip], z_node[ip],
                                      plo, dxi_eb, phi_ls)
                        <= amrex::Real(0)) {
                    vx[ip] = amrex::Real(0);
                    vy[ip] = amrex::Real(0);
                    vz[ip] = amrex::Real(0);
                    return;
                }
                amrex::Real const dm =
                    qdsmc_eb_distance(xs, ys, zs, plo, dxi_eb, phi_ls);
                if (dm <= amrex::Real(0)) {
                    if (qdsmc_eb_mirror(xs, ys, zs, dm,
                                        plo, dxi_eb, phi_ls)
                            <= amrex::Real(0)) {
                        xs = x_node[ip]; ys = y_node[ip]; zs = z_node[ip];
                    }
                }
            }

            auto const v = ablastr::particles::doGatherVectorFieldNodal(
                xs, ys, zs, ux_arr, uy_arr, uz_arr, dxi, plo);

            vx[ip] = v[0];
            vy[ip] = v[1];
            vz[ip] = v[2];
        });
    }

    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::PushX (int lev, amrex::Real dt, amrex::Real frac,
                               amrex::MultiFab const* eb_dist)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::PushX()");

    amrex::Geometry const & geom = Geom(lev);
    auto const plo = geom.ProbLoArray();
    auto const phi = geom.ProbHiArray();
    auto const dx_arr = geom.CellSizeArray();
    auto const dxi = geom.InvCellSizeArray();

    // Effective displacement time: frac*dt of the full-step characteristic.
    amrex::Real const fdt = frac * dt;

    // Per-dimension domain clamp bounds. In non-periodic directions the
    // advected position is clamped just inside the domain (positions at or
    // beyond ProbHi count as outside) rather than handed to Redistribute,
    // which would DELETE the marker: since InitParticles runs only once, the
    // home node would then have no QDSMC marker for the rest of the run and
    // its T_e could never be updated again. Clamping instead accumulates the
    // carried entropy at the boundary nodes and preserves the
    // one-marker-per-node invariant (ResetParticles returns it home).
    // Periodic directions are left unclamped so Redistribute wraps them.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_bnd;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_bnd;
    amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bnd[d] = plo[d];
        hi_bnd[d] = phi[d] - 1.e-6_rt * dx_arr[d];
        is_periodic[d] = geom.isPeriodic(d);
    }

    for (iterator pti(*this, lev); pti.isValid(); ++pti)
    {
        long const np = pti.numParticles();
        auto & attribs = pti.GetStructOfArrays().GetRealData();

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

        bool const has_eb = (eb_dist != nullptr);
        amrex::Array4<amrex::Real const> const phi_ls =
            has_eb ? eb_dist->const_array(pti)
                   : amrex::Array4<amrex::Real const>{};

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

            // Tentative new position (domain-clamped per axis; axes not
            // represented in the field have no enum slot and are simply
            // not tracked, consistent with field dimensionality).
#if defined(WARPX_DIM_3D)
            amrex::Real xn = push_clamp(x_node[ip], vx[ip], 0);
            amrex::Real yn = push_clamp(y_node[ip], vy[ip], 1);
            amrex::Real zn = push_clamp(z_node[ip], vz[ip], 2);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            amrex::Real xn = push_clamp(x_node[ip], vx[ip], 0);
            amrex::Real yn = amrex::Real(0);   // home y is 0 off-plane
            amrex::Real zn = push_clamp(z_node[ip], vz[ip], 1);
#elif defined(WARPX_DIM_1D_Z)
            amrex::Real zn = push_clamp(z_node[ip], vz[ip], 0);
#else
            // WARPX_DIM_RCYLINDER / WARPX_DIM_RSPHERE: x (= r) is the single
            // tracked position (QDSMC is refused at runtime in these
            // geometries); z is a plain attribute, advanced unclamped.
            amrex::Real xn = push_clamp(x_node[ip], vx[ip], 0);
            amrex::Real yn = amrex::Real(0);
            amrex::Real zn = z_node[ip] + vz[ip] * fdt;
#endif

#if !defined(WARPX_DIM_1D_Z)
            if (has_eb) {
                // Markers homed inside the conductor never move (V_e is
                // not physical there); fluid markers pushed into the
                // conductor are specularly mirrored back across the
                // level set (a mirror that lands covered again — deep
                // staircase corner — keeps the marker home). This is
                // the adiabatic E7-replacement at the EB, matching the
                // conduction sweeps' fold-back.
#if defined(WARPX_DIM_3D)
                amrex::Real const hy_eb = y_node[ip];
#else
                amrex::Real const hy_eb = amrex::Real(0);
#endif
                if (qdsmc_eb_distance(x_node[ip], hy_eb, z_node[ip],
                                      plo, dxi, phi_ls)
                        <= amrex::Real(0)) {
                    xn = x_node[ip]; yn = hy_eb; zn = z_node[ip];
                } else {
                    amrex::Real const dn =
                        qdsmc_eb_distance(xn, yn, zn, plo, dxi, phi_ls);
                    if (dn <= amrex::Real(0)) {
                        amrex::Real xr = xn, yr = yn, zr = zn;
                        if (qdsmc_eb_mirror(xr, yr, zr, dn,
                                            plo, dxi, phi_ls)
                                > amrex::Real(0)) {
                            xn = xr; yn = yr; zn = zr;
                        } else {
                            xn = x_node[ip];
                            yn = hy_eb;
                            zn = z_node[ip];
                        }
                    }
                }
            }
#else
            amrex::ignore_unused(has_eb, phi_ls, dxi);
#endif
#if !defined(WARPX_DIM_1D_Z)
            amrex::ignore_unused(yn);
#endif

            // Write the new position to the AMReX-tracked position slots.
#if defined(WARPX_DIM_3D)
            pa_x[ip] = xn;
            pa_y[ip] = yn;
            pa_z[ip] = zn;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            pa_x[ip] = xn;
            pa_z[ip] = zn;
#elif defined(WARPX_DIM_1D_Z)
            pa_z[ip] = zn;
#else
            pa_x[ip] = xn;
            pa_z[ip] = zn;
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
        amrex::GpuArray<amrex::ParticleReal*, 6> slopes = {
            attribs[QdsmcPIdx::entropy_g0].dataPtr(),
            attribs[QdsmcPIdx::entropy_g1].dataPtr(),
            attribs[QdsmcPIdx::entropy_g2].dataPtr(),
            attribs[QdsmcPIdx::np_g0].dataPtr(),
            attribs[QdsmcPIdx::np_g1].dataPtr(),
            attribs[QdsmcPIdx::np_g2].dataPtr()};

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
            for (int c = 0; c < 6; ++c) { slopes[c][ip] = 0; }
        });
    }

    Redistribute();
    amrex::Gpu::synchronize();
}


void
QdsmcParticleContainer::DepositScalar (int lev, int const attr,
                                       int const slope_attr0,
                                       amrex::Real const scale,
                                       amrex::MultiFab & field,
                                       CliffDeposit const & cliff)
{
    auto & warpx = WarpX::GetInstance();
    amrex::Geometry const & geom = warpx.Geom(lev);
    amrex::Periodicity const & period = geom.periodicity();
    amrex::XDim3 const dinv = WarpX::InvCellSize(lev);

    // Cliff-aware blend (see the CliffDeposit doc): only the
    // gradient-corrected path is instrumented.
    bool const cliff_on = cliff.enabled && (slope_attr0 >= 0);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(cliff.enabled && slope_attr0 < 0),
        "QdsmcParticleContainer::DepositScalar: the cliff-limited deposit "
        "is implemented for the gradient-corrected path only "
        "(qdsmc_gradient_deposit = 1)");
    amrex::Real const cl_nfloor = cliff.n_floor;
    amrex::Real const cl_gm1    = cliff.gamma - 1.0_rt;
    amrex::Real const cl_r1     = cliff.r1;
    amrex::Real const cl_inv_dr =
        1.0_rt / amrex::max(cliff.r2 - cliff.r1, 1.0e-12_rt);
    amrex::Real const inv_qe    = 1.0_rt / PhysConst::q_e;
    amrex::Real cl_inv_vol = 1.0_rt;
    {
        auto const * dxs = geom.CellSize();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { cl_inv_vol /= dxs[d]; }
    }

    field.setVal(0);

    // Half-gradient-corrected deposit: field_j += w_ij * scale *
    // (A_i + 1/2 G_i . (x_j - y_i)), with y_i the pushed position and G_i
    // the limited slopes loaded by SetK. Sum_j w_ij (x_j - y_i) = 0 for the
    // linear hat (its weighted mean is the particle position), so the
    // correction conserves Sum(A) exactly; the 1/2 cancels the hat's second
    // moment, making the remap exact through quadratics.
    if (slope_attr0 >= 0)
    {
        auto const plo = geom.ProbLoArray();
        auto const dxi = geom.InvCellSizeArray();
        auto const dx  = geom.CellSizeArray();

        for (iterator pti(*this, lev); pti.isValid(); ++pti)
        {
            long const np = pti.numParticles();
            auto & attribs = pti.GetStructOfArrays().GetRealData();

            const amrex::ParticleReal* AMREX_RESTRICT ap =
                attribs[attr].dataPtr();
            amrex::GpuArray<const amrex::ParticleReal*, AMREX_SPACEDIM> gp;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                gp[d] = attribs[slope_attr0 + d].dataPtr();
            }

            // Cliff-aware blend inputs: destination density and the
            // marker's home (seed-time) density n = N / V_cell.
            amrex::Array4<amrex::Real const> cl_rho{};
            const amrex::ParticleReal* AMREX_RESTRICT cl_np = nullptr;
            if (cliff_on) {
                cl_rho = cliff.rho_new->const_array(pti);
                cl_np  = attribs[QdsmcPIdx::np_real].dataPtr();
            }

            // Pushed positions from the tracked slots, mapped to the
            // (xp, yp, zp) convention compute_weights expects (the
            // out-of-plane components are 0, matching the theta = 0
            // convention of the plain-deposit path).
#if !defined(WARPX_DIM_1D_Z)
            const amrex::ParticleReal* AMREX_RESTRICT pa_x =
                attribs[QdsmcPIdx::x].dataPtr();
#endif
#if defined(WARPX_DIM_3D)
            const amrex::ParticleReal* AMREX_RESTRICT pa_y =
                attribs[QdsmcPIdx::y].dataPtr();
#endif
            const amrex::ParticleReal* AMREX_RESTRICT pa_z =
                attribs[QdsmcPIdx::z].dataPtr();

            auto out = field.array(pti);

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
            {
#if defined(WARPX_DIM_3D)
                amrex::ParticleReal const xp = pa_x[ip];
                amrex::ParticleReal const yp = pa_y[ip];
                amrex::ParticleReal const zp = pa_z[ip];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::ParticleReal const xp = pa_x[ip];
                auto const yp = amrex::ParticleReal(0);
                amrex::ParticleReal const zp = pa_z[ip];
#elif defined(WARPX_DIM_1D_Z)
                auto const xp = amrex::ParticleReal(0);
                auto const yp = amrex::ParticleReal(0);
                amrex::ParticleReal const zp = pa_z[ip];
#else
                amrex::ParticleReal const xp = pa_x[ip];
                auto const yp = amrex::ParticleReal(0);
                auto const zp = amrex::ParticleReal(0);
#endif
                int i = 0, j = 0, k = 0;
                amrex::Real W[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::CellIndex::NODE>(
                    xp, yp, zp, plo, dxi, i, j, k, W);

                amrex::Real const a = ap[ip];

                // Cliff-aware per-destination blend (CliffDeposit doc):
                // factor 1 exactly (no transcendental evaluated) whenever
                // |ln(n_dest/n_home)| <= r1, so the disengaged path is
                // bit-identical; beyond r1 blend toward the isothermal
                // spill factor (n_home/n_dest)^(gamma-1) = e^{-(gamma-1) lr}.
                amrex::Real cl_nh = 1.0_rt;
                if (cliff_on) {
                    cl_nh = amrex::max(cl_np[ip] * cl_inv_vol, cl_nfloor);
                }
                auto const cl_factor = [&] (int const ci, int const cj,
                                            int const ck)
                {
                    if (!cliff_on) { return 1.0_rt; }
                    amrex::Real const nd = amrex::max(
                        cl_rho(ci, cj, ck) * inv_qe, cl_nfloor);
                    amrex::Real const lr = std::log(nd / cl_nh);
                    amrex::Real const ar = amrex::Math::abs(lr);
                    if (ar <= cl_r1) { return 1.0_rt; }
                    amrex::Real const eta =
                        amrex::min((ar - cl_r1) * cl_inv_dr, 1.0_rt);
                    // One-sided: cap the isothermal factor at 1, so the
                    // correction only ever REDUCES the deposited entropy
                    // (kills the up-cliff Te amplification) and never
                    // deposits more than the marker carried (the uncapped
                    // down-cliff factor > 1 measured as a 10x band ENERGY
                    // INJECTOR on the SPN4 arm -- isothermal down-spill
                    // manufactures Sum(K N)).
                    return (1.0_rt - eta)
                        + eta * amrex::min(1.0_rt, std::exp(-cl_gm1 * lr));
                };
#if defined(WARPX_DIM_3D)
                for (int kk = 0; kk < 2; ++kk) {
                for (int jj = 0; jj < 2; ++jj) {
                for (int ii = 0; ii < 2; ++ii) {
                    amrex::Real const w = W[0][ii] * W[1][jj] * W[2][kk];
                    amrex::Real const val = a + amrex::Real(0.5) * (
                        gp[0][ip] * (ii - W[0][1]) * dx[0] +
                        gp[1][ip] * (jj - W[1][1]) * dx[1] +
                        gp[2][ip] * (kk - W[2][1]) * dx[2]);
                    amrex::Gpu::Atomic::AddNoRet(
                        &out(i + ii, j + jj, k + kk),
                        scale * w * val * cl_factor(i + ii, j + jj, k + kk));
                }}}
#elif (AMREX_SPACEDIM == 2)
                for (int jj = 0; jj < 2; ++jj) {
                for (int ii = 0; ii < 2; ++ii) {
                    amrex::Real const w = W[0][ii] * W[1][jj];
                    amrex::Real const val = a + amrex::Real(0.5) * (
                        gp[0][ip] * (ii - W[0][1]) * dx[0] +
                        gp[1][ip] * (jj - W[1][1]) * dx[1]);
                    amrex::Gpu::Atomic::AddNoRet(
                        &out(i + ii, j + jj, k),
                        scale * w * val * cl_factor(i + ii, j + jj, k));
                }}
#else
                for (int ii = 0; ii < 2; ++ii) {
                    amrex::Real const w = W[0][ii];
                    amrex::Real const val = a + amrex::Real(0.5) *
                        gp[0][ip] * (ii - W[0][1]) * dx[0];
                    amrex::Gpu::Atomic::AddNoRet(
                        &out(i + ii, j, k),
                        scale * w * val * cl_factor(i + ii, j, k));
                }
#endif
            });
        }

        amrex::Gpu::synchronize();

        ablastr::utils::communication::SumBoundary(
            field, 0, field.nComp(), field.nGrowVect(), field.nGrowVect(),
            WarpX::do_single_precision_comms, period);
        return;
    }

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
QdsmcParticleContainer::DepositK (int lev, amrex::MultiFab & Kfield,
                                  bool const gradient_corrected,
                                  CliffDeposit const & cliff)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositK()");

    DepositScalar(lev, QdsmcPIdx::entropy,
                  gradient_corrected ? int(QdsmcPIdx::entropy_g0) : -1,
                  1.0_rt, Kfield, cliff);
}


void
QdsmcParticleContainer::DepositK (int lev, amrex::MultiFab & Kfield,
                                  bool const gradient_corrected)
{
    DepositK(lev, Kfield, gradient_corrected, CliffDeposit{});
}


void
QdsmcParticleContainer::DepositField (int lev, amrex::MultiFab & Field,
                                      bool const gradient_corrected)
{
    ABLASTR_PROFILE("QdsmcParticleContainer::DepositField()");

    // np_real carries the electron count n_e * V_cell; the 1/V_cell scale
    // makes the deposited field an electron (number) density.
    auto const * dx_arr = WarpX::GetInstance().Geom(lev).CellSize();
    amrex::Real cell_volume = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cell_volume *= dx_arr[d];
    }
    DepositScalar(lev, QdsmcPIdx::np_real,
                  gradient_corrected ? int(QdsmcPIdx::np_g0) : -1,
                  1.0_rt / cell_volume, Field, CliffDeposit{});
}
