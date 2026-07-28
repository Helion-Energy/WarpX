/* Copyright 2026 S. Eric Clark (Helion Energy)
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SubcycledParticleContainer.H"

#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#   include "EmbeddedBoundary/ParticleBoundaryProcess.H"
#   include "EmbeddedBoundary/ParticleScraper.H"
#endif
#include "Fields.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>

using namespace amrex;

SubcycledParticleContainer::SubcycledParticleContainer (amrex::AmrCore* amr_core,
                                                        int ispecies,
                                                        const std::string& name)
    : PhysicalParticleContainer{amr_core, ispecies, name}
{
    const ParmParse pp_species_name(species_name);

    utils::parser::queryWithParser(pp_species_name, "subcycling_cfl_cyclotron", m_cfl_cyclotron);
    utils::parser::queryWithParser(pp_species_name, "subcycling_cfl_grid", m_cfl_grid);
    utils::parser::queryWithParser(pp_species_name, "subcycling_v_ref", m_v_ref);
    utils::parser::queryWithParser(pp_species_name, "subcycling_max_subcycles", m_max_subcycles);
    pp_species_name.query("do_orbit_averaged_deposition", m_do_orbit_averaged_deposition);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_cfl_cyclotron > 0._rt && m_cfl_grid > 0._rt && m_max_subcycles > 0,
        "<species>.subcycling_cfl_cyclotron, subcycling_cfl_grid and "
        "subcycling_max_subcycles must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !do_splitting,
        "Particle splitting is not supported for subcycled species");
}

void
SubcycledParticleContainer::ComputeSubcycleCount (ablastr::fields::MultiFabRegister& fields,
                                                  int lev, amrex::Real dt)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Conservative upper bound on |B|: combine the per-component max norms
    // (the components live on different staggerings).
    amrex::Real bmax_sq = 0._rt;
    for (int dir = 0; dir < 3; ++dir)
    {
        if (fields.has(FieldType::Bfield_aux, Direction{dir}, lev))
        {
            const auto* B = fields.get(FieldType::Bfield_aux, Direction{dir}, lev);
            amrex::Real comp_max = 0._rt;
            for (int ic = 0; ic < B->nComp(); ++ic)
            {
                comp_max = amrex::max(comp_max, B->norminf(ic));
            }
            bmax_sq += comp_max*comp_max;
        }
    }
    const amrex::Real bmax = std::sqrt(bmax_sq);

    amrex::Real dt_sub = dt;

    if (bmax > 0._rt && m_charge != 0._prt)
    {
        const auto omega_c = static_cast<amrex::Real>(std::abs(m_charge) * bmax / m_mass);
        dt_sub = amrex::min(dt_sub, m_cfl_cyclotron / omega_c);
    }

    const amrex::Real v_ref = (m_v_ref > 0._rt) ? m_v_ref : MaxParticleSpeed();
    if (v_ref > 0._rt)
    {
        const auto dx = WarpX::GetInstance().Geom(lev).CellSizeArray();
        amrex::Real dx_min = dx[0];
        for (int d = 1; d < AMREX_SPACEDIM; ++d) { dx_min = amrex::min(dx_min, dx[d]); }
        dt_sub = amrex::min(dt_sub, m_cfl_grid * dx_min / v_ref);
    }

    // Safety cap on the subcycle count
    dt_sub = amrex::max(dt_sub, dt / static_cast<amrex::Real>(m_max_subcycles));

    m_dt_subcycle = dt_sub;
    // The small tolerance avoids an extra (zero-length) subcycle when
    // dt/dt_sub is an exact integer up to roundoff.
    m_n_subcycles = amrex::max(1, static_cast<int>(std::ceil(dt/dt_sub * (1._rt - 1.e-12))));
}

amrex::Real
SubcycledParticleContainer::MaxParticleSpeed ()
{
    if (TotalNumberOfParticles(true, false) == 0) { return 0._rt; }

    using PType = typename WarpXParticleContainer::SuperParticleType;
    const amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);

    amrex::ParticleReal vmax = amrex::ReduceMax(*this,
        [=] AMREX_GPU_HOST_DEVICE (const PType& p) noexcept -> amrex::ParticleReal
        {
            const amrex::ParticleReal ux = p.rdata(PIdx::ux);
            const amrex::ParticleReal uy = p.rdata(PIdx::uy);
            const amrex::ParticleReal uz = p.rdata(PIdx::uz);
            const amrex::ParticleReal usq = ux*ux + uy*uy + uz*uz;
            return std::sqrt(usq / (1._prt + usq*inv_c2));
        });
    amrex::ParallelDescriptor::ReduceRealMax(vmax);

    return amrex::max(static_cast<amrex::Real>(vmax), 0._rt);
}

void
SubcycledParticleContainer::PrepareMomentAccumulators (ablastr::fields::MultiFabRegister& fields,
                                                       int lev,
                                                       const std::string& current_fp_string)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    m_average_j = fields.has(current_fp_string, Direction{0}, lev);
    for (int dir = 0; dir < 3 && m_average_j; ++dir)
    {
        const auto* j = fields.get(current_fp_string, Direction{dir}, lev);
        if (!m_j_avg[dir]
            || m_j_avg[dir]->boxArray() != j->boxArray()
            || m_j_avg[dir]->DistributionMap() != j->DistributionMap()
            || m_j_avg[dir]->nGrowVect() != j->nGrowVect()
            || m_j_avg[dir]->nComp() != j->nComp())
        {
            m_j_avg[dir] = std::make_unique<amrex::MultiFab>(
                j->boxArray(), j->DistributionMap(), j->nComp(), j->nGrowVect());
            m_j_tmp[dir] = std::make_unique<amrex::MultiFab>(
                j->boxArray(), j->DistributionMap(), j->nComp(), j->nGrowVect());
        }
        m_j_avg[dir]->setVal(0._rt);
    }

    m_average_rho = fields.has(FieldType::rho_fp, lev);
    if (m_average_rho)
    {
        const auto* rho = fields.get(FieldType::rho_fp, lev);
        if (!m_rho_avg
            || m_rho_avg->boxArray() != rho->boxArray()
            || m_rho_avg->DistributionMap() != rho->DistributionMap()
            || m_rho_avg->nGrowVect() != rho->nGrowVect())
        {
            m_rho_avg = std::make_unique<amrex::MultiFab>(
                rho->boxArray(), rho->DistributionMap(), WarpX::ncomps, rho->nGrowVect());
            m_rho_tmp = std::make_unique<amrex::MultiFab>(
                rho->boxArray(), rho->DistributionMap(), WarpX::ncomps, rho->nGrowVect());
        }
        m_rho_avg->setVal(0._rt);
    }
}

void
SubcycledParticleContainer::HandleBoundariesSubcycle (amrex::Real cur_time, amrex::Real dt_sub)
{
    auto& warpx = WarpX::GetInstance();
    auto& boundary_buffer = warpx.GetParticleBoundaryBuffer();

    // Domain boundaries: apply the particle boundary conditions, then record
    // the scraped particles with the sub-step time stamp.
    ApplyBoundaryConditions();
    boundary_buffer.gatherParticlesFromDomainBoundaries(
        *this, getSpeciesId(), cur_time, dt_sub);

    // Move particles to their new boxes/tiles (also removes particles
    // invalidated by absorbing domain boundaries).
    Redistribute();

#ifdef AMREX_USE_EB
    if (EB::enabled())
    {
        using warpx::fields::FieldType;
        const auto distance_to_eb =
            warpx.m_fields.get_mr_levels(FieldType::distance_to_eb, finestLevel());

        // Match the order of WarpX::HandleParticlesAtBoundaries: mark/reflect
        // at the embedded boundary, record the scraped particles, then remove
        // the absorbed ones.
        if (WarpX::eb_particle_boundary == ParticleBoundaryType::Reflecting)
        {
            for (int lev = 0; lev <= finestLevel(); ++lev)
            {
                scrapeParticlesAtEB(*this, distance_to_eb, lev,
                    ParticleBoundaryProcess::Reflect{dt_sub, getMass()});
            }
            boundary_buffer.gatherParticlesFromEmbeddedBoundaries(
                *this, getSpeciesId(), distance_to_eb, cur_time, dt_sub);
            Redistribute();
        }
        else
        {
            scrapeParticlesAtEB(*this, distance_to_eb, ParticleBoundaryProcess::Absorb());
            boundary_buffer.gatherParticlesFromEmbeddedBoundaries(
                *this, getSpeciesId(), distance_to_eb, cur_time, dt_sub);
            deleteInvalidParticles();
        }
    }
#endif
}

void
SubcycledParticleContainer::Evolve (ablastr::fields::MultiFabRegister& fields,
                                    int lev,
                                    const std::string& current_fp_string,
                                    amrex::Real t,
                                    amrex::Real dt,
                                    SubcyclingHalf subcycling_half,
                                    bool skip_deposition,
                                    PositionPushType position_push_type,
                                    MomentumPushType momentum_push_type,
                                    ImplicitOptions const * implicit_options)
{
    ABLASTR_PROFILE("SubcycledParticleContainer::Evolve()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        implicit_options == nullptr,
        "Subcycled species are not compatible with implicit evolution schemes");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        subcycling_half == SubcyclingHalf::None,
        "Subcycled species are not compatible with mesh-refinement subcycling");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0 && finestLevel() == 0,
        "Subcycled species require a single mesh level");

    m_moments_valid = false;

    if (do_not_push || dt <= 0._rt)
    {
        PhysicalParticleContainer::Evolve(fields, lev, current_fp_string, t, dt,
            subcycling_half, skip_deposition, position_push_type, momentum_push_type,
            implicit_options);
        return;
    }

    ComputeSubcycleCount(fields, lev, dt);

    const bool average = m_do_orbit_averaged_deposition && !do_not_deposit;
    if (average) { PrepareMomentAccumulators(fields, lev, current_fp_string); }

    amrex::Real t_sub = t;
    amrex::Real remaining = dt;
    for (int isub = 0; isub < m_n_subcycles && remaining > 0._rt; ++isub)
    {
        // Truncate the final subcycle so the subcycles sum exactly to dt
        // (which makes the orbit-average weights sum exactly to one).
        const amrex::Real dt_sub = amrex::min(m_dt_subcycle, remaining);

        // Gather + push only, against the frozen fields of this step;
        // deposition is handled below so it can be scaled and accumulated.
        PhysicalParticleContainer::Evolve(fields, lev, current_fp_string, t_sub, dt_sub,
            SubcyclingHalf::None, /*skip_deposition=*/true,
            position_push_type, momentum_push_type, nullptr);

        // Boundary handling every subcycle: absorbed particles stop
        // contributing to the averaged moments from this subcycle on, and
        // wall-loss records carry the sub-step time stamp.
        HandleBoundariesSubcycle(t_sub + dt_sub, dt_sub);

        if (average)
        {
            const amrex::Real lambda = dt_sub / dt;

            if (m_average_j)
            {
                for (int dir = 0; dir < 3; ++dir) { m_j_tmp[dir]->setVal(0._rt); }
                const ablastr::fields::MultiLevelVectorField j_tmp{
                    {m_j_tmp[0].get(), m_j_tmp[1].get(), m_j_tmp[2].get()}};
                WarpXParticleContainer::DepositCurrent(j_tmp, dt_sub, -0.5_rt*dt_sub);
                for (int dir = 0; dir < 3; ++dir)
                {
                    amrex::MultiFab::Saxpy(*m_j_avg[dir], lambda, *m_j_tmp[dir],
                        0, 0, m_j_tmp[dir]->nComp(), m_j_tmp[dir]->nGrowVect());
                }
            }

            if (m_average_rho)
            {
                m_rho_tmp->setVal(0._rt);
                const ablastr::fields::MultiLevelScalarField rho_tmp{m_rho_tmp.get()};
                WarpXParticleContainer::DepositCharge(rho_tmp,
                    /*local=*/true, /*reset=*/false,
                    /*apply_boundary_and_scale_volume=*/false,
                    /*interpolate_across_levels=*/false, /*icomp=*/0);
                amrex::MultiFab::Saxpy(*m_rho_avg, lambda, *m_rho_tmp,
                    0, 0, m_rho_tmp->nComp(), m_rho_tmp->nGrowVect());
            }
        }

        t_sub += dt_sub;
        remaining -= dt_sub;
    }

    m_moments_valid = average;

    if (!skip_deposition && !do_not_deposit)
    {
        // The caller expected this Evolve call to deposit the current
        // (electromagnetic path): serve the orbit average, or fall back to a
        // single (approximate) deposit at the final particle positions.
        if (!average)
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                WarpX::current_deposition_algo == CurrentDepositionAlgo::Direct,
                "Subcycled species require do_orbit_averaged_deposition=1 when "
                "used with charge-conserving (Esirkepov/Villasenor) current deposition");
        }
        using ablastr::fields::Direction;
        auto* jx = fields.get(current_fp_string, Direction{0}, lev);
        auto* jy = fields.get(current_fp_string, Direction{1}, lev);
        auto* jz = fields.get(current_fp_string, Direction{2}, lev);
        const ablastr::fields::MultiLevelVectorField J{{jx, jy, jz}};
        DepositCurrent(J, dt, -0.5_rt*dt);
    }
}

void
SubcycledParticleContainer::PushP (int /*lev*/, amrex::Real /*dt*/,
                                   const amrex::MultiFab& /*Ex*/,
                                   const amrex::MultiFab& /*Ey*/,
                                   const amrex::MultiFab& /*Ez*/,
                                   const amrex::MultiFab& /*Bx*/,
                                   const amrex::MultiFab& /*By*/,
                                   const amrex::MultiFab& /*Bz*/,
                                   MomentumPushType /*momentum_push_type*/)
{
    // Intentionally empty: the momenta of a subcycled species stay on the
    // subcycle stagger (synchronized with the positions to within half a
    // subcycle); half-global-step momentum shifts would over-rotate the
    // velocities by up to many gyration periods.
}

void
SubcycledParticleContainer::DepositCurrent (ablastr::fields::MultiLevelVectorField const & J,
                                            const amrex::Real dt, const amrex::Real relative_time)
{
    if (m_moments_valid && m_average_j && m_j_avg[0])
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(J.size()) == 1,
            "Subcycled species require a single mesh level");
        for (int dir = 0; dir < 3; ++dir)
        {
            amrex::IntVect ng = J[0][dir]->nGrowVect();
            ng.min(m_j_avg[dir]->nGrowVect());
            amrex::MultiFab::Add(*J[0][dir], *m_j_avg[dir],
                                 0, 0, m_j_avg[dir]->nComp(), ng);
        }
        return;
    }

    WarpXParticleContainer::DepositCurrent(J, dt, relative_time);
}

void
SubcycledParticleContainer::DepositCharge (const ablastr::fields::MultiLevelScalarField& rho,
                                           const bool local, const bool reset,
                                           const bool apply_boundary_and_scale_volume,
                                           const bool interpolate_across_levels,
                                           const int icomp)
{
    amrex::ignore_unused(interpolate_across_levels);

    if (m_moments_valid && m_average_rho && m_rho_avg)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(rho.size()) == 1,
            "Subcycled species require a single mesh level");

        auto& warpx = WarpX::GetInstance();
        const int nc = WarpX::ncomps;

        // Mirror the flag handling of the standard DepositCharge
        if (reset) { rho[0]->setVal(0._rt, icomp*nc, nc, rho[0]->nGrowVect()); }

        amrex::IntVect ng = rho[0]->nGrowVect();
        ng.min(m_rho_avg->nGrowVect());
        amrex::MultiFab::Add(*rho[0], *m_rho_avg, 0, icomp*nc, nc, ng);

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        if (apply_boundary_and_scale_volume)
        {
            warpx.ApplyInverseVolumeScalingToChargeDensity(rho[0], 0);
        }
#endif
        if (!local)
        {
            ablastr::utils::communication::SumBoundary(
                *rho[0], 0, rho[0]->nComp(), rho[0]->nGrowVect(), rho[0]->nGrowVect(),
                WarpX::do_single_precision_comms,
                warpx.Geom(0).periodicity());
        }
        if (apply_boundary_and_scale_volume)
        {
            warpx.ApplyRhofieldBoundary(0, rho[0], PatchType::fine);
        }
        return;
    }

    WarpXParticleContainer::DepositCharge(rho, local, reset,
        apply_boundary_and_scale_volume, interpolate_across_levels, icomp);
}
