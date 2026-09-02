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
    pp_species_name.query("subcycling_v_ref_meridional", m_v_ref_meridional);
    utils::parser::queryWithParser(pp_species_name, "subcycling_v_theta_margin", m_v_theta_margin);
    utils::parser::queryWithParser(pp_species_name, "subcycling_v_ref_hysteresis", m_v_ref_hysteresis);
    pp_species_name.query("subcycling_cap_is_error", m_cap_is_error);
    pp_species_name.query("do_orbit_averaged_deposition", m_do_orbit_averaged_deposition);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_cfl_cyclotron > 0._rt && m_cfl_grid > 0._rt && m_max_subcycles > 0,
        "<species>.subcycling_cfl_cyclotron, subcycling_cfl_grid and "
        "subcycling_max_subcycles must be positive");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_v_theta_margin >= 0._rt,
        "<species>.subcycling_v_theta_margin must be non-negative");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_v_ref_hysteresis >= 0._rt && m_v_ref_hysteresis < 1._rt,
        "<species>.subcycling_v_ref_hysteresis must lie in [0, 1)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_v_ref_meridional != 0 && m_v_ref > 0._rt),
        "<species>.subcycling_v_ref_meridional sizes the grid CFL on the "
        "measured meridional speed and therefore requires the measured "
        "reference speed, i.e. <species>.subcycling_v_ref <= 0");
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

    // Reference speed for the grid-crossing CFL. A positive input is used as
    // given; otherwise the speed is measured from the particles this step.
    // The measured path can size on the meridional speed instead of the full
    // speed (see m_v_ref_meridional), which requires cylindrical geometry
    // with a single azimuthal mode: only then does motion in theta cross no
    // (r,z) cell, so that the deposition guard band -- the constraint the
    // grid CFL exists to satisfy -- is set by the meridional speed alone.
    const bool measured = (m_v_ref <= 0._rt);
    bool meridional_used = false;
    MaxSpeeds speeds;
    amrex::Real v_ref = m_v_ref;

    if (measured)
    {
#if defined(WARPX_DIM_RZ)
        const bool meridional_ok = (WarpX::n_rz_azimuthal_modes == 1);
#else
        const bool meridional_ok = false;
#endif
        if (m_v_ref_meridional != 0 && !meridional_ok && !m_meridional_gate_reported)
        {
            ablastr::warn_manager::WMRecordWarning("SubcycledParticleContainer",
                "Species '" + species_name + "': subcycling_v_ref_meridional "
                "requires cylindrical geometry with a single azimuthal mode "
                "(warpx.n_rz_azimuthal_modes = 1), because only then does "
                "azimuthal motion cross no cell of the (r,z) deposition grid. "
                "Falling back to sizing the grid CFL on the FULL particle "
                "speed, which is conservative.",
                ablastr::warn_manager::WarnPriority::high);
            m_meridional_gate_reported = true;
        }
        meridional_used = (m_v_ref_meridional != 0) && meridional_ok;

        if (meridional_used)
        {
            // Re-measured every evaluation: never inferred from a cached
            // azimuthal fraction, which is a property of the current regime
            // and would silently go stale under pitch-angle scattering.
            speeds = MaxParticleSpeedsByComponent();
            // Per-particle combination (see MaxSpeeds), then a hard clamp at
            // the full-speed answer. The clamp makes the meridional option
            // unable to size ABOVE the baseline it is trying to improve on:
            // its worst case is parity, never a regression.
            v_ref = amrex::min(speeds.combined, speeds.full);
        }
        else
        {
            v_ref = MaxParticleSpeed();
            speeds.full = v_ref;
        }
    }

    const auto dx = WarpX::GetInstance().Geom(lev).CellSizeArray();
    amrex::Real dx_min = dx[0];
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dx_min = amrex::min(dx_min, dx[d]); }

    if (v_ref > 0._rt)
    {
        dt_sub = amrex::min(dt_sub, m_cfl_grid * dx_min / v_ref);
    }

    // Safety cap on the subcycle count. On the measured path a binding cap
    // means the requested resolution was silently dropped -- the same
    // under-resolution a stale reference speed would produce, but leaving no
    // trace -- so refuse to continue instead of clamping.
    const amrex::Real dt_sub_capped = dt / static_cast<amrex::Real>(m_max_subcycles);
    if (dt_sub < dt_sub_capped)
    {
        const auto n_wanted = static_cast<int>(
            std::ceil(dt/dt_sub * (1._rt - 1.e-12)));
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(measured && m_cap_is_error != 0),
            "Species '" + species_name + "': the subcycle count required by "
            "the CFL conditions (" + std::to_string(n_wanted) + ") exceeds "
            "<species>.subcycling_max_subcycles (" +
            std::to_string(m_max_subcycles) + ") at reference speed " +
            std::to_string(v_ref) + " m/s and |B|max " + std::to_string(bmax) +
            " T. Clamping would silently under-resolve the particle motion "
            "within a subcycle, so the run is stopped instead. Raise the cap, "
            "relax subcycling_cfl_grid / subcycling_cfl_cyclotron, or set "
            "<species>.subcycling_cap_is_error = 0 to restore clamping.");
        dt_sub = dt_sub_capped;
    }

    // The small tolerance avoids an extra (zero-length) subcycle when
    // dt/dt_sub is an exact integer up to roundoff.
    const int n_implied =
        amrex::max(1, static_cast<int>(std::ceil(dt/dt_sub * (1._rt - 1.e-12))));

    // Hold the count through small changes (meridional sizing only, so the
    // other paths stay bit-identical to their previous behavior). The speeds
    // move slightly every step, and acting on that would rescale the
    // deposition cadence for no benefit. Nothing is skipped: the maxima are
    // still measured every step, so a genuine excursion is caught when it
    // happens; only the reaction to a small change is suppressed.
    if (meridional_used && m_n_subcycles_last_reported > 0 &&
        m_v_ref_hysteresis > 0._rt &&
        std::abs(n_implied - m_n_subcycles) <=
            m_v_ref_hysteresis * static_cast<amrex::Real>(m_n_subcycles))
    {
        // Keep m_n_subcycles; make the subcycles uniform over the step so the
        // retained count still covers dt exactly.
        m_dt_subcycle = dt / static_cast<amrex::Real>(m_n_subcycles);
    }
    else
    {
        m_n_subcycles = n_implied;
        m_dt_subcycle = dt_sub;
    }

    if (!m_boot_printed)
    {
        amrex::Print() << "[subcycle] " << species_name << ": "
                       << m_n_subcycles << " subcycles/step (dt_sub = "
                       << m_dt_subcycle << " s, |B|max = " << bmax
                       << " T, cyclotron CFL " << m_cfl_cyclotron
                       << ", grid CFL " << m_cfl_grid
                       << ", v_ref = " << v_ref
                       << " m/s (" << (measured ? "measured" : "fixed")
                       << (measured
                           ? (meridional_used ? ", meridional" : ", full speed")
                           : "")
                       << "), cap " << m_max_subcycles
                       << (measured && m_cap_is_error != 0 ? " (error)" : "")
                       << ", orbit-averaged deposition "
                       << (m_do_orbit_averaged_deposition ? "on" : "off")
                       << ")\n";
        m_boot_printed = true;
    }
    else if (measured && m_n_subcycles != m_n_subcycles_last_reported)
    {
        // Throttled: only when the settled count actually moves, and only
        // once it has moved by more than a small fraction, so that a count
        // hunting between neighbouring values stays quiet.
        const auto n_last = static_cast<amrex::Real>(m_n_subcycles_last_reported);
        const auto n_now = static_cast<amrex::Real>(m_n_subcycles);
        if (m_n_subcycles_last_reported < 0 ||
            std::abs(n_now - n_last) > 0.1_rt * amrex::max(n_last, 1._rt))
        {
            const amrex::Real dt_grid = (v_ref > 0._rt)
                ? m_cfl_grid * dx_min / v_ref : dt;
            amrex::Print() << "[subcycle] " << species_name
                           << ": count " << m_n_subcycles_last_reported
                           << " -> " << m_n_subcycles
                           << " (v_ref = " << v_ref << " m/s";
            if (meridional_used)
            {
                amrex::Print() << ", meridional " << speeds.meridional
                               << ", |v_theta| " << speeds.azimuthal
                               << ", combined " << speeds.combined
                               << ", full " << speeds.full
                               << (speeds.combined > speeds.full
                                   ? " (clamped to full)" : "");
            }
            amrex::Print() << ", binding leg "
                           << (dt_grid <= m_dt_subcycle*(1._rt + 1.e-9) ? "grid" : "cyclotron")
                           << ")\n";
            m_n_subcycles_last_reported = m_n_subcycles;
        }
    }
    if (m_n_subcycles_last_reported < 0) { m_n_subcycles_last_reported = m_n_subcycles; }
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

SubcycledParticleContainer::MaxSpeeds
SubcycledParticleContainer::MaxParticleSpeedsByComponent ()
{
    MaxSpeeds s;
    if (TotalNumberOfParticles(true, false) == 0) { return s; }

    using PType = typename WarpXParticleContainer::SuperParticleType;
    const amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    const auto margin = static_cast<amrex::ParticleReal>(m_v_theta_margin);

    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                     amrex::ReduceOpMax, amrex::ReduceOpMax> reduce_ops;
    auto r = amrex::ParticleReduce<
        amrex::ReduceData<amrex::ParticleReal, amrex::ParticleReal,
                          amrex::ParticleReal, amrex::ParticleReal>>(
        *this,
        [=] AMREX_GPU_DEVICE (const PType& p) noexcept
            -> amrex::GpuTuple<amrex::ParticleReal, amrex::ParticleReal,
                               amrex::ParticleReal, amrex::ParticleReal>
        {
            const amrex::ParticleReal ux = p.rdata(PIdx::ux);
            const amrex::ParticleReal uy = p.rdata(PIdx::uy);
            const amrex::ParticleReal uz = p.rdata(PIdx::uz);
            const amrex::ParticleReal usq = ux*ux + uy*uy + uz*uz;
            // proper velocity u = gamma v  =>  v = u / sqrt(1 + u^2/c^2)
            const amrex::ParticleReal inv_gamma =
                1._prt/std::sqrt(1._prt + usq*inv_c2);
            const amrex::ParticleReal vfull = std::sqrt(usq)*inv_gamma;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // The momenta are lab Cartesian components; theta is the stored
            // azimuthal position angle, so the rotation is exact and has no
            // r -> 0 singularity.
            const amrex::ParticleReal theta = p.rdata(PIdx::theta);
            const amrex::ParticleReal ct = std::cos(theta);
            const amrex::ParticleReal st = std::sin(theta);
            const amrex::ParticleReal vr = ( ux*ct + uy*st)*inv_gamma;
            const amrex::ParticleReal vt = (-ux*st + uy*ct)*inv_gamma;
            const amrex::ParticleReal vz = uz*inv_gamma;
            const amrex::ParticleReal vmer = std::sqrt(vr*vr + vz*vz);
            const amrex::ParticleReal vaz = std::abs(vt);
#else
            const amrex::ParticleReal vmer = vfull;
            const amrex::ParticleReal vaz = 0._prt;
#endif
            // Combined PER PARTICLE: summing the separate maxima would mix
            // two different particles and can exceed every particle's speed.
            return {vfull, vmer, vaz, vmer + margin*vaz};
        },
        reduce_ops);

    auto vfull = amrex::get<0>(r);
    auto vmer  = amrex::get<1>(r);
    auto vaz   = amrex::get<2>(r);
    auto vcomb = amrex::get<3>(r);
    amrex::ParallelDescriptor::ReduceRealMax(vfull);
    amrex::ParallelDescriptor::ReduceRealMax(vmer);
    amrex::ParallelDescriptor::ReduceRealMax(vaz);
    amrex::ParallelDescriptor::ReduceRealMax(vcomb);

    s.full = amrex::max(static_cast<amrex::Real>(vfull), 0._rt);
    s.meridional = amrex::max(static_cast<amrex::Real>(vmer), 0._rt);
    s.azimuthal = amrex::max(static_cast<amrex::Real>(vaz), 0._rt);
    s.combined = amrex::max(static_cast<amrex::Real>(vcomb), 0._rt);
    return s;
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

        // Deposit BEFORE any boundary handling (as in the main PIC loop):
        // charge-conserving (Esirkepov) deposition reconstructs the segment
        // of this subcycle as [x - v*dt_sub, x], which is only valid while
        // the pushed positions are unmodified — reflections or periodic
        // wraps must come after. Out-of-box positions deposit through the
        // guard cells and are folded back by the caller's guard-cell sum.
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

        // Boundary handling every subcycle: absorbed particles stop
        // contributing to the averaged moments from the next subcycle on,
        // and wall-loss records carry the sub-step time stamp.
        HandleBoundariesSubcycle(t_sub + dt_sub, dt_sub);

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

    // Raw fallback (bootstrap deposit before the first subcycled Evolve, or
    // orbit-averaged deposition disabled): a subcycled species can cross many
    // cells per global step, so honoring a global-step relative_time here
    // would shift the deposit positions beyond the guard cells (out-of-bounds
    // writes) while linearly back-extrapolating a strongly curved orbit.
    // Deposit at the instantaneous positions instead.
    amrex::ignore_unused(relative_time);
    WarpXParticleContainer::DepositCurrent(J, dt, 0._rt);
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
