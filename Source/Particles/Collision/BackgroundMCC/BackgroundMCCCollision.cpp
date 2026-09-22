/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BackgroundMCCCollision.H"

#include "ImpactIonization.H"
#include "MCCBackgroundDensity.H"
#include "MCCBackgroundField.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/BinaryCollision/TwoProductUtil.H"
#include "Particles/ParticleCreation/FilterCopyTransform.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <ablastr/particles/DepositCharge.H>

#include <AMReX_Print.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

BackgroundMCCCollision::BackgroundMCCCollision (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
                                     "Background MCC must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);

    amrex::ParticleReal background_density = 0;
    if (utils::parser::queryWithParser(pp_collision_name, "background_density", background_density)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (background_density > 0),
            "The background density must be greater than 0.");
        m_background_density_expression = std::to_string(background_density);
        m_background_density_parser =
            utils::parser::makeParser(
                m_background_density_expression, {"x", "y", "z", "t"});
    }
    else {
        utils::parser::Store_parserString(pp_collision_name, "background_density(x,y,z,t)",
                                          m_background_density_expression);
        m_background_density_parser =
            utils::parser::makeParser(m_background_density_expression, {"x", "y", "z", "t"});
    }

    amrex::ParticleReal background_temperature;
    if (utils::parser::queryWithParser(pp_collision_name, "background_temperature", background_temperature)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (background_temperature >= 0), "The background temperature must be positive."
        );
        m_background_temperature_parser =
            utils::parser::makeParser(std::to_string(background_temperature), {"x", "y", "z", "t"});
    }
    else {
        std::string background_temperature_str;
        utils::parser::Store_parserString(pp_collision_name, "background_temperature(x,y,z,t)", background_temperature_str);
        m_background_temperature_parser =
            utils::parser::makeParser(background_temperature_str, {"x", "y", "z", "t"});
    }

    // compile parsers for background density and temperature
    m_background_density_func = m_background_density_parser.compile<4>();
    m_background_temperature_func = m_background_temperature_parser.compile<4>();

    utils::parser::queryWithParser(
        pp_collision_name, "max_background_density", m_max_background_density);
    // if the background density is constant we can use that number to calculate
    // the maximum collision probability, if `max_background_density` was not
    // specified
    if (m_max_background_density == 0 && background_density != 0) {
        m_max_background_density = background_density;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_max_background_density > 0),
        "The maximum background density must be greater than 0."
    );

    // Optionally carry the background on the mesh so that ionization can
    // deplete it, rather than treating it as an inexhaustible reservoir.
    // Note that the null-collision majorant stays valid either way: it is
    // built from max_background_density once, and depletion only ever
    // decreases the density below it.
    pp_collision_name.query("deplete_background", m_deplete_background);
    if (m_deplete_background)
    {
#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_1D_Z)
        WARPX_ABORT_WITH_MESSAGE(
            collision_name + ".deplete_background is supported in Cartesian geometry "
            "only. The cylindrical and spherical geometries need the radial volume "
            "scaling that is applied to rho, which is not implemented for the "
            "background density.");
#else
        // Collisions acting on one physical gas must name it identically in
        // order to share a single field, so that the gas is not consumed once
        // per collision. The default leaves each collision independent.
        m_background_name = collision_name;
        pp_collision_name.query("background_name", m_background_name);

        // The same order is used to gather the density and to deposit its
        // depletion; matching them is what makes the pair conserve.
        m_background_shape = WarpX::nox;
        utils::parser::queryWithParser(
            pp_collision_name, "background_shape", m_background_shape);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (m_background_shape >= 1) && (m_background_shape <= 4),
            "background_shape must be between 1 and 4.");
#endif
    }

    // if the neutral mass is specified use it, but if ionization is
    // included the mass of the secondary species of that interaction
    // will be used. If no neutral mass is specified and ionization is not
    // included the mass of the colliding species will be used
    m_background_mass = -1;
    utils::parser::queryWithParser(
        pp_collision_name, "background_mass", m_background_mass);

    // Parse the list of scattering processes (these could be elastic,
    // excitation, charge_exchange, etc.) and create a vector of
    // ScatteringProcess objects from each scattering process name.
    amrex::Vector<ScatteringProcess> scattering_processes = BinaryCollisionUtils::parse_scattering_processes(collision_name);

    for (auto& process : scattering_processes) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::INVALID,
                                         "Cannot add an unknown scattering process type");

        // if the scattering process is ionization get the secondary species
        // only one ionization process is supported, the vector
        // m_ionization_processes is only used to make it simple to calculate
        // the maximum collision frequency with the same function used for
        // particle conserving processes
        if (process.type() == ScatteringProcessType::IONIZATION) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!ionization_flag,
                                             "Background MCC only supports a single ionization process");
            ionization_flag = true;

            std::string secondary_species;
            pp_collision_name.get("ionization_species", secondary_species);
            m_species_names.push_back(secondary_species);

            // How the energy left after paying the ionization cost is shared
            // between the incident electron and the one it ejects. The
            // default splits it equally, which is what MCC has always done.
            std::string energy_sharing = "equal";
            pp_collision_name.query("ionization_energy_sharing", energy_sharing);
            if (energy_sharing == "opal") {
                utils::parser::getWithParser(
                    pp_collision_name, "ionization_opal_w", m_ionization_opal_w);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (m_ionization_opal_w > 0),
                    collision_name + ".ionization_opal_w must be greater than 0."
                );
            } else {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    (energy_sharing == "equal"),
                    collision_name + ".ionization_energy_sharing must be either "
                    "'equal' or 'opal'."
                );
            }
            // Printed so that a run's log shows which model was actually
            // read: a build without this parameter ignores it silently.
            amrex::Print() << "  " << collision_name
                           << " ionization energy sharing: " << energy_sharing;
            if (energy_sharing == "opal") {
                amrex::Print() << " (w = " << m_ionization_opal_w << " eV)";
            }
            amrex::Print() << "\n";

            m_ionization_processes.push_back(std::move(process));
        } else {
            m_scattering_processes.push_back(std::move(process));
        }
    }

#ifdef AMREX_USE_GPU
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_ionization_processes_exe;
    for (auto const& p : m_scattering_processes) {
        h_scattering_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_ionization_processes) {
        h_ionization_processes_exe.push_back(p.executor());
    }
    m_scattering_processes_exe.resize(h_scattering_processes_exe.size());
    m_ionization_processes_exe.resize(h_ionization_processes_exe.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
                          h_scattering_processes_exe.end(), m_scattering_processes_exe.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_ionization_processes_exe.begin(),
                          h_ionization_processes_exe.end(), m_ionization_processes_exe.begin());
    amrex::Gpu::streamSynchronize();
#else
    for (auto const& p : m_scattering_processes) {
        m_scattering_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_ionization_processes) {
        m_ionization_processes_exe.push_back(p.executor());
    }
#endif
}

amrex::Vector<DepletableBackgroundSpec>
BackgroundMCCCollision::getDepletableBackgrounds () const
{
    if (!m_deplete_background) { return {}; }

    DepletableBackgroundSpec background;
    background.m_background_name = m_background_name;
    background.m_density_expression = m_background_density_expression;
    background.m_density_func = m_background_density_func;
    background.m_shape = m_background_shape;

    return {background};
}

/** Calculate the maximum collision frequency, from the reduced mass of the
 *  projectile and background pair, using a fixed energy grid that
 *  ranges from 1e-4 to 5000 eV in 0.2 eV increments
 */
amrex::ParticleReal
BackgroundMCCCollision::get_nu_max(amrex::Vector<ScatteringProcess> const& mcc_processes) const
{
    using namespace amrex::literals;
    amrex::ParticleReal nu, nu_max = 0.0;
    amrex::ParticleReal E_start = 1e-4_prt;
    amrex::ParticleReal E_end = 5000._prt;
    amrex::ParticleReal E_step = 0.2_prt;

    // set the energy limits and step size for calculating nu_max based
    // on the given cross-section inputs
    for (const auto &process : mcc_processes) {
        auto energy_lo = process.getMinEnergyInput();
        E_start = (energy_lo < E_start) ? energy_lo : E_start;
        auto energy_hi = process.getMaxEnergyInput();
        E_end = (energy_hi > E_end) ? energy_hi : E_end;
        auto energy_step = process.getEnergyInputStep();
        E_step = (energy_step < E_step) ? energy_step : E_step;
    }

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_background_mass > 0.0_prt,
        "BackgroundMCC: the background mass must be set before nu_max is computed");
    const amrex::ParticleReal reduced_mass =
        m_mass1 * m_background_mass / (m_mass1 + m_background_mass);

    amrex::ParticleReal E = E_start;
    while(E < E_end){
        amrex::ParticleReal sigma_E = 0.0;

        // loop through all collision pathways
        for (const auto &scattering_process : mcc_processes) {
            // get collision cross-section
            sigma_E += scattering_process.getCrossSection(E);
        }

        // calculate collision frequency. E is the collision energy the cross
        // sections are tabulated against, which getCollisionEnergy() computes
        // in the center-of-mass frame, mu*v_rel^2/2, so v_rel follows from
        // the reduced mass. Using the projectile mass instead underestimates
        // nu_max by sqrt(1 + m1/M): negligible for electrons, but sqrt(2) for
        // ions on a gas of their own mass, whose rate is then capped wherever
        // the true frequency exceeds that underestimate.
        nu = (
              m_max_background_density
              * std::sqrt(2.0_prt / reduced_mass * PhysConst::q_e)
              * sigma_E * std::sqrt(E)
              );
        nu_max = std::max(nu_max, nu);
        E+=E_step;
    }
    return nu_max;
}

void
BackgroundMCCCollision::doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doCollisions()");
    using namespace amrex::literals;

    auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    // this is a very ugly hack to have species2 be a reference and be
    // defined in the scope of doCollisions
    auto& species2 = (
                      (m_species_names.size() == 2) ?
                      mypc->GetParticleContainerFromName(m_species_names[1]) :
                      mypc->GetParticleContainerFromName(m_species_names[0])
                      );

    if (!init_flag) {
        m_mass1 = species1.getMass();

        // Resolve the background mass first: get_nu_max needs it for the
        // reduced mass.
        // if an ionization process is included the secondary species mass
        // is taken as the background mass
        if (ionization_flag) {
            m_background_mass = species2.getMass();
        }
        // if no neutral species mass was specified and ionization is not
        // included assume that the collisions will be with neutrals of the
        // same mass as the colliding species (as in ion-neutral collisions)
        else if (m_background_mass == -1) {
            m_background_mass = species1.getMass();
        }

        // calculate maximum collision frequency without ionization
        m_nu_max = get_nu_max(m_scattering_processes);

        // calculate total collision probability
        auto coll_n = m_nu_max * dt;
        m_total_collision_prob = 1.0_prt - std::exp(-coll_n);

        // dt has to be small enough that a linear expansion of the collision
        // probability is sufficiently accurately, otherwise the MCC results
        // will be very heavily affected by small changes in the timestep
        if (coll_n > 0.1_prt) {
            ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                     "dt is too large to ensure accurate MCC results , coll_n: " +
                      std::to_string(coll_n) + " is > 0.1 and collision probability is = " +
                      std::to_string(m_total_collision_prob) + "\n");
        }

        if (ionization_flag) {
            // calculate maximum collision frequency for ionization
            m_nu_max_ioniz = get_nu_max(m_ionization_processes);

            // calculate total ionization probability
            auto coll_n_ioniz = m_nu_max_ioniz * dt;
            m_total_collision_prob_ioniz = 1.0_prt - std::exp(-coll_n_ioniz);

            if (coll_n_ioniz > 0.1_prt) {
                ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                         "dt is too large to ensure accurate MCC ionization , coll_n_ionization: " +
                          std::to_string(coll_n_ioniz) + " is > 0.1 and ionization probability is = " +
                          std::to_string(m_total_collision_prob_ioniz) + "\n");
            }
        }

        amrex::Print() << Utils::TextMsg::Info(
            "Setting up Monte-Carlo collisions for " + m_species_names[0] + " with:\n"
            + "     total non-ionization collision probability: "
            + std::to_string(m_total_collision_prob)
            + "\n     total ionization collision probability: "
            + std::to_string(m_total_collision_prob_ioniz)
        );

        init_flag = true;
    }

    // Loop over refinement levels
    auto const flvl = species1.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // This step's depletion is accumulated separately and applied once
        // every tile has deposited into it; see applyDepletion.
        if (m_deplete_background && ionization_flag) {
            WarpX::GetInstance().m_fields.get(
                MCCBackgroundField::deltaFieldName(m_background_name), lev
            )->setVal(0.0_prt);
        }

        // firstly loop over particles box by box and do all particle conserving
        // scattering
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            doBackgroundCollisionsWithinTile(
                pti, getBackgroundDensity(pti, lev, cur_time), cur_time);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
            }
        }

        // secondly perform ionization through the SmartCopyFactory if needed
        if (ionization_flag) {
            doBackgroundIonization(lev, cost, species1, species2, cur_time);

            if (m_deplete_background) { applyDepletion(lev); }
        }
    }
}


void BackgroundMCCCollision::doBackgroundCollisionsWithinTile
( WarpXParIter& pti, MCCBackgroundDensity const& get_n_a, amrex::Real t )
{
    using namespace amrex::literals;

    // So that CUDA code gets its intrinsic, not the host-only C++ library version
    using std::sqrt;

    // get particle count
    const long np = pti.numParticles();

    // the temperature stays an analytic function of space and time
    auto T_a_func = m_background_temperature_func;

    // get collision parameters
    auto *scattering_processes = m_scattering_processes_exe.data();
    auto const process_count  = static_cast<int>(m_scattering_processes_exe.size());

    auto const total_collision_prob = m_total_collision_prob;
    auto const nu_max = m_nu_max;

    // store projectile and target masses
    auto const m = m_mass1;
    auto const M = m_background_mass;

    // we need particle positions in order to calculate the local density
    // and temperature
    auto GetPosition = GetParticlePosition<PIdx>(pti);

    // get Struct-Of-Array particle data, also called attribs
    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

    amrex::ParallelForRNG(np,
                          [=] AMREX_GPU_HOST_DEVICE (long ip, amrex::RandomEngine const& engine)
                          {
                              // determine if this particle should collide
                              if (amrex::Random(engine) > total_collision_prob) { return; }

                              // The background density and temperature parsers take Cartesian
                              // coordinates as arguments, in all geometries.
                              amrex::ParticleReal x, y, z;
                              GetPosition(ip, x, y, z);

                              const amrex::ParticleReal n_a = get_n_a(x, y, z);
                              const amrex::ParticleReal T_a = T_a_func(x, y, z, t);

                              amrex::ParticleReal v_coll, v_coll2, sigma_E, nu_i = 0;
                              double gamma, E_coll;
                              amrex::ParticleReal ua_x, ua_y, ua_z, vx, vy, vz;
                              const amrex::ParticleReal col_select = amrex::Random(engine);

                              // get velocities of gas particles from a Maxwellian distribution
                              auto const vel_std = sqrt(PhysConst::kb * T_a / M);
                              ua_x = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                              ua_y = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                              ua_z = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);

                              // we assume the target particle is not relativistic (in
                              // the lab frame) and therefore we can transform the projectile
                              // velocity to a frame in which the target is stationary with
                              // a simple Galilean boost
                              // not doing the full Lorentz boost here saves us computation
                              // since most particles will not actually collide
                              vx = ux[ip] - ua_x;
                              vy = uy[ip] - ua_y;
                              vz = uz[ip] - ua_z;
                              v_coll2 = (vx*vx + vy*vy + vz*vz);
                              v_coll = std::sqrt(v_coll2);

                              // calculate the collision energy in eV
                              ParticleUtils::getCollisionEnergy(v_coll2, m, M, gamma, E_coll);

                              // loop through all collision pathways
                              for (int i = 0; i < process_count; i++) {
                                  auto const& scattering_process = *(scattering_processes + i);

                                  // get collision cross-section
                                  sigma_E = scattering_process.getCrossSection(static_cast<amrex::ParticleReal>(E_coll));

                                  // calculate normalized collision frequency
                                  nu_i += n_a * sigma_E * v_coll / nu_max;

                                  // check if this collision should be performed
                                  if (col_select > nu_i) { continue; }

                                  // At this point the given particle has been chosen for a
                                  // collision with a background-gas particle of velocity
                                  // (ua_x, ua_y, ua_z). Compute the post-collision momentum of
                                  // the projectile using conservation of energy and momentum.
                                  // The angular distribution in the center-of-mass frame is set
                                  // by the process's scattering angle model, and any inelastic
                                  // energy loss is passed as the (released) reaction energy.
                                  // The background particle is treated as a reservoir: its recoil
                                  // is computed as the second product but discarded.
                                  amrex::ParticleReal u1x_out, u1y_out, u1z_out;
                                  amrex::ParticleReal u2x_out, u2y_out, u2z_out;
                                  TwoProductComputeProductMomenta(
                                      ux[ip], uy[ip], uz[ip], m,
                                      ua_x, ua_y, ua_z, M,
                                      u1x_out, u1y_out, u1z_out, m,
                                      u2x_out, u2y_out, u2z_out, M,
                                      -scattering_process.m_energy_penalty*PhysConst::q_e,
                                      // TwoProductComputeProductMomenta expects the *released* energy here, hence
                                      // the negative sign; the energy penalty is also converted from eV to Joules.
                                      scattering_process.m_scattering_angle_model,
                                      engine);

                                  // update projectile velocity with new components in labframe
                                  // (the background-gas recoil u2*_out is discarded)
                                  ux[ip] = u1x_out;
                                  uy[ip] = u1y_out;
                                  uz[ip] = u1z_out;
                                  break;
                              }
                          }
                          );
}


void BackgroundMCCCollision::doBackgroundIonization
( int lev, amrex::LayoutData<amrex::Real>* cost,
  WarpXParticleContainer& species1, WarpXParticleContainer& species2, amrex::Real t)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doBackgroundIonization()");
    using namespace amrex::literals;

    const SmartCopyFactory copy_factory_elec(species1, species1);
    const SmartCopyFactory copy_factory_ion(species1, species2);
    const auto CopyElec = copy_factory_elec.getSmartCopy();
    const auto CopyIon = copy_factory_ion.getSmartCopy();

    const amrex::ParticleReal sqrt_kb_m = std::sqrt(PhysConst::kb / m_background_mass);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto& elec_tile = species1.ParticlesAt(lev, pti);
        auto& ion_tile = species2.ParticlesAt(lev, pti);

        const auto np_elec = elec_tile.numParticles();
        const auto np_ion = ion_tile.numParticles();

        // The density accessor, and so the filter that uses it, is per-tile
        // once the background lives on the mesh.
        const auto Filter = ImpactIonizationFilterFunc(
                                                       m_ionization_processes[0],
                                                       m_mass1, m_total_collision_prob_ioniz,
                                                       m_nu_max_ioniz,
                                                       getBackgroundDensity(pti, lev, t)
                                                       );

        auto Transform = ImpactIonizationTransformFunc(
                                                       m_ionization_processes[0].getEnergyPenalty(),
                                                       m_ionization_opal_w,
                                                       m_mass1, sqrt_kb_m, m_background_temperature_func, t
                                                       );

        const auto num_added = filterCopyTransformParticles<1>(species1, species2,
                                                               elec_tile, ion_tile, elec_tile, np_elec, np_ion,
                                                               Filter, CopyElec, CopyIon, Transform
                                                               );

        setNewParticleIDs(elec_tile, np_elec, num_added);
        setNewParticleIDs(ion_tile, np_ion, num_added);

        if (m_deplete_background)
        {
            auto& warpx = WarpX::GetInstance();
            auto * const dn_mf = warpx.m_fields.get(
                MCCBackgroundField::deltaFieldName(m_background_name), lev);

            amrex::Box tilebox = pti.tilebox();
            tilebox.grow(warpx.get_ng_depos_rho());

            // Each ionization event appends one electron and one ion at the
            // source electron's position, so the new electrons -- which land in
            // this same tile, the one pti indexes -- carry exactly the weight
            // of neutrals consumed. Depositing them with unit negative charge
            // gives the number-density decrement, -w/dV, in m^-3, using the
            // same shape factor the density is gathered with.
            amrex::FArrayBox local_dn;
            ablastr::particles::deposit_charge<WarpXParticleContainer>(
                pti, pti.GetAttribs(PIdx::w), /*charge=*/-1.0_prt,
                /*ion_lev=*/nullptr, dn_mf, local_dn, m_background_shape,
                WarpX::InvCellSize(lev),
                WarpX::LowerCorner(tilebox, lev, 0._rt),
                /*n_rz_azimuthal_modes=*/0,
                warpx.get_ng_depos_rho(), lev, amrex::IntVect(1),
                /*offset=*/np_elec, /*np_to_deposit=*/num_added);
        }

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
        }
    }
}


MCCBackgroundDensity
BackgroundMCCCollision::getBackgroundDensity (
    WarpXParIter& pti, int const lev, amrex::Real const t) const
{
    using namespace amrex::literals;

    if (!m_deplete_background) {
        return MCCBackgroundDensity(m_background_density_func, t);
    }

    auto& warpx = WarpX::GetInstance();
    auto const * const n_mf = warpx.m_fields.get(
        MCCBackgroundField::fieldName(m_background_name), lev);

    // Use the box the depletion deposit uses, so that a gather and a deposit at
    // the same position touch the same cells with the same weights.
    amrex::Box tilebox = pti.tilebox();
    tilebox.grow(warpx.get_ng_depos_rho());

    return MCCBackgroundDensity(
        n_mf->const_array(pti), n_mf->ixType().toIntVect(),
        WarpX::InvCellSize(lev),
        WarpX::LowerCorner(tilebox, lev, 0._rt),
        amrex::lbound(tilebox), m_background_shape);
}


void BackgroundMCCCollision::applyDepletion (int const lev)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::applyDepletion()");
    using namespace amrex::literals;

    auto& warpx = WarpX::GetInstance();
    auto * const n_mf = warpx.m_fields.get(
        MCCBackgroundField::fieldName(m_background_name), lev);
    auto * const dn_mf = warpx.m_fields.get(
        MCCBackgroundField::deltaFieldName(m_background_name), lev);

    // Fold the deposit's guard-cell contributions back into the valid region,
    // across tiles and across processes, exactly as rho is treated.
    ablastr::utils::communication::SumBoundary(
        *dn_mf, 0, dn_mf->nComp(), dn_mf->nGrowVect(), dn_mf->nGrowVect(),
        WarpX::do_single_precision_comms, warpx.Geom(lev).periodicity());

    // Reflect back in the part of the stencil that fell outside a domain
    // boundary. Without this the depletion is systematically under-counted in
    // the boundary layer, and more so at higher shape order.
    warpx.ApplyRhofieldBoundary(lev, dn_mf, PatchType::fine);

    // Only now is the decrement complete, so only now may it be applied. The
    // clamp has to come after the fold as well: clamping partial per-tile
    // contributions would destroy the conservation the deposit provides.
    for (amrex::MFIter mfi(*n_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& n_arr = n_mf->array(mfi);
        auto const& dn_arr = dn_mf->const_array(mfi);
        const amrex::Box& bx = mfi.growntilebox();

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // dn is negative; the gas cannot go below empty.
                n_arr(i,j,k) = amrex::max(n_arr(i,j,k) + dn_arr(i,j,k), 0.0_rt);
            });
    }
}
