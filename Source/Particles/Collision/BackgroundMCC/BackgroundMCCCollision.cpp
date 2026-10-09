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

#include <cmath>
#include <string>
#include <type_traits>

namespace
{
    /** Transform for a background reaction (see BackgroundMCCCollision.H).
     *  The filter pass stored, per accepted particle, the velocity of the
     *  molecule it met and the channel it drew, so the products come from the
     *  collision that was accepted. */
    struct BackgroundReactionTransformFunc
    {
        amrex::ParticleReal m_mass1;
        amrex::ParticleReal m_background_mass;
        amrex::ParticleReal m_product_mass1;
        amrex::ParticleReal m_product_mass2;
        int m_mode;
        ScatteringProcess::Executor const* m_exe;
        /** Fragment kinetic energy release per channel, eV */
        amrex::ParticleReal const* m_fragment_energy;
        int const* m_channel;
        amrex::ParticleReal const* m_uax;
        amrex::ParticleReal const* m_uay;
        amrex::ParticleReal const* m_uaz;
        /** Third-product velocities, written here per source particle when
         *  the reaction makes one (nullptr otherwise), read by the copy pass
         *  that creates it: the ejected electron of a target fragmentation,
         *  or the recoiling molecule of a projectile fragmentation */
        amrex::ParticleReal* m_ejx = nullptr;
        amrex::ParticleReal* m_ejy = nullptr;
        amrex::ParticleReal* m_ejz = nullptr;
        amrex::ParticleReal const* m_opal_w = nullptr;
        /** The third product's mass (0 for none) */
        amrex::ParticleReal m_third_mass = 0;

        /** The two products' velocities (u = gamma v); updates or removes the
         *  projectile. */
        template <typename SrcData>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void compute (SrcData& src, int const i_src, amrex::RandomEngine const& engine,
                      amrex::ParticleReal* uA, amrex::ParticleReal* uB) const noexcept
        {
            using namespace amrex::literals;
            using std::sqrt;
            // Read the projectile before writing anything: a product may share
            // the projectile's tile.
            const amrex::ParticleReal ux = src.m_rdata[PIdx::ux][i_src];
            const amrex::ParticleReal uy = src.m_rdata[PIdx::uy][i_src];
            const amrex::ParticleReal uz = src.m_rdata[PIdx::uz][i_src];
            const amrex::ParticleReal uax = m_uax[i_src];
            const amrex::ParticleReal uay = m_uay[i_src];
            const amrex::ParticleReal uaz = m_uaz[i_src];
            auto const& ex = m_exe[m_channel[i_src]];

            // The anisotropy is tabulated against the collision energy.
            const amrex::ParticleReal vx = ux - uax;
            const amrex::ParticleReal vy = uy - uay;
            const amrex::ParticleReal vz = uz - uaz;
            double gamma, E_coll;
            ParticleUtils::getCollisionEnergy(vx*vx + vy*vy + vz*vz, m_mass1,
                                              m_background_mass, gamma, E_coll);
            const amrex::ParticleReal xi =
                ex.getAnisotropy(static_cast<amrex::ParticleReal>(E_coll));
            const amrex::ParticleReal E_release = -ex.m_energy_penalty * PhysConst::q_e;

            if (m_mode == 0) {
                TwoProductComputeProductMomenta(
                    ux, uy, uz, m_mass1, uax, uay, uaz, m_background_mass,
                    uA[0], uA[1], uA[2], m_product_mass1,
                    uB[0], uB[1], uB[2], m_product_mass2,
                    E_release, ex.m_scattering_angle_model, engine, xi);
                src.m_idcpu[i_src] = amrex::ParticleIdCpus::Invalid;
                return;
            }

            // Scatter the pair as two bodies, paying the whole energy cost, then
            // break the fragmenting one about its post-collision velocity. A
            // recoiling molecule that becomes a particle may differ from the
            // background by an electron, which the projectile's fragments
            // then carry or lack: the two bodies take the products' masses,
            // so the relativistic kinematics conserve energy exactly.
            const bool recoils = (m_mode == 2 && m_ejx != nullptr);
            const amrex::ParticleReal m_out1 =
                recoils ? m_product_mass1 + m_product_mass2 : m_mass1;
            const amrex::ParticleReal m_out2 = recoils ? m_third_mass : m_background_mass;
            amrex::ParticleReal u1[3], u2[3];
            TwoProductComputeProductMomenta(
                ux, uy, uz, m_mass1, uax, uay, uaz, m_background_mass,
                u1[0], u1[1], u1[2], m_out1,
                u2[0], u2[1], u2[2], m_out2,
                E_release, ex.m_scattering_angle_model, engine, xi);
            const amrex::ParticleReal* uf = (m_mode == 1) ? u2 : u1;
            if (m_mode == 1 && m_ejx != nullptr) {
                // Share the projectile's energy after the cost with the
                // electron it ejects: equally, or by Opal's law truncated at
                // half (the ejected one is the slower), as ionization does.
                constexpr double c2 = PhysConst::c2_v<double>;
                const double mc2 = static_cast<double>(m_mass1) * c2;
                const double E_avail = Algorithms::KineticEnergy<double>(
                    u1[0], u1[1], u1[2], m_mass1);
                const double w = static_cast<double>(m_opal_w[m_channel[i_src]]) * PhysConst::q_e;
                const double E_ej = (w > 0.0)
                    ? w * std::tan(amrex::Random(engine) * std::atan(E_avail / (2.0*w)))
                    : 0.5 * E_avail;
                const double E_sc = E_avail - E_ej;
                const double u1_norm = std::sqrt(u1[0]*u1[0] + u1[1]*u1[1] + u1[2]*u1[2]);
                const double up_sc = std::sqrt(E_sc * (E_sc + 2.0*mc2) / c2) / m_mass1;
                const double up_ej = std::sqrt(E_ej * (E_ej + 2.0*mc2) / c2) / m_mass1;
                const double scale = (u1_norm > 0.0) ? up_sc / u1_norm : 0.0;
                src.m_rdata[PIdx::ux][i_src] = static_cast<amrex::ParticleReal>(u1[0] * scale);
                src.m_rdata[PIdx::uy][i_src] = static_cast<amrex::ParticleReal>(u1[1] * scale);
                src.m_rdata[PIdx::uz][i_src] = static_cast<amrex::ParticleReal>(u1[2] * scale);
                amrex::ParticleReal ex_, ey_, ez_;
                ParticleUtils::RandomizeVelocity(ex_, ey_, ez_,
                    static_cast<amrex::ParticleReal>(up_ej), engine);
                m_ejx[i_src] = ex_;
                m_ejy[i_src] = ey_;
                m_ejz[i_src] = ez_;
            } else if (m_mode == 1) {
                src.m_rdata[PIdx::ux][i_src] = u1[0];
                src.m_rdata[PIdx::uy][i_src] = u1[1];
                src.m_rdata[PIdx::uz][i_src] = u1[2];
            } else {
                src.m_idcpu[i_src] = amrex::ParticleIdCpus::Invalid;
                if (m_ejx != nullptr) {
                    m_ejx[i_src] = u2[0];
                    m_ejy[i_src] = u2[1];
                    m_ejz[i_src] = u2[2];
                }
            }
            // The fragments share the release back to back in the fragmenting
            // body's frame: equal and opposite momenta p, with p^2/(2 mu) = E.
            const amrex::ParticleReal mu_ab =
                m_product_mass1 * m_product_mass2 / (m_product_mass1 + m_product_mass2);
            const amrex::ParticleReal p_rel =
                sqrt(2.0_prt * mu_ab * m_fragment_energy[m_channel[i_src]] * PhysConst::q_e);
            amrex::ParticleReal nx, ny, nz;
            ParticleUtils::RandomizeVelocity(nx, ny, nz, 1.0_prt, engine);
            uA[0] = uf[0] + p_rel / m_product_mass1 * nx;
            uA[1] = uf[1] + p_rel / m_product_mass1 * ny;
            uA[2] = uf[2] + p_rel / m_product_mass1 * nz;
            uB[0] = uf[0] - p_rel / m_product_mass2 * nx;
            uB[1] = uf[1] - p_rel / m_product_mass2 * ny;
            uB[2] = uf[2] - p_rel / m_product_mass2 * nz;
        }

        /** Products in two species */
        template <typename DstData, typename SrcData>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (DstData& dst1, DstData& dst2, SrcData& src,
                         int const i_src, int const i_dst1, int const i_dst2,
                         amrex::RandomEngine const& engine) const noexcept
        {
            amrex::ParticleReal uA[3], uB[3];
            compute(src, i_src, engine, uA, uB);
            dst1.m_rdata[PIdx::ux][i_dst1] = uA[0];
            dst1.m_rdata[PIdx::uy][i_dst1] = uA[1];
            dst1.m_rdata[PIdx::uz][i_dst1] = uA[2];
            dst2.m_rdata[PIdx::ux][i_dst2] = uB[0];
            dst2.m_rdata[PIdx::uy][i_dst2] = uB[1];
            dst2.m_rdata[PIdx::uz][i_dst2] = uB[2];
        }

        /** Both products in one species, at i_dst and i_dst + 1 */
        template <typename DstData, typename SrcData>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (DstData& dst, SrcData& src, int const i_src, int const i_dst,
                         amrex::RandomEngine const& engine) const noexcept
        {
            amrex::ParticleReal uA[3], uB[3];
            compute(src, i_src, engine, uA, uB);
            dst.m_rdata[PIdx::ux][i_dst] = uA[0];
            dst.m_rdata[PIdx::uy][i_dst] = uA[1];
            dst.m_rdata[PIdx::uz][i_dst] = uA[2];
            dst.m_rdata[PIdx::ux][i_dst + 1] = uB[0];
            dst.m_rdata[PIdx::uy][i_dst + 1] = uB[1];
            dst.m_rdata[PIdx::uz][i_dst + 1] = uB[2];
        }
    };

    /** Sets the third products' velocities the reaction transform stored */
    struct EjectedElectronTransformFunc
    {
        amrex::ParticleReal const* m_ejx;
        amrex::ParticleReal const* m_ejy;
        amrex::ParticleReal const* m_ejz;

        template <typename DstData, typename SrcData>
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        void operator() (DstData& dst, SrcData& /*src*/, int const i_src, int const i_dst,
                         amrex::RandomEngine const& /*engine*/) const noexcept
        {
            dst.m_rdata[PIdx::ux][i_dst] = m_ejx[i_src];
            dst.m_rdata[PIdx::uy][i_dst] = m_ejy[i_src];
            dst.m_rdata[PIdx::uz][i_dst] = m_ejz[i_src];
        }
    };
}

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
    // The names, in the same order, for the per-process reaction inputs.
    amrex::Vector<std::string> process_names;
    pp_collision_name.queryarr("scattering_processes", process_names);

    for (int ip = 0; ip < static_cast<int>(scattering_processes.size()); ++ip) {
        auto& process = scattering_processes[ip];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::INVALID,
                                         "Cannot add an unknown scattering process type");

        // A two-product reaction that names its product species changes the
        // projectile's species, and an excitation or two-product process that
        // names fragment species breaks one partner in two: both get passes of
        // their own, which create the products. Without these keys a process
        // stays a particle conserving scattering, as it has always been here.
        std::string const& name = process_names[ip];
        const bool exchange =
            process.type() == ScatteringProcessType::TWOPRODUCT_REACTION &&
            pp_collision_name.contains((name + "_product_species").c_str());
        const bool fragmentation =
            (process.type() == ScatteringProcessType::TWOPRODUCT_REACTION ||
             process.type() == ScatteringProcessType::EXCITATION) &&
            pp_collision_name.contains((name + "_fragment_species").c_str());
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!(exchange && fragmentation),
            collision_name + "." + name + ": give either product_species or "
            "fragment_species, not both.");
        if (exchange || fragmentation) {
            int mode = ReactionExchange;
            amrex::Vector<std::string> products;
            amrex::ParticleReal fragment_energy = 0;
            if (exchange) {
                pp_collision_name.getarr((name + "_product_species").c_str(), products);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    products.size() == 2 && products[0] != products[1],
                    collision_name + "." + name + "_product_species must name two "
                    "different species.");
            } else {
                pp_collision_name.getarr((name + "_fragment_species").c_str(), products);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(products.size() == 2,
                    collision_name + "." + name + "_fragment_species must name two "
                    "species (the same one twice for identical fragments).");
                std::string fragmenting = "target";
                pp_collision_name.query((name + "_fragmenting").c_str(), fragmenting);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    fragmenting == "target" || fragmenting == "projectile",
                    collision_name + "." + name + "_fragmenting must be 'target' or "
                    "'projectile'.");
                mode = (fragmenting == "target") ? FragmentTarget : FragmentProjectile;
                utils::parser::queryWithParser(
                    pp_collision_name, (name + "_fragment_energy").c_str(), fragment_energy);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    fragment_energy >= 0 && fragment_energy <= process.getEnergyPenalty(),
                    collision_name + "." + name + "_fragment_energy must lie between 0 "
                    "and the process's energy cost, which pays for it.");
            }
            std::string ejected;
            amrex::ParticleReal opal_w = 0;
            if (fragmentation && mode == FragmentProjectile) {
                // The recoiling molecule leaves the background as a particle
                // (e.g. a charge transfer that breaks the projectile).
                pp_collision_name.query((name + "_recoil_species").c_str(), ejected);
            } else if (fragmentation) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !pp_collision_name.contains((name + "_recoil_species").c_str()),
                    collision_name + "." + name + "_recoil_species needs the projectile "
                    "to fragment.");
                pp_collision_name.query((name + "_ejected_species").c_str(), ejected);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ejected.empty() || mode == FragmentTarget,
                    collision_name + "." + name + "_ejected_species needs the target to "
                    "fragment.");
                utils::parser::queryWithParser(
                    pp_collision_name, (name + "_ejected_opal_w").c_str(), opal_w);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(opal_w >= 0,
                    collision_name + "." + name + "_ejected_opal_w must not be negative.");
            }
            int consumed = (mode == FragmentProjectile && ejected.empty()) ? 0 : 1;
            pp_collision_name.query((name + "_background_consumed").c_str(), consumed);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(consumed >= 0,
                collision_name + "." + name + "_background_consumed must not be negative.");
            const auto angle_model = process.scatteringAngleModel();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                angle_model != ScatteringAngleModel::Okhrimovskyy || mode != ReactionExchange,
                collision_name + "." + name + ": an exchange reaction's angle model must "
                "be isotropic, forward or backward.");

            // Fragmentation channels with the same products and mode share a pass.
            BackgroundReaction* group = nullptr;
            if (mode != ReactionExchange) {
                for (auto& r : m_reactions) {
                    if (r.m_mode == mode && r.m_background_consumed == consumed &&
                        r.m_product_species[0] == products[0] &&
                        r.m_product_species[1] == products[1] &&
                        r.m_ejected_species == ejected) { group = &r; }
                }
            }
            if (group == nullptr) {
                m_reactions.emplace_back();
                group = &m_reactions.back();
                group->m_mode = mode;
                group->m_product_species[0] = products[0];
                group->m_product_species[1] = products[1];
                group->m_background_consumed = consumed;
                group->m_ejected_species = ejected;
            }
            group->m_process.push_back(std::move(process));
            group->m_channel_names.push_back(name);
            group->m_fragment_energy_h.push_back(fragment_energy);
            group->m_opal_w_h.push_back(opal_w);
            continue;
        }

        // Ionization channels are sampled in their own pass. Several may be given
        // (`ionization` and `ionization_<name>`): they share the product species, the
        // energy sharing and the ejected electron's angle model, read once at the collision
        // level, while each keeps its own cross section, energy cost and incident angle
        // model. The pass picks each event's channel in proportion to its cross section.
        if (process.type() == ScatteringProcessType::IONIZATION) {
            if (!ionization_flag) {
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

                // The ejected electron's direction: isotropic (the default, which leaves the
                // random stream as it was), or the free kinematics of Magboltz, which ties it
                // to the incident electron's deflection.
                std::string secondary_angle = "isotropic";
                pp_collision_name.query("ionization_secondary_angle_model", secondary_angle);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    secondary_angle == "isotropic" || secondary_angle == "free_kinematics",
                    collision_name + ".ionization_secondary_angle_model must be either "
                    "'isotropic' or 'free_kinematics'."
                );
                m_ionization_secondary_free_kinematics = (secondary_angle == "free_kinematics");
                const bool incident_okh =
                    process.scatteringAngleModel() == ScatteringAngleModel::Okhrimovskyy;
                amrex::Print() << "  " << collision_name << " ionization angle models: incident "
                               << (incident_okh ? "okhrimovskyy" : "isotropic")
                               << ", ejected " << secondary_angle << "\n";
            }

            // The incident electron is deflected by the channel's angle model. Forward and
            // backward have no meaning for a three-body final state and were never applied
            // here, so refuse them rather than run isotropic under a name that says otherwise.
            const auto angle_model = process.scatteringAngleModel();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                angle_model == ScatteringAngleModel::Isotropic ||
                angle_model == ScatteringAngleModel::Okhrimovskyy,
                collision_name + ": the scattering_angle_model of an ionization process must "
                "be either 'isotropic' or 'okhrimovskyy'."
            );

            m_ionization_processes.push_back(std::move(process));
        } else {
            m_scattering_processes.push_back(std::move(process));
        }
    }
    if (m_ionization_processes.size() > 1) {
        amrex::Print() << "  " << collision_name << " ionization channels: "
                       << m_ionization_processes.size() << ", energy costs";
        for (auto const& p : m_ionization_processes) {
            amrex::Print() << " " << p.getEnergyPenalty();
        }
        amrex::Print() << " eV\n";
    }

    for (auto& r : m_reactions) {
        amrex::Gpu::HostVector<ScatteringProcess::Executor> h_exe;
        for (auto const& p : r.m_process) { h_exe.push_back(p.executor()); }
        r.m_exe.resize(h_exe.size());
        r.m_fragment_energy.resize(r.m_fragment_energy_h.size());
        r.m_opal_w.resize(r.m_opal_w_h.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r.m_opal_w_h.begin(),
                              r.m_opal_w_h.end(), r.m_opal_w.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_exe.begin(), h_exe.end(),
                              r.m_exe.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r.m_fragment_energy_h.begin(),
                              r.m_fragment_energy_h.end(), r.m_fragment_energy.begin());
    }
    amrex::Gpu::streamSynchronize();

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

/** Calculate the maximum collision frequency using a fixed energy grid that
 *  ranges from 1e-4 to 5000 eV in 0.2 eV increments
 *
 *  @param[in] mcc_processes the processes whose summed frequency is bounded
 *  @param[in] speed_mass the mass m with which the kernel's collision energy E
 *             and speed v are related, E = m*v^2/2
 */
amrex::ParticleReal
BackgroundMCCCollision::get_nu_max(amrex::Vector<ScatteringProcess> const& mcc_processes,
                                   amrex::ParticleReal speed_mass) const
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

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(speed_mass > 0.0_prt,
        "BackgroundMCC: nu_max needs a positive mass");

    amrex::ParticleReal E = E_start;
    while(E < E_end){
        amrex::ParticleReal sigma_E = 0.0;

        // loop through all collision pathways
        for (const auto &scattering_process : mcc_processes) {
            // get collision cross-section
            sigma_E += scattering_process.getCrossSection(E);
        }

        // calculate collision frequency
        nu = (
              m_max_background_density
              * std::sqrt(2.0_prt / speed_mass * PhysConst::q_e)
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
        // A given background_mass is used as given. Without it, an ionizing
        // collision takes the ion species' mass, and otherwise the
        // projectile's (as in ion-neutral collisions). Ionization used to
        // override a given mass with the ion's, which is one electron mass
        // light: harmless for the scattering, but a reaction that fragments
        // the molecule must balance against the molecule itself.
        if (m_background_mass == -1) {
            m_background_mass = ionization_flag ? species2.getMass() : species1.getMass();
        }

        // calculate maximum collision frequency without ionization. The
        // scattering kernel looks cross sections up at getCollisionEnergy(),
        // the center-of-mass energy mu*v_rel^2/2, and weights them by v_rel,
        // so the majorant must use the reduced mass. The projectile mass
        // underestimates it by sqrt(1 + m1/M): negligible for electrons, but
        // sqrt(2) for ions on a gas of their own mass, whose rate is then
        // capped wherever the true frequency exceeds that underestimate.
        const amrex::ParticleReal reduced_mass =
            m_mass1 * m_background_mass / (m_mass1 + m_background_mass);
        m_nu_max = get_nu_max(m_scattering_processes, reduced_mass);

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
            // calculate maximum collision frequency for ionization. The
            // ionization filter (ImpactIonizationFilterFunc) uses the
            // projectile's own lab-frame kinetic energy and speed, so here
            // the projectile mass is the consistent one.
            m_nu_max_ioniz = get_nu_max(m_ionization_processes, m_mass1);

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

        // Each reaction has its own majorant. Its filter looks the cross
        // section up at the centre-of-mass energy and weights it by the
        // relative speed, as the scattering kernel does, so the reduced mass.
        for (auto& r : m_reactions) {
            for (int k = 0; k < 2; ++k) {
                r.m_product_mass[k] =
                    mypc->GetParticleContainerFromName(r.m_product_species[k]).getMass();
            }
            // The kinematics are relativistic, so a rest-mass mismatch is
            // released as energy: one electron mass too many or too few is
            // 511 keV per event. The products balance the reactants in an
            // exchange, the molecule in a target fragmentation and the
            // projectile in a projectile fragmentation.
            // A third product is made from the rest: an ejected electron
            // leaves the molecule, and a recoiling molecule that becomes a
            // particle takes (or gives) the electron the projectile's
            // fragments lack (or carry).
            const double m_third = r.m_ejected_species.empty() ? 0.0
                : static_cast<double>(
                    mypc->GetParticleContainerFromName(r.m_ejected_species).getMass());
            r.m_third_mass = static_cast<amrex::ParticleReal>(m_third);
            // An ejected electron shares the projectile's energy as an equal:
            // the sharing is only written for an electron projectile.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                r.m_mode != FragmentTarget || r.m_ejected_species.empty() ||
                std::abs(m_third - static_cast<double>(m_mass1)) <= 1.0e-6 * m_third,
                "Background reaction " + r.m_channel_names[0] + ": ejected_species "
                "needs a projectile of the same mass (an electron ejecting an "
                "electron). For a heavy projectile, fragment the target into the "
                "ion and the electron instead.");
            const double m_in =
                (r.m_mode == ReactionExchange)
                    ? static_cast<double>(m_mass1) + static_cast<double>(m_background_mass)
                : (r.m_mode == FragmentTarget)
                    ? static_cast<double>(m_background_mass) - m_third
                : r.m_ejected_species.empty()
                    ? static_cast<double>(m_mass1)
                    : static_cast<double>(m_mass1) + static_cast<double>(m_background_mass)
                        - m_third;
            const double mass_defect_eV =
                (m_in - static_cast<double>(r.m_product_mass[0])
                      - static_cast<double>(r.m_product_mass[1]))
                * PhysConst::c2_v<double> / PhysConst::q_e_v<double>;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(mass_defect_eV) < 1.0e-3,
                "Background reaction " + r.m_channel_names[0] + ": the product masses "
                "differ from what they are made of by " + std::to_string(mass_defect_eV) +
                " eV of rest energy. Give the masses consistently (e.g. ions as atoms "
                "minus electrons) and binding energies through the process energies.");

            // One majorant over the summed channels, looked up at the centre-of-mass
            // energy and weighted by the relative speed: the reduced mass.
            const amrex::ParticleReal reduced_mass =
                m_mass1 * m_background_mass / (m_mass1 + m_background_mass);
            r.m_nu_max = get_nu_max(r.m_process, reduced_mass);
            auto const coll_n_r = r.m_nu_max * dt;
            r.m_total_collision_prob = 1.0_prt - std::exp(-coll_n_r);
            if (coll_n_r > 0.1_prt) {
                ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                         "dt is too large to ensure accurate MCC reaction " +
                         r.m_channel_names[0] + ", coll_n: " + std::to_string(coll_n_r) +
                         " is > 0.1\n");
            }
            std::string channels;
            for (std::size_t c = 0; c < r.m_channel_names.size(); ++c) {
                channels += " " + r.m_channel_names[c] + " ("
                    + std::to_string(r.m_process[c].getEnergyPenalty()) + " eV";
                if (r.m_mode != ReactionExchange) {
                    channels += ", fragments " + std::to_string(r.m_fragment_energy_h[c]) + " eV";
                }
                channels += ")";
            }
            const std::string what =
                (r.m_mode == ReactionExchange) ? m_species_names[0] + " + background -> "
                : (r.m_mode == FragmentTarget) ? "background broken into "
                                               : m_species_names[0] + " broken into ";
            const std::string third = r.m_ejected_species.empty() ? ""
                : (r.m_mode == FragmentTarget) ? " + ejected " + r.m_ejected_species
                                               : " + recoil " + r.m_ejected_species;
            amrex::Print() << Utils::TextMsg::Info(
                "Background reaction: " + what + r.m_product_species[0] + " + "
                + r.m_product_species[1] + third + "; background consumed "
                + std::to_string(r.m_background_consumed) + "; collision probability "
                + std::to_string(r.m_total_collision_prob) + "; channels:" + channels);
        }

        init_flag = true;
    }

    // Loop over refinement levels
    auto const flvl = species1.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // This step's depletion is accumulated separately and applied once
        // every tile has deposited into it; see applyDepletion.
        const bool depletes = m_deplete_background && (ionization_flag || !m_reactions.empty());
        if (depletes) {
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
        }

        // thirdly the species-changing reactions, one pass each
        for (int ir = 0; ir < static_cast<int>(m_reactions.size()); ++ir) {
            doBackgroundReaction(
                lev, cost, species1, ir,
                mypc->GetParticleContainerFromName(m_reactions[ir].m_product_species[0]),
                mypc->GetParticleContainerFromName(m_reactions[ir].m_product_species[1]),
                m_reactions[ir].m_ejected_species.empty() ? nullptr
                    : &mypc->GetParticleContainerFromName(m_reactions[ir].m_ejected_species),
                cur_time);
        }

        if (depletes) { applyDepletion(lev); }
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
                                  // The anisotropy is tabulated against the same collision
                                  // energy as the cross section, and is zero without a table.
                                  const amrex::ParticleReal xi = scattering_process.getAnisotropy(
                                      static_cast<amrex::ParticleReal>(E_coll));
                                  TwoProductComputeProductMomenta(
                                      ux[ip], uy[ip], uz[ip], m,
                                      ua_x, ua_y, ua_z, M,
                                      u1x_out, u1y_out, u1z_out, m,
                                      u2x_out, u2y_out, u2z_out, M,
                                      -scattering_process.m_energy_penalty*PhysConst::q_e,
                                      // TwoProductComputeProductMomenta expects the *released* energy here, hence
                                      // the negative sign; the energy penalty is also converted from eV to Joules.
                                      scattering_process.m_scattering_angle_model,
                                      engine, xi);

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
    auto const* const ionization_processes = m_ionization_processes_exe.data();
    auto const process_count = static_cast<int>(m_ionization_processes_exe.size());

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
                                                       ionization_processes, process_count,
                                                       m_mass1, m_total_collision_prob_ioniz,
                                                       m_nu_max_ioniz,
                                                       getBackgroundDensity(pti, lev, t)
                                                       );

        auto Transform = ImpactIonizationTransformFunc(
                                                       ionization_processes, process_count,
                                                       m_ionization_opal_w,
                                                       m_ionization_secondary_free_kinematics,
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


void BackgroundMCCCollision::doBackgroundReaction
( int lev, amrex::LayoutData<amrex::Real>* cost,
  WarpXParticleContainer& species1, int const ir,
  WarpXParticleContainer& product1, WarpXParticleContainer& product2,
  WarpXParticleContainer* ejected, amrex::Real t)
{
    ABLASTR_PROFILE("BackgroundMCCCollision::doBackgroundReaction()");
    using namespace amrex::literals;
    using std::sqrt;

    auto const& r = m_reactions[ir];
    const bool one_species = (&product1 == &product2);
    const SmartCopyFactory copy_factory_1(species1, product1);
    const SmartCopyFactory copy_factory_2(species1, product2);
    const auto Copy1 = copy_factory_1.getSmartCopy();
    const auto Copy2 = copy_factory_2.getSmartCopy();

    auto const* const exe = r.m_exe.data();
    auto const channel_count = static_cast<int>(r.m_exe.size());
    auto const total_collision_prob = r.m_total_collision_prob;
    auto const nu_max = r.m_nu_max;
    auto const m = m_mass1;
    auto const M = m_background_mass;
    auto T_a_func = m_background_temperature_func;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto& src_tile = species1.ParticlesAt(lev, pti);
        auto& tile1 = product1.ParticlesAt(lev, pti);
        auto& tile2 = product2.ParticlesAt(lev, pti);
        const auto np = src_tile.numParticles();
        if (np == 0) { continue; }
        const auto np1 = tile1.numParticles();
        const auto np2 = tile2.numParticles();

        // Decide the events, keeping the molecule's velocity and the channel
        // each one drew, so that the products are made from the same
        // collision that was accepted.
        using Index = std::remove_const_t<decltype(np1)>;
        amrex::Gpu::DeviceVector<Index> mask(np);
        amrex::Gpu::DeviceVector<int> channel(np);
        amrex::Gpu::DeviceVector<amrex::ParticleReal> uax(np), uay(np), uaz(np);
        auto* const p_mask = mask.dataPtr();
        auto* const p_channel = channel.dataPtr();
        auto* const p_uax = uax.dataPtr();
        auto* const p_uay = uay.dataPtr();
        auto* const p_uaz = uaz.dataPtr();

        const auto get_n_a = getBackgroundDensity(pti, lev, t);
        auto GetPosition = GetParticlePosition<PIdx>(pti);
        const auto ptd = src_tile.getConstParticleTileData();

        amrex::ParallelForRNG(np,
            [=] AMREX_GPU_DEVICE (long ip, amrex::RandomEngine const& engine)
            {
                p_mask[ip] = 0;
                // A projectile an earlier reaction consumed this step is still in
                // the tile until the redistribute; it must not react again.
                if (!amrex::ConstParticleIDWrapper(ptd.m_idcpu[ip]).is_valid()) { return; }
                if (amrex::Random(engine) > total_collision_prob) { return; }

                amrex::ParticleReal x, y, z;
                GetPosition(ip, x, y, z);
                const amrex::ParticleReal n_a = get_n_a(x, y, z);
                const amrex::ParticleReal vel_std =
                    sqrt(PhysConst::kb * T_a_func(x, y, z, t) / M);
                const amrex::ParticleReal ua_x = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                const amrex::ParticleReal ua_y = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                const amrex::ParticleReal ua_z = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);

                const amrex::ParticleReal vx = ptd.m_rdata[PIdx::ux][ip] - ua_x;
                const amrex::ParticleReal vy = ptd.m_rdata[PIdx::uy][ip] - ua_y;
                const amrex::ParticleReal vz = ptd.m_rdata[PIdx::uz][ip] - ua_z;
                const amrex::ParticleReal v_coll2 = vx*vx + vy*vy + vz*vz;
                double gamma, E_coll;
                ParticleUtils::getCollisionEnergy(v_coll2, m, M, gamma, E_coll);
                const auto E = static_cast<amrex::ParticleReal>(E_coll);
                amrex::ParticleReal sigma_total = 0;
                for (int c = 0; c < channel_count; ++c) {
                    sigma_total += exe[c].getCrossSection(E);
                }
                const amrex::ParticleReal nu_i = n_a * sigma_total * sqrt(v_coll2) / nu_max;
                if (amrex::Random(engine) > nu_i) { return; }

                // The channel, in proportion to its cross section; the last open
                // channel takes any rounding at the top.
                int pick = 0;
                if (channel_count > 1) {
                    const amrex::ParticleReal target = amrex::Random(engine) * sigma_total;
                    amrex::ParticleReal cumulative = 0;
                    for (int c = 0; c < channel_count; ++c) {
                        const amrex::ParticleReal sigma_c = exe[c].getCrossSection(E);
                        if (sigma_c <= 0) { continue; }
                        pick = c;
                        cumulative += sigma_c;
                        if (target < cumulative) { break; }
                    }
                }
                p_mask[ip] = 1;
                p_channel[ip] = pick;
                p_uax[ip] = ua_x;
                p_uay[ip] = ua_y;
                p_uaz[ip] = ua_z;
            });

        const bool ejects = !r.m_ejected_species.empty();
        amrex::Gpu::DeviceVector<amrex::ParticleReal> ejx(ejects ? np : 0),
            ejy(ejects ? np : 0), ejz(ejects ? np : 0);
        const BackgroundReactionTransformFunc transform{
            m, M, r.m_product_mass[0], r.m_product_mass[1], r.m_mode, exe,
            r.m_fragment_energy.data(), p_channel, p_uax, p_uay, p_uaz,
            ejects ? ejx.dataPtr() : nullptr, ejects ? ejy.dataPtr() : nullptr,
            ejects ? ejz.dataPtr() : nullptr, r.m_opal_w.data(), r.m_third_mass};

        if (one_species) {
            const auto num_added = filterCopyTransformParticles<2>(
                product1, tile1, src_tile, p_mask, np1, Copy1, transform);
            setNewParticleIDs(tile1, np1, num_added);
        } else {
            const auto num_added = filterCopyTransformParticles<1>(
                product1, product2, tile1, tile2, src_tile, p_mask, np1, np2,
                Copy1, Copy2, transform);
            setNewParticleIDs(tile1, np1, num_added);
            setNewParticleIDs(tile2, np2, num_added);
        }

        // The third products (ejected electrons or recoiling molecules), from
        // the same events (the transform above stored their velocities), at
        // the projectiles' positions.
        if (ejects) {
            auto& tile_e = ejected->ParticlesAt(lev, pti);
            const auto np_e = tile_e.numParticles();
            const SmartCopyFactory copy_factory_e(species1, *ejected);
            const auto CopyE = copy_factory_e.getSmartCopy();
            const auto num_e = filterCopyTransformParticles<1>(
                *ejected, tile_e, src_tile, p_mask, np_e, CopyE,
                EjectedElectronTransformFunc{ejx.dataPtr(), ejy.dataPtr(), ejz.dataPtr()});
            setNewParticleIDs(tile_e, np_e, num_e);
        }

        if (m_deplete_background && r.m_background_consumed > 0)
        {
            auto& warpx = WarpX::GetInstance();
            auto * const dn_mf = warpx.m_fields.get(
                MCCBackgroundField::deltaFieldName(m_background_name), lev);

            amrex::Box tilebox = pti.tilebox();
            tilebox.grow(warpx.get_ng_depos_rho());

            // Deposit the reacted projectiles, at their positions and weighted
            // by the molecules each event consumed, with unit negative charge:
            // the number-density decrement, in m^-3, at the gather's shape.
            WarpXParticleContainer::RealVector dw(np);
            auto* const p_dw = dw.dataPtr();
            // Fetched again: a product written into the projectile's own tile
            // (an ejected electron, or a charge-exchange ion) may have
            // reallocated it since the filter pass.
            auto const* const p_w =
                src_tile.getConstParticleTileData().m_rdata[PIdx::w];
            auto const consumed = static_cast<amrex::ParticleReal>(r.m_background_consumed);
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
            {
                p_dw[ip] = p_mask[ip] ? consumed * p_w[ip] : 0.0_prt;
            });

            amrex::FArrayBox local_dn;
            ablastr::particles::deposit_charge<WarpXParticleContainer>(
                pti, dw, /*charge=*/-1.0_prt,
                /*ion_lev=*/nullptr, dn_mf, local_dn, m_background_shape,
                WarpX::InvCellSize(lev),
                WarpX::LowerCorner(tilebox, lev, 0._rt),
                /*n_rz_azimuthal_modes=*/0,
                warpx.get_ng_depos_rho(), lev, amrex::IntVect(1),
                /*offset=*/0, /*np_to_deposit=*/np);
        }
        // The event arrays must outlive the kernels that read them.
        amrex::Gpu::streamSynchronize();

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
