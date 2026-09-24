/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_
#define WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_

#include "MCCBackgroundDensity.H"
#include "Particles/Collision/ScatteringAngle.H"
#include "Particles/Collision/ScatteringProcess.H"

#include "Particles/Algorithms/KineticEnergy.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>

/** @file
 *
 * This file contains filter and transform functors for impact ionization
 */

/**
 * \brief Filter functor for impact ionization
 */
class ImpactIonizationFilterFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationFilterFunc functor.
    *
    * This function sample a random number and compares it to the total
    * collision probability to see if the given particle should be considered
    * for an ionization event. If the particle passes this stage the collision
    * cross-section is calculated given its energy and another random number
    * is used to determine whether it actually collides.
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] processes executors of the ionization channels, whose cross sections
                 are summed
    * @param[in] process_count number of ionization channels
    * @param[in] mass colliding particle's mass (could also assume electron)
    * @param[in] total_collision_prob total probability for a collision to occur
    * @param[in] nu_max maximum collision frequency
    * @param[in] n_a accessor for the background density in m^-3 at a
                 particle position
    */
    ImpactIonizationFilterFunc(
        ScatteringProcess::Executor const* processes,
        int const process_count,
        double const mass,
        amrex::ParticleReal const total_collision_prob,
        amrex::ParticleReal const nu_max,
        MCCBackgroundDensity const& n_a
    ) : m_processes(processes), m_process_count(process_count), m_mass(mass),
        m_total_collision_prob(total_collision_prob),
        m_nu_max(nu_max), m_n_a(n_a) { }

    /**
    * \brief Functor call. This method determines if a given (electron) particle
    * should undergo an ionization collision.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @param[in] engine the random number state and factory
    * @return true if a collision occurs, false otherwise
    */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (
        const PData& ptd, int const i, amrex::RandomEngine const& engine
    ) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // determine if this particle should collide
        if (Random(engine) > m_total_collision_prob) { return false; }

        // get references to the particle to get its position
        const auto& p = ptd.getSuperParticle(i);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate neutral density at particle location
        const ParticleReal n_a = m_n_a(x, y, z);

        // get the particle velocity
        const ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
        const ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
        const ParticleReal uz = ptd.m_rdata[PIdx::uz][i];

        // calculate kinetic energy
        constexpr auto eV = PhysConst::q_e;
        E_coll = Algorithms::KineticEnergy<double>(ux, uy, uz, static_cast<amrex::ParticleReal>(m_mass))/eV;

        // total ionization cross-section over the channels
        ParticleReal sigma_E = 0;
        for (int k = 0; k < m_process_count; k++) {
            sigma_E += m_processes[k].getCrossSection(static_cast<amrex::ParticleReal>(E_coll));
        }

        // calculate normalized collision frequency
        const ParticleReal u_coll2 = ux*ux + uy*uy + uz*uz;
        const ParticleReal nu_i = n_a * sigma_E * sqrt(u_coll2) / m_nu_max;

        // check if this collision should be performed
        return (Random(engine) <= nu_i);
    }

private:
    ScatteringProcess::Executor const* m_processes;
    int m_process_count;
    double m_mass;
    amrex::ParticleReal m_total_collision_prob = 0;
    amrex::ParticleReal m_nu_max;
    MCCBackgroundDensity m_n_a;
};


/**
 * \brief Transform functor for impact ionization
 */
class ImpactIonizationTransformFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationTransformFunc functor.
    *
    * The transform is responsible for appropriately decreasing the kinetic
    * energy of the colliding particle and assigning appropriate velocities
    * to the two newly created particles. To this end the energy cost of
    * ionization is passed to the constructor as well as the mass of the
    * colliding species and the standard deviation of the ion velocity
    * (normalized temperature).
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] processes executors of the ionization channels. Each event is given one
                 channel, drawn in proportion to its cross section at the incident energy;
                 that channel's energy cost is paid and its anisotropy table (if any) sets
                 the deflection of the incident electron
    * @param[in] process_count number of ionization channels
    * @param[in] opal_w the Opal energy sharing parameter, w, in eV. When zero
                 the energy left after ionization is split equally between the
                 two outgoing electrons instead.
    * @param[in] secondary_free_kinematics if true, the ejected electron's direction is
                 set from the incident electron's deflection by the free kinematics of
                 Magboltz (transverse momentum balanced between the two electrons);
                 otherwise it is isotropic
    * @param[in] mass1 mass of the colliding species
    * @param[in] sqrt_kb_m value of sqrt(kB/m), where kB is Boltzmann's constant
                 and m is the background neutral mass
    * @param[in] T_a_func ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] t the current simulation time
    */
    ImpactIonizationTransformFunc(
        ScatteringProcess::Executor const* processes, int process_count,
        double opal_w, bool secondary_free_kinematics,
        double mass1, amrex::ParticleReal sqrt_kb_m,
        amrex::ParserExecutor<4> const& T_a_func, amrex::Real t
    ) :  m_processes(processes), m_process_count(process_count), m_opal_w(opal_w),
         m_secondary_free_kinematics(secondary_free_kinematics), m_mass1(mass1),
         m_sqrt_kb_m(sqrt_kb_m), m_T_a_func(T_a_func), m_t(t) { }

    /**
    * \brief Functor call. It determines the properties of the generated pair
    * and decreases the kinetic energy of the colliding particle. Inputs are
    * standard from the FilterCopyTransfrom::filterCopyTransformParticles
    * function.
    *
    * @param[in,out] dst1 target species 1 (electrons)
    * @param[in,out] dst2 target species 2 (ions)
    * @param[in] src source species (electrons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst1 particle index of target species 1
    * @param[in] i_dst2 particle index of target species 2
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst1, DstData& dst2, SrcData& src,
        int const i_src, int const i_dst1, int const i_dst2,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // get references to the particle to get its position
        const auto& p = src.getSuperParticle(i_src);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate standard deviation in neutral velocity distribution using
        // the local temperature
        const ParticleReal ion_vel_std = m_sqrt_kb_m * std::sqrt(m_T_a_func(x, y, z, m_t));

        // get references to the original particle's velocity
        auto& ux = src.m_rdata[PIdx::ux][i_src];
        auto& uy = src.m_rdata[PIdx::uy][i_src];
        auto& uz = src.m_rdata[PIdx::uz][i_src];

        // get references to the new particles' velocities
        auto& e_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& e_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& e_uz = dst1.m_rdata[PIdx::uz][i_dst1];
        auto& i_ux = dst2.m_rdata[PIdx::ux][i_dst2];
        auto& i_uy = dst2.m_rdata[PIdx::uy][i_dst2];
        auto& i_uz = dst2.m_rdata[PIdx::uz][i_dst2];

        // calculate kinetic energy
        constexpr auto eV = PhysConst::q_e;
        E_coll = Algorithms::KineticEnergy<double>(ux, uy, uz, m_mass1)/eV;
        const auto E_coll_prt = static_cast<ParticleReal>(E_coll);

        // Pick the channel. The filter accepted this electron with probability proportional
        // to the summed cross section at this same energy (the electron is unchanged since),
        // so drawing channel i with probability sigma_i/sigma_total samples it exactly. A
        // single channel draws nothing, leaving the random stream as it was.
        int channel = 0;
        if (m_process_count > 1) {
            ParticleReal sigma_total = 0;
            for (int i = 0; i < m_process_count; i++) {
                sigma_total += m_processes[i].getCrossSection(E_coll_prt);
            }
            const ParticleReal target = Random(engine) * sigma_total;
            ParticleReal cumulative = 0;
            for (int i = 0; i < m_process_count; i++) {
                const ParticleReal sigma_i = m_processes[i].getCrossSection(E_coll_prt);
                if (sigma_i <= 0) { continue; }
                // The last open channel takes any rounding at the top of the range.
                channel = i;
                cumulative += sigma_i;
                if (target < cumulative) { break; }
            }
        }
        auto const& process = m_processes[channel];

        // Share the energy left after paying the ionization cost between the
        // incident electron and the one it ejects.
        const double E_avail = E_coll - process.m_energy_penalty;
        double E_eject;
        if (m_opal_w > 0.0) {
            // Opal, Peterson & Beaty, J. Chem. Phys. 55, 4100 (1971) measured
            // the ejected electron energy to be distributed as 1/(1+(W/w)^2).
            // Truncating that at half of the available energy, so that the
            // ejected electron is the slower of the two, inverts to the sample
            // below (Vahedi & Surendra, Comput. Phys. Commun. 87, 179 (1995)).
            E_eject = m_opal_w * std::tan(
                Random(engine) * std::atan(E_avail / (2.0*m_opal_w)));
        } else {
            // Equal sharing draws no random number, so it leaves the random
            // stream as it was.
            E_eject = E_avail / 2.0;
        }

        const auto E_out_scat = static_cast<amrex::ParticleReal>((E_avail - E_eject) * eV);
        const auto E_out_eject = static_cast<amrex::ParticleReal>(E_eject * eV);

        // precalculate often used value
        constexpr auto c2 = PhysConst::c2;
        const auto mc2 = m_mass1*c2;

        const amrex::ParticleReal up_scat =
            sqrt(E_out_scat * (E_out_scat + 2.0_prt*mc2) / c2) / m_mass1;
        const amrex::ParticleReal up_eject =
            sqrt(E_out_eject * (E_out_eject + 2.0_prt*mc2) / c2) / m_mass1;

        // The anisotropy at the incident energy (the energy its cross section is tabulated
        // against). Without a table it is zero, i.e. isotropic.
        const auto xi = process.getAnisotropy(E_coll_prt);

        const auto ux_d = static_cast<double>(ux);
        const auto uy_d = static_cast<double>(uy);
        const auto uz_d = static_cast<double>(uz);
        const double u_norm = std::sqrt(ux_d*ux_d + uy_d*uy_d + uz_d*uz_d);
        if (m_secondary_free_kinematics && u_norm > 0.0) {
            // Free kinematics, as in Magboltz (MONTEFTT): draw the incident electron's
            // deflection (cos_p, phi) about its incident direction n, then put the ejected
            // electron at the azimuth phi + pi with p_eject sin(theta_s) = p_scat sin(theta_p),
            // so the transverse momenta of the two electrons cancel and the ion takes only
            // longitudinal momentum. Magboltz writes the ratio as sqrt(E_scat/E_eject), which
            // is the momentum ratio non-relativistically. As in Magboltz, sin(theta_s) is
            // clamped at 1 (the balance is then not exact), and the ejected electron is put
            // in the same hemisphere as the incident one.
            const double nx = ux_d/u_norm;
            const double ny = uy_d/u_norm;
            const double nz = uz_d/u_norm;
            double cos_p, phi;
            ScatteringAngle::drawOkhrimovskyyAngles(xi, engine, cos_p, phi);
            const double sin_p = std::sqrt(amrex::max(1.0 - cos_p*cos_p, 0.0));
            const auto up_scat_d = static_cast<double>(up_scat);
            const auto up_eject_d = static_cast<double>(up_eject);
            const double sin_s = (up_eject_d > 0.0)
                ? amrex::min(sin_p * up_scat_d / up_eject_d, 1.0) : 1.0;
            const double cos_s = std::copysign(
                std::sqrt(amrex::max(1.0 - sin_s*sin_s, 0.0)), cos_p);

            double dir_x, dir_y, dir_z;
            ScatteringAngle::rotateAboutDirection(nx, ny, nz, cos_p, phi, dir_x, dir_y, dir_z);
            ux = static_cast<ParticleReal>(dir_x * up_scat_d);
            uy = static_cast<ParticleReal>(dir_y * up_scat_d);
            uz = static_cast<ParticleReal>(dir_z * up_scat_d);
            ScatteringAngle::rotateAboutDirection(nx, ny, nz, cos_s, phi + MathConst::pi,
                                                  dir_x, dir_y, dir_z);
            e_ux = static_cast<ParticleReal>(dir_x * up_eject_d);
            e_uy = static_cast<ParticleReal>(dir_y * up_eject_d);
            e_uz = static_cast<ParticleReal>(dir_z * up_eject_d);
        } else {
            // Deflect the incident electron about its own direction; xi = 0 is exactly the
            // isotropic draw. The ejected electron is emitted isotropically.
            ScatteringAngle::ScatterVelocityOkhrimovskyy(ux, uy, uz, up_scat, xi, engine);
            ParticleUtils::RandomizeVelocity(e_ux, e_uy, e_uz, up_eject, engine);
        }

        // get velocities for the ion from a Maxwellian distribution
        i_ux = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        i_uy = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        i_uz = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
    }

private:
    ScatteringProcess::Executor const* m_processes;
    int m_process_count;
    /** Opal energy sharing parameter in eV; zero selects equal sharing */
    double m_opal_w;
    /** Set the ejected electron's direction by free kinematics rather than isotropically */
    bool m_secondary_free_kinematics;
    double m_mass1;
    amrex::ParticleReal m_sqrt_kb_m;
    amrex::ParserExecutor<4> m_T_a_func;
    amrex::Real m_t;
};
#endif // WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_
