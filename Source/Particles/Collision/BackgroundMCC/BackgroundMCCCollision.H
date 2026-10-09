/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_
#define WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_

#include "MCCBackgroundDensity.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <string>

class BackgroundMCCCollision final
    : public CollisionBase
{
public:
    explicit BackgroundMCCCollision (std::string const& collision_name);

    ~BackgroundMCCCollision () override = default;

    BackgroundMCCCollision ( BackgroundMCCCollision const &)             = delete;
    BackgroundMCCCollision& operator= ( BackgroundMCCCollision const & ) = delete;
    BackgroundMCCCollision ( BackgroundMCCCollision&& )                  = delete;
    BackgroundMCCCollision& operator= ( BackgroundMCCCollision&& )       = delete;

    [[nodiscard]] amrex::ParticleReal get_nu_max (amrex::Vector<ScatteringProcess> const& mcc_processes,
                                                  amrex::ParticleReal speed_mass) const;

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** \brief This collision's background, when it is set to deplete.
     *
     * Returns a single entry when `deplete_background` is on, and nothing
     * otherwise, in which case the background stays the analytic function it
     * has always been and no mesh field is allocated for it.
     */
    [[nodiscard]] amrex::Vector<DepletableBackgroundSpec>
    getDepletableBackgrounds () const override;

    /** Perform particle conserving MCC collisions within a tile
     *
     * @param pti particle iterator
     * @param get_n_a accessor for the background density over this tile
     * @param t current time
     *
     */
    void doBackgroundCollisionsWithinTile (
        WarpXParIter& pti, MCCBackgroundDensity const& get_n_a, amrex::Real t);

    /** Build the background density accessor for one tile
     *
     * Returns an accessor reading the analytic parser, or, when
     * `deplete_background` is set, one gathering the mesh density field over
     * the same box the depletion deposit writes to.
     *
     * @param[in] pti particle iterator
     * @param[in] lev the mesh-refinement level
     * @param[in] t current time
     */
    [[nodiscard]] MCCBackgroundDensity getBackgroundDensity (
        WarpXParIter& pti, int lev, amrex::Real t) const;

    /** Apply one step's accumulated depletion to the background density
     *
     * Sums the per-step accumulator across tile and process boundaries and
     * reflects it at domain boundaries exactly as rho is treated, then
     * subtracts it from the density, holding the result at or above zero.
     * Must run after every tile of the level has deposited, and never before
     * the accumulator is complete, since the fold is not idempotent.
     *
     * @param[in] lev the mesh-refinement level
     */
    void applyDepletion (int lev);

    /** Perform MCC ionization interactions
     *
     * @param[in] lev the mesh-refinement level
     * @param[in,out] cost pointer to (load balancing) cost corresponding to box where present particles are ionized.
     * @param[in,out] species1,species2 reference to species container used to inject
     * new particles
     * @param t current time
     *
     */
    void doBackgroundIonization (
                                 int lev,
                                 amrex::LayoutData<amrex::Real>* cost,
                                 WarpXParticleContainer& species1,
                                 WarpXParticleContainer& species2,
                                 amrex::Real t
                                 );

    /** Perform the background reactions of one kind
     *
     * The projectile and a background molecule become two particles of the
     * product species, with the process's angle model and reaction energy
     * applied in the centre-of-mass frame. The projectile is removed, and
     * the consumed background molecules are deposited into the depletion
     * accumulator.
     *
     * @param[in] lev the mesh-refinement level
     * @param[in,out] cost pointer to (load balancing) cost
     * @param[in,out] species1 the projectile species
     * @param[in] ir index of the reaction in m_reactions
     * @param[in,out] product1,product2 containers receiving the products
     * @param[in,out] ejected container receiving ejected electrons, or nullptr
     * @param t current time
     */
    void doBackgroundReaction (
        int lev, amrex::LayoutData<amrex::Real>* cost,
        WarpXParticleContainer& species1, int ir,
        WarpXParticleContainer& product1, WarpXParticleContainer& product2,
        WarpXParticleContainer* ejected, amrex::Real t);

private:

    /** A background reaction: a pass of its own that creates particles.
     *
     *  Three modes, all set per process in the inputs:
     *  - exchange: a `two_product_reaction[_<name>]` given
     *    `<process>_product_species = A B`. The projectile and a background
     *    molecule become A + B; the projectile is removed.
     *  - fragment the target: any excitation or two-product process given
     *    `<process>_fragment_species = A B` (A and B may be the same species).
     *    The projectile scatters, paying `<process>_energy`; the molecule
     *    breaks into A + B, which share `<process>_fragment_energy` back to
     *    back about the molecule's post-collision velocity.
     *    With `<process>_ejected_species = E` the projectile also ejects an
     *    electron into E (dissociative ionization); the two electrons share
     *    what is left after the cost, equally or by `<process>_ejected_opal_w`.
     *  - fragment the projectile: the same with
     *    `<process>_fragmenting = projectile`. The projectile breaks into A + B;
     *    the molecule recoils and stays part of the background, or, with
     *    `<process>_recoil_species = R`, leaves it as a particle of R (a charge
     *    transfer that breaks the projectile: D3+ + D2 -> D + D2 + D2+). The
     *    masses must balance over all three products.
     *  Fragmentation channels with the same products and mode share one pass
     *  (one majorant over their summed cross sections; each event picks its
     *  channel in proportion to its cross section), as ionization channels
     *  do. Without these keys a process keeps its historical meaning. */
    static constexpr int ReactionExchange = 0;
    static constexpr int FragmentTarget = 1;
    static constexpr int FragmentProjectile = 2;
    struct BackgroundReaction {
        /** The channels: one for an exchange, every fragmentation channel with
         *  the same products and mode otherwise */
        amrex::Vector<ScatteringProcess> m_process;
        amrex::Vector<std::string> m_channel_names;
        amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_exe;
        /** Kinetic energy release of each channel's fragments, eV */
        amrex::Vector<amrex::ParticleReal> m_fragment_energy_h;
        amrex::Gpu::DeviceVector<amrex::ParticleReal> m_fragment_energy;
        /** The third product's species, empty for none. Target fragmentation:
         *  an electron the projectile ejects (dissociative ionization); the
         *  energy left after the cost is shared between the two electrons,
         *  equally or, per channel, by Opal's law with width w (eV).
         *  Projectile fragmentation: the recoiling molecule. */
        std::string m_ejected_species;
        amrex::Vector<amrex::ParticleReal> m_opal_w_h;
        amrex::Gpu::DeviceVector<amrex::ParticleReal> m_opal_w;
        int m_mode = ReactionExchange;
        std::string m_product_species[2];
        /** Background molecules consumed per event (deposited as depletion) */
        int m_background_consumed = 1;
        amrex::ParticleReal m_product_mass[2] = {0, 0};
        amrex::ParticleReal m_third_mass = 0;
        amrex::ParticleReal m_nu_max = 0;
        amrex::ParticleReal m_total_collision_prob = 0;
    };
    amrex::Vector<BackgroundReaction> m_reactions;

    amrex::Vector<ScatteringProcess> m_scattering_processes;
    amrex::Vector<ScatteringProcess> m_ionization_processes;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_ionization_processes_exe;

    bool init_flag = false;
    bool ionization_flag = false;

    amrex::ParticleReal m_mass1;

    amrex::ParticleReal m_max_background_density = 0;
    amrex::ParticleReal m_background_mass;
    amrex::ParticleReal m_total_collision_prob;
    amrex::ParticleReal m_total_collision_prob_ioniz = 0;
    amrex::ParticleReal m_nu_max;
    amrex::ParticleReal m_nu_max_ioniz;

    /** Opal energy sharing parameter, w, in eV. Zero, the default, splits
     *  the energy left after ionization equally between the two outgoing
     *  electrons, which reproduces the historical behaviour exactly. */
    double m_ionization_opal_w = 0;
    /** Set the ionization-ejected electron's direction by free kinematics */
    bool m_ionization_secondary_free_kinematics = false;

    amrex::Parser m_background_density_parser;
    amrex::Parser m_background_temperature_parser;

    amrex::ParserExecutor<4> m_background_density_func;
    amrex::ParserExecutor<4> m_background_temperature_func;

    /** Whether ionization removes from the background rather than leaving it
     *  as an inexhaustible reservoir. Off by default, which reproduces the
     *  historical behaviour exactly. */
    bool m_deplete_background = false;

    /** Key naming the background. Collisions sharing it share one field. */
    std::string m_background_name;

    /** The raw `background_density` input, kept for consistency checking
     *  against other collisions that declare the same background. */
    std::string m_background_density_expression;

    /** Shape order for gathering the background and depositing its
     *  depletion. Defaults to `algo.particle_shape`. */
    int m_background_shape = 1;
};

#endif // WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_
