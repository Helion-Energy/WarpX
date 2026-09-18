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

    [[nodiscard]] amrex::ParticleReal get_nu_max (amrex::Vector<ScatteringProcess> const& mcc_processes) const;

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

private:

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
