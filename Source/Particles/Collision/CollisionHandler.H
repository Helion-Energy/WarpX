/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_COLLISIONHANDLER_H_
#define WARPX_PARTICLES_COLLISION_COLLISIONHANDLER_H_

#include "CollisionBase.H"

#include "Particles/MultiParticleContainer_fwd.H"

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>

/* \brief CollisionHandler is a light weight class that contains the
 * list of collisions to be done.
 */
class CollisionHandler
{
public:
    explicit CollisionHandler (const MultiParticleContainer*  mypc);

    /* Perform all of the collisions */
    void doCollisions (int step, amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc);

    /**
     * \brief The distinct depletable backgrounds declared by all collisions.
     *
     * Collisions declaring the same background name are collapsed to one
     * entry, so that they end up sharing a single mesh field; conflicting
     * declarations of one background are rejected.
     */
    [[nodiscard]] amrex::Vector<DepletableBackgroundSpec> getDepletableBackgrounds () const;

private:

    amrex::Vector<std::string> collision_names;
    amrex::Vector<std::string> collision_types;
    amrex::Vector< std::unique_ptr<CollisionBase> > allcollisions;

    bool m_use_global_debye_length = false;

};

#endif // WARPX_PARTICLES_COLLISION_COLLISIONHANDLER_H_
