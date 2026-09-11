/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_DEPLETABLEBACKGROUNDSPEC_H_
#define WARPX_PARTICLES_COLLISION_DEPLETABLEBACKGROUNDSPEC_H_

#include <AMReX_Parser.H>

#include <string>

/**
 * \brief Description of a neutral background that a collision can deplete.
 *
 * A depletable background is carried on the mesh so that ionization can remove
 * from it. That makes it shared simulation state rather than a property of one
 * collision: if several collisions act on the same physical fill (say an
 * electron-neutral and an ion-neutral MCC on the same gas) they must read and
 * write one field, or the same gas would be consumed more than once.
 *
 * Collisions therefore declare a background by name and one field is
 * registered per distinct name, with collisions sharing a name sharing the
 * field.
 */
struct DepletableBackgroundSpec
{
    /** Registry key. Collisions declaring the same name share one field. */
    std::string m_background_name;

    /** The raw `background_density` input this collision declared. Kept only
     *  so that two collisions declaring the same background with different
     *  densities can be rejected rather than silently resolved. */
    std::string m_density_expression;

    /** Compiled initial density in m^-3, as a function of (x,y,z,t). */
    amrex::ParserExecutor<4> m_density_func;

    /** Shape order used both to gather the density and to deposit its
     *  depletion. The two must match for the pair to conserve. */
    int m_shape = 1;
};

#endif // WARPX_PARTICLES_COLLISION_DEPLETABLEBACKGROUNDSPEC_H_
