/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_MCCBACKGROUNDDENSITY_H_
#define WARPX_PARTICLES_COLLISION_MCCBACKGROUNDDENSITY_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

/**
 * \brief Device-callable accessor for the MCC background neutral density.
 *
 * The background density is read per-particle at several points in the MCC
 * algorithm. Wrapping those reads behind one accessor keeps the call sites
 * independent of where the density actually comes from, so that a mesh-carried
 * (and therefore depletable) background can be introduced without touching the
 * collision kernels again.
 *
 * At present the only source is the user-supplied `background_density(x,y,z,t)`
 * parser, evaluated at a time fixed when the accessor is constructed. The
 * accessor is captured by value into GPU kernels, so it must stay trivially
 * copyable and free of owning members.
 */
struct MCCBackgroundDensity
{
    /** Construct from the compiled density parser and the current time. */
    MCCBackgroundDensity (
        amrex::ParserExecutor<4> const& parser, amrex::Real t
    ) : m_parser(parser), m_t(t) { }

    /**
    * \brief Return the background density in m^-3 at a particle position.
    *
    * @param[in] x,y,z particle position, Cartesian in all geometries
    */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal operator() (
        amrex::ParticleReal x, amrex::ParticleReal y, amrex::ParticleReal z
    ) const noexcept
    {
        return static_cast<amrex::ParticleReal>(m_parser(x, y, z, m_t));
    }

private:
    amrex::ParserExecutor<4> m_parser;
    amrex::Real m_t;
};

#endif // WARPX_PARTICLES_COLLISION_MCCBACKGROUNDDENSITY_H_
