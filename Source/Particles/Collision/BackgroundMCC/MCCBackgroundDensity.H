/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_MCCBACKGROUNDDENSITY_H_
#define WARPX_PARTICLES_COLLISION_MCCBACKGROUNDDENSITY_H_

#include "Particles/Gather/ScalarFieldGather.H"

#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
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
 * There are two sources. By default the density is the user-supplied
 * `background_density(x,y,z,t)` parser, evaluated at a time fixed when the
 * accessor is constructed. With `deplete_background` on it is instead gathered
 * from a mesh field that ionization removes from, using the same shape factor
 * that deposits the depletion, which is what makes the pair conservative.
 *
 * The accessor is captured by value into GPU kernels, so it must stay
 * trivially copyable and free of owning members.
 */
struct MCCBackgroundDensity
{
    /** Analytic background: evaluate the parser at the particle position. */
    MCCBackgroundDensity (
        amrex::ParserExecutor<4> const& parser, amrex::Real t
    ) : m_parser(parser), m_t(t) { }

    /** Mesh-resident background: gather the field at the particle position.
     *
     * \p xyzmin and \p lo must be the physical and index lower corners of one
     * and the same box. Any box will do, since a matched pair resolves a
     * particle to the same absolute cell either way, but a mismatched pair
     * silently shifts the stencil.
     */
    MCCBackgroundDensity (
        amrex::Array4<amrex::Real const> const& n_arr,
        amrex::IntVect const& n_type,
        amrex::XDim3 const& dinv,
        amrex::XDim3 const& xyzmin,
        amrex::Dim3 const& lo,
        int nox
    ) : m_n_arr(n_arr), m_n_type(n_type), m_dinv(dinv), m_xyzmin(xyzmin),
        m_lo(lo), m_nox(nox), m_use_field(true) { }

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
        if (m_use_field) {
            return static_cast<amrex::ParticleReal>(
                doGatherScalarShapeN(x, y, z, m_n_arr, m_n_type,
                                     m_dinv, m_xyzmin, m_lo, m_nox));
        }
        return static_cast<amrex::ParticleReal>(m_parser(x, y, z, m_t));
    }

    /** Whether this accessor reads the mesh field rather than the parser. */
    [[nodiscard]] bool usesField () const noexcept { return m_use_field; }

private:
    // analytic source
    amrex::ParserExecutor<4> m_parser;
    amrex::Real m_t = 0;

    // mesh source
    amrex::Array4<amrex::Real const> m_n_arr;
    amrex::IntVect m_n_type;
    amrex::XDim3 m_dinv{};
    amrex::XDim3 m_xyzmin{};
    amrex::Dim3 m_lo{};
    int m_nox = 0;

    bool m_use_field = false;
};

#endif // WARPX_PARTICLES_COLLISION_MCCBACKGROUNDDENSITY_H_
