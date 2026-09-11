/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_MCCBACKGROUNDFIELD_H_
#define WARPX_PARTICLES_COLLISION_MCCBACKGROUNDFIELD_H_

#include "Particles/Collision/DepletableBackgroundSpec.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include <string>

/** Mesh-carried neutral density for depletable MCC backgrounds. */
namespace MCCBackgroundField
{
    /**
     * \brief Name a background's density field is registered under.
     *
     * \param[in] background_name the user-facing background name
     */
    std::string fieldName (std::string const& background_name);

    /**
     * \brief Name the per-step depletion accumulator is registered under.
     *
     * This is scratch: it is zeroed at the start of every depletion pass and
     * must not be checkpointed. It exists because the density field is
     * accumulated across steps, so the depletion of one step cannot be summed
     * across guard cells in place -- doing so would re-sum the density already
     * there. The step's decrement is gathered here instead, made consistent
     * across tiles and boundaries, and only then applied.
     *
     * \param[in] background_name the user-facing background name
     */
    std::string deltaFieldName (std::string const& background_name);

    /**
     * \brief Allocate and fill the density field of each given background.
     *
     * Must be called from the level-allocation path, which both a fresh start
     * and a restart pass through, so that the field exists before a checkpoint
     * read can populate it. The parser fill performed here is what a restart
     * from a checkpoint that predates this field falls back to; without it such
     * a restart would silently begin with no gas at all.
     *
     * The field is given rho's staggering and guard cells because the
     * depletion deposit reuses rho's deposition and boundary-folding
     * machinery, which assumes that layout.
     *
     * \param[in,out] fields the register to allocate into
     * \param[in] backgrounds distinct backgrounds to allocate, one field each
     * \param[in] lev mesh refinement level
     * \param[in] ba cell-centered box array of the level
     * \param[in] dm distribution mapping of the level
     * \param[in] nodal_flag staggering to convert \p ba to (rho's)
     * \param[in] ngrow guard cells to allocate (rho's)
     * \param[in] geom geometry of the level
     */
    void allocInit (
        ablastr::fields::MultiFabRegister& fields,
        amrex::Vector<DepletableBackgroundSpec> const& backgrounds,
        int lev,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        amrex::IntVect const& nodal_flag,
        amrex::IntVect const& ngrow,
        amrex::Geometry const& geom
    );
}

#endif // WARPX_PARTICLES_COLLISION_MCCBACKGROUNDFIELD_H_
