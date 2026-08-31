/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEBOUNDARYFLUX_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEBOUNDARYFLUX_H_

#include "ReducedDiags.H"

#include <string>

/**
 * \brief Cumulative particle losses through the Absorbing/Open domain
 * boundary faces, per species and per face: physical weight (count),
 * charge [C], and kinetic energy [J].
 *
 * The cumulative convention matches FieldPoyntingFlux's integrated columns,
 * so windowed differences of any column give the flux through that face over
 * the window; in particular the KE columns are the particle side of an
 * end-loss power ledger (field side: FieldPoyntingFlux; sinks:
 * HybridDissipation).
 *
 * Face columns are labeled <species>_<axis>_<lo|hi>_{count,charge_C,KE_J}
 * with the axis names of the geometry (RZ: r, z). Losses are booked where
 * the particle boundary conditions invalidate particles, at every
 * application (including per-subcycle applications for subcycled species).
 * Thermal boundaries re-inject and are not booked. The tallies are not
 * preserved across checkpoint/restart (v1): analyses across a restart must
 * use windowed deltas within each leg.
 *
 * Relation to ScrapedParticleEnergy: that diagnostic reads the particle
 * boundary buffer (requires <species>.save_particles_at_* and a scraping
 * diagnostic to flush the buffer) and is checkpoint-preserved; this one is
 * buffer-free (no per-particle storage of the losses, suitable for
 * loss-dominated runs) and adds the weight and charge columns.
 */
class ParticleBoundaryFlux : public ReducedDiags
{
public:

    /**
     * \brief Constructor.
     *
     * \param[in] rd_name reduced diags names
     */
    ParticleBoundaryFlux (const std::string& rd_name);

    /**
     * \brief Collect the cumulative per-face tallies into the output row.
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) final;
};

#endif  // WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLEBOUNDARYFLUX_H_
