/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_SCRAPEDPARTICLEENERGY_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_SCRAPEDPARTICLEENERGY_H_

#include "ReducedDiags.H"

#include <string>
#include <vector>

/**
 * \brief This class tallies the kinetic energy carried out of the domain by
 *  particles absorbed at the domain (and, when compiled in, embedded)
 *  boundaries: the particle-outflow advection term of a per-step energy
 *  ledger. For each species and boundary it reports the energy scraped
 *  during the step and the cumulative energy scraped since the start of
 *  the run (preserved across checkpoint/restart).
 *
 *  The tally reads the particle boundary buffer, so it requires the
 *  buffer to be filling for the species/boundaries of interest
 *  (<species>.save_particles_at_* = 1). Particles are attributed to the
 *  step recorded in their "stepScraped" attribute, and each particle is
 *  counted exactly once even if the buffer is later cleared by a
 *  BoundaryScraping diagnostic (the reduced diagnostics compute before
 *  the diagnostic flush within a step).
 */
class ScrapedParticleEnergy : public ReducedDiags
{
public:

    /**
     * \brief Constructor
     *
     * \param[in] rd_name reduced diags names
     */
    explicit ScrapedParticleEnergy (const std::string& rd_name);

    /**
     * \brief Tally the energy of particles newly arrived in the boundary
     * buffer. Runs every step (not gated on the write intervals) so the
     * cumulative integrals stay exact.
     *
     * \param[in] step current time step
     */
    void ComputeDiags (int step) final;

    void WriteCheckpointData (std::string const & dir) final;

    void ReadCheckpointData (std::string const & dir) final;

private:

    /** highest stepScraped value already tallied, per boundary and species */
    std::vector<std::vector<int>> m_last_step_counted;
};

#endif  // WARPX_DIAGNOSTICS_REDUCEDDIAGS_SCRAPEDPARTICLEENERGY_H_
