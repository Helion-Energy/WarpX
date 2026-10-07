/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_SOURCE_ELIGIBILITY_H_
#define WARPX_HYBRID_SOURCE_ELIGIBILITY_H_

#include <AMReX_MultiFab.H>
#include <ablastr/particles/NodalFieldGather.H>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

class HybridPICModel;

/** Default-off, source-only sampling guard. No field-resistivity modification.
 * Counts are resident computational markers, not independent statistical samples.
 * Masks carry overlapping reason bits; zero is eligible. See source_guard.rst.
 */
class HybridSourceEligibility
{
public:
    enum Channel { Ohmic = 0, Thermal = 1, Stopping = 2 };
    enum Reason { Vacuum = 1, LowCount = 2, Axis = 4 };

    void ReadParameters ();
    void Validate (HybridPICModel const& model) const;
    bool enabled () const { return m_enabled; }
    bool guards (Channel c) const { return m_enabled && m_channels[c]; }
    void Prepare (int lev, amrex::MultiFab const& rho, HybridPICModel const& model);
    void BeginBatch (int lev, amrex::MultiFab const& rho, HybridPICModel const& model);
    void EndBatch () { m_batch = false; }

    amrex::MultiFab const& nodes () const { return *m_nodes; }
    amrex::MultiFab const& cells () const { return *m_cells; }
    amrex::MultiFab const& counts () const { return *m_counts; }
    amrex::MultiFab const& contributors () const { return *m_contributors; }
    std::vector<std::string> const& backgroundSpecies () const { return m_background; }
    long epoch () const { return m_epoch; }

    // Shared nodal (nu, nu*T_bath[K], sampled Ti[eV]) for the two relaxation legs.
    amrex::MultiFab& NewRelaxation (std::string const& species, amrex::MultiFab const& te);
    amrex::MultiFab const& Relaxation (std::string const& species) const;
    void ClearRelaxation () { m_relaxation.clear(); m_relaxation_epoch = m_epoch; }

    void BeginChannel (Channel c, amrex::MultiFab const& te);
    void EndChannel (int lev, Channel c, amrex::Real dt, amrex::MultiFab const& te,
                     amrex::MultiFab const& capacity, amrex::MultiFab const* pedestal,
                     amrex::Real gamma);
    void RecordOhmic (int lev, amrex::MultiFab const& tally, amrex::Real dt);
    void RecordStoppingParticle (amrex::Real actual_J, amrex::Real dt);
    void RecordIon (amrex::Real actual_J, amrex::Real thermal_J, amrex::Real redirect_J,
                    amrex::Real dt, bool has_redirect);
    void Phase (int lev, char const* phase, HybridPICModel const& model, bool abort = false);
    // NVCC extended device lambdas require a public enclosing member function.
    void Coverage (int lev, amrex::MultiFab const* previous);
    amrex::Real electronEnergy (Channel c) const { return m_last_electron_J[c]; }
    amrex::Real declinedOhmic () const { return m_declined_ohmic_J; }
    amrex::Real unevaluableOhmic () const { return m_unevaluable_ohmic_nodes; }
    amrex::Real ionEnergy () const { return m_last_ion_J; }

    /** Whole-interaction guard using exactly the stopping deposition weights.
     * Zero-weight corners do not enlarge the support. Called BEFORE particle drag.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static bool
    StoppingEligible (amrex::Real x, amrex::Real y, amrex::Real z,
        amrex::Array4<amrex::Real const> const& reasons,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& invdx)
    {
        int i = 0, j = 0, k = 0;
        amrex::Real w[AMREX_SPACEDIM][2];
        ablastr::particles::compute_weights<amrex::IndexType::NODE>(
            x, y, z, lo, invdx, i, j, k, w);
        for (int c = 0; c < (1 << AMREX_SPACEDIM); ++c) {
            int const di = c & 1;
            int const dj = AMREX_SPACEDIM > 1 ? ((c >> 1) & 1) : 0;
            int const dk = AMREX_SPACEDIM > 2 ? ((c >> 2) & 1) : 0;
            amrex::Real weight = w[0][di];
#if AMREX_SPACEDIM >= 2
            weight *= w[1][dj];
#endif
#if AMREX_SPACEDIM == 3
            weight *= w[2][dk];
#endif
            if (weight > 0 && reasons(i+di,j+dj,k+dk,Stopping) != 0) { return false; }
        }
        return true;
    }

private:
    bool m_enabled = false;
    std::array<bool,3> m_channels{{true,true,true}};
    std::vector<std::string> m_background;
    int m_minimum = 16, m_axis_cells = 1, m_diagnostic_interval = 1, m_export_interval = 0;
    amrex::Real m_n_min = 0;
    std::string m_output_prefix = "source_guard";
    bool m_batch = false;
    long m_epoch = 0, m_relaxation_epoch = -1;
    amrex::Real m_stopping_particle_J = 0, m_stopping_electron_J = 0, m_stopping_dt = 0;
    std::unique_ptr<amrex::MultiFab> m_nodes, m_cells, m_counts, m_contributors;
    std::map<std::string,std::unique_ptr<amrex::MultiFab>> m_relaxation;
    std::array<std::unique_ptr<amrex::MultiFab>,3> m_before, m_delta;
    std::array<amrex::Real,3> m_last_electron_J{};
    amrex::Real m_declined_ohmic_J = 0, m_unevaluable_ohmic_nodes = 0, m_last_ion_J = 0;
    void Export (int lev, char const* phase) const;
};
#endif
