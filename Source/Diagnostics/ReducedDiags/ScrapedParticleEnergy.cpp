/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ScrapedParticleEnergy.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>

#include <fstream>
#include <vector>

using namespace amrex;

// constructor
ScrapedParticleEnergy::ScrapedParticleEnergy (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    auto & pbb = warpx.GetParticleBoundaryBuffer();

    const auto species_names = pbb.getSpeciesNames();
    const auto nSpecies = static_cast<int>(species_names.size());
    constexpr int nBoundaries = ParticleBoundaryBuffer::numBoundaries();

    // per-step scraped energy, then cumulative scraped energy,
    // each (species x boundary)
    m_data.resize(2*nSpecies*nBoundaries, 0.0_rt);

    m_last_step_counted.assign(nBoundaries, std::vector<int>(nSpecies, -1));

    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            for (int i_s = 0; i_s < nSpecies; ++i_s)
            {
                for (int ib = 0; ib < nBoundaries; ++ib)
                {
                    ofs << m_sep;
                    ofs << "[" << c++ << "]outflow_energy_" << species_names[i_s]
                        << "_" << pbb.boundaryName(ib) << "(J)";
                }
            }
            for (int i_s = 0; i_s < nSpecies; ++i_s)
            {
                for (int ib = 0; ib < nBoundaries; ++ib)
                {
                    ofs << m_sep;
                    ofs << "[" << c++ << "]integrated_outflow_" << species_names[i_s]
                        << "_" << pbb.boundaryName(ib) << "(J)";
                }
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}

void ScrapedParticleEnergy::ComputeDiags (int /*step*/)
{
    // Note that this runs every step (independent of the write intervals)
    // so that the cumulative integrals stay exact.

    auto & warpx = WarpX::GetInstance();
    auto & pbb = warpx.GetParticleBoundaryBuffer();
    const auto & mypc = warpx.GetPartContainer();

    const auto species_names = pbb.getSpeciesNames();
    const auto nSpecies = static_cast<int>(species_names.size());
    constexpr int nBoundaries = ParticleBoundaryBuffer::numBoundaries();

    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        const auto & myspc = mypc.GetParticleContainer(i_s);
        const amrex::ParticleReal mass = myspc.getMass();
        const bool is_photon = myspc.AmIA<PhysicalSpecies::photon>();

        for (int ib = 0; ib < nBoundaries; ++ib)
        {
            const int idx_inc = i_s*nBoundaries + ib;
            const int idx_cum = nSpecies*nBoundaries + i_s*nBoundaries + ib;

            m_data[idx_inc] = 0.0_rt;

            auto * buf = pbb.getParticleBufferPointer(species_names[i_s], ib);
            if (buf == nullptr || !buf->isDefined()) { continue; }

            const int last_counted = m_last_step_counted[ib][i_s];
            const int step_comp = buf->GetIntCompIndex("stepScraped");

            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax> reduce_ops;
            amrex::ReduceData<amrex::Real, int> reduce_data(reduce_ops);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            for (int lev = 0; lev <= buf->finestLevel(); ++lev)
            {
                for (auto & kv : buf->GetParticles(lev))
                {
                    const auto & ptile = kv.second;
                    const auto & soa = ptile.GetStructOfArrays();
                    const auto np = ptile.numParticles();
                    if (np == 0) { continue; }

                    const amrex::ParticleReal * const AMREX_RESTRICT w = soa.GetRealData(PIdx::w).data();
                    const amrex::ParticleReal * const AMREX_RESTRICT ux = soa.GetRealData(PIdx::ux).data();
                    const amrex::ParticleReal * const AMREX_RESTRICT uy = soa.GetRealData(PIdx::uy).data();
                    const amrex::ParticleReal * const AMREX_RESTRICT uz = soa.GetRealData(PIdx::uz).data();
                    const int * const AMREX_RESTRICT scraped_step = soa.GetIntData(step_comp).data();

                    reduce_ops.eval(np, reduce_data,
                        [=] AMREX_GPU_DEVICE (int ip) -> ReduceTuple
                        {
                            if (scraped_step[ip] > last_counted) {
                                const amrex::Real ke = is_photon ?
                                    Algorithms::KineticEnergyPhotons(ux[ip], uy[ip], uz[ip]) :
                                    Algorithms::KineticEnergy(ux[ip], uy[ip], uz[ip], mass);
                                return {static_cast<amrex::Real>(w[ip])*ke, scraped_step[ip]};
                            }
                            return {0.0_rt, last_counted};
                        });
                }
            }

            auto reduced = reduce_data.value(reduce_ops);
            amrex::Real e_new = amrex::get<0>(reduced);
            int max_step_seen = amrex::get<1>(reduced);

            ParallelDescriptor::ReduceRealSum(e_new);
            ParallelDescriptor::ReduceIntMax(max_step_seen);

            m_data[idx_inc] = e_new;
            m_data[idx_cum] += e_new;
            m_last_step_counted[ib][i_s] = max_step_seen;
        }
    }
}

void
ScrapedParticleEnergy::WriteCheckpointData (std::string const & dir)
{
    // Write the cumulative sums and per-boundary bookkeeping
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream chkfile{dir + "/" + m_rd_name, std::ofstream::out};
        if (!chkfile.good()) {
            WARPX_ABORT_WITH_MESSAGE(
                "ScrapedParticleEnergy::WriteCheckpointData: could not open file for writing checkpoint data");
        }

        chkfile.precision(17);

        const auto n = m_data.size()/2;
        for (size_t i = 0; i < n; ++i) {
            chkfile << m_data[n + i] << "\n";
        }
        for (const auto & per_boundary : m_last_step_counted) {
            for (const int last : per_boundary) {
                chkfile << last << "\n";
            }
        }
    }
}

void
ScrapedParticleEnergy::ReadCheckpointData (std::string const & dir)
{
    // Read the cumulative sums and per-boundary bookkeeping
    std::ifstream chkfile{dir + "/" + m_rd_name, std::ifstream::in};
    if (!chkfile.good()) {
        WARPX_ABORT_WITH_MESSAGE(
            "ScrapedParticleEnergy::ReadCheckpointData: could not open file for reading checkpoint data");
    }

    const auto n = m_data.size()/2;
    for (size_t i = 0; i < n; ++i) {
        amrex::Real data;
        if (chkfile >> data) {
            m_data[n + i] = data;
        } else {
            WARPX_ABORT_WITH_MESSAGE(
                "ScrapedParticleEnergy::ReadCheckpointData: could not read the checkpoint data");
        }
    }
    for (auto & per_boundary : m_last_step_counted) {
        for (int & last : per_boundary) {
            if (!(chkfile >> last)) {
                WARPX_ABORT_WITH_MESSAGE(
                    "ScrapedParticleEnergy::ReadCheckpointData: could not read the checkpoint bookkeeping");
            }
        }
    }
}
