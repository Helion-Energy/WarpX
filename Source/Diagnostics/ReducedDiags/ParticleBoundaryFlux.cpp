/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleBoundaryFlux.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_Config.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <array>
#include <fstream>
#include <string>

using namespace amrex;

namespace
{
    constexpr int nfaces = 2*AMREX_SPACEDIM;
    constexpr int nquantities = 3;

#if defined(WARPX_DIM_1D_Z)
    const std::array<std::string, AMREX_SPACEDIM> axis_names {"z"};
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    const std::array<std::string, AMREX_SPACEDIM> axis_names {"r"};
#elif defined(WARPX_DIM_RZ)
    const std::array<std::string, AMREX_SPACEDIM> axis_names {"r", "z"};
#elif defined(WARPX_DIM_XZ)
    const std::array<std::string, AMREX_SPACEDIM> axis_names {"x", "z"};
#else
    const std::array<std::string, AMREX_SPACEDIM> axis_names {"x", "y", "z"};
#endif
    const std::array<std::string, 2> side_names {"lo", "hi"};
    const std::array<std::string, nquantities> quantity_names
        {"count", "charge_C", "KE_J"};
}

// constructor
ParticleBoundaryFlux::ParticleBoundaryFlux (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    const auto & mypc = WarpX::GetInstance().GetPartContainer();
    const auto nSpecies = mypc.nSpecies();
    const auto species_names = mypc.GetSpeciesNames();

    m_data.resize(static_cast<std::size_t>(nSpecies)*nfaces*nquantities, 0.0_rt);

    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            for (int i = 0; i < nSpecies; ++i)
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    for (int s = 0; s < 2; ++s)
                    {
                        for (const auto& q : quantity_names)
                        {
                            ofs << m_sep;
                            ofs << "[" << c++ << "]" << species_names[i]
                                << "_" << axis_names[d]
                                << "_" << side_names[s]
                                << "_" << q << "()";
                        }
                    }
                }
            }
            ofs << "\n";
            ofs.close();
        }
    }
}

void ParticleBoundaryFlux::ComputeDiags (int step)
{
    // Check if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    const auto & mypc = WarpX::GetInstance().GetPartContainer();
    const int nSpecies = mypc.nSpecies();

    // Collect the rank-local cumulative tallies, then one global sum.
    int c = 0;
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        const auto & myspc = mypc.GetParticleContainer(i_s);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            for (int s = 0; s < 2; ++s)
            {
                m_data[c++] = myspc.GetBoundaryAbsorbedWeight(d, s);
                m_data[c++] = myspc.GetBoundaryAbsorbedCharge(d, s);
                m_data[c++] = myspc.GetBoundaryAbsorbedEnergy(d, s);
            }
        }
    }

    ParallelDescriptor::ReduceRealSum(
        m_data.data(), static_cast<int>(m_data.size()),
        ParallelDescriptor::IOProcessorNumber());
}
