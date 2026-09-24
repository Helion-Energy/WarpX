/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ScatteringProcess.H"

#include "Utils/TextMsg.H"

#include <algorithm>

ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const std::string& cross_section_file,
                        const amrex::ParticleReal energy,
                        const ScatteringAngleModel scattering_angle_model,
                        const std::string& anisotropy_file )
{
    // read the cross-section data file into memory
    readCrossSectionFile(cross_section_file, m_energies, m_sigmas_h);

    init(scattering_process, energy, scattering_angle_model, anisotropy_file);
}

template <typename InputVector>
ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const InputVector&& energies,
                        const InputVector&& sigmas,
                        const amrex::ParticleReal energy,
                        const ScatteringAngleModel scattering_angle_model )
{
    m_energies.insert(m_energies.begin(), std::begin(energies), std::end(energies));
    m_sigmas_h.insert(m_sigmas_h.begin(), std::begin(sigmas),   std::end(sigmas));

    init(scattering_process, energy, scattering_angle_model);
}

void
ScatteringProcess::init (const std::string& scattering_process, const amrex::ParticleReal energy,
                         const ScatteringAngleModel scattering_angle_model,
                         const std::string& anisotropy_file)
{
    using namespace amrex::literals;
    m_exe_h.m_energies_data = m_energies.data();
    m_exe_h.m_sigmas_data = m_sigmas_h.data();

    // save energy grid parameters for easy use
    const int grid_size = static_cast<int>(m_energies.size());
    m_exe_h.m_grid_size = grid_size;
    m_exe_h.m_energy_lo = m_energies[0];
    m_exe_h.m_energy_hi = m_energies[grid_size-1];
    m_exe_h.m_sigma_lo = m_sigmas_h[0];
    m_exe_h.m_sigma_hi = m_sigmas_h[grid_size-1];
    // The energy grid does not need to be evenly spaced; `m_dE` is only used as a
    // representative energy step (e.g. to set the scan resolution when computing the
    // maximum collision frequency). Use the smallest spacing so that finely resolved
    // regions of a non-uniform grid are not skipped over.
    m_exe_h.m_dE = m_energies[grid_size-1] - m_energies[0];
    for (int i = 1; i < grid_size; i++) {
        m_exe_h.m_dE = std::min(m_exe_h.m_dE, m_energies[i] - m_energies[i-1]);
    }
    m_exe_h.m_energy_penalty = energy;
    m_exe_h.m_type = parseProcessType(scattering_process);
    m_exe_h.m_scattering_angle_model = scattering_angle_model;
    m_exe_h.m_produces_products = (
        m_exe_h.m_type == ScatteringProcessType::IONIZATION ||
        m_exe_h.m_type == ScatteringProcessType::TWOPRODUCT_REACTION ||
        m_exe_h.m_type == ScatteringProcessType::CHARGE_EXCHANGE);

    // sanity check cross-section energy grid
    sanityCheckEnergyGrid(m_energies);

    // The Okhrimovskyy model is defined by its anisotropy table, and a table is meaningless
    // for any other model; requiring the two together keeps either from being silently ignored.
    const bool okhrimovskyy = (scattering_angle_model == ScatteringAngleModel::Okhrimovskyy);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        okhrimovskyy == !anisotropy_file.empty(),
        "Process '" + scattering_process + "': an anisotropy table "
        "(<process>_scattering_anisotropy) must be given with, and only with, "
        "<process>_scattering_angle_model = okhrimovskyy."
    );
    if (okhrimovskyy) {
        readCrossSectionFile(anisotropy_file, m_xi_energies, m_xi_h);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_xi_energies.size() >= 2,
            "Process '" + scattering_process + "': the anisotropy table needs at least two rows."
        );
        sanityCheckEnergyGrid(m_xi_energies);
        for (const auto xi : m_xi_h) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (xi > -1._prt) && (xi < 1._prt),
                "Process '" + scattering_process + "': the anisotropy parameter xi must "
                "satisfy -1 < xi < 1."
            );
        }
        m_exe_h.m_xi_energies_data = m_xi_energies.data();
        m_exe_h.m_xi_data = m_xi_h.data();
        m_exe_h.m_xi_grid_size = static_cast<int>(m_xi_energies.size());
    }

    // check that the cross-section is 0 at the energy cost if the energy
    // cost is > 0 - this is to prevent the possibility of negative left
    // over energy after a collision event
    if (m_exe_h.m_energy_penalty > 0) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (getCrossSection(m_exe_h.m_energy_penalty) == 0),
            "Cross-section > 0 at energy cost for collision."
        );
    }

#ifdef AMREX_USE_GPU
    m_exe_d = m_exe_h;
    m_energies_d.resize(m_energies.size());
    m_sigmas_d.resize(m_sigmas_h.size());
    m_exe_d.m_energies_data = m_energies_d.data();
    m_exe_d.m_sigmas_data = m_sigmas_d.data();
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_energies.begin(), m_energies.end(),
                          m_energies_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_sigmas_h.begin(), m_sigmas_h.end(),
                          m_sigmas_d.begin());
    if (m_exe_h.m_xi_grid_size > 0) {
        m_xi_energies_d.resize(m_xi_energies.size());
        m_xi_d.resize(m_xi_h.size());
        m_exe_d.m_xi_energies_data = m_xi_energies_d.data();
        m_exe_d.m_xi_data = m_xi_d.data();
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_xi_energies.begin(),
                              m_xi_energies.end(), m_xi_energies_d.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_xi_h.begin(), m_xi_h.end(),
                              m_xi_d.begin());
    }
    amrex::Gpu::streamSynchronize();
#endif
}

ScatteringProcessType
ScatteringProcess::parseProcessType(const std::string& scattering_process)
{
    if (scattering_process.find("elastic") != std::string::npos) {
        // `elastic` is matched as a prefix (like `excitationX`) so that several distinct
        // elastic channels (e.g. with different cross-sections and/or scattering angle
        // models) can be included in the same collision under unique names.
        return ScatteringProcessType::ELASTIC;
    } else if (scattering_process == "charge_exchange") {
        return ScatteringProcessType::CHARGE_EXCHANGE;
    } else if (scattering_process == "two_product_reaction") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "ionization" ||
               scattering_process.rfind("ionization_", 0) == 0) {
        // `ionization_<name>` adds further ionization channels beside `ionization`, e.g.
        // one per product vibrational level; background MCC samples among them.
        return ScatteringProcessType::IONIZATION;
    } else if (scattering_process.find("excitation") != std::string::npos) {
        return ScatteringProcessType::EXCITATION;
    } else {
        return ScatteringProcessType::INVALID;
    }
}

void
ScatteringProcess::readCrossSectionFile (
                                  const std::string& cross_section_file,
                                  amrex::Vector<amrex::ParticleReal>& energies,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas )
{
    std::ifstream infile(cross_section_file);
    if(!infile.is_open()) { WARPX_ABORT_WITH_MESSAGE("Failed to open cross-section data file"); }

    amrex::ParticleReal energy, sigma;
    while (infile >> energy >> sigma) {
        energies.push_back(energy);
        sigmas.push_back(sigma);
    }
    if (infile.bad()) { WARPX_ABORT_WITH_MESSAGE("Failed to read cross-section data from file."); }
    infile.close();
}

void
ScatteringProcess::sanityCheckEnergyGrid (
                                   const amrex::Vector<amrex::ParticleReal>& energies
                                   )
{
    // The energy grid does not need to be evenly spaced, but it must be sorted in
    // strictly increasing order for the bisection search and linear interpolation
    // used in `Executor::getCrossSection` to work correctly.
    for (unsigned i = 1; i < energies.size(); i++) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                                         (energies[i] > energies[i-1]),
                                         "Cross-section energy grid must be sorted in "
                                         "strictly increasing order."
                                         );
    }
}
