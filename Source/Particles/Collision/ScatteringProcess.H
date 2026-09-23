/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
#define WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_

#include "Utils/WarpXAlgorithmSelection.H"

#include <AMReX_Algorithm.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_RandomEngine.H>
#include <AMReX_Vector.H>

#include <string>

enum class ScatteringProcessType {
    INVALID,
    ELASTIC,
    TWOPRODUCT_REACTION,
    CHARGE_EXCHANGE,
    EXCITATION,
    IONIZATION,
};

class ScatteringProcess
{
public:
    /**
     * @param[in] scattering_process the process name, e.g. ``elastic`` or ``excitation1``
     * @param[in] cross_section_file path to the two-column cross-section table
     * @param[in] energy energy cost of the process in eV
     * @param[in] scattering_angle_model the angular distribution of the products
     * @param[in] anisotropy_file path to a two-column table of the anisotropy parameter
     *            xi against energy in eV; required for, and only allowed with,
     *            ScatteringAngleModel::Okhrimovskyy
     */
    ScatteringProcess (
                const std::string& scattering_process,
                const std::string& cross_section_file,
                amrex::ParticleReal energy,
                ScatteringAngleModel scattering_angle_model,
                const std::string& anisotropy_file = ""
                );

    template <typename InputVector>
    ScatteringProcess (
                const std::string& scattering_process,
                const InputVector&& energies,
                const InputVector&& sigmas,
                amrex::ParticleReal energy,
                ScatteringAngleModel scattering_angle_model
                );

    ~ScatteringProcess() = default;

    ScatteringProcess (ScatteringProcess const&)            = delete;
    ScatteringProcess& operator= (ScatteringProcess const&) = delete;
    ScatteringProcess (ScatteringProcess &&)                = default;
    ScatteringProcess& operator= (ScatteringProcess &&)     = default;

    /** Read the given cross-section data file to memory.
     *
     * @param cross_section_file the path to the file containing the cross-
     *        section data
     * @param energies vector storing energy values in eV
     * @param sigmas vector storing cross-section values
     *
     */
    static
    void readCrossSectionFile (
                               const std::string& cross_section_file,
                               amrex::Vector<amrex::ParticleReal>& energies,
                               amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas
                               );

    static
    void sanityCheckEnergyGrid (
                                const amrex::Vector<amrex::ParticleReal>& energies
                                );

    struct Executor {
        /** Get the collision cross-section using a simple linear interpolator. If
         * the energy value is lower (higher) than the given energy range the
         * first (last) cross-section value is used.
         *
         * @param E_coll collision energy in eV
         *
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll) const
        {
            if (E_coll < m_energy_lo) {
                return m_sigma_lo;
            } else if (E_coll > m_energy_hi) {
                return m_sigma_hi;
            } else {
                // Locate the pair of energy grid points bracketing E_coll using a
                // bisection search. This does not assume an evenly spaced energy grid.
                // `amrex::bisect` returns idx_1 such that
                // m_energies_data[idx_1] <= E_coll < m_energies_data[idx_1 + 1].
                const int idx_1 = amrex::bisect(m_energies_data, 0, m_grid_size - 1, E_coll);
                const int idx_2 = idx_1 + 1;

                // linearly interpolate to the given energy value
                const amrex::ParticleReal temp =
                    (E_coll - m_energies_data[idx_1])
                    / (m_energies_data[idx_2] - m_energies_data[idx_1]);
                return m_sigmas_data[idx_1] + (m_sigmas_data[idx_2] - m_sigmas_data[idx_1]) * temp;
            }
        }

        /** Get the Okhrimovskyy anisotropy parameter xi by linear interpolation in the
         * anisotropy table, holding the first (last) value below (above) its range.
         * Returns 0 (isotropic) when the process has no anisotropy table.
         *
         * @param E_coll collision energy in eV
         */
        [[nodiscard]]
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::ParticleReal getAnisotropy (amrex::ParticleReal E_coll) const
        {
            if (m_xi_grid_size == 0) {
                return 0;
            } else if (E_coll <= m_xi_energies_data[0]) {
                return m_xi_data[0];
            } else if (E_coll >= m_xi_energies_data[m_xi_grid_size - 1]) {
                return m_xi_data[m_xi_grid_size - 1];
            } else {
                const int idx_1 = amrex::bisect(m_xi_energies_data, 0, m_xi_grid_size - 1, E_coll);
                const int idx_2 = idx_1 + 1;
                const amrex::ParticleReal temp =
                    (E_coll - m_xi_energies_data[idx_1])
                    / (m_xi_energies_data[idx_2] - m_xi_energies_data[idx_1]);
                return m_xi_data[idx_1] + (m_xi_data[idx_2] - m_xi_data[idx_1]) * temp;
            }
        }

        amrex::ParticleReal* m_energies_data = nullptr;
        amrex::ParticleReal* m_sigmas_data = nullptr;
        int m_grid_size = 0;
        amrex::ParticleReal* m_xi_energies_data = nullptr;
        amrex::ParticleReal* m_xi_data = nullptr;
        int m_xi_grid_size = 0;
        amrex::ParticleReal m_energy_lo, m_energy_hi, m_sigma_lo, m_sigma_hi, m_dE;
        amrex::ParticleReal m_energy_penalty;
        ScatteringProcessType m_type;
        ScatteringAngleModel m_scattering_angle_model;
        /** True if this process creates new product particles (ionization, charge exchange,
         *  two-product reaction); false for purely scattering processes (elastic, excitation). **/
        bool m_produces_products = false;
    };

    [[nodiscard]]
    Executor const& executor () const {
#ifdef AMREX_USE_GPU
        return m_exe_d;
#else
        return m_exe_h;
#endif
    }

    [[nodiscard]] amrex::ParticleReal getCrossSection (amrex::ParticleReal E_coll) const
    {
        return m_exe_h.getCrossSection(E_coll);
    }

    [[nodiscard]] amrex::ParticleReal getEnergyPenalty () const { return m_exe_h.m_energy_penalty; }
    [[nodiscard]] amrex::ParticleReal getMinEnergyInput () const { return m_exe_h.m_energy_lo; }
    [[nodiscard]] amrex::ParticleReal getMaxEnergyInput () const { return m_exe_h.m_energy_hi; }
    [[nodiscard]] amrex::ParticleReal getEnergyInputStep () const { return m_exe_h.m_dE; }

    [[nodiscard]] ScatteringProcessType type () const { return m_exe_h.m_type; }

    [[nodiscard]] ScatteringAngleModel scatteringAngleModel () const
    {
        return m_exe_h.m_scattering_angle_model;
    }

    [[nodiscard]] amrex::ParticleReal getAnisotropy (amrex::ParticleReal E_coll) const
    {
        return m_exe_h.getAnisotropy(E_coll);
    }

    static
    ScatteringProcessType parseProcessType(const std::string& process);

private:

    void init (const std::string& scattering_process, amrex::ParticleReal energy,
               ScatteringAngleModel scattering_angle_model,
               const std::string& anisotropy_file = "");

    amrex::Vector<amrex::ParticleReal> m_energies;
    amrex::Vector<amrex::ParticleReal> m_xi_energies;
    amrex::Gpu::HostVector<amrex::ParticleReal> m_xi_h;

#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_energies_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigmas_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_xi_energies_d;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_xi_d;
    Executor m_exe_d;
#endif
    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_h;
    Executor m_exe_h;
};

#endif // WARPX_PARTICLES_COLLISION_SCATTERING_PROCESS_H_
