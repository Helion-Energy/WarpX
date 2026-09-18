/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_COILSET_H_
#define WARPX_CIRCUIT_COILSET_H_

#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace warpx::circuit
{
    /** One circular filament (a coil turn bundle) in the r-z plane,
     * axisymmetric about the machine axis. */
    struct Coil
    {
        std::string name;              //!< unique identifier
        amrex::Real r = 0.0;           //!< filament radius [m], > 0
        amrex::Real z = 0.0;           //!< axial position [m]
        amrex::Real n_turns = 1.0;     //!< turn count (inductances scale with n_i n_j)
        /** Reference current [A]: the external field named field_name holds
         * the coil's field AT I_ref, so a drive scale s = I/I_ref of 1
         * reproduces it. */
        amrex::Real I_ref = 1.0;
        /** Paired entry of external_vector_potential.fields (default: the
         * coil name). */
        std::string field_name;
        /** Fill <field_name>_Aext with the coil's unit field at I_ref from
         * the ring kernel (otherwise the field keeps whatever the
         * external_vector_potential initialization loaded). */
        bool fill_unit_field = true;
    };

    /** Ordered collection of coils read from the `circuit.` inputs:
     *     circuit.coils = drive_a drive_b
     *     circuit.drive_a.r = 0.25
     *     circuit.drive_a.z = 0.0
     *     circuit.drive_a.n_turns = 4
     *     circuit.drive_a.I_ref = 1.0e5
     *     circuit.drive_a.field_name = drive_a   # default: the coil name
     *     circuit.drive_a.fill_unit_field = 1
     */
    class CoilSet
    {
    public:
        /** Read `circuit.coils` and the per-coil parameters; validates
         * r > 0, I_ref != 0 and name uniqueness with clear aborts. */
        void ReadParameters ();

        [[nodiscard]] int size () const { return static_cast<int>(m_coils.size()); }
        [[nodiscard]] const Coil& coil (int i) const { return m_coils[i]; }
        [[nodiscard]] const std::vector<Coil>& coils () const { return m_coils; }

    private:
        std::vector<Coil> m_coils;
    };

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_COILSET_H_
