/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Prabhat Kumar (Helion Energy Inc)
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EXTERNAL_FIELD_SOURCE_H_
#define WARPX_EXTERNAL_FIELD_SOURCE_H_

#include "Fields.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>
#include <set>
#include <vector>
#include <map>

// ========================================================================
// LC Circuit Solver Class
// ========================================================================
/**
 * \brief Simple LC circuit solver for circuit coupling
 */
class LCCircuit {
public:
    /**
     * \brief Constructor for LC circuit
     * \param L Inductance in Henries
     * \param C Capacitance in Farads
     * \param V0 Initial voltage in Volts
     * \param I0 Initial current in Amperes (default 0)
     */
    LCCircuit(amrex::Real L, amrex::Real C, amrex::Real R, amrex::Real V0, amrex::Real I0 = 0.0);

    /**
     * \brief Update circuit state to next time step
     * \param dt Time step in seconds
     */
    void UpdateTimeStep(amrex::Real dt);

    /**
     * \brief Get current voltage across capacitor
     * \return Voltage in Volts
     */
    amrex::Real GetVoltage() const { return m_voltage; }

    /**
     * \brief Get current through inductor
     * \return Current in Amperes
     */
    amrex::Real GetCurrent() const { return m_current; }

    /**
     * \brief Get resonant frequency
     * \return Frequency in Hz
     */
    amrex::Real GetResonantFrequency() const { return 1.0 / (2.0 * M_PI * sqrt(m_L * m_C)); }

    /**
     * \brief Get resonant period
     * \return Period in seconds
     */
    amrex::Real GetResonantPeriod() const { return 2.0 * M_PI * sqrt(m_L * m_C); }

    /**
     * \brief Set external voltage source (for coupling)
     * \param V_ext External voltage in Volts
     */
    void SetExternalVoltage(amrex::Real V_ext) { m_external_voltage = V_ext; }

    /**
     * \brief Reset circuit to initial conditions
     */
    void Reset();

    /**
     * \brief Print circuit parameters
     */
    void PrintInfo() const;

private:
    amrex::Real m_L;                    // Inductance [H]
    amrex::Real m_C;                    // Capacitance [F]
    amrex::Real m_R;                    // Resistance [Ohm]
    amrex::Real m_V0;                   // Initial voltage [V]
    amrex::Real m_I0;                   // Initial current [A]
    amrex::Real m_voltage;              // Current voltage [V]
    amrex::Real m_current;              // Current current [A]
    amrex::Real m_external_voltage;     // External voltage source [V]
    amrex::Real m_time;                 // Current time [s]
};

/**
 * \brief This class represents a space- and time-dependent external electric and magnetic field source
 *        that can be applied directly to the grid. It also supports circuit coupling through voltage
 *        calculation at ports defined by the same flag parsers used for field excitation.
 */
class ExternalFieldSource
{
protected:
    // Initialization Type for External E and B excitation on grid
    static std::string B_excitation_grid_s;
    static std::string E_excitation_grid_s;
    static int ApplyExcitationInPML;
    static int excite_E_source;
    static int excite_B_source;
    static int excite_all;

    // Function for B excitation (x,y,z,t) on the grid
    static std::string str_Bx_excitation_grid_function;
    static std::string str_By_excitation_grid_function;
    static std::string str_Bz_excitation_grid_function;
    // Function for E excitation (x,y,z,t) on the grid
    static std::string str_Ex_excitation_grid_function;
    static std::string str_Ey_excitation_grid_function;
    static std::string str_Ez_excitation_grid_function;

    // Flag for type of excitation (hard/soft)
    static std::string str_Ex_excitation_flag_function;
    static std::string str_Ey_excitation_flag_function;
    static std::string str_Ez_excitation_flag_function;
    static std::string str_Bx_excitation_flag_function;
    static std::string str_By_excitation_flag_function;
    static std::string str_Bz_excitation_flag_function;

    // Parser for B_external excitation on the grid
    std::unique_ptr<amrex::Parser> Bxfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Byfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Bzfield_xt_grid_parser;
    // Parser for E_external excitation on the grid
    std::unique_ptr<amrex::Parser> Exfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Eyfield_xt_grid_parser;
    std::unique_ptr<amrex::Parser> Ezfield_xt_grid_parser;

    std::unique_ptr<amrex::Parser> Exfield_flag_parser;
    std::unique_ptr<amrex::Parser> Eyfield_flag_parser;
    std::unique_ptr<amrex::Parser> Ezfield_flag_parser;
    std::unique_ptr<amrex::Parser> Bxfield_flag_parser;
    std::unique_ptr<amrex::Parser> Byfield_flag_parser;
    std::unique_ptr<amrex::Parser> Bzfield_flag_parser;

    // Circuit coupling members
    amrex::Real m_circuit_current = 0.0;
    bool m_circuit_coupling_enabled = false;
    amrex::Real m_coupling_strength = 0.01;
    std::vector<amrex::Real> m_port_voltages;
    std::vector<int> m_port_ids;
    
    // Cache for port information to avoid repeated parsing
    std::set<int> m_unique_ports;
    bool m_ports_initialized = false;
    
    // Map for quick voltage lookup by port ID
    std::map<int, amrex::Real> m_port_voltage_map;

    std::map<int, amrex::Real> m_port_magnetic_flux_map;
    
    // Circuit coupling input parameters
    static int m_circuit_coupling_interval;

public:

    // Default Constructor
    ExternalFieldSource();

    /** \brief Adds the contribution of user-defined external field-excitation
     *   to Efield and Bfield.
     *   Two types of excitation are supported, namely, hard source and soft source.
     *   An excitation is called hard source if the field value is set exactly equal
     *   to the excitation (similar to a Dirichlet boundary) (field = excitation).
     *   An excitation is called soft source if the field value is updated by adding
     *   to it the contribution from the externally applied excitation (field += excitation).
     *   To ensure that the excitation is appropriately accounted for, a flag parser for
     *   each component of the field must be specified.
     *   The flag can have three values :
     *       If flag_type == 0, field is not updated with external excitation
     *       If flag_type == 1, field is set equal to the
     *                          external excitation (aka hard source)
     *       If flag_type == 2, field is updated to add the contribution from
     *                          external excitation (aka soft source)
     *
     *   \param[in] mfx, mfy, mfz : The field component Multifabs to be updated with
     *                              the external excitation.
     *   \param[in] xfield_parser : external excitation for xcomponent of the field
     *   \param[in] yfield_parser : external excitation for ycomponent of the field
     *   \param[in] zfield_parser : external excitation for zcomponent of the field
     *   \param[in] xflag_parser  : Type xfield excitation (hard source=0/soft source=1)
     *   \param[in] yflag_parser  : Type yfield excitation (hard source=0/soft source=1)
     *   \param[in] zflag_parser  : Type zfield excitation (hard source=0/soft source=1)
     *   \param[in] lev           : level on which the excitation is applied.
     */

    void ApplyExternalFieldExcitationOnGrid (DtType a_dt_type,  bool apply_E, bool apply_B);

    void ApplyExternalFieldExcitationOnGrid (
       warpx::fields::FieldType field,
       amrex::ParserExecutor<4> const& xfield_parser,
       amrex::ParserExecutor<4> const& yfield_parser,
       amrex::ParserExecutor<4> const& zfield_parser,
       amrex::ParserExecutor<3> const& xflag_parser,
       amrex::ParserExecutor<3> const& yflag_parser,
       amrex::ParserExecutor<3> const& zflag_parser, 
       const int lev, DtType a_dt_type);

    /** Parse field excitation functions and flags*/
    void ReadExcitationParser ();

    // ========================================================================
    // Circuit coupling methods
    // ========================================================================

    /**
     * \brief Calculate voltages at all ports defined by flag parsers
     * \param lev AMR level
     */
    void CalculatePortVoltages(int lev);
    
    /**
     * \brief Calculate voltage for a specific port ID
     * \param port_id The port identifier from flag parser
     * \param lev AMR level
     * \return Calculated voltage for the port
     */
    amrex::Real CalculateVoltageForPort(int port_id, int lev);
    amrex::Real CalculateVoltageForPort_EMF(int port_id, int lev);
    
    /**
     * \brief Identify unique port IDs from flag parsers
     * \param lev AMR level
     * \return Set of unique port IDs found in the domain
     */
    std::set<int> GetUniquePortIDs(int lev);
    
    /**
     * \brief Get calculated port voltages
     * \return Vector of voltages corresponding to port IDs
     */
    std::vector<amrex::Real> GetPortVoltages() const { return m_port_voltages; }
    
    /**
     * \brief Get port IDs in the same order as voltages
     * \return Vector of port IDs
     */
    std::vector<int> GetPortIDs() const { return m_port_ids; }
    
    /**
     * \brief Get voltage for a specific port ID
     * \param port_id The port identifier
     * \return Voltage for the specified port (0.0 if port not found)
     */
    amrex::Real GetVoltageForPort(int port_id) const;
    
    /**
     * \brief Check if circuit coupling is enabled
     * \return True if circuit coupling is enabled
     */
    bool IsCircuitCouplingEnabled() const { return m_circuit_coupling_enabled; }
    
    /**
     * \brief Initialize port information (called once during setup)
     * \param lev AMR level
     */
    void InitializePorts(int lev);
    
    /**
     * \brief Update field excitations based on circuit currents
     * \param currents Vector of currents corresponding to port IDs
     */
    void UpdateExcitationsFromCircuitCurrents(const std::vector<amrex::Real>& currents);

    /**
     * \brief Update B-field excitation amplitude from single port current
     * \param current Current in Amperes
     */
    void UpdateBExcitationFromCurrent(amrex::Real current);

    /**
     * \brief Print port information for debugging
     */
    void PrintPortInfo() const;

    /**
     * \brief Initialize LC circuit for testing
     * \param L Inductance in Henries
     * \param C Capacitance in Farads
     * \param V0 Initial voltage in Volts
     */
    void InitializeLCCircuit(amrex::Real L, amrex::Real C, amrex::Real R, amrex::Real V0);

    /**
     * \brief Update LC circuit state
     * \param dt Time step in seconds
     */
    void UpdateLCCircuit(amrex::Real dt);

    /**
     * \brief Get LC circuit voltage
     * \return Voltage in Volts
     */
    amrex::Real GetLCVoltage() const;

    /**
     * \brief Get LC circuit current
     * \return Current in Amperes
     */
    amrex::Real GetLCCurrent() const;

    // LC Circuit for testing circuit coupling
    std::unique_ptr<LCCircuit> m_lc_circuit;
private:

};

#endif // WARPX_EXTERNAL_FIELD_SOURCE_H_
